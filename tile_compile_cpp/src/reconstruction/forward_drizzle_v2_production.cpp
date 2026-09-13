// Gate-10 production wiring: see the header for the contract. The frame
// provider is deliberately serial --- the driver owns the band loop and the
// kernel owns all parallelism.

#include "tile_compile/reconstruction/forward_drizzle_v2_production.hpp"

#include "tile_compile/core/events.hpp"
#include "tile_compile/reconstruction/multiband_fusion.hpp"
#include "tile_compile/reconstruction/output_scale.hpp"
#include "tile_compile/reconstruction/quality_frame_weight_plan.hpp"
#include "tile_compile/reconstruction/source_quality_artifact.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>

namespace tile_compile::reconstruction {
namespace {

constexpr std::size_t kV2ReservoirRecordBytes = 32;   // CpuReservoirRecord
constexpr std::size_t kV2ReservoirQualityBytes = 16;  // CpuReservoirQuality
constexpr std::size_t kV2PixelResultBytes = 64;       // ForwardDrizzleV2PixelResult
constexpr std::size_t kV2ProfileResultBytes = 80;     // ForwardDrizzleV2ProfileResult
constexpr std::size_t kV2FrameMetaBytes = 16;         // per stream slot

std::size_t v2_checked_mul(std::size_t a, std::size_t b) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::overflow_error("FDV2_PROD_SIZE_OVERFLOW");
  return a * b;
}

// central difference with one-sided fallback at borders / non-finite
// neighbours; 0 when no finite neighbour exists.
float v2_central_diff(const Matrix2Df &s, int y, int x, bool x_axis) {
  const int extent = x_axis ? static_cast<int>(s.cols())
                            : static_cast<int>(s.rows());
  const int pm = (x_axis ? x : y) - 1, pp = (x_axis ? x : y) + 1;
  const float vm = (pm >= 0) ? (x_axis ? s(y, pm) : s(pm, x))
                             : std::numeric_limits<float>::quiet_NaN();
  const float vp = (pp < extent) ? (x_axis ? s(y, pp) : s(pp, x))
                                 : std::numeric_limits<float>::quiet_NaN();
  const bool fm = std::isfinite(vm), fp = std::isfinite(vp);
  if (fm && fp) return (vp - vm) * 0.5f;
  const float vc = s(y, x);
  if (fp && std::isfinite(vc)) return vp - vc;
  if (fm && std::isfinite(vc)) return vc - vm;
  return 0.0f;
}

// star_residuals[i].rms_px / applicable from artifacts/global_registration.json,
// positional in source_index order (same `frames` enumeration the registration
// phase wrote). Empty vector on any parse/shape failure --- the caller maps
// that to the absent-sigma2 contract.
struct RegResiduals {
  std::vector<float> rms_px;
  std::vector<uint8_t> applicable;
};
RegResiduals load_reg_residuals(const fs::path &path, std::size_t count) {
  RegResiduals out;
  std::error_code ec;
  if (path.empty() || !fs::is_regular_file(fs::symlink_status(path, ec)))
    return out;
  nlohmann::json j;
  try {
    std::ifstream in(path);
    j = nlohmann::json::parse(in);
  } catch (...) {
    return out;
  }
  if (!j.contains("star_residuals") || !j["star_residuals"].is_array() ||
      j["star_residuals"].size() < count)
    return out;
  out.rms_px.assign(count, std::numeric_limits<float>::quiet_NaN());
  out.applicable.assign(count, 0u);
  for (std::size_t i = 0; i < count; ++i) {
    const auto &e = j["star_residuals"][i];
    if (!e.is_object()) continue;
    const double rms = e.value("rms_px",
                               std::numeric_limits<double>::quiet_NaN());
    const bool appl = e.value("applicable", false);
    out.rms_px[i] = static_cast<float>(rms);
    out.applicable[i] = appl ? 1u : 0u;
  }
  return out;
}

// Per-source-index noise from source_quality_metrics-v1.json. Empty on any
// failure (absent-sigma2 contract).
std::vector<float> load_frame_noise(
    const fs::path &metrics_path, const registration::RegistrationSamplingPlan &sampling,
    const std::string &normalized_cache_hash,
    const std::string &quality_config_hash) {
  std::vector<float> out;
  SourceQualityMetricsArtifact m;
  std::string error;
  if (metrics_path.empty() ||
      !load_source_quality_metrics(metrics_path, sampling.source_identity_hash,
                                   quality_config_hash, normalized_cache_hash,
                                   m, error))
    return out;
  const std::size_t n =
      sampling.frames.empty()
          ? 0
          : std::max_element(sampling.frames.begin(), sampling.frames.end(),
                             [](const auto &a, const auto &b) {
                               return a.source_index < b.source_index;
                             })
                ->source_index +
                1;
  out.assign(n, std::numeric_limits<float>::quiet_NaN());
  for (const auto &f : m.frames)
    if (f.source_index < n) out[f.source_index] = f.noise;
  return out;
}

ForwardDrizzleV2LocalWarp make_local_warp(
    const registration::FrameSamplingTransform &f,
    const ForwardDrizzleSubdivisionParams &sub) {
  ForwardDrizzleV2LocalWarp w{};
  const auto &m = f.smooth_local_model;
  for (int i = 0; i < 16; ++i) {
    w.coeff_x[i] = m.coeff_x[i];
    w.coeff_y[i] = m.coeff_y[i];
  }
  w.image_rows = m.image_rows;
  w.image_cols = m.image_cols;
  w.model_valid = m.valid ? 1 : 0;
  w.model_coordinate_scale = f.model_coordinate_scale;
  w.model_offset_x = f.model_offset_x;
  w.model_offset_y = f.model_offset_y;
  const registration::LocalInversionParams inv{};
  w.max_iter = inv.max_iter;
  w.tol_px = inv.tol_px;
  w.safety_margin_px = inv.safety_margin_px;
  w.position_epsilon_internal_px = sub.position_epsilon_internal_px;
  w.max_subdivision_depth = sub.max_subdivision_depth;
  w.area_relative_epsilon = sub.area_relative_epsilon;
  return w;
}

}  // namespace

std::vector<float> forward_drizzle_v2_sigma2_plane(
    const Matrix2Df &source, double sigma_noise, double sigma_reg_px,
    double droplet_half) {
  const int h = static_cast<int>(source.rows()),
            w = static_cast<int>(source.cols());
  std::vector<float> out(static_cast<std::size_t>(h) * w);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const double gx = v2_central_diff(source, y, x, true);
      const double gy = v2_central_diff(source, y, x, false);
      out[static_cast<std::size_t>(y) * w + x] = static_cast<float>(
          forward_drizzle_v2_sigma2_model(sigma_noise, gx, gy, sigma_reg_px,
                                          droplet_half));
    }
  return out;
}

std::size_t forward_drizzle_v2_cpu_bytes_per_native_pixel(
    int channels, int internal_scale, int reservoir_size, bool sigma2_plane,
    bool emit_profiles) {
  const std::size_t res_slots =
      2 * static_cast<std::size_t>(reservoir_size);
  std::size_t per_ch =
      sizeof(double) * 7 + sizeof(unsigned int) * 2 +
      sizeof(unsigned short) + sizeof(std::uint64_t) +
      res_slots * kV2ReservoirRecordBytes + kV2PixelResultBytes;
  if (emit_profiles)
    per_ch += res_slots * kV2ReservoirQualityBytes + kV2ProfileResultBytes;
  const std::size_t frame_planes =
      (sigma2_plane ? 4 : 3) + (emit_profiles ? 5 : 0);
  const std::size_t internal =
      frame_planes * sizeof(double) * static_cast<std::size_t>(channels) *
      static_cast<std::size_t>(internal_scale) *
      static_cast<std::size_t>(internal_scale);
  return per_ch * static_cast<std::size_t>(channels) +
         sizeof(unsigned int) /* footprint */ + internal;
}

ForwardDrizzleV2RunPlan make_forward_drizzle_v2_run_plan(
    const registration::RegistrationSamplingPlan &sampling,
    const config::ReconstructionDrizzleConfig &drizzle_cfg,
    const config::ReconstructionClippingConfig &clipping_cfg,
    const config::ReconstructionMultibandConfig &multiband_cfg,
    bool emit_profiles, std::uint64_t frame_count,
    const std::string &normalized_cache_hash,
    const std::string &quality_plan_hash,
    const std::string &source_quality_cache_hash,
    const std::string &config_snapshot_hash, std::size_t host_budget_bytes,
    std::size_t device_budget_bytes) {
  const OutputScaleMode osm{drizzle_cfg.internal_scale,
                            drizzle_cfg.output_scale};
  if (!osm.valid())
    throw std::invalid_argument("FDV2_PROD_INVALID_OUTPUT_SCALE");
  // v2 stores on the native output grid; only output_scale == 1 is
  // supported by the cutover contract (2/2 rejected at plan time).
  if (drizzle_cfg.output_scale != 1)
    throw std::invalid_argument("FDV2_PROD_OUTPUT_SCALE_UNSUPPORTED");

  const int channels = sampling.color_mode == ColorMode::MONO ? 1 : 3;
  const std::size_t bpp = forward_drizzle_v2_cpu_bytes_per_native_pixel(
      channels, drizzle_cfg.internal_scale, 64 /*reservoir_size*/,
      true /*sigma2*/, emit_profiles);
  // The band workspace shares the host budget with the source/Q caches and
  // the fusion working set; give it half. When a device path is planned the
  // nominal device bound additionally caps it (deterministic across resume;
  // reserve() still verifies the real device per band).
  const std::size_t meta_bytes = v2_checked_mul(
      static_cast<std::size_t>(frame_count), kV2FrameMetaBytes);
  std::size_t band_budget = host_budget_bytes / 2;
  if (device_budget_bytes > 0)
    band_budget = std::min(band_budget, device_budget_bytes);
  band_budget = band_budget > meta_bytes ? band_budget - meta_bytes : 0;
  const std::size_t row_bytes = v2_checked_mul(
      bpp, static_cast<std::size_t>(sampling.canvas_width_native));
  if (row_bytes == 0 || band_budget / row_bytes < 1)
    throw std::invalid_argument("FDV2_PROD_BAND_BUDGET");
  const int band_rows = static_cast<int>(std::clamp<std::size_t>(
      band_budget / row_bytes, 1,
      static_cast<std::size_t>(sampling.canvas_height_native)));

  ForwardDrizzleV2RunPlan plan;
  plan.source_identity_hash = sampling.source_identity_hash;
  plan.normalized_cache_hash = normalized_cache_hash;
  plan.quality_plan_hash = quality_plan_hash;
  plan.sampling_plan_hash = sampling.plan_hash;
  plan.config_snapshot_hash = config_snapshot_hash;
  plan.source_quality_cache_hash = source_quality_cache_hash;
  plan.min_clip_contributors = drizzle_cfg.min_clip_contributors;
  plan.min_candidates = drizzle_cfg.min_clip_contributors;
  plan.robust_passes = drizzle_cfg.robust_passes;
  plan.sigma_low = clipping_cfg.clip_sigma_low;
  plan.sigma_high = clipping_cfg.clip_sigma_high;
  plan.sigma2_enabled = true;
  bool any_local = false;
  for (const auto &f : sampling.frames)
    if (f.valid && f.has_smooth_local_model) any_local = true;
  plan.local_warp_representation =
      any_local ? "smooth_local_coefficients" : "affine_only";
  plan.emit_profiles = emit_profiles;
  plan.multiband_levels = emit_profiles ? multiband_cfg.levels : 0;
  plan.fine_quality_exponent = multiband_cfg.fine_quality_exponent;
  plan.medium_quality_exponent = multiband_cfg.medium_quality_exponent;
  plan.native_width = sampling.canvas_width_native;
  plan.native_height = sampling.canvas_height_native;
  plan.channels = channels;
  plan.internal_scale = drizzle_cfg.internal_scale;
  plan.color_mode = channels == 1 ? "MONO" : "OSC";
  plan.pixfrac = drizzle_cfg.pixfrac;
  plan.bayer_pattern = static_cast<int>(sampling.bayer_pattern);
  plan.cfa_origin_x = sampling.cfa_origin_x;
  plan.cfa_origin_y = sampling.cfa_origin_y;
  plan.frame_count = frame_count;
  plan.band_rows = band_rows;
  plan.band_count = (plan.native_height + band_rows - 1) / band_rows;
  plan.tile_cols = 0;
  plan.halo_rows =
      emit_profiles ? multiband_fusion_halo_rows(multiband_cfg.levels) : 0;
  plan.x_tiled = false;
  finalize_forward_drizzle_v2_run_plan(plan);
  return plan;
}

ForwardDrizzleV2ProductionResult persist_forward_drizzle_v2_from_predecessors(
    const fs::path &store_root, const fs::path &quality_artifact,
    const registration::RegistrationSamplingPlan &sampling,
    VerifiedNormalizedSourceCache &cache, const GlobalQualityConfig &quality_cfg,
    const config::ReconstructionDrizzleConfig &drizzle_cfg,
    const config::ReconstructionClippingConfig &clipping_cfg,
    const config::ReconstructionMultibandConfig &multiband_cfg,
    const fs::path &source_quality_cache_root,
    const fs::path &source_quality_metrics_json,
    const fs::path &global_registration_json,
    const DrizzleGeometryCacheReader *geometry_cache,
    const ForwardDrizzleSubdivisionParams &subdivision,
    const std::string &config_snapshot_hash,
    const std::string &acceleration_backend,
    const std::function<void(int, int)> &progress) {
  if (source_quality_cache_root.empty())
    throw std::invalid_argument("FDV2_PROD_REQUIRES_SOURCE_QUALITY_CACHE");
  const std::size_t mb =
      drizzle_cfg.memory_budget_mb ? drizzle_cfg.memory_budget_mb : 512;
  const auto quality = load_source_quality_artifact(quality_artifact, sampling,
                                                    cache, quality_cfg, mb);
  const auto weights =
      resolve_quality_frame_weights(quality, sampling, quality_cfg, mb);

  SourceQualityMapCacheReader qreader(
      source_quality_cache_root,
      compute_source_quality_identity_hash(sampling, cache.manifest_hash()),
      /*expected_config_hash=*/"");
  if (!qreader.usable())
    throw std::runtime_error("FDV2_PROD_SOURCE_QUALITY_CACHE_UNUSABLE: " +
                             qreader.error());

  const auto noise =
      load_frame_noise(source_quality_metrics_json, sampling,
                       cache.manifest_hash(),
                       compute_source_quality_config_hash(quality_cfg));
  const auto residuals =
      load_reg_residuals(global_registration_json, sampling.frames.size());

  // Participating stream: valid frames minus the local-model exclusion set,
  // in plan order (source_index-ascending is the plan's canonical order).
  struct Entry {
    const registration::FrameSamplingTransform *f;
    bool has_qc, has_q0, has_q1, has_qa, has_sigma2;
  };
  std::vector<Entry> stream;
  ForwardDrizzleV2ProductionResult out;
  const double half = 0.5 * static_cast<double>(drizzle_cfg.pixfrac);
  for (const auto &f : sampling.frames) {
    if (!f.valid) {
      ++out.frames_skipped_invalid;
      continue;
    }
    if (!f.source_to_canvas_affine_valid || !f.source_to_canvas.allFinite())
      throw std::invalid_argument("DRIZZLE_INVALID_TRANSFORM");
    if (f.has_smooth_local_model) {
      bool excluded = false;
      double rate = 0.0;
      std::uint64_t total = 0, discarded = 0;
      bool decided = false;
      if (geometry_cache != nullptr) {
        const auto st =
            geometry_cache->frame_stats(drizzle_cfg.pixfrac, f.source_index);
        if (st.present) {
          decided = true;
          excluded = st.excluded;
          rate = st.subdivision_error_rate;
          total = st.samples_total;
          discarded = st.samples_discarded;
        }
      }
      if (!decided) {
        // No published cache stats: run the same exclusion sweep the legacy
        // prepare path falls back to (per-sample sample_leaves count).
        std::vector<Leaf> leaves;
        leaves.reserve(16);
        total = static_cast<std::uint64_t>(sampling.source_width) *
                sampling.source_height;
        for (int y = 0; y < sampling.source_height; ++y)
          for (int x = 0; x < sampling.source_width; ++x)
            if (!sample_leaves(sampling, f, x, y, drizzle_cfg.internal_scale,
                               drizzle_cfg.pixfrac, subdivision, leaves))
              ++discarded;
        rate = static_cast<double>(discarded) / static_cast<double>(total);
        excluded =
            rate > subdivision.per_frame_inversion_error_rate_max;
      }
      out.local_model_samples_total += total;
      out.local_model_samples_discarded += discarded;
      if (excluded) {
        out.frames_excluded_subdivision_error_rate.emplace_back(f.frame_id,
                                                              rate);
        continue;
      }
    }
    Entry e;
    e.f = &f;
    e.has_qc = qreader.has("composite", f.source_index);
    e.has_q0 = qreader.has("scale_0", f.source_index);
    e.has_q1 = qreader.has("scale_1", f.source_index);
    e.has_qa = qreader.has("artifact", f.source_index);
    const std::size_t si = f.source_index;
    e.has_sigma2 =
        si < noise.size() && std::isfinite(noise[si]) && noise[si] >= 0.0f &&
        si < residuals.rms_px.size() && residuals.applicable[si] &&
        std::isfinite(residuals.rms_px[si]) && residuals.rms_px[si] >= 0.0f;
    stream.push_back(e);
  }
  if (stream.empty())
    throw std::runtime_error("FDV2_PROD_EMPTY_FRAME_STREAM");
  out.frames_participating = static_cast<int>(stream.size());

  // Nominal device planning bound when the CUDA path is requested: a fixed
  // 4 GiB, NOT measured free VRAM --- the band geometry is bound into
  // plan_hash and must be identical across a resume. A device that cannot
  // actually fit a band fails reserve() and the phase restarts on CPU.
  const std::size_t nominal_device_bytes =
      acceleration_backend == "cuda" ? (std::size_t{4} << 30) : 0;
  out.plan = make_forward_drizzle_v2_run_plan(
      sampling, drizzle_cfg, clipping_cfg, multiband_cfg,
      multiband_cfg.enabled /*emit_profiles*/,
      static_cast<std::uint64_t>(stream.size()), cache.manifest_hash(),
      quality.plan_hash, qreader.metadata().source_quality_cache_hash,
      config_snapshot_hash, mb << 20, nominal_device_bytes);

  // Provider-owned per-frame planes; the driver is serial, so a single
  // reused set satisfies the "valid until the next provider call" contract.
  Matrix2Df comp, s0, s1, art;
  std::vector<float> sigma2_buf;
  const int sh = sampling.source_height, sw = sampling.source_width;
  ForwardDrizzleV2FrameProvider provider =
      [&](std::uint64_t order, ForwardDrizzleV2FrameInput &in) -> bool {
    if (order >= stream.size()) return false;
    const Entry &e = stream[static_cast<std::size_t>(order)];
    const auto &f = *e.f;
    const auto &s2c = f.source_to_canvas;
    in.affine6[0] = s2c(0, 0);
    in.affine6[1] = s2c(0, 1);
    in.affine6[2] = s2c(0, 2);
    in.affine6[3] = s2c(1, 0);
    in.affine6[4] = s2c(1, 1);
    in.affine6[5] = s2c(1, 2);
    in.has_local_model = f.has_smooth_local_model;
    if (in.has_local_model) in.warp = make_local_warp(f, subdivision);

    const Matrix2Df &img = cache.load(f.source_index);
    if (img.rows() != sh || img.cols() != sw) return false;
    in.source = img.data();

    if (e.has_sigma2) {
      sigma2_buf = forward_drizzle_v2_sigma2_plane(
          img, noise[f.source_index], residuals.rms_px[f.source_index], half);
      in.sigma2 = sigma2_buf.data();
    } else {
      in.sigma2 = nullptr;
    }

    ForwardDrizzleV2FrameQuality q{};
    if (e.has_qc) {
      comp = qreader.read_rect("composite", f.source_index, 0, sh, 0, sw);
      q.q_composite = comp.data();
    }
    if (e.has_q0) {
      s0 = qreader.read_rect("scale_0", f.source_index, 0, sh, 0, sw);
      q.q_scale0 = s0.data();
    }
    if (e.has_q1) {
      s1 = qreader.read_rect("scale_1", f.source_index, 0, sh, 0, sw);
      q.q_scale1 = s1.data();
    }
    if (e.has_qa) {
      art = qreader.read_rect("artifact", f.source_index, 0, sh, 0, sw);
      q.q_artifact = art.data();
    }
    in.quality = q;

    const std::size_t si = f.source_index;
    ForwardDrizzleV2FrameMeta meta{};
    meta.g_eff = si < weights.size() ? weights[si] : 0.0f;
    meta.is_direct =
        (!f.model_predicted && f.model_prediction_factor == 1.0f) ? 1u : 0u;
    meta.residual_factor = f.registration_residual_factor;
    in.meta = meta;
    return true;
  };

  ForwardDrizzleV2DriverOptions opts;
  opts.prefer_cuda = acceleration_backend == "cuda";
  if (progress) opts.progress = progress;
  out.driver = run_forward_drizzle_v2(
      store_root, out.plan, sampling.source_width, sampling.source_height,
      provider, opts);
  out.local_model_samples_discarded += out.driver.local_samples_discarded;
  out.q_bin_loads = qreader.bin_loads();
  out.q_bin_cells_decoded = qreader.bin_cells_decoded();
  out.q_expanded_floats = qreader.expanded_floats();
  return out;
}

}  // namespace tile_compile::reconstruction
