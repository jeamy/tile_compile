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
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
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

std::vector<float> forward_drizzle_v2_sigma2_rect(
    const Matrix2Df &window, int win_x0, int win_y0, int x0, int y0, int w,
    int h, int source_w, int source_h, double sigma_noise,
    double sigma_reg_px, double droplet_half) {
  const int wh = static_cast<int>(window.rows());
  const int ww = static_cast<int>(window.cols());
  // Absolute-coordinate accessor: outside the read window (or outside the
  // true source extent) the neighbour counts as non-finite, which reproduces
  // the whole-plane oracle's border/NaN fallback chain.
  auto at = [&](int ax, int ay) -> float {
    if (ax < 0 || ay < 0 || ax >= source_w || ay >= source_h ||
        ax < win_x0 || ay < win_y0 || ax >= win_x0 + ww ||
        ay >= win_y0 + wh)
      return std::numeric_limits<float>::quiet_NaN();
    return window(ay - win_y0, ax - win_x0);
  };
  auto diff = [&](int sx, int sy, bool x_axis) -> float {
    const float vm = at(sx - (x_axis ? 1 : 0), sy - (x_axis ? 0 : 1));
    const float vp = at(sx + (x_axis ? 1 : 0), sy + (x_axis ? 0 : 1));
    const bool fm = std::isfinite(vm), fp = std::isfinite(vp);
    if (fm && fp) return (vp - vm) * 0.5f;
    const float vc = at(sx, sy);
    if (fp && std::isfinite(vc)) return vp - vc;
    if (fm && std::isfinite(vc)) return vc - vm;
    return 0.0f;
  };
  std::vector<float> out(static_cast<std::size_t>(w) * h);
  for (int ly = 0; ly < h; ++ly)
    for (int lx = 0; lx < w; ++lx) {
      const int sx = x0 + lx, sy = y0 + ly;
      out[static_cast<std::size_t>(ly) * w + lx] = static_cast<float>(
          forward_drizzle_v2_sigma2_model(sigma_noise, diff(sx, sy, true),
                                          diff(sx, sy, false), sigma_reg_px,
                                          droplet_half));
    }
  return out;
}

void forward_drizzle_v2_affine_storage_spans(
    const std::vector<DrizzleAffineSourceSpan> &active_spans, int source_w,
    int source_h, bool need_sigma_neighbours,
    std::vector<int> &row_active_scratch,
    std::vector<DrizzleAffineSourceSpan> &storage_spans) {
  storage_spans.clear();
  if (active_spans.empty()) return;
  if (!need_sigma_neighbours) {
    storage_spans.assign(active_spans.begin(), active_spans.end());
    return;
  }
  // Sigma2 needs the in-bounds one-pixel ring: storage row r covers its own
  // active span expanded x+-1 plus the active spans of rows r-1/r+1 (the
  // vertical neighbours of those rows' samples).
  row_active_scratch.assign(static_cast<std::size_t>(source_h), -1);
  for (std::size_t i = 0; i < active_spans.size(); ++i)
    row_active_scratch[static_cast<std::size_t>(
        active_spans[i].source_y)] = static_cast<int>(i);
  for (int r = 0; r < source_h; ++r) {
    const int self =
        row_active_scratch[static_cast<std::size_t>(r)];
    const int up = r > 0 ? row_active_scratch[static_cast<std::size_t>(r - 1)]
                         : -1;
    const int dn = r + 1 < source_h
                       ? row_active_scratch[static_cast<std::size_t>(r + 1)]
                       : -1;
    if (self < 0 && up < 0 && dn < 0) continue;
    int x0 = source_w, x1 = 0;
    auto cover = [&](int idx, int pad) {
      if (idx < 0) return;
      const auto &s = active_spans[static_cast<std::size_t>(idx)];
      x0 = std::min(x0, s.x_begin - pad);
      x1 = std::max(x1, s.x_end + pad);
    };
    cover(self, 1);
    cover(up, 0);
    cover(dn, 0);
    x0 = std::max(0, x0);
    x1 = std::min(source_w, x1);
    if (x0 < x1) storage_spans.push_back({r, x0, x1});
  }
}

void forward_drizzle_v2_build_affine_samples(
    const std::vector<DrizzleAffineSourceSpan> &active_spans,
    const std::vector<DrizzleAffineSourceSpan> &storage_spans,
    const std::vector<float> &storage_values, bool sigma2_present,
    double sigma_noise, double sigma_reg_px, double droplet_half,
    int source_w, int source_h, std::vector<int> &row_storage_scratch,
    std::vector<std::size_t> &storage_offset_scratch,
    std::vector<ForwardDrizzleV2SourceSample> &out_samples) {
  out_samples.clear();
  row_storage_scratch.assign(static_cast<std::size_t>(source_h), -1);
  storage_offset_scratch.clear();
  std::size_t off = 0;
  for (std::size_t i = 0; i < storage_spans.size(); ++i) {
    const auto &s = storage_spans[i];
    row_storage_scratch[static_cast<std::size_t>(s.source_y)] =
        static_cast<int>(i);
    storage_offset_scratch.push_back(off);
    off += static_cast<std::size_t>(s.x_end - s.x_begin);
  }
  // Absolute-coordinate accessor over the packed storage: outside the true
  // source extent or a row's stored interval the neighbour is non-finite,
  // reproducing the whole-plane oracle's border/NaN fallback chain.
  auto at = [&](int ax, int ay) -> float {
    if (ax < 0 || ay < 0 || ax >= source_w || ay >= source_h)
      return std::numeric_limits<float>::quiet_NaN();
    const int idx = row_storage_scratch[static_cast<std::size_t>(ay)];
    if (idx < 0) return std::numeric_limits<float>::quiet_NaN();
    const auto &s = storage_spans[static_cast<std::size_t>(idx)];
    if (ax < s.x_begin || ax >= s.x_end)
      return std::numeric_limits<float>::quiet_NaN();
    return storage_values[storage_offset_scratch[static_cast<std::size_t>(idx)] +
                          static_cast<std::size_t>(ax - s.x_begin)];
  };
  auto diff = [&](int sx, int sy, bool x_axis) -> float {
    const float vm = at(sx - (x_axis ? 1 : 0), sy - (x_axis ? 0 : 1));
    const float vp = at(sx + (x_axis ? 1 : 0), sy + (x_axis ? 0 : 1));
    const bool fm = std::isfinite(vm), fp = std::isfinite(vp);
    if (fm && fp) return (vp - vm) * 0.5f;
    const float vc = at(sx, sy);
    if (fp && std::isfinite(vc)) return vp - vc;
    if (fm && std::isfinite(vc)) return vc - vm;
    return 0.0f;
  };
  for (const auto &span : active_spans) {
    const int sy = span.source_y;
    for (int sx = span.x_begin; sx < span.x_end; ++sx) {
      ForwardDrizzleV2SourceSample smp;
      smp.source_x = static_cast<std::uint32_t>(sx);
      smp.source_y = static_cast<std::uint32_t>(sy);
      smp.value = at(sx, sy);
      smp.sigma2 =
          sigma2_present
              ? static_cast<float>(forward_drizzle_v2_sigma2_model(
                    sigma_noise, diff(sx, sy, true), diff(sx, sy, false),
                    sigma_reg_px, droplet_half))
              : 0.0f;
      out_samples.push_back(smp);
    }
  }
}

std::size_t forward_drizzle_v2_cpu_bytes_per_native_pixel(
    int channels, int internal_scale, int reservoir_slots, bool sigma2_plane,
    bool emit_profiles) {
  const std::size_t res_slots =
      static_cast<std::size_t>(reservoir_slots);
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
  // Reservoir slots are bound by the exact global keep set of the frame
  // stream (default plan reservoir size 64 and the plan default seed),
  // capped at the historical 2*64 overflow bound.
  const auto selected_frame_orders =
      forward_drizzle_v2_selected_frame_orders(
          frame_count, 64, 11400714819323198485ULL);
  const std::size_t bpp = forward_drizzle_v2_cpu_bytes_per_native_pixel(
      channels, drizzle_cfg.internal_scale,
      static_cast<int>(std::max<std::size_t>(
          1, std::min(selected_frame_orders.size(), std::size_t{2 * 64}))),
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
  plan.shared_frame_rejection = clipping_cfg.shared_frame_rejection;
  plan.shared_frame_rejection_consensus =
      clipping_cfg.shared_frame_rejection_consensus;
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
  // The subdivision contract lives in the committed geometry cache; v2
  // never re-runs it (kept in the signature for caller compatibility).
  (void)subdivision;
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
      // v2 fails closed: a local frame without a committed geometry-cache
      // entry can never be served (no inversion/subdivision is re-run in
      // FORWARD_DRIZZLE). The cache's finalised stats drive exclusion.
      if (geometry_cache == nullptr)
        throw std::runtime_error("FDV2_PROD_LOCAL_FRAME_NO_GEOMETRY_CACHE");
      const auto st =
          geometry_cache->frame_stats(drizzle_cfg.pixfrac, f.source_index);
      if (!st.present)
        throw std::runtime_error("FDV2_PROD_LOCAL_FRAME_NO_GEOMETRY_CACHE");
      const double rate = st.subdivision_error_rate;
      const std::uint64_t total = st.samples_total;
      const std::uint64_t discarded = st.samples_discarded;
      out.local_model_samples_total += total;
      out.local_model_samples_discarded += discarded;
      if (st.excluded) {
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
  // 2 GiB for the per-pixel band state, NOT measured free VRAM --- the band
  // geometry is bound into plan_hash and must be identical across a resume
  // (no transient cudaMemGetInfo). CUDA reserve() additionally allocates
  // fixed full-source compatibility/sample/compact-Q buffers (~431 MiB at
  // 3840x2160) and optional cached-leaf storage on top of this dynamic cap.
  // The 2 GiB dynamic cap is proven by bench_m42_v2_spans, which reserved a
  // total of 2,432,577,804 bytes on the GTX 1660 Ti --- deterministic
  // headroom under the observed 3,658 MiB free while remaining
  // resume-stable. A device that cannot actually fit a band fails reserve()
  // and the phase restarts on CPU.
  const std::size_t nominal_device_bytes =
      acceleration_backend == "cuda" ? kV2NominalDeviceDynamicBytes : 0;
  out.plan = make_forward_drizzle_v2_run_plan(
      sampling, drizzle_cfg, clipping_cfg, multiband_cfg,
      multiband_cfg.enabled /*emit_profiles*/,
      static_cast<std::uint64_t>(stream.size()), cache.manifest_hash(),
      quality.plan_hash, qreader.metadata().source_quality_cache_hash,
      config_snapshot_hash, mb << 20, nominal_device_bytes);

  // Reservoir keep set over the frame-order stream: only selected orders can
  // contribute reservoir candidates, so non-selected frames never read their
  // quality maps.
  const auto selected_frame_orders = forward_drizzle_v2_selected_frame_orders(
      out.plan.frame_count, out.plan.reservoir_size,
      out.plan.reservoir_seed);
  std::vector<char> selected_order(stream.size(), 0);
  for (const std::uint64_t o : selected_frame_orders)
    if (o < selected_order.size())
      selected_order[static_cast<std::size_t>(o)] = 1;

  // Provider-owned per-frame buffers; the driver is serial, so a single
  // reused set satisfies the "valid until the next provider call" contract.
  // The source buffer carries the in-bounds one-pixel halo when the inline
  // sigma2 model is active; Q windows are the compact storage-grid cells.
  // All hot-path buffers are pre-reserved to their deterministic upper bound
  // once, outside the provider lambda; any capacity growth inside a call is
  // counted in provider_hotpath_allocations.
  std::vector<float> source_buffer;
  SourceQualityPackedWindow pqc, pq0, pq1, pqa;
  const int sh = sampling.source_height, sw = sampling.source_width;
  // Dense-quality diagnostic baseline: full storage grid at 3 bytes per
  // cell (uint16 cell + veto byte) for every present stream.
  const int qdiv = std::max(1, qreader.metadata().storage_divisor);
  const std::uint64_t q_grid_cells =
      static_cast<std::uint64_t>((sw + qdiv - 1) / qdiv) *
      static_cast<std::uint64_t>((sh + qdiv - 1) / qdiv);
  source_buffer.reserve(static_cast<std::size_t>(sw) * sh);
  for (SourceQualityPackedWindow *pq : {&pqc, &pq0, &pq1, &pqa}) {
    pq->cells.reserve(static_cast<std::size_t>(q_grid_cells));
    pq->veto.reserve(static_cast<std::size_t>(q_grid_cells));
  }
  // Geometry-cache leaf staging (tranche 6): the reader fills leaf_window
  // (leaves and io_scratch capacity preserved), the conversion buffer hands
  // the kernel the POD view. All three are pre-reserved to conservative
  // bounds computed from the committed row indices.
  const std::uint64_t leaf_capacity =
      geometry_cache != nullptr
          ? geometry_cache->max_stripe_leaf_records(
                drizzle_cfg.pixfrac,
                out.plan.band_rows * drizzle_cfg.internal_scale)
          : 0;
  DrizzleCachedLeafWindow leaf_window;
  std::vector<ForwardDrizzleV2CachedLeaf> leaf_buffer;
  leaf_window.leaves.reserve(static_cast<std::size_t>(leaf_capacity));
  leaf_buffer.reserve(static_cast<std::size_t>(leaf_capacity));
  if (geometry_cache != nullptr)
    leaf_window.io_scratch.reserve(
        static_cast<std::size_t>(geometry_cache->max_row_record_count()) *
        72u);
  // Tranche-8 ragged affine path buffers (pre-reserved to the full-source
  // deterministic bounds; capacity tracked by provider_hotpath_allocations).
  const std::size_t src_elems = static_cast<std::size_t>(sw) * sh;
  std::vector<DrizzleAffineSourceSpan> active_spans;
  std::vector<DrizzleAffineSourceSpan> storage_spans;
  std::vector<float> storage_values;
  std::vector<ForwardDrizzleV2SourceSample> samples;
  std::vector<int> row_scratch_a, row_scratch_b;
  std::vector<std::size_t> storage_offsets;
  std::vector<std::uint16_t> aqc, aq0, aq1, aqa;
  std::vector<std::uint8_t> avc, av0, av1, ava;
  SourceQualityPackedWindow aq_scratch;
  active_spans.reserve(static_cast<std::size_t>(sh));
  storage_spans.reserve(static_cast<std::size_t>(sh));
  storage_values.reserve(src_elems);
  samples.reserve(src_elems);
  row_scratch_a.reserve(static_cast<std::size_t>(sh));
  row_scratch_b.reserve(static_cast<std::size_t>(sh));
  storage_offsets.reserve(static_cast<std::size_t>(sh));
  for (std::vector<std::uint16_t> *v : {&aqc, &aq0, &aq1, &aqa})
    v->reserve(src_elems);
  for (std::vector<std::uint8_t> *v : {&avc, &av0, &av1, &ava})
    v->reserve(src_elems);
  aq_scratch.cells.reserve(static_cast<std::size_t>(q_grid_cells));
  aq_scratch.veto.reserve(static_cast<std::size_t>(q_grid_cells));
  for (std::size_t o = 0; o < stream.size(); ++o) {
    if (!selected_order[o]) continue;
    const Entry &e = stream[o];
    out.provider_quality_denominator_bytes +=
        static_cast<std::uint64_t>((e.has_qc ? 1 : 0) + (e.has_q0 ? 1 : 0) +
                                   (e.has_q1 ? 1 : 0) + (e.has_qa ? 1 : 0)) *
        q_grid_cells * 3u;
  }
  ForwardDrizzleV2FrameProvider provider =
      [&](int band_y_begin, int band_rows, std::uint64_t order,
          const ForwardDrizzleV2FramePieceSink &sink) -> bool {
    if (order >= stream.size()) return false;
    const Entry &e = stream[static_cast<std::size_t>(order)];
    const auto &f = *e.f;
    const auto &s2c = f.source_to_canvas;
    ForwardDrizzleV2FrameInput in;
    in.affine6[0] = s2c(0, 0);
    in.affine6[1] = s2c(0, 1);
    in.affine6[2] = s2c(0, 2);
    in.affine6[3] = s2c(1, 0);
    in.affine6[4] = s2c(1, 1);
    in.affine6[5] = s2c(1, 2);
    in.has_local_model = f.has_smooth_local_model;
    in.warp = ForwardDrizzleV2LocalWarp{};
    in.has_cached_geometry = false;
    in.cached_leaves = nullptr;
    in.cached_leaf_count = 0;
    in.cached_unique_source_samples = 0;
    in.quality = {};
    in.sigma2 = nullptr;
    in.sigma2_model = {};
    in.source = nullptr;
    in.source_window = {};
    in.skip = false;
    in.target_x_begin_native = 0;
    in.target_cols_native = 0;

    const std::size_t si = f.source_index;
    ForwardDrizzleV2FrameMeta meta{};
    meta.g_eff = si < weights.size() ? weights[si] : 0.0f;
    meta.is_direct =
        (!f.model_predicted && f.model_prediction_factor == 1.0f) ? 1u : 0u;
    meta.residual_factor = f.registration_residual_factor;
    in.meta = meta;

    const std::size_t cap0 =
        source_buffer.capacity() + leaf_window.leaves.capacity() +
        leaf_window.io_scratch.capacity() + leaf_buffer.capacity() +
        pqc.cells.capacity() +
        pqc.veto.capacity() + pq0.cells.capacity() + pq0.veto.capacity() +
        pq1.cells.capacity() + pq1.veto.capacity() + pqa.cells.capacity() +
        pqa.veto.capacity() + active_spans.capacity() +
        storage_spans.capacity() + storage_values.capacity() +
        samples.capacity() + row_scratch_a.capacity() +
        row_scratch_b.capacity() + storage_offsets.capacity() +
        aqc.capacity() + aq0.capacity() + aq1.capacity() + aqa.capacity() +
        avc.capacity() + av0.capacity() + av1.capacity() + ava.capacity() +
        aq_scratch.cells.capacity() + aq_scratch.veto.capacity();
    auto check_allocations = [&]() {
      if (source_buffer.capacity() + leaf_window.leaves.capacity() +
              leaf_window.io_scratch.capacity() + leaf_buffer.capacity() +
              pqc.cells.capacity() +
              pqc.veto.capacity() + pq0.cells.capacity() +
              pq0.veto.capacity() + pq1.cells.capacity() +
              pq1.veto.capacity() + pqa.cells.capacity() +
              pqa.veto.capacity() + active_spans.capacity() +
              storage_spans.capacity() + storage_values.capacity() +
              samples.capacity() + row_scratch_a.capacity() +
              row_scratch_b.capacity() + storage_offsets.capacity() +
              aqc.capacity() + aq0.capacity() + aq1.capacity() +
              aqa.capacity() + avc.capacity() + av0.capacity() +
              av1.capacity() + ava.capacity() +
              aq_scratch.cells.capacity() + aq_scratch.veto.capacity() >
          cap0)
        ++out.provider_hotpath_allocations;
    };

    // Emit one non-skip piece covering source box `box` for native target
    // range [tx, tx+tw) (tw == 0 = the full-width single-piece form used by
    // the cached-geometry path). The shared buffers are safe to reuse
    // across pieces because the driver fully consumes/enqueues each piece
    // inside the sink call.
    auto emit_piece = [&](const DrizzleSourceScanBox &box, int tx,
                          int tw) -> bool {
      const int bw = box.x1 - box.x0, bh = box.y1 - box.y0;
      if (bw <= 0 || bh <= 0) return false;
      // One read_rect_into over the box plus the in-bounds one-pixel halo
      // when the inline sigma2 model needs the absolute neighbours; the
      // buffer IS the upload (active rect points at the exact box, no core
      // copy).
      const auto src_t0 = std::chrono::steady_clock::now();
      const int hx0 = e.has_sigma2 ? std::max(0, box.x0 - 1) : box.x0;
      const int hy0 = e.has_sigma2 ? std::max(0, box.y0 - 1) : box.y0;
      const int hx1 = e.has_sigma2 ? std::min(sw, box.x1 + 1) : box.x1;
      const int hy1 = e.has_sigma2 ? std::min(sh, box.y1 + 1) : box.y1;
      cache.read_rect_into(f.source_index, hy0, hy1, hx0, hx1, source_buffer);
      out.provider_source_seconds += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - src_t0)
          .count();
      if (source_buffer.size() !=
          static_cast<std::size_t>(hy1 - hy0) * (hx1 - hx0))
        return false;
      in.source = source_buffer.data();
      in.source_window = {hx0, hy0, hx1 - hx0, hy1 - hy0,
                          box.x0 - hx0, box.y0 - hy0, bw, bh};
      if (e.has_sigma2) {
        in.sigma2_model.enabled = true;
        in.sigma2_model.sigma_noise = noise[si];
        in.sigma2_model.sigma_reg_px = residuals.rms_px[si];
        in.sigma2_model.droplet_half = half;
      }

      ForwardDrizzleV2FrameQuality q{};
      if (selected_order[static_cast<std::size_t>(order)]) {
        const auto q_t0 = std::chrono::steady_clock::now();
        auto fill = [&](bool has, const char *stream_name,
                        SourceQualityPackedWindow &dst,
                        ForwardDrizzleV2PackedQualityPlane &desc) {
          if (!has) return;
          qreader.read_packed_rect_into(stream_name, f.source_index, box.y0,
                                        box.y1, box.x0, box.x1, dst);
          if (dst.cells.empty()) return;
          desc.cells = dst.cells.data();
          desc.veto = dst.veto.empty() ? nullptr : dst.veto.data();
          desc.storage_x_begin = dst.storage_x_begin;
          desc.storage_y_begin = dst.storage_y_begin;
          desc.storage_width = dst.storage_width;
          desc.storage_height = dst.storage_height;
          desc.storage_divisor = dst.storage_divisor;
          out.provider_quality_cells_read += dst.cells.size();
        };
        fill(e.has_qc, "composite", pqc, q.qc_packed);
        fill(e.has_q0, "scale_0", pq0, q.q0_packed);
        fill(e.has_q1, "scale_1", pq1, q.q1_packed);
        fill(e.has_qa, "artifact", pqa, q.qa_packed);
        out.provider_quality_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - q_t0)
            .count();
      }
      in.quality = q;
      in.target_x_begin_native = tx;
      in.target_cols_native = tw;
      return sink(in);
    };

    // Exact source box this frame can touch in this band: the same
    // inverse-mapped box the stripe enumerator scans for affine frames;
    // for local frames the committed cache leaves define the source bbox
    // (no inversion/subdivision runs here).
    if (in.has_local_model) {
      geometry_cache->read_stripe_leaves_into(
          drizzle_cfg.pixfrac, f.source_index, drizzle_cfg.internal_scale,
          band_y_begin * drizzle_cfg.internal_scale,
          band_rows * drizzle_cfg.internal_scale, leaf_window);
      ++out.geometry_cache_enumerations;
      out.geometry_leaf_records_read += leaf_window.leaves.size();
      out.geometry_unique_source_samples += leaf_window.unique_source_samples;
      if (leaf_window.leaves.empty()) {
        in.skip = true;
        return sink(in);
      }
      const std::size_t nl = leaf_window.leaves.size();
      leaf_buffer.resize(nl);
      // Explicit field-by-field conversion: the two record types are
      // distinct types and size equality alone is no layout contract.
      for (std::size_t li = 0; li < nl; ++li) {
        const DrizzleCachedLeaf &s = leaf_window.leaves[li];
        ForwardDrizzleV2CachedLeaf &d = leaf_buffer[li];
        d.source_x = s.source_x;
        d.source_y = s.source_y;
        d.channel = s.channel;
        d.leaf_order = s.leaf_order;
        for (int k = 0; k < 4; ++k) {
          d.x[k] = s.x[k];
          d.y[k] = s.y[k];
        }
      }
      in.has_cached_geometry = true;
      in.cached_leaves = leaf_buffer.data();
      in.cached_leaf_count = nl;
      in.cached_unique_source_samples = leaf_window.unique_source_samples;
      // Cached-local frames stay a single full-width piece.
      const DrizzleSourceScanBox box{leaf_window.source_y0,
                                     leaf_window.source_y1,
                                     leaf_window.source_x0,
                                     leaf_window.source_x1};
      if (!emit_piece(box, 0, 0)) return false;
      check_allocations();
      return true;
    }

    // Affine frame (tranche 8): the canonical ragged source-row spans
    // enumerate exactly the source pixels whose pixfrac droplet can
    // intersect the band over the FULL native target width --- one
    // full-target piece per frame, no axis-aligned bounding-box inflation
    // for rotated/sheared transforms.
    const auto src_t0 = std::chrono::steady_clock::now();
    drizzle_affine_source_spans_into(sampling, f, drizzle_cfg.pixfrac,
                                     band_y_begin, band_rows, active_spans);
    if (active_spans.empty()) {
      // No source pixel can reach the band: advance bookkeeping only, no
      // source/Q reads and no launched samples.
      ++out.empty_affine_tiles;
      in.skip = true;
      return sink(in);
    }
    forward_drizzle_v2_affine_storage_spans(active_spans, sw, sh,
                                          e.has_sigma2, row_scratch_a,
                                          storage_spans);
    cache.read_row_intervals_into(f.source_index, storage_spans,
                                  storage_values);
    out.provider_source_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - src_t0)
        .count();
    // noise/rms_px are only populated when the predecessor artifacts carry
    // them; has_sigma2 already proved the bounds, so index conditionally.
    const double sig_noise = e.has_sigma2 ? noise[si] : 0.0;
    const double sig_reg = e.has_sigma2 ? residuals.rms_px[si] : 0.0;
    forward_drizzle_v2_build_affine_samples(
        active_spans, storage_spans, storage_values, e.has_sigma2, sig_noise,
        sig_reg, half, sw, sh, row_scratch_b, storage_offsets, samples);
    in.has_affine_samples = true;
    in.affine_samples = samples.data();
    in.affine_sample_count = samples.size();
    in.sigma2_present = e.has_sigma2;

    if (selected_order[static_cast<std::size_t>(order)]) {
      const auto q_t0 = std::chrono::steady_clock::now();
      const char *names[4] = {"composite", "scale_0", "scale_1", "artifact"};
      const bool hasq[4] = {e.has_qc, e.has_q0, e.has_q1, e.has_qa};
      std::vector<std::uint16_t> *cells4[4] = {&aqc, &aq0, &aq1, &aqa};
      std::vector<std::uint8_t> *veto4[4] = {&avc, &av0, &av1, &ava};
      const std::uint16_t *cptr[4] = {};
      const std::uint8_t *vptr[4] = {};
      std::uint32_t mask = 0;
      for (int k = 0; k < 4; ++k) {
        if (!hasq[k]) continue;
        const std::uint64_t c0 = qreader.bin_cells_decoded();
        qreader.read_packed_samples_into(names[k], f.source_index,
                                         active_spans, *cells4[k],
                                         *veto4[k], aq_scratch);
        out.provider_quality_cells_read +=
            qreader.bin_cells_decoded() - c0;
        if (cells4[k]->empty()) continue;
        cptr[k] = cells4[k]->data();
        vptr[k] = veto4[k]->empty() ? nullptr : veto4[k]->data();
        mask |= 1u << k;
      }
      in.aligned_quality.qc = cptr[0];
      in.aligned_quality.q0 = cptr[1];
      in.aligned_quality.q1 = cptr[2];
      in.aligned_quality.qa = cptr[3];
      in.aligned_quality.vc = vptr[0];
      in.aligned_quality.v0 = vptr[1];
      in.aligned_quality.v1 = vptr[2];
      in.aligned_quality.va = vptr[3];
      in.aligned_quality.presence_mask = mask;
      out.provider_quality_seconds += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - q_t0)
          .count();
    }
    if (!sink(in)) return false;
    check_allocations();
    return true;
  };


  const std::uint64_t src_bytes0 = cache.bytes_read();
  const std::uint64_t src_calls0 = cache.rect_read_calls();
  const std::uint64_t q_expanded0 = qreader.expanded_floats();
  const std::uint64_t geom_bytes0 =
      geometry_cache != nullptr ? geometry_cache->leaf_record_bytes_read() : 0;
  ForwardDrizzleV2DriverOptions opts;
  // shared_frame_rejection has no CUDA implementation yet (see
  // ReconstructionClippingConfig::shared_frame_rejection): force CPU rather
  // than silently ignoring it or breaking CPU/CUDA parity.
  opts.prefer_cuda =
      acceleration_backend == "cuda" && !out.plan.shared_frame_rejection;
  opts.cached_leaf_capacity = leaf_capacity;
  if (progress) opts.progress = progress;
  out.driver = run_forward_drizzle_v2(
      store_root, out.plan, sampling.source_width, sampling.source_height,
      provider, opts);
  out.local_model_samples_discarded += out.driver.local_samples_discarded;
  out.q_bin_loads = qreader.bin_loads();
  out.q_bin_cells_decoded = qreader.bin_cells_decoded();
  out.q_expanded_floats = qreader.expanded_floats();
  out.provider_source_bytes_read = cache.bytes_read() - src_bytes0;
  out.provider_source_read_calls = cache.rect_read_calls() - src_calls0;
  out.provider_quality_expanded_floats =
      qreader.expanded_floats() - q_expanded0;
  out.geometry_leaf_record_bytes_read =
      geometry_cache != nullptr
          ? geometry_cache->leaf_record_bytes_read() - geom_bytes0
          : 0;
  return out;
}

}  // namespace tile_compile::reconstruction
