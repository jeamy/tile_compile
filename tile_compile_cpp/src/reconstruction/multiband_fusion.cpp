#include "tile_compile/reconstruction/multiband_fusion.hpp"

#include "tile_compile/reconstruction/atrous_decomposition.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tile_compile::reconstruction {
namespace {

float nanf_() { return std::numeric_limits<float>::quiet_NaN(); }

AtrousDecomposition decompose_plane(const ProfilePlane &p, int w, int h,
                                    int levels, const char *which) {
  if (p.width != w || p.height != h ||
      p.value.size() != static_cast<std::size_t>(w) * h ||
      p.support.size() != static_cast<std::size_t>(w) * h)
    throw std::invalid_argument(std::string("MULTIBAND_PROFILE_GEOMETRY: ") +
                                which);
  return atrous_decompose(p.value, p.support, w, h, levels);
}

// Which profile's detail band feeds level j (1-based), plan 14.3.
enum class BandSource { kFine, kMedium, kRaw };
BandSource band_source(int level, int levels) {
  if (level == 1) return BandSource::kFine;
  if (level == 2 && levels >= 2) return BandSource::kMedium;
  return BandSource::kRaw;
}

}  // namespace

MultibandChannelResult fuse_multiband_channel(
    const MultibandChannelInput &in, int width, int height,
    const MultibandFusionParams &params,
    const std::vector<std::vector<float>> &alpha_by_band) {
  const int L = params.levels;
  if (L < 1 || L > 4) throw std::invalid_argument("MULTIBAND_LEVELS_RANGE");
  if (!in.uniform || !in.raw)
    throw std::invalid_argument("MULTIBAND_MISSING_U_OR_R");
  if (!in.fine) throw std::invalid_argument("MULTIBAND_MISSING_FINE");
  if (L >= 2 && !in.medium)
    throw std::invalid_argument("MULTIBAND_MISSING_MEDIUM");
  const std::size_t n = static_cast<std::size_t>(width) * height;
  if (!alpha_by_band.empty() &&
      static_cast<int>(alpha_by_band.size()) != L)
    throw std::invalid_argument("MULTIBAND_ALPHA_BAND_COUNT");
  for (const auto &a : alpha_by_band)
    if (!a.empty() && a.size() != n)
      throw std::invalid_argument("MULTIBAND_ALPHA_MAP_SIZE");

  const auto du = decompose_plane(*in.uniform, width, height, L, "uniform");
  const auto dr = decompose_plane(*in.raw, width, height, L, "raw");
  const auto df = decompose_plane(*in.fine, width, height, L, "fine");
  const bool have_m = L >= 2;
  const AtrousDecomposition dm =
      have_m ? decompose_plane(*in.medium, width, height, L, "medium")
             : AtrousDecomposition{};

  MultibandChannelResult out;
  out.width = width;
  out.height = height;
  out.value.assign(n, nanf_());
  out.support.assign(n, 0u);

  auto alpha_at = [&](int band_index /*0-based*/, std::size_t i) -> double {
    if (alpha_by_band.empty()) return 1.0;
    const auto &a = alpha_by_band[static_cast<std::size_t>(band_index)];
    if (a.empty()) return 1.0;
    const double v = a[i];
    return std::isfinite(v) ? std::clamp(v, 0.0, 1.0) : 0.0;
  };

  for (std::size_t i = 0; i < n; ++i) {
    // Missing Raw coarse or any Raw band => whole multi-band pixel invalid
    // (plan 14.2). Uniform coarse must also be valid (it supplies C_U,L).
    if (!du.coarse_support[i] || !dr.coarse_support[i]) continue;
    bool raw_bands_ok = true;
    for (const auto &b : dr.bands)
      if (!b.support[i]) { raw_bands_ok = false; break; }
    if (!raw_bands_ok) continue;

    double x = static_cast<double>(du.coarse[i]);
    for (int j = 1; j <= L; ++j) {
      const std::size_t bj = static_cast<std::size_t>(j - 1);
      const double d_r = dr.bands[bj].detail[i];
      const BandSource src = band_source(j, L);
      if (src == BandSource::kRaw) {
        x += d_r;  // alpha ineffective for R bands (plan 14.3)
        continue;
      }
      const AtrousDecomposition &pd = src == BandSource::kFine ? df : dm;
      double alpha = alpha_at(j - 1, i);
      double d_p = d_r;  // default: invalid detail profile => alpha 0
      if (pd.bands[bj].support[i] && std::isfinite(pd.bands[bj].detail[i]))
        d_p = pd.bands[bj].detail[i];
      else
        alpha = 0.0;
      x += d_r + alpha * (d_p - d_r);
    }

    out.value[i] = static_cast<float>(x);
    out.support[i] = 1u;
    ++out.pixels_supported;
  }
  return out;
}

// ---- Orchestrator ------------------------------------------------------------

namespace {

std::array<const ProfilePlane *, 3> chans(const ForwardDrizzleUniformResult &p,
                                          ColorMode mode) {
  if (mode == ColorMode::MONO) return {&p.L, nullptr, nullptr};
  return {&p.R, &p.G, &p.B};
}

// Luminance-combined detail for band j (0-based) of a profile: MONO -> D_L,j,
// OSC -> 0.25 D_R,j + 0.5 D_G,j + 0.25 D_B,j (plan 14.5). Valid only where
// every contributing channel band is valid.
void luma_band(const ForwardDrizzleUniformResult &profile, ColorMode mode,
               int w, int h, int levels, int band0,
               std::vector<float> &out_detail,
               std::vector<uint8_t> &out_support) {
  const std::size_t n = static_cast<std::size_t>(w) * h;
  out_detail.assign(n, std::numeric_limits<float>::quiet_NaN());
  out_support.assign(n, 0u);
  const auto ch = chans(profile, mode);
  const int nch = mode == ColorMode::MONO ? 1 : 3;
  const double wgt[3] = {mode == ColorMode::MONO ? 1.0 : kWorkingLumaWeightsOsc[0],
                         kWorkingLumaWeightsOsc[1], kWorkingLumaWeightsOsc[2]};
  std::array<AtrousDecomposition, 3> dec;
  for (int c = 0; c < nch; ++c)
    dec[c] = atrous_decompose(ch[c]->value, ch[c]->support, w, h, levels);
  for (std::size_t i = 0; i < n; ++i) {
    double acc = 0.0;
    bool ok = true;
    for (int c = 0; c < nch; ++c) {
      const auto &b = dec[c].bands[static_cast<std::size_t>(band0)];
      if (!b.support[i] || !std::isfinite(b.detail[i])) { ok = false; break; }
      acc += wgt[c] * b.detail[i];
    }
    if (ok) {
      out_detail[i] = static_cast<float>(acc);
      out_support[i] = 1u;
    }
  }
}

}  // namespace

MultibandResult fuse_multiband(
    const ForwardDrizzleUniformResult &uniform,
    const ForwardDrizzleUniformResult &raw,
    const ForwardDrizzleUniformResult &fine,
    const ForwardDrizzleUniformResult &medium, ColorMode mode, int width,
    int height, const config::ReconstructionMultibandConfig &mb_cfg,
    const AdaptiveAlphaParams &alpha_params,
    const EnergyGuardParams &guard_params,
    const std::vector<float> &a_separation, const std::vector<float> &a_artifact,
    const std::vector<float> &a_registration,
    const std::vector<double> &background_band_floor) {
  const int L = mb_cfg.levels;
  if (L < 1 || L > 4) throw std::invalid_argument("MULTIBAND_LEVELS_RANGE");
  if (!background_band_floor.empty() &&
      static_cast<int>(background_band_floor.size()) != L)
    throw std::invalid_argument("MULTIBAND_FLOOR_BAND_COUNT");

  // 1. adaptive alpha (F/M bands only; Raw bands empty).
  auto alpha = compute_adaptive_alpha(uniform, fine, medium, mode, width, height,
                                      L, alpha_params, a_separation, a_artifact,
                                      a_registration);

  // 2. energy guard + 3. B3 smoothing, per Fine/Medium band, on luma.
  MultibandResult out;
  out.alpha_final.assign(static_cast<std::size_t>(L), {});
  for (int j = 1; j <= L; ++j) {
    const std::size_t bj = static_cast<std::size_t>(j - 1);
    if (alpha[bj].empty()) continue;  // Raw-sourced band
    const ForwardDrizzleUniformResult &profile = (j == 1) ? fine : medium;
    std::vector<float> dr_luma, dp_luma;
    std::vector<uint8_t> sr, sp;
    luma_band(raw, mode, width, height, L, j - 1, dr_luma, sr);
    luma_band(profile, mode, width, height, L, j - 1, dp_luma, sp);
    std::vector<uint8_t> support(sr.size(), 0u);
    for (std::size_t i = 0; i < support.size(); ++i)
      support[i] = (sr[i] && sp[i]) ? 1u : 0u;
    const double floor_j =
        background_band_floor.empty() ? 0.0 : background_band_floor[bj];
    auto guarded = apply_energy_guard(alpha[bj], dr_luma, dp_luma, support,
                                      width, height,
                                      energy_guard_window_radius(j), floor_j,
                                      guard_params);
    out.alpha_final[bj] =
        smooth_alpha_b3(guarded, support, width, height);
  }

  // 4. per-channel band blend with the shared, guarded, smoothed alpha.
  MultibandFusionParams fp{L};
  const auto uc = chans(uniform, mode), rc = chans(raw, mode),
             fc = chans(fine, mode), mc = chans(medium, mode);
  const int nch = mode == ColorMode::MONO ? 1 : 3;
  std::vector<float> *outv[3] = {&out.R, &out.G, &out.B};
  std::vector<uint8_t> *outs[3] = {&out.support_R, &out.support_G,
                                   &out.support_B};
  if (mode == ColorMode::MONO) { outv[0] = &out.L; outs[0] = &out.support_L; }
  for (int c = 0; c < nch; ++c) {
    MultibandChannelInput in{uc[c], rc[c], fc[c], L >= 2 ? mc[c] : nullptr};
    auto r = fuse_multiband_channel(in, width, height, fp, out.alpha_final);
    *outv[c] = std::move(r.value);
    *outs[c] = std::move(r.support);
    out.pixels_supported += r.pixels_supported;
  }
  out.width = width;
  out.height = height;
  out.mode = mode;
  return out;
}

// ---- Streamed fusion ------------------------------------------------------

namespace {

ProfilePlane slice_plane(const ProfilePlane &src, int w, int ys, int ye) {
  ProfilePlane p;
  if (src.empty()) return p;
  const int sub_h = ye - ys;
  p.width = w;
  p.height = sub_h;
  const std::size_t off = static_cast<std::size_t>(ys) * w;
  const std::size_t cnt = static_cast<std::size_t>(sub_h) * w;
  auto cut = [&](const auto &v, auto &dst) {
    if (v.empty()) return;
    dst.assign(v.begin() + off, v.begin() + off + cnt);
  };
  cut(src.value, p.value);
  cut(src.weight_sum, p.weight_sum);
  cut(src.n_eff, p.n_eff);
  cut(src.support, p.support);
  return p;
}

ForwardDrizzleUniformResult slice_profiles(const ForwardDrizzleUniformResult &s,
                                           ColorMode mode, int w, int ys,
                                           int ye) {
  ForwardDrizzleUniformResult o;
  o.color_mode = s.color_mode;
  o.internal_width = w;
  o.internal_height = ye - ys;
  if (mode == ColorMode::MONO) {
    o.L = slice_plane(s.L, w, ys, ye);
  } else {
    o.R = slice_plane(s.R, w, ys, ye);
    o.G = slice_plane(s.G, w, ys, ye);
    o.B = slice_plane(s.B, w, ys, ye);
  }
  return o;
}

std::vector<float> slice_vec(const std::vector<float> &v, int w, int ys,
                             int ye) {
  if (v.empty()) return {};
  const std::size_t off = static_cast<std::size_t>(ys) * w;
  const std::size_t cnt = static_cast<std::size_t>(ye - ys) * w;
  return {v.begin() + off, v.begin() + off + cnt};
}

}  // namespace

int multiband_fusion_halo_rows(int levels) {
  if (levels < 1) levels = 1;
  if (levels > 4) levels = 4;
  const int atrous_reach = 2 * ((1 << levels) - 1);
  return atrous_reach + energy_guard_window_radius(levels) + 2;
}

MultibandResult fuse_multiband_streamed(
    const ForwardDrizzleUniformResult &uniform,
    const ForwardDrizzleUniformResult &raw,
    const ForwardDrizzleUniformResult &fine,
    const ForwardDrizzleUniformResult &medium, ColorMode mode, int width,
    int height, const config::ReconstructionMultibandConfig &mb_cfg,
    int chunk_rows, const AdaptiveAlphaParams &alpha_params,
    const EnergyGuardParams &guard_params,
    const std::vector<float> &a_separation, const std::vector<float> &a_artifact,
    const std::vector<float> &a_registration,
    const std::vector<double> &background_band_floor) {
  const int L = mb_cfg.levels;
  if (L < 1 || L > 4) throw std::invalid_argument("MULTIBAND_LEVELS_RANGE");
  if (width <= 0 || height <= 0)
    throw std::invalid_argument("MULTIBAND_STREAM_DIMENSIONS");
  if (chunk_rows <= 0 || chunk_rows >= height) {
    return fuse_multiband(uniform, raw, fine, medium, mode, width, height,
                          mb_cfg, alpha_params, guard_params, a_separation,
                          a_artifact, a_registration, background_band_floor);
  }
  const int halo = multiband_fusion_halo_rows(L);
  const std::size_t n = static_cast<std::size_t>(width) * height;

  MultibandResult out;
  out.width = width;
  out.height = height;
  out.mode = mode;
  std::vector<float> *outv[3];
  std::vector<uint8_t> *outs[3];
  int nch;
  if (mode == ColorMode::MONO) {
    out.L.assign(n, nanf_());
    out.support_L.assign(n, 0u);
    outv[0] = &out.L;
    outs[0] = &out.support_L;
    nch = 1;
  } else {
    out.R.assign(n, nanf_());
    out.G.assign(n, nanf_());
    out.B.assign(n, nanf_());
    out.support_R.assign(n, 0u);
    out.support_G.assign(n, 0u);
    out.support_B.assign(n, 0u);
    outv[0] = &out.R; outv[1] = &out.G; outv[2] = &out.B;
    outs[0] = &out.support_R; outs[1] = &out.support_G; outs[2] = &out.support_B;
    nch = 3;
  }
  out.alpha_final.assign(static_cast<std::size_t>(L), {});

  for (int y0 = 0; y0 < height; y0 += chunk_rows) {
    const int y1 = std::min(height, y0 + chunk_rows);
    const int ys = std::max(0, y0 - halo);
    const int ye = std::min(height, y1 + halo);
    const int sub_h = ye - ys;

    const auto su = slice_profiles(uniform, mode, width, ys, ye);
    const auto sr = slice_profiles(raw, mode, width, ys, ye);
    const auto sf = slice_profiles(fine, mode, width, ys, ye);
    const auto sm = L >= 2 ? slice_profiles(medium, mode, width, ys, ye)
                           : ForwardDrizzleUniformResult{};

    auto stripe = fuse_multiband(
        su, sr, sf, sm, mode, width, sub_h, mb_cfg, alpha_params, guard_params,
        slice_vec(a_separation, width, ys, ye),
        slice_vec(a_artifact, width, ys, ye),
        slice_vec(a_registration, width, ys, ye), background_band_floor);

    const std::size_t core = static_cast<std::size_t>(y1 - y0) * width;
    const std::size_t src_off = static_cast<std::size_t>(y0 - ys) * width;
    const std::size_t dst_off = static_cast<std::size_t>(y0) * width;

    std::vector<float> *sv[3];
    std::vector<uint8_t> *ss[3];
    if (mode == ColorMode::MONO) { sv[0] = &stripe.L; ss[0] = &stripe.support_L; }
    else {
      sv[0] = &stripe.R; sv[1] = &stripe.G; sv[2] = &stripe.B;
      ss[0] = &stripe.support_R; ss[1] = &stripe.support_G; ss[2] = &stripe.support_B;
    }
    for (int c = 0; c < nch; ++c) {
      std::copy(sv[c]->begin() + src_off, sv[c]->begin() + src_off + core,
                outv[c]->begin() + dst_off);
      std::copy(ss[c]->begin() + src_off, ss[c]->begin() + src_off + core,
                outs[c]->begin() + dst_off);
    }
    for (int b = 0; b < L; ++b) {
      if (stripe.alpha_final[static_cast<std::size_t>(b)].empty()) continue;
      auto &dst = out.alpha_final[static_cast<std::size_t>(b)];
      if (dst.empty()) dst.assign(n, nanf_());
      const auto &s = stripe.alpha_final[static_cast<std::size_t>(b)];
      std::copy(s.begin() + src_off, s.begin() + src_off + core,
                dst.begin() + dst_off);
    }
  }

  for (int c = 0; c < nch; ++c)
    for (std::size_t i = 0; i < n; ++i)
      if ((*outs[c])[i]) ++out.pixels_supported;
  return out;
}

MultibandResult reconstruct_multiband_reference(
    const registration::RegistrationSamplingPlan &plan,
    const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &drizzle_cfg,
    const config::ReconstructionClippingConfig &clip_cfg,
    const FrameQualityProvider &quality_of,
    const MultibandReconstructionParams &params,
    const ForwardDrizzleSubdivisionParams &subdivision,
    const std::vector<float> &g_eff_by_source_index) {
  const int L = params.multiband.levels;
  if (L < 1 || L > 4) throw std::invalid_argument("MULTIBAND_LEVELS_RANGE");

  MultibandProfileParams mb;
  mb.emit_fine = true;             // D1 is always Fine
  mb.emit_medium = L >= 2;         // D2 is Medium
  mb.emit_alpha_confidence = true;
  mb.fine_quality_exponent = params.multiband.fine_quality_exponent;
  mb.medium_quality_exponent = params.multiband.medium_quality_exponent;
  mb.alpha_confidence = params.alpha_confidence;

  const auto dz = compute_forward_drizzle_uniform_and_raw(
      plan, source_of, drizzle_cfg, clip_cfg, subdivision, g_eff_by_source_index,
      quality_of, mb);

  const int w = dz.uniform.internal_width;
  const int h = dz.uniform.internal_height;
  return fuse_multiband(dz.uniform, dz.raw, dz.fine, dz.medium, plan.color_mode,
                        w, h, params.multiband, params.alpha, params.guard,
                        dz.a_separation, dz.a_artifact, dz.a_registration,
                        params.background_band_floor);
}

}  // namespace tile_compile::reconstruction
