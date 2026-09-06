// M6 tests for the in-memory reference multi-band fusion (plan 14.2/14.3).
// The acceptance criteria are the plan's identities:
//   * alpha == 0 everywhere  =>  X_out = C_U,L + (R - C_R,L); with U == R this
//     collapses to X_out == R exactly.
//   * U == R == F == M       =>  X_out == R for ANY alpha (plan 14.3).
//   * a missing Raw band     =>  the whole multi-band pixel is invalid.
//   * an invalid detail band =>  alpha forced to 0 there (pixel still valid).

#include "tile_compile/reconstruction/multiband_fusion.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <string>
#include <vector>

using namespace tile_compile::reconstruction;
namespace config = tile_compile::config;
namespace registration = tile_compile::registration;
using tile_compile::ColorMode;
using tile_compile::Matrix2Df;
using tile_compile::WarpMatrix;
using Catch::Approx;

namespace {

ProfilePlane plane_from(const std::vector<float> &value, int w, int h) {
  ProfilePlane p;
  p.allocate(w, h);
  p.value = value;
  for (int i = 0; i < w * h; ++i)
    p.support[i] = std::isfinite(value[i]) ? 1u : 0u;
  return p;
}

std::vector<float> smooth_field(int w, int h, float amp = 10.0f) {
  std::vector<float> v(static_cast<size_t>(w) * h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      v[static_cast<size_t>(y) * w + x] =
          50.0f + amp * std::sin(0.13f * x) + 0.7f * amp * std::cos(0.09f * y) +
          0.02f * amp * (x + y);
  return v;
}

// F = base + a fine checkerboard ripple (energy concentrated in band 1).
std::vector<float> with_fine_ripple(const std::vector<float> &base, int w, int h,
                                    float amp) {
  std::vector<float> v = base;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      v[static_cast<size_t>(y) * w + x] += ((x + y) % 2 == 0 ? amp : -amp);
  return v;
}

std::vector<std::vector<float>> const_alpha(int levels, int n, float a) {
  return std::vector<std::vector<float>>(
      static_cast<size_t>(levels), std::vector<float>(static_cast<size_t>(n), a));
}

}  // namespace

TEST_CASE("multiband: alpha == 0 with U == R reproduces R exactly (identity "
          "X_out = C_U,L + R - C_R,L)") {
  const int w = 64, h = 56, L = 3;
  const auto base = smooth_field(w, h);
  const auto rpl = plane_from(base, w, h);
  const auto fpl = plane_from(with_fine_ripple(base, w, h, 4.0f), w, h);
  const auto mpl = plane_from(with_fine_ripple(base, w, h, 2.0f), w, h);

  MultibandChannelInput in{&rpl, &rpl, &fpl, &mpl};  // U := R
  MultibandFusionParams p{L};
  auto res = fuse_multiband_channel(in, w, h, p, const_alpha(L, w * h, 0.0f));

  int checked = 0;
  for (int i = 0; i < w * h; ++i)
    if (res.support[i]) {
      REQUIRE(res.value[i] == Approx(base[i]).margin(3e-3));
      ++checked;
    }
  REQUIRE(checked > 0);
  REQUIRE(res.pixels_supported == checked);
}

TEST_CASE("multiband: U == R == F == M gives X_out == R for any alpha "
          "(plan 14.3)") {
  const int w = 60, h = 60, L = 3;
  const auto base = smooth_field(w, h);
  const auto pl = plane_from(base, w, h);
  MultibandChannelInput in{&pl, &pl, &pl, &pl};
  MultibandFusionParams p{L};

  for (float a : {0.0f, 0.5f, 1.0f}) {
    auto res = fuse_multiband_channel(in, w, h, p, const_alpha(L, w * h, a));
    for (int i = 0; i < w * h; ++i)
      if (res.support[i]) REQUIRE(res.value[i] == Approx(base[i]).margin(3e-3));
  }
}

TEST_CASE("multiband: alpha == 1 injects the Fine profile's band-1 detail "
          "(levels = 1)") {
  const int w = 64, h = 64, L = 1;
  const auto base = smooth_field(w, h);
  const float ripple = 5.0f;
  const auto upl = plane_from(base, w, h);
  const auto fpl = plane_from(with_fine_ripple(base, w, h, ripple), w, h);
  // U == R, F carries the ripple.
  MultibandChannelInput in{&upl, &upl, &fpl, nullptr};
  MultibandFusionParams p{L};

  auto zero = fuse_multiband_channel(in, w, h, p, const_alpha(L, w * h, 0.0f));
  auto one = fuse_multiband_channel(in, w, h, p, const_alpha(L, w * h, 1.0f));

  double max_diff_zero = 0.0, max_diff_one = 0.0;
  for (int i = 0; i < w * h; ++i)
    if (zero.support[i] && one.support[i]) {
      max_diff_zero = std::max(
          max_diff_zero, static_cast<double>(std::abs(zero.value[i] - base[i])));
      max_diff_one = std::max(
          max_diff_one, static_cast<double>(std::abs(one.value[i] - base[i])));
    }
  // alpha 0 -> ~ base (U==R); alpha 1 -> the injected ripple shows up.
  REQUIRE(max_diff_zero < 0.05);
  REQUIRE(max_diff_one > ripple * 0.5);
}

TEST_CASE("multiband: a masked hole in R invalidates the multi-band pixel; a "
          "hole only in F just forces alpha 0 there") {
  const int w = 72, h = 64, L = 2;
  const auto base = smooth_field(w, h);

  SECTION("hole in R") {
    auto rv = base;
    for (int y = 24; y < 34; ++y)
      for (int x = 30; x < 42; ++x)
        rv[static_cast<size_t>(y) * w + x] =
            std::numeric_limits<float>::quiet_NaN();
    const auto rpl = plane_from(rv, w, h);
    const auto fpl = plane_from(base, w, h);
    const auto mpl = plane_from(base, w, h);
    MultibandChannelInput in{&rpl, &rpl, &fpl, &mpl};
    auto res = fuse_multiband_channel(in, w, h, {L});
    // Deep inside the hole the multi-band pixel must be unsupported.
    for (int y = 27; y < 31; ++y)
      for (int x = 33; x < 39; ++x)
        REQUIRE(res.support[static_cast<size_t>(y) * w + x] == 0u);
  }

  SECTION("hole only in F") {
    auto fv = with_fine_ripple(base, w, h, 3.0f);
    for (int y = 24; y < 34; ++y)
      for (int x = 30; x < 42; ++x)
        fv[static_cast<size_t>(y) * w + x] =
            std::numeric_limits<float>::quiet_NaN();
    const auto rpl = plane_from(base, w, h);       // U == R, fully valid
    const auto fpl = plane_from(fv, w, h);
    const auto mpl = plane_from(base, w, h);
    MultibandChannelInput in{&rpl, &rpl, &fpl, &mpl};
    auto res = fuse_multiband_channel(in, w, h, {L}, const_alpha(L, w * h, 1.0f));
    // Inside the F hole: still supported, and (alpha forced to 0 for band 1,
    // band 2 from M == R) X_out ~ base.
    for (int y = 27; y < 31; ++y)
      for (int x = 33; x < 39; ++x) {
        const size_t i = static_cast<size_t>(y) * w + x;
        REQUIRE(res.support[i] == 1u);
        REQUIRE(res.value[i] == Approx(base[i]).margin(3e-3));
      }
  }
}

namespace {
ForwardDrizzleUniformResult mono_prof(const std::vector<float> &v, int w, int h,
                                      float n_eff = 40.0f, float wsum = 100.0f) {
  ForwardDrizzleUniformResult r;
  r.color_mode = tile_compile::ColorMode::MONO;
  r.internal_width = w;
  r.internal_height = h;
  r.L.allocate(w, h);
  for (int i = 0; i < w * h; ++i) {
    r.L.value[i] = v[i];
    r.L.n_eff[i] = n_eff;
    r.L.weight_sum[i] = wsum;
    r.L.support[i] = std::isfinite(v[i]) ? 1u : 0u;
  }
  return r;
}
}  // namespace

TEST_CASE("multiband orchestrator: U == R == F == M gives X_out == R "
          "(full alpha/guard/smoothing pipeline)") {
  const int w = 56, h = 56;
  const auto base = smooth_field(w, h);
  const auto p = mono_prof(base, w, h);
  config::ReconstructionMultibandConfig cfg;  // levels 3
  auto res = fuse_multiband(p, p, p, p, tile_compile::ColorMode::MONO, w, h, cfg);
  int checked = 0;
  for (int i = 0; i < w * h; ++i)
    if (res.support_L[i]) {
      REQUIRE(res.L[i] == Approx(base[i]).margin(3e-3));
      ++checked;
    }
  REQUIRE(checked > 0);
}

TEST_CASE("multiband orchestrator: a noisy Fine profile makes the energy guard "
          "pull the applied alpha below the un-guarded value") {
  const int w = 48, h = 48;
  const std::size_t n = static_cast<size_t>(w) * h;
  const auto base = smooth_field(w, h, 6.0f);
  std::vector<float> fv = base;
  for (std::size_t i = 0; i < n; ++i)
    fv[i] += (((i * 2654435761u) % 101) / 50.0f - 1.0f) * 3.0f;  // ~+-3 noise

  const auto U = mono_prof(base, w, h);
  const auto F = mono_prof(fv, w, h);
  config::ReconstructionMultibandConfig cfg;
  cfg.levels = 1;

  AdaptiveAlphaParams ap;  // A_neff = A_cov = 1 here -> alpha_pre = 1
  EnergyGuardParams strict;  // limit 1.30
  EnergyGuardParams loose;
  loose.energy_limit = 1e9;  // effectively disabled

  auto guarded = fuse_multiband(U, U, F, {}, tile_compile::ColorMode::MONO, w, h,
                                cfg, ap, strict);
  auto unguarded = fuse_multiband(U, U, F, {}, tile_compile::ColorMode::MONO, w,
                                  h, cfg, ap, loose);

  // Interior band-1 alpha: strictly guarded < effectively-1.
  const int ci = static_cast<int>(24) * w + 24;
  REQUIRE(unguarded.alpha_final[0][ci] == Approx(1.0).margin(1e-3));
  REQUIRE(guarded.alpha_final[0][ci] < 0.98f);
  REQUIRE(guarded.alpha_final[0][ci] >= 0.0f);
}

namespace {

// N MONO frames, all registered with the same integer translation so every
// source pixel maps to one internal pixel (internal_scale 1). Direct
// registration, residual factor 1.
registration::RegistrationSamplingPlan nframe_plan(int n, int sw, int sh) {
  registration::RegistrationSamplingPlan plan;
  plan.source_width = sw;
  plan.source_height = sh;
  plan.canvas_width_native = sw + 8;
  plan.canvas_height_native = sh + 8;
  plan.color_mode = ColorMode::MONO;
  for (int i = 0; i < n; ++i) {
    registration::FrameSamplingTransform f;
    f.frame_id = "f" + std::to_string(i);
    f.source_index = static_cast<std::size_t>(i);
    f.valid = true;
    WarpMatrix m = WarpMatrix::Identity();
    m(0, 2) = 4.0f;  // shift source into the canvas interior
    m(1, 2) = 4.0f;
    f.source_to_canvas = m;
    f.source_to_canvas_affine_valid = true;
    f.model_prediction_factor = 1.0f;        // direct
    f.registration_residual_factor = 1.0f;
    plan.frames.push_back(f);
  }
  return plan;
}

Matrix2Df const_map(int w, int h, float v) {
  Matrix2Df m(h, w);
  m.setConstant(v);
  return m;
}

}  // namespace

TEST_CASE("multiband reference: identical frames + constant quality maps => "
          "X_out == R (drizzle U/R/F/M + alpha maps -> fuse, end to end)") {
  const int nf = 4, sw = 40, sh = 36;
  auto plan = nframe_plan(nf, sw, sh);

  std::vector<Matrix2Df> imgs;
  for (int f = 0; f < nf; ++f) {
    Matrix2Df im(sh, sw);
    for (int y = 0; y < sh; ++y)
      for (int x = 0; x < sw; ++x)
        im(y, x) = 100.0f + 8.0f * std::sin(0.2f * x) + 5.0f * std::cos(0.15f * y);
    imgs.push_back(im);
  }
  SourceImageProvider source_of = [&](std::size_t i) -> const Matrix2Df & {
    return imgs[i];
  };

  const auto comp = const_map(sw, sh, 0.7f);
  const auto s0 = const_map(sw, sh, 0.7f);
  const auto s1 = const_map(sw, sh, 0.7f);
  const auto art = const_map(sw, sh, 0.95f);
  FrameQualityProvider quality_of = [&](std::size_t) -> FrameQualityMaps {
    return {&comp, &s0, &s1, &art};
  };

  config::ReconstructionDrizzleConfig drizzle_cfg;
  drizzle_cfg.internal_scale = 1;
  drizzle_cfg.pixfrac = 0.9f;
  drizzle_cfg.min_clip_contributors = nf + 1;  // no clipping
  config::ReconstructionClippingConfig clip_cfg;
  clip_cfg.min_n_eff = 1.0f;
  clip_cfg.min_fraction = 0.1f;

  MultibandReconstructionParams p;
  p.multiband.levels = 3;

  auto res = reconstruct_multiband_reference(plan, source_of, drizzle_cfg,
                                             clip_cfg, quality_of, p);

  // Compare against a plain Raw drizzle: X_out must reproduce R on the
  // common support (U==R==F==M in value because every quality map is the
  // same constant).
  MultibandProfileParams mbp;
  auto dz = compute_forward_drizzle_uniform_and_raw(
      plan, source_of, drizzle_cfg, clip_cfg, {}, {}, quality_of, mbp);

  int checked = 0;
  for (int i = 0; i < sw * sh; ++i)
    if (res.support_L[i] && dz.raw.L.support[i]) {
      REQUIRE(res.L[i] == Approx(dz.raw.L.value[i]).margin(5e-3));
      ++checked;
    }
  REQUIRE(checked > 0);
  REQUIRE(static_cast<int>(res.alpha_final.size()) == 3);
}

namespace {

// Exact agreement: identical NaN pattern, bitwise-equal finite values.
void require_field_bit_exact(const std::vector<float> &a,
                             const std::vector<float> &b, const char *what) {
  REQUIRE(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    const bool na = std::isnan(a[i]), nb = std::isnan(b[i]);
    if (na || nb) {
      REQUIRE(na == nb);
    } else if (a[i] != b[i]) {
      INFO(what << " mismatch at " << i << ": " << a[i] << " vs " << b[i]);
      REQUIRE(a[i] == b[i]);
    }
  }
}

}  // namespace

TEST_CASE("multiband streamed: byte-identical to whole-frame fuse_multiband "
          "across chunk heights (plan 14.7 streaming path)") {
  const int w = 64, h = 176;  // h > 2*halo + chunk => genuinely interior stripes
  const std::size_t n = static_cast<std::size_t>(w) * h;
  config::ReconstructionMultibandConfig cfg;  // levels 3
  const int L = cfg.levels;
  REQUIRE(h > 2 * multiband_fusion_halo_rows(L) + 32);  // non-vacuous

  const auto base = smooth_field(w, h, 12.0f);
  std::vector<float> fv = with_fine_ripple(base, w, h, 2.5f);
  std::vector<float> mv = base;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      mv[static_cast<std::size_t>(y) * w + x] +=
          1.6f * std::sin(0.31f * x) * std::cos(0.27f * y);  // band-2 energy
  std::vector<float> rv = base;
  for (std::size_t i = 0; i < n; ++i)
    rv[i] += (((i * 2654435761u) % 97) / 48.0f - 1.0f) * 1.1f;  // mild noise

  // Two masked holes at different heights so several stripes see an edge.
  auto punch = [&](std::vector<float> &v, int x0, int y0, int side) {
    for (int y = y0; y < y0 + side; ++y)
      for (int x = x0; x < x0 + side; ++x)
        v[static_cast<std::size_t>(y) * w + x] =
            std::numeric_limits<float>::quiet_NaN();
  };
  punch(fv, 10, 20, 7);
  punch(rv, 40, 118, 6);

  const auto U = mono_prof(base, w, h);
  const auto R = mono_prof(rv, w, h);
  const auto F = mono_prof(fv, w, h);
  const auto M = mono_prof(mv, w, h);

  // Non-constant confidence maps so adaptive alpha varies with row.
  std::vector<float> a_sep(n), a_art(n), a_reg(n);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const std::size_t i = static_cast<std::size_t>(y) * w + x;
      a_sep[i] = 0.3f + 0.6f * (0.5f + 0.5f * std::sin(0.05f * y));
      a_art[i] = 0.8f;
      a_reg[i] = 0.7f + 0.2f * std::cos(0.04f * x);
    }

  AdaptiveAlphaParams ap;
  EnergyGuardParams gp;
  const std::vector<double> floor{};

  const auto full = fuse_multiband(U, R, F, M, ColorMode::MONO, w, h, cfg, ap,
                                   gp, a_sep, a_art, a_reg, floor);

  for (int chunk : {13, 32, 64}) {
    const auto strm =
        fuse_multiband_streamed(U, R, F, M, ColorMode::MONO, w, h, cfg, chunk,
                                ap, gp, a_sep, a_art, a_reg, floor);
    INFO("chunk = " << chunk);
    require_field_bit_exact(full.L, strm.L, "X_out");
    REQUIRE(full.support_L == strm.support_L);
    REQUIRE(full.alpha_final.size() == strm.alpha_final.size());
    for (std::size_t b = 0; b < full.alpha_final.size(); ++b) {
      REQUIRE(full.alpha_final[b].empty() == strm.alpha_final[b].empty());
      require_field_bit_exact(full.alpha_final[b], strm.alpha_final[b],
                              "alpha_final");
    }
    REQUIRE(full.pixels_supported == strm.pixels_supported);
  }
}

TEST_CASE("multiband streamed: OSC (R/G/B slicing branch) is byte-identical to "
          "whole-frame fuse_multiband and pixels_supported agrees across the "
          "count route") {
  const int w = 40, h = 150;
  const std::size_t n = static_cast<std::size_t>(w) * h;
  config::ReconstructionMultibandConfig cfg;
  cfg.levels = 2;
  REQUIRE(h > 2 * multiband_fusion_halo_rows(cfg.levels) + 16);

  auto rgb_prof = [&](float phase) {
    ForwardDrizzleUniformResult r;
    r.color_mode = ColorMode::OSC;
    r.internal_width = w;
    r.internal_height = h;
    ProfilePlane *pl[3] = {&r.R, &r.G, &r.B};
    for (int c = 0; c < 3; ++c) {
      pl[c]->allocate(w, h);
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          const std::size_t i = static_cast<std::size_t>(y) * w + x;
          float v = 40.0f + 6.0f * c + 9.0f * std::sin(0.12f * x + phase) +
                    5.0f * std::cos(0.1f * y + 0.5f * c);
          // A per-channel hole so support differs between R/G/B.
          if (c == 0 && x >= 6 && x < 11 && y >= 44 && y < 49)
            v = std::numeric_limits<float>::quiet_NaN();
          if (c == 2 && x >= 25 && x < 29 && y >= 96 && y < 100)
            v = std::numeric_limits<float>::quiet_NaN();
          pl[c]->value[i] = v;
          pl[c]->n_eff[i] = 30.0f;
          pl[c]->weight_sum[i] = 80.0f;
          pl[c]->support[i] = std::isfinite(v) ? 1u : 0u;
        }
    }
    return r;
  };

  const auto U = rgb_prof(0.0f), R = rgb_prof(0.03f), F = rgb_prof(0.4f),
             M = rgb_prof(0.2f);
  std::vector<float> a_sep(n, 0.6f), a_art(n, 0.85f), a_reg(n, 0.75f);

  const auto full = fuse_multiband(U, R, F, M, ColorMode::OSC, w, h, cfg, {}, {},
                                   a_sep, a_art, a_reg, {});
  for (int chunk : {11, 32}) {
    const auto strm = fuse_multiband_streamed(U, R, F, M, ColorMode::OSC, w, h,
                                              cfg, chunk, {}, {}, a_sep, a_art,
                                              a_reg, {});
    INFO("chunk = " << chunk);
    require_field_bit_exact(full.R, strm.R, "R");
    require_field_bit_exact(full.G, strm.G, "G");
    require_field_bit_exact(full.B, strm.B, "B");
    REQUIRE(full.support_R == strm.support_R);
    REQUIRE(full.support_G == strm.support_G);
    REQUIRE(full.support_B == strm.support_B);
    for (std::size_t b = 0; b < full.alpha_final.size(); ++b)
      require_field_bit_exact(full.alpha_final[b], strm.alpha_final[b], "alpha");
    // full sums fuse_multiband_channel's per-channel count; streamed counts
    // supports directly -- the two routes must land on the same total.
    REQUIRE(full.pixels_supported == strm.pixels_supported);
    REQUIRE(full.pixels_supported > 0);
  }
}

TEST_CASE("multiband streamed: an alpha-support component joined only around a "
          "deep bend below the stripe halo -- the one non-local step (B3 flood "
          "fill, plan 14.7) -- deviates only boundedly, near the support pinch") {
  // Horseshoe support: a narrow vertical notch from the top down to a bend
  // row far deeper than the fusion halo, so a stripe covering the upper arms
  // cannot see that the two arms are one 4-connected component.
  const int w = 72, h = 224;
  const std::size_t n = static_cast<std::size_t>(w) * h;
  config::ReconstructionMultibandConfig cfg;  // levels 3
  const int halo = multiband_fusion_halo_rows(cfg.levels);
  const int bend_row = h - 40;                // >> halo below the upper arms
  const int notch_x0 = w / 2 - 2, notch_x1 = w / 2 + 2;  // 4 px wide

  const auto base = smooth_field(w, h, 10.0f);
  auto with_notch = [&](std::vector<float> v) {
    for (int y = 0; y < bend_row; ++y)
      for (int x = notch_x0; x < notch_x1; ++x)
        v[static_cast<std::size_t>(y) * w + x] =
            std::numeric_limits<float>::quiet_NaN();
    return v;
  };
  const auto U = mono_prof(with_notch(base), w, h);
  const auto F =
      mono_prof(with_notch(with_fine_ripple(base, w, h, 2.0f)), w, h);
  const auto R = mono_prof(with_notch(base), w, h);

  const auto full =
      fuse_multiband(U, R, F, U, ColorMode::MONO, w, h, cfg);
  const auto strm = fuse_multiband_streamed(U, R, F, U, ColorMode::MONO, w, h,
                                            cfg, /*chunk_rows=*/13);

  double max_abs = 0.0;
  long diverged = 0, supported = 0;
  int max_dist_from_notch = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const std::size_t i = static_cast<std::size_t>(y) * w + x;
      if (!full.support_L[i] || !strm.support_L[i]) continue;
      ++supported;
      const double d = std::abs(static_cast<double>(full.L[i]) - strm.L[i]);
      if (d > 0.0) {
        ++diverged;
        max_abs = std::max(max_abs, d);
        const int dx = std::min(std::abs(x - notch_x0), std::abs(x - notch_x1));
        max_dist_from_notch = std::max(max_dist_from_notch, dx);
      }
    }
  INFO("supported=" << supported << " diverged=" << diverged
                    << " max_abs=" << max_abs
                    << " max_dist_from_notch=" << max_dist_from_notch);
  // Bounded, rare, and hugging the pinch -- not a math error in the committed
  // rows. (Empirically the halo is wide enough that this is usually 0.)
  REQUIRE(diverged <= supported / 50);       // < 2% of supported pixels
  REQUIRE(max_abs < 0.05);                    // amplitude ~10 -> < 0.5%
  if (diverged > 0) REQUIRE(max_dist_from_notch <= halo);
}

TEST_CASE("multiband streamed: chunk >= height is exactly fuse_multiband") {
  const int w = 24, h = 20;
  config::ReconstructionMultibandConfig cfg;
  cfg.levels = 2;
  const auto base = smooth_field(w, h);
  const auto U = mono_prof(base, w, h);
  const auto F = mono_prof(with_fine_ripple(base, w, h, 2.0f), w, h);
  const auto full =
      fuse_multiband(U, U, F, U, ColorMode::MONO, w, h, cfg);
  const auto strm = fuse_multiband_streamed(U, U, F, U, ColorMode::MONO, w, h,
                                            cfg, /*chunk_rows=*/999);
  require_field_bit_exact(full.L, strm.L, "X_out");
  REQUIRE(full.support_L == strm.support_L);
}

TEST_CASE("multiband: input validation") {
  const int w = 16, h = 16;
  const auto pl = plane_from(smooth_field(w, h), w, h);
  REQUIRE_THROWS(fuse_multiband_channel({&pl, &pl, &pl, &pl}, w, h, {0}));
  REQUIRE_THROWS(fuse_multiband_channel({&pl, &pl, &pl, &pl}, w, h, {5}));
  REQUIRE_THROWS(fuse_multiband_channel({nullptr, &pl, &pl, &pl}, w, h, {2}));
  REQUIRE_THROWS(fuse_multiband_channel({&pl, &pl, nullptr, &pl}, w, h, {2}));
  REQUIRE_THROWS(
      fuse_multiband_channel({&pl, &pl, &pl, nullptr}, w, h, {2}));  // needs M
  REQUIRE_NOTHROW(
      fuse_multiband_channel({&pl, &pl, &pl, nullptr}, w, h, {1}));  // L=1 ok
}
