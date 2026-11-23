#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/errors.hpp"
#include "tile_compile/reconstruction/luma_denoise.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <random>
#include <vector>

using tile_compile::Matrix2Df;

namespace {
double stddev_of(const Matrix2Df &m) {
  const double mean = m.mean();
  double acc = 0.0;
  for (int y = 0; y < m.rows(); ++y)
    for (int x = 0; x < m.cols(); ++x) {
      const double d = m(y, x) - mean;
      acc += d * d;
    }
  return std::sqrt(acc / static_cast<double>(m.rows() * m.cols()));
}
} // namespace

TEST_CASE("luma denoise off by default and a no-op when disabled",
          "[luma-denoise]") {
  tile_compile::config::Config cfg;
  REQUIRE(cfg.luma_denoise.enabled == false);

  constexpr int W = 96, H = 64;
  Matrix2Df r(H, W), g(H, W), b(H, W);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      r(y, x) = 10.0f + static_cast<float>((x + y) % 7);
      g(y, x) = 10.0f;
      b(y, x) = 10.0f - static_cast<float>((x + y) % 5);
    }
  const Matrix2Df r_before = r, g_before = g, b_before = b;

  const auto stats = tile_compile::reconstruction::luma_denoise_rgb_inplace(
      r, g, b, cfg.luma_denoise);

  REQUIRE(stats.applied == false);
  REQUIRE((r - r_before).cwiseAbs().maxCoeff() == 0.0f);
  REQUIRE((g - g_before).cwiseAbs().maxCoeff() == 0.0f);
  REQUIRE((b - b_before).cwiseAbs().maxCoeff() == 0.0f);
}

TEST_CASE("luma denoise reduces flat-background brightness noise without "
          "shifting color",
          "[luma-denoise]") {
  constexpr int W = 200, H = 160;
  Matrix2Df r(H, W), g(H, W), b(H, W);

  // Flat sky with independent per-pixel luma noise, at a fixed, non-neutral
  // color ratio (R:G:B = 1.10:1.00:0.85) so a color shift would be visible
  // as a change in that ratio, not just in absolute brightness.
  std::mt19937 rng(42);
  std::normal_distribution<float> noise(0.0f, 3.0f);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const float base = 50.0f + noise(rng);
      g(y, x) = base;
      r(y, x) = base * 1.10f;
      b(y, x) = base * 0.85f;
    }
  }

  tile_compile::config::LumaDenoiseConfig cfg;
  cfg.enabled = true;
  cfg.star_protection.enabled = false;
  cfg.structure_protection.enabled = false;
  cfg.blend_amount = 1.0f;
  cfg.wavelet.levels = 4;
  cfg.wavelet.threshold_scale = 2.0f;

  const float g_mean_before = g.mean();
  const float ratio_r_before = r.mean() / g_mean_before;
  const float ratio_b_before = b.mean() / g_mean_before;
  const double g_sigma_before = stddev_of(g);
  const Matrix2Df rg_before = r - g;
  const Matrix2Df bg_before = b - g;

  const auto stats = tile_compile::reconstruction::luma_denoise_rgb_inplace(
      r, g, b, cfg);

  REQUIRE(stats.applied);
  REQUIRE(stats.valid_pixels == static_cast<std::uint64_t>(W * H));

  const double g_sigma_after = stddev_of(g);
  // Flat background noise is substantially reduced.
  REQUIRE(g_sigma_after < 0.5 * g_sigma_before);

  // The mean brightness and the R/G, B/G color ratios survive intact --
  // only per-pixel noise was smoothed, not the signal or its color.
  REQUIRE(g.mean() == Catch::Approx(g_mean_before).margin(0.5f));
  const float ratio_r_after = r.mean() / g.mean();
  const float ratio_b_after = b.mean() / g.mean();
  REQUIRE(ratio_r_after == Catch::Approx(ratio_r_before).margin(0.01f));
  REQUIRE(ratio_b_after == Catch::Approx(ratio_b_before).margin(0.01f));
  REQUIRE(((r - g) - rg_before).cwiseAbs().maxCoeff() < 1.0e-5f);
  REQUIRE(((b - g) - bg_before).cwiseAbs().maxCoeff() < 1.0e-5f);
}

TEST_CASE("luma denoise star protection keeps a bright point source sharp",
          "[luma-denoise]") {
  constexpr int W = 160, H = 160;
  Matrix2Df r(H, W), g(H, W), b(H, W);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      g(y, x) = 20.0f;
      r(y, x) = 20.0f;
      b(y, x) = 20.0f;
    }
  // A compact, very bright "star" at the center.
  const int cy = H / 2, cx = W / 2;
  for (int dy = -2; dy <= 2; ++dy)
    for (int dx = -2; dx <= 2; ++dx)
      if (dx * dx + dy * dy <= 4) {
        g(cy + dy, cx + dx) = 500.0f;
        r(cy + dy, cx + dx) = 500.0f;
        b(cy + dy, cx + dx) = 500.0f;
      }
  const float star_peak_before = g(cy, cx);

  tile_compile::config::LumaDenoiseConfig cfg;
  cfg.enabled = true;
  cfg.star_protection.enabled = true;
  cfg.star_protection.threshold_sigma = 5.0f;
  cfg.star_protection.dilate_px = 3;
  cfg.structure_protection.enabled = false;
  cfg.luma_guard_strength = 1.0f;
  cfg.blend_amount = 1.0f;
  cfg.wavelet.levels = 4;
  cfg.wavelet.threshold_scale = 1.0f;

  const auto stats = tile_compile::reconstruction::luma_denoise_rgb_inplace(
      r, g, b, cfg);

  REQUIRE(stats.applied);
  REQUIRE(stats.star_protected_fraction > 0.0);
  // A fully-protected star (luma_guard_strength: 1.0) keeps its peak intact
  // instead of being smoothed toward the surrounding background.
  REQUIRE(g(cy, cx) == Catch::Approx(star_peak_before).margin(1.0f));
}

TEST_CASE("luma_denoise config ranges are validated", "[config][luma-denoise]") {
  tile_compile::config::Config cfg;

  cfg.luma_denoise.luma_guard_strength = 1.5f;
  REQUIRE_THROWS_AS(cfg.validate(), tile_compile::ValidationError);
  cfg.luma_denoise.luma_guard_strength = 0.85f;

  cfg.luma_denoise.blend_amount = -0.1f;
  REQUIRE_THROWS_AS(cfg.validate(), tile_compile::ValidationError);
  cfg.luma_denoise.blend_amount = 0.85f;

  cfg.luma_denoise.wavelet.levels = 0;
  REQUIRE_THROWS_AS(cfg.validate(), tile_compile::ValidationError);
  cfg.luma_denoise.wavelet.levels = 3;

  REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("luma denoise excludes invalid canvas pixels",
          "[luma-denoise][mask]") {
  constexpr int W = 80, H = 48;
  Matrix2Df r = Matrix2Df::Constant(H, W, 20.0f);
  Matrix2Df g = r;
  Matrix2Df b = r;
  std::vector<std::uint8_t> valid(static_cast<size_t>(W * H), 1);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < 10; ++x) {
      valid[static_cast<size_t>(y * W + x)] = 0;
      r(y, x) = g(y, x) = b(y, x) = -1000.0f;
    }
  const Matrix2Df before = r;
  tile_compile::config::LumaDenoiseConfig cfg;
  cfg.enabled = true;
  const auto stats = tile_compile::reconstruction::luma_denoise_rgb_inplace(
      r, g, b, cfg, nullptr, &valid);
  REQUIRE(stats.valid_pixels == static_cast<std::uint64_t>((W - 10) * H));
  REQUIRE((r.leftCols(10) - before.leftCols(10)).cwiseAbs().maxCoeff() == 0.0f);
  REQUIRE((r.rightCols(W - 10).array() - 20.0f).abs().maxCoeff() < 1.0e-4f);
}

TEST_CASE("luma denoise reports extended-source protection for a bright disc",
          "[luma-denoise][mask]") {
  constexpr int W = 256, H = 192;
  Matrix2Df r(H, W), g(H, W), b(H, W);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const float dx = static_cast<float>(x - W / 2);
      const float dy = static_cast<float>(y - H / 2);
      const bool source = dx * dx + dy * dy < 42.0f * 42.0f;
      r(y, x) = g(y, x) = b(y, x) = source ? 4.0f : 1.0f;
    }
  tile_compile::config::LumaDenoiseConfig cfg;
  cfg.enabled = true;
  cfg.star_protection.enabled = false;
  cfg.structure_protection.enabled = false;
  cfg.extended_source_protection.enabled = true;
  cfg.extended_source_protection.luma_sigma = 2.5f;
  cfg.extended_source_protection.dilate_px = 5;

  const auto stats = tile_compile::reconstruction::luma_denoise_rgb_inplace(
      r, g, b, cfg);
  REQUIRE(stats.applied);
  REQUIRE(stats.extended_source_raw_fraction > 0.05);
  REQUIRE(stats.extended_source_raw_fraction < 0.5);
  REQUIRE(stats.extended_source_protected_fraction >=
          stats.extended_source_raw_fraction);
  REQUIRE(stats.combined_protected_fraction > 0.05);
}

TEST_CASE("luma extended-source mask does not classify a quarter of Gaussian sky",
          "[luma-denoise][mask]") {
  constexpr int W = 320, H = 240;
  Matrix2Df r(H, W), g(H, W), b(H, W);
  std::mt19937 rng(7);
  std::normal_distribution<float> noise(0.0f, 0.02f);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      r(y, x) = g(y, x) = b(y, x) = 1.0f + noise(rng);
  tile_compile::config::LumaDenoiseConfig cfg;
  cfg.enabled = true;
  cfg.star_protection.enabled = false;
  cfg.structure_protection.enabled = false;
  cfg.extended_source_protection.enabled = true;
  cfg.extended_source_protection.dilate_px = 0;
  const auto stats = tile_compile::reconstruction::luma_denoise_rgb_inplace(
      r, g, b, cfg);
  REQUIRE(stats.extended_source_raw_fraction < 0.05);
}

TEST_CASE("luma bilateral is off by default and disabled leaves output unchanged",
          "[luma-denoise][bilateral]") {
  tile_compile::config::Config cfg;
  REQUIRE(cfg.luma_denoise.bilateral.enabled == false);

  constexpr int W = 80, H = 48;
  std::mt19937 rng(7);
  std::normal_distribution<float> noise(0.0f, 0.02f);
  Matrix2Df r(H, W), g(H, W), b(H, W);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const float n = noise(rng);
      r(y, x) = 0.2f + n;
      g(y, x) = 0.2f + n;
      b(y, x) = 0.2f + n;
    }
  const Matrix2Df r_before = r, g_before = g, b_before = b;
  tile_compile::config::LumaDenoiseConfig lcfg;
  lcfg.enabled = true;
  lcfg.wavelet.enabled = false;
  lcfg.bilateral.enabled = false;
  tile_compile::reconstruction::luma_denoise_rgb_inplace(r, g, b, lcfg);
  REQUIRE((r - r_before).cwiseAbs().maxCoeff() < 1.0e-6f);
  REQUIRE((g - g_before).cwiseAbs().maxCoeff() < 1.0e-6f);
  REQUIRE((b - b_before).cwiseAbs().maxCoeff() < 1.0e-6f);
}

TEST_CASE("luma bilateral reduces background noise beyond wavelet alone",
          "[luma-denoise][bilateral]") {
  // Same construction the HMS soft_floor/luma_denoise M42-610f investigation
  // used to find the wavelet stage's noise floor: the coarsest
  // Gaussian-pyramid approximation level is always added back unmodified
  // (see denoise_luma_plane_inplace), so a noisy flat background plateaus
  // under wavelet-only denoise however aggressive levels/threshold_scale
  // get. Bilateral targets that residual directly.
  constexpr int W = 160, H = 120;
  std::mt19937 rng(11);
  std::normal_distribution<float> noise(0.0f, 0.03f);
  auto make_noisy_flat = [&]() {
    std::array<Matrix2Df, 3> rgb{Matrix2Df(H, W), Matrix2Df(H, W),
                                 Matrix2Df(H, W)};
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        const float n = noise(rng);
        rgb[0](y, x) = rgb[1](y, x) = rgb[2](y, x) = 0.15f + n;
      }
    return rgb;
  };
  auto wavelet_only = make_noisy_flat();
  auto with_bilateral = make_noisy_flat();

  // Repo-default wavelet strength (levels=3): representative of real use,
  // unlike a maxed-out levels=8 setting whose coarsest approximation level
  // (sigma ~= 2^7*0.75 px) would crush almost all variance in a small test
  // image by itself and mask any additional effect from bilateral.
  tile_compile::config::LumaDenoiseConfig cfg_wavelet;
  cfg_wavelet.enabled = true;
  cfg_wavelet.blend_amount = 1.0f;
  cfg_wavelet.star_protection.enabled = false;
  cfg_wavelet.structure_protection.enabled = false;
  cfg_wavelet.wavelet.enabled = true;
  cfg_wavelet.wavelet.levels = 3;
  cfg_wavelet.wavelet.threshold_scale = 1.5f;
  cfg_wavelet.bilateral.enabled = false;

  // sigma_spatial matched to the value validated against the real M42-610f
  // series (see the HMS soft_floor/luma_denoise report): a small default
  // sigma_spatial (~1.5) barely widens the kernel beyond the wavelet's own
  // finest level and shows no measurable extra effect in this synthetic
  // scenario, even though a larger radius does measurably help on real data.
  auto cfg_both = cfg_wavelet;
  cfg_both.bilateral.enabled = true;
  cfg_both.bilateral.sigma_spatial = 8.0f;
  cfg_both.bilateral.sigma_range = 3.0f;

  tile_compile::reconstruction::luma_denoise_rgb_inplace(
      wavelet_only[0], wavelet_only[1], wavelet_only[2], cfg_wavelet);
  tile_compile::reconstruction::luma_denoise_rgb_inplace(
      with_bilateral[0], with_bilateral[1], with_bilateral[2], cfg_both);

  const double std_wavelet = stddev_of(wavelet_only[0]);
  const double std_both = stddev_of(with_bilateral[0]);
  REQUIRE(std_both < std_wavelet);
}

TEST_CASE("luma bilateral range is invariant to normalized versus ADU scale",
          "[luma-denoise][bilateral][adaptation]") {
  constexpr int W = 96, H = 72;
  auto make = [&](float scale) {
    std::array<Matrix2Df, 3> rgb{Matrix2Df(H, W), Matrix2Df(H, W),
                                 Matrix2Df(H, W)};
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        const float n = ((x * 17 + y * 11) % 13 - 6) * 0.003f;
        rgb[0](y, x) = rgb[1](y, x) = rgb[2](y, x) = scale * (0.2f + n);
      }
    return rgb;
  };
  auto normalized = make(1.0f);
  auto adu = make(500.0f);
  tile_compile::config::LumaDenoiseConfig cfg;
  cfg.enabled = true;
  cfg.blend_amount = 1.0f;
  cfg.star_protection.enabled = false;
  cfg.structure_protection.enabled = false;
  cfg.wavelet.enabled = false;
  cfg.bilateral.enabled = true;
  cfg.bilateral.sigma_range = 2.0f;
  tile_compile::reconstruction::luma_denoise_rgb_inplace(
      normalized[0], normalized[1], normalized[2], cfg);
  tile_compile::reconstruction::luma_denoise_rgb_inplace(adu[0], adu[1],
                                                          adu[2], cfg);
  REQUIRE((normalized[0] - adu[0] / 500.0f).cwiseAbs().maxCoeff() < 2.0e-5f);
}

TEST_CASE("luma sharpness_guard config ranges are validated",
          "[config][luma-denoise][sharpness-guard]") {
  tile_compile::config::Config cfg;
  cfg.luma_denoise.sharpness_guard.max_fwhm_increase_pct = -0.5f;
  REQUIRE_THROWS_AS(cfg.validate(), tile_compile::ValidationError);
  cfg.luma_denoise.sharpness_guard.max_fwhm_increase_pct = 1.0f;

  cfg.luma_denoise.sharpness_guard.min_matched_stars = 0;
  REQUIRE_THROWS_AS(cfg.validate(), tile_compile::ValidationError);
  cfg.luma_denoise.sharpness_guard.min_matched_stars = 15;

  cfg.luma_denoise.sharpness_guard.max_backoff_steps = -1;
  REQUIRE_THROWS_AS(cfg.validate(), tile_compile::ValidationError);
  cfg.luma_denoise.sharpness_guard.max_backoff_steps = 2;

  REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("luma sharpness_guard rejects fail-closed when unprotected stars blur and passes when protected",
          "[luma-denoise][sharpness-guard]") {
  constexpr int W = 240, H = 200;
  auto make_star_field = [&]() {
    std::array<Matrix2Df, 3> rgb{Matrix2Df::Constant(H, W, 20.0f),
                                 Matrix2Df::Constant(H, W, 20.0f),
                                 Matrix2Df::Constant(H, W, 20.0f)};
    std::mt19937 rng(321);
    std::normal_distribution<float> noise(0.0f, 0.35f);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        const float n = noise(rng);
        rgb[0](y, x) += n;
        rgb[1](y, x) += n;
        rgb[2](y, x) += n;
      }
    // 5x4 = 20 compact Gaussian stars (sigma = 1.1 px, FWHM ~ 2.6 px, ~28 sigma)
    for (int gy = 0; gy < 4; ++gy) {
      for (int gx = 0; gx < 5; ++gx) {
        const float cx = 30.0f + gx * 45.0f;
        const float cy = 30.0f + gy * 45.0f;
        for (int dy = -6; dy <= 6; ++dy) {
          for (int dx = -6; dx <= 6; ++dx) {
            const float val = 10.0f * std::exp(-(dx * dx + dy * dy) / (2.0f * 1.1f * 1.1f));
            const int yy = static_cast<int>(cy) + dy;
            const int xx = static_cast<int>(cx) + dx;
            rgb[0](yy, xx) += val;
            rgb[1](yy, xx) += val;
            rgb[2](yy, xx) += val;
          }
        }
      }
    }
    return rgb;
  };

  // Case 1: Star & structure protection disabled + wide bilateral smoothing -> stars blur -> fail-closed reject
  {
    auto img = make_star_field();
    const Matrix2Df r_orig = img[0];
    tile_compile::config::LumaDenoiseConfig cfg;
    cfg.enabled = true;
    cfg.star_protection.enabled = false;
    cfg.structure_protection.enabled = false;
    cfg.blend_amount = 1.0f;
    cfg.wavelet.enabled = true;
    cfg.wavelet.levels = 4;
    cfg.wavelet.threshold_scale = 8.0f;
    cfg.bilateral.enabled = true;
    cfg.bilateral.sigma_spatial = 6.0f;
    cfg.bilateral.sigma_range = 20.0f;
    cfg.sharpness_guard.enabled = true;
    cfg.sharpness_guard.max_fwhm_increase_pct = 1.0f;
    cfg.sharpness_guard.min_matched_stars = 10;
    cfg.sharpness_guard.max_backoff_steps = 1;

    const auto stats = tile_compile::reconstruction::luma_denoise_rgb_inplace(
        img[0], img[1], img[2], cfg);
    REQUIRE(stats.matched_stars >= 10);
    REQUIRE(stats.guard_rejected == true);
    REQUIRE((stats.guard_reason == "fwhm_regression" ||
             stats.guard_reason == "star_peak_erosion"));
    REQUIRE(stats.applied == false);
    REQUIRE((img[0] - r_orig).cwiseAbs().maxCoeff() == 0.0f);
  }

  // Case 2: Star protection enabled -> stars stay sharp -> guard passes
  {
    auto img = make_star_field();
    tile_compile::config::LumaDenoiseConfig cfg;
    cfg.enabled = true;
    cfg.star_protection.enabled = true;
    cfg.star_protection.threshold_sigma = 5.0f;
    cfg.star_protection.dilate_px = 6;
    cfg.structure_protection.enabled = true;
    cfg.luma_guard_strength = 1.0f;
    cfg.blend_amount = 0.85f;
    cfg.wavelet.levels = 3;
    cfg.wavelet.threshold_scale = 1.5f;
    cfg.sharpness_guard.enabled = true;
    cfg.sharpness_guard.max_fwhm_increase_pct = 1.0f;
    cfg.sharpness_guard.min_matched_stars = 10;
    cfg.sharpness_guard.max_backoff_steps = 2;

    const auto stats = tile_compile::reconstruction::luma_denoise_rgb_inplace(
        img[0], img[1], img[2], cfg);
    REQUIRE(stats.matched_stars >= 10);
    REQUIRE(stats.guard_rejected == false);
    REQUIRE(stats.guard_reason == "passed");
    REQUIRE(stats.applied == true);
    REQUIRE(stats.fwhm_delta_pct <= 1.0);
  }
}

TEST_CASE("luma sharpness_guard rejects when too few stars can be measured and records why",
          "[luma-denoise][sharpness-guard]") {
  constexpr int W = 200, H = 160;
  auto field = [&](int stars) {
    std::array<Matrix2Df, 3> rgb{Matrix2Df::Constant(H, W, 20.0f),
                                 Matrix2Df::Constant(H, W, 20.0f),
                                 Matrix2Df::Constant(H, W, 20.0f)};
    std::mt19937 rng(11);
    std::normal_distribution<float> noise(0.0f, 0.35f);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        const float v = noise(rng);
        for (auto& c : rgb) c(y, x) += v;
      }
    for (int s = 0; s < stars; ++s) {
      const int cx = 40 + 50 * s, cy = 80;
      for (int dy = -6; dy <= 6; ++dy)
        for (int dx = -6; dx <= 6; ++dx) {
          const float v = 10.0f * std::exp(-(dx * dx + dy * dy) / (2.0f * 1.1f * 1.1f));
          for (auto& c : rgb) c(cy + dy, cx + dx) += v;
        }
    }
    return rgb;
  };
  tile_compile::config::LumaDenoiseConfig cfg;
  cfg.enabled = true;
  cfg.sharpness_guard.enabled = true;
  cfg.sharpness_guard.min_matched_stars = 15;

  for (int stars : {0, 3}) {
    auto img = field(stars);
    const Matrix2Df r_orig = img[0];
    const auto st = tile_compile::reconstruction::luma_denoise_rgb_inplace(
        img[0], img[1], img[2], cfg);
    REQUIRE_FALSE(st.applied);
    REQUIRE(st.guard_rejected);
    REQUIRE(st.guard_reason == "insufficient_stars");
    REQUIRE(st.matched_stars < 15);
    REQUIRE((img[0] - r_orig).cwiseAbs().maxCoeff() == 0.0f);
  }

  // Guard off: the pass applies as before and says the guard did not run.
  cfg.sharpness_guard.enabled = false;
  auto img = field(3);
  const Matrix2Df r_orig = img[0];
  const auto st = tile_compile::reconstruction::luma_denoise_rgb_inplace(
      img[0], img[1], img[2], cfg);
  REQUIRE(st.applied);
  REQUIRE_FALSE(st.guard_rejected);
  REQUIRE(st.guard_reason == "disabled");
  REQUIRE((img[0] - r_orig).cwiseAbs().maxCoeff() > 0.0f);
}
