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
