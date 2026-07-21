#if __has_include(<catch2/catch_test_macros.hpp>)
#include "tile_compile/image/hypermetric_stretch.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

TEST_CASE("hypermetric_stretch_curve_is_monotonic_and_bounded") {
  float prev = tile_compile::image::hypermetric_hyperbolic_stretch_value(
      0.0f, 100.0f, 6.0f);
  REQUIRE(prev == Catch::Approx(0.0f).margin(1e-5f));
  for (int i = 1; i <= 100; ++i) {
    const float x = static_cast<float>(i) / 100.0f;
    const float y = tile_compile::image::hypermetric_hyperbolic_stretch_value(
        x, 100.0f, 6.0f);
    REQUIRE(y >= prev);
    REQUIRE(y >= -1e-5f);
    REQUIRE(y <= 1.0f + 1e-5f);
    prev = y;
  }
}

TEST_CASE("hypermetric_log_d_solver_hits_target_median") {
  std::vector<float> sample(1000, 0.04f);
  const float log_d =
      tile_compile::image::hypermetric_solve_log_d(sample, 0.20f, 6.0f);
  const float out = tile_compile::image::hypermetric_hyperbolic_stretch_value(
      0.04f, std::pow(10.0f, log_d), 6.0f);
  REQUIRE(out == Catch::Approx(0.20f).margin(2e-3f));
}

TEST_CASE("hypermetric_ready_to_use_rgb_run_produces_unit_range_output") {
  tile_compile::Matrix2Df R(32, 32);
  tile_compile::Matrix2Df G(32, 32);
  tile_compile::Matrix2Df B(32, 32);
  for (int y = 0; y < 32; ++y) {
    for (int x = 0; x < 32; ++x) {
      const float base = 0.015f + 0.0002f * static_cast<float>(x + y);
      R(y, x) = base * 1.12f;
      G(y, x) = base;
      B(y, x) = base * 0.82f;
    }
  }
  R(10, 10) = 0.75f;
  G(10, 10) = 0.70f;
  B(10, 10) = 0.62f;

  tile_compile::image::HyperMetricStretchConfig cfg;
  cfg.enabled = true;
  cfg.sensor_profile = "rec709";
  cfg.adaptive_anchor = true;
  cfg.log_d_mode = "auto";
  cfg.color_strategy = "fixed";

  const auto diag =
      tile_compile::image::run_hypermetric_stretch_rgb(R, G, B, cfg);
  REQUIRE(diag.success);
  REQUIRE(diag.log_d >= 0.0f);
  REQUIRE(diag.log_d <= 7.0f);

  for (int y = 0; y < R.rows(); ++y) {
    for (int x = 0; x < R.cols(); ++x) {
      REQUIRE(R(y, x) >= 0.0f);
      REQUIRE(G(y, x) >= 0.0f);
      REQUIRE(B(y, x) >= 0.0f);
      REQUIRE(R(y, x) <= 1.0f);
      REQUIRE(G(y, x) <= 1.0f);
      REQUIRE(B(y, x) <= 1.0f);
    }
  }
}

TEST_CASE("hypermetric_resolves_dwarf_ii_imx415_profile") {
  tile_compile::Matrix2Df R(16, 16);
  tile_compile::Matrix2Df G(16, 16);
  tile_compile::Matrix2Df B(16, 16);
  for (int y = 0; y < 16; ++y) {
    for (int x = 0; x < 16; ++x) {
      const float base = 0.02f + 0.0002f * static_cast<float>(x + y);
      R(y, x) = base * 1.05f;
      G(y, x) = base;
      B(y, x) = base * 0.9f;
    }
  }

  tile_compile::image::HyperMetricStretchConfig cfg;
  cfg.enabled = true;
  cfg.sensor_profile = "Sony IMX415 (DWARF II)";

  const auto diag =
      tile_compile::image::run_hypermetric_stretch_rgb(R, G, B, cfg);

  REQUIRE(diag.success);
  REQUIRE(diag.profile == "Sony IMX415 (DWARF II)");
  REQUIRE(diag.profile_source == "configured");
  REQUIRE(diag.weights_r == Catch::Approx(0.2703f).margin(1e-6f));
  REQUIRE(diag.weights_g == Catch::Approx(0.5405f).margin(1e-6f));
  REQUIRE(diag.weights_b == Catch::Approx(0.1892f).margin(1e-6f));
}

TEST_CASE("hypermetric_normalizes_16bit_like_float_input_before_anchor") {
  tile_compile::Matrix2Df R(24, 24);
  tile_compile::Matrix2Df G(24, 24);
  tile_compile::Matrix2Df B(24, 24);
  for (int y = 0; y < 24; ++y) {
    for (int x = 0; x < 24; ++x) {
      const float base = (0.02f + 0.0003f * static_cast<float>(x + y)) * 65535.0f;
      R(y, x) = base * 1.1f;
      G(y, x) = base;
      B(y, x) = base * 0.8f;
    }
  }

  tile_compile::image::HyperMetricStretchConfig cfg;
  cfg.enabled = true;
  cfg.sensor_profile = "rec709";
  cfg.adaptive_anchor = true;

  const auto diag =
      tile_compile::image::run_hypermetric_stretch_rgb(R, G, B, cfg);

  REQUIRE(diag.success);
  REQUIRE(diag.anchor < 0.05f);
  REQUIRE(diag.white_clip_percent < 5.0f);
  REQUIRE(R(12, 12) < 0.95f);
  REQUIRE(G(12, 12) < 0.95f);
  REQUIRE(B(12, 12) < 0.95f);
}

TEST_CASE("hypermetric_auto_color_strategy_keeps_python_slider_default") {
  tile_compile::Matrix2Df R(48, 48);
  tile_compile::Matrix2Df G(48, 48);
  tile_compile::Matrix2Df B(48, 48);
  for (int y = 0; y < 48; ++y) {
    for (int x = 0; x < 48; ++x) {
      const float base = 0.012f + 0.0001f * static_cast<float>((x + y) % 16);
      R(y, x) = base;
      G(y, x) = base;
      B(y, x) = base;
    }
  }
  for (int i = 0; i < 12; ++i) {
    const int p = 4 + i * 3;
    R(p, p) = 1.0f;
    G(p, p) = 0.95f;
    B(p, p) = 0.9f;
  }

  tile_compile::image::HyperMetricStretchConfig cfg;
  cfg.enabled = true;
  cfg.sensor_profile = "rec709";
  cfg.color_strategy = "auto";
  cfg.fixed_color_strategy = 0.0f;

  const auto diag =
      tile_compile::image::run_hypermetric_stretch_rgb(R, G, B, cfg);

  REQUIRE(diag.success);
  REQUIRE(diag.color_strategy == Catch::Approx(0.0f).margin(1e-6f));
  REQUIRE(diag.color_grip == Catch::Approx(1.0f).margin(1e-6f));
  REQUIRE(diag.shadow_convergence == Catch::Approx(0.0f).margin(1e-6f));
}

TEST_CASE("hypermetric_scientific_mode_applies_linear_expansion") {
  tile_compile::Matrix2Df R0(32, 32);
  tile_compile::Matrix2Df G0(32, 32);
  tile_compile::Matrix2Df B0(32, 32);
  for (int y = 0; y < 32; ++y) {
    for (int x = 0; x < 32; ++x) {
      const float base = 0.02f + 0.0004f * static_cast<float>(x + y);
      R0(y, x) = base * 1.05f;
      G0(y, x) = base;
      B0(y, x) = base * 0.92f;
    }
  }
  R0(16, 16) = 0.8f;
  G0(16, 16) = 0.76f;
  B0(16, 16) = 0.7f;
  R0(16, 17) = 0.3f;
  G0(16, 17) = 0.28f;
  B0(16, 17) = 0.25f;

  auto R1 = R0;
  auto G1 = G0;
  auto B1 = B0;

  tile_compile::image::HyperMetricStretchConfig cfg;
  cfg.enabled = true;
  cfg.mode = "scientific";
  cfg.sensor_profile = "rec709";
  cfg.adaptive_anchor = false;
  cfg.log_d_mode = "fixed";
  cfg.fixed_log_d = 2.0f;
  cfg.color_strategy = "fixed";

  const auto diag0 =
      tile_compile::image::run_hypermetric_stretch_rgb(R0, G0, B0, cfg);
  cfg.linear_expansion = 1.0f;
  const auto diag1 =
      tile_compile::image::run_hypermetric_stretch_rgb(R1, G1, B1, cfg);

  REQUIRE(diag0.success);
  REQUIRE(diag1.success);
  REQUIRE(std::abs(R1(8, 8) - R0(8, 8)) > 1e-4f);
  REQUIRE(std::abs(G1(8, 8) - G0(8, 8)) > 1e-4f);
  REQUIRE(std::abs(B1(8, 8) - B0(8, 8)) > 1e-4f);
}

TEST_CASE("hypermetric_uses_common_overlap_statistics_without_cropping_output") {
  constexpr int kSize = 16;
  tile_compile::Matrix2Df R(kSize, kSize);
  tile_compile::Matrix2Df G(kSize, kSize);
  tile_compile::Matrix2Df B(kSize, kSize);
  for (int y = 0; y < kSize; ++y) {
    for (int x = 0; x < kSize; ++x) {
      const float base = 0.02f + 0.001f * static_cast<float>(x + y);
      R(y, x) = base * 1.1f;
      G(y, x) = base;
      B(y, x) = base * 0.9f;
    }
  }
  auto R_common_only = R;
  auto G_common_only = G;
  auto B_common_only = B;

  std::vector<uint8_t> common_mask(kSize * kSize, 0u);
  std::vector<uint8_t> output_mask(kSize * kSize, 1u);
  for (int y = 4; y < 12; ++y) {
    for (int x = 4; x < 12; ++x) {
      common_mask[static_cast<size_t>(y) * kSize + x] = 1u;
    }
  }

  tile_compile::image::HyperMetricStretchConfig cfg;
  cfg.enabled = true;
  cfg.mode = "ready_to_use";
  cfg.adaptive_anchor = true;
  cfg.log_d_mode = "auto";

  const auto full_diag = tile_compile::image::run_hypermetric_stretch_rgb(
      R, G, B, cfg, &common_mask, kSize, kSize, &output_mask);
  const auto common_diag = tile_compile::image::run_hypermetric_stretch_rgb(
      R_common_only, G_common_only, B_common_only, cfg, &common_mask, kSize,
      kSize, &common_mask);

  REQUIRE(full_diag.success);
  REQUIRE(common_diag.success);
  REQUIRE(R(0, 0) > 0.0f);
  REQUIRE(G(0, 0) > 0.0f);
  REQUIRE(B(0, 0) > 0.0f);
  REQUIRE(R(8, 8) == Catch::Approx(R_common_only(8, 8)).margin(1e-6f));
  REQUIRE(G(8, 8) == Catch::Approx(G_common_only(8, 8)).margin(1e-6f));
  REQUIRE(B(8, 8) == Catch::Approx(B_common_only(8, 8)).margin(1e-6f));
}
#endif
