#if __has_include(<catch2/catch_test_macros.hpp>)
#include "tile_compile/image/hypermetric_stretch.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <vector>

namespace {

float median_matrix(const tile_compile::Matrix2Df &m) {
  std::vector<float> values;
  values.reserve(static_cast<size_t>(m.rows()) * static_cast<size_t>(m.cols()));
  for (int y = 0; y < m.rows(); ++y) {
    for (int x = 0; x < m.cols(); ++x) {
      values.push_back(m(y, x));
    }
  }
  std::sort(values.begin(), values.end());
  const size_t mid = values.size() / 2;
  if (values.size() % 2 == 0) {
    return 0.5f * (values[mid - 1] + values[mid]);
  }
  return values[mid];
}

} // namespace

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

// Regression for the M31 CFA magenta cast: R/B carry coverage holes clamped
// to 0 while G's doubled Bayer sampling keeps its minimum near the
// background. A per-channel min-clamped floor then sat floor_g far above the
// R/B floors, shrinking G's expansion gap and producing R/G ~1.7 in the
// output. Floors must leave an identical (median - floor) gap per channel.
TEST_CASE("hypermetric_ready_to_use_floors_do_not_follow_channel_min") {
  constexpr int kSize = 96;
  tile_compile::Matrix2Df R(kSize, kSize);
  tile_compile::Matrix2Df G(kSize, kSize);
  tile_compile::Matrix2Df B(kSize, kSize);
  for (int y = 0; y < kSize; ++y) {
    for (int x = 0; x < kSize; ++x) {
      const float base = 0.15f + 0.0004f * static_cast<float>((x + y) % 17);
      R(y, x) = base;
      G(y, x) = base;
      B(y, x) = base;
    }
  }
  // Coverage holes: R/B get zeroed pixels, G keeps a high minimum.
  for (int y = 0; y < kSize; y += 5) {
    for (int x = 0; x < kSize; x += 5) {
      R(y, x) = 0.0f;
      B(y, x) = 0.0f;
    }
  }
  for (int i = 0; i < 30; ++i) {
    const int y = 6 + (i * 11) % 84;
    const int x = 8 + (i * 13) % 80;
    R(y, x) = G(y, x) = B(y, x) = 0.6f + 0.005f * static_cast<float>(i % 6);
  }

  tile_compile::image::HyperMetricStretchConfig cfg;
  cfg.enabled = true;
  cfg.mode = "ready_to_use";
  cfg.sensor_profile = "rec709";
  cfg.adaptive_anchor = false;
  cfg.log_d_mode = "fixed";
  cfg.fixed_log_d = 2.0f;
  cfg.color_strategy = "fixed";
  cfg.target_bg = 0.15f;

  const auto diag =
      tile_compile::image::run_hypermetric_stretch_rgb(R, G, B, cfg);
  REQUIRE(diag.success);
  const float r_med = median_matrix(R);
  const float g_med = median_matrix(G);
  const float b_med = median_matrix(B);
  REQUIRE(g_med > 0.01f);
  REQUIRE(r_med / g_med == Catch::Approx(1.0f).margin(0.15f));
  REQUIRE(b_med / g_med == Catch::Approx(1.0f).margin(0.15f));
}

// The ready_to_use output scaling places each channel floor a shared margin
// below its sky level; that margin is bounded by the widest channel's
// sky-to-minimum span, so it follows the extremes of the per-channel noise.
// Removing only the chroma part of the noise (same common/luma part) narrows
// the margin. Where the sky then lands before the MTF, and so which way the
// midtone moves, depends on the data (IC434 2026-11-22: margin -9%, midtone
// 0.151 -> 0.137, visible luma noise +12% after a chroma-only denoise). The
// diagnostics expose margin, scale and midtone with what bounded them.
TEST_CASE("hypermetric_output_scaling_margin_follows_chroma_noise_extremes") {
  constexpr int kSize = 128;
  auto run = [&](float chroma_sigma) {
    std::mt19937 rng(7);
    std::normal_distribution<float> common(0.0f, 0.004f);
    std::normal_distribution<float> chroma(0.0f, 1.0f);
    tile_compile::Matrix2Df R(kSize, kSize), G(kSize, kSize), B(kSize, kSize);
    for (int y = 0; y < kSize; ++y) {
      for (int x = 0; x < kSize; ++x) {
        const float sky = 0.05f + common(rng);
        const float c = chroma_sigma * chroma(rng);
        R(y, x) = sky + c;
        G(y, x) = sky;
        B(y, x) = sky - c;
      }
    }
    for (int i = 0; i < 20; ++i) {
      const int y = 8 + (i * 23) % 112;
      const int x = 8 + (i * 37) % 112;
      R(y, x) = G(y, x) = B(y, x) = 0.7f;
    }
    tile_compile::image::HyperMetricStretchConfig cfg;
    cfg.enabled = true;
    cfg.mode = "ready_to_use";
    cfg.sensor_profile = "rec709";
    cfg.adaptive_anchor = false;
    cfg.log_d_mode = "fixed";
    cfg.fixed_log_d = 2.0f;
    cfg.color_strategy = "fixed";
    cfg.target_bg = 0.15f;
    return tile_compile::image::run_hypermetric_stretch_rgb(R, G, B, cfg);
  };
  const auto noisy = run(0.012f);
  const auto clean = run(0.0f);
  REQUIRE(noisy.success);
  REQUIRE(clean.success);
  for (const auto *d : {&noisy, &clean}) {
    REQUIRE(d->output_floor_margin > 0.0f);
    REQUIRE((d->output_margin_source == "sigma" ||
             d->output_margin_source == "minimum"));
    REQUIRE(d->output_scale > 0.0f);
    REQUIRE((d->output_scale_source == "p99_span" ||
             d->output_scale_source == "physical_ceiling"));
    REQUIRE(d->output_mtf_midtone > 0.0f);
    REQUIRE(d->output_mtf_midtone < 1.0f);
  }
  REQUIRE(noisy.output_margin_source == "minimum");
  REQUIRE(clean.output_floor_margin < 0.8f * noisy.output_floor_margin);
}

TEST_CASE("hypermetric_ready_to_use_scaling_preserves_weak_blue_channel") {
  constexpr int kSize = 96;
  tile_compile::Matrix2Df R(kSize, kSize);
  tile_compile::Matrix2Df G(kSize, kSize);
  tile_compile::Matrix2Df B(kSize, kSize);
  for (int y = 0; y < kSize; ++y) {
    for (int x = 0; x < kSize; ++x) {
      const float gradient = 0.00002f * static_cast<float>((x + 2 * y) % 31);
      const float g = 0.0054f + gradient;
      G(y, x) = g;
      R(y, x) = g * 0.99f;
      B(y, x) = g * 0.91f;
    }
  }
  for (int i = 0; i < 24; ++i) {
    const int y = 8 + (i * 7) % 80;
    const int x = 10 + (i * 11) % 76;
    G(y, x) = 0.08f + 0.002f * static_cast<float>(i % 5);
    R(y, x) = G(y, x) * 0.99f;
    B(y, x) = G(y, x) * 0.91f;
  }

  tile_compile::image::HyperMetricStretchConfig cfg;
  cfg.enabled = true;
  cfg.mode = "ready_to_use";
  cfg.sensor_profile = "rec709";
  cfg.adaptive_anchor = false;
  cfg.log_d_mode = "fixed";
  cfg.fixed_log_d = 2.0f;
  cfg.color_strategy = "fixed";
  cfg.target_bg = 0.15f;

  const auto diag =
      tile_compile::image::run_hypermetric_stretch_rgb(R, G, B, cfg);

  REQUIRE(diag.success);
  const float r_med = median_matrix(R);
  const float g_med = median_matrix(G);
  const float b_med = median_matrix(B);
  REQUIRE(g_med > 0.01f);
  REQUIRE(r_med / g_med == Catch::Approx(0.99f).margin(0.12f));
  REQUIRE(b_med / g_med == Catch::Approx(0.91f).margin(0.18f));
  REQUIRE(b_med / g_med > 0.55f);
  REQUIRE(diag.black_clip_percent < 1.0f);
}

TEST_CASE("hypermetric_ready_to_use_preserves_extended_highlight_headroom") {
  constexpr int kSize = 160;
  tile_compile::Matrix2Df R(kSize, kSize);
  tile_compile::Matrix2Df G(kSize, kSize);
  tile_compile::Matrix2Df B(kSize, kSize);
  for (int y = 0; y < kSize; ++y) {
    for (int x = 0; x < kSize; ++x) {
      const float dx = static_cast<float>(x - kSize / 2);
      const float dy = static_cast<float>(y - kSize / 2);
      const float core = 0.72f * std::exp(-(dx * dx + dy * dy) / 800.0f);
      const float base = 0.012f + core;
      R(y, x) = base * 1.03f;
      G(y, x) = base;
      B(y, x) = base * 0.92f;
    }
  }

  tile_compile::image::HyperMetricStretchConfig cfg;
  cfg.enabled = true;
  cfg.mode = "ready_to_use";
  cfg.adaptive_anchor = false;
  cfg.log_d_mode = "fixed";
  cfg.fixed_log_d = 3.5f;
  cfg.target_bg = 0.12f;

  const auto diag =
      tile_compile::image::run_hypermetric_stretch_rgb(R, G, B, cfg);

  REQUIRE(diag.success);
  REQUIRE(diag.white_clip_percent == Catch::Approx(0.0f).margin(1e-4f));
  REQUIRE(R(kSize / 2, kSize / 2) < 1.0f);
  REQUIRE(G(kSize / 2, kSize / 2) < 1.0f);
  REQUIRE(B(kSize / 2, kSize / 2) < 1.0f);
}

TEST_CASE("hypermetric_highlight_ceiling_percentile_trades_clip_for_contrast") {
  // Mirrors the real-world failure mode this parameter addresses: a large,
  // moderately bright extended body (most of the frame, up to ~p99) plus a
  // small, much brighter compact core (a handful of pixels, well above
  // p99) -- e.g. a nebula body vs. its bright Trapezium-like core. At the
  // default ceiling (100 = true max), that one small spike caps the
  // contrast scale for the *entire* frame. Lowering the ceiling percentile
  // should (a) leave the background/median level unchanged -- it stays
  // pinned to target_bg by the final MTF match, independent of the ceiling
  // -- while (b) increasing contrast/brightness for the extended body, in
  // exchange for a small, bounded amount of intentional clipping confined
  // to the spike.
  auto run_with_ceiling = [](float ceiling_percentile) {
    constexpr int kSize = 160;
    tile_compile::Matrix2Df R(kSize, kSize);
    tile_compile::Matrix2Df G(kSize, kSize);
    tile_compile::Matrix2Df B(kSize, kSize);
    for (int y = 0; y < kSize; ++y) {
      for (int x = 0; x < kSize; ++x) {
        const float dx = static_cast<float>(x - kSize / 2);
        const float dy = static_cast<float>(y - kSize / 2);
        const float r2 = dx * dx + dy * dy;
        // Broad, extended body: most of the frame sits on this shoulder.
        const float body = 0.10f * std::exp(-r2 / 4000.0f);
        // Small, compact, much brighter core: only a few dozen pixels.
        const float spike = 0.85f * std::exp(-r2 / 8.0f);
        const float base = 0.006f + body + spike;
        R(y, x) = base * 1.03f;
        G(y, x) = base;
        B(y, x) = base * 0.92f;
      }
    }

    tile_compile::image::HyperMetricStretchConfig cfg;
    cfg.enabled = true;
    cfg.mode = "ready_to_use";
    cfg.adaptive_anchor = false;
    cfg.log_d_mode = "fixed";
    cfg.fixed_log_d = 3.5f;
    cfg.target_bg = 0.12f;
    cfg.highlight_ceiling_percentile = ceiling_percentile;

    const auto diag =
        tile_compile::image::run_hypermetric_stretch_rgb(R, G, B, cfg);
    REQUIRE(diag.success);
    // A point out on the extended body, far from the compact core --
    // representative of the "relatively dark/mid-tone" nebula-body pixels
    // the ceiling change is meant to brighten.
    const int body_pt = kSize / 2 + 30;
    return std::tuple<float, float, float>{
        median_matrix(G), G(body_pt, kSize / 2), diag.white_clip_percent};
  };

  const auto [g_med_default, g_body_default, white_clip_default] =
      run_with_ceiling(100.0f);
  const auto [g_med_relaxed, g_body_relaxed, white_clip_relaxed] =
      run_with_ceiling(99.0f);

  // Background/median stays pinned to target_bg regardless of the ceiling.
  REQUIRE(g_med_relaxed == Catch::Approx(g_med_default).margin(0.01f));
  // The extended body gets materially more contrast once it no longer has
  // to share headroom with the small, much brighter core.
  REQUIRE(g_body_relaxed > g_body_default * 1.05f);
  // The traded-off cost is a small, bounded amount of highlight clipping --
  // not the ~0 % the fully protective default guarantees, but nowhere near
  // the whole frame (it stays confined to the compact core).
  REQUIRE(white_clip_default == Catch::Approx(0.0f).margin(1e-4f));
  REQUIRE(white_clip_relaxed > white_clip_default);
  REQUIRE(white_clip_relaxed < 5.0f);
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

namespace {

// A sky with noise shared by all channels plus, optionally, chroma noise
// (R up, B down), a few stars and a faint nebula: the same seed and levels for
// every call, so two calls differ only in the chroma noise -- what a
// chroma-only denoise removes.
std::array<tile_compile::Matrix2Df, 3> make_sky_rgb(int n, float chroma_sigma) {
  std::mt19937 rng(11);
  std::normal_distribution<float> common(0.0f, 0.004f);
  std::normal_distribution<float> chroma(0.0f, 1.0f);
  std::array<tile_compile::Matrix2Df, 3> rgb{tile_compile::Matrix2Df(n, n),
                                             tile_compile::Matrix2Df(n, n),
                                             tile_compile::Matrix2Df(n, n)};
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      const float sky = 0.05f + common(rng) + ((x > n / 2 && y > n / 2) ? 0.02f : 0.0f);
      const float c = chroma_sigma * chroma(rng);
      rgb[0](y, x) = sky + c;
      rgb[1](y, x) = sky;
      rgb[2](y, x) = sky - c;
    }
  }
  for (int i = 0; i < 20; ++i) {
    const int y = 8 + (i * 23) % (n - 16);
    const int x = 8 + (i * 37) % (n - 16);
    rgb[0](y, x) = rgb[1](y, x) = rgb[2](y, x) = 0.7f;
  }
  return rgb;
}

} // namespace

TEST_CASE("hypermetric_frozen_params_give_the_reference_transfer_curve") {
  constexpr int n = 128;
  tile_compile::image::HyperMetricStretchConfig cfg;
  cfg.enabled = true;
  cfg.mode = "ready_to_use";
  cfg.adaptive_anchor = true;

  const auto noisy = make_sky_rgb(n, 0.012f);   // before the chroma pass
  const auto clean = make_sky_rgb(n, 0.0f);     // after a chroma-only denoise

  auto ref = noisy;
  const auto ref_diag =
      tile_compile::image::run_hypermetric_stretch_rgb(ref[0], ref[1], ref[2], cfg);
  REQUIRE(ref_diag.success);
  REQUIRE(ref_diag.derived.valid);
  REQUIRE(ref_diag.derived.output_scaling);
  REQUIRE(ref_diag.params_source == "image");

  // Derived from the denoised image itself, the floors move.
  auto own = clean;
  const auto own_diag =
      tile_compile::image::run_hypermetric_stretch_rgb(own[0], own[1], own[2], cfg);
  REQUIRE(own_diag.success);
  REQUIRE(std::abs(own_diag.derived.floor_r - ref_diag.derived.floor_r) > 1.0e-2f);
  REQUIRE(std::abs(own_diag.derived.mtf_midtone - ref_diag.derived.mtf_midtone) > 1.0e-2f);

  // Frozen: the denoised image gets exactly the reference's parameters.
  auto frozen = clean;
  const auto frozen_diag = tile_compile::image::run_hypermetric_stretch_rgb(
      frozen[0], frozen[1], frozen[2], cfg, nullptr, 0, 0, nullptr,
      &ref_diag.derived);
  REQUIRE(frozen_diag.success);
  REQUIRE(frozen_diag.params_source == "frozen");
  const auto &a = frozen_diag.derived;
  const auto &b = ref_diag.derived;
  REQUIRE(a.input_scale == b.input_scale);
  REQUIRE(a.anchor == b.anchor);
  REQUIRE(a.log_d == b.log_d);
  REQUIRE(a.star_pressure == b.star_pressure);
  REQUIRE(a.floor_r == b.floor_r);
  REQUIRE(a.floor_g == b.floor_g);
  REQUIRE(a.floor_b == b.floor_b);
  REQUIRE(a.scale == b.scale);
  REQUIRE(a.mtf_applied == b.mtf_applied);
  REQUIRE(a.mtf_midtone == b.mtf_midtone);

  // The same transfer curve: pixels identical in both inputs (the stars) come
  // out identical. Elsewhere HMS stretches each pixel's own luminance and
  // keeps its own channel ratios, so the chroma noise still shows there.
  int stars = 0;
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      if (noisy[0](y, x) != clean[0](y, x) || noisy[2](y, x) != clean[2](y, x)) continue;
      ++stars;
      for (int c = 0; c < 3; ++c) REQUIRE(frozen[c](y, x) == ref[c](y, x));
    }
  }
  REQUIRE(stars >= 15);

  // Replaying an image's own parameters reproduces its own run exactly, and
  // parameters marked invalid are ignored.
  auto replay = noisy;
  tile_compile::image::run_hypermetric_stretch_rgb(
      replay[0], replay[1], replay[2], cfg, nullptr, 0, 0, nullptr, &ref_diag.derived);
  for (int c = 0; c < 3; ++c) REQUIRE((replay[c] - ref[c]).cwiseAbs().maxCoeff() == 0.0f);
  tile_compile::image::HyperMetricDerivedParams invalid = ref_diag.derived;
  invalid.valid = false;
  auto ignored = clean;
  const auto ignored_diag = tile_compile::image::run_hypermetric_stretch_rgb(
      ignored[0], ignored[1], ignored[2], cfg, nullptr, 0, 0, nullptr, &invalid);
  REQUIRE(ignored_diag.params_source == "image");
  for (int c = 0; c < 3; ++c) REQUIRE((ignored[c] - own[c]).cwiseAbs().maxCoeff() == 0.0f);
}
#endif
