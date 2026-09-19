#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/errors.hpp"
#include "tile_compile/reconstruction/chroma_denoise.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <vector>

using tile_compile::Matrix2Df;

TEST_CASE("chroma denoise reports extended-source protection and preserves luma",
          "[chroma-denoise]") {
  constexpr int W = 256;
  constexpr int H = 192;
  Matrix2Df r(H, W), g(H, W), b(H, W);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const float dx = static_cast<float>(x - W / 2);
      const float dy = static_cast<float>(y - H / 2);
      const bool source = dx * dx + dy * dy < 42.0f * 42.0f;
      const float luma = source ? 4.0f : 1.0f;
      const float chroma = ((x + y) & 1) ? 0.12f : -0.12f;
      r(y, x) = luma + chroma;
      g(y, x) = luma;
      b(y, x) = luma - chroma;
    }
  }
  const Matrix2Df y_before = (0.25f * r + 0.5f * g + 0.25f * b).eval();

  tile_compile::config::ChromaDenoiseConfig cfg;
  cfg.enabled = true;
  cfg.protect_luma = true;
  cfg.star_protection.enabled = false;
  cfg.structure_protection.enabled = false;
  cfg.extended_source_protection.enabled = true;
  cfg.extended_source_protection.luma_sigma = 2.5f;
  cfg.extended_source_protection.dilate_px = 5;

  const auto stats = tile_compile::reconstruction::chroma_denoise_rgb_inplace(
      r, g, b, cfg);
  const Matrix2Df y_after = (0.25f * r + 0.5f * g + 0.25f * b).eval();

  REQUIRE(stats.applied);
  REQUIRE(stats.valid_pixels == static_cast<std::uint64_t>(W * H));
  REQUIRE(stats.extended_source_raw_fraction > 0.05);
  REQUIRE(stats.extended_source_raw_fraction < 0.5);
  REQUIRE(stats.extended_source_protected_fraction >=
          stats.extended_source_raw_fraction);
  REQUIRE(stats.combined_protected_fraction > 0.05);
  REQUIRE(stats.mean_denoise_fraction < stats.effective_blend_amount);
  REQUIRE((y_after - y_before).cwiseAbs().maxCoeff() < 1.0e-5f);
}

TEST_CASE("chroma denoise large_scale_bias removes a background-scale color "
          "blotch that the wavelet/bilateral stages cannot reach",
          "[chroma-denoise][large-scale-bias]") {
  constexpr int W = 320;
  constexpr int H = 256;
  Matrix2Df r(H, W), g(H, W), b(H, W);
  // A single, wide (sigma=60px, far beyond chroma_wavelet's coarsest scale
  // of 2^(levels-1)*0.75 <= ~9px) smooth chroma bias covering the whole
  // frame -- the failure mode described in the analysis: a background color
  // cast, not per-pixel chroma noise.
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const float dx = static_cast<float>(x - W / 4);
      const float dy = static_cast<float>(y - H / 3);
      const float blotch = 0.4f * std::exp(-(dx * dx + dy * dy) /
                                            (2.0f * 60.0f * 60.0f));
      const float luma = 1.0f;
      r(y, x) = luma + blotch;
      g(y, x) = luma;
      b(y, x) = luma - blotch;
    }
  }

  tile_compile::config::ChromaDenoiseConfig cfg;
  cfg.enabled = true;
  cfg.protect_luma = false;
  cfg.star_protection.enabled = false;
  cfg.structure_protection.enabled = false;
  cfg.extended_source_protection.enabled = false;
  cfg.blend.amount = 1.0f;

  // Baseline: without large_scale_bias, the wavelet+bilateral stages barely
  // touch the wide blotch (it is almost entirely in the coarsest "low" term,
  // which is never thresholded).
  {
    Matrix2Df r2 = r, g2 = g, b2 = b;
    cfg.large_scale_bias.enabled = false;
    const auto stats = tile_compile::reconstruction::chroma_denoise_rgb_inplace(
        r2, g2, b2, cfg);
    REQUIRE(stats.applied);
    const float center_before = r(H / 3, W / 4) - g(H / 3, W / 4);
    const float center_after = r2(H / 3, W / 4) - g2(H / 3, W / 4);
    // The blotch amplitude survives almost unchanged.
    REQUIRE(center_after > 0.8f * center_before);
  }

  // With large_scale_bias enabled, the blotch is flattened.
  {
    Matrix2Df r2 = r, g2 = g, b2 = b;
    cfg.large_scale_bias.enabled = true;
    cfg.large_scale_bias.block_size = 16;
    cfg.large_scale_bias.blur_sigma = 16.0f;
    cfg.large_scale_bias.strength = 1.0f;
    const auto stats = tile_compile::reconstruction::chroma_denoise_rgb_inplace(
        r2, g2, b2, cfg);
    REQUIRE(stats.applied);
    REQUIRE(stats.large_scale_bias_removed_rms_c1 > 0.0);
    REQUIRE(stats.large_scale_bias_removed_rms_c2 > 0.0);
    const float center_before = r(H / 3, W / 4) - g(H / 3, W / 4);
    const float center_after = r2(H / 3, W / 4) - g2(H / 3, W / 4);
    // The blotch amplitude is substantially reduced (flattened), unlike the
    // baseline above.
    REQUIRE(center_after < 0.4f * center_before);
    // Far from the blotch (flat background), the plane is essentially
    // unchanged: the reference level equals the (unaffected) far-field value.
    const float far_before = r(0, W - 1) - g(0, W - 1);
    const float far_after = r2(0, W - 1) - g2(0, W - 1);
    REQUIRE(std::fabs(far_after - far_before) < 0.05f);
  }
}

TEST_CASE("chroma denoise large_scale_bias desaturates an extended object "
          "toward the background chroma when it is NOT covered by "
          "extended_source_protection (the actual M31 regression)",
          "[chroma-denoise][large-scale-bias]") {
  // Reproduces the M31 regression directly: a large, smooth "galaxy" (a
  // clear MINORITY of the frame, as in the real run) with its own distinct
  // real chroma, while extended_source_protection is disabled (the missing
  // config block). The galaxy's own block legitimately participates in the
  // background-median estimate's surface at its own location regardless of
  // how robust the reference level is elsewhere -- so its real color gets
  // pulled toward the background chroma. Enabling extended_source_protection
  // (mirroring the actual fix applied to the run's config) excludes it from
  // the estimate and preserves its color.
  constexpr int W = 480;
  constexpr int H = 384;
  Matrix2Df r(H, W), g(H, W), b(H, W);
  const float cx = W * 0.5f, cy = H * 0.5f;
  const float galaxy_radius = 60.0f;  // a clear minority of the frame (~5%)
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const float dx = static_cast<float>(x) - cx;
      const float dy = static_cast<float>(y) - cy;
      const bool galaxy = (dx * dx + dy * dy) < galaxy_radius * galaxy_radius;
      const float luma = galaxy ? 2.0f : 1.0f;
      const float chroma = galaxy ? 0.25f : 0.0f;
      r(y, x) = luma + chroma;
      g(y, x) = luma;
      b(y, x) = luma - chroma;
    }
  }

  tile_compile::config::ChromaDenoiseConfig cfg;
  cfg.enabled = true;
  cfg.protect_luma = true;
  cfg.star_protection.enabled = false;
  cfg.structure_protection.enabled = false;
  cfg.chroma_wavelet.enabled = false;  // isolate large_scale_bias's own effect
  cfg.chroma_bilateral.enabled = false;
  cfg.blend.amount = 1.0f;
  cfg.large_scale_bias.enabled = true;
  cfg.large_scale_bias.block_size = 16;
  cfg.large_scale_bias.blur_sigma = 12.0f;
  cfg.large_scale_bias.strength = 1.0f;

  auto galaxy_chroma_after = [&](bool extended_source_protection) {
    Matrix2Df r2 = r, g2 = g, b2 = b;
    cfg.extended_source_protection.enabled = extended_source_protection;
    cfg.extended_source_protection.luma_sigma = 0.5f;  // catches luma=2.0 easily
    cfg.extended_source_protection.dilate_px = 5;
    const auto stats = tile_compile::reconstruction::chroma_denoise_rgb_inplace(
        r2, g2, b2, cfg);
    REQUIRE(stats.applied);
    return r2(static_cast<int>(cy), static_cast<int>(cx)) -
           g2(static_cast<int>(cy), static_cast<int>(cx));
  };

  const float chroma_before = r(static_cast<int>(cy), static_cast<int>(cx)) -
                               g(static_cast<int>(cy), static_cast<int>(cx));
  REQUIRE(chroma_before == Catch::Approx(0.25).margin(1.0e-6));

  // Without extended_source_protection (the actual regression): the galaxy's
  // real color is pulled toward the background (~0), just as observed on
  // the real M31 output (core R-G dropped ~38%).
  const float without_protection = galaxy_chroma_after(false);
  REQUIRE(without_protection < 0.5f * chroma_before);

  // With extended_source_protection (the actual fix): the galaxy is excluded
  // from the background estimate and its real color survives intact.
  const float with_protection = galaxy_chroma_after(true);
  REQUIRE(with_protection > 0.9f * chroma_before);
}

TEST_CASE("chroma denoise large_scale_bias does not subtract the interpolated "
          "surface inside protected regions (faint-halo contamination)",
          "[chroma-denoise][large-scale-bias]") {
  // Real M31 failure mode: extended_source_protection detects only the bright
  // core. A faint halo with the object's own chroma stays below the luma
  // threshold, remains unprotected, and contaminates the block medians around
  // the galaxy. The push-pull fill then carries halo-flavoured chroma under
  // the protected region; subtracting that surface removes the object's own
  // colour even though the object itself was correctly protected.
  constexpr int W = 480;
  constexpr int H = 384;
  Matrix2Df r(H, W), g(H, W), b(H, W);
  const float cx = W * 0.5f, cy = H * 0.5f;
  const float core_radius = 40.0f;
  const float halo_radius = 110.0f;
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const float dx = static_cast<float>(x) - cx;
      const float dy = static_cast<float>(y) - cy;
      const float rr = dx * dx + dy * dy;
      // Bright core is detected; the wide faint halo keeps object chroma at
      // ~60% strength but a luma far below the detection threshold.
      const bool core = rr < core_radius * core_radius;
      const bool halo = !core && rr < halo_radius * halo_radius;
      const float luma = core ? 3.0f : (halo ? 1.02f : 1.0f);
      const float chroma = core ? 0.25f : (halo ? 0.15f : 0.0f);
      r(y, x) = luma + chroma;
      g(y, x) = luma;
      b(y, x) = luma - chroma;
    }
  }

  tile_compile::config::ChromaDenoiseConfig cfg;
  cfg.enabled = true;
  cfg.protect_luma = true;
  cfg.star_protection.enabled = false;
  cfg.structure_protection.enabled = false;
  cfg.chroma_wavelet.enabled = false;  // isolate large_scale_bias's own effect
  cfg.chroma_bilateral.enabled = false;
  cfg.blend.amount = 1.0f;
  cfg.extended_source_protection.enabled = true;
  cfg.extended_source_protection.luma_sigma = 3.0f;  // catches core, not halo
  cfg.extended_source_protection.dilate_px = 3;
  cfg.large_scale_bias.enabled = true;
  cfg.large_scale_bias.block_size = 16;
  cfg.large_scale_bias.blur_sigma = 12.0f;
  cfg.large_scale_bias.strength = 1.0f;

  const auto stats = tile_compile::reconstruction::chroma_denoise_rgb_inplace(
      r, g, b, cfg);
  REQUIRE(stats.applied);
  REQUIRE(stats.extended_source_raw_fraction > 0.01);

  // The protected core must keep its own chroma: the correction may not be
  // applied where the surface is only an interpolation.
  const float core_chroma = r(static_cast<int>(cy), static_cast<int>(cx)) -
                            g(static_cast<int>(cy), static_cast<int>(cx));
  REQUIRE(core_chroma > 0.9f * 0.25f);
}

TEST_CASE("chroma denoise adaptation_reference_ratio keeps adaptation "
          "scale-invariant across normalized vs. ADU-scale data",
          "[chroma-denoise][adaptation]") {
  constexpr int W = 96;
  constexpr int H = 96;
  auto make_planes = [&](float scale) {
    Matrix2Df r(H, W), g(H, W), b(H, W);
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const float luma = scale * 10.0f;
        const float chroma = ((x + y) & 1) ? scale * 0.3f : -scale * 0.3f;
        r(y, x) = luma + chroma;
        g(y, x) = luma;
        b(y, x) = luma - chroma;
      }
    }
    return std::array<Matrix2Df, 3>{r, g, b};
  };

  tile_compile::config::ChromaDenoiseConfig cfg;
  cfg.enabled = true;
  cfg.protect_luma = false;
  cfg.star_protection.enabled = false;
  cfg.structure_protection.enabled = false;
  cfg.extended_source_protection.enabled = false;

  // Same relative chroma-to-luma noise ratio, two absolute scales differing
  // by 500x (roughly [0,1]-normalized vs. real ADU-scale data).
  auto small = make_planes(1.0f);
  auto large = make_planes(500.0f);
  const auto stats_small = tile_compile::reconstruction::chroma_denoise_rgb_inplace(
      small[0], small[1], small[2], cfg);
  const auto stats_large = tile_compile::reconstruction::chroma_denoise_rgb_inplace(
      large[0], large[1], large[2], cfg);

  REQUIRE(stats_small.applied);
  REQUIRE(stats_large.applied);
  // With the old fixed-absolute ref_sigma=0.02f, `large`'s adaptation would
  // saturate at the 1.4 clamp regardless of `small`'s value. The scale-
  // relative formula instead yields (near-)identical adaptation for both,
  // since the underlying chroma-to-luma ratio is the same.
  REQUIRE(stats_small.adaptation == Catch::Approx(stats_large.adaptation).margin(1.0e-3));
}

TEST_CASE("chroma bilateral range is invariant to normalized versus ADU scale",
          "[chroma-denoise][adaptation]") {
  constexpr int W = 96, H = 72;
  auto make = [&](float scale) {
    std::array<Matrix2Df, 3> rgb{Matrix2Df(H, W), Matrix2Df(H, W),
                                 Matrix2Df(H, W)};
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        const float n = ((x * 17 + y * 11) % 13 - 6) * 0.01f;
        const float c = (x < W / 2 ? -0.12f : 0.12f) + n;
        rgb[0](y, x) = scale * (1.0f + c);
        rgb[1](y, x) = scale;
        rgb[2](y, x) = scale * (1.0f - c);
      }
    return rgb;
  };
  auto normalized = make(1.0f);
  auto adu = make(500.0f);
  tile_compile::config::ChromaDenoiseConfig cfg;
  cfg.enabled = true;
  cfg.protect_luma = false;
  cfg.chroma_wavelet.enabled = false;
  cfg.chroma_bilateral.enabled = true;
  cfg.chroma_bilateral.sigma_range = 2.0f;
  cfg.blend.amount = 1.0f;
  tile_compile::reconstruction::chroma_denoise_rgb_inplace(
      normalized[0], normalized[1], normalized[2], cfg);
  tile_compile::reconstruction::chroma_denoise_rgb_inplace(
      adu[0], adu[1], adu[2], cfg);
  for (int ch = 0; ch < 3; ++ch)
    REQUIRE((normalized[ch] - adu[ch] / 500.0f).cwiseAbs().maxCoeff() < 2.0e-5f);
}

TEST_CASE("chroma denoise excludes invalid canvas pixels from statistics and output",
          "[chroma-denoise][mask]") {
  constexpr int W = 64, H = 48;
  Matrix2Df r = Matrix2Df::Constant(H, W, 10.0f);
  Matrix2Df g = r;
  Matrix2Df b = r;
  std::vector<std::uint8_t> valid(static_cast<size_t>(W * H), 1);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < 8; ++x) {
      const size_t i = static_cast<size_t>(y * W + x);
      valid[i] = 0;
      r(y, x) = 10000.0f;
      g(y, x) = -5000.0f;
      b(y, x) = 3000.0f;
    }
  }
  const Matrix2Df r0 = r, g0 = g, b0 = b;
  tile_compile::config::ChromaDenoiseConfig cfg;
  cfg.enabled = true;
  cfg.star_protection.enabled = false;
  cfg.structure_protection.enabled = false;
  cfg.extended_source_protection.enabled = false;
  const auto stats = tile_compile::reconstruction::chroma_denoise_rgb_inplace(
      r, g, b, cfg, nullptr, &valid);
  REQUIRE(stats.valid_pixels == static_cast<std::uint64_t>((W - 8) * H));
  REQUIRE((r.leftCols(8) - r0.leftCols(8)).cwiseAbs().maxCoeff() == 0.0f);
  REQUIRE((g.leftCols(8) - g0.leftCols(8)).cwiseAbs().maxCoeff() == 0.0f);
  REQUIRE((b.leftCols(8) - b0.leftCols(8)).cwiseAbs().maxCoeff() == 0.0f);
  REQUIRE((r.rightCols(W - 8).array() - 10.0f).abs().maxCoeff() < 1.0e-4f);
  REQUIRE((g.rightCols(W - 8).array() - 10.0f).abs().maxCoeff() < 1.0e-4f);
  REQUIRE((b.rightCols(W - 8).array() - 10.0f).abs().maxCoeff() < 1.0e-4f);
}

TEST_CASE("chroma structure protection detects a pure chroma edge",
          "[chroma-denoise][mask]") {
  constexpr int W = 128, H = 64;
  Matrix2Df r(H, W), g(H, W), b(H, W);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const float c = x < W / 2 ? -0.25f : 0.25f;
      r(y, x) = 1.0f + c;
      g(y, x) = 1.0f;
      b(y, x) = 1.0f - c;
    }
  tile_compile::config::ChromaDenoiseConfig cfg;
  cfg.enabled = true;
  cfg.star_protection.enabled = false;
  cfg.extended_source_protection.enabled = false;
  cfg.structure_protection.enabled = true;
  Matrix2Df mask;
  tile_compile::reconstruction::ChromaDenoiseDiagnostics diagnostics;
  const auto stats = tile_compile::reconstruction::chroma_denoise_rgb_inplace(
      r, g, b, cfg, &mask, nullptr, &diagnostics);
  REQUIRE(stats.structure_protected_fraction > 0.0);
  REQUIRE(mask(H / 2, W / 2) > 0.25f);
  REQUIRE(diagnostics.structure_mask.rows() == H);
  REQUIRE(diagnostics.effective_amount.rows() == H);
}

TEST_CASE("chroma extended-source mask does not classify a quarter of Gaussian sky",
          "[chroma-denoise][mask]") {
  constexpr int W = 320, H = 240;
  Matrix2Df r(H, W), g(H, W), b(H, W);
  std::mt19937 rng(7);
  std::normal_distribution<float> noise(0.0f, 0.02f);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      r(y, x) = g(y, x) = b(y, x) = 1.0f + noise(rng);
  tile_compile::config::ChromaDenoiseConfig cfg;
  cfg.enabled = true;
  cfg.star_protection.enabled = false;
  cfg.structure_protection.enabled = false;
  cfg.extended_source_protection.enabled = true;
  cfg.extended_source_protection.dilate_px = 0;
  const auto stats = tile_compile::reconstruction::chroma_denoise_rgb_inplace(
      r, g, b, cfg);
  REQUIRE(stats.extended_source_raw_fraction < 0.05);
}

TEST_CASE("new BGE and chroma protection ranges are validated",
          "[config][chroma-denoise][autobge]") {
  tile_compile::config::Config cfg;

  cfg.chroma_denoise.apply_stage = "pre_stack_tiles";
  REQUIRE_THROWS_AS(cfg.validate(), tile_compile::ValidationError);
  cfg.chroma_denoise.apply_stage = "both";
  REQUIRE_THROWS_AS(cfg.validate(), tile_compile::ValidationError);
  cfg.chroma_denoise.apply_stage = "post_pcc";

  cfg.bge.auto_detect.gradient_threshold = 0.0f;
  REQUIRE_THROWS_AS(cfg.validate(), tile_compile::ValidationError);
  cfg.bge.auto_detect.gradient_threshold = 0.05f;

  cfg.bge.auto_detect.extended_source_sigma = 6.1f;
  REQUIRE_THROWS_AS(cfg.validate(), tile_compile::ValidationError);
  cfg.bge.auto_detect.extended_source_sigma = 3.0f;

  cfg.bge.auto_detect.extended_source_dilate_px = 201;
  REQUIRE_THROWS_AS(cfg.validate(), tile_compile::ValidationError);
  cfg.bge.auto_detect.extended_source_dilate_px = 50;

  cfg.chroma_denoise.extended_source_protection.luma_sigma = 0.5f;
  REQUIRE_THROWS_AS(cfg.validate(), tile_compile::ValidationError);
  cfg.chroma_denoise.extended_source_protection.luma_sigma = 2.5f;

  cfg.chroma_denoise.extended_source_protection.dilate_px = 101;
  REQUIRE_THROWS_AS(cfg.validate(), tile_compile::ValidationError);
  cfg.chroma_denoise.extended_source_protection.dilate_px = 30;

  cfg.bge.method = "auto";
  cfg.bge.autobge.patch_size = 4;
  REQUIRE_THROWS_AS(cfg.validate(), tile_compile::ValidationError);
}

TEST_CASE("chroma denoise adaptation_reference_ratio and large_scale_bias "
          "ranges are validated",
          "[config][chroma-denoise]") {
  tile_compile::config::Config cfg;

  cfg.chroma_denoise.adaptation_reference_ratio = 0.0f;
  REQUIRE_THROWS_AS(cfg.validate(), tile_compile::ValidationError);
  cfg.chroma_denoise.adaptation_reference_ratio = 1.0f;

  cfg.chroma_denoise.large_scale_bias.block_size = 3;
  REQUIRE_THROWS_AS(cfg.validate(), tile_compile::ValidationError);
  cfg.chroma_denoise.large_scale_bias.block_size = 32;

  cfg.chroma_denoise.large_scale_bias.blur_sigma = -1.0f;
  REQUIRE_THROWS_AS(cfg.validate(), tile_compile::ValidationError);
  cfg.chroma_denoise.large_scale_bias.blur_sigma = 24.0f;

  cfg.chroma_denoise.large_scale_bias.strength = 1.5f;
  REQUIRE_THROWS_AS(cfg.validate(), tile_compile::ValidationError);
  cfg.chroma_denoise.large_scale_bias.strength = 1.0f;

  REQUIRE_NOTHROW(cfg.validate());
}
