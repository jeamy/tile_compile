#if __has_include(<catch2/catch_test_macros.hpp>)
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/errors.hpp"
#include "tile_compile/metrics/aqmh_quality_map.hpp"
#include "tile_compile/metrics/aqmh_eps.hpp"
#include "tile_compile/metrics/aqmh_global_quality.hpp"

#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace {

tile_compile::Matrix2Df make_gradient(int h, int w) {
  tile_compile::Matrix2Df frame(h, w);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x)
      frame(y, x) = 100.0f + static_cast<float>(x) * 0.5f +
                    static_cast<float>(y) * 0.25f;
  }
  return frame;
}

} // namespace

TEST_CASE("aqmh_config_validates_q_region") {
  // Config() default sets method="aqmh", so this tests AQMH validation
  tile_compile::config::Config cfg;
  cfg.aqmh.enabled = true;
  cfg.aqmh.diagnostics.q_region = 0.75f;
  REQUIRE_NOTHROW(cfg.validate());

  cfg.aqmh.diagnostics.q_region = 1.1f;
  REQUIRE_THROWS_AS(cfg.validate(), tile_compile::ValidationError);
}

TEST_CASE("aqmh_eps_noise_is_background_offset_invariant") {
  const std::vector<float> a{1000.0f, 1000.001f, 999.999f, 1000.002f};
  const std::vector<float> b{0.0f, 0.001f, -0.001f, 0.002f};
  REQUIRE(tile_compile::metrics::eps_noise(a) ==
          Catch::Approx(tile_compile::metrics::eps_noise(b)).margin(1.0e-10f));
}

TEST_CASE("aqmh_global_quality_is_native_positive_and_bounded") {
  tile_compile::config::AqmhGlobalQualityConfig cfg;
  const auto result = tile_compile::metrics::compute_aqmh_global_quality(
      {1.0f, 2.0f, 3.0f}, {3.0f, 2.0f, 1.0f}, cfg);
  REQUIRE(result.weights.size() == 3);
  for (float value : result.weights) {
    REQUIRE(value > cfg.g_floor);
    REQUIRE(value < 1.0f);
  }
}

TEST_CASE("aqmh_quality_map_applies_frame_specific_valid_mask") {
  tile_compile::config::AqmhPyramidConfig cfg;
  tile_compile::Matrix2Df frame = make_gradient(32, 32);
  std::vector<uint8_t> canvas(32u * 32u, 1u);
  std::vector<uint8_t> frame_mask(32u * 32u, 1u);
  frame_mask[10u * 32u + 11u] = 0u;
  const auto result = tile_compile::metrics::compute_aqmh_quality_map(
      frame, canvas, frame_mask, 32, 32, cfg);
  REQUIRE(result.q_map(10, 11) == 0.0f);
}

TEST_CASE("aqmh_quality_map_all_canvas_invalid_outputs_zero") {
  tile_compile::Matrix2Df frame = make_gradient(24, 24);
  std::vector<uint8_t> mask(static_cast<size_t>(24 * 24), 0u);

  tile_compile::config::AqmhPyramidConfig cfg;
  cfg.scales = 1;
  const auto out =
      tile_compile::metrics::compute_aqmh_quality_map(frame, mask, 24, 24, cfg);

  REQUIRE(out.q_map.rows() == 24);
  REQUIRE(out.q_map.cols() == 24);
  for (int y = 0; y < out.q_map.rows(); ++y) {
    for (int x = 0; x < out.q_map.cols(); ++x)
      REQUIRE(out.q_map(y, x) == Catch::Approx(0.0f).margin(1.0e-7f));
  }
}

TEST_CASE("aqmh_quality_map_ignores_canvas_invalid_pixel_values") {
  constexpr int H = 24;
  constexpr int W = 24;
  tile_compile::Matrix2Df a = make_gradient(H, W);
  tile_compile::Matrix2Df b = a;
  std::vector<uint8_t> mask(static_cast<size_t>(H * W), 1u);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < 8; ++x) {
      mask[static_cast<size_t>(y) * W + x] = 0u;
      a(y, x) = -100000.0f;
      b(y, x) = 100000.0f;
    }
  }

  tile_compile::config::AqmhPyramidConfig cfg;
  cfg.scales = 1;
  const auto qa =
      tile_compile::metrics::compute_aqmh_quality_map(a, mask, W, H, cfg);
  const auto qb =
      tile_compile::metrics::compute_aqmh_quality_map(b, mask, W, H, cfg);

  for (int y = 0; y < H; ++y) {
    for (int x = 10; x < W; ++x) {
      REQUIRE(qa.q_map(y, x) == Catch::Approx(qb.q_map(y, x)).margin(1.0e-6f));
    }
  }
}

TEST_CASE("aqmh_quality_map_is_deterministic_and_clamped") {
  tile_compile::Matrix2Df frame = make_gradient(32, 32);
  frame(12, 12) = 1000.0f;
  std::vector<uint8_t> mask(static_cast<size_t>(32 * 32), 1u);

  tile_compile::config::AqmhPyramidConfig cfg;
  cfg.scales = 2;
  const auto a =
      tile_compile::metrics::compute_aqmh_quality_map(frame, mask, 32, 32, cfg);
  const auto b =
      tile_compile::metrics::compute_aqmh_quality_map(frame, mask, 32, 32, cfg);

  REQUIRE(!a.diagnostics.omitted_scales.empty());
  REQUIRE(std::isnan(a.diagnostics.snr_p50));
  for (int y = 0; y < a.q_map.rows(); ++y) {
    for (int x = 0; x < a.q_map.cols(); ++x) {
      REQUIRE(a.q_map(y, x) == Catch::Approx(b.q_map(y, x)).margin(1.0e-7f));
      REQUIRE(a.q_map(y, x) >= 0.0f);
      REQUIRE(a.q_map(y, x) <= 1.0f);
    }
  }
}

TEST_CASE("aqmh_gpu_sharpness_path_matches_cpu_quality_map") {
  constexpr int H = 64;
  constexpr int W = 64;
  tile_compile::Matrix2Df frame(H, W);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      frame(y, x) = 100.0f + 0.17f * static_cast<float>(x) +
                    0.11f * static_cast<float>(y) +
                    2.0f * std::sin(0.21f * static_cast<float>(x)) *
                        std::cos(0.13f * static_cast<float>(y));
    }
  }
  frame(21, 37) += 80.0f;
  std::vector<uint8_t> mask(static_cast<size_t>(H * W), 1u);
  for (int y = 0; y < H; ++y)
    mask[static_cast<size_t>(y) * W] = 0u;

  tile_compile::config::AqmhPyramidConfig cfg;
  cfg.scales = 2;
  const auto cpu = tile_compile::metrics::compute_aqmh_quality_map(
      frame, mask, W, H, cfg, tile_compile::core::AccelerationBackend::cpu);

  tile_compile::core::AccelerationContext context("auto");
  const auto selection =
      context.selection_for(tile_compile::core::AccelerationPhase::aqmh_maps);
  const auto accelerated = tile_compile::metrics::compute_aqmh_quality_map(
      frame, mask, W, H, cfg, selection.selected);

  if (selection.using_gpu)
    REQUIRE(accelerated.diagnostics.acceleration_used);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      REQUIRE(accelerated.q_map(y, x) ==
              Catch::Approx(cpu.q_map(y, x)).margin(2.0e-3f));
    }
  }
}

TEST_CASE("aqmh_opencl_sharpness_path_matches_cpu_quality_map") {
  constexpr int H = 48;
  constexpr int W = 48;
  tile_compile::Matrix2Df frame = make_gradient(H, W);
  for (int y = 4; y < H; y += 9)
    for (int x = 3; x < W; x += 11)
      frame(y, x) += static_cast<float>((x + y) % 17);
  std::vector<uint8_t> mask(static_cast<size_t>(H * W), 1u);
  tile_compile::config::AqmhPyramidConfig cfg;
  cfg.scales = 1;

  tile_compile::core::AccelerationContext context("opencv_opencl");
  const auto selection =
      context.selection_for(tile_compile::core::AccelerationPhase::aqmh_maps);
  if (!selection.using_gpu)
    return;

  const auto cpu = tile_compile::metrics::compute_aqmh_quality_map(
      frame, mask, W, H, cfg, tile_compile::core::AccelerationBackend::cpu);
  const auto opencl = tile_compile::metrics::compute_aqmh_quality_map(
      frame, mask, W, H, cfg, selection.selected);
  REQUIRE(opencl.diagnostics.acceleration_used);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      REQUIRE(opencl.q_map(y, x) ==
              Catch::Approx(cpu.q_map(y, x)).margin(2.0e-3f));
}
#endif
