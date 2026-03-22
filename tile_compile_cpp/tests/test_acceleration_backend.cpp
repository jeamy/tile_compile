#if __has_include(<catch2/catch_test_macros.hpp>)
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/acceleration.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("runtime_limits_acceleration_backend_parses_and_validates") {
  YAML::Node node = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: MONO
runtime_limits:
  acceleration_backend: auto
)");

  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE(cfg.runtime_limits.acceleration_backend == "auto");
  REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("runtime_limits_acceleration_backend_rejects_invalid_value") {
  YAML::Node node = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: MONO
runtime_limits:
  acceleration_backend: vulkan
)");

  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE_THROWS(cfg.validate());
}

TEST_CASE("acceleration_backend_selection_keeps_cpu_requests") {
  const auto selection = tile_compile::core::select_acceleration_backend(
      "cpu", tile_compile::core::AccelerationPhase::prewarp);

  REQUIRE(selection.request_honored);
  REQUIRE(tile_compile::core::acceleration_backend_name(selection.requested) ==
          "cpu");
  REQUIRE(tile_compile::core::acceleration_backend_name(selection.selected) ==
          "cpu");
  REQUIRE_FALSE(selection.using_gpu);
}

TEST_CASE("acceleration_backend_selection_auto_chooses_supported_stacking_backend") {
  const auto selection = tile_compile::core::select_acceleration_backend(
      "auto", tile_compile::core::AccelerationPhase::stacking);

  REQUIRE(selection.requested_name == "auto");
  REQUIRE(selection.auto_requested);
  REQUIRE(selection.request_honored);
  if (selection.opencv_cuda_headers && selection.opencv_cuda_runtime) {
    REQUIRE(tile_compile::core::acceleration_backend_name(selection.selected) ==
            "opencv_cuda");
    REQUIRE(selection.using_gpu);
  } else {
    REQUIRE(tile_compile::core::acceleration_backend_name(selection.selected) ==
            "cpu");
    REQUIRE_FALSE(selection.using_gpu);
  }
}

TEST_CASE("device_batch_descriptors_report_expected_sizes") {
  const auto frame_batch =
      tile_compile::core::make_device_frame_batch(5, 10, 12, 3);
  REQUIRE(frame_batch.batch_size == 5);
  REQUIRE(frame_batch.frame.rows == 10);
  REQUIRE(frame_batch.frame.cols == 12);
  REQUIRE(frame_batch.frame.channels == 3);
  REQUIRE(frame_batch.frame.bytes ==
          static_cast<size_t>(10 * 12 * 3 * sizeof(float)));
  REQUIRE(frame_batch.total_bytes ==
          static_cast<size_t>(5 * 10 * 12 * 3 * sizeof(float)));

  std::vector<tile_compile::Tile> tiles = {
      {0, 0, 32, 32, 0, 0},
      {16, 16, 32, 24, 0, 1},
  };
  const auto tile_batch = tile_compile::core::make_device_tile_batch(tiles, 3);
  REQUIRE(tile_batch.batch_size == tiles.size());
  REQUIRE(tile_batch.channels == 3);
  REQUIRE(tile_batch.max_tile_width == 32);
  REQUIRE(tile_batch.max_tile_height == 32);
  REQUIRE(tile_batch.total_pixels == static_cast<size_t>(32 * 32 + 32 * 24));
  REQUIRE(tile_batch.total_bytes ==
          static_cast<size_t>((32 * 32 + 32 * 24) * 3 * sizeof(float)));
}
#else
int tile_compile_tests_acceleration_backend_stub() { return 0; }
#endif
