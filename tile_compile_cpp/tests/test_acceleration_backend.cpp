#if __has_include(<catch2/catch_test_macros.hpp>)
#include "../apps/runner_shared.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/acceleration.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <thread>

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

TEST_CASE("runtime_limits_acceleration_backend_accepts_opencl_alias") {
  YAML::Node node = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: MONO
runtime_limits:
  acceleration_backend: opencl
)");

  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE(cfg.runtime_limits.acceleration_backend == "opencl");
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

TEST_CASE("runtime_limits_thresholds_parse_and_validate") {
  YAML::Node node = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: MONO
runtime_limits:
  tile_analysis_max_factor_vs_stack: 2.5
  hard_abort_hours: 3.5
)");

  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE(cfg.runtime_limits.tile_analysis_max_factor_vs_stack ==
          Catch::Approx(2.5f));
  REQUIRE(cfg.runtime_limits.hard_abort_hours == Catch::Approx(3.5f));
  REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("runtime_limits_thresholds_reject_non_positive_values") {
  YAML::Node node = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: MONO
runtime_limits:
  tile_analysis_max_factor_vs_stack: 0.0
  hard_abort_hours: -1.0
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
  } else if (selection.opencv_opencl_headers &&
             selection.opencv_opencl_runtime) {
    REQUIRE(tile_compile::core::acceleration_backend_name(selection.selected) ==
            "opencv_opencl");
    REQUIRE(selection.using_gpu);
  } else {
    REQUIRE(tile_compile::core::acceleration_backend_name(selection.selected) ==
            "cpu");
    REQUIRE_FALSE(selection.using_gpu);
  }
}

TEST_CASE("acceleration_backend_selection_parses_opencl_requests") {
  tile_compile::core::AccelerationBackend backend =
      tile_compile::core::AccelerationBackend::cpu;
  REQUIRE(tile_compile::core::parse_acceleration_backend("opencl", backend));
  REQUIRE(tile_compile::core::acceleration_backend_name(backend) ==
          "opencv_opencl");

  const auto selection = tile_compile::core::select_acceleration_backend(
      "opencl", tile_compile::core::AccelerationPhase::stacking);
  REQUIRE(selection.requested_name == "opencl");
  REQUIRE(tile_compile::core::acceleration_backend_name(selection.requested) ==
          "opencv_opencl");
  if (selection.opencv_opencl_headers && selection.opencv_opencl_runtime) {
    REQUIRE(selection.request_honored);
    REQUIRE(tile_compile::core::acceleration_backend_name(selection.selected) ==
            "opencv_opencl");
    REQUIRE(selection.using_gpu);
  } else {
    REQUIRE_FALSE(selection.request_honored);
    REQUIRE(tile_compile::core::acceleration_backend_name(selection.selected) ==
            "cpu");
    REQUIRE_FALSE(selection.fallback_reason.empty());
    REQUIRE_FALSE(selection.using_gpu);
  }
}

TEST_CASE("acceleration_selection_json_includes_opencl_capabilities") {
  const auto selection = tile_compile::core::select_acceleration_backend(
      "auto", tile_compile::core::AccelerationPhase::prewarp);
  const auto json =
      tile_compile::core::acceleration_selection_to_json(selection);

  REQUIRE(json.find("opencv_cuda_headers") != json.end());
  REQUIRE(json.find("opencv_cuda_runtime") != json.end());
  REQUIRE(json.find("opencv_opencl_headers") != json.end());
  REQUIRE(json.find("opencv_opencl_runtime") != json.end());
}

TEST_CASE("acceleration_context_keeps_run_scoped_cpu_selection") {
  tile_compile::core::AccelerationContext context("cpu");
  for (const auto phase : {
           tile_compile::core::AccelerationPhase::prewarp,
           tile_compile::core::AccelerationPhase::aqmh_maps,
           tile_compile::core::AccelerationPhase::aqmh_reconstruction,
           tile_compile::core::AccelerationPhase::tile_reconstruction,
           tile_compile::core::AccelerationPhase::stacking}) {
    const auto selection = context.selection_for(phase);
    REQUIRE(selection.request_honored);
    REQUIRE(selection.selected ==
            tile_compile::core::AccelerationBackend::cpu);
    REQUIRE_FALSE(selection.using_gpu);
  }
  const auto artifact = context.to_json();
  REQUIRE(artifact["requested_backend"] == "cpu");
  REQUIRE(artifact["phases"].contains("AQMH_MAPS"));
  REQUIRE(artifact["phases"].contains("AQMH_RECONSTRUCTION"));
}

TEST_CASE("acceleration_context_keeps_aqmh_maps_cpu_only") {
  tile_compile::core::AccelerationContext context("opencv_cuda");
  const auto selection =
      context.selection_for(tile_compile::core::AccelerationPhase::aqmh_maps);
  REQUIRE(selection.selected == tile_compile::core::AccelerationBackend::cpu);
  REQUIRE_FALSE(selection.using_gpu);
  REQUIRE_FALSE(selection.request_honored);
  REQUIRE_FALSE(selection.fallback_reason.empty());
}

TEST_CASE("worker_cuda_streams_match_selected_cuda_backend") {
  tile_compile::core::AccelerationContext context("auto");
  const auto selection =
      context.selection_for(tile_compile::core::AccelerationPhase::aqmh_maps);
  const bool use_cuda =
      selection.selected == tile_compile::core::AccelerationBackend::opencv_cuda;
  tile_compile::core::WorkerCudaStreams streams(use_cuda, 3);
  if (use_cuda) {
    REQUIRE(streams.size() == 3);
    REQUIRE(streams.get(0) != nullptr);
    REQUIRE(streams.get(1) != nullptr);
    REQUIRE(streams.get(0) != streams.get(1));
  } else {
    REQUIRE(streams.size() == 0);
    REQUIRE(streams.get(0) == nullptr);
  }
}

TEST_CASE("opencv_opencl_sigma_clip_is_safe_across_worker_threads") {
  tile_compile::core::AccelerationContext context("opencv_opencl");
  const auto selection =
      context.selection_for(tile_compile::core::AccelerationPhase::stacking);
  if (selection.selected !=
      tile_compile::core::AccelerationBackend::opencv_opencl) {
    return;
  }

  const tile_compile::core::AccelerationOps ops(
      context, tile_compile::core::AccelerationPhase::stacking);
  std::vector<tile_compile::Matrix2Df> frames;
  for (int i = 0; i < 4; ++i) {
    frames.push_back(tile_compile::Matrix2Df::Constant(
        64, 64, 10.0f + static_cast<float>(i)));
  }
  const std::vector<float> weights(frames.size(), 1.0f);
  std::vector<tile_compile::Matrix2Df> results(4);
  std::vector<std::thread> workers;
  for (size_t worker = 0; worker < results.size(); ++worker) {
    workers.emplace_back([&, worker] {
      results[worker] =
          ops.sigma_clip_reduce(frames, weights, 3.0f, 3.0f, 2, 0.5f,
                                1.0e-6f)
              .tile;
    });
  }
  for (auto &worker : workers)
    worker.join();
  for (const auto &result : results) {
    REQUIRE(result.rows() == 64);
    REQUIRE(result.cols() == 64);
    REQUIRE(result(0, 0) == Catch::Approx(11.5f).margin(1.0e-3f));
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

TEST_CASE("warp_affine_frame_preserves_support_for_negative_samples") {
  tile_compile::core::AccelerationSelection selection;
  selection.phase = tile_compile::core::AccelerationPhase::prewarp;
  selection.requested = tile_compile::core::AccelerationBackend::cpu;
  selection.selected = tile_compile::core::AccelerationBackend::cpu;
  selection.requested_name = "cpu";

  tile_compile::core::AccelerationOps ops(selection);
  tile_compile::Matrix2Df frame(2, 2);
  frame << -1.0f, -2.0f, -3.0f, -4.0f;

  tile_compile::WarpMatrix warp = tile_compile::WarpMatrix::Identity();
  tile_compile::Matrix2Df warped;
  std::vector<uint8_t> valid_mask;
  bool has_data = false;

  REQUIRE(ops.warp_affine_frame(frame, warp, tile_compile::ColorMode::MONO, 2,
                                2, 0, 0, warped, &valid_mask, &has_data));
  REQUIRE(has_data);
  REQUIRE(valid_mask == std::vector<uint8_t>({1u, 1u, 1u, 1u}));
  REQUIRE(warped(0, 0) == Catch::Approx(-1.0f));
  REQUIRE(warped(1, 1) == Catch::Approx(-4.0f));
}

TEST_CASE("common_overlap_tile_gate_keeps_negative_finite_samples") {
  tile_compile::Matrix2Df tile(2, 2);
  tile << -0.5f, 0.0f, 0.0f, 0.0f;

  const tile_compile::Tile bounds{0, 0, 2, 2, 0, 0};
  const std::vector<uint8_t> common_valid_mask = {1u, 1u, 1u, 1u};

  REQUIRE(tile_compile::runner::apply_common_overlap_to_tile_inplace_and_check_nonzero(
      tile, bounds, common_valid_mask, 2, 2));
  REQUIRE(tile_compile::runner::tile_has_nonzero_common_data(tile, 0,
                                                             {1u}));
}

TEST_CASE("common_overlap_tile_gate_marks_canvas_invalid_pixels_nonfinite") {
  tile_compile::Matrix2Df tile(2, 2);
  tile << 1.0f, 2.0f, 3.0f, 4.0f;

  const tile_compile::Tile bounds{0, 0, 2, 2, 0, 0};
  const std::vector<uint8_t> common_valid_mask = {1u, 0u, 1u, 0u};

  REQUIRE(tile_compile::runner::apply_common_overlap_to_tile_inplace_and_check_nonzero(
      tile, bounds, common_valid_mask, 2, 2));
  REQUIRE(std::isfinite(tile(0, 0)));
  REQUIRE_FALSE(std::isfinite(tile(0, 1)));
  REQUIRE(std::isfinite(tile(1, 0)));
  REQUIRE_FALSE(std::isfinite(tile(1, 1)));
}
#else
int tile_compile_tests_acceleration_backend_stub() { return 0; }
#endif
