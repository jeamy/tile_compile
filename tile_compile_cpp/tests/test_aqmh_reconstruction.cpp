#if __has_include(<catch2/catch_test_macros.hpp>)
#include "tile_compile/metrics/aqmh_quality_map_cache.hpp"
#include "tile_compile/core/acceleration.hpp"
#include "tile_compile/reconstruction/aqmh_reconstruction.hpp"
#include "tile_compile/reconstruction/aqmh_validation.hpp"
#include "../apps/runner_phase_aqmh_reconstruction.hpp"

#if TILE_COMPILE_WITH_CUDA
#include "tile_compile/reconstruction/aqmh_reconstruction_cuda.hpp"
#endif
#include "tile_compile/reconstruction/aqmh_reconstruction_opencl.hpp"

#include <filesystem>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace {

std::filesystem::path unique_recon_cache_dir(const std::string &name) {
  static int counter = 0;
  return std::filesystem::temp_directory_path() /
         ("tile_compile_" + name + "_" + std::to_string(++counter));
}

tile_compile::metrics::QualityMapCache make_cache(
    const std::filesystem::path &dir, int width, int height) {
  std::vector<uint8_t> mask(static_cast<size_t>(width * height), 1u);
  tile_compile::config::AqmhPyramidConfig pyramid;
  tile_compile::config::AqmhStorageConfig storage;
  storage.resolution_divisor = 1;
  storage.dtype = "float32";
  storage.max_resident_maps = 2;
  return tile_compile::metrics::QualityMapCache(
      dir, "luma", width, height, pyramid, storage,
      tile_compile::metrics::compute_aqmh_canvas_mask_hash(mask, width,
                                                           height));
}

tile_compile::reconstruction::AqmhFrameLoader loader_for(
    const std::vector<tile_compile::Matrix2Df> &frames) {
  return [&frames](size_t fi, tile_compile::Matrix2Df &out) {
    if (fi >= frames.size())
      return false;
    out = frames[fi];
    return true;
  };
}

} // namespace

TEST_CASE("aqmh_reconstruction_zero_map_is_explicit_veto") {
  const auto dir = unique_recon_cache_dir("aqmh_recon_zero_veto");
  std::filesystem::remove_all(dir);
  auto cache = make_cache(dir, 3, 3);

  std::vector<tile_compile::Matrix2Df> frames{
      tile_compile::Matrix2Df::Constant(3, 3, 10.0f),
      tile_compile::Matrix2Df::Constant(3, 3, 20.0f)};
  cache.write(0, tile_compile::Matrix2Df::Zero(3, 3));
  cache.write(1, tile_compile::Matrix2Df::Zero(3, 3));
  std::vector<uint8_t> mask(static_cast<size_t>(3 * 3), 1u);
  tile_compile::VectorXf global_weights(2);
  global_weights << 1.0f, 1.0f;

  const auto out = tile_compile::reconstruction::reconstruct_aqmh_weighted(
      frames.size(), loader_for(frames), &cache, global_weights, mask, 3, 3,
      tile_compile::reconstruction::AqmhReconstructionConfig{});

  REQUIRE(out.zero_veto_pixels == 9);
  REQUIRE(out.unsupported_pixels == 9);
  for (int y = 0; y < 3; ++y) {
    for (int x = 0; x < 3; ++x) {
      REQUIRE(out.output(y, x) == Catch::Approx(0.0f).margin(1.0e-6f));
      REQUIRE(out.weight_sum(y, x) == Catch::Approx(0.0f).margin(1.0e-6f));
    }
  }
  std::filesystem::remove_all(dir);
}

TEST_CASE("aqmh_reconstruction_missing_maps_do_not_fallback_to_unweighted_mean") {
  const auto dir = unique_recon_cache_dir("aqmh_recon_missing_maps");
  std::filesystem::remove_all(dir);
  auto cache = make_cache(dir, 2, 2);

  std::vector<tile_compile::Matrix2Df> frames{
      tile_compile::Matrix2Df::Constant(2, 2, 10.0f),
      tile_compile::Matrix2Df::Constant(2, 2, 30.0f)};
  std::vector<uint8_t> mask(static_cast<size_t>(2 * 2), 1u);
  tile_compile::VectorXf global_weights(2);
  global_weights << 1.0f, 1.0f;

  tile_compile::reconstruction::AqmhReconstructionConfig cfg;
  cfg.min_n_eff = 1.0f;
  cfg.clip_iterations = 0;
  const auto out = tile_compile::reconstruction::reconstruct_aqmh_weighted(
      frames.size(), loader_for(frames), &cache, global_weights, mask, 2, 2,
      cfg);

  REQUIRE(out.missing_map_samples == 8);
  REQUIRE(out.unsupported_pixels == 4);
  for (int y = 0; y < 2; ++y) {
    for (int x = 0; x < 2; ++x)
      REQUIRE(out.output(y, x) == Catch::Approx(0.0f).margin(1.0e-6f));
  }
  std::filesystem::remove_all(dir);
}

TEST_CASE("aqmh_reconstruction_uses_per_pixel_quality_weights") {
  const auto dir = unique_recon_cache_dir("aqmh_recon_weights");
  std::filesystem::remove_all(dir);
  auto cache = make_cache(dir, 2, 2);

  std::vector<tile_compile::Matrix2Df> frames{
      tile_compile::Matrix2Df::Constant(2, 2, 10.0f),
      tile_compile::Matrix2Df::Constant(2, 2, 100.0f)};
  tile_compile::Matrix2Df q0 = tile_compile::Matrix2Df::Ones(2, 2);
  tile_compile::Matrix2Df q1 = tile_compile::Matrix2Df::Ones(2, 2);
  q1(0, 0) = 0.0f;
  cache.write(0, q0);
  cache.write(1, q1);
  std::vector<uint8_t> mask(static_cast<size_t>(2 * 2), 1u);
  tile_compile::VectorXf global_weights(2);
  global_weights << 1.0f, 1.0f;

  tile_compile::reconstruction::AqmhReconstructionConfig cfg;
  cfg.min_n_eff = 1.0f;
  cfg.clip_iterations = 0;
  const auto out = tile_compile::reconstruction::reconstruct_aqmh_weighted(
      frames.size(), loader_for(frames), &cache, global_weights, mask, 2, 2,
      cfg);

  REQUIRE(out.output(0, 0) == Catch::Approx(10.0f).margin(1.0e-6f));
  REQUIRE(out.output(1, 1) == Catch::Approx(55.0f).margin(1.0e-6f));
  REQUIRE(out.zero_veto_pixels == 0);
  std::filesystem::remove_all(dir);
}

TEST_CASE("aqmh_cherry_pick_run_gate_forces_small_nominal_selection_off") {
  const auto dir = unique_recon_cache_dir("aqmh_cherry_gate");
  std::filesystem::remove_all(dir);
  auto cache = make_cache(dir, 1, 1);
  std::vector<tile_compile::Matrix2Df> frames;
  for (int i = 0; i < 25; ++i) {
    frames.push_back(tile_compile::Matrix2Df::Constant(1, 1, 10.0f));
    cache.write(static_cast<size_t>(i), tile_compile::Matrix2Df::Ones(1, 1));
  }
  tile_compile::VectorXf weights = tile_compile::VectorXf::Ones(25);
  tile_compile::reconstruction::AqmhReconstructionConfig cfg;
  cfg.cherry_pick = true;
  cfg.cherry_pick_k_frac = 0.30f;
  cfg.cherry_pick_k_min_required = 20;
  const auto out = tile_compile::reconstruction::reconstruct_aqmh_weighted(
      frames.size(), loader_for(frames), &cache, weights, {1u}, 1, 1, cfg);
  REQUIRE(out.cherry_pick_forced_disabled);
  REQUIRE_FALSE(out.cherry_pick_active);
  REQUIRE(out.k_nominal_median == 7.0f);
  REQUIRE(out.output(0, 0) == Catch::Approx(10.0f));
  std::filesystem::remove_all(dir);
}

TEST_CASE("aqmh_cherry_pick_enforces_positive_sample_floor") {
  const auto dir = unique_recon_cache_dir("aqmh_cherry_floor");
  std::filesystem::remove_all(dir);
  auto cache = make_cache(dir, 1, 1);
  std::vector<tile_compile::Matrix2Df> frames;
  tile_compile::VectorXf weights(100);
  for (int i = 0; i < 100; ++i) {
    frames.push_back(tile_compile::Matrix2Df::Constant(1, 1, static_cast<float>(i)));
    cache.write(static_cast<size_t>(i), tile_compile::Matrix2Df::Ones(1, 1));
    weights[i] = 1.0f + static_cast<float>(i) * 0.001f;
  }
  tile_compile::reconstruction::AqmhReconstructionConfig cfg;
  cfg.cherry_pick = true;
  cfg.cherry_pick_k_frac = 0.30f;
  cfg.cherry_pick_k_min_required = 20;
  cfg.clip_iterations = 0;
  const auto out = tile_compile::reconstruction::reconstruct_aqmh_weighted(
      frames.size(), loader_for(frames), &cache, weights, {1u}, 1, 1, cfg);
  REQUIRE_FALSE(out.cherry_pick_forced_disabled);
  REQUIRE(out.cherry_pick_active);
  REQUIRE(out.k_effective_p50 == 30.0f);
  std::filesystem::remove_all(dir);
}

TEST_CASE("aqmh_uniform_control_includes_samples_excluded_by_aqmh_weights") {
  const auto dir = unique_recon_cache_dir("aqmh_uniform_control");
  constexpr int W = 3, H = 2;
  std::vector<tile_compile::Matrix2Df> frames(
      2, tile_compile::Matrix2Df::Constant(H, W, 0.0f));
  frames[0].setConstant(10.0f);
  frames[1].setConstant(20.0f);
  tile_compile::metrics::QualityMapCache cache = make_cache(dir, W, H);
  tile_compile::Matrix2Df q0 = tile_compile::Matrix2Df::Constant(H, W, 0.1f);
  tile_compile::Matrix2Df q1 = tile_compile::Matrix2Df::Constant(H, W, 0.9f);
  q0(0, 0) = 0.0f;
  q1(0, 0) = 0.0f;
  cache.write(0, q0);
  cache.write(1, q1);
  tile_compile::VectorXf global = tile_compile::VectorXf::Ones(2);
  std::vector<uint8_t> mask(static_cast<size_t>(W * H), 1u);
  tile_compile::reconstruction::AqmhReconstructionConfig cfg;
  cfg.uniform_weights = true;
  cfg.compute_uniform_control = true;
  cfg.clip_iterations = 0;
  cfg.min_n_eff = 1.0f;
  const auto zero_q = tile_compile::reconstruction::reconstruct_aqmh_weighted(
      frames.size(), loader_for(frames), &cache, global, mask, W, H, cfg);
  REQUIRE(zero_q.output(0, 0) == 0.0f);
  REQUIRE(zero_q.zero_veto_pixels == 1);
  REQUIRE(zero_q.uniform_control_output(0, 0) == Catch::Approx(15.0f));
  REQUIRE(zero_q.uniform_control_valid_mask[0] == 1u);

  q0.setConstant(0.1f);
  q1.setConstant(0.9f);
  cache.write(0, q0);
  cache.write(1, q1);
  global[1] = 0.0f;
  const auto zero_global = tile_compile::reconstruction::reconstruct_aqmh_weighted(
      frames.size(), loader_for(frames), &cache, global, mask, W, H, cfg);
  REQUIRE(zero_global.output(1, 1) == Catch::Approx(10.0f));
  REQUIRE(zero_global.uniform_control_output(1, 1) == Catch::Approx(15.0f));
  REQUIRE(zero_global.uniform_control_valid_mask[static_cast<size_t>(W + 1)] == 1u);
  std::filesystem::remove_all(dir);
}

TEST_CASE("aqmh_validation_regression_is_zero_for_identical_control") {
  tile_compile::Matrix2Df image(8, 8);
  for (int y = 0; y < image.rows(); ++y)
    for (int x = 0; x < image.cols(); ++x)
      image(y, x) = static_cast<float>(x + 2 * y);
  const auto comparison =
      tile_compile::reconstruction::compare_aqmh_to_uniform_control(image,
                                                                    image);
  REQUIRE(comparison.seam_score_regression == Catch::Approx(0.0f));
  REQUIRE(comparison.fwhm_regression == Catch::Approx(0.0f));
  REQUIRE(comparison.background_rms_regression == Catch::Approx(0.0f));
}

TEST_CASE("aqmh_region_streaming_avoids_full_frame_loads") {
  const auto dir = unique_recon_cache_dir("aqmh_region_streaming");
  std::filesystem::remove_all(dir);
  constexpr int W = 8, H = 6, N = 4;
  auto cache = make_cache(dir, W, H);
  std::vector<tile_compile::Matrix2Df> frames;
  std::vector<uint8_t> mask(static_cast<size_t>(W * H), 1u);
  for (int fi = 0; fi < N; ++fi) {
    frames.push_back(tile_compile::Matrix2Df::Constant(H, W, 10.0f + fi));
    cache.write(static_cast<size_t>(fi),
                tile_compile::Matrix2Df::Constant(H, W, 0.5f), mask);
  }
  tile_compile::VectorXf global = tile_compile::VectorXf::Ones(N);
  int full_frame_loads = 0;
  auto full_loader = [&](size_t fi, tile_compile::Matrix2Df &out) {
    ++full_frame_loads;
    out = frames[fi];
    return true;
  };
  auto full_mask_loader = [&](size_t, std::vector<uint8_t> &out) {
    out = mask;
    return true;
  };
  auto region_loader = [&](size_t fi, int y0, int rows,
                           tile_compile::Matrix2Df &out) {
    out = frames[fi].block(y0, 0, rows, W);
    return true;
  };
  auto mask_region_loader = [&](size_t, int y0, int rows,
                                std::vector<uint8_t> &out) {
    out.assign(mask.begin() + static_cast<long>(y0 * W),
               mask.begin() + static_cast<long>((y0 + rows) * W));
    return true;
  };
  tile_compile::reconstruction::AqmhReconstructionConfig cfg;
  cfg.clip_iterations = 0;
  cfg.min_n_eff = 1.0f;
  cfg.compute_uniform_control = true;
  const auto result = tile_compile::reconstruction::reconstruct_aqmh_weighted(
      N, full_loader, &cache, global, mask, W, H, cfg, full_mask_loader,
      region_loader, mask_region_loader);
  REQUIRE(full_frame_loads == 0);
  REQUIRE(result.region_streaming_used);
  REQUIRE(result.output(2, 3) == Catch::Approx(11.5f));
  REQUIRE(result.uniform_control_output(2, 3) == Catch::Approx(11.5f));
  std::filesystem::remove_all(dir);
}

TEST_CASE("aqmh_cuda_reconstruction_matches_cpu_streaming_reference") {
  constexpr int H = 16;
  constexpr int W = 18;
  const auto dir = unique_recon_cache_dir("aqmh_recon_cuda_reference");
  std::filesystem::remove_all(dir);
  auto cache = make_cache(dir, W, H);

  std::vector<tile_compile::Matrix2Df> frames;
  tile_compile::VectorXf global_weights(4);
  global_weights << 1.0f, 0.8f, 1.2f, 1.0f;
  for (int fi = 0; fi < 4; ++fi) {
    tile_compile::Matrix2Df frame(H, W);
    tile_compile::Matrix2Df q(H, W);
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        frame(y, x) = 10.0f + 0.1f * x + 0.2f * y + fi;
        q(y, x) = 0.3f + 0.15f * fi + 0.001f * (x + y);
      }
    }
    if (fi == 3)
      frame(5, 7) = 500.0f;
    q(2, 3) = 0.0f;
    frames.push_back(std::move(frame));
    cache.write(static_cast<size_t>(fi), q);
  }
  std::vector<uint8_t> mask(static_cast<size_t>(W * H), 1u);
  mask[0] = 0u;
  tile_compile::reconstruction::AqmhReconstructionConfig cfg;
  cfg.clip_sigma = 2.0f;
  cfg.min_fraction = 0.5f;
  cfg.compute_uniform_control = true;

  const auto cpu = tile_compile::reconstruction::reconstruct_aqmh_weighted(
      frames.size(), loader_for(frames), &cache, global_weights, mask, W, H,
      cfg);

  tile_compile::core::AccelerationContext context("opencv_cuda");
  const auto selection = context.selection_for(
      tile_compile::core::AccelerationPhase::aqmh_reconstruction);
  if (selection.selected !=
      tile_compile::core::AccelerationBackend::opencv_cuda) {
    std::filesystem::remove_all(dir);
    return;
  }
  tile_compile::core::AccelerationOps ops(
      context, tile_compile::core::AccelerationPhase::aqmh_reconstruction);
  tile_compile::core::WorkerCudaStreams streams(true, 1);
  const auto gpu = ops.reconstruct_aqmh(
      frames.size(), loader_for(frames), &cache, global_weights, mask, W, H,
      cfg, streams.get(0));

  // The v0.1 CUDA path does not implement v0.2 features (weighted median+MAD,
  // M_f validation, cherry-pick gate). It falls back to the CPU implementation
  // and reports acceleration_fallback=true.
  REQUIRE_FALSE(gpu.acceleration_used);
  REQUIRE(gpu.acceleration_fallback);
  REQUIRE(gpu.unsupported_pixels == cpu.unsupported_pixels);
  REQUIRE(gpu.zero_veto_pixels == cpu.zero_veto_pixels);
  REQUIRE(gpu.finite_map_samples == cpu.finite_map_samples);
  REQUIRE(gpu.missing_map_samples == cpu.missing_map_samples);
  REQUIRE(gpu.cuda_host_prepare_seconds >=
          gpu.cuda_host_chunk_setup_seconds);
  REQUIRE(gpu.cuda_host_frame_read_worker_seconds > 0.0);
  REQUIRE(gpu.cuda_host_q_map_read_worker_seconds > 0.0);
  REQUIRE(gpu.cuda_host_mask_read_worker_seconds >= 0.0);
  REQUIRE(gpu.cuda_host_pack_worker_seconds > 0.0);
  REQUIRE(gpu.cuda_h2d_seconds >= 0.0);
  REQUIRE(gpu.cuda_kernel_seconds >= 0.0);
  REQUIRE(gpu.cuda_d2h_seconds >= 0.0);
  REQUIRE(gpu.cuda_result_commit_seconds >= 0.0);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      REQUIRE(gpu.output(y, x) ==
              Catch::Approx(cpu.output(y, x)).margin(2.0e-4f));
      REQUIRE(gpu.weight_sum(y, x) ==
              Catch::Approx(cpu.weight_sum(y, x)).margin(2.0e-4f));
    }
  }
  std::filesystem::remove_all(dir);
}

TEST_CASE("aqmh_reconstruction_chunking_splits_into_expected_chunks") {
  const auto dir = unique_recon_cache_dir("aqmh_recon_chunking");
  std::filesystem::remove_all(dir);
  constexpr int W = 4, H = 10, N = 3;
  auto cache = make_cache(dir, W, H);
  std::vector<tile_compile::Matrix2Df> frames;
  for (int fi = 0; fi < N; ++fi) {
    frames.push_back(tile_compile::Matrix2Df::Constant(H, W, 10.0f + fi));
    cache.write(static_cast<size_t>(fi),
                tile_compile::Matrix2Df::Constant(H, W, 0.5f));
  }
  std::vector<uint8_t> mask(static_cast<size_t>(W * H), 1u);
  tile_compile::VectorXf global = tile_compile::VectorXf::Ones(N);
  tile_compile::reconstruction::AqmhReconstructionConfig cfg;
  cfg.chunk_rows = 3;
  cfg.clip_iterations = 0;
  cfg.min_n_eff = 1.0f;
  const auto out = tile_compile::reconstruction::reconstruct_aqmh_weighted(
      frames.size(), loader_for(frames), &cache, global, mask, W, H, cfg);
  REQUIRE(out.chunk_rows == 3);
  REQUIRE(out.chunk_count == 4);
  REQUIRE(out.output(0, 0) == Catch::Approx(11.0f));
  std::filesystem::remove_all(dir);
}

#if TILE_COMPILE_WITH_CUDA
TEST_CASE("aqmh_native_cuda_reconstruction_matches_cpu_reference") {
  const auto dir = unique_recon_cache_dir("aqmh_recon_native_cuda");
  std::filesystem::remove_all(dir);
  constexpr int H = 16, W = 18, N = 4;
  auto cache = make_cache(dir, W, H);
  std::vector<tile_compile::Matrix2Df> frames;
  tile_compile::VectorXf global_weights(N);
  global_weights << 1.0f, 0.8f, 1.2f, 1.0f;
  for (int fi = 0; fi < N; ++fi) {
    tile_compile::Matrix2Df frame(H, W);
    tile_compile::Matrix2Df q(H, W);
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        frame(y, x) = 10.0f + 0.1f * x + 0.2f * y + fi;
        q(y, x) = 0.3f + 0.15f * fi + 0.001f * (x + y);
      }
    }
    if (fi == 3)
      frame(5, 7) = 500.0f;
    q(2, 3) = 0.0f;
    frames.push_back(std::move(frame));
    cache.write(static_cast<size_t>(fi), q);
  }
  std::vector<uint8_t> mask(static_cast<size_t>(W * H), 1u);
  mask[0] = 0u;
  tile_compile::reconstruction::AqmhReconstructionConfig cfg;
  cfg.clip_sigma = 2.0f;
  cfg.min_fraction = 0.5f;
  cfg.compute_uniform_control = true;

  const auto cpu = tile_compile::reconstruction::reconstruct_aqmh_weighted(
      frames.size(), loader_for(frames), &cache, global_weights, mask, W, H,
      cfg);
  const auto gpu = tile_compile::reconstruction::reconstruct_aqmh_weighted_cuda(
      frames.size(), loader_for(frames), &cache, global_weights, mask, W, H,
      cfg);

  if (!gpu.acceleration_used) {
    std::filesystem::remove_all(dir);
    SKIP("Native CUDA device is not available");
  }
  REQUIRE(gpu.acceleration_used);
  REQUIRE_FALSE(gpu.acceleration_fallback);
  REQUIRE(gpu.unsupported_pixels == cpu.unsupported_pixels);
  REQUIRE(gpu.zero_veto_pixels == cpu.zero_veto_pixels);
  REQUIRE(gpu.finite_map_samples == cpu.finite_map_samples);
  REQUIRE(gpu.missing_map_samples == cpu.missing_map_samples);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      REQUIRE(gpu.output(y, x) ==
              Catch::Approx(cpu.output(y, x)).margin(2.0e-4f));
      REQUIRE(gpu.weight_sum(y, x) ==
              Catch::Approx(cpu.weight_sum(y, x)).margin(2.0e-4f));
      REQUIRE(gpu.uniform_control_output(y, x) ==
              Catch::Approx(cpu.uniform_control_output(y, x)).margin(2.0e-4f));
      REQUIRE(gpu.uniform_control_valid_mask[static_cast<size_t>(y * W + x)] ==
              cpu.uniform_control_valid_mask[static_cast<size_t>(y * W + x)]);
    }
  }
  std::filesystem::remove_all(dir);
}
#endif

TEST_CASE("aqmh_native_opencl_reconstruction_matches_cpu_reference") {
  const auto dir = unique_recon_cache_dir("aqmh_recon_native_opencl");
  std::filesystem::remove_all(dir);
  constexpr int H = 8, W = 8, N = 3;
  auto cache = make_cache(dir, W, H);
  std::vector<tile_compile::Matrix2Df> frames;
  tile_compile::VectorXf global_weights(N);
  global_weights << 1.0f, 1.0f, 1.0f;
  for (int fi = 0; fi < N; ++fi) {
    tile_compile::Matrix2Df frame(H, W);
    tile_compile::Matrix2Df q(H, W);
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        frame(y, x) = 10.0f + fi + 0.05f * x;
        q(y, x) = 0.5f;
      }
    }
    frames.push_back(std::move(frame));
    cache.write(static_cast<size_t>(fi), q);
  }
  std::vector<uint8_t> mask(static_cast<size_t>(W * H), 1u);
  tile_compile::reconstruction::AqmhReconstructionConfig cfg;
  cfg.clip_iterations = 0;
  cfg.min_n_eff = 1.0f;

  const auto cpu = tile_compile::reconstruction::reconstruct_aqmh_weighted(
      frames.size(), loader_for(frames), &cache, global_weights, mask, W, H,
      cfg);
  const auto gpu = tile_compile::reconstruction::reconstruct_aqmh_weighted_opencl(
      frames.size(), loader_for(frames), &cache, global_weights, mask, W, H,
      cfg);

  if (gpu.acceleration_fallback) {
    std::filesystem::remove_all(dir);
    return;
  }
  REQUIRE(gpu.acceleration_used);
  REQUIRE(gpu.unsupported_pixels == cpu.unsupported_pixels);
  REQUIRE(gpu.zero_veto_pixels == cpu.zero_veto_pixels);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      REQUIRE(gpu.output(y, x) ==
              Catch::Approx(cpu.output(y, x)).margin(2.0e-4f));
    }
  }
  std::filesystem::remove_all(dir);
}

TEST_CASE("aqmh_phase_result_preserves_raw_output_separately") {
  tile_compile::runner::AqmhReconstructionPhaseResult result;
  result.raw_output = tile_compile::Matrix2Df::Constant(2, 2, 1.0f);
  result.output = result.raw_output;
  result.output(0, 0) = 2.0f;

  REQUIRE(result.raw_output(0, 0) == Catch::Approx(1.0f));
  REQUIRE(result.output(0, 0) == Catch::Approx(2.0f));
}
#endif
