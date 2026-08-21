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
#include <iostream>
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
  cfg.cherry_pick_mode = "top_k";
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
  cfg.cherry_pick_mode = "top_k";
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

// WP-E: fp16 Q-Maps + bit-packed masks (gpu_half_qmaps/gpu_packed_masks,
// default-on) must reproduce the CPU reference within the same tolerance as
// the full-precision CUDA path, and must reproduce the full-precision CUDA
// path itself (isolates the dequantize-kernel roundtrip from unrelated
// CUDA-vs-CPU drift).
TEST_CASE("aqmh_native_cuda_gpu_half_qmaps_packed_masks_matches_full_precision") {
  const auto dir = unique_recon_cache_dir("aqmh_recon_native_cuda_wpe");
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

  tile_compile::reconstruction::AqmhReconstructionConfig cfg_full = cfg;
  cfg_full.gpu_half_qmaps = false;
  cfg_full.gpu_packed_masks = false;
  const auto gpu_full = tile_compile::reconstruction::reconstruct_aqmh_weighted_cuda(
      frames.size(), loader_for(frames), &cache, global_weights, mask, W, H,
      cfg_full);
  if (!gpu_full.acceleration_used) {
    std::filesystem::remove_all(dir);
    SKIP("Native CUDA device is not available");
  }

  tile_compile::reconstruction::AqmhReconstructionConfig cfg_compressed = cfg;
  REQUIRE(cfg_compressed.gpu_half_qmaps);   // defaults must stay on
  REQUIRE(cfg_compressed.gpu_packed_masks); // defaults must stay on
  const auto gpu_compressed =
      tile_compile::reconstruction::reconstruct_aqmh_weighted_cuda(
          frames.size(), loader_for(frames), &cache, global_weights, mask, W,
          H, cfg_compressed);

  REQUIRE(gpu_compressed.acceleration_used);
  REQUIRE_FALSE(gpu_compressed.acceleration_fallback);
  REQUIRE(gpu_compressed.unsupported_pixels == cpu.unsupported_pixels);
  REQUIRE(gpu_compressed.zero_veto_pixels == cpu.zero_veto_pixels);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      // vs. CPU reference: same tolerance as the full-precision CUDA path.
      REQUIRE(gpu_compressed.output(y, x) ==
              Catch::Approx(cpu.output(y, x)).margin(2.0e-4f));
      // vs. full-precision CUDA: isolates fp16/bit-packing roundtrip error;
      // slightly wider margin to cover fp16 quantization of the Q-Maps.
      REQUIRE(gpu_compressed.output(y, x) ==
              Catch::Approx(gpu_full.output(y, x)).margin(5.0e-3f));
      REQUIRE(gpu_compressed.uniform_control_valid_mask[static_cast<size_t>(
                  y * W + x)] ==
              gpu_full.uniform_control_valid_mask[static_cast<size_t>(
                  y * W + x)]);
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

TEST_CASE("aqmh_raw_baseline_guard_rejects_unrelated_seam_regression") {
  tile_compile::config::AqmhValidationConfig cfg;
  tile_compile::reconstruction::AqmhValidationComparison raw_vs_control;
  raw_vs_control.background_rms_regression = 0.26f;
  raw_vs_control.fwhm_regression = 0.01f;
  raw_vs_control.seam_score_regression = -0.06f;
  raw_vs_control.tail_applicable = true;
  raw_vs_control.elongation_applicable = true;
  raw_vs_control.tail11_abs_regression = -0.04f;
  raw_vs_control.elongation_regression = 0.0f;

  tile_compile::reconstruction::AqmhValidationComparison candidate_vs_control;
  candidate_vs_control.background_rms_regression = 0.02f;
  candidate_vs_control.fwhm_regression = -0.01f;
  candidate_vs_control.seam_score_regression = 0.01f;
  candidate_vs_control.tail_applicable = true;
  candidate_vs_control.elongation_applicable = true;
  candidate_vs_control.tail11_abs_regression = -0.03f;
  candidate_vs_control.elongation_regression = 0.0f;

  tile_compile::reconstruction::AqmhValidationComparison candidate_vs_raw;
  candidate_vs_raw.background_rms_regression = -0.19f;
  candidate_vs_raw.fwhm_regression = -0.02f;
  candidate_vs_raw.seam_score_regression = 0.073f;
  candidate_vs_raw.tail_applicable = true;
  candidate_vs_raw.elongation_applicable = true;
  candidate_vs_raw.tail11_abs_regression = 0.01f;
  candidate_vs_raw.elongation_regression = 0.002f;

  const auto decision =
      tile_compile::reconstruction::aqmh_raw_baseline_guard_decision(
          candidate_vs_raw, raw_vs_control, candidate_vs_control, cfg);

  REQUIRE_FALSE(decision.ok);
  REQUIRE(decision.reason == "candidate_exceeds_raw_baseline_guard");
}
#endif

// ---------------------------------------------------------------------------
// WP-A/R2 session test: session multichannel result matches individual calls.
// Runs without CUDA (session falls back gracefully; test verifies the CPU
// fallback path is consistent).
// ---------------------------------------------------------------------------
TEST_CASE("aqmh_cuda_session_multichannel_matches_individual_calls") {
  constexpr int H = 12, W = 14, N = 4;
  const auto dir = unique_recon_cache_dir("aqmh_session_multichannel");
  std::filesystem::remove_all(dir);
  auto cache = make_cache(dir, W, H);

  tile_compile::VectorXf global_weights(N);
  global_weights << 1.0f, 0.9f, 1.1f, 0.8f;

  std::vector<tile_compile::Matrix2Df> frames_r(N), frames_g(N), frames_b(N);
  for (int fi = 0; fi < N; ++fi) {
    tile_compile::Matrix2Df q(H, W);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        q(y, x) = 0.4f + 0.1f * fi + 0.005f * (x + y);
    cache.write(static_cast<size_t>(fi), q);
    frames_r[fi].resize(H, W);
    frames_g[fi].resize(H, W);
    frames_b[fi].resize(H, W);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        frames_r[fi](y, x) = 0.1f + 0.01f * x + 0.02f * y + 0.1f * fi;
        frames_g[fi](y, x) = 0.2f + 0.01f * x + 0.02f * y + 0.1f * fi;
        frames_b[fi](y, x) = 0.3f + 0.01f * x + 0.02f * y + 0.1f * fi;
      }
  }
  std::vector<uint8_t> canvas_mask(static_cast<size_t>(W * H), 1u);
  std::vector<uint8_t> frame_has_data(N, 1u);

  tile_compile::reconstruction::AqmhReconstructionConfig plane_cfg;
  plane_cfg.compute_uniform_control = false;
  plane_cfg.clip_sigma = 2.0f;
  plane_cfg.min_fraction = 0.5f;

  // Build region loaders
  auto make_region_loader = [&](const std::vector<tile_compile::Matrix2Df>& fs) {
    return [&fs, &frame_has_data, N, W, H](
        size_t fi, int y0, int rows, tile_compile::Matrix2Df& out) -> bool {
      if (fi >= static_cast<size_t>(N) || frame_has_data[fi] == 0u) return false;
      if (y0 < 0 || y0 + rows > H || rows <= 0) return false;
      out = fs[fi].middleRows(y0, rows);
      return out.rows() == rows && out.cols() == W;
    };
  };
  auto r_region = make_region_loader(frames_r);
  auto g_region = make_region_loader(frames_g);
  auto b_region = make_region_loader(frames_b);

  // Individual calls (reference)
  auto make_full_loader = [&](const std::vector<tile_compile::Matrix2Df>& fs) {
    return [&fs, &frame_has_data, N, W, H](size_t fi, tile_compile::Matrix2Df& out) -> bool {
      if (fi >= static_cast<size_t>(N) || frame_has_data[fi] == 0u) return false;
      out = fs[fi]; return out.rows() == H && out.cols() == W;
    };
  };
  const auto ref_r = tile_compile::reconstruction::reconstruct_aqmh_weighted(
      N, make_full_loader(frames_r), &cache, global_weights, canvas_mask, W, H, plane_cfg);
  const auto ref_g = tile_compile::reconstruction::reconstruct_aqmh_weighted(
      N, make_full_loader(frames_g), &cache, global_weights, canvas_mask, W, H, plane_cfg);
  const auto ref_b = tile_compile::reconstruction::reconstruct_aqmh_weighted(
      N, make_full_loader(frames_b), &cache, global_weights, canvas_mask, W, H, plane_cfg);

#if TILE_COMPILE_WITH_CUDA
  tile_compile::reconstruction::AqmhCudaReconstructionSession session;
  const bool sess_ok = session.init(N, &cache, global_weights, canvas_mask, W, H,
                                    plane_cfg, {}, {});
  if (sess_ok) {
    auto results = session.run_planes_rgb({r_region, g_region, b_region},
                                          {false, false, false});
    REQUIRE(results.size() == 3u);
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        REQUIRE(results[0].output(y, x) == Catch::Approx(ref_r.output(y, x)).margin(1e-4f));
        REQUIRE(results[1].output(y, x) == Catch::Approx(ref_g.output(y, x)).margin(1e-4f));
        REQUIRE(results[2].output(y, x) == Catch::Approx(ref_b.output(y, x)).margin(1e-4f));
      }
    }
  }
#endif
  // CPU path: verify individual results are self-consistent
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      REQUIRE(ref_r.output(y, x) != ref_b.output(y, x));  // different channels differ

  std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------
// WP-B two-stage test: session two-stage result matches single-stage.
// GPU-only; skipped when CUDA is not available.
// ---------------------------------------------------------------------------
#if TILE_COMPILE_WITH_CUDA
TEST_CASE("aqmh_cuda_two_stage_matches_single_stage") {
  constexpr int H = 10, W = 12, N = 6;
  const auto dir = unique_recon_cache_dir("aqmh_two_stage");
  std::filesystem::remove_all(dir);
  auto cache = make_cache(dir, W, H);

  tile_compile::VectorXf global_weights(N);
  for (int fi = 0; fi < N; ++fi) global_weights[fi] = 1.0f + 0.1f * fi;

  std::vector<tile_compile::Matrix2Df> frames_a(N), frames_b(N);
  for (int fi = 0; fi < N; ++fi) {
    tile_compile::Matrix2Df q(H, W);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        q(y, x) = 0.5f + 0.05f * fi;
    cache.write(static_cast<size_t>(fi), q);
    frames_a[fi].resize(H, W); frames_b[fi].resize(H, W);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        frames_a[fi](y, x) = 1.0f + 0.01f * x + 0.02f * y + 0.05f * fi;
        frames_b[fi](y, x) = 2.0f + 0.01f * x + 0.02f * y + 0.05f * fi;
      }
  }
  std::vector<uint8_t> canvas(static_cast<size_t>(W * H), 1u);
  tile_compile::reconstruction::AqmhReconstructionConfig cfg;
  cfg.compute_uniform_control = false;

  // Reference: single-plane calls
  auto make_region = [&](const std::vector<tile_compile::Matrix2Df>& fs) {
    return [&fs, N, W, H](size_t fi, int y0, int rows, tile_compile::Matrix2Df& out) -> bool {
      if (fi >= static_cast<size_t>(N)) return false;
      if (y0 < 0 || y0 + rows > H || rows <= 0) return false;
      out = fs[fi].middleRows(y0, rows);
      return true;
    };
  };

  tile_compile::reconstruction::AqmhCudaReconstructionSession single_sess;
  if (!single_sess.init(N, &cache, global_weights, canvas, W, H, cfg, {}, {})) {
    std::filesystem::remove_all(dir); return;
  }
  auto res_a1 = single_sess.run_plane({}, make_region(frames_a), false);
  auto res_b1 = single_sess.run_plane({}, make_region(frames_b), false);
  if (res_a1.acceleration_fallback || res_b1.acceleration_fallback) {
    std::filesystem::remove_all(dir); return;
  }

  // Two-stage: run_planes_rgb with 2 planes
  tile_compile::reconstruction::AqmhCudaReconstructionSession two_sess;
  if (!two_sess.init(N, &cache, global_weights, canvas, W, H, cfg, {}, {})) {
    std::filesystem::remove_all(dir); return;
  }
  auto two_results = two_sess.run_planes_rgb(
      {make_region(frames_a), make_region(frames_b)}, {false, false});
  if (two_results.size() != 2 ||
      two_results[0].acceleration_fallback || two_results[1].acceleration_fallback) {
    std::filesystem::remove_all(dir); return;
  }

  // Results must match within floating-point tolerance
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      REQUIRE(two_results[0].output(y, x) ==
              Catch::Approx(res_a1.output(y, x)).margin(1e-4f));
      REQUIRE(two_results[1].output(y, x) ==
              Catch::Approx(res_b1.output(y, x)).margin(1e-4f));
    }
  }
  std::filesystem::remove_all(dir);
}
#endif
