#if __has_include(<catch2/catch_test_macros.hpp>)
#include "tile_compile/metrics/aqmh_quality_map_cache.hpp"
#include "tile_compile/core/acceleration.hpp"
#include "tile_compile/reconstruction/reconstruction.hpp"

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

  const auto out = tile_compile::reconstruction::reconstruct_aqmh_weighted(
      frames.size(), loader_for(frames), &cache, global_weights, mask, 2, 2,
      tile_compile::reconstruction::AqmhReconstructionConfig{});

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

  const auto out = tile_compile::reconstruction::reconstruct_aqmh_weighted(
      frames.size(), loader_for(frames), &cache, global_weights, mask, 2, 2,
      tile_compile::reconstruction::AqmhReconstructionConfig{});

  REQUIRE(out.output(0, 0) == Catch::Approx(10.0f).margin(1.0e-6f));
  REQUIRE(out.output(1, 1) == Catch::Approx(55.0f).margin(1.0e-6f));
  REQUIRE(out.zero_veto_pixels == 0);
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
  cfg.sigma_low = 2.0f;
  cfg.sigma_high = 2.0f;
  cfg.min_fraction = 0.5f;

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
    }
  }
  std::filesystem::remove_all(dir);
}
#endif
