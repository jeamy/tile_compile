#if __has_include(<catch2/catch_test_macros.hpp>)
#include "tile_compile/metrics/aqmh_quality_map_cache.hpp"

#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace {

std::filesystem::path unique_cache_dir(const std::string &name) {
  static int counter = 0;
  return std::filesystem::temp_directory_path() /
         ("tile_compile_" + name + "_" + std::to_string(++counter));
}

tile_compile::Matrix2Df make_q_map(int h, int w, float offset = 0.0f) {
  tile_compile::Matrix2Df m(h, w);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x)
      m(y, x) = std::min(1.0f, offset + 0.01f * static_cast<float>(x + y));
  }
  return m;
}

} // namespace

TEST_CASE("aqmh_quality_map_cache_float32_roundtrip_full_resolution") {
  const auto dir = unique_cache_dir("aqmh_cache_float32");
  std::filesystem::remove_all(dir);
  std::vector<uint8_t> mask(static_cast<size_t>(8 * 8), 1u);

  tile_compile::config::AqmhPyramidConfig pyramid;
  tile_compile::config::AqmhStorageConfig storage;
  storage.resolution_divisor = 1;
  storage.dtype = "float32";

  tile_compile::metrics::QualityMapCache cache(
      dir, "luma", 8, 8, pyramid, storage,
      tile_compile::metrics::compute_aqmh_canvas_mask_hash(mask, 8, 8));
  const auto q = make_q_map(8, 8);
  cache.write(0, q);

  REQUIRE(cache.has(0));
  REQUIRE(std::filesystem::exists(dir / "aqmh_luma_000000.bin"));
  const auto read = cache.read(0);
  REQUIRE(read.rows() == 8);
  REQUIRE(read.cols() == 8);
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x)
      REQUIRE(read(y, x) == Catch::Approx(q(y, x)).margin(1.0e-6f));
  }

  std::filesystem::remove_all(dir);
}

TEST_CASE("aqmh_quality_map_cache_missing_file_returns_empty_matrix") {
  const auto dir = unique_cache_dir("aqmh_cache_missing");
  std::filesystem::remove_all(dir);
  std::vector<uint8_t> mask(static_cast<size_t>(4 * 4), 1u);

  tile_compile::config::AqmhPyramidConfig pyramid;
  tile_compile::config::AqmhStorageConfig storage;
  tile_compile::metrics::QualityMapCache cache(
      dir, "luma", 4, 4, pyramid, storage,
      tile_compile::metrics::compute_aqmh_canvas_mask_hash(mask, 4, 4));

  REQUIRE(cache.read(999).size() == 0);
  REQUIRE_FALSE(cache.has(999));
  std::filesystem::remove_all(dir);
}

TEST_CASE("aqmh_quality_map_cache_uint8_downsampled_readback_is_clamped") {
  const auto dir = unique_cache_dir("aqmh_cache_uint8");
  std::filesystem::remove_all(dir);
  std::vector<uint8_t> mask(static_cast<size_t>(8 * 8), 1u);

  tile_compile::config::AqmhPyramidConfig pyramid;
  tile_compile::config::AqmhStorageConfig storage;
  storage.resolution_divisor = 2;
  storage.dtype = "uint8";

  tile_compile::metrics::QualityMapCache cache(
      dir, "luma", 8, 8, pyramid, storage,
      tile_compile::metrics::compute_aqmh_canvas_mask_hash(mask, 8, 8));
  auto q = make_q_map(8, 8);
  q(0, 0) = -1.0f;
  q(7, 7) = 2.0f;
  cache.write(0, q);

  const auto read = cache.read(0);
  REQUIRE(read.rows() == 8);
  REQUIRE(read.cols() == 8);
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      REQUIRE(read(y, x) >= 0.0f);
      REQUIRE(read(y, x) <= 1.0f);
    }
  }
  REQUIRE(read(7, 7) <= 1.0f);
  REQUIRE(read(0, 0) >= 0.0f);

  std::filesystem::remove_all(dir);
}

TEST_CASE("aqmh_format_v2_preserves_full_resolution_zero_veto") {
  const auto dir = unique_cache_dir("aqmh_cache_veto_v2");
  std::filesystem::remove_all(dir);
  tile_compile::config::AqmhPyramidConfig pyramid;
  tile_compile::config::AqmhStorageConfig storage;
  storage.resolution_divisor = 2;
  tile_compile::metrics::QualityMapCache cache(
      dir, "luma", 4, 4, pyramid, storage,
      tile_compile::metrics::compute_aqmh_canvas_mask_hash(
          std::vector<uint8_t>(16, 1u), 4, 4));
  tile_compile::Matrix2Df map = tile_compile::Matrix2Df::Ones(4, 4);
  map(1, 1) = 0.0f;
  cache.write(0, map, std::vector<uint8_t>(16, 1u));
  const auto decoded = cache.read(0);
  REQUIRE(decoded(1, 1) == 0.0f);
  REQUIRE(decoded(0, 0) > 0.0f);
  std::filesystem::remove_all(dir);
}

TEST_CASE("aqmh_quality_map_cache_uint16_roundtrip_is_low_loss") {
  const auto dir = unique_cache_dir("aqmh_cache_uint16");
  std::filesystem::remove_all(dir);
  std::vector<uint8_t> mask(static_cast<size_t>(8 * 8), 1u);

  tile_compile::config::AqmhPyramidConfig pyramid;
  tile_compile::config::AqmhStorageConfig storage;
  storage.resolution_divisor = 1;
  storage.dtype = "uint16";

  tile_compile::metrics::QualityMapCache cache(
      dir, "luma", 8, 8, pyramid, storage,
      tile_compile::metrics::compute_aqmh_canvas_mask_hash(mask, 8, 8));
  const auto q = make_q_map(8, 8, 0.1234f);
  cache.write(0, q);

  const auto read = cache.read(0);
  REQUIRE(read.rows() == 8);
  REQUIRE(read.cols() == 8);
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      REQUIRE(read(y, x) == Catch::Approx(q(y, x)).margin(8.0e-6f));
    }
  }
  const auto stats = cache.stats();
  REQUIRE(stats.bytes_written == static_cast<uint64_t>(8 * 8 * 2));
  REQUIRE(stats.bytes_read == static_cast<uint64_t>(8 * 8 * 2));

  std::filesystem::remove_all(dir);
}

TEST_CASE("aqmh_quality_map_cache_invalidates_on_map_affecting_config_change") {
  const auto dir = unique_cache_dir("aqmh_cache_invalidate");
  std::filesystem::remove_all(dir);
  std::vector<uint8_t> mask(static_cast<size_t>(6 * 6), 1u);

  tile_compile::config::AqmhPyramidConfig pyramid;
  tile_compile::config::AqmhStorageConfig storage;
  storage.resolution_divisor = 1;
  const std::string mask_hash =
      tile_compile::metrics::compute_aqmh_canvas_mask_hash(mask, 6, 6);
  {
    tile_compile::metrics::QualityMapCache cache(dir, "luma", 6, 6, pyramid,
                                                 storage, mask_hash);
    cache.write(0, make_q_map(6, 6));
    REQUIRE(cache.has(0));
  }

  pyramid.base_window_px += 1;
  tile_compile::metrics::QualityMapCache changed(dir, "luma", 6, 6, pyramid,
                                                 storage, mask_hash);
  REQUIRE_FALSE(changed.has(0));

  std::filesystem::remove_all(dir);
}

TEST_CASE("aqmh_quality_map_cache_invalidates_on_execution_backend_change") {
  const auto dir = unique_cache_dir("aqmh_cache_backend_invalidate");
  std::filesystem::remove_all(dir);
  std::vector<uint8_t> mask(static_cast<size_t>(6 * 6), 1u);
  tile_compile::config::AqmhPyramidConfig pyramid;
  tile_compile::config::AqmhStorageConfig storage;
  const std::string mask_hash =
      tile_compile::metrics::compute_aqmh_canvas_mask_hash(mask, 6, 6);
  {
    tile_compile::metrics::QualityMapCache cache(
        dir, "luma", 6, 6, pyramid, storage, mask_hash, "cpu");
    cache.write(0, make_q_map(6, 6));
    REQUIRE(cache.has(0));
  }

  tile_compile::metrics::QualityMapCache changed(
      dir, "luma", 6, 6, pyramid, storage, mask_hash, "opencv_cuda");
  REQUIRE_FALSE(changed.has(0));
  std::filesystem::remove_all(dir);
}

TEST_CASE("aqmh_quality_map_cache_lru_obeys_max_resident_maps") {
  const auto dir = unique_cache_dir("aqmh_cache_lru");
  std::filesystem::remove_all(dir);
  std::vector<uint8_t> mask(static_cast<size_t>(5 * 5), 1u);

  tile_compile::config::AqmhPyramidConfig pyramid;
  tile_compile::config::AqmhStorageConfig storage;
  storage.resolution_divisor = 1;
  storage.max_resident_maps = 2;
  tile_compile::metrics::QualityMapCache cache(
      dir, "luma", 5, 5, pyramid, storage,
      tile_compile::metrics::compute_aqmh_canvas_mask_hash(mask, 5, 5));

  cache.write(0, make_q_map(5, 5, 0.0f));
  cache.write(1, make_q_map(5, 5, 0.1f));
  cache.write(2, make_q_map(5, 5, 0.2f));

  REQUIRE(cache.read_cached(0).size() > 0);
  REQUIRE(cache.read_cached(1).size() > 0);
  REQUIRE(cache.read_cached(2).size() > 0);
  REQUIRE(cache.read_cached(1).size() > 0);
  const auto stats = cache.stats();
  REQUIRE(stats.max_resident_maps_observed <= 2);
  REQUIRE(stats.cache_hits >= 1);

  std::filesystem::remove_all(dir);
}

TEST_CASE("aqmh_quality_map_cache_supports_parallel_distinct_writes") {
  const auto dir = unique_cache_dir("aqmh_cache_parallel_writes");
  std::filesystem::remove_all(dir);
  constexpr int frame_count = 8;
  std::vector<uint8_t> mask(static_cast<size_t>(16 * 12), 1u);

  tile_compile::config::AqmhPyramidConfig pyramid;
  tile_compile::config::AqmhStorageConfig storage;
  storage.resolution_divisor = 2;
  storage.dtype = "uint16";
  tile_compile::metrics::QualityMapCache cache(
      dir, "luma", 16, 12, pyramid, storage,
      tile_compile::metrics::compute_aqmh_canvas_mask_hash(mask, 16, 12));

  std::vector<std::thread> writers;
  writers.reserve(frame_count);
  for (int fi = 0; fi < frame_count; ++fi) {
    writers.emplace_back([&, fi]() {
      cache.write(static_cast<size_t>(fi),
                  make_q_map(12, 16, 0.05f * static_cast<float>(fi)));
    });
  }
  for (auto &writer : writers)
    writer.join();

  for (int fi = 0; fi < frame_count; ++fi) {
    REQUIRE(cache.has(static_cast<size_t>(fi)));
    REQUIRE(cache.read(static_cast<size_t>(fi)).size() == 16 * 12);
  }
  const auto stats = cache.stats();
  REQUIRE(stats.write_count == frame_count);
  REQUIRE(stats.read_count == frame_count);
  REQUIRE(stats.bytes_written ==
          static_cast<uint64_t>(frame_count * 8 * 6 * sizeof(uint16_t)));

  std::filesystem::remove_all(dir);
}
#endif
