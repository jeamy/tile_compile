#if __has_include(<catch2/catch_test_macros.hpp>)
#include "../apps/runner_shared.hpp"

#include <filesystem>
#include <string>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace {

std::filesystem::path unique_rgb_cache_dir(const std::string &name) {
  static int counter = 0;
  return std::filesystem::temp_directory_path() /
         ("tile_compile_rgb_cache_" + name + "_" + std::to_string(++counter));
}

} // namespace

TEST_CASE("disk_cache_frame_store_rgb_store_load_roundtrip") {
  const auto dir = unique_rgb_cache_dir("roundtrip");
  const int rows = 4, cols = 6;
  tile_compile::runner::DiskCacheFrameStoreRGB cache(dir, 2, rows, cols);

  tile_compile::Matrix2Df R(rows, cols), G(rows, cols), B(rows, cols);
  for (int i = 0; i < rows * cols; ++i) {
    R.data()[i] = static_cast<float>(i) * 0.1f;
    G.data()[i] = static_cast<float>(i) * 0.2f;
    B.data()[i] = static_cast<float>(i) * 0.3f;
  }
  cache.store(0, R, G, B);

  REQUIRE(cache.has_data(0));
  REQUIRE_FALSE(cache.has_data(1));

  auto R_back = cache.load_channel(0, 0);
  auto G_back = cache.load_channel(0, 1);
  auto B_back = cache.load_channel(0, 2);
  REQUIRE(R_back.rows() == rows);
  REQUIRE(R_back.cols() == cols);
  for (int i = 0; i < rows * cols; ++i) {
    REQUIRE(R_back.data()[i] == Catch::Approx(R.data()[i]));
    REQUIRE(G_back.data()[i] == Catch::Approx(G.data()[i]));
    REQUIRE(B_back.data()[i] == Catch::Approx(B.data()[i]));
  }
}

TEST_CASE("disk_cache_frame_store_rgb_load_all_three") {
  const auto dir = unique_rgb_cache_dir("loadall");
  const int rows = 2, cols = 3;
  tile_compile::runner::DiskCacheFrameStoreRGB cache(dir, 1, rows, cols);

  tile_compile::Matrix2Df R = tile_compile::Matrix2Df::Constant(rows, cols, 1.0f);
  tile_compile::Matrix2Df G = tile_compile::Matrix2Df::Constant(rows, cols, 2.0f);
  tile_compile::Matrix2Df B = tile_compile::Matrix2Df::Constant(rows, cols, 3.0f);
  cache.store(0, R, G, B);

  auto frame = cache.load(0);
  REQUIRE(frame.R(0, 0) == Catch::Approx(1.0f));
  REQUIRE(frame.G(0, 0) == Catch::Approx(2.0f));
  REQUIRE(frame.B(0, 0) == Catch::Approx(3.0f));
}

TEST_CASE("disk_cache_frame_store_rgb_invalid_channel_returns_empty") {
  const auto dir = unique_rgb_cache_dir("invalid");
  tile_compile::runner::DiskCacheFrameStoreRGB cache(dir, 1, 2, 2);
  tile_compile::Matrix2Df R(2, 2), G(2, 2), B(2, 2);
  cache.store(0, R, G, B);
  auto bad = cache.load_channel(0, 5);
  REQUIRE(bad.size() == 0);
}

TEST_CASE("disk_cache_frame_store_rgb_size_rows_cols") {
  const auto dir = unique_rgb_cache_dir("dims");
  tile_compile::runner::DiskCacheFrameStoreRGB cache(dir, 3, 7, 5);
  REQUIRE(cache.size() == 3);
  REQUIRE(cache.rows() == 7);
  REQUIRE(cache.cols() == 5);
}

TEST_CASE("disk_cache_frame_store_rgb_attach_existing") {
  const auto dir = unique_rgb_cache_dir("attach");
  const int rows = 2, cols = 2;
  {
    tile_compile::runner::DiskCacheFrameStoreRGB cache(dir, 1, rows, cols);
    tile_compile::Matrix2Df R = tile_compile::Matrix2Df::Constant(rows, cols, 10.0f);
    tile_compile::Matrix2Df G = tile_compile::Matrix2Df::Constant(rows, cols, 20.0f);
    tile_compile::Matrix2Df B = tile_compile::Matrix2Df::Constant(rows, cols, 30.0f);
    cache.store(0, R, G, B);
    cache.set_preserve_files(true);
  }
  tile_compile::runner::DiskCacheFrameStoreRGB cache2(dir, 1, rows, cols, true);
  REQUIRE(cache2.has_data(0));
  auto R = cache2.load_channel(0, 0);
  REQUIRE(R(0, 0) == Catch::Approx(10.0f));
}
#endif
