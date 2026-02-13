#if __has_include(<catch2/catch_test_macros.hpp>)
#include "tile_compile/reconstruction/reconstruction.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using tile_compile::Matrix2Df;
using tile_compile::reconstruction::sigma_clip_weighted_tile_with_fallback;

TEST_CASE("tile_weighted_path_uses_all_frames_without_preselection") {
  std::vector<Matrix2Df> tiles(3, Matrix2Df::Zero(1, 1));
  tiles[0](0, 0) = 10.0f;
  tiles[1](0, 0) = 20.0f;
  tiles[2](0, 0) = 40.0f;

  std::vector<float> weights = {1.0f, 1.0f, 1.0f};
  auto out = sigma_clip_weighted_tile_with_fallback(
      tiles, weights, 100.0f, 100.0f, 1, 1.0f, 1e-6f);

  REQUIRE_FALSE(out.fallback_used);
  REQUIRE(out.tile.rows() == 1);
  REQUIRE(out.tile.cols() == 1);
  REQUIRE(out.tile(0, 0) == Catch::Approx((10.0f + 20.0f + 40.0f) / 3.0f));
}

TEST_CASE("tile_weighted_path_falls_back_for_low_weight_tiles") {
  std::vector<Matrix2Df> tiles(2, Matrix2Df::Zero(1, 1));
  tiles[0](0, 0) = 10.0f;
  tiles[1](0, 0) = 30.0f;

  std::vector<float> weights = {0.0f, 0.0f};
  auto out = sigma_clip_weighted_tile_with_fallback(
      tiles, weights, 100.0f, 100.0f, 1, 1.0f, 1e-6f);

  REQUIRE(out.fallback_used);
  REQUIRE(out.effective_weight_sum == Catch::Approx(2.0f));
  REQUIRE(out.tile(0, 0) == Catch::Approx(20.0f));
}

TEST_CASE("tile_weighted_path_is_deterministic") {
  std::vector<Matrix2Df> tiles(3, Matrix2Df::Zero(2, 2));
  tiles[0] << 1.0f, 2.0f, 3.0f, 4.0f;
  tiles[1] << 2.0f, 3.0f, 4.0f, 5.0f;
  tiles[2] << 3.0f, 4.0f, 5.0f, 6.0f;
  std::vector<float> weights = {0.8f, 0.6f, 0.4f};

  auto out_a = sigma_clip_weighted_tile_with_fallback(
      tiles, weights, 3.0f, 3.0f, 3, 0.5f, 1e-6f);
  auto out_b = sigma_clip_weighted_tile_with_fallback(
      tiles, weights, 3.0f, 3.0f, 3, 0.5f, 1e-6f);

  REQUIRE(out_a.fallback_used == out_b.fallback_used);
  REQUIRE(out_a.tile.rows() == out_b.tile.rows());
  REQUIRE(out_a.tile.cols() == out_b.tile.cols());
  for (int y = 0; y < out_a.tile.rows(); ++y) {
    for (int x = 0; x < out_a.tile.cols(); ++x) {
      REQUIRE(out_a.tile(y, x) == Catch::Approx(out_b.tile(y, x)).margin(1e-12));
    }
  }
}
#else
int tile_compile_tests_reconstruction_regression_stub() { return 0; }
#endif
