#if __has_include(<catch2/catch_test_macros.hpp>)
#include "tile_compile/reconstruction/reconstruction.hpp"
#include "tile_compile/reconstruction/tile_boundary_diagnostics.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using tile_compile::Matrix2Df;
using tile_compile::Tile;
using tile_compile::reconstruction::analyze_tile_boundaries;
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

TEST_CASE("tile_boundary_diagnostics_reports_constant_overlap_offset") {
  std::vector<Tile> tiles = {
      Tile{0, 0, 4, 4, 0, 0},
      Tile{2, 0, 4, 4, 0, 1},
  };
  std::vector<Matrix2Df> images(2, Matrix2Df::Zero(4, 4));
  images[1].setConstant(2.0f);
  std::vector<uint8_t> valid = {1u, 1u};

  const auto diagnostics = analyze_tile_boundaries(tiles, images, valid);

  REQUIRE(diagnostics.pair_count == 1);
  REQUIRE(diagnostics.observed_pair_count == 1);
  REQUIRE(diagnostics.sample_count == 8);
  REQUIRE(diagnostics.pair_diagnostics.size() == 1);
  REQUIRE(diagnostics.pair_diagnostics[0].mean_abs_diff ==
          Catch::Approx(2.0f).margin(1e-6));
  REQUIRE(diagnostics.pair_diagnostics[0].mean_signed_diff ==
          Catch::Approx(2.0f).margin(1e-6));
  REQUIRE(diagnostics.pair_diagnostics[0].p95_abs_diff ==
          Catch::Approx(2.0f).margin(1e-6));
}

TEST_CASE("tile_boundary_diagnostics_sorts_worst_pairs_by_mean_abs_diff") {
  std::vector<Tile> tiles = {
      Tile{0, 0, 4, 4, 0, 0},
      Tile{2, 0, 4, 4, 0, 1},
      Tile{4, 0, 4, 4, 0, 2},
  };
  std::vector<Matrix2Df> images(3, Matrix2Df::Zero(4, 4));
  images[1].setConstant(1.0f);
  images[2].setConstant(4.0f);
  std::vector<uint8_t> valid = {1u, 1u, 1u};

  const auto diagnostics = analyze_tile_boundaries(tiles, images, valid);

  REQUIRE(diagnostics.observed_pair_count == 2);
  REQUIRE(diagnostics.pair_diagnostics.size() == 2);
  REQUIRE(diagnostics.pair_diagnostics[0].lhs == 1);
  REQUIRE(diagnostics.pair_diagnostics[0].rhs == 2);
  REQUIRE(diagnostics.pair_diagnostics[0].mean_abs_diff ==
          Catch::Approx(3.0f).margin(1e-6));
  REQUIRE(diagnostics.pair_mean_abs_diff_p95 >=
          diagnostics.pair_mean_abs_diff_mean);
}

TEST_CASE("tile_boundary_diagnostics_skips_invalid_tiles") {
  std::vector<Tile> tiles = {
      Tile{0, 0, 4, 4, 0, 0},
      Tile{2, 0, 4, 4, 0, 1},
  };
  std::vector<Matrix2Df> images(2, Matrix2Df::Zero(4, 4));
  std::vector<uint8_t> valid = {1u, 0u};

  const auto diagnostics = analyze_tile_boundaries(tiles, images, valid);

  REQUIRE(diagnostics.pair_count == 0);
  REQUIRE(diagnostics.observed_pair_count == 0);
  REQUIRE(diagnostics.pair_diagnostics.empty());
}

TEST_CASE("tile_boundary_diagnostics_respects_common_canvas_mask") {
  std::vector<Tile> tiles = {
      Tile{0, 0, 4, 4, 0, 0},
      Tile{2, 0, 4, 4, 0, 1},
  };
  std::vector<Matrix2Df> images(2, Matrix2Df::Zero(4, 4));
  images[1].setConstant(5.0f);
  std::vector<uint8_t> valid = {1u, 1u};
  std::vector<uint8_t> common_mask(24, 0u);
  for (int y = 0; y < 4; ++y) {
    common_mask[static_cast<size_t>(y * 6 + 2)] = 1u;
  }

  const auto diagnostics =
      analyze_tile_boundaries(tiles, images, valid, common_mask, 6, 4);

  REQUIRE(diagnostics.pair_count == 1);
  REQUIRE(diagnostics.observed_pair_count == 1);
  REQUIRE(diagnostics.sample_count == 4);
  REQUIRE(diagnostics.pair_diagnostics.size() == 1);
  REQUIRE(diagnostics.pair_diagnostics[0].mean_abs_diff ==
          Catch::Approx(5.0f).margin(1e-6));
  REQUIRE(diagnostics.pair_diagnostics[0].p95_abs_diff ==
          Catch::Approx(5.0f).margin(1e-6));
}
#else
int tile_compile_tests_reconstruction_regression_stub() { return 0; }
#endif
