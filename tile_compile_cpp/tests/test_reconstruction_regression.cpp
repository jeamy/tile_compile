#if __has_include(<catch2/catch_test_macros.hpp>)
#include "tile_compile/reconstruction/reconstruction.hpp"
#include "tile_compile/reconstruction/local_weight_regularization.hpp"
#include "tile_compile/reconstruction/tile_boundary_diagnostics.hpp"
#include "tile_compile/reconstruction/tile_normalization.hpp"
#include "tile_compile/reconstruction/tile_weight_profile_diagnostics.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using tile_compile::Matrix2Df;
using tile_compile::Tile;
using tile_compile::reconstruction::analyze_tile_boundaries;
using tile_compile::reconstruction::estimate_tile_normalization_stats;
using tile_compile::reconstruction::guard_tile_normalization_stats;
using tile_compile::reconstruction::regularize_local_quality_scores;
using tile_compile::reconstruction::LocalWeightRegularizationConfig;
using tile_compile::reconstruction::analyze_tile_weight_profiles;
using tile_compile::reconstruction::TileNormalizationGuardConfig;
using tile_compile::reconstruction::TileNormalizationStats;
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
  REQUIRE(diagnostics.pair_diagnostics[0].mean_abs_residual ==
          Catch::Approx(0.0f).margin(1e-6));
  REQUIRE(diagnostics.pair_diagnostics[0].p95_abs_residual ==
          Catch::Approx(0.0f).margin(1e-6));
  REQUIRE(diagnostics.pair_diagnostics[0].scale_ratio ==
          Catch::Approx(1.0f).margin(1e-6));
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

TEST_CASE("tile_boundary_diagnostics_separates_offset_from_structural_residual") {
  std::vector<Tile> tiles = {
      Tile{0, 0, 4, 4, 0, 0},
      Tile{2, 0, 4, 4, 0, 1},
  };
  std::vector<Matrix2Df> images(2, Matrix2Df::Zero(4, 4));
  images[1].setConstant(2.0f);
  images[1](0, 0) = 5.0f;
  std::vector<uint8_t> valid = {1u, 1u};

  const auto diagnostics = analyze_tile_boundaries(tiles, images, valid);

  REQUIRE(diagnostics.observed_pair_count == 1);
  REQUIRE(diagnostics.pair_diagnostics[0].mean_abs_diff >
          diagnostics.pair_diagnostics[0].mean_abs_residual);
  REQUIRE(diagnostics.pair_diagnostics[0].mean_signed_diff > 2.0f);
  REQUIRE(diagnostics.pair_diagnostics[0].p95_abs_residual > 0.0f);
  REQUIRE(diagnostics.pair_mean_abs_residual_mean > 0.0f);
}

TEST_CASE("tile_weight_profile_diagnostics_reports_pair_deltas_and_mismatch") {
  std::vector<tile_compile::reconstruction::TileBoundaryPairDiagnostic> pairs = {
      tile_compile::reconstruction::TileBoundaryPairDiagnostic{0, 1, 0, 0.0f,
                                                               0.0f, 0.0f, 0.0f,
                                                               0.0f, 1.0f, true},
  };
  std::vector<std::vector<float>> local_weights = {
      {1.0f, 1.0f},
      {2.0f, 0.0f},
      {3.0f, 1.0f},
      {0.0f, 0.0f},
  };
  std::vector<uint8_t> frame_has_data = {1u, 1u, 1u, 1u};

  const auto diagnostics =
      analyze_tile_weight_profiles(pairs, local_weights, frame_has_data);

  REQUIRE(diagnostics.observed_pair_count == 1);
  REQUIRE(diagnostics.pair_diagnostics.size() == 1);
  const auto &pair = diagnostics.pair_diagnostics.front();
  REQUIRE(pair.usable_frame_count == 4);
  REQUIRE(pair.lhs_active_frame_count == 3);
  REQUIRE(pair.rhs_active_frame_count == 2);
  REQUIRE(pair.shared_active_frame_count == 2);
  REQUIRE(pair.activation_mismatch_count == 1);
  REQUIRE(pair.mean_abs_delta == Catch::Approx(1.0f).margin(1e-6));
  REQUIRE(pair.p95_abs_delta == Catch::Approx(2.0f).margin(1e-6));
  REQUIRE(pair.correlation < 0.5f);
  REQUIRE(diagnostics.pair_activation_mismatch_fraction_mean ==
          Catch::Approx(0.25f).margin(1e-6));
}

TEST_CASE("local_weight_regularization_smooths_neighbor_scores_per_frame") {
  std::vector<Tile> tiles = {
      Tile{0, 0, 16, 16, 0, 0},
      Tile{16, 0, 16, 16, 0, 1},
      Tile{32, 0, 16, 16, 0, 2},
  };
  std::vector<uint8_t> tile_valid = {1u, 1u, 1u};
  std::vector<uint8_t> frame_has_data = {1u, 1u};
  std::vector<std::vector<float>> scores = {
      {3.0f, 0.0f, -3.0f},
      {2.0f, 1.0f, 0.0f},
  };
  LocalWeightRegularizationConfig cfg;
  cfg.enabled = true;
  cfg.lambda = 0.5f;
  cfg.passes = 1;

  const auto summary = regularize_local_quality_scores(
      tiles, tile_valid, frame_has_data, cfg, &scores);

  REQUIRE(summary.tile_edge_count == 2);
  REQUIRE(summary.adjusted_entries == 4);
  REQUIRE(scores[0][0] == Catch::Approx(1.5f).margin(1e-6));
  REQUIRE(scores[0][1] == Catch::Approx(0.0f).margin(1e-6));
  REQUIRE(scores[0][2] == Catch::Approx(-1.5f).margin(1e-6));
  REQUIRE(scores[1][0] == Catch::Approx(1.5f).margin(1e-6));
  REQUIRE(scores[1][1] == Catch::Approx(1.0f).margin(1e-6));
  REQUIRE(scores[1][2] == Catch::Approx(0.5f).margin(1e-6));
}

TEST_CASE("tile_normalization_stats_ignore_masked_zero_pixels") {
  Matrix2Df tile(1, 5);
  tile << 0.0f, 1.0f, 2.0f, 3.0f, 0.0f;

  const auto stats = estimate_tile_normalization_stats(tile);

  REQUIRE(stats.sample_count == 3);
  REQUIRE(stats.total_count == 5);
  REQUIRE(stats.background == Catch::Approx(2.0f).margin(1e-6));
  REQUIRE(stats.scale == Catch::Approx(1.0f).margin(1e-6));
}

TEST_CASE("tile_normalization_guard_clamps_unstable_low_scales") {
  std::vector<TileNormalizationStats> stats = {
      TileNormalizationStats{1.0f, 2.0f, 400, 400},
      TileNormalizationStats{1.1f, 4.0f, 400, 400},
      TileNormalizationStats{0.9f, 0.2f, 400, 400},
      TileNormalizationStats{0.0f, 0.0f, 0, 400},
  };
  std::vector<uint8_t> valid = {1u, 1u, 1u, 1u};
  TileNormalizationGuardConfig cfg;

  const auto summary =
      guard_tile_normalization_stats(&stats, valid, cfg, 1.0e-6f);

  REQUIRE(summary.global_background ==
          Catch::Approx(1.0f).margin(1e-6));
  REQUIRE(summary.global_scale == Catch::Approx(2.0f).margin(1e-6));
  REQUIRE(summary.clamped_low_scale_count == 1);
  REQUIRE(summary.used_global_background_count == 1);
  REQUIRE(summary.used_global_scale_count == 1);
  REQUIRE(stats[2].scale == Catch::Approx(1.0f).margin(1e-6));
  REQUIRE(stats[3].background == Catch::Approx(1.0f).margin(1e-6));
  REQUIRE(stats[3].scale == Catch::Approx(2.0f).margin(1e-6));
}
#else
int tile_compile_tests_reconstruction_regression_stub() { return 0; }
#endif
