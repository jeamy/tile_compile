#if __has_include(<catch2/catch_test_macros.hpp>)
#include "../apps/runner_shared.hpp"
#include "tile_compile/image/normalization.hpp"
#include "tile_compile/reconstruction/reconstruction.hpp"
#include "tile_compile/reconstruction/local_weight_regularization.hpp"
#include "tile_compile/reconstruction/tile_boundary_diagnostics.hpp"
#include "tile_compile/reconstruction/tile_normalization.hpp"
#include "tile_compile/reconstruction/tile_weight_profile_diagnostics.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <limits>

using tile_compile::Matrix2Df;
using tile_compile::Tile;
using tile_compile::image::NormalizationScales;
using tile_compile::image::apply_normalization_inplace;
using tile_compile::image::apply_output_scaling_inplace;
using tile_compile::reconstruction::analyze_tile_boundaries;
using tile_compile::reconstruction::estimate_tile_normalization_stats;
using tile_compile::reconstruction::guard_tile_normalization_stats;
using tile_compile::reconstruction::regularize_local_quality_scores;
using tile_compile::reconstruction::LocalWeightRegularizationConfig;
using tile_compile::reconstruction::analyze_tile_weight_profiles;
using tile_compile::reconstruction::TileNormalizationGuardConfig;
using tile_compile::reconstruction::TileNormalizationStats;
using tile_compile::reconstruction::make_partition_window_1d;
using tile_compile::reconstruction::chroma_denoise_rgb_inplace;
using tile_compile::reconstruction::reconstruct_tiles;
using tile_compile::reconstruction::sigma_clip_weighted_tile_with_fallback;
using tile_compile::reconstruction::wiener_tile_filter;

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

TEST_CASE("tile_weighted_path_keeps_finite_negative_samples") {
  std::vector<Matrix2Df> tiles(2, Matrix2Df::Zero(1, 1));
  tiles[0](0, 0) = -1.0f;
  tiles[1](0, 0) = 1.0f;
  std::vector<float> weights = {1.0f, 1.0f};

  auto out = sigma_clip_weighted_tile_with_fallback(
      tiles, weights, 3.0f, 3.0f, 0, 1.0f, 1e-6f);

  REQUIRE_FALSE(out.fallback_used);
  REQUIRE(out.tile(0, 0) == Catch::Approx(0.0f).margin(1e-6));
}

TEST_CASE("tile_weighted_path_marks_empty_pixels_nonfinite") {
  std::vector<Matrix2Df> tiles(2, Matrix2Df::Zero(1, 2));
  const float invalid = std::numeric_limits<float>::quiet_NaN();
  tiles[0] << 1.0f, invalid;
  tiles[1] << 3.0f, invalid;
  std::vector<float> weights = {1.0f, 1.0f};

  auto out = sigma_clip_weighted_tile_with_fallback(
      tiles, weights, 3.0f, 3.0f, 1, 1.0f, 1e-6f);

  REQUIRE(std::isfinite(out.tile(0, 0)));
  REQUIRE_FALSE(std::isfinite(out.tile(0, 1)));
}

TEST_CASE("synthetic_tile_weighting_seam_guard_falls_back_to_global") {
  const auto decision = tile_compile::runner::decide_synthetic_weighting(
      "tile_weighted", 388, 0.60f, 0.091f, 0.55f, 8.11f, -0.06f);

  REQUIRE(decision.tile_seam_guard_triggered);
  REQUIRE(decision.effective_weighting == "global");
}

TEST_CASE("synthetic_tile_weighting_seam_guard_keeps_tile_weighted_when_stable") {
  const auto decision = tile_compile::runner::decide_synthetic_weighting(
      "tile_weighted", 388, 0.003f, 0.015f, 0.01f, 0.8f, 0.45f);

  REQUIRE_FALSE(decision.tile_seam_guard_triggered);
  REQUIRE(decision.effective_weighting == "tile_weighted");
}

TEST_CASE("synthetic_tile_weighting_seam_guard_keeps_old_m66_like_run") {
  const auto decision = tile_compile::runner::decide_synthetic_weighting(
      "tile_weighted", 809, 0.01061374f, 3.3765922f, 0.0088191f, 3.8725216f,
      -0.1081624f);

  REQUIRE_FALSE(decision.tile_seam_guard_triggered);
  REQUIRE(decision.effective_weighting == "tile_weighted");
}

TEST_CASE("synthetic_tile_weighting_seam_guard_rejects_new_m66_like_regression") {
  const auto decision = tile_compile::runner::decide_synthetic_weighting(
      "tile_weighted", 622, 1.3105514f, 4.5448027f, 1.4548731f, 9.9220877f,
      0.0f);

  REQUIRE(decision.tile_seam_guard_triggered);
  REQUIRE(decision.effective_weighting == "global");
}

TEST_CASE("partition_window_forms_unity_in_overlap") {
  const auto lhs = make_partition_window_1d(4, 0, 2);
  const auto rhs = make_partition_window_1d(4, 2, 0);

  REQUIRE(lhs.size() == 4);
  REQUIRE(rhs.size() == 4);
  REQUIRE(lhs[2] + rhs[0] == Catch::Approx(1.0f).margin(1e-6));
  REQUIRE(lhs[3] + rhs[1] == Catch::Approx(1.0f).margin(1e-6));
  REQUIRE(lhs[0] == Catch::Approx(1.0f).margin(1e-6));
  REQUIRE(rhs[3] == Catch::Approx(1.0f).margin(1e-6));
}

TEST_CASE("reconstruct_tiles_preserves_outer_boundary_support_with_partition_windows") {
  Matrix2Df frame(1, 6);
  frame << 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f;

  tile_compile::TileGrid grid;
  grid.tile_size = 4;
  grid.overlap_fraction = 0.5f;
  grid.rows = 1;
  grid.cols = 2;
  grid.tiles = {
      Tile{0, 0, 4, 1, 0, 0},
      Tile{2, 0, 4, 1, 0, 1},
  };

  const std::vector<Matrix2Df> frames{frame};
  const std::vector<std::vector<float>> tile_weights{{1.0f, 1.0f}};

  const auto out = reconstruct_tiles(frames, grid, tile_weights);

  REQUIRE(out.rows() == 1);
  REQUIRE(out.cols() == 6);
  REQUIRE(out(0, 0) == Catch::Approx(10.0f).margin(1e-6));
  REQUIRE(out(0, 5) == Catch::Approx(60.0f).margin(1e-6));
  REQUIRE(out(0, 2) == Catch::Approx(30.0f).margin(1e-6));
  REQUIRE(out(0, 3) == Catch::Approx(40.0f).margin(1e-6));
}

TEST_CASE("normalization_roundtrip_preserves_affine_scale") {
  Matrix2Df img(1, 3);
  img << 10.0f, 12.0f, 14.0f;

  NormalizationScales s;
  s.background_mono = 10.0f;
  s.scale_mono = 0.5f;

  apply_normalization_inplace(img, s, tile_compile::ColorMode::MONO, "", 0, 0);
  REQUIRE(img(0, 0) == Catch::Approx(0.0f).margin(1e-6));
  REQUIRE(img(0, 1) == Catch::Approx(1.0f).margin(1e-6));
  REQUIRE(img(0, 2) == Catch::Approx(2.0f).margin(1e-6));

  apply_output_scaling_inplace(img, 0, 0, tile_compile::ColorMode::MONO, "",
                               2.0f, 1.0f, 1.0f, 1.0f, 10.0f, 0.0f, 0.0f,
                               0.0f, 0.0f);
  REQUIRE(img(0, 0) == Catch::Approx(10.0f).margin(1e-6));
  REQUIRE(img(0, 1) == Catch::Approx(12.0f).margin(1e-6));
  REQUIRE(img(0, 2) == Catch::Approx(14.0f).margin(1e-6));
}

TEST_CASE("output_scaling_ignores_unstable_near_zero_restore_scale") {
  Matrix2Df img(1, 2);
  img << 1.5f, -0.5f;

  apply_output_scaling_inplace(img, 0, 0, tile_compile::ColorMode::MONO, "",
                               1.0e-7f, 1.0f, 1.0f, 1.0f, 2.0f, 0.0f, 0.0f,
                               0.0f, 0.5f);

  REQUIRE(img(0, 0) == Catch::Approx(4.0f).margin(1e-6));
  REQUIRE(img(0, 1) == Catch::Approx(2.0f).margin(1e-6));
}

TEST_CASE("canvas_mask_zeroes_rgb_channels_consistently") {
  Matrix2Df R(2, 2);
  Matrix2Df G(2, 2);
  Matrix2Df B(2, 2);
  R << 10.0f, 20.0f, 30.0f, 40.0f;
  G << 11.0f, 21.0f, 31.0f, 41.0f;
  B << 12.0f, 22.0f, 32.0f, 42.0f;

  const std::vector<uint8_t> mask = {1u, 0u, 1u, 0u};
  tile_compile::image::enforce_canvas_mask_on_rgb(R, G, B, mask);

  REQUIRE(R(0, 0) == Catch::Approx(10.0f).margin(1e-6));
  REQUIRE(G(0, 0) == Catch::Approx(11.0f).margin(1e-6));
  REQUIRE(B(0, 0) == Catch::Approx(12.0f).margin(1e-6));
  REQUIRE(R(0, 1) == Catch::Approx(0.0f).margin(1e-6));
  REQUIRE(G(0, 1) == Catch::Approx(0.0f).margin(1e-6));
  REQUIRE(B(0, 1) == Catch::Approx(0.0f).margin(1e-6));
  REQUIRE(R(1, 1) == Catch::Approx(0.0f).margin(1e-6));
  REQUIRE(G(1, 1) == Catch::Approx(0.0f).margin(1e-6));
  REQUIRE(B(1, 1) == Catch::Approx(0.0f).margin(1e-6));
}

TEST_CASE("wiener_tile_filter_preserves_high_snr_tiles_and_returns_finite_output") {
  Matrix2Df tile = Matrix2Df::Constant(8, 8, 10.0f);
  tile(3, 3) = 30.0f;
  tile(3, 4) = 24.0f;
  tile(4, 3) = 26.0f;
  tile(4, 4) = 22.0f;

  tile_compile::config::WienerDenoiseConfig cfg;
  cfg.enabled = true;
  cfg.snr_threshold = 5.0f;
  cfg.min_snr = 2.0f;
  cfg.q_min = -0.5f;
  cfg.q_max = 1.0f;
  cfg.q_step = 0.25f;
  cfg.max_iterations = 3;

  const auto unchanged =
      wiener_tile_filter(tile, 0.5f, 8.0f, 1.5f, false, cfg);
  const auto filtered =
      wiener_tile_filter(tile, 1.5f, 0.5f, -0.5f, false, cfg);

  REQUIRE((unchanged - tile).cwiseAbs().maxCoeff() ==
          Catch::Approx(0.0f).margin(1e-6));
  REQUIRE(filtered.rows() == tile.rows());
  REQUIRE(filtered.cols() == tile.cols());
  REQUIRE(filtered.array().isFinite().all());
}

TEST_CASE("chroma_denoise_supports_opponent_linear_color_space") {
  Matrix2Df r_ycbcr = Matrix2Df::Zero(3, 3);
  Matrix2Df g_ycbcr = Matrix2Df::Zero(3, 3);
  Matrix2Df b_ycbcr = Matrix2Df::Zero(3, 3);
  r_ycbcr(1, 1) = 1.0f;
  g_ycbcr(1, 2) = 0.5f;
  b_ycbcr(0, 1) = 0.75f;

  Matrix2Df r_opponent = r_ycbcr;
  Matrix2Df g_opponent = g_ycbcr;
  Matrix2Df b_opponent = b_ycbcr;

  tile_compile::config::ChromaDenoiseConfig cfg_ycbcr;
  cfg_ycbcr.enabled = true;
  cfg_ycbcr.color_space = "ycbcr_linear";
  cfg_ycbcr.protect_luma = false;
  cfg_ycbcr.blend.amount = 1.0f;
  cfg_ycbcr.chroma_bilateral.enabled = false;
  cfg_ycbcr.chroma_wavelet.enabled = true;
  cfg_ycbcr.chroma_wavelet.levels = 1;

  auto cfg_opponent = cfg_ycbcr;
  cfg_opponent.color_space = "opponent_linear";

  chroma_denoise_rgb_inplace(r_ycbcr, g_ycbcr, b_ycbcr, cfg_ycbcr);
  chroma_denoise_rgb_inplace(r_opponent, g_opponent, b_opponent, cfg_opponent);

  REQUIRE(r_opponent.rows() == 3);
  REQUIRE(g_opponent.cols() == 3);
  REQUIRE(std::isfinite(r_opponent(1, 1)));
  REQUIRE(std::isfinite(g_opponent(1, 2)));
  REQUIRE(std::isfinite(b_opponent(0, 1)));
  REQUIRE((r_ycbcr - r_opponent).cwiseAbs().maxCoeff() > 1.0e-6f);
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
