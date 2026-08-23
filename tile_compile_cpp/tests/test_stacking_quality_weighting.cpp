#if __has_include(<catch2/catch_test_macros.hpp>)
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/errors.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>

// bge.enabled was a legacy on/off mirror of bge.method ("none" == disabled)
// that could silently disagree with method -- whichever was written most
// recently by a given caller won, so e.g. enabled:false next to a stale
// method:classic still ran BGE. The field was removed; bge.method alone is
// now the sole on/off switch, and any config still setting bge.enabled must
// fail loudly with an actionable message instead of being reinterpreted.
TEST_CASE("bge_method_is_sole_on_off_switch") {
  YAML::Node method_none = YAML::Load(R"(
bge:
  method: none
)");
  auto none_cfg = tile_compile::config::Config::from_yaml(method_none);
  REQUIRE(none_cfg.bge.method == "none");
  REQUIRE_NOTHROW(none_cfg.validate());

  YAML::Node method_autobge = YAML::Load(R"(
bge:
  method: autobge
  autobge:
    random_seed: 123
    stretch_mode: linear
)");
  auto autobge_cfg = tile_compile::config::Config::from_yaml(method_autobge);
  REQUIRE(autobge_cfg.bge.method == "autobge");
  REQUIRE(autobge_cfg.bge.autobge.random_seed == 123);
  REQUIRE(autobge_cfg.bge.autobge.stretch_mode == "linear");
  REQUIRE_NOTHROW(autobge_cfg.validate());

  // Default (no bge block at all) stays disabled.
  auto default_cfg = tile_compile::config::Config::from_yaml(YAML::Load("{}"));
  REQUIRE(default_cfg.bge.method == "none");
}

TEST_CASE("bge_enabled_legacy_field_is_rejected") {
  YAML::Node legacy_enabled_true = YAML::Load(R"(
bge:
  enabled: true
)");
  REQUIRE_THROWS_AS(
      tile_compile::config::Config::from_yaml(legacy_enabled_true),
      tile_compile::ValidationError);

  YAML::Node legacy_enabled_with_method = YAML::Load(R"(
bge:
  enabled: false
  method: classic
)");
  REQUIRE_THROWS_AS(
      tile_compile::config::Config::from_yaml(legacy_enabled_with_method),
      tile_compile::ValidationError);
}

TEST_CASE("bge_autobge_parameters_validate") {
  YAML::Node invalid_patch_size = YAML::Load(R"(
bge:
  method: autobge
  autobge:
    patch_size: 14
)");
  auto patch_cfg =
      tile_compile::config::Config::from_yaml(invalid_patch_size);
  REQUIRE_THROWS(patch_cfg.validate());

  YAML::Node invalid_method = YAML::Load(R"(
bge:
  method: experimental
)");
  auto method_cfg = tile_compile::config::Config::from_yaml(invalid_method);
  REQUIRE_THROWS(method_cfg.validate());
}

TEST_CASE("bge_autobge_user_points_accept_sequence_and_xy_map") {
  YAML::Node node = YAML::Load(R"(
bge:
  method: autobge
  autobge:
    user_sample_points:
      - [0.25, 0.75]
      - {x: 0.5, y: 0.125}
    exclusion_polygons:
      -
        - [0.1, 0.2]
        - {x: 0.3, y: 0.4}
        - [0.5, 0.6]
)");
  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE(cfg.bge.autobge.user_sample_points.size() == 2);
  REQUIRE(cfg.bge.autobge.exclusion_polygons.size() == 1);
  REQUIRE(cfg.bge.autobge.exclusion_polygons.front().size() == 3);
  REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("stacking_cluster_quality_weighting_parses_and_validates") {
  YAML::Node node = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: OSC
  linear_required: true
stacking:
  method: average
  cluster_quality_weighting:
    enabled: true
    kappa_cluster: 1.7
    cap_enabled: true
    cap_ratio: 15.0
)");

  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE(cfg.stacking.cluster_quality_weighting.enabled);
  REQUIRE(std::fabs(cfg.stacking.cluster_quality_weighting.kappa_cluster - 1.7f) <
          1e-6f);
  REQUIRE(cfg.stacking.cluster_quality_weighting.cap_enabled);
  REQUIRE(std::fabs(cfg.stacking.cluster_quality_weighting.cap_ratio - 15.0f) <
          1e-6f);
  REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("stacking_common_overlap_thresholds_parse_and_validate") {
  YAML::Node node = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: OSC
  linear_required: true
stacking:
  method: average
  common_overlap_required_fraction: 1.0
  tile_common_valid_min_fraction: 1.0
)");

  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE(std::fabs(cfg.stacking.common_overlap_required_fraction - 1.0f) <
          1e-6f);
  REQUIRE(std::fabs(cfg.stacking.tile_common_valid_min_fraction - 1.0f) <
          1e-6f);
  REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("stacking_common_overlap_thresholds_reject_out_of_range_values") {
  YAML::Node node = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: OSC
  linear_required: true
stacking:
  method: average
  common_overlap_required_fraction: 0.0
  tile_common_valid_min_fraction: 1.1
)");

  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE_THROWS(cfg.validate());
}

TEST_CASE("stacking_cluster_quality_weighting_rejects_non_positive_kappa") {
  YAML::Node node = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: OSC
  linear_required: true
stacking:
  method: average
  cluster_quality_weighting:
    enabled: true
    kappa_cluster: 0.0
    cap_enabled: false
    cap_ratio: 10.0
)");

  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE_THROWS(cfg.validate());
}

TEST_CASE("stacking_cluster_quality_weighting_rejects_non_positive_cap_ratio") {
  YAML::Node node = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: OSC
  linear_required: true
stacking:
  method: average
  cluster_quality_weighting:
    enabled: true
    kappa_cluster: 1.0
    cap_enabled: true
    cap_ratio: 0.0
)");

  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE_THROWS(cfg.validate());
}

TEST_CASE("local_metrics_spatial_regularization_parses_and_validates") {
  YAML::Node node = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: OSC
  linear_required: true
local_metrics:
  spatial_regularization:
    enabled: true
    lambda: 0.35
    passes: 2
    tau_local: 1.25
)");

  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE(cfg.local_metrics.spatial_regularization.enabled);
  REQUIRE(std::fabs(cfg.local_metrics.spatial_regularization.lambda - 0.35f) <
          1e-6f);
  REQUIRE(cfg.local_metrics.spatial_regularization.passes == 2);
  REQUIRE(std::fabs(cfg.local_metrics.spatial_regularization.tau_local - 1.25f) <
          1e-6f);
  REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("local_metrics_spatial_regularization_rejects_invalid_lambda") {
  YAML::Node node = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: OSC
  linear_required: true
local_metrics:
  spatial_regularization:
    enabled: true
    lambda: 1.5
    passes: 1
)");

  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE_THROWS(cfg.validate());
}

TEST_CASE("local_metrics_neighborhood_normalization_parses_and_validates") {
  YAML::Node node = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: OSC
  linear_required: true
local_metrics:
  neighborhood_normalization:
    enabled: true
    radius: 1
    blend: 0.5
)");

  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE(cfg.local_metrics.neighborhood_normalization.enabled);
  REQUIRE(cfg.local_metrics.neighborhood_normalization.radius == 1);
  REQUIRE(std::fabs(cfg.local_metrics.neighborhood_normalization.blend - 0.5f) <
          1e-6f);
  REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("local_metrics_neighborhood_normalization_rejects_invalid_blend") {
  YAML::Node node = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: OSC
  linear_required: true
local_metrics:
  neighborhood_normalization:
    enabled: true
    radius: 1
    blend: 1.5
)");

  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE_THROWS(cfg.validate());
}

TEST_CASE("tile_soft_star_count_parses_and_validates") {
  YAML::Node node = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: OSC
  linear_required: true
tile:
  star_min_count: 10
  star_soft_count: 14
)");

  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE(cfg.tile.star_min_count == 10);
  REQUIRE(cfg.tile.star_soft_count == 14);
  REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("normalization_mode_and_per_channel_parse_and_validate") {
  YAML::Node node = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: OSC
  linear_required: true
normalization:
  enabled: true
  mode: median
  per_channel: false
)");

  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE(cfg.normalization.enabled);
  REQUIRE(cfg.normalization.mode == "median");
  REQUIRE_FALSE(cfg.normalization.per_channel);
  REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("pcc_background_neutralization_mode_parses_and_validates") {
  YAML::Node node = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: OSC
  linear_required: true
pcc:
  enabled: true
  background_neutralization_mode: off
)");

  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE(cfg.pcc.enabled);
  REQUIRE(cfg.pcc.background_neutralization_mode == "off");
  REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("pcc_background_neutralization_mode_rejects_invalid_value") {
  YAML::Node node = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: OSC
  linear_required: true
pcc:
  enabled: true
  background_neutralization_mode: maybe
)");

  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE_THROWS(cfg.validate());
}

TEST_CASE("calibration_block_parses_and_validates") {
  YAML::Node node = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: OSC
  linear_required: true
calibration:
  use_bias: true
  use_dark: true
  use_flat: true
  bias_use_master: true
  dark_use_master: false
  dark_already_bias_corrected: true
  flat_use_master: true
  dark_auto_select: true
  dark_match_exposure_tolerance_percent: 7.5
  dark_match_use_temp: true
  dark_match_temp_tolerance_c: 1.5
  bias_master: calib/master_bias.fit
  darks_dir: calib/darks
  flat_master: calib/master_flat.fit
  pattern: "*.fit;*.fits"
)");

  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE(cfg.calibration.use_bias);
  REQUIRE(cfg.calibration.use_dark);
  REQUIRE(cfg.calibration.use_flat);
  REQUIRE(cfg.calibration.bias_use_master);
  REQUIRE_FALSE(cfg.calibration.dark_use_master);
  REQUIRE(cfg.calibration.dark_already_bias_corrected);
  REQUIRE(cfg.calibration.flat_use_master);
  REQUIRE(cfg.calibration.dark_auto_select);
  REQUIRE(std::fabs(
              cfg.calibration.dark_match_exposure_tolerance_percent - 7.5f) <
          1e-6f);
  REQUIRE(cfg.calibration.dark_match_use_temp);
  REQUIRE(std::fabs(cfg.calibration.dark_match_temp_tolerance_c - 1.5f) <
          1e-6f);
  REQUIRE(cfg.calibration.bias_master == "calib/master_bias.fit");
  REQUIRE(cfg.calibration.darks_dir == "calib/darks");
  REQUIRE(cfg.calibration.flat_master == "calib/master_flat.fit");
  REQUIRE(cfg.calibration.pattern == "*.fit;*.fits");
  REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("calibration_rejects_missing_sources_and_negative_tolerances") {
  YAML::Node missing_source = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: OSC
  linear_required: true
calibration:
  use_dark: true
)");

  auto cfg_missing = tile_compile::config::Config::from_yaml(missing_source);
  REQUIRE_THROWS(cfg_missing.validate());

  YAML::Node negative_tolerance = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: OSC
  linear_required: true
calibration:
  use_dark: true
  dark_master: calib/master_dark.fit
  dark_match_exposure_tolerance_percent: -1.0
)");

  auto cfg_negative =
      tile_compile::config::Config::from_yaml(negative_tolerance);
  REQUIRE_THROWS(cfg_negative.validate());
}

TEST_CASE("synthetic_clustering_mode_quantile_parses_and_validates") {
  YAML::Node node = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: OSC
  linear_required: true
synthetic:
  clustering:
    mode: quantile
    cluster_count_range: [3, 7]
)");

  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE(cfg.synthetic.clustering.mode == "quantile");
  REQUIRE(cfg.synthetic.clustering.cluster_count_range[0] == 3);
  REQUIRE(cfg.synthetic.clustering.cluster_count_range[1] == 7);
  REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("chroma_denoise_opponent_color_space_parses_and_validates") {
  YAML::Node node = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: OSC
  linear_required: true
chroma_denoise:
  enabled: true
  color_space: opponent_linear
  apply_stage: post_stack_linear
  blend:
    mode: chroma_only
    amount: 0.75
)");

  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE(cfg.chroma_denoise.enabled);
  REQUIRE(cfg.chroma_denoise.color_space == "opponent_linear");
  REQUIRE(cfg.chroma_denoise.blend.mode == "chroma_only");
  REQUIRE(std::fabs(cfg.chroma_denoise.blend.amount - 0.75f) < 1e-6f);
  REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("validation_background_rms_guard_parses_and_validates") {
  YAML::Node node = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: OSC
  linear_required: true
validation:
  max_background_rms_increase_percent: 2.5
)");

  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE(std::fabs(cfg.validation.max_background_rms_increase_percent - 2.5f) <
          1e-6f);
  REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("local_metrics_spatial_regularization_rejects_non_positive_tau_local") {
  YAML::Node node = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: OSC
  linear_required: true
local_metrics:
  spatial_regularization:
    enabled: true
    lambda: 0.35
    passes: 1
    tau_local: 0.0
)");

  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE_THROWS(cfg.validate());
}
#else
int tile_compile_tests_stacking_quality_weighting_stub() { return 0; }
#endif
