#if __has_include(<catch2/catch_test_macros.hpp>)
#include "tile_compile/config/configuration.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>

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

TEST_CASE("stacking_tile_seam_harmonization_parses_and_validates") {
  YAML::Node node = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: OSC
  linear_required: true
stacking:
  method: average
  tile_seam_harmonization:
    enabled: true
    strength: 0.6
    sample_quantile: 0.25
    gradient_quantile: 0.8
    min_sample_fraction: 0.07
    min_samples: 48
    scale_floor_factor: 0.7
    scale_ceil_factor: 1.6
)");

  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE(cfg.stacking.tile_seam_harmonization.enabled);
  REQUIRE(std::fabs(cfg.stacking.tile_seam_harmonization.strength - 0.6f) <
          1e-6f);
  REQUIRE(std::fabs(
              cfg.stacking.tile_seam_harmonization.sample_quantile - 0.25f) <
          1e-6f);
  REQUIRE(std::fabs(
              cfg.stacking.tile_seam_harmonization.gradient_quantile - 0.8f) <
          1e-6f);
  REQUIRE(std::fabs(cfg.stacking.tile_seam_harmonization.min_sample_fraction -
                    0.07f) < 1e-6f);
  REQUIRE(cfg.stacking.tile_seam_harmonization.min_samples == 48);
  REQUIRE(std::fabs(
              cfg.stacking.tile_seam_harmonization.scale_floor_factor - 0.7f) <
          1e-6f);
  REQUIRE(std::fabs(
              cfg.stacking.tile_seam_harmonization.scale_ceil_factor - 1.6f) <
          1e-6f);
  REQUIRE_NOTHROW(cfg.validate());
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

TEST_CASE("stacking_tile_seam_harmonization_rejects_invalid_scale_window") {
  YAML::Node node = YAML::Load(R"(
data:
  frames_min: 1
  color_mode: OSC
  linear_required: true
stacking:
  method: average
  tile_seam_harmonization:
    enabled: true
    strength: 0.6
    sample_quantile: 0.25
    gradient_quantile: 0.8
    min_sample_fraction: 0.07
    min_samples: 48
    scale_floor_factor: 1.7
    scale_ceil_factor: 1.6
)");

  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE_THROWS(cfg.validate());
}
#else
int tile_compile_tests_stacking_quality_weighting_stub() { return 0; }
#endif
