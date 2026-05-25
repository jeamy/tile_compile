#if __has_include(<catch2/catch_test_macros.hpp>)
#include "tile_compile/preprocessing/contract.hpp"

#include "tile_compile/core/errors.hpp"

#include <catch2/catch_test_macros.hpp>

namespace prep = tile_compile::preprocessing;

TEST_CASE("preprocessing_contract_defaults_are_separate_from_main_pipeline") {
  prep::Config cfg;

  REQUIRE(cfg.mode == "linear_prestack");
  REQUIRE(cfg.raw_formats == "tile_compile");
  REQUIRE(cfg.input_mode == "auto");
  REQUIRE(cfg.cfa_mode == "tile_compile");
  REQUIRE(cfg.mono_mode == "auto");
  REQUIRE(cfg.postprocess.astrometry);
  REQUIRE(cfg.postprocess.bge);
  REQUIRE(cfg.postprocess.pcc);
  REQUIRE(cfg.postprocess.hypermetric_stretch);
  REQUIRE(cfg.hypermetric_stretch.require_successful_pcc);
  REQUIRE(cfg.hypermetric_stretch.mode == "ready_to_use");
  REQUIRE(cfg.hypermetric_stretch.sensor_profile == "rec709");
  REQUIRE(cfg.hypermetric_stretch.fallback_profile == "rec709");
  REQUIRE(cfg.hypermetric_stretch.target_bg == 0.15f);
  REQUIRE(cfg.hypermetric_stretch.protect_b == 6.0f);
  REQUIRE(cfg.hypermetric_stretch.convergence_power == 3.5f);
  REQUIRE(cfg.hypermetric_stretch.log_d_mode == "auto");
  REQUIRE(cfg.hypermetric_stretch.fixed_log_d == 2.0f);
  REQUIRE(cfg.hypermetric_stretch.color_strategy == "fixed");
  REQUIRE(cfg.hypermetric_stretch.fixed_color_strategy == 0.0f);
  REQUIRE(cfg.hypermetric_stretch.color_grip == 1.0f);
  REQUIRE(cfg.hypermetric_stretch.shadow_convergence == 0.0f);
  REQUIRE(cfg.hypermetric_stretch.linear_expansion == 0.0f);
  REQUIRE_FALSE(cfg.hypermetric_stretch.write_channels);
  REQUIRE(cfg.hypermetric_stretch.output_rgb == "stacked_rgb_hms.fits");
  REQUIRE(cfg.report.detailed);
  REQUIRE(cfg.report.formats.size() == 3);
  REQUIRE(cfg.report.formats[2] == "html");
  REQUIRE(cfg.runtime_limits.parallel_workers == 4);
  REQUIRE(cfg.runtime_limits.memory_budget == 512);

  prep::validate(cfg);
}

TEST_CASE("preprocessing_contract_phase_order_is_independent") {
  const auto& phases = prep::phase_order();

  REQUIRE(prep::phase_to_string(phases.front()) == "INPUT_SCAN");
  REQUIRE(prep::phase_to_string(phases.back()) == "REPORT");
  REQUIRE(prep::phase_to_string(prep::Phase::REGISTRATION) == "REGISTRATION");
  REQUIRE(prep::phase_to_string(prep::Phase::PCC) == "PCC");
}

TEST_CASE("preprocessing_contract_parameter_groups_match_studio_sections") {
  const auto& groups = prep::parameter_groups();

  REQUIRE(groups.size() == 10);
  REQUIRE(groups[0] == "input");
  REQUIRE(groups[1] == "calibration");
  REQUIRE(groups[2] == "cfa_mono");
  REQUIRE(groups[3] == "registration");
  REQUIRE(groups[4] == "quality_filter");
  REQUIRE(groups[5] == "stacking");
  REQUIRE(groups[6] == "postprocess");
  REQUIRE(groups[7] == "hypermetric_stretch");
  REQUIRE(groups[8] == "report");
  REQUIRE(groups[9] == "runtime_limits");
}

TEST_CASE("preprocessing_contract_rejects_invalid_cross_option") {
  prep::Config cfg;
  cfg.postprocess.astrometry = false;
  cfg.postprocess.pcc = true;

  REQUIRE_THROWS_AS(prep::validate(cfg), tile_compile::ValidationError);
}

TEST_CASE("preprocessing_contract_validates_manual_frame_overrides") {
  prep::Config cfg;
  cfg.manual_frame_overrides.push_back({3, "", false});
  cfg.manual_frame_overrides.push_back({-1, "light_0004.fit", true});
  REQUIRE_NOTHROW(prep::validate(cfg));

  prep::Config invalid;
  invalid.manual_frame_overrides.push_back({-1, "", true});
  REQUIRE_THROWS_AS(prep::validate(invalid), tile_compile::ValidationError);
}
#else
int tile_compile_tests_preprocessing_contract_stub() { return 0; }
#endif
