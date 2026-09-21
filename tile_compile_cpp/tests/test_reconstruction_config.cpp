// M0 tests for the single-method `reconstruction:` config contract
// (docs/AQMH/aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md
//  sections 6.1-6.3).

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/config/legacy_config_migration.hpp"
#include "tile_compile/core/errors.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>

using namespace tile_compile;
using namespace tile_compile::config;

namespace {

// A minimal but valid config: only the pieces Config::validate() strictly needs,
// plus a `reconstruction:` block we can tweak per test.
std::string base_yaml(const std::string& reconstruction_block) {
  return
      "pipeline:\n  mode: production\n"
      "data:\n  color_mode: MONO\n  bayer_pattern: auto\n"
      + reconstruction_block;
}

Config parse(const std::string& reconstruction_block) {
  return Config::from_yaml_text(base_yaml(reconstruction_block));
}

}  // namespace

TEST_CASE("reconstruction: defaults are present and valid (plan 6.1)") {
  Config cfg = parse("");  // no reconstruction block -> struct defaults
  const auto& r = cfg.reconstruction;
  REQUIRE(r.delete_source_cache_after_run == false);
  REQUIRE(r.diagnostics.preview_forward_drizzle_uniform == false);
  REQUIRE(r.diagnostics.persist_forward_drizzle_uniform_store == false);
  REQUIRE(r.drizzle.internal_scale == 2);
  REQUIRE(r.drizzle.output_scale == 1);
  REQUIRE(r.drizzle.kernel == "square");
  REQUIRE(r.drizzle.chunk_halo_rows == -1);
  REQUIRE(r.multiband.levels == 3);
  REQUIRE_NOTHROW(r.validate());
  REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("reconstruction: a full block round-trips through parse (plan 6.1)") {
  const std::string block =
      "reconstruction:\n"
      "  delete_source_cache_after_run: true\n"
      "  keep_profile_cache_after_run: true\n"
      "  common_overlap_required_fraction: 0.9\n"
      "  diagnostics:\n    level: full\n    preview_forward_drizzle_uniform: true\n"
      "    persist_forward_drizzle_uniform_store: true\n"
      "  drizzle:\n"
      "    internal_scale: 2\n    output_scale: 2\n    kernel: square\n"
      "    pixfrac: 0.75\n    robust_passes: 3\n    min_clip_contributors: 6\n"
      "    chunk_rows: 128\n    chunk_halo_rows: 4\n    memory_budget_mb: 2048\n"
      "  clipping:\n"
      "    clip_sigma_low: 2.5\n    clip_sigma_high: 3.5\n"
      "    min_fraction: 0.5\n    min_n_eff: 4.0\n"
      "  coverage_gate:\n"
      "    min_frames: 12\n    min_supported_fraction: 0.99\n"
      "    min_channel_n_eff_floor: 4.0\n    min_channel_n_eff_fraction: 0.2\n"
      "    min_analysis_pixels: 2048\n    max_internal_hole_area_px: 5\n"
      "  quality:\n    pyramid:\n"
      "      scales: 3\n      base_window_px: 8\n      sharpness_weight: 0.7\n"
      "      snr_weight: 0.5\n      score_scale: 2.2\n      artifact_sigma: 4.0\n"
      "      max_artifact_fraction: 0.4\n"
      "  multiband:\n"
      "    enabled: false\n    levels: 2\n    alpha_cap: 0.8\n"
      "    fine_quality_exponent: 5.0\n    medium_quality_exponent: 1.5\n"
      "    min_quality_separation: 0.03\n    full_quality_separation: 0.25\n"
      "    min_effective_samples: 6.0\n    full_effective_samples: 30.0\n"
      "  multiband_validation:\n"
      "    fwhm_ratio_max: 0.9\n    p90_fwhm_ratio_max: 1.05\n"
      "    tail_ratio_max: 1.2\n    elongation_ratio_max: 1.15\n"
      "    background_rms_ratio_max: 1.2\n    seam_ratio_max: 1.1\n"
      "    min_stars_fwhm: 15\n    min_stars_p90_tail_elongation: 25\n"
      "    max_fwhm_ci_relative_width: 0.15\n";
  Config cfg = parse(block);
  const auto& r = cfg.reconstruction;
  REQUIRE(r.delete_source_cache_after_run == true);
  REQUIRE(r.common_overlap_required_fraction == 0.9f);
  REQUIRE(r.diagnostics.level == "full");
  REQUIRE(r.diagnostics.preview_forward_drizzle_uniform == true);
  REQUIRE(r.diagnostics.persist_forward_drizzle_uniform_store == true);
  REQUIRE(r.drizzle.output_scale == 2);
  REQUIRE(r.drizzle.pixfrac == 0.75f);
  REQUIRE(r.drizzle.min_clip_contributors == 6);
  REQUIRE(r.drizzle.chunk_halo_rows == 4);
  REQUIRE(r.clipping.clip_sigma_high == 3.5f);
  REQUIRE(r.coverage_gate.min_frames == 12);
  REQUIRE(r.coverage_gate.max_internal_hole_area_px == 5);
  REQUIRE(r.quality.pyramid.scales == 3);
  REQUIRE(r.quality.pyramid.base_window_px == 8);
  REQUIRE(r.quality.pyramid.sharpness_weight == 0.7f);
  REQUIRE(r.quality.pyramid.snr_weight == 0.5f);
  REQUIRE(r.quality.pyramid.score_scale == 2.2f);
  REQUIRE(r.quality.pyramid.artifact_sigma == 4.0f);
  REQUIRE(r.quality.pyramid.max_artifact_fraction == 0.4f);
  REQUIRE(r.multiband.enabled == false);
  REQUIRE(r.multiband.levels == 2);
  REQUIRE(r.multiband.full_effective_samples == 30.0f);
  REQUIRE(r.multiband_validation.fwhm_ratio_max == 0.9);
  REQUIRE(r.multiband_validation.background_rms_ratio_max == 1.2);
  REQUIRE(r.multiband_validation.min_stars_fwhm == 15);
  REQUIRE(r.multiband_validation.min_stars_p90_tail_elongation == 25);
  REQUIRE(r.multiband_validation.max_fwhm_ci_relative_width == 0.15);
  REQUIRE_NOTHROW(cfg.validate());

  // serialize -> parse -> compare a few fields
  Config cfg2 = Config::from_yaml(cfg.to_yaml());
  REQUIRE(cfg2.reconstruction.drizzle.pixfrac == r.drizzle.pixfrac);
  REQUIRE(cfg2.reconstruction.multiband.levels == r.multiband.levels);
  REQUIRE(cfg2.reconstruction.coverage_gate.min_frames == r.coverage_gate.min_frames);
  REQUIRE(cfg2.reconstruction.quality.pyramid.score_scale ==
          r.quality.pyramid.score_scale);
  REQUIRE(cfg2.reconstruction.diagnostics.level == "full");
  REQUIRE(cfg2.reconstruction.diagnostics.preview_forward_drizzle_uniform == true);
  REQUIRE(cfg2.reconstruction.diagnostics.persist_forward_drizzle_uniform_store == true);
  REQUIRE(cfg2.reconstruction.multiband_validation.background_rms_ratio_max ==
          r.multiband_validation.background_rms_ratio_max);
  REQUIRE(cfg2.reconstruction.multiband_validation.min_stars_fwhm ==
          r.multiband_validation.min_stars_fwhm);
}

TEST_CASE("reconstruction: validation rejects contract violations (plan 6.3)") {
  auto expect_reject = [](const std::string& block) {
    Config cfg = parse(block);
    REQUIRE_THROWS_AS(cfg.reconstruction.validate(), ValidationError);
  };

  expect_reject("reconstruction:\n  drizzle:\n    internal_scale: 3\n");
  expect_reject("reconstruction:\n  drizzle:\n    internal_scale: 1\n    output_scale: 2\n");
  expect_reject("reconstruction:\n  drizzle:\n    kernel: gaussian\n");
  expect_reject("reconstruction:\n  drizzle:\n    pixfrac: 1.5\n");
  expect_reject("reconstruction:\n  drizzle:\n    pixfrac: 0.0\n");
  expect_reject("reconstruction:\n  drizzle:\n    robust_passes: 7\n");
  expect_reject("reconstruction:\n  drizzle:\n    min_clip_contributors: 1\n");
  expect_reject("reconstruction:\n  drizzle:\n    chunk_halo_rows: 0\n");
  expect_reject("reconstruction:\n  clipping:\n    clip_sigma_low: 0\n");
  expect_reject("reconstruction:\n  clipping:\n    min_n_eff: 0.5\n");
  expect_reject("reconstruction:\n  diagnostics:\n    level: verbose\n");
  expect_reject("reconstruction:\n  quality:\n    pyramid:\n      scales: 9\n");
  expect_reject("reconstruction:\n  quality:\n    pyramid:\n      base_window_px: 0\n");
  expect_reject("reconstruction:\n  quality:\n    pyramid:\n"
                "      sharpness_weight: 0\n      snr_weight: 0\n");
  expect_reject("reconstruction:\n  quality:\n    pyramid:\n      score_scale: 0\n");
  expect_reject("reconstruction:\n  quality:\n    pyramid:\n      artifact_sigma: 0\n");
  expect_reject(
      "reconstruction:\n  quality:\n    pyramid:\n      max_artifact_fraction: 0\n");
  expect_reject("reconstruction:\n  coverage_gate:\n    min_frames: 1\n");
  expect_reject("reconstruction:\n  coverage_gate:\n    min_supported_fraction: 1.5\n");
  expect_reject("reconstruction:\n  coverage_gate:\n    min_channel_n_eff_fraction: 0\n");
  expect_reject("reconstruction:\n  multiband:\n    levels: 5\n");
  expect_reject(
      "reconstruction:\n  quality:\n    pyramid:\n      scales: 1\n"
      "  multiband:\n    levels: 2\n");  // levels>=2 needs scales>=2
  expect_reject("reconstruction:\n  multiband:\n    alpha_cap: 1.5\n");
  expect_reject(
      "reconstruction:\n  multiband:\n"
      "    min_quality_separation: 0.3\n    full_quality_separation: 0.2\n");
  expect_reject(
      "reconstruction:\n  multiband:\n"
      "    min_effective_samples: 0.5\n");
  expect_reject(
      "reconstruction:\n  multiband:\n"
      "    min_effective_samples: 30\n    full_effective_samples: 10\n");
  expect_reject("reconstruction:\n  common_overlap_required_fraction: 1.5\n");
  expect_reject(
      "reconstruction:\n  multiband_validation:\n    fwhm_ratio_max: 0\n");
  expect_reject(
      "reconstruction:\n  multiband_validation:\n    background_rms_ratio_max: -1\n");
  expect_reject(
      "reconstruction:\n  multiband_validation:\n    seam_ratio_max: 0\n");
  expect_reject(
      "reconstruction:\n  multiband_validation:\n    min_stars_fwhm: -1\n");
  expect_reject(
      "reconstruction:\n  multiband_validation:\n    max_fwhm_ci_relative_width: 0\n");
}

TEST_CASE("reconstruction: a legacy aqmh pyramid block is stripped wholesale") {
  ConfigMigrationReport report;
  Config cfg = Config::from_yaml_text_migrated(
      "data:\n  color_mode: MONO\n  bayer_pattern: auto\n"
      "aqmh:\n  pyramid:\n"
      "    scales: 6\n    base_window_px: 12\n    w_sharp: 0.8\n    w_snr: 0.3\n"
      "    score_scale: 2.5\n    k_artifact: 4.5\n    frac_artifact_max: 0.5\n",
      report);
  const auto& p = cfg.reconstruction.quality.pyramid;
  const auto& d = Config{}.reconstruction.quality.pyramid;
  REQUIRE(p.scales == d.scales);
  REQUIRE(p.base_window_px == d.base_window_px);
  REQUIRE(report.applied);
}

TEST_CASE("reconstruction: explicit source quality parameters parse directly") {
  ConfigMigrationReport report;
  Config cfg = Config::from_yaml_text_migrated(
      "data:\n  color_mode: MONO\n  bayer_pattern: auto\n"
      "reconstruction:\n  quality:\n    pyramid:\n"
      "      scales: 2\n      score_scale: 1.1\n",
      report);
  const auto& p = cfg.reconstruction.quality.pyramid;
  REQUIRE(p.scales == 2);
  REQUIRE(p.score_scale == 1.1f);
  REQUIRE(p.sharpness_weight == Config{}.reconstruction.quality.pyramid.sharpness_weight);
}

TEST_CASE("reconstruction: the removed prewarp cache key is not read here "
          "(plan 6.2)") {
  // `delete_prewarped_cache_after_run` belongs to the legacy aqmh block; it must
  // not leak into the new reconstruction contract.
  Config cfg = parse(
      "reconstruction:\n  delete_prewarped_cache_after_run: true\n");
  REQUIRE(cfg.reconstruction.delete_source_cache_after_run == false);
}

TEST_CASE("reconstruction: drizzle.full_frame_estimator defaults off and parses") {
  Config def = parse("reconstruction:\n  drizzle:\n    pixfrac: 0.8\n");
  REQUIRE_FALSE(def.reconstruction.drizzle.full_frame_estimator);
  Config on = parse(
      "reconstruction:\n  drizzle:\n    full_frame_estimator: true\n");
  REQUIRE(on.reconstruction.drizzle.full_frame_estimator);
  REQUIRE_NOTHROW(on.reconstruction.validate());
  // The estimator needs the multiband profile streams.
  Config no_mb = parse(
      "reconstruction:\n  drizzle:\n    full_frame_estimator: true\n"
      "  multiband:\n    enabled: false\n");
  REQUIRE_THROWS_AS(no_mb.reconstruction.validate(), ValidationError);
}

TEST_CASE("hypermetric_stretch.color_cast_correction defaults off, parses and validates") {
  Config def = parse("hypermetric_stretch:\n  enabled: true\n");
  REQUIRE_FALSE(def.hypermetric_stretch.color_cast_correction.enabled);
  REQUIRE(def.hypermetric_stretch.color_cast_correction.max_amount == 1.0f);
  REQUIRE(def.hypermetric_stretch.color_cast_correction.min_excess == 1.02f);
  REQUIRE(def.hypermetric_stretch.color_cast_correction.brightness_bins == 8);
  REQUIRE_FALSE(def.hypermetric_stretch.color_cast_correction.neutralize_sky);
  Config on = parse(
      "hypermetric_stretch:\n  color_cast_correction:\n    enabled: true\n"
      "    max_amount: 0.8\n    target_ratio: 1.05\n    min_excess: 1.03\n"
      "    object_sigma: 4\n    brightness_bins: 5\n    neutralize_sky: true\n");
  const auto& c = on.hypermetric_stretch.color_cast_correction;
  REQUIRE(c.enabled);
  REQUIRE(c.max_amount == 0.8f);
  REQUIRE(c.target_ratio == 1.05f);
  REQUIRE(c.min_excess == 1.03f);
  REQUIRE(c.object_sigma == 4.0f);
  REQUIRE(c.brightness_bins == 5);
  REQUIRE(c.neutralize_sky);
  REQUIRE_NOTHROW(on.validate());
  for (const char* bad : {"max_amount: 0", "max_amount: 1.5", "target_ratio: 0.4",
                          "min_excess: 0.9", "object_sigma: 0", "brightness_bins: 0",
                          "brightness_bins: 33"}) {
    Config b = parse(std::string("hypermetric_stretch:\n  color_cast_correction:\n    ") +
                     bad + "\n");
    REQUIRE_THROWS_AS(b.validate(), ValidationError);
  }
}
