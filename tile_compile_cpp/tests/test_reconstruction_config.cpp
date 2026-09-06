// M0 tests for the single-method `reconstruction:` config contract
// (docs/AQMH/aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md
//  sections 6.1-6.3).

#include "tile_compile/config/configuration.hpp"
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
      "method: aqmh\n"
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
      "  quality:\n    pyramid:\n      scales: 3\n"
      "  multiband:\n"
      "    enabled: false\n    levels: 2\n    alpha_cap: 0.8\n"
      "    fine_quality_exponent: 5.0\n    medium_quality_exponent: 1.5\n"
      "    min_quality_separation: 0.03\n    full_quality_separation: 0.25\n"
      "    min_effective_samples: 6.0\n    full_effective_samples: 30.0\n";
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
  REQUIRE(r.multiband.enabled == false);
  REQUIRE(r.multiband.levels == 2);
  REQUIRE(r.multiband.full_effective_samples == 30.0f);
  REQUIRE_NOTHROW(cfg.validate());

  // serialize -> parse -> compare a few fields
  Config cfg2 = Config::from_yaml(cfg.to_yaml());
  REQUIRE(cfg2.reconstruction.drizzle.pixfrac == r.drizzle.pixfrac);
  REQUIRE(cfg2.reconstruction.multiband.levels == r.multiband.levels);
  REQUIRE(cfg2.reconstruction.coverage_gate.min_frames == r.coverage_gate.min_frames);
  REQUIRE(cfg2.reconstruction.diagnostics.level == "full");
  REQUIRE(cfg2.reconstruction.diagnostics.preview_forward_drizzle_uniform == true);
  REQUIRE(cfg2.reconstruction.diagnostics.persist_forward_drizzle_uniform_store == true);
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
  expect_reject("reconstruction:\n  quality:\n    pyramid:\n      scales: 5\n");
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
}

TEST_CASE("reconstruction: the removed prewarp cache key is not read here "
          "(plan 6.2)") {
  // `delete_prewarped_cache_after_run` belongs to the legacy aqmh block; it must
  // not leak into the new reconstruction contract.
  Config cfg = parse(
      "reconstruction:\n  delete_prewarped_cache_after_run: true\n");
  REQUIRE(cfg.reconstruction.delete_source_cache_after_run == false);
}
