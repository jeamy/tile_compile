// M0 tests for the legacy config migration
// (docs/AQMH/aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md
//  section 6.5, decision 2026-09-03).

#include "tile_compile/config/legacy_config_migration.hpp"
#include "tile_compile/core/errors.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <string>

using namespace tile_compile;
using namespace tile_compile::config;

namespace {

YAML::Node load(const std::string& yaml) { return YAML::Load(yaml); }

bool has(const std::vector<std::string>& v, const std::string& s) {
  for (const auto& e : v) if (e == s) return true;
  return false;
}

}  // namespace

TEST_CASE("migration rejects a top-level 'method' key fail-closed (plan 6.5)") {
  YAML::Node n = load("method: aqmh\nnormalization:\n  enabled: true\n");
  ConfigMigrationReport report;
  REQUIRE_THROWS_AS(migrate_legacy_config_node(n, report), ConfigError);
  try {
    YAML::Node n2 = load("method: classic_tile_compile\n");
    ConfigMigrationReport r2;
    migrate_legacy_config_node(n2, r2);
    FAIL("expected throw");
  } catch (const ConfigError& e) {
    REQUIRE_THAT(std::string(e.what()),
                 Catch::Matchers::ContainsSubstring("UNKNOWN_LEGACY_KEY"));
  }
}

TEST_CASE("migration rejects a reconstruction engine key (plan 6.5)") {
  ConfigMigrationReport report;
  YAML::Node a = load("aqmh:\n  reconstruction:\n    engine: cuda\n");
  REQUIRE_THROWS_AS(migrate_legacy_config_node(a, report), ConfigError);
  YAML::Node b = load("reconstruction:\n  engine: foo\n");
  ConfigMigrationReport r2;
  REQUIRE_THROWS_AS(migrate_legacy_config_node(b, r2), ConfigError);
}

TEST_CASE("migration strips removed structural blocks with a report (plan 6.5)") {
  const std::string yaml =
      "normalization:\n  enabled: true\n"
      "tile:\n  size_factor: 32\n"
      "tile_denoise:\n  wiener:\n    enabled: true\n"
      "local_metrics:\n  k_local: 1\n"
      "synthetic:\n  frames_min: 4\n"
      "registration:\n  engine: triangle_star_matching\n";
  YAML::Node n = load(yaml);
  ConfigMigrationReport report;
  REQUIRE_NOTHROW(migrate_legacy_config_node(n, report));

  REQUIRE(report.applied);
  REQUIRE(has(report.stripped_keys, "tile"));
  REQUIRE(has(report.stripped_keys, "tile_denoise"));
  REQUIRE(has(report.stripped_keys, "local_metrics"));
  REQUIRE(has(report.stripped_keys, "synthetic"));

  // stripped from the node
  REQUIRE_FALSE(n["tile"]);
  REQUIRE_FALSE(n["tile_denoise"]);
  REQUIRE_FALSE(n["local_metrics"]);
  REQUIRE_FALSE(n["synthetic"]);
  // kept blocks untouched
  REQUIRE(n["normalization"]);
  REQUIRE(n["registration"]);
  REQUIRE(n["registration"]["engine"].as<std::string>() ==
          "triangle_star_matching");  // not a reconstruction engine
}

TEST_CASE("migration strips removed sub-keys but keeps their parent block "
          "(plan 6.5 / 17.2)") {
  const std::string yaml =
      "dithering:\n  enabled: true\n  min_shift_px: 0.7\n"
      "stacking:\n  method: rej\n  common_overlap_required_fraction: 1\n"
      "  sigma_clip:\n    sigma_low: 2\n"
      "validation:\n  min_fwhm_improvement_percent: 5\n"
      "  require_no_tile_pattern: true\n  min_tile_weight_variance: 0.1\n";
  YAML::Node n = load(yaml);
  ConfigMigrationReport report;
  REQUIRE_NOTHROW(migrate_legacy_config_node(n, report));

  REQUIRE(has(report.stripped_keys, "dithering.min_shift_px"));
  REQUIRE(has(report.stripped_keys, "stacking.method"));
  REQUIRE(has(report.stripped_keys, "stacking.sigma_clip"));
  REQUIRE(has(report.stripped_keys, "validation.require_no_tile_pattern"));
  REQUIRE(has(report.stripped_keys, "validation.min_tile_weight_variance"));

  REQUIRE(n["dithering"]);
  REQUIRE(n["dithering"]["enabled"]);
  REQUIRE_FALSE(n["dithering"]["min_shift_px"]);

  REQUIRE(n["stacking"]);
  REQUIRE(n["stacking"]["common_overlap_required_fraction"]);  // kept
  REQUIRE_FALSE(n["stacking"]["method"]);
  REQUIRE_FALSE(n["stacking"]["sigma_clip"]);

  REQUIRE(n["validation"]["min_fwhm_improvement_percent"]);  // kept
  REQUIRE_FALSE(n["validation"]["require_no_tile_pattern"]);
}

TEST_CASE("migration is a no-op on an already-clean config (plan 6.5)") {
  YAML::Node n = load(
      "normalization:\n  enabled: true\n"
      "registration:\n  transform_model: affine\n"
      "reconstruction:\n  drizzle:\n    internal_scale: 2\n");
  ConfigMigrationReport report;
  REQUIRE_NOTHROW(migrate_legacy_config_node(n, report));
  REQUIRE_FALSE(report.applied);
  REQUIRE(report.empty());
}

TEST_CASE("ConfigMigrationReport serializes to config_migration.json (plan 6.5)") {
  ConfigMigrationReport report;
  report.applied = true;
  report.stripped_keys = {"tile", "synthetic", "stacking.method"};
  const std::string js = report.to_json_string();
  REQUIRE(js.find("\"schema_version\"") != std::string::npos);
  REQUIRE(js.find("\"applied\": true") != std::string::npos);
  REQUIRE(js.find("\"tile\"") != std::string::npos);
  REQUIRE(js.find("\"stacking.method\"") != std::string::npos);
}
