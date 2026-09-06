#pragma once

// Legacy config migration --- milestone M0 of the CFA-forward-drizzle plan
// (docs/AQMH/aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md
//  section 6.5, decision 2026-09-03 "strip with warning").
//
// Applied to the raw parsed YAML *before* it is mapped to config::Config:
//
//   * `method` (and any reconstruction-selection `engine` key) is a semantic
//     legacy key --- it is REJECTED fail-closed with a ConfigError whose message
//     starts with "UNKNOWN_LEGACY_KEY:". It is never silently stripped or
//     translated (this is the one exception to strip-with-warning).
//
//   * The removed *structural* Classic / tile-era blocks (`tile`, `tile_denoise`,
//     `local_metrics`, `synthetic`) and removed sub-keys are STRIPPED: a WARN is
//     logged, the stripped path is recorded in the ConfigMigrationReport, and
//     the run proceeds with the cleaned config. The report is written to
//     artifacts/config_migration.json so the strip is auditable.
//
// Key *renames* (aqmh -> reconstruction, global_metrics ->
// reconstruction.quality.frame_weights, stacking cosmetic keys ->
// calibration.frame_cleanup) are a separate follow-up coupled to the internal
// struct rename; see the note in legacy_config_migration.cpp.

#include <yaml-cpp/yaml.h>

#include <string>
#include <utility>
#include <vector>

namespace tile_compile::config {

struct ConfigMigrationReport {
  bool applied = false;
  // Dotted paths of blocks / keys removed from the config.
  std::vector<std::string> stripped_keys;
  // {from, to} for renames that were applied.
  std::vector<std::pair<std::string, std::string>> renamed_keys;

  bool empty() const {
    return stripped_keys.empty() && renamed_keys.empty();
  }

  // Serialized payload for artifacts/config_migration.json.
  std::string to_json_string() const;
};

// Rejects legacy method/engine keys (throws core::ConfigError with an
// "UNKNOWN_LEGACY_KEY: ..." message) and strips the removed structural blocks
// listed in plan section 6.5, recording every removal in `report` and emitting a
// "[CONFIG-MIGRATION] WARN ..." line per stripped path. Mutates `node` in place.
void migrate_legacy_config_node(YAML::Node& node, ConfigMigrationReport& report);

}  // namespace tile_compile::config
