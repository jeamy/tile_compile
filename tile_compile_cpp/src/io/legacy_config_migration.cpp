#include "tile_compile/config/legacy_config_migration.hpp"

#include "tile_compile/core/errors.hpp"

#include <nlohmann/json.hpp>

#include <iostream>

namespace tile_compile::config {

namespace {

using json = nlohmann::json;

// Top-level structural blocks removed by the single-method cutover (plan 6.5).
// These are dropped with a warning; they carry no semantics the new pipeline
// understands.
constexpr const char* kRemovedTopLevelBlocks[] = {
    "tile",           // Classic tile grid
    "tile_denoise",   // Classic tile-level soft-threshold + wiener denoise
    "local_metrics",  // per-tile local metric weighting
    "synthetic",      // Classic tile-weighted synthetic frame synthesis
};

// Removed sub-keys under blocks that are otherwise kept. {block, key}.
constexpr std::pair<const char*, const char*> kRemovedSubKeys[] = {
    {"dithering", "min_shift_px"},              // dither is diagnosis-only now (6.5 / 26)
    {"validation", "min_tile_weight_variance"}, // tile-era
    {"validation", "require_no_tile_pattern"},  // tile-era
    {"stacking", "method"},                     // STACKING is a pass-through (17.2)
    {"stacking", "sigma_clip"},
    {"stacking", "cluster_quality_weighting"},
    {"stacking", "output_stretch"},
    {"stacking", "tile_common_valid_min_fraction"},
};

void warn_stripped(const std::string& path, ConfigMigrationReport& report) {
  std::cerr << "[CONFIG-MIGRATION] WARN removed legacy config key '" << path
            << "' (plan section 6.5); run continues with it dropped."
            << std::endl;
  report.stripped_keys.push_back(path);
  report.applied = true;
}

}  // namespace

void migrate_legacy_config_node(YAML::Node& node,
                                ConfigMigrationReport& report) {
  if (!node || !node.IsMap()) {
    return;  // nothing structured to migrate
  }

  // --- 1. semantic legacy keys: fail-closed, never stripped ----------------
  if (node["method"] && node["method"].IsDefined() &&
      !node["method"].IsNull()) {
    throw ConfigError(
        "UNKNOWN_LEGACY_KEY: top-level 'method' is not accepted by the "
        "single-method pipeline. There is exactly one reconstruction "
        "method now (plan section 6.5). Remove the 'method' key, or run "
        "'tile_compile_cli migrate-config <in> <out>'.");
  }
  // A reconstruction-selection 'engine' key, if one ever existed.
  if (node["reconstruction"] && node["reconstruction"].IsMap() &&
      node["reconstruction"]["engine"]) {
    throw ConfigError(
        "UNKNOWN_LEGACY_KEY: 'reconstruction.engine' selects a reconstruction "
        "engine; the single-method pipeline has none (plan section 6.5).");
  }
  if (node["aqmh"] && node["aqmh"].IsMap() &&
      node["aqmh"]["reconstruction"] &&
      node["aqmh"]["reconstruction"].IsMap() &&
      node["aqmh"]["reconstruction"]["engine"]) {
    throw ConfigError(
        "UNKNOWN_LEGACY_KEY: 'aqmh.reconstruction.engine' selects a "
        "reconstruction engine; the single-method pipeline has none "
        "(plan section 6.5).");
  }

  // --- 2. removed structural blocks: strip with warning -------------------
  for (const char* block : kRemovedTopLevelBlocks) {
    if (node[block] && node[block].IsDefined()) {
      node.remove(block);
      warn_stripped(block, report);
    }
  }
  for (const auto& [block, key] : kRemovedSubKeys) {
    if (node[block] && node[block].IsMap() && node[block][key] &&
        node[block][key].IsDefined()) {
      node[block].remove(key);
      warn_stripped(std::string(block) + "." + key, report);
    }
  }

  // --- 3. renames (aqmh -> reconstruction, global_metrics -> ... ) --------
  // Deferred. The renames are coupled to the internal config::Config field
  // rename and to the schema / default-YAML / examples update (plan section
  // 6.4). Until that lands, the parser keeps reading the old key names, so a
  // rename here would break parsing. Tracked in plan section 30.4.
}

std::string ConfigMigrationReport::to_json_string() const {
  json j;
  j["schema_version"] = 1;
  j["applied"] = applied;
  j["stripped_keys"] = stripped_keys;
  json renames = json::array();
  for (const auto& [from, to] : renamed_keys) {
    renames.push_back({{"from", from}, {"to", to}});
  }
  j["renamed_keys"] = std::move(renames);
  return j.dump(2);
}

}  // namespace tile_compile::config
