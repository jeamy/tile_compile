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
    "aqmh",           // removed AQMH reconstruction method
    "pipeline",       // production/test mode selector, never consumed at runtime
    "assumptions",    // Classic reduced/emergency frame-count gating
    "tile",           // Classic tile grid
    "tile_denoise",   // Classic tile-level soft-threshold + wiener denoise
    "local_metrics",  // per-tile local metric weighting
    "synthetic",      // Classic tile-weighted synthetic frame synthesis
    "validation",     // tile-era reconstruction acceptance gates
};

// Removed sub-keys under blocks that are otherwise kept. {block, key}.
constexpr std::pair<const char*, const char*> kRemovedSubKeys[] = {
    {"stacking", "method"},                     // STACKING is a pass-through (17.2)
    {"stacking", "sigma_clip"},
    {"stacking", "cluster_quality_weighting"},
    {"stacking", "output_stretch"},
    {"stacking", "tile_common_valid_min_fraction"},
    {"stacking", "cosmetic_correction"},
    {"stacking", "cosmetic_correction_sigma"},
    {"runtime_limits", "allow_emergency_mode"},
    {"runtime_limits", "tile_analysis_max_factor_vs_stack"},
    {"runtime_limits", "tile_reconstruction_diagnostics"},
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

  // --- 3. renames ---------------------------------------------------------
  // stacking.common_overlap_required_fraction -> reconstruction.* (same knob,
  // moved when STACKING became a pass-through).
  auto move_key = [&](YAML::Node src_parent, const char* src_key,
                      YAML::Node dst_parent, const char* dst_key,
                      const std::string& from, const std::string& to) {
    if (!src_parent || !src_parent.IsMap() || !src_parent[src_key] ||
        !src_parent[src_key].IsDefined()) {
      return;
    }
    // An explicitly set new-style key wins; the legacy value is still dropped.
    if (!dst_parent[dst_key] || !dst_parent[dst_key].IsDefined()) {
      dst_parent[dst_key] = src_parent[src_key];
    }
    src_parent.remove(src_key);
    std::cerr << "[CONFIG-MIGRATION] WARN renamed legacy config key '" << from
              << "' -> '" << to << "'." << std::endl;
    report.renamed_keys.emplace_back(from, to);
    report.applied = true;
  };

  if (node["stacking"] && node["stacking"].IsMap() &&
      node["stacking"]["common_overlap_required_fraction"] &&
      node["stacking"]["common_overlap_required_fraction"].IsDefined()) {
    if (!node["reconstruction"] || !node["reconstruction"].IsMap()) {
      node["reconstruction"] = YAML::Node(YAML::NodeType::Map);
    }
    move_key(node["stacking"], "common_overlap_required_fraction",
             node["reconstruction"], "common_overlap_required_fraction",
             "stacking.common_overlap_required_fraction",
             "reconstruction.common_overlap_required_fraction");
  }
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
