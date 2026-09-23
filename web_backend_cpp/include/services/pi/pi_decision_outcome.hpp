#pragma once

#include <filesystem>
#include <functional>
#include <nlohmann/json.hpp>
#include <string>

namespace tile_compile::pi {

// Loads the effective config of a run as JSON. Default: <run_dir>/config.yaml via yaml-cpp,
// read-only. Injectable so tests need no YAML round trip.
using RunConfigLoader = std::function<nlohmann::json(const std::filesystem::path& run_dir)>;
nlohmann::json load_run_config_yaml(const std::filesystem::path& run_dir);

// Links a finished/started run to Jev proposals that were applied to the config draft BEFORE the run
// started (proposal.status == "applied_to_draft", applied_at <= provenance.started_at), by checking
// whether the run's own config carries each proposed value -- NOT by config-hash equality (the
// apply-time and run-start YAML serializers are not guaranteed byte-identical).
//
// Contract (docs/PI/pi_jev_m0_field_inventory_de.md section 4):
//  * Never touches PiMemoryStore or pi_outcome_recorder; state lives only under `decisions_dir`.
//  * Never writes into the run directory. The idempotence marker is
//    <decisions_dir>/_run_markers/<run_id>.json.
//  * Outcome per proposal: <decisions_dir>/<proposal_id>/outcome.json, atomic replace, one entry per
//    run_id. attribution: paths_present | paths_partial (both recorded) | paths_absent (not
//    recorded). comparison_kind is always "unpaired" and quality_delta always null: the outcome is
//    descriptive and confounded, never a quality claim.
//  * Terminal markers: no_provenance, no_applied_proposals, recorded. A read error is written as a
//    retryable marker so a later call can succeed.
// Returns the marker written or found.
nlohmann::json record_jev_outcome_if_needed(const std::filesystem::path& decisions_dir,
                                            const std::string& run_id,
                                            const std::filesystem::path& run_dir,
                                            const RunConfigLoader& loader = load_run_config_yaml);

} // namespace tile_compile::pi
