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
// Same conversion for YAML text (e.g. the config draft the UI sends): scalar types are recovered
// (bool/int/double/null), quoted scalars stay strings. Throws on malformed YAML.
nlohmann::json yaml_text_to_json(const std::string& yaml_text);

// A Jev draft becomes attributable only when the exact returned config is saved as a revision.
bool matches_applied_jev_config(const std::filesystem::path& decisions_dir,
                                const std::string& proposal_id,
                                const nlohmann::json& saved_config);
void record_jev_saved_revision(const std::filesystem::path& decisions_dir,
                               const std::string& proposal_id,
                               const std::string& revision_id);
// Checks an explicitly named, saved Jev proposal against the input files and effective run config.
bool matches_saved_jev_run(const std::filesystem::path& decisions_dir,
                           const std::string& proposal_id,
                           const std::filesystem::path& input_dir,
                           const nlohmann::json& run_config);
// Queue variant: source_dir must be the scanned directory, and staged_dir must
// contain the same selected FITS metadata after queue materialization.
bool matches_saved_jev_run(const std::filesystem::path& decisions_dir,
                           const std::string& proposal_id,
                           const std::filesystem::path& source_dir,
                           const std::filesystem::path& staged_dir,
                           const nlohmann::json& run_config);

// Records the single proposal explicitly verified at run start and named in run provenance.
// It must have been saved before the run and its proposed values must occur in the run config.
// Config-hash equality is not used because the run-start serializer can inject other fields.
//
// Contract (docs/PI/pi_jev_m0_field_inventory_de.md section 4):
//  * Never touches PiMemoryStore or pi_outcome_recorder; state lives only under `decisions_dir`.
//  * Never writes into the run directory. The idempotence marker is
//    <decisions_dir>/_run_markers/<run_id>.json.
//  * Outcome per proposal: <decisions_dir>/<proposal_id>/outcome.json, atomic replace, one entry per
//    run_id. attribution: paths_present | paths_partial (both recorded) | paths_absent (not
//    recorded). comparison_kind is always "unpaired" and quality_delta always null: the outcome is
//    descriptive and confounded, never a quality claim.
//  * Terminal markers: no_applied_proposals, recorded. Missing provenance and read errors are
//    retryable because a run-completion callback can race with provenance writing.
// Returns the marker written or found.
nlohmann::json record_jev_outcome_if_needed(const std::filesystem::path& decisions_dir,
                                            const std::string& run_id,
                                            const std::filesystem::path& run_dir,
                                            const RunConfigLoader& loader = load_run_config_yaml);

} // namespace tile_compile::pi
