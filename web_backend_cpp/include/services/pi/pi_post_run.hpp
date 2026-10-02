#pragma once

#include "services/pi/pi_decision_policy.hpp"

#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace tile_compile::pi {

// pi.post-run-decision-state.v1: what a finished (or failed) run's own artifacts say, read-only. Nothing is measured
// here: every value is taken from run_provenance.json, forward_drizzle.json, logs/run_events.jsonl, the config and the
// presence of files. Anything that is not available is listed in `missing` and stays absent (never zero).
nlohmann::json build_post_run_state(const std::filesystem::path& run_dir, const nlohmann::json& run_config);

// pi.post-run-advice.v1. Deterministic, no model call, no file is written and no run is started. Outcomes:
//   no_change | diagnose | suggest_downstream | suggest_reconstruction.
// Suggestions come only from the checked candidate catalog (same validation as the pre-run path, against the run's
// effective config). Each carries `min_resume_phase` (latest legal resume start, computed from the runner's section
// scope, never from a model) or null with resume_mode "full_run". Feasibility (caches, provenance) is NOT decided here:
// the existing resume dry run must confirm it before anything is started.
nlohmann::json advise_post_run(const nlohmann::json& post_run_state, const nlohmann::json& run_config,
                               const DecisionPolicy& policy, const DecisionCatalog& catalog,
                               const ConfigValidator& validate_config,
                               const std::vector<std::string>& dismissed_candidates);

} // namespace tile_compile::pi
