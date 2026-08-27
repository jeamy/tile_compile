#pragma once

#include <filesystem>
#include <memory>
#include <string>

struct AppState;

namespace tile_compile::pi {

// Joins a run's provenance (docs/PI/pi_local_learning_plan_de.md, Schritt 1a:
// runs/<run_id>/artifacts/pi_run_provenance.json) and measured quality (Schritt 1b:
// runs/<run_id>/artifacts/pi_run_quality.json) with the PI memory candidate that produced the
// run's config (matched by config_sha256), and attaches the measured outcome to that candidate
// (Schritt 1c) via PiMemoryStore::attach_outcome().
//
// Idempotent and cheap to call repeatedly (e.g. from a status-poll route): once a run's outcome
// has been resolved — matched or not — a marker file (runs/<run_id>/artifacts/pi_outcome_recorded.json)
// short-circuits every subsequent call to a single fs::exists() check.
//
// Also call this once, unconditionally, right before deleting a run directory: it is the last
// chance to harvest an unrecorded outcome before pi_run_provenance.json/pi_run_quality.json are
// gone for good (the attached outcome itself lives in the PI memory store, outside the run
// directory, and survives run deletion once recorded — but only once recorded).
void record_run_outcome_if_needed(const std::shared_ptr<AppState>& state,
                                  const std::string& run_id,
                                  const std::filesystem::path& run_dir);

} // namespace tile_compile::pi
