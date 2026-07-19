#pragma once
#include <filesystem>
#include <string>
#include <vector>
#include <optional>
#include <nlohmann/json.hpp>
#include "job_store.hpp"

namespace fs = std::filesystem;

/// @brief Normalizes a phase event name based on the reconstruction method.
/// For AQMH method, certain Classic phases are hidden and others are relabeled.
std::string normalizePhaseEvent(const std::string& event, const std::string& method);

/// @brief Returns the method-aware pipeline phase order for computing run status and progress.
std::vector<std::string> getPhaseOrderForMethod(const std::string& method);

/// @brief Canonical pipeline phase order used to compute run status and progress.
static const std::vector<std::string> PHASE_ORDER = {
    "SCAN_INPUT",
    "CHANNEL_SPLIT",
    "NORMALIZATION",
    "GLOBAL_METRICS",
    "TILE_GRID",
    "REGISTRATION",
    "PREWARP",
    "COMMON_OVERLAP",
    "LOCAL_METRICS",
    "TILE_RECONSTRUCTION",
    "STATE_CLUSTERING",
    "SYNTHETIC_FRAMES",
    "STACKING",
    "DEBAYER",
    "ASTROMETRY",
    "BGE",
    "PCC",
    "HYPERMETRIC_STRETCH"
};

/// @brief Phases accepted by resume endpoints and UI selectors.
/// Only phases whose required artifacts persist after a normal run are listed.
/// Earlier phases (before STACKING) require cache/prewarped_frames which is normally
/// deleted and are therefore validated dynamically in the resume endpoint.
static const std::vector<std::string> RESUME_FROM_PHASES = {
    "STACKING",
    "DEBAYER",
    "ASTROMETRY",
    "BGE",
    "PCC",
    "HYPERMETRIC_STRETCH"
};

/// @brief Reads events and artifacts from a run directory and builds the public status object.
nlohmann::json read_run_status(const fs::path& run_dir);
/// @brief Returns whether a queue JSON document references a run id.
bool queue_contains_run_id(const nlohmann::json& queue, const std::string& run_id);
/// @brief Returns whether a backend job belongs to or mentions the requested run.
bool job_references_run_id(const Job& job, const std::string& run_id);
/// @brief Finds the newest job associated with a run.
std::optional<Job> latest_run_job(const InMemoryJobStore& store, const std::string& run_id, int limit = 500);
/// @brief Overlays transient backend job state onto a persisted run status JSON object.
void apply_job_state_to_run_status(nlohmann::json& status, const std::optional<Job>& job);
/// @brief Probes the operating system for a still-running tile_compile runner process.
bool has_live_runner_process(const std::string& runner_exe, const std::string& run_id, const std::string& run_dir);
/// @brief Marks apparently abandoned active phases as aborted when no runner process is live.
void apply_runtime_liveness_to_run_status(nlohmann::json& status,
                                          const std::optional<Job>& job,
                                          const std::string& runner_exe,
                                          const std::string& run_id,
                                          const std::string& run_dir);
/// @brief Scans the runs directory and returns recent run summaries.
std::vector<nlohmann::json> discover_runs(const fs::path& runs_dir, int limit = 50);
/// @brief Reads the tail of known run log files for display in the UI.
std::string read_run_logs(const fs::path& run_dir, int tail = 250);
/// @brief Lists reportable files and generated artifacts for a run.
nlohmann::json list_run_artifacts(const fs::path& run_dir);
