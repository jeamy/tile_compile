#pragma once
#include <filesystem>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "job_store.hpp"

namespace fs = std::filesystem;

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
    "PCC"
};

static const std::vector<std::string> RESUME_FROM_PHASES = {
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
    "PCC"
};

nlohmann::json read_run_status(const fs::path& run_dir);
bool queue_contains_run_id(const nlohmann::json& queue, const std::string& run_id);
bool job_references_run_id(const Job& job, const std::string& run_id);
std::optional<Job> latest_run_job(const InMemoryJobStore& store, const std::string& run_id, int limit = 500);
void apply_job_state_to_run_status(nlohmann::json& status, const std::optional<Job>& job);
bool has_live_runner_process(const std::string& runner_exe, const std::string& run_id, const std::string& run_dir);
void apply_runtime_liveness_to_run_status(nlohmann::json& status,
                                          const std::optional<Job>& job,
                                          const std::string& runner_exe,
                                          const std::string& run_id,
                                          const std::string& run_dir);
std::vector<nlohmann::json> discover_runs(const fs::path& runs_dir, int limit = 50);
std::string read_run_logs(const fs::path& run_dir, int tail = 250);
nlohmann::json list_run_artifacts(const fs::path& run_dir);
