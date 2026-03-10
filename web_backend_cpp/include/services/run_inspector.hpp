#pragma once
#include <filesystem>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

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
    "ASTROMETRY",
    "BGE",
    "PCC"
};

nlohmann::json read_run_status(const fs::path& run_dir);
std::vector<nlohmann::json> discover_runs(const fs::path& runs_dir, int limit = 50);
std::string read_run_logs(const fs::path& run_dir, int tail = 250);
nlohmann::json list_run_artifacts(const fs::path& run_dir);
