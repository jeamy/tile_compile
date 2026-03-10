#pragma once
#include <filesystem>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

static const std::vector<std::string> PHASE_ORDER = {
    "scan", "local_metrics", "metrics", "registration",
    "stacking", "clustering", "bge", "pcc", "validation"
};

static const std::vector<std::string> RESUME_FROM_PHASES = {
    "local_metrics", "metrics", "registration",
    "stacking", "clustering", "bge", "pcc", "validation"
};

nlohmann::json read_run_status(const fs::path& run_dir);
std::vector<nlohmann::json> discover_runs(const fs::path& runs_dir, int limit = 50);
std::string read_run_logs(const fs::path& run_dir, int tail = 250);
nlohmann::json list_run_artifacts(const fs::path& run_dir);
