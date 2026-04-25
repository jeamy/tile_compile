#pragma once
#include <filesystem>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

/// @brief Builds the HTML report payload and derived chart data for a completed or partial run.
nlohmann::json generate_run_report(const fs::path& run_dir);
