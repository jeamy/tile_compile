#pragma once
#include <filesystem>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

nlohmann::json generate_run_report(const fs::path& run_dir);
