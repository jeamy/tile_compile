#pragma once

#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace tile_compile::preprocessing_service {

std::vector<std::string> phase_order();
nlohmann::json default_config();
nlohmann::json parameter_groups();
nlohmann::json normalize_scan_result(const nlohmann::json& raw,
                                     const std::string& input_path);

fs::path run_dir_for_job(const fs::path& runs_dir, const std::string& job_id);
nlohmann::json read_status_from_job(const nlohmann::json& job_json);

} // namespace tile_compile::preprocessing_service
