#pragma once

#include "services/pi/pi_parameter_catalog.hpp"

#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>

namespace tile_compile::pi {

nlohmann::json build_scan_pi_context(const SchemaPathMap& schema_paths,
                                     const nlohmann::json& base_config,
                                     const nlohmann::json& scan_result,
                                     const nlohmann::json& scan_metrics,
                                     const std::string& context_kind = "scan");

nlohmann::json build_run_completed_pi_context(const SchemaPathMap& schema_paths,
                                              const nlohmann::json& base_config,
                                              const std::filesystem::path& run_dir,
                                              const nlohmann::json& run_status);

nlohmann::json build_run_chat_pi_context(const std::string& run_id,
                                         const std::filesystem::path& run_dir,
                                         const nlohmann::json& status,
                                         const nlohmann::json& artifacts,
                                         const nlohmann::json& problem_ids);

} // namespace tile_compile::pi
