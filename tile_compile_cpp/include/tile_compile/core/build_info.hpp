#pragma once

#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>

namespace tile_compile::core {

std::filesystem::path current_executable_path();
nlohmann::json binary_provenance_json(const std::filesystem::path& path);
nlohmann::json build_info_json(bool include_runtime_binary = true);
std::string build_info_text();

} // namespace tile_compile::core
