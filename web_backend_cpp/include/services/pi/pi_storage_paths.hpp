#pragma once

#include "app_state.hpp"

#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>

namespace tile_compile::pi {

std::filesystem::path default_pi_storage_dir(const std::shared_ptr<AppState>& state);
std::filesystem::path pi_storage_dir(const std::shared_ptr<AppState>& state);
nlohmann::json pi_storage_status(const std::shared_ptr<AppState>& state);

bool set_pi_storage_dir(const std::shared_ptr<AppState>& state,
                        const std::filesystem::path& requested,
                        std::filesystem::path& resolved,
                        std::string& error_code,
                        std::string& error_message);

} // namespace tile_compile::pi
