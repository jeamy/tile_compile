#pragma once

#include "crow_app.hpp"
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>

struct AppState;

namespace tile_compile::routes {

/// Builds a bounded PNG preview from the immutable pre-BGE/pre-HMS run output.
/// Only outputs/stacked_rgb.fits is accepted; later processed artifacts are never selected.
nlohmann::json build_run_completion_preview_image(const std::filesystem::path& run_dir);

void register_pi_routes(CrowApp& app, std::shared_ptr<AppState> state);

} // namespace tile_compile::routes
