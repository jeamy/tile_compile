#pragma once

#include "crow_app.hpp"
#include <memory>

struct AppState;

namespace tile_compile::routes {

void register_pi_routes(CrowApp& app, std::shared_ptr<AppState> state);

} // namespace tile_compile::routes
