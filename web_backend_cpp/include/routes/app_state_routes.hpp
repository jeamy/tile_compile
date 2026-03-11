#pragma once
#include "crow_app.hpp"
#include <memory>
#include "../app_state.hpp"

void register_app_state_routes(CrowApp& app,
                                std::shared_ptr<AppState> state);
