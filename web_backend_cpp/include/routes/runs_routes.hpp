#pragma once
#include "crow_app.hpp"
#include <memory>
#include "../app_state.hpp"

void register_runs_routes(CrowApp& app,
                           std::shared_ptr<AppState> state);
