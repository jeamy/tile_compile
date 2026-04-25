#pragma once
#include "crow_app.hpp"
#include <memory>
#include "../app_state.hpp"

/// @brief Registers UI state persistence and current-run endpoints.
/// @details The function attaches routes to the shared Crow application and uses AppState for
/// runtime configuration, job tracking, event publication, and filesystem guardrails.
void register_app_state_routes(CrowApp& app,
                                std::shared_ptr<AppState> state);
