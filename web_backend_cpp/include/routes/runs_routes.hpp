#pragma once
#include "crow_app.hpp"
#include <memory>
#include "../app_state.hpp"

/// @brief Registers run discovery, status, resume, artifact, report, and queue endpoints.
/// @details The function attaches routes to the shared Crow application and uses AppState for
/// runtime configuration, job tracking, event publication, and filesystem guardrails.
void register_runs_routes(CrowApp& app,
                           std::shared_ptr<AppState> state);
