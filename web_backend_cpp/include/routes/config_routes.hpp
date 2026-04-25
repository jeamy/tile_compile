#pragma once
#include "crow_app.hpp"
#include <memory>
#include "../app_state.hpp"
#include "../services/config_revisions.hpp"

/// @brief Registers configuration load, edit, save, validation, and revision endpoints.
/// @details The function attaches routes to the shared Crow application and uses AppState for
/// runtime configuration, job tracking, event publication, and filesystem guardrails.
void register_config_routes(CrowApp& app,
                              std::shared_ptr<AppState> state,
                              std::shared_ptr<ConfigRevisionStore> revisions);
