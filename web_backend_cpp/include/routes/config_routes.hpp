#pragma once
#include "crow_app.hpp"
#include <memory>
#include "../app_state.hpp"
#include "../services/config_revisions.hpp"

void register_config_routes(CrowApp& app,
                              std::shared_ptr<AppState> state,
                              std::shared_ptr<ConfigRevisionStore> revisions);
