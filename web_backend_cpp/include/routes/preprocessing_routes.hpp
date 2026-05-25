#pragma once

#include "crow_app.hpp"
#include "../app_state.hpp"
#include <memory>

/// @brief Registers the separate raw-preprocessing tool endpoints.
void register_preprocessing_routes(CrowApp& app,
                                   std::shared_ptr<AppState> state);
