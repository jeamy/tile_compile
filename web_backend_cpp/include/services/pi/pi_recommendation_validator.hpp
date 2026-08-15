#pragma once

#include "services/pi/pi_parameter_catalog.hpp"

#include <memory>
#include <nlohmann/json.hpp>

struct AppState;

namespace tile_compile::pi {

nlohmann::json normalize_candidate_updates(const nlohmann::json& analysis);

nlohmann::json validate_recommendation_updates(const nlohmann::json& candidates,
                                               const SchemaPathMap& schema_paths,
                                               const nlohmann::json& base_config,
                                               const std::shared_ptr<AppState>& state,
                                               const nlohmann::json& pi_context = nlohmann::json::object());

} // namespace tile_compile::pi
