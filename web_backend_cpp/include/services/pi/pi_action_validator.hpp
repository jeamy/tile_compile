#pragma once

#include <nlohmann/json.hpp>

namespace tile_compile::pi {

nlohmann::json validate_action_plan_shape(const nlohmann::json& plan);
nlohmann::json prevalidate_config_updates(const nlohmann::json& candidates,
                                          const nlohmann::json& schema_by_path);

} // namespace tile_compile::pi
