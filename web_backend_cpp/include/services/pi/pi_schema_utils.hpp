#pragma once

#include <nlohmann/json.hpp>

namespace tile_compile::pi {

bool schema_type_matches(const nlohmann::json& schema, const nlohmann::json& value);
bool schema_enum_matches(const nlohmann::json& schema, const nlohmann::json& value);

} // namespace tile_compile::pi
