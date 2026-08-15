#pragma once

#include <map>
#include <nlohmann/json.hpp>
#include <string>

namespace tile_compile::pi {

using SchemaPathMap = std::map<std::string, nlohmann::json>;

nlohmann::json curated_parameter_metadata(const std::string& path);
nlohmann::json build_parameter_catalog(const SchemaPathMap& schema_paths,
                                       const nlohmann::json& base_config);

} // namespace tile_compile::pi
