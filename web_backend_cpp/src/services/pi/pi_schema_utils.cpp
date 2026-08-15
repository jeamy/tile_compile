#include "services/pi/pi_schema_utils.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace tile_compile::pi {

bool schema_type_matches(const nlohmann::json& schema, const nlohmann::json& value) {
    if (!schema.is_object() || !schema.contains("type")) return true;

    std::vector<std::string> types;
    if (schema["type"].is_string()) {
        types.push_back(schema["type"].get<std::string>());
    } else if (schema["type"].is_array()) {
        for (const auto& type : schema["type"]) {
            if (type.is_string()) types.push_back(type.get<std::string>());
        }
    }
    if (types.empty()) return true;

    for (const auto& type : types) {
        if (type == "object" && value.is_object()) return true;
        if (type == "array" && value.is_array()) return true;
        if (type == "string" && value.is_string()) return true;
        if (type == "boolean" && value.is_boolean()) return true;
        if (type == "integer" && value.is_number_integer()) return true;
        if (type == "number" && value.is_number()) return true;
        if (type == "null" && value.is_null()) return true;
    }
    return false;
}

bool schema_enum_matches(const nlohmann::json& schema, const nlohmann::json& value) {
    if (!schema.is_object() || !schema.contains("enum") || !schema["enum"].is_array()) return true;
    return std::find(schema["enum"].begin(), schema["enum"].end(), value) != schema["enum"].end();
}

} // namespace tile_compile::pi
