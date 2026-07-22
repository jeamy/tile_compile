#include "services/pi/pi_action_plan.hpp"

#include <string>

namespace tile_compile::pi {
namespace {

std::string json_string_value(const nlohmann::json& value, const std::string& fallback = "") {
    if (value.is_string()) return value.get<std::string>();
    if (value.is_boolean()) return value.get<bool>() ? "true" : "false";
    if (value.is_number() || value.is_null()) return value.dump();
    return fallback;
}

std::string json_string_field(const nlohmann::json& object,
                              const char* key,
                              const std::string& fallback = "") {
    if (!object.is_object() || !object.contains(key)) return fallback;
    return json_string_value(object[key], fallback);
}

double json_double_field(const nlohmann::json& object, const char* key, double fallback = 0.0) {
    if (!object.is_object() || !object.contains(key)) return fallback;
    const auto& value = object[key];
    if (value.is_number()) return value.get<double>();
    if (value.is_boolean()) return value.get<bool>() ? 1.0 : 0.0;
    if (value.is_string()) {
        try {
            return std::stod(value.get<std::string>());
        } catch (...) {
            return fallback;
        }
    }
    return fallback;
}

} // namespace

nlohmann::json build_scan_analysis_action_plan(const nlohmann::json& analysis,
                                               const nlohmann::json& updates) {
    nlohmann::json actions = nlohmann::json::array();
    if (updates.is_array()) {
        int index = 1;
        for (const auto& update : updates) {
            if (!update.is_object()) continue;
            const std::string path = json_string_field(update, "path");
            if (path.empty() || !update.contains("value")) continue;

            nlohmann::json action = {
                {"id", json_string_field(update, "id", "scan_config_set_" + std::to_string(index))},
                {"type", "config.set"},
                {"path", path},
                {"value", update["value"]},
                {"rationale", json_string_field(update, "reason", json_string_field(update, "rationale"))},
                {"confidence", json_double_field(update, "confidence", 0.0)}
            };
            if (update.contains("current_value")) action["current_value"] = update["current_value"];
            if (update.contains("risk")) action["risk"] = json_string_value(update["risk"], "unknown");
            if (update.contains("review_required")) action["review_required"] = update["review_required"];
            if (update.contains("evidence")) action["evidence"] = update["evidence"];
            actions.push_back(std::move(action));
            ++index;
        }
    }

    nlohmann::json plan = {
        {"schema_version", kActionPlanSchemaVersion},
        {"source_schema_version", json_string_field(analysis, "schema_version")},
        {"source", "pi.scan-analysis.v1"},
        {"goal", "Recommend validated Tile Compile configuration updates from scan analysis"},
        {"summary", json_string_field(analysis, "summary")},
        {"confidence", json_double_field(analysis, "confidence", 0.0)},
        {"actions", actions},
        {"post_conditions", nlohmann::json::array({{{"type", "config.valid"}}})},
        {"warnings", analysis.contains("warnings") ? analysis["warnings"] : nlohmann::json::array()}
    };
    return plan;
}

} // namespace tile_compile::pi
