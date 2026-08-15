#include "services/pi/pi_action_validator.hpp"
#include "services/pi/pi_action_plan.hpp"
#include "services/pi/pi_schema_utils.hpp"

#include <set>
#include <string>

namespace tile_compile::pi {
namespace {

std::string string_field(const nlohmann::json& object, const char* key) {
    if (!object.is_object() || !object.contains(key) || !object[key].is_string()) return "";
    return object[key].get<std::string>();
}

void add_error(nlohmann::json& errors,
               const std::string& code,
               const std::string& message,
               const std::string& path = "") {
    nlohmann::json error = {{"code", code}, {"message", message}};
    if (!path.empty()) error["path"] = path;
    errors.push_back(std::move(error));
}

bool action_type_supported(const std::string& type) {
    static const std::set<std::string> supported = {
        "config.set",
        "config.patch",
        "config.diff.preview",
        "preview.bge.plan",
        "preview.hms.plan",
        "run.resume.plan"
    };
    return supported.find(type) != supported.end();
}

} // namespace

nlohmann::json validate_action_plan_shape(const nlohmann::json& plan) {
    nlohmann::json errors = nlohmann::json::array();

    if (!plan.is_object()) {
        add_error(errors, "not_object", "Action plan must be a JSON object");
        return {{"valid", false}, {"errors", errors}};
    }

    const std::string schema_version = string_field(plan, "schema_version");
    if (schema_version != kActionPlanSchemaVersion) {
        add_error(errors, "unsupported_schema_version",
                  "Action plan schema_version must be pi.action-plan.v1",
                  "schema_version");
    }

    if (!plan.contains("actions") || !plan["actions"].is_array()) {
        add_error(errors, "actions_not_array", "Action plan actions must be an array", "actions");
    } else {
        std::set<std::string> ids;
        int index = 0;
        for (const auto& action : plan["actions"]) {
            const std::string base = "actions[" + std::to_string(index) + "]";
            if (!action.is_object()) {
                add_error(errors, "action_not_object", "Action must be a JSON object", base);
                ++index;
                continue;
            }

            const std::string id = string_field(action, "id");
            if (id.empty()) {
                add_error(errors, "missing_action_id", "Action id is required", base + ".id");
            } else if (!ids.insert(id).second) {
                add_error(errors, "duplicate_action_id", "Action id must be unique", base + ".id");
            }

            const std::string type = string_field(action, "type");
            if (type.empty()) {
                add_error(errors, "missing_action_type", "Action type is required", base + ".type");
            } else if (!action_type_supported(type)) {
                add_error(errors, "unsupported_action_type", "Action type is not supported", base + ".type");
            }

            if (type == "config.set") {
                if (string_field(action, "path").empty()) {
                    add_error(errors, "missing_config_path", "config.set action requires path", base + ".path");
                }
                if (!action.contains("value")) {
                    add_error(errors, "missing_config_value", "config.set action requires value", base + ".value");
                }
            }

            if (type == "config.patch") {
                if (!action.contains("updates") || !action["updates"].is_array()) {
                    add_error(errors, "missing_patch_updates",
                              "config.patch action requires updates array",
                              base + ".updates");
                }
            }

            ++index;
        }
    }

    return {
        {"valid", errors.empty()},
        {"schema_version", kActionPlanSchemaVersion},
        {"errors", errors}
    };
}

nlohmann::json prevalidate_config_updates(const nlohmann::json& candidates,
                                          const nlohmann::json& schema_by_path) {
    nlohmann::json validated = nlohmann::json::array();
    nlohmann::json rejected = nlohmann::json::array();

    if (!candidates.is_array()) {
        return {
            {"validated_updates", validated},
            {"rejected_updates", rejected},
            {"candidate_count", 0}
        };
    }

    for (const auto& update : candidates) {
        if (!update.is_object()) continue;
        const std::string path = string_field(update, "path");
        if (path.empty() || !update.contains("value")) {
            nlohmann::json reject = update;
            reject["applicable"] = false;
            reject["reject_reason"] = path.empty() ? "missing_path" : "missing_value";
            rejected.push_back(std::move(reject));
            continue;
        }

        if (!schema_by_path.is_object() || !schema_by_path.contains(path)) {
            nlohmann::json reject = update;
            reject["applicable"] = false;
            reject["reject_reason"] = "unknown_path";
            rejected.push_back(std::move(reject));
            continue;
        }

        const nlohmann::json& schema = schema_by_path[path];
        const nlohmann::json& value = update["value"];
        if (!schema_type_matches(schema, value)) {
            nlohmann::json reject = update;
            reject["applicable"] = false;
            reject["reject_reason"] = "wrong_type";
            rejected.push_back(std::move(reject));
            continue;
        }
        if (!schema_enum_matches(schema, value)) {
            nlohmann::json reject = update;
            reject["applicable"] = false;
            reject["reject_reason"] = "enum_mismatch";
            rejected.push_back(std::move(reject));
            continue;
        }

        nlohmann::json accepted = update;
        accepted["applicable"] = true;
        validated.push_back(std::move(accepted));
    }

    return {
        {"validated_updates", validated},
        {"rejected_updates", rejected},
        {"candidate_count", candidates.size()}
    };
}

} // namespace tile_compile::pi
