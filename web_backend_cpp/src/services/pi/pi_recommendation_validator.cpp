#include "services/pi/pi_recommendation_validator.hpp"

#include "app_state.hpp"
#include "services/pi/pi_schema_utils.hpp"
#include "subprocess_manager.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <vector>

namespace tile_compile::pi {
namespace {

nlohmann::json parse_scalar_value(const nlohmann::json& raw_value) {
    if (!raw_value.is_string()) return raw_value;
    std::string text = raw_value.get<std::string>();
    text.erase(text.begin(), std::find_if(text.begin(), text.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    text.erase(std::find_if(text.rbegin(), text.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), text.end());
    if (text.empty()) return raw_value;

    std::string lowered = text;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (lowered == "true") return true;
    if (lowered == "false") return false;
    if (lowered == "null" || lowered == "~") return nullptr;

    char* end = nullptr;
    errno = 0;
    const double numeric = std::strtod(text.c_str(), &end);
    if (errno == 0 && end != text.c_str() && end && *end == '\0' && std::isfinite(numeric)) {
        if (text.find('.') == std::string::npos &&
            text.find('e') == std::string::npos &&
            text.find('E') == std::string::npos) {
            try {
                return std::stoll(text);
            } catch (...) {
                return numeric;
            }
        }
        return numeric;
    }

    return raw_value;
}

std::optional<nlohmann::json> parse_json_string(const std::string& raw) {
    auto parsed = nlohmann::json::parse(raw, nullptr, false);
    if (parsed.is_discarded()) return std::nullopt;
    return parsed;
}

std::string config_payload_dump(const nlohmann::json& value) {
    return value.dump(2);
}

void set_dotted(nlohmann::json& root, const std::string& dotted_path, const nlohmann::json& value) {
    nlohmann::json* current = &root;
    size_t start = 0;
    while (start < dotted_path.size()) {
        const size_t dot = dotted_path.find('.', start);
        const std::string key = dotted_path.substr(start, dot == std::string::npos ? std::string::npos : dot - start);
        if (key.empty()) return;
        if (dot == std::string::npos) {
            (*current)[key] = value;
            return;
        }
        if (!(*current).contains(key) || !(*current)[key].is_object()) {
            (*current)[key] = nlohmann::json::object();
        }
        current = &(*current)[key];
        start = dot + 1;
    }
}

std::string json_string_value(const nlohmann::json& value, const std::string& fallback = "") {
    if (value.is_string()) return value.get<std::string>();
    if (value.is_boolean()) return value.get<bool>() ? "true" : "false";
    if (value.is_number() || value.is_null()) return value.dump();
    return fallback;
}

std::string json_string_field(const nlohmann::json& object, const char* key, const std::string& fallback = "") {
    if (!object.is_object() || !object.contains(key)) return fallback;
    return json_string_value(object[key], fallback);
}

double json_double_field(const nlohmann::json& object, const char* key, double fallback = 0.0) {
    if (!object.is_object() || !object.contains(key)) return fallback;
    const nlohmann::json& value = object[key];
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

std::string lower_copy(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text;
}

bool contains_any(const std::string& text, std::initializer_list<const char*> needles) {
    const std::string hay = lower_copy(text);
    for (const char* needle : needles) {
        if (hay.find(needle) != std::string::npos) return true;
    }
    return false;
}

const nlohmann::json* get_dotted_ptr(const nlohmann::json& root, const std::string& dotted_path) {
    const nlohmann::json* cur = &root;
    std::istringstream iss(dotted_path);
    std::string part;
    while (std::getline(iss, part, '.')) {
        if (!cur->is_object() || !cur->contains(part)) return nullptr;
        cur = &(*cur)[part];
    }
    return cur;
}

std::optional<std::string> semantic_update_reject_reason(const nlohmann::json& update,
                                                         const nlohmann::json& schema_node,
                                                         const nlohmann::json& base_config,
                                                         const nlohmann::json& pi_context) {
    const std::string path = json_string_field(update, "path");
    const std::string reason = json_string_field(update, "reason");
    const nlohmann::json value = update.contains("value") ? update["value"] : nlohmann::json(nullptr);
    const nlohmann::json catalog = pi_context.contains("parameter_catalog") && pi_context["parameter_catalog"].is_object()
        ? pi_context["parameter_catalog"]
        : build_parameter_catalog(SchemaPathMap{{path, schema_node}}, base_config);
    const nlohmann::json meta = catalog.contains(path) && catalog[path].is_object()
        ? catalog[path]
        : curated_parameter_metadata(path);

    if (!meta.empty()) {
        const bool mentions_schema_limit = contains_any(reason, {"schema max", "schema-declared maximum", "schema maximum", "schema range", "schema recommends", "schema-recommended", "schema default"});
        const bool lacks_schema_max = !schema_node.is_object() || !schema_node.contains("maximum");
        if (mentions_schema_limit && lacks_schema_max &&
            contains_any(reason, {"max", "maximum", "range", "recommended", "recommends"})) {
            return "unsupported_schema_claim";
        }
        if (contains_any(reason, {"schema default", "default is true"}) &&
            meta.contains("cpp_default") && meta["cpp_default"].is_boolean() && !meta["cpp_default"].get<bool>()) {
            return "unsupported_default_claim";
        }
        if (contains_any(reason, {"misconfiguration", "fehlkonfiguration"})) {
            if (const nlohmann::json* cur = get_dotted_ptr(base_config, path)) {
                if (meta.contains("cpp_default") && *cur == meta["cpp_default"]) {
                    return "default_value_misconfiguration_claim";
                }
            }
        }
        if (meta.value("diagnostic_only", false) &&
            contains_any(reason, {"reconstruction quality", "reconstruction", "degrades aqmh reconstruction", "improves diagnostic accuracy", "improves", "degrades"})) {
            return "diagnostic_only_quality_claim";
        }
    }

    if (path == "pcc.k_max" && value.is_number() && value.get<double>() < 1.0) {
        return "pcc_k_max_below_effective_minimum";
    }

    const nlohmann::json facts = pi_context.contains("facts") && pi_context["facts"].is_object()
        ? pi_context["facts"]
        : nlohmann::json::object();
    auto fact_value = [&](const std::string& id) -> const nlohmann::json* {
        if (!facts.contains(id) || !facts[id].is_object() || !facts[id].contains("value")) return nullptr;
        return &facts[id]["value"];
    };

    if (path == "pcc.max_residual_rms" && value.is_number()) {
        const nlohmann::json* status = fact_value("pcc.status");
        const nlohmann::json* residual = fact_value("pcc.residual_rms");
        if (status && status->is_string() && lower_copy(status->get<std::string>()) == "ok" &&
            residual && residual->is_number() &&
            value.get<double>() < residual->get<double>()) {
            return "below_observed_successful_pcc_residual";
        }
    }

    if (path == "validation.max_background_rms_increase_percent" &&
        contains_any(reason, {"any background rms increase", "automatically disable", "auto-disabled", "0 means any", "0.0 means any"})) {
        return "disabled_sentinel_misinterpreted";
    }

    return std::nullopt;
}

} // namespace

nlohmann::json normalize_candidate_updates(const nlohmann::json& analysis) {
    nlohmann::json updates = nlohmann::json::array();
    const nlohmann::json* source = nullptr;
    if (analysis.contains("updates") && analysis["updates"].is_array()) {
        source = &analysis["updates"];
    } else if (analysis.contains("recommendations") && analysis["recommendations"].is_array()) {
        source = &analysis["recommendations"];
    }
    if (!source) return updates;

    for (const auto& item : *source) {
        if (!item.is_object()) continue;
        const std::string path = item.contains("path") && item["path"].is_string()
            ? item["path"].get<std::string>()
            : std::string();
        if (path.empty() || !item.contains("value")) continue;
        nlohmann::json update = {
            {"path", path},
            {"value", parse_scalar_value(item["value"])},
            {"reason", json_string_field(item, "reason", json_string_field(item, "rationale"))},
            {"confidence", json_double_field(item, "confidence", 0.0)},
            {"risk", json_string_field(item, "risk", "unknown")},
        };
        if (item.contains("id") && item["id"].is_string()) update["id"] = item["id"];
        if (item.contains("current_value")) update["current_value"] = item["current_value"];
        if (item.contains("review_required") && item["review_required"].is_boolean()) {
            update["review_required"] = item["review_required"];
        }
        if (item.contains("evidence") && item["evidence"].is_array()) {
            update["evidence"] = nlohmann::json::array();
            for (const auto& evidence : item["evidence"]) {
                update["evidence"].push_back(json_string_value(evidence));
            }
        }
        updates.push_back(std::move(update));
    }
    return updates;
}

nlohmann::json validate_recommendation_updates(const nlohmann::json& candidates,
                                               const SchemaPathMap& schema_paths,
                                               const nlohmann::json& base_config,
                                               const std::shared_ptr<AppState>& state,
                                               const nlohmann::json& pi_context) {
    nlohmann::json validated = nlohmann::json::array();
    nlohmann::json rejected = nlohmann::json::array();
    nlohmann::json patched = base_config;

    for (const auto& update : candidates) {
        const std::string path = json_string_field(update, "path");
        auto schema_it = schema_paths.find(path);
        if (schema_it == schema_paths.end()) {
            nlohmann::json reject = update;
            reject["applicable"] = false;
            reject["reject_reason"] = "unknown_path";
            rejected.push_back(std::move(reject));
            continue;
        }
        const nlohmann::json value = update.contains("value") ? update["value"] : nlohmann::json(nullptr);
        if (!schema_type_matches(schema_it->second, value)) {
            nlohmann::json reject = update;
            reject["applicable"] = false;
            reject["reject_reason"] = "wrong_type";
            rejected.push_back(std::move(reject));
            continue;
        }
        if (!schema_enum_matches(schema_it->second, value)) {
            nlohmann::json reject = update;
            reject["applicable"] = false;
            reject["reject_reason"] = "enum_mismatch";
            rejected.push_back(std::move(reject));
            continue;
        }
        if (const auto semantic_reject = semantic_update_reject_reason(update, schema_it->second, base_config, pi_context)) {
            nlohmann::json reject = update;
            reject["applicable"] = false;
            reject["reject_reason"] = *semantic_reject;
            rejected.push_back(std::move(reject));
            continue;
        }
        set_dotted(patched, path, value);
        nlohmann::json accepted = update;
        accepted["applicable"] = true;
        validated.push_back(std::move(accepted));
    }

    const std::string yaml_text = config_payload_dump(patched);
    SubprocessResult res = run_subprocess({state->runtime.cli_exe, "validate-config", "--stdin"},
                                          state->runtime.project_root.string(),
                                          yaml_text);
    auto validation = parse_json_string(res.stdout_str).value_or(nlohmann::json::object());
    if (res.exit_code != 0 || !validation.value("valid", false)) {
        static const std::vector<std::vector<std::string>> weight_groups = {
            {"global_metrics.weights.background", "global_metrics.weights.gradient", "global_metrics.weights.noise"},
            {"quality_filter.weights.contrast", "quality_filter.weights.fwhm", "quality_filter.weights.roundness"},
        };

        auto group_for_path = [&](const std::string& path) -> int {
            for (int g = 0; g < static_cast<int>(weight_groups.size()); ++g)
                for (const auto& member : weight_groups[g])
                    if (member == path) return g;
            return -1;
        };

        std::map<std::string, nlohmann::json*> path_to_item;
        for (auto& item : validated)
            path_to_item[json_string_field(item, "path")] = &item;

        nlohmann::json surviving = nlohmann::json::array();
        nlohmann::json current_base = base_config;

        for (int g = 0; g < static_cast<int>(weight_groups.size()); ++g) {
            const auto& group = weight_groups[g];
            bool any_in_group = false;
            for (const auto& member : group) if (path_to_item.count(member)) { any_in_group = true; break; }
            if (!any_in_group) continue;
            nlohmann::json trial = current_base;
            bool all_present = true;
            for (const auto& member : group) {
                if (!path_to_item.count(member)) { all_present = false; break; }
                set_dotted(trial, member, (*path_to_item[member])["value"]);
            }
            if (!all_present) continue;
            const std::string trial_yaml = config_payload_dump(trial);
            SubprocessResult vres = run_subprocess({state->runtime.cli_exe, "validate-config", "--stdin"},
                                                   state->runtime.project_root.string(), trial_yaml);
            auto vresult = parse_json_string(vres.stdout_str).value_or(nlohmann::json::object());
            if (vres.exit_code == 0 && vresult.value("valid", false)) {
                current_base = trial;
                for (const auto& member : group) surviving.push_back(*path_to_item[member]);
            } else {
                for (const auto& member : group) {
                    (*path_to_item[member])["applicable"] = false;
                    (*path_to_item[member])["reject_reason"] = "weight_group_validation_failed";
                    rejected.push_back(*path_to_item[member]);
                }
            }
        }

        for (auto& item : validated) {
            const std::string ipath = json_string_field(item, "path");
            if (group_for_path(ipath) >= 0) continue;
            const nlohmann::json ivalue = item.contains("value") ? item["value"] : nlohmann::json(nullptr);
            nlohmann::json trial = current_base;
            set_dotted(trial, ipath, ivalue);
            const std::string trial_yaml = config_payload_dump(trial);
            SubprocessResult vres = run_subprocess({state->runtime.cli_exe, "validate-config", "--stdin"},
                                                   state->runtime.project_root.string(), trial_yaml);
            auto vresult = parse_json_string(vres.stdout_str).value_or(nlohmann::json::object());
            if (vres.exit_code != 0 || !vresult.value("valid", false)) {
                item["applicable"] = false;
                item["reject_reason"] = "config_validation_failed";
                rejected.push_back(item);
            } else {
                current_base = trial;
                surviving.push_back(item);
            }
        }
        validated = surviving;
        const std::string final_yaml = config_payload_dump(current_base);
        return {
            {"validated_updates", validated},
            {"rejected_updates", rejected},
            {"candidate_count", candidates.size()},
            {"patched_config", current_base},
            {"patched_config_yaml", final_yaml},
            {"validation", validation},
        };
    }

    return {
        {"validated_updates", validated},
        {"rejected_updates", rejected},
        {"candidate_count", candidates.size()},
        {"patched_config", patched},
        {"patched_config_yaml", yaml_text},
        {"validation", validation},
    };
}

} // namespace tile_compile::pi
