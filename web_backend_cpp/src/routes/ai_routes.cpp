#include "routes/ai_routes.hpp"
#include "app_state.hpp"
#include "routes/route_utils.hpp"
#include "subprocess_manager.hpp"
#include "services/ai_service.hpp"
#include "services/scan_summary.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <mutex>
#include <set>
#include <sstream>
#include <system_error>

using namespace tile_compile::routes;

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

fs::path ai_config_path(const std::shared_ptr<AppState>& state) {
    return state->runtime.runtime_dir / "ai_scan_config.json";
}

json read_ai_config_file(const std::shared_ptr<AppState>& state) {
    const fs::path path = ai_config_path(state);
    std::ifstream in(path);
    if (!in) return json::object();
    auto parsed = json::parse(in, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) return json::object();
    return parsed;
}

bool write_ai_config_file(const std::shared_ptr<AppState>& state, const json& config) {
    const fs::path path = ai_config_path(state);
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    if (ec) return false;
    std::ofstream out(path);
    if (!out) return false;
    out << config.dump(2);
    out << '\n';
    return static_cast<bool>(out);
}

std::vector<fs::path> list_json_files(const fs::path& dir, bool newest_first = true) {
    std::vector<fs::path> files;
    if (!fs::is_directory(dir)) return files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (name.size() < 5 || name.substr(name.size() - 5) != ".json") continue;
        files.push_back(entry.path());
    }
    if (newest_first) {
        std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
            return a.filename().string() > b.filename().string();
        });
    } else {
        std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
            return a.filename().string() < b.filename().string();
        });
    }
    return files;
}

std::optional<json> parse_json_file(const fs::path& path) {
    std::ifstream ifs(path);
    if (!ifs) return std::nullopt;
    auto parsed = json::parse(ifs, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) return std::nullopt;
    return parsed;
}

json current_ai_config_json(const std::shared_ptr<AppState>& state) {
    const json file_config = read_ai_config_file(state);
    json memory_config = json::object();
    {
        std::lock_guard<std::mutex> lk(state->state_mutex);
        if (state->ui_state.contains("ai") && state->ui_state["ai"].contains("scan_analysis") &&
            state->ui_state["ai"]["scan_analysis"].is_object()) {
            memory_config = state->ui_state["ai"]["scan_analysis"];
        }
    }
    json merged = tile_compile::ai::merge_ai_config_json(
        tile_compile::ai::ai_config_to_json(tile_compile::ai::default_ai_config(state->runtime)),
        file_config,
        state->runtime);
    return tile_compile::ai::merge_ai_config_json(merged, memory_config, state->runtime);
}

tile_compile::ai::AiConfig current_ai_config(const std::shared_ptr<AppState>& state) {
    return tile_compile::ai::ai_config_from_json(current_ai_config_json(state), state->runtime);
}

bool store_ai_config_json(const std::shared_ptr<AppState>& state, const json& config) {
    {
        std::lock_guard<std::mutex> lk(state->state_mutex);
        if (!state->ui_state.is_object()) state->ui_state = json::object();
        if (!state->ui_state.contains("ai") || !state->ui_state["ai"].is_object()) {
            state->ui_state["ai"] = json::object();
        }
        state->ui_state["ai"]["scan_analysis"] = config;
    }
    return write_ai_config_file(state, config);
}

json sidecar_unavailable_payload(const std::exception& e) {
    return {
        {"available", false},
        {"error", {
            {"code", "AI_AGENT_UNAVAILABLE"},
            {"message", e.what()}
        }}
    };
}

json latest_analysis(const InMemoryJobStore& store) {
    for (const auto& job : store.list(100)) {
        if (job.type == "scan_ai_analysis") {
            return {
                {"has_analysis", true},
                {"analysis_id", job.job_id},
                {"job", job_to_json(job)},
                {"data", job.data},
            };
        }
    }
    return {{"has_analysis", false}};
}


json parse_scalar_value(const json& raw_value) {
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

std::string json_string_value(const json& value, const std::string& fallback = "") {
    if (value.is_string()) return value.get<std::string>();
    if (value.is_boolean()) return value.get<bool>() ? "true" : "false";
    if (value.is_number() || value.is_null()) return value.dump();
    return fallback;
}

std::string json_string_field(const json& object, const char* key, const std::string& fallback = "") {
    if (!object.is_object() || !object.contains(key)) return fallback;
    return json_string_value(object[key], fallback);
}

double json_double_field(const json& object, const char* key, double fallback = 0.0) {
    if (!object.is_object() || !object.contains(key)) return fallback;
    const json& value = object[key];
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

int infer_frame_count(const json& scan_result, const json& scan_metrics = json::object()) {
    if (scan_metrics.is_object()) {
        if (scan_metrics.contains("frames_total") && scan_metrics["frames_total"].is_number()) {
            return scan_metrics["frames_total"].get<int>();
        }
        if (scan_metrics.contains("frame_count") && scan_metrics["frame_count"].is_number()) {
            return scan_metrics["frame_count"].get<int>();
        }
    }
    if (scan_result.is_object()) {
        if (scan_result.contains("frames_detected") && scan_result["frames_detected"].is_number()) {
            return scan_result["frames_detected"].get<int>();
        }
        if (scan_result.contains("frames_total") && scan_result["frames_total"].is_number()) {
            return scan_result["frames_total"].get<int>();
        }
        if (scan_result.contains("frames") && scan_result["frames"].is_array()) {
            return static_cast<int>(scan_result["frames"].size());
        }
    }
    return 0;
}

std::optional<json> load_base_config(const std::shared_ptr<AppState>& state,
                                     const json& body,
                                     fs::path& target,
                                     std::optional<crow::response>& error) {
    target = body.contains("path") && body["path"].is_string()
        ? fs::path(body["path"].get<std::string>())
        : state->runtime.default_config_path;
    if (auto err = validate_path(state, target, false)) {
        error = std::move(*err);
        return std::nullopt;
    }

    if (body.contains("yaml") && body["yaml"].is_string() && !body["yaml"].get<std::string>().empty()) {
        auto parsed = parse_yaml_text(body["yaml"].get<std::string>());
        if (!parsed) {
            error = err_resp("BAD_REQUEST", "Config parse error: invalid YAML", 400);
            return std::nullopt;
        }
        if (!parsed->is_object()) {
            error = err_resp("BAD_REQUEST", "base YAML must be a mapping", 400);
            return std::nullopt;
        }
        return *parsed;
    }
    if (body.contains("config") && body["config"].is_object()) {
        return json(body["config"]);
    }
    if (body.contains("base_config") && body["base_config"].is_object()) {
        return json(body["base_config"]);
    }
    auto parsed = parse_yaml_file(target);
    if (parsed && parsed->is_object()) return *parsed;
    return json::object();
}

struct SchemaInfo {
    std::map<std::string, json> paths;
};

void collect_schema_paths(const json& schema, const std::string& prefix, SchemaInfo& out) {
    if (!schema.is_object()) return;
    if (!prefix.empty()) out.paths[prefix] = schema;
    if (!schema.contains("properties") || !schema["properties"].is_object()) return;
    for (auto it = schema["properties"].begin(); it != schema["properties"].end(); ++it) {
        const std::string path = prefix.empty() ? it.key() : prefix + "." + it.key();
        collect_schema_paths(it.value(), path, out);
    }
}

std::optional<SchemaInfo> load_schema_info(const std::shared_ptr<AppState>& state, std::string& error_message) {
    SubprocessResult res = run_subprocess({state->runtime.cli_exe, "get-schema"}, state->runtime.project_root.string());
    auto parsed = parse_json_string(res.stdout_str);
    if (res.exit_code != 0 || !parsed || !parsed->is_object()) {
        error_message = "failed to fetch config schema";
        return std::nullopt;
    }
    SchemaInfo info;
    collect_schema_paths(*parsed, "", info);
    return info;
}

bool schema_type_matches(const json& schema, const json& value) {
    if (!schema.is_object() || !schema.contains("type")) return true;

    std::vector<std::string> types;
    if (schema["type"].is_string()) {
        types.push_back(schema["type"].get<std::string>());
    } else if (schema["type"].is_array()) {
        for (const auto& t : schema["type"]) {
            if (t.is_string()) types.push_back(t.get<std::string>());
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

bool schema_enum_matches(const json& schema, const json& value) {
    if (!schema.is_object() || !schema.contains("enum") || !schema["enum"].is_array()) return true;
    return std::find(schema["enum"].begin(), schema["enum"].end(), value) != schema["enum"].end();
}

json normalize_candidate_updates(const json& analysis) {
    json updates = json::array();
    const json* source = nullptr;
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
        json update = {
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
            update["evidence"] = json::array();
            for (const auto& evidence : item["evidence"]) {
                update["evidence"].push_back(json_string_value(evidence));
            }
        }
        updates.push_back(std::move(update));
    }
    return updates;
}

json validate_updates_against_schema(const json& candidates,
                                     const SchemaInfo& schema,
                                     const json& base_config,
                                     const std::shared_ptr<AppState>& state) {
    json validated = json::array();
    json rejected = json::array();
    json patched = base_config;

    for (const auto& update : candidates) {
        const std::string path = json_string_field(update, "path");
        auto schema_it = schema.paths.find(path);
        if (schema_it == schema.paths.end()) {
            json reject = update;
            reject["applicable"] = false;
            reject["reject_reason"] = "unknown_path";
            rejected.push_back(std::move(reject));
            continue;
        }
        const json value = update.contains("value") ? update["value"] : json(nullptr);
        if (!schema_type_matches(schema_it->second, value)) {
            json reject = update;
            reject["applicable"] = false;
            reject["reject_reason"] = "wrong_type";
            rejected.push_back(std::move(reject));
            continue;
        }
        if (!schema_enum_matches(schema_it->second, value)) {
            json reject = update;
            reject["applicable"] = false;
            reject["reject_reason"] = "enum_mismatch";
            rejected.push_back(std::move(reject));
            continue;
        }
        set_dotted(patched, path, value);
        json accepted = update;
        accepted["applicable"] = true;
        validated.push_back(std::move(accepted));
    }

    // Validate the full patch first (fast path)
    const std::string yaml_text = yaml_dump(patched);
    SubprocessResult res = run_subprocess({state->runtime.cli_exe, "validate-config", "--stdin"},
                                          state->runtime.project_root.string(),
                                          yaml_text);
    auto validation = parse_json_string(res.stdout_str).value_or(json::object());
    if (res.exit_code != 0 || !validation.value("valid", false)) {
        // Weight groups that must be validated together (members must sum to 1.0)
        static const std::vector<std::vector<std::string>> weight_groups = {
            {"global_metrics.weights.background", "global_metrics.weights.gradient", "global_metrics.weights.noise"},
            {"quality_filter.weights.contrast", "quality_filter.weights.fwhm", "quality_filter.weights.roundness"},
        };

        // Determine which validated paths belong to a weight group
        auto group_for_path = [&](const std::string& p) -> int {
            for (int g = 0; g < (int)weight_groups.size(); ++g)
                for (const auto& m : weight_groups[g])
                    if (m == p) return g;
            return -1;
        };

        // Build set of paths present in validated updates
        std::map<std::string, json*> path_to_item;
        for (auto& item : validated)
            path_to_item[json_string_field(item, "path")] = &item;

        std::set<int> groups_attempted;
        json surviving = json::array();
        json current_base = base_config;

        // First pass: try each complete weight group together
        for (int g = 0; g < (int)weight_groups.size(); ++g) {
            const auto& grp = weight_groups[g];
            bool any_in_group = false;
            for (const auto& m : grp) if (path_to_item.count(m)) { any_in_group = true; break; }
            if (!any_in_group) continue;
            groups_attempted.insert(g);
            json trial = current_base;
            bool all_present = true;
            for (const auto& m : grp) {
                if (!path_to_item.count(m)) { all_present = false; break; }
                set_dotted(trial, m, (*path_to_item[m])["value"]);
            }
            if (!all_present) continue; // partial group – handled in individual pass
            const std::string trial_yaml = yaml_dump(trial);
            SubprocessResult vres = run_subprocess({state->runtime.cli_exe, "validate-config", "--stdin"},
                                                   state->runtime.project_root.string(), trial_yaml);
            auto vresult = parse_json_string(vres.stdout_str).value_or(json::object());
            if (vres.exit_code == 0 && vresult.value("valid", false)) {
                current_base = trial;
                for (const auto& m : grp) surviving.push_back(*path_to_item[m]);
            } else {
                for (const auto& m : grp) {
                    (*path_to_item[m])["applicable"] = false;
                    (*path_to_item[m])["reject_reason"] = "weight_group_validation_failed";
                    rejected.push_back(*path_to_item[m]);
                }
            }
        }

        // Second pass: validate remaining (non-weight-group) updates individually
        for (auto& item : validated) {
            const std::string ipath = json_string_field(item, "path");
            if (group_for_path(ipath) >= 0) continue; // already handled above
            const json ivalue = item.contains("value") ? item["value"] : json(nullptr);
            json trial = current_base;
            set_dotted(trial, ipath, ivalue);
            const std::string trial_yaml = yaml_dump(trial);
            SubprocessResult vres = run_subprocess({state->runtime.cli_exe, "validate-config", "--stdin"},
                                                   state->runtime.project_root.string(), trial_yaml);
            auto vresult = parse_json_string(vres.stdout_str).value_or(json::object());
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
        // Re-generate final yaml from surviving updates
        const std::string final_yaml = yaml_dump(current_base);
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

json selected_validated_updates(const json& data, const json& body) {
    json selected = json::array();
    if (!data.contains("validated_updates") || !data["validated_updates"].is_array()) return selected;

    std::set<std::string> selected_paths;
    if (body.contains("selected_paths") && body["selected_paths"].is_array()) {
        for (const auto& path : body["selected_paths"]) {
            if (path.is_string()) selected_paths.insert(path.get<std::string>());
        }
    }

    if (selected_paths.empty() && body.value("apply_all", true) == false) return selected;
    for (const auto& update : data["validated_updates"]) {
        const std::string path = json_string_field(update, "path");
        if (!selected_paths.empty() && selected_paths.find(path) == selected_paths.end()) continue;
        selected.push_back(update);
    }
    return selected;
}

json scan_result_from_request_or_latest(const std::shared_ptr<AppState>& state, const json& body) {
    if (body.contains("scan_result") && body["scan_result"].is_object()) {
        if (body["scan_result"].contains("has_scan") && !body["scan_result"].value("has_scan", false)) {
            return json(nullptr);
        }
        return body["scan_result"];
    }
    const std::string scan_job_id = body.value("scan_job_id", std::string());
    std::optional<Job> job = scan_job_id.empty()
        ? latest_scan_job(state->job_store)
        : state->job_store.get(scan_job_id);
    if (!job) return json(nullptr);
    if (job->data.is_object() && job->data.contains("result") && job->data["result"].is_object()) {
        return job->data["result"];
    }
    if (job->state != JobState::ok && job->state != JobState::error) return json(nullptr);
    if (!job->data.is_object() ||
        (!job->data.contains("frames_detected") &&
         !job->data.contains("frames") &&
         !job->data.contains("per_dir_results") &&
         !job->data.contains("color_mode"))) {
        return json(nullptr);
    }
    return job->data;
}

fs::path ai_analyses_dir(const std::shared_ptr<AppState>& state) {
    return ".ai_analyses";
}

json extract_scan_metadata(const json& scan_result) {
    json meta = json::object();
    if (!scan_result.is_object()) return meta;
    auto str = [&](const std::string& key) {
        if (scan_result.contains(key) && scan_result[key].is_string())
            meta[key] = scan_result[key];
    };
    auto num = [&](const std::string& key) {
        if (scan_result.contains(key) && scan_result[key].is_number())
            meta[key] = scan_result[key];
    };
    str("color_mode");
    str("bayer_pattern");
    str("input_path");
    num("errors_total");
    num("image_width");
    num("image_height");

    // Prefer frames_detected (authoritative) over frames array length (may be truncated)
    if (scan_result.contains("frames_detected") && scan_result["frames_detected"].is_number()) {
        meta["frame_count"] = scan_result["frames_detected"];
    } else if (scan_result.contains("frames_total") && scan_result["frames_total"].is_number()) {
        meta["frame_count"] = scan_result["frames_total"];
    } else if (scan_result.contains("frames") && scan_result["frames"].is_array()) {
        meta["frame_count"] = scan_result["frames"].size();
    }

    // Extract per-frame metadata from first frame
    if (scan_result.contains("frames") && scan_result["frames"].is_array() && !scan_result["frames"].empty()) {
        const auto& f = scan_result["frames"][0];
        if (f.is_object()) {
            auto fstr = [&](const std::string& key) { if (f.contains(key) && f[key].is_string() && !meta.contains(key)) meta[key] = f[key]; };
            auto fnum = [&](const std::string& key) { if (f.contains(key) && f[key].is_number() && !meta.contains(key)) meta[key] = f[key]; };
            fstr("target");
            fstr("camera");
            fstr("telescope");
            fnum("exposure_seconds");
            fnum("gain");
            fnum("image_width");
            fnum("image_height");
            fnum("temperature_c");
        }
    }
    return meta;
}

json compact_scan_metrics_for_analysis_context(const json& scan_metrics) {
    json out = json::object();
    if (!scan_metrics.is_object()) return out;
    for (const std::string key : {
             "ok", "sample_count", "frames_total", "frames_metrics_total",
             "frames_metrics_truncated", "aggregate", "sampling", "session_geometry"
         }) {
        if (scan_metrics.contains(key)) out[key] = scan_metrics[key];
    }
    if (scan_metrics.contains("frames") && scan_metrics["frames"].is_array()) {
        out["frames"] = json::array();
        for (size_t i = 0; i < scan_metrics["frames"].size(); ++i) {
            const auto& frame = scan_metrics["frames"][i];
            if (!frame.is_object()) continue;
            json item = json::object();
            for (const std::string key : {
                     "index", "frame_index", "file", "filename",
                     "fwhm", "background", "noise", "gradient_energy", "sky_gradient", "roundness",
                     "star_count", "header"
                 }) {
                if (frame.contains(key)) item[key] = frame[key];
            }
            out["frames"].push_back(std::move(item));
        }
        out["frames_context_total"] = scan_metrics["frames"].size();
        out["frames_context_truncated"] = false;
    }
    return out;
}

json build_analysis_context(const json& scan_result, const json& request_or_body) {
    json context = {
        {"schema_version", "pi.scan-analysis-context.v1"},
        {"frame_count", infer_frame_count(scan_result, request_or_body.value("scan_metrics", json::object()))},
    };
    if (request_or_body.contains("scan_metrics") && request_or_body["scan_metrics"].is_object()) {
        context["scan_metrics"] = compact_scan_metrics_for_analysis_context(request_or_body["scan_metrics"]);
    }
    if (request_or_body.contains("base_config") && request_or_body["base_config"].is_object()) {
        context["base_config"] = request_or_body["base_config"];
    }
    if (request_or_body.contains("config_schema") && request_or_body["config_schema"].is_object()) {
        context["config_schema"] = request_or_body["config_schema"];
    }
    if (request_or_body.contains("schema_version") && request_or_body["schema_version"].is_string()) {
        context["request_schema_version"] = request_or_body["schema_version"];
    }
    if (request_or_body.contains("model") && request_or_body["model"].is_string()) {
        context["requested_model"] = request_or_body["model"];
    }
    if (request_or_body.contains("allowed_config_paths") && request_or_body["allowed_config_paths"].is_array()) {
        context["allowed_config_paths"] = request_or_body["allowed_config_paths"];
    }
    if (request_or_body.contains("session_context") && request_or_body["session_context"].is_object()) {
        context["session_context"] = request_or_body["session_context"];
    }
    return context;
}

std::string persist_analysis(const std::shared_ptr<AppState>& state,
                             const json& analysis,
                             const json& scan_meta,
                             const json& analysis_context = json::object()) {
    std::error_code ec;
    fs::path dir = ai_analyses_dir(state);
    fs::create_directories(dir, ec);
    if (ec) return "";

    // Build filename: target_YYYYMMDD_HHMMSS.json
    std::string target = scan_meta.value("target", std::string("unknown"));
    // Sanitize target for filename
    for (auto& c : target) { if (!std::isalnum(c) && c != '-' && c != '_') c = '_'; }

    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    struct tm tm_now{};
#ifdef _WIN32
    gmtime_s(&tm_now, &time_t_now);
#else
    gmtime_r(&time_t_now, &tm_now);
#endif
    char ts_buf[32];
    std::strftime(ts_buf, sizeof(ts_buf), "%Y%m%d_%H%M%S", &tm_now);

    std::string filename = target + "_" + ts_buf + ".json";
    fs::path filepath = dir / filename;

    // Merge analysis with scan metadata
    json persisted = analysis;
    persisted["scan_metadata"] = scan_meta;
    if (analysis_context.is_object() && !analysis_context.empty()) {
        persisted["analysis_context"] = analysis_context;
    }
    persisted["persisted_at"] = std::string(ts_buf);
    persisted["has_analysis"] = true;

    std::ofstream ofs(filepath);
    if (!ofs) return "";
    ofs << persisted.dump(2);
    ofs.close();
    return filepath.string();
}

/// @brief Searches persisted analyses for one that matches the given input_path (and frame_count if > 0).
/// Returns the full analysis JSON with has_analysis=true, or {has_analysis:false} if none found.
json find_cached_analysis(const std::shared_ptr<AppState>& state,
                          const std::string& input_path,
                          int frame_count = 0) {
    if (input_path.empty()) return json({{"has_analysis", false}});
    const fs::path dir = ai_analyses_dir(state);
    for (const auto& path : list_json_files(dir, true)) {
        auto parsed_opt = parse_json_file(path);
        if (!parsed_opt) continue;
        auto& parsed = *parsed_opt;
        if (!parsed.contains("scan_metadata") || !parsed["scan_metadata"].is_object()) continue;
        const auto& meta = parsed["scan_metadata"];
        if (!meta.contains("input_path") || !meta["input_path"].is_string()) continue;
        if (meta["input_path"].get<std::string>() != input_path) continue;
        if (frame_count > 0 && meta.contains("frame_count") && meta["frame_count"].is_number()) {
            if (meta["frame_count"].get<int>() != frame_count) continue;
        }
        parsed["has_analysis"] = true;
        parsed["from_cache"] = true;
        return parsed;
    }
    return json({{"has_analysis", false}});
}

json load_latest_persisted_analysis(const std::shared_ptr<AppState>& state) {
    const fs::path dir = ai_analyses_dir(state);
    const auto files = list_json_files(dir, true);
    if (files.empty()) return json({{"has_analysis", false}});

    auto parsed_opt = parse_json_file(files.front());
    if (!parsed_opt) return json({{"has_analysis", false}});
    (*parsed_opt)["has_analysis"] = true;
    return *parsed_opt;
}

json list_persisted_analyses(const std::shared_ptr<AppState>& state, int limit = 50) {
    const fs::path dir = ai_analyses_dir(state);
    json items = json::array();
    auto files = list_json_files(dir, true);
    if ((int)files.size() > limit) files.resize(limit);

    for (const auto& path : files) {
        auto parsed_opt = parse_json_file(path);
        if (!parsed_opt) continue;
        const auto& parsed = *parsed_opt;
        // Return compact summary only (no full recommendations)
        json entry = json::object();
        entry["filename"] = path.filename().string();
        if (parsed.contains("analysis_id")) entry["analysis_id"] = parsed["analysis_id"];
        if (parsed.contains("scan_metadata")) entry["scan_metadata"] = parsed["scan_metadata"];
        if (parsed.contains("confidence")) entry["confidence"] = parsed["confidence"];
        if (parsed.contains("summary")) entry["summary"] = parsed["summary"];
        if (parsed.contains("validated_count")) entry["validated_count"] = parsed["validated_count"];
        if (parsed.contains("rejected_count")) entry["rejected_count"] = parsed["rejected_count"];
        if (parsed.contains("model")) entry["model"] = parsed["model"];
        if (parsed.contains("provider")) entry["provider"] = parsed["provider"];
        if (parsed.contains("persisted_at")) entry["persisted_at"] = parsed["persisted_at"];
        items.push_back(std::move(entry));
    }
    return items;
}

} // namespace

void tile_compile::routes::register_ai_routes(CrowApp& app, std::shared_ptr<AppState> state) {
    CROW_ROUTE(app, "/api/ai/config").methods("GET"_method)
    ([state]() {
        return json_resp(current_ai_config_json(state));
    });

    CROW_ROUTE(app, "/api/ai/config").methods("PATCH"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const json merged = tile_compile::ai::merge_ai_config_json(
            current_ai_config_json(state),
            *body,
            state->runtime);
        if (!store_ai_config_json(state, merged)) {
            return err_resp("AI_CONFIG_SAVE_FAILED", "AI config could not be written", 500);
        }
        state->ui_event_store.push("ai.config", "ai.config", {{"enabled", merged.value("enabled", false)}});
        return json_resp(merged);
    });

    CROW_ROUTE(app, "/api/ai/models").methods("GET"_method)
    ([state]() {
        try {
            tile_compile::ai::AiSidecarClient client(current_ai_config(state));
            return json_resp(client.get("/models"));
        } catch (const std::exception& e) {
            return json_resp(sidecar_unavailable_payload(e));
        }
    });

    CROW_ROUTE(app, "/api/ai/test").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        try {
            tile_compile::ai::AiSidecarClient client(current_ai_config(state));
            return json_resp(client.post("/test", *body));
        } catch (const std::exception& e) {
            return json_resp(sidecar_unavailable_payload(e), 502);
        }
    });

    CROW_ROUTE(app, "/api/ai/auth").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        try {
            tile_compile::ai::AiSidecarClient client(current_ai_config(state));
            return json_resp(client.post("/auth", *body));
        } catch (const std::exception& e) {
            return json_resp(sidecar_unavailable_payload(e), 502);
        }
    });

    CROW_ROUTE(app, "/api/ai/auth/<string>").methods("DELETE"_method)
    ([state](const crow::request&, std::string provider) {
        try {
            tile_compile::ai::AiSidecarClient client(current_ai_config(state));
            return json_resp(client.del("/auth/" + provider));
        } catch (const std::exception& e) {
            return json_resp(sidecar_unavailable_payload(e), 502);
        }
    });

    CROW_ROUTE(app, "/api/scan/analysis").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);

        auto config = current_ai_config(state);
        const bool force = body->value("force", false);
        if (!config.enabled && !force) {
            return json_resp({
                {"status", "AI_DISABLED"},
                {"enabled", false},
                {"message", "AI scan analysis is disabled"}
            });
        }

        json scan_result = scan_result_from_request_or_latest(state, *body);
        if (scan_result.is_null()) {
            return err_resp("NO_SCAN", "No scan result available for AI analysis", 400);
        }

        // Return cached analysis if one exists for the same input (unless force=true)
        if (!force) {
            const std::string ip = json_string_field(scan_result, "input_path");
            int fc = 0;
            if (scan_result.contains("frames_detected") && scan_result["frames_detected"].is_number())
                fc = scan_result["frames_detected"].get<int>();
            else if (scan_result.contains("frames_total") && scan_result["frames_total"].is_number())
                fc = scan_result["frames_total"].get<int>();
            json cached = find_cached_analysis(state, ip, fc);
            if (cached.value("has_analysis", false)) {
                return json_resp(cached);
            }
            // No cache found and force=false: this endpoint is used as a cache probe only.
            // Do NOT start a real AI analysis here; return immediately so the caller can
            // fall through to the streaming endpoint instead.
            return json_resp({{"has_analysis", false}, {"from_cache", false}});
        }

        fs::path target_config_path;
        std::optional<crow::response> config_error;
        std::optional<json> base_config = load_base_config(state, *body, target_config_path, config_error);
        if (!base_config) return std::move(*config_error);

        std::string schema_error;
        std::optional<SchemaInfo> schema = load_schema_info(state, schema_error);
        if (!schema) return err_resp("SCHEMA_UNAVAILABLE", schema_error, 502);

        json allowed_paths = json::array();
        json config_schema = json::object();
        for (const auto& [path, schema_node] : schema->paths) {
            allowed_paths.push_back(path);
            if (schema_node.is_object()) {
                json entry = json::object();
                if (schema_node.contains("type")) entry["type"] = schema_node["type"];
                if (schema_node.contains("enum")) entry["enum"] = schema_node["enum"];
                if (schema_node.contains("description") && schema_node["description"].is_string()) {
                    entry["desc"] = schema_node["description"];
                }
                if (schema_node.contains("default")) entry["default"] = schema_node["default"];
                if (schema_node.contains("minimum")) entry["minimum"] = schema_node["minimum"];
                if (schema_node.contains("maximum")) entry["maximum"] = schema_node["maximum"];
                if (!entry.empty()) config_schema[path] = std::move(entry);
            }
        }

        json request_payload;
        try {
            request_payload = {
                {"schema_version", "pi.scan-analysis.request.v1"},
                {"scan_result", scan_result},
                {"base_config", *base_config},
                {"allowed_config_paths", body->value("allowed_config_paths", allowed_paths)},
                {"config_schema", config_schema},
                {"model", json_string_field(*body, "model", config.model)},
                {"send_paths", config.send_paths},
                {"force", force},
            };
            if (body->contains("scan_metrics") && (*body)["scan_metrics"].is_object()) {
                request_payload["scan_metrics"] = (*body)["scan_metrics"];
            }
            if (body->contains("session_context") && (*body)["session_context"].is_object()) {
                request_payload["session_context"] = (*body)["session_context"];
            }
        } catch (const nlohmann::json::type_error& e) {
            return err_resp("JSON_TYPE_ERROR", std::string("Failed to build request payload: ") + e.what(), 500);
        } catch (const std::exception& e) {
            return err_resp("PAYLOAD_BUILD_ERROR", std::string("Failed to build request payload: ") + e.what(), 500);
        }

        try {
            tile_compile::ai::AiSidecarClient client(config);
            json analysis = client.post("/analyze", request_payload);
            if (!analysis.is_object() || json_string_field(analysis, "schema_version") != "pi.scan-analysis.v1") {
                return err_resp("INVALID_AI_RESPONSE", "AI sidecar returned an invalid analysis payload", 502);
            }
            const json candidates = normalize_candidate_updates(analysis);
            const json validation = validate_updates_against_schema(candidates, *schema, *base_config, state);
            analysis["updates"] = candidates;
            analysis["validated_updates"] = validation["validated_updates"];
            analysis["rejected_updates"] = validation["rejected_updates"];
            analysis["validation"] = validation["validation"];
            analysis["candidate_count"] = validation["candidate_count"];
            analysis["validated_count"] = validation["validated_updates"].size();
            analysis["rejected_count"] = validation["rejected_updates"].size();
            analysis["config_path"] = target_config_path.string();
            analysis["analysis_context"] = build_analysis_context(scan_result, request_payload);
            if (validation.contains("patched_config") && validation["patched_config"].is_object())
                analysis["analysis_context"]["patched_config"] = validation["patched_config"];
            if (validation.contains("patched_config_yaml") && validation["patched_config_yaml"].is_string())
                analysis["analysis_context"]["patched_config_yaml"] = validation["patched_config_yaml"];
            const std::string job_id = state->job_store.create("scan_ai_analysis");
            json job_data = analysis;
            job_data["analysis_id"] = job_id;
            job_data["model"] = request_payload["model"];
            job_data["provider"] = config.provider;
            state->job_store.update_state(job_id, JobState::ok, job_data);
            analysis["analysis_id"] = job_id;
            persist_analysis(state, analysis, extract_scan_metadata(scan_result),
                             analysis["analysis_context"]);
            return json_resp(analysis);
        } catch (const std::exception& e) {
            return json_resp(sidecar_unavailable_payload(e), 502);
        }
    });

    CROW_ROUTE(app, "/api/scan/analysis/latest").methods("GET"_method)
    ([state]() {
        json result = latest_analysis(state->job_store);
        if (!result.is_object() || result.value("has_analysis", false) == false) {
            result = load_latest_persisted_analysis(state);
        }
        return json_resp(result);
    });

    // Store a pre-computed AI analysis (from sidecar streaming) and validate it
    CROW_ROUTE(app, "/api/scan/analysis/store").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);

        json analysis = body->value("analysis", json::object());
        if (!analysis.is_object() || analysis.empty()) {
            return err_resp("BAD_REQUEST", "analysis object is required", 400);
        }

        const std::string schema_ver = analysis.value("schema_version", analysis.value("schema", std::string()));
        if (schema_ver != "pi.scan-analysis.v1") {
            return err_resp("INVALID_SCHEMA", "analysis must have schema_version pi.scan-analysis.v1", 400);
        }

        fs::path target_config_path;
        std::optional<crow::response> config_error;
        std::optional<json> base_config = load_base_config(state, *body, target_config_path, config_error);
        if (!base_config) return std::move(*config_error);

        std::string schema_error;
        std::optional<SchemaInfo> schema = load_schema_info(state, schema_error);
        if (!schema) return err_resp("SCHEMA_UNAVAILABLE", schema_error, 502);

        json scan_result = scan_result_from_request_or_latest(state, *body);
        const json scan_metrics = body->contains("scan_metrics") && (*body)["scan_metrics"].is_object()
            ? (*body)["scan_metrics"]
            : json::object();
        const json candidates = normalize_candidate_updates(analysis);
        const json validation = validate_updates_against_schema(candidates, *schema, *base_config, state);
        analysis["updates"] = candidates;
        analysis["validated_updates"] = validation["validated_updates"];
        analysis["rejected_updates"] = validation["rejected_updates"];
        analysis["validation"] = validation["validation"];
        analysis["candidate_count"] = validation["candidate_count"];
        analysis["validated_count"] = validation["validated_updates"].size();
        analysis["rejected_count"] = validation["rejected_updates"].size();
        analysis["config_path"] = target_config_path.string();
        analysis["analysis_context"] = build_analysis_context(scan_result, *body);
        // Override base_config with the parsed version so the persisted context
        // contains a proper JSON object, not a raw YAML string.
        analysis["analysis_context"]["base_config"] = *base_config;
        if (validation.contains("patched_config") && validation["patched_config"].is_object())
            analysis["analysis_context"]["patched_config"] = validation["patched_config"];
        if (validation.contains("patched_config_yaml") && validation["patched_config_yaml"].is_string())
            analysis["analysis_context"]["patched_config_yaml"] = validation["patched_config_yaml"];

        const std::string job_id = state->job_store.create("scan_ai_analysis");
        json job_data = analysis;
        job_data["analysis_id"] = job_id;
        job_data["model"] = body->value("model", std::string());
        job_data["provider"] = body->value("provider", std::string());
        state->job_store.update_state(job_id, JobState::ok, job_data);
        analysis["analysis_id"] = job_id;
        analysis["has_analysis"] = true;
        json scan_meta = body->contains("scan_metadata") ? (*body)["scan_metadata"] : json::object();
        if (scan_meta.empty()) {
            scan_meta = extract_scan_metadata(scan_result);
        }
        persist_analysis(state, analysis, scan_meta, analysis["analysis_context"]);
        return json_resp(analysis);
    });

    // Analysis history endpoint
    CROW_ROUTE(app, "/api/scan/analysis/history").methods("GET"_method)
    ([state](const crow::request& req) {
        int limit = parse_int_param(req, "limit", 50);
        return json_resp({{"items", list_persisted_analyses(state, limit)}});
    });

    // Load a single persisted analysis by filename
    CROW_ROUTE(app, "/api/scan/analysis/history/<string>").methods("GET"_method)
    ([state](const crow::request&, std::string filename) {
        // Sanitize: no path traversal
        if (filename.find('/') != std::string::npos || filename.find("..") != std::string::npos) {
            return err_resp("BAD_REQUEST", "Invalid filename", 400);
        }
        if (filename.size() < 5 || filename.substr(filename.size() - 5) != ".json") {
            filename += ".json";
        }
        fs::path filepath = ai_analyses_dir(state) / filename;
        std::ifstream ifs(filepath);
        if (!ifs) return err_resp("NOT_FOUND", "Analysis not found", 404);
        auto parsed = json::parse(ifs, nullptr, false);
        if (parsed.is_discarded() || !parsed.is_object()) {
            return err_resp("PARSE_ERROR", "Failed to parse analysis file", 500);
        }
        parsed["has_analysis"] = true;
        return json_resp(parsed);
    });

    // SSE streaming endpoint for live analysis progress
    CROW_ROUTE(app, "/api/scan/analysis/stream").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);

        auto config = current_ai_config(state);
        const bool force = body->value("force", false);
        if (!config.enabled && !force) {
            return json_resp({
                {"status", "AI_DISABLED"},
                {"enabled", false},
                {"message", "AI scan analysis is disabled"}
            });
        }

        json scan_result = scan_result_from_request_or_latest(state, *body);
        if (scan_result.is_null()) {
            return err_resp("NO_SCAN", "No scan result available for AI analysis", 400);
        }

        // Return cached analysis as synthetic SSE complete event (unless force=true)
        if (!force) {
            const std::string ip = json_string_field(scan_result, "input_path");
            int fc = 0;
            if (scan_result.contains("frames_detected") && scan_result["frames_detected"].is_number())
                fc = scan_result["frames_detected"].get<int>();
            else if (scan_result.contains("frames_total") && scan_result["frames_total"].is_number())
                fc = scan_result["frames_total"].get<int>();
            json cached = find_cached_analysis(state, ip, fc);
            if (cached.value("has_analysis", false)) {
                crow::response sse_res;
                sse_res.code = 200;
                sse_res.set_header("Content-Type", "text/event-stream");
                sse_res.set_header("Cache-Control", "no-cache");
                sse_res.set_header("Connection", "keep-alive");
                sse_res.set_header("X-Accel-Buffering", "no");
                cached["from_cache"] = true;
                const std::string data = cached.dump();
                sse_res.body =
                    "event: complete\ndata: " + data + "\n\n";
                return sse_res;
            }
        }

        fs::path target_config_path;
        std::optional<crow::response> config_error;
        std::optional<json> base_config = load_base_config(state, *body, target_config_path, config_error);
        if (!base_config) return std::move(*config_error);

        std::string schema_error;
        std::optional<SchemaInfo> schema = load_schema_info(state, schema_error);
        if (!schema) return err_resp("SCHEMA_UNAVAILABLE", schema_error, 502);

        json allowed_paths = json::array();
        for (const auto& [path, _] : schema->paths) allowed_paths.push_back(path);

        json request_payload = {
            {"schema_version", "pi.scan-analysis.request.v1"},
            {"scan_result", scan_result},
            {"base_config", *base_config},
            {"allowed_config_paths", body->value("allowed_config_paths", allowed_paths)},
            {"model", json_string_field(*body, "model", config.model)},
            {"send_paths", config.send_paths},
            {"force", force},
        };
        if (body->contains("scan_metrics") && (*body)["scan_metrics"].is_object()) {
            request_payload["scan_metrics"] = (*body)["scan_metrics"];
        }

        // Return SSE stream response
        crow::response res;
        res.code = 200;
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_header("X-Accel-Buffering", "no"); // Disable nginx buffering

        // Note: Full SSE proxy implementation would require async handling
        // For now, return the initial response with headers
        res.body = "event: started\ndata: \"Analysis starting...\"\n\n";
        return res;
    });

    CROW_ROUTE(app, "/api/scan/analysis/apply").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const std::string analysis_id = body->value("analysis_id", std::string());
        if (analysis_id.empty()) return err_resp("BAD_REQUEST", "analysis_id is required", 400);
        auto job = state->job_store.get(analysis_id);

        // Fall back to persisted analyses if not in job store (e.g. after server restart)
        json persisted_data = json::object();
        bool found_persisted = false;
        if (!job || job->type != "scan_ai_analysis") {
            const fs::path dir = ai_analyses_dir(state);
            for (const auto& path : list_json_files(dir, true)) {
                auto parsed_opt = parse_json_file(path);
                if (!parsed_opt) continue;
                if (json_string_field(*parsed_opt, "analysis_id") == analysis_id) {
                    persisted_data = std::move(*parsed_opt);
                    found_persisted = true;
                    break;
                }
            }
            if (!found_persisted) {
                return err_resp("NOT_FOUND", "AI analysis not found", 404);
            }
        }

        const json& analysis_data = job ? job->data : persisted_data;

        fs::path target_config_path;
        std::optional<crow::response> config_error;
        std::optional<json> base_config = load_base_config(state, *body, target_config_path, config_error);
        if (!base_config) return std::move(*config_error);

        // Determine whether the caller requested a subset of updates via selected_paths.
        const bool has_selected_paths = body->contains("selected_paths") &&
            (*body)["selected_paths"].is_array() &&
            !(*body)["selected_paths"].empty();

        // If no specific paths are selected AND the analysis file carries a fully-validated
        // patched_config_yaml (written at analysis time), use it directly — identical to
        // loading a preset config.  This avoids re-patching against a potentially stale
        // live config and guarantees the result matches what was validated originally.
        static const json empty_ctx = json::object();
        const json& ctx = analysis_data.contains("analysis_context") && analysis_data["analysis_context"].is_object()
            ? analysis_data["analysis_context"]
            : empty_ctx;
        const bool has_full_patch = !has_selected_paths
            && ctx.contains("patched_config_yaml")
            && ctx["patched_config_yaml"].is_string()
            && !ctx["patched_config_yaml"].get<std::string>().empty();

        json patched;
        json applied = json::array();
        std::string yaml_text;

        if (has_full_patch) {
            // Preset-style apply: use the complete validated config from the analysis file.
            yaml_text = ctx["patched_config_yaml"].get<std::string>();
            auto patched_opt = parse_yaml_text(yaml_text);
            if (!patched_opt) {
                return err_resp("BAD_REQUEST", "patched_config_yaml parse error: invalid YAML", 400);
            }
            patched = *patched_opt;
            // Populate applied list from validated_updates for the response summary.
            if (analysis_data.contains("validated_updates") && analysis_data["validated_updates"].is_array()) {
                for (const auto& u : analysis_data["validated_updates"]) {
                    const std::string p = json_string_field(u, "path");
                    if (!p.empty() && u.contains("value"))
                        applied.push_back({{"path", p}, {"value", u["value"]}});
                }
            }
            if (applied.empty()) return err_resp("NO_VALID_UPDATES", "No validated AI updates in analysis", 400);
        } else {
            // Selective apply: patch only the requested (or all) validated updates onto the live config.
            json updates = selected_validated_updates(analysis_data, *body);
            if (updates.empty()) {
                return err_resp("NO_VALID_UPDATES", "No validated AI updates selected", 400);
            }
            patched = *base_config;
            for (const auto& update : updates) {
                const std::string path = json_string_field(update, "path");
                if (path.empty() || !update.contains("value")) continue;
                set_dotted(patched, path, update["value"]);
                applied.push_back({{"path", path}, {"value", update["value"]}});
            }
            if (applied.empty()) return err_resp("NO_VALID_UPDATES", "No validated AI updates selected", 400);
            yaml_text = yaml_dump(patched);
        }
        SubprocessResult validate_res = run_subprocess({state->runtime.cli_exe, "validate-config", "--stdin"},
                                                       state->runtime.project_root.string(),
                                                       yaml_text);
        json validation = parse_json_string(validate_res.stdout_str).value_or(json::object());
        if (validate_res.exit_code != 0 || !validation.value("valid", false)) {
            return json_resp({
                {"ok", false},
                {"code", "CONFIG_VALIDATION_FAILED"},
                {"message", "AI updates produced an invalid config"},
                {"analysis_id", analysis_id},
                {"validation", validation},
                {"applied", applied}
            });
        }

        json result = {
            {"ok", true},
            {"analysis_id", analysis_id},
            {"path", target_config_path.string()},
            {"config", patched},
            {"config_yaml", yaml_text},
            {"applied", applied},
            {"applied_paths", json::array()},
            {"validation", validation}
        };
        for (const auto& update : applied) result["applied_paths"].push_back(json_string_field(update, "path"));

        if (body->value("persist", false)) {
            SubprocessResult save_res = run_subprocess({state->runtime.cli_exe, "save-config", target_config_path.string(), "--stdin"},
                                                       state->runtime.project_root.string(),
                                                       yaml_text);
            auto saved = parse_json_string(save_res.stdout_str);
            if (save_res.exit_code != 0 || !saved || !saved->is_object()) {
                return backend_command_failed("save-config failed", save_res);
            }
            fs::path saved_path = saved->contains("path") && (*saved)["path"].is_string()
                ? fs::path((*saved)["path"].get<std::string>())
                : target_config_path;
            std::string rev_id = state->revision_store.add(saved_path, yaml_text, "pi_scan_ai");
            {
                std::lock_guard<std::mutex> lk(state->state_mutex);
                state->active_config_revision_id = rev_id;
            }
            state->ui_event_store.push("config.ai.apply", "config.ai.apply", {
                {"analysis_id", analysis_id},
                {"path", saved_path.string()},
                {"revision_id", rev_id},
                {"applied_count", static_cast<int>(applied.size())}
            });
            result["saved"] = saved->value("saved", false);
            result["revision_id"] = rev_id;
            result["path"] = saved_path.string();
        }

        return json_resp(result);
    });
}
