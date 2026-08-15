#include "routes/ai_routes.hpp"
#include "app_state.hpp"
#include "routes/pi_routes.hpp"
#include "routes/route_utils.hpp"
#include "subprocess_manager.hpp"
#include "services/ai_service.hpp"
#include "services/run_inspector.hpp"
#include "services/pi/pi_action_plan.hpp"
#include "services/pi/pi_action_validator.hpp"
#include "services/pi/pi_ai_request_builder.hpp"
#include "services/pi/pi_context_v2.hpp"
#include "services/pi/pi_memory_store.hpp"
#include "services/pi/pi_recommendation_validator.hpp"
#include "services/pi/pi_storage_paths.hpp"
#include "services/scan_summary.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <mutex>
#include <set>
#include <sstream>
#include <system_error>

using namespace tile_compile::routes;

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

std::string url_encode_query_value(const std::string& value) {
    std::ostringstream out;
    out << std::hex << std::uppercase;
    for (unsigned char c : value) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out << static_cast<char>(c);
        } else {
            out << '%' << std::setw(2) << std::setfill('0') << static_cast<int>(c);
        }
    }
    return out.str();
}

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
        tile_compile::ai::ai_config_to_json(tile_compile::ai::default_ai_config()),
        file_config);
    return tile_compile::ai::merge_ai_config_json(merged, memory_config);
}

tile_compile::ai::AiConfig current_ai_config(const std::shared_ptr<AppState>& state) {
    return tile_compile::ai::ai_config_from_json(current_ai_config_json(state));
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

crow::response sidecar_error_response(const std::exception& e, int fallback_status = 502) {
    if (const auto* upstream = dynamic_cast<const tile_compile::ai::AiSidecarHttpError*>(&e)) {
        json payload = upstream->payload();
        if (!payload.is_object()) payload = json::object();
        payload["available"] = false;
        payload["_upstream_status"] = upstream->status();
        if (!payload.contains("error")) {
            payload["error"] = {
                {"code", "AI_PROVIDER_REQUEST_FAILED"},
                {"message", upstream->what()}
            };
        }
        return json_resp(payload, static_cast<int>(upstream->status()));
    }
    return json_resp(sidecar_unavailable_payload(e), fallback_status);
}

crow::response sidecar_read_error_response(const std::exception& e) {
    if (dynamic_cast<const tile_compile::ai::AiSidecarHttpError*>(&e)) {
        return sidecar_error_response(e);
    }
    // Read-only discovery endpoints remain non-fatal when the sidecar itself is
    // unreachable, preserving the frontend's offline fallback contract.
    return json_resp(sidecar_unavailable_payload(e));
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

void attach_pi_action_plan(json& analysis) {
    const json updates = analysis.contains("validated_updates") && analysis["validated_updates"].is_array()
        ? analysis["validated_updates"]
        : json::array();
    json plan = tile_compile::pi::build_scan_analysis_action_plan(analysis, updates);
    analysis["action_plan"] = plan;
    analysis["action_plan_validation"] = tile_compile::pi::validate_action_plan_shape(plan);
}

json build_memory_query_context_signature(const json& body,
                                          const json& base_config,
                                          const json& allowed_paths,
                                          const json& paths);

json build_apply_candidate_memory(const std::string& analysis_id,
                                  const json& analysis_data,
                                  const json& applied,
                                  const json& validation,
                                  const fs::path& config_path,
                                  bool persisted) {
    json applied_paths = json::array();
    if (applied.is_array()) {
        for (const auto& update : applied) {
            const std::string path = json_string_field(update, "path");
            if (!path.empty()) applied_paths.push_back(path);
        }
    }
    json context = analysis_data.contains("analysis_context") && analysis_data["analysis_context"].is_object()
        ? analysis_data["analysis_context"]
        : json::object();
    json session_context = context.contains("session_context") && context["session_context"].is_object()
        ? context["session_context"]
        : json::object();
    json scan_metrics = context.contains("scan_metrics") && context["scan_metrics"].is_object()
        ? context["scan_metrics"]
        : json::object();
    json signature_body = {
        {"session_context", session_context},
        {"scan_metrics", scan_metrics},
        {"scan_result", context.contains("scan_metadata") && context["scan_metadata"].is_object()
            ? context["scan_metadata"]
            : json::object()}
    };
    if (context.contains("frame_count") && context["frame_count"].is_number()) {
        signature_body["scan_result"]["frames_detected"] = context["frame_count"];
    }
    json context_signature = build_memory_query_context_signature(
        signature_body,
        json::object(),
        applied_paths,
        applied_paths);

    json scope = {
        {"applies_when", json::array({
            "context_signature_matches_target_acquisition_and_affected_config_paths"
        })},
        {"does_not_apply_when", json::array({
            "different_target_class_or_acquisition_setup",
            "contradicting_outcome_memory_exists"
        })},
        {"confidence", analysis_data.value("confidence", 0.0)}
    };

    json memory = {
        {"type", "config_optimization"},
        {"status", "candidate"},
        {"privacy_class", "metadata_only"},
        {"source", "scan_ai_apply"},
        {"analysis_id", analysis_id},
        {"provenance", {
            {"analysis_id", analysis_id},
            {"config_path_name", config_path.filename().string()},
            {"source", "scan_ai_apply"}
        }},
        {"persisted", persisted},
        {"config_updates", applied},
        {"recommendation", {
            {"action_plan_fragment", {
                {"actions", applied}
            }},
            {"explanation", analysis_data.value("summary", std::string())}
        }},
        {"context_signature", context_signature},
        {"scope", scope},
        {"evidence", {
            {"analysis_id", analysis_id},
            {"validation", validation},
            {"human_feedback", nullptr},
            {"ai_summary", analysis_data.value("summary", std::string())}
        }},
        {"validation", validation},
        {"outcome", {
            {"stage", "scan_ai_apply"},
            {"validation_valid", validation.value("valid", false)},
            {"applied_count", applied.is_array() ? applied.size() : 0},
            {"applied_paths", applied_paths},
            {"persist_requested", persisted},
            {"verified", false}
        }},
        {"review", {
            {"status", "candidate"},
            {"reviewed_by", nullptr},
            {"reviewed_at", nullptr},
            {"notes", ""}
        }},
        {"retrieval", {
            {"keywords", applied_paths},
            {"negative", false}
        }},
    };
    if (analysis_data.is_object()) {
        if (analysis_data.contains("summary")) memory["summary"] = analysis_data["summary"];
        if (analysis_data.contains("confidence")) memory["confidence"] = analysis_data["confidence"];
        if (analysis_data.contains("detected_scenarios")) memory["detected_scenarios"] = analysis_data["detected_scenarios"];
        if (analysis_data.contains("warnings")) memory["warnings"] = analysis_data["warnings"];
    }
    return memory;
}

void collect_config_leaf_paths(const json& value, const std::string& prefix, std::set<std::string>& paths) {
    if (value.is_object()) {
        for (auto it = value.begin(); it != value.end(); ++it) {
            const std::string next = prefix.empty() ? it.key() : prefix + "." + it.key();
            collect_config_leaf_paths(it.value(), next, paths);
        }
        return;
    }
    if (!prefix.empty()) paths.insert(prefix);
}

void collect_json_path_fields(const json& value, std::set<std::string>& paths) {
    if (value.is_string()) {
        const std::string path = value.get<std::string>();
        if (!path.empty()) paths.insert(path);
        return;
    }
    if (value.is_object()) {
        if (value.contains("path") && value["path"].is_string()) {
            const std::string path = value["path"].get<std::string>();
            if (!path.empty()) paths.insert(path);
        }
        for (auto it = value.begin(); it != value.end(); ++it) {
            collect_json_path_fields(it.value(), paths);
        }
        return;
    }
    if (value.is_array()) {
        for (const auto& item : value) collect_json_path_fields(item, paths);
    }
}

json compact_memory_for_scan_context(const json& memory) {
    json out = {
        {"memory_id", json_string_field(memory, "memory_id")},
        {"type", json_string_field(memory, "type")},
        {"source", json_string_field(memory, "source")},
        {"status", memory.value("status", std::string("candidate"))},
        {"privacy_class", memory.value("privacy_class", std::string("metadata_only"))},
        {"retrieval_score", memory.value("retrieval_score", 0)}
    };
    if (memory.contains("context_match_score")) out["context_match_score"] = memory["context_match_score"];
    if (memory.contains("path_match_score")) out["path_match_score"] = memory["path_match_score"];
    if (memory.contains("match_explanation")) out["match_explanation"] = memory["match_explanation"];
    if (memory.contains("match_coverage")) out["match_coverage"] = memory["match_coverage"];
    if (memory.contains("summary")) out["summary"] = memory["summary"];
    if (memory.contains("confidence")) out["confidence"] = memory["confidence"];
    if (memory.contains("config_updates")) out["config_updates"] = memory["config_updates"];
    if (memory.contains("context_signature")) out["context_signature"] = memory["context_signature"];
    if (memory.contains("scope")) out["scope"] = memory["scope"];
    if (memory.contains("evidence")) out["evidence"] = memory["evidence"];
    if (memory.contains("outcome")) out["outcome"] = memory["outcome"];
    if (memory.contains("retrieval_warning")) out["retrieval_warning"] = memory["retrieval_warning"];
    if (memory.contains("detected_scenarios")) out["detected_scenarios"] = memory["detected_scenarios"];
    if (memory.contains("warnings")) out["warnings"] = memory["warnings"];
    if (memory.contains("validation") && memory["validation"].is_object()) {
        out["validation"] = {
            {"valid", memory["validation"].value("valid", false)}
        };
    }
    if (memory.contains("review") && memory["review"].is_object()) {
        out["review"] = {
            {"status", memory["review"].value("status", std::string())},
            {"reviewed_at", memory["review"].value("reviewed_at", std::string())},
            {"note", memory["review"].value("note", std::string())}
        };
    }
    return out;
}

json array_or_singleton_string(const json& value) {
    if (value.is_array()) return value;
    if (value.is_string() && !value.get<std::string>().empty()) return json::array({value});
    return json::array();
}

json session_value_or_null(const json& session_context, const char* key) {
    return session_context.is_object() && session_context.contains(key) ? session_context[key] : json(nullptr);
}

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string known_string_or_empty(const json& value) {
    if (!value.is_string()) return "";
    std::string text = value.get<std::string>();
    text.erase(text.begin(), std::find_if(text.begin(), text.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    text.erase(std::find_if(text.rbegin(), text.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), text.end());
    const std::string lowered = lower_ascii(text);
    if (text.empty() || lowered == "unknown" || lowered == "null" || lowered == "n/a") return "";
    return text;
}

json first_known_string_json(const std::vector<json>& values, const std::string& fallback = "unknown") {
    for (const auto& value : values) {
        const std::string text = known_string_or_empty(value);
        if (!text.empty()) return text;
    }
    return fallback;
}

json first_known_number_json(const std::vector<json>& values) {
    for (const auto& value : values) {
        if (value.is_number()) return value;
        if (value.is_string()) {
            try {
                return std::stod(value.get<std::string>());
            } catch (...) {
            }
        }
    }
    return nullptr;
}

json header_value_ci(const json& header, const std::vector<std::string>& aliases) {
    if (!header.is_object()) return nullptr;
    for (const auto& alias : aliases) {
        if (header.contains(alias)) return header[alias];
    }
    std::map<std::string, json> lowered;
    for (auto it = header.begin(); it != header.end(); ++it) {
        lowered[lower_ascii(it.key())] = it.value();
    }
    for (const auto& alias : aliases) {
        auto it = lowered.find(lower_ascii(alias));
        if (it != lowered.end()) return it->second;
    }
    return nullptr;
}

void collect_frame_header_values(const json& frame,
                                 const std::vector<std::string>& frame_keys,
                                 const std::vector<std::string>& header_keys,
                                 std::vector<json>& out) {
    if (!frame.is_object()) return;
    for (const auto& key : frame_keys) {
        if (frame.contains(key)) out.push_back(frame[key]);
    }
    if (frame.contains("header") && frame["header"].is_object()) {
        const json value = header_value_ci(frame["header"], header_keys);
        if (!value.is_null()) out.push_back(value);
    }
}

std::vector<json> scan_values_from_frames(const json& scan_result,
                                          const json& scan_metrics,
                                          const std::vector<std::string>& frame_keys,
                                          const std::vector<std::string>& header_keys) {
    std::vector<json> values;
    if (scan_result.contains("frames") && scan_result["frames"].is_array()) {
        for (const auto& frame : scan_result["frames"]) {
            collect_frame_header_values(frame, frame_keys, header_keys, values);
        }
    }
    if (scan_metrics.contains("frames") && scan_metrics["frames"].is_array()) {
        for (const auto& frame : scan_metrics["frames"]) {
            collect_frame_header_values(frame, frame_keys, header_keys, values);
        }
    }
    return values;
}

json unique_known_strings(const std::vector<json>& values) {
    std::set<std::string> seen;
    json out = json::array();
    for (const auto& value : values) {
        const std::string text = known_string_or_empty(value);
        if (text.empty()) continue;
        const std::string key = lower_ascii(text);
        if (seen.insert(key).second) out.push_back(text);
    }
    return out;
}

json build_memory_query_context_signature(const json& body,
                                          const json& base_config,
                                          const json& allowed_paths,
                                          const json& paths) {
    const json session_context = body.contains("session_context") && body["session_context"].is_object()
        ? body["session_context"]
        : json::object();
    const json scan_result = body.contains("scan_result") && body["scan_result"].is_object()
        ? body["scan_result"]
        : json::object();
    const json scan_metrics = body.contains("scan_metrics") && body["scan_metrics"].is_object()
        ? body["scan_metrics"]
        : json::object();

    std::string color_mode = scan_result.value("color_mode", std::string("unknown"));
    if (color_mode == "unknown" && base_config.contains("data") && base_config["data"].is_object()) {
        color_mode = json_string_field(base_config["data"], "color_mode", "unknown");
    }

    auto target_values = scan_values_from_frames(
        scan_result, scan_metrics, {"target", "object", "object_name"}, {"OBJECT", "TARGET", "OBJNAME"});
    auto camera_values = scan_values_from_frames(
        scan_result, scan_metrics, {"camera", "camera_name", "instrument"}, {"INSTRUME", "CAMERA", "CCDNAME"});
    auto telescope_values = scan_values_from_frames(
        scan_result, scan_metrics, {"telescope", "scope"}, {"TELESCOP", "SCOPE", "INSTRUME"});
    auto filter_values = scan_values_from_frames(
        scan_result, scan_metrics, {"filter", "filter_name"}, {"FILTER", "FILTER1", "FILTER2", "FILTER3", "FILTERID"});
    auto exposure_values = scan_values_from_frames(
        scan_result, scan_metrics, {"exposure_seconds", "exposure", "exptime"}, {"EXPTIME", "EXPOSURE", "EXP-TIME"});
    auto date_values = scan_values_from_frames(
        scan_result, scan_metrics, {"date_obs", "date-obs", "date"}, {"DATE-OBS", "DATEOBS", "DATE"});
    for (const std::string key : {"target", "object", "object_name"}) if (scan_result.contains(key)) target_values.push_back(scan_result[key]);
    for (const std::string key : {"camera", "camera_name", "instrument"}) if (scan_result.contains(key)) camera_values.push_back(scan_result[key]);
    for (const std::string key : {"telescope", "scope"}) if (scan_result.contains(key)) telescope_values.push_back(scan_result[key]);
    for (const std::string key : {"filter", "filter_name"}) if (scan_result.contains(key)) filter_values.push_back(scan_result[key]);
    for (const std::string key : {"exposure_seconds", "exposure", "exptime"}) if (scan_result.contains(key)) exposure_values.push_back(scan_result[key]);
    for (const std::string key : {"date_obs", "date-obs", "date"}) if (scan_result.contains(key)) date_values.push_back(scan_result[key]);

    json filters = session_context.contains("filters")
        ? array_or_singleton_string(session_context["filters"])
        : unique_known_strings(filter_values);

    json affected_paths = paths.is_array() ? paths : json::array();
    if (allowed_paths.is_array()) {
        for (const auto& path : allowed_paths) {
            if (path.is_string()) affected_paths.push_back(path);
        }
    }

    json problem_classes = json::array();
    if (body.contains("problem_classes")) problem_classes = array_or_singleton_string(body["problem_classes"]);
    else if (session_context.contains("problem_classes")) problem_classes = array_or_singleton_string(session_context["problem_classes"]);
    json problem_hints = json::array();
    if (body.contains("problem_hints")) problem_hints = array_or_singleton_string(body["problem_hints"]);
    else if (session_context.contains("problem_hints")) problem_hints = array_or_singleton_string(session_context["problem_hints"]);

    return {
        {"schema_version", "pi.context_signature.v1"},
        {"target", {
            {"object_name", first_known_string_json({
                session_value_or_null(session_context, "target_name"),
                scan_result.contains("target") ? scan_result["target"] : json(nullptr),
                scan_result.contains("object") ? scan_result["object"] : json(nullptr),
                target_values.empty() ? json(nullptr) : target_values.front()
            })},
            {"object_type", session_context.value("target_type", std::string("unknown"))},
            {"angular_size_class", session_context.value("target_angular_size", std::string("unknown"))},
            {"has_extended_emission", session_value_or_null(session_context, "has_extended_emission")}
        }},
        {"acquisition", {
            {"camera_name", first_known_string_json({
                session_value_or_null(session_context, "camera_name"),
                scan_result.contains("camera") ? scan_result["camera"] : json(nullptr),
                camera_values.empty() ? json(nullptr) : camera_values.front()
            })},
            {"camera_type", session_context.value("camera_type", std::string("unknown"))},
            {"color_mode", color_mode},
            {"filters", filters},
            {"frame_count", infer_frame_count(scan_result, scan_metrics)},
            {"exposure_seconds", first_known_number_json(exposure_values)},
            {"date_obs_first", date_values.empty() ? json(nullptr) : first_known_string_json({date_values.front()}, "")},
            {"date_obs_last", date_values.empty() ? json(nullptr) : first_known_string_json({date_values.back()}, "")},
            {"calibration", {
                {"darks", session_context.value("calibration_darks", false)},
                {"flats", session_context.value("calibration_flats", false)},
                {"bias", session_context.value("calibration_bias", false)}
            }}
        }},
        {"optics", {
            {"telescope", first_known_string_json({
                session_value_or_null(session_context, "telescope"),
                scan_result.contains("telescope") ? scan_result["telescope"] : json(nullptr),
                telescope_values.empty() ? json(nullptr) : telescope_values.front()
            })},
            {"focal_length_mm", session_value_or_null(session_context, "focal_length_mm")},
            {"f_ratio", session_value_or_null(session_context, "f_ratio")},
            {"pixel_scale_arcsec", session_value_or_null(session_context, "pixel_scale_arcsec")}
        }},
        {"mount", {
            {"type", session_context.value("mount_type", std::string("unknown"))},
            {"tracking_quality", session_context.value("tracking_quality", std::string("unknown"))}
        }},
        {"pipeline", {
            {"affected_paths", affected_paths},
            {"phases", body.contains("pipeline_phases") ? array_or_singleton_string(body["pipeline_phases"]) : json::array()}
        }},
        {"problem", {
            {"classes", problem_classes},
            {"hints", problem_hints}
        }},
        {"quality", {
            {"aggregate", scan_metrics.contains("aggregate") ? scan_metrics["aggregate"] : json::object()},
            {"session_geometry", scan_metrics.contains("session_geometry") ? scan_metrics["session_geometry"] : json::object()}
        }}
    };
}

json accepted_pi_memories_for_scan_request(const std::shared_ptr<AppState>& state,
                                           const json& body,
                                           const json& base_config,
                                           const json& allowed_paths) {
    std::set<std::string> query_paths;
    collect_config_leaf_paths(base_config, "", query_paths);
    if (allowed_paths.is_array()) {
        for (const auto& path : allowed_paths) {
            if (path.is_string()) query_paths.insert(path.get<std::string>());
        }
    }

    json paths = json::array();
    for (const auto& path : query_paths) paths.push_back(path);
    const json context_signature = build_memory_query_context_signature(body, base_config, allowed_paths, paths);

    tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
    const json retrieved = store.retrieve({
        {"type", "config_optimization"},
        {"paths", paths},
        {"context_signature", context_signature}
    }, 8);

    json accepted = json::array();
    for (const auto& memory : retrieved) {
        if (memory.value("status", std::string()) != "accepted") continue;
        accepted.push_back(compact_memory_for_scan_context(memory));
    }
    return accepted;
}

json negative_pi_memories_for_scan_request(const std::shared_ptr<AppState>& state,
                                           const json& body,
                                           const json& base_config,
                                           const json& allowed_paths) {
    std::set<std::string> query_paths;
    collect_config_leaf_paths(base_config, "", query_paths);
    if (allowed_paths.is_array()) {
        for (const auto& path : allowed_paths) {
            if (path.is_string()) query_paths.insert(path.get<std::string>());
        }
    }

    json paths = json::array();
    for (const auto& path : query_paths) paths.push_back(path);
    const json context_signature = build_memory_query_context_signature(body, base_config, allowed_paths, paths);

    tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
    json negative = json::array();
    for (const auto& memory : store.retrieve_negative({
            {"type", "config_optimization"},
            {"paths", paths},
            {"context_signature", context_signature}
        }, 8)) {
        negative.push_back(compact_memory_for_scan_context(memory));
    }
    return negative;
}

json session_context_with_accepted_memories(const std::shared_ptr<AppState>& state,
                                            const json& body,
                                            const json& base_config,
                                            const json& allowed_paths) {
    json session_context = body.contains("session_context") && body["session_context"].is_object()
        ? body["session_context"]
        : json::object();
    const json memories = accepted_pi_memories_for_scan_request(state, body, base_config, allowed_paths);
    if (!memories.empty()) {
        session_context["accepted_pi_memories"] = memories;
    }
    const json negative_memories = negative_pi_memories_for_scan_request(state, body, base_config, allowed_paths);
    if (!negative_memories.empty()) {
        session_context["negative_pi_memories"] = negative_memories;
    }
    return session_context;
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
    return state->runtime.runtime_dir / ".ai_analyses";
}

std::string completion_decode_base64url(std::string value) {
    for (char& ch : value) {
        if (ch == '-') ch = '+';
        else if (ch == '_') ch = '/';
    }
    while (value.size() % 4 != 0) value.push_back('=');
    static const std::string alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    int buffer = 0;
    int bits_collected = 0;
    for (char ch : value) {
        if (ch == '=') break;
        const auto pos = alphabet.find(ch);
        if (pos == std::string::npos) return "";
        buffer = (buffer << 6) | static_cast<int>(pos);
        bits_collected += 6;
        if (bits_collected >= 8) {
            bits_collected -= 8;
            out.push_back(static_cast<char>((buffer >> bits_collected) & 0xFF));
        }
    }
    return out;
}

std::string decode_completion_run_id(std::string run_id) {
    if (run_id.rfind("b64_", 0) != 0) return run_id;
    const std::string decoded = completion_decode_base64url(run_id.substr(4));
    return decoded.empty() ? run_id : decoded;
}

std::string sha256_hex(const std::string& value) {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(value.data()), value.size(), digest);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned char byte : digest) out << std::setw(2) << static_cast<int>(byte);
    return out.str();
}

std::mutex completion_job_mutex;

fs::path completion_analysis_path(const std::shared_ptr<AppState>& state,
                                  const std::string& run_id) {
    return ai_analyses_dir(state) / ("run_completion_" + sha256_hex(run_id).substr(0, 24) + ".json");
}

fs::path completion_analysis_status_path(const std::shared_ptr<AppState>& state,
                                         const std::string& run_id) {
    return ai_analyses_dir(state) / ("run_completion_" + sha256_hex(run_id).substr(0, 24) + ".status.json");
}

bool persist_completion_payload(const fs::path& path, const json& payload) {
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    if (ec) return false;
    const fs::path temporary = path.string() + ".tmp";
    {
        std::ofstream out(temporary, std::ios::trunc);
        if (!out) return false;
        out << payload.dump(2) << '\n';
        if (!out) return false;
    }
    fs::remove(path, ec);
    ec.clear();
    fs::rename(temporary, path, ec);
    if (ec) {
        fs::remove(temporary, ec);
        return false;
    }
    return true;
}

json load_completion_analysis(const std::shared_ptr<AppState>& state,
                              const std::string& run_id) {
    auto status = parse_json_file(completion_analysis_status_path(state, run_id));
    if (status &&
        status->value("schema_version", std::string()) == "pi.run-completion-analysis.status.v1" &&
        status->value("run_id", std::string()) == run_id) {
        const std::string state_name = status->value("status", std::string());
        if (state_name == "running" || state_name == "pending") {
            const std::string job_id = status->value("analysis_id", std::string());
            const auto job = job_id.empty() ? std::optional<Job>{} : state->job_store.get(job_id);
            if (job && (job->state == JobState::running || job->state == JobState::pending)) {
                (*status)["has_analysis"] = false;
                return *status;
            }
            if (job && job->state == JobState::ok) {
                // The worker publishes the result before marking the in-memory job complete.
            } else {
                (*status)["status"] = "error";
                (*status)["has_analysis"] = false;
                (*status)["error"] = {
                    {"code", "COMPLETION_ANALYSIS_INTERRUPTED"},
                    {"message", job && !job->error_message.empty()
                        ? job->error_message
                        : "Completion analysis was interrupted before a result was persisted"}
                };
                return *status;
            }
        } else if (state_name == "error") {
            (*status)["has_analysis"] = false;
            return *status;
        }
    }

    auto parsed = parse_json_file(completion_analysis_path(state, run_id));
    if (!parsed || parsed->value("schema_version", std::string()) != "pi.run-completion-analysis.v1" ||
        parsed->value("run_id", std::string()) != run_id) {
        return {{"has_analysis", false}, {"run_id", run_id}};
    }
    (*parsed)["has_analysis"] = true;
    (*parsed)["from_cache"] = true;
    return *parsed;
}

bool persist_completion_analysis(const std::shared_ptr<AppState>& state,
                                 const std::string& run_id,
                                 const json& analysis) {
    return persist_completion_payload(completion_analysis_path(state, run_id), analysis);
}

bool persist_completion_analysis_status(const std::shared_ptr<AppState>& state,
                                        const std::string& run_id,
                                        const json& status) {
    return persist_completion_payload(completion_analysis_status_path(state, run_id), status);
}

std::optional<crow::response> resolve_completion_run_dir(
    const std::shared_ptr<AppState>& state,
    const std::string& run_id,
    const std::string& requested_run_dir,
    fs::path& run_dir) {
    if (requested_run_dir.empty()) {
        try {
            run_dir = state->runtime.resolve_run_dir(run_id);
            return std::nullopt;
        } catch (const std::exception& e) {
            return err_resp("RUN_NOT_FOUND", e.what(), 404);
        }
    }
    auto resolved = state->runtime.resolve_input_path(fs::path(requested_run_dir), true);
    run_dir = resolved.path;
    if (resolved.status == PathStatus::not_allowed) {
        return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + run_dir.string(), 403);
    }
    if (resolved.status == PathStatus::not_found || !fs::is_directory(run_dir)) {
        return err_resp("RUN_NOT_FOUND", "Run directory does not exist: " + run_dir.string(), 404);
    }
    return std::nullopt;
}

json completion_config_schema(const SchemaInfo& schema) {
    json result = json::object();
    for (const auto& [path, node] : schema.paths) {
        if (!node.is_object()) continue;
        json entry = json::object();
        for (const std::string key : {"type", "enum", "description", "default", "minimum", "maximum"}) {
            if (node.contains(key)) entry[key == "description" ? "desc" : key] = node[key];
        }
        if (!entry.empty()) result[path] = std::move(entry);
    }
    return result;
}

json extract_resume_recommendation(const json& analysis) {
    static const std::set<std::string> supported = {
        "SCAN_INPUT", "REGISTRATION", "PREWARP", "CHANNEL_SPLIT", "NORMALIZATION",
        "GLOBAL_METRICS", "TILE_GRID", "COMMON_OVERLAP", "LOCAL_METRICS",
        "TILE_RECONSTRUCTION", "STATE_CLUSTERING", "SYNTHETIC_FRAMES", "AQMH_MAPS",
        "AQMH_GLOBAL_QUALITY", "AQMH_METRICS", "AQMH_RECONSTRUCTION", "AQMH_DIAGNOSTICS",
        "STACKING", "DEBAYER", "ASTROMETRY", "BGE", "PCC", "HYPERMETRIC_STRETCH"
    };
    json recommendation = analysis.value("resume_recommendation", json::object());
    if (!recommendation.is_object()) recommendation = json::object();
    const std::string phase = recommendation.value("from_phase", std::string());
    if (!supported.count(phase)) {
        recommendation["from_phase"] = "DEBAYER";
        recommendation["reason"] = "Fallback to the earliest safe phase that regenerates stacked_rgb.fits.";
        recommendation["model_phase_rejected"] = phase;
    }
    recommendation["feasibility"] = "requires_dry_run";
    return recommendation;
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
    str("object_name");
    str("target");
    str("object");
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
            fstr("object");
            fstr("camera");
            fstr("telescope");
            fstr("filter");
            fstr("date_obs");
            fnum("exposure_seconds");
            fnum("exposure");
            fnum("exptime");
            fnum("gain");
            fnum("image_width");
            fnum("image_height");
            fnum("temperature_c");
            if (f.contains("header") && f["header"].is_object()) {
                const auto& h = f["header"];
                auto hval = [&](std::initializer_list<const char*> keys) -> json {
                    for (const char* key : keys) {
                        if (h.contains(key)) return h[key];
                    }
                    for (auto it = h.begin(); it != h.end(); ++it) {
                        const std::string lowered_key = lower_ascii(it.key());
                        for (const char* key : keys) {
                            if (lowered_key == lower_ascii(key)) return it.value();
                        }
                    }
                    return nullptr;
                };
                auto hstr = [&](const std::string& out_key, std::initializer_list<const char*> keys) {
                    if (meta.contains(out_key)) return;
                    json value = hval(keys);
                    if (value.is_string()) meta[out_key] = value;
                };
                auto hnum = [&](const std::string& out_key, std::initializer_list<const char*> keys) {
                    if (meta.contains(out_key)) return;
                    json value = hval(keys);
                    if (value.is_number()) meta[out_key] = value;
                    else if (value.is_string()) {
                        try {
                            meta[out_key] = std::stod(value.get<std::string>());
                        } catch (...) {
                        }
                    }
                };
                hstr("target", {"OBJECT", "TARGET", "OBJNAME"});
                hstr("camera", {"INSTRUME", "CAMERA", "CCDNAME"});
                hstr("telescope", {"TELESCOP", "SCOPE"});
                hstr("filter", {"FILTER", "FILTER1", "FILTER2", "FILTER3", "FILTERID"});
                hstr("date_obs", {"DATE-OBS", "DATEOBS", "DATE"});
                hnum("exposure_seconds", {"EXPTIME", "EXPOSURE", "EXP-TIME"});
            }
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
        {"scan_metadata", extract_scan_metadata(scan_result)}
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
                          int frame_count = 0,
                          const std::string& object_name = "") {
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
        if (!object_name.empty()) {
            const std::string cached_object = meta.value("object_name", meta.value("target", meta.value("object", std::string())));
            if (cached_object != object_name) continue;
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
            *body);
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
            return sidecar_read_error_response(e);
        }
    });

    CROW_ROUTE(app, "/api/ai/account").methods("GET"_method)
    ([state](const crow::request& req) {
        try {
            std::string path = "/account";
            if (const char* provider = req.url_params.get("provider")) {
                path += "?provider=" + url_encode_query_value(provider);
            }
            tile_compile::ai::AiSidecarClient client(current_ai_config(state));
            return json_resp(client.get(path));
        } catch (const std::exception& e) {
            return sidecar_read_error_response(e);
        }
    });

    CROW_ROUTE(app, "/api/ai/traffic").methods("GET"_method)
    ([state](const crow::request& req) {
        try {
            std::string path = "/traffic";
            if (const char* limit = req.url_params.get("limit")) {
                path += "?limit=" + url_encode_query_value(limit);
            }
            tile_compile::ai::AiSidecarClient client(current_ai_config(state));
            return json_resp(client.get(path));
        } catch (const std::exception& e) {
            return sidecar_read_error_response(e);
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
            return sidecar_error_response(e);
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
            return sidecar_error_response(e);
        }
    });

    CROW_ROUTE(app, "/api/ai/auth/<string>").methods("DELETE"_method)
    ([state](const crow::request&, std::string provider) {
        try {
            tile_compile::ai::AiSidecarClient client(current_ai_config(state));
            return json_resp(client.del("/auth/" + provider));
        } catch (const std::exception& e) {
            return sidecar_error_response(e);
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
            const std::string object_name = scan_result.value("object_name", scan_result.value("target", scan_result.value("object", std::string())));
            json cached = find_cached_analysis(state, ip, fc, object_name);
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
                if (schema_node.contains("exclusiveMinimum")) entry["exclusiveMinimum"] = schema_node["exclusiveMinimum"];
                if (schema_node.contains("maximum")) entry["maximum"] = schema_node["maximum"];
                if (!entry.empty()) config_schema[path] = std::move(entry);
            }
        }

        json request_payload;
        try {
            const json session_context = session_context_with_accepted_memories(
                state,
                *body,
                *base_config,
                body->value("allowed_config_paths", allowed_paths));
            request_payload = {
                {"schema_version", "pi.scan-analysis.request.v1"},
                {"scan_result", scan_result},
                {"base_config", *base_config},
                {"allowed_config_paths", body->value("allowed_config_paths", allowed_paths)},
                {"config_schema", config_schema},
                {"pi_context", tile_compile::pi::build_scan_pi_context(
                    schema->paths,
                    *base_config,
                    scan_result,
                    body->contains("scan_metrics") && (*body)["scan_metrics"].is_object()
                        ? (*body)["scan_metrics"]
                        : json::object())},
                {"model", json_string_field(*body, "model", config.model)},
                {"send_paths", config.send_paths},
                {"force", force},
            };
            if (body->contains("scan_metrics") && (*body)["scan_metrics"].is_object()) {
                request_payload["scan_metrics"] = (*body)["scan_metrics"];
            }
            if (!session_context.empty()) {
                request_payload["session_context"] = session_context;
            }
            request_payload["ai_request"] = tile_compile::pi::build_ai_request_v2({
                {"task", "scan_recommendation"},
                {"user_message", "Analyze the transferred scan, config and memory data and propose validated tile_compile parameter changes."},
                {"context_signature", build_memory_query_context_signature(
                    *body,
                    *base_config,
                    body->value("allowed_config_paths", allowed_paths),
                    body->value("allowed_config_paths", allowed_paths))},
                {"scan_context", {
                    {"scan_result", scan_result},
                    {"scan_metrics", body->contains("scan_metrics") ? (*body)["scan_metrics"] : json::object()}
                }},
                {"config", {
                    {"base_config", *base_config},
                    {"config_schema", config_schema},
                    {"parameter_catalog", request_payload["pi_context"]["parameter_catalog"]}
                }},
                {"pi_context", request_payload["pi_context"]},
                {"allowed_config_paths", body->value("allowed_config_paths", allowed_paths)},
                {"session_context", session_context},
                {"expected_response", "pi.scan-analysis.v1 with parameter recommendations, evidence, risks, confidence and action plan candidates"},
                {"provider", config.provider},
                {"model", json_string_field(*body, "model", config.model)},
                {"source_request_schema", "pi.scan-analysis.request.v1"}
            });
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
            const json candidates = tile_compile::pi::normalize_candidate_updates(analysis);
            const json validation = tile_compile::pi::validate_recommendation_updates(
                candidates, schema->paths, *base_config, state, request_payload["pi_context"]);
            analysis["updates"] = candidates;
            analysis["validated_updates"] = validation["validated_updates"];
            analysis["rejected_updates"] = validation["rejected_updates"];
            analysis["validation"] = validation["validation"];
            analysis["candidate_count"] = validation["candidate_count"];
            analysis["validated_count"] = validation["validated_updates"].size();
            analysis["rejected_count"] = validation["rejected_updates"].size();
            analysis["config_path"] = target_config_path.string();
            analysis["analysis_context"] = build_analysis_context(scan_result, request_payload);
            analysis["analysis_context"]["pi_context"] = request_payload["pi_context"];
            if (validation.contains("patched_config") && validation["patched_config"].is_object())
                analysis["analysis_context"]["patched_config"] = validation["patched_config"];
            if (validation.contains("patched_config_yaml") && validation["patched_config_yaml"].is_string())
                analysis["analysis_context"]["patched_config_yaml"] = validation["patched_config_yaml"];
            attach_pi_action_plan(analysis);
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
            return sidecar_error_response(e);
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
        const json pi_context = body->contains("pi_context") && (*body)["pi_context"].is_object()
            ? (*body)["pi_context"]
            : tile_compile::pi::build_scan_pi_context(schema->paths, *base_config, scan_result, scan_metrics);
        const json candidates = tile_compile::pi::normalize_candidate_updates(analysis);
        const json validation = tile_compile::pi::validate_recommendation_updates(
            candidates, schema->paths, *base_config, state, pi_context);
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
        analysis["analysis_context"]["pi_context"] = pi_context;
        if (validation.contains("patched_config") && validation["patched_config"].is_object())
            analysis["analysis_context"]["patched_config"] = validation["patched_config"];
        if (validation.contains("patched_config_yaml") && validation["patched_config_yaml"].is_string())
            analysis["analysis_context"]["patched_config_yaml"] = validation["patched_config_yaml"];
        attach_pi_action_plan(analysis);

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
                if (schema_node.contains("exclusiveMinimum")) entry["exclusiveMinimum"] = schema_node["exclusiveMinimum"];
                if (schema_node.contains("maximum")) entry["maximum"] = schema_node["maximum"];
                if (!entry.empty()) config_schema[path] = std::move(entry);
            }
        }

        const json selected_allowed_paths = body->value("allowed_config_paths", allowed_paths);
        const json scan_metrics = body->contains("scan_metrics") && (*body)["scan_metrics"].is_object()
            ? (*body)["scan_metrics"]
            : json::object();
        const json pi_context = tile_compile::pi::build_scan_pi_context(schema->paths, *base_config, scan_result, scan_metrics);
        const json session_context = session_context_with_accepted_memories(
            state,
            *body,
            *base_config,
            selected_allowed_paths);

        json request_payload = {
            {"schema_version", "pi.scan-analysis.request.v1"},
            {"scan_result", scan_result},
            {"base_config", *base_config},
            {"allowed_config_paths", selected_allowed_paths},
            {"config_schema", config_schema},
            {"pi_context", pi_context},
            {"model", json_string_field(*body, "model", config.model)},
            {"send_paths", config.send_paths},
            {"force", force},
        };
        if (body->contains("scan_metrics") && (*body)["scan_metrics"].is_object()) {
            request_payload["scan_metrics"] = (*body)["scan_metrics"];
        }
        if (!session_context.empty()) {
            request_payload["session_context"] = session_context;
        }
        request_payload["ai_request"] = tile_compile::pi::build_ai_request_v2({
            {"task", "scan_recommendation"},
            {"user_message", "Analyze the transferred scan, config and memory data and propose validated tile_compile parameter changes."},
            {"context_signature", build_memory_query_context_signature(
                *body,
                *base_config,
                selected_allowed_paths,
                selected_allowed_paths)},
            {"scan_context", {
                {"scan_result", scan_result},
                {"scan_metrics", body->contains("scan_metrics") ? (*body)["scan_metrics"] : json::object()}
            }},
            {"config", {
                {"base_config", *base_config},
                {"config_schema", config_schema},
                {"parameter_catalog", pi_context["parameter_catalog"]}
            }},
            {"pi_context", pi_context},
            {"allowed_config_paths", selected_allowed_paths},
            {"session_context", session_context},
            {"expected_response", "pi.scan-analysis.v1 stream with parameter recommendations, evidence, risks and confidence"},
            {"provider", config.provider},
            {"model", json_string_field(*body, "model", config.model)},
            {"source_request_schema", "pi.scan-analysis.request.v1"}
        });

        crow::response res;
        res.code = 200;
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_header("X-Accel-Buffering", "no");
        std::ostringstream stream;
        stream << "event: progress\ndata: " << json({{"phase", "building_prompt"}, {"progress", 10}}).dump() << "\n\n";
        try {
            tile_compile::ai::AiSidecarClient client(config);
            json analysis = client.post("/analyze", request_payload);
            if (!analysis.is_object() || json_string_field(analysis, "schema_version") != "pi.scan-analysis.v1") {
                stream << "event: error\ndata: " << json({{"phase", "error"}, {"message", "AI sidecar returned an invalid analysis payload"}}).dump() << "\n\n";
                res.body = stream.str();
                return res;
            }
            const json candidates = tile_compile::pi::normalize_candidate_updates(analysis);
            const json validation = tile_compile::pi::validate_recommendation_updates(
                candidates, schema->paths, *base_config, state, pi_context);
            analysis["updates"] = candidates;
            analysis["validated_updates"] = validation["validated_updates"];
            analysis["rejected_updates"] = validation["rejected_updates"];
            analysis["validation"] = validation["validation"];
            analysis["candidate_count"] = validation["candidate_count"];
            analysis["validated_count"] = validation["validated_updates"].size();
            analysis["rejected_count"] = validation["rejected_updates"].size();
            analysis["config_path"] = target_config_path.string();
            analysis["analysis_context"] = build_analysis_context(scan_result, request_payload);
            analysis["analysis_context"]["pi_context"] = pi_context;
            if (validation.contains("patched_config") && validation["patched_config"].is_object())
                analysis["analysis_context"]["patched_config"] = validation["patched_config"];
            if (validation.contains("patched_config_yaml") && validation["patched_config_yaml"].is_string())
                analysis["analysis_context"]["patched_config_yaml"] = validation["patched_config_yaml"];
            attach_pi_action_plan(analysis);
            const std::string job_id = state->job_store.create("scan_ai_analysis");
            json job_data = analysis;
            job_data["analysis_id"] = job_id;
            job_data["model"] = request_payload["model"];
            job_data["provider"] = config.provider;
            state->job_store.update_state(job_id, JobState::ok, job_data);
            analysis["analysis_id"] = job_id;
            persist_analysis(state, analysis, extract_scan_metadata(scan_result),
                             analysis["analysis_context"]);
            stream << "event: complete\ndata: " << analysis.dump() << "\n\n";
        } catch (const std::exception& e) {
            stream << "event: error\ndata: " << json({{"phase", "error"}, {"message", e.what()}}).dump() << "\n\n";
        }
        res.body = stream.str();
        return res;
    });

    CROW_ROUTE(app, "/api/runs/<string>/completion-analysis").methods("GET"_method)
    ([state](const crow::request&, std::string run_id) {
        run_id = decode_completion_run_id(std::move(run_id));
        return json_resp(load_completion_analysis(state, run_id));
    });

    CROW_ROUTE(app, "/api/runs/<string>/completion-analysis").methods("POST"_method)
    ([state](const crow::request& req, std::string run_id) {
        run_id = decode_completion_run_id(std::move(run_id));
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);

        fs::path run_dir;
        if (auto error = resolve_completion_run_dir(
                state, run_id, body->value("run_dir", std::string()), run_dir)) {
            return std::move(*error);
        }
        const json run_status = read_run_status(run_dir);
        if (run_status.value("status", std::string()) != "completed") {
            return err_resp("RUN_NOT_COMPLETED", "Completion analysis requires a completed run", 409,
                            {{"status", run_status.value("status", std::string("unknown"))}});
        }

        const fs::path config_path = run_dir / "config.yaml";
        const std::string config_yaml = tile_compile::routes::read_file_str(config_path);
        auto base_config = parse_yaml_text(config_yaml);
        if (!base_config || !base_config->is_object()) {
            return err_resp("RUN_CONFIG_UNAVAILABLE", "Completed run config.yaml is missing or invalid", 409);
        }

        json preview = build_run_completion_preview_image(run_dir);
        if (!preview.value("available", false)) {
            return err_resp("STACKED_RGB_UNAVAILABLE",
                            "outputs/stacked_rgb.fits is required for completion analysis", 404,
                            {{"source_artifact", preview.value("path", std::string("outputs/stacked_rgb.fits"))},
                             {"reason", preview.value("reason", std::string("unavailable"))}});
        }
        const std::string source_fingerprint = sha256_hex(
            preview.value("base64", std::string()) + "\n" + config_yaml);
        const bool force = body->value("force", false);
        if (!force) {
            json cached = load_completion_analysis(state, run_id);
            if (cached.value("has_analysis", false) &&
                cached.value("source_fingerprint", std::string()) == source_fingerprint) {
                return json_resp(cached);
            }
        }

        auto ai_config = current_ai_config(state);
        if (!ai_config.enabled) {
            return json_resp({
                {"schema_version", "pi.run-completion-analysis.v1"},
                {"status", "AI_DISABLED"},
                {"has_analysis", false},
                {"run_id", run_id},
                {"source_artifact", "outputs/stacked_rgb.fits"}
            });
        }

        std::string schema_error;
        auto schema = load_schema_info(state, schema_error);
        if (!schema) return err_resp("SCHEMA_UNAVAILABLE", schema_error, 502);
        json allowed_paths = json::array();
        for (const auto& [path, _] : schema->paths) allowed_paths.push_back(path);
        const json config_schema = completion_config_schema(*schema);
        const json pi_context = tile_compile::pi::build_run_completed_pi_context(schema->paths, *base_config, run_dir, run_status);

        const std::string prompt =
            "You are PI for tile_compile. Analyze the attached preview rendered from the immutable "
            "outputs/stacked_rgb.fits before BGE and hypermetric stretch. Return exactly one JSON object, "
            "without markdown, with schema_version pi.run-completion-analysis.v1. Required fields: summary "
            "(string), findings (array of objects with severity, title, evidence, recommendation), updates "
            "(array of objects with path, value, reason, confidence, risk), and resume_recommendation "
            "(object with from_phase and reason). Recommend object-agnostic parameter improvements for a new "
            "run. Use only allowed_config_paths and config_schema. Do not infer defects that are hidden by the "
            "preview. AQMH and Classic Tile Compile are independent: never feed Classic local/tile quality "
            "metrics into AQMH weights. Select the earliest necessary supported resume phase and explain why; "
            "the backend will verify feasibility separately.";

        json image_info = preview;
        image_info.erase("base64");
        json request_payload = {
            {"schema_version", "pi.run-completion-analysis.request.v1"},
            {"run_id", run_id},
            {"run_status", run_status},
            {"source_artifact", image_info},
            {"source_fingerprint", source_fingerprint},
            {"base_config", *base_config},
            {"config_schema", config_schema},
            {"pi_context", pi_context},
            {"allowed_config_paths", allowed_paths},
            {"image_base64", preview["base64"]},
            {"image_mime", preview.value("mime", std::string("image/png"))},
            {"model", json_string_field(*body, "model", ai_config.model)},
            {"prompt", prompt},
            {"ai_request", tile_compile::pi::build_ai_request_v2({
                {"task", "run_completion_analysis"},
                {"user_message", prompt},
                {"run_context", {{"run_id", run_id}, {"status", run_status}, {"source_artifact", image_info}}},
                {"config", {{"base_config", *base_config}, {"config_schema", config_schema}, {"parameter_catalog", pi_context["parameter_catalog"]}}},
                {"pi_context", pi_context},
                {"allowed_config_paths", allowed_paths},
                {"expected_response", "pi.run-completion-analysis.v1"},
                {"provider", ai_config.provider},
                {"model", json_string_field(*body, "model", ai_config.model)},
                {"source_request_schema", "pi.run-completion-analysis.request.v1"}
            })}
        };

        json initial_job_data = {
            {"run_id", run_id},
            {"run_dir", run_dir.string()},
            {"source_fingerprint", source_fingerprint},
            {"source_artifact", image_info}
        };
        std::string analysis_id;
        {
            std::lock_guard<std::mutex> lock(completion_job_mutex);
            auto status = parse_json_file(completion_analysis_status_path(state, run_id));
            if (status && status->value("status", std::string()) == "running") {
                const std::string existing_id = status->value("analysis_id", std::string());
                const auto existing_job = existing_id.empty()
                    ? std::optional<Job>{}
                    : state->job_store.get(existing_id);
                if (existing_job && (existing_job->state == JobState::running ||
                                     existing_job->state == JobState::pending)) {
                    (*status)["has_analysis"] = false;
                    (*status)["deduplicated"] = true;
                    return json_resp(*status, 202);
                }
            }

            analysis_id = state->job_store.create("run_completion_analysis", run_id);
            initial_job_data["analysis_id"] = analysis_id;
            json running_status = {
                {"schema_version", "pi.run-completion-analysis.status.v1"},
                {"status", "running"},
                {"has_analysis", false},
                {"run_id", run_id},
                {"run_dir", run_dir.string()},
                {"analysis_id", analysis_id},
                {"source_artifact", image_info},
                {"source_fingerprint", source_fingerprint}
            };
            if (!persist_completion_analysis_status(state, run_id, running_status)) {
                state->job_store.update_state(
                    analysis_id, JobState::error, initial_job_data,
                    "Completion analysis status could not be persisted");
                return err_resp("ANALYSIS_PERSIST_FAILED",
                                "Completion analysis status could not be persisted", 500);
            }
            state->job_store.update_state(analysis_id, JobState::running, initial_job_data);
        }

        std::thread([
            state,
            run_id,
            run_dir,
            source_fingerprint,
            image_info,
            request_payload,
            ai_config,
            schema = *schema,
            base_config = *base_config,
            pi_context,
            analysis_id
        ]() mutable {
            try {
                tile_compile::ai::AiSidecarClient client(ai_config);
                json analysis = client.post("/run-completion-analysis", request_payload);
                if (!analysis.is_object() ||
                    analysis.value("schema_version", std::string()) != "pi.run-completion-analysis.v1") {
                    throw std::runtime_error("AI sidecar returned an invalid completion analysis");
                }
                // Never persist image payloads even if a provider mirrors request fields.
                analysis.erase("image_base64");
                analysis.erase("image");
                analysis.erase("images");
                const json candidates = tile_compile::pi::normalize_candidate_updates(analysis);
                const json validated = tile_compile::pi::validate_recommendation_updates(
                    candidates, schema.paths, base_config, state, pi_context);
                analysis["schema_version"] = "pi.run-completion-analysis.v1";
                analysis["status"] = "ready";
                analysis["has_analysis"] = true;
                analysis["from_cache"] = false;
                analysis["run_id"] = run_id;
                analysis["run_dir"] = run_dir.string();
                analysis["source_artifact"] = image_info;
                analysis["source_fingerprint"] = source_fingerprint;
                analysis["pi_context"] = pi_context;
                analysis["updates"] = candidates;
                analysis["validated_updates"] = validated["validated_updates"];
                analysis["rejected_updates"] = validated["rejected_updates"];
                analysis["validation"] = validated["validation"];
                analysis["config_yaml"] = validated["patched_config_yaml"];
                analysis["resume_recommendation"] = extract_resume_recommendation(analysis);
                analysis["analysis_id"] = analysis_id;
                if (!persist_completion_analysis(state, run_id, analysis)) {
                    throw std::runtime_error("Completion analysis could not be persisted");
                }
                const json ready_status = {
                    {"schema_version", "pi.run-completion-analysis.status.v1"},
                    {"status", "ready"},
                    {"has_analysis", true},
                    {"run_id", run_id},
                    {"analysis_id", analysis_id},
                    {"source_fingerprint", source_fingerprint}
                };
                if (!persist_completion_analysis_status(state, run_id, ready_status)) {
                    throw std::runtime_error("Completion analysis status could not be persisted");
                }
                state->job_store.update_state(analysis_id, JobState::ok, analysis);
            } catch (const std::exception& e) {
                json error_status = {
                    {"schema_version", "pi.run-completion-analysis.status.v1"},
                    {"status", "error"},
                    {"has_analysis", false},
                    {"run_id", run_id},
                    {"run_dir", run_dir.string()},
                    {"analysis_id", analysis_id},
                    {"source_artifact", image_info},
                    {"source_fingerprint", source_fingerprint},
                    {"error", {{"code", "COMPLETION_ANALYSIS_FAILED"}, {"message", e.what()}}}
                };
                persist_completion_analysis_status(state, run_id, error_status);
                state->job_store.update_state(analysis_id, JobState::error, error_status, e.what());
            }
        }).detach();

        return json_resp({
            {"schema_version", "pi.run-completion-analysis.status.v1"},
            {"status", "running"},
            {"has_analysis", false},
            {"run_id", run_id},
            {"run_dir", run_dir.string()},
            {"analysis_id", analysis_id},
            {"source_artifact", image_info},
            {"source_fingerprint", source_fingerprint}
        }, 202);
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

        // When the caller supplies an explicit base YAML (e.g. the parameter editor draft),
        // always patch onto that base and never use the analysis-time patched_config_yaml —
        // the editor draft may have diverged from the config that was originally analyzed.
        const bool has_caller_base_yaml = body->contains("yaml") && (*body)["yaml"].is_string()
            && !(*body)["yaml"].get<std::string>().empty();

        // If no specific paths are selected AND no caller base YAML is supplied AND the
        // analysis file carries a fully-validated patched_config_yaml (written at analysis
        // time), use it directly — identical to loading a preset config.  This avoids
        // re-patching against a potentially stale live config and guarantees the result
        // matches what was validated originally.
        static const json empty_ctx = json::object();
        const json& ctx = analysis_data.contains("analysis_context") && analysis_data["analysis_context"].is_object()
            ? analysis_data["analysis_context"]
            : empty_ctx;
        const bool has_full_patch = !has_selected_paths
            && !has_caller_base_yaml
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

        if (body->value("learn", false)) {
            try {
                tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
                json memory = build_apply_candidate_memory(
                    analysis_id,
                    analysis_data,
                    applied,
                    validation,
                    target_config_path,
                    body->value("persist", false));
                result["memory"] = store.append_candidate(std::move(memory));
            } catch (const std::exception& e) {
                result["memory_error"] = e.what();
            }
        }

        return json_resp(result);
    });
}
