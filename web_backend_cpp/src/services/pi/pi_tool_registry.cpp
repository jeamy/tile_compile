#include "services/pi/pi_tool_registry.hpp"

#include "app_state.hpp"
#include "services/pi/pi_context_builder.hpp"
#include "services/run_inspector.hpp"
#include "subprocess_manager.hpp"

#include <fstream>
#include <set>
#include <sstream>

namespace tile_compile::pi {
namespace {

nlohmann::json tool_def(const std::string& name,
                        const std::string& version,
                        const std::string& description) {
    return {
        {"name", name},
        {"tool_version", version},
        {"min_autonomy_level", "L0"},
        {"privacy_class", "metadata_only"},
        {"read_only", true},
        {"mutation_free", true},
        {"write_policy", "no_direct_writes_use_action_plan_preview_apply"},
        {"description", description},
        {"input_schema", {{"type", "object"}, {"additionalProperties", false}}},
        {"output_schema", {{"type", "object"}}}
    };
}

nlohmann::json load_schema_summary(const std::filesystem::path& schema_path) {
    std::ifstream in(schema_path);
    if (!in) {
        return {
            {"available", false},
            {"error", "schema_file_unavailable"}
        };
    }
    auto parsed = nlohmann::json::parse(in, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) {
        return {
            {"available", false},
            {"error", "schema_parse_failed"}
        };
    }

    nlohmann::json top_level = nlohmann::json::array();
    if (parsed.contains("properties") && parsed["properties"].is_object()) {
        for (auto it = parsed["properties"].begin(); it != parsed["properties"].end(); ++it) {
            nlohmann::json item = {
                {"path", it.key()}
            };
            if (it.value().is_object()) {
                if (it.value().contains("type")) item["type"] = it.value()["type"];
                if (it.value().contains("description")) item["description"] = it.value()["description"];
            }
            top_level.push_back(std::move(item));
        }
    }

    return {
        {"available", true},
        {"schema_version", parsed.value("$schema", std::string())},
        {"title", parsed.value("title", std::string())},
        {"top_level_properties", top_level},
        {"top_level_count", top_level.size()}
    };
}

std::string input_string(const nlohmann::json& input, const char* key) {
    if (!input.is_object() || !input.contains(key) || !input[key].is_string()) return "";
    return input[key].get<std::string>();
}

std::filesystem::path resolve_tool_run_dir(const std::shared_ptr<AppState>& state,
                                           const nlohmann::json& input,
                                           std::string& run_id) {
    run_id = input_string(input, "run_id");
    if (run_id.empty()) {
        std::lock_guard<std::mutex> lk(state->state_mutex);
        run_id = state->current_run_id;
    }
    if (run_id.empty()) {
        auto runs = discover_runs(state->runtime.runs_dir, 1);
        if (!runs.empty()) run_id = runs.front().value("run_id", std::string());
    }
    if (run_id.empty()) throw std::runtime_error("No run_id available for PI tool");
    return state->runtime.resolve_run_dir(run_id);
}

nlohmann::json summarize_artifacts(const nlohmann::json& artifacts) {
    nlohmann::json groups = nlohmann::json::object();
    nlohmann::json sample = nlohmann::json::array();
    if (!artifacts.is_array()) {
        return {{"count", 0}, {"groups", groups}, {"sample", sample}};
    }
    for (const auto& item : artifacts) {
        const std::string group = item.value("group", std::string("artifacts"));
        groups[group] = groups.value(group, 0) + 1;
        if (sample.size() < 12) {
            sample.push_back({
                {"path", item.value("relative_path", item.value("path", std::string()))},
                {"name", item.value("name", std::string())},
                {"size_bytes", item.value("size_bytes", item.value("size", 0))}
            });
        }
    }
    return {
        {"count", artifacts.size()},
        {"groups", groups},
        {"sample", sample}
    };
}

nlohmann::json read_json_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) return nullptr;
    auto parsed = nlohmann::json::parse(in, nullptr, false);
    if (parsed.is_discarded()) return nullptr;
    return parsed;
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) return "";
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

nlohmann::json parse_validation_result(const SubprocessResult& result) {
    auto parsed = nlohmann::json::parse(result.stdout_str, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) {
        return {
            {"valid", false},
            {"errors", nlohmann::json::array({"validate-config returned non-json output"})},
            {"warnings", nlohmann::json::array()}
        };
    }
    return parsed;
}

nlohmann::json run_config_validation(const std::shared_ptr<AppState>& state,
                                     const std::string& yaml_text) {
    if (yaml_text.empty()) {
        return {
            {"valid", false},
            {"errors", nlohmann::json::array({"no config yaml available for preview planning"})},
            {"warnings", nlohmann::json::array()}
        };
    }
    const auto result = run_subprocess({state->runtime.cli_exe, "validate-config", "--stdin"},
                                       state->runtime.project_root.string(),
                                       yaml_text);
    auto validation = parse_validation_result(result);
    validation["exit_code"] = result.exit_code;
    return validation;
}

nlohmann::json status_summary(const nlohmann::json& status) {
    return {
        {"status", status.value("status", std::string())},
        {"current_phase", status.contains("current_phase") ? status["current_phase"] : nlohmann::json(nullptr)},
        {"progress", status.contains("progress") ? status["progress"] : nlohmann::json(nullptr)}
    };
}

nlohmann::json phase_state(const nlohmann::json& status, const std::string& phase) {
    if (status.contains("phases") && status["phases"].is_array()) {
        for (const auto& item : status["phases"]) {
            if (!item.is_object()) continue;
            if (item.value("phase", std::string()) == phase) return item;
        }
    }
    return {
        {"phase", phase},
        {"status", "unknown"},
        {"pct", 0.0}
    };
}

nlohmann::json prerequisite_phase_summary(const nlohmann::json& status, const std::string& phase) {
    nlohmann::json result = {
        {"ok", true},
        {"items", nlohmann::json::array()}
    };
    if (!status.contains("phases") || !status["phases"].is_array()) return result;
    for (const auto& item : status["phases"]) {
        if (!item.is_object()) continue;
        const std::string item_phase = item.value("phase", std::string());
        if (item_phase == phase) break;
        const std::string item_status = item.value("status", std::string("unknown"));
        const bool phase_ok = item_status == "ok" || item_status == "skipped";
        if (!phase_ok) result["ok"] = false;
        result["items"].push_back({
            {"phase", item_phase},
            {"status", item_status},
            {"ok", phase_ok}
        });
    }
    return result;
}

nlohmann::json build_phase_preview_plan(const std::shared_ptr<AppState>& state,
                                        const nlohmann::json& input,
                                        const std::string& action_type,
                                        const std::string& phase) {
    std::string run_id;
    const auto run_dir = resolve_tool_run_dir(state, input, run_id);
    const bool run_exists = std::filesystem::exists(run_dir) && std::filesystem::is_directory(run_dir);
    const auto status = run_exists ? read_run_status(run_dir) : nlohmann::json::object();

    std::string yaml_text;
    std::string config_source;
    if (input.contains("yaml") && input["yaml"].is_string()) {
        yaml_text = input["yaml"].get<std::string>();
        config_source = "request.yaml";
    } else {
        const auto run_config = run_dir / "config.yaml";
        yaml_text = read_text_file(run_config);
        if (!yaml_text.empty()) {
            config_source = "run.config";
        } else {
            yaml_text = read_text_file(state->runtime.default_config_path);
            config_source = yaml_text.empty() ? "unavailable" : "default_config";
        }
    }

    const auto validation = run_config_validation(state, yaml_text);
    const bool config_valid = validation.value("valid", false);
    const auto artifacts = run_exists ? list_run_artifacts(run_dir) : nlohmann::json::array();
    const auto stats = run_exists ? read_json_file(run_dir / "artifacts" / "stats.json") : nlohmann::json(nullptr);
    const auto prerequisites = prerequisite_phase_summary(status, phase);
    return {
        {"schema_version", "pi.preview-plan.v1"},
        {"mutation_free", true},
        {"tool_action", action_type},
        {"phase", phase},
        {"run_id", run_id},
        {"run_dir", {
            {"name", run_dir.filename().string()},
            {"exists", run_exists}
        }},
        {"config_source", config_source},
        {"config_valid", config_valid},
        {"validation", validation},
        {"ready", run_exists && config_valid},
        {"status", status_summary(status)},
        {"phase_state", phase_state(status, phase)},
        {"prerequisites", prerequisites},
        {"artifact_summary", summarize_artifacts(artifacts)},
        {"report_stats_available", !stats.is_null()},
        {"report_stats", stats.is_null() ? nlohmann::json::object() : stats},
        {"planned_command", {
            {"executable", "tile_compile_runner"},
            {"args", nlohmann::json::array({"resume", "--run-dir", run_dir.string(), "--from-phase", phase})}
        }},
        {"safety", {
            {"will_start_job", false},
            {"requires_user_review", true},
            {"requires_apply_endpoint", true}
        }}
    };
}

} // namespace

PiToolRegistry::PiToolRegistry(std::shared_ptr<AppState> state)
    : _state(std::move(state)) {}

nlohmann::json PiToolRegistry::list_tools() const {
    return {
        {"schema_version", "pi.tools-list.v1"},
        {"tool_registry_version", "1.1.0"},
        {"tools", nlohmann::json::array({
            tool_def("context.overview", "1.0.0", "Return compact read-only Tile Compile runtime, state, and recent-job context."),
            tool_def("config.schema.summary", "1.0.0", "Return compact read-only summary of the Tile Compile config schema."),
            tool_def("run.artifacts.summary", "1.0.0", "Return compact read-only summary of artifacts for a run."),
            tool_def("run.report.summary", "1.0.0", "Return compact read-only run status and existing report summary metadata."),
            tool_def("preview.bge.plan", "1.0.0", "Validate and return a mutation-free BGE resume preview plan for a run."),
            tool_def("preview.hms.plan", "1.0.0", "Validate and return a mutation-free Hypermetric Stretch resume preview plan for a run."),
            tool_def("preview.resume.plan", "1.0.0", "Validate and return a mutation-free resume preview plan for a requested phase.")
        })}
    };
}

nlohmann::json PiToolRegistry::call_tool(const std::string& name, const nlohmann::json& input) const {
    if (name == "context.overview") {
        PiContextBuilder builder(_state);
        return {
            {"ok", true},
            {"tool", name},
            {"result", builder.build_overview_context()}
        };
    }
    if (name == "config.schema.summary") {
        return {
            {"ok", true},
            {"tool", name},
            {"result", load_schema_summary(_state->runtime.schema_path)}
        };
    }
    if (name == "run.artifacts.summary") {
        try {
            std::string run_id;
            const auto run_dir = resolve_tool_run_dir(_state, input, run_id);
            return {
                {"ok", true},
                {"tool", name},
                {"result", {
                    {"run_id", run_id},
                    {"artifacts", summarize_artifacts(list_run_artifacts(run_dir))}
                }}
            };
        } catch (const std::exception& e) {
            return {{"ok", false}, {"error", {{"code", "RUN_CONTEXT_UNAVAILABLE"}, {"message", e.what()}}}};
        }
    }
    if (name == "run.report.summary") {
        try {
            std::string run_id;
            const auto run_dir = resolve_tool_run_dir(_state, input, run_id);
            const auto status = read_run_status(run_dir);
            const auto artifacts = list_run_artifacts(run_dir);
            const auto stats = read_json_file(run_dir / "artifacts" / "stats.json");
            return {
                {"ok", true},
                {"tool", name},
                {"result", {
                    {"run_id", run_id},
                    {"status", status},
                    {"artifact_summary", summarize_artifacts(artifacts)},
                    {"report_stats_available", !stats.is_null()},
                    {"report_stats", stats.is_null() ? nlohmann::json::object() : stats}
                }}
            };
        } catch (const std::exception& e) {
            return {{"ok", false}, {"error", {{"code", "RUN_CONTEXT_UNAVAILABLE"}, {"message", e.what()}}}};
        }
    }
    if (name == "preview.bge.plan" || name == "preview.hms.plan" || name == "preview.resume.plan") {
        try {
            std::string phase = "BGE";
            std::string action_type = "preview.bge";
            if (name == "preview.hms.plan") {
                phase = "HYPERMETRIC_STRETCH";
                action_type = "preview.hms";
            } else if (name == "preview.resume.plan") {
                phase = input_string(input, "from_phase");
                if (phase.empty()) phase = input_string(input, "phase");
                if (phase.empty()) throw std::runtime_error("from_phase is required for preview.resume.plan");
                action_type = "preview.resume";
            }
            return {
                {"ok", true},
                {"tool", name},
                {"result", build_phase_preview_plan(_state, input, action_type, phase)}
            };
        } catch (const std::exception& e) {
            return {{"ok", false}, {"error", {{"code", "RUN_CONTEXT_UNAVAILABLE"}, {"message", e.what()}}}};
        }
    }
    return {
        {"ok", false},
        {"error", {
            {"code", "UNKNOWN_PI_TOOL"},
            {"message", "Unknown PI tool: " + name}
        }}
    };
}

} // namespace tile_compile::pi
