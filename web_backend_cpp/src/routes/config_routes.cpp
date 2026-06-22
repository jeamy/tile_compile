#include "routes/config_routes.hpp"
#include "routes/route_utils.hpp"
#include "subprocess_manager.hpp"
#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;
using namespace tile_compile::routes;

namespace {

/// @brief Parses scalar value.
/// @details This implementation serves configuration loading, mutation, validation, and revision endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
nlohmann::json parse_scalar_value(const nlohmann::json& raw_value, bool parse_values) {
    if (!parse_values || !raw_value.is_string()) return raw_value;

    const std::string text = raw_value.get<std::string>();
    std::string trimmed = text;
    trimmed.erase(trimmed.begin(), std::find_if(trimmed.begin(), trimmed.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    trimmed.erase(std::find_if(trimmed.rbegin(), trimmed.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), trimmed.end());

    if (trimmed.empty()) return raw_value;

    std::string lowered = trimmed;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    if (lowered == "true") return true;
    if (lowered == "false") return false;
    if (lowered == "null" || lowered == "~") return nullptr;

    char* end = nullptr;
    errno = 0;
    const double numeric = std::strtod(trimmed.c_str(), &end);
    if (errno == 0 && end != trimmed.c_str() && end && *end == '\0' && std::isfinite(numeric)) {
        if (trimmed.find('.') == std::string::npos &&
            trimmed.find('e') == std::string::npos &&
            trimmed.find('E') == std::string::npos) {
            try {
                return std::stoll(trimmed);
            } catch (...) {
                return numeric;
            }
        }
        return numeric;
    }

    const bool looks_structured_yaml =
        !trimmed.empty() &&
        (trimmed.front() == '{' || trimmed.front() == '[' || trimmed.find('\n') != std::string::npos);
    if (looks_structured_yaml) {
        if (auto parsed = parse_yaml_text(trimmed)) {
            return *parsed;
        }
        return raw_value;
    }

    return raw_value;
}

} // namespace

/// @brief Registers configuration endpoints for YAML loading, validation, saving, and revision history.
/// @details This is the route-group entry point called from main during Crow setup.
void register_config_routes(CrowApp& app,
                            std::shared_ptr<AppState> state,
                            std::shared_ptr<ConfigRevisionStore> /*unused*/) {

    CROW_ROUTE(app, "/api/config/schema").methods("GET"_method)
    ([state](const crow::request&) {
        SubprocessResult res = run_subprocess({state->runtime.cli_exe, "get-schema"}, state->runtime.project_root.string());
        auto parsed = parse_json_string(res.stdout_str);
        if (res.exit_code != 0 || !parsed || !parsed->is_object()) return backend_command_failed("failed to fetch schema", res);
        return json_resp(*parsed);
    });

    CROW_ROUTE(app, "/api/config/defaults").methods("GET"_method)
    ([state](const crow::request&) {
        SubprocessResult res = run_subprocess({state->runtime.cli_exe, "dump-default-config"}, state->runtime.project_root.string());
        auto parsed = parse_json_string(res.stdout_str);
        if (res.exit_code != 0 || !parsed || !parsed->is_object()) return backend_command_failed("failed to fetch defaults", res);
        return json_resp(*parsed);
    });

    CROW_ROUTE(app, "/api/config/current").methods("GET"_method)
    ([state](const crow::request& req) {
        fs::path config_path = req.url_params.get("path") ? fs::path(req.url_params.get("path")) : state->runtime.default_config_path;
        if (auto err = validate_path(state, config_path, "config_path", true)) return std::move(*err);

        SubprocessResult res = run_subprocess({state->runtime.cli_exe, "load-config", config_path.string()}, state->runtime.project_root.string());
        auto parsed = parse_json_string(res.stdout_str);
        if (res.exit_code == 0 && parsed && parsed->is_object()) {
            return json_resp({{"config", parsed->value("yaml", std::string())}, {"source", config_path.string()}});
        }

        std::ifstream in(config_path);
        if (!in) return backend_command_failed("failed to load config", res);
        std::string yaml_text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        return json_resp({{"config", yaml_text}, {"source", config_path.string()}, {"fallback", "file_read"}});
    });

    CROW_ROUTE(app, "/api/config/validate").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req).value_or(nlohmann::json::object());

        bool strict = body.value("strict_exit_codes", false);
        std::vector<std::string> cmd = {state->runtime.cli_exe, "validate-config"};
        std::string stdin_text;

        if (body.contains("path") && body["path"].is_string() && !body["path"].get<std::string>().empty()) {
            fs::path config_path = body["path"].get<std::string>();
            if (auto err = validate_path(state, config_path, "config_path", true)) return std::move(*err);
            cmd.push_back("--path");
            cmd.push_back(config_path.string());
        } else if (body.contains("yaml") && body["yaml"].is_string()) {
            stdin_text = body["yaml"].get<std::string>();
            cmd.push_back("--stdin");
        } else if (body.contains("config") && body["config"].is_object()) {
            stdin_text = yaml_dump(body["config"]);
            cmd.push_back("--stdin");
        } else {
            return err_resp("BAD_REQUEST", "provide one of: path, yaml, or config", 400);
        }

        if (strict) cmd.push_back("--strict-exit-codes");
        SubprocessResult res = run_subprocess(cmd, state->runtime.project_root.string(), stdin_text);
        auto parsed = parse_json_string(res.stdout_str);
        if (!parsed || !parsed->is_object()) {
            nlohmann::json details = nlohmann::json::array();
            if (!res.stderr_str.empty()) details.push_back("stderr: " + res.stderr_str);
            if (!res.stdout_str.empty()) details.push_back("stdout: " + res.stdout_str);
            if (details.empty()) details.push_back("validate-config returned non-json output");
            return json_resp({{"ok", false}, {"errors", details}, {"warnings", nlohmann::json::array({"CLI validation backend returned unexpected output"})}});
        }

        return json_resp({
            {"ok", parsed->value("valid", false)},
            {"errors", parsed->value("errors", nlohmann::json::array())},
            {"warnings", parsed->value("warnings", nlohmann::json::array())},
        });
    });

    CROW_ROUTE(app, "/api/config/save").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body_opt = parse_body(req);
        if (!body_opt) return err_resp("Invalid JSON");
        auto& body = *body_opt;

        fs::path target = body.contains("path") && body["path"].is_string()
            ? fs::path(body["path"].get<std::string>())
            : state->runtime.default_config_path;
        if (auto err = validate_path(state, target, "config_path")) return std::move(*err);

        std::string yaml_text;
        if (body.contains("yaml") && body["yaml"].is_string()) {
            yaml_text = body["yaml"].get<std::string>();
        } else if (body.contains("config") && body["config"].is_object()) {
            yaml_text = yaml_dump(body["config"]);
        } else {
            return err_resp("BAD_REQUEST", "provide yaml or config object", 400);
        }

        SubprocessResult res = run_subprocess({state->runtime.cli_exe, "save-config", target.string(), "--stdin"},
                                              state->runtime.project_root.string(),
                                              yaml_text);
        auto parsed = parse_json_string(res.stdout_str);
        if (res.exit_code != 0 || !parsed || !parsed->is_object()) return backend_command_failed("save-config failed", res);

        fs::path saved_path = parsed->contains("path") && (*parsed)["path"].is_string()
            ? fs::path((*parsed)["path"].get<std::string>())
            : target;
        std::string rev_id = state->revision_store.add(saved_path, yaml_text, "save_config");
        {
            std::lock_guard<std::mutex> lk(state->state_mutex);
            state->active_config_revision_id = rev_id;
        }
        state->ui_event_store.push("config.save", "config.save", {{"path", saved_path.string()}, {"saved", parsed->value("saved", false)}, {"revision_id", rev_id}});
        return json_resp({{"path", saved_path.string()}, {"saved", parsed->value("saved", false)}, {"revision_id", rev_id}});
    });

    CROW_ROUTE(app, "/api/config/presets").methods("GET"_method)
    ([state](const crow::request& req) {
        nlohmann::json items = nlohmann::json::array();
        fs::path presets_dir = state->runtime.presets_dir;
        bool fallback_used = false;
        if (const char* dir_arg = req.url_params.get("dir")) {
            std::string raw_dir = dir_arg;
            if (!raw_dir.empty()) {
                fs::path requested_dir = fs::path(raw_dir);
                if (auto err = validate_path(state, requested_dir, "presets_dir", true)) {
                    const auto status = err->code;
                    if (status == 403) return std::move(*err);
                    fallback_used = true;
                } else if (!fs::is_directory(requested_dir)) {
                    fallback_used = true;
                } else {
                    presets_dir = requested_dir;
                }
            }
        }
        if (!fs::is_directory(presets_dir)) return json_resp({{"items", items}, {"dir", presets_dir.string()}, {"fallback_used", fallback_used}});
        for (auto& entry : fs::directory_iterator(presets_dir)) {
            if (entry.is_directory()) {
                items.push_back({{"id", entry.path().filename().string()}, {"name", entry.path().filename().string()}, {"path", entry.path().string()}, {"is_dir", true}});
            } else if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                if (ext != ".yaml" && ext != ".yml") continue;
                items.push_back({{"id", entry.path().stem().string()}, {"name", entry.path().filename().string()}, {"path", entry.path().string()}, {"is_dir", false}});
            }
        }
        return json_resp({{"items", items}, {"dir", presets_dir.string()}, {"fallback_used", fallback_used}});
    });

    CROW_ROUTE(app, "/api/config/presets/apply").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body_opt = parse_body(req);
        if (!body_opt || !body_opt->contains("path") || !(*body_opt)["path"].is_string()) {
            return err_resp("BAD_REQUEST", "path is required", 400);
        }
        auto& body = *body_opt;

        fs::path preset_path = fs::path(body["path"].get<std::string>());
        if (preset_path.is_relative()) preset_path = state->runtime.project_root / preset_path;
        if (auto err = validate_path(state, preset_path, "preset_path", true)) return std::move(*err);

        SubprocessResult res = run_subprocess({state->runtime.cli_exe, "load-config", preset_path.string()}, state->runtime.project_root.string());
        auto parsed = parse_json_string(res.stdout_str);
        if (res.exit_code != 0 || !parsed || !parsed->is_object()) return backend_command_failed("load-config failed", res);

        state->ui_event_store.push("config.preset.apply", "config.presets_apply", {{"preset_path", preset_path.string()}});
        return json_resp({{"config", parsed->value("yaml", std::string())}, {"applied_paths", nlohmann::json::array({preset_path.string()})}});
    });

    CROW_ROUTE(app, "/api/config/revisions").methods("GET"_method)
    ([state]() {
        auto revisions = state->revision_store.list();
        nlohmann::json items = nlohmann::json::array();
        for (const auto& revision : revisions) items.push_back(config_revision_to_json(revision));
        return json_resp({{"items", items}, {"active_revision_id", state->active_config_revision_id}});
    });

    CROW_ROUTE(app, "/api/config/revisions/<string>/restore").methods("POST"_method)
    ([state](const crow::request&, std::string rev_id) {
        auto rev = state->revision_store.get(rev_id);
        if (!rev) return err_resp("NOT_FOUND", "revision '" + rev_id + "' not found", 404);

        fs::path target = rev->path.empty() ? state->runtime.default_config_path : fs::path(rev->path);
        if (auto err = validate_path(state, target, "revision_path")) return std::move(*err);

        if (!target.parent_path().empty()) fs::create_directories(target.parent_path());
        if (!rev->yaml_text.empty()) {
            std::ofstream out(target);
            if (!out) return err_resp("BACKEND_COMMAND_FAILED", "failed to restore revision", 502, {{"path", target.string()}});
            out << rev->yaml_text;
        } else if (!fs::exists(target)) {
            std::ofstream out(target);
            if (!out) return err_resp("BACKEND_COMMAND_FAILED", "failed to restore revision", 502, {{"path", target.string()}});
            out << "{}\n";
        }

        {
            std::lock_guard<std::mutex> lk(state->state_mutex);
            state->active_config_revision_id = rev_id;
        }
        state->ui_event_store.push("config.revision.restore", "config.revision_restore", {{"revision_id", rev_id}, {"path", target.string()}});
        return json_resp({{"ok", true}, {"active_revision_id", rev_id}});
    });

    CROW_ROUTE(app, "/api/config/patch").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body_opt = parse_body(req);
        if (!body_opt) return err_resp("Invalid JSON");
        auto& body = *body_opt;

        fs::path target = body.contains("path") && body["path"].is_string()
            ? fs::path(body["path"].get<std::string>())
            : state->runtime.default_config_path;
        if (auto err = validate_path(state, target, "config_path")) return std::move(*err);

        nlohmann::json base = nlohmann::json::object();
        if (body.contains("yaml") && body["yaml"].is_string()) {
            auto parsed = parse_yaml_text(body["yaml"].get<std::string>());
            if (!parsed) return err_resp("BAD_REQUEST", "YAML parse error: invalid YAML", 400);
            base = *parsed;
        } else if (body.contains("config") && body["config"].is_object()) {
            base = body["config"];
        } else {
            auto parsed = parse_yaml_file(target);
            if (parsed) base = *parsed;
        }

        if (!base.is_object()) {
            return err_resp("BAD_REQUEST", "base config must be a mapping", 400);
        }
        if (body.contains("updates") && !body["updates"].is_array()) {
            return err_resp("BAD_REQUEST", "updates must be a list", 400);
        }

        const bool parse_values = body.value("parse_values", true);
        const bool persist = body.value("persist", false);
        nlohmann::json applied = nlohmann::json::array();
        if (body.contains("updates") && body["updates"].is_array()) {
            for (const auto& entry : body["updates"]) {
                if (!entry.is_object()) continue;
                const std::string dotted = entry.value("path", "");
                if (dotted.empty()) continue;
                nlohmann::json value = entry.contains("value") ? parse_scalar_value(entry["value"], parse_values) : nlohmann::json(nullptr);
                set_dotted(base, dotted, value);
                applied.push_back({{"path", dotted}, {"value", value}});
            }
        }

        const std::string merged_yaml = yaml_dump(base);
        nlohmann::json result = {
            {"path", target.string()},
            {"config", base},
            {"config_yaml", merged_yaml},
            {"applied", applied},
        };

        if (persist) {
            SubprocessResult res = run_subprocess({state->runtime.cli_exe, "save-config", target.string(), "--stdin"},
                                                  state->runtime.project_root.string(),
                                                  merged_yaml);
            auto parsed = parse_json_string(res.stdout_str);
            if (res.exit_code != 0 || !parsed || !parsed->is_object()) return backend_command_failed("save-config failed", res);

            fs::path saved_path = parsed->contains("path") && (*parsed)["path"].is_string()
                ? fs::path((*parsed)["path"].get<std::string>())
                : target;
            std::string rev_id = state->revision_store.add(saved_path, merged_yaml, "config_patch");
            {
                std::lock_guard<std::mutex> lk(state->state_mutex);
                state->active_config_revision_id = rev_id;
            }
            state->ui_event_store.push("config.patch.save", "config.patch", {{"path", saved_path.string()}, {"revision_id", rev_id}, {"applied_count", static_cast<int>(applied.size())}});
            result["saved"] = parsed->value("saved", false);
            result["revision_id"] = rev_id;
            result["path"] = saved_path.string();
        }

        return json_resp(result);
    });
}
