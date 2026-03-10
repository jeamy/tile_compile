#include "routes/config_routes.hpp"
#include "subprocess_manager.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

namespace {

crow::response json_resp(const nlohmann::json& j, int status = 200) {
    crow::response res(status, j.dump());
    res.set_header("Content-Type", "application/json");
    return res;
}

crow::response err_resp(const std::string& msg, int status = 400) {
    return json_resp({{"error", {{"message", msg}}}}, status);
}

crow::response err_resp(const std::string& code,
                        const std::string& msg,
                        int status,
                        const nlohmann::json& details = nlohmann::json::object()) {
    return json_resp({{"error", {{"code", code}, {"message", msg}, {"details", details}}}}, status);
}

std::string read_file_str(const fs::path& p) {
    std::ifstream f(p);
    if (!f) return "";
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

nlohmann::json allowed_roots_json(const BackendRuntime& runtime) {
    nlohmann::json roots = nlohmann::json::array();
    for (const auto& root : runtime.allowed_roots()) roots.push_back(root.string());
    return roots;
}

std::optional<crow::response> validate_path(const std::shared_ptr<AppState>& state,
                                            fs::path& path,
                                            const std::string& label,
                                            bool must_exist = false) {
    auto resolved = state->runtime.resolve_input_path(path, must_exist);
    path = resolved.path;
    if (resolved.status == PathStatus::not_allowed) {
        return err_resp("PATH_NOT_ALLOWED", label + " is outside allowed roots", 422, {{"path", path.string()}, {"allowed_roots", allowed_roots_json(state->runtime)}});
    }
    if (resolved.status == PathStatus::not_found) {
        return err_resp("PATH_NOT_FOUND", label + " does not exist", 422, {{"path", path.string()}});
    }
    return std::nullopt;
}

std::optional<nlohmann::json> parse_json(const std::string& raw) {
    auto parsed = nlohmann::json::parse(raw, nullptr, false);
    if (parsed.is_discarded()) return std::nullopt;
    return parsed;
}

crow::response backend_command_failed(const std::string& message, const SubprocessResult& result) {
    return err_resp("BACKEND_COMMAND_FAILED", message, 502, {
        {"exit_code", result.exit_code},
        {"stdout", result.stdout_str},
        {"stderr", result.stderr_str},
    });
}

YAML::Node json_to_yaml_node(const nlohmann::json& value) {
    if (value.is_object()) {
        YAML::Node node(YAML::NodeType::Map);
        for (auto it = value.begin(); it != value.end(); ++it) node[it.key()] = json_to_yaml_node(it.value());
        return node;
    }
    if (value.is_array()) {
        YAML::Node node(YAML::NodeType::Sequence);
        for (const auto& item : value) node.push_back(json_to_yaml_node(item));
        return node;
    }
    if (value.is_boolean()) return YAML::Node(value.get<bool>());
    if (value.is_number_integer()) return YAML::Node(value.get<long long>());
    if (value.is_number_unsigned()) return YAML::Node(value.get<unsigned long long>());
    if (value.is_number_float()) return YAML::Node(value.get<double>());
    if (value.is_null()) return YAML::Node();
    return YAML::Node(value.get<std::string>());
}

nlohmann::json yaml_to_json(const YAML::Node& node) {
    if (!node || node.IsNull()) return nullptr;
    if (node.IsMap()) {
        nlohmann::json out = nlohmann::json::object();
        for (auto it = node.begin(); it != node.end(); ++it) out[it->first.as<std::string>()] = yaml_to_json(it->second);
        return out;
    }
    if (node.IsSequence()) {
        nlohmann::json out = nlohmann::json::array();
        for (auto it = node.begin(); it != node.end(); ++it) out.push_back(yaml_to_json(*it));
        return out;
    }
    try { return node.as<bool>(); } catch (...) {}
    try { return node.as<long long>(); } catch (...) {}
    try { return node.as<double>(); } catch (...) {}
    try { return node.as<std::string>(); } catch (...) {}
    return nullptr;
}

std::string yaml_dump(const nlohmann::json& value) {
    std::ostringstream oss;
    oss << json_to_yaml_node(value);
    return oss.str();
}

nlohmann::json parse_scalar_value(const nlohmann::json& raw_value, bool parse_values) {
    if (!parse_values || !raw_value.is_string()) return raw_value;
    try {
        return yaml_to_json(YAML::Load(raw_value.get<std::string>()));
    } catch (...) {
        return raw_value;
    }
}

void set_dotted(nlohmann::json& root, const std::string& dotted_path, const nlohmann::json& value) {
    std::vector<std::string> parts;
    std::istringstream iss(dotted_path);
    std::string part;
    while (std::getline(iss, part, '.')) {
        if (!part.empty()) parts.push_back(part);
    }
    if (parts.empty()) return;

    nlohmann::json* node = &root;
    for (size_t i = 0; i + 1 < parts.size(); ++i) {
        if (!node->contains(parts[i]) || !(*node)[parts[i]].is_object()) (*node)[parts[i]] = nlohmann::json::object();
        node = &(*node)[parts[i]];
    }
    (*node)[parts.back()] = value;
}

} // namespace

void register_config_routes(CrowApp& app,
                            std::shared_ptr<AppState> state,
                            std::shared_ptr<ConfigRevisionStore> /*unused*/) {

    CROW_ROUTE(app, "/api/config/schema").methods("GET"_method)
    ([state](const crow::request&) {
        SubprocessResult res = run_subprocess({state->runtime.cli_exe, "get-schema"}, state->runtime.project_root.string());
        auto parsed = parse_json(res.stdout_str);
        if (res.exit_code != 0 || !parsed || !parsed->is_object()) return backend_command_failed("failed to fetch schema", res);
        return json_resp(*parsed);
    });

    CROW_ROUTE(app, "/api/config/current").methods("GET"_method)
    ([state](const crow::request& req) {
        fs::path config_path = req.url_params.get("path") ? fs::path(req.url_params.get("path")) : state->runtime.default_config_path;
        if (auto err = validate_path(state, config_path, "config_path", true)) return std::move(*err);

        SubprocessResult res = run_subprocess({state->runtime.cli_exe, "load-config", config_path.string()}, state->runtime.project_root.string());
        auto parsed = parse_json(res.stdout_str);
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
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded()) body = nlohmann::json::object();

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
        auto parsed = parse_json(res.stdout_str);
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
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded()) return err_resp("Invalid JSON");

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
        auto parsed = parse_json(res.stdout_str);
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
    ([state]() {
        nlohmann::json items = nlohmann::json::array();
        if (!fs::is_directory(state->runtime.presets_dir)) return json_resp({{"items", items}});
        for (auto& entry : fs::directory_iterator(state->runtime.presets_dir)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            if (ext != ".yaml" && ext != ".yml") continue;
            if (entry.path().filename().string().find("example") == std::string::npos) continue;
            items.push_back({{"id", entry.path().stem().string()}, {"name", entry.path().filename().string()}, {"path", entry.path().string()}});
        }
        return json_resp({{"items", items}});
    });

    CROW_ROUTE(app, "/api/config/presets/apply").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("path") || !body["path"].is_string()) {
            return err_resp("BAD_REQUEST", "path is required", 400);
        }

        fs::path preset_path = fs::path(body["path"].get<std::string>());
        if (preset_path.is_relative()) preset_path = state->runtime.project_root / preset_path;
        if (auto err = validate_path(state, preset_path, "preset_path", true)) return std::move(*err);

        SubprocessResult res = run_subprocess({state->runtime.cli_exe, "load-config", preset_path.string()}, state->runtime.project_root.string());
        auto parsed = parse_json(res.stdout_str);
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
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded()) return err_resp("Invalid JSON");

        fs::path target = body.contains("path") && body["path"].is_string()
            ? fs::path(body["path"].get<std::string>())
            : state->runtime.default_config_path;
        if (auto err = validate_path(state, target, "config_path")) return std::move(*err);

        nlohmann::json base = nlohmann::json::object();
        if (body.contains("yaml") && body["yaml"].is_string()) {
            try {
                base = yaml_to_json(YAML::Load(body["yaml"].get<std::string>()));
            } catch (const std::exception& e) {
                return err_resp("BAD_REQUEST", std::string("YAML parse error: ") + e.what(), 400);
            }
        } else if (body.contains("config") && body["config"].is_object()) {
            base = body["config"];
        } else {
            std::string current_text = read_file_str(target);
            if (!current_text.empty()) {
                try {
                    base = yaml_to_json(YAML::Load(current_text));
                } catch (const std::exception& e) {
                    return err_resp("BAD_REQUEST", std::string("YAML parse error: ") + e.what(), 400);
                }
            }
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
            auto parsed = parse_json(res.stdout_str);
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
