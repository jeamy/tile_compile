#include "routes/config_routes.hpp"
#include "subprocess_manager.hpp"
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

static crow::response json_resp(const nlohmann::json& j, int status = 200) {
    crow::response res(status, j.dump());
    res.set_header("Content-Type", "application/json");
    return res;
}
static crow::response err_resp(const std::string& msg, int status = 400) {
    return json_resp({{"error", {{"message", msg}}}}, status);
}
static crow::response err_resp(const std::string& code,
                               const std::string& msg,
                               int status,
                               const nlohmann::json& details) {
    return json_resp({{"error", {{"code", code}, {"message", msg}, {"details", details}}}}, status);
}

static std::string read_file_str(const fs::path& p) {
    std::ifstream f(p);
    if (!f) return "";
    return std::string((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
}

static YAML::Node json_to_yaml_node(const nlohmann::json& value) {
    if (value.is_object()) {
        YAML::Node node(YAML::NodeType::Map);
        for (auto it = value.begin(); it != value.end(); ++it) {
            node[it.key()] = json_to_yaml_node(it.value());
        }
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

static nlohmann::json yaml_to_json(const YAML::Node& node) {
    if (!node || node.IsNull()) return nullptr;
    if (node.IsMap()) {
        nlohmann::json out = nlohmann::json::object();
        for (auto it = node.begin(); it != node.end(); ++it) {
            out[it->first.as<std::string>()] = yaml_to_json(it->second);
        }
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

// Equivalent to Python's _set_dotted(yaml_text, path, value, parse_values=True)
static void set_dotted(YAML::Node& root, const std::string& dotted_path,
                       const std::string& value_str, bool parse_values = true) {
    std::vector<std::string> keys;
    std::istringstream iss(dotted_path);
    std::string key;
    while (std::getline(iss, key, '.')) keys.push_back(key);
    if (keys.empty()) return;

    YAML::Node cur = root;
    for (size_t i = 0; i + 1 < keys.size(); ++i) {
        if (!cur[keys[i]]) cur[keys[i]] = YAML::Node(YAML::NodeType::Map);
        cur = cur[keys[i]];
    }
    if (parse_values) {
        try {
            cur[keys.back()] = YAML::Load(value_str);
            return;
        } catch (...) {}
    }
    cur[keys.back()] = value_str;
}

void register_config_routes(CrowApp& app,
                              std::shared_ptr<AppState> state,
                              std::shared_ptr<ConfigRevisionStore> /*unused*/) {

    CROW_ROUTE(app, "/api/config/current").methods("GET"_method)
    ([state](const crow::request& req) {
        std::string path = req.url_params.get("path") ? req.url_params.get("path") : "";
        fs::path config_path = path.empty() ? state->runtime.default_config_path : fs::path(path);
        if (!state->runtime.is_path_allowed(config_path)) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + config_path.string(), 403, {{"path", config_path.string()}});
        }
        std::string yaml_text = read_file_str(config_path);
        return json_resp({{"config", yaml_text},
                          {"source", config_path.string()}});
    });

    CROW_ROUTE(app, "/api/config/validate").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("yaml"))
            return err_resp("Missing 'yaml'");
        std::string yaml_text = body["yaml"].get<std::string>();

        SubprocessResult res = run_subprocess({
            state->runtime.cli_exe, "validate-config",
            "--stdin"
        }, state->runtime.project_root.string(), yaml_text);

        if (res.exit_code == 0) {
            try {
                auto j = nlohmann::json::parse(res.stdout_str);
                return json_resp({
                    {"ok", j.value("valid", false)},
                    {"valid", j.value("valid", false)},
                    {"errors", j.value("errors", nlohmann::json::array())},
                    {"warnings", nlohmann::json::array()}
                });
            } catch (...) {}
            return json_resp({{"ok", true}, {"errors", nlohmann::json::array()},
                              {"warnings", nlohmann::json::array()}});
        }
        try {
            auto j = nlohmann::json::parse(res.stdout_str);
            return json_resp({
                {"ok", j.value("valid", false)},
                {"valid", j.value("valid", false)},
                {"errors", j.value("errors", nlohmann::json::array())},
                {"warnings", nlohmann::json::array()}
            });
        } catch (...) {}
        return json_resp({{"ok", false},
                          {"errors", nlohmann::json::array({res.stderr_str})},
                          {"warnings", nlohmann::json::array()}});
    });

    CROW_ROUTE(app, "/api/config/save").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded()) return err_resp("Invalid JSON");
        std::string target_path = body.value("path", state->runtime.default_config_path.string());
        if (!state->runtime.is_path_allowed(fs::path(target_path))) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + target_path, 403, {{"path", target_path}});
        }

        std::string yaml_text;
        if (body.contains("yaml") && body["yaml"].is_string()) {
            yaml_text = body["yaml"].get<std::string>();
        } else if (body.contains("config") && body["config"].is_object()) {
            std::ostringstream oss;
            oss << json_to_yaml_node(body["config"]);
            yaml_text = oss.str();
        } else {
            return err_resp("BAD_REQUEST", "provide yaml or config object", 400, nlohmann::json::object());
        }

        std::ofstream out(target_path);
        if (!out) return err_resp("Cannot write: " + target_path, 500);
        out << yaml_text;

        std::string rev_id = state->revision_store.add(yaml_text, "save");
        {
            std::lock_guard<std::mutex> lk(state->state_mutex);
            state->active_config_revision_id = rev_id;
        }
        state->ui_event_store.push("config_saved", {{"path", target_path}, {"revision_id", rev_id}});
        return json_resp({{"path", target_path}, {"saved", true}, {"revision_id", rev_id}});
    });

    CROW_ROUTE(app, "/api/config/patch").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded()) return err_resp("Invalid JSON");

        fs::path target_path = body.contains("path") && body["path"].is_string()
            ? fs::path(body["path"].get<std::string>())
            : state->runtime.default_config_path;
        if (!state->runtime.is_path_allowed(target_path)) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + target_path.string(), 403, {{"path", target_path.string()}});
        }

        std::string base_yaml;
        if (body.contains("yaml") && body["yaml"].is_string())
            base_yaml = body["yaml"].get<std::string>();
        else if (body.contains("config") && body["config"].is_object()) {
            std::ostringstream oss;
            oss << json_to_yaml_node(body["config"]);
            base_yaml = oss.str();
        }
        else
            base_yaml = read_file_str(target_path);

        bool parse_values = body.value("parse_values", true);
        bool persist = body.value("persist", false);
        nlohmann::json applied = nlohmann::json::array();

        YAML::Node root;
        try { root = YAML::Load(base_yaml); } catch (const std::exception& e) {
            return err_resp(std::string("YAML parse error: ") + e.what());
        }
        if (!root || !root.IsMap()) root = YAML::Node(YAML::NodeType::Map);

        if (body.contains("updates") && body["updates"].is_array()) {
            for (auto& upd : body["updates"]) {
                std::string path = upd.value("path", "");
                std::string value = upd.value("value", "");
                if (!path.empty()) {
                    set_dotted(root, path, value, parse_values);
                    YAML::Node current = root;
                    std::istringstream iss(path);
                    std::string key;
                    while (std::getline(iss, key, '.')) {
                        if (key.empty() || !current[key]) { current = YAML::Node(); break; }
                        current = current[key];
                    }
                    applied.push_back({{"path", path}, {"value", yaml_to_json(current)}});
                }
            }
        }

        std::ostringstream oss;
        oss << root;
        std::string patched_yaml = oss.str();

        nlohmann::json response = {
            {"path", target_path.string()},
            {"config", yaml_to_json(root)},
            {"config_yaml", patched_yaml},
            {"applied", applied}
        };

        if (persist) {
            std::ofstream out(target_path);
            if (!out) return err_resp("Cannot write: " + target_path.string(), 500);
            out << patched_yaml;

            std::string rev_id = state->revision_store.add(patched_yaml, "patch");
            {
                std::lock_guard<std::mutex> lk(state->state_mutex);
                state->active_config_revision_id = rev_id;
            }
            state->ui_event_store.push("config_patched", {{"path", target_path.string()}, {"revision_id", rev_id}});
            response["saved"] = true;
            response["revision_id"] = rev_id;
        }

        return json_resp(response);
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
            items.push_back({
                {"id", entry.path().stem().string()},
                {"name", entry.path().filename().string()},
                {"path", entry.path().string()},
            });
        }
        return json_resp({{"items", items}});
    });

    CROW_ROUTE(app, "/api/config/presets/apply").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("path"))
            return err_resp("BAD_REQUEST", "path is required", 400, nlohmann::json::object());
        fs::path path = fs::path(body["path"].get<std::string>());
        if (path.is_relative()) path = state->runtime.project_root / path;
        if (!state->runtime.is_path_allowed(path)) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + path.string(), 403, {{"path", path.string()}});
        }
        std::string yaml_text = read_file_str(path);
        if (yaml_text.empty()) return err_resp("Preset not found: " + path.string(), 404);
        return json_resp({{"config", yaml_text}, {"applied_paths", nlohmann::json::array({path.string()})}});
    });

    CROW_ROUTE(app, "/api/config/revisions").methods("GET"_method)
    ([state]() {
        auto list = state->revision_store.list();
        nlohmann::json items = nlohmann::json::array();
        for (auto& r : list) items.push_back(config_revision_to_json(r));
        return json_resp({{"items", items}, {"active_revision_id", state->active_config_revision_id}});
    });

    CROW_ROUTE(app, "/api/config/revisions/<string>/restore").methods("POST"_method)
    ([state](const crow::request&, std::string rev_id) {
        auto rev = state->revision_store.get(rev_id);
        if (!rev) return err_resp("Revision not found: " + rev_id, 404);
        if (!state->runtime.is_path_allowed(state->runtime.default_config_path)) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + state->runtime.default_config_path.string(), 403, {{"path", state->runtime.default_config_path.string()}});
        }
        std::ofstream out(state->runtime.default_config_path);
        if (!out) return err_resp("Cannot write: " + state->runtime.default_config_path.string(), 500);
        out << rev->yaml_text;
        {
            std::lock_guard<std::mutex> lk(state->state_mutex);
            state->active_config_revision_id = rev_id;
        }
        return json_resp({{"ok", true}, {"active_revision_id", rev_id}});
    });

    CROW_ROUTE(app, "/api/config/schema").methods("GET"_method)
    ([state]() {
        std::string schema_text = read_file_str(state->runtime.schema_path);
        return json_resp({{"schema", schema_text},
                          {"path",   state->runtime.schema_path.string()}});
    });
}
