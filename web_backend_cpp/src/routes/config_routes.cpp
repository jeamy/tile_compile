#include "routes/config_routes.hpp"
#include "subprocess_manager.hpp"
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

static crow::response json_resp(const nlohmann::json& j, int status = 200) {
    crow::response res(status, j.dump());
    res.set_header("Content-Type", "application/json");
    return res;
}
static crow::response err_resp(const std::string& msg, int status = 400) {
    return json_resp({{"error", {{"message", msg}}}}, status);
}

static std::string read_file_str(const fs::path& p) {
    std::ifstream f(p);
    if (!f) return "";
    return std::string((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
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
    ([state](const crow::request&) {
        std::string yaml_text = read_file_str(state->runtime.default_config_path);
        return json_resp({{"config", yaml_text},
                          {"path",   state->runtime.default_config_path.string()}});
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
        if (body.is_discarded() || !body.contains("yaml"))
            return err_resp("Missing 'yaml'");
        std::string yaml_text = body["yaml"].get<std::string>();
        std::string target_path = body.value("path", state->runtime.default_config_path.string());

        std::ofstream out(target_path);
        if (!out) return err_resp("Cannot write: " + target_path, 500);
        out << yaml_text;

        std::string rev_id = state->revision_store.add(yaml_text, "save");
        {
            std::lock_guard<std::mutex> lk(state->state_mutex);
            state->active_config_revision_id = rev_id;
        }
        state->ui_event_store.push("config_saved", {{"path", target_path}, {"revision_id", rev_id}});
        return json_resp({{"ok", true}, {"path", target_path}, {"revision_id", rev_id}});
    });

    CROW_ROUTE(app, "/api/config/patch").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded()) return err_resp("Invalid JSON");

        std::string base_yaml;
        if (body.contains("yaml") && body["yaml"].is_string())
            base_yaml = body["yaml"].get<std::string>();
        else
            base_yaml = read_file_str(state->runtime.default_config_path);

        bool parse_values = body.value("parse_values", true);

        YAML::Node root;
        try { root = YAML::Load(base_yaml); } catch (const std::exception& e) {
            return err_resp(std::string("YAML parse error: ") + e.what());
        }

        if (body.contains("updates") && body["updates"].is_array()) {
            for (auto& upd : body["updates"]) {
                std::string path = upd.value("path", "");
                std::string value = upd.value("value", "");
                if (!path.empty())
                    set_dotted(root, path, value, parse_values);
            }
        }

        std::ostringstream oss;
        oss << root;
        std::string patched_yaml = oss.str();
        return json_resp({{"config_yaml", patched_yaml}, {"ok", true}});
    });

    CROW_ROUTE(app, "/api/config/presets").methods("GET"_method)
    ([state]() {
        nlohmann::json items = nlohmann::json::array();
        if (!fs::is_directory(state->runtime.presets_dir)) return json_resp({{"items", items}});
        for (auto& entry : fs::directory_iterator(state->runtime.presets_dir)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            if (ext != ".yaml" && ext != ".yml") continue;
            items.push_back({
                {"name", entry.path().stem().string()},
                {"path", entry.path().string()},
            });
        }
        return json_resp({{"items", items}});
    });

    CROW_ROUTE(app, "/api/config/presets/apply").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("path"))
            return err_resp("Missing 'path'");
        std::string path = body["path"].get<std::string>();
        std::string yaml_text = read_file_str(fs::path(path));
        if (yaml_text.empty()) return err_resp("Preset not found: " + path, 404);
        return json_resp({{"config", yaml_text}, {"path", path}});
    });

    CROW_ROUTE(app, "/api/config/revisions").methods("GET"_method)
    ([state]() {
        auto list = state->revision_store.list();
        nlohmann::json items = nlohmann::json::array();
        for (auto& r : list) items.push_back(config_revision_to_json(r));
        return json_resp({{"items", items}});
    });

    CROW_ROUTE(app, "/api/config/revisions/<string>/restore").methods("POST"_method)
    ([state](const crow::request&, std::string rev_id) {
        auto rev = state->revision_store.get(rev_id);
        if (!rev) return err_resp("Revision not found: " + rev_id, 404);
        return json_resp({{"config", rev->yaml_text}, {"revision_id", rev_id}});
    });

    CROW_ROUTE(app, "/api/config/schema").methods("GET"_method)
    ([state]() {
        std::string schema_text = read_file_str(state->runtime.schema_path);
        return json_resp({{"schema", schema_text},
                          {"path",   state->runtime.schema_path.string()}});
    });
}
