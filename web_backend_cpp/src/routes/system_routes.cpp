#include "routes/system_routes.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

static crow::response json_response(const nlohmann::json& j, int status = 200) {
    crow::response res(status, j.dump());
    res.set_header("Content-Type", "application/json");
    return res;
}

static crow::response error_response(const std::string& msg, int status = 400) {
    return json_response({{"error", {{"message", msg}}}}, status);
}

void register_system_routes(CrowApp& app,
                              std::shared_ptr<AppState> state) {

    CROW_ROUTE(app, "/api/health")
    ([state]() {
        return json_response({{"status", "ok"}, {"service", "tile_compile_web_backend"}});
    });

    CROW_ROUTE(app, "/api/version")
    ([state]() {
        return json_response({
            {"version",    "1.0.0"},
            {"backend",    "c++"},
            {"project_root", state->runtime.project_root.string()},
        });
    });

    CROW_ROUTE(app, "/api/fs/roots")
    ([state]() {
        return json_response({
            {"items", nlohmann::json::array({state->runtime.project_root.string()})}
        });
    });

    CROW_ROUTE(app, "/api/fs/list").methods("GET"_method)
    ([state](const crow::request& req) {
        std::string path = req.url_params.get("path") ? req.url_params.get("path") : "";
        if (path.empty()) path = state->runtime.project_root.string();
        fs::path dir(path);
        if (!state->runtime.is_path_allowed(dir))
            return error_response("Path not allowed: " + path, 403);
        if (!fs::is_directory(dir))
            return error_response("Not a directory: " + path, 404);
        nlohmann::json items = nlohmann::json::array();
        for (auto& entry : fs::directory_iterator(dir)) {
            items.push_back({
                {"name",  entry.path().filename().string()},
                {"path",  entry.path().string()},
                {"is_dir", entry.is_directory()},
                {"size",  entry.is_regular_file() ? (int64_t)fs::file_size(entry.path()) : 0},
            });
        }
        return json_response({{"path", path}, {"items", items}});
    });

    CROW_ROUTE(app, "/api/fs/grant-root").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("path"))
            return error_response("Missing 'path'");
        std::string path = body["path"].get<std::string>();
        state->runtime.grant_root(fs::path(path));
        return json_response({{"ok", true}, {"path", path}});
    });

    CROW_ROUTE(app, "/api/fs/open").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("path"))
            return error_response("Missing 'path'");
        std::string path = body["path"].get<std::string>();
        if (!state->runtime.is_path_allowed(fs::path(path)))
            return error_response("Path not allowed", 403);
#ifdef __APPLE__
        std::system(("open " + path + " &").c_str());
#elif defined(_WIN32)
        ShellExecuteA(nullptr, "open", path.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#else
        std::system(("xdg-open " + path + " &").c_str());
#endif
        return json_response({{"ok", true}});
    });
}
