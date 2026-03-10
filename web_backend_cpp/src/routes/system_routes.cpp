#include "routes/system_routes.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cstdlib>

namespace fs = std::filesystem;

static crow::response json_response(const nlohmann::json& j, int status = 200) {
    crow::response res(status, j.dump());
    res.set_header("Content-Type", "application/json");
    return res;
}

static crow::response error_response(const std::string& msg, int status = 400) {
    return json_response({{"error", {{"message", msg}}}}, status);
}

static crow::response error_response(const std::string& code,
                                     const std::string& msg,
                                     int status,
                                     const nlohmann::json& details) {
    return json_response({{"error", {{"code", code}, {"message", msg}, {"details", details}}}}, status);
}

static fs::path normalized_existing_path(const fs::path& path) {
    std::error_code ec;
    fs::path normalized = fs::weakly_canonical(path, ec);
    if (ec) return path.lexically_normal();
    return normalized;
}

void register_system_routes(CrowApp& app,
                              std::shared_ptr<AppState> state) {

    CROW_ROUTE(app, "/api/health")
    ([state]() {
        return json_response({{"status", "ok"}, {"service", "tile_compile_web_backend"}});
    });

    CROW_ROUTE(app, "/api/version")
    ([state]() {
        fs::path cli_path = fs::path(state->runtime.cli_exe);
        fs::path runner_path = fs::path(state->runtime.runner_exe);
        return json_response({
            {"cli",    (fs::exists(cli_path) ? "found:" : "missing:") + cli_path.string()},
            {"runner", (fs::exists(runner_path) ? "found:" : "missing:") + runner_path.string()},
        });
    });

    CROW_ROUTE(app, "/api/fs/roots")
    ([state]() {
        std::vector<std::string> roots;
        for (const auto& root : state->runtime.allowed_roots()) {
            std::error_code ec;
            fs::path resolved = fs::weakly_canonical(root, ec);
            if (ec || !fs::exists(resolved) || !fs::is_directory(resolved)) continue;
            roots.push_back(resolved.string());
        }
        std::sort(roots.begin(), roots.end());
        roots.erase(std::unique(roots.begin(), roots.end()), roots.end());

        std::string default_path;
        std::error_code ec;
        fs::path preferred = fs::weakly_canonical(state->runtime.runs_dir, ec);
        if (!ec && std::find(roots.begin(), roots.end(), preferred.string()) != roots.end()) {
            default_path = preferred.string();
        } else if (!roots.empty()) {
            default_path = roots.front();
        }
        return json_response({{"items", roots}, {"default_path", default_path.empty() ? nlohmann::json(nullptr) : nlohmann::json(default_path)}});
    });

    CROW_ROUTE(app, "/api/fs/list").methods("GET"_method)
    ([state](const crow::request& req) {
        std::string path = req.url_params.get("path") ? req.url_params.get("path") : "";
        bool include_files = false;
        if (req.url_params.get("include_files")) {
            std::string raw = req.url_params.get("include_files");
            include_files = raw == "1" || raw == "true" || raw == "yes" || raw == "on";
        }
        if (path.empty()) {
            std::vector<std::string> roots;
            for (const auto& root : state->runtime.allowed_roots()) {
                std::error_code ec;
                fs::path resolved = fs::weakly_canonical(root, ec);
                if (ec || !fs::exists(resolved) || !fs::is_directory(resolved)) continue;
                roots.push_back(resolved.string());
            }
            std::sort(roots.begin(), roots.end());
            roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
            if (roots.empty()) {
                return error_response("NO_ALLOWED_ROOTS", "no readable allowed roots available for file browser", 422, nlohmann::json::object());
            }
            path = roots.front();
        }
        fs::path dir(path);
        if (!state->runtime.is_path_allowed(dir))
            return error_response("PATH_NOT_ALLOWED", "Path not allowed: " + path, 403, {{"path", path}});
        if (!fs::exists(dir))
            return error_response("PATH_NOT_FOUND", "Path not found: " + path, 422, {{"path", path}});
        if (!fs::is_directory(dir))
            return error_response("NOT_A_DIRECTORY", "path is not a directory", 422, {{"path", path}});

        fs::path resolved_dir = normalized_existing_path(dir);
        std::string normalized_path = resolved_dir.string();
        nlohmann::json parent_path = nullptr;
        fs::path parent = resolved_dir.parent_path();
        if (!parent.empty() && parent != resolved_dir && state->runtime.is_path_allowed(parent) && fs::exists(parent)) {
            parent_path = normalized_existing_path(parent).string();
        }

        nlohmann::json items = nlohmann::json::array();
        std::vector<fs::directory_entry> children;
        for (auto& entry : fs::directory_iterator(resolved_dir)) children.push_back(entry);
        std::sort(children.begin(), children.end(), [](const fs::directory_entry& a, const fs::directory_entry& b) {
            bool a_dir = a.is_directory();
            bool b_dir = b.is_directory();
            if (a_dir != b_dir) return a_dir > b_dir;
            return a.path().filename().string() < b.path().filename().string();
        });
        for (auto& entry : children) {
            bool is_dir = entry.is_directory();
            if (!is_dir && !include_files) continue;
            if (!state->runtime.is_path_allowed(entry.path())) continue;
            items.push_back({
                {"name",  entry.path().filename().string()},
                {"path",  normalized_existing_path(entry.path()).string()},
                {"type",  is_dir ? "dir" : "file"},
            });
        }
        return json_response({{"path", normalized_path}, {"parent", parent_path}, {"items", items}});
    });

    CROW_ROUTE(app, "/api/fs/grant-root").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("path"))
            return error_response("BAD_REQUEST", "path is required", 400, nlohmann::json::object());
        std::string path = body["path"].get<std::string>();
        fs::path candidate = fs::path(path);
        if (candidate.is_relative()) {
            return error_response("PATH_INVALID", "path must be absolute", 422, {{"path", path}});
        }
        candidate = normalized_existing_path(candidate);
        if (!fs::exists(candidate)) {
            return error_response("PATH_NOT_FOUND", "path does not exist", 422, {{"path", candidate.string()}});
        }
        if (!fs::is_directory(candidate)) {
            return error_response("NOT_A_DIRECTORY", "path is not a directory", 422, {{"path", candidate.string()}});
        }
        state->runtime.grant_root(candidate);

        nlohmann::json allowed_roots = nlohmann::json::array();
        for (const auto& root : state->runtime.allowed_roots()) {
            allowed_roots.push_back(normalized_existing_path(root).string());
        }
        return json_response({{"ok", true}, {"path", candidate.string()}, {"allowed_roots", allowed_roots}});
    });

    CROW_ROUTE(app, "/api/fs/open").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("path"))
            return error_response("BAD_REQUEST", "path is required", 400, nlohmann::json::object());
        std::string path = body["path"].get<std::string>();
        fs::path target = fs::path(path);
        if (!state->runtime.is_path_allowed(target))
            return error_response("PATH_NOT_ALLOWED", "Path not allowed", 403, {{"path", path}});
        if (!fs::exists(target))
            return error_response("PATH_NOT_FOUND", "path does not exist", 422, {{"path", path}});
        target = normalized_existing_path(target);
#ifdef __APPLE__
        std::vector<std::string> command = {"open", target.string()};
        int rc = std::system(("open '" + target.string() + "' >/dev/null 2>&1 &").c_str());
#elif defined(_WIN32)
        std::vector<std::string> command = {"startfile", target.string()};
        int rc = 0;
        ShellExecuteA(nullptr, "open", target.string().c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#else
        std::vector<std::string> command = {"xdg-open", target.string()};
        int rc = std::system(("xdg-open '" + target.string() + "' >/dev/null 2>&1 &").c_str());
#endif
        if (rc != 0) {
            return error_response("OPEN_FAILED", "failed to open path", 422, {{"path", target.string()}});
        }
        return json_response({{"ok", true}, {"path", target.string()}, {"command", command}});
    });
 }
