#include "routes/system_routes.hpp"
#include "routes/route_utils.hpp"
#include "tile_compile/core/build_info.hpp"
#include "tile_compile/core/utils.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace fs = std::filesystem;
using namespace tile_compile::routes;


/// @brief Implements normalized existing path.
/// @details This implementation serves backend health and runtime filesystem helper endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static fs::path normalized_existing_path(const fs::path& path) {
    std::error_code ec;
    fs::path normalized = fs::weakly_canonical(path, ec);
    if (ec) return path.lexically_normal();
    return normalized;
}

static bool directory_entry_dir_first(const fs::directory_entry& a,
                                      const fs::directory_entry& b) {
    const bool a_dir = a.is_directory();
    const bool b_dir = b.is_directory();
    if (a_dir != b_dir) return a_dir > b_dir;
    return a.path().filename().string() < b.path().filename().string();
}

static nlohmann::json installed_binary_version(const fs::path& path) {
    nlohmann::json result = tile_compile::core::binary_provenance_json(path);
    const fs::path sidecar =
        path.parent_path() / (path.stem().string() + ".build-info.json");
    result["build_info_sidecar"] = sidecar.string();
    result["build_info"] = nullptr;
    if (!fs::is_regular_file(sidecar)) return result;
    try {
        result["build_info"] = nlohmann::json::parse(
            tile_compile::core::read_text(sidecar));
    } catch (const std::exception& e) {
        result["build_info_error"] = e.what();
    }
    return result;
}

/// @brief Registers system endpoints for runtime health, path resolution, and backend metadata.
/// @details This is the route-group entry point called from main during Crow setup.
#ifndef _WIN32
/// @brief The desktop "open" program; TILE_COMPILE_OPEN_COMMAND overrides it.
static std::string desktop_opener() {
    const char* custom = std::getenv("TILE_COMPILE_OPEN_COMMAND");
    if (custom && *custom) return custom;
#ifdef __APPLE__
    return "open";
#else
    return "xdg-open";
#endif
}

/// @brief Starts argv without a shell (the path is one argument, never
/// parsed), its stdio on /dev/null, in a session of its own. A detached
/// thread reaps it, so neither a zombie is left nor SIGCHLD is touched
/// (SubprocessManager waits for its own children).
/// @return 0 if the program was started, an errno value otherwise.
static int spawn_detached(const std::vector<std::string>& command) {
    std::vector<char*> argv;
    argv.reserve(command.size() + 1);
    for (const auto& part : command) argv.push_back(const_cast<char*>(part.c_str()));
    argv.push_back(nullptr);

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    for (int fd = 0; fd <= 2; ++fd)
        posix_spawn_file_actions_addopen(&actions, fd, "/dev/null", fd == 0 ? O_RDONLY : O_WRONLY, 0);
    posix_spawnattr_t attr;
    posix_spawnattr_init(&attr);
    // The backend ignores SIGPIPE; the opener gets the default back.
    sigset_t defaults;
    sigemptyset(&defaults);
    sigaddset(&defaults, SIGPIPE);
    posix_spawnattr_setsigdefault(&attr, &defaults);
    short flags = POSIX_SPAWN_SETSIGDEF;
#ifdef POSIX_SPAWN_SETSID
    flags |= POSIX_SPAWN_SETSID;
#endif
    posix_spawnattr_setflags(&attr, flags);
    pid_t pid = -1;
    const int rc = posix_spawnp(&pid, argv[0], &actions, &attr, argv.data(), environ);
    posix_spawnattr_destroy(&attr);
    posix_spawn_file_actions_destroy(&actions);
    if (rc != 0) return rc;
    std::thread([pid] {
        int status = 0;
        while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    }).detach();
    return 0;
}
#endif

void register_system_routes(CrowApp& app,
                              std::shared_ptr<AppState> state) {

    CROW_ROUTE(app, "/api/health")
    ([state](const crow::request& req) {
        return json_resp({{"status", "ok"}, {"service", "tile_compile_web_backend"}});
    });

    CROW_ROUTE(app, "/api/version")
    ([state](const crow::request& req) {
        const fs::path cli_path = fs::path(state->runtime.cli_exe);
        const fs::path runner_path = fs::path(state->runtime.runner_exe);
        return json_resp({
            {"backend", tile_compile::core::build_info_json(true)},
            {"cli", installed_binary_version(cli_path)},
            {"runner", installed_binary_version(runner_path)},
        });
    });

    CROW_ROUTE(app, "/api/fs/roots")
    ([state](const crow::request& req) {
        std::vector<std::string> roots;
        for (const auto& root : state->runtime.allowed_roots()) {
            std::error_code ec;
            fs::path resolved = fs::weakly_canonical(root, ec);
            if (ec || !fs::exists(resolved) || !fs::is_directory(resolved)) continue;
            roots.push_back(resolved.string());
        }
        sort_unique_inplace(roots);

        std::string default_path;
        std::error_code ec;
        fs::path preferred = fs::weakly_canonical(state->runtime.runs_dir, ec);
        if (!ec && std::find(roots.begin(), roots.end(), preferred.string()) != roots.end()) {
            default_path = preferred.string();
        } else if (!roots.empty()) {
            default_path = roots.front();
        }
        return json_resp({{"items", roots}, {"default_path", default_path.empty() ? nlohmann::json(nullptr) : nlohmann::json(default_path)}});
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
            sort_unique_inplace(roots);
            if (roots.empty()) {
                return err_resp("NO_ALLOWED_ROOTS", "no readable allowed roots available for file browser", 400, nlohmann::json::object());
            }
            path = roots.front();
        }
        fs::path dir(path);
        if (!state->runtime.is_path_allowed(dir))
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + path, 403, {{"path", path}});
        if (!fs::exists(dir))
            return err_resp("PATH_NOT_FOUND", "Path not found: " + path, 400, {{"path", path}});
        if (!fs::is_directory(dir))
            return err_resp("NOT_A_DIRECTORY", "path is not a directory", 400, {{"path", path}});

        fs::path resolved_dir = normalized_existing_path(dir);
        std::string normalized_path = resolved_dir.string();
        nlohmann::json parent_path = nullptr;
        bool parent_allowed = false;
        fs::path parent = resolved_dir.parent_path();
        if (!parent.empty() && parent != resolved_dir && fs::exists(parent)) {
            parent_path = normalized_existing_path(parent).string();
            parent_allowed = state->runtime.is_path_allowed(parent);
        }

        nlohmann::json items = nlohmann::json::array();
        std::vector<fs::directory_entry> children;
        for (auto& entry : fs::directory_iterator(resolved_dir)) children.push_back(entry);
        std::sort(children.begin(), children.end(), directory_entry_dir_first);
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
        return json_resp({{"path", normalized_path}, {"parent", parent_path}, {"parent_allowed", parent_allowed}, {"items", items}});
    });

    CROW_ROUTE(app, "/api/fs/grant-root").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body_opt = parse_body(req);
        if (!body_opt || !body_opt->contains("path"))
            return err_resp("BAD_REQUEST", "path is required", 400, nlohmann::json::object());
        auto& body = *body_opt;
        std::string path = body["path"].get<std::string>();
        fs::path candidate = fs::path(path);
        if (candidate.is_relative()) {
            return err_resp("PATH_INVALID", "path must be absolute", 400, {{"path", path}});
        }
        candidate = normalized_existing_path(candidate);
        if (!fs::exists(candidate)) {
            return err_resp("PATH_NOT_FOUND", "path does not exist", 400, {{"path", candidate.string()}});
        }
        if (!fs::is_directory(candidate)) {
            return err_resp("NOT_A_DIRECTORY", "path is not a directory", 400, {{"path", candidate.string()}});
        }
        state->runtime.grant_root(candidate);

        nlohmann::json allowed_roots = nlohmann::json::array();
        for (const auto& root : state->runtime.allowed_roots()) {
            allowed_roots.push_back(normalized_existing_path(root).string());
        }
        return json_resp({{"ok", true}, {"path", candidate.string()}, {"allowed_roots", allowed_roots}});
    });

    CROW_ROUTE(app, "/api/fs/open").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body_opt = parse_body(req);
        if (!body_opt || !body_opt->contains("path"))
            return err_resp("BAD_REQUEST", "path is required", 400, nlohmann::json::object());
        auto& body = *body_opt;
        std::string path = body["path"].get<std::string>();
        fs::path target = fs::path(path);
        if (!state->runtime.is_path_allowed(target))
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed", 403, {{"path", path}});
        if (!fs::exists(target))
            return err_resp("PATH_NOT_FOUND", "path does not exist", 400, {{"path", path}});
        target = normalized_existing_path(target);
#ifdef _WIN32
        std::vector<std::string> command = {"startfile", target.string()};
        int rc = 0;
        ShellExecuteA(nullptr, "open", target.string().c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#else
        std::vector<std::string> command = {desktop_opener(), target.string()};
        const int rc = spawn_detached(command);
#endif
        if (rc != 0) {
            return err_resp("OPEN_FAILED", "failed to open path", 400, {{"path", target.string()}});
        }
        return json_resp({{"ok", true}, {"path", target.string()}, {"command", command}});
    });
 }
