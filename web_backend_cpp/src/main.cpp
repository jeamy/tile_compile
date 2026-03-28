#include "app_state.hpp"
#include "routes/system_routes.hpp"
#include "routes/jobs_routes.hpp"
#include "routes/app_state_routes.hpp"
#include "routes/scan_routes.hpp"
#include "routes/config_routes.hpp"
#include "routes/runs_routes.hpp"
#include "routes/ws_routes.hpp"
#include "routes/tools_routes.hpp"

#define CROW_MAIN
#include "crow_app.hpp"

#include <iostream>
#include <fstream>
#include <memory>
#include <filesystem>
#include <system_error>
#include <cerrno>
#include <set>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

namespace {

bool is_queue_staging_job_dir(const fs::path& path) {
    if (path.empty()) return false;
    const std::string name = path.filename().string();
    return name.rfind("job_", 0) == 0;
}

#ifdef __linux__
bool process_cmdline_references_path(const fs::path& target_path,
                                     const std::string& runner_name,
                                     int self_pid) {
    const std::string target = target_path.string();
    std::error_code ec;
    if (!fs::exists("/proc", ec)) return false;

    for (const auto& entry : fs::directory_iterator("/proc", ec)) {
        if (ec || !entry.is_directory()) continue;
        const std::string pid_text = entry.path().filename().string();
        if (pid_text.empty() ||
            !std::all_of(pid_text.begin(), pid_text.end(),
                         [](unsigned char ch) { return std::isdigit(ch) != 0; })) {
            continue;
        }

        int pid = 0;
        try {
            pid = std::stoi(pid_text);
        } catch (...) {
            continue;
        }
        if (pid == self_pid) continue;

        std::ifstream cmdline(entry.path() / "cmdline", std::ios::binary);
        if (!cmdline) continue;
        std::string raw((std::istreambuf_iterator<char>(cmdline)),
                        std::istreambuf_iterator<char>());
        if (raw.empty()) continue;

        std::vector<std::string> argv;
        size_t start = 0;
        while (start < raw.size()) {
            size_t end = raw.find('\0', start);
            if (end == std::string::npos) end = raw.size();
            if (end > start) argv.push_back(raw.substr(start, end - start));
            start = end + 1;
        }
        if (argv.empty()) continue;

        const std::string exe_name = fs::path(argv.front()).filename().string();
        const bool is_runner =
            exe_name == runner_name ||
            exe_name.find("tile_compile_runner") != std::string::npos;
        if (!is_runner) continue;

        if (std::any_of(argv.begin(), argv.end(), [&target](const std::string& arg) {
                return arg.find(target) != std::string::npos;
            })) {
            return true;
        }
    }

    return false;
}
#endif

void cleanup_orphan_queue_staging(const BackendRuntime& runtime) {
    const fs::path staging_root = runtime.runs_dir / ".queue_staging";
    std::error_code ec;
    if (!fs::exists(staging_root, ec) || !fs::is_directory(staging_root, ec)) return;

    const std::string runner_name = fs::path(runtime.runner_exe).filename().string();
    int removed = 0;
    int kept = 0;
    int failed = 0;

    for (const auto& entry : fs::directory_iterator(staging_root, ec)) {
        if (ec) break;
        if (!entry.is_directory()) continue;
        if (!is_queue_staging_job_dir(entry.path())) continue;

        bool is_live = false;
#ifdef __linux__
        is_live = process_cmdline_references_path(entry.path(), runner_name, static_cast<int>(::getpid()));
#endif
        if (is_live) {
            ++kept;
            continue;
        }

        fs::remove_all(entry.path(), ec);
        if (ec) {
            ++failed;
            std::cerr << "[tile_compile_web_backend] Failed to remove stale queue staging dir: "
                      << entry.path() << " (" << ec.message() << ")" << std::endl;
            ec.clear();
            continue;
        }
        ++removed;
    }

    bool root_empty = false;
    if (!ec) {
        root_empty = fs::is_empty(staging_root, ec);
    }
    if (!ec && root_empty) {
        fs::remove(staging_root, ec);
        ec.clear();
    }

    if (removed > 0 || kept > 0 || failed > 0) {
        std::cout << "[tile_compile_web_backend] Queue staging cleanup: removed="
                  << removed << " kept_live=" << kept << " failed=" << failed
                  << std::endl;
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        auto state = std::make_shared<AppState>();
        state->runtime = BackendRuntime::from_env();
        state->job_store.configure_retention(state->runtime.guard_limits.retained_jobs);
        state->subprocess_manager.configure_limits(state->runtime.guard_limits);
        state->ui_event_store.configure(state->runtime.ui_events_path);
        cleanup_orphan_queue_staging(state->runtime);

        CrowApp app;

        auto& cors = app.get_middleware<crow::CORSHandler>();
        cors.global()
            .origin("*")
            .methods("GET"_method, "POST"_method, "PUT"_method,
                     "DELETE"_method, "OPTIONS"_method)
            .headers("Content-Type", "Authorization", "Accept");

        register_system_routes(app, state);
        register_jobs_routes(app, state);
        register_app_state_routes(app, state);
        register_scan_routes(app, state);
        register_config_routes(app, state, nullptr);
        register_runs_routes(app, state);
        register_ws_routes(app, state);
        register_tools_routes(app, state);

        auto read_ui_file = [&state](const fs::path& relative_path) -> std::optional<std::pair<fs::path, std::string>> {
            fs::path f = state->runtime.ui_dir / relative_path;
            std::error_code ec;
            f = f.lexically_normal();
            const fs::path ui_root = state->runtime.ui_dir.lexically_normal();
            const std::string ui_root_text = ui_root.string();
            const std::string file_text = f.string();
            if (file_text.compare(0, ui_root_text.size(), ui_root_text) != 0) return std::nullopt;
            if (!fs::exists(f, ec) || fs::is_directory(f, ec)) return std::nullopt;
            std::ifstream in(f, std::ios::binary);
            if (!in) return std::nullopt;
            std::string body((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
            return std::make_pair(f, std::move(body));
        };

        auto serve_ui_file = [&read_ui_file](const fs::path& relative_path, bool spa_fallback = true) {
            auto loaded = read_ui_file(relative_path);
            if (!loaded && spa_fallback) loaded = read_ui_file("index.html");
            if (!loaded) return crow::response(404);
            const auto& [f, body] = *loaded;
            crow::response res(200, body);
            std::string ext = f.extension().string();
            if      (ext == ".html") res.set_header("Content-Type", "text/html");
            else if (ext == ".js")   res.set_header("Content-Type", "application/javascript");
            else if (ext == ".css")  res.set_header("Content-Type", "text/css");
            else if (ext == ".json") res.set_header("Content-Type", "application/json");
            else if (ext == ".png")  res.set_header("Content-Type", "image/png");
            else if (ext == ".svg")  res.set_header("Content-Type", "image/svg+xml");
            else                     res.set_header("Content-Type", "application/octet-stream");
            return res;
        };

        // Static file serving — frontend SPA
        if (fs::is_directory(state->runtime.ui_dir)) {
            CROW_ROUTE(app, "/ui/<path>")
            ([&serve_ui_file](const crow::request&, std::string path) {
                return serve_ui_file(path);
            });

            CROW_ROUTE(app, "/ui")
            ([](const crow::request&) {
                crow::response res(302);
                res.set_header("Location", "/ui/");
                return res;
            });

            CROW_ROUTE(app, "/<path>")
            ([&serve_ui_file](const crow::request&, std::string path) {
                if (path == "ui" || path == "ui/") {
                    return serve_ui_file("index.html");
                }
                return serve_ui_file(path, false);
            });
        }

        CROW_ROUTE(app, "/api/<path>")
        ([](const crow::request&, std::string) {
            nlohmann::json body = {
                {"error", {
                    {"code", "NOT_FOUND"},
                    {"message", "Not Found"}
                }}
            };
            crow::response res(404, body.dump());
            res.set_header("Content-Type", "application/json");
            return res;
        });

        // Redirect root to /ui
        CROW_ROUTE(app, "/")
        ([]() {
            crow::response res(302);
            res.set_header("Location", "/ui/");
            return res;
        });

        int port = state->runtime.port;
        std::cout << "[tile_compile_web_backend] Starting on http://"
                  << state->runtime.host << ":" << port << "/ui" << std::endl;

        app.bindaddr(state->runtime.host)
           .port(port)
           .multithreaded()
           .run();

        return 0;
    } catch (const std::system_error& e) {
        std::cerr << "[tile_compile_web_backend] Fatal system error: "
                  << e.what() << std::endl;
        if (e.code() == std::errc::address_in_use) {
            std::cerr << "[tile_compile_web_backend] Port already in use. "
                      << "Stop the existing listener or start with another PORT."
                      << std::endl;
        } else if (e.code() == std::errc::permission_denied) {
            std::cerr << "[tile_compile_web_backend] Permission denied while opening "
                      << "the listening socket or runtime files." << std::endl;
        } else if (e.code() == std::errc::address_not_available) {
            std::cerr << "[tile_compile_web_backend] Bind address is not available on "
                      << "this host." << std::endl;
        }
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[tile_compile_web_backend] Fatal error: "
                  << e.what() << std::endl;
        return 1;
    }
}
