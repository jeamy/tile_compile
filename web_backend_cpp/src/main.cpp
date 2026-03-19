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
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    auto state = std::make_shared<AppState>();
    state->runtime = BackendRuntime::from_env();
    state->job_store.configure_retention(state->runtime.guard_limits.retained_jobs);
    state->subprocess_manager.configure_limits(state->runtime.guard_limits);
    state->ui_event_store.configure(state->runtime.ui_events_path);

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
}
