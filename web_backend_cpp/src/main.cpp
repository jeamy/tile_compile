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

    // Static file serving — frontend SPA
    if (fs::is_directory(state->runtime.ui_dir)) {
        CROW_ROUTE(app, "/ui/<path>")
        ([&state](const crow::request&, std::string path) {
            fs::path f = state->runtime.ui_dir / path;
            if (!fs::exists(f) || fs::is_directory(f))
                f = state->runtime.ui_dir / "index.html";
            std::ifstream in(f, std::ios::binary);
            if (!in) return crow::response(404);
            std::string body((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
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
        });

        CROW_ROUTE(app, "/ui")
        ([&state](const crow::request&) {
            fs::path f = state->runtime.ui_dir / "index.html";
            std::ifstream in(f, std::ios::binary);
            if (!in) return crow::response(404);
            std::string body((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
            crow::response res(200, body);
            res.set_header("Content-Type", "text/html");
            return res;
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

    // Redirect root to /ui/
    CROW_ROUTE(app, "/")
    ([]() {
        crow::response res(302);
        res.set_header("Location", "/ui/");
        return res;
    });

    int port = state->runtime.port;
    std::cout << "[tile_compile_web_backend] Starting on http://"
              << state->runtime.host << ":" << port << "/ui/" << std::endl;

    app.bindaddr(state->runtime.host)
       .port(port)
       .multithreaded()
       .run();

    return 0;
}
