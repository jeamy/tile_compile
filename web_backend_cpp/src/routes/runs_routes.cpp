#include "routes/runs_routes.hpp"
#include "services/report_generator.hpp"
#include "services/run_inspector.hpp"
#include <nlohmann/json.hpp>
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

void register_runs_routes(CrowApp& app,
                           std::shared_ptr<AppState> state) {

    CROW_ROUTE(app, "/api/runs").methods("GET"_method)
    ([state]() {
        auto runs = discover_runs(state->runtime.runs_dir, 100);
        return json_resp({{"items", runs}});
    });

    CROW_ROUTE(app, "/api/runs/start").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded()) return err_resp("Invalid JSON");

        std::string input_dir  = body.value("input_dir",  "");
        std::string runs_dir   = body.value("runs_dir",   state->runtime.runs_dir.string());
        std::string run_id     = body.value("run_id",     "");
        std::string color_mode = body.value("color_mode", "");

        if (input_dir.empty() && !body.contains("input_dirs"))
            return err_resp("input_dir or input_dirs required");

        if (!state->runtime.is_path_allowed(fs::path(runs_dir))) {
            state->runtime.grant_root(fs::path(runs_dir));
        }

        std::vector<std::string> args = {state->runtime.runner_exe, "run"};
        if (!input_dir.empty()) { args.push_back("--input"); args.push_back(input_dir); }
        if (body.contains("input_dirs") && body["input_dirs"].is_array())
            for (auto& d : body["input_dirs"]) { args.push_back("--input"); args.push_back(d.get<std::string>()); }
        args.push_back("--runs-dir"); args.push_back(runs_dir);
        if (!run_id.empty()) { args.push_back("--run-id"); args.push_back(run_id); }
        if (!color_mode.empty()) { args.push_back("--color-mode"); args.push_back(color_mode); }
        args.push_back("--config"); args.push_back(state->runtime.default_config_path.string());
        args.push_back("--json");

        std::string effective_run_id = run_id.empty() ? "pending" : run_id;
        std::string job_id = state->subprocess_manager.launch("run", args,
                                                               state->runtime.project_root.string(),
                                                               effective_run_id);
        {
            std::lock_guard<std::mutex> lk(state->state_mutex);
            if (!run_id.empty()) state->current_run_id = run_id;
        }
        state->ui_event_store.push("run_started", {{"job_id", job_id}, {"run_id", effective_run_id}});
        return json_resp({{"job_id", job_id}, {"run_id", effective_run_id}, {"state", "running"}});
    });

    CROW_ROUTE(app, "/api/runs/<string>/status").methods("GET"_method)
    ([state](const crow::request&, std::string run_id) {
        try {
            auto run_dir = state->runtime.resolve_run_dir(run_id);
            auto status  = read_run_status(run_dir);
            status["run_dir"] = run_dir.string();
            status["run_id"]  = run_id;
            return json_resp(status);
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }
    });

    CROW_ROUTE(app, "/api/runs/<string>/stop").methods("POST"_method)
    ([state](const crow::request&, std::string run_id) {
        state->subprocess_manager.cancel_by_run(run_id);
        state->ui_event_store.push("run_stopped", {{"run_id", run_id}});
        return json_resp({{"ok", true}, {"run_id", run_id}, {"cancelled_jobs", 1}, {"killed_pids", 0}});
    });

    CROW_ROUTE(app, "/api/runs/<string>/resume").methods("POST"_method)
    ([state](const crow::request& req, std::string run_id) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded()) return err_resp("Invalid JSON");

        std::string from_phase   = body.value("from_phase",   "");
        std::string run_dir_str  = body.value("run_dir",      "");
        std::string rev_id       = body.value("config_revision_id", "");
        std::string filter_ctx   = body.value("filter_context", "");

        if (from_phase.empty()) return err_resp("from_phase required");

        std::vector<std::string> args = {state->runtime.runner_exe, "resume"};
        args.push_back("--run-id"); args.push_back(run_id);
        args.push_back("--from-phase"); args.push_back(from_phase);
        if (!run_dir_str.empty()) { args.push_back("--run-dir"); args.push_back(run_dir_str); }
        if (!rev_id.empty())      { args.push_back("--config-revision"); args.push_back(rev_id); }
        if (!filter_ctx.empty())  { args.push_back("--filter"); args.push_back(filter_ctx); }
        args.push_back("--config"); args.push_back(state->runtime.default_config_path.string());
        args.push_back("--json");

        std::string job_id = state->subprocess_manager.launch("resume", args,
                                                               state->runtime.project_root.string(),
                                                               run_id);
        {
            std::lock_guard<std::mutex> lk(state->state_mutex);
            state->current_run_id = run_id;
        }
        state->ui_event_store.push("run_resumed", {{"job_id", job_id}, {"run_id", run_id}});
        return json_resp({{"job_id", job_id}, {"run_id", run_id}, {"state", "running"}});
    });

    CROW_ROUTE(app, "/api/runs/<string>/logs").methods("GET"_method)
    ([state](const crow::request& req, std::string run_id) {
        int tail = 250;
        if (req.url_params.get("tail"))
            try { tail = std::stoi(req.url_params.get("tail")); } catch (...) {}
        try {
            auto run_dir = state->runtime.resolve_run_dir(run_id);
            std::string text = read_run_logs(run_dir, tail);
            nlohmann::json lines = nlohmann::json::array();
            std::istringstream iss(text);
            std::string line;
            while (std::getline(iss, line)) lines.push_back(line);
            return json_resp({{"lines", lines}, {"run_id", run_id}});
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }
    });

    CROW_ROUTE(app, "/api/runs/<string>/artifacts").methods("GET"_method)
    ([state](const crow::request&, std::string run_id) {
        try {
            auto run_dir = state->runtime.resolve_run_dir(run_id);
            auto items   = list_run_artifacts(run_dir);
            return json_resp({{"items", items}, {"run_id", run_id}});
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }
    });

    CROW_ROUTE(app, "/api/runs/<string>/artifacts/view").methods("GET"_method)
    ([state](const crow::request& req, std::string run_id) {
        std::string rel_path = req.url_params.get("path") ? req.url_params.get("path") : "";
        try {
            auto run_dir = state->runtime.resolve_run_dir(run_id);
            fs::path full = run_dir / rel_path;
            if (!fs::exists(full)) return err_resp("Artifact not found", 404);
            std::ifstream f(full);
            std::string content((std::istreambuf_iterator<char>(f)),
                                  std::istreambuf_iterator<char>());
            std::string ext = full.extension().string();
            if (ext == ".json" || ext == ".jsonl") {
                try {
                    auto j = nlohmann::json::parse(content);
                    return json_resp({{"content", j}, {"filename", full.filename().string()}, {"path", rel_path}});
                } catch (...) {}
            }
            return json_resp({{"content", content}, {"filename", full.filename().string()}, {"path", rel_path}});
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }
    });

    CROW_ROUTE(app, "/api/runs/<string>/artifacts/raw/<path>").methods("GET"_method)
    ([state](const crow::request&, std::string run_id, std::string rel_path) {
        try {
            auto run_dir = state->runtime.resolve_run_dir(run_id);
            fs::path full = run_dir / rel_path;
            if (!fs::exists(full)) return crow::response(404);
            std::ifstream f(full, std::ios::binary);
            std::string body((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
            crow::response res(200, body);
            std::string ext = full.extension().string();
            if      (ext == ".html") res.set_header("Content-Type", "text/html");
            else if (ext == ".json") res.set_header("Content-Type", "application/json");
            else if (ext == ".png")  res.set_header("Content-Type", "image/png");
            else                     res.set_header("Content-Type", "application/octet-stream");
            return res;
        } catch (...) { return crow::response(404); }
    });

    CROW_ROUTE(app, "/api/runs/<string>/delete").methods("POST"_method)
    ([state](const crow::request&, std::string run_id) {
        try {
            auto run_dir = state->runtime.resolve_run_dir(run_id);
            state->subprocess_manager.cancel_by_run(run_id);
            fs::remove_all(run_dir);
            {
                std::lock_guard<std::mutex> lk(state->state_mutex);
                if (state->current_run_id == run_id) state->current_run_id = "";
            }
            state->ui_event_store.push("run_deleted", {{"run_id", run_id}});
            return json_resp({{"ok", true}, {"run_id", run_id}});
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }
    });

    CROW_ROUTE(app, "/api/runs/<string>/set-current").methods("POST"_method)
    ([state](const crow::request&, std::string run_id) {
        {
            std::lock_guard<std::mutex> lk(state->state_mutex);
            state->current_run_id = run_id;
        }
        state->ui_event_store.push("current_run_changed", {{"run_id", run_id}});
        return json_resp({{"ok", true}, {"run_id", run_id}});
    });

    CROW_ROUTE(app, "/api/runs/<string>/stats").methods("POST"_method)
    ([state](const crow::request& req, std::string run_id) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        std::string run_dir_str;
        if (!body.is_discarded()) run_dir_str = body.value("run_dir", "");

        fs::path run_dir;
        try {
            run_dir = run_dir_str.empty() ? state->runtime.resolve_run_dir(run_id) : fs::path(run_dir_str);
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }

        std::string job_id = state->job_store.create("stats", run_id);
        state->job_store.update_state(job_id, JobState::running, {
            {"run_id", run_id},
            {"run_dir", run_dir.string()}
        });
        std::thread([state, job_id, run_id, run_dir]() {
            auto result = generate_run_report(run_dir);
            nlohmann::json data = result;
            data["run_id"] = run_id;
            data["run_dir"] = run_dir.string();
            if (result.value("ok", false)) {
                state->job_store.update_state(job_id, JobState::ok, data);
            } else {
                state->job_store.update_state(job_id, JobState::error, data,
                    result.value("error", std::string("report generation failed")));
            }
        }).detach();
        return json_resp({{"job_id", job_id}, {"run_id", run_id}, {"state", "running"}});
    });

    CROW_ROUTE(app, "/api/runs/<string>/stats/status").methods("GET"_method)
    ([state](const crow::request&, std::string run_id) {
        try {
            auto run_dir = state->runtime.resolve_run_dir(run_id);
            fs::path stats_dir  = run_dir / "artifacts";
            fs::path report_path = stats_dir / "report.html";
            std::string state_str = "unknown";
            if (fs::exists(report_path)) state_str = "ok";

            auto jobs = state->job_store.list(200);
            for (auto& j : jobs) {
                if (j.type == "stats" && j.run_id == run_id) {
                    state_str = job_state_str(j.state);
                    break;
                }
            }
            return json_resp({
                {"state",       state_str},
                {"output_dir",  stats_dir.string()},
                {"report_path", fs::exists(report_path) ? report_path.string() : ""},
            });
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }
    });

    CROW_ROUTE(app, "/api/runs/<string>/config-revisions/<string>/restore").methods("POST"_method)
    ([state](const crow::request&, std::string run_id, std::string rev_id) {
        auto rev = state->revision_store.get(rev_id);
        if (!rev) return err_resp("Revision not found: " + rev_id, 404);
        {
            std::lock_guard<std::mutex> lk(state->state_mutex);
            std::ofstream out(state->runtime.default_config_path);
            if (out) out << rev->yaml_text;
            state->active_config_revision_id = rev_id;
        }
        return json_resp({{"ok", true}, {"revision_id", rev_id}, {"run_id", run_id}});
    });
}
