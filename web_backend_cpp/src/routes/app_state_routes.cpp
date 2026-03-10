#include "routes/app_state_routes.hpp"
#include "services/run_inspector.hpp"
#include "services/scan_summary.hpp"
#include <nlohmann/json.hpp>

static crow::response json_resp(const nlohmann::json& j, int status = 200) {
    crow::response res(status, j.dump());
    res.set_header("Content-Type", "application/json");
    return res;
}

void register_app_state_routes(CrowApp& app,
                                std::shared_ptr<AppState> state) {

    CROW_ROUTE(app, "/api/app/state").methods("GET"_method)
    ([state](const crow::request&) {
        std::lock_guard<std::mutex> lk(state->state_mutex);
        auto& rt = state->runtime;

        auto scan_job = latest_scan_job(state->job_store);
        auto scan_summary = summarize_scan_job(scan_job, state->last_scan_input_path);

        nlohmann::json current_run = nlohmann::json::object();
        if (!state->current_run_id.empty()) {
            try {
                auto run_dir    = rt.resolve_run_dir(state->current_run_id);
                auto run_status = read_run_status(run_dir);
                current_run = {
                    {"run_id",        state->current_run_id},
                    {"run_dir",       run_dir.string()},
                    {"status",        run_status.value("status", "unknown")},
                    {"current_phase", run_status.value("current_phase", nullptr)},
                    {"progress",      run_status.value("progress", 0.0)},
                };
            } catch (...) {
                current_run = {{"run_id", state->current_run_id}, {"status", "unknown"}};
            }
        }

        auto recent_runs = discover_runs(rt.runs_dir, 5);

        nlohmann::json resp = nlohmann::json::object();
        resp["project"] = {
            {"project_root",        rt.project_root.string()},
            {"runs_dir",            rt.runs_dir.string()},
            {"default_config_path", rt.default_config_path.string()},
            {"current_run_id",      state->current_run_id},
        };
        resp["scan"] = {
            {"last_input_path", scan_summary.value("input_path", "")},
            {"last_scan",       scan_summary},
        };
        resp["config"] = {
            {"active_revision_id", state->active_config_revision_id},
            {"revision_count",     state->revision_store.count()},
        };
        resp["queue"]   = nlohmann::json::object();
        resp["run"]     = {{"current", current_run}};
        resp["history"] = {
            {"total_runs", (int)recent_runs.size()},
            {"recent",     recent_runs},
        };
        resp["tools"]   = nlohmann::json::object();
        resp["events"]  = {{"latest_seq", state->ui_event_store.latest_seq()}};
        resp["i18n"]    = {{"locale", "de"}};
        return json_resp(resp);
    });

    CROW_ROUTE(app, "/api/app/constants").methods("GET"_method)
    ([](const crow::request&) {
        return json_resp({
            {"phase_order", PHASE_ORDER},
            {"resume_from", RESUME_FROM_PHASES},
            {"color_modes", {"OSC", "MONO", "NARROW"}},
        });
    });

    CROW_ROUTE(app, "/api/app/ui-events").methods("GET"_method)
    ([state](const crow::request& req) {
        int since = 0;
        if (req.url_params.get("since_seq"))
            try { since = std::stoi(req.url_params.get("since_seq")); } catch (...) {}
        auto events = state->ui_event_store.list(since);
        nlohmann::json items = nlohmann::json::array();
        for (auto& e : events) items.push_back(ui_event_to_json(e));
        return json_resp({{"items", items}, {"latest_seq", state->ui_event_store.latest_seq()}});
    });
}
