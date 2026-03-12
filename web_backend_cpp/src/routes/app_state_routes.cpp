#include "routes/app_state_routes.hpp"
#include "services/run_inspector.hpp"
#include "services/scan_summary.hpp"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

static crow::response json_resp(const nlohmann::json& j, int status = 200) {
    crow::response res(status, j.dump());
    res.set_header("Content-Type", "application/json");
    return res;
}

static fs::path ui_state_path(const std::shared_ptr<AppState>& state) {
    return state->runtime.runtime_dir / "ui_state.json";
}

static void load_ui_state_unlocked(const std::shared_ptr<AppState>& state) {
    if (state->ui_state_loaded) return;
    state->ui_state = nlohmann::json::object();
    const fs::path path = ui_state_path(state);
    std::ifstream in(path);
    if (in) {
        nlohmann::json parsed = nlohmann::json::parse(in, nullptr, false);
        if (!parsed.is_discarded() && parsed.is_object()) {
            state->ui_state = std::move(parsed);
        }
    }
    state->ui_state_loaded = true;
}

static bool save_ui_state_unlocked(const std::shared_ptr<AppState>& state) {
    const fs::path path = ui_state_path(state);
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    std::ofstream out(path, std::ios::trunc);
    if (!out) return false;
    out << state->ui_state.dump(2);
    return static_cast<bool>(out);
}

void register_app_state_routes(CrowApp& app,
                                std::shared_ptr<AppState> state) {

    CROW_ROUTE(app, "/api/app/state").methods("GET"_method)
    ([state](const crow::request&) {
        std::lock_guard<std::mutex> lk(state->state_mutex);
        load_ui_state_unlocked(state);
        auto& rt = state->runtime;

        auto scan_job = latest_scan_job(state->job_store);
        auto scan_summary = summarize_scan_job(scan_job, state->last_scan_input_path);

        nlohmann::json current_run = nlohmann::json::object();
        if (!state->current_run_id.empty()) {
            try {
                auto run_dir    = rt.resolve_run_dir(state->current_run_id);
                auto run_status = read_run_status(run_dir);
                apply_job_state_to_run_status(run_status, latest_run_job(state->job_store, state->current_run_id));
                current_run = {
                    {"run_id",        state->current_run_id},
                    {"run_dir",       run_dir.string()},
                    {"status",        run_status.value("status", "unknown")},
                    {"current_phase", run_status.value("current_phase", nullptr)},
                    {"progress",      run_status.value("progress", 0.0)},
                };
            } catch (...) {
                current_run = {{"run_id", state->current_run_id}, {"status", "unknown"}};
                apply_job_state_to_run_status(current_run, latest_run_job(state->job_store, state->current_run_id));
            }
        }

        auto recent_runs = discover_runs(rt.runs_dir, 5);

        nlohmann::json resp = nlohmann::json::object();
        resp["project"] = {
            {"project_root",        rt.project_root.string()},
            {"runs_dir",            rt.runs_dir.string()},
            {"presets_dir",         rt.presets_dir.string()},
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
        resp["ui_state"] = state->ui_state;
        resp["events"]  = {{"latest_seq", state->ui_event_store.latest_seq()}};
        resp["i18n"]    = {{"locale", state->ui_state.value("gui2.locale", std::string("de"))}};
        return json_resp(resp);
    });

    CROW_ROUTE(app, "/api/app/ui-state").methods("GET"_method)
    ([state](const crow::request&) {
        std::lock_guard<std::mutex> lk(state->state_mutex);
        load_ui_state_unlocked(state);
        return json_resp({{"state", state->ui_state}});
    });

    CROW_ROUTE(app, "/api/app/ui-state").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded()) {
            return json_resp({{"error", {{"code", "BAD_REQUEST"}, {"message", "invalid JSON"}}}}, 400);
        }
        const nlohmann::json next_state = body.contains("state") ? body["state"] : body;
        if (!next_state.is_object()) {
            return json_resp({{"error", {{"code", "BAD_REQUEST"}, {"message", "state must be an object"}}}}, 400);
        }
        {
            std::lock_guard<std::mutex> lk(state->state_mutex);
            load_ui_state_unlocked(state);
            state->ui_state = next_state;
            if (!save_ui_state_unlocked(state)) {
                return json_resp({{"error", {{"code", "INTERNAL_ERROR"}, {"message", "failed to save ui state"}}}}, 500);
            }
        }
        state->ui_event_store.push("app.ui_state.save", "app.ui_state", {{"keys", static_cast<int>(next_state.size())}});
        return json_resp({{"ok", true}, {"state", next_state}});
    });

    CROW_ROUTE(app, "/api/app/constants").methods("GET"_method)
    ([](const crow::request&) {
        return json_resp({
            {"phases", PHASE_ORDER},
            {"resume_from", RESUME_FROM_PHASES},
            {"color_modes", {"OSC", "MONO", "RGB"}},
        });
    });

    CROW_ROUTE(app, "/api/app/ui-events").methods("GET"_method)
    ([state](const crow::request& req) {
        int since = 0;
        int limit = 200;
        if (req.url_params.get("after_seq")) {
            try { since = std::stoi(req.url_params.get("after_seq")); } catch (...) {}
        } else if (req.url_params.get("since_seq")) {
            try { since = std::stoi(req.url_params.get("since_seq")); } catch (...) {}
        }
        if (req.url_params.get("limit"))
            try { limit = std::stoi(req.url_params.get("limit")); } catch (...) {}
        auto events = state->ui_event_store.list(std::max(0, since), std::max(1, limit));
        nlohmann::json items = nlohmann::json::array();
        for (auto& e : events) items.push_back(ui_event_to_json(e));
        return json_resp({{"items", items}, {"latest_seq", state->ui_event_store.latest_seq()}});
    });
}
