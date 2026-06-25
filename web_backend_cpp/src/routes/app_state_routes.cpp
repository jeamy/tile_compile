#include "routes/app_state_routes.hpp"
#include "routes/route_utils.hpp"
#include "services/run_inspector.hpp"
#include "services/scan_summary.hpp"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using namespace tile_compile::routes;

/// @brief Implements ui state path.
/// @details This implementation serves persisted UI state and active-run metadata endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static fs::path ui_state_path(const std::shared_ptr<AppState>& state) {
    return state->runtime.runtime_dir / "ui_state.json";
}

/// @brief Loads ui state unlocked.
/// @details This implementation serves persisted UI state and active-run metadata endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
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

/// @brief Saves ui state unlocked.
/// @details This implementation serves persisted UI state and active-run metadata endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static bool save_ui_state_unlocked(const std::shared_ptr<AppState>& state) {
    const fs::path path = ui_state_path(state);
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    std::ofstream out(path, std::ios::trunc);
    if (!out) return false;
    out << state->ui_state.dump(2);
    return static_cast<bool>(out);
}

/// @brief Implements detect temp root.
/// @details This implementation serves persisted UI state and active-run metadata endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static fs::path detect_temp_root(const std::shared_ptr<AppState>& state) {
    std::error_code ec;
    fs::path temp_root = fs::temp_directory_path(ec);
    if (!ec && !temp_root.empty()) return temp_root;
    return state->runtime.runtime_dir / "tmp";
}

/// @brief Registers UI-state endpoints for persistent frontend state and current-run metadata.
/// @details This is the route-group entry point called from main during Crow setup.
void register_app_state_routes(CrowApp& app,
                                std::shared_ptr<AppState> state) {

    CROW_ROUTE(app, "/api/app/state").methods("GET"_method)
    ([state](const crow::request&) {
        std::string current_run_id;
        std::string current_run_dir_hint;
        std::string active_config_revision_id;
        std::string last_scan_input_path;
        nlohmann::json ui_state = nlohmann::json::object();
        int revision_count = 0;
        auto& rt = state->runtime;
        {
            std::lock_guard<std::mutex> lk(state->state_mutex);
            load_ui_state_unlocked(state);
            current_run_id       = state->current_run_id;
            current_run_dir_hint = state->current_run_dir;
            active_config_revision_id = state->active_config_revision_id;
            last_scan_input_path = state->last_scan_input_path;
            ui_state = state->ui_state;
            revision_count = state->revision_store.count();
        }

        auto scan_job = latest_scan_job(state->job_store);
        auto scan_summary = summarize_scan_job(scan_job, last_scan_input_path);

        nlohmann::json current_run = nlohmann::json::object();
        if (!current_run_id.empty()) {
            const auto job = latest_run_job(state->job_store, current_run_id);
            std::string alt_runs_dir = current_run_dir_hint;
            if (alt_runs_dir.empty() && job && job->data.is_object()) {
                alt_runs_dir = job->data.value("runs_dir", "");
            }
            try {
                auto run_dir    = rt.resolve_run_dir(current_run_id, alt_runs_dir);
                auto run_status = read_run_status(run_dir);
                apply_job_state_to_run_status(run_status, job);
                apply_runtime_liveness_to_run_status(run_status, job, rt.runner_exe, current_run_id, run_dir.string());
                current_run = {
                    {"run_id",        current_run_id},
                    {"run_dir",       run_dir.string()},
                    {"status",        run_status.value("status", "unknown")},
                    {"current_phase", run_status.value("current_phase", nullptr)},
                    {"progress",      run_status.value("progress", 0.0)},
                };
            } catch (...) {
                current_run = {{"run_id", current_run_id}, {"status", "unknown"}};
                apply_job_state_to_run_status(current_run, job);
                apply_runtime_liveness_to_run_status(current_run, job, rt.runner_exe, current_run_id, std::string());
            }
        }

        auto recent_runs = discover_runs(rt.runs_dir, 5);

        nlohmann::json resp = nlohmann::json::object();
        resp["project"] = {
            {"project_root",        rt.project_root.string()},
            {"runs_dir",            rt.runs_dir.string()},
            {"presets_dir",         rt.presets_dir.string()},
            {"default_config_path", rt.default_config_path.string()},
            {"current_run_id",      current_run_id},
        };
        resp["scan"] = {
            {"last_input_path", scan_summary.value("input_path", "")},
            {"last_scan",       scan_summary},
        };
        resp["config"] = {
            {"active_revision_id", active_config_revision_id},
            {"revision_count",     revision_count},
        };
        resp["queue"]   = nlohmann::json::object();
        resp["run"]     = {{"current", current_run}};
        resp["history"] = {
            {"total_runs", (int)recent_runs.size()},
            {"recent",     recent_runs},
        };
        resp["tools"]   = nlohmann::json::object();
        resp["ui_state"] = ui_state;
        resp["events"]  = {{"latest_seq", state->ui_event_store.latest_seq()}};
        resp["i18n"]    = {{"locale", ui_state.value("gui2.locale", std::string("de"))}};
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
        auto body_opt = parse_body(req);
        if (!body_opt) {
            return json_resp({{"error", {{"code", "BAD_REQUEST"}, {"message", "invalid JSON"}}}}, 400);
        }
        auto& body = *body_opt;
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
    ([state](const crow::request&) {
        const fs::path temp_root = detect_temp_root(state);
        return json_resp({
            {"phases", PHASE_ORDER},
            {"resume_from", RESUME_FROM_PHASES},
            {"color_modes", {"OSC", "MONO", "RGB"}},
            {"temp_root", temp_root.string()},
        });
    });

    CROW_ROUTE(app, "/api/app/ui-events").methods("GET"_method)
    ([state](const crow::request& req) {
        int since = 0;
        int limit = 200;
        if (req.url_params.get("after_seq")) {
            since = parse_int_param(req, "after_seq", since);
        } else if (req.url_params.get("since_seq")) {
            since = parse_int_param(req, "since_seq", since);
        }
        limit = parse_int_param(req, "limit", limit);
        auto events = state->ui_event_store.list(std::max(0, since), std::max(1, limit));
        nlohmann::json items = nlohmann::json::array();
        for (auto& e : events) items.push_back(ui_event_to_json(e));
        return json_resp({{"items", items}, {"latest_seq", state->ui_event_store.latest_seq()}});
    });
}
