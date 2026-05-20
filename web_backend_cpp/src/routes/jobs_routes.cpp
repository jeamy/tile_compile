#include "routes/jobs_routes.hpp"
#include "routes/route_utils.hpp"
#include <nlohmann/json.hpp>

using namespace tile_compile::routes;

/// @brief Registers job endpoints for polling recent jobs and requesting cancellation.
/// @details This is the route-group entry point called from main during Crow setup.
void register_jobs_routes(CrowApp& app,
                          std::shared_ptr<AppState> state) {

    CROW_ROUTE(app, "/api/jobs").methods("GET"_method)
    ([state](const crow::request& req) {
        int limit = parse_int_param(req, "limit", 100);
        auto jobs = state->job_store.list(limit);
        nlohmann::json items = nlohmann::json::array();
        for (auto& j : jobs) items.push_back(job_to_json(j));
        return json_resp({{"items", items}});
    });

    CROW_ROUTE(app, "/api/jobs/<string>").methods("GET"_method)
    ([state](const crow::request&, std::string job_id) {
        auto job = state->job_store.get(job_id);
        if (!job) return err_resp("NOT_FOUND", "job '" + job_id + "' not found", 404);
        return json_resp(job_to_json(*job));
    });

    CROW_ROUTE(app, "/api/jobs/<string>/cancel").methods("POST"_method)
    ([state](const crow::request&, std::string job_id) {
        auto job = state->job_store.get(job_id);
        if (!job) return err_resp("NOT_FOUND", "job '" + job_id + "' not found", 404);

        const bool subprocess_cancelled = state->subprocess_manager.cancel(job_id);
        state->job_store.cancel(job_id);
        state->ui_event_store.push(
            "job.cancel",
            "jobs.cancel",
            {
                {"ok", true},
                {"job_type", job->type},
                {"subprocess_cancelled", subprocess_cancelled},
            },
            job->run_id.empty() ? std::nullopt : std::optional<std::string>(job->run_id),
            job_id);
        return json_resp({{"ok", true}});
    });
}
