#include "routes/jobs_routes.hpp"
#include <nlohmann/json.hpp>

static crow::response json_resp(const nlohmann::json& j, int status = 200) {
    crow::response res(status, j.dump());
    res.set_header("Content-Type", "application/json");
    return res;
}
static crow::response err_resp(const std::string& msg, int status = 400) {
    std::string code = "BAD_REQUEST";
    if (status == 404) code = "NOT_FOUND";
    else if (status == 403) code = "FORBIDDEN";
    else if (status == 422) code = "UNPROCESSABLE_ENTITY";
    else if (status >= 500) code = "INTERNAL_ERROR";
    return json_resp({{"error", {{"code", code}, {"message", msg}, {"details", nlohmann::json::object()}}}}, status);
}
static crow::response err_resp(const std::string& code,
                               const std::string& msg,
                               int status,
                               const nlohmann::json& details = nlohmann::json::object()) {
    return json_resp({{"error", {{"code", code}, {"message", msg}, {"details", details}}}}, status);
}

void register_jobs_routes(CrowApp& app,
                          std::shared_ptr<AppState> state) {

    CROW_ROUTE(app, "/api/jobs").methods("GET"_method)
    ([state](const crow::request& req) {
        int limit = 100;
        if (req.url_params.get("limit"))
            try { limit = std::stoi(req.url_params.get("limit")); } catch (...) {}
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
