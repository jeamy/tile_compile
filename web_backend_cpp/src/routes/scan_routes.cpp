#include "routes/scan_routes.hpp"
#include "services/scan_summary.hpp"
#include <nlohmann/json.hpp>

static crow::response json_resp(const nlohmann::json& j, int status = 200) {
    crow::response res(status, j.dump());
    res.set_header("Content-Type", "application/json");
    return res;
}
static crow::response err_resp(const std::string& msg, int status = 400) {
    return json_resp({{"error", {{"message", msg}}}}, status);
}

void register_scan_routes(CrowApp& app,
                           std::shared_ptr<AppState> state) {

    CROW_ROUTE(app, "/api/scan").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded()) return err_resp("Invalid JSON");

        std::string input_dir  = body.value("input_dir",  "");
        int frames_min         = body.value("frames_min", 1);
        bool with_checksums    = body.value("with_checksums", false);

        nlohmann::json input_dirs_arr = nlohmann::json::array();
        if (body.contains("input_dirs") && body["input_dirs"].is_array())
            input_dirs_arr = body["input_dirs"];
        else if (!input_dir.empty())
            input_dirs_arr.push_back(input_dir);

        if (input_dirs_arr.empty()) return err_resp("No input_dir(s) provided");

        if (!input_dir.empty()) {
            std::lock_guard<std::mutex> lk(state->state_mutex);
            state->last_scan_input_path = input_dir;
        }

        std::vector<std::string> args = {state->runtime.cli_exe, "scan"};
        for (auto& d : input_dirs_arr) args.push_back(d.get<std::string>());
        args.push_back("--frames-min"); args.push_back(std::to_string(frames_min));
        if (with_checksums) args.push_back("--with-checksums");
        args.push_back("--json");

        std::string job_id = state->subprocess_manager.launch("scan", args,
                                                               state->runtime.project_root.string());
        state->ui_event_store.push("scan_started", {{"job_id", job_id}});
        return json_resp({{"job_id", job_id}, {"state", "running"}});
    });

    CROW_ROUTE(app, "/api/scan/latest").methods("GET"_method)
    ([state]() {
        auto job     = latest_scan_job(state->job_store);
        auto summary = summarize_scan_job(job, state->last_scan_input_path);
        return json_resp(summary);
    });

    CROW_ROUTE(app, "/api/scan/quality").methods("GET"_method)
    ([state]() {
        return json_resp(scan_quality(state->job_store));
    });

    CROW_ROUTE(app, "/api/guardrails").methods("GET"_method)
    ([state]() {
        return json_resp(scan_guardrails(state->job_store));
    });
}
