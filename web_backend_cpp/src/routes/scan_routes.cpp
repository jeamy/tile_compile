#include "routes/scan_routes.hpp"
#include "services/scan_summary.hpp"
#include <nlohmann/json.hpp>
#include <thread>

namespace fs = std::filesystem;

static crow::response json_resp(const nlohmann::json& j, int status = 200) {
    crow::response res(status, j.dump());
    res.set_header("Content-Type", "application/json");
    return res;
}
static crow::response err_resp(const std::string& msg, int status = 400) {
    return json_resp({{"error", {{"message", msg}}}}, status);
}
static crow::response err_resp(const std::string& code,
                               const std::string& msg,
                               int status,
                               const nlohmann::json& details) {
    return json_resp({{"error", {{"code", code}, {"message", msg}, {"details", details}}}}, status);
}

static std::optional<nlohmann::json> parse_scan_result(const SubprocessResult& res) {
    auto parsed = nlohmann::json::parse(res.stdout_str, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) return std::nullopt;
    return parsed;
}

void register_scan_routes(CrowApp& app,
                           std::shared_ptr<AppState> state) {

    CROW_ROUTE(app, "/api/scan").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded()) return err_resp("Invalid JSON");

        std::string input_dir  = body.value("input_dir", body.value("input_path", ""));
        int frames_min         = body.value("frames_min", 1);
        bool with_checksums    = body.value("with_checksums", false);

        nlohmann::json input_dirs_arr = nlohmann::json::array();
        if (body.contains("input_dirs") && body["input_dirs"].is_array()) {
            for (const auto& item : body["input_dirs"]) {
                if (item.is_string()) {
                    std::string path = item.get<std::string>();
                    if (!path.empty()) input_dirs_arr.push_back(path);
                } else if (item.is_object() && item.contains("input_dir") && item["input_dir"].is_string()) {
                    std::string path = item["input_dir"].get<std::string>();
                    if (!path.empty()) input_dirs_arr.push_back(path);
                } else if (item.is_object() && item.contains("input_path") && item["input_path"].is_string()) {
                    std::string path = item["input_path"].get<std::string>();
                    if (!path.empty()) input_dirs_arr.push_back(path);
                }
            }
        } else if (!input_dir.empty())
            input_dirs_arr.push_back(input_dir);

        if (input_dirs_arr.empty()) return err_resp("No input_dir(s) provided");

        std::vector<std::string> requested_inputs;
        for (const auto& d : input_dirs_arr) requested_inputs.push_back(d.get<std::string>());
        std::vector<std::string> resolved_inputs;
        resolved_inputs.reserve(requested_inputs.size());
        for (const auto& raw : requested_inputs) {
            auto resolved = state->runtime.resolve_input_path(fs::path(raw), !fs::path(raw).is_absolute());
            if (resolved.status == PathStatus::not_allowed) {
                return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + raw, 403, {{"path", raw}});
            }
            if (resolved.status == PathStatus::not_found) {
                return err_resp("PATH_NOT_FOUND", "Path not found: " + raw, 422, {{"path", raw}});
            }
            resolved_inputs.push_back(resolved.path.string());
        }

        if (!resolved_inputs.empty()) input_dir = resolved_inputs.front();
        if (!input_dir.empty()) {
            std::lock_guard<std::mutex> lk(state->state_mutex);
            state->last_scan_input_path = input_dir;
        }

        nlohmann::json initial_data = {
            {"input_path", input_dir},
            {"input_dirs", resolved_inputs},
            {"frames_min", frames_min},
            {"with_checksums", with_checksums},
        };

        std::string job_id;
        if (resolved_inputs.size() == 1) {
            std::vector<std::string> args = {state->runtime.cli_exe, "scan", resolved_inputs.front(), "--frames-min", std::to_string(frames_min), "--json"};
            if (with_checksums) args.push_back("--with-checksums");
            initial_data["command"] = args;
            job_id = state->subprocess_manager.launch("scan", args,
                                                      state->runtime.project_root.string(),
                                                      "",
                                                      initial_data);
        } else {
            job_id = state->job_store.create("scan");
            state->job_store.update_state(job_id, JobState::running, initial_data);
            std::thread([state, job_id, resolved_inputs, frames_min, with_checksums]() {
                try {
                    nlohmann::json per_dir_results = nlohmann::json::array();
                    std::vector<std::string> color_modes_detected;
                    std::vector<std::string> color_candidates;
                    int frames_detected_total = 0;
                    int image_width = 0;
                    int image_height = 0;
                    nlohmann::json bayer_pattern = nullptr;
                    bool requires_confirmation = false;
                    bool ok = true;
                    nlohmann::json all_errors = nlohmann::json::array();
                    nlohmann::json all_warnings = nlohmann::json::array();
                    nlohmann::json all_frames = nlohmann::json::array();

                    for (size_t index = 0; index < resolved_inputs.size(); ++index) {
                        auto snapshot = state->job_store.get(job_id);
                        if (snapshot && snapshot->state == JobState::cancelled) return;

                        std::vector<std::string> args = {
                            state->runtime.cli_exe,
                            "scan",
                            resolved_inputs[index],
                            "--frames-min",
                            std::to_string(frames_min),
                            "--json"
                        };
                        if (with_checksums) args.push_back("--with-checksums");
                        SubprocessResult res = run_subprocess(args, state->runtime.project_root.string());
                        auto parsed_opt = parse_scan_result(res);
                        nlohmann::json parsed = parsed_opt.has_value() ? *parsed_opt : nlohmann::json::object();

                        nlohmann::json item_errors = parsed.contains("errors") && parsed["errors"].is_array()
                            ? parsed["errors"]
                            : nlohmann::json::array();
                        nlohmann::json item_warnings = parsed.contains("warnings") && parsed["warnings"].is_array()
                            ? parsed["warnings"]
                            : nlohmann::json::array();
                        if (res.exit_code != 0 && item_errors.empty()) {
                            item_errors.push_back({
                                {"code", "scan_failed"},
                                {"message", "scan command failed"},
                                {"details", {{"exit_code", res.exit_code}, {"stderr", res.stderr_str}}}
                            });
                        }

                        nlohmann::json item = {
                            {"input_path", resolved_inputs[index]},
                            {"ok", res.exit_code == 0 && item_errors.empty()},
                            {"frames_detected", parsed.value("frames_detected", 0)},
                            {"image_width", parsed.value("image_width", 0)},
                            {"image_height", parsed.value("image_height", 0)},
                            {"color_mode", parsed.value("color_mode", "UNKNOWN")},
                            {"color_mode_candidates", parsed.contains("color_mode_candidates") && parsed["color_mode_candidates"].is_array() ? parsed["color_mode_candidates"] : nlohmann::json::array()},
                            {"bayer_pattern", parsed.contains("bayer_pattern") ? parsed["bayer_pattern"] : nlohmann::json(nullptr)},
                            {"requires_user_confirmation", parsed.value("requires_user_confirmation", false)},
                            {"errors", item_errors},
                            {"warnings", item_warnings},
                            {"frames", parsed.contains("frames") && parsed["frames"].is_array() ? parsed["frames"] : nlohmann::json::array()},
                        };
                        per_dir_results.push_back(item);

                        ok = ok && item.value("ok", false);
                        frames_detected_total += item.value("frames_detected", 0);
                        if (image_width == 0) image_width = item.value("image_width", 0);
                        if (image_height == 0) image_height = item.value("image_height", 0);
                        if (bayer_pattern.is_null() && item.contains("bayer_pattern") && !item["bayer_pattern"].is_null()) bayer_pattern = item["bayer_pattern"];
                        requires_confirmation = requires_confirmation || item.value("requires_user_confirmation", false);
                        for (const auto& err : item_errors) all_errors.push_back(err);
                        for (const auto& warning : item_warnings) all_warnings.push_back(warning);
                        if (item.contains("frames") && item["frames"].is_array()) {
                            for (const auto& frame : item["frames"]) all_frames.push_back(frame);
                        }

                        std::string color_mode = item.value("color_mode", "UNKNOWN");
                        if (!color_mode.empty() && color_mode != "UNKNOWN") {
                            color_modes_detected.push_back(color_mode);
                            if (std::find(color_candidates.begin(), color_candidates.end(), color_mode) == color_candidates.end()) {
                                color_candidates.push_back(color_mode);
                            }
                        }
                        if (item.contains("color_mode_candidates") && item["color_mode_candidates"].is_array()) {
                            for (const auto& candidate_raw : item["color_mode_candidates"]) {
                                std::string candidate = candidate_raw.is_string() ? candidate_raw.get<std::string>() : "";
                                if (!candidate.empty() && std::find(color_candidates.begin(), color_candidates.end(), candidate) == color_candidates.end()) {
                                    color_candidates.push_back(candidate);
                                }
                            }
                        }

                        state->job_store.update_state(job_id, JobState::running, {
                            {"input_path", resolved_inputs[index]},
                            {"input_dirs", resolved_inputs},
                            {"current_index", static_cast<int>(index)},
                            {"progress", static_cast<double>(index + 1) / static_cast<double>(resolved_inputs.size())},
                            {"per_dir_results", per_dir_results}
                        });
                    }

                    std::sort(color_modes_detected.begin(), color_modes_detected.end());
                    color_modes_detected.erase(std::unique(color_modes_detected.begin(), color_modes_detected.end()), color_modes_detected.end());
                    std::string final_color_mode = "UNKNOWN";
                    if (color_modes_detected.size() == 1) final_color_mode = color_modes_detected.front();
                    else if (color_modes_detected.size() > 1) requires_confirmation = true;

                    nlohmann::json summary = {
                        {"ok", ok && all_errors.empty()},
                        {"input_path", resolved_inputs.front()},
                        {"input_dirs", resolved_inputs},
                        {"frames_detected", frames_detected_total},
                        {"image_width", image_width},
                        {"image_height", image_height},
                        {"color_mode", final_color_mode},
                        {"color_mode_candidates", color_candidates},
                        {"bayer_pattern", bayer_pattern},
                        {"requires_user_confirmation", requires_confirmation},
                        {"errors", all_errors},
                        {"warnings", all_warnings},
                        {"frames", all_frames},
                        {"per_dir_results", per_dir_results},
                    };
                    state->job_store.update_state(job_id, summary.value("ok", false) ? JobState::ok : JobState::error, {
                        {"input_path", resolved_inputs.front()},
                        {"input_dirs", resolved_inputs},
                        {"result", summary}
                    });
                } catch (const std::exception& e) {
                    state->job_store.update_state(job_id, JobState::error, {{"error", e.what()}}, e.what());
                }
            }).detach();
        }

        state->ui_event_store.push(
            "scan.start",
            "scan.scan",
            {
                {"input_path", input_dir},
                {"input_dirs", resolved_inputs},
                {"frames_min", frames_min},
                {"with_checksums", with_checksums},
            },
            std::nullopt,
            job_id);
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
