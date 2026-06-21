#include "routes/scan_routes.hpp"
#include "routes/route_utils.hpp"
#include "services/scan_summary.hpp"
#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <thread>
#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;
using namespace tile_compile::routes;

namespace {

using json = nlohmann::json;

/// @brief Compacts array.
/// @details This implementation serves input scan requests and scan result normalization; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
json compact_array(const json& value, size_t max_items) {
    json out = json::array();
    if (!value.is_array()) return out;
    const size_t limit = std::min(value.size(), max_items);
    for (size_t i = 0; i < limit; ++i) out.push_back(value[i]);
    return out;
}

/// @brief Implements append limited unique.
/// @details This implementation serves input scan requests and scan result normalization; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
void append_limited_unique(std::vector<std::string>& items, const std::string& value, size_t limit) {
    if (value.empty()) return;
    if (std::find(items.begin(), items.end(), value) != items.end()) return;
    if (items.size() >= limit) return;
    items.push_back(value);
}

void append_limited_array(json& target,
                          const json& source,
                          size_t limit,
                          size_t& total_count,
                          bool& truncated) {
    if (!source.is_array()) return;
    total_count += source.size();
    for (const auto& item : source) {
        if (target.size() < limit) target.push_back(item);
        else truncated = true;
    }
}

json make_scan_item(const std::string& input_path,
                    const json& parsed,
                    const json& item_errors,
                    const json& item_warnings,
                    bool ok,
                    const BackendGuardLimits& limits) {
    const json frames = parsed.contains("frames") && parsed["frames"].is_array()
        ? parsed["frames"]
        : json::array();
    const json candidates = parsed.contains("color_mode_candidates") && parsed["color_mode_candidates"].is_array()
        ? parsed["color_mode_candidates"]
        : json::array();

    return {
        {"input_path", input_path},
        {"ok", ok},
        {"frames_detected", parsed.value("frames_detected", 0)},
        {"image_width", parsed.value("image_width", 0)},
        {"image_height", parsed.value("image_height", 0)},
        {"color_mode", parsed.value("color_mode", "UNKNOWN")},
        {"color_mode_candidates", compact_array(candidates, limits.scan_color_candidates_preview)},
        {"color_mode_candidates_total", candidates.size()},
        {"color_mode_candidates_truncated", candidates.size() > limits.scan_color_candidates_preview},
        {"bayer_pattern", parsed.contains("bayer_pattern") ? parsed["bayer_pattern"] : json(nullptr)},
        {"requires_user_confirmation", parsed.value("requires_user_confirmation", false)},
        {"errors", compact_array(item_errors, limits.scan_messages_preview)},
        {"errors_total", item_errors.is_array() ? item_errors.size() : 0},
        {"errors_truncated", item_errors.is_array() && item_errors.size() > limits.scan_messages_preview},
        {"warnings", compact_array(item_warnings, limits.scan_messages_preview)},
        {"warnings_total", item_warnings.is_array() ? item_warnings.size() : 0},
        {"warnings_truncated", item_warnings.is_array() && item_warnings.size() > limits.scan_messages_preview},
        {"frames", compact_array(frames, limits.scan_per_dir_frames_preview)},
        {"frames_total", frames.size()},
        {"frames_truncated", frames.size() > limits.scan_per_dir_frames_preview},
    };
}

}  // namespace


/// @brief Parses scan result.
/// @details This implementation serves input scan requests and scan result normalization; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static std::optional<nlohmann::json> parse_scan_result(const SubprocessResult& res) {
    auto parsed = nlohmann::json::parse(res.stdout_str, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) return std::nullopt;
    return parsed;
}

/// @brief Registers scan endpoints that launch input discovery and normalize scanner results.
/// @details This is the route-group entry point called from main during Crow setup.
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
                return err_resp("PATH_NOT_FOUND", "Path not found: " + raw, 400, {{"path", raw}});
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
            std::thread([state, job_id, resolved_inputs, frames_min, with_checksums, limits = state->runtime.guard_limits]() {
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
                    size_t errors_total = 0;
                    size_t warnings_total = 0;
                    size_t frames_total = 0;
                    bool errors_truncated = false;
                    bool warnings_truncated = false;
                    bool frames_truncated = false;
                    bool per_dir_results_truncated = false;

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
                        SubprocessResult res = run_subprocess(args, state->runtime.project_root.string(), "", &limits);
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

                        nlohmann::json item = make_scan_item(
                            resolved_inputs[index],
                            parsed,
                            item_errors,
                            item_warnings,
                            res.exit_code == 0 && item_errors.empty(),
                            limits);
                        if (per_dir_results.size() < limits.scan_per_dir_results_preview) per_dir_results.push_back(item);
                        else per_dir_results_truncated = true;

                        ok = ok && item.value("ok", false);
                        frames_detected_total += item.value("frames_detected", 0);
                        if (image_width == 0) image_width = item.value("image_width", 0);
                        if (image_height == 0) image_height = item.value("image_height", 0);
                        if (bayer_pattern.is_null() && item.contains("bayer_pattern") && !item["bayer_pattern"].is_null()) bayer_pattern = item["bayer_pattern"];
                        requires_confirmation = requires_confirmation || item.value("requires_user_confirmation", false);
                        append_limited_array(all_errors, item_errors, limits.scan_messages_preview, errors_total, errors_truncated);
                        append_limited_array(all_warnings, item_warnings, limits.scan_messages_preview, warnings_total, warnings_truncated);
                        if (parsed.contains("frames") && parsed["frames"].is_array()) {
                            append_limited_array(all_frames, parsed["frames"], limits.scan_frames_preview, frames_total, frames_truncated);
                        }

                        std::string color_mode = item.value("color_mode", "UNKNOWN");
                        if (!color_mode.empty() && color_mode != "UNKNOWN") {
                            color_modes_detected.push_back(color_mode);
                            append_limited_unique(color_candidates, color_mode, limits.scan_color_candidates_preview);
                        }
                        if (item.contains("color_mode_candidates") && item["color_mode_candidates"].is_array()) {
                            for (const auto& candidate_raw : item["color_mode_candidates"]) {
                                std::string candidate = candidate_raw.is_string() ? candidate_raw.get<std::string>() : "";
                                append_limited_unique(color_candidates, candidate, limits.scan_color_candidates_preview);
                            }
                        }

                        state->job_store.update_state(job_id, JobState::running, {
                            {"input_path", resolved_inputs[index]},
                            {"input_dirs", resolved_inputs},
                            {"current_index", static_cast<int>(index)},
                            {"progress", static_cast<double>(index + 1) / static_cast<double>(resolved_inputs.size())},
                            {"frames_detected", frames_detected_total},
                            {"per_dir_results", per_dir_results},
                            {"per_dir_results_total", static_cast<int>(index + 1)},
                            {"per_dir_results_truncated", per_dir_results_truncated}
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
                        {"errors_total", errors_total},
                        {"errors_truncated", errors_truncated},
                        {"warnings", all_warnings},
                        {"warnings_total", warnings_total},
                        {"warnings_truncated", warnings_truncated},
                        {"frames", all_frames},
                        {"frames_total", frames_total},
                        {"frames_truncated", frames_truncated},
                        {"per_dir_results", per_dir_results},
                        {"per_dir_results_total", resolved_inputs.size()},
                        {"per_dir_results_truncated", per_dir_results_truncated},
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

    CROW_ROUTE(app, "/api/scan/metrics").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = json::parse(req.body, nullptr, false);
        std::string input_path;
        if (body.is_object()) {
            if (body.contains("input_path") && body["input_path"].is_string())
                input_path = body["input_path"].get<std::string>();
        }
        if (input_path.empty()) {
            // Try to infer from last scan
            input_path = state->last_scan_input_path;
        }
        if (input_path.empty()) {
            return json_resp({{"error", "no input_path provided and no previous scan"}}, 400);
        }
        std::vector<std::string> args = {state->runtime.cli_exe, "scan-metrics", input_path};
        json initial_data = {{"input_path", input_path}, {"command", args}};
        std::string job_id = state->subprocess_manager.launch(
            "scan-metrics", args, state->runtime.project_root.string(), "", initial_data);
        return json_resp({{"job_id", job_id}, {"state", "running"}});
    });

    CROW_ROUTE(app, "/api/scan/metrics/latest").methods("GET"_method)
    ([state]() {
        auto jobs = state->job_store.list();
        json best_result = nullptr;
        std::string best_ts;
        for (const auto& j : jobs) {
            if (j.type != "scan-metrics") continue;
            if (j.state != JobState::ok) continue;
            if (j.data.contains("result") && j.data["result"].is_object()) {
                const auto& r = j.data["result"];
                if (r.value("ok", false)) {
                    if (j.job_id > best_ts) {
                        best_ts = j.job_id;
                        best_result = r;
                    }
                }
            }
        }
        if (best_result.is_null()) {
            return json_resp({{"ok", false}, {"error", "no scan-metrics result available"}});
        }
        return json_resp(best_result);
    });

    CROW_ROUTE(app, "/api/guardrails").methods("GET"_method)
    ([state]() {
        auto sg = scan_guardrails(state->job_store);
        json checks = json::array();

        // --- scan ---
        std::string scan_status = "check";
        std::string scan_label = "Scan ausstehend";
        if (sg.contains("checks") && sg["checks"].is_array()) {
            for (const auto& c : sg["checks"]) {
                std::string id = c.value("id", "");
                std::string st = c.value("status", "check");
                if (id == "scan_ok") {
                    scan_status = st;
                    scan_label = c.value("label", scan_label);
                }
            }
        }
        checks.push_back({{"id", "scan"}, {"status", scan_status}, {"label", scan_label}});

        // --- config ---
        std::string config_status = "check";
        std::string config_label = "Config fehlt";
        std::string bge_pcc_status = "check";
        std::string bge_pcc_label = "BGE:aus PCC:aus";
        const auto& cfg_path = state->runtime.default_config_path;
        if (fs::exists(cfg_path)) {
            try {
                std::ifstream in(cfg_path);
                std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
                in.close();
                if (!content.empty()) {
                    YAML::Node root = YAML::Load(content);
                    config_status = "ok";
                    config_label = "Config geladen";

                    bool bge_enabled = root["bge"] && root["bge"].IsMap() && root["bge"]["enabled"] && root["bge"]["enabled"].as<bool>();
                    bool pcc_enabled = root["pcc"] && root["pcc"].IsMap() && root["pcc"]["enabled"] && root["pcc"]["enabled"].as<bool>();
                    bge_pcc_status = (bge_enabled || pcc_enabled) ? "ok" : "check";
                    bge_pcc_label = std::string("BGE:") + (bge_enabled ? "an" : "aus") + " PCC:" + (pcc_enabled ? "an" : "aus");
                }
            } catch (...) {
                config_status = "error";
                config_label = "Config ung\u00fcltig";
            }
        }
        checks.push_back({{"id", "config"}, {"status", config_status}, {"label", config_label}});
        checks.push_back({{"id", "bge_pcc"}, {"status", bge_pcc_status}, {"label", bge_pcc_label}});

        // --- calibration --- (from scan job data)
        std::string cal_status = "check";
        std::string cal_label = "Kalibrierung ausstehend";
        auto scan_job = latest_scan_job(state->job_store);
        if (scan_job.has_value()) {
            const auto& data = scan_job->data;
            if (data.contains("calibration") && data["calibration"].is_object()) {
                const auto& cal = data["calibration"];
                bool bias_en = cal.value("bias_enabled", false);
                bool dark_en = cal.value("dark_enabled", false);
                bool flat_en = cal.value("flat_enabled", false);
                if (bias_en || dark_en || flat_en) {
                    cal_status = "ok";
                    cal_label = std::string("Cal:") + (bias_en ? " B" : "") + (dark_en ? " D" : "") + (flat_en ? " F" : "");
                }
            }
        }
        checks.push_back({{"id", "calibration"}, {"status", cal_status}, {"label", cal_label}});

        // Overall status
        std::string overall = "ok";
        for (const auto& c : checks) {
            std::string s = c.value("status", "check");
            if (s == "error") { overall = "error"; break; }
            if (s == "check") overall = "check";
        }
        return json_resp({{"status", overall}, {"checks", checks}});
    });
}
