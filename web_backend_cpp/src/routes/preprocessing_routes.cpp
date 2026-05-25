#include "routes/preprocessing_routes.hpp"
#include "routes/route_utils.hpp"
#include "services/preprocessing_service.hpp"

#include <chrono>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace fs = std::filesystem;
using namespace tile_compile::routes;
namespace prep = tile_compile::preprocessing_service;

namespace {

nlohmann::json parse_body(const crow::request& req) {
    auto body = nlohmann::json::parse(req.body, nullptr, false);
    return body.is_discarded() ? nlohmann::json::object() : body;
}

std::optional<fs::path> resolve_existing_input(const std::shared_ptr<AppState>& state,
                                               const std::string& raw,
                                               crow::response& error) {
    if (raw.empty()) {
        error = err_resp("BAD_REQUEST", "lights_dir or input_dir is required", 400);
        return std::nullopt;
    }
    const auto resolved = state->runtime.resolve_input_path(fs::path(raw), !fs::path(raw).is_absolute());
    if (resolved.status == PathStatus::not_allowed) {
        error = err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + raw, 403, {{"path", raw}});
        return std::nullopt;
    }
    if (resolved.status == PathStatus::not_found) {
        error = err_resp("PATH_NOT_FOUND", "Path not found: " + raw, 422, {{"path", raw}});
        return std::nullopt;
    }
    return resolved.path;
}

nlohmann::json merge_defaults(nlohmann::json overrides) {
    nlohmann::json cfg = prep::default_config();
    if (!overrides.is_object()) return cfg;
    cfg.update(overrides, true);
    return cfg;
}

bool one_of(const std::string& value, std::initializer_list<const char*> allowed) {
    return std::find_if(allowed.begin(), allowed.end(), [&](const char* item) {
        return value == item;
    }) != allowed.end();
}

double number_at(const nlohmann::json& obj, const std::string& key, double fallback) {
    return obj.contains(key) && obj[key].is_number() ? obj[key].get<double>() : fallback;
}

nlohmann::json object_at(const nlohmann::json& obj, const std::string& key) {
    return obj.contains(key) && obj[key].is_object() ? obj[key] : nlohmann::json::object();
}

void validate_preprocessing_config(const nlohmann::json& cfg) {
    if (cfg.value("mode", "") != "linear_prestack") throw std::runtime_error("preprocessing.mode must be linear_prestack");
    if (!one_of(cfg.value("input_mode", ""), {"auto", "cfa_osc", "mono"})) throw std::runtime_error("preprocessing.input_mode must be auto, cfa_osc, or mono");
    if (cfg.value("raw_formats", "") != "tile_compile") throw std::runtime_error("preprocessing.raw_formats must be tile_compile");
    if (cfg.value("cfa_mode", "") != "tile_compile") throw std::runtime_error("preprocessing.cfa_mode must be tile_compile");
    if (!one_of(cfg.value("mono_mode", ""), {"auto", "mono"})) throw std::runtime_error("preprocessing.mono_mode must be auto or mono");
    if (cfg.value("registration_reference", "") != "best_quality") throw std::runtime_error("preprocessing.registration_reference must be best_quality");

    const auto rejection = object_at(cfg, "rejection");
    if (!one_of(rejection.value("method", ""), {"sigma", "median", "winsor"})) throw std::runtime_error("preprocessing.rejection.method must be sigma, median, or winsor");
    if (number_at(rejection, "low", 0.0) <= 0.0 || number_at(rejection, "high", 0.0) <= 0.0) throw std::runtime_error("preprocessing.rejection low/high must be > 0");
    if (number_at(rejection, "max_iters", 0.0) < 1.0) throw std::runtime_error("preprocessing.rejection.max_iters must be >= 1");
    const double min_fraction = number_at(rejection, "min_fraction", -1.0);
    if (min_fraction < 0.0 || min_fraction > 1.0) throw std::runtime_error("preprocessing.rejection.min_fraction must be in [0,1]");

    const auto quality = object_at(cfg, "quality_filter");
    if (!one_of(quality.value("mode", ""), {"auto", "strict", "relaxed", "off"})) throw std::runtime_error("preprocessing.quality_filter.mode must be auto, strict, relaxed, or off");
    if (number_at(quality, "min_stars", 0.0) < 0.0) throw std::runtime_error("preprocessing.quality_filter.min_stars must be >= 0");
    if (number_at(quality, "max_fwhm_sigma", 0.0) <= 0.0) throw std::runtime_error("preprocessing.quality_filter.max_fwhm_sigma must be > 0");
    const double ecc = number_at(quality, "max_eccentricity", -1.0);
    const double cc = number_at(quality, "min_correlation", -1.0);
    if (ecc < 0.0 || ecc > 1.0) throw std::runtime_error("preprocessing.quality_filter.max_eccentricity must be in [0,1]");
    if (cc < 0.0 || cc > 1.0) throw std::runtime_error("preprocessing.quality_filter.min_correlation must be in [0,1]");
    if (quality.contains("manual_overrides") && !quality["manual_overrides"].is_object()) throw std::runtime_error("preprocessing.quality_filter.manual_overrides must be an object");

    const auto stacking = object_at(cfg, "stacking");
    if (!one_of(stacking.value("normalization", ""), {"addscale", "background", "median", "none"})) throw std::runtime_error("preprocessing.stacking.normalization must be addscale, background, median, or none");
    if (!one_of(stacking.value("weighting", ""), {"quality", "uniform"})) throw std::runtime_error("preprocessing.stacking.weighting must be quality or uniform");
    if (number_at(stacking, "cosmetic_correction_sigma", 0.0) <= 0.0) throw std::runtime_error("preprocessing.stacking.cosmetic_correction_sigma must be > 0");
    if (number_at(stacking, "per_frame_cosmetic_correction_sigma", 0.0) <= 0.0) throw std::runtime_error("preprocessing.stacking.per_frame_cosmetic_correction_sigma must be > 0");

    const auto pp = object_at(cfg, "postprocess");
    if (pp.value("pcc", false) && !pp.value("astrometry", false)) throw std::runtime_error("preprocessing.postprocess.pcc requires preprocessing.postprocess.astrometry");

    const auto hms = object_at(cfg, "hypermetric_stretch");
    if (!one_of(hms.value("mode", ""), {"ready_to_use", "scientific"})) throw std::runtime_error("preprocessing.hypermetric_stretch.mode must be ready_to_use or scientific");
    const double target_bg = number_at(hms, "target_bg", 0.0);
    if (target_bg < 0.05 || target_bg > 0.50) throw std::runtime_error("preprocessing.hypermetric_stretch.target_bg must be in [0.05,0.50]");
    if (number_at(hms, "protect_b", 0.0) < 0.1) throw std::runtime_error("preprocessing.hypermetric_stretch.protect_b must be >= 0.1");
    const double convergence = number_at(hms, "convergence_power", 0.0);
    if (convergence < 1.0 || convergence > 10.0) throw std::runtime_error("preprocessing.hypermetric_stretch.convergence_power must be in [1,10]");
    if (!one_of(hms.value("log_d_mode", ""), {"auto", "fixed"})) throw std::runtime_error("preprocessing.hypermetric_stretch.log_d_mode must be auto or fixed");
    if (!one_of(hms.value("color_strategy", ""), {"auto", "fixed"})) throw std::runtime_error("preprocessing.hypermetric_stretch.color_strategy must be auto or fixed");
    const double grip = number_at(hms, "color_grip", -1.0);
    if (grip < 0.0 || grip > 1.0) throw std::runtime_error("preprocessing.hypermetric_stretch.color_grip must be in [0,1]");
    if (hms.value("output_rgb", std::string()).empty()) throw std::runtime_error("preprocessing.hypermetric_stretch.output_rgb must not be empty");

    const auto runtime = object_at(cfg, "runtime_limits");
    if (number_at(runtime, "parallel_workers", 0.0) < 1.0) throw std::runtime_error("preprocessing.runtime_limits.parallel_workers must be >= 1");
    if (number_at(runtime, "memory_budget", 0.0) < 1.0) throw std::runtime_error("preprocessing.runtime_limits.memory_budget must be >= 1");
}

nlohmann::json effective_parameters(const std::shared_ptr<AppState>& state) {
    std::lock_guard<std::mutex> lock(state->state_mutex);
    return state->preprocessing_parameters.is_object() && !state->preprocessing_parameters.empty()
        ? state->preprocessing_parameters
        : prep::default_config();
}

} // namespace

void register_preprocessing_routes(CrowApp& app,
                                   std::shared_ptr<AppState> state) {
    CROW_ROUTE(app, "/api/tools/preprocessing/defaults").methods("GET"_method)
    ([](const crow::request&) {
        return json_resp({
            {"config", prep::default_config()},
            {"phases", prep::phase_order()},
            {"groups", prep::parameter_groups()},
        });
    });

    CROW_ROUTE(app, "/api/tools/preprocessing/parameters").methods("GET"_method)
    ([state](const crow::request&) {
        return json_resp({
            {"config", effective_parameters(state)},
            {"groups", prep::parameter_groups()},
            {"process", "preprocessing"},
            {"separate_from_tile_compile", true},
        });
    });

    CROW_ROUTE(app, "/api/tools/preprocessing/parameters").methods("PATCH"_method)
    ([state](const crow::request& req) {
        try {
            const auto body = parse_body(req);
            const nlohmann::json config = merge_defaults(body.value("config", body));
            validate_preprocessing_config(config);
            {
                std::lock_guard<std::mutex> lock(state->state_mutex);
                state->preprocessing_parameters = config;
            }
            return json_resp({
                {"ok", true},
                {"config", config},
                {"groups", prep::parameter_groups()},
                {"validation", {
                    {"status", "ok"},
                    {"scope", "configuration"},
                    {"process", "preprocessing"},
                    {"checks", {
                        "schema defaults merged",
                        "enum values",
                        "numeric ranges",
                        "postprocess dependencies",
                        "quality filter thresholds",
                        "hypermetric stretch ranges",
                        "runtime limits",
                        "manual frame override shape"
                    }}
                }},
            });
        } catch (const std::exception& e) {
            return err_resp("VALIDATION_ERROR", e.what(), 400);
        }
    });

    CROW_ROUTE(app, "/api/tools/preprocessing/scan").methods("POST"_method)
    ([state](const crow::request& req) {
        const auto body = parse_body(req);
        const std::string raw_input = body.value("lights_dir", body.value("input_dir", ""));
        crow::response error;
        auto resolved = resolve_existing_input(state, raw_input, error);
        if (!resolved) return error;

        nlohmann::json initial_data = {
            {"process", "preprocessing"},
            {"input_path", resolved->string()},
            {"raw_formats", "tile_compile"},
            {"requested_config", merge_defaults(body.value("config", nlohmann::json::object()))},
        };
        std::vector<std::string> args = {
            state->runtime.cli_exe,
            "scan",
            resolved->string(),
            "--frames-min",
            "1",
            "--json",
        };
        initial_data["command"] = args;
        const std::string job_id = state->subprocess_manager.launch(
            "preprocessing_scan",
            args,
            state->runtime.project_root.string(),
            "",
            initial_data);
        return json_resp({{"ok", true}, {"job_id", job_id}, {"status_url", "/api/tools/preprocessing/status?job_id=" + job_id}});
    });

    CROW_ROUTE(app, "/api/tools/preprocessing/run").methods("POST"_method)
    ([state](const crow::request& req) {
        const auto body = parse_body(req);
        nlohmann::json effective_config = body.empty() ? effective_parameters(state)
                                                       : merge_defaults(body.value("config", body));
        try {
            validate_preprocessing_config(effective_config);
        } catch (const std::exception& e) {
            return err_resp("VALIDATION_ERROR", e.what(), 400);
        }
        const std::string raw_input = body.value("lights_dir", effective_config.value("lights_dir", std::string()));
        crow::response error;
        auto resolved = resolve_existing_input(state, raw_input, error);
        if (!resolved) return error;
        effective_config["lights_dir"] = resolved->string();

        const auto now_tp = std::chrono::system_clock::now();
        const auto now_tt = std::chrono::system_clock::to_time_t(now_tp);
        std::tm tm_buf{};
#ifdef _WIN32
        localtime_s(&tm_buf, &now_tt);
#else
        localtime_r(&now_tt, &tm_buf);
#endif
        std::ostringstream ts_ss;
        ts_ss << std::put_time(&tm_buf, "%Y%m%d_%H%M%S");
        const std::string timestamp = ts_ss.str();
        std::string raw_run_name = body.value("run_name", effective_config.value("run_name", std::string()));
        std::string base_name;
        if (!raw_run_name.empty()) {
            for (char& ch : raw_run_name) {
                bool ok = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                          (ch >= '0' && ch <= '9') || ch == '.' || ch == '_' || ch == '-';
                if (!ok) ch = '_';
            }
            while (!raw_run_name.empty() && raw_run_name.front() == '_') raw_run_name.erase(raw_run_name.begin());
            while (!raw_run_name.empty() && raw_run_name.back() == '_') raw_run_name.pop_back();
            base_name = raw_run_name.empty() ? "rs_run" : raw_run_name;
        } else {
            base_name = "rs_run";
        }
        const std::string run_id = base_name + "_" + timestamp;
        const fs::path run_dir = state->runtime.runs_dir / run_id;
        std::vector<std::string> args = {
            state->runtime.runner_exe,
            "preprocess",
            "--config",
            "-",
            "--stdin",
            "--runs-dir",
            state->runtime.runs_dir.string(),
            "--project-root",
            state->runtime.project_root.string(),
            "--run-id",
            run_id,
        };
        nlohmann::json initial_data = {
            {"process", "preprocessing"},
            {"run_id", run_id},
            {"run_dir", run_dir.string()},
            {"effective_config", effective_config},
            {"current_phase", "INPUT_SCAN"},
            {"command", args},
        };
        const std::string job_id = state->subprocess_manager.launch(
            "preprocessing_run",
            args,
            state->runtime.project_root.string(),
            run_id,
            initial_data,
            effective_config.dump());

        return json_resp({{"ok", true}, {"job_id", job_id}, {"run_id", run_id}, {"run_dir", run_dir.string()}});
    });

    CROW_ROUTE(app, "/api/tools/preprocessing/cancel").methods("POST"_method)
    ([state](const crow::request& req) {
        const auto body = parse_body(req);
        const std::string job_id = body.value("job_id", std::string());
        if (job_id.empty()) return err_resp("BAD_REQUEST", "job_id is required", 400);
        const bool subprocess_cancelled = state->subprocess_manager.cancel(job_id);
        state->job_store.cancel(job_id);
        return json_resp({{"ok", true}, {"job_id", job_id}, {"subprocess_cancelled", subprocess_cancelled}});
    });

    CROW_ROUTE(app, "/api/tools/preprocessing/status").methods("GET"_method)
    ([state](const crow::request& req) {
        const std::string job_id = parse_string_param(req, "job_id", "");

        // Maps internal sub-phase names emitted by the runner to the canonical preprocessing phase
        static const std::unordered_map<std::string, std::string> SUBPHASE_MAP = {
            {"CHANNEL_SPLIT",       "CFA_CHANNEL_PREP"},
            {"NORMALIZATION",       "CFA_CHANNEL_PREP"},
            {"GLOBAL_METRICS",      "CFA_CHANNEL_PREP"},
            {"REFERENCE_SELECTION", "REFERENCE_SELECTION"},
            {"PREWARP",             "REGISTRATION"},
            {"FRAME_FILTERING",     "FRAME_FILTERING"},
        };

        // Parse preprocessing events.jsonl to get live phase/progress
        auto enrich_with_live = [&](nlohmann::json result, const nlohmann::json& job_json) -> nlohmann::json {
            const std::string s = result.value("status", std::string());
            if (s != "running") return result;
            const nlohmann::json& data = job_json.contains("data") && job_json["data"].is_object()
                ? job_json["data"] : nlohmann::json::object();
            const std::string run_dir_str = data.value("run_dir", std::string());
            if (run_dir_str.empty()) return result;
            try {
                const fs::path events_path = fs::path(run_dir_str) / "artifacts" / "preprocess" / "events.jsonl";
                if (!fs::exists(events_path)) return result;
                std::ifstream f(events_path);
                if (!f) return result;
                // Build phase map from events: track status and pct per phase
                std::map<std::string, std::string> phase_status;
                std::map<std::string, double> phase_pct;
                std::string current_phase;
                std::string line;
                while (std::getline(f, line)) {
                    if (line.empty()) continue;
                    auto ev = nlohmann::json::parse(line, nullptr, false);
                    if (ev.is_discarded() || !ev.is_object()) continue;
                    const std::string type = ev.value("type", std::string());
                    const std::string raw_phase = ev.value("phase_name", std::string());
                    if (raw_phase.empty()) continue;
                    auto map_it = SUBPHASE_MAP.find(raw_phase);
                    const std::string phase = (map_it != SUBPHASE_MAP.end()) ? map_it->second : raw_phase;
                    if (type == "phase_start") {
                        phase_status[phase] = "running";
                        phase_pct[phase] = 0.0;
                        current_phase = phase;
                    } else if (type == "phase_end") {
                        const std::string st = ev.value("status", "ok");
                        phase_status[phase] = (st == "ok" || st == "skipped") ? st : "failed";
                        phase_pct[phase] = (st == "ok") ? 1.0 : phase_pct[phase];
                        if (current_phase == phase) current_phase = "";
                    } else if (type == "phase_progress") {
                        phase_status[phase] = "running";
                        phase_pct[phase] = ev.value("progress", 0.0);
                        current_phase = phase;
                    }
                }
                if (phase_status.empty()) return result;
                // Rebuild phases array preserving order from result["phases"]
                auto& phases = result["phases"];
                if (phases.is_array()) {
                    for (auto& ph : phases) {
                        const std::string pname = ph.value("phase", std::string());
                        if (phase_status.count(pname)) {
                            ph["status"] = phase_status[pname];
                            ph["pct"] = phase_pct.count(pname) ? phase_pct[pname] : 0.0;
                        }
                    }
                }
                if (!current_phase.empty()) {
                    result["current_phase"] = current_phase;
                    // compute progress: fraction of phases done
                    if (phases.is_array() && !phases.empty()) {
                        int done = 0;
                        double cur_pct = 0.0;
                        for (const auto& ph : phases) {
                            const std::string st = ph.value("status", std::string());
                            if (st == "ok" || st == "skipped") ++done;
                            else if (st == "running") cur_pct = ph.value("pct", 0.0);
                        }
                        result["progress"] = (done + cur_pct) / static_cast<double>(phases.size());
                    }
                }
            } catch (...) {}
            return result;
        };

        if (!job_id.empty()) {
            auto job = state->job_store.get(job_id);
            if (!job) return err_resp("NOT_FOUND", "job '" + job_id + "' not found", 404);
            nlohmann::json job_json = job_to_json(*job);
            if (job->type == "preprocessing_scan" && job_json.contains("data") &&
                job_json["data"].contains("result")) {
                job_json["scan"] = prep::normalize_scan_result(
                    job_json["data"]["result"],
                    job_json["data"].value("input_path", ""));
            }
            return json_resp(enrich_with_live(prep::read_status_from_job(job_json), job_json));
        }
        for (const auto& job : state->job_store.list(200)) {
            if (job.type == "preprocessing_run" || job.type == "preprocessing_scan") {
                nlohmann::json job_json = job_to_json(job);
                return json_resp(enrich_with_live(prep::read_status_from_job(job_json), job_json));
            }
        }
        return json_resp({
            {"status", "idle"},
            {"current_phase", nullptr},
            {"progress", 0.0},
            {"phases", nlohmann::json::array()},
        });
    });

    CROW_ROUTE(app, "/api/tools/preprocessing/report").methods("GET"_method)
    ([state](const crow::request& req) {
        const std::string job_id = parse_string_param(req, "job_id", "");
        if (job_id.empty()) return err_resp("BAD_REQUEST", "job_id is required", 400);
        auto job = state->job_store.get(job_id);
        if (!job) return err_resp("NOT_FOUND", "job '" + job_id + "' not found", 404);
        const fs::path run_dir = (job->data.is_object() && job->data.contains("run_dir") && job->data["run_dir"].is_string())
            ? fs::path(job->data["run_dir"].get<std::string>())
            : prep::run_dir_for_job(state->runtime.runs_dir, job_id);
        const fs::path artifacts_dir = run_dir / "artifacts" / "preprocess";
        return json_resp({
            {"job_id", job_id},
            {"run_id", job->data.is_object() && job->data.contains("run_id") ? job->data["run_id"] : nlohmann::json(nullptr)},
            {"run_dir", run_dir.string()},
            {"artifacts_dir", artifacts_dir.string()},
            {"report_json", (artifacts_dir / "preprocessing_report.json").string()},
            {"report_markdown", (artifacts_dir / "preprocessing_report.md").string()},
            {"report_html", (artifacts_dir / "preprocessing_report.html").string()},
            {"events", (artifacts_dir / "events.jsonl").string()},
            {"manifest", (artifacts_dir / "artifacts_manifest.json").string()},
        });
    });
}
