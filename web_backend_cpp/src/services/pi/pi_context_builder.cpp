#include "services/pi/pi_context_builder.hpp"

#include "app_state.hpp"

#include <algorithm>

namespace tile_compile::pi {
namespace {

nlohmann::json path_summary(const std::filesystem::path& path) {
    return {
        {"name", path.filename().string()},
        {"extension", path.extension().string()},
        {"is_absolute", path.is_absolute()}
    };
}

nlohmann::json job_summary(const Job& job) {
    nlohmann::json out = {
        {"job_id", job.job_id},
        {"type", job.type},
        {"state", job_state_str(job.state)},
        {"created_at", job.created_at},
        {"started_at", job.started_at},
        {"ended_at", job.ended_at}
    };
    if (job.data.is_object()) {
        if (job.data.contains("run_id")) out["run_id"] = job.data["run_id"];
        if (job.data.contains("input_path")) out["input_path"] = path_summary(job.data["input_path"].get<std::string>());
        if (job.data.contains("result") && job.data["result"].is_object()) {
            const auto& result = job.data["result"];
            nlohmann::json compact = nlohmann::json::object();
            for (const std::string key : {
                     "ok", "frames_detected", "frames_total", "image_width", "image_height",
                     "color_mode", "bayer_pattern", "warnings_total", "errors_total"
                 }) {
                if (result.contains(key)) compact[key] = result[key];
            }
            if (!compact.empty()) out["result"] = std::move(compact);
        }
    }
    return out;
}

} // namespace

PiContextBuilder::PiContextBuilder(std::shared_ptr<AppState> state)
    : _state(std::move(state)) {}

nlohmann::json PiContextBuilder::build_overview_context() const {
    nlohmann::json state_snapshot = nlohmann::json::object();
    std::string current_run_id;
    std::string active_config_revision_id;
    std::string last_scan_input_path;
    {
        std::lock_guard<std::mutex> lk(_state->state_mutex);
        current_run_id = _state->current_run_id;
        active_config_revision_id = _state->active_config_revision_id;
        last_scan_input_path = _state->last_scan_input_path;
        if (_state->ui_state.is_object()) {
            for (const std::string key : {"ai", "selectedHistoryRunId", "currentRunId"}) {
                if (_state->ui_state.contains(key)) state_snapshot[key] = _state->ui_state[key];
            }
        }
    }

    nlohmann::json jobs = nlohmann::json::array();
    nlohmann::json latest_scan = nullptr;
    for (const auto& job : _state->job_store.list(20)) {
        const auto compact = job_summary(job);
        jobs.push_back(compact);
        if (latest_scan.is_null() && job.type == "scan") latest_scan = compact;
    }

    return {
        {"schema_version", "pi.context-overview.v1"},
        {"privacy_class", "metadata_only"},
        {"runtime", {
            {"project_root", path_summary(_state->runtime.project_root)},
            {"runs_dir", path_summary(_state->runtime.runs_dir)},
            {"default_config_path", path_summary(_state->runtime.default_config_path)},
            {"schema_path", path_summary(_state->runtime.schema_path)},
            {"host", _state->runtime.host},
            {"port", _state->runtime.port}
        }},
        {"state", {
            {"current_run_id", current_run_id},
            {"active_config_revision_id", active_config_revision_id},
            {"last_scan_input_path", last_scan_input_path.empty() ? nlohmann::json(nullptr) : path_summary(last_scan_input_path)},
            {"ui", state_snapshot}
        }},
        {"jobs", jobs},
        {"latest_scan_job", latest_scan}
    };
}

} // namespace tile_compile::pi
