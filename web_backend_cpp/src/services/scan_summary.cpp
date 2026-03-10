#include "services/scan_summary.hpp"

std::optional<Job> latest_scan_job(const InMemoryJobStore& store) {
    auto jobs = store.list(200);
    for (auto& j : jobs) {
        if (j.type == "scan") return j;
    }
    return std::nullopt;
}

nlohmann::json summarize_scan_job(const std::optional<Job>& job,
                                  const std::string& fallback_input_path) {
    nlohmann::json summary = {
        {"has_scan",     false},
        {"input_path",   fallback_input_path},
        {"input_dirs",   nlohmann::json::array()},
        {"frame_count",  0},
        {"color_mode",   ""},
        {"scan_state",   ""},
    };
    if (!job.has_value()) return summary;

    summary["scan_state"] = job_state_str(job->state);
    const auto& data = job->data;
    if (data.contains("result")) {
        const auto& r = data["result"];
        summary["has_scan"]    = true;
        summary["frame_count"] = r.value("frame_count", 0);
        summary["color_mode"]  = r.value("color_mode", "");
        if (r.contains("input_path"))
            summary["input_path"] = r["input_path"];
        if (r.contains("input_dirs"))
            summary["input_dirs"] = r["input_dirs"];
        if (r.contains("quality_score"))
            summary["quality_score"] = r["quality_score"];
        if (r.contains("warnings"))
            summary["warnings"] = r["warnings"];
    }
    return summary;
}

nlohmann::json scan_quality(const InMemoryJobStore& store) {
    auto job = latest_scan_job(store);
    auto summary = summarize_scan_job(job);
    double score = 0.0;
    if (job && job->data.contains("result"))
        score = job->data["result"].value("quality_score", 0.0);
    return {{"score", score}, {"scan", summary}};
}

nlohmann::json scan_guardrails(const InMemoryJobStore& store) {
    auto job = latest_scan_job(store);
    auto summary = summarize_scan_job(job);
    bool has_scan = summary.value("has_scan", false);
    std::string color_mode = summary.value("color_mode", "");

    nlohmann::json checks = nlohmann::json::array();
    checks.push_back({
        {"id",     "scan_ok"},
        {"status", has_scan ? "ok" : "check"},
        {"label",  has_scan ? "Scan vorhanden" : "Scan ausstehend"},
    });
    checks.push_back({
        {"id",     "color_mode"},
        {"status", color_mode.empty() ? "check" : "ok"},
        {"label",  color_mode.empty() ? "Color mode nicht erkannt" : "Color mode: " + color_mode},
    });

    std::string overall = has_scan ? "ok" : "check";
    return {{"status", overall}, {"checks", checks}};
}
