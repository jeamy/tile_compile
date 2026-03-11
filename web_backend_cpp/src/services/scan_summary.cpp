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
    if (!job.has_value()) {
        return {
            {"has_scan", false},
            {"job_id", nullptr},
            {"job_state", "pending"},
            {"input_path", fallback_input_path},
            {"input_dirs", nlohmann::json::array()},
            {"ok", false},
            {"frames_detected", 0},
            {"image_width", 0},
            {"image_height", 0},
            {"color_mode", "UNKNOWN"},
            {"color_mode_candidates", nlohmann::json::array()},
            {"bayer_pattern", nullptr},
            {"requires_user_confirmation", false},
            {"errors", nlohmann::json::array()},
            {"warnings", nlohmann::json::array()},
            {"frames", nlohmann::json::array()},
            {"per_dir_results", nlohmann::json::array()},
        };
    }

    const auto& data = job->data;
    nlohmann::json result = data.contains("result") && data["result"].is_object()
        ? data["result"]
        : nlohmann::json::object();
    nlohmann::json errors = result.contains("errors") && result["errors"].is_array()
        ? result["errors"]
        : nlohmann::json::array();
    nlohmann::json warnings = result.contains("warnings") && result["warnings"].is_array()
        ? result["warnings"]
        : nlohmann::json::array();
    nlohmann::json input_dirs = result.contains("input_dirs") && result["input_dirs"].is_array()
        ? result["input_dirs"]
        : (data.contains("input_dirs") && data["input_dirs"].is_array() ? data["input_dirs"] : nlohmann::json::array());
    nlohmann::json frames = result.contains("frames") && result["frames"].is_array()
        ? result["frames"]
        : nlohmann::json::array();
    nlohmann::json per_dir_results = result.contains("per_dir_results") && result["per_dir_results"].is_array()
        ? result["per_dir_results"]
        : nlohmann::json::array();

    std::string input_path = result.value("input_path", data.value("input_path", fallback_input_path));
    bool ok = result.contains("ok") && result["ok"].is_boolean()
        ? result["ok"].get<bool>()
        : errors.empty();

    return {
        {"has_scan", true},
        {"job_id", job->job_id},
        {"job_state", job_state_str(job->state)},
        {"input_path", input_path},
        {"input_dirs", input_dirs},
        {"ok", ok},
        {"frames_detected", result.value("frames_detected", 0)},
        {"image_width", result.value("image_width", 0)},
        {"image_height", result.value("image_height", 0)},
        {"color_mode", result.value("color_mode", "UNKNOWN")},
        {"color_mode_candidates", result.contains("color_mode_candidates") && result["color_mode_candidates"].is_array() ? result["color_mode_candidates"] : nlohmann::json::array()},
        {"bayer_pattern", result.contains("bayer_pattern") ? result["bayer_pattern"] : nlohmann::json(nullptr)},
        {"requires_user_confirmation", result.value("requires_user_confirmation", false)},
        {"errors", errors},
        {"warnings", warnings},
        {"frames", frames},
        {"per_dir_results", per_dir_results},
    };
}

nlohmann::json scan_quality(const InMemoryJobStore& store) {
    auto job = latest_scan_job(store);
    auto summary = summarize_scan_job(job);
    if (!summary.value("has_scan", false)) {
        return {
            {"score", 0.0},
            {"factors", nlohmann::json::array({{{"id", "no_scan"}, {"value", 1.0}, {"label", "No scan run yet"}}})},
            {"scan", summary},
        };
    }

    int errors = static_cast<int>(summary.contains("errors") && summary["errors"].is_array() ? summary["errors"].size() : 0);
    int warnings = static_cast<int>(summary.contains("warnings") && summary["warnings"].is_array() ? summary["warnings"].size() : 0);
    double score = std::max(0.0, 1.0 - 0.25 * errors - 0.1 * warnings);
    return {
        {"score", std::round(score * 1000.0) / 1000.0},
        {"factors", nlohmann::json::array({
            {{"id", "errors"}, {"value", errors}, {"label", "scan errors"}},
            {{"id", "warnings"}, {"value", warnings}, {"label", "scan warnings"}},
        })},
        {"scan", summary},
    };
}

nlohmann::json scan_guardrails(const InMemoryJobStore& store) {
    auto job = latest_scan_job(store);
    auto summary = summarize_scan_job(job);
    if (!summary.value("has_scan", false)) {
        return {
            {"status", "check"},
            {"checks", nlohmann::json::array({{{"id", "scan"}, {"status", "check"}, {"label", "Scan ausstehend"}}})},
        };
    }

    nlohmann::json errors = summary.contains("errors") && summary["errors"].is_array()
        ? summary["errors"]
        : nlohmann::json::array();
    nlohmann::json warnings = summary.contains("warnings") && summary["warnings"].is_array()
        ? summary["warnings"]
        : nlohmann::json::array();
    bool requires_confirm = summary.value("requires_user_confirmation", false);
    std::string color_mode = summary.value("color_mode", "UNKNOWN");
    std::string color_mode_status = (requires_confirm || color_mode.empty() || color_mode == "UNKNOWN") ? "check" : "ok";
    std::string color_mode_label = color_mode_status == "check" ? "Color mode bestaetigen" : "Color mode: " + color_mode;

    nlohmann::json checks = nlohmann::json::array({
        {{"id", "scan_ok"}, {"status", errors.empty() ? "ok" : "error"}, {"label", errors.empty() ? "Scan erfolgreich" : "Scan mit Fehlern"}, {"count", errors.size()}},
        {{"id", "color_mode"}, {"status", color_mode_status}, {"label", color_mode_label}, {"value", color_mode}},
        {{"id", "scan_warnings"}, {"status", warnings.empty() ? "ok" : "check"}, {"label", warnings.empty() ? "Keine Scan-Warnungen" : "Warnungen vorhanden"}, {"count", warnings.size()}},
    });
    std::string status = !errors.empty() ? "error" : (!warnings.empty() ? "check" : "ok");
    return {{"status", status}, {"checks", checks}};
}
