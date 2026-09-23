#include "services/scan_metrics_cache.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {
std::string normalized_cache_object(std::string value) {
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), value.end());
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    return value;
}

}  // namespace

std::string scan_metrics_cache_key(const std::string& input_path,
                                   const std::string& object_name,
                                   int frame_count) {
    return input_path + "|" + normalized_cache_object(object_name) + "|" + std::to_string(frame_count);
}

namespace {
std::string fnv1a_hex(const std::string& value) {
    uint64_t hash = 1469598103934665603ull;
    for (unsigned char ch : value) {
        hash ^= ch;
        hash *= 1099511628211ull;
    }
    std::ostringstream out;
    out << std::hex << hash;
    return out.str();
}

fs::path scan_metrics_cache_dir(const std::shared_ptr<AppState>& state) {
    return state->runtime.project_root / "runs" / ".pi_memory" / "scan_metrics_cache";
}

fs::path scan_metrics_cache_path(const std::shared_ptr<AppState>& state, const std::string& cache_key) {
    return scan_metrics_cache_dir(state) / (fnv1a_hex(cache_key) + ".json");
}

}  // namespace

json find_disk_cached_scan_metrics(const std::shared_ptr<AppState>& state,
                                   const std::string& cache_key) {
    const auto path = scan_metrics_cache_path(state, cache_key);
    std::ifstream in(path);
    if (!in) return json::object();
    auto parsed = json::parse(in, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) return json::object();
    if (parsed.value("cache_key", std::string()) != cache_key) return json::object();
    json result = parsed.contains("result") && parsed["result"].is_object()
        ? parsed["result"]
        : parsed;
    if (!result.value("ok", false)) return json::object();
    result["cache_hit"] = true;
    result["cache_source"] = "disk";
    result["cache_key"] = cache_key;
    if (parsed.contains("job_id")) result["cache_source_job_id"] = parsed["job_id"];
    if (parsed.contains("created_at")) result["cache_created_at"] = parsed["created_at"];
    return result;
}

json find_cached_scan_metrics(const std::shared_ptr<AppState>& state,
                              const std::string& input_path,
                              const std::string& object_name,
                              int frame_count) {
    const std::string wanted_key = scan_metrics_cache_key(input_path, object_name, frame_count);
    auto jobs = state->job_store.list();
    json best = nullptr;
    std::string best_job_id;
    for (const auto& j : jobs) {
        if (j.type != "scan-metrics" || j.state != JobState::ok) continue;
        if (!j.data.contains("result") || !j.data["result"].is_object()) continue;
        const auto& r = j.data["result"];
        if (!r.value("ok", false)) continue;
        const std::string cached_input = j.data.value("input_path", r.value("input_path", std::string()));
        const std::string cached_object = j.data.value("object_name", r.value("object_name", r.value("target", std::string())));
        const int cached_frames = r.value("frames_total", r.value("frame_count", 0));
        if (scan_metrics_cache_key(cached_input, cached_object, cached_frames) != wanted_key) continue;
        if (j.job_id > best_job_id) {
            best_job_id = j.job_id;
            best = r;
        }
    }
    if (best.is_null()) return json::object();
    best["cache_hit"] = true;
    best["cache_source"] = "job_store";
    best["cache_source_job_id"] = best_job_id;
    best["cache_key"] = wanted_key;
    return best;
}

