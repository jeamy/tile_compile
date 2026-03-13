#include "services/run_inspector.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <map>
#include <optional>
#include <yaml-cpp/yaml.h>

namespace {

std::optional<fs::path> find_event_file(const fs::path& run_dir) {
    const std::vector<fs::path> candidates = {
        run_dir / "logs" / "run_events.jsonl",
        run_dir / "events.jsonl",
        run_dir / "logs" / "events.jsonl",
    };
    for (const auto& candidate : candidates) {
        if (fs::exists(candidate) && fs::is_regular_file(candidate)) return candidate;
    }
    return std::nullopt;
}

std::vector<nlohmann::json> iter_jsonl(const fs::path& path) {
    std::vector<nlohmann::json> out;
    std::ifstream f(path);
    if (!f) return out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto parsed = nlohmann::json::parse(line, nullptr, false);
        if (!parsed.is_discarded() && parsed.is_object()) out.push_back(parsed);
    }
    return out;
}

std::string phase_name_from_event(const nlohmann::json& ev) {
    if (ev.contains("phase_name") && ev["phase_name"].is_string()) return ev["phase_name"].get<std::string>();
    if (ev.contains("phase") && ev["phase"].is_string()) return ev["phase"].get<std::string>();
    return "";
}

double clamp_progress(const nlohmann::json& value) {
    double v = 0.0;
    try { v = value.get<double>(); } catch (...) { return -1.0; }
    if (v < 0.0) return 0.0;
    if (v > 1.0) return 1.0;
    return v;
}

double overall_progress(const nlohmann::json& phases, const std::string& current_phase, const nlohmann::json& progress_map) {
    if (PHASE_ORDER.empty()) return 0.0;
    int completed = 0;
    for (const auto& phase : PHASE_ORDER) {
        for (const auto& entry : phases) {
            const std::string status = entry.value("status", std::string());
            if (entry.value("phase", std::string()) == phase && (status == "ok" || status == "skipped")) {
                ++completed;
                break;
            }
        }
    }
    double current_component = 0.0;
    if (!current_phase.empty() && progress_map.contains(current_phase)) {
        current_component = progress_map[current_phase].get<double>();
    } else if (!current_phase.empty()) {
        for (const auto& entry : phases) {
            if (entry.value("phase", std::string()) == current_phase) {
                current_component = entry.value("pct", 0.0);
                break;
            }
        }
    }
    double progress = (completed + current_component) / static_cast<double>(PHASE_ORDER.size());
    if (progress < 0.0) return 0.0;
    if (progress > 1.0) return 1.0;
    return progress;
}

std::string read_run_color_mode(const fs::path& run_dir) {
    fs::path config_path = run_dir / "config.yaml";
    std::ifstream f(config_path);
    if (f) {
        try {
            YAML::Node root = YAML::Load(f);
            if (root["data"] && root["data"].IsMap() && root["data"]["color_mode"]) {
                std::string color_mode = root["data"]["color_mode"].as<std::string>();
                if (!color_mode.empty()) {
                    std::transform(color_mode.begin(), color_mode.end(), color_mode.begin(), ::toupper);
                    return color_mode;
                }
            }
        } catch (...) {}
    }
    auto event_file = find_event_file(run_dir);
    if (!event_file) return "UNKNOWN";
    for (const auto& ev : iter_jsonl(*event_file)) {
        if (ev.contains("color_mode") && ev["color_mode"].is_string()) {
            std::string color_mode = ev["color_mode"].get<std::string>();
            if (!color_mode.empty()) {
                std::transform(color_mode.begin(), color_mode.end(), color_mode.begin(), ::toupper);
                return color_mode;
            }
        }
        if (ev.contains("payload") && ev["payload"].is_object() && ev["payload"].contains("color_mode") && ev["payload"]["color_mode"].is_string()) {
            std::string color_mode = ev["payload"]["color_mode"].get<std::string>();
            if (!color_mode.empty()) {
                std::transform(color_mode.begin(), color_mode.end(), color_mode.begin(), ::toupper);
                return color_mode;
            }
        }
    }
    return "UNKNOWN";
}

std::string extract_run_id_from_events(const fs::path& event_file) {
    for (const auto& ev : iter_jsonl(event_file)) {
        if (ev.contains("run_id") && ev["run_id"].is_string()) {
            std::string run_id = ev["run_id"].get<std::string>();
            if (!run_id.empty()) return run_id;
        }
    }
    return "";
}

std::string iso_utc_from_file_time(const fs::file_time_type& file_time) {
    const auto system_now = std::chrono::system_clock::now();
    const auto file_now = fs::file_time_type::clock::now();
    const auto system_tp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(file_time - file_now + system_now);
    const auto tt = std::chrono::system_clock::to_time_t(system_tp);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &tt);
#else
    gmtime_r(&tt, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

}

std::optional<Job> latest_run_job(const InMemoryJobStore& store, const std::string& run_id, int limit) {
    if (run_id.empty()) return std::nullopt;
    for (const auto& job : store.list(limit)) {
        if (job.type.rfind("run", 0) != 0) continue;
        const std::string job_run_id = job.data.is_object()
            ? job.data.value("run_id", job.run_id)
            : job.run_id;
        if (job_run_id == run_id) return job;
    }
    return std::nullopt;
}

void apply_job_state_to_run_status(nlohmann::json& status, const std::optional<Job>& job) {
    if (!job.has_value()) return;

    const std::string state = job_state_str(job->state);
    if (state == "pending" || state == "running") {
        status["status"] = state;
        if (!status.contains("progress") || !status["progress"].is_number()) {
            status["progress"] = 0.0;
        }
        const double job_progress = std::clamp(job->progress / 100.0, 0.0, 1.0);
        try {
            const double current = status["progress"].get<double>();
            status["progress"] = std::max(current, job_progress);
        } catch (...) {
            status["progress"] = job_progress;
        }
        return;
    }

    if (state == "cancelled") {
        status["status"] = "cancelled";
        status["current_phase"] = nullptr;
        return;
    }

    if (state == "error") {
        status["status"] = "failed";
        status["current_phase"] = nullptr;
        return;
    }

    if (state == "ok") {
        status["status"] = "completed";
        status["current_phase"] = nullptr;
        status["progress"] = 1.0;
    }
}

nlohmann::json read_run_status(const fs::path& run_dir) {
    nlohmann::json result = {
        {"run_dir", run_dir.string()},
        {"exists", fs::exists(run_dir)},
        {"status", "unknown"},
        {"color_mode", read_run_color_mode(run_dir)},
        {"current_phase", nullptr},
        {"progress", 0.0},
        {"phases", nlohmann::json::array()},
        {"events", nlohmann::json::array()},
    };
    for (const auto& phase : PHASE_ORDER) {
        result["phases"].push_back(nlohmann::json{{"phase", phase}, {"status", "pending"}, {"pct", 0.0}});
    }

    auto event_file = find_event_file(run_dir);
    if (!event_file) return result;

    nlohmann::json phases = nlohmann::json::object();
    for (const auto& phase : PHASE_ORDER) phases[phase] = {{"phase", phase}, {"status", "pending"}, {"pct", 0.0}};
    nlohmann::json extra_phases = nlohmann::json::object();
    nlohmann::json progress_map = nlohmann::json::object();
    nlohmann::json events_tail = nlohmann::json::array();
    std::string run_status = "unknown";
    std::string current_phase;
    std::string resume_from_phase;

    for (const auto& ev : iter_jsonl(*event_file)) {
        events_tail.push_back(ev);
        if (events_tail.size() > 200) events_tail.erase(events_tail.begin());

        std::string event_type = ev.value("type", std::string());
        std::string phase_name = phase_name_from_event(ev);
        if (!phase_name.empty()) {
            nlohmann::json* phase_state = nullptr;
            if (phases.contains(phase_name)) phase_state = &phases[phase_name];
            else {
                if (!extra_phases.contains(phase_name)) extra_phases[phase_name] = {{"phase", phase_name}, {"status", "pending"}, {"pct", 0.0}};
                phase_state = &extra_phases[phase_name];
            }

            if (event_type == "phase_start") {
                if ((*phase_state).value("status", std::string()) != "running" ||
                    (*phase_state).value("pct", 0.0) >= 1.0) {
                    (*phase_state)["pct"] = 0.0;
                }
                (*phase_state)["status"] = "running";
                current_phase = phase_name;
                if (run_status == "unknown" || run_status == "pending") run_status = "running";
            } else if (event_type == "phase_progress") {
                double progress = ev.contains("progress") ? clamp_progress(ev["progress"]) : -1.0;
                if (progress >= 0.0) {
                    double current_pct = (*phase_state).value("pct", 0.0);
                    (*phase_state)["pct"] = std::max(current_pct, progress);
                    progress_map[phase_name] = (*phase_state)["pct"];
                }
                (*phase_state)["status"] = "running";
                current_phase = phase_name;
                if (run_status == "unknown" || run_status == "pending") run_status = "running";
            } else if (event_type == "phase_end") {
                std::string raw = ev.value("status", std::string("unknown"));
                std::transform(raw.begin(), raw.end(), raw.begin(), ::tolower);
                (*phase_state)["status"] = raw;
                if (raw == "ok" || raw == "skipped") (*phase_state)["pct"] = 1.0;
                if (current_phase == phase_name && (raw == "ok" || raw == "skipped" || raw == "error" || raw == "aborted")) {
                    current_phase.clear();
                }
                if (raw == "error" || raw == "aborted") run_status = "failed";
            }
        }

        if (event_type == "resume_start") {
            if (ev.contains("from_phase") && ev["from_phase"].is_string()) {
                resume_from_phase = ev["from_phase"].get<std::string>();
            } else if (ev.contains("payload") && ev["payload"].is_object() && ev["payload"].contains("from_phase") && ev["payload"]["from_phase"].is_string()) {
                resume_from_phase = ev["payload"]["from_phase"].get<std::string>();
            }
            std::transform(resume_from_phase.begin(), resume_from_phase.end(), resume_from_phase.begin(), ::toupper);
            if (!resume_from_phase.empty()) {
                current_phase = resume_from_phase;
                if (run_status == "unknown" || run_status == "pending" || run_status == "completed") run_status = "running";
                auto it = std::find(PHASE_ORDER.begin(), PHASE_ORDER.end(), resume_from_phase);
                if (it != PHASE_ORDER.end()) {
                    for (auto pit = PHASE_ORDER.begin(); pit != it; ++pit) {
                        if (phases.contains(*pit) && phases[*pit].value("status", std::string()) == "pending") {
                            phases[*pit]["status"] = "ok";
                            phases[*pit]["pct"] = 1.0;
                        }
                    }
                    for (auto pit = it; pit != PHASE_ORDER.end(); ++pit) {
                        if (phases.contains(*pit)) {
                            phases[*pit]["status"] = (*pit == resume_from_phase) ? "running" : "pending";
                            phases[*pit]["pct"] = 0.0;
                        }
                    }
                }
            }
        }

        if (event_type == "resume_end") {
            bool success = ev.value("success", false);
            if (!success && ev.contains("payload") && ev["payload"].is_object()) success = ev["payload"].value("success", false);
            run_status = success ? "completed" : "failed";
            if (success) {
                if (!resume_from_phase.empty()) {
                    nlohmann::json* phase_state = nullptr;
                    if (phases.contains(resume_from_phase)) phase_state = &phases[resume_from_phase];
                    else if (extra_phases.contains(resume_from_phase)) phase_state = &extra_phases[resume_from_phase];
                    if (phase_state) {
                        const std::string status_text = (*phase_state).value("status", std::string());
                        if (status_text == "running" || status_text == "pending") {
                            (*phase_state)["status"] = "ok";
                            (*phase_state)["pct"] = 1.0;
                        }
                    }
                }
                if (current_phase == resume_from_phase) current_phase.clear();
            }
        }

        if (event_type == "run_end") {
            run_status = ev.value("success", false) ? "completed" : "failed";
        }
    }

    if (run_status == "unknown") {
        if (!current_phase.empty()) run_status = "running";
        else {
            for (auto it = phases.begin(); it != phases.end(); ++it) {
                const std::string phase_status = it.value().value("status", std::string());
                if (phase_status == "ok" || phase_status == "skipped") {
                    run_status = "running";
                    break;
                }
            }
        }
    }

    nlohmann::json phase_list = nlohmann::json::array();
    for (const auto& phase : PHASE_ORDER) phase_list.push_back(phases[phase]);
    for (auto it = extra_phases.begin(); it != extra_phases.end(); ++it) phase_list.push_back(it.value());
    double progress = overall_progress(phase_list, current_phase, progress_map);
    if (run_status == "completed") progress = 1.0;

    result["status"] = run_status;
    result["current_phase"] = current_phase.empty() ? nlohmann::json(nullptr) : nlohmann::json(current_phase);
    result["progress"] = std::round(progress * 10000.0) / 10000.0;
    result["phases"] = phase_list;
    result["events"] = events_tail;
    return result;
}

std::vector<nlohmann::json> discover_runs(const fs::path& runs_dir, int limit) {
    std::vector<nlohmann::json> result;
    if (!fs::exists(runs_dir)) return result;

    std::map<fs::path, fs::path> run_paths;
    for (auto& entry : fs::recursive_directory_iterator(runs_dir)) {
        if (!entry.is_regular_file()) continue;
        auto name = entry.path().filename().string();
        if (name != "run_events.jsonl" && name != "events.jsonl") continue;
        fs::path run_dir = (name == "run_events.jsonl" && entry.path().parent_path().filename() == "logs")
            ? entry.path().parent_path().parent_path()
            : entry.path().parent_path();
        run_paths.emplace(run_dir, entry.path());
    }

    for (const auto& item : run_paths) {
        const auto& run_dir = item.first;
        const auto& event_file = item.second;
        if (!fs::exists(run_dir)) continue;
        std::string run_id = extract_run_id_from_events(event_file);
        if (run_id.empty()) {
            std::error_code ec;
            run_id = fs::relative(run_dir, runs_dir, ec).string();
            if (ec) run_id = run_dir.filename().string();
        }
        auto status = read_run_status(run_dir);
        auto modified_time = fs::last_write_time(event_file);
        result.push_back({
            {"name", run_id.find('/') == std::string::npos ? run_id : run_id.substr(run_id.find_last_of('/') + 1)},
            {"path", run_dir.string()},
            {"run_id", run_id},
            {"modified", iso_utc_from_file_time(modified_time)},
            {"status", status.value("status", "unknown")},
        });
    }
    std::sort(result.begin(), result.end(), [](const nlohmann::json& a, const nlohmann::json& b) {
        return a.value("modified", std::string()) > b.value("modified", std::string());
    });
    if (result.size() > static_cast<size_t>(limit)) result.resize(static_cast<size_t>(limit));
    return result;
}

std::string read_run_logs(const fs::path& run_dir, int tail) {
    auto event_file = find_event_file(run_dir);
    if (!event_file) return "";

    std::ifstream in(*event_file);
    if (!in) return "";

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        lines.push_back(line);
    }

    int start = static_cast<int>(lines.size()) - tail;
    if (start < 0) start = 0;
    std::ostringstream oss;
    for (int i = start; i < static_cast<int>(lines.size()); ++i)
        oss << lines[i] << "\n";
    return oss.str();
}

nlohmann::json list_run_artifacts(const fs::path& run_dir) {
    nlohmann::json items = nlohmann::json::array();
    if (!fs::is_directory(run_dir)) return items;

    static const std::vector<std::string> ARTIFACT_EXTS = {
        ".json", ".jsonl", ".html", ".yaml", ".yml", ".png", ".fits", ".log"
    };

    std::function<void(const fs::path&, const std::string&)> scan =
        [&](const fs::path& dir, const std::string& prefix) {
            for (auto& entry : fs::directory_iterator(dir)) {
                std::string name = entry.path().filename().string();
                std::string rel = prefix.empty() ? name : prefix + "/" + name;
                if (entry.is_directory()) {
                    scan(entry.path(), rel);
                } else if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    for (auto& e : ARTIFACT_EXTS) {
                        if (ext == e) {
                            int64_t size_bytes = static_cast<int64_t>(fs::file_size(entry.path()));
                            items.push_back({
                                {"path",          rel},
                                {"relative_path", rel},
                                {"name",          name},
                                {"filename",      name},
                                {"size",          size_bytes},
                                {"size_bytes",    size_bytes},
                            });
                            break;
                        }
                    }
                }
            }
        };
    scan(run_dir, "");
    return items;
}
