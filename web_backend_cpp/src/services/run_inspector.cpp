#include "services/run_inspector.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <functional>
#include <iomanip>
#include <map>
#include <optional>
#include <unordered_set>
#include <yaml-cpp/yaml.h>
#ifndef _WIN32
#include <unistd.h>
#endif

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

bool visit_jsonl(const fs::path& path,
                 const std::function<bool(const nlohmann::json&)>& visitor) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto parsed = nlohmann::json::parse(line, nullptr, false);
        if (parsed.is_discarded() || !parsed.is_object()) continue;
        if (!visitor(parsed)) break;
    }
    return true;
}

std::string phase_name_from_event(const nlohmann::json& ev) {
    if (ev.contains("phase_name") && ev["phase_name"].is_string()) return ev["phase_name"].get<std::string>();
    if (ev.contains("phase")) {
        if (ev["phase"].is_string()) return ev["phase"].get<std::string>();
        if (ev["phase"].is_number_integer()) return std::to_string(ev["phase"].get<int>());
    }
    return "";
}

int phase_order_index(const std::string& phase_name) {
    auto it = std::find(PHASE_ORDER.begin(), PHASE_ORDER.end(), phase_name);
    if (it == PHASE_ORDER.end()) return -1;
    return static_cast<int>(std::distance(PHASE_ORDER.begin(), it));
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
    std::string detected = "UNKNOWN";
    visit_jsonl(*event_file, [&](const nlohmann::json& ev) {
        if (ev.contains("color_mode") && ev["color_mode"].is_string()) {
            std::string color_mode = ev["color_mode"].get<std::string>();
            if (!color_mode.empty()) {
                std::transform(color_mode.begin(), color_mode.end(), color_mode.begin(), ::toupper);
                detected = color_mode;
                return false;
            }
        }
        if (ev.contains("payload") && ev["payload"].is_object() && ev["payload"].contains("color_mode") && ev["payload"]["color_mode"].is_string()) {
            std::string color_mode = ev["payload"]["color_mode"].get<std::string>();
            if (!color_mode.empty()) {
                std::transform(color_mode.begin(), color_mode.end(), color_mode.begin(), ::toupper);
                detected = color_mode;
                return false;
            }
        }
        return true;
    });
    return detected;
}

std::string extract_run_id_from_events(const fs::path& event_file) {
    std::string run_id;
    visit_jsonl(event_file, [&](const nlohmann::json& ev) {
        if (ev.contains("run_id") && ev["run_id"].is_string()) {
            run_id = ev["run_id"].get<std::string>();
            if (!run_id.empty()) return false;
        }
        return true;
    });
    return run_id;
}

std::string normalize_phase_name(std::string phase_name) {
    std::transform(phase_name.begin(), phase_name.end(), phase_name.begin(), ::toupper);
    return phase_name;
}

bool is_run_tracking_job_type(const std::string& type) {
    return type.rfind("run", 0) == 0 || type == "resume";
}

std::string normalize_run_id_path(std::string run_id) {
    std::replace(run_id.begin(), run_id.end(), '\\', '/');
    while (!run_id.empty() && run_id.front() == '/') run_id.erase(run_id.begin());
    while (!run_id.empty() && run_id.back() == '/') run_id.pop_back();
    return run_id;
}

std::string parent_run_id(const std::string& run_id) {
    const std::string normalized = normalize_run_id_path(run_id);
    const auto slash = normalized.find_last_of('/');
    if (slash == std::string::npos) return "";
    return normalized.substr(0, slash);
}

bool run_id_matches_queue_item(const std::string& run_id, const std::string& item_run_id) {
    const std::string normalized_run_id = normalize_run_id_path(run_id);
    const std::string normalized_item_run_id = normalize_run_id_path(item_run_id);
    if (normalized_run_id.empty() || normalized_item_run_id.empty()) return false;
    if (normalized_run_id == normalized_item_run_id) return true;
    return normalized_run_id == parent_run_id(normalized_item_run_id);
}

void ensure_phase_array(nlohmann::json& status) {
    if (status.contains("phases") && status["phases"].is_array()) return;
    status["phases"] = nlohmann::json::array();
    for (const auto& phase : PHASE_ORDER) {
        status["phases"].push_back(nlohmann::json{{"phase", phase}, {"status", "pending"}, {"pct", 0.0}});
    }
}

void apply_resume_job_overlay(nlohmann::json& status, const Job& job) {
    if (!job.data.is_object()) return;

    std::string resume_phase = normalize_phase_name(job.data.value("from_phase", std::string()));
    if (resume_phase.empty()) return;

    ensure_phase_array(status);
    status["current_phase"] = resume_phase;

    const int resume_idx = phase_order_index(resume_phase);
    for (auto& phase_state : status["phases"]) {
        if (!phase_state.is_object()) continue;
        const std::string phase_name = normalize_phase_name(phase_state.value("phase", std::string()));
        const int phase_idx = phase_order_index(phase_name);
        if (phase_idx < 0 || resume_idx < 0) continue;

        if (phase_idx < resume_idx) {
            const std::string phase_status = phase_state.value("status", std::string("pending"));
            if (phase_status == "pending") {
                phase_state["status"] = "ok";
                phase_state["pct"] = 1.0;
            }
            continue;
        }

        if (phase_idx == resume_idx) {
            phase_state["status"] = "running";
            phase_state["pct"] = 0.0;
            continue;
        }

        phase_state["status"] = "pending";
        phase_state["pct"] = 0.0;
    }
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

void mark_active_phase_aborted(nlohmann::json& status) {
    std::string current_phase;
    if (status.contains("current_phase") && status["current_phase"].is_string()) {
        current_phase = status["current_phase"].get<std::string>();
    }
    if (status.contains("phases") && status["phases"].is_array()) {
        for (auto& item : status["phases"]) {
            if (!item.is_object()) continue;
            const std::string phase_name = item.value("phase", std::string());
            const std::string phase_status = item.value("status", std::string());
            if ((!current_phase.empty() && phase_name == current_phase) || phase_status == "running") {
                item["status"] = "aborted";
            }
        }
    }
    status["status"] = "aborted";
    status["current_phase"] = nullptr;
    status["stale_incomplete"] = true;
    status["stale_reason"] = "no_live_job_or_process";
}

}

bool queue_contains_run_id(const nlohmann::json& queue, const std::string& run_id) {
    if (!queue.is_array() || run_id.empty()) return false;
    for (const auto& item : queue) {
        if (!item.is_object()) continue;
        if (!item.contains("run_id") || !item["run_id"].is_string()) continue;
        if (run_id_matches_queue_item(run_id, item["run_id"].get<std::string>())) return true;
    }
    return false;
}

bool job_references_run_id(const Job& job, const std::string& run_id) {
    if (run_id.empty()) return false;
    const std::string job_run_id = job.data.is_object()
        ? job.data.value("run_id", job.run_id)
        : job.run_id;
    if (run_id_matches_queue_item(run_id, job_run_id)) return true;
    if (!job.data.is_object()) return false;
    return queue_contains_run_id(job.data.value("queue", nlohmann::json::array()), run_id);
}

std::optional<Job> latest_run_job(const InMemoryJobStore& store, const std::string& run_id, int limit) {
    if (run_id.empty()) return std::nullopt;
    const std::string normalized_run_id = normalize_run_id_path(run_id);
    std::optional<Job> active_queue_job;
    std::optional<Job> exact_match;
    std::optional<Job> fallback_match;
    for (const auto& job : store.list(limit)) {
        if (!is_run_tracking_job_type(job.type)) continue;
        if (!job_references_run_id(job, run_id)) continue;

        const std::string job_run_id = normalize_run_id_path(
            job.data.is_object() ? job.data.value("run_id", job.run_id) : job.run_id);
        const bool is_exact_match = !normalized_run_id.empty() && job_run_id == normalized_run_id;
        const bool is_active_queue_job =
            job.type == "run_queue" &&
            (job.state == JobState::pending || job.state == JobState::running);

        if (is_exact_match && job.type != "run_queue" &&
            (job.state == JobState::pending || job.state == JobState::running)) {
            return job;
        }
        if (is_active_queue_job && !active_queue_job.has_value()) {
            active_queue_job = job;
            continue;
        }
        if (is_exact_match && !exact_match.has_value()) {
            exact_match = job;
            continue;
        }
        if (!fallback_match.has_value()) fallback_match = job;
    }
    if (active_queue_job.has_value()) return active_queue_job;
    if (exact_match.has_value()) return exact_match;
    return fallback_match;
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
        if (job->type == "resume") apply_resume_job_overlay(status, *job);
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

bool has_live_runner_process(const std::string& runner_exe,
                             const std::string& run_id,
                             const std::string& run_dir) {
#ifdef _WIN32
    (void)runner_exe;
    (void)run_id;
    (void)run_dir;
    return false;
#else
    const std::string runner_name = fs::path(runner_exe).filename().string();
    if (run_id.empty() || runner_name.empty() || !fs::exists("/proc")) return false;
    const int self_pid = static_cast<int>(::getpid());
    for (const auto& entry : fs::directory_iterator("/proc")) {
        if (!entry.is_directory()) continue;
        const std::string pid_text = entry.path().filename().string();
        if (pid_text.empty() || !std::all_of(pid_text.begin(), pid_text.end(), ::isdigit)) continue;
        int pid = 0;
        try {
            pid = std::stoi(pid_text);
        } catch (...) {
            continue;
        }
        if (pid == self_pid) continue;

        std::ifstream cmdline(entry.path() / "cmdline", std::ios::binary);
        if (!cmdline) continue;
        std::string raw((std::istreambuf_iterator<char>(cmdline)), std::istreambuf_iterator<char>());
        if (raw.empty()) continue;

        std::vector<std::string> argv;
        size_t start = 0;
        while (start < raw.size()) {
            size_t end = raw.find('\0', start);
            if (end == std::string::npos) end = raw.size();
            if (end > start) argv.push_back(raw.substr(start, end - start));
            start = end + 1;
        }
        if (argv.empty()) continue;

        const std::string exe_name = fs::path(argv.front()).filename().string();
        if (exe_name != runner_name && exe_name.find("tile_compile_runner") == std::string::npos) continue;
        const bool is_runner = std::any_of(argv.begin() + 1, argv.end(), [](const std::string& arg) {
            return arg == "run" || arg == "resume";
        });
        if (!is_runner) continue;

        std::string joined;
        for (const auto& part : argv) {
            if (!joined.empty()) joined += ' ';
            joined += part;
        }
        if (joined.find(run_id) == std::string::npos &&
            (run_dir.empty() || joined.find(run_dir) == std::string::npos)) {
            continue;
        }
        return true;
    }
    return false;
#endif
}

void apply_runtime_liveness_to_run_status(nlohmann::json& status,
                                          const std::optional<Job>& job,
                                          const std::string& runner_exe,
                                          const std::string& run_id,
                                          const std::string& run_dir) {
    const std::string state = status.value("status", std::string());
    if (state != "running" && state != "pending") return;
    if (job.has_value()) {
        const std::string job_state = job_state_str(job->state);
        if (job_state == "running" || job_state == "pending") return;
    }
    if (has_live_runner_process(runner_exe, run_id, run_dir)) return;
    mark_active_phase_aborted(status);
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
    std::deque<nlohmann::json> events_tail;
    std::string run_status = "unknown";
    std::string current_phase;
    std::string resume_from_phase;
    bool resume_active = false;

    auto is_resume_prereq_phase = [&](const std::string& phase_name) {
        if (!resume_active || resume_from_phase.empty() || phase_name.empty() || phase_name == resume_from_phase) {
            return false;
        }
        const int phase_idx = phase_order_index(phase_name);
        const int resume_idx = phase_order_index(resume_from_phase);
        return phase_idx >= 0 && resume_idx >= 0 && phase_idx < resume_idx;
    };

    visit_jsonl(*event_file, [&](const nlohmann::json& ev) {
        events_tail.push_back(ev);
        if (events_tail.size() > 200) events_tail.pop_front();
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
                if (!is_resume_prereq_phase(phase_name)) current_phase = phase_name;
                if (run_status == "unknown" || run_status == "pending") run_status = "running";
            } else if (event_type == "phase_progress") {
                double progress = ev.contains("progress") ? clamp_progress(ev["progress"]) : -1.0;
                if (progress >= 0.0) {
                    double current_pct = (*phase_state).value("pct", 0.0);
                    (*phase_state)["pct"] = std::max(current_pct, progress);
                    progress_map[phase_name] = (*phase_state)["pct"];
                }
                (*phase_state)["status"] = "running";
                if (!is_resume_prereq_phase(phase_name)) current_phase = phase_name;
                if (run_status == "unknown" || run_status == "pending") run_status = "running";
            } else if (event_type == "phase_end") {
                const std::string previous_status = (*phase_state).value("status", std::string());
                std::string raw = ev.value("status", std::string("unknown"));
                std::transform(raw.begin(), raw.end(), raw.begin(), ::tolower);
                std::string reason = ev.value("reason", std::string());
                if (reason.empty() && ev.contains("payload") && ev["payload"].is_object()) {
                    reason = ev["payload"].value("reason", std::string());
                }
                std::transform(reason.begin(), reason.end(), reason.begin(), ::tolower);
                if (resume_active && phase_name == "ASTROMETRY" &&
                    resume_from_phase != "ASTROMETRY" && raw == "skipped" &&
                    reason == "existing_wcs") {
                    raw = "ok";
                }
                (*phase_state)["status"] = raw;
                if (raw == "ok" || raw == "skipped") (*phase_state)["pct"] = 1.0;
                if (!is_resume_prereq_phase(phase_name) &&
                    current_phase == phase_name &&
                    (raw == "ok" || raw == "skipped" || raw == "error" || raw == "aborted")) {
                    current_phase.clear();
                }
                if (raw == "error" || raw == "aborted") run_status = "failed";
                if ((phase_name == "PCC" || phase_name == "DONE") &&
                    (raw == "ok" || raw == "skipped") &&
                    !resume_active) {
                    run_status = "completed";
                }
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
                resume_active = true;
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
                        if (status_text == "running" || status_text == "pending" || status_text == "skipped") {
                            (*phase_state)["status"] = "ok";
                            (*phase_state)["pct"] = 1.0;
                        }
                    }
                }
                if (current_phase == resume_from_phase) current_phase.clear();
            }
            resume_active = false;
        }

        if (event_type == "run_end") {
            run_status = ev.value("success", false) ? "completed" : "failed";
        }
        return true;
    });

    if (run_status == "running" && current_phase.empty()) {
        run_status = "unknown";
    }

    if (run_status == "unknown") {
        if (!current_phase.empty()) {
            run_status = "running";
        } else {
            if (phases.contains("PCC")) {
                const std::string pcc_status = phases["PCC"].value("status", std::string());
                if (pcc_status == "ok" || pcc_status == "skipped") {
                    run_status = "completed";
                }
            }
            if (run_status == "unknown" && extra_phases.contains("DONE")) {
                const std::string done_status = extra_phases["DONE"].value("status", std::string());
                if (done_status == "ok" || done_status == "skipped") {
                    run_status = "completed";
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
    result["events"] = nlohmann::json::array();
    for (const auto& ev : events_tail) result["events"].push_back(ev);
    return result;
}

std::vector<nlohmann::json> discover_runs(const fs::path& runs_dir, int limit) {
    std::vector<nlohmann::json> result;
    if (!fs::exists(runs_dir)) return result;
    if (limit <= 0) return result;

    std::unordered_set<std::string> seen_run_dirs;
    for (auto& entry : fs::recursive_directory_iterator(runs_dir)) {
        if (!entry.is_regular_file()) continue;
        auto name = entry.path().filename().string();
        if (name != "run_events.jsonl" && name != "events.jsonl") continue;
        const fs::path event_file = entry.path();
        fs::path run_dir = (name == "run_events.jsonl" && entry.path().parent_path().filename() == "logs")
            ? entry.path().parent_path().parent_path()
            : entry.path().parent_path();
        if (!seen_run_dirs.insert(run_dir.string()).second) continue;
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
    // Sort once after collecting all entries (newest first).
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

    std::deque<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        lines.push_back(line);
        if (lines.size() > static_cast<size_t>(std::max(1, tail))) lines.pop_front();
    }

    std::ostringstream oss;
    for (const auto& item : lines) oss << item << "\n";
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
