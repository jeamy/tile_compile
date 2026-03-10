#include "routes/runs_routes.hpp"
#include "services/report_generator.hpp"
#include "services/run_inspector.hpp"
#include "subprocess_manager.hpp"
#include "services/config_revisions.hpp"
#include "services/scan_summary.hpp"
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <regex>
#include <thread>
#include <chrono>
#include <cstdlib>

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

static std::string sanitize_run_id(std::string value) {
    for (char& ch : value) {
        bool ok = (ch >= 'a' && ch <= 'z') ||
                  (ch >= 'A' && ch <= 'Z') ||
                  (ch >= '0' && ch <= '9') ||
                  ch == '.' || ch == '_' || ch == '-';
        if (!ok) ch = '_';
    }
    while (!value.empty() && value.front() == '_') value.erase(value.begin());
    while (!value.empty() && value.back() == '_') value.pop_back();
    if (value.empty()) value = "run";
    return value;
}

static std::string wildcard_to_regex(const std::string& pattern) {
    std::string out;
    out.reserve(pattern.size() * 2 + 4);
    out += '^';
    for (char ch : pattern) {
        switch (ch) {
            case '*': out += ".*"; break;
            case '?': out += '.'; break;
            case '.': case '+': case '(': case ')': case '[': case ']':
            case '{': case '}': case '^': case '$': case '|': case '\\':
                out += '\\';
                out += ch;
                break;
            default:
                out += ch;
                break;
        }
    }
    out += '$';
    return out;
}

static bool wildcard_match(const std::string& pattern, const std::string& value) {
    if (pattern.empty()) return true;
    try {
        return std::regex_match(value, std::regex(wildcard_to_regex(pattern), std::regex::icase));
    } catch (...) {
        return value == pattern;
    }
}

static std::string read_file_str(const fs::path& path) {
    std::ifstream in(path);
    if (!in) return "";
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

static std::optional<fs::path> resolve_artifact_path(const fs::path& run_dir, const std::string& raw_path) {
    const std::string trimmed = raw_path;
    if (trimmed.empty()) return std::nullopt;

    std::error_code ec;
    const fs::path run_dir_resolved = fs::weakly_canonical(run_dir, ec);
    if (ec) return std::nullopt;

    fs::path candidate = fs::path(trimmed);
    if (!candidate.is_absolute()) candidate = run_dir_resolved / candidate;
    candidate = fs::weakly_canonical(candidate, ec);
    if (ec) return std::nullopt;

    if (candidate != run_dir_resolved) {
        const std::string run_prefix = run_dir_resolved.string() + fs::path::preferred_separator;
        const std::string candidate_str = candidate.string();
        if (candidate_str.rfind(run_prefix, 0) != 0) return std::nullopt;
    }
    return candidate;
}

static std::string apply_color_mode_to_yaml(const std::string& base_yaml, const std::string& color_mode) {
    if (base_yaml.empty() || color_mode.empty()) return base_yaml;
    try {
        YAML::Node root = YAML::Load(base_yaml);
        if (!root["data"] || !root["data"].IsMap()) root["data"] = YAML::Node(YAML::NodeType::Map);
        root["data"]["color_mode"] = color_mode;
        std::ostringstream oss;
        oss << root;
        return oss.str();
    } catch (...) {
        return base_yaml;
    }
}

static std::string effective_config_yaml(const std::shared_ptr<AppState>& state,
                                         const std::string& config_yaml,
                                         const std::string& color_mode) {
    std::string yaml = config_yaml.empty() ? read_file_str(state->runtime.default_config_path) : config_yaml;
    return apply_color_mode_to_yaml(yaml, color_mode);
}

static fs::path materialize_queue_input(const fs::path& input_dir,
                                        const std::string& pattern,
                                        const fs::path& staging_root,
                                        int item_index) {
    if (pattern.empty()) return input_dir;
    fs::path staging_dir = staging_root / ("item_" + std::to_string(item_index + 1));
    std::error_code ec;
    fs::remove_all(staging_dir, ec);
    fs::create_directories(staging_dir, ec);
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (!wildcard_match(pattern, name)) continue;
        fs::path target = staging_dir / entry.path().filename();
        std::error_code link_ec;
        fs::create_symlink(entry.path(), target, link_ec);
        if (link_ec) {
            std::error_code copy_ec;
            fs::copy_file(entry.path(), target, fs::copy_options::overwrite_existing, copy_ec);
        }
    }
    return staging_dir;
}

static std::string derive_queue_run_id(const std::string& base_run_id,
                                       const nlohmann::json& item,
                                       int index) {
    std::string explicit_run_id = item.value("run_id", "");
    if (!explicit_run_id.empty()) return sanitize_run_id(explicit_run_id);
    std::string filter = item.value("filter", "mono");
    std::ostringstream oss;
    oss << sanitize_run_id(base_run_id) << "_" << sanitize_run_id(filter) << "_" << (index + 1);
    return oss.str();
}

static nlohmann::json collect_queue_items(const nlohmann::json& body, const std::string& base_run_id) {
    nlohmann::json queue = nlohmann::json::array();
    if (!body.contains("queue") || !body["queue"].is_array()) return queue;
    int index = 0;
    for (const auto& raw : body["queue"]) {
        if (!raw.is_object()) continue;
        std::string input_dir = raw.value("input_dir", raw.value("input_path", ""));
        if (input_dir.empty()) continue;
        nlohmann::json item = {
            {"filter", raw.value("filter", "")},
            {"input_dir", input_dir},
            {"pattern", raw.value("pattern", "")},
            {"run_id", derive_queue_run_id(base_run_id, raw, index)},
            {"state", "pending"}
        };
        queue.push_back(item);
        ++index;
    }
    if (!queue.empty()) return queue;
    if (!body.contains("input_dirs") || !body["input_dirs"].is_array()) return queue;
    for (const auto& raw : body["input_dirs"]) {
        std::string input_dir;
        if (raw.is_string()) {
            input_dir = raw.get<std::string>();
        } else if (raw.is_object()) {
            input_dir = raw.value("input_dir", raw.value("input_path", ""));
        }
        if (input_dir.empty()) continue;
        nlohmann::json item = {
            {"filter", raw.is_object() ? raw.value("filter", "") : ""},
            {"input_dir", input_dir},
            {"pattern", raw.is_object() ? raw.value("pattern", "") : ""},
            {"run_id", derive_queue_run_id(base_run_id, raw.is_object() ? raw : nlohmann::json::object(), index)},
            {"state", "pending"}
        };
        queue.push_back(item);
        ++index;
    }
    return queue;
}

static bool queue_cancel_requested(InMemoryJobStore& store, const std::string& job_id) {
    auto job = store.get(job_id);
    return job && job->state == JobState::cancelled;
}

static bool is_terminal_job_state(const std::string& state) {
    return state == "ok" || state == "error" || state == "cancelled";
}

static nlohmann::json queue_job_payload(const nlohmann::json& queue,
                                        int current_index,
                                        const std::string& current_run_id,
                                        const std::string& runs_dir) {
    int done = 0;
    for (const auto& item : queue) {
        std::string state = item.value("state", "pending");
        if (state == "ok") ++done;
    }
    return {
        {"run_id", current_run_id},
        {"runs_dir", runs_dir},
        {"current_index", current_index},
        {"done", done},
        {"total", static_cast<int>(queue.size())},
        {"queue", queue}
    };
}

static std::vector<std::string> runner_run_args(const std::shared_ptr<AppState>& state,
                                                const std::string& config_path,
                                                const std::string& input_dir,
                                                const std::string& runs_dir,
                                                const std::string& run_id) {
    std::vector<std::string> args = {state->runtime.runner_exe, "run"};
    args.push_back("--config"); args.push_back(config_path);
    args.push_back("--input-dir"); args.push_back(input_dir);
    args.push_back("--runs-dir"); args.push_back(runs_dir);
    args.push_back("--project-root"); args.push_back(state->runtime.project_root.string());
    if (!run_id.empty()) { args.push_back("--run-id"); args.push_back(run_id); }
    return args;
}

static std::vector<std::string> collect_input_dirs(const nlohmann::json& body) {
    std::vector<std::string> input_dirs;

    std::string single = body.value("input_dir", body.value("input_path", ""));
    if (!single.empty()) input_dirs.push_back(single);

    if (body.contains("input_dirs") && body["input_dirs"].is_array()) {
        for (const auto& item : body["input_dirs"]) {
            if (item.is_string()) {
                std::string value = item.get<std::string>();
                if (!value.empty()) input_dirs.push_back(value);
            } else if (item.is_object() && item.contains("input_dir") && item["input_dir"].is_string()) {
                std::string value = item["input_dir"].get<std::string>();
                if (!value.empty()) input_dirs.push_back(value);
            } else if (item.is_object() && item.contains("input_path") && item["input_path"].is_string()) {
                std::string value = item["input_path"].get<std::string>();
                if (!value.empty()) input_dirs.push_back(value);
            }
        }
    }

    return input_dirs;
}

void register_runs_routes(CrowApp& app,
                           std::shared_ptr<AppState> state) {
 
    CROW_ROUTE(app, "/api/runs").methods("GET"_method)
    ([state]() {
        auto runs = discover_runs(state->runtime.runs_dir, 100);
        return json_resp({{"items", runs}, {"total", runs.size()}});
    });

    CROW_ROUTE(app, "/api/runs/start").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded()) return err_resp("Invalid JSON");

        auto guardrails = scan_guardrails(state->job_store);
        if (guardrails.value("status", std::string()) == "error") {
            return err_resp("GUARDRAIL_BLOCKED", "run start blocked by guardrails", 409, guardrails);
        }

        auto input_dirs = collect_input_dirs(body);
        std::string runs_dir    = body.value("runs_dir", state->runtime.runs_dir.string());
        std::string run_id      = body.value("run_id", body.value("run_name", ""));
        std::string color_mode  = body.value("color_mode", "");
        std::string config_yaml = body.value("config_yaml", "");
        std::string base_run_id = sanitize_run_id(run_id.empty() ? "run" : run_id);
        auto queue_items = collect_queue_items(body, base_run_id);

        if (input_dirs.empty() && queue_items.empty())
            return err_resp("BAD_REQUEST", "input_dir is required", 400, nlohmann::json::object());

        for (const auto& dir : input_dirs) {
            if (!state->runtime.is_path_allowed(fs::path(dir))) {
                return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + dir, 403, {{"path", dir}});
            }
        }

        if (!state->runtime.is_path_allowed(fs::path(runs_dir))) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + runs_dir, 403, {{"path", runs_dir}});
        }

        std::string prepared_config_yaml = effective_config_yaml(state, config_yaml, color_mode);
        std::string effective_config_path = state->runtime.default_config_path.string();
        if (!prepared_config_yaml.empty()) {
            std::ofstream out(state->runtime.default_config_path);
            if (!out) return err_resp("Cannot write config: " + state->runtime.default_config_path.string(), 500);
            out << prepared_config_yaml;
        }
        std::string revision_id = state->revision_store.add(prepared_config_yaml, "run_start");
        {
            std::lock_guard<std::mutex> lk(state->state_mutex);
            state->active_config_revision_id = revision_id;
        }

        if (!queue_items.empty()) {
            std::string effective_run_id = queue_items.front().value("run_id", base_run_id);
            std::string job_id = state->job_store.create("run_queue", effective_run_id);
            auto queue_payload = queue_job_payload(queue_items, 0, effective_run_id, runs_dir);
            queue_payload["config_revision_id"] = revision_id;
            state->job_store.update_state(job_id, JobState::running, queue_payload);
            {
                std::lock_guard<std::mutex> lk(state->state_mutex);
                state->current_run_id = effective_run_id;
            }
            state->ui_event_store.push("run_started", {{"job_id", job_id}, {"run_id", effective_run_id}});
            std::thread([state, job_id, queue_items, runs_dir, effective_config_path]() mutable {
                fs::path staging_root = fs::path(runs_dir) / ".queue_staging" / job_id;
                std::error_code ec;
                fs::create_directories(staging_root, ec);
                nlohmann::json queue = queue_items;
                for (size_t i = 0; i < queue.size(); ++i) {
                    if (queue_cancel_requested(state->job_store, job_id)) {
                        std::string cancelled_run_id = (i < queue.size() && queue[i].is_object()) ? queue[i].value("run_id", "") : std::string();
                        state->job_store.update_state(job_id, JobState::cancelled,
                            queue_job_payload(queue, static_cast<int>(i), cancelled_run_id, runs_dir));
                        fs::remove_all(staging_root, ec);
                        return;
                    }

                    for (size_t j = 0; j < queue.size(); ++j) {
                        std::string s = queue[j].value("state", "pending");
                        if (j < i && s != "ok") queue[j]["state"] = "ok";
                        else if (j == i) queue[j]["state"] = "running";
                        else if (s != "ok") queue[j]["state"] = "pending";
                    }

                    std::string current_run_id = queue[i].value("run_id", "");
                    {
                        std::lock_guard<std::mutex> lk(state->state_mutex);
                        state->current_run_id = current_run_id;
                    }

                    state->job_store.update_state(job_id, JobState::running,
                        queue_job_payload(queue, static_cast<int>(i), current_run_id, runs_dir));
                    state->job_store.update_progress(job_id, queue.empty() ? 100.0 : (100.0 * i / queue.size()));

                    fs::path input_dir = fs::path(queue[i].value("input_dir", ""));
                    std::string pattern = queue[i].value("pattern", "");
                    fs::path effective_input_dir = materialize_queue_input(input_dir, pattern, staging_root, static_cast<int>(i));

                    auto args = runner_run_args(state, effective_config_path, effective_input_dir.string(), runs_dir, current_run_id);
                    std::string child_job_id = state->subprocess_manager.launch("run", args,
                                                                                  state->runtime.project_root.string(),
                                                                                  current_run_id);
                    while (true) {
                        auto child_job = state->job_store.get(child_job_id);
                        if (!child_job) break;
                        std::string child_state = job_state_str(child_job->state);
                        queue[i]["result"] = child_job->data;
                        queue[i]["job_id"] = child_job_id;
                        queue[i]["state"] = child_state;
                        state->job_store.update_state(job_id, JobState::running,
                            queue_job_payload(queue, static_cast<int>(i), current_run_id, runs_dir));
                        state->job_store.update_progress(job_id, queue.empty() ? 100.0 : (100.0 * i / queue.size()));
                        if (queue_cancel_requested(state->job_store, job_id) && !is_terminal_job_state(child_state)) {
                            state->subprocess_manager.cancel(child_job_id);
                        }
                        if (is_terminal_job_state(child_state)) {
                            if (child_state == "ok") {
                                queue[i]["state"] = "ok";
                                break;
                            }
                            JobState final_state = child_state == "cancelled" ? JobState::cancelled : JobState::error;
                            state->job_store.update_state(job_id, final_state,
                                queue_job_payload(queue, static_cast<int>(i), current_run_id, runs_dir),
                                child_job->error_message);
                            fs::remove_all(staging_root, ec);
                            return;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(250));
                    }
                }

                std::string final_run_id = queue.empty() ? "" : queue.back().value("run_id", "");
                state->job_store.update_progress(job_id, 100.0);
                state->job_store.update_state(job_id, JobState::ok,
                    queue_job_payload(queue, static_cast<int>(queue.size()) - 1, final_run_id, runs_dir));
                fs::remove_all(staging_root, ec);
            }).detach();
            return json_resp({{"job_id", job_id}, {"run_id", effective_run_id}}, 202);
        }

        std::string effective_run_id = sanitize_run_id(run_id.empty() ? "run" : run_id);
        auto args = runner_run_args(state, effective_config_path, input_dirs.front(), runs_dir, effective_run_id);
        std::string job_id = state->subprocess_manager.launch("run", args,
                                                               state->runtime.project_root.string(),
                                                               effective_run_id);
        state->job_store.update_state(job_id, JobState::running, {
            {"input_dir", input_dirs.front()},
            {"runs_dir", runs_dir},
            {"run_id", effective_run_id},
            {"command", args},
            {"config_revision_id", revision_id}
        });
        {
            std::lock_guard<std::mutex> lk(state->state_mutex);
            state->current_run_id = effective_run_id;
        }
        state->ui_event_store.push("run_started", {{"job_id", job_id}, {"run_id", effective_run_id}});
        return json_resp({{"job_id", job_id}, {"run_id", effective_run_id}}, 202);
    });

    CROW_ROUTE(app, "/api/runs/<string>/status").methods("GET"_method)
    ([state](const crow::request&, std::string run_id) {
        try {
            auto run_dir = state->runtime.resolve_run_dir(run_id);
            auto status  = read_run_status(run_dir);
            return json_resp({
                {"run_id", run_id},
                {"run_dir", run_dir.string()},
                {"status", status.value("status", "unknown")},
                {"color_mode", status.value("color_mode", "UNKNOWN")},
                {"current_phase", status.contains("current_phase") ? status["current_phase"] : nlohmann::json(nullptr)},
                {"progress", status.value("progress", 0.0)},
                {"phases", status.value("phases", nlohmann::json::array())},
                {"events", status.value("events", nlohmann::json::array())},
            });
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }
    });

    CROW_ROUTE(app, "/api/runs/<string>/stop").methods("POST"_method)
    ([state](const crow::request& req, std::string run_id) {
        std::string runs_dir = req.url_params.get("runs_dir") ? req.url_params.get("runs_dir") : state->runtime.runs_dir.string();
        fs::path resolved_run_dir;
        bool has_resolved_run_dir = false;
        if (run_id == "pending") {
            resolved_run_dir = fs::path(runs_dir);
            has_resolved_run_dir = true;
        } else {
            try {
                resolved_run_dir = state->runtime.resolve_run_dir(run_id);
                has_resolved_run_dir = true;
            } catch (...) {}
        }

        auto jobs = state->job_store.list(500);
        nlohmann::json cancelled_jobs = nlohmann::json::array();
        nlohmann::json killed_pids = nlohmann::json::array();
        bool cancelled = false;
        for (const auto& job : jobs) {
            if (job.state != JobState::running && job.state != JobState::pending) continue;

            std::string job_run_dir = job.data.is_object() ? job.data.value("run_dir", std::string()) : std::string();
            std::string job_run_id = job.data.is_object() ? job.data.value("run_id", std::string()) : job.run_id;
            std::string job_runs_dir = job.data.is_object() ? job.data.value("runs_dir", std::string()) : std::string();
            bool matches = job_run_id == run_id;

            if (!matches && has_resolved_run_dir) {
                matches = job_run_dir == resolved_run_dir.string();
            }
            if (!matches && run_id == "pending" && has_resolved_run_dir && !job_runs_dir.empty()) {
                matches = resolved_run_dir.string().rfind(job_runs_dir, 0) == 0;
            }
            if (!matches) continue;

            if (job.pid.has_value()) killed_pids.push_back(*job.pid);
            if (state->subprocess_manager.cancel(job.job_id)) {
                cancelled_jobs.push_back(job.job_id);
                cancelled = true;
            }
        }

        if (!cancelled && run_id == "pending") {
            std::optional<Job> single_running;
            for (const auto& job : jobs) {
                if (job.state != JobState::running) continue;
                if (job.type.rfind("run", 0) != 0) continue;
                if (single_running.has_value()) {
                    single_running.reset();
                    break;
                }
                single_running = job;
            }
            if (single_running.has_value()) {
                if (single_running->pid.has_value()) killed_pids.push_back(*single_running->pid);
                if (state->subprocess_manager.cancel(single_running->job_id)) {
                    cancelled_jobs.push_back(single_running->job_id);
                    cancelled = true;
                }
            }
        }
        state->ui_event_store.push("run_stopped", {{"run_id", run_id}});
        return json_resp({{"ok", cancelled}, {"run_id", run_id}, {"cancelled_jobs", cancelled_jobs}, {"killed_pids", killed_pids}});
    });

    CROW_ROUTE(app, "/api/runs/<string>/resume").methods("POST"_method)
    ([state](const crow::request& req, std::string run_id) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded()) return err_resp("Invalid JSON");

        std::string from_phase   = body.value("from_phase",   "");
        std::string run_dir_str  = body.value("run_dir",      "");
        std::string rev_id       = body.value("config_revision_id", "");
        std::string filter_ctx   = body.value("filter_context", "");

        if (from_phase.empty()) return err_resp("RESUME_PHASE_REQUIRED", "from_phase is required for resume", 409, nlohmann::json::object());

        fs::path run_dir;
        try {
            run_dir = run_dir_str.empty() ? state->runtime.resolve_run_dir(run_id) : fs::path(run_dir_str);
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }

        if (!state->runtime.is_path_allowed(run_dir)) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + run_dir.string(), 403, {{"path", run_dir.string()}});
        }

        if (!rev_id.empty()) {
            auto rev = state->revision_store.get(rev_id);
            if (!rev) return err_resp("NOT_FOUND", "revision '" + rev_id + "' not found", 404, nlohmann::json::object());
            std::ofstream out(run_dir / "config.yaml");
            if (!out) return err_resp("Cannot write: " + (run_dir / "config.yaml").string(), 500);
            out << rev->yaml_text;
        }

        std::vector<std::string> args = {state->runtime.runner_exe, "resume"};
        args.push_back("--run-dir"); args.push_back(run_dir.string());
        args.push_back("--from-phase"); args.push_back(from_phase);

        std::string job_id = state->subprocess_manager.launch("resume", args,
                                                               state->runtime.project_root.string(),
                                                               run_id);
        {
            std::lock_guard<std::mutex> lk(state->state_mutex);
            state->current_run_id = run_id;
        }
        state->ui_event_store.push("run_resumed", {{"job_id", job_id}, {"run_id", run_id}, {"filter_context", filter_ctx}});
        return json_resp({{"job_id", job_id}, {"run_id", run_id}}, 202);
    });

    CROW_ROUTE(app, "/api/runs/<string>/logs").methods("GET"_method)
    ([state](const crow::request& req, std::string run_id) {
        int tail = 250;
        if (req.url_params.get("tail"))
            try { tail = std::stoi(req.url_params.get("tail")); } catch (...) {}
        try {
            auto run_dir = state->runtime.resolve_run_dir(run_id);
            std::string text = read_run_logs(run_dir, tail);
            nlohmann::json lines = nlohmann::json::array();
            std::istringstream iss(text);
            std::string line;
            while (std::getline(iss, line)) lines.push_back(line);
            return json_resp({{"lines", lines}, {"cursor", nullptr}});
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }
    });

    CROW_ROUTE(app, "/api/runs/<string>/artifacts").methods("GET"_method)
    ([state](const crow::request&, std::string run_id) {
        try {
            auto run_dir = state->runtime.resolve_run_dir(run_id);
            auto items   = list_run_artifacts(run_dir);
            return json_resp({{"items", items}, {"run_id", run_id}});
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }
    });

    CROW_ROUTE(app, "/api/runs/<string>/artifacts/view").methods("GET"_method)
    ([state](const crow::request& req, std::string run_id) {
        std::string rel_path = req.url_params.get("path") ? req.url_params.get("path") : "";
        if (rel_path.empty()) {
            return err_resp("BAD_REQUEST", "path is required", 400, nlohmann::json::object());
        }
        try {
            auto run_dir = state->runtime.resolve_run_dir(run_id);
            auto full = resolve_artifact_path(run_dir, rel_path);
            if (!full) {
                return err_resp("ARTIFACT_PATH_INVALID", "artifact path must stay inside run directory", 422, nlohmann::json::object());
            }
            if (!fs::exists(*full) || !fs::is_regular_file(*full)) {
                return err_resp("ARTIFACT_NOT_FILE", "artifact path is not a file", 422, nlohmann::json::object());
            }
            std::ifstream f(*full);
            std::string content((std::istreambuf_iterator<char>(f)),
                                  std::istreambuf_iterator<char>());
            std::string ext = full->extension().string();
            if (ext == ".json" || ext == ".jsonl") {
                try {
                    auto j = nlohmann::json::parse(content);
                    return json_resp({{"json", j}, {"is_json", true}, {"text", content}, {"filename", full->filename().string()}, {"path", full->string()}});
                } catch (...) {}
            }
            return json_resp({{"text", content}, {"is_json", false}, {"filename", full->filename().string()}, {"path", full->string()}});
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }
    });

    CROW_ROUTE(app, "/api/runs/<string>/artifacts/raw/<path>").methods("GET"_method)
    ([state](const crow::request&, std::string run_id, std::string rel_path) {
        try {
            auto run_dir = state->runtime.resolve_run_dir(run_id);
            auto full = resolve_artifact_path(run_dir, rel_path);
            if (!full) return err_resp("ARTIFACT_PATH_INVALID", "artifact path must stay inside run directory", 422, nlohmann::json::object());
            if (!fs::exists(*full) || !fs::is_regular_file(*full)) return err_resp("ARTIFACT_NOT_FILE", "artifact path is not a file", 422, nlohmann::json::object());
            std::ifstream f(*full, std::ios::binary);
            std::string body((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
            crow::response res(200, body);
            std::string ext = full->extension().string();
            if      (ext == ".html") res.set_header("Content-Type", "text/html");
            else if (ext == ".json") res.set_header("Content-Type", "application/json");
            else if (ext == ".png")  res.set_header("Content-Type", "image/png");
            else                     res.set_header("Content-Type", "application/octet-stream");
            return res;
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }
    });

    CROW_ROUTE(app, "/api/runs/<string>/delete").methods("POST"_method)
    ([state](const crow::request&, std::string run_id) {
        try {
            auto run_dir = state->runtime.resolve_run_dir(run_id);
            auto jobs = state->job_store.list(500);
            for (const auto& job : jobs) {
                if (job.state != JobState::running) continue;
                std::string job_run_id = job.data.is_object() ? job.data.value("run_id", std::string()) : std::string();
                std::string job_run_dir = job.data.is_object() ? job.data.value("run_dir", std::string()) : std::string();
                if (job_run_id == run_id || job_run_dir == run_dir.string()) {
                    return err_resp("RUN_ACTIVE", "cannot delete active run", 409, nlohmann::json::object());
                }
            }
            fs::remove_all(run_dir);
            {
                std::lock_guard<std::mutex> lk(state->state_mutex);
                if (state->current_run_id == run_id) state->current_run_id = "";
            }
            state->ui_event_store.push("run_deleted", {{"run_id", run_id}});
            return json_resp({{"ok", true}, {"run_id", run_id}});
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }
    });

    CROW_ROUTE(app, "/api/runs/<string>/set-current").methods("POST"_method)
    ([state](const crow::request&, std::string run_id) {
        {
            std::lock_guard<std::mutex> lk(state->state_mutex);
            state->current_run_id = run_id;
        }
        state->ui_event_store.push("current_run_changed", {{"run_id", run_id}});
        return json_resp({{"ok", true}, {"run_id", run_id}});
    });

    CROW_ROUTE(app, "/api/runs/<string>/stats").methods("POST"_method)
    ([state](const crow::request& req, std::string run_id) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        std::string run_dir_str;
        if (!body.is_discarded()) run_dir_str = body.value("run_dir", "");

        fs::path run_dir;
        try {
            run_dir = run_dir_str.empty() ? state->runtime.resolve_run_dir(run_id) : fs::path(run_dir_str);
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }

        if (!state->runtime.is_path_allowed(run_dir)) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + run_dir.string(), 403, {{"path", run_dir.string()}});
        }

        std::string job_id = state->job_store.create("stats", run_id);
        state->job_store.update_state(job_id, JobState::running, {
            {"run_id", run_id},
            {"run_dir", run_dir.string()}
        });
        std::thread([state, job_id, run_id, run_dir]() {
            auto result = generate_run_report(run_dir);
            nlohmann::json data = result;
            data["run_id"] = run_id;
            data["run_dir"] = run_dir.string();
            if (result.value("ok", false)) {
                state->job_store.update_state(job_id, JobState::ok, data);
            } else {
                state->job_store.update_state(job_id, JobState::error, data,
                    result.value("error", std::string("report generation failed")));
            }
        }).detach();
        return json_resp({{"job_id", job_id}, {"state", "running"}}, 202);
    });

    CROW_ROUTE(app, "/api/runs/<string>/stats/status").methods("GET"_method)
    ([state](const crow::request&, std::string run_id) {
        try {
            auto run_dir = state->runtime.resolve_run_dir(run_id);
            fs::path stats_dir  = run_dir / "artifacts";
            fs::path report_path = stats_dir / "report.html";
            std::string state_str = "unknown";
            if (fs::exists(report_path)) state_str = "ok";

            auto jobs = state->job_store.list(200);
            std::string job_id;
            for (auto& j : jobs) {
                if (j.type == "stats" && j.run_id == run_id) {
                    state_str = job_state_str(j.state);
                    job_id = j.job_id;
                    break;
                }
            }
            return json_resp({
                {"state",       state_str},
                {"output_dir",  stats_dir.string()},
                {"report_path", fs::exists(report_path) ? report_path.string() : ""},
                {"job_id",      job_id},
            });
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }
    });

    CROW_ROUTE(app, "/api/runs/<string>/config-revisions/<string>/restore").methods("POST"_method)
    ([state](const crow::request&, std::string run_id, std::string rev_id) {
        auto rev = state->revision_store.get(rev_id);
        if (!rev) return err_resp("Revision not found: " + rev_id, 404);
        {
            std::lock_guard<std::mutex> lk(state->state_mutex);
            std::ofstream out(state->runtime.default_config_path);
            if (out) out << rev->yaml_text;
            state->active_config_revision_id = rev_id;
        }
        return json_resp({{"ok", true}, {"run_id", run_id}, {"active_revision_id", rev_id}});
    });
}
