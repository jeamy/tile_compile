#include "subprocess_manager.hpp"
#include <algorithm>
#include <sstream>
#include <array>
#include <stdexcept>
#include <thread>
#include <chrono>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <unistd.h>
#  include <sys/wait.h>
#  include <fcntl.h>
#  include <signal.h>
#endif

namespace {

using json = nlohmann::json;
constexpr size_t MAX_JSON_STRING_BYTES = 8 * 1024;
constexpr size_t MAX_JSON_ARRAY_ITEMS = 256;
constexpr size_t MAX_JSON_OBJECT_ITEMS = 256;
constexpr int MAX_JSON_DEPTH = 8;

struct CapturedText {
    std::string text;
    size_t total_bytes{0};
    bool truncated{false};
};

std::string truncate_text(const std::string& raw, size_t max_bytes) {
    if (raw.size() <= max_bytes) return raw;
    if (max_bytes == 0) return {};
    static const std::string suffix = "\n...[truncated]";
    if (max_bytes <= suffix.size()) return raw.substr(0, max_bytes);
    return raw.substr(0, max_bytes - suffix.size()) + suffix;
}

json compact_json_for_job_storage(const json& value,
                                  const BackendGuardLimits& limits,
                                  int depth = 0);

json compact_json_array(const json& value,
                        size_t max_items,
                        const BackendGuardLimits& limits,
                        int depth = 0) {
    json out = json::array();
    if (!value.is_array()) return out;
    const size_t limit = std::min(value.size(), max_items);
    for (size_t i = 0; i < limit; ++i) out.push_back(compact_json_for_job_storage(value[i], limits, depth + 1));
    return out;
}

json compact_json_for_job_storage(const json& value,
                                  const BackendGuardLimits& limits,
                                  int depth) {
    if (depth >= MAX_JSON_DEPTH) return "[truncated depth]";
    if (value.is_string()) return truncate_text(value.get<std::string>(), MAX_JSON_STRING_BYTES);
    if (value.is_array()) return compact_json_array(value, MAX_JSON_ARRAY_ITEMS, limits, depth);
    if (value.is_object()) {
        json out = json::object();
        size_t count = 0;
        for (auto it = value.begin(); it != value.end() && count < MAX_JSON_OBJECT_ITEMS; ++it, ++count) {
            out[it.key()] = compact_json_for_job_storage(it.value(), limits, depth + 1);
        }
        if (value.size() > MAX_JSON_OBJECT_ITEMS) {
            out["_truncated_fields"] = static_cast<long long>(value.size() - MAX_JSON_OBJECT_ITEMS);
        }
        return out;
    }
    return value;
}

json compact_scan_per_dir_result(const json& raw, const BackendGuardLimits& limits) {
    json out = compact_json_for_job_storage(raw, limits);
    if (!raw.is_object()) return out;

    if (raw.contains("errors") && raw["errors"].is_array()) {
        out["errors"] = compact_json_array(raw["errors"], limits.scan_messages_preview, limits);
        out["errors_total"] = raw["errors"].size();
        out["errors_truncated"] = raw["errors"].size() > limits.scan_messages_preview;
    }
    if (raw.contains("warnings") && raw["warnings"].is_array()) {
        out["warnings"] = compact_json_array(raw["warnings"], limits.scan_messages_preview, limits);
        out["warnings_total"] = raw["warnings"].size();
        out["warnings_truncated"] = raw["warnings"].size() > limits.scan_messages_preview;
    }
    if (raw.contains("frames") && raw["frames"].is_array()) {
        out["frames"] = compact_json_array(raw["frames"], limits.scan_per_dir_frames_preview, limits);
        out["frames_total"] = raw["frames"].size();
        out["frames_truncated"] = raw["frames"].size() > limits.scan_per_dir_frames_preview;
    } else if (!out.contains("frames")) {
        out["frames"] = json::array();
        out["frames_total"] = 0;
        out["frames_truncated"] = false;
    }
    if (raw.contains("color_mode_candidates") && raw["color_mode_candidates"].is_array()) {
        out["color_mode_candidates"] = compact_json_array(raw["color_mode_candidates"], limits.scan_color_candidates_preview, limits);
        out["color_mode_candidates_total"] = raw["color_mode_candidates"].size();
        out["color_mode_candidates_truncated"] = raw["color_mode_candidates"].size() > limits.scan_color_candidates_preview;
    }
    return out;
}

json compact_scan_job_result(const json& raw, const BackendGuardLimits& limits) {
    json out = compact_json_for_job_storage(raw, limits);
    if (!raw.is_object()) return out;

    if (raw.contains("errors") && raw["errors"].is_array()) {
        out["errors"] = compact_json_array(raw["errors"], limits.scan_messages_preview, limits);
        out["errors_total"] = raw["errors"].size();
        out["errors_truncated"] = raw["errors"].size() > limits.scan_messages_preview;
    }
    if (raw.contains("warnings") && raw["warnings"].is_array()) {
        out["warnings"] = compact_json_array(raw["warnings"], limits.scan_messages_preview, limits);
        out["warnings_total"] = raw["warnings"].size();
        out["warnings_truncated"] = raw["warnings"].size() > limits.scan_messages_preview;
    }
    if (raw.contains("frames") && raw["frames"].is_array()) {
        out["frames"] = compact_json_array(raw["frames"], limits.scan_frames_preview, limits);
        out["frames_total"] = raw["frames"].size();
        out["frames_truncated"] = raw["frames"].size() > limits.scan_frames_preview;
    } else if (!out.contains("frames")) {
        out["frames"] = json::array();
        out["frames_total"] = 0;
        out["frames_truncated"] = false;
    }
    if (raw.contains("color_mode_candidates") && raw["color_mode_candidates"].is_array()) {
        out["color_mode_candidates"] = compact_json_array(raw["color_mode_candidates"], limits.scan_color_candidates_preview, limits);
        out["color_mode_candidates_total"] = raw["color_mode_candidates"].size();
        out["color_mode_candidates_truncated"] = raw["color_mode_candidates"].size() > limits.scan_color_candidates_preview;
    }
    if (raw.contains("per_dir_results") && raw["per_dir_results"].is_array()) {
        json per_dir = json::array();
        const size_t limit = std::min(raw["per_dir_results"].size(), limits.scan_per_dir_results_preview);
        for (size_t i = 0; i < limit; ++i) per_dir.push_back(compact_scan_per_dir_result(raw["per_dir_results"][i], limits));
        out["per_dir_results"] = std::move(per_dir);
        out["per_dir_results_total"] = raw["per_dir_results"].size();
        out["per_dir_results_truncated"] = raw["per_dir_results"].size() > limits.scan_per_dir_results_preview;
    }
    return out;
}

void store_process_output(json& data,
                          const char* key,
                          const std::string& text,
                          size_t total_bytes,
                          bool truncated,
                          const BackendGuardLimits& limits) {
    data[key] = truncate_text(text, limits.job_stdio_store_bytes);
    data[std::string(key) + "_bytes"] = total_bytes;
    data[std::string(key) + "_truncated"] = truncated || text.size() > limits.job_stdio_store_bytes;
}

#ifndef _WIN32
struct SpawnedProcess {
    pid_t pid{-1};
    int stdout_fd{-1};
    int stderr_fd{-1};
};

CapturedText drain_fd(int fd, size_t capture_limit_bytes) {
    CapturedText out;
    if (fd < 0) return out;
    char buf[4096];
    ssize_t n = 0;
    while ((n = read(fd, buf, sizeof(buf))) > 0) {
        out.total_bytes += static_cast<size_t>(n);
        const size_t remaining = out.text.size() < capture_limit_bytes
            ? (capture_limit_bytes - out.text.size())
            : 0;
        const size_t to_copy = std::min(static_cast<size_t>(n), remaining);
        if (to_copy > 0) out.text.append(buf, to_copy);
        if (to_copy < static_cast<size_t>(n) || out.total_bytes > capture_limit_bytes) out.truncated = true;
    }
    close(fd);
    return out;
}

bool spawn_subprocess(const std::vector<std::string>& args,
                      const std::string& cwd,
                      const std::string& stdin_text,
                      SpawnedProcess& proc_out) {
    int pfd_out[2], pfd_err[2], pfd_in[2];
    if (pipe(pfd_out) || pipe(pfd_err) || pipe(pfd_in)) return false;

    pid_t pid = fork();
    if (pid < 0) {
        // Close all pipe ends to avoid FD leak.
        close(pfd_out[0]); close(pfd_out[1]);
        close(pfd_err[0]); close(pfd_err[1]);
        close(pfd_in[0]);  close(pfd_in[1]);
        return false;
    }

    if (pid == 0) {
        close(pfd_out[0]); close(pfd_err[0]); close(pfd_in[1]);
        dup2(pfd_out[1], STDOUT_FILENO);
        dup2(pfd_err[1], STDERR_FILENO);
        dup2(pfd_in[0], STDIN_FILENO);
        close(pfd_out[1]); close(pfd_err[1]); close(pfd_in[0]);
        setpgid(0, 0);
        if (!cwd.empty() && chdir(cwd.c_str()) != 0) _exit(126);

        std::vector<const char*> argv;
        for (auto& a : args) argv.push_back(a.c_str());
        argv.push_back(nullptr);
        execvp(argv[0], const_cast<char* const*>(argv.data()));
        _exit(127);
    }

    setpgid(pid, pid);
    close(pfd_out[1]); close(pfd_err[1]); close(pfd_in[0]);
    if (!stdin_text.empty()) {
        ssize_t total = 0;
        while (total < static_cast<ssize_t>(stdin_text.size())) {
            ssize_t n = write(pfd_in[1], stdin_text.data() + total, stdin_text.size() - static_cast<size_t>(total));
            if (n <= 0) break;
            total += n;
        }
    }
    close(pfd_in[1]);

    proc_out.pid = pid;
    proc_out.stdout_fd = pfd_out[0];
    proc_out.stderr_fd = pfd_err[0];
    return true;
}

int wait_for_process(BackgroundProcess& proc) {
    int status = 0;
    bool term_sent = false;
    int term_wait_cycles = 0;
    // Allow up to ~3 s for graceful shutdown after SIGTERM before sending SIGKILL.
    constexpr int SIGKILL_AFTER_CYCLES = 20; // 20 * 150 ms = 3 s
    while (true) {
        pid_t rc = waitpid(static_cast<pid_t>(proc.pid.load()), &status, WNOHANG);
        if (rc == static_cast<pid_t>(proc.pid.load())) return status;
        if (rc < 0) return -1;

        if (proc.cancelled.load()) {
            if (!term_sent) {
                kill(-static_cast<pid_t>(proc.pid.load()), SIGTERM);
                term_sent = true;
                term_wait_cycles = 0;
            } else if (++term_wait_cycles >= SIGKILL_AFTER_CYCLES) {
                kill(-static_cast<pid_t>(proc.pid.load()), SIGKILL);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(term_sent ? 150 : 100));
    }
}
#endif

}  // namespace

SubprocessResult run_subprocess(const std::vector<std::string>& args,
                                const std::string& cwd,
                                const std::string& stdin_text,
                                const BackendGuardLimits* limits_override) {
    SubprocessResult res;
    if (args.empty()) { res.exit_code = -1; return res; }
    const BackendGuardLimits limits = limits_override ? *limits_override : backend_guard_limits_from_env();

#ifdef _WIN32
    std::string cmd;
    for (auto& a : args) { cmd += "\"" + a + "\" "; }

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE hStdoutR, hStdoutW, hStderrR, hStderrW, hStdinR, hStdinW;
    CreatePipe(&hStdoutR, &hStdoutW, &sa, 0);
    CreatePipe(&hStderrR, &hStderrW, &sa, 0);
    CreatePipe(&hStdinR, &hStdinW, &sa, 0);
    SetHandleInformation(hStdoutR, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(hStderrR, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(hStdinW, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.hStdOutput = hStdoutW;
    si.hStdError  = hStderrW;
    si.hStdInput  = hStdinR;
    si.dwFlags |= STARTF_USESTDHANDLES;

    PROCESS_INFORMATION pi{};
    bool ok = CreateProcessA(nullptr, cmd.data(), nullptr, nullptr,
                             TRUE, 0, nullptr,
                             cwd.empty() ? nullptr : cwd.c_str(),
                             &si, &pi);
    CloseHandle(hStdoutW);
    CloseHandle(hStderrW);
    CloseHandle(hStdinR);

    if (!ok) { CloseHandle(hStdinW); res.exit_code = -1; return res; }

    if (!stdin_text.empty()) {
        DWORD written = 0;
        WriteFile(hStdinW, stdin_text.data(), static_cast<DWORD>(stdin_text.size()), &written, nullptr);
        // Ignore partial write; child will receive what was written before pipe closes.
    }
    CloseHandle(hStdinW);

    auto read_pipe = [capture_limit = limits.subprocess_capture_bytes](HANDLE h) {
        CapturedText out;
        char buf[4096];
        DWORD n;
        while (ReadFile(h, buf, sizeof(buf), &n, nullptr) && n > 0) {
            out.total_bytes += static_cast<size_t>(n);
            const size_t remaining = out.text.size() < capture_limit
                ? (capture_limit - out.text.size())
                : 0;
            const size_t to_copy = std::min(static_cast<size_t>(n), remaining);
            if (to_copy > 0) out.text.append(buf, to_copy);
            if (to_copy < static_cast<size_t>(n) || out.total_bytes > capture_limit) out.truncated = true;
        }
        return out;
    };
    CapturedText stdout_capture = read_pipe(hStdoutR);
    CapturedText stderr_capture = read_pipe(hStderrR);
    res.stdout_str = std::move(stdout_capture.text);
    res.stderr_str = std::move(stderr_capture.text);
    res.stdout_bytes = stdout_capture.total_bytes;
    res.stderr_bytes = stderr_capture.total_bytes;
    res.stdout_truncated = stdout_capture.truncated;
    res.stderr_truncated = stderr_capture.truncated;
    CloseHandle(hStdoutR);
    CloseHandle(hStderrR);

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD ec;
    GetExitCodeProcess(pi.hProcess, &ec);
    res.exit_code = (int)ec;
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

#else
    int pfd_out[2], pfd_err[2], pfd_in[2];
    if (pipe(pfd_out) || pipe(pfd_err) || pipe(pfd_in)) { res.exit_code = -1; return res; }

    pid_t pid = fork();
    if (pid < 0) {
        close(pfd_out[0]); close(pfd_out[1]);
        close(pfd_err[0]); close(pfd_err[1]);
        close(pfd_in[0]);  close(pfd_in[1]);
        res.exit_code = -1;
        return res;
    }

    if (pid == 0) {
        close(pfd_out[0]); close(pfd_err[0]); close(pfd_in[1]);
        dup2(pfd_out[1], STDOUT_FILENO);
        dup2(pfd_err[1], STDERR_FILENO);
        dup2(pfd_in[0], STDIN_FILENO);
        close(pfd_out[1]); close(pfd_err[1]);
        close(pfd_in[0]);

        if (!cwd.empty() && chdir(cwd.c_str()) != 0) _exit(126);

        std::vector<const char*> argv;
        for (auto& a : args) argv.push_back(a.c_str());
        argv.push_back(nullptr);
        execvp(argv[0], const_cast<char* const*>(argv.data()));
        _exit(127);
    }

    close(pfd_out[1]); close(pfd_err[1]); close(pfd_in[0]);

    if (!stdin_text.empty()) {
        ssize_t total = 0;
        while (total < static_cast<ssize_t>(stdin_text.size())) {
            ssize_t n = write(pfd_in[1], stdin_text.data() + total, stdin_text.size() - static_cast<size_t>(total));
            if (n <= 0) break;
            total += n;
        }
    }
    close(pfd_in[1]);

    // Read stdout and stderr concurrently to avoid deadlock when either pipe
    // buffer fills up while the parent is blocked reading the other pipe.
    CapturedText stdout_capture, stderr_capture;
    std::thread stdout_thread([&stdout_capture, fd = pfd_out[0], cap = limits.subprocess_capture_bytes]() {
        stdout_capture = drain_fd(fd, cap);
    });
    std::thread stderr_thread([&stderr_capture, fd = pfd_err[0], cap = limits.subprocess_capture_bytes]() {
        stderr_capture = drain_fd(fd, cap);
    });
    stdout_thread.join();
    stderr_thread.join();

    res.stdout_str = std::move(stdout_capture.text);
    res.stderr_str = std::move(stderr_capture.text);
    res.stdout_bytes = stdout_capture.total_bytes;
    res.stderr_bytes = stderr_capture.total_bytes;
    res.stdout_truncated = stdout_capture.truncated;
    res.stderr_truncated = stderr_capture.truncated;

    int status = 0;
    waitpid(pid, &status, 0);
    res.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif

    return res;
}

std::string SubprocessManager::launch(const std::string& type,
                                      const std::vector<std::string>& args,
                                      const std::string& cwd,
                                      const std::string& run_id,
                                      const nlohmann::json& initial_data,
                                      const std::string& stdin_text) {
    std::string job_id = _store.create(type, run_id);
    _store.update_state(job_id, JobState::running, initial_data);

    auto proc = std::make_shared<BackgroundProcess>();
    proc->job_id = job_id;

    {
        std::lock_guard<std::mutex> lk(_procs_mutex);
        _procs[job_id] = proc;
    }

    proc->thread = std::thread([this, job_id, type, args, cwd, stdin_text, proc]() {
        SubprocessResult res;
#ifdef _WIN32
        res = run_subprocess(args, cwd, stdin_text, &_limits);
#else
        SpawnedProcess spawned;
        if (!spawn_subprocess(args, cwd, stdin_text, spawned)) {
            res.exit_code = -1;
            res.stderr_str = "failed to spawn subprocess";
        } else {
            proc->pid.store(static_cast<int>(spawned.pid));
            _store.set_pid(job_id, static_cast<int>(spawned.pid));

            CapturedText stdout_capture;
            CapturedText stderr_capture;
            std::thread stdout_thread([&stdout_capture, fd = spawned.stdout_fd, capture_limit = _limits.subprocess_capture_bytes]() {
                stdout_capture = drain_fd(fd, capture_limit);
            });
            std::thread stderr_thread([&stderr_capture, fd = spawned.stderr_fd, capture_limit = _limits.subprocess_capture_bytes]() {
                stderr_capture = drain_fd(fd, capture_limit);
            });

            int status = wait_for_process(*proc);
            stdout_thread.join();
            stderr_thread.join();
            res.stdout_str = std::move(stdout_capture.text);
            res.stderr_str = std::move(stderr_capture.text);
            res.stdout_bytes = stdout_capture.total_bytes;
            res.stderr_bytes = stderr_capture.total_bytes;
            res.stdout_truncated = stdout_capture.truncated;
            res.stderr_truncated = stderr_capture.truncated;
            if (status >= 0 && WIFEXITED(status)) res.exit_code = WEXITSTATUS(status);
            else if (status >= 0 && WIFSIGNALED(status)) res.exit_code = 128 + WTERMSIG(status);
            else res.exit_code = -1;
        }
#endif
        nlohmann::json data = nlohmann::json::object();
        if (auto snapshot = _store.get(job_id); snapshot.has_value() && snapshot->data.is_object()) {
            data = snapshot->data;
        }
        store_process_output(data, "stdout", res.stdout_str, res.stdout_bytes, res.stdout_truncated, _limits);
        store_process_output(data, "stderr", res.stderr_str, res.stderr_bytes, res.stderr_truncated, _limits);
        data["exit_code"] = res.exit_code;
        auto parsed = nlohmann::json::parse(res.stdout_str, nullptr, false);
        if (!parsed.is_discarded()) {
            const json compact = (type == "scan")
                ? compact_scan_job_result(parsed, _limits)
                : compact_json_for_job_storage(parsed, _limits);
            data["result"] = compact;
            if (compact.is_object()) {
                for (auto it = compact.begin(); it != compact.end(); ++it) {
                    if (!data.contains(it.key()) || data[it.key()].is_null()) {
                        data[it.key()] = it.value();
                    }
                }
            }
        }
        if (proc->cancelled.load()) {
            _store.update_state(job_id, JobState::cancelled, data);
        } else if (res.exit_code == 0) {
            _store.update_state(job_id, JobState::ok, data);
        } else {
            _store.update_state(job_id, JobState::error, data,
                                res.stderr_str.empty() ? "exit " + std::to_string(res.exit_code)
                                                        : res.stderr_str.substr(0, 256));
        }
        _store.set_pid(job_id, std::nullopt);
        std::lock_guard<std::mutex> lk(_procs_mutex);
        _procs.erase(job_id);
    });
    proc->thread.detach();
    return job_id;
}

bool SubprocessManager::cancel(const std::string& job_id) {
    std::lock_guard<std::mutex> lk(_procs_mutex);
    auto it = _procs.find(job_id);
    if (it == _procs.end()) return _store.cancel(job_id);
    it->second->cancelled.store(true);
#ifndef _WIN32
    if (it->second->pid.load() > 0) {
        kill(-static_cast<pid_t>(it->second->pid.load()), SIGTERM);
    }
#endif
    _store.cancel(job_id);
    return true;
}

void SubprocessManager::cancel_by_run(const std::string& run_id) {
    auto jobs = _store.list(500);
    for (auto& j : jobs) {
        if (j.run_id == run_id &&
            (j.state == JobState::running || j.state == JobState::pending)) {
            cancel(j.job_id);
        }
    }
}
