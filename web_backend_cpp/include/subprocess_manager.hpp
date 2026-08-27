#pragma once
#include "backend_runtime.hpp"
#include "job_store.hpp"
#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <mutex>
#include <optional>
#include <cstddef>

/// @brief Captured result of a synchronous subprocess execution.
/// @details Tracks exit code, bounded stdout/stderr text, original byte counts, and truncation
/// flags so API responses can expose useful diagnostics without unbounded memory growth.
struct SubprocessResult {
    int exit_code{-1};
    std::string stdout_str;
    std::string stderr_str;
    size_t stdout_bytes{0};
    size_t stderr_bytes{0};
    bool stdout_truncated{false};
    bool stderr_truncated{false};
};

/// @brief Runs a subprocess synchronously and captures bounded output.
/// @details Used for short CLI calls from routes; long-running runner work should go through
/// SubprocessManager so it can be tracked and cancelled by job id.
SubprocessResult run_subprocess(const std::vector<std::string>& args,
                                const std::string& cwd = "",
                                const std::string& stdin_text = "",
                                const BackendGuardLimits* limits = nullptr);

/// @brief Runtime handle for one asynchronous subprocess job.
/// @details Combines the worker thread, cancellation flag, and process id used by the manager
/// to terminate or join background operations.
struct BackgroundProcess {
    std::string job_id;
    std::thread thread;
    std::atomic<bool> cancelled{false};
    std::atomic<int> pid{-1};
#ifdef _WIN32
    void* process_handle{nullptr};
#endif
};

/// @brief Launches, tracks, and cancels asynchronous backend subprocess jobs.
/// @details Bridges the in-memory job store with worker threads that execute scanner, runner,
/// downloader, and tool operations while preserving bounded stdout/stderr diagnostics.
class SubprocessManager {
public:
    /// @brief Creates a manager that reports lifecycle changes into the supplied job store.
    explicit SubprocessManager(InMemoryJobStore& store) : _store(store) {}
    /// @brief Applies subprocess capture and retention limits from BackendRuntime.
    void configure_limits(const BackendGuardLimits& limits) { _limits = limits; }

    /// @brief Starts a tracked background subprocess and returns the created job id.
    /// @brief on_complete, if set, runs on the job's worker thread right after its terminal
    /// JobState (ok/error/cancelled) is stored — before the job's process-table entry is erased.
    /// Kept generic (job_id + final JobState) so SubprocessManager stays a mechanism, not a policy;
    /// callers that need run-specific follow-up (e.g. the PI outcome recorder, Schritt 1c in
    /// docs/PI/pi_local_learning_plan_de.md) supply it per launch() instead of this class knowing
    /// about their business logic.
    std::string launch(const std::string& type,
                       const std::vector<std::string>& args,
                       const std::string& cwd = "",
                       const std::string& run_id = "",
                       const nlohmann::json& initial_data = {},
                       const std::string& stdin_text = "",
                       std::function<void(const std::string& job_id, JobState final_state)> on_complete = nullptr);

    /// @brief Requests cancellation for a single tracked job and its process group.
    bool cancel(const std::string& job_id);
    /// @brief Requests cancellation for all tracked jobs associated with a run id.
    void cancel_by_run(const std::string& run_id);

private:
    InMemoryJobStore& _store;
    BackendGuardLimits _limits{};
    mutable std::mutex _procs_mutex;
    std::unordered_map<std::string, std::shared_ptr<BackgroundProcess>> _procs;
};
