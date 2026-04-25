#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <optional>
#include <nlohmann/json.hpp>

/// @brief Lifecycle state for asynchronous backend jobs.
/// @details Jobs start pending, become running while a subprocess or worker thread is active,
/// and finish as ok, error, or cancelled for polling and WebSocket status updates.
enum class JobState { pending, running, ok, error, cancelled };

/// @brief Converts a JobState to the stable API string used in JSON responses.
inline std::string job_state_str(JobState s) {
    switch (s) {
        case JobState::pending:   return "pending";
        case JobState::running:   return "running";
        case JobState::ok:        return "ok";
        case JobState::error:     return "error";
        case JobState::cancelled: return "cancelled";
    }
    return "unknown";
}

/// @brief In-memory record for one backend-managed asynchronous operation.
/// @details Stores status, progress, process id, timestamps, run association, and arbitrary
/// structured data returned by scanner, runner, downloader, and tool-install tasks.
struct Job {
    std::string job_id;
    std::string type;
    JobState state{JobState::pending};
    nlohmann::json data{};
    double progress{0.0};
    std::optional<int> pid;
    std::string error_message;
    std::string run_id;
    std::string created_at;
    std::string updated_at;
    std::string started_at;
    std::string ended_at;
};

/// @brief Serializes a Job into the public REST/WebSocket JSON shape.
nlohmann::json job_to_json(const Job& j);

/// @brief Thread-safe bounded store for transient backend jobs.
/// @details The store is intentionally process-local; it coordinates API polling, WebSocket
/// overlays, cancellation flags, and recent-job history without persisting job state to disk.
class InMemoryJobStore {
public:
    /// @brief Creates a new pending job and returns its generated id.
    std::string create(const std::string& type, const std::string& run_id = "");
    /// @brief Sets the maximum number of completed/recent jobs retained in memory.
    void configure_retention(size_t max_retained_jobs);
    /// @brief Fetches a copy of a job by id.
    std::optional<Job> get(const std::string& job_id) const;
    /// @brief Updates job state, optional payload data, and optional error text.
    bool update_state(const std::string& job_id, JobState state,
                      const nlohmann::json& data = {}, const std::string& error = "");
    /// @brief Merges a JSON patch object into the job data object.
    bool merge_data(const std::string& job_id, const nlohmann::json& patch);
    /// @brief Stores a normalized progress value for a running job.
    bool update_progress(const std::string& job_id, double progress);
    /// @brief Associates or clears the operating-system process id for a job.
    bool set_pid(const std::string& job_id, std::optional<int> pid);
    /// @brief Returns true when a job has been marked cancelled.
    bool is_cancelled(const std::string& job_id) const;
    /// @brief Marks a job as cancelled so workers and subprocesses can stop cooperatively.
    bool cancel(const std::string& job_id);
    /// @brief Lists recent jobs in newest-first order, capped by limit.
    std::vector<Job> list(int limit = 100) const;

private:
    void prune_locked();

    mutable std::mutex _mutex;
    std::unordered_map<std::string, Job> _jobs;
    std::vector<std::string> _order;
    size_t _max_retained_jobs{128};
    int _counter{0};
};
