#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <optional>
#include <nlohmann/json.hpp>

enum class JobState { pending, running, ok, error, cancelled };

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

nlohmann::json job_to_json(const Job& j);

class InMemoryJobStore {
public:
    std::string create(const std::string& type, const std::string& run_id = "");
    std::optional<Job> get(const std::string& job_id) const;
    bool update_state(const std::string& job_id, JobState state,
                      const nlohmann::json& data = {}, const std::string& error = "");
    bool merge_data(const std::string& job_id, const nlohmann::json& patch);
    bool update_progress(const std::string& job_id, double progress);
    bool set_pid(const std::string& job_id, std::optional<int> pid);
    bool is_cancelled(const std::string& job_id) const;
    bool cancel(const std::string& job_id);
    std::vector<Job> list(int limit = 100) const;

private:
    mutable std::mutex _mutex;
    std::unordered_map<std::string, Job> _jobs;
    std::vector<std::string> _order;
    int _counter{0};
};
