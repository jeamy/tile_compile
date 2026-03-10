#include "job_store.hpp"
#include <sstream>
#include <iomanip>
#include <chrono>

nlohmann::json job_to_json(const Job& j) {
    return {
        {"job_id",    j.job_id},
        {"type",      j.type},
        {"state",     job_state_str(j.state)},
        {"data",      j.data},
        {"progress",  j.progress},
        {"error",     j.error_message},
        {"run_id",    j.run_id},
    };
}

std::string InMemoryJobStore::create(const std::string& type, const std::string& run_id) {
    std::lock_guard<std::mutex> lk(_mutex);
    auto now = std::chrono::system_clock::now().time_since_epoch().count();
    std::ostringstream oss;
    oss << "job_" << (++_counter) << "_" << (now % 100000);
    std::string id = oss.str();
    Job j;
    j.job_id = id;
    j.type   = type;
    j.run_id = run_id;
    j.state  = JobState::pending;
    _jobs[id] = std::move(j);
    _order.push_back(id);
    return id;
}

std::optional<Job> InMemoryJobStore::get(const std::string& job_id) const {
    std::lock_guard<std::mutex> lk(_mutex);
    auto it = _jobs.find(job_id);
    if (it == _jobs.end()) return std::nullopt;
    return it->second;
}

bool InMemoryJobStore::update_state(const std::string& job_id, JobState state,
                                    const nlohmann::json& data, const std::string& error) {
    std::lock_guard<std::mutex> lk(_mutex);
    auto it = _jobs.find(job_id);
    if (it == _jobs.end()) return false;
    it->second.state = state;
    if (!data.is_null()) it->second.data = data;
    if (!error.empty())  it->second.error_message = error;
    return true;
}

bool InMemoryJobStore::update_progress(const std::string& job_id, double progress) {
    std::lock_guard<std::mutex> lk(_mutex);
    auto it = _jobs.find(job_id);
    if (it == _jobs.end()) return false;
    it->second.progress = progress;
    return true;
}

bool InMemoryJobStore::cancel(const std::string& job_id) {
    std::lock_guard<std::mutex> lk(_mutex);
    auto it = _jobs.find(job_id);
    if (it == _jobs.end()) return false;
    if (it->second.state == JobState::running || it->second.state == JobState::pending)
        it->second.state = JobState::cancelled;
    return true;
}

std::vector<Job> InMemoryJobStore::list(int limit) const {
    std::lock_guard<std::mutex> lk(_mutex);
    std::vector<Job> result;
    int start = (int)_order.size() - limit;
    if (start < 0) start = 0;
    for (int i = (int)_order.size() - 1; i >= start; --i) {
        auto it = _jobs.find(_order[i]);
        if (it != _jobs.end()) result.push_back(it->second);
    }
    return result;
}
