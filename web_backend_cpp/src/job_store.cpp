#include "job_store.hpp"
#include <sstream>
#include <iomanip>
#include <chrono>

namespace {

std::string utc_now_iso() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto tt = system_clock::to_time_t(now);
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

nlohmann::json job_to_json(const Job& j) {
    return {
        {"job_id",    j.job_id},
        {"type",      j.type},
        {"run_id",    j.data.is_object() && j.data.contains("run_id") ? j.data["run_id"] : (j.run_id.empty() ? nlohmann::json(nullptr) : nlohmann::json(j.run_id))},
        {"state",     job_state_str(j.state)},
        {"pid",       j.pid.has_value() ? nlohmann::json(*j.pid) : nlohmann::json(nullptr)},
        {"exit_code", j.data.is_object() && j.data.contains("exit_code") ? j.data["exit_code"] : nlohmann::json(nullptr)},
        {"created_at", j.created_at.empty() ? nlohmann::json(nullptr) : nlohmann::json(j.created_at)},
        {"updated_at", j.updated_at.empty() ? nlohmann::json(nullptr) : nlohmann::json(j.updated_at)},
        {"started_at", j.started_at.empty() ? nlohmann::json(nullptr) : nlohmann::json(j.started_at)},
        {"ended_at",   j.ended_at.empty() ? nlohmann::json(nullptr) : nlohmann::json(j.ended_at)},
        {"data",      j.data},
        {"error",     j.error_message},
        {"progress",  j.progress},
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
    j.created_at = utc_now_iso();
    j.updated_at = j.created_at;
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
    if (state == JobState::running && it->second.started_at.empty()) {
        it->second.started_at = utc_now_iso();
    }
    it->second.state = state;
    if (!data.is_null()) it->second.data = data;
    if (!error.empty())  it->second.error_message = error;
    it->second.updated_at = utc_now_iso();
    if ((state == JobState::ok || state == JobState::error || state == JobState::cancelled) && it->second.ended_at.empty()) {
        it->second.ended_at = it->second.updated_at;
    }
    return true;
}

bool InMemoryJobStore::merge_data(const std::string& job_id, const nlohmann::json& patch) {
    std::lock_guard<std::mutex> lk(_mutex);
    auto it = _jobs.find(job_id);
    if (it == _jobs.end()) return false;
    if (!patch.is_object()) {
        it->second.data = patch;
    } else {
        if (!it->second.data.is_object()) it->second.data = nlohmann::json::object();
        for (auto patch_it = patch.begin(); patch_it != patch.end(); ++patch_it) {
            it->second.data[patch_it.key()] = patch_it.value();
        }
    }
    it->second.updated_at = utc_now_iso();
    return true;
}

bool InMemoryJobStore::update_progress(const std::string& job_id, double progress) {
    std::lock_guard<std::mutex> lk(_mutex);
    auto it = _jobs.find(job_id);
    if (it == _jobs.end()) return false;
    it->second.progress = progress;
    it->second.updated_at = utc_now_iso();
    return true;
}

bool InMemoryJobStore::set_pid(const std::string& job_id, std::optional<int> pid) {
    std::lock_guard<std::mutex> lk(_mutex);
    auto it = _jobs.find(job_id);
    if (it == _jobs.end()) return false;
    it->second.pid = pid;
    it->second.updated_at = utc_now_iso();
    return true;
}

bool InMemoryJobStore::is_cancelled(const std::string& job_id) const {
    std::lock_guard<std::mutex> lk(_mutex);
    auto it = _jobs.find(job_id);
    if (it == _jobs.end()) return false;
    return it->second.state == JobState::cancelled;
}

bool InMemoryJobStore::cancel(const std::string& job_id) {
    std::lock_guard<std::mutex> lk(_mutex);
    auto it = _jobs.find(job_id);
    if (it == _jobs.end()) return false;
    if (it->second.state == JobState::running || it->second.state == JobState::pending)
        it->second.state = JobState::cancelled;
    it->second.updated_at = utc_now_iso();
    if (it->second.ended_at.empty()) it->second.ended_at = it->second.updated_at;
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
