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

struct SubprocessResult {
    int exit_code{-1};
    std::string stdout_str;
    std::string stderr_str;
    size_t stdout_bytes{0};
    size_t stderr_bytes{0};
    bool stdout_truncated{false};
    bool stderr_truncated{false};
};

SubprocessResult run_subprocess(const std::vector<std::string>& args,
                                const std::string& cwd = "",
                                const std::string& stdin_text = "",
                                const BackendGuardLimits* limits = nullptr);

struct BackgroundProcess {
    std::string job_id;
    std::thread thread;
    std::atomic<bool> cancelled{false};
    std::atomic<int> pid{-1};
#ifdef _WIN32
    void* process_handle{nullptr};
#endif
};

class SubprocessManager {
public:
    explicit SubprocessManager(InMemoryJobStore& store) : _store(store) {}
    void configure_limits(const BackendGuardLimits& limits) { _limits = limits; }

    std::string launch(const std::string& type,
                       const std::vector<std::string>& args,
                       const std::string& cwd = "",
                       const std::string& run_id = "",
                       const nlohmann::json& initial_data = {},
                       const std::string& stdin_text = "");

    bool cancel(const std::string& job_id);
    void cancel_by_run(const std::string& run_id);

private:
    InMemoryJobStore& _store;
    BackendGuardLimits _limits{};
    mutable std::mutex _procs_mutex;
    std::unordered_map<std::string, std::shared_ptr<BackgroundProcess>> _procs;
};
