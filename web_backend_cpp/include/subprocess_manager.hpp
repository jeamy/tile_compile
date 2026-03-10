#pragma once
#include "job_store.hpp"
#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <mutex>

struct SubprocessResult {
    int exit_code{-1};
    std::string stdout_str;
    std::string stderr_str;
};

SubprocessResult run_subprocess(const std::vector<std::string>& args,
                                const std::string& cwd = "",
                                const std::string& stdin_text = "");

struct BackgroundProcess {
    std::string job_id;
    std::thread thread;
    std::atomic<bool> cancelled{false};
};

class SubprocessManager {
public:
    explicit SubprocessManager(InMemoryJobStore& store) : _store(store) {}

    std::string launch(const std::string& type,
                       const std::vector<std::string>& args,
                       const std::string& cwd = "",
                       const std::string& run_id = "");

    bool cancel(const std::string& job_id);
    void cancel_by_run(const std::string& run_id);

private:
    InMemoryJobStore& _store;
    mutable std::mutex _procs_mutex;
    std::unordered_map<std::string, std::shared_ptr<BackgroundProcess>> _procs;
};
