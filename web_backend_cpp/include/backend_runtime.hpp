#pragma once
#include <filesystem>
#include <string>
#include <unordered_set>
#include <mutex>
#include <memory>

namespace fs = std::filesystem;

struct BackendRuntime {
    fs::path project_root;
    fs::path runs_dir;
    fs::path default_config_path;
    fs::path schema_path;
    fs::path presets_dir;
    fs::path ui_dir;
    std::string host{"127.0.0.1"};
    std::string cli_exe;
    std::string runner_exe;
    int port{8000};

    BackendRuntime() : _roots_mutex(std::make_unique<std::mutex>()) {}
    BackendRuntime(BackendRuntime&&) = default;
    BackendRuntime& operator=(BackendRuntime&&) = default;
    BackendRuntime(const BackendRuntime&) = delete;
    BackendRuntime& operator=(const BackendRuntime&) = delete;

    static BackendRuntime from_env();

    fs::path resolve_run_dir(const std::string& run_id) const;
    bool is_path_allowed(const fs::path& p) const;
    void grant_root(const fs::path& p);

private:
    mutable std::unique_ptr<std::mutex> _roots_mutex;
    std::unordered_set<std::string> _allowed_roots;
};
