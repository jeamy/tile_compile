#pragma once
#include <filesystem>
#include <string>
#include <unordered_set>
#include <mutex>
#include <memory>
#include <vector>
#include <optional>

namespace fs = std::filesystem;

enum class PathStatus { ok, not_allowed, not_found };

struct PathResolution {
    PathStatus status{PathStatus::ok};
    fs::path path;
};

struct BackendRuntime {
    fs::path project_root;
    fs::path runs_dir;
    fs::path default_config_path;
    fs::path schema_path;
    fs::path presets_dir;
    fs::path ui_dir;
    fs::path runtime_dir;
    fs::path ui_events_path;
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
    PathResolution resolve_input_path(const fs::path& p, bool must_exist = false) const;
    bool is_path_allowed(const fs::path& p) const;
    void grant_root(const fs::path& p);
    std::vector<fs::path> allowed_roots() const;
    std::vector<fs::path> input_search_roots() const;

private:
    fs::path normalize_path(const fs::path& p) const;
    bool is_within_root(const fs::path& candidate, const fs::path& root) const;
    mutable std::unique_ptr<std::mutex> _roots_mutex;
    std::unordered_set<std::string> _allowed_roots;
    std::vector<fs::path> _input_search_roots;
};
