#pragma once
#include <filesystem>
#include <string>
#include <unordered_set>
#include <mutex>
#include <memory>
#include <vector>
#include <optional>
#include <cstddef>
#include <cstdint>

namespace fs = std::filesystem;

enum class PathStatus { ok, not_allowed, not_found };

struct PathResolution {
    PathStatus status{PathStatus::ok};
    fs::path path;
};

struct BackendGuardLimits {
    size_t subprocess_capture_bytes{1024 * 1024};
    size_t job_stdio_store_bytes{128 * 1024};
    size_t scan_frames_preview{256};
    size_t scan_per_dir_frames_preview{32};
    size_t scan_per_dir_results_preview{64};
    size_t scan_messages_preview{128};
    size_t scan_color_candidates_preview{32};
    size_t report_events_max{4096};
    size_t report_log_tail{128};
    size_t report_text_bytes{256 * 1024};
    uintmax_t report_json_file_bytes{4 * 1024 * 1024};
    size_t retained_jobs{128};
};

BackendGuardLimits backend_guard_limits_from_env();

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
    BackendGuardLimits guard_limits;

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
