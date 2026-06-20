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

/// @brief Result category for backend path validation.
/// @details Used by route handlers to distinguish accepted paths from missing files and
/// paths that are outside the configured project/input safety roots.
enum class PathStatus { ok, not_allowed, not_found };

/// @brief Normalized path plus validation status.
/// @details Returned when user-supplied paths are resolved against backend guardrails before
/// they are passed to the scanner, runner, or file-serving endpoints.
struct PathResolution {
    PathStatus status{PathStatus::ok};
    fs::path path;
};

/// @brief Tunable retention and payload limits for backend operations.
/// @details The limits cap subprocess output, scan previews, report input size, and retained
/// job history so large datasets do not make API responses or in-memory state unbounded.
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

/// @brief Reads backend guard limits from environment variables.
/// @details Invalid or out-of-range values are clamped to conservative defaults in the
/// implementation so deployments can tune limits without changing code.
BackendGuardLimits backend_guard_limits_from_env();

/// @brief Runtime configuration and filesystem safety policy for the C++ web backend.
/// @details Owns resolved project paths, executable locations, bind settings, input search
/// roots, and path-grant state used by routes before launching tile_compile tools.
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

    /// @brief Builds a runtime configuration from environment variables and repository defaults.
    static BackendRuntime from_env();

    /// @brief Resolves a run id or run path to the concrete run directory.
    fs::path resolve_run_dir(const std::string& run_id) const;
    /// @brief Normalizes and validates a user-provided input path.
    PathResolution resolve_input_path(const fs::path& p, bool must_exist = false) const;
    /// @brief Returns whether a path is inside one of the configured or granted roots.
    bool is_path_allowed(const fs::path& p) const;
    /// @brief Adds a normalized directory to the allowed-root set for the current backend process.
    void grant_root(const fs::path& p);
    /// @brief Returns a snapshot of all roots currently accepted by path guardrails.
    std::vector<fs::path> allowed_roots() const;
    /// @brief Returns roots searched when resolving relative input paths.
    std::vector<fs::path> input_search_roots() const;

private:
    static fs::path normalize_path(const fs::path& p);
    static bool is_within_root(const fs::path& candidate, const fs::path& root);
    mutable std::unique_ptr<std::mutex> _roots_mutex;
    std::unordered_set<std::string> _allowed_roots;
    std::vector<fs::path> _input_search_roots;
};
