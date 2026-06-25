#include "backend_runtime.hpp"
#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <system_error>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <winnetwk.h>
#pragma comment(lib, "mpr.lib")
#endif

namespace {

/// @brief Implements weakly normalize.
/// @details This implementation resolves backend runtime paths, environment overrides, and filesystem safety roots; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
fs::path weakly_normalize(const fs::path& p) {
    if (p.empty()) return {};
    std::error_code ec;
    fs::path normalized = fs::weakly_canonical(p, ec);
    if (ec) return p.lexically_normal();
    return normalized;
}

/// @brief Checks whether like project root.
/// @details This implementation resolves backend runtime paths, environment overrides, and filesystem safety roots; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
bool looks_like_project_root(const fs::path& dir) {
    if (dir.empty()) return false;
    std::error_code ec;
    return fs::is_directory(dir / "web_frontend_v3", ec) &&
           fs::is_directory(dir / "web_backend_cpp", ec) &&
           fs::is_directory(dir / "tile_compile_cpp", ec);
}

/// @brief Discovers project root.
/// @details This implementation resolves backend runtime paths, environment overrides, and filesystem safety roots; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
fs::path discover_project_root(fs::path start) {
    start = weakly_normalize(start);
    if (start.empty()) return {};
    std::error_code ec;
    if (!fs::is_directory(start, ec)) start = start.parent_path();
    for (fs::path current = start; !current.empty(); current = current.parent_path()) {
        if (looks_like_project_root(current)) return current;
        if (current == current.parent_path()) break;
    }
    return {};
}

/// @brief Implements detect default project root.
/// @details This implementation resolves backend runtime paths, environment overrides, and filesystem safety roots; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
fs::path detect_default_project_root() {
    if (auto from_cwd = discover_project_root(fs::current_path()); !from_cwd.empty()) {
        return from_cwd;
    }
#ifdef __linux__
    std::error_code ec;
    fs::path exe_path = fs::read_symlink("/proc/self/exe", ec);
    if (!ec) {
        if (auto from_exe = discover_project_root(exe_path.parent_path()); !from_exe.empty()) {
            return from_exe;
        }
    }
#endif
    return weakly_normalize(fs::current_path());
}

/// @brief Implements env string.
/// @details This implementation resolves backend runtime paths, environment overrides, and filesystem safety roots; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string env_string(const char* name, const char* def = "") {
    const char* v = std::getenv(name);
    return v ? v : def;
}

/// Splits a path list string into individual path strings.
/// On Windows, semicolons are used as separators (colons would break drive letters).
std::vector<std::string> split_colon_paths(const std::string& s) {
    std::vector<std::string> result;
    if (s.empty()) return result;
#ifdef _WIN32
    std::istringstream iss(s);
    std::string part;
    while (std::getline(iss, part, ';')) {
        if (!part.empty()) result.push_back(std::move(part));
    }
#else
    std::istringstream iss(s);
    std::string part;
    while (std::getline(iss, part, ':')) {
        if (!part.empty()) result.push_back(std::move(part));
    }
#endif
    return result;
}

/// @brief Parses size t env.
/// @details This implementation resolves backend runtime paths, environment overrides, and filesystem safety roots; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
size_t parse_size_t_env(const char* name, size_t fallback, size_t min_value, size_t max_value) {
    const std::string raw = env_string(name, "");
    if (raw.empty()) return fallback;
    try {
        const unsigned long long parsed = std::stoull(raw);
        return std::clamp<size_t>(static_cast<size_t>(parsed), min_value, max_value);
    } catch (...) {
        return fallback;
    }
}

/// @brief Parses uintmax env.
/// @details This implementation resolves backend runtime paths, environment overrides, and filesystem safety roots; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
uintmax_t parse_uintmax_env(const char* name, uintmax_t fallback, uintmax_t min_value, uintmax_t max_value) {
    const std::string raw = env_string(name, "");
    if (raw.empty()) return fallback;
    try {
        const unsigned long long parsed = std::stoull(raw);
        return std::clamp<uintmax_t>(static_cast<uintmax_t>(parsed), min_value, max_value);
    } catch (...) {
        return fallback;
    }
}

}

/// @brief Resolves an executable name to an absolute path.
/// @details Tries the value as-is, then relative to the backend executable's directory,
/// then relative to the current working directory, and finally on Windows appends `.exe`.
/// This ensures subprocess launches find the runner/CLI even when the backend is started from a different directory.
fs::path resolve_executable_path(const fs::path& candidate) {
    if (candidate.empty()) return {};
    std::error_code ec;

#ifdef _WIN32
    const bool has_ext = candidate.has_extension();
#else
    const bool has_ext = true;
#endif

    auto probe = [&](const fs::path& p) -> fs::path {
        if (fs::exists(p, ec) && fs::is_regular_file(p, ec)) return p;
#ifdef _WIN32
        if (!has_ext) {
            fs::path with_exe = p;
            with_exe += ".exe";
            if (fs::exists(with_exe, ec) && fs::is_regular_file(with_exe, ec)) return with_exe;
        }
#endif
        return {};
    };

    if (candidate.is_absolute()) return probe(candidate);

    fs::path backend_dir;
#ifdef _WIN32
    char buf[MAX_PATH] = {};
    if (GetModuleFileNameA(nullptr, buf, MAX_PATH) > 0) {
        backend_dir = fs::path(buf).parent_path();
    }
#else
    if (const char* self = std::getenv("_"); self && self[0]) {
        backend_dir = fs::path(self).parent_path();
    }
#endif
    if (!backend_dir.empty()) {
        if (auto p = probe(backend_dir / candidate); !p.empty()) return p;
    }
    if (auto p = probe(fs::current_path() / candidate); !p.empty()) return p;

    return candidate;
}

/// @brief Implements backend guard limits from env.
/// @details This implementation resolves backend runtime paths, environment overrides, and filesystem safety roots; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
BackendGuardLimits backend_guard_limits_from_env() {
    BackendGuardLimits limits;
    limits.subprocess_capture_bytes = parse_size_t_env("TILE_COMPILE_BACKEND_SUBPROCESS_CAPTURE_BYTES",
                                                       limits.subprocess_capture_bytes,
                                                       64 * 1024,
                                                       8 * 1024 * 1024);
    limits.job_stdio_store_bytes = parse_size_t_env("TILE_COMPILE_BACKEND_JOB_STDIO_STORE_BYTES",
                                                    limits.job_stdio_store_bytes,
                                                    8 * 1024,
                                                    512 * 1024);
    limits.scan_frames_preview = parse_size_t_env("TILE_COMPILE_BACKEND_SCAN_FRAMES_PREVIEW",
                                                  limits.scan_frames_preview,
                                                  1,
                                                  4096);
    limits.scan_per_dir_frames_preview = parse_size_t_env("TILE_COMPILE_BACKEND_SCAN_PER_DIR_FRAMES_PREVIEW",
                                                          limits.scan_per_dir_frames_preview,
                                                          1,
                                                          512);
    limits.scan_per_dir_results_preview = parse_size_t_env("TILE_COMPILE_BACKEND_SCAN_PER_DIR_RESULTS_PREVIEW",
                                                           limits.scan_per_dir_results_preview,
                                                           1,
                                                           512);
    limits.scan_messages_preview = parse_size_t_env("TILE_COMPILE_BACKEND_SCAN_MESSAGES_PREVIEW",
                                                    limits.scan_messages_preview,
                                                    1,
                                                    1024);
    limits.scan_color_candidates_preview = parse_size_t_env("TILE_COMPILE_BACKEND_SCAN_COLOR_CANDIDATES_PREVIEW",
                                                            limits.scan_color_candidates_preview,
                                                            1,
                                                            128);
    limits.report_events_max = parse_size_t_env("TILE_COMPILE_BACKEND_REPORT_EVENTS_MAX",
                                                limits.report_events_max,
                                                128,
                                                32768);
    limits.report_log_tail = parse_size_t_env("TILE_COMPILE_BACKEND_REPORT_LOG_TAIL",
                                              limits.report_log_tail,
                                              16,
                                              2048);
    limits.report_text_bytes = parse_size_t_env("TILE_COMPILE_BACKEND_REPORT_TEXT_BYTES",
                                                limits.report_text_bytes,
                                                32 * 1024,
                                                2 * 1024 * 1024);
    limits.report_json_file_bytes = parse_uintmax_env("TILE_COMPILE_BACKEND_REPORT_JSON_FILE_BYTES",
                                                      limits.report_json_file_bytes,
                                                      256 * 1024,
                                                      64 * 1024 * 1024);
    limits.retained_jobs = parse_size_t_env("TILE_COMPILE_BACKEND_RETAINED_JOBS",
                                            limits.retained_jobs,
                                            8,
                                            1000);
    return limits;
}

/// @brief Implements from env.
/// @details This implementation resolves backend runtime paths, environment overrides, and filesystem safety roots; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
BackendRuntime BackendRuntime::from_env() {
    BackendRuntime rt;

    rt.guard_limits = backend_guard_limits_from_env();

    std::string project_root_str = env_string("TILE_COMPILE_PROJECT_ROOT", "");
    if (project_root_str.empty()) {
        rt.project_root = detect_default_project_root();
    } else {
        rt.project_root = weakly_normalize(fs::path(project_root_str));
    }

    std::string runs_dir_str = env_string("TILE_COMPILE_RUNS_DIR", "");
    if (runs_dir_str.empty())
        rt.runs_dir = rt.project_root / "runs";
    else
        rt.runs_dir = weakly_normalize(fs::path(runs_dir_str));

    std::string config_str = env_string("TILE_COMPILE_CONFIG", "");
    if (config_str.empty())
        rt.default_config_path = rt.project_root / "tile_compile_cpp" / "tile_compile.yaml";
    else
        rt.default_config_path = fs::path(config_str);

    std::string schema_str = env_string("TILE_COMPILE_SCHEMA", "");
    if (schema_str.empty())
        rt.schema_path = rt.project_root / "tile_compile_cpp" / "tile_compile.schema.yaml";
    else
        rt.schema_path = fs::path(schema_str);

    std::string presets_str = env_string("TILE_COMPILE_PRESETS_DIR", "");
    if (presets_str.empty())
        rt.presets_dir = rt.project_root / "tile_compile_cpp" / "examples";
    else
        rt.presets_dir = fs::path(presets_str);

    std::string ui_str = env_string("TILE_COMPILE_UI_DIR", "");
    if (ui_str.empty())
        rt.ui_dir = rt.project_root / "web_frontend_v3";
    else
        rt.ui_dir = fs::path(ui_str);

    std::string runtime_dir_str = env_string("TILE_COMPILE_RUNTIME_DIR", "");
    if (runtime_dir_str.empty())
        rt.runtime_dir = rt.project_root / "web_backend_cpp" / "runtime";
    else
        rt.runtime_dir = fs::path(runtime_dir_str);
    rt.ui_events_path = rt.runtime_dir / "ui_events.jsonl";

    rt.host = env_string("TILE_COMPILE_HOST", env_string("HOST", "127.0.0.1").c_str());
    rt.cli_exe    = resolve_executable_path(env_string("TILE_COMPILE_CLI",    "tile_compile_cli")).string();
    rt.runner_exe = resolve_executable_path(env_string("TILE_COMPILE_RUNNER", "tile_compile_runner")).string();

    std::string port_str = env_string("TILE_COMPILE_PORT", "8000");
    try { rt.port = std::stoi(port_str); } catch (...) { rt.port = 8000; }

    for (const auto& root : {
             rt.project_root,
             rt.runs_dir,
#ifdef _WIN32
             fs::path(std::getenv("USERPROFILE") ? std::getenv("USERPROFILE") : ""),
             fs::path(std::getenv("HOMEPATH") ? std::getenv("HOMEPATH") : ""),
#else
             fs::path(std::getenv("HOME") ? std::getenv("HOME") : ""),
             fs::path("/tmp"),
#endif
         }) {
        if (!root.empty()) {
            const std::string normalized = rt.normalize_path(root).string();
            if (!normalized.empty()) rt._allowed_roots.insert(normalized);
        }
    }

    std::string input_dir_str = env_string("TILE_COMPILE_INPUT_DIR", "");
    if (!input_dir_str.empty()) {
        const std::string normalized = rt.normalize_path(fs::path(input_dir_str)).string();
        if (!normalized.empty()) rt._allowed_roots.insert(normalized);
    }

    std::string astap_data_dir_str = env_string("TILE_COMPILE_ASTAP_DATA_DIR", "");
    if (!astap_data_dir_str.empty()) {
        const std::string normalized = rt.normalize_path(fs::path(astap_data_dir_str)).string();
        if (!normalized.empty()) rt._allowed_roots.insert(normalized);
    }

    std::string siril_catalog_dir_str = env_string("TILE_COMPILE_SIRIL_CATALOG_DIR", "");
    if (!siril_catalog_dir_str.empty()) {
        const std::string normalized = rt.normalize_path(fs::path(siril_catalog_dir_str)).string();
        if (!normalized.empty()) rt._allowed_roots.insert(normalized);
    }

#ifdef _WIN32
    {
        const DWORD drives = GetLogicalDrives();
        for (int i = 0; i < 26; ++i) {
            if (drives & (1u << i)) {
                char drive_letter = static_cast<char>('A' + i);
                fs::path drive_root = fs::path(std::string(1, drive_letter) + ":\\");
                const std::string normalized = rt.normalize_path(drive_root).string();
                if (!normalized.empty()) rt._allowed_roots.insert(normalized);
            }
        }
    }
    // Enumerate UNC network shares (\\server\share) and add as allowed roots
    {
        HANDLE hEnum = nullptr;
        if (WNetOpenEnum(RESOURCE_GLOBALNET, RESOURCETYPE_DISK, 0, nullptr, &hEnum) == NO_ERROR) {
            DWORD count = 1;
            DWORD bufSize = 16384;
            std::vector<char> buf(bufSize);
            for (;;) {
                count = 1;
                DWORD result = WNetEnumResource(hEnum, &count, buf.data(), &bufSize);
                if (result == NO_ERROR && count > 0) {
                    auto* res = reinterpret_cast<NETRESOURCEA*>(buf.data());
                    for (DWORD i = 0; i < count; ++i) {
                        if (res[i].lpRemoteName && res[i].dwType == RESOURCETYPE_DISK) {
                            std::string unc_path(res[i].lpRemoteName);
                            // Normalize and add as allowed root
                            const std::string normalized = rt.normalize_path(fs::path(unc_path)).string();
                            if (!normalized.empty()) rt._allowed_roots.insert(normalized);
                        }
                    }
                } else if (result == ERROR_MORE_DATA) {
                    buf.resize(bufSize);
                    continue;
                } else {
                    break;
                }
            }
            if (hEnum) WNetCloseEnum(hEnum);
        }
    }
#endif

    std::string allowed_roots = env_string("TILE_COMPILE_ALLOWED_ROOTS", "");
    for (const auto& root : split_colon_paths(allowed_roots)) {
        rt._allowed_roots.insert(rt.normalize_path(fs::path(root)).string());
    }

    std::string input_roots = env_string("TILE_COMPILE_INPUT_SEARCH_ROOTS", "");
    for (const auto& root : split_colon_paths(input_roots)) {
        rt._input_search_roots.push_back(rt.normalize_path(fs::path(root)));
    }

    return rt;
}

/// @brief Normalizes path.
/// @details This implementation resolves backend runtime paths, environment overrides, and filesystem safety roots; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
fs::path BackendRuntime::normalize_path(const fs::path& p) {
    if (p.empty()) return {};
    std::error_code ec;
    fs::path candidate = p;
    if (!candidate.is_absolute()) candidate = fs::current_path() / candidate;
    fs::path normalized = fs::weakly_canonical(candidate, ec);
    if (ec) normalized = candidate.lexically_normal();
    return normalized;
}

/// @brief Checks whether within root.
/// @details This implementation resolves backend runtime paths, environment overrides, and filesystem safety roots; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
bool BackendRuntime::is_within_root(const fs::path& candidate, const fs::path& root) {
    const fs::path normalized_candidate = normalize_path(candidate);
    const fs::path normalized_root = normalize_path(root);
    if (normalized_candidate.empty() || normalized_root.empty()) return false;
    if (normalized_candidate == normalized_root) return true;

    auto root_it = normalized_root.begin();
    auto candidate_it = normalized_candidate.begin();
    for (; root_it != normalized_root.end() && candidate_it != normalized_candidate.end(); ++root_it, ++candidate_it) {
        if (*root_it != *candidate_it) return false;
    }
    return root_it == normalized_root.end();
}

/// @brief Resolves run dir.
/// @details This implementation resolves backend runtime paths, environment overrides, and filesystem safety roots; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
fs::path BackendRuntime::resolve_run_dir(const std::string& run_id, const std::string& alt_runs_dir) const {
    if (run_id.empty()) throw std::invalid_argument("run_id is empty");
    // If run_id is already an absolute path to an existing directory, return it directly.
    // This handles custom runs_dir / network drives where the frontend passes the full path.
    {
        fs::path as_path(run_id);
        if (as_path.is_absolute()) {
            std::error_code ec;
            if (fs::is_directory(as_path, ec) && !ec) return as_path;
        }
    }
    // Try alt_runs_dir first if provided (e.g. from job data for custom runs_dir)
    if (!alt_runs_dir.empty()) {
        fs::path alt_candidate = fs::path(alt_runs_dir) / run_id;
        if (fs::is_directory(alt_candidate)) return alt_candidate;
        std::error_code ec;
        if (fs::is_directory(alt_runs_dir, ec)) {
            for (auto& entry : fs::directory_iterator(alt_runs_dir, ec)) {
                if (ec) break;
                if (!entry.is_directory()) continue;
                std::string name = entry.path().filename().string();
                if (name == run_id || name.find(run_id) == 0)
                    return entry.path();
            }
        }
    }
    fs::path candidate = runs_dir / run_id;
    if (fs::is_directory(candidate)) return candidate;
    std::error_code ec2;
    if (fs::is_directory(runs_dir, ec2)) {
        for (auto& entry : fs::directory_iterator(runs_dir, ec2)) {
            if (ec2) break;
            if (!entry.is_directory()) continue;
            std::string name = entry.path().filename().string();
            if (name == run_id || name.find(run_id) == 0)
                return entry.path();
        }
    }
    throw std::runtime_error("run_dir not found for run_id: " + run_id);
}

/// @brief Resolves input path.
/// @details This implementation resolves backend runtime paths, environment overrides, and filesystem safety roots; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
PathResolution BackendRuntime::resolve_input_path(const fs::path& p, bool must_exist) const {
    if (p.empty()) return {PathStatus::not_found, {}};

    if (p.is_absolute()) {
        fs::path normalized = normalize_path(p);
        if (!is_path_allowed(normalized)) return {PathStatus::not_allowed, normalized};
        if (must_exist && !fs::exists(normalized)) return {PathStatus::not_found, normalized};
        return {PathStatus::ok, normalized};
    }

    for (const auto& base : _input_search_roots) {
        fs::path candidate = normalize_path(base / p);
        if (!is_path_allowed(candidate)) continue;
        if (fs::exists(candidate)) return {PathStatus::ok, candidate};
    }

    fs::path fallback = normalize_path(project_root / p);
    if (!is_path_allowed(fallback)) return {PathStatus::not_allowed, fallback};
    if (must_exist && !fs::exists(fallback)) return {PathStatus::not_found, fallback};
    return {PathStatus::ok, fallback};
}

/// @brief Checks whether path allowed.
/// @details This implementation resolves backend runtime paths, environment overrides, and filesystem safety roots; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
bool BackendRuntime::is_path_allowed(const fs::path& p) const {
    std::lock_guard<std::mutex> lk(*_roots_mutex);
    for (auto& root : _allowed_roots) {
        if (is_within_root(p, fs::path(root))) return true;
    }
    return false;
}

/// @brief Implements allowed roots.
/// @details This implementation resolves backend runtime paths, environment overrides, and filesystem safety roots; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::vector<fs::path> BackendRuntime::allowed_roots() const {
    std::lock_guard<std::mutex> lk(*_roots_mutex);
    std::vector<fs::path> roots;
    roots.reserve(_allowed_roots.size());
    for (const auto& root : _allowed_roots) roots.emplace_back(root);
    return roots;
}

/// @brief Implements input search roots.
/// @details This implementation resolves backend runtime paths, environment overrides, and filesystem safety roots; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::vector<fs::path> BackendRuntime::input_search_roots() const {
    std::lock_guard<std::mutex> lk(*_roots_mutex);
    return _input_search_roots;
}

/// @brief Implements grant root.
/// @details This implementation resolves backend runtime paths, environment overrides, and filesystem safety roots; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
void BackendRuntime::grant_root(const fs::path& p) {
    std::lock_guard<std::mutex> lk(*_roots_mutex);
    _allowed_roots.insert(normalize_path(p).string());
}
