#include "backend_runtime.hpp"
#include <cstdlib>
#include <stdexcept>
#include <sstream>
#include <system_error>

namespace {

fs::path weakly_normalize(const fs::path& p) {
    if (p.empty()) return {};
    std::error_code ec;
    fs::path normalized = fs::weakly_canonical(p, ec);
    if (ec) return p.lexically_normal();
    return normalized;
}

bool looks_like_project_root(const fs::path& dir) {
    if (dir.empty()) return false;
    std::error_code ec;
    return fs::is_directory(dir / "web_frontend", ec) &&
           fs::is_directory(dir / "web_backend_cpp", ec) &&
           fs::is_directory(dir / "tile_compile_cpp", ec);
}

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

}

BackendRuntime BackendRuntime::from_env() {
    BackendRuntime rt;

    auto env = [](const char* name, const char* def = "") -> std::string {
        const char* v = std::getenv(name);
        return v ? v : def;
    };

    std::string project_root_str = env("TILE_COMPILE_PROJECT_ROOT", "");
    if (project_root_str.empty()) {
        rt.project_root = detect_default_project_root();
    } else {
        rt.project_root = weakly_normalize(fs::path(project_root_str));
    }

    std::string runs_dir_str = env("TILE_COMPILE_RUNS_DIR", "");
    if (runs_dir_str.empty())
        rt.runs_dir = rt.project_root / "runs";
    else
        rt.runs_dir = fs::path(runs_dir_str);

    std::string config_str = env("TILE_COMPILE_CONFIG", "");
    if (config_str.empty())
        rt.default_config_path = rt.project_root / "tile_compile_cpp" / "tile_compile.yaml";
    else
        rt.default_config_path = fs::path(config_str);

    std::string schema_str = env("TILE_COMPILE_SCHEMA", "");
    if (schema_str.empty())
        rt.schema_path = rt.project_root / "tile_compile_cpp" / "tile_compile.schema.yaml";
    else
        rt.schema_path = fs::path(schema_str);

    std::string presets_str = env("TILE_COMPILE_PRESETS_DIR", "");
    if (presets_str.empty())
        rt.presets_dir = rt.project_root / "tile_compile_cpp" / "examples";
    else
        rt.presets_dir = fs::path(presets_str);

    std::string ui_str = env("TILE_COMPILE_UI_DIR", "");
    if (ui_str.empty())
        rt.ui_dir = rt.project_root / "web_frontend";
    else
        rt.ui_dir = fs::path(ui_str);

    std::string runtime_dir_str = env("TILE_COMPILE_RUNTIME_DIR", "");
    if (runtime_dir_str.empty())
        rt.runtime_dir = rt.project_root / "web_backend_cpp" / "runtime";
    else
        rt.runtime_dir = fs::path(runtime_dir_str);
    rt.ui_events_path = rt.runtime_dir / "ui_events.jsonl";

    rt.host = env("TILE_COMPILE_HOST", env("HOST", "127.0.0.1").c_str());
    rt.cli_exe    = env("TILE_COMPILE_CLI",    "tile_compile_cli");
    rt.runner_exe = env("TILE_COMPILE_RUNNER", "tile_compile_runner");

    std::string port_str = env("TILE_COMPILE_PORT", "8000");
    try { rt.port = std::stoi(port_str); } catch (...) { rt.port = 8000; }

    for (const auto& root : {
             rt.project_root,
             rt.runs_dir,
             fs::path(std::getenv("HOME") ? std::getenv("HOME") : ""),
             fs::path("/tmp"),
             fs::path("/media"),
         }) {
        if (!root.empty()) rt._allowed_roots.insert(rt.normalize_path(root).string());
    }

    std::string allowed_roots = env("TILE_COMPILE_ALLOWED_ROOTS", "");
    if (!allowed_roots.empty()) {
        std::istringstream iss(allowed_roots);
        std::string root;
        while (std::getline(iss, root, ':')) {
            if (!root.empty()) rt._allowed_roots.insert(rt.normalize_path(fs::path(root)).string());
        }
    }

    std::string input_roots = env("TILE_COMPILE_INPUT_SEARCH_ROOTS", "");
    if (!input_roots.empty()) {
        std::istringstream iss(input_roots);
        std::string root;
        while (std::getline(iss, root, ':')) {
            if (!root.empty()) rt._input_search_roots.push_back(rt.normalize_path(fs::path(root)));
        }
    }

    return rt;
}

fs::path BackendRuntime::normalize_path(const fs::path& p) const {
    if (p.empty()) return {};
    std::error_code ec;
    fs::path candidate = p;
    if (!candidate.is_absolute()) candidate = fs::current_path() / candidate;
    fs::path normalized = fs::weakly_canonical(candidate, ec);
    if (ec) normalized = candidate.lexically_normal();
    return normalized;
}

bool BackendRuntime::is_within_root(const fs::path& candidate, const fs::path& root) const {
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

fs::path BackendRuntime::resolve_run_dir(const std::string& run_id) const {
    if (run_id.empty()) throw std::invalid_argument("run_id is empty");
    fs::path candidate = runs_dir / run_id;
    if (fs::is_directory(candidate)) return candidate;
    for (auto& entry : fs::directory_iterator(runs_dir)) {
        if (!entry.is_directory()) continue;
        std::string name = entry.path().filename().string();
        if (name == run_id || name.find(run_id) == 0)
            return entry.path();
    }
    throw std::runtime_error("run_dir not found for run_id: " + run_id);
}

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

bool BackendRuntime::is_path_allowed(const fs::path& p) const {
    std::lock_guard<std::mutex> lk(*_roots_mutex);
    for (auto& root : _allowed_roots) {
        if (is_within_root(p, fs::path(root))) return true;
    }
    return false;
}

std::vector<fs::path> BackendRuntime::allowed_roots() const {
    std::lock_guard<std::mutex> lk(*_roots_mutex);
    std::vector<fs::path> roots;
    roots.reserve(_allowed_roots.size());
    for (const auto& root : _allowed_roots) roots.emplace_back(root);
    return roots;
}

std::vector<fs::path> BackendRuntime::input_search_roots() const {
    std::lock_guard<std::mutex> lk(*_roots_mutex);
    return _input_search_roots;
}

void BackendRuntime::grant_root(const fs::path& p) {
    std::lock_guard<std::mutex> lk(*_roots_mutex);
    _allowed_roots.insert(normalize_path(p).string());
}
