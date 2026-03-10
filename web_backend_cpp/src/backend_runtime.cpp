#include "backend_runtime.hpp"
#include <cstdlib>
#include <stdexcept>
#include <sstream>

BackendRuntime BackendRuntime::from_env() {
    BackendRuntime rt;

    auto env = [](const char* name, const char* def = "") -> std::string {
        const char* v = std::getenv(name);
        return v ? v : def;
    };

    std::string project_root_str = env("TILE_COMPILE_PROJECT_ROOT", ".");
    rt.project_root = fs::canonical(fs::path(project_root_str));

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

    rt.host = env("TILE_COMPILE_HOST", env("HOST", "127.0.0.1").c_str());
    rt.cli_exe    = env("TILE_COMPILE_CLI",    "tile_compile_cli");
    rt.runner_exe = env("TILE_COMPILE_RUNNER", "tile_compile_runner");

    std::string port_str = env("TILE_COMPILE_PORT", "8000");
    try { rt.port = std::stoi(port_str); } catch (...) { rt.port = 8000; }

    rt._allowed_roots.insert(rt.project_root.string());
    rt._allowed_roots.insert(rt.runs_dir.string());

    std::string allowed_roots = env("TILE_COMPILE_ALLOWED_ROOTS", "");
    if (!allowed_roots.empty()) {
        std::istringstream iss(allowed_roots);
        std::string root;
        while (std::getline(iss, root, ':')) {
            if (!root.empty()) rt._allowed_roots.insert(root);
        }
    }

    return rt;
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

bool BackendRuntime::is_path_allowed(const fs::path& p) const {
    std::lock_guard<std::mutex> lk(*_roots_mutex);
    std::string ps = p.string();
    for (auto& root : _allowed_roots) {
        if (ps.find(root) == 0) return true;
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

void BackendRuntime::grant_root(const fs::path& p) {
    std::lock_guard<std::mutex> lk(*_roots_mutex);
    _allowed_roots.insert(p.string());
}
