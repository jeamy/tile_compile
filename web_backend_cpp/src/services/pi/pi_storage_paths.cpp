#include "services/pi/pi_storage_paths.hpp"

#include <cstdlib>
#include <fstream>
#include <system_error>

namespace fs = std::filesystem;

namespace tile_compile::pi {
namespace {

constexpr const char* kPiStorageKey = "pi.storage_dir";

fs::path ui_state_path(const std::shared_ptr<AppState>& state) {
    return state->runtime.runtime_dir / "ui_state.json";
}

void load_ui_state_unlocked(const std::shared_ptr<AppState>& state) {
    if (state->ui_state_loaded) return;
    state->ui_state = nlohmann::json::object();
    std::ifstream in(ui_state_path(state));
    if (in) {
        auto parsed = nlohmann::json::parse(in, nullptr, false);
        if (!parsed.is_discarded() && parsed.is_object()) state->ui_state = std::move(parsed);
    }
    state->ui_state_loaded = true;
}

bool save_ui_state_unlocked(const std::shared_ptr<AppState>& state) {
    const fs::path path = ui_state_path(state);
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    if (ec) return false;
    std::ofstream out(path, std::ios::trunc);
    if (!out) return false;
    out << state->ui_state.dump(2);
    return static_cast<bool>(out);
}

fs::path configured_storage_dir_unlocked(const std::shared_ptr<AppState>& state) {
    load_ui_state_unlocked(state);
    if (state->ui_state.contains(kPiStorageKey) && state->ui_state[kPiStorageKey].is_string()) {
        const std::string raw = state->ui_state[kPiStorageKey].get<std::string>();
        if (!raw.empty()) return fs::path(raw);
    }
    if (state->ui_state.contains("pi") && state->ui_state["pi"].is_object() &&
        state->ui_state["pi"].contains("storage_dir") && state->ui_state["pi"]["storage_dir"].is_string()) {
        const std::string raw = state->ui_state["pi"]["storage_dir"].get<std::string>();
        if (!raw.empty()) return fs::path(raw);
    }
    if (const char* raw = std::getenv("TILE_COMPILE_PI_STORAGE_DIR")) {
        if (*raw) return fs::path(raw);
    }
    return {};
}

fs::path normalize_for_storage(const std::shared_ptr<AppState>& state, const fs::path& requested) {
    return state->runtime.resolve_input_path(requested, false).path;
}

} // namespace

fs::path default_pi_storage_dir(const std::shared_ptr<AppState>& state) {
    return state->runtime.runs_dir / ".pi_memory";
}

fs::path pi_storage_dir(const std::shared_ptr<AppState>& state) {
    std::lock_guard<std::mutex> lk(state->state_mutex);
    const fs::path configured = configured_storage_dir_unlocked(state);
    if (configured.empty()) return default_pi_storage_dir(state);
    return normalize_for_storage(state, configured);
}

nlohmann::json pi_storage_status(const std::shared_ptr<AppState>& state) {
    fs::path configured;
    fs::path effective;
    {
        std::lock_guard<std::mutex> lk(state->state_mutex);
        configured = configured_storage_dir_unlocked(state);
        effective = configured.empty() ? default_pi_storage_dir(state) : normalize_for_storage(state, configured);
    }
    return {
        {"schema_version", "pi.storage.v1"},
        {"storage_dir", effective.string()},
        {"default_storage_dir", default_pi_storage_dir(state).string()},
        {"configured", !configured.empty()},
        {"configured_storage_dir", configured.empty() ? std::string() : effective.string()}
    };
}

bool set_pi_storage_dir(const std::shared_ptr<AppState>& state,
                        const fs::path& requested,
                        fs::path& resolved,
                        std::string& error_code,
                        std::string& error_message) {
    if (requested.empty()) {
        error_code = "PATH_INVALID";
        error_message = "storage_dir is required";
        return false;
    }
    const auto resolution = state->runtime.resolve_input_path(requested, false);
    resolved = resolution.path;
    if (resolution.status == PathStatus::not_allowed) {
        error_code = "PATH_NOT_ALLOWED";
        error_message = "PI storage directory is outside allowed roots";
        return false;
    }

    std::error_code ec;
    fs::create_directories(resolved, ec);
    if (ec) {
        error_code = "PATH_NOT_WRITABLE";
        error_message = "failed to create PI storage directory: " + ec.message();
        return false;
    }

    {
        std::lock_guard<std::mutex> lk(state->state_mutex);
        load_ui_state_unlocked(state);
        state->ui_state[kPiStorageKey] = resolved.string();
        if (!state->ui_state.contains("pi") || !state->ui_state["pi"].is_object()) {
            state->ui_state["pi"] = nlohmann::json::object();
        }
        state->ui_state["pi"]["storage_dir"] = resolved.string();
        if (!save_ui_state_unlocked(state)) {
            error_code = "INTERNAL_ERROR";
            error_message = "failed to persist PI storage directory";
            return false;
        }
    }
    return true;
}

} // namespace tile_compile::pi
