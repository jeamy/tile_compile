#pragma once
#include "backend_runtime.hpp"
#include "job_store.hpp"
#include "subprocess_manager.hpp"
#include "ui_event_store.hpp"
#include "services/config_revisions.hpp"
#include <string>
#include <mutex>
#include <nlohmann/json.hpp>

/// @brief Shared mutable state for all HTTP, WebSocket, and background-job handlers.
/// @details The backend passes one AppState instance to every route group so runtime paths,
/// transient UI state, job tracking, subprocess control, and revision history stay consistent
/// across request handlers and background worker threads.
struct AppState {
    BackendRuntime runtime;
    InMemoryJobStore job_store;
    SubprocessManager subprocess_manager{job_store};
    UiEventStore ui_event_store;
    ConfigRevisionStore revision_store;

    mutable std::mutex state_mutex;
    std::string current_run_id;
    std::string current_run_dir;  // absolute path, if known (e.g. network drive / non-default runs_dir)
    std::string active_config_revision_id;
    std::string last_scan_input_path;
    nlohmann::json ui_state = nlohmann::json::object();
    nlohmann::json preprocessing_parameters = nlohmann::json::object();
    bool ui_state_loaded = false;
};
