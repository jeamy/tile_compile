#pragma once
#include "backend_runtime.hpp"
#include "job_store.hpp"
#include "subprocess_manager.hpp"
#include "ui_event_store.hpp"
#include "services/config_revisions.hpp"
#include <string>
#include <mutex>
#include <nlohmann/json.hpp>

struct AppState {
    BackendRuntime runtime;
    InMemoryJobStore job_store;
    SubprocessManager subprocess_manager{job_store};
    UiEventStore ui_event_store;
    ConfigRevisionStore revision_store;

    mutable std::mutex state_mutex;
    std::string current_run_id;
    std::string active_config_revision_id;
    std::string last_scan_input_path;
};
