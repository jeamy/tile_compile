#include "tile_compile/runner/events.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace tile_compile::runner {

EventEmitter::EventEmitter(std::ofstream* log_file)
    : log_file_(log_file) {}

std::string EventEmitter::get_timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count() << 'Z';
    return oss.str();
}

void EventEmitter::emit(const nlohmann::json& event) {
    std::string line = event.dump();
    
    // Emit to stdout
    std::cout << line << std::endl;
    std::cout.flush();
    
    // Emit to log file if available
    if (log_file_ && log_file_->is_open()) {
        (*log_file_) << line << std::endl;
        log_file_->flush();
    }
}

void EventEmitter::phase_start(const std::string& run_id, int phase_id, 
                               const std::string& phase_name,
                               const nlohmann::json& extra) {
    nlohmann::json event;
    event["type"] = "phase_start";
    event["run_id"] = run_id;
    event["phase"] = phase_id;
    event["phase_name"] = phase_name;
    event["ts"] = get_timestamp();
    
    if (!extra.empty() && extra.is_object()) {
        for (auto& [key, value] : extra.items()) {
            event[key] = value;
        }
    }
    
    emit(event);
}

void EventEmitter::phase_end(const std::string& run_id, int phase_id,
                             const std::string& phase_name, const std::string& status,
                             const nlohmann::json& extra) {
    nlohmann::json event;
    event["type"] = "phase_end";
    event["run_id"] = run_id;
    event["phase"] = phase_id;
    event["phase_name"] = phase_name;
    event["status"] = status;
    event["ts"] = get_timestamp();
    
    if (!extra.empty() && extra.is_object()) {
        for (auto& [key, value] : extra.items()) {
            event[key] = value;
        }
    }
    
    emit(event);
}

void EventEmitter::phase_progress(const std::string& run_id, int phase_id,
                                  const std::string& phase_name,
                                  int current, int total,
                                  const nlohmann::json& extra) {
    nlohmann::json event;
    event["type"] = "phase_progress";
    event["run_id"] = run_id;
    event["phase"] = phase_id;
    event["phase_name"] = phase_name;
    event["current"] = current;
    event["total"] = total;
    event["ts"] = get_timestamp();
    
    if (!extra.empty() && extra.is_object()) {
        for (auto& [key, value] : extra.items()) {
            event[key] = value;
        }
    }
    
    emit(event);
}

void EventEmitter::run_start(const std::string& run_id, const nlohmann::json& data) {
    nlohmann::json event;
    event["type"] = "run_start";
    event["run_id"] = run_id;
    event["ts"] = get_timestamp();
    
    if (!data.empty() && data.is_object()) {
        for (auto& [key, value] : data.items()) {
            event[key] = value;
        }
    }
    
    emit(event);
}

void EventEmitter::run_end(const std::string& run_id, bool success, const nlohmann::json& data) {
    nlohmann::json event;
    event["type"] = "run_end";
    event["run_id"] = run_id;
    event["success"] = success;
    event["status"] = success ? "ok" : "error";
    event["ts"] = get_timestamp();
    
    if (!data.empty() && data.is_object()) {
        for (auto& [key, value] : data.items()) {
            event[key] = value;
        }
    }
    
    emit(event);
}

void EventEmitter::run_error(const std::string& run_id, const std::string& error,
                             const std::string& traceback) {
    nlohmann::json event;
    event["type"] = "run_error";
    event["run_id"] = run_id;
    event["error"] = error;
    event["ts"] = get_timestamp();
    
    if (!traceback.empty()) {
        event["traceback"] = traceback;
    }
    
    emit(event);
}

void EventEmitter::stop_requested(const std::string& run_id, int phase_id,
                                  const std::string& phase_name) {
    nlohmann::json event;
    event["type"] = "run_stop_requested";
    event["run_id"] = run_id;
    event["phase"] = phase_id;
    event["phase_name"] = phase_name;
    event["ts"] = get_timestamp();
    
    emit(event);
}

} // namespace tile_compile::runner
