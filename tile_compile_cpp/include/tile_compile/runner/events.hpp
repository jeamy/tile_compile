#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <fstream>

namespace tile_compile::runner {

/**
 * Event emission for runner pipeline.
 * Emits JSON events to stdout and optional log file for GUI consumption.
 */
class EventEmitter {
public:
    explicit EventEmitter(std::ofstream* log_file = nullptr);
    
    void emit(const nlohmann::json& event);
    
    void phase_start(const std::string& run_id, int phase_id, const std::string& phase_name,
                     const nlohmann::json& extra = nlohmann::json::object());
    
    void phase_end(const std::string& run_id, int phase_id, const std::string& phase_name,
                   const std::string& status, const nlohmann::json& extra = nlohmann::json::object());
    
    void phase_progress(const std::string& run_id, int phase_id, const std::string& phase_name,
                       int current, int total, const nlohmann::json& extra = nlohmann::json::object());
    
    void run_start(const std::string& run_id, const nlohmann::json& data);
    void run_end(const std::string& run_id, bool success, const nlohmann::json& data = nlohmann::json::object());
    void run_error(const std::string& run_id, const std::string& error, const std::string& traceback = "");
    
    void stop_requested(const std::string& run_id, int phase_id, const std::string& phase_name);
    
private:
    std::ofstream* log_file_;
    
    std::string get_timestamp() const;
};

} // namespace tile_compile::runner
