#pragma once

#include "types.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>

namespace tile_compile::core {

using json = nlohmann::json;

class EventEmitter {
public:
    EventEmitter() = default;
    
    void run_start(const std::string& run_id, const json& extra, std::ostream& out);
    void run_end(const std::string& run_id, bool success, const std::string& status, std::ostream& out);
    
    void phase_start(const std::string& run_id, Phase phase, const std::string& name, std::ostream& out);
    void phase_progress(const std::string& run_id, Phase phase, float progress, 
                        const std::string& message, std::ostream& out);
    void phase_end(const std::string& run_id, Phase phase, const std::string& status,
                   const json& extra, std::ostream& out);
    
    void frame_processed(const std::string& run_id, Phase phase, int frame_idx, 
                         int total_frames, const std::string& frame_name, std::ostream& out);
    
    void warning(const std::string& run_id, const std::string& message, std::ostream& out);
    void error(const std::string& run_id, const std::string& message, std::ostream& out);
    
private:
    void emit(const json& event, std::ostream& out);
    json base_event(const std::string& type, const std::string& run_id);
};

void emit_event(const std::string& type, const std::string& run_id, 
                const json& data, std::ostream& out);

} // namespace tile_compile::core
