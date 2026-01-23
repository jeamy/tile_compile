#include "tile_compile/core/types.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/config/configuration.hpp"

namespace tile_compile::pipeline {

class PipelineRunner {
public:
    bool run(const std::string& run_id,
             const fs::path& run_dir,
             const fs::path& project_root,
             const config::Config& cfg,
             const std::vector<fs::path>& input_frames,
             std::ostream& log_stream,
             std::atomic<bool>* stop_flag = nullptr) {
        
        core::EventEmitter emitter;
        
        emitter.phase_start(run_id, Phase::SCAN_INPUT, "SCAN_INPUT", log_stream);
        emitter.phase_end(run_id, Phase::SCAN_INPUT, "ok", {}, log_stream);
        
        emitter.phase_start(run_id, Phase::DONE, "DONE", log_stream);
        emitter.phase_end(run_id, Phase::DONE, "ok", {}, log_stream);
        
        return true;
    }
};

} // namespace tile_compile::pipeline
