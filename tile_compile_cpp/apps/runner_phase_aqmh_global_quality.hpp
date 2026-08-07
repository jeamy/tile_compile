#pragma once

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/core/types.hpp"

#include <ostream>
#include <string>
#include <vector>

namespace tile_compile::runner {

bool run_phase_aqmh_global_quality(
    const std::string &run_id, const config::AqmhGlobalQualityConfig &cfg,
    const std::vector<float> &sharpness_summaries,
    const std::vector<float> &snr_summaries,
    const std::vector<float> &background_penalty_summaries,
    const std::vector<uint8_t> &frame_has_data,
    VectorXf &out_weights,
    std::vector<uint8_t> &out_input_invalid, core::EventEmitter &emitter,
    std::ostream &log_file);

} // namespace tile_compile::runner
