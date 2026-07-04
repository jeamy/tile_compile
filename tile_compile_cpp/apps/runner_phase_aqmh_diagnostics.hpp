#pragma once

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/reconstruction/aqmh_reconstruction.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace tile_compile::metrics { class QualityMapCache; }

namespace tile_compile::runner {

bool run_phase_aqmh_diagnostics(
    const std::string &run_id, const config::Config &cfg,
    const std::filesystem::path &run_dir,
    const reconstruction::AqmhReconstructionResult &reconstruction,
    metrics::QualityMapCache *q_map_cache,
    const std::vector<uint8_t> &canvas_mask,
    const std::vector<uint8_t> &frame_has_data,
    int canvas_width, int canvas_height,
    core::EventEmitter &emitter, std::ostream &log_file);

} // namespace tile_compile::runner
