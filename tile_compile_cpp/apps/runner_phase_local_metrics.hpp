#pragma once

#include "runner_shared.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/core/types.hpp"

#include <string>
#include <vector>

namespace tile_compile::runner {

bool run_phase_local_metrics(
    const std::string &run_id, const config::Config &cfg,
    const std::vector<std::filesystem::path> &frames,
    const std::filesystem::path &run_dir,
    const std::vector<uint8_t> &frame_has_data,
    const std::vector<Tile> &tiles_phase56,
    const DiskCacheFrameStore &prewarped_frames, core::EventEmitter &emitter,
    std::ostream &log_file, std::vector<std::vector<TileMetrics>> &local_metrics,
    std::vector<std::vector<float>> &local_weights,
    std::vector<float> &tile_quality_median, std::vector<uint8_t> &tile_is_star,
    std::vector<float> &tile_fwhm_median, int tile_offset_x = 0, int tile_offset_y = 0);

} // namespace tile_compile::runner
