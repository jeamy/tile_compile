#pragma once

#include "runner_shared.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/acceleration.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/image/normalization.hpp"
#include "tile_compile/metrics/metrics.hpp"

#include <memory>

namespace tile_compile::metrics { class QualityMapCache; }

namespace tile_compile::reconstruction {
class AqmhPrefetchCoordinator;
}

namespace tile_compile::runner {

bool run_phase_aqmh_maps(
    const std::string &run_id, const config::Config &cfg,
    const std::vector<std::filesystem::path> &frames,
    const std::filesystem::path &run_dir,
    const std::vector<uint8_t> &frame_has_data,
    const std::vector<uint8_t> &reconstruction_valid_mask,
    const std::vector<uint8_t> &analysis_common_mask, int width, int height,
    const DiskCacheFrameStore &prewarped_frames,
    const std::vector<image::NormalizationScales> &norm_scales,
    ColorMode detected_mode, const std::string &detected_bayer_str,
    bool apply_normalization, core::AccelerationContext &acceleration,
    core::EventEmitter &emitter, std::ostream &log_file,
    std::unique_ptr<metrics::QualityMapCache> &out_cache,
    VectorXf &out_global_weights,
    std::unique_ptr<reconstruction::AqmhPrefetchCoordinator> &out_prefetch_coordinator,
    const std::vector<metrics::FrameStarMetrics> &frame_star_metrics = {});

} // namespace tile_compile::runner
