#pragma once

#include "runner_shared.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/image/normalization.hpp"

#include <memory>
#include <string>
#include <vector>

namespace tile_compile::metrics {
class QualityMapCache;
}

namespace tile_compile::runner {

/// Compute per-frame/per-tile local quality metrics and local reconstruction weights.
///
/// The phase reads prewarped frame tiles from `prewarped_frames`, gates them by
/// the common-overlap masks, optionally applies stored normalization scales,
/// and computes local structure/star metrics for every live tile. The output
/// matrices are indexed as `[frame_index][tile_index]` and feed tile-weighted
/// reconstruction, synthetic-frame generation, diagnostics, and report plots.
///
/// `tile_offset_x` and `tile_offset_y` translate tile-grid coordinates into the
/// prewarped canvas coordinate system when registration expanded the canvas.
/// Returns `true` when all local metric artifacts were written successfully.
bool run_phase_local_metrics(
    const std::string &run_id, const config::Config &cfg,
    const std::vector<std::filesystem::path> &frames,
    const std::filesystem::path &run_dir,
    const std::vector<uint8_t> &frame_has_data,
    const std::vector<Tile> &tiles_phase56,
    const std::vector<uint8_t> &common_valid_mask,
    int common_mask_width, int common_mask_height,
    const std::vector<uint8_t> &tile_common_valid,
    const DiskCacheFrameStore &prewarped_frames,
    const std::vector<image::NormalizationScales> &norm_scales,
    ColorMode detected_mode, const std::string &detected_bayer_str,
    bool apply_normalization_to_tiles, core::EventEmitter &emitter,
    std::ostream &log_file, std::vector<std::vector<TileMetrics>> &local_metrics,
    std::vector<std::vector<float>> &local_weights,
    std::vector<float> &tile_quality_median, std::vector<uint8_t> &tile_is_star,
    std::vector<float> &tile_fwhm_median,
    std::unique_ptr<metrics::QualityMapCache> &out_aqmh_cache,
    int tile_offset_x = 0, int tile_offset_y = 0);

} // namespace tile_compile::runner
