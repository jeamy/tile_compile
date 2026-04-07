#pragma once

#include "tile_compile/core/types.hpp"
#include "tile_compile/reconstruction/progress_reporter.hpp"

#include <cstddef>
#include <functional>
#include <vector>

namespace tile_compile::reconstruction {

struct TileSchedulerConfig {
    int    num_workers          = 1;
    size_t frame_sub_batch_size = 0; // 0 = process all frames at once
    int    gpu_tile_batch_size  = 1; // tiles per GPU dispatch (>1 enables batching)
};

struct TileSchedulerResult {
    int    tiles_processed                 = 0;
    int    tiles_skipped_dead              = 0;
    double processing_time_s               = 0.0;
    double dead_tile_time_saved_estimate_s = 0.0;
    int    workers_used                    = 0;
};

/// Callback invoked by a worker thread for each live tile.
///
/// @param tile              The tile to process.
/// @param tile_index        Index of the tile in grid.tiles.
/// @param sub_batch_start   First frame index in the current sub-batch.
/// @param sub_batch_end     One-past-last frame index in the current sub-batch.
using TileProcessFn = std::function<void(
    const Tile& tile,
    size_t      tile_index,
    size_t      sub_batch_start,
    size_t      sub_batch_end)>;

/// Run the parallel tile scheduler.
///
/// Distributes all non-dead tiles across @p cfg.num_workers threads.
/// Frames are processed in sub-batches of @p cfg.frame_sub_batch_size.
/// Dead tiles (dead_tile_mask[i] == true) are skipped without calling
/// @p process_fn.
///
/// @param grid           Tile grid.
/// @param num_frames     Total number of input frames.
/// @param dead_tile_mask Per-tile dead flag (same size as grid.tiles).
/// @param cfg            Scheduler configuration.
/// @param process_fn     Callback invoked per tile per sub-batch.
/// @param reporter       Optional progress reporter (may be nullptr).
TileSchedulerResult run_tile_scheduler(
    const TileGrid&          grid,
    size_t                   num_frames,
    const std::vector<bool>& dead_tile_mask,
    const TileSchedulerConfig& cfg,
    TileProcessFn            process_fn,
    ProgressReporter*        reporter = nullptr
);

} // namespace tile_compile::reconstruction
