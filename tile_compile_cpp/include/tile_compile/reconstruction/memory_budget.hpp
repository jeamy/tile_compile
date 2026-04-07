#pragma once

#include <cstddef>
#include <string>

namespace tile_compile::reconstruction {

struct MemoryBudgetPlan {
    size_t frame_sub_batch_size        = 0;   // frames per sub-batch
    size_t allocated_frame_batch_bytes = 0;   // bytes reserved for one sub-batch
    size_t allocated_tile_batch_bytes  = 0;   // bytes reserved for OLA output buffers
    int    effective_workers           = 1;   // may be < requested if budget is tight
    bool   budget_warning              = false;
    std::string warning_reason;
};

/// Compute a memory-budget-aware batching plan for TILE_RECONSTRUCTION.
///
/// @param num_frames          Total number of input frames.
/// @param frame_rows          Height of each frame in pixels.
/// @param frame_cols          Width of each frame in pixels.
/// @param frame_channels      Number of channels per frame (1 = mono).
/// @param num_tiles           Total number of tiles in the grid.
/// @param max_tile_width      Maximum tile width in pixels.
/// @param max_tile_height     Maximum tile height in pixels.
/// @param tile_channels       Number of output channels per tile (e.g. 3 for RGB).
/// @param memory_budget_bytes Available RAM in bytes.
/// @param requested_workers   Desired number of parallel worker threads.
///
/// @throws ReconstructionError (memory_budget_too_small_for_frame) if budget
///         cannot hold even a single frame.
/// @throws ReconstructionError (memory_budget_too_small_for_tiles) if budget
///         cannot hold the OLA output buffers.
MemoryBudgetPlan compute_memory_budget_plan(
    int    num_frames,
    int    frame_rows,
    int    frame_cols,
    int    frame_channels,
    int    num_tiles,
    int    max_tile_width,
    int    max_tile_height,
    int    tile_channels,
    size_t memory_budget_bytes,
    int    requested_workers
);

} // namespace tile_compile::reconstruction
