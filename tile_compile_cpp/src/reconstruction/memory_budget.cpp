#include "tile_compile/reconstruction/memory_budget.hpp"
#include "tile_compile/core/errors.hpp"

#include <algorithm>
#include <cstddef>

namespace tile_compile::reconstruction {

/// @brief Computes memory budget plan.
/// @details Part of memory-budget planning for reconstruction batch sizing and worker use; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
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
    int    requested_workers)
{
    (void)num_tiles;

    // Bytes for a single float-precision frame.
    const size_t frame_bytes =
        static_cast<size_t>(frame_rows) *
        static_cast<size_t>(frame_cols) *
        static_cast<size_t>(frame_channels) *
        sizeof(float);

    // OLA output buffers: global accum + weight_sum plus per-worker tile scratch.
    const int safe_workers = std::max(1, requested_workers);
    const size_t output_buffer_bytes =
        static_cast<size_t>(frame_rows) *
        static_cast<size_t>(frame_cols) *
        static_cast<size_t>(tile_channels) *
        sizeof(float) *
        2;
    const size_t tile_pixel_bytes =
        static_cast<size_t>(max_tile_width) *
        static_cast<size_t>(max_tile_height) *
        static_cast<size_t>(tile_channels) *
        sizeof(float);
    const size_t requested_tile_scratch_bytes =
        static_cast<size_t>(safe_workers) * tile_pixel_bytes * 2;
    const size_t initial_tile_batch_bytes =
        output_buffer_bytes + requested_tile_scratch_bytes;

    if (initial_tile_batch_bytes >= memory_budget_bytes) {
        throw ReconstructionError(
            ReconstructionError::Code::memory_budget_too_small_for_tiles,
            "OLA output buffers (" + std::to_string(initial_tile_batch_bytes / (1024 * 1024)) +
            " MB) exceed memory budget (" +
            std::to_string(memory_budget_bytes / (1024 * 1024)) + " MB)");
    }

    // 80 % of budget minus OLA buffers is available for frame data.
    const size_t frame_budget_window = (memory_budget_bytes * 8) / 10;
    const size_t available_for_frames =
        (initial_tile_batch_bytes < frame_budget_window)
            ? (frame_budget_window - initial_tile_batch_bytes)
            : 0;

    if (frame_bytes == 0 || available_for_frames < frame_bytes) {
        throw ReconstructionError(
            ReconstructionError::Code::memory_budget_too_small_for_frame,
            "Memory budget too small to hold a single frame (" +
            std::to_string(frame_bytes / (1024 * 1024)) + " MB per frame, " +
            std::to_string(available_for_frames / (1024 * 1024)) + " MB available)");
    }

    // Sub-batch size: how many frames fit in the available budget.
    size_t sub_batch_size = available_for_frames / frame_bytes;
    sub_batch_size = std::min(sub_batch_size, static_cast<size_t>(num_frames));

    // Each worker needs its own copy of the sub-batch in the worst case.
    // Reduce workers if necessary, but never below max(1, N/2) when budget > 512 MB.
    int effective_workers = safe_workers;
    const size_t per_worker_bytes = sub_batch_size * frame_bytes;
    if (per_worker_bytes > 0 &&
        per_worker_bytes * static_cast<size_t>(effective_workers) > available_for_frames)
    {
        effective_workers = static_cast<int>(available_for_frames / per_worker_bytes);
        effective_workers = std::max(1, effective_workers);
    }

    MemoryBudgetPlan plan;
    plan.frame_sub_batch_size        = sub_batch_size;
    plan.allocated_frame_batch_bytes = sub_batch_size * frame_bytes;
    plan.allocated_tile_batch_bytes  =
        output_buffer_bytes +
        static_cast<size_t>(effective_workers) * tile_pixel_bytes * 2;
    plan.effective_workers           = effective_workers;

    // Warn if workers were reduced below the minimum floor.
    const int min_workers = std::max(1, requested_workers / 2);
    const bool budget_large_enough = memory_budget_bytes > 512ULL * 1024 * 1024;
    if (effective_workers < min_workers && budget_large_enough) {
        plan.budget_warning  = true;
        plan.warning_reason  =
            "Worker count reduced to " + std::to_string(effective_workers) +
            " (requested " + std::to_string(requested_workers) +
            ", minimum floor " + std::to_string(min_workers) +
            ") due to memory budget constraints (" +
            std::to_string(memory_budget_bytes / (1024 * 1024)) + " MB)";
    }

    return plan;
}

} // namespace tile_compile::reconstruction
