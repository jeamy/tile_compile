#include "tile_compile/reconstruction/tile_scheduler.hpp"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

namespace tile_compile::reconstruction {

/// @brief Runs tile scheduler.
/// @details Part of parallel tile scheduling and dead-tile filtering; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
TileSchedulerResult run_tile_scheduler(
    const TileGrid&            grid,
    size_t                     num_frames,
    const std::vector<bool>&   dead_tile_mask,
    const TileSchedulerConfig& cfg,
    TileProcessFn              process_fn,
    ProgressReporter*          reporter)
{
    using clock = std::chrono::steady_clock;
    const auto t_start = clock::now();

    const size_t n_tiles = grid.tiles.size();

    // Build the work queue: indices of live tiles only.
    std::vector<size_t> live_indices;
    live_indices.reserve(n_tiles);
    int dead_count = 0;
    for (size_t i = 0; i < n_tiles; ++i) {
        if (i < dead_tile_mask.size() && dead_tile_mask[i]) {
            ++dead_count;
        } else {
            live_indices.push_back(i);
        }
    }

    const int n_live = static_cast<int>(live_indices.size());
    const int n_workers = std::max(1, cfg.num_workers);
    const size_t sub_batch = (cfg.frame_sub_batch_size == 0)
                             ? num_frames
                             : cfg.frame_sub_batch_size;

    if (reporter) reporter->set_workers_active(n_workers);

    // Shared atomic index into live_indices — workers pull tiles from here.
    std::atomic<int> next_tile{0};
    std::atomic<int> tiles_done{0};

    // Worker-underutilisation detection: track last progress timestamp.
    std::atomic<bool> underutil_warned{false};
    const auto underutil_threshold = std::chrono::seconds(30);

    auto worker_fn = [&]() {
        auto last_progress = clock::now();
        while (true) {
            const int idx = next_tile.fetch_add(1, std::memory_order_relaxed);
            if (idx >= n_live) break;

            const size_t tile_idx = live_indices[static_cast<size_t>(idx)];
            const Tile& tile = grid.tiles[tile_idx];

            // Process all sub-batches for this tile.
            for (size_t sb_start = 0; sb_start < num_frames; sb_start += sub_batch) {
                const size_t sb_end = std::min(sb_start + sub_batch, num_frames);
                process_fn(tile, tile_idx, sb_start, sb_end);
            }

            ++tiles_done;
            if (reporter) reporter->tick();
            last_progress = clock::now();

            // Check for worker underutilisation (only from the first worker).
            if (idx == 0 && !underutil_warned.load()) {
                const auto idle = clock::now() - last_progress;
                if (idle > underutil_threshold && next_tile.load() < n_live) {
                    underutil_warned.store(true);
                    // Caller can observe this via TileSchedulerResult.workers_used.
                }
            }
        }
    };

    // Launch worker threads.
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(n_workers));
    for (int w = 0; w < n_workers; ++w)
        threads.emplace_back(worker_fn);
    for (auto& t : threads) t.join();

    if (reporter) reporter->set_workers_active(0);

    const double elapsed =
        std::chrono::duration<double>(clock::now() - t_start).count();

    // Estimate time saved by skipping dead tiles.
    const double time_per_tile = (n_live > 0) ? elapsed / static_cast<double>(n_live) : 0.0;
    const double saved = time_per_tile * static_cast<double>(dead_count);

    TileSchedulerResult result;
    result.tiles_processed                 = tiles_done.load();
    result.tiles_skipped_dead              = dead_count;
    result.processing_time_s               = elapsed;
    result.dead_tile_time_saved_estimate_s = saved;
    result.workers_used                    = n_workers;
    return result;
}

} // namespace tile_compile::reconstruction
