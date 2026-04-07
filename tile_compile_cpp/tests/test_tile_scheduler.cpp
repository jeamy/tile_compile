#include <catch2/catch_test_macros.hpp>

#include "tile_compile/reconstruction/tile_scheduler.hpp"
#include "tile_compile/core/types.hpp"

#include <atomic>
#include <vector>

using namespace tile_compile;
using namespace tile_compile::reconstruction;

static TileGrid make_linear_grid(int n_tiles) {
    TileGrid g;
    g.tile_size = 10; g.overlap_fraction = 0.0f;
    g.rows = 1; g.cols = n_tiles;
    for (int i = 0; i < n_tiles; ++i)
        g.tiles.push_back(Tile{i * 10, 0, 10, 10, 0, i});
    return g;
}

// ---------------------------------------------------------------------------
// test_scheduler_skips_dead_tiles
// ---------------------------------------------------------------------------
TEST_CASE("tile_scheduler_skips_dead_tiles") {
    const int N = 8;
    const auto grid = make_linear_grid(N);
    // Mark tiles 2 and 5 as dead.
    std::vector<bool> dead(N, false);
    dead[2] = true; dead[5] = true;

    std::atomic<int> processed{0};
    TileSchedulerConfig cfg;
    cfg.num_workers = 1;
    cfg.frame_sub_batch_size = 10;

    const auto result = run_tile_scheduler(
        grid, 10, dead, cfg,
        [&](const Tile&, size_t, size_t, size_t) { ++processed; });

    CHECK(result.tiles_skipped_dead == 2);
    CHECK(result.tiles_processed == N - 2);
    CHECK(processed.load() == N - 2);
}

// ---------------------------------------------------------------------------
// test_scheduler_uses_all_workers
// ---------------------------------------------------------------------------
TEST_CASE("tile_scheduler_uses_requested_worker_count") {
    const int N = 32;
    const auto grid = make_linear_grid(N);
    const std::vector<bool> dead(N, false);

    TileSchedulerConfig cfg;
    cfg.num_workers = 4;
    cfg.frame_sub_batch_size = 10;

    const auto result = run_tile_scheduler(
        grid, 10, dead, cfg,
        [](const Tile&, size_t, size_t, size_t) {});

    CHECK(result.workers_used == 4);
    CHECK(result.tiles_processed == N);
}

// ---------------------------------------------------------------------------
// test_scheduler_worker_count_in_result
// ---------------------------------------------------------------------------
TEST_CASE("tile_scheduler_result_workers_used_matches_config") {
    const auto grid = make_linear_grid(10);
    const std::vector<bool> dead(10, false);

    for (int w : {1, 2, 4}) {
        TileSchedulerConfig cfg;
        cfg.num_workers = w;
        cfg.frame_sub_batch_size = 5;
        const auto result = run_tile_scheduler(
            grid, 5, dead, cfg,
            [](const Tile&, size_t, size_t, size_t) {});
        CHECK(result.workers_used == w);
    }
}

// ---------------------------------------------------------------------------
// test_scheduler_all_dead_processes_nothing
// ---------------------------------------------------------------------------
TEST_CASE("tile_scheduler_all_dead_tiles_processes_nothing") {
    const int N = 5;
    const auto grid = make_linear_grid(N);
    const std::vector<bool> dead(N, true);

    std::atomic<int> called{0};
    TileSchedulerConfig cfg;
    cfg.num_workers = 2;
    cfg.frame_sub_batch_size = 10;

    const auto result = run_tile_scheduler(
        grid, 10, dead, cfg,
        [&](const Tile&, size_t, size_t, size_t) { ++called; });

    CHECK(result.tiles_processed == 0);
    CHECK(result.tiles_skipped_dead == N);
    CHECK(called.load() == 0);
}

// ---------------------------------------------------------------------------
// test_scheduler_sub_batch_calls_correct_ranges
// ---------------------------------------------------------------------------
TEST_CASE("tile_scheduler_sub_batch_covers_all_frames") {
    const int N_TILES = 2, N_FRAMES = 7, SUB = 3;
    const auto grid = make_linear_grid(N_TILES);
    const std::vector<bool> dead(N_TILES, false);

    std::atomic<size_t> total_frames_seen{0};
    TileSchedulerConfig cfg;
    cfg.num_workers = 1;
    cfg.frame_sub_batch_size = SUB;

    run_tile_scheduler(
        grid, N_FRAMES, dead, cfg,
        [&](const Tile&, size_t, size_t sb_start, size_t sb_end) {
            total_frames_seen += (sb_end - sb_start);
        });

    // Each tile should see all N_FRAMES across sub-batches.
    CHECK(total_frames_seen.load() == static_cast<size_t>(N_TILES * N_FRAMES));
}
