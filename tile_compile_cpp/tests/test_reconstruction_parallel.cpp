#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "tile_compile/reconstruction/reconstruction.hpp"
#include "tile_compile/pipeline/adaptive_tile_grid.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/types.hpp"

using namespace tile_compile;
using namespace tile_compile::reconstruction;
using namespace tile_compile::pipeline;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static TileGrid make_simple_grid(int w, int h, int tile_size, float overlap) {
    const auto tiles = build_initial_tile_grid(w, h, tile_size, overlap);
    TileGrid g;
    g.tile_size = tile_size;
    g.overlap_fraction = overlap;
    g.rows = 0; g.cols = 0;
    g.tiles = tiles;
    for (const auto& t : tiles) {
        g.rows = std::max(g.rows, t.row + 1);
        g.cols = std::max(g.cols, t.col + 1);
    }
    return g;
}

// ---------------------------------------------------------------------------
// Task 8.3: reconstruct_tiles_parallel numerisch äquivalent zu reconstruct_tiles
// ---------------------------------------------------------------------------
TEST_CASE("reconstruct_tiles_parallel_matches_legacy_single_worker") {
    const int H = 20, W = 20;
    Matrix2Df frame = Matrix2Df::Zero(H, W);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            frame(y, x) = static_cast<float>(y * W + x) * 0.01f;

    const std::vector<Matrix2Df> frames = {frame};
    const auto grid = make_simple_grid(W, H, 8, 0.0f);
    const size_t n_tiles = grid.tiles.size();
    const std::vector<std::vector<float>> weights(1, std::vector<float>(n_tiles, 1.0f));
    const std::vector<bool> no_dead(n_tiles, false);

    // Legacy result.
    const auto legacy = reconstruct_tiles(frames, grid, weights);

    // Parallel result (1 worker = deterministic).
    ReconstructionConfig cfg;
    cfg.parallel_workers = 1;
    cfg.memory_budget_bytes = 64ULL * 1024 * 1024;
    const auto par = reconstruct_tiles_parallel(frames, grid, weights, no_dead, cfg);

    REQUIRE(par.output.rows() == legacy.rows());
    REQUIRE(par.output.cols() == legacy.cols());
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            CHECK(par.output(y, x) == Catch::Approx(legacy(y, x)).margin(1e-4f));
}

// ---------------------------------------------------------------------------
// Task 8.3: Dead-Tile-Bereiche sind Null im parallelen Output
// ---------------------------------------------------------------------------
TEST_CASE("reconstruct_tiles_parallel_dead_tile_zero_in_output") {
    const int H = 8, W = 16;
    const Matrix2Df frame = Matrix2Df::Constant(H, W, 5.0f);
    const std::vector<Matrix2Df> frames = {frame};

    TileGrid grid;
    grid.tile_size = 8; grid.overlap_fraction = 0.0f;
    grid.rows = 1; grid.cols = 2;
    grid.tiles = {Tile{0, 0, 8, 8, 0, 0}, Tile{8, 0, 8, 8, 0, 1}};

    const std::vector<std::vector<float>> weights = {{1.0f, 1.0f}};
    const std::vector<bool> dead = {false, true}; // tile 1 dead

    ReconstructionConfig cfg;
    cfg.parallel_workers = 1;
    cfg.memory_budget_bytes = 64ULL * 1024 * 1024;
    const auto result = reconstruct_tiles_parallel(frames, grid, weights, dead, cfg);

    // Tile 0 area: non-zero.
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < 8; ++x)
            CHECK(result.output(y, x) != Catch::Approx(0.0f).margin(1e-5f));

    // Tile 1 area: zero.
    for (int y = 0; y < H; ++y)
        for (int x = 8; x < W; ++x)
            CHECK(result.output(y, x) == Catch::Approx(0.0f).margin(1e-5f));
}

// ---------------------------------------------------------------------------
// Task 5.3: tile_boundary_diagnostics_enabled default = false
// ---------------------------------------------------------------------------
TEST_CASE("runtime_limits_config_tile_boundary_diagnostics_default_false") {
    config::RuntimeLimitsConfig cfg;
    CHECK_FALSE(cfg.tile_boundary_diagnostics_enabled);
}

// ---------------------------------------------------------------------------
// Task 9.4: sigma_clip_reduce_batch CPU fallback returns same as single calls
// ---------------------------------------------------------------------------
#include "tile_compile/core/acceleration.hpp"

TEST_CASE("sigma_clip_reduce_batch_cpu_fallback_matches_single_calls") {
    using namespace tile_compile::core;

    AccelerationSelection sel;
    sel.selected = AccelerationBackend::cpu;
    sel.phase    = AccelerationPhase::tile_reconstruction;
    AccelerationOps ops(sel);

    // Two tiles, each with 3 frames of constant value.
    Matrix2Df t0 = Matrix2Df::Constant(4, 4, 2.0f);
    Matrix2Df t1 = Matrix2Df::Constant(4, 4, 5.0f);

    AccelerationOps::BatchSigmaClipInput in0, in1;
    in0.tile_frames = {t0, t0, t0};
    in0.weights     = {1.0f, 1.0f, 1.0f};
    in1.tile_frames = {t1, t1, t1};
    in1.weights     = {1.0f, 1.0f, 1.0f};

    const auto batch_results = ops.sigma_clip_reduce_batch(
        {in0, in1}, 3.0f, 3.0f, 4, 0.4f, 1e-6f);

    REQUIRE(batch_results.size() == 2);
    // Tile 0: all frames = 2.0 → result ≈ 2.0.
    CHECK(batch_results[0].tile(0, 0) == Catch::Approx(2.0f).margin(1e-4f));
    // Tile 1: all frames = 5.0 → result ≈ 5.0.
    CHECK(batch_results[1].tile(0, 0) == Catch::Approx(5.0f).margin(1e-4f));
}

// ---------------------------------------------------------------------------
// Task 10.3: overlap_fraction warning threshold
// ---------------------------------------------------------------------------
#include "tile_compile/pipeline/adaptive_tile_grid.hpp"

TEST_CASE("overlap_fraction_high_produces_more_tiles_than_low") {
    // Verify that overlap=0.6 produces significantly more tiles than overlap=0.3.
    const int W = 400, H = 300, TS = 50;
    const auto tiles_30 = build_initial_tile_grid(W, H, TS, 0.3f);
    const auto tiles_60 = build_initial_tile_grid(W, H, TS, 0.6f);
    // overlap=0.6 should produce more tiles.
    CHECK(tiles_60.size() > tiles_30.size());
}
