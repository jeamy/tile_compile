#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "tile_compile/reconstruction/dead_tile_detector.hpp"
#include "tile_compile/reconstruction/reconstruction.hpp"
#include "tile_compile/core/types.hpp"

using namespace tile_compile;
using namespace tile_compile::reconstruction;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a simple 1-row TileGrid with tiles of given width.
static TileGrid make_grid(int canvas_w, int canvas_h, int tile_w, int tile_h) {
    TileGrid g;
    g.tile_size = tile_w;
    g.overlap_fraction = 0.0f;
    g.rows = canvas_h / tile_h;
    g.cols = canvas_w / tile_w;
    for (int r = 0; r < g.rows; ++r)
        for (int c = 0; c < g.cols; ++c)
            g.tiles.push_back(Tile{c * tile_w, r * tile_h, tile_w, tile_h, r, c});
    return g;
}

// Build a flat canvas mask: all pixels in [x0,x1) × [y0,y1) are set to 1.
static std::vector<uint8_t> make_mask(int w, int h, int x0, int y0, int x1, int y1) {
    std::vector<uint8_t> mask(static_cast<size_t>(w * h), 0);
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x)
            mask[static_cast<size_t>(y * w + x)] = 1;
    return mask;
}

// ---------------------------------------------------------------------------
// Eigenschaft 3: dead_mask[i] == (coverage(tile_i, mask) < min_coverage_fraction)
// ---------------------------------------------------------------------------
TEST_CASE("dead_tile_detector_classification_matches_coverage_threshold") {
    // 4×1 grid, each tile 10×10. Canvas 40×10.
    // Mask covers only the first 10 columns (tile 0 fully covered, rest dead).
    const int W = 40, H = 10, TW = 10, TH = 10;
    const auto grid = make_grid(W, H, TW, TH);
    const auto mask = make_mask(W, H, 0, 0, 10, 10); // only tile 0 covered

    const auto dead = detect_dead_tiles(grid, mask, W, H, 0.01f);

    REQUIRE(dead.size() == 4);
    CHECK_FALSE(dead[0]); // tile 0: fully covered
    CHECK(dead[1]);       // tile 1: no coverage
    CHECK(dead[2]);       // tile 2: no coverage
    CHECK(dead[3]);       // tile 3: no coverage
}

TEST_CASE("dead_tile_detector_partial_coverage_below_threshold_is_dead") {
    // Tile is 10×10 = 100 pixels. Only 0 pixels covered → dead.
    const int W = 10, H = 10;
    TileGrid grid;
    grid.tile_size = 10;
    grid.rows = 1; grid.cols = 1;
    grid.tiles.push_back(Tile{0, 0, 10, 10, 0, 0});

    const std::vector<uint8_t> empty_mask(100, 0);
    const auto dead = detect_dead_tiles(grid, empty_mask, W, H, 0.01f);
    REQUIRE(dead.size() == 1);
    CHECK(dead[0]);
}

TEST_CASE("dead_tile_detector_partial_coverage_above_threshold_is_alive") {
    // Tile 10×10, 5 pixels covered = 5 % > 1 % threshold → alive.
    const int W = 10, H = 10;
    TileGrid grid;
    grid.tile_size = 10;
    grid.rows = 1; grid.cols = 1;
    grid.tiles.push_back(Tile{0, 0, 10, 10, 0, 0});

    auto mask = make_mask(W, H, 0, 0, 5, 1); // 5 pixels
    const auto dead = detect_dead_tiles(grid, mask, W, H, 0.01f);
    REQUIRE(dead.size() == 1);
    CHECK_FALSE(dead[0]);
}

TEST_CASE("dead_tile_detector_empty_mask_treats_all_tiles_as_alive") {
    const auto grid = make_grid(40, 10, 10, 10);
    const std::vector<uint8_t> empty_mask;
    const auto dead = detect_dead_tiles(grid, empty_mask, 40, 10, 0.01f);
    for (bool d : dead) CHECK_FALSE(d);
}

// ---------------------------------------------------------------------------
// Eigenschaft 4: Dead-Tile-Bereiche sind Null im Stack, Dimensionen unverändert
// ---------------------------------------------------------------------------
TEST_CASE("reconstruct_tiles_parallel_dead_tile_area_is_zero") {
    // 2 tiles side by side: tile 0 (x=0..3) and tile 1 (x=4..7).
    // Mark tile 1 as dead. Frames have value 1.0 everywhere.
    // After reconstruction, tile 1 area should be 0.

    const int H = 4, W = 8;
    Matrix2Df frame = Matrix2Df::Constant(H, W, 1.0f);
    const std::vector<Matrix2Df> frames = {frame};

    TileGrid grid;
    grid.tile_size = 4;
    grid.overlap_fraction = 0.0f;
    grid.rows = 1; grid.cols = 2;
    grid.tiles = {
        Tile{0, 0, 4, 4, 0, 0},
        Tile{4, 0, 4, 4, 0, 1},
    };

    // Uniform weights.
    const std::vector<std::vector<float>> weights = {{1.0f, 1.0f}};

    // Tile 1 is dead.
    const std::vector<bool> dead_mask = {false, true};

    ReconstructionConfig cfg;
    cfg.parallel_workers = 1;
    cfg.memory_budget_bytes = 64ULL * 1024 * 1024;

    const auto result = reconstruct_tiles_parallel(frames, grid, weights, dead_mask, cfg);

    REQUIRE(result.output.rows() == H);
    REQUIRE(result.output.cols() == W);
    CHECK(result.tiles_skipped_dead == 1);
    CHECK(result.tiles_processed == 1);

    // Tile 0 area should be non-zero.
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < 4; ++x)
            CHECK(result.output(y, x) == Catch::Approx(1.0f).margin(1e-5f));

    // Tile 1 area should be zero (dead tile, no contribution).
    for (int y = 0; y < H; ++y)
        for (int x = 4; x < W; ++x)
            CHECK(result.output(y, x) == Catch::Approx(0.0f).margin(1e-5f));
}
