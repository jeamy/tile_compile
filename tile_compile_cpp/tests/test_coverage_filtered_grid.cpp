#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "tile_compile/pipeline/adaptive_tile_grid.hpp"
#include "tile_compile/core/errors.hpp"

using namespace tile_compile;
using namespace tile_compile::pipeline;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// All-ones mask (full coverage).
static std::vector<uint8_t> full_mask(int w, int h) {
    return std::vector<uint8_t>(static_cast<size_t>(w * h), 1);
}

// All-zeros mask (no coverage).
static std::vector<uint8_t> empty_mask(int w, int h) {
    return std::vector<uint8_t>(static_cast<size_t>(w * h), 0);
}

// Mask covering only the central rectangle.
static std::vector<uint8_t> central_mask(int w, int h, float fraction) {
    std::vector<uint8_t> mask(static_cast<size_t>(w * h), 0);
    const int x0 = static_cast<int>(w * (1.0f - fraction) / 2.0f);
    const int y0 = static_cast<int>(h * (1.0f - fraction) / 2.0f);
    const int x1 = w - x0;
    const int y1 = h - y0;
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x)
            mask[static_cast<size_t>(y * w + x)] = 1;
    return mask;
}

// ---------------------------------------------------------------------------
// Task 4.3a: leere Maske → identisches Grid wie build_initial_tile_grid
// ---------------------------------------------------------------------------
TEST_CASE("coverage_filter_no_mask_returns_same_as_initial_grid") {
    const int W = 400, H = 300, TS = 50;
    const float OV = 0.3f;

    const auto legacy = build_initial_tile_grid(W, H, TS, OV);
    const auto result = build_coverage_filtered_tile_grid(W, H, TS, OV);

    REQUIRE(result.grid.tiles.size() == legacy.size());
    CHECK(result.coverage_filtered_tiles == 0);
    for (size_t i = 0; i < legacy.size(); ++i) {
        CHECK(result.grid.tiles[i].x == legacy[i].x);
        CHECK(result.grid.tiles[i].y == legacy[i].y);
    }
}

// ---------------------------------------------------------------------------
// Task 4.3b: Canvas-Expansion < 10 % → kein Filter
// ---------------------------------------------------------------------------
TEST_CASE("coverage_filter_expansion_below_threshold_no_filtering") {
    // original = 400×300, expanded = 410×305 → expansion ≈ 4 % < 10 %
    const int W = 410, H = 305, TS = 50;
    const auto mask = full_mask(W, H);
    const auto result = build_coverage_filtered_tile_grid(
        W, H, TS, 0.3f, mask, 400, 300);

    // No filtering should occur.
    CHECK(result.coverage_filtered_tiles == 0);
}

// ---------------------------------------------------------------------------
// Task 4.3c: coverage_filtered_tiles korrekt gezählt
// ---------------------------------------------------------------------------
TEST_CASE("coverage_filter_counts_filtered_tiles_correctly") {
    // Canvas 400×300, original 200×200 → expansion > 10 %.
    // Mask covers only top-left 200×200 → tiles outside that area are dead.
    const int W = 400, H = 300, TS = 50;
    auto mask = empty_mask(W, H);
    // Cover top-left 200×200.
    for (int y = 0; y < 200; ++y)
        for (int x = 0; x < 200; ++x)
            mask[static_cast<size_t>(y * W + x)] = 1;

    const auto result = build_coverage_filtered_tile_grid(
        W, H, TS, 0.0f, mask, 200, 200, 0.15f);

    // Some tiles must have been filtered.
    CHECK(result.coverage_filtered_tiles > 0);
    // Remaining tiles must all have coverage >= 0.15.
    for (const auto& t : result.grid.tiles)
        CHECK(t.coverage_fraction >= 0.15f);
}

// ---------------------------------------------------------------------------
// Eigenschaft 6: full_support_tiles / num_tiles ≥ 0.85 after filtering
// ---------------------------------------------------------------------------
TEST_CASE("coverage_filter_full_support_invariant_holds") {
    // Use a mask that covers 90 % of the canvas centrally.
    // Canvas 400×300, original 200×200 → expansion > 10 %.
    const int W = 400, H = 300, TS = 50;
    const auto mask = central_mask(W, H, 0.9f);

    const auto result = build_coverage_filtered_tile_grid(
        W, H, TS, 0.0f, mask, 200, 200, 0.15f);

    if (!result.grid.tiles.empty()) {
        int full_support = 0;
        for (const auto& t : result.grid.tiles)
            if (t.coverage_fraction >= 0.85f) ++full_support;
        const float ratio = static_cast<float>(full_support) /
                            static_cast<float>(result.grid.tiles.size());
        CHECK(ratio >= 0.85f);
    }
}

// ---------------------------------------------------------------------------
// Task 4.3: build_initial_tile_grid wrapper still works
// ---------------------------------------------------------------------------
TEST_CASE("build_initial_tile_grid_wrapper_backward_compatible") {
    const auto tiles = build_initial_tile_grid(200, 150, 50, 0.3f);
    CHECK_FALSE(tiles.empty());
    // All tiles should have default is_dead = false.
    for (const auto& t : tiles)
        CHECK_FALSE(t.is_dead);
}
