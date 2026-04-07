#pragma once

#include "tile_compile/core/types.hpp"
#include <cstdint>
#include <vector>

namespace tile_compile::pipeline {

struct TileGridBuildResult {
    TileGrid grid;
    int      coverage_filtered_tiles = 0; // tiles removed due to missing coverage
    float    coverage_fraction       = 1.0f; // fraction of canvas with frame coverage
};

/// Build a tile grid filtered by frame coverage.
///
/// When @p canvas_mask is non-empty and the canvas has expanded by more than
/// 10 % relative to @p original_width × @p original_height, tiles whose
/// coverage fraction falls below @p min_tile_coverage_fraction are removed.
///
/// @param image_width               Warped canvas width in pixels.
/// @param image_height              Warped canvas height in pixels.
/// @param tile_size                 Tile side length in pixels.
/// @param overlap_fraction          Overlap between adjacent tiles [0, 0.5].
/// @param canvas_mask               Flat uint8 mask (row-major). Non-zero = covered.
///                                  Pass empty vector to skip coverage filtering.
/// @param original_width            Pre-expansion frame width (for expansion check).
/// @param original_height           Pre-expansion frame height (for expansion check).
/// @param min_tile_coverage_fraction Tiles below this coverage are removed (default 15 %).
///
/// @throws ReconstructionError (coverage_invariant_violated) if after filtering
///         full_support_tiles / num_tiles < 0.85.
TileGridBuildResult build_coverage_filtered_tile_grid(
    int image_width,
    int image_height,
    int tile_size,
    float overlap_fraction,
    const std::vector<uint8_t>& canvas_mask = {},
    int original_width  = 0,
    int original_height = 0,
    float min_tile_coverage_fraction = 0.15f
);

/// Legacy wrapper — kept for backward compatibility.
std::vector<Tile> build_initial_tile_grid(int image_width,
                                          int image_height,
                                          int tile_size,
                                          float overlap_fraction);

} // namespace tile_compile::pipeline
