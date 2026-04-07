#pragma once

#include "tile_compile/core/types.hpp"

#include <cstdint>
#include <vector>

namespace tile_compile::reconstruction {

/// Classify each tile in @p grid as dead (no frame coverage) or alive.
///
/// A tile is considered dead when the fraction of non-zero pixels in the
/// canvas mask within the tile's bounding box is below
/// @p min_coverage_fraction.
///
/// @param grid                  Tile grid to classify.
/// @param canvas_mask           Flat uint8 mask (row-major), same dimensions as
///                              the warped canvas. Non-zero = covered by at least
///                              one frame.
/// @param canvas_width          Width of the canvas in pixels.
/// @param canvas_height         Height of the canvas in pixels.
/// @param min_coverage_fraction Tiles with coverage below this threshold are
///                              marked dead (default 1 %).
///
/// @returns A bool vector of size grid.tiles.size().  true = dead tile.
std::vector<bool> detect_dead_tiles(
    const TileGrid&              grid,
    const std::vector<uint8_t>&  canvas_mask,
    int                          canvas_width,
    int                          canvas_height,
    float                        min_coverage_fraction = 0.01f
);

} // namespace tile_compile::reconstruction
