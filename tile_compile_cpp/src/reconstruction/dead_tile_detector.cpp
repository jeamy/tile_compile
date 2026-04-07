#include "tile_compile/reconstruction/dead_tile_detector.hpp"

#include <algorithm>

namespace tile_compile::reconstruction {

std::vector<bool> detect_dead_tiles(
    const TileGrid&             grid,
    const std::vector<uint8_t>& canvas_mask,
    int                         canvas_width,
    int                         canvas_height,
    float                       min_coverage_fraction)
{
    const size_t n = grid.tiles.size();
    std::vector<bool> dead_mask(n, false);

    // Empty mask means no coverage information — treat all tiles as alive.
    if (canvas_mask.empty() || canvas_width <= 0 || canvas_height <= 0) {
        return dead_mask;
    }

    for (size_t ti = 0; ti < n; ++ti) {
        const Tile& t = grid.tiles[ti];

        // Clamp tile bounds to canvas dimensions.
        const int x0 = std::max(0, t.x);
        const int y0 = std::max(0, t.y);
        const int x1 = std::min(canvas_width,  t.x + t.width);
        const int y1 = std::min(canvas_height, t.y + t.height);

        if (x1 <= x0 || y1 <= y0) {
            // Tile is entirely outside the canvas — definitely dead.
            dead_mask[ti] = true;
            continue;
        }

        const int total_pixels = (x1 - x0) * (y1 - y0);
        int covered = 0;
        for (int y = y0; y < y1; ++y) {
            const uint8_t* row = canvas_mask.data() + y * canvas_width;
            for (int x = x0; x < x1; ++x) {
                if (row[x] != 0) ++covered;
            }
        }

        const float coverage = static_cast<float>(covered) /
                               static_cast<float>(total_pixels);
        dead_mask[ti] = (coverage < min_coverage_fraction);
    }

    return dead_mask;
}

} // namespace tile_compile::reconstruction
