#include "tile_compile/pipeline/adaptive_tile_grid.hpp"
#include "tile_compile/core/errors.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace tile_compile::pipeline {

namespace {

// Build the raw tile list (no coverage filtering).
/// @brief Creates raw tiles.
/// @details Part of adaptive tile-grid construction based on frame size and configuration; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::vector<Tile> make_raw_tiles(int image_width, int image_height,
                                 int tile_size, float overlap_fraction) {
    std::vector<Tile> tiles;
    if (image_width <= 0 || image_height <= 0 || tile_size <= 0) return tiles;

    if (tile_size > image_width || tile_size > image_height) {
        tiles.push_back(Tile{0, 0, image_width, image_height, 0, 0});
        return tiles;
    }

    overlap_fraction = std::min(std::max(overlap_fraction, 0.0f), 0.5f);
    int overlap_px = static_cast<int>(std::floor(overlap_fraction * static_cast<float>(tile_size)));
    int step = std::max(1, tile_size - overlap_px);

    std::vector<int> xs;
    std::vector<int> ys;
    for (int x = 0; x <= image_width - tile_size; x += step) xs.push_back(x);
    for (int y = 0; y <= image_height - tile_size; y += step) ys.push_back(y);

    if (!xs.empty() && xs.back() + tile_size < image_width)  xs.push_back(image_width  - tile_size);
    if (!ys.empty() && ys.back() + tile_size < image_height) ys.push_back(image_height - tile_size);

    int row = 0;
    for (int y0 : ys) {
        int col = 0;
        for (int x0 : xs) {
            tiles.push_back(Tile{x0, y0, tile_size, tile_size, row, col});
            ++col;
        }
        ++row;
    }
    return tiles;
}

// Compute coverage fraction for a single tile against the canvas mask.
/// @brief Implements tile coverage.
/// @details Part of adaptive tile-grid construction based on frame size and configuration; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float tile_coverage(const Tile& t,
                    const std::vector<uint8_t>& mask,
                    int canvas_width, int canvas_height) {
    const int x0 = std::max(0, t.x);
    const int y0 = std::max(0, t.y);
    const int x1 = std::min(canvas_width,  t.x + t.width);
    const int y1 = std::min(canvas_height, t.y + t.height);
    if (x1 <= x0 || y1 <= y0) return 0.0f;

    const int total = (x1 - x0) * (y1 - y0);
    int covered = 0;
    for (int y = y0; y < y1; ++y) {
        const uint8_t* row = mask.data() + y * canvas_width;
        for (int x = x0; x < x1; ++x)
            if (row[x]) ++covered;
    }
    return static_cast<float>(covered) / static_cast<float>(total);
}

} // namespace

/// @brief Builds coverage filtered tile grid.
/// @details Part of adaptive tile-grid construction based on frame size and configuration; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
TileGridBuildResult build_coverage_filtered_tile_grid(
    int image_width,
    int image_height,
    int tile_size,
    float overlap_fraction,
    const std::vector<uint8_t>& canvas_mask,
    int original_width,
    int original_height,
    float min_tile_coverage_fraction)
{
    TileGridBuildResult result;
    auto raw_tiles = make_raw_tiles(image_width, image_height, tile_size, overlap_fraction);

    // Determine grid dimensions from raw tiles.
    int grid_rows = 0, grid_cols = 0;
    for (const auto& t : raw_tiles) {
        grid_rows = std::max(grid_rows, t.row + 1);
        grid_cols = std::max(grid_cols, t.col + 1);
    }

    // Check whether canvas has expanded enough to warrant coverage filtering.
    const bool has_mask = !canvas_mask.empty() && image_width > 0 && image_height > 0;
    const bool canvas_expanded = (original_width > 0 && original_height > 0) &&
        (static_cast<float>(image_width * image_height) >
         static_cast<float>(original_width * original_height) * 1.10f);

    if (!has_mask || !canvas_expanded) {
        // No filtering — return all tiles as-is.
        result.grid.tile_size        = tile_size;
        result.grid.overlap_fraction = overlap_fraction;
        result.grid.rows             = grid_rows;
        result.grid.cols             = grid_cols;
        result.grid.tiles            = std::move(raw_tiles);
        result.coverage_filtered_tiles = 0;
        result.coverage_fraction       = 1.0f;
        return result;
    }

    // Compute per-tile coverage and filter.
    long long total_mask_pixels = 0;
    for (uint8_t v : canvas_mask) if (v) ++total_mask_pixels;
    result.coverage_fraction =
        static_cast<float>(total_mask_pixels) /
        static_cast<float>(image_width * image_height);

    std::vector<Tile> kept;
    kept.reserve(raw_tiles.size());
    int filtered = 0;
    for (auto& t : raw_tiles) {
        const float cov = tile_coverage(t, canvas_mask, image_width, image_height);
        t.coverage_fraction = cov;
        t.is_dead           = (cov < min_tile_coverage_fraction);
        if (!t.is_dead) {
            kept.push_back(t);
        } else {
            ++filtered;
        }
    }

    // Invariant: at least 85 % of kept tiles must be "full-support" (cov >= 0.85).
    if (!kept.empty()) {
        int full_support = 0;
        for (const auto& t : kept)
            if (t.coverage_fraction >= 0.85f) ++full_support;
        const float ratio = static_cast<float>(full_support) / static_cast<float>(kept.size());
        if (ratio < 0.85f) {
            throw ReconstructionError(
                ReconstructionError::Code::coverage_invariant_violated,
                "Coverage-filtered grid has only " +
                std::to_string(static_cast<int>(ratio * 100)) +
                " % full-support tiles (minimum 85 % required)");
        }
    }

    result.grid.tile_size              = tile_size;
    result.grid.overlap_fraction       = overlap_fraction;
    result.grid.rows                   = grid_rows;
    result.grid.cols                   = grid_cols;
    result.grid.tiles                  = std::move(kept);
    result.grid.coverage_filtered_tiles = filtered;
    result.coverage_filtered_tiles     = filtered;
    return result;
}

/// @brief Builds initial tile grid.
/// @details Part of adaptive tile-grid construction based on frame size and configuration; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::vector<Tile> build_initial_tile_grid(int image_width,
                                          int image_height,
                                          int tile_size,
                                          float overlap_fraction) {
    return build_coverage_filtered_tile_grid(
               image_width, image_height, tile_size, overlap_fraction)
           .grid.tiles;
}

} // namespace tile_compile::pipeline
