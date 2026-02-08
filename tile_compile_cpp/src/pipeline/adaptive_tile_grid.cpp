#include "tile_compile/pipeline/adaptive_tile_grid.hpp"

#include <algorithm>
#include <cmath>

namespace tile_compile::pipeline {

std::vector<Tile> build_initial_tile_grid(int image_width,
                                         int image_height,
                                         int tile_size,
                                         float overlap_fraction) {
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

    if (!xs.empty() && xs.back() + tile_size < image_width) xs.push_back(image_width - tile_size);
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

} // namespace tile_compile::pipeline
