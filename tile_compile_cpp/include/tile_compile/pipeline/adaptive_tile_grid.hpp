#pragma once

#include "tile_compile/core/types.hpp"
#include <vector>

namespace tile_compile::pipeline {

std::vector<Tile> build_initial_tile_grid(int image_width,
                                         int image_height,
                                         int tile_size,
                                         float overlap_fraction);

} // namespace tile_compile::pipeline
