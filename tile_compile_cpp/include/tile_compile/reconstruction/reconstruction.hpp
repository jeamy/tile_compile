#pragma once

#include "tile_compile/core/types.hpp"

#include <vector>

namespace tile_compile::reconstruction {

Matrix2Df reconstruct_tiles(const std::vector<Matrix2Df>& frames,
                            const TileGrid& grid,
                            const std::vector<std::vector<float>>& tile_weights);

} // namespace tile_compile::reconstruction
