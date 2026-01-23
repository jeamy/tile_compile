#pragma once

#include "tile_compile/core/types.hpp"

namespace tile_compile::metrics {

TileMetrics calculate_tile_metrics(const Matrix2Df& tile);

} // namespace tile_compile::metrics
