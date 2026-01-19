#include "tile_compile/core/types.hpp"

namespace tile_compile::metrics {

TileMetrics calculate_tile_metrics(const Matrix2Df& tile) {
    TileMetrics m;
    m.fwhm = 3.0f;
    m.roundness = 0.9f;
    m.contrast = tile.maxCoeff() - tile.minCoeff();
    m.sharpness = 1.0f;
    m.quality_score = 1.0f;
    return m;
}

} // namespace tile_compile::metrics
