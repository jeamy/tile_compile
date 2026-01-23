#include "tile_compile/core/types.hpp"

namespace tile_compile::reconstruction {

Matrix2Df reconstruct_tiles(const std::vector<Matrix2Df>& frames,
                            const TileGrid& grid,
                            const std::vector<std::vector<float>>& tile_weights) {
    if (frames.empty()) return Matrix2Df();
    
    int h = frames[0].rows();
    int w = frames[0].cols();
    Matrix2Df result = Matrix2Df::Zero(h, w);
    Matrix2Df weight_sum = Matrix2Df::Zero(h, w);
    
    for (size_t t = 0; t < grid.tiles.size(); ++t) {
        const Tile& tile = grid.tiles[t];
        
        for (size_t f = 0; f < frames.size(); ++f) {
            float weight = tile_weights[f][t];
            
            for (int y = tile.y; y < tile.y + tile.height && y < h; ++y) {
                for (int x = tile.x; x < tile.x + tile.width && x < w; ++x) {
                    result(y, x) += frames[f](y, x) * weight;
                    weight_sum(y, x) += weight;
                }
            }
        }
    }
    
    for (int i = 0; i < result.size(); ++i) {
        if (weight_sum.data()[i] > 0) {
            result.data()[i] /= weight_sum.data()[i];
        }
    }
    
    return result;
}

} // namespace tile_compile::reconstruction
