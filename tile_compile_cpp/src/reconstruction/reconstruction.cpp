#include "tile_compile/core/types.hpp"

#include <cmath>


namespace tile_compile::reconstruction {

Matrix2Df reconstruct_tiles(const std::vector<Matrix2Df>& frames,
                            const TileGrid& grid,
                            const std::vector<std::vector<float>>& tile_weights) {
    if (frames.empty()) return Matrix2Df();
    
    int h = frames[0].rows();
    int w = frames[0].cols();
    Matrix2Df result = Matrix2Df::Zero(h, w);
    Matrix2Df weight_sum = Matrix2Df::Zero(h, w);

    auto hann_1d = [](int n) -> std::vector<float> {
        std::vector<float> out;
        if (n <= 0) return out;
        out.resize(static_cast<size_t>(n), 1.0f);
        if (n == 1) {
            out[0] = 1.0f;
            return out;
        }
        const float pi = 3.14159265358979323846f;
        for (int i = 0; i < n; ++i) {
            float x = static_cast<float>(i) / static_cast<float>(n - 1);
            out[static_cast<size_t>(i)] = 0.5f * (1.0f - std::cos(2.0f * pi * x));
        }
        return out;
    };
    
    for (size_t t = 0; t < grid.tiles.size(); ++t) {
        const Tile& tile = grid.tiles[t];

        const std::vector<float> wx = hann_1d(tile.width);
        const std::vector<float> wy = hann_1d(tile.height);
        
        for (size_t f = 0; f < frames.size(); ++f) {
            float weight = tile_weights[f][t];
            
            for (int y = tile.y; y < tile.y + tile.height && y < h; ++y) {
                for (int x = tile.x; x < tile.x + tile.width && x < w; ++x) {
                    int ly = y - tile.y;
                    int lx = x - tile.x;
                    if (ly < 0 || lx < 0 || ly >= static_cast<int>(wy.size()) || lx >= static_cast<int>(wx.size())) continue;
                    float win = wy[static_cast<size_t>(ly)] * wx[static_cast<size_t>(lx)];
                    float ww = weight * win;
                    result(y, x) += frames[f](y, x) * ww;
                    weight_sum(y, x) += ww;
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
