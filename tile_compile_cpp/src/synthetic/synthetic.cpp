#include "tile_compile/core/types.hpp"

namespace tile_compile::synthetic {

std::vector<Matrix2Df> generate_synthetic_frames(
    const std::vector<Matrix2Df>& frames,
    const ClusteringResult& clustering,
    const VectorXf& global_weights,
    int frames_min, int frames_max) {
    
    std::vector<Matrix2Df> synthetic;
    
    for (int c = 0; c < clustering.n_clusters; ++c) {
        Matrix2Df sum = Matrix2Df::Zero(frames[0].rows(), frames[0].cols());
        float weight_sum = 0.0f;
        int count = 0;
        
        for (size_t f = 0; f < frames.size(); ++f) {
            if (clustering.labels[f] == c) {
                float w = global_weights[f];
                sum += frames[f] * w;
                weight_sum += w;
                count++;
            }
        }
        
        if (count >= frames_min && weight_sum > 0) {
            synthetic.push_back(sum / weight_sum);
        }
    }
    
    while (synthetic.size() > static_cast<size_t>(frames_max)) {
        synthetic.pop_back();
    }
    
    return synthetic;
}

} // namespace tile_compile::synthetic
