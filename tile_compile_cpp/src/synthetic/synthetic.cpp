#include "tile_compile/core/types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace tile_compile::synthetic {

std::vector<Matrix2Df> generate_synthetic_frames(
    const std::vector<Matrix2Df>& frames,
    const ClusteringResult& clustering,
    const VectorXf& global_weights,
    int frames_min, int frames_max);

ClusteringResult cluster_frames_by_state(const Matrix2Df& state_vectors,
                                        int min_k,
                                        int max_k) {
    ClusteringResult result;
    const int n = static_cast<int>(state_vectors.rows());
    const int d = static_cast<int>(state_vectors.cols());

    result.n_clusters = 0;
    result.labels = VectorXi();
    result.centers = Matrix2Df();
    result.silhouette_score = 0.0f;
    result.method = "state_vector_quantile";

    if (n <= 0 || d <= 0) {
        return result;
    }

    int k = n / 10;
    k = std::max(min_k, std::min(max_k, k));
    k = std::max(1, std::min(k, n));

    result.n_clusters = k;
    result.labels = VectorXi::Zero(n);
    result.centers = Matrix2Df::Zero(k, d);

    if (k == 1) {
        result.centers.row(0) = state_vectors.colwise().mean();
        return result;
    }

    std::vector<std::pair<float, int>> order;
    order.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        float key = state_vectors(i, 0);
        if (!std::isfinite(key)) key = 0.0f;
        order.push_back({key, i});
    }
    std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    for (int r = 0; r < n; ++r) {
        int label = (r * k) / std::max(1, n);
        if (label >= k) label = k - 1;
        result.labels[order[static_cast<size_t>(r)].second] = label;
    }

    std::vector<int> counts(static_cast<size_t>(k), 0);
    for (int i = 0; i < n; ++i) {
        int lbl = result.labels[i];
        if (lbl < 0 || lbl >= k) continue;
        result.centers.row(lbl) += state_vectors.row(i);
        counts[static_cast<size_t>(lbl)]++;
    }
    for (int c = 0; c < k; ++c) {
        int cnt = counts[static_cast<size_t>(c)];
        if (cnt > 0) {
            result.centers.row(c) /= static_cast<float>(cnt);
        }
    }

    return result;
}

std::vector<Matrix2Df> build_synthetic_frames(const std::vector<Matrix2Df>& frames,
                                             const ClusteringResult& clustering,
                                             const VectorXf& global_weights,
                                             int frames_min,
                                             int frames_max) {
    return generate_synthetic_frames(frames, clustering, global_weights, frames_min, frames_max);
}

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
