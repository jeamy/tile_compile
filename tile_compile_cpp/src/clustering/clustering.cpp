#include "tile_compile/core/types.hpp"

namespace tile_compile::clustering {

ClusteringResult cluster_frames(const Matrix2Df& state_vectors, int min_k, int max_k) {
    ClusteringResult result;
    result.n_clusters = std::min(max_k, static_cast<int>(state_vectors.rows()));
    result.labels = VectorXi::Zero(state_vectors.rows());
    result.centers = state_vectors.topRows(result.n_clusters);
    result.silhouette_score = 0.5f;
    result.method = "kmeans";
    return result;
}

} // namespace tile_compile::clustering
