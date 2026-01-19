#include "tile_compile/core/types.hpp"
#include <random>

namespace tile_compile::clustering {

std::pair<VectorXi, Matrix2Df> kmeans(const Matrix2Df& data, int k, int max_iters, int n_init) {
    int n = data.rows();
    int d = data.cols();
    
    VectorXi best_labels = VectorXi::Zero(n);
    Matrix2Df best_centers = data.topRows(std::min(k, n));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int init = 0; init < n_init; ++init) {
        Matrix2Df centers(k, d);
        std::uniform_int_distribution<> dis(0, n - 1);
        for (int i = 0; i < k; ++i) {
            centers.row(i) = data.row(dis(gen));
        }
        
        VectorXi labels = VectorXi::Zero(n);
        
        for (int iter = 0; iter < max_iters; ++iter) {
            for (int i = 0; i < n; ++i) {
                float min_dist = std::numeric_limits<float>::max();
                for (int j = 0; j < k; ++j) {
                    float dist = (data.row(i) - centers.row(j)).squaredNorm();
                    if (dist < min_dist) {
                        min_dist = dist;
                        labels[i] = j;
                    }
                }
            }
            
            Matrix2Df new_centers = Matrix2Df::Zero(k, d);
            VectorXi counts = VectorXi::Zero(k);
            for (int i = 0; i < n; ++i) {
                new_centers.row(labels[i]) += data.row(i);
                counts[labels[i]]++;
            }
            for (int j = 0; j < k; ++j) {
                if (counts[j] > 0) {
                    new_centers.row(j) /= counts[j];
                }
            }
            centers = new_centers;
        }
        
        best_labels = labels;
        best_centers = centers;
    }
    
    return {best_labels, best_centers};
}

} // namespace tile_compile::clustering
