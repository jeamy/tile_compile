#include "tile_compile/core/types.hpp"
#include "tile_compile/core/utils.hpp"
#include <opencv2/opencv.hpp>

namespace tile_compile::metrics {

FrameMetrics calculate_frame_metrics(const Matrix2Df& frame) {
    FrameMetrics m;
    m.background = core::compute_median(frame);
    m.noise = core::compute_robust_sigma(frame);
    
    cv::Mat cv_frame(frame.rows(), frame.cols(), CV_32F, const_cast<float*>(frame.data()));
    cv::Mat grad_x, grad_y;
    cv::Sobel(cv_frame, grad_x, CV_32F, 1, 0);
    cv::Sobel(cv_frame, grad_y, CV_32F, 0, 1);
    cv::Mat grad_mag;
    cv::magnitude(grad_x, grad_y, grad_mag);
    m.gradient_energy = cv::mean(grad_mag)[0];
    
    m.quality_score = 1.0f;
    return m;
}

VectorXf calculate_global_weights(const std::vector<FrameMetrics>& metrics,
                                   float w_bg, float w_noise, float w_grad) {
    int n = metrics.size();
    VectorXf weights(n);
    
    VectorXf bg(n), noise(n), grad(n);
    for (int i = 0; i < n; ++i) {
        bg[i] = metrics[i].background;
        noise[i] = metrics[i].noise;
        grad[i] = metrics[i].gradient_energy;
    }
    
    auto normalize = [](VectorXf& v) {
        float vmin = v.minCoeff();
        float vmax = v.maxCoeff();
        if (vmax > vmin) v = (v.array() - vmin) / (vmax - vmin);
    };
    
    normalize(bg);
    normalize(noise);
    normalize(grad);
    
    for (int i = 0; i < n; ++i) {
        weights[i] = w_bg * (1.0f - bg[i]) + w_noise * (1.0f - noise[i]) + w_grad * grad[i];
    }
    
    float sum = weights.sum();
    if (sum > 0) weights /= sum;
    
    return weights;
}

} // namespace tile_compile::metrics
