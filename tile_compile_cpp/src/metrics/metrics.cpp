#include "tile_compile/core/types.hpp"
#include "tile_compile/core/utils.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstring>

namespace tile_compile::metrics {

FrameMetrics calculate_frame_metrics(const Matrix2Df& frame) {
    FrameMetrics m;

    // Avoid large transient allocations (Sobel + gradients) on full-res frames.
    // For GLOBAL_METRICS we only need relative scores, so we can safely compute
    // on a downsampled copy.
    constexpr int kMaxDim = 1024;

    const Matrix2Df* metrics_frame = &frame;
    Matrix2Df down;

    cv::Mat cv_frame(frame.rows(), frame.cols(), CV_32F, const_cast<float*>(frame.data()));
    cv::Mat cv_used = cv_frame;

    int max_dim = std::max(static_cast<int>(frame.rows()), static_cast<int>(frame.cols()));
    if (max_dim > kMaxDim) {
        float scale = static_cast<float>(kMaxDim) / static_cast<float>(max_dim);
        cv::resize(cv_frame, cv_used, cv::Size(), scale, scale, cv::INTER_AREA);

        down = Matrix2Df(cv_used.rows, cv_used.cols);
        if (cv_used.isContinuous()) {
            std::memcpy(down.data(), cv_used.data, static_cast<size_t>(down.size()) * sizeof(float));
        } else {
            for (int r = 0; r < cv_used.rows; ++r) {
                const float* src = cv_used.ptr<float>(r);
                float* dst = down.data() + static_cast<size_t>(r) * static_cast<size_t>(cv_used.cols);
                std::memcpy(dst, src, static_cast<size_t>(cv_used.cols) * sizeof(float));
            }
        }
        metrics_frame = &down;
    }

    m.background = core::compute_median(*metrics_frame);
    m.noise = core::compute_robust_sigma(*metrics_frame);

    cv::Mat grad_x, grad_y;
    cv::Sobel(cv_used, grad_x, CV_32F, 1, 0);
    cv::Sobel(cv_used, grad_y, CV_32F, 0, 1);
    cv::Mat grad_mag;
    cv::magnitude(grad_x, grad_y, grad_mag);
    m.gradient_energy = cv::mean(grad_mag)[0];
    
    m.quality_score = 1.0f;
    return m;
}

VectorXf calculate_global_weights(const std::vector<FrameMetrics>& metrics,
                                   float w_bg, float w_noise, float w_grad,
                                   float clamp_lo, float clamp_hi) {
    int n = metrics.size();
    VectorXf weights(n);
    
    VectorXf bg(n), noise(n), grad(n);
    for (int i = 0; i < n; ++i) {
        bg[i] = metrics[i].background;
        noise[i] = metrics[i].noise;
        grad[i] = metrics[i].gradient_energy;
    }

    auto zscore = [](const VectorXf& v) -> VectorXf {
        if (v.size() <= 0) return v;
        float mean = v.mean();
        VectorXf centered = v.array() - mean;
        float var = centered.array().square().mean();
        float std = std::sqrt(std::max(var, 0.0f));
        if (std < 1.0e-12f) {
            return VectorXf::Zero(v.size());
        }
        return centered / std;
    };

    VectorXf bg_z = zscore(bg);
    VectorXf noise_z = zscore(noise);
    VectorXf grad_z = zscore(grad);

    VectorXf Q = w_bg * (-bg_z.array()) + w_noise * (-noise_z.array()) + w_grad * (grad_z.array());

    for (int i = 0; i < n; ++i) {
        float qc = std::min(std::max(Q[i], clamp_lo), clamp_hi);
        weights[i] = std::exp(qc);
    }

    float sum = weights.sum();
    if (sum > 0.0f) weights /= sum;
    
    return weights;
}

} // namespace tile_compile::metrics
