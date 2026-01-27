#include "tile_compile/core/types.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

namespace tile_compile::metrics {

namespace {

float median_of(std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    const size_t n = v.size();
    const size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    const float hi = v[mid];
    if ((n % 2) == 1) {
        return hi;
    }
    std::nth_element(v.begin(), v.begin() + (mid - 1), v.end());
    const float lo = v[mid - 1];
    return 0.5f * (lo + hi);
}

float stddev_of(const std::vector<float>& v) {
    if (v.size() < 2) return 0.0f;
    double sum = 0.0;
    for (float x : v) sum += static_cast<double>(x);
    const double mean = sum / static_cast<double>(v.size());
    double var = 0.0;
    for (float x : v) {
        const double d = static_cast<double>(x) - mean;
        var += d * d;
    }
    var /= static_cast<double>(v.size());
    if (var <= 0.0) return 0.0f;
    return static_cast<float>(std::sqrt(var));
}

float robust_sigma_mad(std::vector<float>& pixels) {
    if (pixels.empty()) return 0.0f;
    float med = median_of(pixels);
    for (float& x : pixels) x = std::fabs(x - med);
    float mad = median_of(pixels);
    return 1.4826f * mad;
}

cv::Mat1b build_background_mask_sigma_clip(const cv::Mat& frame, float k_sigma, int dilate_radius) {
    const int h = frame.rows;
    const int w = frame.cols;
    cv::Mat1b obj = cv::Mat1b::zeros(h, w);

    std::vector<float> vals;
    vals.reserve(static_cast<size_t>(h) * static_cast<size_t>(w));
    for (int y = 0; y < h; ++y) {
        const float* row = frame.ptr<float>(y);
        for (int x = 0; x < w; ++x) {
            vals.push_back(row[x]);
        }
    }

    float mu = median_of(vals);
    float sigma = robust_sigma_mad(vals);
    if (!(sigma > 0.0f)) {
        return cv::Mat1b(h, w, uint8_t(1));
    }

    const float thr = k_sigma * sigma;
    for (int y = 0; y < h; ++y) {
        const float* row = frame.ptr<float>(y);
        uint8_t* mrow = obj.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) {
            mrow[x] = (std::fabs(row[x] - mu) > thr) ? uint8_t(1) : uint8_t(0);
        }
    }

    cv::Mat1b obj_d;
    const int r = std::max(0, dilate_radius);
    if (r > 0) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * r + 1, 2 * r + 1));
        cv::dilate(obj, obj_d, kernel);
    } else {
        obj_d = obj;
    }

    cv::Mat1b bg = cv::Mat1b::zeros(h, w);
    for (int y = 0; y < h; ++y) {
        const uint8_t* orow = obj_d.ptr<uint8_t>(y);
        uint8_t* brow = bg.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) {
            brow[x] = (orow[x] == 0) ? uint8_t(1) : uint8_t(0);
        }
    }
    return bg;
}

std::vector<float> collect_masked_pixels(const Matrix2Df& frame, const cv::Mat1b& mask) {
    std::vector<float> out;
    out.reserve(static_cast<size_t>(frame.size()));
    for (int y = 0; y < frame.rows(); ++y) {
        const uint8_t* mrow = mask.ptr<uint8_t>(y);
        for (int x = 0; x < frame.cols(); ++x) {
            if (mrow[x] != 0) out.push_back(frame(y, x));
        }
    }
    return out;
}

float masked_median(const Matrix2Df& frame, const cv::Mat1b& mask) {
    std::vector<float> px = collect_masked_pixels(frame, mask);
    if (px.empty()) return 0.0f;
    return median_of(px);
}

float masked_sigma_mad(const Matrix2Df& frame, const cv::Mat1b& mask, float center) {
    std::vector<float> px = collect_masked_pixels(frame, mask);
    if (px.empty()) return 0.0f;
    for (float& x : px) x = std::fabs(x - center);
    float mad = median_of(px);
    return 1.4826f * mad;
}

VectorXf robust_normalize_median_mad(const VectorXf& v) {
    if (v.size() <= 0) return v;
    std::vector<float> vals;
    vals.reserve(static_cast<size_t>(v.size()));
    for (int i = 0; i < v.size(); ++i) vals.push_back(v[i]);
    float med = median_of(vals);
    for (float& x : vals) x = std::fabs(x - med);
    float mad = median_of(vals);
    float sigma_robust = 1.4826f * mad;
    if (!(sigma_robust > 0.0f)) {
        return VectorXf::Zero(v.size());
    }
    return (v.array() - med) / sigma_robust;
}

}

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

    const cv::Mat1b bg_mask = build_background_mask_sigma_clip(cv_used, 3.0f, 3);
    m.background = masked_median(*metrics_frame, bg_mask);
    m.noise = masked_sigma_mad(*metrics_frame, bg_mask, m.background);

    cv::Mat grad_x, grad_y;
    cv::Sobel(cv_used, grad_x, CV_32F, 1, 0);
    cv::Sobel(cv_used, grad_y, CV_32F, 0, 1);
    cv::Mat grad_mag;
    cv::magnitude(grad_x, grad_y, grad_mag);
    cv::Mat grad_mag_sq;
    cv::multiply(grad_mag, grad_mag, grad_mag_sq);
    {
        std::vector<float> gvals;
        gvals.reserve(static_cast<size_t>(grad_mag_sq.rows) * static_cast<size_t>(grad_mag_sq.cols));
        for (int y = 0; y < grad_mag_sq.rows; ++y) {
            const float* row = grad_mag_sq.ptr<float>(y);
            const uint8_t* mrow = bg_mask.ptr<uint8_t>(y);
            for (int x = 0; x < grad_mag_sq.cols; ++x) {
                if (mrow[x] != 0) gvals.push_back(row[x]);
            }
        }
        m.gradient_energy = gvals.empty() ? 0.0f : median_of(gvals);
    }
    
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

    VectorXf bg_n = robust_normalize_median_mad(bg);
    VectorXf noise_n = robust_normalize_median_mad(noise);
    VectorXf grad_n = robust_normalize_median_mad(grad);

    VectorXf Q = w_bg * (-bg_n.array()) + w_noise * (-noise_n.array()) + w_grad * (grad_n.array());

    for (int i = 0; i < n; ++i) {
        float qc = std::min(std::max(Q[i], clamp_lo), clamp_hi);
        weights[i] = std::exp(qc);
    }

    float sum = weights.sum();
    if (std::isfinite(sum) && sum > 1.0e-12f) {
        weights /= sum;
    }

    return weights;
}

} // namespace tile_compile::metrics
