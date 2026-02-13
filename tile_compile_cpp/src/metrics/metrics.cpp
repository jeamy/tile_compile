#include "tile_compile/metrics/metrics.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/core/utils.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

namespace tile_compile::metrics {

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

    float mu = core::median_of(vals);
    float sigma = core::robust_sigma_mad(vals);
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

namespace {

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
    return core::median_of(px);
}

float masked_sigma_mad(const Matrix2Df& frame, const cv::Mat1b& mask, float center) {
    std::vector<float> px = collect_masked_pixels(frame, mask);
    if (px.empty()) return 0.0f;
    for (float& x : px) x = std::fabs(x - center);
    float mad = core::median_of(px);
    return 1.4826f * mad;
}

VectorXf robust_normalize_median_mad(const VectorXf& v) {
    if (v.size() <= 0) return v;
    std::vector<float> vals;
    vals.reserve(static_cast<size_t>(v.size()));
    for (int i = 0; i < v.size(); ++i) vals.push_back(v[i]);
    float med = core::median_of(vals);
    for (float& x : vals) x = std::fabs(x - med);
    float mad = core::median_of(vals);
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
        m.gradient_energy = gvals.empty() ? 0.0f : core::median_of(gvals);
    }
    
    m.quality_score = 1.0f;
    return m;
}

VectorXf calculate_global_weights(const std::vector<FrameMetrics>& metrics,
                                   float w_bg, float w_noise, float w_grad,
                                   float clamp_lo, float clamp_hi,
                                   bool adaptive_weights,
                                   float weight_exponent_scale) {
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

    // Methodik 3.1E §3.2: adaptive variance-based weight adjustment
    if (adaptive_weights && n > 2) {
        // 1. Compute variance of each normalized metric across frames
        auto variance_of = [](const VectorXf& v) -> float {
            float mean = v.mean();
            float var = (v.array() - mean).square().mean();
            return var;
        };

        float var_bg = variance_of(bg_n);
        float var_noise = variance_of(noise_n);
        float var_grad = variance_of(grad_n);
        float var_sum = var_bg + var_noise + var_grad;

        if (var_sum > 1e-12f) {
            // 2. Weights proportional to variance (higher variance → more discriminative)
            float a_bg = var_bg / var_sum;
            float a_noise = var_noise / var_sum;
            float a_grad = var_grad / var_sum;

            // 3. Clip to [0.1, 0.7]
            constexpr float kMinW = 0.1f;
            constexpr float kMaxW = 0.7f;
            a_bg = std::min(std::max(a_bg, kMinW), kMaxW);
            a_noise = std::min(std::max(a_noise, kMinW), kMaxW);
            a_grad = std::min(std::max(a_grad, kMinW), kMaxW);

            // 4. Renormalize to sum = 1
            float s = a_bg + a_noise + a_grad;
            w_bg = a_bg / s;
            w_noise = a_noise / s;
            w_grad = a_grad / s;
        }
        // else: all variances zero → keep static weights (fallback)
    }

    VectorXf Q = w_bg * (-bg_n.array()) + w_noise * (-noise_n.array()) + w_grad * (grad_n.array());

    // Apply exponent scale: G_f = exp(k · Q_f) where k = weight_exponent_scale.
    // k > 1 increases differentiation between good and bad frames.
    float k = (weight_exponent_scale > 0.0f) ? weight_exponent_scale : 1.0f;
    for (int i = 0; i < n; ++i) {
        float qc = std::min(std::max(Q[i], clamp_lo), clamp_hi);
        weights[i] = std::exp(k * qc);
    }

    // NOTE: Do NOT normalize weights to sum=1.
    // Methodology v3 defines G_f = exp(Q_f) with clamping; the absolute scale is
    // meaningful for diagnostics and must not depend on the number of frames.
    return weights;
}

// Fit 1D Gaussian sigma from a profile slice through the peak.
// Uses weighted second-moment around the peak for sub-pixel accuracy.
// Returns FWHM = 2.3548 * sigma.  Returns 0 if invalid.
static float fit_1d_fwhm(const float* data, int len, int peak_idx, float bg) {
    if (len < 3 || peak_idx < 0 || peak_idx >= len) return 0.0f;
    double peak_val = static_cast<double>(data[peak_idx]) - static_cast<double>(bg);
    if (peak_val <= 0.0) return 0.0f;

    // Weighted second moment: sigma^2 = sum(w_i * (i - peak)^2) / sum(w_i)
    double sum_w = 0.0, sum_wd2 = 0.0;
    for (int i = 0; i < len; ++i) {
        double w = std::max(0.0, static_cast<double>(data[i]) - static_cast<double>(bg));
        double d = static_cast<double>(i - peak_idx);
        sum_w += w;
        sum_wd2 += w * d * d;
    }
    if (sum_w <= 0.0) return 0.0f;
    double sigma2 = sum_wd2 / sum_w;
    if (sigma2 <= 0.0) return 0.0f;
    double sigma = std::sqrt(sigma2);
    // FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.3548 * sigma
    double fwhm = 2.3548200450309493 * sigma;
    return (fwhm > 0.2 && fwhm < 50.0) ? static_cast<float>(fwhm) : 0.0f;
}

static std::vector<size_t> keep_indices_by_mad_clip(const std::vector<float>& values,
                                                    float sigma_clip) {
    std::vector<size_t> keep;
    if (values.empty()) return keep;

    std::vector<float> tmp = values;
    const float med = core::median_of(tmp);
    for (float& v : tmp) v = std::fabs(v - med);
    const float mad = core::median_of(tmp);
    const float sigma = 1.4826f * mad;

    keep.reserve(values.size());
    if (!(sigma > 1.0e-8f)) {
        for (size_t i = 0; i < values.size(); ++i) keep.push_back(i);
        return keep;
    }

    const float lo = med - sigma_clip * sigma;
    const float hi = med + sigma_clip * sigma;
    for (size_t i = 0; i < values.size(); ++i) {
        const float v = values[i];
        if (std::isfinite(v) && v >= lo && v <= hi) {
            keep.push_back(i);
        }
    }
    return keep;
}

float estimate_fwhm_from_patch(const cv::Mat& patch) {
    if (patch.empty()) return 0.0f;
    std::vector<float> v;
    v.reserve(static_cast<size_t>(patch.rows) * static_cast<size_t>(patch.cols));
    for (int y = 0; y < patch.rows; ++y) {
        const float* row = patch.ptr<float>(y);
        for (int x = 0; x < patch.cols; ++x) {
            v.push_back(row[x]);
        }
    }
    if (v.empty()) return 0.0f;
    float bg = core::median_of(v);
    float sigma = core::robust_sigma_mad(v);

    double maxv = 0.0;
    cv::minMaxLoc(patch, nullptr, &maxv);
    if (!(maxv > 0.0)) return 0.0f;
    if (maxv <= static_cast<double>(bg) + 3.0 * static_cast<double>(sigma))
        return 0.0f;

    cv::Point peak_loc;
    cv::minMaxLoc(patch, nullptr, nullptr, nullptr, &peak_loc);

    // 1D Gaussian fits in X and Y through peak, return geometric mean
    std::vector<float> slice_x(patch.cols), slice_y(patch.rows);
    for (int x = 0; x < patch.cols; ++x)
        slice_x[x] = patch.at<float>(peak_loc.y, x);
    for (int y = 0; y < patch.rows; ++y)
        slice_y[y] = patch.at<float>(y, peak_loc.x);

    float fx = fit_1d_fwhm(slice_x.data(), patch.cols, peak_loc.x, bg);
    float fy = fit_1d_fwhm(slice_y.data(), patch.rows, peak_loc.y, bg);
    if (fx > 0.0f && fy > 0.0f)
        return std::sqrt(fx * fy);
    if (fx > 0.0f) return fx;
    if (fy > 0.0f) return fy;
    return 0.0f;
}

float measure_fwhm_from_image(const Matrix2Df& img, int max_corners,
                              int patch_radius, size_t min_stars) {
    if (img.size() <= 0) return 0.0f;
    const int patch_sz = 2 * patch_radius + 1;
    cv::Mat img_cv(img.rows(), img.cols(), CV_32F,
                   const_cast<float*>(img.data()));
    cv::Mat blur;
    cv::blur(img_cv, blur, cv::Size(31, 31), cv::Point(-1, -1),
             cv::BORDER_REFLECT_101);
    cv::Mat resid = img_cv - blur;

    std::vector<cv::Point2f> corners;
    try {
        cv::goodFeaturesToTrack(resid, corners, max_corners, 0.01, 6);
    } catch (...) {
        corners.clear();
    }

    std::vector<float> fwhms;
    for (const auto& p : corners) {
        int cx = static_cast<int>(std::round(p.x));
        int cy = static_cast<int>(std::round(p.y));
        int x0 = cx - patch_radius;
        int y0 = cy - patch_radius;
        if (x0 < 0 || y0 < 0 || (x0 + patch_sz) > img_cv.cols ||
            (y0 + patch_sz) > img_cv.rows)
            continue;
        cv::Mat patch = img_cv(cv::Rect(x0, y0, patch_sz, patch_sz));
        float f = estimate_fwhm_from_patch(patch);
        if (f > 0.0f && std::isfinite(f))
            fwhms.push_back(f);
    }

    if (fwhms.size() < min_stars) return 0.0f;
    const std::vector<size_t> keep = keep_indices_by_mad_clip(fwhms, 2.5f);
    if (keep.size() >= min_stars) {
        std::vector<float> clipped;
        clipped.reserve(keep.size());
        for (size_t idx : keep) clipped.push_back(fwhms[idx]);
        return core::median_of(clipped);
    }
    return core::median_of(fwhms);
}

// Estimate FWHM separately in X and Y from a patch using 1D Gaussian fits.
// Returns {fwhm_x, fwhm_y}. Both 0 if invalid.
static std::pair<float, float> estimate_fwhm_xy(const cv::Mat& patch) {
    if (patch.empty()) return {0.0f, 0.0f};

    std::vector<float> v;
    v.reserve(static_cast<size_t>(patch.rows) * static_cast<size_t>(patch.cols));
    for (int y = 0; y < patch.rows; ++y) {
        const float* row = patch.ptr<float>(y);
        for (int x = 0; x < patch.cols; ++x)
            v.push_back(row[x]);
    }
    if (v.empty()) return {0.0f, 0.0f};
    float bg = core::median_of(v);
    float sigma = core::robust_sigma_mad(v);

    double maxv = 0.0;
    cv::minMaxLoc(patch, nullptr, &maxv);
    if (!(maxv > 0.0)) return {0.0f, 0.0f};
    if (maxv <= static_cast<double>(bg) + 3.0 * static_cast<double>(sigma))
        return {0.0f, 0.0f};

    cv::Point peak_loc;
    cv::minMaxLoc(patch, nullptr, nullptr, nullptr, &peak_loc);

    // Extract 1D slices through peak
    std::vector<float> slice_x(patch.cols), slice_y(patch.rows);
    for (int x = 0; x < patch.cols; ++x)
        slice_x[x] = patch.at<float>(peak_loc.y, x);
    for (int y = 0; y < patch.rows; ++y)
        slice_y[y] = patch.at<float>(y, peak_loc.x);

    float fx = fit_1d_fwhm(slice_x.data(), patch.cols, peak_loc.x, bg);
    float fy = fit_1d_fwhm(slice_y.data(), patch.rows, peak_loc.y, bg);
    return {fx, fy};
}

FrameStarMetrics measure_frame_stars(const Matrix2Df& img,
                                     int ref_star_count,
                                     int max_corners,
                                     int patch_radius) {
    FrameStarMetrics result{};

    if (img.size() <= 0) return result;
    const int patch_sz = 2 * patch_radius + 1;
    cv::Mat img_cv(img.rows(), img.cols(), CV_32F,
                   const_cast<float*>(img.data()));
    cv::Mat blur;
    cv::blur(img_cv, blur, cv::Size(31, 31), cv::Point(-1, -1),
             cv::BORDER_REFLECT_101);
    cv::Mat resid = img_cv - blur;

    std::vector<cv::Point2f> corners;
    try {
        cv::goodFeaturesToTrack(resid, corners, max_corners, 0.01, 6);
    } catch (...) {
        corners.clear();
    }

    std::vector<float> fwhms, fwhms_x, fwhms_y, roundnesses;
    for (const auto& pt : corners) {
        int cx = static_cast<int>(std::round(pt.x));
        int cy = static_cast<int>(std::round(pt.y));
        int x0 = cx - patch_radius;
        int y0 = cy - patch_radius;
        if (x0 < 0 || y0 < 0 || (x0 + patch_sz) > img_cv.cols ||
            (y0 + patch_sz) > img_cv.rows)
            continue;
        cv::Mat patch = img_cv(cv::Rect(x0, y0, patch_sz, patch_sz));
        auto [fx, fy] = estimate_fwhm_xy(patch);
        if (fx > 0.0f && fy > 0.0f && std::isfinite(fx) && std::isfinite(fy)) {
            fwhms_x.push_back(fx);
            fwhms_y.push_back(fy);
            float f = std::sqrt(fx * fy);  // geometric mean
            fwhms.push_back(f);
            roundnesses.push_back(fy / fx);
        }
    }

    const std::vector<size_t> keep = keep_indices_by_mad_clip(fwhms, 2.5f);
    if (!keep.empty() && keep.size() < fwhms.size()) {
        std::vector<float> f2, fx2, fy2, r2;
        f2.reserve(keep.size());
        fx2.reserve(keep.size());
        fy2.reserve(keep.size());
        r2.reserve(keep.size());
        for (size_t idx : keep) {
            f2.push_back(fwhms[idx]);
            fx2.push_back(fwhms_x[idx]);
            fy2.push_back(fwhms_y[idx]);
            r2.push_back(roundnesses[idx]);
        }
        fwhms.swap(f2);
        fwhms_x.swap(fx2);
        fwhms_y.swap(fy2);
        roundnesses.swap(r2);
    }

    result.star_count = static_cast<int>(fwhms.size());
    if (result.star_count > 0) {
        result.fwhm = core::median_of(fwhms);
        result.fwhm_x = core::median_of(fwhms_x);
        result.fwhm_y = core::median_of(fwhms_y);
        result.roundness = core::median_of(roundnesses);
        if (ref_star_count > 0) {
            result.wfwhm = result.fwhm *
                static_cast<float>(ref_star_count) /
                static_cast<float>(result.star_count);
        } else {
            result.wfwhm = result.fwhm;
        }
    }

    return result;
}

} // namespace tile_compile::metrics
