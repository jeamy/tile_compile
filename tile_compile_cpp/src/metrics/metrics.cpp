#include "tile_compile/metrics/metrics.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/core/utils.hpp"
#include <opencv2/opencv.hpp>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include <algorithm>

namespace tile_compile::metrics {

/// @brief Builds background mask sigma clip.
/// @details Part of global frame metric and star/PSF estimation helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
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

/// @brief Implements collect masked pixels.
/// @details Part of global frame metric and star/PSF estimation helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
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

/// @brief Implements masked median.
/// @details Part of global frame metric and star/PSF estimation helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float masked_median(const Matrix2Df& frame, const cv::Mat1b& mask) {
    std::vector<float> px = collect_masked_pixels(frame, mask);
    if (px.empty()) return 0.0f;
    return core::median_of(px);
}

/// @brief Implements masked sigma mad.
/// @details Part of global frame metric and star/PSF estimation helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float masked_sigma_mad(const Matrix2Df& frame, const cv::Mat1b& mask, float center) {
    std::vector<float> px = collect_masked_pixels(frame, mask);
    if (px.empty()) return 0.0f;
    for (float& x : px) x = std::fabs(x - center);
    float mad = core::median_of(px);
    return 1.4826f * mad;
}

/// @brief Implements robust normalize median mad.
/// @details Part of global frame metric and star/PSF estimation helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
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

/// @brief Implements positive pearson correlation.
/// @details Part of global frame metric and star/PSF estimation helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float positive_pearson_correlation(const VectorXf& a, const VectorXf& b) {
    if (a.size() <= 1 || a.size() != b.size()) return 0.0f;

    const float mean_a = a.mean();
    const float mean_b = b.mean();
    const VectorXf da = a.array() - mean_a;
    const VectorXf db = b.array() - mean_b;
    const float var_a = da.array().square().mean();
    const float var_b = db.array().square().mean();
    if (!(var_a > 1.0e-12f) || !(var_b > 1.0e-12f)) return 0.0f;

    const float cov = (da.array() * db.array()).mean();
    const float corr = cov / std::sqrt(var_a * var_b);
    if (!std::isfinite(corr)) return 0.0f;
    return std::max(0.0f, corr);
}

}

/// @brief Calculates frame metrics.
/// @details Part of global frame metric and star/PSF estimation helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
FrameMetrics calculate_frame_metrics(const Matrix2Df& frame,
                                     const std::vector<uint8_t>* frame_valid_mask) {
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

    // Large-scale sky gradient: compare background medians of four quadrants.
    // This captures the additive sky gradient (e.g. light pollution, moon glow)
    // separately from gradient_energy which measures local pixel-scale structure.
    // When frame_valid_mask is provided, quadrant medians only consider pixels
    // that are both background (bg_mask) and frame-valid.  A quadrant with no
    // valid pixels is marked NaN.  When background <= 0 or fewer than four
    // valid quadrants exist, sky_gradient is NaN (invalid) per §1.5.
    {
        const int h2 = metrics_frame->rows() / 2;
        const int w2 = metrics_frame->cols() / 2;
        float q[4] = {0, 0, 0, 0};
        int valid_quadrants = 0;
        for (int qi = 0; qi < 4; ++qi) {
            const int y0 = (qi / 2) * h2;
            const int x0 = (qi % 2) * w2;
            const int y1 = (qi / 2 == 0) ? h2 : metrics_frame->rows();
            const int x1 = (qi % 2 == 0) ? w2 : metrics_frame->cols();
            std::vector<float> qvals;
            qvals.reserve(static_cast<size_t>(y1 - y0) * static_cast<size_t>(x1 - x0));
            for (int y = y0; y < y1; ++y) {
                const float* row = metrics_frame->data() + static_cast<size_t>(y) * static_cast<size_t>(metrics_frame->cols());
                const uint8_t* mrow = bg_mask.ptr<uint8_t>(y);
                for (int x = x0; x < x1; ++x) {
                    if (mrow[x] == 0)
                        continue;
                    if (frame_valid_mask) {
                        const size_t fvm_idx = static_cast<size_t>(y) *
                            static_cast<size_t>(metrics_frame->cols()) +
                            static_cast<size_t>(x);
                        if (fvm_idx >= frame_valid_mask->size() ||
                            (*frame_valid_mask)[fvm_idx] == 0)
                            continue;
                    }
                    qvals.push_back(row[x]);
                }
            }
            if (qvals.empty()) {
                q[qi] = std::numeric_limits<float>::quiet_NaN();
            } else {
                q[qi] = core::median_of(qvals);
                ++valid_quadrants;
            }
        }
        if (m.background > 1e-6f && valid_quadrants >= 4) {
            float qmin = q[0], qmax = q[0];
            for (int qi = 1; qi < 4; ++qi) {
                qmin = std::min(qmin, q[qi]);
                qmax = std::max(qmax, q[qi]);
            }
            m.sky_gradient = (qmax - qmin) / m.background;
        } else {
            m.sky_gradient = std::numeric_limits<float>::quiet_NaN();
        }
    }

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

float median_valid_or(const VectorXf& v, float fallback,
                      bool require_positive) {
    std::vector<float> vals;
    vals.reserve(static_cast<size_t>(v.size()));
    for (int i = 0; i < v.size(); ++i) {
        const float x = v[i];
        if (std::isfinite(x) && (!require_positive || x > 0.0f)) {
            vals.push_back(x);
        }
    }
    return vals.empty() ? fallback : core::median_of(vals);
}

void replace_invalid_with(VectorXf& v, float fallback,
                          bool require_positive) {
    for (int i = 0; i < v.size(); ++i) {
        const float x = v[i];
        if (!std::isfinite(x) || (require_positive && !(x > 0.0f))) {
            v[i] = fallback;
        }
    }
}

VectorXf calculate_global_weights_impl(
    const std::vector<FrameMetrics>& metrics,
    const std::vector<FrameStarMetrics>* star_metrics,
    float w_bg, float w_noise, float w_grad, float w_fwhm,
    float w_roundness, float w_star_count, float clamp_lo, float clamp_hi,
    bool adaptive_weights, float weight_exponent_scale) {
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

    VectorXf fwhm_n = VectorXf::Zero(n);
    VectorXf roundness_error_n = VectorXf::Zero(n);
    VectorXf star_count_n = VectorXf::Zero(n);
    if (star_metrics && static_cast<int>(star_metrics->size()) == n &&
        (w_fwhm > 0.0f || w_roundness > 0.0f || w_star_count > 0.0f)) {
        VectorXf fwhm(n), roundness_error(n), star_count(n);
        for (int i = 0; i < n; ++i) {
            const auto& sm = (*star_metrics)[static_cast<size_t>(i)];
            fwhm[i] = sm.fwhm;
            roundness_error[i] =
                (std::isfinite(sm.roundness) && sm.roundness > 0.0f)
                    ? std::fabs(1.0f - sm.roundness)
                    : std::numeric_limits<float>::quiet_NaN();
            star_count[i] = static_cast<float>(sm.star_count);
        }
        replace_invalid_with(
            fwhm, median_valid_or(fwhm, 0.0f, true), true);
        replace_invalid_with(
            roundness_error, median_valid_or(roundness_error, 0.0f, false),
            false);
        replace_invalid_with(
            star_count, median_valid_or(star_count, 0.0f, false), false);

        fwhm_n = robust_normalize_median_mad(fwhm);
        roundness_error_n = robust_normalize_median_mad(roundness_error);
        star_count_n = robust_normalize_median_mad(star_count);
    }

    // Methodik v3.3.9 §5.3.3: optional adaptive weighting must be based on a
    // deterministic predictive-utility criterion, not merely on Var(z(.)).
    //
    // Utility target:
    // - Re-orient the normalized metrics so "higher is better":
    //   s_bg = -z(background), s_noise = -z(noise), s_grad = z(gradient).
    // - For each metric i, predict a leave-one-out consensus target from the
    //   other two signals using the static weights renormalized over the
    //   remaining metrics.
    // - Utility_i = max(corr(signal_i, target_i), 0)^2.
    //
    // Tie-break / fallback:
    // - If utilities are degenerate or nearly tied, keep the static weights.
    // - Otherwise clip to [0.1, 0.7] and renormalize to sum 1.
    const float base_weight_sum = w_bg + w_noise + w_grad + w_fwhm + w_roundness + w_star_count;
    if (adaptive_weights && n > 2 && base_weight_sum > 1.0e-12f) {
        const std::array<float, 6> static_weights{
            w_bg / base_weight_sum,
            w_noise / base_weight_sum,
            w_grad / base_weight_sum,
            w_fwhm / base_weight_sum,
            w_roundness / base_weight_sum,
            w_star_count / base_weight_sum};
        const std::array<VectorXf, 6> signals{-bg_n, -noise_n, grad_n, -fwhm_n, -roundness_error_n, star_count_n};
        std::array<float, 3> utility{0.0f, 0.0f, 0.0f};

        for (int i = 0; i < 3; ++i) {
            float other_weight_sum = 0.0f;
            for (int j = 0; j < 3; ++j) {
                if (j == i) continue;
                other_weight_sum += static_weights[static_cast<size_t>(j)];
            }
            if (!(other_weight_sum > 1.0e-12f)) continue;

            VectorXf target = VectorXf::Zero(n);
            for (int j = 0; j < 3; ++j) {
                if (j == i) continue;
                const float weight =
                    static_weights[static_cast<size_t>(j)] / other_weight_sum;
                target += weight * signals[static_cast<size_t>(j)];
            }

            const float corr = positive_pearson_correlation(
                signals[static_cast<size_t>(i)], target);
            utility[static_cast<size_t>(i)] = corr * corr;
        }

        const float utility_sum = utility[0] + utility[1] + utility[2];
        const float utility_min = std::min({utility[0], utility[1], utility[2]});
        const float utility_max = std::max({utility[0], utility[1], utility[2]});

        if (utility_sum > 1.0e-12f && (utility_max - utility_min) > 1.0e-3f) {
            float a_bg = utility[0] / utility_sum;
            float a_noise = utility[1] / utility_sum;
            float a_grad = utility[2] / utility_sum;

            constexpr float kMinW = 0.1f;
            constexpr float kMaxW = 0.7f;
            a_bg = std::min(std::max(a_bg, kMinW), kMaxW);
            a_noise = std::min(std::max(a_noise, kMinW), kMaxW);
            a_grad = std::min(std::max(a_grad, kMinW), kMaxW);

            float s = a_bg + a_noise + a_grad;
            w_bg = (a_bg / s) * base_weight_sum;
            w_noise = (a_noise / s) * base_weight_sum;
            w_grad = (a_grad / s) * base_weight_sum;
        }
        // else: degenerate / near-tied utilities → keep static defaults
    }

    VectorXf Q = VectorXf::Zero(n);
    Q += w_bg * (-bg_n.array()).matrix();
    Q += w_noise * (-noise_n.array()).matrix();
    Q += w_grad * grad_n;
    Q += w_fwhm * (-fwhm_n.array()).matrix();
    Q += w_roundness * (-roundness_error_n.array()).matrix();
    Q += w_star_count * star_count_n;

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

/// @brief Calculates global weights.
/// @details Part of global frame metric and star/PSF estimation helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
VectorXf calculate_global_weights(const std::vector<FrameMetrics>& metrics,
                                   float w_bg, float w_noise, float w_grad,
                                   float clamp_lo, float clamp_hi,
                                   bool adaptive_weights,
                                   float weight_exponent_scale) {
    return calculate_global_weights_impl(metrics, nullptr, w_bg, w_noise,
                                         w_grad, 0.0f, 0.0f, 0.0f,
                                         clamp_lo, clamp_hi, adaptive_weights,
                                         weight_exponent_scale);
}

VectorXf calculate_global_weights_with_stars(
    const std::vector<FrameMetrics>& metrics,
    const std::vector<FrameStarMetrics>& star_metrics,
    float w_bg, float w_noise, float w_grad,
    float w_fwhm, float w_roundness, float w_star_count,
    float clamp_lo, float clamp_hi, bool adaptive_weights,
    float weight_exponent_scale) {
    return calculate_global_weights_impl(
        metrics, &star_metrics, w_bg, w_noise, w_grad, w_fwhm, w_roundness,
        w_star_count, clamp_lo, clamp_hi, adaptive_weights,
        weight_exponent_scale);
}

struct PsfFit2D {
    float fwhm_major = 0.0f;
    float fwhm_minor = 0.0f;
};

// Fit an elliptical 2D Gaussian proxy using weighted second central moments.
// The principal-axis sigmas come from the covariance eigenvalues.
/// @brief Implements fit elliptical psf 2d.
/// @details Part of global frame metric and star/PSF estimation helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static PsfFit2D fit_elliptical_psf_2d(const cv::Mat& patch, float bg) {
    PsfFit2D out;
    if (patch.empty()) return out;

    constexpr double kFwhmScale = 2.3548200450309493;  // 2*sqrt(2*ln(2))

    double sum_w = 0.0;
    double mx = 0.0;
    double my = 0.0;

    for (int y = 0; y < patch.rows; ++y) {
        const float* row = patch.ptr<float>(y);
        for (int x = 0; x < patch.cols; ++x) {
            const double w = std::max(0.0, static_cast<double>(row[x]) - static_cast<double>(bg));
            sum_w += w;
            mx += w * static_cast<double>(x);
            my += w * static_cast<double>(y);
        }
    }
    if (!(sum_w > 0.0)) return out;

    mx /= sum_w;
    my /= sum_w;

    double cxx = 0.0;
    double cyy = 0.0;
    double cxy = 0.0;
    for (int y = 0; y < patch.rows; ++y) {
        const float* row = patch.ptr<float>(y);
        for (int x = 0; x < patch.cols; ++x) {
            const double w = std::max(0.0, static_cast<double>(row[x]) - static_cast<double>(bg));
            if (!(w > 0.0)) continue;
            const double dx = static_cast<double>(x) - mx;
            const double dy = static_cast<double>(y) - my;
            cxx += w * dx * dx;
            cyy += w * dy * dy;
            cxy += w * dx * dy;
        }
    }

    cxx /= sum_w;
    cyy /= sum_w;
    cxy /= sum_w;

    const double tr = cxx + cyy;
    const double det = cxx * cyy - cxy * cxy;
    const double disc = std::max(0.0, tr * tr - 4.0 * det);
    const double root = std::sqrt(disc);
    const double lambda_major = 0.5 * (tr + root);
    const double lambda_minor = 0.5 * (tr - root);

    if (!(lambda_major > 0.0) || !(lambda_minor > 0.0)) return out;

    const double fwhm_major = kFwhmScale * std::sqrt(lambda_major);
    const double fwhm_minor = kFwhmScale * std::sqrt(lambda_minor);
    if (!(fwhm_major > 0.2 && fwhm_major < 50.0 &&
          fwhm_minor > 0.2 && fwhm_minor < 50.0)) {
        return out;
    }

    out.fwhm_major = static_cast<float>(fwhm_major);
    out.fwhm_minor = static_cast<float>(fwhm_minor);
    return out;
}

/// @brief Implements keep indices by mad clip.
/// @details Part of global frame metric and star/PSF estimation helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
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

/// @brief Estimates fwhm from patch.
/// @details Part of global frame metric and star/PSF estimation helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
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

    const PsfFit2D fit = fit_elliptical_psf_2d(patch, bg);
    const float fx = fit.fwhm_major;
    const float fy = fit.fwhm_minor;
    if (fx > 0.0f && fy > 0.0f)
        return std::sqrt(fx * fy);
    if (fx > 0.0f) return fx;
    if (fy > 0.0f) return fy;
    return 0.0f;
}

/// @brief Implements measure fwhm from image.
/// @details Part of global frame metric and star/PSF estimation helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
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

// Estimate FWHM along principal ellipse axes from a patch.
// Returns {fwhm_major, fwhm_minor}. Both 0 if invalid.
/// @brief Estimates fwhm xy.
/// @details Part of global frame metric and star/PSF estimation helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
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

    const PsfFit2D fit = fit_elliptical_psf_2d(patch, bg);
    return {fit.fwhm_major, fit.fwhm_minor};
}

/// @brief Implements measure frame stars.
/// @details Part of global frame metric and star/PSF estimation helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
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
