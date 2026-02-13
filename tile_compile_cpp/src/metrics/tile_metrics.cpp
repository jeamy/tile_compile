#include "tile_compile/core/types.hpp"
#include "tile_compile/core/utils.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace tile_compile::metrics {

namespace {

// Fit 1D Gaussian sigma from a profile slice through the peak.
// Uses weighted second-moment around the peak for sub-pixel accuracy.
// Returns FWHM = 2.3548 * sigma.  Returns 0 if invalid.
// (Same algorithm as metrics.cpp::fit_1d_fwhm)
float fit_1d_fwhm(const float* data, int len, int peak_idx, float bg) {
    if (len < 3 || peak_idx < 0 || peak_idx >= len) return 0.0f;
    double peak_val = static_cast<double>(data[peak_idx]) - static_cast<double>(bg);
    if (peak_val <= 0.0) return 0.0f;

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
    double fwhm = 2.3548200450309493 * sigma;
    return (fwhm > 0.2 && fwhm < 50.0) ? static_cast<float>(fwhm) : 0.0f;
}

struct StarMeasurement {
    float fwhm_x = 0.0f;
    float fwhm_y = 0.0f;
    float fwhm = 0.0f;       // geometric mean
    float roundness = 0.0f;   // fy / fx
    float contrast = 0.0f;    // peak / background
};

// Measure FWHM, roundness, and contrast from a small patch around a star.
// Returns valid=true only if both X and Y fits succeed.
bool measure_star_patch(const cv::Mat& tile_cv, const cv::Point2f& pt,
                        int patch_radius, float tile_bg, float tile_sigma,
                        StarMeasurement& out) {
    int cx = static_cast<int>(std::round(pt.x));
    int cy = static_cast<int>(std::round(pt.y));
    int psz = 2 * patch_radius + 1;
    int x0 = cx - patch_radius;
    int y0 = cy - patch_radius;

    if (x0 < 0 || y0 < 0 || (x0 + psz) > tile_cv.cols ||
        (y0 + psz) > tile_cv.rows)
        return false;

    cv::Mat patch = tile_cv(cv::Rect(x0, y0, psz, psz));

    // Find peak in patch
    double maxv = 0.0;
    cv::Point peak_loc;
    cv::minMaxLoc(patch, nullptr, &maxv, nullptr, &peak_loc);

    // Reject if peak is not significantly above background
    if (maxv <= static_cast<double>(tile_bg) + 3.0 * static_cast<double>(tile_sigma))
        return false;

    // 1D Gaussian fits in X and Y through peak
    std::vector<float> slice_x(psz), slice_y(psz);
    for (int x = 0; x < psz; ++x)
        slice_x[static_cast<size_t>(x)] = patch.at<float>(peak_loc.y, x);
    for (int y = 0; y < psz; ++y)
        slice_y[static_cast<size_t>(y)] = patch.at<float>(y, peak_loc.x);

    float fx = fit_1d_fwhm(slice_x.data(), psz, peak_loc.x, tile_bg);
    float fy = fit_1d_fwhm(slice_y.data(), psz, peak_loc.y, tile_bg);

    if (!(fx > 0.0f) || !(fy > 0.0f))
        return false;

    out.fwhm_x = fx;
    out.fwhm_y = fy;
    out.fwhm = std::sqrt(fx * fy);
    out.roundness = std::min(fx, fy) / std::max(fx, fy);
    float peak_flux = static_cast<float>(maxv);
    out.contrast = (tile_bg > 1e-8f) ? (peak_flux / tile_bg) : 0.0f;
    return true;
}

}

TileMetrics calculate_tile_metrics(const Matrix2Df& tile) {
    TileMetrics m;
    m.fwhm = 0.0f;
    m.roundness = 0.0f;
    m.contrast = 0.0f;
    m.sharpness = 0.0f;
    m.background = 0.0f;
    m.noise = 0.0f;
    m.gradient_energy = 0.0f;
    m.star_count = 0;
    m.type = TileType::STRUCTURE;
    m.quality_score = 0.0f;

    if (tile.size() <= 0) {
        return m;
    }

    cv::Mat tile_cv(tile.rows(), tile.cols(), CV_32F, const_cast<float*>(tile.data()));
    cv::Mat bg_cv;
    cv::blur(tile_cv, bg_cv, cv::Size(31, 31), cv::Point(-1, -1), cv::BORDER_REFLECT_101);
    cv::Mat resid = tile_cv - bg_cv;

    // Single pass: collect tile pixels and residual pixels simultaneously
    const size_t npx = static_cast<size_t>(tile.rows()) * static_cast<size_t>(tile.cols());
    std::vector<float> px, resid_px;
    px.reserve(npx);
    resid_px.reserve(npx);
    for (int y = 0; y < tile.rows(); ++y) {
        const float* trow = tile_cv.ptr<float>(y);
        const float* rrow = resid.ptr<float>(y);
        for (int x = 0; x < tile.cols(); ++x) {
            px.push_back(trow[x]);
            resid_px.push_back(rrow[x]);
        }
    }

    // Use temporary copies for destructive median/MAD operations
    std::vector<float> px_tmp = px;
    float bg0 = core::median_of(px_tmp);

    std::vector<float> resid_tmp = resid_px;
    float sigma0 = core::robust_sigma_mad(resid_tmp);
    if (!(sigma0 > 0.0f)) {
        double sum = 0.0;
        for (float v : resid_px) sum += static_cast<double>(v);
        double mean = resid_px.empty() ? 0.0 : sum / static_cast<double>(resid_px.size());
        double var = 0.0;
        for (float v : resid_px) {
            double d = static_cast<double>(v) - mean;
            var += d * d;
        }
        var = resid_px.empty() ? 0.0 : var / static_cast<double>(resid_px.size());
        sigma0 = static_cast<float>(std::sqrt(std::max(0.0, var)));
    }

    // Threshold-based split using already-collected vectors (no re-read)
    float thr = bg0 + 3.0f * sigma0;
    std::vector<float> bg_vals;
    std::vector<float> resid_bg;
    bg_vals.reserve(npx);
    resid_bg.reserve(npx);
    for (size_t i = 0; i < px.size(); ++i) {
        if (px[i] <= thr) {
            bg_vals.push_back(px[i]);
            resid_bg.push_back(resid_px[i]);
        }
    }
    if (bg_vals.empty()) {
        bg_vals = std::move(px);
        resid_bg = std::move(resid_px);
    }

    m.background = core::median_of(bg_vals);
    m.noise = core::robust_sigma_mad(resid_bg);

    cv::Mat gx, gy;
    cv::Sobel(resid, gx, CV_32F, 1, 0, 3);
    cv::Sobel(resid, gy, CV_32F, 0, 1, 3);
    cv::Mat gx2, gy2, grad_sq;
    cv::multiply(gx, gx, gx2);
    cv::multiply(gy, gy, gy2);
    grad_sq = gx2 + gy2;
    std::vector<float> grad_vals;
    grad_vals.reserve(static_cast<size_t>(grad_sq.rows) * static_cast<size_t>(grad_sq.cols));
    for (int y = 0; y < grad_sq.rows; ++y) {
        const float* row = grad_sq.ptr<float>(y);
        for (int x = 0; x < grad_sq.cols; ++x) {
            grad_vals.push_back(row[x]);
        }
    }
    m.gradient_energy = grad_vals.empty() ? 0.0f : core::median_of(grad_vals);

    // Star-based metrics: detect stars via goodFeaturesToTrack on residual,
    // then measure FWHM (1D Gauss fit), roundness (axis ratio), and contrast
    // (peak/background) per star.  This replaces the old pixel-counting proxies
    // and is consistent with the global metrics.cpp approach.
    constexpr int kPatchRadius = 5; // 11×11 patch per star
    constexpr int kMaxCorners = 50;
    constexpr float kMinQuality = 0.01f;
    constexpr int kMinDist = 5;

    std::vector<cv::Point2f> corners;
    try {
        cv::Mat resid_u8;
        cv::Mat tmp;
        cv::normalize(resid, tmp, 0.0, 255.0, cv::NORM_MINMAX);
        tmp.convertTo(resid_u8, CV_8U);
        cv::goodFeaturesToTrack(resid_u8, corners, kMaxCorners, kMinQuality, kMinDist);
    } catch (...) {
        corners.clear();
    }

    std::vector<float> fwhms, roundnesses, contrasts;
    for (const auto& pt : corners) {
        StarMeasurement sm;
        if (measure_star_patch(tile_cv, pt, kPatchRadius, bg0, sigma0, sm)) {
            fwhms.push_back(sm.fwhm);
            roundnesses.push_back(sm.roundness);
            contrasts.push_back(sm.contrast);
        }
    }

    m.star_count = static_cast<int>(fwhms.size());
    if (!fwhms.empty()) {
        m.fwhm = core::median_of(fwhms);
        m.roundness = core::median_of(roundnesses);
        m.contrast = core::median_of(contrasts);
    }

    return m;
}

} // namespace tile_compile::metrics
