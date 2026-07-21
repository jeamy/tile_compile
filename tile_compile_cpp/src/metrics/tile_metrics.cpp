#include "tile_compile/core/types.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/metrics/psf_fit.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace tile_compile::metrics {

namespace {

struct StarMeasurement {
    float fwhm_x = 0.0f;
    float fwhm_y = 0.0f;
    float fwhm = 0.0f;       // geometric mean
    float roundness = 0.0f;   // fy / fx
    float contrast = 0.0f;    // peak / background
};

struct TileMetricsScratch {
    std::vector<float> px;
    std::vector<float> resid_px;
    std::vector<float> tmp;
    std::vector<float> bg_vals;
    std::vector<float> resid_bg;
    std::vector<float> grad_vals;
    std::vector<cv::Point2f> corners;
    std::vector<float> fwhms;
    std::vector<float> roundnesses;
    std::vector<float> contrasts;
};

// Measure FWHM, roundness, and contrast from a small patch around a star.
// Returns valid=true only if both X and Y fits succeed.
/// @brief Implements measure star patch.
/// @details Part of per-tile metric and PSF estimation helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
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

    const PsfFit2D fit = fit_elliptical_psf_2d(patch, tile_bg);
    float fx = fit.fwhm_major;
    float fy = fit.fwhm_minor;

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

/// @brief Calculates tile metrics impl.
/// @details Part of per-tile metric and PSF estimation helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
TileMetrics calculate_tile_metrics_impl(const Matrix2Df& tile,
                                        TileMetricsScratch& scratch) {
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

    const size_t npx = static_cast<size_t>(tile.rows()) * static_cast<size_t>(tile.cols());
    scratch.px.clear();
    scratch.resid_px.clear();
    scratch.px.reserve(npx);
    scratch.resid_px.reserve(npx);
    float tile_max = std::numeric_limits<float>::lowest();
    for (int y = 0; y < tile.rows(); ++y) {
        const float* trow = tile_cv.ptr<float>(y);
        const float* rrow = resid.ptr<float>(y);
        for (int x = 0; x < tile.cols(); ++x) {
            const float tv = trow[x];
            scratch.px.push_back(tv);
            scratch.resid_px.push_back(rrow[x]);
            tile_max = std::max(tile_max, tv);
        }
    }

    scratch.tmp = scratch.px;
    const float bg0 = core::median_of(scratch.tmp);

    scratch.tmp = scratch.resid_px;
    float sigma0 = core::robust_sigma_mad(scratch.tmp);
    if (!(sigma0 > 0.0f)) {
        cv::Scalar mu, sd;
        cv::meanStdDev(resid, mu, sd);
        sigma0 = static_cast<float>(sd[0]);
    }

    const float thr = bg0 + 3.0f * sigma0;
    scratch.bg_vals.clear();
    scratch.resid_bg.clear();
    scratch.bg_vals.reserve(npx);
    scratch.resid_bg.reserve(npx);
    for (size_t i = 0; i < scratch.px.size(); ++i) {
        if (scratch.px[i] <= thr) {
            scratch.bg_vals.push_back(scratch.px[i]);
            scratch.resid_bg.push_back(scratch.resid_px[i]);
        }
    }
    if (scratch.bg_vals.empty()) {
        scratch.bg_vals = scratch.px;
        scratch.resid_bg = scratch.resid_px;
    }

    scratch.tmp = scratch.bg_vals;
    m.background = core::median_of(scratch.tmp);
    scratch.tmp = scratch.resid_bg;
    m.noise = core::robust_sigma_mad(scratch.tmp);

    cv::Mat gx, gy;
    cv::Sobel(resid, gx, CV_32F, 1, 0, 3);
    cv::Sobel(resid, gy, CV_32F, 0, 1, 3);
    cv::Mat gx2, gy2, grad_sq;
    cv::multiply(gx, gx, gx2);
    cv::multiply(gy, gy, gy2);
    grad_sq = gx2 + gy2;
    scratch.grad_vals.clear();
    scratch.grad_vals.reserve(static_cast<size_t>(grad_sq.rows) *
                              static_cast<size_t>(grad_sq.cols));
    for (int y = 0; y < grad_sq.rows; ++y) {
        const float* row = grad_sq.ptr<float>(y);
        for (int x = 0; x < grad_sq.cols; ++x) {
            scratch.grad_vals.push_back(row[x]);
        }
    }
    if (!scratch.grad_vals.empty()) {
        scratch.tmp = scratch.grad_vals;
        m.gradient_energy = core::median_of(scratch.tmp);
    }

    constexpr int kPatchRadius = 5;
    constexpr int kMaxCorners = 50;
    constexpr float kMinQuality = 0.01f;
    constexpr int kMinDist = 5;

    scratch.corners.clear();
    if (tile_max > bg0 + 3.0f * sigma0) {
        try {
            cv::goodFeaturesToTrack(resid, scratch.corners, kMaxCorners,
                                    kMinQuality, kMinDist);
        } catch (...) {
            scratch.corners.clear();
        }
    }

    scratch.fwhms.clear();
    scratch.roundnesses.clear();
    scratch.contrasts.clear();
    scratch.fwhms.reserve(scratch.corners.size());
    scratch.roundnesses.reserve(scratch.corners.size());
    scratch.contrasts.reserve(scratch.corners.size());
    for (const auto& pt : scratch.corners) {
        StarMeasurement sm;
        if (measure_star_patch(tile_cv, pt, kPatchRadius, bg0, sigma0, sm)) {
            scratch.fwhms.push_back(sm.fwhm);
            scratch.roundnesses.push_back(sm.roundness);
            scratch.contrasts.push_back(sm.contrast);
        }
    }

    m.star_count = static_cast<int>(scratch.fwhms.size());
    if (!scratch.fwhms.empty()) {
        scratch.tmp = scratch.fwhms;
        m.fwhm = core::median_of(scratch.tmp);
        scratch.tmp = scratch.roundnesses;
        m.roundness = core::median_of(scratch.tmp);
        scratch.tmp = scratch.contrasts;
        m.contrast = core::median_of(scratch.tmp);
    }

    return m;
}

}

/// @brief Calculates tile metrics.
/// @details Part of per-tile metric and PSF estimation helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
TileMetrics calculate_tile_metrics(const Matrix2Df& tile) {
    thread_local TileMetricsScratch scratch;
    return calculate_tile_metrics_impl(tile, scratch);
}

} // namespace tile_compile::metrics
