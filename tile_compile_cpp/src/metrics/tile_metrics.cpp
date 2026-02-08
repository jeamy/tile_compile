#include "tile_compile/core/types.hpp"
#include "tile_compile/core/utils.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace tile_compile::metrics {

namespace {

std::vector<float> collect_pixels(const Matrix2Df& m) {
    std::vector<float> out;
    out.reserve(static_cast<size_t>(m.size()));
    for (Eigen::Index k = 0; k < m.size(); ++k) {
        out.push_back(m.data()[k]);
    }
    return out;
}

float compute_fwhm_proxy(const Matrix2Df& tile) {
    if (tile.size() <= 0) return 0.0f;
    float peak = tile.maxCoeff();
    float half_max = 0.5f * peak;
    int count = 0;
    for (Eigen::Index k = 0; k < tile.size(); ++k) {
        if (tile.data()[k] >= half_max) {
            ++count;
        }
    }
    if (count <= 0) return 0.0f;
    return std::sqrt(static_cast<float>(count) / 3.14159265358979323846f);
}

float compute_roundness_proxy(const Matrix2Df& tile) {
    if (tile.size() <= 0) return 0.0f;
    const float peak = tile.maxCoeff();
    const float eps = std::max(1.0e-12f, std::fabs(peak) * 1.0e-6f);

    std::vector<float> xs;
    std::vector<float> ys;
    xs.reserve(16);
    ys.reserve(16);
    for (int y = 0; y < tile.rows(); ++y) {
        for (int x = 0; x < tile.cols(); ++x) {
            if (std::fabs(tile(y, x) - peak) <= eps) {
                xs.push_back(static_cast<float>(x));
                ys.push_back(static_cast<float>(y));
            }
        }
    }
    if (xs.empty()) return 0.0f;

    auto stddev = [](const std::vector<float>& v) -> float {
        if (v.size() < 2) return 0.0f;
        double sum = 0.0;
        for (float x : v) sum += static_cast<double>(x);
        double mean = sum / static_cast<double>(v.size());
        double var = 0.0;
        for (float x : v) {
            double d = static_cast<double>(x) - mean;
            var += d * d;
        }
        var /= static_cast<double>(v.size());
        return static_cast<float>(std::sqrt(std::max(0.0, var)));
    };

    float sx = stddev(xs);
    float sy = stddev(ys);
    float spread = 0.5f * (sx + sy);
    return 1.0f / (1.0f + spread);
}

float compute_contrast_proxy(const Matrix2Df& tile) {
    if (tile.size() <= 0) return 0.0f;
    float t_max = tile.maxCoeff();
    float t_min = tile.minCoeff();
    return (t_max - t_min) / (t_max + t_min + 1.0e-8f);
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

    m.fwhm = compute_fwhm_proxy(tile);
    m.roundness = compute_roundness_proxy(tile);
    m.contrast = compute_contrast_proxy(tile);

    cv::Mat tile_cv(tile.rows(), tile.cols(), CV_32F, const_cast<float*>(tile.data()));
    cv::Mat bg_cv;
    cv::blur(tile_cv, bg_cv, cv::Size(31, 31), cv::Point(-1, -1), cv::BORDER_REFLECT_101);
    cv::Mat resid = tile_cv - bg_cv;

    std::vector<float> px = collect_pixels(tile);
    float bg0 = core::median_of(px);

    std::vector<float> resid_px;
    resid_px.reserve(static_cast<size_t>(resid.rows) * static_cast<size_t>(resid.cols));
    for (int y = 0; y < resid.rows; ++y) {
        const float* row = resid.ptr<float>(y);
        for (int x = 0; x < resid.cols; ++x) {
            resid_px.push_back(row[x]);
        }
    }
    float sigma0 = core::robust_sigma_mad(resid_px);
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

    float thr = bg0 + 3.0f * sigma0;
    std::vector<float> bg_vals;
    std::vector<float> resid_bg;
    bg_vals.reserve(px.size());
    resid_bg.reserve(resid_px.size());
    for (int y = 0; y < tile.rows(); ++y) {
        const float* trow = tile_cv.ptr<float>(y);
        const float* rrow = resid.ptr<float>(y);
        for (int x = 0; x < tile.cols(); ++x) {
            float tv = trow[x];
            if (tv <= thr) {
                bg_vals.push_back(tv);
                resid_bg.push_back(rrow[x]);
            }
        }
    }
    if (bg_vals.empty()) {
        bg_vals = px;
        resid_bg = resid_px;
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

    try {
        cv::Mat resid_u8;
        cv::Mat tmp;
        cv::normalize(resid, tmp, 0.0, 255.0, cv::NORM_MINMAX);
        tmp.convertTo(resid_u8, CV_8U);
        std::vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(resid_u8, corners, 50, 0.01, 5);
        m.star_count = static_cast<int>(corners.size());
    } catch (...) {
        m.star_count = 0;
    }

    return m;
}

} // namespace tile_compile::metrics
