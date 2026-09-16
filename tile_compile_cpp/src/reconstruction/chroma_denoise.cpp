#include "tile_compile/reconstruction/chroma_denoise.hpp"
#include "tile_compile/core/utils.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace tile_compile::reconstruction {
namespace {
static float robust_sigma_mad_from_mat(const cv::Mat& m) {
    if (m.empty()) return 0.0f;
    std::vector<float> vals;
    vals.reserve(m.total());
    for (int y = 0; y < m.rows; ++y) {
        const float* row = m.ptr<float>(y);
        vals.insert(vals.end(), row, row + m.cols);
    }
    return tile_compile::core::robust_sigma_mad(vals);
}

static float percentile_from_mat(const cv::Mat& m, float p) {
    if (m.empty()) return 0.0f;
    std::vector<float> vals;
    vals.reserve(m.total());
    for (int y = 0; y < m.rows; ++y) {
        const float* row = m.ptr<float>(y);
        vals.insert(vals.end(), row, row + m.cols);
    }
    if (vals.empty()) return 0.0f;
    return tile_compile::core::percentile_of(vals, p);
}

/// @brief Implements quantize to step.
/// @details Part of tile reconstruction, sigma clipping, overlap-add, and synthetic stacking helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void rgb_to_chroma_space(const cv::Mat& R, const cv::Mat& G, const cv::Mat& B,
                         const std::string& color_space,
                         cv::Mat& Y, cv::Mat& C1, cv::Mat& C2) {
    if (color_space == "opponent_linear") {
        Y = (R + G + B) / 3.0f;
        C1 = R - G;
        C2 = B - G;
        return;
    }

    Y = 0.25f * R + 0.5f * G + 0.25f * B;
    C1 = B - Y;
    C2 = R - Y;
}

/// @brief Implements chroma space to rgb.
/// @details Part of tile reconstruction, sigma clipping, overlap-add, and synthetic stacking helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void chroma_space_to_rgb(const cv::Mat& Y, const cv::Mat& C1, const cv::Mat& C2,
                         const std::string& color_space,
                         cv::Mat& R, cv::Mat& G, cv::Mat& B) {
    if (color_space == "opponent_linear") {
        G = Y - (C1 + C2) / 3.0f;
        R = G + C1;
        B = G + C2;
        return;
    }

    R = Y + C2;
    B = Y + C1;
    G = 2.0f * Y - 0.5f * (R + B);
}

/// @brief Implements soft threshold signed.
/// @details Part of tile reconstruction, sigma clipping, overlap-add, and synthetic stacking helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
cv::Mat soft_threshold_signed(const cv::Mat& src, float tau) {
    if (!(tau > 0.0f)) return src.clone();
    cv::Mat abs_src = cv::abs(src);
    cv::Mat shrunk;
    cv::subtract(abs_src, tau, shrunk);
    cv::threshold(shrunk, shrunk, 0.0, 0.0, cv::THRESH_TOZERO);

    cv::Mat neg_mask;
    cv::compare(src, 0.0f, neg_mask, cv::CMP_LT);
    cv::Mat neg_shrunk;
    cv::subtract(cv::Scalar(0.0f), shrunk, neg_shrunk);
    neg_shrunk.copyTo(shrunk, neg_mask);
    return shrunk;
}

/// @brief Builds protection mask.
/// @details Part of tile reconstruction, sigma clipping, overlap-add, and synthetic stacking helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
cv::Mat build_protection_mask(const cv::Mat& y,
                              const config::ChromaDenoiseConfig& cfg) {
    cv::Mat mask = cv::Mat::zeros(y.size(), CV_32F);

    if (cfg.star_protection.enabled) {
        const float sigma = robust_sigma_mad_from_mat(y);
        cv::Scalar mean_y;
        cv::Scalar std_y;
        cv::meanStdDev(y, mean_y, std_y);
        const float med_like = static_cast<float>(mean_y[0]);
        const float thr = med_like + cfg.star_protection.threshold_sigma * (sigma + 1.0e-6f);
        cv::Mat stars;
        cv::threshold(y, stars, thr, 1.0, cv::THRESH_BINARY);
        stars.convertTo(stars, CV_32F);
        if (cfg.star_protection.dilate_px > 0) {
            const int k = std::max(1, cfg.star_protection.dilate_px * 2 + 1);
            cv::Mat ker = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
            cv::dilate(stars, stars, ker);
        }
        cv::max(mask, stars, mask);
    }

    if (cfg.structure_protection.enabled) {
        cv::Mat gx, gy, mag;
        cv::Sobel(y, gx, CV_32F, 1, 0, 3);
        cv::Sobel(y, gy, CV_32F, 0, 1, 3);
        cv::magnitude(gx, gy, mag);
        const float p = percentile_from_mat(mag, cfg.structure_protection.gradient_percentile);
        cv::Mat structures;
        cv::threshold(mag, structures, p, 1.0, cv::THRESH_BINARY);
        structures.convertTo(structures, CV_32F);
        cv::max(mask, structures, mask);
    }

    if (cfg.extended_source_protection.enabled) {
        // Heavily blur Y to suppress point sources; only extended smooth emission
        // survives at this scale (sigma ≈ 2 % of image short axis).
        const float blur_sigma = std::max(5.0f, static_cast<float>(
            std::min(y.rows, y.cols)) * 0.02f);
        cv::Mat y_smooth;
        cv::GaussianBlur(y, y_smooth, cv::Size(0, 0), blur_sigma, blur_sigma,
                         cv::BORDER_REFLECT_101);

        // Estimate sky stats from the lowest-brightness half of the smoothed luma.
        const float p50 = percentile_from_mat(y_smooth, 50.0f);
        std::vector<float> sky_vals;
        sky_vals.reserve(static_cast<size_t>(y_smooth.total()));
        for (int row = 0; row < y_smooth.rows; ++row) {
            const float* ptr = y_smooth.ptr<float>(row);
            for (int col = 0; col < y_smooth.cols; ++col)
                if (ptr[col] <= p50) sky_vals.push_back(ptr[col]);
        }
        const float sky_sigma = tile_compile::core::robust_sigma_mad(sky_vals);
        const float sky_med   = sky_vals.empty() ? p50 :
            tile_compile::core::percentile_of(sky_vals, 50.0f);
        const float thr = sky_med + cfg.extended_source_protection.luma_sigma * sky_sigma;

        cv::Mat ext_src;
        cv::threshold(y_smooth, ext_src, static_cast<double>(thr), 1.0, cv::THRESH_BINARY);
        ext_src.convertTo(ext_src, CV_32F);
        if (cfg.extended_source_protection.dilate_px > 0) {
            const int k = std::max(1, cfg.extended_source_protection.dilate_px * 2 + 1);
            cv::Mat ker = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
            cv::dilate(ext_src, ext_src, ker);
        }
        cv::max(mask, ext_src, mask);
    }

    cv::GaussianBlur(mask, mask, cv::Size(0, 0), 1.0, 1.0, cv::BORDER_REFLECT_101);
    cv::min(mask, 1.0, mask);
    cv::max(mask, 0.0, mask);
    return mask;
}

/// @brief Implements denoise chroma plane inplace.
/// @details Part of tile reconstruction, sigma clipping, overlap-add, and synthetic stacking helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void denoise_chroma_plane_inplace(cv::Mat& c,
                                  const config::ChromaDenoiseConfig& cfg) {
    if (cfg.chroma_wavelet.enabled) {
        cv::Mat cur = c.clone();
        const int levels = std::max(1, cfg.chroma_wavelet.levels);
        for (int lvl = 0; lvl < levels; ++lvl) {
            const double sigma = std::pow(2.0, static_cast<double>(lvl)) * 0.75;
            cv::Mat low;
            cv::GaussianBlur(cur, low, cv::Size(0, 0), sigma, sigma,
                             cv::BORDER_REFLECT_101);
            cv::Mat detail = cur - low;
            const float sigma_n = robust_sigma_mad_from_mat(detail);
            const float tau = cfg.chroma_wavelet.threshold_scale *
                              cfg.chroma_wavelet.soft_k * sigma_n;
            cv::Mat shrunk = soft_threshold_signed(detail, tau);
            cur = low + shrunk;
        }
        c = cur;
    }

    if (cfg.chroma_bilateral.enabled) {
        cv::Mat out;
        cv::bilateralFilter(c, out, 0, cfg.chroma_bilateral.sigma_range,
                            cfg.chroma_bilateral.sigma_spatial,
                            cv::BORDER_REFLECT_101);
        c = out;
    }
}

} // namespace
void chroma_denoise_rgb_inplace(Matrix2Df& r, Matrix2Df& g, Matrix2Df& b,
                                const config::ChromaDenoiseConfig& cfg) {
    if (!cfg.enabled) return;
    if (r.size() <= 0 || g.size() <= 0 || b.size() <= 0) return;
    if (r.rows() != g.rows() || r.cols() != g.cols() ||
        r.rows() != b.rows() || r.cols() != b.cols()) {
        return;
    }

    cv::Mat R(r.rows(), r.cols(), CV_32F, r.data());
    cv::Mat G(g.rows(), g.cols(), CV_32F, g.data());
    cv::Mat B(b.rows(), b.cols(), CV_32F, b.data());

    if (cfg.blend.mode != "chroma_only") return;

    cv::Mat Y, C1, C2;
    rgb_to_chroma_space(R, G, B, cfg.color_space, Y, C1, C2);

    // Dataset-aware adaptation: scale denoise strength from measured chroma noise.
    // This keeps fine detail on clean data and increases suppression on noisy data.
    config::ChromaDenoiseConfig tuned = cfg;
    const float sigma_c1 = robust_sigma_mad_from_mat(C1);
    const float sigma_c2 = robust_sigma_mad_from_mat(C2);
    const float chroma_sigma = 0.5f * (sigma_c1 + sigma_c2);
    const float ref_sigma = 0.02f;
    const float adapt = std::clamp(chroma_sigma / ref_sigma, 0.8f, 1.4f);
    tuned.blend.amount = std::clamp(cfg.blend.amount * adapt, 0.0f, 1.0f);
    tuned.chroma_wavelet.threshold_scale =
        std::max(0.1f, cfg.chroma_wavelet.threshold_scale * adapt);
    tuned.chroma_bilateral.sigma_range =
        std::max(1.0e-4f, cfg.chroma_bilateral.sigma_range * std::sqrt(adapt));

    cv::Mat C1_orig = C1.clone();
    cv::Mat C2_orig = C2.clone();

    denoise_chroma_plane_inplace(C1, tuned);
    denoise_chroma_plane_inplace(C2, tuned);

    cv::Mat amount_map(Y.size(), CV_32F, cv::Scalar(tuned.blend.amount));
    if (tuned.protect_luma) {
        cv::Mat protect = build_protection_mask(Y, tuned);
        amount_map = amount_map.mul(1.0f - tuned.luma_guard_strength * protect);
        cv::min(amount_map, tuned.blend.amount, amount_map);
        cv::max(amount_map, 0.0, amount_map);
    }

    cv::Mat one_minus = 1.0f - amount_map;
    cv::Mat C1_mix = C1_orig.mul(one_minus) + C1.mul(amount_map);
    cv::Mat C2_mix = C2_orig.mul(one_minus) + C2.mul(amount_map);

    cv::Mat R_new, G_new, B_new;
    chroma_space_to_rgb(Y, C1_mix, C2_mix, cfg.color_space, R_new, G_new, B_new);

    R_new.copyTo(R);
    G_new.copyTo(G);
    B_new.copyTo(B);
}

} // namespace tile_compile::reconstruction
