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
                              const config::ChromaDenoiseConfig& cfg,
                              ChromaDenoiseStats* stats) {
    cv::Mat mask = cv::Mat::zeros(y.size(), CV_32F);
    const double pixels = static_cast<double>(y.total());

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
        if (stats && pixels > 0.0)
            stats->star_protected_fraction = cv::countNonZero(stars > 0.5f) / pixels;
        // Feather this component's own boundary, scaled to ITS OWN dilation
        // radius -- not the combined mask's largest radius. star_protection's
        // disk is typically much smaller than extended_source_protection's
        // (a handful of px vs. tens of px for a galaxy disc); feathering it
        // by the latter's scale would smear the precise star mask far more
        // than the ring artifact requires, and the reverse -- feathering a
        // large extended-source disc by only the star mask's small scale --
        // would leave the extended-source boundary just as sharp as before.
        {
            const float star_feather = std::max(
                1.0f, static_cast<float>(cfg.star_protection.dilate_px) / 3.0f);
            cv::GaussianBlur(stars, stars, cv::Size(0, 0), star_feather,
                             star_feather, cv::BORDER_REFLECT_101);
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
        if (stats && pixels > 0.0)
            stats->structure_protected_fraction =
                cv::countNonZero(structures > 0.5f) / pixels;
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
        if (stats) {
            stats->extended_source_sky_median = sky_med;
            stats->extended_source_sky_sigma = sky_sigma;
            stats->extended_source_threshold = thr;
            if (pixels > 0.0)
                stats->extended_source_raw_fraction =
                    cv::countNonZero(ext_src > 0.5f) / pixels;
        }
        if (cfg.extended_source_protection.dilate_px > 0) {
            const int k = std::max(1, cfg.extended_source_protection.dilate_px * 2 + 1);
            cv::Mat ker = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
            cv::dilate(ext_src, ext_src, ker);
        }
        if (stats && pixels > 0.0)
            stats->extended_source_protected_fraction =
                cv::countNonZero(ext_src > 0.5f) / pixels;
        // Own-scale feather -- see the star_protection block above for why
        // this must not share a radius with the other components.
        {
            const float ext_feather = std::max(
                1.0f,
                static_cast<float>(cfg.extended_source_protection.dilate_px) /
                    3.0f);
            cv::GaussianBlur(ext_src, ext_src, cv::Size(0, 0), ext_feather,
                             ext_feather, cv::BORDER_REFLECT_101);
        }
        cv::max(mask, ext_src, mask);
    }

    // Final small anti-aliasing pass on the combined mask (star_protection
    // and extended_source_protection are already feathered at their own
    // scale above; this just smooths the max() seams between components).
    cv::GaussianBlur(mask, mask, cv::Size(0, 0), 1.0, 1.0, cv::BORDER_REFLECT_101);
    cv::min(mask, 1.0, mask);
    cv::max(mask, 0.0, mask);
    if (stats && pixels > 0.0) {
        stats->combined_protected_fraction =
            cv::countNonZero(mask > 0.5f) / pixels;
        stats->mean_protection = cv::mean(mask)[0];
    }
    return mask;
}

/// @brief Removes a smooth, large-scale bias/variation from one chroma plane.
/// @details The wavelet/bilateral stages below only ever shrink *detail*
/// relative to a progressively-blurred low-pass version and by construction
/// never touch that low-pass remainder (see denoise_chroma_plane_inplace), so
/// a background color cast wider than the wavelet's coarsest scale passes
/// through them untouched. This estimates a coarse block-median surface
/// using ONLY unprotected (background) pixels -- so real extended-source /
/// star color never contributes to the estimate -- fills any block that has
/// no unprotected pixel at all via a normalized-convolution ("push-pull")
/// fill so the surface has no holes, smooths the grid into a continuous
/// field, and subtracts `strength` of its deviation from the surface's own
/// global background median (so the overall background chroma level, e.g.
/// real sky-glow color, is preserved -- only spatial variation across it is
/// flattened).
///
/// This relies entirely on `protect_mask` (star_protection / structure_
/// protection / extended_source_protection) to exclude any large foreground
/// object from the estimate. There is deliberately no additional statistical
/// safeguard against an incompletely-protected object: both the median used
/// for the reference level and a MAD-based outlier check share the same
/// ~50%-breakdown point, so neither can distinguish "sky" from "foreground"
/// once the object is large enough to matter, and both are already immune
/// to a strict minority -- there is no regime where a same-channel numeric
/// safeguard adds real protection beyond protect_mask. A target with a large
/// extended object MUST enable extended_source_protection (with a dilation
/// covering the object) for this stage to be safe to use.
///
/// The correction is also faded out inside protected pixels before being
/// subtracted: there the surface is only interpolated from surrounding sky
/// blocks and may be contaminated by unprotected faint halo, so applying it
/// would subtract part of the object's own color.
double flatten_large_scale_chroma_bias(
        cv::Mat& c, const cv::Mat& protect_mask,
        const config::ChromaDenoiseConfig::LargeScaleBiasConfig& cfg,
        double* holes_filled_fraction) {
    if (holes_filled_fraction) *holes_filled_fraction = 0.0;
    if (!cfg.enabled || c.empty()) return 0.0;

    const int bs = std::max(4, cfg.block_size);
    const int gy = std::max(1, (c.rows + bs - 1) / bs);
    const int gx = std::max(1, (c.cols + bs - 1) / bs);
    cv::Mat grid(gy, gx, CV_32F, cv::Scalar(0.0f));
    cv::Mat grid_valid(gy, gx, CV_32F, cv::Scalar(0.0f));

    std::vector<float> block_vals;
    for (int by = 0; by < gy; ++by) {
        const int r0 = by * bs, r1 = std::min(c.rows, r0 + bs);
        for (int bx = 0; bx < gx; ++bx) {
            const int c0 = bx * bs, c1 = std::min(c.cols, c0 + bs);
            block_vals.clear();
            for (int r = r0; r < r1; ++r) {
                const float* crow = c.ptr<float>(r);
                const float* mrow =
                    protect_mask.empty() ? nullptr : protect_mask.ptr<float>(r);
                for (int cc = c0; cc < c1; ++cc) {
                    if (mrow && mrow[cc] > 0.5f) continue;  // background-only
                    block_vals.push_back(crow[cc]);
                }
            }
            if (block_vals.empty()) continue;
            std::nth_element(block_vals.begin(),
                             block_vals.begin() + block_vals.size() / 2,
                             block_vals.end());
            grid.at<float>(by, bx) = block_vals[block_vals.size() / 2];
            grid_valid.at<float>(by, bx) = 1.0f;
        }
    }

    const int n_cells = gy * gx;
    const int n_valid = cv::countNonZero(grid_valid);
    if (n_valid == 0) return 0.0;  // fully protected image: nothing to estimate from
    if (holes_filled_fraction)
        *holes_filled_fraction =
            static_cast<double>(n_cells - n_valid) / static_cast<double>(n_cells);

    // Push-pull fill: propagate valid block values into holes via a growing
    // normalized-convolution box filter, so every cell ends up covered.
    if (n_valid < n_cells) {
        cv::Mat value = grid.mul(grid_valid);
        cv::Mat weight = grid_valid.clone();
        int ksize = 3;
        const int kmax = 2 * std::max(gy, gx) + 1;
        while (cv::countNonZero(weight) < n_cells && ksize <= kmax) {
            cv::Mat blurred_value, blurred_weight;
            cv::boxFilter(value, blurred_value, -1, cv::Size(ksize, ksize),
                          cv::Point(-1, -1), false, cv::BORDER_REPLICATE);
            cv::boxFilter(weight, blurred_weight, -1, cv::Size(ksize, ksize),
                          cv::Point(-1, -1), false, cv::BORDER_REPLICATE);
            cv::Mat still_empty;
            cv::compare(weight, 0.5f, still_empty, cv::CMP_LT);
            cv::Mat reached;
            cv::compare(blurred_weight, 1.0e-6f, reached, cv::CMP_GT);
            cv::Mat newly;
            cv::bitwise_and(still_empty, reached, newly);
            cv::Mat filled_value;
            cv::divide(blurred_value, blurred_weight, filled_value);
            filled_value.copyTo(value, newly);
            cv::Mat ones(weight.size(), CV_32F, cv::Scalar(1.0f));
            ones.copyTo(weight, newly);
            ksize += 2;
        }
        grid = value;
    }

    // Reference level: median of the ORIGINALLY-valid block medians only --
    // the plane's overall background chroma level, preserved unchanged.
    std::vector<float> gv;
    gv.reserve(static_cast<size_t>(n_valid));
    for (int by = 0; by < gy; ++by)
        for (int bx = 0; bx < gx; ++bx)
            if (grid_valid.at<float>(by, bx) > 0.5f)
                gv.push_back(grid.at<float>(by, bx));
    std::nth_element(gv.begin(), gv.begin() + gv.size() / 2, gv.end());
    const float ref = gv[gv.size() / 2];

    cv::Mat surface;
    cv::resize(grid, surface, c.size(), 0, 0, cv::INTER_LINEAR);
    if (cfg.blur_sigma > 0.0f)
        cv::GaussianBlur(surface, surface, cv::Size(0, 0), cfg.blur_sigma,
                         cfg.blur_sigma, cv::BORDER_REFLECT_101);

    cv::Mat correction = (surface - ref) * cfg.strength;
    if (!protect_mask.empty()) {
        // The surface inside protected regions is only a push-pull
        // interpolation -- the least reliable part of the estimate -- and
        // can additionally be contaminated by unprotected faint-halo pixels
        // leaking into neighbouring block medians. Subtracting it there
        // removes the foreground's own chroma (the M31 regression). Fail
        // closed: fade the correction out inside protected areas. Residual
        // sky variation underneath a protected object is preferable to
        // destroying real object color.
        correction = correction.mul(1.0f - protect_mask);
    }
    cv::Scalar mean, stddev;
    cv::meanStdDev(correction, mean, stddev);
    c -= correction;
    return stddev[0];
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
ChromaDenoiseStats chroma_denoise_rgb_inplace(
        Matrix2Df& r, Matrix2Df& g, Matrix2Df& b,
        const config::ChromaDenoiseConfig& cfg) {
    ChromaDenoiseStats stats;
    if (!cfg.enabled) return stats;
    if (r.size() <= 0 || g.size() <= 0 || b.size() <= 0) return stats;
    if (r.rows() != g.rows() || r.cols() != g.cols() ||
        r.rows() != b.rows() || r.cols() != b.cols()) {
        return stats;
    }

    cv::Mat R(r.rows(), r.cols(), CV_32F, r.data());
    cv::Mat G(g.rows(), g.cols(), CV_32F, g.data());
    cv::Mat B(b.rows(), b.cols(), CV_32F, b.data());

    if (cfg.blend.mode != "chroma_only") return stats;

    cv::Mat Y, C1, C2;
    rgb_to_chroma_space(R, G, B, cfg.color_space, Y, C1, C2);

    // Dataset-aware adaptation: scale denoise strength from measured chroma
    // noise, expressed RELATIVE to the image's own luma-noise sigma (not a
    // fixed absolute constant -- the old ref_sigma=0.02f was calibrated for
    // [0,1]-normalized data and saturated the clamp on real, unnormalized
    // ADU-scale data, where absolute chroma sigma is routinely in the tens).
    // This keeps fine detail on clean data and increases suppression on noisy data.
    config::ChromaDenoiseConfig tuned = cfg;
    const float sigma_c1 = robust_sigma_mad_from_mat(C1);
    const float sigma_c2 = robust_sigma_mad_from_mat(C2);
    const float chroma_sigma = 0.5f * (sigma_c1 + sigma_c2);
    const float luma_sigma = std::max(1.0e-6f, robust_sigma_mad_from_mat(Y));
    const float ref_ratio = std::max(1.0e-6f, cfg.adaptation_reference_ratio);
    const float adapt =
        std::clamp((chroma_sigma / luma_sigma) / ref_ratio, 0.8f, 1.4f);
    tuned.blend.amount = std::clamp(cfg.blend.amount * adapt, 0.0f, 1.0f);
    tuned.chroma_wavelet.threshold_scale =
        std::max(0.1f, cfg.chroma_wavelet.threshold_scale * adapt);
    tuned.chroma_bilateral.sigma_range =
        std::max(1.0e-4f, cfg.chroma_bilateral.sigma_range * std::sqrt(adapt));
    stats.applied = true;
    stats.valid_pixels = static_cast<std::uint64_t>(r.size());
    stats.input_chroma_sigma = chroma_sigma;
    stats.input_luma_sigma = luma_sigma;
    stats.adaptation = adapt;
    stats.effective_blend_amount = tuned.blend.amount;

    // Protection mask, computed once up front: it depends only on
    // star_protection/structure_protection/extended_source_protection/
    // luma_guard_strength, none of which the adaptation above touches, and
    // is reused both for the large-scale bias estimate below (background-
    // only pixels) and for the final blend-back amount_map.
    cv::Mat protect;
    if (tuned.protect_luma) protect = build_protection_mask(Y, tuned, &stats);

    cv::Mat C1_orig = C1.clone();
    cv::Mat C2_orig = C2.clone();

    if (tuned.large_scale_bias.enabled) {
        double holes_c1 = 0.0, holes_c2 = 0.0;
        stats.large_scale_bias_removed_rms_c1 = flatten_large_scale_chroma_bias(
            C1, protect, tuned.large_scale_bias, &holes_c1);
        stats.large_scale_bias_removed_rms_c2 = flatten_large_scale_chroma_bias(
            C2, protect, tuned.large_scale_bias, &holes_c2);
        stats.large_scale_bias_grid_holes_filled_fraction =
            std::max(holes_c1, holes_c2);
    }

    denoise_chroma_plane_inplace(C1, tuned);
    denoise_chroma_plane_inplace(C2, tuned);

    cv::Mat amount_map(Y.size(), CV_32F, cv::Scalar(tuned.blend.amount));
    if (tuned.protect_luma) {
        amount_map = amount_map.mul(1.0f - tuned.luma_guard_strength * protect);
        cv::min(amount_map, tuned.blend.amount, amount_map);
        cv::max(amount_map, 0.0, amount_map);
    }
    stats.mean_denoise_fraction = cv::mean(amount_map)[0];

    cv::Mat one_minus = 1.0f - amount_map;
    cv::Mat C1_mix = C1_orig.mul(one_minus) + C1.mul(amount_map);
    cv::Mat C2_mix = C2_orig.mul(one_minus) + C2.mul(amount_map);

    cv::Mat R_new, G_new, B_new;
    chroma_space_to_rgb(Y, C1_mix, C2_mix, cfg.color_space, R_new, G_new, B_new);

    R_new.copyTo(R);
    G_new.copyTo(G);
    B_new.copyTo(B);
    return stats;
}

} // namespace tile_compile::reconstruction
