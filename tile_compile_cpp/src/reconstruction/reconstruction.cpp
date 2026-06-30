#include "tile_compile/reconstruction/reconstruction.hpp"
#include "tile_compile/metrics/aqmh_quality_map_cache.hpp"
#include "tile_compile/reconstruction/tile_grid_key.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

namespace tile_compile::reconstruction {

namespace {

constexpr double kSigmaClipEpsNeff = 1.0e-6;
constexpr double kSigmaClipEpsVar = 1.0e-12;

/// @brief Implements invalid reconstruction sample.
/// @details Part of tile reconstruction, sigma clipping, overlap-add, and synthetic stacking helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
inline float invalid_reconstruction_sample() {
    return std::numeric_limits<float>::quiet_NaN();
}

/// @brief Checks valid sample.
/// @details Part of tile reconstruction, sigma clipping, overlap-add, and synthetic stacking helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
inline bool is_valid_sample(float v) {
    return std::isfinite(v);
}

/// @brief Implements median inplace.
/// @details Part of tile reconstruction, sigma clipping, overlap-add, and synthetic stacking helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float median_inplace(std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<long>(mid), v.end());
    return v[mid];
}

/// @brief Implements robust sigma mad from mat.
/// @details Part of tile reconstruction, sigma clipping, overlap-add, and synthetic stacking helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float robust_sigma_mad_from_mat(const cv::Mat& m) {
    if (m.empty()) return 0.0f;
    std::vector<float> vals;
    vals.reserve(m.total());
    for (int y = 0; y < m.rows; ++y) {
        const float* row = m.ptr<float>(y);
        vals.insert(vals.end(), row, row + m.cols);
    }
    float med = median_inplace(vals);
    for (float& v : vals) v = std::fabs(v - med);
    float mad = median_inplace(vals);
    return 1.4826f * mad;
}

/// @brief Implements percentile from mat.
/// @details Part of tile reconstruction, sigma clipping, overlap-add, and synthetic stacking helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float percentile_from_mat(const cv::Mat& m, float p) {
    if (m.empty()) return 0.0f;
    std::vector<float> vals;
    vals.reserve(m.total());
    for (int y = 0; y < m.rows; ++y) {
        const float* row = m.ptr<float>(y);
        vals.insert(vals.end(), row, row + m.cols);
    }
    if (vals.empty()) return 0.0f;
    p = std::clamp(p, 0.0f, 100.0f);
    const size_t idx = static_cast<size_t>(std::round((p / 100.0f) * static_cast<float>(vals.size() - 1)));
    std::nth_element(vals.begin(), vals.begin() + static_cast<long>(idx), vals.end());
    return vals[idx];
}

/// @brief Implements quantize to step.
/// @details Part of tile reconstruction, sigma clipping, overlap-add, and synthetic stacking helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float quantize_to_step(float value, float lo, float hi, float step) {
    if (!(std::isfinite(value) && std::isfinite(lo) && std::isfinite(hi))) {
        return lo;
    }
    if (!(step > 0.0f) || !(hi > lo)) {
        return std::clamp(value, lo, hi);
    }
    const float clamped = std::clamp(value, lo, hi);
    const float buckets = std::round((clamped - lo) / step);
    return std::clamp(lo + buckets * step, lo, hi);
}

struct TileOverlaps {
    int left = 0;
    int right = 0;
    int top = 0;
    int bottom = 0;
};

TileOverlaps compute_tile_overlaps(
    const Tile& tile,
    const std::unordered_map<uint64_t, size_t>& tile_by_grid,
    const std::vector<Tile>& tiles) {
    TileOverlaps out;
    auto find_neighbor = [&](int row, int col) -> const Tile* {
        auto it = tile_by_grid.find(tile_grid_key(row, col));
        return (it != tile_by_grid.end()) ? &tiles[it->second] : nullptr;
    };
    if (const Tile* nb = find_neighbor(tile.row, tile.col - 1)) {
        out.left = std::max(0, (nb->x + nb->width) - tile.x);
    }
    if (const Tile* nb = find_neighbor(tile.row, tile.col + 1)) {
        out.right = std::max(0, (tile.x + tile.width) - nb->x);
    }
    if (const Tile* nb = find_neighbor(tile.row - 1, tile.col)) {
        out.top = std::max(0, (nb->y + nb->height) - tile.y);
    }
    if (const Tile* nb = find_neighbor(tile.row + 1, tile.col)) {
        out.bottom = std::max(0, (tile.y + tile.height) - nb->y);
    }
    return out;
}

bool same_shape(const Matrix2Df& m, int rows, int cols) {
    return m.rows() == rows && m.cols() == cols;
}

bool canvas_valid_at(const std::vector<uint8_t>& mask, int width, int height,
                     int x, int y) {
    if (x < 0 || y < 0 || x >= width || y >= height) {
        return false;
    }
    if (mask.empty()) {
        return true;
    }
    const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width) +
                       static_cast<size_t>(x);
    return idx < mask.size() && mask[idx] != 0;
}

float aqmh_global_weight(const VectorXf& global_weights, size_t fi) {
    if (fi >= static_cast<size_t>(global_weights.size())) {
        return 1.0f;
    }
    const float w = global_weights[static_cast<Eigen::Index>(fi)];
    return (std::isfinite(w) && w > 0.0f) ? w : 0.0f;
}

/// @brief Implements select wiener quality target.
/// @details Part of tile reconstruction, sigma clipping, overlap-add, and synthetic stacking helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float select_wiener_quality_target(float q_struct_tile,
                                   const config::WienerDenoiseConfig& cfg) {
    float best_q = cfg.q_min;
    float best_cost = std::numeric_limits<float>::infinity();
    int iter = 0;
    for (float q = cfg.q_min;
         q <= cfg.q_max + 0.5f * cfg.q_step && iter < cfg.max_iterations;
         q += cfg.q_step, ++iter) {
        const float q_candidate = std::clamp(q, cfg.q_min, cfg.q_max);
        const float cost = std::fabs(q_struct_tile - q_candidate);
        if (cost < best_cost) {
            best_cost = cost;
            best_q = q_candidate;
        }
    }
    return quantize_to_step(best_q, cfg.q_min, cfg.q_max, cfg.q_step);
}

/// @brief Implements rgb to chroma space.
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

AqmhReconstructionResult reconstruct_aqmh_weighted(
    size_t frame_count, const AqmhFrameLoader& load_frame,
    metrics::QualityMapCache* q_map_cache, const VectorXf& global_weights,
    const std::vector<uint8_t>& canvas_mask, int width, int height,
    const AqmhReconstructionConfig& cfg) {
    AqmhReconstructionResult result;
    result.output = Matrix2Df::Zero(height, width);
    result.weight_sum = Matrix2Df::Zero(height, width);
    if (frame_count == 0 || load_frame == nullptr || q_map_cache == nullptr ||
        width <= 0 || height <= 0) {
        return result;
    }

    // --- Pass 1: Welford online weighted mean + variance in one scan ----------
    // Welford accumulators (West 1979 / Chan 1983 weighted variant):
    //   W    = sum of weights
    //   mean = current weighted mean
    //   M2   = sum of w_i * (x_i - mean_{i-1}) * (x_i - mean_i)
    // Variance = M2 / W  (biased, suitable for sigma clipping)
    // This replaces the original separate mean-pass and variance-pass,
    // halving the number of frame/map disk-IO operations.
    //
    // Cherry-pick pre-pass: if cfg.cherry_pick is true, we first collect the
    // per-pixel score S_f(p) = G_f * Q_map_f(p) for all frames (one pass),
    // then determine the K(p) threshold per pixel, then run the Welford pass
    // only over the selected frames (second pass).  For normal operation the
    // cherry_pick pre-pass is skipped entirely.

    // Helper: per-pixel top-K score threshold map.
    // top_k_threshold(p) = the K-th largest score at pixel p (inclusive).
    // Frames with score < threshold are excluded from reconstruction at p.
    // An all-zero threshold map means "use all frames" (cherry_pick disabled).
    Matrix2Df top_k_threshold = Matrix2Df::Zero(height, width);

    if (cfg.cherry_pick) {
        // Pre-pass: accumulate per-pixel score lists.
        // We store compressed score maps: for each pixel we need the sorted
        // top-K scores, but frame_count can be large.  We use a per-pixel
        // nth_element approach: accumulate all finite scores into a
        // frame_count-length buffer (one per pixel is too large), so instead
        // we do a streaming selection via a small per-pixel max-heap of size K.
        // For memory efficiency we process all frames first and build score
        // maps, then find per-pixel thresholds.
        //
        // Memory bound: frame_count * width * height * 4B can be large.
        // We therefore build an explicit per-pixel sorted vector only for
        // the score threshold, not for intermediate storage.
        //
        // Practical approach: accumulate a compact per-pixel N_valid count and
        // a sorted per-pixel score list using a single score matrix per frame
        // (O(frame_count * W * H) total storage, same as keeping all maps —
        // but we release each score matrix immediately after updating the
        // per-pixel sorted selection structures).
        //
        // To stay within the memory budget we use a row-major approach:
        // build a per-pixel score buffer by streaming frames, keeping only
        // two arrays: a N_valid count and a threshold accumulator using
        // a per-pixel partial sort.  We allocate a vector<float> of length
        // frame_count per-pixel only lazily via a 2D ragged structure — this
        // is only acceptable for small images.  For large images (> 4 Mpx)
        // we fall back to a simpler approach: threshold = K-th smallest of
        // the global per-frame G_f weight (pixel-independent), which is
        // still better than no cherry-pick.
        //
        // For correctness the spec (§5.3) requires per-pixel top-K.
        // Implementation: collect all scores into a transposed buffer, then
        // nth_element per pixel. We limit this to images up to
        // kCherryPickMaxPixels; beyond that use a global threshold.
        constexpr int64_t kCherryPickMaxPixels = 8L * 1024 * 1024; // 8 Mpx
        const int64_t total_px = static_cast<int64_t>(width) * static_cast<int64_t>(height);

        if (total_px <= kCherryPickMaxPixels) {
            // Per-pixel cherry-pick: collect scores[fi][y*width+x].
            std::vector<std::vector<float>> scores(frame_count);
            for (size_t fi = 0; fi < frame_count; ++fi) {
                scores[fi].assign(static_cast<size_t>(width * height), 0.0f);
            }
            for (size_t fi = 0; fi < frame_count; ++fi) {
                Matrix2Df q_map = q_map_cache->read_cached(fi);
                if (!same_shape(q_map, height, width)) continue;
                const float gw = aqmh_global_weight(global_weights, fi);
#pragma omp parallel for schedule(static)
                for (int y = 0; y < height; ++y) {
                    const float* q_ptr = q_map.data() + y * width;
                    float* s_ptr = scores[fi].data() + y * width;
                    for (int x = 0; x < width; ++x) {
                        const float q = q_ptr[x];
                        s_ptr[x] = (std::isfinite(q) && q > 0.0f) ? gw * q : 0.0f;
                    }
                }
            }
            // Per-pixel: compute K(p) and find the K-th largest score.
            // Scratch buffers are reused across pixels to avoid per-pixel
            // heap allocations in this hot loop (one alloc per row instead
            // of one per pixel).
#pragma omp parallel for schedule(static)
            for (int y = 0; y < height; ++y) {
                std::vector<float> valid_scores;
                valid_scores.reserve(frame_count);
                for (int x = 0; x < width; ++x) {
                    if (!canvas_valid_at(canvas_mask, width, height, x, y)) continue;
                    valid_scores.clear();
                    for (size_t fi = 0; fi < frame_count; ++fi) {
                        const float s = scores[fi][static_cast<size_t>(y * width + x)];
                        if (s > 0.0f) valid_scores.push_back(s);
                    }
                    const int n_valid = static_cast<int>(valid_scores.size());
                    const int K = std::min(n_valid,
                        std::max(cfg.cherry_pick_k_min,
                                 static_cast<int>(std::floor(
                                     cfg.cherry_pick_k_frac * static_cast<float>(n_valid)))));
                    if (K <= 0 || K >= n_valid) {
                        top_k_threshold(y, x) = 0.0f; // use all
                        continue;
                    }
                    // nth_element: pivot at position (n_valid - K) in ascending order.
                    // The threshold is the value at that position.
                    const size_t kth_idx = static_cast<size_t>(n_valid - K);
                    std::nth_element(valid_scores.begin(),
                                     valid_scores.begin() + static_cast<long>(kth_idx),
                                     valid_scores.end());
                    top_k_threshold(y, x) = valid_scores[kth_idx];
                }
            }
        } else {
            // Large image fallback: global per-frame score threshold.
            // Use K = max(k_min, floor(k_frac * frame_count)).
            const int n_frames_int = static_cast<int>(frame_count);
            const int K = std::max(cfg.cherry_pick_k_min,
                static_cast<int>(std::floor(cfg.cherry_pick_k_frac *
                                            static_cast<float>(n_frames_int))));
            std::vector<float> gw_scores(frame_count);
            for (size_t fi = 0; fi < frame_count; ++fi)
                gw_scores[fi] = aqmh_global_weight(global_weights, fi);
            const size_t kth_idx = static_cast<size_t>(
                std::max(0, static_cast<int>(frame_count) - K));
            std::vector<float> sorted_gw = gw_scores;
            std::nth_element(sorted_gw.begin(),
                             sorted_gw.begin() + static_cast<long>(kth_idx),
                             sorted_gw.end());
            const float global_thr = sorted_gw[kth_idx];
            // Apply uniform threshold to all valid canvas pixels.
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                    if (canvas_valid_at(canvas_mask, width, height, x, y))
                        top_k_threshold(y, x) = global_thr;
        }
    }

    Matrix2Df welford_W   = Matrix2Df::Zero(height, width); // == result.weight_sum after pass 1
    Matrix2Df welford_mean= Matrix2Df::Zero(height, width);
    Matrix2Df welford_M2  = Matrix2Df::Zero(height, width);
    Matrix2Df finite_map_count      = Matrix2Df::Zero(height, width);
    Matrix2Df positive_weight_count = Matrix2Df::Zero(height, width);

    auto valid_canvas_pixels = [&]() -> uint64_t {
        uint64_t n = 0;
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                if (canvas_valid_at(canvas_mask, width, height, x, y))
                    ++n;
        return n;
    };
    const uint64_t valid_px = valid_canvas_pixels();

    for (size_t fi = 0; fi < frame_count; ++fi) {
        Matrix2Df frame;
        if (!load_frame(fi, frame) || !same_shape(frame, height, width))
            continue;
        Matrix2Df q_map = q_map_cache->read_cached(fi);
        if (!same_shape(q_map, height, width)) {
            result.missing_map_samples += valid_px;
            continue;
        }
        const float gw = aqmh_global_weight(global_weights, fi);
        uint64_t missing_increment = 0;
        uint64_t finite_increment = 0;
#pragma omp parallel for schedule(static) reduction(+:missing_increment, finite_increment)
        for (int y = 0; y < height; ++y) {
            const float* frame_ptr  = frame.data()      + y * width;
            const float* q_map_ptr  = q_map.data()      + y * width;
            float* fin_ptr  = finite_map_count.data()      + y * width;
            float* pos_ptr  = positive_weight_count.data() + y * width;
            float* W_ptr    = welford_W.data()    + y * width;
            float* mean_ptr = welford_mean.data() + y * width;
            float* M2_ptr   = welford_M2.data()   + y * width;

            for (int x = 0; x < width; ++x) {
                if (!canvas_mask.empty() && canvas_mask[y * width + x] == 0)
                    continue;
                const float q = q_map_ptr[x];
                if (!std::isfinite(q)) {
                    ++missing_increment;
                    continue;
                }
                fin_ptr[x] += 1.0f;
                ++finite_increment;
                const float w = gw * std::max(q, 0.0f);
                if (!(w > cfg.eps_weight) || !is_valid_sample(frame_ptr[x]))
                    continue;
                // Cherry-pick gate: skip this frame at this pixel if its score
                // is below the per-pixel K-th threshold.
                if (cfg.cherry_pick) {
                    const float score = gw * std::max(q, 0.0f);
                    const float thr = top_k_threshold.data()[y * width + x];
                    if (thr > 0.0f && score < thr)
                        continue;
                }
                pos_ptr[x] += 1.0f;
                // Welford update
                const float W_new  = W_ptr[x] + w;
                const float delta  = frame_ptr[x] - mean_ptr[x];
                const float mean_new = mean_ptr[x] + (w / W_new) * delta;
                const float delta2 = frame_ptr[x] - mean_new;
                M2_ptr[x]   += w * delta * delta2;
                W_ptr[x]     = W_new;
                mean_ptr[x]  = mean_new;
            }
        }
        result.missing_map_samples += missing_increment;
        result.finite_map_samples += finite_increment;
    }
    // Expose weight_sum from pass 1.
    result.weight_sum = welford_W;

    // --- Cherry-pick K-map diagnostics ----------------------------------------
    // positive_weight_count(y,x) = number of frames that contributed a positive
    // AQMH weight at pixel (x,y) in the Welford pass — after cherry-pick filtering
    // (or equal to N_valid when cherry_pick is disabled).
    // Store this as the K-map and compute aggregate stats.
    if (cfg.cherry_pick) {
        result.cherry_pick_per_pixel_mode =
            (static_cast<int64_t>(width) * static_cast<int64_t>(height) <=
             static_cast<int64_t>(8L * 1024 * 1024));
        result.cherry_pick_k_map = Matrix2Df::Zero(height, width);

        // total valid frames per pixel = finite_map_count (all frames with a
        // finite Q map value, before cherry-pick filtering).
        // K-active pixels: those where positive_weight_count < finite_map_count.
        uint64_t n_canvas_valid = 0;
        uint64_t n_active = 0;
        double k_sum = 0.0;
        std::vector<float> k_vals;
        k_vals.reserve(static_cast<size_t>(width * height / 4));
        int k_min_obs = std::numeric_limits<int>::max();
        int k_max_obs = 0;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (!canvas_valid_at(canvas_mask, width, height, x, y))
                    continue;
                ++n_canvas_valid;
                const float k = positive_weight_count(y, x);
                const float n_all = finite_map_count(y, x);
                result.cherry_pick_k_map(y, x) = k;
                if (k > 0.0f && n_all > k) {
                    // Cherry-pick was active at this pixel (fewer frames used).
                    ++n_active;
                    k_sum += k;
                    k_vals.push_back(k);
                    const int ki = static_cast<int>(k);
                    if (ki < k_min_obs) k_min_obs = ki;
                    if (ki > k_max_obs) k_max_obs = ki;
                }
            }
        }

        result.cherry_pick_active_frac =
            n_canvas_valid > 0
                ? static_cast<float>(n_active) / static_cast<float>(n_canvas_valid)
                : 0.0f;
        result.cherry_pick_mean_k =
            n_active > 0 ? static_cast<float>(k_sum / n_active) : 0.0f;
        if (!k_vals.empty()) {
            std::nth_element(k_vals.begin(),
                             k_vals.begin() + static_cast<long>(k_vals.size() / 2),
                             k_vals.end());
            result.cherry_pick_median_k = k_vals[k_vals.size() / 2];
            result.cherry_pick_k_min_observed = k_min_obs;
            result.cherry_pick_k_max_observed = k_max_obs;
        }
    }

    // --- Pass 2: sigma-clipped accumulate (reads frames/maps a second time) --
    Matrix2Df clipped_accum      = Matrix2Df::Zero(height, width);
    Matrix2Df clipped_weight_sum = Matrix2Df::Zero(height, width);

    for (size_t fi = 0; fi < frame_count; ++fi) {
        Matrix2Df frame;
        if (!load_frame(fi, frame) || !same_shape(frame, height, width))
            continue;
        Matrix2Df q_map = q_map_cache->read_cached(fi);
        if (!same_shape(q_map, height, width))
            continue;
        const float gw = aqmh_global_weight(global_weights, fi);
#pragma omp parallel for schedule(static)
        for (int y = 0; y < height; ++y) {
            const float* frame_ptr  = frame.data()      + y * width;
            const float* q_map_ptr  = q_map.data()      + y * width;
            const float* W_ptr      = welford_W.data()  + y * width;
            const float* mean_ptr   = welford_mean.data()+ y * width;
            const float* M2_ptr     = welford_M2.data() + y * width;
            float* ca_ptr  = clipped_accum.data()      + y * width;
            float* cw_ptr  = clipped_weight_sum.data() + y * width;

            for (int x = 0; x < width; ++x) {
                if ((!canvas_mask.empty() && canvas_mask[y * width + x] == 0) ||
                    W_ptr[x] <= cfg.eps_weight ||
                    !is_valid_sample(frame_ptr[x]))
                    continue;
                const float q = q_map_ptr[x];
                if (!std::isfinite(q))
                    continue;
                const float w = gw * std::max(q, 0.0f);
                if (!(w > cfg.eps_weight))
                    continue;
                // Cherry-pick gate (same threshold as pass 1).
                if (cfg.cherry_pick) {
                    const float thr = top_k_threshold.data()[y * width + x];
                    if (thr > 0.0f && w < thr)
                        continue;
                }
                const float sigma = std::sqrt(
                    std::max(M2_ptr[x] / W_ptr[x], 0.0f));
                const float lo = mean_ptr[x] - cfg.sigma_low  * sigma;
                const float hi = mean_ptr[x] + cfg.sigma_high * sigma;
                const float v  = frame_ptr[x];
                if (sigma <= static_cast<float>(kSigmaClipEpsVar) ||
                    (v >= lo && v <= hi)) {
                    ca_ptr[x] += w * v;
                    cw_ptr[x] += w;
                }
            }
        }
    }

    // --- Final pixel assembly --------------------------------------------------
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (!canvas_valid_at(canvas_mask, width, height, x, y)) {
                result.output(y, x) = 0.0f;
                result.weight_sum(y, x) = 0.0f;
                continue;
            }
            if (finite_map_count(y, x) > 0.0f &&
                positive_weight_count(y, x) <= 0.0f) {
                result.output(y, x) = 0.0f;
                result.weight_sum(y, x) = 0.0f;
                ++result.unsupported_pixels;
                ++result.zero_veto_pixels;
                continue;
            }
            if (result.weight_sum(y, x) <= cfg.eps_weight) {
                result.output(y, x) = 0.0f;
                result.weight_sum(y, x) = 0.0f;
                ++result.unsupported_pixels;
                continue;
            }
            const float min_kept =
                std::max(0.0f, cfg.min_fraction) * result.weight_sum(y, x);
            if (clipped_weight_sum(y, x) > cfg.eps_weight &&
                clipped_weight_sum(y, x) >= min_kept) {
                result.output(y, x) =
                    clipped_accum(y, x) / clipped_weight_sum(y, x);
                result.weight_sum(y, x) = clipped_weight_sum(y, x);
            } else {
                result.output(y, x) = welford_mean(y, x);
            }
        }
    }
    return result;
}

/// @brief Implements reconstruct tiles.
/// @details Part of tile reconstruction, sigma clipping, overlap-add, and synthetic stacking helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Matrix2Df reconstruct_tiles(const std::vector<Matrix2Df>& frames,
                            const TileGrid& grid,
                            const std::vector<std::vector<float>>& tile_weights) {
    if (frames.empty()) return Matrix2Df();
    
    int h = frames[0].rows();
    int w = frames[0].cols();
    Matrix2Df result = Matrix2Df::Zero(h, w);
    Matrix2Df weight_sum = Matrix2Df::Zero(h, w);

    std::unordered_map<uint64_t, size_t> tile_by_grid;
    tile_by_grid.reserve(grid.tiles.size());
    for (size_t ti = 0; ti < grid.tiles.size(); ++ti) {
        tile_by_grid.emplace(tile_grid_key(grid.tiles[ti].row, grid.tiles[ti].col),
                             ti);
    }

    for (size_t t = 0; t < grid.tiles.size(); ++t) {
        const Tile& tile = grid.tiles[t];
        const TileOverlaps overlaps =
            compute_tile_overlaps(tile, tile_by_grid, grid.tiles);

        const std::vector<float> wx =
            make_partition_window_1d(tile.width, overlaps.left, overlaps.right);
        const std::vector<float> wy =
            make_partition_window_1d(tile.height, overlaps.top, overlaps.bottom);
        
        for (size_t f = 0; f < frames.size(); ++f) {
            if (!same_shape(frames[f], h, w)) {
                continue;
            }
            const float weight =
                (f < tile_weights.size() && t < tile_weights[f].size())
                    ? tile_weights[f][t]
                    : 0.0f;
            if (!(std::isfinite(weight) && weight > 0.0f)) {
                continue;
            }
            
            const int y_begin = std::max(0, tile.y);
            const int y_end = std::min(h, tile.y + tile.height);
            const int x_begin = std::max(0, tile.x);
            const int x_end = std::min(w, tile.x + tile.width);
            for (int y = y_begin; y < y_end; ++y) {
                int ly = y - tile.y;
                if (ly < 0 || ly >= static_cast<int>(wy.size())) continue;
                float win_y = wy[static_cast<size_t>(ly)];
                
                const float* frame_ptr = frames[f].data() + y * w;
                float* res_ptr = result.data() + y * w;
                float* w_ptr = weight_sum.data() + y * w;
                
                for (int x = x_begin; x < x_end; ++x) {
                    int lx = x - tile.x;
                    if (lx < 0 || lx >= static_cast<int>(wx.size())) continue;
                    float win = win_y * wx[static_cast<size_t>(lx)];
                    float ww = weight * win;
                    const float v = frame_ptr[x];
                    if (!is_valid_sample(v)) continue;
                    res_ptr[x] += v * ww;
                    w_ptr[x] += ww;
                }
            }
        }
    }
    
    for (int i = 0; i < result.size(); ++i) {
        if (weight_sum.data()[i] > 0) {
            result.data()[i] /= weight_sum.data()[i];
        }
    }
    
    return result;
}

/// @brief Implements wiener tile filter.
/// @details Part of tile reconstruction, sigma clipping, overlap-add, and synthetic stacking helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Matrix2Df wiener_tile_filter(const Matrix2Df& tile, float sigma, float snr_tile,
                             float q_struct_tile, bool is_star_tile,
                             const config::WienerDenoiseConfig& cfg) {
    if (!cfg.enabled) return tile;
    if (is_star_tile) return tile;
    if (!(sigma > 0.0f)) return tile;
    const float q_target = select_wiener_quality_target(q_struct_tile, cfg);
    if (q_target >= cfg.q_max && snr_tile >= cfg.snr_threshold) return tile;

    const float q_span = std::max(1.0e-6f, cfg.q_max - cfg.q_min);
    const float q_factor =
        1.0f - std::clamp((q_target - cfg.q_min) / q_span, 0.0f, 1.0f);

    const float snr_floor = std::max(0.0f, std::min(cfg.min_snr, cfg.snr_threshold));
    const float snr_ceiling = std::max(cfg.min_snr, cfg.snr_threshold);
    const float snr_stable = std::max(snr_tile, snr_floor);
    float snr_factor = 1.0f;
    if (snr_ceiling > snr_floor + 1.0e-6f) {
        snr_factor = 1.0f -
                     std::clamp((snr_stable - snr_floor) /
                                    (snr_ceiling - snr_floor),
                                0.0f, 1.0f);
    } else if (snr_tile >= cfg.snr_threshold) {
        snr_factor = 0.0f;
    }

    const float filter_strength = std::clamp(q_factor * snr_factor, 0.0f, 1.0f);
    if (!(filter_strength > 1.0e-3f)) return tile;

    const int h = static_cast<int>(tile.rows());
    const int w = static_cast<int>(tile.cols());
    if (h <= 0 || w <= 0) return tile;

    const int pad_h = std::max(1, h / 4);
    const int pad_w = std::max(1, w / 4);

    cv::Mat tile_cv(h, w, CV_32F, const_cast<float*>(tile.data()));
    cv::Mat bg;
    int k_bg = std::max(1, (h / 8) | 1);
    cv::blur(tile_cv, bg, cv::Size(k_bg, k_bg), cv::Point(-1, -1), cv::BORDER_REFLECT_101);
    cv::Mat tile_for_fft = tile_cv - bg;

    cv::Mat padded;
    cv::copyMakeBorder(tile_for_fft, padded, pad_h, pad_h, pad_w, pad_w,
                       cv::BORDER_REFLECT_101);
    
    cv::Mat padded_bg;
    cv::copyMakeBorder(bg, padded_bg, pad_h, pad_h, pad_w, pad_w,
                       cv::BORDER_REFLECT_101);

    cv::Mat F;
    cv::dft(padded, F, cv::DFT_COMPLEX_OUTPUT);

    std::vector<cv::Mat> planes(2);
    cv::split(F, planes);
    cv::Mat power = planes[0].mul(planes[0]) + planes[1].mul(planes[1]);

    const float sigma_sq =
        sigma * sigma * (0.25f + 0.75f * filter_strength);
    const float eps = 1.0e-12f;
    cv::Mat H = power - sigma_sq;
    cv::threshold(H, H, 0.0, 0.0, cv::THRESH_TOZERO);
    cv::Mat denom = power + eps;
    cv::divide(H, denom, H);
    cv::min(H, 1.0, H);
    cv::max(H, 0.0, H);

    planes[0] = planes[0].mul(H);
    planes[1] = planes[1].mul(H);
    cv::merge(planes, F);

    cv::Mat filtered;
    cv::dft(F, filtered, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

    // FIX: Add back the background estimate that was subtracted before FFT
    cv::Mat restored = filtered + padded_bg;

    cv::Mat cropped = restored(cv::Rect(pad_w, pad_h, w, h));
    
    Matrix2Df out(h, w);
    if (cropped.isContinuous()) {
        std::memcpy(out.data(), cropped.ptr<float>(),
                    static_cast<size_t>(out.size()) * sizeof(float));
    } else {
        for (int r = 0; r < h; ++r) {
            const float* src = cropped.ptr<float>(r);
            float* dst = out.data() + static_cast<size_t>(r) * static_cast<size_t>(w);
            std::memcpy(dst, src, static_cast<size_t>(w) * sizeof(float));
        }
    }
    if (filter_strength < 1.0f) {
        out = ((1.0f - filter_strength) * tile.array() +
               filter_strength * out.array()).matrix();
    }
    return out;
}

/// @brief Implements soft threshold tile filter.
/// @details Part of tile reconstruction, sigma clipping, overlap-add, and synthetic stacking helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Matrix2Df soft_threshold_tile_filter(const Matrix2Df& tile,
                                      const config::SoftThresholdConfig& cfg) {
    if (!cfg.enabled) return tile;
    const int h = tile.rows();
    const int w = tile.cols();
    if (h <= 0 || w <= 0) return tile;

    cv::Mat tile_cv(h, w, CV_32F, const_cast<float*>(tile.data()));
    cv::Mat valid_mask;
    valid_mask = cv::Mat::zeros(h, w, CV_8U);
    cv::Mat tile_for_filter = tile_cv.clone();
    int valid_count = 0;
    for (int y = 0; y < h; ++y) {
        uchar* mask_row = valid_mask.ptr<uchar>(y);
        float* filter_row = tile_for_filter.ptr<float>(y);
        for (int x = 0; x < w; ++x) {
            if (std::isfinite(filter_row[x])) {
                mask_row[x] = 255;
                ++valid_count;
            } else {
                filter_row[x] = 0.0f;
            }
        }
    }
    if (valid_count <= 0) return tile;

    // 1. Background estimation via box blur over finite support only.
    cv::Mat bg;
    int k = cfg.blur_kernel | 1; // ensure odd
    cv::blur(tile_for_filter, bg, cv::Size(k, k), cv::Point(-1, -1),
             cv::BORDER_REFLECT_101);

    // 2. Highpass residual: R = T - B
    cv::Mat resid = tile_for_filter - bg;

    // 3. Robust noise estimate: σ = 1.4826 · median(|R - median(R)|)
    //    Only use finite pixels so negative calibrated samples remain valid.
    std::vector<float> rv;
    rv.reserve(static_cast<size_t>(resid.total()));
    for (int i = 0; i < static_cast<int>(resid.total()); ++i) {
        if (valid_mask.data[i]) {
            rv.push_back(resid.ptr<float>()[i]);
        }
    }
    if (rv.empty()) return tile;
    size_t mid = rv.size() / 2;
    std::nth_element(rv.begin(), rv.begin() + static_cast<long>(mid), rv.end());
    float med_r = rv[mid];
    for (size_t i = 0; i < rv.size(); ++i)
        rv[i] = std::fabs(rv[i] - med_r);
    std::nth_element(rv.begin(), rv.begin() + static_cast<long>(mid), rv.end());
    float mad = rv[mid];
    float sigma = 1.4826f * mad;

    if (!(sigma > 1e-12f)) return tile; // no noise to remove

    // 4. Soft-threshold: R' = sign(R) · max(|R| - τ, 0)
    float tau = cfg.alpha * sigma;
    cv::Mat result_resid = soft_threshold_signed(resid, tau);

    // 5. Reconstruct: T' = B + R'. Non-finite input support stays invalid.
    cv::Mat out_cv = bg + result_resid;
    
    cv::Mat invalid_mask;
    cv::bitwise_not(valid_mask, invalid_mask);
    out_cv.setTo(0.0f, invalid_mask);

    Matrix2Df out(h, w);
    if (out_cv.isContinuous()) {
        std::memcpy(out.data(), out_cv.ptr<float>(),
                    static_cast<size_t>(out.size()) * sizeof(float));
    } else {
        for (int r = 0; r < h; ++r) {
            const float* src = out_cv.ptr<float>(r);
            float* dst = out.data() + static_cast<size_t>(r) * static_cast<size_t>(w);
            std::memcpy(dst, src, static_cast<size_t>(w) * sizeof(float));
        }
    }
    return out;
}

/// @brief Implements chroma denoise rgb inplace.
/// @details Part of tile reconstruction, sigma clipping, overlap-add, and synthetic stacking helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
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

/// @brief Implements sigma clip stack.
/// @details Part of tile reconstruction, sigma clipping, overlap-add, and synthetic stacking helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Matrix2Df sigma_clip_stack(const std::vector<Matrix2Df>& frames,
                           float sigma_low, float sigma_high,
                           int max_iters, float min_fraction) {
    // Filter out empty (0×0) frames (e.g. empty synthetic cluster outputs)
    std::vector<std::reference_wrapper<const Matrix2Df>> valid;
    valid.reserve(frames.size());
    for (const auto& f : frames) {
        if (f.size() > 0) valid.emplace_back(f);
    }
    if (valid.empty()) return Matrix2Df();
    const int rows = valid[0].get().rows();
    const int cols = valid[0].get().cols();
    valid.erase(std::remove_if(valid.begin(), valid.end(),
                               [&](const auto& f) {
                                   return !same_shape(f.get(), rows, cols);
                               }),
                valid.end());
    if (valid.empty()) return Matrix2Df();
    Matrix2Df out(rows, cols);
    const int n = static_cast<int>(valid.size());

    std::vector<float> values;
    values.reserve(static_cast<size_t>(n));
    std::vector<uint8_t> keep(static_cast<size_t>(n), 1);
    std::vector<uint8_t> proposed_keep(static_cast<size_t>(n), 1);

    for (int idx = 0; idx < out.size(); ++idx) {
        // Collect only finite frames at this pixel.
        values.clear();
        std::fill(keep.begin(), keep.end(), static_cast<uint8_t>(0));
        int n_valid_here = 0;
        for (int i = 0; i < n; ++i) {
            float v = valid[static_cast<size_t>(i)].get().data()[idx];
            if (is_valid_sample(v)) {
                values.push_back(v);
                keep[static_cast<size_t>(i)] = 1;
                n_valid_here++;
            } else {
                values.push_back(0.0f);
            }
        }

        if (n_valid_here <= 0) {
            out.data()[idx] = invalid_reconstruction_sample();
            continue;
        }

        const int min_keep_here = std::max(1, static_cast<int>(std::ceil(min_fraction * n_valid_here)));
        int kept = n_valid_here;
        for (int iter = 0; iter < max_iters; ++iter) {
            if (kept <= 1) break;
            double sum = 0.0;
            double sumsq = 0.0;
            for (int i = 0; i < n; ++i) {
                if (!keep[static_cast<size_t>(i)]) continue;
                float v = values[static_cast<size_t>(i)];
                sum += static_cast<double>(v);
                sumsq += static_cast<double>(v) * static_cast<double>(v);
            }
            double mean = sum / static_cast<double>(kept);
            double var = sumsq / static_cast<double>(kept) - mean * mean;
            if (kept > 1)
                var *= static_cast<double>(kept) / static_cast<double>(kept - 1);
            double sd = (var > 0.0) ? std::sqrt(var) : 0.0;
            if (!(sd > 0.0)) break;

            int new_kept = 0;
            const double lo = mean - static_cast<double>(sigma_low) * sd;
            const double hi = mean + static_cast<double>(sigma_high) * sd;
            proposed_keep = keep;
            for (int i = 0; i < n; ++i) {
                if (!keep[static_cast<size_t>(i)]) continue;
                float v = values[static_cast<size_t>(i)];
                if (v < lo || v > hi) {
                    proposed_keep[static_cast<size_t>(i)] = 0;
                } else {
                    new_kept++;
                }
            }

            if (new_kept < min_keep_here) break;
            keep.swap(proposed_keep);
            kept = new_kept;
        }

        double sum = 0.0;
        int count = 0;
        for (int i = 0; i < n; ++i) {
            if (!keep[static_cast<size_t>(i)]) continue;
            sum += static_cast<double>(values[static_cast<size_t>(i)]);
            count++;
        }
        if (count <= 0) {
            // Fallback: average all finite-data frames at this pixel
            for (int i = 0; i < n; ++i) {
                float v = valid[static_cast<size_t>(i)].get().data()[idx];
                if (is_valid_sample(v)) { sum += static_cast<double>(v); count++; }
            }
        }
        out.data()[idx] =
            (count > 0) ? static_cast<float>(sum / static_cast<double>(count))
                        : invalid_reconstruction_sample();
    }

    return out;
}

/// @brief Implements sigma clip weighted tile.
/// @details Part of tile reconstruction, sigma clipping, overlap-add, and synthetic stacking helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Matrix2Df sigma_clip_weighted_tile(const std::vector<Matrix2Df>& tiles,
                                   const std::vector<float>& weights,
                                   float sigma_low, float sigma_high,
                                   int max_iters, float min_fraction) {
    if (tiles.empty()) return Matrix2Df();
    const int rows = tiles[0].rows();
    const int cols = tiles[0].cols();
    Matrix2Df out = Matrix2Df::Zero(rows, cols);
    std::vector<const float*> tile_ptrs;
    std::vector<double> active_weights;
    tile_ptrs.reserve(tiles.size());
    active_weights.reserve(weights.size());
    for (size_t i = 0; i < weights.size() && i < tiles.size(); ++i) {
        const float w = weights[i];
        if (!(std::isfinite(w) && w > 0.0f)) continue;
        if (!same_shape(tiles[i], rows, cols)) continue;
        active_weights.push_back(static_cast<double>(w));
        tile_ptrs.push_back(tiles[i].data());
    }
    const int n = static_cast<int>(tile_ptrs.size());
    if (n <= 0) {
        return out;
    }
    // Fast path for tiny stacks where clipping is not meaningful and only adds
    // overhead. Keep weighted averaging semantics intact.
    if (n <= 2 || max_iters <= 0) {
        for (int idx = 0; idx < out.size(); ++idx) {
            double wsum = 0.0;
            double wmean = 0.0;
            for (int i = 0; i < n; ++i) {
                const float v = tile_ptrs[static_cast<size_t>(i)][idx];
                if (!is_valid_sample(v)) continue;
                const double wi = active_weights[static_cast<size_t>(i)];
                wsum += wi;
                wmean += wi * static_cast<double>(v);
            }
            out.data()[idx] =
                (wsum > 0.0) ? static_cast<float>(wmean / wsum)
                             : invalid_reconstruction_sample();
        }
        return out;
    }

    std::vector<float> values(static_cast<size_t>(n));
    std::vector<uint8_t> keep(static_cast<size_t>(n));
    std::vector<uint8_t> proposed_keep(static_cast<size_t>(n));

    for (int idx = 0; idx < out.size(); ++idx) {
        int n_valid_here = 0;
        for (int i = 0; i < n; ++i) {
            const float v = tile_ptrs[static_cast<size_t>(i)][idx];
            values[static_cast<size_t>(i)] = v;
            const bool valid_here = is_valid_sample(v);
            keep[static_cast<size_t>(i)] = valid_here ? 1 : 0;
            if (valid_here) n_valid_here++;
        }

        if (n_valid_here <= 0) {
            out.data()[idx] = invalid_reconstruction_sample();
            continue;
        }

        const int min_keep_here = std::max(
            1, static_cast<int>(std::ceil(min_fraction * n_valid_here)));
        int kept = n_valid_here;
        for (int iter = 0; iter < max_iters; ++iter) {
            if (kept <= 1) break;
            // Compute weighted mean and stddev
            double wsum = 0.0, wmean = 0.0;
            for (int i = 0; i < n; ++i) {
                if (!keep[static_cast<size_t>(i)]) continue;
                double wi = active_weights[static_cast<size_t>(i)];
                wsum += wi;
                wmean += wi * static_cast<double>(values[static_cast<size_t>(i)]);
            }
            if (!(wsum > 0.0)) break;
            wmean /= wsum;

            double var = 0.0;
            double wsum2 = 0.0; // sum of squared weights for Bessel correction
            for (int i = 0; i < n; ++i) {
                if (!keep[static_cast<size_t>(i)]) continue;
                double wi = active_weights[static_cast<size_t>(i)];
                double d = static_cast<double>(values[static_cast<size_t>(i)]) - wmean;
                var += wi * d * d;
                wsum2 += wi * wi;
            }
            const double n_eff = (wsum * wsum) / std::max(wsum2, kSigmaClipEpsVar);
            // Bessel correction for reliability (non-frequency) weights:
            // var_unbiased = (Σ wi·d²) / (V1 - V2/V1)  where V1=wsum, V2=Σwi²
            double denom = wsum - wsum2 / std::max(wsum, kSigmaClipEpsVar);
            if (!(n_eff > 2.0 + kSigmaClipEpsNeff) || !(denom > kSigmaClipEpsVar)) {
                break;
            }
            double sd = (var > 0.0 && denom > 0.0) ? std::sqrt(var / denom) : 0.0;
            if (!(sd > 0.0)) break;

            const double lo = wmean - static_cast<double>(sigma_low) * sd;
            const double hi = wmean + static_cast<double>(sigma_high) * sd;
            int new_kept = 0;
            proposed_keep = keep;
            for (int i = 0; i < n; ++i) {
                if (!keep[static_cast<size_t>(i)]) continue;
                double v = static_cast<double>(values[static_cast<size_t>(i)]);
                if (v < lo || v > hi) {
                    proposed_keep[static_cast<size_t>(i)] = 0;
                } else {
                    new_kept++;
                }
            }
            if (new_kept < min_keep_here) break;
            keep.swap(proposed_keep);
            if (new_kept == kept) break; // converged
            kept = new_kept;
        }

        // Final weighted mean of kept values
        double wsum = 0.0, wmean = 0.0;
        for (int i = 0; i < n; ++i) {
            if (!keep[static_cast<size_t>(i)]) continue;
            double wi = active_weights[static_cast<size_t>(i)];
            wsum += wi;
            wmean += wi * static_cast<double>(values[static_cast<size_t>(i)]);
        }
        if (wsum > 0.0) {
            out.data()[idx] = static_cast<float>(wmean / wsum);
        } else {
            // Fallback: use all finite values for this pixel.
            wsum = 0.0; wmean = 0.0;
            for (int i = 0; i < n; ++i) {
                const float v = values[static_cast<size_t>(i)];
                if (!is_valid_sample(v)) continue;
                double wi = active_weights[static_cast<size_t>(i)];
                wsum += wi;
                wmean += wi * static_cast<double>(v);
            }
            out.data()[idx] =
                (wsum > 0.0) ? static_cast<float>(wmean / wsum)
                             : invalid_reconstruction_sample();
        }
    }

    return out;
}

/// @brief Implements sigma clip weighted tile with fallback.
/// @details Part of tile reconstruction, sigma clipping, overlap-add, and synthetic stacking helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
WeightedTileResult sigma_clip_weighted_tile_with_fallback(
    const std::vector<Matrix2Df>& tiles, const std::vector<float>& weights,
    float sigma_low, float sigma_high, int max_iters, float min_fraction,
    float eps_weight) {
    WeightedTileResult out;
    if (tiles.empty() || weights.empty() || tiles.size() != weights.size()) {
        return out;
    }

    std::vector<float> effective_weights(weights);
    double wsum = 0.0;
    for (float w : effective_weights) {
        if (std::isfinite(w) && w > 0.0f) {
            wsum += static_cast<double>(w);
        }
    }
    out.effective_weight_sum = static_cast<float>(wsum);

    if (!(wsum > static_cast<double>(eps_weight))) {
        out.fallback_used = true;
        std::fill(effective_weights.begin(), effective_weights.end(), 1.0f);
        out.effective_weight_sum = static_cast<float>(effective_weights.size());
    }

    out.tile = sigma_clip_weighted_tile(tiles, effective_weights,
                                        sigma_low, sigma_high,
                                        max_iters, min_fraction);
    return out;
}

/// @brief Implements sigma clip weighted rgb tile shared mask.
/// @details Part of tile reconstruction, sigma clipping, overlap-add, and synthetic stacking helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
RGBSharedSigmaClipResult sigma_clip_weighted_rgb_tile_shared_mask(
    const std::vector<Matrix2Df>& tiles_r,
    const std::vector<Matrix2Df>& tiles_g,
    const std::vector<Matrix2Df>& tiles_b,
    const std::vector<float>& weights,
    float sigma_low,
    float sigma_high,
    int max_iters,
    float min_fraction,
    float eps_weight) {
    RGBSharedSigmaClipResult out;
    if (tiles_r.empty() || tiles_g.empty() || tiles_b.empty() || weights.empty()) {
        return out;
    }
    if (tiles_r.size() != tiles_g.size() || tiles_r.size() != tiles_b.size()) {
        return out;
    }

    const int rows = tiles_g[0].rows();
    const int cols = tiles_g[0].cols();
    out.R = Matrix2Df::Zero(rows, cols);
    out.G = Matrix2Df::Zero(rows, cols);
    out.B = Matrix2Df::Zero(rows, cols);

    std::vector<const float*> ptr_r;
    std::vector<const float*> ptr_g;
    std::vector<const float*> ptr_b;
    std::vector<double> active_weights;
    ptr_r.reserve(tiles_r.size());
    ptr_g.reserve(tiles_g.size());
    ptr_b.reserve(tiles_b.size());
    active_weights.reserve(weights.size());

    double wsum = 0.0;
    for (size_t i = 0; i < weights.size() && i < tiles_g.size(); ++i) {
        const float w = weights[i];
        if (!(std::isfinite(w) && w > 0.0f)) {
            continue;
        }
        if (tiles_r[i].rows() != rows || tiles_r[i].cols() != cols ||
            tiles_g[i].rows() != rows || tiles_g[i].cols() != cols ||
            tiles_b[i].rows() != rows || tiles_b[i].cols() != cols) {
            continue;
        }
        ptr_r.push_back(tiles_r[i].data());
        ptr_g.push_back(tiles_g[i].data());
        ptr_b.push_back(tiles_b[i].data());
        active_weights.push_back(static_cast<double>(w));
        wsum += static_cast<double>(w);
    }
    out.effective_weight_sum = static_cast<float>(wsum);
    if (ptr_g.empty()) {
        return out;
    }
    if (!(wsum > static_cast<double>(eps_weight))) {
        out.fallback_used = true;
        std::fill(active_weights.begin(), active_weights.end(), 1.0);
        out.effective_weight_sum = static_cast<float>(active_weights.size());
    }

    const int n = static_cast<int>(ptr_g.size());
    std::vector<float> values(static_cast<size_t>(n));
    std::vector<uint8_t> keep(static_cast<size_t>(n), 0u);
    std::vector<uint8_t> proposed_keep(static_cast<size_t>(n), 0u);

    auto reduce_channel = [&](const std::vector<const float*>& ptrs, int idx) {
        double local_wsum = 0.0;
        double local_wmean = 0.0;
        for (int i = 0; i < n; ++i) {
            if (!keep[static_cast<size_t>(i)]) {
                continue;
            }
            const float v = ptrs[static_cast<size_t>(i)][idx];
            if (!is_valid_sample(v)) {
                continue;
            }
            const double wi = active_weights[static_cast<size_t>(i)];
            local_wsum += wi;
            local_wmean += wi * static_cast<double>(v);
        }
        if (local_wsum > 0.0) {
            return static_cast<float>(local_wmean / local_wsum);
        }
        local_wsum = 0.0;
        local_wmean = 0.0;
        for (int i = 0; i < n; ++i) {
            const float v = ptrs[static_cast<size_t>(i)][idx];
            if (!is_valid_sample(v)) {
                continue;
            }
            const double wi = active_weights[static_cast<size_t>(i)];
            local_wsum += wi;
            local_wmean += wi * static_cast<double>(v);
        }
        return (local_wsum > 0.0)
                   ? static_cast<float>(local_wmean / local_wsum)
                   : invalid_reconstruction_sample();
    };

    for (int idx = 0; idx < out.G.size(); ++idx) {
        int n_valid_here = 0;
        for (int i = 0; i < n; ++i) {
            const float v = ptr_g[static_cast<size_t>(i)][idx];
            values[static_cast<size_t>(i)] = v;
            const bool valid_here = is_valid_sample(v);
            keep[static_cast<size_t>(i)] = valid_here ? 1u : 0u;
            if (valid_here) {
                ++n_valid_here;
            }
        }

        if (n_valid_here <= 0) {
            out.R.data()[idx] = invalid_reconstruction_sample();
            out.G.data()[idx] = invalid_reconstruction_sample();
            out.B.data()[idx] = invalid_reconstruction_sample();
            continue;
        }

        if (n > 2 && max_iters > 0) {
            const int min_keep_here = std::max(
                1, static_cast<int>(std::ceil(min_fraction * n_valid_here)));
            int kept = n_valid_here;
            for (int iter = 0; iter < max_iters; ++iter) {
                if (kept <= 1) {
                    break;
                }
                double local_wsum = 0.0;
                double local_wmean = 0.0;
                for (int i = 0; i < n; ++i) {
                    if (!keep[static_cast<size_t>(i)]) {
                        continue;
                    }
                    const double wi = active_weights[static_cast<size_t>(i)];
                    local_wsum += wi;
                    local_wmean += wi * static_cast<double>(values[static_cast<size_t>(i)]);
                }
                if (!(local_wsum > 0.0)) {
                    break;
                }
                local_wmean /= local_wsum;

                double var = 0.0;
                double local_wsum2 = 0.0;
                for (int i = 0; i < n; ++i) {
                    if (!keep[static_cast<size_t>(i)]) {
                        continue;
                    }
                    const double wi = active_weights[static_cast<size_t>(i)];
                    const double d = static_cast<double>(values[static_cast<size_t>(i)]) - local_wmean;
                    var += wi * d * d;
                    local_wsum2 += wi * wi;
                }
                const double n_eff =
                    (local_wsum * local_wsum) / std::max(local_wsum2, kSigmaClipEpsVar);
                const double denom =
                    local_wsum - local_wsum2 / std::max(local_wsum, kSigmaClipEpsVar);
                if (!(n_eff > 2.0 + kSigmaClipEpsNeff) ||
                    !(denom > kSigmaClipEpsVar)) {
                    break;
                }
                const double sd =
                    (var > 0.0 && denom > 0.0) ? std::sqrt(var / denom) : 0.0;
                if (!(sd > 0.0)) {
                    break;
                }

                const double lo = local_wmean - static_cast<double>(sigma_low) * sd;
                const double hi = local_wmean + static_cast<double>(sigma_high) * sd;
                int new_kept = 0;
                proposed_keep = keep;
                for (int i = 0; i < n; ++i) {
                    if (!keep[static_cast<size_t>(i)]) {
                        continue;
                    }
                    const double v = static_cast<double>(values[static_cast<size_t>(i)]);
                    if (v < lo || v > hi) {
                        proposed_keep[static_cast<size_t>(i)] = 0u;
                    } else {
                        ++new_kept;
                    }
                }
                if (new_kept < min_keep_here) {
                    break;
                }
                keep.swap(proposed_keep);
                if (new_kept == kept) {
                    break;
                }
                kept = new_kept;
            }
        }

        out.R.data()[idx] = reduce_channel(ptr_r, idx);
        out.G.data()[idx] = reduce_channel(ptr_g, idx);
        out.B.data()[idx] = reduce_channel(ptr_b, idx);
    }

    return out;
}

/// @brief Creates partition window 1d.
/// @details Part of tile reconstruction, sigma clipping, overlap-add, and synthetic stacking helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::vector<float> make_partition_window_1d(int n, int left_overlap,
                                            int right_overlap) {
    std::vector<float> w;
    if (n <= 0) {
        return w;
    }
    w.assign(static_cast<size_t>(n), 1.0f);
    left_overlap = std::clamp(left_overlap, 0, n);
    right_overlap = std::clamp(right_overlap, 0, n);
    if (left_overlap + right_overlap > n) {
        const int overflow = left_overlap + right_overlap - n;
        right_overlap = std::max(0, right_overlap - overflow);
    }
    const float pi = 3.14159265358979323846f;
    if (left_overlap > 0) {
        for (int i = 0; i < left_overlap; ++i) {
            const float s =
                (static_cast<float>(i) + 0.5f) / static_cast<float>(left_overlap);
            const float angle = 0.5f * pi * s;
            w[static_cast<size_t>(i)] = std::sin(angle) * std::sin(angle);
        }
    }
    if (right_overlap > 0) {
        for (int i = 0; i < right_overlap; ++i) {
            const int idx = n - right_overlap + i;
            const float s =
                (static_cast<float>(i) + 0.5f) / static_cast<float>(right_overlap);
            const float angle = 0.5f * pi * s;
            w[static_cast<size_t>(idx)] = std::cos(angle) * std::cos(angle);
        }
    }
    return w;
}

} // namespace tile_compile::reconstruction

// ---------------------------------------------------------------------------
// reconstruct_tiles_parallel — parallel implementation (B1 + B2 + B3)
// ---------------------------------------------------------------------------
#include "tile_compile/reconstruction/dead_tile_detector.hpp"
#include "tile_compile/reconstruction/memory_budget.hpp"
#include "tile_compile/reconstruction/progress_reporter.hpp"
#include "tile_compile/reconstruction/tile_scheduler.hpp"

#include <mutex>

namespace tile_compile::reconstruction {

/// @brief Implements reconstruct tiles parallel.
/// @details Part of tile reconstruction, sigma clipping, overlap-add, and synthetic stacking helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
ReconstructTilesResult reconstruct_tiles_parallel(
    const std::vector<Matrix2Df>&          frames,
    const TileGrid&                        grid,
    const std::vector<std::vector<float>>& tile_weights,
    const std::vector<bool>&               dead_tile_mask,
    const ReconstructionConfig&            cfg)
{
    ReconstructTilesResult result;
    if (frames.empty() || grid.tiles.empty()) return result;

    const int h = frames[0].rows();
    const int w = frames[0].cols();
    const int num_frames   = static_cast<int>(frames.size());
    const int num_tiles    = static_cast<int>(grid.tiles.size());
    const int frame_ch     = 1; // single-channel (mono) frames
    const int tile_ch      = 1;

    // Determine max tile dimensions for OLA buffer sizing.
    int max_tw = 0, max_th = 0;
    for (const auto& t : grid.tiles) {
        max_tw = std::max(max_tw, t.width);
        max_th = std::max(max_th, t.height);
    }

    // --- B2: Memory budget plan ---
    const MemoryBudgetPlan plan = compute_memory_budget_plan(
        num_frames, h, w, frame_ch,
        num_tiles, max_tw, max_th, tile_ch,
        cfg.memory_budget_bytes,
        cfg.parallel_workers);

    result.allocated_frame_batch_bytes = plan.allocated_frame_batch_bytes;
    result.allocated_tile_batch_bytes  = plan.allocated_tile_batch_bytes;

    // --- B3: Dead tile mask (caller-supplied; may be empty → all alive) ---
    const std::vector<bool>& dead = dead_tile_mask;

    // --- OLA accumulators (shared, mutex-protected) ---
    Matrix2Df accum      = Matrix2Df::Zero(h, w);
    Matrix2Df weight_sum = Matrix2Df::Zero(h, w);
    std::mutex ola_mutex;

    // Pre-build neighbour lookup for partition windows (same as legacy impl).
    std::unordered_map<uint64_t, size_t> tile_by_grid;
    tile_by_grid.reserve(static_cast<size_t>(num_tiles));
    for (size_t ti = 0; ti < static_cast<size_t>(num_tiles); ++ti) {
        const auto& t = grid.tiles[ti];
        tile_by_grid.emplace(tile_grid_key(t.row, t.col), ti);
    }

    // --- Scheduler config ---
    TileSchedulerConfig sched_cfg;
    sched_cfg.num_workers          = plan.effective_workers;
    sched_cfg.frame_sub_batch_size = plan.frame_sub_batch_size;

    // Process function called per tile per sub-batch.
    auto process_fn = [&](const Tile& tile, size_t tile_idx,
                          size_t sb_start, size_t sb_end) {
        // Build partition window for this tile.
        if (tile.width <= 0 || tile.height <= 0) {
            return;
        }
        const TileOverlaps overlaps =
            compute_tile_overlaps(tile, tile_by_grid, grid.tiles);

        const auto wx =
            make_partition_window_1d(tile.width, overlaps.left, overlaps.right);
        const auto wy =
            make_partition_window_1d(tile.height, overlaps.top, overlaps.bottom);

        // Local accumulators for this sub-batch contribution.
        Matrix2Df local_accum      = Matrix2Df::Zero(tile.height, tile.width);
        Matrix2Df local_weight_sum = Matrix2Df::Zero(tile.height, tile.width);

        const int y_begin = std::max(0, tile.y);
        const int y_end = std::min(h, tile.y + tile.height);
        const int x_begin = std::max(0, tile.x);
        const int x_end = std::min(w, tile.x + tile.width);
        if (y_begin >= y_end || x_begin >= x_end) {
            return;
        }

        for (size_t f = sb_start; f < sb_end; ++f) {
            if (f >= tile_weights.size() || !same_shape(frames[f], h, w)) {
                continue;
            }
            const float weight = (tile_idx < tile_weights[f].size())
                                     ? tile_weights[f][tile_idx]
                                     : 0.0f;
            if (!(std::isfinite(weight) && weight > 0.0f)) continue;

            for (int y = y_begin; y < y_end; ++y) {
                for (int x = x_begin; x < x_end; ++x) {
                    const int ly = y - tile.y;
                    const int lx = x - tile.x;
                    if (ly < 0 || lx < 0 ||
                        ly >= static_cast<int>(wy.size()) ||
                        lx >= static_cast<int>(wx.size())) continue;
                    const float win = wy[static_cast<size_t>(ly)] *
                                      wx[static_cast<size_t>(lx)];
                    const float ww  = weight * win;
                    const float v   = frames[f](y, x);
                    if (!std::isfinite(v)) continue;
                    local_accum(ly, lx)      += v * ww;
                    local_weight_sum(ly, lx) += ww;
                }
            }
        }

        // Merge local accumulators into shared ones.
        {
            std::lock_guard<std::mutex> lk(ola_mutex);
            for (int y = y_begin; y < y_end; ++y)
                for (int x = x_begin; x < x_end; ++x) {
                    const int ly = y - tile.y;
                    const int lx = x - tile.x;
                    accum(y, x)      += local_accum(ly, lx);
                    weight_sum(y, x) += local_weight_sum(ly, lx);
                }
        }
    };

    // --- Run scheduler ---
    const TileSchedulerResult sched = run_tile_scheduler(
        grid, static_cast<size_t>(num_frames), dead, sched_cfg, process_fn);

    // --- Normalise ---
    for (int i = 0; i < accum.size(); ++i) {
        if (weight_sum.data()[i] > 0.0f)
            accum.data()[i] /= weight_sum.data()[i];
    }

    result.output                          = std::move(accum);
    result.tiles_processed                 = sched.tiles_processed;
    result.tiles_skipped_dead              = sched.tiles_skipped_dead;
    result.duration_s                      = sched.processing_time_s;
    result.dead_tile_time_saved_estimate_s = sched.dead_tile_time_saved_estimate_s;
    result.workers_used                    = sched.workers_used;
    return result;
}

} // namespace tile_compile::reconstruction
