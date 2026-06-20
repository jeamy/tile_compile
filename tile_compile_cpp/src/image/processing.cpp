#include "tile_compile/image/processing.hpp"
#include "tile_compile/core/errors.hpp"
#include "tile_compile/core/utils.hpp"

#include <opencv2/opencv.hpp>
#include <array>
#include <algorithm>
#include <cmath>

namespace tile_compile::image {

/// @brief Splits cfa channels.
/// @details Part of general image processing, cosmetic correction, and tile extraction helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::map<std::string, Matrix2Df> split_cfa_channels(const Matrix2Df& mosaic, BayerPattern pattern) {
    int h = mosaic.rows();
    int w = mosaic.cols();
    int hh = h / 2;
    int hw = w / 2;
    
    Matrix2Df R(hh, hw), G(hh, hw), B(hh, hw);
    
    const auto [r_row, r_col, b_row, b_col] = get_bayer_offsets(pattern);
    
    for (int y = 0; y < hh; ++y) {
        for (int x = 0; x < hw; ++x) {
            R(y, x) = mosaic(2*y + r_row, 2*x + r_col);
            B(y, x) = mosaic(2*y + b_row, 2*x + b_col);
            
            float g1 = mosaic(2*y + r_row, 2*x + (1 - r_col));
            float g2 = mosaic(2*y + (1 - r_row), 2*x + r_col);
            G(y, x) = (g1 + g2) / 2.0f;
        }
    }
    
    return {{"R", R}, {"G", G}, {"B", B}};
}

/// @brief Implements reassemble cfa mosaic.
/// @details Part of general image processing, cosmetic correction, and tile extraction helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Matrix2Df reassemble_cfa_mosaic(const Matrix2Df& R, const Matrix2Df& G, const Matrix2Df& B, 
                                 BayerPattern pattern) {
    int hh = R.rows();
    int hw = R.cols();
    int h = hh * 2;
    int w = hw * 2;
    
    Matrix2Df mosaic(h, w);
    
    const auto [r_row, r_col, b_row, b_col] = get_bayer_offsets(pattern);
    
    for (int y = 0; y < hh; ++y) {
        for (int x = 0; x < hw; ++x) {
            mosaic(2*y + r_row, 2*x + r_col) = R(y, x);
            mosaic(2*y + b_row, 2*x + b_col) = B(y, x);
            mosaic(2*y + r_row, 2*x + (1 - r_col)) = G(y, x);
            mosaic(2*y + (1 - r_row), 2*x + r_col) = G(y, x);
        }
    }
    
    return mosaic;
}

/// @brief Normalizes frame.
/// @details Part of general image processing, cosmetic correction, and tile extraction helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Matrix2Df normalize_frame(const Matrix2Df& frame, float target_background, 
                          float target_scale, NormalizationMode mode) {
    std::vector<float> pixels(frame.data(), frame.data() + frame.size());
    float median = core::median_of(pixels);
    
    Matrix2Df result = frame;
    
    if (mode == NormalizationMode::BACKGROUND) {
        float scale = target_background / std::max(median, 1e-6f);
        result = frame * scale;
    } else {
        float offset = target_background - median;
        result = frame.array() + offset;
        result = result * target_scale;
    }
    
    return result;
}

/// @brief Implements cosmetic correction cfa.
/// @details Part of general image processing, cosmetic correction, and tile extraction helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Matrix2Df cosmetic_correction_cfa(const Matrix2Df& mosaic, float sigma_threshold,
                                 bool correct_hot, int origin_x, int origin_y) {
    if (mosaic.size() == 0) return mosaic;
    Matrix2Df result = mosaic;
    const int h = mosaic.rows();
    const int w = mosaic.cols();

    struct Stats {
        float median = 0.0f;
        float mad = 0.0f;
        float sigma = 0.0f;
        float threshold = 0.0f;
        float cold_threshold = 0.0f;
        float neighbor_threshold = 0.0f;
        bool ok = false;
    } stats[2][2];

    auto compute_stats = [&](int py, int px) {
        std::vector<float> vals;
        vals.reserve(static_cast<size_t>((h * w) / 4));
        for (int y = 0; y < h; ++y) {
            const int ypar = (origin_y + y) & 1;
            if (ypar != py) continue;
            for (int x = 0; x < w; ++x) {
                const int xpar = (origin_x + x) & 1;
                if (xpar != px) continue;
                const float value = mosaic(y, x);
                if (std::isfinite(value)) {
                    vals.push_back(value);
                }
            }
        }
        Stats s;
        if (vals.empty()) {
            stats[py][px] = s;
            return;
        }
        s.median = core::median_of(vals);
        s.mad = core::mad_of(vals, s.median);
        s.sigma = 1.4826f * s.mad;
        s.threshold = s.median + sigma_threshold * s.sigma;
        s.cold_threshold = s.median - sigma_threshold * s.sigma;
        s.neighbor_threshold = s.threshold;
        s.ok = true;
        stats[py][px] = s;
    };

    if (!correct_hot) {
        return result;
    }

    compute_stats(0, 0);
    compute_stats(0, 1);
    compute_stats(1, 0);
    compute_stats(1, 1);

    auto in_bounds = [&](int yy, int xx) -> bool {
        return yy >= 0 && yy < h && xx >= 0 && xx < w;
    };

    for (int y = 2; y < h - 2; ++y) {
        const int py = (origin_y + y) & 1;
        for (int x = 2; x < w - 2; ++x) {
            const int px = (origin_x + x) & 1;
            const Stats& s = stats[py][px];
            if (!s.ok) continue;

            const float v = mosaic(y, x);
            if (!std::isfinite(v)) continue;

            std::vector<float> same_color_neighbors;
            same_color_neighbors.reserve(8);
            int hot_neighbor_count = 0;
            for (int dy : {-2, 0, 2}) {
                for (int dx : {-2, 0, 2}) {
                    if (dy == 0 && dx == 0) continue;
                    const int yy = y + dy;
                    const int xx = x + dx;
                    if (!in_bounds(yy, xx)) continue;
                    const float neighbor = mosaic(yy, xx);
                    if (!std::isfinite(neighbor)) continue;
                    same_color_neighbors.push_back(neighbor);
                    if (neighbor > s.neighbor_threshold) {
                        ++hot_neighbor_count;
                    }
                }
            }

            const bool global_candidate_raw =
                (v > s.threshold) &&
                (hot_neighbor_count <= 1);
            const bool global_candidate_cold =
                (v < s.cold_threshold);

            bool local_candidate = false;
            bool extreme_outlier = false;
            bool cold_outlier = false;
            float replacement_value = 0.0f;
            if (same_color_neighbors.size() >= 4u) {
                const float local_median = core::median_of(same_color_neighbors);
                const float local_sigma =
                    1.4826f * core::mad_of(same_color_neighbors, local_median);
                const float local_floor = std::max(
                    {2.0f * local_sigma,
                     0.35f * sigma_threshold * s.sigma,
                     0.01f * std::max(1.0f, std::abs(local_median)),
                     1.0e-6f});
                const float support_threshold =
                    local_median + 0.5f * local_floor;
                int same_color_support = 0;
                for (float neighbor : same_color_neighbors) {
                    if (neighbor > support_threshold) {
                        ++same_color_support;
                    }
                }
                local_candidate =
                    (v > local_median + local_floor) &&
                    (same_color_support <= 1);
                extreme_outlier =
                    (v > local_median + 5.0f * local_floor);
                cold_outlier =
                    (v < local_median - local_floor);
                replacement_value = local_median;
            }

            const bool should_correct =
                (global_candidate_raw || local_candidate || extreme_outlier ||
                 global_candidate_cold || cold_outlier);

            if (should_correct) {
                if (same_color_neighbors.size() >= 4u) {
                    result(y, x) = replacement_value;
                } else {
                    float sum = 0.0f;
                    int n = 0;
                    const int yy4[4] = {y - 2, y + 2, y, y};
                    const int xx4[4] = {x, x, x - 2, x + 2};
                    for (int i = 0; i < 4; ++i) {
                        if (!in_bounds(yy4[i], xx4[i])) continue;
                        const float neighbor = mosaic(yy4[i], xx4[i]);
                        if (!std::isfinite(neighbor)) continue;
                        sum += neighbor;
                        ++n;
                    }
                    if (n >= 2) {
                        result(y, x) = sum / static_cast<float>(n);
                    }
                }
            }
        }
    }

    return result;
}

/// @brief Implements cosmetic correction.
/// @details Part of general image processing, cosmetic correction, and tile extraction helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Matrix2Df cosmetic_correction(const Matrix2Df& frame, float sigma_threshold, bool correct_hot) {
    if (frame.size() == 0) return frame;
    Matrix2Df result = frame;
    int h = frame.rows();
    int w = frame.cols();
    
    std::vector<float> frame_values(frame.data(), frame.data() + frame.size());
    const float median = core::median_of(frame_values);
    const float mad = core::mad_of(frame_values, median);

    float sigma = 1.4826f * mad;
    float threshold = median + sigma_threshold * sigma;
    float neighbor_threshold = median + (0.5f * sigma_threshold) * sigma;
    
    if (correct_hot) {
        for (int y = 1; y < h - 1; ++y) {
            for (int x = 1; x < w - 1; ++x) {
                if (frame(y, x) > threshold) {
                    int hot_neighbor_count = 0;
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (dy == 0 && dx == 0) continue;
                            if (frame(y + dy, x + dx) > neighbor_threshold) {
                                ++hot_neighbor_count;
                            }
                        }
                    }
                    if (hot_neighbor_count <= 1) {
                        float sum = 0.0f;
                        int n = 0;
                        if (y - 2 >= 0) { sum += frame(y - 2, x); ++n; }
                        if (y + 2 < h) { sum += frame(y + 2, x); ++n; }
                        if (x - 2 >= 0) { sum += frame(y, x - 2); ++n; }
                        if (x + 2 < w) { sum += frame(y, x + 2); ++n; }
                        if (n >= 2) {
                            result(y, x) = sum / static_cast<float>(n);
                        } else {
                            float neighbors = (frame(y-1, x) + frame(y+1, x) + 
                                               frame(y, x-1) + frame(y, x+1)) / 4.0f;
                            result(y, x) = neighbors;
                        }
                    }
                }
            }
        }
    }
    
    return result;
}

/// @brief Implements suppress isolated chroma speckles rgb inplace.
/// @details Part of general image processing, cosmetic correction, and tile extraction helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
ChromaSpeckleSuppressionStats suppress_isolated_chroma_speckles_rgb_inplace(
    Matrix2Df& R, Matrix2Df& G, Matrix2Df& B,
    const std::vector<uint8_t>* valid_mask, int mask_rows, int mask_cols) {
    ChromaSpeckleSuppressionStats stats;
    if (R.size() == 0 || G.size() == 0 || B.size() == 0) return stats;
    if (R.rows() != G.rows() || R.rows() != B.rows() ||
        R.cols() != G.cols() || R.cols() != B.cols()) {
        return stats;
    }

    const int h = R.rows();
    const int w = R.cols();
    const bool use_mask =
        (valid_mask != nullptr && mask_rows == h && mask_cols == w &&
         static_cast<int>(valid_mask->size()) == h * w);

    auto is_supported = [&](const Matrix2Df& srcR, const Matrix2Df& srcG,
                            const Matrix2Df& srcB, int y, int x) -> bool {
        if (use_mask && (*valid_mask)[static_cast<size_t>(y * w + x)] == 0u) {
            return false;
        }
        return std::isfinite(srcR(y, x)) && std::isfinite(srcG(y, x)) &&
               std::isfinite(srcB(y, x));
    };

    auto luma = [](float r, float g, float b) -> float {
        return 0.25f * r + 0.50f * g + 0.25f * b;
    };

    constexpr int kRadius = 2;
    constexpr int kPasses = 2;

    // Pre-allocate source buffers once. The per-pass copy is required because
    // the algorithm reads neighbors while writing pixels in-place.
    Matrix2Df srcR(R.rows(), R.cols());
    Matrix2Df srcG(G.rows(), G.cols());
    Matrix2Df srcB(B.rows(), B.cols());
    for (int pass = 0; pass < kPasses; ++pass) {
        srcR = R;
        srcG = G;
        srcB = B;

        for (int y = kRadius; y < h - kRadius; ++y) {
            for (int x = kRadius; x < w - kRadius; ++x) {
                if (!is_supported(srcR, srcG, srcB, y, x)) continue;

                std::vector<float> neighR;
                std::vector<float> neighG;
                std::vector<float> neighB;
                std::vector<float> neighL;
                neighR.reserve(24);
                neighG.reserve(24);
                neighB.reserve(24);
                neighL.reserve(24);

                int bright_support = 0;
                for (int dy = -kRadius; dy <= kRadius; ++dy) {
                    for (int dx = -kRadius; dx <= kRadius; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        const int yy = y + dy;
                        const int xx = x + dx;
                        if (!is_supported(srcR, srcG, srcB, yy, xx)) continue;
                        const float nr = srcR(yy, xx);
                        const float ng = srcG(yy, xx);
                        const float nb = srcB(yy, xx);
                        neighR.push_back(nr);
                        neighG.push_back(ng);
                        neighB.push_back(nb);
                        neighL.push_back(luma(nr, ng, nb));
                    }
                }

                if (neighR.size() < 12u) continue;

                const float medR = core::median_of(neighR);
                const float medG = core::median_of(neighG);
                const float medB = core::median_of(neighB);
                const float medL = core::median_of(neighL);
                if (!(std::isfinite(medL) && medL > 0.0f)) continue;

                const float madL = 1.4826f * core::mad_of(neighL, medL);
                const float curR = srcR(y, x);
                const float curG = srcG(y, x);
                const float curB = srcB(y, x);
                const float curL = luma(curR, curG, curB);

                const float bright_support_threshold =
                    medL + std::max(3.0f * madL, 0.08f * medL);
                for (float nl : neighL) {
                    if (nl > bright_support_threshold) {
                        ++bright_support;
                    }
                }

                const float resR = std::abs(curR - medR);
                const float resG = std::abs(curG - medG);
                const float resB = std::abs(curB - medB);

                const float thrFloor = 0.010f * medL + 1.0e-3f;
                const float thrR =
                    std::max(4.5f * 1.4826f * core::mad_of(neighR, medR), thrFloor);
                const float thrG =
                    std::max(4.5f * 1.4826f * core::mad_of(neighG, medG), thrFloor);
                const float thrB =
                    std::max(4.5f * 1.4826f * core::mad_of(neighB, medB), thrFloor);

                const bool badR = resR > thrR;
                const bool badG = resG > thrG;
                const bool badB = resB > thrB;
                const int badCount = static_cast<int>(badR) + static_cast<int>(badG) +
                                     static_cast<int>(badB);
                if (badCount == 0 || badCount > 2) continue;

                // Guard real compact structure: fix isolated color defects,
                // not stars or coherent bright detail. Allow strongly chromatic
                // defects to pass this guard even if their luma is elevated.
                const float curMax = std::max({curR, curG, curB});
                const float curMin = std::min({curR, curG, curB});
                const float medMax = std::max({medR, medG, medB});
                const float medMin = std::min({medR, medG, medB});
                const float luma_guard = std::max(8.0f * madL, 0.35f * medL);
                const float chroma_guard =
                    std::max(6.0f * madL, 0.20f * std::max(1.0f, medL));
                if (curL > medL + luma_guard &&
                    (curMax - curMin) < (medMax - medMin) + chroma_guard) {
                    continue;
                }
                if (bright_support >= 6) continue;

                std::array<float, 3> residuals{resR, resG, resB};
                std::sort(residuals.begin(), residuals.end(), std::greater<float>());
                if (badCount == 1) {
                    if (!(residuals[0] >
                          std::max(1.20f * residuals[1], 1.10f * thrFloor))) {
                        continue;
                    }
                } else {
                    if (!(residuals[1] >
                          std::max(1.05f * residuals[2], 1.05f * thrFloor))) {
                        continue;
                    }
                }

                const float curSpan = curMax - curMin;
                const float medSpan = medMax - medMin;
                if (curSpan < medSpan + std::max(4.0f * madL, 0.05f * medL)) continue;

                int similar_bad_support = 0;
                for (int dy = -kRadius; dy <= kRadius; ++dy) {
                    for (int dx = -kRadius; dx <= kRadius; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        const int yy = y + dy;
                        const int xx = x + dx;
                        if (!is_supported(srcR, srcG, srcB, yy, xx)) continue;
                        int supporting_channels = 0;
                        if (badR &&
                            std::abs(srcR(yy, xx) - medR) > 0.75f * thrR) {
                            ++supporting_channels;
                        }
                        if (badG &&
                            std::abs(srcG(yy, xx) - medG) > 0.75f * thrG) {
                            ++supporting_channels;
                        }
                        if (badB &&
                            std::abs(srcB(yy, xx) - medB) > 0.75f * thrB) {
                            ++supporting_channels;
                        }
                        if (supporting_channels > 0) {
                            ++similar_bad_support;
                        }
                    }
                }

                if (similar_bad_support > 3) continue;
                if (pass == 0) {
                    ++stats.candidate_pixels;
                }

                if (badR) {
                    R(y, x) = medR;
                }
                if (badG) {
                    G(y, x) = medG;
                }
                if (badB) {
                    B(y, x) = medB;
                }
                ++stats.corrected_pixels;
            }
        }
    }

    return stats;
}

/// @brief Extracts tile.
/// @details Part of general image processing, cosmetic correction, and tile extraction helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Matrix2Df extract_tile(const Matrix2Df& img, const Tile& t) {
    int cols = static_cast<int>(img.cols());
    int rows = static_cast<int>(img.rows());
    int x0 = std::max(0, t.x);
    int y0 = std::max(0, t.y);
    int x1 = std::min(cols, t.x + t.width);
    int y1 = std::min(rows, t.y + t.height);
    int tw = std::max(0, x1 - x0);
    int th = std::max(0, y1 - y0);
    if (tw <= 0 || th <= 0)
        return Matrix2Df();
    return img.block(y0, x0, th, tw);
}

} // namespace tile_compile::image
