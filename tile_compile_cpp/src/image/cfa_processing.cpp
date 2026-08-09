#include "tile_compile/image/cfa_processing.hpp"
#include "tile_compile/core/cfa_warp.hpp"

#include <opencv2/opencv.hpp>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace tile_compile::image {

namespace {

enum class CfaColor : uint8_t {
    Red = 0,
    Green = 1,
    Blue = 2,
};

/// @brief Implements clamp index.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
inline int clamp_index(int v, int lo, int hi) {
    return std::max(lo, std::min(hi, v));
}

/// @brief Implements fill bayer color lut.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
inline void fill_bayer_color_lut(BayerPattern pattern, uint8_t color_lut[4]) {
    if (pattern == BayerPattern::UNKNOWN) {
        pattern = BayerPattern::GBRG;
    }

    switch (pattern) {
        case BayerPattern::RGGB:
            color_lut[0] = static_cast<uint8_t>(CfaColor::Red);
            color_lut[1] = static_cast<uint8_t>(CfaColor::Green);
            color_lut[2] = static_cast<uint8_t>(CfaColor::Green);
            color_lut[3] = static_cast<uint8_t>(CfaColor::Blue);
            break;
        case BayerPattern::BGGR:
            color_lut[0] = static_cast<uint8_t>(CfaColor::Blue);
            color_lut[1] = static_cast<uint8_t>(CfaColor::Green);
            color_lut[2] = static_cast<uint8_t>(CfaColor::Green);
            color_lut[3] = static_cast<uint8_t>(CfaColor::Red);
            break;
        case BayerPattern::GRBG:
            color_lut[0] = static_cast<uint8_t>(CfaColor::Green);
            color_lut[1] = static_cast<uint8_t>(CfaColor::Red);
            color_lut[2] = static_cast<uint8_t>(CfaColor::Blue);
            color_lut[3] = static_cast<uint8_t>(CfaColor::Green);
            break;
        case BayerPattern::GBRG:
        default:
            color_lut[0] = static_cast<uint8_t>(CfaColor::Green);
            color_lut[1] = static_cast<uint8_t>(CfaColor::Blue);
            color_lut[2] = static_cast<uint8_t>(CfaColor::Red);
            color_lut[3] = static_cast<uint8_t>(CfaColor::Green);
            break;
    }
}

/// @brief Implements sample clamped strided.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
inline float sample_clamped_strided(const float* data, int h, int w, int stride,
                                    int y, int x) {
    const int cy = clamp_index(y, 0, h - 1);
    const int cx = clamp_index(x, 0, w - 1);
    return data[static_cast<size_t>(cy) * static_cast<size_t>(stride) +
                static_cast<size_t>(cx)];
}

/// @brief Implements sample clamped.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
inline float sample_clamped(const float* data, int h, int w, int y, int x) {
    return sample_clamped_strided(data, h, w, w, y, x);
}

/// @brief Implements average neighbors of color strided.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
inline float average_neighbors_of_color_strided(const float* data,
                                                int h,
                                                int w,
                                                int stride,
                                                int origin_y,
                                                int origin_x,
                                                const uint8_t color_lut[4],
                                                const int* ys,
                                                const int* xs,
                                                int n,
                                                CfaColor desired_color) {
    float sum = 0.0f;
    int count = 0;
    const uint8_t desired = static_cast<uint8_t>(desired_color);
    for (int i = 0; i < n; ++i) {
        const int cy = clamp_index(ys[i], 0, h - 1);
        const int cx = clamp_index(xs[i], 0, w - 1);
        const int parity_idx = (((origin_y + cy) & 1) << 1) | ((origin_x + cx) & 1);
        if (color_lut[parity_idx] != desired) {
            continue;
        }
        const float v = data[static_cast<size_t>(cy) * static_cast<size_t>(stride) +
                             static_cast<size_t>(cx)];
        if (!(std::isfinite(v) && v > 0.0f)) {
            continue;
        }
        sum += v;
        ++count;
    }
    if (count > 0) {
        return sum / static_cast<float>(count);
    }
    const float fallback = sample_clamped_strided(data, h, w, stride, ys[0], xs[0]);
    return (std::isfinite(fallback) && fallback > 0.0f) ? fallback : 0.0f;
}

/// @brief Implements average neighbors of color.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
inline float average_neighbors_of_color(const float* data,
                                        int h,
                                        int w,
                                        int origin_y,
                                        int origin_x,
                                        const uint8_t color_lut[4],
                                        const int* ys,
                                        const int* xs,
                                        int n,
                                        CfaColor desired_color) {
    return average_neighbors_of_color_strided(
        data, h, w, w, origin_y, origin_x, color_lut, ys, xs, n,
        desired_color);
}

/// @brief Implements debayer bilinear core.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void debayer_bilinear_core(const float* src, int h, int w, int stride,
                           BayerPattern pattern, int origin_x, int origin_y,
                           Matrix2Df& R_out, Matrix2Df& G_out,
                           Matrix2Df& B_out) {
    R_out.resize(h, w);
    G_out.resize(h, w);
    B_out.resize(h, w);
    R_out.setZero();
    G_out.setZero();
    B_out.setZero();

    int r_row = 1, r_col = 0;
    if (pattern == BayerPattern::UNKNOWN) {
        pattern = BayerPattern::GBRG;
    } else if (pattern == BayerPattern::RGGB) {
        r_row = 0; r_col = 0;
    } else if (pattern == BayerPattern::BGGR) {
        r_row = 1; r_col = 1;
    } else if (pattern == BayerPattern::GRBG) {
        r_row = 0; r_col = 1;
    }
    uint8_t color_lut[4] = {0, 0, 0, 0};
    fill_bayer_color_lut(pattern, color_lut);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int parity_idx = (((origin_y + y) & 1) << 1) | ((origin_x + x) & 1);
            const uint8_t color = color_lut[parity_idx];
            float r_val = 0.0f;
            float g_val = 0.0f;
            float b_val = 0.0f;

            if (color == static_cast<uint8_t>(CfaColor::Red)) {
                r_val = sample_clamped_strided(src, h, w, stride, y, x);
                const int gy[4] = {y - 1, y + 1, y, y};
                const int gx[4] = {x, x, x - 1, x + 1};
                g_val = average_neighbors_of_color_strided(
                    src, h, w, stride, origin_y, origin_x, color_lut, gy, gx,
                    4, CfaColor::Green);
                const int by[4] = {y - 1, y - 1, y + 1, y + 1};
                const int bx[4] = {x - 1, x + 1, x - 1, x + 1};
                b_val = average_neighbors_of_color_strided(
                    src, h, w, stride, origin_y, origin_x, color_lut, by, bx,
                    4, CfaColor::Blue);
            } else if (color == static_cast<uint8_t>(CfaColor::Blue)) {
                b_val = sample_clamped_strided(src, h, w, stride, y, x);
                const int gy[4] = {y - 1, y + 1, y, y};
                const int gx[4] = {x, x, x - 1, x + 1};
                g_val = average_neighbors_of_color_strided(
                    src, h, w, stride, origin_y, origin_x, color_lut, gy, gx,
                    4, CfaColor::Green);
                const int ry[4] = {y - 1, y - 1, y + 1, y + 1};
                const int rx[4] = {x - 1, x + 1, x - 1, x + 1};
                r_val = average_neighbors_of_color_strided(
                    src, h, w, stride, origin_y, origin_x, color_lut, ry, rx,
                    4, CfaColor::Red);
            } else {
                g_val = sample_clamped_strided(src, h, w, stride, y, x);
                const bool green_on_red_row = (((origin_y + y) & 1) == r_row);
                if (green_on_red_row) {
                    const int ry[2] = {y, y};
                    const int rx[2] = {x - 1, x + 1};
                    r_val = average_neighbors_of_color_strided(
                        src, h, w, stride, origin_y, origin_x, color_lut,
                        ry, rx, 2, CfaColor::Red);
                    const int by[2] = {y - 1, y + 1};
                    const int bx[2] = {x, x};
                    b_val = average_neighbors_of_color_strided(
                        src, h, w, stride, origin_y, origin_x, color_lut,
                        by, bx, 2, CfaColor::Blue);
                } else {
                    const int ry[2] = {y - 1, y + 1};
                    const int rx[2] = {x, x};
                    r_val = average_neighbors_of_color_strided(
                        src, h, w, stride, origin_y, origin_x, color_lut,
                        ry, rx, 2, CfaColor::Red);
                    const int by[2] = {y, y};
                    const int bx[2] = {x - 1, x + 1};
                    b_val = average_neighbors_of_color_strided(
                        src, h, w, stride, origin_y, origin_x, color_lut,
                        by, bx, 2, CfaColor::Blue);
                }
            }

            R_out(y, x) = r_val;
            G_out(y, x) = g_val;
            B_out(y, x) = b_val;
        }
    }
}

/// @brief Implements debayer nearest neighbor core.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers.
void debayer_nearest_neighbor_core(const float* src, int h, int w, int stride,
                                   BayerPattern pattern, int origin_x, int origin_y,
                                   Matrix2Df& R_out, Matrix2Df& G_out,
                                   Matrix2Df& B_out) {
    R_out.resize(h, w);
    G_out.resize(h, w);
    B_out.resize(h, w);

    uint8_t color_lut[4] = {0, 0, 0, 0};
    fill_bayer_color_lut(pattern, color_lut);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float r = 0, g = 0, b = 0;
            int abs_y = origin_y + y;
            int abs_x = origin_x + x;
            int block_y = abs_y & ~1;
            int block_x = abs_x & ~1;

            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    int py = block_y + dy - origin_y;
                    int px = block_x + dx - origin_x;
                    float val = sample_clamped_strided(src, h, w, stride, py, px);
                    int parity = (dy << 1) | dx;
                    CfaColor c = static_cast<CfaColor>(color_lut[parity]);
                    if (c == CfaColor::Red) r = val;
                    else if (c == CfaColor::Blue) b = val;
                    else g = val;
                }
            }

            float v = sample_clamped_strided(src, h, w, stride, y, x);
            int current_parity = ((abs_y & 1) << 1) | (abs_x & 1);
            CfaColor current_c = static_cast<CfaColor>(color_lut[current_parity]);
            if (current_c == CfaColor::Red) r = v;
            else if (current_c == CfaColor::Blue) b = v;
            else g = v;

            R_out(y, x) = r;
            G_out(y, x) = g;
            B_out(y, x) = b;
        }
    }
}

/// @brief Implements average neighbors of color absolute.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
inline float average_neighbors_of_color_absolute(const float* data,
                                                int h,
                                                int w,
                                                const uint8_t color_lut[4],
                                                const int* ys,
                                                const int* xs,
                                                int n,
                                                CfaColor desired_color) {
    float sum = 0.0f;
    int count = 0;
    const uint8_t desired = static_cast<uint8_t>(desired_color);
    for (int i = 0; i < n; ++i) {
        const int cy = clamp_index(ys[i], 0, h - 1);
        const int cx = clamp_index(xs[i], 0, w - 1);
        const int parity_idx = ((cy & 1) << 1) | (cx & 1);
        if (color_lut[parity_idx] != desired) {
            continue;
        }
        const float v = data[static_cast<size_t>(cy) * static_cast<size_t>(w) +
                             static_cast<size_t>(cx)];
        if (!(std::isfinite(v) && v > 0.0f)) {
            continue;
        }
        sum += v;
        ++count;
    }
    if (count > 0) {
        return sum / static_cast<float>(count);
    }
    const float fallback = sample_clamped(data, h, w, ys[0], xs[0]);
    return (std::isfinite(fallback) && fallback > 0.0f) ? fallback : 0.0f;
}

} // namespace

/// @brief Implements cfa green mask.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Matrix2Df cfa_green_mask(int height, int width, const std::string& bayer_pattern) {
    Matrix2Df mask = Matrix2Df::Zero(height, width);
    
    std::string bp = bayer_pattern;
    std::transform(bp.begin(), bp.end(), bp.begin(), ::toupper);
    if (bp.empty()) bp = "GBRG";
    
    int g0_row, g0_col, g1_row, g1_col;
    
    if (bp == "RGGB" || bp == "BGGR") {
        g0_row = 0; g0_col = 1;
        g1_row = 1; g1_col = 0;
    } else {
        // GBRG, GRBG
        g0_row = 0; g0_col = 0;
        g1_row = 1; g1_col = 1;
    }
    
    for (int y = g0_row; y < height; y += 2) {
        for (int x = g0_col; x < width; x += 2) {
            mask(y, x) = 1.0f;
        }
    }
    for (int y = g1_row; y < height; y += 2) {
        for (int x = g1_col; x < width; x += 2) {
            mask(y, x) = 1.0f;
        }
    }
    
    return mask;
}

/// @brief Implements cfa green proxy.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Matrix2Df cfa_green_proxy(const Matrix2Df& mosaic, const std::string& bayer_pattern) {
    int h = mosaic.rows();
    int w = mosaic.cols();
    
    Matrix2Df gm = cfa_green_mask(h, w, bayer_pattern);
    Matrix2Df out(h, w);
    
    // Pad input
    Matrix2Df x_pad = Matrix2Df::Zero(h + 2, w + 2);
    Matrix2Df gm_pad = Matrix2Df::Zero(h + 2, w + 2);
    
    x_pad.block(1, 1, h, w) = mosaic;
    gm_pad.block(1, 1, h, w) = gm;
    
    // Edge padding
    x_pad.row(0) = x_pad.row(1);
    x_pad.row(h + 1) = x_pad.row(h);
    x_pad.col(0) = x_pad.col(1);
    x_pad.col(w + 1) = x_pad.col(w);
    
    gm_pad.row(0) = gm_pad.row(1);
    gm_pad.row(h + 1) = gm_pad.row(h);
    gm_pad.col(0) = gm_pad.col(1);
    gm_pad.col(w + 1) = gm_pad.col(w);
    
    Matrix2Df g_pad = x_pad.array() * gm_pad.array();
    
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (gm(y, x) > 0.5f) {
                out(y, x) = mosaic(y, x);
            } else {
                // Average of 4 neighbors
                float sum4 = g_pad(y, x + 1) + g_pad(y + 2, x + 1) + 
                             g_pad(y + 1, x) + g_pad(y + 1, x + 2);
                float cnt4 = gm_pad(y, x + 1) + gm_pad(y + 2, x + 1) + 
                             gm_pad(y + 1, x) + gm_pad(y + 1, x + 2);
                out(y, x) = (cnt4 > 0.5f) ? (sum4 / cnt4) : 0.0f;
            }
        }
    }
    
    return out;
}

/// @brief Implements cfa green proxy downsample2x2.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Matrix2Df cfa_green_proxy_downsample2x2(const Matrix2Df& mosaic, const std::string& bayer_pattern) {
    Matrix2Df p = cfa_green_proxy(mosaic, bayer_pattern);
    
    int h = p.rows();
    int w = p.cols();
    int h2 = h - (h % 2);
    int w2 = w - (w % 2);
    
    int out_h = h2 / 2;
    int out_w = w2 / 2;
    
    Matrix2Df out(out_h, out_w);
    
    for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
            int sy = y * 2;
            int sx = x * 2;
            float a = p(sy, sx);
            float b = p(sy, sx + 1);
            float c = p(sy + 1, sx);
            float d = p(sy + 1, sx + 1);
            out(y, x) = 0.25f * (a + b + c + d);
        }
    }
    
    return out;
}

/// @brief Implements warp cfa mosaic via subplanes.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Matrix2Df warp_cfa_mosaic_via_subplanes(
    const Matrix2Df& mosaic,
    const WarpMatrix& warp,
    int out_height,
    int out_width,
    const std::string& border_mode,
    const std::string& interpolation
) {
    const int h = mosaic.rows();
    const int w = mosaic.cols();
    const auto dims = tile_compile::core::compute_cfa_warp_dims(h, w, out_height, out_width);
    auto sub = tile_compile::core::extract_cfa_subplanes(mosaic, dims);
    const auto warps = tile_compile::core::make_all_cfa_subplane_warps(warp);

    cv::Mat a_cv(dims.sub_h, dims.sub_w, CV_32F, sub.a.data());
    cv::Mat b_cv(dims.sub_h, dims.sub_w, CV_32F, sub.b.data());
    cv::Mat c_cv(dims.sub_h, dims.sub_w, CV_32F, sub.c.data());
    cv::Mat d_cv(dims.sub_h, dims.sub_w, CV_32F, sub.d.data());

    int interp_flag = cv::INTER_LINEAR;
    if (interpolation == "nearest") {
        interp_flag = cv::INTER_NEAREST;
    } else if (interpolation == "cubic") {
        interp_flag = cv::INTER_CUBIC;
    } else if (interpolation == "lanczos4") {
        interp_flag = cv::INTER_LANCZOS4;
    }
    int flags = interp_flag | cv::WARP_INVERSE_MAP;

    int border_flag = cv::BORDER_CONSTANT;
    if (border_mode == "replicate") {
        border_flag = cv::BORDER_REPLICATE;
    } else if (border_mode == "reflect") {
        border_flag = cv::BORDER_REFLECT_101;
    }

    cv::Mat a_w, b_w, c_w, d_w;
    cv::warpAffine(a_cv, a_w, warps.a, cv::Size(dims.out_w_sub, dims.out_h_sub), flags, border_flag);
    cv::warpAffine(b_cv, b_w, warps.b, cv::Size(dims.out_w_sub, dims.out_h_sub), flags, border_flag);
    cv::warpAffine(c_cv, c_w, warps.c, cv::Size(dims.out_w_sub, dims.out_h_sub), flags, border_flag);
    cv::warpAffine(d_cv, d_w, warps.d, cv::Size(dims.out_w_sub, dims.out_h_sub), flags, border_flag);

    return tile_compile::core::reassemble_cfa_subplanes(a_w, b_w, c_w, d_w, dims);
}

/// @brief Splits cfa channels.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
CFAChannels split_cfa_channels(const Matrix2Df& mosaic, const std::string& bayer_pattern) {
    std::string bp = bayer_pattern;
    std::transform(bp.begin(), bp.end(), bp.begin(), ::toupper);
    if (bp.empty()) bp = "GBRG";
    
    int h = mosaic.rows();
    int w = mosaic.cols();
    int h2 = h - (h % 2);
    int w2 = w - (w % 2);
    int sub_h = h2 / 2;
    int sub_w = w2 / 2;
    
    // Bayer pattern positions
    int r_row, r_col, b_row, b_col, g1_row, g1_col, g2_row, g2_col;
    
    if (bp == "RGGB") {
        r_row = 0; r_col = 0; g1_row = 0; g1_col = 1; g2_row = 1; g2_col = 0; b_row = 1; b_col = 1;
    } else if (bp == "BGGR") {
        b_row = 0; b_col = 0; g1_row = 0; g1_col = 1; g2_row = 1; g2_col = 0; r_row = 1; r_col = 1;
    } else if (bp == "GBRG") {
        g1_row = 0; g1_col = 0; b_row = 0; b_col = 1; r_row = 1; r_col = 0; g2_row = 1; g2_col = 1;
    } else { // GRBG
        g1_row = 0; g1_col = 0; r_row = 0; r_col = 1; b_row = 1; b_col = 0; g2_row = 1; g2_col = 1;
    }
    
    CFAChannels channels;
    channels.R = Matrix2Df(sub_h, sub_w);
    channels.G = Matrix2Df(sub_h, sub_w);
    channels.B = Matrix2Df(sub_h, sub_w);
    
    for (int y = 0; y < sub_h; ++y) {
        for (int x = 0; x < sub_w; ++x) {
            channels.R(y, x) = mosaic(y * 2 + r_row, x * 2 + r_col);
            channels.B(y, x) = mosaic(y * 2 + b_row, x * 2 + b_col);
            float g1 = mosaic(y * 2 + g1_row, x * 2 + g1_col);
            float g2 = mosaic(y * 2 + g2_row, x * 2 + g2_col);
            channels.G(y, x) = 0.5f * (g1 + g2);
        }
    }
    
    return channels;
}

/// @brief Implements reassemble cfa mosaic.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Matrix2Df reassemble_cfa_mosaic(
    const Matrix2Df& r_plane,
    const Matrix2Df& g_plane, 
    const Matrix2Df& b_plane,
    const std::string& bayer_pattern
) {
    std::string bp = bayer_pattern;
    std::transform(bp.begin(), bp.end(), bp.begin(), ::toupper);
    if (bp.empty()) bp = "GBRG";
    
    int sub_h = r_plane.rows();
    int sub_w = r_plane.cols();
    int h = sub_h * 2;
    int w = sub_w * 2;
    
    int r_row, r_col, b_row, b_col, g1_row, g1_col, g2_row, g2_col;
    
    if (bp == "RGGB") {
        r_row = 0; r_col = 0; g1_row = 0; g1_col = 1; g2_row = 1; g2_col = 0; b_row = 1; b_col = 1;
    } else if (bp == "BGGR") {
        b_row = 0; b_col = 0; g1_row = 0; g1_col = 1; g2_row = 1; g2_col = 0; r_row = 1; r_col = 1;
    } else if (bp == "GBRG") {
        g1_row = 0; g1_col = 0; b_row = 0; b_col = 1; r_row = 1; r_col = 0; g2_row = 1; g2_col = 1;
    } else { // GRBG
        g1_row = 0; g1_col = 0; r_row = 0; r_col = 1; b_row = 1; b_col = 0; g2_row = 1; g2_col = 1;
    }
    
    Matrix2Df mosaic = Matrix2Df::Zero(h, w);
    
    for (int y = 0; y < sub_h; ++y) {
        for (int x = 0; x < sub_w; ++x) {
            mosaic(y * 2 + r_row, x * 2 + r_col) = r_plane(y, x);
            mosaic(y * 2 + b_row, x * 2 + b_col) = b_plane(y, x);
            mosaic(y * 2 + g1_row, x * 2 + g1_col) = g_plane(y, x);
            mosaic(y * 2 + g2_row, x * 2 + g2_col) = g_plane(y, x);
        }
    }
    
    return mosaic;
}

/// @brief Implements bayer offsets.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void bayer_offsets(const std::string& bayer_pattern,
                   int& r_row, int& r_col, int& b_row, int& b_col) {
    std::string bp = bayer_pattern;
    std::transform(bp.begin(), bp.end(), bp.begin(), ::toupper);
    r_row = 1; r_col = 0;
    b_row = 0; b_col = 1;
    if (bp == "RGGB") {
        r_row = 0; r_col = 0;
        b_row = 1; b_col = 1;
    } else if (bp == "BGGR") {
        r_row = 1; r_col = 1;
        b_row = 0; b_col = 0;
    } else if (bp == "GRBG") {
        r_row = 0; r_col = 1;
        b_row = 1; b_col = 0;
    }
}

/// @brief Implements debayer bilinear into.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void debayer_bilinear_into(const Matrix2Df& mosaic,
                           BayerPattern pattern,
                           Matrix2Df& R_out,
                           Matrix2Df& G_out,
                           Matrix2Df& B_out) {
    debayer_bilinear_into(mosaic, pattern, 0, 0, R_out, G_out, B_out);
}

/// @brief Implements debayer bilinear.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
DebayerResult debayer_bilinear(const Matrix2Df& mosaic,
                               BayerPattern pattern) {
    DebayerResult out;
    debayer_bilinear_into(mosaic, pattern, out.R, out.G, out.B);
    return out;
}

/// @brief Implements debayer bilinear into.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void debayer_bilinear_into(const Matrix2Df& mosaic,
                           BayerPattern pattern,
                           int origin_x,
                           int origin_y,
                           Matrix2Df& R_out,
                           Matrix2Df& G_out,
                           Matrix2Df& B_out) {
    const int h = static_cast<int>(mosaic.rows());
    const int w = static_cast<int>(mosaic.cols());
    debayer_bilinear_core(mosaic.data(), h, w, w, pattern, origin_x, origin_y,
                          R_out, G_out, B_out);
}

/// @brief Implements debayer bilinear strided into.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void debayer_bilinear_strided_into(const float* mosaic_data,
                                   int mosaic_rows,
                                   int mosaic_cols,
                                   int mosaic_stride,
                                   BayerPattern pattern,
                                   int origin_x,
                                   int origin_y,
                                   Matrix2Df& R_out,
                                   Matrix2Df& G_out,
                                   Matrix2Df& B_out) {
    if (mosaic_data == nullptr || mosaic_rows <= 0 || mosaic_cols <= 0 ||
        mosaic_stride < mosaic_cols) {
        R_out.resize(0, 0);
        G_out.resize(0, 0);
        B_out.resize(0, 0);
        return;
    }
    debayer_bilinear_core(mosaic_data, mosaic_rows, mosaic_cols, mosaic_stride,
                          pattern, origin_x, origin_y, R_out, G_out, B_out);
}

/// @brief Implements debayer bilinear.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
DebayerResult debayer_bilinear(const Matrix2Df& mosaic,
                               BayerPattern pattern,
                               int origin_x,
                               int origin_y) {
    DebayerResult out;
    debayer_bilinear_into(mosaic, pattern, origin_x, origin_y,
                          out.R, out.G, out.B);
    return out;
}

/// @brief Implements debayer nearest neighbor into.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void debayer_nearest_neighbor_into(const Matrix2Df& mosaic,
                                   BayerPattern pattern,
                                   Matrix2Df& R_out,
                                   Matrix2Df& G_out,
                                   Matrix2Df& B_out) {
    debayer_nearest_neighbor_into(mosaic, pattern, 0, 0, R_out, G_out, B_out);
}

/// @brief Implements debayer nearest neighbor.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
DebayerResult debayer_nearest_neighbor(const Matrix2Df& mosaic,
                                       BayerPattern pattern) {
    DebayerResult out;
    debayer_nearest_neighbor_into(mosaic, pattern, out.R, out.G, out.B);
    return out;
}

/// @brief Implements debayer nearest neighbor into.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void debayer_nearest_neighbor_into(const Matrix2Df& mosaic,
                                   BayerPattern pattern,
                                   int origin_x,
                                   int origin_y,
                                   Matrix2Df& R_out,
                                   Matrix2Df& G_out,
                                   Matrix2Df& B_out) {
    const int h = static_cast<int>(mosaic.rows());
    const int w = static_cast<int>(mosaic.cols());
    debayer_nearest_neighbor_core(mosaic.data(), h, w, w, pattern, origin_x, origin_y,
                                  R_out, G_out, B_out);
}

/// @brief Implements debayer nearest neighbor strided into.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void debayer_nearest_neighbor_strided_into(const float* mosaic_data,
                                           int mosaic_rows,
                                           int mosaic_cols,
                                           int mosaic_stride,
                                           BayerPattern pattern,
                                           int origin_x,
                                           int origin_y,
                                           Matrix2Df& R_out,
                                           Matrix2Df& G_out,
                                           Matrix2Df& B_out) {
    if (mosaic_data == nullptr || mosaic_rows <= 0 || mosaic_cols <= 0 ||
        mosaic_stride < mosaic_cols) {
        R_out.resize(0, 0);
        G_out.resize(0, 0);
        B_out.resize(0, 0);
        return;
    }
    debayer_nearest_neighbor_core(mosaic_data, mosaic_rows, mosaic_cols,
                                  mosaic_stride, pattern, origin_x, origin_y,
                                  R_out, G_out, B_out);
}

/// @brief Implements debayer nearest neighbor.
/// @details Part of CFA/Bayer mask, green-proxy, demosaic, and channel split helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
DebayerResult debayer_nearest_neighbor(const Matrix2Df& mosaic,
                                       BayerPattern pattern,
                                       int origin_x,
                                       int origin_y) {
    DebayerResult out;
    debayer_nearest_neighbor_into(mosaic, pattern, origin_x, origin_y,
                                  out.R, out.G, out.B);
    return out;
}

/// @brief Implements opencv bayer code.
/// @details Maps the Bayer pattern at the given subregion origin to the
/// matching OpenCV demosaicing code so CFA parity is preserved for mosaics
/// living on an offset canvas lattice.
static int opencv_bayer_code(BayerPattern pattern, int origin_x, int origin_y,
                             bool ahd) {
    if (pattern == BayerPattern::UNKNOWN) {
        pattern = BayerPattern::GBRG;
    }
    uint8_t lut[4] = {0, 0, 0, 0};
    fill_bayer_color_lut(pattern, lut);
    const int c00 = lut[((origin_y & 1) << 1) | (origin_x & 1)];
    const int c01 = lut[((origin_y & 1) << 1) | ((origin_x + 1) & 1)];
    // Empirically verified OpenCV code mapping (differs from the naive
    // first-row reading): standard RGGB -> BayerBG, BGGR -> BayerRG,
    // GRBG -> BayerGB, GBRG -> BayerGR.
    if (c00 == static_cast<int>(CfaColor::Blue)) {
        return ahd ? cv::COLOR_BayerRG2RGB_EA : cv::COLOR_BayerRG2RGB_VNG;
    }
    if (c00 == static_cast<int>(CfaColor::Green) &&
        c01 == static_cast<int>(CfaColor::Blue)) {
        return ahd ? cv::COLOR_BayerGR2RGB_EA : cv::COLOR_BayerGR2RGB_VNG;
    }
    if (c00 == static_cast<int>(CfaColor::Red)) {
        return ahd ? cv::COLOR_BayerBG2RGB_EA : cv::COLOR_BayerBG2RGB_VNG;
    }
    return ahd ? cv::COLOR_BayerGB2RGB_EA : cv::COLOR_BayerGB2RGB_VNG;
}

void debayer_opencv_into(const Matrix2Df& mosaic,
                         BayerPattern pattern,
                         int origin_x,
                         int origin_y,
                         bool ahd,
                         Matrix2Df& R_out,
                         Matrix2Df& G_out,
                         Matrix2Df& B_out) {
    const int h = static_cast<int>(mosaic.rows());
    const int w = static_cast<int>(mosaic.cols());
    if (h <= 0 || w <= 0) {
        R_out.resize(0, 0);
        G_out.resize(0, 0);
        B_out.resize(0, 0);
        return;
    }
    // cv::demosaicing supports only integer input: 16U for EA/AHD, 8U for
    // VNG. Map the float mosaic linearly to the full integer range and back;
    // the affine transform preserves channel ratios.
    cv::Mat src(h, w, CV_32F, const_cast<float*>(mosaic.data()),
                static_cast<size_t>(w) * sizeof(float));
    double v_min = 0.0, v_max = 0.0;
    cv::minMaxLoc(src, &v_min, &v_max);
    const double range = v_max - v_min;
    if (!(range > 0.0) || !std::isfinite(range)) {
        R_out = Matrix2Df::Constant(h, w, static_cast<float>(v_min));
        G_out = Matrix2Df::Constant(h, w, static_cast<float>(v_min));
        B_out = Matrix2Df::Constant(h, w, static_cast<float>(v_min));
        return;
    }
    cv::Mat rgb32;
    if (ahd) {
        cv::Mat src16, rgb16;
        src.convertTo(src16, CV_16U, 65535.0 / range, -v_min * 65535.0 / range);
        cv::demosaicing(src16, rgb16,
                        opencv_bayer_code(pattern, origin_x, origin_y, true));
        rgb16.convertTo(rgb32, CV_32F, range / 65535.0, v_min);
    } else {
        cv::Mat src8, rgb8;
        src.convertTo(src8, CV_8U, 255.0 / range, -v_min * 255.0 / range);
        cv::demosaicing(src8, rgb8,
                        opencv_bayer_code(pattern, origin_x, origin_y, false));
        rgb8.convertTo(rgb32, CV_32F, range / 255.0, v_min);
    }
    R_out.resize(h, w);
    G_out.resize(h, w);
    B_out.resize(h, w);
    for (int y = 0; y < h; ++y) {
        const cv::Vec3f* row = rgb32.ptr<cv::Vec3f>(y);
        for (int x = 0; x < w; ++x) {
            R_out(y, x) = row[x][0];
            G_out(y, x) = row[x][1];
            B_out(y, x) = row[x][2];
        }
    }
}

DebayerResult debayer_opencv(const Matrix2Df& mosaic,
                             BayerPattern pattern,
                             int origin_x,
                             int origin_y,
                             bool ahd) {
    DebayerResult out;
    debayer_opencv_into(mosaic, pattern, origin_x, origin_y, ahd,
                        out.R, out.G, out.B);
    return out;
}

} // namespace tile_compile::image
