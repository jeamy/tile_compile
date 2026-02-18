#include "tile_compile/image/cfa_processing.hpp"

#include <opencv2/opencv.hpp>
#include <algorithm>

namespace tile_compile::image {

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

Matrix2Df warp_cfa_mosaic_via_subplanes(
    const Matrix2Df& mosaic,
    const WarpMatrix& warp,
    int out_height,
    int out_width,
    const std::string& border_mode,
    const std::string& interpolation
) {
    int h = mosaic.rows();
    int w = mosaic.cols();
    int h2 = h - (h % 2);
    int w2 = w - (w % 2);
    
    int out_h = (out_height > 0) ? out_height : h;
    int out_w = (out_width > 0) ? out_width : w;
    int out_h2 = out_h - (out_h % 2);
    int out_w2 = out_w - (out_w % 2);
    
    // Extract 4 Bayer subplanes
    int sub_h = h2 / 2;
    int sub_w = w2 / 2;
    
    Matrix2Df a(sub_h, sub_w), b(sub_h, sub_w), c(sub_h, sub_w), d(sub_h, sub_w);
    
    for (int y = 0; y < sub_h; ++y) {
        for (int x = 0; x < sub_w; ++x) {
            a(y, x) = mosaic(y * 2, x * 2);
            b(y, x) = mosaic(y * 2, x * 2 + 1);
            c(y, x) = mosaic(y * 2 + 1, x * 2);
            d(y, x) = mosaic(y * 2 + 1, x * 2 + 1);
        }
    }
    
    // Convert to OpenCV
    cv::Mat a_cv(sub_h, sub_w, CV_32F, a.data());
    cv::Mat b_cv(sub_h, sub_w, CV_32F, b.data());
    cv::Mat c_cv(sub_h, sub_w, CV_32F, c.data());
    cv::Mat d_cv(sub_h, sub_w, CV_32F, d.data());
    
    // Warp matrix components — rotation/scale stay the same,
    // but translation must be halved because subplanes are half-resolution.
    float a2_00 = warp(0, 0), a2_01 = warp(0, 1);
    float a2_10 = warp(1, 0), a2_11 = warp(1, 1);
    float t_x = warp(0, 2) * 0.5f;
    float t_y = warp(1, 2) * 0.5f;
    
    // Compute delta shifts for each subplane
    // delta_a = [-0.25, -0.25], delta_b = [0.25, -0.25], etc.
    auto make_warp = [&](float dx, float dy) -> cv::Mat {
        float new_tx = t_x + (a2_00 * dx + a2_01 * dy) - dx;
        float new_ty = t_y + (a2_10 * dx + a2_11 * dy) - dy;
        cv::Mat w = (cv::Mat_<float>(2, 3) << a2_00, a2_01, new_tx, a2_10, a2_11, new_ty);
        return w;
    };
    
    cv::Mat warp_a = make_warp(-0.25f, -0.25f);
    cv::Mat warp_b = make_warp(0.25f, -0.25f);
    cv::Mat warp_c = make_warp(-0.25f, 0.25f);
    cv::Mat warp_d = make_warp(0.25f, 0.25f);
    
    // Interpolation and border flags
    int interp_flag = (interpolation == "nearest") ? cv::INTER_NEAREST : cv::INTER_LINEAR;
    int flags = interp_flag | cv::WARP_INVERSE_MAP;
    
    int border_flag = cv::BORDER_CONSTANT;
    if (border_mode == "replicate") {
        border_flag = cv::BORDER_REPLICATE;
    } else if (border_mode == "reflect") {
        border_flag = cv::BORDER_REFLECT_101;
    }
    
    int out_w_sub = std::max(1, out_w2 / 2);
    int out_h_sub = std::max(1, out_h2 / 2);
    
    cv::Mat a_w, b_w, c_w, d_w;
    cv::warpAffine(a_cv, a_w, warp_a, cv::Size(out_w_sub, out_h_sub), flags, border_flag);
    cv::warpAffine(b_cv, b_w, warp_b, cv::Size(out_w_sub, out_h_sub), flags, border_flag);
    cv::warpAffine(c_cv, c_w, warp_c, cv::Size(out_w_sub, out_h_sub), flags, border_flag);
    cv::warpAffine(d_cv, d_w, warp_d, cv::Size(out_w_sub, out_h_sub), flags, border_flag);
    
    // Reassemble into exactly out_height x out_width to match DiskCacheFrameStore expectations.
    // out_h2/out_w2 may be smaller than out_h/out_w if out_h/out_w are odd —
    // but canvas is always rounded to even, so this is a safety guard only.
    Matrix2Df out = Matrix2Df::Zero(out_h, out_w);
    
    for (int y = 0; y < out_h_sub; ++y) {
        for (int x = 0; x < out_w_sub; ++x) {
            out(y * 2, x * 2) = a_w.at<float>(y, x);
            out(y * 2, x * 2 + 1) = b_w.at<float>(y, x);
            out(y * 2 + 1, x * 2) = c_w.at<float>(y, x);
            out(y * 2 + 1, x * 2 + 1) = d_w.at<float>(y, x);
        }
    }
    
    return out;
}

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

DebayerResult debayer_bilinear(const Matrix2Df& mosaic,
                               BayerPattern pattern) {
    return debayer_bilinear(mosaic, pattern, 0, 0);
}

DebayerResult debayer_bilinear(const Matrix2Df& mosaic,
                               BayerPattern pattern,
                               int origin_x,
                               int origin_y) {
    const int h = static_cast<int>(mosaic.rows());
    const int w = static_cast<int>(mosaic.cols());

    DebayerResult out;
    out.R = Matrix2Df::Zero(h, w);
    out.G = Matrix2Df::Zero(h, w);
    out.B = Matrix2Df::Zero(h, w);

    // Default to GBRG if unknown.
    if (pattern == BayerPattern::UNKNOWN) {
        pattern = BayerPattern::GBRG;
    }

    int r_row = 1, r_col = 0;
    int b_row = 0, b_col = 1;
    int g1_row = 0, g1_col = 0;
    int g2_row = 1, g2_col = 1;

    if (pattern == BayerPattern::RGGB) {
        r_row = 0; r_col = 0;
        b_row = 1; b_col = 1;
        g1_row = 0; g1_col = 1;
        g2_row = 1; g2_col = 0;
    } else if (pattern == BayerPattern::BGGR) {
        r_row = 1; r_col = 1;
        b_row = 0; b_col = 0;
        g1_row = 0; g1_col = 1;
        g2_row = 1; g2_col = 0;
    } else if (pattern == BayerPattern::GRBG) {
        r_row = 0; r_col = 1;
        b_row = 1; b_col = 0;
        g1_row = 0; g1_col = 0;
        g2_row = 1; g2_col = 1;
    } else { // GBRG
        r_row = 1; r_col = 0;
        b_row = 0; b_col = 1;
        g1_row = 0; g1_col = 0;
        g2_row = 1; g2_col = 1;
    }

    auto clamp_y = [&](int yy) -> int {
        return std::max(0, std::min(h - 1, yy));
    };
    auto clamp_x = [&](int xx) -> int {
        return std::max(0, std::min(w - 1, xx));
    };
    auto parity_match = [&](int yy, int xx, int prow, int pcol) -> bool {
        const int py = (origin_y + yy) & 1;
        const int px = (origin_x + xx) & 1;
        return (py == prow && px == pcol);
    };
    auto is_red = [&](int yy, int xx) -> bool {
        return parity_match(yy, xx, r_row, r_col);
    };
    auto is_blue = [&](int yy, int xx) -> bool {
        return parity_match(yy, xx, b_row, b_col);
    };
    auto is_green = [&](int yy, int xx) -> bool {
        return parity_match(yy, xx, g1_row, g1_col) || parity_match(yy, xx, g2_row, g2_col);
    };
    auto sample = [&](int yy, int xx) -> float {
        return mosaic(clamp_y(yy), clamp_x(xx));
    };
    auto avg_if = [&](const int* ys, const int* xs, int n,
                      const auto& pred) -> float {
        float sum = 0.0f;
        int cnt = 0;
        for (int i = 0; i < n; ++i) {
            const int yy = clamp_y(ys[i]);
            const int xx = clamp_x(xs[i]);
            if (!pred(yy, xx)) continue;
            sum += sample(yy, xx);
            ++cnt;
        }
        return (cnt > 0)
                   ? (sum / static_cast<float>(cnt))
                   : sample(clamp_y(ys[0]), clamp_x(xs[0]));
    };

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float r_val = 0.0f;
            float g_val = 0.0f;
            float b_val = 0.0f;

            if (is_red(y, x)) {
                r_val = sample(y, x);
                const int gy[4] = {y - 1, y + 1, y, y};
                const int gx[4] = {x, x, x - 1, x + 1};
                g_val = avg_if(gy, gx, 4, is_green);
                const int by[4] = {y - 1, y - 1, y + 1, y + 1};
                const int bx[4] = {x - 1, x + 1, x - 1, x + 1};
                b_val = avg_if(by, bx, 4, is_blue);
            } else if (is_blue(y, x)) {
                b_val = sample(y, x);
                const int gy[4] = {y - 1, y + 1, y, y};
                const int gx[4] = {x, x, x - 1, x + 1};
                g_val = avg_if(gy, gx, 4, is_green);
                const int ry[4] = {y - 1, y - 1, y + 1, y + 1};
                const int rx[4] = {x - 1, x + 1, x - 1, x + 1};
                r_val = avg_if(ry, rx, 4, is_red);
            } else if (is_green(y, x)) {
                g_val = sample(y, x);
                const bool green_on_red_row = (((origin_y + y) & 1) == r_row);
                if (green_on_red_row) {
                    const int ry[2] = {y, y};
                    const int rx[2] = {x - 1, x + 1};
                    r_val = avg_if(ry, rx, 2, is_red);
                    const int by[2] = {y - 1, y + 1};
                    const int bx[2] = {x, x};
                    b_val = avg_if(by, bx, 2, is_blue);
                } else {
                    const int ry[2] = {y - 1, y + 1};
                    const int rx[2] = {x, x};
                    r_val = avg_if(ry, rx, 2, is_red);
                    const int by[2] = {y, y};
                    const int bx[2] = {x - 1, x + 1};
                    b_val = avg_if(by, bx, 2, is_blue);
                }
            } else {
                // Should be unreachable with valid Bayer pattern.
                r_val = sample(y, x);
                g_val = sample(y, x);
                b_val = sample(y, x);
            }

            out.R(y, x) = r_val;
            out.G(y, x) = g_val;
            out.B(y, x) = b_val;
        }
    }

    return out;
}

DebayerResult debayer_nearest_neighbor(const Matrix2Df& mosaic,
                                       BayerPattern pattern) {
    return debayer_bilinear(mosaic, pattern, 0, 0);
}

DebayerResult debayer_nearest_neighbor(const Matrix2Df& mosaic,
                                       BayerPattern pattern,
                                       int origin_x,
                                       int origin_y) {
    return debayer_bilinear(mosaic, pattern, origin_x, origin_y);
}

} // namespace tile_compile::image
