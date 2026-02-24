#include "tile_compile/core/types.hpp"
#include "tile_compile/core/errors.hpp"

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>

namespace tile_compile::image {

std::map<std::string, Matrix2Df> split_cfa_channels(const Matrix2Df& mosaic, BayerPattern pattern) {
    int h = mosaic.rows();
    int w = mosaic.cols();
    int hh = h / 2;
    int hw = w / 2;
    
    Matrix2Df R(hh, hw), G(hh, hw), B(hh, hw);
    
    int r_row, r_col, b_row, b_col;
    switch (pattern) {
        case BayerPattern::RGGB:
            r_row = 0; r_col = 0; b_row = 1; b_col = 1;
            break;
        case BayerPattern::BGGR:
            r_row = 1; r_col = 1; b_row = 0; b_col = 0;
            break;
        case BayerPattern::GRBG:
            r_row = 0; r_col = 1; b_row = 1; b_col = 0;
            break;
        case BayerPattern::GBRG:
            r_row = 1; r_col = 0; b_row = 0; b_col = 1;
            break;
        default:
            throw TileCompileError("Unknown Bayer pattern");
    }
    
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

Matrix2Df reassemble_cfa_mosaic(const Matrix2Df& R, const Matrix2Df& G, const Matrix2Df& B, 
                                 BayerPattern pattern) {
    int hh = R.rows();
    int hw = R.cols();
    int h = hh * 2;
    int w = hw * 2;
    
    Matrix2Df mosaic(h, w);
    
    int r_row, r_col, b_row, b_col;
    switch (pattern) {
        case BayerPattern::RGGB:
            r_row = 0; r_col = 0; b_row = 1; b_col = 1;
            break;
        case BayerPattern::BGGR:
            r_row = 1; r_col = 1; b_row = 0; b_col = 0;
            break;
        case BayerPattern::GRBG:
            r_row = 0; r_col = 1; b_row = 1; b_col = 0;
            break;
        case BayerPattern::GBRG:
            r_row = 1; r_col = 0; b_row = 0; b_col = 1;
            break;
        default:
            throw TileCompileError("Unknown Bayer pattern");
    }
    
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

Matrix2Df normalize_frame(const Matrix2Df& frame, float target_background, 
                          float target_scale, NormalizationMode mode) {
    float median = 0.0f;
    {
        std::vector<float> sorted(frame.data(), frame.data() + frame.size());
        std::sort(sorted.begin(), sorted.end());
        size_t n = sorted.size();
        median = (n % 2 == 0) ? (sorted[n/2-1] + sorted[n/2]) / 2.0f : sorted[n/2];
    }
    
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
                vals.push_back(mosaic(y, x));
            }
        }
        Stats s;
        if (vals.empty()) {
            stats[py][px] = s;
            return;
        }
        std::sort(vals.begin(), vals.end());
        const size_t n = vals.size();
        s.median = (n % 2 == 0) ? (vals[n / 2 - 1] + vals[n / 2]) / 2.0f : vals[n / 2];

        std::vector<float> deviations(n);
        for (size_t i = 0; i < n; ++i) {
            deviations[i] = std::abs(vals[i] - s.median);
        }
        std::sort(deviations.begin(), deviations.end());
        s.mad = (n % 2 == 0)
                    ? (deviations[n / 2 - 1] + deviations[n / 2]) / 2.0f
                    : deviations[n / 2];
        s.sigma = 1.4826f * s.mad;
        s.threshold = s.median + sigma_threshold * s.sigma;
        s.neighbor_threshold = s.median + (0.5f * sigma_threshold) * s.sigma;
        s.ok = true;
        stats[py][px] = s;
    };

    compute_stats(0, 0);
    compute_stats(0, 1);
    compute_stats(1, 0);
    compute_stats(1, 1);

    if (!correct_hot) {
        return result;
    }

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
            if (v <= s.threshold) continue;

            int hot_neighbor_count = 0;
            for (int dy : {-2, 0, 2}) {
                for (int dx : {-2, 0, 2}) {
                    if (dy == 0 && dx == 0) continue;
                    const int yy = y + dy;
                    const int xx = x + dx;
                    if (!in_bounds(yy, xx)) continue;
                    if (mosaic(yy, xx) > s.neighbor_threshold) {
                        ++hot_neighbor_count;
                    }
                }
            }

            if (hot_neighbor_count <= 1) {
                float sum = 0.0f;
                int n = 0;
                const int yy4[4] = {y - 2, y + 2, y, y};
                const int xx4[4] = {x, x, x - 2, x + 2};
                for (int i = 0; i < 4; ++i) {
                    if (in_bounds(yy4[i], xx4[i])) {
                        sum += mosaic(yy4[i], xx4[i]);
                        ++n;
                    }
                }
                if (n >= 2) {
                    result(y, x) = sum / static_cast<float>(n);
                } else {
                    float sum8 = 0.0f;
                    int n8 = 0;
                    for (int dy : {-2, 0, 2}) {
                        for (int dx : {-2, 0, 2}) {
                            if (dy == 0 && dx == 0) continue;
                            const int yy = y + dy;
                            const int xx = x + dx;
                            if (!in_bounds(yy, xx)) continue;
                            sum8 += mosaic(yy, xx);
                            ++n8;
                        }
                    }
                    if (n8 > 0) {
                        result(y, x) = sum8 / static_cast<float>(n8);
                    }
                }
            }
        }
    }

    return result;
}

Matrix2Df cosmetic_correction(const Matrix2Df& frame, float sigma_threshold, bool correct_hot) {
    if (frame.size() == 0) return frame;
    Matrix2Df result = frame;
    int h = frame.rows();
    int w = frame.cols();
    
    float median = 0.0f;
    float mad = 0.0f;
    {
        std::vector<float> sorted(frame.data(), frame.data() + frame.size());
        std::sort(sorted.begin(), sorted.end());
        size_t n = sorted.size();
        median = (n % 2 == 0) ? (sorted[n/2-1] + sorted[n/2]) / 2.0f : sorted[n/2];
        
        std::vector<float> deviations(n);
        for (size_t i = 0; i < n; ++i) {
            deviations[i] = std::abs(sorted[i] - median);
        }
        std::sort(deviations.begin(), deviations.end());
        mad = (n % 2 == 0) ? (deviations[n/2-1] + deviations[n/2]) / 2.0f : deviations[n/2];
    }
    
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
