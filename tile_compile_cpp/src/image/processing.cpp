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

Matrix2Df cosmetic_correction(const Matrix2Df& frame, float sigma_threshold, bool correct_hot) {
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
    
    if (correct_hot) {
        for (int y = 1; y < h - 1; ++y) {
            for (int x = 1; x < w - 1; ++x) {
                if (frame(y, x) > threshold) {
                    float neighbors = (frame(y-1, x) + frame(y+1, x) + 
                                       frame(y, x-1) + frame(y, x+1)) / 4.0f;
                    result(y, x) = neighbors;
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
