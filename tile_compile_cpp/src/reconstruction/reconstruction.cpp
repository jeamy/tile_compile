#include "tile_compile/reconstruction/reconstruction.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace tile_compile::reconstruction {

Matrix2Df reconstruct_tiles(const std::vector<Matrix2Df>& frames,
                            const TileGrid& grid,
                            const std::vector<std::vector<float>>& tile_weights) {
    if (frames.empty()) return Matrix2Df();
    
    int h = frames[0].rows();
    int w = frames[0].cols();
    Matrix2Df result = Matrix2Df::Zero(h, w);
    Matrix2Df weight_sum = Matrix2Df::Zero(h, w);

    auto hann_1d = [](int n) -> std::vector<float> {
        std::vector<float> out;
        if (n <= 0) return out;
        out.resize(static_cast<size_t>(n), 1.0f);
        if (n == 1) {
            out[0] = 1.0f;
            return out;
        }
        const float pi = 3.14159265358979323846f;
        for (int i = 0; i < n; ++i) {
            float x = static_cast<float>(i) / static_cast<float>(n - 1);
            out[static_cast<size_t>(i)] = 0.5f * (1.0f - std::cos(2.0f * pi * x));
        }
        return out;
    };
    
    for (size_t t = 0; t < grid.tiles.size(); ++t) {
        const Tile& tile = grid.tiles[t];

        const std::vector<float> wx = hann_1d(tile.width);
        const std::vector<float> wy = hann_1d(tile.height);
        
        for (size_t f = 0; f < frames.size(); ++f) {
            float weight = tile_weights[f][t];
            
            for (int y = tile.y; y < tile.y + tile.height && y < h; ++y) {
                for (int x = tile.x; x < tile.x + tile.width && x < w; ++x) {
                    int ly = y - tile.y;
                    int lx = x - tile.x;
                    if (ly < 0 || lx < 0 || ly >= static_cast<int>(wy.size()) || lx >= static_cast<int>(wx.size())) continue;
                    float win = wy[static_cast<size_t>(ly)] * wx[static_cast<size_t>(lx)];
                    float ww = weight * win;
                    result(y, x) += frames[f](y, x) * ww;
                    weight_sum(y, x) += ww;
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

Matrix2Df wiener_tile_filter(const Matrix2Df& tile, float sigma, float snr_tile,
                             float q_struct_tile, bool is_star_tile,
                             const config::WienerDenoiseConfig& cfg) {
    if (!cfg.enabled) return tile;
    if (is_star_tile) return tile;
    if (!(sigma > 0.0f)) return tile;
    if (snr_tile >= cfg.snr_threshold) return tile;
    if (q_struct_tile <= cfg.q_min) return tile;

    const int h = static_cast<int>(tile.rows());
    const int w = static_cast<int>(tile.cols());
    if (h <= 0 || w <= 0) return tile;

    const int pad_h = std::max(1, h / 4);
    const int pad_w = std::max(1, w / 4);

    cv::Mat tile_cv(h, w, CV_32F, const_cast<float*>(tile.data()));
    cv::Mat padded;
    cv::copyMakeBorder(tile_cv, padded, pad_h, pad_h, pad_w, pad_w,
                       cv::BORDER_REFLECT_101);

    cv::Mat F;
    cv::dft(padded, F, cv::DFT_COMPLEX_OUTPUT);

    std::vector<cv::Mat> planes(2);
    cv::split(F, planes);
    cv::Mat power = planes[0].mul(planes[0]) + planes[1].mul(planes[1]);

    const float sigma_sq = sigma * sigma;
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

    cv::Mat cropped = filtered(cv::Rect(pad_w, pad_h, w, h));
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
    return out;
}

Matrix2Df sigma_clip_stack(const std::vector<Matrix2Df>& frames,
                           float sigma_low, float sigma_high,
                           int max_iters, float min_fraction) {
    if (frames.empty()) return Matrix2Df();
    const int rows = frames[0].rows();
    const int cols = frames[0].cols();
    Matrix2Df out(rows, cols);
    const int n = static_cast<int>(frames.size());
    const int min_keep = std::max(1, static_cast<int>(std::ceil(min_fraction * n)));

    std::vector<float> values;
    values.reserve(static_cast<size_t>(n));
    std::vector<uint8_t> keep(static_cast<size_t>(n), 1);

    for (int idx = 0; idx < out.size(); ++idx) {
        values.clear();
        for (int i = 0; i < n; ++i) {
            values.push_back(frames[static_cast<size_t>(i)].data()[idx]);
            keep[static_cast<size_t>(i)] = 1;
        }

        int kept = n;
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
            double sd = (var > 0.0) ? std::sqrt(var) : 0.0;
            if (!(sd > 0.0)) break;

            int new_kept = 0;
            const double lo = mean - static_cast<double>(sigma_low) * sd;
            const double hi = mean + static_cast<double>(sigma_high) * sd;
            for (int i = 0; i < n; ++i) {
                if (!keep[static_cast<size_t>(i)]) continue;
                float v = values[static_cast<size_t>(i)];
                if (v < lo || v > hi) {
                    keep[static_cast<size_t>(i)] = 0;
                } else {
                    new_kept++;
                }
            }

            if (new_kept < min_keep) break;
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
            for (int i = 0; i < n; ++i)
                sum += static_cast<double>(values[static_cast<size_t>(i)]);
            count = n;
        }
        out.data()[idx] = static_cast<float>(sum / static_cast<double>(count));
    }

    return out;
}

} // namespace tile_compile::reconstruction
