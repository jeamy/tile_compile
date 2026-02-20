#include "tile_compile/reconstruction/reconstruction.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace tile_compile::reconstruction {

namespace {

float median_inplace(std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<long>(mid), v.end());
    return v[mid];
}

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

Matrix2Df reconstruct_tiles(const std::vector<Matrix2Df>& frames,
                            const TileGrid& grid,
                            const std::vector<std::vector<float>>& tile_weights) {
    if (frames.empty()) return Matrix2Df();
    
    int h = frames[0].rows();
    int w = frames[0].cols();
    Matrix2Df result = Matrix2Df::Zero(h, w);
    Matrix2Df weight_sum = Matrix2Df::Zero(h, w);

    // Cache Hanning windows: most grids use uniform tile sizes, so avoid
    // recomputing identical windows for every tile.
    int cached_w = -1, cached_h = -1;
    std::vector<float> wx, wy;

    for (size_t t = 0; t < grid.tiles.size(); ++t) {
        const Tile& tile = grid.tiles[t];

        if (tile.width != cached_w) {
            wx = make_hann_1d(tile.width);
            cached_w = tile.width;
        }
        if (tile.height != cached_h) {
            wy = make_hann_1d(tile.height);
            cached_h = tile.height;
        }
        
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

Matrix2Df soft_threshold_tile_filter(const Matrix2Df& tile,
                                      const config::SoftThresholdConfig& cfg) {
    if (!cfg.enabled) return tile;
    const int h = tile.rows();
    const int w = tile.cols();
    if (h <= 0 || w <= 0) return tile;

    // 1. Background estimation via box blur
    cv::Mat tile_cv(h, w, CV_32F, const_cast<float*>(tile.data()));

    // Build mask: pixels with data (> 0) are valid frame area;
    // zero pixels are canvas dead area and must not influence statistics.
    cv::Mat valid_mask;
    cv::compare(tile_cv, 0.0f, valid_mask, cv::CMP_GT);

    cv::Mat bg;
    int k = cfg.blur_kernel | 1; // ensure odd
    cv::blur(tile_cv, bg, cv::Size(k, k), cv::Point(-1, -1),
             cv::BORDER_REFLECT_101);

    // 2. Highpass residual: R = T - B
    cv::Mat resid = tile_cv - bg;

    // 3. Robust noise estimate: σ = 1.4826 · median(|R - median(R)|)
    //    Only use valid (non-zero) pixels to avoid canvas dead area bias.
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
    cv::Mat abs_resid = cv::abs(resid);
    cv::Mat shrunk;
    cv::subtract(abs_resid, tau, shrunk);
    cv::threshold(shrunk, shrunk, 0.0, 0.0, cv::THRESH_TOZERO);

    // Apply sign: where resid < 0, negate the shrunk value
    cv::Mat sign_mat;
    cv::threshold(resid, sign_mat, 0.0, 0.0, cv::THRESH_TOZERO);     // positive part
    cv::Mat neg_part;
    cv::threshold(-resid, neg_part, 0.0, 0.0, cv::THRESH_TOZERO);    // negative part (abs)
    // sign_mat > 0 → +1, neg_part > 0 → -1, both 0 → 0
    cv::Mat result_resid = shrunk.clone();
    // Where resid was negative, negate shrunk
    cv::Mat neg_mask;
    cv::compare(resid, 0.0f, neg_mask, cv::CMP_LT);
    cv::Mat neg_shrunk;
    cv::subtract(cv::Scalar(0.0f), shrunk, neg_shrunk);
    neg_shrunk.copyTo(result_resid, neg_mask);

    // 5. Reconstruct: T' = B + R'
    //    Preserve zero pixels (canvas dead area) — do not filter them.
    cv::Mat out_cv = bg + result_resid;
    cv::Mat zero_mask;
    cv::compare(tile_cv, 0.0f, zero_mask, cv::CMP_LE);
    out_cv.setTo(0.0f, zero_mask);

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

    // Linear Y + opponent chroma representation.
    cv::Mat Y = 0.25f * R + 0.5f * G + 0.25f * B;
    cv::Mat Cb = B - Y;
    cv::Mat Cr = R - Y;

    // Dataset-aware adaptation: scale denoise strength from measured chroma noise.
    // This keeps fine detail on clean data and increases suppression on noisy data.
    config::ChromaDenoiseConfig tuned = cfg;
    const float sigma_cb = robust_sigma_mad_from_mat(Cb);
    const float sigma_cr = robust_sigma_mad_from_mat(Cr);
    const float chroma_sigma = 0.5f * (sigma_cb + sigma_cr);
    const float ref_sigma = 0.02f;
    const float adapt = std::clamp(chroma_sigma / ref_sigma, 0.8f, 1.4f);
    tuned.blend.amount = std::clamp(cfg.blend.amount * adapt, 0.0f, 1.0f);
    tuned.chroma_wavelet.threshold_scale =
        std::max(0.1f, cfg.chroma_wavelet.threshold_scale * adapt);
    tuned.chroma_bilateral.sigma_range =
        std::max(1.0e-4f, cfg.chroma_bilateral.sigma_range * std::sqrt(adapt));

    cv::Mat Cb_orig = Cb.clone();
    cv::Mat Cr_orig = Cr.clone();

    denoise_chroma_plane_inplace(Cb, tuned);
    denoise_chroma_plane_inplace(Cr, tuned);

    cv::Mat amount_map(Y.size(), CV_32F, cv::Scalar(tuned.blend.amount));
    if (tuned.protect_luma) {
        cv::Mat protect = build_protection_mask(Y, tuned);
        amount_map = amount_map.mul(1.0f - tuned.luma_guard_strength * protect);
        cv::min(amount_map, tuned.blend.amount, amount_map);
        cv::max(amount_map, 0.0, amount_map);
    }

    cv::Mat one_minus = 1.0f - amount_map;
    cv::Mat Cb_mix = Cb_orig.mul(one_minus) + Cb.mul(amount_map);
    cv::Mat Cr_mix = Cr_orig.mul(one_minus) + Cr.mul(amount_map);

    cv::Mat R_new = Y + Cr_mix;
    cv::Mat B_new = Y + Cb_mix;
    cv::Mat G_new = 2.0f * Y - 0.5f * (R_new + B_new);

    R_new.copyTo(R);
    G_new.copyTo(G);
    B_new.copyTo(B);
}

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
    Matrix2Df out(rows, cols);
    const int n = static_cast<int>(valid.size());
    const int min_keep = std::max(1, static_cast<int>(std::ceil(min_fraction * n)));

    std::vector<float> values;
    values.reserve(static_cast<size_t>(n));
    std::vector<uint8_t> keep(static_cast<size_t>(n), 1);

    for (int idx = 0; idx < out.size(); ++idx) {
        // Collect only frames that have valid data (> 0) at this pixel.
        // Zero pixels are canvas dead area and must not enter statistics.
        values.clear();
        std::fill(keep.begin(), keep.end(), static_cast<uint8_t>(0));
        int n_valid_here = 0;
        for (int i = 0; i < n; ++i) {
            float v = valid[static_cast<size_t>(i)].get().data()[idx];
            if (v > 0.0f) {
                values.push_back(v);
                keep[static_cast<size_t>(i)] = 1;
                n_valid_here++;
            } else {
                values.push_back(0.0f);
            }
        }

        if (n_valid_here <= 0) {
            out.data()[idx] = 0.0f; // dead area
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
            for (int i = 0; i < n; ++i) {
                if (!keep[static_cast<size_t>(i)]) continue;
                float v = values[static_cast<size_t>(i)];
                if (v < lo || v > hi) {
                    keep[static_cast<size_t>(i)] = 0;
                } else {
                    new_kept++;
                }
            }

            if (new_kept < min_keep_here) break;
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
            // Fallback: average all valid-data frames at this pixel
            for (int i = 0; i < n; ++i) {
                float v = valid[static_cast<size_t>(i)].get().data()[idx];
                if (v > 0.0f) { sum += static_cast<double>(v); count++; }
            }
        }
        out.data()[idx] = (count > 0) ? static_cast<float>(sum / static_cast<double>(count)) : 0.0f;
    }

    return out;
}

Matrix2Df sigma_clip_weighted_tile(const std::vector<Matrix2Df>& tiles,
                                   const std::vector<float>& weights,
                                   float sigma_low, float sigma_high,
                                   int max_iters, float min_fraction) {
    if (tiles.empty()) return Matrix2Df();
    const int rows = tiles[0].rows();
    const int cols = tiles[0].cols();
    Matrix2Df out(rows, cols);
    const int n = static_cast<int>(tiles.size());
    // Fast path for tiny stacks where clipping is not meaningful and only adds
    // overhead. Keep weighted averaging semantics intact.
    if (n <= 2 || max_iters <= 0) {
        for (int idx = 0; idx < out.size(); ++idx) {
            double wsum = 0.0;
            double wmean = 0.0;
            for (int i = 0; i < n; ++i) {
                const float v = tiles[static_cast<size_t>(i)].data()[idx];
                if (!(std::isfinite(v) && v > 0.0f)) continue;
                const double wi = static_cast<double>(weights[static_cast<size_t>(i)]);
                if (!(wi > 0.0)) continue;
                wsum += wi;
                wmean += wi * static_cast<double>(v);
            }
            out.data()[idx] = (wsum > 0.0) ? static_cast<float>(wmean / wsum) : 0.0f;
        }
        return out;
    }

    std::vector<float> values(static_cast<size_t>(n));
    std::vector<uint8_t> keep(static_cast<size_t>(n));

    for (int idx = 0; idx < out.size(); ++idx) {
        int n_valid_here = 0;
        for (int i = 0; i < n; ++i) {
            const float v = tiles[static_cast<size_t>(i)].data()[idx];
            const float w = weights[static_cast<size_t>(i)];
            values[static_cast<size_t>(i)] = v;
            const bool valid_here = std::isfinite(v) && v > 0.0f &&
                                    std::isfinite(w) && w > 0.0f;
            keep[static_cast<size_t>(i)] = valid_here ? 1 : 0;
            if (valid_here) n_valid_here++;
        }

        if (n_valid_here <= 0) {
            out.data()[idx] = 0.0f;
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
                double wi = static_cast<double>(weights[static_cast<size_t>(i)]);
                wsum += wi;
                wmean += wi * static_cast<double>(values[static_cast<size_t>(i)]);
            }
            if (!(wsum > 0.0)) break;
            wmean /= wsum;

            double var = 0.0;
            double wsum2 = 0.0; // sum of squared weights for Bessel correction
            for (int i = 0; i < n; ++i) {
                if (!keep[static_cast<size_t>(i)]) continue;
                double wi = static_cast<double>(weights[static_cast<size_t>(i)]);
                double d = static_cast<double>(values[static_cast<size_t>(i)]) - wmean;
                var += wi * d * d;
                wsum2 += wi * wi;
            }
            // Bessel correction for reliability (non-frequency) weights:
            // var_unbiased = (Σ wi·d²) / (V1 - V2/V1)  where V1=wsum, V2=Σwi²
            double denom = wsum - wsum2 / wsum;
            double sd = (var > 0.0 && denom > 0.0) ? std::sqrt(var / denom) : 0.0;
            if (!(sd > 0.0)) break;

            const double lo = wmean - static_cast<double>(sigma_low) * sd;
            const double hi = wmean + static_cast<double>(sigma_high) * sd;
            int new_kept = 0;
            for (int i = 0; i < n; ++i) {
                if (!keep[static_cast<size_t>(i)]) continue;
                double v = static_cast<double>(values[static_cast<size_t>(i)]);
                if (v < lo || v > hi) {
                    keep[static_cast<size_t>(i)] = 0;
                } else {
                    new_kept++;
                }
            }
            if (new_kept < min_keep_here) break;
            if (new_kept == kept) break; // converged
            kept = new_kept;
        }

        // Final weighted mean of kept values
        double wsum = 0.0, wmean = 0.0;
        for (int i = 0; i < n; ++i) {
            if (!keep[static_cast<size_t>(i)]) continue;
            double wi = static_cast<double>(weights[static_cast<size_t>(i)]);
            wsum += wi;
            wmean += wi * static_cast<double>(values[static_cast<size_t>(i)]);
        }
        if (wsum > 0.0) {
            out.data()[idx] = static_cast<float>(wmean / wsum);
        } else {
            // Fallback: use all valid (non-zero) values for this pixel.
            wsum = 0.0; wmean = 0.0;
            for (int i = 0; i < n; ++i) {
                const float v = values[static_cast<size_t>(i)];
                if (!(std::isfinite(v) && v > 0.0f)) continue;
                double wi = static_cast<double>(weights[static_cast<size_t>(i)]);
                if (!(wi > 0.0)) continue;
                wsum += wi;
                wmean += wi * static_cast<double>(v);
            }
            out.data()[idx] = (wsum > 0.0) ? static_cast<float>(wmean / wsum) : 0.0f;
        }
    }

    return out;
}

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

std::vector<float> make_hann_1d(int n) {
    std::vector<float> w;
    if (n <= 0)
        return w;
    w.resize(static_cast<size_t>(n));
    if (n == 1) {
        w[0] = 1.0f;
        return w;
    }
    const float pi = 3.14159265358979323846f;
    for (int i = 0; i < n; ++i) {
        float x = static_cast<float>(i) / static_cast<float>(n - 1);
        w[static_cast<size_t>(i)] = 0.5f * (1.0f - std::cos(2.0f * pi * x));
    }
    return w;
}

} // namespace tile_compile::reconstruction
