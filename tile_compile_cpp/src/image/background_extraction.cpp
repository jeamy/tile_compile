#include "tile_compile/image/background_extraction.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace tile_compile::image {

namespace {

constexpr float kTiny = 1.0e-12f;
constexpr float kMinUsableTileFraction = 0.10f;
constexpr float kMinValidSampleFractionForApply = 0.30f;
constexpr int kMinValidSamplesForApply = 96;

float clamp01(float v) { return std::max(0.0f, std::min(1.0f, v)); }

float robust_median(std::vector<float> values) {
    if (values.empty()) return 0.0f;
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    float med = values[mid];
    if ((values.size() & 1U) == 0U) {
        std::nth_element(values.begin(), values.begin() + (mid - 1), values.end());
        med = 0.5f * (med + values[mid - 1]);
    }
    return med;
}

float robust_quantile(std::vector<float> values, float q) {
    if (values.empty()) return 0.0f;
    q = clamp01(q);
    const size_t idx = static_cast<size_t>(q * static_cast<float>(values.size() - 1));
    std::nth_element(values.begin(), values.begin() + idx, values.end());
    return values[idx];
}

float robust_mad(const std::vector<float>& values, float center) {
    if (values.empty()) return 0.0f;
    std::vector<float> abs_dev;
    abs_dev.reserve(values.size());
    for (float v : values) abs_dev.push_back(std::abs(v - center));
    return robust_median(std::move(abs_dev));
}

std::vector<float> box_blur(const Matrix2Df& tile, int radius) {
    const int h = static_cast<int>(tile.rows());
    const int w = static_cast<int>(tile.cols());
    std::vector<float> out(static_cast<size_t>(h * w), 0.0f);
    if (h <= 0 || w <= 0) return out;
    radius = std::max(0, radius);

    std::vector<double> integral(static_cast<size_t>((h + 1) * (w + 1)), 0.0);
    auto idx = [w](int y, int x) { return static_cast<size_t>(y * (w + 1) + x); };
    for (int y = 0; y < h; ++y) {
        double row_sum = 0.0;
        for (int x = 0; x < w; ++x) {
            float v = tile(y, x);
            if (!std::isfinite(v)) v = 0.0f;
            row_sum += static_cast<double>(v);
            integral[idx(y + 1, x + 1)] = integral[idx(y, x + 1)] + row_sum;
        }
    }

    for (int y = 0; y < h; ++y) {
        const int y0 = std::max(0, y - radius);
        const int y1 = std::min(h - 1, y + radius);
        for (int x = 0; x < w; ++x) {
            const int x0 = std::max(0, x - radius);
            const int x1 = std::min(w - 1, x + radius);
            const double sum = integral[idx(y1 + 1, x1 + 1)]
                             - integral[idx(y0, x1 + 1)]
                             - integral[idx(y1 + 1, x0)]
                             + integral[idx(y0, x0)];
            const int area = (y1 - y0 + 1) * (x1 - x0 + 1);
            out[static_cast<size_t>(y * w + x)] = static_cast<float>(sum / std::max(1, area));
        }
    }
    return out;
}

std::vector<uint8_t> dilate_mask(const std::vector<uint8_t>& mask, int w, int h, int radius) {
    if (radius <= 0) return mask;
    std::vector<uint8_t> out(mask.size(), 0);
    const int r2 = radius * radius;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint8_t hit = 0;
            for (int dy = -radius; dy <= radius && !hit; ++dy) {
                const int yy = y + dy;
                if (yy < 0 || yy >= h) continue;
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int xx = x + dx;
                    if (xx < 0 || xx >= w) continue;
                    if (dx * dx + dy * dy > r2) continue;
                    if (mask[static_cast<size_t>(yy * w + xx)] != 0) {
                        hit = 1;
                        break;
                    }
                }
            }
            out[static_cast<size_t>(y * w + x)] = hit;
        }
    }
    return out;
}

float robust_weight_from_loss(const std::string& loss, float r, float param) {
    if (loss == "tukey") return tukey_weight(r, std::max(param, kTiny));
    return huber_weight(r, std::max(param, kTiny));
}

BGEValueStats stats_from_values(const std::vector<float>& values) {
    BGEValueStats st;
    if (values.empty()) return st;

    std::vector<float> finite;
    finite.reserve(values.size());
    for (float v : values) {
        if (std::isfinite(v)) finite.push_back(v);
    }
    if (finite.empty()) return st;

    st.n = static_cast<int>(finite.size());
    st.min = *std::min_element(finite.begin(), finite.end());
    st.max = *std::max_element(finite.begin(), finite.end());
    st.median = robust_median(finite);

    double sum = 0.0;
    for (float v : finite) sum += static_cast<double>(v);
    st.mean = static_cast<float>(sum / static_cast<double>(finite.size()));

    double sum_sq = 0.0;
    for (float v : finite) {
        const double d = static_cast<double>(v) - static_cast<double>(st.mean);
        sum_sq += d * d;
    }
    st.std = static_cast<float>(std::sqrt(sum_sq / static_cast<double>(finite.size())));
    return st;
}

BGEValueStats stats_from_matrix(const Matrix2Df& m) {
    std::vector<float> vals;
    vals.reserve(static_cast<size_t>(m.size()));
    for (int i = 0; i < m.size(); ++i) {
        vals.push_back(m.data()[i]);
    }
    return stats_from_values(vals);
}

float spatial_background_spread(const Matrix2Df& img) {
    const int H = static_cast<int>(img.rows());
    const int W = static_cast<int>(img.cols());
    if (H <= 0 || W <= 0) return std::numeric_limits<float>::infinity();

    std::vector<float> valid_values;
    valid_values.reserve(static_cast<size_t>(img.size()));
    for (int i = 0; i < img.size(); ++i) {
        const float v = img.data()[i];
        if (std::isfinite(v) && v > 0.0f) valid_values.push_back(v);
    }
    if (valid_values.size() < 4096) return std::numeric_limits<float>::infinity();

    const float value_thresh = robust_quantile(valid_values, 0.65f);

    std::vector<float> grad_map(static_cast<size_t>(H * W), 0.0f);
    std::vector<float> grad_values;
    grad_values.reserve(valid_values.size());
    for (int y = 0; y < H; ++y) {
        const int ym = std::max(0, y - 1);
        const int yp = std::min(H - 1, y + 1);
        for (int x = 0; x < W; ++x) {
            const float v = img(y, x);
            if (!(std::isfinite(v) && v > 0.0f)) continue;

            const int xm = std::max(0, x - 1);
            const int xp = std::min(W - 1, x + 1);

            float vxm = img(y, xm);
            float vxp = img(y, xp);
            float vym = img(ym, x);
            float vyp = img(yp, x);

            if (!std::isfinite(vxm)) vxm = v;
            if (!std::isfinite(vxp)) vxp = v;
            if (!std::isfinite(vym)) vym = v;
            if (!std::isfinite(vyp)) vyp = v;

            const float g = std::abs(vxp - vxm) + std::abs(vyp - vym);
            grad_map[static_cast<size_t>(y * W + x)] = g;
            grad_values.push_back(g);
        }
    }
    if (grad_values.size() < 4096) return std::numeric_limits<float>::infinity();

    const float grad_thresh = robust_quantile(grad_values, 0.70f);
    constexpr int kBlockSize = 128;
    constexpr int kMinPixelsPerBlock = 256;

    std::vector<float> block_medians;
    std::vector<float> block_values;
    for (int y0 = 0; y0 < H; y0 += kBlockSize) {
        for (int x0 = 0; x0 < W; x0 += kBlockSize) {
            const int y1 = std::min(H, y0 + kBlockSize);
            const int x1 = std::min(W, x0 + kBlockSize);
            block_values.clear();
            block_values.reserve(static_cast<size_t>((y1 - y0) * (x1 - x0)));

            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                    const float v = img(y, x);
                    if (!(std::isfinite(v) && v > 0.0f)) continue;
                    if (v > value_thresh) continue;
                    const float g = grad_map[static_cast<size_t>(y * W + x)];
                    if (!(std::isfinite(g) && g <= grad_thresh)) continue;
                    block_values.push_back(v);
                }
            }

            if (static_cast<int>(block_values.size()) >= kMinPixelsPerBlock) {
                block_medians.push_back(robust_median(block_values));
            }
        }
    }

    if (block_medians.size() < 8) return std::numeric_limits<float>::infinity();

    const float p10 = robust_quantile(block_medians, 0.10f);
    const float p90 = robust_quantile(block_medians, 0.90f);
    return p90 - p10;
}

} // namespace

// RBF kernel functions (v3.3 §6.3.7)
float rbf_kernel_multiquadric(float d, float mu) {
    return std::sqrt(d * d + mu * mu);
}

float rbf_kernel_thinplate(float d, float epsilon) {
    float d_safe = d + epsilon;
    return (d_safe > epsilon) ? (d_safe * d_safe * std::log(d_safe)) : 0.0f;
}

float rbf_kernel_gaussian(float d, float mu) {
    return std::exp(-d * d / (2.0f * mu * mu));
}

// Robust loss functions (v3.3 §6.3.7)
float huber_loss(float r, float delta) {
    float abs_r = std::abs(r);
    return (abs_r <= delta) ? (0.5f * r * r) : (delta * (abs_r - 0.5f * delta));
}

float tukey_loss(float r, float c) {
    float abs_r = std::abs(r);
    if (abs_r <= c) {
        float u = r / c;
        return (c * c / 6.0f) * (1.0f - std::pow(1.0f - u * u, 3.0f));
    }
    return c * c / 6.0f;
}

float huber_weight(float r, float delta) {
    float abs_r = std::abs(r);
    return (abs_r <= delta) ? 1.0f : (delta / abs_r);
}

float tukey_weight(float r, float c) {
    float abs_r = std::abs(r);
    if (abs_r <= c) {
        float u = r / c;
        return std::pow(1.0f - u * u, 2.0f);
    }
    return 0.0f;
}

// Compute adaptive grid spacing (v3.3 §6.3.8)
int compute_grid_spacing(int image_width, int image_height, int tile_size, const BGEConfig& config) {
    int min_dim = std::min(image_width, image_height);
    
    // Spec intent: G must never drop below max(2*T, G_min).
    // If G_max_fraction would violate this, clamp G_max up to the required floor.
    int G_from_tiles = 2 * tile_size;
    int G_from_resolution = min_dim / std::max(1, config.grid.N_g);
    const int G_floor = std::max(G_from_tiles, config.grid.G_min_px);
    int G_max = static_cast<int>(min_dim * config.grid.G_max_fraction);
    G_max = std::max(G_max, G_floor);

    int G = std::max(G_floor, G_from_resolution);
    G = std::min(G, G_max);
    
    return G;
}

// Extract tile background samples (v3.3 §6.3.2)
std::vector<TileBGSample> extract_tile_background_samples(
    const Matrix2Df& channel,
    const std::vector<TileMetrics>& tile_metrics,
    const TileGrid& tile_grid,
    const BGEConfig& config) {
    
    std::vector<TileBGSample> samples;
    samples.reserve(tile_grid.tiles.size());
    
    if (tile_metrics.size() < tile_grid.tiles.size()) {
        std::cerr << "[BGE] Warning: tile_metrics smaller than tile_grid, truncating to min size" << std::endl;
    }

    const size_t n_tiles = std::min(tile_metrics.size(), tile_grid.tiles.size());

    int informative_metric_tiles = 0;
    for (size_t ti = 0; ti < n_tiles; ++ti) {
        const auto& tm = tile_metrics[ti];
        const bool has_structure =
            std::isfinite(tm.noise) && tm.noise > 1.0e-6f &&
            std::isfinite(tm.gradient_energy) && tm.gradient_energy > 1.0e-6f;
        const bool has_quality = std::isfinite(tm.quality_score) &&
                                 std::abs(tm.quality_score) > 1.0e-3f;
        if (has_structure || has_quality) ++informative_metric_tiles;
    }
    const float informative_fraction =
        (n_tiles > 0)
            ? (static_cast<float>(informative_metric_tiles) /
               static_cast<float>(n_tiles))
            : 0.0f;
    const bool use_tile_metrics = (informative_fraction >= 0.35f);
    if (!use_tile_metrics && n_tiles > 0) {
        std::cerr << "[BGE]   Tile metrics look degenerate (informative="
                  << informative_metric_tiles << "/" << n_tiles
                  << "), switching to image-only sample selection" << std::endl;
    }

    // Compute structure threshold (percentile over E/sigma)
    std::vector<float> structure_scores;
    structure_scores.reserve(n_tiles);
    if (use_tile_metrics) {
        for (size_t ti = 0; ti < n_tiles; ++ti) {
            const auto& tm = tile_metrics[ti];
            if (tm.noise > 1e-6f && std::isfinite(tm.gradient_energy)) {
                structure_scores.push_back(tm.gradient_energy / tm.noise);
            }
        }
    }
    
    float structure_thresh = 0.0f;
    if (!structure_scores.empty()) {
        structure_thresh = robust_quantile(std::move(structure_scores),
                                           config.structure_thresh_percentile);
    }
    
    // Extract background sample per tile
    for (size_t t = 0; t < n_tiles; ++t) {
        const auto& tile = tile_grid.tiles[t];
        const auto& tm = tile_metrics[t];
        
        TileBGSample sample;
        sample.x = tile.x + tile.width / 2.0f;
        sample.y = tile.y + tile.height / 2.0f;
        sample.valid = false;

        // Extremely star-dense STAR tiles are weak background candidates.
        // Keep moderately populated STAR tiles to avoid over-pruning in rich fields.
        if (use_tile_metrics &&
            tm.type == TileType::STAR && tm.star_count >= 16) {
            samples.push_back(sample);
            continue;
        }
        // High local-quality STRUCTURE tiles are often bright extended object
        // signal (nebula/galaxy detail), not sky background. Aggregated
        // quality medians cluster near 0, so use a conservative threshold.
        if (use_tile_metrics &&
            tm.type == TileType::STRUCTURE &&
            std::isfinite(tm.quality_score) && tm.quality_score >= 0.20f) {
            samples.push_back(sample);
            continue;
        }

        // Exclude high-structure tiles (v3.3 §6.3.2a)
        float tile_structure = (tm.noise > 1e-6f) ? (tm.gradient_energy / tm.noise) : 0.0f;
        if (use_tile_metrics && tile_structure > structure_thresh) {
            samples.push_back(sample);
            continue;
        }
        
        // Extract tile region
        int x0 = tile.x;
        int y0 = tile.y;
        int x1 = std::min(x0 + tile.width, static_cast<int>(channel.cols()));
        int y1 = std::min(y0 + tile.height, static_cast<int>(channel.rows()));
        
        if (x1 <= x0 || y1 <= y0) {
            samples.push_back(sample);
            continue;
        }
        
        // Collect tile pixels and local stats for deterministic masks.
        std::vector<float> tile_pixels;
        std::vector<float> tile_gradients;
        tile_pixels.reserve((x1 - x0) * (y1 - y0));
        tile_gradients.reserve((x1 - x0) * (y1 - y0));

        Matrix2Df tile_view = channel.block(y0, x0, y1 - y0, x1 - x0);
        const int tw = x1 - x0;
        const int th = y1 - y0;

        std::vector<float> finite_values;
        finite_values.reserve(static_cast<size_t>(tw * th));
        for (int yy = 0; yy < th; ++yy) {
            for (int xx = 0; xx < tw; ++xx) {
                float v = tile_view(yy, xx);
                if (std::isfinite(v)) finite_values.push_back(v);
            }
        }

        if (finite_values.empty()) {
            samples.push_back(sample);
            continue;
        }

        const float sat_level = robust_quantile(finite_values, 0.999f);

        // Approximate star mask from deterministic DoG fallback (spec §6.3.2a).
        const int r_small = 1;
        const int r_large = std::max(2, std::min(tw, th) / 12);
        std::vector<float> blur_small = box_blur(tile_view, r_small);
        std::vector<float> blur_large = box_blur(tile_view, r_large);
        std::vector<float> dog_vals;
        dog_vals.reserve(static_cast<size_t>(tw * th));
        for (size_t i = 0; i < blur_small.size(); ++i) {
            dog_vals.push_back(blur_small[i] - blur_large[i]);
        }
        const float dog_med = robust_median(dog_vals);
        const float dog_mad = robust_mad(dog_vals, dog_med);
        const float dog_thresh = dog_med + 3.0f * std::max(1.4826f * dog_mad, 1.0e-6f);
        const float bright_thresh = robust_quantile(finite_values, 0.80f);
        std::vector<uint8_t> star_mask(static_cast<size_t>(tw * th), 0);
        for (int yy = 0; yy < th; ++yy) {
            for (int xx = 0; xx < tw; ++xx) {
                const size_t i = static_cast<size_t>(yy * tw + xx);
                const float v = tile_view(yy, xx);
                if (std::isfinite(v) && v >= bright_thresh && dog_vals[i] > dog_thresh) {
                    star_mask[i] = 1;
                }
            }
        }
        int star_dilate_px = std::max(0, config.mask.star_dilate_px);
        if (std::isfinite(tm.fwhm) && tm.fwhm > 0.0f) {
            const int add = static_cast<int>(std::lround(0.25f * tm.fwhm));
            star_dilate_px = std::clamp(star_dilate_px + std::max(0, add),
                                        star_dilate_px, star_dilate_px + 8);
        }
        star_mask = dilate_mask(star_mask, tw, th, star_dilate_px);

        std::vector<uint8_t> sat_mask(static_cast<size_t>(tw * th), 0);
        for (int yy = 0; yy < th; ++yy) {
            for (int xx = 0; xx < tw; ++xx) {
                const size_t i = static_cast<size_t>(yy * tw + xx);
                float v = tile_view(yy, xx);
                if (std::isfinite(v) && v >= sat_level) sat_mask[i] = 1;
            }
        }
        sat_mask = dilate_mask(sat_mask, tw, th, std::max(0, config.mask.sat_dilate_px));

        // Per-pixel structure metric from local gradients.
        for (int yy = 0; yy < th; ++yy) {
            const int ym = std::max(0, yy - 1);
            const int yp = std::min(th - 1, yy + 1);
            for (int xx = 0; xx < tw; ++xx) {
                const int xm = std::max(0, xx - 1);
                const int xp = std::min(tw - 1, xx + 1);
                const float gx = std::abs(tile_view(yy, xp) - tile_view(yy, xm));
                const float gy = std::abs(tile_view(yp, xx) - tile_view(ym, xx));
                tile_gradients.push_back(gx + gy);
            }
        }
        const float grad_thresh = robust_quantile(tile_gradients, config.structure_thresh_percentile);

        for (int yy = 0; yy < th; ++yy) {
            for (int xx = 0; xx < tw; ++xx) {
                const size_t i = static_cast<size_t>(yy * tw + xx);
                float v = tile_view(yy, xx);
                const bool structure_bad = tile_gradients[i] > grad_thresh;
                const bool masked = (star_mask[i] != 0) || (sat_mask[i] != 0) || structure_bad;
                if (!masked && std::isfinite(v) && v > 0.0f) {
                    tile_pixels.push_back(v);
                }
            }
        }
        
        if (tile_pixels.empty()) {
            samples.push_back(sample);
            continue;
        }

        const float usable_fraction = static_cast<float>(tile_pixels.size()) /
                                      static_cast<float>((x1 - x0) * (y1 - y0));
        if (!(std::isfinite(usable_fraction)) ||
            usable_fraction < kMinUsableTileFraction) {
            samples.push_back(sample);
            continue;
        }
        
        // Compute quantile (v3.3 §6.3.2b)
        sample.bg_value = robust_quantile(std::move(tile_pixels), config.sample_quantile);
        if (!(std::isfinite(sample.bg_value) && sample.bg_value > 0.0f)) {
            samples.push_back(sample);
            continue;
        }
        
        // Compute reliability weight (v3.3 §6.3.2c)
        float masked_fraction = 1.0f - usable_fraction;
        const float q =
            (use_tile_metrics && std::isfinite(tm.quality_score))
                ? tm.quality_score
                : 0.0f;
        // Do not up-weight high local-quality tiles for BGE: they are often
        // object-rich and can imprint cloud-like structures into the model.
        const float quality_term =
            std::clamp(1.0f / (1.0f + 0.10f * std::abs(q)), 0.70f, 1.0f);
        const int star_count_for_weight =
            use_tile_metrics ? std::max(0, tm.star_count - 4) : 0;
        const float star_penalty =
            1.0f / (1.0f + 0.04f * static_cast<float>(star_count_for_weight));
        sample.weight = std::exp(-2.0f * tile_structure) *
                        (1.0f - masked_fraction) *
                        quality_term * star_penalty;
        sample.weight = std::max(0.01f, std::min(1.0f, sample.weight));
        
        sample.valid = true;
        samples.push_back(sample);
    }
    
    return samples;
}

// Aggregate tiles to coarse grid (v3.3 §6.3.3)
std::vector<GridCell> aggregate_to_coarse_grid(
    const std::vector<TileBGSample>& tile_samples,
    int image_width, int image_height,
    int grid_spacing,
    const BGEConfig& config) {
    
    // Compute grid dimensions
    int n_cells_x = (image_width + grid_spacing - 1) / grid_spacing;
    int n_cells_y = (image_height + grid_spacing - 1) / grid_spacing;
    
    // Initialize grid cells
    std::vector<std::vector<GridCell>> grid(n_cells_y, std::vector<GridCell>(n_cells_x));
    for (int cy = 0; cy < n_cells_y; ++cy) {
        for (int cx = 0; cx < n_cells_x; ++cx) {
            auto& cell = grid[cy][cx];
            cell.cell_x = cx;
            cell.cell_y = cy;
            cell.center_x = (cx + 0.5f) * grid_spacing;
            cell.center_y = (cy + 0.5f) * grid_spacing;
            cell.n_samples = 0;
            cell.valid = false;
        }
    }
    
    // Assign tile samples to grid cells (v3.3 §6.3.3b)
    std::vector<std::vector<TileBGSample>> cell_samples(n_cells_y * n_cells_x);

    // Robustly suppress globally implausible tile background samples before
    // cell aggregation (e.g., bright-object contamination, near-zero artifacts).
    std::vector<float> valid_bg_values;
    valid_bg_values.reserve(tile_samples.size());
    for (const auto& s : tile_samples) {
        if (s.valid && std::isfinite(s.bg_value) && s.bg_value > 0.0f) {
            valid_bg_values.push_back(s.bg_value);
        }
    }

    float bg_med = 0.0f;
    float bg_sigma = 0.0f;
    bool have_bg_guard = false;
    if (valid_bg_values.size() >= 16) {
        bg_med = robust_median(valid_bg_values);
        const float mad = robust_mad(valid_bg_values, bg_med);
        bg_sigma = 1.4826f * mad;
        have_bg_guard = std::isfinite(bg_sigma) && bg_sigma > kTiny;
    }

    int n_rejected_global_outliers = 0;
    for (const auto& sample : tile_samples) {
        if (!sample.valid) continue;
        if (!(std::isfinite(sample.bg_value) && sample.bg_value > 0.0f)) continue;

        if (have_bg_guard) {
            const float lo = bg_med - 5.0f * bg_sigma;
            const float hi = bg_med + 4.0f * bg_sigma;
            if (sample.bg_value < lo || sample.bg_value > hi) {
                ++n_rejected_global_outliers;
                continue;
            }
        }
        
        int cx = static_cast<int>(sample.x / grid_spacing);
        int cy = static_cast<int>(sample.y / grid_spacing);
        
        if (cx >= 0 && cx < n_cells_x && cy >= 0 && cy < n_cells_y) {
            int cell_idx = cy * n_cells_x + cx;
            cell_samples[cell_idx].push_back(sample);
        }
    }

    if (n_rejected_global_outliers > 0) {
        std::cerr << "[BGE]   Rejected global sample outliers: "
                  << n_rejected_global_outliers << std::endl;
    }
    
    // Aggregate per cell (v3.3 §6.3.3c)
    for (int cy = 0; cy < n_cells_y; ++cy) {
        for (int cx = 0; cx < n_cells_x; ++cx) {
            int cell_idx = cy * n_cells_x + cx;
            auto& cell = grid[cy][cx];
            const auto& samples = cell_samples[cell_idx];
            
            cell.n_samples = samples.size();
            
            // Check if sufficient samples (v3.3 §6.3.3d)
            if (cell.n_samples < config.min_tiles_per_cell) {
                cell.valid = false;
                continue;
            }
            
            // Compute median background and weight
            std::vector<float> bg_values, weights;
            bg_values.reserve(samples.size());
            weights.reserve(samples.size());
            
            for (const auto& s : samples) {
                bg_values.push_back(s.bg_value);
                weights.push_back(s.weight);
            }
            
            cell.bg_value = robust_median(std::move(bg_values));
            cell.weight = robust_median(std::move(weights));
            cell.valid = true;
        }
    }

    // Deterministic handling of insufficient cells (v3.3 §6.3.3d)
    auto is_valid = [&](int cx, int cy) {
        return cy >= 0 && cy < n_cells_y && cx >= 0 && cx < n_cells_x && grid[cy][cx].valid;
    };

    if (config.grid.insufficient_cell_strategy == "nearest") {
        for (int cy = 0; cy < n_cells_y; ++cy) {
            for (int cx = 0; cx < n_cells_x; ++cx) {
                if (grid[cy][cx].valid) continue;
                float best_d2 = std::numeric_limits<float>::infinity();
                int best_x = -1;
                int best_y = -1;
                for (int sy = 0; sy < n_cells_y; ++sy) {
                    for (int sx = 0; sx < n_cells_x; ++sx) {
                        if (!grid[sy][sx].valid) continue;
                        const float dx = grid[cy][cx].center_x - grid[sy][sx].center_x;
                        const float dy = grid[cy][cx].center_y - grid[sy][sx].center_y;
                        const float d2 = dx * dx + dy * dy;
                        if (d2 < best_d2 ||
                            (std::abs(d2 - best_d2) <= 1.0e-6f && (sy < best_y || (sy == best_y && sx < best_x)))) {
                            best_d2 = d2;
                            best_x = sx;
                            best_y = sy;
                        }
                    }
                }
                if (best_x >= 0 && best_y >= 0) {
                    grid[cy][cx].bg_value = grid[best_y][best_x].bg_value;
                    grid[cy][cx].weight = grid[best_y][best_x].weight;
                    grid[cy][cx].valid = true;
                    grid[cy][cx].n_samples = std::max(1, grid[best_y][best_x].n_samples);
                }
            }
        }
    } else if (config.grid.insufficient_cell_strategy == "radius_expand") {
        const int max_radius = std::max(n_cells_x, n_cells_y);
        for (int cy = 0; cy < n_cells_y; ++cy) {
            for (int cx = 0; cx < n_cells_x; ++cx) {
                if (grid[cy][cx].valid) continue;

                std::vector<float> bg_vals;
                std::vector<float> wt_vals;
                int total_samples = 0;
                for (int r = 1; r <= max_radius; ++r) {
                    for (int sy = std::max(0, cy - r); sy <= std::min(n_cells_y - 1, cy + r); ++sy) {
                        for (int sx = std::max(0, cx - r); sx <= std::min(n_cells_x - 1, cx + r); ++sx) {
                            if (std::max(std::abs(sy - cy), std::abs(sx - cx)) != r) continue;
                            const int sidx = sy * n_cells_x + sx;
                            for (const auto& s : cell_samples[sidx]) {
                                bg_vals.push_back(s.bg_value);
                                wt_vals.push_back(s.weight);
                                ++total_samples;
                            }
                        }
                    }
                    if (total_samples >= config.min_tiles_per_cell) {
                        grid[cy][cx].bg_value = robust_median(std::move(bg_vals));
                        grid[cy][cx].weight = robust_median(std::move(wt_vals));
                        grid[cy][cx].valid = true;
                        grid[cy][cx].n_samples = total_samples;
                        break;
                    }
                }
            }
        }
    }

    // Flatten to vector of valid cells
    std::vector<GridCell> valid_cells;
    for (int cy = 0; cy < n_cells_y; ++cy) {
        for (int cx = 0; cx < n_cells_x; ++cx) {
            if (is_valid(cx, cy)) {
                valid_cells.push_back(grid[cy][cx]);
            }
        }
    }

    return valid_cells;
}

// RBF interpolation (v3.3 §6.3.7)
Matrix2Df fit_rbf_surface(
    const std::vector<GridCell>& grid_cells,
    int image_width, int image_height,
    int grid_spacing,
    const BGEConfig& config) {
    
    const int M = grid_cells.size();
    if (M < 3) {
        std::cerr << "[BGE] Too few grid cells for RBF: " << M << std::endl;
        return Matrix2Df::Zero(image_height, image_width);
    }
    
    // Compute mu (shape parameter, v3.3 §6.3.7)
    const int G = std::max(1, grid_spacing);
    const float mu = std::max(kTiny, config.fit.rbf_mu_factor * static_cast<float>(G));
    const float epsilon = std::max(kTiny, config.fit.rbf_epsilon);
    const float lambda_base = std::max(kTiny, config.fit.rbf_lambda);
    
    // Build RBF matrix Phi (M x M)
    Eigen::MatrixXf Phi_base(M, M);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            float dx = grid_cells[i].center_x - grid_cells[j].center_x;
            float dy = grid_cells[i].center_y - grid_cells[j].center_y;
            float d = std::sqrt(dx * dx + dy * dy);
            
            if (config.fit.rbf_phi == "multiquadric") {
                Phi_base(i, j) = rbf_kernel_multiquadric(d, mu);
            } else if (config.fit.rbf_phi == "thinplate") {
                Phi_base(i, j) = rbf_kernel_thinplate(d, epsilon);
            } else { // gaussian
                Phi_base(i, j) = rbf_kernel_gaussian(d, mu);
            }
        }
    }

    // Build target vector b (M x 1)
    Eigen::VectorXf b(M);
    Eigen::VectorXf w_rel(M);
    std::vector<float> bg_values;
    bg_values.reserve(static_cast<size_t>(M));
    for (int i = 0; i < M; ++i) {
        b(i) = grid_cells[i].bg_value;
        w_rel(i) = std::max(1.0e-3f, grid_cells[i].weight);
        bg_values.push_back(grid_cells[i].bg_value);
    }

    auto solve_rbf_coeffs = [&](float lambda,
                                Eigen::VectorXf* out_u,
                                float* out_rms) -> bool {
        const Eigen::MatrixXf Phi_reg =
            Phi_base + lambda * Eigen::MatrixXf::Identity(M, M);

        Eigen::VectorXf u = Eigen::VectorXf::Zero(M);
        Eigen::VectorXf w_rob = Eigen::VectorXf::Ones(M);
        for (int iter = 0; iter < std::max(1, config.fit.irls_max_iterations); ++iter) {
            Eigen::VectorXf w = w_rel.cwiseProduct(w_rob).cwiseMax(1.0e-6f);
            Eigen::MatrixXf W = w.asDiagonal();
            Eigen::MatrixXf lhs = Phi_reg.transpose() * W * Phi_reg +
                                  lambda * Eigen::MatrixXf::Identity(M, M);
            Eigen::VectorXf rhs = Phi_reg.transpose() * W * b;
            Eigen::VectorXf u_new = lhs.ldlt().solve(rhs);

            if (!u_new.allFinite()) {
                return false;
            }

            const float step = (u_new - u).norm();
            const float scale = 1.0f + u.norm();
            u = u_new;

            Eigen::VectorXf residual = b - Phi_base * u;
            for (int i = 0; i < M; ++i) {
                w_rob(i) = robust_weight_from_loss(config.fit.robust_loss,
                                                   residual(i), config.fit.huber_delta);
            }

            if (step <= config.fit.irls_tolerance * scale) break;
        }

        Eigen::VectorXf residual = b - Phi_base * u;
        const float rms = std::sqrt(residual.squaredNorm() / static_cast<float>(M));
        if (!(std::isfinite(rms))) {
            return false;
        }

        *out_u = std::move(u);
        *out_rms = rms;
        return true;
    };

    // Dynamic lambda adaptation: test/adjust/test and prefer the smoothest
    // (highest lambda) model that still fits grid samples well enough.
    const float bg_med = robust_median(bg_values);
    const float bg_sigma = 1.4826f * robust_mad(bg_values, bg_med);
    const float residual_limit = std::max(0.15f, 0.20f * std::max(bg_sigma, kTiny));

    std::vector<float> lambda_trials;
    lambda_trials.reserve(6);
    float l = lambda_base;
    for (int i = 0; i < 6; ++i) {
        lambda_trials.push_back(std::clamp(l, 1.0e-8f, 1.0e-1f));
        l *= 3.0f;
    }
    std::sort(lambda_trials.begin(), lambda_trials.end());
    lambda_trials.erase(std::unique(lambda_trials.begin(), lambda_trials.end()),
                        lambda_trials.end());

    float best_lambda = lambda_base;
    float best_rms = std::numeric_limits<float>::infinity();
    Eigen::VectorXf best_u = Eigen::VectorXf::Zero(M);
    bool have_best = false;

    float accepted_lambda = -1.0f;
    float accepted_rms = std::numeric_limits<float>::infinity();
    Eigen::VectorXf accepted_u = Eigen::VectorXf::Zero(M);

    for (float lambda_try : lambda_trials) {
        Eigen::VectorXf u_try = Eigen::VectorXf::Zero(M);
        float rms_try = std::numeric_limits<float>::infinity();
        const bool ok = solve_rbf_coeffs(lambda_try, &u_try, &rms_try);
        if (!ok) {
            std::cerr << "[BGE]   RBF lambda=" << lambda_try << " fit failed" << std::endl;
            continue;
        }

        std::cerr << "[BGE]   RBF lambda=" << lambda_try
                  << " trial RMS=" << rms_try << std::endl;

        if (!have_best || rms_try < best_rms) {
            best_lambda = lambda_try;
            best_rms = rms_try;
            best_u = u_try;
            have_best = true;
        }

        if (rms_try <= residual_limit) {
            accepted_lambda = lambda_try;
            accepted_rms = rms_try;
            accepted_u = u_try;
        }
    }

    if (!have_best) {
        std::cerr << "[BGE] RBF solve failed for all lambda trials" << std::endl;
        return Matrix2Df::Zero(image_height, image_width);
    }

    float lambda = best_lambda;
    float rms_selected = best_rms;
    Eigen::VectorXf u = best_u;
    if (accepted_lambda > 0.0f) {
        lambda = accepted_lambda;
        rms_selected = accepted_rms;
        u = accepted_u;
    }

    std::cerr << "[BGE]   RBF selected lambda=" << lambda
              << " (limit=" << residual_limit
              << ", rms=" << rms_selected << ")" << std::endl;
    
    // Evaluate RBF at all image pixels
    Matrix2Df surface = Matrix2Df::Zero(image_height, image_width);
    
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < image_height; ++y) {
        for (int x = 0; x < image_width; ++x) {
            float sum = 0.0f;
            for (int i = 0; i < M; ++i) {
                float dx = x - grid_cells[i].center_x;
                float dy = y - grid_cells[i].center_y;
                float d = std::sqrt(dx * dx + dy * dy);
                
                float phi_val;
                if (config.fit.rbf_phi == "multiquadric") {
                    phi_val = rbf_kernel_multiquadric(d, mu);
                } else if (config.fit.rbf_phi == "thinplate") {
                    phi_val = rbf_kernel_thinplate(d, epsilon);
                } else {
                    phi_val = rbf_kernel_gaussian(d, mu);
                }
                
                sum += u(i) * phi_val;
            }
            surface(y, x) = sum;
        }
    }
    
    return surface;
}

// Polynomial surface fitting (v3.3 §6.3.7)
Matrix2Df fit_polynomial_surface(
    const std::vector<GridCell>& grid_cells,
    int image_width, int image_height,
    const BGEConfig& config) {
    
    const int M = grid_cells.size();
    const int order = config.fit.polynomial_order;
    
    // Number of polynomial terms: (order+1)*(order+2)/2
    int n_terms = 0;
    for (int m = 0; m <= order; ++m) {
        for (int n = 0; n <= order - m; ++n) {
            ++n_terms;
        }
    }
    
    if (M < n_terms) {
        std::cerr << "[BGE] Too few grid cells for polynomial order " << order 
                  << ": " << M << " < " << n_terms << std::endl;
        return Matrix2Df::Zero(image_height, image_width);
    }
    
    // Normalize coordinates to [-1, 1]
    float x_scale = 2.0f / image_width;
    float y_scale = 2.0f / image_height;
    float x_offset = -1.0f;
    float y_offset = -1.0f;
    
    // Build design matrix A (M x n_terms)
    Eigen::MatrixXf A(M, n_terms);
    Eigen::VectorXf b(M);
    Eigen::VectorXf w_rel(M);
    
    for (int i = 0; i < M; ++i) {
        float x_norm = grid_cells[i].center_x * x_scale + x_offset;
        float y_norm = grid_cells[i].center_y * y_scale + y_offset;
        
        int col = 0;
        for (int m = 0; m <= order; ++m) {
            for (int n = 0; n <= order - m; ++n) {
                A(i, col) = std::pow(x_norm, m) * std::pow(y_norm, n);
                ++col;
            }
        }
        
        b(i) = grid_cells[i].bg_value;
        w_rel(i) = std::max(1.0e-3f, grid_cells[i].weight);
    }

    // Robust IRLS polynomial fitting (v3.3 §6.3.7)
    Eigen::VectorXf coeffs = Eigen::VectorXf::Zero(n_terms);
    Eigen::VectorXf w_rob = Eigen::VectorXf::Ones(M);
    for (int iter = 0; iter < std::max(1, config.fit.irls_max_iterations); ++iter) {
        Eigen::VectorXf w = w_rel.cwiseProduct(w_rob).cwiseMax(1.0e-6f);
        Eigen::MatrixXf W = w.asDiagonal();
        Eigen::MatrixXf lhs = A.transpose() * W * A;
        Eigen::VectorXf rhs = A.transpose() * W * b;
        Eigen::VectorXf coeffs_new = lhs.ldlt().solve(rhs);

        const float step = (coeffs_new - coeffs).norm();
        const float scale = 1.0f + coeffs.norm();
        coeffs = coeffs_new;

        Eigen::VectorXf residual = b - A * coeffs;
        for (int i = 0; i < M; ++i) {
            w_rob(i) = robust_weight_from_loss(config.fit.robust_loss,
                                               residual(i), config.fit.huber_delta);
        }

        if (step <= config.fit.irls_tolerance * scale) {
            break;
        }
    }
    
    // Evaluate polynomial at all image pixels
    Matrix2Df surface = Matrix2Df::Zero(image_height, image_width);
    
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < image_height; ++y) {
        for (int x = 0; x < image_width; ++x) {
            float x_norm = x * x_scale + x_offset;
            float y_norm = y * y_scale + y_offset;
            
            float sum = 0.0f;
            int col = 0;
            for (int m = 0; m <= order; ++m) {
                for (int n = 0; n <= order - m; ++n) {
                    sum += coeffs(col) * std::pow(x_norm, m) * std::pow(y_norm, n);
                    ++col;
                }
            }
            surface(y, x) = sum;
        }
    }
    
    return surface;
}

// Fit background surface (v3.3 §6.3.7)
BackgroundModel fit_background_surface(
    const std::vector<GridCell>& grid_cells,
    int image_width, int image_height,
    int grid_spacing,
    const BGEConfig& config) {
    
    BackgroundModel result;
    result.grid_cells = grid_cells;
    result.n_valid_cells = grid_cells.size();
    result.success = false;
    
    if (grid_cells.size() < 3) {
        result.error_message = "Too few valid grid cells: " + std::to_string(grid_cells.size());
        return result;
    }
    
    try {
        auto rms_for_model = [&](const Matrix2Df& model) -> float {
            if (model.rows() != image_height || model.cols() != image_width ||
                !model.allFinite()) {
                return std::numeric_limits<float>::infinity();
            }
            float sum_sq = 0.0f;
            int n = 0;
            for (const auto& cell : grid_cells) {
                const int cx = static_cast<int>(cell.center_x);
                const int cy = static_cast<int>(cell.center_y);
                if (cx >= 0 && cx < image_width && cy >= 0 && cy < image_height) {
                    const float residual = cell.bg_value - model(cy, cx);
                    sum_sq += residual * residual;
                    ++n;
                }
            }
            if (n <= 0) return std::numeric_limits<float>::infinity();
            return std::sqrt(sum_sq / static_cast<float>(n));
        };

        const std::string method = config.fit.method;
        Matrix2Df selected_model = Matrix2Df::Zero(image_height, image_width);
        float selected_rms = std::numeric_limits<float>::infinity();

        if (method == "rbf") {
            selected_model = fit_rbf_surface(grid_cells, image_width, image_height,
                                             grid_spacing, config);
            selected_rms = rms_for_model(selected_model);

            // Safety fallback: if RBF fit is unstable on coarse cells, prefer
            // a smoother low-order polynomial model.
            constexpr float kRbfFallbackRmsThreshold = 0.25f;
            if (!std::isfinite(selected_rms) ||
                selected_rms > kRbfFallbackRmsThreshold) {
                BGEConfig poly_cfg = config;
                poly_cfg.fit.method = "poly";
                poly_cfg.fit.polynomial_order = std::clamp(poly_cfg.fit.polynomial_order, 2, 3);
                Matrix2Df poly_model =
                    fit_polynomial_surface(grid_cells, image_width, image_height, poly_cfg);
                const float poly_rms = rms_for_model(poly_model);
                if (std::isfinite(poly_rms) &&
                    poly_rms <= selected_rms * 1.05f) {
                    std::cerr << "[BGE]   RBF fallback -> poly(order="
                              << poly_cfg.fit.polynomial_order
                              << ") rms=" << poly_rms
                              << " (rbf rms=" << selected_rms << ")" << std::endl;
                    selected_model = std::move(poly_model);
                    selected_rms = poly_rms;
                }
            }
        } else if (method == "spline") {
            // Practical spline backend: thin-plate RBF with conservative smoothing.
            BGEConfig spline_cfg = config;
            spline_cfg.fit.method = "rbf";
            spline_cfg.fit.rbf_phi = "thinplate";
            spline_cfg.fit.rbf_lambda = std::max(1.0e-4f, spline_cfg.fit.rbf_lambda);
            selected_model = fit_rbf_surface(grid_cells, image_width, image_height,
                                             grid_spacing, spline_cfg);
            selected_rms = rms_for_model(selected_model);
        } else if (method == "poly") {
            selected_model =
                fit_polynomial_surface(grid_cells, image_width, image_height, config);
            selected_rms = rms_for_model(selected_model);
        } else if (method == "bicubic") {
            // Fallback approximation: cubic polynomial surface.
            BGEConfig cubic_cfg = config;
            cubic_cfg.fit.polynomial_order = std::max(3, cubic_cfg.fit.polynomial_order);
            selected_model =
                fit_polynomial_surface(grid_cells, image_width, image_height, cubic_cfg);
            selected_rms = rms_for_model(selected_model);
        } else {
            result.error_message = "Unsupported fit method: " + method;
            return result;
        }

        if (!std::isfinite(selected_rms)) {
            result.error_message = "Surface fit produced non-finite residuals";
            return result;
        }

        result.model = std::move(selected_model);
        result.rms_residual = selected_rms;
        result.success = true;

    } catch (const std::exception& e) {
        result.error_message = std::string("Fit failed: ") + e.what();
    }
    
    return result;
}

struct BGECandidateResult {
    BGEConfig cfg;
    int grid_spacing = 0;
    int evals = 0;
    float objective = std::numeric_limits<float>::infinity();
    float objective_raw = std::numeric_limits<float>::infinity();
    float objective_normalized = std::numeric_limits<float>::infinity();
    float cv_rms = std::numeric_limits<float>::infinity();
    float flatness = std::numeric_limits<float>::infinity();
    float roughness = std::numeric_limits<float>::infinity();
    bool success = false;
};

static void deterministic_split_indices(int n, float holdout_fraction,
                                        std::vector<int>* train_idx,
                                        std::vector<int>* val_idx) {
    train_idx->clear();
    val_idx->clear();
    if (n <= 0) return;

    const float hf = std::clamp(holdout_fraction, 0.05f, 0.50f);
    const int k = std::max(1, static_cast<int>(std::lround(1.0f / hf)));
    for (int i = 0; i < n; ++i) {
        if ((i % k) == 0) {
            val_idx->push_back(i);
        } else {
            train_idx->push_back(i);
        }
    }
    if (val_idx->empty()) {
        val_idx->push_back(n - 1);
        if (!train_idx->empty()) train_idx->pop_back();
    }
    if (train_idx->empty()) {
        train_idx->push_back(val_idx->back());
        val_idx->pop_back();
        if (val_idx->empty() && n > 1) val_idx->push_back(0);
    }
}

static float eval_model_rms_at_cells(const Matrix2Df& model,
                                     const std::vector<GridCell>& cells) {
    if (cells.empty()) return std::numeric_limits<float>::infinity();
    double sum_sq = 0.0;
    int n = 0;
    for (const auto& c : cells) {
        const int x = static_cast<int>(c.center_x);
        const int y = static_cast<int>(c.center_y);
        if (x < 0 || x >= model.cols() || y < 0 || y >= model.rows()) continue;
        const double r = static_cast<double>(c.bg_value - model(y, x));
        sum_sq += r * r;
        ++n;
    }
    if (n <= 0) return std::numeric_limits<float>::infinity();
    return static_cast<float>(std::sqrt(sum_sq / static_cast<double>(n)));
}

static float eval_model_flatness(const Matrix2Df& model, int step) {
    step = std::max(1, step);
    std::vector<float> grad_energy;
    grad_energy.reserve(
        static_cast<size_t>((model.rows() / step + 1) * (model.cols() / step + 1)));
    for (int y = 0; y + step < model.rows(); y += step) {
        for (int x = 0; x + step < model.cols(); x += step) {
            const float v = model(y, x);
            const float vx = model(y, x + step);
            const float vy = model(y + step, x);
            if (!(std::isfinite(v) && std::isfinite(vx) && std::isfinite(vy))) continue;
            const float gx = (vx - v) / static_cast<float>(step);
            const float gy = (vy - v) / static_cast<float>(step);
            grad_energy.push_back(gx * gx + gy * gy);
        }
    }
    if (grad_energy.size() < 8) return std::numeric_limits<float>::infinity();
    return robust_median(std::move(grad_energy));
}

static float eval_model_roughness(const Matrix2Df& model, int step) {
    step = std::max(1, step);
    std::vector<float> curvature_energy;
    curvature_energy.reserve(
        static_cast<size_t>((model.rows() / step + 1) * (model.cols() / step + 1)));
    for (int y = step; y + step < model.rows(); y += step) {
        for (int x = step; x + step < model.cols(); x += step) {
            const float c = model(y, x);
            const float xp = model(y, x + step);
            const float xm = model(y, x - step);
            const float yp = model(y + step, x);
            const float ym = model(y - step, x);
            const float xyp = model(y + step, x + step);
            const float xym = model(y - step, x + step);
            const float yxp = model(y + step, x - step);
            const float yxm = model(y - step, x - step);
            if (!(std::isfinite(c) && std::isfinite(xp) && std::isfinite(xm) &&
                  std::isfinite(yp) && std::isfinite(ym) && std::isfinite(xyp) &&
                  std::isfinite(xym) && std::isfinite(yxp) && std::isfinite(yxm))) {
                continue;
            }
            const float h = static_cast<float>(step);
            const float dxx = (xp - 2.0f * c + xm) / (h * h);
            const float dyy = (yp - 2.0f * c + ym) / (h * h);
            const float dxy = (xyp - xym - yxp + yxm) / (4.0f * h * h);
            curvature_energy.push_back(std::sqrt(dxx * dxx + dyy * dyy + 2.0f * dxy * dxy));
        }
    }
    if (curvature_energy.size() < 8) return std::numeric_limits<float>::infinity();
    return robust_median(std::move(curvature_energy));
}

static BGECandidateResult try_bge_candidate(const Matrix2Df& channel,
                                            const std::vector<TileMetrics>& tile_metrics,
                                            const TileGrid& tile_grid,
                                            const BGEConfig& cfg_try,
                                            int grid_spacing) {
    BGECandidateResult out;
    out.cfg = cfg_try;
    out.grid_spacing = grid_spacing;

    auto tile_samples = extract_tile_background_samples(channel, tile_metrics, tile_grid, cfg_try);
    auto grid_cells_all = aggregate_to_coarse_grid(tile_samples, channel.cols(), channel.rows(), grid_spacing, cfg_try);

    std::vector<GridCell> cells;
    cells.reserve(grid_cells_all.size());
    for (const auto& gc : grid_cells_all) {
        if (gc.valid) cells.push_back(gc);
    }
    if (cells.size() < 6) return out;

    std::sort(cells.begin(), cells.end(), [](const GridCell& a, const GridCell& b) {
        if (a.cell_y != b.cell_y) return a.cell_y < b.cell_y;
        return a.cell_x < b.cell_x;
    });

    std::vector<int> train_idx;
    std::vector<int> val_idx;
    deterministic_split_indices(static_cast<int>(cells.size()), cfg_try.autotune.holdout_fraction,
                                &train_idx, &val_idx);

    std::vector<GridCell> train_cells;
    std::vector<GridCell> val_cells;
    train_cells.reserve(train_idx.size());
    val_cells.reserve(val_idx.size());
    for (int i : train_idx) train_cells.push_back(cells[static_cast<size_t>(i)]);
    for (int i : val_idx) val_cells.push_back(cells[static_cast<size_t>(i)]);

    auto bg_model = fit_background_surface(train_cells, channel.cols(), channel.rows(), grid_spacing, cfg_try);
    if (!bg_model.success) return out;

    out.cv_rms = eval_model_rms_at_cells(bg_model.model, val_cells);
    const int step = std::max(4, grid_spacing / 4);
    out.flatness = eval_model_flatness(bg_model.model, step);
    out.roughness = eval_model_roughness(bg_model.model, step);

    std::vector<float> bvals;
    bvals.reserve(cells.size());
    for (const auto& gc : cells) bvals.push_back(gc.bg_value);
    const float bmed = std::max(kTiny, robust_median(std::move(bvals)));

    const float n_cv = out.cv_rms / bmed;
    const float n_flat = out.flatness / bmed;
    const float n_rough = out.roughness / bmed;

    out.objective_raw = out.cv_rms + cfg_try.autotune.alpha_flatness * out.flatness +
                        cfg_try.autotune.beta_roughness * out.roughness;
    out.objective_normalized =
        n_cv + cfg_try.autotune.alpha_flatness * n_flat +
        cfg_try.autotune.beta_roughness * n_rough;
    // Deterministic ranking uses raw objective for spec-conform scale behavior.
    out.objective = out.objective_raw;
    out.success =
        std::isfinite(out.objective_raw) && std::isfinite(out.objective_normalized);
    return out;
}

static BGECandidateResult auto_tune_bge_config_conservative(
    const Matrix2Df& channel,
    const std::vector<TileMetrics>& tile_metrics,
    const TileGrid& tile_grid,
    int base_grid_spacing,
    const BGEConfig& base_cfg) {

    const bool extended = (base_cfg.autotune.strategy == "extended");
    auto push_unique = [](std::vector<float>& out, float v) {
        for (float e : out) {
            if (std::fabs(e - v) < 1.0e-6f) return;
        }
        out.push_back(v);
    };

    // Keep conservative candidates strongly background-seeking.
    // High quantiles absorb nebula/object signal and may over-correct.
    std::vector<float> quantiles;
    push_unique(quantiles, std::min(base_cfg.sample_quantile, 0.25f));
    push_unique(quantiles, 0.12f);
    push_unique(quantiles, 0.18f);
    push_unique(quantiles, 0.24f);

    std::vector<float> structure_p;
    push_unique(structure_p, std::max(base_cfg.structure_thresh_percentile, 0.80f));
    push_unique(structure_p, 0.85f);
    push_unique(structure_p, 0.90f);
    std::vector<float> mu_factors = {
        base_cfg.fit.rbf_mu_factor,
        1.4f,
    };
    if (extended) {
        // Extended mode may test one less conservative quantile.
        push_unique(quantiles, 0.30f);
        push_unique(structure_p, 0.95f);
        mu_factors.push_back(1.8f);
    }

    BGECandidateResult best;
    int evals = 0;

    for (float q : quantiles) {
        for (float sp : structure_p) {
            for (float mf : mu_factors) {
                if (evals >= std::max(1, base_cfg.autotune.max_evals)) break;
                BGEConfig cfg_try = base_cfg;
                cfg_try.sample_quantile = std::clamp(q, 0.05f, 0.50f);
                cfg_try.structure_thresh_percentile = std::clamp(sp, 0.50f, 0.99f);
                cfg_try.fit.rbf_mu_factor = std::max(0.2f, mf);

                BGECandidateResult res = try_bge_candidate(
                    channel, tile_metrics, tile_grid, cfg_try, base_grid_spacing);
                ++evals;
                if (!res.success) continue;

                if (!best.success || res.objective < best.objective) {
                    best = res;
                } else if (std::fabs(res.objective - best.objective) < 1e-6f) {
                    if (res.roughness < best.roughness) {
                        best = res;
                    } else if (std::fabs(res.roughness - best.roughness) < 1e-6f) {
                        // Deterministic tie-break: prefer coarser effective model.
                        if (res.cfg.fit.rbf_mu_factor > best.cfg.fit.rbf_mu_factor) {
                            best = res;
                        }
                    }
                }
            }
            if (evals >= std::max(1, base_cfg.autotune.max_evals)) break;
        }
        if (evals >= std::max(1, base_cfg.autotune.max_evals)) break;
    }

    if (!best.success) {
        best.cfg = base_cfg;
        best.grid_spacing = base_grid_spacing;
    }
    best.evals = evals;
    return best;
}

// Main BGE function (v3.3 §6.3)
bool apply_background_extraction(
    Matrix2Df& R, Matrix2Df& G, Matrix2Df& B,
    const std::vector<TileMetrics>& tile_metrics,
    const TileGrid& tile_grid,
    const BGEConfig& config,
    BGEDiagnostics* diagnostics) {

    if (diagnostics != nullptr) {
        diagnostics->attempted = config.enabled;
        diagnostics->success = false;
        diagnostics->image_width = 0;
        diagnostics->image_height = 0;
        diagnostics->grid_spacing = 0;
        diagnostics->method = config.fit.method;
        diagnostics->robust_loss = config.fit.robust_loss;
        diagnostics->insufficient_cell_strategy = config.grid.insufficient_cell_strategy;
        diagnostics->autotune_enabled = config.autotune.enabled;
        diagnostics->autotune_strategy = config.autotune.strategy;
        diagnostics->autotune_max_evals = config.autotune.max_evals;
        diagnostics->autotune_evals = 0;
        diagnostics->autotune_best_objective = 0.0f;
        diagnostics->autotune_best_objective_raw = 0.0f;
        diagnostics->autotune_best_objective_normalized = 0.0f;
        diagnostics->autotune_best_cv_rms = 0.0f;
        diagnostics->autotune_best_flatness = 0.0f;
        diagnostics->autotune_best_roughness = 0.0f;
        diagnostics->autotune_selected_sample_quantile = 0.0f;
        diagnostics->autotune_selected_structure_thresh_percentile = 0.0f;
        diagnostics->autotune_selected_rbf_mu_factor = 0.0f;
        diagnostics->autotune_fallback_used = false;
        diagnostics->channels.clear();
    }
    
    if (!config.enabled) {
        return false;
    }
    
    const int H = R.rows();
    const int W = R.cols();

    if (diagnostics != nullptr) {
        diagnostics->image_width = W;
        diagnostics->image_height = H;
    }
    
    std::cerr << "[BGE] Starting background extraction (v3.3 §6.3)" << std::endl;
    std::cerr << "[BGE] Image size: " << W << "x" << H << std::endl;
    std::cerr << "[BGE] Method: " << config.fit.method << std::endl;
    
    // Compute grid spacing (v3.3 §6.3.8)
    int grid_spacing = compute_grid_spacing(W, H, tile_grid.tile_size, config);
    std::cerr << "[BGE] Grid spacing: " << grid_spacing << " px" << std::endl;

    if (diagnostics != nullptr) {
        diagnostics->grid_spacing = grid_spacing;
    }

    bool any_channel_applied = false;
    bool global_autotune_set = false;
    
    // Process each channel
    for (int c = 0; c < 3; ++c) {
        Matrix2Df* channel = (c == 0) ? &R : (c == 1) ? &G : &B;
        const char* channel_name = (c == 0) ? "R" : (c == 1) ? "G" : "B";
        const Matrix2Df channel_before = *channel;

        BGEChannelDiagnostics ch_diag;
        ch_diag.channel_name = channel_name;
        ch_diag.autotune_enabled = config.autotune.enabled;
        ch_diag.autotune_selected_grid_spacing = grid_spacing;
        ch_diag.input_stats = stats_from_matrix(channel_before);
        
        std::cerr << "[BGE] Processing channel " << channel_name << std::endl;
        
        BGEConfig channel_cfg = config;
        int channel_grid_spacing = grid_spacing;
        if (config.autotune.enabled) {
            BGECandidateResult tune_res =
                auto_tune_bge_config_conservative(*channel, tile_metrics, tile_grid,
                                                  grid_spacing, config);
            channel_cfg = tune_res.cfg;
            channel_grid_spacing = tune_res.grid_spacing;

            if (diagnostics != nullptr) {
                diagnostics->autotune_evals += tune_res.evals;
                diagnostics->autotune_fallback_used =
                    diagnostics->autotune_fallback_used || !tune_res.success;
                if (tune_res.success) {
                    if (!global_autotune_set ||
                        tune_res.objective < diagnostics->autotune_best_objective_raw) {
                        global_autotune_set = true;
                        diagnostics->autotune_best_objective = tune_res.objective_raw;
                        diagnostics->autotune_best_objective_raw = tune_res.objective_raw;
                        diagnostics->autotune_best_objective_normalized =
                            tune_res.objective_normalized;
                        diagnostics->autotune_best_cv_rms = tune_res.cv_rms;
                        diagnostics->autotune_best_flatness = tune_res.flatness;
                        diagnostics->autotune_best_roughness = tune_res.roughness;
                        diagnostics->autotune_selected_sample_quantile =
                            tune_res.cfg.sample_quantile;
                        diagnostics->autotune_selected_structure_thresh_percentile =
                            tune_res.cfg.structure_thresh_percentile;
                        diagnostics->autotune_selected_rbf_mu_factor =
                            tune_res.cfg.fit.rbf_mu_factor;
                    }
                }
            }

            ch_diag.autotune_evals = tune_res.evals;
            ch_diag.autotune_selected_grid_spacing = channel_grid_spacing;
            ch_diag.autotune_fallback_used = !tune_res.success;
            if (tune_res.success) {
                ch_diag.autotune_best_objective = tune_res.objective_raw;
                ch_diag.autotune_best_objective_raw = tune_res.objective_raw;
                ch_diag.autotune_best_objective_normalized =
                    tune_res.objective_normalized;
                ch_diag.autotune_best_cv_rms = tune_res.cv_rms;
                ch_diag.autotune_best_flatness = tune_res.flatness;
                ch_diag.autotune_best_roughness = tune_res.roughness;
                ch_diag.autotune_selected_sample_quantile =
                    tune_res.cfg.sample_quantile;
                ch_diag.autotune_selected_structure_thresh_percentile =
                    tune_res.cfg.structure_thresh_percentile;
                ch_diag.autotune_selected_rbf_mu_factor =
                    tune_res.cfg.fit.rbf_mu_factor;
            }
        }

        // Extract tile background samples (v3.3 §6.3.2)
        auto tile_samples = extract_tile_background_samples(*channel, tile_metrics, tile_grid, channel_cfg);
        int n_valid = std::count_if(tile_samples.begin(), tile_samples.end(), 
                                     [](const auto& s) { return s.valid; });
        std::cerr << "[BGE]   Tile samples: " << n_valid << "/" << tile_samples.size() << " valid" << std::endl;

        ch_diag.tile_samples_total = static_cast<int>(tile_samples.size());
        ch_diag.tile_samples_valid = n_valid;
        ch_diag.sample_bg_values.reserve(static_cast<size_t>(n_valid));
        ch_diag.sample_weight_values.reserve(static_cast<size_t>(n_valid));
        for (const auto& s : tile_samples) {
            if (!s.valid) continue;
            ch_diag.sample_bg_values.push_back(s.bg_value);
            ch_diag.sample_weight_values.push_back(s.weight);
        }
        ch_diag.sample_bg_stats = stats_from_values(ch_diag.sample_bg_values);
        ch_diag.sample_weight_stats = stats_from_values(ch_diag.sample_weight_values);

        const int n_total_samples = std::max(1, ch_diag.tile_samples_total);
        const float valid_fraction =
            static_cast<float>(ch_diag.tile_samples_valid) /
            static_cast<float>(n_total_samples);
        if (ch_diag.tile_samples_valid < kMinValidSamplesForApply ||
            valid_fraction < kMinValidSampleFractionForApply) {
            std::cerr << "[BGE]   Warning: insufficient robust tile samples for channel "
                      << channel_name << " (" << ch_diag.tile_samples_valid << "/"
                      << ch_diag.tile_samples_total << ", frac=" << valid_fraction
                      << "), skipping channel" << std::endl;
            if (diagnostics != nullptr) diagnostics->channels.push_back(std::move(ch_diag));
            continue;
        }
        
        // Aggregate to coarse grid (v3.3 §6.3.3)
        auto grid_cells = aggregate_to_coarse_grid(tile_samples, W, H, channel_grid_spacing, channel_cfg);
        std::cerr << "[BGE]   Grid cells: " << grid_cells.size() << " valid" << std::endl;

        ch_diag.grid_cells_valid = static_cast<int>(grid_cells.size());
        ch_diag.grid_cells = grid_cells;

        if (grid_cells.size() < 3) {
            std::cerr << "[BGE]   Warning: Too few grid cells, skipping channel " << channel_name << std::endl;
            if (diagnostics != nullptr) diagnostics->channels.push_back(std::move(ch_diag));
            continue;
        }
        
        // Fit background surface (v3.3 §6.3.7)
        auto bg_model = fit_background_surface(grid_cells, W, H, channel_grid_spacing, channel_cfg);
        
        if (!bg_model.success) {
            std::cerr << "[BGE]   Error: " << bg_model.error_message << std::endl;
            if (diagnostics != nullptr) diagnostics->channels.push_back(std::move(ch_diag));
            continue;
        }

        std::cerr << "[BGE]   Fit RMS residual: " << bg_model.rms_residual << std::endl;

        ch_diag.fit_success = true;
        ch_diag.fit_rms_residual = bg_model.rms_residual;
        // Clamp model extrapolation to the support range of robust tile samples.
        // This avoids edge over-correction when the fitted surface drifts outside
        // physically observed background values.
        if (ch_diag.sample_bg_values.size() >= 16) {
            std::vector<float> sample_vals = ch_diag.sample_bg_values;
            const float q05 = robust_quantile(sample_vals, 0.05f);
            const float q95 = robust_quantile(std::move(sample_vals), 0.95f);
            const float guard_pad =
                std::max(0.75f, 2.0f * std::max(0.0f, bg_model.rms_residual));
            const float model_min = q05 - guard_pad;
            const float model_max = q95 + guard_pad;
            int clipped = 0;
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    float &mv = bg_model.model(y, x);
                    if (!std::isfinite(mv)) continue;
                    const float clamped = std::clamp(mv, model_min, model_max);
                    if (std::fabs(clamped - mv) > 1.0e-6f) ++clipped;
                    mv = clamped;
                }
            }
            if (clipped > 0) {
                std::cerr << "[BGE]   Model clamp: " << clipped
                          << " px to [" << model_min << ".." << model_max << "]"
                          << std::endl;
            }
        }
        ch_diag.model_stats = stats_from_matrix(bg_model.model);
        const float pedestal = ch_diag.model_stats.median;
        
        // Subtract background (v3.3 §6.3.5) while preserving a channel pedestal
        // for downstream PCC photometry stability. This removes spatial gradient
        // but keeps numeric background level in a positive range.
        Matrix2Df corrected = channel_before;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const float vin = channel_before(y, x);
                if (!(std::isfinite(vin) && vin > 0.0f)) {
                    corrected(y, x) = 0.0f;
                    continue;
                }
                corrected(y, x) = vin - bg_model.model(y, x) + pedestal;
            }
        }

        // Safety guard: never keep a correction that worsens coarse background
        // uniformity in low-intensity, low-gradient regions.
        const float flat_pre = spatial_background_spread(channel_before);
        const float flat_post = spatial_background_spread(corrected);
        bool accept_correction = true;
        constexpr float kMaxFlatnessWorsenFactor = 1.05f;
        if (std::isfinite(flat_pre) && std::isfinite(flat_post) &&
            flat_post > flat_pre * kMaxFlatnessWorsenFactor) {
            std::cerr << "[BGE]   Flatness guard rejected channel " << channel_name
                      << " (pre=" << flat_pre << ", post=" << flat_post << ")"
                      << std::endl;
            accept_correction = false;
        }

        if (!accept_correction) {
            ch_diag.applied = false;
            ch_diag.output_stats = stats_from_matrix(channel_before);
            ch_diag.mean_shift = 0.0f;
            if (diagnostics != nullptr) diagnostics->channels.push_back(std::move(ch_diag));
            continue;
        }

        *channel = std::move(corrected);
        ch_diag.applied = true;
        ch_diag.output_stats = stats_from_matrix(*channel);
        ch_diag.mean_shift = ch_diag.output_stats.mean - ch_diag.input_stats.mean;
        ch_diag.residual_values.reserve(ch_diag.grid_cells.size());
        for (const auto& gc : ch_diag.grid_cells) {
            const int px = static_cast<int>(gc.center_x);
            const int py = static_cast<int>(gc.center_y);
            if (px >= 0 && px < W && py >= 0 && py < H) {
                ch_diag.residual_values.push_back(gc.bg_value - bg_model.model(py, px));
            }
        }
        ch_diag.residual_stats = stats_from_values(ch_diag.residual_values);

        any_channel_applied = true;

        std::cerr << "[BGE]   Background subtracted from channel " << channel_name << std::endl;

        if (diagnostics != nullptr) diagnostics->channels.push_back(std::move(ch_diag));
    }

    std::cerr << "[BGE] Background extraction complete" << std::endl;
    if (diagnostics != nullptr) {
        diagnostics->success = any_channel_applied;
    }
    return any_channel_applied;
}

} // namespace tile_compile::image
