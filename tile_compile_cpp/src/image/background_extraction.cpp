#include "tile_compile/image/background_extraction.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace tile_compile::image {

namespace {

constexpr float kTiny = 1.0e-12f;
constexpr float kMinUsableTileFraction = 0.05f;

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
    
    // G = clip(max(2*T, min(W,H)/N_g), G_min, G_max)
    int G_from_tiles = 2 * tile_size;
    int G_from_resolution = min_dim / config.grid.N_g;
    int G_max = static_cast<int>(min_dim * config.grid.G_max_fraction);
    
    int G = std::max(G_from_tiles, G_from_resolution);
    G = std::max(G, config.grid.G_min_px);
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

    // Compute structure threshold (percentile over E/sigma)
    std::vector<float> structure_scores;
    structure_scores.reserve(n_tiles);
    for (size_t ti = 0; ti < n_tiles; ++ti) {
        const auto& tm = tile_metrics[ti];
        if (tm.noise > 1e-6f && std::isfinite(tm.gradient_energy)) {
            structure_scores.push_back(tm.gradient_energy / tm.noise);
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
        
        // Exclude high-structure tiles (v3.3 §6.3.2a)
        float tile_structure = (tm.noise > 1e-6f) ? (tm.gradient_energy / tm.noise) : 0.0f;
        if (tile_structure > structure_thresh) {
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
        star_mask = dilate_mask(star_mask, tw, th, std::max(0, config.mask.star_dilate_px));

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
        sample.weight = std::exp(-2.0f * tile_structure) * (1.0f - masked_fraction);
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
            const float lo = bg_med - 6.0f * bg_sigma;
            const float hi = bg_med + 6.0f * bg_sigma;
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
        const std::string method = config.fit.method;
        if (method == "rbf") {
            result.model = fit_rbf_surface(grid_cells, image_width, image_height,
                                           grid_spacing, config);
        } else if (method == "spline") {
            // Practical spline backend: thin-plate RBF with conservative smoothing.
            BGEConfig spline_cfg = config;
            spline_cfg.fit.method = "rbf";
            spline_cfg.fit.rbf_phi = "thinplate";
            spline_cfg.fit.rbf_lambda = std::max(1.0e-4f, spline_cfg.fit.rbf_lambda);
            result.model = fit_rbf_surface(grid_cells, image_width, image_height,
                                           grid_spacing, spline_cfg);
        } else if (method == "poly") {
            result.model = fit_polynomial_surface(grid_cells, image_width, image_height, config);
        } else if (method == "bicubic") {
            // Fallback approximation: cubic polynomial surface.
            BGEConfig cubic_cfg = config;
            cubic_cfg.fit.polynomial_order = std::max(3, cubic_cfg.fit.polynomial_order);
            result.model = fit_polynomial_surface(grid_cells, image_width, image_height, cubic_cfg);
        } else {
            result.error_message = "Unsupported fit method: " + method;
            return result;
        }
        
        // Compute RMS residual
        float sum_sq = 0.0f;
        for (const auto& cell : grid_cells) {
            int cx = static_cast<int>(cell.center_x);
            int cy = static_cast<int>(cell.center_y);
            if (cx >= 0 && cx < image_width && cy >= 0 && cy < image_height) {
                float residual = cell.bg_value - result.model(cy, cx);
                sum_sq += residual * residual;
            }
        }
        result.rms_residual = std::sqrt(sum_sq / grid_cells.size());
        result.success = true;
        
    } catch (const std::exception& e) {
        result.error_message = std::string("Fit failed: ") + e.what();
    }
    
    return result;
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
    
    // Process each channel
    for (int c = 0; c < 3; ++c) {
        Matrix2Df* channel = (c == 0) ? &R : (c == 1) ? &G : &B;
        const char* channel_name = (c == 0) ? "R" : (c == 1) ? "G" : "B";
        const Matrix2Df channel_before = *channel;

        BGEChannelDiagnostics ch_diag;
        ch_diag.channel_name = channel_name;
        ch_diag.input_stats = stats_from_matrix(channel_before);
        
        std::cerr << "[BGE] Processing channel " << channel_name << std::endl;
        
        // Extract tile background samples (v3.3 §6.3.2)
        auto tile_samples = extract_tile_background_samples(*channel, tile_metrics, tile_grid, config);
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
        
        // Aggregate to coarse grid (v3.3 §6.3.3)
        auto grid_cells = aggregate_to_coarse_grid(tile_samples, W, H, grid_spacing, config);
        std::cerr << "[BGE]   Grid cells: " << grid_cells.size() << " valid" << std::endl;

        ch_diag.grid_cells_valid = static_cast<int>(grid_cells.size());
        ch_diag.grid_cells = grid_cells;

        if (grid_cells.size() < 3) {
            std::cerr << "[BGE]   Warning: Too few grid cells, skipping channel " << channel_name << std::endl;
            if (diagnostics != nullptr) diagnostics->channels.push_back(std::move(ch_diag));
            continue;
        }
        
        // Fit background surface (v3.3 §6.3.7)
        auto bg_model = fit_background_surface(grid_cells, W, H, grid_spacing, config);
        
        if (!bg_model.success) {
            std::cerr << "[BGE]   Error: " << bg_model.error_message << std::endl;
            if (diagnostics != nullptr) diagnostics->channels.push_back(std::move(ch_diag));
            continue;
        }

        std::cerr << "[BGE]   Fit RMS residual: " << bg_model.rms_residual << std::endl;

        ch_diag.fit_success = true;
        ch_diag.fit_rms_residual = bg_model.rms_residual;
        ch_diag.model_stats = stats_from_matrix(bg_model.model);
        const float pedestal = ch_diag.model_stats.median;
        
        // Subtract background (v3.3 §6.3.5) while preserving a channel pedestal
        // for downstream PCC photometry stability. This removes spatial gradient
        // but keeps numeric background level in a positive range.
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const float vin = channel_before(y, x);
                if (!(std::isfinite(vin) && vin > 0.0f)) {
                    (*channel)(y, x) = 0.0f;
                    continue;
                }
                (*channel)(y, x) = vin - bg_model.model(y, x) + pedestal;
            }
        }

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
