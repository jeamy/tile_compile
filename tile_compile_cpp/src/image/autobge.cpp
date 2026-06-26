#include "tile_compile/image/background_extraction.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace tile_compile::image {

// ===== AutoBGE Implementation =====

namespace {

constexpr float kTiny = 1.0e-12f;

float sorted_quantile_local(const std::vector<float>& sorted_values, float q) {
  if (sorted_values.empty())
    return 0.0f;
  q = std::max(0.0f, std::min(1.0f, q));
  const float pos = q * static_cast<float>(sorted_values.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = std::min(sorted_values.size() - 1, lo + 1);
  const float t = pos - static_cast<float>(lo);
  return sorted_values[lo] * (1.0f - t) + sorted_values[hi] * t;
}

BGEValueStats stats_from_values_local(const std::vector<float>& values) {
  BGEValueStats stats;
  std::vector<float> finite;
  finite.reserve(values.size());
  double sum = 0.0;
  for (float v : values) {
    if (!std::isfinite(v))
      continue;
    finite.push_back(v);
    sum += static_cast<double>(v);
  }
  stats.n = static_cast<int>(finite.size());
  if (finite.empty())
    return stats;
  std::sort(finite.begin(), finite.end());
  stats.min = finite.front();
  stats.max = finite.back();
  stats.median = finite[finite.size() / 2];
  stats.mean = static_cast<float>(sum / static_cast<double>(finite.size()));
  double var = 0.0;
  for (float v : finite) {
    const double d = static_cast<double>(v) - stats.mean;
    var += d * d;
  }
  stats.std = static_cast<float>(std::sqrt(var / static_cast<double>(finite.size())));
  return stats;
}

BGEValueStats stats_from_matrix_local(const Matrix2Df& matrix) {
  std::vector<float> values;
  values.reserve(static_cast<size_t>(matrix.size()));
  for (int i = 0; i < matrix.size(); ++i) {
    const float v = matrix.data()[i];
    if (std::isfinite(v))
      values.push_back(v);
  }
  return stats_from_values_local(values);
}

int autobge_polynomial_term_count(int degree) {
  return (degree + 1) * (degree + 2) / 2;
}

int autobge_min_stage1_points(const BGEConfig::AutoBGEConfig& config) {
  return autobge_polynomial_term_count(config.poly_degree) + 4;
}

constexpr int kAutoBGEMinStage2Points = 16;

float percentile_of_valid(const std::vector<float>& sorted_vals, float pct) {
  if (sorted_vals.empty()) return 0.0f;
  const size_t idx = static_cast<size_t>(pct * (sorted_vals.size() - 1));
  return sorted_vals[idx];
}

std::vector<float> collect_patch_values(const Matrix2Df& img, int cx, int cy,
                                        int patch_size) {
  const int half = patch_size / 2;
  const int r0 = std::max(0, cy - half);
  const int r1 = std::min(static_cast<int>(img.rows()) - 1, cy + half);
  const int c0 = std::max(0, cx - half);
  const int c1 = std::min(static_cast<int>(img.cols()) - 1, cx + half);
  std::vector<float> vals;
  vals.reserve((r1 - r0 + 1) * (c1 - c0 + 1));
  for (int r = r0; r <= r1; ++r)
    for (int c = c0; c <= c1; ++c) {
      const float v = img(r, c);
      if (std::isfinite(v) && v > 0.0f) vals.push_back(v);
    }
  return vals;
}

float median_from_values(std::vector<float> vals) {
  if (vals.empty()) return 0.0f;
  std::sort(vals.begin(), vals.end());
  return vals[vals.size() / 2];
}

float patch_estimate(const Matrix2Df& img, int cx, int cy, int patch_size,
                     const std::string& estimator) {
  std::vector<float> vals = collect_patch_values(img, cx, cy, patch_size);
  if (vals.empty())
    return 0.0f;
  if (estimator != "sigma_clipped_median")
    return median_from_values(std::move(vals));

  const float med = median_from_values(vals);
  std::vector<float> dev;
  dev.reserve(vals.size());
  for (float v : vals)
    dev.push_back(std::fabs(v - med));
  const float mad = median_from_values(std::move(dev));
  const float sigma = 1.4826f * mad;
  if (!(sigma > kTiny))
    return med;

  std::vector<float> clipped;
  clipped.reserve(vals.size());
  const float lo = med - 3.0f * sigma;
  const float hi = med + 3.0f * sigma;
  for (float v : vals) {
    if (v >= lo && v <= hi)
      clipped.push_back(v);
  }
  return clipped.empty() ? med : median_from_values(std::move(clipped));
}

float patch_median(const Matrix2Df& img, int cx, int cy, int patch_size) {
  return patch_estimate(img, cx, cy, patch_size, "median");
}

std::vector<uint8_t> downsample_mask_majority(const std::vector<uint8_t>& mask,
                                             int rows, int cols, int scale,
                                             int out_rows, int out_cols) {
  std::vector<uint8_t> out(static_cast<size_t>(out_rows * out_cols), 0);
  if (scale <= 1) {
    for (int r = 0; r < out_rows; ++r)
      for (int c = 0; c < out_cols; ++c)
        out[static_cast<size_t>(r * out_cols + c)] =
            mask[static_cast<size_t>(r * cols + c)] != 0 ? 1 : 0;
    return out;
  }
  for (int r = 0; r < out_rows; ++r) {
    for (int c = 0; c < out_cols; ++c) {
      int valid = 0;
      int total = 0;
      for (int sy = 0; sy < scale; ++sy) {
        const int sr = r * scale + sy;
        if (sr >= rows)
          break;
        for (int sx = 0; sx < scale; ++sx) {
          const int sc = c * scale + sx;
          if (sc >= cols)
            break;
          ++total;
          if (mask[static_cast<size_t>(sr * cols + sc)] != 0)
            ++valid;
        }
      }
      out[static_cast<size_t>(r * out_cols + c)] =
          valid * 2 >= std::max(1, total) ? 1 : 0;
    }
  }
  return out;
}

} // anonymous namespace

Matrix2Df transform_to_autobge_working_space(
    const Matrix2Df& channel, const BGEConfig::AutoBGEConfig& config,
    StretchParams* params, int channel_index,
    const std::vector<uint8_t>* valid_mask) {

  const int rows = static_cast<int>(channel.rows());
  const int cols = static_cast<int>(channel.cols());

  if (config.stretch_mode == "none") {
    if (params) {
      if (static_cast<int>(params->original_mins.size()) <= channel_index) {
        params->original_mins.resize(channel_index + 1, 0.0f);
        params->original_medians.resize(channel_index + 1, 0.0f);
        params->linear_offsets.resize(channel_index + 1, 0.0f);
        params->linear_scales.resize(channel_index + 1, 1.0f);
        params->mtf_targets.resize(channel_index + 1, 0.25f);
      }
      params->mode = "none";
    }
    return channel;
  }

  // Collect valid, positive, finite pixels for percentile computation
  std::vector<float> valid_vals;
  valid_vals.reserve(rows * cols);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) {
      const float v = channel(r, c);
      if (std::isfinite(v) && v > 0.0f) {
        if (!valid_mask || (*valid_mask)[r * cols + c] != 0)
          valid_vals.push_back(v);
      }
    }

  if (valid_vals.size() < 16) {
    // Fallback to none for this channel
    if (params) {
      if (static_cast<int>(params->original_mins.size()) <= channel_index) {
        params->original_mins.resize(channel_index + 1, 0.0f);
        params->original_medians.resize(channel_index + 1, 0.0f);
        params->linear_offsets.resize(channel_index + 1, 0.0f);
        params->linear_scales.resize(channel_index + 1, 1.0f);
        params->mtf_targets.resize(channel_index + 1, 0.25f);
      }
      params->mode = "none";
    }
    return channel;
  }

  std::sort(valid_vals.begin(), valid_vals.end());
  const float p01 = percentile_of_valid(valid_vals, 0.01f);
  const float p99 = percentile_of_valid(valid_vals, 0.99f);
  const float vmin = valid_vals.front();
  std::vector<float> shifted_vals;
  shifted_vals.reserve(valid_vals.size());
  for (float v : valid_vals)
    shifted_vals.push_back(std::max(0.0f, v - vmin));
  std::sort(shifted_vals.begin(), shifted_vals.end());
  const float median_shifted = percentile_of_valid(shifted_vals, 0.50f);
  const float median = percentile_of_valid(valid_vals, 0.50f);

  if (params) {
    if (static_cast<int>(params->original_mins.size()) <= channel_index) {
      params->original_mins.resize(channel_index + 1, 0.0f);
      params->original_medians.resize(channel_index + 1, 0.0f);
      params->linear_offsets.resize(channel_index + 1, 0.0f);
      params->linear_scales.resize(channel_index + 1, 1.0f);
      params->mtf_targets.resize(channel_index + 1, 0.25f);
    }
    params->original_mins[channel_index] = vmin;
    params->original_medians[channel_index] = median;
  }

  if (config.stretch_mode == "linear") {
    const float scale = std::max(p99 - p01, kTiny);
    if (params) {
      params->linear_offsets[channel_index] = p01;
      params->linear_scales[channel_index] = scale;
      params->mode = "linear";
    }
    Matrix2Df out = (channel.array() - p01) / scale;
    return out;
  }

  // mtf: AutoBGE-style unlinked non-linear stretch on channel-min-shifted data.
  const float target = config.stretch_target_median;
  const float mtf_median = median_shifted > 0.0f ? median_shifted : 1.0f;
  if (params) {
    params->original_mins[channel_index] = vmin;
    params->original_medians[channel_index] = mtf_median;
    params->linear_offsets[channel_index] = 0.0f;
    params->linear_scales[channel_index] = 1.0f;
    params->mtf_targets[channel_index] = target;
    params->mode = "mtf";
  }

  Matrix2Df out(rows, cols);
  const float eps = 1.0e-10f;
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) {
      const float v = channel(r, c);
      if (!std::isfinite(v) || v <= 0.0f) {
        out(r, c) = 0.0f;
        continue;
      }
      const float shifted = std::max(0.0f, v - vmin);
      const float numerator = (mtf_median - 1.0f) * target * shifted;
      const float denominator =
          mtf_median * (target + shifted - 1.0f) - target * shifted;
      out(r, c) = numerator / (denominator + eps);
    }
  return out;
}

Matrix2Df transform_from_autobge_working_space(
    const Matrix2Df& channel, const StretchParams& params, int channel_index) {

  if (params.mode == "none" || channel_index >= static_cast<int>(params.linear_scales.size()))
    return channel;

  if (params.mode == "linear") {
    const float offset = params.linear_offsets[channel_index];
    const float scale = params.linear_scales[channel_index];
    return channel.array() * scale + offset;
  }

  // mtf inverse for:
  // y = ((m - 1) * t * shifted) / (m * (t + shifted - 1) - t * shifted)
  const int rows = static_cast<int>(channel.rows());
  const int cols = static_cast<int>(channel.cols());
  const float median_shifted = params.original_medians[channel_index];
  const float orig_min = params.original_mins[channel_index];
  const float target =
      channel_index < static_cast<int>(params.mtf_targets.size())
          ? params.mtf_targets[channel_index]
          : 0.25f;

  Matrix2Df out(rows, cols);
  const float eps = 1.0e-10f;
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) {
      const float corrected = channel(r, c);
      if (!std::isfinite(corrected)) {
        out(r, c) = corrected;
        continue;
      }
      const float numerator = corrected * median_shifted * (target - 1.0f);
      const float denominator =
          (median_shifted - 1.0f) * target -
          corrected * (median_shifted - target);
      out(r, c) = numerator / (denominator + eps) + orig_min;
    }
  return out;
}

Matrix2Df downsample_area(const Matrix2Df& image, int scale) {
  if (scale <= 1) return image;
  const int src_rows = static_cast<int>(image.rows());
  const int src_cols = static_cast<int>(image.cols());
  const int dst_rows = src_rows / scale;
  const int dst_cols = src_cols / scale;
  if (dst_rows < 1 || dst_cols < 1) return image;

  Matrix2Df out(dst_rows, dst_cols);
  const float inv_area = 1.0f / static_cast<float>(scale * scale);
  for (int dr = 0; dr < dst_rows; ++dr) {
    for (int dc = 0; dc < dst_cols; ++dc) {
      float sum = 0.0f;
      int count = 0;
      for (int sy = 0; sy < scale; ++sy) {
        const int sr = dr * scale + sy;
        if (sr >= src_rows) break;
        for (int sx = 0; sx < scale; ++sx) {
          const int sc = dc * scale + sx;
          if (sc >= src_cols) break;
          const float v = image(sr, sc);
          if (std::isfinite(v)) {
            sum += v;
            ++count;
          }
        }
      }
      out(dr, dc) = count > 0 ? sum * inv_area : 0.0f;
    }
  }
  return out;
}

Matrix2Df upscale_lanczos4(const Matrix2Df& background, int target_rows, int target_cols) {
  const int src_rows = static_cast<int>(background.rows());
  const int src_cols = static_cast<int>(background.cols());
  if (src_rows == target_rows && src_cols == target_cols) return background;

  Matrix2Df out(target_rows, target_cols);
  const float scale_r = static_cast<float>(src_rows) / target_rows;
  const float scale_c = static_cast<float>(src_cols) / target_cols;
  constexpr int a = 4; // Lanczos kernel size

  auto lanczos_kernel = [](float x, int n) -> float {
    if (x == 0.0f) return 1.0f;
    if (std::abs(x) >= n) return 0.0f;
    const float px = static_cast<float>(M_PI) * x;
    const float npx = px / n;
    return std::sin(px) * std::sin(npx) / (px * npx);
  };

  for (int tr = 0; tr < target_rows; ++tr) {
    const float src_r = (tr + 0.5f) * scale_r - 0.5f;
    const int r0 = std::max(0, static_cast<int>(std::floor(src_r)) - a + 1);
    const int r1 = std::min(src_rows - 1, static_cast<int>(std::floor(src_r)) + a);
    for (int tc = 0; tc < target_cols; ++tc) {
      const float src_c = (tc + 0.5f) * scale_c - 0.5f;
      const int c0 = std::max(0, static_cast<int>(std::floor(src_c)) - a + 1);
      const int c1 = std::min(src_cols - 1, static_cast<int>(std::floor(src_c)) + a);

      float weight_sum = 0.0f;
      float value_sum = 0.0f;
      for (int sr = r0; sr <= r1; ++sr) {
        const float wr = lanczos_kernel(src_r - sr, a);
        if (wr == 0.0f) continue;
        for (int sc = c0; sc <= c1; ++sc) {
          const float wc = lanczos_kernel(src_c - sc, a);
          if (wc == 0.0f) continue;
          const float w = wr * wc;
          value_sum += w * background(sr, sc);
          weight_sum += w;
        }
      }
      out(tr, tc) = weight_sum > kTiny ? value_sum / weight_sum : 0.0f;
    }
  }
  return out;
}

// Generate AutoBGE sample points: border + grid + gradient-descent refined points
std::vector<SamplePoint> generate_autobge_sample_points(
    const Matrix2Df& image_downsampled,
    const BGEConfig::AutoBGEConfig& config,
    const std::vector<uint8_t>* valid_mask_downsampled,
    std::mt19937* rng,
    bool random_downselection) {

  const int rows = static_cast<int>(image_downsampled.rows());
  const int cols = static_cast<int>(image_downsampled.cols());
  if (rows < 4 || cols < 4) return {};

  std::mt19937 local_rng;
  if (rng) {
    local_rng = *rng;
  } else {
    local_rng.seed(42);
  }

  // Compute bright exclusion threshold
  std::vector<float> valid_vals;
  valid_vals.reserve(rows * cols);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) {
      const float v = image_downsampled(r, c);
      if (std::isfinite(v) && v > 0.0f) {
        if (!valid_mask_downsampled || (*valid_mask_downsampled)[r * cols + c] != 0)
          valid_vals.push_back(v);
      }
    }
  if (valid_vals.empty()) return {};

  std::sort(valid_vals.begin(), valid_vals.end());
  const int bright_idx = static_cast<int>(
      static_cast<float>(valid_vals.size()) * config.bright_exclusion_fraction);
  const float bright_threshold = valid_vals[std::min(bright_idx,
      static_cast<int>(valid_vals.size()) - 1)];

  // Helper: check if a point is valid (not bright, not masked)
  auto is_valid_point = [&](int x, int y) -> bool {
    if (x < 0 || x >= cols || y < 0 || y >= rows) return false;
    if (valid_mask_downsampled && (*valid_mask_downsampled)[y * cols + x] == 0)
      return false;
    const float v = image_downsampled(y, x);
    if (!std::isfinite(v) || v <= 0.0f) return false;
    if (v >= bright_threshold) return false;
    return true;
  };

  // Gradient descent to dim spot: move point toward dimmer neighbors
  auto gradient_descent_to_dim = [&](int& x, int& y) -> void {
    const int max_iters = config.gradient_descent_max_iters;
    const int half_patch = config.patch_size / 2;
    for (int iter = 0; iter < max_iters; ++iter) {
      float current_val = patch_estimate(image_downsampled, x, y,
                                         config.patch_size,
                                         config.patch_estimator);
      bool moved = false;
      // Check 8 neighbors at step size of half_patch
      const int step = std::max(1, half_patch);
      float best_val = current_val;
      int best_x = x, best_y = y;
      for (int dy = -step; dy <= step; dy += step) {
        for (int dx = -step; dx <= step; dx += step) {
          if (dx == 0 && dy == 0) continue;
          int nx = x + dx, ny = y + dy;
          if (!is_valid_point(nx, ny)) continue;
          float nv = patch_estimate(image_downsampled, nx, ny,
                                    config.patch_size,
                                    config.patch_estimator);
          if (nv < best_val) {
            best_val = nv;
            best_x = nx;
            best_y = ny;
          }
        }
      }
      if (best_x != x || best_y != y) {
        x = best_x;
        y = best_y;
        moved = true;
      }
      if (!moved) break;
    }
  };

  std::vector<SamplePoint> candidates;

  // Determine number of points
  int target_points = config.num_sample_points;
  if (target_points <= 0) {
    // Density-based: ~1 sample per 800 downsampled pixels, clamped to [200, 3000].
    // This ensures sufficient coverage for RBF fitting across typical image sizes.
    target_points = std::clamp((rows * cols) / 800, 200, 3000);
  }

  const int margin = config.border_margin;
  const int usable_w = cols - 2 * margin;
  const int usable_h = rows - 2 * margin;
  if (usable_w < 4 || usable_h < 4) return {};

  auto add_candidate = [&](int x, int y) {
    x = std::clamp(x, margin, cols - margin - 1);
    y = std::clamp(y, margin, rows - margin - 1);
    if (!is_valid_point(x, y))
      return;
    gradient_descent_to_dim(x, y);
    if (is_valid_point(x, y))
      candidates.push_back({x, y});
  };

  // Explicit corners and edge anchors stabilize large-scale planes.
  add_candidate(margin, margin);
  add_candidate(cols - margin - 1, margin);
  add_candidate(margin, rows - margin - 1);
  add_candidate(cols - margin - 1, rows - margin - 1);
  const int edge_steps = std::max(2, static_cast<int>(std::sqrt(target_points)));
  for (int i = 1; i < edge_steps; ++i) {
    const int x = margin + i * usable_w / edge_steps;
    const int y = margin + i * usable_h / edge_steps;
    add_candidate(x, margin);
    add_candidate(x, rows - margin - 1);
    add_candidate(margin, y);
    add_candidate(cols - margin - 1, y);
  }

  // Generate grid points
  const int grid_nx = static_cast<int>(std::sqrt(static_cast<float>(target_points)
      * usable_w / usable_h));
  const int grid_ny = static_cast<int>(std::sqrt(static_cast<float>(target_points)
      * usable_h / usable_w));
  const int actual_nx = std::max(2, grid_nx);
  const int actual_ny = std::max(2, grid_ny);

  for (int gy = 0; gy < actual_ny; ++gy) {
    for (int gx = 0; gx < actual_nx; ++gx) {
      int x = margin + static_cast<int>(static_cast<float>(gx + 0.5f) * usable_w / actual_nx);
      int y = margin + static_cast<int>(static_cast<float>(gy + 0.5f) * usable_h / actual_ny);
      if (!is_valid_point(x, y)) {
        // Try to find nearest valid point
        bool found = false;
        for (int radius = 1; radius < std::max(rows, cols) && !found; ++radius) {
          for (int dy = -radius; dy <= radius && !found; ++dy) {
            for (int dx = -radius; dx <= radius && !found; ++dx) {
              if (std::abs(dx) != radius && std::abs(dy) != radius) continue;
              int nx = x + dx, ny = y + dy;
              if (is_valid_point(nx, ny)) {
                x = nx; y = ny; found = true;
              }
            }
          }
        }
        if (!found) continue;
      }
      // Apply gradient descent to find dim spot
      gradient_descent_to_dim(x, y);
      candidates.push_back({x, y});
    }
  }

  // Remove duplicate points (gradient descent may converge multiple to same spot)
  std::sort(candidates.begin(), candidates.end(), [](const SamplePoint& a, const SamplePoint& b) {
    return a.y * 100000 + a.x < b.y * 100000 + b.x;
  });
  candidates.erase(std::unique(candidates.begin(), candidates.end(), [](const SamplePoint& a, const SamplePoint& b) {
    return a.x == b.x && a.y == b.y;
  }), candidates.end());

  if (!random_downselection) {
    return candidates;
  }

  std::array<std::vector<SamplePoint>, 4> quartiles;
  for (const auto& p : candidates) {
    const int qx = p.x < cols / 2 ? 0 : 1;
    const int qy = p.y < rows / 2 ? 0 : 1;
    quartiles[static_cast<size_t>(qy * 2 + qx)].push_back(p);
  }

  std::vector<SamplePoint> points;
  points.reserve(static_cast<size_t>(target_points));
  for (auto& q : quartiles) {
    std::shuffle(q.begin(), q.end(), local_rng);
    const int quota = std::max(1, target_points / 4);
    for (int i = 0; i < std::min<int>(quota, q.size()); ++i)
      points.push_back(q[static_cast<size_t>(i)]);
  }
  if (static_cast<int>(points.size()) < target_points) {
    std::shuffle(candidates.begin(), candidates.end(), local_rng);
    for (const auto& p : candidates) {
      const bool exists = std::any_of(points.begin(), points.end(),
                                      [&](const SamplePoint& e) {
                                        return e.x == p.x && e.y == p.y;
                                      });
      if (!exists)
        points.push_back(p);
      if (static_cast<int>(points.size()) >= target_points)
        break;
    }
  }

  return points;
}

Matrix2Df fit_polynomial_autobge(
    const Matrix2Df& image_downsampled,
    const std::vector<SamplePoint>& points,
    const BGEConfig::AutoBGEConfig& config,
    int target_rows, int target_cols) {

  const int rows = static_cast<int>(image_downsampled.rows());
  const int cols = static_cast<int>(image_downsampled.cols());
  if (points.empty() || rows < 2 || cols < 2) {
    return Matrix2Df::Zero(target_rows, target_cols);
  }

  const int degree = config.poly_degree;
  // Count polynomial terms: for 2D polynomial of degree d:
  // number of terms = (d+1)*(d+2)/2
  const int n_terms = autobge_polynomial_term_count(degree);
  const int n_points = static_cast<int>(points.size());

  if (n_points < n_terms + 4) {
    return Matrix2Df::Zero(target_rows, target_cols);
  }

  // Build design matrix and observation vector
  Eigen::MatrixXf A(n_points, n_terms);
  Eigen::VectorXf b(n_points);

  // Normalize coordinates to [-1, 1]
  const float norm_x = 2.0f / (cols - 1);
  const float norm_y = 2.0f / (rows - 1);

  for (int i = 0; i < n_points; ++i) {
    const float x = static_cast<float>(points[i].x) * norm_x - 1.0f;
    const float y = static_cast<float>(points[i].y) * norm_y - 1.0f;
    const float val = patch_estimate(image_downsampled, points[i].x, points[i].y,
                                     config.patch_size,
                                     config.patch_estimator);
    b(i) = val;
    int col = 0;
    for (int dy = 0; dy <= degree; ++dy) {
      for (int dx = 0; dx <= degree - dy; ++dx) {
        A(i, col) = std::pow(x, dx) * std::pow(y, dy);
        ++col;
      }
    }
  }

  // Solve least squares
  Eigen::VectorXf coeffs = A.colPivHouseholderQr().solve(b);

  // Render at downsampled resolution, then upscale
  Matrix2Df bg_down(rows, cols);
  for (int r = 0; r < rows; ++r) {
    const float y = static_cast<float>(r) * norm_y - 1.0f;
    for (int c = 0; c < cols; ++c) {
      const float x = static_cast<float>(c) * norm_x - 1.0f;
      float val = 0.0f;
      int idx = 0;
      for (int dy = 0; dy <= degree; ++dy) {
        for (int dx = 0; dx <= degree - dy; ++dx) {
          val += coeffs(idx) * std::pow(x, dx) * std::pow(y, dy);
          ++idx;
        }
      }
      bg_down(r, c) = val;
    }
  }

  return upscale_lanczos4(bg_down, target_rows, target_cols);
}

Matrix2Df fit_rbf_autobge(
    const Matrix2Df& image_downsampled,
    const std::vector<SamplePoint>& points,
    const BGEConfig::AutoBGEConfig& config,
    int target_rows, int target_cols) {

  const int rows = static_cast<int>(image_downsampled.rows());
  const int cols = static_cast<int>(image_downsampled.cols());
  const int n = static_cast<int>(points.size());
  if (n < kAutoBGEMinStage2Points || rows < 2 || cols < 2) {
    return Matrix2Df::Zero(target_rows, target_cols);
  }

  // Compute sample values at each point
  Eigen::VectorXf f(n);
  for (int i = 0; i < n; ++i) {
    f(i) = patch_estimate(image_downsampled, points[i].x, points[i].y,
                          config.patch_size, config.patch_estimator);
  }

  constexpr float epsilon = 1.0f;
  const float smooth = config.rbf_smooth;

  auto norm_x = [&](int x) -> float {
    return cols > 1 ? static_cast<float>(x) / static_cast<float>(cols - 1) : 0.0f;
  };
  auto norm_y = [&](int y) -> float {
    return rows > 1 ? static_cast<float>(y) / static_cast<float>(rows - 1) : 0.0f;
  };
  auto multiquadric_kernel = [](float r, float eps) -> float {
    return std::sqrt((r / eps) * (r / eps) + 1.0f);
  };

  Eigen::MatrixXf W = Eigen::MatrixXf::Zero(n + 3, n + 3);
  Eigen::VectorXf rhs = Eigen::VectorXf::Zero(n + 3);
  for (int i = 0; i < n; ++i) {
    rhs(i) = f(i);
    for (int j = 0; j < n; ++j) {
      const float dx = norm_x(points[i].x) - norm_x(points[j].x);
      const float dy = norm_y(points[i].y) - norm_y(points[j].y);
      const float r = std::sqrt(dx * dx + dy * dy);
      W(i, j) = multiquadric_kernel(r, epsilon);
    }
    // Smoothing diagonal
    W(i, i) += smooth;
    // Polynomial terms (linear: 1, x, y) for TPS with affine component
    W(i, n) = 1.0f;
    W(i, n + 1) = norm_x(points[i].x);
    W(i, n + 2) = norm_y(points[i].y);
    W(n, i) = 1.0f;
    W(n + 1, i) = norm_x(points[i].x);
    W(n + 2, i) = norm_y(points[i].y);
  }

  Eigen::VectorXf weights = W.colPivHouseholderQr().solve(rhs);

  // Render at downsampled resolution
  Matrix2Df bg_down(rows, cols);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const float x = norm_x(c);
      const float y = norm_y(r);
      float val = weights(n) + weights(n + 1) * x + weights(n + 2) * y;
      for (int i = 0; i < n; ++i) {
        const float dx = x - norm_x(points[i].x);
        const float dy = y - norm_y(points[i].y);
        const float dist = std::sqrt(dx * dx + dy * dy);
        val += weights(i) * multiquadric_kernel(dist, epsilon);
      }
      bg_down(r, c) = val;
    }
  }

  return upscale_lanczos4(bg_down, target_rows, target_cols);
}

AutoBGEResult build_autobge_models(
    const Matrix2Df& R, const Matrix2Df& G, const Matrix2Df& B,
    const BGEConfig& config) {

  AutoBGEResult result;
  const int rows = static_cast<int>(R.rows());
  const int cols = static_cast<int>(R.cols());

  // Detect mono input: all three channels identical
  const bool mono = (R.array() == G.array()).all() && (G.array() == B.array()).all();
  result.mono_input = mono;

  // Determine which channels to process
  std::array<const Matrix2Df*, 3> channels = {&R, &G, &B};
  std::array<std::string, 3> names = {"R", "G", "B"};
  int n_channels = mono ? 1 : 3;

  // Determine if we should process mono
  if (mono && config.autobge.mono_mode == "disabled") {
    result.success = false;
    return result;
  }

  const std::vector<uint8_t>* valid_mask = nullptr;
  if (config.common_valid_mask.size() == static_cast<size_t>(rows * cols) &&
      config.common_mask_rows == rows && config.common_mask_cols == cols) {
    valid_mask = &config.common_valid_mask;
  }
  if (!valid_mask) {
    result.success = false;
    return result;
  }

  std::vector<uint8_t> sampling_mask = *valid_mask;
  Matrix2Df luma(rows, cols);
  std::vector<float> valid_luma;
  valid_luma.reserve(static_cast<size_t>(rows * cols));
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      const size_t idx = static_cast<size_t>(y * cols + x);
      const float rv = R(y, x);
      const float gv = G(y, x);
      const float bv = B(y, x);
      const float lv = 0.2126f * rv + 0.7152f * gv + 0.0722f * bv;
      luma(y, x) = lv;
      if ((*valid_mask)[idx] != 0 && std::isfinite(lv) && lv > 0.0f)
        valid_luma.push_back(lv);
    }
  }
  if (valid_luma.size() >= 16) {
    std::sort(valid_luma.begin(), valid_luma.end());
    const float fg_thresh =
        sorted_quantile_local(valid_luma, config.structure_thresh_percentile);
    for (int y = 0; y < rows; ++y) {
      for (int x = 0; x < cols; ++x) {
        const size_t idx = static_cast<size_t>(y * cols + x);
        if ((*valid_mask)[idx] == 0 || !std::isfinite(luma(y, x)) ||
            luma(y, x) >= fg_thresh) {
          sampling_mask[idx] = 0;
        }
      }
    }
  }

  const int scale = config.autobge.downsample_scale;
  std::mt19937 rng(static_cast<uint32_t>(config.autobge.random_seed));

  for (int ch = 0; ch < n_channels; ++ch) {
    const Matrix2Df& channel = *channels[ch];

    // Step 1: Transform to working space
    StretchParams stretch_params;
    Matrix2Df stretched = transform_to_autobge_working_space(
        channel, config.autobge, &stretch_params, ch, valid_mask);

    // Step 2: Downsample
    Matrix2Df downsampled = downsample_area(stretched, scale);

    // Downsample valid mask if present
    std::vector<uint8_t> mask_down;
    const std::vector<uint8_t>* mask_down_ptr = nullptr;
    if (valid_mask) {
      const int ds_rows = static_cast<int>(downsampled.rows());
      const int ds_cols = static_cast<int>(downsampled.cols());
      mask_down = downsample_mask_majority(sampling_mask, rows, cols, scale,
                                           ds_rows, ds_cols);
      mask_down_ptr = &mask_down;
    }

    auto generate_points_with_retry = [&](const Matrix2Df& image,
                                          int min_points,
                                          const char* stage_name) {
      std::vector<SamplePoint> generated = generate_autobge_sample_points(
          image, config.autobge, mask_down_ptr, &rng);
      if (static_cast<int>(generated.size()) >= min_points)
        return generated;

      BGEConfig::AutoBGEConfig relaxed = config.autobge;
      relaxed.bright_exclusion_fraction =
          std::min(0.98f,
                   std::max(relaxed.bright_exclusion_fraction + 0.20f, 0.90f));
      std::vector<SamplePoint> retry = generate_autobge_sample_points(
          image, relaxed, mask_down_ptr, &rng, false);
      std::cerr << "[AutoBGE] Channel " << names[ch] << " " << stage_name
                << ": relaxed sparse sampling retry " << generated.size()
                << " -> " << retry.size() << " points" << std::endl;
      return retry;
    };

    // Step 3: Generate sample points
    const int min_stage1_points = autobge_min_stage1_points(config.autobge);
    std::vector<SamplePoint> points = generate_points_with_retry(
        downsampled, min_stage1_points, "stage1");

    if (static_cast<int>(points.size()) < min_stage1_points) {
      std::cerr << "[AutoBGE] Channel " << names[ch]
                << ": insufficient stage1 sample points (" << points.size()
                << ", required " << min_stage1_points << ")" << std::endl;
      result.channel_models[ch].success = false;
      result.channel_models[ch].error_message = "insufficient_sample_points";
      continue;
    }

    // Step 4: Two-stage fitting — polynomial first, then RBF on residuals
    Matrix2Df poly_bg = fit_polynomial_autobge(
        downsampled, points, config.autobge, rows, cols);

    // Compute residual after polynomial subtraction
    Matrix2Df poly_down = downsample_area(poly_bg, scale);
    Matrix2Df residual_down = downsampled - poly_down;
    if (config.autobge.normalize_between_stages) {
      std::vector<float> source_vals;
      std::vector<float> residual_vals;
      source_vals.reserve(points.size());
      residual_vals.reserve(points.size());
      for (const auto& p : points) {
        source_vals.push_back(patch_estimate(downsampled, p.x, p.y,
                                             config.autobge.patch_size,
                                             config.autobge.patch_estimator));
        residual_vals.push_back(patch_estimate(residual_down, p.x, p.y,
                                               config.autobge.patch_size,
                                               config.autobge.patch_estimator));
      }
      const float source_med = median_from_values(std::move(source_vals));
      const float residual_med = median_from_values(std::move(residual_vals));
      residual_down.array() += (source_med - residual_med);
    }

    // Generate sample points on the residual image
    std::vector<SamplePoint> residual_points = generate_points_with_retry(
        residual_down, kAutoBGEMinStage2Points, "stage2");

    Matrix2Df rbf_bg;
    if (static_cast<int>(residual_points.size()) < kAutoBGEMinStage2Points) {
      std::cerr << "[AutoBGE] Channel " << names[ch]
                << ": insufficient stage2 sample points ("
                << residual_points.size() << ", required "
                << kAutoBGEMinStage2Points << ")" << std::endl;
      result.channel_models[ch].success = false;
      result.channel_models[ch].error_message =
          "insufficient_residual_sample_points";
      continue;
    }
    rbf_bg = fit_rbf_autobge(
        residual_down, residual_points, config.autobge, rows, cols);

    // Combine: total background = polynomial + RBF residual
    Matrix2Df total_bg = poly_bg + rbf_bg;

    // Step 5: derive original-domain model from corrected image. This avoids
    // treating nonlinear working-space deltas as directly additive backgrounds.
    Matrix2Df corrected_working = stretched - total_bg;
    Matrix2Df corrected_original = transform_from_autobge_working_space(
        corrected_working, stretch_params, ch);
    Matrix2Df bg_original = channel - corrected_original;

    // Fill model
    result.channel_models[ch].model = bg_original;
    result.channel_models[ch].success = true;
    result.channel_models[ch].n_valid_cells = static_cast<int>(points.size());
    result.channel_models[ch].grid_cells.clear();
    result.channel_models[ch].grid_cells.reserve(points.size());
    std::vector<float> residual_values;
    residual_values.reserve(points.size());
    for (const auto& p : points) {
      GridCell gc;
      gc.center_x = static_cast<float>(p.x * scale);
      gc.center_y = static_cast<float>(p.y * scale);
      gc.bg_value = patch_estimate(downsampled, p.x, p.y,
                                   config.autobge.patch_size,
                                   config.autobge.patch_estimator);
      gc.weight = 1.0f;
      gc.n_samples = 1;
      gc.valid = true;
      result.channel_models[ch].grid_cells.push_back(gc);
      const int yy = std::clamp(p.y * scale, 0, rows - 1);
      const int xx = std::clamp(p.x * scale, 0, cols - 1);
      residual_values.push_back(gc.bg_value - total_bg(yy, xx));
    }
    result.channel_models[ch].rms_residual = 0.0f;
    if (!residual_values.empty()) {
      double sum_sq = 0.0;
      for (float v : residual_values)
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
      result.channel_models[ch].rms_residual =
          static_cast<float>(std::sqrt(sum_sq / residual_values.size()));
    }

    // Build channel diagnostics
    BGEChannelDiagnostics ch_diag;
    ch_diag.channel_name = names[ch];
    ch_diag.applied = false;
    ch_diag.fit_success = true;
    ch_diag.fit_rms_residual = result.channel_models[ch].rms_residual;
    ch_diag.tile_samples_total = static_cast<int>(points.size());
    ch_diag.tile_samples_valid = static_cast<int>(points.size());
    ch_diag.grid_cells = result.channel_models[ch].grid_cells;
    ch_diag.grid_cells_valid = static_cast<int>(ch_diag.grid_cells.size());
    for (const auto& gc : ch_diag.grid_cells) {
      ch_diag.sample_bg_values.push_back(gc.bg_value);
      ch_diag.sample_weight_values.push_back(gc.weight);
    }
    ch_diag.sample_bg_stats = stats_from_values_local(ch_diag.sample_bg_values);
    ch_diag.sample_weight_stats = stats_from_values_local(ch_diag.sample_weight_values);
    ch_diag.residual_values = residual_values;
    ch_diag.residual_stats = stats_from_values_local(ch_diag.residual_values);
    result.channel_diagnostics.push_back(ch_diag);
  }

  // If mono with rgb_duplicate, copy the R model to G and B
  if (mono && config.autobge.mono_mode == "rgb_duplicate") {
    result.channel_models[1] = result.channel_models[0];
    result.channel_models[2] = result.channel_models[0];
    if (!result.channel_diagnostics.empty()) {
      BGEChannelDiagnostics g_diag = result.channel_diagnostics[0];
      g_diag.channel_name = "G";
      BGEChannelDiagnostics b_diag = result.channel_diagnostics[0];
      b_diag.channel_name = "B";
      result.channel_diagnostics.push_back(g_diag);
      result.channel_diagnostics.push_back(b_diag);
    }
  }

  // Check if at least one channel succeeded
  bool any_success = false;
  for (int ch = 0; ch < 3; ++ch) {
    if (result.channel_models[ch].success) {
      any_success = true;
      break;
    }
  }
  result.success = any_success;
  return result;
}

bool finalize_bge_from_channel_models(
    Matrix2Df& R, Matrix2Df& G, Matrix2Df& B,
    const std::array<BackgroundModel, 3>& channel_models,
    const std::vector<BGEChannelDiagnostics>& channel_diagnostics,
    const BGEConfig& config,
    BGEDiagnostics* diagnostics) {

  const int rows = static_cast<int>(R.rows());
  const int cols = static_cast<int>(R.cols());
  const bool have_canvas_mask =
      config.common_valid_mask.size() == static_cast<size_t>(rows * cols) &&
      config.common_mask_rows == rows && config.common_mask_cols == cols;
  if (!have_canvas_mask) {
    if (diagnostics) {
      diagnostics->attempted = true;
      diagnostics->success = false;
      diagnostics->bge_method = "autobge";
      diagnostics->method = "autobge";
      diagnostics->failure_reason = "missing_canvas_mask";
      diagnostics->channels = channel_diagnostics;
    }
    return false;
  }

  std::array<const Matrix2Df*, 3> inputs = {&R, &G, &B};
  std::array<std::string, 3> names = {"R", "G", "B"};
  std::array<Matrix2Df, 3> corrected;
  std::vector<BGEChannelDiagnostics> out_diags = channel_diagnostics;
  if (out_diags.size() < 3)
    out_diags.resize(3);

  for (int ch = 0; ch < 3; ++ch) {
    if (out_diags[static_cast<size_t>(ch)].channel_name.empty())
      out_diags[static_cast<size_t>(ch)].channel_name = names[ch];
    if (!channel_models[ch].success) {
      std::cerr << "[AutoBGE] Channel " << names[ch]
                << ": model not available, rejecting RGB apply" << std::endl;
      if (diagnostics) {
        diagnostics->attempted = true;
        diagnostics->success = false;
        diagnostics->bge_method = "autobge";
        diagnostics->method = "autobge";
        diagnostics->failure_reason = "partial_channel_model";
        diagnostics->channels = std::move(out_diags);
      }
      return false;
    }

    const Matrix2Df& bg = channel_models[ch].model;
    if (bg.rows() != rows || bg.cols() != cols) {
      std::cerr << "[AutoBGE] Channel " << names[ch]
                << ": model dimension mismatch (" << bg.rows() << "x" << bg.cols()
                << " vs " << rows << "x" << cols << ")" << std::endl;
      if (diagnostics) {
        diagnostics->attempted = true;
        diagnostics->success = false;
        diagnostics->bge_method = "autobge";
        diagnostics->method = "autobge";
        diagnostics->failure_reason = "model_dimension_mismatch";
        diagnostics->channels = std::move(out_diags);
      }
      return false;
    }

    auto& ch_diag = out_diags[static_cast<size_t>(ch)];
    const Matrix2Df& img = *inputs[ch];
    ch_diag.input_stats = stats_from_matrix_local(img);
    ch_diag.model_stats = stats_from_matrix_local(bg);
    const float pedestal = ch_diag.model_stats.median;
    corrected[ch] = img;
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        const size_t idx = static_cast<size_t>(r * cols + c);
        if (config.common_valid_mask[idx] == 0) {
          corrected[ch](r, c) = 0.0f;
          continue;
        }
        const float vin = img(r, c);
        const float mv = bg(r, c);
        if (!std::isfinite(vin) || !std::isfinite(mv)) {
          corrected[ch](r, c) = std::numeric_limits<float>::quiet_NaN();
          continue;
        }
        corrected[ch](r, c) = std::max(0.0f, vin - mv + pedestal);
      }
    }

    if (config.autobge.apply_guards) {
      const float flat_pre =
          spatial_background_spread(img, &config.common_valid_mask);
      const float flat_post =
          spatial_background_spread(corrected[ch], &config.common_valid_mask);
      const float slope_pre =
          coarse_background_plane_slope(img, &config.common_valid_mask);
      const float slope_post =
          coarse_background_plane_slope(corrected[ch], &config.common_valid_mask);
      ch_diag.guard_flat_pre = flat_pre;
      ch_diag.guard_flat_post = flat_post;
      ch_diag.guard_slope_pre = slope_pre;
      ch_diag.guard_slope_post = slope_post;
      const float max_flatness_worsen_factor =
          config.internal_relaxed_channel_guards ? 1.35f : 1.15f;
      const float max_slope_worsen_factor =
          config.internal_relaxed_channel_guards ? 1.15f : 1.08f;
      if (std::isfinite(flat_pre) && std::isfinite(flat_post) &&
          flat_post > flat_pre * max_flatness_worsen_factor) {
        ch_diag.guard_rejected = true;
        ch_diag.guard_reason = "flatness_worsened";
      }
      if (!ch_diag.guard_rejected && std::isfinite(slope_pre) &&
          std::isfinite(slope_post) &&
          slope_post > slope_pre * max_slope_worsen_factor) {
        ch_diag.guard_rejected = true;
        ch_diag.guard_reason = "slope_worsened";
      }
      if (ch_diag.guard_rejected) {
        std::cerr << "[AutoBGE] Channel " << names[ch]
                  << " guard rejected RGB apply: " << ch_diag.guard_reason
                  << std::endl;
        if (diagnostics) {
          diagnostics->attempted = true;
          diagnostics->success = false;
          diagnostics->bge_method = "autobge";
          diagnostics->method = "autobge";
          diagnostics->failure_reason = ch_diag.guard_reason;
          diagnostics->channels = std::move(out_diags);
        }
        return false;
      }
    }

    ch_diag.fit_success = true;
    ch_diag.fit_rms_residual = channel_models[ch].rms_residual;
    ch_diag.output_stats = stats_from_matrix_local(corrected[ch]);
    ch_diag.mean_shift = ch_diag.output_stats.mean - ch_diag.input_stats.mean;
    ch_diag.applied = true;
  }

  if (config.autobge.apply_guards) {
    const std::vector<uint8_t> bg_mask =
        build_chroma_background_mask_from_rgb(R, G, B, &config.common_valid_mask);
    const float pre_rg_std = log_chroma_std_background(R, G, bg_mask);
    const float pre_bg_std = log_chroma_std_background(B, G, bg_mask);
    const float post_rg_std =
        log_chroma_std_background(corrected[0], corrected[1], bg_mask);
    const float post_bg_std =
        log_chroma_std_background(corrected[2], corrected[1], bg_mask);
    constexpr float kMaxChromaStdWorsenFactor = 1.08f;
    if ((std::isfinite(pre_rg_std) && std::isfinite(post_rg_std) &&
         post_rg_std > pre_rg_std * kMaxChromaStdWorsenFactor) ||
        (std::isfinite(pre_bg_std) && std::isfinite(post_bg_std) &&
         post_bg_std > pre_bg_std * kMaxChromaStdWorsenFactor)) {
      if (diagnostics) {
        diagnostics->attempted = true;
        diagnostics->success = false;
        diagnostics->bge_method = "autobge";
        diagnostics->method = "autobge";
        diagnostics->failure_reason = "background_chroma_worsened";
        diagnostics->channels = std::move(out_diags);
      }
      return false;
    }
  }

  R = std::move(corrected[0]);
  G = std::move(corrected[1]);
  B = std::move(corrected[2]);
  enforce_canvas_mask_on_rgb(R, G, B, config.common_valid_mask);

  if (diagnostics) {
    diagnostics->attempted = true;
    diagnostics->success = true;
    diagnostics->bge_method = "autobge";
    diagnostics->method = "autobge";
    diagnostics->channels = std::move(out_diags);
  }

  return true;
}

} // namespace tile_compile::image
