#include "tile_compile/image/background_extraction.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <queue>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace tile_compile::image {

namespace {

constexpr float kTiny = 1.0e-12f;
constexpr float kMinUsableTileFraction = 0.10f;

using SteadyClock = std::chrono::steady_clock;

/// @brief Implements elapsed seconds since.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
double elapsed_seconds_since(const SteadyClock::time_point &start) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(
             SteadyClock::now() - start)
      .count();
}

/// @brief Implements bge parallel worker count.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
int bge_parallel_worker_count(int work_items, int min_items_per_worker) {
  if (work_items <= 0) {
    return 1;
  }
#if defined(_OPENMP)
  if (omp_in_parallel()) {
    return 1;
  }
  const int max_threads = std::max(1, omp_get_max_threads());
  if (max_threads <= 1) {
    return 1;
  }
  const int min_items = std::max(1, min_items_per_worker);
  const int wanted = std::max(1, (work_items + min_items - 1) / min_items);
  return std::max(1, std::min(max_threads, wanted));
#else
  (void)min_items_per_worker;
  return 1;
#endif
}

/// @brief Implements clamp01.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float clamp01(float v) { return std::max(0.0f, std::min(1.0f, v)); }

/// @brief Implements robust median inplace.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float robust_median_inplace(std::vector<float> &values) {
  if (values.empty())
    return 0.0f;
  const size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + mid, values.end());
  float med = values[mid];
  if ((values.size() & 1U) == 0U) {
    std::nth_element(values.begin(), values.begin() + (mid - 1), values.end());
    med = 0.5f * (med + values[mid - 1]);
  }
  return med;
}

/// @brief Implements robust median.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float robust_median(std::vector<float> values) {
  return robust_median_inplace(values);
}

/// @brief Implements robust quantile inplace.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float robust_quantile_inplace(std::vector<float> &values, float q) {
  if (values.empty())
    return 0.0f;
  q = clamp01(q);
  const size_t idx =
      static_cast<size_t>(q * static_cast<float>(values.size() - 1));
  std::nth_element(values.begin(), values.begin() + idx, values.end());
  return values[idx];
}

/// @brief Implements robust quantile.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float robust_quantile(std::vector<float> values, float q) {
  return robust_quantile_inplace(values, q);
}

/// @brief Implements infer bge failure reason.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::string infer_bge_failure_reason(const BGEDiagnostics &diag) {
  if (diag.success) {
    return "";
  }
  if (diag.safety_fallback_triggered && !diag.safety_fallback_reason.empty()) {
    return diag.safety_fallback_reason;
  }
  if (diag.channels.empty()) {
    return diag.attempted ? "no_channels_processed" : "not_attempted";
  }

  int fit_success_count = 0;
  int guard_rejected_count = 0;
  bool all_slope_rejected = true;
  bool all_flatness_rejected = true;

  for (const auto &ch : diag.channels) {
    if (ch.fit_success) {
      ++fit_success_count;
    }
    if (ch.guard_rejected) {
      ++guard_rejected_count;
    }
    all_slope_rejected =
        all_slope_rejected && ch.guard_rejected &&
        ch.guard_reason == "slope_worsened";
    all_flatness_rejected =
        all_flatness_rejected && ch.guard_rejected &&
        ch.guard_reason == "flatness_worsened";
  }

  if (guard_rejected_count == static_cast<int>(diag.channels.size())) {
    if (all_slope_rejected) {
      return "all_channels_guard_rejected_slope";
    }
    if (all_flatness_rejected) {
      return "all_channels_guard_rejected_flatness";
    }
    return "all_channels_guard_rejected";
  }
  if (fit_success_count > 0) {
    return "no_channel_applied";
  }
  return "surface_fit_failed";
}

/// @brief Implements sorted quantile.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float sorted_quantile(const std::vector<float> &sorted_values, float q) {
  if (sorted_values.empty())
    return 0.0f;
  q = clamp01(q);
  const size_t idx =
      static_cast<size_t>(q * static_cast<float>(sorted_values.size() - 1));
  return sorted_values[idx];
}

/// @brief Implements robust mad.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float robust_mad(const std::vector<float> &values, float center) {
  if (values.empty())
    return 0.0f;
  std::vector<float> abs_dev;
  abs_dev.reserve(values.size());
  for (float v : values)
    abs_dev.push_back(std::abs(v - center));
  return robust_median_inplace(abs_dev);
}

float robust_mean(const std::vector<float> &values) {
  if (values.empty())
    return 0.0f;
  double sum = 0.0;
  for (float v : values)
    sum += static_cast<double>(v);
  return static_cast<float>(sum / static_cast<double>(values.size()));
}

std::vector<float> sigma_clipped_values(std::vector<float> values,
                                        float sigma = 3.0f,
                                        int max_iters = 5) {
  values.erase(std::remove_if(values.begin(), values.end(),
                              [](float v) { return !std::isfinite(v); }),
               values.end());
  for (int iter = 0; iter < max_iters && values.size() >= 8; ++iter) {
    std::vector<float> work = values;
    const float center = robust_median_inplace(work);
    const float scale = 1.4826f * robust_mad(values, center);
    if (!(std::isfinite(scale) && scale > kTiny))
      break;
    const float limit = sigma * scale;
    std::vector<float> kept;
    kept.reserve(values.size());
    for (float v : values) {
      if (std::abs(v - center) <= limit)
        kept.push_back(v);
    }
    if (kept.size() == values.size() || kept.size() < 8)
      break;
    values.swap(kept);
  }
  return values;
}

float biweight_location(std::vector<float> values) {
  values.erase(std::remove_if(values.begin(), values.end(),
                              [](float v) { return !std::isfinite(v); }),
               values.end());
  if (values.empty())
    return std::numeric_limits<float>::quiet_NaN();
  std::vector<float> work = values;
  const float m = robust_median_inplace(work);
  const float mad = robust_mad(values, m);
  const float scale = 9.0f * std::max(mad, kTiny);
  double num = 0.0;
  double den = 0.0;
  for (float v : values) {
    const float u = (v - m) / scale;
    if (std::abs(u) >= 1.0f)
      continue;
    const float one_minus_u2 = 1.0f - u * u;
    const float w = one_minus_u2 * one_minus_u2;
    num += static_cast<double>(v) * static_cast<double>(w);
    den += static_cast<double>(w);
  }
  if (den <= 0.0)
    return m;
  return static_cast<float>(num / den);
}

float estimate_tile_background_value(std::vector<float> pixels,
                                     const BGEConfig &config) {
  pixels.erase(std::remove_if(pixels.begin(), pixels.end(),
                              [](float v) { return !std::isfinite(v); }),
               pixels.end());
  if (pixels.empty())
    return std::numeric_limits<float>::quiet_NaN();

  if (config.sample_estimator == "sigma_clipped_median") {
    auto clipped = sigma_clipped_values(std::move(pixels));
    if (clipped.empty())
      return std::numeric_limits<float>::quiet_NaN();
    return robust_median_inplace(clipped);
  }

  if (config.sample_estimator == "sextractor_mode") {
    auto clipped = sigma_clipped_values(std::move(pixels));
    if (clipped.empty())
      return std::numeric_limits<float>::quiet_NaN();
    std::vector<float> med_work = clipped;
    const float median = robust_median_inplace(med_work);
    const float mean = robust_mean(clipped);
    const float mode = 2.5f * median - 1.5f * mean;
    const float denom = std::max(1.0f, std::abs(median));
    if (!std::isfinite(mode) || std::abs(mode - median) > 0.30f * denom)
      return median;
    return mode;
  }

  if (config.sample_estimator == "biweight") {
    return biweight_location(std::move(pixels));
  }

  return robust_quantile_inplace(pixels, config.sample_quantile);
}

float estimate_tile_background_from_sorted(const std::vector<float> &sorted_pixels,
                                           const BGEConfig &config) {
  if (sorted_pixels.empty())
    return std::numeric_limits<float>::quiet_NaN();
  if (config.sample_estimator == "quantile")
    return sorted_quantile(sorted_pixels, config.sample_quantile);
  return estimate_tile_background_value(sorted_pixels, config);
}

/// @brief Implements box blur subregion.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void box_blur_subregion(const Matrix2Df &image, int x0, int y0, int w, int h,
                        int radius, std::vector<double> *integral,
                        std::vector<float> *out) {
  out->assign(static_cast<size_t>(std::max(0, h * w)), 0.0f);
  if (h <= 0 || w <= 0)
    return;
  radius = std::max(0, radius);

  integral->assign(static_cast<size_t>((h + 1) * (w + 1)), 0.0);
  auto idx = [w](int y, int x) { return static_cast<size_t>(y * (w + 1) + x); };
  const int stride = static_cast<int>(image.cols());
  const float *base = image.data();
  for (int y = 0; y < h; ++y) {
    double row_sum = 0.0;
    const float *src =
        base + static_cast<size_t>(y0 + y) * static_cast<size_t>(stride) + x0;
    for (int x = 0; x < w; ++x) {
      float v = src[x];
      if (!std::isfinite(v))
        v = 0.0f;
      row_sum += static_cast<double>(v);
      (*integral)[idx(y + 1, x + 1)] = (*integral)[idx(y, x + 1)] + row_sum;
    }
  }

  for (int y = 0; y < h; ++y) {
    const int y0 = std::max(0, y - radius);
    const int y1 = std::min(h - 1, y + radius);
    for (int x = 0; x < w; ++x) {
      const int x0 = std::max(0, x - radius);
      const int x1 = std::min(w - 1, x + radius);
      const double sum = (*integral)[idx(y1 + 1, x1 + 1)] -
                         (*integral)[idx(y0, x1 + 1)] -
                         (*integral)[idx(y1 + 1, x0)] +
                         (*integral)[idx(y0, x0)];
      const int area = (y1 - y0 + 1) * (x1 - x0 + 1);
      (*out)[static_cast<size_t>(y * w + x)] =
          static_cast<float>(sum / std::max(1, area));
    }
  }
}

/// @brief Implements dilate mask in place.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void dilate_mask_in_place(std::vector<uint8_t> *mask, int w, int h, int radius,
                          std::vector<uint8_t> *scratch) {
  if (radius <= 0)
    return;
  scratch->assign(mask->size(), 0);
  const int r2 = radius * radius;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      uint8_t hit = 0;
      for (int dy = -radius; dy <= radius && !hit; ++dy) {
        const int yy = y + dy;
        if (yy < 0 || yy >= h)
          continue;
        for (int dx = -radius; dx <= radius; ++dx) {
          const int xx = x + dx;
          if (xx < 0 || xx >= w)
            continue;
          if (dx * dx + dy * dy > r2)
            continue;
          if ((*mask)[static_cast<size_t>(yy * w + xx)] != 0) {
            hit = 1;
            break;
          }
        }
      }
      (*scratch)[static_cast<size_t>(y * w + x)] = hit;
    }
  }
  mask->swap(*scratch);
}

struct TileSampleScratch {
  std::vector<float> finite_values;
  std::vector<float> blur_small;
  std::vector<float> blur_large;
  std::vector<float> dog_vals;
  std::vector<float> tile_gradients;
  std::vector<float> supported_gradients;
  std::vector<float> structure_values;
  std::vector<float> tile_pixels;
  std::vector<uint8_t> tile_common_support;
  std::vector<uint8_t> star_mask;
  std::vector<uint8_t> sat_mask;
  std::vector<uint8_t> dilate_scratch;
  std::vector<double> blur_integral;

/// @brief Implements prepare.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
  void prepare(int tw, int th) {
    const size_t tile_px = static_cast<size_t>(std::max(0, tw * th));
    finite_values.clear();
    finite_values.reserve(tile_px);
    blur_small.resize(tile_px);
    blur_large.resize(tile_px);
    dog_vals.resize(tile_px);
    tile_gradients.resize(tile_px);
    supported_gradients.clear();
    supported_gradients.reserve(tile_px);
    structure_values.clear();
    structure_values.reserve(tile_px);
    tile_pixels.clear();
    tile_pixels.reserve(tile_px);
    tile_common_support.resize(tile_px, 1);
    star_mask.assign(tile_px, 0);
    sat_mask.assign(tile_px, 0);
    dilate_scratch.resize(tile_px);
    blur_integral.resize(static_cast<size_t>((th + 1) * (tw + 1)));
  }
};

/// @brief Estimates structure noise scale.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float estimate_structure_noise_scale(const TileMetrics &tm,
                                     const std::vector<float> &bg_pixels) {
  if (std::isfinite(tm.noise) && tm.noise > kTiny) {
    return tm.noise;
  }
  if (!bg_pixels.empty()) {
    std::vector<float> tmp = bg_pixels;
    const float med = robust_median_inplace(tmp);
    const float sigma = 1.4826f * robust_mad(bg_pixels, med);
    if (std::isfinite(sigma) && sigma > kTiny) {
      return sigma;
    }
  }
  return 1.0f;
}

/// @brief Computes structure score from background mask.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float compute_structure_score_from_background_mask(
    const Matrix2Df &channel, int x0, int y0, int tw, int th, float grad_thresh,
    float noise_scale, const TileSampleScratch &scratch,
    TileSampleScratch *scratch_mut) {
  constexpr int kStructureBlurRadiusPx = 5;
  const float noise2 =
      std::max(kTiny * kTiny, noise_scale * noise_scale);
  box_blur_subregion(channel, x0, y0, tw, th, kStructureBlurRadiusPx,
                     &scratch_mut->blur_integral, &scratch_mut->blur_large);
  scratch_mut->structure_values.clear();
  for (int yy = 0; yy < th; ++yy) {
    for (int xx = 0; xx < tw; ++xx) {
      const size_t i = static_cast<size_t>(yy * tw + xx);
      if (scratch.tile_common_support[i] == 0 || scratch.star_mask[i] != 0 ||
          scratch.sat_mask[i] != 0 || scratch.tile_gradients[i] > grad_thresh) {
        continue;
      }
      const float v =
          channel(y0 + yy, x0 + xx);
      if (!std::isfinite(v)) {
        continue;
      }
      const float hp = v - scratch_mut->blur_large[i];
      scratch_mut->structure_values.push_back((hp * hp) / noise2);
    }
  }
  if (scratch_mut->structure_values.empty()) {
    return std::numeric_limits<float>::infinity();
  }
  return robust_median_inplace(scratch_mut->structure_values);
}

struct AutoTunePreparedTileSample {
  float x = 0.0f;
  float y = 0.0f;
  float weight = 0.0f;
  bool valid = false;
  std::vector<float> sorted_pixels;
};

enum class RBFKernelType { Multiquadric, ThinPlate, Gaussian };

/// @brief Resolves rbf kernel type.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
RBFKernelType resolve_rbf_kernel_type(const std::string &phi) {
  if (phi == "thinplate")
    return RBFKernelType::ThinPlate;
  if (phi == "gaussian")
    return RBFKernelType::Gaussian;
  return RBFKernelType::Multiquadric;
}

/// @brief Implements evaluate rbf kernel.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float evaluate_rbf_kernel(RBFKernelType kernel, float d, float mu,
                          float epsilon) {
  switch (kernel) {
  case RBFKernelType::ThinPlate:
    return rbf_kernel_thinplate(d, epsilon);
  case RBFKernelType::Gaussian:
    return rbf_kernel_gaussian(d, mu);
  case RBFKernelType::Multiquadric:
  default:
    return rbf_kernel_multiquadric(d, mu);
  }
}

using RobustWeightFn = float (*)(float, float);

/// @brief Resolves robust weight fn.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
RobustWeightFn resolve_robust_weight_fn(const std::string &loss) {
  return (loss == "tukey") ? &tukey_weight : &huber_weight;
}

/// @brief Implements robust weight from loss.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float robust_weight_from_loss(RobustWeightFn weight_fn, float r, float param) {
  return weight_fn(r, std::max(param, kTiny));
}

/// @brief Updates robust weights.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void update_robust_weights(const Eigen::VectorXf &residual, float param,
                           RobustWeightFn weight_fn,
                           Eigen::VectorXf *out_weights) {
  const int n = static_cast<int>(residual.size());
  for (int i = 0; i < n; ++i) {
    (*out_weights)(i) = robust_weight_from_loss(weight_fn, residual(i), param);
  }
}

/// @brief Implements stats from values.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
BGEValueStats stats_from_values(const std::vector<float> &values) {
  BGEValueStats st;
  if (values.empty())
    return st;

  std::vector<float> finite;
  finite.reserve(values.size());
  for (float v : values) {
    if (std::isfinite(v))
      finite.push_back(v);
  }
  if (finite.empty())
    return st;

  st.n = static_cast<int>(finite.size());
  st.min = *std::min_element(finite.begin(), finite.end());
  st.max = *std::max_element(finite.begin(), finite.end());
  st.median = robust_median_inplace(finite);

  double sum = 0.0;
  for (float v : finite)
    sum += static_cast<double>(v);
  st.mean = static_cast<float>(sum / static_cast<double>(finite.size()));

  double sum_sq = 0.0;
  for (float v : finite) {
    const double d = static_cast<double>(v) - static_cast<double>(st.mean);
    sum_sq += d * d;
  }
  st.std = static_cast<float>(
      std::sqrt(sum_sq / static_cast<double>(finite.size())));
  return st;
}

/// @brief Implements stats from matrix.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
BGEValueStats stats_from_matrix(const Matrix2Df &m) {
  std::vector<float> vals;
  vals.reserve(static_cast<size_t>(m.size()));
  for (int i = 0; i < m.size(); ++i) {
    vals.push_back(m.data()[i]);
  }
  return stats_from_values(vals);
}

} // namespace

static std::vector<AutoTunePreparedTileSample>
extract_autotune_prepared_tile_samples(
    const Matrix2Df &channel, const std::vector<TileMetrics> &tile_metrics,
    const TileGrid &tile_grid, const BGEConfig &config);

/// @brief Implements canvas mask matches image.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool canvas_mask_matches_image(const std::vector<uint8_t> &mask, int rows,
                               int cols) {
  if (rows <= 0 || cols <= 0)
    return false;
  return mask.size() == static_cast<size_t>(rows * cols);
}

/// @brief Implements enforce canvas mask on rgb.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void enforce_canvas_mask_on_rgb(Matrix2Df &R, Matrix2Df &G, Matrix2Df &B,
                                const std::vector<uint8_t> &mask) {
  const int rows = R.rows();
  const int cols = R.cols();
  if (!canvas_mask_matches_image(mask, rows, cols))
    return;
  if (G.rows() != rows || G.cols() != cols || B.rows() != rows ||
      B.cols() != cols) {
    return;
  }
  for (int y = 0; y < rows; ++y) {
    const size_t row_off = static_cast<size_t>(y) * static_cast<size_t>(cols);
    for (int x = 0; x < cols; ++x) {
      const size_t idx = row_off + static_cast<size_t>(x);
      if (mask[idx] == 0) {
        R(y, x) = 0.0f;
        G(y, x) = 0.0f;
        B(y, x) = 0.0f;
      }
    }
  }
}

/// @brief Implements spatial background spread.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float spatial_background_spread(const Matrix2Df &img,
                                const std::vector<uint8_t> *valid_mask) {
  const int H = static_cast<int>(img.rows());
  const int W = static_cast<int>(img.cols());
  if (H <= 0 || W <= 0)
    return std::numeric_limits<float>::infinity();

  const size_t total_px = static_cast<size_t>(H * W);
  const bool use_mask =
      (valid_mask != nullptr && valid_mask->size() == total_px);

  std::vector<float> valid_values;
  valid_values.reserve(total_px);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const size_t i = static_cast<size_t>(y * W + x);
      if (use_mask && (*valid_mask)[i] == 0)
        continue;
      const float v = img.data()[i];
      if (std::isfinite(v) && v > 0.0f)
        valid_values.push_back(v);
    }
  }
  if (valid_values.size() < 4096)
    return std::numeric_limits<float>::infinity();

  const float value_thresh = robust_quantile_inplace(valid_values, 0.65f);

  std::vector<float> grad_map(static_cast<size_t>(H * W), 0.0f);
  std::vector<float> grad_values;
  grad_values.reserve(valid_values.size());
  for (int y = 0; y < H; ++y) {
    const int ym = std::max(0, y - 1);
    const int yp = std::min(H - 1, y + 1);
    for (int x = 0; x < W; ++x) {
      const size_t idx = static_cast<size_t>(y * W + x);
      if (use_mask && (*valid_mask)[idx] == 0)
        continue;
      const float v = img(y, x);
      if (!(std::isfinite(v) && v > 0.0f))
        continue;

      const int xm = std::max(0, x - 1);
      const int xp = std::min(W - 1, x + 1);

      float vxm = img(y, xm);
      float vxp = img(y, xp);
      float vym = img(ym, x);
      float vyp = img(yp, x);

      if (!std::isfinite(vxm))
        vxm = v;
      if (!std::isfinite(vxp))
        vxp = v;
      if (!std::isfinite(vym))
        vym = v;
      if (!std::isfinite(vyp))
        vyp = v;

      const float g = std::abs(vxp - vxm) + std::abs(vyp - vym);
      grad_map[idx] = g;
      grad_values.push_back(g);
    }
  }
  if (grad_values.size() < 4096)
    return std::numeric_limits<float>::infinity();

  const float grad_thresh = robust_quantile_inplace(grad_values, 0.70f);
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
          const size_t idx = static_cast<size_t>(y * W + x);
          if (use_mask && (*valid_mask)[idx] == 0)
            continue;
          const float v = img(y, x);
          if (!(std::isfinite(v) && v > 0.0f))
            continue;
          if (v > value_thresh)
            continue;
          const float g = grad_map[idx];
          if (!(std::isfinite(g) && g <= grad_thresh))
            continue;
          block_values.push_back(v);
        }
      }

      if (static_cast<int>(block_values.size()) >= kMinPixelsPerBlock) {
        block_medians.push_back(robust_median_inplace(block_values));
      }
    }
  }

  if (block_medians.size() < 8)
    return std::numeric_limits<float>::infinity();

  std::sort(block_medians.begin(), block_medians.end());
  const float p10 = sorted_quantile(block_medians, 0.10f);
  const float p90 = sorted_quantile(block_medians, 0.90f);
  return p90 - p10;
}

/// @brief Implements coarse background plane slope.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float coarse_background_plane_slope(const Matrix2Df &img,
                                    const std::vector<uint8_t> *valid_mask) {
  const int H = static_cast<int>(img.rows());
  const int W = static_cast<int>(img.cols());
  if (H <= 0 || W <= 0)
    return std::numeric_limits<float>::infinity();

  const size_t total_px = static_cast<size_t>(H * W);
  const bool use_mask =
      (valid_mask != nullptr && valid_mask->size() == total_px);

  std::vector<float> valid_values;
  valid_values.reserve(total_px);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const size_t i = static_cast<size_t>(y * W + x);
      if (use_mask && (*valid_mask)[i] == 0)
        continue;
      const float v = img.data()[i];
      if (std::isfinite(v) && v > 0.0f)
        valid_values.push_back(v);
    }
  }
  if (valid_values.size() < 4096)
    return std::numeric_limits<float>::infinity();

  const float value_thresh = robust_quantile_inplace(valid_values, 0.65f);

  std::vector<float> grad_map(static_cast<size_t>(H * W), 0.0f);
  std::vector<float> grad_values;
  grad_values.reserve(valid_values.size());
  for (int y = 0; y < H; ++y) {
    const int ym = std::max(0, y - 1);
    const int yp = std::min(H - 1, y + 1);
    for (int x = 0; x < W; ++x) {
      const size_t idx = static_cast<size_t>(y * W + x);
      if (use_mask && (*valid_mask)[idx] == 0)
        continue;
      const float v = img(y, x);
      if (!(std::isfinite(v) && v > 0.0f))
        continue;

      const int xm = std::max(0, x - 1);
      const int xp = std::min(W - 1, x + 1);

      float vxm = img(y, xm);
      float vxp = img(y, xp);
      float vym = img(ym, x);
      float vyp = img(yp, x);
      if (!std::isfinite(vxm))
        vxm = v;
      if (!std::isfinite(vxp))
        vxp = v;
      if (!std::isfinite(vym))
        vym = v;
      if (!std::isfinite(vyp))
        vyp = v;

      const float g = std::abs(vxp - vxm) + std::abs(vyp - vym);
      grad_map[idx] = g;
      grad_values.push_back(g);
    }
  }
  if (grad_values.size() < 4096)
    return std::numeric_limits<float>::infinity();

  const float grad_thresh = robust_quantile_inplace(grad_values, 0.70f);
  constexpr int kBlockSize = 128;
  constexpr int kMinPixelsPerBlock = 256;

  std::vector<std::array<float, 3>> samples;
  std::vector<float> block_values;
  for (int y0 = 0; y0 < H; y0 += kBlockSize) {
    for (int x0 = 0; x0 < W; x0 += kBlockSize) {
      const int y1 = std::min(H, y0 + kBlockSize);
      const int x1 = std::min(W, x0 + kBlockSize);
      block_values.clear();
      block_values.reserve(static_cast<size_t>((y1 - y0) * (x1 - x0)));
      for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
          const size_t idx = static_cast<size_t>(y * W + x);
          if (use_mask && (*valid_mask)[idx] == 0)
            continue;
          const float v = img(y, x);
          if (!(std::isfinite(v) && v > 0.0f))
            continue;
          if (v > value_thresh)
            continue;
          const float g = grad_map[idx];
          if (!(std::isfinite(g) && g <= grad_thresh))
            continue;
          block_values.push_back(v);
        }
      }
      if (static_cast<int>(block_values.size()) < kMinPixelsPerBlock)
        continue;
      const float z = robust_median_inplace(block_values);
      const float cx = 0.5f * static_cast<float>(x0 + x1 - 1);
      const float cy = 0.5f * static_cast<float>(y0 + y1 - 1);
      samples.push_back({cx, cy, z});
    }
  }
  if (samples.size() < 8)
    return std::numeric_limits<float>::infinity();

  double mx = 0.0, my = 0.0, mz = 0.0;
  for (const auto &s : samples) {
    mx += s[0];
    my += s[1];
    mz += s[2];
  }
  const double n = static_cast<double>(samples.size());
  mx /= n;
  my /= n;
  mz /= n;

  double sxx = 0.0, syy = 0.0, sxy = 0.0, sxz = 0.0, syz = 0.0;
  for (const auto &s : samples) {
    const double dx = static_cast<double>(s[0]) - mx;
    const double dy = static_cast<double>(s[1]) - my;
    const double dz = static_cast<double>(s[2]) - mz;
    sxx += dx * dx;
    syy += dy * dy;
    sxy += dx * dy;
    sxz += dx * dz;
    syz += dy * dz;
  }

  const double det = sxx * syy - sxy * sxy;
  if (!(std::isfinite(det)) || std::abs(det) < 1.0e-12) {
    return std::numeric_limits<float>::infinity();
  }

  const double ax = (sxz * syy - syz * sxy) / det;
  const double ay = (syz * sxx - sxz * sxy) / det;
  if (!(std::isfinite(ax) && std::isfinite(ay))) {
    return std::numeric_limits<float>::infinity();
  }
  return static_cast<float>(std::sqrt(ax * ax + ay * ay));
}


namespace { // (re-open anonymous namespace for private helpers)

struct ForegroundComponent {
  int label = 0;
  int area = 0;
  float peak = 0.0f;
  int peak_x = 0;
  int peak_y = 0;
  int min_x = 0;
  int max_x = 0;
  int min_y = 0;
  int max_y = 0;
  int dilation_radius = 0;
  std::vector<int> pixels;
};

struct MeshSkyFitResult {
  Matrix2Df model;
  std::vector<GridCell> grid_cells;
  int mesh_size = 0;
  int n_valid_cells = 0;
  float rms_residual = 0.0f;
  bool success = false;
};

/// @brief Implements sampled positive quantile.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float sampled_positive_quantile(const Matrix2Df &img, float q, int step) {
  step = std::max(1, step);
  std::vector<float> vals;
  vals.reserve(
      static_cast<size_t>((img.rows() / step + 1) * (img.cols() / step + 1)));
  for (int y = 0; y < img.rows(); y += step) {
    for (int x = 0; x < img.cols(); x += step) {
      const float v = img(y, x);
      if (std::isfinite(v) && v > 0.0f)
        vals.push_back(v);
    }
  }
  if (vals.empty())
    return 0.0f;
  return robust_quantile(std::move(vals), q);
}

/// @brief Extracts connected components above threshold.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void extract_connected_components_above_threshold(
    const Matrix2Df &luma, float threshold,
    std::vector<ForegroundComponent> *components, std::vector<int> *labels) {
  components->clear();
  const int H = static_cast<int>(luma.rows());
  const int W = static_cast<int>(luma.cols());
  labels->assign(static_cast<size_t>(H * W), 0);
  if (H <= 0 || W <= 0)
    return;

  auto is_fg = [&](int y, int x) {
    const float v = luma(y, x);
    return std::isfinite(v) && v > threshold;
  };

  int next_label = 1;
  std::queue<int> q;
  const int dx8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
  const int dy8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const int root = y * W + x;
      if ((*labels)[static_cast<size_t>(root)] != 0)
        continue;
      if (!is_fg(y, x))
        continue;

      ForegroundComponent comp;
      comp.label = next_label;
      comp.area = 0;
      comp.peak = luma(y, x);
      comp.peak_x = x;
      comp.peak_y = y;
      comp.min_x = x;
      comp.max_x = x;
      comp.min_y = y;
      comp.max_y = y;

      (*labels)[static_cast<size_t>(root)] = next_label;
      q.push(root);

      while (!q.empty()) {
        const int idx = q.front();
        q.pop();
        const int cy = idx / W;
        const int cx = idx - cy * W;
        const float cv = luma(cy, cx);

        comp.pixels.push_back(idx);
        ++comp.area;
        if (cv > comp.peak) {
          comp.peak = cv;
          comp.peak_x = cx;
          comp.peak_y = cy;
        }
        comp.min_x = std::min(comp.min_x, cx);
        comp.max_x = std::max(comp.max_x, cx);
        comp.min_y = std::min(comp.min_y, cy);
        comp.max_y = std::max(comp.max_y, cy);

        for (int k = 0; k < 8; ++k) {
          const int nx = cx + dx8[k];
          const int ny = cy + dy8[k];
          if (nx < 0 || ny < 0 || nx >= W || ny >= H)
            continue;
          const int nidx = ny * W + nx;
          int &lab = (*labels)[static_cast<size_t>(nidx)];
          if (lab != 0)
            continue;
          if (!is_fg(ny, nx))
            continue;
          lab = next_label;
          q.push(nidx);
        }
      }

      if (comp.area >= 4) {
        components->push_back(std::move(comp));
        ++next_label;
      } else {
        for (int idx : comp.pixels) {
          (*labels)[static_cast<size_t>(idx)] = 0;
        }
      }
    }
  }
}

std::vector<std::vector<std::pair<int, int>>>
/// @brief Builds disk offsets table.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
build_disk_offsets_table(int max_radius) {
  max_radius = std::max(0, max_radius);
  std::vector<std::vector<std::pair<int, int>>> table(
      static_cast<size_t>(max_radius + 1));
  for (int r = 0; r <= max_radius; ++r) {
    auto &offsets = table[static_cast<size_t>(r)];
    const int r2 = r * r;
    for (int dy = -r; dy <= r; ++dy) {
      for (int dx = -r; dx <= r; ++dx) {
        if (dx * dx + dy * dy <= r2)
          offsets.push_back({dx, dy});
      }
    }
  }
  return table;
}

std::vector<uint8_t>
/// @brief Builds modeled foreground mask.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
build_modeled_foreground_mask(const Matrix2Df &luma, const BGEConfig &config,
                              std::vector<ForegroundComponent> *components,
                              float *out_threshold, float *out_sigma) {
  const int H = static_cast<int>(luma.rows());
  const int W = static_cast<int>(luma.cols());
  std::vector<uint8_t> out_mask(static_cast<size_t>(std::max(0, H * W)), 0);
  if (H <= 0 || W <= 0)
    return out_mask;

  std::vector<float> sample_vals;
  sample_vals.reserve(static_cast<size_t>((H / 4 + 1) * (W / 4 + 1)));
  for (int y = 0; y < H; y += 4) {
    for (int x = 0; x < W; x += 4) {
      const float v = luma(y, x);
      if (std::isfinite(v) && v > 0.0f)
        sample_vals.push_back(v);
    }
  }
  if (sample_vals.size() < 256)
    return out_mask;

  const float med = robust_median_inplace(sample_vals);
  const float sigma = std::max(1.0e-6f, 1.4826f * robust_mad(sample_vals, med));
  const float thresh = med + 0.8f * sigma;
  if (out_threshold)
    *out_threshold = thresh;
  if (out_sigma)
    *out_sigma = sigma;

  std::vector<int> labels;
  extract_connected_components_above_threshold(luma, thresh, components,
                                               &labels);
  if (components->empty())
    return out_mask;

  const int base_radius = std::max(1, config.mask.star_dilate_px);
  const int max_radius = base_radius + 20;
  const auto offsets_table = build_disk_offsets_table(max_radius);
  constexpr float kPi = 3.14159265359f;

  for (auto &comp : *components) {
    const float peak_sigma =
        std::max(0.0f, (comp.peak - thresh) / std::max(1.0e-6f, sigma));
    const float area_scale =
        std::sqrt(static_cast<float>(std::max(1, comp.area)) / kPi);
    int rad =
        base_radius + static_cast<int>(std::lround(
                          1.6f * std::log1p(peak_sigma) + 0.15f * area_scale));
    rad = std::clamp(rad, base_radius, max_radius);
    comp.dilation_radius = rad;

    const auto &offsets = offsets_table[static_cast<size_t>(rad)];
    for (int idx : comp.pixels) {
      const int y = idx / W;
      const int x = idx - y * W;
      for (const auto &[dx, dy] : offsets) {
        const int xx = x + dx;
        const int yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= W || yy >= H)
          continue;
        out_mask[static_cast<size_t>(yy * W + xx)] = 1;
      }
    }
  }
  return out_mask;
}

/// @brief Implements robust mesh background estimate.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float robust_mesh_background_estimate(std::vector<float> values) {
  if (values.size() < 16)
    return std::numeric_limits<float>::quiet_NaN();
  for (int iter = 0; iter < 4; ++iter) {
    const float med = robust_median_inplace(values);
    const float sigma = 1.4826f * robust_mad(values, med);
    if (!(std::isfinite(sigma) && sigma > 1.0e-6f))
      break;
    const float clip = 2.5f * sigma;
    std::vector<float> kept;
    kept.reserve(values.size());
    for (float v : values) {
      if (std::isfinite(v) && std::abs(v - med) <= clip)
        kept.push_back(v);
    }
    if (kept.size() < 16 || kept.size() == values.size())
      break;
    values.swap(kept);
  }
  if (values.size() < 8)
    return std::numeric_limits<float>::quiet_NaN();
  return robust_median_inplace(values);
}

/// @brief Implements annulus background median.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float annulus_background_median(const Matrix2Df &img, int cx, int cy, int r_in,
                                int r_out,
                                const std::vector<uint8_t> &fg_mask) {
  const int H = static_cast<int>(img.rows());
  const int W = static_cast<int>(img.cols());
  if (H <= 0 || W <= 0)
    return std::numeric_limits<float>::quiet_NaN();
  r_in = std::max(1, r_in);
  r_out = std::max(r_in + 1, r_out);
  const int r_in2 = r_in * r_in;
  const int r_out2 = r_out * r_out;

  std::vector<float> vals;
  vals.reserve(static_cast<size_t>(6 * r_out));
  const int y0 = std::max(0, cy - r_out);
  const int y1 = std::min(H - 1, cy + r_out);
  const int x0 = std::max(0, cx - r_out);
  const int x1 = std::min(W - 1, cx + r_out);
  for (int y = y0; y <= y1; ++y) {
    for (int x = x0; x <= x1; ++x) {
      const int dx = x - cx;
      const int dy = y - cy;
      const int d2 = dx * dx + dy * dy;
      if (d2 < r_in2 || d2 > r_out2)
        continue;
      const size_t idx = static_cast<size_t>(y * W + x);
      if (idx < fg_mask.size() && fg_mask[idx] != 0)
        continue;
      const float v = img(y, x);
      if (std::isfinite(v) && v > 0.0f)
        vals.push_back(v);
    }
  }
  if (vals.size() < 24)
    return std::numeric_limits<float>::quiet_NaN();
  return robust_median_inplace(vals);
}

/// @brief Implements subtract bright source models.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void subtract_bright_source_models(
    Matrix2Df &sky_work, const Matrix2Df &luma,
    const std::vector<uint8_t> &fg_mask,
    const std::vector<ForegroundComponent> &components, float low_threshold,
    float sigma) {
  if (components.empty())
    return;
  const int H = static_cast<int>(sky_work.rows());
  const int W = static_cast<int>(sky_work.cols());
  if (H <= 0 || W <= 0)
    return;

  const float q995 = sampled_positive_quantile(luma, 0.995f, 4);
  const float bright_thresh =
      std::max(q995, low_threshold + 6.0f * std::max(1.0e-6f, sigma));

  std::vector<int> order(components.size());
  for (size_t i = 0; i < components.size(); ++i)
    order[i] = static_cast<int>(i);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return components[static_cast<size_t>(a)].peak >
           components[static_cast<size_t>(b)].peak;
  });

  constexpr float kPi = 3.14159265359f;
  int modeled = 0;
  const int max_models = 1200;
  for (int oi : order) {
    if (modeled >= max_models)
      break;
    const auto &comp = components[static_cast<size_t>(oi)];
    if (comp.area <= 2 || comp.area > 256)
      continue;
    if (!(std::isfinite(comp.peak) && comp.peak >= bright_thresh))
      continue;

    const int cx = comp.peak_x;
    const int cy = comp.peak_y;
    if (cx < 0 || cy < 0 || cx >= W || cy >= H)
      continue;

    const int ann_in = std::max(3, comp.dilation_radius + 2);
    const int ann_out = ann_in + 10;
    const float bg_local =
        annulus_background_median(sky_work, cx, cy, ann_in, ann_out, fg_mask);
    if (!(std::isfinite(bg_local) && bg_local > 0.0f))
      continue;

    const float peak_v = sky_work(cy, cx);
    const float amp = peak_v - bg_local;
    if (!(std::isfinite(amp) && amp > 0.0f))
      continue;

    const float sigma_px = std::clamp(
        0.35f * std::sqrt(static_cast<float>(std::max(1, comp.area)) / kPi),
        1.2f, 8.0f);
    const float sigma2 = sigma_px * sigma_px;
    const int radius =
        std::clamp(static_cast<int>(std::ceil(3.0f * sigma_px +
                                              0.5f * comp.dilation_radius)),
                   4, 28);
    const int r2_max = radius * radius;

    const int y0 = std::max(0, cy - radius);
    const int y1 = std::min(H - 1, cy + radius);
    const int x0 = std::max(0, cx - radius);
    const int x1 = std::min(W - 1, cx + radius);
    for (int y = y0; y <= y1; ++y) {
      for (int x = x0; x <= x1; ++x) {
        const int dx = x - cx;
        const int dy = y - cy;
        const int r2 = dx * dx + dy * dy;
        if (r2 > r2_max)
          continue;
        const float model = amp * std::exp(-0.5f * static_cast<float>(r2) /
                                           std::max(sigma2, 1.0e-6f));
        const float v = sky_work(y, x);
        if (!std::isfinite(v))
          continue;
        sky_work(y, x) = std::max(bg_local, v - model);
      }
    }
    ++modeled;
  }

  std::cout << "[BGE]   modeled_mask_mesh: bright-source models applied="
            << modeled << std::endl;
}

/// @brief Implements fit modeled mask mesh surface.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
MeshSkyFitResult fit_modeled_mask_mesh_surface(
    const Matrix2Df &channel, const Matrix2Df &luma,
    const std::vector<uint8_t> &fg_mask,
    const std::vector<ForegroundComponent> &components, float low_threshold,
    float sigma, const BGEConfig &config, int tile_size_hint) {
  MeshSkyFitResult out;
  const int H = static_cast<int>(channel.rows());
  const int W = static_cast<int>(channel.cols());
  if (H <= 0 || W <= 0)
    return out;

  Matrix2Df sky_work = channel;
  subtract_bright_source_models(sky_work, luma, fg_mask, components,
                                low_threshold, sigma);

  const int min_dim = std::min(W, H);
  int mesh = min_dim / std::max(8, config.grid.N_g);
  const int mesh_min = std::max(32, config.grid.G_min_px / 2);
  int mesh_max = std::max(config.grid.G_min_px, min_dim / 6);
  if (tile_size_hint > 0)
    mesh_max = std::max(mesh_max, tile_size_hint);
  mesh = std::clamp(mesh, mesh_min, mesh_max);
  out.mesh_size = std::max(16, mesh);

  const int nx = (W + out.mesh_size - 1) / out.mesh_size;
  const int ny = (H + out.mesh_size - 1) / out.mesh_size;
  if (nx <= 0 || ny <= 0)
    return out;

  std::vector<float> cell_values(static_cast<size_t>(nx * ny),
                                 std::numeric_limits<float>::quiet_NaN());
  std::vector<int> cell_counts(static_cast<size_t>(nx * ny), 0);
  std::vector<uint8_t> cell_valid(static_cast<size_t>(nx * ny), 0);

  for (int cy = 0; cy < ny; ++cy) {
    const int y0 = cy * out.mesh_size;
    const int y1 = std::min(H, y0 + out.mesh_size);
    for (int cx = 0; cx < nx; ++cx) {
      const int x0 = cx * out.mesh_size;
      const int x1 = std::min(W, x0 + out.mesh_size);
      const int area = std::max(1, (y1 - y0) * (x1 - x0));
      const int min_required = std::max(48, area / 16);

      std::vector<float> vals;
      vals.reserve(static_cast<size_t>(area / 2));
      for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
          const size_t idx = static_cast<size_t>(y * W + x);
          if (idx < fg_mask.size() && fg_mask[idx] != 0)
            continue;
          const float v = sky_work(y, x);
          if (std::isfinite(v) && v > 0.0f)
            vals.push_back(v);
        }
      }

      const size_t i = static_cast<size_t>(cy * nx + cx);
      cell_counts[i] = static_cast<int>(vals.size());
      if (cell_counts[i] < min_required)
        continue;

      const float bg = robust_mesh_background_estimate(std::move(vals));
      if (std::isfinite(bg) && bg > 0.0f) {
        cell_values[i] = bg;
        cell_valid[i] = 1;
        ++out.n_valid_cells;
      }
    }
  }

  if (out.n_valid_cells < 8)
    return out;

  std::vector<int> valid_idx;
  valid_idx.reserve(static_cast<size_t>(out.n_valid_cells));
  for (int i = 0; i < nx * ny; ++i) {
    if (cell_valid[static_cast<size_t>(i)] != 0)
      valid_idx.push_back(i);
  }

  for (int cy = 0; cy < ny; ++cy) {
    for (int cx = 0; cx < nx; ++cx) {
      const size_t i = static_cast<size_t>(cy * nx + cx);
      if (cell_valid[i] != 0)
        continue;
      float best_d2 = std::numeric_limits<float>::infinity();
      float best_v = std::numeric_limits<float>::quiet_NaN();
      for (int vi : valid_idx) {
        const int vy = vi / nx;
        const int vx = vi - vy * nx;
        const float dx = static_cast<float>(cx - vx);
        const float dy = static_cast<float>(cy - vy);
        const float d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
          best_d2 = d2;
          best_v = cell_values[static_cast<size_t>(vi)];
        }
      }
      if (std::isfinite(best_v)) {
        cell_values[i] = best_v;
      }
    }
  }

  out.model = Matrix2Df::Zero(H, W);
  for (int y = 0; y < H; ++y) {
    const float gy =
        (static_cast<float>(y) + 0.5f) / static_cast<float>(out.mesh_size) -
        0.5f;
    int y0 = static_cast<int>(std::floor(gy));
    float ty = gy - static_cast<float>(y0);
    y0 = std::clamp(y0, 0, ny - 1);
    const int y1 = std::clamp(y0 + 1, 0, ny - 1);
    if (y1 == y0)
      ty = 0.0f;

    for (int x = 0; x < W; ++x) {
      const float gx =
          (static_cast<float>(x) + 0.5f) / static_cast<float>(out.mesh_size) -
          0.5f;
      int x0 = static_cast<int>(std::floor(gx));
      float tx = gx - static_cast<float>(x0);
      x0 = std::clamp(x0, 0, nx - 1);
      const int x1 = std::clamp(x0 + 1, 0, nx - 1);
      if (x1 == x0)
        tx = 0.0f;

      const float v00 = cell_values[static_cast<size_t>(y0 * nx + x0)];
      const float v10 = cell_values[static_cast<size_t>(y0 * nx + x1)];
      const float v01 = cell_values[static_cast<size_t>(y1 * nx + x0)];
      const float v11 = cell_values[static_cast<size_t>(y1 * nx + x1)];
      const float v0 = (1.0f - tx) * v00 + tx * v10;
      const float v1 = (1.0f - tx) * v01 + tx * v11;
      out.model(y, x) = (1.0f - ty) * v0 + ty * v1;
    }
  }

  double sum_sq = 0.0;
  int n_r = 0;
  out.grid_cells.reserve(static_cast<size_t>(nx * ny));
  for (int cy = 0; cy < ny; ++cy) {
    for (int cx = 0; cx < nx; ++cx) {
      const size_t i = static_cast<size_t>(cy * nx + cx);
      GridCell gc;
      gc.cell_x = cx;
      gc.cell_y = cy;
      gc.center_x = std::min(static_cast<float>(W - 1),
                             (static_cast<float>(cx) + 0.5f) * out.mesh_size);
      gc.center_y = std::min(static_cast<float>(H - 1),
                             (static_cast<float>(cy) + 0.5f) * out.mesh_size);
      gc.bg_value = cell_values[i];
      gc.weight = 1.0f;
      gc.n_samples = cell_counts[i];
      gc.valid = (cell_valid[i] != 0);
      out.grid_cells.push_back(gc);

      if (gc.valid && std::isfinite(gc.bg_value)) {
        const int px =
            std::clamp(static_cast<int>(std::lround(gc.center_x)), 0, W - 1);
        const int py =
            std::clamp(static_cast<int>(std::lround(gc.center_y)), 0, H - 1);
        const float r = gc.bg_value - out.model(py, px);
        sum_sq += static_cast<double>(r) * static_cast<double>(r);
        ++n_r;
      }
    }
  }

  if (n_r > 0) {
    out.rms_residual =
        static_cast<float>(std::sqrt(sum_sq / static_cast<double>(n_r)));
  }
  out.success = out.model.size() > 0 && out.model.allFinite();
  return out;
}

} // namespace


/// @brief Builds chroma background mask from rgb.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::vector<uint8_t> build_chroma_background_mask_from_rgb(
    const Matrix2Df &R, const Matrix2Df &G, const Matrix2Df &B,
    const std::vector<uint8_t> *valid_mask) {
  const int H = static_cast<int>(R.rows());
  const int W = static_cast<int>(R.cols());
  std::vector<uint8_t> mask(static_cast<size_t>(std::max(0, H * W)), 0);
  if (H <= 0 || W <= 0)
    return mask;

  const size_t total_px = static_cast<size_t>(H * W);
  const bool use_external_mask =
      (valid_mask != nullptr && valid_mask->size() == total_px);

  std::vector<float> luma(total_px, 0.0f);
  std::vector<float> lum_samples;
  lum_samples.reserve(static_cast<size_t>((H / 2 + 1) * (W / 2 + 1)));
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const size_t idx = static_cast<size_t>(y * W + x);
      if (use_external_mask && (*valid_mask)[idx] == 0)
        continue;
      const float rv = R(y, x);
      const float gv = G(y, x);
      const float bv = B(y, x);
      if (!(std::isfinite(rv) && rv > 0.0f && std::isfinite(gv) && gv > 0.0f &&
            std::isfinite(bv) && bv > 0.0f)) {
        continue;
      }
      const float lv = 0.2126f * rv + 0.7152f * gv + 0.0722f * bv;
      luma[idx] = lv;
      if ((y % 2) == 0 && (x % 2) == 0)
        lum_samples.push_back(lv);
    }
  }
  if (lum_samples.size() < 4096) {
    // Low-data fallback: treat all finite positive pixels as safe background.
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const size_t idx = static_cast<size_t>(y * W + x);
        mask[idx] = (luma[idx] > 0.0f) ? 1 : 0;
      }
    }
    return mask;
  }

  const float lum_thresh = robust_quantile(lum_samples, 0.60f);
  std::vector<float> grad(static_cast<size_t>(H * W), 0.0f);
  std::vector<float> grad_samples;
  grad_samples.reserve(lum_samples.size());
  for (int y = 0; y < H; ++y) {
    const int ym = std::max(0, y - 1);
    const int yp = std::min(H - 1, y + 1);
    for (int x = 0; x < W; ++x) {
      const int xm = std::max(0, x - 1);
      const int xp = std::min(W - 1, x + 1);
      const size_t idx = static_cast<size_t>(y * W + x);
      const float lv = luma[idx];
      if (!(std::isfinite(lv) && lv > 0.0f))
        continue;
      const float gx = std::abs(luma[static_cast<size_t>(y * W + xp)] -
                                luma[static_cast<size_t>(y * W + xm)]);
      const float gy = std::abs(luma[static_cast<size_t>(yp * W + x)] -
                                luma[static_cast<size_t>(ym * W + x)]);
      const float gv = gx + gy;
      grad[idx] = gv;
      if ((y % 2) == 0 && (x % 2) == 0)
        grad_samples.push_back(gv);
    }
  }
  if (grad_samples.size() < 4096) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const size_t idx = static_cast<size_t>(y * W + x);
        mask[idx] = (luma[idx] > 0.0f && luma[idx] <= lum_thresh) ? 1 : 0;
      }
    }
    return mask;
  }

  const float grad_thresh = robust_quantile(grad_samples, 0.70f);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const size_t idx = static_cast<size_t>(y * W + x);
      const float lv = luma[idx];
      const float gv = grad[idx];
      mask[idx] = (std::isfinite(lv) && lv > 0.0f && lv <= lum_thresh &&
                   std::isfinite(gv) && gv <= grad_thresh)
                      ? 1
                      : 0;
    }
  }
  return mask;
}

/// @brief Implements log chroma std background.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float log_chroma_std_background(const Matrix2Df &A, const Matrix2Df &G,
                                const std::vector<uint8_t> &bg_mask) {
  const int H = static_cast<int>(A.rows());
  const int W = static_cast<int>(A.cols());
  if (H <= 0 || W <= 0)
    return std::numeric_limits<float>::infinity();

  std::vector<float> vals;
  vals.reserve(static_cast<size_t>(H * W / 3));
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const size_t idx = static_cast<size_t>(y * W + x);
      if (idx >= bg_mask.size() || bg_mask[idx] == 0)
        continue;
      const float av = A(y, x);
      const float gv = G(y, x);
      if (!(std::isfinite(av) && std::isfinite(gv) && av > 0.0f && gv > 0.0f))
        continue;
      vals.push_back(std::log(av / gv));
    }
  }
  if (vals.size() < 1024)
    return std::numeric_limits<float>::infinity();

  const float mean = stats_from_values(vals).mean;
  double sum_sq = 0.0;
  for (float v : vals) {
    const double d = static_cast<double>(v) - static_cast<double>(mean);
    sum_sq += d * d;
  }
  return static_cast<float>(
      std::sqrt(sum_sq / static_cast<double>(vals.size())));
}

// RBF kernel functions (v3.3 §6.3.7)
/// @brief Implements rbf kernel multiquadric.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float rbf_kernel_multiquadric(float d, float mu) {
  return std::sqrt(d * d + mu * mu);
}

/// @brief Implements rbf kernel thinplate.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float rbf_kernel_thinplate(float d, float epsilon) {
  float d_safe = d + epsilon;
  return (d_safe > epsilon) ? (d_safe * d_safe * std::log(d_safe)) : 0.0f;
}

/// @brief Implements rbf kernel gaussian.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float rbf_kernel_gaussian(float d, float mu) {
  return std::exp(-d * d / (2.0f * mu * mu));
}

/// @brief Implements huber weight.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float huber_weight(float r, float delta) {
  float abs_r = std::abs(r);
  return (abs_r <= delta) ? 1.0f : (delta / abs_r);
}

/// @brief Implements tukey weight.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float tukey_weight(float r, float c) {
  float abs_r = std::abs(r);
  if (abs_r <= c) {
    float u = r / c;
    return std::pow(1.0f - u * u, 2.0f);
  }
  return 0.0f;
}

// Compute adaptive grid spacing (v3.3 §6.3.8)
/// @brief Computes grid spacing.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
int compute_grid_spacing(int image_width, int image_height, int tile_size,
                         const BGEConfig &config) {
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

  // §6.3.9 compact-tile warning: when tile size forces G >> resolution-based
  // estimate, the BGE grid is coarser than ideal (compact-tile mode).
  if (G_from_tiles > G_from_resolution && G_from_resolution > 0) {
    std::cout << "[BGE] Warning: compact-tile mode detected (2*T=" << G_from_tiles
              << " > G_res=" << G_from_resolution
              << "); grid spacing forced to G=" << G
              << " (§6.3.9). BGE accuracy may be reduced."
              << std::endl;
  }

  return G;
}

// Extract tile background samples (v3.3 §6.3.2)
/// @brief Extracts tile background samples.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::vector<TileBGSample> extract_tile_background_samples(
    const Matrix2Df &channel, const std::vector<TileMetrics> &tile_metrics,
    const TileGrid &tile_grid, const BGEConfig &config) {

  const auto prepared = extract_autotune_prepared_tile_samples(
      channel, tile_metrics, tile_grid, config);
  std::vector<TileBGSample> samples;
  samples.reserve(prepared.size());
  for (const auto &p : prepared) {
    TileBGSample sample{};
    sample.x = p.x;
    sample.y = p.y;
    sample.weight = p.weight;
    sample.valid = false;
    if (p.valid && !p.sorted_pixels.empty()) {
      sample.bg_value = estimate_tile_background_from_sorted(p.sorted_pixels,
                                                             config);
      sample.valid = std::isfinite(sample.bg_value) &&
                     std::isfinite(sample.weight) && sample.weight > 0.0f;
    }
    samples.push_back(sample);
  }
  return samples;
}

static std::vector<AutoTunePreparedTileSample>
/// @brief Extracts autotune prepared tile samples.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
extract_autotune_prepared_tile_samples(
    const Matrix2Df &channel, const std::vector<TileMetrics> &tile_metrics,
    const TileGrid &tile_grid, const BGEConfig &config) {

  std::vector<AutoTunePreparedTileSample> prepared_samples;
  const int stride = static_cast<int>(channel.cols());
  const float *channel_data = channel.data();

  if (tile_metrics.size() < tile_grid.tiles.size()) {
    std::cout << "[BGE] Warning: tile_metrics smaller than tile_grid, "
                 "truncating to min size"
              << std::endl;
  }

  const size_t n_tiles = std::min(tile_metrics.size(), tile_grid.tiles.size());
  const bool have_common_mask =
      config.common_mask_rows == channel.rows() &&
      config.common_mask_cols == channel.cols() &&
      config.common_valid_mask.size() ==
          static_cast<size_t>(channel.rows() * channel.cols());
  if (!have_common_mask) {
    std::cout << "[BGE] Error: missing/invalid canvas mask in autotune tile "
                 "sampling"
              << std::endl;
    return prepared_samples;
  }

  int informative_metric_tiles = 0;
  for (size_t ti = 0; ti < n_tiles; ++ti) {
    const auto &tm = tile_metrics[ti];
    const bool has_structure = std::isfinite(tm.noise) && tm.noise > 1.0e-6f &&
                               std::isfinite(tm.gradient_energy) &&
                               tm.gradient_energy > 1.0e-6f;
    const bool has_quality =
        std::isfinite(tm.quality_score) && std::abs(tm.quality_score) > 1.0e-3f;
    if (has_structure || has_quality)
      ++informative_metric_tiles;
  }
  const float informative_fraction =
      (n_tiles > 0) ? (static_cast<float>(informative_metric_tiles) /
                       static_cast<float>(n_tiles))
                    : 0.0f;
  const bool use_tile_metrics = (informative_fraction >= 0.35f);

  std::vector<float> structure_scores;
  structure_scores.reserve(n_tiles);
  if (use_tile_metrics) {
    for (size_t ti = 0; ti < n_tiles; ++ti) {
      const auto &tm = tile_metrics[ti];
      if (tm.noise > 1e-6f && std::isfinite(tm.gradient_energy)) {
        structure_scores.push_back(tm.gradient_energy / tm.noise);
      }
    }
  }

  float structure_thresh = 0.0f;
  if (!structure_scores.empty()) {
    structure_thresh = robust_quantile_inplace(
        structure_scores, config.structure_thresh_percentile);
  }

  prepared_samples.resize(n_tiles);
  const int parallel_workers =
      bge_parallel_worker_count(static_cast<int>(n_tiles), 4);

#pragma omp parallel num_threads(parallel_workers) if(parallel_workers > 1)
  {
    TileSampleScratch scratch;
#pragma omp for schedule(dynamic, 1)
    for (int ti = 0; ti < static_cast<int>(n_tiles); ++ti) {
      const size_t t = static_cast<size_t>(ti);
    const auto &tile = tile_grid.tiles[t];
    const auto &tm = tile_metrics[t];

    AutoTunePreparedTileSample prepared;
    prepared.x = tile.x + tile.width / 2.0f;
    prepared.y = tile.y + tile.height / 2.0f;
    prepared.valid = false;

    if (use_tile_metrics && tm.type == TileType::STAR && tm.star_count >= 16) {
      prepared_samples[t] = std::move(prepared);
      continue;
    }
    if (use_tile_metrics && tm.type == TileType::STRUCTURE &&
        std::isfinite(tm.quality_score) && tm.quality_score >= 0.20f) {
      prepared_samples[t] = std::move(prepared);
      continue;
    }

    const float tile_structure =
        (tm.noise > 1e-6f) ? (tm.gradient_energy / tm.noise) : 0.0f;
    if (use_tile_metrics && tile_structure > structure_thresh) {
      prepared_samples[t] = std::move(prepared);
      continue;
    }

    int x0 = tile.x;
    int y0 = tile.y;
    int x1 = std::min(x0 + tile.width, static_cast<int>(channel.cols()));
    int y1 = std::min(y0 + tile.height, static_cast<int>(channel.rows()));

    if (x1 <= x0 || y1 <= y0) {
      prepared_samples[t] = std::move(prepared);
      continue;
    }

    const int tw = x1 - x0;
    const int th = y1 - y0;
    scratch.prepare(tw, th);
    auto tile_value = [&](int yy, int xx) -> float {
      return channel_data[static_cast<size_t>(y0 + yy) *
                              static_cast<size_t>(stride) +
                          static_cast<size_t>(x0 + xx)];
    };

    int supported_px = 0;
    for (int yy = 0; yy < th; ++yy) {
      const int gy = y0 + yy;
      const size_t row_off = static_cast<size_t>(gy) *
                             static_cast<size_t>(config.common_mask_cols);
      for (int xx = 0; xx < tw; ++xx) {
        const int gx = x0 + xx;
        const uint8_t supported =
            config.common_valid_mask[row_off + static_cast<size_t>(gx)] != 0
                ? 1
                : 0;
        scratch.tile_common_support[static_cast<size_t>(yy * tw + xx)] =
            supported;
        supported_px += (supported != 0) ? 1 : 0;
      }
    }

    if (supported_px <= 0) {
      prepared_samples[t] = std::move(prepared);
      continue;
    }

    int zero_pixel_count = 0;
    for (int yy = 0; yy < th; ++yy) {
      const float *row = channel_data +
                         static_cast<size_t>(y0 + yy) *
                             static_cast<size_t>(stride) +
                         x0;
      for (int xx = 0; xx < tw; ++xx) {
        const size_t i = static_cast<size_t>(yy * tw + xx);
        if (scratch.tile_common_support[i] == 0) {
          continue;
        }
        const float v = row[xx];
        if (std::isfinite(v)) {
          scratch.finite_values.push_back(v);
          if (v == 0.0f)
            ++zero_pixel_count;
        }
      }
    }

    if (scratch.finite_values.empty()) {
      prepared_samples[t] = std::move(prepared);
      continue;
    }

    const float zero_fraction = static_cast<float>(zero_pixel_count) /
                                static_cast<float>(scratch.finite_values.size());
    if (zero_fraction > 0.20f) {
      prepared_samples[t] = std::move(prepared);
      continue;
    }

    const float sat_level =
        robust_quantile_inplace(scratch.finite_values, 0.999f);

    const int r_small = 1;
    const int r_large = std::max(2, std::min(tw, th) / 12);
    box_blur_subregion(channel, x0, y0, tw, th, r_small, &scratch.blur_integral,
                       &scratch.blur_small);
    box_blur_subregion(channel, x0, y0, tw, th, r_large, &scratch.blur_integral,
                       &scratch.blur_large);
    for (size_t i = 0; i < scratch.blur_small.size(); ++i) {
      scratch.dog_vals[i] = scratch.blur_small[i] - scratch.blur_large[i];
    }
    const float dog_med = robust_median_inplace(scratch.dog_vals);
    const float dog_mad = robust_mad(scratch.dog_vals, dog_med);
    const float dog_thresh =
        dog_med + 3.0f * std::max(1.4826f * dog_mad, 1.0e-6f);
    const float bright_thresh =
        robust_quantile_inplace(scratch.finite_values, 0.80f);
    for (int yy = 0; yy < th; ++yy) {
      for (int xx = 0; xx < tw; ++xx) {
        const size_t i = static_cast<size_t>(yy * tw + xx);
        const float v = tile_value(yy, xx);
        if (std::isfinite(v) && v >= bright_thresh &&
            scratch.dog_vals[i] > dog_thresh) {
          scratch.star_mask[i] = 1;
        }
      }
    }
    int star_dilate_px = std::max(0, config.mask.star_dilate_px);
    if (std::isfinite(tm.fwhm) && tm.fwhm > 0.0f) {
      const int add = static_cast<int>(std::lround(0.25f * tm.fwhm));
      star_dilate_px = std::clamp(star_dilate_px + std::max(0, add),
                                  star_dilate_px, star_dilate_px + 8);
    }
    dilate_mask_in_place(&scratch.star_mask, tw, th, star_dilate_px,
                         &scratch.dilate_scratch);

    for (int yy = 0; yy < th; ++yy) {
      for (int xx = 0; xx < tw; ++xx) {
        const size_t i = static_cast<size_t>(yy * tw + xx);
        const float v = tile_value(yy, xx);
        if (std::isfinite(v) && v >= sat_level)
          scratch.sat_mask[i] = 1;
      }
    }
    dilate_mask_in_place(&scratch.sat_mask, tw, th,
                         std::max(0, config.mask.sat_dilate_px),
                         &scratch.dilate_scratch);

    scratch.supported_gradients.clear();
    for (int yy = 0; yy < th; ++yy) {
      const int ym = std::max(0, yy - 1);
      const int yp = std::min(th - 1, yy + 1);
      for (int xx = 0; xx < tw; ++xx) {
        const int xm = std::max(0, xx - 1);
        const int xp = std::min(tw - 1, xx + 1);
        const float gx = std::abs(tile_value(yy, xp) - tile_value(yy, xm));
        const float gy = std::abs(tile_value(yp, xx) - tile_value(ym, xx));
        const float grad = gx + gy;
        scratch.tile_gradients[static_cast<size_t>(yy * tw + xx)] = grad;
        if (scratch.tile_common_support[static_cast<size_t>(yy * tw + xx)] !=
            0) {
          scratch.supported_gradients.push_back(grad);
        }
      }
    }
    std::vector<float> &gradient_source =
        scratch.supported_gradients.empty() ? scratch.tile_gradients
                                            : scratch.supported_gradients;
    const float grad_thresh = robust_quantile_inplace(
        gradient_source, config.structure_thresh_percentile);

    scratch.tile_pixels.clear();
    for (int yy = 0; yy < th; ++yy) {
      for (int xx = 0; xx < tw; ++xx) {
        const size_t i = static_cast<size_t>(yy * tw + xx);
        if (scratch.tile_common_support[i] == 0) {
          continue;
        }
        const float v = tile_value(yy, xx);
        const bool structure_bad = scratch.tile_gradients[i] > grad_thresh;
        const bool masked =
            (scratch.star_mask[i] != 0) || (scratch.sat_mask[i] != 0) ||
            structure_bad;
        if (!masked && std::isfinite(v)) {
          scratch.tile_pixels.push_back(v);
        }
      }
    }

    if (scratch.tile_pixels.empty()) {
      prepared_samples[t] = std::move(prepared);
      continue;
    }

    const float usable_fraction =
        static_cast<float>(scratch.tile_pixels.size()) /
        static_cast<float>(std::max(1, supported_px));
    if (!(std::isfinite(usable_fraction)) ||
        usable_fraction < kMinUsableTileFraction) {
      prepared_samples[t] = std::move(prepared);
      continue;
    }

    std::sort(scratch.tile_pixels.begin(), scratch.tile_pixels.end());
    if (!std::isfinite(scratch.tile_pixels.front())) {
      prepared_samples[t] = std::move(prepared);
      continue;
    }

    const float masked_fraction = 1.0f - usable_fraction;
    const float structure_noise =
        estimate_structure_noise_scale(tm, scratch.tile_pixels);
    const float structure_score = compute_structure_score_from_background_mask(
        channel, x0, y0, tw, th, grad_thresh, structure_noise, scratch,
        &scratch);
    prepared.weight =
        std::exp(-config.tile_weight_lambda_structure * structure_score) *
        (1.0f - masked_fraction);
    prepared.weight = std::clamp(prepared.weight, 0.0f, 1.0f);
    if (!(std::isfinite(prepared.weight) && prepared.weight > 0.0f)) {
      prepared_samples[t] = std::move(prepared);
      continue;
    }
    prepared.sorted_pixels = scratch.tile_pixels;
    prepared.valid = true;
    prepared_samples[t] = std::move(prepared);
    }
  }

  return prepared_samples;
}

// Aggregate tiles to coarse grid (v3.3 §6.3.3)
std::vector<GridCell>
/// @brief Implements aggregate to coarse grid.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
aggregate_to_coarse_grid(const std::vector<TileBGSample> &tile_samples,
                         int image_width, int image_height, int grid_spacing,
                         const BGEConfig &config) {

  // Compute grid dimensions
  int n_cells_x = (image_width + grid_spacing - 1) / grid_spacing;
  int n_cells_y = (image_height + grid_spacing - 1) / grid_spacing;

  // Initialize grid cells
  std::vector<std::vector<GridCell>> grid(n_cells_y,
                                          std::vector<GridCell>(n_cells_x));
  for (int cy = 0; cy < n_cells_y; ++cy) {
    for (int cx = 0; cx < n_cells_x; ++cx) {
      auto &cell = grid[cy][cx];
      cell.cell_x = cx;
      cell.cell_y = cy;
      cell.center_x = (cx + 0.5f) * grid_spacing;
      cell.center_y = (cy + 0.5f) * grid_spacing;
      cell.n_samples = 0;
      cell.valid = false;
    }
  }

  const int total_cells = n_cells_x * n_cells_y;
  std::vector<std::vector<TileBGSample>> cell_samples(
      static_cast<size_t>(total_cells));
  std::vector<int> sample_cell_indices(tile_samples.size(), -1);
  std::vector<int> cell_counts(static_cast<size_t>(total_cells), 0);

  // Robustly suppress globally implausible tile background samples before
  // cell aggregation (e.g., bright-object contamination, near-zero artifacts).
  std::vector<float> valid_bg_values;
  valid_bg_values.reserve(tile_samples.size());
  for (const auto &s : tile_samples) {
    if (s.valid && std::isfinite(s.bg_value) &&
        s.bg_value >= config.min_sample_bg_value) {
      valid_bg_values.push_back(s.bg_value);
    }
  }

  float bg_med = 0.0f;
  float bg_sigma = 0.0f;
  bool have_bg_guard = false;
  if (valid_bg_values.size() >= 16) {
    bg_med = robust_median_inplace(valid_bg_values);
    const float mad = robust_mad(valid_bg_values, bg_med);
    bg_sigma = 1.4826f * mad;
    have_bg_guard = std::isfinite(bg_sigma) && bg_sigma > kTiny;
  }

  int n_rejected_global_outliers = 0;
  for (size_t sample_idx = 0; sample_idx < tile_samples.size(); ++sample_idx) {
    const auto &sample = tile_samples[sample_idx];
    if (!sample.valid)
      continue;
    if (!std::isfinite(sample.bg_value))
      continue;
    if (sample.bg_value < config.min_sample_bg_value)
      continue;

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
      const int cell_idx = cy * n_cells_x + cx;
      sample_cell_indices[sample_idx] = cell_idx;
      ++cell_counts[static_cast<size_t>(cell_idx)];
    }
  }

  for (int cell_idx = 0; cell_idx < total_cells; ++cell_idx) {
    cell_samples[static_cast<size_t>(cell_idx)].resize(
        static_cast<size_t>(cell_counts[static_cast<size_t>(cell_idx)]));
  }
  std::vector<int> cell_write_offsets(static_cast<size_t>(total_cells), 0);
  for (size_t sample_idx = 0; sample_idx < tile_samples.size(); ++sample_idx) {
    const int cell_idx = sample_cell_indices[sample_idx];
    if (cell_idx < 0)
      continue;
    auto &samples = cell_samples[static_cast<size_t>(cell_idx)];
    const size_t write_offset =
        static_cast<size_t>(cell_write_offsets[static_cast<size_t>(cell_idx)]++);
    samples[write_offset] = tile_samples[sample_idx];
  }

  if (n_rejected_global_outliers > 0) {
    std::cout << "[BGE]   Rejected global sample outliers: "
              << n_rejected_global_outliers << std::endl;
  }

  // Aggregate per cell (v3.3 §6.3.3c)
  const int parallel_workers = bge_parallel_worker_count(total_cells, 8);
#pragma omp parallel num_threads(parallel_workers) if(parallel_workers > 1)
  {
    std::vector<float> bg_values;
    std::vector<float> weights;
#pragma omp for schedule(static)
    for (int cell_idx = 0; cell_idx < total_cells; ++cell_idx) {
      const int cy = cell_idx / n_cells_x;
      const int cx = cell_idx - cy * n_cells_x;
      auto &cell = grid[cy][cx];
      const auto &samples = cell_samples[static_cast<size_t>(cell_idx)];

      cell.n_samples = static_cast<int>(samples.size());

      if (cell.n_samples < config.min_tiles_per_cell) {
        cell.valid = false;
        continue;
      }

      bg_values.clear();
      weights.clear();
      bg_values.reserve(samples.size());
      weights.reserve(samples.size());

      for (const auto &s : samples) {
        bg_values.push_back(s.bg_value);
        weights.push_back(s.weight);
      }

      cell.bg_value = robust_median_inplace(bg_values);
      cell.weight = robust_median_inplace(weights);
      cell.valid = true;
    }
  }

  // Deterministic handling of insufficient cells (v3.3 §6.3.3d)
  auto is_valid = [&](int cx, int cy) {
    return cy >= 0 && cy < n_cells_y && cx >= 0 && cx < n_cells_x &&
           grid[cy][cx].valid;
  };

  if (config.grid.insufficient_cell_strategy == "nearest") {
    for (int cy = 0; cy < n_cells_y; ++cy) {
      for (int cx = 0; cx < n_cells_x; ++cx) {
        if (grid[cy][cx].valid)
          continue;
        float best_d2 = std::numeric_limits<float>::infinity();
        int best_x = -1;
        int best_y = -1;
        for (int sy = 0; sy < n_cells_y; ++sy) {
          for (int sx = 0; sx < n_cells_x; ++sx) {
            if (!grid[sy][sx].valid)
              continue;
            const float dx = grid[cy][cx].center_x - grid[sy][sx].center_x;
            const float dy = grid[cy][cx].center_y - grid[sy][sx].center_y;
            const float d2 = dx * dx + dy * dy;
            if (d2 < best_d2 ||
                (std::abs(d2 - best_d2) <= 1.0e-6f &&
                 (sy < best_y || (sy == best_y && sx < best_x)))) {
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
        if (grid[cy][cx].valid)
          continue;

        std::vector<float> bg_vals;
        std::vector<float> wt_vals;
        int total_samples = 0;
        for (int r = 1; r <= max_radius; ++r) {
          for (int sy = std::max(0, cy - r);
               sy <= std::min(n_cells_y - 1, cy + r); ++sy) {
            for (int sx = std::max(0, cx - r);
                 sx <= std::min(n_cells_x - 1, cx + r); ++sx) {
              if (std::max(std::abs(sy - cy), std::abs(sx - cx)) != r)
                continue;
              const int sidx = sy * n_cells_x + sx;
              for (const auto &s : cell_samples[sidx]) {
                bg_vals.push_back(s.bg_value);
                wt_vals.push_back(s.weight);
                ++total_samples;
              }
            }
          }
          if (total_samples >= config.min_tiles_per_cell) {
            grid[cy][cx].bg_value = robust_median_inplace(bg_vals);
            grid[cy][cx].weight = robust_median_inplace(wt_vals);
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
struct RBFModelState {
  const std::vector<GridCell> *grid_cells = nullptr;
  Eigen::VectorXf coeffs;
  Eigen::VectorXf poly_coeffs;
  RBFKernelType kernel = RBFKernelType::Multiquadric;
  float mu = kTiny;
  float epsilon = kTiny;
  float lambda = kTiny;
  float poly_x_scale = 1.0f;
  float poly_y_scale = 1.0f;
  float train_rms = std::numeric_limits<float>::infinity();
  bool success = false;
};

struct PolynomialModelState {
  Eigen::VectorXf coeffs;
  int order = 0;
  float x_scale = 0.0f;
  float y_scale = 0.0f;
  float x_offset = -1.0f;
  float y_offset = -1.0f;
  float train_rms = std::numeric_limits<float>::infinity();
  bool success = false;
};

enum class SurfaceModelKind { None, Rbf, Poly };

struct SurfaceModelSelection {
  SurfaceModelKind kind = SurfaceModelKind::None;
  RBFModelState rbf;
  PolynomialModelState poly;
  float rms = std::numeric_limits<float>::infinity();
  bool success = false;
};

float catmull_rom(float p0, float p1, float p2, float p3, float t) {
  const float t2 = t * t;
  const float t3 = t2 * t;
  return 0.5f * ((2.0f * p1) + (-p0 + p2) * t +
                 (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 +
                 (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
}

Matrix2Df render_bicubic_mesh_surface(const std::vector<GridCell> &grid_cells,
                                      int image_width, int image_height,
                                      int grid_spacing,
                                      float *out_rms) {
  const int nx = std::max(1, (image_width + grid_spacing - 1) / grid_spacing);
  const int ny = std::max(1, (image_height + grid_spacing - 1) / grid_spacing);
  std::vector<float> mesh(static_cast<size_t>(nx * ny),
                          std::numeric_limits<float>::quiet_NaN());
  std::vector<uint8_t> valid(static_cast<size_t>(nx * ny), 0);
  for (const auto &gc : grid_cells) {
    if (!gc.valid || !std::isfinite(gc.bg_value))
      continue;
    if (gc.cell_x < 0 || gc.cell_x >= nx || gc.cell_y < 0 || gc.cell_y >= ny)
      continue;
    const size_t idx = static_cast<size_t>(gc.cell_y * nx + gc.cell_x);
    mesh[idx] = gc.bg_value;
    valid[idx] = 1;
  }

  std::vector<int> valid_idx;
  valid_idx.reserve(grid_cells.size());
  for (int i = 0; i < nx * ny; ++i) {
    if (valid[static_cast<size_t>(i)] != 0)
      valid_idx.push_back(i);
  }
  if (valid_idx.size() < 4)
    return Matrix2Df();

  for (int cy = 0; cy < ny; ++cy) {
    for (int cx = 0; cx < nx; ++cx) {
      const size_t idx = static_cast<size_t>(cy * nx + cx);
      if (valid[idx] != 0)
        continue;
      float best_d2 = std::numeric_limits<float>::infinity();
      float best_v = std::numeric_limits<float>::quiet_NaN();
      for (int vi : valid_idx) {
        const int vy = vi / nx;
        const int vx = vi - vy * nx;
        const float dx = static_cast<float>(cx - vx);
        const float dy = static_cast<float>(cy - vy);
        const float d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
          best_d2 = d2;
          best_v = mesh[static_cast<size_t>(vi)];
        }
      }
      mesh[idx] = best_v;
    }
  }

  auto mesh_at = [&](int y, int x) -> float {
    x = std::clamp(x, 0, nx - 1);
    y = std::clamp(y, 0, ny - 1);
    return mesh[static_cast<size_t>(y * nx + x)];
  };

  Matrix2Df surface(image_height, image_width);
  const int total_pixels = std::max(0, image_width * image_height);
  const int parallel_workers = bge_parallel_worker_count(total_pixels, 32768);
#pragma omp parallel for collapse(2) num_threads(parallel_workers) \
    if(parallel_workers > 1)
  for (int y = 0; y < image_height; ++y) {
    for (int x = 0; x < image_width; ++x) {
      const float gx =
          (static_cast<float>(x) + 0.5f) / static_cast<float>(grid_spacing) -
          0.5f;
      const float gy =
          (static_cast<float>(y) + 0.5f) / static_cast<float>(grid_spacing) -
          0.5f;
      const int ix = static_cast<int>(std::floor(gx));
      const int iy = static_cast<int>(std::floor(gy));
      const float tx = gx - static_cast<float>(ix);
      const float ty = gy - static_cast<float>(iy);
      float row_vals[4];
      for (int ky = -1; ky <= 2; ++ky) {
        const float p0 = mesh_at(iy + ky, ix - 1);
        const float p1 = mesh_at(iy + ky, ix);
        const float p2 = mesh_at(iy + ky, ix + 1);
        const float p3 = mesh_at(iy + ky, ix + 2);
        row_vals[ky + 1] = catmull_rom(p0, p1, p2, p3, tx);
      }
      surface(y, x) =
          catmull_rom(row_vals[0], row_vals[1], row_vals[2], row_vals[3], ty);
    }
  }

  if (out_rms != nullptr) {
    double sum_sq = 0.0;
    int n = 0;
    for (const auto &gc : grid_cells) {
      if (!gc.valid)
        continue;
      const int px =
          std::clamp(static_cast<int>(std::lround(gc.center_x)), 0,
                     image_width - 1);
      const int py =
          std::clamp(static_cast<int>(std::lround(gc.center_y)), 0,
                     image_height - 1);
      const float pred = surface(py, px);
      if (!std::isfinite(pred))
        continue;
      const double r = static_cast<double>(gc.bg_value - pred);
      sum_sq += r * r;
      ++n;
    }
    *out_rms = (n > 0) ? static_cast<float>(
                             std::sqrt(sum_sq / static_cast<double>(n)))
                       : std::numeric_limits<float>::infinity();
  }
  return surface;
}

/// @brief Implements solve rbf model.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool solve_rbf_model(const std::vector<GridCell> &grid_cells, int grid_spacing,
                     const BGEConfig &config, RBFModelState *out) {
  const int M = grid_cells.size();
  if (M < 3) {
    std::cout << "[BGE] Too few grid cells for RBF: " << M << std::endl;
    return false;
  }

  // Compute mu (shape parameter, v3.3 §6.3.7)
  const int G = std::max(1, grid_spacing);
  const float mu =
      std::max(kTiny, config.fit.rbf_mu_factor * static_cast<float>(G));
  const float epsilon = std::max(kTiny, config.fit.rbf_epsilon);
  const float lambda_base = std::max(kTiny, config.fit.rbf_lambda);
  const RBFKernelType kernel = resolve_rbf_kernel_type(config.fit.rbf_phi);

  // Build RBF matrix Phi (M x M)
  Eigen::MatrixXf Phi_base(M, M);
  for (int i = 0; i < M; ++i) {
    Phi_base(i, i) = evaluate_rbf_kernel(kernel, 0.0f, mu, epsilon);
    for (int j = i + 1; j < M; ++j) {
      float dx = grid_cells[i].center_x - grid_cells[j].center_x;
      float dy = grid_cells[i].center_y - grid_cells[j].center_y;
      float d = std::sqrt(dx * dx + dy * dy);
      const float phi = evaluate_rbf_kernel(kernel, d, mu, epsilon);
      Phi_base(i, j) = phi;
      Phi_base(j, i) = phi;
    }
  }

  // Build target vector b (M x 1)
  Eigen::VectorXf b(M);
  Eigen::VectorXf w_rel(M);
  std::vector<float> bg_values;
  bg_values.reserve(static_cast<size_t>(M));
  const auto robust_weight_fn =
      resolve_robust_weight_fn(config.fit.robust_loss);
  float min_x = std::numeric_limits<float>::infinity();
  float max_x = -std::numeric_limits<float>::infinity();
  float min_y = std::numeric_limits<float>::infinity();
  float max_y = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < M; ++i) {
    b(i) = grid_cells[i].bg_value;
    w_rel(i) = std::max(1.0e-3f, grid_cells[i].weight);
    bg_values.push_back(grid_cells[i].bg_value);
    min_x = std::min(min_x, grid_cells[i].center_x);
    max_x = std::max(max_x, grid_cells[i].center_x);
    min_y = std::min(min_y, grid_cells[i].center_y);
    max_y = std::max(max_y, grid_cells[i].center_y);
  }
  const float poly_x_scale =
      2.0f / std::max(1.0f, max_x - min_x + static_cast<float>(G));
  const float poly_y_scale =
      2.0f / std::max(1.0f, max_y - min_y + static_cast<float>(G));

  auto solve_rbf_coeffs = [&](float lambda, Eigen::VectorXf *out_u,
                              Eigen::VectorXf *out_poly,
                              float *out_rms) -> bool {
    Eigen::VectorXf u = Eigen::VectorXf::Zero(M);
    Eigen::VectorXf poly = Eigen::VectorXf::Zero(3);
    Eigen::VectorXf w_rob = Eigen::VectorXf::Ones(M);

    Eigen::MatrixXf P(M, 3);
    for (int i = 0; i < M; ++i) {
      P(i, 0) = 1.0f;
      P(i, 1) = grid_cells[i].center_x * poly_x_scale;
      P(i, 2) = grid_cells[i].center_y * poly_y_scale;
    }

    for (int iter = 0; iter < std::max(1, config.fit.irls_max_iterations);
         ++iter) {
      Eigen::VectorXf w = w_rel.cwiseProduct(w_rob).cwiseMax(1.0e-6f);

      Eigen::MatrixXf lhs = Eigen::MatrixXf::Zero(M + 3, M + 3);
      lhs.block(0, 0, M, M) = Phi_base;
      for (int i = 0; i < M; ++i) {
        lhs(i, i) += lambda / std::max(w(i), 1.0e-6f);
      }
      lhs.block(0, M, M, 3) = P;
      lhs.block(M, 0, 3, M) = P.transpose();

      Eigen::VectorXf rhs = Eigen::VectorXf::Zero(M + 3);
      rhs.segment(0, M) = b;
      Eigen::VectorXf solution = lhs.fullPivLu().solve(rhs);

      if (!solution.allFinite()) {
        return false;
      }

      Eigen::VectorXf u_new = solution.segment(0, M);
      Eigen::VectorXf poly_new = solution.segment(M, 3);
      const float step = (u_new - u).norm() + (poly_new - poly).norm();
      const float scale = 1.0f + u.norm() + poly.norm();
      u = std::move(u_new);
      poly = std::move(poly_new);

      Eigen::VectorXf residual = b - (Phi_base * u + P * poly);
      update_robust_weights(residual, config.fit.huber_delta, robust_weight_fn,
                            &w_rob);

      if (step <= config.fit.irls_tolerance * scale)
        break;
    }

    Eigen::VectorXf residual = b - (Phi_base * u + P * poly);
    const float rms = std::sqrt(residual.squaredNorm() / static_cast<float>(M));
    if (!(std::isfinite(rms))) {
      return false;
    }

    *out_u = std::move(u);
    *out_poly = std::move(poly);
    *out_rms = rms;
    return true;
  };

  // Dynamic lambda adaptation: test/adjust/test and prefer the smoothest
  // (highest lambda) model that still fits grid samples well enough.
  const float bg_med = robust_median_inplace(bg_values);
  const float bg_sigma = 1.4826f * robust_mad(bg_values, bg_med);
  const float residual_limit =
      std::max(0.15f, 0.20f * std::max(bg_sigma, kTiny));

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
  Eigen::VectorXf best_poly = Eigen::VectorXf::Zero(3);
  bool have_best = false;

  float accepted_lambda = -1.0f;
  float accepted_rms = std::numeric_limits<float>::infinity();
  Eigen::VectorXf accepted_u = Eigen::VectorXf::Zero(M);
  Eigen::VectorXf accepted_poly = Eigen::VectorXf::Zero(3);

  for (float lambda_try : lambda_trials) {
    Eigen::VectorXf u_try = Eigen::VectorXf::Zero(M);
    Eigen::VectorXf poly_try = Eigen::VectorXf::Zero(3);
    float rms_try = std::numeric_limits<float>::infinity();
    const bool ok =
        solve_rbf_coeffs(lambda_try, &u_try, &poly_try, &rms_try);
    if (!ok) {
      std::cout << "[BGE]   RBF lambda=" << lambda_try << " fit failed"
                << std::endl;
      continue;
    }

    std::cout << "[BGE]   RBF lambda=" << lambda_try << " trial RMS=" << rms_try
              << std::endl;

    if (!have_best || rms_try < best_rms) {
      best_lambda = lambda_try;
      best_rms = rms_try;
      best_u = u_try;
      best_poly = poly_try;
      have_best = true;
    }

    if (rms_try <= residual_limit) {
      accepted_lambda = lambda_try;
      accepted_rms = rms_try;
      accepted_u = u_try;
      accepted_poly = poly_try;
    }
  }

  if (!have_best) {
    std::cerr << "[BGE] RBF solve failed for all lambda trials" << std::endl;
    return false;
  }

  float lambda = best_lambda;
  float rms_selected = best_rms;
  Eigen::VectorXf u = best_u;
  Eigen::VectorXf poly = best_poly;
  if (accepted_lambda > 0.0f) {
    lambda = accepted_lambda;
    rms_selected = accepted_rms;
    u = accepted_u;
    poly = accepted_poly;
  }

  std::cout << "[BGE]   RBF selected lambda=" << lambda
            << " (limit=" << residual_limit << ", rms=" << rms_selected << ")"
            << std::endl;

  out->grid_cells = &grid_cells;
  out->coeffs = std::move(u);
  out->poly_coeffs = std::move(poly);
  out->kernel = kernel;
  out->mu = mu;
  out->epsilon = epsilon;
  out->lambda = lambda;
  out->poly_x_scale = poly_x_scale;
  out->poly_y_scale = poly_y_scale;
  out->train_rms = rms_selected;
  out->success = true;
  return true;
}

/// @brief Implements eval rbf model at.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float eval_rbf_model_at(const RBFModelState &state, float x, float y) {
  if (!state.success || state.grid_cells == nullptr)
    return std::numeric_limits<float>::quiet_NaN();
  float sum = 0.0f;
  const auto &grid_cells = *state.grid_cells;
  for (int i = 0; i < state.coeffs.size(); ++i) {
    const float dx = x - grid_cells[static_cast<size_t>(i)].center_x;
    const float dy = y - grid_cells[static_cast<size_t>(i)].center_y;
    const float d = std::sqrt(dx * dx + dy * dy);
    sum += state.coeffs(i) *
           evaluate_rbf_kernel(state.kernel, d, state.mu, state.epsilon);
  }
  if (state.poly_coeffs.size() >= 3) {
    sum += state.poly_coeffs(0) + state.poly_coeffs(1) * x * state.poly_x_scale +
           state.poly_coeffs(2) * y * state.poly_y_scale;
  }
  return sum;
}

/// @brief Implements render rbf surface.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Matrix2Df render_rbf_surface(const RBFModelState &state, int image_width,
                             int image_height) {
  Matrix2Df surface = Matrix2Df::Zero(image_height, image_width);
  const int total_pixels = std::max(0, image_width * image_height);
  const int parallel_workers = bge_parallel_worker_count(total_pixels, 32768);
#pragma omp parallel for collapse(2) num_threads(parallel_workers) \
    if(parallel_workers > 1)
  for (int y = 0; y < image_height; ++y) {
    for (int x = 0; x < image_width; ++x) {
      surface(y, x) = eval_rbf_model_at(state, static_cast<float>(x),
                                        static_cast<float>(y));
    }
  }
  return surface;
}

// Polynomial surface fitting (v3.3 §6.3.7)
/// @brief Implements solve polynomial model.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool solve_polynomial_model(const std::vector<GridCell> &grid_cells,
                            int image_width, int image_height,
                            const BGEConfig &config,
                            PolynomialModelState *out) {
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
    std::cout << "[BGE] Too few grid cells for polynomial order " << order
              << ": " << M << " < " << n_terms << std::endl;
    return false;
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
  std::vector<float> x_pows(static_cast<size_t>(order + 1), 1.0f);
  std::vector<float> y_pows(static_cast<size_t>(order + 1), 1.0f);

  for (int i = 0; i < M; ++i) {
    float x_norm = grid_cells[i].center_x * x_scale + x_offset;
    float y_norm = grid_cells[i].center_y * y_scale + y_offset;

    x_pows[0] = 1.0f;
    y_pows[0] = 1.0f;
    for (int p = 1; p <= order; ++p) {
      x_pows[static_cast<size_t>(p)] =
          x_pows[static_cast<size_t>(p - 1)] * x_norm;
      y_pows[static_cast<size_t>(p)] =
          y_pows[static_cast<size_t>(p - 1)] * y_norm;
    }

    int col = 0;
    for (int m = 0; m <= order; ++m) {
      for (int n = 0; n <= order - m; ++n) {
        A(i, col) =
            x_pows[static_cast<size_t>(m)] * y_pows[static_cast<size_t>(n)];
        ++col;
      }
    }

    b(i) = grid_cells[i].bg_value;
    w_rel(i) = std::max(1.0e-3f, grid_cells[i].weight);
  }
  const auto robust_weight_fn =
      resolve_robust_weight_fn(config.fit.robust_loss);

  // Robust IRLS polynomial fitting (v3.3 §6.3.7)
  Eigen::VectorXf coeffs = Eigen::VectorXf::Zero(n_terms);
  Eigen::VectorXf w_rob = Eigen::VectorXf::Ones(M);
  for (int iter = 0; iter < std::max(1, config.fit.irls_max_iterations);
       ++iter) {
    Eigen::VectorXf w = w_rel.cwiseProduct(w_rob).cwiseMax(1.0e-6f);
    Eigen::MatrixXf W = w.asDiagonal();
    Eigen::MatrixXf lhs = A.transpose() * W * A;
    Eigen::VectorXf rhs = A.transpose() * W * b;
    Eigen::VectorXf coeffs_new = lhs.ldlt().solve(rhs);

    const float step = (coeffs_new - coeffs).norm();
    const float scale = 1.0f + coeffs.norm();
    coeffs = coeffs_new;

    Eigen::VectorXf residual = b - A * coeffs;
    update_robust_weights(residual, config.fit.huber_delta, robust_weight_fn,
                          &w_rob);

    if (step <= config.fit.irls_tolerance * scale) {
      break;
    }
  }

  const Eigen::VectorXf residual = b - A * coeffs;
  const float train_rms =
      std::sqrt(residual.squaredNorm() / static_cast<float>(M));
  if (!(std::isfinite(train_rms))) {
    return false;
  }

  out->coeffs = std::move(coeffs);
  out->order = order;
  out->x_scale = x_scale;
  out->y_scale = y_scale;
  out->x_offset = x_offset;
  out->y_offset = y_offset;
  out->train_rms = train_rms;
  out->success = true;
  return true;
}

/// @brief Implements eval polynomial model at.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float eval_polynomial_model_at(const PolynomialModelState &state, float x,
                               float y, std::vector<float> *x_pows,
                               std::vector<float> *y_pows) {
  if (!state.success)
    return std::numeric_limits<float>::quiet_NaN();
  const float x_norm = x * state.x_scale + state.x_offset;
  const float y_norm = y * state.y_scale + state.y_offset;
  x_pows->assign(static_cast<size_t>(state.order + 1), 1.0f);
  y_pows->assign(static_cast<size_t>(state.order + 1), 1.0f);
  for (int p = 1; p <= state.order; ++p) {
    (*x_pows)[static_cast<size_t>(p)] =
        (*x_pows)[static_cast<size_t>(p - 1)] * x_norm;
    (*y_pows)[static_cast<size_t>(p)] =
        (*y_pows)[static_cast<size_t>(p - 1)] * y_norm;
  }

  float sum = 0.0f;
  int col = 0;
  for (int m = 0; m <= state.order; ++m) {
    for (int n = 0; n <= state.order - m; ++n) {
      sum += state.coeffs(col) * (*x_pows)[static_cast<size_t>(m)] *
             (*y_pows)[static_cast<size_t>(n)];
      ++col;
    }
  }
  return sum;
}

/// @brief Implements render polynomial surface.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Matrix2Df render_polynomial_surface(const PolynomialModelState &state,
                                    int image_width, int image_height) {
  Matrix2Df surface = Matrix2Df::Zero(image_height, image_width);
  std::vector<std::vector<float>> x_power_table(
      static_cast<size_t>(image_width),
      std::vector<float>(static_cast<size_t>(state.order + 1), 1.0f));
  std::vector<std::vector<float>> y_power_table(
      static_cast<size_t>(image_height),
      std::vector<float>(static_cast<size_t>(state.order + 1), 1.0f));

  for (int x = 0; x < image_width; ++x) {
    const float x_norm = x * state.x_scale + state.x_offset;
    auto &xp = x_power_table[static_cast<size_t>(x)];
    xp[0] = 1.0f;
    for (int p = 1; p <= state.order; ++p) {
      xp[static_cast<size_t>(p)] = xp[static_cast<size_t>(p - 1)] * x_norm;
    }
  }
  for (int y = 0; y < image_height; ++y) {
    const float y_norm = y * state.y_scale + state.y_offset;
    auto &yp = y_power_table[static_cast<size_t>(y)];
    yp[0] = 1.0f;
    for (int p = 1; p <= state.order; ++p) {
      yp[static_cast<size_t>(p)] = yp[static_cast<size_t>(p - 1)] * y_norm;
    }
  }

  const int total_pixels = std::max(0, image_width * image_height);
  const int parallel_workers = bge_parallel_worker_count(total_pixels, 32768);
#pragma omp parallel for collapse(2) num_threads(parallel_workers) \
    if(parallel_workers > 1)
  for (int y = 0; y < image_height; ++y) {
    for (int x = 0; x < image_width; ++x) {
      const auto &xp = x_power_table[static_cast<size_t>(x)];
      const auto &yp = y_power_table[static_cast<size_t>(y)];

      float sum = 0.0f;
      int col = 0;
      for (int m = 0; m <= state.order; ++m) {
        for (int n = 0; n <= state.order - m; ++n) {
          sum += state.coeffs(col) * xp[static_cast<size_t>(m)] *
                 yp[static_cast<size_t>(n)];
          ++col;
        }
      }
      surface(y, x) = sum;
    }
  }

  return surface;
}

/// @brief Implements select background surface model.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool select_background_surface_model(const std::vector<GridCell> &grid_cells,
                                     int image_width, int image_height,
                                     int grid_spacing, const BGEConfig &config,
                                     SurfaceModelSelection *out,
                                     std::string *error_message) {
  out->kind = SurfaceModelKind::None;
  out->success = false;
  out->rms = std::numeric_limits<float>::infinity();
  const std::string method = config.fit.method;

  auto set_error = [&](const std::string &msg) {
    if (error_message != nullptr)
      *error_message = msg;
  };

  if (method == "rbf") {
    RBFModelState rbf_state;
    const bool have_rbf =
        solve_rbf_model(grid_cells, grid_spacing, config, &rbf_state) &&
        std::isfinite(rbf_state.train_rms);
    if (have_rbf) {
      out->kind = SurfaceModelKind::Rbf;
      out->rbf = rbf_state;
      out->rms = rbf_state.train_rms;
      out->success = true;
    }

    constexpr float kRbfFallbackRmsThreshold = 0.25f;
    if (!have_rbf || rbf_state.train_rms > kRbfFallbackRmsThreshold) {
      BGEConfig poly_cfg = config;
      poly_cfg.fit.method = "poly";
      poly_cfg.fit.polynomial_order =
          std::clamp(poly_cfg.fit.polynomial_order, 2, 3);
      PolynomialModelState poly_state;
      const bool have_poly = solve_polynomial_model(
          grid_cells, image_width, image_height, poly_cfg, &poly_state);
      if (have_poly &&
          (!have_rbf || poly_state.train_rms <= rbf_state.train_rms * 1.05f)) {
        std::cout << "[BGE]   RBF fallback -> poly(order="
                  << poly_cfg.fit.polynomial_order
                  << ") rms=" << poly_state.train_rms << " (rbf rms="
                  << rbf_state.train_rms << ")" << std::endl;
        out->kind = SurfaceModelKind::Poly;
        out->poly = std::move(poly_state);
        out->rms = out->poly.train_rms;
        out->success = true;
      }
    }

    if (!out->success) {
      set_error("RBF solve failed");
      return false;
    }
    return true;
  }

  if (method == "spline") {
    BGEConfig spline_cfg = config;
    spline_cfg.fit.method = "rbf";
    spline_cfg.fit.rbf_phi = "thinplate";
    spline_cfg.fit.rbf_lambda = std::max(1.0e-4f, spline_cfg.fit.rbf_lambda);
    if (!solve_rbf_model(grid_cells, grid_spacing, spline_cfg, &out->rbf)) {
      set_error("Spline fit failed");
      return false;
    }
    out->kind = SurfaceModelKind::Rbf;
    out->rms = out->rbf.train_rms;
    out->success = true;
    return true;
  }

  if (method == "poly") {
    BGEConfig poly_cfg = config;
    if (!solve_polynomial_model(grid_cells, image_width, image_height,
                                poly_cfg, &out->poly)) {
      set_error("Polynomial fit failed");
      return false;
    }
    out->kind = SurfaceModelKind::Poly;
    out->rms = out->poly.train_rms;
    out->success = true;
    return true;
  }

  set_error("Unsupported fit method: " + method);
  return false;
}

// RBF interpolation (v3.3 §6.3.7)
/// @brief Implements fit rbf surface.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Matrix2Df fit_rbf_surface(const std::vector<GridCell> &grid_cells,
                          int image_width, int image_height, int grid_spacing,
                          const BGEConfig &config) {
  RBFModelState state;
  if (!solve_rbf_model(grid_cells, grid_spacing, config, &state)) {
    return Matrix2Df::Zero(image_height, image_width);
  }
  return render_rbf_surface(state, image_width, image_height);
}

// Polynomial surface fitting (v3.3 §6.3.7)
/// @brief Implements fit polynomial surface.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Matrix2Df fit_polynomial_surface(const std::vector<GridCell> &grid_cells,
                                 int image_width, int image_height,
                                 const BGEConfig &config) {
  PolynomialModelState state;
  if (!solve_polynomial_model(grid_cells, image_width, image_height, config,
                              &state)) {
    return Matrix2Df::Zero(image_height, image_width);
  }
  return render_polynomial_surface(state, image_width, image_height);
}

// Fit background surface (v3.3 §6.3.7)
/// @brief Implements fit background surface.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
BackgroundModel fit_background_surface(const std::vector<GridCell> &grid_cells,
                                       int image_width, int image_height,
                                       int grid_spacing,
                                       const BGEConfig &config) {

  BackgroundModel result;
  result.grid_cells = grid_cells;
  result.n_valid_cells = grid_cells.size();
  result.success = false;

  if (grid_cells.size() < 3) {
    result.error_message =
        "Too few valid grid cells: " + std::to_string(grid_cells.size());
    return result;
  }

  try {
    const auto total_start = SteadyClock::now();
    if (config.fit.method == "bicubic") {
      const auto render_start = SteadyClock::now();
      result.model = render_bicubic_mesh_surface(
          grid_cells, image_width, image_height, grid_spacing,
          &result.rms_residual);
      result.render_seconds = elapsed_seconds_since(render_start);
      result.fit_select_seconds = 0.0;
      result.total_seconds = elapsed_seconds_since(total_start);
      if (result.model.size() == 0 || !result.model.allFinite() ||
          !std::isfinite(result.rms_residual)) {
        result.error_message = "Bicubic mesh interpolation failed";
        return result;
      }
      result.success = true;
      return result;
    }

    SurfaceModelSelection selection;
    const auto select_start = SteadyClock::now();
    if (!select_background_surface_model(grid_cells, image_width, image_height,
                                         grid_spacing, config, &selection,
                                         &result.error_message)) {
      result.fit_select_seconds = elapsed_seconds_since(select_start);
      result.total_seconds = elapsed_seconds_since(total_start);
      return result;
    }
    result.fit_select_seconds = elapsed_seconds_since(select_start);

    const auto render_start = SteadyClock::now();
    if (selection.kind == SurfaceModelKind::Rbf) {
      result.model =
          render_rbf_surface(selection.rbf, image_width, image_height);
    } else if (selection.kind == SurfaceModelKind::Poly) {
      result.model =
          render_polynomial_surface(selection.poly, image_width, image_height);
    } else {
      result.error_message = "Surface fit did not select a model";
      result.render_seconds = elapsed_seconds_since(render_start);
      result.total_seconds = elapsed_seconds_since(total_start);
      return result;
    }
    result.render_seconds = elapsed_seconds_since(render_start);

    if (!result.model.allFinite()) {
      result.error_message = "Surface fit produced non-finite residuals";
      result.total_seconds = elapsed_seconds_since(total_start);
      return result;
    }

    result.rms_residual = selection.rms;
    result.total_seconds = elapsed_seconds_since(total_start);
    result.success = true;

  } catch (const std::exception &e) {
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
  float sample_spread = 0.0f;
  float surface_spread = 0.0f;
  double total_seconds = 0.0;
  double model_select_seconds = 0.0;
  double surface_sample_seconds = 0.0;
  double metric_seconds = 0.0;
  bool success = false;
};

/// @brief Implements deterministic split indices.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static void deterministic_split_indices(int n, float holdout_fraction,
                                        std::vector<int> *train_idx,
                                        std::vector<int> *val_idx) {
  train_idx->clear();
  val_idx->clear();
  if (n <= 0)
    return;

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
    if (!train_idx->empty())
      train_idx->pop_back();
  }
  if (train_idx->empty()) {
    train_idx->push_back(val_idx->back());
    val_idx->pop_back();
    if (val_idx->empty() && n > 1)
      val_idx->push_back(0);
  }
}

template <typename EvalFn>
/// @brief Implements eval predictor rms at cells.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static float eval_predictor_rms_at_cells(const std::vector<GridCell> &cells,
                                         EvalFn &&eval_fn) {
  if (cells.empty())
    return std::numeric_limits<float>::infinity();
  double sum_sq = 0.0;
  int n = 0;
  for (const auto &c : cells) {
    const float predicted = eval_fn(c.center_x, c.center_y);
    if (!std::isfinite(predicted))
      continue;
    const double r = static_cast<double>(c.bg_value - predicted);
    sum_sq += r * r;
    ++n;
  }
  if (n <= 0)
    return std::numeric_limits<float>::infinity();
  return static_cast<float>(std::sqrt(sum_sq / static_cast<double>(n)));
}

/// @brief Implements eval model flatness.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static float eval_model_flatness(const Matrix2Df &model, int step) {
  step = std::max(1, step);
  std::vector<float> grad_energy;
  grad_energy.reserve(static_cast<size_t>((model.rows() / step + 1) *
                                          (model.cols() / step + 1)));
  for (int y = 0; y + step < model.rows(); y += step) {
    for (int x = 0; x + step < model.cols(); x += step) {
      const float v = model(y, x);
      const float vx = model(y, x + step);
      const float vy = model(y + step, x);
      if (!(std::isfinite(v) && std::isfinite(vx) && std::isfinite(vy)))
        continue;
      const float gx = (vx - v) / static_cast<float>(step);
      const float gy = (vy - v) / static_cast<float>(step);
      grad_energy.push_back(gx * gx + gy * gy);
    }
  }
  if (grad_energy.size() < 8)
    return std::numeric_limits<float>::infinity();
  return robust_median_inplace(grad_energy);
}

/// @brief Implements eval model roughness.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static float eval_model_roughness(const Matrix2Df &model, int step) {
  step = std::max(1, step);
  std::vector<float> curvature_energy;
  curvature_energy.reserve(static_cast<size_t>((model.rows() / step + 1) *
                                               (model.cols() / step + 1)));
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
      curvature_energy.push_back(
          std::sqrt(dxx * dxx + dyy * dyy + 2.0f * dxy * dxy));
    }
  }
  if (curvature_energy.size() < 8)
    return std::numeric_limits<float>::infinity();
  return robust_median_inplace(curvature_energy);
}

static float robust_p10_p90_spread(std::vector<float> values) {
  values.erase(std::remove_if(values.begin(), values.end(),
                              [](float v) { return !std::isfinite(v); }),
               values.end());
  if (values.size() < 8)
    return 0.0f;
  std::sort(values.begin(), values.end());
  return sorted_quantile(values, 0.90f) - sorted_quantile(values, 0.10f);
}

static float matrix_p10_p90_spread(const Matrix2Df &model) {
  std::vector<float> values;
  values.reserve(static_cast<size_t>(model.size()));
  for (int i = 0; i < model.size(); ++i) {
    const float v = model.data()[i];
    if (std::isfinite(v))
      values.push_back(v);
  }
  return robust_p10_p90_spread(std::move(values));
}

struct SparseEvalGrid {
  int sample_step = 1;
  int width = 0;
  int height = 0;
  std::vector<float> x_coords;
  std::vector<float> y_coords;
};

/// @brief Builds sparse eval grid.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static SparseEvalGrid build_sparse_eval_grid(int image_width, int image_height,
                                             int sample_step) {
  SparseEvalGrid grid;
  grid.sample_step = std::max(1, sample_step);
  grid.width = std::max(1, (image_width + grid.sample_step - 1) / grid.sample_step);
  grid.height =
      std::max(1, (image_height + grid.sample_step - 1) / grid.sample_step);
  grid.x_coords.resize(static_cast<size_t>(grid.width));
  grid.y_coords.resize(static_cast<size_t>(grid.height));
  for (int x = 0; x < grid.width; ++x) {
    grid.x_coords[static_cast<size_t>(x)] =
        static_cast<float>(std::min(image_width - 1, x * grid.sample_step));
  }
  for (int y = 0; y < grid.height; ++y) {
    grid.y_coords[static_cast<size_t>(y)] =
        static_cast<float>(std::min(image_height - 1, y * grid.sample_step));
  }
  return grid;
}

struct BGECandidatePrep {
  bool valid = false;
  std::vector<GridCell> train_cells;
  std::vector<GridCell> val_cells;
  float bg_median = kTiny;
  float tile_bg_spread = 0.0f;
  float grid_bg_spread = 0.0f;
};

/// @brief Builds bge candidate prep.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static BGECandidatePrep build_bge_candidate_prep(
    const std::vector<AutoTunePreparedTileSample> &prepared_tiles,
    float sample_quantile, int image_width, int image_height,
    const BGEConfig &cfg_try, int grid_spacing) {
  BGECandidatePrep prep;

  std::vector<TileBGSample> tile_samples;
  std::vector<float> tile_bg_values;
  tile_samples.reserve(prepared_tiles.size());
  tile_bg_values.reserve(prepared_tiles.size());
  for (const auto &prepared : prepared_tiles) {
    TileBGSample sample{};
    sample.x = prepared.x;
    sample.y = prepared.y;
    sample.valid = false;
    if (prepared.valid && !prepared.sorted_pixels.empty()) {
      BGEConfig sample_cfg = cfg_try;
      sample_cfg.sample_quantile = sample_quantile;
      sample.bg_value =
          estimate_tile_background_from_sorted(prepared.sorted_pixels,
                                               sample_cfg);
      sample.weight = prepared.weight;
      sample.valid =
          std::isfinite(sample.bg_value) &&
          std::isfinite(sample.weight) && sample.weight > 0.0f;
      if (sample.valid)
        tile_bg_values.push_back(sample.bg_value);
    }
    tile_samples.push_back(sample);
  }
  prep.tile_bg_spread = robust_p10_p90_spread(std::move(tile_bg_values));

  auto grid_cells_all = aggregate_to_coarse_grid(
      tile_samples, image_width, image_height, grid_spacing, cfg_try);

  std::vector<GridCell> cells;
  cells.reserve(grid_cells_all.size());
  for (const auto &gc : grid_cells_all) {
    if (gc.valid)
      cells.push_back(gc);
  }
  if (cells.size() < 6)
    return prep;

  std::sort(cells.begin(), cells.end(),
            [](const GridCell &a, const GridCell &b) {
              if (a.cell_y != b.cell_y)
                return a.cell_y < b.cell_y;
              return a.cell_x < b.cell_x;
            });

  std::vector<int> train_idx;
  std::vector<int> val_idx;
  deterministic_split_indices(static_cast<int>(cells.size()),
                              cfg_try.autotune.holdout_fraction, &train_idx,
                              &val_idx);

  prep.train_cells.reserve(train_idx.size());
  prep.val_cells.reserve(val_idx.size());
  for (int i : train_idx)
    prep.train_cells.push_back(cells[static_cast<size_t>(i)]);
  for (int i : val_idx)
    prep.val_cells.push_back(cells[static_cast<size_t>(i)]);

  std::vector<float> bvals;
  bvals.reserve(cells.size());
  for (const auto &gc : cells)
    bvals.push_back(gc.bg_value);
  prep.grid_bg_spread = robust_p10_p90_spread(bvals);
  prep.bg_median = robust_median_inplace(bvals);
  prep.valid = !prep.train_cells.empty() && !prep.val_cells.empty() &&
               std::isfinite(prep.bg_median);
  return prep;
}

static BGECandidateResult
/// @brief Implements try bge candidate prepared.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
try_bge_candidate_prepared(int image_width, int image_height,
                           const BGEConfig &cfg_try, int grid_spacing,
                           const BGECandidatePrep &prep,
                           const SparseEvalGrid &eval_grid) {
  const auto total_start = SteadyClock::now();
  BGECandidateResult out;
  out.cfg = cfg_try;
  out.grid_spacing = grid_spacing;
  if (!prep.valid)
    return out;

  SurfaceModelSelection selection;
  std::string error_message;
  const auto select_start = SteadyClock::now();
  if (!select_background_surface_model(prep.train_cells, image_width,
                                       image_height, grid_spacing, cfg_try,
                                       &selection, &error_message)) {
    out.model_select_seconds = elapsed_seconds_since(select_start);
    out.total_seconds = elapsed_seconds_since(total_start);
    return out;
  }
  out.model_select_seconds = elapsed_seconds_since(select_start);

  std::vector<float> poly_x_pows;
  std::vector<float> poly_y_pows;
  auto eval_selected_model = [&](float x, float y) -> float {
    if (selection.kind == SurfaceModelKind::Rbf) {
      return eval_rbf_model_at(selection.rbf, x, y);
    }
    if (selection.kind == SurfaceModelKind::Poly) {
      return eval_polynomial_model_at(selection.poly, x, y, &poly_x_pows,
                                      &poly_y_pows);
    }
    return std::numeric_limits<float>::quiet_NaN();
  };

  out.cv_rms =
      eval_predictor_rms_at_cells(prep.val_cells, eval_selected_model);

  const auto sample_start = SteadyClock::now();
  Matrix2Df sampled_surface = Matrix2Df::Zero(eval_grid.height, eval_grid.width);
  for (int y = 0; y < eval_grid.height; ++y) {
    for (int x = 0; x < eval_grid.width; ++x) {
      sampled_surface(y, x) =
          eval_selected_model(eval_grid.x_coords[static_cast<size_t>(x)],
                              eval_grid.y_coords[static_cast<size_t>(y)]);
    }
  }
  out.surface_sample_seconds = elapsed_seconds_since(sample_start);

  const float scale = static_cast<float>(eval_grid.sample_step) *
                      static_cast<float>(eval_grid.sample_step);
  const auto metric_start = SteadyClock::now();
  out.flatness = eval_model_flatness(sampled_surface, 1) / std::max(1.0f, scale);
  out.roughness =
      eval_model_roughness(sampled_surface, 1) / std::max(1.0f, scale);
  out.surface_spread = matrix_p10_p90_spread(sampled_surface);
  out.sample_spread = std::max(prep.tile_bg_spread, prep.grid_bg_spread);
  out.metric_seconds = elapsed_seconds_since(metric_start);

  const float bmed = std::max(kTiny, std::abs(prep.bg_median));
  const float bmed2 = std::max(kTiny * kTiny, bmed * bmed);
  const float n_cv = out.cv_rms / bmed;
  const float n_flat = out.flatness / bmed2;
  const float n_rough = out.roughness / bmed;
  float n_degenerate = 0.0f;
  const float spread_floor = std::max(1.5f, 0.003f * bmed);
  if (out.sample_spread > spread_floor) {
    const float spread_ratio = out.surface_spread / std::max(kTiny, out.sample_spread);
    if (spread_ratio < 0.08f) {
      n_degenerate =
          8.0f * (out.sample_spread / bmed) * (0.08f - spread_ratio) / 0.08f;
      if (spread_ratio < 0.015f && out.sample_spread > 2.0f * spread_floor) {
        out.objective_raw = std::numeric_limits<float>::infinity();
        out.objective_normalized = std::numeric_limits<float>::infinity();
        out.objective = std::numeric_limits<float>::infinity();
        out.success = false;
        out.total_seconds = elapsed_seconds_since(total_start);
        return out;
      }
    }
  }

  out.objective_raw = out.cv_rms +
                      cfg_try.autotune.alpha_flatness * out.flatness +
                      cfg_try.autotune.beta_roughness * out.roughness;
  out.objective_normalized = n_cv + cfg_try.autotune.alpha_flatness * n_flat +
                             cfg_try.autotune.beta_roughness * n_rough +
                             n_degenerate;
  out.objective = out.objective_normalized;
  out.success = std::isfinite(out.objective_raw) &&
                std::isfinite(out.objective_normalized);
  out.total_seconds = elapsed_seconds_since(total_start);
  return out;
}

/// @brief Implements auto tune bge config conservative.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static BGECandidateResult auto_tune_bge_config_conservative(
    const Matrix2Df &channel, const std::vector<TileMetrics> &tile_metrics,
    const TileGrid &tile_grid, int base_grid_spacing,
    const BGEConfig &base_cfg, BGEProfileTiming *profile) {

  const bool extended = (base_cfg.autotune.strategy == "extended");
  auto push_unique = [](std::vector<float> &out, float v) {
    for (float e : out) {
      if (std::fabs(e - v) < 1.0e-6f)
        return;
    }
    out.push_back(v);
  };
  auto push_unique_string = [](std::vector<std::string> &out,
                               const std::string &v) {
    if (v.empty())
      return;
    for (const auto &e : out) {
      if (e == v)
        return;
    }
    out.push_back(v);
  };

  // Adaptive quantile range based on field type (nebula vs galaxy).
  // Compute global image statistics to detect diffuse emission.
  std::vector<float> valid_pixels;
  valid_pixels.reserve(static_cast<size_t>(channel.size() / 16));
  for (int i = 0; i < channel.size(); i += 16) {
    if (base_cfg.common_valid_mask.size() == static_cast<size_t>(channel.size()) &&
        base_cfg.common_valid_mask[static_cast<size_t>(i)] == 0) {
      continue;
    }
    const float v = channel.data()[i];
    if (std::isfinite(v) && v > 0.0f)
      valid_pixels.push_back(v);
  }
  float img_median = 0.0f;
  float img_p10 = 0.0f;
  float img_p90 = 0.0f;
  if (valid_pixels.size() >= 1024) {
    std::sort(valid_pixels.begin(), valid_pixels.end());
    img_p10 = sorted_quantile(valid_pixels, 0.10f);
    img_median = sorted_quantile(valid_pixels, 0.50f);
    img_p90 = sorted_quantile(valid_pixels, 0.90f);
  }
  const float lower_span = std::max(kTiny, img_median - img_p10);
  const float upper_span = std::max(kTiny, img_p90 - img_median);
  const bool upper_tail_dominant = upper_span > 1.5f * lower_span;

  std::vector<float> quantiles;
  if (upper_tail_dominant) {
    push_unique(quantiles, std::clamp(base_cfg.sample_quantile, 0.05f, 0.20f));
    push_unique(quantiles, 0.10f);
    push_unique(quantiles, 0.15f);
    push_unique(quantiles, 0.20f);
    push_unique(quantiles, 0.25f);
  } else {
    push_unique(quantiles, std::clamp(base_cfg.sample_quantile, 0.10f, 0.35f));
    push_unique(quantiles, 0.15f);
    push_unique(quantiles, 0.20f);
    push_unique(quantiles, 0.25f);
    push_unique(quantiles, 0.30f);
  }
  if (extended) {
    push_unique(quantiles, 0.35f);
    push_unique(quantiles, 0.40f);
  }

  std::vector<std::string> estimators;
  push_unique_string(estimators, base_cfg.sample_estimator);
  // Always compare the historical quantile estimator against the robust
  // mode-like estimator. In extended mode, sweep all supported estimators.
  push_unique_string(estimators, "quantile");
  push_unique_string(estimators, "sextractor_mode");
  if (extended) {
    push_unique_string(estimators, "sigma_clipped_median");
    push_unique_string(estimators, "biweight");
  }

  std::vector<float> structure_p;
  push_unique(structure_p,
              std::max(base_cfg.structure_thresh_percentile, 0.80f));
  push_unique(structure_p, 0.85f);
  push_unique(structure_p, 0.90f);
  std::vector<float> mu_factors = {
      base_cfg.fit.rbf_mu_factor,
      1.4f,
  };
  if (extended) {
    push_unique(structure_p, 0.95f);
    mu_factors.push_back(1.8f);
  }

  std::vector<std::string> fit_methods;
  fit_methods.push_back(base_cfg.fit.method);
  if (base_cfg.fit.method == "rbf") {
    // Flexible RBF fits can absorb diffuse object halos; always evaluate a
    // low-order polynomial candidate and let validation/roughness decide.
    fit_methods.push_back("poly");
  }

  BGECandidateResult best;
  const int channel_width = static_cast<int>(channel.cols());
  const int channel_height = static_cast<int>(channel.rows());
  const SparseEvalGrid eval_grid = build_sparse_eval_grid(
      channel_width, channel_height, std::max(4, base_grid_spacing / 4));
  const int max_evals = std::max(1, base_cfg.autotune.max_evals);
  int eval_count = 0;

  auto accumulate_profile = [&](const BGECandidateResult &res) {
    if (profile == nullptr)
      return;
    profile->autotune_candidate_jobs += 1;
    profile->autotune_eval_seconds += res.total_seconds;
    profile->autotune_eval_model_select_seconds += res.model_select_seconds;
    profile->autotune_eval_surface_sample_seconds +=
        res.surface_sample_seconds;
    profile->autotune_eval_metric_seconds += res.metric_seconds;
  };

  auto consider_result = [&](const BGECandidateResult &res) {
    if (!res.success)
      return;
    if (!best.success || res.objective < best.objective) {
      best = res;
    } else if (std::fabs(res.objective - best.objective) < 1e-6f) {
      if (res.roughness < best.roughness) {
        best = res;
      } else if (std::fabs(res.roughness - best.roughness) < 1e-6f) {
        if (res.cfg.fit.rbf_mu_factor > best.cfg.fit.rbf_mu_factor) {
          best = res;
        }
      }
    }
  };

  for (float sp : structure_p) {
    if (eval_count >= max_evals)
      break;

    const float sp_clamped = std::clamp(sp, 0.50f, 0.99f);
    BGEConfig prep_cfg = base_cfg;
    prep_cfg.structure_thresh_percentile = sp_clamped;
    const auto prep_extract_start = SteadyClock::now();
    auto prepared_tiles = extract_autotune_prepared_tile_samples(
        channel, tile_metrics, tile_grid, prep_cfg);
    if (profile != nullptr) {
      profile->autotune_prep_seconds +=
          elapsed_seconds_since(prep_extract_start);
      profile->autotune_prep_builds += 1;
    }

    for (float q : quantiles) {
      if (eval_count >= max_evals)
        break;

      const float q_clamped = std::clamp(q, 0.05f, 0.50f);
      for (const auto &estimator : estimators) {
        if (eval_count >= max_evals)
          break;
        if (estimator != "quantile" &&
            std::fabs(q_clamped - quantiles.front()) > 1.0e-6f) {
          continue;
        }

        BGEConfig prep_materialize_cfg = base_cfg;
        prep_materialize_cfg.sample_estimator = estimator;
        prep_materialize_cfg.sample_quantile = q_clamped;
        prep_materialize_cfg.structure_thresh_percentile = sp_clamped;
        const auto prep_materialize_start = SteadyClock::now();
        BGECandidatePrep prep = build_bge_candidate_prep(
            prepared_tiles, q_clamped, channel_width, channel_height,
            prep_materialize_cfg, base_grid_spacing);
        if (profile != nullptr) {
          profile->autotune_prep_seconds +=
              elapsed_seconds_since(prep_materialize_start);
        }
        if (!prep.valid)
          continue;

        for (const auto &fit_method : fit_methods) {
          std::vector<float> method_mu_factors;
          if (fit_method == "rbf") {
            method_mu_factors = mu_factors;
          } else {
            method_mu_factors.push_back(base_cfg.fit.rbf_mu_factor);
          }

          for (float mf : method_mu_factors) {
            if (eval_count >= max_evals)
              break;
            BGEConfig cfg_try = base_cfg;
            cfg_try.fit.method = fit_method;
            cfg_try.sample_estimator = estimator;
            cfg_try.sample_quantile = q_clamped;
            cfg_try.structure_thresh_percentile = sp_clamped;
            if (fit_method == "rbf") {
              cfg_try.fit.rbf_mu_factor = std::max(0.2f, mf);
            }
            BGECandidateResult res = try_bge_candidate_prepared(
                channel_width, channel_height, cfg_try, base_grid_spacing, prep,
                eval_grid);
            ++eval_count;
            accumulate_profile(res);
            consider_result(res);
          }
        }
      }
    }
  }

  if (!best.success) {
    best.cfg = base_cfg;
    best.grid_spacing = base_grid_spacing;
  }
  best.evals = eval_count;
  return best;
}

// Main BGE function (v3.3 §6.3)
/// @brief Applies background extraction.
/// @details Part of background-gradient extraction, mesh sampling, RBF fitting, robust weighting, and autotune evaluation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool apply_background_extraction(Matrix2Df &R, Matrix2Df &G, Matrix2Df &B,
                                 const std::vector<TileMetrics> &tile_metrics,
                                 const TileGrid &tile_grid,
                                 const BGEConfig &config,
                                 BGEDiagnostics *diagnostics) {
  const auto bge_total_start = SteadyClock::now();

  if (diagnostics != nullptr) {
    diagnostics->attempted = config.enabled;
    diagnostics->success = false;
    diagnostics->failure_reason.clear();
    diagnostics->image_width = 0;
    diagnostics->image_height = 0;
    diagnostics->grid_spacing = 0;
    diagnostics->method = config.fit.method;
    diagnostics->robust_loss = config.fit.robust_loss;
    diagnostics->insufficient_cell_strategy =
        config.grid.insufficient_cell_strategy;
    diagnostics->autotune_enabled = config.autotune.enabled;
    diagnostics->autotune_strategy = config.autotune.strategy;
    diagnostics->autotune_max_evals = config.autotune.max_evals;
    diagnostics->autotune_evals = 0;
    diagnostics->autotune_selected_fit_method = config.fit.method;
    diagnostics->autotune_best_objective = 0.0f;
    diagnostics->autotune_best_objective_raw = 0.0f;
    diagnostics->autotune_best_objective_normalized = 0.0f;
    diagnostics->autotune_best_cv_rms = 0.0f;
    diagnostics->autotune_best_flatness = 0.0f;
    diagnostics->autotune_best_roughness = 0.0f;
    diagnostics->autotune_selected_sample_estimator = config.sample_estimator;
    diagnostics->autotune_selected_sample_quantile = 0.0f;
    diagnostics->autotune_selected_structure_thresh_percentile = 0.0f;
    diagnostics->autotune_selected_rbf_mu_factor = 0.0f;
    diagnostics->autotune_fallback_used = false;
    diagnostics->safety_fallback_triggered = false;
    diagnostics->safety_fallback_method.clear();
    diagnostics->safety_fallback_reason.clear();
    diagnostics->profile = BGEProfileTiming{};
    diagnostics->channels.clear();
  }

  if (!config.enabled) {
    return false;
  }

  const int H = R.rows();
  const int W = R.cols();
  const bool have_canvas_mask =
      !config.common_valid_mask.empty() &&
      config.common_mask_rows == H &&
      config.common_mask_cols == W &&
      static_cast<int>(config.common_valid_mask.size()) == H * W;
  if (!have_canvas_mask) {
    std::cerr << "[BGE] Error: missing/invalid canvas mask (canvas_mask.fits required)"
              << std::endl;
    return false;
  }
  // Hard policy: canvas-masked pixels are excluded globally from BGE and kept
  // at zero throughout color processing.
  enforce_canvas_mask_on_rgb(R, G, B, config.common_valid_mask);

  if (diagnostics != nullptr) {
    diagnostics->image_width = W;
    diagnostics->image_height = H;
  }

  std::cout << "[BGE] Starting background extraction (v3.3 §6.3)" << std::endl;
  std::cout << "[BGE] Image size: " << W << "x" << H << std::endl;
  std::cout << "[BGE] Method: " << config.fit.method << std::endl;

  // Compute grid spacing (v3.3 §6.3.8)
  int grid_spacing = compute_grid_spacing(W, H, tile_grid.tile_size, config);
  std::cout << "[BGE] Grid spacing: " << grid_spacing << " px" << std::endl;

  if (diagnostics != nullptr) {
    diagnostics->grid_spacing = grid_spacing;
  }

  const Matrix2Df R_input = R;
  const Matrix2Df G_input = G;
  const Matrix2Df B_input = B;

  bool any_channel_applied = false;
  int channels_applied_total = 0;
  bool global_autotune_set = false;

  const bool use_modeled_mask_mesh = (config.fit.method == "modeled_mask_mesh");
  Matrix2Df modeled_luma;
  std::vector<ForegroundComponent> modeled_components;
  std::vector<uint8_t> modeled_fg_mask;
  float modeled_low_threshold = 0.0f;
  float modeled_sigma = 0.0f;
  if (use_modeled_mask_mesh) {
    const auto modeled_prepass_start = SteadyClock::now();
    modeled_luma = Matrix2Df::Zero(H, W);
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const float rv = R(y, x);
        const float gv = G(y, x);
        const float bv = B(y, x);
        const size_t idx = static_cast<size_t>(y * W + x);
        if (config.common_valid_mask[idx] == 0 ||
            !(std::isfinite(rv) && rv > 0.0f && std::isfinite(gv) &&
              gv > 0.0f && std::isfinite(bv) && bv > 0.0f)) {
          modeled_luma(y, x) = 0.0f;
          continue;
        }
        modeled_luma(y, x) = 0.2126f * rv + 0.7152f * gv + 0.0722f * bv;
      }
    }
    modeled_fg_mask =
        build_modeled_foreground_mask(modeled_luma, config, &modeled_components,
                                      &modeled_low_threshold, &modeled_sigma);
    int mask_count = 0;
    for (uint8_t v : modeled_fg_mask)
      if (v != 0)
        ++mask_count;
    const float mask_frac =
        modeled_fg_mask.empty()
            ? 0.0f
            : static_cast<float>(mask_count) /
                  static_cast<float>(modeled_fg_mask.size());
    std::cout << "[BGE] modeled_mask_mesh prepass: threshold="
              << modeled_low_threshold << " sigma=" << modeled_sigma
              << " components=" << modeled_components.size()
              << " mask_fraction=" << mask_frac << std::endl;
    if (config.autotune.enabled) {
      std::cout << "[BGE] modeled_mask_mesh does not use autotune; "
                   "continuing with deterministic settings"
                << std::endl;
    }
    if (diagnostics != nullptr) {
      diagnostics->profile.modeled_prepass_seconds =
          elapsed_seconds_since(modeled_prepass_start);
    }
  }

  auto finalize_channel_from_model =
      [&](Matrix2Df *channel, const char *channel_name,
          const Matrix2Df &channel_before, BackgroundModel &bg_model,
          BGEChannelDiagnostics &ch_diag) -> bool {
    if (!bg_model.success)
      return false;

    std::cout << "[BGE]   Fit RMS residual: " << bg_model.rms_residual
              << std::endl;
    ch_diag.fit_success = true;
    ch_diag.fit_rms_residual = bg_model.rms_residual;

    if (ch_diag.sample_bg_values.size() >= 16) {
      std::vector<float> sample_vals = ch_diag.sample_bg_values;
      std::sort(sample_vals.begin(), sample_vals.end());
      const float q05 = sorted_quantile(sample_vals, 0.05f);
      const float q95 = sorted_quantile(sample_vals, 0.95f);
      const float guard_pad =
          std::max(0.75f, 2.0f * std::max(0.0f, bg_model.rms_residual));
      const float model_min = q05 - guard_pad;
      const float model_max = q95 + guard_pad;
      int clipped = 0;
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          float &mv = bg_model.model(y, x);
          if (!std::isfinite(mv))
            continue;
          const float clamped = std::clamp(mv, model_min, model_max);
          if (std::fabs(clamped - mv) > 1.0e-6f)
            ++clipped;
          mv = clamped;
        }
      }
      if (clipped > 0) {
        std::cout << "[BGE]   Model clamp: " << clipped << " px to ["
                  << model_min << ".." << model_max << "]" << std::endl;
      }
    }

    const auto correction_start = SteadyClock::now();
    ch_diag.model_stats = stats_from_matrix(bg_model.model);
    const float pedestal = ch_diag.model_stats.median;
    Matrix2Df corrected = channel_before;
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const size_t idx = static_cast<size_t>(y * W + x);
        if (config.common_valid_mask[idx] == 0) {
          corrected(y, x) = 0.0f;
          continue;
        }
        const float vin = channel_before(y, x);
        if (!std::isfinite(vin)) {
          corrected(y, x) = std::numeric_limits<float>::quiet_NaN();
          continue;
        }
        if (vin <= 0.0f) {
          corrected(y, x) = 0.0f;
          continue;
        }
        corrected(y, x) = vin - bg_model.model(y, x) + pedestal;
      }
    }
    ch_diag.profile.apply_correction_seconds +=
        elapsed_seconds_since(correction_start);

    auto flatness_from_grid = [&](bool corrected_values) -> float {
      std::vector<float> vals;
      vals.reserve(ch_diag.grid_cells.size());
      for (const auto &gc : ch_diag.grid_cells) {
        if (!gc.valid)
          continue;
        const int px = static_cast<int>(gc.center_x);
        const int py = static_cast<int>(gc.center_y);
        if (!(px >= 0 && px < W && py >= 0 && py < H))
          continue;
        if (corrected_values) {
          vals.push_back(gc.bg_value - bg_model.model(py, px) + pedestal);
        } else {
          vals.push_back(gc.bg_value);
        }
      }
      if (vals.size() < 8)
        return std::numeric_limits<float>::infinity();
      std::sort(vals.begin(), vals.end());
      const float p10 = sorted_quantile(vals, 0.10f);
      const float p90 = sorted_quantile(vals, 0.90f);
      return p90 - p10;
    };

    const auto guard_start = SteadyClock::now();
    float flat_pre = std::numeric_limits<float>::infinity();
    float flat_post = std::numeric_limits<float>::infinity();
    if (!ch_diag.grid_cells.empty()) {
      flat_pre = flatness_from_grid(false);
      flat_post = flatness_from_grid(true);
    }
    const std::vector<uint8_t> *canvas_mask_ptr =
        (!config.common_valid_mask.empty() && config.common_mask_rows == H &&
         config.common_mask_cols == W &&
         static_cast<int>(config.common_valid_mask.size()) == H * W)
            ? &config.common_valid_mask
            : nullptr;
    const float image_flat_pre =
        spatial_background_spread(channel_before, canvas_mask_ptr);
    const float image_flat_post =
        spatial_background_spread(corrected, canvas_mask_ptr);
    if (std::isfinite(image_flat_pre) && std::isfinite(image_flat_post)) {
      flat_pre = image_flat_pre;
      flat_post = image_flat_post;
    } else if (!(std::isfinite(flat_pre) && std::isfinite(flat_post))) {
      flat_pre = image_flat_pre;
      flat_post = image_flat_post;
    }
    const float slope_pre =
        coarse_background_plane_slope(channel_before, canvas_mask_ptr);
    const float slope_post =
        coarse_background_plane_slope(corrected, canvas_mask_ptr);
    ch_diag.profile.guard_seconds += elapsed_seconds_since(guard_start);
    ch_diag.guard_flat_pre = flat_pre;
    ch_diag.guard_flat_post = flat_post;
    ch_diag.guard_slope_pre = slope_pre;
    ch_diag.guard_slope_post = slope_post;
    ch_diag.guard_reason.clear();
    bool accept_correction = true;
    const float max_flatness_worsen_factor =
        config.internal_relaxed_channel_guards ? 1.35f : 1.15f;
    const float max_slope_worsen_factor =
        config.internal_relaxed_channel_guards ? 1.15f : 1.08f;
    if (std::isfinite(flat_pre) && std::isfinite(flat_post) &&
        flat_post > flat_pre * max_flatness_worsen_factor) {
      std::cout << "[BGE]   Flatness guard rejected channel " << channel_name
                << " (pre=" << flat_pre << ", post=" << flat_post << ")"
                << std::endl;
      accept_correction = false;
      ch_diag.guard_reason = "flatness_worsened";
    }
    if (accept_correction && std::isfinite(slope_pre) &&
        std::isfinite(slope_post) &&
        slope_post > slope_pre * max_slope_worsen_factor) {
      std::cout << "[BGE]   Slope guard rejected channel " << channel_name
                << " (pre=" << slope_pre << ", post=" << slope_post << ")"
                << std::endl;
      accept_correction = false;
      ch_diag.guard_reason = "slope_worsened";
    }

    ch_diag.guard_rejected = !accept_correction;
    if (!accept_correction) {
      if (ch_diag.guard_reason.empty()) {
        ch_diag.guard_reason = "channel_guard_rejected";
      }
      ch_diag.applied = false;
      ch_diag.output_stats = stats_from_matrix(channel_before);
      ch_diag.mean_shift = 0.0f;
      return false;
    }

    *channel = std::move(corrected);
    ch_diag.applied = true;
    ch_diag.output_stats = stats_from_matrix(*channel);
    ch_diag.mean_shift = ch_diag.output_stats.mean - ch_diag.input_stats.mean;
    ch_diag.residual_values.reserve(ch_diag.grid_cells.size());
    for (const auto &gc : ch_diag.grid_cells) {
      const int px = static_cast<int>(gc.center_x);
      const int py = static_cast<int>(gc.center_y);
      if (px >= 0 && px < W && py >= 0 && py < H) {
        ch_diag.residual_values.push_back(gc.bg_value - bg_model.model(py, px));
      }
    }
    ch_diag.residual_stats = stats_from_values(ch_diag.residual_values);

    std::cout << "[BGE]   Background subtracted from channel " << channel_name
              << std::endl;
    return true;
  };

  // Process each channel
  for (int c = 0; c < 3; ++c) {
    const auto channel_total_start = SteadyClock::now();
    Matrix2Df *channel = (c == 0) ? &R : (c == 1) ? &G : &B;
    const char *channel_name = (c == 0) ? "R" : (c == 1) ? "G" : "B";
    const Matrix2Df channel_before = *channel;

    BGEChannelDiagnostics ch_diag;
    ch_diag.channel_name = channel_name;
    ch_diag.autotune_enabled =
        config.autotune.enabled && !use_modeled_mask_mesh;
    ch_diag.autotune_selected_fit_method = config.fit.method;
    ch_diag.autotune_selected_sample_estimator = config.sample_estimator;
    ch_diag.autotune_selected_grid_spacing = grid_spacing;
    ch_diag.input_stats = stats_from_matrix(channel_before);

    std::cout << "[BGE] Processing channel " << channel_name << std::endl;

    if (use_modeled_mask_mesh) {
      auto mesh_model = fit_modeled_mask_mesh_surface(
          channel_before, modeled_luma, modeled_fg_mask, modeled_components,
          modeled_low_threshold, modeled_sigma, config, tile_grid.tile_size);

      if (!mesh_model.success) {
        std::cout << "[BGE]   modeled_mask_mesh fit failed for channel "
                  << channel_name << std::endl;
        ch_diag.profile.total_seconds = elapsed_seconds_since(channel_total_start);
        if (diagnostics != nullptr)
          diagnostics->channels.push_back(std::move(ch_diag));
        continue;
      }

      ch_diag.grid_cells = mesh_model.grid_cells;
      ch_diag.grid_cells_valid = 0;
      for (const auto &gc : ch_diag.grid_cells) {
        if (gc.valid) {
          ++ch_diag.grid_cells_valid;
          ch_diag.sample_bg_values.push_back(gc.bg_value);
          ch_diag.sample_weight_values.push_back(gc.weight);
        }
      }
      ch_diag.tile_samples_total = static_cast<int>(ch_diag.grid_cells.size());
      ch_diag.tile_samples_valid = ch_diag.grid_cells_valid;
      ch_diag.sample_bg_stats = stats_from_values(ch_diag.sample_bg_values);
      ch_diag.sample_weight_stats =
          stats_from_values(ch_diag.sample_weight_values);

      BackgroundModel bg_model;
      bg_model.model = std::move(mesh_model.model);
      bg_model.grid_cells = ch_diag.grid_cells;
      bg_model.n_valid_cells = mesh_model.n_valid_cells;
      bg_model.rms_residual = mesh_model.rms_residual;
      bg_model.success = true;

      if (finalize_channel_from_model(channel, channel_name, channel_before,
                                      bg_model, ch_diag)) {
        any_channel_applied = true;
        ++channels_applied_total;
      }

      ch_diag.profile.total_seconds = elapsed_seconds_since(channel_total_start);
      std::cout << "[BGE]   Profile " << channel_name
                << ": total=" << ch_diag.profile.total_seconds
                << "s apply=" << ch_diag.profile.apply_correction_seconds
                << "s guard=" << ch_diag.profile.guard_seconds << "s"
                << std::endl;
      if (diagnostics != nullptr)
        diagnostics->channels.push_back(std::move(ch_diag));
      continue;
    }

    BGEConfig channel_cfg = config;
    int channel_grid_spacing = grid_spacing;
    if (config.autotune.enabled) {
      const auto autotune_start = SteadyClock::now();
      BGECandidateResult tune_res = auto_tune_bge_config_conservative(
          *channel, tile_metrics, tile_grid, grid_spacing, config,
          &ch_diag.profile);
      ch_diag.profile.autotune_total_seconds +=
          elapsed_seconds_since(autotune_start);
      channel_cfg = tune_res.cfg;
      channel_grid_spacing = tune_res.grid_spacing;

      if (diagnostics != nullptr) {
        diagnostics->autotune_evals += tune_res.evals;
        diagnostics->autotune_fallback_used =
            diagnostics->autotune_fallback_used || !tune_res.success;
        if (tune_res.success) {
          if (!global_autotune_set ||
              tune_res.objective < diagnostics->autotune_best_objective) {
            global_autotune_set = true;
            diagnostics->autotune_best_objective = tune_res.objective;
            diagnostics->autotune_best_objective_raw = tune_res.objective_raw;
            diagnostics->autotune_best_objective_normalized =
                tune_res.objective_normalized;
            diagnostics->autotune_best_cv_rms = tune_res.cv_rms;
            diagnostics->autotune_best_flatness = tune_res.flatness;
            diagnostics->autotune_best_roughness = tune_res.roughness;
            diagnostics->autotune_selected_fit_method = tune_res.cfg.fit.method;
            diagnostics->autotune_selected_sample_estimator =
                tune_res.cfg.sample_estimator;
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
      ch_diag.autotune_selected_fit_method = channel_cfg.fit.method;
      ch_diag.autotune_selected_sample_estimator = channel_cfg.sample_estimator;
      ch_diag.autotune_selected_grid_spacing = channel_grid_spacing;
      ch_diag.autotune_fallback_used = !tune_res.success;
      if (tune_res.success) {
        ch_diag.autotune_best_objective = tune_res.objective;
        ch_diag.autotune_best_objective_raw = tune_res.objective_raw;
        ch_diag.autotune_best_objective_normalized =
            tune_res.objective_normalized;
        ch_diag.autotune_best_cv_rms = tune_res.cv_rms;
        ch_diag.autotune_best_flatness = tune_res.flatness;
        ch_diag.autotune_best_roughness = tune_res.roughness;
        ch_diag.autotune_selected_sample_estimator =
            tune_res.cfg.sample_estimator;
        ch_diag.autotune_selected_sample_quantile =
            tune_res.cfg.sample_quantile;
        ch_diag.autotune_selected_structure_thresh_percentile =
            tune_res.cfg.structure_thresh_percentile;
        ch_diag.autotune_selected_rbf_mu_factor =
            tune_res.cfg.fit.rbf_mu_factor;
      }
    }

    const auto tile_sampling_start = SteadyClock::now();
    auto tile_samples = extract_tile_background_samples(*channel, tile_metrics,
                                                        tile_grid, channel_cfg);
    ch_diag.profile.tile_sampling_seconds +=
        elapsed_seconds_since(tile_sampling_start);
    int n_valid = std::count_if(tile_samples.begin(), tile_samples.end(),
                                [](const auto &s) { return s.valid; });
    std::cout << "[BGE]   Tile samples: " << n_valid << "/"
              << tile_samples.size() << " valid" << std::endl;

    ch_diag.tile_samples_total = static_cast<int>(tile_samples.size());
    ch_diag.tile_samples_valid = n_valid;
    ch_diag.sample_bg_values.reserve(static_cast<size_t>(n_valid));
    ch_diag.sample_weight_values.reserve(static_cast<size_t>(n_valid));
    for (const auto &s : tile_samples) {
      if (!s.valid)
        continue;
      // Exclude near-zero samples (zero-padding tiles that slipped through
      // the >20% zero-pixel filter but still have near-zero bg_value).
      // These corrupt the model clamp range and pull RBF knots to ~0.
      if (s.bg_value < channel_cfg.min_sample_bg_value)
        continue;
      ch_diag.sample_bg_values.push_back(s.bg_value);
      ch_diag.sample_weight_values.push_back(s.weight);
    }
    ch_diag.sample_bg_stats = stats_from_values(ch_diag.sample_bg_values);
    ch_diag.sample_weight_stats =
        stats_from_values(ch_diag.sample_weight_values);

    const int n_total_samples = std::max(1, ch_diag.tile_samples_total);
    const float valid_fraction =
        static_cast<float>(ch_diag.tile_samples_valid) /
        static_cast<float>(n_total_samples);
    const int min_valid_samples_for_apply =
        std::max(1, channel_cfg.min_valid_samples_for_apply);
    const float min_valid_fraction_for_apply =
        std::clamp(channel_cfg.min_valid_sample_fraction_for_apply, 0.0f, 1.0f);
    if (ch_diag.tile_samples_valid < min_valid_samples_for_apply ||
        valid_fraction < min_valid_fraction_for_apply) {
      std::cout
          << "[BGE]   Warning: insufficient robust tile samples for channel "
          << channel_name << " (" << ch_diag.tile_samples_valid << "/"
          << ch_diag.tile_samples_total << ", frac=" << valid_fraction
          << ", required>=" << min_valid_samples_for_apply
          << " and frac>=" << min_valid_fraction_for_apply
          << "), skipping channel" << std::endl;
      ch_diag.profile.total_seconds = elapsed_seconds_since(channel_total_start);
      if (diagnostics != nullptr)
        diagnostics->channels.push_back(std::move(ch_diag));
      continue;
    }

    const auto coarse_grid_start = SteadyClock::now();
    auto grid_cells = aggregate_to_coarse_grid(
        tile_samples, W, H, channel_grid_spacing, channel_cfg);
    ch_diag.profile.coarse_grid_seconds +=
        elapsed_seconds_since(coarse_grid_start);
    std::cout << "[BGE]   Grid cells: " << grid_cells.size() << " valid"
              << std::endl;

    ch_diag.grid_cells_valid = static_cast<int>(grid_cells.size());
    ch_diag.grid_cells = grid_cells;

    if (grid_cells.size() < 3) {
      std::cout << "[BGE]   Warning: Too few grid cells, skipping channel "
                << channel_name << std::endl;
      ch_diag.profile.total_seconds = elapsed_seconds_since(channel_total_start);
      if (diagnostics != nullptr)
        diagnostics->channels.push_back(std::move(ch_diag));
      continue;
    }

    auto bg_model = fit_background_surface(grid_cells, W, H,
                                           channel_grid_spacing, channel_cfg);
    ch_diag.profile.final_fit_total_seconds += bg_model.total_seconds;
    ch_diag.profile.final_fit_select_seconds += bg_model.fit_select_seconds;
    ch_diag.profile.final_fit_render_seconds += bg_model.render_seconds;
    if (!bg_model.success) {
      std::cerr << "[BGE]   Error: " << bg_model.error_message << std::endl;
      ch_diag.profile.total_seconds = elapsed_seconds_since(channel_total_start);
      if (diagnostics != nullptr)
        diagnostics->channels.push_back(std::move(ch_diag));
      continue;
    }

    if (finalize_channel_from_model(channel, channel_name, channel_before,
                                    bg_model, ch_diag)) {
      any_channel_applied = true;
      ++channels_applied_total;
    }

    ch_diag.profile.total_seconds = elapsed_seconds_since(channel_total_start);
    std::cout << "[BGE]   Profile " << channel_name
              << ": total=" << ch_diag.profile.total_seconds
              << "s autotune=" << ch_diag.profile.autotune_total_seconds
              << "s prep=" << ch_diag.profile.autotune_prep_seconds
              << "s eval=" << ch_diag.profile.autotune_eval_seconds
              << "s tile_sampling=" << ch_diag.profile.tile_sampling_seconds
              << "s coarse_grid=" << ch_diag.profile.coarse_grid_seconds
              << "s final_fit=" << ch_diag.profile.final_fit_total_seconds
              << "s render=" << ch_diag.profile.final_fit_render_seconds
              << "s apply=" << ch_diag.profile.apply_correction_seconds
              << "s guard=" << ch_diag.profile.guard_seconds << "s"
              << std::endl;
    if (diagnostics != nullptr)
      diagnostics->channels.push_back(std::move(ch_diag));
  }

  // Hard chroma policy: BGE must be applied atomically across RGB.
  // A partial per-channel apply can introduce color casts (e.g. green bias)
  // that PCC cannot reliably undo in bright nebulosity.
  // In relaxed-guard fallback mode (internal_relaxed_channel_guards): accept
  // 2/3 partial application rather than failing completely. A minor color cast
  // from one unapplied channel is better than skipping BGE entirely.
  // In normal mode: require all 3 channels to avoid color casts that PCC
  // cannot reliably undo.
  const int min_channels_required =
      config.internal_relaxed_channel_guards ? 2 : 3;
  if (channels_applied_total > 0 && channels_applied_total < min_channels_required) {
    std::cerr << "[BGE] Partial channel application (" << channels_applied_total
              << "/3) rejected; reverting all channels to pre-BGE state"
              << std::endl;
    R = R_input;
    G = G_input;
    B = B_input;
    any_channel_applied = false;
    channels_applied_total = 0;
    if (diagnostics != nullptr) {
      diagnostics->safety_fallback_triggered = true;
      diagnostics->safety_fallback_method = "revert_rgb";
      diagnostics->safety_fallback_reason = "partial_channel_application";
      for (auto &ch : diagnostics->channels) {
        if (!ch.applied)
          continue;
        ch.applied = false;
        ch.output_stats = ch.input_stats;
        ch.mean_shift = 0.0f;
        ch.residual_values.clear();
        ch.residual_stats = BGEValueStats{};
      }
    }

    // Safety retry from a clean RGB state. Only do this once from the default
    // guard regime to avoid recursive fallback loops.
    if (!config.internal_relaxed_channel_guards) {
      auto run_fallback = [&](const BGEConfig &fb_cfg, const char *name,
                              Matrix2Df *outR, Matrix2Df *outG,
                              Matrix2Df *outB, BGEDiagnostics *out_diag) {
        *outR = R_input;
        *outG = G_input;
        *outB = B_input;
        out_diag->channels.clear();
        std::cout << "[BGE]   Trying partial-channel safety fallback method: "
                  << name << std::endl;
        return apply_background_extraction(*outR, *outG, *outB, tile_metrics,
                                           tile_grid, fb_cfg, out_diag);
      };

      Matrix2Df R_fb;
      Matrix2Df G_fb;
      Matrix2Df B_fb;
      BGEDiagnostics fb_diag;
      bool fb_ok = false;
      std::string chosen_method;

      // Fallback #1: smoother low-order polynomial with relaxed guards.
      BGEConfig fallback_poly = config;
      fallback_poly.fit.method = "poly";
      fallback_poly.fit.polynomial_order = 2;
      fallback_poly.fit.robust_loss = "tukey";
      fallback_poly.sample_quantile = std::min(config.sample_quantile, 0.16f);
      fallback_poly.structure_thresh_percentile =
          std::min(config.structure_thresh_percentile, 0.80f);
      fallback_poly.autotune.enabled = false;
      fallback_poly.internal_relaxed_channel_guards = true;
      fb_ok =
          run_fallback(fallback_poly, "poly", &R_fb, &G_fb, &B_fb, &fb_diag);
      if (fb_ok) {
        chosen_method = "poly";
      }

      // Fallback #2: conservative RBF with relaxed guards.
      if (!fb_ok) {
        std::cout << "[BGE]   Poly fallback failed; trying conservative RBF"
                  << std::endl;
        BGEConfig fallback_rbf = config;
        fallback_rbf.fit.method = "rbf";
        fallback_rbf.fit.robust_loss = "tukey";
        fallback_rbf.fit.rbf_phi = "multiquadric";
        fallback_rbf.fit.rbf_mu_factor =
            std::max(1.2f, std::min(1.8f, config.fit.rbf_mu_factor));
        fallback_rbf.fit.rbf_lambda =
            std::max(1.0e-5f, config.fit.rbf_lambda);
        fallback_rbf.sample_quantile = std::min(config.sample_quantile, 0.18f);
        fallback_rbf.structure_thresh_percentile =
            std::min(config.structure_thresh_percentile, 0.85f);
        fallback_rbf.autotune.enabled = false;
        fallback_rbf.internal_relaxed_channel_guards = true;
        fb_ok = run_fallback(fallback_rbf, "rbf", &R_fb, &G_fb, &B_fb,
                             &fb_diag);
        if (fb_ok) {
          chosen_method = "rbf";
        }
      }

      if (fb_ok) {
        R = std::move(R_fb);
        G = std::move(G_fb);
        B = std::move(B_fb);
        any_channel_applied = true;
        channels_applied_total = 3;
        if (diagnostics != nullptr) {
          diagnostics->safety_fallback_triggered = true;
          diagnostics->safety_fallback_method = chosen_method;
          diagnostics->safety_fallback_reason = "partial_channel_application";
          diagnostics->method = chosen_method;
          diagnostics->autotune_selected_fit_method = chosen_method;
          diagnostics->autotune_selected_sample_estimator =
              fb_diag.autotune_selected_sample_estimator;
          diagnostics->robust_loss = fb_diag.robust_loss;
          diagnostics->grid_spacing = fb_diag.grid_spacing;
          diagnostics->channels = std::move(fb_diag.channels);
        }
      } else {
        std::cerr
            << "[BGE] partial-channel safety fallbacks failed; keeping pre-BGE RGB"
            << std::endl;
        if (diagnostics != nullptr) {
          diagnostics->safety_fallback_triggered = true;
          diagnostics->safety_fallback_method = "poly->rbf";
          diagnostics->safety_fallback_reason =
              "partial_channel_application_fallback_failed";
        }
      }
    }
  }

  if (any_channel_applied && !config.internal_relaxed_channel_guards) {
    const std::vector<uint8_t> bg_mask =
        build_chroma_background_mask_from_rgb(
            R_input, G_input, B_input, &config.common_valid_mask);
    const float pre_rg_std =
        log_chroma_std_background(R_input, G_input, bg_mask);
    const float pre_bg_std =
        log_chroma_std_background(B_input, G_input, bg_mask);
    const float post_rg_std = log_chroma_std_background(R, G, bg_mask);
    const float post_bg_std = log_chroma_std_background(B, G, bg_mask);

    constexpr float kMaxChromaStdWorsenFactor = 1.08f;
    bool chroma_guard_failed = false;
    std::string chroma_guard_reason;
    if (std::isfinite(pre_rg_std) && std::isfinite(post_rg_std) &&
        post_rg_std > pre_rg_std * kMaxChromaStdWorsenFactor) {
      chroma_guard_failed = true;
      chroma_guard_reason = "background_chroma_rg_worsened";
    }
    if (std::isfinite(pre_bg_std) && std::isfinite(post_bg_std) &&
        post_bg_std > pre_bg_std * kMaxChromaStdWorsenFactor) {
      chroma_guard_failed = true;
      chroma_guard_reason = "background_chroma_bg_worsened";
    }

    if (!chroma_guard_failed && diagnostics != nullptr &&
        diagnostics->channels.size() >= 3) {
      float model_std_max = 0.0f;
      float model_median_max = 0.0f;
      int spatial_channels = 0;
      for (const auto &ch : diagnostics->channels) {
        if (!ch.applied)
          continue;
        model_std_max = std::max(model_std_max, ch.model_stats.std);
        model_median_max =
            std::max(model_median_max, std::abs(ch.model_stats.median));
      }
      const float spatial_threshold =
          std::max(0.25f, 0.001f * std::max(1.0f, model_median_max));
      for (const auto &ch : diagnostics->channels) {
        if (ch.applied && ch.model_stats.std > spatial_threshold)
          ++spatial_channels;
      }

      const float pre_chroma_max = std::max(pre_rg_std, pre_bg_std);
      const float post_chroma_max = std::max(post_rg_std, post_bg_std);
      const bool chroma_is_measurable =
          std::isfinite(pre_chroma_max) && std::isfinite(post_chroma_max) &&
          pre_chroma_max > 0.008f;
      const bool weak_flat_model =
          spatial_channels == 0 && model_std_max <= spatial_threshold &&
          chroma_is_measurable && post_chroma_max >= 0.98f * pre_chroma_max;
      const bool imbalanced_channel_model =
          spatial_channels > 0 && spatial_channels < 3 &&
          chroma_is_measurable && post_chroma_max >= 0.95f * pre_chroma_max;
      if (weak_flat_model || imbalanced_channel_model) {
        chroma_guard_failed = true;
        chroma_guard_reason =
            weak_flat_model ? "flat_model_left_background_chroma"
                            : "imbalanced_channel_spatial_model";
      }
    }

    std::cout << "[BGE] chroma guard: pre_rg_std="
              << pre_rg_std << " post_rg_std=" << post_rg_std
              << " pre_bg_std=" << pre_bg_std << " post_bg_std=" << post_bg_std
              << std::endl;

    if (chroma_guard_failed) {
      std::cout
          << "[BGE] BGE failed background chroma guard; "
             "falling back to conservative fits (reason="
          << chroma_guard_reason << ")"
          << std::endl;

      auto run_fallback = [&](const BGEConfig &fb_cfg, const char *name,
                              Matrix2Df *outR, Matrix2Df *outG, Matrix2Df *outB,
                              BGEDiagnostics *out_diag) {
        *outR = R_input;
        *outG = G_input;
        *outB = B_input;
        out_diag->channels.clear();
        std::cout << "[BGE]   Trying safety fallback method: " << name
                  << std::endl;
        return apply_background_extraction(*outR, *outG, *outB, tile_metrics,
                                           tile_grid, fb_cfg, out_diag);
      };

      // Fallback #1: conservative RBF.
      BGEConfig fallback_rbf = config;
      fallback_rbf.fit.method = "rbf";
      fallback_rbf.fit.robust_loss = "tukey";
      fallback_rbf.fit.rbf_phi = "multiquadric";
      fallback_rbf.fit.rbf_mu_factor =
          std::max(1.2f, std::min(1.8f, config.fit.rbf_mu_factor));
      fallback_rbf.fit.rbf_lambda = std::max(1.0e-5f, config.fit.rbf_lambda);
      fallback_rbf.sample_quantile = std::min(config.sample_quantile, 0.18f);
      fallback_rbf.structure_thresh_percentile =
          std::min(config.structure_thresh_percentile, 0.85f);
      fallback_rbf.autotune.enabled = false;
      fallback_rbf.internal_relaxed_channel_guards = true;

      Matrix2Df R_fb;
      Matrix2Df G_fb;
      Matrix2Df B_fb;
      BGEDiagnostics fb_diag;
      bool fb_ok =
          run_fallback(fallback_rbf, "rbf", &R_fb, &G_fb, &B_fb, &fb_diag);
      std::string chosen_method;

      // Fallback #2: robust low-order polynomial with relaxed guards.
      if (!fb_ok) {
        std::cout << "[BGE]   RBF fallback failed; trying poly fallback"
                  << std::endl;
        BGEConfig fallback_poly = config;
        fallback_poly.fit.method = "poly";
        fallback_poly.fit.polynomial_order = 2;
        fallback_poly.fit.robust_loss = "tukey";
        fallback_poly.sample_quantile = std::min(config.sample_quantile, 0.16f);
        fallback_poly.structure_thresh_percentile =
            std::min(config.structure_thresh_percentile, 0.80f);
        fallback_poly.autotune.enabled = false;
        fallback_poly.internal_relaxed_channel_guards = true;

        fb_ok =
            run_fallback(fallback_poly, "poly", &R_fb, &G_fb, &B_fb, &fb_diag);
        if (fb_ok) {
          chosen_method = "poly";
        }
      } else {
        chosen_method = "rbf";
      }

      if (fb_ok) {
        R = std::move(R_fb);
        G = std::move(G_fb);
        B = std::move(B_fb);
        any_channel_applied = true;
        channels_applied_total = 3;
        if (diagnostics != nullptr) {
          diagnostics->safety_fallback_triggered = true;
          diagnostics->safety_fallback_method = chosen_method;
          diagnostics->safety_fallback_reason =
              chroma_guard_reason.empty() ? "background_chroma_worsened"
                                          : chroma_guard_reason;
          diagnostics->method = chosen_method;
          diagnostics->autotune_selected_fit_method = chosen_method;
          diagnostics->autotune_selected_sample_estimator =
              fb_diag.autotune_selected_sample_estimator;
          diagnostics->robust_loss = fb_diag.robust_loss;
          diagnostics->grid_spacing = fb_diag.grid_spacing;
          diagnostics->channels = std::move(fb_diag.channels);
        }
      } else {
        std::cerr
            << "[BGE] all safety fallbacks failed; reverting BGE for this image"
            << std::endl;
        R = R_input;
        G = G_input;
        B = B_input;
        any_channel_applied = false;
        channels_applied_total = 0;
        if (diagnostics != nullptr) {
          diagnostics->safety_fallback_triggered = true;
          diagnostics->safety_fallback_method = "rbf->poly";
          diagnostics->safety_fallback_reason =
              "background_chroma_worsened_fallback_failed";
          diagnostics->channels.clear();
        }
      }
    }
  }

  enforce_canvas_mask_on_rgb(R, G, B, config.common_valid_mask);
  std::cout << "[BGE] Background extraction complete" << std::endl;
  if (diagnostics != nullptr) {
    diagnostics->profile.total_seconds = elapsed_seconds_since(bge_total_start);
    for (const auto &ch : diagnostics->channels) {
      diagnostics->profile.autotune_total_seconds +=
          ch.profile.autotune_total_seconds;
      diagnostics->profile.autotune_prep_seconds +=
          ch.profile.autotune_prep_seconds;
      diagnostics->profile.autotune_eval_seconds +=
          ch.profile.autotune_eval_seconds;
      diagnostics->profile.autotune_eval_model_select_seconds +=
          ch.profile.autotune_eval_model_select_seconds;
      diagnostics->profile.autotune_eval_surface_sample_seconds +=
          ch.profile.autotune_eval_surface_sample_seconds;
      diagnostics->profile.autotune_eval_metric_seconds +=
          ch.profile.autotune_eval_metric_seconds;
      diagnostics->profile.tile_sampling_seconds +=
          ch.profile.tile_sampling_seconds;
      diagnostics->profile.coarse_grid_seconds +=
          ch.profile.coarse_grid_seconds;
      diagnostics->profile.final_fit_total_seconds +=
          ch.profile.final_fit_total_seconds;
      diagnostics->profile.final_fit_select_seconds +=
          ch.profile.final_fit_select_seconds;
      diagnostics->profile.final_fit_render_seconds +=
          ch.profile.final_fit_render_seconds;
      diagnostics->profile.apply_correction_seconds +=
          ch.profile.apply_correction_seconds;
      diagnostics->profile.guard_seconds += ch.profile.guard_seconds;
      diagnostics->profile.autotune_prep_builds +=
          ch.profile.autotune_prep_builds;
      diagnostics->profile.autotune_candidate_jobs +=
          ch.profile.autotune_candidate_jobs;
    }
    std::cout << "[BGE] Profile total=" << diagnostics->profile.total_seconds
              << "s autotune=" << diagnostics->profile.autotune_total_seconds
              << "s prep=" << diagnostics->profile.autotune_prep_seconds
              << "s eval=" << diagnostics->profile.autotune_eval_seconds
              << "s tile_sampling=" << diagnostics->profile.tile_sampling_seconds
              << "s final_render="
              << diagnostics->profile.final_fit_render_seconds
              << "s apply=" << diagnostics->profile.apply_correction_seconds
              << "s guard=" << diagnostics->profile.guard_seconds << "s"
              << std::endl;
    diagnostics->success = any_channel_applied;
    diagnostics->failure_reason = infer_bge_failure_reason(*diagnostics);
  }
  return any_channel_applied;
}

} // namespace tile_compile::image
