#include "tile_compile/reconstruction/aqmh_sigma_clip.hpp"

#include "tile_compile/metrics/aqmh_eps.hpp"
#include "tile_compile/core/utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tile_compile::reconstruction {
namespace {

struct WeightedVal {
  float val;
  float weight;
};

// Deterministic three-way weighted quickselect. This replaces a complete
// O(N log N) sort for every median and MAD evaluation with expected O(N).
float weighted_median_select(WeightedVal *arr, size_t n) {
  if (n == 0) return 0.0f;
  if (n == 1) return arr[0].val;
  if (n == 2) {
    if (arr[0].val > arr[1].val) std::swap(arr[0], arr[1]);
    return arr[0].weight >= arr[1].weight ? arr[0].val : arr[1].val;
  }
  double target = 0.0;
  for (size_t i = 0; i < n; ++i) target += arr[i].weight;
  target *= 0.5;
  size_t first = 0, last = n;
  while (last - first > 1) {
    const size_t mid_idx = first + (last - first) / 2;
    const float a = arr[first].val;
    const float b = arr[mid_idx].val;
    const float c = arr[last - 1].val;
    const float pivot = std::max(std::min(a, b), std::min(std::max(a, b), c));
    size_t lower = first, scan = first, upper = last;
    while (scan < upper) {
      const float key = arr[scan].val;
      if (key < pivot) {
        std::swap(arr[lower++], arr[scan++]);
      } else if (key > pivot) {
        std::swap(arr[scan], arr[--upper]);
      } else {
        ++scan;
      }
    }
    double lower_weight = 0.0, equal_weight = 0.0;
    for (size_t i = first; i < lower; ++i) lower_weight += arr[i].weight;
    for (size_t i = lower; i < upper; ++i) equal_weight += arr[i].weight;
    if (target <= lower_weight && lower > first) {
      last = lower;
    } else if (target <= lower_weight + equal_weight || upper == last) {
      return pivot;
    } else {
      target -= lower_weight + equal_weight;
      first = upper;
    }
  }
  return arr[first].val;
}

inline float fast_median_inplace(float *data, size_t n) {
  if (n == 0) return 0.0f;
  const size_t mid = n / 2;
  std::nth_element(data, data + mid, data + n);
  const float hi = data[mid];
  if ((n % 2) == 1) return hi;
  std::nth_element(data, data + mid - 1, data + mid);
  const float lo = data[mid - 1];
  return 0.5f * (lo + hi);
}

float noise_floor_fast(const AqmhWeightedSample *samples, size_t n,
                       std::vector<float> &scratch) {
  scratch.resize(n);
  for (size_t i = 0; i < n; ++i) scratch[i] = samples[i].value;
  const float center = fast_median_inplace(scratch.data(), n);
  for (size_t i = 0; i < n; ++i) scratch[i] = std::abs(scratch[i] - center);
  const float mad = fast_median_inplace(scratch.data(), n);
  return std::max(std::nextafter(0.0f, 1.0f),
                  metrics::aqmh_eps_rel * mad);
}

} // namespace

AqmhSigmaClipResult aqmh_sigma_clip(
    std::vector<AqmhWeightedSample> samples, float clip_sigma,
    int iterations, float min_fraction, float min_effective_n) {
  return aqmh_sigma_clip(std::move(samples), clip_sigma, clip_sigma, iterations,
                         min_fraction, min_effective_n);
}

AqmhSigmaClipResult aqmh_sigma_clip(
    std::vector<AqmhWeightedSample> samples, float clip_sigma_low,
    float clip_sigma_high, int iterations, float min_fraction,
    float min_effective_n) {
  AqmhSigmaClipResult result;
  if (samples.empty()) return result;

  // Quick check if filtering needed
  bool has_invalid = false;
  for (const auto &s : samples) {
    if (!std::isfinite(s.value) || !std::isfinite(s.weight) || !(s.weight > 0.0f)) {
      has_invalid = true;
      break;
    }
  }
  if (has_invalid) {
    samples.erase(std::remove_if(samples.begin(), samples.end(), [](const auto &s) {
                    return !std::isfinite(s.value) || !std::isfinite(s.weight) ||
                           !(s.weight > 0.0f);
                  }), samples.end());
    if (samples.empty()) return result;
  }

  const size_t n0 = samples.size();

  // Small-N fast path: for N<=8 use a fixed-size stack array instead of heap
  // allocations for deviations/noise_values. Avoids all dynamic allocation for
  // the common case of pixels with few contributing frames.
  constexpr size_t kSmallN = 8;

  const size_t keep_floor = std::min(
      n0, std::max<size_t>(1, static_cast<size_t>(std::ceil(min_fraction * n0))));

  thread_local std::vector<WeightedVal> wvals;
  thread_local std::vector<float> noise_values;

  for (int iter = 0; iter < iterations; ++iter) {
    const size_t n = samples.size();
    float center;
    float mad;
    float floor_val;
    if (n0 <= kSmallN) {
      // Stack-based path: no heap alloc.
      WeightedVal stack_wvals[kSmallN];
      for (size_t i = 0; i < n; ++i) stack_wvals[i] = {samples[i].value, samples[i].weight};
      center = weighted_median_select(stack_wvals, n);
      for (size_t i = 0; i < n; ++i) stack_wvals[i] = {std::abs(samples[i].value - center), samples[i].weight};
      mad = weighted_median_select(stack_wvals, n);

      float noise_arr[kSmallN];
      for (size_t i = 0; i < n; ++i) noise_arr[i] = samples[i].value;
      const float val_med = fast_median_inplace(noise_arr, n);
      for (size_t i = 0; i < n; ++i) noise_arr[i] = std::abs(noise_arr[i] - val_med);
      const float noise_mad = fast_median_inplace(noise_arr, n);
      floor_val = std::max(std::nextafter(0.0f, 1.0f), metrics::aqmh_eps_rel * noise_mad);
    } else {
      wvals.resize(n);
      for (size_t i = 0; i < n; ++i) wvals[i] = {samples[i].value, samples[i].weight};
      center = weighted_median_select(wvals.data(), n);

      for (size_t i = 0; i < n; ++i) wvals[i] = {std::abs(samples[i].value - center), samples[i].weight};
      mad = weighted_median_select(wvals.data(), n);

      floor_val = noise_floor_fast(samples.data(), n, noise_values);
    }

    size_t keep_count = 0;
    const bool use_noise_floor = (mad <= floor_val);
    const float eps_center = std::numeric_limits<float>::epsilon() * std::max(std::abs(center), 1.0f);
    const float sigma = tile_compile::core::kMadToSigma * mad;
    const float hi_bound = center + clip_sigma_high * sigma;
    const float lo_bound = center - clip_sigma_low * sigma;

    if (use_noise_floor) {
      for (const auto &s : samples)
        keep_count += (std::abs(s.value - center) <= eps_center);
    } else {
      for (const auto &s : samples)
        keep_count += (s.value >= lo_bound && s.value <= hi_bound);
    }

    if (keep_count == samples.size()) {
      // All samples within band: distribution is stable, no further clipping
      // needed. Early exit saves remaining iterations (typically iter 1..N-1).
      break;
    }

    if (keep_count < keep_floor) {
      const float denom = std::max(use_noise_floor ? 0.0f : sigma, floor_val);
      std::sort(samples.begin(), samples.end(), [&](const auto &a, const auto &b) {
        const float ar = std::abs(a.value - center) / denom;
        const float br = std::abs(b.value - center) / denom;
        return ar != br ? ar < br : a.frame_index < b.frame_index;
      });
      samples.resize(keep_floor);
    } else if (keep_count < samples.size()) {
      size_t write_idx = 0;
      if (use_noise_floor) {
        for (size_t i = 0; i < samples.size(); ++i) {
          if (std::abs(samples[i].value - center) <= eps_center) {
            samples[write_idx++] = samples[i];
          }
        }
      } else {
        for (size_t i = 0; i < samples.size(); ++i) {
          if (samples[i].value >= lo_bound && samples[i].value <= hi_bound) {
            samples[write_idx++] = samples[i];
          }
        }
      }
      samples.resize(write_idx);
    } else {
      break;
    }
  }
  double d = 0.0, d2 = 0.0;
  float wmax = 0.0f;
  for (const auto &s : samples) {
    d += s.weight;
    d2 += static_cast<double>(s.weight) * s.weight;
    wmax = std::max(wmax, s.weight);
  }
  result.weight_sum = static_cast<float>(d);
  result.effective_n = d2 > 0.0 ? static_cast<float>(d * d / d2) : 0.0f;
  const float guard = static_cast<float>(samples.size()) *
                      std::numeric_limits<float>::epsilon() * wmax;
  result.denominator_ok = d > guard && result.effective_n >= min_effective_n;
  result.retained = std::move(samples);
  return result;
}

} // namespace tile_compile::reconstruction
