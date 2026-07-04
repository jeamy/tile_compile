#include "tile_compile/reconstruction/aqmh_sigma_clip.hpp"

#include "tile_compile/metrics/aqmh_eps.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tile_compile::reconstruction {
namespace {

float sample_key(const AqmhWeightedSample &sample, bool deviations,
                 float center) {
  return deviations ? std::abs(sample.value - center) : sample.value;
}

// Deterministic three-way weighted quickselect. This replaces a complete
// O(N log N) sort for every median and MAD evaluation with expected O(N).
float weighted_median_select(std::vector<AqmhWeightedSample> &samples,
                             bool deviations, float center = 0.0f) {
  if (samples.empty()) return 0.0f;
  double target = 0.0;
  for (const auto &sample : samples) target += sample.weight;
  target *= 0.5;
  size_t first = 0, last = samples.size();
  while (last - first > 1) {
    const float a = sample_key(samples[first], deviations, center);
    const float b = sample_key(samples[first + (last - first) / 2],
                               deviations, center);
    const float c = sample_key(samples[last - 1], deviations, center);
    const float pivot = std::max(std::min(a, b), std::min(std::max(a, b), c));
    size_t lower = first, scan = first, upper = last;
    while (scan < upper) {
      const float key = sample_key(samples[scan], deviations, center);
      if (key < pivot) {
        std::swap(samples[lower++], samples[scan++]);
      } else if (key > pivot) {
        std::swap(samples[scan], samples[--upper]);
      } else {
        ++scan;
      }
    }
    double lower_weight = 0.0, equal_weight = 0.0;
    for (size_t i = first; i < lower; ++i) lower_weight += samples[i].weight;
    for (size_t i = lower; i < upper; ++i) equal_weight += samples[i].weight;
    if (target <= lower_weight && lower > first) {
      last = lower;
    } else if (target <= lower_weight + equal_weight || upper == last) {
      return pivot;
    } else {
      target -= lower_weight + equal_weight;
      first = upper;
    }
  }
  return sample_key(samples[first], deviations, center);
}

float median_select(std::vector<float> &values) {
  if (values.empty()) return 0.0f;
  const size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + mid, values.end());
  const float hi = values[mid];
  if (values.size() % 2 != 0) return hi;
  const float lo = *std::max_element(values.begin(), values.begin() + mid);
  return 0.5f * (lo + hi);
}

float noise_floor(const std::vector<AqmhWeightedSample> &samples,
                  std::vector<float> &values) {
  values.clear();
  values.reserve(samples.size());
  for (const auto &sample : samples) values.push_back(sample.value);
  const float center = median_select(values);
  for (float &value : values) value = std::abs(value - center);
  const float mad = median_select(values);
  return std::max(std::nextafter(0.0f, 1.0f),
                  metrics::aqmh_eps_rel * mad);
}

} // namespace

AqmhSigmaClipResult aqmh_sigma_clip(
    std::vector<AqmhWeightedSample> samples, float clip_sigma,
    int iterations, float min_fraction, float min_effective_n) {
  samples.erase(std::remove_if(samples.begin(), samples.end(), [](const auto &s) {
                  return !std::isfinite(s.value) || !std::isfinite(s.weight) ||
                         !(s.weight > 0.0f);
                }), samples.end());
  AqmhSigmaClipResult result;
  if (samples.empty()) return result;
  const size_t n0 = samples.size();

  // Small-N fast path: for N<=8 use a fixed-size stack array instead of heap
  // allocations for deviations/noise_values. Avoids all dynamic allocation for
  // the common case of pixels with few contributing frames.
  constexpr size_t kSmallN = 8;

  const size_t keep_floor = std::min(
      n0, std::max<size_t>(1, static_cast<size_t>(std::ceil(min_fraction * n0))));
  for (int iter = 0; iter < iterations; ++iter) {
    const float center = weighted_median_select(samples, false);
    thread_local std::vector<AqmhWeightedSample> deviations;
    thread_local std::vector<float> noise_values;
    float mad;
    float floor_val;
    if (n0 <= kSmallN) {
      // Stack-based path: no heap alloc.
      float dev_arr[kSmallN];
      float noise_arr[kSmallN];
      const size_t n = samples.size();
      for (size_t i = 0; i < n; ++i) dev_arr[i] = std::abs(samples[i].value - center);
      // Selection sort for tiny N (O(N²) but N<=8, ~28 comparisons max).
      for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
          if (dev_arr[j] < dev_arr[i]) std::swap(dev_arr[i], dev_arr[j]);
      const size_t mid = n / 2;
      mad = n % 2 ? dev_arr[mid] : 0.5f * (dev_arr[mid - 1] + dev_arr[mid]);
      for (size_t i = 0; i < n; ++i) noise_arr[i] = samples[i].value;
      for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
          if (noise_arr[j] < noise_arr[i]) std::swap(noise_arr[i], noise_arr[j]);
      const float val_med = n % 2 ? noise_arr[mid] : 0.5f * (noise_arr[mid-1] + noise_arr[mid]);
      for (size_t i = 0; i < n; ++i) noise_arr[i] = std::abs(noise_arr[i] - val_med);
      for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
          if (noise_arr[j] < noise_arr[i]) std::swap(noise_arr[i], noise_arr[j]);
      const float noise_mad = n % 2 ? noise_arr[mid] : 0.5f * (noise_arr[mid-1] + noise_arr[mid]);
      floor_val = std::max(std::nextafter(0.0f, 1.0f), metrics::aqmh_eps_rel * noise_mad);
    } else {
      deviations.assign(samples.begin(), samples.end());
      mad = weighted_median_select(deviations, true, center);
      floor_val = noise_floor(samples, noise_values);
    }
    size_t keep_count = 0;
    if (mad <= floor_val) {
      for (const auto &s : samples) keep_count += s.value == center;
    } else {
      const float sigma = 1.4826f * mad;
      for (const auto &s : samples)
        keep_count += std::abs(s.value - center) <= clip_sigma * sigma;
    }
    if (keep_count == samples.size()) {
      // All samples within band: distribution is stable, no further clipping
      // needed. Early exit saves remaining iterations (typically iter 1..N-1).
      break;
    }
    if (keep_count < keep_floor) {
      std::sort(samples.begin(), samples.end(), [&](const auto &a, const auto &b) {
        const float ar = std::abs(a.value - center) / std::max(1.4826f * mad, floor_val);
        const float br = std::abs(b.value - center) / std::max(1.4826f * mad, floor_val);
        return ar != br ? ar < br : a.frame_index < b.frame_index;
      });
      samples.resize(keep_floor);
    } else if (keep_count < samples.size()) {
      const float sigma = 1.4826f * mad;
      samples.erase(std::remove_if(samples.begin(), samples.end(),
          [&](const auto &s) {
            return mad <= floor_val ? s.value != center
                                   : std::abs(s.value - center) > clip_sigma * sigma;
          }), samples.end());
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
