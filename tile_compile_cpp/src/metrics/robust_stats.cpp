#include "tile_compile/metrics/robust_stats.hpp"
#include "tile_compile/core/utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tile_compile::metrics {
namespace {

std::vector<float> finite_only(const std::vector<float> &values) {
  std::vector<float> out;
  out.reserve(values.size());
  for (float v : values)
    if (std::isfinite(v)) out.push_back(v);
  return out;
}

float positive_floor() {
  return std::nextafter(0.0f, 1.0f);
}

// robust_median's selection on an all-finite, non-empty buffer, in place
// (no non-finite filtering pass, no copy). Same value as robust_median.
float median_inplace_finite(std::vector<float> &values) {
  if (values.empty()) return std::numeric_limits<float>::quiet_NaN();
  const size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + mid, values.end());
  const float upper = values[mid];
  if (values.size() % 2 != 0)
    return upper;
  const float lower =
      *std::max_element(values.begin(), values.begin() + mid);
  return 0.5f * (lower + upper);
}

} // namespace

float robust_median(std::vector<float> values) {
  values.erase(
      std::remove_if(values.begin(), values.end(),
                     [](float value) { return !std::isfinite(value); }),
      values.end());
  if (values.empty()) return std::numeric_limits<float>::quiet_NaN();
  const size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + mid, values.end());
  const float upper = values[mid];
  if (values.size() % 2 != 0)
    return upper;
  const float lower =
      *std::max_element(values.begin(), values.begin() + mid);
  return 0.5f * (lower + upper);
}

float robust_mad(const std::vector<float> &values, float center) {
  if (!std::isfinite(center)) return std::numeric_limits<float>::quiet_NaN();
  std::vector<float> dev;
  dev.reserve(values.size());
  for (float v : values)
    if (std::isfinite(v)) dev.push_back(std::abs(v - center));
  return robust_median(std::move(dev));
}

float eps_scale(const std::vector<float> &values) {
  auto finite = finite_only(values);
  if (finite.empty()) return positive_floor();
  // One scratch buffer reused for all three selections instead of a fresh
  // frame-sized copy per statistic (the medians only need the multiset).
  std::vector<float> scratch;
  scratch.reserve(finite.size());
  for (float v : finite) scratch.push_back(std::abs(v));
  const float med_abs = median_inplace_finite(scratch);
  scratch.assign(finite.begin(), finite.end());
  const float med = median_inplace_finite(scratch);
  for (std::size_t i = 0; i < finite.size(); ++i)
    finite[i] = std::abs(finite[i] - med);
  const float mad = median_inplace_finite(finite);
  return std::max(positive_floor(), robust_eps_rel * std::max(med_abs, mad));
}

float eps_noise(const std::vector<float> &values) {
  auto finite = finite_only(values);
  if (finite.empty()) return positive_floor();
  std::vector<float> scratch(finite);
  const float med = median_inplace_finite(scratch);
  for (std::size_t i = 0; i < finite.size(); ++i)
    finite[i] = std::abs(finite[i] - med);
  return std::max(positive_floor(),
                  robust_eps_rel * median_inplace_finite(finite));
}

std::vector<float> robust_zscore_eps_scale(const std::vector<float> &values) {
  std::vector<float> out(values.size(), std::numeric_limits<float>::quiet_NaN());
  const auto finite = finite_only(values);
  if (finite.empty()) return out;
  const float med = robust_median(finite);
  const float mad = robust_mad(finite, med);
  if (!(mad > 0.0f) || !std::isfinite(mad)) {
    for (size_t i = 0; i < values.size(); ++i)
      if (std::isfinite(values[i])) out[i] = 0.0f;
    return out;
  }
  const float eps = std::max(positive_floor(), robust_eps_rel * std::max(std::abs(med), mad));
  const float scale = std::max(core::kMadToSigma * mad, eps);
  for (size_t i = 0; i < values.size(); ++i)
    if (std::isfinite(values[i])) out[i] = (values[i] - med) / scale;
  return out;
}

} // namespace tile_compile::metrics
