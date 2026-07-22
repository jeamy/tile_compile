#include "tile_compile/metrics/aqmh_eps.hpp"
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

} // namespace

float aqmh_median(std::vector<float> values) {
  values = finite_only(values);
  if (values.empty()) return std::numeric_limits<float>::quiet_NaN();
  std::sort(values.begin(), values.end());
  const size_t mid = values.size() / 2;
  return values.size() % 2 ? values[mid]
                           : 0.5f * (values[mid - 1] + values[mid]);
}

float aqmh_mad(const std::vector<float> &values, float center) {
  if (!std::isfinite(center)) return std::numeric_limits<float>::quiet_NaN();
  std::vector<float> dev;
  dev.reserve(values.size());
  for (float v : values)
    if (std::isfinite(v)) dev.push_back(std::abs(v - center));
  return aqmh_median(std::move(dev));
}

float eps_scale(const std::vector<float> &values) {
  const auto finite = finite_only(values);
  if (finite.empty()) return positive_floor();
  std::vector<float> abs_values;
  abs_values.reserve(finite.size());
  for (float v : finite) abs_values.push_back(std::abs(v));
  const float med_abs = aqmh_median(std::move(abs_values));
  const float med = aqmh_median(finite);
  const float mad = aqmh_mad(finite, med);
  return std::max(positive_floor(), aqmh_eps_rel * std::max(med_abs, mad));
}

float eps_noise(const std::vector<float> &values) {
  const auto finite = finite_only(values);
  if (finite.empty()) return positive_floor();
  const float med = aqmh_median(finite);
  return std::max(positive_floor(), aqmh_eps_rel * aqmh_mad(finite, med));
}

std::vector<float> robust_zscore_eps_scale(const std::vector<float> &values) {
  std::vector<float> out(values.size(), std::numeric_limits<float>::quiet_NaN());
  const auto finite = finite_only(values);
  if (finite.empty()) return out;
  const float med = aqmh_median(finite);
  const float mad = aqmh_mad(finite, med);
  if (!(mad > 0.0f) || !std::isfinite(mad)) {
    for (size_t i = 0; i < values.size(); ++i)
      if (std::isfinite(values[i])) out[i] = 0.0f;
    return out;
  }
  const float eps = std::max(positive_floor(), aqmh_eps_rel * std::max(std::abs(med), mad));
  const float scale = std::max(core::kMadToSigma * mad, eps);
  for (size_t i = 0; i < values.size(); ++i)
    if (std::isfinite(values[i])) out[i] = (values[i] - med) / scale;
  return out;
}

} // namespace tile_compile::metrics
