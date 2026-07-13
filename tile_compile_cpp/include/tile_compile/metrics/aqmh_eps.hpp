#pragma once

#include <vector>

namespace tile_compile::metrics {

inline constexpr float aqmh_eps_rel = 1.0e-6f;

float aqmh_median(std::vector<float> values);
float aqmh_mad(const std::vector<float> &values, float center);
float eps_scale(const std::vector<float> &values);
float eps_noise(const std::vector<float> &values);
std::vector<float> robust_zscore_eps_scale(const std::vector<float> &values);

} // namespace tile_compile::metrics
