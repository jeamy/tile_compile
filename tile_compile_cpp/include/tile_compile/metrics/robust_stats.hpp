#pragma once

#include <vector>

namespace tile_compile::metrics {

inline constexpr float robust_eps_rel = 1.0e-6f;

// Empty-input convention: robust_median and robust_mad return NaN on empty or
// non-finite center, signalling "no data" to callers. This differs from
// core::median_of / core::mad_of which return 0.0f. eps_scale and eps_noise
// return a positive floor value when no finite values are available.
float robust_median(std::vector<float> values);  // returns NaN if empty
float robust_mad(const std::vector<float> &values, float center);  // returns NaN if center non-finite or empty
float eps_scale(const std::vector<float> &values);
float eps_noise(const std::vector<float> &values);
std::vector<float> robust_zscore_eps_scale(const std::vector<float> &values);

} // namespace tile_compile::metrics
