#pragma once

#include "tile_compile/core/types.hpp"

#include <string>

namespace tile_compile::metrics {

struct LinearityThresholds {
  float skewness_max = 1.2f;
  float kurtosis_max = 1.2f;
  float variance_max = 0.5f;
  float energy_ratio_min = 0.95f;
  float gradient_consistency_max = 0.5f;
};

struct LinearityFrameResult {
  bool is_linear = false;
  float score = 0.0f;
  float skewness = 0.0f;
  float kurtosis = 0.0f;
  float variance_coeff = 0.0f;
  float energy_ratio = 0.0f;
  float gradient_consistency = 0.0f;
  bool moment_ok = false;
  bool spectral_ok = false;
  bool spatial_ok = false;
};

LinearityThresholds linearity_thresholds_for(const std::string &strictness);

LinearityFrameResult validate_linearity_frame(const Matrix2Df &img,
                                              const std::string &strictness);

} // namespace tile_compile::metrics
