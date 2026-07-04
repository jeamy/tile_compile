#pragma once

#include "tile_compile/core/types.hpp"

namespace tile_compile::reconstruction {

struct AqmhValidationMetrics {
  float seam_score = 0.0f;
  float fwhm = 0.0f;
  float background_rms = 0.0f;
};

struct AqmhValidationComparison {
  AqmhValidationMetrics aqmh;
  AqmhValidationMetrics control;
  float seam_score_regression = 0.0f;
  float fwhm_regression = 0.0f;
  float background_rms_regression = 0.0f;
};

AqmhValidationMetrics measure_aqmh_validation_metrics(const Matrix2Df &image);
AqmhValidationComparison compare_aqmh_to_uniform_control(
    const Matrix2Df &aqmh, const Matrix2Df &control);

} // namespace tile_compile::reconstruction
