#pragma once

#include "tile_compile/core/types.hpp"

#include <vector>

namespace tile_compile::reconstruction {

struct AqmhValidationMetrics {
  float seam_score = 0.0f;
  float fwhm = 0.0f;
  float background_rms = 0.0f;
  int star_count = 0;
  float tail11_abs_median = 0.0f;
  float tail11_p90 = 0.0f;
  float elongation_median = 0.0f;
};

struct AqmhValidationComparison {
  AqmhValidationMetrics aqmh;
  AqmhValidationMetrics control;
  float seam_score_regression = 0.0f;
  float fwhm_regression = 0.0f;
  float background_rms_regression = 0.0f;
  float tail11_abs_regression = 0.0f;
  float elongation_regression = 0.0f;
  // Gate applicability: false means the metric could not be reliably computed
  // (e.g. too few stars) and must not trigger a fallback.
  bool fwhm_applicable = true;
  bool seam_applicable = true;
  bool background_rms_applicable = true;
  bool tail_applicable = false;       // requires sufficient star_count in both images
  bool elongation_applicable = false;  // requires sufficient star_count in both images
};

struct AqmhValidationStarSample {
  int x = 0;
  int y = 0;
  float peak = 0.0f;
};

// Prepared immutable comparison side. Candidate searches can reuse its star
// positions and metrics instead of rescanning the same control image.
struct AqmhValidationReference {
  AqmhValidationMetrics metrics;
  std::vector<AqmhValidationStarSample> stars;
  int width = 0;
  int height = 0;
};

AqmhValidationMetrics measure_aqmh_validation_metrics(
    const Matrix2Df &image, const std::vector<uint8_t> &validation_mask = {});
AqmhValidationReference prepare_aqmh_validation_reference(
    const Matrix2Df &image, const std::vector<uint8_t> &validation_mask = {});
AqmhValidationComparison compare_aqmh_to_reference(
    const Matrix2Df &aqmh, const AqmhValidationReference &control,
    const std::vector<uint8_t> &validation_mask = {});
AqmhValidationComparison compare_aqmh_to_uniform_control(
    const Matrix2Df &aqmh, const Matrix2Df &control,
    const std::vector<uint8_t> &validation_mask = {});

} // namespace tile_compile::reconstruction
