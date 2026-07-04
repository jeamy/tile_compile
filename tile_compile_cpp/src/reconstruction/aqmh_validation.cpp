#include "tile_compile/reconstruction/aqmh_validation.hpp"

#include "tile_compile/core/utils.hpp"
#include "tile_compile/metrics/aqmh_eps.hpp"
#include "tile_compile/metrics/metrics.hpp"

#include <cmath>
#include <vector>

namespace tile_compile::reconstruction {
namespace {

float regression(float value, float control) {
  return (value - control) /
         std::max(std::abs(control), metrics::eps_scale({value, control}));
}

} // namespace

AqmhValidationMetrics measure_aqmh_validation_metrics(const Matrix2Df &image) {
  AqmhValidationMetrics out;
  if (image.rows() <= 0 || image.cols() <= 0) return out;
  std::vector<float> finite;
  finite.reserve(static_cast<size_t>(image.size()));
  double gradient_sum = 0.0;
  uint64_t gradient_count = 0;
  for (int y = 0; y < image.rows(); ++y) {
    for (int x = 0; x < image.cols(); ++x) {
      const float v = image(y, x);
      if (!std::isfinite(v)) continue;
      finite.push_back(v);
      if (x + 1 < image.cols() && std::isfinite(image(y, x + 1))) {
        gradient_sum += std::abs(v - image(y, x + 1));
        ++gradient_count;
      }
      if (y + 1 < image.rows() && std::isfinite(image(y + 1, x))) {
        gradient_sum += std::abs(v - image(y + 1, x));
        ++gradient_count;
      }
    }
  }
  if (finite.empty()) return out;
  const float med = metrics::aqmh_median(finite);
  const float sigma = 1.4826f * metrics::aqmh_mad(finite, med);
  out.seam_score = gradient_count > 0
      ? static_cast<float>(gradient_sum / gradient_count) /
            std::max(sigma, metrics::eps_scale(finite))
      : 0.0f;
  out.background_rms = sigma;
  out.fwhm = metrics::measure_fwhm_from_image(image);
  if (!std::isfinite(out.fwhm) || out.fwhm < 0.0f) out.fwhm = 0.0f;
  return out;
}

AqmhValidationComparison compare_aqmh_to_uniform_control(
    const Matrix2Df &aqmh, const Matrix2Df &control) {
  AqmhValidationComparison out;
  out.aqmh = measure_aqmh_validation_metrics(aqmh);
  out.control = measure_aqmh_validation_metrics(control);
  out.seam_score_regression = regression(out.aqmh.seam_score,
                                         out.control.seam_score);
  out.fwhm_regression = regression(out.aqmh.fwhm, out.control.fwhm);
  out.background_rms_regression = regression(out.aqmh.background_rms,
                                             out.control.background_rms);
  return out;
}

} // namespace tile_compile::reconstruction
