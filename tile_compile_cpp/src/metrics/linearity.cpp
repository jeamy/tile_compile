#include "tile_compile/metrics/linearity.hpp"
#include "tile_compile/core/utils.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/opencv.hpp>

namespace tile_compile::metrics {

LinearityThresholds
/// @brief Implements linearity thresholds for.
/// @details Part of input-linearity sampling and warning/failure diagnostics; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
linearity_thresholds_for(const std::string &strictness) {
  if (strictness == "moderate") {
    return {1.5f, 0.8f, 0.8f, 0.005f, 0.6f};
  }
  if (strictness == "permissive") {
    return {2.0f, 0.9f, 1.2f, 0.001f, 0.8f};
  }
  // strict: thresholds adjusted for corrected formulas:
  // - variance_coeff now uses pixel_range (not |mean|), typical 0.01-0.30
  // - kurtosis is inverted tail-to-body ratio (p75-p25)/(p95-p5),
  //   low = heavy-tailed (linear), high = light-tailed (non-linear)
  // - energy_ratio excludes DC bin, so values are much lower (~0.01-0.10)
  return {1.2f, 0.7f, 0.6f, 0.01f, 0.4f};
}

LinearityFrameResult
/// @brief Validates linearity frame.
/// @details Part of input-linearity sampling and warning/failure diagnostics; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
validate_linearity_frame(const Matrix2Df &img,
                         const std::string &strictness) {
  LinearityFrameResult out;
  if (img.size() <= 0)
    return out;

  cv::Mat cv_img(img.rows(), img.cols(), CV_32F,
                 const_cast<float *>(img.data()));
  cv::Mat small = cv_img;

  const int max_dim = 256;
  if (cv_img.rows > max_dim || cv_img.cols > max_dim) {
    float scale = static_cast<float>(max_dim) /
                  static_cast<float>(std::max(cv_img.rows, cv_img.cols));
    cv::resize(cv_img, small, cv::Size(), scale, scale, cv::INTER_AREA);
  }

  std::vector<float> values;
  values.reserve(static_cast<size_t>(small.rows) *
                 static_cast<size_t>(small.cols));
  for (int y = 0; y < small.rows; ++y) {
    const float *row = small.ptr<float>(y);
    for (int x = 0; x < small.cols; ++x) {
      float v = row[x];
      if (std::isfinite(v))
        values.push_back(v);
    }
  }

  if (values.empty()) {
    return out;
  }

  double mean = 0.0;
  double m2 = 0.0;
  for (size_t i = 0; i < values.size(); ++i) {
    double x = static_cast<double>(values[i]);
    double delta = x - mean;
    mean += delta / static_cast<double>(i + 1);
    double delta2 = x - mean;
    m2 += delta * delta2;
  }
  double var =
      (values.size() > 1) ? (m2 / static_cast<double>(values.size() - 1)) : 0.0;
  double stddev = std::sqrt(std::max(0.0, var));

  std::vector<float> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  float p0 = sorted.front();
  float p1 = core::percentile_from_sorted(sorted, 1.0f);
  float p5 = core::percentile_from_sorted(sorted, 5.0f);
  float p50 = core::percentile_from_sorted(sorted, 50.0f);
  float p95 = core::percentile_from_sorted(sorted, 95.0f);
  float p99 = core::percentile_from_sorted(sorted, 99.0f);
  float p100 = sorted.back();

  // Use robust percentiles for skewness/kurtosis that exclude stars.
  // The previous formula (p99-p50)/(p50-p1) was dominated by bright stars
  // in astro images, giving skewness~18 regardless of actual linearity.
  // Trim to p25-p75 interquartile range for the moment-based checks.
  float p25 = core::percentile_from_sorted(sorted, 25.0f);
  float p75 = core::percentile_from_sorted(sorted, 75.0f);
  float iqr = (p75 - p25) + 1.0e-12f;
  out.skewness = (p75 - p50) / iqr;
  // Kurtosis: inverted tail-to-body ratio.  Low values = heavy-tailed
  // (linear astro with sharp star peaks), high values = light-tailed
  // (non-linear stretched, compressed dynamic range).
  out.kurtosis = (p75 - p25) / ((p95 - p5) + 1.0e-12f);
  // Variance coefficient: normalize by pixel range (not |mean|), because
  // calibrated frames have mean≈0 which would make this diverge.
  float pixel_range = (p99 - p1) + 1.0e-6f;
  out.variance_coeff = static_cast<float>(stddev / static_cast<double>(pixel_range));

  cv::Mat gx, gy, mag;
  cv::Sobel(small, gx, CV_32F, 1, 0, 3);
  cv::Sobel(small, gy, CV_32F, 0, 1, 3);
  cv::magnitude(gx, gy, mag);
  double mean_grad = cv::mean(mag)[0];
  // Normalize gradient by pixel range (p99-p1) instead of mean, because
  // background-subtracted frames have mean≈0 which would make this diverge.
  out.gradient_consistency =
      static_cast<float>(mean_grad / static_cast<double>(pixel_range));

  out.energy_ratio = 0.0f;
  if (small.rows >= 8 && small.cols >= 8) {
    cv::Mat dft_img;
    cv::dft(small, dft_img, cv::DFT_COMPLEX_OUTPUT);
    std::vector<cv::Mat> planes;
    cv::split(dft_img, planes);
    cv::Mat mag2 = planes[0].mul(planes[0]) + planes[1].mul(planes[1]);
    // Exclude DC bin (0,0) from both numerator and denominator: the DC
    // is just |sum(pixels)|^2 and carries no linearity information.
    // For uncalibrated frames it dominates and inflates energy_ratio;
    // for calibrated frames (mean≈0) it's near-zero.  Excluding it
    // makes the ratio measure actual AC spectral shape.
    double dc_energy = static_cast<double>(mag2.at<float>(0, 0));
    double total_energy = cv::sum(mag2)[0] - dc_energy;
    // Only the top-left corner is the true low-frequency region in
    // standard (non-shifted) DFT layout. The other 3 corners are Nyquist/
    // high-frequency aliases and must NOT be counted as low-frequency.
    int r = std::max(1, std::min(mag2.rows, mag2.cols) / 8);
    double low_energy = cv::sum(mag2(cv::Rect(0, 0, r, r)))[0] - dc_energy;
    if (total_energy > 0.0) {
      out.energy_ratio = static_cast<float>(low_energy / total_energy);
    }
  }

  LinearityThresholds th = linearity_thresholds_for(strictness);
  const float clip_eps =
      std::max(1.0e-3f, (p99 - p1 + 1.0e-6f) * 1.0e-5f);
  size_t low_clip_count = 0;
  size_t high_clip_count = 0;
  for (float v : values) {
    if (v <= p0 + clip_eps) ++low_clip_count;
    if (v >= p100 - clip_eps) ++high_clip_count;
  }
  const float low_clip_frac =
      static_cast<float>(low_clip_count) / static_cast<float>(values.size());
  const float high_clip_frac =
      static_cast<float>(high_clip_count) / static_cast<float>(values.size());
  float clip_fraction_max = 0.02f;
  if (strictness == "moderate") {
    clip_fraction_max = 0.05f;
  } else if (strictness == "permissive") {
    clip_fraction_max = 0.10f;
  }
  const bool low_hard_clip =
      low_clip_frac > clip_fraction_max && p0 <= 1.0f + clip_eps;
  const bool high_hard_clip =
      high_clip_frac > clip_fraction_max && p100 >= 4095.0f - clip_eps;
  const bool clipping_ok = !low_hard_clip && !high_hard_clip;

  // A single light frame cannot prove camera-response linearity for every
  // possible object. Keep the hard verdict deliberately conservative and
  // object-agnostic: robust distribution shape and obvious clipping/compression
  // are hard checks; variance, gradients and spectral layout remain diagnostics
  // because they depend strongly on field content (empty sky, star clusters,
  // nebulosity, galaxy cores, CFA texture).
  out.moment_ok = (std::fabs(out.skewness) < th.skewness_max) &&
                  (std::fabs(out.kurtosis) < th.kurtosis_max) && clipping_ok;
  out.spectral_ok = (out.energy_ratio >= th.energy_ratio_min);
  out.spatial_ok = (out.gradient_consistency < th.gradient_consistency_max);

  out.score =
      (static_cast<float>(out.moment_ok) + static_cast<float>(out.spectral_ok) +
       static_cast<float>(out.spatial_ok)) /
      3.0f;
  out.is_linear = out.moment_ok;
  return out;
}

} // namespace tile_compile::metrics
