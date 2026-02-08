#include "tile_compile/metrics/linearity.hpp"
#include "tile_compile/core/utils.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/opencv.hpp>

namespace tile_compile::metrics {

LinearityThresholds
linearity_thresholds_for(const std::string &strictness) {
  if (strictness == "moderate") {
    return {1.2f, 1.2f, 0.7f, 0.9f, 0.7f};
  }
  if (strictness == "permissive") {
    return {1.5f, 1.5f, 1.0f, 0.8f, 1.0f};
  }
  return {1.2f, 1.2f, 0.5f, 0.95f, 0.5f};
}

LinearityFrameResult
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
  float p1 = core::percentile_from_sorted(sorted, 1.0f);
  float p5 = core::percentile_from_sorted(sorted, 5.0f);
  float p50 = core::percentile_from_sorted(sorted, 50.0f);
  float p95 = core::percentile_from_sorted(sorted, 95.0f);
  float p99 = core::percentile_from_sorted(sorted, 99.0f);

  float denom_skew = (p50 - p1) + 1.0e-12f;
  float denom_kurt = (p50 - p5) + 1.0e-12f;
  out.skewness = (p99 - p50) / denom_skew;
  out.kurtosis = (p95 - p50) / denom_kurt;
  out.variance_coeff = static_cast<float>(stddev / (std::fabs(mean) + 1.0e-12));

  cv::Mat gx, gy, mag;
  cv::Sobel(small, gx, CV_32F, 1, 0, 3);
  cv::Sobel(small, gy, CV_32F, 0, 1, 3);
  cv::magnitude(gx, gy, mag);
  double mean_grad = cv::mean(mag)[0];
  double mean_frame = cv::mean(small)[0];
  out.gradient_consistency =
      static_cast<float>(2.0 * (mean_grad / (std::fabs(mean_frame) + 1.0e-12)));

  out.energy_ratio = 0.0f;
  if (small.rows >= 8 && small.cols >= 8) {
    cv::Mat dft;
    cv::dft(small, dft, cv::DFT_COMPLEX_OUTPUT);
    std::vector<cv::Mat> planes;
    cv::split(dft, planes);
    cv::Mat mag2 = planes[0].mul(planes[0]) + planes[1].mul(planes[1]);
    double total_energy = cv::sum(mag2)[0];
    int r = std::max(1, std::min(mag2.rows, mag2.cols) / 8);
    double low_energy = 0.0;
    low_energy += cv::sum(mag2(cv::Rect(0, 0, r, r)))[0];
    low_energy += cv::sum(mag2(cv::Rect(0, mag2.rows - r, r, r)))[0];
    low_energy += cv::sum(mag2(cv::Rect(mag2.cols - r, 0, r, r)))[0];
    low_energy +=
        cv::sum(mag2(cv::Rect(mag2.cols - r, mag2.rows - r, r, r)))[0];
    if (total_energy > 0.0) {
      out.energy_ratio = static_cast<float>(low_energy / total_energy);
    }
  }

  LinearityThresholds th = linearity_thresholds_for(strictness);
  out.moment_ok = (std::fabs(out.skewness) < th.skewness_max) &&
                  (std::fabs(out.kurtosis) < th.kurtosis_max) &&
                  (out.variance_coeff < th.variance_max);
  out.spectral_ok = (out.energy_ratio >= th.energy_ratio_min);
  out.spatial_ok = (out.gradient_consistency < th.gradient_consistency_max);

  out.score =
      (static_cast<float>(out.moment_ok) + static_cast<float>(out.spectral_ok) +
       static_cast<float>(out.spatial_ok)) /
      3.0f;
  out.is_linear = out.moment_ok && out.spectral_ok && out.spatial_ok;
  return out;
}

} // namespace tile_compile::metrics
