#include "tile_compile/reconstruction/post_run_star_shape.hpp"

#include "tile_compile/metrics/metrics.hpp"
#include "tile_compile/reconstruction/multiband_validation.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>

namespace tile_compile::reconstruction {
namespace {

constexpr int kMaxControlStars = 2000;
constexpr int kMinMatchedStars = 20;
constexpr int kBootstrapSamples = 2000;
constexpr uint64_t kBootstrapSeed = 20261024;
constexpr double kMinFinitePatchFraction = kMultibandValidationStarPatchMinFiniteFraction;

using nlohmann::json;

json unavailable(const char* reason, int detected = 0, int matched = 0) {
  return {{"schema_version", "pi.post-run-star-shape.v1"},
          {"status", "not_applicable"},
          {"reason_code", reason},
          {"detected_control_stars", detected},
          {"matched_stars", matched},
          {"before_stage", "selected_reconstruction"},
          {"after_stage", "final_hms_rgb_g"},
          {"interpretation", "descriptive_measurement_only"}};
}

bool extract_patch(const Matrix2Df& image, const std::vector<uint8_t>& valid,
                   int x, int y, cv::Mat& patch) {
  constexpr int radius = kMultibandValidationStarPatchRadius;
  const int side = 2 * radius + 1;
  if (x < radius || y < radius || x + radius >= image.cols() ||
      y + radius >= image.rows()) return false;
  if (!valid[static_cast<size_t>(y) * image.cols() + x]) return false;

  std::vector<float> finite;
  finite.reserve(static_cast<size_t>(side) * side);
  for (int py = y - radius; py <= y + radius; ++py)
    for (int px = x - radius; px <= x + radius; ++px) {
      const size_t index = static_cast<size_t>(py) * image.cols() + px;
      if (valid[index]) finite.push_back(image(py, px));
    }
  const size_t total = static_cast<size_t>(side) * side;
  if (static_cast<double>(finite.size()) < kMinFinitePatchFraction * total) return false;
  std::nth_element(finite.begin(), finite.begin() + finite.size() / 2, finite.end());
  const float fill = finite[finite.size() / 2];
  patch.create(side, side, CV_32F);
  for (int py = 0; py < side; ++py)
    for (int px = 0; px < side; ++px) {
      const int iy = y - radius + py;
      const int ix = x - radius + px;
      const size_t index = static_cast<size_t>(iy) * image.cols() + ix;
      patch.at<float>(py, px) = valid[index] ? image(iy, ix) : fill;
    }
  return true;
}

double median(std::vector<double> values) {
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(values.begin(), values.end());
  const size_t n = values.size();
  return n % 2 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

double percentile(std::vector<double> values, double p) {
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(values.begin(), values.end());
  const double pos = std::clamp(p, 0.0, 1.0) * (values.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = std::min(lo + 1, values.size() - 1);
  return values[lo] + (values[hi] - values[lo]) * (pos - lo);
}

std::array<double, 2> bootstrap_star_median_ci(const std::vector<double>& ratios) {
  std::mt19937_64 rng(kBootstrapSeed);
  std::uniform_int_distribution<size_t> draw(0, ratios.size() - 1);
  std::vector<double> medians;
  medians.reserve(kBootstrapSamples);
  std::vector<double> sample(ratios.size());
  for (int i = 0; i < kBootstrapSamples; ++i) {
    for (double& value : sample) value = ratios[draw(rng)];
    medians.push_back(median(sample));
  }
  return {percentile(medians, 0.025), percentile(medians, 0.975)};
}

}  // namespace

nlohmann::json measure_post_run_star_shape(const Matrix2Df& before,
                                          const Matrix2Df& after,
                                          const std::vector<uint8_t>& common_valid_mask) {
  if (before.rows() < 32 || before.cols() < 32 || before.rows() != after.rows() ||
      before.cols() != after.cols()) return unavailable("raster_mismatch_or_too_small");
  const size_t pixels = static_cast<size_t>(before.rows()) * before.cols();
  if (common_valid_mask.size() != pixels) return unavailable("valid_mask_mismatch");

  std::vector<uint8_t> valid(pixels, 0);
  cv::Mat1f safe(before.rows(), before.cols(), 0.0f);
  cv::Mat1b detector_mask(before.rows(), before.cols(), static_cast<uint8_t>(0));
  size_t valid_count = 0;
  for (int y = 0; y < before.rows(); ++y) {
    for (int x = 0; x < before.cols(); ++x) {
      const size_t i = static_cast<size_t>(y) * before.cols() + x;
      const float a = before(y, x), b = after(y, x);
      if (!common_valid_mask[i] || !std::isfinite(a) || !std::isfinite(b)) continue;
      valid[i] = 1;
      detector_mask(y, x) = 255;
      safe(y, x) = a;
      ++valid_count;
    }
  }
  if (valid_count < static_cast<size_t>(32 * 32)) return unavailable("insufficient_common_support");

  // Match the established frame-star detector's broad-background subtraction while keeping all
  // star coordinates sourced from the selected reconstruction only.
  cv::Mat background, residual;
  cv::blur(safe, background, cv::Size(31, 31), cv::Point(-1, -1), cv::BORDER_REFLECT_101);
  cv::subtract(safe, background, residual);
  for (int y = 0; y < residual.rows; ++y)
    for (int x = 0; x < residual.cols; ++x)
      if (!detector_mask(y, x)) residual.at<float>(y, x) = 0.0f;

  std::vector<cv::Point2f> corners;
  try {
    cv::goodFeaturesToTrack(residual, corners, kMaxControlStars, 0.01, 15.0,
                            detector_mask, 3, false, 0.04);
  } catch (...) {
    return unavailable("star_detection_failed");
  }
  if (corners.empty()) return unavailable("no_control_stars");

  std::vector<double> ratios;
  ratios.reserve(corners.size());
  cv::Mat before_patch, after_patch;
  for (const auto& corner : corners) {
    const int x = static_cast<int>(std::lround(corner.x));
    const int y = static_cast<int>(std::lround(corner.y));
    if (x < kMultibandValidationStarPatchRadius || y < kMultibandValidationStarPatchRadius ||
        x + kMultibandValidationStarPatchRadius >= before.cols() ||
        y + kMultibandValidationStarPatchRadius >= before.rows()) continue;
    if (!extract_patch(before, valid, x, y, before_patch) ||
        !extract_patch(after, valid, x, y, after_patch)) continue;
    const float fwhm_before = metrics::estimate_fwhm_from_patch(before_patch);
    const float fwhm_after = metrics::estimate_fwhm_from_patch(after_patch);
    if (!std::isfinite(fwhm_before) || !std::isfinite(fwhm_after) ||
        fwhm_before <= 0.0f || fwhm_after <= 0.0f) continue;
    const double ratio = static_cast<double>(fwhm_after) / fwhm_before;
    if (std::isfinite(ratio) && ratio > 0.0) ratios.push_back(ratio);
  }

  if (ratios.size() < kMinMatchedStars)
    return unavailable("too_few_matched_stars", static_cast<int>(corners.size()),
                       static_cast<int>(ratios.size()));
  const auto ci = bootstrap_star_median_ci(ratios);
  const double median_ratio = median(ratios);
  const double p90_ratio = percentile(ratios, 0.90);
  if (!std::isfinite(median_ratio) || !std::isfinite(p90_ratio) ||
      !std::isfinite(ci[0]) || !std::isfinite(ci[1]))
    return unavailable("non_finite_result", static_cast<int>(corners.size()),
                       static_cast<int>(ratios.size()));

  return {{"schema_version", "pi.post-run-star-shape.v1"},
          {"status", "measured"},
          {"method", "fixed_control_positions_psf_patch_fwhm_v1"},
          {"before_stage", "selected_reconstruction"},
          {"after_stage", "final_hms_rgb_g"},
          {"metric", "matched_star_fwhm_ratio_after_over_before"},
          {"median_ratio", median_ratio},
          {"p90_ratio", p90_ratio},
          {"ci95_star_bootstrap", {ci[0], ci[1]}},
          {"detected_control_stars", static_cast<int>(corners.size())},
          {"matched_stars", static_cast<int>(ratios.size())},
          {"min_matched_stars", kMinMatchedStars},
          {"unit", "ratio"},
          {"interpretation", "descriptive_measurement_only"}};
}

}  // namespace tile_compile::reconstruction
