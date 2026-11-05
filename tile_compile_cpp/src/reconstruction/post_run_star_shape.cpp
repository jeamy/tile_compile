#include "tile_compile/reconstruction/post_run_star_shape.hpp"

#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/reconstruction/multiband_validation.hpp"
#include "tile_compile/core/utils.hpp"

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
constexpr int kMomentPatchRadius = 10;
constexpr int kMomentApertureRadius = 6;

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

double median(std::vector<double> values);

bool estimate_paired_fwhm(const Matrix2Df& image, const std::vector<uint8_t>& valid,
                          int x, int y, double& fwhm) {
  constexpr int patch_radius = kMomentPatchRadius;
  constexpr int aperture_sq = kMomentApertureRadius * kMomentApertureRadius;
  if (x < patch_radius || y < patch_radius || x + patch_radius >= image.cols() ||
      y + patch_radius >= image.rows()) return false;
  std::vector<double> annulus;
  annulus.reserve(128);
  int aperture_total = 0, aperture_valid = 0, annulus_total = 0;
  for (int dy = -patch_radius; dy <= patch_radius; ++dy)
    for (int dx = -patch_radius; dx <= patch_radius; ++dx) {
      const int r2 = dx * dx + dy * dy;
      const size_t i = static_cast<size_t>(y + dy) * image.cols() + x + dx;
      if (r2 <= aperture_sq) {
        ++aperture_total;
        if (valid[i]) ++aperture_valid;
      }
      if (r2 >= 64 && r2 <= patch_radius * patch_radius) {
        ++annulus_total;
        if (valid[i]) annulus.push_back(image(y + dy, x + dx));
      }
    }
  if (static_cast<double>(aperture_valid) < kMinFinitePatchFraction * aperture_total ||
      static_cast<double>(annulus.size()) < kMinFinitePatchFraction * annulus_total)
    return false;
  const double background = median(annulus);
  double sum = 0.0, mx = 0.0, my = 0.0;
  for (int dy = -kMomentApertureRadius; dy <= kMomentApertureRadius; ++dy)
    for (int dx = -kMomentApertureRadius; dx <= kMomentApertureRadius; ++dx) {
      if (dx * dx + dy * dy > aperture_sq) continue;
      const size_t i = static_cast<size_t>(y + dy) * image.cols() + x + dx;
      if (!valid[i]) continue;
      const double weight = std::max(0.0, static_cast<double>(image(y + dy, x + dx)) - background);
      sum += weight;
      mx += weight * dx;
      my += weight * dy;
    }
  if (!(sum > 0.0) || !std::isfinite(sum)) return false;
  mx /= sum;
  my /= sum;
  double cxx = 0.0, cyy = 0.0;
  for (int dy = -kMomentApertureRadius; dy <= kMomentApertureRadius; ++dy)
    for (int dx = -kMomentApertureRadius; dx <= kMomentApertureRadius; ++dx) {
      if (dx * dx + dy * dy > aperture_sq) continue;
      const size_t i = static_cast<size_t>(y + dy) * image.cols() + x + dx;
      if (!valid[i]) continue;
      const double weight = std::max(0.0, static_cast<double>(image(y + dy, x + dx)) - background);
      cxx += weight * (dx - mx) * (dx - mx);
      cyy += weight * (dy - my) * (dy - my);
    }
  const double sigma_sq = (cxx + cyy) / (2.0 * sum);
  if (!(sigma_sq > 0.0) || !std::isfinite(sigma_sq)) return false;
  fwhm = 2.3548200450309493 * std::sqrt(sigma_sq);
  return std::isfinite(fwhm) && fwhm > 0.2 && fwhm < 50.0;
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
    cv::goodFeaturesToTrack(residual, corners, kMaxControlStars, 0.001, 8.0,
                            detector_mask, 3, false, 0.04);
  } catch (...) {
    return unavailable("star_detection_failed");
  }
  if (corners.empty()) return unavailable("no_control_stars");

  std::vector<double> ratios;
  ratios.reserve(corners.size());
  for (const auto& corner : corners) {
    const int x = static_cast<int>(std::lround(corner.x));
    const int y = static_cast<int>(std::lround(corner.y));
    double fwhm_before = 0.0, fwhm_after = 0.0;
    if (!estimate_paired_fwhm(before, valid, x, y, fwhm_before) ||
        !estimate_paired_fwhm(after, valid, x, y, fwhm_after)) continue;
    const double ratio = fwhm_after / fwhm_before;
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
          {"method", "fixed_control_positions_annulus_aperture_moments_v1"},
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

void write_post_run_star_shape_artifact(const std::filesystem::path& run_dir,
                                        const std::filesystem::path& configured_hms_output,
                                        bool hms_enabled) {
  json diagnostic;
  try {
    auto final_path = configured_hms_output;
    if (final_path.is_relative()) final_path = run_dir / "outputs" / final_path;
    const auto before_path = run_dir / "outputs" / "reconstructed_G.fit";
    const auto mask_path = run_dir / "outputs" / "canvas_mask.fits";
    if (!hms_enabled || !std::filesystem::is_regular_file(before_path) ||
        !std::filesystem::is_regular_file(final_path) || !std::filesystem::is_regular_file(mask_path)) {
      diagnostic = unavailable("required_output_missing");
    } else {
      const auto before = io::read_fits_pixels_float(before_path);
      const auto after_rgb = io::read_fits_rgb(final_path);
      const auto mask = io::read_fits_pixels_float(mask_path);
      if (before.rows() != after_rgb.G.rows() || before.cols() != after_rgb.G.cols() ||
          before.rows() != mask.rows() || before.cols() != mask.cols()) {
        diagnostic = unavailable("raster_mismatch");
      } else {
        std::vector<uint8_t> valid(static_cast<size_t>(before.rows()) * before.cols(), 0);
        for (int y = 0; y < before.rows(); ++y)
          for (int x = 0; x < before.cols(); ++x)
            valid[static_cast<size_t>(y) * before.cols() + x] =
                std::isfinite(mask(y, x)) && mask(y, x) > 0.0f ? 1 : 0;
        diagnostic = measure_post_run_star_shape(before, after_rgb.G, valid);
      }
    }
  } catch (...) {
    diagnostic = unavailable("measurement_io_failed");
  }
  try {
    const auto artifact_dir = run_dir / "artifacts";
    std::filesystem::create_directories(artifact_dir);
    core::write_text_atomic(artifact_dir / "pi_post_run_star_shape.json", diagnostic.dump(2));
  } catch (...) {
    // This descriptive artifact must not turn a successful image run into a failure.
  }
}

}  // namespace tile_compile::reconstruction
