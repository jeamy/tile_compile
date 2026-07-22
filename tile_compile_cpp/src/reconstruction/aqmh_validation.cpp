#include "tile_compile/reconstruction/aqmh_validation.hpp"

#include "tile_compile/core/utils.hpp"
#include "tile_compile/metrics/aqmh_eps.hpp"
#include "tile_compile/metrics/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace tile_compile::reconstruction {
namespace {

// Safe regression: returns 0 when both values are degenerate (near-zero or
// non-finite). This prevents artificial large regressions when control has
// no meaningful signal (e.g. background_rms=0 on a synthetic image, or
// tail metrics on a star-free field).
float regression(float value, float control) {
  if (!std::isfinite(value) || !std::isfinite(control))
    return 0.0f;
  const float denom = std::max(std::abs(control),
                               metrics::eps_scale({value, control}));
  // If both value and control are near-zero, the regression is meaningless.
  if (std::abs(value) < 1.0e-12f && std::abs(control) < 1.0e-12f)
    return 0.0f;
  return (value - control) / denom;
}

struct StarSample {
  int x = 0;
  int y = 0;
  float peak = 0.0f;
};

bool mask_valid(const std::vector<uint8_t> &mask, int width, int height,
                int x, int y) {
  if (mask.empty()) return true;
  if (mask.size() != static_cast<size_t>(width) * height) return false;
  return mask[static_cast<size_t>(y) * width + x] != 0u;
}

float percentile(std::vector<float> values, float q) {
  if (values.empty()) return 0.0f;
  const size_t idx = static_cast<size_t>(std::clamp(q, 0.0f, 1.0f) *
                                        static_cast<float>(values.size() - 1));
  std::nth_element(values.begin(), values.begin() + idx, values.end());
  return values[idx];
}

float robust_local_noise_sigma(const Matrix2Df &image,
                               const std::vector<uint8_t> &validation_mask) {
  if (image.rows() <= 0 || image.cols() <= 0) return 0.0f;

  // First differences reject slowly varying astronomical signal while
  // retaining pixel-scale noise. Their sigma is sqrt(2) times source noise.
  std::vector<float> differences;
  differences.reserve(static_cast<size_t>(image.rows()) *
                      static_cast<size_t>(image.cols()) * 2u);
  constexpr float inv_sqrt_two = 0.7071067811865475f;
  for (int y = 0; y < image.rows(); ++y) {
    for (int x = 0; x < image.cols(); ++x) {
      if (!mask_valid(validation_mask, image.cols(), image.rows(), x, y))
        continue;
      const float v = image(y, x);
      if (!std::isfinite(v)) continue;
      if (x + 1 < image.cols() &&
          mask_valid(validation_mask, image.cols(), image.rows(), x + 1, y)) {
        const float right = image(y, x + 1);
        if (std::isfinite(right))
          differences.push_back((v - right) * inv_sqrt_two);
      }
      if (y + 1 < image.rows() &&
          mask_valid(validation_mask, image.cols(), image.rows(), x, y + 1)) {
        const float below = image(y + 1, x);
        if (std::isfinite(below))
          differences.push_back((v - below) * inv_sqrt_two);
      }
    }
  }
  if (differences.empty()) return 0.0f;
  const float median = metrics::aqmh_median(differences);
  return tile_compile::core::kMadToSigma * metrics::aqmh_mad(differences, median);
}

std::vector<StarSample> detect_validation_stars(
    const Matrix2Df &image, const std::vector<uint8_t> &validation_mask) {
  constexpr int border = 16;
  constexpr int max_stars = 250;
  std::vector<StarSample> stars;
  if (image.rows() < 2 * border + 1 || image.cols() < 2 * border + 1)
    return stars;

  std::vector<float> finite;
  finite.reserve(static_cast<size_t>(image.size()));
  for (int y = 0; y < image.rows(); ++y)
    for (int x = 0; x < image.cols(); ++x)
      if (mask_valid(validation_mask, image.cols(), image.rows(), x, y) &&
          std::isfinite(image(y, x)))
        finite.push_back(image(y, x));
  if (finite.empty()) return stars;
  const float med = metrics::aqmh_median(finite);
  const float sigma = std::max(tile_compile::core::kMadToSigma * metrics::aqmh_mad(finite, med),
                               metrics::eps_scale(finite));
  const float threshold = std::max(med + 8.0f * sigma,
                                   percentile(finite, 0.999f));
  const float sat = percentile(finite, 0.9998f);

  stars.reserve(max_stars);
  for (int y = border; y < image.rows() - border; ++y) {
    for (int x = border; x < image.cols() - border; ++x) {
      const float peak = image(y, x);
      if (!(std::isfinite(peak) && peak >= threshold && peak < sat)) continue;
      bool is_max = true;
      for (int dy = -2; dy <= 2 && is_max; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
          if (dy == 0 && dx == 0) continue;
          const float v = image(y + dy, x + dx);
          if (std::isfinite(v) && v > peak) {
            is_max = false;
            break;
          }
        }
      }
      if (!is_max) continue;
      bool too_close = false;
      for (const auto &s : stars) {
        const int dx = s.x - x;
        const int dy = s.y - y;
        if (dx * dx + dy * dy < 12 * 12) {
          too_close = true;
          break;
        }
      }
      if (too_close) continue;
      stars.push_back({x, y, peak});
      if (stars.size() >= max_stars) break;
    }
    if (stars.size() >= max_stars) break;
  }

  return stars;
}

void measure_star_tail_metrics_at_samples(
    const Matrix2Df &image, const std::vector<StarSample> &stars,
    const std::vector<uint8_t> &validation_mask, AqmhValidationMetrics &out) {
  constexpr float pi = 3.14159265358979323846f;
  const float target = -3.0f * pi / 4.0f;
  const float opposite = pi / 4.0f;
  auto angle_diff = [](float a, float b) {
    float d = std::fmod(a - b + 3.0f * 3.14159265358979323846f,
                       2.0f * 3.14159265358979323846f) -
              3.14159265358979323846f;
    return std::abs(d);
  };

  std::vector<float> tail_abs;
  std::vector<float> tail_raw;
  std::vector<float> elongations;
  for (const auto &s : stars) {
    constexpr int sample_radius = 14;
    if (s.x < sample_radius || s.y < sample_radius ||
        s.x + sample_radius >= image.cols() ||
        s.y + sample_radius >= image.rows())
      continue;
    bool support_complete = true;
    for (int dy = -sample_radius; dy <= sample_radius && support_complete; ++dy)
      for (int dx = -sample_radius; dx <= sample_radius; ++dx)
        if (!mask_valid(validation_mask, image.cols(), image.rows(),
                        s.x + dx, s.y + dy)) {
          support_complete = false;
          break;
        }
    if (!support_complete) continue;
    float bg_values[76];
    int bg_count = 0;
    for (int dy = -14; dy <= 14; ++dy) {
      for (int dx = -14; dx <= 14; ++dx) {
        const float r = std::sqrt(static_cast<float>(dx * dx + dy * dy));
        if (r >= 10.0f && r <= 14.5f && bg_count < 76) {
          const float v = image(s.y + dy, s.x + dx);
          if (std::isfinite(v)) bg_values[bg_count++] = v;
        }
      }
    }
    if (bg_count < 8) continue;
    std::nth_element(bg_values, bg_values + bg_count / 2,
                     bg_values + bg_count);
    const float bg = bg_values[bg_count / 2];

    double nw = 0.0, se = 0.0, full = 0.0;
    int nw_n = 0, se_n = 0, full_n = 0;
    double sum = 0.0, sx = 0.0, sy = 0.0;
    for (int dy = -14; dy <= 14; ++dy) {
      for (int dx = -14; dx <= 14; ++dx) {
        const float r = std::sqrt(static_cast<float>(dx * dx + dy * dy));
        const float v_raw = image(s.y + dy, s.x + dx);
        if (!std::isfinite(v_raw)) continue;
        const float v = std::max(0.0f, v_raw - bg);
        if (r >= 3.5f && r <= 12.0f) {
          const float a = std::atan2(static_cast<float>(dy),
                                     static_cast<float>(dx));
          full += v; ++full_n;
          if (angle_diff(a, target) <= 25.0f * pi / 180.0f) {
            nw += v; ++nw_n;
          }
          if (angle_diff(a, opposite) <= 25.0f * pi / 180.0f) {
            se += v; ++se_n;
          }
        }
        if (r <= 8.0f) {
          sum += v;
          sx += v * dx;
          sy += v * dy;
        }
      }
    }
    if (full_n == 0 || nw_n == 0 || se_n == 0 || !(full > 0.0)) continue;
    const float score =
        static_cast<float>((nw / nw_n - se / se_n) / std::max(full / full_n,
                                                              1.0e-6));
    tail_raw.push_back(score);
    tail_abs.push_back(std::abs(score));

    if (sum > 0.0) {
      const double mx = sx / sum;
      const double my = sy / sum;
      double xx = 0.0, yy = 0.0, xy = 0.0;
      for (int dy = -8; dy <= 8; ++dy) {
        for (int dx = -8; dx <= 8; ++dx) {
          const float r = std::sqrt(static_cast<float>(dx * dx + dy * dy));
          if (r > 8.0f) continue;
          const float v = std::max(0.0f, image(s.y + dy, s.x + dx) - bg);
          xx += v * (dx - mx) * (dx - mx);
          yy += v * (dy - my) * (dy - my);
          xy += v * (dx - mx) * (dy - my);
        }
      }
      xx /= sum; yy /= sum; xy /= sum;
      const double tr = xx + yy;
      const double det = xx * yy - xy * xy;
      const double disc = std::max(0.0, tr * tr / 4.0 - det);
      const double l1 = tr / 2.0 + std::sqrt(disc);
      const double l2 = std::max(tr / 2.0 - std::sqrt(disc), 1.0e-6);
      elongations.push_back(static_cast<float>(std::sqrt(l1 / l2)));
    }
  }

  out.star_count = static_cast<int>(tail_abs.size());
  if (!tail_abs.empty()) {
    out.tail11_abs_median = percentile(tail_abs, 0.5f);
    out.tail11_p90 = percentile(tail_raw, 0.9f);
  }
  if (!elongations.empty()) {
    out.elongation_median = percentile(elongations, 0.5f);
  }
}

AqmhValidationMetrics measure_aqmh_validation_metrics_at_samples(
    const Matrix2Df &image, const std::vector<StarSample> &stars,
    const std::vector<uint8_t> &validation_mask) {
  AqmhValidationMetrics out;
  if (image.rows() <= 0 || image.cols() <= 0) return out;
  std::vector<float> finite;
  finite.reserve(static_cast<size_t>(image.size()));
  double gradient_sum = 0.0;
  uint64_t gradient_count = 0;
  for (int y = 0; y < image.rows(); ++y) {
    for (int x = 0; x < image.cols(); ++x) {
      if (!mask_valid(validation_mask, image.cols(), image.rows(), x, y))
        continue;
      const float v = image(y, x);
      if (!std::isfinite(v)) continue;
      finite.push_back(v);
      if (x + 1 < image.cols() &&
          mask_valid(validation_mask, image.cols(), image.rows(), x + 1, y) &&
          std::isfinite(image(y, x + 1))) {
        gradient_sum += std::abs(v - image(y, x + 1));
        ++gradient_count;
      }
      if (y + 1 < image.rows() &&
          mask_valid(validation_mask, image.cols(), image.rows(), x, y + 1) &&
          std::isfinite(image(y + 1, x))) {
        gradient_sum += std::abs(v - image(y + 1, x));
        ++gradient_count;
      }
    }
  }
  if (finite.empty()) return out;
  const float med = metrics::aqmh_median(finite);
  const float sigma = tile_compile::core::kMadToSigma * metrics::aqmh_mad(finite, med);
  Matrix2Df fwhm_image = image;
  if (!validation_mask.empty()) {
    for (int y = 0; y < image.rows(); ++y)
      for (int x = 0; x < image.cols(); ++x)
        if (!mask_valid(validation_mask, image.cols(), image.rows(), x, y))
          fwhm_image(y, x) = med;
  }
  out.seam_score = gradient_count > 0
      ? static_cast<float>(gradient_sum / gradient_count) /
            std::max(sigma, metrics::eps_scale(finite))
      : 0.0f;
  out.background_rms = robust_local_noise_sigma(image, validation_mask);
  // FWHM is a scale-invariant comparison metric here, but its corner/PSF
  // fitting cost grows sharply with the full canvas size. Keep the original
  // image for background, seam, and star-tail metrics; use a bounded-size
  // copy only for the FWHM estimate so validation remains usable on large
  // astronomical canvases.
  constexpr int max_fwhm_dimension = 800;
  if (std::max(image.rows(), image.cols()) > max_fwhm_dimension) {
    const float scale = static_cast<float>(max_fwhm_dimension) /
                        static_cast<float>(std::max(image.rows(), image.cols()));
    const int rows = std::max(1, static_cast<int>(std::lround(image.rows() * scale)));
    const int cols = std::max(1, static_cast<int>(std::lround(image.cols() * scale)));
    cv::Mat source(fwhm_image.rows(), fwhm_image.cols(), CV_32F,
                   fwhm_image.data(),
                   static_cast<size_t>(fwhm_image.outerStride()) * sizeof(float));
    Matrix2Df reduced(rows, cols);
    cv::Mat target(rows, cols, CV_32F, reduced.data(),
                   static_cast<size_t>(reduced.outerStride()) * sizeof(float));
    cv::resize(source, target, target.size(), 0.0, 0.0, cv::INTER_AREA);
    out.fwhm = metrics::measure_fwhm_from_image(reduced, 80, 8, 6);
  } else {
    out.fwhm = metrics::measure_fwhm_from_image(fwhm_image);
  }
  if (!std::isfinite(out.fwhm) || out.fwhm < 0.0f) out.fwhm = 0.0f;
  measure_star_tail_metrics_at_samples(image, stars, validation_mask, out);
  return out;
}

} // namespace

AqmhValidationMetrics measure_aqmh_validation_metrics(
    const Matrix2Df &image, const std::vector<uint8_t> &validation_mask) {
  return measure_aqmh_validation_metrics_at_samples(
      image, detect_validation_stars(image, validation_mask), validation_mask);
}

AqmhValidationComparison compare_aqmh_to_uniform_control(
    const Matrix2Df &aqmh, const Matrix2Df &control,
    const std::vector<uint8_t> &validation_mask) {
  AqmhValidationComparison out;
  // Candidate and control must use the same stars. Independent detections
  // compare different populations when sharpness or contrast changes and can
  // manufacture a tail/elongation regression that is not present in the
  // image pair.
  const auto common_stars = detect_validation_stars(control, validation_mask);
  out.aqmh = measure_aqmh_validation_metrics_at_samples(
      aqmh, common_stars, validation_mask);
  out.control = measure_aqmh_validation_metrics_at_samples(
      control, common_stars, validation_mask);
  out.seam_score_regression = regression(out.aqmh.seam_score,
                                         out.control.seam_score);
  out.fwhm_regression = regression(out.aqmh.fwhm, out.control.fwhm);
  out.background_rms_regression = regression(out.aqmh.background_rms,
                                             out.control.background_rms);
  out.tail11_abs_regression = regression(out.aqmh.tail11_abs_median,
                                         out.control.tail11_abs_median);
  out.elongation_regression = regression(out.aqmh.elongation_median,
                                         out.control.elongation_median);

  // Applicability: metrics are only comparable when the control side has a
  // finite, non-degenerate reference value. Non-applicable metrics must never
  // trigger fallback decisions.
  out.fwhm_applicable = out.aqmh.fwhm > 0.0f && out.control.fwhm > 0.0f;
  const float seam_eps = metrics::eps_scale(
      {out.aqmh.seam_score, out.control.seam_score});
  out.seam_applicable =
      std::isfinite(out.aqmh.seam_score) &&
      std::isfinite(out.control.seam_score) &&
      out.control.seam_score > std::max(seam_eps, 1.0e-12f);
  const float background_eps = metrics::eps_scale(
      {out.aqmh.background_rms, out.control.background_rms});
  out.background_rms_applicable =
      std::isfinite(out.aqmh.background_rms) &&
      std::isfinite(out.control.background_rms) &&
      out.control.background_rms > std::max(background_eps, 1.0e-12f);
  // Tail and elongation require sufficient stars in BOTH images
  constexpr int min_stars_for_tail = 12;
  out.tail_applicable = out.aqmh.star_count >= min_stars_for_tail &&
                        out.control.star_count >= min_stars_for_tail;
  out.elongation_applicable = out.tail_applicable;

  // Zero out regressions for non-applicable metrics to prevent spurious triggers
  if (!out.fwhm_applicable) out.fwhm_regression = 0.0f;
  if (!out.seam_applicable) out.seam_score_regression = 0.0f;
  if (!out.background_rms_applicable) out.background_rms_regression = 0.0f;
  if (!out.tail_applicable) out.tail11_abs_regression = 0.0f;
  if (!out.elongation_applicable) out.elongation_regression = 0.0f;

  return out;
}

} // namespace tile_compile::reconstruction
