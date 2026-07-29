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

float median_inplace(std::vector<float> &values) {
  if (values.empty()) return std::numeric_limits<float>::quiet_NaN();
  const size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + mid, values.end());
  const float upper = values[mid];
  if (values.size() % 2 != 0)
    return upper;
  const float lower =
      *std::max_element(values.begin(), values.begin() + mid);
  return 0.5f * (lower + upper);
}

struct RobustFiniteStats {
  float median = 0.0f;
  float mad = 0.0f;
  float eps = std::nextafter(0.0f, 1.0f);
};

RobustFiniteStats robust_finite_stats_inplace(std::vector<float> &values) {
  RobustFiniteStats out;
  if (values.empty()) return out;
  out.median = median_inplace(values);

  std::vector<float> work(values.size());
  for (size_t i = 0; i < values.size(); ++i)
    work[i] = std::abs(values[i] - out.median);
  out.mad = median_inplace(work);

  for (size_t i = 0; i < values.size(); ++i)
    work[i] = std::abs(values[i]);
  const float median_abs = median_inplace(work);
  out.eps = std::max(std::nextafter(0.0f, 1.0f),
                     metrics::aqmh_eps_rel *
                         std::max(median_abs, out.mad));
  return out;
}

float robust_noise_from_differences_inplace(
    std::vector<float> &differences) {
  if (differences.empty()) return 0.0f;
  const float median = median_inplace(differences);
  for (float &value : differences)
    value = std::abs(value - median);
  const float mad = median_inplace(differences);
  return tile_compile::core::kMadToSigma * mad;
}

RobustFiniteStats finite_image_stats(
    const Matrix2Df &image,
    const std::vector<uint8_t> &validation_mask,
    std::vector<float> &finite) {
  finite.clear();
  finite.reserve(static_cast<size_t>(image.size()));
  for (int y = 0; y < image.rows(); ++y) {
    for (int x = 0; x < image.cols(); ++x) {
      if (mask_valid(validation_mask, image.cols(), image.rows(), x, y) &&
          std::isfinite(image(y, x))) {
        finite.push_back(image(y, x));
      }
    }
  }
  return robust_finite_stats_inplace(finite);
}

std::vector<AqmhValidationStarSample> detect_validation_stars(
    const Matrix2Df &image, const std::vector<uint8_t> &validation_mask) {
  constexpr int border = 16;
  constexpr int max_stars = 250;
  std::vector<AqmhValidationStarSample> stars;
  if (image.rows() < 2 * border + 1 || image.cols() < 2 * border + 1)
    return stars;

  std::vector<float> finite;
  const auto stats = finite_image_stats(image, validation_mask, finite);
  if (finite.empty()) return stars;
  const float sigma =
      std::max(tile_compile::core::kMadToSigma * stats.mad, stats.eps);
  const float threshold =
      std::max(stats.median + 8.0f * sigma,
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
    const Matrix2Df &image,
    const std::vector<AqmhValidationStarSample> &stars,
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
    const Matrix2Df &image,
    const std::vector<AqmhValidationStarSample> &stars,
    const std::vector<uint8_t> &validation_mask) {
  AqmhValidationMetrics out;
  if (image.rows() <= 0 || image.cols() <= 0) return out;
  std::vector<float> finite;
  finite.reserve(static_cast<size_t>(image.size()));
  std::vector<float> differences;
  differences.reserve(static_cast<size_t>(image.size()) * 2u);
  double gradient_sum = 0.0;
  uint64_t gradient_count = 0;
  constexpr float inv_sqrt_two = 0.7071067811865475f;
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
        const float delta = v - image(y, x + 1);
        gradient_sum += std::abs(delta);
        differences.push_back(delta * inv_sqrt_two);
        ++gradient_count;
      }
      if (y + 1 < image.rows() &&
          mask_valid(validation_mask, image.cols(), image.rows(), x, y + 1) &&
          std::isfinite(image(y + 1, x))) {
        const float delta = v - image(y + 1, x);
        gradient_sum += std::abs(delta);
        differences.push_back(delta * inv_sqrt_two);
        ++gradient_count;
      }
    }
  }
  if (finite.empty()) return out;
  const auto stats = robust_finite_stats_inplace(finite);
  const float med = stats.median;
  const float sigma = tile_compile::core::kMadToSigma * stats.mad;
  Matrix2Df fwhm_image = image;
  if (!validation_mask.empty()) {
    for (int y = 0; y < image.rows(); ++y)
      for (int x = 0; x < image.cols(); ++x)
        if (!mask_valid(validation_mask, image.cols(), image.rows(), x, y))
          fwhm_image(y, x) = med;
  }
  out.seam_score = gradient_count > 0
      ? static_cast<float>(gradient_sum / gradient_count) /
            std::max(sigma, stats.eps)
      : 0.0f;
  out.background_rms =
      robust_noise_from_differences_inplace(differences);
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

AqmhRawBaselineGuardDecision aqmh_raw_baseline_guard_decision(
    const AqmhValidationComparison &candidate_vs_raw,
    const AqmhValidationComparison &raw_vs_control,
    const AqmhValidationComparison &candidate_vs_control,
    const config::AqmhValidationConfig &cfg) {
  auto background_ok = [](const AqmhValidationComparison &v, float threshold) {
    return !v.background_rms_applicable ||
           v.background_rms_regression <= threshold;
  };
  auto fwhm_ok = [](const AqmhValidationComparison &v, float threshold) {
    return !v.fwhm_applicable || v.fwhm_regression <= threshold;
  };
  auto seam_ok = [](const AqmhValidationComparison &v, float threshold) {
    return !v.seam_applicable || v.seam_score_regression <= threshold;
  };
  auto tail_ok = [](const AqmhValidationComparison &v, float tail_threshold,
                    float elongation_threshold) {
    return !v.tail_applicable ||
           (v.tail11_abs_regression <= tail_threshold &&
            v.elongation_regression <= elongation_threshold);
  };
  auto all_ok = [&](const AqmhValidationComparison &v) {
    return background_ok(v, cfg.max_background_rms_regression) &&
           fwhm_ok(v, cfg.max_fwhm_regression) &&
           seam_ok(v, cfg.max_seam_score_regression) &&
           tail_ok(v, cfg.max_tail11_abs_regression,
                   cfg.max_elongation_regression);
  };

  if (all_ok(candidate_vs_raw)) {
    return {true, false, "strict_raw_baseline_pass"};
  }
  if (all_ok(raw_vs_control)) {
    return {false, false, "raw_baseline_valid_and_candidate_regresses_raw"};
  }
  if (!all_ok(candidate_vs_control)) {
    return {false, false, "candidate_fails_uniform_control"};
  }

  bool repairs_failed_raw_gate = false;
  if (!background_ok(raw_vs_control, cfg.max_background_rms_regression)) {
    repairs_failed_raw_gate =
        repairs_failed_raw_gate ||
        (!candidate_vs_raw.background_rms_applicable ||
         candidate_vs_raw.background_rms_regression <= 0.0f);
  }
  if (!fwhm_ok(raw_vs_control, cfg.max_fwhm_regression)) {
    repairs_failed_raw_gate =
        repairs_failed_raw_gate ||
        (!candidate_vs_raw.fwhm_applicable ||
         candidate_vs_raw.fwhm_regression <= 0.0f);
  }
  if (!seam_ok(raw_vs_control, cfg.max_seam_score_regression)) {
    repairs_failed_raw_gate =
        repairs_failed_raw_gate ||
        (!candidate_vs_raw.seam_applicable ||
         candidate_vs_raw.seam_score_regression <= 0.0f);
  }
  if (!tail_ok(raw_vs_control, cfg.max_tail11_abs_regression,
               cfg.max_elongation_regression)) {
    repairs_failed_raw_gate =
        repairs_failed_raw_gate ||
        (!candidate_vs_raw.tail_applicable ||
         (candidate_vs_raw.tail11_abs_regression <= 0.0f &&
          candidate_vs_raw.elongation_regression <= 0.0f));
  }
  if (!repairs_failed_raw_gate) {
    return {false, false, "candidate_does_not_repair_failed_raw_gate"};
  }

  const bool background_relaxed_ok =
      background_ok(candidate_vs_raw, cfg.max_background_rms_regression) ||
      (candidate_vs_raw.background_rms_applicable &&
       candidate_vs_raw.background_rms_regression <= 0.0f);
  const bool fwhm_relaxed_ok =
      fwhm_ok(candidate_vs_raw, cfg.max_fwhm_regression) ||
      (candidate_vs_raw.fwhm_applicable &&
       candidate_vs_raw.fwhm_regression <= 0.0f);
  const bool seam_relaxed_ok =
      seam_ok(candidate_vs_raw, cfg.max_seam_score_regression * 2.0f);
  const bool tail_relaxed_ok =
      tail_ok(candidate_vs_raw, cfg.max_tail11_abs_regression,
              cfg.max_elongation_regression);
  if (background_relaxed_ok && fwhm_relaxed_ok && seam_relaxed_ok &&
      tail_relaxed_ok) {
    return {true, true, "raw_invalid_candidate_repairs_failed_gate"};
  }
  return {false, false, "candidate_exceeds_relaxed_raw_baseline_guard"};
}

AqmhValidationMetrics measure_aqmh_validation_metrics(
    const Matrix2Df &image, const std::vector<uint8_t> &validation_mask) {
  return measure_aqmh_validation_metrics_at_samples(
      image, detect_validation_stars(image, validation_mask), validation_mask);
}

AqmhValidationReference prepare_aqmh_validation_reference(
    const Matrix2Df &image, const std::vector<uint8_t> &validation_mask) {
  AqmhValidationReference reference;
  reference.width = static_cast<int>(image.cols());
  reference.height = static_cast<int>(image.rows());
  reference.stars = detect_validation_stars(image, validation_mask);
  reference.metrics = measure_aqmh_validation_metrics_at_samples(
      image, reference.stars, validation_mask);
  return reference;
}

AqmhValidationComparison compare_aqmh_to_reference(
    const Matrix2Df &aqmh, const AqmhValidationReference &control,
    const std::vector<uint8_t> &validation_mask) {
  AqmhValidationComparison out;
  if (aqmh.cols() != control.width || aqmh.rows() != control.height) {
    return out;
  }
  out.aqmh = measure_aqmh_validation_metrics_at_samples(
      aqmh, control.stars, validation_mask);
  out.control = control.metrics;
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

AqmhValidationComparison compare_aqmh_to_uniform_control(
    const Matrix2Df &aqmh, const Matrix2Df &control,
    const std::vector<uint8_t> &validation_mask) {
  // Candidate and control must use the same stars. Independent detections
  // compare different populations when sharpness or contrast changes and can
  // manufacture a tail/elongation regression that is not present in the
  // image pair.
  return compare_aqmh_to_reference(
      aqmh, prepare_aqmh_validation_reference(control, validation_mask),
      validation_mask);
}

} // namespace tile_compile::reconstruction
