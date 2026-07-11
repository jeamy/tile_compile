#include "tile_compile/reconstruction/aqmh_validation.hpp"

#include "tile_compile/core/utils.hpp"
#include "tile_compile/metrics/aqmh_eps.hpp"
#include "tile_compile/metrics/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace tile_compile::reconstruction {
namespace {

float regression(float value, float control) {
  return (value - control) /
         std::max(std::abs(control), metrics::eps_scale({value, control}));
}

struct StarSample {
  int x = 0;
  int y = 0;
  float peak = 0.0f;
};

float percentile(std::vector<float> values, float q) {
  if (values.empty()) return 0.0f;
  const size_t idx = static_cast<size_t>(std::clamp(q, 0.0f, 1.0f) *
                                        static_cast<float>(values.size() - 1));
  std::nth_element(values.begin(), values.begin() + idx, values.end());
  return values[idx];
}

void measure_star_tail_metrics(const Matrix2Df &image,
                               AqmhValidationMetrics &out) {
  constexpr int border = 16;
  constexpr int max_stars = 250;
  if (image.rows() < 2 * border + 1 || image.cols() < 2 * border + 1) return;

  std::vector<float> finite;
  finite.reserve(static_cast<size_t>(image.size()));
  for (int y = 0; y < image.rows(); ++y)
    for (int x = 0; x < image.cols(); ++x)
      if (std::isfinite(image(y, x))) finite.push_back(image(y, x));
  if (finite.empty()) return;
  const float med = metrics::aqmh_median(finite);
  const float sigma = std::max(1.4826f * metrics::aqmh_mad(finite, med),
                               metrics::eps_scale(finite));
  const float threshold = std::max(med + 8.0f * sigma,
                                   percentile(finite, 0.999f));
  const float sat = percentile(finite, 0.9998f);

  std::vector<StarSample> stars;
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
  measure_star_tail_metrics(image, out);
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
  out.tail11_abs_regression = regression(out.aqmh.tail11_abs_median,
                                         out.control.tail11_abs_median);
  out.elongation_regression = regression(out.aqmh.elongation_median,
                                         out.control.elongation_median);
  return out;
}

} // namespace tile_compile::reconstruction
