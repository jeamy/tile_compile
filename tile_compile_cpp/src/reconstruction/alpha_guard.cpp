#include "tile_compile/reconstruction/alpha_guard.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tile_compile::reconstruction {

namespace {

float nanf_() { return std::numeric_limits<float>::quiet_NaN(); }

double median_inplace(std::vector<float> &v) {
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  const size_t m = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + m, v.end());
  const double hi = v[m];
  if (v.size() % 2 == 1) return hi;
  const double lo = *std::max_element(v.begin(), v.begin() + m);
  return 0.5 * (lo + hi);
}

constexpr double kB3[5] = {1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0,
                           1.0 / 16.0};

}  // namespace

double mad_sigma(std::vector<float> values) {
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  const double med = median_inplace(values);
  std::vector<float> dev;
  dev.reserve(values.size());
  for (float v : values) dev.push_back(static_cast<float>(std::abs(v - med)));
  return 1.4826 * median_inplace(dev);
}

int energy_guard_window_radius(int level) {
  if (level < 1) throw std::invalid_argument("ENERGY_GUARD_LEVEL");
  return std::max(3, 1 << (level + 1));
}

std::vector<float> apply_energy_guard(const std::vector<float> &alpha_pre,
                                      const std::vector<float> &d_r_luma,
                                      const std::vector<float> &d_profile_luma,
                                      const std::vector<uint8_t> &support,
                                      int width, int height, int window_radius,
                                      double background_floor,
                                      const EnergyGuardParams &params) {
  if (width <= 0 || height <= 0)
    throw std::invalid_argument("ENERGY_GUARD_DIMENSIONS");
  const std::size_t n = static_cast<std::size_t>(width) * height;
  for (const auto *v :
       {&alpha_pre, &d_r_luma, &d_profile_luma})
    if (v->size() != n) throw std::invalid_argument("ENERGY_GUARD_SIZE");
  if (support.size() != n) throw std::invalid_argument("ENERGY_GUARD_SIZE");
  if (window_radius < 1) throw std::invalid_argument("ENERGY_GUARD_RADIUS");
  if (!(background_floor >= 0.0))
    throw std::invalid_argument("ENERGY_GUARD_FLOOR");
  if (params.bisection_iters < 1)
    throw std::invalid_argument("ENERGY_GUARD_ITERS");

  std::vector<float> out(n, 0.0f);
  std::vector<float> win_r, win_mix;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t i = static_cast<std::size_t>(y) * width + x;
      if (!support[i]) continue;
      const double a_pre =
          std::isfinite(alpha_pre[i]) ? std::clamp<double>(alpha_pre[i], 0.0, 1.0)
                                      : 0.0;
      if (a_pre <= 0.0) continue;  // out stays 0

      win_r.clear();
      win_mix.clear();  // filled per candidate alpha below
      std::vector<float> dr_vals, dp_vals;
      for (int yy = std::max(0, y - window_radius);
           yy <= std::min(height - 1, y + window_radius); ++yy)
        for (int xx = std::max(0, x - window_radius);
             xx <= std::min(width - 1, x + window_radius); ++xx) {
          const std::size_t j = static_cast<std::size_t>(yy) * width + xx;
          if (!support[j] || !std::isfinite(d_r_luma[j]) ||
              !std::isfinite(d_profile_luma[j]))
            continue;
          dr_vals.push_back(d_r_luma[j]);
          dp_vals.push_back(d_profile_luma[j]);
        }
      if (static_cast<int>(dr_vals.size()) < params.min_window_pixels) continue;

      const double mad_r = mad_sigma(dr_vals);
      const double scale_raw = std::max(mad_r, background_floor);
      auto ratio_at = [&](double a) -> double {
        std::vector<float> mix(dr_vals.size());
        for (std::size_t k = 0; k < dr_vals.size(); ++k)
          mix[k] = static_cast<float>(dr_vals[k] +
                                      a * (dp_vals[k] - dr_vals[k]));
        const double mad_mix = mad_sigma(std::move(mix));
        return scale_raw > 0.0 ? mad_mix / scale_raw
                               : (mad_mix > 0.0 ? std::numeric_limits<double>::infinity()
                                                : 0.0);
      };

      if (ratio_at(a_pre) <= params.energy_limit) {
        out[i] = static_cast<float>(a_pre);
        continue;
      }
      // Bisection: lo always feasible (ratio_at(0) = mad_r/scale_raw <= 1).
      double lo = 0.0, hi = a_pre;
      for (int it = 0; it < params.bisection_iters; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (ratio_at(mid) <= params.energy_limit)
          lo = mid;
        else
          hi = mid;
      }
      out[i] = static_cast<float>(lo);
    }
  }
  return out;
}

std::vector<float> smooth_alpha_b3(const std::vector<float> &alpha_guarded,
                                   const std::vector<uint8_t> &support,
                                   int width, int height) {
  if (width <= 0 || height <= 0)
    throw std::invalid_argument("ALPHA_SMOOTH_DIMENSIONS");
  const std::size_t n = static_cast<std::size_t>(width) * height;
  if (alpha_guarded.size() != n || support.size() != n)
    throw std::invalid_argument("ALPHA_SMOOTH_SIZE");

  // 4-connected component labels of the support (plan 14.7: no convolution
  // across separate support islands).
  std::vector<int> label(n, -1);
  int next = 0;
  std::vector<std::size_t> stack;
  for (std::size_t s = 0; s < n; ++s) {
    if (!support[s] || label[s] >= 0) continue;
    label[s] = next;
    stack.push_back(s);
    while (!stack.empty()) {
      const std::size_t p = stack.back();
      stack.pop_back();
      const int px = static_cast<int>(p % width);
      const int py = static_cast<int>(p / width);
      const int dx[4] = {-1, 1, 0, 0};
      const int dy[4] = {0, 0, -1, 1};
      for (int k = 0; k < 4; ++k) {
        const int nx = px + dx[k], ny = py + dy[k];
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
        const std::size_t q = static_cast<std::size_t>(ny) * width + nx;
        if (support[q] && label[q] < 0) {
          label[q] = next;
          stack.push_back(q);
        }
      }
    }
    ++next;
  }

  auto masked_num = [&](bool weighted) {
    // separable B3 over (alpha*support) if weighted else support, restricted
    // to the centre pixel's component.
    std::vector<double> src(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      if (support[i])
        src[i] = weighted ? std::max(0.0f, alpha_guarded[i]) : 1.0;
    std::vector<double> hx(n, 0.0);
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x) {
        const std::size_t i = static_cast<std::size_t>(y) * width + x;
        if (label[i] < 0) continue;
        double acc = 0.0;
        for (int t = -2; t <= 2; ++t) {
          const int xx = x + t;
          if (xx < 0 || xx >= width) continue;
          const std::size_t j = static_cast<std::size_t>(y) * width + xx;
          if (label[j] != label[i]) continue;
          acc += kB3[t + 2] * src[j];
        }
        hx[i] = acc;
      }
    std::vector<double> out(n, 0.0);
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x) {
        const std::size_t i = static_cast<std::size_t>(y) * width + x;
        if (label[i] < 0) continue;
        double acc = 0.0;
        for (int t = -2; t <= 2; ++t) {
          const int yy = y + t;
          if (yy < 0 || yy >= height) continue;
          const std::size_t j = static_cast<std::size_t>(yy) * width + x;
          if (label[j] != label[i]) continue;
          acc += kB3[t + 2] * hx[j];
        }
        out[i] = acc;
      }
    return out;
  };

  const auto num = masked_num(true);
  const auto den = masked_num(false);

  std::vector<float> final_(n, nanf_());
  for (std::size_t i = 0; i < n; ++i) {
    if (!support[i]) continue;
    const double blur = den[i] > 0.0 ? num[i] / den[i] : 0.0;
    const double g = std::max(0.0f, alpha_guarded[i]);
    final_[i] = static_cast<float>(std::min(g, blur));  // mandatory min cap
  }
  return final_;
}

}  // namespace tile_compile::reconstruction
