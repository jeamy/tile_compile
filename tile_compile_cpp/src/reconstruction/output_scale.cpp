#include "tile_compile/reconstruction/output_scale.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tile_compile::reconstruction {


namespace {
// overlap of interval [a, b] with [0, 1]
double overlap_unit(double a, double b) {
  return std::max(0.0, std::min(b, 1.0) - std::max(a, 0.0));
}
// K(0, x0) for a top-hat of side d centred at x0, integrated over cell [0,1]
double kernel_overlap(double x0, double d) {
  return overlap_unit(x0 - d / 2.0, x0 + d / 2.0);
}
}  // namespace

std::vector<double> kernel_noise_autocorrelation_1d(float pixfrac, int internal_scale,
                                                    int max_lag) {
  if (!(pixfrac > 0.0f) || pixfrac > 1.0f || (internal_scale != 1 && internal_scale != 2) ||
      max_lag < 0)
    throw std::invalid_argument("KERNEL_NOISE_INVALID_ARGS");
  const double d = static_cast<double>(pixfrac) * internal_scale;
  // Input samples at j + 0.5; a top-hat of side d reaches at most
  // ceil(d/2)+1 cells either way, so this range is generous.
  const int jr = static_cast<int>(std::ceil(d)) + max_lag + 3;

  auto s_at_lag = [&](int lag) {
    double s = 0.0;
    for (int j = -jr; j <= jr; ++j) {
      const double x0 = j + 0.5;
      s += kernel_overlap(x0, d) * kernel_overlap(x0 - lag, d);
    }
    return s;
  };
  const double s0 = s_at_lag(0);
  std::vector<double> rho(static_cast<size_t>(max_lag) + 1, 0.0);
  if (s0 <= 0.0) {
    rho[0] = 1.0;
    return rho;
  }
  for (int lag = 0; lag <= max_lag; ++lag) rho[static_cast<size_t>(lag)] = s_at_lag(lag) / s0;
  return rho;
}

double kernel_noise_correlation_sigma_factor(float pixfrac, int internal_scale) {
  if (!(pixfrac > 0.0f) || pixfrac > 1.0f || (internal_scale != 1 && internal_scale != 2))
    throw std::invalid_argument("KERNEL_NOISE_INVALID_ARGS");
  const double d = static_cast<double>(pixfrac) * internal_scale;
  const int jr = static_cast<int>(std::ceil(d)) + 4;
  double s0 = 0.0;
  for (int j = -jr; j <= jr; ++j) {
    const double k = kernel_overlap(j + 0.5, d);
    s0 += k * k;
  }
  // W = sum_k K(k, x0) = d for a top-hat of side d.
  return (s0 > 0.0) ? d / std::sqrt(s0) : 1.0;
}


}  // namespace tile_compile::reconstruction
