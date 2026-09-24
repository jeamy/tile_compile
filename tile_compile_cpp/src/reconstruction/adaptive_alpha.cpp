#include "tile_compile/reconstruction/adaptive_alpha.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tile_compile::reconstruction {

double alpha_smoothstep(double edge0, double edge1, double x) {
  if (!(edge1 > edge0)) return x >= edge1 ? 1.0 : 0.0;
  const double t = std::clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

namespace {

std::array<const ProfilePlane *, 3> channel_planes(
    const ForwardDrizzleUniformResult &p, ColorMode mode) {
  if (mode == ColorMode::MONO) return {&p.L, nullptr, nullptr};
  return {&p.R, &p.G, &p.B};
}

bool plane_ok(const ProfilePlane *p, std::size_t n) {
  return p && p->width > 0 && p->value.size() == n &&
         p->weight_sum.size() == n && p->n_eff.size() == n &&
         p->support.size() == n;
}

float ext_at(const std::vector<float> &m, std::size_t i) {
  if (m.empty()) return 1.0f;
  const float v = m[i];
  return std::isfinite(v) ? std::clamp(v, 0.0f, 1.0f) : 0.0f;
}

}  // namespace

std::vector<std::vector<float>> compute_adaptive_alpha(
    const ForwardDrizzleUniformResult &uniform,
    const ForwardDrizzleUniformResult &fine,
    const ForwardDrizzleUniformResult &medium, ColorMode mode, int width,
    int height, int levels, const AdaptiveAlphaParams &params,
    const std::vector<float> &a_separation, const std::vector<float> &a_artifact,
    const std::vector<float> &a_registration) {
  if (levels < 1 || levels > 4)
    throw std::invalid_argument("ADAPTIVE_ALPHA_LEVELS_RANGE");
  if (width <= 0 || height <= 0)
    throw std::invalid_argument("ADAPTIVE_ALPHA_DIMENSIONS");
  if (!(params.full_effective_samples > params.min_effective_samples))
    throw std::invalid_argument("ADAPTIVE_ALPHA_EFF_SAMPLE_ORDER");
  if (params.alpha_cap < 0.0f || params.alpha_cap > 1.0f)
    throw std::invalid_argument("ADAPTIVE_ALPHA_CAP_RANGE");

  const std::size_t n = static_cast<std::size_t>(width) * height;
  for (const auto *m : {&a_separation, &a_artifact, &a_registration})
    if (!m->empty() && m->size() != n)
      throw std::invalid_argument("ADAPTIVE_ALPHA_EXT_MAP_SIZE");

  const int nch = mode == ColorMode::MONO ? 1 : 3;
  const auto u_ch = channel_planes(uniform, mode);
  for (int c = 0; c < nch; ++c)
    if (!plane_ok(u_ch[c], n))
      throw std::invalid_argument("ADAPTIVE_ALPHA_UNIFORM_PLANE");

  auto band_alpha = [&](const ForwardDrizzleUniformResult &profile)
      -> std::vector<float> {
    const auto p_ch = channel_planes(profile, mode);
    for (int c = 0; c < nch; ++c)
      if (!plane_ok(p_ch[c], n))
        throw std::invalid_argument("ADAPTIVE_ALPHA_PROFILE_PLANE");
    std::vector<float> alpha(n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
      double a_neff = std::numeric_limits<double>::infinity();
      double a_cov = std::numeric_limits<double>::infinity();
      int active = 0;
      for (int c = 0; c < nch; ++c) {
        if (!p_ch[c]->support[i] || !u_ch[c]->support[i]) continue;
        ++active;
        a_neff = std::min(
            a_neff, alpha_smoothstep(params.min_effective_samples,
                                     params.full_effective_samples,
                                     p_ch[c]->n_eff[i]));
        const double uw = u_ch[c]->weight_sum[i];
        const double pw = p_ch[c]->weight_sum[i];
        const double cov = uw > 0.0 ? std::clamp(pw / uw, 0.0, 1.0) : 0.0;
        a_cov = std::min(a_cov, cov);
      }
      if (active == 0) { alpha[i] = 0.0f; continue; }
      const double a =
          static_cast<double>(params.alpha_cap) * a_neff * a_cov *
          ext_at(a_separation, i) * ext_at(a_artifact, i) *
          ext_at(a_registration, i);
      alpha[i] = static_cast<float>(std::clamp(a, 0.0, 1.0));
    }
    return alpha;
  };

  std::vector<std::vector<float>> out(static_cast<std::size_t>(levels));
  out[0] = band_alpha(fine);
  if (levels >= 2) out[1] = band_alpha(medium);
  // Bands 3..L are Raw-sourced: alpha ignored (plan 14.3) -> empty.
  return out;
}

}  // namespace tile_compile::reconstruction
