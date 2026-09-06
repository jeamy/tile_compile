#include "tile_compile/reconstruction/alpha_confidence.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace tile_compile::reconstruction {

namespace {

double smoothstep(double e0, double e1, double x) {
  if (!(e1 > e0)) return x >= e1 ? 1.0 : 0.0;
  const double t = std::clamp((x - e0) / (e1 - e0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

}  // namespace

double weighted_percentile(std::span<const double> values,
                           std::span<const double> weights, double p) {
  if (values.size() != weights.size())
    throw std::invalid_argument("WEIGHTED_PERCENTILE_SIZE");
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  p = std::clamp(p, 0.0, 1.0);

  std::vector<std::size_t> idx(values.size());
  std::iota(idx.begin(), idx.end(), std::size_t{0});
  std::sort(idx.begin(), idx.end(),
            [&](std::size_t a, std::size_t b) { return values[a] < values[b]; });

  double total = 0.0;
  for (double w : weights) {
    if (!(w >= 0.0)) throw std::invalid_argument("WEIGHTED_PERCENTILE_WEIGHT");
    total += w;
  }
  if (!(total > 0.0)) return values[idx.front()];

  // Hazen plotting position: CDF at sample k is (cum_k - w_k/2) / total.
  double cum = 0.0;
  double prev_cdf = 0.0;
  double prev_val = values[idx.front()];
  for (std::size_t k = 0; k < idx.size(); ++k) {
    const double w = weights[idx[k]];
    cum += w;
    const double cdf = (cum - 0.5 * w) / total;
    const double val = values[idx[k]];
    if (p <= cdf) {
      if (k == 0 || cdf <= prev_cdf) return val;
      const double frac = (p - prev_cdf) / (cdf - prev_cdf);
      return prev_val + std::clamp(frac, 0.0, 1.0) * (val - prev_val);
    }
    prev_cdf = cdf;
    prev_val = val;
  }
  return values[idx.back()];
}

AlphaConfidenceFactors compute_alpha_confidence_channel(
    std::span<const AlphaFactorContribution> accepted,
    const AlphaConfidenceParams &params) {
  AlphaConfidenceFactors out;
  if (accepted.empty()) return out;

  std::vector<double> b, q, resid;
  b.reserve(accepted.size());
  q.reserve(accepted.size());
  resid.reserve(accepted.size());
  std::vector<double> art_v, art_w;
  double b_total = 0.0, b_direct = 0.0;
  for (const auto &c : accepted) {
    if (!(c.b > 0.0)) continue;
    b.push_back(c.b);
    q.push_back(std::clamp(c.q_composite, 0.0, 1.0));
    resid.push_back(c.residual_factor);
    b_total += c.b;
    if (c.is_direct) b_direct += c.b;
    if (std::isfinite(c.artifact_conf)) {
      art_v.push_back(std::clamp(c.artifact_conf, 0.0, 1.0));
      art_w.push_back(c.b);
    }
  }
  if (b.empty()) return out;

  // A_separation.
  const double q_p50 = weighted_percentile(q, b, 0.50);
  const double q_p90 = weighted_percentile(q, b, 0.90);
  out.a_separation = smoothstep(params.min_quality_separation,
                                params.full_quality_separation,
                                std::max(0.0, q_p90 - q_p50));

  // A_artifact. Plan 14.4: fewer than min_artifact_contributors valid a_f
  // => not applicable => 0 (not full confidence).
  if (static_cast<int>(art_v.size()) >= params.min_artifact_contributors) {
    const double a_p10 = weighted_percentile(art_v, art_w, 0.10);
    out.a_artifact = smoothstep(params.artifact_lo, params.artifact_hi, a_p10);
    out.artifact_applicable = true;
  } else {
    out.a_artifact = 0.0;
    out.artifact_applicable = false;
  }

  // A_registration.
  const double direct_fraction = b_total > 0.0 ? b_direct / b_total : 0.0;
  const double residual_p20 = weighted_percentile(resid, b, 0.20);
  out.a_registration =
      std::min(smoothstep(params.direct_fraction_lo, params.direct_fraction_hi,
                          direct_fraction),
               smoothstep(params.residual_p20_lo, params.residual_p20_hi,
                          residual_p20));
  return out;
}

}  // namespace tile_compile::reconstruction
