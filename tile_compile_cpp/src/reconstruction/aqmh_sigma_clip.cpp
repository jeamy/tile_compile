#include "tile_compile/reconstruction/aqmh_sigma_clip.hpp"

#include "tile_compile/metrics/aqmh_eps.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tile_compile::reconstruction {
namespace {

float weighted_median(std::vector<AqmhWeightedSample> samples, bool deviations,
                      float center = 0.0f) {
  std::sort(samples.begin(), samples.end(), [&](const auto &a, const auto &b) {
    const float av = deviations ? std::abs(a.value - center) : a.value;
    const float bv = deviations ? std::abs(b.value - center) : b.value;
    return av != bv ? av < bv : a.frame_index < b.frame_index;
  });
  double total = 0.0;
  for (const auto &s : samples) total += s.weight;
  double cumulative = 0.0;
  for (const auto &s : samples) {
    cumulative += s.weight;
    if (cumulative >= 0.5 * total)
      return deviations ? std::abs(s.value - center) : s.value;
  }
  return samples.empty() ? 0.0f : samples.back().value;
}

} // namespace

AqmhSigmaClipResult aqmh_sigma_clip(
    std::vector<AqmhWeightedSample> samples, float clip_sigma,
    int iterations, float min_fraction, float min_effective_n) {
  samples.erase(std::remove_if(samples.begin(), samples.end(), [](const auto &s) {
                  return !std::isfinite(s.value) || !std::isfinite(s.weight) ||
                         !(s.weight > 0.0f);
                }), samples.end());
  AqmhSigmaClipResult result;
  if (samples.empty()) return result;
  const size_t n0 = samples.size();
  const size_t keep_floor = std::min(
      n0, std::max<size_t>(1, static_cast<size_t>(std::ceil(min_fraction * n0))));
  for (int iter = 0; iter < iterations; ++iter) {
    const float center = weighted_median(samples, false);
    const float mad = weighted_median(samples, true, center);
    std::vector<float> values;
    values.reserve(samples.size());
    for (const auto &s : samples) values.push_back(s.value);
    const float floor = metrics::eps_noise(values);
    std::vector<AqmhWeightedSample> next;
    if (mad <= floor) {
      for (const auto &s : samples) if (s.value == center) next.push_back(s);
    } else {
      const float sigma = 1.4826f * mad;
      for (const auto &s : samples)
        if (std::abs(s.value - center) <= clip_sigma * sigma) next.push_back(s);
    }
    if (next.size() < keep_floor) {
      std::sort(samples.begin(), samples.end(), [&](const auto &a, const auto &b) {
        const float ar = std::abs(a.value - center) / std::max(1.4826f * mad, floor);
        const float br = std::abs(b.value - center) / std::max(1.4826f * mad, floor);
        return ar != br ? ar < br : a.frame_index < b.frame_index;
      });
      next.assign(samples.begin(), samples.begin() + static_cast<long>(keep_floor));
    }
    std::sort(next.begin(), next.end(), [](const auto &a, const auto &b) {
      return a.frame_index < b.frame_index;
    });
    std::sort(samples.begin(), samples.end(), [](const auto &a, const auto &b) {
      return a.frame_index < b.frame_index;
    });
    if (next.size() == samples.size() &&
        std::equal(next.begin(), next.end(), samples.begin(),
                   [](const auto &a, const auto &b) { return a.frame_index == b.frame_index; })) {
      samples = std::move(next);
      break;
    }
    samples = std::move(next);
  }
  double d = 0.0, d2 = 0.0;
  float wmax = 0.0f;
  for (const auto &s : samples) {
    d += s.weight;
    d2 += static_cast<double>(s.weight) * s.weight;
    wmax = std::max(wmax, s.weight);
  }
  result.weight_sum = static_cast<float>(d);
  result.effective_n = d2 > 0.0 ? static_cast<float>(d * d / d2) : 0.0f;
  const float guard = static_cast<float>(samples.size()) *
                      std::numeric_limits<float>::epsilon() * wmax;
  result.denominator_ok = d > guard && result.effective_n >= min_effective_n;
  result.retained = std::move(samples);
  return result;
}

} // namespace tile_compile::reconstruction
