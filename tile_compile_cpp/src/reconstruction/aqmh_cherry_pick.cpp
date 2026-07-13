#include "tile_compile/reconstruction/aqmh_cherry_pick.hpp"

#include <algorithm>
#include <cmath>

namespace tile_compile::reconstruction {

float aqmh_effective_k_frac(
    int n_rankable, float base,
    const std::vector<config::AqmhCherryPickConfig::Tier> &tiers) {
  float fraction = base;
  for (const auto &tier : tiers)
    if (tier.min_n_rankable <= n_rankable)
      fraction = std::max(fraction, tier.k_frac);
  return fraction;
}

int aqmh_k_nominal(int n_rankable, float fraction) {
  return std::max(0, static_cast<int>(std::floor(fraction * n_rankable)));
}

std::vector<AqmhWeightedSample> aqmh_select_top_k(
    std::vector<AqmhWeightedSample> samples, int k_min_required,
    float fraction,
    const std::vector<config::AqmhCherryPickConfig::Tier> &tiers,
    int *nominal_k, float *rank_margin) {
  samples.erase(std::remove_if(samples.begin(), samples.end(), [](const auto &s) {
                  return !(s.score > 0.0f) || !std::isfinite(s.score);
                }), samples.end());
  const int n = static_cast<int>(samples.size());
  const float effective = aqmh_effective_k_frac(n, fraction, tiers);
  const int nominal = aqmh_k_nominal(n, effective);
  if (nominal_k) *nominal_k = nominal;
  if (rank_margin) *rank_margin = -1.0f;
  if (n < k_min_required) return {};
  const int k = std::min(n, std::max(k_min_required, nominal));
  auto score_cmp = [](const auto &a, const auto &b) {
    return a.score != b.score ? a.score > b.score : a.frame_index < b.frame_index;
  };
  // Partition around the k-th element: O(N) average instead of O(N log N) sort.
  std::nth_element(samples.begin(), samples.begin() + k - 1,
                   samples.end(), score_cmp);
  // Sort just the top-k for deterministic ordering: O(k log k).
  std::sort(samples.begin(), samples.begin() + k, score_cmp);
  if (rank_margin && k < n && samples.front().score > 0.0f) {
    // The element at position k is the best of the remaining partition.
    const float kth_score = samples[static_cast<size_t>(k - 1)].score;
    const float next_score = std::max_element(
        samples.begin() + k, samples.end(),
        [](const auto &a, const auto &b) { return a.score < b.score; })->score;
    *rank_margin = (kth_score - next_score) / samples.front().score;
  }
  samples.resize(static_cast<size_t>(k));
  return samples;
}

} // namespace tile_compile::reconstruction
