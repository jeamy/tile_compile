#include "tile_compile/reconstruction/tile_weight_profile_diagnostics.hpp"

#include "tile_compile/core/utils.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace tile_compile::reconstruction {

namespace {

float compute_correlation(const std::vector<float> &lhs,
                          const std::vector<float> &rhs) {
  if (lhs.size() != rhs.size() || lhs.empty()) {
    return 0.0f;
  }
  if (lhs.size() == 1u) {
    return (std::fabs(lhs[0] - rhs[0]) <= 1.0e-6f) ? 1.0f : 0.0f;
  }

  const double lhs_mean =
      std::accumulate(lhs.begin(), lhs.end(), 0.0) / static_cast<double>(lhs.size());
  const double rhs_mean =
      std::accumulate(rhs.begin(), rhs.end(), 0.0) / static_cast<double>(rhs.size());

  double cov = 0.0;
  double lhs_var = 0.0;
  double rhs_var = 0.0;
  double max_abs_delta = 0.0;
  for (size_t i = 0; i < lhs.size(); ++i) {
    const double dl = static_cast<double>(lhs[i]) - lhs_mean;
    const double dr = static_cast<double>(rhs[i]) - rhs_mean;
    cov += dl * dr;
    lhs_var += dl * dl;
    rhs_var += dr * dr;
    max_abs_delta =
        std::max(max_abs_delta, std::fabs(static_cast<double>(lhs[i] - rhs[i])));
  }

  constexpr double kVarEps = 1.0e-12;
  if (lhs_var <= kVarEps && rhs_var <= kVarEps) {
    return (max_abs_delta <= 1.0e-6) ? 1.0f : 0.0f;
  }
  if (lhs_var <= kVarEps || rhs_var <= kVarEps) {
    return 0.0f;
  }
  return static_cast<float>(cov / std::sqrt(lhs_var * rhs_var));
}

} // namespace

TileWeightProfileDiagnostics analyze_tile_weight_profiles(
    const std::vector<TileBoundaryPairDiagnostic> &boundary_pairs,
    const std::vector<std::vector<float>> &local_weights,
    const std::vector<uint8_t> &frame_has_data) {
  TileWeightProfileDiagnostics out;
  if (boundary_pairs.empty() || local_weights.empty() || frame_has_data.empty()) {
    return out;
  }

  const size_t frame_count = std::min(local_weights.size(), frame_has_data.size());
  std::vector<float> pair_mean_abs_delta;
  std::vector<float> pair_p95_abs_delta;
  std::vector<float> pair_activation_mismatch_fraction;
  std::vector<float> pair_correlation;
  pair_mean_abs_delta.reserve(boundary_pairs.size());
  pair_p95_abs_delta.reserve(boundary_pairs.size());
  pair_activation_mismatch_fraction.reserve(boundary_pairs.size());
  pair_correlation.reserve(boundary_pairs.size());
  out.pair_diagnostics.reserve(boundary_pairs.size());

  constexpr float kActiveEps = 1.0e-6f;

  for (const auto &boundary_pair : boundary_pairs) {
    TileWeightProfilePairDiagnostic pair;
    pair.lhs = boundary_pair.lhs;
    pair.rhs = boundary_pair.rhs;

    std::vector<float> deltas;
    std::vector<float> lhs_profile;
    std::vector<float> rhs_profile;
    deltas.reserve(frame_count);
    lhs_profile.reserve(frame_count);
    rhs_profile.reserve(frame_count);

    double abs_delta_sum = 0.0;
    for (size_t fi = 0; fi < frame_count; ++fi) {
      if (frame_has_data[fi] == 0u) {
        continue;
      }
      if (boundary_pair.lhs >= local_weights[fi].size() ||
          boundary_pair.rhs >= local_weights[fi].size()) {
        continue;
      }
      const float lhs_w = local_weights[fi][boundary_pair.lhs];
      const float rhs_w = local_weights[fi][boundary_pair.rhs];
      if (!std::isfinite(lhs_w) || !std::isfinite(rhs_w)) {
        continue;
      }

      ++pair.usable_frame_count;
      lhs_profile.push_back(lhs_w);
      rhs_profile.push_back(rhs_w);

      const bool lhs_active = lhs_w > kActiveEps;
      const bool rhs_active = rhs_w > kActiveEps;
      if (lhs_active) {
        ++pair.lhs_active_frame_count;
      }
      if (rhs_active) {
        ++pair.rhs_active_frame_count;
      }
      if (lhs_active && rhs_active) {
        ++pair.shared_active_frame_count;
      } else if (lhs_active != rhs_active) {
        ++pair.activation_mismatch_count;
      }

      const float abs_delta = std::fabs(lhs_w - rhs_w);
      abs_delta_sum += abs_delta;
      deltas.push_back(abs_delta);
    }

    if (pair.usable_frame_count == 0u) {
      continue;
    }

    pair.mean_abs_delta = static_cast<float>(
        abs_delta_sum / static_cast<double>(pair.usable_frame_count));
    std::sort(deltas.begin(), deltas.end());
    pair.p95_abs_delta = core::percentile_from_sorted(deltas, 95.0f);
    pair.correlation = compute_correlation(lhs_profile, rhs_profile);
    pair.valid = true;

    ++out.observed_pair_count;
    pair_mean_abs_delta.push_back(pair.mean_abs_delta);
    pair_p95_abs_delta.push_back(pair.p95_abs_delta);
    pair_activation_mismatch_fraction.push_back(
        static_cast<float>(pair.activation_mismatch_count) /
        static_cast<float>(pair.usable_frame_count));
    pair_correlation.push_back(pair.correlation);
    out.pair_diagnostics.push_back(pair);
  }

  if (out.pair_diagnostics.empty()) {
    return out;
  }

  out.pair_mean_abs_delta_mean =
      std::accumulate(pair_mean_abs_delta.begin(), pair_mean_abs_delta.end(), 0.0f) /
      static_cast<float>(pair_mean_abs_delta.size());
  out.pair_p95_abs_delta_mean =
      std::accumulate(pair_p95_abs_delta.begin(), pair_p95_abs_delta.end(), 0.0f) /
      static_cast<float>(pair_p95_abs_delta.size());
  out.pair_activation_mismatch_fraction_mean =
      std::accumulate(pair_activation_mismatch_fraction.begin(),
                      pair_activation_mismatch_fraction.end(), 0.0f) /
      static_cast<float>(pair_activation_mismatch_fraction.size());
  out.pair_correlation_mean =
      std::accumulate(pair_correlation.begin(), pair_correlation.end(), 0.0f) /
      static_cast<float>(pair_correlation.size());

  std::sort(pair_mean_abs_delta.begin(), pair_mean_abs_delta.end());
  std::sort(pair_p95_abs_delta.begin(), pair_p95_abs_delta.end());
  std::sort(pair_activation_mismatch_fraction.begin(),
            pair_activation_mismatch_fraction.end());
  std::sort(pair_correlation.begin(), pair_correlation.end());

  out.pair_mean_abs_delta_p95 =
      core::percentile_from_sorted(pair_mean_abs_delta, 95.0f);
  out.pair_p95_abs_delta_p95 =
      core::percentile_from_sorted(pair_p95_abs_delta, 95.0f);
  out.pair_activation_mismatch_fraction_p95 =
      core::percentile_from_sorted(pair_activation_mismatch_fraction, 95.0f);
  out.pair_correlation_p05 =
      core::percentile_from_sorted(pair_correlation, 5.0f);

  return out;
}

} // namespace tile_compile::reconstruction
