#pragma once

#include "tile_compile/reconstruction/tile_boundary_diagnostics.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tile_compile::reconstruction {

struct TileWeightProfilePairDiagnostic {
  size_t lhs = 0;
  size_t rhs = 0;
  size_t usable_frame_count = 0;
  size_t lhs_active_frame_count = 0;
  size_t rhs_active_frame_count = 0;
  size_t shared_active_frame_count = 0;
  size_t activation_mismatch_count = 0;
  float mean_abs_delta = 0.0f;
  float p95_abs_delta = 0.0f;
  float correlation = 0.0f;
  bool valid = false;
};

struct TileWeightProfileDiagnostics {
  size_t observed_pair_count = 0;
  float pair_mean_abs_delta_mean = 0.0f;
  float pair_mean_abs_delta_p95 = 0.0f;
  float pair_p95_abs_delta_mean = 0.0f;
  float pair_p95_abs_delta_p95 = 0.0f;
  float pair_activation_mismatch_fraction_mean = 0.0f;
  float pair_activation_mismatch_fraction_p95 = 0.0f;
  float pair_correlation_mean = 0.0f;
  float pair_correlation_p05 = 0.0f;
  std::vector<TileWeightProfilePairDiagnostic> pair_diagnostics;
};

TileWeightProfileDiagnostics analyze_tile_weight_profiles(
    const std::vector<TileBoundaryPairDiagnostic> &boundary_pairs,
    const std::vector<std::vector<float>> &local_weights,
    const std::vector<uint8_t> &frame_has_data);

} // namespace tile_compile::reconstruction
