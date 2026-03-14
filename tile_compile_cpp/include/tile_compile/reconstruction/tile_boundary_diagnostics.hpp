#pragma once

#include "tile_compile/core/types.hpp"

#include <cstddef>
#include <vector>

namespace tile_compile::reconstruction {

struct TileBoundaryPairDiagnostic {
  size_t lhs = 0;
  size_t rhs = 0;
  size_t sample_count = 0;
  float mean_abs_diff = 0.0f;
  float p95_abs_diff = 0.0f;
  float mean_signed_diff = 0.0f;
  bool valid = false;
};

struct TileBoundaryDiagnostics {
  size_t pair_count = 0;
  size_t observed_pair_count = 0;
  size_t sample_count = 0;
  float pair_mean_abs_diff_mean = 0.0f;
  float pair_mean_abs_diff_p95 = 0.0f;
  float pair_p95_abs_diff_mean = 0.0f;
  float pair_p95_abs_diff_p95 = 0.0f;
  float pair_mean_signed_diff_mean_abs = 0.0f;
  std::vector<TileBoundaryPairDiagnostic> pair_diagnostics;
};

TileBoundaryDiagnostics analyze_tile_boundaries(
    const std::vector<Tile> &tiles, const std::vector<Matrix2Df> &images,
    const std::vector<uint8_t> &tile_valid,
    const std::vector<uint8_t> &common_valid_mask = {},
    int common_mask_width = 0, int common_mask_height = 0);

} // namespace tile_compile::reconstruction
