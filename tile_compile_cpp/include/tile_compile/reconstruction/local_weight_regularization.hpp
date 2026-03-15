#pragma once

#include "tile_compile/core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tile_compile::reconstruction {

struct LocalWeightRegularizationConfig {
  bool enabled = true;
  float lambda = 0.35f;
  int passes = 1;
};

struct LocalWeightRegularizationSummary {
  size_t tile_edge_count = 0;
  size_t adjusted_entries = 0;
  float mean_abs_q_delta = 0.0f;
  float p95_abs_q_delta = 0.0f;
};

LocalWeightRegularizationSummary regularize_local_quality_scores(
    const std::vector<Tile> &tiles, const std::vector<uint8_t> &tile_valid,
    const std::vector<uint8_t> &frame_has_data,
    const LocalWeightRegularizationConfig &cfg,
    std::vector<std::vector<float>> *quality_scores);

} // namespace tile_compile::reconstruction
