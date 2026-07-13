#pragma once

#include "tile_compile/core/types.hpp"

#include <cstdint>
#include <vector>

namespace tile_compile::metrics {

struct AqmhRegion {
  int label = 0;
  int area = 0;
  float mean_quality = 0.0f;
  float compactness = 0.0f;
  float score = 0.0f;
};

std::vector<AqmhRegion> extract_aqmh_regions(
    const Matrix2Df &quality_map, const std::vector<uint8_t> &source_valid_mask,
    float quantile, int morphology_radius_canvas_px);

} // namespace tile_compile::metrics
