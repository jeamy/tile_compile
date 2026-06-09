#pragma once

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/types.hpp"

#include <limits>
#include <vector>

namespace tile_compile::metrics {

inline constexpr float eps_aqmh = 1.0e-6f;

struct AqmhQualityMapDiagnostics {
  float sharpness_p50 = std::numeric_limits<float>::quiet_NaN();
  float snr_p50 = std::numeric_limits<float>::quiet_NaN();
  bool scene_dependent_snr = false;
  std::vector<int> omitted_scales;
};

struct AqmhQualityMapResult {
  Matrix2Df q_map;
  AqmhQualityMapDiagnostics diagnostics;
};

AqmhQualityMapResult compute_aqmh_quality_map(
    const Matrix2Df &frame, const std::vector<uint8_t> &canvas_mask,
    int canvas_mask_width, int canvas_mask_height,
    const config::AqmhPyramidConfig &cfg);

} // namespace tile_compile::metrics
