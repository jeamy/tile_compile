#pragma once

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/acceleration.hpp"
#include "tile_compile/core/types.hpp"

#include <limits>
#include <vector>

namespace tile_compile::metrics {

struct AqmhQualityMapDiagnostics {
  float sharpness_p50 = std::numeric_limits<float>::quiet_NaN();
  float snr_p50 = std::numeric_limits<float>::quiet_NaN();
  bool scene_dependent_snr = false;
  bool acceleration_used = false;
  bool acceleration_fallback = false;
  float g_sharp_summary = std::numeric_limits<float>::quiet_NaN();
  float g_snr_summary = std::numeric_limits<float>::quiet_NaN();
  bool g_summary_invalid = false;
  std::vector<int> omitted_scales;
};

struct AqmhQualityMapResult {
  Matrix2Df q_map;
  AqmhQualityMapDiagnostics diagnostics;
};

// NaN-aware population variance over a clipped square neighborhood. Exposed so
// the linear sliding-window implementation can be regression-tested directly.
Matrix2Df compute_aqmh_local_variance(const Matrix2Df &image, int radius);

AqmhQualityMapResult compute_aqmh_quality_map(
    const Matrix2Df &frame, const std::vector<uint8_t> &canvas_mask,
    const std::vector<uint8_t> &frame_valid_mask,
    int canvas_mask_width, int canvas_mask_height,
    const config::AqmhPyramidConfig &cfg,
    core::AccelerationBackend backend = core::AccelerationBackend::cpu,
    cv::cuda::Stream *stream = nullptr);

inline AqmhQualityMapResult compute_aqmh_quality_map(
    const Matrix2Df &frame, const std::vector<uint8_t> &canvas_mask,
    int canvas_mask_width, int canvas_mask_height,
    const config::AqmhPyramidConfig &cfg,
    core::AccelerationBackend backend = core::AccelerationBackend::cpu,
    cv::cuda::Stream *stream = nullptr) {
  return compute_aqmh_quality_map(frame, canvas_mask, {}, canvas_mask_width,
                                  canvas_mask_height, cfg, backend, stream);
}

} // namespace tile_compile::metrics
