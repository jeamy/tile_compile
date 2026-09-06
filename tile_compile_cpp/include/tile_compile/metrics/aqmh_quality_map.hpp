#pragma once

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/acceleration.hpp"
#include "tile_compile/core/types.hpp"

#include <functional>
#include <limits>
#include <vector>

namespace tile_compile::metrics {

// Optional per-scale observation hook (CFA-forward-drizzle plan M5, section
// 13.3). Invoked once per computed pyramid scale, immediately after `psi` is
// formed and BEFORE it is folded into the geometric-mean composite, with:
//   scale_index      - pyramid level s (0-based)
//   downsample_factor - legacy convention 1 << (2*s), relative to the frame
//                       passed to compute_aqmh_quality_map()
//   psi              - the scale's quality map at that scale's (downsampled)
//                      grid, values in [0,1], NaN where unsupported
//   artifact         - phi_artifact for that scale on the same grid
//                      (1 = clean, 0 = fully artefacted, NaN where unsupported)
// Default-constructed (null) hook => byte-for-byte identical behaviour to the
// pre-hook implementation; the legacy prewarped Q-map path passes no hook.
using PerScaleQualityHook = std::function<void(
    int scale_index, int downsample_factor, const Matrix2Df &psi,
    const Matrix2Df &artifact)>;

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
  double timing_source_mask_seconds = 0.0;
  double timing_pyramid_prepare_seconds = 0.0;
  double timing_sharpness_seconds = 0.0;
  double timing_local_background_seconds = 0.0;
  double timing_snr_seconds = 0.0;
  double timing_artifact_seconds = 0.0;
  double timing_summary_seconds = 0.0;
  double timing_psi_accumulate_seconds = 0.0;
  double timing_finalize_seconds = 0.0;
  double timing_total_seconds = 0.0;
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
    cv::cuda::Stream *stream = nullptr,
    const PerScaleQualityHook &per_scale_hook = {});

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
