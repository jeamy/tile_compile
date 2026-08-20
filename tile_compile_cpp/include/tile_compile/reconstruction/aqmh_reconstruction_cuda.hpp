#pragma once

#include "tile_compile/reconstruction/aqmh_reconstruction.hpp"

#include <memory>
#include <vector>

// Only declare CUDA functions if CUDA is enabled
#if TILE_COMPILE_WITH_CUDA

namespace tile_compile::reconstruction {

/// CUDA-accelerated AQMH weighted reconstruction (single plane).
/// Implements row-chunked weighted median/MAD/sigma-clip on GPU.
AqmhReconstructionResult reconstruct_aqmh_weighted_cuda(
    size_t frame_count,
    const AqmhFrameLoader &load_frame,
    metrics::QualityMapCache *q_map_cache,
    const VectorXf &global_weights,
    const std::vector<uint8_t> &canvas_mask,
    int width, int height,
    const AqmhReconstructionConfig &cfg,
    const AqmhMaskLoader &load_frame_valid_mask = {},
    const AqmhFrameRegionLoader &load_frame_region = {},
    const AqmhMaskRegionLoader &load_frame_valid_mask_region = {},
    const AqmhProgressCallback &progress = {});

/// Session object for multi-plane RGB reconstruction (WP-A/R1).
/// Shares Q-Map/Mask GPU resources across planes: Q-Maps are uploaded to the
/// GPU once per chunk, then all planes (Luma, R, G, B) share them.
/// Use run_planes_rgb() for the 4-plane debayer-first-RGB case.
class AqmhCudaReconstructionSession {
 public:
  AqmhCudaReconstructionSession();
  ~AqmhCudaReconstructionSession();

  /// Initialise GPU resources and upload shared per-session data.
  /// Must be called before run_plane() / run_planes_rgb().
  bool init(size_t frame_count,
            metrics::QualityMapCache *q_map_cache,
            const VectorXf &global_weights,
            const std::vector<uint8_t> &canvas_mask,
            int width, int height,
            const AqmhReconstructionConfig &cfg,
            const AqmhMaskLoader &load_mask = {},
            const AqmhMaskRegionLoader &load_mask_region = {});

  /// Run all planes in one pass (chunk-outside / plane-inside, R1).
  /// frame_region_loaders[i] provides frames for plane i.
  /// compute_uniform_control[i] controls the uniform-control output for plane i.
  std::vector<AqmhReconstructionResult> run_planes_rgb(
      const std::vector<AqmhFrameRegionLoader> &frame_region_loaders,
      const std::vector<bool> &compute_uniform_control,
      const AqmhProgressCallback &progress = {});

  /// Run a single plane (backward-compatible wrapper; uses run_planes_rgb).
  AqmhReconstructionResult run_plane(
      const AqmhFrameLoader &load_frame,
      const AqmhFrameRegionLoader &load_frame_region,
      bool compute_uniform_control,
      const AqmhProgressCallback &progress = {});

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace tile_compile::reconstruction

#else
// CUDA not available - no CUDA functions declared
#endif
