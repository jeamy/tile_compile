#pragma once

#include "tile_compile/reconstruction/aqmh_reconstruction.hpp"

// Only declare CUDA functions if CUDA is enabled
#if TILE_COMPILE_WITH_CUDA

namespace tile_compile::reconstruction {

/// CUDA-accelerated AQMH weighted reconstruction.
/// Implements row-chunked weighted median/MAD/sigma-clip on GPU.
/// Note: This uses row-chunking (not frame-batching) to stay within GPU memory limits
/// while processing ALL frames per pixel, preserving statistical correctness.
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

} // namespace tile_compile::reconstruction

#else
// CUDA not available - no CUDA functions declared
#endif
