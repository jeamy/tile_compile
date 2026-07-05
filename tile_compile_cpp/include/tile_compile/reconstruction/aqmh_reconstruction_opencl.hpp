#pragma once

#include "tile_compile/reconstruction/aqmh_reconstruction.hpp"

namespace tile_compile::reconstruction {

/// OpenCL-accelerated AQMH weighted reconstruction.
/// Implements row-chunked weighted median/MAD/sigma-clip on GPU using OpenCL.
/// Note: This uses row-chunking (not frame-batching) to stay within GPU memory limits
/// while processing ALL frames per pixel, preserving statistical correctness.
AqmhReconstructionResult reconstruct_aqmh_weighted_opencl(
    size_t frame_count,
    const AqmhFrameLoader &load_frame,
    metrics::QualityMapCache *q_map_cache,
    const VectorXf &global_weights,
    const std::vector<uint8_t> &canvas_mask,
    int width, int height,
    const AqmhReconstructionConfig &cfg,
    const AqmhMaskLoader &load_frame_valid_mask = {},
    const AqmhProgressCallback &progress = {});

} // namespace tile_compile::reconstruction
