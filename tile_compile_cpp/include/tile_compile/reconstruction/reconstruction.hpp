#pragma once

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/types.hpp"

#include <vector>

namespace tile_compile::reconstruction {

Matrix2Df reconstruct_tiles(const std::vector<Matrix2Df>& frames,
                            const TileGrid& grid,
                            const std::vector<std::vector<float>>& tile_weights);

// Wiener denoising filter for a single tile
Matrix2Df wiener_tile_filter(const Matrix2Df& tile, float sigma, float snr_tile,
                             float q_struct_tile, bool is_star_tile,
                             const config::WienerDenoiseConfig& cfg);

// Sigma-clipped mean stack of multiple frames
Matrix2Df sigma_clip_stack(const std::vector<Matrix2Df>& frames,
                           float sigma_low, float sigma_high,
                           int max_iters, float min_fraction);

// Per-pixel weighted sigma-clipped mean of tile stack.
// Rejects outlier pixels (e.g. star trails) before computing weighted mean.
Matrix2Df sigma_clip_weighted_tile(const std::vector<Matrix2Df>& tiles,
                                   const std::vector<float>& weights,
                                   float sigma_low, float sigma_high,
                                   int max_iters, float min_fraction);

// Highpass + Soft-Threshold denoising for a single tile (Methodik 3.1E §3.3.1)
Matrix2Df soft_threshold_tile_filter(const Matrix2Df& tile,
                                      const config::SoftThresholdConfig& cfg);

// Generate a 1D Hann (raised cosine) window of length n.
std::vector<float> make_hann_1d(int n);

} // namespace tile_compile::reconstruction
