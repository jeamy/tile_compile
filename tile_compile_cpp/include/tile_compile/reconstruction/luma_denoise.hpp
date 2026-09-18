#pragma once

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/types.hpp"

#include <cstdint>

namespace tile_compile::reconstruction {

struct LumaDenoiseStats {
  bool applied = false;
  std::uint64_t valid_pixels = 0;
  double input_luma_sigma = 0.0;
  double star_protected_fraction = 0.0;
  double structure_protected_fraction = 0.0;
  double combined_protected_fraction = 0.0;
  double mean_protection = 0.0;
  double mean_denoise_fraction = 0.0;
};

// Denoises the RGB planes' shared luminance in place, reconstructing R/G/B
// by adding the same smooth per-pixel brightness delta to each channel
// (color is unaffected -- every channel difference is preserved exactly;
// only brightness noise is smoothed). Called on the post-stack linear RGB,
// before BGE/PCC/HMS -- see LumaDenoiseConfig's comment for why this stage
// exists, and why the reconstruction is additive rather than ratio-based.
// `protection_mask_out`, when non-null, is resized to r's shape and filled
// with the combined protection mask used for this call -- see
// chroma_denoise_rgb_inplace's equivalent parameter.
LumaDenoiseStats luma_denoise_rgb_inplace(
    Matrix2Df& r, Matrix2Df& g, Matrix2Df& b,
    const config::LumaDenoiseConfig& cfg,
    Matrix2Df* protection_mask_out = nullptr);

} // namespace tile_compile::reconstruction
