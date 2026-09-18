#pragma once

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/types.hpp"

#include <cstdint>

namespace tile_compile::reconstruction {

struct ChromaDenoiseStats {
  bool applied = false;
  std::uint64_t valid_pixels = 0;
  double input_chroma_sigma = 0.0;
  double adaptation = 1.0;
  double effective_blend_amount = 0.0;
  double star_protected_fraction = 0.0;
  double structure_protected_fraction = 0.0;
  double extended_source_raw_fraction = 0.0;
  double extended_source_protected_fraction = 0.0;
  double combined_protected_fraction = 0.0;
  double mean_protection = 0.0;
  double mean_denoise_fraction = 0.0;
  double extended_source_sky_median = 0.0;
  double extended_source_sky_sigma = 0.0;
  double extended_source_threshold = 0.0;
  // Large-scale (block-median-surface) chroma bias removal, see
  // config::ChromaDenoiseConfig::LargeScaleBiasConfig.
  double input_luma_sigma = 0.0;
  double large_scale_bias_removed_rms_c1 = 0.0;
  double large_scale_bias_removed_rms_c2 = 0.0;
  double large_scale_bias_grid_holes_filled_fraction = 0.0;
};

// Applies the configured chroma denoise to the RGB planes in place. Shared by
// the downstream post-stack output path (apply_stage == "post_pcc").
// `protection_mask_out`, when non-null, is resized to r's shape and filled
// with the combined protection mask (star_protection/structure_protection/
// extended_source_protection, feathered and combined, in [0,1]) used for
// this call -- the same mask that scales denoise strength via
// luma_guard_strength. Diagnostic only; callers may write it out (e.g. as a
// FITS file) to inspect where and how strongly denoise was held back.
ChromaDenoiseStats chroma_denoise_rgb_inplace(
    Matrix2Df& r, Matrix2Df& g, Matrix2Df& b,
    const config::ChromaDenoiseConfig& cfg,
    Matrix2Df* protection_mask_out = nullptr);

} // namespace tile_compile::reconstruction
