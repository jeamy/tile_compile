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
};

// Applies the configured chroma denoise to the RGB planes in place. Shared by
// the downstream post-stack output path (apply_stage == "post_pcc").
ChromaDenoiseStats chroma_denoise_rgb_inplace(
    Matrix2Df& r, Matrix2Df& g, Matrix2Df& b,
    const config::ChromaDenoiseConfig& cfg);

} // namespace tile_compile::reconstruction
