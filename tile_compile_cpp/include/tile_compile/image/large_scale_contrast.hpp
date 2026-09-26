#pragma once

// Scale-selective ("large-scale") contrast for the stretched RGB image.
//
// The single global stretch curve compresses faint, extended sky structure
// (dust lanes, reflection nebulae, gradients of the sky glow) into a very narrow
// band above the background level. This stage lifts ONLY that large-scale
// component and leaves everything at pixel scale untouched, so noise, star
// profiles and fine detail are not amplified (unlike a local-contrast boost,
// which acts on fine scales).
//
// Mechanism: the luminance is reduced to a star-free large-scale map
//   1. masked area downsampling to about sigma/6 pixels per cell,
//   2. a 5x5 median on that grid (removes stars and hot pixels),
//   3. a masked (normalised) Gaussian of `sigma_px`, so the border of the valid
//      area does not pull the map towards a fill value.
// The deviation of that map from the sky level (its median over the valid
// area) is added back, scaled by `amount`:
//   I' = I + amount * (LS(I) - sky)          (same delta on R, G and B)
// `remove_vignette` first subtracts a radially symmetric component (robust
// polynomial in r^2 about the valid-area centre) from the deviation, so an
// uncorrected vignette or radial sky glow is not amplified together with the
// nebula. `chroma_amount` does the same for the colour differences R-G and B-G
// (G stays unchanged), which brings out large-scale colour structure without
// touching brightness.
//
// Properties: a flat image is left unchanged; pixels outside the valid mask are
// never touched; values are kept within [0, per-channel maximum of the input];
// deterministic; no dependence on the frame count. Off by default.

#include "tile_compile/core/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace tile_compile::image {

struct LargeScaleContrastConfig {
  bool enabled = false;
  float amount = 1.0f;         // [0, 6]: factor on the large-scale luminance deviation (0 = none)
  float sigma_px = 48.0f;      // (0, 512]: Gaussian sigma of the "large" component, output pixels
  float chroma_amount = 0.0f;  // [0, 6]: same factor for R-G and B-G; 0 = colour unchanged
  bool remove_vignette = true; // exclude the radially symmetric component from the boost
};

struct LargeScaleContrastResult {
  bool applied = false;
  // "applied" | "not_needed" (amount and chroma_amount are 0) |
  // "too_few_valid_pixels" | "disabled" | "error"
  std::string status = "disabled";
  std::string error_message;
  double sky_reference = 0.0;       // median of the large-scale luminance map
  double span_before = 0.0;         // p5-p95 of the large-scale luminance map (after vignette removal)
  double span_after = 0.0;          // the same after the boost
  bool vignette_removed = false;
  int downsample_factor = 1;
};

// Adds the boost in place. `valid_mask` (rows*cols, non-zero = valid) may be
// null: then a pixel is valid when its channels are finite and not all zero.
LargeScaleContrastResult apply_large_scale_contrast(
    Matrix2Df &R, Matrix2Df &G, Matrix2Df &B, const LargeScaleContrastConfig &cfg,
    const std::vector<std::uint8_t> *valid_mask);

}  // namespace tile_compile::image
