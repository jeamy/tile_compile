#pragma once

// Adaptive "average neutral" colour-cast correction for the stretched RGB
// image (Siril's SCNR average-neutral idea, made automatic).
//
// Per pixel the green channel is pulled back towards the mean of red and blue,
//   G' = G - amount * max(0, (G - b_g) - max(0, ((R - b_r) + (B - b_b)) / 2)),
// with b_c the per-channel sky level, so the sky itself is left at its level and
// only excess above the sky is reduced. The amount is not a user constant: it is
// chosen so that the median green excess over the object pixels (blurred
// luminance clearly above the sky, brightest 1 % of the object excluded, which
// removes star cores) reaches `target_ratio`, capped at `max_amount`. When the
// measured excess is already <= `min_excess` nothing is changed (protects
// objects that are genuinely green/teal, e.g. OIII emission).

#include "tile_compile/core/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace tile_compile::image {

struct ColorCastCorrectionConfig {
  bool enabled = false;
  float max_amount = 1.0f;    // (0, 1]: cap on the correction strength
  float target_ratio = 1.0f;  // median G/((R+B)/2) on object pixels to reach
  float min_excess = 1.02f;   // >= 1: measured ratio at/below this -> no-op
  float object_sigma = 3.0f;  // object = blurred luminance > sky + k * sigma
};

struct ColorCastCorrectionResult {
  bool applied = false;
  // "applied" | "not_needed" | "too_few_object_pixels" | "disabled" | "error"
  std::string status = "disabled";
  std::string error_message;
  float amount = 0.0f;         // chosen strength in [0, max_amount]
  double ratio_before = 0.0;   // median G/((R+B)/2) on object pixels
  double ratio_after = 0.0;
  std::size_t object_pixels = 0;
  double sky_r = 0.0, sky_g = 0.0, sky_b = 0.0;  // per-channel sky level
};

// Corrects G in place. `valid_mask` (rows*cols, non-zero = valid) may be null:
// then a pixel is valid when its three channels are finite and not all zero.
ColorCastCorrectionResult apply_color_cast_correction(
    Matrix2Df &R, Matrix2Df &G, Matrix2Df &B,
    const ColorCastCorrectionConfig &cfg,
    const std::vector<std::uint8_t> *valid_mask);

}  // namespace tile_compile::image
