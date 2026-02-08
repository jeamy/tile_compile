#pragma once

#include "tile_compile/core/types.hpp"

#include <string>

namespace tile_compile::image {

struct NormalizationScales {
  bool is_osc = false;
  float scale_mono = 1.0f;
  float scale_r = 1.0f;
  float scale_g = 1.0f;
  float scale_b = 1.0f;
};

void apply_normalization_inplace(Matrix2Df &img, const NormalizationScales &s,
                                 ColorMode mode,
                                 const std::string &bayer_pattern,
                                 int origin_x, int origin_y);

// Apply output scaling (denormalization) in-place: multiply by per-channel
// background level and add pedestal. CFA-aware for OSC data.
void apply_output_scaling_inplace(Matrix2Df &img, int origin_x, int origin_y,
                                  ColorMode mode,
                                  const std::string &bayer_pattern,
                                  float bg_mono, float bg_r, float bg_g,
                                  float bg_b, float pedestal);

// Apply a global warp to a full-resolution frame (CFA-aware for OSC).
Matrix2Df apply_global_warp(const Matrix2Df &img, const WarpMatrix &warp,
                            ColorMode mode);

} // namespace tile_compile::image
