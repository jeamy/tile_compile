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

} // namespace tile_compile::image
