#pragma once

#include "tile_compile/astrometry/photometric_color_cal.hpp"

namespace tile_compile::astrometry::detail {

// Applies a fitted diagonal PCC matrix around each channel background, then
// handles background neutralization independently according to the PCC mode.
void apply_diagonal_color_correction(
    Matrix2Df &R, Matrix2Df &G, Matrix2Df &B,
    const ColorMatrix &matrix, double chroma_strength,
    const std::string &background_neutralization_mode,
    const std::vector<uint8_t> &analysis_mask,
    const std::vector<uint8_t> *output_mask = nullptr,
    bool verbose = false);

} // namespace tile_compile::astrometry::detail
