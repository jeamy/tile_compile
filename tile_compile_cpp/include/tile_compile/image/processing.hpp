#pragma once

#include "tile_compile/core/types.hpp"
#include <map>
#include <string>

namespace tile_compile::image {

std::map<std::string, Matrix2Df> split_cfa_channels(const Matrix2Df& mosaic, BayerPattern pattern);

Matrix2Df reassemble_cfa_mosaic(const Matrix2Df& R, const Matrix2Df& G, const Matrix2Df& B,
                                BayerPattern pattern);

Matrix2Df normalize_frame(const Matrix2Df& frame, float target_background,
                          float target_scale, NormalizationMode mode);

Matrix2Df cosmetic_correction(const Matrix2Df& frame, float sigma_threshold, bool correct_hot);

// Extract a tile sub-region from an image, clamped to image bounds.
Matrix2Df extract_tile(const Matrix2Df& img, const Tile& t);

} // namespace tile_compile::image
