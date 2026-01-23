#pragma once

#include "tile_compile/core/types.hpp"

#include <utility>

namespace tile_compile::registration {

Matrix2Df prepare_ecc_image(const Matrix2Df& img);

std::pair<float, float> phasecorr_translation(const Matrix2Df& moving, const Matrix2Df& ref);

RegistrationResult ecc_warp(const Matrix2Df& moving, const Matrix2Df& ref,
                            bool allow_rotation, const WarpMatrix& init_warp,
                            int max_iterations, float epsilon);

WarpMatrix identity_warp();

Matrix2Df apply_warp(const Matrix2Df& img, const WarpMatrix& warp);

} // namespace tile_compile::registration
