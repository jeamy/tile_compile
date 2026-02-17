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

/**
 * Compute bounding box that contains all warped frame corners.
 * Returns (min_x, min_y, max_x, max_y) in output space.
 * Used for field rotation: output canvas must be large enough to contain all rotated frames.
 */
struct BoundingBox {
    int min_x;
    int min_y;
    int max_x;
    int max_y;
    int width() const { return max_x - min_x; }
    int height() const { return max_y - min_y; }
};

BoundingBox compute_warps_bounding_box(int frame_width, int frame_height,
                                       const std::vector<WarpMatrix>& warps);

} // namespace tile_compile::registration
