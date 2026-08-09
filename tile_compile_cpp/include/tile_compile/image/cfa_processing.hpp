#pragma once

#include "tile_compile/core/types.hpp"

#include <string>

namespace tile_compile::image {

/**
 * Create green mask for CFA pattern.
 */
Matrix2Df cfa_green_mask(int height, int width, const std::string& bayer_pattern);

/**
 * Compute green proxy from CFA mosaic (interpolate non-green pixels).
 */
Matrix2Df cfa_green_proxy(const Matrix2Df& mosaic, const std::string& bayer_pattern);

/**
 * Downsample CFA mosaic using green proxy (half resolution).
 * Used for registration of OSC data.
 */
Matrix2Df cfa_green_proxy_downsample2x2(const Matrix2Df& mosaic, const std::string& bayer_pattern);

/**
 * Warp CFA mosaic by warping each Bayer subplane separately.
 * This preserves Bayer pattern integrity during warping.
 */
Matrix2Df warp_cfa_mosaic_via_subplanes(
    const Matrix2Df& mosaic,
    const WarpMatrix& warp,
    int out_height = -1,
    int out_width = -1,
    const std::string& border_mode = "constant",
    const std::string& interpolation = "linear"
);

/**
 * Split CFA mosaic into R, G, B subplanes (half resolution).
 */
struct CFAChannels {
    Matrix2Df R;
    Matrix2Df G;
    Matrix2Df B;
};

CFAChannels split_cfa_channels(const Matrix2Df& mosaic, const std::string& bayer_pattern);

/**
 * Reassemble R, G, B subplanes back into CFA mosaic.
 */
Matrix2Df reassemble_cfa_mosaic(
    const Matrix2Df& r_plane,
    const Matrix2Df& g_plane, 
    const Matrix2Df& b_plane,
    const std::string& bayer_pattern
);

/**
 * Get Bayer pixel offsets for R and B channels.
 * r_row/r_col: row/col parity (0 or 1) of R pixels
 * b_row/b_col: row/col parity (0 or 1) of B pixels
 */
void bayer_offsets(const std::string& bayer_pattern,
                   int& r_row, int& r_col, int& b_row, int& b_col);

/**
 * Simple nearest-neighbor debayer of a CFA mosaic into separate R, G, B planes.
 * Output planes have the same size as the input mosaic.
 */
struct DebayerResult {
    Matrix2Df R;
    Matrix2Df G;
    Matrix2Df B;
};

void debayer_nearest_neighbor_into(const Matrix2Df& mosaic,
                                   BayerPattern pattern,
                                   Matrix2Df& R_out,
                                   Matrix2Df& G_out,
                                   Matrix2Df& B_out);

DebayerResult debayer_nearest_neighbor(const Matrix2Df& mosaic,
                                       BayerPattern pattern);

// Variant that treats the input mosaic as a subregion of a larger Bayer mosaic.
// origin_x/origin_y are the top-left pixel coordinates of this subregion in the
// full image. This is required when demosaicing tiles extracted from a full
// frame, to keep Bayer parity consistent across tiles.
void debayer_nearest_neighbor_into(const Matrix2Df& mosaic,
                                   BayerPattern pattern,
                                   int origin_x,
                                   int origin_y,
                                   Matrix2Df& R_out,
                                   Matrix2Df& G_out,
                                   Matrix2Df& B_out);

void debayer_nearest_neighbor_strided_into(const float* mosaic_data,
                                           int mosaic_rows,
                                           int mosaic_cols,
                                           int mosaic_stride,
                                           BayerPattern pattern,
                                           int origin_x,
                                           int origin_y,
                                           Matrix2Df& R_out,
                                           Matrix2Df& G_out,
                                           Matrix2Df& B_out);

DebayerResult debayer_nearest_neighbor(const Matrix2Df& mosaic,
                                       BayerPattern pattern,
                                       int origin_x,
                                       int origin_y);

// Bilinear debayer with Bayer-parity aware tile origin handling.
void debayer_bilinear_into(const Matrix2Df& mosaic,
                           BayerPattern pattern,
                           Matrix2Df& R_out,
                           Matrix2Df& G_out,
                           Matrix2Df& B_out);

DebayerResult debayer_bilinear(const Matrix2Df& mosaic,
                               BayerPattern pattern);

void debayer_bilinear_into(const Matrix2Df& mosaic,
                           BayerPattern pattern,
                           int origin_x,
                           int origin_y,
                           Matrix2Df& R_out,
                           Matrix2Df& G_out,
                           Matrix2Df& B_out);

void debayer_bilinear_strided_into(const float* mosaic_data,
                                   int mosaic_rows,
                                   int mosaic_cols,
                                   int mosaic_stride,
                                   BayerPattern pattern,
                                   int origin_x,
                                   int origin_y,
                                   Matrix2Df& R_out,
                                   Matrix2Df& G_out,
                                   Matrix2Df& B_out);

DebayerResult debayer_bilinear(const Matrix2Df& mosaic,
                               BayerPattern pattern,
                               int origin_x,
                               int origin_y);

DebayerResult debayer_bilinear_region(const float* mosaic,
                                       int mosaic_height,
                                       int mosaic_width,
                                       int region_x,
                                       int region_y,
                                       int region_width,
                                       int region_height,
                                       BayerPattern pattern);

// Edge-directed demosaicing via OpenCV (VNG; ahd=true selects AHD/EA).
// origin_x/origin_y define the Bayer parity of the mosaic's top-left pixel,
// same convention as the other debayer functions.
void debayer_opencv_into(const Matrix2Df& mosaic,
                         BayerPattern pattern,
                         int origin_x,
                         int origin_y,
                         bool ahd,
                         Matrix2Df& R_out,
                         Matrix2Df& G_out,
                         Matrix2Df& B_out);

DebayerResult debayer_opencv(const Matrix2Df& mosaic,
                             BayerPattern pattern,
                             int origin_x,
                             int origin_y,
                             bool ahd = false);

} // namespace tile_compile::image
