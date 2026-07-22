#pragma once

#include "tile_compile/core/types.hpp"

#include <opencv2/core.hpp>

namespace tile_compile::core {

struct CfaWarpDims {
  int h2 = 0, w2 = 0;
  int out_h = 0, out_w = 0;
  int out_h2 = 0, out_w2 = 0;
  int sub_h = 0, sub_w = 0;
  int out_h_sub = 0, out_w_sub = 0;
};

CfaWarpDims compute_cfa_warp_dims(int h, int w, int out_height, int out_width);

struct CfaSubplanes {
  Matrix2Df a, b, c, d;
};

CfaSubplanes extract_cfa_subplanes(const Matrix2Df& mosaic, const CfaWarpDims& dims);

cv::Mat make_cfa_subplane_warp(const WarpMatrix& warp, float dx, float dy);

struct CfaSubplaneWarps {
  cv::Mat a, b, c, d;
};

CfaSubplaneWarps make_all_cfa_subplane_warps(const WarpMatrix& warp);

Matrix2Df reassemble_cfa_subplanes(const cv::Mat& a_w, const cv::Mat& b_w,
                                   const cv::Mat& c_w, const cv::Mat& d_w,
                                   const CfaWarpDims& dims);

} // namespace tile_compile::core
