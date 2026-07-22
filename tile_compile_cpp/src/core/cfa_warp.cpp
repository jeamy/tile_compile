#include "tile_compile/core/cfa_warp.hpp"

#include <algorithm>

namespace tile_compile::core {

CfaWarpDims compute_cfa_warp_dims(int h, int w, int out_height, int out_width) {
  CfaWarpDims d;
  d.h2 = h - (h % 2);
  d.w2 = w - (w % 2);
  d.out_h = (out_height > 0) ? out_height : h;
  d.out_w = (out_width > 0) ? out_width : w;
  d.out_h2 = d.out_h - (d.out_h % 2);
  d.out_w2 = d.out_w - (d.out_w % 2);
  d.sub_h = d.h2 / 2;
  d.sub_w = d.w2 / 2;
  d.out_h_sub = std::max(1, d.out_h2 / 2);
  d.out_w_sub = std::max(1, d.out_w2 / 2);
  return d;
}

CfaSubplanes extract_cfa_subplanes(const Matrix2Df& mosaic, const CfaWarpDims& dims) {
  CfaSubplanes s;
  s.a = Matrix2Df(dims.sub_h, dims.sub_w);
  s.b = Matrix2Df(dims.sub_h, dims.sub_w);
  s.c = Matrix2Df(dims.sub_h, dims.sub_w);
  s.d = Matrix2Df(dims.sub_h, dims.sub_w);
  for (int y = 0; y < dims.sub_h; ++y) {
    for (int x = 0; x < dims.sub_w; ++x) {
      s.a(y, x) = mosaic(y * 2, x * 2);
      s.b(y, x) = mosaic(y * 2, x * 2 + 1);
      s.c(y, x) = mosaic(y * 2 + 1, x * 2);
      s.d(y, x) = mosaic(y * 2 + 1, x * 2 + 1);
    }
  }
  return s;
}

cv::Mat make_cfa_subplane_warp(const WarpMatrix& warp, float dx, float dy) {
  const float a2_00 = warp(0, 0);
  const float a2_01 = warp(0, 1);
  const float a2_10 = warp(1, 0);
  const float a2_11 = warp(1, 1);
  const float t_x = warp(0, 2) * 0.5f;
  const float t_y = warp(1, 2) * 0.5f;
  const float new_tx = t_x + (a2_00 * dx + a2_01 * dy) - dx;
  const float new_ty = t_y + (a2_10 * dx + a2_11 * dy) - dy;
  return (cv::Mat_<float>(2, 3) << a2_00, a2_01, new_tx, a2_10, a2_11, new_ty);
}

CfaSubplaneWarps make_all_cfa_subplane_warps(const WarpMatrix& warp) {
  return CfaSubplaneWarps{
    make_cfa_subplane_warp(warp, -0.25f, -0.25f),
    make_cfa_subplane_warp(warp, 0.25f, -0.25f),
    make_cfa_subplane_warp(warp, -0.25f, 0.25f),
    make_cfa_subplane_warp(warp, 0.25f, 0.25f),
  };
}

Matrix2Df reassemble_cfa_subplanes(const cv::Mat& a_w, const cv::Mat& b_w,
                                   const cv::Mat& c_w, const cv::Mat& d_w,
                                   const CfaWarpDims& dims) {
  Matrix2Df out = Matrix2Df::Zero(dims.out_h, dims.out_w);
  for (int y = 0; y < dims.out_h_sub; ++y) {
    for (int x = 0; x < dims.out_w_sub; ++x) {
      out(y * 2, x * 2) = a_w.at<float>(y, x);
      out(y * 2, x * 2 + 1) = b_w.at<float>(y, x);
      out(y * 2 + 1, x * 2) = c_w.at<float>(y, x);
      out(y * 2 + 1, x * 2 + 1) = d_w.at<float>(y, x);
    }
  }
  return out;
}

} // namespace tile_compile::core
