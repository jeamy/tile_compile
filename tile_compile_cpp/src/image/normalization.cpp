#include "tile_compile/image/normalization.hpp"
#include "tile_compile/image/cfa_processing.hpp"

#include <opencv2/opencv.hpp>

#include <cstring>

namespace tile_compile::image {

void apply_normalization_inplace(Matrix2Df &img, const NormalizationScales &s,
                                 ColorMode mode,
                                 const std::string &bayer_pattern,
                                 int origin_x, int origin_y) {
  if (img.size() <= 0)
    return;
  if (mode != ColorMode::OSC) {
    img *= s.scale_mono;
    return;
  }

  int r_row, r_col, b_row, b_col;
  bayer_offsets(bayer_pattern, r_row, r_col, b_row, b_col);
  for (int y = 0; y < img.rows(); ++y) {
    const int gy = origin_y + y;
    for (int x = 0; x < img.cols(); ++x) {
      const int gx = origin_x + x;
      const int py = gy & 1;
      const int px = gx & 1;
      if (py == r_row && px == r_col) {
        img(y, x) *= s.scale_r;
      } else if (py == b_row && px == b_col) {
        img(y, x) *= s.scale_b;
      } else {
        img(y, x) *= s.scale_g;
      }
    }
  }
}

void apply_output_scaling_inplace(Matrix2Df &img, int origin_x, int origin_y,
                                  ColorMode mode,
                                  const std::string &bayer_pattern,
                                  float bg_mono, float bg_r, float bg_g,
                                  float bg_b, float pedestal) {
  if (img.size() <= 0)
    return;
  if (mode != ColorMode::OSC) {
    img *= bg_mono;
    img.array() += pedestal;
    return;
  }

  int r_row, r_col, b_row, b_col;
  bayer_offsets(bayer_pattern, r_row, r_col, b_row, b_col);
  for (int y = 0; y < img.rows(); ++y) {
    const int gy = origin_y + y;
    for (int x = 0; x < img.cols(); ++x) {
      const int gx = origin_x + x;
      const int py = gy & 1;
      const int px = gx & 1;
      if (py == r_row && px == r_col) {
        img(y, x) = img(y, x) * bg_r + pedestal;
      } else if (py == b_row && px == b_col) {
        img(y, x) = img(y, x) * bg_b + pedestal;
      } else {
        img(y, x) = img(y, x) * bg_g + pedestal;
      }
    }
  }
}

Matrix2Df apply_global_warp(const Matrix2Df &img, const WarpMatrix &warp,
                            ColorMode mode, int out_rows, int out_cols) {
  if (img.size() <= 0 || out_rows <= 0 || out_cols <= 0) {
    return Matrix2Df();
  }
  if (mode == ColorMode::OSC) {
    return warp_cfa_mosaic_via_subplanes(img, warp, out_rows, out_cols);
  }
  // For MONO/RGB: use standard warp with WARP_INVERSE_MAP
  // (imported via registration header would create circular dep — inline here)
  cv::Mat cv_img(img.rows(), img.cols(), CV_32F,
                 const_cast<float *>(img.data()));
  cv::Mat warp_matrix(2, 3, CV_32F);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      warp_matrix.at<float>(i, j) = warp(i, j);
  cv::Mat warped;
  cv::warpAffine(cv_img, warped, warp_matrix, cv::Size(out_cols, out_rows),
                 cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);
  Matrix2Df result(out_rows, out_cols);
  std::memcpy(result.data(), warped.data,
              static_cast<size_t>(result.size()) * sizeof(float));
  return result;
}

Matrix2Df apply_global_warp(const Matrix2Df &img, const WarpMatrix &warp,
                            ColorMode mode) {
  return apply_global_warp(img, warp, mode, img.rows(), img.cols());
}

} // namespace tile_compile::image
