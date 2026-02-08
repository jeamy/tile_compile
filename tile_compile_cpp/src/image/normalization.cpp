#include "tile_compile/image/normalization.hpp"
#include "tile_compile/image/cfa_processing.hpp"

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

} // namespace tile_compile::image
