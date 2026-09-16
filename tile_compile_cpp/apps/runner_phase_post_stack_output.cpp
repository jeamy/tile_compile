#include "runner_phase_post_stack_output.hpp"

#include "tile_compile/core/utils.hpp"
#include "tile_compile/image/cfa_processing.hpp"
#include "tile_compile/image/processing.hpp"
#include "tile_compile/image/normalization.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace tile_compile::runner {
void write_stretched_rgb_snapshot(
    const std::filesystem::path &path,
    const Matrix2Df &R_src,
    const Matrix2Df &G_src,
    const Matrix2Df &B_src,
    const std::vector<uint8_t> &canvas_mask,
    const std::vector<uint8_t> &statistics_mask,
    int canvas_rows,
    int canvas_cols,
    const io::FitsHeader &hdr,
    bool apply_stretch,
    const char *stage_tag) {
  Matrix2Df R_disk = R_src;
  Matrix2Df G_disk = G_src;
  Matrix2Df B_disk = B_src;

  if (!canvas_mask.empty() &&
      canvas_mask.size() == static_cast<size_t>(canvas_rows) * canvas_cols) {
    image::enforce_canvas_mask_on_rgb(R_disk, G_disk, B_disk, canvas_mask);
  }

  if (apply_stretch) {
    const auto stretch = core::stretch_rgb_to_u32_linear_from_zero_inplace(
        R_disk, G_disk, B_disk, statistics_mask);
    if (stretch.applied) {
      std::cout << "[" << stage_tag << "] RGB output stretch ["
                << stretch.low << ".." << stretch.high
                << "] -> [0..4294967295] (robust p99.9)"
                << " samples=" << stretch.sample_count << std::endl;
    }
  }

  std::error_code ec;
  std::filesystem::remove(path, ec);
  if (apply_stretch) {
    io::write_fits_rgb_u32(path, R_disk, G_disk, B_disk, hdr);
  } else {
    io::write_fits_rgb(path, R_disk, G_disk, B_disk, hdr);
  }
}

} // namespace tile_compile::runner
