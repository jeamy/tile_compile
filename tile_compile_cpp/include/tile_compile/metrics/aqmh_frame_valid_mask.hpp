#pragma once

#include "tile_compile/core/types.hpp"

#include <filesystem>
#include <vector>

namespace tile_compile::metrics {

std::vector<uint8_t> compute_aqmh_frame_valid_mask(
    const Matrix2Df &frame, const std::vector<uint8_t> &canvas_mask,
    int width, int height);

class FrameValidMaskStore {
public:
  FrameValidMaskStore(std::filesystem::path directory, int width, int height);
  void write(size_t frame_index, const std::vector<uint8_t> &mask);
  std::vector<uint8_t> read(size_t frame_index) const;
  std::vector<uint8_t> read_region(size_t frame_index, int y0, int rows) const;
  std::string hash(size_t frame_index) const;

private:
  std::filesystem::path path(size_t frame_index) const;
  std::filesystem::path directory_;
  int width_ = 0;
  int height_ = 0;
};

} // namespace tile_compile::metrics
