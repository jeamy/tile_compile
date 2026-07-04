#pragma once

#include "tile_compile/core/types.hpp"

#include <filesystem>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace tile_compile::metrics {

std::vector<uint8_t> compute_aqmh_frame_valid_mask(
    const Matrix2Df &frame, const std::vector<uint8_t> &canvas_mask,
    int width, int height);

class FrameValidMaskStore {
public:
  FrameValidMaskStore(std::filesystem::path directory, int width, int height);
  ~FrameValidMaskStore();
  void write(size_t frame_index, const std::vector<uint8_t> &mask);
  std::vector<uint8_t> read(size_t frame_index) const;
  std::vector<uint8_t> read_region(size_t frame_index, int y0, int rows) const;
  std::string hash(size_t frame_index) const;

private:
  std::filesystem::path path(size_t frame_index) const;
  const uint8_t *mapped_bytes(size_t frame_index) const;
  void clear_mappings() const;
  std::filesystem::path directory_;
  int width_ = 0;
  int height_ = 0;
  mutable std::mutex mapping_mutex_;
  mutable std::unordered_map<size_t, void *> mappings_;
};

} // namespace tile_compile::metrics
