#pragma once

#include "tile_compile/core/types.hpp"

#include <cstdint>
#include <filesystem>
#include <streambuf>
#include <string>
#include <vector>

namespace tile_compile::runner {

std::string format_bytes(uint64_t bytes);

uint64_t estimate_total_file_bytes(const std::vector<std::filesystem::path> &paths);

bool message_indicates_disk_full(const std::string &message);

class TeeBuf : public std::streambuf {
public:
  TeeBuf(std::streambuf *a, std::streambuf *b);

protected:
  int overflow(int c) override;
  int sync() override;

private:
  std::streambuf *a_;
  std::streambuf *b_;
};

class DiskCacheFrameStore {
public:
  DiskCacheFrameStore();
  DiskCacheFrameStore(const std::filesystem::path &cache_dir, size_t n_frames,
                      int rows, int cols);
  ~DiskCacheFrameStore();

  DiskCacheFrameStore(const DiskCacheFrameStore &) = delete;
  DiskCacheFrameStore &operator=(const DiskCacheFrameStore &) = delete;
  DiskCacheFrameStore(DiskCacheFrameStore &&o) noexcept;
  DiskCacheFrameStore &operator=(DiskCacheFrameStore &&o) noexcept;

  void store(size_t fi, const Matrix2Df &frame);
  Matrix2Df load(size_t fi) const;
  Matrix2Df extract_tile(size_t fi, const Tile &t, int offset_x = 0,
                         int offset_y = 0) const;

  bool has_data(size_t fi) const;
  size_t size() const;
  int rows() const;
  int cols() const;

  void cleanup();

private:
  std::filesystem::path frame_path(size_t fi) const;

  std::filesystem::path cache_dir_;
  int rows_ = 0;
  int cols_ = 0;
  size_t frame_bytes_ = 0;
  std::vector<bool> has_data_;
};

} // namespace tile_compile::runner
