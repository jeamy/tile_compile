#include "tile_compile/metrics/aqmh_frame_valid_mask.hpp"

#include "tile_compile/core/utils.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace tile_compile::metrics {

std::vector<uint8_t> compute_aqmh_frame_valid_mask(
    const Matrix2Df &frame, const std::vector<uint8_t> &canvas_mask,
    int width, int height) {
  if (frame.cols() != width || frame.rows() != height || width <= 0 || height <= 0)
    throw std::invalid_argument("AQMH frame-valid mask shape mismatch");
  if (!canvas_mask.empty() && canvas_mask.size() != static_cast<size_t>(width * height))
    throw std::invalid_argument("AQMH canvas mask shape mismatch");
  std::vector<uint8_t> mask(static_cast<size_t>(width * height), 0u);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      const size_t i = static_cast<size_t>(y * width + x);
      mask[i] = (canvas_mask.empty() || canvas_mask[i] != 0u) &&
                        std::isfinite(frame(y, x))
                    ? 1u : 0u;
    }
  return mask;
}

FrameValidMaskStore::FrameValidMaskStore(std::filesystem::path directory,
                                         int width, int height)
    : directory_(std::move(directory)), width_(width), height_(height) {
  if (width_ <= 0 || height_ <= 0) throw std::invalid_argument("invalid mask-store size");
  std::filesystem::create_directories(directory_);
}

FrameValidMaskStore::~FrameValidMaskStore() { clear_mappings(); }

std::filesystem::path FrameValidMaskStore::path(size_t frame_index) const {
  std::ostringstream name;
  name << "frame_valid_" << std::setw(6) << std::setfill('0') << frame_index << ".bin";
  return directory_ / name.str();
}

void FrameValidMaskStore::write(size_t frame_index, const std::vector<uint8_t> &mask) {
  const size_t pixels = static_cast<size_t>(width_) * static_cast<size_t>(height_);
  if (mask.size() != pixels) throw std::invalid_argument("frame-valid mask size mismatch");
  std::vector<uint8_t> packed((pixels + 7u) / 8u, 0u);
  for (size_t i = 0; i < pixels; ++i)
    if (mask[i]) packed[i / 8u] |= static_cast<uint8_t>(1u << (i % 8u));
  std::ofstream out(path(frame_index), std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write AQMH frame-valid mask");
  out.write(reinterpret_cast<const char *>(packed.data()),
            static_cast<std::streamsize>(packed.size()));
}

std::vector<uint8_t> FrameValidMaskStore::read(size_t frame_index) const {
  const size_t pixels = static_cast<size_t>(width_) * static_cast<size_t>(height_);
  std::vector<uint8_t> packed((pixels + 7u) / 8u, 0u);
  std::ifstream in(path(frame_index), std::ios::binary);
  if (!in) return {};
  in.read(reinterpret_cast<char *>(packed.data()), static_cast<std::streamsize>(packed.size()));
  if (!in) return {};
  std::vector<uint8_t> mask(pixels, 0u);
  for (size_t i = 0; i < pixels; ++i)
    mask[i] = (packed[i / 8u] >> (i % 8u)) & 1u;
  return mask;
}

std::vector<uint8_t> FrameValidMaskStore::read_region(
    size_t frame_index, int y0, int rows) const {
  if (y0 < 0 || rows <= 0 || y0 + rows > height_) return {};
  const size_t first_bit = static_cast<size_t>(y0) * width_;
  const size_t bit_count = static_cast<size_t>(rows) * width_;
  const size_t first_byte = first_bit / 8u;
  const size_t last_byte = (first_bit + bit_count + 7u) / 8u;
  std::vector<uint8_t> packed(last_byte - first_byte, 0u);
  const uint8_t *mapped = mapped_bytes(frame_index);
  if (mapped) {
    std::copy(mapped + first_byte, mapped + last_byte, packed.begin());
  } else {
    std::ifstream in(path(frame_index), std::ios::binary);
    if (!in) return {};
    in.seekg(static_cast<std::streamoff>(first_byte));
    in.read(reinterpret_cast<char *>(packed.data()),
            static_cast<std::streamsize>(packed.size()));
    if (!in) return {};
  }
  std::vector<uint8_t> mask(bit_count, 0u);
  for (size_t i = 0; i < bit_count; ++i) {
    const size_t global_bit = first_bit + i;
    const size_t local_byte = global_bit / 8u - first_byte;
    mask[i] = static_cast<uint8_t>(
        (packed[local_byte] >> (global_bit % 8u)) & 1u);
  }
  return mask;
}

const uint8_t *FrameValidMaskStore::mapped_bytes(size_t frame_index) const {
#ifdef _WIN32
  (void)frame_index;
  return nullptr;
#else
  std::lock_guard<std::mutex> lock(mapping_mutex_);
  auto it = mappings_.find(frame_index);
  if (it != mappings_.end()) return static_cast<const uint8_t *>(it->second);
  const auto p = path(frame_index);
  const int fd = ::open(p.c_str(), O_RDONLY);
  if (fd < 0) return nullptr;
  const size_t bytes =
      (static_cast<size_t>(width_) * height_ + 7u) / 8u;
  void *view = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
  ::close(fd);
  if (view == MAP_FAILED) return nullptr;
  mappings_.emplace(frame_index, view);
  return static_cast<const uint8_t *>(view);
#endif
}

void FrameValidMaskStore::clear_mappings() const {
#ifndef _WIN32
  std::lock_guard<std::mutex> lock(mapping_mutex_);
  const size_t bytes =
      (static_cast<size_t>(width_) * height_ + 7u) / 8u;
  for (const auto &[_, view] : mappings_)
    if (view) ::munmap(view, bytes);
  mappings_.clear();
#endif
}

std::string FrameValidMaskStore::hash(size_t frame_index) const {
  const auto mask = read(frame_index);
  return mask.empty() ? std::string() : core::sha256_bytes(mask);
}

} // namespace tile_compile::metrics
