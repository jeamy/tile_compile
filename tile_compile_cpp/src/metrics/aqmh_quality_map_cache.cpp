#include "tile_compile/metrics/aqmh_quality_map_cache.hpp"

#include "tile_compile/core/utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace tile_compile::metrics {
namespace fs = std::filesystem;
namespace core = tile_compile::core;
using json = nlohmann::json;

namespace {

constexpr int kAqmhMapFormatVersion = 2;

float clamp_q(float v) {
  if (!std::isfinite(v))
    return 0.0f;
  return std::clamp(v, 0.0f, 1.0f);
}

size_t dtype_bytes(const std::string &dtype) {
  if (dtype == "float32")
    return sizeof(float);
  if (dtype == "uint16")
    return sizeof(uint16_t);
  return sizeof(uint8_t);
}

std::string make_config_hash(const config::AqmhPyramidConfig &pyramid,
                             const config::AqmhStorageConfig &storage,
                             int full_width, int full_height,
                             const std::string &stream_id,
                             const std::string &mask_hash,
                             const std::string &execution_backend) {
  std::ostringstream oss;
  oss << "v=" << kAqmhMapFormatVersion << '\n';
  oss << "stream=" << stream_id << '\n';
  oss << "full=" << full_width << "x" << full_height << '\n';
  oss << "pyramid.scales=" << pyramid.scales << '\n';
  oss << "pyramid.base_window_px=" << pyramid.base_window_px << '\n';
  oss << std::setprecision(9);
  oss << "pyramid.w_sharp=" << pyramid.w_sharp << '\n';
  oss << "pyramid.w_snr=" << pyramid.w_snr << '\n';
  oss << "pyramid.k_artifact=" << pyramid.k_artifact << '\n';
  oss << "pyramid.frac_artifact_max=" << pyramid.frac_artifact_max << '\n';
  oss << "storage.resolution_divisor=" << storage.resolution_divisor << '\n';
  oss << "storage.dtype=" << storage.dtype << '\n';
  oss << "mask=" << mask_hash << '\n';
  oss << "execution_backend=" << execution_backend << '\n';
  const std::string payload = oss.str();
  return core::sha256_bytes(
      std::vector<uint8_t>(payload.begin(), payload.end()));
}

Matrix2Df empty_matrix() { return Matrix2Df(); }

} // namespace

QualityMapCache::QualityMapCache(
    fs::path cache_dir, std::string map_stream_id, int full_width,
    int full_height, const config::AqmhPyramidConfig &pyramid_cfg,
    const config::AqmhStorageConfig &storage_cfg, std::string canvas_mask_hash,
    std::string execution_backend)
    : cache_dir_(std::move(cache_dir)), map_stream_id_(std::move(map_stream_id)),
      full_width_(full_width), full_height_(full_height),
      pyramid_cfg_(pyramid_cfg), storage_cfg_(storage_cfg),
      canvas_mask_hash_(std::move(canvas_mask_hash)),
      execution_backend_(std::move(execution_backend)) {
  if (full_width_ <= 0 || full_height_ <= 0) {
    throw std::invalid_argument("QualityMapCache requires positive full size");
  }
  if (storage_cfg_.resolution_divisor != 1 &&
      storage_cfg_.resolution_divisor != 2 &&
      storage_cfg_.resolution_divisor != 4) {
    throw std::invalid_argument(
        "QualityMapCache resolution_divisor must be 1, 2, or 4");
  }
  if (storage_cfg_.dtype != "float32" && storage_cfg_.dtype != "uint16" &&
      storage_cfg_.dtype != "uint8") {
    throw std::invalid_argument(
        "QualityMapCache dtype must be float32, uint16, or uint8");
  }
  stored_width_ =
      std::max(1, (full_width_ + storage_cfg_.resolution_divisor - 1) /
                      storage_cfg_.resolution_divisor);
  stored_height_ =
      std::max(1, (full_height_ + storage_cfg_.resolution_divisor - 1) /
                      storage_cfg_.resolution_divisor);
  config_hash_ = make_config_hash(pyramid_cfg_, storage_cfg_, full_width_,
                                  full_height_, map_stream_id_,
                                  canvas_mask_hash_, execution_backend_);
  fs::create_directories(cache_dir_);
  if (!metadata_matches()) {
    cleanup();
    fs::create_directories(cache_dir_);
  }
  write_metadata();
}

QualityMapCache::~QualityMapCache() { clear_file_mappings(); }

void QualityMapCache::write(size_t fi, const Matrix2Df &q_map,
                            const std::vector<uint8_t> &source_valid_mask) {
  if (q_map.rows() != full_height_ || q_map.cols() != full_width_) {
    throw std::invalid_argument("AQMH quality map shape does not match cache");
  }
  fs::create_directories(cache_dir_);

  if (!source_valid_mask.empty() &&
      source_valid_mask.size() != static_cast<size_t>(q_map.size()))
    throw std::invalid_argument("AQMH source-valid mask shape mismatch");
  const Matrix2Df stored = downsample_for_storage(q_map, source_valid_mask);
  const fs::path path = map_path(fi);
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open AQMH quality-map cache file for write: " +
                             path.string());
  }

  if (storage_cfg_.resolution_divisor > 1) {
    const size_t pixels = static_cast<size_t>(q_map.size());
    std::vector<uint8_t> packed((pixels + 7u) / 8u, 0u);
    for (size_t i = 0; i < pixels; ++i)
      if (std::isfinite(q_map.data()[i]) && q_map.data()[i] == 0.0f &&
          (source_valid_mask.empty() || source_valid_mask[i] != 0u))
        packed[i / 8u] |= static_cast<uint8_t>(1u << (i % 8u));
    std::ofstream veto_out(veto_path(fi), std::ios::binary | std::ios::trunc);
    if (!veto_out) throw std::runtime_error("failed to write AQMH zero-veto mask");
    veto_out.write(reinterpret_cast<const char *>(packed.data()),
                   static_cast<std::streamsize>(packed.size()));
    if (!veto_out) throw std::runtime_error("failed while writing AQMH zero-veto mask");
  }
  if (!source_valid_mask.empty()) {
    core::write_text(source_mask_hash_path(fi),
                     core::sha256_bytes(source_valid_mask));
  }

  if (storage_cfg_.dtype == "float32") {
    std::vector<float> encoded(static_cast<size_t>(stored.size()));
    const auto n = static_cast<std::ptrdiff_t>(stored.size());
#pragma omp simd
    for (std::ptrdiff_t i = 0; i < n; ++i)
      encoded[static_cast<size_t>(i)] = clamp_q(stored.data()[i]);
    out.write(reinterpret_cast<const char *>(encoded.data()),
              static_cast<std::streamsize>(encoded.size() * sizeof(float)));
  } else if (storage_cfg_.dtype == "uint16") {
    std::vector<uint16_t> encoded(static_cast<size_t>(stored.size()));
    const auto n = static_cast<std::ptrdiff_t>(stored.size());
#pragma omp simd
    for (std::ptrdiff_t i = 0; i < n; ++i) {
      encoded[static_cast<size_t>(i)] = static_cast<uint16_t>(
          std::lround(clamp_q(stored.data()[i]) * 65535.0f));
    }
    out.write(reinterpret_cast<const char *>(encoded.data()),
              static_cast<std::streamsize>(encoded.size() * sizeof(uint16_t)));
  } else {
    std::vector<uint8_t> encoded(static_cast<size_t>(stored.size()));
    const auto n = static_cast<std::ptrdiff_t>(stored.size());
#pragma omp simd
    for (std::ptrdiff_t i = 0; i < n; ++i) {
      encoded[static_cast<size_t>(i)] = static_cast<uint8_t>(
          std::lround(clamp_q(stored.data()[i]) * 255.0f));
    }
    out.write(reinterpret_cast<const char *>(encoded.data()),
              static_cast<std::streamsize>(encoded.size() * sizeof(uint8_t)));
  }
  if (!out) {
    throw std::runtime_error("failed while writing AQMH quality-map cache file: " +
                             path.string());
  }

  std::lock_guard<std::mutex> lock(mutex_);
  stats_.write_count += 1;
  stats_.bytes_written += static_cast<uint64_t>(
      stored_width_ * stored_height_ * dtype_bytes(storage_cfg_.dtype));
  auto it = resident_.find(fi);
  if (it != resident_.end()) {
    lru_.erase(it->second.second);
    resident_.erase(it);
  }
}

Matrix2Df QualityMapCache::read(size_t fi) const {
  if (!metadata_matches())
    return empty_matrix();
  Matrix2Df decoded = decode_file(fi);
  if (decoded.size() == 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.cache_misses += 1;
    return decoded;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.read_count += 1;
    stats_.bytes_read += static_cast<uint64_t>(
        stored_width_ * stored_height_ * dtype_bytes(storage_cfg_.dtype));
  }
  Matrix2Df full = upsample_to_full_resolution(decoded);
  apply_zero_veto_mask(fi, full);
  return full;
}

Matrix2Df QualityMapCache::read_region(size_t fi, int y0, int rows) const {
  if (!metadata_matches() || y0 < 0 || rows <= 0 || y0 + rows > full_height_)
    return empty_matrix();

  // Route through resident cache/LRU (Fix for §3.2: read_region was bypassing the cache)
  if (storage_cfg_.max_resident_maps > 0) {
    // 1. Fast path: if map is resident, extract region from it.
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = resident_.find(fi);
      if (it != resident_.end()) {
        lru_.erase(it->second.second);
        lru_.push_front(fi);
        it->second.second = lru_.begin();
        stats_.cache_hits += 1;
        // Extract region from resident map
        Matrix2Df region(rows, full_width_);
        for (int ry = 0; ry < rows; ++ry) {
          const int src_y = y0 + ry;
          if (src_y >= 0 && src_y < full_height_) {
            const float* src_row = it->second.first.row(src_y).data();
            float* dst_row = region.row(ry).data();
            std::copy(src_row, src_row + full_width_, dst_row);
          }
        }
        return region;
      }
    }

    // 2. Cache miss: load full map WITHOUT holding the cache mutex.
    Matrix2Df full_map = read(fi);
    if (full_map.size() == 0) {
      return empty_matrix();
    }

    // 3. Double-check: another thread may have inserted while we were doing I/O.
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = resident_.find(fi);
      if (it != resident_.end()) {
        lru_.erase(it->second.second);
        lru_.push_front(fi);
        it->second.second = lru_.begin();
        stats_.cache_hits += 1;
        // Extract region from resident map
        Matrix2Df region(rows, full_width_);
        for (int ry = 0; ry < rows; ++ry) {
          const int src_y = y0 + ry;
          if (src_y >= 0 && src_y < full_height_) {
            const float* src_row = it->second.first.row(src_y).data();
            float* dst_row = region.row(ry).data();
            std::copy(src_row, src_row + full_width_, dst_row);
          }
        }
        return region;
      }

      // 4. Insert into cache, then evict until resident_.size() <= max_resident_maps.
      lru_.push_front(fi);
      auto inserted = resident_.emplace(
          fi, std::make_pair(std::move(full_map), lru_.begin())).first;
      evict_to_limit_locked();
      stats_.max_resident_maps_observed =
          std::max(stats_.max_resident_maps_observed, resident_.size());

      // Extract region from newly cached map
      Matrix2Df region(rows, full_width_);
      for (int ry = 0; ry < rows; ++ry) {
        const int src_y = y0 + ry;
        if (src_y >= 0 && src_y < full_height_) {
          const float* src_row = inserted->second.first.row(src_y).data();
          float* dst_row = region.row(ry).data();
          std::copy(src_row, src_row + full_width_, dst_row);
        }
      }
      return region;
    }
  }

  // Fallback: original direct read path when cache is disabled
  const int d = storage_cfg_.resolution_divisor;
  const auto stored_coord = [d](int y) {
    return (static_cast<float>(y) + 0.5f) / static_cast<float>(d) - 0.5f;
  };
  const int sy0 = std::clamp(
      static_cast<int>(std::floor(stored_coord(y0))), 0, stored_height_ - 1);
  const int sy1 = std::clamp(
      static_cast<int>(std::floor(stored_coord(y0 + rows - 1))) + 1,
      0, stored_height_ - 1);
  Matrix2Df stored = decode_stored_rows(fi, sy0, sy1 - sy0 + 1);
  if (stored.size() == 0) return stored;
  Matrix2Df out(rows, full_width_);
#pragma omp parallel for schedule(static)
  for (int ry = 0; ry < rows; ++ry) {
    const int y = y0 + ry;
    const float sy = stored_coord(y);
    const int base_y = static_cast<int>(std::floor(sy));
    const float ty = sy - base_y;
    const int ay0 = std::clamp(base_y, 0, stored_height_ - 1) - sy0;
    const int ay1 = std::clamp(base_y + 1, 0, stored_height_ - 1) - sy0;
    for (int x = 0; x < full_width_; ++x) {
      const float sx = (static_cast<float>(x) + 0.5f) /
                           static_cast<float>(d) - 0.5f;
      const int base_x = static_cast<int>(std::floor(sx));
      const float tx = sx - base_x;
      const int ax0 = std::clamp(base_x, 0, stored_width_ - 1);
      const int ax1 = std::clamp(base_x + 1, 0, stored_width_ - 1);
      const float v0 = (1.0f - tx) * stored(ay0, ax0) +
                       tx * stored(ay0, ax1);
      const float v1 = (1.0f - tx) * stored(ay1, ax0) +
                       tx * stored(ay1, ax1);
      out(ry, x) = clamp_q((1.0f - ty) * v0 + ty * v1);
    }
  }
  if (d > 1) {
    const size_t first_bit = static_cast<size_t>(y0) * full_width_;
    const size_t bit_count = static_cast<size_t>(rows) * full_width_;
    const size_t first_byte = first_bit / 8u;
    const size_t last_byte = (first_bit + bit_count + 7u) / 8u;
    std::vector<uint8_t> packed(last_byte - first_byte, 0u);
    const uint8_t *veto = mapped_veto_bytes(fi);
    if (veto != nullptr) {
      std::copy(veto + first_byte, veto + last_byte, packed.begin());
    } else {
      std::ifstream in(veto_path(fi), std::ios::binary);
      if (!in) return empty_matrix();
      in.seekg(static_cast<std::streamoff>(first_byte));
      in.read(reinterpret_cast<char *>(packed.data()),
              static_cast<std::streamsize>(packed.size()));
      if (!in) return empty_matrix();
    }
    for (size_t i = 0; i < bit_count; ++i) {
      const size_t global_bit = first_bit + i;
      if (((packed[global_bit / 8u - first_byte] >> (global_bit % 8u)) & 1u) != 0u)
        out.data()[i] = 0.0f;
    }
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.read_count;
    stats_.bytes_read += static_cast<uint64_t>(stored.size()) *
                         dtype_bytes(storage_cfg_.dtype);
  }
  return out;
}

Matrix2Df QualityMapCache::read_cached(size_t fi) const {
  if (storage_cfg_.max_resident_maps <= 0)
    return read(fi);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = resident_.find(fi);
    if (it != resident_.end()) {
      lru_.erase(it->second.second);
      lru_.push_front(fi);
      it->second.second = lru_.begin();
      stats_.cache_hits += 1;
      return it->second.first;
    }
  }

  Matrix2Df m = read(fi);
  if (m.size() == 0)
    return m;

  std::lock_guard<std::mutex> lock(mutex_);
  auto existing = resident_.find(fi);
  if (existing != resident_.end()) {
    lru_.erase(existing->second.second);
    resident_.erase(existing);
  }
  lru_.push_front(fi);
  auto inserted =
      resident_.emplace(fi, std::make_pair(std::move(m), lru_.begin())).first;
  evict_to_limit_locked();
  stats_.max_resident_maps_observed =
      std::max(stats_.max_resident_maps_observed, resident_.size());
  return inserted->second.first;
}

bool QualityMapCache::has(size_t fi) const {
  if (!metadata_matches())
    return false;
  std::error_code ec;
  const bool map_exists = fs::is_regular_file(map_path(fi), ec) && !ec;
  if (!map_exists) return false;
  if (storage_cfg_.resolution_divisor > 1)
    return fs::is_regular_file(veto_path(fi), ec) && !ec;
  return true;
}

void QualityMapCache::clear_memory_cache() const {
  std::lock_guard<std::mutex> lock(mutex_);
  resident_.clear();
  lru_.clear();
}

void QualityMapCache::cleanup() {
  clear_memory_cache();
  clear_file_mappings();
  std::error_code ec;
  fs::remove_all(cache_dir_, ec);
}

const fs::path &QualityMapCache::cache_dir() const { return cache_dir_; }

fs::path QualityMapCache::map_path(size_t fi) const {
  std::ostringstream name;
  name << "aqmh_" << map_stream_id_ << "_" << std::setw(6) << std::setfill('0')
       << fi << ".bin";
  return cache_dir_ / name.str();
}

fs::path QualityMapCache::veto_path(size_t fi) const {
  std::ostringstream name;
  name << "aqmh_" << map_stream_id_ << "_" << std::setw(6) << std::setfill('0')
       << fi << ".veto";
  return cache_dir_ / name.str();
}

fs::path QualityMapCache::source_mask_hash_path(size_t fi) const {
  std::ostringstream name;
  name << "aqmh_" << map_stream_id_ << "_" << std::setw(6) << std::setfill('0')
       << fi << ".maskhash";
  return cache_dir_ / name.str();
}

std::string QualityMapCache::source_mask_hash(size_t fi) const {
  try { return core::read_text(source_mask_hash_path(fi)); }
  catch (...) { return {}; }
}

AqmhQualityMapCacheStats QualityMapCache::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

int QualityMapCache::full_width() const { return full_width_; }
int QualityMapCache::full_height() const { return full_height_; }
int QualityMapCache::stored_width() const { return stored_width_; }
int QualityMapCache::stored_height() const { return stored_height_; }
const std::string &QualityMapCache::map_stream_id() const {
  return map_stream_id_;
}

fs::path QualityMapCache::metadata_path() const {
  return cache_dir_ / "aqmh_cache.json";
}

void QualityMapCache::write_metadata() const {
  std::lock_guard<std::mutex> lock(mutex_);
  json j;
  j["format_version"] = kAqmhMapFormatVersion;
  j["map_stream_id"] = map_stream_id_;
  j["full_width"] = full_width_;
  j["full_height"] = full_height_;
  j["stored_width"] = stored_width_;
  j["stored_height"] = stored_height_;
  j["dtype"] = storage_cfg_.dtype;
  j["resolution_divisor"] = storage_cfg_.resolution_divisor;
  j["canvas_mask_hash"] = canvas_mask_hash_;
  j["config_hash"] = config_hash_;
  j["execution_backend"] = execution_backend_;
  j["pyramid"] = {{"scales", pyramid_cfg_.scales},
                  {"base_window_px", pyramid_cfg_.base_window_px},
                  {"w_sharp", pyramid_cfg_.w_sharp},
                  {"w_snr", pyramid_cfg_.w_snr},
                  {"k_artifact", pyramid_cfg_.k_artifact},
                  {"frac_artifact_max", pyramid_cfg_.frac_artifact_max}};
  core::write_text(metadata_path(), j.dump(2));
}

bool QualityMapCache::metadata_matches() const {
  const fs::path path = metadata_path();
  std::error_code ec;
  if (!fs::exists(path, ec))
    return true;
  try {
    const json j = json::parse(core::read_text(path));
    return j.value("format_version", -1) == kAqmhMapFormatVersion &&
           j.value("map_stream_id", std::string()) == map_stream_id_ &&
           j.value("full_width", -1) == full_width_ &&
           j.value("full_height", -1) == full_height_ &&
           j.value("stored_width", -1) == stored_width_ &&
           j.value("stored_height", -1) == stored_height_ &&
           j.value("dtype", std::string()) == storage_cfg_.dtype &&
           j.value("resolution_divisor", -1) ==
               storage_cfg_.resolution_divisor &&
           j.value("canvas_mask_hash", std::string()) == canvas_mask_hash_ &&
           j.value("execution_backend", std::string("cpu")) ==
               execution_backend_ &&
           j.value("config_hash", std::string()) == config_hash_;
  } catch (...) {
    return false;
  }
}

void QualityMapCache::evict_to_limit_locked() const {
  const size_t limit = static_cast<size_t>(storage_cfg_.max_resident_maps);
  while (resident_.size() > limit && !lru_.empty()) {
    const size_t victim = lru_.back();
    lru_.pop_back();
    resident_.erase(victim);
  }
}

Matrix2Df QualityMapCache::decode_file(size_t fi) const {
  const fs::path path = map_path(fi);
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return empty_matrix();
  Matrix2Df stored(stored_height_, stored_width_);
  const auto n = static_cast<std::ptrdiff_t>(stored.size());
  if (storage_cfg_.dtype == "float32") {
    std::vector<float> raw(static_cast<size_t>(n));
    in.read(reinterpret_cast<char *>(raw.data()),
            static_cast<std::streamsize>(raw.size() * sizeof(float)));
    if (!in)
      return empty_matrix();
#pragma omp simd
    for (std::ptrdiff_t i = 0; i < n; ++i)
      stored.data()[i] = clamp_q(raw[static_cast<size_t>(i)]);
  } else if (storage_cfg_.dtype == "uint16") {
    std::vector<uint16_t> raw(static_cast<size_t>(n));
    in.read(reinterpret_cast<char *>(raw.data()),
            static_cast<std::streamsize>(raw.size() * sizeof(uint16_t)));
    if (!in)
      return empty_matrix();
#pragma omp simd
    for (std::ptrdiff_t i = 0; i < n; ++i)
      stored.data()[i] = static_cast<float>(raw[static_cast<size_t>(i)]) / 65535.0f;
  } else {
    std::vector<uint8_t> raw(static_cast<size_t>(n));
    in.read(reinterpret_cast<char *>(raw.data()),
            static_cast<std::streamsize>(raw.size() * sizeof(uint8_t)));
    if (!in)
      return empty_matrix();
#pragma omp simd
    for (std::ptrdiff_t i = 0; i < n; ++i)
      stored.data()[i] = static_cast<float>(raw[static_cast<size_t>(i)]) / 255.0f;
  }
  return stored;
}

Matrix2Df QualityMapCache::decode_stored_rows(
    size_t fi, int y0, int rows) const {
  if (y0 < 0 || rows <= 0 || y0 + rows > stored_height_) return empty_matrix();
  const size_t count = static_cast<size_t>(rows) * stored_width_;
  const size_t offset = static_cast<size_t>(y0) * stored_width_ *
                        dtype_bytes(storage_cfg_.dtype);
  Matrix2Df out(rows, stored_width_);
  const uint8_t *mapped = mapped_map_bytes(fi);
  std::ifstream in;
  if (mapped == nullptr) {
    in.open(map_path(fi), std::ios::binary);
    if (!in) return empty_matrix();
    in.seekg(static_cast<std::streamoff>(offset));
  }
  if (storage_cfg_.dtype == "float32") {
    if (mapped) std::memcpy(out.data(), mapped + offset, count * sizeof(float));
    else {
      in.read(reinterpret_cast<char *>(out.data()),
              static_cast<std::streamsize>(count * sizeof(float)));
      if (!in) return empty_matrix();
    }
    for (size_t i = 0; i < count; ++i) out.data()[i] = clamp_q(out.data()[i]);
  } else if (storage_cfg_.dtype == "uint16") {
    std::vector<uint16_t> raw(count);
    if (mapped) std::memcpy(raw.data(), mapped + offset, count * sizeof(uint16_t));
    else {
      in.read(reinterpret_cast<char *>(raw.data()),
              static_cast<std::streamsize>(count * sizeof(uint16_t)));
      if (!in) return empty_matrix();
    }
    for (size_t i = 0; i < count; ++i)
      out.data()[i] = static_cast<float>(raw[i]) / 65535.0f;
  } else {
    std::vector<uint8_t> raw(count);
    if (mapped) std::memcpy(raw.data(), mapped + offset, count);
    else {
      in.read(reinterpret_cast<char *>(raw.data()),
              static_cast<std::streamsize>(count));
      if (!in) return empty_matrix();
    }
    for (size_t i = 0; i < count; ++i)
      out.data()[i] = static_cast<float>(raw[i]) / 255.0f;
  }
  return out;
}

const uint8_t *QualityMapCache::mapped_map_bytes(size_t fi) const {
#ifdef _WIN32
  (void)fi;
  return nullptr;
#else
  std::lock_guard<std::mutex> lock(mutex_);
  auto &entry = file_mappings_[fi];
  if (entry.map_data) return static_cast<const uint8_t *>(entry.map_data);
  const auto path = map_path(fi);
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return nullptr;
  entry.map_size = static_cast<size_t>(stored_width_) * stored_height_ *
                   dtype_bytes(storage_cfg_.dtype);
  entry.map_data = ::mmap(nullptr, entry.map_size, PROT_READ, MAP_SHARED, fd, 0);
  ::close(fd);
  if (entry.map_data == MAP_FAILED) {
    entry.map_data = nullptr;
    entry.map_size = 0;
  }
  return static_cast<const uint8_t *>(entry.map_data);
#endif
}

const uint8_t *QualityMapCache::mapped_veto_bytes(size_t fi) const {
#ifdef _WIN32
  (void)fi;
  return nullptr;
#else
  std::lock_guard<std::mutex> lock(mutex_);
  auto &entry = file_mappings_[fi];
  if (entry.veto_data) return static_cast<const uint8_t *>(entry.veto_data);
  const auto path = veto_path(fi);
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return nullptr;
  entry.veto_size = (static_cast<size_t>(full_width_) * full_height_ + 7u) / 8u;
  entry.veto_data = ::mmap(nullptr, entry.veto_size, PROT_READ, MAP_SHARED, fd, 0);
  ::close(fd);
  if (entry.veto_data == MAP_FAILED) {
    entry.veto_data = nullptr;
    entry.veto_size = 0;
  }
  return static_cast<const uint8_t *>(entry.veto_data);
#endif
}

void QualityMapCache::clear_file_mappings() const {
#ifndef _WIN32
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto &[_, entry] : file_mappings_) {
    if (entry.map_data) ::munmap(entry.map_data, entry.map_size);
    if (entry.veto_data) ::munmap(entry.veto_data, entry.veto_size);
  }
  file_mappings_.clear();
#endif
}

Matrix2Df QualityMapCache::downsample_for_storage(
    const Matrix2Df &q_map,
    const std::vector<uint8_t> &source_valid_mask) const {
  const int d = storage_cfg_.resolution_divisor;
  Matrix2Df out(stored_height_, stored_width_);
  out.setZero();
  for (int oy = 0; oy < stored_height_; ++oy) {
    for (int ox = 0; ox < stored_width_; ++ox) {
      double sum = 0.0;
      int count = 0;
      for (int y = oy * d; y < std::min(full_height_, (oy + 1) * d); ++y) {
        for (int x = ox * d; x < std::min(full_width_, (ox + 1) * d); ++x) {
          const size_t i = static_cast<size_t>(y * full_width_ + x);
          if ((source_valid_mask.empty() || source_valid_mask[i] != 0u) &&
              std::isfinite(q_map(y, x))) {
            sum += clamp_q(q_map(y, x));
            ++count;
          }
        }
      }
      out(oy, ox) = count > 0 ? static_cast<float>(sum / count) : 0.0f;
    }
  }
  return out;
}

void QualityMapCache::apply_zero_veto_mask(size_t fi, Matrix2Df &map) const {
  if (storage_cfg_.resolution_divisor <= 1 || map.size() == 0) return;
  const size_t pixels = static_cast<size_t>(map.size());
  std::vector<uint8_t> packed((pixels + 7u) / 8u, 0u);
  std::ifstream in(veto_path(fi), std::ios::binary);
  if (!in) { map.resize(0, 0); return; }
  in.read(reinterpret_cast<char *>(packed.data()), static_cast<std::streamsize>(packed.size()));
  if (!in) { map.resize(0, 0); return; }
  for (size_t i = 0; i < pixels; ++i)
    if (((packed[i / 8u] >> (i % 8u)) & 1u) != 0u) map.data()[i] = 0.0f;
}

Matrix2Df QualityMapCache::upsample_to_full_resolution(
    const Matrix2Df &stored) const {
  const int d = storage_cfg_.resolution_divisor;
  if (d <= 1) {
    // No upsampling needed — stored resolution matches full resolution.
    Matrix2Df out(full_height_, full_width_);
    for (int y = 0; y < full_height_; ++y)
      for (int x = 0; x < full_width_; ++x)
        out(y, x) = clamp_q(stored(std::min(stored_height_ - 1, y),
                                   std::min(stored_width_ - 1, x)));
    return out;
  }

  // Mask-aware bilinear upsampling.
  // Each stored pixel covers a d×d block in full-resolution space.
  // We use the centre of the stored pixel as the sample point:
  //   centre_x = (sx + 0.5) * d - 0.5   (in full-res coordinates)
  // Invalid stored samples (clamped to 0 by the NaN→0 storage convention)
  // are treated as valid for the bilinear weights because all stored values
  // are finite after decode_file().  The bilinear interpolation therefore
  // always produces a finite result.
  Matrix2Df out(full_height_, full_width_);
  for (int y = 0; y < full_height_; ++y) {
    for (int x = 0; x < full_width_; ++x) {
      // Map to continuous stored coordinate (pixel-centre convention).
      const float sx = (static_cast<float>(x) + 0.5f) / static_cast<float>(d) - 0.5f;
      const float sy = (static_cast<float>(y) + 0.5f) / static_cast<float>(d) - 0.5f;
      const int x0 = static_cast<int>(std::floor(sx));
      const int y0 = static_cast<int>(std::floor(sy));
      const float tx = sx - static_cast<float>(x0);
      const float ty = sy - static_cast<float>(y0);

      // Clamp to stored bounds.
      const int x1 = std::min(stored_width_ - 1,  std::max(0, x0 + 1));
      const int y1 = std::min(stored_height_ - 1, std::max(0, y0 + 1));
      const int cx0 = std::clamp(x0, 0, stored_width_ - 1);
      const int cy0 = std::clamp(y0, 0, stored_height_ - 1);

      const float v00 = stored(cy0, cx0);
      const float v10 = stored(cy0, x1);
      const float v01 = stored(y1,  cx0);
      const float v11 = stored(y1,  x1);

      const float v = (1.0f - ty) * ((1.0f - tx) * v00 + tx * v10)
                    +         ty  * ((1.0f - tx) * v01 + tx * v11);
      out(y, x) = clamp_q(v);
    }
  }
  return out;
}

std::string compute_aqmh_canvas_mask_hash(const std::vector<uint8_t> &mask,
                                          int width, int height) {
  std::vector<uint8_t> payload;
  payload.reserve(sizeof(int) * 2 + mask.size());
  auto append_int = [&](int v) {
    for (int i = 0; i < 4; ++i)
      payload.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xff));
  };
  append_int(width);
  append_int(height);
  payload.insert(payload.end(), mask.begin(), mask.end());
  return core::sha256_bytes(payload);
}

} // namespace tile_compile::metrics
