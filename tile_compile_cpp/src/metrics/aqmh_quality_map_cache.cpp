#include "tile_compile/metrics/aqmh_quality_map_cache.hpp"

#include "tile_compile/core/utils.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace tile_compile::metrics {
namespace fs = std::filesystem;
namespace core = tile_compile::core;
using json = nlohmann::json;

namespace {

constexpr int kAqmhMapFormatVersion = 1;

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

void QualityMapCache::write(size_t fi, const Matrix2Df &q_map) {
  if (q_map.rows() != full_height_ || q_map.cols() != full_width_) {
    throw std::invalid_argument("AQMH quality map shape does not match cache");
  }
  fs::create_directories(cache_dir_);

  const Matrix2Df stored = downsample_for_storage(q_map);
  const fs::path path = map_path(fi);
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open AQMH quality-map cache file for write: " +
                             path.string());
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
  return upsample_to_full_resolution(decoded);
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
  return fs::is_regular_file(map_path(fi), ec) && !ec;
}

void QualityMapCache::clear_memory_cache() const {
  std::lock_guard<std::mutex> lock(mutex_);
  resident_.clear();
  lru_.clear();
}

void QualityMapCache::cleanup() {
  clear_memory_cache();
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
  if (storage_cfg_.dtype == "float32") {
    for (int y = 0; y < stored_height_; ++y) {
      for (int x = 0; x < stored_width_; ++x) {
        float v = 0.0f;
        in.read(reinterpret_cast<char *>(&v), sizeof(float));
        if (!in)
          return empty_matrix();
        stored(y, x) = clamp_q(v);
      }
    }
  } else if (storage_cfg_.dtype == "uint16") {
    for (int y = 0; y < stored_height_; ++y) {
      for (int x = 0; x < stored_width_; ++x) {
        uint16_t v = 0;
        in.read(reinterpret_cast<char *>(&v), sizeof(uint16_t));
        if (!in)
          return empty_matrix();
        stored(y, x) = static_cast<float>(v) / 65535.0f;
      }
    }
  } else {
    for (int y = 0; y < stored_height_; ++y) {
      for (int x = 0; x < stored_width_; ++x) {
        uint8_t v = 0;
        in.read(reinterpret_cast<char *>(&v), sizeof(uint8_t));
        if (!in)
          return empty_matrix();
        stored(y, x) = static_cast<float>(v) / 255.0f;
      }
    }
  }
  return stored;
}

Matrix2Df QualityMapCache::downsample_for_storage(const Matrix2Df &q_map) const {
  const int d = storage_cfg_.resolution_divisor;
  Matrix2Df out(stored_height_, stored_width_);
  out.setZero();
  for (int oy = 0; oy < stored_height_; ++oy) {
    for (int ox = 0; ox < stored_width_; ++ox) {
      double sum = 0.0;
      int count = 0;
      for (int y = oy * d; y < std::min(full_height_, (oy + 1) * d); ++y) {
        for (int x = ox * d; x < std::min(full_width_, (ox + 1) * d); ++x) {
          sum += clamp_q(q_map(y, x));
          ++count;
        }
      }
      out(oy, ox) = count > 0 ? static_cast<float>(sum / count) : 0.0f;
    }
  }
  return out;
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
