#pragma once

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>

namespace tile_compile::metrics {

struct AqmhQualityMapCacheStats {
  uint64_t bytes_written = 0;
  uint64_t bytes_read = 0;
  uint64_t write_count = 0;
  uint64_t read_count = 0;
  uint64_t cache_hits = 0;
  uint64_t cache_misses = 0;
  size_t max_resident_maps_observed = 0;
};

class QualityMapCache {
public:
  QualityMapCache(std::filesystem::path cache_dir, std::string map_stream_id,
                  int full_width, int full_height,
                  const config::AqmhPyramidConfig &pyramid_cfg,
                  const config::AqmhStorageConfig &storage_cfg,
                  std::string canvas_mask_hash,
                  std::string execution_backend = "cpu");

  QualityMapCache(const QualityMapCache &) = delete;
  QualityMapCache &operator=(const QualityMapCache &) = delete;

  void write(size_t fi, const Matrix2Df &q_map);
  Matrix2Df read(size_t fi) const;
  Matrix2Df read_cached(size_t fi) const;
  bool has(size_t fi) const;
  void clear_memory_cache() const;
  void cleanup();

  const std::filesystem::path &cache_dir() const;
  std::filesystem::path map_path(size_t fi) const;
  AqmhQualityMapCacheStats stats() const;

  int full_width() const;
  int full_height() const;
  int stored_width() const;
  int stored_height() const;
  const std::string &map_stream_id() const;

private:
  std::filesystem::path metadata_path() const;
  void write_metadata() const;
  bool metadata_matches() const;
  void evict_to_limit_locked() const;
  Matrix2Df decode_file(size_t fi) const;
  Matrix2Df downsample_for_storage(const Matrix2Df &q_map) const;
  Matrix2Df upsample_to_full_resolution(const Matrix2Df &stored) const;

  std::filesystem::path cache_dir_;
  std::string map_stream_id_;
  int full_width_ = 0;
  int full_height_ = 0;
  int stored_width_ = 0;
  int stored_height_ = 0;
  config::AqmhPyramidConfig pyramid_cfg_;
  config::AqmhStorageConfig storage_cfg_;
  std::string canvas_mask_hash_;
  std::string config_hash_;
  std::string execution_backend_;

  mutable std::mutex mutex_;
  mutable AqmhQualityMapCacheStats stats_;
  mutable std::list<size_t> lru_;
  mutable std::unordered_map<size_t, std::pair<Matrix2Df, std::list<size_t>::iterator>>
      resident_;
};

std::string compute_aqmh_canvas_mask_hash(const std::vector<uint8_t> &mask,
                                          int width, int height);

} // namespace tile_compile::metrics
