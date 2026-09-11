#pragma once
#include "tile_compile/reconstruction/forward_drizzle.hpp"
#include <array>
#include <cstdint>
#include <list>
#include <map>
#include <unordered_map>
#include <vector>

namespace tile_compile::reconstruction {

// Existing runner cache format: <source_index>.raw, row-major native float32.
// Publication records existing files; it does not normalize, repair, or copy
// them. A later changed/truncated file fails closed when its bytes are loaded.
std::string publish_normalized_source_manifest(
    const fs::path &root, const registration::RegistrationSamplingPlan &plan);

// Trusted-Run source cache (§T1): no SHA-256 in the hot path. The manifest
// records file sizes and metadata identity only; a load reads the file and
// keeps up to `memory_budget_mb`-worth of frames resident in an LRU (plan
// §30.72 O2). A re-load of a resident index re-stats the file and serves the
// cached matrix (no read) ONLY if size and mtime are unchanged AND the mtime
// is strictly older than the instant we last verified it. Any size change,
// any forward mtime move, a same-tick rewrite, or a stat failure forces a
// full re-read. Truncation is detected via the size check; in-place content
// rewrites at the same size are NOT detected (trusted-run contract, §1).
// `load()` is NOT thread-safe --- for a parallel consumer, give each worker
// its own instance via the (proto, budget_mb) constructor, which copies the
// already-parsed manifest and starts with an empty LRU (plan §30.72 O3).
class VerifiedNormalizedSourceCache {
  fs::path root_;
  int width_ = 0, height_ = 0;
  std::map<size_t, std::string> hashes_;  // source_index -> (unused in trusted run)
  std::string manifest_hash_, context_hash_;
  size_t capacity_ = 1;
  struct Entry {
    size_t index = 0;
    Matrix2Df image;
    std::uintmax_t file_size = 0;
    std::filesystem::file_time_type mtime{};
    std::filesystem::file_time_type verified_at{};
  };
  std::list<Entry> lru_;                                // front = most recent
  std::unordered_map<size_t, std::list<Entry>::iterator> resident_;
  std::uint64_t load_calls_ = 0, lru_hits_ = 0, evictions_ = 0;
  std::uint64_t bytes_read_ = 0, expanded_floats_ = 0;
  const Matrix2Df &verify_and_insert(size_t source_index);
public:
  VerifiedNormalizedSourceCache(const fs::path &root,
      const registration::RegistrationSamplingPlan &expected,
      size_t memory_budget_mb = 512);
  // Per-worker clone: same verified manifest, independent (empty) LRU sized to
  // `memory_budget_mb`. Cheap --- no file stat, no manifest re-parse.
  VerifiedNormalizedSourceCache(const VerifiedNormalizedSourceCache &proto,
                                size_t memory_budget_mb);
  const Matrix2Df &load(size_t source_index);

  // Partial read: returns a (y1-y0) x (x1-x0) matrix whose element (r, c)
  // equals load(source_index)(y0 + r, x0 + c). Reads only the covering Y rows
  // from disk (no SHA verification — trusted run). Ranges are clamped to the
  // frame; an empty range returns a 0x0 matrix. Not thread-safe (like load()).
  Matrix2Df read_rect(size_t source_index, int y0, int y1, int x0, int x1);
  Matrix2Df read_region(size_t source_index, int y0, int y1) {
    return read_rect(source_index, y0, y1, 0, width_);
  }
  // I/O diagnostics. `bytes_read` counts every byte read from a `.raw` file:
  // a load() whole-frame read and each read_rect window.
  std::uint64_t bytes_read() const { return bytes_read_; }
  std::uint64_t expanded_floats() const { return expanded_floats_; }
  void reset_io_counters() { bytes_read_ = expanded_floats_ = 0; }

  const std::string &manifest_hash() const { return manifest_hash_; }
  bool matches(const registration::RegistrationSamplingPlan &plan) const;
  size_t resident_frame_count() const { return lru_.size(); }
  size_t capacity_frames() const { return capacity_; }
  std::uint64_t load_call_count() const { return load_calls_; }
  std::uint64_t lru_hit_count() const { return lru_hits_; }
  std::uint64_t eviction_count() const { return evictions_; }
  // Bytes of one decoded frame (row-major float32). Lets a caller size a
  // per-worker clone.
  size_t frame_byte_size() const {
    return static_cast<size_t>(width_) * height_ * sizeof(float);
  }
};

} // namespace tile_compile::reconstruction
