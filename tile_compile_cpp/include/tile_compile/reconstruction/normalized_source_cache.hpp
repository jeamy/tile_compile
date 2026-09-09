#pragma once
#include "tile_compile/reconstruction/forward_drizzle.hpp"
#include <cstdint>
#include <list>
#include <map>
#include <unordered_map>

namespace tile_compile::reconstruction {

// Existing runner cache format: <source_index>.raw, row-major native float32.
// Publication records existing files; it does not normalize, repair, or copy
// them. A later changed/truncated file fails closed when its bytes are loaded.
std::string publish_normalized_source_manifest(
    const fs::path &root, const registration::RegistrationSamplingPlan &plan);

// Verifies each <source_index>.raw against the published manifest SHA-256 on
// first touch, then keeps up to `memory_budget_mb`-worth of frames resident in
// an LRU (plan §30.72 O2). A re-load of a resident index re-stats the file and
// serves the cached matrix (no read, no SHA-256) ONLY if size and mtime are
// unchanged AND the mtime is strictly older than the instant we last verified
// it. Any size change, any forward mtime move, a same-tick rewrite, or a stat
// failure forces a full re-read + re-hash, so truncation and ordinary rewrites
// still fail closed. Not defended on the hit path: a rewrite that also rolls
// the file's mtime back to the previously seen value (deliberate tampering) ---
// the phase-start preflight and a fresh cache instance still catch that.
// `load()` is NOT thread-safe
// --- for a parallel consumer, give each worker its own instance via the
// (proto, budget_mb) constructor, which copies the already-parsed manifest and
// starts with an empty LRU (plan §30.72 O3). The returned reference stays valid
// until the next `load()` on THAT instance evicts the entry; the
// most-recently-loaded entry is never evicted, so the pre-existing "ref valid
// until the next load()" contract holds.
class VerifiedNormalizedSourceCache {
  fs::path root_;
  int width_ = 0, height_ = 0;
  std::map<size_t, std::string> hashes_;
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
  std::uint64_t hash_computations_ = 0;
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
  const std::string &manifest_hash() const { return manifest_hash_; }
  bool matches(const registration::RegistrationSamplingPlan &plan) const;
  // Diagnostics: number of SHA-256 verifications actually performed (one per
  // first touch / re-insert after eviction). Lets a test prove the LRU hit
  // path skips hashing.
  std::uint64_t hash_computation_count() const { return hash_computations_; }
  size_t resident_frame_count() const { return lru_.size(); }
  size_t capacity_frames() const { return capacity_; }
  // Bytes of one decoded frame (row-major float32). Lets a caller size a
  // per-worker clone.
  size_t frame_byte_size() const {
    return static_cast<size_t>(width_) * height_ * sizeof(float);
  }
};

} // namespace tile_compile::reconstruction
