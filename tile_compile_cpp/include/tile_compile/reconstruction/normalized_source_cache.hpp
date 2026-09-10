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

  // ---- §30.81 step 3a-3: run-internal source block-check index -------------
  // Independent of the image LRU. Built from the SAME bytes as the whole-file
  // SHA-256 on first touch, and published only after that whole-file hash
  // matches the manifest. It is built once: either from the buffer load()
  // already verified (no extra read), or --- on a region-first touch --- by
  // streaming the file one block at a time (peak scratch = one block). A later
  // read_region / read_rect reads and SHA-256-verifies ONLY the blocks
  // covering its Y range. Blocks span whole rows (row-major float32), so a Y
  // window never straddles a partial block; an X-narrowed rect verifies the
  // SAME blocks and reads the SAME bytes as the full-width rect with the same
  // Y range --- narrowing X costs nothing extra but saves no I/O either, only
  // output memory and the copy. Contract boundary: only blocks a read actually
  // touches are verified --- a change confined to an unread block is detected
  // only when a later read covers it. The index is trusted across image-LRU
  // eviction while the file's (size, mtime) are unchanged and mtime is strictly
  // older than the instant it was built (the same guard load() uses on the hit
  // path); any drift rebuilds it, which re-verifies the whole file. RAM-only:
  // a fresh cache instance (process restart) starts with no block index and
  // re-verifies on first touch.
  struct BlockIndex {
    std::uintmax_t file_size = 0;
    std::filesystem::file_time_type mtime{};
    std::filesystem::file_time_type verified_at{};
    std::vector<std::array<unsigned char, 32>> block_sha;
  };
  size_t block_rows_ = 1;  // rows per block (whole rows, >= 1)
  std::map<size_t, BlockIndex> block_index_;
  std::uint64_t blocks_verified_ = 0, block_index_builds_ = 0;
  std::uint64_t bytes_read_ = 0, expanded_floats_ = 0;
  const BlockIndex &ensure_block_index(size_t source_index);
  // Per-block SHA-256 over an in-memory frame buffer (the shared digest
  // routine; load() feeds it the buffer it already verified, so no extra read
  // and no extra peak).
  std::vector<std::array<unsigned char, 32>> digest_blocks(
      const unsigned char *data, size_t bytes) const;
  // Store `digs` as the block index for `source_index` with the file's current
  // (size, mtime) and a fresh verified_at stamp; replaces any stale entry.
  void publish_block_index(size_t source_index, const std::filesystem::path &path,
                           std::vector<std::array<unsigned char, 32>> digs);
public:
  VerifiedNormalizedSourceCache(const fs::path &root,
      const registration::RegistrationSamplingPlan &expected,
      size_t memory_budget_mb = 512);
  // Per-worker clone: same verified manifest, independent (empty) LRU sized to
  // `memory_budget_mb`. Cheap --- no file stat, no manifest re-parse.
  VerifiedNormalizedSourceCache(const VerifiedNormalizedSourceCache &proto,
                                size_t memory_budget_mb);
  const Matrix2Df &load(size_t source_index);

  // Block-verified partial read: returns a (y1-y0) x (x1-x0) matrix whose
  // element (r, c) equals load(source_index)(y0 + r, x0 + c). Only the blocks
  // covering [y0, y1) are read and SHA-256-checked (see BlockIndex above).
  // Ranges are clamped to the frame; an empty range returns a 0x0 matrix.
  // Throws NORMALIZED_CACHE_BLOCK_MISMATCH if a covered block no longer matches
  // its published digest, NORMALIZED_CACHE_CONTENT_MISMATCH if the whole-file
  // hash fails when the index is (re)built. Not thread-safe (like load()).
  Matrix2Df read_rect(size_t source_index, int y0, int y1, int x0, int x1);
  Matrix2Df read_region(size_t source_index, int y0, int y1) {
    return read_rect(source_index, y0, y1, 0, width_);
  }
  // Rows per block index entry (whole rows). Lets a test reason about block
  // read amplification for a given Y window.
  size_t block_row_span() const { return block_rows_; }
  // Diagnostics for the block-check path. `bytes_read` counts every byte read
  // from a `.raw` file on either entry point: a load() / re-verify whole-frame
  // read, a region-first-touch streaming build, and each read_rect window.
  std::uint64_t blocks_verified() const { return blocks_verified_; }
  std::uint64_t block_index_builds() const { return block_index_builds_; }
  std::uint64_t bytes_read() const { return bytes_read_; }
  std::uint64_t expanded_floats() const { return expanded_floats_; }
  void reset_io_counters() {
    blocks_verified_ = block_index_builds_ = bytes_read_ = expanded_floats_ = 0;
  }

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
