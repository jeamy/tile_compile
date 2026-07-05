#pragma once

#include <cstddef>
#include <memory>

namespace tile_compile::metrics { class QualityMapCache; }

namespace tile_compile::reconstruction {

/// Prefetch coordinator for overlapping AQMH_MAPS with Q-map I/O in
/// AQMH_RECONSTRUCTION (Option C — see plan §3.1).
/// This does NOT let reconstruction compute weighted-MAD output before
/// global_weights are available; it only lets Q-map loading start early
/// so maps are resident in QualityMapCache by the time AQMH_GLOBAL_QUALITY
/// finishes. It deliberately does not promise frame-pixel prefetch because
/// no frame-pixel cache is part of this plan.
class AqmhPrefetchCoordinator {
public:
  explicit AqmhPrefetchCoordinator(
      size_t frame_count, metrics::QualityMapCache* q_map_cache);
  ~AqmhPrefetchCoordinator();

  /// Called by the maps phase: mark frame fi's Q-map as written to cache.
  /// Triggers async prefetch of that frame's Q-map into the reconstruction-side
  /// QualityMapCache resident LRU.
  void publish_frame(size_t fi);

  /// Called by the maps phase: signal that all frames are published.
  void finish();

  /// Called by reconstruction before starting the sigma-clip pass: block
  /// until every published Q-map has been prefetched (a no-op if maps already
  /// finished before reconstruction reached this point).
  void wait_all_prefetched();

  /// Returns true if prefetch is active (no errors occurred).
  bool prefetch_active() const;

  /// Returns the number of frames successfully prefetched.
  size_t prefetched_count() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace tile_compile::reconstruction
