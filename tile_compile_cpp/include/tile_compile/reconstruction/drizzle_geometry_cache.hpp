// Plan section 11.14 P1/P2 --- authoritative forward-drizzle geometry, built
// once per (variant, frame) and read back per stripe from a spatial index,
// replacing the per-stripe full-source re-enumeration in
// `enumerate_drizzle_stripe_leaf_cells` for LOCAL-WARP frames.
//
// Contract (plan 11.14.3 / 11.14.4):
//   * The leaves are produced by the UNCHANGED CPU evaluation
//     (`sample_leaves` -> `subdivide_local` -> `invert_local_source_to_canvas`).
//     No minimax, no re-interpolation of image values. Corner coordinates are
//     stored with their exact `double` bits.
//   * Each source sample passes through `sample_leaves` EXACTLY ONCE per
//     (variant, frame). Subdivision within a sample is still allowed and is
//     counted separately.
//   * The dense footprint (pixfrac == 1) and the drizzle coverage droplet
//     (pixfrac == cfg.pixfrac) are DISTINCT variants. Identical variants
//     (cfg.pixfrac == 1) are deduplicated by identity.
//   * Frame-exclusion rates are finalised by the build; a discarded frame or
//     sample never reappears in a later coverage / profile phase. Chunk height
//     does not influence that set.
//   * Transactional generation commit + per-file checksums + schema/identity
//     verification, like `drizzle_profile_store`. No silent use of an
//     incomplete generation after an abort. Disk pre-flight before writing.
//   * No full-leaf cache of all frames in RAM: the builder streams budgeted
//     record batches to disk; the reader holds only a bounded working window
//     plus the compact per-frame band index.
//
// Affine frames are NOT cached: their per-stripe cost is already O(source_px)
// (the enumerator inverse-maps the stripe corners and bounds the source rows),
// and they are not "eligible_local_frames" for the P2 acceptance counter.

#pragma once

#include "tile_compile/reconstruction/forward_drizzle.hpp"

#include <cstdint>
#include <functional>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace tile_compile::reconstruction {

namespace fs = std::filesystem;

// One explicit geometry variant. `pixfrac` is the only discriminator today
// (kernel/scale/subdivision are shared), but it is a struct so a future kernel
// variant slots in without changing the identity surface.
struct GeometryVariant {
  float pixfrac = 0.8f;
  bool operator==(const GeometryVariant &) const = default;
};

// Binds everything plan 11.14.3 lists: sampling + model identity, canvas, CFA
// contract, internal_scale, the variant pixfrac, subdivision + inversion
// params, coordinate convention and algorithm version. Per variant.
struct GeometryCacheIdentity {
  std::string geometry_hash;   // sha256 over all of the above, incl. pixfrac
  std::string sampling_plan_hash;
  int canvas_width_native = 0;
  int canvas_height_native = 0;
  int internal_scale = 0;
  float pixfrac = 0.0f;
  ColorMode color_mode = ColorMode::MONO;
  bool operator==(const GeometryCacheIdentity &) const = default;
};

GeometryCacheIdentity make_geometry_cache_identity(
    const registration::RegistrationSamplingPlan &plan,
    const config::ReconstructionDrizzleConfig &cfg,
    const GeometryVariant &variant,
    const ForwardDrizzleSubdivisionParams &subdivision = {});

// Per-(variant, frame) build/read statistics. `top_level_sample_leaves_calls`
// is the P2 acceptance number: for a fresh build it must equal
// `source_width * source_height` per eligible local frame per variant, and a
// reader must add ZERO further calls.
struct GeometryCacheFrameStats {
  std::uint64_t source_index = 0;
  std::uint64_t samples_total = 0;   // source_width * source_height
  std::uint64_t top_level_sample_leaves_calls = 0;  // == samples_total for a build
  std::uint64_t leaves_written = 0;
  std::uint64_t samples_discarded = 0;
  double subdivision_error_rate = 0.0;
  bool excluded = false;   // rate > per_frame_inversion_error_rate_max
};

struct GeometryCacheBuildResult {
  fs::path generation_dir;
  // One entry per variant, in the order passed to build_drizzle_geometry_cache.
  std::vector<GeometryCacheIdentity> identities;
  std::vector<GeometryCacheFrameStats> frames;  // flattened over variants
  std::uint64_t total_leaves = 0;
  std::uint64_t total_record_bytes = 0;
  std::uint64_t index_bytes = 0;
  // TOTAL work-time (summed over frames, not wall) split of the build: the
  // sample_leaves geometry sweep (the O(V*N*P) term the cache pays ONCE) vs.
  // the record/index write + fsync (the disk-I/O term). With workers > 1 the
  // wall time is roughly these divided by the worker count.
  double sample_leaves_seconds = 0.0;
  double write_seconds = 0.0;
  double wall_seconds = 0.0;   // measured end-to-end build wall time
  int workers_used = 1;
  // Frames excluded by subdivision-error rate (finalised here, plan 11.14.3).
  std::vector<std::pair<std::string, double>> excluded_frames;
};

// Build the geometry cache for every requested variant. Only local-warp frames
// are materialised (affine frames stay on the cheap inverse-map path).
// `memory_budget_bytes` is a floor check on the one-source-row staging buffer.
//
// `max_workers` (plan 11.14.5 P3): each (variant, frame) is an independent
// task --- its own sample_leaves sweep, its own .rows/.leaves files, no shared
// mutable state. With max_workers > 1 they run on an OpenMP team; results are
// still assembled into the manifest in deterministic (variant, source_index)
// order, so the committed store is byte-identical to the max_workers == 1
// reference regardless of the worker count or scheduling. The per-worker
// footprint is one source row of leaf records (independent of frame count and
// canvas height).
// Progress callback: called once after each COMPLETED (variant, frame) task
// that was actually a local-warp frame (the cheap non-local tasks are not
// reported --- there can be hundreds of them and they finish instantly).
// `done`/`total` count present tasks only. Called under an internal lock, so
// it is safe to write to a shared log/emitter without its own locking.
using GeometryCacheProgressFn =
    std::function<void(std::size_t done, std::size_t total,
                       const std::string &detail)>;

GeometryCacheBuildResult build_drizzle_geometry_cache(
    const fs::path &root, const registration::RegistrationSamplingPlan &plan,
    const config::ReconstructionDrizzleConfig &cfg,
    const std::vector<GeometryVariant> &variants,
    const ForwardDrizzleSubdivisionParams &subdivision,
    std::uint64_t memory_budget_bytes, int max_workers = 1,
    const GeometryCacheProgressFn &on_progress = nullptr);

// Verified reader over ONE committed generation.
//
// On open it loads and hashes only the per-frame ROW INDEX (`.rows`, a few tens
// of KiB per frame) and validates structure: schema + algorithm version,
// per-variant identity, exact expected local-frame population (no missing / no
// extra / no duplicate), dimension + exclusion-stat consistency, and every
// row's (offset, count) against the real `.leaves` file length. Leaf RECORDS
// are NEVER fully resident --- `enumerate_stripe` seek-reads only the row
// blocks that intersect the requested stripe into a bounded reused buffer.
//
// `verify_record_bytes` additionally streams and SHA-256-checks every `.leaves`
// file on open (correct but O(total store bytes) of I/O --- the caller budgets
// this); when false, record integrity rests on the `.rows` hash, the structural
// checks, and the transactional commit chain.
class DrizzleGeometryCacheReader {
public:
  DrizzleGeometryCacheReader(
      const fs::path &root,
      const std::vector<GeometryCacheIdentity> &expected,
      const std::vector<std::size_t> &expected_local_source_indices,
      bool verify_record_bytes = false);
  ~DrizzleGeometryCacheReader();
  DrizzleGeometryCacheReader(const DrizzleGeometryCacheReader &) = delete;
  DrizzleGeometryCacheReader &
  operator=(const DrizzleGeometryCacheReader &) = delete;

  // True if this (variant pixfrac, source_index) pair is a materialised
  // local-warp frame in the cache. Affine frames and unknown pairs return
  // false --- the caller then uses the legacy enumerator.
  bool has_frame(float pixfrac, std::size_t source_index) const;

  // Per-(variant, frame) stats finalised by the build. `present` is false if
  // this pixfrac/source_index pair is not a materialised local frame.
  struct FrameStatsView {
    bool present = false;
    bool excluded = false;
    std::uint64_t samples_total = 0;
    std::uint64_t samples_discarded = 0;
    double subdivision_error_rate = 0.0;
  };
  FrameStatsView frame_stats(float pixfrac, std::size_t source_index) const;

  // Drop-in for `enumerate_drizzle_stripe_leaf_cells`: replays every leaf cell
  // whose target bbox intersects the internal-canvas stripe
  // [y_begin, y_begin + rows), in the SAME canonical
  // (source_y, source_x, leaf_order, y, x) order the source scan produces, so
  // downstream double accumulation is bit-identical.
  void enumerate_stripe(float pixfrac, std::size_t source_index, int scale,
                        int y_begin, int rows,
                        const DrizzleLeafCellSink &sink) const;

  // Frames excluded by subdivision-error rate, as finalised by the build.
  const std::vector<std::pair<std::string, double>> &excluded_frames() const;

  // Bytes the reader keeps resident (row indices + fixed overhead). Does NOT
  // grow with the leaf-record volume --- the P1/P2 "no full-leaf cache"
  // property is checkable against this.
  std::uint64_t resident_bytes() const;

  // Number of enumerate_stripe() calls served so far (process-lifetime of this
  // reader). Thread-safe. Plan 11.14.5 P3 Teil 2 uses it to assert that a
  // band-parallel reduction still delegates local-warp geometry to the cache
  // at workers > 1 (byte-identity alone would not catch a silent fallback to
  // full re-enumeration, which is also correct but defeats P1/P2).
  std::uint64_t enumerate_call_count() const;

  // Largest per-source-row leaf-record count across every materialised
  // (variant, frame). `enumerate_stripe` reads at most this many records
  // (* sizeof(LeafRecord)) into one reused buffer, so it bounds the
  // per-concurrent-caller working set of a band-parallel reduction
  // (plan 11.14.5 P3 Teil 2 "shared budget" term).
  std::uint64_t max_row_record_count() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Publishes `reader` as the geometry source for the current thread for the
// lifetime of the guard. While one is active, `enumerate_drizzle_stripe_leaf_cells`
// serves LOCAL-WARP frames that the reader has (matching pixfrac +
// source_index) from the cache instead of re-running sample_leaves; affine
// frames and cache-misses fall through to the legacy enumerator unchanged.
// `nullptr` is allowed and disables delegation (identical to no guard).
class ScopedActiveGeometryCache {
public:
  explicit ScopedActiveGeometryCache(const DrizzleGeometryCacheReader *reader);
  ~ScopedActiveGeometryCache();
  ScopedActiveGeometryCache(const ScopedActiveGeometryCache &) = delete;
  ScopedActiveGeometryCache &operator=(const ScopedActiveGeometryCache &) = delete;

private:
  const DrizzleGeometryCacheReader *prev_;
};

// The current thread's active geometry cache reader, or nullptr.
const DrizzleGeometryCacheReader *active_geometry_cache();

} // namespace tile_compile::reconstruction
