#pragma once

// Forward-Drizzle-v2 Gate-10: band driver. Connects the Gate-7 transactional
// store to the shared kernel interface (CUDA preferred, host CPU fallback)
// behind one entry point:
//
//   inspect -> adopt(resumable)/begin(fresh) -> per band:
//   kernel.reserve(band) -> accumulate every frame -> kernel.finalize ->
//   commit_band(pixels, profiles) -> finish(commit gate) -> current.json
//
// Backend contract (spec wiring_contract/fallback): CUDA is attempted when
// `prefer_cuda` and a runtime device exists. Any device-side failure ---
// a kernel call reporting false, or ForwardDrizzleCudaError --- discards the
// unpublished generation (including any adopted committed prefix) and
// restarts the WHOLE phase on the host CPU port. The retry is not
// recursive: a CPU failure propagates.
//
// The provider contract keeps the driver free of cache/quality wiring:
// for each (band, stream position [0, plan.frame_count)) it supplies the
// canvas affine6, the optional local warp, the normalized source window
// (packed, absolute coordinates --- or `skip` for an empty window), the
// optional sigma2 plane, the quality streams and the frame meta row.
// Pointers must stay valid until the next provider call; windows must be
// contained in the plan-bound source dims. Returning false aborts the phase.

#include "tile_compile/reconstruction/forward_drizzle_v2_cpu.hpp"
#include "tile_compile/reconstruction/forward_drizzle_v2_store.hpp"

#include <cstddef>
#include <functional>
#include <string>

namespace tile_compile::reconstruction {

struct ForwardDrizzleV2FrameInput {
  // Persisted source->canvas transform in CANVAS coordinates (the band
  // origin lives in the kernel config, not here).
  double affine6[6] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  bool has_local_model = false;
  ForwardDrizzleV2LocalWarp warp{};
  const float *source = nullptr;   // packed source_window.width*height
  // Explicit sigma2 plane packed over the ACTIVE window rect (compatibility
  // path) OR the inline model computed from the (halo-extended) source
  // buffer --- both may be null/disabled (degraded stream), never both set.
  const float *sigma2 = nullptr;
  ForwardDrizzleV2Sigma2FrameModel sigma2_model{};
  // Packed source window in absolute source coordinates; default = full
  // source. Ignored when `skip` is set.
  ForwardDrizzleV2SourceWindow source_window{};
  // True when the frame cannot reach this band at all: the driver advances
  // the stream bookkeeping without any scatter/fold.
  bool skip = false;
  // Geometry-cache path (tranche 6): when has_cached_geometry is set the
  // kernel scatters `cached_leaves` (raw internal canvas coordinates,
  // canonical order) instead of running the local inversion; `warp` is then
  // unused. The leaf source coordinates must lie inside the active window
  // rect.
  bool has_cached_geometry = false;
  const ForwardDrizzleV2CachedLeaf *cached_leaves = nullptr;
  std::size_t cached_leaf_count = 0;
  std::uint64_t cached_unique_source_samples = 0;
  // Affine target-tile range (tranche 7): the piece scatters/folds only
  // native target columns [target_x_begin_native, +target_cols_native) of
  // the band. target_cols_native == 0 is the full-width compatibility
  // default. Ignored on skip and on the single-piece local/cached paths.
  int target_x_begin_native = 0;
  int target_cols_native = 0;
  // Tranche 8 canonical ragged affine path: one-shot full-target piece fed
  // by the active source-sample list (canonical ascending (y,x) order) with
  // aligned packed quality. `source_window`/`source`/`sigma2`/
  // `sigma2_model`/`quality`/`target_*` stay unused on this path;
  // `sigma2_present` selects whether each sample's sigma2 field applies.
  bool has_affine_samples = false;
  const ForwardDrizzleV2SourceSample *affine_samples = nullptr;
  std::size_t affine_sample_count = 0;
  bool sigma2_present = false;
  ForwardDrizzleV2AlignedQuality aligned_quality{};
  ForwardDrizzleV2FrameQuality quality{};   // all-null = absent streams
  ForwardDrizzleV2FrameMeta meta{};
};

// Piece sink (tranche 7): the provider invokes it once with a skip piece or
// one-or-more non-skip pieces. All non-skip pieces of a frame must share
// the same affine6/meta/geometry mode; affine pieces must be ordered by
// target x and non-overlapping; cached/local frames emit exactly one
// full-width piece. Returning false aborts the provider (and the phase).
using ForwardDrizzleV2FramePieceSink =
    std::function<bool(const ForwardDrizzleV2FrameInput &piece)>;
using ForwardDrizzleV2FrameProvider =
    std::function<bool(int band_y_begin, int band_rows,
                       std::uint64_t frame_order,
                       const ForwardDrizzleV2FramePieceSink &sink)>;

// Test-only: thrown by run_forward_drizzle_v2 when the driver option
// simulate_kill_after_bands triggers. Mimics SIGKILL mid-phase: the writer
// is intentionally leaked so the unpublished generation (with its committed
// prefix) survives on disk, exactly like a killed process leaves it.
struct ForwardDrizzleV2SimulatedKill : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct ForwardDrizzleV2DriverOptions {
  bool prefer_cuda = true;
  std::function<void(int band_index, int band_count)> progress;
  // Test-only: after this many band commits of the current attempt, throw
  // ForwardDrizzleV2SimulatedKill and leak the writer (crash simulation).
  // < 0 disables (default).
  int simulate_kill_after_bands = -1;
  // Reserved leaf-record capacity for the geometry-cache scatter path,
  // copied into every band's kernel config (fixed across begin_band). Not
  // part of the run plan/hash. 0 disables the cached-leaf path.
  std::uint64_t cached_leaf_capacity = 0;
};

struct ForwardDrizzleV2DriverResult {
  fs::path generation_dir;
  bool committed = false;         // finish() published current.json
  int bands_total = 0;
  int bands_committed = 0;        // committed during THIS invocation
  int bands_reused = 0;           // adopted committed prefix
  std::string backend_used;       // "cuda_v2" | "cpu_v2" | "reused"
  std::string cuda_fallback_reason;
  ForwardDrizzleV2PrototypeStats totals{};   // summed over computed bands
  std::uint64_t local_samples_discarded = 0;
  std::string commit_hash;        // commit.json hash (complete only)
  // Spec telemetry gate: the slowest single (frame, band) device event time
  // reported by the kernel (max of kernel->stats().max_frame_seconds over
  // computed bands).
  double max_frame_seconds = 0.0;
  // Slowest single provider + enqueue call, wall seconds. Computed bands
  // only; not device event timing.
  double max_provider_enqueue_seconds = 0.0;
  // Wall seconds around the complete backend attempt(s) of this call
  // (provider + kernel + finalize + store writes for computed bands).
  double phase_wall_seconds = 0.0;
  // Wall seconds inside commit_band plus finish (the store-side commit
  // operations only, summed over the computed bands).
  double commit_seconds = 0.0;
  // Reusable per-band record buffers are pre-reserved outside the band
  // loop; any capacity growth inside a band lands here (must stay 0).
  std::uint64_t driver_hotpath_allocations = 0;
};

// Runs (or resumes/completes) the banded v2 production under `store_root`.
// `plan` must be finalized (plan_hash set); x_tiled plans are rejected ---
// the driver covers full-width bands only. `prefer_cuda` selects the device
// backend first; a device failure restarts the phase on CPU per the spec.
// `options.progress` fires before each computed band.
// Throws std::runtime_error on provider failure, contract violations, a
// corrupt store, or a CPU-backend failure.
ForwardDrizzleV2DriverResult run_forward_drizzle_v2(
    const fs::path &store_root, const ForwardDrizzleV2RunPlan &plan,
    int source_width, int source_height,
    const ForwardDrizzleV2FrameProvider &provider,
    const ForwardDrizzleV2DriverOptions &options = {});

// Test-only fault injection, mirroring
// set_forward_drizzle_cuda_fault_after_chunks: n >= 0 makes the CUDA attempt
// report a device failure before committing band n (0 = immediate). -1
// (default) disables. Process-global; the environment variable
// TILE_COMPILE_FD_V2_CUDA_FAULT_AFTER_BANDS arms it at first read.
void set_forward_drizzle_v2_cuda_fault_after_bands(int n);
int forward_drizzle_v2_cuda_fault_after_bands();

}  // namespace tile_compile::reconstruction
