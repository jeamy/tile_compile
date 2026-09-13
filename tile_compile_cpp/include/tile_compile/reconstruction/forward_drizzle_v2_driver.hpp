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
// for each stream position [0, plan.frame_count) it supplies the canvas
// affine6, the optional local warp, the normalized source plane, the
// optional sigma2 plane, the quality streams and the frame meta row.
// Pointers must stay valid until the next provider call; all frames must
// share plan-bound source dims. Returning false aborts the phase.

#include "tile_compile/reconstruction/forward_drizzle_v2_cpu.hpp"
#include "tile_compile/reconstruction/forward_drizzle_v2_store.hpp"

#include <functional>
#include <string>

namespace tile_compile::reconstruction {

struct ForwardDrizzleV2FrameInput {
  // Persisted source->canvas transform in CANVAS coordinates (the band
  // origin lives in the kernel config, not here).
  double affine6[6] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  bool has_local_model = false;
  ForwardDrizzleV2LocalWarp warp{};
  const float *source = nullptr;   // source_w*source_h, required
  const float *sigma2 = nullptr;   // optional (absent = degraded stream)
  ForwardDrizzleV2FrameQuality quality{};   // all-null = absent streams
  ForwardDrizzleV2FrameMeta meta{};
};

using ForwardDrizzleV2FrameProvider =
    std::function<bool(std::uint64_t frame_order,
                       ForwardDrizzleV2FrameInput &out)>;

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
