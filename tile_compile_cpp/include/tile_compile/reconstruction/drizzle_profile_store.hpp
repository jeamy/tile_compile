#pragma once

#include "tile_compile/reconstruction/adaptive_alpha.hpp"
#include "tile_compile/reconstruction/alpha_confidence.hpp"
#include "tile_compile/reconstruction/alpha_guard.hpp"
#include "tile_compile/reconstruction/forward_drizzle.hpp"
#include "tile_compile/reconstruction/forward_drizzle_cuda.hpp"

namespace tile_compile::reconstruction {

// Caller-derived expectations, never inferred from the store being verified.
struct DrizzleStorePredecessors {
  std::string normalized_cache_hash, quality_plan_hash;
  // M5: canonical hash of the source composite Q-map cache when Raw consumes
  // Q_composite. Empty => no Q-maps applied; then it does not enter the
  // reconstruction hash (older stores stay comparable).
  std::string source_quality_cache_hash;
};

// M6: the full multiband contract that enters `multiband_config_hash`
// (plan 16.4: profile hashes plus levels, alpha, energy, support, downsample
// and validation contract). `enabled == false` => a plain uniform_raw store,
// byte-identical to a pre-M6 store. The versioned a-trous constants
// (kAtrousDenMinFraction, kAtrousDecompositionVersion) are folded in by
// make_drizzle_store_identity itself, not carried here.
struct MultibandStoreContract {
  bool enabled = false;
  int levels = 3;  // [1,4]; medium plane emitted only for levels >= 2
  float fine_quality_exponent = 4.0f;
  float medium_quality_exponent = 2.0f;
  AdaptiveAlphaParams alpha{};
  EnergyGuardParams guard{};
  AlphaConfidenceParams confidence{};
};

struct DrizzleStoreIdentity {
  std::string source_identity_hash, sampling_plan_hash, reconstruction_hash;
  std::string normalized_cache_hash, quality_plan_hash;
  // uniform_unclipped | uniform_raw_clipped | uniform_raw_multiband_clipped
  std::string mode;
  int width = 0, height = 0;
  ColorMode color_mode = ColorMode::MONO;
  // 0 => not a multiband store. >0 => band count; plane_names() reproduces the
  // fine/(medium)/alpha_* plane set from this, so it must be on the identity,
  // not merely folded into reconstruction_hash.
  int multiband_levels = 0;
  bool operator==(const DrizzleStoreIdentity &) const = default;
};

DrizzleStoreIdentity make_drizzle_store_identity(
    const registration::RegistrationSamplingPlan &plan,
    const config::ReconstructionDrizzleConfig &cfg,
    const ForwardDrizzleSubdivisionParams &subdivision = {},
    const config::ReconstructionClippingConfig *clipping = nullptr,
    const std::vector<float> &g_eff = {},
    const DrizzleStorePredecessors &predecessors = {},
    const MultibandStoreContract &multiband = {});

// Plan 19.4 / 19.6: telemetry for a multiband store built via the CUDA
// per-stripe path (accumulate_pair_by_frame_cuda driven by run_cuda_chunked).
// `used` stays false for the CPU reference path; the timings are wall clock.
struct DrizzleCudaStoreTiming {
  bool used = false;
  int bands = 0;                       // bands committed by run_cuda_chunked
  int resolved_chunk_rows = 0;         // initial band height from plan_cuda_chunking
  int min_chunk_rows = 0;
  std::size_t bytes_per_row = 0;       // device+host working-set estimate used
  std::size_t device_free_bytes = 0;   // free VRAM at plan time
  // §30.80 (P6 priority-3 groundwork): the three-way split of bytes_per_row.
  // `cand_row_bytes` is the HOST flat ClipCandidate buffer per internal row
  // (channels * dims.width * frame_count * sizeof(ClipCandidate)); it is
  // charged against device VRAM today even though it lives in host RAM and is
  // separately capped by `host_budget_bytes`. At real geometry it is ~99 % of
  // bytes_per_row -> the band height is set by a host term against VRAM.
  std::size_t cand_row_bytes = 0;
  std::size_t rec_row_bytes = 0;       // device per-frame contribution vector / row
  std::size_t acc_row_bytes = 0;       // 8 double stripe accumulators / channel / row
  std::size_t host_budget_bytes = 0;   // absolute host ClipCandidate ceiling used
  // Observed band structure from run_cuda_chunked (§30.80): halvings counts
  // CudaAllocFailure catches (band-collapse indicator); the min/max band rows
  // are the actual processed heights.
  int band_halvings = 0;
  int min_band_rows = 0;
  int max_band_rows = 0;
  // §30.81 (P6 priority-3 2D target tiling): each device band is additionally
  // split into column tiles so the HOST ClipCandidate buffer (cand_row * band
  // rows * tile_w / dims.width) stays under `host_budget_bytes`. `chunk_plan`
  // is now planned against the DEVICE term only (rec_row + acc_row). The tile
  // count is 1 and `resolved_tile_w == dims.width` when the whole band's host
  // buffer already fits. `min_tile_w` is the narrowest tile any band used.
  int resolved_tile_w = 0;   // tile width of the first band's tiling
  int min_tile_w = 0;
  int max_tiles_per_band = 0;
  double stripe_seconds = 0.0;         // sum of time inside accumulate_pair_by_frame_cuda
  double total_seconds = 0.0;          // whole chunked drive incl. sink / store I/O
  // plan 19.6.2: how many frames took the hybrid CPU-geometry -> GPU-raster
  // path (local-warp frames). > 0 => the committed store is labelled
  // "cuda_hybrid" rather than "cuda". Affine-only CUDA runs leave this 0.
  int hybrid_local_frames = 0;
  // plan 19.6.2 wall-clock split of the hybrid path, summed over all local-warp
  // frames of all bands (0 on an affine-only CUDA run). `hybrid_cpu_seconds` is
  // the CPU leaf geometry + marshalling + record assembly; the raster figure is
  // time inside the device polygon-area kernel (transfer + kernel, lumped).
  double hybrid_cpu_seconds = 0.0;
  double hybrid_gpu_raster_seconds = 0.0;
  long long hybrid_leaf_cells = 0;
};

struct DrizzleStoreResult {
  fs::path generation_dir;
  ForwardDrizzleDiagnostics diagnostics;
  ForwardDrizzleClippingDiagnostics clipping;
  DrizzleCudaStoreTiming cuda_timing;
  // The identity actually written (populated by persist_* entry points). A
  // consumer that reads the store back should use THIS, never a re-derived
  // one, so a write/read identity divergence fails at write time.
  DrizzleStoreIdentity identity;
};

// Immutable generation directories; current.json is the sole commit point.
// An interrupted writer preserves the previous commit. Old generations are
// retained for readers; automatic deletion/garbage collection is not performed.
DrizzleStoreResult persist_forward_drizzle_uniform(
    const fs::path &root, const registration::RegistrationSamplingPlan &plan,
    const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg,
    const ForwardDrizzleSubdivisionParams &subdivision = {});
DrizzleStoreResult persist_forward_drizzle_uniform_and_raw(
    const fs::path &root, const registration::RegistrationSamplingPlan &plan,
    const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg,
    const config::ReconstructionClippingConfig &clipping,
    const ForwardDrizzleSubdivisionParams &subdivision = {},
    const std::vector<float> &g_eff = {},
    const DrizzleStorePredecessors &predecessors = {},
    const FrameQualityProvider &quality_of = {},
    // Plan 11.14.5 P3 Teil 2: output-row-band workers for the CPU streaming
    // reduction. Forwarded verbatim to stream_forward_drizzle_uniform_and_raw();
    // bit-identical (store commit hash included) to `workers == 1` (default).
    int workers = 1);

// M6: uniform + raw + fine + (medium, when levels >= 2) profile planes plus
// the four channel-min alpha-confidence maps (alpha_separation / alpha_artifact
// / alpha_registration / alpha_support), each a single pseudo-channel "X"
// plane. Requires a quality provider that supplies composite + artifact (and
// scale0/scale1 for levels >= 1/2). Output scale 2/1 is rejected --- the
// channel-min confidence maps have no defined 2x2 area-average
// (2x2-mean(min_c) != min_c(2x2-mean)); use 1/1 or 2/2.
// `cuda.attempt` requests the plan-19 CUDA path. Slice 1 has no kernels: with
// `attempt` set it throws ForwardDrizzleCudaError (immediately, or --- for the
// fault-injection test hook --- after N committed stripes) so the caller can
// exercise the plan-19.4 full-phase CPU restart. The thrown-from generation
// directory is never committed (StoreWriter discards it).
DrizzleStoreResult persist_forward_drizzle_multiband(
    const fs::path &root, const registration::RegistrationSamplingPlan &plan,
    const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg,
    const config::ReconstructionClippingConfig &clipping,
    const MultibandStoreContract &multiband,
    // §30.81 step 3a-2: rect provider. The CUDA stripe path decodes Q maps for
    // only the source rectangle each column tile's records touch; the CPU
    // streaming sub-path adapts it back to full via a plain wrapper.
    const FrameQualityRectProvider &quality_of,
    const ForwardDrizzleSubdivisionParams &subdivision = {},
    const std::vector<float> &g_eff = {},
    const DrizzleStorePredecessors &predecessors = {},
    const ForwardDrizzleCudaOptions &cuda = {},
    // Plan 11.14.5 P3 Teil 2: output-row-band workers for the CPU reference
    // streaming reduction only. The CUDA stripe path (cuda.attempt with a
    // usable device) has its own device-band chunking and ignores this; on a
    // CUDA->CPU restart the CPU path picks it up. Bit-identical (store commit
    // hash included) to `workers == 1` (default).
    int workers = 1);

// Convenience overload for callers that only have a full-source-geometry
// FrameQualityProvider (tests, non-cache paths): adapts it via
// to_rect_provider (the CUDA path then decodes full maps, no rectangle win).
inline DrizzleStoreResult persist_forward_drizzle_multiband(
    const fs::path &root, const registration::RegistrationSamplingPlan &plan,
    const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg,
    const config::ReconstructionClippingConfig &clipping,
    const MultibandStoreContract &multiband,
    const FrameQualityProvider &quality_of,
    const ForwardDrizzleSubdivisionParams &subdivision = {},
    const std::vector<float> &g_eff = {},
    const DrizzleStorePredecessors &predecessors = {},
    const ForwardDrizzleCudaOptions &cuda = {}, int workers = 1) {
  return persist_forward_drizzle_multiband(
      root, plan, source_of, cfg, clipping, multiband,
      to_rect_provider(quality_of), subdivision, g_eff, predecessors, cuda,
      workers);
}

struct DrizzleStoreValidation {
  bool usable = false;
  fs::path generation_dir;
  std::string error;
};
// Checks exact plane set, context, hashes, FITS shape/type/row order. This
// establishes store integrity, not availability of pipeline predecessors.
DrizzleStoreValidation verify_drizzle_profile_store(
    const fs::path &root, const DrizzleStoreIdentity &expected);

// Verifies one immutable generation (rehashes it), then reads a bounded native
// store ROI. The budget covers the returned plane and IO scratch, in MiB.
ProfilePlane read_drizzle_profile_region(
    const fs::path &root, const DrizzleStoreIdentity &expected,
    const std::string &profile, const std::string &channel,
    int x, int y, int width, int height, size_t memory_budget_mb = 64);

// Same read, against an ALREADY-VERIFIED generation directory (from
// verify_drizzle_profile_store) --- skips the per-call rehash. For loops that
// pull many regions from one store (e.g. striped fusion): verify once, then
// read every stripe with this. The caller owns having verified the directory
// against a trusted identity.
ProfilePlane read_drizzle_profile_region_preverified(
    const fs::path &generation_dir, const DrizzleStoreIdentity &expected,
    const std::string &profile, const std::string &channel,
    int x, int y, int width, int height, size_t memory_budget_mb = 64);

} // namespace tile_compile::reconstruction
