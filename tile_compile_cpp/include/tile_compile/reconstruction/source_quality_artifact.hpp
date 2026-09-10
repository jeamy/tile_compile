#pragma once
#include "tile_compile/reconstruction/normalized_source_cache.hpp"
#include "tile_compile/reconstruction/quality_frame_weight_plan.hpp"
#include "tile_compile/reconstruction/drizzle_profile_store.hpp"

#include <array>
#include <cstdint>

namespace tile_compile::reconstruction {

// Returns source-index-addressed weights after checking identities, factors and
// the complete frame set. Output size is bounded before allocation.
std::vector<float> resolve_quality_frame_weights(
    const QualityFrameWeightPlan &quality,
    const registration::RegistrationSamplingPlan &sampling,
    const GlobalQualityConfig &cfg, size_t memory_budget_mb = 512);

QualityFrameWeightPlan persist_source_quality_artifact(
    const fs::path &path, const registration::RegistrationSamplingPlan &sampling,
    VerifiedNormalizedSourceCache &cache, const GlobalQualityConfig &cfg,
    size_t memory_budget_mb = 512, int workers = 1);
QualityFrameWeightPlan load_source_quality_artifact(
    const fs::path &path, const registration::RegistrationSamplingPlan &sampling,
    const VerifiedNormalizedSourceCache &cache, const GlobalQualityConfig &cfg,
    size_t memory_budget_mb = 512);

// Library orchestration with mandatory predecessor checks. Existing runner
// phases are not resumed or bypassed by this entry point.
DrizzleStoreResult persist_forward_drizzle_from_predecessors(
    const fs::path &store_root, const fs::path &quality_artifact,
    const registration::RegistrationSamplingPlan &sampling,
    VerifiedNormalizedSourceCache &cache, const GlobalQualityConfig &quality_cfg,
    const config::ReconstructionDrizzleConfig &drizzle_cfg,
    const config::ReconstructionClippingConfig &clipping_cfg,
    const ForwardDrizzleSubdivisionParams &subdivision = {},
    // M5: when set, Raw consumes the source composite Q-maps from this cache
    // root as Q_composite_f,c(q). Empty => Q_composite = 1.0 (Raw unchanged).
    const fs::path &source_quality_cache_root = {},
    // Plan 11.14.5 P3 Teil 2: CPU-reduction output-row-band workers. Forwarded
    // to the streaming reduction; bit-identical to `workers == 1` (default).
    int workers = 1);

// Maps the public multiband config onto the store's full hashed contract.
// Fields not yet in config (energy guard, most confidence edges) take their
// documented defaults but still enter multiband_config_hash.
MultibandStoreContract multiband_store_contract_from_config(
    const config::ReconstructionMultibandConfig &cfg);

struct MultibandStoreBuildResult {
  DrizzleStoreResult store;
  DrizzleStoreIdentity identity;
  // Plan 19: which path actually produced the committed store. "cuda" when a
  // CUDA attempt ran to completion with affine-only frames; "cuda_hybrid" when
  // that run also took at least one local-warp frame through the plan-19.6.2
  // hybrid CPU-geometry -> GPU-rasterization path; "cpu" for the plain path AND
  // for a CUDA attempt that failed and was restarted on the CPU reference path.
  std::string backend_used = "cpu";
  // Non-empty iff a CUDA attempt was made and did not commit: the reason the
  // phase fell back to the CPU reference path (plan 19.4).
  std::string cuda_fallback_reason;
  // §30.81 step 3a-2/3a-2b: SourceQualityMapCacheReader I/O totals for this
  // build --- lets a run report whether the Q read is actually bounded.
  //   q_bin_loads         .bin files opened (one per read_rect call)
  //   q_bin_cells_decoded  storage-grid cells read (3a-2b: only the covering
  //                        cells; before it, storage_w*storage_h per call)
  //   q_expanded_floats    map elements materialised
  std::uint64_t q_bin_loads = 0;
  std::uint64_t q_bin_cells_decoded = 0;
  std::uint64_t q_expanded_floats = 0;
};

// M6 phase 1: build the multiband profile store (uniform+raw+fine+(medium)+
// the four alpha-confidence maps) from the M5 predecessors --- the Q-map cache
// supplies composite + scale0/scale1 + artifact. This is the durable artefact.
// Requires `source_quality_cache_root`; output scale 2/1 is rejected.
//
// `acceleration_backend` is the resolved backend name ("cpu" | "cuda"). "cuda"
// attempts the plan-19 CUDA path and, on ForwardDrizzleCudaError, discards the
// uncommitted generation and restarts the ENTIRE build on the CPU reference
// path (plan 19.4) --- the committed store is then bit-identical to a "cpu"
// build. The retry is not recursive: a CPU restart that itself fails throws.
MultibandStoreBuildResult persist_multiband_store_from_predecessors(
    const fs::path &store_root, const fs::path &quality_artifact,
    const registration::RegistrationSamplingPlan &sampling,
    VerifiedNormalizedSourceCache &cache, const GlobalQualityConfig &quality_cfg,
    const config::ReconstructionDrizzleConfig &drizzle_cfg,
    const config::ReconstructionClippingConfig &clipping_cfg,
    const config::ReconstructionMultibandConfig &multiband_cfg,
    const fs::path &source_quality_cache_root,
    const ForwardDrizzleSubdivisionParams &subdivision = {},
    const std::string &acceleration_backend = "cpu",
    // Plan 11.14.5 P3 Teil 2: CPU-reference-path output-row-band workers.
    // Ignored on the CUDA stripe path (own device band chunking); picked up on
    // a CUDA->CPU restart. Bit-identical (store commit hash) to 1 (default).
    int workers = 1);

// The three plan-15 candidate images, each reduced to the fixed working
// luminance (`kWorkingLumaDefinition`), assembled during the single fusion
// pass so the store is read only once. `alpha_final_by_band[j]` is the shared
// alpha actually applied to fused band j (empty inner vector = Raw-sourced /
// inactive band), row-major width*height --- fed to `prepare_validation_samples`
// for the per-star `multiband_effective` flag.
struct MultibandCandidateLuma {
  int width = 0;
  int height = 0;
  std::vector<float> uniform_luma, raw_luma, multiband_luma;  // NaN off support
  std::vector<uint8_t> uniform_support;  // luma support (every active channel)
  std::vector<std::vector<float>> alpha_final_by_band;
};

// Plan 11.13(2): the three plan-15 candidate images at FULL per-channel
// resolution are streamed stripe-wise into a caller-owned scratch directory
// instead of held whole in RAM. Peak resident candidate data is then one fusion
// stripe (nch * chunk * W floats) during the fuse pass and one plane during
// delivery read-back --- not 3*nch whole planes. `nch` is 1 (MONO -> "L") or 3
// (OSC -> R,G,B). Each spooled plane is host-native `float`, row-major
// width*height, NaN off support. The runner reads back only the delivered
// candidates (Raw baseline + selected + optionally the `full`-diagnostics
// controls), one plane at a time, straight to FITS.
struct MultibandCandidateSpool {
  fs::path dir;  // caller-created scratch directory; must exist before the call
  int width = 0;
  int height = 0;
  int nch = 1;
  bool mono = true;
  bool populated = false;
  // "<candidate>_<c>.f32" under `dir`; candidate in {uniform,raw,multiband}.
  fs::path plane_path(const std::string &candidate, int c) const;
};

// Read one spooled candidate plane back (width*height host-native floats).
// Throws std::runtime_error on a shape / size / I/O mismatch. The caller writes
// it straight to FITS and frees it, so peak delivery memory is one plane.
std::vector<float> read_candidate_spool_plane(const MultibandCandidateSpool &spool,
                                              const std::string &candidate, int c);

// Plan 11.13(1)+(4): the shared, overflow-safe host working-set estimate for the
// MULTIBAND fuse / validation / export phase, computed BEFORE any large
// allocation. Every term is an upper bound on data that is concurrently
// resident. `fuse_multiband_store_to_image` fails closed
// (throws "MULTIBAND_MEMORY_BUDGET") when `estimated_peak_bytes > budget_bytes`,
// before it touches the final image, the candidate buffers or the spool ---
// nothing already committed is disturbed.
struct MultibandFusionMemoryPlan {
  std::size_t final_image_bytes = 0;       // nch * N * 4  (held whole)
  std::size_t stripe_working_bytes = 0;    // U/R/F/M + fuse + alpha + luma scratch
  std::size_t candidate_luma_bytes = 0;    // with_candidate_luma: 3 luma + support + alpha
  std::size_t spool_stripe_bytes = 0;      // with_candidate_spool: 3 * nch * chunk * W * 4
  std::size_t delivery_readback_bytes = 0; // one plane read back for FITS export
  std::size_t margin_bytes = 0;            // max(64 MiB, 5% of the raw sum)
  std::size_t estimated_peak_bytes = 0;    // RAM sum of the above
  std::size_t budget_bytes = 0;
  bool fits = false;                       // estimated_peak_bytes <= budget_bytes
  // Plan 11.13(2) + 11.11: bytes the candidate spool writes to the temp
  // filesystem (3 candidates * nch * N * 4), and the free space that must be
  // available for it: estimated_temp_peak * 1.20 + max(2 GiB, 5% of capacity).
  // `available_temp_bytes` / `temp_space_ok` are filled by
  // fuse_multiband_store_to_image once the spool directory is known (0 / true
  // when no spool is requested).
  std::size_t spool_temp_bytes = 0;
  std::size_t required_free_temp_bytes = 0;
  std::size_t available_temp_bytes = 0;
  bool temp_space_ok = true;
};
MultibandFusionMemoryPlan plan_multiband_fusion_memory(
    int width, int height, int nch, int levels, int chunk_rows, int halo_rows,
    bool with_candidate_luma, bool with_candidate_spool, std::size_t budget_bytes);

// M6 phase 2: fuse the durable multiband store (plan 14, streamed path) into a
// single final X_out image at `final_image_path` (MONO -> float FITS, OSC ->
// RGB FITS). `chunk_rows <= 0` uses a default. Returns supported-pixel count.
// When `candidates_out` is non-null it is also filled with the plan-15
// uniform / raw / multiband working-luminance candidates + the fused per-band
// alpha maps, at no extra store I/O.
//
// `memory_budget_mb == 0` means "unset" --- an internal floor is applied. A
// non-zero value is an EXPLICIT budget and is used verbatim (plan 11.13: no
// silent raising of an explicit budget); the phase fails closed if the
// pre-planned working set does not fit it.
long long fuse_multiband_store_to_image(
    const fs::path &store_root, const DrizzleStoreIdentity &identity,
    const fs::path &final_image_path,
    const config::ReconstructionMultibandConfig &multiband_cfg,
    int chunk_rows = 0, size_t memory_budget_mb = 512,
    MultibandCandidateLuma *candidates_out = nullptr,
    // When non-null, streams the three candidates at full per-channel resolution
    // to `spool_out->dir` (which must already exist). `final_image_path` still
    // receives the multiband X_out as before.
    MultibandCandidateSpool *spool_out = nullptr,
    // When non-null, receives the pre-allocation working-set plan that gated the
    // phase (plan 11.13(4): reported separately from the measured RSS peak).
    MultibandFusionMemoryPlan *mem_plan_out = nullptr);

} // namespace tile_compile::reconstruction
