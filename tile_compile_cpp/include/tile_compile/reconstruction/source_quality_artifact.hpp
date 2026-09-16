#pragma once
#include "tile_compile/reconstruction/normalized_source_cache.hpp"
#include "tile_compile/reconstruction/quality_frame_weight_plan.hpp"
#include "tile_compile/reconstruction/drizzle_profile_store.hpp"
#include "tile_compile/reconstruction/forward_drizzle_v2_store.hpp"

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

// T3: when `metrics_path` is non-empty and the file exists, load the
// pre-computed per-frame metrics from SOURCE_QUALITY_MAPS instead of
// reloading and re-running compute_source_quality_proxy_v1 per frame.
// The resulting QualityFrameWeightPlan is bit-identical to the recompute path.
QualityFrameWeightPlan persist_source_quality_artifact(
    const fs::path &path, const registration::RegistrationSamplingPlan &sampling,
    VerifiedNormalizedSourceCache &cache, const GlobalQualityConfig &cfg,
    const fs::path &metrics_path,
    size_t memory_budget_mb = 512, int workers = 1);

QualityFrameWeightPlan load_source_quality_artifact(
    const fs::path &path, const registration::RegistrationSamplingPlan &sampling,
    const VerifiedNormalizedSourceCache &cache, const GlobalQualityConfig &cfg,
    size_t memory_budget_mb = 512);

// Maps the public multiband config onto the store's full hashed contract.
// Fields not yet in config (energy guard, most confidence edges) take their
// documented defaults but still enter multiband_config_hash.
MultibandStoreContract multiband_store_contract_from_config(
    const config::ReconstructionMultibandConfig &cfg);

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

// Fuse the durable multiband v2 store into a single final X_out image at
// `final_image_path` (MONO -> float FITS, OSC -> RGB FITS). The v2 store's
// committed band profile records are read with a rolling row window (every
// band decoded once), each fusion stripe is adapted through
// forward_drizzle_v2_profiles_to_uniform_result +
// forward_drizzle_v2_profile_alpha_plane (channel-min alpha) and fused with
// the shared fuse_multiband. `plan` binds the store; inspection must report
// status complete or the call throws (fail closed). Same output products,
// budget pre-plan and fail-closed semantics as the legacy variant.
struct ForwardDrizzleV2FusionStats {
  std::uint64_t bands_decoded = 0;         // distinct bands read from store
  std::uint64_t record_bytes_read = 0;     // decoded profile bytes
  std::uint64_t record_bytes_no_reuse = 0; // window bytes if re-read per stripe
  double read_amplification = 1.0;         // read / logical store bytes
  ForwardDrizzleV2AlphaDiagnostic alpha{};
};
long long fuse_multiband_v2_store_to_image(
    const fs::path &store_root, const ForwardDrizzleV2RunPlan &plan,
    const fs::path &final_image_path,
    const config::ReconstructionMultibandConfig &multiband_cfg,
    int chunk_rows = 0, size_t memory_budget_mb = 512,
    MultibandCandidateLuma *candidates_out = nullptr,
    MultibandCandidateSpool *spool_out = nullptr,
    MultibandFusionMemoryPlan *mem_plan_out = nullptr,
    ForwardDrizzleV2FusionStats *stats_out = nullptr);

} // namespace tile_compile::reconstruction
