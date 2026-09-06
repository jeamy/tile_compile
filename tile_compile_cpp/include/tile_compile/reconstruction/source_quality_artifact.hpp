#pragma once
#include "tile_compile/reconstruction/normalized_source_cache.hpp"
#include "tile_compile/reconstruction/quality_frame_weight_plan.hpp"
#include "tile_compile/reconstruction/drizzle_profile_store.hpp"

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
    size_t memory_budget_mb = 512);
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
    const fs::path &source_quality_cache_root = {});

// Maps the public multiband config onto the store's full hashed contract.
// Fields not yet in config (energy guard, most confidence edges) take their
// documented defaults but still enter multiband_config_hash.
MultibandStoreContract multiband_store_contract_from_config(
    const config::ReconstructionMultibandConfig &cfg);

struct MultibandStoreBuildResult {
  DrizzleStoreResult store;
  DrizzleStoreIdentity identity;
  // Plan 19: which path actually produced the committed store. "cuda" only
  // when a CUDA attempt ran to completion; "cpu" for the plain path AND for a
  // CUDA attempt that failed and was restarted on the CPU reference path.
  std::string backend_used = "cpu";
  // Non-empty iff a CUDA attempt was made and did not commit: the reason the
  // phase fell back to the CPU reference path (plan 19.4).
  std::string cuda_fallback_reason;
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
    const std::string &acceleration_backend = "cpu");

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

// M6 phase 2: fuse the durable multiband store (plan 14, streamed path) into a
// single final X_out image at `final_image_path` (MONO -> float FITS, OSC ->
// RGB FITS). `chunk_rows <= 0` uses a default. Returns supported-pixel count.
// When `candidates_out` is non-null it is also filled with the plan-15
// uniform / raw / multiband working-luminance candidates + the fused per-band
// alpha maps, at no extra store I/O.
long long fuse_multiband_store_to_image(
    const fs::path &store_root, const DrizzleStoreIdentity &identity,
    const fs::path &final_image_path,
    const config::ReconstructionMultibandConfig &multiband_cfg,
    int chunk_rows = 0, size_t memory_budget_mb = 512,
    MultibandCandidateLuma *candidates_out = nullptr);

} // namespace tile_compile::reconstruction
