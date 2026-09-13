#pragma once

// Forward-Drizzle-v2 Gate-10 production wiring. Connects the pipeline's
// predecessor artifacts to the Gate-10 band driver:
//
//   VerifiedNormalizedSourceCache      -> normalized source planes
//   SourceQualityMapCacheReader        -> q_composite/q_scale0/q_scale1/
//                                         q_artifact source planes
//   source_quality_metrics-v1.json     -> per-frame sigma_noise
//   global_registration.json           -> per-frame sigma_reg (star_residuals
//                                         rms_px)
//   QualityFrameWeightPlan             -> g_eff
//   RegistrationSamplingPlan           -> transforms, is_direct,
//                                         residual_factor, local models
//   DrizzleGeometryCacheReader         -> local-warp frame exclusion stats
//
// The run plan binds every predecessor identity; a config or predecessor
// change yields a different plan_hash and the store fails closed.

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/reconstruction/drizzle_geometry_cache.hpp"
#include "tile_compile/reconstruction/forward_drizzle.hpp"
#include "tile_compile/reconstruction/forward_drizzle_v2_driver.hpp"
#include "tile_compile/reconstruction/forward_drizzle_v2_store.hpp"
#include "tile_compile/reconstruction/normalized_source_cache.hpp"
#include "tile_compile/reconstruction/global_quality.hpp"
#include "tile_compile/reconstruction/source_quality_map_cache.hpp"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace tile_compile::reconstruction {

// Gate-10 sigma2 input plane (spec wiring_contract/inputs/sigma2):
//   sigma2(y,x) = sigma_noise^2 + (gx^2 + gy^2) * sigma_reg_px^2 + half^2/3
// with gx/gy the central differences of the normalized source (one-sided at
// borders; a difference with a non-finite neighbour is taken one-sided, and
// 0 when no finite neighbour exists). Non-finite/negative inputs produce
// non-finite plane values, which the kernel maps onto the degraded-sigma
// contract. Returns source.rows()*source.cols() row-major values.
std::vector<float> forward_drizzle_v2_sigma2_plane(
    const Matrix2Df &source, double sigma_noise, double sigma_reg_px,
    double droplet_half);

// Host-side per-native-pixel workspace of the CPU kernel, replicating the
// reserve() accounting (frame planes at internal resolution + per-channel
// accumulators + reservoir records + output records). Used to size the
// production band plan against the host budget.
std::size_t forward_drizzle_v2_cpu_bytes_per_native_pixel(
    int channels, int internal_scale, int reservoir_size, bool sigma2_plane,
    bool emit_profiles);

// Builds and finalizes the v2 run plan bound to the given predecessor
// identities. `frame_count` is the participating-frame stream length the
// provider will serve. Band geometry: `band_rows` core rows per band so one
// band's workspace stays within min(`host_budget_bytes`/2,
// `device_budget_bytes`); `band_count` covers [0, native_height). The
// budgets must be deterministic across resume (the plan_hash binds the band
// geometry) --- callers pass a NOMINAL device bound, not measured free VRAM;
// reserve() still verifies the real device per band and a device failure
// restarts the phase on CPU per the cutover contract. `halo_rows` records
// the multiband fusion halo for provenance/read accounting (the stored
// bands are contiguous core rows without padding). Throws invalid_argument
// on an unsupported output grid (2/2 or output_scale != 1) or a budget that
// cannot fit a single band row.
ForwardDrizzleV2RunPlan make_forward_drizzle_v2_run_plan(
    const registration::RegistrationSamplingPlan &sampling,
    const config::ReconstructionDrizzleConfig &drizzle_cfg,
    const config::ReconstructionClippingConfig &clipping_cfg,
    const config::ReconstructionMultibandConfig &multiband_cfg,
    bool emit_profiles, std::uint64_t frame_count,
    const std::string &normalized_cache_hash,
    const std::string &quality_plan_hash,
    const std::string &source_quality_cache_hash,
    const std::string &config_snapshot_hash,
    std::size_t host_budget_bytes,
    std::size_t device_budget_bytes = 0);

struct ForwardDrizzleV2ProductionResult {
  ForwardDrizzleV2RunPlan plan;
  ForwardDrizzleV2DriverResult driver;
  int frames_participating = 0;
  int frames_skipped_invalid = 0;
  // Frames excluded by the per-frame local-model inversion error rate
  // (same contract as the legacy prepare_drizzle_frames path).
  std::vector<std::pair<std::string, double>>
      frames_excluded_subdivision_error_rate;
  std::uint64_t local_model_samples_total = 0;     // incl. excluded frames
  std::uint64_t local_model_samples_discarded = 0; // incl. excluded frames
  // SourceQualityMapCacheReader I/O totals (section 30.81 accounting).
  std::uint64_t q_bin_loads = 0;
  std::uint64_t q_bin_cells_decoded = 0;
  std::uint64_t q_expanded_floats = 0;
};

// Runs (or resumes) the v2 FORWARD_DRIZZLE production under `store_root`
// using the same predecessor artifacts as the legacy producer. When
// `acceleration_backend` is "cuda" the driver attempts the device path first
// and restarts the whole phase on the host port after a device failure;
// `backend_used` in the result records which path committed.
//
// `geometry_cache` must be the published reader whenever the plan contains
// local-model frames; a local frame without cache stats is re-checked by a
// one-shot exclusion sweep (same contract as prepare_drizzle_frames).
// `source_quality_metrics_json` / `global_registration_json` may be missing
// or fail validation: the affected frames then carry an absent sigma2
// stream (n_eff fallback) exactly per spec.
ForwardDrizzleV2ProductionResult persist_forward_drizzle_v2_from_predecessors(
    const fs::path &store_root, const fs::path &quality_artifact,
    const registration::RegistrationSamplingPlan &sampling,
    VerifiedNormalizedSourceCache &cache, const GlobalQualityConfig &quality_cfg,
    const config::ReconstructionDrizzleConfig &drizzle_cfg,
    const config::ReconstructionClippingConfig &clipping_cfg,
    const config::ReconstructionMultibandConfig &multiband_cfg,
    const fs::path &source_quality_cache_root,
    const fs::path &source_quality_metrics_json,
    const fs::path &global_registration_json,
    const DrizzleGeometryCacheReader *geometry_cache,
    const ForwardDrizzleSubdivisionParams &subdivision,
    const std::string &config_snapshot_hash,
    const std::string &acceleration_backend,
    const std::function<void(int, int)> &progress = {});

}  // namespace tile_compile::reconstruction
