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

struct DrizzleStoreResult {
  fs::path generation_dir;
  ForwardDrizzleDiagnostics diagnostics;
  ForwardDrizzleClippingDiagnostics clipping;
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
    const FrameQualityProvider &quality_of = {});

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
    const FrameQualityProvider &quality_of,
    const ForwardDrizzleSubdivisionParams &subdivision = {},
    const std::vector<float> &g_eff = {},
    const DrizzleStorePredecessors &predecessors = {},
    const ForwardDrizzleCudaOptions &cuda = {});

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
