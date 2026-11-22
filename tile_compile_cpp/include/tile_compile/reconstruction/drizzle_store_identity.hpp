#pragma once

// Store identity and multiband contract of the forward-drizzle reconstruction.
// They outlived the v1 profile store (drizzle_profile_store, removed
// 2026-11-21); the v2 producer fills DrizzleStoreIdentity and the v2 fusion
// reads MultibandStoreContract.

#include "tile_compile/core/types.hpp"
#include "tile_compile/reconstruction/adaptive_alpha.hpp"
#include "tile_compile/reconstruction/alpha_confidence.hpp"
#include "tile_compile/reconstruction/alpha_guard.hpp"

#include <string>

namespace tile_compile::reconstruction {

// M6: the full multiband contract that enters `multiband_config_hash`
// (plan 16.4: profile hashes plus levels, alpha, energy, support, downsample
// and validation contract), consumed by the v2 multiband fusion.
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

} // namespace tile_compile::reconstruction
