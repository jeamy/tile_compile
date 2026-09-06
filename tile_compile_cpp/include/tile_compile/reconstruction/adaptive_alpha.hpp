#pragma once

// Adaptive per-band alpha for the controlled multi-band reconstruction ---
// milestone M6 (plan section 14.4).
//
//   alpha_j = alpha_cap * A_neff * A_coverage * A_separation
//                       * A_artifact * A_registration
//
// All factors are in [0, 1]; alpha_j is therefore at most its smallest
// factor. One alpha_j map is SHARED across R/G/B (plan 14.6); every
// channel-specific factor uses the conservative min over the active channels.
//
// This module computes the two factors that need only the profile planes:
//   A_neff,c     = smoothstep(min_effective_samples, full_effective_samples,
//                             n_eff_profile,c)
//   A_coverage,c = clamp(profile_support_weight_c / uniform_support_weight_c,
//                        0, 1)
// with A_* = min over active channels. A_separation / A_artifact /
// A_registration need per-frame stripe statistics and are supplied here as
// pre-reduced per-pixel maps (default: 1 everywhere) --- a later M6 batch
// computes them.
//
// Only bands whose profile source is Fine or Medium get a computed alpha map;
// bands sourced from Raw ignore alpha (plan 14.3) and get an empty map.

#include "tile_compile/core/types.hpp"
#include "tile_compile/reconstruction/forward_drizzle.hpp"

#include <vector>

namespace tile_compile::reconstruction {

struct AdaptiveAlphaParams {
  float alpha_cap = 1.0f;              // [0, 1]
  float min_effective_samples = 8.0f;  // 1 <= min < full
  float full_effective_samples = 24.0f;
};

// `fine` / `medium` carry the F / M profile planes (per channel R/G/B or L);
// `medium` may be an empty ForwardDrizzleUniformResult when levels < 2.
// External factor maps, if non-empty, must be row-major size width*height with
// values in [0, 1] (already reduced to the channel min by the caller).
//
// Returns `levels` maps. Map j (0-based) is the shared alpha for band j+1 when
// that band is sourced from Fine (j==0) or Medium (j==1); every other band
// (Raw-sourced) returns an empty vector.
std::vector<std::vector<float>> compute_adaptive_alpha(
    const ForwardDrizzleUniformResult &uniform,
    const ForwardDrizzleUniformResult &fine,
    const ForwardDrizzleUniformResult &medium, ColorMode mode, int width,
    int height, int levels, const AdaptiveAlphaParams &params,
    const std::vector<float> &a_separation = {},
    const std::vector<float> &a_artifact = {},
    const std::vector<float> &a_registration = {});

// Hermite smoothstep clamped to [0,1]; exposed for unit tests.
double alpha_smoothstep(double edge0, double edge1, double x);

}  // namespace tile_compile::reconstruction
