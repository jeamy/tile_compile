#pragma once

// GLOBAL_QUALITY (plan sections 5.2, 11.9): per-frame global quality factor
// G_quality(f), computed from the source-space CFA analysis proxy (plan
// 13.2) instead of a prewarped luminance image --- the direct consequence
// of removing PREWARP from the reconstruction path (plan 11.9's "Herkunft
// von G_quality(f) nach PREWARP-Entfernung").
//
// Reuses metrics::calculate_global_weights_with_stars() EXACTLY as-is
// ("dieselbe mathematische Definition wie bisher", plan 11.9) on the new
// input. That function's raw output is w = exp(k * clamp(Q, lo, hi)), an
// UNBOUNDED positive weight (explicitly documented in metrics.cpp as never
// normalized to sum 1, since its absolute, frame-count-independent scale is
// meaningful) --- it does NOT satisfy the plan's "G_quality(f) in [0,1]"
// contract (11.9), which A_coverage's clamp(w_profile/w_uniform, 0, 1)
// (14.4) actually depends on: w_profile <= w_uniform pointwise requires
// G_eff <= 1, and G_eff already multiplies two other [0,1] factors, so a
// G_quality > 1 on any single frame would silently break that invariant
// rather than error anywhere. Discovered while implementing this, not
// stated in the plan text --- see the progress notes for the reasoning.
//
// Closed with the minimal, algebraically exact fix: the logistic squashing
// w/(1+w) = sigmoid(k*Q) applied on top of the *unmodified* legacy weight,
// rather than re-deriving or duplicating its internals. This preserves the
// reused formula's frame-set-independent absolute scale (still monotonic in
// the same underlying Q score) while landing in the open interval (0,1) ---
// and deliberately never exactly 0 or 1, so it can never be confused with
// the separate, explicit Q=0 veto semantics (plan 11.9).

#include "tile_compile/core/types.hpp"
#include "tile_compile/reconstruction/forward_drizzle.hpp"

#include <vector>

namespace tile_compile::reconstruction {

struct GlobalQualityConfig {
  float w_bg = 0.4f;
  float w_noise = 0.3f;
  float w_grad = 0.3f;
  float w_fwhm = 0.0f;
  float w_roundness = 0.0f;
  float w_star_count = 0.0f;
  float clamp_lo = -3.0f;
  float clamp_hi = 3.0f;
  bool adaptive_weights = false;
  float weight_exponent_scale = 1.0f;
  int star_max_corners = 400;
  int star_patch_radius = 10;
};

// `sources` are the normalized CFA (OSC) or L (MONO) frames from the
// existing normalized cache (plan 10.1) --- never prewarped images. Returns
// one G_quality(f) per frame, in the open interval (0,1), same order as
// `sources`. The first frame is used as the reference star count for wFWHM
// (matches the existing legacy convention).
// Provider overload retains one frame's proxy plus scalar metrics for all
// frames. The provider must release its previous source before loading another.
VectorXf compute_global_quality_weights(size_t frame_count,
                                        const SourceImageProvider &source_of,
                                        ColorMode color_mode, BayerPattern bayer_pattern,
                                        int cfa_origin_x, int cfa_origin_y,
                                        const GlobalQualityConfig &cfg);

VectorXf compute_global_quality_weights(const std::vector<Matrix2Df> &sources,
                                        ColorMode color_mode, BayerPattern bayer_pattern,
                                        int cfa_origin_x, int cfa_origin_y,
                                        const GlobalQualityConfig &cfg);

}  // namespace tile_compile::reconstruction
