#pragma once

// A_separation / A_artifact / A_registration --- the adaptive-alpha
// confidence factors that need per-frame stripe statistics (plan 14.4).
// Milestone M6.
//
// Status: this is the reviewed, tested PRIMITIVE. Wiring it into the
// forward-drizzle stripe loop (which must expose, per accepted frame
// contribution, the geometric weight B_f,c, the K-averaged Q_composite and
// artifact_confidence, and the frame's registration flags) is a separate
// integration step --- exactly as apply_robust_clipping() was landed before
// its drizzle wiring.
//
// Definitions (plan 14.4), per target pixel, per active channel c:
//
//   separation      = weighted_p90(q) - weighted_p50(q)          weight = B
//   A_separation,c  = smoothstep(min_quality_separation,
//                                full_quality_separation, separation)
//
//   a_f,c           = K-averaged artifact_confidence_f (per contribution)
//   a_p10,c         = weighted_p10(a_f,c)                          weight = B
//   A_artifact,c    = smoothstep(0.25, 0.75, a_p10,c)
//                     ; < min_artifact_contributors valid contributions => 0
//
//   direct_fraction_c = sum_f B * is_direct / sum_f B
//   residual_p20_c    = weighted_p20(registration_residual_factor_f)  weight=B
//   A_registration,c  = min(smoothstep(0.50, 0.85, direct_fraction_c),
//                           smoothstep(0.55, 0.90, residual_p20_c))
//
// A_* over the frame is min over active channels (done by the caller).

#include <cstdint>
#include <span>
#include <vector>

namespace tile_compile::reconstruction {

// One accepted frame's contribution to a single target pixel/channel.
struct AlphaFactorContribution {
  double b = 0.0;                // geometric weight B_f,c(q) (> 0)
  double q_composite = 1.0;      // K-averaged Q_composite_f,c(q), in [0,1]
  double artifact_conf = 1.0;    // K-averaged artifact_confidence_f,c(q), [0,1]
                                 // (non-finite => "not applicable", excluded)
  bool is_direct = false;        // is_direct_registration_f
  double residual_factor = 1.0;  // registration_residual_factor_f, in (0,1]
};

struct AlphaConfidenceParams {
  double min_quality_separation = 0.05;
  double full_quality_separation = 0.20;
  int min_artifact_contributors = 8;
  // The plan fixes the registration smoothstep edges.
  double direct_fraction_lo = 0.50, direct_fraction_hi = 0.85;
  double residual_p20_lo = 0.55, residual_p20_hi = 0.90;
  double artifact_lo = 0.25, artifact_hi = 0.75;
};

struct AlphaConfidenceFactors {
  double a_separation = 0.0;
  double a_artifact = 0.0;
  double a_registration = 0.0;
  bool artifact_applicable = false;  // >= min_artifact_contributors valid a_f
};

// One channel, one target pixel. `accepted` is the post-clipping accepted
// frame contribution set (each b > 0). An empty set yields all-zero factors.
AlphaConfidenceFactors compute_alpha_confidence_channel(
    std::span<const AlphaFactorContribution> accepted,
    const AlphaConfidenceParams &params = {});

// Weighted percentile in [0,1]: values sorted ascending, running weight, the
// value at which the cumulative weight first reaches `p * total_weight`
// (linear interpolation between the bracketing samples). Exposed for tests.
double weighted_percentile(std::span<const double> values,
                           std::span<const double> weights, double p);

}  // namespace tile_compile::reconstruction
