#pragma once

#include <span>

namespace tile_compile::registration {

// One measured (valid) registration used as a plausibility anchor for
// model-predicted warps. Anchors must be sorted by fi ascending and ang_rad
// must already be unwrapped along the sequence.
struct WarpPredictionAnchor {
  float fi = 0.0f;
  float ang_rad = 0.0f;
  float tx = 0.0f;
  float ty = 0.0f;
};

enum class WarpPredictionGateReason {
  ok,
  no_anchor,
  extrapolation_too_long,
  angle_discontinuity,
  shift_discontinuity,
};

struct WarpPredictionGateResult {
  bool ok = false;
  WarpPredictionGateReason reason = WarpPredictionGateReason::no_anchor;
  float angle_dev_deg = 0.0f;
  float angle_allowed_deg = 0.0f;
  float shift_dev_px = 0.0f;
  float shift_allowed_px = 0.0f;
  int anchor_distance = 0;
};

// Gate constants (always on, no user-facing parameter):
//   kWarpGateAngleAbsTolDeg: base angle tolerance added to the rate term.
//   kWarpGateRateMult: multiplier on the measured local/global rate.
//   kWarpGateAngleHardCapDeg: upper bound of the allowed angle deviation.
//   kWarpGateShiftAbsTolPx: base shift tolerance in pixels.
//   kWarpGateShiftHardCapPx: upper bound of the allowed shift deviation.
//   kWarpGateMaxExtrapolationFrames: max distance past the last/first anchor.
//   kWarpGateLocalRatePairs: consecutive anchor pairs sampled for the local
//     rate on each side of the prediction point.
inline constexpr float kWarpGateAngleAbsTolDeg = 1.0f;
inline constexpr float kWarpGateRateMult = 2.0f;
inline constexpr float kWarpGateAngleHardCapDeg = 5.0f;
inline constexpr float kWarpGateShiftAbsTolPx = 60.0f;
inline constexpr float kWarpGateShiftHardCapPx = 400.0f;
inline constexpr int kWarpGateMaxExtrapolationFrames = 3;
inline constexpr int kWarpGateLocalRatePairs = 5;

// Checks whether a model-predicted warp (angle/tx/ty at frame index fi) is
// consistent with the neighbouring measured anchors. A prediction that
// deviates more than the rate-scaled tolerance from the bracketing anchors
// (or lies too far outside the anchor span) is implausible and must leave
// the frame unresolved instead of receiving a bogus model warp.
WarpPredictionGateResult evaluate_warp_prediction(
    float fi, float pred_ang_rad, float pred_tx, float pred_ty,
    std::span<const WarpPredictionAnchor> anchors);

const char *warp_prediction_gate_reason_name(WarpPredictionGateReason reason);

}  // namespace tile_compile::registration
