#include "tile_compile/core/mode_gating.hpp"

namespace tile_compile::core {

/// @brief Implements evaluate mode gate.
/// @details Part of pipeline mode selection and emergency/reduced/full-mode gating; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
ModeGateDecision evaluate_mode_gate(int usable_frames, int reduced_threshold,
                                    bool allow_emergency_mode,
                                    int reduced_min_frames) {
  ModeGateDecision out;
  if (usable_frames < reduced_min_frames) {
    if (allow_emergency_mode) {
      out.mode = FrameMode::EmergencyReduced;
      out.reduced_mode = true;
      out.emergency_mode = true;
      out.should_abort = false;
    }
    return out;
  }

  if (usable_frames < reduced_threshold) {
    out.mode = FrameMode::Reduced;
    out.reduced_mode = true;
    out.emergency_mode = false;
    out.should_abort = false;
    return out;
  }

  out.mode = FrameMode::Full;
  out.reduced_mode = false;
  out.emergency_mode = false;
  out.should_abort = false;
  return out;
}

} // namespace tile_compile::core
