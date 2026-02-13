#pragma once

namespace tile_compile::core {

enum class FrameMode {
  AbortInsufficient,
  EmergencyReduced,
  Reduced,
  Full,
};

struct ModeGateDecision {
  FrameMode mode = FrameMode::AbortInsufficient;
  bool reduced_mode = false;
  bool emergency_mode = false;
  bool should_abort = true;
};

ModeGateDecision evaluate_mode_gate(int usable_frames, int reduced_threshold,
                                    bool allow_emergency_mode,
                                    int reduced_min_frames = 50);

} // namespace tile_compile::core
