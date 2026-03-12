#pragma once

#include <algorithm>
#include <cmath>

#include "tile_compile/core/types.hpp"

namespace tile_compile::runner {

struct RegistrationShiftDiagnostics {
  float shift_magnitude = 0.0f;
  bool half_turn_family = false;
};

inline RegistrationShiftDiagnostics registration_shift_diagnostics(
    const WarpMatrix &w, int width, int height) {
  constexpr float kPi = 3.14159265358979323846f;
  RegistrationShiftDiagnostics diag;

  const float angle = std::atan2(w(0, 1), w(0, 0));
  const float angle_abs = std::fabs(std::remainder(angle, 2.0f * kPi));
  diag.half_turn_family = angle_abs > (0.75f * kPi);

  if (!diag.half_turn_family) {
    diag.shift_magnitude = std::hypot(w(0, 2), w(1, 2));
    return diag;
  }

  const float cx = 0.5f * static_cast<float>(std::max(0, width - 1));
  const float cy = 0.5f * static_cast<float>(std::max(0, height - 1));
  const float expected_tx = cx - (w(0, 0) * cx + w(0, 1) * cy);
  const float expected_ty = cy - (w(1, 0) * cx + w(1, 1) * cy);

  diag.shift_magnitude = std::hypot(w(0, 2) - expected_tx, w(1, 2) - expected_ty);
  return diag;
}

} // namespace tile_compile::runner
