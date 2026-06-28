#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

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

// A physical meridian flip changes the orientation of a temporally coherent
// part of a session.  Isolated half-turn solutions are the common 180-degree
// ambiguity of star-pattern matching and must not be admitted as valid model
// anchors.  Allow small holes so an occasional failed registration inside a
// real flip segment does not split it.
inline std::vector<uint8_t> persistent_half_turn_support(
    const std::vector<uint8_t> &candidates, int radius = 8,
    int minimum_support = 4) {
  std::vector<uint8_t> supported(candidates.size(), 0);
  if (radius < 0 || minimum_support <= 0) {
    return supported;
  }
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i] == 0) {
      continue;
    }
    const size_t lo = i > static_cast<size_t>(radius)
                          ? i - static_cast<size_t>(radius)
                          : 0;
    const size_t hi = std::min(candidates.size(),
                               i + static_cast<size_t>(radius) + 1);
    int count = 0;
    for (size_t j = lo; j < hi; ++j) {
      count += candidates[j] != 0 ? 1 : 0;
    }
    supported[i] = count >= minimum_support ? 1 : 0;
  }
  return supported;
}

} // namespace tile_compile::runner
