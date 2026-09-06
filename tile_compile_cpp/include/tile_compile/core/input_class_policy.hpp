#pragma once

// Input-class policy --- milestone M0 of the CFA-forward-drizzle plan
// (docs/AQMH/aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md
//  sections 3.1.1, 23/M0).
//
// The single-method pipeline supports exactly two input classes: OSC raw with a
// known Bayer pattern, and MONO raw. Already-debayered RGB cubes are rejected
// fail-closed at SCAN_INPUT with a stable "UNSUPPORTED_INPUT" error; unknown /
// contradictory colour metadata is also rejected fail-closed.

#include "tile_compile/core/types.hpp"

#include <string>

namespace tile_compile::core {

enum class InputClassDecision {
  accept_osc,       // OSC with a known Bayer pattern
  accept_mono,      // MONO single plane
  reject_rgb,       // already-debayered RGB cube --- out of scope for the cutover
  reject_osc_no_bayer,  // OSC without a resolvable Bayer pattern
  reject_unknown,   // no usable colour classification
};

inline InputClassDecision classify_input_for_single_method(
    ColorMode mode, BayerPattern bayer) {
  switch (mode) {
    case ColorMode::MONO:
      return InputClassDecision::accept_mono;
    case ColorMode::OSC:
      return bayer == BayerPattern::UNKNOWN
                 ? InputClassDecision::reject_osc_no_bayer
                 : InputClassDecision::accept_osc;
    case ColorMode::RGB:
      return InputClassDecision::reject_rgb;
    default:
      return InputClassDecision::reject_unknown;
  }
}

inline bool input_class_accepted(InputClassDecision d) {
  return d == InputClassDecision::accept_osc ||
         d == InputClassDecision::accept_mono;
}

inline std::string input_class_rejection_message(InputClassDecision d) {
  switch (d) {
    case InputClassDecision::reject_rgb:
      return "UNSUPPORTED_INPUT: already-debayered RGB frames are not accepted "
             "by the CFA-forward-drizzle single-method pipeline (plan section "
             "3.1.1). Supported input classes: OSC raw (with a known Bayer "
             "pattern) and MONO raw.";
    case InputClassDecision::reject_osc_no_bayer:
      return "UNSUPPORTED_INPUT: OSC input without a resolvable Bayer pattern "
             "(BAYERPAT / COLORTYP / data.bayer_pattern). The single-method "
             "pipeline does not guess a CFA pattern (plan section 3.1.1).";
    case InputClassDecision::reject_unknown:
    default:
      return "UNSUPPORTED_INPUT: the colour mode of the input frames could not "
             "be classified as OSC or MONO (plan section 3.1.1).";
  }
}

}  // namespace tile_compile::core
