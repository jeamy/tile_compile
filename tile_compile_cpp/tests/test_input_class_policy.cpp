// M0 tests for the single-method input-class policy
// (docs/AQMH/aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md
//  section 3.1.1).

#include "tile_compile/core/input_class_policy.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>

using namespace tile_compile;
using namespace tile_compile::core;

TEST_CASE("MONO raw is accepted (plan 3.1.1)") {
  const auto d = classify_input_for_single_method(ColorMode::MONO,
                                                  BayerPattern::UNKNOWN);
  REQUIRE(d == InputClassDecision::accept_mono);
  REQUIRE(input_class_accepted(d));
}

TEST_CASE("OSC raw with a known Bayer pattern is accepted (plan 3.1.1)") {
  for (BayerPattern b : {BayerPattern::RGGB, BayerPattern::BGGR,
                         BayerPattern::GRBG, BayerPattern::GBRG}) {
    const auto d = classify_input_for_single_method(ColorMode::OSC, b);
    REQUIRE(d == InputClassDecision::accept_osc);
    REQUIRE(input_class_accepted(d));
  }
}

TEST_CASE("OSC without a Bayer pattern is rejected fail-closed (plan 3.1.1)") {
  const auto d = classify_input_for_single_method(ColorMode::OSC,
                                                  BayerPattern::UNKNOWN);
  REQUIRE(d == InputClassDecision::reject_osc_no_bayer);
  REQUIRE_FALSE(input_class_accepted(d));
  REQUIRE(input_class_rejection_message(d).find("UNSUPPORTED_INPUT") !=
          std::string::npos);
}

TEST_CASE("already-debayered RGB is rejected fail-closed (plan 3.1.1)") {
  const auto d = classify_input_for_single_method(ColorMode::RGB,
                                                  BayerPattern::UNKNOWN);
  REQUIRE(d == InputClassDecision::reject_rgb);
  REQUIRE_FALSE(input_class_accepted(d));
  const std::string msg = input_class_rejection_message(d);
  REQUIRE(msg.find("UNSUPPORTED_INPUT") != std::string::npos);
  REQUIRE(msg.find("RGB") != std::string::npos);
}
