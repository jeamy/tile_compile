#if __has_include(<catch2/catch_test_macros.hpp>)
#include "tile_compile/core/mode_gating.hpp"

#include <catch2/catch_test_macros.hpp>

using tile_compile::core::FrameMode;
using tile_compile::core::ModeGateDecision;
using tile_compile::core::evaluate_mode_gate;

TEST_CASE("mode_gate_aborts_below_50_without_emergency") {
  ModeGateDecision d = evaluate_mode_gate(49, 200, false, 50);
  REQUIRE(d.should_abort);
  REQUIRE_FALSE(d.reduced_mode);
  REQUIRE_FALSE(d.emergency_mode);
  REQUIRE(d.mode == FrameMode::AbortInsufficient);
}

TEST_CASE("mode_gate_emergency_reduced_below_50_with_emergency") {
  ModeGateDecision d = evaluate_mode_gate(49, 200, true, 50);
  REQUIRE_FALSE(d.should_abort);
  REQUIRE(d.reduced_mode);
  REQUIRE(d.emergency_mode);
  REQUIRE(d.mode == FrameMode::EmergencyReduced);
}

TEST_CASE("mode_gate_reduced_between_50_and_threshold") {
  ModeGateDecision d = evaluate_mode_gate(150, 200, false, 50);
  REQUIRE_FALSE(d.should_abort);
  REQUIRE(d.reduced_mode);
  REQUIRE_FALSE(d.emergency_mode);
  REQUIRE(d.mode == FrameMode::Reduced);
}

TEST_CASE("mode_gate_full_at_or_above_threshold") {
  ModeGateDecision d = evaluate_mode_gate(200, 200, false, 50);
  REQUIRE_FALSE(d.should_abort);
  REQUIRE_FALSE(d.reduced_mode);
  REQUIRE_FALSE(d.emergency_mode);
  REQUIRE(d.mode == FrameMode::Full);
}
#else
int tile_compile_tests_mode_gating_stub() { return 0; }
#endif
