// Tests for the warp-prediction plausibility gate used by the registration
// field-rotation model (SECTION 6 of runner_phase_registration.cpp).

#include "tile_compile/registration/warp_prediction_gate.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstring>
#include <vector>

using namespace tile_compile::registration;
using Catch::Approx;

namespace {

constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;

std::vector<WarpPredictionAnchor> make_smooth_anchors() {
  // fi = 0..40 with a 3-frame gap at 17..19; gentle linear trends.
  std::vector<WarpPredictionAnchor> v;
  for (int fi = 0; fi <= 40; ++fi) {
    if (fi >= 17 && fi <= 19) {
      continue;
    }
    v.push_back({static_cast<float>(fi), 0.0005f * static_cast<float>(fi),
                 0.3f * static_cast<float>(fi),
                 -0.1f * static_cast<float>(fi)});
  }
  return v;
}

}  // namespace

TEST_CASE("warp prediction gate accepts smooth interpolation",
          "[registration][warp-prediction-gate]") {
  const auto anchors = make_smooth_anchors();
  // Linear interpolation between anchors 16 and 20 at fi=18.
  const auto r = evaluate_warp_prediction(18.0f, 0.0005f * 18.0f, 0.3f * 18.0f,
                                          -0.1f * 18.0f, anchors);
  REQUIRE(r.ok);
  REQUIRE(r.reason == WarpPredictionGateReason::ok);
}

TEST_CASE("warp prediction gate rejects discontinuity",
          "[registration][warp-prediction-gate]") {
  // Anchors 0..20 at ~0.3 rad, 40..60 at ~2.9 rad, with a 4000 px tx jump.
  std::vector<WarpPredictionAnchor> anchors;
  for (int fi = 0; fi <= 20; ++fi) {
    anchors.push_back({static_cast<float>(fi), 0.3f,
                       5.0f * static_cast<float>(fi), 0.0f});
  }
  for (int fi = 40; fi <= 60; ++fi) {
    anchors.push_back({static_cast<float>(fi), 2.9f,
                       4000.0f + 5.0f * static_cast<float>(fi), 0.0f});
  }

  SECTION("mid-angle prediction across the jump is an angle discontinuity") {
    const auto r =
        evaluate_warp_prediction(30.0f, 1.6f, 2100.0f, 0.0f, anchors);
    REQUIRE_FALSE(r.ok);
    REQUIRE(r.reason == WarpPredictionGateReason::angle_discontinuity);
    REQUIRE(r.angle_dev_deg > r.angle_allowed_deg);
  }

  SECTION("angle-consistent but mid-tx prediction is a shift discontinuity") {
    // Angle matches the left anchor (checked first), tx is 2000 px off.
    const auto r =
        evaluate_warp_prediction(30.0f, 0.3f, 2100.0f, 0.0f, anchors);
    REQUIRE_FALSE(r.ok);
    REQUIRE(r.reason == WarpPredictionGateReason::shift_discontinuity);
    REQUIRE(r.shift_dev_px > r.shift_allowed_px);
  }
}

TEST_CASE("warp prediction gate extrapolation limits",
          "[registration][warp-prediction-gate]") {
  std::vector<WarpPredictionAnchor> anchors;
  for (int fi = 0; fi <= 100; ++fi) {
    anchors.push_back({static_cast<float>(fi), 0.001f * static_cast<float>(fi),
                       2.0f * static_cast<float>(fi),
                       -0.5f * static_cast<float>(fi)});
  }

  SECTION("3 frames past the last anchor is accepted") {
    const auto r = evaluate_warp_prediction(103.0f, 0.001f * 103.0f,
                                            2.0f * 103.0f, -0.5f * 103.0f,
                                            anchors);
    REQUIRE(r.ok);
  }
  SECTION("4 frames past the last anchor is rejected") {
    const auto r = evaluate_warp_prediction(104.0f, 0.001f * 104.0f,
                                            2.0f * 104.0f, -0.5f * 104.0f,
                                            anchors);
    REQUIRE_FALSE(r.ok);
    REQUIRE(r.reason == WarpPredictionGateReason::extrapolation_too_long);
  }
  SECTION("4 frames before the first anchor is rejected") {
    const auto r = evaluate_warp_prediction(-4.0f, -0.004f, -8.0f, 2.0f,
                                            anchors);
    REQUIRE_FALSE(r.ok);
    REQUIRE(r.reason == WarpPredictionGateReason::extrapolation_too_long);
  }
}

TEST_CASE("warp prediction gate angle hard cap",
          "[registration][warp-prediction-gate]") {
  // Fast legit local rate: 0.014 rad/frame (~0.8 deg/frame) near the anchor.
  // allowed = min(5 deg, 1 + 2*0.014*57.3*dist); at dist=3 the cap binds.
  std::vector<WarpPredictionAnchor> anchors = {
      {0.0f, 0.0f, 0.0f, 0.0f},
      {5.0f, 0.07f, 10.0f, 0.0f},
      {10.0f, 0.14f, 20.0f, 0.0f},
  };
  const float base_ang = 0.14f;
  const float base_tx = 20.0f + 3.0f * 2.0f;  // follow the 2 px/frame trend

  SECTION("4.5 deg deviation at 3 frames distance is accepted") {
    const auto r = evaluate_warp_prediction(
        13.0f, base_ang + 4.5f * kDegToRad, base_tx, 0.0f, anchors);
    REQUIRE(r.ok);
    REQUIRE(r.angle_allowed_deg == Approx(5.0f));
  }
  SECTION("6 deg deviation at 3 frames distance is rejected") {
    const auto r = evaluate_warp_prediction(
        13.0f, base_ang + 6.0f * kDegToRad, base_tx, 0.0f, anchors);
    REQUIRE_FALSE(r.ok);
    REQUIRE(r.reason == WarpPredictionGateReason::angle_discontinuity);
  }
}

TEST_CASE("warp prediction gate reports no_anchor on empty input",
          "[registration][warp-prediction-gate]") {
  const std::vector<WarpPredictionAnchor> anchors;
  const auto r = evaluate_warp_prediction(5.0f, 0.0f, 0.0f, 0.0f, anchors);
  REQUIRE_FALSE(r.ok);
  REQUIRE(r.reason == WarpPredictionGateReason::no_anchor);
}

TEST_CASE("warp prediction gate reason names round trip",
          "[registration][warp-prediction-gate]") {
  for (auto reason :
       {WarpPredictionGateReason::ok, WarpPredictionGateReason::no_anchor,
        WarpPredictionGateReason::extrapolation_too_long,
        WarpPredictionGateReason::angle_discontinuity,
        WarpPredictionGateReason::shift_discontinuity}) {
    const char *name = warp_prediction_gate_reason_name(reason);
    REQUIRE(name != nullptr);
    REQUIRE(std::strlen(name) > 0);
    REQUIRE(std::strcmp(name, "unknown") != 0);
  }
  REQUIRE(std::strcmp(warp_prediction_gate_reason_name(
                          WarpPredictionGateReason::angle_discontinuity),
                      "angle_discontinuity") == 0);
}

TEST_CASE("warp prediction gate rejects the IC5070 discontinuity blend",
          "[registration][warp-prediction-gate]") {
  // Real anchors from ic5070_20260921_171913 global_registration.json:
  // frames 355..363 and 401..410 (ang = atan2(a10,a00), unwrapped).
  const std::vector<WarpPredictionAnchor> anchors = {
      {355.0f, 0.241492943f, 691.4274f, -422.2586f},
      {356.0f, 0.249673944f, 709.6264f, -433.1476f},
      {357.0f, 0.257781390f, 726.5285f, -444.7246f},
      {358.0f, 0.266840053f, 746.2296f, -455.8566f},
      {359.0f, 0.275481188f, 762.6429f, -468.1804f},
      {360.0f, 0.284933240f, 783.0626f, -483.2629f},
      {361.0f, 0.297245541f, 805.0042f, -507.7739f},
      {362.0f, 0.309991106f, 826.1989f, -531.5249f},
      {363.0f, 0.324163594f, 857.3435f, -556.1102f},
      {401.0f, 2.976684254f, 4984.8159f, 1773.8778f},
      {402.0f, 2.985715702f, 4966.3809f, 1792.1912f},
      {403.0f, 2.993255396f, 4949.6787f, 1804.7067f},
      {404.0f, 3.003202634f, 4934.1802f, 1820.2456f},
      {405.0f, 3.018832917f, 4897.5190f, 1850.4194f},
      {406.0f, 3.026084011f, 4876.7144f, 1863.6760f},
      {407.0f, 3.033755000f, 4850.8604f, 1877.4496f},
      {408.0f, 3.041354572f, 4831.5479f, 1890.9537f},
      {409.0f, 3.047694795f, 4810.7319f, 1903.0869f},
      {410.0f, 3.053720927f, 4792.5093f, 1914.7662f},
  };
  // The model_blended prediction seen in the real run at fi=380.
  const auto r = evaluate_warp_prediction(
      380.0f, 73.58f * kDegToRad, 2372.4f, 143.0f, anchors);
  REQUIRE_FALSE(r.ok);
  REQUIRE(r.reason == WarpPredictionGateReason::angle_discontinuity);
}

TEST_CASE("warp prediction gate keeps the ic4605 single missing frame",
          "[registration][warp-prediction-gate]") {
  // Real anchors from ic4605_20260921_180332 global_registration.json:
  // frames 13..16 and 18..21; frame 17 is the only gap in a smooth trend.
  const std::vector<WarpPredictionAnchor> anchors = {
      {13.0f, -0.131068328f, -286.9598f, 275.5132f},
      {14.0f, -0.130413237f, -284.4441f, 275.6318f},
      {15.0f, -0.129886607f, -283.0788f, 269.4312f},
      {16.0f, -0.129685368f, -281.1294f, 265.8961f},
      {18.0f, -0.128142200f, -272.8920f, 251.3580f},
      {19.0f, -0.127635254f, -271.7801f, 248.3300f},
      {20.0f, -0.127024865f, -270.5806f, 243.7693f},
      {21.0f, -0.124699411f, -264.8810f, 228.6157f},
  };
  const auto r = evaluate_warp_prediction(
      17.0f, -7.409f * kDegToRad, -276.7f, 258.5f, anchors);
  REQUIRE(r.ok);
  REQUIRE(r.reason == WarpPredictionGateReason::ok);
}
