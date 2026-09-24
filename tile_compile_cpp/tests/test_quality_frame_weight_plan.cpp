// M3 tests for QualityFrameWeightPlan (plan section 11.9): G_eff(f) computed
// exactly once, registration factors read verbatim from the sampling plan,
// canonical hash and fail-closed loader.

#include "tile_compile/reconstruction/quality_frame_weight_plan.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace tile_compile;
using namespace tile_compile::reconstruction;
using registration::FrameSamplingTransform;
using registration::RegistrationSamplingPlan;
using Catch::Approx;

namespace {
RegistrationSamplingPlan two_frame_plan() {
  RegistrationSamplingPlan plan;
  plan.source_identity_hash = "src-hash-abc";
  plan.plan_hash = "sampling-plan-hash-xyz";
  FrameSamplingTransform f0;
  f0.frame_id = "src-hash-abc:0";
  f0.source_index = 0;
  f0.valid = true;
  f0.model_prediction_factor = 0.8f;
  f0.registration_residual_factor = 0.9f;
  FrameSamplingTransform f1;
  f1.frame_id = "src-hash-abc:1";
  f1.source_index = 1;
  f1.valid = true;
  f1.model_prediction_factor = 1.0f;
  f1.registration_residual_factor = 0.6f;
  plan.frames = {f0, f1};
  return plan;
}
}  // namespace

TEST_CASE("quality frame weight plan: g_eff is exactly the product of its "
          "three factors, registration factors taken verbatim (plan 11.9)") {
  auto sampling = two_frame_plan();
  VectorXf g_quality(2);
  g_quality << 0.5f, 0.75f;

  auto plan = build_quality_frame_weight_plan(sampling, g_quality, "cfg-hash-1");

  REQUIRE(plan.frames.size() == 2);
  REQUIRE(plan.frames[0].g_quality == Approx(0.5f));
  REQUIRE(plan.frames[0].model_prediction_factor == Approx(0.8f));
  REQUIRE(plan.frames[0].registration_residual_factor == Approx(0.9f));
  REQUIRE(plan.frames[0].g_eff == Approx(0.5f * 0.8f * 0.9f));  // 0.36
  REQUIRE(plan.frames[1].g_eff == Approx(0.75f * 1.0f * 0.6f));  // 0.45
  REQUIRE(plan.source_identity_hash == "src-hash-abc");
  REQUIRE(plan.sampling_plan_hash == "sampling-plan-hash-xyz");
  REQUIRE(plan.source_quality_config_hash == "cfg-hash-1");
  REQUIRE_FALSE(plan.plan_hash.empty());
}

TEST_CASE("quality frame weight plan: size mismatch between g_quality and "
          "the sampling plan is rejected") {
  auto sampling = two_frame_plan();
  VectorXf g_quality(3);
  g_quality << 0.1f, 0.2f, 0.3f;
  REQUIRE_THROWS(build_quality_frame_weight_plan(sampling, g_quality, "cfg"));
}

TEST_CASE("quality frame weight plan: round-trips through JSON losslessly "
          "and re-validates its hash") {
  auto sampling = two_frame_plan();
  VectorXf g_quality(2);
  g_quality << 0.5f, 0.75f;
  auto plan = build_quality_frame_weight_plan(sampling, g_quality, "cfg-hash-1");

  const std::string js = serialize_quality_frame_weight_plan(plan);
  QualityFrameWeightPlan parsed;
  std::string error;
  REQUIRE(parse_quality_frame_weight_plan(js, parsed, error));
  REQUIRE(error.empty());
  REQUIRE(parsed.plan_hash == plan.plan_hash);
  REQUIRE(parsed.frames.size() == 2);
  REQUIRE(parsed.frames[1].g_eff == Approx(plan.frames[1].g_eff));
}

TEST_CASE("quality frame weight plan: a tampered artifact is rejected "
          "fail-closed (hash mismatch, not silently accepted)") {
  auto sampling = two_frame_plan();
  VectorXf g_quality(2);
  g_quality << 0.5f, 0.75f;
  auto plan = build_quality_frame_weight_plan(sampling, g_quality, "cfg-hash-1");
  std::string js = serialize_quality_frame_weight_plan(plan);

  // Flip a g_quality digit without touching plan_hash.
  const auto pos = js.find("0.5");
  REQUIRE(pos != std::string::npos);
  js[pos + 2] = '9';  // 0.5 -> 0.9

  QualityFrameWeightPlan parsed;
  std::string error;
  REQUIRE_FALSE(parse_quality_frame_weight_plan(js, parsed, error));
  REQUIRE_FALSE(error.empty());
}

TEST_CASE("quality frame weight plan: an inconsistent g_eff (not the product "
          "of its factors) is rejected even if the hash matches it") {
  // Build a plan by hand where g_eff is deliberately wrong, then hash it so
  // the hash check passes --- the separate product check must still catch it.
  QualityFrameWeightPlan plan;
  plan.source_identity_hash = "s";
  plan.sampling_plan_hash = "p";
  plan.source_quality_config_hash = "c";
  QualityFrameWeight w;
  w.frame_id = "s:0";
  w.g_quality = 0.5f;
  w.model_prediction_factor = 0.8f;
  w.registration_residual_factor = 0.9f;
  w.g_eff = 0.99f;  // wrong: should be 0.36
  plan.frames = {w};
  plan.plan_hash = compute_quality_frame_weight_plan_hash(plan);

  const std::string js = serialize_quality_frame_weight_plan(plan);
  QualityFrameWeightPlan parsed;
  std::string error;
  REQUIRE_FALSE(parse_quality_frame_weight_plan(js, parsed, error));
  REQUIRE(error.find("g_eff") != std::string::npos);
}

TEST_CASE("source quality config hash: stable for equal configs, changes "
          "when any weighting parameter changes (plan 18.3 hash domain)") {
  GlobalQualityConfig a;
  GlobalQualityConfig b = a;
  REQUIRE(compute_source_quality_config_hash(a) == compute_source_quality_config_hash(b));

  b.w_noise += 0.01f;
  REQUIRE(compute_source_quality_config_hash(a) != compute_source_quality_config_hash(b));

  GlobalQualityConfig c = a;
  c.clamp_hi = 4.0f;
  REQUIRE(compute_source_quality_config_hash(a) != compute_source_quality_config_hash(c));

  GlobalQualityConfig d = a;
  d.star_patch_radius = 12;
  REQUIRE(compute_source_quality_config_hash(a) != compute_source_quality_config_hash(d));
}

TEST_CASE("quality frame weight plan hash: changes when g_quality changes, "
          "stable otherwise (frame-ID and formula stability, plan 11.9)") {
  auto sampling = two_frame_plan();
  VectorXf g1(2);
  g1 << 0.5f, 0.75f;
  VectorXf g2(2);
  g2 << 0.5f, 0.751f;

  auto p1 = build_quality_frame_weight_plan(sampling, g1, "cfg");
  auto p1_again = build_quality_frame_weight_plan(sampling, g1, "cfg");
  auto p2 = build_quality_frame_weight_plan(sampling, g2, "cfg");

  REQUIRE(p1.plan_hash == p1_again.plan_hash);
  REQUIRE(p1.plan_hash != p2.plan_hash);
}

TEST_CASE("quality frame weight plan: rehashed invalid factors and duplicate identities fail closed", "[drizzle-audit]") {
  auto sampling = two_frame_plan();
  VectorXf quality = VectorXf::Constant(2, 0.5f);
  auto valid = build_quality_frame_weight_plan(sampling, quality, "cfg");
  QualityFrameWeightPlan parsed;
  std::string error;
  auto bad = valid;
  bad.frames[0].g_quality = -0.5f;
  bad.frames[0].g_eff = bad.frames[0].g_quality * bad.frames[0].model_prediction_factor * bad.frames[0].registration_residual_factor;
  bad.plan_hash = compute_quality_frame_weight_plan_hash(bad);
  REQUIRE_FALSE(parse_quality_frame_weight_plan(serialize_quality_frame_weight_plan(bad), parsed, error));
  bad = valid;
  bad.frames[1].frame_id = bad.frames[0].frame_id;
  bad.plan_hash = compute_quality_frame_weight_plan_hash(bad);
  REQUIRE_FALSE(parse_quality_frame_weight_plan(serialize_quality_frame_weight_plan(bad), parsed, error));
  quality[0] = 2.0f;
  REQUIRE_THROWS(build_quality_frame_weight_plan(sampling, quality, "cfg"));
}
