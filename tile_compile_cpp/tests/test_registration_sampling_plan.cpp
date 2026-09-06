// M0 tests for RegistrationSamplingPlan
// (docs/AQMH/aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md
//  sections 7.1-7.4, 20.1).

#include "tile_compile/registration/registration_sampling_plan.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <nlohmann/json.hpp>
#include <string>

using namespace tile_compile;
using namespace tile_compile::registration;
using Catch::Approx;

namespace {

WarpMatrix make_affine(float a, float b, float tx, float c, float d, float ty) {
  WarpMatrix m;
  m(0, 0) = a; m(0, 1) = b; m(0, 2) = tx;
  m(1, 0) = c; m(1, 1) = d; m(1, 2) = ty;
  return m;
}

// A small plan with two frames: one affine-only, one with a local model.
RegistrationSamplingPlan make_sample_plan() {
  RegistrationSamplingPlan p;
  p.source_width = 3840;
  p.source_height = 2160;
  p.canvas_width_native = 3926;
  p.canvas_height_native = 2312;
  p.canvas_offset_x_native = 42;
  p.canvas_offset_y_native = 76;
  p.internal_scale = 2;
  p.output_scale = 1;
  p.color_mode = ColorMode::OSC;
  p.bayer_pattern = BayerPattern::RGGB;
  p.cfa_origin_x = 0;
  p.cfa_origin_y = 0;

  FrameSamplingTransform f0;
  f0.frame_id = "frame-0000-abcdef";
  f0.source_index = 0;
  f0.valid = true;
  // small rotation + translation
  const float th = 0.01f;
  f0.canvas_to_source =
      make_affine(std::cos(th), -std::sin(th), 12.5f,
                  std::sin(th), std::cos(th), -7.25f);
  REQUIRE(invert_affine_2x3(f0.canvas_to_source, 0.80f, 1.25f,
                            f0.source_to_canvas));
  f0.source_to_canvas_affine_valid = true;
  f0.registration_residual_factor = 0.87f;
  f0.model_prediction_factor = 1.0f;
  f0.chain_depth = 0;
  f0.provenance = "direct_global";
  p.frames.push_back(f0);

  FrameSamplingTransform f1;
  f1.frame_id = "frame-0001-123456";
  f1.source_index = 1;
  f1.valid = true;
  f1.canvas_to_source = make_affine(1.0f, 0.0f, -3.0f, 0.0f, 1.0f, 5.0f);
  REQUIRE(invert_affine_2x3(f1.canvas_to_source, 0.80f, 1.25f,
                            f1.source_to_canvas));
  f1.source_to_canvas_affine_valid = true;
  f1.has_smooth_local_model = true;
  f1.smooth_local_model.valid = true;
  f1.smooth_local_model.image_rows = 2160;
  f1.smooth_local_model.image_cols = 3840;
  for (int i = 0; i < f1.smooth_local_model.coeff_x.size(); ++i) {
    f1.smooth_local_model.coeff_x[i] = 0.001f * static_cast<float>(i + 1);
    f1.smooth_local_model.coeff_y[i] = -0.002f * static_cast<float>(i + 1);
  }
  f1.model_coordinate_scale = 1.0f;
  f1.model_offset_x = 42.0f;
  f1.model_offset_y = 76.0f;
  f1.registration_residual_factor = 0.72f;
  f1.model_prediction_factor = 0.75f;
  f1.model_predicted = true;
  f1.chain_depth = 2;
  f1.provenance = "sequential_refined";
  p.frames.push_back(f1);

  p.plan_hash = compute_plan_hash(p);
  return p;
}

}  // namespace

TEST_CASE("affine 2x3 inversion round-trips a point (plan 7.2)") {
  const WarpMatrix c2s = make_affine(std::cos(0.03f), -std::sin(0.03f), 10.0f,
                                     std::sin(0.03f), std::cos(0.03f), -4.0f);
  WarpMatrix s2c;
  REQUIRE(invert_affine_2x3(c2s, 0.80f, 1.25f, s2c));

  for (float qx : {0.0f, 100.0f, 3925.0f}) {
    for (float qy : {0.0f, 250.0f, 2311.0f}) {
      const float sx = c2s(0, 0) * qx + c2s(0, 1) * qy + c2s(0, 2);
      const float sy = c2s(1, 0) * qx + c2s(1, 1) * qy + c2s(1, 2);
      const float rqx = s2c(0, 0) * sx + s2c(0, 1) * sy + s2c(0, 2);
      const float rqy = s2c(1, 0) * sx + s2c(1, 1) * sy + s2c(1, 2);
      REQUIRE(rqx == Approx(qx).margin(1e-2));
      REQUIRE(rqy == Approx(qy).margin(1e-2));
    }
  }
}

TEST_CASE("affine inversion rejects singular / out-of-bounds matrices (plan 7.2)") {
  WarpMatrix out;
  // singular: second row is a multiple of the first
  REQUIRE_FALSE(invert_affine_2x3(make_affine(1, 2, 0, 2, 4, 0),
                                  0.80f, 1.25f, out));
  // determinant far above scale bounds (2x scale -> det 4)
  REQUIRE_FALSE(invert_affine_2x3(make_affine(2, 0, 0, 0, 2, 0),
                                  0.80f, 1.25f, out));
  // reflection (negative determinant) is rejected by design
  REQUIRE_FALSE(invert_affine_2x3(make_affine(1, 0, 0, 0, -1, 0),
                                  0.80f, 1.25f, out));
  // non-finite coefficient
  REQUIRE_FALSE(invert_affine_2x3(
      make_affine(std::nanf(""), 0, 0, 0, 1, 0), 0.80f, 1.25f, out));
  // identity is fine
  REQUIRE(invert_affine_2x3(make_affine(1, 0, 0, 0, 1, 0), 0.80f, 1.25f, out));
}

TEST_CASE("serialization is lossless (plan 7.4 / 20.1)") {
  const RegistrationSamplingPlan p = make_sample_plan();
  const std::string text = serialize_to_json_string(p);

  RegistrationSamplingPlan q;
  std::string err;
  REQUIRE(parse_from_json_string(text, q, err));
  REQUIRE(err.empty());

  REQUIRE(q.source_width == p.source_width);
  REQUIRE(q.canvas_offset_x_native == p.canvas_offset_x_native);
  REQUIRE(q.internal_scale == p.internal_scale);
  REQUIRE(q.output_scale == p.output_scale);
  REQUIRE(q.color_mode == p.color_mode);
  REQUIRE(q.bayer_pattern == p.bayer_pattern);
  REQUIRE(q.frames.size() == p.frames.size());

  for (std::size_t i = 0; i < p.frames.size(); ++i) {
    const auto& a = p.frames[i];
    const auto& b = q.frames[i];
    REQUIRE(b.frame_id == a.frame_id);
    REQUIRE(b.source_index == a.source_index);
    REQUIRE(b.valid == a.valid);
    REQUIRE(b.source_to_canvas_affine_valid == a.source_to_canvas_affine_valid);
    REQUIRE(b.has_smooth_local_model == a.has_smooth_local_model);
    REQUIRE(b.chain_depth == a.chain_depth);
    REQUIRE(b.provenance == a.provenance);
    REQUIRE(b.model_coordinate_scale == Approx(a.model_coordinate_scale));
    REQUIRE(b.model_prediction_factor == Approx(a.model_prediction_factor));
    REQUIRE(b.registration_residual_factor ==
            Approx(a.registration_residual_factor));
    for (int r = 0; r < 2; ++r)
      for (int c = 0; c < 3; ++c) {
        REQUIRE(b.canvas_to_source(r, c) == Approx(a.canvas_to_source(r, c)));
        REQUIRE(b.source_to_canvas(r, c) == Approx(a.source_to_canvas(r, c)));
      }
    for (int k = 0; k < a.smooth_local_model.coeff_x.size(); ++k) {
      REQUIRE(b.smooth_local_model.coeff_x[k] ==
              Approx(a.smooth_local_model.coeff_x[k]));
      REQUIRE(b.smooth_local_model.coeff_y[k] ==
              Approx(a.smooth_local_model.coeff_y[k]));
    }
  }

  // Round-trip preserves the hash exactly.
  REQUIRE(compute_plan_hash(q) == compute_plan_hash(p));
  REQUIRE(compute_plan_hash(q) == p.plan_hash);
}

TEST_CASE("plan_hash is stable against diagnostic-only changes (plan 7.4 / 20.1)") {
  const RegistrationSamplingPlan base = make_sample_plan();
  const std::string h0 = compute_plan_hash(base);

  SECTION("provenance string does not change the hash") {
    RegistrationSamplingPlan p = base;
    p.frames[0].provenance = "totally-different-provenance";
    p.frames[1].provenance = "";
    REQUIRE(compute_plan_hash(p) == h0);
  }
  SECTION("chain_depth does not change the hash") {
    RegistrationSamplingPlan p = base;
    p.frames[1].chain_depth = 99;
    REQUIRE(compute_plan_hash(p) == h0);
  }
  SECTION("model_predicted flag does not change the hash") {
    RegistrationSamplingPlan p = base;
    p.frames[0].model_predicted = true;
    REQUIRE(compute_plan_hash(p) == h0);
  }
  SECTION("internal_scale / output_scale are not part of plan_hash") {
    RegistrationSamplingPlan p = base;
    p.internal_scale = 1;
    p.output_scale = 2;
    REQUIRE(compute_plan_hash(p) == h0);
  }
}

TEST_CASE("plan_hash changes on semantic changes (plan 7.4 / 20.1)") {
  const RegistrationSamplingPlan base = make_sample_plan();
  const std::string h0 = compute_plan_hash(base);

  SECTION("canvas offset") {
    RegistrationSamplingPlan p = base;
    p.canvas_offset_x_native += 1;
    REQUIRE(compute_plan_hash(p) != h0);
  }
  SECTION("bayer pattern") {
    RegistrationSamplingPlan p = base;
    p.bayer_pattern = BayerPattern::GBRG;
    REQUIRE(compute_plan_hash(p) != h0);
  }
  SECTION("cfa origin parity") {
    RegistrationSamplingPlan p = base;
    p.cfa_origin_x = 1;
    REQUIRE(compute_plan_hash(p) != h0);
  }
  SECTION("frame reordering") {
    RegistrationSamplingPlan p = base;
    std::swap(p.frames[0], p.frames[1]);
    REQUIRE(compute_plan_hash(p) != h0);
  }
  SECTION("a warp coefficient") {
    RegistrationSamplingPlan p = base;
    p.frames[0].canvas_to_source(0, 2) += 0.5f;
    REQUIRE(compute_plan_hash(p) != h0);
  }
  SECTION("a local-model coefficient") {
    RegistrationSamplingPlan p = base;
    p.frames[1].smooth_local_model.coeff_x[3] += 1e-4f;
    REQUIRE(compute_plan_hash(p) != h0);
  }
  SECTION("model coordinate scale") {
    RegistrationSamplingPlan p = base;
    p.frames[1].model_coordinate_scale = 2.0f;
    REQUIRE(compute_plan_hash(p) != h0);
  }
  SECTION("registration residual factor") {
    RegistrationSamplingPlan p = base;
    p.frames[0].registration_residual_factor = 0.5f;
    REQUIRE(compute_plan_hash(p) != h0);
  }
  SECTION("frame validity") {
    RegistrationSamplingPlan p = base;
    p.frames[0].valid = false;
    REQUIRE(compute_plan_hash(p) != h0);
  }
}

TEST_CASE("2x scaling does not change native warp semantics (plan 20.1)") {
  // internal_scale is not part of plan_hash, and the native warps are unchanged,
  // so a plan rendered at 1x and 2x must hash identically.
  RegistrationSamplingPlan p1 = make_sample_plan();
  p1.internal_scale = 1;
  p1.output_scale = 1;
  RegistrationSamplingPlan p2 = p1;
  p2.internal_scale = 2;
  p2.output_scale = 2;
  REQUIRE(compute_plan_hash(p1) == compute_plan_hash(p2));
}

TEST_CASE("local source->canvas inversion converges for identity local model "
          "(plan 7.3)") {
  // With a zero local model the inversion must reproduce the affine inverse.
  RegistrationSamplingPlan p = make_sample_plan();
  FrameSamplingTransform& f = p.frames[1];
  for (int i = 0; i < f.smooth_local_model.coeff_x.size(); ++i) {
    f.smooth_local_model.coeff_x[i] = 0.0f;
    f.smooth_local_model.coeff_y[i] = 0.0f;
  }

  LocalInversionParams params;
  for (float sx : {50.0f, 1920.0f, 3800.0f}) {
    for (float sy : {50.0f, 1080.0f, 2100.0f}) {
      float qx = 0.0f, qy = 0.0f;
      REQUIRE(invert_local_source_to_canvas(f, sx, sy, p.canvas_width_native,
                                            p.canvas_height_native, params,
                                            qx, qy));
      const float ux = f.source_to_canvas(0, 0) * sx +
                       f.source_to_canvas(0, 1) * sy + f.source_to_canvas(0, 2);
      const float uy = f.source_to_canvas(1, 0) * sx +
                       f.source_to_canvas(1, 1) * sy + f.source_to_canvas(1, 2);
      REQUIRE(qx == Approx(ux).margin(1e-3));
      REQUIRE(qy == Approx(uy).margin(1e-3));
    }
  }
}

TEST_CASE("local inversion fails deterministically on an invalid coordinate "
          "scale (plan 7.3)") {
  RegistrationSamplingPlan p = make_sample_plan();
  FrameSamplingTransform f = p.frames[1];
  f.model_coordinate_scale = 0.0f;  // broken local model
  float qx = 0.0f, qy = 0.0f;
  REQUIRE_FALSE(invert_local_source_to_canvas(
      f, 1920.0f, 1080.0f, p.canvas_width_native, p.canvas_height_native,
      LocalInversionParams{}, qx, qy));
}

TEST_CASE("sampling audit: OpenCV center adapter preserves physical control points", "[drizzle-audit]") {
  auto cv=make_affine(0.8f,-0.6f,12,0.6f,0.8f,-3);
  auto edge=opencv_to_edge_sampling_map(cv);
  for(float x:{0.0f,7.3f,30.0f}) for(float y:{0.0f,2.7f,21.0f}) {
    const float sx=cv(0,0)*x+cv(0,1)*y+cv(0,2);
    const float sy=cv(1,0)*x+cv(1,1)*y+cv(1,2);
    REQUIRE(edge(0,0)*(x+0.5f)+edge(0,1)*(y+0.5f)+edge(0,2)==Approx(sx+0.5f).margin(1e-5));
    REQUIRE(edge(1,0)*(x+0.5f)+edge(1,1)*(y+0.5f)+edge(1,2)==Approx(sy+0.5f).margin(1e-5));
  }
}

TEST_CASE("sampling audit: loader rejects missing fields stale hashes and inconsistent inverses", "[drizzle-audit]") {
  auto plan=make_sample_plan();
  plan.plan_hash=compute_plan_hash(plan);
  auto valid=nlohmann::json::parse(serialize_to_json_string(plan));
  RegistrationSamplingPlan loaded;std::string error;
  REQUIRE(parse_from_json_string(valid.dump(),loaded,error));
  auto missing=valid;missing.erase("source_width");
  REQUIRE_FALSE(parse_from_json_string(missing.dump(),loaded,error));
  auto future=valid;future["schema_version"]=999;
  REQUIRE_FALSE(parse_from_json_string(future.dump(),loaded,error));
  auto changed=valid;changed["source_identity_hash"]="changed cache/source";
  REQUIRE_FALSE(parse_from_json_string(changed.dump(),loaded,error));
  plan.frames[0].source_to_canvas(0,2)+=1;
  plan.plan_hash=compute_plan_hash(plan);
  REQUIRE_FALSE(parse_from_json_string(serialize_to_json_string(plan),loaded,error));
}
