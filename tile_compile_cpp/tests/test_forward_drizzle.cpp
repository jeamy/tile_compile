// M2 tests for the CFA-aware forward-drizzle CPU reference (Uniform-Control
// only; docs/AQMH/aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md
// section 11).

#include "tile_compile/reconstruction/forward_drizzle.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

using namespace tile_compile;
using namespace tile_compile::reconstruction;
using Catch::Approx;
using tile_compile::registration::FrameSamplingTransform;
using tile_compile::registration::RegistrationSamplingPlan;

namespace {

WarpMatrix make_source_to_canvas(double a, double b, double tx, double c,
                                 double d, double ty) {
  WarpMatrix m;
  m(0, 0) = static_cast<float>(a);
  m(0, 1) = static_cast<float>(b);
  m(0, 2) = static_cast<float>(tx);
  m(1, 0) = static_cast<float>(c);
  m(1, 1) = static_cast<float>(d);
  m(1, 2) = static_cast<float>(ty);
  return m;
}

FrameSamplingTransform make_affine_frame(const std::string &id, size_t idx,
                                         const WarpMatrix &source_to_canvas) {
  FrameSamplingTransform f;
  f.frame_id = id;
  f.source_index = idx;
  f.valid = true;
  f.source_to_canvas = source_to_canvas;
  f.source_to_canvas_affine_valid = true;
  return f;
}

// Single 1x1-pixel-source plan, one frame, centered well away from the
// canvas border so the droplet is never boundary-clipped.
RegistrationSamplingPlan single_sample_plan(ColorMode mode) {
  RegistrationSamplingPlan plan;
  plan.source_width = 1;
  plan.source_height = 1;
  plan.canvas_width_native = 40;
  plan.canvas_height_native = 40;
  plan.color_mode = mode;
  return plan;
}

} // namespace

TEST_CASE("forward drizzle: affine area identity holds for translation, "
          "rotation, scale and combined warps (plan 11.6)") {
  struct Case {
    std::string name;
    double a, b, c, d;
  };
  const double deg30 = 30.0 * M_PI / 180.0;
  const double deg20 = 20.0 * M_PI / 180.0;
  const std::vector<Case> cases = {
      {"translation", 1.0, 0.0, 0.0, 1.0},
      {"rotation30", std::cos(deg30), -std::sin(deg30), std::sin(deg30),
       std::cos(deg30)},
      {"scale1.5", 1.5, 0.0, 0.0, 1.5},
      {"rotation20+scale1.5", 1.5 * std::cos(deg20), -1.5 * std::sin(deg20),
       1.5 * std::sin(deg20), 1.5 * std::cos(deg20)},
  };

  for (const auto &tc : cases) {
    for (int internal_scale : {1, 2}) {
      DYNAMIC_SECTION(tc.name << " internal_scale=" << internal_scale) {
        RegistrationSamplingPlan plan = single_sample_plan(ColorMode::MONO);
        plan.frames.push_back(make_affine_frame(
            "f0", 0,
            make_source_to_canvas(tc.a, tc.b, 20.0, tc.c, tc.d, 20.0)));

        config::ReconstructionDrizzleConfig cfg;
        cfg.internal_scale = internal_scale;
        cfg.pixfrac = 0.8f;

        Matrix2Df img(1, 1);
        img(0, 0) = 5.0f; // value is irrelevant to the area identity (checked
                          // via B, not A)
        SourceImageProvider source_of = [&](std::size_t) -> const Matrix2Df & {
          return img;
        };

        auto result = compute_forward_drizzle_uniform(plan, source_of, cfg);

        double sum_b = 0.0;
        for (float w : result.L.weight_sum)
          sum_b += w;

        const double det = tc.a * tc.d - tc.b * tc.c;
        const double expected = static_cast<double>(cfg.pixfrac) * cfg.pixfrac *
                                static_cast<double>(internal_scale) *
                                internal_scale * std::abs(det);
        REQUIRE(sum_b == Approx(expected).epsilon(1e-5));
      }
    }
  }
}

TEST_CASE("forward drizzle: OSC CFA colour segregation matches CFA origin "
          "(plan 11.4)") {
  RegistrationSamplingPlan plan;
  plan.source_width = 2;
  plan.source_height = 2;
  plan.canvas_width_native = 20;
  plan.canvas_height_native = 20;
  plan.color_mode = ColorMode::OSC;
  plan.bayer_pattern = BayerPattern::RGGB; // (0,0)=R (1,0)=G (0,1)=G (1,1)=B
  plan.cfa_origin_x = 0;
  plan.cfa_origin_y = 0;
  plan.frames.push_back(
      make_affine_frame("f0", 0, make_source_to_canvas(1, 0, 8, 0, 1, 8)));

  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = 1;
  cfg.pixfrac = 0.8f;

  Matrix2Df img(2, 2);
  img(0, 0) = 100.0f; // R
  img(0, 1) = 200.0f; // G (x odd, y even)
  img(1, 0) = 300.0f; // G (x even, y odd)
  img(1, 1) = 400.0f; // B
  SourceImageProvider source_of = [&](std::size_t) -> const Matrix2Df & {
    return img;
  };

  auto result = compute_forward_drizzle_uniform(plan, source_of, cfg);

  REQUIRE_FALSE(result.R.empty());
  REQUIRE_FALSE(result.G.empty());
  REQUIRE_FALSE(result.B.empty());
  REQUIRE(result.L.empty()); // MONO plane must stay absent for OSC (plan 11.4)

  auto max_value = [](const ProfilePlane &p) {
    float m = 0.0f;
    for (size_t i = 0; i < p.value.size(); ++i)
      if (p.support[i] && p.value[i] > m)
        m = p.value[i];
    return m;
  };
  REQUIRE(max_value(result.R) == Approx(100.0).epsilon(1e-3));
  REQUIRE(max_value(result.B) == Approx(400.0).epsilon(1e-3));
  // Both G sites (200, 300) accumulate into the same green channel (plan 11.4).
  float g_total_weighted = 0.0f;
  for (size_t i = 0; i < result.G.value.size(); ++i) {
    if (result.G.support[i])
      g_total_weighted += result.G.value[i] * result.G.weight_sum[i];
  }
  REQUIRE(g_total_weighted > 0.0f);
}

TEST_CASE("forward drizzle: MONO path fills only L, never copies into R/G/B "
          "(plan 11.4)") {
  RegistrationSamplingPlan plan = single_sample_plan(ColorMode::MONO);
  plan.frames.push_back(
      make_affine_frame("f0", 0, make_source_to_canvas(1, 0, 20, 0, 1, 20)));
  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = 1;
  cfg.pixfrac = 0.8f;
  Matrix2Df img(1, 1);
  img(0, 0) = 7.0f;
  SourceImageProvider source_of = [&](std::size_t) -> const Matrix2Df & {
    return img;
  };
  auto result = compute_forward_drizzle_uniform(plan, source_of, cfg);
  REQUIRE_FALSE(result.L.empty());
  REQUIRE(result.R.empty());
  REQUIRE(result.G.empty());
  REQUIRE(result.B.empty());
}

TEST_CASE(
    "forward drizzle: two identically-registered frames average their "
    "values (frame-local aggregation + Uniform-Control, plan 11.7/11.9)") {
  RegistrationSamplingPlan plan = single_sample_plan(ColorMode::MONO);
  plan.frames.push_back(
      make_affine_frame("f0", 0, make_source_to_canvas(1, 0, 20, 0, 1, 20)));
  plan.frames.push_back(
      make_affine_frame("f1", 1, make_source_to_canvas(1, 0, 20, 0, 1, 20)));

  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = 1;
  cfg.pixfrac = 0.8f;

  Matrix2Df img0(1, 1), img1(1, 1);
  img0(0, 0) = 10.0f;
  img1(0, 0) = 30.0f;
  SourceImageProvider source_of = [&](std::size_t idx) -> const Matrix2Df & {
    return idx == 0 ? img0 : img1;
  };

  auto result = compute_forward_drizzle_uniform(plan, source_of, cfg);
  float peak = 0.0f;
  float peak_n_eff = 0.0f;
  for (size_t i = 0; i < result.L.value.size(); ++i) {
    if (result.L.support[i] && result.L.weight_sum[i] > peak) {
      peak = result.L.value[i];
      peak_n_eff = result.L.n_eff[i];
    }
  }
  // Identical geometry -> identical B_f,c(q) for both frames -> plain average.
  REQUIRE(peak == Approx(20.0).epsilon(1e-3));
  // n_eff = (sum w)^2 / sum w^2 == 2 for two equal weights (plan 11.10).
  REQUIRE(peak_n_eff == Approx(2.0).epsilon(1e-2));
}

TEST_CASE("forward drizzle: local-warp subdivision with zero displacement "
          "reproduces the affine area exactly (plan 11.6)") {
  RegistrationSamplingPlan plan = single_sample_plan(ColorMode::MONO);
  FrameSamplingTransform f =
      make_affine_frame("f0", 0, make_source_to_canvas(1, 0, 20, 0, 1, 20));
  f.has_smooth_local_model = true;
  f.smooth_local_model.valid = true;
  f.smooth_local_model.image_rows = 40;
  f.smooth_local_model.image_cols = 40;
  // coeff_x/coeff_y default-constructed to zero: d(q) == 0 everywhere, so the
  // local model is exactly the affine seed.
  f.model_coordinate_scale = 1.0f;
  plan.frames.push_back(f);

  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = 1;
  cfg.pixfrac = 0.8f;
  Matrix2Df img(1, 1);
  img(0, 0) = 5.0f;
  SourceImageProvider source_of = [&](std::size_t) -> const Matrix2Df & {
    return img;
  };

  auto result = compute_forward_drizzle_uniform(plan, source_of, cfg);
  double sum_b = 0.0;
  for (float w : result.L.weight_sum)
    sum_b += w;
  REQUIRE(sum_b == Approx(0.8 * 0.8).epsilon(5e-3));
  REQUIRE(result.diagnostics.local_model_samples_discarded == 0);
  REQUIRE(result.diagnostics.frames_excluded_subdivision_error_rate.empty());
}

TEST_CASE("forward drizzle: a frame whose local model cannot be inverted at "
          "all is excluded, not silently degraded (plan 11.6)") {
  RegistrationSamplingPlan plan = single_sample_plan(ColorMode::MONO);
  FrameSamplingTransform f =
      make_affine_frame("f0", 0, make_source_to_canvas(1, 0, 20, 0, 1, 20));
  f.has_smooth_local_model = true;
  f.smooth_local_model.valid = true;
  f.smooth_local_model.image_rows = 40;
  f.smooth_local_model.image_cols = 40;
  f.model_coordinate_scale =
      0.0f; // broken: invert_local_source_to_canvas fails
  plan.frames.push_back(f);

  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = 1;
  cfg.pixfrac = 0.8f;
  Matrix2Df img(1, 1);
  img(0, 0) = 5.0f;
  SourceImageProvider source_of = [&](std::size_t) -> const Matrix2Df & {
    return img;
  };

  auto result = compute_forward_drizzle_uniform(plan, source_of, cfg);
  REQUIRE(result.diagnostics.frames_excluded_subdivision_error_rate.size() ==
          1);
  REQUIRE(result.diagnostics.frames_excluded_subdivision_error_rate[0].first ==
          "f0");
  // Fully excluded -> no support anywhere.
  for (uint8_t s : result.L.support)
    REQUIRE(s == 0);
}

TEST_CASE("polygon_rectangle_intersection_area: axis-aligned square fully "
          "inside one pixel") {
  const double px[4] = {2.1, 2.9, 2.9, 2.1};
  const double py[4] = {2.1, 2.1, 2.9, 2.9};
  const double area =
      polygon_rectangle_intersection_area(px, py, 2.0, 2.0, 3.0, 3.0);
  REQUIRE(area == Approx(0.8 * 0.8).epsilon(1e-9));
}

TEST_CASE("polygon_rectangle_intersection_area: square straddling two pixels "
          "splits proportionally") {
  // A 1x1 square centered on the vertical boundary x=3, spanning x in
  // [2.5,3.5].
  const double px[4] = {2.5, 3.5, 3.5, 2.5};
  const double py[4] = {0.0, 0.0, 1.0, 1.0};
  const double left =
      polygon_rectangle_intersection_area(px, py, 2.0, 0.0, 3.0, 1.0);
  const double right =
      polygon_rectangle_intersection_area(px, py, 3.0, 0.0, 4.0, 1.0);
  REQUIRE(left == Approx(0.5).epsilon(1e-9));
  REQUIRE(right == Approx(0.5).epsilon(1e-9));
  REQUIRE((left + right) == Approx(1.0).epsilon(1e-9));
}

TEST_CASE("polygon_rectangle_intersection_area: no overlap returns exactly 0") {
  const double px[4] = {10.0, 11.0, 11.0, 10.0};
  const double py[4] = {10.0, 10.0, 11.0, 11.0};
  REQUIRE(polygon_rectangle_intersection_area(px, py, 0.0, 0.0, 1.0, 1.0) ==
          0.0);
}

TEST_CASE(
    "forward drizzle: stripe boundaries preserve rotated scaled OSC values",
    "[drizzle-audit]") {
  RegistrationSamplingPlan plan;
  plan.source_width = 13;
  plan.source_height = 11;
  plan.canvas_width_native = 40;
  plan.canvas_height_native = 40;
  plan.color_mode = ColorMode::OSC;
  plan.bayer_pattern = BayerPattern::GBRG;
  plan.cfa_origin_x = 1;
  plan.cfa_origin_y = 1;
  const double angle = 0.7853981633974483;
  for (size_t i = 0; i < 3; ++i)
    plan.frames.push_back(make_affine_frame(
        "f" + std::to_string(i), i,
        make_source_to_canvas(1.2 * std::cos(angle), -1.2 * std::sin(angle),
                              20 + i * 0.13, 1.2 * std::sin(angle),
                              1.2 * std::cos(angle), 4 + i * 0.27)));
  Matrix2Df source(11, 13);
  for (int y = 0; y < 11; ++y)
    for (int x = 0; x < 13; ++x)
      source(y, x) = x - y * 0.3f;
  source(4, 4) = std::numeric_limits<float>::quiet_NaN();
  SourceImageProvider provider = [&](size_t) -> const Matrix2Df & {
    return source;
  };
  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = 2;
  cfg.chunk_rows = 80;
  auto full = compute_forward_drizzle_uniform(plan, provider, cfg);
  cfg.chunk_rows = 1;
  auto striped = compute_forward_drizzle_uniform(plan, provider, cfg);
  auto compare = [](const ProfilePlane &a, const ProfilePlane &b) {
    REQUIRE(a.support == b.support);
    REQUIRE(a.weight_sum == b.weight_sum);
    REQUIRE(a.n_eff == b.n_eff);
    for (size_t i = 0; i < a.value.size(); ++i)
      if (a.support[i])
        REQUIRE(a.value[i] == b.value[i]);
  };
  compare(full.R, striped.R);
  compare(full.G, striped.G);
  compare(full.B, striped.B);
}

TEST_CASE(
    "forward drizzle: budget rejects huge materialization before source IO",
    "[drizzle-audit]") {
  RegistrationSamplingPlan plan;
  plan.source_width = 6000;
  plan.source_height = 4000;
  plan.canvas_width_native = 6000;
  plan.canvas_height_native = 4000;
  plan.color_mode = ColorMode::OSC;
  plan.bayer_pattern = BayerPattern::RGGB;
  config::ReconstructionDrizzleConfig cfg;
  cfg.memory_budget_mb = 512;
  const auto memory = plan_drizzle_memory(plan, cfg, 3 * 53);
  REQUIRE(memory.rows >= 1);
  REQUIRE(memory.rows < 8000);
  REQUIRE(memory.estimated_peak_bytes <= 512ull * 1024 * 1024);
  bool loaded = false;
  Matrix2Df empty;
  SourceImageProvider provider = [&](size_t) -> const Matrix2Df & {
    loaded = true;
    return empty;
  };
  REQUIRE_THROWS_WITH(
      compute_forward_drizzle_uniform(plan, provider, cfg),
      "DRIZZLE_MEMORY_BUDGET: retained/source buffers exceed budget");
  REQUIRE_FALSE(loaded);
  cfg.chunk_rows = 8000;
  REQUIRE_THROWS(plan_drizzle_memory(plan, cfg, 3 * 53));
}

TEST_CASE(
    "forward drizzle: bounded streaming succeeds when full output cannot fit",
    "[drizzle-audit][drizzle-memory]") {
  RegistrationSamplingPlan plan;
  plan.source_width = 512;
  plan.source_height = 512;
  plan.canvas_width_native = 512;
  plan.canvas_height_native = 512;
  plan.color_mode = ColorMode::OSC;
  plan.bayer_pattern = BayerPattern::RGGB;
  plan.frames.push_back(
      make_affine_frame("f0", 0, make_source_to_canvas(1, 0, 0, 0, 1, 0)));
  Matrix2Df source = Matrix2Df::Constant(512, 512, -7.0f);
  SourceImageProvider provider = [&](size_t) -> const Matrix2Df & {
    return source;
  };
  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = 1;
  cfg.memory_budget_mb = 8;
  REQUIRE_THROWS(compute_forward_drizzle_uniform(plan, provider, cfg));
  size_t pixels = 0;
  int next_row = 0;
  const auto diagnostics = stream_forward_drizzle_uniform(
      plan, provider, cfg,
      [&](int y, const ForwardDrizzleUniformResult &stripe) {
        REQUIRE(y == next_row);
        next_row += stripe.internal_height;
        for (const auto *p : {&stripe.R, &stripe.G, &stripe.B})
          for (size_t i = 0; i < p->value.size(); ++i)
            if (p->support[i]) {
              ++pixels;
              if (p->value[i] != -7.0f)
                FAIL("constant surface brightness changed");
            }
      });
  REQUIRE(pixels == 512 * 512);
  REQUIRE(next_row == 512);
  REQUIRE(diagnostics.estimated_peak_bytes <= 8 * 1024 * 1024);
  REQUIRE(diagnostics.resolved_chunk_rows < 512);
}

TEST_CASE("forward drizzle: local rejection counts source samples once",
          "[drizzle-audit]") {
  auto plan = single_sample_plan(ColorMode::MONO);
  auto f =
      make_affine_frame("f0", 0, make_source_to_canvas(1, 0, 20, 0, 1, 20));
  f.has_smooth_local_model = true;
  f.smooth_local_model.valid = true;
  f.smooth_local_model.image_rows = 40;
  f.smooth_local_model.image_cols = 40;
  f.smooth_local_model.coeff_x.setZero();
  f.smooth_local_model.coeff_y.setZero();
  f.smooth_local_model.coeff_x[0] = 2.0f;
  f.smooth_local_model.coeff_y[4] = 0.7f;
  plan.frames.push_back(f);
  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = 2;
  ForwardDrizzleSubdivisionParams sub;
  sub.position_epsilon_internal_px =
      100; // isolate area convergence at depth zero
  sub.area_relative_epsilon = 0;
  sub.max_subdivision_depth = 0;
  auto prepared = prepare_drizzle_frames(plan, cfg, sub);
  REQUIRE(prepared.diagnostics.local_model_samples_total == 1);
  REQUIRE(prepared.diagnostics.local_model_samples_discarded == 1);
  REQUIRE(prepared.frames.empty());
  REQUIRE(
      prepared.diagnostics.frames_excluded_subdivision_error_rate[0].second ==
      1.0);
  sub.area_relative_epsilon = 0.005f;
  sub.max_subdivision_depth = 2;
  sub.position_epsilon_internal_px = 0.05f;
  auto accepted = prepare_drizzle_frames(plan, cfg, sub);
  REQUIRE(accepted.frames.size() == 1);
}

TEST_CASE("forward drizzle: aperture flux and centroid survive fractional "
          "shifts at 1x and 2x",
          "[drizzle-audit]") {
  RegistrationSamplingPlan plan;
  plan.source_width = 24;
  plan.source_height = 24;
  plan.canvas_width_native = 48;
  plan.canvas_height_native = 48;
  plan.color_mode = ColorMode::MONO;
  plan.frames.push_back(make_affine_frame(
      "f0", 0, make_source_to_canvas(1, 0, 10.25, 0, 1, 9.5)));
  Matrix2Df source = Matrix2Df::Zero(24, 24);
  source(9, 8) = 100;
  SourceImageProvider provider = [&](size_t) -> const Matrix2Df & {
    return source;
  };
  for (int scale : {1, 2}) {
    config::ReconstructionDrizzleConfig cfg;
    cfg.internal_scale = scale;
    cfg.pixfrac = 1;
    cfg.chunk_rows = 3;
    auto result = compute_forward_drizzle_uniform(plan, provider, cfg);
    double flux = 0, mx = 0, my = 0;
    for (int y = 0; y < result.internal_height; ++y)
      for (int x = 0; x < result.internal_width; ++x) {
        const size_t i = static_cast<size_t>(y) * result.internal_width + x;
        if (!result.L.support[i])
          continue;
        const double v = result.L.value[i] / (scale * scale);
        flux += v;
        mx += v * (x + 0.5) / scale;
        my += v * (y + 0.5) / scale;
      }
    REQUIRE(flux == Approx(100).epsilon(1e-6));
    REQUIRE(mx / flux == Approx(18.75).epsilon(1e-6));
    REQUIRE(my / flux == Approx(19.0).epsilon(1e-6));
  }
}

// --- M3 (plan 11.8): shared robust clipping ---------------------------------

namespace {
std::vector<ClipCandidate> make_candidates(const std::vector<double> &values) {
  std::vector<ClipCandidate> c;
  c.reserve(values.size());
  for (size_t i = 0; i < values.size(); ++i) c.push_back({i, values[i], 1.0});
  return c;
}
}  // namespace

TEST_CASE("robust clipping: below min_clip_contributors, nothing is clipped "
          "(plan 11.8 step 2)") {
  auto candidates = make_candidates({10, 10, 10, 1000});  // an obvious outlier
  auto result = apply_robust_clipping(candidates, /*min_clip_contributors=*/5,
                                      /*robust_passes=*/3, 3.0f, 3.0f, 0.0f, 0.0f);
  REQUIRE(result.accepted.size() == 4);
  for (bool a : result.accepted) REQUIRE(a);
  REQUIRE_FALSE(result.pixel_rejected);
}

TEST_CASE("robust clipping: a clear outlier is excluded via degenerate-MAD "
          "guard (plan 11.8 steps 4-6)") {
  auto candidates = make_candidates({10, 10, 10, 10, 100});
  auto result = apply_robust_clipping(candidates, /*min_clip_contributors=*/5,
                                      /*robust_passes=*/3, 3.0f, 3.0f, 0.0f, 0.0f);
  REQUIRE(result.accepted == std::vector<bool>{true, true, true, true, false});
}

TEST_CASE("robust clipping: identical values all stay valid, no arbitrary "
          "epsilon clipping of a constant background (plan 11.8)") {
  auto candidates = make_candidates({5, 5, 5, 5, 5});
  auto result = apply_robust_clipping(candidates, 5, 3, 3.0f, 3.0f, 0.0f, 0.0f);
  for (bool a : result.accepted) REQUIRE(a);
}

TEST_CASE("robust clipping: asymmetric sigma bounds clip only on the "
          "configured side (plan 11.8 step 5)") {
  // Single pass in isolation (robust_passes=1): median=12, mad=1 (both
  // hand-verified), bounds = [12 - 5*1, 12 + 0.5*1] = [7, 12.5]. 13 and 14
  // fall outside the tight upper bound; the loose lower bound keeps 10, 11.
  // (A further pass would keep re-tightening as the survivor set shrinks ---
  // that cascading behaviour is covered separately by the multi-pass test
  // below, not conflated with this single-pass side-selectivity check.)
  auto candidates = make_candidates({10, 11, 12, 13, 14});
  auto result = apply_robust_clipping(candidates, 5, /*robust_passes=*/1,
                                      /*sigma_low=*/5.0f, /*sigma_high=*/0.5f,
                                      0.0f, 0.0f);
  REQUIRE(result.accepted == std::vector<bool>{true, true, true, false, false});
}

TEST_CASE("robust clipping: a single pass is not enough when a wide-spread "
          "outlier group inflates MAD, but a second pass catches the "
          "residual (plan 11.8 step 7)") {
  // Good cluster {10,11,12,13}; a moderate anomaly (18) that survives pass 1
  // only because two far outliers (60,62) inflate MAD; pass 2 recomputes
  // without them and catches 18. Hand-verified: see plan-doc progress notes.
  auto candidates = make_candidates({10, 11, 12, 13, 18, 60, 62});

  auto one_pass = apply_robust_clipping(candidates, 4, /*robust_passes=*/1,
                                        3.0f, 3.0f, 0.0f, 0.0f);
  REQUIRE(one_pass.accepted ==
          std::vector<bool>{true, true, true, true, true, false, false});

  auto two_passes = apply_robust_clipping(candidates, 4, /*robust_passes=*/2,
                                          3.0f, 3.0f, 0.0f, 0.0f);
  REQUIRE(two_passes.accepted ==
          std::vector<bool>{true, true, true, true, false, false, false});

  // A third pass changes nothing further (already converged).
  auto three_passes = apply_robust_clipping(candidates, 4, /*robust_passes=*/3,
                                            3.0f, 3.0f, 0.0f, 0.0f);
  REQUIRE(three_passes.accepted == two_passes.accepted);
}

TEST_CASE("robust clipping: min_fraction/min_n_eff veto rejects the whole "
          "pixel when too much was clipped (plan 11.8 step 8)") {
  auto candidates = make_candidates({10, 10, 10, 10, 100});
  // Same geometry as the degenerate-MAD test (1 of 5 clipped -> fraction=0.8),
  // but now require a fraction the surviving 4/5 cannot reach.
  auto result = apply_robust_clipping(candidates, 5, 3, 3.0f, 3.0f,
                                      /*min_fraction=*/0.9f, 0.0f);
  REQUIRE(result.pixel_rejected);

  auto ok = apply_robust_clipping(candidates, 5, 3, 3.0f, 3.0f,
                                  /*min_fraction=*/0.7f, 0.0f);
  REQUIRE_FALSE(ok.pixel_rejected);
}

TEST_CASE("robust clipping: result is independent of input order (plan 11.8 "
          "step 3 determinism)") {
  std::vector<ClipCandidate> forward = {
      {0, 10.0, 1.0}, {1, 11.0, 1.0}, {2, 12.0, 1.0},
      {3, 13.0, 1.0}, {4, 18.0, 1.0}, {5, 60.0, 1.0}, {6, 62.0, 1.0}};
  std::vector<ClipCandidate> shuffled = {
      forward[6], forward[2], forward[4], forward[0],
      forward[5], forward[1], forward[3]};

  auto a = apply_robust_clipping(forward, 4, 2, 3.0f, 3.0f, 0.0f, 0.0f);
  auto b = apply_robust_clipping(shuffled, 4, 2, 3.0f, 3.0f, 0.0f, 0.0f);

  std::array<bool, 7> accepted_by_frame_a{}, accepted_by_frame_b{};
  for (size_t i = 0; i < forward.size(); ++i)
    accepted_by_frame_a[forward[i].frame_index] = a.accepted[i];
  for (size_t i = 0; i < shuffled.size(); ++i)
    accepted_by_frame_b[shuffled[i].frame_index] = b.accepted[i];
  REQUIRE(accepted_by_frame_a == accepted_by_frame_b);
}

TEST_CASE("robust clipping: empty input is rejected, not a crash (plan 11.8)") {
  auto result = apply_robust_clipping({}, 5, 3, 3.0f, 3.0f, 0.0f, 0.0f);
  REQUIRE(result.accepted.empty());
  REQUIRE(result.pixel_rejected);
}

// --- M3 (plan 11.8/11.9): compute_forward_drizzle_uniform_and_raw ----------

namespace {
// 5 identically-registered MONO frames sharing one internal pixel, so every
// candidate's geometric weight B is equal and hand-computable n_eff/fraction
// checks are exact.
RegistrationSamplingPlan five_identical_frames_plan(const std::vector<float> &values) {
  RegistrationSamplingPlan plan = single_sample_plan(ColorMode::MONO);
  for (size_t i = 0; i < values.size(); ++i)
    plan.frames.push_back(make_affine_frame("f" + std::to_string(i), i,
                                            make_source_to_canvas(1, 0, 20, 0, 1, 20)));
  return plan;
}
}  // namespace

TEST_CASE("uniform+raw: an outlier excluded by clipping is absent from both "
          "profiles identically (plan 11.8/11.9)") {
  auto plan = five_identical_frames_plan({10, 10, 10, 10, 100});
  std::vector<Matrix2Df> images;
  for (float v : {10.0f, 10.0f, 10.0f, 10.0f, 100.0f}) {
    Matrix2Df img(1, 1);
    img(0, 0) = v;
    images.push_back(img);
  }
  SourceImageProvider source_of = [&](std::size_t idx) -> const Matrix2Df & {
    return images[idx];
  };

  config::ReconstructionDrizzleConfig drizzle_cfg;
  drizzle_cfg.internal_scale = 1;
  drizzle_cfg.pixfrac = 0.8f;
  drizzle_cfg.min_clip_contributors = 5;  // exactly the frame count: clipping engages
  drizzle_cfg.robust_passes = 2;
  config::ReconstructionClippingConfig clip_cfg;  // defaults: sigma 3/3, min_fraction 0.4, min_n_eff 3.0

  auto result =
      compute_forward_drizzle_uniform_and_raw(plan, source_of, drizzle_cfg, clip_cfg);

  double peak_u = 0.0, peak_r = 0.0;
  float peak_weight = 0.0f;
  for (size_t i = 0; i < result.uniform.L.value.size(); ++i) {
    if (result.uniform.L.support[i] && result.uniform.L.weight_sum[i] > peak_weight) {
      peak_weight = result.uniform.L.weight_sum[i];
      peak_u = result.uniform.L.value[i];
      peak_r = result.raw.L.value[i];
    }
  }
  // The 100-outlier is excluded (degenerate-MAD case, same as the pure
  // apply_robust_clipping test) -> mean of the remaining four 10s.
  REQUIRE(peak_u == Approx(10.0).epsilon(1e-3));
  // Raw is numerically identical to Uniform (G_eff/Q_composite stubbed at 1).
  REQUIRE(peak_r == Approx(peak_u).epsilon(1e-9));
  REQUIRE(result.clipping.pixel_channel_rejected == 0);
  REQUIRE(result.clipping.candidate_contributions_clipped == 1);
}

TEST_CASE("uniform+raw: Q_composite reweights Raw per plan 11.7/11.9 without "
          "vetoing the output pixel") {
  // Two identically-registered MONO frames on a 1x1 source: equal geometric
  // weight B, no clipping. Uniform is the plain mean; Raw is the
  // Q_composite-weighted mean (G_eff left at 1).
  RegistrationSamplingPlan plan = single_sample_plan(ColorMode::MONO);
  plan.frames.push_back(
      make_affine_frame("f0", 0, make_source_to_canvas(1, 0, 20, 0, 1, 20)));
  plan.frames.push_back(
      make_affine_frame("f1", 1, make_source_to_canvas(1, 0, 20, 0, 1, 20)));

  std::vector<Matrix2Df> images(2, Matrix2Df(1, 1));
  images[0](0, 0) = 10.0f;
  images[1](0, 0) = 20.0f;
  SourceImageProvider source_of = [&](std::size_t idx) -> const Matrix2Df & {
    return images[idx];
  };

  config::ReconstructionDrizzleConfig drizzle_cfg;
  drizzle_cfg.internal_scale = 1;
  drizzle_cfg.pixfrac = 0.8f;
  drizzle_cfg.min_clip_contributors = 3;  // > 2 frames: clipping never engages
  config::ReconstructionClippingConfig clip_cfg;
  clip_cfg.min_n_eff = 1.0f;  // 2 equal-weight frames -> n_eff 2

  auto peak = [](const ForwardDrizzleUniformResult &r) {
    double v = 0.0;
    float w = 0.0f;
    for (size_t i = 0; i < r.L.value.size(); ++i)
      if (r.L.support[i] && r.L.weight_sum[i] > w) {
        w = r.L.weight_sum[i];
        v = r.L.value[i];
      }
    return v;
  };

  SECTION("no quality provider -> Raw == Uniform == plain mean") {
    auto res = compute_forward_drizzle_uniform_and_raw(plan, source_of,
                                                       drizzle_cfg, clip_cfg);
    REQUIRE(peak(res.uniform) == Approx(15.0).epsilon(1e-6));
    REQUIRE(peak(res.raw) == Approx(15.0).epsilon(1e-9));
  }

  SECTION("Q_composite 1.0 vs 0.25 pulls Raw toward the high-Q frame") {
    std::vector<Matrix2Df> qmaps(2, Matrix2Df(1, 1));
    qmaps[0](0, 0) = 1.0f;
    qmaps[1](0, 0) = 0.25f;
    FrameQualityProvider quality_of = [&](std::size_t idx) -> FrameQualityMaps {
      return {&qmaps[idx], nullptr, nullptr};
    };
    auto res = compute_forward_drizzle_uniform_and_raw(
        plan, source_of, drizzle_cfg, clip_cfg, {}, {}, quality_of);
    // Uniform unchanged; Raw = (10*1 + 20*0.25) / (1 + 0.25) = 12.
    REQUIRE(peak(res.uniform) == Approx(15.0).epsilon(1e-6));
    REQUIRE(peak(res.raw) == Approx(12.0).epsilon(1e-6));
  }

  SECTION("Q_composite == 0 drops that frame from Raw but never vetoes the "
          "pixel (plan 11.7: no pixel veto)") {
    std::vector<Matrix2Df> qmaps(2, Matrix2Df(1, 1));
    qmaps[0](0, 0) = 1.0f;
    qmaps[1](0, 0) = 0.0f;  // explicit per-sample veto for frame 1
    FrameQualityProvider quality_of = [&](std::size_t idx) -> FrameQualityMaps {
      return {&qmaps[idx], nullptr, nullptr};
    };
    auto res = compute_forward_drizzle_uniform_and_raw(
        plan, source_of, drizzle_cfg, clip_cfg, {}, {}, quality_of);
    REQUIRE(peak(res.uniform) == Approx(15.0).epsilon(1e-6));  // both frames
    REQUIRE(peak(res.raw) == Approx(10.0).epsilon(1e-6));      // frame 0 only
    // The pixel is still supported in Raw (not rejected).
    bool raw_supported = false;
    for (size_t i = 0; i < res.raw.L.support.size(); ++i)
      if (res.raw.L.support[i]) raw_supported = true;
    REQUIRE(raw_supported);
    REQUIRE(res.clipping.pixel_channel_rejected == 0);
  }

  SECTION("a NaN Q_composite behaves as 0 for that sample (missing Q is not "
          "an unweighted fallback, plan 11.9)") {
    std::vector<Matrix2Df> qmaps(2, Matrix2Df(1, 1));
    qmaps[0](0, 0) = 1.0f;
    qmaps[1](0, 0) = std::numeric_limits<float>::quiet_NaN();
    FrameQualityProvider quality_of = [&](std::size_t idx) -> FrameQualityMaps {
      return {&qmaps[idx], nullptr, nullptr};
    };
    auto res = compute_forward_drizzle_uniform_and_raw(
        plan, source_of, drizzle_cfg, clip_cfg, {}, {}, quality_of);
    REQUIRE(peak(res.raw) == Approx(10.0).epsilon(1e-6));
  }

  SECTION("Fine profile uses pow(Q_scale0, fine_quality_exponent); Medium uses "
          "pow(Q_scale1, medium_quality_exponent); Uniform/Raw unchanged") {
    std::vector<Matrix2Df> s0(2, Matrix2Df(1, 1)), s1(2, Matrix2Df(1, 1));
    s0[0](0, 0) = 1.0f; s0[1](0, 0) = 0.5f;
    s1[0](0, 0) = 1.0f; s1[1](0, 0) = 0.25f;
    FrameQualityProvider quality_of = [&](std::size_t idx) -> FrameQualityMaps {
      return {nullptr, &s0[idx], &s1[idx]};
    };
    MultibandProfileParams mb;
    mb.emit_fine = true;
    mb.emit_medium = true;
    mb.fine_quality_exponent = 2.0f;    // frame1 fine weight (0.5)^2 = 0.25
    mb.medium_quality_exponent = 1.0f;  // frame1 medium weight 0.25

    auto res = compute_forward_drizzle_uniform_and_raw(
        plan, source_of, drizzle_cfg, clip_cfg, {}, {}, quality_of, mb);

    REQUIRE(peak(res.uniform) == Approx(15.0).epsilon(1e-6));
    REQUIRE(peak(res.raw) == Approx(15.0).epsilon(1e-6));  // no composite map
    // Fine  = (10*1 + 20*0.25) / (1 + 0.25) = 12.
    REQUIRE(peak(res.fine) == Approx(12.0).epsilon(1e-6));
    // Medium = (10*1 + 20*0.25) / (1 + 0.25) = 12.
    REQUIRE(peak(res.medium) == Approx(12.0).epsilon(1e-6));
  }

  SECTION("the default fine_quality_exponent (4) is more selective than 2") {
    std::vector<Matrix2Df> s0(2, Matrix2Df(1, 1));
    s0[0](0, 0) = 1.0f; s0[1](0, 0) = 0.5f;
    FrameQualityProvider quality_of = [&](std::size_t idx) -> FrameQualityMaps {
      return {nullptr, &s0[idx], nullptr};
    };
    MultibandProfileParams mb;
    mb.emit_fine = true;  // fine_quality_exponent defaults to 4
    auto res = compute_forward_drizzle_uniform_and_raw(
        plan, source_of, drizzle_cfg, clip_cfg, {}, {}, quality_of, mb);
    // frame1 weight (0.5)^4 = 0.0625 => Fine = (10 + 20*0.0625)/1.0625.
    REQUIRE(peak(res.fine) == Approx(11.25 / 1.0625).epsilon(1e-6));
    REQUIRE(res.medium.L.value.empty());  // medium not requested
  }

  SECTION("requesting Fine/Medium without a quality provider is rejected") {
    MultibandProfileParams mb;
    mb.emit_fine = true;
    REQUIRE_THROWS(compute_forward_drizzle_uniform_and_raw(
        plan, source_of, drizzle_cfg, clip_cfg, {}, {}, {}, mb));
  }

  SECTION("emit_alpha_confidence produces the three per-pixel factor maps; "
          "< 8 accepted contributions => A_artifact not applicable => 0") {
    std::vector<Matrix2Df> comp(2, Matrix2Df(1, 1)), art(2, Matrix2Df(1, 1));
    comp[0](0, 0) = 0.9f; comp[1](0, 0) = 0.3f;
    art[0](0, 0) = 0.95f; art[1](0, 0) = 0.95f;
    FrameQualityProvider quality_of = [&](std::size_t idx) -> FrameQualityMaps {
      return {&comp[idx], nullptr, nullptr, &art[idx]};
    };
    MultibandProfileParams mb;
    mb.emit_alpha_confidence = true;

    auto res = compute_forward_drizzle_uniform_and_raw(
        plan, source_of, drizzle_cfg, clip_cfg, {}, {}, quality_of, mb);

    REQUIRE(res.a_separation.size() == res.uniform.L.value.size());
    REQUIRE(res.a_artifact.size() == res.uniform.L.value.size());
    REQUIRE(res.a_registration.size() == res.uniform.L.value.size());
    bool any_supported = false;
    for (size_t i = 0; i < res.alpha_confidence_support.size(); ++i)
      if (res.alpha_confidence_support[i]) {
        any_supported = true;
        REQUIRE(std::isfinite(res.a_separation[i]));
        REQUIRE(res.a_separation[i] >= 0.0f);
        REQUIRE(res.a_separation[i] <= 1.0f);
        REQUIRE(res.a_artifact[i] == Approx(0.0f));  // 2 frames < 8 -> N/A
        REQUIRE(res.a_registration[i] >= 0.0f);
      }
    REQUIRE(any_supported);
  }

  SECTION("emit_alpha_confidence without a composite or artifact map is "
          "rejected") {
    std::vector<Matrix2Df> comp(2, Matrix2Df(1, 1));
    comp[0](0, 0) = 0.9f; comp[1](0, 0) = 0.3f;
    FrameQualityProvider composite_only =
        [&](std::size_t idx) -> FrameQualityMaps {
      return {&comp[idx], nullptr, nullptr, nullptr};
    };
    MultibandProfileParams mb;
    mb.emit_alpha_confidence = true;
    REQUIRE_THROWS(compute_forward_drizzle_uniform_and_raw(
        plan, source_of, drizzle_cfg, clip_cfg, {}, {}, composite_only, mb));
    REQUIRE_THROWS(compute_forward_drizzle_uniform_and_raw(
        plan, source_of, drizzle_cfg, clip_cfg, {}, {}, {}, mb));
  }
}

TEST_CASE("emit_alpha_confidence: the < 8 threshold counts contributions with "
          "REAL artifact data, not mere frame presence (plan 14.4)") {
  // 10 identically-registered frames. Every frame carries a composite map;
  // 9 carry an artifact map (0.95), one carries none. Under the correct rule
  // A_artifact is applicable (9 real artifact contributions >= 8) and, with a
  // clean 0.95 p10, saturates to 1. A rule that counted frame presence would
  // treat the map-less frame as artifact_conf == 0, drag the weighted p10 far
  // below 0.75 and report A_artifact well under 1.
  auto plan = five_identical_frames_plan(std::vector<float>(10, 12.0f));
  std::vector<Matrix2Df> images(10, Matrix2Df(1, 1));
  for (auto &im : images) im(0, 0) = 12.0f;
  SourceImageProvider source_of = [&](std::size_t idx) -> const Matrix2Df & {
    return images[idx];
  };

  Matrix2Df comp(1, 1); comp(0, 0) = 0.7f;
  Matrix2Df art(1, 1);  art(0, 0) = 0.95f;
  FrameQualityProvider quality_of = [&](std::size_t idx) -> FrameQualityMaps {
    return {&comp, nullptr, nullptr, idx == 3 ? nullptr : &art};
  };

  config::ReconstructionDrizzleConfig drizzle_cfg;
  drizzle_cfg.internal_scale = 1;
  drizzle_cfg.pixfrac = 0.8f;
  drizzle_cfg.min_clip_contributors = 11;  // > 10 frames: no clipping
  config::ReconstructionClippingConfig clip_cfg;
  clip_cfg.min_n_eff = 1.0f;
  clip_cfg.min_fraction = 0.1f;

  MultibandProfileParams mb;
  mb.emit_alpha_confidence = true;

  auto res = compute_forward_drizzle_uniform_and_raw(
      plan, source_of, drizzle_cfg, clip_cfg, {}, {}, quality_of, mb);

  bool any = false;
  for (size_t i = 0; i < res.alpha_confidence_support.size(); ++i)
    if (res.alpha_confidence_support[i]) {
      any = true;
      REQUIRE(res.a_artifact[i] == Approx(1.0f).margin(1e-4));
    }
  REQUIRE(any);
}

TEST_CASE("emit_alpha_confidence: CHARACTERISATION -- at production OSC "
          "geometry (internal_scale 2, pixfrac 0.8, RGGB) the per-channel "
          "8-contribution A_artifact bar suppresses Fine/Medium alpha on a "
          "material fraction of the interior at typical frame counts "
          "(section-15 gate design input; see plan 30.40)") {
  // Measured: nf=30 -> ~60% of interior supported pixels keep A_artifact>0;
  // nf=60 -> ~84%. The per-channel "< 8 valid contributions => A_artifact=0"
  // rule with min_c (plan 14.4) genuinely bites on sparse OSC R/B. This test
  // pins the behaviour so a further collapse is caught; the design response
  // is a plan decision, not a threshold tweak here.
  const int nf = 30, sw = 28, sh = 24;
  RegistrationSamplingPlan plan;
  plan.source_width = sw;
  plan.source_height = sh;
  plan.canvas_width_native = sw + 12;
  plan.canvas_height_native = sh + 12;
  plan.color_mode = ColorMode::OSC;
  plan.bayer_pattern = BayerPattern::RGGB;
  for (int i = 0; i < nf; ++i) {
    // Deterministic sub-pixel dither on a 6x6 lattice + tiny rotation.
    const double dx = 6.0 + ((i * 7) % 6) / 6.0;
    const double dy = 6.0 + ((i * 5) % 6) / 6.0;
    const double a = 0.02 * std::sin(0.3 * i);
    plan.frames.push_back(make_affine_frame(
        "f" + std::to_string(i), i,
        make_source_to_canvas(std::cos(a), -std::sin(a), dx, std::sin(a),
                              std::cos(a), dy)));
  }

  std::vector<Matrix2Df> imgs(nf, Matrix2Df(sh, sw));
  for (int f = 0; f < nf; ++f)
    for (int y = 0; y < sh; ++y)
      for (int x = 0; x < sw; ++x)
        imgs[f](y, x) = 200.0f + 20.0f * std::sin(0.25f * x) +
                        12.0f * std::cos(0.2f * y);
  SourceImageProvider source_of = [&](std::size_t i) -> const Matrix2Df & {
    return imgs[i];
  };

  // Per-frame constant composite Q spanning [0.35, 0.9] so A_separation is
  // non-trivial; artifact map a clean 0.9 for every frame.
  std::vector<Matrix2Df> comp(nf, Matrix2Df(sh, sw)), art(nf, Matrix2Df(sh, sw));
  for (int f = 0; f < nf; ++f) {
    comp[f].setConstant(0.35f + 0.55f * static_cast<float>(f) / (nf - 1));
    art[f].setConstant(0.9f);
  }
  FrameQualityProvider quality_of = [&](std::size_t i) -> FrameQualityMaps {
    return {&comp[i], nullptr, nullptr, &art[i]};
  };

  config::ReconstructionDrizzleConfig drizzle_cfg;
  drizzle_cfg.internal_scale = 2;
  drizzle_cfg.pixfrac = 0.8f;
  drizzle_cfg.min_clip_contributors = nf + 1;  // isolate the A_artifact bar
  config::ReconstructionClippingConfig clip_cfg;
  clip_cfg.min_n_eff = 1.0f;
  clip_cfg.min_fraction = 0.1f;

  MultibandProfileParams mb;
  mb.emit_alpha_confidence = true;
  auto res = compute_forward_drizzle_uniform_and_raw(
      plan, source_of, drizzle_cfg, clip_cfg, {}, {}, quality_of, mb);

  // Interior mask: drop a 3-internal-px frame so we measure where coverage is
  // full, not the dither-thinned rim.
  const int iw = res.uniform.R.width, ih = res.uniform.R.height;
  long supported = 0, live = 0;
  for (int y = 3; y < ih - 3; ++y)
    for (int x = 3; x < iw - 3; ++x) {
      const std::size_t i = static_cast<std::size_t>(y) * iw + x;
      if (!res.alpha_confidence_support[i]) continue;
      ++supported;
      if (res.a_artifact[i] > 0.0f) ++live;
    }
  const double live_frac =
      supported > 0 ? static_cast<double>(live) / supported : 0.0;
  INFO("interior supported=" << supported << " A_artifact>0=" << live
                             << " frac=" << live_frac);
  REQUIRE(supported > 0);
  // Majority of the interior stays live at nf=30 (regression floor); the
  // remainder is the documented per-channel-<8 collapse. If live_frac falls
  // through 0.45 something regressed in the confidence wiring.
  REQUIRE(live_frac > 0.45);
  REQUIRE(live_frac < 0.80);  // and it is NOT ~full -- the collapse is real
}

TEST_CASE("uniform+raw: below min_clip_contributors, no clipping happens "
          "(matches M2's unclipped behaviour, plan 11.8 step 2)") {
  auto plan = five_identical_frames_plan({10, 10, 10, 10, 100});
  std::vector<Matrix2Df> images;
  for (float v : {10.0f, 10.0f, 10.0f, 10.0f, 100.0f}) {
    Matrix2Df img(1, 1);
    img(0, 0) = v;
    images.push_back(img);
  }
  SourceImageProvider source_of = [&](std::size_t idx) -> const Matrix2Df & {
    return images[idx];
  };

  config::ReconstructionDrizzleConfig drizzle_cfg;
  drizzle_cfg.internal_scale = 1;
  drizzle_cfg.pixfrac = 0.8f;
  drizzle_cfg.min_clip_contributors = 6;  // > 5 frames present: clipping never engages
  config::ReconstructionClippingConfig clip_cfg;

  auto result =
      compute_forward_drizzle_uniform_and_raw(plan, source_of, drizzle_cfg, clip_cfg);
  double peak_u = 0.0;
  float peak_weight = 0.0f;
  for (size_t i = 0; i < result.uniform.L.value.size(); ++i)
    if (result.uniform.L.support[i] && result.uniform.L.weight_sum[i] > peak_weight) {
      peak_weight = result.uniform.L.weight_sum[i];
      peak_u = result.uniform.L.value[i];
    }
  // Mean of all five values including the outlier: (10*4+100)/5 = 28.
  REQUIRE(peak_u == Approx(28.0).epsilon(1e-3));
  REQUIRE(result.clipping.candidate_contributions_clipped == 0);
}

TEST_CASE("uniform+raw: min_fraction veto rejects the pixel in both "
          "profiles identically, no partial fill (plan 11.8 step 8)") {
  auto plan = five_identical_frames_plan({10, 10, 10, 10, 100});
  std::vector<Matrix2Df> images;
  for (float v : {10.0f, 10.0f, 10.0f, 10.0f, 100.0f}) {
    Matrix2Df img(1, 1);
    img(0, 0) = v;
    images.push_back(img);
  }
  SourceImageProvider source_of = [&](std::size_t idx) -> const Matrix2Df & {
    return images[idx];
  };

  config::ReconstructionDrizzleConfig drizzle_cfg;
  drizzle_cfg.internal_scale = 1;
  drizzle_cfg.pixfrac = 0.8f;
  drizzle_cfg.min_clip_contributors = 5;
  config::ReconstructionClippingConfig clip_cfg;
  clip_cfg.min_fraction = 0.9f;  // 4/5 = 0.8 accepted after clipping < 0.9

  auto result =
      compute_forward_drizzle_uniform_and_raw(plan, source_of, drizzle_cfg, clip_cfg);
  REQUIRE(result.clipping.pixel_channel_rejected == 1);
  for (size_t i = 0; i < result.uniform.L.support.size(); ++i) {
    REQUIRE(result.uniform.L.support[i] == 0);
    REQUIRE(result.raw.L.support[i] == 0);
  }
}

TEST_CASE("uniform+raw: MONO fills only L in both profiles (plan 11.4)") {
  auto plan = five_identical_frames_plan({1, 2, 3});
  std::vector<Matrix2Df> images;
  for (float v : {1.0f, 2.0f, 3.0f}) {
    Matrix2Df img(1, 1);
    img(0, 0) = v;
    images.push_back(img);
  }
  SourceImageProvider source_of = [&](std::size_t idx) -> const Matrix2Df & {
    return images[idx];
  };
  config::ReconstructionDrizzleConfig drizzle_cfg;
  drizzle_cfg.internal_scale = 1;
  drizzle_cfg.pixfrac = 0.8f;
  config::ReconstructionClippingConfig clip_cfg;
  auto result =
      compute_forward_drizzle_uniform_and_raw(plan, source_of, drizzle_cfg, clip_cfg);
  REQUIRE_FALSE(result.uniform.L.empty());
  REQUIRE(result.uniform.R.empty());
  REQUIRE_FALSE(result.raw.L.empty());
  REQUIRE(result.raw.R.empty());
}

TEST_CASE("uniform+raw: with no g_eff supplied, raw is bit-identical to the "
          "clipped uniform (backward compatible, plan 11.9)") {
  RegistrationSamplingPlan plan = single_sample_plan(ColorMode::MONO);
  plan.frames.push_back(make_affine_frame("f0", 0, make_source_to_canvas(1, 0, 20, 0, 1, 20)));
  plan.frames.push_back(make_affine_frame("f1", 1, make_source_to_canvas(1, 0, 20, 0, 1, 20)));
  Matrix2Df i0(1, 1), i1(1, 1);
  i0(0, 0) = 10.0f;
  i1(0, 0) = 30.0f;
  SourceImageProvider source_of = [&](std::size_t idx) -> const Matrix2Df & {
    return idx == 0 ? i0 : i1;
  };
  config::ReconstructionDrizzleConfig drizzle_cfg;
  drizzle_cfg.internal_scale = 1;
  drizzle_cfg.pixfrac = 0.8f;
  config::ReconstructionClippingConfig clip_cfg;
  clip_cfg.min_n_eff = 1.0f;  // isolate the raw/uniform comparison from the 2-frame n_eff veto

  auto r = compute_forward_drizzle_uniform_and_raw(plan, source_of, drizzle_cfg, clip_cfg);
  REQUIRE(r.uniform.L.support == r.raw.L.support);
  REQUIRE(r.uniform.L.weight_sum == r.raw.L.weight_sum);
  size_t supported = 0;
  for (size_t i = 0; i < r.uniform.L.support.size(); ++i)
    if (r.uniform.L.support[i]) {  // value is NaN at unsupported pixels -> compare only here
      REQUIRE(r.uniform.L.value[i] == r.raw.L.value[i]);
      ++supported;
    }
  REQUIRE(supported > 0);
}

TEST_CASE("uniform+raw: a per-frame g_eff pulls the raw value toward the "
          "higher-weighted frame while uniform stays the plain mean "
          "(plan 11.9)") {
  RegistrationSamplingPlan plan = single_sample_plan(ColorMode::MONO);
  plan.frames.push_back(make_affine_frame("f0", 0, make_source_to_canvas(1, 0, 20, 0, 1, 20)));
  plan.frames.push_back(make_affine_frame("f1", 1, make_source_to_canvas(1, 0, 20, 0, 1, 20)));
  Matrix2Df i0(1, 1), i1(1, 1);
  i0(0, 0) = 10.0f;
  i1(0, 0) = 30.0f;
  SourceImageProvider source_of = [&](std::size_t idx) -> const Matrix2Df & {
    return idx == 0 ? i0 : i1;
  };
  config::ReconstructionDrizzleConfig drizzle_cfg;
  drizzle_cfg.internal_scale = 1;
  drizzle_cfg.pixfrac = 0.8f;
  config::ReconstructionClippingConfig clip_cfg;
  clip_cfg.min_n_eff = 1.0f;  // isolate the g_eff behaviour from the 2-frame n_eff veto

  // Frame 1 down-weighted to 0.25: raw = (1*10 + 0.25*30) / 1.25 = 14.
  const std::vector<float> g_eff = {1.0f, 0.25f};
  auto r = compute_forward_drizzle_uniform_and_raw(plan, source_of, drizzle_cfg, clip_cfg, {},
                                                   g_eff);

  float u_peak = 0.0f, r_peak = 0.0f, w = 0.0f;
  for (size_t i = 0; i < r.uniform.L.value.size(); ++i)
    if (r.uniform.L.support[i] && r.uniform.L.weight_sum[i] > w) {
      w = r.uniform.L.weight_sum[i];
      u_peak = r.uniform.L.value[i];
      r_peak = r.raw.L.value[i];
    }
  REQUIRE(u_peak == Approx(20.0).epsilon(1e-3));  // plain mean, unaffected by g_eff
  REQUIRE(r_peak == Approx(14.0).epsilon(1e-3));  // pulled toward frame 0
  // Same support: g_eff > 0 zeroes nothing.
  REQUIRE(r.uniform.L.support == r.raw.L.support);
}

TEST_CASE("uniform+raw: a g_eff vector of the wrong length is rejected") {
  RegistrationSamplingPlan plan = single_sample_plan(ColorMode::MONO);
  plan.frames.push_back(make_affine_frame("f0", 0, make_source_to_canvas(1, 0, 20, 0, 1, 20)));
  Matrix2Df img(1, 1);
  img(0, 0) = 5.0f;
  SourceImageProvider source_of = [&](std::size_t) -> const Matrix2Df & { return img; };
  config::ReconstructionDrizzleConfig drizzle_cfg;
  drizzle_cfg.internal_scale = 1;
  config::ReconstructionClippingConfig clip_cfg;
  const std::vector<float> g_eff_wrong = {1.0f, 0.5f};  // 2 entries, 1 frame
  REQUIRE_THROWS(compute_forward_drizzle_uniform_and_raw(plan, source_of, drizzle_cfg, clip_cfg,
                                                         {}, g_eff_wrong));
}

TEST_CASE("uniform+raw: per-pixel candidate lists exceeding the memory "
          "budget fail closed instead of risking an uncontrolled allocation "
          "(plan 11.11 memory contract)",
          "[drizzle-audit]") {
  // 50 identically-registered frames, every one of a 50x50 canvas's pixels
  // touched by all 50 --- ~50 * 2500 * sizeof(ClipCandidate) of candidate
  // storage in a single stripe. Auto sizing must reduce rows before IO.
  RegistrationSamplingPlan plan;
  plan.source_width = 50;
  plan.source_height = 50;
  plan.canvas_width_native = 50;
  plan.canvas_height_native = 50;
  plan.color_mode = ColorMode::MONO;
  for (size_t i = 0; i < 50; ++i)
    plan.frames.push_back(
        make_affine_frame("f" + std::to_string(i), i, make_source_to_canvas(1, 0, 0, 0, 1, 0)));

  Matrix2Df source = Matrix2Df::Constant(50, 50, 1.0f);
  SourceImageProvider source_of = [&](std::size_t) -> const Matrix2Df & { return source; };

  config::ReconstructionDrizzleConfig drizzle_cfg;
  drizzle_cfg.internal_scale = 1;
  drizzle_cfg.pixfrac = 1.0f;
  drizzle_cfg.memory_budget_mb = 2;  // too small for ~3 MB of candidates
  config::ReconstructionClippingConfig clip_cfg;

  const auto bounded = compute_forward_drizzle_uniform_and_raw(plan, source_of, drizzle_cfg, clip_cfg);
  REQUIRE(bounded.diagnostics.resolved_chunk_rows < 50);
  REQUIRE(bounded.diagnostics.estimated_peak_bytes <= 2 * 1024 * 1024);
  REQUIRE(bounded.uniform.L.value[100] == 1.0f);
  size_t calls = 0;
  SourceImageProvider counted = [&](size_t) -> const Matrix2Df & { ++calls; return source; };
  drizzle_cfg.chunk_rows = 50;
  REQUIRE_THROWS_WITH(
      compute_forward_drizzle_uniform_and_raw(plan, counted, drizzle_cfg, clip_cfg),
      Catch::Matchers::ContainsSubstring("DRIZZLE_MEMORY_BUDGET"));
  REQUIRE(calls == 0);
  drizzle_cfg.chunk_rows = 0;

  // A realistic budget for the same input succeeds.
  drizzle_cfg.memory_budget_mb = 64;
  REQUIRE_NOTHROW(
      compute_forward_drizzle_uniform_and_raw(plan, source_of, drizzle_cfg, clip_cfg));
}

TEST_CASE("uniform+raw: streaming avoids both full output canvases", "[drizzle-audit][drizzle-memory]") {
  RegistrationSamplingPlan plan;
  plan.source_width = plan.source_height = 512;
  plan.canvas_width_native = plan.canvas_height_native = 512;
  plan.color_mode = ColorMode::MONO;
  plan.frames.push_back(make_affine_frame("f", 0, make_source_to_canvas(1, 0, 0, 0, 1, 0)));
  Matrix2Df source = Matrix2Df::Constant(512, 512, -7.0f);
  size_t calls = 0;
  SourceImageProvider provider = [&](size_t) -> const Matrix2Df & { ++calls; return source; };
  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = 1;
  cfg.pixfrac = 1;
  cfg.memory_budget_mb = 8;
  config::ReconstructionClippingConfig clipping;
  clipping.min_n_eff = 1;
  REQUIRE_THROWS(compute_forward_drizzle_uniform_and_raw(plan, provider, cfg, clipping));
  REQUIRE(calls == 0);
  int next_y = 0;
  const auto diag = stream_forward_drizzle_uniform_and_raw(plan, provider, cfg, clipping,
      [&](int y, const ForwardDrizzleUniformAndRawResult &stripe) {
        REQUIRE(y == next_y);
        next_y += stripe.uniform.internal_height;
        for (size_t i = 0; i < stripe.uniform.L.value.size(); ++i) {
          REQUIRE(stripe.uniform.L.value[i] == -7.0f);
          REQUIRE(stripe.raw.L.value[i] == -7.0f);
        }
      });
  REQUIRE(next_y == 512);
  REQUIRE(diag.diagnostics.estimated_peak_bytes <= 8 * 1024 * 1024);
  REQUIRE(diag.clipping.pixel_channel_evaluations == 512 * 512);
}

TEST_CASE("uniform+raw: invalid weights fail before source loading", "[drizzle-audit]") {
  RegistrationSamplingPlan plan;
  plan.source_width = plan.source_height = 1;
  plan.canvas_width_native = plan.canvas_height_native = 1;
  plan.color_mode = ColorMode::MONO;
  plan.frames.push_back(make_affine_frame("f", 0, make_source_to_canvas(1, 0, 0, 0, 1, 0)));
  Matrix2Df source = Matrix2Df::Ones(1, 1);
  size_t calls = 0;
  SourceImageProvider provider = [&](size_t) -> const Matrix2Df & { ++calls; return source; };
  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = 1;
  config::ReconstructionClippingConfig clipping;
  for (float weight : {-1.0f, 2.0f, std::numeric_limits<float>::quiet_NaN()}) {
    REQUIRE_THROWS(compute_forward_drizzle_uniform_and_raw(plan, provider, cfg, clipping, {}, {weight}));
  }
  REQUIRE(calls == 0);
}

TEST_CASE("uniform+raw: clipped OSC results are invariant across stripe boundaries", "[drizzle-audit]") {
  RegistrationSamplingPlan plan;
  plan.source_width = plan.source_height = 16;
  plan.canvas_width_native = plan.canvas_height_native = 16;
  plan.color_mode = ColorMode::OSC;
  plan.bayer_pattern = BayerPattern::RGGB;
  std::vector<Matrix2Df> sources;
  for (size_t i = 0; i < 6; ++i) {
    plan.frames.push_back(make_affine_frame("f" + std::to_string(i), i,
        make_source_to_canvas(0.99, -0.07, 0.1*i, 0.07, 0.99, 0.13*i)));
    Matrix2Df source(16, 16);
    for (int y = 0; y < 16; ++y) for (int x = 0; x < 16; ++x)
      source(y,x) = x + y * 0.2f + i;
    source(8, 8) = i == 5 ? 10000.0f : 10.0f;
    sources.push_back(std::move(source));
  }
  SourceImageProvider provider = [&](size_t i) -> const Matrix2Df & { return sources.at(i); };
  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = 2;
  cfg.pixfrac = 0.8f;
  cfg.chunk_rows = 1;
  config::ReconstructionClippingConfig clipping;
  clipping.min_n_eff = 1;
  const std::vector<float> weights = {0.1f,0.2f,0.4f,0.8f,0.9f,1.0f};
  auto a = compute_forward_drizzle_uniform_and_raw(plan, provider, cfg, clipping, {}, weights);
  cfg.chunk_rows = 13;
  auto b = compute_forward_drizzle_uniform_and_raw(plan, provider, cfg, clipping, {}, weights);
  auto compare = [&](const ProfilePlane &x, const ProfilePlane &y) {
    REQUIRE(x.support == y.support);
    REQUIRE(x.weight_sum == y.weight_sum);
    REQUIRE(x.n_eff == y.n_eff);
    for (size_t i = 0; i < x.value.size(); ++i)
      REQUIRE((x.value[i] == y.value[i] || (std::isnan(x.value[i]) && std::isnan(y.value[i]))));
  };
  compare(a.uniform.R,b.uniform.R); compare(a.uniform.G,b.uniform.G); compare(a.uniform.B,b.uniform.B);
  compare(a.raw.R,b.raw.R); compare(a.raw.G,b.raw.G); compare(a.raw.B,b.raw.B);
  REQUIRE(a.clipping.pixel_channel_rejected == b.clipping.pixel_channel_rejected);
  REQUIRE(a.clipping.candidate_contributions_clipped == b.clipping.candidate_contributions_clipped);
}
