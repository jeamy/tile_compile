// M1 tests for geometric coverage without image PREWARP
// (docs/AQMH/aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md
//  sections 9.2, 9.3, 9.5, 26).

#include "tile_compile/registration/sampling_geometry.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "tile_compile/reconstruction/forward_drizzle.hpp"
#include <algorithm>
#include <array>
#include <cmath>

using namespace tile_compile;
using namespace tile_compile::registration;
using Catch::Approx;

namespace {

WarpMatrix make_affine(float a, float b, float tx, float c, float d, float ty) {
  WarpMatrix m;
  m(0, 0) = a;
  m(0, 1) = b;
  m(0, 2) = tx;
  m(1, 0) = c;
  m(1, 1) = d;
  m(1, 2) = ty;
  return m;
}

FrameSamplingTransform make_frame(const std::string &id, size_t idx,
                                  const WarpMatrix &canvas_to_source) {
  FrameSamplingTransform f;
  f.frame_id = id;
  f.source_index = idx;
  f.valid = true;
  f.canvas_to_source = canvas_to_source;
  REQUIRE(invert_affine_2x3(canvas_to_source, 0.5f, 2.0f, f.source_to_canvas));
  f.source_to_canvas_affine_valid = true;
  return f;
}

config::ReconstructionCoverageGateConfig lenient_gate(int min_frames = 1) {
  config::ReconstructionCoverageGateConfig g;
  g.min_frames = min_frames;
  g.min_supported_fraction = 0.5f;
  g.min_channel_n_eff_floor = 1.0f;
  g.min_channel_n_eff_fraction = 0.0f;
  g.min_analysis_pixels = 1;
  g.max_internal_hole_area_px = 1000000;
  return g;
}

} // namespace

TEST_CASE("MONO identity warp, single frame: full coverage (plan 9.2)") {
  RegistrationSamplingPlan plan;
  plan.source_width = 20;
  plan.source_height = 20;
  plan.canvas_width_native = 20;
  plan.canvas_height_native = 20;
  plan.color_mode = ColorMode::MONO;
  plan.bayer_pattern = BayerPattern::UNKNOWN;
  plan.frames.push_back(
      make_frame("f0", 0, make_affine(1, 0, 0, 0, 1, 0))); // identity

  auto cov =
      compute_geometric_coverage(plan, /*internal_scale=*/1,
                                 /*pixfrac=*/1.0f, lenient_gate(1), 1.0f);
  REQUIRE(cov.internal_width == 20);
  REQUIRE(cov.internal_height == 20);
  REQUIRE(cov.support_count_l.size() == 400);
  // every pixel should be touched exactly once
  for (uint16_t v : cov.support_count_l)
    REQUIRE(v == 1);
  for (uint8_t v : cov.reconstruction_support_mask)
    REQUIRE(v == 1);
  for (uint8_t v : cov.analysis_common_mask)
    REQUIRE(v == 1);
  REQUIRE(cov.gate.passed);
  REQUIRE(cov.gate.valid_frame_count == 1);
  REQUIRE(cov.gate.analysis_pixels == 400);
}

TEST_CASE("MONO two frames with partial overlap: masks distinguish common vs "
          "any-frame support (plan 9.3)") {
  RegistrationSamplingPlan plan;
  plan.source_width = 20;
  plan.source_height = 20;
  plan.canvas_width_native = 30; // wider than a single frame's footprint
  plan.canvas_height_native = 20;
  plan.color_mode = ColorMode::MONO;
  plan.bayer_pattern = BayerPattern::UNKNOWN;
  // frame 0 sits at canvas x in [0,20); frame 1 shifted +10 -> canvas x in
  // [10,30) canvas_to_source: s = q - shift  =>  for frame1, s = q - (10,0)
  plan.frames.push_back(make_frame("f0", 0, make_affine(1, 0, 0, 0, 1, 0)));
  plan.frames.push_back(make_frame("f1", 1, make_affine(1, 0, -10, 0, 1, 0)));

  auto cov = compute_geometric_coverage(plan, 1, 1.0f, lenient_gate(2), 1.0f);
  REQUIRE(cov.internal_width == 30);

  auto at = [&](int x, int y) { return cov.support_count_l[y * 30 + x]; };
  // x in [0,10): only frame 0
  REQUIRE(at(2, 5) == 1);
  // x in [10,20): both frames overlap
  REQUIRE(at(15, 5) == 2);
  // x in [20,30): only frame 1
  REQUIRE(at(25, 5) == 1);

  auto mask_at = [&](const std::vector<uint8_t> &m, int x, int y) {
    return m[y * 30 + x];
  };
  // analysis_common_mask (fraction=1.0 -> needs both frames) only in overlap
  REQUIRE(mask_at(cov.analysis_common_mask, 2, 5) == 0);
  REQUIRE(mask_at(cov.analysis_common_mask, 15, 5) == 1);
  REQUIRE(mask_at(cov.analysis_common_mask, 25, 5) == 0);
  // reconstruction_support_mask (>=1 frame) everywhere touched
  REQUIRE(mask_at(cov.reconstruction_support_mask, 2, 5) == 1);
  REQUIRE(mask_at(cov.reconstruction_support_mask, 25, 5) == 1);
}

TEST_CASE(
    "OSC RGGB single frame identity: R/B are sparser than G (plan 11.4/9.2)") {
  RegistrationSamplingPlan plan;
  plan.source_width = 16;
  plan.source_height = 16;
  plan.canvas_width_native = 16;
  plan.canvas_height_native = 16;
  plan.color_mode = ColorMode::OSC;
  plan.bayer_pattern = BayerPattern::RGGB;
  plan.frames.push_back(make_frame("f0", 0, make_affine(1, 0, 0, 0, 1, 0)));

  auto cov = compute_geometric_coverage(plan, 1, 1.0f, lenient_gate(1), 1.0f);
  const int touched_r =
      std::count_if(cov.support_count_r.begin(), cov.support_count_r.end(),
                    [](uint16_t v) { return v > 0; });
  const int touched_g =
      std::count_if(cov.support_count_g.begin(), cov.support_count_g.end(),
                    [](uint16_t v) { return v > 0; });
  const int touched_b =
      std::count_if(cov.support_count_b.begin(), cov.support_count_b.end(),
                    [](uint16_t v) { return v > 0; });
  // RGGB on a 16x16 grid: 64 R sites, 128 G sites, 64 B sites; with pixfrac=1
  // and identity warp each site's droplet stays within its own pixel cell, so
  // touched counts should equal site counts exactly.
  REQUIRE(touched_r == 64);
  REQUIRE(touched_g == 128);
  REQUIRE(touched_b == 64);
  // R and B never touch the same pixel as each other in this configuration.
  for (int i = 0; i < 256; ++i) {
    REQUIRE_FALSE((cov.support_count_r[i] > 0 && cov.support_count_b[i] > 0));
  }
}

TEST_CASE("coverage_gate fails fail-closed on too few frames (plan 26)") {
  RegistrationSamplingPlan plan;
  plan.source_width = 20;
  plan.source_height = 20;
  plan.canvas_width_native = 20;
  plan.canvas_height_native = 20;
  plan.color_mode = ColorMode::MONO;
  plan.frames.push_back(make_frame("f0", 0, make_affine(1, 0, 0, 0, 1, 0)));

  config::ReconstructionCoverageGateConfig g = lenient_gate(1);
  g.min_frames = 5; // require more frames than are present
  auto cov = compute_geometric_coverage(plan, 1, 1.0f, g, 1.0f);
  REQUIRE_FALSE(cov.gate.passed);
  REQUIRE_FALSE(cov.gate.violations.empty());
  bool found = false;
  for (const auto &v : cov.gate.violations)
    if (v.find("min_frames") != std::string::npos)
      found = true;
  REQUIRE(found);
}

TEST_CASE("coverage_gate reports the hole check as implemented and finds no "
          "hole on a fully-covered mask (plan 26)") {
  RegistrationSamplingPlan plan;
  plan.source_width = 10;
  plan.source_height = 10;
  plan.canvas_width_native = 10;
  plan.canvas_height_native = 10;
  plan.color_mode = ColorMode::MONO;
  plan.frames.push_back(make_frame("f0", 0, make_affine(1, 0, 0, 0, 1, 0)));
  auto cov = compute_geometric_coverage(plan, 1, 1.0f, lenient_gate(1), 1.0f);
  REQUIRE(cov.gate.hole_check_implemented);
  REQUIRE(cov.gate.largest_internal_hole_area_px == 0);
}

TEST_CASE("largest_interior_hole_area finds a true interior hole but ignores "
          "a border-touching unsupported region (plan 9.5/26)") {
  // 10x10 all-supported mask except:
  //   - a 2x2 hole fully inside (rows/cols 4..5)      -> interior hole, area 4
  //   - a 1x3 unsupported strip touching the top edge  -> NOT a hole (exterior)
  const int W = 10, H = 10;
  std::vector<uint8_t> mask(static_cast<size_t>(W * H), 1);
  auto clear = [&](int x, int y) { mask[static_cast<size_t>(y * W + x)] = 0; };
  clear(4, 4);
  clear(5, 4);
  clear(4, 5);
  clear(5, 5); // interior 2x2 hole
  clear(1, 0);
  clear(2, 0);
  clear(3, 0); // touches border (y=0)

  REQUIRE(largest_interior_hole_area(mask, W, H) == 4);
}

TEST_CASE("largest_interior_hole_area returns 0 for a fully supported or "
          "fully unsupported mask (plan 9.5/26)") {
  std::vector<uint8_t> full(100, 1);
  REQUIRE(largest_interior_hole_area(full, 10, 10) == 0);
  // fully unsupported: every 0-pixel is reachable from the border -> no
  // interior hole by definition (the "hole" IS the exterior).
  std::vector<uint8_t> empty(100, 0);
  REQUIRE(largest_interior_hole_area(empty, 10, 10) == 0);
}

TEST_CASE("largest_interior_hole_area picks the largest of several interior "
          "holes (plan 9.5/26)") {
  const int W = 20, H = 10;
  std::vector<uint8_t> mask(static_cast<size_t>(W * H), 1);
  auto clear = [&](int x, int y) { mask[static_cast<size_t>(y * W + x)] = 0; };
  // small hole: 1 pixel
  clear(2, 5);
  // bigger hole: 3x3 = 9 pixels
  for (int y = 4; y <= 6; ++y)
    for (int x = 10; x <= 12; ++x)
      clear(x, y);
  REQUIRE(largest_interior_hole_area(mask, W, H) == 9);
}

TEST_CASE("coverage_gate fails when the interior hole exceeds "
          "max_internal_hole_area_px (plan 26)") {
  // Two frames whose footprints together leave an interior gap: frame 0
  // covers the full 20x20 canvas, frame 1 is irrelevant to this test other
  // than being a second valid frame; instead we drive the hole via a plan
  // whose single frame simply doesn't cover a sub-rectangle at all, by
  // shrinking its source so a band of the canvas is never sampled while
  // canvas pixels on both sides of that band ARE sampled by a second,
  // wider frame that surrounds it.
  RegistrationSamplingPlan plan;
  plan.source_width = 20;
  plan.source_height = 20;
  plan.canvas_width_native = 20;
  plan.canvas_height_native = 20;
  plan.color_mode = ColorMode::MONO;
  plan.frames.push_back(make_frame("f0", 0, make_affine(1, 0, 0, 0, 1, 0)));

  config::ReconstructionCoverageGateConfig g = lenient_gate(1);
  g.max_internal_hole_area_px = -1; // any hole at all fails; canvas has none
  auto cov = compute_geometric_coverage(plan, 1, 1.0f, g, 1.0f);
  // Full identity coverage has no interior hole, so this specific config only
  // documents that the check participates in gate.passed when triggered;
  // the actual triggering behaviour is exercised via largest_interior_hole_area
  // directly above (unit-level) since constructing a coverage-shaped hole
  // through the public forward-mapping API requires multi-frame geometry that
  // is easier to state precisely on the mask directly.
  REQUIRE(cov.gate.largest_internal_hole_area_px == 0);
  REQUIRE_FALSE(cov.gate.passed); // 0 > -1 triggers the gate
  bool found = false;
  for (const auto &v : cov.gate.violations)
    if (v.find("max_internal_hole_area_px") != std::string::npos)
      found = true;
  REQUIRE(found);
}

TEST_CASE("compute_geometric_coverage is bit-exact independent of worker "
          "count (plan 9.2 parallelization determinism)") {
  RegistrationSamplingPlan plan;
  plan.source_width = 40;
  plan.source_height = 40;
  plan.canvas_width_native = 50;
  plan.canvas_height_native = 50;
  plan.color_mode = ColorMode::OSC;
  plan.bayer_pattern = BayerPattern::RGGB;
  // several overlapping frames with different affine warps (translation +
  // small rotation) so coverage counts vary spatially
  plan.frames.push_back(make_frame("f0", 0, make_affine(1, 0, 2, 0, 1, 3)));
  plan.frames.push_back(make_frame("f1", 1, make_affine(1, 0, 5, 0, 1, 1)));
  const float c = std::cos(0.05f), s = std::sin(0.05f);
  plan.frames.push_back(make_frame("f2", 2, make_affine(c, -s, 4, s, c, 2)));
  plan.frames.push_back(make_frame("f3", 3, make_affine(1, 0, -1, 0, 1, 4)));
  plan.frames.push_back(make_frame("f4", 4, make_affine(1, 0, 3, 0, 1, -2)));

  auto g = lenient_gate(1);
  const auto cov1 = compute_geometric_coverage(plan, 2, 0.8f, g, 1.0f, 1);
  const auto cov4 = compute_geometric_coverage(plan, 2, 0.8f, g, 1.0f, 4);
  const auto cov7 = compute_geometric_coverage(plan, 2, 0.8f, g, 1.0f, 7);

  REQUIRE(cov1.support_count_r == cov4.support_count_r);
  REQUIRE(cov1.support_count_g == cov4.support_count_g);
  REQUIRE(cov1.support_count_b == cov4.support_count_b);
  REQUIRE(cov1.support_count_r == cov7.support_count_r);
  REQUIRE(cov1.support_count_g == cov7.support_count_g);
  REQUIRE(cov1.support_count_b == cov7.support_count_b);
  REQUIRE(cov1.analysis_common_mask == cov4.analysis_common_mask);
  REQUIRE(cov1.reconstruction_support_mask == cov4.reconstruction_support_mask);
  REQUIRE(cov1.gate.analysis_pixels == cov4.gate.analysis_pixels);
  REQUIRE(cov1.gate.min_supported_fraction ==
          Approx(cov4.gate.min_supported_fraction));
  REQUIRE(cov1.gate.largest_internal_hole_area_px ==
          cov4.gate.largest_internal_hole_area_px);
}

TEST_CASE("2x internal_scale doubles the coverage canvas (plan 12.1)") {
  RegistrationSamplingPlan plan;
  plan.source_width = 10;
  plan.source_height = 10;
  plan.canvas_width_native = 10;
  plan.canvas_height_native = 10;
  plan.color_mode = ColorMode::MONO;
  plan.frames.push_back(make_frame("f0", 0, make_affine(1, 0, 0, 0, 1, 0)));
  auto cov = compute_geometric_coverage(plan, 2, 0.8f, lenient_gate(1), 1.0f);
  REQUIRE(cov.internal_width == 20);
  REQUIRE(cov.internal_height == 20);
}

TEST_CASE("serialize_sampling_geometry_json produces the documented fields "
          "(plan 9.4)") {
  RegistrationSamplingPlan plan;
  plan.source_width = 10;
  plan.source_height = 10;
  plan.canvas_width_native = 10;
  plan.canvas_height_native = 10;
  plan.color_mode = ColorMode::MONO;
  plan.frames.push_back(make_frame("f0", 0, make_affine(1, 0, 0, 0, 1, 0)));
  plan.plan_hash = compute_plan_hash(plan);
  auto cov = compute_geometric_coverage(plan, 1, 0.8f, lenient_gate(1), 1.0f);
  const std::string js =
      serialize_sampling_geometry_json(plan, "abc123", "square", 0.8f, 1, cov);
  REQUIRE(js.find("\"coverage_source\": \"forward_drizzle_geometry\"") !=
          std::string::npos);
  REQUIRE(js.find("\"sampling_plan_hash\"") != std::string::npos);
  REQUIRE(js.find("\"coverage_geometry_hash\": \"abc123\"") !=
          std::string::npos);
  REQUIRE(js.find("\"passed\": true") != std::string::npos);
  REQUIRE(js.find("\"dither_spread_circular_px_diagnostic\"") !=
          std::string::npos);
  REQUIRE(js.find("\"x_p10\"") != std::string::npos);
  REQUIRE(js.find("\"y_p10\"") != std::string::npos);
}

TEST_CASE("dither spread circular diagnostic: identical frames (zero dither) "
          "give near-zero sigma (plan 9.3/8.x)") {
  RegistrationSamplingPlan plan;
  plan.canvas_width_native = 100;
  plan.canvas_height_native = 100;
  // 5 frames, all with the *same* integer-pixel-aligned identity warp: every
  // frame lands on exactly the same phase mod 2 at every site, so the
  // circular resultant length R == 1 and sigma_circ_px == 0 exactly.
  for (int i = 0; i < 5; ++i) {
    plan.frames.push_back(
        make_frame("f" + std::to_string(i), i, make_affine(1, 0, 0, 0, 1, 0)));
  }
  auto diag = compute_dither_spread_circular_diagnostic(plan);
  // Exactly 0 mathematically (R == 1); the clamp to 1 - 1e-12 that guards
  // ln() against a literal R == 1 leaves a ~1e-6 floor, not true zero.
  REQUIRE(diag.x_p10 == Approx(0.0).margin(1e-5));
  REQUIRE(diag.y_p10 == Approx(0.0).margin(1e-5));
}

TEST_CASE("dither spread circular diagnostic: 4-way quadrature phase spread "
          "gives the maximal Rayleigh sigma (plan 9.3/8.x)") {
  RegistrationSamplingPlan plan;
  plan.canvas_width_native = 100;
  plan.canvas_height_native = 100;
  // 4 frames offset by 0, 0.5, 1.0, 1.5 px in x (and y): theta = pi * offset
  // lands exactly at 0, pi/2, pi, 3pi/2 --- perfectly uniform on the circle,
  // so the mean resultant vector is exactly (0,0), R == 0, sigma is the
  // (clamp-bounded) maximum the estimator can report.
  const float offsets[4] = {0.0f, 0.5f, 1.0f, 1.5f};
  for (int i = 0; i < 4; ++i) {
    plan.frames.push_back(
        make_frame("f" + std::to_string(i), i,
                   make_affine(1, 0, offsets[i], 0, 1, offsets[i])));
  }
  auto diag = compute_dither_spread_circular_diagnostic(plan);
  // sqrt(-2*ln(1e-12)) / pi, the clamp ceiling used to keep ln() finite.
  const double kMaxSigma = std::sqrt(-2.0 * std::log(1e-12)) / M_PI;
  REQUIRE(diag.x_p10 == Approx(kMaxSigma).epsilon(1e-6));
  REQUIRE(diag.y_p10 == Approx(kMaxSigma).epsilon(1e-6));
}

TEST_CASE("dither spread circular diagnostic: no valid frames returns zero, "
          "not NaN (plan 9.3/8.x)") {
  RegistrationSamplingPlan plan;
  plan.canvas_width_native = 100;
  plan.canvas_height_native = 100;
  auto diag = compute_dither_spread_circular_diagnostic(plan);
  REQUIRE(diag.x_p10 == 0.0);
  REQUIRE(diag.y_p10 == 0.0);
}

TEST_CASE("coverage audit: footprint mask is independent of sparse CFA support",
          "[drizzle-audit]") {
  RegistrationSamplingPlan plan;
  plan.source_width = 16;
  plan.source_height = 16;
  plan.canvas_width_native = 16;
  plan.canvas_height_native = 16;
  plan.color_mode = ColorMode::OSC;
  plan.bayer_pattern = BayerPattern::RGGB;
  for (size_t i = 0; i < 3; ++i)
    plan.frames.push_back(
        make_frame("f" + std::to_string(i), i, make_affine(1, 0, 0, 0, 1, 0)));
  auto cov = compute_geometric_coverage(plan, 2, 0.8f, lenient_gate(), 1.0f);
  REQUIRE(cov.gate.analysis_pixels == 1024);
  REQUIRE(std::count(cov.analysis_common_mask.begin(),
                     cov.analysis_common_mask.end(), 1) == 1024);
  REQUIRE(cov.gate.min_supported_fraction == Approx(0.25));
  REQUIRE_FALSE(cov.gate.passed);
}

TEST_CASE(
    "coverage audit: unequal areas do not masquerade as three effective frames",
    "[drizzle-audit]") {
  RegistrationSamplingPlan plan;
  plan.source_width = 1;
  plan.source_height = 1;
  plan.canvas_width_native = 1;
  plan.canvas_height_native = 1;
  plan.color_mode = ColorMode::MONO;
  plan.frames.push_back(make_frame("f0", 0, make_affine(1, 0, 0, 0, 1, 0)));
  plan.frames.push_back(
      make_frame("f1", 1, make_affine(1, 0, -0.99f, 0, 1, 0)));
  plan.frames.push_back(
      make_frame("f2", 2, make_affine(1, 0, -0.99f, 0, 1, 0)));
  auto gate = lenient_gate();
  gate.min_channel_n_eff_floor = 3;
  auto cov = compute_geometric_coverage(plan, 1, 1, gate, 1);
  REQUIRE(cov.support_count_l[0] == 3);
  REQUIRE(cov.gate.min_channel_n_eff_p10 ==
          Approx(1.02 * 1.02 / 1.0002).epsilon(1e-5));
  REQUIRE_FALSE(cov.gate.passed);
}

TEST_CASE(
    "coverage audit: stripes preserve exact geometry and weighted percentile",
    "[drizzle-audit]") {
  RegistrationSamplingPlan plan;
  plan.source_width = 16;
  plan.source_height = 12;
  plan.canvas_width_native = 30;
  plan.canvas_height_native = 28;
  plan.color_mode = ColorMode::MONO;
  plan.frames.push_back(make_frame("f0", 0, make_affine(1, 0, -3, 0, 1, -2)));
  plan.frames.push_back(
      make_frame("f1", 1, make_affine(0.8f, -0.3f, 2, 0.3f, 0.8f, -4)));
  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = 2;
  cfg.chunk_rows = 1;
  auto a =
      compute_geometric_coverage(plan, 2, 0.8f, lenient_gate(), 0.5f, 128, cfg);
  cfg.chunk_rows = 56;
  auto b =
      compute_geometric_coverage(plan, 2, 0.8f, lenient_gate(), 0.5f, 1, cfg);
  REQUIRE(a.support_count_l == b.support_count_l);
  REQUIRE(a.analysis_common_mask == b.analysis_common_mask);
  REQUIRE(a.reconstruction_support_mask == b.reconstruction_support_mask);
  REQUIRE(a.gate.min_channel_n_eff_p10 == b.gate.min_channel_n_eff_p10);
  REQUIRE(a.gate.largest_internal_hole_area_px ==
          b.gate.largest_internal_hole_area_px);
  REQUIRE(a.gate.workers_used == 1);
  cfg.memory_budget_mb = 1;
  REQUIRE_THROWS(compute_geometric_coverage(plan, 2, 0.8f, lenient_gate(), 0.5f,
                                            128, cfg));
}

TEST_CASE("coverage audit: semantic hashes invalidate geometry without rounded "
          "float collisions",
          "[drizzle-audit]") {
  RegistrationSamplingPlan plan;
  config::ReconstructionDrizzleConfig cfg;
  auto initial = compute_coverage_geometry_hash(plan, cfg, 1);
  REQUIRE(initial != compute_coverage_geometry_hash(plan, cfg, 0.5));
  cfg.pixfrac = std::nextafter(cfg.pixfrac, 1.0f);
  REQUIRE(initial != compute_coverage_geometry_hash(plan, cfg, 1));
  cfg.pixfrac = 0.8f;
  cfg.chunk_rows = 1;
  cfg.memory_budget_mb = 16;
  REQUIRE(initial == compute_coverage_geometry_hash(plan, cfg, 1));
}

TEST_CASE("coverage audit: exact support and neff match uniform under rotation "
          "and shear",
          "[drizzle-audit]") {
  RegistrationSamplingPlan plan;
  plan.source_width = 12;
  plan.source_height = 10;
  plan.canvas_width_native = 24;
  plan.canvas_height_native = 24;
  plan.color_mode = ColorMode::OSC;
  plan.bayer_pattern = BayerPattern::RGGB;
  for (size_t i = 0; i < 3; ++i)
    plan.frames.push_back(make_frame(
        "f" + std::to_string(i), i,
        make_affine(0.9f, -0.25f, -1 - i * 0.2f, 0.3f, 1.1f, -4 + i * 0.17f)));
  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = 2;
  cfg.chunk_rows = 3;
  auto cov =
      compute_geometric_coverage(plan, 2, 0.8f, lenient_gate(), 0.5f, 1, cfg);
  Matrix2Df source = Matrix2Df::Ones(10, 12);
  auto uniform = reconstruction::compute_forward_drizzle_uniform(
      plan, [&](size_t) -> const Matrix2Df & { return source; }, cfg);
  std::array<const reconstruction::ProfilePlane *, 3> planes = {
      &uniform.R, &uniform.G, &uniform.B};
  std::array<const std::vector<uint32_t> *, 3> counts = {
      &cov.support_count_r, &cov.support_count_g, &cov.support_count_b};
  double expected = 100;
  for (int c = 0; c < 3; ++c) {
    std::vector<float> neff;
    for (size_t i = 0; i < cov.analysis_common_mask.size(); ++i) {
      REQUIRE(planes[c]->support[i] == ((*counts[c])[i] > 0));
      if (cov.analysis_common_mask[i])
        neff.push_back(planes[c]->n_eff[i]);
    }
    REQUIRE_FALSE(neff.empty());
    std::sort(neff.begin(), neff.end());
    const double rank = (neff.size() - 1) * 0.1;
    const size_t lo = std::floor(rank), hi = std::ceil(rank);
    expected =
        std::min(expected, neff[lo] + (neff[hi] - neff[lo]) * (rank - lo));
  }
  REQUIRE(cov.gate.min_channel_n_eff_p10 == Approx(expected).margin(1e-7));
}
