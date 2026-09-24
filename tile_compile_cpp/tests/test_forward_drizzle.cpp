// M2 tests for the CFA-aware forward-drizzle CPU reference (Uniform-Control
// only; docs/AQMH/aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md
// section 11).

#include "tile_compile/reconstruction/forward_drizzle.hpp"
#include "tile_compile/reconstruction/forward_drizzle_cuda.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <tuple>
#include <utility>
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

// P6 CPU FORWARD_DRIZZLE acceleration probe: a candidate fast path skips
// polygon_rectangle_intersection_area for a unit cell that lies strictly
// inside the leaf quad and takes k = 1.0 directly. That is only a *bit-exact*
// win if the full clip also returns exactly 1.0 for such a cell. This probe
// asserts that across rotated / sheared leaves and realistic internal-canvas
// coordinate magnitudes (integers to ~7680, so shoelace products ~6e7, inside
// exact double). If any case is not exactly 1.0 the shortcut becomes a
// numeric-gated change instead of a free one.
TEST_CASE("polygon_rectangle_intersection_area returns EXACTLY 1.0 for a unit "
          "cell strictly inside the leaf quad",
          "[drizzle-audit]") {
  struct Leaf2 {
    double x[4], y[4];
  };
  auto mapped_leaf = [](double a, double b, double c, double d, double ox,
                        double oy, double s) {
    // image of a large source square [-s,s]^2 under [[a,b],[c,d]] + (ox,oy)
    const double sx[4] = {-s, s, s, -s};
    const double sy[4] = {-s, -s, s, s};
    Leaf2 L;
    for (int i = 0; i < 4; ++i) {
      L.x[i] = a * sx[i] + b * sy[i] + ox;
      L.y[i] = c * sx[i] + d * sy[i] + oy;
    }
    return L;
  };
  const std::array<std::array<double, 4>, 5> lin = {{
      {1.0, 0.0, 0.0, 1.0},        // identity
      {1.0, 0.18, 0.05, 1.0},      // shear
      {0.9063, -0.4226, 0.4226, 0.9063},  // 25 deg rotation
      {1.0, 0.70, 0.0, 1.0},       // strong shear
      {1.2, 0.15, -0.1, 0.85},     // rotation + anisotropic scale
  }};
  int checked = 0;
  for (const auto &m : lin)
    for (double centre : {40.0, 512.0, 4000.0, 7600.0}) {
      // A leaf that comfortably contains a 6x6 block of unit cells around
      // (centre, centre): source half-size 8 -> mapped extent >= 8 * min|.|.
      const Leaf2 L = mapped_leaf(m[0], m[1], m[2], m[3], centre + 0.37,
                                  centre - 0.21, 8.0);
      for (int gy = static_cast<int>(centre) - 1;
           gy <= static_cast<int>(centre) + 1; ++gy)
        for (int gx = static_cast<int>(centre) - 1;
             gx <= static_cast<int>(centre) + 1; ++gx) {
          const double k = polygon_rectangle_intersection_area(
              L.x, L.y, gx, gy, gx + 1.0, gy + 1.0);
          INFO("lin " << m[0] << "," << m[1] << "," << m[2] << "," << m[3]
                      << " cell " << gx << "," << gy);
          REQUIRE(k == 1.0);
          ++checked;
        }
    }
  REQUIRE(checked == 5 * 4 * 9);
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

TEST_CASE("forward drizzle: §30.81 target-column window partitions the cell "
          "set exactly, default window is byte-identical",
          "[drizzle-audit][fd-tile-window]") {
  RegistrationSamplingPlan plan;
  plan.source_width = 17;
  plan.source_height = 13;
  plan.canvas_width_native = 30;
  plan.canvas_height_native = 24;
  plan.color_mode = ColorMode::OSC;
  plan.bayer_pattern = BayerPattern::GBRG;
  plan.cfa_origin_x = 1;
  plan.cfa_origin_y = 0;
  const double angle = 0.6;  // rotation + shear + non-unit scale
  plan.frames.push_back(make_affine_frame(
      "f0", 0,
      make_source_to_canvas(1.3 * std::cos(angle), -1.1 * std::sin(angle) + 0.12,
                            5.4, 1.2 * std::sin(angle), 1.25 * std::cos(angle),
                            3.7)));
  const auto &f = plan.frames.front();

  const int scale = 2;
  const int W = plan.canvas_width_native * scale;  // 60
  const float pixfrac = 1.0f;
  const int y_begin = 3;
  const int rows = 11;
  const ForwardDrizzleSubdivisionParams sub{};

  using Cell = std::tuple<int, int, int, int, int, int>;  // sx,sy,c,leaf,cx,cy
  auto collect = [&](int x_begin, int cols) {
    std::vector<Cell> cells;
    enumerate_drizzle_stripe_leaf_cells(
        plan, f, scale, pixfrac, y_begin, rows,
        [&](int sx, int sy, int c, int leaf, int cx, int cy, const double *,
            const double *) { cells.emplace_back(sx, sy, c, leaf, cx, cy); },
        sub, x_begin, cols);
    std::sort(cells.begin(), cells.end());
    return cells;
  };

  const auto full = collect(0, -1);
  REQUIRE_FALSE(full.empty());
  for (const auto &cell : full) {
    const int cx = std::get<4>(cell);
    REQUIRE(cx >= 0);
    REQUIRE(cx < W);
  }

  // cols < 0 and an over-wide window both mean "full internal width".
  REQUIRE(collect(0, W) == full);
  REQUIRE(collect(0, W + 40) == full);
  REQUIRE(collect(-8, W + 40) == full);

  // Ragged partitions: a split that does not divide W, and one that leaves a
  // width-1 right remainder (W % w1 == 1) --- where a rebase off-by-one lives.
  for (int w1 : {7, 23, 31, 59}) {
    const auto left = collect(0, w1);
    const auto right = collect(w1, W - w1);
    for (const auto &cell : left) REQUIRE(std::get<4>(cell) < w1);
    for (const auto &cell : right) REQUIRE(std::get<4>(cell) >= w1);
    std::vector<Cell> merged;
    merged.reserve(left.size() + right.size());
    merged.insert(merged.end(), left.begin(), left.end());
    merged.insert(merged.end(), right.begin(), right.end());
    std::sort(merged.begin(), merged.end());
    INFO("split at w1=" << w1);
    REQUIRE(merged == full);  // disjoint by cx, so union == full, no duplicates
  }

  // Three-way ragged partition covering the same claim with more seams.
  {
    const auto a = collect(0, 13);
    const auto b = collect(13, 27);  // [13, 40)
    const auto c = collect(40, W - 40);
    std::vector<Cell> merged;
    merged.insert(merged.end(), a.begin(), a.end());
    merged.insert(merged.end(), b.begin(), b.end());
    merged.insert(merged.end(), c.begin(), c.end());
    std::sort(merged.begin(), merged.end());
    REQUIRE(merged == full);
  }

  // rasterize_drizzle_stripe: the windowed `index` is rebased to the window,
  // and reassembling the per-window (cx,cy,k) triples reproduces the
  // full-width rasterization exactly.
  using Area = std::tuple<int, int, double>;  // cx, cy, k
  auto raster = [&](int x_begin, int cols) {
    const int xb = std::clamp(x_begin, 0, W);
    const int win_w = cols < 0 ? W : std::clamp(x_begin + cols, xb, W) - xb;
    std::vector<Area> out;
    rasterize_drizzle_stripe(
        plan, f, scale, pixfrac, y_begin, rows,
        [&](int, int, int, int, size_t index, double k) {
          const int cy = static_cast<int>(index) / win_w + y_begin;
          const int cx = static_cast<int>(index) % win_w + xb;
          out.emplace_back(cx, cy, k);
        },
        sub, x_begin, cols);
    std::sort(out.begin(), out.end());
    return out;
  };
  const auto raster_full = raster(0, -1);
  REQUIRE_FALSE(raster_full.empty());
  {
    auto lo = raster(0, 23);
    auto hi = raster(23, W - 23);
    std::vector<Area> merged;
    merged.insert(merged.end(), lo.begin(), lo.end());
    merged.insert(merged.end(), hi.begin(), hi.end());
    std::sort(merged.begin(), merged.end());
    REQUIRE(merged == raster_full);
  }
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
      Catch::Matchers::ContainsSubstring(
          "DRIZZLE_MEMORY_BUDGET: retained/source buffers exceed budget"));
  REQUIRE_FALSE(loaded);
  cfg.chunk_rows = 8000;
  REQUIRE_THROWS(plan_drizzle_memory(plan, cfg, 3 * 53));
}

TEST_CASE(
    "forward drizzle: autogrow raises a too-small budget within headroom",
    "[drizzle-audit][drizzle-memory]") {
  RegistrationSamplingPlan plan;
  plan.source_width = 6000;
  plan.source_height = 4000;
  plan.canvas_width_native = 6000;
  plan.canvas_height_native = 4000;
  plan.color_mode = ColorMode::OSC;
  plan.bayer_pattern = BayerPattern::RGGB;
  plan.frames.push_back(
      make_affine_frame("f0", 0, make_source_to_canvas(1, 0, 0, 0, 1, 0)));
  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = 1;
  cfg.memory_budget_mb = 512;  // too small for the retained set below
  // ~1.1 GiB retained: needs roughly a 2-3 GiB effective budget.
  const size_t retained = 1100ull * 1024 * 1024;
  std::vector<std::string> warnings;
  const auto grown = plan_drizzle_memory_autogrow(
      plan, cfg, 3 * 53, retained, true,
      [&](const std::string &m) { warnings.push_back(m); });
  REQUIRE(grown.rows >= 1);
  REQUIRE(cfg.memory_budget_mb > 512);  // grew in +1 GiB steps until it fit
  REQUIRE(!warnings.empty());
  for (const auto &w : warnings)
    REQUIRE(w.find("raising drizzle memory budget") != std::string::npos);
}

TEST_CASE(
    "forward drizzle: autogrow still fails closed beyond the headroom cap",
    "[drizzle-audit][drizzle-memory]") {
  RegistrationSamplingPlan plan;
  plan.source_width = 6000;
  plan.source_height = 4000;
  plan.canvas_width_native = 6000;
  plan.canvas_height_native = 4000;
  plan.color_mode = ColorMode::OSC;
  plan.bayer_pattern = BayerPattern::RGGB;
  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = 1;
  cfg.memory_budget_mb = 512;
  // ~1 TiB retained cannot fit any realistic headroom: the +1 GiB steps stop
  // once the 80%-of-available cap binds.
  const size_t retained = (1ull << 40);
  std::vector<std::string> warnings;
  REQUIRE_THROWS_WITH(
      plan_drizzle_memory_autogrow(plan, cfg, 3 * 53, retained, true,
                                   [&](const std::string &m) {
                                     warnings.push_back(m);
                                   }),
      Catch::Matchers::ContainsSubstring("DRIZZLE_MEMORY_BUDGET"));
  REQUIRE(!warnings.empty());
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

// --- M7 slice 2: CPU <-> CUDA parity for the exact droplet/cell overlap area.
// The device kernel is a 1:1 port of polygon_rectangle_intersection_area; this
// is the first entry in the plan-19.5 parity matrix. Skips cleanly when no
// CUDA device is available (CUDA-free build, or CI without a GPU).
TEST_CASE("plan-19.5 parity: CUDA polygon_rect_area == CPU reference",
          "[forward-drizzle][cuda-parity]") {
  if (reconstruction::forward_drizzle_cuda_device_memory().free_bytes == 0) {
    SUCCEED("no CUDA device -- parity check skipped");
    return;
  }

  // A spread of cases: full cover, partial, straddle, disjoint, degenerate,
  // rotated parallelograms (affine droplets), sub-pixel shifts.
  std::vector<std::array<double, 8>> quads;
  std::vector<std::array<double, 4>> rects;
  auto add = [&](std::array<double, 8> q, std::array<double, 4> r) {
    quads.push_back(q);
    rects.push_back(r);
  };
  add({0, 0, 1, 0, 1, 1, 0, 1}, {0, 0, 1, 1});            // exact cell
  add({0.3, 0.3, 1.3, 0.3, 1.3, 1.3, 0.3, 1.3}, {0, 0, 1, 1});  // shifted
  add({2, 0, 3, 0, 3, 1, 2, 1}, {0, 0, 1, 1});            // disjoint -> 0
  add({-0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5}, {0, 0, 1, 1});  // corner
  add({0.1, 0.0, 1.1, 0.2, 0.9, 1.2, -0.1, 1.0}, {0, 0, 1, 1});  // parallelogram
  add({0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}, {0, 0, 1, 1});  // degenerate pt
  // A deterministic pseudo-random sweep of small affine droplets.
  uint64_t s = 0x2545F4914F6CDD1Dull;
  auto nextf = [&]() {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return static_cast<double>(s >> 11) / static_cast<double>(1ull << 53);
  };
  for (int i = 0; i < 4000; ++i) {
    const double cx = nextf() * 4.0, cy = nextf() * 4.0;
    const double a = nextf() * 6.28318, sc = 0.3 + nextf() * 1.4;
    const double ca = std::cos(a) * sc, sa = std::sin(a) * sc;
    // unit square [-0.5,0.5]^2 rotated+scaled about (cx,cy)
    const double lx[4] = {-0.5, 0.5, 0.5, -0.5}, ly[4] = {-0.5, -0.5, 0.5, 0.5};
    std::array<double, 8> q{};
    for (int k = 0; k < 4; ++k) {
      q[2 * k] = cx + ca * lx[k] - sa * ly[k];
      q[2 * k + 1] = cy + sa * lx[k] + ca * ly[k];
    }
    const int rx = static_cast<int>(nextf() * 4), ry = static_cast<int>(nextf() * 4);
    add(q, {double(rx), double(ry), double(rx + 1), double(ry + 1)});
  }

  const int n = static_cast<int>(quads.size());
  std::vector<double> flat_q(n * 8), flat_r(n * 4), gpu(n, -1.0);
  for (int i = 0; i < n; ++i) {
    std::copy(quads[i].begin(), quads[i].end(), flat_q.begin() + i * 8);
    std::copy(rects[i].begin(), rects[i].end(), flat_r.begin() + i * 4);
  }
  REQUIRE(reconstruction::forward_drizzle_cuda_polygon_rect_area_batch(
      flat_q.data(), flat_r.data(), n, gpu.data()));

  // Plan 19.6 numerical contract: with -ffp-contract=off (CPU) and --fmad=false
  // (CUDA) on this path, the device and host run the *same* double-precision
  // algorithm with no contraction on either side, so every area is
  // BIT-IDENTICAL. This is what lets the downstream discrete decisions
  // (clip accept/reject, veto, mask) match exactly CPU<->GPU.
  int bit_exact = 0;
  for (int i = 0; i < n; ++i) {
    const double px[4] = {quads[i][0], quads[i][2], quads[i][4], quads[i][6]};
    const double py[4] = {quads[i][1], quads[i][3], quads[i][5], quads[i][7]};
    const double cpu = polygon_rectangle_intersection_area(
        px, py, rects[i][0], rects[i][1], rects[i][2], rects[i][3]);
    if (cpu == gpu[i]) ++bit_exact;
    INFO("i=" << i << " cpu=" << cpu << " gpu=" << gpu[i]);
    REQUIRE(cpu == gpu[i]);
  }
  INFO("bit-exact " << bit_exact << " / " << n);
  REQUIRE(bit_exact == n);
}

// Plan-19.5 parity entry 2: the affine droplet corner map (build_affine_leaf +
// to_internal). Pure per-sample arithmetic -> expected bit-exact CPU<->CUDA.
TEST_CASE("plan-19.5 parity: CUDA affine leaf corners == CPU reference",
          "[forward-drizzle][cuda-parity]") {
  if (reconstruction::forward_drizzle_cuda_device_memory().free_bytes == 0) {
    SUCCEED("no CUDA device -- parity check skipped");
    return;
  }
  // A non-trivial affine: rotation + anisotropic scale + shear + translation.
  const double aff[6] = {1.017, -0.033, 12.5, 0.041, 0.994, -7.25};
  const int scale = 2;
  const double half = 0.4;  // pixfrac/2

  std::vector<double> samples;
  uint64_t s = 0x9E3779B97F4A7C15ull;
  auto nextf = [&]() {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return static_cast<double>(s >> 11) / static_cast<double>(1ull << 53);
  };
  for (int i = 0; i < 5000; ++i) {
    samples.push_back(nextf() * 4000.0);   // sx over a realistic sensor extent
    samples.push_back(nextf() * 3000.0);   // sy
  }
  const int n = static_cast<int>(samples.size() / 2);
  std::vector<double> gpu(n * 8, 0.0);
  REQUIRE(reconstruction::forward_drizzle_cuda_affine_leaf_corners_batch(
      aff, scale, half, samples.data(), n, gpu.data()));

  int bit_exact = 0;
  for (int i = 0; i < n; ++i) {
    const double sx = samples[2 * i], sy = samples[2 * i + 1];
    const double csx[4] = {sx - half, sx + half, sx + half, sx - half};
    const double csy[4] = {sy - half, sy - half, sy + half, sy + half};
    for (int k = 0; k < 4; ++k) {
      const double qx = aff[0] * csx[k] + aff[1] * csy[k] + aff[2];
      const double qy = aff[3] * csx[k] + aff[4] * csy[k] + aff[5];
      const double ex = qx * scale, ey = qy * scale;
      const double gx = gpu[i * 8 + 2 * k], gy = gpu[i * 8 + 2 * k + 1];
      if (ex == gx && ey == gy) ++bit_exact;
      INFO("i=" << i << " k=" << k << " cpu=(" << ex << "," << ey
                << ") gpu=(" << gx << "," << gy << ")");
      // Plan 19.6: -ffp-contract=off + --fmad=false -> the affine dot product
      // is not fused on either side -> bit-identical.
      REQUIRE(ex == gx);
      REQUIRE(ey == gy);
    }
  }
  INFO("bit-exact " << bit_exact << " / " << (n * 4));
  REQUIRE(bit_exact == n * 4);
}

namespace {

// Brute-force oracle for drizzle_affine_source_spans: source pixel (sx,sy)
// is active iff its transformed pixfrac droplet bounding box intersects the
// full native target width and the band's native y interval.
bool g8_brute_active(const WarpMatrix &m, float pixfrac, int sw, int sh,
                     int canvas_w, int band_y0, int band_y1, int sx, int sy) {
  const double a0 = m(0, 0), a1 = m(0, 1), a2 = m(0, 2);
  const double a3 = m(1, 0), a4 = m(1, 1), a5 = m(1, 2);
  const double half = pixfrac / 2.0;
  const double rx = half * (std::fabs(a0) + std::fabs(a1));
  const double ry = half * (std::fabs(a3) + std::fabs(a4));
  const double qx = a0 * (sx + 0.5) + a1 * (sy + 0.5) + a2;
  const double qy = a3 * (sx + 0.5) + a4 * (sy + 0.5) + a5;
  (void)sw;
  (void)sh;
  return qx + rx > 0.0 && qx - rx < static_cast<double>(canvas_w) &&
         qy + ry > static_cast<double>(band_y0) &&
         qy - ry < static_cast<double>(band_y1);
}

}  // namespace

TEST_CASE("affine source spans: analytic intervals equal brute-force droplet "
          "membership (tranche 8)",
          "[affine-spans]") {
  const int sw = 24, sh = 18;
  const double th = 10.0 * M_PI / 180.0;
  const std::vector<WarpMatrix> mats = {
      make_source_to_canvas(1, 0, 0, 0, 1, 0),                       // identity
      make_source_to_canvas(std::cos(th), -std::sin(th), 5.0,
                            std::sin(th), std::cos(th), 3.0),       // +10 deg
      make_source_to_canvas(std::cos(th), std::sin(th), 5.0,
                            -std::sin(th), std::cos(th), 3.0),      // -10 deg
      make_source_to_canvas(1, 0.25, 2.0, 0.1, 1, 1.0),             // shear
      make_source_to_canvas(1.4, 0, -3.0, 0, 0.7, 4.0),             // scale
      make_source_to_canvas(1, 0, 11.5, 0, 1, -4.5),                // translation
      make_source_to_canvas(std::cos(th), -std::sin(th), 30.0,
                            std::sin(th), std::cos(th), -6.0),      // edge clip
  };
  for (std::size_t mi = 0; mi < mats.size(); ++mi) {
    for (float pf : {0.2f, 0.8f, 1.0f}) {
      RegistrationSamplingPlan plan;
      plan.source_width = sw;
      plan.source_height = sh;
      plan.canvas_width_native = 40;
      plan.canvas_height_native = 30;
      plan.color_mode = ColorMode::MONO;
      plan.frames.push_back(
          make_affine_frame("f" + std::to_string(mi), 0, mats[mi]));
      const std::vector<std::pair<int, int>> bands = {
          {0, 30}, {0, 7}, {9, 5}, {14, 2}, {27, 3}, {29, 1}};
      for (const auto &band : bands) {
        const auto spans =
            drizzle_affine_source_spans(plan, plan.frames[0], pf, band.first,
                                        band.second);
        // Spans are ascending sy with non-empty half-open x intervals and
        // membership identical to the oracle at every coordinate.
        int prev_y = -1;
        for (const auto &s : spans) {
          REQUIRE(s.source_y > prev_y);
          REQUIRE(s.x_begin < s.x_end);
          REQUIRE(s.x_begin >= 0);
          REQUIRE(s.x_end <= sw);
          prev_y = s.source_y;
        }
        std::size_t si = 0;
        for (int sy = 0; sy < sh; ++sy) {
          const bool row_active =
              si < spans.size() && spans[si].source_y == sy;
          for (int sx = 0; sx < sw; ++sx) {
            const bool expect =
                g8_brute_active(mats[mi], pf, sw, sh,
                                plan.canvas_width_native, band.first,
                                band.first + band.second, sx, sy);
            const bool got =
                row_active && sx >= spans[si].x_begin && sx < spans[si].x_end;
            INFO("mat=" << mi << " pf=" << pf << " band=" << band.first
                        << "/" << band.second << " sx=" << sx << " sy=" << sy);
            REQUIRE(got == expect);
          }
          if (row_active) ++si;
        }
        REQUIRE(si == spans.size());
      }
    }
  }
}

TEST_CASE("affine source spans: invalid inputs and caller-owned output "
          "capacity",
          "[affine-spans]") {
  RegistrationSamplingPlan plan;
  plan.source_width = 8;
  plan.source_height = 8;
  plan.canvas_width_native = 16;
  plan.canvas_height_native = 16;
  plan.color_mode = ColorMode::MONO;
  plan.frames.push_back(
      make_affine_frame("f", 0, make_source_to_canvas(1, 0, 0, 0, 1, 0)));
  FrameSamplingTransform local = plan.frames[0];
  local.has_smooth_local_model = true;
  std::vector<DrizzleAffineSourceSpan> out;
  drizzle_affine_source_spans_into(plan, local, 1.0f, 0, 16, out);
  REQUIRE(out.empty());
  drizzle_affine_source_spans_into(plan, plan.frames[0], 0.0f, 0, 16, out);
  REQUIRE(out.empty());
  drizzle_affine_source_spans_into(plan, plan.frames[0], 1.0f, -1, 4, out);
  REQUIRE(out.empty());
  drizzle_affine_source_spans_into(plan, plan.frames[0], 1.0f, 0, 0, out);
  REQUIRE(out.empty());
  drizzle_affine_source_spans_into(plan, plan.frames[0], 1.0f, 14, 4, out);
  REQUIRE(out.empty());
  // Caller-owned output: capacity is preserved across fills.
  out.reserve(64);
  const DrizzleAffineSourceSpan *p0 = out.data();
  const std::size_t c0 = out.capacity();
  drizzle_affine_source_spans_into(plan, plan.frames[0], 1.0f, 0, 16, out);
  REQUIRE(out.size() == 8);
  REQUIRE(out.data() == p0);
  REQUIRE(out.capacity() == c0);
  drizzle_affine_source_spans_into(plan, plan.frames[0], 1.0f, 4, 4, out);
  REQUIRE(out.data() == p0);
  // Band fully below the transform's reach: empty, not a throw.
  RegistrationSamplingPlan plan2 = plan;
  plan2.frames[0] =
      make_affine_frame("g", 0, make_source_to_canvas(1, 0, 100, 0, 1, 100));
  drizzle_affine_source_spans_into(plan2, plan2.frames[0], 1.0f, 0, 16, out);
  REQUIRE(out.empty());
}
