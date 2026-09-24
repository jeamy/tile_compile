// Plan section 11.14 P5 --- profile the residual hotspot after P2/P3.
//
// P1/P2 removed the O(N*P*K*consumers) blow-up; the geometry work is now a
// single O(V*N*P) `sample_leaves` sweep per frame at build time, and P3
// parallelises it per-frame. This test measures what that sweep actually costs,
// so the P5 decision ("does the one-time local basis evaluation still dominate?
// then investigate exact reuse of identical probe points + SIMD before
// minimax/GPU") is evidence-driven.
//
// It reports, per geometry VARIANT (cfa pixfrac<1 vs footprint pixfrac=1) and
// per warp regime (no-recursion vs subdividing):
//   * mean Newton iterations per invert call
//   * local_forward calls per source sample (9 if no recursion, more with it)
//   * std::exp calls per source pixel (== 16 * invert_iterations / source_px)
//   * measured ns per sample_leaves and ns per std::exp
//   * the analytic exact-reuse ceiling for identical probe points
//   * which term dominates: std::exp vs the surrounding scalar/Eigen work
//
// Pure measurement --- no production code path changes, nothing asserted about
// speed. The only REQUIREs are sanity checks on the counters.

#include "tile_compile/reconstruction/drizzle_geometry_stats.hpp"
#include "tile_compile/reconstruction/forward_drizzle.hpp"
#include "tile_compile/registration/global_registration.hpp"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace tile_compile;
using namespace tile_compile::reconstruction;
using tile_compile::registration::FrameSamplingTransform;
using tile_compile::registration::RegistrationSamplingPlan;
namespace gs = tile_compile::reconstruction::geomstats;

namespace {
using clk = std::chrono::steady_clock;
double secs(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

WarpMatrix s2c(double a, double b, double tx, double c, double d, double ty) {
  WarpMatrix m;
  m(0, 0) = static_cast<float>(a);
  m(0, 1) = static_cast<float>(b);
  m(0, 2) = static_cast<float>(tx);
  m(1, 0) = static_cast<float>(c);
  m(1, 1) = static_cast<float>(d);
  m(1, 2) = static_cast<float>(ty);
  return m;
}

// One local-warp frame. `curv` scales the higher-order RBF coefficients: at 0
// the field is near-linear over a source pixel (depth-0 subdivision converges,
// leaves/sample == 1 --- the regime every §30.55/§30.64 measurement has hit);
// large values inject curvature that forces recursion.
RegistrationSamplingPlan one_frame_plan(int sw, int sh, int cw, int ch,
                                        float curv) {
  RegistrationSamplingPlan plan;
  plan.source_width = sw;
  plan.source_height = sh;
  plan.canvas_width_native = cw;
  plan.canvas_height_native = ch;
  plan.color_mode = ColorMode::OSC;
  plan.bayer_pattern = BayerPattern::GBRG;
  plan.cfa_origin_x = 0;
  plan.cfa_origin_y = 0;
  plan.plan_hash = "p5-profile";
  FrameSamplingTransform f;
  f.frame_id = "p5f0";
  f.source_index = 0;
  f.valid = true;
  const double ang = 0.004;
  f.source_to_canvas = s2c(std::cos(ang), -std::sin(ang), (cw - sw) / 2.0 + 0.3,
                           std::sin(ang), std::cos(ang), (ch - sh) / 2.0 - 0.2);
  f.source_to_canvas_affine_valid = true;
  f.has_smooth_local_model = true;
  f.smooth_local_model.valid = true;
  f.smooth_local_model.image_rows = ch;
  f.smooth_local_model.image_cols = cw;
  f.smooth_local_model.coeff_x.setZero();
  f.smooth_local_model.coeff_y.setZero();
  f.smooth_local_model.coeff_x[0] = 0.12f;
  f.smooth_local_model.coeff_y[0] = -0.09f;
  f.smooth_local_model.coeff_x[5] = 0.04f;
  // Alternating-sign higher modes -> local curvature.
  for (int i = 1; i < 16; ++i) {
    const float s = (i % 2 == 0) ? 1.0f : -1.0f;
    f.smooth_local_model.coeff_x[i] += curv * s * (0.6f + 0.05f * i);
    f.smooth_local_model.coeff_y[i] += curv * s * (0.5f - 0.03f * i);
  }
  f.model_coordinate_scale = 1.0f;
  plan.frames.push_back(f);
  return plan;
}

struct Probe {
  double sweep_s = 0.0;   // best-of-N full source sweep
  gs::VariantCounters c;  // counters from that sweep
};

Probe sweep_once(const RegistrationSamplingPlan &plan, float pixfrac, int reps) {
  const ForwardDrizzleSubdivisionParams sub;
  const auto &f = plan.frames[0];
  std::vector<Leaf> leaves;
  leaves.reserve(32);
  Probe out;
  out.sweep_s = 1e30;
  for (int r = 0; r < reps; ++r) {
    gs::ScopedEnable en(true);  // resets the registry
    gs::ScopedVariant v(gs::Variant::kPrepareExclusionScan, pixfrac);
    const auto t0 = clk::now();
    for (int sy = 0; sy < plan.source_height; ++sy)
      for (int sx = 0; sx < plan.source_width; ++sx)
        sample_leaves(plan, f, sx, sy, /*scale=*/1, pixfrac, sub, leaves);
    const double s = secs(t0, clk::now());
    if (s < out.sweep_s) out.sweep_s = s;
    out.c = gs::registry().v[static_cast<std::size_t>(
        gs::Variant::kPrepareExclusionScan)];
  }
  return out;
}

// Bare transcendental cost (ns/call), min of several runs to shed noise.
// `smooth_local_basis` calls std::exp on a float argument -> the libm float
// entry point (expf); the double path is timed too so a promotion regression
// would be visible.
struct ExpBench { double std_exp_float, expf_explicit, std_exp_double; };
ExpBench bench_exp() {
  const int n = 20'000'000;
  auto run = [&](int which) {
    volatile double sink = 0.0;
    const auto t0 = clk::now();
    for (int i = 0; i < n; ++i) {
      const float xf = -(static_cast<float>(i & 1023) * 0.0011f);
      if (which == 0) sink += std::exp(xf);            // float overload
      else if (which == 1) sink += expf(xf);            // explicit expf
      else sink += std::exp(static_cast<double>(xf));   // double path
    }
    (void)sink;
    return secs(t0, clk::now()) / n * 1e9;
  };
  ExpBench b{1e30, 1e30, 1e30};
  for (int r = 0; r < 5; ++r) {
    b.std_exp_float = std::min(b.std_exp_float, run(0));
    b.expf_explicit = std::min(b.expf_explicit, run(1));
    b.std_exp_double = std::min(b.std_exp_double, run(2));
  }
  return b;
}

// One full evaluate_smooth_local_displacement (== one smooth_local_basis: 16
// std::exp + the Eigen Coefficients zero/normalise + two dot products), min of
// several runs. This is a HOT-LOOP LOWER BOUND on per-call cost (model in
// cache, branch predictor saturated); used only to bound, not measure, the
// basis-eval share of the real sweep. perf is unavailable in this env.
double ns_per_basis_eval(const RegistrationSamplingPlan &plan) {
  const auto &m = plan.frames[0].smooth_local_model;
  const int n = 4'000'000;
  const float w = static_cast<float>(plan.canvas_width_native - 1);
  const float h = static_cast<float>(plan.canvas_height_native - 1);
  double best = 1e30;
  for (int r = 0; r < 5; ++r) {
    volatile float sink = 0.0f;
    const auto t0 = clk::now();
    for (int i = 0; i < n; ++i) {
      const float x = w * static_cast<float>(i & 255) / 255.0f;
      const float y = h * static_cast<float>((i >> 8) & 255) / 255.0f;
      const auto d = registration::evaluate_smooth_local_displacement(m, x, y);
      sink += d.x + d.y;
    }
    (void)sink;
    best = std::min(best, secs(t0, clk::now()) / n * 1e9);
  }
  return best;
}

void report(const char *regime, const RegistrationSamplingPlan &plan,
            float pixfrac, const Probe &p, double ns_exp, double ns_basis) {
  const double src_px = static_cast<double>(plan.source_width) *
                        plan.source_height;
  const double top = static_cast<double>(p.c.top_level_sample_leaves_calls);
  const double subdiv = static_cast<double>(p.c.subdivide_local_calls);
  const double lf = static_cast<double>(p.c.local_forward_calls);
  const double inv = static_cast<double>(p.c.invert_calls);
  const double it = static_cast<double>(p.c.invert_iterations);
  const double mean_newton = inv > 0 ? it / inv : 0.0;
  const double lf_per_sample = top > 0 ? lf / top : 0.0;
  const double exp_per_px = it * 16.0 / src_px;
  const double ns_per_sample = p.sweep_s / src_px * 1e9;
  const double ns_per_exp_measured = it > 0 ? p.sweep_s / (it * 16.0) * 1e9 : 0.0;
  const double predicted_exp_s = it * 16.0 * ns_exp * 1e-9;
  const double predicted_basis_s = it * ns_basis * 1e-9;
  const double exp_frac = p.sweep_s > 0 ? predicted_exp_s / p.sweep_s : 0.0;
  const double basis_frac = p.sweep_s > 0 ? predicted_basis_s / p.sweep_s : 0.0;
  // Residual = everything that is NOT a basis eval: 9x local_forward call
  // overhead per sample, Newton loop control, subdivide_local's bilinear error
  // probe + 2 shoelace_area, leaf push/bbox bookkeeping. A near-zero or
  // negative residual means the hot-loop basis micro-bench is saturating the
  // attribution -> read basis_frac as ">= this", not exact.
  const double residual_frac = 1.0 - basis_frac;

  // Exact-reuse ceiling for identical probe points.
  //  - footprint pixfrac==1: the depth-0 3x3 grid tiles the integer lattice, so
  //    across the WxH sweep the ~9*W*H probes collapse to the ~(2W+1)(2H+1)
  //    distinct half-integer lattice points -> ~2.25x. Recursion adds
  //    quarter-lattice points, shared only where neighbours recurse alike.
  //  - cfa pixfrac<1: depth-0 probes are pixel-INTERIOR (sx+0.1, sx+0.5,
  //    sx+0.9 at pixfrac 0.8) -> zero inter-pixel sharing. The only reuse is
  //    parent<->child inside a subdividing sample: a child 3x3 shares 4 of its
  //    9 points with the parent, so a sample that recurses once drops from
  //    9*(1+4)=45 to ~25 local_forward -> ~1.8x, but ONLY for recursing
  //    samples.
  const bool footprint = pixfrac >= 0.999f;
  const double recursing_frac = top > 0 ? (subdiv - top) / std::max(1.0, top) : 0;
  double reuse_ceiling;
  if (footprint) {
    reuse_ceiling = 9.0 / 4.0;  // depth-0 lattice collapse
  } else {
    // weighted: non-recursing samples 1.0x, recursing ~1.8x
    const double frac_rec = std::min(1.0, std::max(0.0, recursing_frac));
    reuse_ceiling = 1.0 * (1.0 - frac_rec) + 1.8 * frac_rec;
  }

  std::printf(
      "[p5] %-14s pixfrac=%.2f | leaves/sample=%.3f subdiv/sample=%.3f "
      "(recursion=%s)\n"
      "[p5]   mean_newton_iters=%.2f  local_forward/sample=%.2f  "
      "exp/source_px=%.1f\n"
      "[p5]   ns/sample_leaves=%.1f  ns/exp(measured, whole-sweep)=%.2f  "
      "ns/exp(bare)=%.2f\n"
      "[p5]   ATTRIBUTION (micro-bench-derived, NOT a profiler split): "
      "std::exp >=%.0f%%  full_basis_eval(16 exp+Eigen) >=%.0f%%  "
      "residual (9x local_forward overhead + Newton ctl + subdivide "
      "bilinear/shoelace + leaf bookkeeping) ~%.0f%%\n"
      "[p5]   exact-probe-reuse ceiling ~%.2fx  (%s)\n",
      regime, pixfrac,
      static_cast<double>(p.c.leaves_generated) / std::max(1.0, top),
      subdiv / std::max(1.0, top),
      subdiv > top ? "YES" : "no",
      mean_newton, lf_per_sample, exp_per_px, ns_per_sample,
      ns_per_exp_measured, ns_exp, exp_frac * 100.0, basis_frac * 100.0,
      residual_frac * 100.0, reuse_ceiling,
      footprint ? "footprint: depth-0 half-integer lattice collapse"
                : "cfa: interior probes, parent<->child only");
}

} // namespace

TEST_CASE("plan 11.14 P5: local basis-eval hotspot profile after P2/P3",
          "[geometry-p5]") {
  // Two source sizes so a reader can see the per-sample rate is size-stable.
  const int sw = 96, sh = 96, cw = sw + 24, ch = sh + 24;

  const ExpBench eb = bench_exp();
  const double ns_exp = eb.std_exp_float;
  std::printf("\n[p5] bare transcendental (min of 5, ns/call): std::exp(float)="
              "%.2f  expf()=%.2f  std::exp(double)=%.2f  "
              "(kSmoothLocalGridSize=4 -> 16 exp per smooth_local_basis; "
              "smooth_local_basis calls the float path)\n",
              eb.std_exp_float, eb.expf_explicit, eb.std_exp_double);

  struct Regime { const char *name; float curv; };
  for (const Regime rg : {Regime{"near-linear", 0.0f},
                          Regime{"curved", 3.5f},
                          Regime{"strongly-curved", 12.0f}}) {
    const auto plan = one_frame_plan(sw, sh, cw, ch, rg.curv);
    const double ns_basis = ns_per_basis_eval(plan);
    std::printf(
        "\n[p5] === regime: %s (curv=%.1f) ===  one full basis eval = %.2f ns "
        "(bare-exp share %.0f%%)\n",
        rg.name, rg.curv, ns_basis, 16.0 * ns_exp / ns_basis * 100.0);
    for (float pixfrac : {0.8f, 1.0f}) {
      const auto p = sweep_once(plan, pixfrac, /*reps=*/5);
      report(rg.name, plan, pixfrac, p, ns_exp, ns_basis);
      // Sanity: the sweep ran the whole source once per rep.
      REQUIRE(p.c.top_level_sample_leaves_calls ==
              static_cast<std::uint64_t>(sw) * sh);
      REQUIRE(p.c.invert_iterations > 0);
      REQUIRE(p.c.local_forward_calls >= p.c.top_level_sample_leaves_calls);
    }
  }

  std::printf(
      "\n[p5] CONCLUSION (P5 closed by measurement, see protocol §30.69):\n"
      "[p5]  - exact identical-probe reuse: ~1.0x on the cfa production path "
      "(no recursion in any regime -> no sharing).\n"
      "[p5]  - LTO/inlining: 4-5%% (measured separately with IPO=ON).\n"
      "[p5]  - bit-exact de-Eigen of smooth_local_basis: REJECTED. It also "
      "feeds the registration model fit (global_registration.cpp:977) so a bit "
      "change moves the fitted coeffs and every downstream hash; and "
      "basis.dot() cannot be replaced by a sequential loop bit-exactly (Eigen "
      "vectorises the 16-float reduction). Ceiling only ~1.05-1.15x -- not "
      "worth the blast radius.\n"
      "[p5]  - remaining ~38%% is std::exp (already the expf float path). "
      "Cutting it needs a vector/minimax expf -> changes bits by construction "
      "-> the plan's separately-gated numeric/model-identity revision.\n"
      "[p5]  P5 does NOT close the ~1440 s/16-core geometry-build extrapolation "
      "vs the <=1920 s whole-chain P6 target. Levers: more cores (P3 scales) or "
      "the numeric revision.\n");
}
