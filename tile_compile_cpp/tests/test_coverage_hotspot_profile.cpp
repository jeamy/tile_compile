// Plan section 11.14.7 P6 follow-up --- SAMPLING_GEOMETRY hotspot attribution.
//
// The real 600-frame M31 (affine) run spent 2 h 21 min in SAMPLING_GEOMETRY.
// Static reading of compute_geometric_coverage (sampling_geometry.cpp:347+)
// shows two candidate costs per (stripe, frame):
//   A. geometry: two rasterize_drizzle_stripe passes (CFA droplet + dense
//      footprint). For an AFFINE frame the source scan is bounded to the
//      stripe's back-projected row band (forward_drizzle.cpp:492-512), so this
//      is ~1x the source across all stripes, not K x.
//   B. dense O(n) per-frame buffer traffic: std::fill(B[c]) + the
//      `for c, for i<n: if (B[c][i] > 0)` reduction + the footprint touched[]
//      scan --- run in full for EVERY frame regardless of how few stripe
//      pixels that frame actually covers.
//
// This hidden benchmark runs coverage on a scaled-down affine OSC scene and
// prints geometry_wall_s (from the geomstats ScopedGeometryTimer on both
// rasterize calls) against the measured total wall. The ratio says whether to
// attack geometry or memory traffic first. Hidden ([.]) so it never runs in
// the default suite; invoke explicitly:
//
//   ./tests "[coverage-profile]"
//
// Tunable via env (defaults chosen to finish in ~1 min):
//   TC_COVPROF_SRC_W, TC_COVPROF_SRC_H     source dims           (1920 x 1080)
//   TC_COVPROF_SCALE                        internal_scale        (2)
//   TC_COVPROF_FRAMES                       affine frame count    (24)
//   TC_COVPROF_CHUNK                        chunk_rows            (256)
//   TC_COVPROF_BUDGET_MB                    memory_budget_mb      (16384)

#include "tile_compile/reconstruction/drizzle_geometry_stats.hpp"
#include "tile_compile/reconstruction/forward_drizzle.hpp" // WarpMatrix, ColorMode
#include "tile_compile/registration/sampling_geometry.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace tile_compile;
using namespace tile_compile::reconstruction;
using tile_compile::registration::FrameSamplingTransform;
using tile_compile::registration::RegistrationSamplingPlan;

namespace {

int env_int(const char *k, int dflt) {
  if (const char *v = std::getenv(k)) {
    const int parsed = std::atoi(v);
    if (parsed > 0)
      return parsed;
  }
  return dflt;
}

WarpMatrix affine(double a, double b, double tx, double c, double d, double ty) {
  WarpMatrix m;
  m(0, 0) = static_cast<float>(a);
  m(0, 1) = static_cast<float>(b);
  m(0, 2) = static_cast<float>(tx);
  m(1, 0) = static_cast<float>(c);
  m(1, 1) = static_cast<float>(d);
  m(1, 2) = static_cast<float>(ty);
  return m;
}

// A pure-affine OSC plan: N frames with small sub-pixel dithers and a tiny
// per-frame shift, no local model. Mirrors the M31 class (0 local models).
RegistrationSamplingPlan make_affine_osc_plan(int src_w, int src_h, int frames) {
  RegistrationSamplingPlan plan;
  plan.source_width = src_w;
  plan.source_height = src_h;
  plan.canvas_width_native = src_w + 64;
  plan.canvas_height_native = src_h + 64;
  plan.color_mode = ColorMode::OSC;
  plan.bayer_pattern = BayerPattern::RGGB;
  plan.cfa_origin_x = 0;
  plan.cfa_origin_y = 0;

  for (int i = 0; i < frames; ++i) {
    FrameSamplingTransform f;
    f.frame_id = "aff" + std::to_string(i);
    f.source_index = static_cast<size_t>(i);
    f.valid = true;
    // Dither across a ~3 px box, deterministic, no rotation.
    const double dx = 32.0 + std::fmod(i * 0.618, 1.0) * 3.0 + (i % 5) * 0.25;
    const double dy = 32.0 + std::fmod(i * 0.414, 1.0) * 3.0 + (i % 3) * 0.25;
    f.source_to_canvas = affine(1.0, 0.0, dx, 0.0, 1.0, dy);
    f.source_to_canvas_affine_valid = true;
    f.has_smooth_local_model = false;
    plan.frames.push_back(f);
  }
  return plan;
}

} // namespace

TEST_CASE("SAMPLING_GEOMETRY hotspot: geometry wall vs total wall on an affine "
          "coverage run",
          "[.][coverage-profile]") {
  namespace gs = geomstats;

  const int src_w = env_int("TC_COVPROF_SRC_W", 1920);
  const int src_h = env_int("TC_COVPROF_SRC_H", 1080);
  const int scale = env_int("TC_COVPROF_SCALE", 2);
  const int frames = env_int("TC_COVPROF_FRAMES", 24);
  const int chunk = env_int("TC_COVPROF_CHUNK", 256);
  const int budget_mb = env_int("TC_COVPROF_BUDGET_MB", 16384);
  const int workers = env_int("TC_COVPROF_WORKERS", 1);

  const RegistrationSamplingPlan plan =
      make_affine_osc_plan(src_w, src_h, frames);

  config::ReconstructionDrizzleConfig res;
  res.internal_scale = scale;
  res.pixfrac = 0.8f;
  res.chunk_rows = chunk;
  res.memory_budget_mb = static_cast<size_t>(budget_mb);
  const config::ReconstructionCoverageGateConfig gate_cfg{};

  gs::registry().reset();
  gs::ScopedEnable _e(true);

  const auto t0 = std::chrono::steady_clock::now();
  auto coverage = registration::compute_geometric_coverage(
      plan, scale, 0.8f, gate_cfg, /*fraction=*/0.5f, /*num_workers=*/workers,
      res, /*retain_channel_counts=*/false);
  const auto t1 = std::chrono::steady_clock::now();
  const double total_s =
      std::chrono::duration<double>(t1 - t0).count();

  const auto &reg = gs::registry();
  const auto &cfa = reg.v[static_cast<std::size_t>(gs::Variant::kCoverageCfa)];
  const auto &foot =
      reg.v[static_cast<std::size_t>(gs::Variant::kCoverageFootprint)];
  const double geom_s = cfa.geometry_wall_s + foot.geometry_wall_s;
  const double other_s = total_s - geom_s;

  const long long internal_w =
      static_cast<long long>(coverage.internal_width);
  const long long internal_h =
      static_cast<long long>(coverage.internal_height);
  const int stripes = reg.stripe_count;
  const long long n_per_stripe =
      internal_w * std::min<long long>(chunk, internal_h);

  std::printf("\n=== SAMPLING_GEOMETRY coverage hotspot profile ===\n");
  std::printf("scene: src %dx%d  scale %d  internal %lldx%lld  frames %d  "
              "chunk %d  stripes %d  workers req %d used %d\n",
              src_w, src_h, scale, internal_w, internal_h, frames, chunk,
              stripes, workers, coverage.gate.workers_used);
  std::printf("n per stripe (internal_w * chunk) : %lld cells\n", n_per_stripe);
  std::printf("dense O(n) reduction touches      : ~%.2e "
              "(stripes * frames * 3ch * n)\n",
              static_cast<double>(stripes) * frames * 3.0 *
                  static_cast<double>(n_per_stripe));
  std::printf("cfa   rasterize wall_s            : %.3f  "
              "(source_samples_visited %llu)\n",
              cfa.geometry_wall_s,
              (unsigned long long)cfa.source_samples_visited);
  std::printf("foot  rasterize wall_s            : %.3f  "
              "(source_samples_visited %llu)\n",
              foot.geometry_wall_s,
              (unsigned long long)foot.source_samples_visited);
  std::printf("------------------------------------------------\n");
  std::printf("TOTAL coverage wall_s             : %.3f\n", total_s);
  if (geom_s > 0.0) {
    std::printf("  geometry (2x rasterize)        : %.3f  (%.1f%%)\n", geom_s,
                100.0 * geom_s / total_s);
    std::printf("  everything else (fill+scan+holes+quantile+prepare) : "
                "%.3f  (%.1f%%)\n",
                other_s, 100.0 * other_s / total_s);
  } else {
    std::printf("  (geomstats suppressed under workers>1 --- total wall "
                "only)\n");
  }
  std::printf("================================================\n\n");

  // No behavioural assertion --- this is a measurement harness. Sanity only.
  REQUIRE(total_s > 0.0);
  REQUIRE(stripes >= 1);
  REQUIRE(coverage.gate.valid_frame_count == frames);
}
