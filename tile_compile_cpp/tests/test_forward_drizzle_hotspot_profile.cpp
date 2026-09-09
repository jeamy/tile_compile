// P6 CPU FORWARD_DRIZZLE hotspot attribution (plan 11.14.7 follow-up).
//
// The 600-frame real runs never finished FORWARD_DRIZZLE. §30.69/§30.70
// extrapolated the phase from micro-benches; this hidden benchmark measures
// the actual CPU cost split on a scaled-down AFFINE OSC scene, so the
// optimisation effort attaches to the dominant term rather than an assumed one.
//
// Four-way split (single stripe, one reduction worker):
//   enum   : enumerate_drizzle_stripe_leaf_cells with a no-op sink
//   clip   : + polygon_rectangle_intersection_area per emitted cell
//   accum  : + the real A[c]+=k*v / B[c]+=k accumulate lambda
//   reduce : compute_forward_drizzle_uniform_and_raw total minus the above
// Plus band-parallel reduction scaling (workers 1..16) on the same scene.
//
// Hidden ([.]) --- never runs in the default suite. Invoke explicitly:
//   ./tests "[fd-hotspot]"
// Tunable via env (defaults ~1 min):
//   TC_FDPROF_SRC_W / TC_FDPROF_SRC_H     source dims        (1920 x 1080)
//   TC_FDPROF_SCALE                        internal_scale     (2)
//   TC_FDPROF_FRAMES                       affine frame count (16)
//   TC_FDPROF_BUDGET_MB                    memory_budget_mb   (16384)
//   TC_FDPROF_SCALING                      "1,2,4,8,16" band-worker sweep

#include "tile_compile/reconstruction/forward_drizzle.hpp"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

using namespace tile_compile;
using namespace tile_compile::reconstruction;
using tile_compile::registration::FrameSamplingTransform;
using tile_compile::registration::RegistrationSamplingPlan;

namespace {

int env_int(const char *k, int dflt) {
  if (const char *v = std::getenv(k)) {
    const int p = std::atoi(v);
    if (p > 0) return p;
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

// Pure-affine OSC plan, mirrors the M31 class (0 local models): small
// sub-pixel dithers + a tiny per-frame shift, no rotation.
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
    const double dx = 32.0 + std::fmod(i * 0.618, 1.0) * 3.0 + (i % 5) * 0.25;
    const double dy = 32.0 + std::fmod(i * 0.414, 1.0) * 3.0 + (i % 3) * 0.25;
    f.source_to_canvas = affine(1.0, 0.0, dx, 0.0, 1.0, dy);
    f.source_to_canvas_affine_valid = true;
    f.has_smooth_local_model = false;
    plan.frames.push_back(f);
  }
  return plan;
}

double secs(std::chrono::steady_clock::time_point a,
            std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

} // namespace

TEST_CASE("FORWARD_DRIZZLE hotspot: CPU cost split + band-parallel scaling on "
          "an affine OSC scene",
          "[.][fd-hotspot]") {
  const int src_w = env_int("TC_FDPROF_SRC_W", 1920);
  const int src_h = env_int("TC_FDPROF_SRC_H", 1080);
  const int scale = env_int("TC_FDPROF_SCALE", 2);
  const int frames = env_int("TC_FDPROF_FRAMES", 16);
  const int budget_mb = env_int("TC_FDPROF_BUDGET_MB", 16384);
  std::vector<int> sweep;
  {
    std::string s = "1,2,4,8,16";
    if (const char *v = std::getenv("TC_FDPROF_SCALING")) s = v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
      if (!tok.empty()) sweep.push_back(std::atoi(tok.c_str()));
  }

  const RegistrationSamplingPlan plan = make_affine_osc_plan(src_w, src_h, frames);
  const int iw = (src_w + 64) * scale;
  const int ih = (src_h + 64) * scale;
  const float pixfrac = 0.8f;

  // One synthetic source, reused for every frame (the geometry cost --- what
  // this profiles --- does not depend on pixel values).
  Matrix2Df source(src_h, src_w);
  for (int y = 0; y < src_h; ++y)
    for (int x = 0; x < src_w; ++x)
      source(y, x) = 100.0f + 0.01f * static_cast<float>((x * 7 + y * 13) % 512);
  SourceImageProvider source_of = [&](std::size_t) -> const Matrix2Df & {
    return source;
  };

  const int chunk = env_int("TC_FDPROF_CHUNK", 256);
  config::ReconstructionDrizzleConfig dz;
  dz.internal_scale = scale;
  dz.pixfrac = pixfrac;
  dz.chunk_rows = chunk;
  dz.memory_budget_mb = static_cast<size_t>(budget_mb);
  config::ReconstructionClippingConfig clip_cfg;
  dz.min_clip_contributors = 5;
  dz.robust_passes = 2;

  std::printf("\n=== FORWARD_DRIZZLE CPU hotspot profile ===\n");
  std::printf("scene: src %dx%d  scale %d  internal %dx%d  frames %d  "
              "chunk_rows %d  budget %d MB\n",
              src_w, src_h, scale, iw, ih, frames, chunk, budget_mb);

  // --- Six-way split on ONE production-like stripe, all frames, 1 thread. ---
  // rows = chunk (or ih if chunk == 0); the stripe sits mid-canvas so every
  // frame's dithered footprint fully covers it.
  const int channels = 3;
  const int rows = chunk > 0 ? std::min(chunk, ih) : ih;
  const int y_begin = std::max(0, (ih - rows) / 2);
  const size_t n = static_cast<size_t>(iw) * static_cast<size_t>(rows);
  const int nstripes = (ih + rows - 1) / rows;

  std::vector<std::vector<double>> A(channels), Bv(channels);
  std::vector<std::vector<uint32_t>> cnt(channels);
  std::vector<std::vector<ClipCandidate>> cand(channels);
  for (int c = 0; c < channels; ++c) {
    A[c].assign(n, 0.0);
    Bv[c].assign(n, 0.0);
    cnt[c].assign(n, 0);
    cand[c].assign(n * static_cast<size_t>(frames), ClipCandidate{});
  }

  auto t = [] { return std::chrono::steady_clock::now(); };

  // enum only
  std::uint64_t cells = 0;
  const auto e0 = t();
  for (const auto &f : plan.frames)
    enumerate_drizzle_stripe_leaf_cells(
        plan, f, scale, pixfrac, y_begin, rows,
        [&](int, int, int, int, int, int, const double *, const double *) {
          ++cells;
        });
  const auto e1 = t();
  // enum + clip
  volatile double clip_sink = 0.0;
  for (const auto &f : plan.frames)
    enumerate_drizzle_stripe_leaf_cells(
        plan, f, scale, pixfrac, y_begin, rows,
        [&](int, int, int, int, int cx, int cy, const double *lx,
            const double *ly) {
          clip_sink += polygon_rectangle_intersection_area(lx, ly, cx, cy,
                                                           cx + 1.0, cy + 1.0);
        });
  const auto e2 = t();
  // enum + clip + accumulate (A/B), with a per-frame fill like production
  for (const auto &f : plan.frames) {
    for (int c = 0; c < channels; ++c) {
      std::fill(A[c].begin(), A[c].end(), 0.0);
      std::fill(Bv[c].begin(), Bv[c].end(), 0.0);
    }
    rasterize_drizzle_stripe(
        plan, f, scale, pixfrac, y_begin, rows,
        [&](int sx, int sy, int c, int, size_t i, double k) {
          const double v = source(sy, sx);
          if (!std::isfinite(v)) return;
          A[c][i] += k * v;
          Bv[c][i] += k;
        });
  }
  const auto e3 = t();
  // isolated: per-frame fill of A/B over the whole stripe (12x full sweeps)
  for (const auto &f : plan.frames) {
    (void)f;
    for (int c = 0; c < channels; ++c) {
      std::fill(A[c].begin(), A[c].end(), 0.0);
      std::fill(Bv[c].begin(), Bv[c].end(), 0.0);
    }
  }
  const auto e4 = t();
  // isolated: per-frame candidate gather sweep (reads B, writes ClipCandidate)
  for (const auto &f : plan.frames)
    for (int c = 0; c < channels; ++c)
      for (size_t i = 0; i < n; ++i)
        if (Bv[c][i] > 0.0)
          cand[c][i * frames + (cnt[c][i]++)] = {
              f.source_index, A[c][i] / Bv[c][i], Bv[c][i], 1.0, 1.0, 1.0, 1.0,
              false};
  const auto e5 = t();
  // isolated: reduce_pixel_profiles over every pixel's candidate span
  const DrizzleProfileReduceConfig rc{dz.min_clip_contributors,
                                      dz.robust_passes,
                                      clip_cfg.clip_sigma_low,
                                      clip_cfg.clip_sigma_high,
                                      clip_cfg.min_fraction,
                                      clip_cfg.min_n_eff,
                                      false,
                                      false,
                                      false,
                                      4.0f,
                                      2.0f,
                                      {}};
  ProfilePlane up, rp;
  up.allocate(iw, rows);
  rp.allocate(iw, rows);
  auto g_eff_for = [](std::size_t) { return 1.0; };
  const std::vector<std::pair<std::uint8_t, float>> reg_by_source;
  ForwardDrizzleClippingDiagnostics diag;
  for (int c = 0; c < channels; ++c)
    for (size_t i = 0; i < n; ++i) {
      if (!cnt[c][i]) continue;
      const std::span<const ClipCandidate> pix(cand[c].data() + i * frames,
                                               cnt[c][i]);
      reduce_pixel_profiles(pix, rc, g_eff_for, reg_by_source, i, &up, &rp,
                            nullptr, nullptr, nullptr, nullptr, nullptr, diag);
    }
  const auto e6 = t();

  const double d_enum = secs(e0, e1);
  const double d_clip = secs(e1, e2) - d_enum;
  const double d_accum = secs(e2, e3) - secs(e1, e2);
  const double d_fill = secs(e3, e4);
  const double d_gather = secs(e4, e5);
  const double d_reduce = secs(e5, e6);
  const double d_stripe_sum =
      d_enum + d_clip + d_accum + d_fill + d_gather + d_reduce;

  // --- Full pipeline (ground truth) + band-parallel sweep ---
  double base = 0.0, t_full1 = 0.0;
  int resolved_chunk = 0;
  std::vector<std::pair<int, double>> sweep_res;
  for (int w : sweep) {
    const auto w0 = t();
    auto r = compute_forward_drizzle_uniform_and_raw(plan, source_of, dz,
                                                     clip_cfg, {}, {}, {}, {}, w);
    const double tw = secs(w0, t());
    if (w == sweep.front()) {
      base = tw;
      t_full1 = tw;
      resolved_chunk = r.diagnostics.resolved_chunk_rows;
    }
    sweep_res.push_back({w, tw});
    const bool has_plane =
        !r.uniform.R.value.empty() || !r.uniform.L.value.empty();
    REQUIRE(has_plane);
  }

  std::printf("one stripe: y_begin %d rows %d  (%d stripes cover the canvas)  "
              "cells emitted (all frames) %llu\n",
              y_begin, rows, nstripes, (unsigned long long)cells);
  std::printf("full run resolved_chunk_rows: %d\n", resolved_chunk);
  std::printf("-------- per-stripe six-way split (1 thread, all frames) "
              "--------\n");
  auto line = [&](const char *nm, double s) {
    std::printf("  %-8s %8.3f s  (%5.1f%% of stripe)  x %d stripes = %7.2f s\n",
                nm, s, 100.0 * s / (d_stripe_sum > 0 ? d_stripe_sum : 1),
                nstripes, s * nstripes);
  };
  line("enum", d_enum);
  line("clip", d_clip);
  line("accum", d_accum);
  line("fill", d_fill);
  line("gather", d_gather);
  line("reduce", d_reduce);
  std::printf("  -------------------------------------------\n");
  std::printf("  stripe sum %8.3f s   x %d = %7.2f s   (full 1-worker run "
              "%7.2f s)\n",
              d_stripe_sum, nstripes, d_stripe_sum * nstripes, t_full1);
  std::printf("-------- band-parallel scaling (full compute_..._uniform_and_raw)"
              " --------\n");
  for (auto [w, tw] : sweep_res)
    std::printf("  workers %2d : %8.3f s   speedup %5.2fx\n", w, tw,
                base > 0 ? base / tw : 0.0);
  std::printf("==========================================\n\n");

  REQUIRE(t_full1 > 0.0);
  REQUIRE(cells > 0);
  REQUIRE(d_stripe_sum > 0.0);
}
