// P6 CPU FORWARD_DRIZZLE hotspot attribution (plan 11.14.7 follow-up).
//
// The 600-frame real runs never finished FORWARD_DRIZZLE. §30.69/§30.70
// extrapolated the phase from micro-benches; this hidden benchmark measures
// the actual CPU cost split on a scaled-down AFFINE OSC scene, so the
// optimisation effort attaches to the dominant term rather than an assumed
// one.
//
// Repair 2026-09-10 (§30.77, plan §0.2 priority 1). The pre-repair version
// zeroed A/B in an isolated fill section BEFORE the isolated gather/reduce
// sections, so both ran on empty state (`B > 0` never fired, `cnt` stayed 0)
// and measured a read sweep / a skip loop instead of production work. The
// repaired structure mirrors the production per-frame order
// (fill -> raster/accumulate -> gather, each timed separately) so the reduce
// section processes real candidates, plus:
//   * assert that candidates were actually written and reductions executed;
//   * deterministic per-frame source variation + sparse outliers so robust
//     clipping engages; bit-exact comparison of the split-computed stripe
//     against the production full-canvas result (same providers, same
//     reduce config) as the reference check;
//   * the one-time hoist allocation timed separately (production pays it
//     once per run, the per-stripe alloc timer does not include it);
//   * big micro-benchmark buffers freed before the production sweep so they
//     cannot skew chunk planning / memory pressure; every sweep point prints
//     requested/budgeted/used workers, resolved chunk rows and peak bytes;
//     speedups are stated explicitly against the W=1 point;
//   * actual source-visit counters (unique source pixels, leaf visits)
//     replace the blanket /4 OSC denominator.
// enum-vs-clip attribution remains a DIFFERENCE estimate of two runs with
// different callback costs -- an order-of-magnitude indicator, not an
// isolated profile (§30.77).
//
// Hidden ([.]) --- never runs in the default suite. Invoke explicitly:
//   ./tests "[fd-hotspot]"
// Tunable via env (defaults ~1 min):
//   TC_FDPROF_SRC_W / TC_FDPROF_SRC_H     source dims        (1920 x 1080)
//   TC_FDPROF_SCALE                        internal_scale     (2)
//   TC_FDPROF_FRAMES                       affine frame count (16)
//   TC_FDPROF_CHUNK                        chunk_rows, 0 = auto (256)
//   TC_FDPROF_BUDGET_MB                    memory_budget_mb   (16384)
//   TC_FDPROF_SCALING                      "1,2,4,8,16" band-worker sweep
//   TC_FD_PROFILE                          additionally prints the
//                                          production stripe timers

#include "tile_compile/reconstruction/forward_drizzle.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
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

// Like env_int but 0 is a valid value (used for chunk_rows = 0 = auto).
int env_int0(const char *k, int dflt) {
  if (const char *v = std::getenv(k)) {
    if (*v) return std::atoi(v);
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

// Deterministic per-frame source: base gradient + small frame-dependent
// jitter + sparse strong outliers, so the robust clipping stage actually
// performs rejections (uniform inputs would leave it idle).
void fill_source(Matrix2Df &dst, std::size_t fi) {
  const int w = dst.cols(), h = dst.rows();
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float v = 100.0f + 0.01f * static_cast<float>((x * 7 + y * 13) % 512);
      v += 0.5f * static_cast<float>(
          (fi * 31 + static_cast<std::size_t>(x) * 3 +
           static_cast<std::size_t>(y) * 5) % 11);
      const std::uint64_t mix =
          static_cast<std::uint64_t>(x) * 2654435761ull +
          static_cast<std::uint64_t>(y) * 40503ull +
          static_cast<std::uint64_t>(fi) * 97ull;
      if (mix % 4096ull == 0ull) v *= 8.0f;  // sparse deterministic outlier
      dst(y, x) = v;
    }
}

bool float_bits_equal(float a, float b) {
  return a == b || (std::isnan(a) && std::isnan(b));
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

  // Single shared provider buffer, refilled deterministically per frame index
  // (same contract as production: re-loaded per frame).
  Matrix2Df src_cur(src_h, src_w);
  SourceImageProvider source_of = [&](std::size_t fi) -> const Matrix2Df & {
    fill_source(src_cur, fi);
    return src_cur;
  };

  const int chunk = env_int0("TC_FDPROF_CHUNK", 256);  // 0 => auto
  config::ReconstructionDrizzleConfig dz;
  dz.internal_scale = scale;
  dz.pixfrac = pixfrac;
  dz.chunk_rows = chunk;
  dz.memory_budget_mb = static_cast<size_t>(budget_mb);
  config::ReconstructionClippingConfig clip_cfg;
  dz.min_clip_contributors = 5;
  dz.robust_passes = 2;
  // The micro-split runs on a single bounded stripe so its own scratch stays
  // small; the full-pipeline section below uses dz.chunk_rows as given.
  const int split_rows = std::min(chunk > 0 ? chunk : 256, ih);

  std::printf("\n=== FORWARD_DRIZZLE CPU hotspot profile (repaired 2026-09-10) ===\n");
  std::printf("scene: src %dx%d  scale %d  internal %dx%d  frames %d  "
              "chunk_rows %d  budget %d MB\n",
              src_w, src_h, scale, iw, ih, frames, chunk, budget_mb);

  // --- Split on ONE production-like stripe, all frames, 1 thread. ---
  // rows = chunk (or ih if chunk == 0); the stripe sits mid-canvas so every
  // frame's dithered footprint fully covers it.
  const int channels = 3;
  const int rows = split_rows;
  const int y_begin = std::max(0, (ih - rows) / 2);
  const size_t n = static_cast<size_t>(iw) * static_cast<size_t>(rows);
  const int nstripes = (ih + rows - 1) / rows;

  auto t = [] { return std::chrono::steady_clock::now(); };

  // One-time hoist allocation: production allocates these buffers once per
  // run sized to the largest stripe (the per-stripe alloc timer does not
  // include this). Timed separately here.
  const auto h0 = t();
  std::vector<std::vector<double>> A(channels), Bv(channels);
  std::vector<std::vector<uint32_t>> cnt(channels);
  std::vector<std::vector<ClipCandidate>> cand(channels);
  for (int c = 0; c < channels; ++c) {
    A[c].assign(n, 0.0);
    Bv[c].assign(n, 0.0);
    cnt[c].assign(n, 0);
    cand[c].assign(n * static_cast<size_t>(frames), ClipCandidate{});
  }
  const double d_hoist_alloc = secs(h0, t());

  // enum only -- with actual source-visit counters (unique source pixels and
  // leaf visits) instead of a blanket /4 source-pixel estimate.
  std::uint64_t cells = 0, leaf_visits = 0;
  std::vector<std::uint8_t> src_seen(static_cast<size_t>(src_w) * src_h, 0);
  const auto e0 = t();
  for (const auto &f : plan.frames) {
    std::uint32_t prev_visit = 0xFFFFFFFFu;
    enumerate_drizzle_stripe_leaf_cells(
        plan, f, scale, pixfrac, y_begin, rows,
        [&](int sx, int sy, int, int leaf_order, int, int, const double *,
            const double *) {
          ++cells;
          src_seen[static_cast<size_t>(sy) * src_w + sx] = 1;
          // Canonical callback order is (sy, sx, leaf_order, cy, cx): a key
          // change marks the start of a new leaf's cell enumeration.
          const std::uint32_t key =
              (static_cast<std::uint32_t>(sy * src_w + sx) << 4) ^
              static_cast<std::uint32_t>(leaf_order);
          if (key != prev_visit) {
            ++leaf_visits;
            prev_visit = key;
          }
        });
  }
  const auto e1 = t();
  std::uint64_t unique_src_px = 0;
  for (auto b : src_seen) unique_src_px += b;
  // enum + clip. NOTE: clip-vs-enum attribution is a difference estimate of
  // two runs whose callbacks cost differently (this one also writes the
  // visit bitmap) -- order of magnitude only, see header comment.
  volatile double clip_sink = 0.0;
  std::uint64_t clip_calls = 0, clip_zero = 0, clip_full = 0;
  for (const auto &f : plan.frames)
    enumerate_drizzle_stripe_leaf_cells(
        plan, f, scale, pixfrac, y_begin, rows,
        [&](int, int, int, int, int cx, int cy, const double *lx,
            const double *ly) {
          const double a = polygon_rectangle_intersection_area(lx, ly, cx, cy,
                                                              cx + 1.0, cy + 1.0);
          clip_sink += a;
          ++clip_calls;
          if (a == 0.0) ++clip_zero;
          if (a == 1.0) ++clip_full;  // discarded k=1.0 shortcut's key stat
        });
  const auto e2 = t();
  std::printf("clip calls %llu  zero-area %llu (%.1f%%)  full-cover %llu  "
              "leaf visits %llu  unique src px %llu  "
              "clips/leaf-visit %.2f  leaf-visits/src-px %.2f\n",
              (unsigned long long)clip_calls, (unsigned long long)clip_zero,
              100.0 * static_cast<double>(clip_zero) /
                  static_cast<double>(clip_calls ? clip_calls : 1),
              (unsigned long long)clip_full, (unsigned long long)leaf_visits,
              (unsigned long long)unique_src_px,
              static_cast<double>(clip_calls) /
                  static_cast<double>(leaf_visits ? leaf_visits : 1),
              static_cast<double>(leaf_visits) /
                  static_cast<double>(unique_src_px ? unique_src_px : 1));

  // Per-frame production order: fill -> raster/accumulate -> gather, each
  // timed separately and accumulated. After the frame loop the candidate
  // buffer holds the full N-frame state the reduce stage sees.
  double d_fill = 0.0, d_accum = 0.0, d_gather = 0.0;
  std::uint64_t cand_written = 0;
  for (const auto &f : plan.frames) {
    auto s = t();
    for (int c = 0; c < channels; ++c) {
      std::fill(A[c].begin(), A[c].end(), 0.0);
      std::fill(Bv[c].begin(), Bv[c].end(), 0.0);
    }
    d_fill += secs(s, t());
    fill_source(src_cur, f.source_index);
    s = t();
    rasterize_drizzle_stripe(
        plan, f, scale, pixfrac, y_begin, rows,
        [&](int sx, int sy, int c, int, size_t i, double k) {
          const double v = src_cur(sy, sx);
          if (!std::isfinite(v)) return;
          A[c][i] += k * v;
          Bv[c][i] += k;
        });
    d_accum += secs(s, t());
    s = t();
    for (int c = 0; c < channels; ++c)
      for (size_t i = 0; i < n; ++i)
        if (Bv[c][i] > 0.0) {
          cand[c][i * frames + (cnt[c][i]++)] = {
              f.source_index, A[c][i] / Bv[c][i], Bv[c][i], 1.0, 1.0, 1.0,
              1.0, false};
          ++cand_written;
        }
    d_gather += secs(s, t());
  }
  std::uint64_t counts_sum = 0;
  for (int c = 0; c < channels; ++c)
    for (size_t i = 0; i < n; ++i) counts_sum += cnt[c][i];
  REQUIRE(cand_written == counts_sum);
  REQUIRE(cand_written > 0);

  // reduce over every pixel's populated candidate span.
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
  std::uint64_t reductions_executed = 0;
  const auto r5 = t();
  for (int c = 0; c < channels; ++c)
    for (size_t i = 0; i < n; ++i) {
      if (!cnt[c][i]) continue;
      const std::span<const ClipCandidate> pix(cand[c].data() + i * frames,
                                               cnt[c][i]);
      reduce_pixel_profiles(pix, rc, g_eff_for, reg_by_source, i, &up, &rp,
                            nullptr, nullptr, nullptr, nullptr, nullptr, diag);
      ++reductions_executed;
    }
  const double d_reduce = secs(r5, t());
  REQUIRE(reductions_executed > 0);

  // Isolated gather on EXPLICITLY prepared, fully populated inputs: worst-
  // case occupancy (every pixel covered). NOT a production simulation ---
  // printed separately and excluded from the stripe sum.
  double d_gather_worst = 0.0;
  {
    for (int c = 0; c < channels; ++c) {
      std::fill(A[c].begin(), A[c].end(), 123.0);
      std::fill(Bv[c].begin(), Bv[c].end(), 1.0);
      std::fill(cnt[c].begin(), cnt[c].end(), 0u);
    }
    const auto g0 = t();
    for (const auto &f : plan.frames) {
      (void)f;
      for (int c = 0; c < channels; ++c)
        for (size_t i = 0; i < n; ++i)
          cand[c][i * frames + (cnt[c][i]++)] = {0, A[c][i], Bv[c][i], 1.0,
                                                 1.0, 1.0, 1.0, false};
    }
    d_gather_worst = secs(g0, t());
  }

  const double d_enum = secs(e0, e1);
  const double d_clip = secs(e1, e2) - d_enum;
  const double d_stripe_sum =
      d_enum + d_clip + d_fill + d_accum + d_gather + d_reduce;

  // --- Free the big micro-benchmark buffers BEFORE the production sweep so
  // they cannot skew available memory / chunk planning of the sweep runs. ---
  for (int c = 0; c < channels; ++c) {
    std::vector<double>().swap(A[c]);
    std::vector<double>().swap(Bv[c]);
    std::vector<std::uint32_t>().swap(cnt[c]);
    std::vector<ClipCandidate>().swap(cand[c]);
  }
  std::vector<std::uint8_t>().swap(src_seen);

  // --- Full pipeline (ground truth) + band-parallel sweep ---
  double t_w1 = -1.0;
  ForwardDrizzleUniformAndRawResult ref;  // W=1 reference for the compare
  std::vector<std::pair<int, double>> sweep_res;
  auto run_full = [&](int w, bool keep_ref) {
    const auto w0 = t();
    auto r = compute_forward_drizzle_uniform_and_raw(plan, source_of, dz,
                                                     clip_cfg, {}, {}, {}, {}, w);
    const double tw = secs(w0, t());
    const bool has_plane =
        !r.uniform.R.value.empty() || !r.uniform.L.value.empty();
    REQUIRE(has_plane);
    std::printf("  workers req %2d  budgeted %2d  used %2d  chunk %4d  "
                "peak %zu MB : %8.3f s\n",
                r.diagnostics.workers_requested, r.diagnostics.workers_budgeted,
                r.diagnostics.workers_used, r.diagnostics.resolved_chunk_rows,
                r.diagnostics.estimated_peak_bytes / (1024 * 1024), tw);
    if (keep_ref) ref = std::move(r);
    return tw;
  };
  std::printf("-------- band-parallel scaling (full compute_..._uniform_and_raw;"
              " speedup vs W=1) --------\n");
  bool have_w1 = false;
  for (int w : sweep) if (w == 1) have_w1 = true;
  if (!have_w1) t_w1 = run_full(1, true);
  for (int w : sweep) {
    const double tw = run_full(w, w == 1);
    if (w == 1) t_w1 = tw;
    sweep_res.push_back({w, tw});
  }

  // Reference check: the split-computed stripe planes must be bit-identical
  // to the production full-canvas result over [y_begin, y_begin + rows).
  // Same provider values, same candidate construction, same reduce config,
  // frame-ordered per-pixel candidate spans on both sides.
  std::uint64_t mismatches = 0;
  const std::array<const ProfilePlane *, 3> ref_up{&ref.uniform.R,
                                                   &ref.uniform.G,
                                                   &ref.uniform.B};
  const std::array<const ProfilePlane *, 3> ref_rp{&ref.raw.R, &ref.raw.G,
                                                   &ref.raw.B};
  // Recompute the split planes (buffers were freed): rerun fill->accum->
  // gather->reduce cheaply into fresh planes.
  {
    std::vector<std::vector<double>> A2(channels), B2(channels);
    std::vector<std::vector<std::uint32_t>> cnt2(channels);
    std::vector<std::vector<ClipCandidate>> cand2(channels);
    for (int c = 0; c < channels; ++c) {
      A2[c].assign(n, 0.0);
      B2[c].assign(n, 0.0);
      cnt2[c].assign(n, 0);
      cand2[c].assign(n * static_cast<size_t>(frames), ClipCandidate{});
    }
    ProfilePlane up2, rp2;
    up2.allocate(iw, rows);
    rp2.allocate(iw, rows);
    for (const auto &f : plan.frames) {
      for (int c = 0; c < channels; ++c) {
        std::fill(A2[c].begin(), A2[c].end(), 0.0);
        std::fill(B2[c].begin(), B2[c].end(), 0.0);
      }
      fill_source(src_cur, f.source_index);
      rasterize_drizzle_stripe(
          plan, f, scale, pixfrac, y_begin, rows,
          [&](int sx, int sy, int c, int, size_t i, double k) {
            const double v = src_cur(sy, sx);
            if (!std::isfinite(v)) return;
            A2[c][i] += k * v;
            B2[c][i] += k;
          });
      for (int c = 0; c < channels; ++c)
        for (size_t i = 0; i < n; ++i)
          if (B2[c][i] > 0.0)
            cand2[c][i * frames + (cnt2[c][i]++)] = {
                f.source_index, A2[c][i] / B2[c][i], B2[c][i], 1.0, 1.0, 1.0,
                1.0, false};
    }
    ForwardDrizzleClippingDiagnostics diag2;
    for (int c = 0; c < channels; ++c)
      for (size_t i = 0; i < n; ++i) {
        if (!cnt2[c][i]) continue;
        const std::span<const ClipCandidate> pix(cand2[c].data() + i * frames,
                                                 cnt2[c][i]);
        reduce_pixel_profiles(pix, rc, g_eff_for, reg_by_source, i, &up2, &rp2,
                              nullptr, nullptr, nullptr, nullptr, nullptr,
                              diag2);
      }
    auto cmp_plane = [&](const ProfilePlane &got,
                         const ProfilePlane &fullref) {
      if (fullref.empty()) return;
      for (size_t i = 0; i < n; ++i) {
        const size_t p = static_cast<size_t>(y_begin + i / iw) * iw + (i % iw);
        if (!float_bits_equal(got.value[i], fullref.value[p])) ++mismatches;
        if (!float_bits_equal(got.weight_sum[i], fullref.weight_sum[p]))
          ++mismatches;
        if (!float_bits_equal(got.n_eff[i], fullref.n_eff[p])) ++mismatches;
        if (got.support[i] != fullref.support[p]) ++mismatches;
      }
    };
    // Production keeps three separate per-channel planes; run the reduce per
    // channel into separate plane pairs for the comparison.
    for (int c = 0; c < channels; ++c) {
      ProfilePlane uc, rc2;
      uc.allocate(iw, rows);
      rc2.allocate(iw, rows);
      ForwardDrizzleClippingDiagnostics d3;
      for (size_t i = 0; i < n; ++i) {
        if (!cnt2[c][i]) continue;
        const std::span<const ClipCandidate> pix(cand2[c].data() + i * frames,
                                                 cnt2[c][i]);
        reduce_pixel_profiles(pix, rc, g_eff_for, reg_by_source, i, &uc, &rc2,
                              nullptr, nullptr, nullptr, nullptr, nullptr, d3);
      }
      cmp_plane(uc, *ref_up[c]);
      cmp_plane(rc2, *ref_rp[c]);
    }
    std::printf("reference compare (split stripe vs production full run): "
                "%llu mismatches over %zu cells x 2 profiles x 3 channels\n",
                (unsigned long long)mismatches, n);
    REQUIRE(mismatches == 0);
  }

  std::printf("one stripe: y_begin %d rows %d  (%d stripes cover the canvas)  "
              "cells emitted (all frames) %llu\n",
              y_begin, rows, nstripes, (unsigned long long)cells);
  std::printf("hoist alloc (once per production run): %.3f s\n",
              d_hoist_alloc);
  std::printf("-------- per-stripe split (1 thread, all frames; production "
              "order: fill + accum + gather per frame) --------\n");
  auto line = [&](const char *nm, double s) {
    std::printf("  %-8s %8.3f s  (%5.1f%% of stripe)  x %d stripes = %7.2f s\n",
                nm, s, 100.0 * s / (d_stripe_sum > 0 ? d_stripe_sum : 1),
                nstripes, s * nstripes);
  };
  line("enum", d_enum);
  line("clip*", d_clip);  // * difference estimate, see header comment
  line("fill", d_fill);
  line("accum", d_accum);
  line("gather", d_gather);
  line("reduce", d_reduce);
  std::printf("  gather-worst (isolated, fully populated, NOT in sum): "
              "%8.3f s\n", d_gather_worst);
  std::printf("  -------------------------------------------\n");
  std::printf("  stripe sum %8.3f s   x %d = %7.2f s   (full 1-worker run "
              "%7.2f s)\n",
              d_stripe_sum, nstripes, d_stripe_sum * nstripes, t_w1);
  std::printf("  scaling (vs W=1 = %.3f s):\n", t_w1);
  for (auto [w, tw] : sweep_res)
    std::printf("    workers %2d : %8.3f s   speedup %5.2fx\n", w, tw,
                t_w1 > 0 ? t_w1 / tw : 0.0);
  std::printf("==========================================\n\n");

  REQUIRE(t_w1 > 0.0);
  REQUIRE(cells > 0);
  REQUIRE(leaf_visits > 0);
  REQUIRE(unique_src_px > 0);
  REQUIRE(clip_calls > 0);
  REQUIRE(d_stripe_sum > 0.0);
  // Outlier engagement evidence from the production path itself.
  REQUIRE(ref.clipping.pixel_channel_evaluations > 0);
  REQUIRE(ref.clipping.candidate_contributions_clipped > 0);
}
