// §30.81 step-5 / step-3a benchmark: attribute the per-column-tile repetition
// factor of the CUDA affine pair path.
//
// Hidden ([.]), needs a real device AND `TC_FD_CUDA_PROFILE` set. It runs the
// same affine multiband build three ways over one internal band --- full width
// (A, 1 tile), N external per-tile calls (B), and the internal tiled driver
// (C) --- and prints the coarse phase split (producer / device
// malloc+upload+kernel+download / host sort / host reduce) plus the
// tiled/single ratio. It asserts every reassembly is bit-identical to A.
//
// §30.81 step 3a removed the per-band record memo: each column tile is now
// produced through a WINDOWED producer, so total records sorted stays 1x the
// full-width call (asserted) instead of Nx. The device phases are still ~Nx
// because the affine kernel does not yet take an X-window --- that is step 3;
// this bench is its baseline.
//
//   TC_FD_CUDA_PROFILE=1 ./tests "[fd-cuda-tile]" --success

#include "tile_compile/reconstruction/forward_drizzle.hpp"
#include "tile_compile/reconstruction/forward_drizzle_contrib_list.hpp"
#include "tile_compile/reconstruction/forward_drizzle_cuda.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace tile_compile;
using namespace tile_compile::reconstruction;
using tile_compile::registration::FrameSamplingTransform;
using tile_compile::registration::RegistrationSamplingPlan;

namespace {

WarpMatrix s2c(double a, double b, double tx, double c, double d, double ty) {
  WarpMatrix m;
  m(0, 0) = static_cast<float>(a); m(0, 1) = static_cast<float>(b);
  m(0, 2) = static_cast<float>(tx);
  m(1, 0) = static_cast<float>(c); m(1, 1) = static_cast<float>(d);
  m(1, 2) = static_cast<float>(ty);
  return m;
}

struct Snap {
  double malloc_s, upload_s, kernel_s, download_s, produce_s, sort_s, reduce_s;
  unsigned long long calls, records_sorted;
};
Snap snap() {
  const auto &p = forward_drizzle_cuda_profile();
  return {p.dev_malloc_s.load(),   p.dev_upload_s.load(),
          p.dev_kernel_s.load(),   p.dev_download_s.load(),
          p.produce_s.load(),      p.sort_s.load(),
          p.reduce_s.load(),       p.calls.load(),
          p.records_sorted.load()};
}

}  // namespace

TEST_CASE("§30.81 step-5 baseline: CUDA affine pair path column-tile repetition "
          "factor",
          "[.][fd-cuda-tile]") {
  if (forward_drizzle_cuda_device_memory().free_bytes == 0) {
    SUCCEED("no CUDA device");
    return;
  }
  if (!forward_drizzle_cuda_profile_enabled()) {
    SUCCEED("set TC_FD_CUDA_PROFILE=1 to collect the phase split");
    return;
  }

  RegistrationSamplingPlan plan;
  plan.source_width = 320;
  plan.source_height = 48;
  plan.canvas_width_native = 480;   // W = 960 at internal_scale 2
  plan.canvas_height_native = 40;   // H = 80
  plan.color_mode = ColorMode::OSC;
  plan.bayer_pattern = BayerPattern::RGGB;
  plan.cfa_origin_x = plan.cfa_origin_y = 0;

  const int nf = 6;
  for (int i = 0; i < nf; ++i) {
    const double ang = 0.010 * std::sin(0.5 * i);
    FrameSamplingTransform f;
    f.frame_id = "f" + std::to_string(i);
    f.source_index = static_cast<size_t>(i);
    f.valid = f.source_to_canvas_affine_valid = true;
    f.model_prediction_factor = 1.0f;
    f.registration_residual_factor = 1.0f;
    f.source_to_canvas =
        s2c(1.25 * std::cos(ang), -1.15 * std::sin(ang), 6.0 + 0.3 * i,
            1.10 * std::sin(ang), 1.20 * std::cos(ang), 4.0 + 0.2 * i);
    plan.frames.push_back(f);
  }

  std::vector<Matrix2Df> src, qc, q0, q1, qa;
  for (int i = 0; i < nf; ++i) {
    Matrix2Df s(plan.source_height, plan.source_width);
    Matrix2Df c(plan.source_height, plan.source_width);
    Matrix2Df a0(plan.source_height, plan.source_width);
    Matrix2Df a1(plan.source_height, plan.source_width);
    Matrix2Df ar(plan.source_height, plan.source_width);
    for (int y = 0; y < plan.source_height; ++y)
      for (int x = 0; x < plan.source_width; ++x) {
        s(y, x) = 30.0f + 0.4f * x - 0.2f * y + 0.5f * i +
                  9.0f * std::sin(0.3f * (x + y));
        c(y, x) = 0.4f + 0.4f * std::fabs(std::sin(0.2f * (x + y) + i));
        a0(y, x) = 0.3f + 0.5f * std::fabs(std::cos(0.15f * x - 0.1f * y + i));
        a1(y, x) = 0.35f + 0.4f * std::fabs(std::sin(0.1f * x + 0.2f * y - i));
        ar(y, x) = ((x + y + i) % 5 == 0) ? std::nanf("")
                                          : 0.6f + 0.3f * std::sin(0.5f * x + i);
      }
    s(5 + i, 40) = 900.0f;  // clip outlier
    src.push_back(std::move(s));
    qc.push_back(std::move(c));
    q0.push_back(std::move(a0));
    q1.push_back(std::move(a1));
    qa.push_back(std::move(ar));
  }
  SourceImageProvider provider = [&](size_t i) -> const Matrix2Df & {
    return src[i];
  };
  // §30.81 step 3a-2: a real rect provider --- slices the requested source
  // rectangle of the full maps into fresh matrices with the matching origin,
  // exactly like the SQM cache reader's read_rect. Exercises the rebasing on
  // the CUDA store path in this bench.
  std::vector<Matrix2Df> rc(nf), r0(nf), r1(nf), ra(nf);
  FrameQualityRectProvider quality_of =
      [&](size_t i, int y0, int y1, int x0, int x1) -> FrameQualityMaps {
    const int h = qc[i].rows(), w = qc[i].cols();
    if (y1 < 0) y1 = h;
    if (x1 < 0) x1 = w;
    y0 = std::clamp(y0, 0, h); y1 = std::clamp(y1, y0, h);
    x0 = std::clamp(x0, 0, w); x1 = std::clamp(x1, x0, w);
    auto slice = [&](const Matrix2Df &full, Matrix2Df &dst) {
      dst.resize(y1 - y0, x1 - x0);
      for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x) dst(y - y0, x - x0) = full(y, x);
    };
    slice(qc[i], rc[i]);
    slice(q0[i], r0[i]);
    slice(q1[i], r1[i]);
    slice(qa[i], ra[i]);
    FrameQualityMaps m{&rc[i], &r0[i], &r1[i], &ra[i]};
    m.y_origin = y0;
    m.x_origin = x0;
    return m;
  };

  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = 2;
  cfg.pixfrac = 0.9f;
  cfg.min_clip_contributors = 3;
  cfg.robust_passes = 2;
  config::ReconstructionClippingConfig clip;
  clip.clip_sigma_low = clip.clip_sigma_high = 2.5f;
  clip.min_fraction = 0.1f;
  clip.min_n_eff = 1.0f;
  MultibandProfileParams mb;
  mb.emit_fine = mb.emit_medium = mb.emit_alpha_confidence = true;
  const std::vector<float> g_eff = {1.0f, 0.95f, 0.9f, 0.88f, 0.92f, 0.97f};
  const std::size_t budget = static_cast<std::size_t>(1) << 32;

  const int W = plan.canvas_width_native * cfg.internal_scale;   // 960
  const int H = plan.canvas_height_native * cfg.internal_scale;  // 80

  // --- Run A: one full-width call -------------------------------------------
  forward_drizzle_cuda_profile().reset();
  const auto single = accumulate_pair_by_frame_cuda(
      plan, provider, cfg, clip, 0, H, {}, g_eff, quality_of, mb, budget);
  const Snap a = snap();

  // --- Run B: the same band split into 8 column tiles ---------------------
  const int tiles = 8;
  const int tile_w = (W + tiles - 1) / tiles;
  forward_drizzle_cuda_profile().reset();
  ForwardDrizzleUniformAndRawResult tiled;
  auto init = [&](ForwardDrizzleUniformResult &p, bool on) {
    p.color_mode = ColorMode::OSC;
    p.internal_width = W;
    p.internal_height = on ? H : 0;
    if (on) { p.R.allocate(W, H); p.G.allocate(W, H); p.B.allocate(W, H); }
  };
  init(tiled.uniform, true); init(tiled.raw, true);
  init(tiled.fine, true); init(tiled.medium, true);
  const std::size_t N = static_cast<std::size_t>(W) * H;
  tiled.a_separation.assign(N, std::nanf(""));
  tiled.a_artifact.assign(N, std::nanf(""));
  tiled.a_registration.assign(N, std::nanf(""));
  tiled.alpha_confidence_support.assign(N, 0u);

  for (int xb = 0; xb < W; xb += tile_w) {
    const int tw = std::min(tile_w, W - xb);
    const auto part = accumulate_pair_by_frame_cuda(
        plan, provider, cfg, clip, 0, H, {}, g_eff, quality_of, mb, budget, 32,
        static_cast<std::size_t>(1) << 20, nullptr, xb, tw);
    auto blit = [&](ForwardDrizzleUniformResult &d,
                    const ForwardDrizzleUniformResult &s) {
      const std::array<ProfilePlane *, 3> dp{&d.R, &d.G, &d.B};
      const std::array<const ProfilePlane *, 3> sp{&s.R, &s.G, &s.B};
      for (int c = 0; c < 3; ++c) {
        if (sp[c]->value.empty()) continue;
        for (int y = 0; y < H; ++y)
          for (int x = 0; x < tw; ++x) {
            const std::size_t si = static_cast<std::size_t>(y) * tw + x;
            const std::size_t di = static_cast<std::size_t>(y) * W + (xb + x);
            dp[c]->value[di] = sp[c]->value[si];
            dp[c]->weight_sum[di] = sp[c]->weight_sum[si];
            dp[c]->n_eff[di] = sp[c]->n_eff[si];
            dp[c]->support[di] = sp[c]->support[si];
          }
      }
    };
    blit(tiled.uniform, part.uniform);
    blit(tiled.raw, part.raw);
    blit(tiled.fine, part.fine);
    blit(tiled.medium, part.medium);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < tw; ++x) {
        const std::size_t si = static_cast<std::size_t>(y) * tw + x;
        const std::size_t di = static_cast<std::size_t>(y) * W + (xb + x);
        tiled.a_separation[di] = part.a_separation[si];
        tiled.a_artifact[di] = part.a_artifact[si];
        tiled.a_registration[di] = part.a_registration[si];
        tiled.alpha_confidence_support[di] = part.alpha_confidence_support[si];
      }
  }
  const Snap b = snap();

  // --- Run C: the §30.81 step-3a internal tiled driver -----------------
  // One accumulate_pair_by_frame_cuda call with a tile sink: the tiled branch
  // reduces each column tile from a WINDOWED producer call (no record memo),
  // stitching the parts through the sink.
  forward_drizzle_cuda_profile().reset();
  ForwardDrizzleUniformAndRawResult memo;
  init(memo.uniform, true); init(memo.raw, true);
  init(memo.fine, true); init(memo.medium, true);
  memo.a_separation.assign(N, std::nanf(""));
  memo.a_artifact.assign(N, std::nanf(""));
  memo.a_registration.assign(N, std::nanf(""));
  memo.alpha_confidence_support.assign(N, 0u);
  PairTileSink memo_sink = [&](int xb, int tw,
                               const ForwardDrizzleUniformAndRawResult &part) {
    auto blit = [&](ForwardDrizzleUniformResult &d,
                    const ForwardDrizzleUniformResult &s) {
      const std::array<ProfilePlane *, 3> dp{&d.R, &d.G, &d.B};
      const std::array<const ProfilePlane *, 3> sp{&s.R, &s.G, &s.B};
      for (int c = 0; c < 3; ++c) {
        if (sp[c]->value.empty()) continue;
        for (int y = 0; y < H; ++y)
          for (int x = 0; x < tw; ++x) {
            const std::size_t si = static_cast<std::size_t>(y) * tw + x;
            const std::size_t di = static_cast<std::size_t>(y) * W + (xb + x);
            dp[c]->value[di] = sp[c]->value[si];
            dp[c]->weight_sum[di] = sp[c]->weight_sum[si];
            dp[c]->n_eff[di] = sp[c]->n_eff[si];
            dp[c]->support[di] = sp[c]->support[si];
          }
      }
    };
    blit(memo.uniform, part.uniform);
    blit(memo.raw, part.raw);
    blit(memo.fine, part.fine);
    blit(memo.medium, part.medium);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < tw; ++x) {
        const std::size_t si = static_cast<std::size_t>(y) * tw + x;
        const std::size_t di = static_cast<std::size_t>(y) * W + (xb + x);
        memo.a_separation[di] = part.a_separation[si];
        memo.a_artifact[di] = part.a_artifact[si];
        memo.a_registration[di] = part.a_registration[si];
        memo.alpha_confidence_support[di] = part.alpha_confidence_support[si];
      }
  };
  accumulate_pair_by_frame_cuda(plan, provider, cfg, clip, 0, H, {}, g_eff,
                                quality_of, mb, budget, 32,
                                static_cast<std::size_t>(1) << 20, nullptr, 0,
                                -1, &memo_sink, tile_w);
  const Snap cc = snap();

  // --- correctness: both reassemblies == single -------------------------
  auto same_plane = [](const ProfilePlane &x, const ProfilePlane &y) {
    REQUIRE(x.support == y.support);
    REQUIRE(x.weight_sum == y.weight_sum);
    REQUIRE(x.n_eff == y.n_eff);
    for (size_t i = 0; i < x.value.size(); ++i)
      if (x.support[i]) REQUIRE(x.value[i] == y.value[i]);
  };
  for (auto *t : {&tiled, &memo})
    for (auto pr : {std::pair{&single.uniform, &t->uniform},
                    std::pair{&single.raw, &t->raw},
                    std::pair{&single.fine, &t->fine},
                    std::pair{&single.medium, &t->medium}}) {
      same_plane(pr.first->R, pr.second->R);
      same_plane(pr.first->G, pr.second->G);
      same_plane(pr.first->B, pr.second->B);
    }
  REQUIRE(single.alpha_confidence_support == tiled.alpha_confidence_support);
  REQUIRE(single.alpha_confidence_support == memo.alpha_confidence_support);

  // §30.81 step 3a gate: the windowed producer keeps TOTAL record work at 1x
  // the full-width call --- each tile emits only its own window's records.
  // (The device phases are still ~Nx; the kernel X-window is step 3.)
  REQUIRE(b.records_sorted == a.records_sorted);
  REQUIRE(cc.records_sorted == a.records_sorted);

  // --- report ------------------------------------------------------------
  auto dev = [](const Snap &s) {
    return s.malloc_s + s.upload_s + s.kernel_s + s.download_s;
  };
  auto tot = [&](const Snap &s) {
    return s.produce_s + s.sort_s + s.reduce_s;
  };
  auto row = [&](const char *name, double va, double vb, double vc) {
    std::printf("  %-12s  %8.4f  %8.4f  %8.4f   %6.2fx  %6.2fx\n", name, va, vb,
                vc, va > 0 ? vb / va : 0.0, va > 0 ? vc / va : 0.0);
  };
  std::printf("\n=== §30.81 step-3a: CUDA affine pair, %d frames, W=%d H=%d, "
              "%d column tiles (tile_w=%d) ===\n",
              nf, W, H, tiles, tile_w);
  std::printf("  %-12s  %8s  %8s  %8s   %6s  %6s\n", "phase", "A single",
              "B per-tile", "C tiled", "B/A", "C/A");
  row("produce", a.produce_s, b.produce_s, cc.produce_s);
  row("  .malloc", a.malloc_s, b.malloc_s, cc.malloc_s);
  row("  .upload", a.upload_s, b.upload_s, cc.upload_s);
  row("  .kernel", a.kernel_s, b.kernel_s, cc.kernel_s);
  row("  .download", a.download_s, b.download_s, cc.download_s);
  row("  dev sum", dev(a), dev(b), dev(cc));
  row("sort", a.sort_s, b.sort_s, cc.sort_s);
  row("reduce", a.reduce_s, b.reduce_s, cc.reduce_s);
  row("TOTAL", tot(a), tot(b), tot(cc));
  std::printf("  accumulate_pair_impl calls : A %llu  B %llu  C %llu\n", a.calls,
              b.calls, cc.calls);
  std::printf("  records sorted (sum)       : A %llu  B %llu (%.2fx)  "
              "C %llu (%.2fx)\n",
              a.records_sorted, b.records_sorted,
              a.records_sorted ? static_cast<double>(b.records_sorted) /
                                     static_cast<double>(a.records_sorted)
                               : 0.0,
              cc.records_sorted,
              a.records_sorted ? static_cast<double>(cc.records_sorted) /
                                     static_cast<double>(a.records_sorted)
                               : 0.0);
  std::printf("==============================================================="
              "\n");
}
