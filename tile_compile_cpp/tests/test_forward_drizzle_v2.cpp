#include "tile_compile/reconstruction/forward_drizzle_contrib_list.hpp"
#include "tile_compile/reconstruction/forward_drizzle_cuda.hpp"
#include "tile_compile/reconstruction/forward_drizzle_v2.hpp"
#include "tile_compile/registration/registration_sampling_plan.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace tile_compile;
using namespace tile_compile::reconstruction;
using tile_compile::registration::FrameSamplingTransform;
using tile_compile::registration::RegistrationSamplingPlan;

namespace {

WarpMatrix affine(double a, double b, double tx, double c, double d,
                  double ty) {
  WarpMatrix m;
  m(0, 0) = static_cast<float>(a);
  m(0, 1) = static_cast<float>(b);
  m(0, 2) = static_cast<float>(tx);
  m(1, 0) = static_cast<float>(c);
  m(1, 1) = static_cast<float>(d);
  m(1, 2) = static_cast<float>(ty);
  return m;
}

struct Fixture {
  RegistrationSamplingPlan plan;
  std::vector<Matrix2Df> images;
  config::ReconstructionDrizzleConfig cfg;

  SourceImageProvider provider() const {
    return [this](std::size_t i) -> const Matrix2Df & { return images.at(i); };
  }
};

Fixture make_fixture(ColorMode mode, BayerPattern pattern, int ox, int oy,
                     int scale) {
  Fixture f;
  f.plan.source_width = 7;
  f.plan.source_height = 6;
  f.plan.canvas_width_native = 14;
  f.plan.canvas_height_native = 13;
  f.plan.color_mode = mode;
  f.plan.bayer_pattern = pattern;
  f.plan.cfa_origin_x = ox;
  f.plan.cfa_origin_y = oy;
  const std::vector<WarpMatrix> transforms = {
      affine(1.0, 0.0, 2.0, 0.0, 1.0, 2.0),
      affine(0.997, -0.071, 3.13, 0.071, 0.997, 1.37),
      affine(1.04, 0.13, 1.61, -0.07, 0.96, 2.43),
      affine(0.84, 0.22, 2.41, -0.16, 1.11, 1.83),
      affine(1.13, -0.18, 1.29, 0.09, 0.88, 2.71)};
  for (std::size_t i = 0; i < transforms.size(); ++i) {
    FrameSamplingTransform frame;
    frame.frame_id = "v2-" + std::to_string(i);
    frame.source_index = i;
    frame.valid = true;
    frame.source_to_canvas = transforms[i];
    frame.source_to_canvas_affine_valid = true;
    f.plan.frames.push_back(frame);
    Matrix2Df image(f.plan.source_height, f.plan.source_width);
    for (int y = 0; y < image.rows(); ++y)
      for (int x = 0; x < image.cols(); ++x)
        image(y, x) = static_cast<float>(11.0 + 3.0 * i + 0.7 * x +
                                         1.3 * y + std::sin(x + 0.2 * y));
    image(1, static_cast<int>(i) + 1) =
        std::numeric_limits<float>::quiet_NaN();
    f.images.push_back(std::move(image));
  }
  f.cfg.internal_scale = scale;
  f.cfg.pixfrac = 0.8f;
  f.cfg.kernel = "square";
  return f;
}

void require_exact(const DrizzleUniformAccum &a,
                   const DrizzleUniformAccum &b) {
  REQUIRE(a.width == b.width);
  REQUIRE(a.rows == b.rows);
  REQUIRE(a.channels == b.channels);
  for (int c = 0; c < a.channels; ++c) {
    REQUIRE(a.wx[c].size() == b.wx[c].size());
    REQUIRE(std::memcmp(a.wx[c].data(), b.wx[c].data(),
                        a.wx[c].size() * sizeof(double)) == 0);
    REQUIRE(std::memcmp(a.w[c].data(), b.w[c].data(),
                        a.w[c].size() * sizeof(double)) == 0);
    REQUIRE(std::memcmp(a.w2[c].data(), b.w2[c].data(),
                        a.w2[c].size() * sizeof(double)) == 0);
  }
}

}  // namespace

TEST_CASE("forward drizzle v2 affine target gather is the exact record oracle",
          "[forward-drizzle-v2][oracle]") {
  for (int scale : {1, 2}) {
    for (BayerPattern pattern : {BayerPattern::RGGB, BayerPattern::BGGR,
                                 BayerPattern::GRBG, BayerPattern::GBRG}) {
      for (const auto origin : {std::pair{0, 0}, std::pair{1, -1}}) {
        for (ColorMode mode : {ColorMode::MONO, ColorMode::OSC}) {
          auto f = make_fixture(mode, pattern, origin.first, origin.second,
                                scale);
          const int full_h = f.plan.canvas_height_native * scale;
          for (const float pixfrac : {0.2f, 0.8f, 1.0f}) {
            f.cfg.pixfrac = pixfrac;
            for (const auto stripe : {std::pair{0, full_h},
                                      std::pair{1, full_h - 2}}) {
              const auto oracle = accumulate_uniform_by_frame(
                  f.plan, f.provider(), f.cfg, stripe.first, stripe.second);
              const auto got = gather_affine_uniform_v2(
                  f.plan, f.provider(), f.cfg, stripe.first, stripe.second);
              require_exact(oracle, got.accum);
              REQUIRE(got.stats.target_cells ==
                      static_cast<std::uint64_t>(oracle.width) * oracle.rows);
              REQUIRE(got.stats.source_candidates > 0);
              REQUIRE(got.stats.positive_overlaps > 0);
              REQUIRE(got.stats.workspace_bytes ==
                      static_cast<std::size_t>(oracle.channels) * oracle.width *
                          oracle.rows * 3 * sizeof(double));
            }
          }
        }
      }
    }
  }
}

TEST_CASE("forward drizzle v2 CUDA target gather matches the CPU gather",
          "[forward-drizzle-v2][cuda-parity]") {
  if (!forward_drizzle_cuda_runtime_available()) {
    SUCCEED("CUDA device unavailable");
    return;
  }
  for (int scale : {1, 2}) {
    for (ColorMode mode : {ColorMode::MONO, ColorMode::OSC}) {
      const auto f = make_fixture(mode, BayerPattern::GBRG, 1, -1, scale);
      const int y0 = 1;
      const int rows = f.plan.canvas_height_native * scale - 2;
      const auto cpu =
          gather_affine_uniform_v2(f.plan, f.provider(), f.cfg, y0, rows);
      ForwardDrizzleV2UniformResult gpu;
      REQUIRE(gather_affine_uniform_v2_cuda(f.plan, f.provider(), f.cfg, y0,
                                            rows, gpu));
      require_exact(cpu.accum, gpu.accum);
      REQUIRE(gpu.stats.source_candidates == cpu.stats.source_candidates);
      REQUIRE(gpu.stats.positive_overlaps == cpu.stats.positive_overlaps);
    }
  }
}

TEST_CASE("forward drizzle v2 dense scatter has the same affine support",
          "[forward-drizzle-v2][cuda-parity][scatter]") {
  if (!forward_drizzle_cuda_runtime_available()) {
    SUCCEED("CUDA device unavailable");
    return;
  }
  auto f = make_fixture(ColorMode::OSC, BayerPattern::GRBG, -1, 1, 2);
  f.plan.frames.resize(1);
  const int width = f.plan.canvas_width_native * f.cfg.internal_scale;
  const int rows = f.plan.canvas_height_native * f.cfg.internal_scale;
  const auto gather =
      gather_affine_uniform_v2(f.plan, f.provider(), f.cfg, 0, rows);
  const std::size_t n = static_cast<std::size_t>(width) * rows;
  std::vector<double> a(3 * n), b(3 * n);
  const auto &m = f.plan.frames.front().source_to_canvas;
  const double affine6[6] = {m(0, 0), m(0, 1), m(0, 2),
                             m(1, 0), m(1, 1), m(1, 2)};
  unsigned long long overlaps = 0;
  REQUIRE(forward_drizzle_cuda_affine_dense_scatter(
      affine6, f.cfg.internal_scale, 0.5 * f.cfg.pixfrac, 0, 0, width, rows,
      f.plan.source_width, f.plan.source_height, f.images.front().data(),
      static_cast<int>(f.plan.bayer_pattern), f.plan.cfa_origin_x,
      f.plan.cfa_origin_y, false, a.data(), b.data(), &overlaps));
  REQUIRE(overlaps == gather.stats.positive_overlaps);
  for (int c = 0; c < 3; ++c) {
    for (std::size_t i = 0; i < n; ++i) {
      REQUIRE(a[static_cast<std::size_t>(c) * n + i] ==
              Catch::Approx(gather.accum.wx[c][i]).margin(1e-12));
      REQUIRE(b[static_cast<std::size_t>(c) * n + i] ==
              Catch::Approx(gather.accum.w[c][i]).margin(1e-12));
    }
  }

  ForwardDrizzleV2CudaWorkspace workspace;
  REQUIRE_FALSE(workspace.reserve(std::numeric_limits<std::size_t>::max(),
                                  std::numeric_limits<std::size_t>::max(), 3));
  REQUIRE(workspace.stats().allocations == 0);
  REQUIRE(workspace.reserve(f.images.front().size(), n, 3));
  std::vector<double> wa(3 * n), wb(3 * n);
  REQUIRE(workspace.run_dense_scatter(
      affine6, f.cfg.internal_scale, 0.5 * f.cfg.pixfrac, 0, 0, width, rows,
      f.plan.source_width, f.plan.source_height, f.images.front().data(),
      static_cast<int>(f.plan.bayer_pattern), f.plan.cfa_origin_x,
      f.plan.cfa_origin_y, false, wa.data(), wb.data()));
  REQUIRE(workspace.run_dense_scatter(
      affine6, f.cfg.internal_scale, 0.5 * f.cfg.pixfrac, 0, 0, width, rows,
      f.plan.source_width, f.plan.source_height, f.images.front().data(),
      static_cast<int>(f.plan.bayer_pattern), f.plan.cfa_origin_x,
      f.plan.cfa_origin_y, false, wa.data(), wb.data()));
  REQUIRE(std::memcmp(a.data(), wa.data(), a.size() * sizeof(double)) == 0);
  REQUIRE(std::memcmp(b.data(), wb.data(), b.size() * sizeof(double)) == 0);
  REQUIRE(workspace.stats().allocations == 1);
  REQUIRE(workspace.stats().calls == 2);
  REQUIRE(workspace.stats().positive_overlaps == 2 * overlaps);
}

TEST_CASE("forward drizzle v2 rejects local warp before its architecture gate",
          "[forward-drizzle-v2][oracle]") {
  auto f = make_fixture(ColorMode::MONO, BayerPattern::RGGB, 0, 0, 1);
  f.plan.frames.front().has_smooth_local_model = true;
  f.plan.frames.front().smooth_local_model.valid = true;
  REQUIRE_THROWS_AS(gather_affine_uniform_v2(
                        f.plan, f.provider(), f.cfg, 0,
                        f.plan.canvas_height_native),
                    std::invalid_argument);
}

TEST_CASE("forward drizzle v2 native fold retains same-frame cross terms",
          "[forward-drizzle-v2][fold]") {
  const std::vector<double> area{0.25, 0.25, 0.25, 0.25};
  std::vector<ForwardDrizzleV2FrameSubpixel> v;
  for (std::size_t f = 0; f < 2; ++f)
    for (int j = 0; j < 4; ++j)
      v.push_back({f, (10.0 + f) * (1.0 + j), 1.0 + j});
  const auto got = fold_native_pixel_v2(v, 2, area);
  REQUIRE(got.a == 52.5);
  REQUIRE(got.b == 5.0);
  // Each frame folds to b_f=2.5.  Correct B2 is 2.5^2+2.5^2=12.5;
  // sum_j area_j^2*sum_f b_fj^2 would incorrectly produce 3.75.
  REQUIRE(got.b2 == 12.5);
  REQUIRE(got.n_eff == 2.0);
  REQUIRE(got.value == 10.5);
  REQUIRE(got.supported_area_fraction == 1.0);
  REQUIRE(got.source_support);
}

TEST_CASE("forward drizzle v2 partial fold preserves supported surface value",
          "[forward-drizzle-v2][fold]") {
  const std::vector<double> area{0.25, 0.25, 0.25, 0.25};
  const std::vector<ForwardDrizzleV2FrameSubpixel> v{
      {0, 7.0, 1.0}, {0, 0.0, 0.0}, {0, 21.0, 3.0}, {0, 0.0, 0.0}};
  const auto got = fold_native_pixel_v2(v, 1, area);
  REQUIRE(got.source_support);
  REQUIRE(got.value == 7.0);
  REQUIRE(got.supported_area_fraction == 0.5);
}

TEST_CASE("forward drizzle v2 robust reducer is bounded and never deletes support",
          "[forward-drizzle-v2][robust]") {
  std::vector<ForwardDrizzleV2RobustCandidate> c;
  for (std::size_t i = 0; i < 34; ++i)
    c.push_back({i, 100.0 + static_cast<double>(i % 3) - 1.0, 1.0});
  c[5].x = 100000.0;  // isolated cosmic ray
  int passes = 0;
  const auto replay = [&](const ForwardDrizzleV2CandidateSink &sink) {
    ++passes;
    for (const auto &v : c) sink(v);
  };
  const auto got = robust_reduce_v2(replay);
  REQUIRE(passes == 2);
  REQUIRE(got.state ==
          ForwardDrizzleV2RobustState::primary_winsorized_mom);
  REQUIRE(got.b == 34.0);
  REQUIRE(got.b2 == 34.0);
  REQUIRE(got.n_eff == 34.0);
  REQUIRE(got.value > 99.0);
  REQUIRE(got.value < 101.5);

  std::vector<ForwardDrizzleV2RobustCandidate> sparse{{0, 42.0, 0.7}};
  const auto fallback = robust_reduce_v2(
      [&](const ForwardDrizzleV2CandidateSink &sink) {
        for (const auto &v : sparse) sink(v);
      });
  REQUIRE(fallback.state ==
          ForwardDrizzleV2RobustState::too_few_candidates_fallback);
  REQUIRE(fallback.value == 42.0);
  REQUIRE(fallback.b == 0.7);

  std::vector<ForwardDrizzleV2RobustCandidate> flat;
  for (std::size_t i = 0; i < 34; ++i) flat.push_back({i, 12.0, 1.0});
  flat[7].x = 1.0e9;
  const auto degenerate = robust_reduce_v2(
      [&](const ForwardDrizzleV2CandidateSink &sink) {
        for (const auto &v : flat) sink(v);
      });
  REQUIRE(degenerate.state ==
          ForwardDrizzleV2RobustState::degenerate_scale);
  REQUIRE(degenerate.value == 12.0);
  REQUIRE(degenerate.b == 34.0);
}

TEST_CASE("forward drizzle v2 memory plan is checked and N-independent in X",
          "[forward-drizzle-v2][memory]") {
  ForwardDrizzleV2MemoryInputs in;
  in.target_width = 7680;
  in.target_height = 4320;
  in.channels = 3;
  in.robust_groups = 17;
  in.device_budget_bytes = static_cast<std::size_t>(4) << 30;
  in.host_budget_bytes = static_cast<std::size_t>(16) << 30;
  in.device_fixed_bytes = static_cast<std::size_t>(512) << 20;
  in.host_fixed_bytes = static_cast<std::size_t>(1) << 30;
  in.pinned_bytes_per_frame_target_row = 3840 * 5 * sizeof(float) / 2;
  in.frame_count = 40;
  const auto p40 = plan_forward_drizzle_v2_memory(in);
  REQUIRE(p40.feasible);
  REQUIRE(p40.tile_cols == 7680);
  REQUIRE_FALSE(p40.x_tiled);
  in.frame_count = 600;
  const auto p600 = plan_forward_drizzle_v2_memory(in);
  REQUIRE(p600.feasible);
  REQUIRE(p600.tile_cols == p40.tile_cols);
  // The device limit may bind both plans; increasing N must never increase
  // the resolved band height, but equality is valid in a device-bound case.
  REQUIRE(p600.band_rows <= p40.band_rows);
  REQUIRE(p600.device_peak_bytes <= in.device_budget_bytes);
  REQUIRE(p600.host_peak_bytes <= in.host_budget_bytes);

  in.device_budget_bytes = 1024;
  REQUIRE_FALSE(plan_forward_drizzle_v2_memory(in).feasible);
  in.device_budget_bytes = std::numeric_limits<std::size_t>::max();
  in.pinned_bytes_per_frame_target_row =
      std::numeric_limits<std::size_t>::max();
  REQUIRE_FALSE(plan_forward_drizzle_v2_memory(in).feasible);
}

TEST_CASE("forward drizzle v2 affine enumeration benchmark",
          "[.][forward-drizzle-v2-bench]") {
  if (!forward_drizzle_cuda_runtime_available()) {
    SUCCEED("CUDA device unavailable");
    return;
  }
  constexpr int sw = 1920, sh = 1080;
  Matrix2Df source(sh, sw);
  for (int y = 0; y < sh; ++y)
    for (int x = 0; x < sw; ++x)
      source(y, x) = 100.0f + 0.001f * x + 0.002f * y;
  const double angle = 0.025;
  const double ca = std::cos(angle), sa = std::sin(angle);
  const double affine6[6] = {ca, -sa, 15.0, sa, ca, -8.0};
  const double det = affine6[0] * affine6[4] - affine6[1] * affine6[3];
  const double inverse6[6] = {
      affine6[4] / det, -affine6[1] / det,
      -(affine6[4] * affine6[2] - affine6[1] * affine6[5]) / det,
      -affine6[3] / det, affine6[0] / det,
      -(-affine6[3] * affine6[2] + affine6[0] * affine6[5]) / det};
  using clock = std::chrono::steady_clock;
  for (int scale : {1, 2}) {
    const int tw = sw * scale, th = sh * scale;
    const std::size_t n = static_cast<std::size_t>(tw) * th * 3;
    std::vector<double> ga(n), gb(n), sa0(n), sb0(n), sa1(n), sb1(n);
    unsigned long long gc = 0, go = 0, so0 = 0, so1 = 0;
    const auto tg0 = clock::now();
    REQUIRE(forward_drizzle_cuda_affine_target_gather(
        affine6, inverse6, scale, 0.4, 0, 0, tw, th, sw, sh, source.data(),
        static_cast<int>(BayerPattern::RGGB), 0, 0, false, ga.data(), gb.data(),
        &gc, &go));
    const auto tg1 = clock::now();
    REQUIRE(forward_drizzle_cuda_affine_dense_scatter(
        affine6, scale, 0.4, 0, 0, tw, th, sw, sh, source.data(),
        static_cast<int>(BayerPattern::RGGB), 0, 0, false, sa0.data(),
        sb0.data(), &so0));
    const auto ts1 = clock::now();
    REQUIRE(forward_drizzle_cuda_affine_dense_scatter(
        affine6, scale, 0.4, 0, 0, tw, th, sw, sh, source.data(),
        static_cast<int>(BayerPattern::RGGB), 0, 0, false, sa1.data(),
        sb1.data(), &so1));
    const auto ts2 = clock::now();
    double max_a = 0.0, max_b = 0.0;
    std::uint64_t support_mismatch = 0;
    for (std::size_t i = 0; i < n; ++i) {
      max_a = std::max(max_a, std::abs(ga[i] - sa0[i]));
      max_b = std::max(max_b, std::abs(gb[i] - sb0[i]));
      support_mismatch += ((gb[i] > 0.0) != (sb0[i] > 0.0));
    }
    const bool repeat_equal =
        std::memcmp(sa0.data(), sa1.data(), n * sizeof(double)) == 0 &&
        std::memcmp(sb0.data(), sb1.data(), n * sizeof(double)) == 0;
    std::printf("[v2-enum] canvas=%dx%d scale=%d gather_s=%.6f "
                "scatter_s=%.6f scatter_repeat_s=%.6f candidates=%llu "
                "overlaps=%llu max_abs_A=%.17g max_abs_B=%.17g "
                "support_mismatch=%llu repeat_byte_equal=%d\n",
                tw, th, scale,
                std::chrono::duration<double>(tg1 - tg0).count(),
                std::chrono::duration<double>(ts1 - tg1).count(),
                std::chrono::duration<double>(ts2 - ts1).count(), gc, go,
                max_a, max_b,
                static_cast<unsigned long long>(support_mismatch),
                repeat_equal ? 1 : 0);
    REQUIRE(go == so0);
    REQUIRE(so0 == so1);
    REQUIRE(support_mismatch == 0);
  }
}
