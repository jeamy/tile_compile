#include "tile_compile/reconstruction/forward_drizzle_contrib_list.hpp"
#include "tile_compile/reconstruction/forward_drizzle_cuda.hpp"
#include "tile_compile/reconstruction/forward_drizzle_v2.hpp"
#include "tile_compile/registration/registration_sampling_plan.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
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
    for (int j = 0; j < 4; ++j) {
      const double b = 1.0 + j;
      v.push_back({f, b, b, b, (10.0 + f) * b, b});
    }
  const auto got = fold_native_pixel_v2(v, 2, area);
  REQUIRE(got.a == 52.5);
  REQUIRE(got.b == 5.0);
  // Each frame folds to b_f=2.5.  Correct B2 is 2.5^2+2.5^2=12.5;
  // sum_j area_j^2*sum_f b_fj^2 would incorrectly produce 3.75.
  REQUIRE(got.b2 == 12.5);
  REQUIRE(got.n_eff == 2.0);
  REQUIRE(got.value == 10.5);
  REQUIRE(got.geometry_area_fraction == 1.0);
  REQUIRE(got.source_area_fraction == 1.0);
  REQUIRE(got.estimator_area_fraction == 1.0);
  REQUIRE(got.profile_area_fraction == 1.0);
  REQUIRE(got.geometry_support);
  REQUIRE(got.source_support);
  REQUIRE(got.estimator_support);
  REQUIRE(got.profile_support);
}

TEST_CASE("forward drizzle v2 partial fold preserves supported surface value",
          "[forward-drizzle-v2][fold]") {
  const std::vector<double> area{0.25, 0.25, 0.25, 0.25};
  const std::vector<ForwardDrizzleV2FrameSubpixel> v{
      {0, 1.0, 1.0, 1.0, 7.0, 1.0},
      {0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0, 3.0, 3.0, 3.0, 21.0, 3.0},
      {0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  const auto got = fold_native_pixel_v2(v, 1, area);
  REQUIRE(got.geometry_support);
  REQUIRE(got.source_support);
  REQUIRE(got.estimator_support);
  REQUIRE(got.profile_support);
  REQUIRE(got.value == 7.0);
  REQUIRE(got.geometry_area_fraction == 0.5);
  REQUIRE(got.source_area_fraction == 0.5);
  REQUIRE(got.estimator_area_fraction == 0.5);
  REQUIRE(got.profile_area_fraction == 0.5);
}

TEST_CASE("forward drizzle v2 fold keeps four support layers distinct",
          "[forward-drizzle-v2][fold][support]") {
  const std::vector<double> area{0.25, 0.25, 0.25, 0.25};
  const std::vector<ForwardDrizzleV2FrameSubpixel> v{
      {0, 1.0, 0.0, 0.0, 0.0, 0.0},
      {0, 1.0, 1.0, 0.0, 0.0, 0.0},
      {0, 1.0, 1.0, 1.0, 0.0, 0.0},
      {0, 1.0, 1.0, 1.0, 5.0, 1.0}};
  const auto got = fold_native_pixel_v2(v, 1, area);
  REQUIRE(got.geometry_support);
  REQUIRE(got.source_support);
  REQUIRE(got.estimator_support);
  REQUIRE(got.profile_support);
  REQUIRE(got.geometry_area_fraction == 1.0);
  REQUIRE(got.source_area_fraction == 0.75);
  REQUIRE(got.estimator_area_fraction == 0.5);
  REQUIRE(got.profile_area_fraction == 0.25);
  REQUIRE(got.value == 5.0);
  REQUIRE(got.n_eff == 1.0);

  const std::vector<ForwardDrizzleV2FrameSubpixel> no_profile{
      {0, 1.0, 1.0, 1.0, 0.0, 0.0},
      {0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  const auto absent = fold_native_pixel_v2(no_profile, 1, area);
  REQUIRE(absent.geometry_support);
  REQUIRE(absent.source_support);
  REQUIRE(absent.estimator_support);
  REQUIRE_FALSE(absent.profile_support);
  REQUIRE(absent.profile_area_fraction == 0.0);

  const std::vector<ForwardDrizzleV2FrameSubpixel> point{
      {0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0, 2.0, 2.0, 2.0, 14.0, 2.0},
      {0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  const auto point_result = fold_native_pixel_v2(point, 1, area);
  REQUIRE(point_result.geometry_area_fraction == 0.25);
  REQUIRE(point_result.profile_area_fraction == 0.25);
  REQUIRE(point_result.a == 3.5);
  REQUIRE(point_result.value == 7.0);
  REQUIRE(point_result.b2 == Catch::Approx(0.25).margin(1e-15));
  REQUIRE(point_result.n_eff == Catch::Approx(1.0).margin(1e-15));

  const std::vector<ForwardDrizzleV2FrameSubpixel> empty(4);
  const auto none = fold_native_pixel_v2(empty, 1, area);
  REQUIRE_FALSE(none.geometry_support);
  REQUIRE_FALSE(none.source_support);
  REQUIRE_FALSE(none.estimator_support);
  REQUIRE_FALSE(none.profile_support);
  REQUIRE(none.b == 0.0);
  REQUIRE(none.value == 0.0);
}

TEST_CASE("forward drizzle v2 fold rejects invalid support implications",
          "[forward-drizzle-v2][fold][support]") {
  const std::vector<double> area{1.0};
  REQUIRE_THROWS_AS(fold_native_pixel_v2(
                        std::vector<ForwardDrizzleV2FrameSubpixel>{
                            {0, 0.0, 1.0, 0.0, 0.0, 0.0}},
                        1, area),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(fold_native_pixel_v2(
                        std::vector<ForwardDrizzleV2FrameSubpixel>{
                            {0, 1.0, 1.0, 0.0, 2.0, 1.0}},
                        1, area),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(fold_native_pixel_v2(
                        std::vector<ForwardDrizzleV2FrameSubpixel>{
                            {0, 1.0, 1.0, 1.0,
                             std::numeric_limits<double>::quiet_NaN(), 1.0}},
                        1, area),
                    std::invalid_argument);
}

TEST_CASE("forward drizzle v2 fold preserves weighted gradient algebra",
          "[forward-drizzle-v2][fold][flux]") {
  const std::vector<double> area{0.1, 0.2, 0.3, 0.4};
  const double values[4] = {1.0, 2.0, 3.0, 4.0};
  const double weights[2][4] = {{1.0, 2.0, 1.0, 2.0},
                                {2.0, 1.0, 2.0, 1.0}};
  std::vector<ForwardDrizzleV2FrameSubpixel> v;
  for (std::size_t f = 0; f < 2; ++f)
    for (std::size_t j = 0; j < 4; ++j) {
      const double b = weights[f][j];
      v.push_back({f, b, b, b, b * values[j], b});
    }
  const auto got = fold_native_pixel_v2(v, 2, area);
  REQUIRE(got.a == Catch::Approx(9.0).margin(1e-15));
  REQUIRE(got.b == Catch::Approx(3.0).margin(1e-15));
  REQUIRE(got.b2 == Catch::Approx(4.52).margin(1e-15));
  REQUIRE(got.value == Catch::Approx(3.0).margin(1e-15));
  REQUIRE(got.n_eff == Catch::Approx(9.0 / 4.52).margin(1e-15));
  REQUIRE(got.geometry_area_fraction == 1.0);
  REQUIRE(got.source_area_fraction == 1.0);
  REQUIRE(got.estimator_area_fraction == 1.0);
  REQUIRE(got.profile_area_fraction == 1.0);
}

TEST_CASE("forward drizzle v2 CUDA fold matches all support layers",
          "[forward-drizzle-v2][fold][cuda-parity]") {
  if (!forward_drizzle_cuda_runtime_available()) {
    SUCCEED("CUDA device unavailable");
    return;
  }
  const std::vector<double> area{0.1, 0.2, 0.3, 0.4};
  const std::vector<ForwardDrizzleV2FrameSubpixel> v{
      {0, 1.0, 1.0, 1.0, 2.0, 1.0},
      {0, 2.0, 2.0, 0.0, 0.0, 0.0},
      {0, 3.0, 0.0, 0.0, 0.0, 0.0},
      {0, 4.0, 4.0, 4.0, 20.0, 4.0},
      {1, 1.5, 1.5, 1.5, 4.5, 1.5},
      {1, 0.0, 0.0, 0.0, 0.0, 0.0},
      {1, 2.5, 2.5, 2.5, 10.0, 2.5},
      {1, 3.5, 3.5, 0.0, 0.0, 0.0}};
  const auto cpu = fold_native_pixel_v2(v, 2, area);
  ForwardDrizzleV2FoldResult gpu;
  REQUIRE(fold_native_pixel_v2_cuda(v, 2, area, gpu));
  REQUIRE(gpu.a == Catch::Approx(cpu.a).margin(1e-12));
  REQUIRE(gpu.b == Catch::Approx(cpu.b).margin(1e-12));
  REQUIRE(gpu.b2 == Catch::Approx(cpu.b2).margin(1e-12));
  REQUIRE(gpu.value == Catch::Approx(cpu.value).margin(1e-12));
  REQUIRE(gpu.n_eff == Catch::Approx(cpu.n_eff).margin(1e-12));
  REQUIRE(gpu.geometry_area_fraction ==
          Catch::Approx(cpu.geometry_area_fraction).margin(1e-12));
  REQUIRE(gpu.source_area_fraction ==
          Catch::Approx(cpu.source_area_fraction).margin(1e-12));
  REQUIRE(gpu.estimator_area_fraction ==
          Catch::Approx(cpu.estimator_area_fraction).margin(1e-12));
  REQUIRE(gpu.profile_area_fraction ==
          Catch::Approx(cpu.profile_area_fraction).margin(1e-12));
  REQUIRE(gpu.geometry_support == cpu.geometry_support);
  REQUIRE(gpu.source_support == cpu.source_support);
  REQUIRE(gpu.estimator_support == cpu.estimator_support);
  REQUIRE(gpu.profile_support == cpu.profile_support);
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

TEST_CASE("forward drizzle v2 Gate-1 production affine matrix",
          "[.][forward-drizzle-v2-gate1-production]") {
  if (!forward_drizzle_cuda_runtime_available()) {
    SUCCEED("CUDA device unavailable");
    return;
  }

  constexpr char kSpecSha[] =
      "9be00e5a367b2ec4e3d15dc23a49270fa1c2cd0e2455fc2f03a5506809b5b149";
  constexpr int sw = 3840, sh = 2160, tw = 7868, th = 4540;
  constexpr double kMaxAbs = 1e-10;
  constexpr double kMaxRel = 1e-12;
  constexpr double kMaxSteadySeconds = 0.75;
  constexpr double kConstantFluxRel = 1e-10;
  constexpr double kPointFluxRel = 1e-9;
  constexpr std::array<double, 3> pixfracs{0.2, 0.8, 1.0};

  struct TransformCase {
    const char *id;
    std::array<double, 6> m;
  };
  const std::array<TransformCase, 7> transforms{{
      {"m42_reference_frame_29", {1.0, 0.0, 32.0, 0.0, 1.0, 50.0}},
      {"m42_positive_rotation_frame_0",
       {0.9998640418052673, -0.016167480498552322, 35.636409759521484,
        0.01610037311911583, 0.9999631643295288, 46.23441696166992}},
      {"m42_negative_rotation_frame_59",
       {0.9998466372489929, 0.017386717721819878, 55.92152404785156,
        -0.01743965968489647, 0.9998936057090759, 68.48160552978516}},
      {"m42_max_condition_frame_25",
       {0.9999858140945435, -0.0024168870877474546, 31.47933578491211,
        0.0021460296120494604, 1.000150442123413, 56.29893493652344}},
      {"m42_scale_extreme_frame_17",
       {1.0000033378601074, -0.006749980617314577, 28.21784782409668,
        0.006637410260736942, 1.0002058744430542, 58.95621871948242}},
      {"stress_shear_scale_a",
       {0.84, 0.22, 420.0, -0.16, 1.11, 380.0}},
      {"stress_shear_scale_b",
       {1.13, -0.18, 510.0, 0.09, 0.88, 430.0}},
  }};

  auto inverse = [](const std::array<double, 6> &m) {
    const double det = m[0] * m[4] - m[1] * m[3];
    REQUIRE(std::isfinite(det));
    REQUIRE(std::abs(det) > 1e-12);
    return std::array<double, 6>{
        m[4] / det, -m[1] / det,
        -(m[4] * m[2] - m[1] * m[5]) / det,
        -m[3] / det, m[0] / det,
        -(-m[3] * m[2] + m[0] * m[5]) / det};
  };
  auto rel_error = [](double a, double b) {
    return std::abs(a - b) / std::max(1.0, std::abs(a));
  };

  Matrix2Df source(sh, sw);
  for (int y = 0; y < sh; ++y)
    for (int x = 0; x < sw; ++x)
      source(y, x) = static_cast<float>(
          100.0 + 0.001 * x + 0.002 * y +
          0.01 * ((17 * x + 31 * y) % 29));

  const std::size_t plane_n = static_cast<std::size_t>(tw) * th;
  const std::size_t n = 3 * plane_n;
  using clock = std::chrono::steady_clock;
  for (const auto &tc : transforms) {
    const auto inv = inverse(tc.m);
    for (double pixfrac : pixfracs) {
      std::vector<double> ga(n), gb(n), sa0(n), sb0(n), sa1(n), sb1(n);
      unsigned long long candidates = 0, gather_overlaps = 0;
      const auto gather_begin = clock::now();
      REQUIRE(forward_drizzle_cuda_affine_target_gather(
          tc.m.data(), inv.data(), 2, 0.5 * pixfrac, 0, 0, tw, th, sw, sh,
          source.data(), static_cast<int>(BayerPattern::GBRG), 0, 0, false,
          ga.data(), gb.data(), &candidates, &gather_overlaps));
      const double gather_seconds =
          std::chrono::duration<double>(clock::now() - gather_begin).count();

      ForwardDrizzleV2CudaWorkspace workspace;
      REQUIRE(workspace.reserve(source.size(), plane_n, 3));
      REQUIRE(workspace.run_dense_scatter(
          tc.m.data(), 2, 0.5 * pixfrac, 0, 0, tw, th, sw, sh,
          source.data(), static_cast<int>(BayerPattern::GBRG), 0, 0, false,
          sa0.data(), sb0.data()));
      const auto first_stats = workspace.stats();
      REQUIRE(workspace.run_dense_scatter(
          tc.m.data(), 2, 0.5 * pixfrac, 0, 0, tw, th, sw, sh,
          source.data(), static_cast<int>(BayerPattern::GBRG), 0, 0, false,
          sa1.data(), sb1.data()));
      const auto final_stats = workspace.stats();

      double max_abs = 0.0, max_rel = 0.0;
      double repeat_abs = 0.0, repeat_rel = 0.0;
      std::uint64_t support_mismatches = 0;
      for (std::size_t i = 0; i < n; ++i) {
        max_abs = std::max({max_abs, std::abs(ga[i] - sa0[i]),
                            std::abs(gb[i] - sb0[i])});
        max_rel = std::max({max_rel, rel_error(ga[i], sa0[i]),
                            rel_error(gb[i], sb0[i])});
        repeat_abs = std::max({repeat_abs, std::abs(sa0[i] - sa1[i]),
                               std::abs(sb0[i] - sb1[i])});
        repeat_rel = std::max({repeat_rel, rel_error(sa0[i], sa1[i]),
                               rel_error(sb0[i], sb1[i])});
        support_mismatches +=
            static_cast<std::uint64_t>((gb[i] > 0.0) != (sb0[i] > 0.0));
      }
      const double first_upload = first_stats.upload_seconds;
      const double first_kernel = first_stats.kernel_seconds;
      const double first_download = first_stats.download_seconds;
      const double second_upload =
          final_stats.upload_seconds - first_stats.upload_seconds;
      const double second_kernel =
          final_stats.kernel_seconds - first_stats.kernel_seconds;
      const double second_download =
          final_stats.download_seconds - first_stats.download_seconds;
      // Gate-1 steady state ends with frame-local A/B resident on device. The
      // full A/B D2H below exists only so this correctness harness can compare
      // every cell; v2's selected pipeline reduces those planes on device.
      const double first_steady = first_upload + first_kernel;
      const double second_steady = second_upload + second_kernel;

      std::printf(
          "{\"gate\":1,\"spec_sha256\":\"%s\",\"case\":\"%s\","
          "\"pixfrac\":%.1f,\"gather_correctness_seconds\":%.9f,"
          "\"scatter_first_upload_seconds\":%.9f,"
          "\"scatter_first_kernel_seconds\":%.9f,"
          "\"scatter_first_download_seconds\":%.9f,"
          "\"scatter_first_steady_seconds\":%.9f,"
          "\"scatter_second_upload_seconds\":%.9f,"
          "\"scatter_second_kernel_seconds\":%.9f,"
          "\"scatter_second_download_seconds\":%.9f,"
          "\"scatter_second_steady_seconds\":%.9f,\"candidates\":%llu,"
          "\"overlaps\":%llu,\"max_abs_error\":%.17g,"
          "\"max_relative_error\":%.17g,\"repeat_max_abs_error\":%.17g,"
          "\"repeat_max_relative_error\":%.17g,"
          "\"support_mismatches\":%llu,\"global_syncs\":%llu,"
          "\"stream_syncs\":%llu}\n",
          kSpecSha, tc.id, pixfrac, gather_seconds, first_upload, first_kernel,
          first_download, first_steady, second_upload, second_kernel,
          second_download, second_steady, candidates, gather_overlaps, max_abs,
          max_rel, repeat_abs, repeat_rel,
          static_cast<unsigned long long>(support_mismatches),
          static_cast<unsigned long long>(
              final_stats.device_global_synchronizations),
          static_cast<unsigned long long>(final_stats.stream_synchronizations));

      CHECK(final_stats.positive_overlaps == 2 * gather_overlaps);
      CHECK(support_mismatches == 0);
      CHECK(max_abs <= kMaxAbs);
      CHECK(max_rel <= kMaxRel);
      CHECK(repeat_abs <= kMaxAbs);
      CHECK(repeat_rel <= kMaxRel);
      CHECK(first_steady <= kMaxSteadySeconds);
      CHECK(second_steady <= kMaxSteadySeconds);
      CHECK(final_stats.allocations == 1);
      CHECK(final_stats.device_global_synchronizations == 0);
    }
  }

  const std::array<double, 6> identity{1.0, 0.0, 32.0, 0.0, 1.0, 50.0};
  std::vector<double> a(n), b(n);
  ForwardDrizzleV2CudaWorkspace flux_workspace;
  REQUIRE(flux_workspace.reserve(source.size(), plane_n, 3));

  source.setConstant(7.0f);
  REQUIRE(flux_workspace.run_dense_scatter(
      identity.data(), 2, 0.4, 0, 0, tw, th, sw, sh, source.data(),
      static_cast<int>(BayerPattern::GBRG), 0, 0, false, a.data(), b.data()));
  double constant_max_rel = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    if (b[i] > 0.0)
      constant_max_rel =
          std::max(constant_max_rel, std::abs(a[i] / b[i] - 7.0) / 7.0);
  CHECK(constant_max_rel <= kConstantFluxRel);

  source.setConstant(std::numeric_limits<float>::quiet_NaN());
  const int point_x = sw / 2, point_y = sh / 2;
  source(point_y, point_x) = 100.0f;
  REQUIRE(flux_workspace.run_dense_scatter(
      identity.data(), 2, 0.4, 0, 0, tw, th, sw, sh, source.data(),
      static_cast<int>(BayerPattern::GBRG), 0, 0, false, a.data(), b.data()));
  double point_a = 0.0, point_b = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    point_a += a[i];
    point_b += b[i];
  }
  const double expected_b = 0.8 * 0.8 * 2.0 * 2.0;
  const double expected_a = 100.0 * expected_b;
  CHECK(std::abs(point_b - expected_b) / expected_b <= kPointFluxRel);
  CHECK(std::abs(point_a - expected_a) / expected_a <= kPointFluxRel);
  std::printf(
      "{\"gate\":1,\"spec_sha256\":\"%s\",\"flux\":true,"
      "\"constant_max_relative_error\":%.17g,"
      "\"point_a_relative_error\":%.17g,"
      "\"point_b_relative_error\":%.17g}\n",
      kSpecSha, constant_max_rel,
      std::abs(point_a - expected_a) / expected_a,
      std::abs(point_b - expected_b) / expected_b);
}

namespace {

constexpr const char *kGate3SpecSha =
    "b7d7e1311695fa10aec4dae43e48995a1b9699672a8eb95a9925215b01241c53";

// Deterministic splitmix-style noise in [0,1); stable across platforms because
// it only uses integer arithmetic.
double g3_unit_noise(std::uint64_t i) {
  std::uint64_t h = i * 6364136223846793005ULL + 1442695040888963407ULL;
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53ULL;
  h ^= h >> 33;
  return static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0);
}

// Roughly unit-variance noise built from three uniforms.
double g3_noise(std::uint64_t f) {
  return 2.0 * (g3_unit_noise(f * 3) + g3_unit_noise(f * 3 + 1) +
                g3_unit_noise(f * 3 + 2) - 1.5);
}

struct G3Case {
  const char *name;
  double truth = 0.0;
  std::vector<ForwardDrizzleV2RobustCandidate> c;
};

std::vector<G3Case> g3_cases() {
  std::vector<G3Case> cases;
  const auto clean = [](int n) {
    G3Case k;
    k.name = "clean_gauss";
    k.truth = 100.0;
    for (std::size_t f = 0; f < static_cast<std::size_t>(n); ++f)
      k.c.push_back({f, 100.0 + g3_noise(f), 1.0});
    return k;
  };

  cases.push_back(clean(40));

  {
    G3Case k = clean(40);
    k.name = "cosmic_ray_single";
    k.c[20].x += 200.0;
    cases.push_back(k);
  }
  {
    G3Case k = clean(40);
    k.name = "cosmic_ray_single_edge_frame";
    k.c[0].x += 200.0;
    cases.push_back(k);
  }
  {
    G3Case k = clean(40);
    k.name = "extreme_outlier_dominated_weight";
    k.c[7].x += 200.0;
    k.c[7].b = 10.0;
    cases.push_back(k);
  }
  {
    G3Case k;
    k.name = "all_identical";
    k.truth = 100.0;
    for (std::size_t f = 0; f < 40; ++f) k.c.push_back({f, 100.0, 1.0});
    cases.push_back(k);
  }
  {
    G3Case k;
    k.name = "sparse_5_rb";
    k.truth = 100.0;
    const double xs[] = {99.0, 100.0, 101.0, 99.5, 100.5};
    for (std::size_t f = 0; f < 5; ++f) k.c.push_back({f, xs[f], 1.0});
    cases.push_back(k);
  }
  {
    G3Case k = clean(40);
    k.name = "satellite_trail";
    for (std::size_t f = 15; f <= 19; ++f) k.c[f].x += 30.0;
    cases.push_back(k);
  }
  {
    G3Case k = clean(40);
    k.name = "grouped_contamination_30pct";
    for (std::size_t f = 0; f < k.c.size(); ++f)
      if (f % 10 < 3) k.c[f].x += 8.0;
    cases.push_back(k);
  }
  {
    G3Case k = clean(40);
    k.name = "symmetric_heavy_tail";
    int sign = 1;
    for (std::size_t f = 0; f < k.c.size(); ++f)
      if (f % 5 == 0) {
        k.c[f].x += 50.0 * sign;
        sign = -sign;
      }
    cases.push_back(k);
  }
  {
    G3Case k = clean(40);
    k.name = "same_group_outliers";
    k.c[8].x += 50.0;
    k.c[25].x += 50.0;  // 8 and 25 share group 8 for K=17
    cases.push_back(k);
  }
  {
    G3Case k;
    k.name = "alternating_amplitude";
    k.truth = 100.0;
    for (std::size_t f = 0; f < 40; ++f)
      k.c.push_back(
          {f, 100.0 + (f % 2 == 0 ? g3_noise(f) : 5.0 * g3_noise(f)), 1.0});
    cases.push_back(k);
  }
  {
    G3Case k;
    k.name = "monotone_drift";
    k.truth = 109.75;
    for (std::size_t f = 0; f < 40; ++f) k.c.push_back({f, 100.0 + 0.5 * f, 1.0});
    cases.push_back(k);
  }
  {
    G3Case k;
    k.name = "bimodal";
    k.truth = 100.0;
    for (std::size_t f = 0; f < 40; ++f)
      k.c.push_back({f, f < 20 ? 98.0 : 102.0, 1.0});
    cases.push_back(k);
  }
  {
    G3Case k;
    k.name = "mad_zero_one_outlier";
    k.truth = 100.0;
    for (std::size_t f = 0; f < 40; ++f) k.c.push_back({f, 100.0, 1.0});
    k.c[13].x = 150.0;
    cases.push_back(k);
  }
  {
    G3Case k;
    k.name = "sparse_3";
    k.truth = 42.0;
    k.c = {{0, 41.0, 1.0}, {1, 42.0, 1.0}, {2, 43.0, 1.0}};
    cases.push_back(k);
  }
  {
    G3Case k;
    k.name = "single";
    k.truth = 77.0;
    k.c = {{0, 77.0, 1.0}};
    cases.push_back(k);
  }
  {
    G3Case k = clean(40);
    k.name = "unequal_weights";
    for (std::size_t f = 0; f < k.c.size(); ++f)
      k.c[f].b = 0.05 + 9.95 * g3_unit_noise(0x9000 + f);
    cases.push_back(k);
  }
  {
    G3Case k = clean(40);
    k.name = "nonfinite";
    k.c[10].x = std::numeric_limits<double>::quiet_NaN();
    k.c[20].b = 0.0;
    k.c[21].b = -1.0;
    k.c[22].x = std::numeric_limits<double>::infinity();
    cases.push_back(k);
  }
  {
    G3Case k;
    k.name = "empty";
    k.truth = 0.0;
    cases.push_back(k);
  }
  {
    G3Case k = clean(600);
    k.name = "clean_gauss_N600";
    cases.push_back(k);
  }
  {
    G3Case k = clean(600);
    k.name = "grouped_contamination_30pct_N600";
    for (std::size_t f = 0; f < k.c.size(); ++f)
      if (f % 10 < 3) k.c[f].x += 8.0;
    cases.push_back(k);
  }
  return cases;
}

const char *g3_estimator_name(ForwardDrizzleV2Estimator e) {
  switch (e) {
    case ForwardDrizzleV2Estimator::uniform:
      return "uniform";
    case ForwardDrizzleV2Estimator::mom_median:
      return "mom_weighted_median";
    case ForwardDrizzleV2Estimator::mom_winsorized_groups:
      return "mom_winsorized_groups";
    case ForwardDrizzleV2Estimator::mom_trimmed_groups:
      return "mom_trimmed_groups";
    case ForwardDrizzleV2Estimator::reservoir_sigma_clip:
      return "reservoir_sigma_clip";
    case ForwardDrizzleV2Estimator::two_pass_winsorized_frames:
      return "two_pass_winsorized_frames";
  }
  return "unknown";
}

const char *g3_state_name(ForwardDrizzleV2RobustState s) {
  switch (s) {
    case ForwardDrizzleV2RobustState::no_source_support:
      return "no_source_support";
    case ForwardDrizzleV2RobustState::too_few_candidates_fallback:
      return "too_few_candidates_fallback";
    case ForwardDrizzleV2RobustState::too_few_groups_fallback:
      return "too_few_groups_fallback";
    case ForwardDrizzleV2RobustState::degenerate_scale:
      return "degenerate_scale";
    case ForwardDrizzleV2RobustState::primary_winsorized_mom:
      return "primary_winsorized_mom";
    case ForwardDrizzleV2RobustState::primary_mom_median:
      return "primary_mom_median";
    case ForwardDrizzleV2RobustState::primary_mom_winsorized_groups:
      return "primary_mom_winsorized_groups";
    case ForwardDrizzleV2RobustState::primary_mom_trimmed_groups:
      return "primary_mom_trimmed_groups";
    case ForwardDrizzleV2RobustState::primary_uniform:
      return "primary_uniform";
    case ForwardDrizzleV2RobustState::primary_reservoir_sigma_clip:
      return "primary_reservoir_sigma_clip";
    case ForwardDrizzleV2RobustState::oracle_sigma_clip:
      return "oracle_sigma_clip";
  }
  return "unknown";
}

}  // namespace

TEST_CASE("forward drizzle v2 gate-3 fallback state machine preserves support",
          "[forward-drizzle-v2][robust][gate3]") {
  ForwardDrizzleV2RobustConfig cfg;
  std::vector<ForwardDrizzleV2RobustCandidate> empty;
  const auto none = robust_reduce_candidates_v2(
      empty, ForwardDrizzleV2Estimator::mom_winsorized_groups, cfg);
  REQUIRE(none.state == ForwardDrizzleV2RobustState::no_source_support);
  REQUIRE(none.b == 0.0);

  std::vector<ForwardDrizzleV2RobustCandidate> sparse{
      {0, 41.0, 1.0}, {1, 42.0, 1.0}, {2, 43.0, 1.0}};
  const auto few = robust_reduce_candidates_v2(
      sparse, ForwardDrizzleV2Estimator::mom_winsorized_groups, cfg);
  REQUIRE(few.state ==
          ForwardDrizzleV2RobustState::too_few_candidates_fallback);
  REQUIRE(few.value == 42.0);
  REQUIRE(few.b == 3.0);

  std::vector<ForwardDrizzleV2RobustCandidate> few_groups{
      {0, 41.0, 1.0}, {1, 42.0, 1.0}, {17, 43.0, 1.0},
      {18, 44.0, 1.0}, {34, 45.0, 1.0}};
  const auto fg = robust_reduce_candidates_v2(
      few_groups, ForwardDrizzleV2Estimator::mom_winsorized_groups, cfg);
  REQUIRE(fg.state == ForwardDrizzleV2RobustState::too_few_groups_fallback);
  REQUIRE(fg.value == Catch::Approx(43.0));
  REQUIRE(fg.b == 5.0);

  std::vector<ForwardDrizzleV2RobustCandidate> flat;
  for (std::size_t f = 0; f < 40; ++f) flat.push_back({f, 100.0, 1.0});
  flat[13].x = 150.0;
  const auto deg = robust_reduce_candidates_v2(
      flat, ForwardDrizzleV2Estimator::mom_winsorized_groups, cfg);
  REQUIRE(deg.state == ForwardDrizzleV2RobustState::degenerate_scale);
  REQUIRE(deg.value == 100.0);
  REQUIRE(deg.b == 40.0);
  const auto deg_median = robust_reduce_candidates_v2(
      flat, ForwardDrizzleV2Estimator::mom_median, cfg);
  REQUIRE(deg_median.state == ForwardDrizzleV2RobustState::primary_mom_median);
  REQUIRE(deg_median.value == 100.0);

  std::vector<ForwardDrizzleV2RobustCandidate> mixed;
  for (std::size_t f = 0; f < 40; ++f) mixed.push_back({f, 100.0 + f, 1.0});
  mixed[10].x = std::numeric_limits<double>::quiet_NaN();
  mixed[20].b = -2.0;
  for (auto est : {ForwardDrizzleV2Estimator::uniform,
                   ForwardDrizzleV2Estimator::mom_median,
                   ForwardDrizzleV2Estimator::mom_winsorized_groups,
                   ForwardDrizzleV2Estimator::mom_trimmed_groups}) {
    const auto r = robust_reduce_candidates_v2(mixed, est, cfg);
    REQUIRE(r.candidates == 38);
    REQUIRE(std::isfinite(r.value));
    REQUIRE(r.b == 38.0);
    REQUIRE(r.n_eff == Catch::Approx(38.0));
  }
  REQUIRE_THROWS_AS(
      robust_reduce_candidates_v2(
          mixed, ForwardDrizzleV2Estimator::two_pass_winsorized_frames, cfg),
      std::invalid_argument);
}

TEST_CASE("forward drizzle v2 fold and scatter reject overflowing/nonfinite inputs",
          "[forward-drizzle-v2][robust][gate3]") {
  // Review fix: frame_count * area.size() must not overflow before the shape
  // check, and the device scatter must reject non-finite affine/pixfrac
  // parameters (floor(NaN) to int is undefined behaviour on device).
  std::vector<ForwardDrizzleV2FrameSubpixel> e(1);
  std::vector<double> area{1.0};
  REQUIRE_THROWS_AS(
      fold_native_pixel_v2(e, std::numeric_limits<std::size_t>::max(), area),
      std::invalid_argument);

  if (forward_drizzle_cuda_runtime_available()) {
    double bad[6] = {1, 0, 0, 0, 1, std::numeric_limits<double>::quiet_NaN()};
    ForwardDrizzleV2CudaWorkspace ws;
    REQUIRE(ws.reserve(64, 64, 1));
    std::vector<float> src(64, 1.0f);
    std::vector<double> a(64, 0.0), b(64, 0.0);
    REQUIRE_FALSE(ws.run_dense_scatter(bad, 1, 0.5, 0, 0, 8, 8, 8, 8,
                                       src.data(), 0, 0, 0, true, a.data(),
                                       b.data()));
    double good[6] = {1, 0, 0, 0, 1, 0};
    REQUIRE_FALSE(ws.run_dense_scatter(good, 1,
                                       std::numeric_limits<double>::quiet_NaN(),
                                       0, 0, 8, 8, 8, 8, src.data(), 0, 0, 0,
                                       true, a.data(), b.data()));
    REQUIRE(ws.run_dense_scatter(good, 1, 0.5, 0, 0, 8, 8, 8, 8, src.data(), 0,
                                 0, 0, true, a.data(), b.data()));
  }
}

TEST_CASE("forward drizzle v2 Gate-3 robust estimator adversarial matrix",
          "[.][forward-drizzle-v2-gate3]") {
  const auto cases = g3_cases();
  const ForwardDrizzleV2Estimator grouped[] = {
      ForwardDrizzleV2Estimator::uniform,
      ForwardDrizzleV2Estimator::mom_median,
      ForwardDrizzleV2Estimator::mom_winsorized_groups,
      ForwardDrizzleV2Estimator::mom_trimmed_groups};
  const int ks[] = {17, 41};
  // Cases excluded from the absolute vs-oracle bound per the frozen spec.
  const auto oracle_gated = [](const char *name) {
    return std::string(name) != "bimodal" &&
           std::string(name) != "clean_gauss_N600";
  };

  // tag -> case -> result
  std::map<std::string, std::map<std::string, ForwardDrizzleV2RobustResult>>
      results;
  std::map<std::string, std::map<std::string, double>> dev_oracle, dev_truth;

  const auto run = [&](const G3Case &k, const std::string &tag,
                       ForwardDrizzleV2Estimator est,
                       const ForwardDrizzleV2RobustConfig &cfg) {
    const auto r = robust_reduce_candidates_v2(k.c, est, cfg);
    const auto r2 = robust_reduce_candidates_v2(k.c, est, cfg);
    // Determinism: identical inputs produce bit-identical outputs.
    REQUIRE(r.value == r2.value);
    REQUIRE(r.b == r2.b);
    REQUIRE(r.b2 == r2.b2);
    REQUIRE(r.n_eff == r2.n_eff);
    REQUIRE(r.state == r2.state);
    results[tag][k.name] = r;
    return r;
  };

  for (const auto &k : cases) {
    const auto oracle = robust_frame_oracle_v2(k.c);
    const auto oracle2 = robust_frame_oracle_v2(k.c);
    REQUIRE(oracle.value == oracle2.value);
    REQUIRE(oracle.b == oracle2.b);
    REQUIRE(oracle.n_eff == oracle2.n_eff);
    const bool empty = oracle.b <= 0.0;

    std::vector<std::pair<std::string, ForwardDrizzleV2RobustResult>> rows;
    for (auto est : grouped)
      for (int groups : ks) {
        ForwardDrizzleV2RobustConfig cfg;
        cfg.groups = groups;
        const std::string tag = std::string(g3_estimator_name(est)) + "/K" +
                                std::to_string(groups);
        rows.emplace_back(tag, run(k, tag, est, cfg));
      }
    {
      ForwardDrizzleV2RobustConfig cfg;
      const std::string tag =
          std::string("reservoir_sigma_clip/R") +
          std::to_string(cfg.reservoir_size);
      rows.emplace_back(tag, run(k, tag,
                                 ForwardDrizzleV2Estimator::reservoir_sigma_clip,
                                 cfg));
    }
    // Two-pass quality reference (not selectable): K=17.
    {
      ForwardDrizzleV2RobustConfig cfg;
      const auto tp = robust_reduce_v2(
          [&](const ForwardDrizzleV2CandidateSink &sink) {
            for (const auto &v : k.c) sink(v);
          },
          cfg);
      const auto tp2 = robust_reduce_v2(
          [&](const ForwardDrizzleV2CandidateSink &sink) {
            for (const auto &v : k.c) sink(v);
          },
          cfg);
      REQUIRE(tp.value == tp2.value);
      rows.emplace_back("two_pass_winsorized_frames/K17", tp);
      results["two_pass_winsorized_frames/K17"][k.name] = tp;
    }

    for (const auto &[tag, r] : rows) {
      if (empty) {
        REQUIRE(r.state == ForwardDrizzleV2RobustState::no_source_support);
        std::printf(
            "{\"gate\":3,\"spec_sha256\":\"%s\",\"case\":\"%s\","
            "\"estimator\":\"%s\",\"state\":\"%s\"}\n",
            kGate3SpecSha, k.name, tag.c_str(), g3_state_name(r.state));
        continue;
      }
      // Support preservation: the estimator never deletes source support
      // and never produces nonfinite outputs.
      REQUIRE(r.b > 0.0);
      REQUIRE(std::isfinite(r.value));
      REQUIRE(std::isfinite(r.n_eff));
      REQUIRE(r.n_eff > 0.0);
      const double do_ = std::abs(r.value - oracle.value);
      const double dt_ = std::abs(r.value - k.truth);
      dev_oracle[tag][k.name] = do_;
      dev_truth[tag][k.name] = dt_;
      std::printf(
          "{\"gate\":3,\"spec_sha256\":\"%s\",\"case\":\"%s\","
          "\"estimator\":\"%s\",\"state\":\"%s\",\"value\":%.17g,"
          "\"b\":%.17g,\"b2\":%.17g,\"n_eff\":%.17g,\"center\":%.17g,"
          "\"scale\":%.17g,\"oracle_value\":%.17g,\"dev_oracle\":%.17g,"
          "\"dev_truth\":%.17g}\n",
          kGate3SpecSha, k.name, tag.c_str(), g3_state_name(r.state),
          r.value, r.b, r.b2, r.n_eff, r.center, r.scale, oracle.value, do_,
          dt_);
    }
  }

  // Selection rule: lowest worst-case absolute deviation from the oracle over
  // the gated cases among selectable single-pass candidates; tie-broken by
  // worst-case deviation from truth.
  std::map<std::string, double> worst_o, worst_t;
  for (const auto &[tag, per_case] : dev_oracle) {
    double wo = 0.0, wt = 0.0;
    for (const auto &[cname, d] : per_case) {
      if (oracle_gated(cname.c_str())) wo = std::max(wo, d);
      wt = std::max(wt, dev_truth[tag][cname]);
    }
    worst_o[tag] = wo;
    worst_t[tag] = wt;
    std::printf(
        "{\"gate\":3,\"spec_sha256\":\"%s\",\"summary\":true,"
        "\"estimator\":\"%s\",\"worst_vs_oracle\":%.17g,"
        "\"worst_vs_truth\":%.17g}\n",
        kGate3SpecSha, tag.c_str(), wo, wt);
  }

  std::string winner;
  double winner_wo = std::numeric_limits<double>::infinity();
  double winner_wt = std::numeric_limits<double>::infinity();
  for (const auto &[tag, wo] : worst_o) {
    if (tag.rfind("uniform", 0) == 0 ||
        tag.rfind("two_pass", 0) == 0)
      continue;  // not selectable as primary
    if (wo < winner_wo - 1e-15 ||
        (std::abs(wo - winner_wo) <= 1e-15 && worst_t[tag] < winner_wt)) {
      winner = tag;
      winner_wo = wo;
      winner_wt = worst_t[tag];
    }
  }
  REQUIRE_FALSE(winner.empty());
  std::printf(
      "{\"gate\":3,\"spec_sha256\":\"%s\",\"selection\":true,"
      "\"estimator\":\"%s\",\"worst_vs_oracle\":%.17g,"
      "\"worst_vs_truth\":%.17g}\n",
      kGate3SpecSha, winner.c_str(), winner_wo, winner_wt);

  // The selected estimator must satisfy every frozen threshold.
  REQUIRE(winner_wo <= 2.0);
  REQUIRE(dev_oracle[winner]["clean_gauss"] <= 0.25);
  REQUIRE(results[winner]["mad_zero_one_outlier"].value == 100.0);
  REQUIRE(std::abs(results[winner]["bimodal"].value - 100.0) <= 2.0);
  for (const char *cr : {"cosmic_ray_single",
                         "cosmic_ray_single_edge_frame"}) {
    const double uni_dev =
        std::abs(results["uniform/K17"][cr].value - 100.0);
    REQUIRE(dev_truth[winner][cr] <= uni_dev);
  }
  // Symmetric contamination must be handled at least as well as the oracle.
  REQUIRE(dev_oracle[winner]["symmetric_heavy_tail"] <= 0.25);
}

namespace {

constexpr const char *kGate4SpecSha =
    "808d98db7862130e9aa59ffa0c33d8d10c9c2991f789feba0a2f8c966e454bfe";

struct SumFP32 {
  float v = 0.0f;
  void add(double x) { v += static_cast<float>(x); }
  double get() const { return v; }
};

// Neumaier compensated summation in FP32.
struct SumNeu32 {
  float s = 0.0f, c = 0.0f;
  void add(double xd) {
    const float x = static_cast<float>(xd);
    const float t = s + x;
    if (std::fabs(s) >= std::fabs(x)) c += (s - t) + x;
    else c += (x - t) + s;
    s = t;
  }
  double get() const { return static_cast<double>(s) + c; }
};

struct SumFP64 {
  double v = 0.0;
  void add(double x) { v += x; }
  double get() const { return v; }
};

const char *g4_conf_state_name(ForwardDrizzleV2ConfidenceState s) {
  switch (s) {
    case ForwardDrizzleV2ConfidenceState::no_source_support:
      return "no_source_support";
    case ForwardDrizzleV2ConfidenceState::fallback_n_eff:
      return "fallback_n_eff";
    case ForwardDrizzleV2ConfidenceState::modeled:
      return "modeled";
  }
  return "unknown";
}

}  // namespace

TEST_CASE("forward drizzle v2 gate-4 sigma model and confidence states",
          "[forward-drizzle-v2][gate4]") {
  // sigma2 = sn^2 + (gx^2+gy^2)*sr^2 + half^2/3
  REQUIRE(forward_drizzle_v2_sigma2_model(1.0, 0.0, 0.0, 0.5, 0.6) ==
          Catch::Approx(1.0 + 0.0 + 0.36 / 3.0));
  REQUIRE(forward_drizzle_v2_sigma2_model(0.0, 3.0, 4.0, 0.5, 0.6) ==
          Catch::Approx(6.25 + 0.36 / 3.0));
  REQUIRE(std::isnan(
      forward_drizzle_v2_sigma2_model(-1.0, 0.0, 0.0, 0.5, 0.5)));
  REQUIRE(std::isnan(forward_drizzle_v2_sigma2_model(
      1.0, std::numeric_limits<double>::infinity(), 0.0, 0.5, 0.5)));
  REQUIRE(std::isnan(
      forward_drizzle_v2_sigma2_model(1.0, 0.0, 0.0, -0.1, 0.5)));

  std::vector<ForwardDrizzleV2RobustCandidate> c;
  for (std::size_t f = 0; f < 40; ++f)
    c.push_back({f, 100.0 + g3_noise(f), 1.0});
  std::vector<double> s2(c.size(), 1.0);

  // Modeled confidence with constant sigma equals n_eff/(n_eff+1) over the
  // effective set.
  const auto modeled = robust_reduce_candidates_v2(
      c, ForwardDrizzleV2Estimator::reservoir_sigma_clip, {}, 0, s2);
  REQUIRE(modeled.conf_state == ForwardDrizzleV2ConfidenceState::modeled);
  REQUIRE(modeled.conf_b >= 35.0);
  REQUIRE(modeled.conf_b <= 40.0);
  REQUIRE(modeled.confidence ==
          Catch::Approx(modeled.conf_b / (modeled.conf_b + 1.0))
              .epsilon(1e-12));
  REQUIRE(modeled.conf_degraded == 0);

  // Missing sigma inputs fall back to the n_eff formula; support untouched.
  const auto fb = robust_reduce_candidates_v2(
      c, ForwardDrizzleV2Estimator::reservoir_sigma_clip);
  REQUIRE(fb.conf_state == ForwardDrizzleV2ConfidenceState::fallback_n_eff);
  REQUIRE(fb.confidence == Catch::Approx(fb.n_eff / (fb.n_eff + 1.0)));
  REQUIRE(fb.b == 40.0);

  // Partially invalid sigma inputs are degraded, not fatal.
  std::vector<double> s2d(c.size(), 1.0);
  s2d[3] = std::numeric_limits<double>::quiet_NaN();
  s2d[9] = -2.0;
  const auto deg = robust_reduce_candidates_v2(
      c, ForwardDrizzleV2Estimator::reservoir_sigma_clip, {}, 0, s2d);
  REQUIRE(deg.conf_degraded == 2);
  REQUIRE(deg.conf_state == ForwardDrizzleV2ConfidenceState::modeled);
  REQUIRE(deg.conf_b >= 35.0);

  // sigma2 span of wrong size is a contract violation.
  std::vector<double> wrong(3, 1.0);
  REQUIRE_THROWS_AS(
      robust_reduce_candidates_v2(
          c, ForwardDrizzleV2Estimator::reservoir_sigma_clip, {}, 0, wrong),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      robust_frame_oracle_v2(c, 5, 3, 3.0, 3.0, wrong), std::invalid_argument);

  // No support: no confidence at all.
  std::vector<ForwardDrizzleV2RobustCandidate> empty;
  const auto none = robust_reduce_candidates_v2(
      empty, ForwardDrizzleV2Estimator::reservoir_sigma_clip);
  REQUIRE(none.conf_state ==
          ForwardDrizzleV2ConfidenceState::no_source_support);
  REQUIRE(none.confidence == 0.0);
}

TEST_CASE("forward drizzle v2 Gate-4 numerics measurement",
          "[.][forward-drizzle-v2-gate4]") {
  struct Seq {
    const char *name;
    std::vector<double> terms;
  };
  std::vector<Seq> seqs;
  {
    Seq s{"equal_weights_600", {}};
    for (int i = 0; i < 600; ++i) s.terms.push_back(1.0);
    seqs.push_back(std::move(s));
  }
  {
    Seq s{"wide_weights", {}};
    for (int i = -8; i <= 8; ++i)
      for (int j = 0; j < 37; ++j)
        s.terms.push_back(std::pow(10.0, i) * (1.0 + 0.01 * j));
    seqs.push_back(std::move(s));
  }
  {
    Seq s{"alternating_magnitude", {}};
    for (int i = 0; i < 512; ++i)
      s.terms.push_back(i % 2 == 0 ? 1.0e6 : 1.0e-6);
    seqs.push_back(std::move(s));
  }
  {
    Seq s{"cancellation", {}};
    for (int i = 0; i < 256; ++i) {
      s.terms.push_back(1.0e8);
      s.terms.push_back(-1.0e8);
      s.terms.push_back(1.0);
    }
    seqs.push_back(std::move(s));
  }
  {
    Seq s{"long_stream_1e5", {}};
    for (int i = 0; i < 100000; ++i)
      s.terms.push_back(1.0 + 0.001 * (i % 97));
    seqs.push_back(std::move(s));
  }

  for (const auto &s : seqs) {
    SumFP32 f32;
    SumNeu32 neu;
    SumFP64 f64;
    for (double t : s.terms) {
      f32.add(t);
      neu.add(t);
      f64.add(t);
    }
    const double ref = f64.get();
    const auto rel = [&](double v) {
      return ref != 0.0 ? std::abs(v - ref) / std::abs(ref)
                        : std::abs(v - ref);
    };
    std::printf(
        "{\"gate\":4,\"spec_sha256\":\"%s\",\"sequence\":\"%s\","
        "\"n\":%zu,\"ref_fp64\":%.17g,\"fp32\":%.17g,"
        "\"fp32_rel_err\":%.17g,\"neumaier_fp32\":%.17g,"
        "\"neumaier_rel_err\":%.17g,\"fp64\":%.17g}\n",
        kGate4SpecSha, s.name, s.terms.size(), ref, f32.get(), rel(f32.get()),
        neu.get(), rel(neu.get()), f64.get());
  }

  // Confidence over the gate-3 adversarial matrix with per-candidate sigma2
  // inputs derived from the sigma model (frame-dependent gradient).
  const auto cases = g3_cases();
  for (const auto &k : cases) {
    std::vector<double> s2(k.c.size());
    for (std::size_t i = 0; i < k.c.size(); ++i)
      s2[i] = forward_drizzle_v2_sigma2_model(
          1.0, 0.1 * std::sin(0.3 * static_cast<double>(k.c[i].frame_order)),
          0.1 * std::cos(0.2 * static_cast<double>(k.c[i].frame_order)), 0.5,
          0.4);
    const auto r = robust_reduce_candidates_v2(
        k.c, ForwardDrizzleV2Estimator::reservoir_sigma_clip, {}, 0, s2);
    if (r.b <= 0.0) {
      std::printf(
          "{\"gate\":4,\"spec_sha256\":\"%s\",\"case\":\"%s\","
          "\"state\":\"no_source_support\"}\n",
          kGate4SpecSha, k.name);
      continue;
    }
    REQUIRE(std::isfinite(r.confidence));
    REQUIRE(r.confidence > 0.0);
    REQUIRE(r.confidence <= 1.0);
    std::printf(
        "{\"gate\":4,\"spec_sha256\":\"%s\",\"case\":\"%s\","
        "\"conf_state\":\"%s\",\"confidence\":%.17g,\"conf_b\":%.17g,"
        "\"conf_s\":%.17g,\"conf_c\":%.17g,\"conf_degraded\":%llu,"
        "\"value\":%.17g,\"n_eff\":%.17g}\n",
        kGate4SpecSha, k.name, g4_conf_state_name(r.conf_state), r.confidence,
        r.conf_b, r.conf_s, r.conf_c,
        static_cast<unsigned long long>(r.conf_degraded), r.value, r.n_eff);
  }
}
