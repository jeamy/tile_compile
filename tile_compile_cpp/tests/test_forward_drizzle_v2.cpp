#include "tile_compile/reconstruction/drizzle_geometry_cache.hpp"
#include "tile_compile/reconstruction/forward_drizzle_cuda.hpp"
#include "tile_compile/reconstruction/forward_drizzle_v2_cpu.hpp"
#include "tile_compile/reconstruction/forward_drizzle_v2.hpp"
#include "tile_compile/reconstruction/forward_drizzle_v2_driver.hpp"
#include "tile_compile/reconstruction/forward_drizzle_v2_production.hpp"
#include "tile_compile/reconstruction/forward_drizzle_v2_store.hpp"
#include "tile_compile/reconstruction/source_quality_artifact.hpp"
#include "tile_compile/reconstruction/multiband_fusion.hpp"
#include "tile_compile/reconstruction/multiband_validation.hpp"
#include "tile_compile/core/atomic_output.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/registration/registration_sampling_plan.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
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
          "[forward-drizzle-v2][memory][gate5]") {
  ForwardDrizzleV2MemoryInputs in;
  in.target_width = 7868;
  in.target_height = 4540;
  in.channels = 3;
  in.frame_count = 40;
  in.source_device_slots = 2;
  in.source_slot_device_bytes = std::size_t{3840} * 2160 * sizeof(float);
  in.quality_device_slots = 2;
  in.quality_slot_device_bytes = std::size_t{3840} * 2160 * sizeof(float);
  in.device_reserve_bytes = std::size_t{256} << 20;
  in.pinned_source_slots = 4;
  in.pinned_quality_slots = 4;
  in.pinned_source_bytes_per_frame_row = 3840 * 4 * sizeof(float);
  in.pinned_quality_bytes_per_frame_row = 3840 * 4 * sizeof(float);
  in.host_fixed_bytes = std::size_t{2} << 30;
  in.device_budget_bytes = std::size_t{5} << 30;
  in.host_budget_bytes = std::size_t{16} << 30;

  // Exact per-pixel role arithmetic (gate-6 amended table, scale 1):
  // reservoir 128 slots*32 + kept 4 + (3+2+2+3)*8 + 8 counter + 22 coverage
  // + 64 result record = 4096+4+80+8+22+64 = 4274 per channel; +4 footprint
  // per pixel; frame planes 4*8*3 = 96 per internal pixel (sigma2 plane on).
  const auto p40 = plan_forward_drizzle_v2_memory(in);
  REQUIRE(p40.device_bytes_per_target_pixel == 3 * 4274 + 4 + 96);
  REQUIRE(p40.feasible);
  REQUIRE(p40.tile_cols == 7868);
  REQUIRE_FALSE(p40.x_tiled);
  REQUIRE(p40.band_core_rows == p40.band_rows);
  REQUIRE(p40.predicted_read_amplification == Catch::Approx(1.0));
  REQUIRE(p40.device_peak_bytes <= in.device_budget_bytes);
  REQUIRE(p40.host_peak_bytes <= in.host_budget_bytes);

  // frame_count must not change the plan at all.
  in.frame_count = 600;
  const auto p600 = plan_forward_drizzle_v2_memory(in);
  REQUIRE(p600.feasible);
  REQUIRE(p600.tile_cols == p40.tile_cols);
  REQUIRE(p600.band_rows == p40.band_rows);
  REQUIRE(p600.device_peak_bytes == p40.device_peak_bytes);

  // X-tile fallback: shrink the device budget below one full-width row.
  ForwardDrizzleV2MemoryInputs xt = in;
  xt.device_budget_bytes = xt.source_slot_device_bytes * 2 +
                           xt.quality_slot_device_bytes * 2 +
                           xt.device_reserve_bytes +
                           p40.device_bytes_per_target_pixel * 2000;
  const auto pxt = plan_forward_drizzle_v2_memory(xt);
  REQUIRE(pxt.feasible);
  REQUIRE(pxt.x_tiled);
  REQUIRE(pxt.tile_cols == 2000);
  REQUIRE(pxt.band_rows == 1);

  // Halo accounting: padded rows charged, core rows reported, RA grows.
  ForwardDrizzleV2MemoryInputs hh = in;
  hh.halo_rows_per_band_edge = 2;
  hh.device_budget_bytes = p40.device_peak_bytes;  // force tight bands
  const auto ph = plan_forward_drizzle_v2_memory(hh);
  REQUIRE(ph.feasible);
  REQUIRE(ph.band_rows == ph.band_core_rows + 4);
  REQUIRE(ph.band_count ==
          (in.target_height + ph.band_core_rows - 1) / ph.band_core_rows);
  REQUIRE(ph.predicted_read_amplification ==
          Catch::Approx((in.target_height + 4.0 * ph.band_count) /
                        in.target_height));

  // RA bound rejection: tiny bound with nonzero halo must be infeasible when
  // more than one band is required.
  ForwardDrizzleV2MemoryInputs rb = hh;
  rb.max_source_read_amplification = 1.0;
  rb.device_budget_bytes =
      rb.device_reserve_bytes + rb.source_slot_device_bytes * 2 +
      rb.quality_slot_device_bytes * 2 +
      p40.device_bytes_per_target_pixel * rb.target_width * 5;
  const auto prb = plan_forward_drizzle_v2_memory(rb);
  REQUIRE_FALSE(prb.feasible);

  // Infeasible and overflow paths.
  ForwardDrizzleV2MemoryInputs bad = in;
  bad.device_budget_bytes = 1024;
  REQUIRE_FALSE(plan_forward_drizzle_v2_memory(bad).feasible);
  bad = in;
  bad.device_budget_bytes = std::numeric_limits<std::size_t>::max();
  bad.pinned_source_bytes_per_frame_row =
      std::numeric_limits<std::size_t>::max();
  REQUIRE_FALSE(plan_forward_drizzle_v2_memory(bad).feasible);
  bad = in;
  bad.pinned_source_slots = 0;
  REQUIRE_FALSE(plan_forward_drizzle_v2_memory(bad).feasible);
  bad = in;
  bad.channels = 2;
  REQUIRE_FALSE(plan_forward_drizzle_v2_memory(bad).feasible);
  bad = in;
  bad.max_source_read_amplification = 0.5;
  REQUIRE_FALSE(plan_forward_drizzle_v2_memory(bad).feasible);
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

TEST_CASE("forward drizzle v2 Gate-5 memory plan measurement",
          "[.][forward-drizzle-v2-gate5]") {
  const char *sha =
      "0f47490138ffb9de7a7308ce0e27a6cb20a4068074dc7952a69e346b2887dea4";
  struct S {
    const char *name;
    int halo;
    std::size_t dev_gib, host_gib;
    int channels;
  };
  const S scenarios[] = {
      {"production_osc_5gib_16gib", 0, 5, 16, 3},
      {"production_osc_4gib_16gib", 0, 4, 16, 3},
      {"production_osc_5gib_halo14", 14, 5, 16, 3},
      {"production_osc_5gib_halo62", 62, 5, 16, 3},
      {"production_mono_5gib_16gib", 0, 5, 16, 1},
      {"production_osc_2gib_16gib", 0, 2, 16, 3},
  };
  for (const auto &s : scenarios) {
    ForwardDrizzleV2MemoryInputs in;
    in.target_width = 7868;
    in.target_height = 4540;
    in.channels = s.channels;
    in.frame_count = 600;
    in.source_device_slots = 2;
    in.source_slot_device_bytes = std::size_t{3840} * 2160 * sizeof(float);
    in.quality_device_slots = 2;
    in.quality_slot_device_bytes = std::size_t{3840} * 2160 * sizeof(float);
    in.device_reserve_bytes = std::size_t{256} << 20;
    in.pinned_source_slots = 4;
    in.pinned_quality_slots = 4;
    in.pinned_source_bytes_per_frame_row = 3840 * 4 * sizeof(float);
    in.pinned_quality_bytes_per_frame_row = 3840 * 4 * sizeof(float);
    in.host_fixed_bytes = std::size_t{2} << 30;
    in.device_budget_bytes = s.dev_gib << 30;
    in.host_budget_bytes = s.host_gib << 30;
    in.halo_rows_per_band_edge = s.halo;
    const auto p = plan_forward_drizzle_v2_memory(in);
    std::printf(
        "{\"gate\":5,\"spec_sha256\":\"%s\",\"scenario\":\"%s\","
        "\"feasible\":%s,\"tile_cols\":%d,\"x_tiled\":%s,"
        "\"band_rows\":%d,\"band_core_rows\":%d,\"band_count\":%d,"
        "\"bytes_per_pixel\":%llu,\"device_dynamic\":%llu,"
        "\"device_peak\":%llu,\"host_pinned\":%llu,\"host_peak\":%llu,"
        "\"predicted_ra\":%.6f}\n",
        sha, s.name, p.feasible ? "true" : "false", p.tile_cols,
        p.x_tiled ? "true" : "false", p.band_rows, p.band_core_rows,
        p.band_count,
        static_cast<unsigned long long>(p.device_bytes_per_target_pixel),
        static_cast<unsigned long long>(p.device_dynamic_bytes),
        static_cast<unsigned long long>(p.device_peak_bytes),
        static_cast<unsigned long long>(p.host_pinned_bytes),
        static_cast<unsigned long long>(p.host_peak_bytes),
        p.predicted_read_amplification);
  }
}

// --- Gate 6: minimal affine prototype kernel -------------------------------

namespace {

// CPU oracle for the gate-6 kernel: per-frame CPU gathers give the
// frame-local internal planes (A, B_src via the finite-value image; B_geo via
// an all-finite image; S2 via a sigma2-valued image masked to finite source
// samples). The fold, stream confidence and reservoir clip are exactly the
// gate-2/3/4 CPU contracts applied to the same candidates.
struct Gate6CpuRef {
  std::vector<ForwardDrizzleV2RobustResult> results;  // [c][px] channel-major
  std::vector<unsigned short> masks;                  // geo|src<<4
  std::vector<unsigned int> footprint;
  std::uint64_t dense_overlap = 0;
};

Gate6CpuRef gate6_cpu_reference(
    const Fixture &f, const std::vector<float> &sigma2_or_empty,
    const ForwardDrizzleV2KernelConfig &kcfg) {
  const int scale = kcfg.internal_scale;
  const int nc = f.plan.canvas_width_native;
  const int nr = f.plan.canvas_height_native;
  const int ic = nc * scale;
  const int ir = nr * scale;
  const int channels = kcfg.mono ? 1 : 3;
  const std::size_t nplane = static_cast<std::size_t>(nc) * nr;
  const std::size_t iplane = static_cast<std::size_t>(ic) * ir;
  const double inv_s2 = 1.0 / (static_cast<double>(scale) * scale);
  const std::size_t n_frames = f.plan.frames.size();

  Matrix2Df geo_img(f.plan.source_height, f.plan.source_width);
  geo_img.setConstant(1.0f);
  std::vector<Matrix2Df> s2_img;
  if (!sigma2_or_empty.empty()) {
    s2_img.reserve(n_frames);
    for (std::size_t i = 0; i < n_frames; ++i) {
      Matrix2Df m(f.plan.source_height, f.plan.source_width);
      for (int y = 0; y < m.rows(); ++y)
        for (int x = 0; x < m.cols(); ++x) {
          const std::size_t si = static_cast<std::size_t>(y) * m.cols() + x;
          const float v = f.images[i](y, x);
          m(y, x) = std::isfinite(v) ? sigma2_or_empty[si]
                                     : std::numeric_limits<float>::quiet_NaN();
        }
      s2_img.push_back(std::move(m));
    }
  }

  Gate6CpuRef ref;
  ref.results.assign(nplane * channels, ForwardDrizzleV2RobustResult{});
  ref.masks.assign(nplane * channels, 0);
  ref.footprint.assign(nplane, 0);
  std::vector<std::vector<ForwardDrizzleV2RobustCandidate>> cands(
      nplane * channels);
  std::vector<std::vector<double>> s2v(nplane * channels);

  for (std::size_t fr = 0; fr < n_frames; ++fr) {
    RegistrationSamplingPlan one = f.plan;
    one.frames = {f.plan.frames[fr]};
    const auto &img = f.images[fr];
    auto real_of = [&](std::size_t) -> const Matrix2Df & { return img; };
    auto geo_of = [&](std::size_t) -> const Matrix2Df & { return geo_img; };
    const auto g_a = gather_affine_uniform_v2(one, real_of, f.cfg, 0, ir);
    const auto g_g = gather_affine_uniform_v2(one, geo_of, f.cfg, 0, ir);
    ForwardDrizzleV2UniformResult g_s;
    if (!sigma2_or_empty.empty()) {
      auto s2_of = [&](std::size_t) -> const Matrix2Df & {
        return s2_img[fr];
      };
      g_s = gather_affine_uniform_v2(one, s2_of, f.cfg, 0, ir);
    }
    for (std::size_t px = 0; px < nplane; ++px) {
      const int nx = static_cast<int>(px % nc);
      const int ny = static_cast<int>(px / nc);
      bool any_geo = false;
      for (int c = 0; c < channels; ++c) {
        const std::size_t pc = static_cast<std::size_t>(c) * nplane + px;
        double a = 0.0, bs = 0.0, bg = 0.0, sw = 0.0;
        unsigned int geo_bits = 0, src_bits = 0;
        for (int iy = 0; iy < scale; ++iy)
          for (int ix = 0; ix < scale; ++ix) {
            const int j = iy * scale + ix;
            const std::size_t ii =
                static_cast<std::size_t>(ny * scale + iy) * ic + nx * scale +
                ix;
            // w[c]/wx[c] are already the per-channel planes of size iplane.
            a += inv_s2 * g_a.accum.wx[c][ii];
            bs += inv_s2 * g_a.accum.w[c][ii];
            bg += inv_s2 * g_g.accum.w[c][ii];
            if (!sigma2_or_empty.empty())
              sw += inv_s2 * g_s.accum.wx[c][ii];
            if (g_g.accum.w[c][ii] > 0.0) geo_bits |= 1u << j;
            if (g_a.accum.w[c][ii] > 0.0) src_bits |= 1u << j;
          }
        ref.masks[pc] |=
            static_cast<unsigned short>(geo_bits | (src_bits << 4));
        if (bg > 0.0) any_geo = true;
        if (!(bs > 0.0)) continue;
        cands[pc].push_back({fr, a / bs, bs});
        if (!sigma2_or_empty.empty()) s2v[pc].push_back(sw / bs);
      }
      if (any_geo) ++ref.footprint[px];
    }
  }

  ForwardDrizzleV2RobustConfig rcfg;
  rcfg.reservoir_size = kcfg.reservoir_size;
  rcfg.reservoir_seed = kcfg.reservoir_seed;
  rcfg.oracle_min_clip_contributors = kcfg.min_clip_contributors;
  rcfg.oracle_passes = kcfg.robust_passes;
  rcfg.oracle_sigma_low = kcfg.sigma_low;
  rcfg.oracle_sigma_high = kcfg.sigma_high;
  rcfg.min_candidates = kcfg.min_candidates;
  for (std::size_t pc = 0; pc < nplane * channels; ++pc) {
    if (cands[pc].empty()) continue;
    ref.results[pc] = robust_reduce_candidates_v2(
        cands[pc], ForwardDrizzleV2Estimator::reservoir_sigma_clip, rcfg,
        static_cast<std::uint64_t>(n_frames), s2v[pc]);
  }
  for (std::size_t px = 0; px < nplane; ++px)
    if (ref.footprint[px] == n_frames && n_frames > 0) ++ref.dense_overlap;
  return ref;
}

void require_gate6_parity(const Gate6CpuRef &ref,
                          const std::vector<ForwardDrizzleV2PixelResult> &gpu,
                          int ncols, int nrows, int channels) {
  const std::size_t nplane = static_cast<std::size_t>(ncols) * nrows;
  for (std::size_t pc = 0; pc < nplane * channels; ++pc) {
    const auto &e = ref.results[pc];
    const auto &g = gpu[pc];
    if (g.robust_state != static_cast<std::uint8_t>(e.state) ||
        g.contributors != e.candidates)
      std::fprintf(stderr,
                   "gate6 mismatch pc=%llu dev(state=%u contrib=%u b=%.6g "
                   "degraded=%llu) cpu(state=%d cand=%llu b=%.6g "
                   "degraded=%llu)\n",
                   (unsigned long long)pc, (unsigned)g.robust_state,
                   g.contributors, g.b, (unsigned long long)g.conf_degraded,
                   (int)e.state, (unsigned long long)e.candidates, e.b,
                   (unsigned long long)e.conf_degraded);
    REQUIRE(g.robust_state ==
            static_cast<std::uint8_t>(e.state));
    REQUIRE(g.confidence_state ==
            static_cast<std::uint8_t>(e.conf_state));
    REQUIRE(g.contributors == e.candidates);
    REQUIRE(g.conf_degraded == e.conf_degraded);
    REQUIRE(g.value == Catch::Approx(e.value).epsilon(1e-12).margin(1e-12));
    REQUIRE(g.b == Catch::Approx(e.b).epsilon(1e-12).margin(1e-12));
    REQUIRE(g.n_eff == Catch::Approx(e.n_eff).epsilon(1e-12).margin(1e-12));
    REQUIRE(g.confidence ==
            Catch::Approx(e.confidence).epsilon(1e-12).margin(1e-12));
  }
}

}  // namespace

TEST_CASE("forward drizzle v2 gate6 prototype kernel matches the CPU oracle",
          "[forward-drizzle-v2][cuda-parity][gate6]") {
  if (!forward_drizzle_cuda_runtime_available()) {
    SUCCEED("CUDA device unavailable");
    return;
  }
  const BayerPattern patterns[] = {BayerPattern::RGGB, BayerPattern::BGGR,
                                   BayerPattern::GRBG, BayerPattern::GBRG};
  for (int scale : {1, 2}) {
    for (ColorMode mode : {ColorMode::MONO, ColorMode::OSC}) {
      for (BayerPattern pattern : patterns) {
        auto f = make_fixture(mode, pattern, 1, -1, scale);
        const int nc = f.plan.canvas_width_native;
        const int nr = f.plan.canvas_height_native;
        const int channels = mode == ColorMode::MONO ? 1 : 3;
        const std::size_t nplane = static_cast<std::size_t>(nc) * nr;
        const std::size_t n_src = static_cast<std::size_t>(
            f.plan.source_width) * f.plan.source_height;
        std::vector<float> s2(n_src);
        for (std::size_t i = 0; i < n_src; ++i)
          s2[i] = 0.01f + 0.0001f * static_cast<float>(i % 97);
        s2[3] = -1.0f;  // degraded-confidence coverage on both paths
        ForwardDrizzleV2KernelConfig kcfg;
        kcfg.internal_scale = scale;
        kcfg.stream_length = f.plan.frames.size();
        kcfg.half = 0.5 * f.cfg.pixfrac;
        kcfg.bayer_pattern = static_cast<int>(pattern);
        kcfg.cfa_origin_x = f.plan.cfa_origin_x;
        kcfg.cfa_origin_y = f.plan.cfa_origin_y;
        kcfg.mono = mode == ColorMode::MONO;

        const auto ref = gate6_cpu_reference(f, s2, kcfg);
        ForwardDrizzleV2CudaPrototypeKernel kernel;
        REQUIRE(kernel.reserve(nc, nr, f.plan.source_width,
                               f.plan.source_height, kcfg));
        for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
          const auto &m = f.plan.frames[fr].source_to_canvas;
          const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                                m(1, 0), m(1, 1), m(1, 2)};
          REQUIRE(kernel.accumulate_frame(a6, f.images[fr].data(), s2.data(),
                                          fr));
        }
        std::vector<ForwardDrizzleV2PixelResult> got(nplane * channels);
        std::uint64_t dense = 0;
        REQUIRE(kernel.finalize(got.data(), nullptr, &dense));
        require_gate6_parity(ref, got, nc, nr, channels);
        REQUIRE(dense == ref.dense_overlap);
        // Fractions must equal the CPU mask-derived values bit for bit.
        const double subpixels = static_cast<double>(scale) * scale;
        for (std::size_t pc = 0; pc < nplane * channels; ++pc) {
          const unsigned int m = ref.masks[pc];
          const float geo = static_cast<float>(
              static_cast<double>(__builtin_popcount(m & 0xFu)) / subpixels);
          const float src = static_cast<float>(
              static_cast<double>(__builtin_popcount((m >> 4) & 0xFu)) /
              subpixels);
          REQUIRE(got[pc].geometry_fraction == geo);
          REQUIRE(got[pc].source_fraction == src);
          REQUIRE(got[pc].estimator_fraction == src);
          REQUIRE(got[pc].profile_fraction == src);
        }
        const auto &st = kernel.stats();
        REQUIRE(st.allocations == 1);
        REQUIRE(st.device_global_synchronizations == 0);
        REQUIRE(st.stream_synchronizations == 1);
        REQUIRE(st.frames_processed == f.plan.frames.size());
        REQUIRE(st.positive_overlaps > 0);
        REQUIRE(st.candidates_streamed > 0);
        REQUIRE(st.reservoir_kept_total > 0);
        REQUIRE(st.slot_transitions == f.plan.frames.size());
        REQUIRE(st.source_bytes_uploaded ==
                2 * f.plan.frames.size() * n_src * sizeof(float));
        REQUIRE(st.result_bytes_downloaded > 0);
      }
    }
  }
}

TEST_CASE("forward drizzle v2 gate6 kernel reservoir sampling N > R",
          "[forward-drizzle-v2][cuda-parity][gate6]") {
  if (!forward_drizzle_cuda_runtime_available()) {
    SUCCEED("CUDA device unavailable");
    return;
  }
  auto f = make_fixture(ColorMode::OSC, BayerPattern::GBRG, 0, 0, 2);
  const std::size_t keep = f.plan.frames.size();
  for (std::size_t i = keep; i < 80; ++i) {
    FrameSamplingTransform frame = f.plan.frames[i % keep];
    frame.frame_id = "v2-extra-" + std::to_string(i);
    frame.source_index = i;
    // Small deterministic transform perturbation per frame.
    frame.source_to_canvas(0, 2) += static_cast<float>(0.01 * (i - keep));
    f.plan.frames.push_back(frame);
    f.images.push_back(f.images[i % keep]);
  }
  const int nc = f.plan.canvas_width_native;
  const int nr = f.plan.canvas_height_native;
  const std::size_t nplane = static_cast<std::size_t>(nc) * nr;
  const std::size_t n_src = static_cast<std::size_t>(f.plan.source_width) *
                            f.plan.source_height;
  ForwardDrizzleV2KernelConfig kcfg;
  kcfg.internal_scale = 2;
  kcfg.stream_length = f.plan.frames.size();
  kcfg.half = 0.5 * f.cfg.pixfrac;
  kcfg.bayer_pattern = static_cast<int>(BayerPattern::GBRG);
  kcfg.mono = false;
  const auto ref = gate6_cpu_reference(f, {}, kcfg);
  ForwardDrizzleV2CudaPrototypeKernel kernel;
  REQUIRE(kernel.reserve(nc, nr, f.plan.source_width, f.plan.source_height,
                         kcfg));
  for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
    const auto &m = f.plan.frames[fr].source_to_canvas;
    const double a6[6] = {m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1), m(1, 2)};
    REQUIRE(kernel.accumulate_frame(a6, f.images[fr].data(), nullptr, fr));
  }
  std::vector<ForwardDrizzleV2PixelResult> got(nplane * 3);
  std::uint64_t dense = 0;
  REQUIRE(kernel.finalize(got.data(), nullptr, &dense));
  require_gate6_parity(ref, got, nc, nr, 3);
  REQUIRE(dense == ref.dense_overlap);
  // The kept set is a strict subset of the 80-frame stream on covered
  // pixels, so reservoir_kept_total < candidates_streamed.
  REQUIRE(kernel.stats().reservoir_kept_total <=
          kernel.stats().candidates_streamed);
  REQUIRE(kernel.stats().reservoir_kept_total > 0);
  (void)n_src;
}

TEST_CASE("forward drizzle v2 gate6 kernel degenerate and repeatability",
          "[forward-drizzle-v2][cuda-parity][gate6]") {
  if (!forward_drizzle_cuda_runtime_available()) {
    SUCCEED("CUDA device unavailable");
    return;
  }
  auto f = make_fixture(ColorMode::MONO, BayerPattern::RGGB, 0, 0, 2);
  const int nc = f.plan.canvas_width_native;
  const int nr = f.plan.canvas_height_native;
  const std::size_t nplane = static_cast<std::size_t>(nc) * nr;
  ForwardDrizzleV2KernelConfig kcfg;
  kcfg.internal_scale = 2;
  kcfg.stream_length = 4;
  kcfg.half = 0.5 * f.cfg.pixfrac;
  kcfg.mono = true;

  // Empty band: no frames -> every result is no_source_support, dense 0.
  {
    ForwardDrizzleV2CudaPrototypeKernel kernel;
    REQUIRE(kernel.reserve(nc, nr, f.plan.source_width, f.plan.source_height,
                           kcfg));
    std::vector<ForwardDrizzleV2PixelResult> got(nplane);
    std::uint64_t dense = 999;
    REQUIRE(kernel.finalize(got.data(), nullptr, &dense));
    REQUIRE(dense == 0);
    for (const auto &r : got) {
      REQUIRE(r.robust_state == static_cast<std::uint8_t>(
                                  ForwardDrizzleV2RobustState::no_source_support));
      REQUIRE(r.contributors == 0);
      REQUIRE(r.b == 0.0);
      REQUIRE(r.confidence == 0.0);
    }
    REQUIRE(kernel.stats().frames_processed == 0);
    REQUIRE(kernel.stats().stream_synchronizations == 1);
  }

  // Single frame: N=1 <= R, uniform value, n_eff = 1 on covered pixels.
  {
    ForwardDrizzleV2KernelConfig one = kcfg;
    one.stream_length = 1;
    ForwardDrizzleV2CudaPrototypeKernel kernel;
    REQUIRE(kernel.reserve(nc, nr, f.plan.source_width, f.plan.source_height,
                           one));
    const auto &m = f.plan.frames[0].source_to_canvas;
    const double a6[6] = {m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1), m(1, 2)};
    REQUIRE(kernel.accumulate_frame(a6, f.images[0].data(), nullptr, 0));
    // A second frame must be rejected (stream_length reached).
    REQUIRE_FALSE(kernel.accumulate_frame(a6, f.images[0].data(), nullptr, 1));
    std::vector<ForwardDrizzleV2PixelResult> got(nplane);
    std::uint64_t dense = 0;
    REQUIRE(kernel.finalize(got.data(), nullptr, &dense));
    REQUIRE(dense > 0);
    bool saw_supported = false;
    for (const auto &r : got) {
      if (r.contributors == 0) continue;
      saw_supported = true;
      REQUIRE(r.robust_state ==
              static_cast<std::uint8_t>(
                  ForwardDrizzleV2RobustState::too_few_candidates_fallback));
      REQUIRE(r.n_eff == Catch::Approx(1.0).margin(1e-12));
      REQUIRE(std::isfinite(r.value));
    }
    REQUIRE(saw_supported);
  }

  // Repeatability: two identical runs agree bit-exact on states/masks and to
  // 1e-12 on values (atomic ordering is not a guaranteed contract).
  {
    auto run_once = [&] {
      ForwardDrizzleV2CudaPrototypeKernel kernel;
      REQUIRE(kernel.reserve(nc, nr, f.plan.source_width,
                             f.plan.source_height, kcfg));
      for (std::size_t fr = 0; fr < 4; ++fr) {
        const auto &m = f.plan.frames[fr].source_to_canvas;
        const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                              m(1, 0), m(1, 1), m(1, 2)};
        REQUIRE(kernel.accumulate_frame(a6, f.images[fr].data(), nullptr, fr));
      }
      std::vector<ForwardDrizzleV2PixelResult> got(nplane);
      std::uint64_t dense = 0;
      REQUIRE(kernel.finalize(got.data(), nullptr, &dense));
      return std::pair{got, dense};
    };
    const auto r1 = run_once();
    const auto r2 = run_once();
    REQUIRE(r1.second == r2.second);
    for (std::size_t i = 0; i < nplane; ++i) {
      REQUIRE(r1.first[i].robust_state == r2.first[i].robust_state);
      REQUIRE(r1.first[i].confidence_state == r2.first[i].confidence_state);
      REQUIRE(r1.first[i].contributors == r2.first[i].contributors);
      REQUIRE(r1.first[i].geometry_fraction ==
              r2.first[i].geometry_fraction);
      REQUIRE(r1.first[i].source_fraction == r2.first[i].source_fraction);
      REQUIRE(r1.first[i].value ==
              Catch::Approx(r2.first[i].value).epsilon(1e-12).margin(1e-12));
      REQUIRE(r1.first[i].b ==
              Catch::Approx(r2.first[i].b).epsilon(1e-12).margin(1e-12));
    }
  }
}

TEST_CASE("forward drizzle v2 gate6 workspace memory stays within the plan",
          "[forward-drizzle-v2][cuda-parity][gate6]") {
  if (!forward_drizzle_cuda_runtime_available()) {
    SUCCEED("CUDA device unavailable");
    return;
  }
  ForwardDrizzleV2KernelConfig kcfg;
  kcfg.internal_scale = 2;
  kcfg.stream_length = 4;
  kcfg.mono = false;
  ForwardDrizzleV2CudaPrototypeKernel kernel;
  REQUIRE(kernel.reserve(64, 8, 32, 16, kcfg));
  ForwardDrizzleV2MemoryInputs in;
  in.target_width = 64;
  in.target_height = 8;
  in.channels = 3;
  in.frame_count = 4;
  in.internal_scale = 2;
  in.source_device_slots = 1;
  in.source_slot_device_bytes = std::size_t{32} * 16 * sizeof(float);
  in.quality_device_slots = 1;
  in.quality_slot_device_bytes = std::size_t{32} * 16 * sizeof(float);
  in.device_reserve_bytes = 1 << 20;
  in.device_budget_bytes = std::size_t{4} << 30;
  in.host_budget_bytes = std::size_t{4} << 30;
  const auto p = plan_forward_drizzle_v2_memory(in);
  REQUIRE(p.feasible);
  const std::size_t nplane = std::size_t{64} * 8;
  // The workspace's per-native-pixel footprint must fit the gate-5 role
  // budget for the same band (single frame plane set, no X-tiling).
  REQUIRE(kernel.device_bytes_per_native_pixel() <=
          p.device_bytes_per_target_pixel);
  (void)nplane;
}

TEST_CASE("forward drizzle v2 gate6 production geometry pipeline",
          "[.][forward-drizzle-v2-gate6-production]") {
  if (!forward_drizzle_cuda_runtime_available()) {
    SUCCEED("CUDA device unavailable");
    return;
  }
  constexpr char kSpecSha[] =
      "19e2fb471cd321781e36e963572f8fc6137f49c2202f575f2928bf236c17c749";
  constexpr int sw = 3840, sh = 2160;      // source
  constexpr int nc = 3934, nr = 2270;      // native canvas
  constexpr double kMaxSteadySeconds = 0.75;

  // Pick the band height through the gate-5 planner against the actual
  // device budget (80% of free memory, role table defaults).
  const auto mem = forward_drizzle_cuda_device_memory();
  ForwardDrizzleV2MemoryInputs in;
  in.target_width = nc;
  in.target_height = nr;
  in.channels = 3;
  in.frame_count = 60;
  in.internal_scale = 2;
  in.source_device_slots = 2;
  in.source_slot_device_bytes = std::size_t{sw} * sh * sizeof(float);
  in.quality_device_slots = 1;
  in.quality_slot_device_bytes = std::size_t{sw} * sh * sizeof(float);
  in.device_reserve_bytes = std::size_t{256} << 20;
  in.device_budget_bytes =
      static_cast<std::size_t>(static_cast<double>(mem.free_bytes) * 0.8);
  in.host_budget_bytes = std::size_t{8} << 30;
  const auto plan = plan_forward_drizzle_v2_memory(in);
  REQUIRE(plan.feasible);
  REQUIRE(plan.tile_cols == nc);
  const int band_rows = plan.band_rows;

  Matrix2Df source(sh, sw);
  for (int y = 0; y < sh; ++y)
    for (int x = 0; x < sw; ++x)
      source(y, x) = static_cast<float>(
          100.0 + 0.001 * x + 0.002 * y + 0.01 * ((17 * x + 31 * y) % 29));

  // 60 affine transforms from the five real M42 cases with a small
  // deterministic dither. Mild transforms only: the shared footprint then
  // keeps a nonempty dense-overlap core, which the telemetry check needs.
  const std::array<std::array<double, 6>, 5> base{{
      {1.0, 0.0, 32.0, 0.0, 1.0, 50.0},
      {0.9998640418052673, -0.016167480498552322, 35.636409759521484,
       0.01610037311911583, 0.9999631643295288, 46.23441696166992},
      {0.9998466372489929, 0.017386717721819878, 55.92152404785156,
       -0.01743965968489647, 0.9998936057090759, 68.48160552978516},
      {0.9999858140945435, -0.0024168870877474546, 31.47933578491211,
       0.0021460296120494604, 1.000150442123413, 56.29893493652344},
      {1.0000033378601074, -0.006749980617314577, 28.21784782409668,
       0.006637410260736942, 1.0002058744430542, 58.95621871948242},
  }};

  ForwardDrizzleV2KernelConfig kcfg;
  kcfg.internal_scale = 2;
  kcfg.stream_length = 60;
  kcfg.half = 0.4;  // pixfrac 0.8
  kcfg.bayer_pattern = static_cast<int>(BayerPattern::GBRG);
  kcfg.mono = false;

  ForwardDrizzleV2CudaPrototypeKernel kernel;
  REQUIRE(kernel.reserve(nc, band_rows, sw, sh, kcfg));
  const auto &st0 = kernel.stats();
  REQUIRE(st0.allocations == 1);
  for (std::uint64_t f = 0; f < 60; ++f) {
    auto m = base[f % base.size()];
    m[2] += 0.05 * static_cast<double>(f);  // deterministic dither
    m[5] += 0.03 * static_cast<double>(f);
    REQUIRE(kernel.accumulate_frame(m.data(), source.data(), nullptr, f));
  }
  // Independent dense-overlap oracle: CPU gather per frame over the same
  // band, footprint = any positive B in the native pixel's subpixels.
  RegistrationSamplingPlan cpu_plan;
  cpu_plan.source_width = sw;
  cpu_plan.source_height = sh;
  cpu_plan.canvas_width_native = nc;
  cpu_plan.canvas_height_native = nr;
  cpu_plan.color_mode = ColorMode::OSC;
  cpu_plan.bayer_pattern = BayerPattern::GBRG;
  config::ReconstructionDrizzleConfig cpu_cfg;
  cpu_cfg.internal_scale = 2;
  cpu_cfg.pixfrac = 0.8f;
  cpu_cfg.kernel = "square";
  std::vector<unsigned int> fp_cpu(
      static_cast<std::size_t>(nc) * band_rows, 0);
  for (std::uint64_t f = 0; f < 60; ++f) {
    auto m = base[f % base.size()];
    m[2] += 0.05 * static_cast<double>(f);
    m[5] += 0.03 * static_cast<double>(f);
    RegistrationSamplingPlan one = cpu_plan;
    FrameSamplingTransform fr;
    fr.frame_id = "g6-" + std::to_string(f);
    fr.source_index = 0;
    fr.valid = true;
    fr.source_to_canvas = affine(m[0], m[1], m[2], m[3], m[4], m[5]);
    fr.source_to_canvas_affine_valid = true;
    one.frames = {fr};
    auto one_of = [&](std::size_t) -> const Matrix2Df & { return source; };
    const auto g = gather_affine_uniform_v2(one, one_of, cpu_cfg, 0,
                                            band_rows * 2);
    const std::size_t icols = static_cast<std::size_t>(nc) * 2;
    for (int ny = 0; ny < band_rows; ++ny)
      for (int nx = 0; nx < nc; ++nx) {
        bool any = false;
        for (int c = 0; c < 3 && !any; ++c)
          for (int iy = 0; iy < 2 && !any; ++iy)
            for (int ix = 0; ix < 2 && !any; ++ix)
              any = g.accum.w[c][static_cast<std::size_t>(ny * 2 + iy) *
                                     icols + nx * 2 + ix] > 0.0;
        if (any) ++fp_cpu[static_cast<std::size_t>(ny) * nc + nx];
      }
  }
  std::uint64_t dense_cpu = 0;
  for (unsigned int v : fp_cpu) dense_cpu += (v == 60);

  const std::size_t n_out =
      static_cast<std::size_t>(nc) * band_rows * 3;
  std::vector<ForwardDrizzleV2PixelResult> results(n_out);
  std::uint64_t dense = 0;
  REQUIRE(kernel.finalize(results.data(), nullptr, &dense));
  const auto &st = kernel.stats();

  std::uint64_t supported = 0, clipped = 0, fallbacks = 0;
  for (const auto &r : results) {
    if (r.contributors == 0) continue;
    ++supported;
    if (r.robust_state ==
        static_cast<std::uint8_t>(
            ForwardDrizzleV2RobustState::primary_reservoir_sigma_clip))
      ++clipped;
    else if (r.robust_state == static_cast<std::uint8_t>(
                 ForwardDrizzleV2RobustState::reservoir_overflow_fallback))
      ++fallbacks;
    REQUIRE(std::isfinite(r.value));
    REQUIRE(std::isfinite(r.b));
    REQUIRE(r.geometry_fraction > 0.0f);
  }
  std::printf(
      "{\"gate\":6,\"spec_sha256\":\"%s\",\"case\":\"production_band\","
      "\"native_cols\":%d,\"band_rows\":%d,\"frames\":60,"
      "\"reserved_bytes\":%llu,\"planned_band_rows\":%d,"
      "\"per_px_device\":%llu,\"per_px_plan\":%llu,"
      "\"upload_s\":%.6f,\"kernel_s\":%.6f,\"max_frame_s\":%.6f,"
      "\"mean_frame_s\":%.6f,\"positive_overlaps\":%llu,"
      "\"candidates_streamed\":%llu,\"reservoir_kept\":%llu,"
      "\"dense_overlap\":%llu,\"supported\":%llu,\"clipped\":%llu,"
      "\"overflow_fallbacks\":%llu,\"allocations\":%llu,"
      "\"global_syncs\":%llu,\"stream_syncs\":%llu}\n",
      kSpecSha, nc, band_rows,
      static_cast<unsigned long long>(st.reserved_device_bytes), band_rows,
      static_cast<unsigned long long>(kernel.device_bytes_per_native_pixel()),
      static_cast<unsigned long long>(plan.device_bytes_per_target_pixel),
      st.upload_seconds, st.kernel_seconds, st.max_frame_seconds,
      (st.upload_seconds + st.kernel_seconds) / 60.0,
      st.positive_overlaps, st.candidates_streamed, st.reservoir_kept_total,
      static_cast<unsigned long long>(dense),
      static_cast<unsigned long long>(supported),
      static_cast<unsigned long long>(clipped),
      static_cast<unsigned long long>(fallbacks),
      static_cast<unsigned long long>(st.allocations),
      static_cast<unsigned long long>(st.device_global_synchronizations),
      static_cast<unsigned long long>(st.stream_synchronizations));

  REQUIRE(supported > 0);
  REQUIRE(clipped > 0);
  REQUIRE(fallbacks == 0);
  REQUIRE(dense == dense_cpu);
  REQUIRE(st.allocations == 1);
  REQUIRE(st.device_global_synchronizations == 0);
  REQUIRE(st.stream_synchronizations == 1);
  REQUIRE(st.frames_processed == 60);
  REQUIRE(st.max_frame_seconds <= kMaxSteadySeconds);
  REQUIRE(st.reserved_device_bytes <= plan.device_peak_bytes);
  REQUIRE(kernel.device_bytes_per_native_pixel() <=
          plan.device_bytes_per_target_pixel);
}

namespace {

struct V2StoreFixture {
  core::AtomicOutput staging{fs::temp_directory_path() / "fdv2-store-test"};
  fs::path root = staging.path();
  V2StoreFixture() { fs::create_directories(root); }
  ~V2StoreFixture() {
    std::error_code ec;
    fs::remove_all(root, ec);
  }
};

ForwardDrizzleV2RunPlan v2_store_plan(int width = 8, int height = 12,
                                    int band_rows = 4, int channels = 1) {
  ForwardDrizzleV2RunPlan p;
  p.source_identity_hash = "src-hash";
  p.normalized_cache_hash = "cache-hash";
  p.quality_plan_hash = "q-hash";
  p.sampling_plan_hash = "sampling-hash";
  p.config_snapshot_hash = "cfg-hash";
  p.native_width = width;
  p.native_height = height;
  p.channels = channels;
  p.internal_scale = 2;
  p.color_mode = channels == 1 ? "MONO" : "OSC";
  p.frame_count = 5;
  p.band_rows = band_rows;
  p.band_count = (height + band_rows - 1) / band_rows;
  finalize_forward_drizzle_v2_run_plan(p);
  return p;
}

std::vector<ForwardDrizzleV2PixelResult> v2_band_records(int rows, int cols,
                                                       int channels,
                                                       double seed) {
  std::vector<ForwardDrizzleV2PixelResult> r(
      static_cast<std::size_t>(rows) * cols * channels);
  for (std::size_t i = 0; i < r.size(); ++i) {
    r[i].value = seed + static_cast<double>(i) * 0.5;
    r[i].b = 1.25;
    r[i].n_eff = 4.0;
    r[i].confidence = 0.5;
    r[i].contributors = 5;
    r[i].robust_state = static_cast<std::uint8_t>(
        ForwardDrizzleV2RobustState::primary_reservoir_sigma_clip);
    r[i].confidence_state = static_cast<std::uint8_t>(
        ForwardDrizzleV2ConfidenceState::modeled);
  }
  return r;
}

// Simulates a process crash: the writer is intentionally leaked so its
// destructor (which would remove the unpublished generation) never runs.
ForwardDrizzleV2StoreWriter *v2_crashed_writer(const fs::path &root,
                                               ForwardDrizzleV2RunPlan plan,
                                               int bands_to_commit) {
  auto *writer = new ForwardDrizzleV2StoreWriter(root, std::move(plan));
  writer->begin();
  int y = 0;
  for (int i = 0; i < bands_to_commit; ++i) {
    const int rows =
        std::min(writer->plan().band_rows, writer->plan().native_height - y);
    writer->commit_band(i, y, rows,
                        v2_band_records(rows, writer->plan().native_width,
                                        writer->plan().channels, 100.0 + i),
                        {}, 7 + i);
    y += rows;
  }
  return writer;
}

void v2_commit_all_bands(ForwardDrizzleV2StoreWriter &writer, int from_band,
                         double seed_base = 100.0) {
  int y = 0;
  for (int i = 0; i < from_band; ++i) {
    const auto &p = writer.plan();
    y += std::min(p.band_rows, p.native_height - y);
  }
  for (int i = from_band; i < writer.plan().band_count; ++i) {
    const auto &p = writer.plan();
    const int rows = std::min(p.band_rows, p.native_height - y);
    writer.commit_band(i, y, rows,
                       v2_band_records(rows, p.native_width, p.channels,
                                       seed_base + i),
                       {}, 7 + i);
    y += rows;
  }
}

ForwardDrizzleV2CommitGate v2_gate(std::uint64_t bands,
                                 std::uint64_t nonfinite = 0) {
  ForwardDrizzleV2CommitGate g;
  g.bands_processed = bands;
  g.nonfinite_pixels_inside_source_support = nonfinite;
  g.telemetry_json = "{\"frames_processed\":5}";
  return g;
}

std::string v2_file_text(const fs::path &path) {
  std::ifstream f(path);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

fs::path v2_single_generation(const fs::path &root) {
  for (const auto &e : fs::directory_iterator(root))
    if (e.is_directory() &&
        e.path().filename().string().rfind("forward_drizzle_v2_generation-",
                                           0) == 0)
      return e.path();
  return {};
}

}  // namespace

TEST_CASE("forward drizzle v2 store publishes only fully committed bands",
          "[forward-drizzle-v2][gate7]") {
  V2StoreFixture fx;
  const auto plan = v2_store_plan();

  {
    ForwardDrizzleV2StoreWriter writer(fx.root, plan);
    writer.begin();
    v2_commit_all_bands(writer, 0);
    REQUIRE(writer.committed_bands() == plan.band_count);
    REQUIRE_NOTHROW(writer.finish(v2_gate(plan.band_count)));
    REQUIRE(writer.published());
  }
  REQUIRE(fs::exists(fx.root / "current.json"));

  const auto inspection = inspect_forward_drizzle_v2_store(fx.root, plan);
  REQUIRE(inspection.status == ForwardDrizzleV2StoreStatus::complete);
  REQUIRE(!inspection.commit_hash.empty());

  // Band records round-trip byte-identically.
  ForwardDrizzleV2Checkpoint cp;
  std::string error;
  REQUIRE(parse_forward_drizzle_v2_checkpoint(
      v2_file_text(inspection.generation / "checkpoint.json"), cp, error));
  REQUIRE(cp.bands.size() == static_cast<std::size_t>(plan.band_count));
  const auto records = read_forward_drizzle_v2_band(inspection.generation,
                                                    cp.bands.front());
  const auto expected = v2_band_records(plan.band_rows, plan.native_width,
                                        plan.channels, 100.0);
  REQUIRE(records.size() == expected.size());
  REQUIRE(std::memcmp(records.data(), expected.data(),
                      records.size() * sizeof(records[0])) == 0);

  // Writer contract violations are rejected before any mutation.
  ForwardDrizzleV2StoreWriter bad(fx.root / "second", plan);
  bad.begin();
  REQUIRE_THROWS(bad.begin());
  REQUIRE_THROWS(bad.commit_band(1, 0, plan.band_rows,
                                 v2_band_records(plan.band_rows,
                                                 plan.native_width,
                                                 plan.channels, 0.0),
                                 {}, 0));
  REQUIRE_THROWS(bad.finish(v2_gate(plan.band_count)));
}

TEST_CASE("forward drizzle v2 store resume replays only the missing tail",
          "[forward-drizzle-v2][gate7]") {
  V2StoreFixture crashed;
  V2StoreFixture reference;

  // Reference: uninterrupted run.
  std::string reference_checkpoint_hash;
  {
    ForwardDrizzleV2StoreWriter writer(reference.root, v2_store_plan());
    writer.begin();
    v2_commit_all_bands(writer, 0);
    writer.finish(v2_gate(writer.plan().band_count));
    nlohmann::json cp = nlohmann::json::parse(v2_file_text(
        v2_single_generation(reference.root) / "checkpoint.json"));
    reference_checkpoint_hash = cp.at("checkpoint_hash").get<std::string>();
  }

  // Crashed run: band 0 committed, process died before band 1.
  auto *leaked = v2_crashed_writer(crashed.root, v2_store_plan(), 1);
  (void)leaked;
  const fs::path generation = v2_single_generation(crashed.root);
  REQUIRE(!generation.empty());

  // Residue of the interrupted band-1 write and a stale stage directory
  // must be ignored: only checkpoint.json establishes resumability.
  {
    std::ofstream residue(generation / "band-0001.bin", std::ios::binary);
    residue << "garbage";
    fs::create_directories(generation / "band-0002.bin.stage-1");
  }

  auto inspection = inspect_forward_drizzle_v2_store(crashed.root,
                                                     v2_store_plan());
  REQUIRE(inspection.status == ForwardDrizzleV2StoreStatus::resumable);
  REQUIRE(inspection.next_band == 1);
  REQUIRE(inspection.committed.size() == 1);
  REQUIRE(inspection.generation == generation);

  {
    ForwardDrizzleV2StoreWriter resumed(crashed.root, v2_store_plan());
    resumed.adopt(inspection.generation, inspection.next_band,
                  inspection.committed);
    REQUIRE(resumed.committed_bands() == 1);
    // The residue band-0001.bin is overwritten by the atomic rename.
    v2_commit_all_bands(resumed, 1);
    resumed.finish(v2_gate(resumed.plan().band_count));
  }

  const auto done =
      inspect_forward_drizzle_v2_store(crashed.root, v2_store_plan());
  REQUIRE(done.status == ForwardDrizzleV2StoreStatus::complete);

  // Determinism: identical inputs produce identical band artifacts and the
  // same checkpoint hash as the uninterrupted run.
  nlohmann::json cp = nlohmann::json::parse(
      v2_file_text(done.generation / "checkpoint.json"));
  REQUIRE(cp.at("checkpoint_hash").get<std::string>() ==
          reference_checkpoint_hash);
  REQUIRE(cp.at("plan_hash").get<std::string>() ==
          v2_store_plan().plan_hash);
  for (const auto &band : cp.at("bands")) {
    const auto records =
        read_forward_drizzle_v2_band(done.generation,
                                     [&band] {
                                       ForwardDrizzleV2BandCommit b;
                                       b.band_index = band.at("band_index");
                                       b.y_begin = band.at("y_begin");
                                       b.rows = band.at("rows");
                                       b.native_cols = band.at("native_cols");
                                       b.channels = band.at("channels");
                                       b.dense_overlap_count =
                                           band.at("dense_overlap_count");
                                       b.artifact = band.at("artifact");
                                       b.bytes = band.at("bytes");
                                       b.sha256 = band.at("sha256");
                                       return b;
                                     }());
    const int i = band.at("band_index").get<int>();
    const auto expected =
        v2_band_records(band.at("rows").get<int>(),
                        band.at("native_cols").get<int>(),
                        band.at("channels").get<int>(), 100.0 + i);
    REQUIRE(std::memcmp(records.data(), expected.data(),
                        records.size() * sizeof(records[0])) == 0);
  }
}

TEST_CASE("forward drizzle v2 store fails closed on unbound or corrupt state",
          "[forward-drizzle-v2][gate7]") {
  const auto expect_corrupt = [](const fs::path &root,
                                 const ForwardDrizzleV2RunPlan &plan) {
    const auto s = inspect_forward_drizzle_v2_store(root, plan);
    REQUIRE(s.status == ForwardDrizzleV2StoreStatus::corrupt);
    REQUIRE(!s.error.empty());
    return s.error;
  };

  SECTION("corrupt band artifact content") {
    V2StoreFixture fx;
    v2_crashed_writer(fx.root, v2_store_plan(), 1);
    const auto gen = v2_single_generation(fx.root);
    std::fstream f(gen / "band-0000.bin",
                   std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(80);
    f.put('X');
    f.close();
    expect_corrupt(fx.root, v2_store_plan());
  }
  SECTION("truncated band artifact") {
    V2StoreFixture fx;
    v2_crashed_writer(fx.root, v2_store_plan(), 1);
    const auto gen = v2_single_generation(fx.root);
    fs::resize_file(gen / "band-0000.bin", 40);
    expect_corrupt(fx.root, v2_store_plan());
  }
  SECTION("checkpoint lists a missing artifact") {
    V2StoreFixture fx;
    v2_crashed_writer(fx.root, v2_store_plan(), 1);
    const auto gen = v2_single_generation(fx.root);
    fs::remove(gen / "band-0000.bin");
    expect_corrupt(fx.root, v2_store_plan());
  }
  SECTION("non-contiguous checkpoint") {
    V2StoreFixture fx;
    v2_crashed_writer(fx.root, v2_store_plan(), 2);
    const auto gen = v2_single_generation(fx.root);
    nlohmann::json cp =
        nlohmann::json::parse(v2_file_text(gen / "checkpoint.json"));
    cp["bands"].erase(0);  // prefix now starts at index 1
    cp.erase("checkpoint_hash");
    const auto text = cp.dump();
    cp["checkpoint_hash"] = core::sha256_bytes(
        std::vector<std::uint8_t>(text.begin(), text.end()));
    core::write_text_atomic(gen / "checkpoint.json", cp.dump(2));
    expect_corrupt(fx.root, v2_store_plan());
  }
  SECTION("checkpoint plan hash mismatch") {
    V2StoreFixture fx;
    v2_crashed_writer(fx.root, v2_store_plan(), 1);
    const auto gen = v2_single_generation(fx.root);
    nlohmann::json cp =
        nlohmann::json::parse(v2_file_text(gen / "checkpoint.json"));
    cp["plan_hash"] = "different";
    cp.erase("checkpoint_hash");
    const auto text = cp.dump();
    cp["checkpoint_hash"] = core::sha256_bytes(
        std::vector<std::uint8_t>(text.begin(), text.end()));
    core::write_text_atomic(gen / "checkpoint.json", cp.dump(2));
    expect_corrupt(fx.root, v2_store_plan());
  }
  SECTION("checkpoint tampered without hash repair") {
    V2StoreFixture fx;
    v2_crashed_writer(fx.root, v2_store_plan(), 1);
    const auto gen = v2_single_generation(fx.root);
    nlohmann::json cp =
        nlohmann::json::parse(v2_file_text(gen / "checkpoint.json"));
    cp["band_count"] = 99;
    core::write_text_atomic(gen / "checkpoint.json", cp.dump(2));
    expect_corrupt(fx.root, v2_store_plan());
  }
  SECTION("tampered plan.json") {
    V2StoreFixture fx;
    v2_crashed_writer(fx.root, v2_store_plan(), 1);
    const auto gen = v2_single_generation(fx.root);
    nlohmann::json p = nlohmann::json::parse(v2_file_text(gen / "plan.json"));
    p["pixfrac"] = 0.1;
    core::write_text_atomic(gen / "plan.json", p.dump(2));
    expect_corrupt(fx.root, v2_store_plan());
  }
  SECTION("plan schema_version mismatch") {
    V2StoreFixture fx;
    v2_crashed_writer(fx.root, v2_store_plan(), 1);
    const auto gen = v2_single_generation(fx.root);
    nlohmann::json p = nlohmann::json::parse(v2_file_text(gen / "plan.json"));
    p["schema_version"] = 2;
    p.erase("plan_hash");
    const auto text = p.dump();
    p["plan_hash"] = core::sha256_bytes(
        std::vector<std::uint8_t>(text.begin(), text.end()));
    core::write_text_atomic(gen / "plan.json", p.dump(2));
    expect_corrupt(fx.root, v2_store_plan());
  }
  SECTION("expected plan differs from stored context") {
    V2StoreFixture fx;
    v2_crashed_writer(fx.root, v2_store_plan(), 1);
    auto other = v2_store_plan();
    other.config_snapshot_hash = "other-config";
    finalize_forward_drizzle_v2_run_plan(other);
    expect_corrupt(fx.root, other);
  }
  SECTION("ambiguous multiple generations") {
    V2StoreFixture fx;
    v2_crashed_writer(fx.root, v2_store_plan(), 1);
    fs::create_directories(fx.root / "forward_drizzle_v2_generation-x");
    expect_corrupt(fx.root, v2_store_plan());
  }
  SECTION("current.json points at missing generation") {
    V2StoreFixture fx;
    {
      ForwardDrizzleV2StoreWriter writer(fx.root, v2_store_plan());
      writer.begin();
      v2_commit_all_bands(writer, 0);
      writer.finish(v2_gate(writer.plan().band_count));
    }
    fs::remove_all(v2_single_generation(fx.root));
    expect_corrupt(fx.root, v2_store_plan());
  }
  SECTION("unpublished writer removes its generation") {
    V2StoreFixture fx;
    fs::path generation;
    {
      ForwardDrizzleV2StoreWriter writer(fx.root, v2_store_plan());
      writer.begin();
      writer.commit_band(0, 0, writer.plan().band_rows,
                         v2_band_records(writer.plan().band_rows,
                                         writer.plan().native_width,
                                         writer.plan().channels, 0.0),
                         {}, 0);
      generation = writer.generation();
    }
    REQUIRE(!fs::exists(generation));
    const auto s = inspect_forward_drizzle_v2_store(fx.root, v2_store_plan());
    REQUIRE(s.status == ForwardDrizzleV2StoreStatus::fresh);
  }
  SECTION("commit gate blocks publish") {
    V2StoreFixture fx;
    fs::path generation;
    {
      ForwardDrizzleV2StoreWriter writer(fx.root, v2_store_plan());
      writer.begin();
      v2_commit_all_bands(writer, 0);
      generation = writer.generation();
      REQUIRE_THROWS(writer.finish(v2_gate(writer.plan().band_count,
                                           /*nonfinite=*/1)));
      REQUIRE_THROWS(writer.finish(v2_gate(writer.plan().band_count - 1)));
    }
    REQUIRE(!fs::exists(generation));
    REQUIRE(!fs::exists(fx.root / "current.json"));
  }
}

// --- Gate 8: local warp -----------------------------------------------------

namespace {

registration::SmoothLocalWarpModel
v2_local_model(int rows, int cols, const float cx[16], const float cy[16]) {
  registration::SmoothLocalWarpModel m;
  m.valid = true;
  m.image_rows = rows;
  m.image_cols = cols;
  for (int i = 0; i < 16; ++i) {
    m.coeff_x[i] = cx[i];
    m.coeff_y[i] = cy[i];
  }
  return m;
}

FrameSamplingTransform v2_local_frame(const FrameSamplingTransform &base,
                                      const registration::SmoothLocalWarpModel &m,
                                      float coord_scale = 1.0f,
                                      float off_x = 0.0f, float off_y = 0.0f) {
  FrameSamplingTransform f = base;
  f.has_smooth_local_model = true;
  f.smooth_local_model = m;
  f.model_coordinate_scale = coord_scale;
  f.model_offset_x = off_x;
  f.model_offset_y = off_y;
  return f;
}

ForwardDrizzleV2LocalWarp
v2_warp_descriptor(const FrameSamplingTransform &f,
                   const ForwardDrizzleSubdivisionParams &sub = {}) {
  ForwardDrizzleV2LocalWarp w;
  const auto &m = f.smooth_local_model;
  for (int i = 0; i < 16; ++i) {
    w.coeff_x[i] = m.coeff_x[i];
    w.coeff_y[i] = m.coeff_y[i];
  }
  w.image_rows = m.image_rows;
  w.image_cols = m.image_cols;
  w.model_valid = m.valid ? 1 : 0;
  w.model_coordinate_scale = f.model_coordinate_scale;
  w.model_offset_x = f.model_offset_x;
  w.model_offset_y = f.model_offset_y;
  w.position_epsilon_internal_px = sub.position_epsilon_internal_px;
  w.max_subdivision_depth = sub.max_subdivision_depth;
  w.area_relative_epsilon = sub.area_relative_epsilon;
  return w;
}

// CPU leaf-replay oracle for one frame: sample_leaves plus the production
// bbox clamp and polygon_rectangle_intersection_area accumulation into
// channel-major internal planes --- the exact semantics k_scatter_v2_local
// reproduces on device. `s2w` accumulates area*sigma2 over finite samples.
struct Gate8FramePlanes {
  std::vector<double> a, bs, bg, s2w;
  std::uint64_t overlaps = 0;
  std::uint64_t discarded = 0;
};

Gate8FramePlanes v2_frame_planes_cpu(
    const RegistrationSamplingPlan &plan, const FrameSamplingTransform &frame,
    const Matrix2Df &img, int scale, float pixfrac, int channels,
    BayerPattern pattern, int ox, int oy,
    const ForwardDrizzleSubdivisionParams &sub = {},
    const float *sigma2 = nullptr) {
  const int ic = plan.canvas_width_native * scale;
  const int ir = plan.canvas_height_native * scale;
  const std::size_t iplane = static_cast<std::size_t>(ic) * ir;
  Gate8FramePlanes p;
  p.a.assign(channels * iplane, 0.0);
  p.bs.assign(channels * iplane, 0.0);
  p.bg.assign(channels * iplane, 0.0);
  if (sigma2 != nullptr) p.s2w.assign(channels * iplane, 0.0);
  std::vector<Leaf> leaves;
  for (int sy = 0; sy < plan.source_height; ++sy)
    for (int sx = 0; sx < plan.source_width; ++sx) {
      if (!sample_leaves(plan, frame, sx, sy, scale, pixfrac, sub, leaves)) {
        ++p.discarded;
        continue;
      }
      const float v = img(sy, sx);
      const bool finite = std::isfinite(v);
      const double s2 =
          sigma2 != nullptr
              ? static_cast<double>(
                    sigma2[static_cast<std::size_t>(sy) * plan.source_width +
                           sx])
              : 0.0;
      const int c =
          channels == 1
              ? 0
              : static_cast<int>(cfa_channel_for_source_pixel(
                    sx, sy, pattern, ox, oy));
      for (const auto &leaf : leaves) {
        const double minx =
            *std::min_element(leaf.x, leaf.x + 4);
        const double maxx =
            *std::max_element(leaf.x, leaf.x + 4);
        const double miny =
            *std::min_element(leaf.y, leaf.y + 4);
        const double maxy =
            *std::max_element(leaf.y, leaf.y + 4);
        const int x0 = static_cast<int>(
            std::clamp(std::floor(minx), 0.0, static_cast<double>(ic)));
        const int x1 = static_cast<int>(
            std::clamp(std::ceil(maxx), 0.0, static_cast<double>(ic)));
        const int y0 = static_cast<int>(
            std::clamp(std::floor(miny), 0.0, static_cast<double>(ir)));
        const int y1 = static_cast<int>(
            std::clamp(std::ceil(maxy), 0.0, static_cast<double>(ir)));
        for (int ty = y0; ty < y1; ++ty)
          for (int tx = x0; tx < x1; ++tx) {
            const double k = polygon_rectangle_intersection_area(
                leaf.x, leaf.y, tx, ty, tx + 1.0, ty + 1.0);
            if (!(k > 0.0)) continue;
            const std::size_t o =
                static_cast<std::size_t>(c) * iplane +
                static_cast<std::size_t>(ty) * ic + tx;
            p.bg[o] += k;
            if (finite) {
              p.a[o] += k * v;
              p.bs[o] += k;
              if (sigma2 != nullptr) p.s2w[o] += k * s2;
            }
            ++p.overlaps;
          }
      }
    }
  return p;
}

// fp32 inversion internals quantize leaf corners (~1e-7 relative); decisions
// and support masks must be exact, magnitudes within a small margin.
void require_gate8_plane_parity(const Gate8FramePlanes &ref,
                                const std::vector<double> &da,
                                const std::vector<double> &dbs,
                                const std::vector<double> &dbg, double tol) {
  REQUIRE(da.size() == ref.a.size());
  REQUIRE(dbs.size() == ref.bs.size());
  REQUIRE(dbg.size() == ref.bg.size());
  for (std::size_t i = 0; i < ref.a.size(); ++i) {
    REQUIRE((dbg[i] > 0.0) == (ref.bg[i] > 0.0));
    REQUIRE((dbs[i] > 0.0) == (ref.bs[i] > 0.0));
    REQUIRE(da[i] == Catch::Approx(ref.a[i]).margin(tol));
    REQUIRE(dbs[i] == Catch::Approx(ref.bs[i]).margin(tol));
    REQUIRE(dbg[i] == Catch::Approx(ref.bg[i]).margin(tol));
  }
}

struct Gate8DevicePlanes {
  std::vector<double> a, bs, bg;
  unsigned long long overlaps = 0;
  unsigned long long discarded = 0;
};

Gate8DevicePlanes v2_local_scatter_device(
    const double a6[6], const ForwardDrizzleV2LocalWarp &w, int scale,
    float pixfrac, int ic, int ir, const Matrix2Df &img, int channels,
    BayerPattern pattern, int ox, int oy, int canvas_w, int canvas_h) {
  Gate8DevicePlanes p;
  const std::size_t n = static_cast<std::size_t>(channels) * ic * ir;
  p.a.assign(n, -1.0);
  p.bs.assign(n, -1.0);
  p.bg.assign(n, -1.0);
  REQUIRE(forward_drizzle_cuda_local_dense_scatter(
      a6, w, scale, 0.5 * static_cast<double>(pixfrac), ic, ir,
      static_cast<int>(img.cols()), static_cast<int>(img.rows()), img.data(),
      static_cast<int>(pattern), ox, oy, channels == 1, canvas_w, canvas_h,
      p.a.data(), p.bs.data(), p.bg.data(), &p.overlaps, &p.discarded));
  return p;
}

// Gate-8 variant of require_gate6_parity: the device reproduces the CPU
// oracle's fp32 inversion internals only up to expf/FMA contraction
// differences (~few ulp on the corner positions), so magnitudes are
// checked at a documented tolerance while every discrete decision
// (states, contributor counts, masks, discards) stays exact.
void require_gate8_parity(const Gate6CpuRef &ref,
                          const std::vector<ForwardDrizzleV2PixelResult> &gpu,
                          int ncols, int nrows, int channels) {
  const std::size_t nplane = static_cast<std::size_t>(ncols) * nrows;
  for (std::size_t pc = 0; pc < nplane * channels; ++pc) {
    const auto &e = ref.results[pc];
    const auto &g = gpu[pc];
    REQUIRE(g.robust_state == static_cast<std::uint8_t>(e.state));
    REQUIRE(g.confidence_state == static_cast<std::uint8_t>(e.conf_state));
    REQUIRE(g.contributors == e.candidates);
    REQUIRE(g.conf_degraded == e.conf_degraded);
    REQUIRE(g.value == Catch::Approx(e.value).epsilon(1e-5).margin(1e-9));
    REQUIRE(g.b == Catch::Approx(e.b).epsilon(1e-5).margin(1e-9));
    REQUIRE(g.n_eff == Catch::Approx(e.n_eff).epsilon(1e-5).margin(1e-9));
    REQUIRE(g.confidence ==
            Catch::Approx(e.confidence).epsilon(1e-5).margin(1e-9));
  }
}

// Full-band CPU reference for the mixed affine+local stream: identical to
// gate6_cpu_reference but every frame's planes come from the leaf replay,
// which covers both affine (build_affine_leaf) and local (subdivide_local)
// frames with the same fold/reservoir semantics downstream.
Gate6CpuRef gate8_cpu_reference(
    const Fixture &f, const std::vector<float> &sigma2_or_empty,
    const ForwardDrizzleV2KernelConfig &kcfg,
    const ForwardDrizzleSubdivisionParams &sub = {}) {
  const int scale = kcfg.internal_scale;
  const int nc = f.plan.canvas_width_native;
  const int nr = f.plan.canvas_height_native;
  const int ic = nc * scale;
  const int ir = nr * scale;
  const int channels = kcfg.mono ? 1 : 3;
  const std::size_t nplane = static_cast<std::size_t>(nc) * nr;
  const std::size_t iplane = static_cast<std::size_t>(ic) * ir;
  const double inv_s2 = 1.0 / (static_cast<double>(scale) * scale);
  const std::size_t n_frames = f.plan.frames.size();

  Gate6CpuRef ref;
  ref.results.assign(nplane * channels, ForwardDrizzleV2RobustResult{});
  ref.masks.assign(nplane * channels, 0);
  ref.footprint.assign(nplane, 0);
  std::vector<std::vector<ForwardDrizzleV2RobustCandidate>> cands(
      nplane * channels);
  std::vector<std::vector<double>> s2v(nplane * channels);

  const std::size_t n_src = static_cast<std::size_t>(f.plan.source_width) *
                            f.plan.source_height;
  for (std::size_t fr = 0; fr < n_frames; ++fr) {
    const float *s2 = sigma2_or_empty.empty() ? nullptr
                                              : sigma2_or_empty.data();
    const auto planes = v2_frame_planes_cpu(
        f.plan, f.plan.frames[fr], f.images[fr], scale, f.cfg.pixfrac,
        channels, f.plan.bayer_pattern, f.plan.cfa_origin_x,
        f.plan.cfa_origin_y, sub, s2);
    (void)n_src;
    for (std::size_t px = 0; px < nplane; ++px) {
      const int nx = static_cast<int>(px % nc);
      const int ny = static_cast<int>(px / nc);
      bool any_geo = false;
      for (int c = 0; c < channels; ++c) {
        const std::size_t pc = static_cast<std::size_t>(c) * nplane + px;
        double a = 0.0, bs = 0.0, bg = 0.0, sw = 0.0;
        unsigned int geo_bits = 0, src_bits = 0;
        for (int iy = 0; iy < scale; ++iy)
          for (int ix = 0; ix < scale; ++ix) {
            const int j = iy * scale + ix;
            const std::size_t ii =
                static_cast<std::size_t>(c) * iplane +
                static_cast<std::size_t>(ny * scale + iy) * ic + nx * scale +
                ix;
            a += inv_s2 * planes.a[ii];
            bs += inv_s2 * planes.bs[ii];
            bg += inv_s2 * planes.bg[ii];
            if (sigma2_or_empty.empty() == false) sw += inv_s2 * planes.s2w[ii];
            if (planes.bg[ii] > 0.0) geo_bits |= 1u << j;
            if (planes.bs[ii] > 0.0) src_bits |= 1u << j;
          }
        ref.masks[pc] |=
            static_cast<unsigned short>(geo_bits | (src_bits << 4));
        if (bg > 0.0) any_geo = true;
        if (!(bs > 0.0)) continue;
        cands[pc].push_back({fr, a / bs, bs});
        if (!sigma2_or_empty.empty()) s2v[pc].push_back(sw / bs);
      }
      if (any_geo) ++ref.footprint[px];
    }
  }

  ForwardDrizzleV2RobustConfig rcfg;
  rcfg.reservoir_size = kcfg.reservoir_size;
  rcfg.reservoir_seed = kcfg.reservoir_seed;
  rcfg.oracle_min_clip_contributors = kcfg.min_clip_contributors;
  rcfg.oracle_passes = kcfg.robust_passes;
  rcfg.oracle_sigma_low = kcfg.sigma_low;
  rcfg.oracle_sigma_high = kcfg.sigma_high;
  rcfg.min_candidates = kcfg.min_candidates;
  for (std::size_t pc = 0; pc < nplane * channels; ++pc) {
    if (cands[pc].empty()) continue;
    ref.results[pc] = robust_reduce_candidates_v2(
        cands[pc], ForwardDrizzleV2Estimator::reservoir_sigma_clip, rcfg,
        static_cast<std::uint64_t>(n_frames), s2v[pc]);
  }
  for (std::size_t px = 0; px < nplane; ++px)
    if (ref.footprint[px] == n_frames && n_frames > 0) ++ref.dense_overlap;
  return ref;
}

}  // namespace

TEST_CASE("forward drizzle v2 gate8 local scatter matches the CPU oracle",
          "[forward-drizzle-v2][cuda-parity][gate8]") {
  if (!forward_drizzle_cuda_runtime_available()) {
    SUCCEED("CUDA device unavailable");
    return;
  }
  const double tol = 2.0e-3;
  for (int scale : {1, 2}) {
    auto f = make_fixture(ColorMode::OSC, BayerPattern::GBRG, 0, 0, scale);
    const int nc = f.plan.canvas_width_native;
    const int nr = f.plan.canvas_height_native;
    const int ic = nc * scale, ir = nr * scale;
    const auto &base = f.plan.frames[0];
    const auto &m = base.source_to_canvas;
    const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                          m(1, 0), m(1, 1), m(1, 2)};
    const auto run_case = [&](const FrameSamplingTransform &fr,
                              const ForwardDrizzleSubdivisionParams &sub =
                                  {}) {
      const auto ref = v2_frame_planes_cpu(
          f.plan, fr, f.images[0], scale, f.cfg.pixfrac, 3,
          f.plan.bayer_pattern, f.plan.cfa_origin_x, f.plan.cfa_origin_y, sub);
      const auto got =
          v2_local_scatter_device(a6, v2_warp_descriptor(fr, sub), scale,
                                  f.cfg.pixfrac, ic, ir, f.images[0], 3,
                                  f.plan.bayer_pattern, f.plan.cfa_origin_x,
                                  f.plan.cfa_origin_y, nc, nr);
      REQUIRE(got.discarded == ref.discarded);
      REQUIRE(got.overlaps == ref.overlaps);
      require_gate8_plane_parity(ref, got.a, got.bs, got.bg, tol);
      return ref;
    };

    SECTION("zero coefficients reproduce the affine droplet") {
      const float z[16] = {};
      const auto fr =
          v2_local_frame(base, v2_local_model(nr, nc, z, z));
      const auto ref = run_case(fr);
      REQUIRE(ref.discarded == 0);
      // The affine kernel computes the droplet in fp64 while the local
      // path --- like the production CPU oracle --- inverts in fp32. A
      // knife-edge boundary cell may therefore differ; the overlap counts
      // agree up to that boundary quantization.
      std::vector<double> aa(3 * static_cast<std::size_t>(ic) * ir),
          ab(aa.size());
      unsigned long long aov = 0;
      REQUIRE(forward_drizzle_cuda_affine_dense_scatter(
          a6, scale, 0.5 * static_cast<double>(f.cfg.pixfrac), 0, 0, ic, ir,
          f.plan.source_width, f.plan.source_height, f.images[0].data(),
          static_cast<int>(f.plan.bayer_pattern), f.plan.cfa_origin_x,
          f.plan.cfa_origin_y, false, aa.data(), ab.data(), &aov));
      REQUIRE(std::abs(static_cast<long long>(aov) -
                       static_cast<long long>(ref.overlaps)) <= 4);
    }

    SECTION("mild single bump") {
      float cx[16] = {}, cy[16] = {};
      cx[5] = 1.5f;   // one interior knot
      cy[9] = -1.0f;
      run_case(v2_local_frame(base, v2_local_model(nr, nc, cx, cy)));
    }

    SECTION("strong curvature forces depth-2 nodes") {
      float cx[16] = {}, cy[16] = {};
      // Checkerboard saddle on a compressed 4x4 model domain (knot
      // spacing ~4.3 render px). Calibrated amplitude: strong enough to
      // force depth-2 subdivision on some samples, with the rest failing
      // inversion/subdivision all-or-nothing (discard parity is asserted).
      cx[5] = 4.0f;
      cx[6] = -4.0f;
      cx[9] = -4.0f;
      cx[10] = 4.0f;
      cy[5] = -4.0f;
      cy[6] = 4.0f;
      cy[9] = 4.0f;
      cy[10] = -4.0f;
      const auto fr = v2_local_frame(base, v2_local_model(4, 4, cx, cy),
                                     3.0f / 13.0f);
      // Prove the case actually exercises depth-2 acceptance: some sample
      // must emit more leaves than the 4 depth-1 nodes alone could.
      std::vector<Leaf> leaves;
      std::size_t max_leaves = 0;
      for (int sy = 0; sy < f.plan.source_height; ++sy)
        for (int sx = 0; sx < f.plan.source_width; ++sx)
          if (sample_leaves(f.plan, fr, sx, sy, scale, f.cfg.pixfrac, {},
                            leaves))
            max_leaves = std::max(max_leaves, leaves.size());
      REQUIRE(max_leaves > 4);
      run_case(fr);
    }

    SECTION("boundary taper") {
      float cx[16] = {}, cy[16] = {};
      for (int i : {0, 1, 4, 5, 10, 11, 14, 15}) {
        cx[i] = 4.0f;
        cy[i] = -3.0f;
      }
      run_case(v2_local_frame(base, v2_local_model(nr, nc, cx, cy)));
    }

    SECTION("partial discards: max_iter too small to converge") {
      float cx[16] = {}, cy[16] = {};
      cx[5] = 0.8f;
      cx[6] = -0.6f;
      auto fr = v2_local_frame(base, v2_local_model(nr, nc, cx, cy));
      auto w = v2_warp_descriptor(fr);
      w.max_iter = 1;  // cannot converge wherever d != 0
      // No oracle parity here: sample_leaves always uses the default
      // LocalInversionParams, so only the discard accounting is checked.
      const auto got =
          v2_local_scatter_device(a6, w, scale, f.cfg.pixfrac, ic, ir,
                                  f.images[0], 3, f.plan.bayer_pattern,
                                  f.plan.cfa_origin_x, f.plan.cfa_origin_y,
                                  nc, nr);
      REQUIRE(got.discarded > 0);
      REQUIRE(got.discarded <=
              static_cast<std::uint64_t>(f.plan.source_width) *
                  f.plan.source_height);
    }

    SECTION("invalid model discards every sample") {
      float cx[16] = {}, cy[16] = {};
      cx[5] = 1.0f;
      auto m = v2_local_model(nr, nc, cx, cy);
      m.valid = false;
      const auto fr = v2_local_frame(base, m);
      const auto got =
          v2_local_scatter_device(a6, v2_warp_descriptor(fr), scale,
                                  f.cfg.pixfrac, ic, ir, f.images[0], 3,
                                  f.plan.bayer_pattern, f.plan.cfa_origin_x,
                                  f.plan.cfa_origin_y, nc, nr);
      const std::uint64_t n_src = static_cast<std::uint64_t>(
          f.plan.source_width) * f.plan.source_height;
      REQUIRE(got.discarded == n_src);
      REQUIRE(got.overlaps == 0);
      for (double v : got.bg) REQUIRE(v == 0.0);
    }

    SECTION("nonfinite coefficient discards every sample") {
      float cx[16] = {}, cy[16] = {};
      cx[5] = std::numeric_limits<float>::quiet_NaN();
      const auto fr =
          v2_local_frame(base, v2_local_model(nr, nc, cx, cy));
      const auto got =
          v2_local_scatter_device(a6, v2_warp_descriptor(fr), scale,
                                  f.cfg.pixfrac, ic, ir, f.images[0], 3,
                                  f.plan.bayer_pattern, f.plan.cfa_origin_x,
                                  f.plan.cfa_origin_y, nc, nr);
      const std::uint64_t n_src = static_cast<std::uint64_t>(
          f.plan.source_width) * f.plan.source_height;
      REQUIRE(got.discarded == n_src);
      REQUIRE(got.overlaps == 0);
    }

    SECTION("subdivision depth beyond the tree rejects the call") {
      float cx[16] = {}, cy[16] = {};
      const auto fr =
          v2_local_frame(base, v2_local_model(nr, nc, cx, cy));
      auto w = v2_warp_descriptor(fr);
      w.max_subdivision_depth = 3;
      std::vector<double> dummy(3 * static_cast<std::size_t>(ic) * ir);
      REQUIRE_FALSE(forward_drizzle_cuda_local_dense_scatter(
          a6, w, scale, 0.4, ic, ir, f.plan.source_width,
          f.plan.source_height, f.images[0].data(),
          static_cast<int>(f.plan.bayer_pattern), f.plan.cfa_origin_x,
          f.plan.cfa_origin_y, false, nc, nr, dummy.data(), dummy.data(),
          dummy.data(), nullptr, nullptr));
    }

    SECTION("translated M42 affine seed plus mild model") {
      const double m42[6] = {0.9998640418052673, -0.016167480498552322,
                             2.6364097595214844, 0.01610037311911583,
                             0.9999631643295288, 2.2344169616699219};
      float cx[16] = {}, cy[16] = {};
      cx[5] = 1.2f;
      cx[10] = -0.9f;
      cy[6] = 0.7f;
      const auto fr =
          v2_local_frame(base, v2_local_model(nr, nc, cx, cy));
      // Swap the fixture affine for the M42 seed on both sides.
      auto fr2 = fr;
      fr2.source_to_canvas = affine(m42[0], m42[1], m42[2], m42[3], m42[4],
                                    m42[5]);
      const auto ref2 = v2_frame_planes_cpu(
          f.plan, fr2, f.images[0], scale, f.cfg.pixfrac, 3,
          f.plan.bayer_pattern, f.plan.cfa_origin_x, f.plan.cfa_origin_y);
      const auto got =
          v2_local_scatter_device(m42, v2_warp_descriptor(fr2), scale,
                                  f.cfg.pixfrac, ic, ir, f.images[0], 3,
                                  f.plan.bayer_pattern, f.plan.cfa_origin_x,
                                  f.plan.cfa_origin_y, nc, nr);
      REQUIRE(got.discarded == ref2.discarded);
      REQUIRE(got.overlaps == ref2.overlaps);
      require_gate8_plane_parity(ref2, got.a, got.bs, got.bg, tol);
    }

    SECTION("mono path maps every sample to channel 0") {
      auto mf = make_fixture(ColorMode::MONO, BayerPattern::RGGB, 0, 0, scale);
      const auto &mbase = mf.plan.frames[0];
      const auto &mm = mbase.source_to_canvas;
      const double ma6[6] = {mm(0, 0), mm(0, 1), mm(0, 2),
                             mm(1, 0), mm(1, 1), mm(1, 2)};
      float cx[16] = {}, cy[16] = {};
      cx[5] = 1.4f;
      cy[10] = 0.8f;
      const auto fr =
          v2_local_frame(mbase, v2_local_model(nr, nc, cx, cy));
      const auto ref = v2_frame_planes_cpu(
          mf.plan, fr, mf.images[0], scale, mf.cfg.pixfrac, 1,
          mf.plan.bayer_pattern, mf.plan.cfa_origin_x, mf.plan.cfa_origin_y);
      const auto got =
          v2_local_scatter_device(ma6, v2_warp_descriptor(fr), scale,
                                  mf.cfg.pixfrac, ic, ir, mf.images[0], 1,
                                  mf.plan.bayer_pattern, mf.plan.cfa_origin_x,
                                  mf.plan.cfa_origin_y, nc, nr);
      REQUIRE(got.discarded == ref.discarded);
      REQUIRE(got.overlaps == ref.overlaps);
      require_gate8_plane_parity(ref, got.a, got.bs, got.bg, tol);
    }
  }
}

TEST_CASE("forward drizzle v2 gate8 displacement bound is conservative",
          "[forward-drizzle-v2][gate8]") {
  float cx[16] = {}, cy[16] = {};
  cx[5] = 1.5f;
  cx[9] = -2.0f;
  cy[5] = 0.5f;
  cy[9] = 1.0f;
  const double bound = forward_drizzle_v2_local_displacement_bound(
      std::span<const float>(cx, 16), std::span<const float>(cy, 16));
  REQUIRE(bound == Catch::Approx(std::hypot(2.0, 1.0)));
  const auto model = v2_local_model(13, 14, cx, cy);
  // Dense sweep: |d_model(q)| <= bound everywhere the model is defined.
  double observed = 0.0;
  for (int iy = 0; iy <= 60; ++iy)
    for (int ix = 0; ix <= 60; ++ix) {
      const float mx = static_cast<float>(ix) * 13.0f / 60.0f;
      const float my = static_cast<float>(iy) * 12.0f / 60.0f;
      const auto d = registration::evaluate_smooth_local_displacement(
          model, mx, my);
      observed = std::max(observed,
                          std::hypot(static_cast<double>(d.x),
                                     static_cast<double>(d.y)));
    }
  REQUIRE(observed <= bound + 1e-6);
  // Invalid inputs fail closed: span-shape violations throw, nonfinite
  // coefficients yield a nonfinite bound.
  REQUIRE_THROWS_AS(forward_drizzle_v2_local_displacement_bound(
                        std::span<const float>(cx, 16),
                        std::span<const float>(cy, 8)),
                    std::invalid_argument);
  float bad[16] = {};
  bad[3] = std::numeric_limits<float>::quiet_NaN();
  REQUIRE(!std::isfinite(forward_drizzle_v2_local_displacement_bound(
      std::span<const float>(bad, 16), std::span<const float>(cy, 16))));
}

TEST_CASE("forward drizzle v2 gate8 mixed affine/local stream parity",
          "[forward-drizzle-v2][cuda-parity][gate8]") {
  if (!forward_drizzle_cuda_runtime_available()) {
    SUCCEED("CUDA device unavailable");
    return;
  }
  for (int scale : {1, 2}) {
    for (ColorMode mode : {ColorMode::MONO, ColorMode::OSC}) {
      auto f = make_fixture(mode, BayerPattern::GBRG, 0, 0, scale);
      const int nc = f.plan.canvas_width_native;
      const int nr = f.plan.canvas_height_native;
      const int channels = mode == ColorMode::MONO ? 1 : 3;
      const std::size_t nplane = static_cast<std::size_t>(nc) * nr;
      const std::size_t n_src = static_cast<std::size_t>(
          f.plan.source_width) * f.plan.source_height;
      std::vector<float> s2(n_src);
      for (std::size_t i = 0; i < n_src; ++i)
        s2[i] = 0.01f + 0.0001f * static_cast<float>(i % 97);

      // Interleave local models on frames 1 and 3.
      float cx1[16] = {}, cy1[16] = {}, cx3[16] = {}, cy3[16] = {};
      cx1[5] = 1.3f;
      cy1[6] = -0.8f;
      cx3[9] = -1.7f;
      cy3[10] = 1.1f;
      f.plan.frames[1] = v2_local_frame(
          f.plan.frames[1], v2_local_model(nr, nc, cx1, cy1));
      f.plan.frames[3] = v2_local_frame(
          f.plan.frames[3], v2_local_model(nr, nc, cx3, cy3));

      ForwardDrizzleV2KernelConfig kcfg;
      kcfg.internal_scale = scale;
      kcfg.stream_length = f.plan.frames.size();
      kcfg.half = 0.5 * f.cfg.pixfrac;
      kcfg.bayer_pattern = static_cast<int>(f.plan.bayer_pattern);
      kcfg.cfa_origin_x = f.plan.cfa_origin_x;
      kcfg.cfa_origin_y = f.plan.cfa_origin_y;
      kcfg.mono = mode == ColorMode::MONO;

      // Reference: same fold on the leaf-replay planes; also count oracle
      // discards for the stats check.
      const auto ref = gate8_cpu_reference(f, s2, kcfg);
      std::uint64_t expected_discards = 0;
      for (const auto &fr : f.plan.frames) {
        if (!fr.has_smooth_local_model) continue;
        expected_discards += v2_frame_planes_cpu(
                                 f.plan, fr, f.images[fr.source_index],
                                 scale, f.cfg.pixfrac, channels,
                                 f.plan.bayer_pattern, f.plan.cfa_origin_x,
                                 f.plan.cfa_origin_y)
                                 .discarded;
      }

      ForwardDrizzleV2CudaPrototypeKernel kernel;
      REQUIRE(kernel.reserve(nc, nr, f.plan.source_width,
                             f.plan.source_height, kcfg));
      for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
        const auto &frame = f.plan.frames[fr];
        const auto &m = frame.source_to_canvas;
        const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                              m(1, 0), m(1, 1), m(1, 2)};
        if (frame.has_smooth_local_model) {
          REQUIRE(kernel.accumulate_frame_local(
              a6, v2_warp_descriptor(frame), f.images[fr].data(),
              s2.data(), fr));
        } else {
          REQUIRE(kernel.accumulate_frame(a6, f.images[fr].data(),
                                          s2.data(), fr));
        }
      }
      std::vector<ForwardDrizzleV2PixelResult> got(nplane * channels);
      std::uint64_t dense = 0;
      REQUIRE(kernel.finalize(got.data(), nullptr, &dense));
      require_gate8_parity(ref, got, nc, nr, channels);
      REQUIRE(dense == ref.dense_overlap);
      REQUIRE(kernel.stats().local_samples_discarded == expected_discards);
      REQUIRE(kernel.stats().allocations == 1);
      REQUIRE(kernel.stats().stream_synchronizations == 1);
    }
  }
}

TEST_CASE("forward drizzle v2 gate8 local production geometry",
          "[.][forward-drizzle-v2-gate8-production]") {
  if (!forward_drizzle_cuda_runtime_available()) {
    SUCCEED("CUDA device unavailable");
    return;
  }
  constexpr char kSpecSha[] =
      "2fa20cd7e3b5293e391fa869b2972de8a9d8c3220935ade6a47c0ff8fbd6094b";
  constexpr int sw = 3840, sh = 2160;
  constexpr int nc = 3934, nr = 2270;
  constexpr double kMaxSteadySeconds = 0.75;

  const auto mem = forward_drizzle_cuda_device_memory();
  ForwardDrizzleV2MemoryInputs in;
  in.target_width = nc;
  in.target_height = nr;
  in.channels = 3;
  in.frame_count = 60;
  in.internal_scale = 2;
  in.source_device_slots = 2;
  in.source_slot_device_bytes = std::size_t{sw} * sh * sizeof(float);
  in.quality_device_slots = 1;
  in.quality_slot_device_bytes = std::size_t{sw} * sh * sizeof(float);
  in.device_reserve_bytes = std::size_t{256} << 20;
  in.device_budget_bytes =
      static_cast<std::size_t>(static_cast<double>(mem.free_bytes) * 0.8);
  in.host_budget_bytes = std::size_t{8} << 30;
  const auto plan = plan_forward_drizzle_v2_memory(in);
  REQUIRE(plan.feasible);
  REQUIRE(plan.tile_cols == nc);
  const int band_rows = plan.band_rows;

  Matrix2Df source(sh, sw);
  for (int y = 0; y < sh; ++y)
    for (int x = 0; x < sw; ++x)
      source(y, x) = static_cast<float>(
          100.0 + 0.001 * x + 0.002 * y + 0.01 * ((17 * x + 31 * y) % 29));

  const std::array<std::array<double, 6>, 5> base{{
      {1.0, 0.0, 32.0, 0.0, 1.0, 50.0},
      {0.9998640418052673, -0.016167480498552322, 35.636409759521484,
       0.01610037311911583, 0.9999631643295288, 46.23441696166992},
      {0.9998466372489929, 0.017386717721819878, 55.92152404785156,
       -0.01743965968489647, 0.9998936057090759, 68.48160552978516},
      {0.9999858140945435, -0.0024168870877474546, 31.47933578491211,
       0.0021460296120494604, 1.000150442123413, 56.29893493652344},
      {1.0000033378601074, -0.006749980617314577, 28.21784782409668,
       0.006637410260736942, 1.0002058744430542, 58.95621871948242},
  }};

  ForwardDrizzleV2KernelConfig kcfg;
  kcfg.internal_scale = 2;
  kcfg.stream_length = 60;
  kcfg.half = 0.4;
  kcfg.bayer_pattern = static_cast<int>(BayerPattern::GBRG);
  kcfg.mono = false;
  kcfg.canvas_width_native = nc;
  kcfg.canvas_height_native = nr;

  struct LocalCase {
    const char *name;
    registration::SmoothLocalWarpModel model;
    float coord_scale;
    float off_x, off_y;
  };
  // Model image in reference-image dimensions mapped onto the native
  // canvas: scale = (image_cols - 1) / canvas_cols, like production models
  // estimated in source-image coordinates.
  const float mscale = static_cast<float>(sw - 1) / static_cast<float>(nc);
  auto make_case = [&](const char *name, float amp, bool checker, int mr,
                       int mc, float cs, float ox = 0.0f, float oy = 0.0f) {
    float cx[16] = {}, cy[16] = {};
    if (checker) {
      cx[5] = amp;
      cx[6] = -amp;
      cx[9] = -amp;
      cx[10] = amp;
      cy[5] = -amp;
      cy[6] = amp;
      cy[9] = amp;
      cy[10] = -amp;
    } else {
      cx[5] = amp;
      cx[10] = -0.6f * amp;
      cy[6] = 0.7f * amp;
      cy[9] = -0.4f * amp;
    }
    return LocalCase{name, v2_local_model(mr, mc, cx, cy), cs, ox, oy};
  };
  const LocalCase cases[] = {
      // Realistic mild model on the full canvas (production-shaped).
      make_case("mild_realistic", 1.6f, false, sh, sw, mscale),
      // Inversion stress: displacement gradient near the contraction
      // limit forces the full max-iteration path per node.
      make_case("inversion_stress", 500.0f, true, sh, sw, mscale),
      // Compressed 13x13 native-px model domain inside the band
      // (canvas x in [1000,1013], y in [50,63]): the only way a 4x4
      // Gaussian model produces depth-2 subdivision on production
      // geometry, since canvas-fitted knot spacing is ~nc/3.
      make_case("depth2_patch", 4.0f, true, 4, 4, 3.0f / 13.0f, 1000.0f,
                50.0f),
  };

  for (const auto &lc : cases) {
    FrameSamplingTransform proto;
    proto.has_smooth_local_model = true;
    proto.smooth_local_model = lc.model;
    proto.model_coordinate_scale = lc.coord_scale;
    proto.model_offset_x = lc.off_x;
    proto.model_offset_y = lc.off_y;
    const auto warp = v2_warp_descriptor(proto);

    ForwardDrizzleV2CudaPrototypeKernel kernel;
    REQUIRE(kernel.reserve(nc, band_rows, sw, sh, kcfg));
    for (std::uint64_t f = 0; f < 60; ++f) {
      auto m = base[f % base.size()];
      m[2] += 0.05 * static_cast<double>(f);
      m[5] += 0.03 * static_cast<double>(f);
      REQUIRE(kernel.accumulate_frame_local(m.data(), warp, source.data(),
                                            nullptr, f));
    }
    const std::size_t n_out =
        static_cast<std::size_t>(nc) * band_rows * 3;
    std::vector<ForwardDrizzleV2PixelResult> results(n_out);
    std::uint64_t dense = 0;
    REQUIRE(kernel.finalize(results.data(), nullptr, &dense));
    const auto &st = kernel.stats();
    std::uint64_t supported = 0;
    for (const auto &r : results) {
      if (r.contributors == 0) continue;
      ++supported;
      REQUIRE(std::isfinite(r.value));
      REQUIRE(std::isfinite(r.b));
    }
    std::printf(
        "{\"gate\":8,\"spec_sha256\":\"%s\",\"case\":\"%s\","
        "\"native_cols\":%d,\"band_rows\":%d,\"frames\":60,"
        "\"reserved_bytes\":%llu,\"per_px_device\":%llu,"
        "\"per_px_plan\":%llu,\"max_frame_s\":%.6f,"
        "\"mean_frame_s\":%.6f,\"positive_overlaps\":%llu,"
        "\"discarded\":%llu,\"dense_overlap\":%llu,\"supported\":%llu,"
        "\"allocations\":%llu,\"global_syncs\":%llu,"
        "\"stream_syncs\":%llu}\n",
        kSpecSha, lc.name, nc, band_rows,
        static_cast<unsigned long long>(st.reserved_device_bytes),
        static_cast<unsigned long long>(
            kernel.device_bytes_per_native_pixel()),
        static_cast<unsigned long long>(plan.device_bytes_per_target_pixel),
        st.max_frame_seconds,
        (st.upload_seconds + st.kernel_seconds) / 60.0,
        st.positive_overlaps, st.local_samples_discarded,
        static_cast<unsigned long long>(dense),
        static_cast<unsigned long long>(supported),
        static_cast<unsigned long long>(st.allocations),
        static_cast<unsigned long long>(
            st.device_global_synchronizations),
        static_cast<unsigned long long>(st.stream_synchronizations));
    REQUIRE(supported > 0);
    REQUIRE(st.allocations == 1);
    REQUIRE(st.device_global_synchronizations == 0);
    REQUIRE(st.stream_synchronizations == 1);
    REQUIRE(st.frames_processed == 60);
    REQUIRE(st.max_frame_seconds <= kMaxSteadySeconds);
    REQUIRE(st.reserved_device_bytes <= plan.device_peak_bytes);
    REQUIRE(kernel.device_bytes_per_native_pixel() <=
            plan.device_bytes_per_target_pixel);
  }
}

TEST_CASE("forward drizzle v2 gate9 profile candidate fold",
          "[forward_drizzle_v2][gate9]") {
  using reconstruction::ForwardDrizzleV2QualityFold;
  const ForwardDrizzleV2QualityFold absent{};
  SECTION("present streams fold to area-weighted means") {
    const auto cd = reconstruction::forward_drizzle_v2_fold_profile_candidate(
        7, 8.0, 4.0, ForwardDrizzleV2QualityFold{2.0, 4.0, true},
        ForwardDrizzleV2QualityFold{1.0, 4.0, true},
        ForwardDrizzleV2QualityFold{3.0, 4.0, true},
        ForwardDrizzleV2QualityFold{1.2, 4.0, true});
    REQUIRE(cd.frame_order == 7);
    REQUIRE(cd.x == Catch::Approx(2.0));
    REQUIRE(cd.b == Catch::Approx(4.0));
    REQUIRE(cd.q == Catch::Approx(0.5));
    REQUIRE(cd.q0 == Catch::Approx(0.25));
    REQUIRE(cd.q1 == Catch::Approx(0.75));
    REQUIRE(cd.qa == Catch::Approx(0.3));
    REQUIRE(cd.qa_has_data);
  }
  SECTION("absent streams fold to 1.0 and artifact has no data") {
    const auto cd = reconstruction::forward_drizzle_v2_fold_profile_candidate(
        0, 5.0, 2.0, absent, absent, absent, absent);
    REQUIRE(cd.q == Catch::Approx(1.0));
    REQUIRE(cd.q0 == Catch::Approx(1.0));
    REQUIRE(cd.q1 == Catch::Approx(1.0));
    REQUIRE(cd.qa == Catch::Approx(1.0));
    REQUIRE_FALSE(cd.qa_has_data);
  }
  SECTION("artifact stream present but all samples nonfinite") {
    const auto cd = reconstruction::forward_drizzle_v2_fold_profile_candidate(
        0, 5.0, 2.0, absent, absent, absent,
        ForwardDrizzleV2QualityFold{0.0, 0.0, true});
    REQUIRE(cd.qa == Catch::Approx(0.0));
    REQUIRE_FALSE(cd.qa_has_data);
  }
  SECTION("zero geometric weight yields a zero-weight candidate") {
    const auto cd = reconstruction::forward_drizzle_v2_fold_profile_candidate(
        3, 0.0, 0.0, ForwardDrizzleV2QualityFold{1.0, 0.0, true}, absent,
        absent, absent);
    REQUIRE(cd.x == Catch::Approx(0.0));
    REQUIRE(cd.b == Catch::Approx(0.0));
    REQUIRE(cd.q == Catch::Approx(1.0));
  }
  SECTION("nonfinite accumulators throw") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(
        reconstruction::forward_drizzle_v2_fold_profile_candidate(
            0, nan, 1.0, absent, absent, absent, absent),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        reconstruction::forward_drizzle_v2_fold_profile_candidate(
            0, 1.0, -1.0, absent, absent, absent, absent),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        reconstruction::forward_drizzle_v2_fold_profile_candidate(
            0, 1.0, 1.0, ForwardDrizzleV2QualityFold{nan, 0.0, true}, absent,
            absent, absent),
        std::invalid_argument);
  }
}

TEST_CASE("forward drizzle v2 gate9 profile reduce weights and mask",
          "[forward_drizzle_v2][gate9]") {
  using reconstruction::ForwardDrizzleV2FrameMeta;
  using reconstruction::ForwardDrizzleV2ProfileCandidate;
  std::vector<ForwardDrizzleV2ProfileCandidate> cands = {
      {0, 10.0, 2.0, 1.0, 0.5, 0.8, 0.9, true},
      {1, 20.0, 1.0, 0.5, 1.0, 0.5, 0.4, true},
      {2, 999.0, 3.0, 1.0, 1.0, 1.0, 1.0, true}};
  std::vector<ForwardDrizzleV2FrameMeta> meta = {
      {2.0f, 0.9f, 1, 0}, {0.5f, 0.7f, 0, 0}, {1.0f, 1.0f, 1, 0}};

  SECTION("weights follow the profile contract on the accepted set") {
    std::vector<std::uint8_t> acc = {1, 1, 0};
    const auto r = reconstruction::forward_drizzle_v2_profile_reduce(
        cands, acc, meta);
    // uniform: (2*10 + 1*20) / 3 = 13.33; raw: (4*10 + 0.25*20)/4.25;
    // fine: (2*2*0.5^4 + 1*0.5*1^4) = 0.75;
    // medium: (2*2*0.8^2 + 1*0.5*0.5^2) = 2.685
    REQUIRE(r.uniform.value == Catch::Approx(40.0 / 3.0));
    REQUIRE(r.uniform.weight_sum == Catch::Approx(3.0));
    REQUIRE(r.raw.value == Catch::Approx(45.0 / 4.25).margin(1e-5));
    REQUIRE(r.raw.weight_sum == Catch::Approx(4.25));
    REQUIRE(r.fine.weight_sum == Catch::Approx(0.75).margin(1e-6));
    REQUIRE(r.medium.weight_sum == Catch::Approx(2.685).margin(1e-5));
    for (const auto *p : {&r.uniform, &r.raw, &r.fine, &r.medium}) {
      REQUIRE(p->support == 1);
      REQUIRE(p->n_eff > 0.0f);
    }
    // Rejected candidate 2 (x = 999) must not leak into any profile.
    for (const auto *p : {&r.uniform, &r.raw, &r.fine, &r.medium})
      REQUIRE(p->value < 100.0f);
  }

  SECTION("shared mask is external: high-q rejection removes the outlier") {
    std::vector<std::uint8_t> acc_with_outlier = {1, 1, 1};
    std::vector<std::uint8_t> acc_clipped = {1, 1, 0};
    const auto with_outlier =
        reconstruction::forward_drizzle_v2_profile_reduce(cands,
                                                          acc_with_outlier,
                                                          meta);
    const auto clipped = reconstruction::forward_drizzle_v2_profile_reduce(
        cands, acc_clipped, meta);
    REQUIRE(with_outlier.uniform.value > 100.0f);
    // Dropping the candidate by mask is bit-identical to dropping it from
    // the stream entirely.
    std::vector<ForwardDrizzleV2ProfileCandidate> two(cands.begin(),
                                                      cands.begin() + 2);
    std::vector<std::uint8_t> acc_two = {1, 1};
    const auto r2 = reconstruction::forward_drizzle_v2_profile_reduce(
        two, acc_two, meta);
    REQUIRE(clipped.uniform.value == r2.uniform.value);
    REQUIRE(clipped.raw.value == r2.raw.value);
    REQUIRE(clipped.fine.weight_sum == r2.fine.weight_sum);
  }

  SECTION("empty stream yields unsupported NaN profiles") {
    const auto r = reconstruction::forward_drizzle_v2_profile_reduce(
        {}, {}, meta);
    for (const auto *p : {&r.uniform, &r.raw, &r.fine, &r.medium}) {
      REQUIRE(p->support == 0);
      REQUIRE(std::isnan(p->value));
      REQUIRE(p->weight_sum == 0.0f);
    }
  }

  SECTION("zero-weight accepted candidates leave profiles unsupported") {
    std::vector<ForwardDrizzleV2ProfileCandidate> zero = {
        {0, 10.0, 0.0, 1.0, 1.0, 1.0, 1.0, true}};
    std::vector<std::uint8_t> acc = {1};
    const auto r = reconstruction::forward_drizzle_v2_profile_reduce(
        zero, acc, meta);
    REQUIRE(r.uniform.support == 0);
    REQUIRE(std::isnan(r.uniform.value));
  }

  SECTION("malformed inputs throw invalid_argument") {
    std::vector<std::uint8_t> acc = {1, 1, 0};
    std::vector<std::uint8_t> short_mask = {1};
    REQUIRE_THROWS_AS(
        reconstruction::forward_drizzle_v2_profile_reduce(cands, short_mask,
                                                        meta),
        std::invalid_argument);
    std::vector<ForwardDrizzleV2FrameMeta> short_meta = {meta[0]};
    REQUIRE_THROWS_AS(
        reconstruction::forward_drizzle_v2_profile_reduce(cands, acc,
                                                          short_meta),
        std::invalid_argument);
    std::vector<ForwardDrizzleV2FrameMeta> bad_g = meta;
    bad_g[0].g_eff = -1.0f;
    REQUIRE_THROWS_AS(
        reconstruction::forward_drizzle_v2_profile_reduce(cands, acc, bad_g),
        std::invalid_argument);
    bad_g = meta;
    bad_g[1].g_eff = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(
        reconstruction::forward_drizzle_v2_profile_reduce(cands, acc, bad_g),
        std::invalid_argument);
    std::vector<ForwardDrizzleV2ProfileCandidate> bad_cd = cands;
    bad_cd[0].x = std::numeric_limits<double>::infinity();
    REQUIRE_THROWS_AS(
        reconstruction::forward_drizzle_v2_profile_reduce(bad_cd, acc, meta),
        std::invalid_argument);
    bad_cd = cands;
    bad_cd[1].q = -0.5;
    REQUIRE_THROWS_AS(
        reconstruction::forward_drizzle_v2_profile_reduce(bad_cd, acc, meta),
        std::invalid_argument);
    bad_cd = cands;
    bad_cd[0].qa_has_data = true;
    bad_cd[0].qa = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(
        reconstruction::forward_drizzle_v2_profile_reduce(bad_cd, acc, meta),
        std::invalid_argument);
    // Rejected candidates are not validated: NaN in a clipped candidate ok.
    bad_cd = cands;
    bad_cd[2].x = std::numeric_limits<double>::quiet_NaN();
    bad_cd[2].b = -1.0;
    REQUIRE_NOTHROW(reconstruction::forward_drizzle_v2_profile_reduce(
        bad_cd, acc, meta));
  }
}

TEST_CASE("forward drizzle v2 gate9 alpha factors match the oracle",
          "[forward_drizzle_v2][gate9]") {
  using reconstruction::ForwardDrizzleV2FrameMeta;
  using reconstruction::ForwardDrizzleV2ProfileCandidate;
  std::vector<ForwardDrizzleV2ProfileCandidate> cands;
  std::vector<std::uint8_t> acc;
  std::vector<ForwardDrizzleV2FrameMeta> meta(12);
  std::vector<reconstruction::AlphaFactorContribution> oracle_contribs;
  for (std::size_t f = 0; f < 12; ++f) {
    const double q = 0.4 + 0.05 * static_cast<double>(f % 5);
    const double qa = 0.2 + 0.06 * static_cast<double>(f % 7);
    const double b = 1.0 + 0.1 * static_cast<double>(f % 3);
    cands.push_back({f, 5.0 + static_cast<double>(f), b, q, q, q, qa, true});
    acc.push_back(f == 11 ? 0 : 1);
    meta[f] = {1.0f, 0.6f + 0.03f * static_cast<float>(f),
               static_cast<std::uint8_t>(f % 2), 0};
    if (f != 11)
      oracle_contribs.push_back({b, q, qa, f % 2 != 0, meta[f].residual_factor});
  }
  const auto r = reconstruction::forward_drizzle_v2_profile_reduce(
      cands, acc, meta);
  const auto oracle = reconstruction::compute_alpha_confidence_channel(
      oracle_contribs);
  REQUIRE(r.a_separation == Catch::Approx(oracle.a_separation));
  REQUIRE(r.a_artifact == Catch::Approx(oracle.a_artifact));
  REQUIRE(r.a_registration == Catch::Approx(oracle.a_registration));
  REQUIRE(r.artifact_applicable == oracle.artifact_applicable);
  REQUIRE(r.artifact_applicable);

  SECTION("fewer artifact contributors than the minimum is non-applicable") {
    std::vector<ForwardDrizzleV2ProfileCandidate> few(cands.begin(),
                                                      cands.begin() + 4);
    std::vector<std::uint8_t> few_acc = {1, 1, 1, 1};
    const auto rf = reconstruction::forward_drizzle_v2_profile_reduce(
        few, few_acc, meta);
    REQUIRE_FALSE(rf.artifact_applicable);
    REQUIRE(rf.a_artifact == Catch::Approx(0.0));
  }

  SECTION("qa_has_data=false excludes the frame from the artifact pool") {
    std::vector<ForwardDrizzleV2ProfileCandidate> no_qa(cands.begin(),
                                                      cands.begin() + 11);
    for (auto &c : no_qa) c.qa_has_data = false;
    std::vector<std::uint8_t> a(11, 1);
    const auto rn = reconstruction::forward_drizzle_v2_profile_reduce(
        no_qa, a, meta);
    REQUIRE_FALSE(rn.artifact_applicable);
    REQUIRE(rn.a_artifact == Catch::Approx(0.0));
  }

  SECTION("emit_alpha_confidence=false leaves zero factors") {
    reconstruction::ForwardDrizzleV2ProfileConfig cfg;
    cfg.emit_alpha_confidence = false;
    const auto rd = reconstruction::forward_drizzle_v2_profile_reduce(
        cands, acc, meta, cfg);
    REQUIRE(rd.a_separation == 0.0f);
    REQUIRE(rd.a_artifact == 0.0f);
    REQUIRE(rd.a_registration == 0.0f);
    REQUIRE_FALSE(rd.artifact_applicable);
  }
}

TEST_CASE("forward drizzle v2 gate9 ring plan and streaming read amplification",
          "[forward_drizzle_v2][gate9]") {
  SECTION("ring plan matches the fusion halo contract") {
    for (int l = 1; l <= 4; ++l) {
      const auto p = reconstruction::forward_drizzle_v2_multiband_ring_plan(
          4540, 59, l);
      REQUIRE(p.levels == l);
      REQUIRE(p.halo_rows ==
              reconstruction::multiband_fusion_halo_rows(l));
      REQUIRE(p.band_core_rows == 59);
      REQUIRE(p.canvas_rows == 4540);
      // floor(2*halo/core) + 2
      REQUIRE(p.resident_bands == (2 * p.halo_rows) / 59 + 2);
    }
    const auto p4 = reconstruction::forward_drizzle_v2_multiband_ring_plan(
        4540, 59, 4);
    REQUIRE(p4.halo_rows == 64);
    REQUIRE(p4.resident_bands == 4);
  }
  SECTION("invalid ring arguments throw") {
    REQUIRE_THROWS_AS(
        reconstruction::forward_drizzle_v2_multiband_ring_plan(0, 59, 4),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        reconstruction::forward_drizzle_v2_multiband_ring_plan(100, 0, 4),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        reconstruction::forward_drizzle_v2_multiband_ring_plan(100, 10, 0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        reconstruction::forward_drizzle_v2_multiband_ring_plan(100, 10, 5),
        std::invalid_argument);
  }
  SECTION("reuse-aware read amplification") {
    // 4540 rows x 1 KB, halo 64, 37 bands (gate-5 production geometry).
    const auto ra = reconstruction::forward_drizzle_v2_streaming_ra(
        4540, 1024, 64, 37);
    REQUIRE(ra.logical_bytes == 4540ull * 1024);
    REQUIRE(ra.application_bytes_no_reuse ==
            4540ull * 1024 + 2ull * 37 * 64 * 1024);
    REQUIRE(ra.application_bytes_reuse ==
            4540ull * 1024 + 2ull * 64 * 1024);
    REQUIRE(ra.reused_bytes ==
            ra.application_bytes_no_reuse - ra.application_bytes_reuse);
    REQUIRE(ra.ra_no_reuse ==
            Catch::Approx(1.0 + 2.0 * 37 * 64.0 / 4540.0));
    REQUIRE(ra.ra_reuse == Catch::Approx(1.0 + 2.0 * 64.0 / 4540.0));
    REQUIRE(ra.ra_reuse < 1.03);
  }
  SECTION("degenerate and overflowing streaming inputs throw") {
    REQUIRE_THROWS_AS(
        reconstruction::forward_drizzle_v2_streaming_ra(0, 1024, 64, 37),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        reconstruction::forward_drizzle_v2_streaming_ra(4540, 0, 64, 37),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        reconstruction::forward_drizzle_v2_streaming_ra(4540, 1024, 64, 0),
        std::invalid_argument);
    const std::uint64_t huge = std::numeric_limits<std::uint64_t>::max();
    REQUIRE_THROWS_AS(
        reconstruction::forward_drizzle_v2_streaming_ra(huge, 1024, 64, 37),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        reconstruction::forward_drizzle_v2_streaming_ra(4540, 1024, huge, 2),
        std::invalid_argument);
  }
}

TEST_CASE("forward drizzle v2 gate9 profile adapter layout",
          "[forward_drizzle_v2][gate9]") {
  using reconstruction::ForwardDrizzleV2ProfileResult;
  constexpr int nc = 4, nr = 3;
  const std::size_t nplane = static_cast<std::size_t>(nc) * nr;

  SECTION("OSC maps R/G/B record planes and selects the profile") {
    std::vector<ForwardDrizzleV2ProfileResult> rec(3 * nplane);
    for (int c = 0; c < 3; ++c)
      for (std::size_t px = 0; px < nplane; ++px) {
        auto &r = rec[static_cast<std::size_t>(c) * nplane + px];
        r.uniform = {100.0f + static_cast<float>(10 * c + px), 2.0f, 1.5f, 1};
        r.raw = {200.0f + static_cast<float>(10 * c + px), 3.0f, 1.2f, 1};
        r.fine = {300.0f, 4.0f, 1.1f, 1};
        r.medium = {400.0f, 5.0f, 1.0f, 1};
      }
    const auto u = reconstruction::forward_drizzle_v2_profiles_to_uniform_result(
        rec, nc, nr, 3, ColorMode::OSC, 0);
    REQUIRE(u.color_mode == ColorMode::OSC);
    REQUIRE(u.internal_width == nc);
    REQUIRE(u.internal_height == nr);
    REQUIRE(u.R.value[0] == Catch::Approx(100.0));
    REQUIRE(u.G.value[0] == Catch::Approx(110.0));
    REQUIRE(u.B.value[nplane - 1] ==
            Catch::Approx(100.0 + 20 + nplane - 1));
    REQUIRE(u.G.support[5] == 1);
    REQUIRE(u.G.weight_sum[5] == Catch::Approx(2.0));
    REQUIRE(u.L.empty());
    const auto raw = reconstruction::forward_drizzle_v2_profiles_to_uniform_result(
        rec, nc, nr, 3, ColorMode::OSC, 1);
    REQUIRE(raw.R.value[0] == Catch::Approx(200.0));
    const auto med = reconstruction::forward_drizzle_v2_profiles_to_uniform_result(
        rec, nc, nr, 3, ColorMode::OSC, 3);
    REQUIRE(med.B.value[2] == Catch::Approx(400.0));
    REQUIRE(med.B.n_eff[2] == Catch::Approx(1.0));
  }

  SECTION("MONO maps slot 0 and preserves no-support NaN") {
    std::vector<ForwardDrizzleV2ProfileResult> rec(nplane);
    for (std::size_t px = 0; px < nplane; ++px)
      rec[px].uniform = {7.0f + static_cast<float>(px), 1.0f, 1.0f, 1};
    rec[3].uniform = {0.0f, 0.0f, 0.0f, 0};
    const auto m = reconstruction::forward_drizzle_v2_profiles_to_uniform_result(
        rec, nc, nr, 1, ColorMode::MONO, 0);
    REQUIRE(m.L.value[0] == Catch::Approx(7.0));
    REQUIRE(m.L.value[3] == Catch::Approx(0.0));
    REQUIRE(m.L.support[3] == 0);
    REQUIRE(m.R.empty());
    REQUIRE(m.G.empty());
    REQUIRE(m.B.empty());
  }

  SECTION("adapter rejects malformed geometry and sizes") {
    std::vector<ForwardDrizzleV2ProfileResult> rec(3 * nplane);
    REQUIRE_THROWS_AS(
        reconstruction::forward_drizzle_v2_profiles_to_uniform_result(
            rec, 0, nr, 3, ColorMode::OSC, 0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        reconstruction::forward_drizzle_v2_profiles_to_uniform_result(
            rec, nc, nr, 3, ColorMode::OSC, 4),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        reconstruction::forward_drizzle_v2_profiles_to_uniform_result(
            rec, nc, nr, 2, ColorMode::OSC, 0),
        std::invalid_argument);
    std::vector<ForwardDrizzleV2ProfileResult> short_rec(2 * nplane);
    REQUIRE_THROWS_AS(
        reconstruction::forward_drizzle_v2_profiles_to_uniform_result(
            short_rec, nc, nr, 3, ColorMode::OSC, 0),
        std::invalid_argument);
    std::vector<ForwardDrizzleV2ProfileResult> mono_rec(nplane);
    REQUIRE_THROWS_AS(
        reconstruction::forward_drizzle_v2_profiles_to_uniform_result(
            mono_rec, nc, nr, 3, ColorMode::MONO, 0),
        std::invalid_argument);
  }
}

namespace {

// Gate-9 clip-mask oracle: replicates robust_frame_oracle_v2's accepted-mask
// logic (valid filter, (x, order) sort, iterative weighted-MAD clip) so the
// test can feed the shared mask into forward_drizzle_v2_profile_reduce.
std::vector<std::uint8_t> g9_clip_mask(
    const std::vector<ForwardDrizzleV2RobustCandidate> &cands,
    int min_clip, int passes, double s_low, double s_high) {
  std::vector<std::size_t> valid;
  for (std::size_t i = 0; i < cands.size(); ++i)
    if (std::isfinite(cands[i].x) && std::isfinite(cands[i].b) &&
        cands[i].b > 0.0)
      valid.push_back(i);
  std::vector<std::uint8_t> mask(cands.size(), std::uint8_t{0});
  for (std::size_t i : valid) mask[i] = 1;
  if (valid.size() < static_cast<std::size_t>(min_clip)) return mask;
  std::vector<std::size_t> order = valid;
  std::sort(order.begin(), order.end(), [&](std::size_t i, std::size_t j) {
    if (cands[i].x != cands[j].x) return cands[i].x < cands[j].x;
    return cands[i].frame_order < cands[j].frame_order;
  });
  for (int pass = 0; pass < passes; ++pass) {
    std::vector<std::size_t> active;
    for (std::size_t idx : order)
      if (mask[idx]) active.push_back(idx);
    if (active.empty()) break;
    double total_w = 0.0;
    for (std::size_t idx : active) total_w += cands[idx].b;
    double median = cands[active.back()].x;
    if (total_w > 0.0) {
      double cum = 0.0;
      for (std::size_t idx : active) {
        cum += cands[idx].b;
        if (cum >= total_w / 2.0) {
          median = cands[idx].x;
          break;
        }
      }
    }
    std::vector<std::size_t> dev = active;
    std::sort(dev.begin(), dev.end(), [&](std::size_t i, std::size_t j) {
      const double di = std::abs(cands[i].x - median);
      const double dj = std::abs(cands[j].x - median);
      if (di != dj) return di < dj;
      return cands[i].frame_order < cands[j].frame_order;
    });
    double mad = std::abs(cands[dev.back()].x - median);
    if (total_w > 0.0) {
      double cum = 0.0;
      for (std::size_t idx : dev) {
        cum += cands[idx].b;
        if (cum >= total_w / 2.0) {
          mad = std::abs(cands[idx].x - median);
          break;
        }
      }
    }
    const double lower = median - s_low * mad;
    const double upper = median + s_high * mad;
    bool changed = false;
    for (std::size_t idx : active) {
      const double x = cands[idx].x;
      if (!(x >= lower && x <= upper)) {
        mask[idx] = 0;
        changed = true;
      }
    }
    if (!changed) break;
  }
  return mask;
}

void require_g9_profile_parity(
    const ForwardDrizzleV2ProfileResult &e,
    const ForwardDrizzleV2ProfileResult &g, std::size_t pc) {
  auto check = [&](const ForwardDrizzleV2ProfileOutput &ep,
                   const ForwardDrizzleV2ProfileOutput &gp,
                   const char *name) {
    if (ep.support != gp.support ||
        (ep.support && !std::isfinite(ep.value) != !std::isfinite(gp.value))) {
      std::fprintf(stderr,
                   "gate9 %s mismatch pc=%llu dev(sup=%u v=%g w=%g) "
                   "cpu(sup=%u v=%g w=%g)\n",
                   name, (unsigned long long)pc, (unsigned)gp.support,
                   gp.value, gp.weight_sum, (unsigned)ep.support, ep.value,
                   ep.weight_sum);
    }
    REQUIRE(gp.support == ep.support);
    if (!ep.support) return;
    if (std::abs(gp.weight_sum - ep.weight_sum) >
        1e-5 + 1e-5 * std::abs(ep.weight_sum))
      std::fprintf(stderr, "gate9 %s weight pc=%llu dev=%g cpu=%g\n", name,
                   (unsigned long long)pc, gp.weight_sum, ep.weight_sum);
    REQUIRE(gp.value == Catch::Approx(ep.value).epsilon(1e-5).margin(1e-5));
    REQUIRE(gp.weight_sum ==
            Catch::Approx(ep.weight_sum).epsilon(1e-5).margin(1e-5));
    REQUIRE(gp.n_eff == Catch::Approx(ep.n_eff).epsilon(1e-4).margin(1e-4));
  };
  check(e.uniform, g.uniform, "uniform");
  check(e.raw, g.raw, "raw");
  check(e.fine, g.fine, "fine");
  check(e.medium, g.medium, "medium");
  REQUIRE(g.a_separation == Catch::Approx(e.a_separation).margin(1e-5));
  REQUIRE(g.a_artifact == Catch::Approx(e.a_artifact).margin(1e-4));
  REQUIRE(g.a_registration == Catch::Approx(e.a_registration).margin(1e-4));
  REQUIRE(g.artifact_applicable == e.artifact_applicable);
}

}  // namespace

TEST_CASE("forward drizzle v2 gate9 device profiles match the CPU oracle",
          "[forward-drizzle-v2][cuda-parity][gate9]") {
  if (!forward_drizzle_cuda_runtime_available()) {
    SUCCEED("CUDA device unavailable");
    return;
  }
  const BayerPattern patterns[] = {BayerPattern::GRBG, BayerPattern::GBRG};
  for (int scale : {1, 2}) {
    for (ColorMode mode : {ColorMode::MONO, ColorMode::OSC}) {
      for (BayerPattern pattern : patterns) {
        auto f = make_fixture(mode, pattern, 1, -1, scale);
        // Extend to 12 frames: the artifact factor needs >= 8 accepted
        // contributors and fallback coverage needs sparse pixels.
        const std::size_t base_frames = f.plan.frames.size();
        for (std::size_t i = base_frames; i < 12; ++i) {
          FrameSamplingTransform frame = f.plan.frames[i % base_frames];
          frame.frame_id = "v2-g9-" + std::to_string(i);
          frame.source_index = i;
          frame.source_to_canvas(0, 2) +=
              static_cast<float>(0.011 * static_cast<double>(i - base_frames));
          f.plan.frames.push_back(frame);
          f.images.push_back(f.images[i % base_frames]);
        }
        const int nc = f.plan.canvas_width_native;
        const int nr = f.plan.canvas_height_native;
        const int channels = mode == ColorMode::MONO ? 1 : 3;
        const std::size_t nplane = static_cast<std::size_t>(nc) * nr;
        const std::size_t n_src = static_cast<std::size_t>(
            f.plan.source_width) * f.plan.source_height;
        const int n_frames = static_cast<int>(f.plan.frames.size());

        // Deterministic per-frame quality maps with veto coverage.
        std::vector<Matrix2Df> qc(n_frames), q0(n_frames), q1(n_frames),
            qa(n_frames);
        std::vector<ForwardDrizzleV2FrameMeta> meta(n_frames);
        for (int fr = 0; fr < n_frames; ++fr) {
          qc[fr].resize(f.plan.source_height, f.plan.source_width);
          q0[fr].resize(f.plan.source_height, f.plan.source_width);
          q1[fr].resize(f.plan.source_height, f.plan.source_width);
          qa[fr].resize(f.plan.source_height, f.plan.source_width);
          for (int y = 0; y < f.plan.source_height; ++y)
            for (int x = 0; x < f.plan.source_width; ++x) {
              const std::size_t si =
                  static_cast<std::size_t>(y) * f.plan.source_width + x;
              const double s = static_cast<double>((si + 7 * fr) % 61) / 61.0;
              qc[fr](y, x) = static_cast<float>(0.4 + 0.6 * s);
              q0[fr](y, x) = static_cast<float>(0.3 + 0.7 * ((s * fr) -
                                  std::floor(s * fr)));
              q1[fr](y, x) = static_cast<float>(0.5 + 0.4 * s);
              qa[fr](y, x) = static_cast<float>(0.2 + 0.6 * s);
            }
          // Veto coverage: nonfinite / nonpositive samples per stream.
          qc[fr](0, 0) = std::numeric_limits<float>::quiet_NaN();
          if (f.plan.source_width > 1) qc[fr](0, 1) = -1.0f;
          qa[fr](1 % f.plan.source_height, 0) =
              std::numeric_limits<float>::quiet_NaN();
          meta[fr] = {static_cast<float>(0.6 + 0.05 * (fr % 5)),
                      static_cast<float>(0.55 + 0.05 * fr),
                      static_cast<std::uint8_t>(fr % 2), 0};
        }

        ForwardDrizzleV2KernelConfig kcfg;
        kcfg.internal_scale = scale;
        kcfg.stream_length = n_frames;
        kcfg.half = 0.5 * f.cfg.pixfrac;
        kcfg.bayer_pattern = static_cast<int>(pattern);
        kcfg.cfa_origin_x = f.plan.cfa_origin_x;
        kcfg.cfa_origin_y = f.plan.cfa_origin_y;
        kcfg.mono = mode == ColorMode::MONO;
        kcfg.emit_profiles = true;

        ForwardDrizzleV2CudaPrototypeKernel kernel;
        REQUIRE(kernel.reserve(nc, nr, f.plan.source_width,
                               f.plan.source_height, kcfg));
        for (int fr = 0; fr < n_frames; ++fr) {
          const auto &m = f.plan.frames[fr].source_to_canvas;
          const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                                m(1, 0), m(1, 1), m(1, 2)};
          ForwardDrizzleV2FrameQuality q;
          q.q_composite = qc[fr].data();
          q.q_scale0 = q0[fr].data();
          q.q_scale1 = q1[fr].data();
          // Frame 2 exercises an absent artifact stream (qa_has_data=false).
          q.q_artifact = fr == 2 ? nullptr : qa[fr].data();
          REQUIRE(kernel.accumulate_frame(a6, f.images[fr].data(), nullptr,
                                          static_cast<std::uint64_t>(fr), &q,
                                          &meta[fr]));
        }
        std::vector<ForwardDrizzleV2PixelResult> got(nplane * channels);
        std::vector<ForwardDrizzleV2ProfileResult> gp(nplane * channels);
        std::uint64_t dense = 0;
        REQUIRE(kernel.finalize(got.data(), gp.data(), &dense));

        // CPU oracle: gather per-frame planes for value, quality sums and
        // the finite-artifact weight; fold, clip-mask, profile reduce.
        const int ic = nc * scale, ir = nr * scale;
        const double inv_s2 = 1.0 / (static_cast<double>(scale) * scale);
        std::vector<std::vector<ForwardDrizzleV2ProfileCandidate>> cands(
            nplane * channels);
        std::vector<std::vector<ForwardDrizzleV2RobustCandidate>> rcands(
            nplane * channels);
        for (int fr = 0; fr < n_frames; ++fr) {
          RegistrationSamplingPlan one = f.plan;
          one.frames = {f.plan.frames[fr]};
          const auto &img = f.images[fr];
          auto real_of = [&](std::size_t) -> const Matrix2Df & { return img; };
          const auto g_a =
              gather_affine_uniform_v2(one, real_of, f.cfg, 0, ir);
          // Clamped images: nonfinite/<=0 samples contribute 0 to the
          // area-weighted mean, and quality attaches only to samples whose
          // source value is finite (the same population as b_src). The
          // artifact-flag image carries 1.0 where finite artifact data
          // exists, so its wx sum is the finite-artifact area weight.
          Matrix2Df cqc(f.plan.source_height, f.plan.source_width),
              cq0(f.plan.source_height, f.plan.source_width),
              cq1(f.plan.source_height, f.plan.source_width),
              cqa(f.plan.source_height, f.plan.source_width),
              qaf(f.plan.source_height, f.plan.source_width);
          for (int y = 0; y < f.plan.source_height; ++y)
            for (int x = 0; x < f.plan.source_width; ++x) {
              const bool finite_src = std::isfinite(img(y, x));
              auto cl = [](float v) {
                return std::isfinite(v) && v > 0.0f ? v : 0.0f;
              };
              cqc(y, x) = finite_src ? cl(qc[fr](y, x)) : 0.0f;
              cq0(y, x) = finite_src ? cl(q0[fr](y, x)) : 0.0f;
              cq1(y, x) = finite_src ? cl(q1[fr](y, x)) : 0.0f;
              cqa(y, x) = finite_src ? cl(qa[fr](y, x)) : 0.0f;
              qaf(y, x) =
                  finite_src && std::isfinite(qa[fr](y, x)) ? 1.0f : 0.0f;
            }
          auto qc_of = [&](std::size_t) -> const Matrix2Df & { return cqc; };
          auto q0_of = [&](std::size_t) -> const Matrix2Df & { return cq0; };
          auto q1_of = [&](std::size_t) -> const Matrix2Df & { return cq1; };
          auto qa_of = [&](std::size_t) -> const Matrix2Df & { return cqa; };
          auto qaf_of = [&](std::size_t) -> const Matrix2Df & { return qaf; };
          const auto g_qc = gather_affine_uniform_v2(one, qc_of, f.cfg, 0, ir);
          const auto g_q0 = gather_affine_uniform_v2(one, q0_of, f.cfg, 0, ir);
          const auto g_q1 = gather_affine_uniform_v2(one, q1_of, f.cfg, 0, ir);
          const auto g_qa = gather_affine_uniform_v2(one, qa_of, f.cfg, 0, ir);
          const auto g_qaf =
              gather_affine_uniform_v2(one, qaf_of, f.cfg, 0, ir);
          const bool qa_present = fr != 2;
          for (std::size_t px = 0; px < nplane; ++px) {
            const int nx = static_cast<int>(px % nc);
            const int ny = static_cast<int>(px / nc);
            for (int c = 0; c < channels; ++c) {
              const std::size_t pc =
                  static_cast<std::size_t>(c) * nplane + px;
              double a = 0.0, bs = 0.0, sq = 0.0, s0 = 0.0, s1 = 0.0,
                     sa = 0.0, saf = 0.0;
              for (int iy = 0; iy < scale; ++iy)
                for (int ix = 0; ix < scale; ++ix) {
                  const std::size_t ii =
                      static_cast<std::size_t>(ny * scale + iy) * ic +
                      nx * scale + ix;
                  a += inv_s2 * g_a.accum.wx[c][ii];
                  bs += inv_s2 * g_a.accum.w[c][ii];
                  sq += inv_s2 * g_qc.accum.wx[c][ii];
                  s0 += inv_s2 * g_q0.accum.wx[c][ii];
                  s1 += inv_s2 * g_q1.accum.wx[c][ii];
                  sa += inv_s2 * g_qa.accum.wx[c][ii];
                  saf += inv_s2 * g_qaf.accum.wx[c][ii];
                }
              if (!(bs > 0.0)) continue;
              rcands[pc].push_back({static_cast<std::size_t>(fr), a / bs, bs});
              ForwardDrizzleV2ProfileCandidate cd;
              cd.frame_order = fr;
              cd.x = a / bs;
              cd.b = bs;
              cd.q = sq / bs;
              cd.q0 = s0 / bs;
              cd.q1 = s1 / bs;
              cd.qa = qa_present ? sa / bs : 1.0;
              cd.qa_has_data = qa_present && saf > 0.0;
              cands[pc].push_back(cd);
            }
          }
        }

        ForwardDrizzleV2RobustConfig rcfg;
        rcfg.reservoir_size = kcfg.reservoir_size;
        rcfg.reservoir_seed = kcfg.reservoir_seed;
        rcfg.oracle_min_clip_contributors = kcfg.min_clip_contributors;
        rcfg.oracle_passes = kcfg.robust_passes;
        rcfg.oracle_sigma_low = kcfg.sigma_low;
        rcfg.oracle_sigma_high = kcfg.sigma_high;
        rcfg.min_candidates = kcfg.min_candidates;
        ForwardDrizzleV2ProfileConfig pcfg;
        pcfg.fine_quality_exponent = kcfg.fine_quality_exponent;
        pcfg.medium_quality_exponent = kcfg.medium_quality_exponent;
        bool saw_primary = false, saw_fallback = false,
             saw_artifact_applicable = false;
        for (std::size_t pc = 0; pc < nplane * channels; ++pc) {
          const auto &r = got[pc];
          if (r.robust_state ==
              static_cast<std::uint8_t>(
                  ForwardDrizzleV2RobustState::no_source_support)) {
            REQUIRE(gp[pc].uniform.support == 0);
            continue;
          }
          if (r.robust_state !=
              static_cast<std::uint8_t>(
                  ForwardDrizzleV2RobustState::primary_reservoir_sigma_clip)) {
            // Fallback contract: every profile carries the uniform stream
            // value, alphas are zero.
            saw_fallback = true;
            const float v = static_cast<float>(r.value);
            for (const auto *o :
                 {&gp[pc].uniform, &gp[pc].raw, &gp[pc].fine,
                  &gp[pc].medium}) {
              REQUIRE(o->support == 1);
              REQUIRE(o->value == Catch::Approx(v).margin(1e-4));
            }
            REQUIRE(gp[pc].a_separation == 0.0f);
            REQUIRE(gp[pc].a_artifact == 0.0f);
            REQUIRE(gp[pc].a_registration == 0.0f);
            continue;
          }
          saw_primary = true;
          const auto mask = g9_clip_mask(
              rcands[pc], kcfg.min_clip_contributors, kcfg.robust_passes,
              kcfg.sigma_low, kcfg.sigma_high);
          const auto ref = robust_reduce_candidates_v2(
              rcands[pc], ForwardDrizzleV2Estimator::reservoir_sigma_clip,
              rcfg, static_cast<std::uint64_t>(n_frames), {});
          const auto expected = forward_drizzle_v2_profile_reduce(
              cands[pc], mask, meta, pcfg, ref.confidence,
              ref.candidates > 0 && ref.conf_degraded == ref.candidates);
          require_g9_profile_parity(expected, gp[pc], pc);
          if (gp[pc].artifact_applicable) saw_artifact_applicable = true;
        }
        REQUIRE(saw_primary);
        REQUIRE(saw_fallback);
        REQUIRE(saw_artifact_applicable);
        REQUIRE(kernel.stats().allocations == 1);
        REQUIRE(kernel.stats().device_global_synchronizations == 0);
        REQUIRE(kernel.stats().stream_synchronizations == 1);
      }
    }
  }
}

namespace {

// Deterministic synthetic star field for the gate-9 validation integration
// test (same construction as test_multiband_validation.cpp's fixture).
Matrix2Df g9_field(int w, int h, double sigma, double amp, double bg,
                   uint32_t seed) {
  Matrix2Df img(h, w);
  std::mt19937 rng(seed);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      img(y, x) = static_cast<float>(
          bg + 2.0 * std::sin(0.31 * x + 0.17 * y + seed));
  const int r = static_cast<int>(std::ceil(4 * sigma));
  for (int gy = 16; gy < h - 16; gy += 27)
    for (int gx = 16; gx < w - 16; gx += 27)
      for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
          const double g =
              amp * std::exp(-static_cast<double>(dx * dx + dy * dy) /
                             (2 * sigma * sigma));
          img(gy + dy, gx + dx) += static_cast<float>(g);
        }
  return img;
}

}  // namespace

TEST_CASE("forward drizzle v2 gate9 ring immutability and reuse counters",
          "[forward-drizzle-v2][gate9][ring]") {
  const int H = 100, core = 10;
  for (int levels : {1, 2, 4}) {
    const auto plan =
        forward_drizzle_v2_multiband_ring_plan(H, core, levels);
    const int h = plan.halo_rows;
    REQUIRE(h == multiband_fusion_halo_rows(levels));
    REQUIRE(plan.resident_bands == 2 * h / core + 2);
    const int nb = (H + core - 1) / core;
    REQUIRE(plan.band_core_rows == core);
    REQUIRE(plan.canvas_rows == H);

    ForwardDrizzleV2MultibandRing ring(plan, 8, 4);
    REQUIRE(ring.band_count() == nb);
    std::vector<char> seen_immutable(nb, 0);
    for (int b = 0; b < nb; ++b) {
      const auto newly = ring.finalize_band(b);
      for (int i : newly) seen_immutable[i] = 1;
      REQUIRE(ring.resident_bands() <= plan.resident_bands);
      // Core b may only be immutable once its full halo window is covered by
      // finalized input bands.
      for (int cb = 0; cb < nb; ++cb) {
        if (!ring.core_immutable(cb)) continue;
        const int lo = std::max(0, cb * core - h);
        const int hi = std::min(H, (cb + 1) * core + h);
        const int ilo = lo / core;
        const int ihi = std::min(nb - 1, (hi - 1) / core);
        for (int i = ilo; i <= ihi; ++i) REQUIRE(ring.band_finalized(i));
      }
    }
    for (int cb = 0; cb < nb; ++cb) {
      REQUIRE(seen_immutable[cb]);
      REQUIRE(ring.core_immutable(cb));
    }
    // First and last edges: core 0 cannot be immutable before finalize(0)
    // covers the upper edge; the last core becomes immutable only with the
    // last band.
    // Reuse accounting: every canvas row is read exactly once; the total
    // padded demand minus fresh reads is the reused share.
    REQUIRE(ring.source_read_bytes() ==
            static_cast<std::uint64_t>(H) * 8);
    REQUIRE(ring.quality_read_bytes() ==
            static_cast<std::uint64_t>(H) * 4);
    REQUIRE(ring.source_reused_bytes() > 0);
    REQUIRE(ring.source_reused_bytes() ==
            2 * ring.quality_reused_bytes());
    // The recorded reuse rate matches the reuse-aware RA model; the <= 1.1
    // bound applies to the selected tile plan (production-scale canvas, not
    // this toy geometry).
    const auto ra = forward_drizzle_v2_streaming_ra(H, 8, h, nb);
    REQUIRE(ring.source_read_bytes() <= ra.application_bytes_reuse);
    const auto ra_prod = forward_drizzle_v2_streaming_ra(
        4540, 8, h, (4540 + 99) / 100);
    REQUIRE(ra_prod.ra_reuse <= 1.1);
    REQUIRE(ring.resident_bands() <= plan.resident_bands);
  }
  // Edge: band smaller than halo -> first bands cover the whole window
  // neighbourhood before any core is immutable.
  const auto plan = forward_drizzle_v2_multiband_ring_plan(20, 4, 4);
  ForwardDrizzleV2MultibandRing ring(plan, 1, 1);
  const auto n0 = ring.finalize_band(0);
  REQUIRE(n0.empty());  // window of core 0 still needs later bands
  // Double finalize and out-of-range are rejected.
  REQUIRE_THROWS_AS(ring.finalize_band(0), std::invalid_argument);
  REQUIRE_THROWS_AS(ring.finalize_band(-1), std::invalid_argument);
  REQUIRE_THROWS_AS(ring.finalize_band(ring.band_count()),
                    std::invalid_argument);
  // Degenerate plan inputs throw.
  REQUIRE_THROWS_AS(forward_drizzle_v2_multiband_ring_plan(0, 4, 2),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(forward_drizzle_v2_multiband_ring_plan(20, 0, 2),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(forward_drizzle_v2_multiband_ring_plan(20, 4, 0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(forward_drizzle_v2_multiband_ring_plan(20, 4, 5),
                    std::invalid_argument);
}

TEST_CASE("forward drizzle v2 gate9 confidence calibration and alpha "
          "diagnostics",
          "[forward-drizzle-v2][gate9][confidence]") {
  using reconstruction::ForwardDrizzleV2ProfileCandidate;
  using reconstruction::ForwardDrizzleV2FrameMeta;
  // 12 uniform candidates: clean modeled sigma2 => confidence ~12/13 >= 0.9.
  std::vector<ForwardDrizzleV2ProfileCandidate> cands;
  std::vector<ForwardDrizzleV2RobustCandidate> rcands;
  std::vector<ForwardDrizzleV2FrameMeta> meta;
  std::vector<double> sigma2;
  for (int i = 0; i < 12; ++i) {
    cands.push_back({static_cast<std::size_t>(i), 10.0 + 0.01 * i, 1.0, 1.0,
                     1.0, 1.0, 0.8, true});
    rcands.push_back(
        {static_cast<std::size_t>(i), 10.0 + 0.01 * i, 1.0});
    meta.push_back({1.0f, 0.8f, 1, 0});
    sigma2.push_back(0.01);
  }
  ForwardDrizzleV2RobustConfig rcfg;
  const auto rr = robust_reduce_candidates_v2(
      rcands, ForwardDrizzleV2Estimator::reservoir_sigma_clip, rcfg, 12,
      sigma2);
  REQUIRE(rr.state ==
          ForwardDrizzleV2RobustState::primary_reservoir_sigma_clip);
  REQUIRE(rr.confidence >= 0.9);
  std::vector<std::uint8_t> acc(cands.size(), 1);
  const auto clean = forward_drizzle_v2_profile_reduce(
      cands, acc, meta, {}, rr.confidence);
  REQUIRE(clean.a_separation >= 0.9f);

  // Degraded: every contributor's sigma2 invalid -> a_separation collapses.
  std::vector<double> bad_sigma(sigma2.size(),
                                std::numeric_limits<double>::quiet_NaN());
  const auto rbad = robust_reduce_candidates_v2(
      rcands, ForwardDrizzleV2Estimator::reservoir_sigma_clip, rcfg, 12,
      bad_sigma);
  REQUIRE(rbad.conf_degraded == rbad.candidates);
  const auto degraded = forward_drizzle_v2_profile_reduce(
      cands, acc, meta, {}, rbad.confidence, true);
  REQUIRE(degraded.a_separation <= 0.1f);

  // Channel-min: per-pixel min over supported channels; a pixel without
  // support stays NaN.
  std::vector<ForwardDrizzleV2ProfileResult> recs(3 * 4);
  recs[0 * 4 + 0] = clean;   // ch0 px0: sep ~0.92
  recs[1 * 4 + 0] = degraded;  // ch1 px0: sep 0 -> channel min 0
  recs[2 * 4 + 0] = clean;
  const auto sep_plane =
      forward_drizzle_v2_profile_alpha_plane(recs, 2, 2, 3, 0);
  REQUIRE(sep_plane[0] == Catch::Approx(0.0f));
  REQUIRE(std::isnan(sep_plane[1]));

  // Near-zero diagnostic: the field above is globally near-zero at px0 only.
  const auto diag =
      forward_drizzle_v2_alpha_diagnostic(recs, 2, 2, 3);
  REQUIRE(diag.supported_pixels == 1);
  REQUIRE(diag.near_zero_pixels == 1);
  REQUIRE(diag.global_near_zero);

  // A field with finite nonzero alpha is not flagged.
  std::vector<ForwardDrizzleV2ProfileResult> ok(3 * 4);
  for (int c = 0; c < 3; ++c)
    for (int px = 0; px < 4; ++px) {
      auto r = clean;
      r.a_artifact = 0.9f;
      r.a_registration = 0.9f;
      r.artifact_applicable = true;
      ok[static_cast<std::size_t>(c) * 4 + px] = r;
    }
  const auto diag_ok = forward_drizzle_v2_alpha_diagnostic(ok, 2, 2, 3);
  REQUIRE(diag_ok.supported_pixels == 4);
  REQUIRE(diag_ok.near_zero_pixels == 0);
  REQUIRE_FALSE(diag_ok.global_near_zero);

  // Artifact not applicable folds in as 1: high sep/reg alone is not
  // near-zero.
  for (auto &r : ok) r.artifact_applicable = false;
  const auto diag_na = forward_drizzle_v2_alpha_diagnostic(ok, 2, 2, 3);
  REQUIRE(diag_na.near_zero_pixels == 0);
}

TEST_CASE("forward drizzle v2 gate9 fusion parity and support subset",
          "[forward-drizzle-v2][gate9][fusion]") {
  const int W = 40, H = 36;
  const std::size_t n = static_cast<std::size_t>(W) * H;
  // Synthetic profile records: uniform = smooth base, raw = base + high
  // frequency, fine = base + fine detail, medium = base + mid detail.
  std::vector<ForwardDrizzleV2ProfileResult> recs(n);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const std::size_t i = static_cast<std::size_t>(y) * W + x;
      const float base = 100.0f + 0.5f * x + 0.3f * y;
      auto set = [](ForwardDrizzleV2ProfileOutput &o, float v) {
        o.value = v;
        o.weight_sum = 4.0f;
        o.n_eff = 3.0f;
        o.support = 1;
      };
      auto &r = recs[i];
      set(r.uniform, base);
      set(r.raw, base + std::sin(0.7f * x) * std::cos(0.9f * y));
      set(r.fine, base + 0.5f * std::sin(1.7f * x + 0.3f * y));
      set(r.medium, base + 0.8f * std::sin(0.2f * x));
      r.a_separation = 0.8f;
      r.a_artifact = 0.9f;
      r.a_registration = 0.95f;
      r.artifact_applicable = true;
    }
  // A few unsupported pixels exercise the support-subset contract.
  for (int x = 0; x < 5; ++x) {
    auto &r = recs[x];
    for (auto *o : {&r.uniform, &r.raw, &r.fine, &r.medium}) {
      o->support = 0;
      o->value = std::numeric_limits<float>::quiet_NaN();
    }
  }
  const auto uniform = forward_drizzle_v2_profiles_to_uniform_result(
      recs, W, H, 1, ColorMode::MONO, 0);
  const auto raw = forward_drizzle_v2_profiles_to_uniform_result(
      recs, W, H, 1, ColorMode::MONO, 1);
  const auto fine = forward_drizzle_v2_profiles_to_uniform_result(
      recs, W, H, 1, ColorMode::MONO, 2);
  const auto medium = forward_drizzle_v2_profiles_to_uniform_result(
      recs, W, H, 1, ColorMode::MONO, 3);
  REQUIRE(uniform.internal_width == W);
  REQUIRE(uniform.L.value.size() == n);
  const auto a_sep =
      forward_drizzle_v2_profile_alpha_plane(recs, W, H, 1, 0);
  const auto a_art =
      forward_drizzle_v2_profile_alpha_plane(recs, W, H, 1, 1);
  const auto a_reg =
      forward_drizzle_v2_profile_alpha_plane(recs, W, H, 1, 2);

  for (int levels : {1, 2, 3, 4}) {
    config::ReconstructionMultibandConfig mb_cfg;
    mb_cfg.levels = levels;
    const auto whole = fuse_multiband(uniform, raw, fine, medium,
                                      ColorMode::MONO, W, H, mb_cfg, {}, {},
                                      a_sep, a_art, a_reg);
    const auto streamed =
        fuse_multiband_streamed(uniform, raw, fine, medium, ColorMode::MONO,
                                W, H, mb_cfg, 5, {}, {}, a_sep, a_art,
                                a_reg);
    REQUIRE(streamed.L.size() == whole.L.size());
    REQUIRE(streamed.support_L == whole.support_L);
    for (std::size_t i = 0; i < n; ++i) {
      // Fused support is a subset of the uniform/raw support and the
      // streamed result is identical on this fully connected field (NaN
      // border pixels compare equal by position).
      if (whole.support_L[i])
        REQUIRE(uniform.L.support[i] == 1);
      const float sv = streamed.L[i], wv = whole.L[i];
      REQUIRE((sv == wv || (std::isnan(sv) && std::isnan(wv))));
    }
  }
  // Support subset: a pixel unsupported in the uniform profile is
  // unsupported in the fused output.
  config::ReconstructionMultibandConfig mb_cfg;
  mb_cfg.levels = 3;
  const auto fused = fuse_multiband(uniform, raw, fine, medium,
                                    ColorMode::MONO, W, H, mb_cfg, {}, {},
                                    a_sep, a_art, a_reg);
  for (int x = 0; x < 5; ++x) REQUIRE(fused.support_L[x] == 0);
}

TEST_CASE("forward drizzle v2 gate9 matched-star gates on v2-produced "
          "images",
          "[forward-drizzle-v2][gate9][validation]") {
  const int W = 200, H = 180;
  const auto U = g9_field(W, H, 2.4, 900.0, 100.0, 5);
  const auto R = g9_field(W, H, 1.9, 900.0, 100.0, 5);
  const std::size_t n = static_cast<std::size_t>(W) * H;

  // Build v2 profile records: uniform/raw from the fields, fine/medium equal
  // to raw so any nonzero alpha stays on the raw manifold.
  std::vector<ForwardDrizzleV2ProfileResult> recs(n);
  for (std::size_t i = 0; i < n; ++i) {
    auto set = [](ForwardDrizzleV2ProfileOutput &o, float v) {
      o.value = v;
      o.weight_sum = 8.0f;
      o.n_eff = 7.0f;
      o.support = 1;
    };
    set(recs[i].uniform, U(i / W, i % W));
    set(recs[i].raw, R(i / W, i % W));
    set(recs[i].fine, R(i / W, i % W));
    set(recs[i].medium, R(i / W, i % W));
  }
  const auto uniform = forward_drizzle_v2_profiles_to_uniform_result(
      recs, W, H, 1, ColorMode::MONO, 0);
  const auto raw = forward_drizzle_v2_profiles_to_uniform_result(
      recs, W, H, 1, ColorMode::MONO, 1);
  const auto fine = forward_drizzle_v2_profiles_to_uniform_result(
      recs, W, H, 1, ColorMode::MONO, 2);
  const auto medium = forward_drizzle_v2_profiles_to_uniform_result(
      recs, W, H, 1, ColorMode::MONO, 3);

  // Case 1: alpha identically zero -> every band is raw-sourced -> the fused
  // field is uniform-coarse plus raw detail and every star's
  // multiband_effective flag is cleared by the empty alpha_final maps.
  // Zero effective stars can never select Multiband -> Raw.
  std::vector<float> zero_alpha(n, 0.0f);
  config::ReconstructionMultibandConfig mb_cfg;
  mb_cfg.levels = 3;
  const auto fused0 = fuse_multiband(uniform, raw, fine, medium,
                                     ColorMode::MONO, W, H, mb_cfg, {}, {},
                                     zero_alpha, zero_alpha, zero_alpha);
  Matrix2Df fused_img(H, W);
  for (std::size_t i = 0; i < n; ++i) fused_img(i / W, i % W) = fused0.L[i];
  const auto diag = forward_drizzle_v2_alpha_diagnostic(recs, W, H, 1);
  // All alpha factors are 0 on the records (defaults) -> globally near-zero.
  REQUIRE(diag.supported_pixels == n);
  REQUIRE(diag.global_near_zero);

  auto stars =
      prepare_validation_samples(U, W, H, {}, fused0.alpha_final);
  REQUIRE(stars.size() >= 20);
  REQUIRE(std::none_of(stars.begin(), stars.end(), [](const ValidationStar &s) {
    return s.multiband_effective;
  }));
  const auto sel0 = select_reconstruction_candidate(U, R, fused_img, W, H,
                                                    stars);
  INFO("reason: " << sel0.reason);
  REQUIRE(sel0.stars_multiband_effective == 0);
  REQUIRE(sel0.selected == SelectedCandidate::kDrizzleRaw);

  // Case 2: multiband identical to raw (the alpha==0 end-to-end outcome) ->
  // the FWHM ratio gate fails at equality -> Raw.
  for (auto &s : stars) s.multiband_effective = true;
  const auto sel1 = select_reconstruction_candidate(U, R, R, W, H, stars);
  INFO("reason: " << sel1.reason);
  REQUIRE(sel1.selected == SelectedCandidate::kDrizzleRaw);

  // Case 3: raw fails an applicable safety gate (background RMS regresses
  // vs. uniform) -> Uniform.
  Matrix2Df raw_bad = R;
  std::mt19937 rng(17);
  std::normal_distribution<double> gn(0.0, 40.0);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      raw_bad(y, x) += static_cast<float>(gn(rng));
  const auto sel2 = select_reconstruction_candidate(U, raw_bad, fused_img, W,
                                                    H, stars);
  INFO("reason: " << sel2.reason);
  REQUIRE(sel2.selected == SelectedCandidate::kDrizzleUniform);
}

TEST_CASE("forward drizzle v2 gate9 memory plan profiled amendment",
          "[forward-drizzle-v2][gate9][memory]") {
  static_assert(sizeof(ForwardDrizzleV2ProfileResult) <= 80,
                "profile record role bound");
  ForwardDrizzleV2MemoryInputs in;
  in.target_width = 7868;
  in.target_height = 4540;
  in.channels = 3;
  in.internal_scale = 2;
  in.frame_count = 60;
  in.device_budget_bytes = 16ull << 30;
  in.host_budget_bytes = 32ull << 30;
  const auto base = plan_forward_drizzle_v2_memory(in);
  REQUIRE(base.feasible);
  in.emit_profiles = true;
  in.frame_meta_slots = 64;
  const auto prof = plan_forward_drizzle_v2_memory(in);
  REQUIRE(prof.feasible);
  // The amended per-pixel role table: base + reservoir quality side array +
  // profile record + five quality frame planes, all per channel.
  const auto &r = in.roles;
  const std::size_t slots = static_cast<std::size_t>(r.reservoir_size) *
                            r.reservoir_slot_factor;
  const std::size_t expected = base.device_bytes_per_target_pixel +
      in.channels * (slots * r.reservoir_quality_bytes_per_slot +
                     r.profile_result_bytes +
                     static_cast<std::size_t>(r.quality_frame_plane_doubles) *
                         sizeof(double) * in.internal_scale *
                         in.internal_scale);
  REQUIRE(prof.device_bytes_per_target_pixel == expected);
  // Gate-9 documented bound: ~14.6 KB/px estimate holds for the MONO,
  // internal_scale-1 profiled plan; the OSC production plan is larger.
  ForwardDrizzleV2MemoryInputs mono = in;
  mono.channels = 1;
  mono.internal_scale = 1;
  const auto mono_plan = plan_forward_drizzle_v2_memory(mono);
  REQUIRE(mono_plan.feasible);
  REQUIRE(mono_plan.device_bytes_per_target_pixel <= 14600);
  // Non-profiled mode is bit-identical to the Gate-6 footprint.
  REQUIRE(base.device_bytes_per_target_pixel <
          prof.device_bytes_per_target_pixel);
  in.emit_profiles = false;
  REQUIRE(plan_forward_drizzle_v2_memory(in)
              .device_bytes_per_target_pixel ==
          base.device_bytes_per_target_pixel);
}

TEST_CASE("forward drizzle v2 gate9 degraded sigma2 collapses device "
          "separation",
          "[forward-drizzle-v2][cuda-parity][gate9]") {
  if (!forward_drizzle_cuda_runtime_available()) {
    SUCCEED("CUDA device unavailable");
    return;
  }
  auto f = make_fixture(ColorMode::MONO, BayerPattern::GRBG, 1, -1, 1);
  const std::size_t base_frames = f.plan.frames.size();
  for (std::size_t i = base_frames; i < 10; ++i) {
    FrameSamplingTransform frame = f.plan.frames[i % base_frames];
    frame.frame_id = "v2-g9d-" + std::to_string(i);
    frame.source_index = i;
    frame.source_to_canvas(0, 2) += 0.01f * static_cast<float>(i);
    f.plan.frames.push_back(frame);
    f.images.push_back(f.images[i % base_frames]);
  }
  const int nc = f.plan.canvas_width_native;
  const int nr = f.plan.canvas_height_native;
  const int n_frames = static_cast<int>(f.plan.frames.size());
  const std::size_t n_src =
      static_cast<std::size_t>(f.plan.source_width) * f.plan.source_height;
  const std::size_t nplane = static_cast<std::size_t>(nc) * nr;

  // All-invalid sigma2 plane: every contributor degrades.
  std::vector<float> bad_sigma(
      n_src, std::numeric_limits<float>::quiet_NaN());
  std::vector<ForwardDrizzleV2FrameMeta> meta(
      n_frames, ForwardDrizzleV2FrameMeta{1.0f, 1.0f, 1, 0});
  ForwardDrizzleV2KernelConfig kcfg;
  kcfg.internal_scale = 1;
  kcfg.stream_length = n_frames;
  kcfg.half = 0.5 * f.cfg.pixfrac;
  kcfg.bayer_pattern = static_cast<int>(BayerPattern::GRBG);
  kcfg.mono = true;
  kcfg.emit_profiles = true;
  ForwardDrizzleV2CudaPrototypeKernel kernel;
  REQUIRE(kernel.reserve(nc, nr, f.plan.source_width, f.plan.source_height,
                         kcfg));
  for (int fr = 0; fr < n_frames; ++fr) {
    const auto &m = f.plan.frames[fr].source_to_canvas;
    const double a6[6] = {m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1),
                          m(1, 2)};
    ForwardDrizzleV2FrameQuality q;  // all streams absent -> q = 1.0
    REQUIRE(kernel.accumulate_frame(a6, f.images[fr].data(),
                                    bad_sigma.data(),
                                    static_cast<std::uint64_t>(fr), &q,
                                    &meta[fr]));
  }
  std::vector<ForwardDrizzleV2PixelResult> got(nplane);
  std::vector<ForwardDrizzleV2ProfileResult> gp(nplane);
  REQUIRE(kernel.finalize(got.data(), gp.data(), nullptr));
  int supported = 0, degraded_sep0 = 0;
  for (std::size_t i = 0; i < nplane; ++i) {
    if (got[i].robust_state ==
        static_cast<std::uint8_t>(
            ForwardDrizzleV2RobustState::primary_reservoir_sigma_clip)) {
      ++supported;
      // Every contributor carried invalid sigma2: no separation evidence.
      if (gp[i].a_separation == 0.0f) ++degraded_sep0;
      // Missing quality streams fold to 1.0: raw == uniform * g_eff.
      REQUIRE(gp[i].raw.value ==
              Catch::Approx(gp[i].uniform.value).margin(1e-4));
    }
  }
  REQUIRE(supported > 0);
  REQUIRE(degraded_sep0 == supported);
}

TEST_CASE("forward drizzle v2 gate9 production geometry profiled pipeline",
          "[.][forward-drizzle-v2-gate9-production]") {
  if (!forward_drizzle_cuda_runtime_available()) {
    SUCCEED("CUDA device unavailable");
    return;
  }
  constexpr char kSpecSha[] =
      "920206712ffef61310de397bf855b34b998100fb6c18c74e1b58fe2306ad2bdc";
  constexpr int sw = 3840, sh = 2160;
  constexpr int nc = 3934, nr = 2270;
  constexpr double kMaxSteadySeconds = 0.75;
  constexpr int n_frames = 60;

  const auto mem = forward_drizzle_cuda_device_memory();
  ForwardDrizzleV2MemoryInputs in;
  in.target_width = nc;
  in.target_height = nr;
  in.channels = 3;
  in.frame_count = n_frames;
  in.internal_scale = 2;
  in.emit_profiles = true;
  in.frame_meta_slots = 128;
  in.source_device_slots = 2;
  in.source_slot_device_bytes = std::size_t{sw} * sh * sizeof(float);
  in.quality_device_slots = 4;
  in.quality_slot_device_bytes = std::size_t{sw} * sh * sizeof(float);
  in.device_reserve_bytes = std::size_t{256} << 20;
  in.device_budget_bytes =
      static_cast<std::size_t>(static_cast<double>(mem.free_bytes) * 0.8);
  in.host_budget_bytes = std::size_t{8} << 30;
  const auto plan = plan_forward_drizzle_v2_memory(in);
  REQUIRE(plan.feasible);
  REQUIRE(plan.tile_cols == nc);
  const int band_rows = plan.band_rows;

  Matrix2Df source(sh, sw);
  for (int y = 0; y < sh; ++y)
    for (int x = 0; x < sw; ++x)
      source(y, x) = static_cast<float>(
          100.0 + 0.001 * x + 0.002 * y + 0.01 * ((17 * x + 31 * y) % 29));
  Matrix2Df qplane(sh, sw);
  for (int y = 0; y < sh; ++y)
    for (int x = 0; x < sw; ++x)
      qplane(y, x) = static_cast<float>(
          0.6 + 0.3 * std::sin(0.01 * x) * std::cos(0.013 * y));

  const std::array<std::array<double, 6>, 5> base{{
      {1.0, 0.0, 32.0, 0.0, 1.0, 50.0},
      {0.9998640418052673, -0.016167480498552322, 35.636409759521484,
       0.01610037311911583, 0.9999631643295288, 46.23441696166992},
      {0.9998466372489929, 0.017386717721819878, 55.92152404785156,
       -0.01743965968489647, 0.9998936057090759, 68.48160552978516},
      {0.9999858140945435, -0.0024168870877474546, 31.47933578491211,
       0.0021460296120494604, 1.000150442123413, 56.29893493652344},
      {1.0000033378601074, -0.006749980617314577, 28.21784782409668,
       0.006637410260736942, 1.0002058744430542, 58.95621871948242},
  }};

  ForwardDrizzleV2KernelConfig kcfg;
  kcfg.internal_scale = 2;
  kcfg.stream_length = n_frames;
  kcfg.half = 0.4;
  kcfg.bayer_pattern = static_cast<int>(BayerPattern::GBRG);
  kcfg.mono = false;
  kcfg.emit_profiles = true;

  ForwardDrizzleV2CudaPrototypeKernel kernel;
  REQUIRE(kernel.reserve(nc, band_rows, sw, sh, kcfg));
  for (std::uint64_t fr = 0; fr < static_cast<std::uint64_t>(n_frames);
       ++fr) {
    auto m = base[fr % base.size()];
    m[2] += 0.05 * static_cast<double>(fr);
    m[5] += 0.03 * static_cast<double>(fr);
    ForwardDrizzleV2FrameQuality q;
    q.q_composite = qplane.data();
    q.q_scale0 = qplane.data();
    q.q_scale1 = qplane.data();
    q.q_artifact = qplane.data();
    const ForwardDrizzleV2FrameMeta fm{0.9f, 0.8f,
                                       static_cast<std::uint8_t>(fr % 2), 0};
    REQUIRE(kernel.accumulate_frame(m.data(), source.data(), nullptr, fr,
                                    &q, &fm));
  }
  const std::size_t n_out =
      static_cast<std::size_t>(nc) * band_rows * 3;
  std::vector<ForwardDrizzleV2PixelResult> results(n_out);
  std::vector<ForwardDrizzleV2ProfileResult> profiles(n_out);
  std::uint64_t dense = 0;
  REQUIRE(kernel.finalize(results.data(), profiles.data(), &dense));
  const auto &st = kernel.stats();

  // Bounded-memory contract: the profiled footprint must fit the amended
  // per-pixel plan bound.
  const std::size_t bpp = kernel.device_bytes_per_native_pixel();
  std::fprintf(stderr,
               "gate9 production: bpp=%llu plan_bpp=%llu max_frame=%.4fs "
               "allocs=%llu gsyncs=%llu\n",
               (unsigned long long)bpp,
               (unsigned long long)plan.device_bytes_per_target_pixel,
               st.max_frame_seconds, (unsigned long long)st.allocations,
               (unsigned long long)st.device_global_synchronizations);
  REQUIRE(bpp <= plan.device_bytes_per_target_pixel);
  REQUIRE(st.allocations == 1);
  REQUIRE(st.device_global_synchronizations == 0);
  REQUIRE(st.max_frame_seconds < kMaxSteadySeconds);

  // Spot-check profile sanity on the dense core.
  std::uint64_t supported = 0, raw_supported = 0, art_applicable = 0;
  for (std::size_t i = 0; i < n_out; ++i) {
    if (results[i].robust_state !=
        static_cast<std::uint8_t>(
            ForwardDrizzleV2RobustState::primary_reservoir_sigma_clip))
      continue;
    ++supported;
    const auto &p = profiles[i];
    REQUIRE(p.uniform.support == 1);
    if (p.raw.support) ++raw_supported;
    if (p.artifact_applicable) ++art_applicable;
    REQUIRE(std::isfinite(p.uniform.value));
  }
  REQUIRE(supported > 0);
  REQUIRE(raw_supported > 0);
  REQUIRE(art_applicable > 0);

  // Persist the production measurement for the decision artifact.
  const std::filesystem::path results_path =
      std::filesystem::path("..") / "docs" /
      "forward_drizzle_v2_gate9_streaming_multiband_results_2026-09-12."
      "jsonl";
  nlohmann::json rec;
  rec["case"] = "production_geometry_profiled";
  rec["spec_sha256"] = kSpecSha;
  rec["canvas_native"] = {nc, nr};
  rec["frames"] = n_frames;
  rec["device_bytes_per_native_pixel"] = bpp;
  rec["plan_bytes_per_native_pixel"] = plan.device_bytes_per_target_pixel;
  rec["band_rows"] = band_rows;
  rec["max_frame_seconds"] = st.max_frame_seconds;
  rec["allocations"] = st.allocations;
  rec["global_synchronizations"] = st.device_global_synchronizations;
  rec["stream_synchronizations"] = st.stream_synchronizations;
  rec["primary_pixels"] = supported;
  rec["artifact_applicable_pixels"] = art_applicable;
  rec["dense_overlap"] = dense;
  std::ofstream(results_path, std::ios::app)
      << rec.dump() << "\n";
}

TEST_CASE("forward drizzle v2 gate10 store persists profile artifacts",
          "[gate10][store]") {
  V2StoreFixture fx;
  auto plan = v2_store_plan();
  plan.emit_profiles = true;
  plan.multiband_levels = 2;
  finalize_forward_drizzle_v2_run_plan(plan);

  auto make_profiles = [](std::size_t n, double seed) {
    std::vector<ForwardDrizzleV2ProfileResult> r(n);
    for (std::size_t i = 0; i < n; ++i) {
      r[i].uniform.value = seed + static_cast<double>(i);
      r[i].uniform.weight_sum = 1.0f;
      r[i].uniform.n_eff = 1.0f;
      r[i].uniform.support = 1;
      r[i].raw.value = seed * 2.0f + static_cast<float>(i);
      r[i].raw.weight_sum = 2.0f;
      r[i].fine.value = seed * 0.5f;
      r[i].fine.weight_sum = 0.5f;
      r[i].medium.value = seed * 0.25f;
      r[i].medium.weight_sum = 0.25f;
      r[i].a_separation = 0.3f;
      r[i].a_artifact = 0.2f;
      r[i].a_registration = 0.9f;
      r[i].artifact_applicable = true;
    }
    return r;
  };

  // Roundtrip: committed profile payloads hash-verify and read back
  // identical; checkpoint and commit record both artifacts.
  {
    ForwardDrizzleV2StoreWriter writer(fx.root / "ok", plan);
    writer.begin();
    int y = 0;
    for (int i = 0; i < plan.band_count; ++i) {
      const int rows = std::min(plan.band_rows, plan.native_height - y);
      const std::size_t n = static_cast<std::size_t>(rows) * plan.native_width *
                            plan.channels;
      const auto records = v2_band_records(rows, plan.native_width,
                                           plan.channels, 10.0 + i);
      const auto profiles = make_profiles(n, 1.0 + i);
      writer.commit_band(i, y, rows, records, profiles, 3 + i);
      const auto insp = inspect_forward_drizzle_v2_store(fx.root / "ok", plan);
      REQUIRE(insp.status == ForwardDrizzleV2StoreStatus::resumable);
      const auto &band = insp.committed.back();
      REQUIRE(band.profiles_artifact ==
              "band-" + std::string(4 - std::to_string(i).size(), '0') +
                  std::to_string(i) + ".profiles.bin");
      REQUIRE(band.profiles_bytes ==
              64 + n * sizeof(ForwardDrizzleV2ProfileResult));
      REQUIRE_FALSE(band.profiles_sha256.empty());
      const auto back = read_forward_drizzle_v2_band_profiles(
          writer.generation(), band);
      REQUIRE(back.size() == n);
      REQUIRE(std::memcmp(back.data(), profiles.data(),
                          n * sizeof(profiles[0])) == 0);
      y += rows;
    }
    writer.finish(v2_gate(plan.band_count));
    const auto insp = inspect_forward_drizzle_v2_store(fx.root / "ok", plan);
    REQUIRE(insp.status == ForwardDrizzleV2StoreStatus::complete);
  }

  // emit_profiles plan without payloads rejects the band.
  {
    ForwardDrizzleV2StoreWriter writer(fx.root / "mismatch", plan);
    writer.begin();
    REQUIRE_THROWS(writer.commit_band(
        0, 0, plan.band_rows,
        v2_band_records(plan.band_rows, plan.native_width, plan.channels, 0.0),
        {}, 0));
    // destructor cleans up the unpublished generation
  }

  // Crash after two profiled bands: resume is resumable, missing or
  // tampered profile payload fails closed.
  {
    ForwardDrizzleV2StoreWriter *w = new ForwardDrizzleV2StoreWriter(
        fx.root / "resumed2", plan);
    w->begin();
    int y = 0;
    for (int i = 0; i < 2; ++i) {
      const int rows = std::min(plan.band_rows, plan.native_height - y);
      const std::size_t n = static_cast<std::size_t>(rows) * plan.native_width *
                            plan.channels;
      w->commit_band(i, y, rows,
                     v2_band_records(rows, plan.native_width, plan.channels,
                                     10.0 + i),
                     make_profiles(n, 1.0 + i), 3 + i);
      y += rows;
    }
    const auto generation = w->generation();
    auto insp = inspect_forward_drizzle_v2_store(fx.root / "resumed2", plan);
    REQUIRE(insp.status == ForwardDrizzleV2StoreStatus::resumable);
    REQUIRE(insp.next_band == 2);

    // Missing profile artifact fails closed.
    fs::remove(generation / "band-0000.profiles.bin");
    insp = inspect_forward_drizzle_v2_store(fx.root / "resumed2", plan);
    REQUIRE(insp.status == ForwardDrizzleV2StoreStatus::corrupt);
    delete w;
  }

  // Profile fields present while emit_profiles=false are rejected.
  {
    auto plain = v2_store_plan();
    ForwardDrizzleV2StoreWriter *w = new ForwardDrizzleV2StoreWriter(
        fx.root / "foreign", plain);
    w->begin();
    w->commit_band(0, 0, plain.band_rows,
                   v2_band_records(plain.band_rows, plain.native_width,
                                   plain.channels, 0.0),
                   {}, 0);
    auto insp = inspect_forward_drizzle_v2_store(fx.root / "foreign", plan);
    REQUIRE(insp.status == ForwardDrizzleV2StoreStatus::corrupt);
    delete w;
  }
}

// ---------------------------------------------------------------------------
// Gate 10: host CPU port of the v2 band kernel (forward_drizzle_v2_cpu)
// ---------------------------------------------------------------------------

TEST_CASE("forward drizzle v2 gate10 host kernel affine parity",
          "[forward-drizzle-v2][gate10]") {
  const BayerPattern patterns[] = {BayerPattern::RGGB, BayerPattern::BGGR,
                                   BayerPattern::GRBG, BayerPattern::GBRG};
  for (int scale : {1, 2}) {
    for (ColorMode mode : {ColorMode::MONO, ColorMode::OSC}) {
      for (BayerPattern pattern : patterns) {
        auto f = make_fixture(mode, pattern, 1, -1, scale);
        const int nc = f.plan.canvas_width_native;
        const int nr = f.plan.canvas_height_native;
        const int channels = mode == ColorMode::MONO ? 1 : 3;
        const std::size_t nplane = static_cast<std::size_t>(nc) * nr;
        const std::size_t n_src = static_cast<std::size_t>(
            f.plan.source_width) * f.plan.source_height;
        std::vector<float> s2(n_src);
        for (std::size_t i = 0; i < n_src; ++i)
          s2[i] = 0.01f + 0.0001f * static_cast<float>(i % 97);
        s2[3] = -1.0f;  // degraded-confidence coverage
        ForwardDrizzleV2KernelConfig kcfg;
        kcfg.internal_scale = scale;
        kcfg.stream_length = f.plan.frames.size();
        kcfg.half = 0.5 * f.cfg.pixfrac;
        kcfg.bayer_pattern = static_cast<int>(pattern);
        kcfg.cfa_origin_x = f.plan.cfa_origin_x;
        kcfg.cfa_origin_y = f.plan.cfa_origin_y;
        kcfg.mono = mode == ColorMode::MONO;
        kcfg.canvas_width_native = nc;
        kcfg.canvas_height_native = nr;

        const auto ref = gate6_cpu_reference(f, s2, kcfg);
        ForwardDrizzleV2CpuKernel kernel;
        REQUIRE(kernel.reserve(nc, nr, f.plan.source_width,
                               f.plan.source_height, kcfg));
        for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
          const auto &m = f.plan.frames[fr].source_to_canvas;
          const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                                m(1, 0), m(1, 1), m(1, 2)};
          REQUIRE(kernel.accumulate_frame(a6, f.images[fr].data(), s2.data(),
                                          fr));
        }
        std::vector<ForwardDrizzleV2PixelResult> got(nplane * channels);
        std::uint64_t dense = 0;
        REQUIRE(kernel.finalize(got.data(), nullptr, &dense));
        require_gate6_parity(ref, got, nc, nr, channels);
        REQUIRE(dense == ref.dense_overlap);
        const double subpixels = static_cast<double>(scale) * scale;
        for (std::size_t pc = 0; pc < nplane * channels; ++pc) {
          const unsigned int m = ref.masks[pc];
          const float geo = static_cast<float>(
              static_cast<double>(__builtin_popcount(m & 0xFu)) / subpixels);
          const float src = static_cast<float>(
              static_cast<double>(__builtin_popcount((m >> 4) & 0xFu)) /
              subpixels);
          REQUIRE(got[pc].geometry_fraction == geo);
          REQUIRE(got[pc].source_fraction == src);
          REQUIRE(got[pc].estimator_fraction == src);
          REQUIRE(got[pc].profile_fraction == src);
        }
        const auto &st = kernel.stats();
        REQUIRE(st.allocations == 1);
        REQUIRE(st.device_global_synchronizations == 0);
        REQUIRE(st.stream_synchronizations == 1);
        REQUIRE(st.frames_processed == f.plan.frames.size());
        REQUIRE(st.positive_overlaps > 0);
        REQUIRE(st.candidates_streamed > 0);
        REQUIRE(st.reservoir_kept_total > 0);
        REQUIRE(st.slot_transitions == f.plan.frames.size());
        // Bounded-memory accounting: reserved host bytes per native pixel
        // must fit the gate-5 plan budget comfortably.
        REQUIRE(kernel.host_bytes_per_native_pixel() > 0);
        REQUIRE(kernel.host_bytes_per_native_pixel() < 20000);
      }
    }
  }
}

TEST_CASE("forward drizzle v2 gate10 host kernel reservoir sampling N > R",
          "[forward-drizzle-v2][gate10]") {
  auto f = make_fixture(ColorMode::OSC, BayerPattern::GBRG, 0, 0, 2);
  const std::size_t keep = f.plan.frames.size();
  for (std::size_t i = keep; i < 80; ++i) {
    FrameSamplingTransform frame = f.plan.frames[i % keep];
    frame.frame_id = "v2-extra-" + std::to_string(i);
    frame.source_index = i;
    frame.source_to_canvas(0, 2) += static_cast<float>(0.01 * (i - keep));
    f.plan.frames.push_back(frame);
    f.images.push_back(f.images[i % keep]);
  }
  const int nc = f.plan.canvas_width_native;
  const int nr = f.plan.canvas_height_native;
  const std::size_t nplane = static_cast<std::size_t>(nc) * nr;
  ForwardDrizzleV2KernelConfig kcfg;
  kcfg.internal_scale = 2;
  kcfg.stream_length = f.plan.frames.size();
  kcfg.half = 0.5 * f.cfg.pixfrac;
  kcfg.bayer_pattern = static_cast<int>(BayerPattern::GBRG);
  kcfg.mono = false;
  const auto ref = gate6_cpu_reference(f, {}, kcfg);
  ForwardDrizzleV2CpuKernel kernel;
  REQUIRE(kernel.reserve(nc, nr, f.plan.source_width, f.plan.source_height,
                         kcfg));
  for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
    const auto &m = f.plan.frames[fr].source_to_canvas;
    const double a6[6] = {m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1), m(1, 2)};
    REQUIRE(kernel.accumulate_frame(a6, f.images[fr].data(), nullptr, fr));
  }
  std::vector<ForwardDrizzleV2PixelResult> got(nplane * 3);
  std::uint64_t dense = 0;
  REQUIRE(kernel.finalize(got.data(), nullptr, &dense));
  require_gate6_parity(ref, got, nc, nr, 3);
  REQUIRE(dense == ref.dense_overlap);
  REQUIRE(kernel.stats().reservoir_kept_total <=
          kernel.stats().candidates_streamed);
  REQUIRE(kernel.stats().reservoir_kept_total > 0);
}

TEST_CASE("forward drizzle v2 selected frame orders are the exact keep set",
          "[forward-drizzle-v2]") {
  // Golden keep set for the shared splitmix64 predicate.
  const std::vector<std::uint64_t> want{
      9,   35,  46,  48,  54,  60,  71,  74,  89,  95,  104, 108, 109,
      111, 115, 143, 152, 153, 163, 190, 205, 207, 219, 220, 234, 252,
      254, 275, 290, 295, 302, 304, 306, 313, 314, 320, 331, 334, 343,
      350, 355, 361, 373, 389, 390, 400, 401, 404, 409, 479, 492, 513,
      519, 531, 545, 551, 596, 605};
  REQUIRE(forward_drizzle_v2_selected_frame_orders(
              610, 64, 11400714819323198485ULL) == want);

  // N <= R selects every order, ascending.
  const auto all = forward_drizzle_v2_selected_frame_orders(
      5, 64, 0x9e3779b97f4a7c15ULL);
  REQUIRE(all == std::vector<std::uint64_t>{0, 1, 2, 3, 4});

  REQUIRE_THROWS_AS(forward_drizzle_v2_selected_frame_orders(0, 64, 0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(forward_drizzle_v2_selected_frame_orders(10, 0, 0),
                    std::invalid_argument);
}

TEST_CASE("forward drizzle v2 keeps the historical 2R overflow fallback",
          "[forward-drizzle-v2]") {
  // Rare keep sets can exceed 2*R; the slot count must stay at the
  // historical cap so the kept > slots overflow fallback survives.
  // Bounded deterministic seed search for a (N=3, R=1) keep set > 2.
  std::uint64_t seed = 0;
  std::vector<std::uint64_t> sel;
  for (; seed < 4096; ++seed) {
    sel = forward_drizzle_v2_selected_frame_orders(3, 1, seed);
    if (sel.size() > 2) break;
  }
  REQUIRE(sel.size() == 3);

  auto f = make_fixture(ColorMode::MONO, BayerPattern::RGGB, 0, 0, 1);
  const int nc = f.plan.canvas_width_native;
  const int nr = f.plan.canvas_height_native;
  const std::size_t nplane = static_cast<std::size_t>(nc) * nr;
  ForwardDrizzleV2KernelConfig kcfg;
  kcfg.internal_scale = 1;
  kcfg.reservoir_size = 1;
  kcfg.reservoir_seed = seed;
  kcfg.stream_length = 3;
  kcfg.half = 0.5 * f.cfg.pixfrac;
  kcfg.bayer_pattern = static_cast<int>(BayerPattern::RGGB);
  kcfg.mono = true;
  kcfg.sigma2_plane = false;

  ForwardDrizzleV2CpuKernel kernel;
  REQUIRE(kernel.reserve(nc, nr, f.plan.source_width, f.plan.source_height,
                         kcfg));
  // Planned slots remain the historical 2*R.
  REQUIRE(kernel.host_bytes_per_native_pixel() ==
          forward_drizzle_v2_cpu_bytes_per_native_pixel(1, 1, 2, false,
                                                      false));
  // Three identical frames: every covered pixel collects all three kept
  // candidates and overflows the two slots.
  const auto &m = f.plan.frames[0].source_to_canvas;
  const double a6[6] = {m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1), m(1, 2)};
  for (std::uint64_t fr = 0; fr < 3; ++fr)
    REQUIRE(kernel.accumulate_frame(a6, f.images[0].data(), nullptr, fr));
  std::vector<ForwardDrizzleV2PixelResult> got(nplane);
  std::uint64_t dense = 0;
  REQUIRE(kernel.finalize(got.data(), nullptr, &dense));
  bool saw_overflow = false;
  for (const auto &r : got) {
    if (r.contributors == 0) continue;
    REQUIRE(r.robust_state ==
            static_cast<std::uint8_t>(
                ForwardDrizzleV2RobustState::reservoir_overflow_fallback));
    saw_overflow = true;
  }
  REQUIRE(saw_overflow);
}

TEST_CASE("forward drizzle v2 gate10 host kernel mixed local stream parity",
          "[forward-drizzle-v2][gate10]") {
  for (int scale : {1, 2}) {
    for (ColorMode mode : {ColorMode::MONO, ColorMode::OSC}) {
      auto f = make_fixture(mode, BayerPattern::GBRG, 0, 0, scale);
      const int nc = f.plan.canvas_width_native;
      const int nr = f.plan.canvas_height_native;
      const int channels = mode == ColorMode::MONO ? 1 : 3;
      const std::size_t nplane = static_cast<std::size_t>(nc) * nr;
      const std::size_t n_src = static_cast<std::size_t>(
          f.plan.source_width) * f.plan.source_height;
      std::vector<float> s2(n_src);
      for (std::size_t i = 0; i < n_src; ++i)
        s2[i] = 0.01f + 0.0001f * static_cast<float>(i % 97);

      float cx1[16] = {}, cy1[16] = {}, cx3[16] = {}, cy3[16] = {};
      cx1[5] = 1.3f;
      cy1[6] = -0.8f;
      cx3[9] = -1.7f;
      cy3[10] = 1.1f;
      f.plan.frames[1] = v2_local_frame(
          f.plan.frames[1], v2_local_model(nr, nc, cx1, cy1));
      f.plan.frames[3] = v2_local_frame(
          f.plan.frames[3], v2_local_model(nr, nc, cx3, cy3));

      ForwardDrizzleV2KernelConfig kcfg;
      kcfg.internal_scale = scale;
      kcfg.stream_length = f.plan.frames.size();
      kcfg.half = 0.5 * f.cfg.pixfrac;
      kcfg.bayer_pattern = static_cast<int>(f.plan.bayer_pattern);
      kcfg.cfa_origin_x = f.plan.cfa_origin_x;
      kcfg.cfa_origin_y = f.plan.cfa_origin_y;
      kcfg.mono = mode == ColorMode::MONO;
      kcfg.canvas_width_native = nc;
      kcfg.canvas_height_native = nr;

      const auto ref = gate8_cpu_reference(f, s2, kcfg);
      std::uint64_t expected_discards = 0;
      for (const auto &fr : f.plan.frames) {
        if (!fr.has_smooth_local_model) continue;
        expected_discards += v2_frame_planes_cpu(
                                 f.plan, fr, f.images[fr.source_index],
                                 scale, f.cfg.pixfrac, channels,
                                 f.plan.bayer_pattern, f.plan.cfa_origin_x,
                                 f.plan.cfa_origin_y)
                                 .discarded;
      }

      ForwardDrizzleV2CpuKernel kernel;
      REQUIRE(kernel.reserve(nc, nr, f.plan.source_width,
                             f.plan.source_height, kcfg));
      for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
        const auto &frame = f.plan.frames[fr];
        const auto &m = frame.source_to_canvas;
        const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                              m(1, 0), m(1, 1), m(1, 2)};
        if (frame.has_smooth_local_model) {
          REQUIRE(kernel.accumulate_frame_local(
              a6, v2_warp_descriptor(frame), f.images[fr].data(),
              s2.data(), fr));
        } else {
          REQUIRE(kernel.accumulate_frame(a6, f.images[fr].data(),
                                          s2.data(), fr));
        }
      }
      std::vector<ForwardDrizzleV2PixelResult> got(nplane * channels);
      std::uint64_t dense = 0;
      REQUIRE(kernel.finalize(got.data(), nullptr, &dense));
      // The host kernel runs the SAME inversion/subdivision functions as the
      // oracle, so magnitudes match far tighter than the device tolerance.
      require_gate6_parity(ref, got, nc, nr, channels);
      REQUIRE(dense == ref.dense_overlap);
      REQUIRE(kernel.stats().local_samples_discarded == expected_discards);
      REQUIRE(kernel.stats().allocations == 1);
    }
  }
}

TEST_CASE("forward drizzle v2 gate10 host kernel band windows compose",
          "[forward-drizzle-v2][gate10]") {
  // Banding contract: identical canvas-coords affine6 + band_origin selects
  // the band window; per-band records must equal the whole-canvas run's
  // rows (discrete fields bit-exact, magnitudes within 1e-12).
  for (int scale : {1, 2}) {
    for (ColorMode mode : {ColorMode::MONO, ColorMode::OSC}) {
      auto f = make_fixture(mode, BayerPattern::GRBG, 0, 0, scale);
      const int nc = f.plan.canvas_width_native;
      const int nr = f.plan.canvas_height_native;
      const int channels = mode == ColorMode::MONO ? 1 : 3;
      const std::size_t nplane = static_cast<std::size_t>(nc) * nr;
      const std::size_t n_src = static_cast<std::size_t>(
          f.plan.source_width) * f.plan.source_height;
      std::vector<float> s2(n_src, 0.02f);

      // One local frame so the band test covers the canvas-coords
      // inversion + shifted emit path.
      float cx[16] = {}, cy[16] = {};
      cx[5] = 1.2f;
      cy[6] = -0.7f;
      f.plan.frames[2] = v2_local_frame(
          f.plan.frames[2], v2_local_model(nr, nc, cx, cy));

      auto make_kcfg = [&](int y_begin, int rows) {
        ForwardDrizzleV2KernelConfig k;
        k.internal_scale = scale;
        k.stream_length = f.plan.frames.size();
        k.half = 0.5 * f.cfg.pixfrac;
        k.bayer_pattern = static_cast<int>(f.plan.bayer_pattern);
        k.cfa_origin_x = f.plan.cfa_origin_x;
        k.cfa_origin_y = f.plan.cfa_origin_y;
        k.mono = mode == ColorMode::MONO;
        k.canvas_width_native = nc;
        k.canvas_height_native = nr;
        k.band_origin_y_native = y_begin;
        return k;
      };
      auto run = [&](int y_begin, int rows) {
        ForwardDrizzleV2CpuKernel kernel;
        REQUIRE(kernel.reserve(nc, rows, f.plan.source_width,
                               f.plan.source_height,
                               make_kcfg(y_begin, rows)));
        for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
          const auto &frame = f.plan.frames[fr];
          const auto &m = frame.source_to_canvas;
          const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                                m(1, 0), m(1, 1), m(1, 2)};
          if (frame.has_smooth_local_model)
            REQUIRE(kernel.accumulate_frame_local(
                a6, v2_warp_descriptor(frame), f.images[fr].data(),
                s2.data(), fr));
          else
            REQUIRE(kernel.accumulate_frame(a6, f.images[fr].data(),
                                            s2.data(), fr));
        }
        std::vector<ForwardDrizzleV2PixelResult> out(
            static_cast<std::size_t>(rows) * nc * channels);
        std::uint64_t dense = 0;
        REQUIRE(kernel.finalize(out.data(), nullptr, &dense));
        return out;
      };

      const auto whole = run(0, nr);
      // Resume contract: recomputing the same band with the same plan
      // yields bitwise-identical records (field-wise to skip padding).
      const auto band1a = run(5, 5);
      const auto band1b = run(5, 5);
      auto bit_eq = [](const auto &a, const auto &b) {
        return std::memcmp(&a, &b, sizeof(a)) == 0;
      };
      for (std::size_t i = 0; i < band1a.size(); ++i) {
        const auto &a = band1a[i];
        const auto &b = band1b[i];
        REQUIRE(bit_eq(a.value, b.value));
        REQUIRE(bit_eq(a.b, b.b));
        REQUIRE(bit_eq(a.n_eff, b.n_eff));
        REQUIRE(bit_eq(a.confidence, b.confidence));
        REQUIRE(bit_eq(a.geometry_fraction, b.geometry_fraction));
        REQUIRE(bit_eq(a.source_fraction, b.source_fraction));
        REQUIRE(bit_eq(a.estimator_fraction, b.estimator_fraction));
        REQUIRE(bit_eq(a.profile_fraction, b.profile_fraction));
        REQUIRE(bit_eq(a.contributors, b.contributors));
        REQUIRE(bit_eq(a.robust_state, b.robust_state));
        REQUIRE(bit_eq(a.confidence_state, b.confidence_state));
        REQUIRE(bit_eq(a.conf_degraded, b.conf_degraded));
      }

      // Whole-vs-band: all discrete fields bit-exact and the full-stream
      // magnitudes within 1e-12. The clip-accepted mean `value` may flip a
      // boundary candidate because the banded emit shifts corners before
      // scaling (one ulp); count and bound those flips.
      std::uint64_t compared = 0, flips = 0;
      int y = 0;
      int band_index = 0;
      while (y < nr) {
        const int rows = std::min(5, nr - y);
        const auto band = run(y, rows);
        for (int r = 0; r < rows; ++r)
          for (int x = 0; x < nc; ++x)
            for (int c = 0; c < channels; ++c) {
              const std::size_t pband =
                  static_cast<std::size_t>(c) * rows * nc +
                  static_cast<std::size_t>(r) * nc + x;
              const std::size_t pwhole =
                  static_cast<std::size_t>(c) * nplane +
                  static_cast<std::size_t>(y + r) * nc + x;
              const auto &e = whole[pwhole];
              const auto &g = band[pband];
              ++compared;
              // A clip-boundary candidate may flip because the banded emit
              // shifts corners before scaling (one ulp). On a flip, both
              // `value` and `confidence` (accepted-set Gate-4) move; all
              // discrete fields and full-stream magnitudes must stay exact.
              const double vd = std::fabs(static_cast<double>(g.value) -
                                          static_cast<double>(e.value));
              const double cd =
                  std::fabs(static_cast<double>(g.confidence) -
                            static_cast<double>(e.confidence));
              const bool flip = vd > 1e-12 || cd > 1e-12;
              if (flip)
                INFO("clip flip band=" << band_index << " r=" << r
                                       << " x=" << x << " c=" << c);
              REQUIRE(g.robust_state == e.robust_state);
              REQUIRE(g.confidence_state == e.confidence_state);
              REQUIRE(g.contributors == e.contributors);
              REQUIRE(g.conf_degraded == e.conf_degraded);
              REQUIRE(g.geometry_fraction == e.geometry_fraction);
              REQUIRE(g.source_fraction == e.source_fraction);
              REQUIRE(g.b ==
                      Catch::Approx(e.b).epsilon(1e-12).margin(1e-12));
              REQUIRE(g.n_eff ==
                      Catch::Approx(e.n_eff).epsilon(1e-12).margin(1e-12));
              if (!flip) {
                REQUIRE(g.value ==
                        Catch::Approx(e.value).epsilon(1e-12).margin(1e-12));
                REQUIRE(g.confidence ==
                        Catch::Approx(e.confidence).epsilon(1e-12)
                            .margin(1e-12));
              } else {
                REQUIRE(g.robust_state ==
                        static_cast<std::uint8_t>(
                            ForwardDrizzleV2RobustState::
                                primary_reservoir_sigma_clip));
                REQUIRE(vd <=
                        0.1 * std::fabs(static_cast<double>(e.value)));
                ++flips;
              }
            }
        y += rows;
        ++band_index;
      }
      REQUIRE(band_index == 3);
      REQUIRE(flips <= compared / 200 + 1);
    }
  }
}

TEST_CASE("forward drizzle v2 gate10 host kernel profile parity",
          "[forward-drizzle-v2][gate10]") {
  const BayerPattern patterns[] = {BayerPattern::GRBG, BayerPattern::GBRG};
  for (int scale : {1, 2}) {
    for (ColorMode mode : {ColorMode::MONO, ColorMode::OSC}) {
      for (BayerPattern pattern : patterns) {
        auto f = make_fixture(mode, pattern, 1, -1, scale);
        const std::size_t base_frames = f.plan.frames.size();
        for (std::size_t i = base_frames; i < 12; ++i) {
          FrameSamplingTransform frame = f.plan.frames[i % base_frames];
          frame.frame_id = "v2-g10-" + std::to_string(i);
          frame.source_index = i;
          frame.source_to_canvas(0, 2) +=
              static_cast<float>(0.011 * static_cast<double>(i - base_frames));
          f.plan.frames.push_back(frame);
          f.images.push_back(f.images[i % base_frames]);
        }
        const int nc = f.plan.canvas_width_native;
        const int nr = f.plan.canvas_height_native;
        const int channels = mode == ColorMode::MONO ? 1 : 3;
        const std::size_t nplane = static_cast<std::size_t>(nc) * nr;
        const int n_frames = static_cast<int>(f.plan.frames.size());

        std::vector<Matrix2Df> qc(n_frames), q0(n_frames), q1(n_frames),
            qa(n_frames);
        std::vector<ForwardDrizzleV2FrameMeta> meta(n_frames);
        for (int fr = 0; fr < n_frames; ++fr) {
          qc[fr].resize(f.plan.source_height, f.plan.source_width);
          q0[fr].resize(f.plan.source_height, f.plan.source_width);
          q1[fr].resize(f.plan.source_height, f.plan.source_width);
          qa[fr].resize(f.plan.source_height, f.plan.source_width);
          for (int yy = 0; yy < f.plan.source_height; ++yy)
            for (int xx = 0; xx < f.plan.source_width; ++xx) {
              const std::size_t si =
                  static_cast<std::size_t>(yy) * f.plan.source_width + xx;
              const double s = static_cast<double>((si + 7 * fr) % 61) / 61.0;
              qc[fr](yy, xx) = static_cast<float>(0.4 + 0.6 * s);
              q0[fr](yy, xx) = static_cast<float>(0.3 + 0.7 * ((s * fr) -
                                  std::floor(s * fr)));
              q1[fr](yy, xx) = static_cast<float>(0.5 + 0.4 * s);
              qa[fr](yy, xx) = static_cast<float>(0.2 + 0.6 * s);
            }
          qc[fr](0, 0) = std::numeric_limits<float>::quiet_NaN();
          if (f.plan.source_width > 1) qc[fr](0, 1) = -1.0f;
          qa[fr](1 % f.plan.source_height, 0) =
              std::numeric_limits<float>::quiet_NaN();
          meta[fr] = {static_cast<float>(0.6 + 0.05 * (fr % 5)),
                      static_cast<float>(0.55 + 0.05 * fr),
                      static_cast<std::uint8_t>(fr % 2), 0};
        }

        ForwardDrizzleV2KernelConfig kcfg;
        kcfg.internal_scale = scale;
        kcfg.stream_length = n_frames;
        kcfg.half = 0.5 * f.cfg.pixfrac;
        kcfg.bayer_pattern = static_cast<int>(pattern);
        kcfg.cfa_origin_x = f.plan.cfa_origin_x;
        kcfg.cfa_origin_y = f.plan.cfa_origin_y;
        kcfg.mono = mode == ColorMode::MONO;
        kcfg.emit_profiles = true;

        ForwardDrizzleV2CpuKernel kernel;
        REQUIRE(kernel.reserve(nc, nr, f.plan.source_width,
                               f.plan.source_height, kcfg));
        for (int fr = 0; fr < n_frames; ++fr) {
          const auto &m = f.plan.frames[fr].source_to_canvas;
          const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                                m(1, 0), m(1, 1), m(1, 2)};
          ForwardDrizzleV2FrameQuality q;
          q.q_composite = qc[fr].data();
          q.q_scale0 = q0[fr].data();
          q.q_scale1 = q1[fr].data();
          q.q_artifact = fr == 2 ? nullptr : qa[fr].data();
          REQUIRE(kernel.accumulate_frame(a6, f.images[fr].data(), nullptr,
                                          static_cast<std::uint64_t>(fr), &q,
                                          &meta[fr]));
        }
        std::vector<ForwardDrizzleV2PixelResult> got(nplane * channels);
        std::vector<ForwardDrizzleV2ProfileResult> gp(nplane * channels);
        std::uint64_t dense = 0;
        REQUIRE(kernel.finalize(got.data(), gp.data(), &dense));

        const int ic = nc * scale, ir = nr * scale;
        const double inv_s2 = 1.0 / (static_cast<double>(scale) * scale);
        std::vector<std::vector<ForwardDrizzleV2ProfileCandidate>> cands(
            nplane * channels);
        std::vector<std::vector<ForwardDrizzleV2RobustCandidate>> rcands(
            nplane * channels);
        for (int fr = 0; fr < n_frames; ++fr) {
          RegistrationSamplingPlan one = f.plan;
          one.frames = {f.plan.frames[fr]};
          const auto &img = f.images[fr];
          auto real_of = [&](std::size_t) -> const Matrix2Df & { return img; };
          const auto g_a =
              gather_affine_uniform_v2(one, real_of, f.cfg, 0, ir);
          Matrix2Df cqc(f.plan.source_height, f.plan.source_width),
              cq0(f.plan.source_height, f.plan.source_width),
              cq1(f.plan.source_height, f.plan.source_width),
              cqa(f.plan.source_height, f.plan.source_width),
              qaf(f.plan.source_height, f.plan.source_width);
          for (int yy = 0; yy < f.plan.source_height; ++yy)
            for (int xx = 0; xx < f.plan.source_width; ++xx) {
              const bool finite_src = std::isfinite(img(yy, xx));
              auto cl = [](float v) {
                return std::isfinite(v) && v > 0.0f ? v : 0.0f;
              };
              cqc(yy, xx) = finite_src ? cl(qc[fr](yy, xx)) : 0.0f;
              cq0(yy, xx) = finite_src ? cl(q0[fr](yy, xx)) : 0.0f;
              cq1(yy, xx) = finite_src ? cl(q1[fr](yy, xx)) : 0.0f;
              cqa(yy, xx) = finite_src ? cl(qa[fr](yy, xx)) : 0.0f;
              qaf(yy, xx) =
                  finite_src && std::isfinite(qa[fr](yy, xx)) ? 1.0f : 0.0f;
            }
          auto qc_of = [&](std::size_t) -> const Matrix2Df & { return cqc; };
          auto q0_of = [&](std::size_t) -> const Matrix2Df & { return cq0; };
          auto q1_of = [&](std::size_t) -> const Matrix2Df & { return cq1; };
          auto qa_of = [&](std::size_t) -> const Matrix2Df & { return cqa; };
          auto qaf_of = [&](std::size_t) -> const Matrix2Df & { return qaf; };
          const auto g_qc = gather_affine_uniform_v2(one, qc_of, f.cfg, 0, ir);
          const auto g_q0 = gather_affine_uniform_v2(one, q0_of, f.cfg, 0, ir);
          const auto g_q1 = gather_affine_uniform_v2(one, q1_of, f.cfg, 0, ir);
          const auto g_qa = gather_affine_uniform_v2(one, qa_of, f.cfg, 0, ir);
          const auto g_qaf =
              gather_affine_uniform_v2(one, qaf_of, f.cfg, 0, ir);
          const bool qa_present = fr != 2;
          for (std::size_t px = 0; px < nplane; ++px) {
            const int nx = static_cast<int>(px % nc);
            const int ny = static_cast<int>(px / nc);
            for (int c = 0; c < channels; ++c) {
              const std::size_t pc =
                  static_cast<std::size_t>(c) * nplane + px;
              double a = 0.0, bs = 0.0, sq = 0.0, s0 = 0.0, s1 = 0.0,
                     sa = 0.0, saf = 0.0;
              for (int iy = 0; iy < scale; ++iy)
                for (int ix = 0; ix < scale; ++ix) {
                  const std::size_t ii =
                      static_cast<std::size_t>(ny * scale + iy) * ic +
                      nx * scale + ix;
                  a += inv_s2 * g_a.accum.wx[c][ii];
                  bs += inv_s2 * g_a.accum.w[c][ii];
                  sq += inv_s2 * g_qc.accum.wx[c][ii];
                  s0 += inv_s2 * g_q0.accum.wx[c][ii];
                  s1 += inv_s2 * g_q1.accum.wx[c][ii];
                  sa += inv_s2 * g_qa.accum.wx[c][ii];
                  saf += inv_s2 * g_qaf.accum.wx[c][ii];
                }
              if (!(bs > 0.0)) continue;
              rcands[pc].push_back({static_cast<std::size_t>(fr), a / bs, bs});
              ForwardDrizzleV2ProfileCandidate cd;
              cd.frame_order = fr;
              cd.x = a / bs;
              cd.b = bs;
              cd.q = sq / bs;
              cd.q0 = s0 / bs;
              cd.q1 = s1 / bs;
              cd.qa = qa_present ? sa / bs : 1.0;
              cd.qa_has_data = qa_present && saf > 0.0;
              cands[pc].push_back(cd);
            }
          }
        }

        ForwardDrizzleV2RobustConfig rcfg;
        rcfg.reservoir_size = kcfg.reservoir_size;
        rcfg.reservoir_seed = kcfg.reservoir_seed;
        rcfg.oracle_min_clip_contributors = kcfg.min_clip_contributors;
        rcfg.oracle_passes = kcfg.robust_passes;
        rcfg.oracle_sigma_low = kcfg.sigma_low;
        rcfg.oracle_sigma_high = kcfg.sigma_high;
        rcfg.min_candidates = kcfg.min_candidates;
        ForwardDrizzleV2ProfileConfig pcfg;
        pcfg.fine_quality_exponent = kcfg.fine_quality_exponent;
        pcfg.medium_quality_exponent = kcfg.medium_quality_exponent;
        bool saw_primary = false, saw_fallback = false,
             saw_artifact_applicable = false;
        for (std::size_t pc = 0; pc < nplane * channels; ++pc) {
          const auto &r = got[pc];
          if (r.robust_state ==
              static_cast<std::uint8_t>(
                  ForwardDrizzleV2RobustState::no_source_support)) {
            REQUIRE(gp[pc].uniform.support == 0);
            continue;
          }
          if (r.robust_state !=
              static_cast<std::uint8_t>(
                  ForwardDrizzleV2RobustState::primary_reservoir_sigma_clip)) {
            saw_fallback = true;
            const float v = static_cast<float>(r.value);
            for (const auto *o :
                 {&gp[pc].uniform, &gp[pc].raw, &gp[pc].fine,
                  &gp[pc].medium}) {
              REQUIRE(o->support == 1);
              REQUIRE(o->value == Catch::Approx(v).margin(1e-4));
            }
            REQUIRE(gp[pc].a_separation == 0.0f);
            REQUIRE(gp[pc].a_artifact == 0.0f);
            REQUIRE(gp[pc].a_registration == 0.0f);
            continue;
          }
          saw_primary = true;
          const auto mask = g9_clip_mask(
              rcands[pc], kcfg.min_clip_contributors, kcfg.robust_passes,
              kcfg.sigma_low, kcfg.sigma_high);
          const auto ref = robust_reduce_candidates_v2(
              rcands[pc], ForwardDrizzleV2Estimator::reservoir_sigma_clip,
              rcfg, static_cast<std::uint64_t>(n_frames), {});
          const auto expected = forward_drizzle_v2_profile_reduce(
              cands[pc], mask, meta, pcfg, ref.confidence,
              ref.candidates > 0 && ref.conf_degraded == ref.candidates);
          require_g9_profile_parity(expected, gp[pc], pc);
          if (gp[pc].artifact_applicable) saw_artifact_applicable = true;
        }
        REQUIRE(saw_primary);
        REQUIRE(saw_fallback);
        REQUIRE(saw_artifact_applicable);
      }
    }
  }
}

TEST_CASE("forward drizzle v2 gate10 host kernel matches the device",
          "[forward-drizzle-v2][cuda-parity][gate10]") {
  if (!forward_drizzle_cuda_runtime_available()) {
    SUCCEED("CUDA device unavailable");
    return;
  }
  const BayerPattern patterns[] = {BayerPattern::GRBG, BayerPattern::GBRG};
  for (int scale : {1, 2}) {
    for (ColorMode mode : {ColorMode::MONO, ColorMode::OSC}) {
      for (BayerPattern pattern : patterns) {
        auto f = make_fixture(mode, pattern, 1, -1, scale);
        // N > R stream so the reservoir keep set is a strict subset.
        const std::size_t keep_n = f.plan.frames.size();
        for (std::size_t i = keep_n; i < 80; ++i) {
          FrameSamplingTransform frame = f.plan.frames[i % keep_n];
          frame.frame_id = "v2-extra-" + std::to_string(i);
          frame.source_index = i;
          frame.source_to_canvas(0, 2) +=
              static_cast<float>(0.01 * (i - keep_n));
          f.plan.frames.push_back(frame);
          f.images.push_back(f.images[i % keep_n]);
        }
        const int nc = f.plan.canvas_width_native;
        const int nr = f.plan.canvas_height_native;
        const int channels = mode == ColorMode::MONO ? 1 : 3;
        const std::size_t nplane = static_cast<std::size_t>(nc) * nr;
        const std::size_t n_src = static_cast<std::size_t>(
            f.plan.source_width) * f.plan.source_height;
        std::vector<float> s2(n_src);
        for (std::size_t i = 0; i < n_src; ++i)
          s2[i] = 0.01f + 0.0001f * static_cast<float>(i % 97);
        s2[3] = -1.0f;

        // Local warp on one frame covers the CPU/device local path too.
        float cx[16] = {}, cy[16] = {};
        cx[5] = 1.1f;
        cy[9] = -0.6f;
        f.plan.frames[1] = v2_local_frame(
            f.plan.frames[1], v2_local_model(nr, nc, cx, cy));

        // Nontrivial quality: four streams present on every frame.
        std::vector<float> qc(n_src), q0(n_src), q1(n_src), qa(n_src);
        for (std::size_t i = 0; i < n_src; ++i) {
          qc[i] = 0.5f + 0.25f * static_cast<float>(i % 7);
          q0[i] = 0.8f;
          q1[i] = 0.6f;
          qa[i] = static_cast<float>(i % 11) * 0.1f;
        }
        ForwardDrizzleV2FrameQuality quality;
        quality.q_composite = qc.data();
        quality.q_scale0 = q0.data();
        quality.q_scale1 = q1.data();
        quality.q_artifact = qa.data();

        ForwardDrizzleV2KernelConfig kcfg;
        kcfg.internal_scale = scale;
        kcfg.stream_length = f.plan.frames.size();
        kcfg.half = 0.5 * f.cfg.pixfrac;
        kcfg.bayer_pattern = static_cast<int>(pattern);
        kcfg.cfa_origin_x = f.plan.cfa_origin_x;
        kcfg.cfa_origin_y = f.plan.cfa_origin_y;
        kcfg.mono = mode == ColorMode::MONO;
        kcfg.canvas_width_native = nc;
        kcfg.canvas_height_native = nr;
        kcfg.emit_profiles = true;

        std::vector<ForwardDrizzleV2FrameMeta> meta(f.plan.frames.size());
        for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr)
          meta[fr] = {0.8f + 0.05f * static_cast<float>(fr % 3),
                      0.7f + 0.04f * static_cast<float>(fr),
                      static_cast<std::uint8_t>(fr % 2), 0};

        auto run = [&](ForwardDrizzleV2Kernel &kernel,
                       std::vector<ForwardDrizzleV2PixelResult> &out,
                       std::vector<ForwardDrizzleV2ProfileResult> &pout) {
          REQUIRE(kernel.reserve(nc, nr, f.plan.source_width,
                                 f.plan.source_height, kcfg));
          for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
            const auto &frame = f.plan.frames[fr];
            const auto &m = frame.source_to_canvas;
            const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                                  m(1, 0), m(1, 1), m(1, 2)};
            if (frame.has_smooth_local_model)
              REQUIRE(kernel.accumulate_frame_local(
                  a6, v2_warp_descriptor(frame), f.images[fr].data(),
                  s2.data(), fr, &quality, &meta[fr]));
            else
              REQUIRE(kernel.accumulate_frame(a6, f.images[fr].data(),
                                              s2.data(), fr, &quality,
                                              &meta[fr]));
          }
          std::uint64_t dense = 0;
          REQUIRE(kernel.finalize(out.data(), pout.data(), &dense));
          return dense;
        };

        ForwardDrizzleV2CpuKernel cpu;
        ForwardDrizzleV2CudaKernel gpu;
        std::vector<ForwardDrizzleV2PixelResult> hc(nplane * channels),
            dev(nplane * channels);
        std::vector<ForwardDrizzleV2ProfileResult> hcp(nplane * channels),
            devp(nplane * channels);
        const auto dense_c = run(cpu, hc, hcp);
        const auto dense_g = run(gpu, dev, devp);
        REQUIRE(dense_c == dense_g);
        REQUIRE(cpu.stats().local_samples_discarded ==
                gpu.stats().local_samples_discarded);
        // Quality work ran exactly for the deterministic keep set on both
        // backends.
        const auto selected = forward_drizzle_v2_selected_frame_orders(
            f.plan.frames.size(), kcfg.reservoir_size,
            kcfg.reservoir_seed);
        REQUIRE(selected.size() < f.plan.frames.size());
        REQUIRE(cpu.stats().quality_frames_processed == selected.size());
        REQUIRE(gpu.stats().quality_frames_processed == selected.size());
        REQUIRE(cpu.stats().quality_bytes_uploaded ==
                selected.size() * 4 * n_src * sizeof(float));
        REQUIRE(gpu.stats().quality_bytes_uploaded ==
                selected.size() * 4 * n_src * sizeof(float));
        REQUIRE(cpu.stats().source_samples_launched ==
                f.plan.frames.size() * n_src);
        REQUIRE(gpu.stats().source_samples_launched ==
                f.plan.frames.size() * n_src);
        for (std::size_t pc = 0; pc < nplane * channels; ++pc) {
          const auto &h = hc[pc];
          const auto &d = dev[pc];
          // The cpu_fallback acceptance gate: discrete state bit-identical;
          // magnitudes at the gate-8 local tolerance (both sides run fp32
          // inversion; affine-only diffs are far tighter, proven at 1e-12
          // against the fp64 oracle above).
          REQUIRE(h.robust_state == d.robust_state);
          REQUIRE(h.confidence_state == d.confidence_state);
          REQUIRE(h.contributors == d.contributors);
          REQUIRE(h.conf_degraded == d.conf_degraded);
          REQUIRE(h.geometry_fraction == d.geometry_fraction);
          REQUIRE(h.source_fraction == d.source_fraction);
          REQUIRE(h.value ==
                  Catch::Approx(d.value).epsilon(1e-5).margin(1e-9));
          REQUIRE(h.b == Catch::Approx(d.b).epsilon(1e-5).margin(1e-9));
          REQUIRE(h.n_eff ==
                  Catch::Approx(d.n_eff).epsilon(1e-5).margin(1e-9));
          REQUIRE(h.confidence ==
                  Catch::Approx(d.confidence).epsilon(1e-5).margin(1e-9));
          const auto &hp = hcp[pc];
          const auto &dp = devp[pc];
          for (const auto &pair :
               {std::pair{&hp.uniform, &dp.uniform},
                std::pair{&hp.raw, &dp.raw},
                std::pair{&hp.fine, &dp.fine},
                std::pair{&hp.medium, &dp.medium}}) {
            REQUIRE(pair.first->support == pair.second->support);
            if (!pair.first->support) continue;
            REQUIRE(pair.first->value ==
                    Catch::Approx(pair.second->value)
                        .epsilon(1e-6)
                        .margin(1e-6));
            REQUIRE(pair.first->weight_sum ==
                    Catch::Approx(pair.second->weight_sum)
                        .epsilon(1e-6)
                        .margin(1e-9));
          }
          REQUIRE(hp.a_separation ==
                  Catch::Approx(dp.a_separation).margin(1e-6));
          REQUIRE(hp.a_artifact ==
                  Catch::Approx(dp.a_artifact).margin(1e-6));
          REQUIRE(hp.a_registration ==
                  Catch::Approx(dp.a_registration).margin(1e-6));
          REQUIRE(hp.artifact_applicable == dp.artifact_applicable);
        }
      }
    }
  }
}

TEST_CASE("forward drizzle v2 gate10 host kernel validation",
          "[forward-drizzle-v2][gate10]") {
  auto f = make_fixture(ColorMode::MONO, BayerPattern::RGGB, 0, 0, 1);
  const int nc = f.plan.canvas_width_native;
  const int nr = f.plan.canvas_height_native;
  auto base_cfg = [&] {
    ForwardDrizzleV2KernelConfig k;
    k.internal_scale = 1;
    k.stream_length = f.plan.frames.size();
    k.half = 0.4;
    k.bayer_pattern = static_cast<int>(BayerPattern::RGGB);
    k.mono = true;
    k.canvas_width_native = nc;
    k.canvas_height_native = nr;
    return k;
  };

  SECTION("band origin must be contained in the canvas") {
    auto k = base_cfg();
    k.band_origin_y_native = -1;
    ForwardDrizzleV2CpuKernel kernel;
    REQUIRE_FALSE(kernel.reserve(nc, 4, f.plan.source_width,
                                 f.plan.source_height, k));
    k = base_cfg();
    k.band_origin_y_native = nr - 2;
    REQUIRE_FALSE(kernel.reserve(nc, 4, f.plan.source_width,
                                 f.plan.source_height, k));
    k = base_cfg();
    k.band_origin_y_native = nr - 4;  // exactly fits
    REQUIRE(kernel.reserve(nc, 4, f.plan.source_width,
                           f.plan.source_height, k));
  }
  SECTION("band origin without canvas dims must be zero") {
    auto k = base_cfg();
    k.canvas_width_native = 0;
    k.canvas_height_native = 0;
    k.band_origin_y_native = 2;
    ForwardDrizzleV2CpuKernel kernel;
    REQUIRE_FALSE(kernel.reserve(nc, nr, f.plan.source_width,
                                 f.plan.source_height, k));
  }
  SECTION("emit_profiles requires meta rows and a profile output") {
    auto k = base_cfg();
    k.emit_profiles = true;
    ForwardDrizzleV2CpuKernel kernel;
    REQUIRE(kernel.reserve(nc, nr, f.plan.source_width,
                           f.plan.source_height, k));
    const auto &m = f.plan.frames[0].source_to_canvas;
    const double a6[6] = {m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1),
                          m(1, 2)};
    REQUIRE_FALSE(kernel.accumulate_frame(a6, f.images[0].data(), nullptr,
                                          0));
    ForwardDrizzleV2FrameMeta meta{1.0f, 0.9f, 1, 0};
    for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
      const auto &mm = f.plan.frames[fr].source_to_canvas;
      const double aa[6] = {mm(0, 0), mm(0, 1), mm(0, 2),
                            mm(1, 0), mm(1, 1), mm(1, 2)};
      REQUIRE(kernel.accumulate_frame(aa, f.images[fr].data(), nullptr, fr,
                                      nullptr, &meta));
    }
    const std::size_t n =
        static_cast<std::size_t>(nc) * nr;
    std::vector<ForwardDrizzleV2PixelResult> out(n);
    std::uint64_t dense = 0;
    REQUIRE_FALSE(kernel.finalize(out.data(), nullptr, &dense));
    std::vector<ForwardDrizzleV2ProfileResult> pout(n);
    REQUIRE(kernel.finalize(out.data(), pout.data(), &dense));
  }
  SECTION("malformed meta rows fail the call") {
    auto k = base_cfg();
    k.emit_profiles = true;
    ForwardDrizzleV2CpuKernel kernel;
    REQUIRE(kernel.reserve(nc, nr, f.plan.source_width,
                           f.plan.source_height, k));
    const auto &m = f.plan.frames[0].source_to_canvas;
    const double a6[6] = {m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1),
                          m(1, 2)};
    ForwardDrizzleV2FrameMeta meta{-1.0f, 1.0f, 0, 0};
    REQUIRE_FALSE(kernel.accumulate_frame(a6, f.images[0].data(), nullptr,
                                          0, nullptr, &meta));
  }
  SECTION("local depth beyond the implicit tree rejects the call") {
    auto k = base_cfg();
    ForwardDrizzleV2CpuKernel kernel;
    REQUIRE(kernel.reserve(nc, nr, f.plan.source_width,
                           f.plan.source_height, k));
    const auto &m = f.plan.frames[0].source_to_canvas;
    const double a6[6] = {m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1),
                          m(1, 2)};
    auto w = v2_warp_descriptor(f.plan.frames[0]);
    w.max_subdivision_depth = 3;
    REQUIRE_FALSE(kernel.accumulate_frame_local(a6, w,
                                                f.images[0].data(), nullptr,
                                                0));
  }
  SECTION("double reserve / post-finalize accumulate fail") {
    auto k = base_cfg();
    ForwardDrizzleV2CpuKernel kernel;
    REQUIRE(kernel.reserve(nc, nr, f.plan.source_width,
                           f.plan.source_height, k));
    REQUIRE_FALSE(kernel.reserve(nc, nr, f.plan.source_width,
                                 f.plan.source_height, k));
    for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
      const auto &m = f.plan.frames[fr].source_to_canvas;
      const double a6[6] = {m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1),
                            m(1, 2)};
      REQUIRE(kernel.accumulate_frame(a6, f.images[fr].data(), nullptr,
                                      fr));
    }
    const auto &m = f.plan.frames[0].source_to_canvas;
    const double a6[6] = {m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1),
                          m(1, 2)};
    // Stream-length cap.
    REQUIRE_FALSE(kernel.accumulate_frame(a6, f.images[0].data(), nullptr,
                                          f.plan.frames.size()));
    std::vector<ForwardDrizzleV2PixelResult> out(
        static_cast<std::size_t>(nc) * nr);
    std::uint64_t dense = 0;
    REQUIRE(kernel.finalize(out.data(), nullptr, &dense));
    REQUIRE_FALSE(kernel.finalize(out.data(), nullptr, &dense));
  }
}

namespace {

// 12-frame fixture (extended from make_fixture's 5 via small y-jitter,
// mirroring the gate9 test's extension) where `outlier_frame`'s whole image
// has been pushed +500 off the rest, so every (pixel, channel) it
// contributes a candidate to reliably trips the sigma clip -- this exercises
// shared_frame_rejection's consensus step without depending on incidental
// noise to produce a clip event.
Fixture sfr_fixture(ColorMode mode, int outlier_frame) {
  auto f = make_fixture(mode, BayerPattern::GRBG, 1, -1, 1);
  const std::size_t base_frames = f.plan.frames.size();
  for (std::size_t i = base_frames; i < 12; ++i) {
    FrameSamplingTransform frame = f.plan.frames[i % base_frames];
    frame.frame_id = "v2-sfr-" + std::to_string(i);
    frame.source_index = i;
    frame.source_to_canvas(0, 2) +=
        static_cast<float>(0.011 * static_cast<double>(i - base_frames));
    f.plan.frames.push_back(frame);
    f.images.push_back(f.images[i % base_frames]);
  }
  auto &img = f.images[static_cast<std::size_t>(outlier_frame)];
  for (int y = 0; y < img.rows(); ++y)
    for (int x = 0; x < img.cols(); ++x)
      if (std::isfinite(img(y, x))) img(y, x) += 500.0f;
  return f;
}

struct SfrRun {
  std::vector<ForwardDrizzleV2PixelResult> results;
  std::vector<ForwardDrizzleV2ProfileResult> profiles;
  int channels = 0;
};

SfrRun run_sfr(const Fixture &f, bool mono, bool shared, double consensus,
              bool use_cuda = false) {
  const int nc = f.plan.canvas_width_native;
  const int nr = f.plan.canvas_height_native;
  const int channels = mono ? 1 : 3;
  const std::size_t nplane = static_cast<std::size_t>(nc) * nr;

  ForwardDrizzleV2KernelConfig kcfg;
  kcfg.internal_scale = f.cfg.internal_scale;
  kcfg.stream_length = static_cast<int>(f.plan.frames.size());
  kcfg.half = 0.5 * f.cfg.pixfrac;
  kcfg.bayer_pattern = static_cast<int>(f.plan.bayer_pattern);
  kcfg.cfa_origin_x = f.plan.cfa_origin_x;
  kcfg.cfa_origin_y = f.plan.cfa_origin_y;
  kcfg.mono = mono;
  kcfg.emit_profiles = true;
  kcfg.shared_frame_rejection = shared;
  kcfg.shared_frame_rejection_consensus = consensus;

  std::unique_ptr<ForwardDrizzleV2Kernel> kernel_holder =
      use_cuda ? std::unique_ptr<ForwardDrizzleV2Kernel>(
                     std::make_unique<ForwardDrizzleV2CudaKernel>())
               : std::unique_ptr<ForwardDrizzleV2Kernel>(
                     std::make_unique<ForwardDrizzleV2CpuKernel>());
  ForwardDrizzleV2Kernel &kernel = *kernel_holder;
  REQUIRE(kernel.reserve(nc, nr, f.plan.source_width, f.plan.source_height,
                         kcfg));
  ForwardDrizzleV2FrameMeta meta{1.0f, 0.9f, 1, 0};
  for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
    const auto &m = f.plan.frames[fr].source_to_canvas;
    const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                          m(1, 0), m(1, 1), m(1, 2)};
    REQUIRE(kernel.accumulate_frame(a6, f.images[fr].data(), nullptr, fr,
                                    nullptr, &meta));
  }
  SfrRun out;
  out.channels = channels;
  out.results.resize(nplane * channels);
  out.profiles.resize(nplane * channels);
  std::uint64_t dense = 0;
  REQUIRE(kernel.finalize(out.results.data(), out.profiles.data(), &dense));
  return out;
}

void require_pixel_result_equal(const ForwardDrizzleV2PixelResult &a,
                                const ForwardDrizzleV2PixelResult &b) {
  REQUIRE(a.value == b.value);
  REQUIRE(a.b == b.b);
  REQUIRE(a.n_eff == b.n_eff);
  REQUIRE(a.confidence == b.confidence);
  REQUIRE(a.geometry_fraction == b.geometry_fraction);
  REQUIRE(a.source_fraction == b.source_fraction);
  REQUIRE(a.estimator_fraction == b.estimator_fraction);
  REQUIRE(a.profile_fraction == b.profile_fraction);
  REQUIRE(a.contributors == b.contributors);
  REQUIRE(a.robust_state == b.robust_state);
  REQUIRE(a.confidence_state == b.confidence_state);
  REQUIRE(a.conf_degraded == b.conf_degraded);
}

void require_profile_output_equal(const ForwardDrizzleV2ProfileOutput &a,
                                  const ForwardDrizzleV2ProfileOutput &b) {
  REQUIRE(a.support == b.support);
  if (a.support) {
    REQUIRE(a.value == b.value);
    REQUIRE(a.weight_sum == b.weight_sum);
    REQUIRE(a.n_eff == b.n_eff);
  }
}

void require_profile_result_equal(const ForwardDrizzleV2ProfileResult &a,
                                  const ForwardDrizzleV2ProfileResult &b) {
  require_profile_output_equal(a.uniform, b.uniform);
  require_profile_output_equal(a.raw, b.raw);
  require_profile_output_equal(a.fine, b.fine);
  require_profile_output_equal(a.medium, b.medium);
  REQUIRE(a.a_separation == b.a_separation);
  REQUIRE(a.a_artifact == b.a_artifact);
  REQUIRE(a.a_registration == b.a_registration);
  REQUIRE(a.artifact_applicable == b.artifact_applicable);
}

}  // namespace

TEST_CASE("forward drizzle v2 shared_frame_rejection consensus=1.0 is "
          "bit-identical to disabled",
          "[forward-drizzle-v2][shared-frame-rejection]") {
  // shared_frame_rejection_consensus=1.0 means `frac > 1.0` never fires, so
  // the consensus step never revises any channel's own clip decision --
  // the accepted set it produces must reduce exactly to what the
  // shared_frame_rejection=false fast path computes. This is the
  // discriminating test for the finalize() restructuring itself (the
  // ChannelWork/px-ch reorganization): any divergence here is a bug in the
  // hand-copy of the clip algorithm into the two branches, not a property
  // of the consensus feature.
  for (ColorMode mode : {ColorMode::OSC, ColorMode::MONO}) {
    const auto f = sfr_fixture(mode, /*outlier_frame=*/5);
    const bool mono = mode == ColorMode::MONO;
    const auto off = run_sfr(f, mono, false, 0.5);
    const auto on = run_sfr(f, mono, true, 1.0);
    REQUIRE(off.channels == on.channels);
    REQUIRE(off.results.size() == on.results.size());
    bool saw_clip = false;
    for (std::size_t i = 0; i < off.results.size(); ++i) {
      require_pixel_result_equal(off.results[i], on.results[i]);
      require_profile_result_equal(off.profiles[i], on.profiles[i]);
      if (off.results[i].robust_state ==
          static_cast<std::uint8_t>(
              ForwardDrizzleV2RobustState::primary_reservoir_sigma_clip))
        saw_clip = true;
    }
    // The +500 outlier frame must actually have triggered the clip somewhere
    // -- otherwise this test would pass vacuously without exercising the
    // clip/finalize code path at all.
    REQUIRE(saw_clip);
  }
}

TEST_CASE("forward drizzle v2 shared_frame_rejection consensus drops a "
          "cross-channel outlier frame",
          "[forward-drizzle-v2][shared-frame-rejection]") {
  // OSC only: consensus across channels requires nch > 1. `r.b` is the raw
  // full-stream folded weight (set once from the pre-clip accumulator and
  // never revised by the clip or consensus step), so it can't show a
  // consensus effect; `r.value` (the accepted-candidate weighted mean) and
  // `r.confidence` are what the consensus step's `cwv.accepted[i] = 0`
  // revision actually feeds into. This sweeps outlier magnitudes at the
  // fixture's production-range pixfrac (0.8, from make_fixture) and
  // requires that at least one configuration produces a pixel where a
  // SUBSET (not all) of the clipped channels change `value` -- a uniform
  // change across every clipped channel at a pixel wouldn't distinguish
  // "consensus fired" from some unrelated global effect, but a change
  // isolated to some channels and not others at the same pixel can only
  // come from the per-frame vote-counting logic actually revising that
  // specific channel's accepted set.
  bool saw_partial_value_change = false;
  for (const double magnitude : {4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0}) {
    for (const int outlier_frame : {3, 5, 7}) {
      auto f = sfr_fixture(ColorMode::OSC, outlier_frame);
      // Undo the fixture's uniform +500 bump and instead perturb only the
      // source pixels that feed the R and G channel planes, leaving B's
      // candidate for this frame at baseline -- since each channel plane is
      // built only from its own CFA-colored source pixels, this makes the
      // outlier frame a clear per-channel outlier in R/G while B's own clip
      // pass has no reason to reject it. If R and G's rejection then also
      // shows up in B's accepted set, that can only be the cross-channel
      // consensus vote, not B's own independent clip.
      auto &img = f.images[static_cast<std::size_t>(outlier_frame)];
      for (int y = 0; y < img.rows(); ++y)
        for (int x = 0; x < img.cols(); ++x) {
          if (!std::isfinite(img(y, x))) continue;
          img(y, x) -= 500.0f;
          const auto c = cfa_channel_for_source_pixel(
              x, y, f.plan.bayer_pattern, f.plan.cfa_origin_x,
              f.plan.cfa_origin_y);
          if (c == CfaChannel::R || c == CfaChannel::G)
            img(y, x) += static_cast<float>(magnitude);
        }
      const auto off = run_sfr(f, false, false, 0.5);
      const auto on = run_sfr(f, false, true, 0.5);
      REQUIRE(off.results.size() == on.results.size());
      const std::size_t nplane = off.results.size() / 3;
      for (std::size_t px = 0; px < nplane; ++px) {
        int clipped = 0, changed = 0;
        for (int c = 0; c < 3; ++c) {
          const std::size_t i = static_cast<std::size_t>(c) * nplane + px;
          const auto &o = off.results[i];
          const auto &n = on.results[i];
          if (o.robust_state !=
              static_cast<std::uint8_t>(
                  ForwardDrizzleV2RobustState::primary_reservoir_sigma_clip))
            continue;
          ++clipped;
          if (std::fabs(o.value - n.value) > 1e-6) ++changed;
        }
        if (clipped >= 2 && changed >= 1 && changed < clipped)
          saw_partial_value_change = true;
      }
    }
  }
  REQUIRE(saw_partial_value_change);
}

TEST_CASE("forward drizzle v2 shared_frame_rejection CUDA matches the "
          "original (non-SFR) CUDA kernel at consensus=1.0",
          "[forward-drizzle-v2][shared-frame-rejection][cuda-parity]") {
  if (!forward_drizzle_cuda_runtime_available()) {
    SUCCEED("CUDA device unavailable");
    return;
  }
  // Bit-exactness against the CPU oracle is NOT checkable here: comparing
  // run_sfr(..., shared=false, use_cuda=false) against
  // run_sfr(..., shared=false, use_cuda=true) on THIS fixture already
  // disagrees in the last 2-3 ULPs of `value` on a per-pixel basis --
  // confirmed independent of shared_frame_rejection (same mismatch at
  // magnitude=0, i.e. no injected outlier at all, and identical whether
  // shared_frame_rejection is compiled in or not). That is a pre-existing
  // CPU<->CUDA gap in the original k_finalize_v2/CPU-oracle pair on this
  // specific fixture shape (12 frames, GRBG/(1,-1), uniform per-frame meta),
  // not something introduced by the SFR kernels -- see the session notes;
  // out of scope to chase here.
  //
  // What IS checkable, and is the actual correctness claim for the new
  // device code (k_finalize_v2_sfr_build/_vote/_reduce in
  // forward_drizzle_cuda_device.cu): with shared_frame_rejection_consensus
  // = 1.0, kernel B's vote never revises anything (frac > 1.0 is never
  // true), so the SFR kernel path must reduce to bit-identical output
  // against the ORIGINAL k_finalize_v2 kernel on the SAME device, for both
  // OSC (kernel B's cross-channel loop active) and MONO (kernel B's grid is
  // skipped: channels <= 1).
  for (ColorMode mode : {ColorMode::OSC, ColorMode::MONO}) {
    const auto f = sfr_fixture(mode, /*outlier_frame=*/5);
    const bool mono = mode == ColorMode::MONO;
    const auto off = run_sfr(f, mono, false, 1.0, /*use_cuda=*/true);
    const auto on = run_sfr(f, mono, true, 1.0, /*use_cuda=*/true);
    REQUIRE(off.results.size() == on.results.size());
    bool saw_clip = false;
    for (std::size_t i = 0; i < off.results.size(); ++i) {
      require_pixel_result_equal(off.results[i], on.results[i]);
      require_profile_result_equal(off.profiles[i], on.profiles[i]);
      if (off.results[i].robust_state ==
          static_cast<std::uint8_t>(
              ForwardDrizzleV2RobustState::primary_reservoir_sigma_clip))
        saw_clip = true;
    }
    REQUIRE(saw_clip);
  }
}

TEST_CASE("forward drizzle v2 shared_frame_rejection CUDA consensus vote "
          "fires and matches the CPU vote's qualitative behavior",
          "[forward-drizzle-v2][shared-frame-rejection][cuda-parity]") {
  if (!forward_drizzle_cuda_runtime_available()) {
    SUCCEED("CUDA device unavailable");
    return;
  }
  // GPU counterpart of the CPU-only "consensus drops a cross-channel
  // outlier frame" sweep above -- same R/G-only perturbation technique (so
  // B's own clip has no reason to reject the outlier frame, isolating the
  // cross-channel vote), but comparing GPU consensus=0.5 against GPU
  // consensus=1.0 instead of CPU on vs off, to stay clear of the pre-existing
  // CPU<->CUDA gap noted in the test above.
  bool saw_partial_value_change = false;
  for (const double magnitude : {4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0}) {
    for (const int outlier_frame : {3, 5, 7}) {
      auto f = sfr_fixture(ColorMode::OSC, outlier_frame);
      auto &img = f.images[static_cast<std::size_t>(outlier_frame)];
      for (int y = 0; y < img.rows(); ++y)
        for (int x = 0; x < img.cols(); ++x) {
          if (!std::isfinite(img(y, x))) continue;
          img(y, x) -= 500.0f;
          const auto c = cfa_channel_for_source_pixel(
              x, y, f.plan.bayer_pattern, f.plan.cfa_origin_x,
              f.plan.cfa_origin_y);
          if (c == CfaChannel::R || c == CfaChannel::G)
            img(y, x) += static_cast<float>(magnitude);
        }
      const auto off = run_sfr(f, false, true, 1.0, /*use_cuda=*/true);
      const auto on = run_sfr(f, false, true, 0.5, /*use_cuda=*/true);
      REQUIRE(off.results.size() == on.results.size());
      const std::size_t nplane = off.results.size() / 3;
      for (std::size_t px = 0; px < nplane; ++px) {
        int clipped = 0, changed = 0;
        for (int c = 0; c < 3; ++c) {
          const std::size_t i = static_cast<std::size_t>(c) * nplane + px;
          const auto &o = off.results[i];
          const auto &n = on.results[i];
          if (o.robust_state !=
              static_cast<std::uint8_t>(
                  ForwardDrizzleV2RobustState::primary_reservoir_sigma_clip))
            continue;
          ++clipped;
          if (std::fabs(o.value - n.value) > 1e-6) ++changed;
        }
        if (clipped >= 2 && changed >= 1 && changed < clipped)
          saw_partial_value_change = true;
      }
    }
  }
  REQUIRE(saw_partial_value_change);
}

namespace {

// Builds a driver-ready run plan over the small fixture canvas.
ForwardDrizzleV2RunPlan g10_driver_plan(const Fixture &f, int band_rows,
                                        int channels, bool emit_profiles) {
  ForwardDrizzleV2RunPlan p;
  p.source_identity_hash = "src";
  p.normalized_cache_hash = "cache";
  p.quality_plan_hash = "q";
  p.sampling_plan_hash = "sampling";
  p.config_snapshot_hash = "cfg";
  p.native_width = f.plan.canvas_width_native;
  p.native_height = f.plan.canvas_height_native;
  p.channels = channels;
  p.internal_scale = f.cfg.internal_scale;
  p.color_mode = channels == 1 ? "MONO" : "OSC";
  p.pixfrac = f.cfg.pixfrac;
  p.bayer_pattern = static_cast<int>(f.plan.bayer_pattern);
  p.cfa_origin_x = f.plan.cfa_origin_x;
  p.cfa_origin_y = f.plan.cfa_origin_y;
  p.frame_count = f.plan.frames.size();
  p.band_rows = band_rows;
  p.band_count =
      (p.native_height + band_rows - 1) / band_rows;
  p.emit_profiles = emit_profiles;
  p.multiband_levels = emit_profiles ? 3 : 0;
  finalize_forward_drizzle_v2_run_plan(p);
  return p;
}

// Frame provider over a Fixture: sigma2 plane + meta rows are constant
// fixtures; local models are forwarded when present.
ForwardDrizzleV2FrameProvider g10_provider(const Fixture &f,
                                           const std::vector<float> &s2,
                                           const std::vector<ForwardDrizzleV2FrameMeta> *meta) {
  return [&f, &s2, meta](int, int, std::uint64_t order,
                         const ForwardDrizzleV2FramePieceSink &sink) {
    if (order >= f.plan.frames.size()) return false;
    const auto &frame = f.plan.frames[order];
    const auto &m = frame.source_to_canvas;
    ForwardDrizzleV2FrameInput out;
    const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                          m(1, 0), m(1, 1), m(1, 2)};
    std::copy_n(a6, 6, out.affine6);
    out.has_local_model = frame.has_smooth_local_model;
    if (out.has_local_model) out.warp = v2_warp_descriptor(frame);
    out.source = f.images[order].data();
    out.sigma2 = s2.data();
    out.source_window = {0, 0, f.plan.source_width, f.plan.source_height};
    if (meta) out.meta = (*meta)[order];
    return sink(out);
  };
}

// Reads every committed band back and returns the whole-canvas record and
// profile vectors (band rows stacked in order, channel-planar per band).
void g10_read_store(const fs::path &root, const ForwardDrizzleV2RunPlan &plan,
                    std::vector<ForwardDrizzleV2PixelResult> &records,
                    std::vector<ForwardDrizzleV2ProfileResult> *profiles) {
  const auto insp = inspect_forward_drizzle_v2_store(root, plan);
  REQUIRE(insp.status == ForwardDrizzleV2StoreStatus::complete);
  records.clear();
  if (profiles) profiles->clear();
  ForwardDrizzleV2Checkpoint cp;
  std::string error;
  REQUIRE(parse_forward_drizzle_v2_checkpoint(
      v2_file_text(insp.generation / "checkpoint.json"), cp, error));
  REQUIRE(cp.bands.size() == static_cast<std::size_t>(plan.band_count));
  const std::size_t nplane = static_cast<std::size_t>(plan.native_width) *
                             plan.native_height;
  records.assign(nplane * plan.channels,
                 ForwardDrizzleV2PixelResult{});
  if (profiles)
    profiles->assign(nplane * plan.channels,
                     ForwardDrizzleV2ProfileResult{});
  for (const auto &b : cp.bands) {
    const auto band = read_forward_drizzle_v2_band(insp.generation, b);
    REQUIRE(band.size() ==
            static_cast<std::size_t>(b.rows) * b.native_cols * b.channels);
    for (int c = 0; c < b.channels; ++c)
      for (int r = 0; r < b.rows; ++r)
        std::copy_n(band.data() + static_cast<std::size_t>(c) * b.rows *
                                     b.native_cols + static_cast<std::size_t>(r) * b.native_cols,
                    b.native_cols,
                    records.data() +
                        static_cast<std::size_t>(c) * nplane +
                        static_cast<std::size_t>(b.y_begin + r) *
                            b.native_cols);
    if (profiles) {
      const auto pb =
          read_forward_drizzle_v2_band_profiles(insp.generation, b);
      REQUIRE(pb.size() == band.size());
      for (int c = 0; c < b.channels; ++c)
        for (int r = 0; r < b.rows; ++r)
          std::copy_n(pb.data() + static_cast<std::size_t>(c) * b.rows *
                                      b.native_cols + static_cast<std::size_t>(r) * b.native_cols,
                      b.native_cols,
                      profiles->data() +
                          static_cast<std::size_t>(c) * nplane +
                          static_cast<std::size_t>(b.y_begin + r) *
                              b.native_cols);
    }
  }
}

bool g10_bit_eq(const ForwardDrizzleV2PixelResult &a,
                const ForwardDrizzleV2PixelResult &b) {
  auto eq = [](const auto &x, const auto &y) {
    return std::memcmp(&x, &y, sizeof(x)) == 0;
  };
  return eq(a.value, b.value) && eq(a.b, b.b) && eq(a.n_eff, b.n_eff) &&
         eq(a.confidence, b.confidence) &&
         eq(a.geometry_fraction, b.geometry_fraction) &&
         eq(a.source_fraction, b.source_fraction) &&
         eq(a.estimator_fraction, b.estimator_fraction) &&
         eq(a.profile_fraction, b.profile_fraction) &&
         eq(a.contributors, b.contributors) &&
         eq(a.robust_state, b.robust_state) &&
         eq(a.confidence_state, b.confidence_state) &&
         eq(a.conf_degraded, b.conf_degraded);
}

bool g10_bit_eq(const ForwardDrizzleV2ProfileResult &a,
                const ForwardDrizzleV2ProfileResult &b) {
  auto eq = [](const auto &x, const auto &y) {
    return std::memcmp(&x, &y, sizeof(x)) == 0;
  };
  for (const auto &pair :
       {std::pair{&a.uniform, &b.uniform}, std::pair{&a.raw, &b.raw},
        std::pair{&a.fine, &b.fine}, std::pair{&a.medium, &b.medium}}) {
    if (!eq(pair.first->value, pair.second->value) ||
        !eq(pair.first->weight_sum, pair.second->weight_sum) ||
        !eq(pair.first->n_eff, pair.second->n_eff) ||
        pair.first->support != pair.second->support)
      return false;
  }
  return eq(a.a_separation, b.a_separation) &&
         eq(a.a_artifact, b.a_artifact) &&
         eq(a.a_registration, b.a_registration) &&
         a.artifact_applicable == b.artifact_applicable;
}

}  // namespace

TEST_CASE("forward drizzle v2 gate10 driver banded run round-trips",
          "[forward-drizzle-v2][gate10][driver]") {
  V2StoreFixture fx;
  auto f = make_fixture(ColorMode::MONO, BayerPattern::GRBG, 0, 0, 1);
  const int nc = f.plan.canvas_width_native;
  const int nr = f.plan.canvas_height_native;
  const std::size_t n_src =
      static_cast<std::size_t>(f.plan.source_width) * f.plan.source_height;
  std::vector<float> s2(n_src, 0.02f);
  std::vector<ForwardDrizzleV2FrameMeta> meta(f.plan.frames.size());
  for (std::size_t i = 0; i < meta.size(); ++i)
    meta[i] = {0.9f, 0.8f, static_cast<std::uint8_t>(i % 2), 0};
  float cx[16] = {}, cy[16] = {};
  cx[5] = 1.2f;
  cy[6] = -0.7f;
  f.plan.frames[2] =
      v2_local_frame(f.plan.frames[2], v2_local_model(nr, nc, cx, cy));

  const auto plan = g10_driver_plan(f, 4, 1, true);
  REQUIRE(plan.band_count == 4);  // 13 rows over 4-row bands
  ForwardDrizzleV2DriverOptions opts;
  opts.prefer_cuda = false;
  std::vector<int> progressed;
  opts.progress = [&](int band, int) { progressed.push_back(band); };
  const auto result =
      run_forward_drizzle_v2(fx.root, plan, f.plan.source_width,
                             f.plan.source_height,
                             g10_provider(f, s2, &meta), opts);
  REQUIRE(result.committed);
  REQUIRE(result.backend_used == "cpu_v2");
  REQUIRE(result.bands_committed == plan.band_count);
  REQUIRE(result.bands_reused == 0);
  REQUIRE(progressed == std::vector<int>({0, 1, 2, 3}));
  REQUIRE(result.totals.frames_processed == plan.frame_count *
                                               plan.band_count);

  std::vector<ForwardDrizzleV2PixelResult> records;
  std::vector<ForwardDrizzleV2ProfileResult> profiles;
  g10_read_store(fx.root, plan, records, &profiles);

  // Records must be bitwise identical to direct per-band kernel runs.
  for (int band = 0; band < plan.band_count; ++band) {
    const int y0 = band * plan.band_rows;
    const int rows = std::min(plan.band_rows, nr - y0);
    ForwardDrizzleV2KernelConfig k;
    k.internal_scale = plan.internal_scale;
    k.stream_length = plan.frame_count;
    k.min_clip_contributors = plan.min_clip_contributors;
    k.min_candidates = plan.min_candidates;
    k.robust_passes = plan.robust_passes;
    k.sigma_low = plan.sigma_low;
    k.sigma_high = plan.sigma_high;
    k.half = 0.5 * plan.pixfrac;
    k.bayer_pattern = plan.bayer_pattern;
    k.cfa_origin_x = plan.cfa_origin_x;
    k.cfa_origin_y = plan.cfa_origin_y;
    k.mono = true;
    k.sigma2_plane = plan.sigma2_enabled;
    k.canvas_width_native = nc;
    k.canvas_height_native = nr;
    k.band_origin_y_native = y0;
    k.emit_profiles = true;
    k.fine_quality_exponent = plan.fine_quality_exponent;
    k.medium_quality_exponent = plan.medium_quality_exponent;
    ForwardDrizzleV2CpuKernel kernel;
    REQUIRE(kernel.reserve(nc, rows, f.plan.source_width,
                           f.plan.source_height, k));
    for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
      const auto &frame = f.plan.frames[fr];
      const auto &m = frame.source_to_canvas;
      const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                            m(1, 0), m(1, 1), m(1, 2)};
      if (frame.has_smooth_local_model)
        REQUIRE(kernel.accumulate_frame_local(
            a6, v2_warp_descriptor(frame), f.images[fr].data(), s2.data(),
            fr, nullptr, &meta[fr]));
      else
        REQUIRE(kernel.accumulate_frame(a6, f.images[fr].data(), s2.data(),
                                        fr, nullptr, &meta[fr]));
    }
    std::vector<ForwardDrizzleV2PixelResult> direct(
        static_cast<std::size_t>(rows) * nc);
    std::vector<ForwardDrizzleV2ProfileResult> directp(direct.size());
    std::uint64_t dense = 0;
    REQUIRE(kernel.finalize(direct.data(), directp.data(), &dense));
    for (int r = 0; r < rows; ++r)
      for (int x = 0; x < nc; ++x) {
        const std::size_t wb =
            static_cast<std::size_t>(y0 + r) * nc + x;
        const std::size_t rb = static_cast<std::size_t>(r) * nc + x;
        REQUIRE(g10_bit_eq(records[wb], direct[rb]));
        REQUIRE(g10_bit_eq(profiles[wb], directp[rb]));
      }
  }
}

TEST_CASE("forward drizzle v2 gate10 driver resume reuses the prefix",
          "[forward-drizzle-v2][gate10][driver]") {
  V2StoreFixture crashed;
  V2StoreFixture reference;
  auto f = make_fixture(ColorMode::MONO, BayerPattern::GRBG, 0, 0, 1);
  const std::size_t n_src =
      static_cast<std::size_t>(f.plan.source_width) * f.plan.source_height;
  std::vector<float> s2(n_src, 0.02f);
  const auto plan = g10_driver_plan(f, 4, 1, true);
  const auto provider = g10_provider(f, s2, nullptr);

  ForwardDrizzleV2DriverOptions cpu;
  cpu.prefer_cuda = false;

  const auto ref =
      run_forward_drizzle_v2(reference.root, plan, f.plan.source_width,
                             f.plan.source_height, provider, cpu);
  REQUIRE(ref.committed);

  ForwardDrizzleV2DriverOptions kill = cpu;
  kill.simulate_kill_after_bands = 2;
  REQUIRE_THROWS_AS(run_forward_drizzle_v2(crashed.root, plan,
                                           f.plan.source_width,
                                           f.plan.source_height, provider,
                                           kill),
                    ForwardDrizzleV2SimulatedKill);

  const auto insp = inspect_forward_drizzle_v2_store(crashed.root, plan);
  REQUIRE(insp.status == ForwardDrizzleV2StoreStatus::resumable);
  REQUIRE(insp.next_band == 2);

  const auto done =
      run_forward_drizzle_v2(crashed.root, plan, f.plan.source_width,
                             f.plan.source_height, provider, cpu);
  REQUIRE(done.committed);
  REQUIRE(done.bands_reused == 2);
  REQUIRE(done.bands_committed == plan.band_count - 2);
  // The commit hash binds the generation name, so it differs between runs;
  // the deterministic content identity is the checkpoint hash.
  REQUIRE(!done.commit_hash.empty());
  REQUIRE(!ref.commit_hash.empty());
  {
    ForwardDrizzleV2Checkpoint cp_done, cp_ref;
    std::string cp_error;
    REQUIRE(parse_forward_drizzle_v2_checkpoint(
        v2_file_text(done.generation_dir / "checkpoint.json"), cp_done,
        cp_error));
    REQUIRE(parse_forward_drizzle_v2_checkpoint(
        v2_file_text(ref.generation_dir / "checkpoint.json"), cp_ref,
        cp_error));
    REQUIRE(cp_done.checkpoint_hash == cp_ref.checkpoint_hash);
  }

  std::vector<ForwardDrizzleV2PixelResult> a, b;
  std::vector<ForwardDrizzleV2ProfileResult> pa, pb;
  g10_read_store(crashed.root, plan, a, &pa);
  g10_read_store(reference.root, plan, b, &pb);
  REQUIRE(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    REQUIRE(g10_bit_eq(a[i], b[i]));
    REQUIRE(g10_bit_eq(pa[i], pb[i]));
  }
}

TEST_CASE("forward drizzle v2 gate10 cuda fault restarts on cpu",
          "[forward-drizzle-v2][gate10][driver]") {
  if (!forward_drizzle_cuda_runtime_available()) {
    SUCCEED("CUDA device unavailable");
    return;
  }
  V2StoreFixture fx;
  V2StoreFixture reference;
  auto f = make_fixture(ColorMode::MONO, BayerPattern::GRBG, 0, 0, 1);
  const std::size_t n_src =
      static_cast<std::size_t>(f.plan.source_width) * f.plan.source_height;
  std::vector<float> s2(n_src, 0.02f);
  const auto plan = g10_driver_plan(f, 4, 1, true);
  const auto provider = g10_provider(f, s2, nullptr);

  ForwardDrizzleV2DriverOptions cpu;
  cpu.prefer_cuda = false;
  const auto ref =
      run_forward_drizzle_v2(reference.root, plan, f.plan.source_width,
                             f.plan.source_height, provider, cpu);

  // The device attempt commits band 0, then fails: the whole phase
  // restarts on CPU and the discarded band-0 artifact must not leak into
  // the result.
  set_forward_drizzle_v2_cuda_fault_after_bands(1);
  ForwardDrizzleV2DriverOptions pref;
  pref.prefer_cuda = true;
  const auto done =
      run_forward_drizzle_v2(fx.root, plan, f.plan.source_width,
                             f.plan.source_height, provider, pref);
  set_forward_drizzle_v2_cuda_fault_after_bands(-1);
  REQUIRE(done.committed);
  REQUIRE(done.backend_used == "cpu_v2");
  REQUIRE(!done.cuda_fallback_reason.empty());
  REQUIRE(done.bands_reused == 0);
  // The commit hash binds the generation name and backend telemetry, so it
  // legitimately differs between runs; the deterministic content identity is
  // the checkpoint hash (plan + committed band records).
  REQUIRE(!done.commit_hash.empty());
  REQUIRE(!ref.commit_hash.empty());
  {
    ForwardDrizzleV2Checkpoint cp_done, cp_ref;
    std::string cp_error;
    REQUIRE(parse_forward_drizzle_v2_checkpoint(
        v2_file_text(done.generation_dir / "checkpoint.json"), cp_done,
        cp_error));
    REQUIRE(parse_forward_drizzle_v2_checkpoint(
        v2_file_text(ref.generation_dir / "checkpoint.json"), cp_ref,
        cp_error));
    REQUIRE(cp_done.checkpoint_hash == cp_ref.checkpoint_hash);
  }

  std::vector<ForwardDrizzleV2PixelResult> a, b;
  std::vector<ForwardDrizzleV2ProfileResult> pa, pb;
  g10_read_store(fx.root, plan, a, &pa);
  g10_read_store(reference.root, plan, b, &pb);
  for (std::size_t i = 0; i < a.size(); ++i) {
    REQUIRE(g10_bit_eq(a[i], b[i]));
    REQUIRE(g10_bit_eq(pa[i], pb[i]));
  }
}

TEST_CASE("forward drizzle v2 gate10 driver fails closed on corruption",
          "[forward-drizzle-v2][gate10][driver]") {
  auto f = make_fixture(ColorMode::MONO, BayerPattern::GRBG, 0, 0, 1);
  const std::size_t n_src =
      static_cast<std::size_t>(f.plan.source_width) * f.plan.source_height;
  std::vector<float> s2(n_src, 0.02f);
  const auto plan = g10_driver_plan(f, 4, 1, true);
  const auto provider = g10_provider(f, s2, nullptr);
  ForwardDrizzleV2DriverOptions cpu;
  cpu.prefer_cuda = false;

  // Corruption is detected on the resume inspection of an UNCOMMITTED
  // generation; a complete store short-circuits via current.json (payload
  // verification happens when a reader touches the artifacts).
  auto crashed = [&](const fs::path &root) {
    ForwardDrizzleV2DriverOptions kill = cpu;
    kill.simulate_kill_after_bands = 3;
    REQUIRE_THROWS_AS(run_forward_drizzle_v2(root, plan, f.plan.source_width,
                                           f.plan.source_height, provider,
                                           kill),
                      ForwardDrizzleV2SimulatedKill);
  };
  auto reruns_throw = [&](const fs::path &root) {
    REQUIRE_THROWS(run_forward_drizzle_v2(root, plan, f.plan.source_width,
                                          f.plan.source_height, provider,
                                          cpu));
  };

  // Bit flip inside a committed profile payload.
  {
    V2StoreFixture fx;
    crashed(fx.root);
    const auto gen = v2_single_generation(fx.root);
    const auto file = gen / "band-0001.profiles.bin";
    std::fstream io(file, std::ios::in | std::ios::out | std::ios::binary);
    io.seekp(64);
    char byte = 0;
    io.read(&byte, 1);
    byte ^= 0x40;
    io.seekp(64);
    io.write(&byte, 1);
    io.close();
    reruns_throw(fx.root);
  }
  // Truncated pixel band artifact.
  {
    V2StoreFixture fx;
    crashed(fx.root);
    const auto gen = v2_single_generation(fx.root);
    fs::resize_file(gen / "band-0002.bin", 100);
    reruns_throw(fx.root);
  }
  // Non-contiguous checkpoint.
  {
    V2StoreFixture fx;
    crashed(fx.root);
    const auto gen = v2_single_generation(fx.root);
    auto cp = nlohmann::json::parse(
        v2_file_text(gen / "checkpoint.json"));
    cp["bands"].erase(cp["bands"].begin() + 1);
    std::ofstream(gen / "checkpoint.json") << cp.dump(2);
    reruns_throw(fx.root);
  }
  // A plan-hash mismatch (foreign context) is rejected the same way.
  {
    V2StoreFixture fx;
    crashed(fx.root);
    auto foreign = plan;
    foreign.sigma_low = 2.5;
    finalize_forward_drizzle_v2_run_plan(foreign);
    REQUIRE_THROWS(run_forward_drizzle_v2(
        fx.root, foreign, f.plan.source_width, f.plan.source_height,
        provider, cpu));
  }
}

TEST_CASE("forward drizzle v2 gate10 store feeds streamed fusion",
          "[forward-drizzle-v2][gate10][driver][fusion]") {
  V2StoreFixture fx;
  auto f = make_fixture(ColorMode::MONO, BayerPattern::GRBG, 0, 0, 1);
  const int nc = f.plan.canvas_width_native;
  const int nr = f.plan.canvas_height_native;
  const std::size_t n_src =
      static_cast<std::size_t>(f.plan.source_width) * f.plan.source_height;
  std::vector<float> s2(n_src, 0.02f);
  std::vector<ForwardDrizzleV2FrameMeta> meta(f.plan.frames.size());
  for (std::size_t i = 0; i < meta.size(); ++i)
    meta[i] = {1.0f, 0.75f, 1, 0};

  const auto plan = g10_driver_plan(f, 3, 1, true);
  ForwardDrizzleV2DriverOptions cpu;
  cpu.prefer_cuda = false;
  const auto done =
      run_forward_drizzle_v2(fx.root, plan, f.plan.source_width,
                             f.plan.source_height,
                             g10_provider(f, s2, &meta), cpu);
  REQUIRE(done.committed);

  std::vector<ForwardDrizzleV2PixelResult> records;
  std::vector<ForwardDrizzleV2ProfileResult> profiles;
  g10_read_store(fx.root, plan, records, &profiles);
  REQUIRE(profiles.size() ==
          static_cast<std::size_t>(nc) * nr);

  const auto uniform = forward_drizzle_v2_profiles_to_uniform_result(
      profiles, nc, nr, 1, ColorMode::MONO, 0);
  const auto raw = forward_drizzle_v2_profiles_to_uniform_result(
      profiles, nc, nr, 1, ColorMode::MONO, 1);
  const auto fine = forward_drizzle_v2_profiles_to_uniform_result(
      profiles, nc, nr, 1, ColorMode::MONO, 2);
  const auto medium = forward_drizzle_v2_profiles_to_uniform_result(
      profiles, nc, nr, 1, ColorMode::MONO, 3);
  const auto a_sep =
      forward_drizzle_v2_profile_alpha_plane(profiles, nc, nr, 1, 0);
  const auto a_art =
      forward_drizzle_v2_profile_alpha_plane(profiles, nc, nr, 1, 1);
  const auto a_reg =
      forward_drizzle_v2_profile_alpha_plane(profiles, nc, nr, 1, 2);

  // The committed store must feed the production streaming fusion: whole
  // vs. streamed fusion stay identical on real v2 records.
  for (int levels : {1, 2, 3}) {
    config::ReconstructionMultibandConfig mb;
    mb.levels = levels;
    const auto whole =
        fuse_multiband(uniform, raw, fine, medium, ColorMode::MONO, nc, nr,
                       mb, {}, {}, a_sep, a_art, a_reg);
    const auto streamed = fuse_multiband_streamed(
        uniform, raw, fine, medium, ColorMode::MONO, nc, nr, mb, 2, {}, {},
        a_sep, a_art, a_reg);
    REQUIRE(streamed.L.size() == whole.L.size());
    REQUIRE(streamed.support_L == whole.support_L);
    for (std::size_t i = 0; i < whole.L.size(); ++i) {
      const float sv = streamed.L[i], wv = whole.L[i];
      REQUIRE((sv == wv || (std::isnan(sv) && std::isnan(wv))));
      if (whole.support_L[i])
        REQUIRE(uniform.L.support[i] == 1);
    }
  }
}

TEST_CASE("forward drizzle v2 gate10 production plan binds predecessors",
          "[forward-drizzle-v2][gate10][production]") {
  auto f = make_fixture(ColorMode::MONO, BayerPattern::GRBG, 0, 0, 1);
  f.plan.source_identity_hash = "src-id";
  f.plan.plan_hash = "sampling-hash";
  config::ReconstructionDrizzleConfig dc;
  dc.internal_scale = 1;
  dc.output_scale = 1;
  dc.pixfrac = 0.8f;
  config::ReconstructionClippingConfig cc;
  config::ReconstructionMultibandConfig mc;
  mc.enabled = true;
  mc.levels = 3;

  const auto base = make_forward_drizzle_v2_run_plan(
      f.plan, dc, cc, mc, true,
      static_cast<std::uint64_t>(f.plan.frames.size()), "cache-a", "q-a",
      "sqc-a", "cfg-a", std::size_t{512} << 20);
  REQUIRE(base.native_width == f.plan.canvas_width_native);
  REQUIRE(base.native_height == f.plan.canvas_height_native);
  REQUIRE(base.channels == 1);
  REQUIRE(base.internal_scale == 1);
  REQUIRE_FALSE(base.x_tiled);
  REQUIRE(base.emit_profiles);
  REQUIRE(base.multiband_levels == 3);
  REQUIRE(base.halo_rows == multiband_fusion_halo_rows(3));
  REQUIRE(base.band_rows >= 1);
  REQUIRE(base.band_rows <= base.native_height);
  REQUIRE(base.band_count * base.band_rows >= base.native_height);
  REQUIRE(base.band_count == (base.native_height + base.band_rows - 1) /
                                 base.band_rows);
  REQUIRE(base.frame_count == f.plan.frames.size());
  REQUIRE(base.min_clip_contributors == dc.min_clip_contributors);
  REQUIRE(base.robust_passes == dc.robust_passes);
  REQUIRE(base.sigma_low == Catch::Approx(cc.clip_sigma_low));
  REQUIRE(base.sigma_high == Catch::Approx(cc.clip_sigma_high));
  REQUIRE(base.fine_quality_exponent == Catch::Approx(mc.fine_quality_exponent));
  REQUIRE(base.medium_quality_exponent == Catch::Approx(mc.medium_quality_exponent));
  REQUIRE_FALSE(base.plan_hash.empty());
  REQUIRE(base.local_warp_representation == "affine_only");

  SECTION("every bound predecessor identity changes the hash") {
    for (int which = 0; which < 5; ++which) {
      std::array<std::string, 4> ids{"cache-a", "q-a", "sqc-a", "cfg-a"};
      ids[which % 4] = "changed";
      std::uint64_t frames = f.plan.frames.size();
      if (which == 4) frames += 1;
      const auto other = make_forward_drizzle_v2_run_plan(
          f.plan, dc, cc, mc, true, frames, ids[0], ids[1], ids[2], ids[3],
          std::size_t{512} << 20);
      REQUIRE(other.plan_hash != base.plan_hash);
    }
    // Sampling identity is bound, too.
    auto s2 = f.plan;
    s2.plan_hash = "sampling-hash-changed";
    const auto other = make_forward_drizzle_v2_run_plan(
        s2, dc, cc, mc, true,
        static_cast<std::uint64_t>(f.plan.frames.size()), "cache-a", "q-a",
        "sqc-a", "cfg-a", std::size_t{512} << 20);
    REQUIRE(other.plan_hash != base.plan_hash);
  }

  SECTION("band geometry follows the deterministic budgets") {
    // The planner sizes reservoir slots by the exact global keep set, not
    // 2*R.
    const auto selected = forward_drizzle_v2_selected_frame_orders(
        f.plan.frames.size(), 64, 11400714819323198485ULL);
    const std::size_t bpp = forward_drizzle_v2_cpu_bytes_per_native_pixel(
        1, 1, static_cast<int>(std::max<std::size_t>(1, selected.size())),
        true, true);
    // N=610 keeps 58 slots; the old 2*R sizing reserved 128.
    const auto sel610 = forward_drizzle_v2_selected_frame_orders(
        610, 64, 11400714819323198485ULL);
    REQUIRE(sel610.size() == 58);
    REQUIRE(forward_drizzle_v2_cpu_bytes_per_native_pixel(
                1, 1, static_cast<int>(sel610.size()), true, true) <
            forward_drizzle_v2_cpu_bytes_per_native_pixel(1, 1, 128, true,
                                                          true));
    const std::size_t row_bytes = bpp * base.native_width;
    // host_budget/2 bounds the band workspace (minus the meta table).
    REQUIRE(static_cast<std::size_t>(base.band_rows) * row_bytes <=
            (std::size_t{512} << 20) / 2);
    // A tighter nominal device bound shrinks band_rows deterministically.
    const auto capped = make_forward_drizzle_v2_run_plan(
        f.plan, dc, cc, mc, true,
        static_cast<std::uint64_t>(f.plan.frames.size()), "cache-a", "q-a",
        "sqc-a", "cfg-a", std::size_t{512} << 20, 4 * row_bytes);
    REQUIRE(capped.band_rows <= 4);
    // A budget that cannot fit one row fails closed.
    REQUIRE_THROWS_AS(make_forward_drizzle_v2_run_plan(
                          f.plan, dc, cc, mc, true,
                          static_cast<std::uint64_t>(f.plan.frames.size()),
                          "cache-a", "q-a", "sqc-a", "cfg-a",
                          row_bytes - 1),
                      std::invalid_argument);
  }

  SECTION("CUDA nominal device budget is 2 GiB and bounds A2 band geometry") {
    // Production passes kV2NominalDeviceDynamicBytes as the device budget
    // when "cuda" is requested. For the real A2 geometry (OSC 3ch,
    // internal_scale 2, emit_profiles, 4526-wide canvas, N=610 -> 58
    // reservoir slots) the dynamic per-pixel band state must resolve to
    // band_rows <= 48 so it stays within 2 GiB; the fixed full-source
    // compatibility/sample/compact-Q buffers (~431 MiB at 3840x2160) plus
    // this cap stay under 2.5 GiB total, matching the proven
    // bench_m42_v2_spans reservation of 2,432,577,804 bytes.
    REQUIRE(kV2NominalDeviceDynamicBytes == (std::size_t{2} << 30));
    constexpr std::size_t kFixedDeviceBytes =
        std::size_t{431} << 20;  // documented fixed-buffer bound at 3840x2160
    REQUIRE(kV2NominalDeviceDynamicBytes + kFixedDeviceBytes <=
            (std::size_t{5} << 29));  // 2.5 GiB
    RegistrationSamplingPlan a2;
    a2.source_width = 3840;
    a2.source_height = 2160;
    a2.canvas_width_native = 4526;
    a2.canvas_height_native = 3370;
    a2.color_mode = ColorMode::OSC;
    a2.bayer_pattern = BayerPattern::GRBG;
    a2.source_identity_hash = "a2-src";
    a2.plan_hash = "a2-sampling";
    auto d2 = dc;
    d2.internal_scale = 2;
    d2.output_scale = 1;
    const auto sel610 = forward_drizzle_v2_selected_frame_orders(
        610, 64, 11400714819323198485ULL);
    const std::size_t bpp = forward_drizzle_v2_cpu_bytes_per_native_pixel(
        3, 2, static_cast<int>(sel610.size()), true, true);
    const std::size_t row_bytes = bpp * 4526;
    const auto a2plan = make_forward_drizzle_v2_run_plan(
        a2, d2, cc, mc, true, 610, "cache-a", "q-a", "sqc-a", "cfg-a",
        std::size_t{16384} << 20, kV2NominalDeviceDynamicBytes);
    INFO("a2 band_rows=" << a2plan.band_rows
                         << " row_bytes=" << row_bytes);
    REQUIRE(a2plan.band_rows <= 48);
    REQUIRE(static_cast<std::size_t>(a2plan.band_rows) * row_bytes <=
            kV2NominalDeviceDynamicBytes);
    // Sanity: the old 4 GiB cap allowed ~96 rows; the new cap must not
    // silently re-admit it.
    const auto old = make_forward_drizzle_v2_run_plan(
        a2, d2, cc, mc, true, 610, "cache-a", "q-a", "sqc-a", "cfg-a",
        std::size_t{16384} << 20, std::size_t{4} << 30);
    REQUIRE(old.band_rows > a2plan.band_rows);
  }

  SECTION("unsupported output grids reject") {
    auto d22 = dc;
    d22.internal_scale = 2;
    d22.output_scale = 2;
    REQUIRE_THROWS_AS(make_forward_drizzle_v2_run_plan(
                          f.plan, d22, cc, mc, true, 5, "c", "q", "s", "g",
                          std::size_t{512} << 20),
                      std::invalid_argument);
    auto d21 = dc;
    d21.internal_scale = 2;  // 2/1 is supported (native output grid)
    const auto p21 = make_forward_drizzle_v2_run_plan(
        f.plan, d21, cc, mc, true, 5, "c", "q", "s", "g",
        std::size_t{512} << 20);
    REQUIRE(p21.internal_scale == 2);
    REQUIRE(p21.native_width == f.plan.canvas_width_native);
  }

  SECTION("local warp presence binds the representation") {
    auto s_local = f.plan;
    float cx[16] = {}, cy[16] = {};
    cx[5] = 0.4f;
    s_local.frames[0] = v2_local_frame(
        s_local.frames[0],
        v2_local_model(f.plan.canvas_height_native,
                       f.plan.canvas_width_native, cx, cy));
    const auto local = make_forward_drizzle_v2_run_plan(
        s_local, dc, cc, mc, true,
        static_cast<std::uint64_t>(f.plan.frames.size()), "cache-a", "q-a",
        "sqc-a", "cfg-a", std::size_t{512} << 20);
    REQUIRE(local.local_warp_representation == "smooth_local_coefficients");
    REQUIRE(local.plan_hash != base.plan_hash);
  }
}

TEST_CASE("forward drizzle v2 gate10 sigma2 plane follows the model",
          "[forward-drizzle-v2][gate10][production]") {
  // Linear ramp: central differences are exact, one-sided borders agree.
  Matrix2Df ramp(4, 5);
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 5; ++x) ramp(y, x) = static_cast<float>(x + 2 * y);
  const auto s2 =
      forward_drizzle_v2_sigma2_plane(ramp, 0.3, 0.05, 0.4 /*half*/);
  REQUIRE(s2.size() == 20);
  const double expect =
      forward_drizzle_v2_sigma2_model(0.3, 1.0, 2.0, 0.05, 0.4);
  for (float v : s2) REQUIRE(v == Catch::Approx(expect).epsilon(1e-6));

  SECTION("non-finite or negative inputs degrade the whole plane") {
    const auto bad = forward_drizzle_v2_sigma2_plane(ramp, -1.0, 0.05, 0.4);
    for (float v : bad) REQUIRE_FALSE(std::isfinite(v));
    const auto nan = forward_drizzle_v2_sigma2_plane(
        ramp, std::numeric_limits<double>::quiet_NaN(), 0.05, 0.4);
    for (float v : nan) REQUIRE_FALSE(std::isfinite(v));
  }

  SECTION("non-finite source neighbours degrade only their footprint") {
    Matrix2Df img = ramp;
    img(1, 2) = std::numeric_limits<float>::quiet_NaN();
    const auto p = forward_drizzle_v2_sigma2_plane(img, 0.3, 0.05, 0.4);
    REQUIRE(p.size() == 20);
    // The NaN pixel itself keeps a finite sigma2 (its gradient is built from
    // finite neighbours); the kernel treats a non-finite source as absent.
    REQUIRE(std::isfinite(p[1 * 5 + 2]));
    // All other pixels stay finite --- differences are one-sided at the
    // hole rather than poisoning it.
    for (float v : p) REQUIRE(std::isfinite(v));
  }

  SECTION("degenerate single-column source yields zero x-gradient") {
    Matrix2Df col(4, 1);
    for (int y = 0; y < 4; ++y) col(y, 0) = static_cast<float>(y);
    const auto p = forward_drizzle_v2_sigma2_plane(col, 0.3, 0.05, 0.4);
    const double e = forward_drizzle_v2_sigma2_model(0.3, 0.0, 1.0, 0.05, 0.4);
    for (float v : p) REQUIRE(v == Catch::Approx(e).epsilon(1e-6));
  }
}

TEST_CASE("forward drizzle v2 gate10 v2 store fuses to the reference image",
          "[forward-drizzle-v2][gate10][fusion][production]") {
  V2StoreFixture fx;
  auto f = make_fixture(ColorMode::MONO, BayerPattern::GRBG, 0, 0, 1);
  const int nc = f.plan.canvas_width_native;
  const int nr = f.plan.canvas_height_native;
  const std::size_t n_src =
      static_cast<std::size_t>(f.plan.source_width) * f.plan.source_height;
  std::vector<float> s2(n_src, 0.02f);
  std::vector<ForwardDrizzleV2FrameMeta> meta(f.plan.frames.size());
  for (std::size_t i = 0; i < meta.size(); ++i)
    meta[i] = {1.0f, 0.9f, static_cast<std::uint8_t>(i % 2), 0};

  // One fusion level: the atrous coarse support at level 3 needs a 14-pixel
  // reach, which the 14x13 fixture canvas can never satisfy; level 1 keeps
  // the full store -> adapter -> striped-fusion -> FITS/spool contract while
  // still producing supported pixels.
  auto plan = g10_driver_plan(f, 4, 1, true);
  plan.multiband_levels = 1;
  finalize_forward_drizzle_v2_run_plan(plan);
  ForwardDrizzleV2DriverOptions opts;
  opts.prefer_cuda = false;
  const auto done =
      run_forward_drizzle_v2(fx.root, plan, f.plan.source_width,
                             f.plan.source_height,
                             g10_provider(f, s2, &meta), opts);
  REQUIRE(done.committed);

  // Store autodetection: the published plan round-trips and binds.
  ForwardDrizzleV2RunPlan detected;
  std::string load_error;
  REQUIRE(load_forward_drizzle_v2_published_plan(fx.root, detected,
                                               load_error));
  REQUIRE(detected.plan_hash == plan.plan_hash);
  // A complete inspection now exposes the verified band list.
  const auto insp = inspect_forward_drizzle_v2_store(fx.root, detected);
  REQUIRE(insp.status == ForwardDrizzleV2StoreStatus::complete);
  REQUIRE(insp.committed.size() ==
          static_cast<std::size_t>(plan.band_count));

  config::ReconstructionMultibandConfig mb;
  mb.enabled = true;
  mb.levels = 1;
  const auto contract = multiband_store_contract_from_config(mb);

  MultibandCandidateLuma cand;
  MultibandCandidateSpool spool;
  spool.dir = fx.root / "spool";
  fs::create_directories(spool.dir);
  MultibandFusionMemoryPlan mp;
  ForwardDrizzleV2FusionStats stats;
  const auto fits_path = fx.root / "reconstruction_multiband.fits";
  const long long px = fuse_multiband_v2_store_to_image(
      fx.root, detected, fits_path, mb, /*chunk_rows=*/2, 256, &cand, &spool,
      &mp, &stats);
  REQUIRE(px > 0);
  REQUIRE(stats.bands_decoded ==
          static_cast<std::uint64_t>(plan.band_count));
  const std::size_t nplane = static_cast<std::size_t>(nc) * nr;
  REQUIRE(stats.record_bytes_read ==
          static_cast<std::uint64_t>(nplane) *
              sizeof(ForwardDrizzleV2ProfileResult));
  REQUIRE(stats.record_bytes_no_reuse > stats.record_bytes_read);
  REQUIRE(mp.fits);
  REQUIRE(spool.populated);
  REQUIRE(fs::file_size(fits_path) > 0);
  REQUIRE(cand.width == nc);
  REQUIRE(cand.height == nr);

  // Parity: the v2-adapted striped fusion must equal the whole-canvas
  // fusion over the same committed profile records.
  std::vector<ForwardDrizzleV2PixelResult> records;
  std::vector<ForwardDrizzleV2ProfileResult> profiles;
  g10_read_store(fx.root, plan, records, &profiles);
  const auto U = forward_drizzle_v2_profiles_to_uniform_result(
      profiles, nc, nr, 1, ColorMode::MONO, 0);
  const auto R = forward_drizzle_v2_profiles_to_uniform_result(
      profiles, nc, nr, 1, ColorMode::MONO, 1);
  const auto F = forward_drizzle_v2_profiles_to_uniform_result(
      profiles, nc, nr, 1, ColorMode::MONO, 2);
  const auto M = forward_drizzle_v2_profiles_to_uniform_result(
      profiles, nc, nr, 1, ColorMode::MONO, 3);
  const auto a_sep =
      forward_drizzle_v2_profile_alpha_plane(profiles, nc, nr, 1, 0);
  const auto a_art =
      forward_drizzle_v2_profile_alpha_plane(profiles, nc, nr, 1, 1);
  const auto a_reg =
      forward_drizzle_v2_profile_alpha_plane(profiles, nc, nr, 1, 2);
  const auto whole =
      fuse_multiband(U, R, F, M, ColorMode::MONO, nc, nr, mb, contract.alpha,
                     contract.guard, a_sep, a_art, a_reg, {});
  REQUIRE(cand.multiband_luma.size() == whole.L.size());
  REQUIRE(cand.uniform_luma.size() == whole.L.size());
  long long support_parity = 0;
  for (std::size_t i = 0; i < whole.L.size(); ++i) {
    const float mv = cand.multiband_luma[i], wv = whole.L[i];
    REQUIRE((mv == wv || (std::isnan(mv) && std::isnan(wv))));
    const float uv = cand.uniform_luma[i], uref = U.L.value[i];
    REQUIRE((uv == uref || (std::isnan(uv) && std::isnan(uref))));
    REQUIRE((cand.uniform_support[i] != 0) == (U.L.support[i] != 0));
    if (cand.uniform_support[i]) ++support_parity;
  }
  REQUIRE(support_parity > 0);
  // The spool's multiband plane carries the same fused values.
  const auto mb_plane =
      read_candidate_spool_plane(spool, "multiband", 0);
  REQUIRE(mb_plane.size() == nplane);
  const auto raw_plane = read_candidate_spool_plane(spool, "raw", 0);
  for (std::size_t i = 0; i < nplane; ++i) {
    const float sv = mb_plane[i], wv = whole.L[i];
    REQUIRE((sv == wv || (std::isnan(sv) && std::isnan(wv))));
    const float rv = raw_plane[i], rref = R.L.value[i];
    REQUIRE((rv == rref || (std::isnan(rv) && std::isnan(rref))));
  }
}

TEST_CASE("forward drizzle v2 gate10 fusion fails closed",
          "[forward-drizzle-v2][gate10][fusion][production]") {
  auto f = make_fixture(ColorMode::MONO, BayerPattern::GRBG, 0, 0, 1);
  config::ReconstructionMultibandConfig mb;
  mb.enabled = true;
  mb.levels = 3;
  const auto plan = g10_driver_plan(f, 4, 1, true);

  SECTION("no published store") {
    V2StoreFixture fx;
    ForwardDrizzleV2RunPlan p;
    std::string e;
    REQUIRE_FALSE(load_forward_drizzle_v2_published_plan(fx.root, p, e));
    REQUIRE_THROWS_AS(
        fuse_multiband_v2_store_to_image(fx.root, plan, fx.root / "o.fits",
                                         mb, 2, 256),
        std::runtime_error);
  }

  SECTION("resumable-only store is not fused") {
    V2StoreFixture fx;
    const std::size_t n_src =
        static_cast<std::size_t>(f.plan.source_width) * f.plan.source_height;
    std::vector<float> s2(n_src, 0.02f);
    ForwardDrizzleV2DriverOptions opts;
    opts.prefer_cuda = false;
    opts.simulate_kill_after_bands = 1;
    try {
      (void)run_forward_drizzle_v2(fx.root, plan, f.plan.source_width,
                                   f.plan.source_height,
                                   g10_provider(f, s2, nullptr), opts);
      FAIL("expected simulated kill");
    } catch (const ForwardDrizzleV2SimulatedKill &) {
    }
    ForwardDrizzleV2RunPlan p;
    std::string e;
    REQUIRE_FALSE(load_forward_drizzle_v2_published_plan(fx.root, p, e));
    const auto insp = inspect_forward_drizzle_v2_store(fx.root, plan);
    REQUIRE(insp.status == ForwardDrizzleV2StoreStatus::resumable);
    REQUIRE_THROWS_AS(
        fuse_multiband_v2_store_to_image(fx.root, plan, fx.root / "o.fits",
                                         mb, 2, 256),
        std::runtime_error);
  }

  SECTION("foreign plan rejected") {
    V2StoreFixture fx;
    const std::size_t n_src =
        static_cast<std::size_t>(f.plan.source_width) * f.plan.source_height;
    std::vector<float> s2(n_src, 0.02f);
    ForwardDrizzleV2DriverOptions opts;
    opts.prefer_cuda = false;
    const auto done =
        run_forward_drizzle_v2(fx.root, plan, f.plan.source_width,
                               f.plan.source_height,
                               g10_provider(f, s2, nullptr), opts);
    REQUIRE(done.committed);
    auto foreign = plan;
    foreign.pixfrac = 0.5;
    finalize_forward_drizzle_v2_run_plan(foreign);
    REQUIRE_THROWS_AS(
        fuse_multiband_v2_store_to_image(fx.root, foreign, fx.root / "o.fits",
                                         mb, 2, 256),
        std::runtime_error);
  }

  SECTION("corrupted profile payload rejected") {
    V2StoreFixture fx;
    const std::size_t n_src =
        static_cast<std::size_t>(f.plan.source_width) * f.plan.source_height;
    std::vector<float> s2(n_src, 0.02f);
    ForwardDrizzleV2DriverOptions opts;
    opts.prefer_cuda = false;
    const auto done =
        run_forward_drizzle_v2(fx.root, plan, f.plan.source_width,
                               f.plan.source_height,
                               g10_provider(f, s2, nullptr), opts);
    REQUIRE(done.committed);
    std::ofstream corrupt(done.generation_dir / "band-0000.profiles.bin",
                          std::ios::binary | std::ios::app);
    const char junk[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    corrupt.write(junk, 8);
    corrupt.close();
    ForwardDrizzleV2RunPlan detected;
    std::string e;
    REQUIRE(load_forward_drizzle_v2_published_plan(fx.root, detected, e));
    REQUIRE_THROWS_AS(
        fuse_multiband_v2_store_to_image(fx.root, detected,
                                         fx.root / "o.fits", mb, 2, 256),
        std::runtime_error);
  }
}

TEST_CASE("forward drizzle v2 sigma2 rect slices the whole-plane oracle",
          "[forward-drizzle-v2]") {
  const int sw = 9, sh = 7;
  Matrix2Df src(sh, sw);
  for (int y = 0; y < sh; ++y)
    for (int x = 0; x < sw; ++x)
      src(y, x) = static_cast<float>(5.0 + 0.9 * x + 1.1 * y +
                                     std::sin(0.7 * x - 0.3 * y));
  src(2, 3) = std::numeric_limits<float>::quiet_NaN();
  src(0, 4) = std::numeric_limits<float>::quiet_NaN();
  src(6, 0) = -1.0f;  // negative -> non-finite plane value
  const auto whole = forward_drizzle_v2_sigma2_plane(src, 0.3, 0.05, 0.4);
  REQUIRE(whole.size() == static_cast<std::size_t>(sw) * sh);

  // Interior + every true source border + NaN-adjacent rects.
  const int rects[][4] = {{2, 2, 4, 3}, {0, 0, 3, 2}, {6, 0, 3, 4},
                          {0, 5, 4, 2}, {5, 4, 4, 3}, {0, 0, 9, 7},
                          {1, 1, 4, 3}};
  for (const auto &rc : rects) {
    const int x0 = rc[0], y0 = rc[1], w = rc[2], h = rc[3];
    const int hx0 = std::max(0, x0 - 1), hy0 = std::max(0, y0 - 1);
    const int hx1 = std::min(sw, x0 + w + 1), hy1 = std::min(sh, y0 + h + 1);
    Matrix2Df win(hy1 - hy0, hx1 - hx0);
    for (int r = 0; r < win.rows(); ++r)
      for (int c = 0; c < win.cols(); ++c)
        win(r, c) = src(hy0 + r, hx0 + c);
    const auto rect = forward_drizzle_v2_sigma2_rect(
        win, hx0, hy0, x0, y0, w, h, sw, sh, 0.3, 0.05, 0.4);
    REQUIRE(rect.size() == static_cast<std::size_t>(w) * h);
    for (int r = 0; r < h; ++r)
      for (int c = 0; c < w; ++c) {
        const float want =
            whole[static_cast<std::size_t>(y0 + r) * sw + x0 + c];
        const float got = rect[static_cast<std::size_t>(r) * w + c];
        REQUIRE(std::memcmp(&got, &want, sizeof(float)) == 0);
      }
  }
}

TEST_CASE("forward drizzle v2 gate10 host kernel source windows match "
          "full source",
          "[forward-drizzle-v2][gate10]") {
  const BayerPattern patterns[] = {BayerPattern::RGGB, BayerPattern::BGGR,
                                   BayerPattern::GRBG, BayerPattern::GBRG};
  for (int scale : {1, 2})
    for (ColorMode mode : {ColorMode::MONO, ColorMode::OSC})
      for (BayerPattern pattern : patterns) {
        auto f = make_fixture(mode, pattern, 1, -1, scale);
        const int nc = f.plan.canvas_width_native;
        const int nr = f.plan.canvas_height_native;
        const int channels = mode == ColorMode::MONO ? 1 : 3;
        const int sw = f.plan.source_width, sh = f.plan.source_height;
        const std::size_t n_src = static_cast<std::size_t>(sw) * sh;
        std::vector<float> s2(n_src);
        for (std::size_t i = 0; i < n_src; ++i)
          s2[i] = 0.01f + 0.0001f * static_cast<float>(i % 97);

        auto make_kcfg = [&](int y_begin) {
          ForwardDrizzleV2KernelConfig k;
          k.internal_scale = scale;
          k.stream_length = f.plan.frames.size();
          k.half = 0.5 * f.cfg.pixfrac;
          k.bayer_pattern = static_cast<int>(pattern);
          k.cfa_origin_x = f.plan.cfa_origin_x;
          k.cfa_origin_y = f.plan.cfa_origin_y;
          k.mono = mode == ColorMode::MONO;
          k.canvas_width_native = nc;
          k.canvas_height_native = nr;
          k.band_origin_y_native = y_begin;
          return k;
        };
        auto pack = [](const Matrix2Df &img, int x0, int y0, int w, int h) {
          std::vector<float> out(static_cast<std::size_t>(w) * h);
          for (int r = 0; r < h; ++r)
            for (int c = 0; c < w; ++c)
              out[static_cast<std::size_t>(r) * w + c] = img(y0 + r, x0 + c);
          return out;
        };
        auto run = [&](int y_begin, int rows, bool windowed) {
          ForwardDrizzleV2CpuKernel kernel;
          REQUIRE(kernel.reserve(nc, rows, sw, sh, make_kcfg(y_begin)));
          std::uint64_t skipped = 0;
          for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
            const auto &m = f.plan.frames[fr].source_to_canvas;
            const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                                  m(1, 0), m(1, 1), m(1, 2)};
            if (!windowed) {
              REQUIRE(kernel.accumulate_frame(a6, f.images[fr].data(),
                                              s2.data(), fr));
              continue;
            }
            const auto box = drizzle_source_scan_box(
                f.plan, f.plan.frames[fr], scale, y_begin * scale,
                rows * scale);
            const int bw = box.x1 - box.x0, bh = box.y1 - box.y0;
            if (bw <= 0 || bh <= 0) {
              REQUIRE(kernel.skip_frame(fr));
              ++skipped;
              continue;
            }
            const ForwardDrizzleV2SourceWindow w{box.x0, box.y0, bw, bh};
            const auto src_w = pack(f.images[fr], box.x0, box.y0, bw, bh);
            std::vector<float> s2w(static_cast<std::size_t>(bw) * bh);
            for (int r = 0; r < bh; ++r)
              for (int c = 0; c < bw; ++c)
                s2w[static_cast<std::size_t>(r) * bw + c] =
                    s2[static_cast<std::size_t>(box.y0 + r) * sw + box.x0 +
                       c];
            REQUIRE(kernel.accumulate_frame_window(a6, w, src_w.data(),
                                                   s2w.data(), nullptr, fr));
          }
          std::vector<ForwardDrizzleV2PixelResult> out(
              static_cast<std::size_t>(rows) * nc * channels);
          std::uint64_t dense = 0;
          REQUIRE(kernel.finalize(out.data(), nullptr, &dense));
          REQUIRE(kernel.stats().frames_skipped_empty_window == skipped);
          return std::pair{out, kernel.stats()};
        };

        for (int y0 = 0; y0 < nr; y0 += 5) {
          const int rows = std::min(5, nr - y0);
          const auto full = run(y0, rows, false);
          const auto win = run(y0, rows, true);
          // The exact scan box can only drop source pixels whose leaves
          // clamp empty: CPU output is bit-identical.
          for (std::size_t i = 0; i < full.first.size(); ++i)
            REQUIRE(g10_bit_eq(full.first[i], win.first[i]));
          REQUIRE(win.second.frames_processed + win.second
                          .frames_skipped_empty_window ==
                  f.plan.frames.size());
          REQUIRE(win.second.slot_transitions == f.plan.frames.size());
        }
      }
}

TEST_CASE("forward drizzle v2 gate10 driver windowed provider contract",
          "[forward-drizzle-v2][gate10][driver]") {
  V2StoreFixture fx;
  auto f = make_fixture(ColorMode::MONO, BayerPattern::GRBG, 0, 0, 1);
  const int nc = f.plan.canvas_width_native;
  const int nr = f.plan.canvas_height_native;
  const int sw = f.plan.source_width, sh = f.plan.source_height;
  const std::size_t n_src = static_cast<std::size_t>(sw) * sh;
  std::vector<float> s2(n_src, 0.02f);
  std::vector<ForwardDrizzleV2FrameMeta> meta(f.plan.frames.size());
  for (std::size_t i = 0; i < meta.size(); ++i)
    meta[i] = {0.9f, 0.8f, static_cast<std::uint8_t>(i % 2), 0};

  const auto plan = g10_driver_plan(f, 4, 1, true);
  std::vector<std::pair<int, int>> bands_seen;
  std::uint64_t expected_samples = 0, expected_skips = 0;
  Matrix2Df win;
  std::vector<float> s2w;
  ForwardDrizzleV2FrameProvider provider =
      [&](int band_y_begin, int band_rows, std::uint64_t order,
          const ForwardDrizzleV2FramePieceSink &sink) -> bool {
    if (order >= f.plan.frames.size()) return false;
    bands_seen.emplace_back(band_y_begin, band_rows);
    const auto &frame = f.plan.frames[order];
    const auto &m = frame.source_to_canvas;
    ForwardDrizzleV2FrameInput in;
    const double a6[6] = {m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1),
                          m(1, 2)};
    std::copy_n(a6, 6, in.affine6);
    in.has_local_model = false;
    in.quality = {};
    in.skip = false;
    in.meta = meta[order];
    const auto box = drizzle_source_scan_box(
        f.plan, frame, plan.internal_scale,
        band_y_begin * plan.internal_scale,
        band_rows * plan.internal_scale);
    const int bw = box.x1 - box.x0, bh = box.y1 - box.y0;
    if (bw <= 0 || bh <= 0) {
      in.skip = true;
      ++expected_skips;
      return sink(in);
    }
    in.source_window = {box.x0, box.y0, bw, bh};
    win.resize(bh, bw);
    s2w.resize(static_cast<std::size_t>(bw) * bh);
    for (int r = 0; r < bh; ++r)
      for (int c = 0; c < bw; ++c) {
        win(r, c) = f.images[order](box.y0 + r, box.x0 + c);
        s2w[static_cast<std::size_t>(r) * bw + c] =
            s2[static_cast<std::size_t>(box.y0 + r) * sw + box.x0 + c];
      }
    in.source = win.data();
    in.sigma2 = s2w.data();
    expected_samples += static_cast<std::uint64_t>(bw) * bh;
    return sink(in);
  };

  ForwardDrizzleV2DriverOptions opts;
  opts.prefer_cuda = false;
  const auto result =
      run_forward_drizzle_v2(fx.root, plan, sw, sh, provider, opts);
  REQUIRE(result.committed);
  REQUIRE(result.backend_used == "cpu_v2");

  // The provider observed every (band, frame) pair with the driver's band
  // coordinates.
  REQUIRE(bands_seen.size() ==
          static_cast<std::size_t>(plan.band_count) * plan.frame_count);
  for (std::size_t i = 0; i < bands_seen.size(); ++i) {
    const int band = static_cast<int>(i / plan.frame_count);
    REQUIRE(bands_seen[i].first == band * plan.band_rows);
    REQUIRE(bands_seen[i].second ==
            std::min(plan.band_rows, nr - band * plan.band_rows));
  }

  // Honest accounting: only nonempty windows launched samples; skips are
  // counted separately and still advance the stream.
  REQUIRE(result.totals.source_samples_launched == expected_samples);
  REQUIRE(result.totals.frames_skipped_empty_window == expected_skips);
  REQUIRE(expected_skips > 0);
  REQUIRE(result.totals.frames_processed + expected_skips ==
          plan.frame_count * static_cast<std::uint64_t>(plan.band_count));
  REQUIRE(result.totals.slot_transitions ==
          plan.frame_count * static_cast<std::uint64_t>(plan.band_count));

  // Records must be bitwise identical to direct full-source per-band runs.
  std::vector<ForwardDrizzleV2PixelResult> records;
  g10_read_store(fx.root, plan, records, nullptr);
  for (int band = 0; band < plan.band_count; ++band) {
    const int y0 = band * plan.band_rows;
    const int rows = std::min(plan.band_rows, nr - y0);
    ForwardDrizzleV2KernelConfig k;
    k.internal_scale = plan.internal_scale;
    k.stream_length = plan.frame_count;
    k.min_clip_contributors = plan.min_clip_contributors;
    k.min_candidates = plan.min_candidates;
    k.robust_passes = plan.robust_passes;
    k.sigma_low = plan.sigma_low;
    k.sigma_high = plan.sigma_high;
    k.half = 0.5 * plan.pixfrac;
    k.bayer_pattern = plan.bayer_pattern;
    k.cfa_origin_x = plan.cfa_origin_x;
    k.cfa_origin_y = plan.cfa_origin_y;
    k.mono = true;
    k.sigma2_plane = plan.sigma2_enabled;
    k.canvas_width_native = nc;
    k.canvas_height_native = nr;
    k.band_origin_y_native = y0;
    k.emit_profiles = true;
    k.fine_quality_exponent = plan.fine_quality_exponent;
    k.medium_quality_exponent = plan.medium_quality_exponent;
    ForwardDrizzleV2CpuKernel kernel;
    REQUIRE(kernel.reserve(nc, rows, sw, sh, k));
    for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
      const auto &m = f.plan.frames[fr].source_to_canvas;
      const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                            m(1, 0), m(1, 1), m(1, 2)};
      REQUIRE(kernel.accumulate_frame(a6, f.images[fr].data(), s2.data(),
                                      fr, nullptr, &meta[fr]));
    }
    std::vector<ForwardDrizzleV2PixelResult> direct(
        static_cast<std::size_t>(rows) * nc);
    std::vector<ForwardDrizzleV2ProfileResult> directp(direct.size());
    std::uint64_t dense = 0;
    REQUIRE(kernel.finalize(direct.data(), directp.data(), &dense));
    for (int r = 0; r < rows; ++r)
      for (int x = 0; x < nc; ++x)
        REQUIRE(g10_bit_eq(records[static_cast<std::size_t>(y0 + r) * nc + x],
                           direct[static_cast<std::size_t>(r) * nc + x]));
  }
}

TEST_CASE("forward drizzle v2 inline sigma2 model matches the explicit "
          "plane over halo windows",
          "[forward-drizzle-v2][gate10][kernel][sigma2-model]") {
  // CPU bit-parity: a halo-extended source buffer + enabled sigma2 model
  // must reproduce the explicit whole-plane sigma2 path exactly, including
  // true-border one-sided fallbacks and NaN neighbours (the fixture plants
  // one NaN per frame).
  const BayerPattern patterns[] = {BayerPattern::RGGB, BayerPattern::BGGR,
                                   BayerPattern::GRBG, BayerPattern::GBRG};
  const double noise = 0.3, reg = 0.05;
  for (ColorMode mode : {ColorMode::MONO, ColorMode::OSC})
    for (BayerPattern pattern : patterns)
      for (int scale : {1, 2}) {
        auto f = make_fixture(mode, pattern, 0, 0, scale);
        const int sw = f.plan.source_width, sh = f.plan.source_height;
        const int nc = f.plan.canvas_width_native,
                  nr = f.plan.canvas_height_native;
        const int channels = mode == ColorMode::MONO ? 1 : 3;
        const double half = 0.5 * static_cast<double>(f.cfg.pixfrac);
        std::vector<std::vector<float>> s2_full(f.plan.frames.size());
        for (std::size_t i = 0; i < f.plan.frames.size(); ++i)
          s2_full[i] = forward_drizzle_v2_sigma2_plane(f.images[i], noise,
                                                     reg, half);
        auto make_kcfg = [&](int y_begin) {
          ForwardDrizzleV2KernelConfig k;
          k.internal_scale = scale;
          k.stream_length = f.plan.frames.size();
          k.sigma2_plane = true;
          k.half = half;
          k.bayer_pattern = static_cast<int>(pattern);
          k.cfa_origin_x = f.plan.cfa_origin_x;
          k.cfa_origin_y = f.plan.cfa_origin_y;
          k.mono = mode == ColorMode::MONO;
          k.canvas_width_native = nc;
          k.canvas_height_native = nr;
          k.band_origin_y_native = y_begin;
          return k;
        };
        auto run = [&](int y_begin, int rows, bool model_path) {
          ForwardDrizzleV2CpuKernel kernel;
          REQUIRE(kernel.reserve(nc, rows, sw, sh, make_kcfg(y_begin)));
          for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
            const auto &m = f.plan.frames[fr].source_to_canvas;
            const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                                  m(1, 0), m(1, 1), m(1, 2)};
            if (!model_path) {
              REQUIRE(kernel.accumulate_frame(a6, f.images[fr].data(),
                                              s2_full[fr].data(), fr));
              continue;
            }
            const auto box = drizzle_source_scan_box(
                f.plan, f.plan.frames[fr], scale, y_begin * scale,
                rows * scale);
            const int bw = box.x1 - box.x0, bh = box.y1 - box.y0;
            if (bw <= 0 || bh <= 0) {
              REQUIRE(kernel.skip_frame(fr));
              continue;
            }
            // Halo buffer = box + one in-bounds pixel ring; the kernel's
            // inline model reads the neighbours straight out of it.
            const int hx0 = std::max(0, box.x0 - 1),
                      hy0 = std::max(0, box.y0 - 1);
            const int hx1 = std::min(sw, box.x1 + 1),
                      hy1 = std::min(sh, box.y1 + 1);
            const int hw = hx1 - hx0, hh = hy1 - hy0;
            std::vector<float> buf(static_cast<std::size_t>(hw) * hh);
            for (int r = 0; r < hh; ++r)
              for (int c = 0; c < hw; ++c)
                buf[static_cast<std::size_t>(r) * hw + c] =
                    f.images[fr](hy0 + r, hx0 + c);
            const ForwardDrizzleV2SourceWindow w{
                hx0, hy0, hw, hh,
                box.x0 - hx0, box.y0 - hy0, bw, bh};
            const ForwardDrizzleV2Sigma2FrameModel s2m{true, noise, reg,
                                                     half};
            REQUIRE(kernel.accumulate_frame_window(a6, w, buf.data(),
                                                   nullptr, &s2m, fr));
          }
          std::vector<ForwardDrizzleV2PixelResult> out(
              static_cast<std::size_t>(rows) * nc * channels);
          std::uint64_t dense = 0;
          REQUIRE(kernel.finalize(out.data(), nullptr, &dense));
          return std::pair{out, kernel.stats()};
        };
        for (int y0 = 0; y0 < nr; y0 += 5) {
          const int rows = std::min(5, nr - y0);
          const auto ref = run(y0, rows, false);
          const auto mod = run(y0, rows, true);
          for (std::size_t i = 0; i < ref.first.size(); ++i)
            REQUIRE(g10_bit_eq(ref.first[i], mod.first[i]));
          // Model mode uploads no explicit sigma2 plane: source bytes are
          // exactly the halo buffers, launched samples the ACTIVE pixels.
          std::uint64_t expect_src = 0, expect_act = 0;
          for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
            const auto box = drizzle_source_scan_box(
                f.plan, f.plan.frames[fr], scale, y0 * scale, rows * scale);
            const int bw = box.x1 - box.x0, bh = box.y1 - box.y0;
            if (bw <= 0 || bh <= 0) continue;
            const int hx0 = std::max(0, box.x0 - 1),
                      hy0 = std::max(0, box.y0 - 1);
            const int hx1 = std::min(sw, box.x1 + 1),
                      hy1 = std::min(sh, box.y1 + 1);
            expect_src += static_cast<std::uint64_t>(hx1 - hx0) *
                          (hy1 - hy0) * sizeof(float);
            expect_act += static_cast<std::uint64_t>(bw) * bh;
          }
          REQUIRE(mod.second.source_bytes_uploaded == expect_src);
          REQUIRE(mod.second.source_samples_launched == expect_act);
          REQUIRE(mod.second.frames_processed +
                          mod.second.frames_skipped_empty_window ==
                      f.plan.frames.size());
        }
      }
}

TEST_CASE("forward drizzle v2 compact quality windows match float planes",
          "[forward-drizzle-v2][gate10][kernel][packed-quality]") {
  auto f = make_fixture(ColorMode::MONO, BayerPattern::GRBG, 0, 0, 1);
  const int sw = f.plan.source_width, sh = f.plan.source_height;
  const int nc = f.plan.canvas_width_native, nr = f.plan.canvas_height_native;
  const std::size_t n_src = static_cast<std::size_t>(sw) * sh;
  std::vector<float> s2(n_src, 0.02f);
  std::vector<ForwardDrizzleV2FrameMeta> meta(f.plan.frames.size());
  for (std::size_t i = 0; i < meta.size(); ++i)
    meta[i] = {0.9f, 0.8f, static_cast<std::uint8_t>(i % 2), 0};

  // Synthetic storage grid (divisor 2) for all four streams: quantised
  // cells, one hard-veto cell and one zero (no-data) cell.
  const int d = 2;
  const int gw = (sw + d - 1) / d, gh = (sh + d - 1) / d;
  const std::size_t gcells = static_cast<std::size_t>(gw) * gh;
  std::vector<std::uint16_t> cells[4];
  std::vector<std::uint8_t> veto[4];
  for (int s = 0; s < 4; ++s) {
    cells[s].resize(gcells);
    veto[s].resize(gcells, 0);
    for (std::size_t i = 0; i < gcells; ++i)
      cells[s][i] = static_cast<std::uint16_t>(
          1 + (i * 977 + s * 131) % 65534);
    cells[s][gcells / 3] = 0;              // zero cell -> NaN
    veto[s][(2 * gcells) / 3] = 1;         // hard veto -> NaN
  }
  // Float equivalent planes: decode cell by ABSOLUTE source coordinate.
  auto float_plane = [&](int s) {
    std::vector<float> out(n_src);
    for (int y = 0; y < sh; ++y)
      for (int x = 0; x < sw; ++x) {
        const std::size_t i = static_cast<std::size_t>(y / d) * gw + x / d;
        out[static_cast<std::size_t>(y) * sw + x] =
            veto[s][i] ? std::numeric_limits<float>::quiet_NaN()
                       : dequantize_quality(cells[s][i]);
      }
    return out;
  };
  std::vector<float> fq[4] = {float_plane(0), float_plane(1),
                              float_plane(2), float_plane(3)};

  // Non-divisor-aligned window {x=1,y=1,w=5,h=4}: covering storage cells
  // are cx in [0,3), cy in [0,3).
  const ForwardDrizzleV2SourceWindow win{1, 1, 5, 4};
  const int cx0 = 1 / d, cy0 = 1 / d;
  const int cx1 = (1 + 5 - 1) / d, cy1 = (1 + 4 - 1) / d;
  const int pww = cx1 - cx0 + 1, pwh = cy1 - cy0 + 1;
  std::vector<std::uint16_t> pcells[4];
  std::vector<std::uint8_t> pveto[4];
  for (int s = 0; s < 4; ++s)
    for (int cy = cy0; cy <= cy1; ++cy)
      for (int cx = cx0; cx <= cx1; ++cx) {
        pcells[s].push_back(cells[s][static_cast<std::size_t>(cy) * gw + cx]);
        pveto[s].push_back(veto[s][static_cast<std::size_t>(cy) * gw + cx]);
      }

  auto make_kcfg = [&] {
    ForwardDrizzleV2KernelConfig k;
    k.internal_scale = 1;
    k.stream_length = f.plan.frames.size();
    k.sigma2_plane = true;
    k.emit_profiles = true;
    k.half = 0.5 * f.cfg.pixfrac;
    k.bayer_pattern = static_cast<int>(BayerPattern::GRBG);
    k.canvas_width_native = nc;
    k.canvas_height_native = nr;
    k.mono = true;
    return k;
  };
  auto run = [&](bool packed, std::vector<ForwardDrizzleV2PixelResult> &recs,
                 std::vector<ForwardDrizzleV2ProfileResult> &profs) {
    ForwardDrizzleV2CpuKernel kernel;
    REQUIRE(kernel.reserve(nc, nr, sw, sh, make_kcfg()));
    for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
      const auto &m = f.plan.frames[fr].source_to_canvas;
      const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                            m(1, 0), m(1, 1), m(1, 2)};
      std::vector<float> src_w(20), s2w(20);
      std::vector<float> qw[4];
      ForwardDrizzleV2FrameQuality q;
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 5; ++c) {
          const std::size_t t = static_cast<std::size_t>(r) * 5 + c;
          src_w[t] = f.images[fr](1 + r, 1 + c);
          s2w[t] = s2[static_cast<std::size_t>(1 + r) * sw + 1 + c];
          for (int s = 0; s < 4; ++s)
            qw[s].push_back(fq[s][static_cast<std::size_t>(1 + r) * sw +
                                  1 + c]);
        }
      if (packed) {
        auto fill = [&](int s, ForwardDrizzleV2PackedQualityPlane &p) {
          p.cells = pcells[s].data();
          p.veto = pveto[s].data();
          p.storage_x_begin = cx0;
          p.storage_y_begin = cy0;
          p.storage_width = pww;
          p.storage_height = pwh;
          p.storage_divisor = d;
        };
        fill(0, q.qc_packed);
        fill(1, q.q0_packed);
        fill(2, q.q1_packed);
        fill(3, q.qa_packed);
      } else {
        q.q_composite = qw[0].data();
        q.q_scale0 = qw[1].data();
        q.q_scale1 = qw[2].data();
        q.q_artifact = qw[3].data();
      }
      REQUIRE(kernel.accumulate_frame_window(a6, win, src_w.data(),
                                             s2w.data(), nullptr, fr, &q,
                                             &meta[fr]));
    }
    recs.assign(static_cast<std::size_t>(nc) * nr,
                ForwardDrizzleV2PixelResult{});
    profs.assign(recs.size(), ForwardDrizzleV2ProfileResult{});
    std::uint64_t dense = 0;
    REQUIRE(kernel.finalize(recs.data(), profs.data(), &dense));
    return kernel.stats();
  };
  std::vector<ForwardDrizzleV2PixelResult> ra, rb;
  std::vector<ForwardDrizzleV2ProfileResult> pa, pb;
  const auto sa = run(false, ra, pa);
  const auto sb = run(true, rb, pb);
  for (std::size_t i = 0; i < ra.size(); ++i) {
    REQUIRE(g10_bit_eq(ra[i], rb[i]));
    REQUIRE(g10_bit_eq(pa[i], pb[i]));
  }
  // Compact accounting: 3 bytes per storage cell per present stream.
  const std::uint64_t frame_cells =
      static_cast<std::uint64_t>(pww) * pwh * 3u;
  REQUIRE(sb.quality_bytes_uploaded ==
          f.plan.frames.size() * 4 * frame_cells);
  REQUIRE(sa.quality_bytes_uploaded ==
          f.plan.frames.size() * 4 * 20 * sizeof(float));
}

TEST_CASE("forward drizzle v2 driver model+packed-Q accounting",
          "[forward-drizzle-v2][gate10][driver]") {
  V2StoreFixture fx;
  auto f = make_fixture(ColorMode::MONO, BayerPattern::GRBG, 0, 0, 1);
  const int nc = f.plan.canvas_width_native,
            nr = f.plan.canvas_height_native;
  const int sw = f.plan.source_width, sh = f.plan.source_height;
  const auto plan = g10_driver_plan(f, 4, 1, true);
  const int d = 2;
  const int gw = (sw + d - 1) / d, gh = (sh + d - 1) / d;
  const std::size_t gcells = static_cast<std::size_t>(gw) * gh;
  std::vector<std::uint16_t> cells(gcells);
  std::vector<std::uint8_t> veto(gcells, 0);
  for (std::size_t i = 0; i < gcells; ++i)
    cells[i] = static_cast<std::uint16_t>(1 + (i * 613) % 65534);
  const double noise = 0.25, reg = 0.04,
               half = 0.5 * static_cast<double>(f.cfg.pixfrac);
  Matrix2Df buf;
  std::uint64_t expect_halo_bytes = 0, expect_active = 0;
  ForwardDrizzleV2FrameProvider provider =
      [&](int band_y_begin, int band_rows, std::uint64_t order,
          const ForwardDrizzleV2FramePieceSink &sink) -> bool {
    if (order >= f.plan.frames.size()) return false;
    const auto &frame = f.plan.frames[order];
    const auto &m = frame.source_to_canvas;
    ForwardDrizzleV2FrameInput in;
    const double a6[6] = {m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1),
                          m(1, 2)};
    std::copy_n(a6, 6, in.affine6);
    in.has_local_model = false;
    in.quality = {};
    in.skip = false;
    in.sigma2 = nullptr;
    in.sigma2_model = {true, noise, reg, half};
    in.meta = {0.9f, 0.8f, 1u, 0};
    const auto box = drizzle_source_scan_box(
        f.plan, frame, plan.internal_scale,
        band_y_begin * plan.internal_scale,
        band_rows * plan.internal_scale);
    const int bw = box.x1 - box.x0, bh = box.y1 - box.y0;
    if (bw <= 0 || bh <= 0) {
      in.skip = true;
      return sink(in);
    }
    const int hx0 = std::max(0, box.x0 - 1), hy0 = std::max(0, box.y0 - 1);
    const int hx1 = std::min(sw, box.x1 + 1), hy1 = std::min(sh, box.y1 + 1);
    const int hw = hx1 - hx0, hh = hy1 - hy0;
    buf.resize(hh, hw);
    for (int r = 0; r < hh; ++r)
      for (int c = 0; c < hw; ++c)
        buf(r, c) = f.images[order](hy0 + r, hx0 + c);
    in.source = buf.data();
    in.source_window = {hx0, hy0, hw, hh,
                        box.x0 - hx0, box.y0 - hy0, bw, bh};
    expect_halo_bytes += static_cast<std::uint64_t>(hw) * hh * sizeof(float);
    expect_active += static_cast<std::uint64_t>(bw) * bh;
    // Full-grid packed Q for the composite stream only.
    ForwardDrizzleV2PackedQualityPlane p;
    p.cells = cells.data();
    p.veto = veto.data();
    p.storage_x_begin = 0;
    p.storage_y_begin = 0;
    p.storage_width = gw;
    p.storage_height = gh;
    p.storage_divisor = d;
    in.quality.qc_packed = p;
    return sink(in);
  };
  ForwardDrizzleV2DriverOptions opts;
  opts.prefer_cuda = false;
  const auto done = run_forward_drizzle_v2(fx.root, plan, sw, sh, provider,
                                           opts);
  REQUIRE(done.committed);
  // No explicit sigma2 plane was ever supplied in model mode.
  REQUIRE(done.totals.source_bytes_uploaded == expect_halo_bytes);
  // Launched samples count ACTIVE pixels, never the halo ring.
  REQUIRE(done.totals.source_samples_launched == expect_active);
  // Compact Q bytes: 3 bytes per storage cell per present stream. Only the
  // composite stream was present; non-selected frames upload nothing.
  const auto selected = forward_drizzle_v2_selected_frame_orders(
      plan.frame_count, plan.reservoir_size, plan.reservoir_seed);
  // Quality work happens for selected frames only, and only where the
  // frame's band window is non-empty (empty windows take skip_frame).
  std::uint64_t expect_qframes = 0;
  for (int band = 0; band < plan.band_count; ++band) {
    const int by = band * plan.band_rows;
    const int brows = std::min(plan.band_rows, nr - by);
    for (std::uint64_t order : selected) {
      const auto box = drizzle_source_scan_box(
          f.plan, f.plan.frames[order], plan.internal_scale,
          by * plan.internal_scale, brows * plan.internal_scale);
      if (box.x1 > box.x0 && box.y1 > box.y0) ++expect_qframes;
    }
  }
  REQUIRE(done.totals.quality_frames_processed == expect_qframes);
  REQUIRE(done.totals.quality_bytes_uploaded ==
          done.totals.quality_frames_processed * gcells * 3u);
}

TEST_CASE("forward drizzle v2 CUDA windowed scatter matches full-source",
          "[forward-drizzle-v2][cuda-parity][gate10]") {
  if (!forward_drizzle_cuda_runtime_available()) {
    SUCCEED("CUDA device unavailable");
    return;
  }
  const BayerPattern patterns[] = {BayerPattern::RGGB, BayerPattern::BGGR,
                                   BayerPattern::GRBG, BayerPattern::GBRG};
  const int scale = 2;
  for (BayerPattern pattern : patterns) {
    auto f = make_fixture(ColorMode::OSC, pattern, 0, 0, scale);
    const int sw = f.plan.source_width, sh = f.plan.source_height;
    const int nc = f.plan.canvas_width_native,
              nr = f.plan.canvas_height_native;
    const std::size_t n_src = static_cast<std::size_t>(sw) * sh;
    const std::size_t nplane = static_cast<std::size_t>(nc) * nr * 3;
    std::vector<float> s2(n_src, 0.02f);
    auto make_kcfg = [&](int y_begin) {
      ForwardDrizzleV2KernelConfig k;
      k.internal_scale = scale;
      k.stream_length = f.plan.frames.size();
      k.sigma2_plane = true;
      k.half = 0.5 * f.cfg.pixfrac;
      k.bayer_pattern = static_cast<int>(pattern);
      k.cfa_origin_x = f.plan.cfa_origin_x;
      k.cfa_origin_y = f.plan.cfa_origin_y;
      k.mono = false;
      k.canvas_width_native = nc;
      k.canvas_height_native = nr;
      k.band_origin_y_native = y_begin;
      return k;
    };
    auto run = [&](int y_begin, int rows, bool windowed) {
      ForwardDrizzleV2CudaPrototypeKernel kernel;
      REQUIRE(kernel.reserve(nc, rows, sw, sh, make_kcfg(y_begin)));
      for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
        const auto &m = f.plan.frames[fr].source_to_canvas;
        const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                              m(1, 0), m(1, 1), m(1, 2)};
        if (!windowed) {
          REQUIRE(kernel.accumulate_frame(a6, f.images[fr].data(),
                                          s2.data(), fr));
          continue;
        }
        const auto box = drizzle_source_scan_box(
            f.plan, f.plan.frames[fr], scale, y_begin * scale,
            rows * scale);
        const int bw = box.x1 - box.x0, bh = box.y1 - box.y0;
        if (bw <= 0 || bh <= 0) {
          REQUIRE(kernel.skip_frame(fr));
          continue;
        }
        const ForwardDrizzleV2SourceWindow w{box.x0, box.y0, bw, bh};
        std::vector<float> src_w(static_cast<std::size_t>(bw) * bh),
            s2w(src_w.size());
        for (int r = 0; r < bh; ++r)
          for (int c = 0; c < bw; ++c) {
            src_w[static_cast<std::size_t>(r) * bw + c] =
                f.images[fr](box.y0 + r, box.x0 + c);
            s2w[static_cast<std::size_t>(r) * bw + c] =
                s2[static_cast<std::size_t>(box.y0 + r) * sw + box.x0 + c];
          }
        REQUIRE(kernel.accumulate_frame_window(a6, w, src_w.data(),
                                               s2w.data(), nullptr, fr));
      }
      std::vector<ForwardDrizzleV2PixelResult> out(
          static_cast<std::size_t>(rows) * nc * 3);
      std::uint64_t dense = 0;
      REQUIRE(kernel.finalize(out.data(), nullptr, &dense));
      return std::pair{out, kernel.stats()};
    };
    const int y0 = 4, rows = 5;
    const auto full = run(y0, rows, false);
    const auto win = run(y0, rows, true);
    REQUIRE(full.first.size() == win.first.size());
    for (std::size_t i = 0; i < full.first.size(); ++i) {
      const auto &a = full.first[i];
      const auto &b = win.first[i];
      // Discrete fields and fractions are device-exact; magnitudes use the
      // established parity tolerance.
      REQUIRE(a.geometry_fraction == b.geometry_fraction);
      REQUIRE(a.source_fraction == b.source_fraction);
      REQUIRE(a.estimator_fraction == b.estimator_fraction);
      REQUIRE(a.profile_fraction == b.profile_fraction);
      REQUIRE(a.contributors == b.contributors);
      REQUIRE(a.robust_state == b.robust_state);
      REQUIRE(a.confidence_state == b.confidence_state);
      REQUIRE(a.conf_degraded == b.conf_degraded);
      REQUIRE(a.value == Catch::Approx(b.value).epsilon(1e-9).margin(1e-9));
      REQUIRE(a.b == Catch::Approx(b.b).epsilon(1e-9).margin(1e-9));
      REQUIRE(a.n_eff == Catch::Approx(b.n_eff).epsilon(1e-9).margin(1e-9));
      REQUIRE(a.confidence ==
              Catch::Approx(b.confidence).epsilon(1e-9).margin(1e-9));
    }
    // Windowed launch: samples bounded by the exact scan boxes, never the
    // full source per frame.
    std::uint64_t expect_act = 0;
    for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
      const auto box = drizzle_source_scan_box(
          f.plan, f.plan.frames[fr], scale, y0 * scale, rows * scale);
      const int bw = box.x1 - box.x0, bh = box.y1 - box.y0;
      if (bw > 0 && bh > 0)
        expect_act += static_cast<std::uint64_t>(bw) * bh;
    }
    REQUIRE(win.second.source_samples_launched == expect_act);
  }
}

TEST_CASE("forward drizzle v2 begin_band reuses one workspace across bands",
          "[forward-drizzle-v2][gate10][kernel][persistent-workspace]") {
  auto f = make_fixture(ColorMode::MONO, BayerPattern::GRBG, 0, 0, 1);
  const int sw = f.plan.source_width, sh = f.plan.source_height;
  const int nc = f.plan.canvas_width_native, nr = f.plan.canvas_height_native;
  const std::size_t n_src = static_cast<std::size_t>(sw) * sh;
  std::vector<float> s2(n_src, 0.02f);
  // One local-model frame exercises the local scatter under reset.
  float lx[16] = {}, ly[16] = {};
  auto lf = v2_local_frame(f.plan.frames[0], v2_local_model(sh, sw, lx, ly));
  const auto lw = v2_warp_descriptor(lf);
  std::vector<ForwardDrizzleV2FrameMeta> meta(f.plan.frames.size());
  for (std::size_t i = 0; i < meta.size(); ++i)
    meta[i] = {0.9f, 0.8f, static_cast<std::uint8_t>(i % 2), 0};
  auto make_kcfg = [&](int y_begin) {
    ForwardDrizzleV2KernelConfig k;
    k.internal_scale = 1;
    k.stream_length = f.plan.frames.size();
    k.sigma2_plane = true;
    k.emit_profiles = true;
    k.half = 0.5 * f.cfg.pixfrac;
    k.bayer_pattern = static_cast<int>(BayerPattern::GRBG);
    k.canvas_width_native = nc;
    k.canvas_height_native = nr;
    k.band_origin_y_native = y_begin;
    k.mono = true;
    return k;
  };
  auto run_band = [&](ForwardDrizzleV2Kernel &kernel, int y_begin,
                      int rows) {
    for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
      const auto &m = f.plan.frames[fr].source_to_canvas;
      const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                            m(1, 0), m(1, 1), m(1, 2)};
      if (fr == 1 && y_begin >= 4) {
        REQUIRE(kernel.skip_frame(fr, &meta[fr]));  // empty-window frame
        continue;
      }
      if (fr == 0)
        REQUIRE(kernel.accumulate_frame_local(a6, lw, f.images[fr].data(),
                                              s2.data(), fr, nullptr,
                                              &meta[fr]));
      else
        REQUIRE(kernel.accumulate_frame(a6, f.images[fr].data(), s2.data(),
                                        fr, nullptr, &meta[fr]));
    }
    std::vector<ForwardDrizzleV2PixelResult> recs(
        static_cast<std::size_t>(rows) * nc);
    std::vector<ForwardDrizzleV2ProfileResult> profs(recs.size());
    std::uint64_t dense = 0;
    REQUIRE(kernel.finalize(recs.data(), profs.data(), &dense));
    return std::tuple{recs, profs, dense};
  };

  // Reference: one fresh kernel per band (the pre-tranche-5 shape).
  std::vector<std::tuple<std::vector<ForwardDrizzleV2PixelResult>,
                         std::vector<ForwardDrizzleV2ProfileResult>,
                         std::uint64_t>>
      ref;
  for (int y0 = 0; y0 < nr; y0 += 5) {
    const int rows = std::min(5, nr - y0);
    ForwardDrizzleV2CpuKernel kernel;
    REQUIRE(kernel.reserve(nc, rows, sw, sh, make_kcfg(y0)));
    ref.push_back(run_band(kernel, y0, rows));
  }

  // Persistent: one reserve for the max band height, begin_band per band.
  ForwardDrizzleV2CpuKernel kernel;
  REQUIRE(kernel.reserve(nc, 5, sw, sh, make_kcfg(0)));
  REQUIRE(kernel.stats().workspace_reservations == 1);
  REQUIRE(kernel.stats().allocations == 1);
  std::size_t band_i = 0;
  for (int y0 = 0; y0 < nr; y0 += 5, ++band_i) {
    const int rows = std::min(5, nr - y0);
    REQUIRE(kernel.begin_band(rows, make_kcfg(y0)));
    REQUIRE(kernel.stats().band_resets == 1);
    REQUIRE(kernel.stats().workspace_reservations == (band_i == 0 ? 1 : 0));
    auto got = run_band(kernel, y0, rows);
    const auto &[rr, rp, rd] = ref[band_i];
    const auto &[gr, gp, gd] = got;
    REQUIRE(rd == gd);
    REQUIRE(gr.size() == rr.size());
    for (std::size_t i = 0; i < rr.size(); ++i) {
      REQUIRE(g10_bit_eq(rr[i], gr[i]));
      REQUIRE(g10_bit_eq(rp[i], gp[i]));
    }
  }

  // Contract violations.
  {
    ForwardDrizzleV2CpuKernel fresh;
    REQUIRE(fresh.reserve(nc, 5, sw, sh, make_kcfg(0)));
    REQUIRE_FALSE(fresh.begin_band(6, make_kcfg(0)));        // oversized rows
    REQUIRE_FALSE(fresh.begin_band(0, make_kcfg(0)));        // zero rows
    auto bad = make_kcfg(0);
    bad.band_origin_y_native = nr;                           // origin outside
    REQUIRE_FALSE(fresh.begin_band(1, bad));
    bad = make_kcfg(nr - 2);
    bad.canvas_height_native = nr;  // origin + rows must fit the canvas
    REQUIRE_FALSE(fresh.begin_band(5, bad));
    bad = make_kcfg(0);
    bad.reservoir_seed += 1;                                 // fixed field
    REQUIRE_FALSE(fresh.begin_band(5, bad));
    bad = make_kcfg(0);
    bad.internal_scale = 2;
    REQUIRE_FALSE(fresh.begin_band(5, bad));
    bad = make_kcfg(0);
    bad.band_origin_x_native = 1;                            // x is fixed
    REQUIRE_FALSE(fresh.begin_band(5, bad));
    // Mid-stream: begin_band after a frame but before finalize fails.
    REQUIRE(fresh.begin_band(5, make_kcfg(0)));
    const auto &m = f.plan.frames[0].source_to_canvas;
    const double a6[6] = {m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1),
                          m(1, 2)};
    REQUIRE(fresh.accumulate_frame(a6, f.images[0].data(), s2.data(), 0,
                                   nullptr, &meta[0]));
    REQUIRE_FALSE(fresh.begin_band(5, make_kcfg(0)));
    // ... but after finalize it rebinds fine.
    std::vector<ForwardDrizzleV2PixelResult> recs(5ull * nc);
    std::vector<ForwardDrizzleV2ProfileResult> profs(recs.size());
    for (std::size_t fr = 1; fr < f.plan.frames.size(); ++fr) {
      const auto &m2 = f.plan.frames[fr].source_to_canvas;
      const double b6[6] = {m2(0, 0), m2(0, 1), m2(0, 2),
                            m2(1, 0), m2(1, 1), m2(1, 2)};
      REQUIRE(fresh.accumulate_frame(b6, f.images[fr].data(), s2.data(), fr,
                                     nullptr, &meta[fr]));
    }
    std::uint64_t dense = 0;
    REQUIRE(fresh.finalize(recs.data(), profs.data(), &dense));
    REQUIRE(fresh.begin_band(3, make_kcfg(2)));
    REQUIRE(fresh.stats().band_resets == 1);
    REQUIRE(fresh.stats().workspace_reservations == 0);  // consumed earlier
  }
}

TEST_CASE("forward drizzle v2 driver keeps one workspace per attempt",
          "[forward-drizzle-v2][gate10][driver][persistent-workspace]") {
  V2StoreFixture fx;
  auto f = make_fixture(ColorMode::MONO, BayerPattern::GRBG, 0, 0, 1);
  const int sw = f.plan.source_width, sh = f.plan.source_height;
  const auto plan = g10_driver_plan(f, 4, 1, true);
  ForwardDrizzleV2FrameProvider provider =
      [&](int, int, std::uint64_t order,
          const ForwardDrizzleV2FramePieceSink &sink) -> bool {
    if (order >= f.plan.frames.size()) return false;
    const auto &m = f.plan.frames[order].source_to_canvas;
    ForwardDrizzleV2FrameInput in;
    const double a6[6] = {m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1),
                          m(1, 2)};
    std::copy_n(a6, 6, in.affine6);
    in.has_local_model = false;
    in.quality = {};
    in.skip = false;
    in.source = f.images[order].data();
    in.sigma2 = nullptr;
    in.sigma2_model = {};
    in.source_window = {0, 0, sw, sh};
    in.meta = {0.9f, 0.8f, 1u, 0};
    return sink(in);
  };
  ForwardDrizzleV2DriverOptions opts;
  opts.prefer_cuda = false;
  const auto done = run_forward_drizzle_v2(fx.root, plan, sw, sh, provider,
                                           opts);
  REQUIRE(done.committed);
  // Exactly one workspace reservation and one logical allocation for the
  // whole backend attempt; one begin_band per computed band.
  REQUIRE(done.totals.workspace_reservations == 1);
  REQUIRE(done.totals.allocations == 1);
  REQUIRE(done.totals.band_resets ==
          static_cast<std::uint64_t>(plan.band_count));
  REQUIRE(done.driver_hotpath_allocations == 0);
  REQUIRE(done.totals.reserved_device_bytes > 0);
}

TEST_CASE("forward drizzle v2 CUDA persistent workspace matches "
          "independent kernels",
          "[forward-drizzle-v2][cuda-parity][gate10][persistent-workspace]") {
  if (!forward_drizzle_cuda_runtime_available()) {
    SUCCEED("CUDA device unavailable");
    return;
  }
  auto f = make_fixture(ColorMode::OSC, BayerPattern::RGGB, 0, 0, 1);
  const int sw = f.plan.source_width, sh = f.plan.source_height;
  const int nc = f.plan.canvas_width_native, nr = f.plan.canvas_height_native;
  const std::size_t n_src = static_cast<std::size_t>(sw) * sh;
  std::vector<float> s2(n_src, 0.02f);
  std::vector<ForwardDrizzleV2FrameMeta> meta(f.plan.frames.size());
  for (std::size_t i = 0; i < meta.size(); ++i)
    meta[i] = {0.9f, 0.8f, static_cast<std::uint8_t>(i % 2), 0};
  auto make_kcfg = [&](int y_begin) {
    ForwardDrizzleV2KernelConfig k;
    k.internal_scale = 1;
    k.stream_length = f.plan.frames.size();
    k.sigma2_plane = true;
    k.emit_profiles = true;
    k.half = 0.5 * f.cfg.pixfrac;
    k.bayer_pattern = static_cast<int>(BayerPattern::RGGB);
    k.canvas_width_native = nc;
    k.canvas_height_native = nr;
    k.band_origin_y_native = y_begin;
    k.mono = false;
    return k;
  };
  auto run_band = [&](ForwardDrizzleV2CudaPrototypeKernel &kernel, int rows) {
    for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
      const auto &m = f.plan.frames[fr].source_to_canvas;
      const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                            m(1, 0), m(1, 1), m(1, 2)};
      if (fr == 2) {  // skip-event case: no events recorded for this order
        REQUIRE(kernel.skip_frame(fr, &meta[fr]));
        continue;
      }
      REQUIRE(kernel.accumulate_frame(a6, f.images[fr].data(), s2.data(), fr,
                                      nullptr, &meta[fr]));
    }
    std::vector<ForwardDrizzleV2PixelResult> recs(
        static_cast<std::size_t>(rows) * nc * 3);
    std::vector<ForwardDrizzleV2ProfileResult> profs(recs.size());
    std::uint64_t dense = 0;
    REQUIRE(kernel.finalize(recs.data(), profs.data(), &dense));
    return std::tuple{recs, profs, dense};
  };
  // Short last band (nr % 4 != 0 for the fixture canvas).
  const std::vector<std::pair<int, int>> bands = {{0, 4}, {4, 4}, {8, 5}};
  using BandOut = std::tuple<std::vector<ForwardDrizzleV2PixelResult>,
                             std::vector<ForwardDrizzleV2ProfileResult>,
                             std::uint64_t>;
  std::vector<BandOut> ref;
  for (const auto &[y0, rows] : bands) {
    ForwardDrizzleV2CudaPrototypeKernel kernel;
    REQUIRE(kernel.reserve(nc, rows, sw, sh, make_kcfg(y0)));
    ref.push_back(run_band(kernel, rows));
  }
  ForwardDrizzleV2CudaPrototypeKernel kernel;
  REQUIRE(kernel.reserve(nc, 5, sw, sh, make_kcfg(0)));
  for (std::size_t b = 0; b < bands.size(); ++b) {
    const auto &[y0, rows] = bands[b];
    REQUIRE(kernel.begin_band(rows, make_kcfg(y0)));
    REQUIRE(kernel.stats().band_resets == 1);
    auto got = run_band(kernel, rows);
    const auto &[rr, rp, rd] = ref[b];
    const auto &[gr, gp, gd] = got;
    REQUIRE(rd == gd);
    for (std::size_t i = 0; i < rr.size(); ++i) {
      // Discrete fields are device-exact; magnitudes use the established
      // parity tolerance.
      REQUIRE(gr[i].contributors == rr[i].contributors);
      REQUIRE(gr[i].robust_state == rr[i].robust_state);
      REQUIRE(gr[i].confidence_state == rr[i].confidence_state);
      REQUIRE(gr[i].conf_degraded == rr[i].conf_degraded);
      REQUIRE(gr[i].geometry_fraction == rr[i].geometry_fraction);
      REQUIRE(gr[i].value ==
              Catch::Approx(rr[i].value).epsilon(1e-9).margin(1e-9));
      REQUIRE(gp[i].uniform.support == rp[i].uniform.support);
      REQUIRE(std::isnan(gp[i].uniform.value) ==
              std::isnan(rp[i].uniform.value));
      if (std::isfinite(rp[i].uniform.value))
        REQUIRE(gp[i].uniform.value ==
                Catch::Approx(rp[i].uniform.value).epsilon(1e-9).margin(1e-9));
    }
  }
  // Skipped order: event flags gate finalize timing; stats are per-band.
  REQUIRE(kernel.stats().workspace_reservations == 0);
}

// --- Tranche 6: committed geometry-cache leaves drive local frames ---------

namespace {

// Provider that serves local frames from a DrizzleGeometryCacheReader
// exactly like the production provider: stripe leaf window, skip on empty,
// full-source upload window for simplicity (the kernel validates coverage
// against the ACTIVE rect, which is the whole source here).
ForwardDrizzleV2FrameProvider g10_cached_provider(
    const Fixture &f, const DrizzleGeometryCacheReader &reader, float pixfrac,
    int scale, const std::vector<float> &s2,
    const std::vector<ForwardDrizzleV2FrameMeta> *meta,
    DrizzleCachedLeafWindow &win,
    std::vector<ForwardDrizzleV2CachedLeaf> &conv,
    std::uint64_t *leaves_read = nullptr) {
  return [&](int band_y0, int band_rows, std::uint64_t order,
             const ForwardDrizzleV2FramePieceSink &sink) {
    if (order >= f.plan.frames.size()) return false;
    const auto &frame = f.plan.frames[static_cast<std::size_t>(order)];
    const auto &m = frame.source_to_canvas;
    ForwardDrizzleV2FrameInput in;
    const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                          m(1, 0), m(1, 1), m(1, 2)};
    std::copy_n(a6, 6, in.affine6);
    in.has_local_model = false;
    in.warp = ForwardDrizzleV2LocalWarp{};
    in.has_cached_geometry = false;
    in.cached_leaves = nullptr;
    in.cached_leaf_count = 0;
    in.cached_unique_source_samples = 0;
    in.quality = {};
    in.sigma2_model = {};
    in.skip = false;
    in.source = f.images[static_cast<std::size_t>(order)].data();
    in.sigma2 = s2.data();
    in.source_window = {0, 0, f.plan.source_width, f.plan.source_height};
    if (meta) in.meta = (*meta)[static_cast<std::size_t>(order)];
    if (!frame.has_smooth_local_model) return sink(in);
    reader.read_stripe_leaves_into(
        pixfrac, frame.source_index, scale, band_y0 * scale,
        band_rows * scale, win);
    if (leaves_read) *leaves_read += win.leaves.size();
    if (win.leaves.empty()) {
      in.skip = true;
      return sink(in);
    }
    conv.resize(win.leaves.size());
    for (std::size_t li = 0; li < win.leaves.size(); ++li) {
      const DrizzleCachedLeaf &s = win.leaves[li];
      ForwardDrizzleV2CachedLeaf &d = conv[li];
      d.source_x = s.source_x;
      d.source_y = s.source_y;
      d.channel = s.channel;
      d.leaf_order = s.leaf_order;
      for (int k = 0; k < 4; ++k) {
        d.x[k] = s.x[k];
        d.y[k] = s.y[k];
      }
    }
    in.has_cached_geometry = true;
    in.cached_leaves = conv.data();
    in.cached_leaf_count = conv.size();
    in.cached_unique_source_samples = win.unique_source_samples;
    return sink(in);
  };
}

}  // namespace

TEST_CASE("forward drizzle v2 driver replays committed cache leaves "
          "bit-identically to the local-window path (tranche 6)",
          "[forward-drizzle-v2][gate10][geometry-cache]") {
  for (int scale : {1, 2}) {
    auto f = make_fixture(ColorMode::OSC, BayerPattern::GBRG, 0, 0, scale);
    const int nc = f.plan.canvas_width_native;
    const int nr = f.plan.canvas_height_native;
    const std::size_t n_src =
        static_cast<std::size_t>(f.plan.source_width) * f.plan.source_height;
    std::vector<float> s2(n_src, 0.02f);
    std::vector<ForwardDrizzleV2FrameMeta> meta(f.plan.frames.size());
    for (std::size_t i = 0; i < meta.size(); ++i)
      meta[i] = {0.9f, 0.8f, static_cast<std::uint8_t>(i % 2), 0};
    float cx1[16] = {}, cy1[16] = {}, cx3[16] = {}, cy3[16] = {};
    cx1[5] = 1.3f;
    cy1[6] = -0.8f;
    cx3[9] = -1.7f;
    cy3[10] = 1.1f;
    f.plan.frames[1] = v2_local_frame(
        f.plan.frames[1], v2_local_model(nr, nc, cx1, cy1));
    f.plan.frames[3] = v2_local_frame(
        f.plan.frames[3], v2_local_model(nr, nc, cx3, cy3));

    // Commit the geometry cache for the two local frames.
    config::ReconstructionDrizzleConfig dcfg;
    dcfg.internal_scale = scale;
    dcfg.pixfrac = f.cfg.pixfrac;
    V2StoreFixture geom_fx;
    const auto built = build_drizzle_geometry_cache(
        geom_fx.root, f.plan, dcfg, {{f.cfg.pixfrac}}, {}, 64ull << 20);
    DrizzleGeometryCacheReader greader(geom_fx.root, built.identities,
                                       {1, 3});

    const auto plan = g10_driver_plan(f, 4, 3, true);
    const std::uint64_t cap = greader.max_stripe_leaf_records(
        f.cfg.pixfrac, plan.band_rows * scale);
    REQUIRE(cap > 0);

    V2StoreFixture ref_fx;
    ForwardDrizzleV2DriverOptions cpu_opts;
    cpu_opts.prefer_cuda = false;
    const auto ref = run_forward_drizzle_v2(
        ref_fx.root, plan, f.plan.source_width, f.plan.source_height,
        g10_provider(f, s2, &meta), cpu_opts);
    REQUIRE(ref.committed);
    std::vector<ForwardDrizzleV2PixelResult> ref_records;
    std::vector<ForwardDrizzleV2ProfileResult> ref_profiles;
    g10_read_store(ref_fx.root, plan, ref_records, &ref_profiles);

    V2StoreFixture cx;
    DrizzleCachedLeafWindow win;
    win.leaves.reserve(static_cast<std::size_t>(cap));
    std::vector<ForwardDrizzleV2CachedLeaf> conv;
    conv.reserve(static_cast<std::size_t>(cap));
    std::uint64_t leaves_read = 0;
    ForwardDrizzleV2DriverOptions copts;
    copts.prefer_cuda = false;
    copts.cached_leaf_capacity = cap;
    const auto got = run_forward_drizzle_v2(
        cx.root, plan, f.plan.source_width, f.plan.source_height,
        g10_cached_provider(f, greader, f.cfg.pixfrac, scale, s2, &meta,
                            win, conv, &leaves_read),
        copts);
    REQUIRE(got.committed);
    std::vector<ForwardDrizzleV2PixelResult> got_records;
    std::vector<ForwardDrizzleV2ProfileResult> got_profiles;
    g10_read_store(cx.root, plan, got_records, &got_profiles);

    // Records and profiles are bit-identical: the cached scatter emits the
    // same corner bits in the same canonical order as the direct local
    // path (the internal band origin shift is exact for power-of-two
    // scales).
    REQUIRE(got_records.size() == ref_records.size());
    for (std::size_t i = 0; i < ref_records.size(); ++i)
      REQUIRE(g10_bit_eq(got_records[i], ref_records[i]));
    REQUIRE(got_profiles.size() == ref_profiles.size());
    for (std::size_t i = 0; i < ref_profiles.size(); ++i)
      REQUIRE(g10_bit_eq(got_profiles[i], ref_profiles[i]));

    // No inversion/subdivision discard work ran in the consume phase; the
    // cache finalised it at build time.
    REQUIRE(got.totals.cached_leaf_records_launched == leaves_read);
    REQUIRE(got.totals.cached_leaf_records_launched > 0);
    REQUIRE(got.local_samples_discarded == 0);
    REQUIRE(got.totals.source_samples_launched <=
            ref.totals.source_samples_launched);
  }
}

TEST_CASE("forward drizzle v2 cached-leaf path fails closed on malformed "
          "records (tranche 6)",
          "[forward-drizzle-v2][gate10][geometry-cache]") {
  auto f = make_fixture(ColorMode::MONO, BayerPattern::RGGB, 0, 0, 1);
  const int nc = f.plan.canvas_width_native;
  const int nr = f.plan.canvas_height_native;
  const int sw = f.plan.source_width, sh = f.plan.source_height;
  auto kcfg = [&] {
    ForwardDrizzleV2KernelConfig k;
    k.internal_scale = 1;
    k.stream_length = 1;
    k.half = 0.5 * f.cfg.pixfrac;
    k.bayer_pattern = static_cast<int>(f.plan.bayer_pattern);
    k.mono = true;
    k.canvas_width_native = nc;
    k.canvas_height_native = nr;
    k.cached_leaf_capacity = 8;
    return k;
  };
  const ForwardDrizzleV2SourceWindow full{0, 0, sw, sh};
  const auto &m = f.plan.frames[0].source_to_canvas;
  const double a6[6] = {m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1), m(1, 2)};

  // One valid leaf (absolute internal coords; scale 1 here).
  ForwardDrizzleV2CachedLeaf good;
  good.source_x = 1;
  good.source_y = 1;
  good.channel = 0;
  good.leaf_order = 0;
  for (int k = 0; k < 4; ++k) {
    good.x[k] = 2.0 + (k == 1 || k == 2 ? 0.5 : -0.5);
    good.y[k] = 2.0 + (k >= 2 ? 0.5 : -0.5);
  }

  SECTION("accepts a well-formed leaf") {
    ForwardDrizzleV2CpuKernel kernel;
    REQUIRE(kernel.reserve(nc, nr, sw, sh, kcfg()));
    REQUIRE(kernel.accumulate_frame_cached_leaves(
        a6, full, f.images[0].data(), nullptr, nullptr, &good, 1, 1, 0));
    REQUIRE(kernel.stats().cached_leaf_records_launched == 1);
    REQUIRE(kernel.stats().source_samples_launched == 1);
  }
  SECTION("leaf_count over the reserved capacity fails") {
    ForwardDrizzleV2CpuKernel kernel;
    REQUIRE(kernel.reserve(nc, nr, sw, sh, kcfg()));
    std::vector<ForwardDrizzleV2CachedLeaf> many(9, good);
    REQUIRE_FALSE(kernel.accumulate_frame_cached_leaves(
        a6, full, f.images[0].data(), nullptr, nullptr, many.data(), 9, 9,
        0));
  }
  SECTION("zero capacity disables the path") {
    auto k = kcfg();
    k.cached_leaf_capacity = 0;
    ForwardDrizzleV2CpuKernel kernel;
    REQUIRE(kernel.reserve(nc, nr, sw, sh, k));
    REQUIRE_FALSE(kernel.accumulate_frame_cached_leaves(
        a6, full, f.images[0].data(), nullptr, nullptr, &good, 1, 1, 0));
  }
  SECTION("source coordinate outside the active window fails") {
    ForwardDrizzleV2CpuKernel kernel;
    REQUIRE(kernel.reserve(nc, nr, sw, sh, kcfg()));
    ForwardDrizzleV2CachedLeaf bad = good;
    bad.source_x = static_cast<std::uint32_t>(sw);  // outside the source
    REQUIRE_FALSE(kernel.accumulate_frame_cached_leaves(
        a6, full, f.images[0].data(), nullptr, nullptr, &bad, 1, 1, 0));
    bad = good;
    bad.source_y = 0;
    const ForwardDrizzleV2SourceWindow sub{2, 2, sw - 3, sh - 3};
    REQUIRE_FALSE(kernel.accumulate_frame_cached_leaves(
        a6, sub, f.images[0].data() + 2 * sw + 2, nullptr, nullptr, &bad, 1,
        1, 0));
  }
  SECTION("channel out of range fails") {
    ForwardDrizzleV2CpuKernel kernel;
    REQUIRE(kernel.reserve(nc, nr, sw, sh, kcfg()));
    ForwardDrizzleV2CachedLeaf bad = good;
    bad.channel = 1;  // MONO allows only channel 0
    REQUIRE_FALSE(kernel.accumulate_frame_cached_leaves(
        a6, full, f.images[0].data(), nullptr, nullptr, &bad, 1, 1, 0));
  }
  SECTION("nonfinite corners fail") {
    ForwardDrizzleV2CpuKernel kernel;
    REQUIRE(kernel.reserve(nc, nr, sw, sh, kcfg()));
    ForwardDrizzleV2CachedLeaf bad = good;
    bad.x[2] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_FALSE(kernel.accumulate_frame_cached_leaves(
        a6, full, f.images[0].data(), nullptr, nullptr, &bad, 1, 1, 0));
    bad = good;
    bad.y[0] = std::numeric_limits<double>::infinity();
    REQUIRE_FALSE(kernel.accumulate_frame_cached_leaves(
        a6, full, f.images[0].data(), nullptr, nullptr, &bad, 1, 1, 0));
  }
  SECTION("empty leaf list is rejected (the provider's skip contract)") {
    ForwardDrizzleV2CpuKernel kernel;
    REQUIRE(kernel.reserve(nc, nr, sw, sh, kcfg()));
    REQUIRE_FALSE(kernel.accumulate_frame_cached_leaves(
        a6, full, f.images[0].data(), nullptr, nullptr, &good, 0, 0, 0));
  }
}

TEST_CASE("forward drizzle v2 CUDA cached-leaf path matches the CPU replay "
          "across persistent bands (tranche 6)",
          "[forward-drizzle-v2][gate10][geometry-cache][cuda-parity]") {
  if (!forward_drizzle_cuda_runtime_available()) {
    SUCCEED("CUDA device unavailable");
    return;
  }
  const int scale = 2;
  auto f = make_fixture(ColorMode::OSC, BayerPattern::GBRG, 0, 0, scale);
  const int nc = f.plan.canvas_width_native;
  const int nr = f.plan.canvas_height_native;
  const std::size_t n_src =
      static_cast<std::size_t>(f.plan.source_width) * f.plan.source_height;
  std::vector<float> s2(n_src);
  for (std::size_t i = 0; i < n_src; ++i)
    s2[i] = 0.01f + 0.0001f * static_cast<float>(i % 97);
  float cx[16] = {}, cy[16] = {};
  cx[5] = 1.2f;
  cy[6] = -0.7f;
  f.plan.frames[2] = v2_local_frame(
      f.plan.frames[2], v2_local_model(nr, nc, cx, cy));

  config::ReconstructionDrizzleConfig dcfg;
  dcfg.internal_scale = scale;
  dcfg.pixfrac = f.cfg.pixfrac;
  V2StoreFixture geom_fx;
  const auto built = build_drizzle_geometry_cache(
      geom_fx.root, f.plan, dcfg, {{f.cfg.pixfrac}}, {}, 64ull << 20);
  DrizzleGeometryCacheReader greader(geom_fx.root, built.identities, {2});

  auto kcfg = [&](int y0) {
    ForwardDrizzleV2KernelConfig k;
    k.internal_scale = scale;
    k.stream_length = f.plan.frames.size();
    k.half = 0.5 * f.cfg.pixfrac;
    k.bayer_pattern = static_cast<int>(f.plan.bayer_pattern);
    k.cfa_origin_x = f.plan.cfa_origin_x;
    k.cfa_origin_y = f.plan.cfa_origin_y;
    k.canvas_width_native = nc;
    k.canvas_height_native = nr;
    k.band_origin_y_native = y0;
    return k;
  };
  const int band_rows = 5;
  const std::uint64_t cap = greader.max_stripe_leaf_records(
      f.cfg.pixfrac, band_rows * scale);
  REQUIRE(cap > 0);

  // One persistent CUDA workspace over three bands (short last band);
  // compare each band's records to the CPU cached path.
  ForwardDrizzleV2CudaPrototypeKernel kernel;
  auto rk = kcfg(0);
  rk.cached_leaf_capacity = cap;
  REQUIRE(kernel.reserve(nc, band_rows, f.plan.source_width,
                         f.plan.source_height, rk));
  DrizzleCachedLeafWindow win;
  win.leaves.reserve(static_cast<std::size_t>(cap));
  std::vector<ForwardDrizzleV2CachedLeaf> conv;
  conv.reserve(static_cast<std::size_t>(cap));
  std::uint64_t leaf_records_total = 0, leaf_bytes_total = 0;
  int resets = 0;

  for (int band = 0, y = 0; y < nr; ++band, y += band_rows) {
    const int rows = std::min(band_rows, nr - y);
    auto bk = kcfg(y);
    bk.cached_leaf_capacity = cap;
    REQUIRE(kernel.begin_band(rows, bk));

    ForwardDrizzleV2CpuKernel cpu;
    REQUIRE(cpu.reserve(nc, rows, f.plan.source_width, f.plan.source_height,
                        bk));
    for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
      const auto &frame = f.plan.frames[fr];
      const auto &m = frame.source_to_canvas;
      const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                            m(1, 0), m(1, 1), m(1, 2)};
      const ForwardDrizzleV2SourceWindow full{0, 0, f.plan.source_width,
                                              f.plan.source_height};
      if (!frame.has_smooth_local_model) {
        REQUIRE(kernel.accumulate_frame_window(
            a6, full, f.images[fr].data(), s2.data(), nullptr, fr));
        REQUIRE(cpu.accumulate_frame_window(
            a6, full, f.images[fr].data(), s2.data(), nullptr, fr));
        continue;
      }
      greader.read_stripe_leaves_into(f.cfg.pixfrac, frame.source_index,
                                      scale, y * scale, rows * scale, win);
      if (win.leaves.empty()) {
        REQUIRE(kernel.skip_frame(fr));
        REQUIRE(cpu.skip_frame(fr));
        continue;
      }
      conv.resize(win.leaves.size());
      std::memcpy(conv.data(), win.leaves.data(),
                  win.leaves.size() * sizeof(ForwardDrizzleV2CachedLeaf));
      REQUIRE(kernel.accumulate_frame_cached_leaves(
          a6, full, f.images[fr].data(), s2.data(), nullptr, conv.data(),
          conv.size(), win.unique_source_samples, fr));
      REQUIRE(cpu.accumulate_frame_cached_leaves(
          a6, full, f.images[fr].data(), s2.data(), nullptr, conv.data(),
          conv.size(), win.unique_source_samples, fr));
    }
    const std::size_t n =
        static_cast<std::size_t>(rows) * nc * 3;
    std::vector<ForwardDrizzleV2PixelResult> gpu(n), ref(n);
    std::uint64_t dense_g = 0, dense_r = 0;
    REQUIRE(kernel.finalize(gpu.data(), nullptr, &dense_g));
    REQUIRE(cpu.finalize(ref.data(), nullptr, &dense_r));
    REQUIRE(dense_g == dense_r);
    for (std::size_t i = 0; i < n; ++i) {
      REQUIRE(gpu[i].contributors == ref[i].contributors);
      REQUIRE(gpu[i].robust_state == ref[i].robust_state);
      REQUIRE(gpu[i].confidence_state == ref[i].confidence_state);
      REQUIRE(gpu[i].conf_degraded == ref[i].conf_degraded);
      REQUIRE(gpu[i].geometry_fraction == ref[i].geometry_fraction);
      REQUIRE(std::isnan(gpu[i].value) == std::isnan(ref[i].value));
      if (std::isfinite(ref[i].value))
        REQUIRE(gpu[i].value ==
                Catch::Approx(ref[i].value).epsilon(1e-9).margin(1e-9));
      REQUIRE(gpu[i].b ==
              Catch::Approx(ref[i].b).epsilon(1e-9).margin(1e-9));
      REQUIRE(gpu[i].n_eff ==
              Catch::Approx(ref[i].n_eff).epsilon(1e-9).margin(1e-9));
    }
    // stats() is a per-band delta; accumulate the persistent-buffer proof.
    leaf_records_total += kernel.stats().cached_leaf_records_launched;
    leaf_bytes_total += kernel.stats().cached_leaf_bytes_uploaded;
    resets += static_cast<int>(kernel.stats().band_resets);
  }
  // One workspace for the whole multi-band run; the leaf buffer was
  // allocated once at reserve() and re-used on every band.
  REQUIRE(resets == 3);
  REQUIRE(leaf_records_total > 0);
  REQUIRE(leaf_bytes_total ==
          leaf_records_total * sizeof(ForwardDrizzleV2CachedLeaf));
}

// ---- tranche 7: affine target-X pieces ------------------------------------

namespace {

struct G7BandOut {
  std::vector<ForwardDrizzleV2PixelResult> records;
  std::vector<ForwardDrizzleV2ProfileResult> profiles;
  ForwardDrizzleV2PrototypeStats stats;
  std::uint64_t dense = 0;
  std::uint64_t empty_tiles = 0;
  std::uint64_t expected_launched = 0;
};

// Runs one band of `f` on `kernel` (already reserved for the band): every
// frame is emitted either as a single full-width piece through the
// compatibility wrapper (tile_cols == 0) or as ordered per-tile pieces of
// `tile_cols` native columns through the begin/piece/finish lifecycle.
// Empty tiles are omitted; a frame whose band box is empty is skipped.
G7BandOut g7_run_band(ForwardDrizzleV2Kernel &kernel, const Fixture &f,
                      const std::vector<float> &s2,
                      const std::vector<ForwardDrizzleV2FrameMeta> &meta,
                      bool profiles, int y_begin, int rows, int tile_cols,
                      bool inline_sigma) {
  const int nc = f.plan.canvas_width_native;
  const int channels = f.plan.color_mode == ColorMode::MONO ? 1 : 3;
  const int scale = f.cfg.internal_scale;
  const int sw = f.plan.source_width, sh = f.plan.source_height;
  G7BandOut out;
  std::vector<float> win_buf, s2w, qw;
  for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
    const auto &frame = f.plan.frames[fr];
    const auto &m = frame.source_to_canvas;
    const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                          m(1, 0), m(1, 1), m(1, 2)};
    const auto band_box = drizzle_source_scan_box(
        f.plan, frame, scale, y_begin * scale, rows * scale);
    if (band_box.x1 <= band_box.x0 || band_box.y1 <= band_box.y0) {
      REQUIRE(kernel.skip_frame(fr, profiles ? &meta[fr] : nullptr));
      continue;
    }
    // Pack the source (+optional halo) and sigma2/quality planes for the
    // given scan box, then dispatch either the one-shot wrapper or one
    // affine piece.
    auto emit = [&](const DrizzleSourceScanBox &box, int tx, int tw) {
      const int bw = box.x1 - box.x0, bh = box.y1 - box.y0;
      REQUIRE(bw > 0);
      REQUIRE(bh > 0);
      out.expected_launched += static_cast<std::uint64_t>(bw) * bh;
      const int hx0 = inline_sigma ? std::max(0, box.x0 - 1) : box.x0;
      const int hy0 = inline_sigma ? std::max(0, box.y0 - 1) : box.y0;
      const int hx1 = inline_sigma ? std::min(sw, box.x1 + 1) : box.x1;
      const int hy1 = inline_sigma ? std::min(sh, box.y1 + 1) : box.y1;
      const int hw = hx1 - hx0, hh = hy1 - hy0;
      win_buf.assign(static_cast<std::size_t>(hw) * hh, 0.0f);
      for (int r = 0; r < hh; ++r)
        for (int c = 0; c < hw; ++c)
          win_buf[static_cast<std::size_t>(r) * hw + c] =
              f.images[fr](hy0 + r, hx0 + c);
      ForwardDrizzleV2SourceWindow w{hx0, hy0, hw, hh,
                                     box.x0 - hx0, box.y0 - hy0, bw, bh};
      ForwardDrizzleV2Sigma2FrameModel sm{};
      if (inline_sigma) {
        sm.enabled = true;
        sm.sigma_noise = 0.05;
        sm.sigma_reg_px = 0.02;
        sm.droplet_half = 0.5 * f.cfg.pixfrac;
      } else {
        s2w.assign(static_cast<std::size_t>(bw) * bh, 0.0f);
        for (int r = 0; r < bh; ++r)
          for (int c = 0; c < bw; ++c)
            s2w[static_cast<std::size_t>(r) * bw + c] =
                s2[static_cast<std::size_t>(box.y0 + r) * sw + box.x0 + c];
      }
      ForwardDrizzleV2FrameQuality q{};
      if (profiles) {
        qw.assign(static_cast<std::size_t>(bw) * bh, 0.0f);
        for (int r = 0; r < bh; ++r)
          for (int c = 0; c < bw; ++c)
            qw[static_cast<std::size_t>(r) * bw + c] =
                0.5f + 0.01f * static_cast<float>((box.x0 + c + box.y0 + r) %
                                                  37);
        q.q_composite = qw.data();
      }
      if (tile_cols == 0) {
        return kernel.accumulate_frame_window(
            a6, w, win_buf.data(), inline_sigma ? nullptr : s2w.data(),
            inline_sigma ? &sm : nullptr, fr, profiles ? &q : nullptr,
            profiles ? &meta[fr] : nullptr);
      }
      return kernel.accumulate_affine_piece(
          a6, tx, tw, w, win_buf.data(),
          inline_sigma ? nullptr : s2w.data(), inline_sigma ? &sm : nullptr,
          profiles ? &q : nullptr);
    };
    if (tile_cols == 0) {
      REQUIRE(emit(band_box, 0, nc));
      continue;
    }
    // Pre-scan the tiles: a margin-inflated band box can be non-empty while
    // every tile's inverse image clips out on different axes (some x-out,
    // some y-out). Production emits one skip piece in exactly this case.
    struct TilePiece {
      int tx, tw;
      DrizzleSourceScanBox box;
    };
    std::vector<TilePiece> tiles;
    for (int tx = 0; tx < nc; tx += tile_cols) {
      const int tw = std::min(tile_cols, nc - tx);
      const auto box = drizzle_source_scan_box(f.plan, frame, scale,
                                               y_begin * scale, rows * scale,
                                               tx * scale, tw * scale);
      if (box.x1 <= box.x0 || box.y1 <= box.y0) {
        ++out.empty_tiles;
        continue;
      }
      tiles.push_back({tx, tw, box});
    }
    if (tiles.empty()) {
      REQUIRE(kernel.skip_frame(fr, profiles ? &meta[fr] : nullptr));
      continue;
    }
    REQUIRE(kernel.begin_affine_frame(fr, profiles ? &meta[fr] : nullptr));
    for (const auto &t : tiles) REQUIRE(emit(t.box, t.tx, t.tw));
    REQUIRE(kernel.finish_affine_frame(fr));
  }
  const std::size_t np =
      static_cast<std::size_t>(rows) * nc * channels;
  out.records.assign(np, ForwardDrizzleV2PixelResult{});
  if (profiles) out.profiles.assign(np, ForwardDrizzleV2ProfileResult{});
  REQUIRE(kernel.finalize(out.records.data(),
                          profiles ? out.profiles.data() : nullptr,
                          &out.dense));
  out.stats = kernel.stats();
  return out;
}

ForwardDrizzleV2KernelConfig g7_kcfg(const Fixture &f, int scale,
                                     ColorMode mode, BayerPattern pattern,
                                     int y_begin, bool profiles) {
  ForwardDrizzleV2KernelConfig k;
  k.internal_scale = scale;
  k.stream_length = f.plan.frames.size();
  k.half = 0.5 * f.cfg.pixfrac;
  k.bayer_pattern = static_cast<int>(pattern);
  k.cfa_origin_x = f.plan.cfa_origin_x;
  k.cfa_origin_y = f.plan.cfa_origin_y;
  k.mono = mode == ColorMode::MONO;
  k.canvas_width_native = f.plan.canvas_width_native;
  k.canvas_height_native = f.plan.canvas_height_native;
  k.band_origin_y_native = y_begin;
  k.emit_profiles = profiles;
  return k;
}

}  // namespace

TEST_CASE("forward drizzle v2 CPU affine target-x pieces are bit-identical "
          "to the full-width piece (tranche 7)",
          "[forward-drizzle-v2][gate10][target-pieces]") {
  const BayerPattern patterns[] = {BayerPattern::RGGB, BayerPattern::BGGR,
                                   BayerPattern::GRBG, BayerPattern::GBRG};
  const int nr = 13;
  for (int scale : {1, 2})
    for (ColorMode mode : {ColorMode::MONO, ColorMode::OSC})
      for (BayerPattern pattern : patterns)
        for (bool profiles : {false, true})
          for (bool inline_sigma : {false, true}) {
            auto f = make_fixture(mode, pattern, 1, -1, scale);
            const int nc = f.plan.canvas_width_native;
            const int sw = f.plan.source_width, sh = f.plan.source_height;
            const std::size_t n_src = static_cast<std::size_t>(sw) * sh;
            std::vector<float> s2(n_src);
            for (std::size_t i = 0; i < n_src; ++i)
              s2[i] = 0.01f + 0.0001f * static_cast<float>(i % 97);
            std::vector<ForwardDrizzleV2FrameMeta> meta(
                f.plan.frames.size());
            for (std::size_t i = 0; i < meta.size(); ++i)
              meta[i] = {0.9f, 0.8f, static_cast<std::uint8_t>(i % 2), 0};
            // First, middle and short last bands.
            for (const auto band : {std::pair{0, 5}, std::pair{4, 5},
                                    std::pair{9, 4}}) {
              const int y0 = band.first, rows = band.second;
              ForwardDrizzleV2CpuKernel ref;
              REQUIRE(ref.reserve(nc, rows, sw, sh,
                                  g7_kcfg(f, scale, mode, pattern, y0,
                                          profiles)));
              const auto full = g7_run_band(ref, f, s2, meta, profiles, y0,
                                            rows, 0, inline_sigma);
              for (int tile : {2, 3}) {
                ForwardDrizzleV2CpuKernel tiled;
                REQUIRE(tiled.reserve(
                    nc, rows, sw, sh,
                    g7_kcfg(f, scale, mode, pattern, y0, profiles)));
                const auto got = g7_run_band(tiled, f, s2, meta, profiles,
                                             y0, rows, tile, inline_sigma);
                for (std::size_t i = 0; i < full.records.size(); ++i)
                  REQUIRE(g10_bit_eq(full.records[i], got.records[i]));
                for (std::size_t i = 0; i < full.profiles.size(); ++i)
                  REQUIRE(g10_bit_eq(full.profiles[i], got.profiles[i]));
                REQUIRE(got.dense == full.dense);
                REQUIRE(got.stats.frames_processed ==
                        full.stats.frames_processed);
                REQUIRE(got.stats.slot_transitions ==
                        full.stats.slot_transitions);
                REQUIRE(got.stats.affine_pieces_processed >=
                        got.stats.frames_processed);
                // Launched samples are the exact per-piece window sums
                // (overlapping tile windows may exceed one band box).
                REQUIRE(got.stats.source_samples_launched ==
                        got.expected_launched);
                REQUIRE(full.stats.source_samples_launched ==
                        full.expected_launched);
              }
            }
          }
  (void)nr;
}

TEST_CASE("forward drizzle v2 affine piece lifecycle rejects malformed "
          "streams (tranche 7)",
          "[forward-drizzle-v2][gate10][target-pieces]") {
  auto f = make_fixture(ColorMode::MONO, BayerPattern::RGGB, 0, 0, 1);
  const int nc = f.plan.canvas_width_native;
  const int nr = f.plan.canvas_height_native;
  const int sw = f.plan.source_width, sh = f.plan.source_height;
  const std::size_t n_src = static_cast<std::size_t>(sw) * sh;
  std::vector<float> s2(n_src, 0.02f);
  std::vector<ForwardDrizzleV2FrameMeta> meta(f.plan.frames.size());
  for (std::size_t i = 0; i < meta.size(); ++i)
    meta[i] = {0.9f, 0.8f, static_cast<std::uint8_t>(i % 2), 0};
  const auto &m = f.plan.frames[0].source_to_canvas;
  const double a6[6] = {m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1), m(1, 2)};
  const ForwardDrizzleV2SourceWindow full{0, 0, sw, sh};
  ForwardDrizzleV2FrameQuality q{};
  std::vector<float> qv(n_src, 0.7f);
  q.q_composite = qv.data();

  SECTION("piece without begin, double begin, finish without piece") {
    ForwardDrizzleV2CpuKernel k;
    REQUIRE(k.reserve(nc, nr, sw, sh,
                      g7_kcfg(f, 1, ColorMode::MONO, BayerPattern::RGGB, 0,
                              true)));
    REQUIRE_FALSE(k.accumulate_affine_piece(a6, 0, 4, full,
                                            f.images[0].data(), s2.data(),
                                            nullptr, &q));
    REQUIRE(k.begin_affine_frame(0, &meta[0]));
    REQUIRE_FALSE(k.begin_affine_frame(1, &meta[1]));  // frame still open
    REQUIRE_FALSE(k.finish_affine_frame(0));           // no pieces yet
    REQUIRE(k.accumulate_affine_piece(a6, 0, nc, full, f.images[0].data(),
                                      s2.data(), nullptr, &q));
    REQUIRE_FALSE(k.finish_affine_frame(1));  // wrong order
    REQUIRE(k.finish_affine_frame(0));
  }
  SECTION("overlap, out-of-order and out-of-range pieces") {
    ForwardDrizzleV2CpuKernel k;
    REQUIRE(k.reserve(nc, nr, sw, sh,
                      g7_kcfg(f, 1, ColorMode::MONO, BayerPattern::RGGB, 0,
                              true)));
    REQUIRE(k.begin_affine_frame(0, &meta[0]));
    REQUIRE(k.accumulate_affine_piece(a6, 0, 8, full, f.images[0].data(),
                                      s2.data(), nullptr, &q));
    REQUIRE_FALSE(k.accumulate_affine_piece(a6, 4, 4, full,
                                            f.images[0].data(), s2.data(),
                                            nullptr, &q));  // overlaps [0,8)
    REQUIRE_FALSE(k.accumulate_affine_piece(a6, nc, 1, full,
                                            f.images[0].data(), s2.data(),
                                            nullptr, &q));  // out of range
    REQUIRE_FALSE(k.accumulate_affine_piece(a6, 8, nc, full,
                                            f.images[0].data(), s2.data(),
                                            nullptr, &q));  // 8+nc > nc
    REQUIRE(k.accumulate_affine_piece(a6, 8, nc - 8, full,
                                      f.images[0].data(), s2.data(), nullptr,
                                      &q));
    REQUIRE(k.finish_affine_frame(0));
    // A later frame cannot backtrack either.
    REQUIRE(k.begin_affine_frame(1, &meta[1]));
    REQUIRE(k.accumulate_affine_piece(a6, 7, 4, full, f.images[1].data(),
                                      s2.data(), nullptr, &q));
    REQUIRE_FALSE(k.accumulate_affine_piece(a6, 0, 4, full,
                                            f.images[1].data(), s2.data(),
                                            nullptr, &q));  // out of order
  }
  SECTION("qmask mismatch across pieces") {
    ForwardDrizzleV2CpuKernel k;
    REQUIRE(k.reserve(nc, nr, sw, sh,
                      g7_kcfg(f, 1, ColorMode::MONO, BayerPattern::RGGB, 0,
                              true)));
    REQUIRE(k.begin_affine_frame(0, &meta[0]));
    REQUIRE(k.accumulate_affine_piece(a6, 0, 7, full, f.images[0].data(),
                                      s2.data(), nullptr, &q));
    ForwardDrizzleV2FrameQuality none{};
    REQUIRE_FALSE(k.accumulate_affine_piece(a6, 7, nc - 7, full,
                                            f.images[0].data(), s2.data(),
                                            nullptr, &none));
    REQUIRE(k.accumulate_affine_piece(a6, 7, nc - 7, full,
                                      f.images[0].data(), s2.data(), nullptr,
                                      &q));
    REQUIRE(k.finish_affine_frame(0));
  }
  SECTION("one-shot calls, skip, begin_band and finalize reject an open "
          "frame") {
    ForwardDrizzleV2CpuKernel k;
    const auto kcfg =
        g7_kcfg(f, 1, ColorMode::MONO, BayerPattern::RGGB, 0, true);
    REQUIRE(k.reserve(nc, nr, sw, sh, kcfg));
    REQUIRE(k.begin_affine_frame(0, &meta[0]));
    REQUIRE_FALSE(k.skip_frame(1, &meta[1]));
    REQUIRE_FALSE(k.accumulate_frame(a6, f.images[1].data(), s2.data(), 1,
                                     &q, &meta[1]));
    REQUIRE_FALSE(k.accumulate_frame_window(a6, full, f.images[1].data(),
                                            s2.data(), nullptr, 1, &q,
                                            &meta[1]));
    REQUIRE_FALSE(k.begin_band(4, kcfg));
    std::vector<ForwardDrizzleV2PixelResult> junk(
        static_cast<std::size_t>(nr) * nc);
    std::vector<ForwardDrizzleV2ProfileResult> junkp(junk.size());
    std::uint64_t dense = 0;
    REQUIRE_FALSE(k.finalize(junk.data(), junkp.data(), &dense));
    // The open frame can still be completed afterwards.
    REQUIRE(k.accumulate_affine_piece(a6, 0, nc, full, f.images[0].data(),
                                      s2.data(), nullptr, &q));
    REQUIRE(k.finish_affine_frame(0));
  }
}

TEST_CASE("forward drizzle v2 CUDA affine target-x pieces match the "
          "full-width piece (tranche 7)",
          "[forward-drizzle-v2][gate10][target-pieces][cuda-parity]") {
  if (!forward_drizzle_cuda_runtime_available()) {
    SUCCEED("CUDA device unavailable");
    return;
  }
  for (int scale : {1, 2})
    for (ColorMode mode : {ColorMode::MONO, ColorMode::OSC}) {
      auto f = make_fixture(mode, BayerPattern::GBRG, 1, -1, scale);
      const int nc = f.plan.canvas_width_native;
      const int nr = f.plan.canvas_height_native;
      const int sw = f.plan.source_width, sh = f.plan.source_height;
      const std::size_t n_src = static_cast<std::size_t>(sw) * sh;
      std::vector<float> s2(n_src, 0.02f);
      std::vector<ForwardDrizzleV2FrameMeta> meta(f.plan.frames.size());
      for (std::size_t i = 0; i < meta.size(); ++i)
        meta[i] = {0.9f, 0.8f, static_cast<std::uint8_t>(i % 2), 0};
      // Persistent workspace across two bands, piece width 3 native cols.
      ForwardDrizzleV2CudaKernel persistent;
      const int max_rows = 5;
      REQUIRE(persistent.reserve(
          nc, max_rows, sw, sh,
          g7_kcfg(f, scale, mode, BayerPattern::GBRG, 0, true)));
      for (const auto band : {std::pair{0, 5}, std::pair{5, 5},
                              std::pair{10, nr - 10}}) {
        const int y0 = band.first, rows = band.second;
        REQUIRE(persistent.begin_band(
            rows, g7_kcfg(f, scale, mode, BayerPattern::GBRG, y0, true)));
        const auto tiled = g7_run_band(persistent, f, s2, meta, true, y0,
                                       rows, 3, false);
        ForwardDrizzleV2CudaKernel ref;
        REQUIRE(ref.reserve(
            nc, rows, sw, sh,
            g7_kcfg(f, scale, mode, BayerPattern::GBRG, y0, true)));
        const auto full =
            g7_run_band(ref, f, s2, meta, true, y0, rows, 0, false);
        REQUIRE(tiled.records.size() == full.records.size());
        for (std::size_t i = 0; i < tiled.records.size(); ++i) {
          const auto &a = full.records[i];
          const auto &b = tiled.records[i];
          REQUIRE(a.contributors == b.contributors);
          REQUIRE(a.robust_state == b.robust_state);
          REQUIRE(a.confidence_state == b.confidence_state);
          REQUIRE(a.conf_degraded == b.conf_degraded);
          if (std::isfinite(a.value))
            REQUIRE(b.value ==
                    Catch::Approx(a.value).epsilon(1e-9).margin(1e-9));
          REQUIRE(b.b == Catch::Approx(a.b).epsilon(1e-9).margin(1e-9));
          REQUIRE(b.n_eff ==
                  Catch::Approx(a.n_eff).epsilon(1e-9).margin(1e-9));
        }
        REQUIRE(tiled.dense == full.dense);
        REQUIRE(tiled.stats.affine_pieces_processed >=
                tiled.stats.frames_processed);
      }
    }
}

TEST_CASE("forward drizzle v2 driver replays multi-piece affine frames "
          "bit-identically (tranche 7)",
          "[forward-drizzle-v2][gate10][driver][target-pieces]") {
  V2StoreFixture fx;
  V2StoreFixture reference;
  auto f = make_fixture(ColorMode::OSC, BayerPattern::GRBG, 0, 0, 1);
  const int nc = f.plan.canvas_width_native;
  const int sw = f.plan.source_width, sh = f.plan.source_height;
  const std::size_t n_src = static_cast<std::size_t>(sw) * sh;
  std::vector<float> s2(n_src, 0.02f);
  std::vector<ForwardDrizzleV2FrameMeta> meta(f.plan.frames.size());
  for (std::size_t i = 0; i < meta.size(); ++i)
    meta[i] = {0.9f, 0.8f, static_cast<std::uint8_t>(i % 2), 0};
  // A local-warp frame exercises the single-piece path inside the same run.
  float cx[16] = {}, cy[16] = {};
  cx[5] = 1.2f;
  cy[6] = -0.7f;
  const auto plan0 = g10_driver_plan(f, 4, 1, true);
  f.plan.frames[2] =
      v2_local_frame(f.plan.frames[2],
                     v2_local_model(plan0.native_height, nc, cx, cy));
  const auto plan = g10_driver_plan(f, 4, 1, true);
  ForwardDrizzleV2DriverOptions cpu;
  cpu.prefer_cuda = false;

  const auto ref =
      run_forward_drizzle_v2(reference.root, plan, sw, sh,
                             g10_provider(f, s2, &meta), cpu);
  REQUIRE(ref.committed);

  // Piece provider: affine frames are emitted as ordered 5-native-column
  // pieces (empty tiles omitted); local frames stay one full-width piece.
  const int tile = 5;
  std::vector<float> win_buf, s2w;
  ForwardDrizzleV2FrameProvider provider =
      [&](int band_y0, int band_rows, std::uint64_t order,
          const ForwardDrizzleV2FramePieceSink &sink) -> bool {
    if (order >= f.plan.frames.size()) return false;
    const auto &frame = f.plan.frames[order];
    const auto &m = frame.source_to_canvas;
    const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                          m(1, 0), m(1, 1), m(1, 2)};
    ForwardDrizzleV2FrameInput in;
    std::copy_n(a6, 6, in.affine6);
    in.meta = meta[order];
    in.has_local_model = frame.has_smooth_local_model;
    if (in.has_local_model) {
      in.warp = v2_warp_descriptor(frame);
      in.source = f.images[order].data();
      in.sigma2 = s2.data();
      in.source_window = {0, 0, sw, sh};
      return sink(in);  // exactly one full-width piece
    }
    bool emitted = false;
    for (int tx = 0; tx < nc; tx += tile) {
      const int tw = std::min(tile, nc - tx);
      const auto box = drizzle_source_scan_box(
          f.plan, frame, plan.internal_scale, band_y0 * plan.internal_scale,
          band_rows * plan.internal_scale, tx * plan.internal_scale,
          tw * plan.internal_scale);
      const int bw = box.x1 - box.x0, bh = box.y1 - box.y0;
      if (bw <= 0 || bh <= 0) continue;
      win_buf.assign(static_cast<std::size_t>(bw) * bh, 0.0f);
      s2w.assign(static_cast<std::size_t>(bw) * bh, 0.0f);
      for (int r = 0; r < bh; ++r)
        for (int c = 0; c < bw; ++c) {
          win_buf[static_cast<std::size_t>(r) * bw + c] =
              f.images[order](box.y0 + r, box.x0 + c);
          s2w[static_cast<std::size_t>(r) * bw + c] =
              s2[static_cast<std::size_t>(box.y0 + r) * sw + box.x0 + c];
        }
      in.source = win_buf.data();
      in.sigma2 = s2w.data();
      in.source_window = {box.x0, box.y0, bw, bh};
      in.target_x_begin_native = tx;
      in.target_cols_native = tw;
      if (!sink(in)) return false;
      emitted = true;
    }
    if (!emitted) {
      in.skip = true;
      return sink(in);
    }
    return true;
  };
  const auto got = run_forward_drizzle_v2(fx.root, plan, sw, sh, provider,
                                          cpu);
  REQUIRE(got.committed);
  std::vector<ForwardDrizzleV2PixelResult> rr, gr;
  std::vector<ForwardDrizzleV2ProfileResult> rp, gp;
  g10_read_store(reference.root, plan, rr, &rp);
  g10_read_store(fx.root, plan, gr, &gp);
  REQUIRE(rr.size() == gr.size());
  for (std::size_t i = 0; i < rr.size(); ++i)
    REQUIRE(g10_bit_eq(rr[i], gr[i]));
  for (std::size_t i = 0; i < rp.size(); ++i)
    REQUIRE(g10_bit_eq(rp[i], gp[i]));
  // One slot transition per frame per band; pieces add up beyond frames.
  REQUIRE(got.totals.slot_transitions ==
          static_cast<std::uint64_t>(plan.band_count) * plan.frame_count);
  REQUIRE(got.totals.affine_pieces_processed >
          got.totals.frames_processed);
  REQUIRE(got.totals.source_samples_launched <=
          ref.totals.source_samples_launched);
}

TEST_CASE("forward drizzle v2 driver rejects malformed piece streams "
          "(tranche 7)",
          "[forward-drizzle-v2][gate10][driver][target-pieces]") {
  auto f = make_fixture(ColorMode::MONO, BayerPattern::GRBG, 0, 0, 1);
  const int nc = f.plan.canvas_width_native;
  const int sw = f.plan.source_width, sh = f.plan.source_height;
  const std::size_t n_src = static_cast<std::size_t>(sw) * sh;
  std::vector<float> s2(n_src, 0.02f);
  const auto plan = g10_driver_plan(f, 4, 1, false);
  ForwardDrizzleV2DriverOptions cpu;
  cpu.prefer_cuda = false;
  auto piece = [&](std::uint64_t order, int tx, int tw, bool skip) {
    ForwardDrizzleV2FrameInput in;
    const auto &m = f.plan.frames[order].source_to_canvas;
    const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                          m(1, 0), m(1, 1), m(1, 2)};
    std::copy_n(a6, 6, in.affine6);
    in.skip = skip;
    in.source = f.images[order].data();
    in.sigma2 = s2.data();
    in.source_window = {0, 0, sw, sh};
    in.target_x_begin_native = tx;
    in.target_cols_native = tw;
    return in;
  };
  auto expect_throw = [&](auto body) {
    V2StoreFixture fx;
    ForwardDrizzleV2FrameProvider p =
        [&](int, int, std::uint64_t order,
            const ForwardDrizzleV2FramePieceSink &sink) -> bool {
      if (order != 0) {
        ForwardDrizzleV2FrameInput in;
        in.skip = true;
        return sink(in);
      }
      return body(sink);
    };
    REQUIRE_THROWS_AS(
        run_forward_drizzle_v2(fx.root, plan, sw, sh, p, cpu),
        std::runtime_error);
  };
  expect_throw([&](const ForwardDrizzleV2FramePieceSink &) {
    return true;  // zero emissions
  });
  expect_throw([&](const ForwardDrizzleV2FramePieceSink &s) {
    if (!s(piece(0, 0, nc, true))) return false;  // skip ...
    return s(piece(0, 0, nc, false));            // ... then a piece
  });
  expect_throw([&](const ForwardDrizzleV2FramePieceSink &s) {
    if (!s(piece(0, 0, 8, false))) return false;
    return s(piece(0, 4, 4, false));  // overlaps [0,8)
  });
  expect_throw([&](const ForwardDrizzleV2FramePieceSink &s) {
    if (!s(piece(0, 7, 4, false))) return false;
    return s(piece(0, 0, 4, false));  // out of order
  });
  expect_throw([&](const ForwardDrizzleV2FramePieceSink &s) {
    return s(piece(0, nc - 2, 4, false));  // out of range
  });
  expect_throw([&](const ForwardDrizzleV2FramePieceSink &s) {
    auto a = piece(0, 0, 7, false), b = piece(0, 7, nc - 7, false);
    b.affine6[2] += 1.0;  // changed transform between pieces
    if (!s(a)) return false;
    return s(b);
  });
  expect_throw([&](const ForwardDrizzleV2FramePieceSink &s) {
    auto local = piece(0, 0, 0, false);
    local.has_local_model = true;
    local.warp = ForwardDrizzleV2LocalWarp{};
    if (!s(local)) return false;
    return s(local);  // local frames are single-piece
  });
}

TEST_CASE("forward drizzle v2 affine target tiles bound read amplification "
          "(tranche 7, geometry only)",
          "[forward-drizzle-v2][gate10][target-pieces]") {
  // Synthetic ~4k source rotated ~1.7 deg + shear onto a 4526-wide canvas:
  // per-band full-width union boxes over-read; 512-native-col tiles do not.
  // (The per-tile axis-aligned bbox inflation of a rotated w x h rect is
  // 1 + sin(2*theta)*(w/h + h/w)/2, so ~8 deg cannot reach <= 1.1 no matter
  // the tile width; ~2 deg registration rotation is the realistic regime.)
  RegistrationSamplingPlan sampling{};
  sampling.source_width = 3840;
  sampling.source_height = 2160;
  sampling.canvas_width_native = 4526;
  sampling.canvas_height_native = 2600;
  FrameSamplingTransform fr;
  fr.valid = true;
  fr.source_to_canvas_affine_valid = true;
  const double c = std::cos(0.03), s = std::sin(0.03);
  const double m00 = c + 0.02 * s, m01 = -s + 0.02 * c;
  const double m10 = s, m11 = c;
  const double tx =
      0.5 * 4526.0 - (m00 * 1920.0 + m01 * 1080.0);
  const double ty = 0.5 * 2600.0 - (m10 * 1920.0 + m11 * 1080.0);
  fr.source_to_canvas = affine(m00, m01, tx, m10, m11, ty);
  const int scale = 2;
  const int band_rows_native = 256;
  std::uint64_t tiled_area = 0, untiled_area = 0;
  for (int y = 0; y < sampling.canvas_height_native;
       y += band_rows_native) {
    const int rows =
        std::min(band_rows_native, sampling.canvas_height_native - y);
    const auto whole = drizzle_source_scan_box(sampling, fr, scale,
                                               y * scale, rows * scale);
    untiled_area += static_cast<std::uint64_t>(
        std::max(0, whole.x1 - whole.x0)) *
        std::max(0, whole.y1 - whole.y0);
    for (int x = 0; x < sampling.canvas_width_native;
         x += kAffineTargetTileColsNative) {
      const int tw = std::min(kAffineTargetTileColsNative,
                              sampling.canvas_width_native - x);
      const auto box = drizzle_source_scan_box(
          sampling, fr, scale, y * scale, rows * scale, x * scale,
          tw * scale);
      if (box.x1 <= box.x0 || box.y1 <= box.y0) continue;
      tiled_area += static_cast<std::uint64_t>(box.x1 - box.x0) *
                    (box.y1 - box.y0);
    }
  }
  const double base = static_cast<double>(sampling.source_width) *
                      sampling.source_height;
  REQUIRE(tiled_area > 0);
  REQUIRE(tiled_area / base <= 1.1);
  REQUIRE(untiled_area / base > 1.1);
  REQUIRE(tiled_area < untiled_area);
}

// ---- tranche 8: canonical ragged affine sample list ----------------------

namespace {

// Builds the canonical sample list + aligned packed quality for one frame/
// band the same way the production provider does (spans -> storage spans ->
// packed values -> samples; divisor-1 quality codes so the reference window
// path can consume the identical quantized values through a packed
// descriptor).
struct G8Frame {
  std::vector<DrizzleAffineSourceSpan> active, storage;
  std::vector<float> storage_vals;
  std::vector<ForwardDrizzleV2SourceSample> samples;
  std::vector<std::uint16_t> qc;
  std::vector<std::uint8_t> vc;
  std::vector<std::uint16_t> cells_plane;
  std::vector<std::uint8_t> veto_plane;
  bool empty = false;
};

float g8_qvalue(int sx, int sy) {
  return 0.5f + 0.01f * static_cast<float>((sx + sy) % 37);
}
bool g8_vetoed(int sx, int sy) { return (sx * 5 + sy * 3) % 29 == 0; }

G8Frame g8_build_frame(const Fixture &f, std::size_t fr, float pixfrac,
                       int y_begin, int rows, bool sigma_plane,
                       bool profiles, int sw, int sh) {
  G8Frame g;
  const auto &frame = f.plan.frames[fr];
  drizzle_affine_source_spans_into(f.plan, frame, pixfrac, y_begin, rows,
                                   g.active);
  if (g.active.empty()) {
    g.empty = true;
    return g;
  }
  std::vector<int> ra, rs;
  std::vector<std::size_t> soff;
  forward_drizzle_v2_affine_storage_spans(g.active, sw, sh, sigma_plane, ra,
                                        g.storage);
  // Equivalent of the packed interval reads, straight from the image.
  for (const auto &s : g.storage)
    for (int x = s.x_begin; x < s.x_end; ++x)
      g.storage_vals.push_back(f.images[fr](s.source_y, x));
  forward_drizzle_v2_build_affine_samples(
      g.active, g.storage, g.storage_vals, sigma_plane, 0.05, 0.02,
      0.5 * pixfrac, sw, sh, rs, soff, g.samples);
  if (profiles) {
    const std::size_t n = g.samples.size();
    g.qc.resize(n);
    g.vc.resize(n);
    g.cells_plane.assign(static_cast<std::size_t>(sw) * sh, 0);
    g.veto_plane.assign(static_cast<std::size_t>(sw) * sh, 0);
    for (int y = 0; y < sh; ++y)
      for (int x = 0; x < sw; ++x) {
        const std::size_t i = static_cast<std::size_t>(y) * sw + x;
        g.cells_plane[i] = quantize_quality(g8_qvalue(x, y));
        g.veto_plane[i] = g8_vetoed(x, y) ? 1 : 0;
      }
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t p =
          static_cast<std::size_t>(g.samples[i].source_y) * sw +
          g.samples[i].source_x;
      g.qc[i] = g.cells_plane[p];
      g.vc[i] = g.veto_plane[p];
    }
  }
  return g;
}

// Runs one band on `kernel` through the canonical sample-list path.
G7BandOut g8_run_band(ForwardDrizzleV2Kernel &kernel, const Fixture &f,
                      const std::vector<ForwardDrizzleV2FrameMeta> &meta,
                      bool profiles, int y_begin, int rows,
                      bool sigma_plane) {
  const int nc = f.plan.canvas_width_native;
  const int channels = f.plan.color_mode == ColorMode::MONO ? 1 : 3;
  const int sw = f.plan.source_width, sh = f.plan.source_height;
  G7BandOut out;
  for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
    const auto &frame = f.plan.frames[fr];
    const auto &m = frame.source_to_canvas;
    const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                          m(1, 0), m(1, 1), m(1, 2)};
    G8Frame g = g8_build_frame(f, fr, f.cfg.pixfrac, y_begin, rows,
                               sigma_plane, profiles, sw, sh);
    if (g.empty) {
      REQUIRE(kernel.skip_frame(fr, profiles ? &meta[fr] : nullptr));
      continue;
    }
    ForwardDrizzleV2AlignedQuality aq{};
    if (profiles) {
      aq.qc = g.qc.data();
      aq.vc = g.vc.data();
      aq.presence_mask = 1u;
    }
    REQUIRE(kernel.accumulate_frame_affine_samples(
        a6, g.samples.data(), g.samples.size(), sigma_plane,
        profiles ? &aq : nullptr, fr, profiles ? &meta[fr] : nullptr));
    out.expected_launched += g.samples.size();
  }
  const std::size_t np =
      static_cast<std::size_t>(rows) * nc * channels;
  out.records.assign(np, ForwardDrizzleV2PixelResult{});
  if (profiles) out.profiles.assign(np, ForwardDrizzleV2ProfileResult{});
  REQUIRE(kernel.finalize(out.records.data(),
                          profiles ? out.profiles.data() : nullptr,
                          &out.dense));
  out.stats = kernel.stats();
  return out;
}

// Full-window rectangle reference consuming the identical quantized quality
// through a divisor-1 packed descriptor and the identical explicit sigma2
// plane (the builder's oracle at every included coordinate).
G7BandOut g8_ref_band(ForwardDrizzleV2Kernel &kernel, const Fixture &f,
                      const std::vector<float> &s2,
                      const std::vector<ForwardDrizzleV2FrameMeta> &meta,
                      bool profiles, int y_begin, int rows,
                      bool sigma_plane) {
  const int nc = f.plan.canvas_width_native;
  const int channels = f.plan.color_mode == ColorMode::MONO ? 1 : 3;
  const int sw = f.plan.source_width, sh = f.plan.source_height;
  G7BandOut out;
  for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
    const auto &frame = f.plan.frames[fr];
    const auto &m = frame.source_to_canvas;
    const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                          m(1, 0), m(1, 1), m(1, 2)};
    const auto band_box = drizzle_source_scan_box(
        f.plan, frame, f.cfg.internal_scale, y_begin * f.cfg.internal_scale,
        rows * f.cfg.internal_scale);
    if (band_box.x1 <= band_box.x0 || band_box.y1 <= band_box.y0) {
      REQUIRE(kernel.skip_frame(fr, profiles ? &meta[fr] : nullptr));
      continue;
    }
    ForwardDrizzleV2SourceWindow w{0, 0, sw, sh, 0, 0, sw, sh};
    ForwardDrizzleV2FrameQuality q{};
    ForwardDrizzleV2PackedQualityPlane pk{};
    std::vector<std::uint16_t> cells;
    std::vector<std::uint8_t> veto;
    if (profiles) {
      cells.assign(static_cast<std::size_t>(sw) * sh, 0);
      veto.assign(static_cast<std::size_t>(sw) * sh, 0);
      for (int y = 0; y < sh; ++y)
        for (int x = 0; x < sw; ++x) {
          const std::size_t i = static_cast<std::size_t>(y) * sw + x;
          cells[i] = quantize_quality(g8_qvalue(x, y));
          veto[i] = g8_vetoed(x, y) ? 1 : 0;
        }
      pk.storage_x_begin = 0;
      pk.storage_y_begin = 0;
      pk.storage_width = sw;
      pk.storage_height = sh;
      pk.storage_divisor = 1;
      pk.cells = cells.data();
      pk.veto = veto.data();
      q.qc_packed = pk;
    }
    out.expected_launched += static_cast<std::uint64_t>(sw) * sh;
    REQUIRE(kernel.accumulate_frame_window(
        a6, w, f.images[fr].data(), sigma_plane ? s2.data() : nullptr,
        nullptr, fr, profiles ? &q : nullptr,
        profiles ? &meta[fr] : nullptr));
  }
  const std::size_t np =
      static_cast<std::size_t>(rows) * nc * channels;
  out.records.assign(np, ForwardDrizzleV2PixelResult{});
  if (profiles) out.profiles.assign(np, ForwardDrizzleV2ProfileResult{});
  REQUIRE(kernel.finalize(out.records.data(),
                          profiles ? out.profiles.data() : nullptr,
                          &out.dense));
  out.stats = kernel.stats();
  return out;
}

// Numeric parity between two PixelResult vectors (CUDA vs CPU tolerance).
void g8_near(const ForwardDrizzleV2PixelResult &e,
             const ForwardDrizzleV2PixelResult &g) {
  REQUIRE(g.robust_state == e.robust_state);
  REQUIRE(g.confidence_state == e.confidence_state);
  REQUIRE(g.contributors == e.contributors);
  REQUIRE(g.conf_degraded == e.conf_degraded);
  REQUIRE(g.value == Catch::Approx(e.value).epsilon(1e-5).margin(1e-9));
  REQUIRE(g.b == Catch::Approx(e.b).epsilon(1e-5).margin(1e-9));
  REQUIRE(g.n_eff == Catch::Approx(e.n_eff).epsilon(1e-5).margin(1e-9));
  REQUIRE(g.confidence ==
          Catch::Approx(e.confidence).epsilon(1e-5).margin(1e-9));
}

}  // namespace

TEST_CASE("forward drizzle v2 CPU ragged affine sample list is bit-identical "
          "to the full-source window path (tranche 8)",
          "[forward-drizzle-v2][gate10][affine-samples]") {
  const BayerPattern patterns[] = {BayerPattern::RGGB, BayerPattern::BGGR,
                                   BayerPattern::GRBG, BayerPattern::GBRG};
  for (int scale : {1, 2})
    for (ColorMode mode : {ColorMode::MONO, ColorMode::OSC})
      for (BayerPattern pattern : patterns)
        for (bool profiles : {false, true})
          for (bool sigma_plane : {false, true}) {
            auto f = make_fixture(mode, pattern, 1, -1, scale);
            const int nc = f.plan.canvas_width_native;
            const int sw = f.plan.source_width, sh = f.plan.source_height;
            const std::size_t n_src = static_cast<std::size_t>(sw) * sh;
            std::vector<ForwardDrizzleV2FrameMeta> meta(
                f.plan.frames.size());
            for (std::size_t i = 0; i < meta.size(); ++i)
              meta[i] = {0.9f, 0.8f, static_cast<std::uint8_t>(i % 2), 0};
            for (const auto band :
                 {std::pair{0, 5}, std::pair{4, 5}, std::pair{9, 4}}) {
              const int y0 = band.first, rows = band.second;
              INFO("scale=" << scale << " mode=" << static_cast<int>(mode)
                            << " pat=" << static_cast<int>(pattern)
                            << " profiles=" << profiles
                            << " sigma=" << sigma_plane << " band=" << y0
                            << "/" << rows);
              ForwardDrizzleV2KernelConfig k =
                  g7_kcfg(f, scale, mode, pattern, y0, profiles);
              k.sigma2_plane = sigma_plane;
              ForwardDrizzleV2CpuKernel ref;
              REQUIRE(ref.reserve(nc, rows, sw, sh, k));
              // Reference: full window, per-frame explicit sigma2 plane and
              // divisor-1 packed quality identical to the aligned arrays.
              G7BandOut full;
              const int channels = mode == ColorMode::MONO ? 1 : 3;
              for (std::size_t fr = 0; fr < f.plan.frames.size(); ++fr) {
                const auto &frame = f.plan.frames[fr];
                const auto &m = frame.source_to_canvas;
                const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                                      m(1, 0), m(1, 1), m(1, 2)};
                const auto band_box = drizzle_source_scan_box(
                    f.plan, frame, scale, y0 * scale, rows * scale);
                if (band_box.x1 <= band_box.x0 || band_box.y1 <= band_box.y0) {
                  REQUIRE(ref.skip_frame(fr,
                                         profiles ? &meta[fr] : nullptr));
                  continue;
                }
                const auto plane = forward_drizzle_v2_sigma2_plane(
                    f.images[fr], 0.05, 0.02, 0.5 * f.cfg.pixfrac);
                ForwardDrizzleV2SourceWindow w{0, 0, sw, sh, 0, 0, sw, sh};
                ForwardDrizzleV2FrameQuality q{};
                ForwardDrizzleV2PackedQualityPlane pk{};
                std::vector<std::uint16_t> cells;
                std::vector<std::uint8_t> veto;
                if (profiles) {
                  cells.assign(n_src, 0);
                  veto.assign(n_src, 0);
                  for (int y = 0; y < sh; ++y)
                    for (int x = 0; x < sw; ++x) {
                      const std::size_t i =
                          static_cast<std::size_t>(y) * sw + x;
                      cells[i] = quantize_quality(g8_qvalue(x, y));
                      veto[i] = g8_vetoed(x, y) ? 1 : 0;
                    }
                  pk.storage_width = sw;
                  pk.storage_height = sh;
                  pk.storage_divisor = 1;
                  pk.cells = cells.data();
                  pk.veto = veto.data();
                  q.qc_packed = pk;
                }
                REQUIRE(ref.accumulate_frame_window(
                    a6, w, f.images[fr].data(),
                    sigma_plane ? plane.data() : nullptr, nullptr, fr,
                    profiles ? &q : nullptr,
                    profiles ? &meta[fr] : nullptr));
              }
              full.records.assign(static_cast<std::size_t>(rows) * nc *
                                      channels,
                                  ForwardDrizzleV2PixelResult{});
              if (profiles)
                full.profiles.assign(full.records.size(),
                                     ForwardDrizzleV2ProfileResult{});
              REQUIRE(ref.finalize(full.records.data(),
                                   profiles ? full.profiles.data() : nullptr,
                                   &full.dense));
              full.stats = ref.stats();
              ForwardDrizzleV2CpuKernel smp;
              REQUIRE(smp.reserve(nc, rows, sw, sh, k));
              const auto got = g8_run_band(smp, f, meta, profiles, y0, rows,
                                           sigma_plane);
              for (std::size_t i = 0; i < full.records.size(); ++i)
                REQUIRE(g10_bit_eq(full.records[i], got.records[i]));
              for (std::size_t i = 0; i < full.profiles.size(); ++i)
                REQUIRE(g10_bit_eq(full.profiles[i], got.profiles[i]));
              REQUIRE(got.dense == full.dense);
              // The ragged path may legitimately skip a frame whose
              // margin-inflated band box was non-empty but whose true
              // droplet coverage is empty (processed-but-zero-overlap on
              // the rectangle path); every processed frame is bit-equal.
              REQUIRE(got.stats.frames_processed <=
                      full.stats.frames_processed);
              // Skipped frames still consume a stream slot.
              REQUIRE(got.stats.slot_transitions ==
                      full.stats.slot_transitions);
              REQUIRE(got.stats.source_samples_launched ==
                      got.expected_launched);
              REQUIRE(got.stats.affine_samples_processed ==
                      got.stats.source_samples_launched);
              REQUIRE((got.stats.affine_span_rows > 0) ==
                      (got.stats.affine_samples_processed > 0));
              REQUIRE(got.stats.source_samples_launched <=
                      full.stats.source_samples_launched);
            }
          }
}

TEST_CASE("forward drizzle v2 CPU affine sample validation fails closed "
          "(tranche 8)", "[forward-drizzle-v2][gate10][affine-samples]") {
  auto f = make_fixture(ColorMode::MONO, BayerPattern::GRBG, 0, 0, 1);
  const int nc = f.plan.canvas_width_native;
  const int sw = f.plan.source_width, sh = f.plan.source_height;
  auto k = g7_kcfg(f, 1, ColorMode::MONO, BayerPattern::GRBG, 0, true);
  ForwardDrizzleV2CpuKernel kernel;
  REQUIRE(kernel.reserve(nc, 4, sw, sh, k));
  const double a6[6] = {1, 0, 2, 0, 1, 2};
  std::vector<ForwardDrizzleV2SourceSample> s = {{0, 0, 1.0f, 0.0f},
                                               {1, 0, 2.0f, 0.0f},
                                               {0, 1, 3.0f, 0.0f}};
  // Out of extent, duplicate and non-canonical order are rejected.
  auto bad = s;
  bad[2].source_x = static_cast<std::uint32_t>(sw);
  REQUIRE_FALSE(kernel.accumulate_frame_affine_samples(
      a6, bad.data(), bad.size(), false, nullptr, 0, nullptr));
  bad = s;
  bad[1].source_y = static_cast<std::uint32_t>(sh);
  REQUIRE_FALSE(kernel.accumulate_frame_affine_samples(
      a6, bad.data(), bad.size(), false, nullptr, 0, nullptr));
  bad = s;
  bad[1] = s[2];
  bad[2] = s[1];
  REQUIRE_FALSE(kernel.accumulate_frame_affine_samples(
      a6, bad.data(), bad.size(), false, nullptr, 0, nullptr));
  bad = s;
  bad[1] = s[0];  // duplicate
  REQUIRE_FALSE(kernel.accumulate_frame_affine_samples(
      a6, bad.data(), bad.size(), false, nullptr, 0, nullptr));
  REQUIRE_FALSE(kernel.accumulate_frame_affine_samples(
      a6, nullptr, s.size(), false, nullptr, 0, nullptr));
  REQUIRE_FALSE(kernel.accumulate_frame_affine_samples(
      a6, s.data(), 0, false, nullptr, 0, nullptr));
  // Non-finite affine rejected.
  const double bad6[6] = {1, 0, std::numeric_limits<double>::infinity(),
                          0, 1, 2};
  REQUIRE_FALSE(kernel.accumulate_frame_affine_samples(
      bad6, s.data(), s.size(), false, nullptr, 0, nullptr));
  // Presence bit without the code array rejected.
  ForwardDrizzleV2AlignedQuality aq{};
  aq.presence_mask = 1u;
  REQUIRE_FALSE(kernel.accumulate_frame_affine_samples(
      a6, s.data(), s.size(), false, &aq, 0, nullptr));
  // A valid frame then succeeds and the kernel stays consistent.
  const ForwardDrizzleV2FrameMeta meta{0.9f, 0.8f, 1, 0};
  REQUIRE(kernel.accumulate_frame_affine_samples(
      a6, s.data(), s.size(), false, nullptr, 0, &meta));
}

TEST_CASE("forward drizzle v2 CUDA ragged affine sample list matches the CPU "
          "sample path (tranche 8)", "[forward-drizzle-v2][gate10][cuda]") {
  if (!forward_drizzle_cuda_runtime_available()) return;
  for (int scale : {1, 2})
    for (ColorMode mode : {ColorMode::MONO, ColorMode::OSC})
      for (bool profiles : {false, true})
        for (bool sigma_plane : {false, true}) {
          auto f = make_fixture(mode, BayerPattern::GBRG, 1, -1, scale);
          const int nc = f.plan.canvas_width_native;
          const int sw = f.plan.source_width, sh = f.plan.source_height;
          std::vector<ForwardDrizzleV2FrameMeta> meta(
              f.plan.frames.size());
          for (std::size_t i = 0; i < meta.size(); ++i)
            meta[i] = {0.9f, 0.8f, static_cast<std::uint8_t>(i % 2), 0};
          for (const auto band : {std::pair{0, 5}, std::pair{9, 4}}) {
            const int y0 = band.first, rows = band.second;
            ForwardDrizzleV2KernelConfig k =
                g7_kcfg(f, scale, mode, BayerPattern::GBRG, y0, profiles);
            k.sigma2_plane = sigma_plane;
            ForwardDrizzleV2CpuKernel cpu;
            REQUIRE(cpu.reserve(nc, rows, sw, sh, k));
            const auto ref =
                g8_run_band(cpu, f, meta, profiles, y0, rows, sigma_plane);
            ForwardDrizzleV2CudaKernel gpu;
            REQUIRE(gpu.reserve(nc, rows, sw, sh, k));
            const auto got =
                g8_run_band(gpu, f, meta, profiles, y0, rows, sigma_plane);
            REQUIRE(got.records.size() == ref.records.size());
            for (std::size_t i = 0; i < ref.records.size(); ++i)
              g8_near(ref.records[i], got.records[i]);
            REQUIRE(got.dense == ref.dense);
            REQUIRE(got.stats.source_samples_launched ==
                    ref.stats.source_samples_launched);
            REQUIRE(got.stats.affine_samples_processed ==
                    ref.stats.affine_samples_processed);
            REQUIRE(got.stats.affine_span_rows ==
                    ref.stats.affine_span_rows);
          }
        }
}

TEST_CASE("forward drizzle v2 driver sample-list provider produces identical "
          "store bytes to the rectangle path (tranche 8)",
          "[forward-drizzle-v2][gate10][driver][affine-samples]") {
  V2StoreFixture fx_rect, fx_smp;
  auto f = make_fixture(ColorMode::MONO, BayerPattern::GRBG, 0, 0, 2);
  const int sw = f.plan.source_width, sh = f.plan.source_height;
  std::vector<ForwardDrizzleV2FrameMeta> meta(f.plan.frames.size());
  for (std::size_t i = 0; i < meta.size(); ++i)
    meta[i] = {0.9f, 0.8f, static_cast<std::uint8_t>(i % 2), 0};
  const auto plan = g10_driver_plan(f, 4, 1, true);
  ForwardDrizzleV2DriverOptions opts;
  opts.prefer_cuda = false;

  // Rectangle reference with the same per-frame oracle sigma2 the sample
  // builder computes (0.05 / 0.02 / half).
  auto rect_provider =
      [&f, &meta](int, int, std::uint64_t order,
                  const ForwardDrizzleV2FramePieceSink &sink) {
        if (order >= f.plan.frames.size()) return false;
        const auto &frame = f.plan.frames[order];
        const auto &m = frame.source_to_canvas;
        ForwardDrizzleV2FrameInput in;
        const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                              m(1, 0), m(1, 1), m(1, 2)};
        std::copy_n(a6, 6, in.affine6);
        in.meta = meta[order];
        in.source = f.images[order].data();
        in.source_window = {0, 0, f.plan.source_width,
                            f.plan.source_height};
        static thread_local std::vector<float> s2p;
        s2p = forward_drizzle_v2_sigma2_plane(f.images[order], 0.05, 0.02,
                                              0.5 * f.cfg.pixfrac);
        in.sigma2 = s2p.data();
        return sink(in);
      };
  const auto rect_result = run_forward_drizzle_v2(
      fx_rect.root, plan, sw, sh, rect_provider, opts);
  REQUIRE(rect_result.committed);

  // Sample-list provider: canonical spans + samples per band, one
  // full-target piece per affine frame.
  auto sample_provider =
      [&f, &meta](int y_begin, int rows, std::uint64_t order,
                  const ForwardDrizzleV2FramePieceSink &sink) {
        if (order >= f.plan.frames.size()) return false;
        const auto &frame = f.plan.frames[order];
        const int sw = f.plan.source_width, sh = f.plan.source_height;
        ForwardDrizzleV2FrameInput in;
        const auto &m = frame.source_to_canvas;
        const double a6[6] = {m(0, 0), m(0, 1), m(0, 2),
                              m(1, 0), m(1, 1), m(1, 2)};
        std::copy_n(a6, 6, in.affine6);
        in.meta = meta[order];
        if (frame.has_smooth_local_model) {
          // Compatibility fallback: the full-source rectangle path.
          in.has_local_model = true;
          in.warp = v2_warp_descriptor(frame);
          in.source = f.images[order].data();
          in.source_window = {0, 0, sw, sh};
          return sink(in);
        }
        // The driver consumes the pointers synchronously inside sink().
        G8Frame g = g8_build_frame(f, order, f.cfg.pixfrac, y_begin, rows,
                                   true, false, sw, sh);
        if (g.empty) {
          in.skip = true;
          return sink(in);
        }
        in.has_affine_samples = true;
        in.affine_samples = g.samples.data();
        in.affine_sample_count = g.samples.size();
        in.sigma2_present = true;
        return sink(in);
      };
  const auto smp_result = run_forward_drizzle_v2(
      fx_smp.root, plan, sw, sh, sample_provider, opts);
  REQUIRE(smp_result.committed);
  // The ragged path legitimately skips band x frame combinations whose
  // margin-inflated scan box was non-empty but whose true droplet coverage
  // is empty; every processed frame is bit-identical.
  REQUIRE(smp_result.totals.frames_processed <=
          rect_result.totals.frames_processed);
  REQUIRE(smp_result.totals.affine_samples_processed > 0);
  REQUIRE(smp_result.totals.affine_span_rows > 0);
  // Launched samples never exceed the rectangle path's per-frame full
  // source launches.
  REQUIRE(smp_result.totals.source_samples_launched <=
          rect_result.totals.source_samples_launched);
  std::vector<ForwardDrizzleV2PixelResult> ra, rb;
  std::vector<ForwardDrizzleV2ProfileResult> pa, pb;
  g10_read_store(fx_rect.root, plan, ra, &pa);
  g10_read_store(fx_smp.root, plan, rb, &pb);
  REQUIRE(ra.size() == rb.size());
  REQUIRE(pa.size() == pb.size());
  for (std::size_t i = 0; i < ra.size(); ++i)
    REQUIRE(g10_bit_eq(ra[i], rb[i]));
  for (std::size_t i = 0; i < pa.size(); ++i)
    REQUIRE(g10_bit_eq(pa[i], pb[i]));
}

TEST_CASE("forward drizzle v2 ragged spans bound source launches under "
          "rotation where bounding boxes cannot (tranche 8 gate)",
          "[forward-drizzle-v2][gate10][affine-samples]") {
  // Synthetic A2-like geometry: 3840x2160 source, ~10 deg rotation + shear
  // on a 4526-wide canvas, band_rows 96. Ragged spans must keep the summed
  // launched-sample count <=1.1x the full-source baseline; the axis-aligned
  // band box of the SAME transform exceeds it.
  const int sw = 3840, sh = 2160, cw = 4526, ch = 3200;
  const double th = 10.0 * M_PI / 180.0;
  RegistrationSamplingPlan plan;
  plan.source_width = sw;
  plan.source_height = sh;
  plan.canvas_width_native = cw;
  plan.canvas_height_native = ch;
  plan.color_mode = ColorMode::MONO;
  FrameSamplingTransform fr;
  fr.frame_id = "a2-like";
  fr.source_index = 0;
  fr.valid = true;
  fr.source_to_canvas = affine(std::cos(th) * 1.02, -std::sin(th) * 1.02,
                               120.0, std::sin(th) * 0.98 + 0.01,
                               std::cos(th) * 0.98, 60.0);
  fr.source_to_canvas_affine_valid = true;
  plan.frames.push_back(fr);
  std::uint64_t launched = 0, box_elems = 0;
  for (int y0 = 0; y0 < ch; y0 += 96) {
    const int rows = std::min(96, ch - y0);
    const auto spans =
        drizzle_affine_source_spans(plan, fr, 0.8f, y0, rows);
    for (const auto &s : spans)
      launched += static_cast<std::uint64_t>(s.x_end - s.x_begin);
    const auto box = drizzle_source_scan_box(plan, fr, 2, y0 * 2, rows * 2);
    if (box.x1 > box.x0 && box.y1 > box.y0)
      box_elems += static_cast<std::uint64_t>(box.x1 - box.x0) *
                   (box.y1 - box.y0);
  }
  const double baseline = static_cast<double>(sw) * sh;
  INFO("launched=" << launched << " baseline=" << baseline
                   << " box=" << box_elems);
  REQUIRE(static_cast<double>(launched) / baseline <= 1.1);
  // The rotated band's full bounding box legitimately exceeds 1.1 (the
  // inflation grows with canvas height, not with tile width) — this is the
  // regression signal the ragged path removes.
  REQUIRE(static_cast<double>(box_elems) / baseline > 1.1);

  // Conditional real-artifact gate (A2 registration sampling): when the
  // recorded run artifact exists, every affine transform's summed launched
  // amplification over planned 96-row bands must be <=1.1.
  const fs::path art =
      "runs/gate10_v2/a2_m42_affine/artifacts/registration_sampling.json";
  std::error_code ec;
  if (fs::is_regular_file(fs::symlink_status(art, ec))) {
    std::ifstream in(art);
    const nlohmann::json j = nlohmann::json::parse(in);
    RegistrationSamplingPlan rplan;
    rplan.source_width = j.at("source_width").get<int>();
    rplan.source_height = j.at("source_height").get<int>();
    rplan.canvas_width_native = j.at("canvas_width_native").get<int>();
    rplan.canvas_height_native = j.at("canvas_height_native").get<int>();
    rplan.color_mode = ColorMode::MONO;
    std::vector<FrameSamplingTransform> affs;
    for (const auto &fj : j.at("frames")) {
      FrameSamplingTransform tf;
      tf.frame_id = fj.at("frame_id").get<std::string>();
      tf.source_index = affs.size();
      tf.valid = fj.value("valid", true);
      const auto &m = fj.at("source_to_canvas");
      tf.source_to_canvas = affine(m[0].get<double>(), m[1].get<double>(),
                                   m[2].get<double>(), m[3].get<double>(),
                                   m[4].get<double>(), m[5].get<double>());
      tf.source_to_canvas_affine_valid =
          fj.value("source_to_canvas_affine_valid", true);
      tf.has_smooth_local_model = fj.value("has_smooth_local_model", false);
      rplan.frames.push_back(tf);
      if (tf.valid && tf.source_to_canvas_affine_valid &&
          !tf.has_smooth_local_model)
        affs.push_back(tf);
    }
    REQUIRE(affs.size() == 610);
    std::uint64_t rlaunched = 0;
    for (const auto &tf : affs)
      for (int y0 = 0; y0 < rplan.canvas_height_native; y0 += 96) {
        const int rows =
            std::min(96, rplan.canvas_height_native - y0);
        const auto spans =
            drizzle_affine_source_spans(rplan, tf, 0.8f, y0, rows);
        for (const auto &s : spans)
          rlaunched += static_cast<std::uint64_t>(s.x_end - s.x_begin);
      }
    const double rbaseline =
        static_cast<double>(affs.size()) * rplan.source_width *
        rplan.source_height;
    INFO("real A2 launched=" << rlaunched << " baseline=" << rbaseline);
    REQUIRE(static_cast<double>(rlaunched) / rbaseline <= 1.1);
  }
}

// ---------------------------------------------------------------------------
// Pilot + full-frame estimator (plan estimator "reservoir_pilot_full_frame")
// ---------------------------------------------------------------------------
namespace {

struct FullFrameRun {
  std::vector<ForwardDrizzleV2PixelResult> results;
  std::vector<ForwardDrizzleV2ProfileResult> profiles;
  ForwardDrizzleV2PrototypeStats stats;
};

// Deterministic per-(frame, pixel) noise in [-sqrt(3), sqrt(3)] (variance 1).
double ff_noise(std::uint64_t frame, int x, int y) {
  std::uint64_t z = frame * 0x9e3779b97f4a7c15ULL +
                    static_cast<std::uint64_t>(x) * 0xbf58476d1ce4e5b9ULL +
                    static_cast<std::uint64_t>(y) * 0x94d049bb133111ebULL;
  z ^= z >> 30;
  z *= 0xbf58476d1ce4e5b9ULL;
  z ^= z >> 27;
  z *= 0x94d049bb133111ebULL;
  z ^= z >> 31;
  const double u = static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
  return (2.0 * u - 1.0) * 1.7320508075688772;
}

// N frames sharing frame 0's geometry; each frame is the fixture's base image
// plus noise*sigma and, for frames with (frame % 10) < outlier_tenths, a
// +outlier_shift bump. full=true streams pilot frames first, calls end_pilot
// and then streams the rest; full=false streams the historical 0..N-1 order.
FullFrameRun run_full_frame(const Fixture &f, std::uint64_t n, double sigma,
                            int outlier_tenths, double outlier_shift,
                            bool full, std::uint64_t noise_seed = 0,
                            bool use_cuda = false) {
  const int nc = f.plan.canvas_width_native;
  const int nr = f.plan.canvas_height_native;
  const std::size_t nplane = static_cast<std::size_t>(nc) * nr;
  ForwardDrizzleV2KernelConfig kcfg;
  kcfg.internal_scale = f.cfg.internal_scale;
  kcfg.stream_length = n;
  kcfg.half = 0.5 * f.cfg.pixfrac;
  kcfg.bayer_pattern = static_cast<int>(f.plan.bayer_pattern);
  kcfg.cfa_origin_x = f.plan.cfa_origin_x;
  kcfg.cfa_origin_y = f.plan.cfa_origin_y;
  kcfg.mono = false;
  kcfg.emit_profiles = true;
  kcfg.robust_passes = 4;
  // Wide symmetric bounds (raw MAD): the frozen pilot bounds only protect
  // against outliers. With the production 2/4 asymmetric bounds the pilot
  // sample's bound jitter dominates the full-frame error (see the plan).
  kcfg.sigma_low = 4.0;
  kcfg.sigma_high = 4.0;
  kcfg.min_clip_contributors = 5;
  kcfg.min_candidates = 5;
  kcfg.shared_frame_rejection = true;
  kcfg.shared_frame_rejection_consensus = 0.5;
  kcfg.full_frame_estimator = full;
  std::unique_ptr<ForwardDrizzleV2Kernel> kernel_holder =
      use_cuda ? std::unique_ptr<ForwardDrizzleV2Kernel>(
                     std::make_unique<ForwardDrizzleV2CudaKernel>())
               : std::unique_ptr<ForwardDrizzleV2Kernel>(
                     std::make_unique<ForwardDrizzleV2CpuKernel>());
  ForwardDrizzleV2Kernel &kernel = *kernel_holder;
  REQUIRE(kernel.reserve(nc, nr, f.plan.source_width, f.plan.source_height,
                         kcfg));
  const auto &m = f.plan.frames[0].source_to_canvas;
  const double a6[6] = {m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1), m(1, 2)};
  ForwardDrizzleV2FrameMeta meta{1.0f, 0.9f, 1, 0};
  std::vector<std::uint64_t> seq;
  std::size_t pilot_n = 0;
  if (full) {
    const auto pilot = forward_drizzle_v2_selected_frame_orders(
        n, kcfg.reservoir_size, kcfg.reservoir_seed);
    std::vector<char> is_pilot(static_cast<std::size_t>(n), 0);
    for (const auto o : pilot) {
      seq.push_back(o);
      is_pilot[static_cast<std::size_t>(o)] = 1;
    }
    pilot_n = seq.size();
    for (std::uint64_t o = 0; o < n; ++o)
      if (!is_pilot[static_cast<std::size_t>(o)]) seq.push_back(o);
  } else {
    for (std::uint64_t o = 0; o < n; ++o) seq.push_back(o);
  }
  Matrix2Df img = f.images[0];
  const Matrix2Df base = f.images[0];
  for (std::size_t pos = 0; pos < seq.size(); ++pos) {
    const std::uint64_t fr = seq[pos];
    for (int y = 0; y < img.rows(); ++y)
      for (int x = 0; x < img.cols(); ++x) {
        if (!std::isfinite(base(y, x))) {
          img(y, x) = base(y, x);
          continue;
        }
        double v = static_cast<double>(base(y, x)) +
                   sigma * ff_noise(fr + noise_seed * 1000003ULL, x, y);
        if (static_cast<int>(fr % 10) < outlier_tenths) v += outlier_shift;
        img(y, x) = static_cast<float>(v);
      }
    REQUIRE(kernel.accumulate_frame(a6, img.data(), nullptr, fr, nullptr,
                                    &meta));
    if (full && pos + 1 == pilot_n) REQUIRE(kernel.end_pilot());
  }
  FullFrameRun out;
  out.results.resize(nplane * 3);
  out.profiles.resize(nplane * 3);
  std::uint64_t dense = 0;
  REQUIRE(kernel.finalize(out.results.data(), out.profiles.data(), &dense));
  out.stats = kernel.stats();
  return out;
}

constexpr std::uint8_t kStatePilotFull = static_cast<std::uint8_t>(
    ForwardDrizzleV2RobustState::primary_reservoir_pilot_full_frame);

}  // namespace

TEST_CASE("forward drizzle v2 full-frame estimator equals the reservoir value "
          "when every frame is a pilot frame",
          "[forward-drizzle-v2][full-frame]") {
  // N <= reservoir size keeps every frame in the reservoir, so the pilot IS
  // the full stream: value and the four profile outputs must be bit-equal
  // to the historical path (same (x, order) summation order).
  const auto f = sfr_fixture(ColorMode::OSC, /*outlier_frame=*/5);
  const auto ref = run_full_frame(f, 12, 1.0, 0, 0.0, false);
  const auto ff = run_full_frame(f, 12, 1.0, 0, 0.0, true);
  REQUIRE(ref.results.size() == ff.results.size());
  int compared = 0;
  for (std::size_t i = 0; i < ref.results.size(); ++i) {
    const auto &a = ref.results[i];
    const auto &b = ff.results[i];
    if (a.robust_state !=
        static_cast<std::uint8_t>(
            ForwardDrizzleV2RobustState::primary_reservoir_sigma_clip))
      continue;
    REQUIRE(b.robust_state == kStatePilotFull);
    REQUIRE(a.value == b.value);
    require_profile_output_equal(ref.profiles[i].uniform,
                                 ff.profiles[i].uniform);
    require_profile_output_equal(ref.profiles[i].raw, ff.profiles[i].raw);
    require_profile_output_equal(ref.profiles[i].fine, ff.profiles[i].fine);
    require_profile_output_equal(ref.profiles[i].medium,
                                 ff.profiles[i].medium);
    ++compared;
  }
  REQUIRE(compared > 0);
  REQUIRE(ff.stats.full_frame_rejected == 0);
}

TEST_CASE("forward drizzle v2 full-frame estimator lowers the noise floor "
          "beyond the reservoir at N=610-like streams",
          "[forward-drizzle-v2][full-frame]") {
  // 200 frames, pure noise, no contamination. The reservoir keeps ~64 of them
  // (the keep set is bounded by the reservoir size), the full-frame value
  // uses every accepted frame: its error against the noise-free value must
  // be at least 1.5x smaller (theory: sqrt(200/64) = 1.77 before the clip's
  // own efficiency loss).
  const auto f = sfr_fixture(ColorMode::OSC, /*outlier_frame=*/5);
  const std::uint64_t n = 200;
  // Two independent noise realisations per estimator. The clip's upward bias
  // is a property of the pixel's scatter, identical in both realisations, so
  // it cancels in the difference; what remains is the estimator noise.
  const auto res_a = run_full_frame(f, n, 3.0, 0, 0.0, false, 1);
  const auto res_b = run_full_frame(f, n, 3.0, 0, 0.0, false, 2);
  const auto ff_a = run_full_frame(f, n, 3.0, 0, 0.0, true, 1);
  const auto ff_b = run_full_frame(f, n, 3.0, 0, 0.0, true, 2);
  double q_res = 0.0, q_ff = 0.0;
  int used = 0;
  for (std::size_t i = 0; i < res_a.results.size(); ++i) {
    if (ff_a.results[i].robust_state != kStatePilotFull ||
        ff_b.results[i].robust_state != kStatePilotFull)
      continue;
    const double dr = res_a.results[i].value - res_b.results[i].value;
    const double df = ff_a.results[i].value - ff_b.results[i].value;
    q_res += dr * dr;
    q_ff += df * df;
    ++used;
  }
  REQUIRE(used > 5);
  REQUIRE(ff_a.stats.full_frame_accepted > 0);
  const double sd_res = std::sqrt(q_res / used), sd_ff = std::sqrt(q_ff / used);
  double neff_ff = 0.0, neff_res = 0.0;
  int nn = 0;
  for (std::size_t i = 0; i < ff_a.results.size(); ++i) {
    if (ff_a.results[i].robust_state != kStatePilotFull) continue;
    neff_ff += ff_a.profiles[i].uniform.n_eff;
    neff_res += res_a.profiles[i].uniform.n_eff;
    ++nn;
  }
  INFO("noise reservoir " << sd_res << " noise full-frame " << sd_ff
       << " mean uniform n_eff res " << neff_res / nn << " ff " << neff_ff / nn
       << " accepted " << ff_a.stats.full_frame_accepted << " rejected "
       << ff_a.stats.full_frame_rejected << " no_bounds "
       << ff_a.stats.full_frame_no_bounds << " degenerate "
       << ff_a.stats.full_frame_degenerate_pilot);
  REQUIRE(sd_ff * 1.5 < sd_res);
}

TEST_CASE("forward drizzle v2 full-frame estimator keeps the contamination "
          "protection of the clip",
          "[forward-drizzle-v2][full-frame]") {
  // 30% of the frames (periodic, so also inside the pilot) carry a +8 shift
  // on top of unit noise, the gate-3 contamination shape. A plain mean would
  // be biased by ~+2.4; the frozen pilot bounds must keep the mean error
  // well below that.
  const auto f = sfr_fixture(ColorMode::OSC, /*outlier_frame=*/5);
  const std::uint64_t n = 200;
  const auto truth = run_full_frame(f, n, 0.0, 0, 0.0, false);
  const auto ff = run_full_frame(f, n, 1.0, 3, 8.0, true);
  double bias = 0.0;
  int used = 0;
  for (std::size_t i = 0; i < truth.results.size(); ++i) {
    if (ff.results[i].robust_state != kStatePilotFull) continue;
    bias += ff.results[i].value - truth.results[i].value;
    ++used;
  }
  REQUIRE(used > 5);
  INFO("mean bias " << bias / used);
  REQUIRE(std::fabs(bias / used) < 0.6);
  REQUIRE(ff.stats.full_frame_rejected > 0);
}

TEST_CASE("forward drizzle v2 full-frame estimator enforces its stream "
          "contract",
          "[forward-drizzle-v2][full-frame]") {
  const auto f = sfr_fixture(ColorMode::OSC, /*outlier_frame=*/5);
  const int nc = f.plan.canvas_width_native, nr = f.plan.canvas_height_native;
  const std::uint64_t n = 200;
  ForwardDrizzleV2KernelConfig kcfg;
  kcfg.internal_scale = f.cfg.internal_scale;
  kcfg.stream_length = n;
  kcfg.half = 0.5 * f.cfg.pixfrac;
  kcfg.bayer_pattern = static_cast<int>(f.plan.bayer_pattern);
  kcfg.cfa_origin_x = f.plan.cfa_origin_x;
  kcfg.cfa_origin_y = f.plan.cfa_origin_y;
  kcfg.emit_profiles = true;
  kcfg.full_frame_estimator = true;
  {
    auto bad = kcfg;
    bad.emit_profiles = false;  // profile weights need the Q/meta streams
    ForwardDrizzleV2CpuKernel k;
    REQUIRE_FALSE(k.reserve(nc, nr, f.plan.source_width, f.plan.source_height,
                            bad));
  }
  ForwardDrizzleV2CpuKernel k;
  REQUIRE(k.reserve(nc, nr, f.plan.source_width, f.plan.source_height, kcfg));
  const auto pilot = forward_drizzle_v2_selected_frame_orders(
      n, kcfg.reservoir_size, kcfg.reservoir_seed);
  std::vector<char> is_pilot(static_cast<std::size_t>(n), 0);
  for (const auto o : pilot) is_pilot[static_cast<std::size_t>(o)] = 1;
  std::uint64_t non_pilot = 0, pilot0 = pilot.front();
  while (is_pilot[static_cast<std::size_t>(non_pilot)]) ++non_pilot;
  const auto &m = f.plan.frames[0].source_to_canvas;
  const double a6[6] = {m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1), m(1, 2)};
  ForwardDrizzleV2FrameMeta meta{1.0f, 0.9f, 1, 0};
  // A non-pilot frame before the barrier and a pilot frame after it fail.
  REQUIRE_FALSE(k.accumulate_frame(a6, f.images[0].data(), nullptr, non_pilot,
                                   nullptr, &meta));
  std::vector<ForwardDrizzleV2PixelResult> res(
      static_cast<std::size_t>(nc) * nr * 3);
  std::vector<ForwardDrizzleV2ProfileResult> prof(res.size());
  std::uint64_t dense = 0;
  REQUIRE_FALSE(k.finalize(res.data(), prof.data(), &dense));  // no barrier
  REQUIRE(k.accumulate_frame(a6, f.images[0].data(), nullptr, pilot0, nullptr,
                             &meta));
  REQUIRE(k.end_pilot());
  REQUIRE_FALSE(k.end_pilot());  // once per band
  REQUIRE_FALSE(k.accumulate_frame(a6, f.images[0].data(), nullptr,
                                   pilot.back(), nullptr, &meta));
  REQUIRE(k.accumulate_frame(a6, f.images[0].data(), nullptr, non_pilot,
                             nullptr, &meta));
}

TEST_CASE("forward drizzle v2 full-frame estimator CUDA matches the CPU "
          "oracle",
          "[forward-drizzle-v2][full-frame][cuda-parity]") {
  if (!forward_drizzle_cuda_runtime_available()) {
    SUCCEED("CUDA device unavailable");
    return;
  }
  // Discrete decisions (states, accepted/rejected counts) must be identical;
  // values may differ in the last ULPs (pre-existing CPU<->CUDA gap of the
  // reservoir pair on this fixture shape, see the shared_frame_rejection
  // parity test), so they are compared with a tight relative tolerance.
  const auto f = sfr_fixture(ColorMode::OSC, /*outlier_frame=*/5);
  struct Case { std::uint64_t n; double sigma; int tenths; double shift; };
  for (const Case c : {Case{12, 1.0, 0, 0.0}, Case{200, 1.0, 3, 8.0},
                       Case{200, 3.0, 0, 0.0}}) {
    const auto cpu = run_full_frame(f, c.n, c.sigma, c.tenths, c.shift, true,
                                    0, false);
    const auto gpu = run_full_frame(f, c.n, c.sigma, c.tenths, c.shift, true,
                                    0, true);
    REQUIRE(cpu.results.size() == gpu.results.size());
    int full_px = 0;
    for (std::size_t i = 0; i < cpu.results.size(); ++i) {
      const auto &a = cpu.results[i];
      const auto &b = gpu.results[i];
      REQUIRE(a.robust_state == b.robust_state);
      REQUIRE(a.contributors == b.contributors);
      if (a.robust_state == kStatePilotFull) ++full_px;
      if (std::isfinite(a.value))
        REQUIRE(std::fabs(a.value - b.value) <=
                1e-9 * (1.0 + std::fabs(a.value)));
      for (int k = 0; k < 4; ++k) {
        const ForwardDrizzleV2ProfileOutput *pa[4] = {
            &cpu.profiles[i].uniform, &cpu.profiles[i].raw,
            &cpu.profiles[i].fine, &cpu.profiles[i].medium};
        const ForwardDrizzleV2ProfileOutput *pb[4] = {
            &gpu.profiles[i].uniform, &gpu.profiles[i].raw,
            &gpu.profiles[i].fine, &gpu.profiles[i].medium};
        REQUIRE(pa[k]->support == pb[k]->support);
        if (pa[k]->support)
          REQUIRE(std::fabs(pa[k]->value - pb[k]->value) <=
                  1e-5f * (1.0f + std::fabs(pa[k]->value)));
      }
    }
    REQUIRE(full_px > 5);
    REQUIRE(cpu.stats.full_frame_accepted == gpu.stats.full_frame_accepted);
    REQUIRE(cpu.stats.full_frame_rejected == gpu.stats.full_frame_rejected);
    REQUIRE(cpu.stats.full_frame_degenerate_pilot ==
            gpu.stats.full_frame_degenerate_pilot);
    REQUIRE(cpu.stats.full_frame_no_bounds == gpu.stats.full_frame_no_bounds);
  }
}
