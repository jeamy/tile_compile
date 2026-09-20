#if __has_include(<catch2/catch_test_macros.hpp>)
#include "tile_compile/image/color_cast_correction.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace tile_compile;
using image::ColorCastCorrectionConfig;
using image::apply_color_cast_correction;

namespace {

constexpr int kN = 300;

struct Img {
  Matrix2Df R, G, B;
};

// Sky 0.12 + noise; a disc "nebula" (radius 70) adds 0.25 to R and B and
// 0.25 * green_factor to G; a small saturated star sits in a corner.
Img make_image(double green_factor, double noise = 0.01) {
  Img im;
  im.R = Matrix2Df::Constant(kN, kN, 0.12f);
  im.G = im.R;
  im.B = im.R;
  std::mt19937 rng(42);
  std::normal_distribution<float> nd(0.0f, static_cast<float>(noise));
  for (int y = 0; y < kN; ++y) {
    for (int x = 0; x < kN; ++x) {
      const double d = std::hypot(y - 150.0, x - 150.0);
      const float neb = d < 70.0 ? 1.0f : 0.0f;
      im.R(y, x) += 0.25f * neb + nd(rng);
      im.B(y, x) += 0.25f * neb + nd(rng);
      im.G(y, x) += static_cast<float>(0.25 * green_factor) * neb + nd(rng);
    }
  }
  for (int y = 285; y < 289; ++y)
    for (int x = 285; x < 289; ++x) im.R(y, x) = im.G(y, x) = im.B(y, x) = 0.9f;
  return im;
}

double region_median(const Matrix2Df &m, int y0, int y1, int x0, int x1) {
  std::vector<float> v;
  for (int y = y0; y < y1; ++y)
    for (int x = x0; x < x1; ++x) v.push_back(m(y, x));
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];
}

// Green excess over the sky inside the nebula core.
double nebula_ratio(const Img &im) {
  const double r = region_median(im.R, 120, 180, 120, 180) - 0.12;
  const double g = region_median(im.G, 120, 180, 120, 180) - 0.12;
  const double b = region_median(im.B, 120, 180, 120, 180) - 0.12;
  return g / (0.5 * (r + b));
}

ColorCastCorrectionConfig on() {
  ColorCastCorrectionConfig c;
  c.enabled = true;
  return c;
}

}  // namespace

TEST_CASE("color cast correction pulls a green excess to the target ratio",
          "[color-cast]") {
  Img im = make_image(1.2);
  REQUIRE(nebula_ratio(im) == Catch::Approx(1.2).margin(0.03));
  const double sky_g_before = region_median(im.G, 0, 60, 0, 60);
  const auto r = apply_color_cast_correction(im.R, im.G, im.B, on(), nullptr);
  REQUIRE(r.applied);
  REQUIRE(r.status == "applied");
  REQUIRE(r.ratio_before == Catch::Approx(1.2).margin(0.04));
  REQUIRE(r.ratio_after == Catch::Approx(1.0).margin(0.01));
  REQUIRE(r.amount > 0.4f);
  REQUIRE(r.amount <= 1.0f);
  REQUIRE(nebula_ratio(im) == Catch::Approx(1.0).margin(0.03));
  // The sky keeps its level (only noise excursions above the sky are trimmed).
  REQUIRE(std::fabs(region_median(im.G, 0, 60, 0, 60) - sky_g_before) < 0.006);
  // R and B are never touched.
  const Img ref = make_image(1.2);
  REQUIRE(im.R == ref.R);
  REQUIRE(im.B == ref.B);
  // The saturated star is neutral and stays put.
  REQUIRE(im.G(286, 286) == Catch::Approx(0.9f).margin(0.02));
}

TEST_CASE("color cast correction leaves a neutral image untouched",
          "[color-cast]") {
  Img im = make_image(1.0);
  const Img before = im;
  const auto r = apply_color_cast_correction(im.R, im.G, im.B, on(), nullptr);
  REQUIRE_FALSE(r.applied);
  REQUIRE(r.status == "not_needed");
  REQUIRE(im.G == before.G);
}

TEST_CASE("color cast correction protects a genuinely green object via "
          "min_excess", "[color-cast]") {
  Img im = make_image(1.2);
  const Img before = im;
  auto cfg = on();
  cfg.min_excess = 1.3f;  // measured 1.2 <= 1.3: leave it alone
  const auto r = apply_color_cast_correction(im.R, im.G, im.B, cfg, nullptr);
  REQUIRE_FALSE(r.applied);
  REQUIRE(r.status == "not_needed");
  REQUIRE(im.G == before.G);
}

TEST_CASE("color cast correction respects max_amount and target_ratio",
          "[color-cast]") {
  {
    Img im = make_image(1.2);
    auto cfg = on();
    cfg.max_amount = 0.3f;
    const auto r = apply_color_cast_correction(im.R, im.G, im.B, cfg, nullptr);
    REQUIRE(r.applied);
    REQUIRE(r.amount == Catch::Approx(0.3f).margin(1e-4));
    REQUIRE(r.ratio_after > 1.05);  // capped before reaching the target
  }
  {
    Img im = make_image(1.2);
    auto cfg = on();
    cfg.target_ratio = 1.1f;
    const auto r = apply_color_cast_correction(im.R, im.G, im.B, cfg, nullptr);
    REQUIRE(r.applied);
    REQUIRE(r.ratio_after == Catch::Approx(1.1).margin(0.01));
  }
}

TEST_CASE("color cast correction honours the valid mask and disabled state",
          "[color-cast]") {
  {
    Img im = make_image(1.2);
    const Img before = im;
    ColorCastCorrectionConfig cfg;  // enabled = false
    const auto r = apply_color_cast_correction(im.R, im.G, im.B, cfg, nullptr);
    REQUIRE(r.status == "disabled");
    REQUIRE(im.G == before.G);
  }
  {
    Img im = make_image(1.2);
    const Img before = im;
    std::vector<std::uint8_t> mask(static_cast<std::size_t>(kN) * kN, 1);
    for (int y = 0; y < 40; ++y)
      for (int x = 0; x < kN; ++x) mask[static_cast<std::size_t>(y) * kN + x] = 0;
    const auto r = apply_color_cast_correction(im.R, im.G, im.B, on(), &mask);
    REQUIRE(r.applied);
    for (int y = 0; y < 40; ++y)
      for (int x = 0; x < kN; ++x) REQUIRE(im.G(y, x) == before.G(y, x));
  }
  {
    Img im = make_image(1.2);
    Matrix2Df small = Matrix2Df::Constant(10, 10, 0.1f);
    const auto r = apply_color_cast_correction(im.R, small, im.B, on(), nullptr);
    REQUIRE(r.status == "error");
  }
}
#endif
