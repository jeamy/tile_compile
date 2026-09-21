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
    // The largest class amount sits at the cap (the faintest class, made of
    // marginal edge pixels, may legitimately get less).
    REQUIRE(*std::max_element(r.amounts.begin(), r.amounts.end()) ==
            Catch::Approx(0.3f).margin(1e-4));
    REQUIRE(r.amount <= 0.3f + 1e-4f);
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

namespace {

// Brightness-dependent cast: a bright core (r < 30) with a strong green excess
// (x1.25), surrounded by a faint ring (30 < r < 70) that is already neutral.
Img make_two_zone_image() {
  Img im;
  im.R = Matrix2Df::Constant(kN, kN, 0.12f);
  im.G = im.R;
  im.B = im.R;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 0.01f);
  for (int y = 0; y < kN; ++y) {
    for (int x = 0; x < kN; ++x) {
      const double d = std::hypot(y - 150.0, x - 150.0);
      double s = 0.0, gf = 1.0;
      if (d < 30.0) { s = 0.30; gf = 1.25; }
      else if (d < 70.0) { s = 0.05; gf = 1.0; }
      im.R(y, x) += static_cast<float>(s) + nd(rng);
      im.B(y, x) += static_cast<float>(s) + nd(rng);
      im.G(y, x) += static_cast<float>(s * gf) + nd(rng);
    }
  }
  return im;
}

double zone_ratio(const Img &im, int y0, int y1, int x0, int x1) {
  const double r = region_median(im.R, y0, y1, x0, x1) - 0.12;
  const double g = region_median(im.G, y0, y1, x0, x1) - 0.12;
  const double b = region_median(im.B, y0, y1, x0, x1) - 0.12;
  return g / (0.5 * (r + b));
}

}  // namespace

TEST_CASE("color cast correction is brightness dependent: bright core and "
          "neutral faint ring are both handled", "[color-cast]") {
  // Core box r < ~20, ring box well inside 30 < r < 70 (rows 100..115, cols 130..170).
  {
    Img im = make_two_zone_image();
    REQUIRE(zone_ratio(im, 135, 165, 135, 165) == Catch::Approx(1.25).margin(0.04));
    REQUIRE(zone_ratio(im, 95, 110, 130, 170) == Catch::Approx(1.0).margin(0.06));
    auto cfg = on();
    cfg.brightness_bins = 8;
    const auto r = apply_color_cast_correction(im.R, im.G, im.B, cfg, nullptr);
    REQUIRE(r.applied);
    REQUIRE(r.amounts.size() == 8);
    REQUIRE(zone_ratio(im, 135, 165, 135, 165) == Catch::Approx(1.0).margin(0.05));
    REQUIRE(zone_ratio(im, 95, 110, 130, 170) == Catch::Approx(1.0).margin(0.06));
    // Bright classes get a larger amount than the faint ones.
    REQUIRE(r.amounts.back() > r.amounts.front() + 0.2f);
  }
  {
    // One global amount cannot do both: it either leaves the core green or
    // pushes the neutral ring below neutral.
    Img im = make_two_zone_image();
    auto cfg = on();
    cfg.brightness_bins = 1;
    const auto r = apply_color_cast_correction(im.R, im.G, im.B, cfg, nullptr);
    REQUIRE(r.amounts.size() == 1);
    const double core = zone_ratio(im, 135, 165, 135, 165);
    const double ring = zone_ratio(im, 95, 110, 130, 170);
    REQUIRE((core > 1.05 || ring < 0.97));
  }
}

TEST_CASE("color cast correction can neutralize the sky", "[color-cast]") {
  // Neutral nebula on a slightly magenta sky (G sky 0.105 instead of 0.12).
  auto build = [] {
    Img im = make_image(1.0);
    for (int y = 0; y < kN; ++y)
      for (int x = 0; x < kN; ++x) im.G(y, x) -= 0.015f;
    return im;
  };
  {
    Img im = build();
    const double sky_before = region_median(im.G, 0, 60, 0, 60);
    REQUIRE(sky_before == Catch::Approx(0.105).margin(0.003));
    auto cfg = on();
    cfg.neutralize_sky = true;
    const auto r = apply_color_cast_correction(im.R, im.G, im.B, cfg, nullptr);
    REQUIRE(r.applied);
    REQUIRE(r.sky_neutralized);
    REQUIRE(r.sky_offset_g == Catch::Approx(0.015).margin(0.003));
    REQUIRE(r.status == "sky_neutralized");  // no cast excess to correct
    REQUIRE(region_median(im.G, 0, 60, 0, 60) == Catch::Approx(0.12).margin(0.003));
    // The whole G plane moves (nebula included), R and B do not.
    const Img ref = build();
    REQUIRE(im.R == ref.R);
    REQUIRE(im.B == ref.B);
  }
  {
    // Off by default: nothing changes although the sky is tinted.
    Img im = build();
    const Img before = im;
    const auto r = apply_color_cast_correction(im.R, im.G, im.B, on(), nullptr);
    REQUIRE_FALSE(r.sky_neutralized);
    REQUIRE(im.G == before.G);
  }
  {
    // Combined with a real cast: excess corrected and sky neutral.
    Img im = build();
    for (int y = 0; y < kN; ++y)
      for (int x = 0; x < kN; ++x) {
        const double d = std::hypot(y - 150.0, x - 150.0);
        if (d < 70.0) im.G(y, x) += 0.25f * 0.2f;
      }
    auto cfg = on();
    cfg.neutralize_sky = true;
    const auto r = apply_color_cast_correction(im.R, im.G, im.B, cfg, nullptr);
    REQUIRE(r.status == "applied");
    REQUIRE(r.sky_neutralized);
    REQUIRE(region_median(im.G, 0, 60, 0, 60) == Catch::Approx(0.12).margin(0.004));
    REQUIRE(nebula_ratio(im) == Catch::Approx(1.0).margin(0.04));
  }
  {
    // An implausibly large offset is not applied.
    Img im = make_image(1.0);
    for (int y = 0; y < kN; ++y)
      for (int x = 0; x < kN; ++x) im.G(y, x) -= 0.06f;  // 50 % of the sky level
    const Img before = im;
    auto cfg = on();
    cfg.neutralize_sky = true;
    const auto r = apply_color_cast_correction(im.R, im.G, im.B, cfg, nullptr);
    REQUIRE_FALSE(r.sky_neutralized);
  }
}
#endif

