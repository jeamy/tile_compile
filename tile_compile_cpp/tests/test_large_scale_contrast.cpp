#if __has_include(<catch2/catch_test_macros.hpp>)
#include "tile_compile/image/large_scale_contrast.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace tile_compile;
using image::LargeScaleContrastConfig;
using image::apply_large_scale_contrast;

namespace {

constexpr int kN = 400;

struct Img {
  Matrix2Df R, G, B;
};

// Sky 0.12 + Gaussian noise; a broad nebula blob (sigma 60 px, amplitude `blob` on all channels, `blob_r_only` extra on R);
// a radial vignette (centre bright, corners `vignette` darker); one small bright star far from the blob.
Img make_image(double blob, double noise, double vignette = 0.0, double blob_r_only = 0.0) {
  Img im;
  im.R = Matrix2Df::Constant(kN, kN, 0.12f);
  im.G = im.R;
  im.B = im.R;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, static_cast<float>(noise));
  const double rmax2 = 2.0 * 200.0 * 200.0;
  for (int y = 0; y < kN; ++y)
    for (int x = 0; x < kN; ++x) {
      const double r2 = (y - 200.0) * (y - 200.0) + (x - 200.0) * (x - 200.0);
      const double v = -vignette * (r2 / rmax2);
      const double b = std::exp(-((y - 150.0) * (y - 150.0) + (x - 230.0) * (x - 230.0)) / (2.0 * 60.0 * 60.0));
      const double s = 0.6 * std::exp(-((y - 340.0) * (y - 340.0) + (x - 60.0) * (x - 60.0)) / (2.0 * 1.5 * 1.5));
      im.R(y, x) += static_cast<float>(v + blob * b + blob_r_only * b + s) + (noise > 0 ? nd(rng) : 0.0f);
      im.G(y, x) += static_cast<float>(v + blob * b + s) + (noise > 0 ? nd(rng) : 0.0f);
      im.B(y, x) += static_cast<float>(v + blob * b + s) + (noise > 0 ? nd(rng) : 0.0f);
    }
  return im;
}

double region_mean(const Matrix2Df &m, int y0, int y1, int x0, int x1) {
  double acc = 0;
  for (int y = y0; y < y1; ++y)
    for (int x = x0; x < x1; ++x) acc += m(y, x);
  return acc / ((y1 - y0) * (x1 - x0));
}

// Pixel-to-pixel noise: standard deviation of the difference to the right neighbour, over a sky patch.
double pixel_noise(const Matrix2Df &m, int y0, int y1, int x0, int x1) {
  double acc = 0, acc2 = 0;
  int n = 0;
  for (int y = y0; y < y1; ++y)
    for (int x = x0; x < x1 - 1; ++x) {
      const double d = m(y, x + 1) - m(y, x);
      acc += d;
      acc2 += d * d;
      ++n;
    }
  const double mean = acc / n;
  return std::sqrt(acc2 / n - mean * mean);
}

double max_abs_diff(const Matrix2Df &a, const Matrix2Df &b) { return (a - b).cwiseAbs().maxCoeff(); }

LargeScaleContrastConfig on(float amount = 2.0f) {
  LargeScaleContrastConfig c;
  c.enabled = true;
  c.amount = amount;
  return c;
}

}  // namespace

TEST_CASE("large-scale contrast: disabled and zero amounts change nothing", "[large-scale-contrast]") {
  Img im = make_image(0.03, 0.004);
  const Img ref = im;
  LargeScaleContrastConfig off;  // enabled = false
  auto r = apply_large_scale_contrast(im.R, im.G, im.B, off, nullptr);
  REQUIRE(r.status == "disabled");
  REQUIRE_FALSE(r.applied);
  REQUIRE(max_abs_diff(im.R, ref.R) == 0.0f);
  LargeScaleContrastConfig zero = on(0.0f);
  r = apply_large_scale_contrast(im.R, im.G, im.B, zero, nullptr);
  REQUIRE(r.status == "not_needed");
  REQUIRE_FALSE(r.applied);
  REQUIRE(max_abs_diff(im.G, ref.G) == 0.0f);
}

TEST_CASE("large-scale contrast lifts the broad structure and leaves noise and stars alone", "[large-scale-contrast]") {
  Img im = make_image(0.03, 0.004);
  const Img ref = im;
  const double sky_before = region_mean(ref.G, 300, 380, 300, 380);
  const double blob_before = region_mean(ref.G, 140, 160, 220, 240) - sky_before;
  const double noise_before = pixel_noise(ref.G, 300, 380, 200, 380);
  const float star_before = ref.G(340, 60);
  LargeScaleContrastConfig cfg = on(2.0f);
  cfg.remove_vignette = false;
  const auto r = apply_large_scale_contrast(im.R, im.G, im.B, cfg, nullptr);
  REQUIRE(r.applied);
  REQUIRE(r.status == "applied");
  const double blob_after = region_mean(im.G, 140, 160, 220, 240) - region_mean(im.G, 300, 380, 300, 380);
  // The large-scale map of a sigma-60 blob smoothed with sigma 48 carries about 60 % of its amplitude; amount 2 adds twice that.
  REQUIRE(blob_after / blob_before > 1.9);
  REQUIRE(blob_after / blob_before < 2.7);
  REQUIRE(pixel_noise(im.G, 300, 380, 200, 380) / noise_before == Catch::Approx(1.0).margin(0.02));
  REQUIRE(std::fabs(im.G(340, 60) - star_before) < 0.01f);  // a star far from the structure is not brightened
  REQUIRE(r.span_after > 1.5 * r.span_before);
  REQUIRE(r.sky_reference == Catch::Approx(0.12).margin(0.02));
}

TEST_CASE("large-scale contrast: a flat image and pixels outside the mask are left untouched", "[large-scale-contrast]") {
  Img flat = make_image(0.0, 0.0);
  for (int y = 0; y < kN; ++y)
    for (int x = 0; x < kN; ++x) flat.R(y, x) = flat.G(y, x) = flat.B(y, x) = 0.2f;
  const Img ref = flat;
  auto r = apply_large_scale_contrast(flat.R, flat.G, flat.B, on(3.0f), nullptr);
  REQUIRE(r.applied);
  REQUIRE(max_abs_diff(flat.G, ref.G) < 1e-6f);

  // A quarter of the frame is invalid (zero); the valid part carries a blob. The invalid part must stay exactly zero,
  // and a flat valid area next to the border must not get a halo from the fill value.
  Img im = make_image(0.03, 0.0);
  std::vector<std::uint8_t> mask(static_cast<std::size_t>(kN) * kN, 1);
  for (int y = 0; y < kN; ++y)
    for (int x = 0; x < 100; ++x) {
      mask[static_cast<std::size_t>(y) * kN + x] = 0;
      im.R(y, x) = im.G(y, x) = im.B(y, x) = 0.0f;
    }
  const Img before = im;
  r = apply_large_scale_contrast(im.R, im.G, im.B, on(2.0f), &mask);
  REQUIRE(r.applied);
  for (int y = 0; y < kN; ++y)
    for (int x = 0; x < 100; ++x) REQUIRE(im.G(y, x) == 0.0f);
  // near the border, far from the blob (blob is at x=230): the sky level is unchanged within a small tolerance
  REQUIRE(std::fabs(region_mean(im.G, 300, 380, 110, 150) - region_mean(before.G, 300, 380, 110, 150)) < 0.004);
}

TEST_CASE("large-scale contrast: a vignette is not amplified when remove_vignette is on", "[large-scale-contrast]") {
  const Img ref = make_image(0.03, 0.0, 0.04);
  auto corner_to_centre = [](const Img &im) {
    return region_mean(im.G, 190, 210, 190, 210) - region_mean(im.G, 350, 395, 350, 395);
  };
  const double vignette_before = corner_to_centre(ref);
  REQUIRE(vignette_before > 0.01);
  Img keep = ref;
  LargeScaleContrastConfig cfg = on(3.0f);
  cfg.remove_vignette = true;
  REQUIRE(apply_large_scale_contrast(keep.R, keep.G, keep.B, cfg, nullptr).applied);
  Img amplified = ref;
  cfg.remove_vignette = false;
  REQUIRE(apply_large_scale_contrast(amplified.R, amplified.G, amplified.B, cfg, nullptr).applied);
  const double with_removal = corner_to_centre(keep) / vignette_before;
  const double without_removal = corner_to_centre(amplified) / vignette_before;
  REQUIRE(without_removal > 2.0);            // amplified about (1 + amount)
  REQUIRE(with_removal < 1.5);               // mostly left alone
  REQUIRE(with_removal < 0.6 * without_removal);
  // the off-centre nebula is still lifted with the removal on
  const double blob_ref = region_mean(ref.G, 140, 160, 220, 240) - region_mean(ref.G, 130, 170, 120, 140);
  const double blob_keep = region_mean(keep.G, 140, 160, 220, 240) - region_mean(keep.G, 130, 170, 120, 140);
  REQUIRE(blob_keep > 1.5 * blob_ref);
}

TEST_CASE("large-scale contrast: chroma amount changes colour structure only", "[large-scale-contrast]") {
  Img im = make_image(0.0, 0.0, 0.0, 0.03);  // blob only in R: an R-G colour blob
  const Img ref = im;
  LargeScaleContrastConfig cfg = on(0.0f);
  cfg.chroma_amount = 2.0f;
  cfg.remove_vignette = false;
  const auto r = apply_large_scale_contrast(im.R, im.G, im.B, cfg, nullptr);
  REQUIRE(r.applied);
  REQUIRE(max_abs_diff(im.G, ref.G) == 0.0f);  // G is the reference channel and never moves
  REQUIRE(max_abs_diff(im.B, ref.B) < 1e-6f);  // there is no B-G structure in this image
  const double rg_before = region_mean(ref.R, 140, 160, 220, 240) - region_mean(ref.G, 140, 160, 220, 240);
  const double rg_after = region_mean(im.R, 140, 160, 220, 240) - region_mean(im.G, 140, 160, 220, 240);
  REQUIRE(rg_after / rg_before > 1.9);
}

TEST_CASE("large-scale contrast keeps every channel inside its own input range", "[large-scale-contrast]") {
  Img im = make_image(-0.1, 0.01);  // a dark dip instead of a bump; large amount tries to push below zero
  float max_g = 0;
  for (int y = 0; y < kN; ++y)
    for (int x = 0; x < kN; ++x) max_g = std::max(max_g, im.G(y, x));
  const auto r = apply_large_scale_contrast(im.R, im.G, im.B, on(6.0f), nullptr);
  REQUIRE(r.applied);
  REQUIRE(im.G.minCoeff() >= 0.0f);
  REQUIRE(im.G.maxCoeff() <= max_g);
}

TEST_CASE("large-scale contrast is deterministic and reports the grid it used", "[large-scale-contrast]") {
  Img a = make_image(0.03, 0.004), b = a;
  const auto ra = apply_large_scale_contrast(a.R, a.G, a.B, on(), nullptr);
  const auto rb = apply_large_scale_contrast(b.R, b.G, b.B, on(), nullptr);
  REQUIRE(max_abs_diff(a.G, b.G) == 0.0f);
  REQUIRE(ra.downsample_factor == 8);
  REQUIRE(rb.span_before == ra.span_before);
  for (auto [sigma, factor] : {std::pair<float, int>{6.0f, 1}, {48.0f, 8}, {512.0f, 16}}) {
    Img c = make_image(0.03, 0.004);
    LargeScaleContrastConfig cfg = on();
    cfg.sigma_px = sigma;
    REQUIRE(apply_large_scale_contrast(c.R, c.G, c.B, cfg, nullptr).downsample_factor == factor);
  }
}

TEST_CASE("large-scale contrast refuses unusable input", "[large-scale-contrast]") {
  Img im = make_image(0.03, 0.004);
  Matrix2Df small = Matrix2Df::Constant(16, 16, 0.1f);
  auto r = apply_large_scale_contrast(small, small, small, on(), nullptr);
  REQUIRE(r.status == "error");
  Matrix2Df other = Matrix2Df::Constant(kN, kN - 1, 0.1f);
  r = apply_large_scale_contrast(im.R, im.G, other, on(), nullptr);
  REQUIRE(r.status == "error");
  r = apply_large_scale_contrast(im.R, im.G, im.B, on(-1.0f), nullptr);
  REQUIRE(r.status == "error");
  std::vector<std::uint8_t> wrong(10, 1);
  r = apply_large_scale_contrast(im.R, im.G, im.B, on(), &wrong);
  REQUIRE(r.status == "error");
  std::vector<std::uint8_t> nearly_empty(static_cast<std::size_t>(kN) * kN, 0);
  for (int i = 0; i < 200; ++i) nearly_empty[i] = 1;
  const Img before = im;
  r = apply_large_scale_contrast(im.R, im.G, im.B, on(), &nearly_empty);
  REQUIRE(r.status == "too_few_valid_pixels");
  REQUIRE(max_abs_diff(im.G, before.G) == 0.0f);
}

#endif  // __has_include(<catch2/catch_test_macros.hpp>)
