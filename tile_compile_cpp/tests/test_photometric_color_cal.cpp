#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "tile_compile/astrometry/detail/photometric_color_cal_detail.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using tile_compile::Matrix2Df;
using tile_compile::astrometry::ColorMatrix;

namespace {

float median(Matrix2Df channel) {
  std::vector<float> values(channel.data(), channel.data() + channel.size());
  const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
  std::nth_element(values.begin(), middle, values.end());
  return *middle;
}

struct TestImage {
  Matrix2Df R = Matrix2Df(64, 64);
  Matrix2Df G = Matrix2Df(64, 64);
  Matrix2Df B = Matrix2Df(64, 64);
  std::vector<uint8_t> mask = std::vector<uint8_t>(64 * 64, 1);

  TestImage() {
    for (int y = 0; y < 64; ++y) {
      for (int x = 0; x < 64; ++x) {
        const float noise = 0.01f * static_cast<float>((x + 3 * y) % 7 - 3);
        R(y, x) = 24.0f + noise;
        G(y, x) = 25.0f + noise;
        B(y, x) = 17.0f + noise;
      }
    }
    R(32, 32) += 10.0f;
    G(32, 32) += 10.0f;
    B(32, 32) += 10.0f;
  }
};

const ColorMatrix kDiagonalMatrix = {{{1.2, 0.0, 0.0},
                                      {0.0, 1.0, 0.0},
                                      {0.0, 0.0, 1.6}}};

} // namespace

TEST_CASE("diagonal PCC keeps channel backgrounds when neutralization is off") {
  TestImage image;

  tile_compile::astrometry::detail::apply_diagonal_color_correction(
      image.R, image.G, image.B, kDiagonalMatrix, 0.5, "off", image.mask);

  REQUIRE(median(image.R) == Catch::Approx(24.0f).margin(0.02f));
  REQUIRE(median(image.G) == Catch::Approx(25.0f).margin(0.02f));
  REQUIRE(median(image.B) == Catch::Approx(17.0f).margin(0.02f));
  REQUIRE(image.R(32, 32) - 24.0f == Catch::Approx(11.0f).margin(0.04f));
  REQUIRE(image.G(32, 32) - 25.0f == Catch::Approx(10.0f).margin(0.04f));
  REQUIRE(image.B(32, 32) - 17.0f == Catch::Approx(13.0f).margin(0.04f));
}

TEST_CASE("diagonal PCC always neutralizes independently of chroma strength") {
  TestImage image;

  tile_compile::astrometry::detail::apply_diagonal_color_correction(
      image.R, image.G, image.B, kDiagonalMatrix, 0.5, "always", image.mask);

  const float expected_background = (24.0f + 25.0f + 17.0f) / 3.0f;
  REQUIRE(median(image.R) == Catch::Approx(expected_background).margin(0.02f));
  REQUIRE(median(image.G) == Catch::Approx(expected_background).margin(0.02f));
  REQUIRE(median(image.B) == Catch::Approx(expected_background).margin(0.02f));
  REQUIRE(image.R(32, 32) - expected_background ==
          Catch::Approx(11.0f).margin(0.04f));
  REQUIRE(image.G(32, 32) - expected_background ==
          Catch::Approx(10.0f).margin(0.04f));
  REQUIRE(image.B(32, 32) - expected_background ==
          Catch::Approx(13.0f).margin(0.04f));
}

TEST_CASE("diagonal PCC auto neutralizes stable low-chroma background") {
  TestImage image;

  tile_compile::astrometry::detail::apply_diagonal_color_correction(
      image.R, image.G, image.B, kDiagonalMatrix, 0.5, "auto", image.mask);

  const float expected_background = (24.0f + 25.0f + 17.0f) / 3.0f;
  REQUIRE(median(image.R) == Catch::Approx(expected_background).margin(0.02f));
  REQUIRE(median(image.G) == Catch::Approx(expected_background).margin(0.02f));
  REQUIRE(median(image.B) == Catch::Approx(expected_background).margin(0.02f));
}

TEST_CASE("diagonal PCC auto neutralizes realistic noisy background with green color cast") {
  Matrix2Df R(128, 128);
  Matrix2Df G(128, 128);
  Matrix2Df B(128, 128);
  std::vector<uint8_t> mask(128 * 128, 1);

  // Realistic OSC noisy background: G=15.4, R=8.2, B=6.0 with ~0.6 ADU noise
  for (int y = 0; y < 128; ++y) {
    for (int x = 0; x < 128; ++x) {
      const float noise = 0.35f * static_cast<float>((x * 7 + y * 13) % 11 - 5) / 5.0f;
      R(y, x) = 8.2f + noise;
      G(y, x) = 15.4f + noise;
      B(y, x) = 6.0f + noise;
    }
  }

  tile_compile::astrometry::detail::apply_diagonal_color_correction(
      R, G, B, kDiagonalMatrix, 0.85, "auto", mask);

  const float expected_bg = (8.2f + 15.4f + 6.0f) / 3.0f;
  REQUIRE(median(R) == Catch::Approx(expected_bg).margin(0.15f));
  REQUIRE(median(G) == Catch::Approx(expected_bg).margin(0.15f));
  REQUIRE(median(B) == Catch::Approx(expected_bg).margin(0.15f));
}
