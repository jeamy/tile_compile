// M4 tests: internal 2x raster -> output geometry (plan section 12).

#include "tile_compile/reconstruction/output_scale.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace tile_compile;
using namespace tile_compile::reconstruction;
using Catch::Approx;

TEST_CASE("output_scale mode: 2/1 needs a downsample; 1/1 and 2/2 do not") {
  REQUIRE(OutputScaleMode{2, 1}.needs_2x2_downsample());
  REQUIRE_FALSE(OutputScaleMode{1, 1}.needs_2x2_downsample());
  REQUIRE_FALSE(OutputScaleMode{2, 2}.needs_2x2_downsample());
  REQUIRE(OutputScaleMode{2, 1}.valid());
  REQUIRE_FALSE(OutputScaleMode{1, 2}.valid());  // output > internal
  REQUIRE_FALSE(OutputScaleMode{3, 1}.valid());
}


// --- plan 12.4: kernel-induced noise correlation --------------------------

TEST_CASE("kernel noise: pixfrac*internal_scale == 1 gives no correlation "
          "(factor exactly 1, plan 12.4)") {
  REQUIRE(kernel_noise_correlation_sigma_factor(1.0f, 1) == Approx(1.0).epsilon(1e-9));
  REQUIRE(kernel_noise_correlation_sigma_factor(0.5f, 2) == Approx(1.0).epsilon(1e-9));
  auto rho = kernel_noise_autocorrelation_1d(1.0f, 1, 4);
  REQUIRE(rho[0] == Approx(1.0));
  for (size_t i = 1; i < rho.size(); ++i) REQUIRE(rho[i] == Approx(0.0).margin(1e-12));
}

TEST_CASE("kernel noise: pixfrac=1, internal_scale=2 (d=2) matches the "
          "hand-computed value W/sqrt(S0) = 2/sqrt(1.5) (plan 12.4)") {
  // d = 2: overlaps of [j-0.5, j+1.5] with [0,1] over j = -1,0,1 are
  // 0.5, 1.0, 0.5 -> S0 = 0.25 + 1 + 0.25 = 1.5; W = d = 2.
  REQUIRE(kernel_noise_correlation_sigma_factor(1.0f, 2) ==
          Approx(2.0 / std::sqrt(1.5)).epsilon(1e-9));
  auto rho = kernel_noise_autocorrelation_1d(1.0f, 2, 3);
  // S_1 = sum over j of K(0,j+0.5)*K(1,j+0.5):
  //   j=-1: 0.5 * (overlap of [-1.5,0.5] shifted by -1 = [-0.5,1.5]&[0,1]=1.0) ... compute:
  //   K(0, x0) uses x0=j+0.5; K(1, x0) = kernel_overlap(x0 - 1, d).
  //   j=-1: K0=overlap([-1.5,0.5])=0.5 ; K1=overlap([-2.5,-0.5])=0
  //   j=0 : K0=overlap([-0.5,1.5])=1.0 ; K1=overlap([-1.5,0.5])=0.5
  //   j=1 : K0=overlap([0.5,2.5])=0.5  ; K1=overlap([-0.5,1.5])=1.0
  //   j=2 : K0=0                        ; K1=overlap([0.5,2.5])=0.5
  //   S_1 = 0 + 0.5 + 0.5 + 0 = 1.0 -> rho_1 = 1.0 / 1.5
  REQUIRE(rho[1] == Approx(1.0 / 1.5).epsilon(1e-9));
  // sqrt(sum of rho over ALL lags, both signs) must equal the scalar factor.
  double sum = rho[0];
  for (size_t i = 1; i < rho.size(); ++i) sum += 2.0 * rho[i];
  REQUIRE(std::sqrt(sum) == Approx(kernel_noise_correlation_sigma_factor(1.0f, 2)).epsilon(1e-9));
}

TEST_CASE("kernel noise: production default pixfrac=0.8, internal_scale=2 "
          "(d=1.6) matches 1.6/sqrt(1.18) (plan 12.4)") {
  // d = 1.6: overlaps of [j+0.5-0.8, j+0.5+0.8] with [0,1] over j=-1,0,1
  //   j=-1: [-1.3,0.3]&[0,1] = 0.3
  //   j=0 : [-0.3,1.3]&[0,1] = 1.0
  //   j=1 : [ 0.7,2.3]&[0,1] = 0.3
  //   S0 = 0.09 + 1 + 0.09 = 1.18; W = 1.6
  REQUIRE(kernel_noise_correlation_sigma_factor(0.8f, 2) ==
          Approx(1.6 / std::sqrt(1.18)).epsilon(1e-6));
  REQUIRE(kernel_noise_correlation_sigma_factor(0.8f, 2) > 1.0);
}

TEST_CASE("kernel noise: invalid arguments are rejected") {
  REQUIRE_THROWS(kernel_noise_correlation_sigma_factor(0.0f, 2));
  REQUIRE_THROWS(kernel_noise_correlation_sigma_factor(1.5f, 2));
  REQUIRE_THROWS(kernel_noise_correlation_sigma_factor(0.8f, 3));
}
