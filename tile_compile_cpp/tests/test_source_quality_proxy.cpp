// M3 tests for the CFA-aware source-space quality analysis proxy
// (docs/AQMH/aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md
//  section 13.2, proxy_version=1). Minimal M3 scope: the analysis proxy and
// its global MAD-based noise statistic --- scale-specific maps, the cache
// sink and region reads are M5.

#include "tile_compile/reconstruction/source_quality_proxy.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace tile_compile;
using namespace tile_compile::reconstruction;
using Catch::Approx;

TEST_CASE("b3_spline_blur preserves a constant image exactly (DC gain 1)") {
  Matrix2Df img = Matrix2Df::Constant(6, 6, 7.5f);
  auto blurred = b3_spline_blur(img);
  for (int y = 0; y < 6; ++y)
    for (int x = 0; x < 6; ++x) REQUIRE(blurred(y, x) == Approx(7.5f).epsilon(1e-5));
}

TEST_CASE("b3_spline_blur redistributes a single spike per the [1,4,6,4,1]/16 "
          "kernel (hand-computed)") {
  // A single 1.0 spike in an otherwise-zero row, blurred with clamp-to-edge
  // boundaries. Interior taps follow the kernel weights exactly; near the
  // border the clamp folds the out-of-range taps back onto the edge pixel.
  Matrix2Df img = Matrix2Df::Zero(1, 9);
  img(0, 4) = 1.0f;  // interior position, 2 taps clear on both sides
  auto blurred = b3_spline_blur(img);
  REQUIRE(blurred(0, 2) == Approx(1.0 / 16.0).epsilon(1e-5));
  REQUIRE(blurred(0, 3) == Approx(4.0 / 16.0).epsilon(1e-5));
  REQUIRE(blurred(0, 4) == Approx(6.0 / 16.0).epsilon(1e-5));
  REQUIRE(blurred(0, 5) == Approx(4.0 / 16.0).epsilon(1e-5));
  REQUIRE(blurred(0, 6) == Approx(1.0 / 16.0).epsilon(1e-5));
  REQUIRE(blurred(0, 0) == Approx(0.0).margin(1e-6));
  REQUIRE(blurred(0, 8) == Approx(0.0).margin(1e-6));
}

TEST_CASE("median_absolute_deviation_sigma matches the exact hand-computed "
          "formula (plan 13.2 step 2)") {
  Matrix2Df values(1, 5);
  values << 1, 2, 3, 4, 5;
  // median = 3; deviations = {2,1,0,1,2}; median(deviations) = 1;
  // sigma = 1.4826 * 1.
  REQUIRE(median_absolute_deviation_sigma(values) == Approx(1.4826).epsilon(1e-6));
}

TEST_CASE("median_absolute_deviation_sigma of a constant image is exactly 0") {
  Matrix2Df values = Matrix2Df::Constant(4, 4, 42.0f);
  REQUIRE(median_absolute_deviation_sigma(values) == Approx(0.0).margin(1e-9));
}

TEST_CASE("quad-green grid: G_quad = 0.5*(G1+G2), hand-computed on an RGGB "
          "checkerboard (plan 13.2 step 1)") {
  // RGGB, cfa_origin=(0,0): (even,even)=R (odd,even)=G1 (even,odd)=G2 (odd,odd)=B.
  Matrix2Df source = Matrix2Df::Zero(4, 4);
  // Quad (0,0): R=100@(0,0) G1=10@(1,0) G2=1@(0,1) B=200@(1,1) -> 0.5*(10+1)=5.5
  source(0, 0) = 100; source(0, 1) = 10; source(1, 0) = 1; source(1, 1) = 200;
  // Quad (1,0): R=100@(2,0) G1=20@(3,0) G2=2@(2,1) B=300@(3,1) -> 0.5*(20+2)=11
  source(0, 2) = 100; source(0, 3) = 20; source(1, 2) = 2; source(1, 3) = 300;
  // Quad (0,1): R=100@(0,2) G1=30@(1,2) G2=3@(0,3) B=400@(1,3) -> 0.5*(30+3)=16.5
  source(2, 0) = 100; source(2, 1) = 30; source(3, 0) = 3; source(3, 1) = 400;
  // Quad (1,1): R=100@(2,2) G1=40@(3,2) G2=4@(2,3) B=500@(3,3) -> 0.5*(40+4)=22
  source(2, 2) = 100; source(2, 3) = 40; source(3, 2) = 4; source(3, 3) = 500;

  auto result = compute_source_quality_proxy_v1(source, ColorMode::OSC, BayerPattern::RGGB, 0, 0);
  REQUIRE(result.quad_width == 2);
  REQUIRE(result.quad_height == 2);
  REQUIRE(result.quad_green(0, 0) == Approx(5.5));
  REQUIRE(result.quad_green(0, 1) == Approx(11.0));
  REQUIRE(result.quad_green(1, 0) == Approx(16.5));
  REQUIRE(result.quad_green(1, 1) == Approx(22.0));
}

TEST_CASE("edge-aware full proxy: an R/B site prefers the neighbour-pair "
          "direction with the lower gradient (plan 13.2 step 3)") {
  // RGGB, cfa_origin=(0,0): (1,1) is odd/odd -> B.
  // West=(0,1)=G2, East=(2,1)=G2, North=(1,0)=G1, South=(1,2)=G1.
  Matrix2Df source = Matrix2Df::Zero(5, 5);
  source(1, 0) = 10;  // west
  source(1, 2) = 90;  // east  -> horizontal gradient = 80 (a real edge in x)
  source(0, 1) = 50;  // north
  source(2, 1) = 52;  // south -> vertical gradient = 2 (smooth)
  auto result = compute_source_quality_proxy_v1(source, ColorMode::OSC, BayerPattern::RGGB, 0, 0);
  // Must pick the smooth (vertical) direction, not average across the edge.
  REQUIRE(result.proxy_full(1, 1) == Approx(51.0).epsilon(1e-6));

  // Swap so the edge is now vertical and the smooth direction is horizontal.
  Matrix2Df source2 = Matrix2Df::Zero(5, 5);
  source2(1, 0) = 50;  // west
  source2(1, 2) = 52;  // east  -> horizontal gradient = 2 (smooth)
  source2(0, 1) = 10;  // north
  source2(2, 1) = 90;  // south -> vertical gradient = 80 (a real edge in y)
  auto result2 =
      compute_source_quality_proxy_v1(source2, ColorMode::OSC, BayerPattern::RGGB, 0, 0);
  REQUIRE(result2.proxy_full(1, 1) == Approx(51.0).epsilon(1e-6));
}

TEST_CASE("edge-aware full proxy leaves native green samples untouched "
          "(plan 13.2 step 3)") {
  Matrix2Df source = Matrix2Df::Zero(5, 5);
  source(1, 1) = 77.0f;  // odd,odd -> B for RGGB, irrelevant here
  source(0, 1) = 123.0f;  // odd,even -> G1: must pass through unchanged
  auto result = compute_source_quality_proxy_v1(source, ColorMode::OSC, BayerPattern::RGGB, 0, 0);
  REQUIRE(result.proxy_full(0, 1) == Approx(123.0f));
}

TEST_CASE("MONO path: proxy_full equals the input exactly, no quad grid "
          "(plan 13.2: no CFA interpolation for MONO)") {
  Matrix2Df source(3, 3);
  source << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  auto result =
      compute_source_quality_proxy_v1(source, ColorMode::MONO, BayerPattern::UNKNOWN, 0, 0);
  REQUIRE(result.quad_width == 0);
  REQUIRE(result.quad_height == 0);
  REQUIRE(result.quad_green.size() == 0);
  for (int y = 0; y < 3; ++y)
    for (int x = 0; x < 3; ++x) REQUIRE(result.proxy_full(y, x) == source(y, x));
}

TEST_CASE("global sigma_green is 0 for a perfectly uniform CFA frame "
          "(plan 13.2 step 2)") {
  // Constant value everywhere: every quad's G_quad is identical -> highpass
  // is exactly 0 everywhere -> MAD-sigma is exactly 0.
  Matrix2Df source = Matrix2Df::Constant(8, 8, 500.0f);
  auto result = compute_source_quality_proxy_v1(source, ColorMode::OSC, BayerPattern::RGGB, 0, 0);
  REQUIRE(result.sigma_green == Approx(0.0).margin(1e-6));
}

// --- plan 13.2 mandated proxy tests: Bayer-checkerboard, colored stars,
//     narrowband MONO. (Veto-leakage needs the zero-veto mask, which is not
//     yet implemented in proxy_version=1 --- tracked in 30.18/30.24.)

TEST_CASE("proxy 13.2: a pure R/B chroma checkerboard with a flat green "
          "grid does NOT leak into sigma_green (step 5: R/B chroma enters "
          "neither sigma_green nor the sharpness ranking)") {
  // 16x16 RGGB, cfa_origin (0,0): G at (odd,even) and (even,odd).
  // Green kept perfectly flat; R and B alternate between two extremes.
  Matrix2Df src = Matrix2Df::Zero(16, 16);
  for (int y = 0; y < 16; ++y)
    for (int x = 0; x < 16; ++x) {
      const bool xodd = x & 1, yodd = y & 1;
      const bool is_green = (xodd != yodd);
      if (is_green) {
        src(y, x) = 1000.0f;  // flat green
      } else if (!xodd && !yodd) {
        src(y, x) = ((x / 2 + y / 2) & 1) ? 4000.0f : 50.0f;  // R checkerboard
      } else {
        src(y, x) = ((x / 2 + y / 2) & 1) ? 60.0f : 3800.0f;  // B checkerboard
      }
    }
  auto r = compute_source_quality_proxy_v1(src, ColorMode::OSC, BayerPattern::RGGB, 0, 0);
  // Quad-Green grid is the mean of two flat greens -> perfectly flat ->
  // highpass exactly 0 -> sigma_green exactly 0, regardless of R/B chaos.
  REQUIRE(r.sigma_green == Approx(0.0).margin(1e-6));
  // The Quad-Green grid itself must be flat (no chroma bleed).
  for (int i = 0; i < r.quad_green.size(); ++i)
    REQUIRE(r.quad_green.data()[i] == Approx(1000.0f));
}

TEST_CASE("proxy 13.2: a strongly colored (red) star is represented by its "
          "real green flux, not by its red chroma (step 5)") {
  // 12x12 RGGB. Uniform faint green background; one 'star' that is bright in
  // R but only modestly above background in G, and dark in B.
  Matrix2Df src = Matrix2Df::Constant(12, 12, 100.0f);
  auto set_channel = [&](CfaChannel want, int cx, int cy, float val) {
    for (int y = cy - 1; y <= cy + 1; ++y)
      for (int x = cx - 1; x <= cx + 1; ++x)
        if (cfa_channel_for_source_pixel(x, y, BayerPattern::RGGB, 0, 0) == want)
          src(y, x) = val;
  };
  set_channel(CfaChannel::R, 6, 6, 9000.0f);  // very bright red
  set_channel(CfaChannel::G, 6, 6, 400.0f);   // modest green
  set_channel(CfaChannel::B, 6, 6, 30.0f);    // dark blue

  auto r = compute_source_quality_proxy_v1(src, ColorMode::OSC, BayerPattern::RGGB, 0, 0);
  // At the star's Quad-Green location the proxy value follows the ~400 green
  // sample, nowhere near the 9000 red. Bound it well below red, comfortably
  // above the 100 background.
  const int qx = 6 / 2, qy = 6 / 2;
  const float g = r.quad_green(qy, qx);
  REQUIRE(g > 150.0f);
  REQUIRE(g < 1500.0f);  // categorically not the 9000 red
  // The full-res green proxy at a red site near the star interpolates from
  // green neighbours (~400), never the red value.
  REQUIRE(r.proxy_full(6, 6) < 2000.0f);
}

TEST_CASE("proxy 13.2: narrowband MONO uses the L plane directly with the "
          "same highpass/MAD, no CFA interpolation, matching a hand value") {
  // A MONO frame: smooth ramp + a single sharp spike. sigma_green must be
  // computed as 1.4826 * median(|hp - median(hp)|) on L - B3_blur(L),
  // identical to feeding the same array to median_absolute_deviation_sigma.
  Matrix2Df L(6, 6);
  for (int y = 0; y < 6; ++y)
    for (int x = 0; x < 6; ++x) L(y, x) = 10.0f * x + 3.0f * y;
  L(3, 3) += 500.0f;  // narrowband emission spike

  auto r = compute_source_quality_proxy_v1(L, ColorMode::MONO, BayerPattern::UNKNOWN, 0, 0);
  REQUIRE(r.quad_green.size() == 0);
  for (int y = 0; y < 6; ++y)
    for (int x = 0; x < 6; ++x) REQUIRE(r.proxy_full(y, x) == L(y, x));

  const Matrix2Df hp = L - b3_spline_blur(L);
  REQUIRE(r.sigma_green == Approx(median_absolute_deviation_sigma(hp)).epsilon(1e-6));
}
