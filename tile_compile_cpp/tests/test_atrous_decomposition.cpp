// M6 tests for the masked, support-propagating a-trous decomposition
// (docs/AQMH/aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md
//  section 14.2). Acceptance is fully hand-computable: DC gain 1 on a
// constant image, C_L + sum_j D_j == input exactly on the tightest common
// valid support, level-1 spike response = the B3 kernel, and monotonically
// shrinking level support M_L <= ... <= M_1 <= M_0.

#include "tile_compile/reconstruction/atrous_decomposition.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <numeric>
#include <vector>

using namespace tile_compile::reconstruction;
using Catch::Approx;

namespace {

int support_count(const std::vector<uint8_t> &m) {
  return std::accumulate(m.begin(), m.end(), 0,
                         [](int a, uint8_t b) { return a + (b ? 1 : 0); });
}

}  // namespace

TEST_CASE("atrous: den_min is the fixed 0.5 hash-domain constant") {
  REQUIRE(kAtrousDenMinFraction == 0.5);
  REQUIRE(kAtrousDecompositionVersion == 1);
}

TEST_CASE("atrous: a constant image has ~zero detail on its support and the "
          "coarse residual carries the DC (gain 1)") {
  const int w = 96, h = 96;
  std::vector<float> img(static_cast<size_t>(w) * h, 7.25f);
  auto d = atrous_decompose(img, {}, w, h, 3);

  REQUIRE(d.bands.size() == 3);
  // Border pixels legitimately fall below den_min (out-of-image taps
  // contribute nothing) and drop out; the deep interior stays fully
  // supported with zero detail.
  int interior_supported = 0;
  for (const auto &b : d.bands)
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = static_cast<size_t>(y) * w + x;
        if (b.support[i]) {
          REQUIRE(std::abs(b.detail[i]) < 1e-5f);
          if (x >= 32 && x < w - 32 && y >= 32 && y < h - 32)
            ++interior_supported;
        }
      }
  REQUIRE(interior_supported > 0);
  for (size_t i = 0; i < d.coarse.size(); ++i)
    if (d.coarse_support[i])
      REQUIRE(d.coarse[i] == Approx(7.25f).margin(1e-4));
  // Deep interior (well inside the cumulative kernel halo) must survive.
  REQUIRE(d.coarse_support[static_cast<size_t>(h / 2) * w + w / 2] == 1u);
  REQUIRE(atrous_reconstruction_max_error(d, img) < 1e-4);
}

TEST_CASE("atrous: reconstruction identity C_L + sum_j D_j == input on a "
          "structured image (all valid)") {
  const int w = 48, h = 40;
  std::vector<float> img(static_cast<size_t>(w) * h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const double ramp = 3.0 + 0.05 * x - 0.03 * y;
      const double bump =
          15.0 * std::exp(-((x - 20) * (x - 20) + (y - 18) * (y - 18)) / 18.0);
      img[static_cast<size_t>(y) * w + x] = static_cast<float>(ramp + bump);
    }
  for (int levels = 1; levels <= 4; ++levels) {
    auto d = atrous_decompose(img, {}, w, h, levels);
    INFO("levels = " << levels);
    REQUIRE(atrous_reconstruction_max_error(d, img) < 2e-3);
  }
}

TEST_CASE("atrous: level-1 spike response equals the separable [1,4,6,4,1]/16 "
          "kernel (dilation 1, hand-computed)") {
  const int w = 21, h = 21;
  std::vector<float> img(static_cast<size_t>(w) * h, 0.0f);
  const int cx = 10, cy = 10;
  img[static_cast<size_t>(cy) * w + cx] = 1.0f;

  auto d = atrous_decompose(img, {}, w, h, 1);
  // C_1 = conv(spike, h (x)) then h (y): C_1(cx,cy) = (6/16)*(6/16) = 36/256.
  // D_1 = C_0 - C_1 => at the centre 1 - 36/256 = 220/256.
  const float d1_centre = d.bands[0].detail[static_cast<size_t>(cy) * w + cx];
  REQUIRE(d1_centre == Approx(220.0f / 256.0f).margin(1e-5));
  // One pixel to the right: C_0 = 0, C_1 = (4/16)*(6/16) = 24/256 =>
  // D_1 = -24/256.
  const float d1_right = d.bands[0].detail[static_cast<size_t>(cy) * w + cx + 1];
  REQUIRE(d1_right == Approx(-24.0f / 256.0f).margin(1e-5));
  // Diagonal (dx=1, dy=1): C_1 = (4/16)*(4/16) = 16/256 => D_1 = -16/256.
  const float d1_diag =
      d.bands[0].detail[static_cast<size_t>(cy + 1) * w + cx + 1];
  REQUIRE(d1_diag == Approx(-16.0f / 256.0f).margin(1e-5));
  // Two pixels out (still in kernel reach): C_1 = (1/16)*(6/16) = 6/256.
  const float d1_two = d.bands[0].detail[static_cast<size_t>(cy) * w + cx + 2];
  REQUIRE(d1_two == Approx(-6.0f / 256.0f).margin(1e-5));
  // Three pixels out is beyond the undilated kernel -> exactly 0.
  const float d1_three = d.bands[0].detail[static_cast<size_t>(cy) * w + cx + 3];
  REQUIRE(d1_three == Approx(0.0f).margin(1e-6));
}

TEST_CASE("atrous: level 2 is dilated (one zero between taps) so it reaches "
          "farther than level 1") {
  const int w = 31, h = 31;
  std::vector<float> img(static_cast<size_t>(w) * h, 0.0f);
  const int cx = 15, cy = 15;
  img[static_cast<size_t>(cy) * w + cx] = 1.0f;

  auto d = atrous_decompose(img, {}, w, h, 2);
  // Level 1 spreads C_1 to +-2. Level 2 with dilation 2 has taps at
  // offsets {-4,-2,0,+2,+4}; applied to C_1 (nonzero within +-2) it reaches
  // +-6. So the coarse (C_2) must be nonzero at offset 5 and 6 from centre,
  // where level 1 alone (C_1) was exactly zero.
  REQUIRE(std::abs(d.coarse[static_cast<size_t>(cy) * w + cx + 5]) > 1e-6f);
  REQUIRE(std::abs(d.coarse[static_cast<size_t>(cy) * w + cx + 6]) > 1e-6f);
  REQUIRE(std::abs(d.coarse[static_cast<size_t>(cy) * w + cx + 7]) < 1e-6f);
}

TEST_CASE("atrous: level support shrinks monotonically and a masked-out hole "
          "erodes the support outward each level") {
  const int w = 60, h = 60;
  std::vector<float> img(static_cast<size_t>(w) * h);
  for (size_t i = 0; i < img.size(); ++i)
    img[i] = static_cast<float>(1.0 + 0.01 * (i % 37));
  std::vector<uint8_t> mask(img.size(), 1u);
  // A solid invalid block in the middle.
  for (int y = 25; y < 35; ++y)
    for (int x = 25; x < 35; ++x) mask[static_cast<size_t>(y) * w + x] = 0u;
  // Value there is NaN (unsupported).
  for (int y = 25; y < 35; ++y)
    for (int x = 25; x < 35; ++x)
      img[static_cast<size_t>(y) * w + x] =
          std::numeric_limits<float>::quiet_NaN();

  auto d = atrous_decompose(img, mask, w, h, 3);

  const int m0 = support_count(mask);
  const int m1 = support_count(d.bands[0].support);
  const int m2 = support_count(d.bands[1].support);
  const int m3 = support_count(d.bands[2].support);
  const int mc = support_count(d.coarse_support);
  REQUIRE(m1 <= m0);
  REQUIRE(m2 <= m1);
  REQUIRE(m3 <= m2);
  REQUIRE(mc == m3);  // coarse_support == M_levels == last band's M_j
  REQUIRE(m3 < m0);   // the hole really did erode support

  // D_j validity is M_(j-1) && M_j: any pixel that dropped out at level j
  // must have BOTH support==0 and detail==NaN in band j.
  for (const auto &b : d.bands)
    for (size_t i = 0; i < b.support.size(); ++i)
      if (!b.support[i]) REQUIRE(std::isnan(b.detail[i]));

  // Reconstruction identity still holds exactly on the tightest common
  // support (M_levels).
  // Rebuild a NaN-free "original on support" for the checker: it only reads
  // pixels where coarse_support is set, and those were valid in M_0.
  REQUIRE(atrous_reconstruction_max_error(d, img) < 2e-3);
}

TEST_CASE("atrous: levels out of [1,4] is rejected") {
  std::vector<float> img(16, 1.0f);
  REQUIRE_THROWS(atrous_decompose(img, {}, 4, 4, 0));
  REQUIRE_THROWS(atrous_decompose(img, {}, 4, 4, 5));
  REQUIRE_NOTHROW(atrous_decompose(img, {}, 4, 4, 1));
  REQUIRE_NOTHROW(atrous_decompose(img, {}, 4, 4, 4));
}
