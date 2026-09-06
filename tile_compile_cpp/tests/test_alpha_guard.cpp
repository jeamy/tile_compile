// M6 tests for the local energy guard (plan 14.5) and support-aware alpha
// smoothing (plan 14.7).

#include "tile_compile/reconstruction/alpha_guard.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <numeric>
#include <vector>

using namespace tile_compile::reconstruction;
using Catch::Approx;

TEST_CASE("energy guard: window radius is max(3, 2^(j+1))") {
  REQUIRE(energy_guard_window_radius(1) == 4);
  REQUIRE(energy_guard_window_radius(2) == 8);
  REQUIRE(energy_guard_window_radius(3) == 16);
}

TEST_CASE("mad_sigma: 1.4826 * median(|x - median(x)|), hand-computed") {
  REQUIRE(mad_sigma({1, 2, 3, 4, 5}) == Approx(1.4826));
  REQUIRE(mad_sigma({5, 5, 5, 5}) == Approx(0.0));
  // [0,0,0,10]: median 0, deviations [0,0,0,10] -> median 0.
  REQUIRE(mad_sigma({0, 0, 0, 10}) == Approx(0.0));
}

namespace {
std::vector<float> ramp(int w, int h, float slope) {
  std::vector<float> v(static_cast<size_t>(w) * h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      v[static_cast<size_t>(y) * w + x] = slope * (x - y);
  return v;
}
}  // namespace

TEST_CASE("energy guard: D_profile == D_R => mixing changes nothing => alpha "
          "unchanged") {
  const int w = 24, h = 24;
  const std::size_t n = static_cast<size_t>(w) * h;
  const auto dr = ramp(w, h, 0.3f);
  const std::vector<float> alpha_pre(n, 0.8f);
  const std::vector<uint8_t> support(n, 1u);
  auto g = apply_energy_guard(alpha_pre, dr, dr, support, w, h, 6, 0.0);
  for (float v : g) REQUIRE(v == Approx(0.8));
}

TEST_CASE("energy guard: a noisy detail profile inflates local energy and the "
          "bisection reduces alpha until energy_ratio <= 1.30") {
  const int w = 32, h = 32;
  const std::size_t n = static_cast<size_t>(w) * h;
  const auto dr = ramp(w, h, 0.02f);  // low local MAD
  std::vector<float> dp = dr;
  for (std::size_t i = 0; i < n; ++i)
    dp[i] += ((i * 2654435761u) % 101) / 50.0f - 1.0f;  // deterministic +-1
  const std::vector<float> alpha_pre(n, 1.0f);
  const std::vector<uint8_t> support(n, 1u);

  EnergyGuardParams p;
  auto g = apply_energy_guard(alpha_pre, dr, dp, support, w, h,
                              energy_guard_window_radius(1), 0.0, p);

  // Interior pixel (full window).
  const int cx = 16, cy = 16;
  const float a = g[static_cast<size_t>(cy) * w + cx];
  REQUIRE(a > 0.0f);
  REQUIRE(a < 1.0f);

  // Recompute the ratio at the guarded alpha over the same window; must be
  // within the limit (bisection returns the feasible `lo`).
  std::vector<float> mix;
  for (int yy = cy - 4; yy <= cy + 4; ++yy)
    for (int xx = cx - 4; xx <= cx + 4; ++xx) {
      const size_t j = static_cast<size_t>(yy) * w + xx;
      mix.push_back(dr[j] + a * (dp[j] - dr[j]));
    }
  std::vector<float> rawwin;
  for (int yy = cy - 4; yy <= cy + 4; ++yy)
    for (int xx = cx - 4; xx <= cx + 4; ++xx)
      rawwin.push_back(dr[static_cast<size_t>(yy) * w + xx]);
  const double scale_raw = std::max(mad_sigma(rawwin), 0.0);
  REQUIRE(mad_sigma(mix) / scale_raw <= p.energy_limit + 1e-6);
}

TEST_CASE("energy guard: fewer than min_window_pixels valid neighbours => "
          "alpha 0 for that band pixel") {
  const int w = 20, h = 20;
  const std::size_t n = static_cast<size_t>(w) * h;
  const auto dr = ramp(w, h, 0.1f);
  std::vector<uint8_t> support(n, 0u);
  // A 4x4 valid island = 16 < 25 pixels.
  for (int y = 8; y < 12; ++y)
    for (int x = 8; x < 12; ++x) support[static_cast<size_t>(y) * w + x] = 1u;
  const std::vector<float> alpha_pre(n, 0.9f);
  auto g = apply_energy_guard(alpha_pre, dr, dr, support, w, h, 6, 0.0);
  for (int y = 8; y < 12; ++y)
    for (int x = 8; x < 12; ++x)
      REQUIRE(g[static_cast<size_t>(y) * w + x] == Approx(0.0));
}

TEST_CASE("energy guard: background_band_floor sets the noise scale when the "
          "raw band MAD is tiny") {
  const int w = 30, h = 30;
  const std::size_t n = static_cast<size_t>(w) * h;
  const std::vector<float> dr(n, 0.0f);  // MAD 0
  std::vector<float> dp(n, 0.0f);
  // Non-degenerate spread: 101 distinct deterministic levels in ~[-0.2, 0.2].
  for (std::size_t i = 0; i < n; ++i)
    dp[i] = (((i * 2654435761u) % 101) / 50.0f - 1.0f) * 0.2f;
  const std::vector<float> alpha_pre(n, 1.0f);
  const std::vector<uint8_t> support(n, 1u);
  const int cx = 15, cy = 15;
  // floor 1.0: mixed MAD at alpha 1 is ~0.1 -> ratio ~0.1 <= 1.30 -> unchanged.
  auto g = apply_energy_guard(alpha_pre, dr, dp, support, w, h,
                              energy_guard_window_radius(1), 1.0);
  REQUIRE(g[static_cast<size_t>(cy) * w + cx] == Approx(1.0));
  // floor 0.01: ratio >> 1.30 -> bisection pulls alpha well below 1.
  auto g2 = apply_energy_guard(alpha_pre, dr, dp, support, w, h,
                               energy_guard_window_radius(1), 0.01);
  REQUIRE(g2[static_cast<size_t>(cy) * w + cx] < 0.5f);
}

TEST_CASE("alpha smoothing: alpha_guarded == 0 stays exactly 0 and the min "
          "cap prevents smoothing from lifting a low-alpha pixel") {
  const int w = 16, h = 16;
  const std::size_t n = static_cast<size_t>(w) * h;
  std::vector<uint8_t> support(n, 1u);
  std::vector<float> a(n, 0.2f);
  a[static_cast<size_t>(8) * w + 8] = 1.0f;  // one high spike
  a[static_cast<size_t>(0) * w + 0] = 0.0f;  // one exact zero

  auto s = smooth_alpha_b3(a, support, w, h);

  REQUIRE(s[static_cast<size_t>(0) * w + 0] == Approx(0.0));  // stays 0
  // Neighbour of the spike: blur pulls it up, min cap holds it at 0.2.
  REQUIRE(s[static_cast<size_t>(8) * w + 9] == Approx(0.2).margin(1e-6));
  // The spike itself: blur reduces it below 1.
  REQUIRE(s[static_cast<size_t>(8) * w + 8] < 1.0f);
  REQUIRE(s[static_cast<size_t>(8) * w + 8] > 0.2f);
}

TEST_CASE("alpha smoothing: no bleed across separate 4-connected support "
          "islands") {
  const int w = 21, h = 8;
  const std::size_t n = static_cast<size_t>(w) * h;
  std::vector<uint8_t> support(n, 0u);
  std::vector<float> a(n, 0.0f);
  // Island A: cols [0,9), alpha 1. Island B: cols [11,20), alpha 0.
  // Column 10 is an invalid gap -> the two are separate components.
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < 9; ++x) {
      support[static_cast<size_t>(y) * w + x] = 1u;
      a[static_cast<size_t>(y) * w + x] = 1.0f;
    }
    for (int x = 11; x < 20; ++x)
      support[static_cast<size_t>(y) * w + x] = 1u;  // alpha stays 0
  }
  auto s = smooth_alpha_b3(a, support, w, h);
  for (int y = 0; y < h; ++y)
    for (int x = 11; x < 20; ++x)
      REQUIRE(s[static_cast<size_t>(y) * w + x] == Approx(0.0));
  // Island A interior stays ~1 (constant, renormalised).
  REQUIRE(s[static_cast<size_t>(4) * w + 4] == Approx(1.0).margin(1e-6));
}
