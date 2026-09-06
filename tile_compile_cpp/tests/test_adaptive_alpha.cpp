// M6 tests for the adaptive per-band alpha (plan 14.4). This batch covers the
// two factors computable from the profile planes alone --- A_neff and
// A_coverage --- plus the alpha_cap and external-factor product. A_separation
// / A_artifact / A_registration are supplied here as pre-reduced maps.

#include "tile_compile/reconstruction/adaptive_alpha.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <vector>

using namespace tile_compile::reconstruction;
using tile_compile::ColorMode;
using Catch::Approx;

namespace {

ForwardDrizzleUniformResult mono_profile(int w, int h, float n_eff,
                                         float weight_sum) {
  ForwardDrizzleUniformResult r;
  r.color_mode = ColorMode::MONO;
  r.internal_width = w;
  r.internal_height = h;
  r.L.allocate(w, h);
  for (int i = 0; i < w * h; ++i) {
    r.L.value[i] = 1.0f;
    r.L.weight_sum[i] = weight_sum;
    r.L.n_eff[i] = n_eff;
    r.L.support[i] = 1u;
  }
  return r;
}

}  // namespace

TEST_CASE("adaptive alpha: smoothstep is clamped Hermite") {
  REQUIRE(alpha_smoothstep(8.0, 24.0, 4.0) == Approx(0.0));
  REQUIRE(alpha_smoothstep(8.0, 24.0, 24.0) == Approx(1.0));
  REQUIRE(alpha_smoothstep(8.0, 24.0, 16.0) == Approx(0.5));  // midpoint
  REQUIRE(alpha_smoothstep(8.0, 24.0, 12.0) == Approx(0.15625));  // t=.25
}

TEST_CASE("adaptive alpha: A_neff drives the Fine band --- below "
          "min_effective_samples alpha is 0, at/above full it saturates") {
  const int w = 8, h = 8;
  const auto uniform = mono_profile(w, h, 40.0f, 100.0f);
  AdaptiveAlphaParams p;  // 8 / 24, cap 1

  SECTION("n_eff below min -> alpha 0") {
    const auto fine = mono_profile(w, h, 5.0f, 100.0f);
    auto a = compute_adaptive_alpha(uniform, fine, {}, ColorMode::MONO, w, h, 1,
                                    p);
    for (float v : a[0]) REQUIRE(v == Approx(0.0));
  }
  SECTION("n_eff at full, full coverage -> alpha == alpha_cap") {
    const auto fine = mono_profile(w, h, 24.0f, 100.0f);
    auto a = compute_adaptive_alpha(uniform, fine, {}, ColorMode::MONO, w, h, 1,
                                    p);
    for (float v : a[0]) REQUIRE(v == Approx(1.0));
  }
  SECTION("alpha_cap scales the result") {
    const auto fine = mono_profile(w, h, 24.0f, 100.0f);
    p.alpha_cap = 0.4f;
    auto a = compute_adaptive_alpha(uniform, fine, {}, ColorMode::MONO, w, h, 1,
                                    p);
    for (float v : a[0]) REQUIRE(v == Approx(0.4));
  }
}

TEST_CASE("adaptive alpha: A_coverage = clamp(profile_weight / uniform_weight, "
          "0, 1)") {
  const int w = 6, h = 6;
  const auto uniform = mono_profile(w, h, 40.0f, 100.0f);
  const auto fine = mono_profile(w, h, 40.0f, 30.0f);  // 30/100 coverage
  AdaptiveAlphaParams p;
  auto a = compute_adaptive_alpha(uniform, fine, {}, ColorMode::MONO, w, h, 1, p);
  // A_neff = 1 (n_eff 40 >= full 24); A_coverage = 0.3.
  for (float v : a[0]) REQUIRE(v == Approx(0.3));
}

TEST_CASE("adaptive alpha: OSC uses the conservative channel minimum for both "
          "n_eff and coverage") {
  const int w = 4, h = 4;
  const std::size_t n = static_cast<std::size_t>(w) * h;
  ForwardDrizzleUniformResult uniform;
  uniform.color_mode = ColorMode::OSC;
  ForwardDrizzleUniformResult fine;
  fine.color_mode = ColorMode::OSC;
  for (auto *pl : {&uniform.R, &uniform.G, &uniform.B}) pl->allocate(w, h);
  for (auto *pl : {&fine.R, &fine.G, &fine.B}) pl->allocate(w, h);
  auto fill = [&](ProfilePlane &pl, float ne, float ws) {
    for (std::size_t i = 0; i < n; ++i) {
      pl.value[i] = 1.0f; pl.n_eff[i] = ne; pl.weight_sum[i] = ws;
      pl.support[i] = 1u;
    }
  };
  fill(uniform.R, 50, 100); fill(uniform.G, 50, 100); fill(uniform.B, 50, 100);
  // G is the weak channel: n_eff 16 -> A_neff,G = smoothstep(8,24,16) = 0.5;
  // R/B saturate. Coverage: R 1.0, G 0.8, B 1.0 -> min 0.8.
  fill(fine.R, 40, 100); fill(fine.G, 16, 80); fill(fine.B, 40, 100);
  AdaptiveAlphaParams p;
  auto a = compute_adaptive_alpha(uniform, fine, {}, ColorMode::OSC, w, h, 1, p);
  for (float v : a[0]) REQUIRE(v == Approx(0.5 * 0.8).epsilon(1e-5));
}

TEST_CASE("adaptive alpha: external A_separation / A_artifact / A_registration "
          "multiply in and any one at 0 forces alpha 0") {
  const int w = 5, h = 5;
  const std::size_t n = static_cast<std::size_t>(w) * h;
  const auto uniform = mono_profile(w, h, 40.0f, 100.0f);
  const auto fine = mono_profile(w, h, 40.0f, 100.0f);  // A_neff=A_cov=1
  AdaptiveAlphaParams p;

  std::vector<float> sep(n, 0.5f), art(n, 0.6f), reg(n, 1.0f);
  auto a = compute_adaptive_alpha(uniform, fine, {}, ColorMode::MONO, w, h, 1, p,
                                  sep, art, reg);
  for (float v : a[0]) REQUIRE(v == Approx(0.5 * 0.6 * 1.0).epsilon(1e-5));

  std::fill(art.begin(), art.end(), 0.0f);
  auto a0 = compute_adaptive_alpha(uniform, fine, {}, ColorMode::MONO, w, h, 1,
                                   p, sep, art, reg);
  for (float v : a0[0]) REQUIRE(v == Approx(0.0));
}

TEST_CASE("adaptive alpha: an unsupported profile pixel gets alpha 0; Raw "
          "bands (>=3) return an empty map") {
  const int w = 4, h = 4;
  auto uniform = mono_profile(w, h, 40.0f, 100.0f);
  auto fine = mono_profile(w, h, 40.0f, 100.0f);
  auto medium = mono_profile(w, h, 40.0f, 100.0f);
  fine.L.support[5] = 0u;  // one unsupported Fine pixel
  AdaptiveAlphaParams p;
  auto a = compute_adaptive_alpha(uniform, fine, medium, ColorMode::MONO, w, h,
                                  3, p);
  REQUIRE(a.size() == 3);
  REQUIRE(a[0][5] == Approx(0.0));  // unsupported -> 0
  REQUIRE(a[0][0] == Approx(1.0));
  REQUIRE(a[1].size() == static_cast<size_t>(w * h));  // medium band computed
  REQUIRE(a[2].empty());                               // Raw band -> ignored
}

TEST_CASE("adaptive alpha: input validation") {
  const int w = 4, h = 4;
  const auto pr = mono_profile(w, h, 10.0f, 10.0f);
  AdaptiveAlphaParams p;
  REQUIRE_THROWS(
      compute_adaptive_alpha(pr, pr, {}, ColorMode::MONO, w, h, 0, p));
  REQUIRE_THROWS(
      compute_adaptive_alpha(pr, pr, {}, ColorMode::MONO, w, h, 5, p));
  AdaptiveAlphaParams bad = p;
  bad.full_effective_samples = 4.0f;  // < min
  REQUIRE_THROWS(
      compute_adaptive_alpha(pr, pr, {}, ColorMode::MONO, w, h, 1, bad));
}
