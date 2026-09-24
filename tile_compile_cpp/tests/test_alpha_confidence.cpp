// M6 tests for A_separation / A_artifact / A_registration (plan 14.4) --- the
// adaptive-alpha confidence factors derived from per-frame stripe statistics.

#include "tile_compile/reconstruction/alpha_confidence.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <vector>

using namespace tile_compile::reconstruction;
using Catch::Approx;

TEST_CASE("weighted_percentile: equal weights reduce to the ordinary Hazen "
          "percentile") {
  const std::vector<double> v{1, 2, 3, 4, 5};
  const std::vector<double> w(5, 1.0);
  REQUIRE(weighted_percentile(v, w, 0.5) == Approx(3.0));
  REQUIRE(weighted_percentile(v, w, 0.9) == Approx(5.0));
  REQUIRE(weighted_percentile(v, w, 0.0) == Approx(1.0));
  REQUIRE(weighted_percentile(v, w, 1.0) == Approx(5.0));
  REQUIRE(weighted_percentile(v, w, 0.1) == Approx(1.0));
}

TEST_CASE("weighted_percentile: weights shift the percentile toward the "
          "heavier samples") {
  const std::vector<double> v{10, 20};
  const std::vector<double> w{3, 1};  // total 4
  // Hazen CDF: 0.375 at v=10, 0.875 at v=20. p=0.5 -> 10 + 0.25*(20-10).
  REQUIRE(weighted_percentile(v, w, 0.5) == Approx(12.5));
  REQUIRE(weighted_percentile(v, w, 0.375) == Approx(10.0));
}

TEST_CASE("weighted_percentile: validation") {
  const std::vector<double> v{1, 2, 3};
  REQUIRE_THROWS(weighted_percentile(v, std::vector<double>{1, 2}, 0.5));
  REQUIRE(std::isnan(weighted_percentile({}, {}, 0.5)));
}

namespace {
AlphaFactorContribution mk(double b, double q, double art, bool direct,
                           double resid) {
  return AlphaFactorContribution{b, q, art, direct, resid};
}
}  // namespace

TEST_CASE("alpha confidence: empty contribution set -> all factors zero") {
  auto f = compute_alpha_confidence_channel({});
  REQUIRE(f.a_separation == Approx(0.0));
  REQUIRE(f.a_artifact == Approx(0.0));
  REQUIRE(f.a_registration == Approx(0.0));
  REQUIRE_FALSE(f.artifact_applicable);
}

TEST_CASE("alpha confidence: A_separation is 0 for a uniform Q population and "
          "saturates for a wide Q spread") {
  std::vector<AlphaFactorContribution> flat, wide;
  for (int i = 0; i < 12; ++i) {
    flat.push_back(mk(1.0, 0.5, 1.0, true, 1.0));
    wide.push_back(mk(1.0, i < 6 ? 0.15 : 0.85, 1.0, true, 1.0));
  }
  REQUIRE(compute_alpha_confidence_channel(flat).a_separation == Approx(0.0));
  // spread q_p90 - q_p50 ~ 0.7 >> full_quality_separation 0.20 -> ~1.
  REQUIRE(compute_alpha_confidence_channel(wide).a_separation > 0.99);
}

TEST_CASE("alpha confidence: A_artifact needs >= min_artifact_contributors "
          "valid a_f, and a low-artifact tail pulls the weighted p10 down") {
  std::vector<AlphaFactorContribution> few, clean, tail;
  for (int i = 0; i < 5; ++i) few.push_back(mk(1.0, 0.5, 0.9, true, 1.0));
  for (int i = 0; i < 10; ++i) clean.push_back(mk(1.0, 0.5, 0.95, true, 1.0));
  for (int i = 0; i < 10; ++i)
    tail.push_back(mk(1.0, 0.5, i < 8 ? 0.95 : 0.10, true, 1.0));

  auto ffew = compute_alpha_confidence_channel(few);
  REQUIRE_FALSE(ffew.artifact_applicable);
  REQUIRE(ffew.a_artifact == Approx(0.0));

  auto fclean = compute_alpha_confidence_channel(clean);
  REQUIRE(fclean.artifact_applicable);
  REQUIRE(fclean.a_artifact > 0.99);  // a_p10 ~ 0.95 -> smoothstep(0.25,0.75)=1

  auto ftail = compute_alpha_confidence_channel(tail);
  REQUIRE(ftail.artifact_applicable);
  REQUIRE(ftail.a_artifact < fclean.a_artifact);  // tail drags p10 down
}

TEST_CASE("alpha confidence: a non-finite artifact_conf is 'not applicable' "
          "and excluded from the count") {
  std::vector<AlphaFactorContribution> v;
  for (int i = 0; i < 7; ++i) v.push_back(mk(1.0, 0.5, 0.9, true, 1.0));
  for (int i = 0; i < 5; ++i)
    v.push_back(mk(1.0, 0.5, std::numeric_limits<double>::quiet_NaN(), true,
                   1.0));
  // Only 7 finite a_f < 8 -> not applicable.
  auto f = compute_alpha_confidence_channel(v);
  REQUIRE_FALSE(f.artifact_applicable);
  REQUIRE(f.a_artifact == Approx(0.0));
}

TEST_CASE("alpha confidence: A_registration = min(direct-fraction gate, "
          "residual-p20 gate)") {
  std::vector<AlphaFactorContribution> all_direct, half_direct, weak_resid;
  for (int i = 0; i < 10; ++i) {
    all_direct.push_back(mk(1.0, 0.5, 0.9, true, 1.0));
    half_direct.push_back(mk(1.0, 0.5, 0.9, i < 5, 1.0));
    weak_resid.push_back(mk(1.0, 0.5, 0.9, true, 0.55));
  }
  REQUIRE(compute_alpha_confidence_channel(all_direct).a_registration >
          0.99);
  // direct_fraction 0.5 -> smoothstep(0.50,0.85,0.5) = 0.
  REQUIRE(compute_alpha_confidence_channel(half_direct).a_registration ==
          Approx(0.0));
  // residual_p20 0.55 -> smoothstep(0.55,0.90,0.55) = 0.
  REQUIRE(compute_alpha_confidence_channel(weak_resid).a_registration ==
          Approx(0.0));
}

TEST_CASE("alpha confidence: geometric weight B skews the percentiles") {
  // Nine light contributions at high Q + one very heavy one at low Q: the
  // weighted p50 is dragged down toward the heavy sample.
  std::vector<AlphaFactorContribution> v;
  for (int i = 0; i < 9; ++i) v.push_back(mk(0.1, 0.9, 0.9, true, 1.0));
  v.push_back(mk(10.0, 0.1, 0.9, true, 1.0));
  auto f = compute_alpha_confidence_channel(v);
  // p50 near 0.1 (heavy), p90 near 0.9 -> large separation -> saturates.
  REQUIRE(f.a_separation > 0.99);
}
