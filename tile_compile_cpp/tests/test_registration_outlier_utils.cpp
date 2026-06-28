#include "tile_compile/runner/registration_outlier_utils.hpp"

#if __has_include(<catch2/catch_test_macros.hpp>)
#include <catch2/catch_test_macros.hpp>

#include <cmath>

namespace tile_compile::runner {

TEST_CASE("registration shift diagnostics keep genuine 180 degree flips plausible") {
  WarpMatrix half_turn = WarpMatrix::Zero();
  half_turn(0, 0) = -1.0f;
  half_turn(1, 1) = -1.0f;
  half_turn(0, 2) = 5201.0f;
  half_turn(1, 2) = 3463.0f;

  const auto diag = registration_shift_diagnostics(half_turn, 5202, 3464);
  REQUIRE(diag.half_turn_family);
  REQUIRE(std::fabs(diag.shift_magnitude) < 1.0e-4f);
}

TEST_CASE("registration shift diagnostics preserve normal translation behaviour") {
  WarpMatrix translate = WarpMatrix::Identity();
  translate(0, 2) = 24.0f;
  translate(1, 2) = -18.0f;

  const auto diag = registration_shift_diagnostics(translate, 5202, 3464);
  REQUIRE_FALSE(diag.half_turn_family);
  REQUIRE(std::fabs(diag.shift_magnitude - 30.0f) < 1.0e-4f);
}

TEST_CASE("isolated half-turn matches are not treated as meridian flips") {
  std::vector<uint8_t> candidates(40, 0);
  candidates[5] = 1;
  candidates[17] = 1;
  candidates[18] = 1;
  candidates[33] = 1;

  const auto supported = persistent_half_turn_support(candidates);
  REQUIRE(std::count(supported.begin(), supported.end(), uint8_t{1}) == 0);
}

TEST_CASE("temporally coherent half-turn matches support a meridian flip") {
  std::vector<uint8_t> candidates(40, 0);
  candidates[15] = 1;
  candidates[16] = 1;
  candidates[18] = 1; // tolerate one failed frame inside the segment
  candidates[19] = 1;

  const auto supported = persistent_half_turn_support(candidates);
  REQUIRE(supported[15] == 1);
  REQUIRE(supported[16] == 1);
  REQUIRE(supported[18] == 1);
  REQUIRE(supported[19] == 1);
}

} // namespace tile_compile::runner
#endif
