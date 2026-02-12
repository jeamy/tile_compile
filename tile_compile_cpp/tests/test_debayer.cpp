#if __has_include(<catch2/catch_test_macros.hpp>)
#include "tile_compile/image/cfa_processing.hpp"
#include "tile_compile/core/types.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace {

struct Offsets {
  int r_row, r_col;
  int b_row, b_col;
};

Offsets offsets_for(tile_compile::BayerPattern p) {
  switch (p) {
  case tile_compile::BayerPattern::RGGB:
    return {0, 0, 1, 1};
  case tile_compile::BayerPattern::BGGR:
    return {1, 1, 0, 0};
  case tile_compile::BayerPattern::GRBG:
    return {0, 1, 1, 0};
  case tile_compile::BayerPattern::GBRG:
    return {1, 0, 0, 1};
  default:
    return {1, 0, 0, 1};
  }
}

} // namespace

TEST_CASE("debayer_nearest_neighbor_outputs_constant_channels_for_constant_CFA") {
  using tile_compile::Matrix2Df;
  using tile_compile::BayerPattern;

  const float Rv = 100.0f;
  const float Gv = 200.0f;
  const float Bv = 300.0f;

  for (BayerPattern p : {BayerPattern::RGGB, BayerPattern::BGGR, BayerPattern::GRBG,
                        BayerPattern::GBRG}) {
    Matrix2Df mosaic(4, 4);
    auto off = offsets_for(p);

    for (int y = 0; y < mosaic.rows(); ++y) {
      for (int x = 0; x < mosaic.cols(); ++x) {
        int py = y & 1;
        int px = x & 1;
        if (py == off.r_row && px == off.r_col) {
          mosaic(y, x) = Rv;
        } else if (py == off.b_row && px == off.b_col) {
          mosaic(y, x) = Bv;
        } else {
          mosaic(y, x) = Gv;
        }
      }
    }

    auto out = tile_compile::image::debayer_nearest_neighbor(mosaic, p);

    for (int y = 0; y < mosaic.rows(); ++y) {
      for (int x = 0; x < mosaic.cols(); ++x) {
        REQUIRE(out.R(y, x) == Catch::Approx(Rv));
        REQUIRE(out.G(y, x) == Catch::Approx(Gv));
        REQUIRE(out.B(y, x) == Catch::Approx(Bv));
      }
    }
  }
}

#else
int tile_compile_tests_debayer_stub() { return 0; }
#endif
