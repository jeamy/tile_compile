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

TEST_CASE("debayer_bilinear_respects_tile_origin_parity") {
  using tile_compile::Matrix2Df;
  using tile_compile::BayerPattern;

  // Constant-color CFA: correct origin must reproduce the channel values.
  // A wrong (flipped) origin swaps R and B, which is exactly the failure
  // mode of forcing origin (0,0) for mosaics living on an offset lattice.
  const float Rv = 100.0f;
  const float Gv = 200.0f;
  const float Bv = 300.0f;

  for (BayerPattern p : {BayerPattern::RGGB, BayerPattern::BGGR, BayerPattern::GRBG,
                        BayerPattern::GBRG}) {
    auto off = offsets_for(p);

    for (int origin_y = 0; origin_y < 2; ++origin_y) {
      for (int origin_x = 0; origin_x < 2; ++origin_x) {
        Matrix2Df mosaic(6, 6);
        for (int y = 0; y < mosaic.rows(); ++y) {
          for (int x = 0; x < mosaic.cols(); ++x) {
            int py = (origin_y + y) & 1;
            int px = (origin_x + x) & 1;
            if (py == off.r_row && px == off.r_col) {
              mosaic(y, x) = Rv;
            } else if (py == off.b_row && px == off.b_col) {
              mosaic(y, x) = Bv;
            } else {
              mosaic(y, x) = Gv;
            }
          }
        }

        auto out = tile_compile::image::debayer_bilinear(
            mosaic, p, origin_x, origin_y);
        // Interior pixels must be unaffected by border clamping.
        for (int y = 2; y < mosaic.rows() - 2; ++y) {
          for (int x = 2; x < mosaic.cols() - 2; ++x) {
            REQUIRE(out.R(y, x) == Catch::Approx(Rv).margin(1.0));
            REQUIRE(out.G(y, x) == Catch::Approx(Gv).margin(1.0));
            REQUIRE(out.B(y, x) == Catch::Approx(Bv).margin(1.0));
          }
        }
      }
    }
  }
}

TEST_CASE("debayer_bilinear_wrong_origin_swaps_channels") {
  using tile_compile::Matrix2Df;
  using tile_compile::BayerPattern;

  // Guard against regressions to a forced (0,0) origin: for a mosaic whose
  // true origin is odd, forcing (0,0) must detectably corrupt the channels
  // (G clearly wrong on interior pixels of a constant-color CFA).
  const float Rv = 100.0f;
  const float Gv = 200.0f;
  const float Bv = 300.0f;

  const BayerPattern p = BayerPattern::GBRG;
  auto off = offsets_for(p);
  const int origin_y = 1;
  const int origin_x = 1;

  Matrix2Df mosaic(6, 6);
  for (int y = 0; y < mosaic.rows(); ++y) {
    for (int x = 0; x < mosaic.cols(); ++x) {
      int py = (origin_y + y) & 1;
      int px = (origin_x + x) & 1;
      if (py == off.r_row && px == off.r_col) {
        mosaic(y, x) = Rv;
      } else if (py == off.b_row && px == off.b_col) {
        mosaic(y, x) = Bv;
      } else {
        mosaic(y, x) = Gv;
      }
    }
  }

  auto wrong = tile_compile::image::debayer_bilinear(mosaic, p, 0, 0);
  // Flipped parity swaps R and B (G stays correct by symmetry of the
  // constant-CFA average), so detect the swap on R and B directly.
  const float err_r = (wrong.R.block(2, 2, 2, 2).array() - Rv).abs().maxCoeff();
  const float err_b = (wrong.B.block(2, 2, 2, 2).array() - Bv).abs().maxCoeff();
  REQUIRE(err_r > 50.0f);
  REQUIRE(err_b > 50.0f);
}

TEST_CASE("debayer_opencv_wrong_origin_swaps_channels") {
  using tile_compile::Matrix2Df;
  using tile_compile::BayerPattern;

  const float Rv = 100.0f;
  const float Gv = 200.0f;
  const float Bv = 300.0f;
  const BayerPattern p = BayerPattern::GBRG;
  const auto off = offsets_for(p);
  const int origin_y = 1;
  const int origin_x = 1;
  Matrix2Df mosaic(8, 8);
  for (int y = 0; y < mosaic.rows(); ++y) {
    for (int x = 0; x < mosaic.cols(); ++x) {
      const int py = (origin_y + y) & 1;
      const int px = (origin_x + x) & 1;
      if (py == off.r_row && px == off.r_col)
        mosaic(y, x) = Rv;
      else if (py == off.b_row && px == off.b_col)
        mosaic(y, x) = Bv;
      else
        mosaic(y, x) = Gv;
    }
  }

  const auto wrong = tile_compile::image::debayer_opencv(mosaic, p, 0, 0,
                                                           true);
  const float err_r =
      (wrong.R.block(3, 3, 2, 2).array() - Rv).abs().maxCoeff();
  const float err_b =
      (wrong.B.block(3, 3, 2, 2).array() - Bv).abs().maxCoeff();
  REQUIRE(err_r > 50.0f);
  REQUIRE(err_b > 50.0f);
}

TEST_CASE("debayer_opencv_respects_tile_origin_parity") {
  using tile_compile::Matrix2Df;
  using tile_compile::BayerPattern;

  const float Rv = 100.0f;
  const float Gv = 200.0f;
  const float Bv = 300.0f;

  for (BayerPattern p : {BayerPattern::RGGB, BayerPattern::BGGR, BayerPattern::GRBG,
                        BayerPattern::GBRG}) {
    auto off = offsets_for(p);

    for (int origin_y = 0; origin_y < 2; ++origin_y) {
      for (int origin_x = 0; origin_x < 2; ++origin_x) {
        Matrix2Df mosaic(8, 8);
        for (int y = 0; y < mosaic.rows(); ++y) {
          for (int x = 0; x < mosaic.cols(); ++x) {
            int py = (origin_y + y) & 1;
            int px = (origin_x + x) & 1;
            if (py == off.r_row && px == off.r_col) {
              mosaic(y, x) = Rv;
            } else if (py == off.b_row && px == off.b_col) {
              mosaic(y, x) = Bv;
            } else {
              mosaic(y, x) = Gv;
            }
          }
        }

        for (bool ahd : {false, true}) {
          auto out = tile_compile::image::debayer_opencv(
              mosaic, p, origin_x, origin_y, ahd);
          for (int y = 3; y < mosaic.rows() - 3; ++y) {
            for (int x = 3; x < mosaic.cols() - 3; ++x) {
              REQUIRE(out.R(y, x) == Catch::Approx(Rv).margin(2.0));
              REQUIRE(out.G(y, x) == Catch::Approx(Gv).margin(2.0));
              REQUIRE(out.B(y, x) == Catch::Approx(Bv).margin(2.0));
            }
          }
        }
      }
    }
  }
}

#else
int tile_compile_tests_debayer_stub() { return 0; }
#endif
