#include "tile_compile/registration/global_registration.hpp"

#if __has_include(<catch2/catch_test_macros.hpp>)
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

namespace tile_compile::registration {

namespace {

Matrix2Df make_registration_pattern(int rows, int cols) {
  Matrix2Df img(rows, cols);
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      const float xf = static_cast<float>(x);
      const float yf = static_cast<float>(y);
      img(y, x) = 0.1f * xf + 0.07f * yf +
                  0.5f * std::sin(0.17f * xf) +
                  0.3f * std::cos(0.11f * yf);
    }
  }
  return img;
}

} // namespace

TEST_CASE("register_single_frame accepts near-perfect identity directly") {
  const Matrix2Df ref = make_registration_pattern(48, 64);
  const Matrix2Df mov = ref;
  const config::RegistrationConfig rcfg;

  const auto res = register_single_frame(mov, ref, rcfg);

  REQUIRE(res.reg.success);
  REQUIRE(res.method_used == "identity");
  REQUIRE(res.ncc_identity > 0.999f);
  REQUIRE(res.ncc_warped == Catch::Approx(res.ncc_identity).margin(1.0e-6));
  REQUIRE(res.reg.correlation == Catch::Approx(res.ncc_identity).margin(1.0e-6));
  REQUIRE(std::fabs(res.reg.warp(0, 0) - 1.0f) < 1.0e-6f);
  REQUIRE(std::fabs(res.reg.warp(0, 1)) < 1.0e-6f);
  REQUIRE(std::fabs(res.reg.warp(0, 2)) < 1.0e-6f);
  REQUIRE(std::fabs(res.reg.warp(1, 0)) < 1.0e-6f);
  REQUIRE(std::fabs(res.reg.warp(1, 1) - 1.0f) < 1.0e-6f);
  REQUIRE(std::fabs(res.reg.warp(1, 2)) < 1.0e-6f);
}

TEST_CASE("register_frames_to_reference does not mark accepted identity as failure") {
  const Matrix2Df frame = make_registration_pattern(48, 64);
  const std::vector<Matrix2Df> frames{frame, frame};
  const config::RegistrationConfig rcfg;

  const auto out = register_frames_to_reference(frames, ColorMode::MONO,
                                                BayerPattern::UNKNOWN, rcfg);

  REQUIRE(out.success.size() == 2);
  REQUIRE(out.success[0]);
  REQUIRE(out.success[1]);
  REQUIRE(out.ref_idx == 1);
  REQUIRE(out.scores[0] > 0.999f);
  REQUIRE(out.scores[1] == Catch::Approx(1.0f).margin(1.0e-6));
  REQUIRE(std::fabs(out.warps_fullres[0](0, 2)) < 1.0e-6f);
  REQUIRE(std::fabs(out.warps_fullres[0](1, 2)) < 1.0e-6f);
}

} // namespace tile_compile::registration
#endif
