#include "tile_compile/registration/global_registration.hpp"
#include "tile_compile/registration/registration.hpp"

#if __has_include(<catch2/catch_test_macros.hpp>)
#include <catch2/catch_approx.hpp>
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

float compute_test_ncc(const Matrix2Df &a, const Matrix2Df &b) {
  REQUIRE(a.rows() == b.rows());
  REQUIRE(a.cols() == b.cols());
  const int n = a.size();
  REQUIRE(n > 1);

  double ma = 0.0;
  double mb = 0.0;
  for (int i = 0; i < n; ++i) {
    ma += a.data()[i];
    mb += b.data()[i];
  }
  ma /= static_cast<double>(n);
  mb /= static_cast<double>(n);

  double sab = 0.0;
  double saa = 0.0;
  double sbb = 0.0;
  for (int i = 0; i < n; ++i) {
    const double va = static_cast<double>(a.data()[i]) - ma;
    const double vb = static_cast<double>(b.data()[i]) - mb;
    sab += va * vb;
    saa += va * va;
    sbb += vb * vb;
  }
  const double den = std::sqrt(saa * sbb);
  return (den > 1.0e-10) ? static_cast<float>(sab / den) : 0.0f;
}

Matrix2Df make_star_field(int rows, int cols,
                          const std::vector<std::pair<float, float>> &stars) {
  Matrix2Df img = Matrix2Df::Zero(rows, cols);
  for (const auto &[cx, cy] : stars) {
    for (int y = std::max(0, static_cast<int>(std::floor(cy - 4.0f)));
         y < std::min(rows, static_cast<int>(std::ceil(cy + 5.0f))); ++y) {
      for (int x = std::max(0, static_cast<int>(std::floor(cx - 4.0f)));
           x < std::min(cols, static_cast<int>(std::ceil(cx + 5.0f))); ++x) {
        const float dx = static_cast<float>(x) - cx;
        const float dy = static_cast<float>(y) - cy;
        img(y, x) += 180.0f * std::exp(-(dx * dx + dy * dy) / 2.8f);
      }
    }
  }
  return img;
}

std::vector<std::pair<float, float>> make_dense_ring_star_positions() {
  std::vector<std::pair<float, float>> stars;
  stars.reserve(24);
  for (int i = 0; i < 16; ++i) {
    const float ang =
        static_cast<float>(i) * 2.0f * static_cast<float>(M_PI) / 16.0f;
    const float radius = 46.0f + static_cast<float>((i % 3) - 1) * 2.5f;
    stars.emplace_back(64.0f + radius * std::cos(ang),
                       64.0f + radius * std::sin(ang));
  }
  for (int i = 0; i < 8; ++i) {
    const float ang =
        static_cast<float>(i) * 2.0f * static_cast<float>(M_PI) / 8.0f + 0.17f;
    const float radius = 22.0f + static_cast<float>(i % 2) * 3.0f;
    stars.emplace_back(64.0f + radius * std::cos(ang),
                       64.0f + radius * std::sin(ang));
  }
  return stars;
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

TEST_CASE("triangle star matching remains stable on dense near-symmetric fields") {
  const auto stars = make_dense_ring_star_positions();
  const Matrix2Df ref = make_star_field(128, 128, stars);

  const float theta = 0.11f;
  const float ct = std::cos(theta);
  const float st = std::sin(theta);
  std::vector<std::pair<float, float>> moved_stars;
  moved_stars.reserve(stars.size());
  for (const auto &[x, y] : stars) {
    const float dx = x - 64.0f;
    const float dy = y - 64.0f;
    moved_stars.emplace_back(64.0f + ct * dx - st * dy + 6.0f,
                             64.0f + st * dx + ct * dy - 4.0f);
  }
  const Matrix2Df mov = make_star_field(128, 128, moved_stars);

  const auto res =
      triangle_star_matching(mov, ref, true, 64, 4, 4.0f, "affine");

  REQUIRE(res.success);

  const Matrix2Df warped = apply_warp(mov, res.warp);
  REQUIRE(compute_test_ncc(warped, ref) > 0.85f);
}

} // namespace tile_compile::registration
#endif
