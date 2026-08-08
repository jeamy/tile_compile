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

TEST_CASE("phase correlation displacement is inverted for apply_warp") {
  const int rows = 96;
  const int cols = 128;
  const Matrix2Df moving = make_star_field(
      rows, cols, {{22.0f, 24.0f}, {51.0f, 68.0f}, {93.0f, 37.0f}});

  // Build a reference whose content moved +7 px in x and -3 px in y.
  const int content_dx = 7;
  const int content_dy = -3;
  Matrix2Df ref = Matrix2Df::Zero(rows, cols);
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      const int rx = x + content_dx;
      const int ry = y + content_dy;
      if (rx >= 0 && rx < cols && ry >= 0 && ry < rows) {
        ref(ry, rx) = moving(y, x);
      }
    }
  }

  const auto [dx, dy] = phasecorr_translation(moving, ref);
  REQUIRE(dx == Catch::Approx(static_cast<float>(content_dx)).margin(0.25f));
  REQUIRE(dy == Catch::Approx(static_cast<float>(content_dy)).margin(0.25f));

  // apply_warp uses WARP_INVERSE_MAP, therefore the phase-correlation
  // displacement must be negated before it is placed in the warp matrix.
  WarpMatrix inverse_shift = identity_warp();
  inverse_shift(0, 2) = -dx;
  inverse_shift(1, 2) = -dy;
  const Matrix2Df aligned = apply_warp(moving, inverse_shift);
  REQUIRE(compute_test_ncc(aligned, ref) > 0.95f);
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

TEST_CASE("affine star refinement robustly recovers a small residual warp") {
  constexpr int rows = 300;
  constexpr int cols = 400;
  constexpr double theta = 0.0025;
  const double a00 = 1.003 * std::cos(theta);
  const double a01 = -std::sin(theta) + 0.0008;
  const double a10 = std::sin(theta);
  const double a11 = 0.998 * std::cos(theta);
  const double cx = 0.5 * (cols - 1);
  const double cy = 0.5 * (rows - 1);
  const double tx = cx + 0.45 - a00 * cx - a01 * cy;
  const double ty = cy - 0.35 - a10 * cx - a11 * cy;
  const double det = a00 * a11 - a01 * a10;

  std::vector<StarPoint> ref_stars;
  std::vector<StarPoint> warped_stars;
  for (int gy = 0; gy < 6; ++gy) {
    for (int gx = 0; gx < 6; ++gx) {
      const double rx = 40.0 + gx * 64.0;
      const double ry = 35.0 + gy * 46.0;
      const double ux = rx - tx;
      const double uy = ry - ty;
      double wx = (a11 * ux - a01 * uy) / det;
      double wy = (-a10 * ux + a00 * uy) / det;
      if (ref_stars.size() < 3) {
        wx += 2.0;
        wy -= 1.4;
      }
      ref_stars.push_back(
          {static_cast<float>(rx), static_cast<float>(ry), 100.0f});
      warped_stars.push_back(
          {static_cast<float>(wx), static_cast<float>(wy), 100.0f});
    }
  }

  const auto fit = estimate_affine_star_refinement(
      ref_stars, warped_stars, rows, cols, 3.0f);

  REQUIRE(fit.valid);
  REQUIRE(fit.rejection_reason == "accepted");
  REQUIRE(fit.matched_stars == 36);
  REQUIRE(fit.inlier_stars >= 30);
  REQUIRE(fit.spatial_coverage > 0.5f);
  REQUIRE(fit.median_after_px < fit.median_before_px);
  REQUIRE(fit.p90_after_px < fit.p90_before_px);

  const auto &ref = ref_stars.back();
  const auto &warped = warped_stars.back();
  const float predicted_x = fit.correction_warp(0, 0) * ref.x +
                            fit.correction_warp(0, 1) * ref.y +
                            fit.correction_warp(0, 2);
  const float predicted_y = fit.correction_warp(1, 0) * ref.x +
                            fit.correction_warp(1, 1) * ref.y +
                            fit.correction_warp(1, 2);
  REQUIRE(predicted_x == Catch::Approx(warped.x).margin(0.05f));
  REQUIRE(predicted_y == Catch::Approx(warped.y).margin(0.05f));
}

TEST_CASE("affine star refinement rejects spatially concentrated matches") {
  std::vector<StarPoint> ref_stars;
  std::vector<StarPoint> warped_stars;
  for (int gy = 0; gy < 5; ++gy) {
    for (int gx = 0; gx < 6; ++gx) {
      const float x = 100.0f + gx * 8.0f;
      const float y = 100.0f + gy * 8.0f;
      ref_stars.push_back({x, y, 100.0f});
      warped_stars.push_back({x - 0.4f, y + 0.2f, 100.0f});
    }
  }

  const auto fit =
      estimate_affine_star_refinement(ref_stars, warped_stars, 300, 400);

  REQUIRE_FALSE(fit.valid);
  REQUIRE(fit.rejection_reason == "insufficient_spatial_coverage");
}

} // namespace tile_compile::registration
#endif
