#include "../apps/runner_registration_refinement_state.hpp"
#include "tile_compile/registration/global_registration.hpp"
#include "tile_compile/registration/registration.hpp"

#if __has_include(<catch2/catch_test_macros.hpp>)
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
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

cv::Point2f evaluate_expected_smooth_local(
    const SmoothLocalWarpModel &model, float x, float y) {
  if (!model.valid || model.image_rows <= 1 || model.image_cols <= 1 ||
      x < 0.0f || y < 0.0f ||
      x > static_cast<float>(model.image_cols - 1) ||
      y > static_cast<float>(model.image_rows - 1)) {
    return {};
  }
  const float nx = x / static_cast<float>(model.image_cols - 1);
  const float ny = y / static_cast<float>(model.image_rows - 1);
  const float edge_distance = std::min({nx, 1.0f - nx, ny, 1.0f - ny});
  const float taper_t = std::clamp(edge_distance / 0.08f, 0.0f, 1.0f);
  const float taper = taper_t * taper_t * (3.0f - 2.0f * taper_t);
  constexpr float inverse_two_sigma_squared =
      1.0f / (2.0f * 0.28f * 0.28f);
  float weighted_x = 0.0f;
  float weighted_y = 0.0f;
  float weight_sum = 0.0f;
  int index = 0;
  for (int gy = 0; gy < 4; ++gy) {
    const float center_y = static_cast<float>(gy) / 3.0f;
    for (int gx = 0; gx < 4; ++gx, ++index) {
      const float center_x = static_cast<float>(gx) / 3.0f;
      const float dx = nx - center_x;
      const float dy = ny - center_y;
      const float weight =
          std::exp(-(dx * dx + dy * dy) * inverse_two_sigma_squared);
      weight_sum += weight;
      weighted_x += weight * model.coeff_x[index];
      weighted_y += weight * model.coeff_y[index];
    }
  }
  if (weight_sum <= 1.0e-8f) {
    return {};
  }
  return {taper * weighted_x / weight_sum,
          taper * weighted_y / weight_sum};
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

TEST_CASE("smooth local refinement improves training and held-out stars") {
  constexpr int rows = 300;
  constexpr int cols = 400;
  SmoothLocalWarpModel truth;
  truth.valid = true;
  truth.image_rows = rows;
  truth.image_cols = cols;
  for (int gy = 0; gy < 4; ++gy) {
    for (int gx = 0; gx < 4; ++gx) {
      const int index = gy * 4 + gx;
      truth.coeff_x[index] =
          0.38f + 0.05f * static_cast<float>(gx) -
          0.03f * static_cast<float>(gy);
      truth.coeff_y[index] =
          -0.31f + 0.04f * static_cast<float>(gy) -
          0.02f * static_cast<float>(gx);
    }
  }
  cv::Mat displacement_x;
  cv::Mat displacement_y;
  render_smooth_local_displacement(truth, rows, cols, 1.0f, 0.0f, 0.0f,
                                   displacement_x, displacement_y);

  std::vector<StarPoint> ref_stars;
  std::vector<StarPoint> warped_stars;
  for (int gy = 0; gy < 8; ++gy) {
    for (int gx = 0; gx < 8; ++gx) {
      const int x = 35 + gx * 47;
      const int y = 28 + gy * 34;
      ref_stars.push_back(
          {static_cast<float>(x), static_cast<float>(y), 100.0f});
      warped_stars.push_back(
          {static_cast<float>(x) + displacement_x.at<float>(y, x),
           static_cast<float>(y) + displacement_y.at<float>(y, x), 100.0f});
    }
  }

  const auto fit = estimate_smooth_local_star_refinement(
      ref_stars, warped_stars, rows, cols, 3.0f);

  REQUIRE(fit.valid);
  REQUIRE(fit.rejection_reason == "accepted");
  REQUIRE(fit.matched_stars == 64);
  REQUIRE(fit.training_stars == 48);
  REQUIRE(fit.validation_stars == 16);
  REQUIRE(fit.spatial_coverage > 0.5f);
  REQUIRE(fit.median_after_px < fit.median_before_px);
  REQUIRE(fit.p90_after_px < fit.p90_before_px);
  REQUIRE(fit.validation_median_after_px <
          fit.validation_median_before_px);
  REQUIRE(fit.validation_p90_after_px < fit.validation_p90_before_px);
  REQUIRE(fit.max_displacement_px < 1.5f);
  REQUIRE(fit.min_jacobian_determinant >= 0.94f);
  REQUIRE(fit.max_jacobian_determinant <= 1.06f);
}

TEST_CASE("smooth local renderer preserves coordinate scaling and offset") {
  SmoothLocalWarpModel model;
  model.valid = true;
  model.image_rows = 120;
  model.image_cols = 160;
  model.coeff_x.setConstant(0.4f);
  model.coeff_y.setConstant(-0.25f);

  cv::Mat proxy_x;
  cv::Mat proxy_y;
  cv::Mat full_x;
  cv::Mat full_y;
  cv::Mat offset_x;
  cv::Mat offset_y;
  render_smooth_local_displacement(model, 120, 160, 1.0f, 0.0f, 0.0f,
                                   proxy_x, proxy_y);
  render_smooth_local_displacement(model, 240, 320, 0.5f, 0.0f, 0.0f,
                                   full_x, full_y);
  render_smooth_local_displacement(model, 280, 360, 0.5f, 20.0f, 10.0f,
                                   offset_x, offset_y);

  REQUIRE(full_x.at<float>(120, 160) ==
          Catch::Approx(2.0f * proxy_x.at<float>(60, 80)).margin(0.02f));
  REQUIRE(full_y.at<float>(120, 160) ==
          Catch::Approx(2.0f * proxy_y.at<float>(60, 80)).margin(0.02f));
  REQUIRE(offset_x.at<float>(130, 180) == Catch::Approx(0.8f).margin(0.02f));
  REQUIRE(offset_y.at<float>(130, 180) == Catch::Approx(-0.5f).margin(0.02f));
  REQUIRE(std::fabs(full_x.at<float>(0, 0)) < 1.0e-6f);
  REQUIRE(std::fabs(full_y.at<float>(0, 0)) < 1.0e-6f);

  SmoothLocalWarpModel spatial_model = model;
  for (int index = 0; index < spatial_model.coeff_x.size(); ++index) {
    spatial_model.coeff_x[index] = -0.2f + 0.03f * index;
    spatial_model.coeff_y[index] = 0.15f - 0.02f * index;
  }
  cv::Mat spatial_x;
  cv::Mat spatial_y;
  constexpr float spatial_scale = 0.5f;
  constexpr float spatial_offset_x = 20.0f;
  constexpr float spatial_offset_y = 10.0f;
  constexpr int sample_x = 179;
  constexpr int sample_y = 129;
  render_smooth_local_displacement(
      spatial_model, 280, 360, spatial_scale, spatial_offset_x,
      spatial_offset_y, spatial_x, spatial_y);
  const cv::Point2f expected = evaluate_expected_smooth_local(
      spatial_model, (sample_x - spatial_offset_x) * spatial_scale,
      (sample_y - spatial_offset_y) * spatial_scale);
  REQUIRE(spatial_x.at<float>(sample_y, sample_x) ==
          Catch::Approx(expected.x / spatial_scale).margin(1.0e-6f));
  REQUIRE(spatial_y.at<float>(sample_y, sample_x) ==
          Catch::Approx(expected.y / spatial_scale).margin(1.0e-6f));
}

TEST_CASE("smooth local prepared remap composes maps and shares RGB support") {
  constexpr int source_rows = 220;
  constexpr int source_cols = 280;
  constexpr int output_rows = 180;
  constexpr int output_cols = 240;
  SmoothLocalWarpModel model;
  model.valid = true;
  model.image_rows = 120;
  model.image_cols = 160;
  model.coeff_x.setConstant(0.2f);
  model.coeff_y.setConstant(-0.1f);

  WarpMatrix inverse_warp;
  inverse_warp << 1.0f, 0.01f, -8.0f, -0.02f, 1.0f, -6.0f;
  SmoothLocalRemapPlan plan;
  REQUIRE(prepare_smooth_local_remap(
      source_rows, source_cols, inverse_warp, model, output_rows, output_cols,
      0.5f, 20.0f, 10.0f, plan));
  REQUIRE(plan.valid_mask.size() ==
          static_cast<size_t>(output_rows * output_cols));

  constexpr int sample_x = 120;
  constexpr int sample_y = 90;
  constexpr float corrected_x = static_cast<float>(sample_x) + 0.4f;
  constexpr float corrected_y = static_cast<float>(sample_y) - 0.2f;
  const float expected_map_x = corrected_x + 0.01f * corrected_y - 8.0f;
  const float expected_map_y = -0.02f * corrected_x + corrected_y - 6.0f;
  REQUIRE(plan.map_x.at<float>(sample_y, sample_x) ==
          Catch::Approx(expected_map_x).margin(0.02f));
  REQUIRE(plan.map_y.at<float>(sample_y, sample_x) ==
          Catch::Approx(expected_map_y).margin(0.02f));

  Matrix2Df red = make_registration_pattern(source_rows, source_cols);
  Matrix2Df green = 2.0f * red;
  Matrix2Df blue = 0.5f * red;
  Matrix2Df warped_red;
  Matrix2Df warped_green;
  Matrix2Df warped_blue;
  bool has_red = false;
  bool has_green = false;
  bool has_blue = false;
  REQUIRE(remap_frame_with_smooth_local_plan(red, plan, "linear", warped_red,
                                             &has_red));
  REQUIRE(remap_frame_with_smooth_local_plan(
      green, plan, "linear", warped_green, &has_green));
  REQUIRE(remap_frame_with_smooth_local_plan(
      blue, plan, "linear", warped_blue, &has_blue));
  REQUIRE(has_red);
  REQUIRE(has_green);
  REQUIRE(has_blue);

  Matrix2Df warped_mono;
  std::vector<uint8_t> mono_mask;
  bool has_mono = false;
  REQUIRE(warp_frame_with_smooth_local_model(
      red, inverse_warp, model, output_rows, output_cols, 0.5f, 20.0f,
      10.0f, "linear", warped_mono, &mono_mask, &has_mono));
  REQUIRE(has_mono);
  REQUIRE(mono_mask == plan.valid_mask);

  int valid_pixels = 0;
  int invalid_pixels = 0;
  bool invalid_pixels_are_nan = true;
  bool valid_pixels_are_finite = true;
  float max_mono_parity_error = 0.0f;
  float max_green_scaling_error = 0.0f;
  float max_blue_scaling_error = 0.0f;
  for (int y = 0; y < output_rows; ++y) {
    for (int x = 0; x < output_cols; ++x) {
      const size_t index = static_cast<size_t>(y) * output_cols + x;
      if (plan.valid_mask[index] == 0) {
        ++invalid_pixels;
        invalid_pixels_are_nan =
            invalid_pixels_are_nan && !std::isfinite(warped_red(y, x)) &&
            !std::isfinite(warped_green(y, x)) &&
            !std::isfinite(warped_blue(y, x)) &&
            !std::isfinite(warped_mono(y, x));
      } else {
        ++valid_pixels;
        valid_pixels_are_finite =
            valid_pixels_are_finite && std::isfinite(warped_red(y, x)) &&
            std::isfinite(warped_green(y, x)) &&
            std::isfinite(warped_blue(y, x)) &&
            std::isfinite(warped_mono(y, x));
        max_mono_parity_error =
            std::max(max_mono_parity_error,
                     std::fabs(warped_mono(y, x) - warped_red(y, x)));
        max_green_scaling_error = std::max(
            max_green_scaling_error,
            std::fabs(warped_green(y, x) - 2.0f * warped_red(y, x)));
        max_blue_scaling_error = std::max(
            max_blue_scaling_error,
            std::fabs(warped_blue(y, x) - 0.5f * warped_red(y, x)));
      }
    }
  }
  REQUIRE(valid_pixels > 0);
  REQUIRE(invalid_pixels > 0);
  REQUIRE(invalid_pixels_are_nan);
  REQUIRE(valid_pixels_are_finite);
  REQUIRE(max_mono_parity_error < 1.0e-6f);
  REQUIRE(max_green_scaling_error < 1.0e-4f);
  REQUIRE(max_blue_scaling_error < 1.0e-4f);
}

TEST_CASE("smooth local prepared remap rejects full-resolution geometry") {
  SmoothLocalWarpModel model;
  model.valid = true;
  model.image_rows = 120;
  model.image_cols = 160;
  model.coeff_x.setConstant(1.0f);
  model.coeff_y.setZero();
  WarpMatrix identity;
  identity << 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f;
  SmoothLocalRemapPlan plan;

  REQUIRE_FALSE(prepare_smooth_local_remap(
      240, 320, identity, model, 280, 360, 0.5f, 20.0f, 10.0f, plan));
  REQUIRE_FALSE(plan.has_data);
}

TEST_CASE("smooth local refinement rejects spatially concentrated matches") {
  std::vector<StarPoint> ref_stars;
  std::vector<StarPoint> warped_stars;
  for (int gy = 0; gy < 4; ++gy) {
    for (int gx = 0; gx < 8; ++gx) {
      const float x = 150.0f + gx * 7.0f;
      const float y = 110.0f + gy * 7.0f;
      ref_stars.push_back({x, y, 100.0f});
      warped_stars.push_back({x + 0.4f, y - 0.25f, 100.0f});
    }
  }

  const auto fit = estimate_smooth_local_star_refinement(
      ref_stars, warped_stars, 300, 400, 3.0f);

  REQUIRE_FALSE(fit.valid);
  REQUIRE(fit.rejection_reason == "insufficient_spatial_coverage");
}

TEST_CASE("smooth local refinement independently rejects displacement") {
  std::vector<StarPoint> ref_stars;
  std::vector<StarPoint> warped_stars;
  for (int gy = 0; gy < 8; ++gy) {
    for (int gx = 0; gx < 8; ++gx) {
      const float x = 35.0f + gx * 47.0f;
      const float y = 28.0f + gy * 34.0f;
      ref_stars.push_back({x, y, 100.0f});
      warped_stars.push_back({x + 2.2f, y, 100.0f});
    }
  }

  const auto fit = estimate_smooth_local_star_refinement(
      ref_stars, warped_stars, 300, 400, 3.0f);

  REQUIRE_FALSE(fit.valid);
  REQUIRE(fit.max_displacement_px > 1.5f);
  REQUIRE(fit.rejection_reason == "displacement_out_of_bounds");
}

TEST_CASE("smooth local refinement independently rejects Jacobian distortion") {
  std::vector<StarPoint> ref_stars;
  std::vector<StarPoint> warped_stars;
  for (int gy = 0; gy < 8; ++gy) {
    for (int gx = 0; gx < 8; ++gx) {
      const float x = 35.0f + gx * 47.0f;
      const float y = 28.0f + gy * 34.0f;
      ref_stars.push_back({x, y, 100.0f});
      warped_stars.push_back({x + 1.0f, y, 100.0f});
    }
  }

  const auto fit = estimate_smooth_local_star_refinement(
      ref_stars, warped_stars, 300, 400, 3.0f);

  REQUIRE_FALSE(fit.valid);
  REQUIRE(fit.max_displacement_px < 1.5f);
  REQUIRE(fit.rejection_reason == "jacobian_out_of_bounds");
}

TEST_CASE("smooth local refinement rejects held-out tail regression") {
  std::vector<StarPoint> ref_stars;
  std::vector<StarPoint> warped_stars;
  int sorted_index = 0;
  for (int gy = 0; gy < 8; ++gy) {
    for (int gx = 0; gx < 8; ++gx, ++sorted_index) {
      const float x = 35.0f + gx * 47.0f;
      const float y = 28.0f + gy * 34.0f;
      float dx = 0.5f;
      if (sorted_index == 1 || sorted_index == 5) {
        dx = -1.5f;
      }
      ref_stars.push_back({x, y, 100.0f});
      warped_stars.push_back({x + dx, y, 100.0f});
    }
  }

  const auto fit = estimate_smooth_local_star_refinement(
      ref_stars, warped_stars, 300, 400, 3.0f);

  REQUIRE_FALSE(fit.valid);
  REQUIRE((fit.rejection_reason == "validation_p90_not_improved" ||
           fit.rejection_reason == "validation_rms_regressed"));
}

TEST_CASE("runner refinement proxy cache commits accepted candidates only") {
  tile_compile::runner::detail::RefinementProxyCache cache;
  cache.proxy = Matrix2Df::Constant(2, 3, 1.0f);
  cache.stars = {{1.0f, 2.0f, 3.0f}};
  Matrix2Df rejected_candidate = Matrix2Df::Constant(2, 3, 2.0f);
  int detector_calls = 0;
  const auto detector = [&](const Matrix2Df &proxy) {
    ++detector_calls;
    return std::vector<StarPoint>{{0.0f, 0.0f, proxy.sum()}};
  };

  REQUIRE_FALSE(cache.commit_candidate(false, rejected_candidate, detector));
  REQUIRE(detector_calls == 0);
  REQUIRE(cache.proxy.sum() == Catch::Approx(6.0f));
  REQUIRE(cache.stars.front().flux == Catch::Approx(3.0f));

  Matrix2Df accepted_candidate = Matrix2Df::Constant(2, 3, 4.0f);
  REQUIRE(cache.commit_candidate(true, std::move(accepted_candidate),
                                 detector));
  REQUIRE(detector_calls == 1);
  REQUIRE(cache.proxy.sum() == Catch::Approx(24.0f));
  REQUIRE(cache.stars.size() == 1);
  REQUIRE(cache.stars.front().flux == Catch::Approx(24.0f));
}

TEST_CASE("runner refinement rollback restores frame and aggregate state") {
  using tile_compile::runner::detail::AffineRefinementFrameStats;
  using tile_compile::runner::detail::RefinementAggregateState;
  using tile_compile::runner::detail::RegistrationResidualStats;
  using tile_compile::runner::detail::SmoothLocalRefinementFrameStats;

  WarpMatrix original_warp;
  original_warp << 1.0f, 0.0f, 2.0f, 0.0f, 1.0f, -3.0f;
  WarpMatrix warp = original_warp;
  RefinementAggregateState aggregate;
  aggregate.residual_applicable = 4;
  aggregate.residual_damped = 2;
  aggregate.affine_attempted = 3;
  aggregate.affine_applied = 2;
  aggregate.affine_rejected = 1;
  aggregate.local_attempted = 2;
  aggregate.local_applied = 1;
  aggregate.local_rejected = 1;
  aggregate.residual_medians = {0.2f, 0.3f};
  aggregate.residual_p90s = {0.6f, 0.7f};
  aggregate.residual_factors = {0.9f, 0.8f};
  const auto snapshot =
      tile_compile::runner::detail::make_refinement_rollback_snapshot(
          warp, aggregate);

  float residual_weight = 0.5f;
  RegistrationResidualStats residual_stats;
  residual_stats.applicable = true;
  AffineRefinementFrameStats affine;
  affine.attempted = true;
  affine.applied = true;
  affine.reason = "applied";
  SmoothLocalRefinementFrameStats local;
  local.attempted = true;
  local.applied = true;
  local.reason = "applied";
  local.fit.model.valid = true;
  warp(0, 2) = 9.0f;
  ++aggregate.residual_applicable;
  ++aggregate.residual_damped;
  ++aggregate.affine_attempted;
  ++aggregate.affine_applied;
  ++aggregate.local_attempted;
  ++aggregate.local_applied;
  aggregate.residual_medians.push_back(1.0f);
  aggregate.residual_p90s.push_back(1.0f);
  aggregate.residual_factors.push_back(0.5f);

  tile_compile::runner::detail::rollback_refinement_frame(
      snapshot, warp, residual_weight, residual_stats, affine, local,
      aggregate);

  REQUIRE((warp - original_warp).norm() < 1.0e-6f);
  REQUIRE(residual_weight == Catch::Approx(1.0f));
  REQUIRE_FALSE(residual_stats.applicable);
  REQUIRE(aggregate.residual_applicable == 4);
  REQUIRE(aggregate.residual_damped == 2);
  REQUIRE(aggregate.affine_attempted == 4);
  REQUIRE(aggregate.affine_applied == 2);
  REQUIRE(aggregate.affine_rejected == 2);
  REQUIRE(aggregate.local_attempted == 3);
  REQUIRE(aggregate.local_applied == 1);
  REQUIRE(aggregate.local_rejected == 2);
  REQUIRE(aggregate.residual_medians.size() == 2);
  REQUIRE(aggregate.residual_p90s.size() == 2);
  REQUIRE(aggregate.residual_factors.size() == 2);
  REQUIRE_FALSE(affine.applied);
  REQUIRE(affine.reason == "exception");
  REQUIRE_FALSE(local.applied);
  REQUIRE_FALSE(local.fit.model.valid);
  REQUIRE(local.reason == "exception");
}

} // namespace tile_compile::registration
#endif
