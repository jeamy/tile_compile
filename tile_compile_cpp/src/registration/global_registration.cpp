#include "tile_compile/registration/global_registration.hpp"

#include "tile_compile/core/utils.hpp"
#include "tile_compile/image/cfa_processing.hpp"
#include "tile_compile/registration/registration.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#if CV_MAJOR_VERSION >= 5
#include <opencv2/features.hpp>
#else
#include <opencv2/features2d.hpp>
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <unordered_map>

namespace tile_compile::registration {

/// @brief Implements downsample2x2 mean.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Matrix2Df downsample2x2_mean(const Matrix2Df &in) {
  const int h = in.rows();
  const int w = in.cols();
  const int h2 = h - (h % 2);
  const int w2 = w - (w % 2);
  const int out_h = std::max(1, h2 / 2);
  const int out_w = std::max(1, w2 / 2);
  Matrix2Df out(out_h, out_w);
  for (int y = 0; y < out_h; ++y) {
    for (int x = 0; x < out_w; ++x) {
      const int sy = y * 2;
      const int sx = x * 2;
      const float a = in(sy, sx);
      const float b = in(sy, sx + 1);
      const float c = in(sy + 1, sx);
      const float d = in(sy + 1, sx + 1);
      out(y, x) = 0.25f * (a + b + c + d);
    }
  }
  return out;
}

/// @brief Implements scale translation warp.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
WarpMatrix scale_translation_warp(const WarpMatrix &w, float scale) {
  WarpMatrix out = w;
  out(0, 2) *= scale;
  out(1, 2) *= scale;
  return out;
}

// Invert a 2×3 affine warp matrix: given M→R, return R→M (or vice versa).
/// @brief Inverts warp 2x3.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static WarpMatrix invert_warp_2x3(const WarpMatrix &w) {
  const float a00 = w(0, 0), a01 = w(0, 1), tx = w(0, 2);
  const float a10 = w(1, 0), a11 = w(1, 1), ty = w(1, 2);
  const float det = a00 * a11 - a01 * a10;
  if (std::fabs(det) < 1e-12f)
    return w; // degenerate — return as-is
  const float inv_det = 1.0f / det;
  WarpMatrix inv;
  inv(0, 0) = a11 * inv_det;
  inv(0, 1) = -a01 * inv_det;
  inv(1, 0) = -a10 * inv_det;
  inv(1, 1) = a00 * inv_det;
  inv(0, 2) = -(inv(0, 0) * tx + inv(0, 1) * ty);
  inv(1, 2) = -(inv(1, 0) * tx + inv(1, 1) * ty);
  return inv;
}

// --- Sub-functions (exported via header) ---

// Estimate rotation (deg) between ref and mov using log-polar phase correlation
// on magnitude spectrum.
/// @brief Estimates rotation logpolar.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float estimate_rotation_logpolar(const cv::Mat &ref, const cv::Mat &mov) {
  cv::Mat ref_dft, mov_dft;
  cv::dft(ref, ref_dft, cv::DFT_COMPLEX_OUTPUT);
  cv::dft(mov, mov_dft, cv::DFT_COMPLEX_OUTPUT);

  std::vector<cv::Mat> planes_ref, planes_mov;
  cv::split(ref_dft, planes_ref);
  cv::split(mov_dft, planes_mov);
  cv::Mat mag_ref, mag_mov;
  cv::magnitude(planes_ref[0], planes_ref[1], mag_ref);
  cv::magnitude(planes_mov[0], planes_mov[1], mag_mov);

  mag_ref += 1.0e-9f;
  mag_mov += 1.0e-9f;

  cv::Mat lp_ref, lp_mov;
  const cv::Point2f center(static_cast<float>(ref.cols) / 2.0f,
                           static_cast<float>(ref.rows) / 2.0f);
  const double M = ref.cols;
  cv::warpPolar(mag_ref, lp_ref, mag_ref.size(), center, M,
               static_cast<int>(cv::WARP_FILL_OUTLIERS) | cv::WARP_POLAR_LOG);
  cv::warpPolar(mag_mov, lp_mov, mag_mov.size(), center, M,
               static_cast<int>(cv::WARP_FILL_OUTLIERS) | cv::WARP_POLAR_LOG);

  cv::Point2d shift = cv::phaseCorrelate(lp_mov, lp_ref);
  double rotation_deg = -shift.y * 360.0 / static_cast<double>(lp_ref.rows);
  return static_cast<float>(rotation_deg);
}

/// @brief Converts uint8 stretch.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
cv::Mat to_uint8_stretch(const Matrix2Df &src) {
  cv::Mat f(src.rows(), src.cols(), CV_32F, const_cast<float *>(src.data()));
  std::vector<float> vals;
  vals.reserve(static_cast<size_t>(src.size()));
  for (int r = 0; r < f.rows; ++r) {
    const float *p = f.ptr<float>(r);
    vals.insert(vals.end(), p, p + f.cols);
  }
  if (vals.empty())
    return cv::Mat();
  const size_t n = vals.size();
  auto nth = [&](size_t k) {
    std::nth_element(vals.begin(), vals.begin() + k, vals.end());
    return vals[k];
  };
  float lo = nth(static_cast<size_t>(0.01 * n));
  float hi = nth(static_cast<size_t>(0.99 * n));
  if (hi <= lo)
    hi = lo + 1.0f;
  cv::Mat out;
  f.convertTo(out, CV_8U, 255.0 / (hi - lo), -255.0 * lo / (hi - lo));
  return out;
}

/// @brief Estimates affine family transform.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static cv::Mat estimate_affine_family_transform(
    const std::vector<cv::Point2f> &pts_mov,
    const std::vector<cv::Point2f> &pts_ref,
    const std::string &transform_model, cv::Mat &inliers,
    double ransac_threshold = 3.0, size_t max_iters = 2000,
    double confidence = 0.99) {
  if (transform_model == "affine") {
    return cv::estimateAffine2D(pts_mov, pts_ref, inliers, cv::RANSAC,
                                ransac_threshold, max_iters, confidence);
  }
  return cv::estimateAffinePartial2D(pts_mov, pts_ref, inliers, cv::RANSAC,
                                     ransac_threshold, max_iters, confidence);
}

/// @brief Inverts forward affine to warp.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static bool invert_forward_affine_to_warp(const cv::Mat &A,
                                          bool allow_rotation, WarpMatrix &warp,
                                          std::string *error_message = nullptr) {
  if (A.empty()) {
    if (error_message) {
      *error_message = "transform_fail";
    }
    return false;
  }

  float a00_fw = static_cast<float>(A.at<double>(0, 0));
  float a01_fw = static_cast<float>(A.at<double>(0, 1));
  float a10_fw = static_cast<float>(A.at<double>(1, 0));
  float a11_fw = static_cast<float>(A.at<double>(1, 1));
  float tx_fw = static_cast<float>(A.at<double>(0, 2));
  float ty_fw = static_cast<float>(A.at<double>(1, 2));

  if (!allow_rotation) {
    a00_fw = 1.0f;
    a01_fw = 0.0f;
    a10_fw = 0.0f;
    a11_fw = 1.0f;
  }

  const float det = a00_fw * a11_fw - a01_fw * a10_fw;
  if (std::fabs(det) < 1e-8f) {
    if (error_message) {
      *error_message = "singular_matrix";
    }
    return false;
  }
  const float inv_det = 1.0f / det;
  const float a00_inv = a11_fw * inv_det;
  const float a01_inv = -a01_fw * inv_det;
  const float a10_inv = -a10_fw * inv_det;
  const float a11_inv = a00_fw * inv_det;
  const float tx_inv = -(a00_inv * tx_fw + a01_inv * ty_fw);
  const float ty_inv = -(a10_inv * tx_fw + a11_inv * ty_fw);
  warp << a00_inv, a01_inv, tx_inv, a10_inv, a11_inv, ty_inv;
  return true;
}

/// @brief Implements feature registration similarity.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
RegistrationResult feature_registration_similarity(const Matrix2Df &mov,
                                                   const Matrix2Df &ref,
                                                   bool allow_rotation,
                                                   const std::string &transform_model) {
  cv::Mat ref_cv = to_uint8_stretch(ref);
  cv::Mat mov_cv = to_uint8_stretch(mov);

  RegistrationResult res;
  res.warp = identity_warp();
  res.correlation = 0.0f;
  res.success = false;

  if (ref_cv.empty() || mov_cv.empty()) {
    res.error_message = "empty image";
    return res;
  }

  cv::Ptr<cv::ORB> orb = cv::ORB::create();
  std::vector<cv::KeyPoint> kps_ref, kps_mov;
  cv::Mat desc_ref, desc_mov;
  orb->detectAndCompute(ref_cv, cv::noArray(), kps_ref, desc_ref);
  orb->detectAndCompute(mov_cv, cv::noArray(), kps_mov, desc_mov);

  if (desc_ref.empty() || desc_mov.empty()) {
    res.error_message = "no features";
    return res;
  }

  cv::BFMatcher matcher(cv::NORM_HAMMING, true);
  std::vector<cv::DMatch> matches;
  matcher.match(desc_mov, desc_ref, matches);
  if (matches.size() < 8) {
    res.error_message = "few matches";
    return res;
  }

  std::sort(matches.begin(), matches.end(),
            [](const cv::DMatch &a, const cv::DMatch &b) {
              return a.distance < b.distance;
            });
  const int keep = std::max<size_t>(15, matches.size() * 0.3);
  matches.resize(std::min<size_t>(static_cast<size_t>(keep), matches.size()));

  std::vector<cv::Point2f> pts_mov, pts_ref;
  pts_mov.reserve(matches.size());
  pts_ref.reserve(matches.size());
  for (const auto &m : matches) {
    pts_mov.push_back(kps_mov[m.queryIdx].pt);
    pts_ref.push_back(kps_ref[m.trainIdx].pt);
  }

  cv::Mat inliers;
  cv::Mat A = estimate_affine_family_transform(pts_mov, pts_ref,
                                               transform_model, inliers);
  if (A.empty()) {
    res.error_message = "transform_fail";
    return res;
  }
  if (!invert_forward_affine_to_warp(A, allow_rotation, res.warp,
                                     &res.error_message)) {
    return res;
  }
  int inl = inliers.empty() ? 0 : cv::countNonZero(inliers);
  res.correlation = matches.empty() ? 0.0f
                                    : static_cast<float>(inl) /
                                          static_cast<float>(matches.size());
  res.success = res.correlation > 0.1f;
  return res;
}

// Internal helper for star detection with configurable threshold
/// @brief Detects stars with threshold.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static std::vector<StarPoint> detect_stars_with_threshold(
    const Matrix2Df &img, int topk, float sigma_multiplier, 
    float med, float sigma) {
  const int h = img.rows();
  const int w = img.cols();
  const float thresh = med + sigma_multiplier * sigma;

  std::vector<StarPoint> stars;
  stars.reserve(static_cast<size_t>(topk) * 2);
  for (int y = 1; y < h - 1; ++y) {
    const float *row = img.data() + static_cast<size_t>(y) * w;
    for (int x = 1; x < w - 1; ++x) {
      const float v = row[x];
      if (v < thresh)
        continue;
      if (v <= row[x - 1] || v <= row[x + 1])
        continue;
        
      bool is_max = true;
      for (int dy = -1; dy <= 1 && is_max; ++dy) {
        if (dy == 0) continue; // Already checked horizontal neighbors
        const float *r2 = img.data() + static_cast<size_t>(y + dy) * w;
        if (r2[x - 1] >= v || r2[x] >= v || r2[x + 1] >= v) {
          is_max = false;
        }
      }
      if (!is_max)
        continue;

      // Compute flux-weighted centroid and hot pixel / elongation metrics
      // in a 5×5 neighborhood for better discrimination
      const int hw = 2; // half-width for analysis
      float wsum = 0.0f;
      float xs = 0.0f;
      float ys = 0.0f;
      float central_flux = row[x] - med;
      if (central_flux < 0.0f)
        central_flux = 0.0f;
      float Ixx = 0.0f, Iyy = 0.0f, Ixy = 0.0f;
      for (int dy = -hw; dy <= hw; ++dy) {
        if (y + dy < 0 || y + dy >= h)
          continue;
        const float *r2 = img.data() + static_cast<size_t>(y + dy) * w;
        for (int dx = -hw; dx <= hw; ++dx) {
          if (x + dx < 0 || x + dx >= w)
            continue;
          const float val = r2[x + dx] - med;
          if (val <= 0.0f)
            continue;
          wsum += val;
          xs += (static_cast<float>(x + dx) * val);
          ys += (static_cast<float>(y + dy) * val);
          Ixx += static_cast<float>(dx * dx) * val;
          Iyy += static_cast<float>(dy * dy) * val;
          Ixy += static_cast<float>(dx * dy) * val;
        }
      }
      if (wsum <= 0.0f)
        continue;

      // Hot pixel rejection: real stars spread flux over multiple pixels.
      // Hot pixels concentrate >80% in the central pixel.
      const float concentration = central_flux / wsum;
      if (concentration > 0.8f)
        continue;

      // Roundness filter via second moments: reject elongated sources
      if (Ixx + Iyy > 1e-6f) {
        float trace = Ixx + Iyy;
        float det2 = Ixx * Iyy - Ixy * Ixy;
        float disc = trace * trace - 4.0f * det2;
        if (disc < 0.0f) disc = 0.0f;
        float sqrt_disc = std::sqrt(disc);
        float lam1 = (trace + sqrt_disc) * 0.5f;
        float lam2 = (trace - sqrt_disc) * 0.5f;
        float roundness = (lam1 > 1e-6f) ? (lam2 / lam1) : 0.0f;
        if (roundness < 0.15f)
          continue; // too elongated (trail or artifact)
      }

      StarPoint s;
      s.x = xs / wsum;
      s.y = ys / wsum;
      s.flux = wsum;
      stars.push_back(s);
    }
  }
  std::sort(
      stars.begin(), stars.end(),
      [](const StarPoint &a, const StarPoint &b) { return a.flux > b.flux; });
  if (static_cast<int>(stars.size()) > topk) {
    stars.resize(static_cast<size_t>(topk));
  }
  return stars;
}

/// @brief Detects stars simple.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::vector<StarPoint> detect_stars_simple(const Matrix2Df &img, int topk,
                                           bool enable_local_background_subtraction) {
  const int h = img.rows();
  const int w = img.cols();
  if (h < 5 || w < 5)
    return {};

  // §4.4, §8.D — Lokale Hintergrundsubtraktion bei Gradienten/Mondlicht
  Matrix2Df processed_img;
  const Matrix2Df *img_ptr = &img;
  if (enable_local_background_subtraction) {
    // Lokale Hintergrundschätzung mit Box-Blur (31x31 Pixel)
    processed_img = img;  // Kopie
    const int kernel_size = 31;
    const int half_k = kernel_size / 2;
    // Einfacher Box-Blur für Hintergrundschätzung
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        float sum = 0.0f;
        int count = 0;
        for (int dy = -half_k; dy <= half_k; ++dy) {
          int py = std::clamp(y + dy, 0, h - 1);
          for (int dx = -half_k; dx <= half_k; ++dx) {
            int px = std::clamp(x + dx, 0, w - 1);
            sum += img(py, px);
            ++count;
          }
        }
        float background = sum / count;
        processed_img(y, x) = std::max(0.0f, img(y, x) - background);
      }
    }
    img_ptr = &processed_img;
  }

  std::vector<float> pixels;
  pixels.reserve(static_cast<size_t>(img_ptr->size()));
  for (int y = 0; y < h; ++y) {
    const float *row = img_ptr->data() + static_cast<size_t>(y) * w;
    pixels.insert(pixels.end(), row, row + w);
  }
  float med = core::median_of(pixels);
  float sigma = core::robust_sigma_mad(pixels);
  if (sigma < 1.0e-6f)
    sigma = 1.0f;

  // Try standard threshold (3.5σ) — mit optionaler Hintergrundsubtraktion
  std::vector<StarPoint> stars = detect_stars_with_threshold(*img_ptr, topk, 3.5f, med, sigma);

  // Adaptive fallback: if we found very few stars, retry with lower threshold
  // This helps with clouds/nebula where stars appear fainter
  const int min_expected = std::max(4, topk / 2);
  if (static_cast<int>(stars.size()) < min_expected) {
    std::vector<StarPoint> stars_relaxed = detect_stars_with_threshold(*img_ptr, topk, 2.5f, med, sigma);
    if (stars_relaxed.size() > stars.size()) {
      stars = std::move(stars_relaxed);
    }
  }

  return stars;
}

AffineStarRefinementResult estimate_affine_star_refinement(
    const std::vector<StarPoint> &ref_stars,
    const std::vector<StarPoint> &warped_stars, int image_rows, int image_cols,
    float match_radius_px) {
  AffineStarRefinementResult out;
  out.correction_warp = identity_warp();
  out.rejection_reason = "too_few_stars";
  if (ref_stars.size() < 24 || warped_stars.size() < 24 || image_rows <= 0 ||
      image_cols <= 0 || match_radius_px <= 0.0f) {
    return out;
  }

  struct Match {
    size_t warped_idx = 0;
    size_t ref_idx = 0;
  };
  std::vector<Match> matches;
  matches.reserve(std::min(ref_stars.size(), warped_stars.size()));
  const float radius_sq = match_radius_px * match_radius_px;
  constexpr float kAmbiguityRatioSq = 0.8f * 0.8f;

  for (size_t wi = 0; wi < warped_stars.size(); ++wi) {
    float best_d2 = radius_sq;
    float second_d2 = std::numeric_limits<float>::max();
    int best_ref = -1;
    for (size_t ri = 0; ri < ref_stars.size(); ++ri) {
      const float dx = warped_stars[wi].x - ref_stars[ri].x;
      const float dy = warped_stars[wi].y - ref_stars[ri].y;
      const float d2 = dx * dx + dy * dy;
      if (d2 < best_d2) {
        second_d2 = best_d2;
        best_d2 = d2;
        best_ref = static_cast<int>(ri);
      } else if (d2 < second_d2) {
        second_d2 = d2;
      }
    }
    if (best_ref < 0 ||
        (std::isfinite(second_d2) &&
         best_d2 > kAmbiguityRatioSq * second_d2)) {
      continue;
    }

    int nearest_warped = -1;
    float reverse_best_d2 = radius_sq;
    for (size_t other_wi = 0; other_wi < warped_stars.size(); ++other_wi) {
      const float dx = warped_stars[other_wi].x -
                       ref_stars[static_cast<size_t>(best_ref)].x;
      const float dy = warped_stars[other_wi].y -
                       ref_stars[static_cast<size_t>(best_ref)].y;
      const float d2 = dx * dx + dy * dy;
      if (d2 < reverse_best_d2) {
        reverse_best_d2 = d2;
        nearest_warped = static_cast<int>(other_wi);
      }
    }
    if (nearest_warped == static_cast<int>(wi)) {
      matches.push_back({wi, static_cast<size_t>(best_ref)});
    }
  }

  out.matched_stars = static_cast<int>(matches.size());
  if (matches.size() < 24) {
    out.rejection_reason = "too_few_mutual_matches";
    return out;
  }

  std::vector<cv::Point2f> points_warped;
  std::vector<cv::Point2f> points_ref;
  points_warped.reserve(matches.size());
  points_ref.reserve(matches.size());
  for (const Match &match : matches) {
    const auto &warped = warped_stars[match.warped_idx];
    const auto &ref = ref_stars[match.ref_idx];
    points_warped.emplace_back(warped.x, warped.y);
    points_ref.emplace_back(ref.x, ref.y);
  }

  cv::Mat inlier_mask;
  cv::Mat forward = estimate_affine_family_transform(
      points_warped, points_ref, "affine", inlier_mask, 0.75, 3000, 0.995);
  if (forward.empty()) {
    out.rejection_reason = "ransac_failed";
    return out;
  }
  out.inlier_stars = inlier_mask.empty() ? 0 : cv::countNonZero(inlier_mask);
  out.inlier_ratio = static_cast<float>(out.inlier_stars) /
                     static_cast<float>(matches.size());
  if (out.inlier_stars < 18 || out.inlier_ratio < 0.60f) {
    out.rejection_reason = "insufficient_ransac_consensus";
    return out;
  }

  std::vector<cv::Point2f> inlier_ref_points;
  inlier_ref_points.reserve(static_cast<size_t>(out.inlier_stars));
  for (size_t i = 0; i < points_ref.size(); ++i) {
    if (inlier_mask.at<uint8_t>(static_cast<int>(i)) != 0) {
      inlier_ref_points.push_back(points_ref[i]);
    }
  }
  std::vector<cv::Point2f> hull;
  cv::convexHull(inlier_ref_points, hull);
  const double image_area = static_cast<double>(image_rows) * image_cols;
  out.spatial_coverage = image_area > 0.0
                             ? static_cast<float>(cv::contourArea(hull) /
                                                  image_area)
                             : 0.0f;
  const cv::Rect2f bounds = cv::boundingRect(inlier_ref_points);
  const float x_span = bounds.width / static_cast<float>(image_cols);
  const float y_span = bounds.height / static_cast<float>(image_rows);
  if (out.spatial_coverage < 0.12f || x_span < 0.35f || y_span < 0.35f) {
    out.rejection_reason = "insufficient_spatial_coverage";
    return out;
  }

  const double a00 = forward.at<double>(0, 0);
  const double a01 = forward.at<double>(0, 1);
  const double tx = forward.at<double>(0, 2);
  const double a10 = forward.at<double>(1, 0);
  const double a11 = forward.at<double>(1, 1);
  const double ty = forward.at<double>(1, 2);
  if (!std::isfinite(a00) || !std::isfinite(a01) || !std::isfinite(tx) ||
      !std::isfinite(a10) || !std::isfinite(a11) || !std::isfinite(ty)) {
    out.rejection_reason = "non_finite_transform";
    return out;
  }
  const double determinant = a00 * a11 - a01 * a10;
  if (determinant <= 0.0) {
    out.rejection_reason = "invalid_determinant";
    return out;
  }

  cv::Mat linear = (cv::Mat_<double>(2, 2) << a00, a01, a10, a11);
  cv::SVD svd(linear, cv::SVD::NO_UV);
  out.max_scale = static_cast<float>(svd.w.at<double>(0));
  out.min_scale = static_cast<float>(svd.w.at<double>(1));
  if (out.min_scale < 0.99f || out.max_scale > 1.01f ||
      out.max_scale / out.min_scale > 1.01f) {
    out.rejection_reason = "scale_or_shear_out_of_bounds";
    return out;
  }

  out.rotation_deg = static_cast<float>(
      std::atan2(a10 - a01, a00 + a11) * 180.0 / CV_PI);
  if (std::fabs(out.rotation_deg) > 0.5f) {
    out.rejection_reason = "rotation_out_of_bounds";
    return out;
  }
  const double center_x = 0.5 * static_cast<double>(image_cols - 1);
  const double center_y = 0.5 * static_cast<double>(image_rows - 1);
  const double center_dx = a00 * center_x + a01 * center_y + tx - center_x;
  const double center_dy = a10 * center_x + a11 * center_y + ty - center_y;
  out.center_displacement_px =
      static_cast<float>(std::hypot(center_dx, center_dy));
  if (out.center_displacement_px > 2.0f) {
    out.rejection_reason = "center_displacement_out_of_bounds";
    return out;
  }

  auto percentile = [](std::vector<float> values, float q) {
    std::sort(values.begin(), values.end());
    if (values.empty()) {
      return 0.0f;
    }
    const float pos = std::clamp(q, 0.0f, 1.0f) *
                      static_cast<float>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = std::min(values.size() - 1, lo + 1);
    const float t = pos - static_cast<float>(lo);
    return values[lo] * (1.0f - t) + values[hi] * t;
  };
  std::vector<float> residuals_before;
  std::vector<float> residuals_after;
  residuals_before.reserve(matches.size());
  residuals_after.reserve(matches.size());
  double before_sq_sum = 0.0;
  double after_sq_sum = 0.0;
  for (size_t i = 0; i < points_ref.size(); ++i) {
    const double before_dx = points_warped[i].x - points_ref[i].x;
    const double before_dy = points_warped[i].y - points_ref[i].y;
    const float before = static_cast<float>(std::hypot(before_dx, before_dy));
    const double predicted_x =
        a00 * points_warped[i].x + a01 * points_warped[i].y + tx;
    const double predicted_y =
        a10 * points_warped[i].x + a11 * points_warped[i].y + ty;
    const float after = static_cast<float>(
        std::hypot(predicted_x - points_ref[i].x,
                   predicted_y - points_ref[i].y));
    residuals_before.push_back(before);
    residuals_after.push_back(after);
    before_sq_sum += static_cast<double>(before) * before;
    after_sq_sum += static_cast<double>(after) * after;
  }
  out.median_before_px = percentile(residuals_before, 0.5f);
  out.p90_before_px = percentile(residuals_before, 0.9f);
  out.rms_before_px = static_cast<float>(
      std::sqrt(before_sq_sum / static_cast<double>(matches.size())));
  out.median_after_px = percentile(residuals_after, 0.5f);
  out.p90_after_px = percentile(residuals_after, 0.9f);
  out.rms_after_px = static_cast<float>(
      std::sqrt(after_sq_sum / static_cast<double>(matches.size())));

  const float required_median_gain =
      std::max(0.01f, 0.05f * out.median_before_px);
  const float required_p90_gain = std::max(0.03f, 0.05f * out.p90_before_px);
  if (out.median_after_px > out.median_before_px - required_median_gain) {
    out.rejection_reason = "median_not_improved";
    return out;
  }
  if (out.p90_after_px > out.p90_before_px - required_p90_gain) {
    out.rejection_reason = "p90_not_improved";
    return out;
  }
  if (out.rms_after_px > out.rms_before_px + 1.0e-4f) {
    out.rejection_reason = "rms_regressed";
    return out;
  }

  if (!invert_forward_affine_to_warp(forward, true, out.correction_warp,
                                     &out.rejection_reason)) {
    return out;
  }
  out.valid = true;
  out.rejection_reason = "accepted";
  return out;
}

// =====================================================================
// Similarity helpers (used by trail, star pair, and triangle matching)
// =====================================================================

struct SimilarityResult {
  bool ok = false;
  float scale = 1.0f;
  float theta = 0.0f; // radians
  Eigen::Vector2f t{0.0f, 0.0f};
  int inliers = 0;
  float mean_err = 1.0e9f;
};

/// @brief Implements score similarity.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
SimilarityResult score_similarity(const std::vector<StarPoint> &mov,
                                  const std::vector<StarPoint> &ref,
                                  float scale, float theta,
                                  const Eigen::Vector2f &t,
                                  float inlier_tol_px) {
  if (mov.empty() || ref.empty())
    return {};
  const float ct = std::cos(theta);
  const float st = std::sin(theta);
  int inl = 0;
  float err_sum = 0.0f;
  for (const auto &m : mov) {
    const float xr = scale * (ct * m.x - st * m.y) + t.x();
    const float yr = scale * (st * m.x + ct * m.y) + t.y();
    float best = std::numeric_limits<float>::max();
    for (const auto &r : ref) {
      const float dx = xr - r.x;
      const float dy = yr - r.y;
      const float d = std::sqrt(dx * dx + dy * dy);
      if (d < best)
        best = d;
    }
    if (best < inlier_tol_px) {
      ++inl;
      err_sum += best;
    }
  }
  SimilarityResult res;
  res.ok = inl > 0;
  res.inliers = inl;
  res.mean_err = (inl > 0) ? (err_sum / static_cast<float>(inl)) : res.mean_err;
  res.scale = scale;
  res.theta = theta;
  res.t = t;
  return res;
}

static std::vector<std::pair<cv::Point2f, cv::Point2f>>
/// @brief Builds similarity consensus pairs.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
build_similarity_consensus_pairs(const std::vector<StarPoint> &mov,
                                 const std::vector<StarPoint> &ref,
                                 const SimilarityResult &best,
                                 float inlier_tol_px) {
  struct Candidate {
    int mov_idx = -1;
    int ref_idx = -1;
    float dist = 0.0f;
  };

  std::vector<Candidate> candidates;
  candidates.reserve(mov.size());
  const float ct = std::cos(best.theta);
  const float st = std::sin(best.theta);
  for (size_t mi = 0; mi < mov.size(); ++mi) {
    const auto &m = mov[mi];
    const float xr = best.scale * (ct * m.x - st * m.y) + best.t.x();
    const float yr = best.scale * (st * m.x + ct * m.y) + best.t.y();
    int best_ref_idx = -1;
    float best_dist = std::numeric_limits<float>::max();
    for (size_t ri = 0; ri < ref.size(); ++ri) {
      const float dx = xr - ref[ri].x;
      const float dy = yr - ref[ri].y;
      const float d = std::sqrt(dx * dx + dy * dy);
      if (d < best_dist) {
        best_dist = d;
        best_ref_idx = static_cast<int>(ri);
      }
    }
    if (best_ref_idx >= 0 && best_dist < inlier_tol_px) {
      candidates.push_back(
          {static_cast<int>(mi), best_ref_idx, best_dist});
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &a, const Candidate &b) {
              return a.dist < b.dist;
            });
  std::vector<char> ref_used(ref.size(), 0);
  std::vector<std::pair<cv::Point2f, cv::Point2f>> pairs;
  pairs.reserve(candidates.size());
  for (const Candidate &c : candidates) {
    if (c.ref_idx < 0 || c.ref_idx >= static_cast<int>(ref.size()) ||
        ref_used[static_cast<size_t>(c.ref_idx)] != 0) {
      continue;
    }
    ref_used[static_cast<size_t>(c.ref_idx)] = 1;
    pairs.emplace_back(cv::Point2f(mov[static_cast<size_t>(c.mov_idx)].x,
                                   mov[static_cast<size_t>(c.mov_idx)].y),
                       cv::Point2f(ref[static_cast<size_t>(c.ref_idx)].x,
                                   ref[static_cast<size_t>(c.ref_idx)].y));
  }
  return pairs;
}

/// @brief Implements maybe refine similarity to affine.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static bool maybe_refine_similarity_to_affine(
    const std::vector<StarPoint> &mov, const std::vector<StarPoint> &ref,
    const SimilarityResult &best, bool allow_rotation, float inlier_tol_px,
    int min_inliers, const std::string &transform_model, WarpMatrix &warp,
    float &correlation, std::string &error_message) {
  if (transform_model != "affine") {
    return false;
  }

  const auto pairs =
      build_similarity_consensus_pairs(mov, ref, best, inlier_tol_px);
  if (pairs.size() < 3) {
    error_message = "few_affine_pairs";
    return false;
  }

  std::vector<cv::Point2f> pts_mov;
  std::vector<cv::Point2f> pts_ref;
  pts_mov.reserve(pairs.size());
  pts_ref.reserve(pairs.size());
  for (const auto &p : pairs) {
    pts_mov.push_back(p.first);
    pts_ref.push_back(p.second);
  }

  cv::Mat inliers;
  cv::Mat A = estimate_affine_family_transform(
      pts_mov, pts_ref, "affine", inliers,
      std::max(3.0f, inlier_tol_px * 1.5f));
  if (A.empty()) {
    error_message = "affine_refine_fail";
    return false;
  }
  if (!invert_forward_affine_to_warp(A, allow_rotation, warp, &error_message)) {
    return false;
  }
  const int inl = inliers.empty() ? 0 : cv::countNonZero(inliers);
  if (inl < min_inliers) {
    error_message = "few_affine_inliers";
    return false;
  }
  correlation = pts_mov.empty()
                    ? 0.0f
                    : static_cast<float>(inl) /
                          static_cast<float>(pts_mov.size());
  return true;
}

/// @brief Implements similarity from pairs.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool similarity_from_pairs(const Eigen::Vector2f &m1, const Eigen::Vector2f &m2,
                           const Eigen::Vector2f &r1, const Eigen::Vector2f &r2,
                           bool allow_rotation, float &scale, float &theta,
                           Eigen::Vector2f &t) {
  Eigen::Vector2f v_m = m2 - m1;
  Eigen::Vector2f v_r = r2 - r1;
  const float len_m = v_m.norm();
  const float len_r = v_r.norm();
  if (len_m < 1.0e-3f || len_r < 1.0e-3f)
    return false;
  scale = len_r / len_m;
  if (!std::isfinite(scale) || scale <= 0.0f)
    return false;
  theta = allow_rotation
              ? (std::atan2(v_r.y(), v_r.x()) - std::atan2(v_m.y(), v_m.x()))
              : 0.0f;
  const float ct = std::cos(theta);
  const float st = std::sin(theta);
  Eigen::Matrix2f R;
  R << ct, -st, st, ct;
  t = r1 - scale * (R * m1);
  return true;
}

// =====================================================================
// Robust multi-scale Phase+ECC with gradient pre-processing
// Handles nebula/cloud gradients and large rotations better.
// =====================================================================

/// @brief Implements gradient preprocess.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static Matrix2Df gradient_preprocess(const Matrix2Df &img) {
  cv::Mat f(img.rows(), img.cols(), CV_32F,
            const_cast<float *>(img.data()));
  // Laplacian of Gaussian: removes low-frequency gradients (nebula, clouds)
  // while preserving high-frequency structure (stars, edges)
  cv::Mat blurred;
  cv::GaussianBlur(f, blurred, cv::Size(0, 0), 2.0);
  cv::Mat laplacian;
  cv::Laplacian(blurred, laplacian, CV_32F, 3);

  // Normalize to [0,1]
  double minV, maxV;
  cv::minMaxLoc(laplacian, &minV, &maxV);
  if (maxV - minV < 1e-10)
    maxV = minV + 1.0;
  cv::Mat norm = (laplacian - minV) / (maxV - minV);

  Matrix2Df result(img.rows(), img.cols());
  std::memcpy(result.data(), norm.data,
              static_cast<size_t>(img.size()) * sizeof(float));
  return result;
}

/// @brief Implements robust phase ecc.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
RegistrationResult robust_phase_ecc(const Matrix2Df &mov,
                                    const Matrix2Df &ref,
                                    bool allow_rotation) {
  RegistrationResult res;
  res.warp = identity_warp();
  res.success = false;
  res.correlation = 0.0f;

  // Gradient pre-processing: removes nebula/cloud gradients
  Matrix2Df mov_grad = gradient_preprocess(mov);
  Matrix2Df ref_grad = gradient_preprocess(ref);

  // Multi-scale pyramid: coarse-to-fine for large displacements/rotations
  // Level 2 (4x downsampled) → Level 1 (2x) → Level 0 (original)
  std::vector<Matrix2Df> mov_pyr = {mov_grad};
  std::vector<Matrix2Df> ref_pyr = {ref_grad};

  // Build 2 pyramid levels
  for (int level = 0; level < 2; ++level) {
    mov_pyr.push_back(downsample2x2_mean(mov_pyr.back()));
    ref_pyr.push_back(downsample2x2_mean(ref_pyr.back()));
  }

  // Start from coarsest level
  WarpMatrix current_warp = identity_warp();

  for (int level = static_cast<int>(mov_pyr.size()) - 1; level >= 0; --level) {
    const Matrix2Df &m = mov_pyr[static_cast<size_t>(level)];
    const Matrix2Df &r = ref_pyr[static_cast<size_t>(level)];

    Matrix2Df m_ecc = prepare_ecc_image(m);
    Matrix2Df r_ecc = prepare_ecc_image(r);

    if (level == static_cast<int>(mov_pyr.size()) - 1) {
      // Coarsest level: estimate initial translation + rotation
      auto [dx, dy] = phasecorr_translation(m_ecc, r_ecc);
      // phaseCorrelate reports the forward content displacement.  ECC and
      // apply_warp() consume the inverse-map translation, so negate it for
      // the initial warp seed.
      dx = -dx;
      dy = -dy;
      current_warp(0, 2) = dx;
      current_warp(1, 2) = dy;

      if (allow_rotation) {
        cv::Mat r_cv(r_ecc.rows(), r_ecc.cols(), CV_32F,
                     const_cast<float *>(r_ecc.data()));
        cv::Mat m_cv(m_ecc.rows(), m_ecc.cols(), CV_32F,
                     const_cast<float *>(m_ecc.data()));
        float rot = estimate_rotation_logpolar(r_cv, m_cv);
        float th = rot * 3.14159265f / 180.0f;
        float ct = std::cos(th);
        float st = std::sin(th);
        float cx = static_cast<float>(m_ecc.cols()) * 0.5f;
        float cy = static_cast<float>(m_ecc.rows()) * 0.5f;
        float tx_rot = cx * (1.0f - ct) + cy * st;
        float ty_rot = cy * (1.0f - ct) - cx * st;
        current_warp << ct, -st, dx + tx_rot, st, ct, dy + ty_rot;
      }
    } else {
      // Scale warp from previous (coarser) level to current level
      current_warp(0, 2) *= 2.0f;
      current_warp(1, 2) *= 2.0f;
    }

    // ECC refinement at this level
    int iters = (level == 0) ? 200 : 100;
    float eps = (level == 0) ? 1e-6f : 1e-4f;
    RegistrationResult level_res =
        ecc_warp(m_ecc, r_ecc, allow_rotation, current_warp, iters, eps);
    if (level_res.success) {
      current_warp = level_res.warp;
      res = level_res;
    }
    // If ECC fails at coarse level, still try finer levels with current seed
  }

  // NOTE: OpenCV findTransformECC returns a warp that is intended to be used
  // directly with warpAffine(..., WARP_INVERSE_MAP) to align moving→reference.
  // Our apply_warp() also uses WARP_INVERSE_MAP, so we must NOT invert here.
  return res;
}

/// @brief Implements robust phase ecc seeded.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
RegistrationResult robust_phase_ecc_seeded(const Matrix2Df &mov,
                                           const Matrix2Df &ref,
                                           bool allow_rotation,
                                           const WarpMatrix &init_warp) {
  RegistrationResult res;
  res.warp = init_warp;
  res.success = false;
  res.correlation = 0.0f;

  Matrix2Df mov_grad = gradient_preprocess(mov);
  Matrix2Df ref_grad = gradient_preprocess(ref);

  std::vector<Matrix2Df> mov_pyr = {mov_grad};
  std::vector<Matrix2Df> ref_pyr = {ref_grad};
  for (int level = 0; level < 2; ++level) {
    mov_pyr.push_back(downsample2x2_mean(mov_pyr.back()));
    ref_pyr.push_back(downsample2x2_mean(ref_pyr.back()));
  }

  WarpMatrix current_warp = init_warp;
  const int coarsest_level = static_cast<int>(mov_pyr.size()) - 1;
  const float coarsest_scale = std::pow(2.0f, static_cast<float>(coarsest_level));
  current_warp(0, 2) /= coarsest_scale;
  current_warp(1, 2) /= coarsest_scale;

  for (int level = coarsest_level; level >= 0; --level) {
    const Matrix2Df &m = mov_pyr[static_cast<size_t>(level)];
    const Matrix2Df &r = ref_pyr[static_cast<size_t>(level)];

    Matrix2Df m_ecc = prepare_ecc_image(m);
    Matrix2Df r_ecc = prepare_ecc_image(r);

    if (level < coarsest_level) {
      current_warp(0, 2) *= 2.0f;
      current_warp(1, 2) *= 2.0f;
    }

    const int iters = (level == 0) ? 250 : 120;
    const float eps = (level == 0) ? 1e-6f : 1e-4f;
    RegistrationResult level_res =
        ecc_warp(m_ecc, r_ecc, allow_rotation, current_warp, iters, eps);
    if (level_res.success) {
      current_warp = level_res.warp;
      res = level_res;
    }
  }

  return res;
}

namespace {
struct SimilarityPair {
  int i;
  int j;
  float dist;
};
} // namespace

RegistrationResult
/// @brief Implements star registration similarity.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
star_registration_similarity(const Matrix2Df &mov, const Matrix2Df &ref,
                             bool allow_rotation,
                             int topk_stars, int min_inliers,
                             float inlier_tol_px, float dist_bin_px,
                             const std::string &transform_model,
                             bool enable_local_background_subtraction) {
  RegistrationResult res;
  res.warp = identity_warp();
  res.success = false;
  res.correlation = 0.0f;

  auto mov_stars = detect_stars_simple(mov, topk_stars, enable_local_background_subtraction);
  auto ref_stars = detect_stars_simple(ref, topk_stars, enable_local_background_subtraction);
  if (mov_stars.size() < 3 || ref_stars.size() < 3) {
    res.error_message = "too_few_stars";
    return res;
  }

  std::vector<SimilarityPair> ref_pairs;
  std::vector<SimilarityPair> mov_pairs;

  auto build_pairs = [&](const std::vector<StarPoint> &stars,
                         std::vector<SimilarityPair> &out) {
    const int n = static_cast<int>(stars.size());
    out.reserve(static_cast<size_t>(n) * static_cast<size_t>(n) / 2);
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        const float dx = stars[j].x - stars[i].x;
        const float dy = stars[j].y - stars[i].y;
        const float d = std::sqrt(dx * dx + dy * dy);
        if (d > 1.0f)
          out.push_back({i, j, d});
      }
    }
    std::sort(out.begin(), out.end(),
              [](const SimilarityPair &a, const SimilarityPair &b) { return a.dist > b.dist; });
    const size_t limit = std::min<size_t>(out.size(), 800);
    out.resize(limit);
  };

  build_pairs(ref_stars, ref_pairs);
  build_pairs(mov_stars, mov_pairs);
  if (ref_pairs.empty() || mov_pairs.empty()) {
    res.error_message = "no_pairs";
    return res;
  }

  std::unordered_map<int, std::vector<SimilarityPair>> ref_bucket;
  ref_bucket.reserve(ref_pairs.size() * 2);
  for (const auto &p : ref_pairs) {
    int key = static_cast<int>(std::round(p.dist / dist_bin_px));
    ref_bucket[key].push_back(p);
  }

  SimilarityResult best;
  int attempts = 0;
  for (const auto &pm : mov_pairs) {
    int key = static_cast<int>(std::round(pm.dist / dist_bin_px));
    for (int dk = -1; dk <= 1; ++dk) {
      auto it = ref_bucket.find(key + dk);
      if (it == ref_bucket.end())
        continue;
      for (const auto &pr : it->second) {
        ++attempts;
        float scale = 1.0f;
        float theta = 0.0f;
        Eigen::Vector2f t;
        const Eigen::Vector2f m1(mov_stars[pm.i].x, mov_stars[pm.i].y);
        const Eigen::Vector2f m2(mov_stars[pm.j].x, mov_stars[pm.j].y);
        const Eigen::Vector2f r1(ref_stars[pr.i].x, ref_stars[pr.i].y);
        const Eigen::Vector2f r2(ref_stars[pr.j].x, ref_stars[pr.j].y);
        if (!similarity_from_pairs(m1, m2, r1, r2, allow_rotation, scale, theta,
                                   t))
          continue;
        // No hard rotation limit — RANSAC consensus handles outliers.
        if (!std::isfinite(scale) || scale < 0.5f || scale > 2.0f)
          continue;

        SimilarityResult sr = score_similarity(mov_stars, ref_stars, scale,
                                               theta, t, inlier_tol_px);
        if (!sr.ok)
          continue;
        if (sr.inliers > best.inliers ||
            (sr.inliers == best.inliers && sr.mean_err < best.mean_err)) {
          best = sr;
        }
      }
    }
    if (attempts > 4000 && best.inliers >= min_inliers)
      break;
  }

  if (best.inliers >= min_inliers && best.mean_err < inlier_tol_px * 1.2f) {
    res.success = true;
    res.correlation = static_cast<float>(best.inliers) /
                      static_cast<float>(std::max<size_t>(1, mov_stars.size()));
    if (maybe_refine_similarity_to_affine(
            mov_stars, ref_stars, best, allow_rotation, inlier_tol_px,
            min_inliers, transform_model, res.warp, res.correlation,
            res.error_message)) {
      return res;
    }
    // Construct Forward Matrix (M -> R)
    float s_fw = best.scale;
    float th_fw = best.theta;
    float tx_fw = best.t.x();
    float ty_fw = best.t.y();

    float c_fw = std::cos(th_fw);
    float sn_fw = std::sin(th_fw);

    // We need the Inverse Matrix (R -> M) for apply_warp with WARP_INVERSE_MAP
    // M = (1/s) * R^T * (R_coord - t)
    // Scale_inv = 1/s
    // Rot_inv = -theta
    // T_inv = - (1/s) * R^T * t

    float s_inv = 1.0f / s_fw;
    float c_inv = c_fw;    // cos(-th) = cos(th)
    float sn_inv = -sn_fw; // sin(-th) = -sin(th)

    float a00 = s_inv * c_inv;
    float a01 = s_inv * -sn_inv; // -sin inside rot matrix
    float a10 = s_inv * sn_inv;
    float a11 = s_inv * c_inv;

    // t_inv = - S_inv * t_fw = - [a00 a01; a10 a11] * [tx; ty]
    float tx_inv = -(a00 * tx_fw + a01 * ty_fw);
    float ty_inv = -(a10 * tx_fw + a11 * ty_fw);

    res.warp << a00, a01, tx_inv, a10, a11, ty_inv;
  } else {
    res.error_message = "no_consensus";
  }
  return res;
}

// =====================================================================
// Triangle-based asterism matching (astroalign-style, rotation-invariant)
// =====================================================================

struct Triangle {
  int i, j, k;          // star indices
  float sides[3];       // sorted side lengths (ascending)
  float perimeter;
  float ratios[2];      // sides[0]/sides[2], sides[1]/sides[2] — invariants
  float signed_area2;   // orientation-preserving triangle discriminator
};

static std::vector<Triangle>
/// @brief Builds triangles.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
build_triangles(const std::vector<StarPoint> &stars, int max_triangles) {
  const int n = static_cast<int>(stars.size());
  std::vector<Triangle> tris;
  if (n < 3)
    return tris;

  // Limit combinatorial explosion: use a wider star subset. The previous
  // hard cap of 30 caused dense fields to collapse into many ambiguous
  // asterisms, even though plenty of stars were available.
  const int limit = std::min(n, 60);
  tris.reserve(static_cast<size_t>(limit * (limit - 1) * (limit - 2) / 6));

  for (int i = 0; i < limit; ++i) {
    for (int j = i + 1; j < limit; ++j) {
      for (int k = j + 1; k < limit; ++k) {
        float dx_ij = stars[j].x - stars[i].x;
        float dy_ij = stars[j].y - stars[i].y;
        float dx_ik = stars[k].x - stars[i].x;
        float dy_ik = stars[k].y - stars[i].y;
        float dx_jk = stars[k].x - stars[j].x;
        float dy_jk = stars[k].y - stars[j].y;

        float d_ij = std::sqrt(dx_ij * dx_ij + dy_ij * dy_ij);
        float d_ik = std::sqrt(dx_ik * dx_ik + dy_ik * dy_ik);
        float d_jk = std::sqrt(dx_jk * dx_jk + dy_jk * dy_jk);
        const float signed_area2 = dx_ij * dy_ik - dy_ij * dx_ik;

        // Skip degenerate triangles
        if (d_ij < 3.0f || d_ik < 3.0f || d_jk < 3.0f)
          continue;
        if (std::fabs(signed_area2) < 9.0f)
          continue;

        Triangle t;
        t.i = i;
        t.j = j;
        t.k = k;

        // Sort sides ascending
        float s[3] = {d_ij, d_ik, d_jk};
        if (s[0] > s[1]) std::swap(s[0], s[1]);
        if (s[1] > s[2]) std::swap(s[1], s[2]);
        if (s[0] > s[1]) std::swap(s[0], s[1]);

        t.sides[0] = s[0];
        t.sides[1] = s[1];
        t.sides[2] = s[2];
        t.perimeter = s[0] + s[1] + s[2];

        // Invariant ratios (scale + rotation invariant)
        t.ratios[0] = s[0] / s[2];
        t.ratios[1] = s[1] / s[2];
        t.signed_area2 = signed_area2;

        // Near-equilateral and near-isosceles triangles are highly ambiguous
        // under ratio-only matching and poison the downstream affine fit.
        if (t.ratios[1] > 0.97f || (t.ratios[1] - t.ratios[0]) < 0.02f) {
          continue;
        }

        tris.push_back(t);
      }
    }
  }

  // Keep largest triangles (more robust)
  if (static_cast<int>(tris.size()) > max_triangles) {
    std::sort(tris.begin(), tris.end(),
              [](const Triangle &a, const Triangle &b) {
                return a.perimeter > b.perimeter;
              });
    tris.resize(static_cast<size_t>(max_triangles));
  }

  return tris;
}

RegistrationResult
/// @brief Implements triangle star matching.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
triangle_star_matching(const Matrix2Df &mov, const Matrix2Df &ref,
                       bool allow_rotation,
                       int topk_stars, int min_inliers,
                       float inlier_tol_px,
                       const std::string &transform_model,
                       bool enable_local_background_subtraction,
                       float shift_radius_px) {
  RegistrationResult res;
  res.warp = identity_warp();
  res.success = false;
  res.correlation = 0.0f;

  auto mov_stars = detect_stars_simple(mov, topk_stars, enable_local_background_subtraction);
  auto ref_stars = detect_stars_simple(ref, topk_stars, enable_local_background_subtraction);

  if (mov_stars.size() < 3 || ref_stars.size() < 3) {
    res.error_message = "too_few_stars";
    return res;
  }

  const int max_tris = 600;
  auto mov_tris = build_triangles(mov_stars, max_tris);
  auto ref_tris = build_triangles(ref_stars, max_tris);

  if (mov_tris.empty() || ref_tris.empty()) {
    res.error_message = "no_triangles";
    return res;
  }

  // Match triangles by invariant ratios
  // Calibrated via Python simulation on M104 star data:
  // ratio_tol=0.03 -> too many false matches (crowded fields)
  // ratio_tol=0.008, ambiguity_margin=0.007 -> correct balance
  const float ratio_tol = 0.008f;
  const float ambiguity_margin = 0.007f;

  struct PairVote {
    int mov_idx = -1;
    int ref_idx = -1;
    int votes = 0;
    float score_sum = 0.0f;
  };
  std::unordered_map<std::uint64_t, PairVote> pair_votes;
  pair_votes.reserve(static_cast<size_t>(max_tris));

  int matches_found = 0;
  for (const auto &mt : mov_tris) {
    float best_dist = ratio_tol * 2.0f;
    float second_best_dist = ratio_tol * 2.0f;
    const Triangle *best_rt = nullptr;

    for (const auto &rt : ref_tris) {
      if ((mt.signed_area2 > 0.0f) != (rt.signed_area2 > 0.0f)) {
        continue;
      }
      float dr0 = mt.ratios[0] - rt.ratios[0];
      float dr1 = mt.ratios[1] - rt.ratios[1];
      float d = std::sqrt(dr0 * dr0 + dr1 * dr1);
      if (d < best_dist) {
        second_best_dist = best_dist;
        best_dist = d;
        best_rt = &rt;
      } else if (d < second_best_dist) {
        second_best_dist = d;
      }
    }

    if (!best_rt || best_dist >= ratio_tol)
      continue;
    if (second_best_dist - best_dist < ambiguity_margin)
      continue;

    matches_found++;

    // Determine vertex correspondence by matching sorted side lengths
    // The vertices opposite to the shortest/medium/longest sides correspond
    // mov triangle vertices: i, j, k
    // We need to figure out which vertex is opposite which side
    // Side ij is between i,j — opposite vertex is k
    // Side ik is between i,k — opposite vertex is j
    // Side jk is between j,k — opposite vertex is i
    auto vertex_order = [](const std::vector<StarPoint> &stars,
                           const Triangle &t) -> std::array<int, 3> {
      float d_ij = 0, d_ik = 0, d_jk = 0;
      {
        float dx = stars[t.j].x - stars[t.i].x;
        float dy = stars[t.j].y - stars[t.i].y;
        d_ij = std::sqrt(dx * dx + dy * dy);
      }
      {
        float dx = stars[t.k].x - stars[t.i].x;
        float dy = stars[t.k].y - stars[t.i].y;
        d_ik = std::sqrt(dx * dx + dy * dy);
      }
      {
        float dx = stars[t.k].x - stars[t.j].x;
        float dy = stars[t.k].y - stars[t.j].y;
        d_jk = std::sqrt(dx * dx + dy * dy);
      }
      // vertex opposite shortest side, medium side, longest side
      struct SideVtx {
        float len;
        int opposite;
      };
      SideVtx sv[3] = {{d_ij, t.k}, {d_ik, t.j}, {d_jk, t.i}};
      if (sv[0].len > sv[1].len) std::swap(sv[0], sv[1]);
      if (sv[1].len > sv[2].len) std::swap(sv[1], sv[2]);
      if (sv[0].len > sv[1].len) std::swap(sv[0], sv[1]);
      return {sv[0].opposite, sv[1].opposite, sv[2].opposite};
    };

    auto mov_order = vertex_order(mov_stars, mt);
    auto ref_order = vertex_order(ref_stars, *best_rt);

    for (int v = 0; v < 3; ++v) {
      const int mov_idx = mov_order[v];
      const int ref_idx = ref_order[v];
      const std::uint64_t key =
          (static_cast<std::uint64_t>(static_cast<std::uint32_t>(mov_idx))
           << 32) |
          static_cast<std::uint32_t>(ref_idx);
      auto &vote = pair_votes[key];
      vote.mov_idx = mov_idx;
      vote.ref_idx = ref_idx;
      vote.votes += 1;
      vote.score_sum += best_dist;
    }

    if (matches_found > 200)
      break;
  }

  std::vector<PairVote> ranked_pairs;
  ranked_pairs.reserve(pair_votes.size());
  for (const auto &entry : pair_votes) {
    ranked_pairs.push_back(entry.second);
  }
  std::sort(ranked_pairs.begin(), ranked_pairs.end(),
            [](const PairVote &a, const PairVote &b) {
              if (a.votes != b.votes) {
                return a.votes > b.votes;
              }
              return a.score_sum < b.score_sum;
            });

  // Shift-consistency filter: each candidate pair implies a translation
  // (dx,dy). Compute per-pair support = sum of votes of other pairs whose
  // implied shift is within shift_radius px. The pair with highest support
  // is the shift-cluster anchor; keep only pairs within shift_radius of it.
  // This eliminates false matches that dominate vote counts in crowded fields.
  const float shift_radius = shift_radius_px;
  struct PairShift {
    int mov_idx, ref_idx, votes;
    float dx, dy;
    int support;
  };
  std::vector<PairShift> cand_pairs;
  cand_pairs.reserve(ranked_pairs.size());
  for (const PairVote &vote : ranked_pairs) {
    if (vote.mov_idx < 0 || vote.ref_idx < 0) continue;
    if (vote.mov_idx >= static_cast<int>(mov_stars.size()) ||
        vote.ref_idx >= static_cast<int>(ref_stars.size())) continue;
    float dx = ref_stars[static_cast<size_t>(vote.ref_idx)].x -
               mov_stars[static_cast<size_t>(vote.mov_idx)].x;
    float dy = ref_stars[static_cast<size_t>(vote.ref_idx)].y -
               mov_stars[static_cast<size_t>(vote.mov_idx)].y;
    cand_pairs.push_back({vote.mov_idx, vote.ref_idx, vote.votes, dx, dy, 0});
  }
  // Compute shift support for each pair (O(n^2), n <= ~150 pairs — fast enough)
  for (size_t i = 0; i < cand_pairs.size(); ++i) {
    int sup = 0;
    for (size_t j = 0; j < cand_pairs.size(); ++j) {
      if (i == j) continue;
      float ddx = cand_pairs[i].dx - cand_pairs[j].dx;
      float ddy = cand_pairs[i].dy - cand_pairs[j].dy;
      if (std::sqrt(ddx * ddx + ddy * ddy) <= shift_radius)
        sup += cand_pairs[j].votes;
    }
    cand_pairs[i].support = sup;
  }
  // Find anchor = pair with maximum weighted shift support
  size_t best_anchor = 0;
  int best_anchor_sup = -1;
  for (size_t i = 0; i < cand_pairs.size(); ++i) {
    if (cand_pairs[i].support > best_anchor_sup) {
      best_anchor_sup = cand_pairs[i].support;
      best_anchor = i;
    }
  }
  // Keep only pairs consistent with the anchor shift
  if (!cand_pairs.empty()) {
    float anchor_dx = cand_pairs[best_anchor].dx;
    float anchor_dy = cand_pairs[best_anchor].dy;
    cand_pairs.erase(
        std::remove_if(cand_pairs.begin(), cand_pairs.end(),
                       [anchor_dx, anchor_dy, shift_radius](const PairShift &p) {
                         float ddx = p.dx - anchor_dx;
                         float ddy = p.dy - anchor_dy;
                         return std::sqrt(ddx * ddx + ddy * ddy) > shift_radius;
                       }),
        cand_pairs.end());
  }
  // Re-sort by vote count
  std::sort(cand_pairs.begin(), cand_pairs.end(),
            [](const PairShift &a, const PairShift &b) {
              return a.votes > b.votes;
            });

  std::vector<cv::Point2f> pts_mov, pts_ref;
  pts_mov.reserve(cand_pairs.size());
  pts_ref.reserve(cand_pairs.size());
  std::vector<uint8_t> mov_used(mov_stars.size(), 0);
  std::vector<uint8_t> ref_used(ref_stars.size(), 0);
  for (const PairShift &p : cand_pairs) {
    if (mov_used[static_cast<size_t>(p.mov_idx)] != 0 ||
        ref_used[static_cast<size_t>(p.ref_idx)] != 0) {
      continue;
    }
    mov_used[static_cast<size_t>(p.mov_idx)] = 1;
    ref_used[static_cast<size_t>(p.ref_idx)] = 1;
    pts_mov.push_back(cv::Point2f(mov_stars[static_cast<size_t>(p.mov_idx)].x,
                                  mov_stars[static_cast<size_t>(p.mov_idx)].y));
    pts_ref.push_back(cv::Point2f(ref_stars[static_cast<size_t>(p.ref_idx)].x,
                                  ref_stars[static_cast<size_t>(p.ref_idx)].y));
  }

  if (pts_mov.size() < static_cast<size_t>(std::max(3, min_inliers))) {
    res.error_message = "few_triangle_matches";
    return res;
  }

  // Use RANSAC to find the best similarity transform from correspondences
  cv::Mat inliers;
  cv::Mat A = estimate_affine_family_transform(pts_mov, pts_ref,
                                               transform_model, inliers);
  if (A.empty()) {
    res.error_message = "transform_fail";
    return res;
  }
  if (!invert_forward_affine_to_warp(A, allow_rotation, res.warp,
                                     &res.error_message)) {
    return res;
  }
  int inl = inliers.empty() ? 0 : cv::countNonZero(inliers);
  res.correlation =
      pts_mov.empty()
          ? 0.0f
          : static_cast<float>(inl) / static_cast<float>(pts_mov.size());
  res.success = inl >= min_inliers;
  return res;
}

/// @brief Implements hybrid phase ecc.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
RegistrationResult hybrid_phase_ecc(const Matrix2Df &mov, const Matrix2Df &ref,
                                    bool allow_rotation) {
  RegistrationResult res;
  res.warp = identity_warp();
  res.success = false;
  res.correlation = 0.0f;

  Matrix2Df mov_ecc = prepare_ecc_image(mov);
  Matrix2Df ref_ecc = prepare_ecc_image(ref);

  auto [dx, dy] = phasecorr_translation(mov_ecc, ref_ecc);
  // phaseCorrelate returns the forward content displacement; the warp used
  // with WARP_INVERSE_MAP must carry the opposite translation.
  dx = -dx;
  dy = -dy;

  WarpMatrix init = identity_warp();
  init(0, 2) = dx;
  init(1, 2) = dy;

  if (allow_rotation) {
    cv::Mat ref_cv(ref_ecc.rows(), ref_ecc.cols(), CV_32F,
                   const_cast<float *>(ref_ecc.data()));
    cv::Mat mov_cv(mov_ecc.rows(), mov_ecc.cols(), CV_32F,
                   const_cast<float *>(mov_ecc.data()));
    float rot = estimate_rotation_logpolar(ref_cv, mov_cv);
    // Use actual detected rotation as ECC seed — clamping causes
    // convergence failures when real rotation exceeds the limit.
    const float th = rot * 3.14159265f / 180.0f;
    const float ct = std::cos(th);
    const float st = std::sin(th);

    // Correct for center of rotation
    const float cx = static_cast<float>(mov_ecc.cols()) * 0.5f;
    const float cy = static_cast<float>(mov_ecc.rows()) * 0.5f;
    const float tx_rot = cx * (1.0f - ct) + cy * st;
    const float ty_rot = cy * (1.0f - ct) - cx * st;

    init << ct, -st, dx + tx_rot, st, ct, dy + ty_rot;
  }

  res = ecc_warp(mov_ecc, ref_ecc, allow_rotation, init, 200, 1e-6f);

  // NOTE: OpenCV findTransformECC returns a warp that is intended to be used
  // directly with warpAffine(..., WARP_INVERSE_MAP) to align moving→reference.
  // Our apply_warp() also uses WARP_INVERSE_MAP, so we must NOT invert here.
  return res;
}

// =====================================================================
// Canonical single-frame registration cascade with NCC validation
// =====================================================================

/// @brief Computes ncc.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static float compute_ncc(const Matrix2Df &a, const Matrix2Df &b) {
  const int n = a.size();
  if (n <= 0 || n != b.size())
    return 0.0f;
  const float *da = a.data();
  const float *db = b.data();
  double ma = 0, mb = 0;
  for (int i = 0; i < n; ++i) {
    ma += da[i];
    mb += db[i];
  }
  ma /= n;
  mb /= n;
  double sab = 0, saa = 0, sbb = 0;
  for (int i = 0; i < n; ++i) {
    double va = da[i] - ma;
    double vb = db[i] - mb;
    sab += va * vb;
    saa += va * va;
    sbb += vb * vb;
  }
  double den = std::sqrt(saa * sbb);
  return (den > 1e-10) ? static_cast<float>(sab / den) : 0.0f;
}

/// @brief Implements warp valid mask.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
cv::Mat warp_valid_mask(const Matrix2Df &img, const WarpMatrix &warp) {
  cv::Mat ones(img.rows(), img.cols(), CV_32F, cv::Scalar(1.0f));
  cv::Mat warp_matrix(2, 3, CV_32F);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      warp_matrix.at<float>(i, j) = warp(i, j);
    }
  }
  cv::Mat warped_mask;
  cv::warpAffine(ones, warped_mask, warp_matrix, ones.size(),
                 cv::INTER_NEAREST | cv::WARP_INVERSE_MAP,
                 cv::BORDER_CONSTANT, cv::Scalar(0.0f));
  return warped_mask;
}

/// @brief Computes ncc masked.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float compute_ncc_masked(const Matrix2Df &a, const Matrix2Df &b,
                         const cv::Mat &mask, int *used_pixels) {
  if (a.size() <= 0 || a.size() != b.size() || mask.empty() ||
      mask.rows != a.rows() || mask.cols != a.cols()) {
    if (used_pixels) {
      *used_pixels = 0;
    }
    return 0.0f;
  }

  const float *da = a.data();
  const float *db = b.data();
  double ma = 0.0;
  double mb = 0.0;
  int n = 0;
  for (int y = 0; y < mask.rows; ++y) {
    const float *pm = mask.ptr<float>(y);
    const int row_off = y * mask.cols;
    for (int x = 0; x < mask.cols; ++x) {
      if (pm[x] <= 0.5f) {
        continue;
      }
      const int idx = row_off + x;
      ma += da[idx];
      mb += db[idx];
      ++n;
    }
  }
  if (used_pixels) {
    *used_pixels = n;
  }
  if (n <= 1) {
    return 0.0f;
  }

  ma /= static_cast<double>(n);
  mb /= static_cast<double>(n);
  double sab = 0.0;
  double saa = 0.0;
  double sbb = 0.0;
  for (int y = 0; y < mask.rows; ++y) {
    const float *pm = mask.ptr<float>(y);
    const int row_off = y * mask.cols;
    for (int x = 0; x < mask.cols; ++x) {
      if (pm[x] <= 0.5f) {
        continue;
      }
      const int idx = row_off + x;
      const double va = static_cast<double>(da[idx]) - ma;
      const double vb = static_cast<double>(db[idx]) - mb;
      sab += va * vb;
      saa += va * va;
      sbb += vb * vb;
    }
  }
  const double den = std::sqrt(saa * sbb);
  return (den > 1e-10) ? static_cast<float>(sab / den) : 0.0f;
}

/// @brief Implements register single frame.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
SingleFrameRegResult register_single_frame(const Matrix2Df &mov,
                                           const Matrix2Df &ref,
                                           const config::RegistrationConfig &rcfg,
                                           float min_ncc_improvement) {
  // Thread-safe diagnostic counter (only log first few calls)
  static std::atomic<int> diag_counter{0};
  const int diag_id = diag_counter.fetch_add(1);
  const bool diag = (diag_id < 10);

  SingleFrameRegResult out;
  out.reg.warp = identity_warp();
  out.reg.success = false;
  out.reg.correlation = 0.0f;
  out.reg.error_message.clear();
  out.method_used = "identity";
  out.ncc_identity = compute_ncc(mov, ref);
  out.ncc_warped = out.ncc_identity;

  const float identity_gate_margin = std::max(min_ncc_improvement, 0.0f);
  if (out.ncc_identity >= 1.0f - identity_gate_margin) {
    // Spec §4.2 binding edge case: if identity is already near-perfect,
    // no warp can satisfy the strict NCC improvement gate. Accept identity
    // directly and do not count it as a cascade failure.
    out.reg.success = true;
    out.reg.correlation = out.ncc_identity;
    if (diag) {
      std::cout << "[REG-DIAG#" << diag_id
                << "] identity accepted directly: ncc_identity="
                << out.ncc_identity
                << " threshold=" << (1.0f - identity_gate_margin)
                << std::endl;
    }
    return out;
  }

  if (diag) {
    auto stars_ref = detect_stars_simple(ref, rcfg.star_topk, rcfg.enable_local_background_subtraction);
    auto stars_mov = detect_stars_simple(mov, rcfg.star_topk, rcfg.enable_local_background_subtraction);
    std::cout << "[REG-DIAG#" << diag_id << "] ncc_identity=" << out.ncc_identity
              << " stars_ref=" << stars_ref.size()
              << " stars_mov=" << stars_mov.size()
              << " img=" << mov.rows() << "x" << mov.cols()
              << std::endl;
  }

  bool accepted = false;

  // Try a cascade method and validate with NCC
  auto try_method = [&](RegistrationResult rr,
                        const std::string &method) -> bool {
    if (!rr.success) {
      if (diag) {
        std::cout << "[REG-DIAG#" << diag_id << "] " << method
                  << " FAIL: " << rr.error_message << std::endl;
      }
      return false;
    }
    Matrix2Df warped = apply_warp(mov, rr.warp);
    const cv::Mat valid_mask = warp_valid_mask(mov, rr.warp);
    int overlap_pixels = 0;
    // Blur before NCC: makes the quality metric robust against hot pixels and
    // sharp point sources that cause extreme NCC sensitivity to sub-pixel shifts.
    // sigma=1.5 matches the ECC pre-processing blur (prepare_ecc_image).
    // Important: clamp to zero first so that negative background-subtracted
    // values do not bleed into star peaks via Gaussian blur.
    auto blur_for_ncc = [](const Matrix2Df &img) -> Matrix2Df {
      cv::Mat m(img.rows(), img.cols(), CV_32F,
                const_cast<float *>(img.data()));
      cv::Mat clamped;
      cv::max(m, 0.0f, clamped);          // remove negative background artefacts
      cv::GaussianBlur(clamped, clamped, cv::Size(0, 0), 1.5);
      Matrix2Df result(img.rows(), img.cols());
      std::memcpy(result.data(), clamped.data,
                  static_cast<size_t>(img.size()) * sizeof(float));
      return result;
    };
    const Matrix2Df mov_b    = blur_for_ncc(mov);
    const Matrix2Df warped_b = blur_for_ncc(warped);
    const Matrix2Df ref_b    = blur_for_ncc(ref);
    const float ncc_identity_overlap =
        compute_ncc_masked(mov_b, ref_b, valid_mask, &overlap_pixels);
    float ncc = compute_ncc_masked(warped_b, ref_b, valid_mask);
    if (diag) {
      const float angle_deg = std::atan2(-rr.warp(0,1), rr.warp(0,0)) * (180.0f / 3.14159265f);
      const float scale = std::sqrt(rr.warp(0,0)*rr.warp(0,0) + rr.warp(0,1)*rr.warp(0,1));
      std::cout << "[REG-DIAG#" << diag_id << "] " << method
                << " success=" << rr.success << " ncc_warped=" << ncc
                << " ncc_identity_overlap=" << ncc_identity_overlap
                << " overlap_px=" << overlap_pixels
                << " threshold=" << (ncc_identity_overlap + min_ncc_improvement)
                << " tx=" << rr.warp(0,2) << " ty=" << rr.warp(1,2)
                << " rot_deg=" << angle_deg << " scale=" << scale
                << " a00=" << rr.warp(0,0) << " a01=" << rr.warp(0,1)
                << " a10=" << rr.warp(1,0) << " a11=" << rr.warp(1,1)
                << std::endl;
    }
    // Near-identity bypass: if the warp is geometrically trivial (sub-pixel
    // shift + negligible rotation) AND does not degrade NCC significantly,
    // bilinear interpolation may slightly lower NCC vs. identity. Accept the
    // warp and report ncc_identity_overlap as the quality score so downstream
    // metrics are not penalised by interpolation blur.
    const float shift_total = std::sqrt(rr.warp(0,2)*rr.warp(0,2) +
                                        rr.warp(1,2)*rr.warp(1,2));
    const float angle_rad   = std::atan2(-rr.warp(0,1), rr.warp(0,0));
    const float angle_abs   = std::fabs(angle_rad) * (180.0f / 3.14159265f);
    // Only bypass the NCC gate when the warp is small AND non-degrading AND
    // the frame is already well-aligned with the reference (ncc_identity > 0.7).
    // For frames far from the reference a near-zero warp from star_pair just
    // means the method found no valid shift, not that the frame is already
    // aligned. A drop > 0.02 below blurred-identity indicates a false warp
    // even at small shifts, so we keep the NCC gate in that case.
    const bool near_identity = (shift_total < rcfg.star_inlier_tol_px) &&
                               (angle_abs   < 0.1f) &&
                               (ncc >= ncc_identity_overlap - 0.02f) &&
                               (out.ncc_identity > 0.7f);
    if (overlap_pixels <= 16 ||
        (!near_identity && ncc < ncc_identity_overlap + min_ncc_improvement))
      return false; // warp doesn't improve alignment — reject
    out.reg = rr;
    // For near-identity warps use the uninterpolated NCC so downstream
    // quality metrics are not degraded by interpolation blur.
    out.reg.correlation = near_identity ? ncc_identity_overlap : ncc;
    out.method_used = method;
    out.ncc_warped   = out.reg.correlation;
    return true;
  };

  auto try_star_methods = [&]() -> bool {
    bool ok = try_method(
        triangle_star_matching(mov, ref, rcfg.allow_rotation,
                               rcfg.star_topk, rcfg.star_min_inliers,
                               rcfg.star_inlier_tol_px,
                               rcfg.transform_model,
                               rcfg.enable_local_background_subtraction,
                               rcfg.star_shift_radius_px),
        "triangle");
    if (!ok && rcfg.enable_star_pair_fallback) {
      ok = try_method(
          star_registration_similarity(
              mov, ref, rcfg.allow_rotation, rcfg.star_topk,
              rcfg.star_min_inliers, rcfg.star_inlier_tol_px,
              rcfg.star_dist_bin_px, rcfg.transform_model,
              rcfg.enable_local_background_subtraction),
          "star_pair");
    }
    return ok;
  };

  // 1) Primary engine
  if (!accepted) {
    if (rcfg.engine == "triangle_star_matching" ||
        rcfg.engine == "star_similarity" || rcfg.engine.empty()) {
      accepted = try_star_methods();
    } else if (rcfg.engine == "opencv_feature") {
      accepted = try_method(
          feature_registration_similarity(mov, ref, rcfg.allow_rotation,
                                          rcfg.transform_model),
          "akaze");
    } else if (rcfg.engine == "robust_phase_ecc") {
      accepted = try_method(
          robust_phase_ecc(mov, ref, rcfg.allow_rotation),
          "robust_phase_ecc");
    } else {
      // Default: triangle
      accepted = try_method(
          triangle_star_matching(mov, ref, rcfg.allow_rotation,
                                rcfg.star_topk, rcfg.star_min_inliers,
                                rcfg.star_inlier_tol_px,
                                rcfg.transform_model,
                                rcfg.enable_local_background_subtraction,
                                rcfg.star_shift_radius_px),
          "triangle");
    }
  }

  // 2) Fallback cascade
  if (!accepted) {
    accepted = try_method(
        feature_registration_similarity(mov, ref, rcfg.allow_rotation,
                                        rcfg.transform_model),
        "akaze");
  }
  if (!accepted) {
    accepted = try_method(
        robust_phase_ecc(mov, ref, rcfg.allow_rotation),
        "robust_phase_ecc");
  }

  // 3) Final fallback: identity
  if (!accepted) {
    out.reg.warp = identity_warp();
    out.reg.correlation = 0.0f;
    out.reg.success = false;
    out.reg.error_message = "identity_fallback";
    out.method_used = "identity";
    out.ncc_warped = out.ncc_identity;
  }

  return out;
}

// =====================================================================
// register_frames_to_reference — NOT used by runner (runner calls
// register_single_frame directly with its own rescue passes).
// Kept as a thin wrapper for tests/CLI that may call it.
// =====================================================================

/// @brief Implements concatenate warps.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static WarpMatrix concatenate_warps(const WarpMatrix &w1, const WarpMatrix &w2) {
  WarpMatrix result;
  result(0,0) = w2(0,0)*w1(0,0) + w2(0,1)*w1(1,0);
  result(0,1) = w2(0,0)*w1(0,1) + w2(0,1)*w1(1,1);
  result(1,0) = w2(1,0)*w1(0,0) + w2(1,1)*w1(1,0);
  result(1,1) = w2(1,0)*w1(0,1) + w2(1,1)*w1(1,1);
  result(0,2) = w2(0,0)*w1(0,2) + w2(0,1)*w1(1,2) + w2(0,2);
  result(1,2) = w2(1,0)*w1(0,2) + w2(1,1)*w1(1,2) + w2(1,2);
  return result;
}

GlobalRegistrationOutput
/// @brief Implements register frames to reference.
/// @details Part of global registration algorithms, star matching, ECC fallback, and warp validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
register_frames_to_reference(const std::vector<Matrix2Df> &frames_fullres,
                             ColorMode mode, BayerPattern bayer,
                             const config::RegistrationConfig &rcfg,
                             const std::vector<FrameMetrics> *frame_metrics_opt,
                             const VectorXf *global_weights_opt) {

  GlobalRegistrationOutput out;
  const int n = static_cast<int>(frames_fullres.size());
  out.warps_fullres.assign(static_cast<size_t>(n), identity_warp());
  out.scores.assign(static_cast<size_t>(n), 0.0f);
  out.success.assign(static_cast<size_t>(n), false);
  out.errors.assign(static_cast<size_t>(n), "");
  out.engine_used = rcfg.engine;

  if (n == 0)
    return out;

  // Reference selection: prefer global weights (if available), then quality
  // score, else middle frame.
  out.ref_idx = n / 2;
  out.ref_selection_method = "middle";
  out.ref_selection_value = 0.0f;

  if (global_weights_opt && global_weights_opt->size() == n) {
    int best = 0;
    float best_v = (*global_weights_opt)[0];
    for (int i = 1; i < n; ++i) {
      float v = (*global_weights_opt)[i];
      if (v > best_v) {
        best_v = v;
        best = i;
      }
    }
    out.ref_idx = best;
    out.ref_selection_method = "global_weight";
    out.ref_selection_value = best_v;
  } else if (frame_metrics_opt &&
             static_cast<int>(frame_metrics_opt->size()) == n) {
    int best = 0;
    float best_v = (*frame_metrics_opt)[0].quality_score;
    for (int i = 1; i < n; ++i) {
      float v = (*frame_metrics_opt)[i].quality_score;
      if (v > best_v) {
        best_v = v;
        best = i;
      }
    }
    out.ref_idx = best;
    out.ref_selection_method = "quality_score";
    out.ref_selection_value = best_v;
  }

  // Build proxy images
  std::vector<Matrix2Df> proxy;
  proxy.reserve(static_cast<size_t>(n));
  const bool is_osc = (mode == ColorMode::OSC);
  for (int i = 0; i < n; ++i) {
    if (is_osc) {
      proxy.push_back(tile_compile::image::cfa_green_proxy_downsample2x2(
          frames_fullres[i], tile_compile::bayer_pattern_to_string(bayer)));
    } else {
      proxy.push_back(downsample2x2_mean(frames_fullres[i]));
    }
  }
  out.downsample_scale = 2.0f;

  const Matrix2Df ref_p = proxy[static_cast<size_t>(out.ref_idx)];

  for (int i = 0; i < n; ++i) {
    if (i == out.ref_idx) {
      out.success[static_cast<size_t>(i)] = true;
      out.scores[static_cast<size_t>(i)] = 1.0f;
      out.warps_fullres[static_cast<size_t>(i)] = identity_warp();
      continue;
    }

    const Matrix2Df mov_p = proxy[static_cast<size_t>(i)];

    // 1) Try direct registration to reference
    SingleFrameRegResult sfr = register_single_frame(mov_p, ref_p, rcfg);

    if (sfr.reg.success) {
      out.success[static_cast<size_t>(i)] = true;
      out.scores[static_cast<size_t>(i)] = sfr.reg.correlation;
      out.warps_fullres[static_cast<size_t>(i)] =
          scale_translation_warp(sfr.reg.warp, out.downsample_scale);
    } else {
      out.warps_fullres[static_cast<size_t>(i)] = identity_warp();
      out.scores[static_cast<size_t>(i)] = 0.0f;
      out.success[static_cast<size_t>(i)] = false;
      out.errors[static_cast<size_t>(i)] = sfr.reg.error_message;
    }
  }

  return out;
}

} // namespace tile_compile::registration
