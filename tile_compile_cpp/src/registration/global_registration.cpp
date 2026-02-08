#include "tile_compile/registration/global_registration.hpp"

#include "tile_compile/core/utils.hpp"
#include "tile_compile/image/cfa_processing.hpp"
#include "tile_compile/registration/registration.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>

namespace tile_compile::registration {

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

WarpMatrix scale_translation_warp(const WarpMatrix &w, float scale) {
  WarpMatrix out = w;
  out(0, 2) *= scale;
  out(1, 2) *= scale;
  return out;
}

// --- Sub-functions (exported via header) ---

// Estimate rotation (deg) between ref and mov using log-polar phase correlation
// on magnitude spectrum.
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
  cv::logPolar(mag_ref, lp_ref, center, M, cv::WARP_FILL_OUTLIERS);
  cv::logPolar(mag_mov, lp_mov, center, M, cv::WARP_FILL_OUTLIERS);

  cv::Point2d shift = cv::phaseCorrelate(lp_mov, lp_ref);
  double rotation_deg = -shift.y * 360.0 / static_cast<double>(lp_ref.rows);
  return static_cast<float>(rotation_deg);
}

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

RegistrationResult feature_registration_similarity(const Matrix2Df &mov,
                                                   const Matrix2Df &ref,
                                                   bool allow_rotation) {
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

  cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
  std::vector<cv::KeyPoint> kps_ref, kps_mov;
  cv::Mat desc_ref, desc_mov;
  akaze->detectAndCompute(ref_cv, cv::noArray(), kps_ref, desc_ref);
  akaze->detectAndCompute(mov_cv, cv::noArray(), kps_mov, desc_mov);

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
  cv::Mat A = cv::estimateAffinePartial2D(pts_mov, pts_ref, inliers, cv::RANSAC,
                                          3.0, 2000, 0.99);
  if (A.empty()) {
    res.error_message = "affine fail";
    return res;
  }

  // Forward warp (M→R) from estimateAffinePartial2D
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
  // No rotation limit: AKAZE features are rotation-invariant.

  // Invert to (R→M) for apply_warp with WARP_INVERSE_MAP
  const float det = a00_fw * a11_fw - a01_fw * a10_fw;
  if (std::fabs(det) < 1e-8f) {
    res.error_message = "singular matrix";
    return res;
  }
  const float inv_det = 1.0f / det;
  const float a00_inv = a11_fw * inv_det;
  const float a01_inv = -a01_fw * inv_det;
  const float a10_inv = -a10_fw * inv_det;
  const float a11_inv = a00_fw * inv_det;
  const float tx_inv = -(a00_inv * tx_fw + a01_inv * ty_fw);
  const float ty_inv = -(a10_inv * tx_fw + a11_inv * ty_fw);

  res.warp << a00_inv, a01_inv, tx_inv, a10_inv, a11_inv, ty_inv;
  int inl = inliers.empty() ? 0 : cv::countNonZero(inliers);
  res.correlation = matches.empty() ? 0.0f
                                    : static_cast<float>(inl) /
                                          static_cast<float>(matches.size());
  res.success = res.correlation > 0.1f;
  return res;
}

struct StarPoint {
  float x = 0.0f;
  float y = 0.0f;
  float flux = 0.0f;
};

std::vector<StarPoint> detect_stars_simple(const Matrix2Df &img, int topk) {
  const int h = img.rows();
  const int w = img.cols();
  if (h < 5 || w < 5)
    return {};

  std::vector<float> pixels;
  pixels.reserve(static_cast<size_t>(img.size()));
  for (int y = 0; y < h; ++y) {
    const float *row = img.data() + static_cast<size_t>(y) * w;
    pixels.insert(pixels.end(), row, row + w);
  }
  float med = core::median_of(pixels);
  float sigma = core::robust_sigma_mad(pixels);
  if (sigma < 1.0e-6f)
    sigma = 1.0f;
  const float thresh = med + 3.5f * sigma;

  std::vector<StarPoint> stars;
  stars.reserve(static_cast<size_t>(topk) * 2);
  for (int y = 1; y < h - 1; ++y) {
    const float *row = img.data() + static_cast<size_t>(y) * w;
    for (int x = 1; x < w - 1; ++x) {
      const float v = row[x];
      if (v < thresh)
        continue;
      bool is_max = true;
      for (int dy = -1; dy <= 1 && is_max; ++dy) {
        const float *r2 = img.data() + static_cast<size_t>(y + dy) * w;
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0)
            continue;
          if (r2[x + dx] >= v) {
            is_max = false;
            break;
          }
        }
      }
      if (!is_max)
        continue;

      float wsum = 0.0f;
      float xs = 0.0f;
      float ys = 0.0f;
      for (int dy = -1; dy <= 1; ++dy) {
        const float *r2 = img.data() + static_cast<size_t>(y + dy) * w;
        for (int dx = -1; dx <= 1; ++dx) {
          const float val = r2[x + dx] - med;
          if (val <= 0.0f)
            continue;
          wsum += val;
          xs += (static_cast<float>(x + dx) * val);
          ys += (static_cast<float>(y + dy) * val);
        }
      }
      if (wsum <= 0.0f)
        continue;
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

struct SimilarityResult {
  bool ok = false;
  float scale = 1.0f;
  float theta = 0.0f; // radians
  Eigen::Vector2f t{0.0f, 0.0f};
  int inliers = 0;
  float mean_err = 1.0e9f;
};

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

RegistrationResult
star_registration_similarity(const Matrix2Df &mov, const Matrix2Df &ref,
                             bool allow_rotation,
                             int topk_stars, int min_inliers,
                             float inlier_tol_px, float dist_bin_px) {
  RegistrationResult res;
  res.warp = identity_warp();
  res.success = false;
  res.correlation = 0.0f;

  auto mov_stars = detect_stars_simple(mov, topk_stars);
  auto ref_stars = detect_stars_simple(ref, topk_stars);
  if (mov_stars.size() < 3 || ref_stars.size() < 3) {
    res.error_message = "too_few_stars";
    return res;
  }

  struct Pair {
    int i;
    int j;
    float dist;
  };
  std::vector<Pair> ref_pairs;
  std::vector<Pair> mov_pairs;

  auto build_pairs = [&](const std::vector<StarPoint> &stars,
                         std::vector<Pair> &out) {
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
              [](const Pair &a, const Pair &b) { return a.dist > b.dist; });
    const size_t limit = std::min<size_t>(out.size(), 800);
    out.resize(limit);
  };

  build_pairs(ref_stars, ref_pairs);
  build_pairs(mov_stars, mov_pairs);
  if (ref_pairs.empty() || mov_pairs.empty()) {
    res.error_message = "no_pairs";
    return res;
  }

  std::unordered_map<int, std::vector<Pair>> ref_bucket;
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
};

static std::vector<Triangle>
build_triangles(const std::vector<StarPoint> &stars, int max_triangles) {
  const int n = static_cast<int>(stars.size());
  std::vector<Triangle> tris;
  if (n < 3)
    return tris;

  // Limit combinatorial explosion: use top stars only
  const int limit = std::min(n, 30);
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

        // Skip degenerate triangles
        if (d_ij < 3.0f || d_ik < 3.0f || d_jk < 3.0f)
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
triangle_star_matching(const Matrix2Df &mov, const Matrix2Df &ref,
                       bool allow_rotation,
                       int topk_stars, int min_inliers,
                       float inlier_tol_px) {
  RegistrationResult res;
  res.warp = identity_warp();
  res.success = false;
  res.correlation = 0.0f;

  auto mov_stars = detect_stars_simple(mov, topk_stars);
  auto ref_stars = detect_stars_simple(ref, topk_stars);

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
  const float ratio_tol = 0.03f; // tolerance for ratio matching

  // Collect star correspondences from matched triangles
  std::vector<cv::Point2f> pts_mov, pts_ref;
  pts_mov.reserve(mov_tris.size() * 3);
  pts_ref.reserve(ref_tris.size() * 3);

  int matches_found = 0;
  for (const auto &mt : mov_tris) {
    float best_dist = ratio_tol * 2.0f;
    const Triangle *best_rt = nullptr;

    for (const auto &rt : ref_tris) {
      float dr0 = mt.ratios[0] - rt.ratios[0];
      float dr1 = mt.ratios[1] - rt.ratios[1];
      float d = std::sqrt(dr0 * dr0 + dr1 * dr1);
      if (d < best_dist) {
        best_dist = d;
        best_rt = &rt;
      }
    }

    if (!best_rt || best_dist >= ratio_tol)
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
      pts_mov.push_back(cv::Point2f(mov_stars[mov_order[v]].x,
                                     mov_stars[mov_order[v]].y));
      pts_ref.push_back(cv::Point2f(ref_stars[ref_order[v]].x,
                                     ref_stars[ref_order[v]].y));
    }

    if (matches_found > 200)
      break;
  }

  if (pts_mov.size() < 6) {
    res.error_message = "few_triangle_matches";
    return res;
  }

  // Use RANSAC to find the best similarity transform from correspondences
  cv::Mat inliers;
  cv::Mat A = cv::estimateAffinePartial2D(pts_mov, pts_ref, inliers,
                                          cv::RANSAC, 3.0, 2000, 0.99);
  if (A.empty()) {
    res.error_message = "affine_fail";
    return res;
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
  // No rotation limit: triangle asterism matching is rotation-invariant.

  // Invert to (R→M) for apply_warp with WARP_INVERSE_MAP
  const float det = a00_fw * a11_fw - a01_fw * a10_fw;
  if (std::fabs(det) < 1e-8f) {
    res.error_message = "singular_matrix";
    return res;
  }
  const float inv_det = 1.0f / det;
  const float a00_inv = a11_fw * inv_det;
  const float a01_inv = -a01_fw * inv_det;
  const float a10_inv = -a10_fw * inv_det;
  const float a11_inv = a00_fw * inv_det;
  const float tx_inv = -(a00_inv * tx_fw + a01_inv * ty_fw);
  const float ty_inv = -(a10_inv * tx_fw + a11_inv * ty_fw);

  res.warp << a00_inv, a01_inv, tx_inv, a10_inv, a11_inv, ty_inv;
  int inl = inliers.empty() ? 0 : cv::countNonZero(inliers);
  res.correlation =
      pts_mov.empty()
          ? 0.0f
          : static_cast<float>(inl) / static_cast<float>(pts_mov.size());
  res.success = inl >= min_inliers && res.correlation > 0.05f;
  return res;
}

RegistrationResult hybrid_phase_ecc(const Matrix2Df &mov, const Matrix2Df &ref,
                                    bool allow_rotation) {
  RegistrationResult res;
  res.warp = identity_warp();
  res.success = false;
  res.correlation = 0.0f;

  Matrix2Df mov_ecc = prepare_ecc_image(mov);
  Matrix2Df ref_ecc = prepare_ecc_image(ref);

  auto [dx, dy] = phasecorr_translation(mov_ecc, ref_ecc);

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
  return res;
}

GlobalRegistrationOutput
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

  // Prepare ECC images once for ref
  const Matrix2Df ref_ecc = prepare_ecc_image(ref_p);
  cv::Mat ref_cv(ref_ecc.rows(), ref_ecc.cols(), CV_32F,
                 const_cast<float *>(ref_ecc.data()));

  for (int i = 0; i < n; ++i) {
    if (i == out.ref_idx) {
      out.success[static_cast<size_t>(i)] = true;
      out.scores[static_cast<size_t>(i)] = 1.0f;
      out.warps_fullres[static_cast<size_t>(i)] = identity_warp();
      continue;
    }

    const Matrix2Df mov_p = proxy[static_cast<size_t>(i)];

    RegistrationResult rr;

    // Cascaded registration: try primary engine, then fallbacks.
    // v3: ALL frames must be used — failed registration gets identity warp.

    // 1) Primary engine
    if (rcfg.engine == "triangle_star_matching" ||
        rcfg.engine == "star_similarity") {
      rr = triangle_star_matching(mov_p, ref_p, rcfg.allow_rotation,
                                  rcfg.star_topk, rcfg.star_min_inliers,
                                  rcfg.star_inlier_tol_px);
      if (!rr.success) {
        rr = star_registration_similarity(
            mov_p, ref_p, rcfg.allow_rotation,
            rcfg.star_topk, rcfg.star_min_inliers, rcfg.star_inlier_tol_px,
            rcfg.star_dist_bin_px);
      }
    } else if (rcfg.engine == "opencv_feature") {
      rr = feature_registration_similarity(mov_p, ref_p, rcfg.allow_rotation);
    } else if (rcfg.engine == "hybrid_phase_ecc") {
      rr = hybrid_phase_ecc(mov_p, ref_p, rcfg.allow_rotation);
    } else {
      rr = triangle_star_matching(mov_p, ref_p, rcfg.allow_rotation,
                                  rcfg.star_topk, rcfg.star_min_inliers,
                                  rcfg.star_inlier_tol_px);
    }

    // 2) Fallback cascade if primary failed
    if (!rr.success) {
      rr = feature_registration_similarity(mov_p, ref_p, rcfg.allow_rotation);
    }
    if (!rr.success) {
      rr = hybrid_phase_ecc(mov_p, ref_p, rcfg.allow_rotation);
    }

    if (!rr.success) {
      out.warps_fullres[static_cast<size_t>(i)] = identity_warp();
      out.scores[static_cast<size_t>(i)] = 0.0f;
      out.success[static_cast<size_t>(i)] = false;
      out.errors[static_cast<size_t>(i)] = rr.error_message;
      continue;
    }

    out.success[static_cast<size_t>(i)] = true;
    out.scores[static_cast<size_t>(i)] = rr.correlation;
    out.warps_fullres[static_cast<size_t>(i)] =
        scale_translation_warp(rr.warp, out.downsample_scale);
  }

  return out;
}

} // namespace tile_compile::registration
