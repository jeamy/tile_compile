#include "tile_compile/reconstruction/source_quality_proxy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace tile_compile::reconstruction {

namespace {

// Separable [1,4,6,4,1]/16 pass along one axis, clamp-to-edge boundary.
void b3_pass_rows(const Matrix2Df &in, Matrix2Df &out) {
  const int H = static_cast<int>(in.rows());
  const int W = static_cast<int>(in.cols());
  out.resize(H, W);
  static const double k[5] = {1.0, 4.0, 6.0, 4.0, 1.0};
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      double sum = 0.0;
      for (int t = -2; t <= 2; ++t) {
        const int xx = std::clamp(x + t, 0, W - 1);
        sum += k[t + 2] * static_cast<double>(in(y, xx));
      }
      out(y, x) = static_cast<float>(sum / 16.0);
    }
  }
}

void b3_pass_cols(const Matrix2Df &in, Matrix2Df &out) {
  const int H = static_cast<int>(in.rows());
  const int W = static_cast<int>(in.cols());
  out.resize(H, W);
  static const double k[5] = {1.0, 4.0, 6.0, 4.0, 1.0};
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      double sum = 0.0;
      for (int t = -2; t <= 2; ++t) {
        const int yy = std::clamp(y + t, 0, H - 1);
        sum += k[t + 2] * static_cast<double>(in(yy, x));
      }
      out(y, x) = static_cast<float>(sum / 16.0);
    }
  }
}

}  // namespace

Matrix2Df b3_spline_blur(const Matrix2Df &input) {
  Matrix2Df rows_pass, result;
  b3_pass_rows(input, rows_pass);
  b3_pass_cols(rows_pass, result);
  return result;
}

double median_absolute_deviation_sigma(const Matrix2Df &values) {
  std::vector<float> finite;
  finite.reserve(static_cast<size_t>(values.rows()) * static_cast<size_t>(values.cols()));
  for (int y = 0; y < values.rows(); ++y)
    for (int x = 0; x < values.cols(); ++x) {
      const float v = values(y, x);
      if (std::isfinite(v)) finite.push_back(v);
    }
  if (finite.empty()) return 0.0;

  auto median_of = [](std::vector<float> &v) -> double {
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    double m = v[mid];
    if (v.size() % 2 == 0) {
      std::nth_element(v.begin(), v.begin() + mid - 1, v.begin() + mid);
      m = 0.5 * (m + v[mid - 1]);
    }
    return m;
  };

  const double med = median_of(finite);
  for (float &v : finite) v = static_cast<float>(std::abs(v - med));
  const double mad = median_of(finite);
  return 1.4826 * mad;
}

namespace {

// Edge-aware full-resolution green proxy at one R/B site (plan 13.2 step 3):
// prefers the neighbour-pair direction (horizontal or vertical) with the
// lower local gradient, so the proxy never interpolates across an edge.
float edge_aware_green_at(const Matrix2Df &source, int x, int y, int width, int height) {
  const bool have_w = x > 0, have_e = x < width - 1;
  const bool have_n = y > 0, have_s = y < height - 1;
  const double west = have_w ? static_cast<double>(source(y, x - 1)) : 0.0;
  const double east = have_e ? static_cast<double>(source(y, x + 1)) : 0.0;
  const double north = have_n ? static_cast<double>(source(y - 1, x)) : 0.0;
  const double south = have_s ? static_cast<double>(source(y + 1, x)) : 0.0;

  const bool horiz_ok = have_w && have_e;
  const bool vert_ok = have_n && have_s;
  const bool horiz_avail = have_w || have_e;
  const bool vert_avail = have_n || have_s;

  const double horiz_val =
      horiz_ok ? 0.5 * (west + east) : (have_w ? west : (have_e ? east : 0.0));
  const double vert_val =
      vert_ok ? 0.5 * (north + south) : (have_n ? north : (have_s ? south : 0.0));
  const double horiz_grad = horiz_ok ? std::abs(west - east) : std::numeric_limits<double>::infinity();
  const double vert_grad = vert_ok ? std::abs(north - south) : std::numeric_limits<double>::infinity();

  if (!horiz_avail && !vert_avail) return source(y, x);  // 1x1 degenerate case
  if (!vert_avail) return static_cast<float>(horiz_val);
  if (!horiz_avail) return static_cast<float>(vert_val);
  return static_cast<float>(horiz_grad <= vert_grad ? horiz_val : vert_val);
}

}  // namespace

SourceQualityProxyResult compute_source_quality_proxy_v1(const Matrix2Df &source,
                                                          ColorMode color_mode,
                                                          BayerPattern bayer_pattern,
                                                          int cfa_origin_x, int cfa_origin_y) {
  SourceQualityProxyResult result;
  const int H = static_cast<int>(source.rows());
  const int W = static_cast<int>(source.cols());

  if (color_mode == ColorMode::MONO) {
    // Plan 13.2: "MONO verwendet direkt die normalisierte L-Ebene als Proxy
    // ... ohne CFA-Interpolation."
    result.proxy_full = source;
    result.sigma_green = median_absolute_deviation_sigma(source - b3_spline_blur(source));
    return result;
  }

  // --- Quad-Green grid (plan 13.2 step 1) ---
  const int qw = W / 2, qh = H / 2;
  result.quad_width = qw;
  result.quad_height = qh;
  result.quad_green.resize(qh, qw);
  for (int qy = 0; qy < qh; ++qy) {
    for (int qx = 0; qx < qw; ++qx) {
      const int sx = qx * 2, sy = qy * 2;
      double green_sum = 0.0;
      for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
          if (cfa_channel_for_source_pixel(sx + dx, sy + dy, bayer_pattern, cfa_origin_x,
                                           cfa_origin_y) == CfaChannel::G) {
            green_sum += source(sy + dy, sx + dx);
          }
        }
      }
      result.quad_green(qy, qx) = static_cast<float>(green_sum * 0.5);  // 0.5*(G1+G2)
    }
  }

  // --- global sigma_green (plan 13.2 step 2), on the Quad-Green grid ---
  if (qw > 0 && qh > 0) {
    result.sigma_green =
        median_absolute_deviation_sigma(result.quad_green - b3_spline_blur(result.quad_green));
  }

  // --- full-resolution edge-aware green proxy (plan 13.2 step 3) ---
  result.proxy_full.resize(H, W);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const CfaChannel c = cfa_channel_for_source_pixel(x, y, bayer_pattern, cfa_origin_x,
                                                        cfa_origin_y);
      result.proxy_full(y, x) =
          (c == CfaChannel::G) ? source(y, x) : edge_aware_green_at(source, x, y, W, H);
    }
  }

  return result;
}

}  // namespace tile_compile::reconstruction
