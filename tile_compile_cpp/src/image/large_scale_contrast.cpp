#include "tile_compile/image/large_scale_contrast.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace tile_compile::image {
namespace {

// Large-scale map on a coarse grid: NaN-free values in `map`, `ok` marks cells
// that carry a trustworthy value (enough valid pixels, enough Gaussian support).
struct CoarseMap {
  cv::Mat map;  // CV_32F
  cv::Mat ok;   // CV_8U, 1 = usable
  int rows = 0, cols = 0;
};

double median_of(std::vector<float> v) {
  if (v.empty()) return 0.0;
  const std::size_t mid = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  double m = v[mid];
  if (v.size() % 2 == 0) {
    const double lo = *std::max_element(v.begin(), v.begin() + mid);
    m = 0.5 * (m + lo);
  }
  return m;
}

double percentile_of(std::vector<float> v, double p) {
  if (v.empty()) return 0.0;
  const std::size_t k = std::min(
      v.size() - 1, static_cast<std::size_t>(p * static_cast<double>(v.size() - 1)));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

CoarseMap build_map(const cv::Mat &plane, const cv::Mat &valid, int factor, float sigma_px) {
  CoarseMap out;
  const int rows = std::max(1, plane.rows / factor), cols = std::max(1, plane.cols / factor);
  out.rows = rows;
  out.cols = cols;
  cv::Mat num, den, valid_f;
  valid.convertTo(valid_f, CV_32F);
  cv::multiply(plane, valid_f, num);
  cv::Mat num_s, den_s;
  cv::resize(num, num_s, cv::Size(cols, rows), 0, 0, cv::INTER_AREA);
  cv::resize(valid_f, den_s, cv::Size(cols, rows), 0, 0, cv::INTER_AREA);
  cv::Mat cell_ok = den_s > 0.9f;  // cell mostly inside the valid area
  cv::Mat small(rows, cols, CV_32F, cv::Scalar(0));
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x)
      if (cell_ok.at<std::uint8_t>(y, x)) small.at<float>(y, x) = num_s.at<float>(y, x) / den_s.at<float>(y, x);
  // Fill unusable cells with the median of the usable ones so that the median
  // filter below does not see zeros at the border.
  std::vector<float> usable;
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x)
      if (cell_ok.at<std::uint8_t>(y, x)) usable.push_back(small.at<float>(y, x));
  const float fill = static_cast<float>(median_of(usable));
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x)
      if (!cell_ok.at<std::uint8_t>(y, x)) small.at<float>(y, x) = fill;
  cv::Mat med;
  if (rows >= 5 && cols >= 5) cv::medianBlur(small, med, 5);  // float median: ksize <= 5
  else med = small;
  // Masked (normalised) Gaussian: border cells are averaged only over usable neighbours.
  const double sigma = std::max(0.5, static_cast<double>(sigma_px) / factor);
  cv::Mat ok_f, med_ok, blur_num, blur_den;
  cell_ok.convertTo(ok_f, CV_32F, 1.0 / 255.0);
  cv::multiply(med, ok_f, med_ok);
  cv::GaussianBlur(med_ok, blur_num, cv::Size(0, 0), sigma, sigma, cv::BORDER_CONSTANT);
  cv::GaussianBlur(ok_f, blur_den, cv::Size(0, 0), sigma, sigma, cv::BORDER_CONSTANT);
  out.map = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
  out.ok = cv::Mat(rows, cols, CV_8U, cv::Scalar(0));
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x) {
      const float d = blur_den.at<float>(y, x);
      if (cell_ok.at<std::uint8_t>(y, x) && d > 0.5f) {
        out.map.at<float>(y, x) = blur_num.at<float>(y, x) / d;
        out.ok.at<std::uint8_t>(y, x) = 1;
      }
    }
  return out;
}

// Robust radial fit in r^2 (degree 3) over the usable cells; returns the fitted
// value at every cell. IRLS with a 2-sigma soft clip keeps a bright nebula
// from driving the vignette estimate.
cv::Mat radial_fit(const CoarseMap &m) {
  const int rows = m.rows, cols = m.cols;
  double cy = 0, cx = 0;
  std::size_t n = 0;
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x)
      if (m.ok.at<std::uint8_t>(y, x)) { cy += y; cx += x; ++n; }
  cv::Mat fit(rows, cols, CV_32F, cv::Scalar(0));
  if (n < 16) return fit;
  cy /= n; cx /= n;
  const double scale = std::max(cy, cx) * std::max(cy, cx) + 1e-9;
  constexpr int kDeg = 3;
  auto basis = [&](int y, int x, double *b) {
    const double r2 = ((y - cy) * (y - cy) + (x - cx) * (x - cx)) / scale;
    double p = 1.0;
    for (int k = 0; k <= kDeg; ++k) { b[k] = p; p *= r2; }
  };
  std::vector<double> w(n, 1.0);
  double coef[kDeg + 1] = {0, 0, 0, 0};
  for (int iter = 0; iter < 5; ++iter) {
    double ata[kDeg + 1][kDeg + 1] = {}, atb[kDeg + 1] = {};
    std::size_t i = 0;
    for (int y = 0; y < rows; ++y)
      for (int x = 0; x < cols; ++x) {
        if (!m.ok.at<std::uint8_t>(y, x)) continue;
        double b[kDeg + 1];
        basis(y, x, b);
        const double v = m.map.at<float>(y, x);
        for (int r = 0; r <= kDeg; ++r) {
          atb[r] += w[i] * b[r] * v;
          for (int c = 0; c <= kDeg; ++c) ata[r][c] += w[i] * b[r] * b[c];
        }
        ++i;
      }
    // Gaussian elimination on the 4x4 normal equations.
    double a[kDeg + 1][kDeg + 2];
    for (int r = 0; r <= kDeg; ++r) {
      for (int c = 0; c <= kDeg; ++c) a[r][c] = ata[r][c] + (r == c ? 1e-12 : 0.0);
      a[r][kDeg + 1] = atb[r];
    }
    for (int col = 0; col <= kDeg; ++col) {
      int piv = col;
      for (int r = col + 1; r <= kDeg; ++r)
        if (std::fabs(a[r][col]) > std::fabs(a[piv][col])) piv = r;
      if (std::fabs(a[piv][col]) < 1e-14) return fit;
      for (int c = 0; c <= kDeg + 1; ++c) std::swap(a[col][c], a[piv][c]);
      for (int r = col + 1; r <= kDeg; ++r) {
        const double f = a[r][col] / a[col][col];
        for (int c = col; c <= kDeg + 1; ++c) a[r][c] -= f * a[col][c];
      }
    }
    for (int r = kDeg; r >= 0; --r) {
      double s = a[r][kDeg + 1];
      for (int c = r + 1; c <= kDeg; ++c) s -= a[r][c] * coef[c];
      coef[r] = s / a[r][r];
    }
    std::vector<float> res;
    res.reserve(n);
    i = 0;
    std::vector<double> resid(n);
    for (int y = 0; y < rows; ++y)
      for (int x = 0; x < cols; ++x) {
        if (!m.ok.at<std::uint8_t>(y, x)) continue;
        double b[kDeg + 1];
        basis(y, x, b);
        double f = 0;
        for (int k = 0; k <= kDeg; ++k) f += coef[k] * b[k];
        resid[i] = m.map.at<float>(y, x) - f;
        res.push_back(static_cast<float>(resid[i]));
        ++i;
      }
    const double med = median_of(res);
    std::vector<float> dev;
    dev.reserve(n);
    for (float r : res) dev.push_back(std::fabs(r - static_cast<float>(med)));
    const double s = 1.4826 * median_of(dev) + 1e-9;
    for (std::size_t j = 0; j < n; ++j) w[j] = 1.0 / std::max(1.0, std::fabs(resid[j]) / (2.0 * s));
  }
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x) {
      double b[kDeg + 1];
      basis(y, x, b);
      double f = 0;
      for (int k = 0; k <= kDeg; ++k) f += coef[k] * b[k];
      fit.at<float>(y, x) = static_cast<float>(f);
    }
  return fit;
}

// Deviation map of a plane from its own sky level (median over usable cells),
// vignette component removed if asked; zero outside the usable cells.
cv::Mat deviation_map(const cv::Mat &plane, const cv::Mat &valid, int factor, float sigma_px,
                      bool remove_vignette, CoarseMap *map_out, double *reference_out) {
  CoarseMap m = build_map(plane, valid, factor, sigma_px);
  std::vector<float> usable;
  for (int y = 0; y < m.rows; ++y)
    for (int x = 0; x < m.cols; ++x)
      if (m.ok.at<std::uint8_t>(y, x)) usable.push_back(m.map.at<float>(y, x));
  const double ref = median_of(usable);
  cv::Mat dev(m.rows, m.cols, CV_32F, cv::Scalar(0));
  cv::Mat radial;
  if (remove_vignette) radial = radial_fit(m);
  for (int y = 0; y < m.rows; ++y)
    for (int x = 0; x < m.cols; ++x) {
      if (!m.ok.at<std::uint8_t>(y, x)) continue;
      float d = m.map.at<float>(y, x) - static_cast<float>(ref);
      if (remove_vignette) d -= radial.at<float>(y, x) - static_cast<float>(ref);
      dev.at<float>(y, x) = d;
    }
  if (map_out) *map_out = m;
  if (reference_out) *reference_out = ref;
  return dev;
}

double span_p5_p95(const cv::Mat &dev, const cv::Mat &ok) {
  std::vector<float> v;
  for (int y = 0; y < dev.rows; ++y)
    for (int x = 0; x < dev.cols; ++x)
      if (ok.at<std::uint8_t>(y, x)) v.push_back(dev.at<float>(y, x));
  return percentile_of(v, 0.95) - percentile_of(v, 0.05);
}

cv::Mat wrap(Matrix2Df &m) { return cv::Mat(static_cast<int>(m.rows()), static_cast<int>(m.cols()), CV_32F, m.data()); }
cv::Mat wrap_const(const Matrix2Df &m) {
  return cv::Mat(static_cast<int>(m.rows()), static_cast<int>(m.cols()), CV_32F, const_cast<float *>(m.data()));
}

}  // namespace

LargeScaleContrastResult apply_large_scale_contrast(
    Matrix2Df &R, Matrix2Df &G, Matrix2Df &B, const LargeScaleContrastConfig &cfg,
    const std::vector<std::uint8_t> *valid_mask) {
  LargeScaleContrastResult res;
  if (!cfg.enabled) return res;
  const int rows = static_cast<int>(R.rows()), cols = static_cast<int>(R.cols());
  if (rows < 32 || cols < 32 || G.rows() != R.rows() || G.cols() != R.cols() || B.rows() != R.rows() || B.cols() != R.cols()) {
    res.status = "error";
    res.error_message = "channels must have the same size and at least 32x32 pixels";
    return res;
  }
  if (!(cfg.amount >= 0.0f && cfg.chroma_amount >= 0.0f && cfg.sigma_px > 0.0f)) {
    res.status = "error";
    res.error_message = "amount and chroma_amount must be >= 0 and sigma_px > 0";
    return res;
  }
  if (cfg.amount == 0.0f && cfg.chroma_amount == 0.0f) {
    res.status = "not_needed";
    return res;
  }
  if (valid_mask && valid_mask->size() != static_cast<std::size_t>(rows) * cols) {
    res.status = "error";
    res.error_message = "valid mask size does not match the image";
    return res;
  }
  cv::Mat valid(rows, cols, CV_8U, cv::Scalar(0));
  std::size_t n_valid = 0;
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x) {
      const std::size_t i = static_cast<std::size_t>(y) * cols + x;
      bool v;
      if (valid_mask) {
        v = (*valid_mask)[i] != 0;
      } else {
        const float r = R(y, x), g = G(y, x), b = B(y, x);
        v = std::isfinite(r) && std::isfinite(g) && std::isfinite(b) && (r != 0.0f || g != 0.0f || b != 0.0f);
      }
      if (v) { valid.at<std::uint8_t>(y, x) = 1; ++n_valid; }
    }
  if (n_valid < static_cast<std::size_t>(rows) * cols / 20) {  // less than 5 % of the frame
    res.status = "too_few_valid_pixels";
    return res;
  }
  // Coarse grid of about sigma/6 pixels per cell (at least 1, at most 16).
  const int factor = std::clamp(static_cast<int>(std::floor(cfg.sigma_px / 6.0f)), 1, 16);
  res.downsample_factor = factor;

  Matrix2Df lum(rows, cols);
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x)
      lum(y, x) = valid.at<std::uint8_t>(y, x) ? (R(y, x) + G(y, x) + B(y, x)) / 3.0f : 0.0f;

  // Per-channel ceilings: the boost never pushes a channel beyond its own input maximum.
  float max_r = 0, max_g = 0, max_b = 0;
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x)
      if (valid.at<std::uint8_t>(y, x)) {
        max_r = std::max(max_r, R(y, x));
        max_g = std::max(max_g, G(y, x));
        max_b = std::max(max_b, B(y, x));
      }

  cv::Mat delta_l, delta_rg, delta_bg;
  CoarseMap lum_map;
  double ref = 0.0;
  cv::Mat dev_l = deviation_map(wrap_const(lum), valid, factor, cfg.sigma_px, cfg.remove_vignette, &lum_map, &ref);
  res.sky_reference = ref;
  res.vignette_removed = cfg.remove_vignette;
  res.span_before = span_p5_p95(dev_l, lum_map.ok);
  if (cfg.amount > 0.0f) {
    cv::Mat scaled = dev_l * cfg.amount;
    cv::resize(scaled, delta_l, cv::Size(cols, rows), 0, 0, cv::INTER_LINEAR);
    res.span_after = span_p5_p95(scaled, lum_map.ok);
  } else {
    res.span_after = res.span_before;
  }
  if (cfg.chroma_amount > 0.0f) {
    Matrix2Df rg(rows, cols), bg(rows, cols);
    for (int y = 0; y < rows; ++y)
      for (int x = 0; x < cols; ++x) {
        rg(y, x) = valid.at<std::uint8_t>(y, x) ? R(y, x) - G(y, x) : 0.0f;
        bg(y, x) = valid.at<std::uint8_t>(y, x) ? B(y, x) - G(y, x) : 0.0f;
      }
    cv::Mat d_rg = deviation_map(wrap_const(rg), valid, factor, cfg.sigma_px, cfg.remove_vignette, nullptr, nullptr) * cfg.chroma_amount;
    cv::Mat d_bg = deviation_map(wrap_const(bg), valid, factor, cfg.sigma_px, cfg.remove_vignette, nullptr, nullptr) * cfg.chroma_amount;
    cv::resize(d_rg, delta_rg, cv::Size(cols, rows), 0, 0, cv::INTER_LINEAR);
    cv::resize(d_bg, delta_bg, cv::Size(cols, rows), 0, 0, cv::INTER_LINEAR);
  }
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x) {
      if (!valid.at<std::uint8_t>(y, x)) continue;
      const float dl = delta_l.empty() ? 0.0f : delta_l.at<float>(y, x);
      const float dr = delta_rg.empty() ? 0.0f : delta_rg.at<float>(y, x);
      const float db = delta_bg.empty() ? 0.0f : delta_bg.at<float>(y, x);
      R(y, x) = std::clamp(R(y, x) + dl + dr, 0.0f, std::max(max_r, R(y, x)));
      G(y, x) = std::clamp(G(y, x) + dl, 0.0f, std::max(max_g, G(y, x)));
      B(y, x) = std::clamp(B(y, x) + dl + db, 0.0f, std::max(max_b, B(y, x)));
    }
  res.applied = true;
  res.status = "applied";
  return res;
}

}  // namespace tile_compile::image
