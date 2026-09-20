#include "tile_compile/image/color_cast_correction.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tile_compile::image {
namespace {

constexpr std::size_t kMaxSamples = 400000;

double median_of(std::vector<double> &v) {
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

double percentile_of(std::vector<double> &v, double p) {
  if (v.empty()) return 0.0;
  const std::size_t k = std::min(
      v.size() - 1, static_cast<std::size_t>(p * static_cast<double>(v.size() - 1)));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

// 9x9 box blur (edge-clamped) via separable running sums.
std::vector<float> box_blur9(const std::vector<float> &src, int rows, int cols) {
  const int r = 4;
  std::vector<float> tmp(src.size()), out(src.size());
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      double acc = 0.0;
      for (int k = -r; k <= r; ++k) {
        const int xx = std::clamp(x + k, 0, cols - 1);
        acc += src[static_cast<std::size_t>(y) * cols + xx];
      }
      tmp[static_cast<std::size_t>(y) * cols + x] = static_cast<float>(acc / (2 * r + 1));
    }
  }
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      double acc = 0.0;
      for (int k = -r; k <= r; ++k) {
        const int yy = std::clamp(y + k, 0, rows - 1);
        acc += tmp[static_cast<std::size_t>(yy) * cols + x];
      }
      out[static_cast<std::size_t>(y) * cols + x] = static_cast<float>(acc / (2 * r + 1));
    }
  }
  return out;
}

}  // namespace

ColorCastCorrectionResult apply_color_cast_correction(
    Matrix2Df &R, Matrix2Df &G, Matrix2Df &B,
    const ColorCastCorrectionConfig &cfg,
    const std::vector<std::uint8_t> *valid_mask) {
  ColorCastCorrectionResult res;
  if (!cfg.enabled) return res;
  const int rows = static_cast<int>(R.rows()), cols = static_cast<int>(R.cols());
  if (rows <= 0 || cols <= 0 || G.rows() != rows || B.rows() != rows ||
      G.cols() != cols || B.cols() != cols) {
    res.status = "error";
    res.error_message = "RGB dimensions mismatch";
    return res;
  }
  if (!(cfg.max_amount > 0.0f && cfg.max_amount <= 1.0f) ||
      !(cfg.target_ratio > 0.0f) || !(cfg.min_excess >= 1.0f) ||
      !(cfg.object_sigma > 0.0f)) {
    res.status = "error";
    res.error_message = "invalid color_cast_correction parameters";
    return res;
  }
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  const bool use_mask = valid_mask != nullptr && valid_mask->size() == n;

  std::vector<std::uint8_t> valid(n, 0);
  std::vector<float> lum(n, 0.0f);
  std::size_t nvalid = 0;
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      const std::size_t i = static_cast<std::size_t>(y) * cols + x;
      const float r = R(y, x), g = G(y, x), b = B(y, x);
      bool ok = std::isfinite(r) && std::isfinite(g) && std::isfinite(b);
      if (ok) ok = use_mask ? ((*valid_mask)[i] != 0) : !(r == 0.0f && g == 0.0f && b == 0.0f);
      if (!ok) continue;
      valid[i] = 1;
      lum[i] = (r + g + b) / 3.0f;
      ++nvalid;
    }
  }
  if (nvalid < 1000) {
    res.status = "too_few_object_pixels";
    return res;
  }
  // Invalid pixels take the median luminance so they do not bias the blur.
  {
    std::vector<double> t;
    const std::size_t step = std::max<std::size_t>(1, n / kMaxSamples);
    for (std::size_t i = 0; i < n; i += step)
      if (valid[i]) t.push_back(lum[i]);
    const float fill = static_cast<float>(median_of(t));
    for (std::size_t i = 0; i < n; ++i)
      if (!valid[i]) lum[i] = fill;
  }
  // Blurred luminance WITHOUT the centre pixel: the object/sky decision must not
  // depend on the pixel's own noise, otherwise the green excess of marginal
  // pixels is biased (selection on a variable that contains G).
  std::vector<float> lb = box_blur9(lum, rows, cols);
  for (std::size_t i = 0; i < n; ++i) lb[i] = (81.0f * lb[i] - lum[i]) / 80.0f;

  // Sky reference: the darkest 30 % of the blurred luminance.
  const std::size_t step = std::max<std::size_t>(1, n / kMaxSamples);
  std::vector<double> lb_samples;
  for (std::size_t i = 0; i < n; i += step)
    if (valid[i]) lb_samples.push_back(lb[i]);
  const double dark_thr = percentile_of(lb_samples, 0.30);
  std::vector<double> dark_l, dark_r, dark_g, dark_b;
  for (std::size_t i = 0; i < n; i += step) {
    if (!valid[i] || lb[i] > dark_thr) continue;
    const int y = static_cast<int>(i / cols), x = static_cast<int>(i % cols);
    dark_l.push_back(lb[i]);
    dark_r.push_back(R(y, x));
    dark_g.push_back(G(y, x));
    dark_b.push_back(B(y, x));
  }
  if (dark_l.size() < 100) {
    res.status = "too_few_object_pixels";
    return res;
  }
  const double sky_l = median_of(dark_l);
  {
    // Noise of the blurred luminance: robust spread of differences between
    // blurred values 9 px apart (disjoint windows), immune to the truncation of
    // the dark set and to slowly varying structure.
    std::vector<double> diffs;
    for (int y = 0; y < rows; y += 3) {
      for (int x = 0; x + 9 < cols; x += 3) {
        const std::size_t i = static_cast<std::size_t>(y) * cols + x;
        if (valid[i] && valid[i + 9]) diffs.push_back(static_cast<double>(lb[i]) - lb[i + 9]);
      }
    }
    double sigma = 0.0;
    if (diffs.size() >= 100) {
      const double med = median_of(diffs);
      for (double& d : diffs) d = std::fabs(d - med);
      sigma = 1.4826 * median_of(diffs) / std::sqrt(2.0);
    }
    res.sky_r = median_of(dark_r);
    res.sky_g = median_of(dark_g);
    res.sky_b = median_of(dark_b);
    // Object pixels: clearly above the sky, without the brightest 1 % (stars).
    std::vector<double> excess;
    for (std::size_t i = 0; i < n; i += step)
      if (valid[i]) excess.push_back(lb[i] - sky_l);
    const double upper = percentile_of(excess, 0.99);
    const double lower = static_cast<double>(cfg.object_sigma) * std::max(sigma, 1e-9);
    std::vector<double> xg, avg;
    std::size_t n_obj = 0;
    for (std::size_t i = 0; i < n; ++i) {
      if (!valid[i]) continue;
      const double e = lb[i] - sky_l;
      if (e > lower && e <= upper) ++n_obj;
    }
    const std::size_t ostep = std::max<std::size_t>(1, n_obj / kMaxSamples);
    std::size_t seen = 0;
    for (std::size_t i = 0; i < n; ++i) {
      if (!valid[i]) continue;
      const double e = lb[i] - sky_l;
      if (!(e > lower && e <= upper)) continue;
      if ((seen++ % ostep) != 0) continue;
      const int y = static_cast<int>(i / cols), x = static_cast<int>(i % cols);
      xg.push_back(G(y, x) - res.sky_g);
      avg.push_back(0.5 * ((R(y, x) - res.sky_r) + (B(y, x) - res.sky_b)));
    }
    res.object_pixels = xg.size();
    if (xg.size() < 1000) {
      res.status = "too_few_object_pixels";
      return res;
    }
    auto ratio_at = [&](double a) {
      std::vector<double> g(xg.size()), m(xg.size());
      for (std::size_t k = 0; k < xg.size(); ++k) {
        const double ref = std::max(avg[k], 0.0);
        g[k] = xg[k] - a * std::max(0.0, xg[k] - ref);
        m[k] = avg[k];
      }
      const double mg = median_of(g), mm = median_of(m);
      return mm > 0.0 ? mg / mm : 0.0;
    };
    res.ratio_before = ratio_at(0.0);
    res.ratio_after = res.ratio_before;
    if (!(res.ratio_before > static_cast<double>(cfg.min_excess))) {
      res.status = "not_needed";
      return res;
    }
    double amount = static_cast<double>(cfg.max_amount);
    if (ratio_at(amount) < static_cast<double>(cfg.target_ratio)) {
      double lo = 0.0, hi = amount;  // ratio decreases with the amount
      for (int it = 0; it < 24; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (ratio_at(mid) > static_cast<double>(cfg.target_ratio)) lo = mid;
        else hi = mid;
      }
      amount = 0.5 * (lo + hi);
    }
    res.amount = static_cast<float>(amount);
    res.ratio_after = ratio_at(amount);
  }
  // Apply to every valid pixel.
  const double a = static_cast<double>(res.amount);
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      if (!valid[static_cast<std::size_t>(y) * cols + x]) continue;
      const double xg = G(y, x) - res.sky_g;
      const double ref = std::max(0.0, 0.5 * ((R(y, x) - res.sky_r) + (B(y, x) - res.sky_b)));
      const double ex = std::max(0.0, xg - ref);
      if (ex > 0.0) G(y, x) = static_cast<float>(G(y, x) - a * ex);
    }
  }
  res.applied = true;
  res.status = "applied";
  return res;
}

}  // namespace tile_compile::image
