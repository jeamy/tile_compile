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
      !(cfg.object_sigma > 0.0f) || cfg.brightness_bins < 1 ||
      cfg.brightness_bins > 32) {
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
    struct Sample { double e, xg, avg; };
    std::vector<Sample> smp;
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
      smp.push_back({e, G(y, x) - res.sky_g,
                     0.5 * ((R(y, x) - res.sky_r) + (B(y, x) - res.sky_b))});
    }
    res.object_pixels = smp.size();
    if (smp.size() < 1000) {
      res.status = "too_few_object_pixels";
      return res;
    }
    std::sort(smp.begin(), smp.end(),
              [](const Sample &a, const Sample &b) { return a.e < b.e; });
    // Equal-count brightness classes (fewer when the object is small).
    const int bins = std::max(1, std::min<int>(cfg.brightness_bins,
                                               static_cast<int>(smp.size() / 1000)));
    auto class_ratio = [&](std::size_t b0, std::size_t b1, double a) {
      std::vector<double> g, m;
      g.reserve(b1 - b0);
      m.reserve(b1 - b0);
      for (std::size_t k = b0; k < b1; ++k) {
        const double ref = std::max(smp[k].avg, 0.0);
        g.push_back(smp[k].xg - a * std::max(0.0, smp[k].xg - ref));
        m.push_back(smp[k].avg);
      }
      const double mg = median_of(g), mm = median_of(m);
      return mm > 0.0 ? mg / mm : 0.0;
    };
    std::vector<double> centers(static_cast<std::size_t>(bins)), amt(static_cast<std::size_t>(bins), 0.0);
    std::vector<std::size_t> edge(static_cast<std::size_t>(bins) + 1);
    for (int k = 0; k <= bins; ++k) edge[k] = smp.size() * static_cast<std::size_t>(k) / bins;
    const double max_a = static_cast<double>(cfg.max_amount);
    const double target = static_cast<double>(cfg.target_ratio);
    for (int k = 0; k < bins; ++k) {
      const std::size_t b0 = edge[k], b1 = edge[k + 1];
      centers[k] = smp[(b0 + b1) / 2].e;
      if (!(class_ratio(b0, b1, 0.0) > static_cast<double>(cfg.min_excess))) continue;
      double a = max_a;
      if (class_ratio(b0, b1, a) < target) {
        double lo = 0.0, hi = a;  // the ratio decreases with the amount
        for (int it = 0; it < 24; ++it) {
          const double mid = 0.5 * (lo + hi);
          if (class_ratio(b0, b1, mid) > target) lo = mid;
          else hi = mid;
        }
        a = 0.5 * (lo + hi);
      }
      amt[k] = a;
    }
    // The class amounts are used as they are: the classes hold many pixels, and
    // the interpolation over the pixel brightness below already smooths the
    // transition between them.
    const std::vector<double> &sm = amt;
    // Amount as a function of a pixel's own blurred excess: 0 at/below half the
    // object threshold, rising to the first class amount at the threshold,
    // linear between class centres, constant above the last centre.
    auto amount_at = [&](double e) {
      if (e <= 0.5 * lower) return 0.0;
      if (e < lower) return sm[0] * (e - 0.5 * lower) / (0.5 * lower);
      if (e <= centers.front() || bins == 1) return sm.front();
      if (e >= centers.back()) return sm.back();
      const auto it = std::upper_bound(centers.begin(), centers.end(), e);
      const std::size_t j = static_cast<std::size_t>(it - centers.begin());
      const double t = (e - centers[j - 1]) / std::max(centers[j] - centers[j - 1], 1e-12);
      return sm[j - 1] + t * (sm[j] - sm[j - 1]);
    };
    // Overall ratios (median over all samples) before / after with per-pixel amounts.
    {
      std::vector<double> g0, g1, m;
      for (const auto &q : smp) {
        const double ref = std::max(q.avg, 0.0);
        g0.push_back(q.xg);
        g1.push_back(q.xg - amount_at(q.e) * std::max(0.0, q.xg - ref));
        m.push_back(q.avg);
      }
      const double mm = median_of(m);
      res.ratio_before = mm > 0.0 ? median_of(g0) / mm : 0.0;
      res.ratio_after = mm > 0.0 ? median_of(g1) / mm : 0.0;
    }
    res.amounts.assign(sm.begin(), sm.end());
    double asum = 0.0, amax = 0.0;
    for (const auto &q : smp) {
      const double v = amount_at(q.e);
      asum += v;
      amax = std::max(amax, v);
    }
    res.amount = static_cast<float>(asum / static_cast<double>(smp.size()));
    if (!(amax > 1e-6)) {
      res.status = "not_needed";
      res.amount = 0.0f;
      res.ratio_after = res.ratio_before;
      return res;
    }
    // Apply to every valid pixel with the amount of its own brightness.
    for (int y = 0; y < rows; ++y) {
      for (int x = 0; x < cols; ++x) {
        const std::size_t i = static_cast<std::size_t>(y) * cols + x;
        if (!valid[i]) continue;
        const double a = amount_at(static_cast<double>(lb[i]) - sky_l);
        if (a <= 0.0) continue;
        const double xg = G(y, x) - res.sky_g;
        const double ref = std::max(0.0, 0.5 * ((R(y, x) - res.sky_r) + (B(y, x) - res.sky_b)));
        const double ex = std::max(0.0, xg - ref);
        if (ex > 0.0) G(y, x) = static_cast<float>(G(y, x) - a * ex);
      }
    }
  }
  res.applied = true;
  res.status = "applied";
  return res;
}

}  // namespace tile_compile::image
