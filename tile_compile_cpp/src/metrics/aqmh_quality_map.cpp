#include "tile_compile/metrics/aqmh_quality_map.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <numeric>
#include <thread>

namespace tile_compile::metrics {
namespace {

bool finite(float v) { return std::isfinite(v); }

float nan_value() { return std::numeric_limits<float>::quiet_NaN(); }

bool mask_valid(const std::vector<uint8_t> &mask, int w, int h, int x, int y) {
  if (x < 0 || y < 0 || x >= w || y >= h)
    return false;
  if (mask.empty())
    return true;
  if (w <= 0 || h <= 0 ||
      mask.size() != static_cast<size_t>(w) * static_cast<size_t>(h))
    return false;
  return mask[static_cast<size_t>(y) * static_cast<size_t>(w) +
              static_cast<size_t>(x)] != 0;
}

std::vector<float> finite_values(const Matrix2Df &m) {
  std::vector<float> values;
  values.reserve(static_cast<size_t>(m.size()));
  for (int y = 0; y < m.rows(); ++y) {
    for (int x = 0; x < m.cols(); ++x) {
      const float v = m(y, x);
      if (finite(v))
        values.push_back(v);
    }
  }
  return values;
}

float median_of(std::vector<float> values) {
  if (values.empty())
    return nan_value();
  const size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + mid, values.end());
  float med = values[mid];
  if (values.size() % 2 == 0) {
    std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
    med = 0.5f * (med + values[mid - 1]);
  }
  return med;
}

float median_buf(WindowBuf &buf) {
  if (buf.empty())
    return nan_value();
  const int n = buf.size();
  const int mid = n / 2;
  std::nth_element(buf.begin(), buf.begin() + mid, buf.end());
  float med = buf.data[static_cast<size_t>(mid)];
  if (n % 2 == 0) {
    std::nth_element(buf.begin(), buf.begin() + mid - 1, buf.end());
    med = 0.5f * (med + buf.data[static_cast<size_t>(mid - 1)]);
  }
  return med;
}

float mad_buf(WindowBuf &buf, float center, WindowBuf &tmp) {
  if (buf.empty() || !finite(center))
    return nan_value();
  tmp.clear();
  for (int i = 0; i < buf.size(); ++i)
    tmp.push(std::abs(buf.data[static_cast<size_t>(i)] - center));
  return median_buf(tmp);
}

float finite_median(const Matrix2Df &m) { return median_of(finite_values(m)); }

float mad_of(const std::vector<float> &values, float center) {
  if (values.empty() || !finite(center))
    return nan_value();
  std::vector<float> dev;
  dev.reserve(values.size());
  for (float v : values)
    dev.push_back(std::abs(v - center));
  return median_of(std::move(dev));
}

// Stack-allocated window buffer — avoids heap allocation in hot path.
// Max window diameter = 2*R+1; R_max=4 => 9x9=81 values.
constexpr int kMaxWindowR = 8;
constexpr int kMaxWindowN = (2 * kMaxWindowR + 1) * (2 * kMaxWindowR + 1);

struct WindowBuf {
  std::array<float, kMaxWindowN> data{};
  int n = 0;
  void clear() { n = 0; }
  void push(float v) { data[static_cast<size_t>(n++)] = v; }
  bool empty() const { return n == 0; }
  int size() const { return n; }
  float *begin() { return data.data(); }
  float *end() { return data.data() + n; }
  const float *begin() const { return data.data(); }
  const float *end() const { return data.data() + n; }
};

void fill_window(const Matrix2Df &m, int cx, int cy, int r, WindowBuf &buf) {
  buf.clear();
  const int rows = static_cast<int>(m.rows());
  const int cols = static_cast<int>(m.cols());
  const int r_clamped = std::min(r, kMaxWindowR);
  for (int yy = std::max(0, cy - r_clamped); yy <= std::min(rows - 1, cy + r_clamped); ++yy) {
    for (int xx = std::max(0, cx - r_clamped); xx <= std::min(cols - 1, cx + r_clamped); ++xx) {
      const float v = m(yy, xx);
      if (finite(v))
        buf.push(v);
    }
  }
}

Matrix2Df canvas_masked_frame(const Matrix2Df &frame,
                              const std::vector<uint8_t> &mask, int mask_w,
                              int mask_h) {
  Matrix2Df out(frame.rows(), frame.cols());
  for (int y = 0; y < frame.rows(); ++y) {
    for (int x = 0; x < frame.cols(); ++x) {
      const float v = frame(y, x);
      out(y, x) = (mask_valid(mask, mask_w, mask_h, x, y) && finite(v)) ? v
                                                                        : nan_value();
    }
  }
  return out;
}

Matrix2Df downsample_valid_mean(const Matrix2Df &src, int factor) {
  if (factor <= 1)
    return src;
  const int rows = static_cast<int>(src.rows());
  const int cols = static_cast<int>(src.cols());
  const int out_h = std::max(1, (rows + factor - 1) / factor);
  const int out_w = std::max(1, (cols + factor - 1) / factor);
  Matrix2Df out(out_h, out_w);
  out.setConstant(nan_value());
  for (int oy = 0; oy < out_h; ++oy) {
    for (int ox = 0; ox < out_w; ++ox) {
      double sum = 0.0;
      int count = 0;
      for (int y = oy * factor; y < std::min(rows, (oy + 1) * factor);
           ++y) {
        for (int x = ox * factor; x < std::min(cols, (ox + 1) * factor);
             ++x) {
          const float v = src(y, x);
          if (finite(v)) {
            sum += v;
            ++count;
          }
        }
      }
      if (count > 0)
        out(oy, ox) = static_cast<float>(sum / count);
    }
  }
  return out;
}

Matrix2Df masked_laplacian(const Matrix2Df &img) {
  Matrix2Df out(img.rows(), img.cols());
  out.setConstant(nan_value());
  for (int y = 0; y < img.rows(); ++y) {
    for (int x = 0; x < img.cols(); ++x) {
      const float c = img(y, x);
      if (!finite(c))
        continue;
      double sum = 0.0;
      int count = 0;
      constexpr int dx[4] = {-1, 1, 0, 0};
      constexpr int dy[4] = {0, 0, -1, 1};
      for (int k = 0; k < 4; ++k) {
        const int xx = x + dx[k];
        const int yy = y + dy[k];
        if (xx < 0 || yy < 0 || xx >= img.cols() || yy >= img.rows())
          continue;
        const float v = img(yy, xx);
        if (finite(v)) {
          sum += v;
          ++count;
        }
      }
      if (count >= 2)
        out(y, x) = std::abs(static_cast<float>(sum - count * c));
    }
  }
  return out;
}

Matrix2Df local_variance(const Matrix2Df &m, int r) {
  Matrix2Df out(m.rows(), m.cols());
  out.setConstant(nan_value());
  WindowBuf buf;
  for (int y = 0; y < m.rows(); ++y) {
    for (int x = 0; x < m.cols(); ++x) {
      fill_window(m, x, y, r, buf);
      if (buf.empty())
        continue;
      if (buf.size() < 3) {
        out(y, x) = 0.0f;
        continue;
      }
      double mean = 0.0;
      for (int i = 0; i < buf.size(); ++i) mean += buf.data[static_cast<size_t>(i)];
      mean /= buf.size();
      double var = 0.0;
      for (int i = 0; i < buf.size(); ++i) {
        const double d = buf.data[static_cast<size_t>(i)] - mean;
        var += d * d;
      }
      out(y, x) = static_cast<float>(var / buf.size());
    }
  }
  return out;
}

Matrix2Df phi_snr(const Matrix2Df &img, int r, bool &scene_dependent) {
  Matrix2Df out(img.rows(), img.cols());
  out.setConstant(nan_value());
  WindowBuf buf, tmp;
  for (int y = 0; y < img.rows(); ++y) {
    for (int x = 0; x < img.cols(); ++x) {
      fill_window(img, x, y, r, buf);
      if (buf.empty())
        continue;
      float signal = 0.0f;
      float sigma = eps_aqmh;
      if (buf.size() >= 3) {
        const float bg = median_buf(buf);
        double sum = 0.0;
        for (int i = 0; i < buf.size(); ++i)
          sum += std::max(buf.data[static_cast<size_t>(i)] - bg, 0.0f);
        signal = static_cast<float>(sum / buf.size());
        sigma = std::max(1.4826f * mad_buf(buf, bg, tmp), eps_aqmh);
      } else {
        scene_dependent = true;
        double sum = 0.0;
        for (int i = 0; i < buf.size(); ++i)
          sum += std::max(buf.data[static_cast<size_t>(i)], 0.0f);
        signal = static_cast<float>(sum / buf.size());
      }
      out(y, x) = signal / sigma;
    }
  }
  return out;
}

// Separable box filter: O(W*H*R) instead of O(W*H*R²)
Matrix2Df local_mean(const Matrix2Df &m, int r) {
  const int rows = static_cast<int>(m.rows());
  const int cols = static_cast<int>(m.cols());
  // Horizontal pass
  Matrix2Df hsum(rows, cols);
  Matrix2Df hcnt(rows, cols);
  hsum.setConstant(0.0f);
  hcnt.setConstant(0.0f);
  for (int y = 0; y < rows; ++y) {
    double s = 0.0;
    int c = 0;
    for (int x = 0; x <= std::min(r, cols - 1); ++x) {
      const float v = m(y, x);
      if (finite(v)) { s += v; ++c; }
    }
    for (int x = 0; x < cols; ++x) {
      hsum(y, x) = static_cast<float>(s);
      hcnt(y, x) = static_cast<float>(c);
      const int xadd = x + r + 1;
      if (xadd < cols) { const float v = m(y, xadd); if (finite(v)) { s += v; ++c; } }
      const int xrem = x - r;
      if (xrem >= 0) { const float v = m(y, xrem); if (finite(v)) { s -= v; --c; } }
    }
  }
  // Vertical pass
  Matrix2Df out(rows, cols);
  out.setConstant(nan_value());
  for (int x = 0; x < cols; ++x) {
    double s = 0.0, c = 0.0;
    for (int y = 0; y <= std::min(r, rows - 1); ++y) { s += hsum(y, x); c += hcnt(y, x); }
    for (int y = 0; y < rows; ++y) {
      if (c > 0.0) out(y, x) = static_cast<float>(s / c);
      const int yadd = y + r + 1;
      if (yadd < rows) { s += hsum(yadd, x); c += hcnt(yadd, x); }
      const int yrem = y - r;
      if (yrem >= 0) { s -= hsum(yrem, x); c -= hcnt(yrem, x); }
    }
  }
  return out;
}

Matrix2Df phi_artifact(const Matrix2Df &img, int r, float k_artifact,
                       float frac_artifact_max) {
  const Matrix2Df blur = local_mean(img, r);
  Matrix2Df hp(img.rows(), img.cols());
  hp.setConstant(nan_value());
  for (int y = 0; y < img.rows(); ++y) {
    for (int x = 0; x < img.cols(); ++x) {
      if (finite(img(y, x)) && finite(blur(y, x)))
        hp(y, x) = img(y, x) - blur(y, x);
    }
  }

  Matrix2Df out(img.rows(), img.cols());
  out.setConstant(nan_value());
  WindowBuf buf, tmp;
  for (int y = 0; y < img.rows(); ++y) {
    for (int x = 0; x < img.cols(); ++x) {
      fill_window(hp, x, y, r, buf);
      if (buf.empty())
        continue;
      const float center = median_buf(buf);
      const float tau = std::max(1.4826f * mad_buf(buf, center, tmp), eps_aqmh);
      int outliers = 0;
      for (int i = 0; i < buf.size(); ++i) {
        if (std::abs(buf.data[static_cast<size_t>(i)]) > k_artifact * tau)
          ++outliers;
      }
      const float frac = static_cast<float>(outliers) / buf.size();
      // Mindest-Quality von 0.01 beibehalten, damit glatte Regionen nicht auf 0 fallen
      constexpr float min_quality = 0.01f;
      out(y, x) = std::clamp(
          min_quality + (1.0f - min_quality) * (1.0f - frac / std::max(frac_artifact_max, eps_aqmh)),
          min_quality, 1.0f);
    }
  }
  return out;
}

Matrix2Df robust_zscore(const Matrix2Df &m) {
  Matrix2Df out(m.rows(), m.cols());
  out.setConstant(nan_value());
  const auto values = finite_values(m);
  const float med = median_of(values);
  const float scale = std::max(1.4826f * mad_of(values, med), eps_aqmh);
  if (!finite(med))
    return out;
  for (int y = 0; y < m.rows(); ++y) {
    for (int x = 0; x < m.cols(); ++x) {
      const float v = m(y, x);
      if (finite(v))
        out(y, x) = (v - med) / scale;
    }
  }
  return out;
}

Matrix2Df mask_aware_bilinear_upsample(const Matrix2Df &src, int out_w,
                                       int out_h, int factor) {
  Matrix2Df out(out_h, out_w);
  out.setConstant(nan_value());
  const int rows = static_cast<int>(src.rows());
  const int cols = static_cast<int>(src.cols());
  if (factor <= 1 && src.rows() == out_h && src.cols() == out_w)
    return src;
  for (int y = 0; y < out_h; ++y) {
    for (int x = 0; x < out_w; ++x) {
      const float sx = (static_cast<float>(x) + 0.5f) / factor - 0.5f;
      const float sy = (static_cast<float>(y) + 0.5f) / factor - 0.5f;
      const int x0 = static_cast<int>(std::floor(sx));
      const int y0 = static_cast<int>(std::floor(sy));
      double num = 0.0;
      double den = 0.0;
      for (int j = 0; j <= 1; ++j) {
        for (int i = 0; i <= 1; ++i) {
          const int xx = std::clamp(x0 + i, 0, cols - 1);
          const int yy = std::clamp(y0 + j, 0, rows - 1);
          const float wx = 1.0f - std::abs(sx - static_cast<float>(x0 + i));
          const float wy = 1.0f - std::abs(sy - static_cast<float>(y0 + j));
          const float w = std::max(wx, 0.0f) * std::max(wy, 0.0f);
          const float v = src(yy, xx);
          if (w > 0.0f && finite(v)) {
            num += w * v;
            den += w;
          }
        }
      }
      if (den > 0.0)
        out(y, x) = static_cast<float>(num / den);
    }
  }
  return out;
}

Matrix2Df compute_psi(const Matrix2Df &sharp, const Matrix2Df &snr,
                      const Matrix2Df &artifact,
                      const config::AqmhPyramidConfig &cfg) {
  const Matrix2Df z_sharp = robust_zscore(sharp);
  const Matrix2Df z_snr = robust_zscore(snr);
  Matrix2Df out(sharp.rows(), sharp.cols());
  out.setConstant(nan_value());
  for (int y = 0; y < out.rows(); ++y) {
    for (int x = 0; x < out.cols(); ++x) {
      if (!finite(z_sharp(y, x)) || !finite(z_snr(y, x)) ||
          !finite(artifact(y, x)))
        continue;
      const float score = cfg.w_sharp * z_sharp(y, x) + cfg.w_snr * z_snr(y, x);
      const float sigmoid = 1.0f / (1.0f + std::exp(-score));
      out(y, x) = std::clamp(sigmoid * artifact(y, x), 0.0f, 1.0f);
    }
  }
  return out;
}

} // namespace

AqmhQualityMapResult compute_aqmh_quality_map(
    const Matrix2Df &frame, const std::vector<uint8_t> &canvas_mask,
    int canvas_mask_width, int canvas_mask_height,
    const config::AqmhPyramidConfig &cfg) {
  AqmhQualityMapResult result;
  result.q_map = Matrix2Df::Zero(frame.rows(), frame.cols());
  if (frame.rows() <= 0 || frame.cols() <= 0)
    return result;

  const Matrix2Df masked =
      canvas_masked_frame(frame, canvas_mask, canvas_mask_width,
                          canvas_mask_height);
  const int min_dim = std::min(frame.rows(), frame.cols());
  std::vector<Matrix2Df> psi_full_res;
  psi_full_res.reserve(static_cast<size_t>(std::max(cfg.scales, 0)));

  for (int s = 0; s < cfg.scales; ++s) {
    const int factor = 1 << (2 * s);
    if (factor > std::max(1, min_dim / 16)) {
      result.diagnostics.omitted_scales.push_back(s);
      continue;
    }

    const Matrix2Df img_s = downsample_valid_mean(masked, factor);
    const int radius = std::max(1, cfg.base_window_px);
    const Matrix2Df sharp = local_variance(masked_laplacian(img_s), radius);
    bool scene_dependent = false;
    const Matrix2Df snr = phi_snr(img_s, radius, scene_dependent);
    const Matrix2Df artifact =
        phi_artifact(img_s, radius, cfg.k_artifact, cfg.frac_artifact_max);
    result.diagnostics.scene_dependent_snr =
        result.diagnostics.scene_dependent_snr || scene_dependent;

    if (s == 0)
      result.diagnostics.sharpness_p50 = finite_median(sharp);
    if (s == 1)
      result.diagnostics.snr_p50 = finite_median(snr);

    psi_full_res.push_back(
        mask_aware_bilinear_upsample(compute_psi(sharp, snr, artifact, cfg),
                                     frame.cols(), frame.rows(), factor));
  }

  if (psi_full_res.empty())
    return result;

  for (int y = 0; y < frame.rows(); ++y) {
    for (int x = 0; x < frame.cols(); ++x) {
      if (!mask_valid(canvas_mask, canvas_mask_width, canvas_mask_height, x,
                      y)) {
        result.q_map(y, x) = 0.0f;
        continue;
      }
      double log_sum = 0.0;
      bool veto = false;
      for (const Matrix2Df &psi : psi_full_res) {
        const float v = psi(y, x);
        if (!finite(v) || v <= 0.0f) {
          veto = true;
          break;
        }
        log_sum += std::log(static_cast<double>(std::clamp(v, eps_aqmh, 1.0f)));
      }
      result.q_map(y, x) =
          veto ? 0.0f
               : static_cast<float>(std::exp(log_sum / psi_full_res.size()));
    }
  }

  return result;
}

} // namespace tile_compile::metrics
