#include "tile_compile/reconstruction/atrous_decomposition.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tile_compile::reconstruction {
namespace {

constexpr double kW[5] = {1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0,
                          1.0 / 16.0};
float nanf_() { return std::numeric_limits<float>::quiet_NaN(); }

// One separable pass with the dilated B3 kernel over TWO channels that share
// the same geometry (redundant-reload analysis N8): bounds checks and index
// math run once for both accumulators instead of twice. `src` is a full
// field; taps that fall outside [0, len) simply do not contribute.
// `horizontal` selects the axis; `dilation` is the a-trous hole spacing
// (1 << (level-1)).
void conv_axis_pair(const std::vector<double> &src_a,
                    const std::vector<double> &src_b, int width, int height,
                    int dilation, bool horizontal,
                    std::vector<double> &out_a, std::vector<double> &out_b) {
  out_a.assign(src_a.size(), 0.0);
  out_b.assign(src_b.size(), 0.0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      double acc_a = 0.0, acc_b = 0.0;
      for (int t = -2; t <= 2; ++t) {
        const int xx = horizontal ? x + t * dilation : x;
        const int yy = horizontal ? y : y + t * dilation;
        if (xx < 0 || xx >= width || yy < 0 || yy >= height) continue;
        const std::size_t j = static_cast<std::size_t>(yy) * width + xx;
        acc_a += kW[t + 2] * src_a[j];
        acc_b += kW[t + 2] * src_b[j];
      }
      const std::size_t i = static_cast<std::size_t>(y) * width + x;
      out_a[i] = acc_a;
      out_b[i] = acc_b;
    }
  }
}

// 2D masked convolution as two separable passes of the SAME kernel (linear,
// so separable application == the 2D convolution the plan writes). Returns
// den (convolved mask) and num (convolved value*mask). hx_* are caller-owned
// scratch (N8: reused across decomposition levels instead of per-level
// allocations).
void masked_convolve(const std::vector<double> &value_times_mask,
                     const std::vector<double> &mask, int width, int height,
                     int dilation, std::vector<double> &hx_num,
                     std::vector<double> &hx_den, std::vector<double> &num,
                     std::vector<double> &den) {
  conv_axis_pair(value_times_mask, mask, width, height, dilation, true,
                 hx_num, hx_den);
  conv_axis_pair(hx_num, hx_den, width, height, dilation, false, num, den);
}

}  // namespace

AtrousDecomposition atrous_decompose(const std::vector<float> &value,
                                     const std::vector<uint8_t> &mask,
                                     int width, int height, int levels) {
  if (width <= 0 || height <= 0)
    throw std::invalid_argument("ATROUS_BAD_DIMENSIONS");
  const std::size_t n = static_cast<std::size_t>(width) * height;
  if (value.size() != n)
    throw std::invalid_argument("ATROUS_VALUE_SIZE_MISMATCH");
  if (!mask.empty() && mask.size() != n)
    throw std::invalid_argument("ATROUS_MASK_SIZE_MISMATCH");
  if (levels < 1 || levels > 4)
    throw std::invalid_argument("ATROUS_LEVELS_OUT_OF_RANGE");

  // C_0, M_0.
  std::vector<double> c_prev(n, 0.0);
  std::vector<uint8_t> m_prev(n, 0u);
  for (std::size_t i = 0; i < n; ++i) {
    const bool valid = (mask.empty() || mask[i] != 0u) &&
                       std::isfinite(value[i]);
    m_prev[i] = valid ? 1u : 0u;
    c_prev[i] = valid ? static_cast<double>(value[i]) : 0.0;
  }

  AtrousDecomposition d;
  d.width = width;
  d.height = height;
  d.levels = levels;
  d.bands.resize(static_cast<std::size_t>(levels));

  // N8: per-level scratch is hoisted; every level rewrites each buffer fully
  // (masked_convolve assigns; c_cur is only read at entries m_cur marked this
  // level).
  std::vector<double> vm(n), md(n), hx_num, hx_den, num, den, c_cur(n);
  std::vector<uint8_t> m_cur(n);
  for (int j = 1; j <= levels; ++j) {
    const int dilation = 1 << (j - 1);
    for (std::size_t i = 0; i < n; ++i) {
      md[i] = m_prev[i] ? 1.0 : 0.0;
      vm[i] = m_prev[i] ? c_prev[i] : 0.0;
    }
    masked_convolve(vm, md, width, height, dilation, hx_num, hx_den, num,
                    den);
    auto &band = d.bands[static_cast<std::size_t>(j - 1)];
    band.level = j;
    band.detail.assign(n, nanf_());
    band.support.assign(n, 0u);

    for (std::size_t i = 0; i < n; ++i) {
      const bool den_ok = den[i] >= kAtrousDenMinFraction;
      m_cur[i] = (m_prev[i] != 0u && den_ok) ? 1u : 0u;
      if (den_ok) c_cur[i] = num[i] / den[i];
      // D_j valid only on M_(j-1) && M_j.
      if (m_prev[i] != 0u && m_cur[i] != 0u) {
        band.support[i] = 1u;
        band.detail[i] = static_cast<float>(c_prev[i] - c_cur[i]);
      }
    }
    c_prev.swap(c_cur);
    m_prev.swap(m_cur);
  }

  d.coarse.assign(n, nanf_());
  d.coarse_support.assign(n, 0u);
  for (std::size_t i = 0; i < n; ++i)
    if (m_prev[i] != 0u) {
      d.coarse_support[i] = 1u;
      d.coarse[i] = static_cast<float>(c_prev[i]);
    }
  return d;
}

double atrous_reconstruction_max_error(const AtrousDecomposition &d,
                                       const std::vector<float> &original) {
  const std::size_t n =
      static_cast<std::size_t>(d.width) * static_cast<std::size_t>(d.height);
  if (original.size() != n) throw std::invalid_argument("ATROUS_ORIG_SIZE");
  double max_err = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!d.coarse_support[i]) continue;  // tightest common valid support
    double sum = d.coarse[i];
    bool ok = std::isfinite(sum);
    for (const auto &b : d.bands) {
      if (!b.support[i] || !std::isfinite(b.detail[i])) { ok = false; break; }
      sum += b.detail[i];
    }
    if (!ok || !std::isfinite(original[i])) continue;
    max_err = std::max(max_err, std::abs(static_cast<double>(original[i]) - sum));
  }
  return max_err;
}

}  // namespace tile_compile::reconstruction
