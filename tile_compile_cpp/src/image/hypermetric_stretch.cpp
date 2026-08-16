#include "tile_compile/image/hypermetric_stretch.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace tile_compile::image {
namespace {

struct ProfileDef {
  const char *name;
  std::array<float, 3> weights;
};

std::string normalize_key(std::string s) {
  std::string out;
  out.reserve(s.size());
  for (unsigned char c : s) {
    if (std::isalnum(c)) {
      out.push_back(static_cast<char>(std::tolower(c)));
    }
  }
  return out;
}

const std::vector<ProfileDef> &profiles() {
  static const std::vector<ProfileDef> kProfiles = {
      {"rec709", {0.2126f, 0.7152f, 0.0722f}},
      {"Rec.709 (Recommended)", {0.2126f, 0.7152f, 0.0722f}},
      {"Sony IMX571 (ASI2600/QHY268)", {0.2944f, 0.5021f, 0.2035f}},
      {"Sony IMX455 (ASI6200/QHY600)", {0.2987f, 0.5001f, 0.2013f}},
      {"Sony IMX410 (ASI2400)", {0.3015f, 0.5050f, 0.1935f}},
      {"Sony IMX269 (Altair/ToupTek)", {0.3040f, 0.5010f, 0.1950f}},
      {"Sony IMX294 (ASI294)", {0.3068f, 0.5008f, 0.1925f}},
      {"Sony IMX533 (ASI533)", {0.2910f, 0.5072f, 0.2018f}},
      {"Sony IMX676 (ASI676)", {0.2880f, 0.5100f, 0.2020f}},
      {"Sony IMX585 (ASI585) - STARVIS 2", {0.3431f, 0.4822f, 0.1747f}},
      {"Sony IMX662 (ASI662) - STARVIS 2", {0.3430f, 0.4821f, 0.1749f}},
      {"Sony IMX678 (ASI678) - STARVIS 2", {0.3426f, 0.4825f, 0.1750f}},
      {"Sony IMX415 (DWARF II)", {0.2703f, 0.5405f, 0.1892f}},
      {"Sony IMX462 (ASI462)", {0.3333f, 0.4866f, 0.1801f}},
      {"Sony IMX715 (ASI715)", {0.3410f, 0.4840f, 0.1750f}},
      {"Sony IMX482 (ASI482)", {0.3150f, 0.4950f, 0.1900f}},
      {"Sony IMX183 (ASI183)", {0.2967f, 0.4983f, 0.2050f}},
      {"Sony IMX178 (ASI178)", {0.2346f, 0.5206f, 0.2448f}},
      {"Sony IMX224 (ASI224)", {0.3402f, 0.4765f, 0.1833f}},
      {"Canon EOS (Modern - 60D/600D/500D)", {0.2600f, 0.5200f, 0.2200f}},
      {"Canon EOS (Legacy - 300D/40D/20D)", {0.2450f, 0.5350f, 0.2200f}},
      {"Nikon DSLR (Modern - D5100/D7200)", {0.2650f, 0.5100f, 0.2250f}},
      {"Nikon DSLR (Legacy - D3/D300/D90)", {0.2500f, 0.5300f, 0.2200f}},
      {"Fujifilm X-Trans 5 HR", {0.2800f, 0.5100f, 0.2100f}},
      {"Panasonic MN34230 (ASI1600)", {0.2650f, 0.5250f, 0.2100f}},
      {"ZWO Seestar S50", {0.3333f, 0.4866f, 0.1801f}},
      {"ZWO Seestar S30", {0.2928f, 0.5053f, 0.2019f}},
      {"Narrowband HOO", {0.5000f, 0.2500f, 0.2500f}},
      {"Narrowband SHO", {0.3333f, 0.3400f, 0.3267f}},
  };
  return kProfiles;
}

bool mask_valid(const std::vector<uint8_t> *mask, int rows, int cols, int y,
                int x) {
  if (mask == nullptr || rows <= 0 || cols <= 0) {
    return true;
  }
  if (y < 0 || x < 0 || y >= rows || x >= cols) {
    return false;
  }
  const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(cols) +
                     static_cast<size_t>(x);
  return idx < mask->size() && (*mask)[idx] != 0;
}

float sanitize01(float v) {
  if (!std::isfinite(v)) {
    return 0.0f;
  }
  return std::clamp(v, 0.0f, 1.0f);
}

void normalize_rgb_input_inplace(
    Matrix2Df &R, Matrix2Df &G, Matrix2Df &B,
    const std::vector<uint8_t> *statistics_mask,
    const std::vector<uint8_t> *output_mask, int mask_rows, int mask_cols) {
  const int rows = static_cast<int>(R.rows());
  const int cols = static_cast<int>(R.cols());
  float max_v = 0.0f;
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      if (!mask_valid(statistics_mask, mask_rows, mask_cols, y, x)) {
        continue;
      }
      const float vals[3] = {R(y, x), G(y, x), B(y, x)};
      for (float v : vals) {
        if (std::isfinite(v)) {
          max_v = std::max(max_v, v);
        }
      }
    }
  }

  float scale = 1.0f;
  if (max_v > 1.1f && max_v < 100000.0f) {
    scale = 65535.0f;
  } else if (max_v >= 100000.0f) {
    scale = 4294967295.0f;
  }

  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      if (!mask_valid(output_mask, mask_rows, mask_cols, y, x)) {
        R(y, x) = G(y, x) = B(y, x) = 0.0f;
        continue;
      }
      R(y, x) = sanitize01(R(y, x) / scale);
      G(y, x) = sanitize01(G(y, x) / scale);
      B(y, x) = sanitize01(B(y, x) / scale);
    }
  }
}

float percentile(std::vector<float> values, float pct) {
  values.erase(std::remove_if(values.begin(), values.end(),
                              [](float v) { return !std::isfinite(v); }),
               values.end());
  if (values.empty()) {
    return 0.0f;
  }
  pct = std::clamp(pct, 0.0f, 100.0f);
  const double pos = (static_cast<double>(values.size()) - 1.0) *
                     static_cast<double>(pct) / 100.0;
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = static_cast<size_t>(std::ceil(pos));
  std::nth_element(values.begin(), values.begin() + lo, values.end());
  const float vlo = values[lo];
  if (hi == lo) {
    return vlo;
  }
  std::nth_element(values.begin(), values.begin() + hi, values.end());
  const float vhi = values[hi];
  const float t = static_cast<float>(pos - static_cast<double>(lo));
  return vlo * (1.0f - t) + vhi * t;
}

float median(std::vector<float> values) { return percentile(std::move(values), 50.0f); }

float mean(const std::vector<float> &values) {
  if (values.empty()) {
    return 0.0f;
  }
  const double sum =
      std::accumulate(values.begin(), values.end(), 0.0);
  return static_cast<float>(sum / static_cast<double>(values.size()));
}

float stddev(const std::vector<float> &values, float m) {
  if (values.empty()) {
    return 0.0f;
  }
  double acc = 0.0;
  for (float v : values) {
    const double d = static_cast<double>(v) - static_cast<double>(m);
    acc += d * d;
  }
  return static_cast<float>(std::sqrt(acc / static_cast<double>(values.size())));
}

std::vector<float> sample_luminance(const Matrix2Df &R, const Matrix2Df &G,
                                    const Matrix2Df &B,
                                    const std::array<float, 3> &w,
                                    const std::vector<uint8_t> *mask,
                                    int mask_rows, int mask_cols,
                                    size_t max_samples) {
  const int rows = static_cast<int>(R.rows());
  const int cols = static_cast<int>(R.cols());
  const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t stride = std::max<size_t>(1, total / std::max<size_t>(1, max_samples));
  std::vector<float> out;
  out.reserve(std::min(total, max_samples));
  size_t linear = 0;
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x, ++linear) {
      if (linear % stride != 0 ||
          !mask_valid(mask, mask_rows, mask_cols, y, x)) {
        continue;
      }
      const float l = w[0] * sanitize01(R(y, x)) +
                      w[1] * sanitize01(G(y, x)) +
                      w[2] * sanitize01(B(y, x));
      out.push_back(l);
    }
  }
  return out;
}

struct RgbSample {
  std::vector<float> r;
  std::vector<float> g;
  std::vector<float> b;
};

RgbSample sample_rgb_pixels(const Matrix2Df &R, const Matrix2Df &G,
                            const Matrix2Df &B,
                            const std::vector<uint8_t> *mask, int mask_rows,
                            int mask_cols, size_t target_pixels) {
  const int rows = static_cast<int>(R.rows());
  const int cols = static_cast<int>(R.cols());
  const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t stride =
      std::max<size_t>(1, total / std::max<size_t>(1, target_pixels));
  RgbSample sample;
  sample.r.reserve(std::min(total, target_pixels));
  sample.g.reserve(std::min(total, target_pixels));
  sample.b.reserve(std::min(total, target_pixels));
  size_t linear = 0;
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x, ++linear) {
      if (linear % stride != 0 ||
          !mask_valid(mask, mask_rows, mask_cols, y, x)) {
        continue;
      }
      sample.r.push_back(sanitize01(R(y, x)));
      sample.g.push_back(sanitize01(G(y, x)));
      sample.b.push_back(sanitize01(B(y, x)));
    }
  }
  return sample;
}

std::vector<float> luminance_from_sample(const RgbSample &sample,
                                         const std::array<float, 3> &w) {
  const size_t n = std::min({sample.r.size(), sample.g.size(), sample.b.size()});
  std::vector<float> out;
  out.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    out.push_back(w[0] * sample.r[i] + w[1] * sample.g[i] + w[2] * sample.b[i]);
  }
  return out;
}

float calculate_anchor_statistical_sample(const RgbSample &sample) {
  if (sample.r.empty() || sample.g.empty() || sample.b.empty()) {
    return 0.0f;
  }
  const float floor =
      std::min({percentile(sample.r, 0.5f), percentile(sample.g, 0.5f),
                percentile(sample.b, 0.5f)});
  return std::max(0.0f, floor - 0.00025f);
}

float calculate_anchor_adaptive_luminance(std::vector<float> sample) {
  if (sample.empty()) {
    return 0.0f;
  }
  constexpr int kBins = 65536;
  std::vector<float> hist(kBins, 0.0f);
  for (float v : sample) {
    const float c = sanitize01(v);
    int idx = static_cast<int>(c * static_cast<float>(kBins));
    idx = std::clamp(idx, 0, kBins - 1);
    hist[idx] += 1.0f;
  }

  std::vector<float> smooth(kBins, 0.0f);
  constexpr int kWidth = 50;
  // NumPy parity: np.convolve(hist, ones(50)/50, mode='same').
  // For an even kernel this corresponds to full-convolution indices
  // [24, 24 + len(hist)), i.e. histogram bins [i - 25, i + 24].
  for (int i = 0; i < kBins; ++i) {
    float sum = 0.0f;
    const int begin = std::max(0, i - 25);
    const int end = std::min(kBins - 1, i + 24);
    for (int j = begin; j <= end; ++j) {
      sum += hist[j];
    }
    smooth[i] = sum / static_cast<float>(kWidth);
  }

  int search_start = 100;
  if (*std::max_element(smooth.begin(), smooth.begin() + search_start) > 0.0f) {
    search_start = 0;
  }
  auto peak_it = std::max_element(smooth.begin() + search_start, smooth.end());
  const int peak_idx = static_cast<int>(std::distance(smooth.begin(), peak_it));
  const float target = *peak_it * 0.06f;
  int anchor_idx = -1;
  for (int i = 0; i < peak_idx; ++i) {
    if (smooth[i] < target) {
      anchor_idx = i;
    }
  }
  if (anchor_idx >= 0) {
    return static_cast<float>(anchor_idx) / static_cast<float>(kBins);
  }
  return percentile(std::move(sample), 0.5f);
}

float calculate_anchor_statistical(const Matrix2Df &R, const Matrix2Df &G,
                                   const Matrix2Df &B,
                                   const std::vector<uint8_t> *mask,
                                   int mask_rows, int mask_cols) {
  const int rows = static_cast<int>(R.rows());
  const int cols = static_cast<int>(R.cols());
  const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t stride = std::max<size_t>(1, (total * 3u) / 500000u);
  std::vector<float> sr;
  std::vector<float> sg;
  std::vector<float> sb;
  size_t linear = 0;
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x, ++linear) {
      if (linear % stride != 0 ||
          !mask_valid(mask, mask_rows, mask_cols, y, x)) {
        continue;
      }
      sr.push_back(sanitize01(R(y, x)));
      sg.push_back(sanitize01(G(y, x)));
      sb.push_back(sanitize01(B(y, x)));
    }
  }
  const float floor =
      std::min({percentile(sr, 0.5f), percentile(sg, 0.5f), percentile(sb, 0.5f)});
  return std::max(0.0f, floor - 0.00025f);
}

float calculate_anchor_adaptive(const Matrix2Df &R, const Matrix2Df &G,
                                const Matrix2Df &B,
                                const std::array<float, 3> &w,
                                const std::vector<uint8_t> *mask,
                                int mask_rows, int mask_cols) {
  std::vector<float> sample =
      sample_luminance(R, G, B, w, mask, mask_rows, mask_cols, 2000000u);
  if (sample.empty()) {
    return 0.0f;
  }
  return calculate_anchor_adaptive_luminance(std::move(sample));
}

float estimate_star_pressure(std::vector<float> luma) {
  luma.erase(std::remove_if(luma.begin(), luma.end(),
                            [](float v) { return !std::isfinite(v) || v <= 1e-7f; }),
             luma.end());
  if (luma.size() < 100) {
    return 0.0f;
  }
  const float p999 = percentile(luma, 99.9f);
  const float p9999 = percentile(luma, 99.99f);
  const size_t bright =
      static_cast<size_t>(std::count_if(luma.begin(), luma.end(),
                                        [p999](float v) { return v > p999; }));
  const float bright_frac =
      static_cast<float>(bright) / static_cast<float>(luma.size());
  float p_term = std::clamp(p9999 / (p999 + 1e-9f), 1.0f, 5.0f);
  p_term = (p_term - 1.0f) / 4.0f;
  const float f_term = std::clamp(bright_frac * 200.0f, 0.0f, 1.0f);
  return std::clamp(0.7f * p_term + 0.3f * f_term, 0.0f, 1.0f);
}

float solve_auto_log_d(std::vector<float> valid, float target_bg, float b,
                       float star_pressure, bool ready_to_use) {
  if (valid.empty()) {
    return 2.0f;
  }
  float target = target_bg;
  float best = 2.0f;
  for (int i = 0; i < 15; ++i) {
    best = hypermetric_solve_log_d(valid, target, b);
    if (star_pressure > 0.6f) {
      target *= (1.0f - 0.15f * star_pressure);
    }
    if (!ready_to_use) {
      break;
    }
    const float D = std::pow(10.0f, best);
    std::vector<float> stretched;
    stretched.reserve(valid.size());
    for (float v : valid) {
      stretched.push_back(hypermetric_hyperbolic_stretch_value(v, D, b));
    }
    const float med = median(stretched);
    const float sd = stddev(stretched, med);
    const float min_v = *std::min_element(stretched.begin(), stretched.end());
    const float global_floor = std::max(min_v, med - 2.7f * sd);
    if (global_floor <= 0.001f) {
      break;
    }
    target -= 0.015f;
    if (target < 0.05f) {
      break;
    }
  }
  return best;
}

std::vector<float> build_auto_luma_sample(const Matrix2Df &R, const Matrix2Df &G,
                                          const Matrix2Df &B,
                                          const std::array<float, 3> &w,
                                          bool adaptive_anchor,
                                          const std::vector<uint8_t> *mask,
                                          int mask_rows, int mask_cols) {
  RgbSample sample = sample_rgb_pixels(R, G, B, mask, mask_rows, mask_cols, 100000u);
  const float anchor =
      adaptive_anchor
          ? calculate_anchor_adaptive_luminance(luminance_from_sample(sample, w))
          : calculate_anchor_statistical_sample(sample);
  const size_t n = std::min({sample.r.size(), sample.g.size(), sample.b.size()});
  std::vector<float> valid;
  valid.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    const float ra = std::max(sample.r[i] - anchor, 0.0f);
    const float ga = std::max(sample.g[i] - anchor, 0.0f);
    const float ba = std::max(sample.b[i] - anchor, 0.0f);
    const float l = w[0] * ra + w[1] * ga + w[2] * ba;
    if (l > 1e-7f && std::isfinite(l)) {
      valid.push_back(l);
    }
  }
  return valid;
}

float mtf_value(float data, float m) {
  const float denom = (2.0f * m - 1.0f) * data - m;
  if (std::abs(denom) < 1e-12f) {
    return 0.0f;
  }
  const float res = ((m - 1.0f) * data) / denom;
  if (!std::isfinite(res)) {
    return res > 0.0f ? 1.0f : 0.0f;
  }
  return std::clamp(res, 0.0f, 1.0f);
}

void adaptive_output_scaling(
    Matrix2Df &R, Matrix2Df &G, Matrix2Df &B,
    const std::array<float, 3> &w, float target_bg,
    const std::vector<uint8_t> *statistics_mask,
    const std::vector<uint8_t> *output_mask, int mask_rows, int mask_cols) {
  const int rows = static_cast<int>(R.rows());
  const int cols = static_cast<int>(R.cols());

  // Sample per-channel statistics for channel-specific floor computation.
  // Using a luma-based floor (with rec709 weights where G=71.5%) causes the
  // floor to exceed the B channel background, clipping B to near-zero and
  // producing a green tint. Use per-channel floors, but keep a shared robust
  // expansion span so the final output scaling does not independently
  // white-balance the channels.
  std::vector<float> sr;
  std::vector<float> sg;
  std::vector<float> sb;
  const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t stride = std::max<size_t>(1, total / 500000u);
  size_t linear = 0;
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x, ++linear) {
      if (linear % stride != 0 ||
          !mask_valid(statistics_mask, mask_rows, mask_cols, y, x)) {
        continue;
      }
      sr.push_back(sanitize01(R(y, x)));
      sg.push_back(sanitize01(G(y, x)));
      sb.push_back(sanitize01(B(y, x)));
    }
  }
  if (sr.empty()) {
    return;
  }

  auto channel_floor = [](std::vector<float> &vals) {
    const float med = median(vals);
    const float sd = stddev(vals, med);
    const float min_v = *std::min_element(vals.begin(), vals.end());
    return std::max(min_v, med - 2.7f * sd);
  };

  const float floor_r = channel_floor(sr);
  const float floor_g = channel_floor(sg);
  const float floor_b = channel_floor(sb);
  constexpr float pedestal = 0.001f;

  // Per-channel soft ceilings define candidate dynamic ranges. The expansion
  // uses a shared span, preventing low-dynamic-range channels from being
  // amplified with a different color scale than the others.
  const float ceil_r = percentile(sr, 99.0f);
  const float ceil_g = percentile(sg, 99.0f);
  const float ceil_b = percentile(sb, 99.0f);
  const float span_r = std::max(ceil_r - floor_r, 1e-6f);
  const float span_g = std::max(ceil_g - floor_g, 1e-6f);
  const float span_b = std::max(ceil_b - floor_b, 1e-6f);
  const float shared_span = std::max({span_r, span_g, span_b});

  // Channel floor + shared scale: map [floor_ch, floor_ch + shared_span] to
  // [pedestal, 0.98].
  auto make_expand = [&](float floor_ch) {
    const float scale = (0.98f - pedestal) / (shared_span + 1e-9f);
    return [floor_ch, scale, pedestal](float v) {
      return std::clamp((v - floor_ch) * scale + pedestal, 0.0f, 1.0f);
    };
  };

  auto expand_r = make_expand(floor_r);
  auto expand_g = make_expand(floor_g);
  auto expand_b = make_expand(floor_b);

  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      if (!mask_valid(output_mask, mask_rows, mask_cols, y, x)) {
        R(y, x) = G(y, x) = B(y, x) = 0.0f;
        continue;
      }
      R(y, x) = expand_r(R(y, x));
      G(y, x) = expand_g(G(y, x));
      B(y, x) = expand_b(B(y, x));
    }
  }

  const std::vector<float> scaled_luma = sample_luminance(
      R, G, B, w, statistics_mask, mask_rows, mask_cols, 500000u);
  const float current_bg = median(scaled_luma);
  if (current_bg > 0.0f && current_bg < 1.0f &&
      std::abs(current_bg - target_bg) > 1e-3f) {
    const float denom =
        current_bg * (2.0f * target_bg - 1.0f) - target_bg;
    if (std::abs(denom) > 1e-9f) {
      const float m = (current_bg * (target_bg - 1.0f)) / denom;
      for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
          if (!mask_valid(output_mask, mask_rows, mask_cols, y, x)) {
            continue;
          }
          R(y, x) = mtf_value(R(y, x), m);
          G(y, x) = mtf_value(G(y, x), m);
          B(y, x) = mtf_value(B(y, x), m);
        }
      }
    }
  }
}

void soft_clip(Matrix2Df &M, float threshold, float rolloff) {
  for (int y = 0; y < M.rows(); ++y) {
    for (int x = 0; x < M.cols(); ++x) {
      float v = M(y, x);
      if (v > threshold) {
        const float t =
            std::clamp((v - threshold) / (1.0f - threshold + 1e-9f), 0.0f, 1.0f);
        v = threshold + (1.0f - threshold) *
                            (1.0f - std::pow(1.0f - t, rolloff));
      }
      M(y, x) = std::clamp(v, 0.0f, 1.0f);
    }
  }
}

void apply_linear_expansion(Matrix2Df &L, float factor,
                            const std::vector<uint8_t> *mask, int mask_rows,
                            int mask_cols) {
  if (factor <= 0.001f) {
    return;
  }
  factor = std::clamp(factor, 0.0f, 1.0f);
  const int rows = static_cast<int>(L.rows());
  const int cols = static_cast<int>(L.cols());

  float abs_max = 0.0f;
  int max_y = 0;
  int max_x = 0;
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      if (!mask_valid(mask, mask_rows, mask_cols, y, x)) {
        continue;
      }
      const float v = sanitize01(L(y, x));
      if (v > abs_max) {
        abs_max = v;
        max_y = y;
        max_x = x;
      }
    }
  }

  bool use_absolute_max = false;
  if (abs_max > 0.001f) {
    float max_neighbor = -std::numeric_limits<float>::infinity();
    bool has_neighbor = false;
    for (int yy = std::max(0, max_y - 1); yy <= std::min(rows - 1, max_y + 1);
         ++yy) {
      for (int xx = std::max(0, max_x - 1); xx <= std::min(cols - 1, max_x + 1);
           ++xx) {
        if ((yy == max_y && xx == max_x) ||
            !mask_valid(mask, mask_rows, mask_cols, yy, xx)) {
          continue;
        }
        const float v = sanitize01(L(yy, xx));
        if (v < abs_max) {
          max_neighbor = std::max(max_neighbor, v);
          has_neighbor = true;
        }
      }
    }
    if (has_neighbor && max_neighbor >= abs_max * 0.20f) {
      use_absolute_max = true;
    }
  }

  const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t stride = std::max<size_t>(1, total / 500000u);
  std::vector<float> sample;
  sample.reserve(std::min<size_t>(total, 500000u));
  size_t linear = 0;
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x, ++linear) {
      if (linear % stride != 0 ||
          !mask_valid(mask, mask_rows, mask_cols, y, x)) {
        continue;
      }
      sample.push_back(sanitize01(L(y, x)));
    }
  }
  if (sample.empty()) {
    return;
  }
  const float low = percentile(sample, 0.001f);
  const float high =
      use_absolute_max ? abs_max : percentile(std::move(sample), 99.999f);
  if (high <= low) {
    return;
  }
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      if (!mask_valid(mask, mask_rows, mask_cols, y, x)) {
        L(y, x) = 0.0f;
        continue;
      }
      const float normalized = std::clamp((L(y, x) - low) / (high - low),
                                          0.0f, 1.0f);
      L(y, x) = L(y, x) * (1.0f - factor) + normalized * factor;
    }
  }
}

} // namespace

std::array<float, 3> hypermetric_profile_weights(
    const std::string &profile, const std::string &fallback_profile,
    std::string *resolved_profile, std::string *profile_source) {
  const std::string key = normalize_key(profile == "auto" ? fallback_profile : profile);
  for (const auto &p : profiles()) {
    if (normalize_key(p.name) == key) {
      if (resolved_profile != nullptr) {
        *resolved_profile = p.name;
      }
      if (profile_source != nullptr) {
        *profile_source = profile == "auto" ? "fallback" : "configured";
      }
      return p.weights;
    }
  }
  if (resolved_profile != nullptr) {
    *resolved_profile = "rec709";
  }
  if (profile_source != nullptr) {
    *profile_source = "fallback";
  }
  return {0.2126f, 0.7152f, 0.0722f};
}

float hypermetric_hyperbolic_stretch_value(float value, float D, float b,
                                           float SP) {
  D = std::max(D, 0.1f);
  b = std::max(b, 0.1f);
  const float term1 = std::asinh(D * (value - SP) + b);
  const float term2 = std::asinh(b);
  float norm = std::asinh(D * (1.0f - SP) + b) - term2;
  if (std::abs(norm) < 1e-12f) {
    norm = 1e-6f;
  }
  return (term1 - term2) / norm;
}

float hypermetric_solve_log_d(std::vector<float> luma_sample,
                              float target_median, float b) {
  luma_sample.erase(
      std::remove_if(luma_sample.begin(), luma_sample.end(),
                     [](float v) { return !std::isfinite(v) || v <= 0.0f; }),
      luma_sample.end());
  if (luma_sample.empty()) {
    return 2.0f;
  }
  const float median_in = median(std::move(luma_sample));
  if (median_in < 1e-9f) {
    return 2.0f;
  }
  float low = 0.0f;
  float high = 7.0f;
  float best = 2.0f;
  for (int i = 0; i < 40; ++i) {
    const float mid = (low + high) * 0.5f;
    const float val =
        hypermetric_hyperbolic_stretch_value(median_in, std::pow(10.0f, mid), b);
    best = mid;
    if (std::abs(val - target_median) < 0.0001f) {
      break;
    }
    if (val < target_median) {
      low = mid;
    } else {
      high = mid;
    }
  }
  return best;
}

HyperMetricStretchDiagnostics run_hypermetric_stretch_rgb(
    Matrix2Df &R, Matrix2Df &G, Matrix2Df &B,
    const HyperMetricStretchConfig &cfg,
    const std::vector<uint8_t> *statistics_mask, int mask_rows, int mask_cols,
    const std::vector<uint8_t> *output_mask) {
  HyperMetricStretchDiagnostics diag;
  diag.target_bg = cfg.target_bg;
  diag.protect_b = cfg.protect_b;
  diag.convergence_power = cfg.convergence_power;
  diag.linear_expansion = cfg.linear_expansion;

  if (R.rows() <= 0 || R.cols() <= 0 || G.rows() != R.rows() ||
      B.rows() != R.rows() || G.cols() != R.cols() || B.cols() != R.cols()) {
    diag.status = "error";
    diag.error_message = "RGB dimensions mismatch";
    return diag;
  }
  const size_t mask_size =
      static_cast<size_t>(R.rows()) * static_cast<size_t>(R.cols());
  if (statistics_mask != nullptr &&
      (mask_rows != R.rows() || mask_cols != R.cols() ||
       statistics_mask->size() != mask_size)) {
    diag.status = "error";
    diag.error_message = "statistics mask dimensions mismatch";
    return diag;
  }
  if (output_mask != nullptr &&
      (mask_rows != R.rows() || mask_cols != R.cols() ||
       output_mask->size() != mask_size)) {
    diag.status = "error";
    diag.error_message = "output mask dimensions mismatch";
    return diag;
  }
  if (output_mask == nullptr) {
    output_mask = statistics_mask;
  }

  normalize_rgb_input_inplace(R, G, B, statistics_mask, output_mask,
                              mask_rows, mask_cols);

  std::string resolved_profile;
  std::string profile_source;
  const std::array<float, 3> w = hypermetric_profile_weights(
      cfg.sensor_profile, cfg.fallback_profile, &resolved_profile, &profile_source);
  diag.profile = resolved_profile;
  diag.profile_source = profile_source;
  diag.weights_r = w[0];
  diag.weights_g = w[1];
  diag.weights_b = w[2];

  diag.anchor =
      cfg.adaptive_anchor
          ? calculate_anchor_adaptive(R, G, B, w, statistics_mask, mask_rows,
                                      mask_cols)
          : calculate_anchor_statistical(R, G, B, statistics_mask, mask_rows,
                                         mask_cols);

  const int rows = static_cast<int>(R.rows());
  const int cols = static_cast<int>(R.cols());
  Matrix2Df L(rows, cols);
  std::vector<float> valid_luma;
  valid_luma.reserve(std::min<size_t>(static_cast<size_t>(rows) *
                                          static_cast<size_t>(cols),
                                      500000u));
  const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t stride = std::max<size_t>(1, total / 500000u);
  size_t linear = 0;
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x, ++linear) {
      if (!mask_valid(output_mask, mask_rows, mask_cols, y, x)) {
        L(y, x) = 0.0f;
        continue;
      }
      const float ra = std::max(sanitize01(R(y, x)) - diag.anchor, 0.0f);
      const float ga = std::max(sanitize01(G(y, x)) - diag.anchor, 0.0f);
      const float ba = std::max(sanitize01(B(y, x)) - diag.anchor, 0.0f);
      const float l = w[0] * ra + w[1] * ga + w[2] * ba;
      L(y, x) = l;
      if (linear % stride == 0 && l > 1e-7f &&
          mask_valid(statistics_mask, mask_rows, mask_cols, y, x)) {
        valid_luma.push_back(l);
      }
    }
  }

  diag.star_pressure = estimate_star_pressure(valid_luma);
  float target_bg = cfg.target_bg;
  target_bg = std::clamp(target_bg, 0.05f, 0.50f);
  diag.target_bg = target_bg;

  std::vector<float> auto_luma;
  if (cfg.log_d_mode != "fixed") {
    auto_luma = build_auto_luma_sample(R, G, B, w, cfg.adaptive_anchor,
                                       statistics_mask, mask_rows, mask_cols);
  }
  diag.log_d = cfg.log_d_mode == "fixed"
                   ? std::clamp(cfg.fixed_log_d, 0.0f, 7.0f)
                   : solve_auto_log_d(std::move(auto_luma), target_bg,
                                      cfg.protect_b, diag.star_pressure,
                                      cfg.mode == "ready_to_use");

  float strategy = cfg.fixed_color_strategy;
  if (cfg.color_strategy == "auto") {
    strategy = cfg.fixed_color_strategy;
  }
  strategy = std::clamp(strategy, -1.0f, 1.0f);
  diag.color_strategy = strategy;
  float color_grip = cfg.color_grip;
  float shadow_conv = cfg.shadow_convergence;
  if (cfg.mode == "ready_to_use") {
    if (strategy < 0.0f) {
      shadow_conv = std::abs(strategy) * 3.0f;
      color_grip = 1.0f;
    } else {
      color_grip = 1.0f - strategy * 0.6f;
      shadow_conv = 0.0f;
    }
  }
  color_grip = std::clamp(color_grip, 0.0f, 1.0f);
  shadow_conv = std::max(0.0f, shadow_conv);
  diag.color_grip = color_grip;
  diag.shadow_convergence = shadow_conv;

  const float D = std::pow(10.0f, diag.log_d);
  Matrix2Df Ls(rows, cols);
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      Ls(y, x) = std::clamp(
          hypermetric_hyperbolic_stretch_value(L(y, x), D, cfg.protect_b),
          0.0f, 1.0f);
    }
  }
  if (cfg.mode != "ready_to_use" && cfg.linear_expansion > 0.001f) {
    apply_linear_expansion(Ls, cfg.linear_expansion, output_mask, mask_rows,
                           mask_cols);
  }

  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      if (!mask_valid(output_mask, mask_rows, mask_cols, y, x)) {
        R(y, x) = G(y, x) = B(y, x) = 0.0f;
        continue;
      }
      const float ra = std::max(sanitize01(R(y, x)) - diag.anchor, 0.0f);
      const float ga = std::max(sanitize01(G(y, x)) - diag.anchor, 0.0f);
      const float ba = std::max(sanitize01(B(y, x)) - diag.anchor, 0.0f);
      const float safe_l = L(y, x) + 1e-9f;
      const float k = std::pow(Ls(y, x), cfg.convergence_power);
      float rf = (ra / safe_l) * (1.0f - k) + k;
      float gf = (ga / safe_l) * (1.0f - k) + k;
      float bf = (ba / safe_l) * (1.0f - k) + k;
      float rv = Ls(y, x) * rf;
      float gv = Ls(y, x) * gf;
      float bv = Ls(y, x) * bf;

      if (color_grip < 1.0f || shadow_conv > 0.01f) {
        const float scalar_r = std::clamp(
            hypermetric_hyperbolic_stretch_value(ra, D, cfg.protect_b), 0.0f, 1.0f);
        const float scalar_g = std::clamp(
            hypermetric_hyperbolic_stretch_value(ga, D, cfg.protect_b), 0.0f, 1.0f);
        const float scalar_b = std::clamp(
            hypermetric_hyperbolic_stretch_value(ba, D, cfg.protect_b), 0.0f, 1.0f);
        float grip = color_grip;
        if (shadow_conv > 0.01f) {
          grip *= std::pow(Ls(y, x), shadow_conv);
        }
        rv = rv * grip + scalar_r * (1.0f - grip);
        gv = gv * grip + scalar_g * (1.0f - grip);
        bv = bv * grip + scalar_b * (1.0f - grip);
      }

      R(y, x) = std::clamp(rv * 0.995f + 0.005f, 0.0f, 1.0f);
      G(y, x) = std::clamp(gv * 0.995f + 0.005f, 0.0f, 1.0f);
      B(y, x) = std::clamp(bv * 0.995f + 0.005f, 0.0f, 1.0f);
    }
  }

  if (cfg.mode == "ready_to_use") {
    adaptive_output_scaling(R, G, B, w, target_bg, statistics_mask,
                            output_mask, mask_rows, mask_cols);
    soft_clip(R, 0.98f, 2.0f);
    soft_clip(G, 0.98f, 2.0f);
    soft_clip(B, 0.98f, 2.0f);
  }

  size_t valid_px = 0;
  size_t black = 0;
  size_t white = 0;
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      if (!mask_valid(statistics_mask, mask_rows, mask_cols, y, x)) {
        continue;
      }
      ++valid_px;
      if (R(y, x) <= 1e-7f || G(y, x) <= 1e-7f || B(y, x) <= 1e-7f) {
        ++black;
      }
      if (R(y, x) >= 1.0f - 1e-7f || G(y, x) >= 1.0f - 1e-7f ||
          B(y, x) >= 1.0f - 1e-7f) {
        ++white;
      }
    }
  }
  if (valid_px > 0) {
    diag.black_clip_percent = 100.0f * static_cast<float>(black) /
                              static_cast<float>(valid_px);
    diag.white_clip_percent = 100.0f * static_cast<float>(white) /
                              static_cast<float>(valid_px);
  }
  diag.success = true;
  diag.status = "ok";
  return diag;
}

} // namespace tile_compile::image
