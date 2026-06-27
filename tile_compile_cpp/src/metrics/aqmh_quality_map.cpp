#include "tile_compile/metrics/aqmh_quality_map.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <thread>
#include <unordered_map>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#if __has_include(<opencv2/core/cuda.hpp>) && \
    __has_include(<opencv2/cudafilters.hpp>)
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#define TILE_COMPILE_AQMH_CUDA_FILTERS 1
#else
#define TILE_COMPILE_AQMH_CUDA_FILTERS 0
#endif

#if __has_include(<opencv2/core/ocl.hpp>)
#include <opencv2/core/ocl.hpp>
#define TILE_COMPILE_AQMH_OPENCL 1
#else
#define TILE_COMPILE_AQMH_OPENCL 0
#endif

namespace tile_compile::metrics {
namespace {

// IEEE-754 exponent test is branch-free and vectorizes without enabling
// -ffast-math (which would invalidate AQMH NaN/canvas semantics).
#pragma omp declare simd notinbranch
bool finite(float v) {
  return (std::bit_cast<uint32_t>(v) & 0x7f800000u) != 0x7f800000u;
}

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
  const bool use_mask = !mask.empty();
  const bool mask_shape_valid =
      !use_mask || (mask_w == frame.cols() && mask_h == frame.rows() &&
                    mask.size() == static_cast<size_t>(frame.size()));
  const auto n = static_cast<std::ptrdiff_t>(frame.size());
  const float *src = frame.data();
  float *dst = out.data();
#pragma omp simd
  for (std::ptrdiff_t i = 0; i < n; ++i) {
    const float v = src[i];
    const bool valid = mask_shape_valid &&
                       (!use_mask || mask[static_cast<size_t>(i)] != 0u);
    dst[i] = (valid && finite(v)) ? v : nan_value();
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

bool accelerated_local_variance(const Matrix2Df &m, int r,
                                core::AccelerationBackend backend,
                                Matrix2Df &out,
                                cv::cuda::Stream *stream) {
  if (backend != core::AccelerationBackend::opencv_cuda &&
      backend != core::AccelerationBackend::opencv_opencl)
    return false;
  const int rows = static_cast<int>(m.rows());
  const int cols = static_cast<int>(m.cols());
  if (rows <= 0 || cols <= 0)
    return false;

  Matrix2Df values(rows, cols);
  Matrix2Df squares(rows, cols);
  Matrix2Df support(rows, cols);
  const auto n = static_cast<std::ptrdiff_t>(m.size());
#pragma omp simd
  for (std::ptrdiff_t i = 0; i < n; ++i) {
    const float v = m.data()[i];
    const bool valid = finite(v);
    const float clean = valid ? v : 0.0f;
    values.data()[i] = clean;
    squares.data()[i] = clean * clean;
    support.data()[i] = valid ? 1.0f : 0.0f;
  }

  Matrix2Df sums(rows, cols), square_sums(rows, cols), counts(rows, cols);
  const cv::Size kernel_size(2 * r + 1, 2 * r + 1);
  try {
    if (backend == core::AccelerationBackend::opencv_cuda) {
#if TILE_COMPILE_AQMH_CUDA_FILTERS
      cv::Mat h_values(rows, cols, CV_32F, values.data());
      cv::Mat h_squares(rows, cols, CV_32F, squares.data());
      cv::Mat h_support(rows, cols, CV_32F, support.data());
      cv::cuda::GpuMat d_values, d_squares, d_support;
      cv::cuda::GpuMat d_sums, d_square_sums, d_counts;
      // CUDA filters keep internal work buffers and are not safe to share
      // across workers. Cache by kernel size per worker thread instead.
      thread_local std::unordered_map<int, cv::Ptr<cv::cuda::Filter>>
          box_filter_cache;
      const int kernel_key = kernel_size.width;
      auto &filter = box_filter_cache[kernel_key];
      if (!filter) {
        filter = cv::cuda::createBoxFilter(
            CV_32F, CV_32F, kernel_size, cv::Point(-1, -1),
            cv::BORDER_CONSTANT);
      }
      cv::cuda::Stream &cuda_stream =
          stream ? *stream : cv::cuda::Stream::Null();
      d_values.upload(h_values, cuda_stream);
      d_squares.upload(h_squares, cuda_stream);
      d_support.upload(h_support, cuda_stream);
      filter->apply(d_values, d_sums, cuda_stream);
      filter->apply(d_squares, d_square_sums, cuda_stream);
      filter->apply(d_support, d_counts, cuda_stream);
      cv::Mat h_sums(rows, cols, CV_32F, sums.data());
      cv::Mat h_square_sums(rows, cols, CV_32F, square_sums.data());
      cv::Mat h_counts(rows, cols, CV_32F, counts.data());
      d_sums.download(h_sums, cuda_stream);
      d_square_sums.download(h_square_sums, cuda_stream);
      d_counts.download(h_counts, cuda_stream);
      cuda_stream.waitForCompletion();
#else
      return false;
#endif
    } else {
#if TILE_COMPILE_AQMH_OPENCL
      cv::Mat h_values(rows, cols, CV_32F, values.data());
      cv::Mat h_squares(rows, cols, CV_32F, squares.data());
      cv::Mat h_support(rows, cols, CV_32F, support.data());
      cv::UMat u_values, u_squares, u_support;
      h_values.copyTo(u_values);
      h_squares.copyTo(u_squares);
      h_support.copyTo(u_support);
      cv::UMat u_sums, u_square_sums, u_counts;
      cv::boxFilter(u_values, u_sums, CV_32F, kernel_size, cv::Point(-1, -1),
                    true, cv::BORDER_CONSTANT);
      cv::boxFilter(u_squares, u_square_sums, CV_32F, kernel_size,
                    cv::Point(-1, -1), true, cv::BORDER_CONSTANT);
      cv::boxFilter(u_support, u_counts, CV_32F, kernel_size,
                    cv::Point(-1, -1), true, cv::BORDER_CONSTANT);
      cv::Mat h_sums(rows, cols, CV_32F, sums.data());
      cv::Mat h_square_sums(rows, cols, CV_32F, square_sums.data());
      cv::Mat h_counts(rows, cols, CV_32F, counts.data());
      u_sums.copyTo(h_sums);
      u_square_sums.copyTo(h_square_sums);
      u_counts.copyTo(h_counts);
#else
      return false;
#endif
    }
  } catch (...) {
    return false;
  }

  out.resize(rows, cols);
  out.setConstant(nan_value());
  const float window_area = static_cast<float>(kernel_size.area());
#pragma omp simd
  for (std::ptrdiff_t i = 0; i < n; ++i) {
    const float support_fraction = counts.data()[i];
    const float count = support_fraction * window_area;
    if (count > 0.0f && count < 3.0f) {
      out.data()[i] = 0.0f;
    } else if (count >= 3.0f) {
      const double mean =
          static_cast<double>(sums.data()[i]) / support_fraction;
      const double mean_square =
          static_cast<double>(square_sums.data()[i]) / support_fraction;
      out.data()[i] = static_cast<float>(
          std::max(0.0, mean_square - mean * mean));
    }
  }
  return true;
}

struct LocalMeanResult {
  Matrix2Df mean;
  Matrix2Df count;
};

// Separable box filter. Returning the support count lets callers reuse the
// same traversal for masked fallback decisions.
LocalMeanResult local_mean_and_count(const Matrix2Df &m, int r) {
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
  LocalMeanResult result{Matrix2Df(rows, cols), Matrix2Df(rows, cols)};
  result.mean.setConstant(nan_value());
  result.count.setZero();
  for (int x = 0; x < cols; ++x) {
    double s = 0.0, c = 0.0;
    for (int y = 0; y <= std::min(r, rows - 1); ++y) { s += hsum(y, x); c += hcnt(y, x); }
    for (int y = 0; y < rows; ++y) {
      result.count(y, x) = static_cast<float>(c);
      if (c > 0.0) result.mean(y, x) = static_cast<float>(s / c);
      const int yadd = y + r + 1;
      if (yadd < rows) { s += hsum(yadd, x); c += hcnt(yadd, x); }
      const int yrem = y - r;
      if (yrem >= 0) { s -= hsum(yrem, x); c -= hcnt(yrem, x); }
    }
  }
  return result;
}

Matrix2Df local_mean(const Matrix2Df &m, int r) {
  return local_mean_and_count(m, r).mean;
}

Matrix2Df phi_snr(const Matrix2Df &img, const Matrix2Df &bg,
                  const Matrix2Df &valid_cnt, int r,
                  bool &scene_dependent) {
  // Performance optimisation: replace the O(W*H*R²) per-pixel median window
  // with a two-step approach:
  //   1. Local background b_s via separable O(W*H) box mean (local_mean).
  //      The mean is a fast approximation; for smooth background regions the
  //      difference from the true median is negligible at the signal/noise
  //      scale.  For pixels with fewer than 3 valid neighbours the fallback
  //      path is taken and scene_dependent is set.
  //   2. Local noise sigma via separable O(W*H) MAD approximation:
  //      sigma ≈ 1.4826 * local_mean(|img - b_s|, r).
  //      Because |img - bg| is always non-negative, the mean of absolute
  //      deviations is proportional to the true MAD; the 1.4826 factor
  //      converts it to a consistent sigma estimate under Gaussian noise.
  //
  // For tiny windows (< 3 valid neighbours) the original O(R²) window
  // fallback is used and scene_dependent is set.  At scale 0 with R=4 on
  // a 24 Mpx sensor this reduces the hot-path from O(W*H*81*log81) to
  // O(W*H).
  const int rows = static_cast<int>(img.rows());
  const int cols = static_cast<int>(img.cols());

  // Signal = local_mean(max(img - bg, 0), r).
  Matrix2Df sig_img(rows, cols);
  sig_img.setConstant(nan_value());
  const auto n = static_cast<std::ptrdiff_t>(img.size());
  const float *img_data = img.data();
  const float *bg_data = bg.data();
  float *sig_data = sig_img.data();
#pragma omp simd
  for (std::ptrdiff_t i = 0; i < n; ++i) {
    const bool valid = finite(img_data[i]) && finite(bg_data[i]);
    sig_data[i] = valid ? std::max(img_data[i] - bg_data[i], 0.0f)
                        : nan_value();
  }
  const Matrix2Df mu = local_mean(sig_img, r);

  // Step 4: noise = 1.4826 * local_mean(|img - bg|, r)  (MAD approximation).
  Matrix2Df abs_dev(rows, cols);
  abs_dev.setConstant(nan_value());
  float *abs_dev_data = abs_dev.data();
#pragma omp simd
  for (std::ptrdiff_t i = 0; i < n; ++i) {
    const bool valid = finite(img_data[i]) && finite(bg_data[i]);
    abs_dev_data[i] = valid ? std::abs(img_data[i] - bg_data[i]) : nan_value();
  }
  const Matrix2Df noise_map = local_mean(abs_dev, r);

  // Assemble phi_snr, with O(R²) fallback for pixels with < 3 valid neighbours.
  Matrix2Df out(rows, cols);
  out.setConstant(nan_value());
  WindowBuf buf, tmp;
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      const float n_valid = valid_cnt(y, x);
      if (n_valid <= 0.0f)
        continue;

      if (n_valid < 3.0f) {
        // Fallback: original O(R²) window path.
        scene_dependent = true;
        fill_window(img, x, y, r, buf);
        if (buf.empty()) continue;
        double sum = 0.0;
        for (int i = 0; i < buf.size(); ++i)
          sum += std::max(buf.data[static_cast<size_t>(i)], 0.0f);
        out(y, x) = static_cast<float>(sum / buf.size()) / eps_aqmh;
        continue;
      }

      if (!finite(mu(y, x)) || !finite(noise_map(y, x)))
        continue;
      const float sigma = std::max(1.4826f * noise_map(y, x), eps_aqmh);
      out(y, x) = mu(y, x) / sigma;
    }
  }
  return out;
}

Matrix2Df phi_artifact(const Matrix2Df &img, const Matrix2Df &blur, int r,
                       float k_artifact, float frac_artifact_max) {
  // Performance optimisation: replace the O(W*H*R²) per-pixel MAD+outlier-
  // fraction loop with a separable approach.
  //
  // Algorithm:
  //   1. hp = img - local_mean(img, r)              — separable O(W*H)
  //   2. tau_map = 1.4826 * local_mean(|hp|, r)     — separable O(W*H),
  //      MAD approximation (same as phi_snr noise estimate)
  //   3. outlier_ind = (|hp| > k * tau_map) ? 1 : 0 — pixel-wise O(W*H)
  //   4. frac_out = local_mean(outlier_ind, r)       — separable O(W*H)
  //   5. phi_artifact = min_quality + (1-min_quality) * (1 - clip(frac_out/frac_max, 0,1))
  //
  // Step 2 uses local_mean(|hp|) as a fast proxy for the true local MAD.
  // Under symmetric noise this is proportional: E[|x - mu|] ≈ sigma*sqrt(2/pi)
  // and MAD ≈ 0.6745*sigma, so the ratio is ~1.22 × mean_abs_dev.  The 1.4826
  // constant scales to a Gaussian-consistent sigma regardless of the proxy
  // used, so relative outlier detection is stable.
  //
  // The overall complexity is O(W*H) per scale instead of O(W*H*R²).

  // Step 1: high-pass residual.
  Matrix2Df hp(img.rows(), img.cols());
  hp.setConstant(nan_value());
  const auto n = static_cast<std::ptrdiff_t>(img.size());
  const float *img_data = img.data();
  const float *blur_data = blur.data();
  float *hp_data = hp.data();
#pragma omp simd
  for (std::ptrdiff_t i = 0; i < n; ++i) {
    const bool valid = finite(img_data[i]) && finite(blur_data[i]);
    hp_data[i] = valid ? img_data[i] - blur_data[i] : nan_value();
  }

  // Step 2: local robust scale = 1.4826 * local_mean(|hp|, r).
  Matrix2Df abs_hp(img.rows(), img.cols());
  abs_hp.setConstant(nan_value());
  float *abs_hp_data = abs_hp.data();
#pragma omp simd
  for (std::ptrdiff_t i = 0; i < n; ++i) {
    abs_hp_data[i] = finite(hp_data[i]) ? std::abs(hp_data[i]) : nan_value();
  }
  const Matrix2Df mean_abs = local_mean(abs_hp, r);

  // Step 3: binary outlier indicator.
  Matrix2Df outlier_ind(img.rows(), img.cols());
  outlier_ind.setConstant(nan_value());
  const float *mean_abs_data = mean_abs.data();
  float *outlier_data = outlier_ind.data();
#pragma omp simd
  for (std::ptrdiff_t i = 0; i < n; ++i) {
    const bool valid = finite(hp_data[i]) && finite(mean_abs_data[i]);
    const float tau = std::max(1.4826f * mean_abs_data[i], eps_aqmh);
    outlier_data[i] = valid ? ((std::abs(hp_data[i]) > k_artifact * tau)
                                   ? 1.0f
                                   : 0.0f)
                              : nan_value();
  }

  // Step 4: local outlier fraction = local_mean(outlier_ind, r).
  const Matrix2Df frac_out = local_mean(outlier_ind, r);

  // Step 5: assemble phi_artifact with min_quality floor.
  Matrix2Df out(img.rows(), img.cols());
  out.setConstant(nan_value());
  constexpr float min_quality = 0.01f; // prevents black blocks in smooth regions
  const float *frac_data = frac_out.data();
  float *out_data = out.data();
#pragma omp simd
  for (std::ptrdiff_t i = 0; i < n; ++i) {
      const float value = std::clamp(
          min_quality + (1.0f - min_quality) *
              (1.0f - std::clamp(frac_data[i] /
                                     std::max(frac_artifact_max, eps_aqmh),
                                 0.0f, 1.0f)),
          min_quality, 1.0f);
      out_data[i] = finite(frac_data[i]) ? value : nan_value();
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
  const auto n = static_cast<std::ptrdiff_t>(m.size());
  const float *m_data = m.data();
  float *out_data = out.data();
#pragma omp simd
  for (std::ptrdiff_t i = 0; i < n; ++i) {
    const float v = m_data[i];
    out_data[i] = finite(v) ? (v - med) / scale : nan_value();
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
    const config::AqmhPyramidConfig &cfg,
    core::AccelerationBackend backend,
    cv::cuda::Stream *stream) {
  AqmhQualityMapResult result;
  result.q_map = Matrix2Df::Zero(frame.rows(), frame.cols());
  if (frame.rows() <= 0 || frame.cols() <= 0)
    return result;

  const Matrix2Df masked =
      canvas_masked_frame(frame, canvas_mask, canvas_mask_width,
                          canvas_mask_height);
  const int min_dim = std::min(frame.rows(), frame.cols());
  Matrix2Dd log_sum = Matrix2Dd::Zero(frame.rows(), frame.cols());
  std::vector<uint8_t> veto(static_cast<size_t>(frame.size()), 0u);
  size_t computed_scales = 0;

  for (int s = 0; s < cfg.scales; ++s) {
    const int factor = 1 << (2 * s);
    if (factor > std::max(1, min_dim / 16)) {
      result.diagnostics.omitted_scales.push_back(s);
      continue;
    }

    const Matrix2Df img_s = downsample_valid_mean(masked, factor);
    const int radius = std::max(1, cfg.base_window_px);
    const Matrix2Df laplacian = masked_laplacian(img_s);
    Matrix2Df sharp;
    if (backend == core::AccelerationBackend::cpu) {
      sharp = local_variance(laplacian, radius);
    } else if (accelerated_local_variance(laplacian, radius, backend, sharp, stream)) {
      result.diagnostics.acceleration_used = true;
    } else {
      auto fallback = compute_aqmh_quality_map(
          frame, canvas_mask, canvas_mask_width, canvas_mask_height, cfg,
          core::AccelerationBackend::cpu, nullptr);
      fallback.diagnostics.acceleration_fallback = true;
      return fallback;
    }
    const LocalMeanResult local_img = local_mean_and_count(img_s, radius);
    bool scene_dependent = false;
    const Matrix2Df snr = phi_snr(img_s, local_img.mean, local_img.count,
                                  radius, scene_dependent);
    const Matrix2Df artifact = phi_artifact(
        img_s, local_img.mean, radius, cfg.k_artifact,
        cfg.frac_artifact_max);
    result.diagnostics.scene_dependent_snr =
        result.diagnostics.scene_dependent_snr || scene_dependent;

    if (s == 0)
      result.diagnostics.sharpness_p50 = finite_median(sharp);
    if (s == 1)
      result.diagnostics.snr_p50 = finite_median(snr);

    const Matrix2Df psi = mask_aware_bilinear_upsample(
        compute_psi(sharp, snr, artifact, cfg), frame.cols(), frame.rows(),
        factor);
    for (int y = 0; y < frame.rows(); ++y) {
      for (int x = 0; x < frame.cols(); ++x) {
        const size_t idx = static_cast<size_t>(y) *
                               static_cast<size_t>(frame.cols()) +
                           static_cast<size_t>(x);
        if (veto[idx] != 0u)
          continue;
        const float v = psi(y, x);
        if (!finite(v) || v <= 0.0f) {
          veto[idx] = 1u;
        } else {
          log_sum(y, x) +=
              std::log(static_cast<double>(std::clamp(v, eps_aqmh, 1.0f)));
        }
      }
    }
    ++computed_scales;
  }

  if (computed_scales == 0)
    return result;

  for (int y = 0; y < frame.rows(); ++y) {
    for (int x = 0; x < frame.cols(); ++x) {
      if (!mask_valid(canvas_mask, canvas_mask_width, canvas_mask_height, x,
                      y)) {
        result.q_map(y, x) = 0.0f;
        continue;
      }
      const size_t idx = static_cast<size_t>(y) *
                             static_cast<size_t>(frame.cols()) +
                         static_cast<size_t>(x);
      result.q_map(y, x) =
          veto[idx] != 0u
              ? 0.0f
              : static_cast<float>(
                    std::exp(log_sum(y, x) / computed_scales));
    }
  }

  return result;
}

} // namespace tile_compile::metrics
