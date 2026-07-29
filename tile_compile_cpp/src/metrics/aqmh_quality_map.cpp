#include "tile_compile/metrics/aqmh_quality_map.hpp"
#include "tile_compile/metrics/aqmh_eps.hpp"
#include "tile_compile/core/utils.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <thread>
#include <unordered_map>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#if __has_include(<opencv2/core/cuda.hpp>) && \
    __has_include(<opencv2/cudafilters.hpp>) && \
    __has_include(<opencv2/cudaarithm.hpp>)
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
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
    med = 0.5f * (med + *std::max_element(values.begin(), values.begin() + mid));
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
    med = 0.5f * (med + *std::max_element(buf.begin(), buf.begin() + mid));
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

Matrix2Df source_masked_frame(const Matrix2Df &frame,
                              const std::vector<uint8_t> &canvas_mask,
                              const std::vector<uint8_t> &frame_mask,
                              int mask_w, int mask_h) {
  Matrix2Df out(frame.rows(), frame.cols());
  const bool use_mask = !canvas_mask.empty();
  const bool use_frame_mask = !frame_mask.empty();
  const bool mask_shape_valid =
      !use_mask || (mask_w == frame.cols() && mask_h == frame.rows() &&
                    canvas_mask.size() == static_cast<size_t>(frame.size()));
  const bool frame_mask_shape_valid =
      !use_frame_mask || frame_mask.size() == static_cast<size_t>(frame.size());
  if (use_mask && !mask_shape_valid) {
    std::cerr << "[AQMH-QM] Warning: canvas mask shape mismatch — mask "
              << mask_w << "x" << mask_h << " (" << canvas_mask.size()
              << ") vs frame " << frame.cols() << "x" << frame.rows()
              << " (" << frame.size() << ")\n";
  }
  if (use_frame_mask && !frame_mask_shape_valid) {
    std::cerr << "[AQMH-QM] Warning: frame mask size mismatch — mask "
              << frame_mask.size() << " vs frame " << frame.size() << "\n";
  }
  const auto n = static_cast<std::ptrdiff_t>(frame.size());
  const float *src = frame.data();
  float *dst = out.data();
#pragma omp simd
  for (std::ptrdiff_t i = 0; i < n; ++i) {
    const float v = src[i];
    const bool valid = mask_shape_valid && frame_mask_shape_valid &&
                       (!use_mask || canvas_mask[static_cast<size_t>(i)] != 0u) &&
                       (!use_frame_mask || frame_mask[static_cast<size_t>(i)] != 0u);
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
#pragma omp parallel for collapse(2) schedule(static)
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
#pragma omp parallel for collapse(2) schedule(static)
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
        out(y, x) = c - static_cast<float>(sum / static_cast<double>(count));
    }
  }
  return out;
}

Matrix2Df local_variance_linear(const Matrix2Df &m, int r) {
  const int rows = static_cast<int>(m.rows());
  const int cols = static_cast<int>(m.cols());
  const int radius = std::clamp(r, 0, kMaxWindowR);
  Matrix2Df horizontal_sum = Matrix2Df::Zero(rows, cols);
  Matrix2Df horizontal_square_sum = Matrix2Df::Zero(rows, cols);
  Matrix2Df horizontal_count = Matrix2Df::Zero(rows, cols);

  // Sliding horizontal moments. This replaces a complete window scan per
  // pixel while retaining the source-valid (finite) support contract.
#pragma omp parallel for schedule(static)
  for (int y = 0; y < rows; ++y) {
    double sum = 0.0;
    double square_sum = 0.0;
    int count = 0;
    for (int x = 0; x <= std::min(radius, cols - 1); ++x) {
      const float value = m(y, x);
      if (finite(value)) {
        sum += value;
        square_sum += static_cast<double>(value) * value;
        ++count;
      }
    }
    for (int x = 0; x < cols; ++x) {
      horizontal_sum(y, x) = static_cast<float>(sum);
      horizontal_square_sum(y, x) = static_cast<float>(square_sum);
      horizontal_count(y, x) = static_cast<float>(count);
      const int add_x = x + radius + 1;
      if (add_x < cols) {
        const float value = m(y, add_x);
        if (finite(value)) {
          sum += value;
          square_sum += static_cast<double>(value) * value;
          ++count;
        }
      }
      const int remove_x = x - radius;
      if (remove_x >= 0) {
        const float value = m(y, remove_x);
        if (finite(value)) {
          sum -= value;
          square_sum -= static_cast<double>(value) * value;
          --count;
        }
      }
    }
  }

  Matrix2Df out(rows, cols);
  out.setConstant(nan_value());
#pragma omp parallel for schedule(static)
  for (int x = 0; x < cols; ++x) {
    double sum = 0.0;
    double square_sum = 0.0;
    double count = 0.0;
    for (int y = 0; y <= std::min(radius, rows - 1); ++y) {
      sum += horizontal_sum(y, x);
      square_sum += horizontal_square_sum(y, x);
      count += horizontal_count(y, x);
    }
    for (int y = 0; y < rows; ++y) {
      if (count > 0.0 && count < 3.0) {
        out(y, x) = 0.0f;
      } else if (count >= 3.0) {
        const double mean = sum / count;
        out(y, x) = static_cast<float>(
            std::max(0.0, square_sum / count - mean * mean));
      }
      const int add_y = y + radius + 1;
      if (add_y < rows) {
        sum += horizontal_sum(add_y, x);
        square_sum += horizontal_square_sum(add_y, x);
        count += horizontal_count(add_y, x);
      }
      const int remove_y = y - radius;
      if (remove_y >= 0) {
        sum -= horizontal_sum(remove_y, x);
        square_sum -= horizontal_square_sum(remove_y, x);
        count -= horizontal_count(remove_y, x);
      }
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
  Matrix2Df support(rows, cols);
  const auto n = static_cast<std::ptrdiff_t>(m.size());
#pragma omp simd
  for (std::ptrdiff_t i = 0; i < n; ++i) {
    const float v = m.data()[i];
    const bool valid = finite(v);
    const float clean = valid ? v : 0.0f;
    values.data()[i] = clean;
    support.data()[i] = valid ? 1.0f : 0.0f;
  }

  Matrix2Df sums(rows, cols), square_sums(rows, cols), counts(rows, cols);
  const cv::Size kernel_size(2 * r + 1, 2 * r + 1);
  try {
    if (backend == core::AccelerationBackend::opencv_cuda) {
#if TILE_COMPILE_AQMH_CUDA_FILTERS
      cv::Mat h_values(rows, cols, CV_32F, values.data());
      cv::Mat h_support(rows, cols, CV_32F, support.data());
      struct CudaMomentsWorkspace {
        cv::cuda::GpuMat values, squares, support;
        cv::cuda::GpuMat sums, square_sums, counts;
      };
      thread_local CudaMomentsWorkspace workspace;
      auto &d_values = workspace.values;
      auto &d_squares = workspace.squares;
      auto &d_support = workspace.support;
      auto &d_sums = workspace.sums;
      auto &d_square_sums = workspace.square_sums;
      auto &d_counts = workspace.counts;
      for (auto *buffer : {&d_values, &d_squares, &d_support, &d_sums,
                           &d_square_sums, &d_counts})
        buffer->create(rows, cols, CV_32F);
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
      d_support.upload(h_support, cuda_stream);
      cv::cuda::multiply(d_values, d_values, d_squares, 1.0, -1,
                         cuda_stream);
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
      cv::Mat h_support(rows, cols, CV_32F, support.data());
      cv::UMat u_values, u_squares, u_support;
      h_values.copyTo(u_values);
      h_support.copyTo(u_support);
      cv::multiply(u_values, u_values, u_squares);
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

struct LocalMeanPairResult {
  Matrix2Df first_mean;
  Matrix2Df second_mean;
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
#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static)
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

// Fused separable means for matrices with identical finite support. This
// avoids building and traversing the same support-count intermediates twice.
LocalMeanPairResult local_mean_pair_and_count(
    const Matrix2Df &first, const Matrix2Df &second, int r) {
  const int rows = static_cast<int>(first.rows());
  const int cols = static_cast<int>(first.cols());
  Matrix2Df first_hsum = Matrix2Df::Zero(rows, cols);
  Matrix2Df second_hsum = Matrix2Df::Zero(rows, cols);
  Matrix2Df hcnt = Matrix2Df::Zero(rows, cols);

#pragma omp parallel for schedule(static)
  for (int y = 0; y < rows; ++y) {
    double first_sum = 0.0;
    double second_sum = 0.0;
    int count = 0;
    for (int x = 0; x <= std::min(r, cols - 1); ++x) {
      const float a = first(y, x);
      const float b = second(y, x);
      if (finite(a) && finite(b)) {
        first_sum += a;
        second_sum += b;
        ++count;
      }
    }
    for (int x = 0; x < cols; ++x) {
      first_hsum(y, x) = static_cast<float>(first_sum);
      second_hsum(y, x) = static_cast<float>(second_sum);
      hcnt(y, x) = static_cast<float>(count);
      const int xadd = x + r + 1;
      if (xadd < cols) {
        const float a = first(y, xadd);
        const float b = second(y, xadd);
        if (finite(a) && finite(b)) {
          first_sum += a;
          second_sum += b;
          ++count;
        }
      }
      const int xrem = x - r;
      if (xrem >= 0) {
        const float a = first(y, xrem);
        const float b = second(y, xrem);
        if (finite(a) && finite(b)) {
          first_sum -= a;
          second_sum -= b;
          --count;
        }
      }
    }
  }

  LocalMeanPairResult result{
      Matrix2Df(rows, cols), Matrix2Df(rows, cols), Matrix2Df(rows, cols)};
  result.first_mean.setConstant(nan_value());
  result.second_mean.setConstant(nan_value());
  result.count.setZero();
#pragma omp parallel for schedule(static)
  for (int x = 0; x < cols; ++x) {
    double first_sum = 0.0;
    double second_sum = 0.0;
    double count = 0.0;
    for (int y = 0; y <= std::min(r, rows - 1); ++y) {
      first_sum += first_hsum(y, x);
      second_sum += second_hsum(y, x);
      count += hcnt(y, x);
    }
    for (int y = 0; y < rows; ++y) {
      result.count(y, x) = static_cast<float>(count);
      if (count > 0.0) {
        result.first_mean(y, x) =
            static_cast<float>(first_sum / count);
        result.second_mean(y, x) =
            static_cast<float>(second_sum / count);
      }
      const int yadd = y + r + 1;
      if (yadd < rows) {
        first_sum += first_hsum(yadd, x);
        second_sum += second_hsum(yadd, x);
        count += hcnt(yadd, x);
      }
      const int yrem = y - r;
      if (yrem >= 0) {
        first_sum -= first_hsum(yrem, x);
        second_sum -= second_hsum(yrem, x);
        count -= hcnt(yrem, x);
      }
    }
  }
  return result;
}

Matrix2Df phi_snr(const Matrix2Df &img, const Matrix2Df &bg,
                  const Matrix2Df &valid_cnt, int r,
                  bool &scene_dependent) {
  // Separable O(W*H) fast path (restored from pre-v0.2):
  //   signal = local_mean(max(img - bg, 0), r)
  //   noise  = 1.4826 * local_mean(|img - bg|, r)  (MAD approximation)
  // eps_noise is computed once globally over source-valid pixels as a floor
  // (spec §2.2: eps_noise(I_s over W_s_valid)) — not per-window.
  const int rows = static_cast<int>(img.rows());
  const int cols = static_cast<int>(img.cols());

  // Global eps_noise floor: computed once over all finite pixels.
  const float global_eps_noise = eps_noise(finite_values(img));

  // Signal = local_mean(max(img - bg, 0), r).
  Matrix2Df sig_img(rows, cols);
  Matrix2Df abs_dev(rows, cols);
  sig_img.setConstant(nan_value());
  abs_dev.setConstant(nan_value());
  const auto np = static_cast<std::ptrdiff_t>(img.size());
  const float *img_data = img.data();
  const float *bg_data = bg.data();
  float *sig_data = sig_img.data();
  float *abs_dev_data = abs_dev.data();
#pragma omp simd
  for (std::ptrdiff_t i = 0; i < np; ++i) {
    const bool valid = finite(img_data[i]) && finite(bg_data[i]);
    if (valid) {
      const float delta = img_data[i] - bg_data[i];
      sig_data[i] = std::max(delta, 0.0f);
      abs_dev_data[i] = std::abs(delta);
    } else {
      sig_data[i] = nan_value();
      abs_dev_data[i] = nan_value();
    }
  }
  const LocalMeanPairResult local_signal_noise =
      local_mean_pair_and_count(sig_img, abs_dev, r);
  const Matrix2Df &mu = local_signal_noise.first_mean;
  const Matrix2Df &noise_map = local_signal_noise.second_mean;

  // Assemble phi_snr, with O(R²) fallback for pixels with < 3 valid neighbours.
  Matrix2Df out(rows, cols);
  out.setConstant(nan_value());
#pragma omp parallel for schedule(static)
  for (int y = 0; y < rows; ++y) {
    WindowBuf buf, tmp;
    for (int x = 0; x < cols; ++x) {
      const float n_valid = valid_cnt(y, x);
      if (n_valid <= 0.0f)
        continue;

      if (n_valid < 3.0f) {
#pragma omp atomic write
        scene_dependent = true;
        fill_window(img, x, y, r, buf);
        if (buf.empty()) continue;
        double sum = 0.0;
        for (int i = 0; i < buf.size(); ++i)
          sum += std::max(buf.data[static_cast<size_t>(i)], 0.0f);
        out(y, x) = static_cast<float>(sum / buf.size()) / global_eps_noise;
        continue;
      }

      if (!finite(mu(y, x)) || !finite(noise_map(y, x)))
        continue;
      const float sigma = std::max(core::kMadToSigma * noise_map(y, x), global_eps_noise);
      out(y, x) = mu(y, x) / sigma;
    }
  }
  return out;
}

Matrix2Df phi_artifact(const Matrix2Df &img, const Matrix2Df &blur, int r,
                       float k_artifact, float frac_artifact_max) {
  // Separable O(W*H) fast path (restored from pre-v0.2):
  //   hp = img - local_mean(img, r)
  //   tau_map = 1.4826 * local_mean(|hp|, r)  (MAD approximation)
  //   frac_out = local_mean((|hp| > k*tau_map) ? 1 : 0, r)
  // eps_scale is computed once globally over hp (spec §2.3: eps_scale(hp_s over H_s_valid)).
  const auto n = static_cast<std::ptrdiff_t>(img.size());
  const float *img_data = img.data();
  const float *blur_data = blur.data();

  // Step 1: high-pass residual hp = img - blur (where blur = local_mean).
  Matrix2Df hp(img.rows(), img.cols());
  hp.setConstant(nan_value());
  float *hp_data = hp.data();
#pragma omp simd
  for (std::ptrdiff_t i = 0; i < n; ++i) {
    const bool valid = finite(img_data[i]) && finite(blur_data[i]);
    hp_data[i] = valid ? img_data[i] - blur_data[i] : nan_value();
  }

  // Global eps_scale floor: computed once over all finite hp pixels.
  const float global_eps_scale = eps_scale(finite_values(hp));

  // Step 2: local robust scale = 1.4826 * local_mean(|hp|, r).
  Matrix2Df abs_hp(img.rows(), img.cols());
  abs_hp.setConstant(nan_value());
  float *abs_hp_data = abs_hp.data();
#pragma omp simd
  for (std::ptrdiff_t i = 0; i < n; ++i) {
    abs_hp_data[i] = finite(hp_data[i]) ? std::abs(hp_data[i]) : nan_value();
  }
  const Matrix2Df mean_abs = local_mean(abs_hp, r);

  // Step 3: binary outlier indicator (NaN where hp is invalid).
  // Spec §2.3.2c: frac_out denom = |H_s_valid| (finite hp pixels in window),
  // not the total window size. We track this via local_mean_and_count on both
  // the outlier indicator and the finite-hp indicator separately.
  Matrix2Df outlier_ind(img.rows(), img.cols());
  outlier_ind.setConstant(nan_value());
  const float *mean_abs_data = mean_abs.data();
  float *outlier_data = outlier_ind.data();
#pragma omp simd
  for (std::ptrdiff_t i = 0; i < n; ++i) {
    const bool valid = finite(hp_data[i]) && finite(mean_abs_data[i]);
    const float tau = std::max(core::kMadToSigma * mean_abs_data[i], global_eps_scale);
    outlier_data[i] = valid ? ((std::abs(hp_data[i]) > k_artifact * tau)
                                   ? 1.0f
                                   : 0.0f)
                              : nan_value();
  }

  // Step 4: outlier_count = local_sum(outlier_ind, r);
  //         h_valid_count = |H_s_valid| per pixel = local count of finite hp.
  // local_mean_and_count returns mean and count over finite values, so:
  //   outlier_count(p) = outlier_mean(p) * outlier_count_map(p)  [count of finite outlier_ind]
  //   h_valid_count(p) = count of finite hp = same finite support as outlier_ind
  // Since outlier_ind is NaN iff hp is NaN, both share the same finite support.
  const LocalMeanResult outlier_lm = local_mean_and_count(outlier_ind, r);
  const Matrix2Df &h_valid_count = outlier_lm.count; // = |H_s_valid| per pixel

  // Step 5: assemble phi_artifact with spec-correct frac_out = outlier_count / |H_s_valid|.
  // |H_s_valid| = 0 → invalid; < 3 → 1.0f (insufficient support, no false veto, §2.3.2c).
  Matrix2Df out(img.rows(), img.cols());
  out.setConstant(nan_value());
  const float *mean_out_data = outlier_lm.mean.data();
  const float *hvc_data = h_valid_count.data();
  float *out_data = out.data();
  const float safe_frac_max = std::max(frac_artifact_max, global_eps_scale);
#pragma omp simd
  for (std::ptrdiff_t i = 0; i < n; ++i) {
    const float hvc = hvc_data[i];
    if (hvc <= 0.0f) continue;                    // |H_s_valid| = 0 → invalid
    if (hvc < 3.0f) { out_data[i] = 1.0f; continue; } // < 3 → no false veto
    const float frac = finite(mean_out_data[i]) ? mean_out_data[i] : 0.0f;
    out_data[i] = 1.0f - std::clamp(frac / safe_frac_max, 0.0f, 1.0f);
  }
  return out;
}

Matrix2Df robust_zscore(const Matrix2Df &m) {
  Matrix2Df out(m.rows(), m.cols());
  out.setConstant(nan_value());
  const auto values = finite_values(m);
  const auto z = robust_zscore_eps_scale(values);
  if (values.empty())
    return out;
  const auto n = static_cast<std::ptrdiff_t>(m.size());
  const float *m_data = m.data();
  float *out_data = out.data();
  size_t zi = 0;
  for (std::ptrdiff_t i = 0; i < n; ++i)
    if (finite(m_data[i])) out_data[i] = z[zi++];
  return out;
}

void accumulate_upsampled_log_psi(
    const Matrix2Df &src, int out_w, int out_h, int factor,
    Matrix2Dd &log_sum, std::vector<uint8_t> &veto) {
  const int rows = static_cast<int>(src.rows());
  const int cols = static_cast<int>(src.cols());
#pragma omp parallel for collapse(2) schedule(static)
  for (int y = 0; y < out_h; ++y) {
    for (int x = 0; x < out_w; ++x) {
      float value = nan_value();
      if (factor <= 1 && rows == out_h && cols == out_w) {
        value = src(y, x);
      } else {
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
            const float wx =
                1.0f - std::abs(sx - static_cast<float>(x0 + i));
            const float wy =
                1.0f - std::abs(sy - static_cast<float>(y0 + j));
            const float weight =
                std::max(wx, 0.0f) * std::max(wy, 0.0f);
            const float sample = src(yy, xx);
            if (weight > 0.0f && finite(sample)) {
              num += weight * sample;
              den += weight;
            }
          }
        }
        if (den > 0.0)
          value = static_cast<float>(num / den);
      }

      const size_t idx =
          static_cast<size_t>(y) * static_cast<size_t>(out_w) +
          static_cast<size_t>(x);
      if (veto[idx] != 0u)
        continue;
      if (!finite(value) || value <= 0.0f) {
        veto[idx] = 1u;
      } else {
        log_sum(y, x) += std::log(static_cast<double>(value));
      }
    }
  }
}

Matrix2Df compute_psi(const Matrix2Df &sharp, const Matrix2Df &snr,
                      const Matrix2Df &artifact,
                      const config::AqmhPyramidConfig &cfg) {
  const Matrix2Df z_sharp = robust_zscore(sharp);
  const Matrix2Df z_snr = robust_zscore(snr);
  Matrix2Df out(sharp.rows(), sharp.cols());
  out.setConstant(nan_value());
#pragma omp parallel for collapse(2) schedule(static)
  for (int y = 0; y < out.rows(); ++y) {
    for (int x = 0; x < out.cols(); ++x) {
      if (!finite(z_sharp(y, x)) || !finite(z_snr(y, x)) ||
          !finite(artifact(y, x)))
        continue;
      const float score = cfg.score_scale *
          (cfg.w_sharp * z_sharp(y, x) + cfg.w_snr * z_snr(y, x));
      const float sigmoid = 1.0f / (1.0f + std::exp(-score));
      out(y, x) = std::clamp(sigmoid * artifact(y, x), 0.0f, 1.0f);
    }
  }
  return out;
}

} // namespace

Matrix2Df compute_aqmh_local_variance(const Matrix2Df &image, int radius) {
  return local_variance_linear(image, radius);
}

AqmhQualityMapResult compute_aqmh_quality_map(
    const Matrix2Df &frame, const std::vector<uint8_t> &canvas_mask,
    const std::vector<uint8_t> &frame_valid_mask,
    int canvas_mask_width, int canvas_mask_height,
    const config::AqmhPyramidConfig &cfg,
    core::AccelerationBackend backend,
    cv::cuda::Stream *stream) {
  const auto total_start = std::chrono::steady_clock::now();
  const auto elapsed_since = [](const auto &start) {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now() - start)
        .count();
  };
  AqmhQualityMapResult result;
  result.q_map = Matrix2Df::Zero(frame.rows(), frame.cols());
  if (frame.rows() <= 0 || frame.cols() <= 0) {
    result.diagnostics.timing_total_seconds = elapsed_since(total_start);
    return result;
  }

  const auto source_mask_start = std::chrono::steady_clock::now();
  const Matrix2Df masked =
      source_masked_frame(frame, canvas_mask, frame_valid_mask,
                          canvas_mask_width, canvas_mask_height);
  result.diagnostics.timing_source_mask_seconds +=
      elapsed_since(source_mask_start);
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

    const auto pyramid_prepare_start = std::chrono::steady_clock::now();
    const Matrix2Df img_s = downsample_valid_mean(masked, factor);
    const int radius = std::max(1, cfg.base_window_px);
    const Matrix2Df laplacian = masked_laplacian(img_s);
    result.diagnostics.timing_pyramid_prepare_seconds +=
        elapsed_since(pyramid_prepare_start);

    const auto sharpness_start = std::chrono::steady_clock::now();
    Matrix2Df sharp;
    if (backend == core::AccelerationBackend::cpu) {
      sharp = compute_aqmh_local_variance(laplacian, radius);
    } else if (accelerated_local_variance(laplacian, radius, backend, sharp, stream)) {
      result.diagnostics.acceleration_used = true;
    } else {
      // Per-scale CPU fallback: avoids discarding already-computed scales
      // by recursing into a full pyramid restart.
      sharp = compute_aqmh_local_variance(laplacian, radius);
      result.diagnostics.acceleration_fallback = true;
    }
    result.diagnostics.timing_sharpness_seconds +=
        elapsed_since(sharpness_start);

    const auto local_background_start = std::chrono::steady_clock::now();
    const LocalMeanResult local_img = local_mean_and_count(img_s, radius);
    result.diagnostics.timing_local_background_seconds +=
        elapsed_since(local_background_start);

    bool scene_dependent = false;
    const auto snr_start = std::chrono::steady_clock::now();
    const Matrix2Df snr = phi_snr(img_s, local_img.mean, local_img.count,
                                  radius, scene_dependent);
    result.diagnostics.timing_snr_seconds += elapsed_since(snr_start);

    const auto artifact_start = std::chrono::steady_clock::now();
    const Matrix2Df artifact = phi_artifact(
        img_s, local_img.mean, radius, cfg.k_artifact,
        cfg.frac_artifact_max);
    result.diagnostics.timing_artifact_seconds +=
        elapsed_since(artifact_start);
    result.diagnostics.scene_dependent_snr =
        result.diagnostics.scene_dependent_snr || scene_dependent;

    const auto summary_start = std::chrono::steady_clock::now();
    if (s == 0) {
      result.diagnostics.sharpness_p50 = finite_median(sharp);
      result.diagnostics.g_sharp_summary = result.diagnostics.sharpness_p50;
      result.diagnostics.g_snr_summary = finite_median(snr);
    }
    if (s == 1) {
      result.diagnostics.snr_p50 = finite_median(snr);
      result.diagnostics.g_snr_summary = result.diagnostics.snr_p50;
    }
    result.diagnostics.timing_summary_seconds +=
        elapsed_since(summary_start);

    const auto psi_accumulate_start = std::chrono::steady_clock::now();
    const Matrix2Df psi = compute_psi(sharp, snr, artifact, cfg);
    accumulate_upsampled_log_psi(
        psi, frame.cols(), frame.rows(), factor, log_sum, veto);
    result.diagnostics.timing_psi_accumulate_seconds +=
        elapsed_since(psi_accumulate_start);
    ++computed_scales;
  }

  if (computed_scales == 0) {
    result.diagnostics.timing_total_seconds = elapsed_since(total_start);
    return result;
  }

  const auto finalize_start = std::chrono::steady_clock::now();
  for (int y = 0; y < frame.rows(); ++y) {
    for (int x = 0; x < frame.cols(); ++x) {
      const size_t idx = static_cast<size_t>(y) *
                             static_cast<size_t>(frame.cols()) +
                         static_cast<size_t>(x);
      const bool frame_ok = frame_valid_mask.empty() ||
                            (idx < frame_valid_mask.size() && frame_valid_mask[idx] != 0u);
      if (!mask_valid(canvas_mask, canvas_mask_width, canvas_mask_height, x, y) ||
          !frame_ok) {
        result.q_map(y, x) = 0.0f;
        continue;
      }
      result.q_map(y, x) =
          veto[idx] != 0u
              ? 0.0f
              : static_cast<float>(
                    std::exp(log_sum(y, x) / computed_scales));
    }
  }

  result.diagnostics.g_summary_invalid =
      !finite(result.diagnostics.g_sharp_summary) ||
      !finite(result.diagnostics.g_snr_summary);
  result.diagnostics.timing_finalize_seconds += elapsed_since(finalize_start);
  result.diagnostics.timing_total_seconds = elapsed_since(total_start);

  return result;
}

} // namespace tile_compile::metrics
