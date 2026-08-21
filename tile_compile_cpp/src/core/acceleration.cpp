#include "tile_compile/core/acceleration.hpp"
#include "tile_compile/core/cfa_warp.hpp"

#include "tile_compile/core/utils.hpp"
#include "tile_compile/image/normalization.hpp"
#include "tile_compile/metrics/aqmh_quality_map_cache.hpp"
#include "tile_compile/reconstruction/aqmh_reconstruction_cuda.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <sstream>

#if __has_include(<opencv2/core/cuda.hpp>)
#include <opencv2/core/cuda.hpp>
#define TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS 1
#else
#define TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS 0
#endif

#if __has_include(<opencv2/cudawarping.hpp>)
#include <opencv2/cudawarping.hpp>
#define TILE_COMPILE_HAS_OPENCV_CUDA_WARPING 1
#else
#define TILE_COMPILE_HAS_OPENCV_CUDA_WARPING 0
#endif

#if __has_include(<opencv2/cudaarithm.hpp>)
#include <opencv2/cudaarithm.hpp>
#define TILE_COMPILE_HAS_OPENCV_CUDA_ARITHM 1
#else
#define TILE_COMPILE_HAS_OPENCV_CUDA_ARITHM 0
#endif

#if __has_include(<opencv2/cudafilters.hpp>)
#include <opencv2/cudafilters.hpp>
#define TILE_COMPILE_HAS_OPENCV_CUDA_FILTERS 1
#else
#define TILE_COMPILE_HAS_OPENCV_CUDA_FILTERS 0
#endif

#if __has_include(<opencv2/core/ocl.hpp>)
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>
#define TILE_COMPILE_HAS_OPENCV_OPENCL 1
#else
#define TILE_COMPILE_HAS_OPENCV_OPENCL 0
#endif

namespace tile_compile::core {

namespace {

/// @brief Implements opencv cuda headers available.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool opencv_cuda_headers_available(AccelerationPhase phase) {
#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS
  switch (phase) {
  case AccelerationPhase::prewarp:
    return TILE_COMPILE_HAS_OPENCV_CUDA_WARPING != 0;
  case AccelerationPhase::aqmh_maps:
    return TILE_COMPILE_HAS_OPENCV_CUDA_FILTERS != 0;
  case AccelerationPhase::aqmh_reconstruction:
    return false;
  case AccelerationPhase::tile_reconstruction:
    return TILE_COMPILE_HAS_OPENCV_CUDA_ARITHM != 0;
  case AccelerationPhase::stacking:
    return TILE_COMPILE_HAS_OPENCV_CUDA_ARITHM != 0;
  }
#endif
  (void)phase;
  return false;
}

int interpolation_flag_from_name(const std::string &name) {
  if (name == "nearest")
    return cv::INTER_NEAREST;
  if (name == "cubic")
    return cv::INTER_CUBIC;
  if (name == "lanczos4")
    return cv::INTER_LANCZOS4;
  return cv::INTER_LINEAR;
}

/// @brief Implements phase supports backend.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool phase_supports_backend(AccelerationPhase phase,
                            AccelerationBackend backend) {
  switch (backend) {
  case AccelerationBackend::cpu:
    return true;
  case AccelerationBackend::opencv_cuda:
    return phase == AccelerationPhase::prewarp ||
           phase == AccelerationPhase::aqmh_maps ||
           phase == AccelerationPhase::tile_reconstruction ||
           phase == AccelerationPhase::stacking;
  case AccelerationBackend::opencv_opencl:
    return phase == AccelerationPhase::prewarp ||
           phase == AccelerationPhase::tile_reconstruction ||
           phase == AccelerationPhase::stacking;
  case AccelerationBackend::cuda:
    return phase == AccelerationPhase::aqmh_reconstruction;
  }
  return false;
}

/// @brief Implements opencv cuda runtime available.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool opencv_cuda_runtime_available() {
#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS && \
    (TILE_COMPILE_HAS_OPENCV_CUDA_WARPING || TILE_COMPILE_HAS_OPENCV_CUDA_FILTERS)
  try {
    return cv::cuda::getCudaEnabledDeviceCount() > 0;
  } catch (...) {
    return false;
  }
#else
  return false;
#endif
}

/// @brief Implements opencv opencl runtime available.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool opencv_opencl_runtime_available() {
#if TILE_COMPILE_HAS_OPENCV_OPENCL
  try {
    if (!cv::ocl::haveOpenCL()) {
      return false;
    }
    cv::ocl::setUseOpenCL(true);
    return cv::ocl::useOpenCL();
  } catch (...) {
    return false;
  }
#else
  return false;
#endif
}

/// @brief Lists missing backend reason.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::string missing_backend_reason(AccelerationBackend backend,
                                   bool tile_compile_with_cuda,
                                   bool opencv_cuda_headers,
                                   bool opencv_cuda_runtime,
                                   bool opencv_opencl_headers,
                                   bool opencv_opencl_runtime) {
  switch (backend) {
  case AccelerationBackend::cpu:
    return {};
  case AccelerationBackend::opencv_cuda:
    if (!opencv_cuda_headers) {
      return "opencv_cuda_headers_unavailable";
    }
    if (!opencv_cuda_runtime) {
      return "opencv_cuda_runtime_unavailable";
    }
    return {};
  case AccelerationBackend::opencv_opencl:
    if (!opencv_opencl_headers) {
      return "opencv_opencl_headers_unavailable";
    }
    if (!opencv_opencl_runtime) {
      return "opencv_opencl_runtime_unavailable";
    }
    return {};
  case AccelerationBackend::cuda:
    if (!tile_compile_with_cuda) {
      return "cuda_backend_not_built";
    }
    return {};
  }
  return "unknown_backend";
}

/// @brief Implements unsupported phase reason.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::string unsupported_phase_reason(AccelerationBackend backend,
                                     AccelerationPhase phase) {
  return acceleration_backend_name(backend) + "_backend_not_implemented_for_" +
         acceleration_phase_name(phase);
}

/// @brief Implements safe frame bytes.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
size_t safe_frame_bytes(int rows, int cols, int channels) {
  const size_t r = static_cast<size_t>(std::max(0, rows));
  const size_t c = static_cast<size_t>(std::max(0, cols));
  const size_t ch = static_cast<size_t>(std::max(1, channels));
  return r * c * ch * sizeof(float);
}

/// @brief Implements warp is identity.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool warp_is_identity(const WarpMatrix &warp) {
  const float eps = 1.0e-6f;
  return std::fabs(warp(0, 0) - 1.0f) < eps &&
         std::fabs(warp(0, 1)) < eps && std::fabs(warp(1, 0)) < eps &&
         std::fabs(warp(1, 1) - 1.0f) < eps && std::fabs(warp(0, 2)) < eps &&
         std::fabs(warp(1, 2)) < eps;
}

/// @brief Implements warp matrix to cv.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
cv::Mat warp_matrix_to_cv(const WarpMatrix &warp) {
  cv::Mat warp_matrix(2, 3, CV_32F);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      warp_matrix.at<float>(i, j) = warp(i, j);
    }
  }
  return warp_matrix;
}

/// @brief Creates host finite mask.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
cv::Mat make_host_finite_mask(const Matrix2Df &matrix) {
  cv::Mat mask(static_cast<int>(matrix.rows()), static_cast<int>(matrix.cols()),
               CV_8U, cv::Scalar(0));
  for (int y = 0; y < mask.rows; ++y) {
    uchar *mask_row = mask.ptr<uchar>(y);
    for (int x = 0; x < mask.cols; ++x) {
      if (std::isfinite(matrix(y, x))) {
        mask_row[x] = 255;
      }
    }
  }
  return mask;
}

/// @brief Writes valid outputs from mask.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void write_valid_outputs_from_mask(const cv::Mat &mask,
                                   std::vector<uint8_t> *valid_mask_out,
                                   bool *has_data_out) {
  if (valid_mask_out == nullptr && has_data_out == nullptr) {
    return;
  }

  const int rows = std::max(0, mask.rows);
  const int cols = std::max(0, mask.cols);
  const size_t pixel_count =
      static_cast<size_t>(rows) * static_cast<size_t>(cols);
  if (valid_mask_out != nullptr) {
    valid_mask_out->assign(pixel_count, 0u);
  }

  bool has_data = false;
  for (int y = 0; y < rows; ++y) {
    const uchar *row = mask.ptr<uchar>(y);
    const size_t row_off = static_cast<size_t>(y) * static_cast<size_t>(cols);
    for (int x = 0; x < cols; ++x) {
      if (row[x] == 0) {
        continue;
      }
      has_data = true;
      if (valid_mask_out != nullptr) {
        (*valid_mask_out)[row_off + static_cast<size_t>(x)] = 1u;
      }
    }
  }

  if (has_data_out != nullptr) {
    *has_data_out = has_data;
  }
}

void invalidate_matrix_outside_support(Matrix2Df &matrix,
                                       const cv::Mat &support_mask) {
  if (matrix.rows() != support_mask.rows ||
      matrix.cols() != support_mask.cols || support_mask.type() != CV_8U) {
    return;
  }
  const float invalid = std::numeric_limits<float>::quiet_NaN();
  for (int y = 0; y < support_mask.rows; ++y) {
    const uchar *mask_row = support_mask.ptr<uchar>(y);
    for (int x = 0; x < support_mask.cols; ++x) {
      if (mask_row[x] == 0) {
        matrix(y, x) = invalid;
      }
    }
  }
}

/// @brief Builds warped support mask.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool build_warped_support_mask(const cv::Mat &warp_matrix, int src_rows,
                               int src_cols, cv::Size output_size,
                               cv::Mat &support_mask) {
  if (src_rows <= 0 || src_cols <= 0 || output_size.width <= 0 ||
      output_size.height <= 0) {
    support_mask.release();
    return false;
  }

  cv::Mat ones(src_rows, src_cols, CV_32F, cv::Scalar(1.0f));
  cv::Mat warped_support;
  cv::warpAffine(ones, warped_support, warp_matrix, output_size,
                 cv::INTER_NEAREST | cv::WARP_INVERSE_MAP,
                 cv::BORDER_CONSTANT, cv::Scalar(0.0f));
  cv::compare(warped_support, 0.5f, support_mask, cv::CMP_GT);
  return !support_mask.empty();
}

#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS && TILE_COMPILE_HAS_OPENCV_CUDA_WARPING
/// @brief Implements cuda warp affine impl.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool cuda_warp_affine_impl(const cv::Mat &src, const cv::Mat &warp_matrix,
                           cv::Size output_size, cv::Mat &dst,
                           int interpolation_flag, cv::cuda::Stream *stream) {
  try {
    cv::cuda::GpuMat d_src;
    cv::cuda::GpuMat d_dst;
    cv::cuda::Stream &cuda_stream =
        stream ? *stream : cv::cuda::Stream::Null();
    d_src.upload(src, cuda_stream);
    cv::cuda::warpAffine(d_src, d_dst, warp_matrix, output_size,
                         interpolation_flag | cv::WARP_INVERSE_MAP,
                         cv::BORDER_CONSTANT, cv::Scalar(0), cuda_stream);
    d_dst.download(dst, cuda_stream);
    cuda_stream.waitForCompletion();
    return !dst.empty();
  } catch (...) {
    return false;
  }
}

/// @brief Implements fused 3-channel (RGB) cuda warp affine with single stream synchronization.
bool cuda_warp_affine_rgb_impl(
    const cv::Mat &src_r, const cv::Mat &src_g, const cv::Mat &src_b,
    const cv::Mat &warp_matrix, cv::Size output_size,
    cv::Mat &dst_r, cv::Mat &dst_g, cv::Mat &dst_b,
    int interpolation_flag, cv::cuda::Stream *stream) {
  try {
    cv::cuda::GpuMat d_src_r, d_src_g, d_src_b;
    cv::cuda::GpuMat d_dst_r, d_dst_g, d_dst_b;
    cv::cuda::Stream &cuda_stream =
        stream ? *stream : cv::cuda::Stream::Null();
    d_src_r.upload(src_r, cuda_stream);
    d_src_g.upload(src_g, cuda_stream);
    d_src_b.upload(src_b, cuda_stream);
    cv::cuda::warpAffine(d_src_r, d_dst_r, warp_matrix, output_size,
                         interpolation_flag | cv::WARP_INVERSE_MAP,
                         cv::BORDER_CONSTANT, cv::Scalar(0), cuda_stream);
    cv::cuda::warpAffine(d_src_g, d_dst_g, warp_matrix, output_size,
                         interpolation_flag | cv::WARP_INVERSE_MAP,
                         cv::BORDER_CONSTANT, cv::Scalar(0), cuda_stream);
    cv::cuda::warpAffine(d_src_b, d_dst_b, warp_matrix, output_size,
                         interpolation_flag | cv::WARP_INVERSE_MAP,
                         cv::BORDER_CONSTANT, cv::Scalar(0), cuda_stream);
    d_dst_r.download(dst_r, cuda_stream);
    d_dst_g.download(dst_g, cuda_stream);
    d_dst_b.download(dst_b, cuda_stream);
    cuda_stream.waitForCompletion();
    return !dst_r.empty() && !dst_g.empty() && !dst_b.empty();
  } catch (...) {
    return false;
  }
}

/// @brief Implements cuda warp cfa mosaic.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool cuda_warp_cfa_mosaic(const Matrix2Df &mosaic, const WarpMatrix &warp,
                          int out_height, int out_width, Matrix2Df &out,
                          int interpolation_flag, cv::cuda::Stream *stream) {
  const int h = static_cast<int>(mosaic.rows());
  const int w = static_cast<int>(mosaic.cols());
  const auto dims = compute_cfa_warp_dims(h, w, out_height, out_width);
  auto sub = extract_cfa_subplanes(mosaic, dims);
  const auto warps = make_all_cfa_subplane_warps(warp);

  cv::Mat a_cv(dims.sub_h, dims.sub_w, CV_32F, sub.a.data());
  cv::Mat b_cv(dims.sub_h, dims.sub_w, CV_32F, sub.b.data());
  cv::Mat c_cv(dims.sub_h, dims.sub_w, CV_32F, sub.c.data());
  cv::Mat d_cv(dims.sub_h, dims.sub_w, CV_32F, sub.d.data());
  cv::Mat a_w, b_w, c_w, d_w;
  const cv::Size out_size(dims.out_w_sub, dims.out_h_sub);
  if (stream) {
    // Keep all four CFA planes in one stream and synchronize once. Calling the
    // single-plane wrapper here would force four upload/warp/download barriers.
    try {
      std::array<cv::cuda::GpuMat, 4> d_src;
      std::array<cv::cuda::GpuMat, 4> d_dst;
      const std::array<cv::Mat, 4> src = {a_cv, b_cv, c_cv, d_cv};
      std::array<cv::Mat *, 4> dst = {&a_w, &b_w, &c_w, &d_w};
      const std::array<cv::Mat, 4> warp_arr = {warps.a, warps.b, warps.c, warps.d};
      for (size_t i = 0; i < src.size(); ++i) {
        d_src[i].upload(src[i], *stream);
        cv::cuda::warpAffine(d_src[i], d_dst[i], warp_arr[i], out_size,
                             interpolation_flag | cv::WARP_INVERSE_MAP,
                             cv::BORDER_CONSTANT, cv::Scalar(0), *stream);
        d_dst[i].download(*dst[i], *stream);
      }
      stream->waitForCompletion();
    } catch (...) {
      return false;
    }
  } else if (!cuda_warp_affine_impl(a_cv, warps.a, out_size, a_w,
                                    interpolation_flag, nullptr) ||
             !cuda_warp_affine_impl(b_cv, warps.b, out_size, b_w,
                                    interpolation_flag, nullptr) ||
             !cuda_warp_affine_impl(c_cv, warps.c, out_size, c_w,
                                    interpolation_flag, nullptr) ||
             !cuda_warp_affine_impl(d_cv, warps.d, out_size, d_w,
                                    interpolation_flag, nullptr)) {
    return false;
  }

  out = reassemble_cfa_subplanes(a_w, b_w, c_w, d_w, dims);
  return out.size() > 0;
}
#endif

#if TILE_COMPILE_HAS_OPENCV_OPENCL
/// @brief Implements opencl warp affine impl locked.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool opencl_warp_affine_impl_locked(const cv::Mat &src,
                                    const cv::Mat &warp_matrix,
                                    cv::Size output_size, cv::Mat &dst,
                                    int interpolation_flag) {
  cv::UMat u_src;
  src.copyTo(u_src);
  cv::UMat u_dst;
  cv::warpAffine(u_src, u_dst, warp_matrix, output_size,
                 interpolation_flag | cv::WARP_INVERSE_MAP,
                 cv::BORDER_CONSTANT, cv::Scalar(0));

  cv::Mat host_dst;
  u_dst.copyTo(host_dst);
  dst = std::move(host_dst);
  return !dst.empty();
}

/// @brief Implements opencl warp affine impl.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool opencl_warp_affine_impl(const cv::Mat &src, const cv::Mat &warp_matrix,
                             cv::Size output_size, cv::Mat &dst,
                             int interpolation_flag) {
  try {
    return opencl_warp_affine_impl_locked(src, warp_matrix, output_size, dst,
                                          interpolation_flag);
  } catch (...) {
    return false;
  }
}

/// @brief Implements opencl warp cfa mosaic.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool opencl_warp_cfa_mosaic(const Matrix2Df &mosaic, const WarpMatrix &warp,
                             int out_height, int out_width, Matrix2Df &out,
                             int interpolation_flag) {
  const int h = static_cast<int>(mosaic.rows());
  const int w = static_cast<int>(mosaic.cols());
  const auto dims = compute_cfa_warp_dims(h, w, out_height, out_width);
  auto sub = extract_cfa_subplanes(mosaic, dims);
  const auto warps = make_all_cfa_subplane_warps(warp);

  cv::Mat a_cv(dims.sub_h, dims.sub_w, CV_32F, sub.a.data());
  cv::Mat b_cv(dims.sub_h, dims.sub_w, CV_32F, sub.b.data());
  cv::Mat c_cv(dims.sub_h, dims.sub_w, CV_32F, sub.c.data());
  cv::Mat d_cv(dims.sub_h, dims.sub_w, CV_32F, sub.d.data());
  cv::Mat a_w, b_w, c_w, d_w;
  const cv::Size out_size(dims.out_w_sub, dims.out_h_sub);
  try {
    if (!opencl_warp_affine_impl_locked(a_cv, warps.a, out_size, a_w,
                                        interpolation_flag) ||
        !opencl_warp_affine_impl_locked(b_cv, warps.b, out_size, b_w,
                                        interpolation_flag) ||
        !opencl_warp_affine_impl_locked(c_cv, warps.c, out_size, c_w,
                                        interpolation_flag) ||
        !opencl_warp_affine_impl_locked(d_cv, warps.d, out_size, d_w,
                                        interpolation_flag)) {
      return false;
    }
  } catch (...) {
    return false;
  }

  out = reassemble_cfa_subplanes(a_w, b_w, c_w, d_w, dims);
  return out.size() > 0;
}

cv::Mat compute_min_keep_host(const cv::Mat &valid_count_host, int rows,
                              int cols, float min_fraction) {
  cv::Mat min_keep_host(rows, cols, CV_32F);
  for (int y = 0; y < rows; ++y) {
    const float *count_row = valid_count_host.ptr<float>(y);
    float *min_keep_row = min_keep_host.ptr<float>(y);
    for (int x = 0; x < cols; ++x) {
      const int n_valid_here = static_cast<int>(std::lround(count_row[x]));
      min_keep_row[x] = static_cast<float>(
          std::max(1, static_cast<int>(std::ceil(min_fraction * n_valid_here))));
    }
  }
  return min_keep_host;
}

cv::UMat opencl_clip_update_keep_masks(
    const std::vector<cv::UMat> &gpu_mats,
    std::vector<cv::UMat> &keep_masks, const cv::UMat &lo, const cv::UMat &hi,
    const cv::UMat &active_mask, const cv::UMat &can_continue,
    const cv::UMat &min_keep, int rows, int cols) {
  std::vector<cv::UMat> new_keep_masks;
  new_keep_masks.reserve(keep_masks.size());
  cv::UMat new_count(rows, cols, CV_32F);
  new_count.setTo(cv::Scalar(0.0f));
  for (size_t i = 0; i < gpu_mats.size(); ++i) {
    cv::UMat ge_lo, le_hi, in_range;
    cv::compare(gpu_mats[i], lo, ge_lo, cv::CMP_GE);
    cv::compare(gpu_mats[i], hi, le_hi, cv::CMP_LE);
    cv::bitwise_and(ge_lo, le_hi, in_range);

    cv::UMat new_keep;
    cv::bitwise_and(keep_masks[i], in_range, new_keep);
    new_keep_masks.push_back(new_keep);

    cv::UMat new_keep_f32;
    new_keep.convertTo(new_keep_f32, CV_32F, 1.0 / 255.0);
    cv::add(new_count, new_keep_f32, new_count);
  }

  cv::UMat meets_min;
  cv::compare(new_count, min_keep, meets_min, cv::CMP_GE);
  cv::UMat update_mask;
  cv::bitwise_and(active_mask, can_continue, update_mask);
  cv::UMat apply_new_keep;
  cv::bitwise_and(update_mask, meets_min, apply_new_keep);
  cv::UMat next_active;
  apply_new_keep.copyTo(next_active);

  cv::UMat apply_new_keep_inv;
  cv::bitwise_not(apply_new_keep, apply_new_keep_inv);
  for (size_t i = 0; i < keep_masks.size(); ++i) {
    cv::UMat old_region, new_region;
    cv::bitwise_and(keep_masks[i], apply_new_keep_inv, old_region);
    cv::bitwise_and(new_keep_masks[i], apply_new_keep, new_region);
    cv::bitwise_or(old_region, new_region, keep_masks[i]);
  }
  return next_active;
}

/// @brief Implements opencl sigma clip weighted tile impl.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool opencl_sigma_clip_weighted_tile_impl(
    const std::vector<Matrix2Df> &tiles, const std::vector<float> &weights,
    float sigma_low, float sigma_high, int max_iters, float min_fraction,
    float eps_weight, reconstruction::WeightedTileResult &out) {
  out = reconstruction::WeightedTileResult{};
  const float invalid_sample = std::numeric_limits<float>::quiet_NaN();
  if (tiles.empty() || weights.empty() || tiles.size() != weights.size()) {
    return true;
  }

  std::vector<float> effective_weights(weights);
  double host_weight_sum = 0.0;
  for (float &w : effective_weights) {
    if (std::isfinite(w) && w > 0.0f) {
      host_weight_sum += static_cast<double>(w);
    } else {
      w = 0.0f;
    }
  }
  out.effective_weight_sum = static_cast<float>(host_weight_sum);
  if (!(host_weight_sum > static_cast<double>(eps_weight))) {
    out.fallback_used = true;
    std::fill(effective_weights.begin(), effective_weights.end(), 1.0f);
    out.effective_weight_sum = static_cast<float>(effective_weights.size());
  }

  std::vector<std::reference_wrapper<const Matrix2Df>> active_tiles;
  std::vector<float> active_weights;
  active_tiles.reserve(tiles.size());
  active_weights.reserve(weights.size());
  for (size_t i = 0; i < tiles.size(); ++i) {
    if (effective_weights[i] > 0.0f && tiles[i].size() > 0) {
      active_tiles.emplace_back(tiles[i]);
      active_weights.push_back(effective_weights[i]);
    }
  }
  if (active_tiles.empty()) {
    out.tile = Matrix2Df();
    return true;
  }

  const int rows = static_cast<int>(active_tiles[0].get().rows());
  const int cols = static_cast<int>(active_tiles[0].get().cols());
  if (rows <= 0 || cols <= 0) {
    out.tile = Matrix2Df();
    return true;
  }

  try {
    std::vector<cv::UMat> gpu_tiles;
    std::vector<cv::UMat> keep_masks;
    std::vector<cv::UMat> valid_masks;
    gpu_tiles.reserve(active_tiles.size());
    keep_masks.reserve(active_tiles.size());
    valid_masks.reserve(active_tiles.size());

    cv::UMat zeros(rows, cols, CV_32F);
    cv::UMat eps(rows, cols, CV_32F);
    cv::UMat valid_count(rows, cols, CV_32F);
    cv::Mat valid_count_host(rows, cols, CV_32F);
    cv::Mat min_keep_host(rows, cols, CV_32F);
    zeros.setTo(cv::Scalar(0.0f));
    eps.setTo(cv::Scalar(1.0e-6f));
    valid_count.setTo(cv::Scalar(0.0f));

    for (const Matrix2Df &tile : active_tiles) {
      cv::Mat host_view(rows, cols, CV_32F, const_cast<float *>(tile.data()));
      cv::UMat gpu_tile;
      host_view.copyTo(gpu_tile);
      gpu_tiles.push_back(gpu_tile);

      cv::Mat valid_mask_host = make_host_finite_mask(tile);
      cv::UMat valid_mask;
      valid_mask_host.copyTo(valid_mask);
      valid_masks.push_back(valid_mask.clone());
      keep_masks.push_back(valid_mask.clone());

      cv::UMat valid_mask_f32;
      valid_mask.convertTo(valid_mask_f32, CV_32F, 1.0 / 255.0);
      cv::add(valid_count, valid_mask_f32, valid_count);
    }

    valid_count.copyTo(valid_count_host);
    min_keep_host = compute_min_keep_host(valid_count_host, rows, cols, min_fraction);

    const bool enable_clipping =
        static_cast<int>(gpu_tiles.size()) > 2 && max_iters > 0;
    // OpenCV UMat operations are thread-safe - no global mutex needed
    cv::UMat min_keep;
    min_keep_host.copyTo(min_keep);

    cv::UMat active_mask;
    cv::compare(valid_count, 0.0f, active_mask, cv::CMP_GT);

    if (enable_clipping) {
      for (int iter = 0; iter < max_iters; ++iter) {
        cv::UMat wsum(rows, cols, CV_32F);
        cv::UMat wsum2(rows, cols, CV_32F);
        cv::UMat wmean_num(rows, cols, CV_32F);
        wsum.setTo(cv::Scalar(0.0f));
        wsum2.setTo(cv::Scalar(0.0f));
        wmean_num.setTo(cv::Scalar(0.0f));

        for (size_t i = 0; i < gpu_tiles.size(); ++i) {
          cv::UMat keep_f32;
          keep_masks[i].convertTo(keep_f32, CV_32F, 1.0 / 255.0);

          cv::UMat weighted_keep;
          keep_f32.convertTo(weighted_keep, CV_32F, active_weights[i]);
          cv::add(wsum, weighted_keep, wsum);

          cv::UMat weighted_value;
          cv::multiply(gpu_tiles[i], weighted_keep, weighted_value);
          cv::add(wmean_num, weighted_value, wmean_num);

          cv::UMat weighted_keep_sq;
          keep_f32.convertTo(weighted_keep_sq, CV_32F,
                             active_weights[i] * active_weights[i]);
          cv::add(wsum2, weighted_keep_sq, wsum2);
        }

        cv::UMat wsum_safe;
        cv::max(wsum, eps, wsum_safe);
        cv::UMat mean;
        cv::divide(wmean_num, wsum_safe, mean);

        cv::UMat var_num(rows, cols, CV_32F);
        var_num.setTo(cv::Scalar(0.0f));
        for (size_t i = 0; i < gpu_tiles.size(); ++i) {
          cv::UMat diff;
          cv::subtract(gpu_tiles[i], mean, diff);
          cv::UMat diff_sq;
          cv::multiply(diff, diff, diff_sq);

          cv::UMat keep_f32;
          keep_masks[i].convertTo(keep_f32, CV_32F, 1.0 / 255.0);
          cv::UMat weighted_keep;
          keep_f32.convertTo(weighted_keep, CV_32F, active_weights[i]);
          cv::UMat weighted_diff;
          cv::multiply(diff_sq, weighted_keep, weighted_diff);
          cv::add(var_num, weighted_diff, var_num);
        }

        cv::UMat wsum2_over_wsum;
        cv::divide(wsum2, wsum_safe, wsum2_over_wsum);
        cv::UMat denom;
        cv::subtract(wsum, wsum2_over_wsum, denom);
        cv::UMat denom_safe;
        cv::max(denom, eps, denom_safe);
        cv::UMat var;
        cv::divide(var_num, denom_safe, var);
        cv::max(var, zeros, var);

        cv::UMat wsum_sq;
        cv::multiply(wsum, wsum, wsum_sq);
        cv::UMat wsum2_safe;
        cv::max(wsum2, eps, wsum2_safe);
        cv::UMat n_eff;
        cv::divide(wsum_sq, wsum2_safe, n_eff);
        cv::UMat neff_mask;
        cv::compare(n_eff, 2.0f + 1.0e-6f, neff_mask, cv::CMP_GT);
        cv::UMat denom_positive_mask;
        cv::compare(denom, 1.0e-12f, denom_positive_mask, cv::CMP_GT);
        cv::UMat sd;
        cv::sqrt(var, sd);
        cv::UMat sd_positive_mask;
        cv::compare(sd, 0.0f, sd_positive_mask, cv::CMP_GT);
        cv::UMat can_continue;
        cv::bitwise_and(neff_mask, denom_positive_mask, can_continue);
        cv::bitwise_and(can_continue, sd_positive_mask, can_continue);

        cv::UMat sigma_low_sd;
        cv::UMat sigma_high_sd;
        sd.convertTo(sigma_low_sd, CV_32F, sigma_low);
        sd.convertTo(sigma_high_sd, CV_32F, sigma_high);
        cv::UMat lo;
        cv::UMat hi;
        cv::subtract(mean, sigma_low_sd, lo);
        cv::add(mean, sigma_high_sd, hi);

        active_mask = opencl_clip_update_keep_masks(
            gpu_tiles, keep_masks, lo, hi, active_mask, can_continue,
            min_keep, rows, cols);
      }
    }

    cv::UMat final_wsum(rows, cols, CV_32F);
    cv::UMat final_num(rows, cols, CV_32F);
    cv::UMat fallback_wsum(rows, cols, CV_32F);
    cv::UMat fallback_num(rows, cols, CV_32F);
    final_wsum.setTo(cv::Scalar(0.0f));
    final_num.setTo(cv::Scalar(0.0f));
    fallback_wsum.setTo(cv::Scalar(0.0f));
    fallback_num.setTo(cv::Scalar(0.0f));

    for (size_t i = 0; i < gpu_tiles.size(); ++i) {
      cv::UMat keep_f32;
      keep_masks[i].convertTo(keep_f32, CV_32F, 1.0 / 255.0);
      cv::UMat valid_f32;
      valid_masks[i].convertTo(valid_f32, CV_32F, 1.0 / 255.0);

      cv::UMat weighted_keep;
      keep_f32.convertTo(weighted_keep, CV_32F, active_weights[i]);
      cv::UMat weighted_valid;
      valid_f32.convertTo(weighted_valid, CV_32F, active_weights[i]);
      cv::add(final_wsum, weighted_keep, final_wsum);
      cv::add(fallback_wsum, weighted_valid, fallback_wsum);

      cv::UMat weighted_value;
      cv::multiply(gpu_tiles[i], weighted_keep, weighted_value);
      cv::UMat fallback_value;
      cv::multiply(gpu_tiles[i], weighted_valid, fallback_value);
      cv::add(final_num, weighted_value, final_num);
      cv::add(fallback_num, fallback_value, fallback_num);
    }

    cv::UMat final_wsum_safe;
    cv::UMat fallback_wsum_safe;
    cv::max(final_wsum, eps, final_wsum_safe);
    cv::max(fallback_wsum, eps, fallback_wsum_safe);
    cv::UMat final_out;
    cv::UMat fallback_out;
    cv::divide(final_num, final_wsum_safe, final_out);
    cv::divide(fallback_num, fallback_wsum_safe, fallback_out);

    cv::UMat zero_wsum_mask;
    cv::compare(final_wsum, eps_weight, zero_wsum_mask, cv::CMP_LE);
    fallback_out.copyTo(final_out, zero_wsum_mask);
    cv::UMat zero_fallback_mask;
    cv::compare(fallback_wsum, eps_weight, zero_fallback_mask, cv::CMP_LE);
    final_out.setTo(cv::Scalar(invalid_sample), zero_fallback_mask);

    out.tile.resize(rows, cols);
    cv::Mat out_host(rows, cols, CV_32F, out.tile.data());
    final_out.copyTo(out_host);
    return true;
  } catch (...) {
    return false;
  }
}

/// @brief Implements opencl sigma clip stack impl.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool opencl_sigma_clip_stack_impl(
    const std::vector<Matrix2Df> &frames, float sigma_low, float sigma_high,
    int max_iters, float min_fraction, Matrix2Df &out) {
  const float invalid_sample = std::numeric_limits<float>::quiet_NaN();
  std::vector<std::reference_wrapper<const Matrix2Df>> valid;
  valid.reserve(frames.size());
  for (const auto &frame : frames) {
    if (frame.size() > 0) {
      valid.emplace_back(frame);
    }
  }
  if (valid.empty()) {
    out.resize(0, 0);
    return true;
  }

  const int rows = static_cast<int>(valid[0].get().rows());
  const int cols = static_cast<int>(valid[0].get().cols());
  if (rows <= 0 || cols <= 0) {
    out.resize(0, 0);
    return true;
  }

  try {
    std::vector<cv::UMat> gpu_frames;
    std::vector<cv::UMat> keep_masks;
    gpu_frames.reserve(valid.size());
    keep_masks.reserve(valid.size());

    cv::UMat ones(rows, cols, CV_32F);
    cv::UMat zeros(rows, cols, CV_32F);
    cv::UMat initial_count(rows, cols, CV_32F);
    cv::Mat initial_count_host(rows, cols, CV_32F);
    cv::Mat min_keep_host(rows, cols, CV_32F);
    // OpenCV UMat operations are thread-safe - no global mutex needed
    ones.setTo(cv::Scalar(1.0f));
    zeros.setTo(cv::Scalar(0.0f));
    initial_count.setTo(cv::Scalar(0.0f));

    for (const Matrix2Df &frame : valid) {
      cv::Mat host_view(rows, cols, CV_32F, const_cast<float *>(frame.data()));
      cv::UMat gpu_frame;
      host_view.copyTo(gpu_frame);
      gpu_frames.push_back(gpu_frame);

      cv::Mat valid_mask_host = make_host_finite_mask(frame);
      cv::UMat valid_mask;
      valid_mask_host.copyTo(valid_mask);
      keep_masks.push_back(valid_mask.clone());

      cv::UMat valid_mask_f32;
      valid_mask.convertTo(valid_mask_f32, CV_32F, 1.0 / 255.0);
      cv::add(initial_count, valid_mask_f32, initial_count);
    }

    initial_count.copyTo(initial_count_host);
    min_keep_host = compute_min_keep_host(initial_count_host, rows, cols, min_fraction);

    // OpenCV UMat operations are thread-safe - no global mutex needed
    cv::UMat min_keep;
    min_keep_host.copyTo(min_keep);

    cv::UMat active_mask;
    cv::compare(initial_count, 0.0f, active_mask, cv::CMP_GT);

    for (int iter = 0; iter < max_iters; ++iter) {
      cv::UMat sum(rows, cols, CV_32F);
      cv::UMat sumsq(rows, cols, CV_32F);
      cv::UMat count(rows, cols, CV_32F);
      sum.setTo(cv::Scalar(0.0f));
      sumsq.setTo(cv::Scalar(0.0f));
      count.setTo(cv::Scalar(0.0f));

      for (size_t i = 0; i < gpu_frames.size(); ++i) {
        cv::UMat keep_f32;
        keep_masks[i].convertTo(keep_f32, CV_32F, 1.0 / 255.0);

        cv::UMat masked_frame;
        cv::multiply(gpu_frames[i], keep_f32, masked_frame);
        cv::add(sum, masked_frame, sum);

        cv::UMat frame_sq;
        cv::multiply(gpu_frames[i], gpu_frames[i], frame_sq);
        cv::UMat masked_sq;
        cv::multiply(frame_sq, keep_f32, masked_sq);
        cv::add(sumsq, masked_sq, sumsq);
        cv::add(count, keep_f32, count);
      }

      cv::UMat count_denom;
      cv::max(count, ones, count_denom);
      cv::UMat mean;
      cv::divide(sum, count_denom, mean);

      cv::UMat sumsq_mean;
      cv::divide(sumsq, count_denom, sumsq_mean);
      cv::UMat mean_sq;
      cv::multiply(mean, mean, mean_sq);
      cv::UMat var;
      cv::subtract(sumsq_mean, mean_sq, var);
      cv::max(var, zeros, var);

      cv::UMat kept_gt_one_mask;
      cv::compare(count, 1.0f, kept_gt_one_mask, cv::CMP_GT);
      cv::UMat kept_gt_one_f32;
      kept_gt_one_mask.convertTo(kept_gt_one_f32, CV_32F, 1.0 / 255.0);

      cv::UMat count_minus_one;
      cv::subtract(count, ones, count_minus_one);
      cv::max(count_minus_one, ones, count_minus_one);
      cv::UMat factor;
      cv::divide(count, count_minus_one, factor);

      cv::UMat one_minus_gt_one;
      cv::subtract(ones, kept_gt_one_f32, one_minus_gt_one);
      cv::UMat factor_when_gt_one;
      cv::multiply(factor, kept_gt_one_f32, factor_when_gt_one);
      cv::add(factor_when_gt_one, one_minus_gt_one, factor);
      cv::multiply(var, factor, var);
      cv::max(var, zeros, var);

      cv::UMat sd;
      cv::sqrt(var, sd);
      cv::UMat sd_positive_mask;
      cv::compare(sd, 0.0f, sd_positive_mask, cv::CMP_GT);
      cv::UMat can_continue;
      cv::bitwise_and(kept_gt_one_mask, sd_positive_mask, can_continue);

      cv::UMat sigma_low_sd;
      cv::UMat sigma_high_sd;
      sd.convertTo(sigma_low_sd, CV_32F, sigma_low);
      sd.convertTo(sigma_high_sd, CV_32F, sigma_high);
      cv::UMat lo;
      cv::UMat hi;
      cv::subtract(mean, sigma_low_sd, lo);
      cv::add(mean, sigma_high_sd, hi);

      active_mask = opencl_clip_update_keep_masks(
          gpu_frames, keep_masks, lo, hi, active_mask, can_continue,
          min_keep, rows, cols);
    }

    cv::UMat final_sum(rows, cols, CV_32F);
    cv::UMat final_count(rows, cols, CV_32F);
    final_sum.setTo(cv::Scalar(0.0f));
    final_count.setTo(cv::Scalar(0.0f));
    for (size_t i = 0; i < gpu_frames.size(); ++i) {
      cv::UMat keep_f32;
      keep_masks[i].convertTo(keep_f32, CV_32F, 1.0 / 255.0);
      cv::UMat masked_frame;
      cv::multiply(gpu_frames[i], keep_f32, masked_frame);
      cv::add(final_sum, masked_frame, final_sum);
      cv::add(final_count, keep_f32, final_count);
    }

    cv::UMat final_count_denom;
    cv::max(final_count, ones, final_count_denom);
    cv::UMat final_out;
    cv::divide(final_sum, final_count_denom, final_out);
    cv::UMat dead_mask;
    cv::compare(final_count, 0.0f, dead_mask, cv::CMP_LE);
    final_out.setTo(cv::Scalar(invalid_sample), dead_mask);

    out.resize(rows, cols);
    cv::Mat out_host(rows, cols, CV_32F, out.data());
    final_out.copyTo(out_host);
    return true;
  } catch (...) {
    return false;
  }
}
#endif

#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS && TILE_COMPILE_HAS_OPENCV_CUDA_ARITHM
/// @brief Implements cuda sigma clip weighted tile impl.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool cuda_sigma_clip_weighted_tile_impl(
    const std::vector<Matrix2Df> &tiles, const std::vector<float> &weights,
    float sigma_low, float sigma_high, int max_iters, float min_fraction,
    float eps_weight, reconstruction::WeightedTileResult &out,
    cv::cuda::Stream *stream) {
  out = reconstruction::WeightedTileResult{};
  const float invalid_sample = std::numeric_limits<float>::quiet_NaN();
  if (tiles.empty() || weights.empty() || tiles.size() != weights.size()) {
    return true;
  }

  std::vector<float> effective_weights(weights);
  double host_weight_sum = 0.0;
  for (float &w : effective_weights) {
    if (std::isfinite(w) && w > 0.0f) {
      host_weight_sum += static_cast<double>(w);
    } else {
      w = 0.0f;
    }
  }
  out.effective_weight_sum = static_cast<float>(host_weight_sum);
  if (!(host_weight_sum > static_cast<double>(eps_weight))) {
    out.fallback_used = true;
    std::fill(effective_weights.begin(), effective_weights.end(), 1.0f);
    out.effective_weight_sum = static_cast<float>(effective_weights.size());
  }

  std::vector<std::reference_wrapper<const Matrix2Df>> active_tiles;
  std::vector<float> active_weights;
  active_tiles.reserve(tiles.size());
  active_weights.reserve(weights.size());
  for (size_t i = 0; i < tiles.size(); ++i) {
    if (effective_weights[i] > 0.0f && tiles[i].size() > 0) {
      active_tiles.emplace_back(tiles[i]);
      active_weights.push_back(effective_weights[i]);
    }
  }
  if (active_tiles.empty()) {
    out.tile = Matrix2Df();
    return true;
  }

  const int rows = static_cast<int>(active_tiles[0].get().rows());
  const int cols = static_cast<int>(active_tiles[0].get().cols());
  if (rows <= 0 || cols <= 0) {
    out.tile = Matrix2Df();
    return true;
  }

  try {
    cv::cuda::Stream& s = stream ? *stream : cv::cuda::Stream::Null();
    std::vector<cv::cuda::GpuMat> gpu_tiles;
    std::vector<cv::cuda::GpuMat> keep_masks;
    std::vector<cv::cuda::GpuMat> valid_masks;
    gpu_tiles.reserve(active_tiles.size());
    keep_masks.reserve(active_tiles.size());
    valid_masks.reserve(active_tiles.size());

    cv::cuda::GpuMat ones(rows, cols, CV_32F);
    cv::cuda::GpuMat zeros(rows, cols, CV_32F);
    cv::cuda::GpuMat eps(rows, cols, CV_32F);
    cv::cuda::GpuMat valid_count(rows, cols, CV_32F);
    ones.setTo(cv::Scalar(1.0f), s);
    zeros.setTo(cv::Scalar(0.0f), s);
    eps.setTo(cv::Scalar(1.0e-6f), s);
    valid_count.setTo(cv::Scalar(0.0f), s);

    for (const Matrix2Df &tile : active_tiles) {
      cv::Mat host_view(rows, cols, CV_32F,
                        const_cast<float *>(tile.data()));
      cv::cuda::GpuMat gpu_tile;
      gpu_tile.upload(host_view, s);
      gpu_tiles.push_back(gpu_tile);

      cv::Mat valid_mask_host = make_host_finite_mask(tile);
      cv::cuda::GpuMat valid_mask;
      valid_mask.upload(valid_mask_host, s);
      valid_masks.push_back(valid_mask.clone());
      keep_masks.push_back(valid_mask.clone());

      cv::cuda::GpuMat valid_mask_f32;
      valid_mask.convertTo(valid_mask_f32, CV_32F, 1.0 / 255.0, s);
      cv::cuda::add(valid_count, valid_mask_f32, valid_count, cv::noArray(), -1, s);
    }

    cv::Mat valid_count_host(rows, cols, CV_32F);
    cv::Mat min_keep_host(rows, cols, CV_32F);
    valid_count.download(valid_count_host, s);
    s.waitForCompletion();
    for (int y = 0; y < rows; ++y) {
      const float *count_row = valid_count_host.ptr<float>(y);
      float *min_keep_row = min_keep_host.ptr<float>(y);
      for (int x = 0; x < cols; ++x) {
        const int n_valid_here = static_cast<int>(std::lround(count_row[x]));
        min_keep_row[x] = static_cast<float>(
            std::max(1, static_cast<int>(std::ceil(min_fraction * n_valid_here))));
      }
    }

    cv::cuda::GpuMat min_keep;
    min_keep.upload(min_keep_host, s);

    const bool enable_clipping =
        static_cast<int>(gpu_tiles.size()) > 2 && max_iters > 0;
    cv::cuda::GpuMat active_mask;
    cv::cuda::compare(valid_count, 0.0f, active_mask, cv::CMP_GT, s);

    if (enable_clipping) {
      for (int iter = 0; iter < max_iters; ++iter) {
        cv::cuda::GpuMat wsum(rows, cols, CV_32F);
        cv::cuda::GpuMat wsum2(rows, cols, CV_32F);
        cv::cuda::GpuMat wmean_num(rows, cols, CV_32F);
        wsum.setTo(cv::Scalar(0.0f), s);
        wsum2.setTo(cv::Scalar(0.0f), s);
        wmean_num.setTo(cv::Scalar(0.0f), s);

        for (size_t i = 0; i < gpu_tiles.size(); ++i) {
          cv::cuda::GpuMat keep_f32;
          keep_masks[i].convertTo(keep_f32, CV_32F, 1.0 / 255.0, s);

          cv::cuda::GpuMat weighted_keep;
          cv::cuda::multiply(keep_f32, cv::Scalar(active_weights[i]),
                             weighted_keep, 1, -1, s);
          cv::cuda::add(wsum, weighted_keep, wsum, cv::noArray(), -1, s);

          cv::cuda::GpuMat weighted_value;
          cv::cuda::multiply(gpu_tiles[i], weighted_keep, weighted_value, 1, -1, s);
          cv::cuda::add(wmean_num, weighted_value, wmean_num, cv::noArray(), -1, s);

          cv::cuda::GpuMat weighted_keep_sq;
          cv::cuda::multiply(keep_f32,
                             cv::Scalar(active_weights[i] * active_weights[i]),
                             weighted_keep_sq, 1, -1, s);
          cv::cuda::add(wsum2, weighted_keep_sq, wsum2, cv::noArray(), -1, s);
        }

        cv::cuda::GpuMat wsum_safe;
        cv::cuda::max(wsum, eps, wsum_safe, s);
        cv::cuda::GpuMat mean;
        cv::cuda::divide(wmean_num, wsum_safe, mean, 1, -1, s);

        cv::cuda::GpuMat var_num(rows, cols, CV_32F);
        var_num.setTo(cv::Scalar(0.0f), s);
        for (size_t i = 0; i < gpu_tiles.size(); ++i) {
          cv::cuda::GpuMat diff;
          cv::cuda::subtract(gpu_tiles[i], mean, diff, cv::noArray(), -1, s);
          cv::cuda::GpuMat diff_sq;
          cv::cuda::multiply(diff, diff, diff_sq, 1, -1, s);

          cv::cuda::GpuMat keep_f32;
          keep_masks[i].convertTo(keep_f32, CV_32F, 1.0 / 255.0, s);
          cv::cuda::GpuMat weighted_keep;
          cv::cuda::multiply(keep_f32, cv::Scalar(active_weights[i]),
                             weighted_keep, 1, -1, s);
          cv::cuda::GpuMat weighted_diff;
          cv::cuda::multiply(diff_sq, weighted_keep, weighted_diff, 1, -1, s);
          cv::cuda::add(var_num, weighted_diff, var_num, cv::noArray(), -1, s);
        }

        cv::cuda::GpuMat wsum2_over_wsum;
        cv::cuda::divide(wsum2, wsum_safe, wsum2_over_wsum, 1, -1, s);
        cv::cuda::GpuMat denom;
        cv::cuda::subtract(wsum, wsum2_over_wsum, denom, cv::noArray(), -1, s);
        cv::cuda::GpuMat denom_safe;
        cv::cuda::max(denom, eps, denom_safe, s);
        cv::cuda::GpuMat var;
        cv::cuda::divide(var_num, denom_safe, var, 1, -1, s);
        cv::cuda::max(var, zeros, var, s);

        cv::cuda::GpuMat wsum_sq;
        cv::cuda::multiply(wsum, wsum, wsum_sq, 1, -1, s);
        cv::cuda::GpuMat wsum2_safe;
        cv::cuda::max(wsum2, eps, wsum2_safe, s);
        cv::cuda::GpuMat n_eff;
        cv::cuda::divide(wsum_sq, wsum2_safe, n_eff, 1, -1, s);
        cv::cuda::GpuMat neff_mask;
        cv::cuda::compare(n_eff, 2.0f + 1.0e-6f, neff_mask, cv::CMP_GT, s);
        cv::cuda::GpuMat denom_positive_mask;
        cv::cuda::compare(denom, 1.0e-12f, denom_positive_mask, cv::CMP_GT, s);
        cv::cuda::GpuMat sd;
        cv::cuda::sqrt(var, sd, s);
        cv::cuda::GpuMat sd_positive_mask;
        cv::cuda::compare(sd, 0.0f, sd_positive_mask, cv::CMP_GT, s);
        cv::cuda::GpuMat can_continue;
        cv::cuda::bitwise_and(neff_mask, denom_positive_mask, can_continue, cv::noArray(), s);
        cv::cuda::bitwise_and(can_continue, sd_positive_mask, can_continue, cv::noArray(), s);

        cv::cuda::GpuMat sigma_low_sd;
        cv::cuda::GpuMat sigma_high_sd;
        cv::cuda::multiply(sd, cv::Scalar(sigma_low), sigma_low_sd, 1, -1, s);
        cv::cuda::multiply(sd, cv::Scalar(sigma_high), sigma_high_sd, 1, -1, s);
        cv::cuda::GpuMat lo;
        cv::cuda::GpuMat hi;
        cv::cuda::subtract(mean, sigma_low_sd, lo, cv::noArray(), -1, s);
        cv::cuda::add(mean, sigma_high_sd, hi, cv::noArray(), -1, s);

        std::vector<cv::cuda::GpuMat> new_keep_masks;
        new_keep_masks.reserve(keep_masks.size());
        cv::cuda::GpuMat new_valid_count(rows, cols, CV_32F);
        new_valid_count.setTo(cv::Scalar(0.0f), s);
        for (size_t i = 0; i < gpu_tiles.size(); ++i) {
          cv::cuda::GpuMat ge_lo;
          cv::cuda::GpuMat le_hi;
          cv::cuda::GpuMat in_range;
          cv::cuda::compare(gpu_tiles[i], lo, ge_lo, cv::CMP_GE, s);
          cv::cuda::compare(gpu_tiles[i], hi, le_hi, cv::CMP_LE, s);
          cv::cuda::bitwise_and(ge_lo, le_hi, in_range, cv::noArray(), s);

          cv::cuda::GpuMat new_keep;
          cv::cuda::bitwise_and(keep_masks[i], in_range, new_keep, cv::noArray(), s);
          new_keep_masks.push_back(new_keep);

          cv::cuda::GpuMat new_keep_f32;
          new_keep.convertTo(new_keep_f32, CV_32F, 1.0 / 255.0, s);
          cv::cuda::add(new_valid_count, new_keep_f32, new_valid_count, cv::noArray(), -1, s);
        }

        cv::cuda::GpuMat meets_min;
        cv::cuda::compare(new_valid_count, min_keep, meets_min, cv::CMP_GE, s);
        cv::cuda::GpuMat update_mask;
        cv::cuda::bitwise_and(active_mask, can_continue, update_mask, cv::noArray(), s);
        cv::cuda::GpuMat apply_new_keep;
        cv::cuda::bitwise_and(update_mask, meets_min, apply_new_keep, cv::noArray(), s);
        cv::cuda::GpuMat next_active;
        apply_new_keep.copyTo(next_active, s);

        cv::cuda::GpuMat apply_new_keep_inv;
        cv::cuda::bitwise_not(apply_new_keep, apply_new_keep_inv, cv::noArray(), s);
        for (size_t i = 0; i < keep_masks.size(); ++i) {
          cv::cuda::GpuMat old_region;
          cv::cuda::GpuMat new_region;
          cv::cuda::bitwise_and(keep_masks[i], apply_new_keep_inv, old_region, cv::noArray(), s);
          cv::cuda::bitwise_and(new_keep_masks[i], apply_new_keep, new_region, cv::noArray(), s);
          cv::cuda::bitwise_or(old_region, new_region, keep_masks[i], cv::noArray(), s);
        }
        active_mask = next_active;
      }
    }

    cv::cuda::GpuMat final_wsum(rows, cols, CV_32F);
    cv::cuda::GpuMat final_num(rows, cols, CV_32F);
    cv::cuda::GpuMat fallback_wsum(rows, cols, CV_32F);
    cv::cuda::GpuMat fallback_num(rows, cols, CV_32F);
    final_wsum.setTo(cv::Scalar(0.0f), s);
    final_num.setTo(cv::Scalar(0.0f), s);
    fallback_wsum.setTo(cv::Scalar(0.0f), s);
    fallback_num.setTo(cv::Scalar(0.0f), s);

    for (size_t i = 0; i < gpu_tiles.size(); ++i) {
      cv::cuda::GpuMat keep_f32;
      keep_masks[i].convertTo(keep_f32, CV_32F, 1.0 / 255.0, s);
      cv::cuda::GpuMat valid_f32;
      valid_masks[i].convertTo(valid_f32, CV_32F, 1.0 / 255.0, s);

      cv::cuda::GpuMat weighted_keep;
      cv::cuda::multiply(keep_f32, cv::Scalar(active_weights[i]), weighted_keep, 1, -1, s);
      cv::cuda::GpuMat weighted_valid;
      cv::cuda::multiply(valid_f32, cv::Scalar(active_weights[i]), weighted_valid, 1, -1, s);
      cv::cuda::add(final_wsum, weighted_keep, final_wsum, cv::noArray(), -1, s);
      cv::cuda::add(fallback_wsum, weighted_valid, fallback_wsum, cv::noArray(), -1, s);

      cv::cuda::GpuMat weighted_value;
      cv::cuda::multiply(gpu_tiles[i], weighted_keep, weighted_value, 1, -1, s);
      cv::cuda::GpuMat fallback_value;
      cv::cuda::multiply(gpu_tiles[i], weighted_valid, fallback_value, 1, -1, s);
      cv::cuda::add(final_num, weighted_value, final_num, cv::noArray(), -1, s);
      cv::cuda::add(fallback_num, fallback_value, fallback_num, cv::noArray(), -1, s);
    }

    cv::cuda::GpuMat final_wsum_safe;
    cv::cuda::GpuMat fallback_wsum_safe;
    cv::cuda::max(final_wsum, eps, final_wsum_safe, s);
    cv::cuda::max(fallback_wsum, eps, fallback_wsum_safe, s);
    cv::cuda::GpuMat final_out;
    cv::cuda::GpuMat fallback_out;
    cv::cuda::divide(final_num, final_wsum_safe, final_out, 1, -1, s);
    cv::cuda::divide(fallback_num, fallback_wsum_safe, fallback_out, 1, -1, s);

    cv::cuda::GpuMat zero_wsum_mask;
    cv::cuda::compare(final_wsum, eps_weight, zero_wsum_mask, cv::CMP_LE, s);
    fallback_out.copyTo(final_out, zero_wsum_mask, s);
    cv::cuda::GpuMat zero_fallback_mask;
    cv::cuda::compare(fallback_wsum, eps_weight, zero_fallback_mask, cv::CMP_LE, s);
    final_out.setTo(cv::Scalar(invalid_sample), zero_fallback_mask, s);

    out.tile.resize(rows, cols);
    cv::Mat out_host(rows, cols, CV_32F, out.tile.data());
    final_out.download(out_host, s);
    s.waitForCompletion();
    return true;
  } catch (...) {
    return false;
  }
}

/// @brief Implements cuda sigma clip stack impl.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool cuda_sigma_clip_stack_impl(
    const std::vector<Matrix2Df> &frames, float sigma_low, float sigma_high,
    int max_iters, float min_fraction, Matrix2Df &out,
    cv::cuda::Stream *stream) {
  const float invalid_sample = std::numeric_limits<float>::quiet_NaN();
  std::vector<std::reference_wrapper<const Matrix2Df>> valid;
  valid.reserve(frames.size());
  for (const auto &frame : frames) {
    if (frame.size() > 0) {
      valid.emplace_back(frame);
    }
  }
  if (valid.empty()) {
    out.resize(0, 0);
    return true;
  }

  const int rows = static_cast<int>(valid[0].get().rows());
  const int cols = static_cast<int>(valid[0].get().cols());
  if (rows <= 0 || cols <= 0) {
    out.resize(0, 0);
    return true;
  }

  try {
    cv::cuda::Stream& s = stream ? *stream : cv::cuda::Stream::Null();
    std::vector<cv::cuda::GpuMat> gpu_frames;
    std::vector<cv::cuda::GpuMat> keep_masks;
    gpu_frames.reserve(valid.size());
    keep_masks.reserve(valid.size());

    cv::cuda::GpuMat ones(rows, cols, CV_32F);
    cv::cuda::GpuMat zeros(rows, cols, CV_32F);
    cv::cuda::GpuMat initial_count(rows, cols, CV_32F);
    ones.setTo(cv::Scalar(1.0f), s);
    zeros.setTo(cv::Scalar(0.0f), s);
    initial_count.setTo(cv::Scalar(0.0f), s);

    for (const Matrix2Df &frame : valid) {
      cv::Mat host_view(rows, cols, CV_32F,
                        const_cast<float *>(frame.data()));
      cv::cuda::GpuMat gpu_frame;
      gpu_frame.upload(host_view, s);
      gpu_frames.push_back(gpu_frame);

      cv::Mat valid_mask_host = make_host_finite_mask(frame);
      cv::cuda::GpuMat valid_mask;
      valid_mask.upload(valid_mask_host, s);
      keep_masks.push_back(valid_mask.clone());

      cv::cuda::GpuMat valid_mask_f32;
      valid_mask.convertTo(valid_mask_f32, CV_32F, 1.0 / 255.0, s);
      cv::cuda::add(initial_count, valid_mask_f32, initial_count, cv::noArray(), -1, s);
    }

    cv::Mat initial_count_host(rows, cols, CV_32F);
    cv::Mat min_keep_host(rows, cols, CV_32F);
    initial_count.download(initial_count_host, s);
    s.waitForCompletion();
    for (int y = 0; y < rows; ++y) {
      const float *count_row = initial_count_host.ptr<float>(y);
      float *min_keep_row = min_keep_host.ptr<float>(y);
      for (int x = 0; x < cols; ++x) {
        const int n_valid_here = static_cast<int>(std::lround(count_row[x]));
        min_keep_row[x] = static_cast<float>(
            std::max(1, static_cast<int>(std::ceil(min_fraction * n_valid_here))));
      }
    }

    cv::cuda::GpuMat min_keep;
    min_keep.upload(min_keep_host, s);

    cv::cuda::GpuMat active_mask;
    cv::cuda::compare(initial_count, 0.0f, active_mask, cv::CMP_GT, s);

    for (int iter = 0; iter < max_iters; ++iter) {
      cv::cuda::GpuMat sum(rows, cols, CV_32F);
      cv::cuda::GpuMat sumsq(rows, cols, CV_32F);
      cv::cuda::GpuMat count(rows, cols, CV_32F);
      sum.setTo(cv::Scalar(0.0f), s);
      sumsq.setTo(cv::Scalar(0.0f), s);
      count.setTo(cv::Scalar(0.0f), s);

      for (size_t i = 0; i < gpu_frames.size(); ++i) {
        cv::cuda::GpuMat keep_f32;
        keep_masks[i].convertTo(keep_f32, CV_32F, 1.0 / 255.0, s);

        cv::cuda::GpuMat masked_frame;
        cv::cuda::multiply(gpu_frames[i], keep_f32, masked_frame, 1, -1, s);
        cv::cuda::add(sum, masked_frame, sum, cv::noArray(), -1, s);

        cv::cuda::GpuMat frame_sq;
        cv::cuda::multiply(gpu_frames[i], gpu_frames[i], frame_sq, 1, -1, s);
        cv::cuda::GpuMat masked_sq;
        cv::cuda::multiply(frame_sq, keep_f32, masked_sq, 1, -1, s);
        cv::cuda::add(sumsq, masked_sq, sumsq, cv::noArray(), -1, s);
        cv::cuda::add(count, keep_f32, count, cv::noArray(), -1, s);
      }

      cv::cuda::GpuMat count_denom;
      cv::cuda::max(count, ones, count_denom, s);

      cv::cuda::GpuMat mean;
      cv::cuda::divide(sum, count_denom, mean, 1, -1, s);

      cv::cuda::GpuMat sumsq_mean;
      cv::cuda::divide(sumsq, count_denom, sumsq_mean, 1, -1, s);
      cv::cuda::GpuMat mean_sq;
      cv::cuda::multiply(mean, mean, mean_sq, 1, -1, s);
      cv::cuda::GpuMat var;
      cv::cuda::subtract(sumsq_mean, mean_sq, var, cv::noArray(), -1, s);
      cv::cuda::max(var, zeros, var, s);

      cv::cuda::GpuMat kept_gt_one_mask;
      cv::cuda::compare(count, 1.0f, kept_gt_one_mask, cv::CMP_GT, s);
      cv::cuda::GpuMat kept_gt_one_f32;
      kept_gt_one_mask.convertTo(kept_gt_one_f32, CV_32F, 1.0 / 255.0, s);

      cv::cuda::GpuMat count_minus_one;
      cv::cuda::subtract(count, ones, count_minus_one, cv::noArray(), -1, s);
      cv::cuda::max(count_minus_one, ones, count_minus_one, s);
      cv::cuda::GpuMat factor;
      cv::cuda::divide(count, count_minus_one, factor, 1, -1, s);

      cv::cuda::GpuMat one_minus_gt_one;
      cv::cuda::subtract(ones, kept_gt_one_f32, one_minus_gt_one, cv::noArray(), -1, s);
      cv::cuda::GpuMat factor_when_gt_one;
      cv::cuda::multiply(factor, kept_gt_one_f32, factor_when_gt_one, 1, -1, s);
      cv::cuda::add(factor_when_gt_one, one_minus_gt_one, factor, cv::noArray(), -1, s);
      cv::cuda::multiply(var, factor, var, 1, -1, s);
      cv::cuda::max(var, zeros, var, s);

      cv::cuda::GpuMat sd;
      cv::cuda::sqrt(var, sd, s);

      cv::cuda::GpuMat sd_positive_mask;
      cv::cuda::compare(sd, 0.0f, sd_positive_mask, cv::CMP_GT, s);
      cv::cuda::GpuMat can_continue;
      cv::cuda::bitwise_and(kept_gt_one_mask, sd_positive_mask, can_continue, cv::noArray(), s);

      cv::cuda::GpuMat sigma_low_sd;
      cv::cuda::GpuMat sigma_high_sd;
      cv::cuda::multiply(sd, cv::Scalar(sigma_low), sigma_low_sd, 1, -1, s);
      cv::cuda::multiply(sd, cv::Scalar(sigma_high), sigma_high_sd, 1, -1, s);
      cv::cuda::GpuMat lo;
      cv::cuda::GpuMat hi;
      cv::cuda::subtract(mean, sigma_low_sd, lo, cv::noArray(), -1, s);
      cv::cuda::add(mean, sigma_high_sd, hi, cv::noArray(), -1, s);

      std::vector<cv::cuda::GpuMat> new_keep_masks;
      new_keep_masks.reserve(keep_masks.size());
      cv::cuda::GpuMat new_count(rows, cols, CV_32F);
      new_count.setTo(cv::Scalar(0.0f), s);

      for (size_t i = 0; i < gpu_frames.size(); ++i) {
        cv::cuda::GpuMat ge_lo;
        cv::cuda::GpuMat le_hi;
        cv::cuda::GpuMat in_range;
        cv::cuda::compare(gpu_frames[i], lo, ge_lo, cv::CMP_GE, s);
        cv::cuda::compare(gpu_frames[i], hi, le_hi, cv::CMP_LE, s);
        cv::cuda::bitwise_and(ge_lo, le_hi, in_range, cv::noArray(), s);

        cv::cuda::GpuMat new_keep;
        cv::cuda::bitwise_and(keep_masks[i], in_range, new_keep, cv::noArray(), s);
        new_keep_masks.push_back(new_keep);

        cv::cuda::GpuMat new_keep_f32;
        new_keep.convertTo(new_keep_f32, CV_32F, 1.0 / 255.0, s);
        cv::cuda::add(new_count, new_keep_f32, new_count, cv::noArray(), -1, s);
      }

      cv::cuda::GpuMat meets_min;
      cv::cuda::compare(new_count, min_keep, meets_min, cv::CMP_GE, s);
      cv::cuda::GpuMat update_mask;
      cv::cuda::bitwise_and(active_mask, can_continue, update_mask, cv::noArray(), s);
      cv::cuda::GpuMat apply_new_keep;
      cv::cuda::bitwise_and(update_mask, meets_min, apply_new_keep, cv::noArray(), s);
      cv::cuda::GpuMat next_active;
      apply_new_keep.copyTo(next_active, s);

      cv::cuda::GpuMat apply_new_keep_inv;
      cv::cuda::bitwise_not(apply_new_keep, apply_new_keep_inv, cv::noArray(), s);
      for (size_t i = 0; i < keep_masks.size(); ++i) {
        cv::cuda::GpuMat old_region;
        cv::cuda::GpuMat new_region;
        cv::cuda::bitwise_and(keep_masks[i], apply_new_keep_inv, old_region, cv::noArray(), s);
        cv::cuda::bitwise_and(new_keep_masks[i], apply_new_keep, new_region, cv::noArray(), s);
        cv::cuda::bitwise_or(old_region, new_region, keep_masks[i], cv::noArray(), s);
      }
      active_mask = next_active;
    }

    cv::cuda::GpuMat final_sum(rows, cols, CV_32F);
    cv::cuda::GpuMat final_count(rows, cols, CV_32F);
    final_sum.setTo(cv::Scalar(0.0f), s);
    final_count.setTo(cv::Scalar(0.0f), s);
    for (size_t i = 0; i < gpu_frames.size(); ++i) {
      cv::cuda::GpuMat keep_f32;
      keep_masks[i].convertTo(keep_f32, CV_32F, 1.0 / 255.0, s);
      cv::cuda::GpuMat masked_frame;
      cv::cuda::multiply(gpu_frames[i], keep_f32, masked_frame, 1, -1, s);
      cv::cuda::add(final_sum, masked_frame, final_sum, cv::noArray(), -1, s);
      cv::cuda::add(final_count, keep_f32, final_count, cv::noArray(), -1, s);
    }

    cv::cuda::GpuMat final_count_denom;
    cv::cuda::max(final_count, ones, final_count_denom, s);
    cv::cuda::GpuMat final_out;
    cv::cuda::divide(final_sum, final_count_denom, final_out, 1, -1, s);
    cv::cuda::GpuMat dead_mask;
    cv::cuda::compare(final_count, 0.0f, dead_mask, cv::CMP_LE, s);
    final_out.setTo(cv::Scalar(invalid_sample), dead_mask, s);

    out.resize(rows, cols);
    cv::Mat out_host(rows, cols, CV_32F, out.data());
    final_out.download(out_host, s);
    s.waitForCompletion();
    return true;
  } catch (...) {
    return false;
  }
}

bool cuda_reconstruct_aqmh_impl(
    size_t frame_count,
    const reconstruction::AqmhFrameLoader &load_frame,
    metrics::QualityMapCache *q_map_cache, const VectorXf &global_weights,
    const std::vector<uint8_t> &canvas_mask, int width, int height,
    const reconstruction::AqmhReconstructionConfig &cfg,
    reconstruction::AqmhReconstructionResult &result,
    cv::cuda::Stream *stream) {
  if (cfg.cherry_pick || frame_count == 0 || !load_frame || !q_map_cache ||
      width <= 0 || height <= 0) {
    return false;
  }
  constexpr float kAqmhSigmaClipEpsVar = 1.0e-12f;

  try {
    cv::cuda::Stream &s = stream ? *stream : cv::cuda::Stream::Null();
    const cv::Size size(width, height);
    cv::cuda::GpuMat zeros(size, CV_32F);
    cv::cuda::GpuMat eps(size, CV_32F);
    zeros.setTo(cv::Scalar(0.0f), s);
    eps.setTo(cv::Scalar(std::numeric_limits<float>::epsilon()), s);

    std::vector<uint8_t> canvas_u8(static_cast<size_t>(width) * height,
                                   canvas_mask.empty() ? 255u : 0u);
    if (!canvas_mask.empty()) {
      for (size_t i = 0; i < canvas_u8.size() && i < canvas_mask.size(); ++i)
        canvas_u8[i] = canvas_mask[i] ? 255u : 0u;
    }
    cv::Mat canvas_host(height, width, CV_8U, canvas_u8.data());
    cv::cuda::GpuMat canvas;
    canvas.upload(canvas_host, s);

    cv::cuda::GpuMat W(size, CV_32F), mean(size, CV_32F), M2(size, CV_32F);
    cv::cuda::GpuMat finite_count(size, CV_32F), positive_count(size, CV_32F);
    W.setTo(cv::Scalar(0.0f), s);
    mean.setTo(cv::Scalar(0.0f), s);
    M2.setTo(cv::Scalar(0.0f), s);
    finite_count.setTo(cv::Scalar(0.0f), s);
    positive_count.setTo(cv::Scalar(0.0f), s);

    auto finite_mask = [&](const cv::cuda::GpuMat &src,
                           cv::cuda::GpuMat &mask) {
      cv::cuda::GpuMat lo, hi;
      cv::cuda::compare(src, -std::numeric_limits<float>::max(), lo,
                        cv::CMP_GE, s);
      cv::cuda::compare(src, std::numeric_limits<float>::max(), hi,
                        cv::CMP_LE, s);
      cv::cuda::bitwise_and(lo, hi, mask, cv::noArray(), s);
    };
    auto mask_to_float = [&](const cv::cuda::GpuMat &mask,
                             cv::cuda::GpuMat &out) {
      mask.convertTo(out, CV_32F, 1.0 / 255.0, s);
    };

    uint64_t valid_px = 0;
    for (uint8_t v : canvas_u8)
      valid_px += v != 0;

    for (size_t fi = 0; fi < frame_count; ++fi) {
      Matrix2Df frame;
      if (!load_frame(fi, frame) || frame.rows() != height ||
          frame.cols() != width)
        continue;
      Matrix2Df q_map = q_map_cache->read_cached(fi);
      if (q_map.rows() != height || q_map.cols() != width) {
        result.missing_map_samples += valid_px;
        continue;
      }

      const float gw = fi >= static_cast<size_t>(global_weights.size())
                           ? 1.0f
                           : (std::isfinite(global_weights[fi]) &&
                                      global_weights[fi] > 0.0f
                                  ? global_weights[fi]
                                  : 0.0f);
      cv::Mat frame_host(height, width, CV_32F, frame.data());
      cv::Mat q_host(height, width, CV_32F, q_map.data());
      cv::cuda::GpuMat d_frame, d_q;
      d_frame.upload(frame_host, s);
      d_q.upload(q_host, s);

      cv::cuda::GpuMat finite_q, finite_frame;
      finite_mask(d_q, finite_q);
      finite_mask(d_frame, finite_frame);
      cv::cuda::GpuMat finite_canvas;
      cv::cuda::bitwise_and(finite_q, canvas, finite_canvas, cv::noArray(), s);
      cv::cuda::GpuMat finite_f32;
      mask_to_float(finite_canvas, finite_f32);
      cv::cuda::add(finite_count, finite_f32, finite_count, cv::noArray(), -1,
                    s);

      cv::cuda::GpuMat q_clean(size, CV_32F);
      q_clean.setTo(cv::Scalar(0.0f), s);
      d_q.copyTo(q_clean, finite_q, s);
      cv::cuda::max(q_clean, zeros, q_clean, s);
      cv::cuda::GpuMat weight;
      cv::cuda::multiply(q_clean, cv::Scalar(gw), weight, 1.0, -1, s);
      cv::cuda::GpuMat positive;
      cv::cuda::compare(weight, 0.0, positive, cv::CMP_GT, s);
      cv::cuda::bitwise_and(positive, finite_frame, positive, cv::noArray(), s);
      cv::cuda::bitwise_and(positive, canvas, positive, cv::noArray(), s);

      cv::cuda::GpuMat positive_f32;
      mask_to_float(positive, positive_f32);
      cv::cuda::add(positive_count, positive_f32, positive_count,
                    cv::noArray(), -1, s);
      cv::cuda::multiply(weight, positive_f32, weight, 1.0, -1, s);

      cv::cuda::GpuMat frame_clean(size, CV_32F);
      frame_clean.setTo(cv::Scalar(0.0f), s);
      d_frame.copyTo(frame_clean, positive, s);
      cv::cuda::GpuMat W_new;
      cv::cuda::add(W, weight, W_new, cv::noArray(), -1, s);
      cv::cuda::GpuMat W_safe;
      cv::cuda::max(W_new, eps, W_safe, s);
      cv::cuda::GpuMat delta;
      cv::cuda::subtract(frame_clean, mean, delta, cv::noArray(), -1, s);
      cv::cuda::GpuMat ratio;
      cv::cuda::divide(weight, W_safe, ratio, 1.0, -1, s);
      cv::cuda::GpuMat update;
      cv::cuda::multiply(ratio, delta, update, 1.0, -1, s);
      cv::cuda::GpuMat mean_new;
      cv::cuda::add(mean, update, mean_new, cv::noArray(), -1, s);
      cv::cuda::GpuMat delta2, term;
      cv::cuda::subtract(frame_clean, mean_new, delta2, cv::noArray(), -1, s);
      cv::cuda::multiply(delta, delta2, term, 1.0, -1, s);
      cv::cuda::multiply(term, weight, term, 1.0, -1, s);
      cv::cuda::add(M2, term, M2, cv::noArray(), -1, s);
      W = W_new;
      mean = mean_new;

      for (int y = 0; y < height; ++y) {
        const float *q = q_map.data() + static_cast<size_t>(y) * width;
        for (int x = 0; x < width; ++x) {
          if (!canvas_u8[static_cast<size_t>(y) * width + x])
            continue;
          if (std::isfinite(q[x]))
            ++result.finite_map_samples;
          else
            ++result.missing_map_samples;
        }
      }
    }

    cv::cuda::GpuMat W_safe;
    cv::cuda::max(W, eps, W_safe, s);
    cv::cuda::GpuMat variance;
    cv::cuda::divide(M2, W_safe, variance, 1.0, -1, s);
    cv::cuda::max(variance, zeros, variance, s);
    cv::cuda::GpuMat sigma;
    cv::cuda::sqrt(variance, sigma, s);
    cv::cuda::GpuMat lo_delta, hi_delta, lo, hi;
    cv::cuda::multiply(sigma, cv::Scalar(cfg.clip_sigma), lo_delta, 1.0, -1,
                       s);
    cv::cuda::multiply(sigma, cv::Scalar(cfg.clip_sigma), hi_delta, 1.0, -1,
                       s);
    cv::cuda::subtract(mean, lo_delta, lo, cv::noArray(), -1, s);
    cv::cuda::add(mean, hi_delta, hi, cv::noArray(), -1, s);

    cv::cuda::GpuMat clipped_accum(size, CV_32F);
    cv::cuda::GpuMat clipped_weight(size, CV_32F);
    clipped_accum.setTo(cv::Scalar(0.0f), s);
    clipped_weight.setTo(cv::Scalar(0.0f), s);

    for (size_t fi = 0; fi < frame_count; ++fi) {
      Matrix2Df frame;
      if (!load_frame(fi, frame) || frame.rows() != height ||
          frame.cols() != width)
        continue;
      Matrix2Df q_map = q_map_cache->read_cached(fi);
      if (q_map.rows() != height || q_map.cols() != width)
        continue;
      const float gw = fi >= static_cast<size_t>(global_weights.size())
                           ? 1.0f
                           : (std::isfinite(global_weights[fi]) &&
                                      global_weights[fi] > 0.0f
                                  ? global_weights[fi]
                                  : 0.0f);
      cv::Mat frame_host(height, width, CV_32F, frame.data());
      cv::Mat q_host(height, width, CV_32F, q_map.data());
      cv::cuda::GpuMat d_frame, d_q;
      d_frame.upload(frame_host, s);
      d_q.upload(q_host, s);
      cv::cuda::GpuMat finite_q, finite_frame;
      finite_mask(d_q, finite_q);
      finite_mask(d_frame, finite_frame);
      cv::cuda::GpuMat q_clean(size, CV_32F);
      q_clean.setTo(cv::Scalar(0.0f), s);
      d_q.copyTo(q_clean, finite_q, s);
      cv::cuda::max(q_clean, zeros, q_clean, s);
      cv::cuda::GpuMat weight;
      cv::cuda::multiply(q_clean, cv::Scalar(gw), weight, 1.0, -1, s);
      cv::cuda::GpuMat valid_weight;
      cv::cuda::compare(weight, 0.0, valid_weight, cv::CMP_GT, s);
      cv::cuda::bitwise_and(valid_weight, finite_frame, valid_weight,
                            cv::noArray(), s);
      cv::cuda::bitwise_and(valid_weight, canvas, valid_weight, cv::noArray(),
                            s);
      cv::cuda::GpuMat ge_lo, le_hi, in_range, sigma_small, keep;
      cv::cuda::compare(d_frame, lo, ge_lo, cv::CMP_GE, s);
      cv::cuda::compare(d_frame, hi, le_hi, cv::CMP_LE, s);
      cv::cuda::bitwise_and(ge_lo, le_hi, in_range, cv::noArray(), s);
      cv::cuda::compare(sigma, kAqmhSigmaClipEpsVar,
                        sigma_small, cv::CMP_LE, s);
      cv::cuda::bitwise_or(in_range, sigma_small, keep, cv::noArray(), s);
      cv::cuda::bitwise_and(keep, valid_weight, keep, cv::noArray(), s);
      cv::cuda::GpuMat keep_f32;
      mask_to_float(keep, keep_f32);
      cv::cuda::multiply(weight, keep_f32, weight, 1.0, -1, s);
      cv::cuda::GpuMat frame_clean(size, CV_32F);
      frame_clean.setTo(cv::Scalar(0.0f), s);
      d_frame.copyTo(frame_clean, keep, s);
      cv::cuda::GpuMat weighted_value;
      cv::cuda::multiply(frame_clean, weight, weighted_value, 1.0, -1, s);
      cv::cuda::add(clipped_accum, weighted_value, clipped_accum,
                    cv::noArray(), -1, s);
      cv::cuda::add(clipped_weight, weight, clipped_weight, cv::noArray(), -1,
                    s);
    }

    Matrix2Df host_W(height, width), host_mean(height, width);
    Matrix2Df host_finite(height, width), host_positive(height, width);
    Matrix2Df host_clipped_accum(height, width), host_clipped_weight(height,
                                                                        width);
    cv::Mat h_W(height, width, CV_32F, host_W.data());
    cv::Mat h_mean(height, width, CV_32F, host_mean.data());
    cv::Mat h_finite(height, width, CV_32F, host_finite.data());
    cv::Mat h_positive(height, width, CV_32F, host_positive.data());
    cv::Mat h_ca(height, width, CV_32F, host_clipped_accum.data());
    cv::Mat h_cw(height, width, CV_32F, host_clipped_weight.data());
    W.download(h_W, s);
    mean.download(h_mean, s);
    finite_count.download(h_finite, s);
    positive_count.download(h_positive, s);
    clipped_accum.download(h_ca, s);
    clipped_weight.download(h_cw, s);
    s.waitForCompletion();

    result.output = Matrix2Df::Zero(height, width);
    result.weight_sum = host_W;
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const size_t idx = static_cast<size_t>(y) * width + x;
        if (!canvas_u8[idx]) {
          result.weight_sum(y, x) = 0.0f;
          continue;
        }
        if (host_finite(y, x) > 0.0f && host_positive(y, x) <= 0.0f) {
          result.weight_sum(y, x) = 0.0f;
          ++result.unsupported_pixels;
          ++result.zero_veto_pixels;
          continue;
        }
        if (host_W(y, x) <= 0.0f) {
          result.weight_sum(y, x) = 0.0f;
          ++result.unsupported_pixels;
          continue;
        }
        const float min_kept =
            std::max(0.0f, cfg.min_fraction) * host_W(y, x);
        if (host_clipped_weight(y, x) > 0.0f &&
            host_clipped_weight(y, x) >= min_kept) {
          result.output(y, x) =
              host_clipped_accum(y, x) / host_clipped_weight(y, x);
          result.weight_sum(y, x) = host_clipped_weight(y, x);
        } else {
          result.output(y, x) = host_mean(y, x);
        }
      }
    }
    result.acceleration_used = true;
    return true;
  } catch (...) {
    return false;
  }
}
#endif

/// @brief Implements auto backend requested.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool auto_backend_requested(const std::string &name) {
  return core::to_lower(name) == "auto";
}

/// @brief Implements choose auto backend.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
AccelerationBackend choose_auto_backend(AccelerationPhase phase,
                                        bool tile_compile_with_cuda,
                                        bool opencv_cuda_runtime,
                                        bool opencv_opencl_runtime) {
  const bool opencv_cuda_headers = opencv_cuda_headers_available(phase);
  const bool opencv_opencl_headers = TILE_COMPILE_HAS_OPENCV_OPENCL != 0;
  const AccelerationBackend candidates[] = {
      AccelerationBackend::cuda,
      AccelerationBackend::opencv_cuda,
      AccelerationBackend::opencv_opencl,
      AccelerationBackend::cpu,
  };
  for (AccelerationBackend candidate : candidates) {
    if (!phase_supports_backend(phase, candidate)) {
      continue;
    }
    if (missing_backend_reason(candidate, tile_compile_with_cuda,
                               opencv_cuda_headers,
                               opencv_cuda_runtime,
                               opencv_opencl_headers,
                               opencv_opencl_runtime)
        .empty()) {
      return candidate;
    }
  }
  return AccelerationBackend::cpu;
}

// Keep stateless selection and run-scoped selection on one code path. The
// caller supplies either freshly probed or context-snapshotted capabilities.
AccelerationSelection select_with_capabilities(
    const std::string &requested_backend_name, AccelerationPhase phase,
    const AccelerationCapabilities &capabilities) {
  AccelerationSelection selection;
  selection.phase = phase;
  selection.tile_compile_with_cuda = capabilities.tile_compile_with_cuda;
  selection.opencv_cuda_headers = opencv_cuda_headers_available(phase);
  selection.opencv_cuda_runtime = capabilities.opencv_cuda_runtime;
  selection.opencv_opencl_headers = capabilities.opencv_opencl_headers;
  selection.opencv_opencl_runtime = capabilities.opencv_opencl_runtime;
  selection.requested_name = core::to_lower(requested_backend_name);
  if (selection.requested_name.empty()) selection.requested_name = "auto";
  selection.auto_requested = auto_backend_requested(selection.requested_name);

  if (selection.auto_requested) {
    selection.selected = choose_auto_backend(
        phase, selection.tile_compile_with_cuda,
        selection.opencv_cuda_runtime, selection.opencv_opencl_runtime);
    selection.requested = selection.selected;
    selection.gpu_requested = selection.selected != AccelerationBackend::cpu;
    selection.using_gpu = selection.gpu_requested;
    return selection;
  }

  AccelerationBackend requested = AccelerationBackend::cpu;
  if (!parse_acceleration_backend(selection.requested_name, requested)) {
    selection.request_honored = false;
    selection.fallback_reason = "invalid_requested_backend";
    return selection;
  }
  selection.requested = requested;
  selection.selected = requested;
  selection.gpu_requested = requested != AccelerationBackend::cpu;

  const std::string missing = missing_backend_reason(
      requested, selection.tile_compile_with_cuda,
      selection.opencv_cuda_headers, selection.opencv_cuda_runtime,
      selection.opencv_opencl_headers, selection.opencv_opencl_runtime);
  if (!missing.empty()) {
    selection.selected = AccelerationBackend::cpu;
    selection.request_honored = false;
    selection.fallback_reason = missing;
    return selection;
  }
  if (!phase_supports_backend(phase, requested)) {
    selection.selected = AccelerationBackend::cpu;
    selection.request_honored = false;
    selection.fallback_reason = unsupported_phase_reason(requested, phase);
    return selection;
  }
  selection.using_gpu = selection.gpu_requested;
  return selection;
}

} // namespace

/// @brief Implements acceleration phase name.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::string acceleration_phase_name(AccelerationPhase phase) {
  switch (phase) {
  case AccelerationPhase::prewarp:
    return "PREWARP";
  case AccelerationPhase::aqmh_maps:
    return "AQMH_MAPS";
  case AccelerationPhase::aqmh_reconstruction:
    return "AQMH_RECONSTRUCTION";
  case AccelerationPhase::tile_reconstruction:
    return "TILE_RECONSTRUCTION";
  case AccelerationPhase::stacking:
    return "STACKING";
  }
  return "UNKNOWN";
}

/// @brief Implements acceleration backend name.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::string acceleration_backend_name(AccelerationBackend backend) {
  switch (backend) {
  case AccelerationBackend::cpu:
    return "cpu";
  case AccelerationBackend::opencv_cuda:
    return "opencv_cuda";
  case AccelerationBackend::opencv_opencl:
    return "opencv_opencl";
  case AccelerationBackend::cuda:
    return "cuda";
  }
  return "cpu";
}

/// @brief Parses acceleration backend.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool parse_acceleration_backend(const std::string &name,
                                AccelerationBackend &backend_out) {
  const std::string normalized = core::to_lower(name);
  if (normalized == "cpu") {
    backend_out = AccelerationBackend::cpu;
    return true;
  }
  if (normalized == "opencv_cuda") {
    backend_out = AccelerationBackend::opencv_cuda;
    return true;
  }
  if (normalized == "opencv_opencl" || normalized == "opencl") {
    backend_out = AccelerationBackend::opencv_opencl;
    return true;
  }
  if (normalized == "cuda") {
    backend_out = AccelerationBackend::cuda;
    return true;
  }
  return false;
}

/// @brief Implements select acceleration backend.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
AccelerationSelection select_acceleration_backend(
    const std::string &requested_backend_name, AccelerationPhase phase) {
  AccelerationCapabilities capabilities;
  capabilities.tile_compile_with_cuda = TILE_COMPILE_WITH_CUDA != 0;
  capabilities.opencv_cuda_runtime = opencv_cuda_runtime_available();
  capabilities.opencv_opencl_headers = TILE_COMPILE_HAS_OPENCV_OPENCL != 0;
  capabilities.opencv_opencl_runtime = opencv_opencl_runtime_available();
  return select_with_capabilities(requested_backend_name, phase,
                                  capabilities);
}

AccelerationContext::AccelerationContext(std::string requested_backend_name,
                                         int device_id)
    : requested_backend_name_(core::to_lower(requested_backend_name)) {
  if (requested_backend_name_.empty())
    requested_backend_name_ = "auto";
  capabilities_.tile_compile_with_cuda = TILE_COMPILE_WITH_CUDA != 0;
  capabilities_.opencv_cuda_runtime = opencv_cuda_runtime_available();
  capabilities_.opencv_opencl_headers = TILE_COMPILE_HAS_OPENCV_OPENCL != 0;
  capabilities_.opencv_opencl_runtime = opencv_opencl_runtime_available();
  capabilities_.device_id = std::max(0, device_id);

#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS
  if (capabilities_.opencv_cuda_runtime) {
    try {
      const int count = cv::cuda::getCudaEnabledDeviceCount();
      if (capabilities_.device_id >= count)
        capabilities_.device_id = 0;
      cv::cuda::setDevice(capabilities_.device_id);
      cv::cuda::DeviceInfo info(capabilities_.device_id);
      capabilities_.device_name = info.name();
    } catch (...) {
      capabilities_.opencv_cuda_runtime = false;
    }
  }
#endif
#if TILE_COMPILE_HAS_OPENCV_OPENCL
  if (capabilities_.device_name.empty() &&
      capabilities_.opencv_opencl_runtime) {
    try {
      capabilities_.device_name = cv::ocl::Device::getDefault().name();
    } catch (...) {
    }
  }
#endif
}

AccelerationSelection
AccelerationContext::selection_for(AccelerationPhase phase) const {
  return select_with_capabilities(requested_backend_name_, phase,
                                  capabilities_);
}

json AccelerationContext::to_json() const {
  json phases = json::object();
  for (AccelerationPhase phase : {AccelerationPhase::prewarp,
                                  AccelerationPhase::aqmh_maps,
                                  AccelerationPhase::aqmh_reconstruction,
                                  AccelerationPhase::tile_reconstruction,
                                  AccelerationPhase::stacking}) {
    phases[acceleration_phase_name(phase)] =
        acceleration_selection_to_json(selection_for(phase));
  }
  return {{"requested_backend", requested_backend_name_},
          {"device_id", capabilities_.device_id},
          {"device_name", capabilities_.device_name},
          {"opencv_cuda_runtime", capabilities_.opencv_cuda_runtime},
          {"opencv_opencl_runtime", capabilities_.opencv_opencl_runtime},
          {"phases", std::move(phases)}};
}

void AccelerationContext::synchronize() const {
#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS
  if (capabilities_.opencv_cuda_runtime) {
    try {
      cv::cuda::Stream::Null().waitForCompletion();
    } catch (...) {
    }
  }
#endif
#if TILE_COMPILE_HAS_OPENCV_OPENCL
  if (capabilities_.opencv_opencl_runtime) {
    try {
      cv::ocl::finish();
    } catch (...) {
    }
  }
#endif
}

struct WorkerCudaStreams::Impl {
#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS
  std::vector<cv::cuda::Stream> streams;
#endif
};

WorkerCudaStreams::WorkerCudaStreams(bool enabled, size_t worker_count)
    : impl_(std::make_unique<Impl>()) {
#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS
  if (enabled && worker_count > 0) {
    impl_->streams.resize(worker_count);
  }
#else
  (void)enabled;
  (void)worker_count;
#endif
}

WorkerCudaStreams::~WorkerCudaStreams() = default;
WorkerCudaStreams::WorkerCudaStreams(WorkerCudaStreams &&) noexcept = default;
WorkerCudaStreams &
WorkerCudaStreams::operator=(WorkerCudaStreams &&) noexcept = default;

cv::cuda::Stream *WorkerCudaStreams::get(size_t worker_index) noexcept {
#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS
  return worker_index < impl_->streams.size() ? &impl_->streams[worker_index]
                                               : nullptr;
#else
  (void)worker_index;
  return nullptr;
#endif
}

size_t WorkerCudaStreams::size() const noexcept {
#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS
  return impl_->streams.size();
#else
  return 0;
#endif
}

/// @brief Implements acceleration selection to json.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
json acceleration_selection_to_json(const AccelerationSelection &selection) {
  json out = {
      {"phase", acceleration_phase_name(selection.phase)},
      {"requested_backend", selection.requested_name},
      {"selected_backend", acceleration_backend_name(selection.selected)},
      {"auto_requested", selection.auto_requested},
      {"request_honored", selection.request_honored},
      {"gpu_requested", selection.gpu_requested},
      {"using_gpu", selection.using_gpu},
      {"tile_compile_with_cuda", selection.tile_compile_with_cuda},
      {"opencv_cuda_headers", selection.opencv_cuda_headers},
      {"opencv_cuda_runtime", selection.opencv_cuda_runtime},
      {"opencv_opencl_headers", selection.opencv_opencl_headers},
      {"opencv_opencl_runtime", selection.opencv_opencl_runtime},
  };
  if (!selection.fallback_reason.empty()) {
    out["fallback_reason"] = selection.fallback_reason;
  }
  return out;
}

/// @brief Implements acceleration selection summary.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::string acceleration_selection_summary(
    const AccelerationSelection &selection) {
  std::ostringstream oss;
  oss << "requested=" << selection.requested_name
      << " selected=" << acceleration_backend_name(selection.selected)
      << " execution=" << (selection.using_gpu ? "GPU" : "CPU");
  if (selection.auto_requested) {
    oss << " (auto-detected)";
  }
  if (!selection.request_honored && !selection.fallback_reason.empty()) {
    oss << " [fallback: " << selection.fallback_reason << "]";
  }
  if (selection.using_gpu) {
    if (selection.selected == AccelerationBackend::opencv_cuda) {
      oss << " [OpenCV CUDA]";
    } else if (selection.selected == AccelerationBackend::opencv_opencl) {
      oss << " [OpenCL]";
    } else if (selection.selected == AccelerationBackend::cuda) {
      oss << " [native CUDA]";
    }
  }
  return oss.str();
}

/// @brief Creates device frame.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
DeviceFrame make_device_frame(int rows, int cols, int channels) {
  DeviceFrame frame;
  frame.rows = rows;
  frame.cols = cols;
  frame.channels = std::max(1, channels);
  frame.bytes = safe_frame_bytes(rows, cols, frame.channels);
  return frame;
}

/// @brief Creates device frame batch.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
DeviceFrameBatch make_device_frame_batch(size_t batch_size, int rows, int cols,
                                         int channels) {
  DeviceFrameBatch batch;
  batch.batch_size = batch_size;
  batch.frame = make_device_frame(rows, cols, channels);
  batch.total_bytes = batch.frame.bytes * batch.batch_size;
  return batch;
}

/// @brief Creates device tile batch.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
DeviceTileBatch make_device_tile_batch(const std::vector<Tile> &tiles,
                                       int channels) {
  DeviceTileBatch batch;
  batch.batch_size = tiles.size();
  batch.channels = std::max(1, channels);
  for (const Tile &tile : tiles) {
    batch.max_tile_width = std::max(batch.max_tile_width, tile.width);
    batch.max_tile_height = std::max(batch.max_tile_height, tile.height);
    const size_t tile_pixels =
        static_cast<size_t>(std::max(0, tile.width)) *
        static_cast<size_t>(std::max(0, tile.height));
    batch.total_pixels += tile_pixels;
    batch.total_bytes += tile_pixels * static_cast<size_t>(batch.channels) *
                         sizeof(float);
  }
  return batch;
}

/// @brief Implements device frame to json.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
json device_frame_to_json(const DeviceFrame &frame) {
  return {
      {"rows", frame.rows},
      {"cols", frame.cols},
      {"channels", frame.channels},
      {"bytes", frame.bytes},
  };
}

/// @brief Implements device frame batch to json.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
json device_frame_batch_to_json(const DeviceFrameBatch &batch) {
  return {
      {"batch_size", batch.batch_size},
      {"frame", device_frame_to_json(batch.frame)},
      {"total_bytes", batch.total_bytes},
  };
}

/// @brief Implements device tile batch to json.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
json device_tile_batch_to_json(const DeviceTileBatch &batch) {
  return {
      {"batch_size", batch.batch_size},
      {"channels", batch.channels},
      {"total_pixels", batch.total_pixels},
      {"total_bytes", batch.total_bytes},
      {"max_tile_width", batch.max_tile_width},
      {"max_tile_height", batch.max_tile_height},
  };
}

/// @brief Implements AccelerationOps.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
AccelerationOps::AccelerationOps(AccelerationSelection selection,
                                 std::string prewarp_interpolation)
    : selection_(std::move(selection)),
      prewarp_interpolation_(std::move(prewarp_interpolation)) {}

AccelerationOps::AccelerationOps(const AccelerationContext &context,
                                 AccelerationPhase phase,
                                 std::string prewarp_interpolation)
    : selection_(context.selection_for(phase)),
      prewarp_interpolation_(std::move(prewarp_interpolation)) {}

struct AccelerationOps::OverlapAddState {
  int rows = 0;
  int cols = 0;
#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS && TILE_COMPILE_HAS_OPENCV_CUDA_ARITHM
  cv::cuda::GpuMat gpu_mat;
#endif
#if TILE_COMPILE_HAS_OPENCV_OPENCL
  cv::UMat u_mat;
#endif
};

/// @brief Implements warp affine frame.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool AccelerationOps::warp_affine_frame(Matrix2Df img, const WarpMatrix &warp,
                                        ColorMode mode, int canvas_height,
                                        int canvas_width, int offset_x,
                                        int offset_y, Matrix2Df &warped_out,
                                        std::vector<uint8_t> *valid_mask_out,
                                        bool *has_data_out,
                                        cv::cuda::Stream *stream) const {
  auto update_valid_outputs_from_rect = [&](int dst_y, int dst_x, int copy_h,
                                            int copy_w) {
    cv::Mat mask(canvas_height, canvas_width, CV_8U, cv::Scalar(0));
    if (copy_h > 0 && copy_w > 0 && dst_y >= 0 && dst_x >= 0 &&
        (dst_y + copy_h) <= canvas_height &&
        (dst_x + copy_w) <= canvas_width) {
      mask(cv::Rect(dst_x, dst_y, copy_w, copy_h)).setTo(cv::Scalar(255));
    }
    invalidate_matrix_outside_support(warped_out, mask);
    write_valid_outputs_from_mask(mask, valid_mask_out, has_data_out);
  };

  auto update_valid_outputs_from_warp = [&](int src_height,
                                            int src_width) {
    cv::Mat support_mask;
    if (!build_warped_support_mask(warp_matrix_to_cv(warp), src_height,
                                   src_width,
                                   cv::Size(canvas_width, canvas_height),
                                   support_mask)) {
      support_mask = cv::Mat(canvas_height, canvas_width, CV_8U,
                             cv::Scalar(0));
    }
    invalidate_matrix_outside_support(warped_out, support_mask);
    write_valid_outputs_from_mask(support_mask, valid_mask_out, has_data_out);
  };

  if (img.size() <= 0) {
    warped_out.resize(0, 0);
    if (valid_mask_out != nullptr) {
      valid_mask_out->clear();
    }
    if (has_data_out != nullptr) {
      *has_data_out = false;
    }
    return false;
  }

  const int src_height = static_cast<int>(img.rows());
  const int src_width = static_cast<int>(img.cols());
  const int interpolation_flag =
      interpolation_flag_from_name(prewarp_interpolation_);
  if (warp_is_identity(warp)) {
    int dst_y = 0;
    int dst_x = 0;
    int copy_h = src_height;
    int copy_w = src_width;
    if (canvas_width > src_width || canvas_height > src_height) {
      warped_out = Matrix2Df::Zero(canvas_height, canvas_width);
      dst_y = std::max(0, offset_y);
      dst_x = std::max(0, offset_x);
      copy_h = std::min(src_height, canvas_height - dst_y);
      copy_w = std::min(src_width, canvas_width - dst_x);
      if (copy_h > 0 && copy_w > 0) {
        warped_out.block(dst_y, dst_x, copy_h, copy_w) =
            img.block(0, 0, copy_h, copy_w);
      }
    } else {
      warped_out = std::move(img);
      copy_h = std::min(src_height, canvas_height);
      copy_w = std::min(src_width, canvas_width);
    }
    update_valid_outputs_from_rect(dst_y, dst_x, copy_h, copy_w);
    return warped_out.size() > 0;
  }

#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS && TILE_COMPILE_HAS_OPENCV_CUDA_WARPING
  if (selection_.selected == AccelerationBackend::opencv_cuda &&
      selection_.phase == AccelerationPhase::prewarp) {
    if (mode == ColorMode::OSC) {
      if (cuda_warp_cfa_mosaic(img, warp, canvas_height, canvas_width,
                               warped_out, interpolation_flag, stream)) {
        update_valid_outputs_from_warp(src_height, src_width);
        return true;
      }
    } else {
      const cv::Mat src(static_cast<int>(img.rows()), static_cast<int>(img.cols()),
                        CV_32F, const_cast<float *>(img.data()));
      const cv::Mat warp_matrix = warp_matrix_to_cv(warp);
      cv::Mat dst;
      if (cuda_warp_affine_impl(src, warp_matrix,
                                cv::Size(canvas_width, canvas_height), dst,
                                interpolation_flag, stream)) {
        warped_out.resize(canvas_height, canvas_width);
        std::memcpy(warped_out.data(), dst.data,
                    static_cast<size_t>(warped_out.size()) * sizeof(float));
        update_valid_outputs_from_warp(src_height, src_width);
        return true;
      }
    }
  }
#endif

#if TILE_COMPILE_HAS_OPENCV_OPENCL
  if (selection_.selected == AccelerationBackend::opencv_opencl &&
      selection_.phase == AccelerationPhase::prewarp) {
    if (mode == ColorMode::OSC) {
      if (opencl_warp_cfa_mosaic(img, warp, canvas_height, canvas_width,
                                 warped_out, interpolation_flag)) {
        update_valid_outputs_from_warp(src_height, src_width);
        return true;
      }
    } else {
      const cv::Mat src(static_cast<int>(img.rows()), static_cast<int>(img.cols()),
                        CV_32F, const_cast<float *>(img.data()));
      const cv::Mat warp_matrix = warp_matrix_to_cv(warp);
      cv::Mat dst;
      if (opencl_warp_affine_impl(src, warp_matrix,
                                  cv::Size(canvas_width, canvas_height), dst,
                                  interpolation_flag)) {
        warped_out.resize(canvas_height, canvas_width);
        std::memcpy(warped_out.data(), dst.data,
                    static_cast<size_t>(warped_out.size()) * sizeof(float));
        update_valid_outputs_from_warp(src_height, src_width);
        return true;
      }
    }
  }
#endif

  warped_out =
      image::apply_global_warp(img, warp, mode, canvas_height, canvas_width,
                               prewarp_interpolation_);
  update_valid_outputs_from_warp(src_height, src_width);
  return warped_out.size() > 0;
}

/// @brief Implements fused 3-channel (RGB) warp affine frame with support mask update.
bool AccelerationOps::warp_affine_rgb_frame(
    Matrix2Df img_r, Matrix2Df img_g, Matrix2Df img_b,
    const WarpMatrix &warp, int canvas_height, int canvas_width,
    int offset_x, int offset_y,
    Matrix2Df &warped_r_out, Matrix2Df &warped_g_out, Matrix2Df &warped_b_out,
    std::vector<uint8_t> *valid_mask_out,
    bool *has_data_out,
    cv::cuda::Stream *stream) const {
  if (img_r.size() <= 0 || img_g.size() <= 0 || img_b.size() <= 0) {
    warped_r_out.resize(0, 0);
    warped_g_out.resize(0, 0);
    warped_b_out.resize(0, 0);
    if (valid_mask_out != nullptr) valid_mask_out->clear();
    if (has_data_out != nullptr) *has_data_out = false;
    return false;
  }

  const int src_height = static_cast<int>(img_r.rows());
  const int src_width = static_cast<int>(img_r.cols());
  const int interpolation_flag =
      interpolation_flag_from_name(prewarp_interpolation_);

  auto update_valid_outputs_from_warp = [&](int sh, int sw) {
    cv::Mat support_mask;
    if (!build_warped_support_mask(warp_matrix_to_cv(warp), sh, sw,
                                   cv::Size(canvas_width, canvas_height),
                                   support_mask)) {
      support_mask = cv::Mat(canvas_height, canvas_width, CV_8U, cv::Scalar(0));
    }
    invalidate_matrix_outside_support(warped_r_out, support_mask);
    invalidate_matrix_outside_support(warped_g_out, support_mask);
    invalidate_matrix_outside_support(warped_b_out, support_mask);
    write_valid_outputs_from_mask(support_mask, valid_mask_out, has_data_out);
  };

#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS && TILE_COMPILE_HAS_OPENCV_CUDA_WARPING
  if (selection_.selected == AccelerationBackend::opencv_cuda &&
      selection_.phase == AccelerationPhase::prewarp) {
    const cv::Mat src_r(src_height, src_width, CV_32F, const_cast<float *>(img_r.data()));
    const cv::Mat src_g(src_height, src_width, CV_32F, const_cast<float *>(img_g.data()));
    const cv::Mat src_b(src_height, src_width, CV_32F, const_cast<float *>(img_b.data()));
    const cv::Mat warp_matrix = warp_matrix_to_cv(warp);
    cv::Mat dst_r, dst_g, dst_b;
    if (cuda_warp_affine_rgb_impl(src_r, src_g, src_b, warp_matrix,
                                  cv::Size(canvas_width, canvas_height),
                                  dst_r, dst_g, dst_b,
                                  interpolation_flag, stream)) {
      warped_r_out.resize(canvas_height, canvas_width);
      warped_g_out.resize(canvas_height, canvas_width);
      warped_b_out.resize(canvas_height, canvas_width);
      const size_t byte_count = static_cast<size_t>(canvas_height * canvas_width) * sizeof(float);
      std::memcpy(warped_r_out.data(), dst_r.data, byte_count);
      std::memcpy(warped_g_out.data(), dst_g.data, byte_count);
      std::memcpy(warped_b_out.data(), dst_b.data, byte_count);
      update_valid_outputs_from_warp(src_height, src_width);
      return true;
    }
  }
#endif

  // Fallback: per-channel affine warp
  bool r_ok = warp_affine_frame(std::move(img_r), warp, ColorMode::MONO,
                                canvas_height, canvas_width, offset_x, offset_y,
                                warped_r_out, valid_mask_out, has_data_out, stream);
  bool g_ok = warp_affine_frame(std::move(img_g), warp, ColorMode::MONO,
                                canvas_height, canvas_width, offset_x, offset_y,
                                warped_g_out, nullptr, nullptr, stream);
  bool b_ok = warp_affine_frame(std::move(img_b), warp, ColorMode::MONO,
                                canvas_height, canvas_width, offset_x, offset_y,
                                warped_b_out, nullptr, nullptr, stream);
  return r_ok && g_ok && b_ok;
}

/// @brief Implements sigma clip reduce.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
reconstruction::WeightedTileResult AccelerationOps::sigma_clip_reduce(
    const std::vector<Matrix2Df> &tiles, const std::vector<float> &weights,
    float sigma_low, float sigma_high, int max_iters, float min_fraction,
    float eps_weight,
    cv::cuda::Stream *stream) const {
#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS && TILE_COMPILE_HAS_OPENCV_CUDA_ARITHM
  if (selection_.selected == AccelerationBackend::opencv_cuda &&
      (selection_.phase == AccelerationPhase::tile_reconstruction ||
       selection_.phase == AccelerationPhase::stacking)) {
    reconstruction::WeightedTileResult gpu_out;
    if (cuda_sigma_clip_weighted_tile_impl(
            tiles, weights, sigma_low, sigma_high, max_iters, min_fraction,
            eps_weight, gpu_out, stream)) {
      return gpu_out;
    }
  }
#endif
#if TILE_COMPILE_HAS_OPENCV_OPENCL
  if (selection_.selected == AccelerationBackend::opencv_opencl &&
      (selection_.phase == AccelerationPhase::tile_reconstruction ||
       selection_.phase == AccelerationPhase::stacking)) {
    reconstruction::WeightedTileResult gpu_out;
    if (opencl_sigma_clip_weighted_tile_impl(
            tiles, weights, sigma_low, sigma_high, max_iters, min_fraction,
            eps_weight, gpu_out)) {
      return gpu_out;
    }
  }
#endif
  return reconstruction::sigma_clip_weighted_tile_with_fallback(
      tiles, weights, sigma_low, sigma_high, max_iters, min_fraction,
      eps_weight);
}

/// @brief Implements sigma clip stack.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Matrix2Df AccelerationOps::sigma_clip_stack(const std::vector<Matrix2Df> &frames,
                                            float sigma_low,
                                            float sigma_high, int max_iters,
                                            float min_fraction,
                                            cv::cuda::Stream *stream) const {
#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS && TILE_COMPILE_HAS_OPENCV_CUDA_ARITHM
  if (selection_.selected == AccelerationBackend::opencv_cuda &&
      selection_.phase == AccelerationPhase::stacking) {
    Matrix2Df gpu_out;
    if (cuda_sigma_clip_stack_impl(frames, sigma_low, sigma_high, max_iters,
                                   min_fraction, gpu_out, stream)) {
      return gpu_out;
    }
  }
#endif
#if TILE_COMPILE_HAS_OPENCV_OPENCL
  if (selection_.selected == AccelerationBackend::opencv_opencl &&
      selection_.phase == AccelerationPhase::stacking) {
    Matrix2Df gpu_out;
    if (opencl_sigma_clip_stack_impl(frames, sigma_low, sigma_high, max_iters,
                                     min_fraction, gpu_out)) {
      return gpu_out;
    }
  }
#endif
  return reconstruction::sigma_clip_stack(frames, sigma_low, sigma_high,
                                          max_iters, min_fraction);
}

reconstruction::AqmhReconstructionResult AccelerationOps::reconstruct_aqmh(
    size_t frame_count,
    const reconstruction::AqmhFrameLoader &load_frame,
    metrics::QualityMapCache *q_map_cache, const VectorXf &global_weights,
    const std::vector<uint8_t> &canvas_mask, int width, int height,
    const reconstruction::AqmhReconstructionConfig &cfg,
    cv::cuda::Stream *stream,
    const reconstruction::AqmhMaskLoader &load_frame_valid_mask,
    const reconstruction::AqmhFrameRegionLoader &load_frame_region,
    const reconstruction::AqmhMaskRegionLoader &load_frame_valid_mask_region,
    const reconstruction::AqmhProgressCallback &progress) const {
  (void)stream;
  std::string cuda_fallback_reason;
#if TILE_COMPILE_WITH_CUDA
  if (selection_.selected == AccelerationBackend::cuda &&
      selection_.phase == AccelerationPhase::aqmh_reconstruction) {
    auto result = reconstruction::reconstruct_aqmh_weighted_cuda(
        frame_count, load_frame, q_map_cache, global_weights, canvas_mask,
        width, height, cfg, load_frame_valid_mask, load_frame_region,
        load_frame_valid_mask_region, progress);
    if (result.acceleration_used && !result.acceleration_fallback) {
      return result;
    }
    cuda_fallback_reason = result.acceleration_fallback_reason.empty()
        ? "cuda_path_declined_without_specific_reason"
        : result.acceleration_fallback_reason;
  }
#endif
  auto result = reconstruction::reconstruct_aqmh_weighted(
      frame_count, load_frame, q_map_cache, global_weights, canvas_mask, width,
      height, cfg, load_frame_valid_mask, load_frame_region,
      load_frame_valid_mask_region, progress);
  if (selection_.using_gpu &&
      selection_.phase == AccelerationPhase::aqmh_reconstruction) {
    result.acceleration_fallback = true;
    result.acceleration_fallback_reason = cuda_fallback_reason;
  }
  return result;
}

/// @brief Implements overlap add.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void AccelerationOps::overlap_add(
    const Matrix2Df &tile, const Tile &tile_bounds,
    const std::vector<float> &hann_x, const std::vector<float> &hann_y,
    const std::vector<uint8_t> &common_valid_mask, int canvas_width,
    Matrix2Df &accum, Matrix2Df &weight_sum, bool accumulate_weight) const {
  if (tile.rows() != tile_bounds.height || tile.cols() != tile_bounds.width) {
    return;
  }

  Matrix2Df coeff(tile_bounds.height, tile_bounds.width);
  coeff.setZero();
  const int x0 = std::max(0, tile_bounds.x);
  const int y0 = std::max(0, tile_bounds.y);
  for (int yy = 0; yy < tile.rows(); ++yy) {
    const int iy = y0 + yy;
    if (iy < 0 || iy >= accum.rows()) {
      continue;
    }
    for (int xx = 0; xx < tile.cols(); ++xx) {
      const int ix = x0 + xx;
      if (ix < 0 || ix >= accum.cols()) {
        continue;
      }
      const size_t common_idx =
          static_cast<size_t>(iy) * static_cast<size_t>(canvas_width) +
          static_cast<size_t>(ix);
      if (common_idx >= common_valid_mask.size() ||
          common_valid_mask[common_idx] == 0) {
        continue;
      }
      coeff(yy, xx) =
          hann_y[static_cast<size_t>(yy)] * hann_x[static_cast<size_t>(xx)];
    }
  }
  overlap_add(tile, tile_bounds, coeff, accum, weight_sum, accumulate_weight);
}

/// @brief Implements overlap add.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void AccelerationOps::overlap_add(const Matrix2Df &tile, const Tile &tile_bounds,
                                  const Matrix2Df &coeff, Matrix2Df &accum,
                                  Matrix2Df &weight_sum,
                                  bool accumulate_weight) const {
  if (tile.rows() != tile_bounds.height || tile.cols() != tile_bounds.width) {
    return;
  }
  if (coeff.rows() != tile.rows() || coeff.cols() != tile.cols()) {
    return;
  }

#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS && TILE_COMPILE_HAS_OPENCV_CUDA_ARITHM
  if (selection_.selected == AccelerationBackend::opencv_cuda &&
      selection_.phase == AccelerationPhase::tile_reconstruction) {
    auto ensure_state = [&](Matrix2Df &host_matrix)
        -> std::shared_ptr<OverlapAddState> {
      std::lock_guard<std::shared_mutex> lock(overlap_add_mutex_);
      auto &state = overlap_add_states_[&host_matrix];
      if (!state) {
        state = std::make_shared<OverlapAddState>();
      }
      if (host_matrix.rows() <= 0 || host_matrix.cols() <= 0) {
        return state;
      }
      if (state->rows != host_matrix.rows() || state->cols != host_matrix.cols() ||
          state->gpu_mat.empty()) {
        state->rows = static_cast<int>(host_matrix.rows());
        state->cols = static_cast<int>(host_matrix.cols());
        cv::Mat host_view(state->rows, state->cols, CV_32F,
                          host_matrix.data());
        state->gpu_mat.upload(host_view);
      }
      return state;
    };
    auto ensure_coeff_state = [&](const Matrix2Df &host_matrix)
        -> std::shared_ptr<OverlapAddState> {
      std::lock_guard<std::shared_mutex> lock(overlap_add_mutex_);
      auto &state = overlap_add_coeff_states_[&host_matrix];
      if (!state) {
        state = std::make_shared<OverlapAddState>();
      }
      if (host_matrix.rows() <= 0 || host_matrix.cols() <= 0) {
        return state;
      }
      if (state->rows != host_matrix.rows() || state->cols != host_matrix.cols() ||
          state->gpu_mat.empty()) {
        state->rows = static_cast<int>(host_matrix.rows());
        state->cols = static_cast<int>(host_matrix.cols());
        cv::Mat host_view(state->rows, state->cols, CV_32F,
                          const_cast<float *>(host_matrix.data()));
        state->gpu_mat.upload(host_view);
      }
      return state;
    };

    const int x0 = std::max(0, tile_bounds.x);
    const int y0 = std::max(0, tile_bounds.y);
    const int clip_w = std::min<int>(tile.cols(), accum.cols() - x0);
    const int clip_h = std::min<int>(tile.rows(), accum.rows() - y0);
    if (clip_w <= 0 || clip_h <= 0) {
      return;
    }

    thread_local Matrix2Df weighted_tile;
    weighted_tile.resize(clip_h, clip_w);
    weighted_tile.setZero();

    bool has_valid_pixels = false;
    for (int yy = 0; yy < clip_h; ++yy) {
      for (int xx = 0; xx < clip_w; ++xx) {
        const float coeff_value = coeff(yy, xx);
        if (!(coeff_value > 0.0f)) {
          continue;
        }
        const float tile_value = tile(yy, xx);
        if (!std::isfinite(tile_value)) {
          continue;
        }
        weighted_tile(yy, xx) = tile_value * coeff_value;
        has_valid_pixels = true;
      }
    }

    if (!has_valid_pixels) {
      return;
    }

    auto accum_state = ensure_state(accum);
    if (!accum_state || accum_state->gpu_mat.empty()) {
      return;
    }

    const cv::Rect roi(x0, y0, clip_w, clip_h);
    const cv::Rect coeff_roi_rect(0, 0, clip_w, clip_h);
    cv::cuda::GpuMat accum_roi(accum_state->gpu_mat, roi);
    cv::Mat weighted_tile_host(clip_h, clip_w, CV_32F, weighted_tile.data());
    thread_local cv::cuda::GpuMat weighted_tile_gpu;
    weighted_tile_gpu.upload(weighted_tile_host);
    cv::cuda::add(accum_roi, weighted_tile_gpu, accum_roi);

    if (accumulate_weight) {
      auto weight_state = ensure_state(weight_sum);
      auto coeff_state = ensure_coeff_state(coeff);
      if (weight_state && !weight_state->gpu_mat.empty() && coeff_state &&
          !coeff_state->gpu_mat.empty()) {
        cv::cuda::GpuMat weight_roi(weight_state->gpu_mat, roi);
        cv::cuda::GpuMat coeff_roi(coeff_state->gpu_mat, coeff_roi_rect);
        cv::cuda::add(weight_roi, coeff_roi, weight_roi);
      }
    }
    return;
  }
#endif

#if TILE_COMPILE_HAS_OPENCV_OPENCL
  if (selection_.selected == AccelerationBackend::opencv_opencl &&
      selection_.phase == AccelerationPhase::tile_reconstruction) {
    auto ensure_state = [&](Matrix2Df &host_matrix)
        -> std::shared_ptr<OverlapAddState> {
      std::lock_guard<std::shared_mutex> lock(overlap_add_mutex_);
      auto &state = overlap_add_states_[&host_matrix];
      if (!state) {
        state = std::make_shared<OverlapAddState>();
      }
      if (host_matrix.rows() <= 0 || host_matrix.cols() <= 0) {
        return state;
      }
      if (state->rows != host_matrix.rows() || state->cols != host_matrix.cols() ||
          state->u_mat.empty()) {
        state->rows = static_cast<int>(host_matrix.rows());
        state->cols = static_cast<int>(host_matrix.cols());
        cv::Mat host_view(state->rows, state->cols, CV_32F,
                          host_matrix.data());
        host_view.copyTo(state->u_mat);
      }
      return state;
    };
    auto ensure_coeff_state = [&](const Matrix2Df &host_matrix)
        -> std::shared_ptr<OverlapAddState> {
      std::lock_guard<std::shared_mutex> lock(overlap_add_mutex_);
      auto &state = overlap_add_coeff_states_[&host_matrix];
      if (!state) {
        state = std::make_shared<OverlapAddState>();
      }
      if (host_matrix.rows() <= 0 || host_matrix.cols() <= 0) {
        return state;
      }
      if (state->rows != host_matrix.rows() || state->cols != host_matrix.cols() ||
          state->u_mat.empty()) {
        state->rows = static_cast<int>(host_matrix.rows());
        state->cols = static_cast<int>(host_matrix.cols());
        cv::Mat host_view(state->rows, state->cols, CV_32F,
                          const_cast<float *>(host_matrix.data()));
        host_view.copyTo(state->u_mat);
      }
      return state;
    };

    const int x0 = std::max(0, tile_bounds.x);
    const int y0 = std::max(0, tile_bounds.y);
    const int clip_w = std::min<int>(tile.cols(), accum.cols() - x0);
    const int clip_h = std::min<int>(tile.rows(), accum.rows() - y0);
    if (clip_w <= 0 || clip_h <= 0) {
      return;
    }

    thread_local Matrix2Df weighted_tile;
    weighted_tile.resize(clip_h, clip_w);
    weighted_tile.setZero();

    bool has_valid_pixels = false;
    for (int yy = 0; yy < clip_h; ++yy) {
      for (int xx = 0; xx < clip_w; ++xx) {
        const float coeff_value = coeff(yy, xx);
        if (!(coeff_value > 0.0f)) {
          continue;
        }
        const float tile_value = tile(yy, xx);
        if (!std::isfinite(tile_value)) {
          continue;
        }
        weighted_tile(yy, xx) = tile_value * coeff_value;
        has_valid_pixels = true;
      }
    }

    if (!has_valid_pixels) {
      return;
    }

    auto accum_state = ensure_state(accum);
    if (!accum_state || accum_state->u_mat.empty()) {
      return;
    }

    const cv::Rect roi(x0, y0, clip_w, clip_h);
    const cv::Rect coeff_roi_rect(0, 0, clip_w, clip_h);
    cv::UMat accum_roi = accum_state->u_mat(roi);
    cv::Mat weighted_tile_host(clip_h, clip_w, CV_32F, weighted_tile.data());
    thread_local cv::UMat weighted_tile_gpu;
    if (weighted_tile_gpu.rows != clip_h || weighted_tile_gpu.cols != clip_w ||
        weighted_tile_gpu.type() != CV_32F) {
      weighted_tile_gpu.create(clip_h, clip_w, CV_32F);
    }
    weighted_tile_host.copyTo(weighted_tile_gpu);
    cv::add(accum_roi, weighted_tile_gpu, accum_roi);

    if (accumulate_weight) {
      auto weight_state = ensure_state(weight_sum);
      auto coeff_state = ensure_coeff_state(coeff);
      if (weight_state && !weight_state->u_mat.empty() && coeff_state &&
          !coeff_state->u_mat.empty()) {
        cv::UMat weight_roi = weight_state->u_mat(roi);
        cv::UMat coeff_roi = coeff_state->u_mat(coeff_roi_rect);
        cv::add(weight_roi, coeff_roi, weight_roi);
      }
    }
    return;
  }
#endif

  const int x0 = std::max(0, tile_bounds.x);
  const int y0 = std::max(0, tile_bounds.y);
  for (int yy = 0; yy < tile.rows(); ++yy) {
    const int iy = y0 + yy;
    if (iy < 0 || iy >= accum.rows()) {
      continue;
    }
    for (int xx = 0; xx < tile.cols(); ++xx) {
      const int ix = x0 + xx;
      if (ix < 0 || ix >= accum.cols()) {
        continue;
      }
      const float coeff_value = coeff(yy, xx);
      if (!(coeff_value > 0.0f)) {
        continue;
      }
      const float tile_value = tile(yy, xx);
      if (!std::isfinite(tile_value)) {
        continue;
      }
      accum(iy, ix) += tile_value * coeff_value;
      if (accumulate_weight) {
        weight_sum(iy, ix) += coeff_value;
      }
    }
  }
}

/// @brief Implements overlap add preweighted.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void AccelerationOps::overlap_add_preweighted(const Matrix2Df &weighted_tile,
                                              const Tile &tile_bounds,
                                              Matrix2Df &accum,
                                              Matrix2Df &weight_sum,
                                              const Matrix2Df *weight_mask,
                                              bool accumulate_weight) const {
  if (weighted_tile.rows() != tile_bounds.height ||
      weighted_tile.cols() != tile_bounds.width) {
    return;
  }

  const int x0 = std::max(0, tile_bounds.x);
  const int y0 = std::max(0, tile_bounds.y);
  const int clip_w = std::min<int>(weighted_tile.cols(), accum.cols() - x0);
  const int clip_h = std::min<int>(weighted_tile.rows(), accum.rows() - y0);
  if (clip_w <= 0 || clip_h <= 0) {
    return;
  }

  bool has_any = false;
  for (int yy = 0; yy < clip_h && !has_any; ++yy) {
    const float *row = weighted_tile.data() +
                       static_cast<size_t>(yy) * static_cast<size_t>(weighted_tile.cols());
    for (int xx = 0; xx < clip_w; ++xx) {
      if (std::isfinite(row[xx]) && row[xx] != 0.0f) {
        has_any = true;
        break;
      }
    }
  }
  if (!has_any) {
    return;
  }

  const Matrix2Df clipped_weighted =
      weighted_tile.block(0, 0, clip_h, clip_w);
  Matrix2Df clipped_weight_mask;
  const bool has_weight_mask =
      accumulate_weight && weight_mask != nullptr &&
      weight_mask->rows() == weighted_tile.rows() &&
      weight_mask->cols() == weighted_tile.cols();
  if (has_weight_mask) {
    clipped_weight_mask = weight_mask->block(0, 0, clip_h, clip_w);
  }

#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS && TILE_COMPILE_HAS_OPENCV_CUDA_ARITHM
  if (selection_.selected == AccelerationBackend::opencv_cuda &&
      selection_.phase == AccelerationPhase::tile_reconstruction) {
    auto ensure_state = [&](Matrix2Df &host_matrix)
        -> std::shared_ptr<OverlapAddState> {
      std::lock_guard<std::shared_mutex> lock(overlap_add_mutex_);
      auto &state = overlap_add_states_[&host_matrix];
      if (!state) {
        state = std::make_shared<OverlapAddState>();
      }
      if (host_matrix.rows() <= 0 || host_matrix.cols() <= 0) {
        return state;
      }
      if (state->rows != host_matrix.rows() || state->cols != host_matrix.cols() ||
          state->gpu_mat.empty()) {
        state->rows = static_cast<int>(host_matrix.rows());
        state->cols = static_cast<int>(host_matrix.cols());
        cv::Mat host_view(state->rows, state->cols, CV_32F,
                          host_matrix.data());
        state->gpu_mat.upload(host_view);
      }
      return state;
    };

    auto accum_state = ensure_state(accum);
    if (!accum_state || accum_state->gpu_mat.empty()) {
      return;
    }
    const cv::Rect roi(x0, y0, clip_w, clip_h);
    cv::cuda::GpuMat accum_roi(accum_state->gpu_mat, roi);
    cv::Mat weighted_tile_host(clip_h, clip_w, CV_32F,
                               const_cast<float *>(clipped_weighted.data()));
    cv::cuda::GpuMat weighted_tile_gpu;
    weighted_tile_gpu.upload(weighted_tile_host);
    cv::cuda::add(accum_roi, weighted_tile_gpu, accum_roi);

    if (has_weight_mask) {
      auto weight_state = ensure_state(weight_sum);
      if (weight_state && !weight_state->gpu_mat.empty()) {
        cv::cuda::GpuMat weight_roi(weight_state->gpu_mat, roi);
        cv::Mat weight_mask_host(clip_h, clip_w, CV_32F,
                                 const_cast<float *>(clipped_weight_mask.data()));
        cv::cuda::GpuMat weight_mask_gpu;
        weight_mask_gpu.upload(weight_mask_host);
        cv::cuda::add(weight_roi, weight_mask_gpu, weight_roi);
      }
    }
    return;
  }
#endif

#if TILE_COMPILE_HAS_OPENCV_OPENCL
  if (selection_.selected == AccelerationBackend::opencv_opencl &&
      selection_.phase == AccelerationPhase::tile_reconstruction) {
    auto ensure_state = [&](Matrix2Df &host_matrix)
        -> std::shared_ptr<OverlapAddState> {
      std::lock_guard<std::shared_mutex> lock(overlap_add_mutex_);
      auto &state = overlap_add_states_[&host_matrix];
      if (!state) {
        state = std::make_shared<OverlapAddState>();
      }
      if (host_matrix.rows() <= 0 || host_matrix.cols() <= 0) {
        return state;
      }
      if (state->rows != host_matrix.rows() || state->cols != host_matrix.cols() ||
          state->u_mat.empty()) {
        state->rows = static_cast<int>(host_matrix.rows());
        state->cols = static_cast<int>(host_matrix.cols());
        cv::Mat host_view(state->rows, state->cols, CV_32F,
                          host_matrix.data());
        host_view.copyTo(state->u_mat);
      }
      return state;
    };

    auto accum_state = ensure_state(accum);
    if (!accum_state || accum_state->u_mat.empty()) {
      return;
    }
    const cv::Rect roi(x0, y0, clip_w, clip_h);
    cv::UMat accum_roi = accum_state->u_mat(roi);
    cv::Mat weighted_tile_host(clip_h, clip_w, CV_32F,
                               const_cast<float *>(clipped_weighted.data()));
    thread_local cv::UMat weighted_tile_gpu;
    if (weighted_tile_gpu.rows != clip_h || weighted_tile_gpu.cols != clip_w ||
        weighted_tile_gpu.type() != CV_32F) {
      weighted_tile_gpu.create(clip_h, clip_w, CV_32F);
    }
    weighted_tile_host.copyTo(weighted_tile_gpu);
    cv::add(accum_roi, weighted_tile_gpu, accum_roi);

    if (has_weight_mask) {
      auto weight_state = ensure_state(weight_sum);
      if (weight_state && !weight_state->u_mat.empty()) {
        cv::UMat weight_roi = weight_state->u_mat(roi);
        cv::Mat weight_mask_host(clip_h, clip_w, CV_32F,
                                 const_cast<float *>(clipped_weight_mask.data()));
        thread_local cv::UMat weight_mask_gpu;
        if (weight_mask_gpu.rows != clip_h || weight_mask_gpu.cols != clip_w ||
            weight_mask_gpu.type() != CV_32F) {
          weight_mask_gpu.create(clip_h, clip_w, CV_32F);
        }
        weight_mask_host.copyTo(weight_mask_gpu);
        cv::add(weight_roi, weight_mask_gpu, weight_roi);
      }
    }
    return;
  }
#endif

  accum.block(y0, x0, clip_h, clip_w).array() +=
      clipped_weighted.array();
  if (has_weight_mask) {
    weight_sum.block(y0, x0, clip_h, clip_w).array() +=
        clipped_weight_mask.array();
  }
}

/// @brief Normalizes overlap accum.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool AccelerationOps::normalize_overlap_accum(Matrix2Df &accum,
                                              Matrix2Df &weight_sum,
                                              float eps_weight,
                                              float invalid_value) const {
#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS && TILE_COMPILE_HAS_OPENCV_CUDA_ARITHM
  if (selection_.selected == AccelerationBackend::opencv_cuda &&
      selection_.phase == AccelerationPhase::tile_reconstruction) {
    std::lock_guard<std::shared_mutex> lock(overlap_add_mutex_);
    auto accum_it = overlap_add_states_.find(&accum);
    auto weight_it = overlap_add_states_.find(&weight_sum);
    if (accum_it == overlap_add_states_.end() ||
        weight_it == overlap_add_states_.end() || !accum_it->second ||
        !weight_it->second) {
      return false;
    }

    OverlapAddState &accum_state = *accum_it->second;
    OverlapAddState &weight_state = *weight_it->second;
    if (accum_state.gpu_mat.empty() || weight_state.gpu_mat.empty() ||
        accum_state.rows != weight_state.rows ||
        accum_state.cols != weight_state.cols) {
      return false;
    }

    cv::cuda::GpuMat eps_mat(accum_state.rows, accum_state.cols, CV_32F);
    eps_mat.setTo(cv::Scalar(eps_weight));
    cv::cuda::GpuMat denom;
    cv::cuda::max(weight_state.gpu_mat, eps_mat, denom);
    cv::cuda::divide(accum_state.gpu_mat, denom, accum_state.gpu_mat);
    cv::cuda::GpuMat invalid_mask;
    cv::cuda::compare(weight_state.gpu_mat, eps_weight, invalid_mask,
                      cv::CMP_LE);
    accum_state.gpu_mat.setTo(cv::Scalar(invalid_value), invalid_mask);
    return true;
  }
#endif

#if TILE_COMPILE_HAS_OPENCV_OPENCL
  if (selection_.selected == AccelerationBackend::opencv_opencl &&
      selection_.phase == AccelerationPhase::tile_reconstruction) {
    std::lock_guard<std::shared_mutex> lock(overlap_add_mutex_);
    auto accum_it = overlap_add_states_.find(&accum);
    auto weight_it = overlap_add_states_.find(&weight_sum);
    if (accum_it == overlap_add_states_.end() ||
        weight_it == overlap_add_states_.end() || !accum_it->second ||
        !weight_it->second) {
      return false;
    }

    OverlapAddState &accum_state = *accum_it->second;
    OverlapAddState &weight_state = *weight_it->second;
    if (accum_state.u_mat.empty() || weight_state.u_mat.empty() ||
        accum_state.rows != weight_state.rows ||
        accum_state.cols != weight_state.cols) {
      return false;
    }

    cv::UMat eps_mat(accum_state.rows, accum_state.cols, CV_32F);
    eps_mat.setTo(cv::Scalar(eps_weight));
    cv::UMat denom;
    cv::max(weight_state.u_mat, eps_mat, denom);
    cv::divide(accum_state.u_mat, denom, accum_state.u_mat);
    cv::UMat invalid_mask;
    cv::compare(weight_state.u_mat, eps_weight, invalid_mask, cv::CMP_LE);
    accum_state.u_mat.setTo(cv::Scalar(invalid_value), invalid_mask);
    return true;
  }
#endif

  (void)accum;
  (void)weight_sum;
  (void)eps_weight;
  (void)invalid_value;
  return false;
}

/// @brief Implements flush overlap state.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void AccelerationOps::flush_overlap_state(Matrix2Df &accum,
                                          Matrix2Df &weight_sum) const {
#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS && TILE_COMPILE_HAS_OPENCV_CUDA_ARITHM
  if (selection_.selected == AccelerationBackend::opencv_cuda &&
      selection_.phase == AccelerationPhase::tile_reconstruction) {
    std::lock_guard<std::shared_mutex> lock(overlap_add_mutex_);
    auto flush_cuda = [&](Matrix2Df &host_matrix) {
      auto it = overlap_add_states_.find(&host_matrix);
      if (it == overlap_add_states_.end() || !it->second) {
        return;
      }
      OverlapAddState &state = *it->second;
      if (state.rows > 0 && state.cols > 0 && !state.gpu_mat.empty()) {
        if (host_matrix.rows() != state.rows ||
            host_matrix.cols() != state.cols) {
          host_matrix.resize(state.rows, state.cols);
        }
        cv::Mat host_view(state.rows, state.cols, CV_32F, host_matrix.data());
        state.gpu_mat.download(host_view);
      }
      overlap_add_states_.erase(it);
    };

    flush_cuda(accum);
    flush_cuda(weight_sum);
    overlap_add_coeff_states_.clear();
    return;
  }
#endif

#if TILE_COMPILE_HAS_OPENCV_OPENCL
  if (selection_.selected == AccelerationBackend::opencv_opencl &&
      selection_.phase == AccelerationPhase::tile_reconstruction) {
    std::lock_guard<std::shared_mutex> lock(overlap_add_mutex_);
    auto flush_opencl = [&](Matrix2Df &host_matrix) {
      auto it = overlap_add_states_.find(&host_matrix);
      if (it == overlap_add_states_.end() || !it->second) {
        return;
      }
      OverlapAddState &state = *it->second;
      if (state.rows > 0 && state.cols > 0 && !state.u_mat.empty()) {
        if (host_matrix.rows() != state.rows ||
            host_matrix.cols() != state.cols) {
          host_matrix.resize(state.rows, state.cols);
        }
        cv::Mat host_view(state.rows, state.cols, CV_32F, host_matrix.data());
        state.u_mat.copyTo(host_view);
      }
      overlap_add_states_.erase(it);
    };

    flush_opencl(accum);
    flush_opencl(weight_sum);
    overlap_add_coeff_states_.clear();
    return;
  }
#endif

  (void)accum;
  (void)weight_sum;
  overlap_add_coeff_states_.clear();
}

} // namespace tile_compile::core

// ---------------------------------------------------------------------------
// AccelerationOps::sigma_clip_reduce_batch — GPU batch interface (B6)
// ---------------------------------------------------------------------------
namespace tile_compile::core {

std::vector<reconstruction::WeightedTileResult>
/// @brief Implements sigma clip reduce batch.
/// @details Part of GPU/CPU backend selection and accelerated image-operation wrappers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
AccelerationOps::sigma_clip_reduce_batch(
    const std::vector<BatchSigmaClipInput>& tile_inputs,
    float sigma_low,
    float sigma_high,
    int   max_iters,
    float min_fraction,
    float eps_weight,
    cv::cuda::Stream *stream) const
{
    std::vector<reconstruction::WeightedTileResult> results;
    results.resize(tile_inputs.size());

    const size_t n_tiles = tile_inputs.size();
    if (n_tiles == 0) {
        return results;
    }

    // For GPU backends: process tiles in parallel using multiple streams/workers
    // For CPU backend: use OpenMP parallel for if available
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic, 1) if(selection_.selected == AccelerationBackend::cpu)
    #endif
    for (size_t i = 0; i < n_tiles; ++i) {
        results[i] = sigma_clip_reduce(
            tile_inputs[i].tile_frames,
            tile_inputs[i].weights,
            sigma_low, sigma_high,
            max_iters, min_fraction, eps_weight, stream);
    }

    return results;
}

} // namespace tile_compile::core
