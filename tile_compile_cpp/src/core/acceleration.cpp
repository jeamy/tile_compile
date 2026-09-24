#include "tile_compile/core/acceleration.hpp"
#include "tile_compile/core/cfa_warp.hpp"

#include "tile_compile/core/utils.hpp"
#include "tile_compile/image/normalization.hpp"

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
  case AccelerationPhase::forward_drizzle:
    return false;  // custom CUDA path, not OpenCV-CUDA
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
    return phase == AccelerationPhase::prewarp;
  case AccelerationBackend::opencv_opencl:
    return phase == AccelerationPhase::prewarp;
  case AccelerationBackend::cuda:
    return phase == AccelerationPhase::forward_drizzle;
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
  case AccelerationPhase::forward_drizzle:
    return "FORWARD_DRIZZLE";
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
                                  AccelerationPhase::forward_drizzle}) {
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
} // namespace tile_compile::core
