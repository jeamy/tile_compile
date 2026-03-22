#include "tile_compile/core/acceleration.hpp"

#include "tile_compile/core/utils.hpp"
#include "tile_compile/image/cfa_processing.hpp"
#include "tile_compile/image/normalization.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
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

namespace tile_compile::core {

namespace {

bool opencv_cuda_headers_available(AccelerationPhase phase) {
#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS
  switch (phase) {
  case AccelerationPhase::prewarp:
    return TILE_COMPILE_HAS_OPENCV_CUDA_WARPING != 0;
  case AccelerationPhase::tile_reconstruction:
    return TILE_COMPILE_HAS_OPENCV_CUDA_ARITHM != 0;
  case AccelerationPhase::stacking:
    return TILE_COMPILE_HAS_OPENCV_CUDA_ARITHM != 0;
  }
#endif
  (void)phase;
  return false;
}

bool phase_supports_backend(AccelerationPhase phase,
                            AccelerationBackend backend) {
  switch (backend) {
  case AccelerationBackend::cpu:
    return true;
  case AccelerationBackend::opencv_cuda:
    return phase == AccelerationPhase::prewarp ||
           phase == AccelerationPhase::tile_reconstruction ||
           phase == AccelerationPhase::stacking;
  case AccelerationBackend::cuda:
    return false;
  }
  return false;
}

bool opencv_cuda_runtime_available() {
#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS && TILE_COMPILE_HAS_OPENCV_CUDA_WARPING
  try {
    return cv::cuda::getCudaEnabledDeviceCount() > 0;
  } catch (...) {
    return false;
  }
#else
  return false;
#endif
}

std::string missing_backend_reason(AccelerationBackend backend,
                                   bool tile_compile_with_cuda,
                                   bool opencv_cuda_headers,
                                   bool opencv_cuda_runtime) {
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
  case AccelerationBackend::cuda:
    if (!tile_compile_with_cuda) {
      return "cuda_backend_not_built";
    }
    return {};
  }
  return "unknown_backend";
}

std::string unsupported_phase_reason(AccelerationBackend backend,
                                     AccelerationPhase phase) {
  return acceleration_backend_name(backend) + "_backend_not_implemented_for_" +
         acceleration_phase_name(phase);
}

size_t safe_frame_bytes(int rows, int cols, int channels) {
  const size_t r = static_cast<size_t>(std::max(0, rows));
  const size_t c = static_cast<size_t>(std::max(0, cols));
  const size_t ch = static_cast<size_t>(std::max(1, channels));
  return r * c * ch * sizeof(float);
}

bool warp_is_identity(const WarpMatrix &warp) {
  const float eps = 1.0e-6f;
  return std::fabs(warp(0, 0) - 1.0f) < eps &&
         std::fabs(warp(0, 1)) < eps && std::fabs(warp(1, 0)) < eps &&
         std::fabs(warp(1, 1) - 1.0f) < eps && std::fabs(warp(0, 2)) < eps &&
         std::fabs(warp(1, 2)) < eps;
}

cv::Mat warp_matrix_to_cv(const WarpMatrix &warp) {
  cv::Mat warp_matrix(2, 3, CV_32F);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      warp_matrix.at<float>(i, j) = warp(i, j);
    }
  }
  return warp_matrix;
}

#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS && TILE_COMPILE_HAS_OPENCV_CUDA_WARPING
bool cuda_warp_affine_impl(const cv::Mat &src, const cv::Mat &warp_matrix,
                           cv::Size output_size, cv::Mat &dst,
                           cv::Mat *valid_mask_host = nullptr,
                           int *nonzero_count = nullptr) {
  try {
    cv::cuda::GpuMat d_src;
    cv::cuda::GpuMat d_dst;
    d_src.upload(src);
    cv::cuda::warpAffine(d_src, d_dst, warp_matrix, output_size,
                         cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                         cv::BORDER_CONSTANT, cv::Scalar(0));
#if TILE_COMPILE_HAS_OPENCV_CUDA_ARITHM
    if (valid_mask_host != nullptr || nonzero_count != nullptr) {
      cv::cuda::GpuMat d_mask;
      cv::cuda::compare(d_dst, 0.0f, d_mask, cv::CMP_GT);
      if (nonzero_count != nullptr) {
        *nonzero_count = cv::cuda::countNonZero(d_mask);
      }
      if (valid_mask_host != nullptr) {
        d_mask.download(*valid_mask_host);
      }
    }
#else
    (void)valid_mask_host;
    (void)nonzero_count;
#endif
    d_dst.download(dst);
    return !dst.empty();
  } catch (...) {
    return false;
  }
}

bool cuda_warp_cfa_mosaic(const Matrix2Df &mosaic, const WarpMatrix &warp,
                          int out_height, int out_width, Matrix2Df &out) {
  const int h = static_cast<int>(mosaic.rows());
  const int w = static_cast<int>(mosaic.cols());
  const int h2 = h - (h % 2);
  const int w2 = w - (w % 2);
  const int out_h = (out_height > 0) ? out_height : h;
  const int out_w = (out_width > 0) ? out_width : w;
  const int out_h2 = out_h - (out_h % 2);
  const int out_w2 = out_w - (out_w % 2);
  const int sub_h = h2 / 2;
  const int sub_w = w2 / 2;
  const int out_h_sub = std::max(1, out_h2 / 2);
  const int out_w_sub = std::max(1, out_w2 / 2);

  Matrix2Df a(sub_h, sub_w), b(sub_h, sub_w), c(sub_h, sub_w), d(sub_h, sub_w);
  for (int y = 0; y < sub_h; ++y) {
    for (int x = 0; x < sub_w; ++x) {
      a(y, x) = mosaic(y * 2, x * 2);
      b(y, x) = mosaic(y * 2, x * 2 + 1);
      c(y, x) = mosaic(y * 2 + 1, x * 2);
      d(y, x) = mosaic(y * 2 + 1, x * 2 + 1);
    }
  }

  const float a2_00 = warp(0, 0);
  const float a2_01 = warp(0, 1);
  const float a2_10 = warp(1, 0);
  const float a2_11 = warp(1, 1);
  const float t_x = warp(0, 2) * 0.5f;
  const float t_y = warp(1, 2) * 0.5f;
  auto make_warp = [&](float dx, float dy) -> cv::Mat {
    const float new_tx = t_x + (a2_00 * dx + a2_01 * dy) - dx;
    const float new_ty = t_y + (a2_10 * dx + a2_11 * dy) - dy;
    return (cv::Mat_<float>(2, 3) << a2_00, a2_01, new_tx, a2_10, a2_11,
            new_ty);
  };

  cv::Mat a_cv(sub_h, sub_w, CV_32F, a.data());
  cv::Mat b_cv(sub_h, sub_w, CV_32F, b.data());
  cv::Mat c_cv(sub_h, sub_w, CV_32F, c.data());
  cv::Mat d_cv(sub_h, sub_w, CV_32F, d.data());
  cv::Mat a_w, b_w, c_w, d_w;
  const cv::Size out_size(out_w_sub, out_h_sub);
  if (!cuda_warp_affine_impl(a_cv, make_warp(-0.25f, -0.25f), out_size, a_w) ||
      !cuda_warp_affine_impl(b_cv, make_warp(0.25f, -0.25f), out_size, b_w) ||
      !cuda_warp_affine_impl(c_cv, make_warp(-0.25f, 0.25f), out_size, c_w) ||
      !cuda_warp_affine_impl(d_cv, make_warp(0.25f, 0.25f), out_size, d_w)) {
    return false;
  }

  out = Matrix2Df::Zero(out_h, out_w);
  for (int y = 0; y < out_h_sub; ++y) {
    for (int x = 0; x < out_w_sub; ++x) {
      out(y * 2, x * 2) = a_w.at<float>(y, x);
      out(y * 2, x * 2 + 1) = b_w.at<float>(y, x);
      out(y * 2 + 1, x * 2) = c_w.at<float>(y, x);
      out(y * 2 + 1, x * 2 + 1) = d_w.at<float>(y, x);
    }
  }
  return out.size() > 0;
}
#endif

#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS && TILE_COMPILE_HAS_OPENCV_CUDA_ARITHM
bool cuda_sigma_clip_weighted_tile_impl(
    const std::vector<Matrix2Df> &tiles, const std::vector<float> &weights,
    float sigma_low, float sigma_high, int max_iters, float min_fraction,
    float eps_weight, reconstruction::WeightedTileResult &out) {
  out = reconstruction::WeightedTileResult{};
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
    ones.setTo(cv::Scalar(1.0f));
    zeros.setTo(cv::Scalar(0.0f));
    eps.setTo(cv::Scalar(1.0e-6f));
    valid_count.setTo(cv::Scalar(0.0f));

    for (const Matrix2Df &tile : active_tiles) {
      cv::Mat host_view(rows, cols, CV_32F,
                        const_cast<float *>(tile.data()));
      cv::cuda::GpuMat gpu_tile;
      gpu_tile.upload(host_view);
      gpu_tiles.push_back(gpu_tile);

      cv::cuda::GpuMat valid_mask;
      cv::cuda::compare(gpu_tiles.back(), 0.0f, valid_mask, cv::CMP_GT);
      valid_masks.push_back(valid_mask.clone());
      keep_masks.push_back(valid_mask.clone());

      cv::cuda::GpuMat valid_mask_f32;
      valid_mask.convertTo(valid_mask_f32, CV_32F, 1.0 / 255.0);
      cv::cuda::add(valid_count, valid_mask_f32, valid_count);
    }

    cv::Mat valid_count_host(rows, cols, CV_32F);
    cv::Mat min_keep_host(rows, cols, CV_32F);
    valid_count.download(valid_count_host);
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
    min_keep.upload(min_keep_host);

    const bool enable_clipping =
        static_cast<int>(gpu_tiles.size()) > 2 && max_iters > 0;
    cv::cuda::GpuMat active_mask;
    cv::cuda::compare(valid_count, 0.0f, active_mask, cv::CMP_GT);

    if (enable_clipping) {
      for (int iter = 0; iter < max_iters; ++iter) {
        cv::cuda::GpuMat wsum(rows, cols, CV_32F);
        cv::cuda::GpuMat wsum2(rows, cols, CV_32F);
        cv::cuda::GpuMat wmean_num(rows, cols, CV_32F);
        wsum.setTo(cv::Scalar(0.0f));
        wsum2.setTo(cv::Scalar(0.0f));
        wmean_num.setTo(cv::Scalar(0.0f));

        for (size_t i = 0; i < gpu_tiles.size(); ++i) {
          cv::cuda::GpuMat keep_f32;
          keep_masks[i].convertTo(keep_f32, CV_32F, 1.0 / 255.0);

          cv::cuda::GpuMat weighted_keep;
          cv::cuda::multiply(keep_f32, cv::Scalar(active_weights[i]),
                             weighted_keep);
          cv::cuda::add(wsum, weighted_keep, wsum);

          cv::cuda::GpuMat weighted_value;
          cv::cuda::multiply(gpu_tiles[i], weighted_keep, weighted_value);
          cv::cuda::add(wmean_num, weighted_value, wmean_num);

          cv::cuda::GpuMat weighted_keep_sq;
          cv::cuda::multiply(keep_f32,
                             cv::Scalar(active_weights[i] * active_weights[i]),
                             weighted_keep_sq);
          cv::cuda::add(wsum2, weighted_keep_sq, wsum2);
        }

        cv::cuda::GpuMat wsum_safe;
        cv::cuda::max(wsum, eps, wsum_safe);
        cv::cuda::GpuMat mean;
        cv::cuda::divide(wmean_num, wsum_safe, mean);

        cv::cuda::GpuMat var_num(rows, cols, CV_32F);
        var_num.setTo(cv::Scalar(0.0f));
        for (size_t i = 0; i < gpu_tiles.size(); ++i) {
          cv::cuda::GpuMat diff;
          cv::cuda::subtract(gpu_tiles[i], mean, diff);
          cv::cuda::GpuMat diff_sq;
          cv::cuda::multiply(diff, diff, diff_sq);

          cv::cuda::GpuMat keep_f32;
          keep_masks[i].convertTo(keep_f32, CV_32F, 1.0 / 255.0);
          cv::cuda::GpuMat weighted_keep;
          cv::cuda::multiply(keep_f32, cv::Scalar(active_weights[i]),
                             weighted_keep);
          cv::cuda::GpuMat weighted_diff;
          cv::cuda::multiply(diff_sq, weighted_keep, weighted_diff);
          cv::cuda::add(var_num, weighted_diff, var_num);
        }

        cv::cuda::GpuMat wsum2_over_wsum;
        cv::cuda::divide(wsum2, wsum_safe, wsum2_over_wsum);
        cv::cuda::GpuMat denom;
        cv::cuda::subtract(wsum, wsum2_over_wsum, denom);
        cv::cuda::GpuMat denom_safe;
        cv::cuda::max(denom, eps, denom_safe);
        cv::cuda::GpuMat var;
        cv::cuda::divide(var_num, denom_safe, var);
        cv::cuda::max(var, zeros, var);

        cv::cuda::GpuMat kept_gt_one_mask;
        cv::cuda::compare(valid_count, 1.0f, kept_gt_one_mask, cv::CMP_GT);
        cv::cuda::GpuMat denom_positive_mask;
        cv::cuda::compare(denom, 0.0f, denom_positive_mask, cv::CMP_GT);
        cv::cuda::GpuMat sd;
        cv::cuda::sqrt(var, sd);
        cv::cuda::GpuMat sd_positive_mask;
        cv::cuda::compare(sd, 0.0f, sd_positive_mask, cv::CMP_GT);
        cv::cuda::GpuMat can_continue;
        cv::cuda::bitwise_and(kept_gt_one_mask, denom_positive_mask,
                              can_continue);
        cv::cuda::bitwise_and(can_continue, sd_positive_mask, can_continue);

        cv::cuda::GpuMat sigma_low_sd;
        cv::cuda::GpuMat sigma_high_sd;
        cv::cuda::multiply(sd, cv::Scalar(sigma_low), sigma_low_sd);
        cv::cuda::multiply(sd, cv::Scalar(sigma_high), sigma_high_sd);
        cv::cuda::GpuMat lo;
        cv::cuda::GpuMat hi;
        cv::cuda::subtract(mean, sigma_low_sd, lo);
        cv::cuda::add(mean, sigma_high_sd, hi);

        std::vector<cv::cuda::GpuMat> new_keep_masks;
        new_keep_masks.reserve(keep_masks.size());
        cv::cuda::GpuMat new_valid_count(rows, cols, CV_32F);
        new_valid_count.setTo(cv::Scalar(0.0f));
        for (size_t i = 0; i < gpu_tiles.size(); ++i) {
          cv::cuda::GpuMat ge_lo;
          cv::cuda::GpuMat le_hi;
          cv::cuda::GpuMat in_range;
          cv::cuda::compare(gpu_tiles[i], lo, ge_lo, cv::CMP_GE);
          cv::cuda::compare(gpu_tiles[i], hi, le_hi, cv::CMP_LE);
          cv::cuda::bitwise_and(ge_lo, le_hi, in_range);

          cv::cuda::GpuMat new_keep;
          cv::cuda::bitwise_and(keep_masks[i], in_range, new_keep);
          new_keep_masks.push_back(new_keep);

          cv::cuda::GpuMat new_keep_f32;
          new_keep.convertTo(new_keep_f32, CV_32F, 1.0 / 255.0);
          cv::cuda::add(new_valid_count, new_keep_f32, new_valid_count);
        }

        cv::cuda::GpuMat meets_min;
        cv::cuda::compare(new_valid_count, min_keep, meets_min, cv::CMP_GE);
        cv::cuda::GpuMat update_mask;
        cv::cuda::bitwise_and(active_mask, can_continue, update_mask);
        cv::cuda::GpuMat next_active;
        cv::cuda::bitwise_and(update_mask, meets_min, next_active);

        cv::cuda::GpuMat update_mask_inv;
        cv::cuda::bitwise_not(update_mask, update_mask_inv);
        for (size_t i = 0; i < keep_masks.size(); ++i) {
          cv::cuda::GpuMat old_region;
          cv::cuda::GpuMat new_region;
          cv::cuda::bitwise_and(keep_masks[i], update_mask_inv, old_region);
          cv::cuda::bitwise_and(new_keep_masks[i], update_mask, new_region);
          cv::cuda::bitwise_or(old_region, new_region, keep_masks[i]);
        }
        valid_count = new_valid_count;
        active_mask = next_active;
      }
    }

    cv::cuda::GpuMat final_wsum(rows, cols, CV_32F);
    cv::cuda::GpuMat final_num(rows, cols, CV_32F);
    cv::cuda::GpuMat fallback_wsum(rows, cols, CV_32F);
    cv::cuda::GpuMat fallback_num(rows, cols, CV_32F);
    final_wsum.setTo(cv::Scalar(0.0f));
    final_num.setTo(cv::Scalar(0.0f));
    fallback_wsum.setTo(cv::Scalar(0.0f));
    fallback_num.setTo(cv::Scalar(0.0f));

    for (size_t i = 0; i < gpu_tiles.size(); ++i) {
      cv::cuda::GpuMat keep_f32;
      keep_masks[i].convertTo(keep_f32, CV_32F, 1.0 / 255.0);
      cv::cuda::GpuMat valid_f32;
      valid_masks[i].convertTo(valid_f32, CV_32F, 1.0 / 255.0);

      cv::cuda::GpuMat weighted_keep;
      cv::cuda::multiply(keep_f32, cv::Scalar(active_weights[i]), weighted_keep);
      cv::cuda::GpuMat weighted_valid;
      cv::cuda::multiply(valid_f32, cv::Scalar(active_weights[i]), weighted_valid);
      cv::cuda::add(final_wsum, weighted_keep, final_wsum);
      cv::cuda::add(fallback_wsum, weighted_valid, fallback_wsum);

      cv::cuda::GpuMat weighted_value;
      cv::cuda::multiply(gpu_tiles[i], weighted_keep, weighted_value);
      cv::cuda::GpuMat fallback_value;
      cv::cuda::multiply(gpu_tiles[i], weighted_valid, fallback_value);
      cv::cuda::add(final_num, weighted_value, final_num);
      cv::cuda::add(fallback_num, fallback_value, fallback_num);
    }

    cv::cuda::GpuMat final_wsum_safe;
    cv::cuda::GpuMat fallback_wsum_safe;
    cv::cuda::max(final_wsum, eps, final_wsum_safe);
    cv::cuda::max(fallback_wsum, eps, fallback_wsum_safe);
    cv::cuda::GpuMat final_out;
    cv::cuda::GpuMat fallback_out;
    cv::cuda::divide(final_num, final_wsum_safe, final_out);
    cv::cuda::divide(fallback_num, fallback_wsum_safe, fallback_out);

    cv::cuda::GpuMat zero_wsum_mask;
    cv::cuda::compare(final_wsum, eps_weight, zero_wsum_mask, cv::CMP_LE);
    fallback_out.copyTo(final_out, zero_wsum_mask);
    cv::cuda::GpuMat zero_fallback_mask;
    cv::cuda::compare(fallback_wsum, eps_weight, zero_fallback_mask, cv::CMP_LE);
    final_out.setTo(cv::Scalar(0.0f), zero_fallback_mask);

    out.tile.resize(rows, cols);
    cv::Mat out_host(rows, cols, CV_32F, out.tile.data());
    final_out.download(out_host);
    return true;
  } catch (...) {
    return false;
  }
}

bool cuda_sigma_clip_stack_impl(
    const std::vector<Matrix2Df> &frames, float sigma_low, float sigma_high,
    int max_iters, float min_fraction, Matrix2Df &out) {
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
    std::vector<cv::cuda::GpuMat> gpu_frames;
    std::vector<cv::cuda::GpuMat> keep_masks;
    gpu_frames.reserve(valid.size());
    keep_masks.reserve(valid.size());

    cv::cuda::GpuMat ones(rows, cols, CV_32F);
    cv::cuda::GpuMat zeros(rows, cols, CV_32F);
    cv::cuda::GpuMat initial_count(rows, cols, CV_32F);
    ones.setTo(cv::Scalar(1.0f));
    zeros.setTo(cv::Scalar(0.0f));
    initial_count.setTo(cv::Scalar(0.0f));

    for (const Matrix2Df &frame : valid) {
      cv::Mat host_view(rows, cols, CV_32F,
                        const_cast<float *>(frame.data()));
      cv::cuda::GpuMat gpu_frame;
      gpu_frame.upload(host_view);
      gpu_frames.push_back(gpu_frame);

      cv::cuda::GpuMat valid_mask;
      cv::cuda::compare(gpu_frames.back(), 0.0f, valid_mask, cv::CMP_GT);
      keep_masks.push_back(valid_mask.clone());

      cv::cuda::GpuMat valid_mask_f32;
      valid_mask.convertTo(valid_mask_f32, CV_32F, 1.0 / 255.0);
      cv::cuda::add(initial_count, valid_mask_f32, initial_count);
    }

    cv::Mat initial_count_host(rows, cols, CV_32F);
    cv::Mat min_keep_host(rows, cols, CV_32F);
    initial_count.download(initial_count_host);
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
    min_keep.upload(min_keep_host);

    cv::cuda::GpuMat active_mask;
    cv::cuda::compare(initial_count, 0.0f, active_mask, cv::CMP_GT);

    for (int iter = 0; iter < max_iters; ++iter) {
      cv::cuda::GpuMat sum(rows, cols, CV_32F);
      cv::cuda::GpuMat sumsq(rows, cols, CV_32F);
      cv::cuda::GpuMat count(rows, cols, CV_32F);
      sum.setTo(cv::Scalar(0.0f));
      sumsq.setTo(cv::Scalar(0.0f));
      count.setTo(cv::Scalar(0.0f));

      for (size_t i = 0; i < gpu_frames.size(); ++i) {
        cv::cuda::GpuMat keep_f32;
        keep_masks[i].convertTo(keep_f32, CV_32F, 1.0 / 255.0);

        cv::cuda::GpuMat masked_frame;
        cv::cuda::multiply(gpu_frames[i], keep_f32, masked_frame);
        cv::cuda::add(sum, masked_frame, sum);

        cv::cuda::GpuMat frame_sq;
        cv::cuda::multiply(gpu_frames[i], gpu_frames[i], frame_sq);
        cv::cuda::GpuMat masked_sq;
        cv::cuda::multiply(frame_sq, keep_f32, masked_sq);
        cv::cuda::add(sumsq, masked_sq, sumsq);
        cv::cuda::add(count, keep_f32, count);
      }

      cv::cuda::GpuMat count_denom;
      cv::cuda::max(count, ones, count_denom);

      cv::cuda::GpuMat mean;
      cv::cuda::divide(sum, count_denom, mean);

      cv::cuda::GpuMat sumsq_mean;
      cv::cuda::divide(sumsq, count_denom, sumsq_mean);
      cv::cuda::GpuMat mean_sq;
      cv::cuda::multiply(mean, mean, mean_sq);
      cv::cuda::GpuMat var;
      cv::cuda::subtract(sumsq_mean, mean_sq, var);
      cv::cuda::max(var, zeros, var);

      cv::cuda::GpuMat kept_gt_one_mask;
      cv::cuda::compare(count, 1.0f, kept_gt_one_mask, cv::CMP_GT);
      cv::cuda::GpuMat kept_gt_one_f32;
      kept_gt_one_mask.convertTo(kept_gt_one_f32, CV_32F, 1.0 / 255.0);

      cv::cuda::GpuMat count_minus_one;
      cv::cuda::subtract(count, ones, count_minus_one);
      cv::cuda::max(count_minus_one, ones, count_minus_one);
      cv::cuda::GpuMat factor;
      cv::cuda::divide(count, count_minus_one, factor);

      cv::cuda::GpuMat one_minus_gt_one;
      cv::cuda::subtract(ones, kept_gt_one_f32, one_minus_gt_one);
      cv::cuda::GpuMat factor_when_gt_one;
      cv::cuda::multiply(factor, kept_gt_one_f32, factor_when_gt_one);
      cv::cuda::add(factor_when_gt_one, one_minus_gt_one, factor);
      cv::cuda::multiply(var, factor, var);
      cv::cuda::max(var, zeros, var);

      cv::cuda::GpuMat sd;
      cv::cuda::sqrt(var, sd);

      cv::cuda::GpuMat sd_positive_mask;
      cv::cuda::compare(sd, 0.0f, sd_positive_mask, cv::CMP_GT);
      cv::cuda::GpuMat can_continue;
      cv::cuda::bitwise_and(kept_gt_one_mask, sd_positive_mask, can_continue);

      cv::cuda::GpuMat sigma_low_sd;
      cv::cuda::GpuMat sigma_high_sd;
      cv::cuda::multiply(sd, cv::Scalar(sigma_low), sigma_low_sd);
      cv::cuda::multiply(sd, cv::Scalar(sigma_high), sigma_high_sd);
      cv::cuda::GpuMat lo;
      cv::cuda::GpuMat hi;
      cv::cuda::subtract(mean, sigma_low_sd, lo);
      cv::cuda::add(mean, sigma_high_sd, hi);

      std::vector<cv::cuda::GpuMat> new_keep_masks;
      new_keep_masks.reserve(keep_masks.size());
      cv::cuda::GpuMat new_count(rows, cols, CV_32F);
      new_count.setTo(cv::Scalar(0.0f));

      for (size_t i = 0; i < gpu_frames.size(); ++i) {
        cv::cuda::GpuMat ge_lo;
        cv::cuda::GpuMat le_hi;
        cv::cuda::GpuMat in_range;
        cv::cuda::compare(gpu_frames[i], lo, ge_lo, cv::CMP_GE);
        cv::cuda::compare(gpu_frames[i], hi, le_hi, cv::CMP_LE);
        cv::cuda::bitwise_and(ge_lo, le_hi, in_range);

        cv::cuda::GpuMat new_keep;
        cv::cuda::bitwise_and(keep_masks[i], in_range, new_keep);
        new_keep_masks.push_back(new_keep);

        cv::cuda::GpuMat new_keep_f32;
        new_keep.convertTo(new_keep_f32, CV_32F, 1.0 / 255.0);
        cv::cuda::add(new_count, new_keep_f32, new_count);
      }

      cv::cuda::GpuMat meets_min;
      cv::cuda::compare(new_count, min_keep, meets_min, cv::CMP_GE);
      cv::cuda::GpuMat update_mask;
      cv::cuda::bitwise_and(active_mask, can_continue, update_mask);
      cv::cuda::GpuMat next_active;
      cv::cuda::bitwise_and(update_mask, meets_min, next_active);

      cv::cuda::GpuMat update_mask_inv;
      cv::cuda::bitwise_not(update_mask, update_mask_inv);
      for (size_t i = 0; i < keep_masks.size(); ++i) {
        cv::cuda::GpuMat old_region;
        cv::cuda::GpuMat new_region;
        cv::cuda::bitwise_and(keep_masks[i], update_mask_inv, old_region);
        cv::cuda::bitwise_and(new_keep_masks[i], update_mask, new_region);
        cv::cuda::bitwise_or(old_region, new_region, keep_masks[i]);
      }
      active_mask = next_active;
    }

    cv::cuda::GpuMat final_sum(rows, cols, CV_32F);
    cv::cuda::GpuMat final_count(rows, cols, CV_32F);
    final_sum.setTo(cv::Scalar(0.0f));
    final_count.setTo(cv::Scalar(0.0f));
    for (size_t i = 0; i < gpu_frames.size(); ++i) {
      cv::cuda::GpuMat keep_f32;
      keep_masks[i].convertTo(keep_f32, CV_32F, 1.0 / 255.0);
      cv::cuda::GpuMat masked_frame;
      cv::cuda::multiply(gpu_frames[i], keep_f32, masked_frame);
      cv::cuda::add(final_sum, masked_frame, final_sum);
      cv::cuda::add(final_count, keep_f32, final_count);
    }

    cv::cuda::GpuMat final_count_denom;
    cv::cuda::max(final_count, ones, final_count_denom);
    cv::cuda::GpuMat final_out;
    cv::cuda::divide(final_sum, final_count_denom, final_out);
    cv::cuda::GpuMat dead_mask;
    cv::cuda::compare(final_count, 0.0f, dead_mask, cv::CMP_LE);
    final_out.setTo(cv::Scalar(0.0f), dead_mask);

    out.resize(rows, cols);
    cv::Mat out_host(rows, cols, CV_32F, out.data());
    final_out.download(out_host);
    return true;
  } catch (...) {
    return false;
  }
}
#endif

bool auto_backend_requested(const std::string &name) {
  return core::to_lower(name) == "auto";
}

AccelerationBackend choose_auto_backend(AccelerationPhase phase,
                                        bool tile_compile_with_cuda,
                                        bool opencv_cuda_runtime) {
  const bool opencv_cuda_headers = opencv_cuda_headers_available(phase);
  const AccelerationBackend candidates[] = {
      AccelerationBackend::cuda,
      AccelerationBackend::opencv_cuda,
      AccelerationBackend::cpu,
  };
  for (AccelerationBackend candidate : candidates) {
    if (!phase_supports_backend(phase, candidate)) {
      continue;
    }
    if (missing_backend_reason(candidate, tile_compile_with_cuda,
                               opencv_cuda_headers,
                               opencv_cuda_runtime)
        .empty()) {
      return candidate;
    }
  }
  return AccelerationBackend::cpu;
}

} // namespace

std::string acceleration_phase_name(AccelerationPhase phase) {
  switch (phase) {
  case AccelerationPhase::prewarp:
    return "PREWARP";
  case AccelerationPhase::tile_reconstruction:
    return "TILE_RECONSTRUCTION";
  case AccelerationPhase::stacking:
    return "STACKING";
  }
  return "UNKNOWN";
}

std::string acceleration_backend_name(AccelerationBackend backend) {
  switch (backend) {
  case AccelerationBackend::cpu:
    return "cpu";
  case AccelerationBackend::opencv_cuda:
    return "opencv_cuda";
  case AccelerationBackend::cuda:
    return "cuda";
  }
  return "cpu";
}

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
  if (normalized == "cuda") {
    backend_out = AccelerationBackend::cuda;
    return true;
  }
  return false;
}

AccelerationSelection select_acceleration_backend(
    const std::string &requested_backend_name, AccelerationPhase phase) {
  AccelerationSelection selection;
  selection.phase = phase;
  selection.tile_compile_with_cuda = TILE_COMPILE_WITH_CUDA != 0;
  selection.opencv_cuda_headers = opencv_cuda_headers_available(phase);
  selection.opencv_cuda_runtime = opencv_cuda_runtime_available();

  selection.requested_name = core::to_lower(requested_backend_name);
  if (selection.requested_name.empty()) {
    selection.requested_name = "auto";
  }
  selection.auto_requested = auto_backend_requested(selection.requested_name);

  if (selection.auto_requested) {
    selection.selected =
        choose_auto_backend(phase, selection.tile_compile_with_cuda,
                            selection.opencv_cuda_runtime);
    selection.requested = selection.selected;
    selection.gpu_requested = selection.selected != AccelerationBackend::cpu;
    selection.using_gpu = selection.selected != AccelerationBackend::cpu;
    return selection;
  }

  AccelerationBackend requested_backend = AccelerationBackend::cpu;
  if (!parse_acceleration_backend(selection.requested_name, requested_backend)) {
    selection.request_honored = false;
    selection.fallback_reason = "invalid_requested_backend";
    selection.requested = AccelerationBackend::cpu;
    selection.selected = AccelerationBackend::cpu;
    return selection;
  }

  selection.requested = requested_backend;
  selection.selected = requested_backend;
  selection.gpu_requested = requested_backend != AccelerationBackend::cpu;

  const std::string missing_reason =
      missing_backend_reason(requested_backend, selection.tile_compile_with_cuda,
                             selection.opencv_cuda_headers,
                             selection.opencv_cuda_runtime);
  if (!missing_reason.empty()) {
    selection.selected = AccelerationBackend::cpu;
    selection.request_honored = false;
    selection.fallback_reason = missing_reason;
    return selection;
  }

  if (!phase_supports_backend(phase, requested_backend)) {
    selection.selected = AccelerationBackend::cpu;
    selection.request_honored = false;
    selection.fallback_reason =
        unsupported_phase_reason(requested_backend, phase);
    return selection;
  }

  selection.using_gpu = requested_backend != AccelerationBackend::cpu;
  return selection;
}

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
  };
  if (!selection.fallback_reason.empty()) {
    out["fallback_reason"] = selection.fallback_reason;
  }
  return out;
}

std::string acceleration_selection_summary(
    const AccelerationSelection &selection) {
  std::ostringstream oss;
  oss << "requested=" << selection.requested_name
      << " selected=" << acceleration_backend_name(selection.selected)
      << " execution=" << (selection.using_gpu ? "gpu" : "cpu");
  if (selection.auto_requested) {
    oss << " auto=true";
  }
  if (!selection.request_honored && !selection.fallback_reason.empty()) {
    oss << " reason=" << selection.fallback_reason;
  }
  if (selection.using_gpu) {
    if (selection.selected == AccelerationBackend::opencv_cuda) {
      oss << " gpu_backend=opencv_cuda";
    } else if (selection.selected == AccelerationBackend::cuda) {
      oss << " gpu_backend=cuda";
    }
  }
  return oss.str();
}

DeviceFrame make_device_frame(int rows, int cols, int channels) {
  DeviceFrame frame;
  frame.rows = rows;
  frame.cols = cols;
  frame.channels = std::max(1, channels);
  frame.bytes = safe_frame_bytes(rows, cols, frame.channels);
  return frame;
}

DeviceFrameBatch make_device_frame_batch(size_t batch_size, int rows, int cols,
                                         int channels) {
  DeviceFrameBatch batch;
  batch.batch_size = batch_size;
  batch.frame = make_device_frame(rows, cols, channels);
  batch.total_bytes = batch.frame.bytes * batch.batch_size;
  return batch;
}

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

json device_frame_to_json(const DeviceFrame &frame) {
  return {
      {"rows", frame.rows},
      {"cols", frame.cols},
      {"channels", frame.channels},
      {"bytes", frame.bytes},
  };
}

json device_frame_batch_to_json(const DeviceFrameBatch &batch) {
  return {
      {"batch_size", batch.batch_size},
      {"frame", device_frame_to_json(batch.frame)},
      {"total_bytes", batch.total_bytes},
  };
}

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

AccelerationOps::AccelerationOps(AccelerationSelection selection)
    : selection_(std::move(selection)) {}

struct AccelerationOps::OverlapAddState {
  int rows = 0;
  int cols = 0;
#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS && TILE_COMPILE_HAS_OPENCV_CUDA_ARITHM
  cv::cuda::GpuMat gpu_mat;
#endif
};

bool AccelerationOps::warp_affine_frame(Matrix2Df img, const WarpMatrix &warp,
                                        ColorMode mode, int canvas_height,
                                        int canvas_width, int offset_x,
                                        int offset_y, Matrix2Df &warped_out,
                                        std::vector<uint8_t> *valid_mask_out,
                                        bool *has_data_out) const {
  auto update_valid_outputs = [&](const Matrix2Df &matrix) {
    if (valid_mask_out == nullptr && has_data_out == nullptr) {
      return;
    }
    const size_t pixel_count =
        static_cast<size_t>(std::max<Eigen::Index>(0, matrix.size()));
    if (valid_mask_out != nullptr) {
      valid_mask_out->assign(pixel_count, 0u);
    }
    bool has_data = false;
    for (size_t i = 0; i < pixel_count; ++i) {
      if (matrix.data()[static_cast<Eigen::Index>(i)] > 0.0f) {
        has_data = true;
        if (valid_mask_out != nullptr) {
          (*valid_mask_out)[i] = 1u;
        }
      }
    }
    if (has_data_out != nullptr) {
      *has_data_out = has_data;
    }
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
  if (warp_is_identity(warp)) {
    if (canvas_width > src_width || canvas_height > src_height) {
      warped_out = Matrix2Df::Zero(canvas_height, canvas_width);
      const int dst_y = std::max(0, offset_y);
      const int dst_x = std::max(0, offset_x);
      const int copy_h = std::min(src_height, canvas_height - dst_y);
      const int copy_w = std::min(src_width, canvas_width - dst_x);
      if (copy_h > 0 && copy_w > 0) {
        warped_out.block(dst_y, dst_x, copy_h, copy_w) =
            img.block(0, 0, copy_h, copy_w);
      }
    } else {
      warped_out = std::move(img);
    }
    update_valid_outputs(warped_out);
    return warped_out.size() > 0;
  }

#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS && TILE_COMPILE_HAS_OPENCV_CUDA_WARPING
  if (selection_.selected == AccelerationBackend::opencv_cuda &&
      selection_.phase == AccelerationPhase::prewarp) {
    if (mode == ColorMode::OSC) {
      if (cuda_warp_cfa_mosaic(img, warp, canvas_height, canvas_width,
                               warped_out)) {
        update_valid_outputs(warped_out);
        return true;
      }
    } else {
      const cv::Mat src(static_cast<int>(img.rows()), static_cast<int>(img.cols()),
                        CV_32F, const_cast<float *>(img.data()));
      const cv::Mat warp_matrix = warp_matrix_to_cv(warp);
      cv::Mat dst;
      cv::Mat valid_mask_host;
      int nonzero_count = -1;
      if (cuda_warp_affine_impl(src, warp_matrix,
                                cv::Size(canvas_width, canvas_height), dst,
                                (valid_mask_out != nullptr) ? &valid_mask_host
                                                            : nullptr,
                                (has_data_out != nullptr) ? &nonzero_count
                                                          : nullptr)) {
        warped_out.resize(canvas_height, canvas_width);
        std::memcpy(warped_out.data(), dst.data,
                    static_cast<size_t>(warped_out.size()) * sizeof(float));
        if (valid_mask_out != nullptr) {
          valid_mask_out->assign(static_cast<size_t>(canvas_height) *
                                     static_cast<size_t>(canvas_width),
                                 0u);
          if (!valid_mask_host.empty()) {
            for (int y = 0; y < valid_mask_host.rows; ++y) {
              const uchar *row = valid_mask_host.ptr<uchar>(y);
              for (int x = 0; x < valid_mask_host.cols; ++x) {
                (*valid_mask_out)[static_cast<size_t>(y) *
                                      static_cast<size_t>(canvas_width) +
                                  static_cast<size_t>(x)] =
                    (row[x] != 0) ? 1u : 0u;
              }
            }
          }
        }
        if (has_data_out != nullptr) {
          *has_data_out = nonzero_count > 0;
        }
        return true;
      }
    }
  }
#endif

  warped_out =
      image::apply_global_warp(img, warp, mode, canvas_height, canvas_width);
  update_valid_outputs(warped_out);
  return warped_out.size() > 0;
}

reconstruction::WeightedTileResult AccelerationOps::sigma_clip_reduce(
    const std::vector<Matrix2Df> &tiles, const std::vector<float> &weights,
    float sigma_low, float sigma_high, int max_iters, float min_fraction,
    float eps_weight) const {
#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS && TILE_COMPILE_HAS_OPENCV_CUDA_ARITHM
  if (selection_.selected == AccelerationBackend::opencv_cuda &&
      (selection_.phase == AccelerationPhase::tile_reconstruction ||
       selection_.phase == AccelerationPhase::stacking)) {
    reconstruction::WeightedTileResult gpu_out;
    if (cuda_sigma_clip_weighted_tile_impl(
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

Matrix2Df AccelerationOps::sigma_clip_stack(const std::vector<Matrix2Df> &frames,
                                            float sigma_low,
                                            float sigma_high, int max_iters,
                                            float min_fraction) const {
#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS && TILE_COMPILE_HAS_OPENCV_CUDA_ARITHM
  if (selection_.selected == AccelerationBackend::opencv_cuda &&
      selection_.phase == AccelerationPhase::stacking) {
    Matrix2Df gpu_out;
    if (cuda_sigma_clip_stack_impl(frames, sigma_low, sigma_high, max_iters,
                                   min_fraction, gpu_out)) {
      return gpu_out;
    }
  }
#endif
  return reconstruction::sigma_clip_stack(frames, sigma_low, sigma_high,
                                          max_iters, min_fraction);
}

void AccelerationOps::overlap_add(
    const Matrix2Df &tile, const Tile &tile_bounds,
    const std::vector<float> &hann_x, const std::vector<float> &hann_y,
    const std::vector<uint8_t> &common_valid_mask, int canvas_width,
    Matrix2Df &accum, Matrix2Df &weight_sum, bool accumulate_weight) const {
  if (tile.rows() != tile_bounds.height || tile.cols() != tile_bounds.width) {
    return;
  }

#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS && TILE_COMPILE_HAS_OPENCV_CUDA_ARITHM
  if (selection_.selected == AccelerationBackend::opencv_cuda &&
      selection_.phase == AccelerationPhase::tile_reconstruction) {
    auto ensure_state = [&](Matrix2Df &host_matrix)
        -> std::shared_ptr<OverlapAddState> {
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

    const int x0 = std::max(0, tile_bounds.x);
    const int y0 = std::max(0, tile_bounds.y);
    const int clip_w = std::min<int>(tile.cols(), accum.cols() - x0);
    const int clip_h = std::min<int>(tile.rows(), accum.rows() - y0);
    if (clip_w <= 0 || clip_h <= 0) {
      return;
    }

    Matrix2Df weighted_tile = Matrix2Df::Zero(clip_h, clip_w);
    Matrix2Df weighted_mask;
    if (accumulate_weight) {
      weighted_mask = Matrix2Df::Zero(clip_h, clip_w);
    }

    bool has_valid_pixels = false;
    for (int yy = 0; yy < clip_h; ++yy) {
      const int iy = y0 + yy;
      for (int xx = 0; xx < clip_w; ++xx) {
        const int ix = x0 + xx;
        const size_t common_idx =
            static_cast<size_t>(iy) * static_cast<size_t>(canvas_width) +
            static_cast<size_t>(ix);
        if (common_idx >= common_valid_mask.size() ||
            common_valid_mask[common_idx] == 0) {
          continue;
        }
        const float tile_value = tile(yy, xx);
        if (!(std::isfinite(tile_value) && tile_value > 0.0f)) {
          continue;
        }
        const float win =
            hann_y[static_cast<size_t>(yy)] * hann_x[static_cast<size_t>(xx)];
        weighted_tile(yy, xx) = tile_value * win;
        if (accumulate_weight) {
          weighted_mask(yy, xx) = win;
        }
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
    cv::cuda::GpuMat accum_roi(accum_state->gpu_mat, roi);
    cv::Mat weighted_tile_host(clip_h, clip_w, CV_32F, weighted_tile.data());
    cv::cuda::GpuMat weighted_tile_gpu;
    weighted_tile_gpu.upload(weighted_tile_host);
    cv::cuda::add(accum_roi, weighted_tile_gpu, accum_roi);

    if (accumulate_weight) {
      auto weight_state = ensure_state(weight_sum);
      if (weight_state && !weight_state->gpu_mat.empty()) {
        cv::cuda::GpuMat weight_roi(weight_state->gpu_mat, roi);
        cv::Mat weighted_mask_host(clip_h, clip_w, CV_32F, weighted_mask.data());
        cv::cuda::GpuMat weighted_mask_gpu;
        weighted_mask_gpu.upload(weighted_mask_host);
        cv::cuda::add(weight_roi, weighted_mask_gpu, weight_roi);
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
      const size_t common_idx =
          static_cast<size_t>(iy) * static_cast<size_t>(canvas_width) +
          static_cast<size_t>(ix);
      if (common_idx >= common_valid_mask.size() ||
          common_valid_mask[common_idx] == 0) {
        continue;
      }
      const float tile_value = tile(yy, xx);
      if (!(std::isfinite(tile_value) && tile_value > 0.0f)) {
        continue;
      }
      const float win =
          hann_y[static_cast<size_t>(yy)] * hann_x[static_cast<size_t>(xx)];
      accum(iy, ix) += tile_value * win;
      if (accumulate_weight) {
        weight_sum(iy, ix) += win;
      }
    }
  }
}

bool AccelerationOps::normalize_overlap_accum(Matrix2Df &accum,
                                              Matrix2Df &weight_sum,
                                              float eps_weight,
                                              float invalid_value) const {
#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS && TILE_COMPILE_HAS_OPENCV_CUDA_ARITHM
  if (selection_.selected != AccelerationBackend::opencv_cuda ||
      selection_.phase != AccelerationPhase::tile_reconstruction) {
    return false;
  }

  auto accum_it = overlap_add_states_.find(&accum);
  auto weight_it = overlap_add_states_.find(&weight_sum);
  if (accum_it == overlap_add_states_.end() || weight_it == overlap_add_states_.end() ||
      !accum_it->second || !weight_it->second) {
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
  cv::cuda::compare(weight_state.gpu_mat, eps_weight, invalid_mask, cv::CMP_LE);
  accum_state.gpu_mat.setTo(cv::Scalar(invalid_value), invalid_mask);
  return true;
#else
  (void)accum;
  (void)weight_sum;
  (void)eps_weight;
  (void)invalid_value;
  return false;
#endif
}

void AccelerationOps::flush_overlap_state(Matrix2Df &accum,
                                          Matrix2Df &weight_sum) const {
#if TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS && TILE_COMPILE_HAS_OPENCV_CUDA_ARITHM
  if (selection_.selected != AccelerationBackend::opencv_cuda ||
      selection_.phase != AccelerationPhase::tile_reconstruction) {
    return;
  }

  auto flush_one = [&](Matrix2Df &host_matrix) {
    auto it = overlap_add_states_.find(&host_matrix);
    if (it == overlap_add_states_.end() || !it->second) {
      return;
    }
    OverlapAddState &state = *it->second;
    if (state.rows > 0 && state.cols > 0 && !state.gpu_mat.empty()) {
      if (host_matrix.rows() != state.rows || host_matrix.cols() != state.cols) {
        host_matrix.resize(state.rows, state.cols);
      }
      cv::Mat host_view(state.rows, state.cols, CV_32F, host_matrix.data());
      state.gpu_mat.download(host_view);
    }
    overlap_add_states_.erase(it);
  };

  flush_one(accum);
  flush_one(weight_sum);
#else
  (void)accum;
  (void)weight_sum;
#endif
}

} // namespace tile_compile::core
