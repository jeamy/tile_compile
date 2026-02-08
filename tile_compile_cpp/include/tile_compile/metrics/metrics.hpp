#pragma once

#include "tile_compile/core/types.hpp"
#include <opencv2/core.hpp>
#include <vector>

namespace tile_compile::metrics {

FrameMetrics calculate_frame_metrics(const Matrix2Df& frame);

VectorXf calculate_global_weights(const std::vector<FrameMetrics>& metrics,
                                 float w_bg, float w_noise, float w_grad,
                                 float clamp_lo = -3.0f, float clamp_hi = 3.0f);

// Estimate FWHM from a small image patch (e.g. around a detected star)
float estimate_fwhm_from_patch(const cv::Mat& patch);

// Build a binary background mask via sigma-clipping
cv::Mat1b build_background_mask_sigma_clip(const cv::Mat& frame,
                                           float k_sigma, int dilate_radius);

} // namespace tile_compile::metrics
