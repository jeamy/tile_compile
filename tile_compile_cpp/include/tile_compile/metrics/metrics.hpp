#pragma once

#include "tile_compile/core/types.hpp"
#include <opencv2/core.hpp>
#include <vector>

namespace tile_compile::metrics {

FrameMetrics calculate_frame_metrics(const Matrix2Df& frame);

VectorXf calculate_global_weights(const std::vector<FrameMetrics>& metrics,
                                 float w_bg, float w_noise, float w_grad,
                                 float clamp_lo = -3.0f, float clamp_hi = 3.0f,
                                 bool adaptive_weights = false,
                                 float weight_exponent_scale = 1.0f);

// Estimate FWHM from a small image patch (e.g. around a detected star)
float estimate_fwhm_from_patch(const cv::Mat& patch);

// Build a binary background mask via sigma-clipping
cv::Mat1b build_background_mask_sigma_clip(const cv::Mat& frame,
                                           float k_sigma, int dilate_radius);

// Measure median FWHM from an image by detecting star-like features and
// fitting Gaussian profiles. Returns 0 if fewer than min_stars are found.
float measure_fwhm_from_image(const Matrix2Df& img, int max_corners = 400,
                              int patch_radius = 10, size_t min_stars = 25);

// Per-frame star metrics (Siril-style diagnostics)
struct FrameStarMetrics {
    float fwhm;         // median FWHM (px)
    float fwhm_x;       // median FWHM in X direction (px)
    float fwhm_y;       // median FWHM in Y direction (px)
    float roundness;     // median roundness = fwhm_y / fwhm_x
    float wfwhm;        // weighted FWHM = fwhm * (n_ref_stars / n_stars)
    int   star_count;    // number of detected stars with valid FWHM
};

// Measure per-frame star metrics: FWHM, roundness, wFWHM, star count.
// ref_star_count is the star count of the reference frame (for wFWHM).
// If ref_star_count <= 0, wfwhm = fwhm.
FrameStarMetrics measure_frame_stars(const Matrix2Df& img,
                                     int ref_star_count = 0,
                                     int max_corners = 400,
                                     int patch_radius = 10);

} // namespace tile_compile::metrics
