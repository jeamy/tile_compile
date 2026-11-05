#pragma once

// Helpers shared verbatim by chroma_denoise.cpp and luma_denoise.cpp. Internal
// to the reconstruction library (not an installed header).

#include "tile_compile/core/types.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace tile_compile::reconstruction::denoise_detail {

float robust_sigma_mad_from_mat(const cv::Mat& m,
                                const cv::Mat& valid = cv::Mat());

float percentile_from_mat(const cv::Mat& m, float p,
                          const cv::Mat& valid = cv::Mat());

cv::Mat build_valid_mask(const cv::Mat& r, const cv::Mat& g, const cv::Mat& b,
                         const std::vector<std::uint8_t>* supplied);

void copy_to_matrix(const cv::Mat& src, Matrix2Df& dst);

float high_frequency_sigma(const cv::Mat& src, const cv::Mat& valid);

cv::Mat masked_gaussian_blur(const cv::Mat& src, const cv::Mat& valid,
                             double sigma);

}  // namespace tile_compile::reconstruction::denoise_detail
