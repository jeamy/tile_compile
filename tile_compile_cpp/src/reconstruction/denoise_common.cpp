#include "denoise_common.hpp"

#include "tile_compile/core/utils.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace tile_compile::reconstruction::denoise_detail {

float robust_sigma_mad_from_mat(const cv::Mat& m,
                                const cv::Mat& valid) {
  if (m.empty()) return 0.0f;
  std::vector<float> vals;
  vals.reserve(m.total());
  for (int y = 0; y < m.rows; ++y) {
    const float* row = m.ptr<float>(y);
    const std::uint8_t* vrow = valid.empty() ? nullptr : valid.ptr<std::uint8_t>(y);
    for (int x = 0; x < m.cols; ++x)
      if ((!vrow || vrow[x] != 0) && std::isfinite(row[x])) vals.push_back(row[x]);
  }
  return tile_compile::core::robust_sigma_mad(vals);
}

float percentile_from_mat(const cv::Mat& m, float p,
                          const cv::Mat& valid) {
  if (m.empty()) return 0.0f;
  std::vector<float> vals;
  vals.reserve(m.total());
  for (int y = 0; y < m.rows; ++y) {
    const float* row = m.ptr<float>(y);
    const std::uint8_t* vrow = valid.empty() ? nullptr : valid.ptr<std::uint8_t>(y);
    for (int x = 0; x < m.cols; ++x)
      if ((!vrow || vrow[x] != 0) && std::isfinite(row[x])) vals.push_back(row[x]);
  }
  if (vals.empty()) return 0.0f;
  return tile_compile::core::percentile_of(vals, p);
}

cv::Mat build_valid_mask(const cv::Mat& r, const cv::Mat& g, const cv::Mat& b,
                         const std::vector<std::uint8_t>* supplied) {
  cv::Mat valid(r.size(), CV_8U, cv::Scalar(0));
  const bool have_supplied = supplied && supplied->size() == r.total();
  for (int y = 0; y < r.rows; ++y) {
    const float* rr = r.ptr<float>(y);
    const float* gg = g.ptr<float>(y);
    const float* bb = b.ptr<float>(y);
    std::uint8_t* vv = valid.ptr<std::uint8_t>(y);
    for (int x = 0; x < r.cols; ++x) {
      const size_t i = static_cast<size_t>(y) * r.cols + x;
      vv[x] = static_cast<std::uint8_t>(
          (!have_supplied || (*supplied)[i] != 0) && std::isfinite(rr[x]) &&
          std::isfinite(gg[x]) && std::isfinite(bb[x]));
    }
  }
  return valid;
}

void copy_to_matrix(const cv::Mat& src, Matrix2Df& dst) {
  dst.resize(src.rows, src.cols);
  cv::Mat view(src.rows, src.cols, CV_32F, dst.data());
  src.copyTo(view);
}

float high_frequency_sigma(const cv::Mat& src, const cv::Mat& valid) {
  cv::Mat valid_f;
  valid.convertTo(valid_f, CV_32F);
  cv::Mat numerator, denominator, low;
  cv::GaussianBlur(src.mul(valid_f), numerator, cv::Size(0, 0), 0.75, 0.75,
                   cv::BORDER_REFLECT_101);
  cv::GaussianBlur(valid_f, denominator, cv::Size(0, 0), 0.75, 0.75,
                   cv::BORDER_REFLECT_101);
  cv::divide(numerator, denominator + 1.0e-12f, low);
  return robust_sigma_mad_from_mat(src - low, valid);
}

cv::Mat masked_gaussian_blur(const cv::Mat& src, const cv::Mat& valid,
                             double sigma) {
  cv::Mat valid_f;
  valid.convertTo(valid_f, CV_32F);
  cv::Mat numerator, denominator, out;
  cv::GaussianBlur(src.mul(valid_f), numerator, cv::Size(0, 0), sigma, sigma,
                   cv::BORDER_REFLECT_101);
  cv::GaussianBlur(valid_f, denominator, cv::Size(0, 0), sigma, sigma,
                   cv::BORDER_REFLECT_101);
  cv::divide(numerator, denominator + 1.0e-12f, out);
  src.copyTo(out, denominator < 1.0e-6f);
  return out;
}

}  // namespace tile_compile::reconstruction::denoise_detail
