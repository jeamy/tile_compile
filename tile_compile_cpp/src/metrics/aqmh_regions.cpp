#include "tile_compile/metrics/aqmh_regions.hpp"

#include "tile_compile/metrics/aqmh_eps.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace tile_compile::metrics {

std::vector<AqmhRegion> extract_aqmh_regions(
    const Matrix2Df &quality_map, const std::vector<uint8_t> &source_valid_mask,
    float q, int radius) {
  const int rows = quality_map.rows(), cols = quality_map.cols();
  if (rows <= 0 || cols <= 0 ||
      (!source_valid_mask.empty() &&
       source_valid_mask.size() != static_cast<size_t>(rows * cols))) return {};
  std::vector<float> values;
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x) {
      const size_t i = static_cast<size_t>(y * cols + x);
      if ((source_valid_mask.empty() || source_valid_mask[i]) &&
          std::isfinite(quality_map(y, x))) values.push_back(quality_map(y, x));
    }
  if (values.empty()) return {};
  std::sort(values.begin(), values.end());
  const double pos = std::clamp<double>(q, 0.0, 1.0) * (values.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = static_cast<size_t>(std::ceil(pos));
  const float t = static_cast<float>(pos - static_cast<double>(lo));
  const float threshold = values[lo] * (1.0f - t) + values[hi] * t;
  cv::Mat binary(rows, cols, CV_8U, cv::Scalar(0));
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x) {
      const size_t i = static_cast<size_t>(y * cols + x);
      if ((source_valid_mask.empty() || source_valid_mask[i]) &&
          quality_map(y, x) >= threshold) binary.at<uint8_t>(y, x) = 255;
    }
  if (radius > 0) {
    const int d = 2 * radius + 1;
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {d, d});
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
  }
  cv::Mat labels, stats, centroids;
  const int count = cv::connectedComponentsWithStats(binary, labels, stats,
                                                      centroids, 8, CV_32S);
  std::vector<AqmhRegion> regions;
  for (int label = 1; label < count; ++label) {
    const int area = stats.at<int>(label, cv::CC_STAT_AREA);
    cv::Mat component = labels == label;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(component, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    double perimeter = 0.0, sum = 0.0;
    for (const auto &c : contours) perimeter += cv::arcLength(c, true);
    for (int y = 0; y < rows; ++y)
      for (int x = 0; x < cols; ++x)
        if (labels.at<int>(y, x) == label) sum += quality_map(y, x);
    AqmhRegion region;
    region.label = label;
    region.area = area;
    region.mean_quality = area > 0 ? static_cast<float>(sum / area) : 0.0f;
    region.compactness = perimeter > 0.0
        ? static_cast<float>(4.0 * std::acos(-1.0) * area / (perimeter * perimeter))
        : 0.0f;
    region.score = region.mean_quality * std::log1p(static_cast<float>(area));
    regions.push_back(region);
  }
  std::sort(regions.begin(), regions.end(), [](const auto &a, const auto &b) {
    return a.score != b.score ? a.score > b.score : a.label < b.label;
  });
  return regions;
}

} // namespace tile_compile::metrics
