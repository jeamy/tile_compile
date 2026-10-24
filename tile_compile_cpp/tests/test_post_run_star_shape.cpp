#include "tile_compile/reconstruction/post_run_star_shape.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

using namespace tile_compile;
using namespace tile_compile::reconstruction;

namespace {
Matrix2Df star_field(int width, int height, double sigma) {
  Matrix2Df image(height, width);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      double value = 100.0;
      for (int cy = 30; cy < height - 20; cy += 45)
        for (int cx = 30; cx < width - 20; cx += 45) {
          const double dx = x - cx, dy = y - cy;
          value += 1000.0 * std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
        }
      image(y, x) = static_cast<float>(value);
    }
  return image;
}
}  // namespace

TEST_CASE("post-run star-shape diagnostic measures fixed control positions") {
  constexpr int width = 420, height = 360;
  const auto before = star_field(width, height, 2.0);
  const auto after = star_field(width, height, 2.6);
  const std::vector<uint8_t> valid(static_cast<size_t>(width) * height, 1);

  const auto measured = measure_post_run_star_shape(before, after, valid);
  INFO(measured.dump());
  REQUIRE(measured["status"] == "measured");
  REQUIRE(measured["metric"] == "matched_star_fwhm_ratio_after_over_before");
  REQUIRE(measured["matched_stars"].get<int>() >= 20);
  REQUIRE(measured["median_ratio"].get<double>() > 1.05);
  REQUIRE(measured["interpretation"] == "descriptive_measurement_only");
}

TEST_CASE("post-run star-shape diagnostic fails closed with insufficient support") {
  Matrix2Df image(80, 80);
  image.setZero();
  const std::vector<uint8_t> valid(80 * 80, 0);
  const auto result = measure_post_run_star_shape(image, image, valid);
  REQUIRE(result["status"] == "not_applicable");
  REQUIRE(result["reason_code"] == "insufficient_common_support");
}
