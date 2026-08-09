#if __has_include(<catch2/catch_test_macros.hpp>)
#include "tile_compile/metrics/linearity.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>

namespace {

tile_compile::Matrix2Df make_linear_raw_like_frame(const std::string &kind) {
  using tile_compile::Matrix2Df;

  Matrix2Df img(144, 256);
  for (int y = 0; y < img.rows(); ++y) {
    for (int x = 0; x < img.cols(); ++x) {
      float low_freq = 0.0f;
      if (kind == "nebula") {
        const float dx = (static_cast<float>(x) - 0.55f * img.cols()) /
                         (0.28f * img.cols());
        const float dy = (static_cast<float>(y) - 0.48f * img.rows()) /
                         (0.20f * img.rows());
        low_freq += 18.0f * std::exp(-(dx * dx + dy * dy));
        low_freq += 8.0f * std::sin(static_cast<float>(x) * 0.025f);
        low_freq += 5.0f * std::cos(static_cast<float>(y) * 0.034f);
      } else if (kind == "galaxy_core") {
        const float dx = (static_cast<float>(x) - 0.50f * img.cols()) /
                         (0.11f * img.cols());
        const float dy = (static_cast<float>(y) - 0.50f * img.rows()) /
                         (0.11f * img.rows());
        const float r = std::sqrt(dx * dx + dy * dy);
        low_freq += 160.0f / (1.0f + 8.0f * r);
      } else if (kind != "empty") {
        low_freq += 1.6f * std::sin(static_cast<float>(x) * 0.035f);
        low_freq += 1.1f * std::cos(static_cast<float>(y) * 0.041f);
      }
      const float deterministic_noise =
          static_cast<float>(((x * 17 + y * 31) % 11) - 5) * 0.22f;
      img(y, x) = 220.0f + low_freq + deterministic_noise;
    }
  }

  int star_count = 0;
  if (kind == "star_rich") {
    star_count = 80;
  } else if (kind == "nebula" || kind == "galaxy_core") {
    star_count = 35;
  }
  for (int i = 0; i < star_count; ++i) {
    const int x = (i * 47 + 13) % img.cols();
    const int y = (i * 29 + 7) % img.rows();
    const float peak = 35.0f + static_cast<float>((i * 19) % 220);
    img(y, x) += peak;
    if (x > 0) img(y, x - 1) += peak * 0.20f;
    if (x + 1 < img.cols()) img(y, x + 1) += peak * 0.20f;
    if (y > 0) img(y - 1, x) += peak * 0.20f;
    if (y + 1 < img.rows()) img(y + 1, x) += peak * 0.20f;
  }
  return img;
}

tile_compile::Matrix2Df hard_stretch_with_black_clip(const tile_compile::Matrix2Df &src) {
  tile_compile::Matrix2Df out(src.rows(), src.cols());
  float min_v = src.minCoeff();
  float max_v = src.maxCoeff();
  const float range = std::max(1.0e-6f, max_v - min_v);
  for (int y = 0; y < src.rows(); ++y) {
    for (int x = 0; x < src.cols(); ++x) {
      float v = (src(y, x) - min_v) / range;
      v = std::max(0.0f, v - 0.08f) / 0.92f;
      v = std::sqrt(std::clamp(v, 0.0f, 1.0f));
      out(y, x) = 4095.0f * v;
    }
  }
  return out;
}

}  // namespace

TEST_CASE("linearity accepts linear raw-like frames independent of object") {
  for (const std::string kind :
       {"empty", "star_rich", "nebula", "galaxy_core"}) {
    const auto img = make_linear_raw_like_frame(kind);
    const auto result =
        tile_compile::metrics::validate_linearity_frame(img, "strict");

    CAPTURE(kind);
    CAPTURE(result.skewness);
    CAPTURE(result.kurtosis);
    CAPTURE(result.variance_coeff);
    CAPTURE(result.energy_ratio);
    CAPTURE(result.gradient_consistency);
    REQUIRE(result.moment_ok);
    REQUIRE(result.is_linear);
  }
}

TEST_CASE("linearity flags obvious clipped stretched frames") {
  const auto linear = make_linear_raw_like_frame("star_rich");
  const auto stretched = hard_stretch_with_black_clip(linear);

  const auto result =
      tile_compile::metrics::validate_linearity_frame(stretched, "strict");

  REQUIRE_FALSE(result.moment_ok);
  REQUIRE_FALSE(result.is_linear);
}

#else
int tile_compile_tests_linearity_stub() { return 0; }
#endif
