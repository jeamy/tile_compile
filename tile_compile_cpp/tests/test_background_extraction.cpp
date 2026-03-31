#if __has_include(<catch2/catch_test_macros.hpp>)
#include "tile_compile/core/types.hpp"
#include "tile_compile/image/background_extraction.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

namespace {

tile_compile::TileGrid make_single_tile_grid(int w, int h) {
  tile_compile::TileGrid grid{};
  grid.tile_size = w;
  grid.overlap_fraction = 0.0f;
  grid.rows = 1;
  grid.cols = 1;
  grid.tiles.push_back(tile_compile::Tile{0, 0, w, h, 0, 0});
  return grid;
}

tile_compile::TileMetrics make_structure_tile_metrics(float scale) {
  tile_compile::TileMetrics tm{};
  tm.fwhm = 2.5f;
  tm.roundness = 0.0f;
  tm.contrast = 0.0f;
  tm.sharpness = 0.0f;
  tm.background = 100.0f * scale;
  tm.noise = 1.8f * scale;
  tm.gradient_energy = 0.75f * scale * scale;
  tm.star_count = 0;
  tm.type = tile_compile::TileType::STRUCTURE;
  tm.quality_score = 0.0f;
  return tm;
}

tile_compile::image::BGEConfig make_bge_config(int w, int h) {
  tile_compile::image::BGEConfig cfg{};
  cfg.enabled = true;
  cfg.sample_quantile = 0.20f;
  cfg.structure_thresh_percentile = 0.95f;
  cfg.mask.star_dilate_px = 0;
  cfg.mask.sat_dilate_px = 0;
  cfg.common_mask_rows = h;
  cfg.common_mask_cols = w;
  cfg.common_valid_mask.assign(static_cast<size_t>(w * h), 1);
  return cfg;
}

tile_compile::Matrix2Df make_synthetic_tile(int w, int h, float scale) {
  tile_compile::Matrix2Df img(h, w);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float trend = 100.0f + 0.35f * static_cast<float>(x) +
                          0.20f * static_cast<float>(y);
      const float ripple =
          1.2f * std::sin(0.45f * static_cast<float>(x)) +
          0.8f * std::cos(0.35f * static_cast<float>(y));
      const float checker = ((x + 2 * y) % 5 == 0) ? 1.0f : -0.6f;
      img(y, x) = scale * (trend + ripple + checker);
    }
  }
  return img;
}

} // namespace

TEST_CASE("bge_tile_weight_is_stable_under_global_intensity_scaling") {
  constexpr int kWidth = 16;
  constexpr int kHeight = 16;
  constexpr float kScale = 37.0f;

  const auto grid = make_single_tile_grid(kWidth, kHeight);
  const auto cfg = make_bge_config(kWidth, kHeight);

  const auto base_img = make_synthetic_tile(kWidth, kHeight, 1.0f);
  const auto scaled_img = make_synthetic_tile(kWidth, kHeight, kScale);

  const std::vector<tile_compile::TileMetrics> base_metrics{
      make_structure_tile_metrics(1.0f)};
  const std::vector<tile_compile::TileMetrics> scaled_metrics{
      make_structure_tile_metrics(kScale)};

  const auto base_samples = tile_compile::image::extract_tile_background_samples(
      base_img, base_metrics, grid, cfg);
  const auto scaled_samples =
      tile_compile::image::extract_tile_background_samples(
          scaled_img, scaled_metrics, grid, cfg);

  REQUIRE(base_samples.size() == 1);
  REQUIRE(scaled_samples.size() == 1);
  REQUIRE(base_samples[0].valid);
  REQUIRE(scaled_samples[0].valid);
  REQUIRE(base_samples[0].weight > 0.0f);
  REQUIRE(scaled_samples[0].weight > 0.0f);

  REQUIRE(scaled_samples[0].bg_value ==
          Catch::Approx(base_samples[0].bg_value * kScale).epsilon(1e-4));
  REQUIRE(scaled_samples[0].weight ==
          Catch::Approx(base_samples[0].weight).epsilon(1e-3));
}
#else
int tile_compile_tests_background_extraction_stub() { return 0; }
#endif
