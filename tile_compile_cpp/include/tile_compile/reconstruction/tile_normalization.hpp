#pragma once

#include "tile_compile/core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tile_compile::reconstruction {

struct PositiveMedianEstimate {
  float value = 0.0f;
  size_t sample_count = 0;
  size_t total_count = 0;
};

struct TileNormalizationStats {
  float background = 0.0f;
  float scale = 1.0f;
  size_t sample_count = 0;
  size_t total_count = 0;
};

struct TileNormalizationGuardConfig {
  size_t min_samples = 64;
  float min_fraction = 0.05f;
  float scale_floor_factor = 0.5f;
  float scale_ceil_factor = 2.0f;
};

struct TileNormalizationGuardSummary {
  float global_background = 0.0f;
  float global_scale = 1.0f;
  size_t used_global_background_count = 0;
  size_t used_global_scale_count = 0;
  size_t clamped_low_scale_count = 0;
  size_t clamped_high_scale_count = 0;
};

PositiveMedianEstimate positive_median(const Matrix2Df &src,
                                       bool abs_values = false,
                                       float center = 0.0f);

TileNormalizationStats estimate_tile_normalization_stats(const Matrix2Df &src);

size_t minimum_tile_normalization_samples(
    size_t total_count, const TileNormalizationGuardConfig &config);

TileNormalizationGuardSummary guard_tile_normalization_stats(
    std::vector<TileNormalizationStats> *stats,
    const std::vector<uint8_t> &valid_tiles,
    const TileNormalizationGuardConfig &config, float eps);

} // namespace tile_compile::reconstruction
