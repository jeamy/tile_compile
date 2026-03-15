#include "tile_compile/reconstruction/tile_normalization.hpp"

#include "tile_compile/core/utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tile_compile::reconstruction {

PositiveMedianEstimate positive_median(const Matrix2Df &src, bool abs_values,
                                       float center) {
  PositiveMedianEstimate out;
  if (src.size() <= 0) {
    return out;
  }
  out.total_count = static_cast<size_t>(src.size());
  std::vector<float> values;
  values.reserve(out.total_count);
  for (Eigen::Index i = 0; i < src.size(); ++i) {
    const float v = src.data()[i];
    if (!(std::isfinite(v) && v > 0.0f)) {
      continue;
    }
    values.push_back(abs_values ? std::fabs(v - center) : v);
  }
  out.sample_count = values.size();
  if (values.empty()) {
    return out;
  }
  const size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + static_cast<long>(mid),
                   values.end());
  out.value = values[mid];
  return out;
}

TileNormalizationStats estimate_tile_normalization_stats(const Matrix2Df &src) {
  TileNormalizationStats out;
  const auto bg = positive_median(src, false, 0.0f);
  const auto scale = positive_median(src, true, bg.value);
  out.background = bg.value;
  out.scale = scale.value;
  out.sample_count = bg.sample_count;
  out.total_count = bg.total_count;
  return out;
}

size_t minimum_tile_normalization_samples(
    size_t total_count, const TileNormalizationGuardConfig &config) {
  const size_t fraction_floor = static_cast<size_t>(
      std::ceil(static_cast<double>(total_count) *
                static_cast<double>(config.min_fraction)));
  return std::max(config.min_samples, fraction_floor);
}

TileNormalizationGuardSummary guard_tile_normalization_stats(
    std::vector<TileNormalizationStats> *stats,
    const std::vector<uint8_t> &valid_tiles,
    const TileNormalizationGuardConfig &config, float eps) {
  TileNormalizationGuardSummary summary;
  if (stats == nullptr || stats->empty()) {
    return summary;
  }

  std::vector<float> valid_backgrounds;
  std::vector<float> valid_scales;
  valid_backgrounds.reserve(stats->size());
  valid_scales.reserve(stats->size());
  for (size_t i = 0; i < stats->size(); ++i) {
    if (i >= valid_tiles.size() || valid_tiles[i] == 0u) {
      continue;
    }
    const auto &entry = (*stats)[i];
    const size_t min_required =
        minimum_tile_normalization_samples(entry.total_count, config);
    if (entry.sample_count < min_required || !std::isfinite(entry.background) ||
        !std::isfinite(entry.scale) || entry.scale <= eps) {
      continue;
    }
    valid_backgrounds.push_back(entry.background);
    valid_scales.push_back(entry.scale);
  }

  summary.global_background =
      valid_backgrounds.empty() ? 0.0f : core::median_of(valid_backgrounds);
  summary.global_scale =
      std::max(eps, valid_scales.empty() ? 1.0f : core::median_of(valid_scales));

  const float scale_floor =
      std::max(eps, summary.global_scale * config.scale_floor_factor);
  const float scale_ceil =
      std::max(scale_floor, summary.global_scale * config.scale_ceil_factor);

  for (size_t i = 0; i < stats->size(); ++i) {
    if (i >= valid_tiles.size() || valid_tiles[i] == 0u) {
      continue;
    }
    auto &entry = (*stats)[i];
    const size_t min_required =
        minimum_tile_normalization_samples(entry.total_count, config);

    if (entry.sample_count < min_required || !std::isfinite(entry.background)) {
      entry.background = summary.global_background;
      ++summary.used_global_background_count;
    }

    if (entry.sample_count < min_required || !std::isfinite(entry.scale) ||
        entry.scale <= eps) {
      entry.scale = summary.global_scale;
      ++summary.used_global_scale_count;
      continue;
    }

    if (entry.scale < scale_floor) {
      entry.scale = scale_floor;
      ++summary.clamped_low_scale_count;
    } else if (entry.scale > scale_ceil) {
      entry.scale = scale_ceil;
      ++summary.clamped_high_scale_count;
    }
  }

  return summary;
}

} // namespace tile_compile::reconstruction
