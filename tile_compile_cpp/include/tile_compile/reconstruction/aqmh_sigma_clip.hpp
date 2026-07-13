#pragma once

#include <cstddef>
#include <vector>

namespace tile_compile::reconstruction {

struct AqmhWeightedSample {
  float value = 0.0f;
  float weight = 0.0f;
  float score = 0.0f;
  size_t frame_index = 0;
};

struct AqmhSigmaClipResult {
  std::vector<AqmhWeightedSample> retained;
  float weight_sum = 0.0f;
  float effective_n = 0.0f;
  bool denominator_ok = false;
};

AqmhSigmaClipResult aqmh_sigma_clip(
    std::vector<AqmhWeightedSample> samples, float clip_sigma,
    int iterations, float min_fraction, float min_effective_n);

AqmhSigmaClipResult aqmh_sigma_clip(
    std::vector<AqmhWeightedSample> samples, float clip_sigma_low,
    float clip_sigma_high, int iterations, float min_fraction,
    float min_effective_n);

} // namespace tile_compile::reconstruction
