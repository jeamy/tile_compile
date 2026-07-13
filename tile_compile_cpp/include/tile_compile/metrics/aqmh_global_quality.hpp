#pragma once

#include "tile_compile/config/configuration.hpp"

#include <cstdint>
#include <vector>

namespace tile_compile::metrics {

struct AqmhGlobalQualityResult {
  std::vector<float> weights;
  std::vector<uint8_t> input_invalid;
};

AqmhGlobalQualityResult compute_aqmh_global_quality(
    const std::vector<float> &sharpness_summaries,
    const std::vector<float> &snr_summaries,
    const std::vector<float> &background_penalty_summaries,
    const config::AqmhGlobalQualityConfig &cfg);

} // namespace tile_compile::metrics
