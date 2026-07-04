#include "tile_compile/metrics/aqmh_global_quality.hpp"

#include "tile_compile/metrics/aqmh_eps.hpp"

#include <cmath>
#include <stdexcept>

namespace tile_compile::metrics {

AqmhGlobalQualityResult compute_aqmh_global_quality(
    const std::vector<float> &sharpness_summaries,
    const std::vector<float> &snr_summaries,
    const config::AqmhGlobalQualityConfig &cfg) {
  if (sharpness_summaries.size() != snr_summaries.size())
    throw std::invalid_argument("AQMH global-quality summary size mismatch");
  AqmhGlobalQualityResult result;
  result.weights.resize(sharpness_summaries.size(), 1.0f);
  result.input_invalid.resize(sharpness_summaries.size(), 0u);
  auto sharp_z = robust_zscore_eps_scale(sharpness_summaries);
  auto snr_z = robust_zscore_eps_scale(snr_summaries);
  for (size_t i = 0; i < result.weights.size(); ++i) {
    const bool invalid = !std::isfinite(sharpness_summaries[i]) ||
                         !std::isfinite(snr_summaries[i]);
    result.input_invalid[i] = invalid ? 1u : 0u;
    const float zs = std::isfinite(sharp_z[i]) ? sharp_z[i] : 0.0f;
    const float zn = std::isfinite(snr_z[i]) ? snr_z[i] : 0.0f;
    const float score = cfg.g_w_sharp * zs + cfg.g_w_snr * zn;
    const float sigmoid = 1.0f / (1.0f + std::exp(-score));
    result.weights[i] = cfg.g_floor + (1.0f - cfg.g_floor) * sigmoid;
  }
  return result;
}

} // namespace tile_compile::metrics
