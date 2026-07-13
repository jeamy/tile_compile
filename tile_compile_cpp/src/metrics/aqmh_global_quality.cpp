#include "tile_compile/metrics/aqmh_global_quality.hpp"

#include "tile_compile/metrics/aqmh_eps.hpp"

#include <cmath>
#include <stdexcept>

namespace tile_compile::metrics {

AqmhGlobalQualityResult compute_aqmh_global_quality(
    const std::vector<float> &sharpness_summaries,
    const std::vector<float> &snr_summaries,
    const std::vector<float> &background_penalty_summaries,
    const config::AqmhGlobalQualityConfig &cfg) {
  if (sharpness_summaries.size() != snr_summaries.size() ||
      sharpness_summaries.size() != background_penalty_summaries.size())
    throw std::invalid_argument("AQMH global-quality summary size mismatch");
  AqmhGlobalQualityResult result;
  result.weights.resize(sharpness_summaries.size(), 1.0f);
  result.input_invalid.resize(sharpness_summaries.size(), 0u);
  auto sharp_z = robust_zscore_eps_scale(sharpness_summaries);
  auto snr_z = robust_zscore_eps_scale(snr_summaries);
  auto background_z =
      robust_zscore_eps_scale(background_penalty_summaries);

  // Detect inputs with insufficient variance. When the MAD is tiny relative
  // to the median (CV < 1%), the z-score amplifies noise to extreme values
  // (e.g. z=18 for a 0.3% spread), which can invert the quality ranking.
  // For astrophotos, SNR is nearly constant across frames and positively
  // correlated with FWHM (blurry frames spread signal over more pixels),
  // so amplifying its tiny variance gives worst frames the highest weights.
  auto effective_weight = [](const std::vector<float> &values,
                             const std::vector<float> &z,
                             float configured_weight) -> float {
    if (configured_weight <= 0.0f)
      return 0.0f;
    std::vector<float> finite;
    finite.reserve(values.size());
    for (float v : values)
      if (std::isfinite(v) && v > 0.0f)
        finite.push_back(v);
    if (finite.size() < 3)
      return 0.0f;
    std::sort(finite.begin(), finite.end());
    const size_t mid = finite.size() / 2;
    const float med = finite.size() % 2
                          ? finite[mid]
                          : 0.5f * (finite[mid - 1] + finite[mid]);
    if (!(med > 0.0f))
      return 0.0f;
    std::vector<float> abs_dev;
    abs_dev.reserve(finite.size());
    for (float v : finite)
      abs_dev.push_back(std::fabs(v - med));
    std::sort(abs_dev.begin(), abs_dev.end());
    const float mad = abs_dev.size() % 2
                          ? abs_dev[abs_dev.size() / 2]
                          : 0.5f * (abs_dev[abs_dev.size() / 2 - 1] +
                                    abs_dev[abs_dev.size() / 2]);
    const float cv = mad / med;
    if (cv < 0.01f)
      return 0.0f;
    return configured_weight;
  };

  const float w_sharp_eff =
      effective_weight(sharpness_summaries, sharp_z, cfg.g_w_sharp);
  const float w_snr_eff =
      effective_weight(snr_summaries, snr_z, cfg.g_w_snr);
  const float w_background_eff = effective_weight(
      background_penalty_summaries, background_z,
      cfg.g_w_background_penalty);

  const float w_total = w_sharp_eff + w_snr_eff + w_background_eff;
  const float w_sharp_norm = (w_total > 0.0f) ? w_sharp_eff / w_total : 0.0f;
  const float w_snr_norm = (w_total > 0.0f) ? w_snr_eff / w_total : 0.0f;
  const float w_background_norm =
      (w_total > 0.0f) ? w_background_eff / w_total : 0.0f;

  for (size_t i = 0; i < result.weights.size(); ++i) {
    const bool invalid = !std::isfinite(sharpness_summaries[i]) ||
                         !std::isfinite(snr_summaries[i]) ||
                         !std::isfinite(background_penalty_summaries[i]);
    result.input_invalid[i] = invalid ? 1u : 0u;
    float zs = std::isfinite(sharp_z[i]) ? sharp_z[i] : 0.0f;
    float zn = std::isfinite(snr_z[i]) ? snr_z[i] : 0.0f;
    float zb = std::isfinite(background_z[i]) ? background_z[i] : 0.0f;
    zs = std::clamp(zs, -5.0f, 5.0f);
    zn = std::clamp(zn, -5.0f, 5.0f);
    zb = std::clamp(zb, -5.0f, 5.0f);
    const float score =
        w_sharp_norm * zs + w_snr_norm * zn - w_background_norm * zb;
    const float sigmoid = 1.0f / (1.0f + std::exp(-score));
    result.weights[i] = cfg.g_floor + (1.0f - cfg.g_floor) * sigmoid;
  }
  return result;
}

} // namespace tile_compile::metrics
