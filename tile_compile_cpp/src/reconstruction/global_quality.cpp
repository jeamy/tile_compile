#include "tile_compile/reconstruction/global_quality.hpp"

#include "tile_compile/metrics/metrics.hpp"
#include "tile_compile/reconstruction/source_quality_proxy.hpp"

namespace tile_compile::reconstruction {

VectorXf compute_global_quality_weights(size_t n, const SourceImageProvider &source_of,
                                        ColorMode color_mode, BayerPattern bayer_pattern,
                                        int cfa_origin_x, int cfa_origin_y,
                                        const GlobalQualityConfig &cfg) {
  VectorXf g_quality(static_cast<int>(n));
  if (n == 0) return g_quality;

  // Keep only scalar metrics across frames; release the analysis proxy
  // before requesting the next source image.
  std::vector<FrameMetrics> frame_metrics(n);
  std::vector<metrics::FrameStarMetrics> star_metrics(n);
  int ref_star_count = 0;
  for (size_t i = 0; i < n; ++i) {
    auto proxy = compute_source_quality_proxy_v1(source_of(i), color_mode, bayer_pattern,
                                                 cfa_origin_x, cfa_origin_y);
    frame_metrics[i] = metrics::calculate_frame_metrics(proxy.proxy_full);
    star_metrics[i] = metrics::measure_frame_stars(
        proxy.proxy_full, ref_star_count, cfg.star_max_corners, cfg.star_patch_radius);
    if (i == 0) ref_star_count = star_metrics[i].star_count;
  }

  const VectorXf raw = metrics::calculate_global_weights_with_stars(
      frame_metrics, star_metrics, cfg.w_bg, cfg.w_noise, cfg.w_grad, cfg.w_fwhm,
      cfg.w_roundness, cfg.w_star_count, cfg.clamp_lo, cfg.clamp_hi, cfg.adaptive_weights,
      cfg.weight_exponent_scale);

  // The exact, documented bounding transform: w/(1+w) = sigmoid(k*Q). See
  // the header for why this is necessary and why it is the minimal fix.
  for (size_t i = 0; i < n; ++i) {
    const float w = raw[static_cast<int>(i)];
    g_quality[static_cast<int>(i)] = w / (1.0f + w);
  }
  return g_quality;
}

VectorXf compute_global_quality_weights(const std::vector<Matrix2Df> &sources,
                                        ColorMode color_mode, BayerPattern bayer_pattern,
                                        int cfa_origin_x, int cfa_origin_y,
                                        const GlobalQualityConfig &cfg) {
  return compute_global_quality_weights(
      sources.size(), [&](size_t i) -> const Matrix2Df & { return sources.at(i); },
      color_mode, bayer_pattern, cfa_origin_x, cfa_origin_y, cfg);
}

}  // namespace tile_compile::reconstruction
