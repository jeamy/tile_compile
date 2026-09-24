#include "tile_compile/reconstruction/global_quality.hpp"

#include "tile_compile/metrics/metrics.hpp"
#include "tile_compile/reconstruction/source_quality_proxy.hpp"

#include <atomic>
#include <exception>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tile_compile::reconstruction {

namespace {

// The exact, documented bounding transform: w/(1+w) = sigmoid(k*Q). See
// global_quality.hpp for why this is necessary and why it is the minimal fix.
// Applied identically in both compute_global_quality_weights and
// compute_global_quality_weights_from_metrics.
VectorXf apply_sigmoid_bounding(const VectorXf& raw) {
  VectorXf out(raw.size());
  for (long i = 0; i < raw.size(); ++i) {
    const float w = raw[i];
    out[i] = w / (1.0f + w);
  }
  return out;
}

}  // namespace

VectorXf compute_global_quality_weights(
    size_t n, const SourceImageProvider &source_of, ColorMode color_mode,
    BayerPattern bayer_pattern, int cfa_origin_x, int cfa_origin_y,
    const GlobalQualityConfig &cfg, int workers,
    const std::function<SourceImageProvider()> &make_thread_provider) {
  VectorXf g_quality(static_cast<int>(n));
  if (n == 0) return g_quality;

  // Keep only scalar metrics across frames; release the analysis proxy
  // before requesting the next source image.
  std::vector<FrameMetrics> frame_metrics(n);
  std::vector<metrics::FrameStarMetrics> star_metrics(n);
  int ref_star_count = 0;

  auto one_frame = [&](size_t i, const Matrix2Df &src) {
    auto proxy = compute_source_quality_proxy_v1(src, color_mode, bayer_pattern,
                                                 cfa_origin_x, cfa_origin_y);
    frame_metrics[i] = metrics::calculate_frame_metrics(proxy.proxy_full);
    star_metrics[i] = metrics::measure_frame_stars(proxy.proxy_full,
                                                   ref_star_count,
                                                   cfg.star_max_corners,
                                                   cfg.star_patch_radius);
  };

  // Frame 0 first: it fixes ref_star_count for every other frame (legacy
  // convention). Its own measure_frame_stars is called with ref_star_count 0.
  one_frame(0, source_of(0));
  ref_star_count = star_metrics[0].star_count;

  const int nw =
      static_cast<int>(std::min<size_t>(std::max(1, workers), n));
#ifdef _OPENMP
  if (nw > 1 && make_thread_provider) {
    std::atomic<bool> failed{false};
    std::exception_ptr err;
#pragma omp parallel num_threads(nw)
    {
      SourceImageProvider tp;
      try {
        tp = make_thread_provider();
      } catch (...) {
#pragma omp critical(gq_err)
        if (!err) err = std::current_exception();
        failed.store(true, std::memory_order_relaxed);
      }
#pragma omp for schedule(dynamic, 1)
      for (long long i = 1; i < static_cast<long long>(n); ++i) {
        if (failed.load(std::memory_order_relaxed) || !tp)
          continue;
        try {
          one_frame(static_cast<size_t>(i), tp(static_cast<size_t>(i)));
        } catch (...) {
#pragma omp critical(gq_err)
          if (!err) err = std::current_exception();
          failed.store(true, std::memory_order_relaxed);
        }
      }
    }
    if (err) std::rethrow_exception(err);
  } else
#endif
  {
    (void)make_thread_provider;
    for (size_t i = 1; i < n; ++i)
      one_frame(i, source_of(i));
  }

  const VectorXf raw = metrics::calculate_global_weights_with_stars(
      frame_metrics, star_metrics, cfg.w_bg, cfg.w_noise, cfg.w_grad, cfg.w_fwhm,
      cfg.w_roundness, cfg.w_star_count, cfg.clamp_lo, cfg.clamp_hi, cfg.adaptive_weights,
      cfg.weight_exponent_scale);

  return apply_sigmoid_bounding(raw);
}

VectorXf compute_global_quality_weights(const std::vector<Matrix2Df> &sources,
                                        ColorMode color_mode, BayerPattern bayer_pattern,
                                        int cfa_origin_x, int cfa_origin_y,
                                        const GlobalQualityConfig &cfg) {
  return compute_global_quality_weights(
      sources.size(), [&](size_t i) -> const Matrix2Df & { return sources.at(i); },
      color_mode, bayer_pattern, cfa_origin_x, cfa_origin_y, cfg);
}

VectorXf compute_global_quality_weights_from_metrics(
    const std::vector<FrameMetrics> &frame_metrics,
    const std::vector<metrics::FrameStarMetrics> &star_metrics,
    const GlobalQualityConfig &cfg) {
  const size_t n = frame_metrics.size();
  VectorXf g_quality(static_cast<int>(n));
  if (n == 0) return g_quality;
  if (star_metrics.size() != n)
    throw std::invalid_argument("GLOBAL_QUALITY_METRICS_SIZE_MISMATCH");

  const VectorXf raw = metrics::calculate_global_weights_with_stars(
      frame_metrics, star_metrics, cfg.w_bg, cfg.w_noise, cfg.w_grad,
      cfg.w_fwhm, cfg.w_roundness, cfg.w_star_count, cfg.clamp_lo, cfg.clamp_hi,
      cfg.adaptive_weights, cfg.weight_exponent_scale);

  return apply_sigmoid_bounding(raw);
}

}  // namespace tile_compile::reconstruction
