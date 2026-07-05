#include "runner_phase_aqmh_maps.hpp"

#include "runner_phase_local_metrics.hpp"
#include "tile_compile/metrics/metrics.hpp"
#include "tile_compile/reconstruction/aqmh_pipeline_overlap.hpp"

namespace tile_compile::runner {

bool run_phase_aqmh_maps(
    const std::string &run_id, const config::Config &cfg,
    const std::vector<std::filesystem::path> &frames,
    const std::filesystem::path &run_dir,
    const std::vector<uint8_t> &frame_has_data,
    const std::vector<uint8_t> &canvas_mask, int width, int height,
    const DiskCacheFrameStore &prewarped_frames,
    const std::vector<image::NormalizationScales> &norm_scales,
    ColorMode detected_mode, const std::string &detected_bayer_str,
    bool apply_normalization, core::AccelerationContext &acceleration,
    core::EventEmitter &emitter, std::ostream &log_file,
    std::unique_ptr<metrics::QualityMapCache> &out_cache,
    VectorXf &out_global_weights,
    std::unique_ptr<reconstruction::AqmhPrefetchCoordinator> &out_prefetch_coordinator,
    const std::vector<metrics::FrameStarMetrics> &frame_star_metrics) {
  std::vector<std::vector<TileMetrics>> unused_metrics;
  std::vector<std::vector<float>> unused_weights;
  std::vector<float> unused_quality, unused_fwhm;
  std::vector<uint8_t> unused_star;
  const std::vector<Tile> no_tiles;
  const std::vector<uint8_t> no_tile_mask;
  return run_phase_local_metrics(
      run_id, cfg, frames, run_dir, frame_has_data, no_tiles, canvas_mask,
      width, height, no_tile_mask, prewarped_frames, norm_scales,
      detected_mode, detected_bayer_str, apply_normalization, acceleration,
      emitter, log_file, unused_metrics, unused_weights, unused_quality,
      unused_star, unused_fwhm, out_cache, out_global_weights, 0, 0,
      frame_star_metrics, &out_prefetch_coordinator);
}

} // namespace tile_compile::runner
