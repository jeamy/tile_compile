#pragma once

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/image/normalization.hpp"
#include "tile_compile/metrics/metrics.hpp"

#include <filesystem>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace tile_compile::runner {

class RunnerFrameCache;
class DiskCacheFrameStoreRGB;

/// Shared outputs from CHANNEL_SPLIT, NORMALIZATION, and GLOBAL_METRICS.
///
/// The runner keeps these three phases together because normalized frames are
/// immediately reused for global metric extraction and later registration. The
/// context therefore carries both the per-frame normalization scales and the
/// disk-backed cache of normalized frame data.
struct PhaseMetricsContext {
  /// Per-frame scalar/background normalization factors, including OSC channels.
  std::vector<image::NormalizationScales> norm_scales;
  /// Per-frame global quality metrics such as background, noise, FWHM, stars.
  std::vector<FrameMetrics> frame_metrics;
  /// Per-frame star metrics (FWHM, roundness, star_count) from GLOBAL_METRICS.
  std::vector<metrics::FrameStarMetrics> frame_star_metrics;
  /// Final global frame weights derived from `frame_metrics`.
  VectorXf global_weights;
  /// Disk cache containing normalized full-frame images for later phases.
  std::shared_ptr<RunnerFrameCache> frame_cache;
  /// Disk cache containing debayered RGB frames (Debayer-First-AQMH only).
  /// Populated when `aqmh.reconstruction.debayer_first` is true and the input
  /// mode is OSC. Later phases (prewarp, Q-maps, reconstruction) read from this
  /// cache instead of the CFA `frame_cache`.
  std::shared_ptr<DiskCacheFrameStoreRGB> rgb_frame_cache;
  /// Pedestal applied when restoring output scaling after reconstruction.
  float output_pedestal = 0.0f;
  /// Mono/luma output scale used by final output restoration.
  float output_scale_mono = 1.0f;
  /// Red-channel output scale for OSC/RGB restoration.
  float output_scale_r = 1.0f;
  /// Green-channel output scale for OSC/RGB restoration.
  float output_scale_g = 1.0f;
  /// Blue-channel output scale for OSC/RGB restoration.
  float output_scale_b = 1.0f;
  /// Mono/luma background reference measured during normalization.
  float output_bg_mono = 0.0f;
  /// Red-channel background reference measured during normalization.
  float output_bg_r = 0.0f;
  /// Green-channel background reference measured during normalization.
  float output_bg_g = 0.0f;
  /// Blue-channel background reference measured during normalization.
  float output_bg_b = 0.0f;
};

/// Execute metadata channel handling, per-frame normalization, and global metrics.
///
/// For OSC input this phase records channel metadata but defers actual channel
/// splitting to tile processing. It normalizes each input frame, stores the
/// normalized result in `PhaseMetricsContext::frame_cache`, computes global
/// frame metrics, derives global weights, and writes the corresponding JSON
/// artifacts under `run_dir/artifacts`.
///
/// Returns `true` when all required artifacts and cached normalized frames were
/// produced; returns `false` after emitting a phase error.
bool run_phase_channel_split_normalization_global_metrics(
    const std::string &run_id, const config::Config &cfg,
    const std::vector<std::filesystem::path> &frames,
    const std::filesystem::path &run_dir, ColorMode detected_mode,
    const std::string &detected_bayer_str, core::EventEmitter &emitter,
    std::ostream &log_file, PhaseMetricsContext &out);

} // namespace tile_compile::runner
