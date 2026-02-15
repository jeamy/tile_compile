#pragma once

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/image/normalization.hpp"

#include <filesystem>
#include <ostream>
#include <string>
#include <vector>

namespace tile_compile::runner {

struct PhaseMetricsContext {
  std::vector<image::NormalizationScales> norm_scales;
  std::vector<FrameMetrics> frame_metrics;
  VectorXf global_weights;
  float output_pedestal = 0.0f;
  float output_bg_mono = 1.0f;
  float output_bg_r = 1.0f;
  float output_bg_g = 1.0f;
  float output_bg_b = 1.0f;
};

bool run_phase_channel_split_normalization_global_metrics(
    const std::string &run_id, const config::Config &cfg,
    const std::vector<std::filesystem::path> &frames,
    const std::filesystem::path &run_dir, ColorMode detected_mode,
    const std::string &detected_bayer_str, core::EventEmitter &emitter,
    std::ostream &log_file, PhaseMetricsContext &out);

} // namespace tile_compile::runner
