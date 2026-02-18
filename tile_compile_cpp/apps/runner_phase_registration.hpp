#pragma once

#include "runner_shared.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/image/normalization.hpp"
#include "tile_compile/io/fits_io.hpp"

#include <filesystem>
#include <ostream>
#include <string>
#include <vector>

namespace tile_compile::runner {

struct PhaseRegistrationContext {
  DiskCacheFrameStore prewarped_frames;
  std::vector<uint8_t> frame_has_data;
  int n_usable_frames = 0;
  int min_valid_frames = 1;
  int canvas_width = 0;   // Expanded canvas width for field rotation
  int canvas_height = 0;  // Expanded canvas height for field rotation
  int tile_offset_x = 0;  // Tile coordinate offset for field rotation
  int tile_offset_y = 0;  // Tile coordinate offset for field rotation
};

bool run_phase_registration_prewarp(
    const std::string &run_id, const config::Config &cfg,
    const std::vector<std::filesystem::path> &frames,
    const std::filesystem::path &run_dir, int height, int width,
    ColorMode detected_mode, const std::string &detected_bayer_str,
    const std::vector<image::NormalizationScales> &norm_scales,
    const std::vector<FrameMetrics> &frame_metrics,
    const VectorXf &global_weights, const io::FitsHeader &first_hdr,
    core::EventEmitter &emitter, std::ostream &log_file,
    PhaseRegistrationContext &out);

} // namespace tile_compile::runner
