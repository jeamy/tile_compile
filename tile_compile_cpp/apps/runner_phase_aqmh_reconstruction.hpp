#pragma once

#include "runner_shared.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/acceleration.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/reconstruction/aqmh_reconstruction.hpp"
#include "tile_compile/reconstruction/aqmh_validation.hpp"

#include <chrono>
#include <filesystem>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace tile_compile::metrics { class QualityMapCache; }

namespace tile_compile::reconstruction {
class AqmhPrefetchCoordinator;
}

namespace tile_compile::runner {

struct AqmhReconstructionPhaseResult {
  reconstruction::AqmhReconstructionResult recon;
  reconstruction::AqmhValidationComparison control_validation;
  Matrix2Df raw_output;
  Matrix2Df output;
  Matrix2Df weight_sum;
  bool osc_rgb_cleared = false;
  // Debayer-First-AQMH: per-channel reconstructed outputs.
  Matrix2Df df_output_R;
  Matrix2Df df_output_G;
  Matrix2Df df_output_B;
  Matrix2Df df_weight_sum_R;
  Matrix2Df df_weight_sum_G;
  Matrix2Df df_weight_sum_B;
  std::vector<uint8_t> df_valid_mask_R;
  std::vector<uint8_t> df_valid_mask_G;
  std::vector<uint8_t> df_valid_mask_B;
  bool debayer_first_used = false;
  // Stufe B: accumulated per-frame background model grids on the canvas grid
  // (reference domain). The output stage upsamples and adds these before
  // applying output-scale restoration.
  BackgroundModelGrid background_map_canvas_grid;
  // Stufe C: per-channel uniform control maps for Debayer-First-AQMH.
  Matrix2Df df_control_R;
  Matrix2Df df_control_G;
  Matrix2Df df_control_B;
};

bool run_phase_aqmh_reconstruction(
    const std::string &run_id, const config::Config &cfg,
    const std::filesystem::path &run_dir,
    const std::vector<std::filesystem::path> &frames,
    const std::vector<uint8_t> &frame_has_data,
    const std::vector<uint8_t> &reconstruction_valid_mask,
    const std::vector<uint8_t> &validation_common_mask,
    int canvas_width, int canvas_height,
    bool osc_mode,
    const DiskCacheFrameStore &prewarped_frames,
    std::unique_ptr<metrics::QualityMapCache> &aqmh_cache,
    const VectorXf &aqmh_global_weights,
    core::AccelerationContext &acceleration,
    core::EventEmitter &emitter, std::ostream &log_file,
    const std::chrono::steady_clock::time_point &phase_started_at,
    int prev_cv_threads,
    AqmhReconstructionPhaseResult &out,
    reconstruction::AqmhPrefetchCoordinator* prefetch_coordinator = nullptr,
    const DiskCacheFrameStoreRGB *prewarped_frames_rgb = nullptr,
    const BackgroundModelGridStore *prewarped_background_grid_store = nullptr
);

} // namespace tile_compile::runner
