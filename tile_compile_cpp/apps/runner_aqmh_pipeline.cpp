#include "runner_aqmh_pipeline.hpp"

#include "runner_phase_aqmh_diagnostics.hpp"
#include "runner_phase_aqmh_maps.hpp"
#include "runner_phase_aqmh_reconstruction.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/reconstruction/aqmh_pipeline_overlap.hpp"

#include <iostream>
#include <memory>

namespace tile_compile::runner {

bool run_aqmh_phases(
    const std::string &run_id, const config::Config &cfg,
    const std::filesystem::path &run_dir,
    const std::vector<std::filesystem::path> &frames,
    const std::vector<uint8_t> &frame_has_data,
    const std::vector<uint8_t> &common_valid_mask,
    int canvas_width, int canvas_height, bool osc_mode,
    const DiskCacheFrameStore &prewarped_frames,
    const std::vector<image::NormalizationScales> &norm_scales,
    ColorMode detected_mode, const std::string &detected_bayer_str,
    core::AccelerationContext &acceleration,
    core::EventEmitter &emitter, std::ostream &log_file,
    AqmhPipelineContext &out) {

  // Stage 1: AQMH_MAPS → AQMH_GLOBAL_QUALITY (global quality called internally
  // at end of maps phase, before returning, so both complete before Stage 2).
  if (!run_phase_aqmh_maps(
          run_id, cfg, frames, run_dir, frame_has_data, common_valid_mask,
          common_valid_mask, canvas_width, canvas_height, prewarped_frames,
          norm_scales,
          detected_mode, detected_bayer_str, false, acceleration, emitter,
          log_file, out.aqmh_cache, out.aqmh_global_weights,
          out.prefetch_coordinator)) {
    return false;
  }

  // Stage 2: AQMH_RECONSTRUCTION (without inline diagnostics — Stage 3 below).
  const auto recon_started_at = std::chrono::steady_clock::now();
  const int prev_cv_threads = cv::getNumThreads();
  cv::setNumThreads(1);

  AqmhReconstructionPhaseResult recon_phase_result;
  if (!run_phase_aqmh_reconstruction(
          run_id, cfg, run_dir, frames, frame_has_data,
          common_valid_mask, common_valid_mask, canvas_width, canvas_height,
          osc_mode,
          prewarped_frames, out.aqmh_cache, out.aqmh_global_weights,
          acceleration, emitter, log_file,
          recon_started_at, prev_cv_threads, recon_phase_result,
          out.prefetch_coordinator.get(), nullptr, nullptr)) {
    return false;
  }

  out.recon_result     = recon_phase_result.recon;
  out.recon_output     = recon_phase_result.output;
  out.weight_sum       = recon_phase_result.weight_sum;
  out.control_validation = recon_phase_result.control_validation;

  // Stage 3: AQMH_DIAGNOSTICS — independent stage, runs after reconstruction.
  // Computes block-level Q-map statistics and heatmaps (§6.2, §6.3), and
  // writes run-level cherry-pick fields to aqmh_metrics.json.
  if (!run_phase_aqmh_diagnostics(
          run_id, cfg, run_dir, out.recon_result,
          out.aqmh_cache.get(), common_valid_mask, frame_has_data,
          canvas_width, canvas_height,
          emitter, log_file)) {
    return false;
  }

  return true;
}

} // namespace tile_compile::runner
