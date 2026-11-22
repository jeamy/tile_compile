#pragma once

#include "runner_shared.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/acceleration.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/image/normalization.hpp"
#include "tile_compile/io/fits_io.hpp"

#include <filesystem>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace tile_compile::runner {

class RunnerFrameCache;

/// Output of the REGISTRATION phase.
///
/// The registration phase estimates one full-resolution affine warp per input
/// frame and rejects or models weak registrations. Forward drizzle consumes the
/// resulting sampling plan directly; no prewarped canvas is materialized.
struct PhaseRegistrationContext {
  /// Also carries the per-frame weight factors (model_prediction_factor,
  /// registration_residual_factor) that forward drizzle folds into g_eff.
  registration::RegistrationSamplingPlan sampling_plan;
};

/// True when a model-predicted frame is left out of the stack: its median
/// matched-star residual (registration proxy px) exceeds `max_residual_px`, or
/// it could not be measured. `max_residual_px <= 0` keeps every frame
/// (registration.model_prediction_max_residual_px).
inline bool drop_model_predicted_frame(bool residual_applicable, float median_px,
                                       float max_residual_px) {
  if (max_residual_px <= 0.0f) return false;
  return !residual_applicable || !(median_px <= max_residual_px);
}

/// Run the REGISTRATION and PREWARP phases and fill downstream phase context.
///
/// This phase builds registration proxies from normalized frames, selects one
/// or more temporal/quality anchors, registers frames through the configured
/// cascade, applies rescue/modeling for weak frames, writes
/// `global_registration.json`, and finally prewarps accepted frames to a common
/// canvas. The function owns phase event emission and writes all registration
/// diagnostics into `run_dir/artifacts`.
///
/// Returns `true` when enough frames were registered/prewarped for the pipeline
/// to continue; returns `false` after emitting an error event when the phase
/// cannot produce a valid common canvas.
bool run_phase_registration_prewarp(
    const std::string &run_id, const config::Config &cfg,
    const std::vector<std::filesystem::path> &frames,
    const std::filesystem::path &run_dir, int height, int width,
    ColorMode detected_mode, const std::string &detected_bayer_str,
    const std::shared_ptr<RunnerFrameCache> &frame_cache,
    const std::vector<image::NormalizationScales> &norm_scales,
    const std::vector<FrameMetrics> &frame_metrics,
    const VectorXf &global_weights, const io::FitsHeader &first_hdr,
    core::AccelerationContext &acceleration, core::EventEmitter &emitter,
    std::ostream &log_file,
    PhaseRegistrationContext &out);

} // namespace tile_compile::runner
