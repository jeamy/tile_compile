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

/// Output bundle produced by the combined REGISTRATION and PREWARP phases.
///
/// The registration phase estimates one full-resolution affine warp per input
/// frame, rejects or models weak registrations, and then prewarps usable frames
/// onto a common canvas. Later phases consume the disk-backed frame store and
/// the validity masks here instead of re-reading or re-warping the original
/// FITS files.
struct PhaseRegistrationContext {
  /// Disk-backed prewarped frames on the common canvas.
  DiskCacheFrameStore prewarped_frames;
  /// Disk-backed prewarped RGB frames (Debayer-First-AQMH only).
  /// Populated when `aqmh.reconstruction.debayer_first` is true and the input
  /// mode is OSC. When active, downstream phases read from this store instead
  /// of `prewarped_frames`.
  DiskCacheFrameStoreRGB prewarped_frames_rgb;
  /// Per-input-frame flag indicating whether the prewarp store contains data.
  std::vector<uint8_t> frame_has_data;
  /// Per-canvas-pixel count of usable prewarped frames contributing data.
  std::vector<uint16_t> overlap_coverage_count;
  /// Binary common-overlap mask used to gate local metrics and reconstruction.
  std::vector<uint8_t> common_valid_mask;
  /// Per-input-frame flag: 1 if the warp was supplied by the field-rotation
  /// model (model_interpolated / model_blended / model_global_poly /
  /// model_local_poly / model_nearest_copy) rather than directly measured.
  /// Used by the pipeline to apply a configurable weight penalty so that
  /// model-predicted frames contribute less to the stack than directly
  /// registered frames even when their image quality metrics are good.
  std::vector<uint8_t> model_predicted_mask;
  /// Number of frames retained after registration, modeling, and prewarp.
  int n_usable_frames = 0;
  /// Minimum frame support required by downstream common-overlap decisions.
  int min_valid_frames = 1;
  /// Width of the prewarped common canvas in full-resolution pixels.
  int canvas_width = 0;
  /// Height of the prewarped common canvas in full-resolution pixels.
  int canvas_height = 0;
  /// X offset from original frame coordinates into canvas coordinates.
  int tile_offset_x = 0;
  /// Y offset from original frame coordinates into canvas coordinates.
  int tile_offset_y = 0;
  /// Disk-backed prewarped per-frame background model grids on the canvas grid.
  /// Populated in Stufe B and consumed by the reconstruction phase.
  std::shared_ptr<BackgroundModelGridStore> prewarped_background_grid_store;
  /// Number of rows/cols in the prewarped background grid canvas domain.
  int background_grid_rows = 0;
  int background_grid_cols = 0;
};

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
    PhaseRegistrationContext &out,
    const std::shared_ptr<DiskCacheFrameStoreRGB> &rgb_frame_cache = nullptr,
    const std::shared_ptr<BackgroundModelGridStore> &background_grid_store = nullptr);

} // namespace tile_compile::runner
