#pragma once

#include "runner_shared.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/image/normalization.hpp"
#include "tile_compile/preprocessing/contract.hpp"

#include <filesystem>
#include <ostream>
#include <string>
#include <vector>

namespace tile_compile::runner {

/// Output bundle from the preprocessing pipeline phases (6):
/// INPUT_FRAMES, CALIBRATION, CFA_CHANNEL_PREP, REFERENCE_SELECTION,
/// REGISTRATION, NORMALIZATION.
///
/// These outputs are consumed by QUALITY_ANALYSIS (phase 7) and STACKING.
struct PreprocessPipelineContext {
    /// Calibrated (or uncalibrated) light frames ready for registration.
    std::vector<std::filesystem::path> effective_frames;
    /// Whether calibration was requested and applied.
    bool calibration_applied = false;
    /// Detected color mode (MONO or OSC).
    ColorMode color_mode = ColorMode::MONO;
    /// Detected Bayer pattern string ("RGGB", "BGGR", …, or "" for mono).
    std::string bayer_pattern;
    /// Index of the selected reference frame within `effective_frames`.
    int reference_frame_index = 0;
    /// Metric used for reference selection ("best_quality", "temporal_center").
    std::string reference_selection_strategy;
    /// Per-frame normalization scales computed during CFA_CHANNEL_PREP.
    std::vector<image::NormalizationScales> norm_scales;
    /// Per-frame global quality metrics (background, noise, FWHM, stars).
    std::vector<FrameMetrics> frame_metrics;
    /// Global stacking weights derived from frame_metrics.
    VectorXf global_weights;
    /// Per-frame registration warps (identity for reference frame).
    std::vector<WarpMatrix> frame_warps;
    /// Per-frame registration correlation coefficients.
    std::vector<float> frame_cc;
    /// Disk-backed normalized + prewarped frame store for STACKING.
    DiskCacheFrameStore prewarped_frames;
    /// Per-frame flag: true if this frame was successfully prewarped.
    std::vector<uint8_t> frame_has_data;
    /// Canvas dimensions after registration.
    int canvas_width = 0;
    int canvas_height = 0;
};

/// Execute the full preprocessing pipeline cut (Plan §6) for LINEAR_PRESTACK.
///
/// Phases driven:
///   - INPUT_FRAMES   : scans lights_dir, validates dimensions, reads headers
///   - CALIBRATION    : applies bias/dark/flat if configured; otherwise skipped
///   - CFA_CHANNEL_PREP: determines color mode, bayer pattern, normalization
///   - REFERENCE_SELECTION: selects reference frame via `best_quality` or config
///   - REGISTRATION   : reuses existing global registration; writes metrics
///   - (Tile phases TILE_GRID, TILE_RECONSTRUCTION, SYNTHETIC_FRAMES,
///      STATE_CLUSTERING are explicitly NOT started)
///
/// @param run_id     Unique run identifier used in event emission.
/// @param cfg        Preprocessing configuration (preprocessing::Config).
/// @param run_dir    Root directory for this run's artifacts.
/// @param proj_root  Project root for resolving relative paths.
/// @param emitter    Event emitter writing to events.jsonl.
/// @param log_file   Log output stream.
/// @param out        Populated on success.
/// @return true on success, false after emitting an error event.
bool run_preprocess_pipeline(
    const std::string& run_id,
    const preprocessing::Config& cfg,
    const std::filesystem::path& run_dir,
    const std::filesystem::path& proj_root,
    core::EventEmitter& emitter,
    std::ostream& log_file,
    PreprocessPipelineContext& out);

} // namespace tile_compile::runner
