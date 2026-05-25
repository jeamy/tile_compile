#pragma once

#include "runner_phase_preprocess_pipeline.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/metrics/metrics.hpp"
#include "tile_compile/preprocessing/contract.hpp"

#include <filesystem>
#include <ostream>
#include <string>
#include <vector>

namespace tile_compile::runner {

/// Per-frame exclusion reason codes.
enum class FrameExclusionReason {
    not_excluded,
    fwhm_too_high,
    eccentricity_too_high,
    star_count_too_low,
    registration_cc_too_low,
    manual_exclude,
};

std::string frame_exclusion_reason_to_string(FrameExclusionReason r);

/// Full quality record for one frame, written into frame_quality.csv and the
/// quality analysis JSON artifact. All per-frame diagnostic values from
/// GLOBAL_METRICS, REGISTRATION, and additional passes are aggregated here.
struct FrameQualityRecord {
    int   index             = 0;
    std::string filename;

    // --- Photometric / star metrics (from GLOBAL_METRICS) ---
    int   star_count        = 0;      ///< detected stars with valid FWHM
    float fwhm              = 0.0f;   ///< median FWHM in pixels
    float fwhm_x            = 0.0f;   ///< median FWHM in X
    float fwhm_y            = 0.0f;   ///< median FWHM in Y
    float eccentricity      = 0.0f;   ///< 1 - (min_axis / max_axis); 0 = round
    float background_median = 0.0f;   ///< median background level
    float background_rms    = 0.0f;   ///< background noise RMS
    float snr_estimate      = 0.0f;   ///< S/N = (signal_peak - bg) / noise
    float clip_fraction     = 0.0f;   ///< fraction of pixels at or above saturation

    // --- Registration metrics ---
    float registration_cc   = 0.0f;   ///< NCC correlation coefficient vs reference

    // --- Derived quality score ---
    float quality_score     = 0.0f;   ///< combined quality score [0,1]

    // --- Frame filtering decision ---
    bool  included          = true;
    FrameExclusionReason exclusion_reason = FrameExclusionReason::not_excluded;
    std::string exclusion_detail;     ///< human-readable detail

    // --- Manual override ---
    bool  manual_override   = false;
    bool  manual_included   = true;   ///< effective value when manual_override is true
};

/// Output of the QUALITY_ANALYSIS and FRAME_FILTERING phases.
struct QualityAnalysisContext {
    std::vector<FrameQualityRecord> records;
    /// Indices into PreprocessPipelineContext::effective_frames that pass filtering.
    std::vector<int> accepted_indices;
    /// Indices of rejected frames with their reasons.
    std::vector<int> rejected_indices;
    /// Path to the written frame_quality.csv.
    std::filesystem::path csv_path;
};

/// Execute QUALITY_ANALYSIS and FRAME_FILTERING phases (Plan §7).
///
/// Uses per-frame metrics from `pipeline_ctx` and registration correlations,
/// runs auto-rejection according to `cfg.quality_filter`, and writes:
///   - `artifacts/preprocess/frame_quality.csv`
///   - `artifacts/preprocess/quality_analysis.json`
///
/// @param run_id        Unique run identifier.
/// @param cfg           Preprocessing configuration.
/// @param pipeline_ctx  Result from run_preprocess_pipeline().
/// @param star_metrics  Per-frame star metrics (fwhm, roundness, star_count).
/// @param run_dir       Root directory for this run.
/// @param emitter       Event emitter.
/// @param log_file      Log output stream.
/// @param out           Populated on success.
/// @return true on success (≥1 frame accepted), false on error.
bool run_quality_analysis(
    const std::string& run_id,
    const preprocessing::Config& cfg,
    const PreprocessPipelineContext& pipeline_ctx,
    const std::vector<metrics::FrameStarMetrics>& star_metrics,
    const std::filesystem::path& run_dir,
    core::EventEmitter& emitter,
    std::ostream& log_file,
    QualityAnalysisContext& out);

} // namespace tile_compile::runner
