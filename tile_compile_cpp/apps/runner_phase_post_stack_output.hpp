#pragma once

#include "runner_shared.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/io/fits_io.hpp"

#include <filesystem>
#include <ostream>
#include <string>
#include <vector>

namespace tile_compile::runner {

/// Parameters for writing post-stack output files (stacked.fits, stacked_rgb,
/// reconstructed_L/R/G/B, stacked_rgb_solve). Used by both the main pipeline
/// and the resume path to ensure identical behavior.
struct PostStackOutputConfig {
  bool output_stretch = false;
  bool crop_to_nonzero_bbox = true;
  bool aqmh_enabled = false;
  bool cosmetic_correction = false;
  float cosmetic_correction_sigma = 4.0f;
};

/// Output scaling factors restored from the normalization artifact.
/// Used to re-add the photometric background to the linear reconstruction.
struct OutputScaling {
  float scale_mono = 1.0f;
  float scale_r = 1.0f;
  float scale_g = 1.0f;
  float scale_b = 1.0f;
  float bg_mono = 0.0f;
  float bg_r = 0.0f;
  float bg_g = 0.0f;
  float bg_b = 0.0f;
  float pedestal = 0.0f;
};

/// Result struct returned by the post-stack output writer.
struct PostStackOutputResult {
  bool success = false;
  CropBox crop_box{0, 0, 0, 0};
  bool crop_applied = false;
  std::string error;
};

/// Write all post-stack output FITS files with consistent stretch, crop,
/// scaling, and format logic.
///
/// This function handles:
///   - Output scaling restoration (re-add background + photometric scale)
///   - Canvas mask enforcement
///   - Crop to non-zero bounding box
///   - Luma stretch (u16 robust p99.9)
///   - RGB stretch (u32 robust p99.9)
///   - Writing stacked.fits, reconstructed_L.fit, reconstructed_R/G/B.fit,
///     stacked_rgb.fits, stacked_rgb_solve.fits
///   - Canvas mask FITS update after crop
///
/// Both runner_pipeline.cpp and runner_resume.cpp must call this function
/// instead of implementing their own output logic.
///
/// @param recon         Mono/luma reconstruction (modified in place for crop).
/// @param recon_R       Red channel (empty for MONO).
/// @param recon_G       Green channel (empty for MONO).
/// @param recon_B       Blue channel (empty for MONO).
/// @param common_valid_mask  Canvas validity mask.
/// @param analysis_valid_mask  Common-overlap analysis mask (may differ from canvas).
/// @param scaling       Output scaling factors from normalization.
/// @param detected_mode OSC or MONO.
/// @param detected_bayer_str  Bayer pattern string.
/// @param debayer_tile_offset_x  CFA alignment offset X.
/// @param debayer_tile_offset_y  CFA alignment offset Y.
/// @param first_hdr     FITS header for output files.
/// @param cfg           Post-stack output configuration.
/// @param run_dir       Run directory.
/// @param run_id        Run identifier.
/// @param emitter       Event emitter.
/// @param log_file      Log stream.
/// @param out           Result (crop box, success).
/// @return true on success, false on error.
bool write_post_stack_outputs(
    Matrix2Df &recon,
    Matrix2Df &recon_R,
    Matrix2Df &recon_G,
    Matrix2Df &recon_B,
    std::vector<uint8_t> &common_valid_mask,
    std::vector<uint8_t> &analysis_valid_mask,
    const OutputScaling &scaling,
    ColorMode detected_mode,
    const std::string &detected_bayer_str,
    int debayer_tile_offset_x,
    int debayer_tile_offset_y,
    const io::FitsHeader &first_hdr,
    const PostStackOutputConfig &cfg,
    const std::filesystem::path &run_dir,
    const std::string &run_id,
    core::EventEmitter &emitter,
    std::ostream &log_file,
    PostStackOutputResult &out);

/// Write a stretched RGB snapshot (used for BGE/PCC intermediate outputs).
/// Applies canvas mask, then robust p99.9 stretch if requested, writes as
/// uint32 or float32 depending on stretch flag.
void write_stretched_rgb_snapshot(
    const std::filesystem::path &path,
    const Matrix2Df &R_src,
    const Matrix2Df &G_src,
    const Matrix2Df &B_src,
    const std::vector<uint8_t> &canvas_mask,
    int canvas_rows,
    int canvas_cols,
    const io::FitsHeader &hdr,
    bool apply_stretch,
    const char *stage_tag);

} // namespace tile_compile::runner
