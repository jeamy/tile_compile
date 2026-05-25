#pragma once

#include "tile_compile/config/configuration.hpp"

#include <string>
#include <vector>

namespace tile_compile::preprocessing {

enum class Phase {
  INPUT_SCAN,
  CALIBRATION,
  CFA_CHANNEL_PREP,
  REFERENCE_SELECTION,
  REGISTRATION,
  QUALITY_ANALYSIS,
  FRAME_FILTERING,
  STACKING,
  ASTROMETRY,
  BGE,
  PCC,
  HYPERMETRIC_STRETCH,
  REPORT
};

struct RejectionConfig {
  std::string method = "sigma";
  float low = 3.0f;
  float high = 3.0f;
  int max_iters = 3;
  float min_fraction = 0.4f;
};

struct QualityFilterConfig {
  std::string mode = "auto";
  int min_stars = 30;
  float max_fwhm_sigma = 2.0f;
  float max_eccentricity = 0.65f;
  float min_correlation = 0.75f;
};

struct ManualFrameOverride {
  int index = -1;
  std::string filename;
  bool include = true;
};

struct StackingConfig {
  std::string normalization = "addscale";
  std::string weighting = "quality";
  bool cosmetic_correction = false;
  float cosmetic_correction_sigma = 5.0f;
  bool per_frame_cosmetic_correction = false;
  float per_frame_cosmetic_correction_sigma = 5.0f;
};

struct CalibrationConfig {
  bool use_bias = false;
  bool use_dark = false;
  bool use_flat = false;
  bool bias_use_master = false;
  bool dark_use_master = false;
  bool flat_use_master = false;
  bool darkflat_use_master = false;
  bool dark_auto_select = true;
  bool dark_match_use_temp = false;
  float dark_match_exposure_tolerance_percent = 8.0f;
  float dark_match_temp_tolerance_c = 3.0f;
  std::string bias_master;
  std::string dark_master;
  std::string flat_master;
  std::string darkflat_master;
  std::string pattern = "*.fit;*.fits;*.fts;*.fit.fz;*.fits.fz;*.fts.fz";
};

struct PostprocessConfig {
  bool astrometry = true;
  bool bge = true;
  bool pcc = true;
  bool hypermetric_stretch = true;
};

struct HyperMetricStretchConfig {
  bool require_successful_pcc = true;
  std::string mode = "ready_to_use";
  std::string sensor_profile = "rec709";
  std::string fallback_profile = "rec709";
  bool adaptive_anchor = true;
  float target_bg = 0.15f;
  float protect_b = 6.0f;
  float convergence_power = 3.5f;
  std::string log_d_mode = "auto";
  float fixed_log_d = 2.0f;
  std::string color_strategy = "fixed";
  float fixed_color_strategy = 0.0f;
  float color_grip = 1.0f;
  float shadow_convergence = 0.0f;
  float linear_expansion = 0.0f;
  bool write_channels = false;
  std::string output_rgb = "stacked_rgb_hms.fits";
};

struct ReportConfig {
  bool detailed = true;
  std::vector<std::string> formats{"json", "markdown", "html"};
};

struct RuntimeLimitsConfig {
  int parallel_workers = 4;
  int memory_budget = 512;
};

struct Config {
  std::string mode = "linear_prestack";
  std::string lights_dir;
  std::string bias_dir;
  std::string darks_dir;
  std::string flats_dir;
  std::string darkflats_dir;
  std::string input_mode = "auto";
  std::string raw_formats = "tile_compile";
  std::string bayer_pattern = "auto";
  std::string cfa_mode = "tile_compile";
  std::string mono_mode = "auto";
  std::string registration_reference = "best_quality";
  CalibrationConfig calibration;
  RejectionConfig rejection;
  QualityFilterConfig quality_filter;
  std::vector<ManualFrameOverride> manual_frame_overrides;
  StackingConfig stacking;
  PostprocessConfig postprocess;
  bool has_astrometry_config = false;
  bool has_bge_config = false;
  bool has_tile_config = false;
  bool has_pcc_config = false;
  config::AstrometryConfig astrometry;
  config::BGEConfig bge;
  config::TileConfig tile;
  config::PCCConfig pcc;
  HyperMetricStretchConfig hypermetric_stretch;
  ReportConfig report;
  RuntimeLimitsConfig runtime_limits;
};

std::string phase_to_string(Phase phase);
const std::vector<Phase>& phase_order();
const std::vector<std::string>& parameter_groups();
void validate(const Config& config);

} // namespace tile_compile::preprocessing
