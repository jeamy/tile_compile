#include "tile_compile/preprocessing/contract.hpp"

#include "tile_compile/core/errors.hpp"

#include <algorithm>
#include <initializer_list>

namespace tile_compile::preprocessing {

namespace {

bool one_of(const std::string& value, std::initializer_list<const char*> allowed) {
  return std::find_if(allowed.begin(), allowed.end(), [&](const char* item) {
           return value == item;
         }) != allowed.end();
}

bool in_unit_interval(float value) {
  return value >= 0.0f && value <= 1.0f;
}

} // namespace

std::string phase_to_string(Phase phase) {
  switch (phase) {
    case Phase::INPUT_SCAN:
      return "INPUT_SCAN";
    case Phase::CALIBRATION:
      return "CALIBRATION";
    case Phase::CFA_CHANNEL_PREP:
      return "CFA_CHANNEL_PREP";
    case Phase::REFERENCE_SELECTION:
      return "REFERENCE_SELECTION";
    case Phase::REGISTRATION:
      return "REGISTRATION";
    case Phase::QUALITY_ANALYSIS:
      return "QUALITY_ANALYSIS";
    case Phase::FRAME_FILTERING:
      return "FRAME_FILTERING";
    case Phase::STACKING:
      return "STACKING";
    case Phase::ASTROMETRY:
      return "ASTROMETRY";
    case Phase::BGE:
      return "BGE";
    case Phase::PCC:
      return "PCC";
    case Phase::HYPERMETRIC_STRETCH:
      return "HYPERMETRIC_STRETCH";
    case Phase::REPORT:
      return "REPORT";
  }
  return "UNKNOWN";
}

const std::vector<Phase>& phase_order() {
  static const std::vector<Phase> phases = {
      Phase::INPUT_SCAN,
      Phase::CALIBRATION,
      Phase::CFA_CHANNEL_PREP,
      Phase::REFERENCE_SELECTION,
      Phase::REGISTRATION,
      Phase::QUALITY_ANALYSIS,
      Phase::FRAME_FILTERING,
      Phase::STACKING,
      Phase::ASTROMETRY,
      Phase::BGE,
      Phase::PCC,
      Phase::HYPERMETRIC_STRETCH,
      Phase::REPORT,
  };
  return phases;
}

const std::vector<std::string>& parameter_groups() {
  static const std::vector<std::string> groups = {
      "input",
      "calibration",
      "cfa_mono",
      "registration",
      "quality_filter",
      "stacking",
      "postprocess",
      "hypermetric_stretch",
      "report",
      "runtime_limits",
  };
  return groups;
}

void validate(const Config& config) {
  if (config.mode != "linear_prestack") {
    throw ValidationError("preprocessing.mode must be 'linear_prestack'");
  }
  if (!one_of(config.input_mode, {"auto", "cfa_osc", "mono"})) {
    throw ValidationError("preprocessing.input_mode must be auto, cfa_osc, or mono");
  }
  if (config.raw_formats != "tile_compile") {
    throw ValidationError("preprocessing.raw_formats must be 'tile_compile'");
  }
  if (config.cfa_mode != "tile_compile") {
    throw ValidationError("preprocessing.cfa_mode must be 'tile_compile'");
  }
  if (!one_of(config.mono_mode, {"auto", "mono"})) {
    throw ValidationError("preprocessing.mono_mode must be auto or mono");
  }
  if (config.input_mode == "mono" && config.bayer_pattern != "auto" &&
      config.bayer_pattern != "UNKNOWN" && !config.bayer_pattern.empty()) {
    throw ValidationError(
        "preprocessing.input_mode=mono requires bayer_pattern auto, UNKNOWN, or empty");
  }
  if (config.registration_reference != "best_quality") {
    throw ValidationError(
        "preprocessing.registration_reference must be 'best_quality'");
  }
  if (!one_of(config.rejection.method, {"sigma", "median", "winsor"})) {
    throw ValidationError("preprocessing.rejection.method must be sigma, median, or winsor");
  }
  if (config.rejection.low <= 0.0f || config.rejection.high <= 0.0f) {
    throw ValidationError("preprocessing.rejection low/high must be > 0");
  }
  if (config.rejection.max_iters < 1) {
    throw ValidationError("preprocessing.rejection.max_iters must be >= 1");
  }
  if (!in_unit_interval(config.rejection.min_fraction)) {
    throw ValidationError("preprocessing.rejection.min_fraction must be in [0,1]");
  }
  if (!one_of(config.quality_filter.mode, {"auto", "strict", "relaxed", "off"})) {
    throw ValidationError(
        "preprocessing.quality_filter.mode must be auto, strict, relaxed, or off");
  }
  if (config.quality_filter.min_stars < 0) {
    throw ValidationError("preprocessing.quality_filter.min_stars must be >= 0");
  }
  if (config.quality_filter.max_fwhm_sigma <= 0.0f) {
    throw ValidationError("preprocessing.quality_filter.max_fwhm_sigma must be > 0");
  }
  if (!in_unit_interval(config.quality_filter.max_eccentricity)) {
    throw ValidationError(
        "preprocessing.quality_filter.max_eccentricity must be in [0,1]");
  }
  if (!in_unit_interval(config.quality_filter.min_correlation)) {
    throw ValidationError(
        "preprocessing.quality_filter.min_correlation must be in [0,1]");
  }
  for (const auto& override_item : config.manual_frame_overrides) {
    if (override_item.index < 0 && override_item.filename.empty()) {
      throw ValidationError(
          "preprocessing.quality_filter.manual_overrides entries need index or filename");
    }
  }
  if (!one_of(config.stacking.normalization,
              {"addscale", "background", "median", "none"})) {
    throw ValidationError(
        "preprocessing.stacking.normalization must be addscale, background, median, or none");
  }
  if (!one_of(config.stacking.weighting, {"quality", "uniform"})) {
    throw ValidationError("preprocessing.stacking.weighting must be quality or uniform");
  }
  if (config.stacking.cosmetic_correction_sigma <= 0.0f) {
    throw ValidationError("preprocessing.stacking.cosmetic_correction_sigma must be > 0");
  }
  if (config.stacking.per_frame_cosmetic_correction_sigma <= 0.0f) {
    throw ValidationError("preprocessing.stacking.per_frame_cosmetic_correction_sigma must be > 0");
  }
  if (config.calibration.use_bias && config.calibration.bias_use_master &&
      config.calibration.bias_master.empty()) {
    throw ValidationError("preprocessing.calibration.bias_master must be set when bias_use_master=true");
  }
  if (config.calibration.use_dark && config.calibration.dark_use_master &&
      config.calibration.dark_master.empty()) {
    throw ValidationError("preprocessing.calibration.dark_master must be set when dark_use_master=true");
  }
  if (config.calibration.use_flat && config.calibration.flat_use_master &&
      config.calibration.flat_master.empty()) {
    throw ValidationError("preprocessing.calibration.flat_master must be set when flat_use_master=true");
  }
  if (config.calibration.dark_match_exposure_tolerance_percent < 0.0f) {
    throw ValidationError("preprocessing.calibration.dark_match_exposure_tolerance_percent must be >= 0");
  }
  if (config.calibration.dark_match_temp_tolerance_c < 0.0f) {
    throw ValidationError("preprocessing.calibration.dark_match_temp_tolerance_c must be >= 0");
  }
  if (config.postprocess.pcc && !config.postprocess.astrometry) {
    throw ValidationError(
        "preprocessing.postprocess.pcc requires preprocessing.postprocess.astrometry");
  }
  if (!one_of(config.hypermetric_stretch.mode, {"ready_to_use", "scientific"})) {
    throw ValidationError("preprocessing.hypermetric_stretch.mode must be ready_to_use or scientific");
  }
  if (config.hypermetric_stretch.target_bg < 0.05f ||
      config.hypermetric_stretch.target_bg > 0.50f) {
    throw ValidationError("preprocessing.hypermetric_stretch.target_bg must be in [0.05,0.50]");
  }
  if (config.hypermetric_stretch.protect_b < 0.1f) {
    throw ValidationError("preprocessing.hypermetric_stretch.protect_b must be >= 0.1");
  }
  if (config.hypermetric_stretch.convergence_power < 1.0f ||
      config.hypermetric_stretch.convergence_power > 10.0f) {
    throw ValidationError("preprocessing.hypermetric_stretch.convergence_power must be in [1,10]");
  }
  if (!one_of(config.hypermetric_stretch.log_d_mode, {"auto", "fixed"})) {
    throw ValidationError("preprocessing.hypermetric_stretch.log_d_mode must be auto or fixed");
  }
  if (config.hypermetric_stretch.fixed_log_d < 0.0f ||
      config.hypermetric_stretch.fixed_log_d > 7.0f) {
    throw ValidationError("preprocessing.hypermetric_stretch.fixed_log_d must be in [0,7]");
  }
  if (!one_of(config.hypermetric_stretch.color_strategy, {"auto", "fixed"})) {
    throw ValidationError("preprocessing.hypermetric_stretch.color_strategy must be auto or fixed");
  }
  if (config.hypermetric_stretch.fixed_color_strategy < -1.0f ||
      config.hypermetric_stretch.fixed_color_strategy > 1.0f) {
    throw ValidationError("preprocessing.hypermetric_stretch.fixed_color_strategy must be in [-1,1]");
  }
  if (!in_unit_interval(config.hypermetric_stretch.color_grip)) {
    throw ValidationError("preprocessing.hypermetric_stretch.color_grip must be in [0,1]");
  }
  if (config.hypermetric_stretch.shadow_convergence < 0.0f) {
    throw ValidationError("preprocessing.hypermetric_stretch.shadow_convergence must be >= 0");
  }
  if (!in_unit_interval(config.hypermetric_stretch.shadow_color_floor)) {
    throw ValidationError("preprocessing.hypermetric_stretch.shadow_color_floor must be in [0,1]");
  }
  if (!in_unit_interval(config.hypermetric_stretch.linear_expansion)) {
    throw ValidationError("preprocessing.hypermetric_stretch.linear_expansion must be in [0,1]");
  }
  if (config.hypermetric_stretch.output_rgb.empty()) {
    throw ValidationError("preprocessing.hypermetric_stretch.output_rgb must not be empty");
  }
  if (config.report.formats.empty()) {
    throw ValidationError("preprocessing.report.formats must not be empty");
  }
  for (const auto& format : config.report.formats) {
    if (!one_of(format, {"json", "markdown", "html"})) {
      throw ValidationError(
          "preprocessing.report.formats entries must be json, markdown, or html");
    }
  }
  if (config.tile.size_factor < 1) {
    throw ValidationError("preprocessing.tile.size_factor must be >= 1");
  }
  if (config.tile.min_size < 1) {
    throw ValidationError("preprocessing.tile.min_size must be >= 1");
  }
  if (config.tile.max_divisor < 1) {
    throw ValidationError("preprocessing.tile.max_divisor must be >= 1");
  }
  if (config.tile.overlap_fraction < 0.0f || config.tile.overlap_fraction > 0.5f) {
    throw ValidationError("preprocessing.tile.overlap_fraction must be in [0,0.5]");
  }
  if (config.runtime_limits.parallel_workers < 1) {
    throw ValidationError("preprocessing.runtime_limits.parallel_workers must be >= 1");
  }
  if (config.runtime_limits.memory_budget < 1) {
    throw ValidationError("preprocessing.runtime_limits.memory_budget must be >= 1");
  }
  if (!one_of(config.runtime_limits.acceleration_backend,
              {"auto", "cpu", "opencv_cuda", "opencv_opencl", "opencl",
               "cuda"})) {
    throw ValidationError(
        "preprocessing.runtime_limits.acceleration_backend is invalid");
  }
}

} // namespace tile_compile::preprocessing
