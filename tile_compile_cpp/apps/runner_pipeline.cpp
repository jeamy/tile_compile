#include "runner_pipeline.hpp"

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/acceleration.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/core/mode_gating.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/image/background_extraction.hpp"
#include "tile_compile/image/cfa_processing.hpp"
#include "tile_compile/image/hypermetric_stretch.hpp"
#include "tile_compile/image/normalization.hpp"
#include "tile_compile/image/processing.hpp"
#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/metrics/linearity.hpp"
#include "tile_compile/metrics/aqmh_quality_map_cache.hpp"
#include "tile_compile/metrics/aqmh_frame_valid_mask.hpp"
#include "tile_compile/metrics/metrics.hpp"
#include "tile_compile/pipeline/adaptive_tile_grid.hpp"
#include "tile_compile/reconstruction/reconstruction.hpp"
#include "tile_compile/reconstruction/aqmh_pipeline_overlap.hpp"
#include "tile_compile/reconstruction/aqmh_reconstruction.hpp"
#include "tile_compile/reconstruction/aqmh_validation.hpp"
#include "tile_compile/reconstruction/tile_boundary_diagnostics.hpp"
#include "tile_compile/reconstruction/tile_normalization.hpp"
#include "tile_compile/reconstruction/tile_weight_profile_diagnostics.hpp"
#include "tile_compile/astrometry/wcs.hpp"
#include "tile_compile/astrometry/gaia_catalog.hpp"
#include "tile_compile/astrometry/photometric_color_cal.hpp"

#include "runner_phase_metrics.hpp"
#include "runner_phase_local_metrics.hpp"
#include "runner_phase_aqmh_maps.hpp"
#include "runner_phase_aqmh_diagnostics.hpp"
#include "runner_phase_aqmh_reconstruction.hpp"
#include "runner_phase_registration.hpp"
#include "runner_shared.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <opencv2/opencv.hpp>

#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {
using tile_compile::ColorMode;
using tile_compile::Matrix2Df;
using tile_compile::Tile;
using tile_compile::WarpMatrix;

namespace image = tile_compile::image;
namespace astro = tile_compile::astrometry;
namespace core = tile_compile::core;
namespace io = tile_compile::io;
namespace reconstruction = tile_compile::reconstruction;
namespace runner = tile_compile::runner;
using tile_compile::runner::TeeBuf;
using tile_compile::runner::estimate_total_file_bytes;
using tile_compile::runner::format_bytes;
using tile_compile::runner::message_indicates_disk_full;
using tile_compile::runner::resolve_astap_binary_path;
using tile_compile::runner::shell_quote;
using tile_compile::runner::system_cmd;

using NormalizationScales = image::NormalizationScales;

image::HyperMetricStretchConfig to_image_hms_config(
    const tile_compile::config::HyperMetricStretchConfig &src) {
  image::HyperMetricStretchConfig dst;
  dst.enabled = src.enabled;
  dst.require_successful_pcc = src.require_successful_pcc;
  dst.mode = src.mode;
  dst.sensor_profile = src.sensor_profile;
  dst.fallback_profile = src.fallback_profile;
  dst.adaptive_anchor = src.adaptive_anchor;
  dst.target_bg = src.target_bg;
  dst.protect_b = src.protect_b;
  dst.convergence_power = src.convergence_power;
  dst.log_d_mode = src.log_d_mode;
  dst.fixed_log_d = src.fixed_log_d;
  dst.color_strategy = src.color_strategy;
  dst.fixed_color_strategy = src.fixed_color_strategy;
  dst.color_grip = src.color_grip;
  dst.shadow_convergence = src.shadow_convergence;
  dst.linear_expansion = src.linear_expansion;
  dst.write_channels = src.write_channels;
  dst.output_rgb = src.output_rgb;
  return dst;
}

constexpr float kTileNormBoundaryRegressionFactor = 8.0f;
constexpr float kTileNormBoundaryRegressionAbsP95 = 0.25f;
constexpr float kCalibrationFlatFloor = 1.0e-6f;
constexpr double kCalibrationGainMismatchWarningAbs = 0.25;
constexpr double kCalibrationGainMatchTolerance = 0.25;

struct CalibrationMaster {
  Matrix2Df data;
  std::string source_kind;
  std::string source_path;
  std::vector<fs::path> input_frames;
  float normalization_reference = 1.0f;
};

struct CalibrationRunResult {
  bool requested = false;
  bool applied = false;
  std::vector<fs::path> calibrated_frames;
  core::json artifact = core::json::object();
};

/// @brief Implements trim copy.
/// @details Part of the production runner pipeline that coordinates scan, registration, metrics, reconstruction, stacking, astrometry, BGE, and PCC phases; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::string trim_copy(std::string value) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(), not_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(),
              value.end());
  return value;
}

/// @brief Parses double string.
/// @details Part of the production runner pipeline that coordinates scan, registration, metrics, reconstruction, stacking, astrometry, BGE, and PCC phases; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::optional<double> parse_double_string(const std::string &text) {
  const std::string trimmed = trim_copy(text);
  if (trimmed.empty()) {
    return std::nullopt;
  }
  char *end = nullptr;
  errno = 0;
  const double value = std::strtod(trimmed.c_str(), &end);
  if (errno != 0 || end == trimmed.c_str() || (end != nullptr && *end != '\0') ||
      !std::isfinite(value)) {
    return std::nullopt;
  }
  return value;
}

/// @brief Reads header numeric.
/// @details Part of the production runner pipeline that coordinates scan, registration, metrics, reconstruction, stacking, astrometry, BGE, and PCC phases; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::optional<double> read_header_numeric(
    const io::FitsHeader &header, std::initializer_list<const char *> keys,
    bool require_positive) {
  for (const char *key : keys) {
    if (auto value = header.get_double(key);
        value && std::isfinite(*value) &&
        (!require_positive || *value > 0.0)) {
      return value;
    }
    if (auto value = header.get_int(key);
        value && (!require_positive || *value > 0)) {
      return static_cast<double>(*value);
    }
    if (auto value = header.get_string(key)) {
      if (auto parsed = parse_double_string(*value);
          parsed && std::isfinite(*parsed) &&
          (!require_positive || *parsed > 0.0)) {
        return parsed;
      }
    }
  }
  return std::nullopt;
}

/// @brief Extracts exposure seconds.
/// @details Part of the production runner pipeline that coordinates scan, registration, metrics, reconstruction, stacking, astrometry, BGE, and PCC phases; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::optional<double> extract_exposure_seconds(const io::FitsHeader &header) {
  return read_header_numeric(
      header, {"EXPTIME", "EXPOSURE", "EXPOSURETIME", "EXPOSURE_TIME",
               "EXP_TIME", "DURATION", "EXPOS"},
      true);
}

/// @brief Extracts temperature celsius.
/// @details Part of the production runner pipeline that coordinates scan, registration, metrics, reconstruction, stacking, astrometry, BGE, and PCC phases; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::optional<double> extract_temperature_celsius(const io::FitsHeader &header) {
  return read_header_numeric(
      header, {"CCD-TEMP", "CCD_TEMP", "CCD_TEMP_C", "SENSOR_T",
               "SENSORTEMP", "TEMP", "TEMPERAT"},
      false);
}

/// @brief Extracts gain value.
/// @details Part of the production runner pipeline that coordinates scan, registration, metrics, reconstruction, stacking, astrometry, BGE, and PCC phases; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::optional<double> extract_gain_value(const io::FitsHeader &header) {
  return read_header_numeric(header, {"GAIN"}, true);
}

template <typename Extractor>
/// @brief Implements sample header median.
/// @details Part of the production runner pipeline that coordinates scan, registration, metrics, reconstruction, stacking, astrometry, BGE, and PCC phases; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::optional<double> sample_header_median(const std::vector<fs::path> &paths,
                                           size_t max_samples,
                                           Extractor extractor) {
  if (paths.empty()) {
    return std::nullopt;
  }
  const size_t sample_count = std::min(max_samples, paths.size());
  std::vector<float> values;
  values.reserve(sample_count);
  for (size_t i = 0; i < sample_count; ++i) {
    try {
      const io::FitsHeader hdr = io::read_fits_header(paths[i]);
      if (auto value = extractor(hdr);
          value && std::isfinite(*value)) {
        values.push_back(static_cast<float>(*value));
      }
    } catch (const std::exception &) {
    }
  }
  if (values.empty()) {
    return std::nullopt;
  }
  return static_cast<double>(core::median_of(values));
}

/// @brief Implements warn if gain mismatch.
/// @details Part of the production runner pipeline that coordinates scan, registration, metrics, reconstruction, stacking, astrometry, BGE, and PCC phases; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void warn_if_gain_mismatch(const std::vector<fs::path> &light_frames,
                           const std::vector<fs::path> &calibration_frames,
                           const std::string &calibration_label,
                           const std::string &run_id,
                           core::EventEmitter &emitter,
                           std::ostream &log_file,
                           core::json &artifact_step) {
  const auto light_gain =
      sample_header_median(light_frames, 10, extract_gain_value);
  const auto calibration_gain =
      sample_header_median(calibration_frames, 10, extract_gain_value);
  if (!light_gain || !calibration_gain) {
    return;
  }
  artifact_step["light_gain"] = *light_gain;
  artifact_step["calibration_gain"] = *calibration_gain;
  const double diff = std::fabs(*light_gain - *calibration_gain);
  if (diff <= kCalibrationGainMismatchWarningAbs) {
    return;
  }
  artifact_step["gain_mismatch_warning"] = true;
  emitter.warning(
      run_id,
      "Calibration " + calibration_label + " gain mismatch: lights use GAIN " +
          std::to_string(*light_gain) + ", calibration uses GAIN " +
          std::to_string(*calibration_gain),
      log_file);
}

bool require_gain_match(const std::vector<fs::path> &light_frames,
                        const std::vector<fs::path> &calibration_frames,
                        const std::string &calibration_label,
                        core::json &artifact_step,
                        std::string &error_out) {
  const auto light_gain =
      sample_header_median(light_frames, 10, extract_gain_value);
  const auto calibration_gain =
      sample_header_median(calibration_frames, 10, extract_gain_value);
  if (!light_gain) {
    error_out = "Calibration " + calibration_label +
                " rejected: light GAIN header is missing";
    return false;
  }
  if (!calibration_gain) {
    error_out = "Calibration " + calibration_label +
                " rejected: calibration GAIN header is missing";
    artifact_step["light_gain"] = *light_gain;
    return false;
  }
  artifact_step["light_gain"] = *light_gain;
  artifact_step["calibration_gain"] = *calibration_gain;
  const double diff = std::fabs(*light_gain - *calibration_gain);
  if (diff <= kCalibrationGainMatchTolerance) {
    return true;
  }
  artifact_step["gain_mismatch_error"] = true;
  error_out = "Calibration " + calibration_label +
              " rejected: lights use GAIN " + std::to_string(*light_gain) +
              ", calibration uses GAIN " +
              std::to_string(*calibration_gain);
  return false;
}

/// @brief Resolves config path.
/// @details Part of the production runner pipeline that coordinates scan, registration, metrics, reconstruction, stacking, astrometry, BGE, and PCC phases; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
fs::path resolve_config_path(const fs::path &project_root,
                             const std::string &raw_path) {
  const std::string trimmed = trim_copy(raw_path);
  if (trimmed.empty()) {
    return {};
  }
  fs::path path(trimmed);
  if (path.is_relative()) {
    path = project_root / path;
  }
  std::error_code ec;
  const fs::path absolute = fs::absolute(path, ec);
  return ec ? path : absolute;
}

/// @brief Loads average master from files.
/// @details Part of the production runner pipeline that coordinates scan, registration, metrics, reconstruction, stacking, astrometry, BGE, and PCC phases; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool load_average_master_from_files(const std::vector<fs::path> &paths,
                                    int expected_height, int expected_width,
                                    Matrix2Df &out, std::string &error_out) {
  if (paths.empty()) {
    error_out = "no calibration frames found";
    return false;
  }

  Matrix2Df accum;
  bool first = true;
  size_t loaded = 0;
  for (const auto &path : paths) {
    Matrix2Df img;
    try {
      img = io::read_fits_pixels_float(path);
    } catch (const std::exception &e) {
      error_out = "failed to read calibration frame '" + path.string() +
                  "': " + e.what();
      return false;
    }
    if (img.rows() != expected_height || img.cols() != expected_width) {
      error_out = "calibration frame dimension mismatch for '" + path.string() +
                  "': expected " + std::to_string(expected_width) + "x" +
                  std::to_string(expected_height) + ", got " +
                  std::to_string(img.cols()) + "x" +
                  std::to_string(img.rows());
      return false;
    }
    if (first) {
      accum = img;
      first = false;
    } else {
      accum += img;
    }
    ++loaded;
  }
  if (loaded == 0) {
    error_out = "no readable calibration frames found";
    return false;
  }
  out = accum / static_cast<float>(loaded);
  return true;
}

/// @brief Normalizes flat master.
/// @details Part of the production runner pipeline that coordinates scan, registration, metrics, reconstruction, stacking, astrometry, BGE, and PCC phases; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool normalize_flat_master(Matrix2Df &flat, float &median_out,
                           std::string &error_out) {
  std::vector<float> samples;
  samples.reserve(static_cast<size_t>(flat.size()));
  for (Eigen::Index i = 0; i < flat.size(); ++i) {
    const float v = flat.data()[i];
    if (std::isfinite(v) && v > kCalibrationFlatFloor) {
      samples.push_back(v);
    }
  }
  if (samples.empty()) {
    error_out = "flat master has no finite positive samples";
    return false;
  }
  median_out = core::median_of(samples);
  if (!(std::isfinite(median_out) && median_out > kCalibrationFlatFloor)) {
    error_out = "flat master normalization median is invalid";
    return false;
  }
  flat.array() /= median_out;
  return true;
}

/// @brief Implements discover calibration frames.
/// @details Part of the production runner pipeline that coordinates scan, registration, metrics, reconstruction, stacking, astrometry, BGE, and PCC phases; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::vector<fs::path> discover_calibration_frames(const fs::path &dir,
                                                  const std::string &pattern) {

  auto frames = core::discover_frames(dir, pattern);
  frames.erase(
      std::remove_if(frames.begin(), frames.end(),
                     [](const fs::path &p) { return !io::is_fits_image_path(p); }),
      frames.end());
  std::sort(frames.begin(), frames.end());
  return frames;
}

/// @brief Implements select dark inputs.
/// @details Part of the production runner pipeline that coordinates scan, registration, metrics, reconstruction, stacking, astrometry, BGE, and PCC phases; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::vector<fs::path> select_dark_inputs(
    const std::vector<fs::path> &all_darks, const std::vector<fs::path> &lights,
    const tile_compile::config::CalibrationConfig &cfg,
    core::json &selection_info) {
  selection_info = {
      {"enabled", cfg.dark_auto_select},
      {"candidate_count", static_cast<int>(all_darks.size())},
  };
  if (!cfg.dark_auto_select || all_darks.empty()) {
    selection_info["used_all_candidates"] = true;
    return all_darks;
  }

  const size_t sample_count = std::min<size_t>(10, lights.size());
  std::vector<float> light_exposures;
  std::vector<float> light_temps;
  std::vector<float> light_gains;
  light_exposures.reserve(sample_count);
  light_temps.reserve(sample_count);
  light_gains.reserve(sample_count);
  for (size_t i = 0; i < sample_count; ++i) {
    try {
      const io::FitsHeader hdr = io::read_fits_header(lights[i]);
      if (auto exptime = extract_exposure_seconds(hdr)) {
        light_exposures.push_back(static_cast<float>(*exptime));
      }
      if (cfg.dark_match_use_temp) {
        if (auto temp = extract_temperature_celsius(hdr)) {
          light_temps.push_back(static_cast<float>(*temp));
        }
      }
      if (auto gain = extract_gain_value(hdr)) {
        light_gains.push_back(static_cast<float>(*gain));
      }
    } catch (const std::exception &) {
    }
  }

  if (light_gains.empty()) {
    selection_info["used_all_candidates"] = false;
    selection_info["fallback_reason"] = "light_gain_unknown";
    selection_info["matched_count"] = 0;
    return {};
  }
  const float light_gain_median = core::median_of(light_gains);
  selection_info["light_gain"] = light_gain_median;

  if (light_exposures.empty()) {
    selection_info["used_all_candidates"] = false;
    selection_info["fallback_reason"] = "light_exposure_unknown";
    selection_info["matched_count"] = 0;
    return {};
  }

  const float light_exposure_median = core::median_of(light_exposures);
  selection_info["light_exposure_seconds"] = light_exposure_median;
  const bool require_temp =
      cfg.dark_match_use_temp && !light_temps.empty();
  float light_temp_median = 0.0f;
  if (require_temp) {
    light_temp_median = core::median_of(light_temps);
    selection_info["light_temperature_c"] = light_temp_median;
  }

  std::vector<fs::path> matched;
  matched.reserve(all_darks.size());
  int missing_exposure = 0;
  int missing_temp = 0;
  int missing_gain = 0;
  int gain_mismatch = 0;
  const float exposure_tolerance =
      std::max(0.0f, cfg.dark_match_exposure_tolerance_percent) / 100.0f;
  for (const auto &path : all_darks) {
    io::FitsHeader hdr;
    try {
      hdr = io::read_fits_header(path);
    } catch (const std::exception &) {
      continue;
    }
    const auto dark_gain = extract_gain_value(hdr);
    if (!dark_gain || !std::isfinite(*dark_gain)) {
      ++missing_gain;
      continue;
    }
    if (std::fabs(*dark_gain - light_gain_median) >
        kCalibrationGainMatchTolerance) {
      ++gain_mismatch;
      continue;
    }
    const auto dark_exposure = extract_exposure_seconds(hdr);
    if (!dark_exposure || !std::isfinite(*dark_exposure) ||
        *dark_exposure <= 0.0) {
      ++missing_exposure;
      continue;
    }
    const double rel_diff =
        std::fabs(*dark_exposure - light_exposure_median) /
        std::max<double>(light_exposure_median, 1.0e-12);
    if (rel_diff > exposure_tolerance) {
      continue;
    }
    if (require_temp) {
      const auto dark_temp = extract_temperature_celsius(hdr);
      if (!dark_temp || !std::isfinite(*dark_temp)) {
        ++missing_temp;
        continue;
      }
      if (std::fabs(*dark_temp - light_temp_median) >
          cfg.dark_match_temp_tolerance_c) {
        continue;
      }
    }
    matched.push_back(path);
  }

  selection_info["missing_exposure_headers"] = missing_exposure;
  selection_info["missing_gain_headers"] = missing_gain;
  selection_info["gain_mismatch_count"] = gain_mismatch;
  selection_info["gain_tolerance_abs"] = kCalibrationGainMatchTolerance;
  if (require_temp) {
    selection_info["missing_temperature_headers"] = missing_temp;
    selection_info["temperature_tolerance_c"] =
        cfg.dark_match_temp_tolerance_c;
  }
  selection_info["exposure_tolerance_percent"] =
      cfg.dark_match_exposure_tolerance_percent;
  selection_info["matched_count"] = static_cast<int>(matched.size());

  if (!matched.empty()) {
    selection_info["used_all_candidates"] = false;
    return matched;
  }

  // No darks matched gain/exposure — fall back to all darks with warning
  selection_info["used_all_candidates"] = true;
  selection_info["fallback_reason"] = "no_matching_darks_gain_or_exposure";
  return all_darks;
}

/// @brief Resolves calibration master.
/// @details Part of the production runner pipeline that coordinates scan, registration, metrics, reconstruction, stacking, astrometry, BGE, and PCC phases; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool resolve_calibration_master(
    const fs::path &project_root, const std::string &explicit_master_raw,
    const std::string &dir_raw, const std::string &pattern,
    bool prefer_explicit_master, int expected_height, int expected_width,
    CalibrationMaster &out, std::string &error_out,
    const std::vector<fs::path> *preset_inputs = nullptr) {
  const fs::path explicit_master =
      resolve_config_path(project_root, explicit_master_raw);
  const fs::path dir = resolve_config_path(project_root, dir_raw);

  auto load_explicit = [&](const fs::path &path) -> bool {
    if (path.empty()) {
      error_out = "explicit master path is empty";
      return false;
    }
    if (!fs::exists(path)) {
      error_out = "explicit master not found: " + path.string();
      return false;
    }
    Matrix2Df master;
    if (!load_average_master_from_files({path}, expected_height, expected_width,
                                        master, error_out)) {
      return false;
    }
    out.data = std::move(master);
    out.source_kind = "explicit_master";
    out.source_path = path.string();
    out.input_frames = {path};
    return true;
  };

  auto load_from_dir = [&](const fs::path &directory) -> bool {
    if (directory.empty()) {
      error_out = "calibration directory path is empty";
      return false;
    }
    if (!fs::exists(directory) || !fs::is_directory(directory)) {
      error_out = "calibration directory not found: " + directory.string();
      return false;
    }
    std::vector<fs::path> frames =
        preset_inputs ? *preset_inputs
                      : discover_calibration_frames(directory, pattern);
    if (frames.empty()) {
      error_out = "no calibration frames found in " + directory.string();
      return false;
    }
    Matrix2Df master;
    if (!load_average_master_from_files(frames, expected_height, expected_width,
                                        master, error_out)) {
      return false;
    }
    out.data = std::move(master);
    out.source_kind = "directory_average";
    out.source_path = directory.string();
    out.input_frames = std::move(frames);
    return true;
  };

  if (prefer_explicit_master && !explicit_master.empty()) {
    if (load_explicit(explicit_master)) {
      return true;
    }
    if (!dir.empty()) {
      error_out.clear();
      if (load_from_dir(dir)) {
        return true;
      }
    }
    return false;
  }

  if (!dir.empty()) {
    if (load_from_dir(dir)) {
      return true;
    }
    if (!explicit_master.empty()) {
      error_out.clear();
      if (load_explicit(explicit_master)) {
        return true;
      }
    }
    return false;
  }

  if (!explicit_master.empty()) {
    return load_explicit(explicit_master);
  }

  error_out = "no master file or calibration directory configured";
  return false;
}

/// @brief Runs scan input calibration.
/// @details Part of the production runner pipeline that coordinates scan, registration, metrics, reconstruction, stacking, astrometry, BGE, and PCC phases; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool run_scan_input_calibration(
    const tile_compile::config::Config &cfg, const fs::path &project_root,
    const std::vector<fs::path> &input_frames, const fs::path &run_dir,
    const std::string &run_id, core::EventEmitter &emitter,
    std::ostream &log_file, ColorMode detected_mode, int image_height,
    int image_width, CalibrationRunResult &out, std::string &error_out) {
  out = CalibrationRunResult{};
  const auto &cal = cfg.calibration;
  out.requested = cal.use_bias || cal.use_dark || cal.use_flat;
  out.artifact["requested"] = out.requested;
  out.artifact["steps"] = {
      {"bias", {{"enabled", cal.use_bias}}},
      {"dark", {{"enabled", cal.use_dark}}},
      {"flat", {{"enabled", cal.use_flat}}},
  };

  if (!out.requested) {
    out.calibrated_frames = input_frames;
    out.artifact["applied"] = false;
    return true;
  }

  if (detected_mode == ColorMode::RGB) {
    error_out =
        "calibration.* is only supported for mono/CFA FITS inputs, not RGB cubes";
    return false;
  }

  CalibrationMaster bias_master;
  CalibrationMaster dark_master;
  CalibrationMaster flat_master;

  if (cal.use_bias) {
    if (!resolve_calibration_master(
            project_root, cal.bias_master, cal.bias_dir, cal.pattern,
            cal.bias_use_master, image_height, image_width, bias_master,
            error_out)) {
      return false;
    }
    out.artifact["steps"]["bias"]["source"] = bias_master.source_kind;
    out.artifact["steps"]["bias"]["path"] = bias_master.source_path;
    out.artifact["steps"]["bias"]["input_count"] =
        static_cast<int>(bias_master.input_frames.size());
    warn_if_gain_mismatch(input_frames, bias_master.input_frames, "bias", run_id,
                          emitter, log_file, out.artifact["steps"]["bias"]);
  }

  std::vector<fs::path> selected_dark_inputs;
  if (cal.use_dark) {
    core::json dark_selection;
    const fs::path dark_dir = resolve_config_path(project_root, cal.darks_dir);
    if (!dark_dir.empty() && fs::exists(dark_dir) && fs::is_directory(dark_dir)) {
      const auto all_darks = discover_calibration_frames(dark_dir, cal.pattern);
      selected_dark_inputs =
          select_dark_inputs(all_darks, input_frames, cal, dark_selection);
      if (selected_dark_inputs.empty() && !all_darks.empty()) {
        // No darks matched but candidates exist — warn and use all
        emitter.warning(
            run_id,
            "Calibration dark: no dark frames matched light GAIN and exposure, using all darks as fallback",
            log_file);
        selected_dark_inputs = all_darks;
      } else if (selected_dark_inputs.empty()) {
        out.artifact["steps"]["dark"]["selection"] = dark_selection;
        error_out =
            "Calibration dark rejected: no dark frames found in darks_dir";
        return false;
      }
      if (dark_selection.value("used_all_candidates", false) &&
          dark_selection.contains("fallback_reason")) {
        emitter.warning(
            run_id,
            "Calibration dark auto-selection fell back to all darks: " +
                dark_selection["fallback_reason"].get<std::string>(),
            log_file);
      }
    }
    if (!resolve_calibration_master(
            project_root, cal.dark_master, cal.darks_dir, cal.pattern,
            cal.dark_use_master, image_height, image_width, dark_master,
            error_out,
            selected_dark_inputs.empty() ? nullptr : &selected_dark_inputs)) {
      return false;
    }
    out.artifact["steps"]["dark"]["source"] = dark_master.source_kind;
    out.artifact["steps"]["dark"]["path"] = dark_master.source_path;
    out.artifact["steps"]["dark"]["input_count"] =
        static_cast<int>(dark_master.input_frames.size());
    out.artifact["steps"]["dark"]["selection"] = dark_selection;
    warn_if_gain_mismatch(input_frames, dark_master.input_frames, "dark",
                           run_id, emitter, log_file,
                           out.artifact["steps"]["dark"]);
    const bool dark_needs_bias_correction =
        cal.use_bias && !cal.dark_already_bias_corrected;
    out.artifact["steps"]["dark"]["bias_corrected_before_apply"] =
        dark_needs_bias_correction;
    if (dark_needs_bias_correction) {
      dark_master.data -= bias_master.data;
    }
  }

  if (cal.use_flat) {
    if (!resolve_calibration_master(
            project_root, cal.flat_master, cal.flats_dir, cal.pattern,
            cal.flat_use_master, image_height, image_width, flat_master,
            error_out)) {
      return false;
    }
    float flat_median = 1.0f;
    if (!normalize_flat_master(flat_master.data, flat_median, error_out)) {
      return false;
    }
    flat_master.normalization_reference = flat_median;
    out.artifact["steps"]["flat"]["source"] = flat_master.source_kind;
    out.artifact["steps"]["flat"]["path"] = flat_master.source_path;
    out.artifact["steps"]["flat"]["input_count"] =
        static_cast<int>(flat_master.input_frames.size());
    out.artifact["steps"]["flat"]["normalization_median"] = flat_median;
    warn_if_gain_mismatch(input_frames, flat_master.input_frames, "flat", run_id,
                          emitter, log_file, out.artifact["steps"]["flat"]);
  }

  const fs::path calibrated_dir = run_dir / "outputs" / "calibrated";
  fs::create_directories(calibrated_dir);
  out.calibrated_frames.clear();
  out.calibrated_frames.reserve(input_frames.size());

  for (size_t i = 0; i < input_frames.size(); ++i) {
    Matrix2Df light;
    io::FitsHeader header;
    try {
      std::tie(light, header) = io::read_fits_float(input_frames[i]);
    } catch (const std::exception &e) {
      error_out = "failed to read light frame '" + input_frames[i].string() +
                  "': " + e.what();
      return false;
    }
    if (light.rows() != image_height || light.cols() != image_width) {
      error_out = "light frame dimension mismatch during calibration for '" +
                  input_frames[i].string() + "'";
      return false;
    }

    Matrix2Df calibrated = light;
    if (cal.use_bias) {
      calibrated -= bias_master.data;
    }
    if (cal.use_dark) {
      calibrated -= dark_master.data;
    }
    if (cal.use_flat) {
      for (Eigen::Index px = 0; px < calibrated.size(); ++px) {
        const float denom = flat_master.data.data()[px];
        if (std::isfinite(denom) && denom > kCalibrationFlatFloor) {
          calibrated.data()[px] /= denom;
        }
      }
    }

    header.set("CALIBRAT", true);
    header.set("BIASCORR", cal.use_bias);
    header.set("DARKCORR", cal.use_dark);
    header.set("FLATCORR", cal.use_flat);

    std::ostringstream name;
    name << "cal_" << std::setfill('0') << std::setw(5) << (i + 1) << ".fit";
    const fs::path out_path = calibrated_dir / name.str();
    try {
      io::write_fits_float(out_path, calibrated, header);
    } catch (const std::exception &e) {
      error_out = "failed to write calibrated frame '" + out_path.string() +
                  "': " + e.what();
      return false;
    }
    out.calibrated_frames.push_back(out_path);
  }

  out.applied = true;
  out.artifact["applied"] = true;
  out.artifact["frame_count"] = static_cast<int>(out.calibrated_frames.size());
  out.artifact["output_dir"] = calibrated_dir.string();
  return true;
}

/// @brief Implements tile grid key.
/// @details Part of the production runner pipeline that coordinates scan, registration, metrics, reconstruction, stacking, astrometry, BGE, and PCC phases; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
uint64_t tile_grid_key(int row, int col) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(row)) << 32) ^
         static_cast<uint32_t>(col);
}

struct TileWindowCacheEntry {
  std::vector<float> x;
  std::vector<float> y;
};

struct TileOlaCoeffCacheEntry {
  Matrix2Df coeff;
  bool has_nonzero = false;
};

std::vector<TileWindowCacheEntry>
/// @brief Builds tile window cache.
/// @details Part of the production runner pipeline that coordinates scan, registration, metrics, reconstruction, stacking, astrometry, BGE, and PCC phases; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
build_tile_window_cache(const std::vector<Tile> &tiles) {
  std::vector<TileWindowCacheEntry> out(tiles.size());
  std::unordered_map<uint64_t, size_t> tile_by_grid;
  tile_by_grid.reserve(tiles.size());
  for (size_t ti = 0; ti < tiles.size(); ++ti) {
    tile_by_grid.emplace(tile_grid_key(tiles[ti].row, tiles[ti].col), ti);
  }

  for (size_t ti = 0; ti < tiles.size(); ++ti) {
    const auto &tile = tiles[ti];
    int left_overlap = 0;
    int right_overlap = 0;
    int top_overlap = 0;
    int bottom_overlap = 0;

    auto left_it = tile_by_grid.find(tile_grid_key(tile.row, tile.col - 1));
    if (left_it != tile_by_grid.end()) {
      const auto &nbr = tiles[left_it->second];
      left_overlap = std::max(0, (nbr.x + nbr.width) - tile.x);
    }
    auto right_it = tile_by_grid.find(tile_grid_key(tile.row, tile.col + 1));
    if (right_it != tile_by_grid.end()) {
      const auto &nbr = tiles[right_it->second];
      right_overlap = std::max(0, (tile.x + tile.width) - nbr.x);
    }
    auto up_it = tile_by_grid.find(tile_grid_key(tile.row - 1, tile.col));
    if (up_it != tile_by_grid.end()) {
      const auto &nbr = tiles[up_it->second];
      top_overlap = std::max(0, (nbr.y + nbr.height) - tile.y);
    }
    auto down_it = tile_by_grid.find(tile_grid_key(tile.row + 1, tile.col));
    if (down_it != tile_by_grid.end()) {
      const auto &nbr = tiles[down_it->second];
      bottom_overlap = std::max(0, (tile.y + tile.height) - nbr.y);
    }

    out[ti].x = reconstruction::make_partition_window_1d(
        tile.width, left_overlap, right_overlap);
    out[ti].y = reconstruction::make_partition_window_1d(
        tile.height, top_overlap, bottom_overlap);
  }
  return out;
}

/// @brief Builds tile ola coeff cache.
/// @details Part of the production runner pipeline that coordinates scan, registration, metrics, reconstruction, stacking, astrometry, BGE, and PCC phases; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::vector<TileOlaCoeffCacheEntry> build_tile_ola_coeff_cache(
    const std::vector<Tile> &tiles,
    const std::vector<TileWindowCacheEntry> &tile_window_cache,
    const std::vector<uint8_t> &common_valid_mask, int canvas_width,
    int canvas_height) {
  std::vector<TileOlaCoeffCacheEntry> out(tiles.size());
  for (size_t ti = 0; ti < tiles.size(); ++ti) {
    if (ti >= tile_window_cache.size()) {
      continue;
    }
    const Tile &tile = tiles[ti];
    auto &entry = out[ti];
    if (tile.width <= 0 || tile.height <= 0) {
      continue;
    }
    entry.coeff.resize(tile.height, tile.width);
    entry.coeff.setZero();
    for (int yy = 0; yy < tile.height; ++yy) {
      const int iy = tile.y + yy;
      if (iy < 0 || iy >= canvas_height) {
        continue;
      }
      const float wy = tile_window_cache[ti].y[static_cast<size_t>(yy)];
      for (int xx = 0; xx < tile.width; ++xx) {
        const int ix = tile.x + xx;
        if (ix < 0 || ix >= canvas_width) {
          continue;
        }
        const size_t common_idx =
            static_cast<size_t>(iy) * static_cast<size_t>(canvas_width) +
            static_cast<size_t>(ix);
        if (common_idx >= common_valid_mask.size() ||
            common_valid_mask[common_idx] == 0) {
          continue;
        }
        const float coeff =
            wy * tile_window_cache[ti].x[static_cast<size_t>(xx)];
        entry.coeff(yy, xx) = coeff;
        entry.has_nonzero = entry.has_nonzero || (coeff > 0.0f);
      }
    }
  }
  return out;
}

/// @brief Implements safe boundary metric.
/// @details Part of the production runner pipeline that coordinates scan, registration, metrics, reconstruction, stacking, astrometry, BGE, and PCC phases; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float safe_boundary_metric(float value) {
  constexpr float kBoundaryMetricFloor = 1.0e-4f;
  if (!std::isfinite(value)) {
    return kBoundaryMetricFloor;
  }
  return std::max(value, kBoundaryMetricFloor);
}

/// @brief Decides whether to disable phase7 tile norm.
/// @details Part of the production runner pipeline that coordinates scan, registration, metrics, reconstruction, stacking, astrometry, BGE, and PCC phases; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool should_disable_phase7_tile_norm(
    const reconstruction::TileBoundaryDiagnostics &raw,
    const reconstruction::TileBoundaryDiagnostics &normalized,
    float *ratio_out) {
  const float raw_metric = safe_boundary_metric(raw.pair_mean_abs_diff_p95);
  const float normalized_metric =
      safe_boundary_metric(normalized.pair_mean_abs_diff_p95);
  const float ratio = normalized_metric / raw_metric;
  if (ratio_out != nullptr) {
    *ratio_out = ratio;
  }
  if (!std::isfinite(normalized.pair_mean_abs_diff_p95)) {
    return true;
  }
  return normalized_metric > kTileNormBoundaryRegressionAbsP95 &&
         ratio > kTileNormBoundaryRegressionFactor;
}

/// @brief Writes canvas mask fits.
/// @details Part of the production runner pipeline that coordinates scan, registration, metrics, reconstruction, stacking, astrometry, BGE, and PCC phases; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool write_canvas_mask_fits(const fs::path &mask_path,
                            const std::vector<uint8_t> &mask,
                            int rows, int cols,
                            const tile_compile::io::FitsHeader &header,
                            std::string &error_out) {
  if (rows <= 0 || cols <= 0) {
    error_out = "invalid canvas mask dimensions";
    return false;
  }
  if (mask.size() != static_cast<size_t>(rows * cols)) {
    error_out = "canvas mask size mismatch while writing";
    return false;
  }
  Matrix2Df mask_img(rows, cols);
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      mask_img(y, x) =
          (mask[static_cast<size_t>(y * cols + x)] != 0) ? 1.0f : 0.0f;
    }
  }
  try {
    fs::create_directories(mask_path.parent_path());
    tile_compile::io::write_fits_float(mask_path, mask_img, header);
    return true;
  } catch (const std::exception &e) {
    error_out = std::string("cannot write canvas mask: ") + e.what();
    return false;
  }
}
} // namespace

/// @brief Runs pipeline command.
/// @details Part of the production runner pipeline that coordinates scan, registration, metrics, reconstruction, stacking, astrometry, BGE, and PCC phases; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
int run_pipeline_command(const std::string &config_path, const std::string &input_dir,
                const std::string &runs_dir, const std::string &project_root,
                const std::string &run_id_override,
                bool dry_run, int max_frames, int max_tiles,
                bool config_from_stdin) {
  using namespace tile_compile;

  fs::path cfg_path(config_path);
  fs::path in_dir(input_dir);
  fs::path runs(runs_dir);

  const bool use_stdin_config = config_from_stdin || (config_path == "-");
  fs::path proj_root;

  if (!fs::exists(in_dir)) {
    std::cerr << "Error: Input directory not found: " << input_dir << std::endl;
    return 1;
  }

  config::Config cfg;
  std::string cfg_text;
  if (use_stdin_config) {
    std::ostringstream ss;
    ss << std::cin.rdbuf();
    cfg_text = ss.str();
    if (cfg_text.empty()) {
      std::cerr << "Error: --stdin provided but no config YAML received"
                << std::endl;
      return 1;
    }
    cfg = config::Config::from_yaml_text(cfg_text);
    cfg.validate();
    proj_root =
        project_root.empty() ? fs::current_path() : fs::path(project_root);
  } else {
    if (!fs::exists(cfg_path)) {
      std::cerr << "Error: Config file not found: " << config_path << std::endl;
      return 1;
    }
    cfg = config::Config::load(cfg_path);
    cfg.validate();
    proj_root = project_root.empty() ? core::resolve_project_root(cfg_path)
                                     : fs::path(project_root);
  }


  cfg.method = config::getEffectiveMethod(cfg);
  cfg.aqmh.enabled = cfg.method == "aqmh";

  auto frames = core::discover_frames(in_dir, "*");
  frames.erase(
      std::remove_if(frames.begin(), frames.end(),
                     [](const fs::path &p) { return !io::is_fits_image_path(p); }),
      frames.end());
  std::sort(frames.begin(), frames.end());
  if (max_frames > 0 && frames.size() > static_cast<size_t>(max_frames)) {
    frames.resize(static_cast<size_t>(max_frames));
  }
  if (frames.empty()) {
    std::cerr << "Error: No FITS frames found in " << input_dir << std::endl;
    return 1;
  }

  std::string run_id = run_id_override.empty() ? core::get_run_id() : run_id_override;
  fs::path run_dir;
  try {
    run_dir = fs::absolute(runs / run_id);
  } catch (...) {
    // fs::absolute() can fail on Windows with UNC paths (\\server\share).
    // Fall back to lexical normalization — the path is still usable.
    run_dir = (runs / run_id).lexically_normal();
  }
  try {
    fs::create_directories(run_dir / "logs");
    fs::create_directories(run_dir / "outputs");
    fs::create_directories(run_dir / "artifacts");
  } catch (const std::exception& e) {
    std::cerr << "Error: cannot create run directories in " << run_dir
              << ": " << e.what() << std::endl;
    return 1;
  }

  if (use_stdin_config) {
    std::ofstream out(run_dir / "config.yaml", std::ios::out);
    out << cfg_text;
  } else {
    core::copy_config(cfg_path, run_dir / "config.yaml");
  }

  std::ofstream event_log_file(run_dir / "logs" / "run_events.jsonl",
                               std::ios::out | std::ios::trunc);
  if (!event_log_file.is_open()) {
    std::cerr << "Error: cannot open events log file: "
              << (run_dir / "logs" / "run_events.jsonl") << std::endl;
    return 1;
  }
  TeeBuf tee_buf(std::cout.rdbuf(), event_log_file.rdbuf());
  std::ostream log_file(&tee_buf);

  core::EventEmitter emitter;
  emitter.run_start(run_id,
                    {{"config_path", config_path},
                     {"input_dir", input_dir},
                     {"run_dir", run_dir.string()},
                     {"frames_discovered", frames.size()},
                     {"dry_run", dry_run}},
                    log_file);
  core::AccelerationContext acceleration(
      cfg.runtime_limits.acceleration_backend);
  core::write_text(run_dir / "artifacts" / "acceleration_context.json",
                   acceleration.to_json().dump(2));
  const auto run_started_at = std::chrono::steady_clock::now();
  auto abort_if_runtime_limit_exceeded =
      [&](const std::string &checkpoint) -> bool {
    const double elapsed_hours =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      run_started_at)
            .count() /
        3600.0;
    if (elapsed_hours <= cfg.runtime_limits.hard_abort_hours) {
      return false;
    }
    emitter.warning(
        run_id,
        "Runtime limit exceeded at " + checkpoint + " (" +
            std::to_string(elapsed_hours) + " h > " +
            std::to_string(cfg.runtime_limits.hard_abort_hours) + " h)",
        log_file);
    core::emit_event("runtime_limit_exceeded", run_id,
                     {{"checkpoint", checkpoint},
                      {"elapsed_hours", elapsed_hours},
                      {"hard_abort_hours",
                       cfg.runtime_limits.hard_abort_hours}},
                     log_file);
    {
      std::ostringstream oss;
      oss << "runtime limit exceeded at " << checkpoint << " ("
          << elapsed_hours << " h > "
          << cfg.runtime_limits.hard_abort_hours << " h)";
      emitter.run_end(run_id, false, "runtime_limit_exceeded", log_file,
                      {{"message", oss.str()}});
      std::cerr << "Error: " << oss.str() << std::endl;
    }
    return true;
  };

  std::cout << "Run ID: " << run_id << std::endl;
  std::cout << "Frames: " << frames.size() << std::endl;
  std::cout << "Output: " << run_dir.string() << std::endl;
  if (max_tiles > 0) {
    std::cout << "Max tiles (Phase 5/6): " << max_tiles << std::endl;
  }

  if (dry_run) {
    emitter.phase_start(run_id, Phase::SCAN_INPUT, "SCAN_INPUT", log_file);
    emitter.phase_end(run_id, Phase::SCAN_INPUT, "skipped",
                      {{"reason", "dry_run"}, {"input_dir", input_dir}},
                      log_file);

    std::cout << "Dry run - no processing" << std::endl;
    emitter.run_end(run_id, true, "ok", log_file);
    return 0;
  }

  // Phase 0: SCAN_INPUT (// Methodik v3)
  emitter.phase_start(run_id, Phase::SCAN_INPUT, "SCAN_INPUT", log_file);

  int width = 0;
  int height = 0;
  int naxis = 0;
  ColorMode detected_mode = ColorMode::MONO;
  BayerPattern detected_bayer = BayerPattern::UNKNOWN;
  Matrix2Df first_frame;
  io::FitsHeader first_header;

  try {
    std::tie(width, height, naxis) = io::get_fits_dimensions(frames.front());
    auto first = io::read_fits_float(frames.front());
    first_frame = std::move(first.first);
    first_header = std::move(first.second);

    detected_mode = io::detect_color_mode(first_header, naxis);
    detected_bayer = io::detect_bayer_pattern(first_header);
  } catch (const std::exception &e) {
    emitter.phase_end(run_id, Phase::SCAN_INPUT, "error",
                      {{"error", e.what()}, {"input_dir", input_dir}},
                      log_file);
    emitter.run_end(run_id, false, "error", log_file,
                    {{"message", std::string("Error during SCAN_INPUT: ") + e.what()}});
    std::cerr << "Error during SCAN_INPUT: " << e.what() << std::endl;
    return 1;
  }

  std::string detected_mode_str = color_mode_to_string(detected_mode);
  std::string detected_bayer_str = bayer_pattern_to_string(detected_bayer);

  const bool header_has_color_hint =
      (naxis >= 3) || (detected_bayer != BayerPattern::UNKNOWN) ||
      first_header.get_string("COLORTYP").has_value();
  const ColorMode cfg_color_mode =
      cfg.data.color_mode.empty() ? ColorMode::MONO
                                  : (cfg.data.color_mode == "RGB"
                                         ? ColorMode::RGB
                                         : (cfg.data.color_mode == "OSC"
                                                ? ColorMode::OSC
                                                : ColorMode::MONO));
  const bool cfg_color_mode_valid =
      cfg.data.color_mode == "MONO" || cfg.data.color_mode == "OSC" ||
      cfg.data.color_mode == "RGB";
  const BayerPattern cfg_bayer = cfg.data.bayer_pattern.empty()
                                     ? BayerPattern::UNKNOWN
                                     : string_to_bayer_pattern(
                                           cfg.data.bayer_pattern);
  const bool cfg_looks_like_default_osc =
      (cfg.data.color_mode == "OSC" &&
       (cfg.data.bayer_pattern.empty() || cfg.data.bayer_pattern == "GBRG"));

  if (!header_has_color_hint && cfg_color_mode_valid) {
    // For hint-less FITS, avoid forcing OSC from implicit defaults.
    // This keeps MONO/SW datasets processable without BAYERPAT.
    if (!cfg_looks_like_default_osc) {
      detected_mode = cfg_color_mode;
      detected_mode_str = color_mode_to_string(detected_mode);
      emitter.warning(run_id,
                      "FITS header has no clear color hint; using "
                      "config.data.color_mode='" +
                          cfg.data.color_mode + "' as fallback",
                      log_file);
    } else {
      detected_mode = ColorMode::MONO;
      detected_mode_str = color_mode_to_string(detected_mode);
      emitter.warning(run_id,
                      "FITS header has no clear color hint; default OSC/BAYER "
                      "config would be ambiguous, using MONO fallback",
                      log_file);
    }
  }
  if (detected_mode == ColorMode::OSC &&
      detected_bayer == BayerPattern::UNKNOWN &&
      cfg_bayer != BayerPattern::UNKNOWN) {
    detected_bayer = cfg_bayer;
    detected_bayer_str = bayer_pattern_to_string(detected_bayer);
    emitter.warning(run_id,
                    "FITS header has no valid BAYER pattern; using "
                    "config.data.bayer_pattern='" +
                        cfg.data.bayer_pattern + "' as fallback",
                    log_file);
  } else if (detected_mode != ColorMode::OSC) {
    detected_bayer = BayerPattern::UNKNOWN;
    detected_bayer_str = bayer_pattern_to_string(detected_bayer);
  }
  if (detected_mode == ColorMode::OSC &&
      detected_bayer == BayerPattern::UNKNOWN) {
    const std::string msg =
        "OSC input has no Bayer metadata (BAYERPAT/COLORTYP) and "
        "data.bayer_pattern is auto; refusing to guess a CFA pattern";
    emitter.phase_end(run_id, Phase::SCAN_INPUT, "error",
                      {{"error", msg},
                       {"input_dir", input_dir},
                       {"bayer_pattern", "UNKNOWN"}},
                      log_file);
    emitter.run_end(run_id, false, "error", log_file,
                    {{"message", std::string("Error during SCAN_INPUT: ") + msg}});
    std::cerr << "Error during SCAN_INPUT: " << msg << std::endl;
    return 1;
  }
  if (width <= 0 && cfg.data.image_width > 0) {
    width = cfg.data.image_width;
    emitter.warning(run_id,
                    "FITS header missing image_width; using "
                    "config.data.image_width fallback",
                    log_file);
  }
  if (height <= 0 && cfg.data.image_height > 0) {
    height = cfg.data.image_height;
    emitter.warning(run_id,
                    "FITS header missing image_height; using "
                    "config.data.image_height fallback",
                    log_file);
  }

  for (size_t idx = 1; idx < frames.size(); ++idx) {
    try {
      auto [frame_width, frame_height, frame_naxis] =
          io::get_fits_dimensions(frames[idx]);
      (void)frame_naxis;
      if (frame_width <= 0 && cfg.data.image_width > 0) {
        frame_width = cfg.data.image_width;
      }
      if (frame_height <= 0 && cfg.data.image_height > 0) {
        frame_height = cfg.data.image_height;
      }
      if (frame_width != width || frame_height != height) {
        const std::string msg =
            "Inconsistent image size: expected " + std::to_string(width) + "x" +
            std::to_string(height) + ", got " + std::to_string(frame_width) +
            "x" + std::to_string(frame_height) + " in " +
            frames[idx].filename().string();
        emitter.phase_end(run_id, Phase::SCAN_INPUT, "error",
                          {{"error", msg},
                           {"input_dir", input_dir},
                           {"expected_width", width},
                           {"expected_height", height},
                           {"frame", frames[idx].filename().string()},
                           {"frame_width", frame_width},
                           {"frame_height", frame_height}},
                          log_file);
        emitter.run_end(run_id, false, "error", log_file,
                        {{"message", std::string("Error during SCAN_INPUT: ") + msg}});
        std::cerr << "Error during SCAN_INPUT: " << msg << std::endl;
        return 1;
      }
    } catch (const std::exception &e) {
      emitter.phase_end(run_id, Phase::SCAN_INPUT, "error",
                        {{"error", e.what()},
                         {"input_dir", input_dir},
                         {"frame", frames[idx].filename().string()}},
                        log_file);
      emitter.run_end(run_id, false, "error", log_file,
                      {{"message", std::string("Error during SCAN_INPUT: ") + e.what()}});
      std::cerr << "Error during SCAN_INPUT: " << e.what() << std::endl;
      return 1;
    }
  }

  if (header_has_color_hint && !cfg.data.color_mode.empty() &&
      cfg.data.color_mode != detected_mode_str) {
    emitter.warning(run_id,
                    "Detected color mode '" + detected_mode_str +
                        "' differs from config.data.color_mode '" +
                        cfg.data.color_mode + "'",
                    log_file);
  }
  if (!cfg.data.bayer_pattern.empty() && detected_mode == ColorMode::OSC &&
      cfg.data.bayer_pattern != detected_bayer_str &&
      detected_bayer != BayerPattern::UNKNOWN) {
    emitter.warning(run_id,
                    "Detected bayer pattern '" + detected_bayer_str +
                        "' differs from config.data.bayer_pattern '" +
                        cfg.data.bayer_pattern + "'",
                    log_file);
  }

  core::json linearity_info;
  std::vector<size_t> rejected_indices;
  if (cfg.linearity.enabled || cfg.data.linear_required) {
    auto indices = core::sample_indices(frames.size(), cfg.linearity.max_frames);
    int failed = 0;
    float score_sum = 0.0f;
    std::vector<std::string> failed_names;
    for (size_t idx : indices) {
      Matrix2Df frame_img;
      if (idx == 0) {
        frame_img = first_frame;
      } else {
        frame_img = io::read_fits_pixels_float(frames[idx]);
      }
      metrics::LinearityFrameResult res =
          metrics::validate_linearity_frame(frame_img, cfg.linearity.strictness);
      score_sum += res.is_linear ? 1.0f : 0.0f;
      if (!res.is_linear) {
        failed++;
        rejected_indices.push_back(idx);
        if (failed_names.size() < 5) {
          failed_names.push_back(frames[idx].filename().string());
        }
      }
    }

    float overall_linearity =
        indices.empty() ? 0.0f
                        : (score_sum / static_cast<float>(indices.size()));
    linearity_info["enabled"] = true;
    linearity_info["sampled_frames"] = static_cast<int>(indices.size());
    linearity_info["overall_linearity"] = overall_linearity;
    linearity_info["min_overall_linearity"] =
        cfg.linearity.min_overall_linearity;
    linearity_info["failed_frames"] = failed;
    if (!failed_names.empty()) {
      linearity_info["failed_frame_names"] = failed_names;
    }

    if (failed > 0) {
      emitter.warning(
          run_id,
          "Linearity check: " + std::to_string(failed) +
              " sampled frames flagged non-linear (overall_linearity=" +
              std::to_string(overall_linearity) + ")",
          log_file);
    }
  } else {
    emitter.warning(
        run_id,
        "Linearity check disabled by config; continuing without enforcement.",
        log_file);
    linearity_info["enabled"] = false;
  }

  if (!rejected_indices.empty()) {
    std::sort(rejected_indices.begin(), rejected_indices.end());
    rejected_indices.erase(
        std::unique(rejected_indices.begin(), rejected_indices.end()),
        rejected_indices.end());
    linearity_info["flagged_indices"] = core::json::array();
    linearity_info["flagged_names"] = core::json::array();
    for (size_t idx : rejected_indices) {
      linearity_info["flagged_indices"].push_back(static_cast<int>(idx));
      if (idx < frames.size()) {
        linearity_info["flagged_names"].push_back(
            frames[idx].filename().string());
      }
    }

    emitter.warning(
        run_id,
        "Linearity: " + std::to_string(rejected_indices.size()) +
            " frames flagged non-linear (kept, warn-only mode)",
        log_file);
    linearity_info["action"] = "warn_only";
    linearity_info["frames_remaining"] = static_cast<int>(frames.size());
  }

  CalibrationRunResult calibration_result;
  if (cfg.calibration.use_bias || cfg.calibration.use_dark ||
      cfg.calibration.use_flat) {
    std::string calibration_error;
    if (!run_scan_input_calibration(cfg, proj_root, frames, run_dir, run_id,
                                    emitter, log_file, detected_mode, height,
                                    width, calibration_result,
                                    calibration_error)) {
      emitter.phase_end(run_id, Phase::SCAN_INPUT, "error",
                        {{"error", calibration_error},
                         {"input_dir", input_dir},
                         {"substep", "calibration"}},
                        log_file);
      emitter.run_end(run_id, false, "error", log_file,
                      {{"message", std::string("Error during SCAN_INPUT: ") + calibration_error}});
      std::cerr << "Error during SCAN_INPUT: " << calibration_error
                << std::endl;
      return 1;
    }
    if (calibration_result.applied) {
      frames = calibration_result.calibrated_frames;
      emitter.warning(
          run_id,
          "Calibration applied during SCAN_INPUT; downstream phases will use "
          "outputs/calibrated/cal_*.fit",
          log_file);
    }
  } else {
    calibration_result.requested = false;
    calibration_result.applied = false;
    calibration_result.calibrated_frames = frames;
    calibration_result.artifact = {{"requested", false}, {"applied", false}};
  }

  core::json scan_extra = {
      {"input_dir", input_dir},
      {"frames_scanned", frames.size()},
      {"image_width", width},
      {"image_height", height},
      {"color_mode", detected_mode_str},
      {"bayer_pattern", detected_bayer_str},
      {"calibration", calibration_result.artifact},
  };

  {
    const uint64_t scan_dir_bytes = estimate_total_file_bytes(frames);
    const uint64_t required_min_bytes =
        (scan_dir_bytes > std::numeric_limits<uint64_t>::max() / 4)
            ? std::numeric_limits<uint64_t>::max()
            : (scan_dir_bytes * 4ULL);

    std::error_code ec_space;
    const auto space_info = fs::space(runs, ec_space);
    if (!ec_space) {
      const uint64_t available_bytes =
          static_cast<uint64_t>(space_info.available);
      scan_extra["runs_device_available_bytes"] = available_bytes;
      scan_extra["scan_input_total_bytes"] = scan_dir_bytes;
      scan_extra["required_min_bytes_scandir_x4"] = required_min_bytes;

      if (available_bytes < required_min_bytes) {
        const std::string msg =
            "Insufficient disk space on runs device: available=" +
            format_bytes(available_bytes) +
            ", required_min(scandir*4)=" + format_bytes(required_min_bytes);
        emitter.phase_end(run_id, Phase::SCAN_INPUT, "error",
                          {{"error", msg},
                           {"runs_device_available_bytes", available_bytes},
                           {"scan_input_total_bytes", scan_dir_bytes},
                           {"required_min_bytes_scandir_x4", required_min_bytes},
                           {"runs_dir", runs.string()}},
                          log_file);
        emitter.run_end(run_id, false, "insufficient_disk_space", log_file,
                        {{"message", msg}});
        std::cerr << "Error during SCAN_INPUT: " << msg << std::endl;
        return 1;
      }
    } else {
      emitter.warning(run_id,
                      "Disk-space precheck skipped: cannot query free space for " +
                          runs.string() + " (" + ec_space.message() + ")",
                      log_file);
    }
  }

  if (!linearity_info.is_null()) {
    scan_extra["linearity"] = linearity_info;
  }

  emitter.phase_end(run_id, Phase::SCAN_INPUT, "ok", scan_extra, log_file);
  if (abort_if_runtime_limit_exceeded("SCAN_INPUT")) {
    return 1;
  }

  runner::PhaseRegistrationContext phase_registration_ctx;

  runner::PhaseMetricsContext phase_metrics_ctx;
  if (!runner::run_phase_channel_split_normalization_global_metrics(
          run_id, cfg, frames, run_dir, detected_mode, detected_bayer_str,
          emitter, log_file, phase_metrics_ctx)) {
    return 1;
  }
  if (abort_if_runtime_limit_exceeded("CHANNEL_SPLIT_NORMALIZATION_GLOBAL_METRICS")) {
    return 1;
  }

  auto &norm_scales = phase_metrics_ctx.norm_scales;
  auto &frame_metrics = phase_metrics_ctx.frame_metrics;
  VectorXf global_weights = phase_metrics_ctx.global_weights;
  VectorXf aqmh_global_weights;
  const auto frame_cache = phase_metrics_ctx.frame_cache;
  const float output_pedestal = phase_metrics_ctx.output_pedestal;
  const float output_scale_mono = phase_metrics_ctx.output_scale_mono;
  const float output_scale_r = phase_metrics_ctx.output_scale_r;
  const float output_scale_g = phase_metrics_ctx.output_scale_g;
  const float output_scale_b = phase_metrics_ctx.output_scale_b;
  const float output_bg_mono = phase_metrics_ctx.output_bg_mono;
  const float output_bg_r = phase_metrics_ctx.output_bg_r;
  const float output_bg_g = phase_metrics_ctx.output_bg_g;
  const float output_bg_b = phase_metrics_ctx.output_bg_b;

  // Seeing is shared by registration validation and Classic tile sizing.
  float seeing_fwhm_med = 3.0f;
  bool have_seeing_fwhm = false;
  {
    // Robust FWHM probing: measure on up to 5 evenly-spaced frames,
    // take median of all successful measurements (not just the first).
    const size_t n_probe = std::min<size_t>(5, frames.size());
    std::vector<float> fwhm_probes;
    fwhm_probes.reserve(n_probe);
    for (size_t pi = 0; pi < n_probe; ++pi) {
      size_t fi =
          (n_probe <= 1) ? 0 : (pi * (frames.size() - 1)) / (n_probe - 1);
      const int roi_w = std::min(width, 1024);
      const int roi_h = std::min(height, 1024);
      const int roi_x0 = std::max(0, (width - roi_w) / 2);
      const int roi_y0 = std::max(0, (height - roi_h) / 2);

      Matrix2Df img =
          io::read_fits_region_float(frames[fi], roi_x0, roi_y0, roi_w, roi_h);
      image::apply_normalization_inplace(img, norm_scales[fi], detected_mode,
                                  detected_bayer_str, roi_x0, roi_y0);
      float fwhm = metrics::measure_fwhm_from_image(img);
      if (fwhm > 0.0f && std::isfinite(fwhm))
        fwhm_probes.push_back(fwhm);
    }
    if (!fwhm_probes.empty()) {
      seeing_fwhm_med = core::median_of(fwhm_probes);
      have_seeing_fwhm = true;
    }
  }

  int seeing_tile_size = 0;
  float overlap_fraction = cfg.tile.overlap_fraction;
  float overlap_clipped = cfg.tile.overlap_fraction;
  int overlap_px = 0;
  int stride_px = 0;
  {
    float F = seeing_fwhm_med;
    if (!(F > 0.0f) || !std::isfinite(F)) {
      F = 3.0f;
    }

    const int tmin = std::max(16, cfg.tile.min_size);
    const int D = std::max(1, cfg.tile.max_divisor);
    int tmax = std::max(1, std::min(width, height) / D);
    if (tmax < tmin) tmax = tmin;

    const float t0 = static_cast<float>(cfg.tile.size_factor) * F;
    const float tc = std::min(std::max(t0, static_cast<float>(tmin)),
                              static_cast<float>(tmax));
    seeing_tile_size = static_cast<int>(std::floor(tc));
    if (seeing_tile_size < tmin) seeing_tile_size = tmin;

    overlap_clipped = std::min(0.5f, std::max(0.0f, overlap_fraction));
    overlap_fraction = overlap_clipped;
    overlap_px = static_cast<int>(
        std::floor(overlap_clipped * static_cast<float>(seeing_tile_size)));
    stride_px = seeing_tile_size - overlap_px;
    if (stride_px <= 0) {
      overlap_clipped = std::min(0.5f, std::max(0.0f, 0.25f));
      overlap_fraction = overlap_clipped;
      overlap_px = static_cast<int>(
          std::floor(overlap_clipped * static_cast<float>(seeing_tile_size)));
      stride_px = seeing_tile_size - overlap_px;
    }
  }

  std::vector<Tile> tiles;
  int uniform_tile_size = seeing_tile_size;
  if (!cfg.aqmh.enabled) {
    // Classic-only Phase 4: TILE_GRID (with adaptive optimization).
    emitter.phase_start(run_id, Phase::TILE_GRID, "TILE_GRID", log_file);
    tiles = tile_compile::pipeline::build_initial_tile_grid(
        width, height, uniform_tile_size, overlap_fraction);

  // Overlap-fraction performance warning (Anforderung 9.1, 9.3).
  if (overlap_fraction >= 0.5f) {
    // Estimate tile count at overlap=0.3 for comparison.
    const auto tiles_at_30 = tile_compile::pipeline::build_initial_tile_grid(
        width, height, uniform_tile_size, 0.3f);
    const float ratio = (tiles_at_30.empty())
        ? 1.0f
        : static_cast<float>(tiles.size()) / static_cast<float>(tiles_at_30.size());
    std::ostringstream msg;
    msg << "overlap_fraction=" << overlap_fraction
        << " produces " << tiles.size() << " tiles"
        << " vs. ~" << tiles_at_30.size() << " at overlap=0.3"
        << " (factor " << ratio << "x more tiles, proportionally longer TILE_RECONSTRUCTION)";
    emitter.warning(run_id, msg.str(), log_file);
  }

  {
    core::json artifact;
    artifact["image_width"] = width;
    artifact["image_height"] = height;
    artifact["num_tiles"] = static_cast<int>(tiles.size());
    artifact["overlap_fraction"] = overlap_fraction;
    artifact["seeing_fwhm_median"] = seeing_fwhm_med;
    artifact["seeing_tile_size"] = seeing_tile_size;
    artifact["seeing_overlap_px"] = overlap_px;
    artifact["stride_px"] = stride_px;
    artifact["tile_config"] = {
        {"size_factor", cfg.tile.size_factor},
        {"min_size", cfg.tile.min_size},
        {"max_divisor", cfg.tile.max_divisor},
        {"overlap_fraction", overlap_fraction},
        {"overlap_clipped", overlap_clipped},
    };
    artifact["uniform_tile_size"] = uniform_tile_size;

    // Estimated TILE_RECONSTRUCTION time (Anforderung 9.2).
    // Calibration constant k ≈ 0.012 s/tile/frame/worker (empirical).
    constexpr float k_tile_frame_worker = 0.012f;
    const int pw = std::max(1, cfg.runtime_limits.parallel_workers);
    artifact["estimated_reconstruction_time_s"] =
        static_cast<float>(tiles.size()) *
        static_cast<float>(frames.size()) /
        static_cast<float>(pw) *
        k_tile_frame_worker;
    artifact["coverage_filtered_tiles"] = 0; // updated after PREWARP if canvas expands

    artifact["tiles"] = core::json::array();
    for (const auto &t : tiles) {
      artifact["tiles"].push_back({
          {"x", t.x},
          {"y", t.y},
          {"width", t.width},
          {"height", t.height},
      });
    }

    core::write_text(run_dir / "artifacts" / "tile_grid.json",
                     artifact.dump(2));
  }

    emitter.phase_end(run_id, Phase::TILE_GRID, "ok",
                      {
                          {"num_tiles", static_cast<int>(tiles.size())},
                          {"gradient_field", false},
                      },
                      log_file);
    if (abort_if_runtime_limit_exceeded("TILE_GRID")) {
      return 1;
    }
  }

  // Helpers for Phase 5/6
  auto load_frame_normalized = [&](size_t frame_index) -> Matrix2Df {
    if (frame_cache && frame_cache->has_normalized(frame_index)) {
      return frame_cache->load_normalized(frame_index);
    }
    Matrix2Df img = io::read_fits_pixels_float(frames[frame_index]);
    image::apply_normalization_inplace(img, norm_scales[frame_index],
                                       detected_mode, detected_bayer_str, 0,
                                       0);
    if (frame_cache && img.size() > 0) {
      frame_cache->store_normalized(frame_index, img);
    }
    return img;
  };

  // extract_tile is now image::extract_tile (canonical module function)

  std::vector<std::vector<TileMetrics>> local_metrics;
  std::vector<std::vector<float>> local_weights;
  std::vector<TileMetrics> bge_tile_metrics_cache;
  TileGrid bge_tile_grid_cache;
  std::vector<float> tile_fwhm_median;
  std::vector<int> tile_valid_counts;
  std::vector<uint8_t> tile_fallback_used;
  std::vector<float> tile_warp_variances;
  std::vector<float> tile_mean_correlations;
  std::vector<float> tile_post_contrast;
  std::vector<float> tile_post_background;
  std::vector<float> tile_norm_bg_r;
  std::vector<float> tile_norm_bg_g;
  std::vector<float> tile_norm_bg_b;
  std::vector<float> tile_norm_scale;
  std::vector<float> tile_post_snr;
  std::vector<float> tile_mean_dx;
  std::vector<float> tile_mean_dy;
  std::vector<float> tile_quality_median;
  std::vector<uint8_t> tile_is_star;
  std::unique_ptr<tile_compile::metrics::QualityMapCache> aqmh_cache;
  std::vector<std::atomic<int>> frame_valid_tile_counts(frames.size());
  Matrix2Df recon;
  Matrix2Df recon_R;
  Matrix2Df recon_G;
  Matrix2Df recon_B;
  std::vector<uint8_t> df_valid_mask_R;
  std::vector<uint8_t> df_valid_mask_G;
  std::vector<uint8_t> df_valid_mask_B;
  Matrix2Df weight_sum;
  runner::BackgroundModelGrid aqmh_background_map_canvas_grid;

  Matrix2Df first_img;
  io::FitsHeader first_hdr;
  {
    first_img = load_frame_normalized(0);
    first_hdr = first_header;
  }

  if (!runner::run_phase_registration_prewarp(
          run_id, cfg, frames, run_dir, height, width, detected_mode,
          detected_bayer_str, frame_cache, norm_scales, frame_metrics, global_weights,
          first_header, acceleration, emitter, log_file,
          phase_registration_ctx, phase_metrics_ctx.rgb_frame_cache,
          phase_metrics_ctx.background_grid_store)) {
    return 1;
  }
  if (abort_if_runtime_limit_exceeded("REGISTRATION_PREWARP")) {
    return 1;
  }

  // Apply weight penalty to frames whose warp was predicted by the
  // field-rotation model rather than directly measured.  These frames have
  // cc≈0.0001 (minimum clamp), yet their image-quality metrics (FWHM, noise…)
  // may be good, which can give them disproportionately high global_weights.
  // Scaling down their weight limits their contribution to the stack without
  // excluding them entirely (the predicted geometry is usually correct).
  {
    constexpr float kModelPredictedWeightFactor = 0.05f;
    const auto &mp_mask = phase_registration_ctx.model_predicted_mask;
    int n_penalized = 0;
    for (Eigen::Index fi = 0; fi < global_weights.size(); ++fi) {
      if (static_cast<size_t>(fi) < mp_mask.size() && mp_mask[static_cast<size_t>(fi)]) {
        global_weights[fi] *= kModelPredictedWeightFactor;
        ++n_penalized;
      }
    }
    if (n_penalized > 0) {
      std::cout << "[PIPELINE] Applied model-predicted weight penalty ("
                << kModelPredictedWeightFactor << "x) to " << n_penalized
                << " frame(s) with model-interpolated/blended registration."
                << std::endl;
    }
  }

  auto &prewarped_frames = phase_registration_ctx.prewarped_frames;
  prewarped_frames.set_preserve_files(
      !cfg.aqmh.reconstruction.delete_prewarped_cache_after_run);
  auto &prewarped_frames_rgb = phase_registration_ctx.prewarped_frames_rgb;
  if (prewarped_frames_rgb.size() > 0) {
    prewarped_frames_rgb.set_preserve_files(
        !cfg.aqmh.reconstruction.delete_prewarped_cache_after_run);
  }
  auto &frame_has_data = phase_registration_ctx.frame_has_data;
  const int n_usable_frames = phase_registration_ctx.n_usable_frames;
  int min_valid_frames = phase_registration_ctx.min_valid_frames;
  const int canvas_tile_offset_x = phase_registration_ctx.tile_offset_x;
  const int canvas_tile_offset_y = phase_registration_ctx.tile_offset_y;
  int debayer_tile_offset_x = canvas_tile_offset_x;
  int debayer_tile_offset_y = canvas_tile_offset_y;
  // Canvas dimensions may be larger than original frame due to field rotation.
  const int canvas_height = (phase_registration_ctx.canvas_height > 0)
      ? phase_registration_ctx.canvas_height : height;
  const int canvas_width  = (phase_registration_ctx.canvas_width  > 0)
      ? phase_registration_ctx.canvas_width  : width;

  if (!cfg.aqmh.enabled &&
      (canvas_width != width || canvas_height != height)) {
    tiles = tile_compile::pipeline::build_initial_tile_grid(
        canvas_width, canvas_height, uniform_tile_size, overlap_fraction);

    std::ostringstream msg;
    msg << "TILE_GRID updated for expanded canvas: " << width << "x" << height
        << " -> " << canvas_width << "x" << canvas_height
        << " (tiles=" << tiles.size() << ")";
    emitter.warning(run_id, msg.str(), log_file);

    core::json artifact;
    artifact["image_width"] = canvas_width;
    artifact["image_height"] = canvas_height;
    artifact["num_tiles"] = static_cast<int>(tiles.size());
    artifact["overlap_fraction"] = overlap_fraction;
    artifact["seeing_fwhm_median"] = seeing_fwhm_med;
    artifact["seeing_tile_size"] = seeing_tile_size;
    artifact["seeing_overlap_px"] = overlap_px;
    artifact["stride_px"] = stride_px;
    artifact["tile_config"] = {
        {"size_factor", cfg.tile.size_factor},
        {"min_size", cfg.tile.min_size},
        {"max_divisor", cfg.tile.max_divisor},
        {"overlap_fraction", overlap_fraction},
        {"overlap_clipped", overlap_clipped},
    };
    artifact["uniform_tile_size"] = uniform_tile_size;
    // Estimated TILE_RECONSTRUCTION time for expanded canvas.
    constexpr float k_tile_frame_worker_exp = 0.012f;
    const int pw_exp = std::max(1, cfg.runtime_limits.parallel_workers);
    artifact["estimated_reconstruction_time_s"] =
        static_cast<float>(tiles.size()) *
        static_cast<float>(frames.size()) /
        static_cast<float>(pw_exp) *
        k_tile_frame_worker_exp;
    artifact["coverage_filtered_tiles"] = 0; // canvas mask not yet available here
    artifact["tiles"] = core::json::array();
    for (const auto &t : tiles) {
      artifact["tiles"].push_back({
          {"x", t.x},
          {"y", t.y},
          {"width", t.width},
          {"height", t.height},
      });
    }
    core::write_text(run_dir / "artifacts" / "tile_grid.json", artifact.dump(2));
  }

  std::vector<Tile> tiles_phase56 = tiles;
  if (max_tiles > 0 && tiles_phase56.size() > static_cast<size_t>(max_tiles)) {
    tiles_phase56.resize(static_cast<size_t>(max_tiles));
  }

  emitter.phase_start(run_id, Phase::COMMON_OVERLAP, "COMMON_OVERLAP", log_file);

  const size_t canvas_px =
      static_cast<size_t>(std::max(0, canvas_height)) *
      static_cast<size_t>(std::max(0, canvas_width));
  const int required_common_frames = std::max(
      1, static_cast<int>(std::ceil(
             cfg.stacking.common_overlap_required_fraction *
             static_cast<float>(std::max(1, n_usable_frames)))));
  std::vector<uint16_t> overlap_coverage_count_fallback;
  std::vector<uint16_t> *overlap_coverage_count_ptr =
      &phase_registration_ctx.overlap_coverage_count;
  std::vector<uint8_t> reconstruction_valid_mask;
  std::vector<float> tile_common_overlap_ratio(tiles_phase56.size(), 0.0f);
  std::vector<uint8_t> tile_common_valid(tiles_phase56.size(), 0);
  std::vector<uint8_t> tile_reconstruction_valid(tiles_phase56.size(), 0);

  std::mutex common_overlap_progress_mutex;
  size_t loaded_frames = static_cast<size_t>(n_usable_frames);

  if (phase_registration_ctx.overlap_coverage_count.size() != canvas_px ||
      phase_registration_ctx.common_valid_mask.size() != canvas_px) {
    overlap_coverage_count_fallback.assign(canvas_px, 0);
    overlap_coverage_count_ptr = &overlap_coverage_count_fallback;
    loaded_frames = 0;
    for (size_t fi = 0; fi < frames.size(); ++fi) {
      if (!frame_has_data[fi]) {
        continue;
      }
      const float *p = prewarped_frames.frame_data(fi);
      if (p == nullptr) {
        continue;
      }
      ++loaded_frames;
      for (size_t i = 0; i < canvas_px; ++i) {
        if (std::isfinite(p[i]) &&
            overlap_coverage_count_fallback[i] <
                std::numeric_limits<uint16_t>::max()) {
          ++overlap_coverage_count_fallback[i];
        }
      }

      const size_t done = fi + 1;
      if (done % 5 == 0 || done == frames.size()) {
        std::lock_guard<std::mutex> lock(common_overlap_progress_mutex);
        emitter.phase_progress_counts(
            run_id, Phase::COMMON_OVERLAP, static_cast<int>(done),
            static_cast<int>(frames.size()),
            "common_overlap fallback coverage " + std::to_string(done) + "/" +
                std::to_string(frames.size()),
            "frames", log_file);
      }
    }

  }

  auto &overlap_coverage_count = *overlap_coverage_count_ptr;
  std::vector<uint8_t> common_valid_mask;
  std::vector<uint8_t> analysis_valid_mask;
  if (overlap_coverage_count.size() == canvas_px) {
    auto masks = runner::compute_overlap_masks(overlap_coverage_count,
                                               required_common_frames);
    common_valid_mask = std::move(masks.analysis_common);
    reconstruction_valid_mask = std::move(masks.reconstruction_support);
    analysis_valid_mask = std::move(masks.analysis_valid);
  } else {
    common_valid_mask.assign(canvas_px, 0u);
    reconstruction_valid_mask.assign(canvas_px, 0u);
    analysis_valid_mask.assign(canvas_px, 0u);
  }
  const bool aqmh_uses_reconstruction_canvas =
      cfg.aqmh.enabled && reconstruction_valid_mask.size() == canvas_px;
  const std::vector<uint8_t> &aqmh_canvas_valid_mask =
      aqmh_uses_reconstruction_canvas ? reconstruction_valid_mask
                                      : common_valid_mask;
  const std::vector<uint8_t> &output_valid_mask =
      cfg.aqmh.enabled ? reconstruction_valid_mask : common_valid_mask;

  {
    size_t common_pixels = 0;
    size_t reconstruction_pixels = 0;
    for (size_t i = 0; i < canvas_px; ++i) {
      if (common_valid_mask[i] != 0) {
        ++common_pixels;
      }
      if (i < reconstruction_valid_mask.size() &&
          reconstruction_valid_mask[i] != 0) {
        ++reconstruction_pixels;
      }
    }

	    for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
	      const Tile &t = tiles_phase56[ti];
	      const int x0 = std::max(0, t.x);
	      const int y0 = std::max(0, t.y);
	      const int x1 = std::min(canvas_width, t.x + t.width);
	      const int y1 = std::min(canvas_height, t.y + t.height);
	      const int tile_total = std::max(0, t.width) * std::max(0, t.height);
	      int tile_common = 0;
	      int tile_reconstruction = 0;
	      for (int y = y0; y < y1; ++y) {
	        const size_t row_off = static_cast<size_t>(y) *
	                               static_cast<size_t>(canvas_width);
	        for (int x = x0; x < x1; ++x) {
	          const size_t idx = row_off + static_cast<size_t>(x);
	          if (common_valid_mask[idx] != 0) {
	            ++tile_common;
	          }
	          if (idx < reconstruction_valid_mask.size() &&
	              reconstruction_valid_mask[idx] != 0) {
	            ++tile_reconstruction;
	          }
	        }
	      }
      const float ratio =
          (tile_total > 0)
              ? (static_cast<float>(tile_common) / static_cast<float>(tile_total))
              : 0.0f;
      tile_common_overlap_ratio[ti] = ratio;
      if (ratio + 1.0e-6f >= cfg.stacking.tile_common_valid_min_fraction) {
        tile_common_valid[ti] = 1;
      }
      if (tile_reconstruction > 0) {
        tile_reconstruction_valid[ti] = 1;
      }

      const size_t done = ti + 1;
      if (done % 50 == 0 || done == tiles_phase56.size()) {
        std::lock_guard<std::mutex> lock(common_overlap_progress_mutex);
        emitter.phase_progress_counts(
            run_id, Phase::COMMON_OVERLAP, static_cast<int>(done),
            static_cast<int>(tiles_phase56.size()),
            "common_overlap tile-gating " + std::to_string(done) + "/" +
                std::to_string(tiles_phase56.size()),
            "tiles", log_file);
      }
    }

    core::json overlap_artifact;
    overlap_artifact["canvas_width"] = canvas_width;
    overlap_artifact["canvas_height"] = canvas_height;
    overlap_artifact["usable_frames"] = n_usable_frames;
    overlap_artifact["loaded_frames"] = static_cast<int>(loaded_frames);
    overlap_artifact["required_common_frames"] = required_common_frames;
    overlap_artifact["common_overlap_required_fraction"] =
        cfg.stacking.common_overlap_required_fraction;
    overlap_artifact["tile_common_valid_min_fraction"] =
        cfg.stacking.tile_common_valid_min_fraction;
    overlap_artifact["common_pixels"] = static_cast<uint64_t>(common_pixels);
    overlap_artifact["reconstruction_pixels"] =
        static_cast<uint64_t>(reconstruction_pixels);
    overlap_artifact["aqmh_canvas_mask"] =
        aqmh_uses_reconstruction_canvas ? "reconstruction" : "common";
    overlap_artifact["aqmh_canvas_pixels"] =
        static_cast<uint64_t>(aqmh_uses_reconstruction_canvas
                                  ? reconstruction_pixels
                                  : common_pixels);
    overlap_artifact["analysis_mask"] =
        (run_dir / "outputs" / "common_overlap_mask.fits").string();
    overlap_artifact["output_mask"] =
        (run_dir / "outputs" / "canvas_mask.fits").string();
    overlap_artifact["output_mask_source"] =
        cfg.aqmh.enabled ? "reconstruction_support" : "common_overlap";
    overlap_artifact["common_fraction"] =
        (canvas_px > 0)
            ? (static_cast<double>(common_pixels) /
               static_cast<double>(canvas_px))
            : 0.0;
    overlap_artifact["reconstruction_fraction"] =
        (canvas_px > 0)
            ? (static_cast<double>(reconstruction_pixels) /
               static_cast<double>(canvas_px))
            : 0.0;
    overlap_artifact["tiles"] = core::json::array();
    for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
      const Tile &t = tiles_phase56[ti];
      overlap_artifact["tiles"].push_back({
          {"index", static_cast<int>(ti)},
          {"x", t.x},
          {"y", t.y},
          {"width", t.width},
          {"height", t.height},
          {"common_ratio", tile_common_overlap_ratio[ti]},
          {"common_valid", tile_common_valid[ti] != 0},
          {"reconstruction_valid", tile_reconstruction_valid[ti] != 0},
      });
    }
    core::write_text(run_dir / "artifacts" / "common_overlap.json",
                     overlap_artifact.dump(2));

    {
      const fs::path common_mask_path =
          run_dir / "outputs" / "common_overlap_mask.fits";
      const fs::path mask_path = run_dir / "outputs" / "canvas_mask.fits";
      std::string mask_write_error;
      if (!write_canvas_mask_fits(common_mask_path, analysis_valid_mask,
                                  canvas_height, canvas_width, first_hdr,
                                  mask_write_error) ||
          !write_canvas_mask_fits(mask_path, output_valid_mask, canvas_height,
                                  canvas_width, first_hdr, mask_write_error)) {
        emitter.phase_end(run_id, Phase::COMMON_OVERLAP, "error",
                          {{"reason", "canvas_mask_write_failed"},
                           {"error", mask_write_error},
                           {"canvas_mask", mask_path.string()}},
                          log_file);
        emitter.run_end(run_id, false, "error", log_file,
                        {{"message", mask_write_error}});
        std::cerr << "Error: " << mask_write_error << std::endl;
        return 1;
      }
      std::cout << "[COMMON_OVERLAP] Analysis mask saved: "
                << common_mask_path << " (" << canvas_width << "x"
                << canvas_height << ", valid=" << common_pixels << "/"
                << canvas_px << ")" << std::endl;
      std::cout << "[COMMON_OVERLAP] Output canvas mask saved: " << mask_path
                << " (valid="
                << (cfg.aqmh.enabled ? reconstruction_pixels : common_pixels)
                << "/" << canvas_px << ")" << std::endl;
      if (aqmh_uses_reconstruction_canvas) {
        std::cout << "[COMMON_OVERLAP] AQMH canvas uses reconstruction support "
                  << "(" << reconstruction_pixels << "/" << canvas_px
                  << " pixels); COMMON_OVERLAP remains analysis mask"
                  << std::endl;
      }
    }

    emitter.phase_end(
        run_id, Phase::COMMON_OVERLAP, "ok",
        {
            {"usable_frames", n_usable_frames},
	            {"loaded_frames", static_cast<int>(loaded_frames)},
	            {"required_common_frames", required_common_frames},
	            {"common_overlap_required_fraction",
	             cfg.stacking.common_overlap_required_fraction},
	            {"tile_common_valid_min_fraction",
	             cfg.stacking.tile_common_valid_min_fraction},
	            {"common_pixels", static_cast<uint64_t>(common_pixels)},
            {"reconstruction_pixels",
             static_cast<uint64_t>(reconstruction_pixels)},
            {"aqmh_canvas_mask",
             aqmh_uses_reconstruction_canvas ? "reconstruction" : "common"},
            {"common_fraction",
             (canvas_px > 0)
                 ? (static_cast<double>(common_pixels) /
                    static_cast<double>(canvas_px))
                 : 0.0},
            {"tile_common_valid",
             std::count_if(tile_common_valid.begin(), tile_common_valid.end(),
                           [&](uint8_t v) { return v != 0; })},
        },
        log_file);
  }
  if (abort_if_runtime_limit_exceeded("COMMON_OVERLAP")) {
    return 1;
  }

  const int kReducedModeMinFrames = cfg.assumptions.frames_min;
  const int reduced_threshold = cfg.assumptions.frames_reduced_threshold;
  const core::ModeGateDecision gate = core::evaluate_mode_gate(
      n_usable_frames, reduced_threshold,
      cfg.runtime_limits.allow_emergency_mode, kReducedModeMinFrames);
  const bool emergency_mode = gate.emergency_mode;
  if (gate.should_abort) {
    std::ostringstream oss;
    oss << "Insufficient usable frames after registration/warp: "
        << n_usable_frames << " (<" << kReducedModeMinFrames
        << "). Set runtime_limits.allow_emergency_mode=true to force "
           "emergency reduced mode.";
    emitter.run_end(run_id, false, "insufficient_frames", log_file,
                    {{"message", oss.str()},
                     {"usable_frames", n_usable_frames},
                     {"frames_min", kReducedModeMinFrames}});
    std::cerr << "Error: " << oss.str() << std::endl;
    return 1;
  }

  const bool reduced_mode = gate.reduced_mode;
  const bool skip_clustering_in_reduced =
      (reduced_mode && cfg.assumptions.reduced_mode_skip_clustering);
  constexpr float kEpsWeight = 1.0e-6f;
  constexpr float kEpsMedian = 1.0e-6f;
  constexpr float kEpsWeightSum = 1.0e-6f;

  bool run_validation_failed = false;
  const auto tile_analysis_started_at = std::chrono::steady_clock::now();
  double tile_analysis_runtime_seconds = 0.0;
  double stacking_runtime_seconds = 0.0;
  std::optional<reconstruction::AqmhValidationComparison>
      aqmh_control_validation;

  // Prefetch coordinator for AQMH Q-map I/O overlap
  std::unique_ptr<reconstruction::AqmhPrefetchCoordinator> aqmh_prefetch_coordinator;

  while (true) {
    const bool metrics_ok = cfg.aqmh.enabled
        ? runner::run_phase_aqmh_maps(
              run_id, cfg, frames, run_dir, frame_has_data, aqmh_canvas_valid_mask,
              common_valid_mask, canvas_width, canvas_height, prewarped_frames,
              norm_scales,
              detected_mode, detected_bayer_str, false, acceleration, emitter,
              log_file, aqmh_cache, aqmh_global_weights,
              aqmh_prefetch_coordinator,
              phase_metrics_ctx.frame_star_metrics)
        : runner::run_phase_local_metrics(
              run_id, cfg, frames, run_dir, frame_has_data, tiles_phase56,
              common_valid_mask, canvas_width, canvas_height,
              tile_common_valid, prewarped_frames, norm_scales, detected_mode,
              detected_bayer_str, false, acceleration, emitter, log_file,
              local_metrics, local_weights, tile_quality_median, tile_is_star,
              tile_fwhm_median, aqmh_cache, aqmh_global_weights,
              canvas_tile_offset_x, canvas_tile_offset_y);
    if (!metrics_ok) {
      return 1;
    }
    if (abort_if_runtime_limit_exceeded("LOCAL_METRICS")) {
      return 1;
    }

    // Phase 6: TILE_RECONSTRUCTION (Methodik v3)
    const Phase reconstruction_phase = cfg.aqmh.enabled
                                           ? Phase::AQMH_RECONSTRUCTION
                                           : Phase::TILE_RECONSTRUCTION;
    emitter.phase_start(run_id, reconstruction_phase,
                        phase_to_string(reconstruction_phase), log_file);
    const auto tile_reconstruction_started_at = std::chrono::steady_clock::now();

    const int passes_total = 1;
    // Helper: post-warp metrics (// Methodik v3 §6)
	    auto compute_post_warp_metrics =
	        [&](const Matrix2Df &warped) -> std::tuple<float, float, float> {
	      if (warped.size() <= 0)
	        return {0.0f, 0.0f, 0.0f};
	      Matrix2Df finite_only = warped;
	      cv::Mat valid_mask(warped.rows(), warped.cols(), CV_8U,
	                         cv::Scalar(0));
	      int valid_count = 0;
	      for (int y = 0; y < warped.rows(); ++y) {
	        uchar *mrow = valid_mask.ptr<uchar>(y);
	        for (int x = 0; x < warped.cols(); ++x) {
	          if (std::isfinite(warped(y, x))) {
	            mrow[x] = 255;
	            ++valid_count;
	          } else {
	            finite_only(y, x) = 0.0f;
	          }
	        }
	      }
	      if (valid_count <= 0) {
	        return {0.0f, 0.0f, 0.0f};
	      }
	      cv::Mat wcv(finite_only.rows(), finite_only.cols(), CV_32F,
	                  finite_only.data());
	      cv::Mat lap;
      cv::Laplacian(wcv, lap, CV_32F);
      cv::Scalar mean_sd, stddev_sd;
      cv::meanStdDev(lap, mean_sd, stddev_sd, valid_mask);
      float contrast = static_cast<float>(stddev_sd[0] * stddev_sd[0]);

      std::vector<float> px;
      px.reserve(static_cast<size_t>(valid_count));
      for (Eigen::Index k = 0; k < warped.size(); ++k) {
        const float v = warped.data()[k];
        if (std::isfinite(v))
          px.push_back(v);
      }
      if (px.empty())
        return {0.0f, 0.0f, 0.0f};
      float background = core::median_of(px);

      float snr = 0.0f;
      if (!px.empty()) {
        float mad = core::robust_sigma_mad(px);
        std::vector<float> sorted_px = px;
        std::sort(sorted_px.begin(), sorted_px.end());
        float p99 = core::percentile_from_sorted(sorted_px, 99.0f);
        snr = (p99 - background) / (mad + 1.0e-6f);
      }

      return {contrast, background, snr};
    };

    const bool osc_mode = (detected_mode == ColorMode::OSC);
    const auto tile_reconstruction_acceleration = acceleration.selection_for(
        core::AccelerationPhase::tile_reconstruction);
    const core::AccelerationOps tile_reconstruction_ops(
        acceleration, core::AccelerationPhase::tile_reconstruction);
    const auto tile_reconstruction_frame_batch = core::make_device_frame_batch(
        static_cast<size_t>(std::max(0, n_usable_frames)), canvas_height,
        canvas_width, 1);
    const auto tile_reconstruction_tile_batch = core::make_device_tile_batch(
        tiles_phase56, osc_mode ? 3 : 1);
    {
      std::ostringstream msg;
      msg << "TILE_RECONSTRUCTION acceleration "
          << core::acceleration_selection_summary(
                 tile_reconstruction_acceleration);
      if (!tile_reconstruction_acceleration.request_honored &&
          !tile_reconstruction_acceleration.fallback_reason.empty()) {
        emitter.warning(run_id, msg.str(), log_file);
      }
      std::cout << "[Phase 6] " << msg.str() << std::endl;
    }

    recon = Matrix2Df::Zero(canvas_height, canvas_width);
    weight_sum = Matrix2Df::Zero(canvas_height, canvas_width);
    if (osc_mode) {
      recon_R = Matrix2Df::Zero(canvas_height, canvas_width);
      recon_G = Matrix2Df::Zero(canvas_height, canvas_width);
      recon_B = Matrix2Df::Zero(canvas_height, canvas_width);
    }

    const int prev_cv_threads_recon = cv::getNumThreads();
    cv::setNumThreads(1);

    runner::SyntheticWeightingDecision synthetic_weighting_decision;
    synthetic_weighting_decision.requested_weighting = cfg.synthetic.weighting;
    synthetic_weighting_decision.effective_weighting =
        cfg.aqmh.enabled ? "global" : cfg.synthetic.weighting;
    const auto tile_window_cache = build_tile_window_cache(tiles_phase56);
    const float eps_ws = kEpsWeightSum;

    if (cfg.aqmh.enabled) {
      runner::AqmhReconstructionPhaseResult aqmh_recon_result;
      if (!runner::run_phase_aqmh_reconstruction(
              run_id, cfg, run_dir, frames, frame_has_data,
              aqmh_canvas_valid_mask, common_valid_mask, canvas_width,
              canvas_height, osc_mode,
              prewarped_frames, aqmh_cache, aqmh_global_weights,
              acceleration, emitter, log_file,
              tile_reconstruction_started_at, prev_cv_threads_recon,
              aqmh_recon_result, aqmh_prefetch_coordinator.get(),
              phase_registration_ctx.prewarped_frames_rgb.size() > 0
                  ? &phase_registration_ctx.prewarped_frames_rgb
                  : nullptr,
              phase_registration_ctx.prewarped_background_grid_store.get())) {
        return 1;
      }
      recon = aqmh_recon_result.output;
      weight_sum = aqmh_recon_result.weight_sum;
      aqmh_control_validation = aqmh_recon_result.control_validation;
      aqmh_background_map_canvas_grid =
          std::move(aqmh_recon_result.background_map_canvas_grid);
      try {
        io::write_fits_float(
            run_dir / "outputs" / "aqmh_reconstructed_raw.fit",
            aqmh_recon_result.raw_output, first_hdr);
      } catch (const std::exception &e) {
        core::emit_event(
            "artifact_write_failed", run_id,
            {{"phase_name", "AQMH_RECONSTRUCTION"},
             {"reason", "persist_raw_reconstruction_failed"},
             {"output",
              (run_dir / "outputs" / "aqmh_reconstructed_raw.fit").string()},
             {"error", e.what()}},
            log_file);
        emitter.run_end(run_id, false, "persist_raw_reconstruction_failed",
                        log_file,
                        {{"message", std::string("cannot persist immutable AQMH reconstruction: ") + e.what()}});
        std::cerr << "Error: cannot persist immutable AQMH reconstruction: "
                  << e.what() << std::endl;
        return 1;
      }
      if (aqmh_recon_result.debayer_first_used) {
        // Debayer-First-AQMH: populate RGB channels from per-channel
        // reconstruction results so downstream output writes RGB FITS.
        recon_R = std::move(aqmh_recon_result.df_output_R);
        recon_G = std::move(aqmh_recon_result.df_output_G);
        recon_B = std::move(aqmh_recon_result.df_output_B);
        df_valid_mask_R = std::move(aqmh_recon_result.df_valid_mask_R);
        df_valid_mask_G = std::move(aqmh_recon_result.df_valid_mask_G);
        df_valid_mask_B = std::move(aqmh_recon_result.df_valid_mask_B);
      } else if (aqmh_recon_result.osc_rgb_cleared) {
        recon_R.resize(0, 0);
        recon_G.resize(0, 0);
        recon_B.resize(0, 0);
      }
      weight_sum.resize(0, 0);
      first_img.resize(0, 0);
      if (abort_if_runtime_limit_exceeded("TILE_RECONSTRUCTION")) {
        return 1;
      }
      // Phase: AQMH_DIAGNOSTICS — block-level Q-map statistics and heatmaps
      if (!runner::run_phase_aqmh_diagnostics(
              run_id, cfg, run_dir, aqmh_recon_result.recon,
              aqmh_cache.get(), common_valid_mask, frame_has_data,
              canvas_width, canvas_height, emitter, log_file)) {
        return 1;
      }
    } else {

    // Parallel processing configuration
    int parallel_tiles = cfg.runtime_limits.parallel_workers;
    int cpu_cores = std::thread::hardware_concurrency();
    if (cpu_cores == 0)
      cpu_cores = 1;
    if (parallel_tiles > cpu_cores) {
      std::cout << "[WARNING] parallel_tiles (" << parallel_tiles
                << ") exceeds CPU cores (" << cpu_cores << "), capping to "
                << cpu_cores << std::endl;
      parallel_tiles = cpu_cores;
    }
    if (parallel_tiles < 1)
      parallel_tiles = 1;

    // OSC RGB stacking can be memory-heavy. Instead of reducing worker count,
    // compute a frame sub-batch size so that N workers × sub_batch × tile_px × 3ch
    // fits within the memory budget. This preserves full parallelism.
    size_t frame_sub_batch_size = static_cast<size_t>(n_usable_frames); // default: all frames
    if (osc_mode && !tiles_phase56.empty()) {
      size_t max_tile_px = 0;
      for (const auto &t : tiles_phase56) {
        size_t px = static_cast<size_t>(std::max(0, t.width)) *
                    static_cast<size_t>(std::max(0, t.height));
        if (px > max_tile_px)
          max_tile_px = px;
      }
      const auto batch_plan = tile_compile::runner::compute_memory_capped_frame_sub_batch(
          static_cast<size_t>(n_usable_frames), max_tile_px, 3, parallel_tiles,
          cfg.runtime_limits.memory_budget);
      parallel_tiles = batch_plan.effective_workers;
      frame_sub_batch_size = batch_plan.frame_sub_batch_size > 0
                                 ? batch_plan.frame_sub_batch_size
                                 : static_cast<size_t>(n_usable_frames);
      if (batch_plan.budget_too_small_for_requested_workers) {
        std::cout << "[Phase 6] OSC memory cap: budget too small, using 1 worker"
                  << std::endl;
      } else if (batch_plan.sub_batch_limited) {
        std::cout << "[Phase 6] OSC sub-batch: " << parallel_tiles
                  << " workers × " << frame_sub_batch_size
                  << " frames/batch (budget "
                  << (batch_plan.memory_budget_bytes / (1024 * 1024)) << " MB, "
                  << (batch_plan.bytes_per_frame_per_worker / (1024 * 1024))
                  << " MB/frame/worker)" << std::endl;
      }
    }

    std::cout << "[Phase 6] Using " << parallel_tiles
              << " parallel workers for " << tiles_phase56.size() << " tiles"
              << " cpu_workers=" << parallel_tiles
              << " gpu="
              << (tile_reconstruction_acceleration.using_gpu ? "yes" : "no")
              << " backend=" << core::acceleration_backend_name(
                                     tile_reconstruction_acceleration.selected)
              << std::endl;

    tile_valid_counts.assign(tiles_phase56.size(), 0);
    tile_fallback_used.assign(tiles_phase56.size(), 0);
    tile_warp_variances.assign(tiles_phase56.size(), 0.0f);
    tile_mean_correlations.assign(tiles_phase56.size(), 0.0f);
    tile_post_contrast.assign(tiles_phase56.size(), 0.0f);
    tile_post_background.assign(tiles_phase56.size(), 0.0f);
    tile_norm_bg_r.assign(tiles_phase56.size(), 0.0f);
    tile_norm_bg_g.assign(tiles_phase56.size(), 0.0f);
    tile_norm_bg_b.assign(tiles_phase56.size(), 0.0f);
    tile_norm_scale.assign(tiles_phase56.size(), 1.0f);
    std::vector<reconstruction::TileNormalizationStats> tile_norm_stats(
        tiles_phase56.size());
    std::vector<reconstruction::PositiveMedianEstimate> tile_bg_r_estimates(
        tiles_phase56.size());
    std::vector<reconstruction::PositiveMedianEstimate> tile_bg_g_estimates(
        tiles_phase56.size());
    std::vector<reconstruction::PositiveMedianEstimate> tile_bg_b_estimates(
        tiles_phase56.size());
    reconstruction::TileNormalizationGuardSummary tile_norm_guard_summary;
    tile_post_snr.assign(tiles_phase56.size(), 0.0f);
    tile_mean_dx.assign(tiles_phase56.size(), 0.0f);
    tile_mean_dy.assign(tiles_phase56.size(), 0.0f);
    std::vector<uint8_t> tile_reconstructed_valid(tiles_phase56.size(), 0u);
    reconstruction::TileBoundaryDiagnostics boundary_diagnostics_raw;
    reconstruction::TileBoundaryDiagnostics boundary_diagnostics_normalized;
    reconstruction::TileWeightProfileDiagnostics boundary_weight_profile_diagnostics;
    float boundary_valid_count_delta_mean_abs = 0.0f;
    float boundary_valid_count_delta_p95_abs = 0.0f;
    float boundary_post_background_delta_mean_abs = 0.0f;
    float boundary_post_background_delta_p95_abs = 0.0f;
    float boundary_post_snr_delta_mean_abs = 0.0f;
    float boundary_post_snr_delta_p95_abs = 0.0f;
    float boundary_mean_correlation_delta_mean_abs = 0.0f;
    float boundary_mean_correlation_delta_p95_abs = 0.0f;
    int boundary_fallback_mismatch_count = 0;
    std::vector<Matrix2Df> reconstructed_tiles(tiles_phase56.size());
    std::vector<Matrix2Df> reconstructed_tiles_R;
    std::vector<Matrix2Df> reconstructed_tiles_G;
    std::vector<Matrix2Df> reconstructed_tiles_B;
    if (osc_mode) {
      reconstructed_tiles_R.resize(tiles_phase56.size());
      reconstructed_tiles_G.resize(tiles_phase56.size());
      reconstructed_tiles_B.resize(tiles_phase56.size());
    }
    for (auto &c : frame_valid_tile_counts)
      c.store(0);

    // v3.3.9: tile-wise median/MAD normalization before OLA is no longer part
    // of the mandatory linear reconstruction core.
    const bool phase7_tile_norm_requested = false;
    bool apply_phase7_tile_norm = false;
    bool tile_norm_disabled_due_boundary_regression = false;
    float tile_norm_boundary_regression_ratio = 1.0f;
    std::string tile_norm_application = "disabled_v3_3_9_linear_core";
    const std::string &tile_reconstruction_diagnostics_mode =
        cfg.runtime_limits.tile_reconstruction_diagnostics;
    // Use the explicit bool field (set by YAML or derived from the legacy string field).
    const bool tile_reconstruction_diagnostics_enabled =
        cfg.runtime_limits.tile_boundary_diagnostics_enabled;
    const bool tile_reconstruction_diagnostics_full =
        tile_reconstruction_diagnostics_enabled &&
        (tile_reconstruction_diagnostics_mode == "full");

    std::mutex progress_mutex;
    std::atomic<size_t> tiles_completed{0};
    std::atomic<size_t> tiles_failed{0};

    core::WorkerCudaStreams tile_rec_streams(
        tile_reconstruction_acceleration.selected ==
            core::AccelerationBackend::opencv_cuda,
        static_cast<size_t>(std::max(1, parallel_tiles)));

    const auto tile_ola_coeff_cache = build_tile_ola_coeff_cache(
        tiles_phase56, tile_window_cache, common_valid_mask, canvas_width,
        canvas_height);

    std::unique_ptr<tile_compile::runner::DiskCacheFrameStore> osc_rgb_cache_r;
    std::unique_ptr<tile_compile::runner::DiskCacheFrameStore> osc_rgb_cache_g;
    std::unique_ptr<tile_compile::runner::DiskCacheFrameStore> osc_rgb_cache_b;
    bool use_full_frame_osc_rgb_cache = false;
    if (osc_mode && !frames.empty()) {
      const size_t budget_bytes =
          static_cast<size_t>(std::max(1, cfg.runtime_limits.memory_budget)) *
          1024ull * 1024ull;
      const size_t canvas_pixels =
          static_cast<size_t>(std::max(0, canvas_width)) *
          static_cast<size_t>(std::max(0, canvas_height));
      const size_t full_frame_rgb_cache_bytes =
          canvas_pixels * sizeof(float) * 3u *
          static_cast<size_t>(std::max(1, n_usable_frames));
      // The RGB cache is only safe when its full footprint fits into a
      // conservative slice of the configured memory budget. Otherwise, fall
      // back to tile-local debayering to avoid phase-9 OOM kills.
      use_full_frame_osc_rgb_cache =
          full_frame_rgb_cache_bytes <= (budget_bytes / 2u);
      if (!use_full_frame_osc_rgb_cache) {
        std::cout << "[Phase 6] Disabling OSC full-frame RGB cache: estimated "
                  << (full_frame_rgb_cache_bytes / (1024.0 * 1024.0 * 1024.0))
                  << " GiB exceeds conservative budget slice "
                  << ((budget_bytes / 2u) / (1024.0 * 1024.0))
                  << " MiB; using tile-local debayer fallback" << std::endl;
      }
    }
    if (osc_mode && use_full_frame_osc_rgb_cache && !frames.empty()) {
      const fs::path cache_root = run_dir / "cache" / "phase9_osc_rgb";
      osc_rgb_cache_r = std::make_unique<tile_compile::runner::DiskCacheFrameStore>(
          cache_root / "R", frames.size(), canvas_height, canvas_width);
      osc_rgb_cache_g = std::make_unique<tile_compile::runner::DiskCacheFrameStore>(
          cache_root / "G", frames.size(), canvas_height, canvas_width);
      osc_rgb_cache_b = std::make_unique<tile_compile::runner::DiskCacheFrameStore>(
          cache_root / "B", frames.size(), canvas_height, canvas_width);

      const int osc_cache_workers = std::max(
          1, compute_adaptive_worker_count(cfg, frames.size(), frames,
                                           tile_compile::runner::WorkerParallelProfile::MixedIo));
      std::cout << "[Phase 6] Building OSC full-frame RGB cache with "
                << osc_cache_workers << " workers" << std::endl;
      std::atomic<size_t> rgb_next{0};
      std::atomic<size_t> rgb_done{0};
      std::atomic<bool> rgb_failed{false};
      std::mutex rgb_error_mutex;
      std::string rgb_error;

      auto build_rgb_cache_worker = [&]() {
        Matrix2Df deb_r;
        Matrix2Df deb_g;
        Matrix2Df deb_b;
        while (true) {
          const size_t fi = rgb_next.fetch_add(1);
          if (fi >= frames.size()) {
            break;
          }
          try {
            if (frame_has_data[fi]) {
              Matrix2Df frame_mosaic = prewarped_frames.load(fi);
              if (frame_mosaic.rows() == canvas_height &&
                  frame_mosaic.cols() == canvas_width) {
                image::debayer_nearest_neighbor_into(
                    frame_mosaic, detected_bayer, 0, 0, deb_r, deb_g, deb_b);
                if (tile_compile::runner::
                        apply_common_overlap_to_rgb_frames_inplace_and_check_nonzero(
                            deb_r, deb_g, deb_b, common_valid_mask,
                            canvas_width, canvas_height)) {
                  osc_rgb_cache_r->store(fi, deb_r);
                  osc_rgb_cache_g->store(fi, deb_g);
                  osc_rgb_cache_b->store(fi, deb_b);
                }
              }
            }
          } catch (const std::exception &e) {
            rgb_failed.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(rgb_error_mutex);
            if (rgb_error.empty()) {
              rgb_error = e.what();
            }
          } catch (...) {
            rgb_failed.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(rgb_error_mutex);
            if (rgb_error.empty()) {
              rgb_error = "phase6_osc_rgb_cache_unknown_error";
            }
          }

          const size_t done = rgb_done.fetch_add(1) + 1;
          if (done % 32 == 0 || done == frames.size()) {
            std::cout << "[Phase 6] OSC RGB cache " << done << "/"
                      << frames.size() << std::endl;
          }
        }
      };

      if (osc_cache_workers > 1) {
        std::vector<std::thread> rgb_workers;
        rgb_workers.reserve(static_cast<size_t>(osc_cache_workers));
        for (int w = 0; w < osc_cache_workers; ++w) {
          rgb_workers.emplace_back(build_rgb_cache_worker);
        }
        for (auto &worker : rgb_workers) {
          worker.join();
        }
      } else {
        build_rgb_cache_worker();
      }

      if (rgb_failed.load(std::memory_order_relaxed)) {
        const std::string err =
            rgb_error.empty() ? "phase6_osc_rgb_cache_unknown_error" : rgb_error;
        emitter.phase_end(run_id, reconstruction_phase, "error",
                          {{"error", err}}, log_file);
        emitter.run_end(run_id, false, "error", log_file,
                        {{"message", std::string("Phase 6 OSC RGB cache build: ") + err}});
        std::cerr << "Error during Phase 6 OSC RGB cache build: " << err
                  << std::endl;
        return 1;
      }
    }

    // Worker function for parallel tile processing (v3: global warp only, no
    // local ECC)
    auto process_tile = [&](size_t ti, cv::cuda::Stream *tile_stream) {
      const Tile &t = tiles_phase56[ti];
      const bool tile_has_common_overlap =
          ti < tile_common_valid.size() && tile_common_valid[ti] != 0;
      const bool tile_has_reconstruction_support =
          ti < tile_reconstruction_valid.size() &&
          tile_reconstruction_valid[ti] != 0;
      auto compute_frame_tile_weight = [&](size_t fi) -> float {
        const float G_f = (fi < static_cast<size_t>(global_weights.size()))
                              ? global_weights[static_cast<int>(fi)]
                              : 1.0f;
        const float L_ft =
            (fi < local_weights.size() && ti < local_weights[fi].size())
                ? local_weights[fi][ti]
                : 1.0f;
        const float local_weight =
            (!(L_ft > 0.0f) && !tile_has_common_overlap &&
             tile_has_reconstruction_support)
                ? 1.0f
                : L_ft;
        const float w = G_f * local_weight;
        return (std::isfinite(w) && w > 0.0f) ? w : 0.0f;
      };

      std::vector<float> weights; // kept for potential future use

      Matrix2Df tile_rec;
      Matrix2Df tile_rec_R = Matrix2Df::Zero(t.height, t.width);
      Matrix2Df tile_rec_G = Matrix2Df::Zero(t.height, t.width);
      Matrix2Df tile_rec_B = Matrix2Df::Zero(t.height, t.width);
      size_t n_valid = 0;
      bool used_weight_fallback = false;
      std::vector<uint8_t> tile_support_mask;
      std::vector<uint8_t> tile_support_mask_r;
      std::vector<uint8_t> tile_support_mask_g;
      std::vector<uint8_t> tile_support_mask_b;
      auto capture_finite_mask = [](const Matrix2Df &img) {
        std::vector<uint8_t> mask(static_cast<size_t>(std::max<Eigen::Index>(0, img.size())),
                                  0u);
        for (Eigen::Index i = 0; i < img.size(); ++i) {
          if (std::isfinite(img.data()[i])) {
            mask[static_cast<size_t>(i)] = 1u;
          }
        }
        return mask;
      };
      auto has_any_supported = [](const std::vector<uint8_t> &mask) {
        return std::any_of(mask.begin(), mask.end(),
                           [](uint8_t v) { return v != 0u; });
      };
      auto replace_nonfinite_with_zero = [](Matrix2Df &img) {
        for (Eigen::Index i = 0; i < img.size(); ++i) {
          if (!std::isfinite(img.data()[i])) {
            img.data()[i] = 0.0f;
          }
        }
      };
      auto apply_support_mask = [](Matrix2Df &img,
                                   const std::vector<uint8_t> &mask) {
        const float invalid = std::numeric_limits<float>::quiet_NaN();
        for (Eigen::Index i = 0; i < img.size(); ++i) {
          if (static_cast<size_t>(i) >= mask.size() ||
              mask[static_cast<size_t>(i)] == 0u) {
            img.data()[i] = invalid;
          }
        }
      };

      // Sub-batch size: limits peak RAM per worker to sub_batch × tile_pixels × channels.
      const size_t n_frames_total = frames.size();
      const size_t sub_batch = (frame_sub_batch_size > 0 && frame_sub_batch_size < n_frames_total)
                               ? frame_sub_batch_size : n_frames_total;

      if (osc_mode) {
        // Methodik v3 (OSC): stack in RGB space (debayer-before-stack).
        // Prefer a full-frame RGB cache only when it fits the memory model;
        // otherwise fall back to tile-local debayering.
        if (!tile_has_reconstruction_support) {
          tiles_failed++;
          return;
        }

        // Sub-batch stacking: process frames in chunks of frame_sub_batch_size
        // to keep peak RAM at sub_batch × tile_pixels × 3ch × workers
        // instead of all_frames × tile_pixels × 3ch × workers.
        // Results are accumulated as a weighted mean across batches.
        Matrix2Df accum_R = Matrix2Df::Zero(t.height, t.width);
        Matrix2Df accum_G = Matrix2Df::Zero(t.height, t.width);
        Matrix2Df accum_B = Matrix2Df::Zero(t.height, t.width);
        Matrix2Df wsum_R  = Matrix2Df::Zero(t.height, t.width);
        Matrix2Df wsum_G  = Matrix2Df::Zero(t.height, t.width);
        Matrix2Df wsum_B  = Matrix2Df::Zero(t.height, t.width);
        size_t total_valid = 0;

        for (size_t batch_start = 0; batch_start < n_frames_total; batch_start += sub_batch) {
          const size_t batch_end = std::min(batch_start + sub_batch, n_frames_total);

          std::vector<float> channel_weights;
          std::vector<Matrix2Df> valid_tiles_R;
          std::vector<Matrix2Df> valid_tiles_G;
          std::vector<Matrix2Df> valid_tiles_B;
          channel_weights.reserve(batch_end - batch_start);
          valid_tiles_R.reserve(batch_end - batch_start);
          valid_tiles_G.reserve(batch_end - batch_start);
          valid_tiles_B.reserve(batch_end - batch_start);
        if (use_full_frame_osc_rgb_cache && osc_rgb_cache_r && osc_rgb_cache_g &&
            osc_rgb_cache_b) {
          Matrix2Df tile_r;
          Matrix2Df tile_g;
          Matrix2Df tile_b;

          // Single pass: extract RGB tiles from the prebuilt full-frame cache
          // and compute shared channel weights.
          for (size_t fi = batch_start; fi < batch_end; ++fi) {
            if (!frame_has_data[fi])
              continue;
            const float frame_weight = compute_frame_tile_weight(fi);
            if (!(frame_weight > 0.0f))
              continue;
            if (!osc_rgb_cache_r->extract_tile_into(fi, t, tile_r, 0, 0) ||
                !osc_rgb_cache_g->extract_tile_into(fi, t, tile_g, 0, 0) ||
                !osc_rgb_cache_b->extract_tile_into(fi, t, tile_b, 0, 0) ||
                tile_r.rows() != t.height || tile_r.cols() != t.width ||
                tile_g.rows() != t.height || tile_g.cols() != t.width ||
                tile_b.rows() != t.height || tile_b.cols() != t.width) {
              continue;
            }
            if (!tile_compile::runner::tile_has_nonzero_common_data(tile_g, ti,
                                                                    tile_reconstruction_valid)) {
              continue;
            }

            valid_tiles_R.push_back(std::move(tile_r));
            valid_tiles_G.push_back(std::move(tile_g));
            valid_tiles_B.push_back(std::move(tile_b));
            frame_valid_tile_counts[fi].fetch_add(1);
            channel_weights.push_back(frame_weight);
          }
        } else {
          const int origin_x = std::max(0, t.x);
          const int origin_y = std::max(0, t.y);
          Matrix2Df tile_r;
          Matrix2Df tile_g;
          Matrix2Df tile_b;
          for (size_t fi = batch_start; fi < batch_end; ++fi) {
            if (!frame_has_data[fi])
              continue;
            const float frame_weight = compute_frame_tile_weight(fi);
            if (!(frame_weight > 0.0f))
              continue;
            const float *frame_ptr = prewarped_frames.frame_data(fi);
            if (frame_ptr == nullptr || origin_x < 0 || origin_y < 0 ||
                origin_x + t.width > canvas_width ||
                origin_y + t.height > canvas_height) {
              continue;
            }
            const float *tile_ptr =
                frame_ptr +
                static_cast<size_t>(origin_y) *
                    static_cast<size_t>(canvas_width) +
                static_cast<size_t>(origin_x);
            image::debayer_nearest_neighbor_strided_into(
                tile_ptr, t.height, t.width, canvas_width, detected_bayer,
                origin_x, origin_y, tile_r, tile_g, tile_b);
            if (!tile_compile::runner::
                    apply_common_overlap_to_rgb_tiles_inplace_and_check_nonzero(
                        tile_r, tile_g, tile_b, t, common_valid_mask,
                        canvas_width, canvas_height)) {
              continue;
            }

            valid_tiles_R.push_back(std::move(tile_r));
            valid_tiles_G.push_back(std::move(tile_g));
            valid_tiles_B.push_back(std::move(tile_b));
            frame_valid_tile_counts[fi].fetch_add(1);
            channel_weights.push_back(frame_weight);
          }
        }

        if (valid_tiles_G.empty()) {
          continue; // skip empty batch, try next
        }

        // Stack this sub-batch and accumulate into weighted mean.
        auto stack_channel_batch = [&](const std::vector<Matrix2Df> &chan_tiles)
            -> Matrix2Df {
          if (channel_weights.size() != chan_tiles.size()) {
            std::vector<float> fw(chan_tiles.size(), 1.0f);
            auto wr = tile_reconstruction_ops.sigma_clip_reduce(
                chan_tiles, fw,
                cfg.stacking.sigma_clip.sigma_low,
                cfg.stacking.sigma_clip.sigma_high,
                cfg.stacking.sigma_clip.max_iters,
                cfg.stacking.sigma_clip.min_fraction, kEpsWeight, tile_stream);
            used_weight_fallback = used_weight_fallback || wr.fallback_used;
            return std::move(wr.tile);
          }
          auto wr = tile_reconstruction_ops.sigma_clip_reduce(
              chan_tiles, channel_weights,
              cfg.stacking.sigma_clip.sigma_low,
              cfg.stacking.sigma_clip.sigma_high,
              cfg.stacking.sigma_clip.max_iters,
              cfg.stacking.sigma_clip.min_fraction, kEpsWeight, tile_stream);
          used_weight_fallback = used_weight_fallback || wr.fallback_used;
          return std::move(wr.tile);
        };

        Matrix2Df batch_R = stack_channel_batch(valid_tiles_R);
        Matrix2Df batch_G = stack_channel_batch(valid_tiles_G);
        Matrix2Df batch_B = stack_channel_batch(valid_tiles_B);

        if (batch_G.size() <= 0) continue;

        // Weighted accumulation: weight = number of valid frames in this batch.
        const float batch_w = static_cast<float>(valid_tiles_G.size());
        total_valid += valid_tiles_G.size();
        for (int py = 0; py < t.height; ++py) {
          for (int px = 0; px < t.width; ++px) {
            if (std::isfinite(batch_R(py, px))) { accum_R(py, px) += batch_R(py, px) * batch_w; wsum_R(py, px) += batch_w; }
            if (std::isfinite(batch_G(py, px))) { accum_G(py, px) += batch_G(py, px) * batch_w; wsum_G(py, px) += batch_w; }
            if (std::isfinite(batch_B(py, px))) { accum_B(py, px) += batch_B(py, px) * batch_w; wsum_B(py, px) += batch_w; }
          }
        }
      } // end sub-batch loop

        if (total_valid == 0) {
          tiles_failed++;
          return;
        }

        // Normalise accumulators.
        for (int py = 0; py < t.height; ++py) {
          for (int px = 0; px < t.width; ++px) {
            tile_rec_R(py, px) = (wsum_R(py, px) > 0.0f) ? accum_R(py, px) / wsum_R(py, px) : std::numeric_limits<float>::quiet_NaN();
            tile_rec_G(py, px) = (wsum_G(py, px) > 0.0f) ? accum_G(py, px) / wsum_G(py, px) : std::numeric_limits<float>::quiet_NaN();
            tile_rec_B(py, px) = (wsum_B(py, px) > 0.0f) ? accum_B(py, px) / wsum_B(py, px) : std::numeric_limits<float>::quiet_NaN();
          }
        }

        if (tile_rec_R.size() <= 0 || tile_rec_G.size() <= 0 ||
            tile_rec_B.size() <= 0) {
          tiles_failed++;
          return;
        }

        tile_support_mask_r = capture_finite_mask(tile_rec_R);
        tile_support_mask_g = capture_finite_mask(tile_rec_G);
        tile_support_mask_b = capture_finite_mask(tile_rec_B);
        if (!has_any_supported(tile_support_mask_g)) {
          tiles_failed++;
          return;
        }
        replace_nonfinite_with_zero(tile_rec_R);
        replace_nonfinite_with_zero(tile_rec_G);
        replace_nonfinite_with_zero(tile_rec_B);
        // Post-metrics are computed on G as a stable luminance proxy.
        tile_rec = tile_rec_G;
        n_valid = total_valid;
      } else {
        if (!tile_has_reconstruction_support) {
          tiles_failed++;
          return;
        }

        // MONO sub-batch stacking: accumulate weighted mean across frame batches.
        Matrix2Df accum_mono = Matrix2Df::Zero(t.height, t.width);
        Matrix2Df wsum_mono  = Matrix2Df::Zero(t.height, t.width);
        size_t total_valid_mono = 0;

        for (size_t batch_start = 0; batch_start < frames.size(); batch_start += sub_batch) {
          const size_t batch_end = std::min(batch_start + sub_batch, frames.size());          std::vector<Matrix2Df> warped_tiles;
          warped_tiles.reserve(batch_end - batch_start);
          std::vector<float> batch_weights;
          batch_weights.reserve(batch_end - batch_start);

          for (size_t fi = batch_start; fi < batch_end; ++fi) {
            if (!frame_has_data[fi])
              continue;
            const float frame_weight = compute_frame_tile_weight(fi);
            if (!(frame_weight > 0.0f))
              continue;
            warped_tiles.emplace_back();
            Matrix2Df &tile_img = warped_tiles.back();
            if (!prewarped_frames.extract_tile_into(fi, t, tile_img, 0, 0) ||
                tile_img.rows() != t.height || tile_img.cols() != t.width) {
              warped_tiles.pop_back();
              continue;
            }
            if (!tile_compile::runner::
                    apply_common_overlap_to_tile_inplace_and_check_nonzero(
                        tile_img, t, reconstruction_valid_mask, canvas_width,
                        canvas_height)) {
              warped_tiles.pop_back();
              continue;
            }
            frame_valid_tile_counts[fi].fetch_add(1);
            batch_weights.push_back(frame_weight);
          }

          if (warped_tiles.empty()) continue;

          auto wr = tile_reconstruction_ops.sigma_clip_reduce(
              warped_tiles, batch_weights, cfg.stacking.sigma_clip.sigma_low,
              cfg.stacking.sigma_clip.sigma_high,
              cfg.stacking.sigma_clip.max_iters,
              cfg.stacking.sigma_clip.min_fraction, kEpsWeight, tile_stream);
          used_weight_fallback = used_weight_fallback || wr.fallback_used;

          const float bw = static_cast<float>(warped_tiles.size());
          total_valid_mono += warped_tiles.size();
          for (int py = 0; py < t.height; ++py)
            for (int px = 0; px < t.width; ++px)
              if (std::isfinite(wr.tile(py, px))) {
                accum_mono(py, px) += wr.tile(py, px) * bw;
                wsum_mono(py, px)  += bw;
              }
        }

        if (total_valid_mono == 0) {
          tiles_failed++;
          return;
        }

        tile_rec = Matrix2Df(t.height, t.width);
        for (int py = 0; py < t.height; ++py)
          for (int px = 0; px < t.width; ++px)
            tile_rec(py, px) = (wsum_mono(py, px) > 0.0f)
                               ? accum_mono(py, px) / wsum_mono(py, px)
                               : std::numeric_limits<float>::quiet_NaN();
        tile_support_mask = capture_finite_mask(tile_rec);
        if (!has_any_supported(tile_support_mask)) {
          tiles_failed++;
          return;
        }
        replace_nonfinite_with_zero(tile_rec);
        n_valid = total_valid_mono;
      }

      tile_valid_counts[ti] = static_cast<int>(n_valid);
      tile_fallback_used[ti] = used_weight_fallback ? 1u : 0u;
      tile_warp_variances[ti] = 0.0f;
      tile_mean_correlations[ti] = 1.0f;
      tile_mean_dx[ti] = 0.0f;
      tile_mean_dy[ti] = 0.0f;

      // Methodik 3.1E §3.3.1: Tile denoising after stacking, before OLA.
      // 1. Soft-Threshold (Highpass + shrinkage) — always first (spatial domain)
      bool is_star = (ti < tile_is_star.size()) && tile_is_star[ti];
      if (cfg.tile_denoise.soft_threshold.enabled &&
          !(cfg.tile_denoise.soft_threshold.skip_star_tiles && is_star)) {
        tile_rec = reconstruction::soft_threshold_tile_filter(
            tile_rec, cfg.tile_denoise.soft_threshold);
        if (osc_mode) {
          tile_rec_R = reconstruction::soft_threshold_tile_filter(
              tile_rec_R, cfg.tile_denoise.soft_threshold);
          tile_rec_G = reconstruction::soft_threshold_tile_filter(
              tile_rec_G, cfg.tile_denoise.soft_threshold);
          tile_rec_B = reconstruction::soft_threshold_tile_filter(
              tile_rec_B, cfg.tile_denoise.soft_threshold);
        }
      }

      // 2. Wiener filter (frequency domain) — applied after soft-threshold
      float tile_noise = (ti < tile_quality_median.size())
                             ? tile_quality_median[ti]
                             : 0.0f;
      float tile_snr = (tile_post_snr.size() > ti) ? tile_post_snr[ti] : 0.0f;
      float tile_q = (ti < tile_quality_median.size())
                          ? tile_quality_median[ti]
                          : 0.0f;
      if (cfg.tile_denoise.wiener.enabled) {
        // Estimate noise from tile residual for Wiener filter
        auto estimate_tile_noise = [](const Matrix2Df &t_img) -> float {
          if (t_img.size() <= 0) return 0.0f;
          cv::Mat m(t_img.rows(), t_img.cols(), CV_32F,
                    const_cast<float *>(t_img.data()));
          cv::Mat bg_m;
          cv::blur(m, bg_m, cv::Size(31, 31), cv::Point(-1, -1),
                   cv::BORDER_REFLECT_101);
          cv::Mat r = m - bg_m;
          
          std::vector<float> rv;
          rv.reserve(static_cast<size_t>(r.total()));
          for (int i = 0; i < static_cast<int>(r.total()); ++i) {
            float v = r.ptr<float>()[i];
            if (std::isfinite(v)) rv.push_back(v);
          }
          if (rv.empty()) return 0.0f;
          size_t mid = rv.size() / 2;
          std::nth_element(rv.begin(), rv.begin() + mid, rv.end());
          float med = rv[mid];
          for (float &v : rv) v = std::fabs(v - med);
          std::nth_element(rv.begin(), rv.begin() + mid, rv.end());
          return 1.4826f * rv[mid];
        };
        float sigma_est = estimate_tile_noise(tile_rec);
        tile_rec = reconstruction::wiener_tile_filter(
            tile_rec, sigma_est, tile_snr, tile_q, is_star,
            cfg.tile_denoise.wiener);
        if (osc_mode) {
          float sig_r = estimate_tile_noise(tile_rec_R);
          tile_rec_R = reconstruction::wiener_tile_filter(
              tile_rec_R, sig_r, tile_snr, tile_q, is_star,
              cfg.tile_denoise.wiener);
          float sig_g = estimate_tile_noise(tile_rec_G);
          tile_rec_G = reconstruction::wiener_tile_filter(
              tile_rec_G, sig_g, tile_snr, tile_q, is_star,
              cfg.tile_denoise.wiener);
          float sig_b = estimate_tile_noise(tile_rec_B);
          tile_rec_B = reconstruction::wiener_tile_filter(
              tile_rec_B, sig_b, tile_snr, tile_q, is_star,
              cfg.tile_denoise.wiener);
        }
      }

      if (osc_mode && cfg.chroma_denoise.enabled &&
          cfg.chroma_denoise.apply_stage == "pre_stack_tiles") {
        reconstruction::chroma_denoise_rgb_inplace(
            tile_rec_R, tile_rec_G, tile_rec_B, cfg.chroma_denoise);
      }

      if (osc_mode) {
        apply_support_mask(tile_rec_R, tile_support_mask_r);
        apply_support_mask(tile_rec_G, tile_support_mask_g);
        apply_support_mask(tile_rec_B, tile_support_mask_b);
        tile_rec = tile_rec_G;
      } else {
        apply_support_mask(tile_rec, tile_support_mask);
      }

      auto [c, b, s] = compute_post_warp_metrics(tile_rec);
      tile_post_contrast[ti] = c;
      tile_post_background[ti] = b;
      tile_post_snr[ti] = s;
      reconstructed_tiles[ti] = std::move(tile_rec);
      if (osc_mode) {
        reconstructed_tiles_R[ti] = std::move(tile_rec_R);
        reconstructed_tiles_G[ti] = std::move(tile_rec_G);
        reconstructed_tiles_B[ti] = std::move(tile_rec_B);
      }
      tile_reconstructed_valid[ti] = 1u;

      size_t done = ++tiles_completed;
      if (done % 20 == 0 || done == tiles_phase56.size()) {
        std::lock_guard<std::mutex> lock(progress_mutex);
        emitter.phase_progress_counts(
            run_id, reconstruction_phase, static_cast<int>(done),
            static_cast<int>(tiles_phase56.size()),
            "workers=" + std::to_string(parallel_tiles) + " cpu_workers=" +
                std::to_string(parallel_tiles) + " gpu=" +
                (tile_reconstruction_acceleration.using_gpu ? "yes" : "no") +
                " backend=" + core::acceleration_backend_name(
                                   tile_reconstruction_acceleration.selected),
            "tiles", log_file);
      }
    };

    // Execute tiles in parallel or serial based on parallel_tiles setting
    if (parallel_tiles > 1) {
      std::cout << "  Processing " << tiles_phase56.size() << " tiles with "
                << parallel_tiles << " workers..." << std::endl;

      std::vector<std::thread> workers;
      std::atomic<size_t> next_tile{0};

      for (int w = 0; w < parallel_tiles; ++w) {
        workers.emplace_back([&, w]() {
          cv::cuda::Stream *stream_ptr = nullptr;
          stream_ptr = tile_rec_streams.get(static_cast<size_t>(w));
          while (true) {
            size_t ti = next_tile.fetch_add(1);
            if (ti >= tiles_phase56.size())
              break;
            process_tile(ti, stream_ptr);
          }
        });
      }

      for (auto &worker : workers) {
        worker.join();
      }

      std::cout << "  Completed " << tiles_completed.load() << " tiles ("
                << tiles_failed.load() << " failed)" << std::endl;
    } else {
      std::cout << "  Processing " << tiles_phase56.size()
                << " tiles serially..." << std::endl;
      for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
        process_tile(ti, tile_rec_streams.get(0));
      }
    }

    if (osc_rgb_cache_r) {
      osc_rgb_cache_r->cleanup();
      osc_rgb_cache_g->cleanup();
      osc_rgb_cache_b->cleanup();
      osc_rgb_cache_r.reset();
      osc_rgb_cache_g.reset();
      osc_rgb_cache_b.reset();
    }

    if (apply_phase7_tile_norm) {
      constexpr reconstruction::TileNormalizationGuardConfig kTileNormGuardCfg{};
      for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
        if (tile_reconstructed_valid[ti] == 0u) {
          continue;
        }
        tile_norm_stats[ti] = reconstruction::estimate_tile_normalization_stats(
            reconstructed_tiles[ti]);
        if (osc_mode) {
          tile_bg_r_estimates[ti] =
              reconstruction::positive_median(reconstructed_tiles_R[ti]);
          tile_bg_g_estimates[ti] =
              reconstruction::positive_median(reconstructed_tiles_G[ti]);
          tile_bg_b_estimates[ti] =
              reconstruction::positive_median(reconstructed_tiles_B[ti]);
        }
      }
      tile_norm_guard_summary = reconstruction::guard_tile_normalization_stats(
          &tile_norm_stats, tile_reconstructed_valid, kTileNormGuardCfg,
          kEpsMedian);

      std::vector<float> valid_bg_r_vals;
      std::vector<float> valid_bg_g_vals;
      std::vector<float> valid_bg_b_vals;
      valid_bg_r_vals.reserve(tiles_phase56.size());
      valid_bg_g_vals.reserve(tiles_phase56.size());
      valid_bg_b_vals.reserve(tiles_phase56.size());
      for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
        if (tile_reconstructed_valid[ti] == 0u) {
          continue;
        }
        const size_t min_required = reconstruction::minimum_tile_normalization_samples(
            tile_norm_stats[ti].total_count, kTileNormGuardCfg);
        if (!osc_mode) {
          valid_bg_r_vals.push_back(tile_norm_stats[ti].background);
          continue;
        }
        if (tile_bg_r_estimates[ti].sample_count >= min_required &&
            std::isfinite(tile_bg_r_estimates[ti].value)) {
          valid_bg_r_vals.push_back(tile_bg_r_estimates[ti].value);
        }
        if (tile_bg_g_estimates[ti].sample_count >= min_required &&
            std::isfinite(tile_bg_g_estimates[ti].value)) {
          valid_bg_g_vals.push_back(tile_bg_g_estimates[ti].value);
        }
        if (tile_bg_b_estimates[ti].sample_count >= min_required &&
            std::isfinite(tile_bg_b_estimates[ti].value)) {
          valid_bg_b_vals.push_back(tile_bg_b_estimates[ti].value);
        }
      }
      const float global_bg_r = valid_bg_r_vals.empty()
                                    ? tile_norm_guard_summary.global_background
                                    : core::median_of(valid_bg_r_vals);
      const float global_bg_g = valid_bg_g_vals.empty() ? global_bg_r
                                                        : core::median_of(valid_bg_g_vals);
      const float global_bg_b = valid_bg_b_vals.empty() ? global_bg_r
                                                        : core::median_of(valid_bg_b_vals);
      for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
        if (tile_reconstructed_valid[ti] == 0u) {
          continue;
        }
        const size_t min_required = reconstruction::minimum_tile_normalization_samples(
            tile_norm_stats[ti].total_count, kTileNormGuardCfg);
        tile_norm_scale[ti] = std::max(kEpsMedian, tile_norm_stats[ti].scale);
        if (osc_mode) {
          tile_norm_bg_r[ti] =
              (tile_bg_r_estimates[ti].sample_count >= min_required &&
               std::isfinite(tile_bg_r_estimates[ti].value))
                  ? tile_bg_r_estimates[ti].value
                  : global_bg_r;
          tile_norm_bg_g[ti] =
              (tile_bg_g_estimates[ti].sample_count >= min_required &&
               std::isfinite(tile_bg_g_estimates[ti].value))
                  ? tile_bg_g_estimates[ti].value
                  : global_bg_g;
          tile_norm_bg_b[ti] =
              (tile_bg_b_estimates[ti].sample_count >= min_required &&
               std::isfinite(tile_bg_b_estimates[ti].value))
                  ? tile_bg_b_estimates[ti].value
                  : global_bg_b;
        } else {
          tile_norm_bg_r[ti] = tile_norm_stats[ti].background;
        }
      }
    }

    if (tile_reconstruction_diagnostics_enabled) {
      auto summarize_abs_metric = [](std::vector<float> values) {
        std::pair<float, float> summary{0.0f, 0.0f};
        if (values.empty()) {
          return summary;
        }
        summary.first =
            std::accumulate(values.begin(), values.end(), 0.0f) /
            static_cast<float>(values.size());
        std::sort(values.begin(), values.end());
        summary.second = core::percentile_from_sorted(values, 95.0f);
        return summary;
      };

      auto build_boundary_input_tiles = [&](bool normalized) {
        std::vector<Matrix2Df> boundary_input_tiles(tiles_phase56.size());
        for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
          if (tile_reconstructed_valid[ti] == 0u) {
            continue;
          }

          if (osc_mode) {
            Matrix2Df diag_r = reconstructed_tiles_R[ti];
            Matrix2Df diag_g = reconstructed_tiles_G[ti];
            Matrix2Df diag_b = reconstructed_tiles_B[ti];
            if (normalized) {
              const float inv = 1.0f / std::max(kEpsMedian, tile_norm_scale[ti]);
              for (Eigen::Index i = 0; i < diag_r.size(); ++i) {
                diag_r.data()[i] = (diag_r.data()[i] - tile_norm_bg_r[ti]) * inv;
                diag_g.data()[i] = (diag_g.data()[i] - tile_norm_bg_g[ti]) * inv;
                diag_b.data()[i] = (diag_b.data()[i] - tile_norm_bg_b[ti]) * inv;
              }
            }
            boundary_input_tiles[ti] =
                0.25f * diag_r + 0.5f * diag_g + 0.25f * diag_b;
          } else {
            Matrix2Df diag_tile = reconstructed_tiles[ti];
            if (normalized) {
              const float inv = 1.0f / std::max(kEpsMedian, tile_norm_scale[ti]);
              for (Eigen::Index i = 0; i < diag_tile.size(); ++i) {
                diag_tile.data()[i] =
                    (diag_tile.data()[i] - tile_norm_bg_r[ti]) * inv;
              }
            }
            boundary_input_tiles[ti] = std::move(diag_tile);
          }
        }
        return boundary_input_tiles;
      };

      if (osc_mode) {
        const auto boundary_input_tiles_raw = build_boundary_input_tiles(false);
        boundary_diagnostics_raw = reconstruction::analyze_tile_boundaries(
            tiles_phase56, boundary_input_tiles_raw, tile_reconstructed_valid,
            common_valid_mask, canvas_width, canvas_height);
      } else {
        boundary_diagnostics_raw = reconstruction::analyze_tile_boundaries(
            tiles_phase56, reconstructed_tiles, tile_reconstructed_valid,
            common_valid_mask, canvas_width, canvas_height);
      }

      if (phase7_tile_norm_requested) {
        const auto boundary_input_tiles_normalized = build_boundary_input_tiles(true);
        boundary_diagnostics_normalized = reconstruction::analyze_tile_boundaries(
            tiles_phase56, boundary_input_tiles_normalized,
            tile_reconstructed_valid, common_valid_mask, canvas_width,
            canvas_height);
        if (should_disable_phase7_tile_norm(boundary_diagnostics_raw,
                                            boundary_diagnostics_normalized,
                                            &tile_norm_boundary_regression_ratio)) {
          apply_phase7_tile_norm = false;
          tile_norm_disabled_due_boundary_regression = true;
          tile_norm_application = "disabled_boundary_regression";
          std::ostringstream msg;
          msg << "TILE_RECONSTRUCTION disabled phase7 tile normalization: "
              << "boundary p95 regressed from "
              << boundary_diagnostics_raw.pair_mean_abs_diff_p95 << " to "
              << boundary_diagnostics_normalized.pair_mean_abs_diff_p95
              << " (ratio=" << tile_norm_boundary_regression_ratio << ")";
          emitter.warning(run_id, msg.str(), log_file);
          std::cout << "[TILE_RECONSTRUCTION] " << msg.str() << std::endl;
        }
      } else {
        boundary_diagnostics_normalized = boundary_diagnostics_raw;
      }
      const auto &boundary_diagnostics_active =
          apply_phase7_tile_norm ? boundary_diagnostics_normalized
                                 : boundary_diagnostics_raw;
      if (tile_reconstruction_diagnostics_full) {
        boundary_weight_profile_diagnostics =
            reconstruction::analyze_tile_weight_profiles(
                boundary_diagnostics_active.pair_diagnostics, local_weights,
                frame_has_data);
      }

      std::vector<float> valid_count_deltas;
      std::vector<float> background_deltas;
      std::vector<float> snr_deltas;
      std::vector<float> correlation_deltas;
      valid_count_deltas.reserve(
          boundary_diagnostics_active.pair_diagnostics.size());
      background_deltas.reserve(
          boundary_diagnostics_active.pair_diagnostics.size());
      if (tile_reconstruction_diagnostics_full) {
        snr_deltas.reserve(boundary_diagnostics_active.pair_diagnostics.size());
        correlation_deltas.reserve(
            boundary_diagnostics_active.pair_diagnostics.size());
      }
      for (const auto &pair : boundary_diagnostics_active.pair_diagnostics) {
        valid_count_deltas.push_back(static_cast<float>(
            std::abs(tile_valid_counts[pair.lhs] - tile_valid_counts[pair.rhs])));
        background_deltas.push_back(std::fabs(
            tile_post_background[pair.lhs] - tile_post_background[pair.rhs]));
        if (tile_reconstruction_diagnostics_full) {
          snr_deltas.push_back(
              std::fabs(tile_post_snr[pair.lhs] - tile_post_snr[pair.rhs]));
          correlation_deltas.push_back(
              std::fabs(tile_mean_correlations[pair.lhs] -
                        tile_mean_correlations[pair.rhs]));
        }
        if ((tile_fallback_used[pair.lhs] != 0u) !=
            (tile_fallback_used[pair.rhs] != 0u)) {
          ++boundary_fallback_mismatch_count;
        }
      }

      auto valid_count_summary = summarize_abs_metric(std::move(valid_count_deltas));
      boundary_valid_count_delta_mean_abs = valid_count_summary.first;
      boundary_valid_count_delta_p95_abs = valid_count_summary.second;
      auto background_summary = summarize_abs_metric(std::move(background_deltas));
      boundary_post_background_delta_mean_abs = background_summary.first;
      boundary_post_background_delta_p95_abs = background_summary.second;
      if (tile_reconstruction_diagnostics_full) {
        auto snr_summary = summarize_abs_metric(std::move(snr_deltas));
        boundary_post_snr_delta_mean_abs = snr_summary.first;
        boundary_post_snr_delta_p95_abs = snr_summary.second;
        auto correlation_summary =
            summarize_abs_metric(std::move(correlation_deltas));
        boundary_mean_correlation_delta_mean_abs = correlation_summary.first;
        boundary_mean_correlation_delta_p95_abs = correlation_summary.second;
      }
    }

    for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
      if (tile_reconstructed_valid[ti] == 0u) {
        continue;
      }
      const Tile &t = tiles_phase56[ti];
      if (ti >= tile_ola_coeff_cache.size() ||
          !tile_ola_coeff_cache[ti].has_nonzero) {
        continue;
      }
      const Matrix2Df &tile_ola_coeff = tile_ola_coeff_cache[ti].coeff;

      if (osc_mode) {
        Matrix2Df tile_r = reconstructed_tiles_R[ti];
        Matrix2Df tile_g = reconstructed_tiles_G[ti];
        Matrix2Df tile_b = reconstructed_tiles_B[ti];
        if (apply_phase7_tile_norm) {
          const float inv = 1.0f / std::max(kEpsMedian, tile_norm_scale[ti]);
          for (Eigen::Index i = 0; i < tile_r.size(); ++i) {
            tile_r.data()[i] = (tile_r.data()[i] - tile_norm_bg_r[ti]) * inv;
            tile_g.data()[i] = (tile_g.data()[i] - tile_norm_bg_g[ti]) * inv;
            tile_b.data()[i] = (tile_b.data()[i] - tile_norm_bg_b[ti]) * inv;
          }
        }

        tile_reconstruction_ops.overlap_add(tile_r, t, tile_ola_coeff, recon_R,
                                            weight_sum, true);
        tile_reconstruction_ops.overlap_add(tile_g, t, tile_ola_coeff, recon_G,
                                            weight_sum, false);
        tile_reconstruction_ops.overlap_add(tile_b, t, tile_ola_coeff, recon_B,
                                            weight_sum, false);
      } else {
        Matrix2Df tile = reconstructed_tiles[ti];
        if (apply_phase7_tile_norm) {
          const float inv = 1.0f / std::max(kEpsMedian, tile_norm_scale[ti]);
          for (Eigen::Index i = 0; i < tile.size(); ++i) {
            tile.data()[i] = (tile.data()[i] - tile_norm_bg_r[ti]) * inv;
          }
        }

        tile_reconstruction_ops.overlap_add(tile, t, tile_ola_coeff, recon,
                                            weight_sum, true);
      }
    }

    cv::setNumThreads(prev_cv_threads_recon);

    bool overlap_normalized_on_device = false;
    const float invalid_ws = std::numeric_limits<float>::quiet_NaN();
    if (osc_mode) {
      const bool norm_r = tile_reconstruction_ops.normalize_overlap_accum(
          recon_R, weight_sum, eps_ws, invalid_ws);
      const bool norm_g = tile_reconstruction_ops.normalize_overlap_accum(
          recon_G, weight_sum, eps_ws, invalid_ws);
      const bool norm_b = tile_reconstruction_ops.normalize_overlap_accum(
          recon_B, weight_sum, eps_ws, invalid_ws);
      overlap_normalized_on_device = norm_r && norm_g && norm_b;
    } else {
      overlap_normalized_on_device = tile_reconstruction_ops
                                         .normalize_overlap_accum(
                                             recon, weight_sum, eps_ws,
                                             invalid_ws);
    }

    if (osc_mode) {
      tile_reconstruction_ops.flush_overlap_state(recon_R, weight_sum);
      tile_reconstruction_ops.flush_overlap_state(recon_G, weight_sum);
      tile_reconstruction_ops.flush_overlap_state(recon_B, weight_sum);
    } else {
      tile_reconstruction_ops.flush_overlap_state(recon, weight_sum);
    }

    // Normalize reconstruction
    if (osc_mode) {
      if (!overlap_normalized_on_device) {
        for (int i = 0; i < recon.size(); ++i) {
          float ws = weight_sum.data()[i];
          if (ws > eps_ws) {
            recon_R.data()[i] /= ws;
            recon_G.data()[i] /= ws;
            recon_B.data()[i] /= ws;
          } else {
            // Mark canvas dead area with NaN sentinel (impossible sensor value)
            // so downstream logic can reject it robustly via std::isfinite().
            recon_R.data()[i] = invalid_ws;
            recon_G.data()[i] = invalid_ws;
            recon_B.data()[i] = invalid_ws;
          }
        }
      }

      if (apply_phase7_tile_norm) {
        // Methodik v3.2 §5.7.2 (optional): global robust tile background
        // restore.
        std::vector<float> bg_r_vals;
        std::vector<float> bg_g_vals;
        std::vector<float> bg_b_vals;
        std::vector<float> m_vals;
        bg_r_vals.reserve(tiles_phase56.size());
        bg_g_vals.reserve(tiles_phase56.size());
        bg_b_vals.reserve(tiles_phase56.size());
        m_vals.reserve(tiles_phase56.size());
        for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
          if (tile_valid_counts[ti] <= 0) {
            continue;
          }
          bg_r_vals.push_back(tile_norm_bg_r[ti]);
          bg_g_vals.push_back(tile_norm_bg_g[ti]);
          bg_b_vals.push_back(tile_norm_bg_b[ti]);
          m_vals.push_back(tile_norm_scale[ti]);
        }
        if (!bg_r_vals.empty() && !bg_g_vals.empty() && !bg_b_vals.empty() &&
            !m_vals.empty()) {
          const float bg_r = core::median_of(bg_r_vals);
          const float bg_g = core::median_of(bg_g_vals);
          const float bg_b = core::median_of(bg_b_vals);
          const float m_global = std::max(kEpsMedian, core::median_of(m_vals));
          for (int i = 0; i < recon_R.size(); ++i) {
            if (weight_sum.data()[i] > eps_ws) {
              recon_R.data()[i] = recon_R.data()[i] * m_global + bg_r;
              recon_G.data()[i] = recon_G.data()[i] * m_global + bg_g;
              recon_B.data()[i] = recon_B.data()[i] * m_global + bg_b;
            }
          }
        }
      }

      // Keep a luminance proxy for validation + downstream metrics.
      recon = 0.25f * recon_R + 0.5f * recon_G + 0.25f * recon_B;
    } else {
      if (!overlap_normalized_on_device) {
        for (int i = 0; i < recon.size(); ++i) {
          float ws = weight_sum.data()[i];
          if (ws > eps_ws) {
            recon.data()[i] /= ws;
          } else {
            // Mark canvas dead area with NaN sentinel (impossible sensor value)
            // so downstream logic can reject it robustly via std::isfinite().
            recon.data()[i] = invalid_ws;
          }
        }
      }

      if (apply_phase7_tile_norm) {
        std::vector<float> bg_vals;
        std::vector<float> m_vals;
        bg_vals.reserve(tiles_phase56.size());
        m_vals.reserve(tiles_phase56.size());
        for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
          if (tile_valid_counts[ti] > 0) {
            bg_vals.push_back(tile_norm_bg_r[ti]);
            m_vals.push_back(tile_norm_scale[ti]);
          }
        }
        if (!bg_vals.empty() && !m_vals.empty()) {
          const float bg = core::median_of(bg_vals);
          const float m_global = std::max(kEpsMedian, core::median_of(m_vals));
          for (int i = 0; i < recon.size(); ++i) {
            if (weight_sum.data()[i] > eps_ws) {
              recon.data()[i] = recon.data()[i] * m_global + bg;
            }
          }
        }
      }
    }

    // --- Memory release: weight_sum and first_img no longer needed ---
    weight_sum.resize(0, 0);
    first_img.resize(0, 0);

    const int valid_tile_count = std::count_if(
        tile_valid_counts.begin(), tile_valid_counts.end(),
        [&](int c) { return c >= min_valid_frames; });
    const int dead_tile_count = static_cast<int>(tiles_phase56.size()) -
                                std::count_if(tile_reconstructed_valid.begin(),
                                              tile_reconstructed_valid.end(),
                                              [&](uint8_t v) { return v != 0u; });
    const int full_support_tile_count = std::count_if(
        tile_valid_counts.begin(), tile_valid_counts.end(),
        [&](int c) { return c == static_cast<int>(frames.size()); });

    // Write reconstruction artifacts (v3)
    {
      core::json artifact;
      const bool write_full_tile_reconstruction_artifacts =
          tile_reconstruction_diagnostics_full;
      std::unordered_map<uint64_t,
                         reconstruction::TileWeightProfilePairDiagnostic>
          weight_profile_by_pair;
      auto pair_key = [](size_t lhs, size_t rhs) -> uint64_t {
        return (static_cast<uint64_t>(lhs) << 32) ^ static_cast<uint64_t>(rhs);
      };
      if (write_full_tile_reconstruction_artifacts) {
        weight_profile_by_pair.reserve(
            boundary_weight_profile_diagnostics.pair_diagnostics.size());
        for (const auto &pair :
             boundary_weight_profile_diagnostics.pair_diagnostics) {
          weight_profile_by_pair.emplace(pair_key(pair.lhs, pair.rhs), pair);
        }
      }
      auto append_boundary_pairs = [&](const char *key,
                                       const reconstruction::TileBoundaryDiagnostics
                                           &diagnostics) {
        artifact[key] = core::json::array();
        const size_t top_pair_count =
            std::min<size_t>(8, diagnostics.pair_diagnostics.size());
        for (size_t i = 0; i < top_pair_count; ++i) {
          const auto &pair = diagnostics.pair_diagnostics[i];
          core::json entry;
          entry["lhs_index"] = static_cast<int>(pair.lhs);
          entry["rhs_index"] = static_cast<int>(pair.rhs);
          entry["lhs_row"] = tiles_phase56[pair.lhs].row;
          entry["lhs_col"] = tiles_phase56[pair.lhs].col;
          entry["rhs_row"] = tiles_phase56[pair.rhs].row;
          entry["rhs_col"] = tiles_phase56[pair.rhs].col;
          entry["sample_count"] = static_cast<int>(pair.sample_count);
          entry["mean_abs_diff"] = pair.mean_abs_diff;
          entry["p95_abs_diff"] = pair.p95_abs_diff;
          entry["mean_signed_diff"] = pair.mean_signed_diff;
          entry["mean_abs_residual"] = pair.mean_abs_residual;
          entry["p95_abs_residual"] = pair.p95_abs_residual;
          entry["scale_ratio"] = pair.scale_ratio;
          entry["lhs_valid_count"] = tile_valid_counts[pair.lhs];
          entry["rhs_valid_count"] = tile_valid_counts[pair.rhs];
          entry["lhs_fallback_used"] = tile_fallback_used[pair.lhs] != 0u;
          entry["rhs_fallback_used"] = tile_fallback_used[pair.rhs] != 0u;
          entry["lhs_post_background"] = tile_post_background[pair.lhs];
          entry["rhs_post_background"] = tile_post_background[pair.rhs];
          entry["lhs_post_snr_proxy"] = tile_post_snr[pair.lhs];
          entry["rhs_post_snr_proxy"] = tile_post_snr[pair.rhs];
          entry["lhs_mean_correlation"] = tile_mean_correlations[pair.lhs];
          entry["rhs_mean_correlation"] = tile_mean_correlations[pair.rhs];
          const auto weight_it =
              weight_profile_by_pair.find(pair_key(pair.lhs, pair.rhs));
          if (weight_it != weight_profile_by_pair.end()) {
            const auto &weights = weight_it->second;
            entry["local_weight_usable_frame_count"] =
                static_cast<int>(weights.usable_frame_count);
            entry["local_weight_lhs_active_frame_count"] =
                static_cast<int>(weights.lhs_active_frame_count);
            entry["local_weight_rhs_active_frame_count"] =
                static_cast<int>(weights.rhs_active_frame_count);
            entry["local_weight_shared_active_frame_count"] =
                static_cast<int>(weights.shared_active_frame_count);
            entry["local_weight_activation_mismatch_count"] =
                static_cast<int>(weights.activation_mismatch_count);
            entry["local_weight_activation_mismatch_fraction"] =
                weights.usable_frame_count > 0u
                    ? static_cast<float>(weights.activation_mismatch_count) /
                          static_cast<float>(weights.usable_frame_count)
                    : 0.0f;
            entry["local_weight_mean_abs_delta"] = weights.mean_abs_delta;
            entry["local_weight_p95_abs_delta"] = weights.p95_abs_delta;
            entry["local_weight_correlation"] = weights.correlation;
          }
          artifact[key].push_back(std::move(entry));
        }
      };
      artifact["num_frames"] = static_cast<int>(frames.size());
      artifact["num_tiles"] = static_cast<int>(tiles_phase56.size());
      artifact["valid_tiles"] = valid_tile_count;
      artifact["dead_tiles"] = dead_tile_count;
      artifact["tile_boundary_diagnostics_enabled"] =
          tile_reconstruction_diagnostics_enabled;
      artifact["full_support_tiles"] = full_support_tile_count;
      artifact["tile_reconstruction_diagnostics_mode"] =
          tile_reconstruction_diagnostics_mode;
      artifact["tile_norm_application"] = tile_norm_application;
      artifact["tile_boundary_analysis_uses_common_canvas_mask"] =
          tile_reconstruction_diagnostics_enabled;
      artifact["tile_boundary_analysis_input"] =
          tile_reconstruction_diagnostics_enabled ? "pre_ola_raw" : "disabled";
      artifact["tile_boundary_raw_pair_count"] =
          static_cast<int>(boundary_diagnostics_raw.pair_count);
      artifact["tile_boundary_raw_observation_count"] =
          static_cast<int>(boundary_diagnostics_raw.observed_pair_count);
      artifact["tile_boundary_raw_sample_count"] =
          static_cast<int>(boundary_diagnostics_raw.sample_count);
      artifact["tile_boundary_raw_pair_mean_abs_diff_mean"] =
          boundary_diagnostics_raw.pair_mean_abs_diff_mean;
      artifact["tile_boundary_raw_pair_mean_abs_diff_p95"] =
          boundary_diagnostics_raw.pair_mean_abs_diff_p95;
      artifact["tile_boundary_raw_pair_p95_abs_diff_mean"] =
          boundary_diagnostics_raw.pair_p95_abs_diff_mean;
      artifact["tile_boundary_raw_pair_p95_abs_diff_p95"] =
          boundary_diagnostics_raw.pair_p95_abs_diff_p95;
      artifact["tile_boundary_raw_pair_mean_signed_diff_mean_abs"] =
          boundary_diagnostics_raw.pair_mean_signed_diff_mean_abs;
      artifact["tile_boundary_raw_pair_mean_abs_residual_mean"] =
          boundary_diagnostics_raw.pair_mean_abs_residual_mean;
      artifact["tile_boundary_raw_pair_mean_abs_residual_p95"] =
          boundary_diagnostics_raw.pair_mean_abs_residual_p95;
      artifact["tile_boundary_raw_pair_p95_abs_residual_mean"] =
          boundary_diagnostics_raw.pair_p95_abs_residual_mean;
      artifact["tile_boundary_raw_pair_p95_abs_residual_p95"] =
          boundary_diagnostics_raw.pair_p95_abs_residual_p95;
      artifact["tile_boundary_raw_pair_scale_ratio_deviation_mean"] =
          boundary_diagnostics_raw.pair_scale_ratio_deviation_mean;
      artifact["tile_boundary_raw_pair_scale_ratio_deviation_p95"] =
          boundary_diagnostics_raw.pair_scale_ratio_deviation_p95;
      artifact["tile_boundary_pair_count"] =
          static_cast<int>(boundary_diagnostics_raw.pair_count);
      artifact["tile_boundary_observation_count"] =
          static_cast<int>(boundary_diagnostics_raw.observed_pair_count);
      artifact["tile_boundary_sample_count"] =
          static_cast<int>(boundary_diagnostics_raw.sample_count);
      artifact["tile_boundary_pair_mean_abs_diff_mean"] =
          boundary_diagnostics_raw.pair_mean_abs_diff_mean;
      artifact["tile_boundary_pair_mean_abs_diff_p95"] =
          boundary_diagnostics_raw.pair_mean_abs_diff_p95;
      artifact["tile_boundary_pair_p95_abs_diff_mean"] =
          boundary_diagnostics_raw.pair_p95_abs_diff_mean;
      artifact["tile_boundary_pair_p95_abs_diff_p95"] =
          boundary_diagnostics_raw.pair_p95_abs_diff_p95;
      artifact["tile_boundary_pair_mean_signed_diff_mean_abs"] =
          boundary_diagnostics_raw.pair_mean_signed_diff_mean_abs;
      artifact["tile_boundary_pair_mean_abs_residual_mean"] =
          boundary_diagnostics_raw.pair_mean_abs_residual_mean;
      artifact["tile_boundary_pair_mean_abs_residual_p95"] =
          boundary_diagnostics_raw.pair_mean_abs_residual_p95;
      artifact["tile_boundary_pair_p95_abs_residual_mean"] =
          boundary_diagnostics_raw.pair_p95_abs_residual_mean;
      artifact["tile_boundary_pair_p95_abs_residual_p95"] =
          boundary_diagnostics_raw.pair_p95_abs_residual_p95;
      artifact["tile_boundary_pair_scale_ratio_deviation_mean"] =
          boundary_diagnostics_raw.pair_scale_ratio_deviation_mean;
      artifact["tile_boundary_pair_scale_ratio_deviation_p95"] =
          boundary_diagnostics_raw.pair_scale_ratio_deviation_p95;
      artifact["tile_boundary_valid_count_delta_mean_abs"] =
          boundary_valid_count_delta_mean_abs;
      artifact["tile_boundary_valid_count_delta_p95_abs"] =
          boundary_valid_count_delta_p95_abs;
      artifact["tile_boundary_post_background_delta_mean_abs"] =
          boundary_post_background_delta_mean_abs;
      artifact["tile_boundary_post_background_delta_p95_abs"] =
          boundary_post_background_delta_p95_abs;
      artifact["tile_boundary_fallback_mismatch_count"] =
          boundary_fallback_mismatch_count;
      artifact["common_overlap_source"] = "prewarp_inline_coverage";
      artifact["acceleration"] =
          core::acceleration_selection_to_json(
              tile_reconstruction_acceleration);
      artifact["device_frame_batch"] =
          core::device_frame_batch_to_json(tile_reconstruction_frame_batch);
      artifact["device_tile_batch"] =
          core::device_tile_batch_to_json(tile_reconstruction_tile_batch);
      if (write_full_tile_reconstruction_artifacts) {
        artifact["tile_valid_counts"] = core::json::array();
        artifact["tile_fallback_used"] = core::json::array();
        artifact["tile_mean_correlations"] = core::json::array();
        artifact["tile_post_contrast"] = core::json::array();
        artifact["tile_post_background"] = core::json::array();
        artifact["tile_post_snr_proxy"] = core::json::array();
        artifact["tile_boundary_post_snr_delta_mean_abs"] =
            boundary_post_snr_delta_mean_abs;
        artifact["tile_boundary_post_snr_delta_p95_abs"] =
            boundary_post_snr_delta_p95_abs;
        artifact["tile_boundary_mean_correlation_delta_mean_abs"] =
            boundary_mean_correlation_delta_mean_abs;
        artifact["tile_boundary_mean_correlation_delta_p95_abs"] =
            boundary_mean_correlation_delta_p95_abs;
        artifact["tile_boundary_local_weight_observation_count"] =
            static_cast<int>(
                boundary_weight_profile_diagnostics.observed_pair_count);
        artifact["tile_boundary_local_weight_mean_abs_delta_mean"] =
            boundary_weight_profile_diagnostics.pair_mean_abs_delta_mean;
        artifact["tile_boundary_local_weight_mean_abs_delta_p95"] =
            boundary_weight_profile_diagnostics.pair_mean_abs_delta_p95;
        artifact["tile_boundary_local_weight_p95_abs_delta_mean"] =
            boundary_weight_profile_diagnostics.pair_p95_abs_delta_mean;
        artifact["tile_boundary_local_weight_p95_abs_delta_p95"] =
            boundary_weight_profile_diagnostics.pair_p95_abs_delta_p95;
        artifact["tile_boundary_local_weight_activation_mismatch_fraction_mean"] =
            boundary_weight_profile_diagnostics
                .pair_activation_mismatch_fraction_mean;
        artifact["tile_boundary_local_weight_activation_mismatch_fraction_p95"] =
            boundary_weight_profile_diagnostics
                .pair_activation_mismatch_fraction_p95;
        artifact["tile_boundary_local_weight_correlation_mean"] =
            boundary_weight_profile_diagnostics.pair_correlation_mean;
        artifact["tile_boundary_local_weight_correlation_p05"] =
            boundary_weight_profile_diagnostics.pair_correlation_p05;
        for (size_t i = 0; i < tiles_phase56.size(); ++i) {
          artifact["tile_valid_counts"].push_back(tile_valid_counts[i]);
          artifact["tile_fallback_used"].push_back(
              tile_fallback_used[i] != 0u);
          artifact["tile_mean_correlations"].push_back(
              tile_mean_correlations[i]);
          artifact["tile_post_contrast"].push_back(tile_post_contrast[i]);
          artifact["tile_post_background"].push_back(tile_post_background[i]);
          artifact["tile_post_snr_proxy"].push_back(tile_post_snr[i]);
        }
        append_boundary_pairs("tile_boundary_raw_top_pairs",
                              boundary_diagnostics_raw);
        artifact["tile_boundary_top_pairs"] =
            artifact["tile_boundary_raw_top_pairs"];
      }
      core::write_text(run_dir / "artifacts" / "tile_reconstruction.json",
                       artifact.dump(2));
    }

    synthetic_weighting_decision = runner::decide_synthetic_weighting(
        cfg.synthetic.weighting,
        static_cast<int>(boundary_diagnostics_raw.observed_pair_count),
        boundary_diagnostics_raw.pair_mean_abs_diff_p95,
        boundary_diagnostics_raw.pair_scale_ratio_deviation_p95,
        boundary_post_background_delta_p95_abs,
        boundary_weight_profile_diagnostics.pair_mean_abs_delta_p95,
        boundary_weight_profile_diagnostics.pair_correlation_p05);
    if (synthetic_weighting_decision.tile_seam_guard_triggered) {
      std::ostringstream msg;
      msg << "SYNTHETIC_FRAMES seam guard: fallback tile_weighted -> global"
          << " (pairs=" << synthetic_weighting_decision.boundary_pair_count
          << ", boundary_mean_abs_diff_p95="
          << synthetic_weighting_decision.boundary_pair_mean_abs_diff_p95
          << ", boundary_scale_ratio_dev_p95="
          << synthetic_weighting_decision
                 .boundary_pair_scale_ratio_deviation_p95
          << ", boundary_post_background_delta_p95_abs="
          << synthetic_weighting_decision.boundary_post_background_delta_p95_abs
          << ", local_weight_delta_p95="
          << synthetic_weighting_decision.local_weight_mean_abs_delta_p95
          << ", local_weight_corr_p05="
          << synthetic_weighting_decision.local_weight_correlation_p05 << ")";
      emitter.warning(run_id, msg.str(), log_file);
    }
    emitter.phase_end(
        run_id, reconstruction_phase, "ok",
        {
            {"duration_s",
             std::chrono::duration<double>(
                 std::chrono::steady_clock::now() -
                 tile_reconstruction_started_at).count()},
            {"output", (run_dir / "outputs" / "reconstructed_L.fit").string()},
            {"valid_tiles", valid_tile_count},
            {"fallback_tiles",
             std::count_if(tile_fallback_used.begin(), tile_fallback_used.end(),
                           [&](uint8_t v) { return v != 0u; })},
            {"tile_norm_global_scale", tile_norm_guard_summary.global_scale},
            {"tile_norm_guard_clamped_low_scale_count",
             static_cast<int>(tile_norm_guard_summary.clamped_low_scale_count)},
            {"tile_norm_guard_used_global_scale_count",
             static_cast<int>(tile_norm_guard_summary.used_global_scale_count)},
            {"tile_norm_application", tile_norm_application},
            {"tile_boundary_analysis_input",
             tile_reconstruction_diagnostics_enabled ? "pre_ola_raw" : "disabled"},
            {"tile_boundary_pairs",
             static_cast<int>(boundary_diagnostics_raw.observed_pair_count)},
            {"tile_boundary_raw_pair_mean_abs_diff_p95",
             boundary_diagnostics_raw.pair_mean_abs_diff_p95},
            {"tile_boundary_pair_mean_abs_diff_p95",
             boundary_diagnostics_raw.pair_mean_abs_diff_p95},
            {"tile_boundary_post_background_delta_p95_abs",
             boundary_post_background_delta_p95_abs},
            {"tile_boundary_local_weight_mean_abs_delta_p95",
             boundary_weight_profile_diagnostics.pair_mean_abs_delta_p95},
            {"tile_boundary_local_weight_correlation_p05",
             boundary_weight_profile_diagnostics.pair_correlation_p05},
            {"common_overlap_source", "prewarp_inline_coverage"},
            {"acceleration",
             core::acceleration_selection_to_json(
                 tile_reconstruction_acceleration)},
            {"device_frame_batch",
             core::device_frame_batch_to_json(
                 tile_reconstruction_frame_batch)},
            {"device_tile_batch",
             core::device_tile_batch_to_json(tile_reconstruction_tile_batch)},
            {"tile_boundary_fallback_mismatch_count",
             boundary_fallback_mismatch_count},
        },
        log_file);
    if (abort_if_runtime_limit_exceeded("TILE_RECONSTRUCTION")) {
      return 1;
    }
    }

    // Phase 7: STATE_CLUSTERING (// Methodik v3 §10)
    // AQMH uses independent per-pixel reconstruction; clustering and
    // synthetic frames are classic-only and produce no meaningful output.
    bool use_synthetic_frames = true;
    std::string synthetic_skip_reason;
    float synthetic_skip_weight_spread = 0.0f;
    float synthetic_skip_quality_spread = 0.0f;
    int synthetic_skip_eligible_clusters = 0;
    std::vector<int> cluster_labels(static_cast<size_t>(frames.size()), 0);
    int n_clusters = 1;
    const bool skip_clustering_for_aqmh = cfg.aqmh.enabled;
    const bool skip_synthetic_for_aqmh = cfg.aqmh.enabled;
    if (!skip_clustering_for_aqmh) {
      emitter.phase_start(run_id, Phase::STATE_CLUSTERING, "STATE_CLUSTERING",
                          log_file);
    }
    if (skip_clustering_for_aqmh) {
      use_synthetic_frames = false;
      synthetic_skip_reason = "aqmh_independent_reconstruction";
    } else if (skip_clustering_in_reduced) {
      use_synthetic_frames = false;
      synthetic_skip_reason = emergency_mode ? "emergency_mode"
                                             : "reduced_mode";
      emitter.phase_end(run_id, Phase::STATE_CLUSTERING, "skipped",
                        {{"reason", synthetic_skip_reason},
                         {"usable_frame_count", n_usable_frames},
                         {"frames_reduced_threshold",
                          cfg.assumptions.frames_reduced_threshold},
                         {"emergency_mode", emergency_mode}},
                        log_file);
    }

    if (!skip_clustering_in_reduced && !skip_clustering_for_aqmh) {
      // Build state vectors for clustering (v3.2 core vector):
      // [G_f, mean_local_quality, var_local_quality, B_f, sigma_f]
      const int n_frames_cluster = static_cast<int>(frames.size());
      std::vector<std::vector<float>> state_vectors(
          static_cast<size_t>(n_frames_cluster));

      std::vector<float> G_for_cluster(static_cast<size_t>(n_frames_cluster),
                                       1.0f);

      for (size_t fi = 0; fi < frames.size(); ++fi) {
        float G_f = (fi < static_cast<size_t>(global_weights.size()))
                        ? global_weights[static_cast<int>(fi)]
                        : 1.0f;
        float bg =
            (fi < frame_metrics.size()) ? frame_metrics[fi].background : 0.0f;
        float noise =
            (fi < frame_metrics.size()) ? frame_metrics[fi].noise : 0.0f;

        // Compute mean/var of local tile quality for this frame
        float mean_local = 0.0f, var_local = 0.0f;
        if (fi < local_metrics.size() && !local_metrics[fi].empty()) {
          for (const auto &tm : local_metrics[fi]) {
            mean_local += tm.quality_score;
          }
          mean_local /= static_cast<float>(local_metrics[fi].size());
          for (const auto &tm : local_metrics[fi]) {
            float diff = tm.quality_score - mean_local;
            var_local += diff * diff;
          }
          var_local /= static_cast<float>(local_metrics[fi].size());
        }
        state_vectors[fi] = {G_f, mean_local, var_local, bg, noise};
        G_for_cluster[fi] = G_f;
      }

      std::vector<std::vector<float>> X = state_vectors;
      std::vector<float> state_means;
      std::vector<float> state_stds;
      std::vector<std::string> final_feature_list = {
          "global_weight",
          "mean_local_quality",
          "var_local_quality",
          "background",
          "noise"};
      if (n_frames_cluster > 0) {
        const size_t D = X[0].size();
        state_means.assign(D, 0.0f);
        state_stds.assign(D, 0.0f);

        for (size_t d = 0; d < D; ++d) {
          double sum = 0.0;
          for (size_t i = 0; i < X.size(); ++i)
            sum += static_cast<double>(X[i][d]);
          state_means[d] = static_cast<float>(sum / static_cast<double>(X.size()));
          double var = 0.0;
          for (size_t i = 0; i < X.size(); ++i) {
            double diff =
                static_cast<double>(X[i][d]) - static_cast<double>(state_means[d]);
            var += diff * diff;
          }
          var /= std::max<double>(1.0, static_cast<double>(X.size()));
          state_stds[d] = static_cast<float>(std::sqrt(std::max(0.0, var)));
        }

        const float eps = kEpsWeight;
        for (size_t i = 0; i < X.size(); ++i) {
          for (size_t d = 0; d < D; ++d) {
            float sd = state_stds[d];
            X[i][d] = (sd > eps) ? ((X[i][d] - state_means[d]) / sd) : 0.0f;
          }
        }
      }

      // Determine cluster count: K = clip(floor(N/10), K_min, K_max)
      int k_min = cfg.synthetic.clustering.cluster_count_range[0];
      int k_max = cfg.synthetic.clustering.cluster_count_range[1];
      if (reduced_mode) {
        k_min = cfg.assumptions.reduced_mode_cluster_range[0];
        k_max = cfg.assumptions.reduced_mode_cluster_range[1];
      }
      int k_default = std::max(k_min, std::min(k_max, n_frames_cluster / 10));

      // Simple k-means clustering
      n_clusters = std::min(k_default, n_frames_cluster);

      std::string clustering_method = cfg.synthetic.clustering.mode;
      auto assign_quantile_clusters = [&]() {
        std::vector<std::pair<float, int>> order;
        order.reserve(G_for_cluster.size());
        for (size_t i = 0; i < G_for_cluster.size(); ++i) {
          order.push_back({G_for_cluster[i], static_cast<int>(i)});
        }
        std::sort(
            order.begin(), order.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
        for (size_t r = 0; r < order.size(); ++r) {
          int label = static_cast<int>((r * static_cast<size_t>(n_clusters)) /
                                       std::max<size_t>(1, order.size()));
          if (label >= n_clusters)
            label = n_clusters - 1;
          cluster_labels[static_cast<size_t>(order[r].second)] = label;
        }
      };

      if (n_clusters > 1 && n_frames_cluster > 1) {
        if (cfg.synthetic.clustering.mode == "quantile") {
          assign_quantile_clusters();
        } else {
          // K-means++ initialization: pick first center uniformly at random,
          // then each subsequent center with probability proportional to D(x)²
          // (squared distance to nearest existing center).
          std::mt19937 rng(42); // fixed seed for reproducibility
          std::vector<std::vector<float>> centers;
          centers.reserve(static_cast<size_t>(n_clusters));

          // First center: pick middle frame (deterministic, reproducible)
          centers.push_back(X[static_cast<size_t>(n_frames_cluster / 2)]);

          std::vector<double> min_dist_sq(X.size(),
                                          std::numeric_limits<double>::max());
          for (int c = 1; c < n_clusters; ++c) {
            // Update min distances to nearest center (only need to check latest)
            const auto &last_center = centers.back();
            for (size_t fi = 0; fi < X.size(); ++fi) {
              double d2 = 0.0;
              for (size_t d = 0; d < X[fi].size(); ++d) {
                double diff = static_cast<double>(X[fi][d]) -
                              static_cast<double>(last_center[d]);
                d2 += diff * diff;
              }
              if (d2 < min_dist_sq[fi])
                min_dist_sq[fi] = d2;
            }
            // Sample next center with probability proportional to D(x)²
            std::discrete_distribution<size_t> dist(min_dist_sq.begin(),
                                                    min_dist_sq.end());
            size_t next = dist(rng);
            centers.push_back(X[next]);
          }

          // K-means iterations
          for (int iter = 0; iter < 20; ++iter) {
            // Assign labels
            for (size_t fi = 0; fi < X.size(); ++fi) {
              float best_dist = std::numeric_limits<float>::max();
              int best_c = 0;
              for (int c = 0; c < n_clusters; ++c) {
                float dist = 0.0f;
                for (size_t d = 0; d < X[fi].size(); ++d) {
                  float diff = X[fi][d] - centers[static_cast<size_t>(c)][d];
                  dist += diff * diff;
                }
                if (dist < best_dist) {
                  best_dist = dist;
                  best_c = c;
                }
              }
              cluster_labels[fi] = best_c;
            }

            // Update centers
            std::vector<std::vector<float>> new_centers(
                static_cast<size_t>(n_clusters),
                std::vector<float>(X[0].size(), 0.0f));
            std::vector<int> counts(static_cast<size_t>(n_clusters), 0);
            for (size_t fi = 0; fi < X.size(); ++fi) {
              int c = cluster_labels[fi];
              for (size_t d = 0; d < X[fi].size(); ++d) {
                new_centers[static_cast<size_t>(c)][d] += X[fi][d];
              }
              counts[static_cast<size_t>(c)]++;
            }
            for (int c = 0; c < n_clusters; ++c) {
              if (counts[static_cast<size_t>(c)] > 0) {
                for (size_t d = 0;
                     d < new_centers[static_cast<size_t>(c)].size(); ++d) {
                  new_centers[static_cast<size_t>(c)][d] /=
                      static_cast<float>(counts[static_cast<size_t>(c)]);
                }
              }
            }
            centers = new_centers;
          }
        }
      }

      {
        std::vector<int> counts(static_cast<size_t>(n_clusters), 0);
        for (int lbl : cluster_labels) {
          if (lbl >= 0 && lbl < n_clusters)
            counts[static_cast<size_t>(lbl)]++;
        }

        bool degenerate = false;
        for (int c = 0; c < n_clusters; ++c) {
          if (counts[static_cast<size_t>(c)] <= 0) {
            degenerate = true;
            break;
          }
        }

        if (degenerate && n_clusters > 1) {
          clustering_method = "quantile";
          assign_quantile_clusters();
        }
      }

      {
        core::json artifact;
        artifact["n_clusters"] = n_clusters;
        artifact["k_min"] = k_min;
        artifact["k_max"] = k_max;
        artifact["method"] = clustering_method;
        artifact["feature_names"] = core::json::array();
        for (const auto &name : final_feature_list)
          artifact["feature_names"].push_back(name);
        artifact["standardization"] = {
            {"method", "zscore"},
            {"eps", kEpsWeight},
            {"means", core::json::array()},
            {"stds", core::json::array()},
        };
        for (float v : state_means)
          artifact["standardization"]["means"].push_back(v);
        for (float v : state_stds)
          artifact["standardization"]["stds"].push_back(v);
        artifact["state_vectors_raw"] = core::json::array();
        artifact["state_vectors_standardized"] = core::json::array();
        for (size_t i = 0; i < state_vectors.size(); ++i) {
          core::json raw = core::json::array();
          core::json stdv = core::json::array();
          for (float v : state_vectors[i])
            raw.push_back(v);
          for (float v : X[i])
            stdv.push_back(v);
          artifact["state_vectors_raw"].push_back(std::move(raw));
          artifact["state_vectors_standardized"].push_back(std::move(stdv));
        }
        artifact["cluster_labels"] = core::json::array();
        for (int lbl : cluster_labels)
          artifact["cluster_labels"].push_back(lbl);
        artifact["cluster_sizes"] = core::json::array();
        for (int c = 0; c < n_clusters; ++c) {
          int count = static_cast<int>(
              std::count(cluster_labels.begin(), cluster_labels.end(), c));
          artifact["cluster_sizes"].push_back(count);
        }
        core::write_text(run_dir / "artifacts" / "state_clustering.json",
                         artifact.dump(2));
      }

      emitter.phase_end(run_id, Phase::STATE_CLUSTERING, "ok",
                        {{"n_clusters", n_clusters}}, log_file);
    }
    if (abort_if_runtime_limit_exceeded("STATE_CLUSTERING")) {
      return 1;
    }

    // Aggregate tile metrics over frames for downstream BGE/PCC usage.
    if (!local_metrics.empty()) {
      const size_t n_tiles = local_metrics.front().size();
      const bool consistent = std::all_of(
          local_metrics.begin(), local_metrics.end(),
          [n_tiles](const auto &fm) { return fm.size() == n_tiles; });
      if (!consistent || n_tiles == 0) {
        bge_tile_metrics_cache = local_metrics.front();
      } else {
        auto median_or_zero = [](std::vector<float> vals) -> float {
          if (vals.empty()) return 0.0f;
          return core::median_of(vals);
        };

        bge_tile_metrics_cache.assign(n_tiles, TileMetrics{});
        for (size_t ti = 0; ti < n_tiles; ++ti) {
          std::vector<float> fwhm_vals;
          std::vector<float> round_vals;
          std::vector<float> contrast_vals;
          std::vector<float> sharp_vals;
          std::vector<float> bg_vals;
          std::vector<float> noise_vals;
          std::vector<float> grad_vals;
          std::vector<float> q_vals;
          std::vector<float> star_count_vals;
          int star_votes = 0;
          int structure_votes = 0;

          fwhm_vals.reserve(local_metrics.size());
          round_vals.reserve(local_metrics.size());
          contrast_vals.reserve(local_metrics.size());
          sharp_vals.reserve(local_metrics.size());
          bg_vals.reserve(local_metrics.size());
          noise_vals.reserve(local_metrics.size());
          grad_vals.reserve(local_metrics.size());
          q_vals.reserve(local_metrics.size());
          star_count_vals.reserve(local_metrics.size());

          for (const auto &fm : local_metrics) {
            const auto &tm = fm[ti];
            if (std::isfinite(tm.fwhm)) fwhm_vals.push_back(tm.fwhm);
            if (std::isfinite(tm.roundness)) round_vals.push_back(tm.roundness);
            if (std::isfinite(tm.contrast)) contrast_vals.push_back(tm.contrast);
            if (std::isfinite(tm.sharpness)) sharp_vals.push_back(tm.sharpness);
            if (std::isfinite(tm.background)) bg_vals.push_back(tm.background);
            if (std::isfinite(tm.noise)) noise_vals.push_back(tm.noise);
            if (std::isfinite(tm.gradient_energy)) grad_vals.push_back(tm.gradient_energy);
            if (std::isfinite(tm.quality_score)) q_vals.push_back(tm.quality_score);
            star_count_vals.push_back(static_cast<float>(tm.star_count));
            if (tm.type == TileType::STAR) {
              ++star_votes;
            } else {
              ++structure_votes;
            }
          }

          TileMetrics agg{};
          agg.fwhm = median_or_zero(std::move(fwhm_vals));
          agg.roundness = median_or_zero(std::move(round_vals));
          agg.contrast = median_or_zero(std::move(contrast_vals));
          agg.sharpness = median_or_zero(std::move(sharp_vals));
          agg.background = median_or_zero(std::move(bg_vals));
          agg.noise = median_or_zero(std::move(noise_vals));
          agg.gradient_energy = median_or_zero(std::move(grad_vals));
          agg.quality_score = median_or_zero(std::move(q_vals));
          agg.star_count = static_cast<int>(
              std::lround(median_or_zero(std::move(star_count_vals))));
          agg.type = (star_votes >= structure_votes) ? TileType::STAR
                                                     : TileType::STRUCTURE;
          bge_tile_metrics_cache[ti] = agg;
        }
      }
    } else {
      bge_tile_metrics_cache.clear();
    }

    // --- Memory release: local_metrics no longer needed after clustering ---
    { std::vector<std::vector<TileMetrics>>().swap(local_metrics); }

    // Phase 8: SYNTHETIC_FRAMES (// Methodik v3 §11)
    // AQMH uses independent per-pixel reconstruction; synthetic frames are
    // classic-only and produce no output here.
    if (!skip_synthetic_for_aqmh) {
      emitter.phase_start(run_id, Phase::SYNTHETIC_FRAMES, "SYNTHETIC_FRAMES",
                          log_file);
    }

    struct RGBFrame {
      Matrix2Df R;
      Matrix2Df G;
      Matrix2Df B;
    };

    std::vector<Matrix2Df> synthetic_frames;
    std::vector<RGBFrame> synthetic_rgb_frames;
    std::vector<float> synthetic_cluster_quality;
    std::vector<float> synthetic_cluster_mass;

    auto reconstruct_subset =
        [&](const std::vector<char> &frame_mask) -> Matrix2Df {
      if (synthetic_weighting_decision.effective_weighting == "tile_weighted") {
        Matrix2Df out = Matrix2Df::Zero(canvas_height, canvas_width);
        Matrix2Df weight_ola = Matrix2Df::Zero(canvas_height, canvas_width);
        std::atomic<bool> any_tile{false};
        const int recon_rows = std::max(1, canvas_height);
        const size_t recon_stripe_count =
            static_cast<size_t>(std::max(1, std::min<int>(
                cfg.runtime_limits.parallel_workers, recon_rows)));
        std::vector<std::mutex> recon_stripe_mutexes(recon_stripe_count);
        auto recon_stripe_for_row = [&](int y) -> size_t {
          if (recon_rows <= 1 || recon_stripe_mutexes.size() <= 1) {
            return 0;
          }
          const int y_clamped = std::clamp(y, 0, recon_rows - 1);
          const size_t num =
              static_cast<size_t>(y_clamped) * recon_stripe_mutexes.size();
          const size_t den = static_cast<size_t>(recon_rows);
          return std::min(recon_stripe_mutexes.size() - 1, num / den);
        };

        std::atomic<size_t> next_tile{0};

        int subset_workers = 1;
        if (tiles_phase56.size() > 1) {
          int cpu_cores = static_cast<int>(std::thread::hardware_concurrency());
          if (cpu_cores <= 0)
            cpu_cores = 1;
          subset_workers = std::min<int>(cfg.runtime_limits.parallel_workers,
                                         cpu_cores);
          subset_workers =
              std::min<int>(subset_workers, static_cast<int>(tiles_phase56.size()));
          subset_workers = std::max(1, subset_workers);
        }
        core::WorkerCudaStreams subset_streams(
            tile_reconstruction_acceleration.selected ==
                core::AccelerationBackend::opencv_cuda,
            static_cast<size_t>(subset_workers));

        auto process_tile = [&](int worker_index) {
          std::vector<Matrix2Df> cluster_tiles;
          std::vector<float> cluster_weights;
          cluster_tiles.reserve(frame_mask.size());
          cluster_weights.reserve(frame_mask.size());
          while (true) {
            const size_t ti = next_tile.fetch_add(1);
            if (ti >= tiles_phase56.size())
              break;

            const Tile &t = tiles_phase56[ti];
            cluster_tiles.clear();
            cluster_weights.clear();

            for (size_t fi = 0; fi < frame_mask.size() && fi < frames.size(); ++fi) {
              if (!frame_mask[fi] || !frame_has_data[fi])
                continue;
              if (ti >= tile_common_valid.size() || tile_common_valid[ti] == 0)
                continue;
              float G_f = (fi < static_cast<size_t>(global_weights.size()))
                              ? global_weights[static_cast<int>(fi)]
                              : 1.0f;
              float L_ft =
                  (fi < local_weights.size() && ti < local_weights[fi].size())
                      ? local_weights[fi][ti]
                      : 1.0f;
              float w = G_f * L_ft;
              if (!(std::isfinite(w) && w > 0.0f))
                continue;
              cluster_tiles.emplace_back();
              Matrix2Df &tile_img = cluster_tiles.back();
              if (!prewarped_frames.extract_tile_into(fi, t, tile_img, 0, 0) ||
                  tile_img.rows() != t.height || tile_img.cols() != t.width) {
                cluster_tiles.pop_back();
                continue;
              }
              if (!tile_compile::runner::
                      apply_common_overlap_to_tile_inplace_and_check_nonzero(
                          tile_img, t, common_valid_mask, canvas_width,
                          canvas_height)) {
                cluster_tiles.pop_back();
                continue;
              }
              cluster_weights.push_back(w);
            }

            if (cluster_tiles.empty())
              continue;

            Matrix2Df tile_rec;
            if (cluster_tiles.size() == 1) {
              tile_rec = std::move(cluster_tiles.front());
            } else {
              auto wr = tile_reconstruction_ops.sigma_clip_reduce(
                  cluster_tiles, cluster_weights,
                  cfg.stacking.sigma_clip.sigma_low,
                  cfg.stacking.sigma_clip.sigma_high,
                  cfg.stacking.sigma_clip.max_iters,
                  cfg.stacking.sigma_clip.min_fraction, kEpsWeight,
                  subset_streams.get(static_cast<size_t>(worker_index)));
              tile_rec = std::move(wr.tile);
            }
            if (tile_rec.rows() != t.height || tile_rec.cols() != t.width)
              continue;

            if (ti >= tile_window_cache.size())
              continue;
            const std::vector<float> &window_x = tile_window_cache[ti].x;
            const std::vector<float> &window_y = tile_window_cache[ti].y;

            const int x0 = std::max(0, t.x);
            const int y0 = std::max(0, t.y);

            for (int yy = 0; yy < tile_rec.rows(); ++yy) {
              const int iy = y0 + yy;
              if (iy < 0 || iy >= out.rows())
                continue;
              const size_t stripe = recon_stripe_for_row(iy);
              std::lock_guard<std::mutex> lock(recon_stripe_mutexes[stripe]);
              for (int xx = 0; xx < tile_rec.cols(); ++xx) {
                const int ix = x0 + xx;
                if (ix < 0 || ix >= out.cols())
                  continue;
                const size_t common_idx =
                    static_cast<size_t>(iy) * static_cast<size_t>(canvas_width) +
                    static_cast<size_t>(ix);
                if (common_idx >= common_valid_mask.size() ||
                    common_valid_mask[common_idx] == 0) {
                  continue;
                }
                const float tile_value = tile_rec(yy, xx);
                if (!std::isfinite(tile_value)) {
                  continue;
                }
                const float win = window_y[static_cast<size_t>(yy)] *
                                  window_x[static_cast<size_t>(xx)];
                out(iy, ix) += tile_value * win;
                weight_ola(iy, ix) += win;
              }
            }
            any_tile.store(true, std::memory_order_relaxed);
          }
        };

        if (subset_workers > 1) {
          std::vector<std::thread> workers;
          workers.reserve(static_cast<size_t>(subset_workers));
          for (int w = 0; w < subset_workers; ++w) {
            workers.emplace_back(process_tile, w);
          }
          for (auto &worker : workers) {
            if (worker.joinable())
              worker.join();
          }
        } else {
          process_tile(0);
        }

        if (!any_tile.load(std::memory_order_relaxed))
          return Matrix2Df();

        for (Eigen::Index i = 0; i < out.size(); ++i) {
          float ws = weight_ola.data()[i];
          if (ws > kEpsWeightSum) {
            out.data()[i] /= ws;
          } else {
            out.data()[i] = std::numeric_limits<float>::quiet_NaN();
          }
        }
        return out;
      }

      // Accumulate weighted sum directly to avoid copying full-res frames.
      Matrix2Df out;
      float wsum = 0.0f;

      for (size_t fi = 0; fi < frame_mask.size() && fi < frames.size(); ++fi) {
        if (!frame_mask[fi] || !frame_has_data[fi])
          continue;
        Matrix2Df src = prewarped_frames.load(fi);
        if (src.size() <= 0)
          continue;
        float w = (fi < static_cast<size_t>(global_weights.size()))
                      ? global_weights[static_cast<int>(fi)]
                      : 1.0f;
        if (out.size() == 0) {
          out = Matrix2Df::Zero(src.rows(), src.cols());
        }
        const size_t px_count = static_cast<size_t>(std::max<Eigen::Index>(0, src.size()));
        for (size_t i = 0; i < px_count && i < common_valid_mask.size(); ++i) {
          if (common_valid_mask[i] != 0 &&
              std::isfinite(src.data()[static_cast<Eigen::Index>(i)])) {
            out.data()[static_cast<Eigen::Index>(i)] +=
                src.data()[static_cast<Eigen::Index>(i)] * w;
          }
        }
        wsum += w;
      }

      if (out.size() == 0)
        return Matrix2Df();
      if (wsum > kEpsWeight)
        out /= wsum;
      return out;
    };
    int synth_min = cfg.synthetic.frames_min;
    int synth_max = cfg.synthetic.frames_max;

    if (!use_synthetic_frames && !skip_synthetic_for_aqmh) {
      core::json extra;
      if (!synthetic_skip_reason.empty()) {
        extra["reason"] = synthetic_skip_reason;
      } else {
        extra["reason"] = emergency_mode ? "emergency_mode"
                                          : "reduced_mode";
      }
      if (synthetic_skip_eligible_clusters > 0) {
        extra["eligible_clusters"] = synthetic_skip_eligible_clusters;
        extra["weight_spread"] = synthetic_skip_weight_spread;
        extra["quality_spread"] = synthetic_skip_quality_spread;
      }
      extra["usable_frame_count"] = n_usable_frames;
      extra["frames_reduced_threshold"] =
          cfg.assumptions.frames_reduced_threshold;
      extra["emergency_mode"] = emergency_mode;
      emitter.phase_end(run_id, Phase::SYNTHETIC_FRAMES, "skipped", extra,
                        log_file);
    } else if (use_synthetic_frames) {
      std::vector<int> cluster_sizes(n_clusters, 0);
      for (size_t fi = 0; fi < frames.size(); ++fi) {
        int c = cluster_labels[fi];
        if (c >= 0 && c < n_clusters)
          cluster_sizes[c]++;
      }
      int eligible_clusters = 0;
      for (int c = 0; c < n_clusters; ++c) {
        if (cluster_sizes[c] >= synth_min)
          eligible_clusters++;
      }
      const int target_synth = std::min(eligible_clusters, synth_max);
      int clusters_done = 0;
      int synth_done = 0;

      for (int c = 0; c < n_clusters; ++c) {
        std::vector<char> use_frame(frames.size(), 0);
        int count = 0;
        std::vector<float> cluster_q_values;
        cluster_q_values.reserve(frames.size());
        float cluster_mass = 0.0f;
        const float k_global =
            std::max(cfg.global_metrics.weight_exponent_scale, kEpsWeight);
        const float q_min = cfg.global_metrics.clamp[0];
        const float q_max = cfg.global_metrics.clamp[1];
        for (size_t fi = 0; fi < frames.size(); ++fi) {
          if (cluster_labels[fi] != c)
            continue;
          use_frame[fi] = 1;
          if (frame_has_data[fi]) {
            count++;
            const float G_f = (fi < static_cast<size_t>(global_weights.size()))
                                  ? global_weights[static_cast<int>(fi)]
                                  : 1.0f;
            const float q_f = std::clamp(
                std::log(std::max(G_f, kEpsWeight)) / k_global, q_min, q_max);
            if (std::isfinite(q_f)) {
              cluster_q_values.push_back(q_f);
            }
            if (std::isfinite(G_f) && G_f > 0.0f) {
              cluster_mass += G_f;
            }
          }
        }
        clusters_done++;
        emitter.phase_progress_counts(
            run_id, Phase::SYNTHETIC_FRAMES, clusters_done, n_clusters,
            "Cluster " + std::to_string(c + 1) + " von " +
                std::to_string(n_clusters),
            "synthetic " + std::to_string(synth_done) + "/" +
                std::to_string(target_synth),
            log_file);
        if (count < synth_min)
          continue;
        Matrix2Df syn = reconstruct_subset(use_frame);
        if (syn.size() == 0)
          continue;
        const float q_k =
            cluster_q_values.empty() ? 0.0f : core::median_of(cluster_q_values);

        if (detected_mode == ColorMode::OSC) {
          auto deb = image::debayer_nearest_neighbor(
              syn, detected_bayer, -canvas_tile_offset_x, -canvas_tile_offset_y);
          RGBFrame rgb;
          rgb.R = std::move(deb.R);
          rgb.G = std::move(deb.G);
          rgb.B = std::move(deb.B);
          synthetic_rgb_frames.push_back(std::move(rgb));
        }

        synthetic_frames.push_back(std::move(syn));
        synthetic_cluster_quality.push_back(q_k);
        synthetic_cluster_mass.push_back(
            (std::isfinite(cluster_mass) && cluster_mass > kEpsWeight)
                ? cluster_mass
                : static_cast<float>(count));
        synth_done = static_cast<int>(synthetic_frames.size());
        if (static_cast<int>(synthetic_frames.size()) >= synth_max)
          break;
      }

      if (synthetic_frames.empty()) {
        // If there are not enough frames to satisfy frames_min, treat as a
        // valid skip.
        if (static_cast<int>(frames.size()) < synth_min) {
          use_synthetic_frames = false;
          emitter.phase_end(run_id, Phase::SYNTHETIC_FRAMES, "skipped",
                            {{"reason", "insufficient_frames"},
                             {"frame_count", static_cast<int>(frames.size())},
                             {"frames_min", synth_min}},
                            log_file);
        } else {
          emitter.phase_end(
              run_id, Phase::SYNTHETIC_FRAMES, "error",
              {{"error", "SYNTHETIC_FRAMES: no synthetic frames"}}, log_file);
          emitter.run_end(run_id, false, "error", log_file,
                          {{"message", "SYNTHETIC_FRAMES: no synthetic frames"}});
          return 1;
        }
      }

    }

    if (use_synthetic_frames) {
      // Save synthetic frames
      for (size_t si = 0; si < synthetic_frames.size(); ++si) {
        std::string fname = "synthetic_" + std::to_string(si) + ".fit";
        Matrix2Df out = synthetic_frames[si];
        Matrix2Df valid_mask = Matrix2Df::Zero(out.rows(), out.cols());
        for (Eigen::Index pi = 0; pi < out.size(); ++pi) {
          const float v = out.data()[pi];
          valid_mask.data()[pi] =
              (std::isfinite(v) &&
               static_cast<size_t>(pi) < common_valid_mask.size() &&
               common_valid_mask[static_cast<size_t>(pi)] != 0)
                  ? 1.0f
                  : 0.0f;
        }
        image::apply_output_scaling_inplace(out, -canvas_tile_offset_x,
            -canvas_tile_offset_y, detected_mode,
            detected_bayer_str, output_scale_mono, output_scale_r,
            output_scale_g, output_scale_b, output_bg_mono, output_bg_r,
            output_bg_g, output_bg_b, output_pedestal);
        for (Eigen::Index pi = 0; pi < out.size(); ++pi) {
          if (valid_mask.data()[pi] == 0.0f) {
            out.data()[pi] = 0.0f;
          }
        }
        io::write_fits_float(run_dir / "outputs" / fname, out, first_hdr);
      }

      {
        core::json artifact;
        artifact["num_synthetic"] = static_cast<int>(synthetic_frames.size());
        artifact["frames_min"] = synth_min;
        artifact["frames_max"] = synth_max;
        artifact["requested_weighting"] =
            synthetic_weighting_decision.requested_weighting;
        artifact["effective_weighting"] =
            synthetic_weighting_decision.effective_weighting;
        artifact["tile_seam_guard_triggered"] =
            synthetic_weighting_decision.tile_seam_guard_triggered;
        artifact["tile_seam_guard_boundary_pair_count"] =
            synthetic_weighting_decision.boundary_pair_count;
        artifact["tile_seam_guard_boundary_pair_mean_abs_diff_p95"] =
            synthetic_weighting_decision.boundary_pair_mean_abs_diff_p95;
        artifact["tile_seam_guard_boundary_pair_scale_ratio_deviation_p95"] =
            synthetic_weighting_decision
                .boundary_pair_scale_ratio_deviation_p95;
        artifact["tile_seam_guard_boundary_post_background_delta_p95_abs"] =
            synthetic_weighting_decision
                .boundary_post_background_delta_p95_abs;
        artifact["tile_seam_guard_local_weight_mean_abs_delta_p95"] =
            synthetic_weighting_decision.local_weight_mean_abs_delta_p95;
        artifact["tile_seam_guard_local_weight_correlation_p05"] =
            synthetic_weighting_decision.local_weight_correlation_p05;
        artifact["weighting"] = synthetic_weighting_decision.effective_weighting;
        artifact["cluster_quality"] = core::json::array();
        artifact["cluster_mass"] = core::json::array();
        for (float qk : synthetic_cluster_quality) {
          artifact["cluster_quality"].push_back(qk);
        }
        for (float mk : synthetic_cluster_mass) {
          artifact["cluster_mass"].push_back(mk);
        }
        core::write_text(run_dir / "artifacts" / "synthetic_frames.json",
                         artifact.dump(2));
      }

      emitter.phase_end(
          run_id, Phase::SYNTHETIC_FRAMES, "ok",
          {{"num_synthetic", static_cast<int>(synthetic_frames.size())},
           {"requested_weighting",
            synthetic_weighting_decision.requested_weighting},
           {"effective_weighting",
            synthetic_weighting_decision.effective_weighting},
           {"tile_seam_guard_triggered",
            synthetic_weighting_decision.tile_seam_guard_triggered}},
          log_file);
    }
    tile_analysis_runtime_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      tile_analysis_started_at)
            .count();
    if (!skip_synthetic_for_aqmh &&
        abort_if_runtime_limit_exceeded("SYNTHETIC_FRAMES")) {
      return 1;
    }

    // --- Memory release: prewarped_frames disk cache no longer needed ---
    // The configured cleanup policy controls whether files remain available
    // for an AQMH reconstruction resume after the run ends.
    if (cfg.aqmh.reconstruction.delete_prewarped_cache_after_run) {
      prewarped_frames.cleanup();
      if (prewarped_frames_rgb.size() > 0) prewarped_frames_rgb.cleanup();
    } else {
      prewarped_frames.clear_mappings();
      if (prewarped_frames_rgb.size() > 0) prewarped_frames_rgb.clear_mappings();
    }
    { std::vector<uint8_t>().swap(frame_has_data); }

    // Phase 9: STACKING (final overlap-add already done in Phase 6)
    emitter.phase_start(run_id, Phase::STACKING, "STACKING", log_file);
    const auto stacking_started_at = std::chrono::steady_clock::now();
    const auto stacking_acceleration =
        acceleration.selection_for(core::AccelerationPhase::stacking);
    const core::AccelerationOps stacking_ops(
        acceleration, core::AccelerationPhase::stacking);
    core::WorkerCudaStreams stacking_streams(
        stacking_acceleration.selected ==
            core::AccelerationBackend::opencv_cuda,
        detected_mode == ColorMode::OSC ? 3u : 1u);
    size_t stacking_input_count = 0;
    {
      std::ostringstream msg;
      msg << "STACKING acceleration "
          << core::acceleration_selection_summary(stacking_acceleration)
          << " cpu_workers=" << (detected_mode == ColorMode::OSC ? 3 : 1)
          << " gpu=" << (stacking_acceleration.using_gpu ? "yes" : "no")
          << " backend="
          << core::acceleration_backend_name(stacking_acceleration.selected);
      if (!stacking_acceleration.request_honored &&
          !stacking_acceleration.fallback_reason.empty()) {
        emitter.warning(run_id, msg.str(), log_file);
      }
      std::cout << "[STACKING] " << msg.str() << std::endl;
    }

    if (use_synthetic_frames) {
      // Filter out empty (0×0) synthetic frames (empty cluster outputs)
      std::vector<Matrix2Df> valid_synth;
      valid_synth.reserve(synthetic_frames.size());
      std::vector<float> valid_synth_q;
      valid_synth_q.reserve(synthetic_frames.size());
      std::vector<float> valid_synth_mass;
      valid_synth_mass.reserve(synthetic_frames.size());

      // For OSC: keep a parallel list of per-frame RGB planes so we can
      // stack in RGB space and avoid debayering after sigma-clipped stacking.
      std::vector<Matrix2Df> synth_R;
      std::vector<Matrix2Df> synth_G;
      std::vector<Matrix2Df> synth_B;
      if (detected_mode == ColorMode::OSC) {
        synth_R.reserve(synthetic_frames.size());
        synth_G.reserve(synthetic_frames.size());
        synth_B.reserve(synthetic_frames.size());
      }

      for (size_t i = 0; i < synthetic_frames.size(); ++i) {
        auto &sf = synthetic_frames[i];
        if (sf.size() <= 0)
          continue;

        if (detected_mode == ColorMode::OSC) {
          if (i < synthetic_rgb_frames.size() &&
              synthetic_rgb_frames[i].R.size() > 0) {
            synth_R.push_back(std::move(synthetic_rgb_frames[i].R));
            synth_G.push_back(std::move(synthetic_rgb_frames[i].G));
            synth_B.push_back(std::move(synthetic_rgb_frames[i].B));
          } else {
            auto deb = image::debayer_nearest_neighbor(
                sf, detected_bayer, -canvas_tile_offset_x, -canvas_tile_offset_y);
            synth_R.push_back(std::move(deb.R));
            synth_G.push_back(std::move(deb.G));
            synth_B.push_back(std::move(deb.B));
          }
        }

        valid_synth.push_back(std::move(sf));
        if (i < synthetic_cluster_quality.size()) {
          valid_synth_q.push_back(synthetic_cluster_quality[i]);
        } else {
          valid_synth_q.push_back(0.0f);
        }
        if (i < synthetic_cluster_mass.size()) {
          valid_synth_mass.push_back(synthetic_cluster_mass[i]);
        } else {
          valid_synth_mass.push_back(1.0f);
        }
      }

      std::cerr << "[STACKING] " << valid_synth.size() << " / "
                << synthetic_frames.size() << " non-empty synthetic frames"
                << std::endl;

      if (!valid_synth.empty()) {
        stacking_input_count = valid_synth.size();
        const bool use_quality_weighting =
            cfg.stacking.cluster_quality_weighting.enabled;
        std::vector<float> cluster_stack_weights;
        if (use_quality_weighting) {
          cluster_stack_weights.resize(valid_synth_q.size(), 1.0f);
          const float kappa = cfg.stacking.cluster_quality_weighting.kappa_cluster;
          std::vector<float> q_values = valid_synth_q;
          const float q_ref =
              q_values.empty() ? 0.0f : core::median_of(q_values);
          for (size_t i = 0; i < valid_synth_q.size(); ++i) {
            const float mass =
                (i < valid_synth_mass.size() && std::isfinite(valid_synth_mass[i]) &&
                 valid_synth_mass[i] > kEpsWeight)
                    ? valid_synth_mass[i]
                    : 1.0f;
            const float q_rel =
                std::clamp(valid_synth_q[i] - q_ref, -3.0f, 3.0f);
            cluster_stack_weights[i] = mass * std::exp(kappa * q_rel);
            if (!std::isfinite(cluster_stack_weights[i]) ||
                cluster_stack_weights[i] <= 0.0f) {
              cluster_stack_weights[i] = 1.0f;
            }
          }
          if (cfg.stacking.cluster_quality_weighting.cap_enabled &&
              !cluster_stack_weights.empty()) {
            std::vector<float> tmp_w = cluster_stack_weights;
            const float med_w = core::median_of(tmp_w);
            const float cap =
                std::max(kEpsWeight,
                         cfg.stacking.cluster_quality_weighting.cap_ratio * med_w);
            for (float &w : cluster_stack_weights) {
              if (w > cap)
                w = cap;
            }
          }
        }

        if (detected_mode == ColorMode::OSC &&
            !synth_R.empty() && synth_R.size() == valid_synth.size()) {
          if (!use_quality_weighting && cfg.stacking.method == "rej") {
            auto stack_channel = [&](const std::vector<Matrix2Df> &channel,
                                     size_t stream_index) {
              return stacking_ops.sigma_clip_stack(
                  channel, cfg.stacking.sigma_clip.sigma_low,
                  cfg.stacking.sigma_clip.sigma_high,
                  cfg.stacking.sigma_clip.max_iters,
                  cfg.stacking.sigma_clip.min_fraction,
                  stacking_streams.get(stream_index));
            };
            auto future_r = std::async(std::launch::async, stack_channel,
                                       std::cref(synth_R), 0u);
            auto future_g = std::async(std::launch::async, stack_channel,
                                       std::cref(synth_G), 1u);
            recon_B = stack_channel(synth_B, 2u);
            recon_R = future_r.get();
            recon_G = future_g.get();
          } else {
            std::vector<float> stack_weights(synth_R.size(), 1.0f);
            if (use_quality_weighting &&
                cluster_stack_weights.size() == synth_R.size()) {
              stack_weights = cluster_stack_weights;
            }
            auto reduce_channel = [&](const std::vector<Matrix2Df> &channel,
                                      size_t stream_index) {
              return stacking_ops.sigma_clip_reduce(
                  channel, stack_weights, cfg.stacking.sigma_clip.sigma_low,
                  cfg.stacking.sigma_clip.sigma_high,
                  cfg.stacking.sigma_clip.max_iters,
                  cfg.stacking.sigma_clip.min_fraction, kEpsWeight,
                  stacking_streams.get(stream_index));
            };
            auto future_r = std::async(std::launch::async, reduce_channel,
                                       std::cref(synth_R), 0u);
            auto future_g = std::async(std::launch::async, reduce_channel,
                                       std::cref(synth_G), 1u);
            auto wr_b = reduce_channel(synth_B, 2u);
            auto wr_r = future_r.get();
            auto wr_g = future_g.get();
            recon_R = std::move(wr_r.tile);
            recon_G = std::move(wr_g.tile);
            recon_B = std::move(wr_b.tile);
          }
          recon = 0.25f * recon_R + 0.5f * recon_G + 0.25f * recon_B;
        } else {
          if (!use_quality_weighting && cfg.stacking.method == "rej") {
            recon = stacking_ops.sigma_clip_stack(
                valid_synth, cfg.stacking.sigma_clip.sigma_low,
                cfg.stacking.sigma_clip.sigma_high,
                cfg.stacking.sigma_clip.max_iters,
                cfg.stacking.sigma_clip.min_fraction,
                stacking_streams.get(0));
          } else {
            std::vector<float> stack_weights(valid_synth.size(), 1.0f);
            if (use_quality_weighting &&
                cluster_stack_weights.size() == valid_synth.size()) {
              stack_weights = cluster_stack_weights;
            }
            auto wr = stacking_ops.sigma_clip_reduce(
                valid_synth, stack_weights, cfg.stacking.sigma_clip.sigma_low,
                cfg.stacking.sigma_clip.sigma_high,
                cfg.stacking.sigma_clip.max_iters,
                cfg.stacking.sigma_clip.min_fraction, kEpsWeight,
                stacking_streams.get(0));
            recon = std::move(wr.tile);
          }
        }
      }
    }

    const auto stacking_input_batch = core::make_device_frame_batch(
        stacking_input_count, recon.rows(), recon.cols(),
        detected_mode == ColorMode::OSC ? 3 : 1);

    // Optional post-processing (not part of the linear quality core).
    if (cfg.stacking.cosmetic_correction) {
      const float cosmetic_sigma = cfg.stacking.cosmetic_correction_sigma;
      recon = image::cosmetic_correction(recon, cosmetic_sigma, true);
      const bool have_rgb_recon =
          detected_mode == ColorMode::OSC && recon_R.size() == recon.size() &&
          recon_G.size() == recon.size() && recon_B.size() == recon.size() &&
          recon_R.size() > 0;
      if (have_rgb_recon) {
        recon_R = image::cosmetic_correction(recon_R, cosmetic_sigma, true);
        recon_G = image::cosmetic_correction(recon_G, cosmetic_sigma, true);
        recon_B = image::cosmetic_correction(recon_B, cosmetic_sigma, true);
      }
    }

    if (detected_mode == ColorMode::OSC && cfg.chroma_denoise.enabled &&
        cfg.chroma_denoise.apply_stage == "post_stack_linear" &&
        recon_R.size() == recon.size() && recon_G.size() == recon.size() &&
        recon_B.size() == recon.size() && recon_R.size() > 0) {
      reconstruction::chroma_denoise_rgb_inplace(
          recon_R, recon_G, recon_B, cfg.chroma_denoise);
      recon = 0.25f * recon_R + 0.5f * recon_G + 0.25f * recon_B;
    }

	    auto write_stacking_outputs = [&](const Matrix2Df &stack_luma) -> bool {
	      Matrix2Df recon_out = stack_luma;
      // Stufe B: add accumulated background map to residual before output-scale
      // restoration; no extra scalar background is added.
      {
        const auto &bg_grid = aqmh_background_map_canvas_grid;
        if (bg_grid.channels() > 0 && bg_grid.rows() > 0 &&
            bg_grid.cols() > 0) {
          Matrix2Df bg_luma;
          if (bg_grid.channels() == 4) {
            const Matrix2Df bg_R =
                bg_grid.upsample_channel(0, canvas_height, canvas_width);
            const Matrix2Df G1 =
                bg_grid.upsample_channel(1, canvas_height, canvas_width);
            const Matrix2Df G2 =
                bg_grid.upsample_channel(2, canvas_height, canvas_width);
            const Matrix2Df bg_B =
                bg_grid.upsample_channel(3, canvas_height, canvas_width);
            bg_luma = 0.25f * bg_R + 0.25f * (G1 + G2) + 0.25f * bg_B;
          } else if (bg_grid.channels() == 1) {
            bg_luma =
                bg_grid.upsample_channel(0, canvas_height, canvas_width);
          }
          if (bg_luma.size() > 0 &&
              static_cast<size_t>(bg_luma.size()) ==
                  static_cast<size_t>(recon_out.size())) {
            recon_out += bg_luma;
          }
        }
      }
      if (detected_mode == ColorMode::OSC) {
        const float scale_luma = 0.25f * output_scale_r + 0.5f * output_scale_g +
                                 0.25f * output_scale_b;
        recon_out *= scale_luma;
        recon_out.array() += output_pedestal;
      } else {
        recon_out *= output_scale_mono;
        recon_out.array() += output_pedestal;
      }
      for (Eigen::Index k = 0; k < recon_out.size(); ++k) {
	        if (static_cast<size_t>(k) >= output_valid_mask.size() ||
	            output_valid_mask[static_cast<size_t>(k)] == 0) {
	          recon_out.data()[k] = 0.0f;
	        }
	      }

      if (cfg.stacking.output_stretch) {
        const auto stretch =
            core::stretch_to_u16_linear_from_zero_inplace(recon_out);
        if (stretch.applied) {
          std::cout << "[Stacking] Output linear stretch ["
                    << stretch.low << ".." << stretch.high
                    << "] -> [0..65535] samples=" << stretch.sample_count
                    << std::endl;
        }
      }

      try {
        std::error_code ec_space;
        const auto space_info = fs::space(run_dir, ec_space);
        if (!ec_space) {
          const uint64_t required_stack_bytes =
              static_cast<uint64_t>(std::max<Eigen::Index>(0, recon_out.size())) *
              sizeof(float) * 2ULL;
          const uint64_t available_bytes =
              static_cast<uint64_t>(space_info.available);
          if (available_bytes < required_stack_bytes) {
            const std::string msg =
                "Disk full risk before STACKING write: available=" +
                format_bytes(available_bytes) +
                ", required_estimate=" + format_bytes(required_stack_bytes);
            emitter.phase_end(run_id, Phase::STACKING, "error",
                              {{"error", msg},
                               {"runs_device_available_bytes", available_bytes},
                               {"required_estimate_bytes", required_stack_bytes},
                               {"outputs_dir", (run_dir / "outputs").string()}},
                              log_file);
            emitter.run_end(run_id, false, "insufficient_disk_space", log_file,
                            {{"message", msg}});
            std::cerr << "Error during STACKING: " << msg << std::endl;
            return false;
          }
        }

        io::write_fits_float(run_dir / "outputs" / "stacked.fits", recon_out,
                             first_hdr);
        io::write_fits_float(run_dir / "outputs" / "reconstructed_L.fit",
                             recon_out, first_hdr);
      } catch (const std::exception &e) {
        const bool disk_full = message_indicates_disk_full(e.what());
        const std::string msg =
            disk_full
                ? ("Disk full while writing STACKING outputs to " +
                   (run_dir / "outputs").string() + ": " + e.what())
                : (std::string("STACKING output write failed: ") + e.what());
        emitter.phase_end(run_id, Phase::STACKING, "error",
                          {{"error", msg},
                           {"outputs_dir", (run_dir / "outputs").string()}},
                          log_file);
        emitter.run_end(run_id, false,
                        disk_full ? "insufficient_disk_space" : "error",
                        log_file,
                        {{"message", msg}});
        std::cerr << "Error during STACKING: " << msg << std::endl;
        return false;
      }
      return true;
    };

    {
      bool validation_ok = true;
      core::json v;
      if (cfg.aqmh.enabled && aqmh_control_validation) {
        const auto &comparison = *aqmh_control_validation;
        const bool seam_ok = comparison.seam_score_regression <=
            cfg.aqmh.validation.max_seam_score_regression;
        const bool fwhm_ok = comparison.fwhm_regression <=
            cfg.aqmh.validation.max_fwhm_regression;
        const bool background_ok = comparison.background_rms_regression <=
            cfg.aqmh.validation.max_background_rms_regression;
        v["aqmh_uniform_control"] = {
            {"same_samples_masks_and_clipping", true},
            {"aqmh_seam_score", comparison.aqmh.seam_score},
            {"control_seam_score", comparison.control.seam_score},
            {"seam_score_regression", comparison.seam_score_regression},
            {"max_seam_score_regression",
             cfg.aqmh.validation.max_seam_score_regression},
            {"seam_score_ok", seam_ok},
            {"aqmh_fwhm", comparison.aqmh.fwhm},
            {"control_fwhm", comparison.control.fwhm},
            {"fwhm_regression", comparison.fwhm_regression},
            {"max_fwhm_regression",
             cfg.aqmh.validation.max_fwhm_regression},
            {"fwhm_ok", fwhm_ok},
            {"aqmh_background_rms", comparison.aqmh.background_rms},
            {"control_background_rms", comparison.control.background_rms},
            {"background_rms_regression",
             comparison.background_rms_regression},
            {"max_background_rms_regression",
             cfg.aqmh.validation.max_background_rms_regression},
            {"background_rms_ok", background_ok}};
        validation_ok = validation_ok && seam_ok && fwhm_ok && background_ok;
      }

      float output_fwhm_med = metrics::measure_fwhm_from_image(recon);

      float fwhm_improvement_percent = 0.0f;
      if (seeing_fwhm_med > 1.0e-6f && output_fwhm_med > 0.0f) {
        fwhm_improvement_percent =
            (seeing_fwhm_med - output_fwhm_med) / seeing_fwhm_med * 100.0f;
      }
      v["method"] = cfg.method;
      v["seeing_fwhm_median"] = seeing_fwhm_med;
      v["output_fwhm_median"] = output_fwhm_med;
      v["fwhm_improvement_percent"] = fwhm_improvement_percent;
      if (!cfg.aqmh.enabled) {
        if (fwhm_improvement_percent <
            cfg.validation.min_fwhm_improvement_percent) {
          validation_ok = false;
          v["fwhm_improvement_ok"] = false;
        } else {
          v["fwhm_improvement_ok"] = true;
        }
      } else {
        // AQMH: FWHM measurement only informational; no pass/fail threshold
        // because the FWHM measurer operates on the luma/proxy channel and
        // may return 0 when no stars are detected in the luminance map.
        if (output_fwhm_med > 0.0f && seeing_fwhm_med > 1.0e-6f) {
          v["fwhm_improvement_ok"] = true;
        } else {
          v["fwhm_improvement_ok"] = nullptr;  // not evaluated for AQMH
        }
      }

      {
        std::vector<float> input_noise_values;
        input_noise_values.reserve(frame_metrics.size());
        for (const auto &fm : frame_metrics) {
          if (std::isfinite(fm.noise) && fm.noise > 0.0f) {
            input_noise_values.push_back(fm.noise);
          }
        }
        const float input_background_rms =
            core::median_finite_positive(input_noise_values, 0.0f);

        Matrix2Df recon_background = recon;
        for (Eigen::Index i = 0; i < recon_background.size(); ++i) {
          if (!std::isfinite(recon_background.data()[i])) {
            recon_background.data()[i] = 0.0f;
          }
        }
        cv::Mat recon_background_cv(
            recon_background.rows(), recon_background.cols(), CV_32F,
            const_cast<float *>(recon_background.data()));
        const cv::Mat1b bg_mask =
            metrics::build_background_mask_sigma_clip(
                recon_background_cv, 3.0f, 3);
        std::vector<float> bg_samples;
        bg_samples.reserve(static_cast<size_t>(recon_background.size()));
        for (int y = 0; y < recon_background.rows(); ++y) {
          const uint8_t *mrow = bg_mask.ptr<uint8_t>(y);
          for (int x = 0; x < recon_background.cols(); ++x) {
            const float sample = recon_background(y, x);
            if (mrow[x] != 0 && std::isfinite(sample)) {
              bg_samples.push_back(sample);
            }
          }
        }
        std::vector<float> bg_samples_copy = bg_samples;
        const float output_background_rms =
            core::robust_sigma_mad(bg_samples_copy);
        float background_rms_increase_percent = 0.0f;
        if (input_background_rms > 1.0e-12f) {
          background_rms_increase_percent =
              (output_background_rms - input_background_rms) /
              input_background_rms * 100.0f;
        }
        v["input_background_rms"] = input_background_rms;
        v["output_background_rms"] = output_background_rms;
        v["background_rms_increase_percent"] =
            background_rms_increase_percent;

        const float max_bg_rms_increase =
            cfg.validation.max_background_rms_increase_percent;
        const bool enforce_bg_rms_limit = max_bg_rms_increase > 0.0f;
        const bool background_rms_ok =
            !enforce_bg_rms_limit ||
            background_rms_increase_percent <= max_bg_rms_increase;
        v["background_rms_ok"] = background_rms_ok;
        if (!background_rms_ok) {
          validation_ok = false;
        }
      }

      if (!cfg.aqmh.enabled) {
        float tile_weight_variance = 0.0f;
        {
          std::vector<float> tile_means;
          tile_means.reserve(tiles_phase56.size());
          for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
            double sum = 0.0;
            int cnt = 0;
            for (size_t fi = 0; fi < frames.size(); ++fi) {
              float G_f = (fi < static_cast<size_t>(global_weights.size()))
                              ? global_weights[static_cast<int>(fi)]
                              : 1.0f;
              float L_ft =
                  (fi < local_weights.size() && ti < local_weights[fi].size())
                      ? local_weights[fi][ti]
                      : 1.0f;
              sum += static_cast<double>(G_f * L_ft);
              cnt++;
            }
            tile_means.push_back(
                cnt > 0 ? static_cast<float>(sum / static_cast<double>(cnt))
                        : 0.0f);
          }
          double mean = 0.0;
          for (float x : tile_means)
            mean += static_cast<double>(x);
          mean /= std::max<double>(1.0, static_cast<double>(tile_means.size()));
          double var = 0.0;
          for (float x : tile_means) {
            double d = static_cast<double>(x) - mean;
            var += d * d;
          }
          var /= std::max<double>(1.0, static_cast<double>(tile_means.size()));
          tile_weight_variance =
              static_cast<float>(var / (mean * mean + 1.0e-12));
        }
        v["tile_weight_variance"] = tile_weight_variance;
        if (tile_weight_variance < cfg.validation.min_tile_weight_variance) {
          validation_ok = false;
          v["tile_weight_variance_ok"] = false;
        } else {
          v["tile_weight_variance_ok"] = true;
        }
      } else {
        // AQMH: derive quality-weight spread from per-frame map_mean values
        // instead of Classic tile weights (which are not computed for AQMH).
        const auto aqmh_metrics_path =
            run_dir / "artifacts" / "aqmh_metrics.json";
        const bool aqmh_metrics_exists =
            std::filesystem::exists(aqmh_metrics_path);
        const std::string aqmh_metrics_str =
            aqmh_metrics_exists ? core::read_text(aqmh_metrics_path) : "";
        if (!aqmh_metrics_str.empty()) {
          try {
            const auto am = core::json::parse(aqmh_metrics_str);
            if (am.contains("diagnostics") && am["diagnostics"].is_array() &&
                !am["diagnostics"].empty()) {
              double sum_mean = 0.0, sum_frac = 0.0;
              int n = 0;
              for (const auto &fr : am["diagnostics"]) {
                if (fr.contains("map_mean") && fr["map_mean"].is_number()) {
                  sum_mean += fr["map_mean"].get<double>();
                  ++n;
                }
                if (fr.contains("artifact_frac") &&
                    fr["artifact_frac"].is_number()) {
                  sum_frac += fr["artifact_frac"].get<double>();
                }
              }
              if (n > 0) {
                const double mean_map = sum_mean / n;
                double var_map = 0.0;
                for (const auto &fr : am["diagnostics"]) {
                  if (fr.contains("map_mean") && fr["map_mean"].is_number()) {
                    const double d = fr["map_mean"].get<double>() - mean_map;
                    var_map += d * d;
                  }
                }
                var_map /= std::max(1, n);
                v["aqmh_map_mean_variance"] = static_cast<float>(var_map);
                v["aqmh_map_mean_avg"] = static_cast<float>(mean_map);
                v["aqmh_artifact_frac_avg"] =
                    static_cast<float>(sum_frac / n);
                v["aqmh_frames_evaluated"] = n;
              }
            }
          } catch (const std::exception &) {
          }
        }
      }

      bool tile_pattern_ok = true;
      if (!cfg.aqmh.enabled && cfg.validation.require_no_tile_pattern) {
        Matrix2Df recon_validation = recon;
        for (Eigen::Index i = 0; i < recon_validation.size(); ++i) {
          if (!std::isfinite(recon_validation.data()[i])) {
            recon_validation.data()[i] = 0.0f;
          }
        }
        cv::Mat img_cv(recon_validation.rows(), recon_validation.cols(), CV_32F,
                       const_cast<float *>(recon_validation.data()));
        cv::Mat gx, gy;
        cv::Sobel(img_cv, gx, CV_32F, 1, 0, 3);
        cv::Sobel(img_cv, gy, CV_32F, 0, 1, 3);
        cv::Mat mag;
        cv::magnitude(gx, gy, mag);

        std::vector<int> xb;
        std::vector<int> yb;
        xb.reserve(tiles.size());
        yb.reserve(tiles.size());
        for (const auto &t : tiles) {
          if (t.x > 0)
            xb.push_back(t.x);
          if (t.y > 0)
            yb.push_back(t.y);
        }
        std::sort(xb.begin(), xb.end());
        xb.erase(std::unique(xb.begin(), xb.end()), xb.end());
        std::sort(yb.begin(), yb.end());
        yb.erase(std::unique(yb.begin(), yb.end()), yb.end());

        auto line_mean_x = [&](int x) -> float {
          if (x < 0 || x >= mag.cols)
            return 0.0f;
          double sum = 0.0;
          for (int y = 0; y < mag.rows; ++y)
            sum += static_cast<double>(mag.at<float>(y, x));
          return static_cast<float>(sum / static_cast<double>(mag.rows));
        };
        auto line_mean_y = [&](int y) -> float {
          if (y < 0 || y >= mag.rows)
            return 0.0f;
          double sum = 0.0;
          for (int x = 0; x < mag.cols; ++x)
            sum += static_cast<double>(mag.at<float>(y, x));
          return static_cast<float>(sum / static_cast<double>(mag.cols));
        };

        std::vector<float> boundary_ratios;
        boundary_ratios.reserve(xb.size() + yb.size());

        float worst_ratio = 1.0f;
        for (int x : xb) {
          float b = line_mean_x(x);
          float n = 0.5f * (line_mean_x(x - 2) + line_mean_x(x + 2));
          float r = b / (n + 1.0e-12f);
          boundary_ratios.push_back(r);
          if (r > worst_ratio)
            worst_ratio = r;
        }
        for (int y : yb) {
          float b = line_mean_y(y);
          float n = 0.5f * (line_mean_y(y - 2) + line_mean_y(y + 2));
          float r = b / (n + 1.0e-12f);
          boundary_ratios.push_back(r);
          if (r > worst_ratio)
            worst_ratio = r;
        }

        float p95_ratio = worst_ratio;
        if (!boundary_ratios.empty()) {
          if (eps_ws > 0.0f) {
            const size_t p95_idx = static_cast<size_t>(
                std::floor(0.95 * static_cast<double>(boundary_ratios.size() - 1)));
            std::nth_element(boundary_ratios.begin(),
                             boundary_ratios.begin() + static_cast<long>(p95_idx),
                             boundary_ratios.end());
            p95_ratio = boundary_ratios[p95_idx];
          }
        }

        v["tile_pattern_ratio"] = worst_ratio;
        v["tile_pattern_ratio_p95"] = p95_ratio;
        v["tile_pattern_boundary_count"] = static_cast<int>(boundary_ratios.size());
        tile_pattern_ok = (worst_ratio < 1.5f) && (p95_ratio < 1.25f);
        v["tile_pattern_ok"] = tile_pattern_ok;
        if (!tile_pattern_ok)
          validation_ok = false;
      }

      core::write_text(run_dir / "artifacts" / "validation.json", v.dump(2));

      // Do not abort here: we still want to run DEBAYER so GUI gets outputs.
      // We will mark the run as validation_failed at the end.
      if (!validation_ok) {
        run_validation_failed = true;
      }
    }

    runner::CropBox stacking_crop_box{
        0, 0, static_cast<int>(recon.cols()), static_cast<int>(recon.rows())};
    bool stacking_crop_applied = false;
    if (cfg.output.crop_to_nonzero_bbox && recon.size() > 0) {
      const int full_rows = recon.rows();
      const int full_cols = recon.cols();
      const bool have_rgb_full =
          (recon_R.rows() == full_rows && recon_R.cols() == full_cols &&
           recon_G.rows() == full_rows && recon_G.cols() == full_cols &&
           recon_B.rows() == full_rows && recon_B.cols() == full_cols);
      const size_t full_mask_px =
          static_cast<size_t>(full_rows) * static_cast<size_t>(full_cols);

      if (common_valid_mask.size() != full_mask_px) {
        const std::string msg =
            "internal canvas mask size mismatch during crop";
        emitter.phase_end(run_id, Phase::STACKING, "error",
                          {{"reason", "canvas_mask_size_mismatch"},
                           {"error", msg},
                           {"mask_pixels",
                            static_cast<uint64_t>(common_valid_mask.size())},
                           {"expected_mask_pixels",
                            static_cast<uint64_t>(full_mask_px)}},
                          log_file);
        emitter.run_end(run_id, false, "error", log_file,
                        {{"message", msg}});
        std::cerr << "Error during STACKING: " << msg << std::endl;
        return 1;
      }

      stacking_crop_box = cfg.aqmh.enabled
          ? runner::compute_support_mask_bbox(reconstruction_valid_mask,
                                               full_rows, full_cols)
          : runner::compute_nonzero_data_bbox(
                recon, have_rgb_full ? &recon_R : nullptr,
                have_rgb_full ? &recon_G : nullptr,
                have_rgb_full ? &recon_B : nullptr);
      if (!stacking_crop_box.valid()) {
        const std::string msg =
            "crop_to_nonzero_bbox produced empty valid canvas";
        emitter.phase_end(run_id, Phase::STACKING, "error",
                          {{"reason", "empty_valid_crop"},
                           {"error", msg},
                           {"full_width", full_cols},
                           {"full_height", full_rows}},
                          log_file);
        emitter.run_end(run_id, false, "error", log_file,
                        {{"message", msg}});
        std::cerr << "Error during STACKING: " << msg << std::endl;
        return 1;
      }

      const int crop_x = stacking_crop_box.x;
      const int crop_y = stacking_crop_box.y;
      const int crop_w = stacking_crop_box.width;
      const int crop_h = stacking_crop_box.height;
      stacking_crop_applied =
          (crop_x != 0 || crop_y != 0 || crop_w != full_cols ||
           crop_h != full_rows);

      if (stacking_crop_applied) {
        recon = recon.block(crop_y, crop_x, crop_h, crop_w).eval();
        if (have_rgb_full) {
          recon_R = recon_R.block(crop_y, crop_x, crop_h, crop_w).eval();
          recon_G = recon_G.block(crop_y, crop_x, crop_h, crop_w).eval();
          recon_B = recon_B.block(crop_y, crop_x, crop_h, crop_w).eval();
        }
        debayer_tile_offset_x -= crop_x;
        debayer_tile_offset_y -= crop_y;

        std::vector<uint8_t> cropped_common_mask(
            static_cast<size_t>(crop_h * crop_w), static_cast<uint8_t>(0));
        std::vector<uint8_t> cropped_analysis_mask(
            static_cast<size_t>(crop_h * crop_w), static_cast<uint8_t>(0));
        std::vector<uint8_t> cropped_recon_mask;
        const bool have_recon_mask =
            (reconstruction_valid_mask.size() == full_mask_px);
        if (have_recon_mask) {
          cropped_recon_mask.assign(static_cast<size_t>(crop_h * crop_w),
                                     static_cast<uint8_t>(0));
        }

        std::vector<uint8_t> cropped_df_valid_mask_R;
        std::vector<uint8_t> cropped_df_valid_mask_G;
        std::vector<uint8_t> cropped_df_valid_mask_B;
        const bool have_df_valid_masks =
            df_valid_mask_R.size() == static_cast<size_t>(full_mask_px) &&
            df_valid_mask_G.size() == static_cast<size_t>(full_mask_px) &&
            df_valid_mask_B.size() == static_cast<size_t>(full_mask_px);
        if (have_df_valid_masks) {
          cropped_df_valid_mask_R.assign(static_cast<size_t>(crop_h * crop_w), 0u);
          cropped_df_valid_mask_G.assign(static_cast<size_t>(crop_h * crop_w), 0u);
          cropped_df_valid_mask_B.assign(static_cast<size_t>(crop_h * crop_w), 0u);
        }

        for (int y = 0; y < crop_h; ++y) {
          const int sy = crop_y + y;
          const size_t src_row_off =
              static_cast<size_t>(sy) * static_cast<size_t>(full_cols);
          const size_t dst_row_off =
              static_cast<size_t>(y) * static_cast<size_t>(crop_w);
          for (int x = 0; x < crop_w; ++x) {
            const int sx = crop_x + x;
            cropped_common_mask[dst_row_off + static_cast<size_t>(x)] =
                common_valid_mask[src_row_off + static_cast<size_t>(sx)];
            cropped_analysis_mask[dst_row_off + static_cast<size_t>(x)] =
                analysis_valid_mask[src_row_off + static_cast<size_t>(sx)];
            if (have_recon_mask) {
              cropped_recon_mask[dst_row_off + static_cast<size_t>(x)] =
                  reconstruction_valid_mask[src_row_off + static_cast<size_t>(sx)];
            }
            if (have_df_valid_masks) {
              cropped_df_valid_mask_R[dst_row_off + static_cast<size_t>(x)] =
                  df_valid_mask_R[src_row_off + static_cast<size_t>(sx)];
              cropped_df_valid_mask_G[dst_row_off + static_cast<size_t>(x)] =
                  df_valid_mask_G[src_row_off + static_cast<size_t>(sx)];
              cropped_df_valid_mask_B[dst_row_off + static_cast<size_t>(x)] =
                  df_valid_mask_B[src_row_off + static_cast<size_t>(sx)];
            }
          }
        }
        common_valid_mask.swap(cropped_common_mask);
        analysis_valid_mask.swap(cropped_analysis_mask);
        if (have_df_valid_masks) {
          df_valid_mask_R.swap(cropped_df_valid_mask_R);
          df_valid_mask_G.swap(cropped_df_valid_mask_G);
          df_valid_mask_B.swap(cropped_df_valid_mask_B);
        }
        if (have_recon_mask) {
          reconstruction_valid_mask.swap(cropped_recon_mask);
        }

        const fs::path common_mask_path =
            run_dir / "outputs" / "common_overlap_mask.fits";
        const fs::path mask_path = run_dir / "outputs" / "canvas_mask.fits";
        std::string mask_write_error;
        if (!write_canvas_mask_fits(common_mask_path, analysis_valid_mask, crop_h,
                                    crop_w, first_hdr, mask_write_error) ||
            !write_canvas_mask_fits(mask_path, output_valid_mask, crop_h,
                                    crop_w, first_hdr, mask_write_error)) {
          emitter.phase_end(run_id, Phase::STACKING, "error",
                            {{"reason", "canvas_mask_write_failed"},
                             {"error", mask_write_error},
                             {"canvas_mask", mask_path.string()}},
                            log_file);
          emitter.run_end(run_id, false, "error", log_file,
                          {{"message", mask_write_error}});
          std::cerr << "Error during STACKING: " << mask_write_error
                    << std::endl;
          return 1;
        }
        std::cout << "[COMMON_OVERLAP] Analysis and output masks updated after "
                     "crop: "
                  << crop_w << "x" << crop_h << std::endl;
      }
    }

    if (!write_stacking_outputs(recon)) {
      return 1;
    }

    emitter.phase_end(
        run_id, Phase::STACKING, "ok",
        {{"acceleration",
          core::acceleration_selection_to_json(stacking_acceleration)},
         {"device_frame_batch_input",
          core::device_frame_batch_to_json(stacking_input_batch)},
         {"input_frames", static_cast<int>(stacking_input_count)},
         {"crop_applied", stacking_crop_applied},
         {"crop_source", cfg.aqmh.enabled ? "reconstruction_support_mask"
                                           : "nonzero_data_bbox"},
         {"crop_x", stacking_crop_box.x},
         {"crop_y", stacking_crop_box.y},
         {"crop_width", stacking_crop_box.width},
         {"crop_height", stacking_crop_box.height},
         {"output_luma", (run_dir / "outputs" / "stacked.fits").string()}},
        log_file);
    stacking_runtime_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      stacking_started_at)
            .count();
    {
      core::json runtime_limits_artifact = {
          {"tile_analysis_runtime_seconds", tile_analysis_runtime_seconds},
          {"stacking_runtime_seconds", stacking_runtime_seconds},
          {"tile_analysis_max_factor_vs_stack",
           cfg.runtime_limits.tile_analysis_max_factor_vs_stack},
      };
      if (stacking_runtime_seconds > 1.0e-9) {
        const double ratio =
            tile_analysis_runtime_seconds / stacking_runtime_seconds;
        runtime_limits_artifact["tile_analysis_to_stack_ratio"] = ratio;
        // For AQMH the tile_analysis timer spans AQMH map computation which is
        // much longer than classic tile analysis; ratio check is not meaningful.
        const bool ratio_check_applicable = !cfg.aqmh.enabled;
        runtime_limits_artifact["tile_analysis_ratio_applicable"] =
            ratio_check_applicable;
        const bool ratio_ok =
            ratio_check_applicable &&
            ratio <= cfg.runtime_limits.tile_analysis_max_factor_vs_stack;
        runtime_limits_artifact["tile_analysis_ratio_ok"] =
            ratio_check_applicable ? core::json(ratio_ok) : core::json(nullptr);
        if (ratio_check_applicable && !ratio_ok) {
          emitter.warning(
              run_id,
              "Tile-analysis runtime anomaly: ratio=" + std::to_string(ratio) +
                  " exceeds runtime_limits.tile_analysis_max_factor_vs_stack=" +
                  std::to_string(
                      cfg.runtime_limits.tile_analysis_max_factor_vs_stack),
              log_file);
        }
        // Anforderung 7.2: always warn when ratio > 10.
        if (ratio_check_applicable && ratio > 10.0 && ratio_ok) {
          emitter.warning(
              run_id,
              "TILE_RECONSTRUCTION took " + std::to_string(ratio) +
                  "x longer than STACKING (threshold: 10). "
                  "Consider reducing overlap_fraction or increasing memory_budget.",
              log_file);
        }
      } else {
        runtime_limits_artifact["tile_analysis_to_stack_ratio"] = nullptr;
        runtime_limits_artifact["tile_analysis_ratio_ok"] = false;
      }
      core::write_text(run_dir / "artifacts" / "runtime_limits.json",
                       runtime_limits_artifact.dump(2));
    }
    if (abort_if_runtime_limit_exceeded("STACKING")) {
      return 1;
    }

    // Phase 10: DEBAYER (for OSC data)
    emitter.phase_start(run_id, Phase::DEBAYER, "DEBAYER", log_file);

    Matrix2Df R_out, G_out, B_out;
    bool have_rgb = false;
    fs::path stacked_rgb_path = run_dir / "outputs" / "stacked_rgb.fits";
    fs::path stacked_rgb_solve_path = run_dir / "outputs" / "stacked_rgb_solve.fits";
    auto stretch_rgb_for_output = [&](Matrix2Df& R_ch, Matrix2Df& G_ch,
                                      Matrix2Df& B_ch,
                                      const char* stage_tag) -> bool {
      const std::vector<uint8_t>& statistics_mask =
          analysis_valid_mask.size() == static_cast<size_t>(R_ch.size())
              ? analysis_valid_mask
              : (common_valid_mask.size() == static_cast<size_t>(R_ch.size())
                     ? common_valid_mask
                     : reconstruction_valid_mask);
      const auto stretch =
          core::stretch_rgb_to_u32_linear_from_zero_inplace(
              R_ch, G_ch, B_ch, statistics_mask);
      if (!stretch.applied) return false;
      std::cout << "[" << stage_tag
                << "] RGB output "
                << "linear"
                << " stretch ["
                << stretch.low << ".." << stretch.high << "] -> [0..4294967295]"
                << " samples=" << stretch.sample_count << std::endl;
      return true;
    };

    auto write_output_rgb_snapshot = [&](const fs::path &path,
                                         const Matrix2Df &R_src,
                                         const Matrix2Df &G_src,
                                         const Matrix2Df &B_src,
                                         const io::FitsHeader &hdr,
                                         const char *stage_tag) {
      Matrix2Df R_disk = R_src;
      Matrix2Df G_disk = G_src;
      Matrix2Df B_disk = B_src;
      const bool stretched = cfg.stacking.output_stretch;
      if (stretched) {
        stretch_rgb_for_output(R_disk, G_disk, B_disk, stage_tag);
      }
      image::enforce_canvas_mask_on_rgb(R_disk, G_disk, B_disk,
                                        reconstruction_valid_mask);
      std::error_code ec;
      fs::remove(path, ec);
      if (stretched) {
        io::write_fits_rgb_u32(path, R_disk, G_disk, B_disk, hdr);
      } else {
        io::write_fits_rgb(path, R_disk, G_disk, B_disk, hdr);
      }
    };

    bool have_successful_bge = false;
    fs::path stacked_rgb_bge_path = run_dir / "outputs" / "stacked_rgb_bge.fits";
    fs::path stacked_rgb_bge_linear_path =
        run_dir / "outputs" / "stacked_rgb_bge_linear.fits";

    if (detected_mode == ColorMode::OSC) {
      std::string debayer_method = "precomputed_rgb";
      io::FitsHeader rgb_output_hdr = first_hdr;
      rgb_output_hdr.set("DEBAYER", "PRE_STACK");
      if (recon_R.size() == recon.size() && recon_R.size() > 0 &&
          recon_G.size() == recon.size() && recon_B.size() == recon.size()) {
        R_out = std::move(recon_R);
        G_out = std::move(recon_G);
        B_out = std::move(recon_B);
      } else {
        // Fallback (should be rare): use the same edge-aware CFA output path
        // as resume so normal and resumed AQMH outputs share debayer semantics.
        auto debayer = image::debayer_opencv(
            recon, detected_bayer, -debayer_tile_offset_x,
            -debayer_tile_offset_y, /*ahd=*/true);
        debayer_method = "edge_aware";
        R_out = std::move(debayer.R);
        G_out = std::move(debayer.G);
        B_out = std::move(debayer.B);
      }
      have_rgb = true;
      // Stufe B: add accumulated background map to residual, then apply
      // output-scale restoration (no extra scalar background term).
      {
        const auto &bg_grid = aqmh_background_map_canvas_grid;
        if (bg_grid.channels() > 0 && bg_grid.rows() > 0 &&
            bg_grid.cols() > 0) {
          Matrix2Df bg_R, bg_G, bg_B;
          if (bg_grid.channels() == 4) {
            bg_R = bg_grid.upsample_channel(0, canvas_height, canvas_width);
            const Matrix2Df G1 =
                bg_grid.upsample_channel(1, canvas_height, canvas_width);
            const Matrix2Df G2 =
                bg_grid.upsample_channel(2, canvas_height, canvas_width);
            bg_G = 0.5f * (G1 + G2);
            bg_B = bg_grid.upsample_channel(3, canvas_height, canvas_width);
          } else if (bg_grid.channels() == 1) {
            bg_R = bg_grid.upsample_channel(0, canvas_height, canvas_width);
            bg_G = bg_R;
            bg_B = bg_R;
          } else if (bg_grid.channels() == 3) {
            bg_R = bg_grid.upsample_channel(0, canvas_height, canvas_width);
            bg_G = bg_grid.upsample_channel(1, canvas_height, canvas_width);
            bg_B = bg_grid.upsample_channel(2, canvas_height, canvas_width);
          }
          if (bg_R.size() > 0 &&
              static_cast<size_t>(bg_R.size()) ==
                  static_cast<size_t>(R_out.size())) {
            R_out += bg_R;
          }
          if (bg_G.size() > 0 &&
              static_cast<size_t>(bg_G.size()) ==
                  static_cast<size_t>(G_out.size())) {
            G_out += bg_G;
          }
          if (bg_B.size() > 0 &&
              static_cast<size_t>(bg_B.size()) ==
                  static_cast<size_t>(B_out.size())) {
            B_out += bg_B;
          }
        }
      }
      R_out *= output_scale_r;
      G_out *= output_scale_g;
      B_out *= output_scale_b;
      R_out.array() += output_pedestal;
      G_out.array() += output_pedestal;
      B_out.array() += output_pedestal;
      image::enforce_canvas_mask_on_rgb(R_out, G_out, B_out,
                                        reconstruction_valid_mask);

      // Apply per-channel valid masks. Pixels with no contributing frame
      // (weight_sum == 0) are marked as NaN instead of a physical 0.
      if (df_valid_mask_R.size() == static_cast<size_t>(R_out.size()) &&
          R_out.size() > 0) {
        for (size_t i = 0; i < static_cast<size_t>(R_out.size()); ++i) {
          if (df_valid_mask_R[i] == 0u) R_out.data()[i] = std::numeric_limits<float>::quiet_NaN();
          if (df_valid_mask_G[i] == 0u) G_out.data()[i] = std::numeric_limits<float>::quiet_NaN();
          if (df_valid_mask_B[i] == 0u) B_out.data()[i] = std::numeric_limits<float>::quiet_NaN();
        }
      }

      io::write_fits_float(run_dir / "outputs" / "reconstructed_R.fit", R_out,
                           rgb_output_hdr);
      io::write_fits_float(run_dir / "outputs" / "reconstructed_G.fit", G_out,
                           rgb_output_hdr);
      io::write_fits_float(run_dir / "outputs" / "reconstructed_B.fit", B_out,
                           rgb_output_hdr);
      write_output_rgb_snapshot(stacked_rgb_path, R_out, G_out, B_out,
                                rgb_output_hdr, "STACKING");
      // Write an additional linear (non-stretched) cube for plate solving.
      io::write_fits_rgb(stacked_rgb_solve_path, R_out, G_out, B_out,
                         rgb_output_hdr);

      emitter.phase_end(
          run_id, Phase::DEBAYER, "ok",
          {{"mode", "OSC"},
           {"bayer_pattern", bayer_pattern_to_string(detected_bayer)},
           {"debayer_method", debayer_method},
           {"output_rgb", stacked_rgb_path.string()},
           {"output_rgb_solve", stacked_rgb_solve_path.string()}},
          log_file);
    } else {
      emitter.phase_end(run_id, Phase::DEBAYER, "ok", {{"mode", "MONO"}},
                        log_file);
    }
    if (abort_if_runtime_limit_exceeded("DEBAYER")) {
      return 1;
    }

    // Phase 11: ASTROMETRY (plate solve via ASTAP)
    emitter.phase_start(run_id, Phase::ASTROMETRY, "ASTROMETRY", log_file);

    astro::WCS wcs;
    bool have_wcs = false;

    if (!cfg.astrometry.enabled) {
      emitter.phase_end(run_id, Phase::ASTROMETRY, "skipped",
                        {{"reason", "disabled"}}, log_file);
    } else if (!have_rgb) {
      emitter.phase_end(run_id, Phase::ASTROMETRY, "skipped",
                        {{"reason", "no_rgb_data"}}, log_file);
    } else {
      // Determine ASTAP paths (config or defaults)
      std::string astap_data = cfg.astrometry.astap_data_dir;
      if (astap_data.empty()) {
#ifdef _WIN32
        if (const char *la = std::getenv("LOCALAPPDATA"); la && la[0] != '\0') {
          astap_data = std::string(la) + "\\tile_compile\\astap";
        }
#else
        const char *home = std::getenv("HOME");
        if (home) astap_data = std::string(home) + "/.local/share/tile_compile/astap";
#endif
      }
      fs::path astap_bin_path = resolve_astap_binary_path(cfg.astrometry.astap_bin, astap_data);
      // If the resolved binary lives outside the configured data dir, use its parent as data dir
      if (!astap_bin_path.empty()) {
        std::error_code ec;
        fs::path data_dir_path(astap_data);
        auto relative = fs::relative(astap_bin_path, data_dir_path, ec);
        if (ec || relative.empty() || relative.begin() == relative.end() || *relative.begin() == "..") {
          astap_data = astap_bin_path.parent_path().string();
        }
      }

      if (astap_bin_path.empty()) {
        emitter.phase_end(run_id, Phase::ASTROMETRY, "skipped",
                          {{"reason", "astap_not_found"},
                           {"astap_bin", cfg.astrometry.astap_bin.empty() ? astap_data + "/astap_cli" : cfg.astrometry.astap_bin}}, log_file);
      } else {
        // Run ASTAP plate solve on the linear (non-stretched) RGB cube
        std::string cmd = shell_quote(astap_bin_path.string()) + " -f " +
            shell_quote(stacked_rgb_solve_path.string()) +
            " -d " + shell_quote(astap_data) +
            " -r " + std::to_string(cfg.astrometry.search_radius);

        std::cout << "[ASTROMETRY] Running: " << cmd << std::endl;
        int ret = std::system(system_cmd(cmd).c_str());

        // ASTAP writes a .wcs file next to the input
        fs::path wcs_path = stacked_rgb_solve_path;
        wcs_path.replace_extension(".wcs");

        if (ret == 0 && fs::exists(wcs_path)) {
          try {
            wcs = astro::parse_wcs_file(wcs_path.string());
            have_wcs = wcs.valid();
          } catch (const std::exception &e) {
            std::cerr << "[ASTROMETRY] WCS parse error: " << e.what() << std::endl;
          }
        }

        if (have_wcs) {
          // Inject WCS keywords into first_hdr so all subsequent
          // FITS outputs (PCC etc.) inherit the astrometric solution.
          // ASTAP does not write WCS into FLOAT_IMG FITS (BITPIX=-32).
          first_hdr.numeric_values["CRVAL1"] = wcs.crval1;
          first_hdr.numeric_values["CRVAL2"] = wcs.crval2;
          first_hdr.numeric_values["CRPIX1"] = wcs.crpix1;
          first_hdr.numeric_values["CRPIX2"] = wcs.crpix2;
          first_hdr.numeric_values["CD1_1"]  = wcs.cd1_1;
          first_hdr.numeric_values["CD1_2"]  = wcs.cd1_2;
          first_hdr.numeric_values["CD2_1"]  = wcs.cd2_1;
          first_hdr.numeric_values["CD2_2"]  = wcs.cd2_2;
          first_hdr.numeric_values["EQUINOX"] = 2000.0;
          first_hdr.string_values["CTYPE1"]  = "RA---TAN";
          first_hdr.string_values["CTYPE2"]  = "DEC--TAN";
          first_hdr.string_values["CUNIT1"]  = "deg";
          first_hdr.string_values["CUNIT2"]  = "deg";
          first_hdr.bool_values["PLTSOLVD"] = true;

          // Re-write outputs with WCS keywords
          if (have_rgb) {
            try {
              io::update_fits_header_in_place(stacked_rgb_path, first_hdr);
              std::cout << "[ASTROMETRY] WCS keywords written to " << stacked_rgb_path << std::endl;
            } catch (const std::exception &e) {
              std::cout << "[ASTROMETRY] Could not update stacked_rgb.fits: " << e.what() << std::endl;
            }
            try {
              io::update_fits_header_in_place(stacked_rgb_solve_path, first_hdr);
            } catch (const std::exception &) {
            }
          }

          // Copy .wcs to run artifacts directory
          fs::path wcs_artifact = run_dir / "artifacts" / "stacked_rgb.wcs";
          try {
            fs::copy_file(wcs_path, wcs_artifact,
                          fs::copy_options::overwrite_existing);
            std::cout << "[ASTROMETRY] WCS saved to " << wcs_artifact << std::endl;
          } catch (const std::exception &e) {
            std::cerr << "[ASTROMETRY] Could not copy .wcs: " << e.what() << std::endl;
          }

          emitter.phase_end(run_id, Phase::ASTROMETRY, "ok",
                            {{"ra", wcs.crval1},
                             {"dec", wcs.crval2},
                             {"pixel_scale_arcsec", wcs.pixel_scale_arcsec()},
                             {"rotation_deg", wcs.rotation_deg()},
                             {"fov_w_deg", wcs.fov_width_deg()},
                             {"fov_h_deg", wcs.fov_height_deg()},
                             {"wcs_file", wcs_artifact.string()}},
                            log_file);
        } else {
          emitter.phase_end(run_id, Phase::ASTROMETRY, "skipped",
                            {{"reason", "solve_failed"},
                             {"exit_code", ret}}, log_file);
        }
      }
    }
    if (abort_if_runtime_limit_exceeded("ASTROMETRY")) {
      return 1;
    }

    auto assign_output_canvas_mask_for_rgb =
        [&](std::vector<uint8_t> &out_mask, int &rows_out, int &cols_out,
            std::string &error_out) -> bool {
      rows_out = 0;
      cols_out = 0;
      if (R_out.rows() <= 0 || R_out.cols() <= 0 || R_out.rows() != G_out.rows() ||
          R_out.rows() != B_out.rows() || R_out.cols() != G_out.cols() ||
          R_out.cols() != B_out.cols()) {
        error_out = "invalid RGB dimensions";
        return false;
      }

      rows_out = static_cast<int>(R_out.rows());
      cols_out = static_cast<int>(R_out.cols());
      const size_t expected_size =
          static_cast<size_t>(rows_out) * static_cast<size_t>(cols_out);
      if (reconstruction_valid_mask.size() != expected_size) {
        error_out = "reconstruction_valid_mask size mismatch";
        return false;
      }
      out_mask = reconstruction_valid_mask;
      return true;
    };

    // Phase 11.5: BGE (Background Gradient Extraction) - v3.3 §6.3
    // Must run BEFORE PCC to remove gradients that would bias color calibration
    const std::string bge_phase_label =
        (cfg.bge.method == "none")    ? "BGE (Skipped)" :
        (cfg.bge.method == "classic") ? "BGE (Classic)" :
                                        "BGE (AutoBGE)";
    emitter.phase_start(run_id, Phase::BGE, "BGE", log_file,
                        {{"label", bge_phase_label},
                         {"bge_method", cfg.bge.method}});

    if (cfg.bge.method == "none") {
      std::error_code ec_linear;
      std::error_code ec_display;
      fs::remove(stacked_rgb_bge_linear_path, ec_linear);
      fs::remove(stacked_rgb_bge_path, ec_display);
      emitter.phase_end(run_id, Phase::BGE, "skipped",
                        {{"reason", "disabled"},
                         {"bge_method", cfg.bge.method}}, log_file);
    } else if (!have_rgb) {
      std::error_code ec_linear;
      std::error_code ec_display;
      fs::remove(stacked_rgb_bge_linear_path, ec_linear);
      fs::remove(stacked_rgb_bge_path, ec_display);
      emitter.phase_end(run_id, Phase::BGE, "skipped",
                        {{"reason", "no_rgb_data"}}, log_file);
    } else {
      constexpr int bge_progress_total = 4;
      emitter.phase_progress_counts(run_id, Phase::BGE, 0, bge_progress_total,
                                    "prepare", "BGE", log_file);
      std::cerr << "[BGE] Starting background gradient extraction (v3.3 §6.3)" << std::endl;

      // BGE requires tile metrics from LOCAL_METRICS and matching tile geometry.
      // Prefer final post-PREWARP canvas tiles, but ensure strict compatibility
      // with metric vector length when available.
      TileGrid bge_tile_grid;
      bge_tile_grid.tile_size = uniform_tile_size;
      bge_tile_grid.overlap_fraction = overlap_fraction;
      bge_tile_grid.rows = 0;
      bge_tile_grid.cols = 0;

      const std::vector<Tile>* bge_tiles_source = &tiles;
      std::vector<Tile> aqmh_bge_tiles;
      if (!bge_tile_metrics_cache.empty()) {
        const size_t metrics_tile_count = bge_tile_metrics_cache.size();
        if (metrics_tile_count == tiles_phase56.size()) {
          bge_tiles_source = &tiles_phase56;
        } else if (metrics_tile_count == tiles.size()) {
          bge_tiles_source = &tiles;
        }
      } else if (cfg.aqmh.enabled) {
        aqmh_bge_tiles = tile_compile::pipeline::build_initial_tile_grid(
            R_out.cols(), R_out.rows(), std::max(32, uniform_tile_size),
            overlap_fraction);
        bge_tiles_source = &aqmh_bge_tiles;
      }
      bge_tile_grid.tiles = *bge_tiles_source;

      if (!bge_tile_grid.tiles.empty()) {
        int max_row = 0;
        int max_col = 0;
        for (const auto &t : bge_tile_grid.tiles) {
          max_row = std::max(max_row, t.row + 1);
          max_col = std::max(max_col, t.col + 1);
        }
        bge_tile_grid.rows = max_row;
        bge_tile_grid.cols = max_col;
      }
      bge_tile_grid_cache = bge_tile_grid;

      std::string bge_tile_metrics_source;
      // Set BGE tile metrics source based on reconstruction method
      if (cfg.method == "aqmh") {
        bge_tile_metrics_source = "aqmh_output";
      } else {
        bge_tile_metrics_source = bge_tile_metrics_cache.empty() ? "none" : "classic_local_metrics";
      }
      auto build_aqmh_bge_tile_metrics =
          [&](const TileGrid &grid, const std::vector<uint8_t> &valid_mask,
              int mask_rows, int mask_cols) {
            std::vector<TileMetrics> out;
            out.reserve(grid.tiles.size());
            const bool mask_ok =
                mask_rows == R_out.rows() && mask_cols == R_out.cols() &&
                valid_mask.size() ==
                    static_cast<size_t>(R_out.rows() * R_out.cols());
            for (const auto &tile : grid.tiles) {
              TileMetrics tm{};
              tm.fwhm = 0.0f;
              tm.roundness = 0.0f;
              tm.contrast = 0.0f;
              tm.sharpness = 0.0f;
              tm.background = 0.0f;
              tm.noise = 0.0f;
              tm.gradient_energy = 0.0f;
              tm.star_count = 0;
              tm.type = TileType::STRUCTURE;
              tm.quality_score = 0.0f;

              const int x0 = std::max(0, tile.x);
              const int y0 = std::max(0, tile.y);
              const int x1 = std::min(tile.x + tile.width,
                                      static_cast<int>(R_out.cols()));
              const int y1 = std::min(tile.y + tile.height,
                                      static_cast<int>(R_out.rows()));
              if (x1 <= x0 || y1 <= y0 || !mask_ok) {
                out.push_back(tm);
                continue;
              }

              std::vector<float> values;
              values.reserve(static_cast<size_t>((x1 - x0) * (y1 - y0)));
              double gradient_sum = 0.0;
              size_t gradient_count = 0;
              for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                  const size_t idx = static_cast<size_t>(y) *
                                         static_cast<size_t>(mask_cols) +
                                     static_cast<size_t>(x);
                  if (valid_mask[idx] == 0) {
                    continue;
                  }
                  const float rv = R_out(y, x);
                  const float gv = G_out(y, x);
                  const float bv = B_out(y, x);
                  if (!(std::isfinite(rv) && std::isfinite(gv) &&
                        std::isfinite(bv))) {
                    continue;
                  }
                  const float luma =
                      0.2126f * rv + 0.7152f * gv + 0.0722f * bv;
                  if (!std::isfinite(luma)) {
                    continue;
                  }
                  values.push_back(luma);
                  const int xm = std::max(x0, x - 1);
                  const int xp = std::min(x1 - 1, x + 1);
                  const int ym = std::max(y0, y - 1);
                  const int yp = std::min(y1 - 1, y + 1);
                  const float l_xm = 0.2126f * R_out(y, xm) +
                                     0.7152f * G_out(y, xm) +
                                     0.0722f * B_out(y, xm);
                  const float l_xp = 0.2126f * R_out(y, xp) +
                                     0.7152f * G_out(y, xp) +
                                     0.0722f * B_out(y, xp);
                  const float l_ym = 0.2126f * R_out(ym, x) +
                                     0.7152f * G_out(ym, x) +
                                     0.0722f * B_out(ym, x);
                  const float l_yp = 0.2126f * R_out(yp, x) +
                                     0.7152f * G_out(yp, x) +
                                     0.0722f * B_out(yp, x);
                  if (std::isfinite(l_xm) && std::isfinite(l_xp) &&
                      std::isfinite(l_ym) && std::isfinite(l_yp)) {
                    gradient_sum += std::fabs(l_xp - l_xm) +
                                    std::fabs(l_yp - l_ym);
                    ++gradient_count;
                  }
                }
              }
              if (!values.empty()) {
                std::vector<float> median_values = values;
                tm.background = core::median_of(median_values);
                std::vector<float> noise_values = values;
                tm.noise = core::robust_sigma_mad(noise_values);
                tm.gradient_energy =
                    gradient_count > 0
                        ? static_cast<float>(gradient_sum /
                                             static_cast<double>(gradient_count))
                        : 0.0f;
                tm.contrast = tm.gradient_energy;
                tm.sharpness = tm.gradient_energy;
              }
              out.push_back(tm);
            }
            return out;
          };

      image::BGEConfig bge_cfg =
          tile_compile::runner::to_image_bge_config(cfg.bge);
      bge_cfg.max_workers = cfg.runtime_limits.parallel_workers;
      std::string mask_error;
      int rows = 0;
      int cols = 0;
      if (!assign_output_canvas_mask_for_rgb(bge_cfg.common_valid_mask, rows,
                                             cols, mask_error)) {
        emitter.phase_end(run_id, Phase::BGE, "error",
                          {{"reason", "output_canvas_mask_invalid"},
                           {"error", mask_error}},
                          log_file);
        emitter.run_end(run_id, false, "error", log_file,
                        {{"message", mask_error}});
        std::cerr << "Error: " << mask_error << std::endl;
        return 1;
      }
      bge_cfg.common_mask_rows = rows;
      bge_cfg.common_mask_cols = cols;
      tile_compile::runner::apply_autobge_exclusion_polygons(
          cfg.bge, rows, cols, bge_cfg);
      std::cout << "[BGE] Using reconstruction output canvas mask (" << cols
                << "x" << rows << ")" << std::endl;

      if (cfg.method == "aqmh" && bge_tile_metrics_cache.empty() &&
          !bge_tile_grid.tiles.empty()) {
        bge_tile_metrics_cache = build_aqmh_bge_tile_metrics(
            bge_tile_grid, bge_cfg.common_valid_mask, bge_cfg.common_mask_rows,
            bge_cfg.common_mask_cols);
      }

      // BGE requires method-specific tile sampling helpers.
      image::BGEDiagnostics bge_diag;
      bool bge_have_local_metrics =
          bge_tile_metrics_source == "classic_local_metrics";
      bool bge_have_tile_metrics = !bge_tile_metrics_cache.empty();
      bool bge_have_bge_grid = !bge_tile_grid.tiles.empty();
      bool bge_have_tile_data = bge_have_tile_metrics && bge_have_bge_grid;
      bool bge_metrics_tiles_match = false;
      bool bge_compact_tile_mode = false;
      if (bge_have_bge_grid && bge_tile_grid.tiles.size() == 1) {
        const auto &only_tile = bge_tile_grid.tiles.front();
        bge_compact_tile_mode =
            only_tile.x == 0 && only_tile.y == 0 &&
            only_tile.width == R_out.cols() &&
            only_tile.height == R_out.rows();
      }
      emitter.phase_progress_counts(run_id, Phase::BGE, 1, bge_progress_total,
                                    "collect_tiles", "BGE", log_file);

      if (cfg.bge.method == "autobge" ||
          (!bge_tile_metrics_cache.empty() && !bge_tile_grid.tiles.empty())) {
        const auto& tile_metrics_for_bge = bge_tile_metrics_cache;
        bge_metrics_tiles_match = cfg.bge.method == "autobge" ||
            (tile_metrics_for_bge.size() == bge_tile_grid.tiles.size());

        if (cfg.bge.method != "autobge" && bge_compact_tile_mode) {
          const std::string compact_reason = "compact_tile_mode_auto_disabled";
          std::ostringstream msg;
          msg << "BGE auto-disabled: compact-tile mode detected (single tile "
              << bge_tile_grid.tiles.front().width << "x"
              << bge_tile_grid.tiles.front().height
              << " covers full output canvas).";
          emitter.warning(run_id, msg.str(), log_file);
          std::cerr << "[BGE] Warning: " << msg.str() << std::endl;
          bge_diag.failure_reason = compact_reason;
        } else if (!bge_metrics_tiles_match) {
          std::cerr << "[BGE] Warning: tile metric/grid size mismatch (metrics="
                    << tile_metrics_for_bge.size() << ", tiles="
                    << bge_tile_grid.tiles.size() << "), skipping BGE"
                    << std::endl;
        } else {
          emitter.phase_progress_counts(run_id, Phase::BGE, 2, bge_progress_total,
                                        "fit_apply", "BGE", log_file);
          Matrix2Df R_bge = R_out;
          Matrix2Df G_bge = G_out;
          Matrix2Df B_bge = B_out;
          bool bge_success = image::apply_background_extraction(
              R_bge, G_bge, B_bge,
              tile_metrics_for_bge,
              bge_tile_grid,
              bge_cfg,
              &bge_diag);

          if (bge_success) {
            R_out = std::move(R_bge);
            G_out = std::move(G_bge);
            B_out = std::move(B_bge);
            have_successful_bge = true;
            io::write_fits_rgb(stacked_rgb_bge_linear_path, R_out, G_out, B_out,
                               first_hdr);
            write_output_rgb_snapshot(stacked_rgb_bge_path, R_out, G_out, B_out,
                                      first_hdr, "BGE");
            std::cerr << "[BGE] Background extraction completed successfully" << std::endl;
          } else {
            std::error_code ec;
            fs::remove(stacked_rgb_bge_linear_path, ec);
            fs::remove(stacked_rgb_bge_path, ec);
            std::cerr << "[BGE] Background extraction skipped or failed" << std::endl;
          }
        }
      } else {
        std::cerr << "[BGE] Warning: No tile metrics available, skipping BGE" << std::endl;
      }

      core::json bge_artifact = tile_compile::runner::bge_diag_to_json(
          bge_diag, cfg.bge.enabled, bge_have_tile_data, bge_metrics_tiles_match);
      bge_artifact["have_local_metrics"] = bge_have_local_metrics;
      bge_artifact["have_tile_metrics"] = bge_have_tile_metrics;
      bge_artifact["tile_metrics_source"] = bge_tile_metrics_source;
      bge_artifact["have_bge_grid"] = bge_have_bge_grid;
      bge_artifact["local_metrics_tiles"] =
          static_cast<int>(bge_tile_metrics_cache.size());
      bge_artifact["bge_grid_tiles"] = static_cast<int>(bge_tile_grid.tiles.size());
      bge_artifact["compact_tile_mode_detected"] = bge_compact_tile_mode;
      bge_artifact["auto_disabled_reason"] =
          bge_compact_tile_mode ? core::json("compact_tile_mode_auto_disabled")
                                : core::json(nullptr);
      bge_artifact["config"] = {
          {"enabled", cfg.bge.enabled},
          {"method", cfg.bge.method},
          {"autobge",
           {
               {"num_sample_points", cfg.bge.autobge.num_sample_points},
               {"poly_degree", cfg.bge.autobge.poly_degree},
               {"rbf_smooth", cfg.bge.autobge.rbf_smooth},
               {"downsample_scale", cfg.bge.autobge.downsample_scale},
               {"patch_size", cfg.bge.autobge.patch_size},
               {"patch_estimator", cfg.bge.autobge.patch_estimator},
               {"stretch_mode", cfg.bge.autobge.stretch_mode},
               {"stretch_target_median",
                cfg.bge.autobge.stretch_target_median},
               {"border_margin", cfg.bge.autobge.border_margin},
               {"bright_exclusion_fraction",
                cfg.bge.autobge.bright_exclusion_fraction},
               {"gradient_descent_max_iters",
                cfg.bge.autobge.gradient_descent_max_iters},
               {"random_seed", cfg.bge.autobge.random_seed},
               {"normalize_between_stages",
                cfg.bge.autobge.normalize_between_stages},
               {"apply_guards", cfg.bge.autobge.apply_guards},
               {"mono_mode", cfg.bge.autobge.mono_mode},
           }},
          {"classic",
           {
          {"sample_quantile", cfg.bge.sample_quantile},
          {"sample_estimator", cfg.bge.sample_estimator},
          {"min_sample_bg_value", cfg.bge.min_sample_bg_value},
          {"structure_thresh_percentile", cfg.bge.structure_thresh_percentile},
          {"min_tiles_per_cell", cfg.bge.min_tiles_per_cell},
          {"min_valid_sample_fraction_for_apply",
           cfg.bge.min_valid_sample_fraction_for_apply},
          {"min_valid_samples_for_apply", cfg.bge.min_valid_samples_for_apply},
          {"tile_weight_lambda_structure",
           cfg.bge.tile_weight_lambda_structure},
          {"mask",
           {
               {"star_dilate_px", cfg.bge.mask.star_dilate_px},
               {"sat_dilate_px", cfg.bge.mask.sat_dilate_px},
           }},
          {"grid",
           {
               {"N_g", cfg.bge.grid.N_g},
               {"G_min_px", cfg.bge.grid.G_min_px},
               {"G_max_fraction", cfg.bge.grid.G_max_fraction},
               {"insufficient_cell_strategy",
                cfg.bge.grid.insufficient_cell_strategy},
           }},
          {"fit",
           {
               {"method", cfg.bge.fit.method},
               {"robust_loss", cfg.bge.fit.robust_loss},
               {"huber_delta", cfg.bge.fit.huber_delta},
               {"irls_max_iterations", cfg.bge.fit.irls_max_iterations},
               {"irls_tolerance", cfg.bge.fit.irls_tolerance},
               {"polynomial_order", cfg.bge.fit.polynomial_order},
               {"rbf_phi", cfg.bge.fit.rbf_phi},
               {"rbf_mu_factor", cfg.bge.fit.rbf_mu_factor},
               {"rbf_lambda", cfg.bge.fit.rbf_lambda},
               {"rbf_epsilon", cfg.bge.fit.rbf_epsilon},
           }},
          {"autotune",
           {
               {"enabled", cfg.bge.autotune.enabled},
               {"max_evals", cfg.bge.autotune.max_evals},
               {"holdout_fraction", cfg.bge.autotune.holdout_fraction},
               {"alpha_flatness", cfg.bge.autotune.alpha_flatness},
               {"beta_roughness", cfg.bge.autotune.beta_roughness},
               {"strategy", cfg.bge.autotune.strategy},
           }},
          }},
      };
      fs::path bge_artifact_path = run_dir / "artifacts" / "bge.json";
      core::write_text(bge_artifact_path, bge_artifact.dump(2));
      emitter.phase_progress_counts(run_id, Phase::BGE, 3, bge_progress_total,
                                    "write_artifact", "BGE", log_file);

      core::json bge_phase_extra = {
          {"requested", cfg.bge.enabled},
          {"bge_method", cfg.bge.method},
          {"attempted", bge_diag.attempted},
          {"success", bge_diag.success},
          {"have_tile_data", bge_have_tile_data},
          {"metrics_tiles_match", bge_metrics_tiles_match},
          {"compact_tile_mode_detected", bge_compact_tile_mode},
          {"artifact", bge_artifact_path.string()},
      };
      if (!bge_have_tile_data) {
        bge_phase_extra["reason"] = "no_tile_data";
      } else if (bge_compact_tile_mode) {
        bge_phase_extra["reason"] = "compact_tile_mode_auto_disabled";
      } else if (!bge_metrics_tiles_match) {
        bge_phase_extra["reason"] = "tile_metric_grid_mismatch";
      } else if (bge_diag.attempted && !bge_diag.success) {
        bge_phase_extra["reason"] =
            bge_diag.failure_reason.empty() ? "fit_failed"
                                            : bge_diag.failure_reason;
      }

      emitter.phase_progress_counts(run_id, Phase::BGE, 4, bge_progress_total,
                                    "finalize", "BGE", log_file);

      emitter.phase_end(run_id, Phase::BGE,
                        bge_diag.success ? "ok" : "skipped",
                        bge_phase_extra, log_file);
    }
    if (abort_if_runtime_limit_exceeded("BGE")) {
      return 1;
    }

    // Phase 12: PCC (Photometric Color Calibration)
    bool have_successful_pcc = false;
    emitter.phase_start(run_id, Phase::PCC, "PCC", log_file);
    const fs::path pcc_input_rgb_path =
        have_successful_bge ? stacked_rgb_bge_linear_path : stacked_rgb_solve_path;
    const fs::path stacked_rgb_pcc_path = run_dir / "outputs" / "stacked_rgb_pcc.fits";
    {
      std::error_code ec;
      fs::remove(stacked_rgb_pcc_path, ec);
    }

    if (!cfg.pcc.enabled) {
      emitter.phase_end(run_id, Phase::PCC, "skipped",
                        {{"reason", "disabled"},
                         {"input_rgb", pcc_input_rgb_path.string()}},
                        log_file);
    } else if (!have_wcs) {
      emitter.phase_end(run_id, Phase::PCC, "skipped",
                        {{"reason", "no_wcs"},
                         {"input_rgb", pcc_input_rgb_path.string()}},
                        log_file);
    } else if (!have_rgb) {
      emitter.phase_end(run_id, Phase::PCC, "skipped",
                        {{"reason", "no_rgb_data"}}, log_file);
    } else {
      // Catalog source selection with fallback
      // auto: siril → vizier_gaia → vizier_apass
      double search_r = wcs.search_radius_deg();
      std::string source = cfg.pcc.source;
      double pcc_auto_fwhm_px = 0.0;
      std::string pcc_auto_fwhm_source = "disabled";
      tile_compile::runner::PCCCatalogQueryResult catalog =
          tile_compile::runner::query_pcc_catalog_stars(
              wcs, cfg.pcc, std::cerr, "[PCC]");
      std::string used_source = catalog.used_source;
      std::vector<astro::GaiaStar> stars = std::move(catalog.stars);

      if (stars.empty()) {
        emitter.phase_end(run_id, Phase::PCC, "skipped",
                          {{"reason", "no_catalog_stars"},
                           {"search_radius_deg", search_r},
                           {"source", source},
                           {"input_rgb", pcc_input_rgb_path.string()}},
                          log_file);
      } else {
        // Build PCC config from pipeline config
        astro::PCCConfig pcc_cfg =
            tile_compile::runner::to_astrometry_pcc_config(cfg.pcc);
        // PCC uses COMMON_OVERLAP only as an analysis/support mask. The visible
        // output canvas must remain intact inside the cropped output image.
        std::string mask_error;
        int rows = 0;
        int cols = 0;
        if (!assign_output_canvas_mask_for_rgb(pcc_cfg.output_valid_mask, rows,
                                               cols, mask_error)) {
          emitter.phase_end(run_id, Phase::PCC, "error",
                            {{"reason", "output_canvas_mask_invalid"},
                             {"error", mask_error},
                             {"input_rgb_bge", pcc_input_rgb_path.string()}},
                            log_file);
          emitter.run_end(run_id, false, "error", log_file,
                          {{"message", mask_error}});
          std::cerr << "Error: " << mask_error << std::endl;
          return 1;
        }
        const size_t expected_size =
            static_cast<size_t>(rows) * static_cast<size_t>(cols);
        if (common_valid_mask.size() != expected_size) {
          emitter.phase_end(run_id, Phase::PCC, "error",
                            {{"reason", "analysis_mask_invalid"},
                             {"error", "common_valid_mask size mismatch"},
                             {"mask_pixels",
                              static_cast<uint64_t>(common_valid_mask.size())},
                             {"expected_pixels",
                              static_cast<uint64_t>(expected_size)},
                             {"input_rgb_bge", pcc_input_rgb_path.string()}},
                            log_file);
          emitter.run_end(run_id, false, "error", log_file,
                          {{"message", "common_valid_mask size mismatch for PCC analysis"}});
          std::cerr << "Error: common_valid_mask size mismatch for PCC analysis"
                    << std::endl;
          return 1;
        }
        pcc_cfg.common_valid_mask = common_valid_mask;
        pcc_cfg.common_mask_rows = rows;
        pcc_cfg.common_mask_cols = cols;
        pcc_cfg.output_mask_rows = rows;
        pcc_cfg.output_mask_cols = cols;
        std::cout << "[PCC] Using COMMON_OVERLAP analysis mask and reconstruction output canvas mask ("
                  << cols << "x" << rows << ")" << std::endl;

        if (pcc_cfg.radii_mode == "auto_fwhm") {
          pcc_auto_fwhm_px = runner::resolve_pcc_auto_fwhm_px(
              R_out, G_out, B_out, have_seeing_fwhm,
              static_cast<double>(seeing_fwhm_med), &pcc_auto_fwhm_source);
          const double F = pcc_auto_fwhm_px;
          const double r_ap = std::max(static_cast<double>(pcc_cfg.min_aperture_px),
                                       pcc_cfg.aperture_fwhm_mult * F);
          const double r_in = std::max(r_ap + 1.0,
                                       pcc_cfg.annulus_inner_fwhm_mult * F);
          const double r_out = std::max(r_in + 2.0,
                                        pcc_cfg.annulus_outer_fwhm_mult * F);

          pcc_cfg.aperture_radius_px = r_ap;
          pcc_cfg.annulus_inner_px = r_in;
          pcc_cfg.annulus_outer_px = r_out;
          std::cout << "[PCC] auto_fwhm radii source: "
                    << pcc_auto_fwhm_source << " (F=" << F << ")"
                    << std::endl;
        }

        auto result = astro::run_pcc(R_out, G_out, B_out, wcs, stars, pcc_cfg);

        if (result.success) {
          have_successful_pcc = true;
          const auto chroma_speckle_stats =
              image::suppress_isolated_chroma_speckles_rgb_inplace(
                  R_out, G_out, B_out, &pcc_cfg.output_valid_mask,
                  pcc_cfg.output_mask_rows, pcc_cfg.output_mask_cols);
          if (chroma_speckle_stats.corrected_pixels > 0) {
            std::cout << "[PCC] Post-PCC chroma speckle suppressor corrected "
                      << chroma_speckle_stats.corrected_pixels
                      << " isolated pixels (candidates="
                      << chroma_speckle_stats.candidate_pixels << ")"
                      << std::endl;
          }
          if (cfg.chroma_denoise.enabled &&
              cfg.chroma_denoise.apply_stage == "post_pcc") {
            reconstruction::chroma_denoise_rgb_inplace(
                R_out, G_out, B_out, cfg.chroma_denoise);
          }

          // Keep per-channel PCC outputs linear, but write the visible RGB
          // snapshot with the configured output stretch semantics.
          io::write_fits_float(run_dir / "outputs" / "pcc_R.fit",
                               R_out, first_hdr);
          io::write_fits_float(run_dir / "outputs" / "pcc_G.fit",
                               G_out, first_hdr);
          io::write_fits_float(run_dir / "outputs" / "pcc_B.fit",
                               B_out, first_hdr);
          // stacked_rgb_pcc.fits must remain LINEAR float32 — it is the HMS input.
          // Never apply output_stretch here; HMS needs the original linear data.
          io::write_fits_rgb(stacked_rgb_pcc_path, R_out, G_out, B_out, first_hdr);

          core::json matrix_json = core::json::array();
          for (int r = 0; r < 3; ++r) {
            matrix_json.push_back({result.matrix[r][0],
                                   result.matrix[r][1],
                                   result.matrix[r][2]});
          }

          emitter.phase_end(run_id, Phase::PCC, "ok",
                            {{"stars_matched", result.n_stars_matched},
                             {"stars_used", result.n_stars_used},
                             {"residual_rms", result.residual_rms},
                             {"determinant", result.determinant},
                             {"condition_number", result.condition_number},
                             {"apply_mode", result.apply_mode},
                             {"apply_attenuation", pcc_cfg.apply_attenuation},
                             {"chroma_strength", pcc_cfg.chroma_strength},
                             {"k_max", pcc_cfg.k_max},
                             {"radii_mode", pcc_cfg.radii_mode},
                             {"auto_fwhm_px", pcc_auto_fwhm_px},
                             {"auto_fwhm_source", pcc_auto_fwhm_source},
                             {"aperture_radius_px", pcc_cfg.aperture_radius_px},
                             {"annulus_inner_px", pcc_cfg.annulus_inner_px},
                             {"annulus_outer_px", pcc_cfg.annulus_outer_px},
                             {"isolated_chroma_speckles_corrected",
                              chroma_speckle_stats.corrected_pixels},
                             {"isolated_chroma_speckle_candidates",
                              chroma_speckle_stats.candidate_pixels},
                             {"matrix", matrix_json},
                             {"source", used_source},
                             {"input_rgb_bge", pcc_input_rgb_path.string()}},
                            log_file);
        } else {
          emitter.phase_end(run_id, Phase::PCC, "skipped",
                            {{"reason", "fit_failed"},
                             {"error", result.error_message},
                             {"stars_matched", result.n_stars_matched},
                             {"stars_used", result.n_stars_used},
                             {"residual_rms", result.residual_rms},
                             {"determinant", result.determinant},
                             {"condition_number", result.condition_number},
                             {"apply_mode", result.apply_mode},
                             {"apply_attenuation", pcc_cfg.apply_attenuation},
                             {"chroma_strength", pcc_cfg.chroma_strength},
                             {"k_max", pcc_cfg.k_max},
                             {"radii_mode", pcc_cfg.radii_mode},
                             {"auto_fwhm_px", pcc_auto_fwhm_px},
                             {"auto_fwhm_source", pcc_auto_fwhm_source},
                             {"aperture_radius_px", pcc_cfg.aperture_radius_px},
                             {"annulus_inner_px", pcc_cfg.annulus_inner_px},
                             {"annulus_outer_px", pcc_cfg.annulus_outer_px},
                             {"source", used_source},
                             {"input_rgb_bge", pcc_input_rgb_path.string()}},
                            log_file);
        }
      }
    }

    if (abort_if_runtime_limit_exceeded("PCC")) {
      return 1;
    }

    // Phase 13: HyperMetric Stretch (final nonlinear RGB stretch after PCC)
    emitter.phase_start(run_id, Phase::HYPERMETRIC_STRETCH,
                        "HYPERMETRIC_STRETCH", log_file);
    if (!cfg.hypermetric_stretch.enabled) {
      emitter.phase_end(run_id, Phase::HYPERMETRIC_STRETCH, "skipped",
                        {{"reason", "disabled"}}, log_file);
    } else if (!have_rgb) {
      emitter.phase_end(run_id, Phase::HYPERMETRIC_STRETCH, "skipped",
                        {{"reason", "no_rgb_data"}}, log_file);
    } else if (!have_wcs) {
      emitter.phase_end(run_id, Phase::HYPERMETRIC_STRETCH, "skipped",
                        {{"reason", "missing_successful_astrometry"}},
                        log_file);
    } else if (!have_successful_pcc) {
      emitter.phase_end(run_id, Phase::HYPERMETRIC_STRETCH, "skipped",
                        {{"reason", "missing_successful_pcc"},
                         {"require_successful_pcc", true}},
                        log_file);
    } else {
      try {
        image::HyperMetricStretchConfig hms_cfg =
            to_image_hms_config(cfg.hypermetric_stretch);
        const int hms_rows = static_cast<int>(R_out.rows());
        const int hms_cols = static_cast<int>(R_out.cols());
        const size_t hms_pixels =
            static_cast<size_t>(hms_rows) * static_cast<size_t>(hms_cols);
        const std::vector<uint8_t> *hms_statistics_mask = nullptr;
        const std::vector<uint8_t> *hms_output_mask = nullptr;
        if (common_valid_mask.size() == hms_pixels) {
          hms_statistics_mask = &common_valid_mask;
        }
        if (output_valid_mask.size() == hms_pixels) {
          hms_output_mask = &output_valid_mask;
        }

        auto hms_diag = image::run_hypermetric_stretch_rgb(
            R_out, G_out, B_out, hms_cfg, hms_statistics_mask, hms_rows,
            hms_cols, hms_output_mask);
        if (!hms_diag.success) {
          emitter.phase_end(run_id, Phase::HYPERMETRIC_STRETCH, "error",
                            {{"reason", "stretch_failed"},
                             {"error", hms_diag.error_message}},
                            log_file);
          emitter.run_end(run_id, false, "error", log_file,
                          {{"message", hms_diag.error_message}});
          return 1;
        }

        io::FitsHeader hms_hdr = first_hdr;
        hms_hdr.set("HMS", true);
        hms_hdr.set("HMSVER", std::string("1"));
        hms_hdr.set("HMSMODE", hms_cfg.mode);
        hms_hdr.set("HMSPROF", hms_diag.profile);
        hms_hdr.set("HMSWR", static_cast<double>(hms_diag.weights_r));
        hms_hdr.set("HMSWG", static_cast<double>(hms_diag.weights_g));
        hms_hdr.set("HMSWB", static_cast<double>(hms_diag.weights_b));
        hms_hdr.set("HMSANCH", static_cast<double>(hms_diag.anchor));
        hms_hdr.set("HMSLOGD", static_cast<double>(hms_diag.log_d));
        hms_hdr.set("HMSB", static_cast<double>(hms_diag.protect_b));
        hms_hdr.set("HMSTGBG", static_cast<double>(hms_diag.target_bg));
        hms_hdr.set("HMSCONV", static_cast<double>(hms_diag.convergence_power));
        hms_hdr.set("HMSSTAR", static_cast<double>(hms_diag.star_pressure));

        fs::path hms_rgb_path(hms_cfg.output_rgb);
        if (hms_rgb_path.is_relative()) {
          hms_rgb_path = run_dir / "outputs" / hms_rgb_path;
        }
        {
          std::error_code ec;
          fs::remove(hms_rgb_path, ec);
        }
        io::write_fits_rgb(hms_rgb_path, R_out, G_out, B_out, hms_hdr);
        if (hms_cfg.write_channels) {
          io::write_fits_float(run_dir / "outputs" / "hms_R.fit", R_out,
                               hms_hdr);
          io::write_fits_float(run_dir / "outputs" / "hms_G.fit", G_out,
                               hms_hdr);
          io::write_fits_float(run_dir / "outputs" / "hms_B.fit", B_out,
                               hms_hdr);
        }

        emitter.phase_end(
            run_id, Phase::HYPERMETRIC_STRETCH, "ok",
            {{"output_rgb", hms_rgb_path.string()},
             {"profile", hms_diag.profile},
             {"profile_source", hms_diag.profile_source},
             {"weights_r", hms_diag.weights_r},
             {"weights_g", hms_diag.weights_g},
             {"weights_b", hms_diag.weights_b},
             {"anchor", hms_diag.anchor},
             {"log_d", hms_diag.log_d},
             {"target_bg", hms_diag.target_bg},
             {"protect_b", hms_diag.protect_b},
             {"convergence_power", hms_diag.convergence_power},
             {"star_pressure", hms_diag.star_pressure},
             {"color_strategy", hms_diag.color_strategy},
             {"color_grip", hms_diag.color_grip},
             {"shadow_convergence", hms_diag.shadow_convergence},
             {"black_clip_percent", hms_diag.black_clip_percent},
             {"white_clip_percent", hms_diag.white_clip_percent},
             {"input_stage", "pcc"}},
            log_file);
      } catch (const std::exception &e) {
        emitter.phase_end(run_id, Phase::HYPERMETRIC_STRETCH, "error",
                          {{"reason", "exception"}, {"error", e.what()}},
                          log_file);
        emitter.run_end(run_id, false, "error", log_file,
                        {{"message", std::string("HYPERMETRIC_STRETCH exception: ") + e.what()}});
        return 1;
      }
    }

    if (abort_if_runtime_limit_exceeded("HYPERMETRIC_STRETCH")) {
      return 1;
    }

    // --- Memory release: all large image buffers before final exit ---
    R_out.resize(0, 0);
    G_out.resize(0, 0);
    B_out.resize(0, 0);
    recon.resize(0, 0);
    recon_R.resize(0, 0);
    recon_G.resize(0, 0);
    recon_B.resize(0, 0);
    { std::vector<std::vector<float>>().swap(local_weights); }
    { std::vector<Matrix2Df>().swap(synthetic_frames); }

    // Phase 14: DONE
    emitter.phase_start(run_id, Phase::DONE, "DONE", log_file);
    emitter.phase_end(run_id, Phase::DONE, "ok", {}, log_file);

    if (run_validation_failed) {
      emitter.run_end(run_id, false, "validation_failed", log_file,
                      {{"message", "Pipeline completed but validation failed"}});

      std::cout << "Pipeline completed with validation_failed" << std::endl;
      return 1;
    }

    emitter.run_end(run_id, true, "ok", log_file);

    std::cout << "Pipeline completed successfully" << std::endl;
    return 0;
  }

}
