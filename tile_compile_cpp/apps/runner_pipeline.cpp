#include "runner_forward_drizzle.hpp"
#include "tile_compile/reconstruction/forward_drizzle.hpp"
#include "runner_pipeline.hpp"

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/build_info.hpp"
#include "tile_compile/core/acceleration.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/config/legacy_config_migration.hpp"
#include "tile_compile/core/input_class_policy.hpp"
#include "tile_compile/core/pipeline_contract.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/image/background_extraction.hpp"
#include "tile_compile/image/cfa_processing.hpp"
#include "tile_compile/image/hypermetric_stretch.hpp"
#include "tile_compile/image/normalization.hpp"
#include "tile_compile/image/processing.hpp"
#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/metrics/linearity.hpp"
#include "tile_compile/metrics/metrics.hpp"
#include "tile_compile/pipeline/adaptive_tile_grid.hpp"
#include "tile_compile/astrometry/wcs.hpp"
#include "tile_compile/astrometry/gaia_catalog.hpp"
#include "tile_compile/astrometry/photometric_color_cal.hpp"

#include "runner_phase_metrics.hpp"
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
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <opencv2/opencv.hpp>

#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {
using tile_compile::ColorMode;
using tile_compile::Matrix2Df;

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
using tile_compile::runner::estimate_astap_sensor_fov_deg;
using tile_compile::runner::resolve_astap_binary_path;
using tile_compile::runner::shell_quote;
using tile_compile::runner::system_cmd;

using NormalizationScales = image::NormalizationScales;

core::json provenance_file_json(const fs::path &path,
                                const fs::path &source_path = {}) {
  std::error_code ec;
  fs::path normalized = fs::absolute(path, ec);
  if (ec) normalized = path.lexically_normal();
  if (!fs::is_regular_file(normalized, ec) || ec) {
    throw std::runtime_error("Provenance input is not a readable file: " +
                             normalized.string());
  }
  const auto size_bytes = fs::file_size(normalized, ec);
  if (ec) {
    throw std::runtime_error("Cannot determine provenance file size: " +
                             normalized.string() + ": " + ec.message());
  }
  fs::path recorded_path = source_path;
  if (recorded_path.empty()) {
    recorded_path = fs::weakly_canonical(normalized, ec);
    if (ec) recorded_path = normalized;
  }
  core::json result = {{"path", recorded_path.string()},
                       {"size_bytes", size_bytes},
                       {"sha256", core::sha256_file(normalized)}};
  if (recorded_path != normalized) result["materialized_input"] = true;
  return result;
}

core::json make_run_provenance(
    const fs::path &config_snapshot,
    const std::vector<fs::path> &ordered_frames) {
  core::json entries = core::json::array();
  std::unordered_map<std::string, core::json> origin_manifests;
  for (size_t index = 0; index < ordered_frames.size(); ++index) {
    const fs::path &frame = ordered_frames[index];
    const fs::path origin_manifest_path =
        frame.parent_path() / ".tile_compile_input_origins.json";
    const std::string manifest_key = origin_manifest_path.string();
    auto manifest_it = origin_manifests.find(manifest_key);
    if (manifest_it == origin_manifests.end()) {
      core::json origin_manifest = core::json::object();
      if (fs::is_regular_file(origin_manifest_path)) {
        try {
          origin_manifest = core::json::parse(
              core::read_text(origin_manifest_path));
          if (!origin_manifest.is_object()) {
            origin_manifest = core::json::object();
          }
        } catch (...) {
          origin_manifest = core::json::object();
        }
      }
      manifest_it = origin_manifests.emplace(
          manifest_key, std::move(origin_manifest)).first;
    }

    fs::path source_path;
    const std::string filename = frame.filename().string();
    if (manifest_it->second.contains(filename) &&
        manifest_it->second[filename].is_string()) {
      source_path = manifest_it->second[filename].get<std::string>();
    }
    core::json entry = provenance_file_json(frame, source_path);
    entry["index"] = index;
    entries.push_back(std::move(entry));
  }
  const std::string canonical_manifest = entries.dump();
  const std::vector<uint8_t> manifest_bytes(canonical_manifest.begin(),
                                             canonical_manifest.end());
  const core::json config_file = provenance_file_json(config_snapshot);
  return {
      {"schema_version", 1},
      {"created_at", core::get_iso_timestamp()},
      {"build", core::build_info_json(true)},
      {"config", config_file},
      // Milestone M0: identifies which reconstruction contract wrote this run.
      // 0 == legacy / cutover-in-progress (see core/pipeline_contract.hpp).
      {"pipeline_contract_version", core::kPipelineContractVersionActive},
      {"pipeline_contract_label",
       core::pipeline_contract_label(core::kPipelineContractVersionActive)},
      {"input_manifest",
       {{"ordering", "lexicographic_path_after_max_frames"},
        {"entry_count", entries.size()},
        {"entries", std::move(entries)},
        {"sha256", core::sha256_bytes(manifest_bytes)}}}};
}

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

// The web backend pre-claims a run directory before launching `reconstruct`:
// it writes the immutable config snapshot plus the run-start claim/provenance
// artifacts, then points --config at that snapshot. A run directory containing
// only those pre-start files is still fresh; any other entry means a prior run
// or foreign content already occupies it.
bool run_dir_contains_only_backend_claim(const fs::path &run_dir) {
  std::error_code ec;
  if (!fs::is_directory(run_dir, ec) || ec) return false;
  static const std::unordered_set<std::string> kAllowedTopLevel = {
      "config.yaml", ".config.yaml.pending", "artifacts",
      ".run_start_config_claim.lock"};
  static const std::unordered_set<std::string> kAllowedArtifacts = {
      "run_start_config_claim.json", ".run_start_config_claim.json.pending",
      "pi_run_provenance.json", "config_revisions"};
  for (const auto &entry : fs::directory_iterator(run_dir, ec)) {
    const std::string name = entry.path().filename().string();
    if (!kAllowedTopLevel.count(name)) return false;
    if (name != "artifacts") continue;
    std::error_code art_ec;
    for (const auto &artifact :
         fs::directory_iterator(entry.path(), art_ec)) {
      if (!kAllowedArtifacts.count(artifact.path().filename().string()))
        return false;
    }
    if (art_ec) return false;
  }
  return !ec;
}

} // namespace

/// @brief Runs pipeline command.
/// @details Part of the production runner pipeline that coordinates scan, registration, metrics, reconstruction, stacking, astrometry, BGE, and PCC phases; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
int run_pipeline_command(const std::string &config_path, const std::string &input_dir,
                const std::string &runs_dir, const std::string &project_root,
                const std::string &run_id_override,
                bool dry_run, int max_frames,
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
  // M0 / plan section 6.5: the production run path migrates a legacy config
  // (reject method/engine fail-closed, strip removed structural blocks) and
  // records the changes in config_migration.json below. The frozen
  // legacy-reference runner intentionally loads the config verbatim.
  config::ConfigMigrationReport config_migration_report;
  if (use_stdin_config) {
    std::ostringstream ss;
    ss << std::cin.rdbuf();
    cfg_text = ss.str();
    if (cfg_text.empty()) {
      std::cerr << "Error: --stdin provided but no config YAML received"
                << std::endl;
      return 1;
    }
    proj_root =
        project_root.empty() ? fs::current_path() : fs::path(project_root);
  } else {
    if (!fs::exists(cfg_path)) {
      std::cerr << "Error: Config file not found: " << config_path << std::endl;
      return 1;
    }
    cfg_text = core::read_text(cfg_path);
    proj_root = project_root.empty() ? core::resolve_project_root(cfg_path)
                                     : fs::path(project_root);
  }
  try {
    cfg = config::Config::from_yaml_text_migrated(cfg_text,
                                                  config_migration_report);
    cfg.validate();
  } catch (const std::exception &e) {
    std::cerr << "Error: failed to load/validate config: " << e.what()
              << std::endl;
    return 1;
  }


  // Forward-drizzle resolves CPU workers against its shared memory plan.
  // Keep the configured count for fresh runs as well as reconstruction resume.

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
  {
    if (run_id.empty() || fs::path(run_id).filename() != run_id || run_id == "." || run_id == "..") {
      std::cerr << "FORWARD_RUN_INVALID_ID" << std::endl;
      return 1;
    }
    fs::create_directories(runs);
    if (!fs::create_directory(run_dir) &&
        !run_dir_contains_only_backend_claim(run_dir)) {
      std::cerr << "FORWARD_RUN_REQUIRES_FRESH_DIRECTORY" << std::endl;
      return 1;
    }
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

  const fs::path config_snapshot_path = run_dir / "config.yaml";
  if (use_stdin_config) {
    std::ofstream out(config_snapshot_path, std::ios::out);
    if (!out) {
      std::cerr << "Error: cannot write run config snapshot: "
                << config_snapshot_path << std::endl;
      return 1;
    }
    out << cfg_text;
  } else {
    core::copy_config(cfg_path, config_snapshot_path);
  }

  // M0 / plan section 6.5: record any legacy-config migration that was applied
  // so the strip is auditable. Written even when nothing changed is not useful,
  // so only emit it when the migration touched the config.
  if (config_migration_report.applied) {
    core::write_text(run_dir / "artifacts" / "config_migration.json",
                     config_migration_report.to_json_string());
  }

  core::json run_provenance;
  const fs::path run_provenance_path =
      run_dir / "artifacts" / "run_provenance.json";
  try {
    run_provenance = make_run_provenance(config_snapshot_path, frames);
    run_provenance["execution_scope"] = "forward_drizzle_m1_m3";
    core::write_text(run_provenance_path, run_provenance.dump(2));
  } catch (const std::exception &e) {
    std::cerr << "Error: cannot establish immutable run provenance: "
              << e.what() << std::endl;
    return 1;
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
                     {"config_snapshot", config_snapshot_path.string()},
                     {"config_sha256", run_provenance["config"]["sha256"]},
                     {"input_dir", input_dir},
                     {"input_manifest_sha256",
                      run_provenance["input_manifest"]["sha256"]},
                     {"run_dir", run_dir.string()},
                     {"provenance_path", run_provenance_path.string()},
                     {"build_id", run_provenance["build"]["build_id"]},
                     {"git_sha", run_provenance["build"]["source"]["git_sha"]},
                     {"git_dirty",
                      run_provenance["build"]["source"]["git_dirty"]},
                     {"binary_sha256",
                      run_provenance["build"]["binary"]["sha256"]},
                     {"frames_discovered", frames.size()},
                     {"pipeline_contract_version",
                      core::kPipelineContractVersionActive},
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

  // M0 / plan section 3.1.1: the single-method pipeline supports only OSC raw
  // (with a known Bayer pattern) and MONO raw. Already-debayered RGB cubes and
  // unclassifiable colour metadata are rejected fail-closed here so no such
  // input is silently forced down a wrong compute path.
  {
    const core::InputClassDecision input_decision =
        core::classify_input_for_single_method(detected_mode, detected_bayer);
    if (!core::input_class_accepted(input_decision)) {
      const std::string msg =
          core::input_class_rejection_message(input_decision);
      emitter.phase_end(run_id, Phase::SCAN_INPUT, "error",
                        {{"error", msg},
                         {"input_dir", input_dir},
                         {"color_mode", detected_mode_str},
                         {"bayer_pattern", detected_bayer_str}},
                        log_file);
      emitter.run_end(
          run_id, false, "error", log_file,
          {{"message", std::string("Error during SCAN_INPUT: ") + msg}});
      std::cerr << "Error during SCAN_INPUT: " << msg << std::endl;
      return 1;
    }
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
  // `auto` intentionally delegates Bayer-pattern selection to FITS metadata;
  // a detected concrete pattern is therefore not a configuration mismatch.
  // Compare parsed concrete patterns so case/whitespace variations such as
  // `gbrg` are treated equivalently to the canonical `GBRG` spelling.
  std::string normalized_cfg_bayer = cfg.data.bayer_pattern;
  normalized_cfg_bayer.erase(
      normalized_cfg_bayer.begin(),
      std::find_if(normalized_cfg_bayer.begin(), normalized_cfg_bayer.end(),
                   [](unsigned char c) { return !std::isspace(c); }));
  normalized_cfg_bayer.erase(
      std::find_if(normalized_cfg_bayer.rbegin(), normalized_cfg_bayer.rend(),
                   [](unsigned char c) { return !std::isspace(c); })
          .base(),
      normalized_cfg_bayer.end());
  std::transform(normalized_cfg_bayer.begin(), normalized_cfg_bayer.end(),
                 normalized_cfg_bayer.begin(), [](unsigned char c) {
                   return static_cast<char>(std::toupper(c));
                 });
  const bool cfg_bayer_is_auto = normalized_cfg_bayer == "AUTO";
  if (detected_mode == ColorMode::OSC &&
      !cfg.data.bayer_pattern.empty() && !cfg_bayer_is_auto &&
      detected_bayer != BayerPattern::UNKNOWN &&
      (cfg_bayer == BayerPattern::UNKNOWN || cfg_bayer != detected_bayer)) {
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

  {
    try {
      registration::RegistrationSamplingPlan resources;
      resources.source_width = resources.canvas_width_native = width;
      resources.source_height = resources.canvas_height_native = height;
      resources.color_mode = detected_mode;
      resources.bayer_pattern = detected_bayer;
      resources.frames.resize(frames.size());
      auto limits = cfg.reconstruction.drizzle;
      limits.internal_scale = 1;
      limits.chunk_rows = height;
      if (!limits.memory_budget_mb) limits.memory_budget_mb = std::max(1, cfg.runtime_limits.memory_budget);
      const size_t pixels = static_cast<size_t>(width) * height;
      if (frames.size() > std::numeric_limits<size_t>::max() / sizeof(float) / pixels)
        throw std::runtime_error("FORWARD_RUN_MEMORY_SIZE_OVERFLOW");
      // Plan 11.14.6 P4: the only frame-count-dependent RAM held concurrently
      // with the reconstruction working set is the in-memory registration
      // proxy set (RunnerFrameCache::registration_proxies_). Each proxy is a
      // 2x2 downsample (build_registration_proxy -> cfa_green_proxy_downsample2x2
      // / downsample2x2_mean), i.e. pixels/4 floats --- NOT a full-resolution
      // plane. Normalized full frames are disk-backed (DiskCacheFrameStore) and
      // the geometry cache reader keeps only the row index (~KiB/frame). Charge
      // the real proxy footprint; the earlier pixels*4*N term over-estimated it
      // 4x and needlessly rejected 100+ frame runs (dev protocol s30.61).
      const size_t proxy_bytes_per_frame = (pixels / 4) * sizeof(float);
      // When the configured budget cannot hold the retained proxy set, grow
      // it in 1 GiB steps within the live memory headroom instead of
      // aborting; each step is logged as a run warning.
      reconstruction::plan_drizzle_memory_autogrow(
          resources, limits, 128, proxy_bytes_per_frame * frames.size(), true,
          [&](const std::string &msg) {
            emitter.warning(run_id, msg, log_file);
          });
      // Persist the (possibly grown) budget so the reconstruction stages
      // resolve the same effective limit instead of re-inheriting
      // runtime_limits.
      cfg.reconstruction.drizzle.memory_budget_mb = limits.memory_budget_mb;
    } catch (const std::exception &e) {
      emitter.run_end(run_id, false, "error", log_file, {{"message", e.what()}});
      return 1;
    }
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
  const auto frame_cache = phase_metrics_ctx.frame_cache;

  {
    try {
      if (!frame_cache) throw std::runtime_error("FORWARD_RUN_REQUIRES_NORMALIZED_CACHE");
      for (size_t i = 0; i < frames.size(); ++i)
        if (!frame_cache->has_normalized(i))
          throw std::runtime_error("FORWARD_RUN_NORMALIZATION_INCOMPLETE");
      frame_cache->release_registration_proxies();
      if (!runner::run_phase_registration_prewarp(run_id, cfg, frames, run_dir,
              height, width, detected_mode, detected_bayer_str, frame_cache,
              norm_scales, frame_metrics, global_weights, first_header,
              acceleration, emitter, log_file, phase_registration_ctx, true)) {
        emitter.run_end(run_id, false, "error", log_file, {{"message", "registration failed"}});
        return 1;
      }
      return runner::run_forward_drizzle_stages(run_id, cfg, run_dir,
          phase_registration_ctx.sampling_plan, frame_cache.get(), emitter, log_file) ? 0 : 1;
    } catch (const std::exception &e) {
      emitter.run_end(run_id, false, "error", log_file, {{"message", e.what()}});
      return 1;
    }
  }
}
