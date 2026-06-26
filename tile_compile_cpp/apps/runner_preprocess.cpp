#include "runner_preprocess.hpp"

#include "runner_phase_preprocess_pipeline.hpp"
#include "runner_phase_quality_analysis.hpp"
#include "runner_shared.hpp"

#include "tile_compile/astrometry/photometric_color_cal.hpp"
#include "tile_compile/astrometry/wcs.hpp"
#include "tile_compile/image/hypermetric_stretch.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/image/background_extraction.hpp"
#include "tile_compile/image/cfa_processing.hpp"
#include "tile_compile/image/normalization.hpp"
#include "tile_compile/image/processing.hpp"
#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/metrics/metrics.hpp"
#include "tile_compile/metrics/tile_metrics.hpp"
#include "tile_compile/pipeline/adaptive_tile_grid.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <numeric>
#include <sstream>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

using tile_compile::runner::parallel_for_indices;
using tile_compile::runner::default_parallel_workers;

using json = nlohmann::json;
namespace core = tile_compile::core;
namespace astro = tile_compile::astrometry;
namespace config = tile_compile::config;
namespace image = tile_compile::image;
namespace io = tile_compile::io;
namespace metrics = tile_compile::metrics;
namespace prep = tile_compile::preprocessing;
namespace runner = tile_compile::runner;
using tile_compile::Matrix2Df;
using tile_compile::Tile;
using tile_compile::TileGrid;
using tile_compile::TileMetrics;
using tile_compile::TileType;



std::string read_all_stdin() {
  std::ostringstream ss;
  ss << std::cin.rdbuf();
  return ss.str();
}

std::string json_string(const json& j, const std::string& key,
                        const std::string& fallback = "") {
  return j.contains(key) && j[key].is_string() ? j[key].get<std::string>()
                                               : fallback;
}

float json_float(const json& j, const std::string& key, float fallback) {
  return j.contains(key) && j[key].is_number() ? j[key].get<float>() : fallback;
}

int json_int(const json& j, const std::string& key, int fallback) {
  return j.contains(key) && j[key].is_number_integer() ? j[key].get<int>() : fallback;
}

bool json_bool(const json& j, const std::string& key, bool fallback) {
  return j.contains(key) && j[key].is_boolean() ? j[key].get<bool>() : fallback;
}

json json_object(const json& j, const std::string& key) {
  return j.contains(key) && j[key].is_object() ? j[key] : json::object();
}

prep::Config parse_preprocessing_config(const json& j) {
  prep::Config cfg;
  cfg.mode = json_string(j, "mode", cfg.mode);
  cfg.lights_dir = json_string(j, "lights_dir", cfg.lights_dir);
  cfg.bias_dir = json_string(j, "bias_dir", cfg.bias_dir);
  cfg.darks_dir = json_string(j, "darks_dir", cfg.darks_dir);
  cfg.flats_dir = json_string(j, "flats_dir", cfg.flats_dir);
  cfg.darkflats_dir = json_string(j, "darkflats_dir", cfg.darkflats_dir);
  cfg.input_mode = json_string(j, "input_mode", cfg.input_mode);
  cfg.raw_formats = json_string(j, "raw_formats", cfg.raw_formats);
  cfg.bayer_pattern = json_string(j, "bayer_pattern", cfg.bayer_pattern);
  cfg.cfa_mode = json_string(j, "cfa_mode", cfg.cfa_mode);
  cfg.mono_mode = json_string(j, "mono_mode", cfg.mono_mode);
  cfg.registration_reference =
      json_string(j, "registration_reference", cfg.registration_reference);

  if (j.contains("rejection") && j["rejection"].is_object()) {
    const auto& r = j["rejection"];
    cfg.rejection.method = json_string(r, "method", cfg.rejection.method);
    cfg.rejection.low = json_float(r, "low", cfg.rejection.low);
    cfg.rejection.high = json_float(r, "high", cfg.rejection.high);
    cfg.rejection.max_iters = json_int(r, "max_iters", cfg.rejection.max_iters);
    cfg.rejection.min_fraction = json_float(r, "min_fraction", cfg.rejection.min_fraction);
  }
  if (j.contains("quality_filter") && j["quality_filter"].is_object()) {
    const auto& q = j["quality_filter"];
    cfg.quality_filter.mode = json_string(q, "mode", cfg.quality_filter.mode);
    cfg.quality_filter.min_stars = json_int(q, "min_stars", cfg.quality_filter.min_stars);
    cfg.quality_filter.max_fwhm_sigma =
        json_float(q, "max_fwhm_sigma", cfg.quality_filter.max_fwhm_sigma);
    cfg.quality_filter.max_eccentricity =
        json_float(q, "max_eccentricity", cfg.quality_filter.max_eccentricity);
    cfg.quality_filter.min_correlation =
        json_float(q, "min_correlation", cfg.quality_filter.min_correlation);
    if (q.contains("manual_overrides") && q["manual_overrides"].is_object()) {
      cfg.manual_frame_overrides.clear();
      for (auto it = q["manual_overrides"].begin(); it != q["manual_overrides"].end(); ++it) {
        prep::ManualFrameOverride item;
        const std::string key = it.key();
        char* end = nullptr;
        const long idx = std::strtol(key.c_str(), &end, 10);
        if (end != key.c_str() && end != nullptr && *end == '\0') item.index = static_cast<int>(idx);
        else item.filename = key;
        if (it.value().is_boolean()) {
          item.include = it.value().get<bool>();
        } else if (it.value().is_object()) {
          item.include = json_bool(it.value(), "include", item.include);
          if (it.value().contains("index") && it.value()["index"].is_number_integer()) {
            item.index = it.value()["index"].get<int>();
          }
          item.filename = json_string(it.value(), "filename", item.filename);
        }
        cfg.manual_frame_overrides.push_back(item);
      }
    }
  }
  if (j.contains("stacking") && j["stacking"].is_object()) {
    const auto& s = j["stacking"];
    cfg.stacking.normalization = json_string(s, "normalization", cfg.stacking.normalization);
    cfg.stacking.weighting = json_string(s, "weighting", cfg.stacking.weighting);
    cfg.stacking.cosmetic_correction =
        json_bool(s, "cosmetic_correction", cfg.stacking.cosmetic_correction);
    cfg.stacking.cosmetic_correction_sigma =
        json_float(s, "cosmetic_correction_sigma", cfg.stacking.cosmetic_correction_sigma);
    cfg.stacking.per_frame_cosmetic_correction =
        json_bool(s, "per_frame_cosmetic_correction", cfg.stacking.per_frame_cosmetic_correction);
    cfg.stacking.per_frame_cosmetic_correction_sigma =
        json_float(s, "per_frame_cosmetic_correction_sigma",
                   cfg.stacking.per_frame_cosmetic_correction_sigma);
    if (s.contains("sigma_clip") && s["sigma_clip"].is_object()) {
      const auto& sc = s["sigma_clip"];
      cfg.rejection.low = json_float(sc, "sigma_low", cfg.rejection.low);
      cfg.rejection.high = json_float(sc, "sigma_high", cfg.rejection.high);
      cfg.rejection.max_iters = json_int(sc, "max_iters", cfg.rejection.max_iters);
      cfg.rejection.min_fraction = json_float(sc, "min_fraction", cfg.rejection.min_fraction);
    }
  }
  if (j.contains("calibration") && j["calibration"].is_object()) {
    const auto& c = j["calibration"];
    cfg.calibration.use_bias = json_bool(c, "use_bias", cfg.calibration.use_bias);
    cfg.calibration.use_dark = json_bool(c, "use_dark", cfg.calibration.use_dark);
    cfg.calibration.use_flat = json_bool(c, "use_flat", cfg.calibration.use_flat);
    cfg.calibration.bias_use_master =
        json_bool(c, "bias_use_master", cfg.calibration.bias_use_master);
    cfg.calibration.dark_use_master =
        json_bool(c, "dark_use_master", cfg.calibration.dark_use_master);
    cfg.calibration.flat_use_master =
        json_bool(c, "flat_use_master", cfg.calibration.flat_use_master);
    cfg.calibration.darkflat_use_master =
        json_bool(c, "darkflat_use_master", cfg.calibration.darkflat_use_master);
    cfg.calibration.dark_auto_select =
        json_bool(c, "dark_auto_select", cfg.calibration.dark_auto_select);
    cfg.calibration.dark_match_use_temp =
        json_bool(c, "dark_match_use_temp", cfg.calibration.dark_match_use_temp);
    cfg.calibration.dark_match_exposure_tolerance_percent =
        json_float(c, "dark_match_exposure_tolerance_percent",
                   cfg.calibration.dark_match_exposure_tolerance_percent);
    cfg.calibration.dark_match_temp_tolerance_c =
        json_float(c, "dark_match_temp_tolerance_c",
                   cfg.calibration.dark_match_temp_tolerance_c);
    cfg.calibration.bias_master = json_string(c, "bias_master", cfg.calibration.bias_master);
    cfg.calibration.dark_master = json_string(c, "dark_master", cfg.calibration.dark_master);
    cfg.calibration.flat_master = json_string(c, "flat_master", cfg.calibration.flat_master);
    cfg.calibration.darkflat_master =
        json_string(c, "darkflat_master", cfg.calibration.darkflat_master);
    cfg.calibration.pattern = json_string(c, "pattern", cfg.calibration.pattern);
  }
  if (j.contains("postprocess") && j["postprocess"].is_object()) {
    const auto& p = j["postprocess"];
    cfg.postprocess.astrometry = json_bool(p, "astrometry", cfg.postprocess.astrometry);
    cfg.postprocess.bge = json_bool(p, "bge", cfg.postprocess.bge);
    cfg.postprocess.pcc = json_bool(p, "pcc", cfg.postprocess.pcc);
    cfg.postprocess.hypermetric_stretch =
        json_bool(p, "hypermetric_stretch", cfg.postprocess.hypermetric_stretch);
  }
  if (j.contains("astrometry") && j["astrometry"].is_object()) {
    const auto& a = j["astrometry"];
    cfg.has_astrometry_config = true;
    cfg.astrometry.enabled = json_bool(a, "enabled", cfg.astrometry.enabled);
    cfg.astrometry.astap_bin = json_string(a, "astap_bin", cfg.astrometry.astap_bin);
    cfg.astrometry.astap_data_dir =
        json_string(a, "astap_data_dir", cfg.astrometry.astap_data_dir);
    cfg.astrometry.search_radius = json_int(a, "search_radius", cfg.astrometry.search_radius);
  }
  if (j.contains("bge") && j["bge"].is_object()) {
    const auto& b = j["bge"];
    cfg.has_bge_config = true;
    cfg.bge.enabled = json_bool(b, "enabled", cfg.bge.enabled);
    if (b.contains("method") && b["method"].is_string()) {
      cfg.bge.method = b["method"].get<std::string>();
      cfg.bge.enabled = (cfg.bge.method != "none");
    } else {
      cfg.bge.method = cfg.bge.enabled ? "classic" : "none";
    }
    const json autobge = json_object(b, "autobge");
    cfg.bge.autobge.num_sample_points =
        json_int(autobge, "num_sample_points", cfg.bge.autobge.num_sample_points);
    cfg.bge.autobge.poly_degree =
        json_int(autobge, "poly_degree", cfg.bge.autobge.poly_degree);
    cfg.bge.autobge.rbf_smooth =
        json_float(autobge, "rbf_smooth", cfg.bge.autobge.rbf_smooth);
    cfg.bge.autobge.downsample_scale =
        json_int(autobge, "downsample_scale", cfg.bge.autobge.downsample_scale);
    cfg.bge.autobge.patch_size =
        json_int(autobge, "patch_size", cfg.bge.autobge.patch_size);
    cfg.bge.autobge.patch_estimator =
        json_string(autobge, "patch_estimator", cfg.bge.autobge.patch_estimator);
    cfg.bge.autobge.stretch_mode =
        json_string(autobge, "stretch_mode", cfg.bge.autobge.stretch_mode);
    cfg.bge.autobge.stretch_target_median =
        json_float(autobge, "stretch_target_median",
                   cfg.bge.autobge.stretch_target_median);
    cfg.bge.autobge.border_margin =
        json_int(autobge, "border_margin", cfg.bge.autobge.border_margin);
    cfg.bge.autobge.bright_exclusion_fraction =
        json_float(autobge, "bright_exclusion_fraction",
                   cfg.bge.autobge.bright_exclusion_fraction);
    cfg.bge.autobge.gradient_descent_max_iters =
        json_int(autobge, "gradient_descent_max_iters",
                 cfg.bge.autobge.gradient_descent_max_iters);
    cfg.bge.autobge.random_seed =
        json_int(autobge, "random_seed", cfg.bge.autobge.random_seed);
    cfg.bge.autobge.normalize_between_stages =
        json_bool(autobge, "normalize_between_stages",
                  cfg.bge.autobge.normalize_between_stages);
    cfg.bge.autobge.apply_guards =
        json_bool(autobge, "apply_guards", cfg.bge.autobge.apply_guards);
    cfg.bge.autobge.mono_mode =
        json_string(autobge, "mono_mode", cfg.bge.autobge.mono_mode);
    cfg.bge.sample_quantile = json_float(b, "sample_quantile", cfg.bge.sample_quantile);
    cfg.bge.sample_estimator = json_string(b, "sample_estimator", cfg.bge.sample_estimator);
    cfg.bge.min_sample_bg_value = json_float(b, "min_sample_bg_value", cfg.bge.min_sample_bg_value);
    cfg.bge.structure_thresh_percentile =
        json_float(b, "structure_thresh_percentile", cfg.bge.structure_thresh_percentile);
    cfg.bge.min_tiles_per_cell = json_int(b, "min_tiles_per_cell", cfg.bge.min_tiles_per_cell);
    cfg.bge.min_valid_sample_fraction_for_apply =
        json_float(b, "min_valid_sample_fraction_for_apply",
                   cfg.bge.min_valid_sample_fraction_for_apply);
    cfg.bge.min_valid_samples_for_apply =
        json_int(b, "min_valid_samples_for_apply", cfg.bge.min_valid_samples_for_apply);
    const json mask = json_object(b, "mask");
    cfg.bge.mask.star_dilate_px = json_int(mask, "star_dilate_px", cfg.bge.mask.star_dilate_px);
    cfg.bge.mask.sat_dilate_px = json_int(mask, "sat_dilate_px", cfg.bge.mask.sat_dilate_px);
    const json grid = json_object(b, "grid");
    cfg.bge.grid.N_g = json_int(grid, "N_g", cfg.bge.grid.N_g);
    cfg.bge.grid.G_min_px = json_int(grid, "G_min_px", cfg.bge.grid.G_min_px);
    cfg.bge.grid.G_max_fraction = json_float(grid, "G_max_fraction", cfg.bge.grid.G_max_fraction);
    cfg.bge.grid.insufficient_cell_strategy =
        json_string(grid, "insufficient_cell_strategy", cfg.bge.grid.insufficient_cell_strategy);
    const json fit = json_object(b, "fit");
    cfg.bge.fit.method = json_string(fit, "method", cfg.bge.fit.method);
    cfg.bge.fit.robust_loss = json_string(fit, "robust_loss", cfg.bge.fit.robust_loss);
    cfg.bge.fit.huber_delta = json_float(fit, "huber_delta", cfg.bge.fit.huber_delta);
    cfg.bge.fit.irls_max_iterations =
        json_int(fit, "irls_max_iterations", cfg.bge.fit.irls_max_iterations);
    cfg.bge.fit.irls_tolerance = json_float(fit, "irls_tolerance", cfg.bge.fit.irls_tolerance);
    cfg.bge.fit.polynomial_order = json_int(fit, "polynomial_order", cfg.bge.fit.polynomial_order);
    cfg.bge.fit.rbf_phi = json_string(fit, "rbf_phi", cfg.bge.fit.rbf_phi);
    cfg.bge.fit.rbf_mu_factor = json_float(fit, "rbf_mu_factor", cfg.bge.fit.rbf_mu_factor);
    cfg.bge.fit.rbf_lambda = json_float(fit, "rbf_lambda", cfg.bge.fit.rbf_lambda);
    cfg.bge.fit.rbf_epsilon = json_float(fit, "rbf_epsilon", cfg.bge.fit.rbf_epsilon);
    const json autotune = json_object(b, "autotune");
    cfg.bge.autotune.enabled = json_bool(autotune, "enabled", cfg.bge.autotune.enabled);
    cfg.bge.autotune.max_evals = json_int(autotune, "max_evals", cfg.bge.autotune.max_evals);
    cfg.bge.autotune.holdout_fraction =
        json_float(autotune, "holdout_fraction", cfg.bge.autotune.holdout_fraction);
    cfg.bge.autotune.alpha_flatness =
        json_float(autotune, "alpha_flatness", cfg.bge.autotune.alpha_flatness);
    cfg.bge.autotune.beta_roughness =
        json_float(autotune, "beta_roughness", cfg.bge.autotune.beta_roughness);
    cfg.bge.autotune.strategy = json_string(autotune, "strategy", cfg.bge.autotune.strategy);
    cfg.bge.tile_weight_lambda_structure =
        json_float(b, "tile_weight_lambda_structure", cfg.bge.tile_weight_lambda_structure);
  }
  if (j.contains("tile") && j["tile"].is_object()) {
    const auto& t = j["tile"];
    cfg.has_tile_config = true;
    cfg.tile.size_factor = json_int(t, "size_factor", cfg.tile.size_factor);
    cfg.tile.min_size = json_int(t, "min_size", cfg.tile.min_size);
    cfg.tile.max_divisor = json_int(t, "max_divisor", cfg.tile.max_divisor);
    cfg.tile.overlap_fraction = json_float(t, "overlap_fraction", cfg.tile.overlap_fraction);
    cfg.tile.star_min_count = json_int(t, "star_min_count", cfg.tile.star_min_count);
    cfg.tile.star_soft_count = json_int(t, "star_soft_count", cfg.tile.star_soft_count);
  }
  if (j.contains("pcc") && j["pcc"].is_object()) {
    const auto& p = j["pcc"];
    cfg.has_pcc_config = true;
    cfg.pcc.enabled = json_bool(p, "enabled", cfg.pcc.enabled);
    cfg.pcc.source = json_string(p, "source", cfg.pcc.source);
    cfg.pcc.mag_limit = json_float(p, "mag_limit", cfg.pcc.mag_limit);
    cfg.pcc.mag_bright_limit = json_float(p, "mag_bright_limit", cfg.pcc.mag_bright_limit);
    cfg.pcc.aperture_radius_px = json_float(p, "aperture_radius_px", cfg.pcc.aperture_radius_px);
    cfg.pcc.annulus_inner_px = json_float(p, "annulus_inner_px", cfg.pcc.annulus_inner_px);
    cfg.pcc.annulus_outer_px = json_float(p, "annulus_outer_px", cfg.pcc.annulus_outer_px);
    cfg.pcc.min_stars = json_int(p, "min_stars", cfg.pcc.min_stars);
    cfg.pcc.sigma_clip = json_float(p, "sigma_clip", cfg.pcc.sigma_clip);
    cfg.pcc.background_model = json_string(p, "background_model", cfg.pcc.background_model);
    cfg.pcc.max_condition_number =
        json_float(p, "max_condition_number", cfg.pcc.max_condition_number);
    cfg.pcc.max_residual_rms = json_float(p, "max_residual_rms", cfg.pcc.max_residual_rms);
    cfg.pcc.radii_mode = json_string(p, "radii_mode", cfg.pcc.radii_mode);
    cfg.pcc.aperture_fwhm_mult = json_float(p, "aperture_fwhm_mult", cfg.pcc.aperture_fwhm_mult);
    cfg.pcc.annulus_inner_fwhm_mult =
        json_float(p, "annulus_inner_fwhm_mult", cfg.pcc.annulus_inner_fwhm_mult);
    cfg.pcc.annulus_outer_fwhm_mult =
        json_float(p, "annulus_outer_fwhm_mult", cfg.pcc.annulus_outer_fwhm_mult);
    cfg.pcc.min_aperture_px = json_float(p, "min_aperture_px", cfg.pcc.min_aperture_px);
    cfg.pcc.siril_catalog_dir = json_string(p, "siril_catalog_dir", cfg.pcc.siril_catalog_dir);
    cfg.pcc.apply_attenuation = json_bool(p, "apply_attenuation", cfg.pcc.apply_attenuation);
    cfg.pcc.chroma_strength = json_float(p, "chroma_strength", cfg.pcc.chroma_strength);
    cfg.pcc.k_max = json_float(p, "k_max", cfg.pcc.k_max);
    cfg.pcc.background_neutralization_mode =
        json_string(p, "background_neutralization_mode",
                    cfg.pcc.background_neutralization_mode);
  }
  if (j.contains("hypermetric_stretch") && j["hypermetric_stretch"].is_object()) {
    const auto& h = j["hypermetric_stretch"];
    cfg.hypermetric_stretch.require_successful_pcc =
        json_bool(h, "require_successful_pcc", cfg.hypermetric_stretch.require_successful_pcc);
    cfg.hypermetric_stretch.mode = json_string(h, "mode", cfg.hypermetric_stretch.mode);
    cfg.hypermetric_stretch.sensor_profile =
        json_string(h, "sensor_profile", cfg.hypermetric_stretch.sensor_profile);
    cfg.hypermetric_stretch.fallback_profile =
        json_string(h, "fallback_profile", cfg.hypermetric_stretch.fallback_profile);
    cfg.hypermetric_stretch.adaptive_anchor =
        json_bool(h, "adaptive_anchor", cfg.hypermetric_stretch.adaptive_anchor);
    cfg.hypermetric_stretch.target_bg =
        json_float(h, "target_bg", cfg.hypermetric_stretch.target_bg);
    cfg.hypermetric_stretch.protect_b =
        json_float(h, "protect_b", cfg.hypermetric_stretch.protect_b);
    cfg.hypermetric_stretch.convergence_power =
        json_float(h, "convergence_power", cfg.hypermetric_stretch.convergence_power);
    cfg.hypermetric_stretch.log_d_mode =
        json_string(h, "log_d_mode", cfg.hypermetric_stretch.log_d_mode);
    cfg.hypermetric_stretch.fixed_log_d =
        json_float(h, "fixed_log_d", cfg.hypermetric_stretch.fixed_log_d);
    cfg.hypermetric_stretch.color_strategy =
        json_string(h, "color_strategy", cfg.hypermetric_stretch.color_strategy);
    cfg.hypermetric_stretch.fixed_color_strategy =
        json_float(h, "fixed_color_strategy", cfg.hypermetric_stretch.fixed_color_strategy);
    cfg.hypermetric_stretch.color_grip =
        json_float(h, "color_grip", cfg.hypermetric_stretch.color_grip);
    cfg.hypermetric_stretch.shadow_convergence =
        json_float(h, "shadow_convergence", cfg.hypermetric_stretch.shadow_convergence);
    cfg.hypermetric_stretch.linear_expansion =
        json_float(h, "linear_expansion", cfg.hypermetric_stretch.linear_expansion);
    cfg.hypermetric_stretch.write_channels =
        json_bool(h, "write_channels", cfg.hypermetric_stretch.write_channels);
    cfg.hypermetric_stretch.output_rgb =
        json_string(h, "output_rgb", cfg.hypermetric_stretch.output_rgb);
  }
  if (j.contains("report") && j["report"].is_object()) {
    const auto& r = j["report"];
    cfg.report.detailed = json_bool(r, "detailed", cfg.report.detailed);
    if (r.contains("formats") && r["formats"].is_array()) {
      cfg.report.formats.clear();
      for (const auto& item : r["formats"]) {
        if (item.is_string()) cfg.report.formats.push_back(item.get<std::string>());
      }
    }
  }
  if (j.contains("runtime_limits") && j["runtime_limits"].is_object()) {
    const auto& r = j["runtime_limits"];
    cfg.runtime_limits.parallel_workers =
        json_int(r, "parallel_workers", cfg.runtime_limits.parallel_workers);
    cfg.runtime_limits.memory_budget =
        json_int(r, "memory_budget", cfg.runtime_limits.memory_budget);
  }
  prep::validate(cfg);
  return cfg;
}

json config_to_json(const prep::Config& cfg) {
  json manual_overrides = json::object();
  for (const auto& item : cfg.manual_frame_overrides) {
    const std::string key = item.index >= 0 ? std::to_string(item.index) : item.filename;
    manual_overrides[key] = {
        {"index", item.index},
        {"filename", item.filename},
        {"include", item.include},
    };
  }
  return {
      {"mode", cfg.mode},
      {"lights_dir", cfg.lights_dir},
      {"bias_dir", cfg.bias_dir},
      {"darks_dir", cfg.darks_dir},
      {"flats_dir", cfg.flats_dir},
      {"darkflats_dir", cfg.darkflats_dir},
      {"input_mode", cfg.input_mode},
      {"raw_formats", cfg.raw_formats},
      {"bayer_pattern", cfg.bayer_pattern},
      {"cfa_mode", cfg.cfa_mode},
      {"mono_mode", cfg.mono_mode},
      {"registration_reference", cfg.registration_reference},
      {"calibration", {
          {"use_bias", cfg.calibration.use_bias},
          {"use_dark", cfg.calibration.use_dark},
          {"use_flat", cfg.calibration.use_flat},
          {"bias_use_master", cfg.calibration.bias_use_master},
          {"dark_use_master", cfg.calibration.dark_use_master},
          {"flat_use_master", cfg.calibration.flat_use_master},
          {"darkflat_use_master", cfg.calibration.darkflat_use_master},
          {"dark_auto_select", cfg.calibration.dark_auto_select},
          {"dark_match_use_temp", cfg.calibration.dark_match_use_temp},
          {"dark_match_exposure_tolerance_percent", cfg.calibration.dark_match_exposure_tolerance_percent},
          {"dark_match_temp_tolerance_c", cfg.calibration.dark_match_temp_tolerance_c},
          {"bias_master", cfg.calibration.bias_master},
          {"dark_master", cfg.calibration.dark_master},
          {"flat_master", cfg.calibration.flat_master},
          {"darkflat_master", cfg.calibration.darkflat_master},
          {"pattern", cfg.calibration.pattern},
      }},
      {"rejection", {{"method", cfg.rejection.method}, {"low", cfg.rejection.low}, {"high", cfg.rejection.high}, {"max_iters", cfg.rejection.max_iters}, {"min_fraction", cfg.rejection.min_fraction}}},
      {"quality_filter", {
          {"mode", cfg.quality_filter.mode},
          {"min_stars", cfg.quality_filter.min_stars},
          {"max_fwhm_sigma", cfg.quality_filter.max_fwhm_sigma},
          {"max_eccentricity", cfg.quality_filter.max_eccentricity},
          {"min_correlation", cfg.quality_filter.min_correlation},
          {"manual_overrides", manual_overrides},
      }},
      {"stacking", {
          {"normalization", cfg.stacking.normalization},
          {"weighting", cfg.stacking.weighting},
          {"cosmetic_correction", cfg.stacking.cosmetic_correction},
          {"cosmetic_correction_sigma", cfg.stacking.cosmetic_correction_sigma},
          {"per_frame_cosmetic_correction", cfg.stacking.per_frame_cosmetic_correction},
          {"per_frame_cosmetic_correction_sigma", cfg.stacking.per_frame_cosmetic_correction_sigma},
      }},
      {"postprocess", {
          {"astrometry", cfg.postprocess.astrometry},
          {"bge", cfg.postprocess.bge},
          {"pcc", cfg.postprocess.pcc},
          {"hypermetric_stretch", cfg.postprocess.hypermetric_stretch},
      }},
      {"tile", {
          {"size_factor", cfg.tile.size_factor},
          {"min_size", cfg.tile.min_size},
          {"max_divisor", cfg.tile.max_divisor},
          {"overlap_fraction", cfg.tile.overlap_fraction},
          {"star_min_count", cfg.tile.star_min_count},
          {"star_soft_count", cfg.tile.star_soft_count},
      }},
      {"hypermetric_stretch", {
          {"require_successful_pcc", cfg.hypermetric_stretch.require_successful_pcc},
          {"mode", cfg.hypermetric_stretch.mode},
          {"sensor_profile", cfg.hypermetric_stretch.sensor_profile},
          {"fallback_profile", cfg.hypermetric_stretch.fallback_profile},
          {"adaptive_anchor", cfg.hypermetric_stretch.adaptive_anchor},
          {"target_bg", cfg.hypermetric_stretch.target_bg},
          {"protect_b", cfg.hypermetric_stretch.protect_b},
          {"convergence_power", cfg.hypermetric_stretch.convergence_power},
          {"log_d_mode", cfg.hypermetric_stretch.log_d_mode},
          {"fixed_log_d", cfg.hypermetric_stretch.fixed_log_d},
          {"color_strategy", cfg.hypermetric_stretch.color_strategy},
          {"fixed_color_strategy", cfg.hypermetric_stretch.fixed_color_strategy},
          {"color_grip", cfg.hypermetric_stretch.color_grip},
          {"shadow_convergence", cfg.hypermetric_stretch.shadow_convergence},
          {"linear_expansion", cfg.hypermetric_stretch.linear_expansion},
          {"write_channels", cfg.hypermetric_stretch.write_channels},
          {"output_rgb", cfg.hypermetric_stretch.output_rgb},
      }},
      {"report", {{"detailed", cfg.report.detailed}, {"formats", cfg.report.formats}}},
      {"runtime_limits", {
          {"parallel_workers", cfg.runtime_limits.parallel_workers},
          {"memory_budget", cfg.runtime_limits.memory_budget},
      }},
  };
}

std::vector<metrics::FrameStarMetrics> measure_stars_for_quality(
    const runner::PreprocessPipelineContext& ctx) {
  const size_t n = ctx.effective_frames.size();
  std::vector<metrics::FrameStarMetrics> out(n);
  int ref_stars = 0;
  if (ctx.reference_frame_index >= 0 &&
      static_cast<size_t>(ctx.reference_frame_index) < n &&
      ctx.prewarped_frames.has_data(static_cast<size_t>(ctx.reference_frame_index))) {
    const auto ref = ctx.prewarped_frames.load(static_cast<size_t>(ctx.reference_frame_index));
    out[static_cast<size_t>(ctx.reference_frame_index)] = metrics::measure_frame_stars(ref, 0);
    ref_stars = out[static_cast<size_t>(ctx.reference_frame_index)].star_count;
  }
  const int workers = default_parallel_workers(n, 0);
  parallel_for_indices(n, workers, [&](size_t i) {
    if (static_cast<int>(i) == ctx.reference_frame_index) return;
    if (!ctx.prewarped_frames.has_data(i)) return;
    const auto img = ctx.prewarped_frames.load(i);
    out[i] = metrics::measure_frame_stars(img, ref_stars);
  });
  return out;
}

struct PreprocessStackResult {
  fs::path stacked_linear_path;
  fs::path stacked_rgb_path;
  fs::path diagnostics_path;
  int frames_used = 0;
  int rejected_input_frames = 0;
  int rejected_samples = 0;
  int64_t total_samples = 0;
  std::string method;
  std::string weighting;
};

struct PreprocessPostprocessResult {
  json phases = json::array();
  json artifacts = json::array();
  fs::path astrometry_wcs_path;
  fs::path bge_rgb_path;
  fs::path bge_diag_path;
  fs::path pcc_rgb_path;
  fs::path pcc_diag_path;
  fs::path hms_rgb_path;
  fs::path hms_diag_path;
};

float weighted_mean(const std::vector<float>& values,
                    const std::vector<float>& weights) {
  double sum = 0.0;
  double wsum = 0.0;
  for (size_t i = 0; i < values.size(); ++i) {
    const float v = values[i];
    const float w = i < weights.size() ? weights[i] : 1.0f;
    if (!std::isfinite(v) || !std::isfinite(w) || w <= 0.0f) continue;
    sum += static_cast<double>(v) * static_cast<double>(w);
    wsum += static_cast<double>(w);
  }
  return wsum > 0.0 ? static_cast<float>(sum / wsum) : 0.0f;
}

float median_value(std::vector<float> values) {
  values.erase(std::remove_if(values.begin(), values.end(), [](float v) {
                 return !std::isfinite(v);
               }), values.end());
  if (values.empty()) return 0.0f;
  std::sort(values.begin(), values.end());
  const size_t mid = values.size() / 2;
  if (values.size() % 2 == 1) return values[mid];
  return 0.5f * (values[mid - 1] + values[mid]);
}

std::pair<float, float> mean_stddev(const std::vector<float>& values) {
  std::vector<float> finite;
  finite.reserve(values.size());
  for (float v : values) if (std::isfinite(v)) finite.push_back(v);
  if (finite.empty()) return {0.0f, 0.0f};
  const double mean = std::accumulate(finite.begin(), finite.end(), 0.0) /
                      static_cast<double>(finite.size());
  double var = 0.0;
  for (float v : finite) {
    const double d = static_cast<double>(v) - mean;
    var += d * d;
  }
  var /= static_cast<double>(finite.size());
  return {static_cast<float>(mean), static_cast<float>(std::sqrt(var))};
}

float reduce_pixel(std::vector<float>& values,
                   const std::vector<float>& weights,
                   const prep::RejectionConfig& rejection,
                   int& rejected_samples,
                   std::vector<float>& kept_values,
                   std::vector<float>& kept_weights,
                   std::vector<float>& next_values,
                   std::vector<float>& next_weights) {
  if (values.empty()) return 0.0f;
  const std::string method = rejection.method;
  if (method == "median") {
    return median_value(values);
  }

  if (method == "winsor") {
    auto [mean, sigma] = mean_stddev(values);
    if (sigma > 0.0f) {
      const float lo = mean - rejection.low * sigma;
      const float hi = mean + rejection.high * sigma;
      for (float& v : values) {
        if (!std::isfinite(v)) continue;
        if (v < lo) {
          v = lo;
          ++rejected_samples;
        } else if (v > hi) {
          v = hi;
          ++rejected_samples;
        }
      }
    }
    return weighted_mean(values, weights);
  }

  kept_values.clear();
  kept_weights.clear();
  kept_values.reserve(values.size());
  kept_weights.reserve(weights.size());
  for (size_t i = 0; i < values.size(); ++i) {
    const float v = values[i];
    if (!std::isfinite(v)) continue;
    kept_values.push_back(v);
    kept_weights.push_back(i < weights.size() ? weights[i] : 1.0f);
  }
  const size_t min_keep = static_cast<size_t>(
      std::ceil(static_cast<float>(kept_values.size()) * rejection.min_fraction));
  for (int iter = 0; iter < std::max(1, rejection.max_iters); ++iter) {
    if (kept_values.size() <= std::max<size_t>(2, min_keep)) break;
    auto [mean, sigma] = mean_stddev(kept_values);
    if (sigma <= 0.0f || !std::isfinite(sigma)) break;
    const float lo = mean - rejection.low * sigma;
    const float hi = mean + rejection.high * sigma;
    next_values.clear();
    next_weights.clear();
    next_values.reserve(kept_values.size());
    next_weights.reserve(kept_weights.size());
    int iter_rejected = 0;
    for (size_t i = 0; i < kept_values.size(); ++i) {
      const float v = kept_values[i];
      if (v < lo || v > hi) {
        ++iter_rejected;
        continue;
      }
      next_values.push_back(v);
      next_weights.push_back(i < kept_weights.size() ? kept_weights[i] : 1.0f);
    }
    if (iter_rejected == 0 || next_values.empty() || next_values.size() < min_keep) break;
    rejected_samples += iter_rejected;
    kept_values.swap(next_values);
    kept_weights.swap(next_weights);
  }
  if (kept_values.empty()) return weighted_mean(values, weights);
  return weighted_mean(kept_values, kept_weights);
}

Matrix2Df stack_frame_planes(const std::vector<Matrix2Df>& frames,
                             const std::vector<float>& weights,
                             const prep::RejectionConfig& rejection,
                             int workers,
                             int& rejected_samples,
                             int64_t& total_samples,
                             const std::function<void(int, int)>& row_progress) {
  if (frames.empty()) return {};
  const int rows = static_cast<int>(frames.front().rows());
  const int cols = static_cast<int>(frames.front().cols());
  Matrix2Df stacked(rows, cols);
  std::atomic<int> rejected_total{0};
  std::atomic<int64_t> sample_total{0};
  std::mutex progress_mutex;
  parallel_for_indices(static_cast<size_t>(rows),
                       default_parallel_workers(static_cast<size_t>(rows), workers),
                       [&](size_t row_idx) {
    const int y = static_cast<int>(row_idx);
    std::vector<float> values;
    std::vector<float> kept_values;
    std::vector<float> kept_weights;
    std::vector<float> next_values;
    std::vector<float> next_weights;
    values.reserve(frames.size());
    kept_values.reserve(frames.size());
    kept_weights.reserve(frames.size());
    next_values.reserve(frames.size());
    next_weights.reserve(frames.size());
    int rejected_local = 0;
    int64_t total_local = 0;
    for (int x = 0; x < cols; ++x) {
      values.clear();
      for (const auto& frame : frames) values.push_back(frame(y, x));
      stacked(y, x) = reduce_pixel(values, weights, rejection, rejected_local,
                                   kept_values, kept_weights, next_values, next_weights);
      total_local += static_cast<int64_t>(frames.size());
    }
    rejected_total.fetch_add(rejected_local, std::memory_order_relaxed);
    sample_total.fetch_add(total_local, std::memory_order_relaxed);
    if (row_progress) {
      std::lock_guard<std::mutex> lock(progress_mutex);
      row_progress(y, rows);
    }
  });
  rejected_samples += rejected_total.load(std::memory_order_relaxed);
  total_samples += sample_total.load(std::memory_order_relaxed);
  return stacked;
}

Matrix2Df stack_frame_ptrs(const std::vector<const float*>& frames,
                           int rows,
                           int cols,
                           const std::vector<float>& weights,
                           const prep::RejectionConfig& rejection,
                           int workers,
                           int& rejected_samples,
                           int64_t& total_samples) {
  if (frames.empty() || rows <= 0 || cols <= 0) return {};
  Matrix2Df stacked(rows, cols);
  std::atomic<int> rejected_total{0};
  std::atomic<int64_t> sample_total{0};
  parallel_for_indices(static_cast<size_t>(rows),
                       default_parallel_workers(static_cast<size_t>(rows), workers),
                       [&](size_t row_idx) {
    const int y = static_cast<int>(row_idx);
    const size_t row_off = static_cast<size_t>(y) * static_cast<size_t>(cols);
    std::vector<float> values;
    std::vector<float> kept_values;
    std::vector<float> kept_weights;
    std::vector<float> next_values;
    std::vector<float> next_weights;
    values.reserve(frames.size());
    kept_values.reserve(frames.size());
    kept_weights.reserve(frames.size());
    next_values.reserve(frames.size());
    next_weights.reserve(frames.size());
    int rejected_local = 0;
    int64_t total_local = 0;
    for (int x = 0; x < cols; ++x) {
      values.clear();
      const size_t offset = row_off + static_cast<size_t>(x);
      for (const float* frame : frames) {
        values.push_back(frame ? frame[offset] : std::numeric_limits<float>::quiet_NaN());
      }
      stacked(y, x) = reduce_pixel(values, weights, rejection, rejected_local,
                                   kept_values, kept_weights, next_values, next_weights);
      total_local += static_cast<int64_t>(frames.size());
    }
    rejected_total.fetch_add(rejected_local, std::memory_order_relaxed);
    sample_total.fetch_add(total_local, std::memory_order_relaxed);
  });
  rejected_samples += rejected_total.load(std::memory_order_relaxed);
  total_samples += sample_total.load(std::memory_order_relaxed);
  return stacked;
}

std::vector<float> stack_weights_for_accepted(const prep::Config& cfg,
                                              const runner::QualityAnalysisContext& qa) {
  std::vector<float> weights;
  weights.reserve(qa.accepted_indices.size());
  for (int idx : qa.accepted_indices) {
    float w = 1.0f;
    if (cfg.stacking.weighting == "quality") {
      const auto& rec = qa.records[static_cast<size_t>(idx)];
      w = std::max(1.0e-6f, rec.quality_score);
      if (!std::isfinite(w)) w = 1.0f;
    }
    weights.push_back(w);
  }
  return weights;
}

runner::FrameSubBatchPlan raw_stack_sub_batch_plan(
    int rows,
    int cols,
    int channels,
    size_t frame_count,
    const prep::RuntimeLimitsConfig& runtime_limits) {
  if (rows <= 0 || cols <= 0 || frame_count == 0) return {};
  return runner::compute_memory_capped_frame_sub_batch(
      frame_count,
      static_cast<size_t>(rows) * static_cast<size_t>(cols),
      channels,
      1,
      runtime_limits.memory_budget);
}

void accumulate_batch_stack(const Matrix2Df& batch,
                            float batch_weight,
                            Matrix2Df& accum,
                            Matrix2Df& wsum) {
  if (batch.size() <= 0) return;
  if (accum.size() <= 0) {
    accum = Matrix2Df::Zero(batch.rows(), batch.cols());
    wsum = Matrix2Df::Zero(batch.rows(), batch.cols());
  }
  for (int y = 0; y < batch.rows(); ++y) {
    for (int x = 0; x < batch.cols(); ++x) {
      const float v = batch(y, x);
      if (std::isfinite(v)) {
        accum(y, x) += v * batch_weight;
        wsum(y, x) += batch_weight;
      }
    }
  }
}

Matrix2Df finalize_batch_accum(const Matrix2Df& accum, const Matrix2Df& wsum) {
  if (accum.size() <= 0 || wsum.size() != accum.size()) return {};
  Matrix2Df out(accum.rows(), accum.cols());
  for (int y = 0; y < accum.rows(); ++y) {
    for (int x = 0; x < accum.cols(); ++x) {
      out(y, x) = wsum(y, x) > 0.0f ? accum(y, x) / wsum(y, x)
                                    : std::numeric_limits<float>::quiet_NaN();
    }
  }
  return out;
}

PreprocessStackResult run_preprocess_stacking(
    const std::string& run_id,
    const prep::Config& cfg,
    const runner::PreprocessPipelineContext& pp,
    const runner::QualityAnalysisContext& qa,
    const fs::path& run_dir,
    core::EventEmitter& emitter,
    std::ostream& event_out) {
  PreprocessStackResult result;
  result.method = cfg.rejection.method;
  result.weighting = cfg.stacking.weighting;
  result.rejected_input_frames = static_cast<int>(qa.rejected_indices.size());

  emitter.phase_start(run_id, prep::phase_to_string(prep::Phase::STACKING), event_out);
  const fs::path output_dir = run_dir / "outputs";
  const fs::path artifact_dir = run_dir / "artifacts" / "preprocess";
  fs::create_directories(output_dir);
  fs::create_directories(artifact_dir);

  std::vector<int> accepted;
  accepted.reserve(qa.accepted_indices.size());
  for (int idx : qa.accepted_indices) {
    if (idx >= 0 && static_cast<size_t>(idx) < pp.prewarped_frames.size() &&
        pp.prewarped_frames.has_data(static_cast<size_t>(idx))) {
      accepted.push_back(idx);
    }
  }
  if (accepted.empty()) {
    emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::STACKING), "error",
                      {{"error", "no accepted registered frames to stack"}}, event_out);
    emitter.run_end(run_id, false, "error", event_out);
    throw std::runtime_error("no accepted registered frames to stack");
  }

  const int rows = pp.prewarped_frames.rows();
  const int cols = pp.prewarped_frames.cols();
  const std::vector<float> weights = stack_weights_for_accepted(cfg, qa);

  const auto linear_batch_plan =
      raw_stack_sub_batch_plan(rows, cols, 1, accepted.size(), cfg.runtime_limits);
  const size_t linear_sub_batch =
      linear_batch_plan.frame_sub_batch_size > 0 ? linear_batch_plan.frame_sub_batch_size
                                                 : accepted.size();
  Matrix2Df linear_accum;
  Matrix2Df linear_wsum;
  for (size_t batch_start = 0; batch_start < accepted.size(); batch_start += linear_sub_batch) {
    const size_t batch_end = std::min(batch_start + linear_sub_batch, accepted.size());
    const size_t batch_size = batch_end - batch_start;
    std::vector<const float*> batch_frames;
    std::vector<float> batch_weights;
    batch_frames.reserve(batch_size);
    batch_weights.reserve(batch_size);
    for (size_t i = batch_start; i < batch_end; ++i) {
      const float* frame_ptr =
          pp.prewarped_frames.frame_data(static_cast<size_t>(accepted[i]));
      if (frame_ptr == nullptr) continue;
      batch_frames.push_back(frame_ptr);
      batch_weights.push_back(i < weights.size() ? weights[i] : 1.0f);
    }
    if (batch_frames.empty()) continue;
    int batch_rejected = 0;
    int64_t batch_total = 0;
    Matrix2Df batch_stack = stack_frame_ptrs(batch_frames, rows, cols, batch_weights,
                                             cfg.rejection,
                                             cfg.runtime_limits.parallel_workers,
                                             batch_rejected, batch_total);
    result.rejected_samples += batch_rejected;
    result.total_samples += batch_total;
    accumulate_batch_stack(batch_stack, static_cast<float>(batch_frames.size()),
                           linear_accum, linear_wsum);
    pp.prewarped_frames.clear_mappings();
    emitter.phase_progress(run_id, prep::phase_to_string(prep::Phase::STACKING),
                           0.45f * static_cast<float>(batch_end) /
                               static_cast<float>(accepted.size()),
                           "stacked linear batch " + std::to_string(batch_end) +
                               "/" + std::to_string(accepted.size()),
                           event_out);
  }
  Matrix2Df stacked = finalize_batch_accum(linear_accum, linear_wsum);
  if (stacked.size() <= 0) {
    emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::STACKING), "error",
                      {{"error", "linear stack produced no output"}}, event_out);
    emitter.run_end(run_id, false, "error", event_out);
    throw std::runtime_error("linear stack produced no output");
  }

  // Restore output scaling: the prewarped frames are stored after normalization
  // (background subtracted, scaled). We need to undo this by applying the
  // median background + scale back to the stacked output.
  {
    std::vector<float> P_mono, P_r, P_g, P_b;
    std::vector<float> B_mono, B_r, B_g, B_b;
    for (const auto& s : pp.norm_scales) {
      if (std::isfinite(s.scale_mono) && s.scale_mono > 0.0f) P_mono.push_back(s.scale_mono);
      if (std::isfinite(s.scale_r)    && s.scale_r    > 0.0f) P_r.push_back(s.scale_r);
      if (std::isfinite(s.scale_g)    && s.scale_g    > 0.0f) P_g.push_back(s.scale_g);
      if (std::isfinite(s.scale_b)    && s.scale_b    > 0.0f) P_b.push_back(s.scale_b);
      if (std::isfinite(s.background_mono)) B_mono.push_back(s.background_mono);
      if (std::isfinite(s.background_r))    B_r.push_back(s.background_r);
      if (std::isfinite(s.background_g))    B_g.push_back(s.background_g);
      if (std::isfinite(s.background_b))    B_b.push_back(s.background_b);
    }
    const float out_scale_mono = core::median_finite_positive(P_mono, 1.0f);
    const float out_scale_r    = core::median_finite_positive(P_r,    1.0f);
    const float out_scale_g    = core::median_finite_positive(P_g,    1.0f);
    const float out_scale_b    = core::median_finite_positive(P_b,    1.0f);
    const float out_bg_mono    = core::median_finite(B_mono, 0.0f);
    const float out_bg_r       = core::median_finite(B_r,    0.0f);
    const float out_bg_g       = core::median_finite(B_g,    0.0f);
    const float out_bg_b       = core::median_finite(B_b,    0.0f);
    image::apply_output_scaling_inplace(stacked, 0, 0,
        pp.color_mode, pp.bayer_pattern,
        out_scale_mono, out_scale_r, out_scale_g, out_scale_b,
        out_bg_mono, out_bg_r, out_bg_g, out_bg_b, 0.0f);
  }

  if (cfg.stacking.cosmetic_correction) {
    stacked = image::cosmetic_correction(
        stacked, cfg.stacking.cosmetic_correction_sigma, true);
  }

  io::FitsHeader header;
  try {
    if (!pp.effective_frames.empty()) header = io::read_fits_header(pp.effective_frames.front());
  } catch (...) {}
  header.set("TC_PRE", std::string("LINEAR_PRESTACK"));
  header.set("TC_STKFR", static_cast<double>(accepted.size()));
  header.set("TC_REJ", cfg.rejection.method);
  header.set("TC_WEIGHT", cfg.stacking.weighting);

  result.stacked_linear_path = output_dir / "stacked_linear.fits";
  io::write_fits_float(result.stacked_linear_path, stacked, header);
  result.frames_used = static_cast<int>(accepted.size());

  if (pp.color_mode == tile_compile::ColorMode::OSC) {
    const auto pattern = tile_compile::string_to_bayer_pattern(pp.bayer_pattern);
    if (pattern != tile_compile::BayerPattern::UNKNOWN) {
      int rgb_rejected_samples = 0;
      int64_t rgb_total_samples = 0;
      const auto rgb_batch_plan =
          raw_stack_sub_batch_plan(rows, cols, 3, accepted.size(), cfg.runtime_limits);
      const size_t rgb_sub_batch =
          rgb_batch_plan.frame_sub_batch_size > 0 ? rgb_batch_plan.frame_sub_batch_size
                                                  : accepted.size();
      Matrix2Df accum_R;
      Matrix2Df accum_G;
      Matrix2Df accum_B;
      Matrix2Df wsum_R;
      Matrix2Df wsum_G;
      Matrix2Df wsum_B;
      for (size_t batch_start = 0; batch_start < accepted.size(); batch_start += rgb_sub_batch) {
        const size_t batch_end = std::min(batch_start + rgb_sub_batch, accepted.size());
        std::vector<Matrix2Df> r_frames;
        std::vector<Matrix2Df> g_frames;
        std::vector<Matrix2Df> b_frames;
        std::vector<float> batch_weights;
        r_frames.reserve(batch_end - batch_start);
        g_frames.reserve(batch_end - batch_start);
        b_frames.reserve(batch_end - batch_start);
        batch_weights.reserve(batch_end - batch_start);
        Matrix2Df deb_r;
        Matrix2Df deb_g;
        Matrix2Df deb_b;
        for (size_t i = batch_start; i < batch_end; ++i) {
          const float* frame_ptr = pp.prewarped_frames.frame_data(static_cast<size_t>(accepted[i]));
          if (frame_ptr == nullptr) continue;
          image::debayer_nearest_neighbor_strided_into(
              frame_ptr, rows, cols, cols, pattern, 0, 0, deb_r, deb_g, deb_b);
          if (deb_r.rows() == rows && deb_r.cols() == cols &&
              deb_g.rows() == rows && deb_g.cols() == cols &&
              deb_b.rows() == rows && deb_b.cols() == cols) {
            r_frames.push_back(std::move(deb_r));
            g_frames.push_back(std::move(deb_g));
            b_frames.push_back(std::move(deb_b));
            batch_weights.push_back(i < weights.size() ? weights[i] : 1.0f);
          }
        }
        if (g_frames.empty()) continue;
        int batch_rejected = 0;
        int64_t batch_total = 0;
        Matrix2Df batch_R = stack_frame_planes(r_frames, batch_weights, cfg.rejection,
                                               cfg.runtime_limits.parallel_workers,
                                               batch_rejected, batch_total, nullptr);
        Matrix2Df batch_G = stack_frame_planes(g_frames, batch_weights, cfg.rejection,
                                               cfg.runtime_limits.parallel_workers,
                                               batch_rejected, batch_total, nullptr);
        Matrix2Df batch_B = stack_frame_planes(b_frames, batch_weights, cfg.rejection,
                                               cfg.runtime_limits.parallel_workers,
                                               batch_rejected, batch_total, nullptr);
        rgb_rejected_samples += batch_rejected;
        rgb_total_samples += batch_total;
        const float batch_weight = static_cast<float>(g_frames.size());
        accumulate_batch_stack(batch_R, batch_weight, accum_R, wsum_R);
        accumulate_batch_stack(batch_G, batch_weight, accum_G, wsum_G);
        accumulate_batch_stack(batch_B, batch_weight, accum_B, wsum_B);
        pp.prewarped_frames.clear_mappings();
        emitter.phase_progress(run_id, prep::phase_to_string(prep::Phase::STACKING),
                               0.45f + 0.55f * static_cast<float>(batch_end) /
                                           static_cast<float>(accepted.size()),
                               "stacked RGB batch " + std::to_string(batch_end) +
                                   "/" + std::to_string(accepted.size()),
                               event_out);
      }
      Matrix2Df R = finalize_batch_accum(accum_R, wsum_R);
      Matrix2Df G = finalize_batch_accum(accum_G, wsum_G);
      Matrix2Df B = finalize_batch_accum(accum_B, wsum_B);
      if (R.size() <= 0 || G.size() <= 0 || B.size() <= 0) {
        emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::STACKING), "error",
                          {{"error", "RGB stack produced no output"}}, event_out);
        emitter.run_end(run_id, false, "error", event_out);
        throw std::runtime_error("RGB stack produced no output");
      }
      // Restore output scaling for R, G, B channels separately.
      {
        std::vector<float> P_r, P_g, P_b, B_r, B_g, B_b;
        for (const auto& s : pp.norm_scales) {
          if (std::isfinite(s.scale_r) && s.scale_r > 0.0f) P_r.push_back(s.scale_r);
          if (std::isfinite(s.scale_g) && s.scale_g > 0.0f) P_g.push_back(s.scale_g);
          if (std::isfinite(s.scale_b) && s.scale_b > 0.0f) P_b.push_back(s.scale_b);
          if (std::isfinite(s.background_r)) B_r.push_back(s.background_r);
          if (std::isfinite(s.background_g)) B_g.push_back(s.background_g);
          if (std::isfinite(s.background_b)) B_b.push_back(s.background_b);
        }
        const float os_r = core::median_finite_positive(P_r, 1.0f);
        const float os_g = core::median_finite_positive(P_g, 1.0f);
        const float os_b = core::median_finite_positive(P_b, 1.0f);
        const float ob_r = core::median_finite(B_r, 0.0f);
        const float ob_g = core::median_finite(B_g, 0.0f);
        const float ob_b = core::median_finite(B_b, 0.0f);
        // R, G, B are already debayered full-res channels (no CFA offset needed)
        R.array() = R.array() * os_r + ob_r;
        G.array() = G.array() * os_g + ob_g;
        B.array() = B.array() * os_b + ob_b;
      }
      if (cfg.stacking.cosmetic_correction) {
        R = image::cosmetic_correction(R, cfg.stacking.cosmetic_correction_sigma, true);
        G = image::cosmetic_correction(G, cfg.stacking.cosmetic_correction_sigma, true);
        B = image::cosmetic_correction(B, cfg.stacking.cosmetic_correction_sigma, true);
      }
      result.rejected_samples += rgb_rejected_samples;
      result.total_samples += rgb_total_samples;
      result.stacked_rgb_path = output_dir / "stacked_rgb.fits";
      io::write_fits_rgb(result.stacked_rgb_path, R, G, B, header);
    }
  }

  result.diagnostics_path = artifact_dir / "stacking_diagnostics.json";
  json diag = {
      {"method", result.method},
      {"weighting", result.weighting},
      {"frames_used", result.frames_used},
      {"frames_rejected_before_stack", result.rejected_input_frames},
      {"sample_rejections_or_winsor_clamps", result.rejected_samples},
      {"total_pixel_samples", result.total_samples},
      {"normalization", cfg.stacking.normalization},
      {"sigma_clip", {
          {"sigma_low", cfg.rejection.low},
          {"sigma_high", cfg.rejection.high},
          {"max_iters", cfg.rejection.max_iters},
          {"min_fraction", cfg.rejection.min_fraction},
      }},
      {"cosmetic_correction", cfg.stacking.cosmetic_correction},
      {"cosmetic_correction_sigma", cfg.stacking.cosmetic_correction_sigma},
      {"per_frame_cosmetic_correction", cfg.stacking.per_frame_cosmetic_correction},
      {"per_frame_cosmetic_correction_sigma", cfg.stacking.per_frame_cosmetic_correction_sigma},
      {"output_linear", result.stacked_linear_path.string()},
      {"accepted_indices", accepted},
      {"weights", weights},
  };
  if (!result.stacked_rgb_path.empty()) diag["output_rgb"] = result.stacked_rgb_path.string();
  if (pp.color_mode == tile_compile::ColorMode::OSC) {
    diag["rgb_stack_mode"] = "tile_compile_style_sub_batch_debayer_before_stack";
    diag["rgb_channels_stacked_separately"] = !result.stacked_rgb_path.empty();
    diag["linear_sub_batch_size"] = linear_batch_plan.frame_sub_batch_size;
    diag["rgb_sub_batch_size"] =
        raw_stack_sub_batch_plan(rows, cols, 3, accepted.size(), cfg.runtime_limits)
            .frame_sub_batch_size;
    diag["runtime_limits"] = {
        {"parallel_workers", cfg.runtime_limits.parallel_workers},
        {"memory_budget", cfg.runtime_limits.memory_budget},
    };
  }
  core::write_text(result.diagnostics_path, diag.dump(2));

  emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::STACKING), "ok",
                    {{"frames_used", result.frames_used},
                     {"frames_rejected_before_stack", result.rejected_input_frames},
                     {"sample_rejections_or_winsor_clamps", result.rejected_samples},
                     {"output_linear", result.stacked_linear_path.string()},
                     {"output_rgb", result.stacked_rgb_path.empty() ? json(nullptr) : json(result.stacked_rgb_path.string())},
                     {"diagnostics", result.diagnostics_path.string()}},
                    event_out);
  return result;
}

std::string shell_quote(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 2);
  out.push_back(static_cast<char>(39));
  for (char c : s) {
    if (c == static_cast<char>(39)) out += "'\\''";
    else out.push_back(c);
  }
  out.push_back(static_cast<char>(39));
  return out;
}

void add_phase_result(json& phases,
                      const std::string& phase,
                      const std::string& status,
                      json detail = json::object()) {
  detail["phase"] = phase;
  detail["status"] = status;
  phases.push_back(std::move(detail));
}

void add_artifact(json& artifacts,
                  const std::string& type,
                  const std::string& phase,
                  const fs::path& path) {
  if (path.empty()) return;
  artifacts.push_back({{"type", type}, {"phase", phase}, {"path", path.string()}});
}

float accepted_median_fwhm(const runner::QualityAnalysisContext& qa) {
  std::vector<float> values;
  values.reserve(qa.accepted_indices.size());
  for (int idx : qa.accepted_indices) {
    if (idx < 0 || static_cast<size_t>(idx) >= qa.records.size()) continue;
    const float fwhm = qa.records[static_cast<size_t>(idx)].fwhm;
    if (std::isfinite(fwhm) && fwhm > 0.0f) values.push_back(fwhm);
  }
  return median_value(std::move(values));
}

TileGrid build_preprocess_bge_grid(int rows,
                                   int cols,
                                   const config::TileConfig& tile_cfg,
                                   float seeing_fwhm) {
  if (!(seeing_fwhm > 0.0f) || !std::isfinite(seeing_fwhm)) {
    seeing_fwhm = 3.0f;
  }
  const int tmin = std::max(16, tile_cfg.min_size);
  const int divisor = std::max(1, tile_cfg.max_divisor);
  int tmax = std::max(1, std::min(rows, cols) / divisor);
  if (tmax < tmin) tmax = tmin;

  const float requested = static_cast<float>(std::max(1, tile_cfg.size_factor)) *
                          seeing_fwhm;
  int tile_size = static_cast<int>(
      std::floor(std::min(std::max(requested, static_cast<float>(tmin)),
                          static_cast<float>(tmax))));
  if (tile_size < tmin) tile_size = tmin;

  const float overlap =
      std::min(0.5f, std::max(0.0f, tile_cfg.overlap_fraction));
  TileGrid grid;
  grid.tile_size = tile_size;
  grid.overlap_fraction = overlap;
  grid.rows = 0;
  grid.cols = 0;
  grid.tiles = tile_compile::pipeline::build_initial_tile_grid(cols, rows,
                                                               tile_size,
                                                               overlap);
  for (const auto& t : grid.tiles) {
    grid.rows = std::max(grid.rows, t.row + 1);
    grid.cols = std::max(grid.cols, t.col + 1);
  }
  return grid;
}

std::vector<TileMetrics> measure_bge_tile_metrics(const Matrix2Df& R,
                                                  const Matrix2Df& G,
                                                  const Matrix2Df& B,
                                                  const TileGrid& grid) {
  const size_t n_tiles = grid.tiles.size();
  std::vector<TileMetrics> out(n_tiles);
  const int workers = default_parallel_workers(n_tiles, 0);
  parallel_for_indices(n_tiles, workers, [&](size_t ti) {
    const Tile& t = grid.tiles[ti];
    Matrix2Df tile(t.height, t.width);
    for (int y = 0; y < t.height; ++y) {
      for (int x = 0; x < t.width; ++x) {
        tile(y, x) = (R(t.y + y, t.x + x) + G(t.y + y, t.x + x) +
                      B(t.y + y, t.x + x)) / 3.0f;
      }
    }
    TileMetrics tm = metrics::calculate_tile_metrics(tile);
    if (!std::isfinite(tm.background)) tm.background = 0.0f;
    if (!std::isfinite(tm.noise)) tm.noise = 0.0f;
    if (!std::isfinite(tm.gradient_energy)) tm.gradient_energy = 0.0f;
    if (!std::isfinite(tm.quality_score)) tm.quality_score = 1.0f;
    if (tm.star_count <= 0) tm.type = TileType::STRUCTURE;
    out[ti] = tm;
  });
  return out;
}

config::BGEConfig default_preprocess_bge_config() {
  config::BGEConfig cfg;
  cfg.enabled = true;
  cfg.method = "classic";
  cfg.min_valid_samples_for_apply = 16;
  cfg.min_valid_sample_fraction_for_apply = 0.10f;
  cfg.grid.N_g = 32;
  return cfg;
}

config::PCCConfig default_preprocess_pcc_config() {
  config::PCCConfig cfg;
  cfg.enabled = true;
  return cfg;
}

PreprocessPostprocessResult run_preprocess_postprocess(
    const std::string& run_id,
    const prep::Config& cfg,
    const PreprocessStackResult& stack,
    const runner::PreprocessPipelineContext& pp,
    const runner::QualityAnalysisContext& qa,
    const fs::path& run_dir,
    core::EventEmitter& emitter,
    std::ostream& event_out) {
  PreprocessPostprocessResult result;
  const fs::path artifact_dir = run_dir / "artifacts" / "preprocess";
  const fs::path output_dir = run_dir / "outputs";
  fs::create_directories(artifact_dir);
  fs::create_directories(output_dir);

  astro::WCS wcs;
  bool have_wcs = false;

  emitter.phase_start(run_id, prep::phase_to_string(prep::Phase::ASTROMETRY), event_out);
  if (!cfg.postprocess.astrometry) {
    add_phase_result(result.phases, "ASTROMETRY", "skipped", {{"reason", "disabled"}});
    emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::ASTROMETRY), "skipped",
                      {{"reason", "disabled"}}, event_out);
  } else if (stack.stacked_rgb_path.empty()) {
    add_phase_result(result.phases, "ASTROMETRY", "skipped", {{"reason", "no_rgb_stack"}});
    emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::ASTROMETRY), "skipped",
                      {{"reason", "no_rgb_stack"}}, event_out);
  } else {
    std::string astap_data = cfg.has_astrometry_config ? cfg.astrometry.astap_data_dir : "";
    if (astap_data.empty()) {
#ifdef _WIN32
      if (const char *la = std::getenv("LOCALAPPDATA"); la && la[0] != '\0') {
        astap_data = std::string(la) + "\\tile_compile\\astap";
      }
#else
      if (const char* home = std::getenv("HOME")) {
        astap_data = std::string(home) + "/.local/share/tile_compile/astap";
      }
#endif
    }
    const std::string astap_bin_cfg = cfg.has_astrometry_config ? cfg.astrometry.astap_bin : "";
    fs::path astap_bin_path = runner::resolve_astap_binary_path(astap_bin_cfg, astap_data);
    // If the resolved binary lives outside the configured data dir, use its parent as data dir
    if (!astap_bin_path.empty()) {
      std::error_code ec;
      fs::path data_dir_path(astap_data);
      auto relative = fs::relative(astap_bin_path, data_dir_path, ec);
      if (ec || relative.empty() || relative.begin() == relative.end() || *relative.begin() == "..") {
        astap_data = astap_bin_path.parent_path().string();
      }
    }
    const int search_radius = cfg.has_astrometry_config ? cfg.astrometry.search_radius : 180;
    if (astap_bin_path.empty()) {
      const std::string reported_bin = astap_bin_cfg.empty() ? astap_data + "/astap_cli" : astap_bin_cfg;
      add_phase_result(result.phases, "ASTROMETRY", "skipped",
                       {{"reason", "astap_not_found"}, {"astap_bin", reported_bin}});
      emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::ASTROMETRY), "skipped",
                        {{"reason", "astap_not_found"}, {"astap_bin", reported_bin}},
                        event_out);
    } else {
      const std::string cmd = runner::shell_quote(astap_bin_path.string()) + " -f " +
                              runner::shell_quote(stack.stacked_rgb_path.string()) + " -d " +
                              runner::shell_quote(astap_data) + " -r " + std::to_string(search_radius);
      const int ret = std::system(runner::system_cmd(cmd).c_str());
      fs::path wcs_path = stack.stacked_rgb_path;
      wcs_path.replace_extension(".wcs");
      if (ret == 0 && fs::exists(wcs_path)) {
        try {
          wcs = astro::parse_wcs_file(wcs_path.string());
          have_wcs = wcs.valid();
        } catch (...) {
          have_wcs = false;
        }
      }
      if (have_wcs) {
        result.astrometry_wcs_path = artifact_dir / "stacked_rgb.wcs";
        fs::copy_file(wcs_path, result.astrometry_wcs_path,
                      fs::copy_options::overwrite_existing);
        io::FitsHeader hdr = io::read_fits_header(stack.stacked_rgb_path);
        hdr.numeric_values["CRVAL1"] = wcs.crval1;
        hdr.numeric_values["CRVAL2"] = wcs.crval2;
        hdr.numeric_values["CRPIX1"] = wcs.crpix1;
        hdr.numeric_values["CRPIX2"] = wcs.crpix2;
        hdr.numeric_values["CD1_1"] = wcs.cd1_1;
        hdr.numeric_values["CD1_2"] = wcs.cd1_2;
        hdr.numeric_values["CD2_1"] = wcs.cd2_1;
        hdr.numeric_values["CD2_2"] = wcs.cd2_2;
        hdr.numeric_values["EQUINOX"] = 2000.0;
        hdr.string_values["CTYPE1"] = "RA---TAN";
        hdr.string_values["CTYPE2"] = "DEC--TAN";
        hdr.string_values["CUNIT1"] = "deg";
        hdr.string_values["CUNIT2"] = "deg";
        hdr.bool_values["PLTSOLVD"] = true;
        io::update_fits_header_in_place(stack.stacked_rgb_path, hdr);
        add_artifact(result.artifacts, "wcs", "ASTROMETRY", result.astrometry_wcs_path);
        add_phase_result(result.phases, "ASTROMETRY", "ok",
                         {{"ra", wcs.crval1}, {"dec", wcs.crval2},
                          {"pixel_scale_arcsec", wcs.pixel_scale_arcsec()},
                          {"rotation_deg", wcs.rotation_deg()},
                          {"wcs_file", result.astrometry_wcs_path.string()}});
        emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::ASTROMETRY), "ok",
                          result.phases.back(), event_out);
      } else {
        add_phase_result(result.phases, "ASTROMETRY", "skipped",
                         {{"reason", "solve_failed"}, {"exit_code", ret}});
        emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::ASTROMETRY), "skipped",
                          {{"reason", "solve_failed"}, {"exit_code", ret}}, event_out);
      }
    }
  }

  fs::path current_rgb = stack.stacked_rgb_path;
  // Determine BGE method for phase label before emitting phase_start
  std::string bge_method_for_label;
  if (!cfg.postprocess.bge) {
    bge_method_for_label = "none";
  } else if (cfg.has_bge_config) {
    bge_method_for_label = cfg.bge.method;
  } else {
    bge_method_for_label = "classic";
  }
  const std::string bge_phase_label =
      (bge_method_for_label == "none")    ? "BGE (Skipped)" :
      (bge_method_for_label == "classic") ? "BGE (Classic)" :
                                            "BGE (AutoBGE)";
  emitter.phase_start(run_id, prep::phase_to_string(prep::Phase::BGE), event_out,
                      {{"label", bge_phase_label},
                       {"bge_method", bge_method_for_label}});
  if (!cfg.postprocess.bge) {
    add_phase_result(result.phases, "BGE", "skipped", {{"reason", "disabled"}});
    emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::BGE), "skipped",
                      {{"reason", "disabled"}}, event_out);
  } else if (current_rgb.empty()) {
    add_phase_result(result.phases, "BGE", "skipped", {{"reason", "no_rgb_stack"}});
    emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::BGE), "skipped",
                      {{"reason", "no_rgb_stack"}}, event_out);
  } else {
    auto rgb = io::read_fits_rgb(current_rgb);
    config::BGEConfig bge_source = cfg.has_bge_config ? cfg.bge : default_preprocess_bge_config();
    if (!cfg.has_bge_config) {
      bge_source.enabled = true;
      bge_source.method = "classic";
    }
    float seeing_fwhm = accepted_median_fwhm(qa);
    if (pp.color_mode == tile_compile::ColorMode::OSC && seeing_fwhm > 0.0f) {
      seeing_fwhm *= 2.0f;
    }
    TileGrid grid = build_preprocess_bge_grid(rgb.R.rows(), rgb.R.cols(),
                                             cfg.tile, seeing_fwhm);
    std::vector<TileMetrics> tile_metrics = measure_bge_tile_metrics(rgb.R, rgb.G, rgb.B, grid);
    image::BGEConfig bge_cfg = runner::to_image_bge_config(bge_source);
    bge_cfg.common_valid_mask.assign(static_cast<size_t>(rgb.R.rows() * rgb.R.cols()), 1);
    bge_cfg.common_mask_rows = rgb.R.rows();
    bge_cfg.common_mask_cols = rgb.R.cols();
    image::BGEDiagnostics diag;
    const bool ok = image::apply_background_extraction(rgb.R, rgb.G, rgb.B,
                                                       tile_metrics, grid, bge_cfg, &diag);
    result.bge_diag_path = artifact_dir / "bge_diagnostics.json";
    json bge_diag_json = runner::bge_diag_to_json(diag, true, true, true);
    bge_diag_json["local_metrics_tiles"] = static_cast<int>(tile_metrics.size());
    bge_diag_json["bge_grid_tiles"] = static_cast<int>(grid.tiles.size());
    bge_diag_json["preprocess_bge_tile_size"] = grid.tile_size;
    bge_diag_json["preprocess_bge_overlap_fraction"] = grid.overlap_fraction;
    bge_diag_json["preprocess_bge_seeing_fwhm"] = seeing_fwhm;
    bge_diag_json["config"] = {
        {"sample_quantile", bge_source.sample_quantile},
        {"sample_estimator", bge_source.sample_estimator},
        {"min_sample_bg_value", bge_source.min_sample_bg_value},
        {"structure_thresh_percentile", bge_source.structure_thresh_percentile},
        {"min_tiles_per_cell", bge_source.min_tiles_per_cell},
        {"min_valid_sample_fraction_for_apply",
         bge_source.min_valid_sample_fraction_for_apply},
        {"min_valid_samples_for_apply", bge_source.min_valid_samples_for_apply},
        {"tile_weight_lambda_structure", bge_source.tile_weight_lambda_structure},
        {"mask", {
            {"star_dilate_px", bge_source.mask.star_dilate_px},
            {"sat_dilate_px", bge_source.mask.sat_dilate_px},
        }},
        {"grid", {
            {"N_g", bge_source.grid.N_g},
            {"G_min_px", bge_source.grid.G_min_px},
            {"G_max_fraction", bge_source.grid.G_max_fraction},
            {"insufficient_cell_strategy", bge_source.grid.insufficient_cell_strategy},
        }},
        {"fit", {
            {"method", bge_source.fit.method},
            {"robust_loss", bge_source.fit.robust_loss},
            {"huber_delta", bge_source.fit.huber_delta},
            {"irls_max_iterations", bge_source.fit.irls_max_iterations},
            {"irls_tolerance", bge_source.fit.irls_tolerance},
            {"polynomial_order", bge_source.fit.polynomial_order},
            {"rbf_phi", bge_source.fit.rbf_phi},
            {"rbf_mu_factor", bge_source.fit.rbf_mu_factor},
            {"rbf_lambda", bge_source.fit.rbf_lambda},
            {"rbf_epsilon", bge_source.fit.rbf_epsilon},
        }},
        {"autotune", {
            {"enabled", bge_source.autotune.enabled},
            {"max_evals", bge_source.autotune.max_evals},
            {"holdout_fraction", bge_source.autotune.holdout_fraction},
            {"alpha_flatness", bge_source.autotune.alpha_flatness},
            {"beta_roughness", bge_source.autotune.beta_roughness},
            {"strategy", bge_source.autotune.strategy},
        }},
    };
    core::write_text(result.bge_diag_path, bge_diag_json.dump(2));
    add_artifact(result.artifacts, "bge_diagnostics", "BGE", result.bge_diag_path);
    if (ok) {
      result.bge_rgb_path = output_dir / "stacked_rgb_bge.fits";
      io::write_fits_rgb(result.bge_rgb_path, rgb.R, rgb.G, rgb.B,
                         io::read_fits_header(current_rgb));
      current_rgb = result.bge_rgb_path;
      add_artifact(result.artifacts, "stacked_rgb_bge", "BGE", result.bge_rgb_path);
      add_phase_result(result.phases, "BGE", "ok",
                       {{"output_rgb", result.bge_rgb_path.string()},
                        {"diagnostics", result.bge_diag_path.string()}});
      emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::BGE), "ok",
                        result.phases.back(), event_out);
    } else {
      add_phase_result(result.phases, "BGE", "skipped",
                       {{"reason", diag.failure_reason.empty() ? "bge_not_applied" : diag.failure_reason},
                        {"diagnostics", result.bge_diag_path.string()}});
      emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::BGE), "skipped",
                        result.phases.back(), event_out);
    }
  }

  bool have_successful_pcc = false;
  emitter.phase_start(run_id, prep::phase_to_string(prep::Phase::PCC), event_out);
  if (!cfg.postprocess.pcc) {
    add_phase_result(result.phases, "PCC", "skipped", {{"reason", "disabled"}});
    emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::PCC), "skipped",
                      {{"reason", "disabled"}}, event_out);
  } else if (current_rgb.empty()) {
    add_phase_result(result.phases, "PCC", "skipped", {{"reason", "no_rgb_stack"}});
    emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::PCC), "skipped",
                      {{"reason", "no_rgb_stack"}}, event_out);
  } else if (!have_wcs) {
    add_phase_result(result.phases, "PCC", "skipped", {{"reason", "missing_wcs"}});
    emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::PCC), "skipped",
                      {{"reason", "missing_wcs"}}, event_out);
  } else {
    config::PCCConfig pcc_config = cfg.has_pcc_config ? cfg.pcc : default_preprocess_pcc_config();
    if (!cfg.has_pcc_config) pcc_config.enabled = true;
    auto catalog = runner::query_pcc_catalog_stars(wcs, pcc_config, event_out, "[PCC][preprocess]");
    if (catalog.stars.empty()) {
      add_phase_result(result.phases, "PCC", "skipped", {{"reason", "no_catalog_stars"}});
      emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::PCC), "skipped",
                        result.phases.back(), event_out);
    } else {
      auto rgb = io::read_fits_rgb(current_rgb);
      astro::PCCConfig pc = runner::to_astrometry_pcc_config(pcc_config);
      pc.common_valid_mask.assign(static_cast<size_t>(rgb.R.rows() * rgb.R.cols()), 1);
      pc.output_valid_mask = pc.common_valid_mask;
      pc.common_mask_rows = pc.output_mask_rows = rgb.R.rows();
      pc.common_mask_cols = pc.output_mask_cols = rgb.R.cols();
      astro::PCCResult pcc = astro::run_pcc(rgb.R, rgb.G, rgb.B, wcs, catalog.stars, pc);
      result.pcc_diag_path = artifact_dir / "pcc_diagnostics.json";
      json pcc_diag = {
          {"success", pcc.success},
          {"used_source", catalog.used_source},
          {"stars_matched", pcc.n_stars_matched},
          {"stars_used", pcc.n_stars_used},
          {"residual_rms", pcc.residual_rms},
          {"determinant", pcc.determinant},
          {"condition_number", pcc.condition_number},
          {"apply_mode", pcc.apply_mode},
          {"error", pcc.error_message},
      };
      core::write_text(result.pcc_diag_path, pcc_diag.dump(2));
      add_artifact(result.artifacts, "pcc_diagnostics", "PCC", result.pcc_diag_path);
      if (pcc.success) {
        have_successful_pcc = true;
        result.pcc_rgb_path = output_dir / "stacked_rgb_pcc.fits";
        io::write_fits_rgb(result.pcc_rgb_path, rgb.R, rgb.G, rgb.B,
                           io::read_fits_header(current_rgb));
        add_artifact(result.artifacts, "stacked_rgb_pcc", "PCC", result.pcc_rgb_path);
        add_phase_result(result.phases, "PCC", "ok",
                         {{"output_rgb", result.pcc_rgb_path.string()},
                          {"diagnostics", result.pcc_diag_path.string()},
                          {"stars_used", pcc.n_stars_used}});
        emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::PCC), "ok",
                          result.phases.back(), event_out);
      } else {
        add_phase_result(result.phases, "PCC", "skipped",
                         {{"reason", pcc.error_message},
                          {"diagnostics", result.pcc_diag_path.string()}});
        emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::PCC), "skipped",
                          result.phases.back(), event_out);
      }
    }
  }

  emitter.phase_start(run_id, prep::phase_to_string(prep::Phase::HYPERMETRIC_STRETCH), event_out);
  if (!cfg.postprocess.hypermetric_stretch) {
    add_phase_result(result.phases, "HYPERMETRIC_STRETCH", "skipped", {{"reason", "disabled"}});
    emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::HYPERMETRIC_STRETCH), "skipped",
                      result.phases.back(), event_out);
  } else if (!have_wcs) {
    add_phase_result(result.phases, "HYPERMETRIC_STRETCH", "skipped",
                     {{"reason", "missing_successful_astrometry"}});
    emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::HYPERMETRIC_STRETCH), "skipped",
                      result.phases.back(), event_out);
  } else if (!have_successful_pcc) {
    add_phase_result(result.phases, "HYPERMETRIC_STRETCH", "skipped",
                     {{"reason", "missing_successful_pcc"},
                      {"require_successful_pcc", true}});
    emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::HYPERMETRIC_STRETCH), "skipped",
                      result.phases.back(), event_out);
  } else {
    const fs::path hms_input = result.pcc_rgb_path;
    if (hms_input.empty()) {
      add_phase_result(result.phases, "HYPERMETRIC_STRETCH", "skipped", {{"reason", "no_rgb_input"}});
      emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::HYPERMETRIC_STRETCH), "skipped",
                        result.phases.back(), event_out);
    } else {
      try {
        auto rgb = io::read_fits_rgb(hms_input);
        image::HyperMetricStretchConfig hms_cfg;
        hms_cfg.enabled = true;
        hms_cfg.require_successful_pcc = cfg.hypermetric_stretch.require_successful_pcc;
        hms_cfg.mode = cfg.hypermetric_stretch.mode;
        hms_cfg.sensor_profile = cfg.hypermetric_stretch.sensor_profile;
        hms_cfg.fallback_profile = cfg.hypermetric_stretch.fallback_profile;
        hms_cfg.adaptive_anchor = cfg.hypermetric_stretch.adaptive_anchor;
        hms_cfg.target_bg = cfg.hypermetric_stretch.target_bg;
        hms_cfg.protect_b = cfg.hypermetric_stretch.protect_b;
        hms_cfg.convergence_power = cfg.hypermetric_stretch.convergence_power;
        hms_cfg.log_d_mode = cfg.hypermetric_stretch.log_d_mode;
        hms_cfg.fixed_log_d = cfg.hypermetric_stretch.fixed_log_d;
        hms_cfg.color_strategy = cfg.hypermetric_stretch.color_strategy;
        hms_cfg.fixed_color_strategy = cfg.hypermetric_stretch.fixed_color_strategy;
        hms_cfg.color_grip = cfg.hypermetric_stretch.color_grip;
        hms_cfg.shadow_convergence = cfg.hypermetric_stretch.shadow_convergence;
        hms_cfg.linear_expansion = cfg.hypermetric_stretch.linear_expansion;
        hms_cfg.write_channels = cfg.hypermetric_stretch.write_channels;
        hms_cfg.output_rgb = cfg.hypermetric_stretch.output_rgb;
        const auto hms_diag = image::run_hypermetric_stretch_rgb(rgb.R, rgb.G, rgb.B, hms_cfg);
        fs::path hms_out(hms_cfg.output_rgb);
        if (hms_out.is_relative()) hms_out = output_dir / hms_out;
        const fs::path hms_diag_path = artifact_dir / "hms_diagnostics.json";
        if (hms_diag.success) {
          io::write_fits_rgb(hms_out, rgb.R, rgb.G, rgb.B, io::read_fits_header(hms_input));
          if (hms_cfg.write_channels) {
            io::write_fits_float(output_dir / "hms_R.fit", rgb.R, io::read_fits_header(hms_input));
            io::write_fits_float(output_dir / "hms_G.fit", rgb.G, io::read_fits_header(hms_input));
            io::write_fits_float(output_dir / "hms_B.fit", rgb.B, io::read_fits_header(hms_input));
          }
          const json hms_diag_json = {
              {"success", hms_diag.success},
              {"profile", hms_diag.profile},
              {"profile_source", hms_diag.profile_source},
              {"anchor", hms_diag.anchor},
              {"log_d", hms_diag.log_d},
              {"target_bg", hms_diag.target_bg},
              {"protect_b", hms_diag.protect_b},
              {"convergence_power", hms_diag.convergence_power},
              {"color_strategy", hms_diag.color_strategy},
              {"color_grip", hms_diag.color_grip},
              {"input", hms_input.string()},
              {"output", hms_out.string()},
          };
          core::write_text(hms_diag_path, hms_diag_json.dump(2));
          result.hms_rgb_path  = hms_out;
          result.hms_diag_path = hms_diag_path;
          add_artifact(result.artifacts, "stacked_rgb_hms", "HYPERMETRIC_STRETCH", hms_out);
          add_artifact(result.artifacts, "hms_diagnostics", "HYPERMETRIC_STRETCH", hms_diag_path);
          add_phase_result(result.phases, "HYPERMETRIC_STRETCH", "ok",
                           {{"output_rgb", hms_out.string()},
                            {"diagnostics", hms_diag_path.string()},
                            {"log_d", hms_diag.log_d},
                            {"anchor", hms_diag.anchor}});
          emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::HYPERMETRIC_STRETCH), "ok",
                            result.phases.back(), event_out);
        } else {
          const json hms_diag_json = {{"success", false}, {"error", hms_diag.error_message}};
          core::write_text(hms_diag_path, hms_diag_json.dump(2));
          add_phase_result(result.phases, "HYPERMETRIC_STRETCH", "skipped",
                           {{"reason", hms_diag.error_message.empty() ? "hms_failed" : hms_diag.error_message},
                            {"diagnostics", hms_diag_path.string()}});
          emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::HYPERMETRIC_STRETCH), "skipped",
                            result.phases.back(), event_out);
        }
      } catch (const std::exception& e) {
        add_phase_result(result.phases, "HYPERMETRIC_STRETCH", "skipped",
                         {{"reason", std::string("exception: ") + e.what()}});
        emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::HYPERMETRIC_STRETCH), "skipped",
                          result.phases.back(), event_out);
      }
    }
  }
  return result;
}

void write_rejected_frames(const fs::path& path,
                           const runner::QualityAnalysisContext& qa) {
  std::ofstream out(path);
  for (int idx : qa.rejected_indices) {
    const auto& r = qa.records[static_cast<size_t>(idx)];
    out << r.index << "," << r.filename << ","
        << runner::frame_exclusion_reason_to_string(r.exclusion_reason)
        << "," << r.exclusion_detail << "\n";
  }
}

void write_report_and_manifest(const fs::path& run_dir,
                               const std::string& run_id,
                               const prep::Config& cfg,
                               const runner::PreprocessPipelineContext& pp,
                               const runner::QualityAnalysisContext& qa,
                               const PreprocessStackResult& stack,
                               const PreprocessPostprocessResult& post) {
  const fs::path artifact_dir = run_dir / "artifacts" / "preprocess";
  fs::create_directories(artifact_dir);

  const fs::path report_path = artifact_dir / "preprocessing_report.json";
  const fs::path manifest_path = artifact_dir / "artifacts_manifest.json";
  const fs::path rejected_path = artifact_dir / "rejected_frames.txt";
  write_rejected_frames(rejected_path, qa);

  json phases = json::array({
      {{"phase", "INPUT_SCAN"}, {"status", "ok"}, {"frames_total", static_cast<int>(pp.effective_frames.size())}},
      {{"phase", "CALIBRATION"}, {"status", pp.calibration_applied ? "ok" : "skipped"},
       {"reason", pp.calibration_applied ? json(nullptr) : json("no calibration configured")}},
      {{"phase", "CFA_CHANNEL_PREP"}, {"status", "ok"},
       {"color_mode", pp.color_mode == tile_compile::ColorMode::OSC ? "OSC" :
                      pp.color_mode == tile_compile::ColorMode::RGB ? "RGB" : "MONO"},
       {"bayer_pattern", pp.bayer_pattern}},
      {{"phase", "REFERENCE_SELECTION"}, {"status", "ok"},
       {"strategy", pp.reference_selection_strategy}, {"index", pp.reference_frame_index}},
      {{"phase", "REGISTRATION"}, {"status", "ok"},
       {"canvas_width", pp.canvas_width}, {"canvas_height", pp.canvas_height}},
      {{"phase", "QUALITY_ANALYSIS"}, {"status", "ok"},
       {"frames_total", static_cast<int>(qa.records.size())}, {"frame_quality_csv", qa.csv_path.string()}},
      {{"phase", "FRAME_FILTERING"}, {"status", "ok"},
       {"accepted", static_cast<int>(qa.accepted_indices.size())},
       {"rejected", static_cast<int>(qa.rejected_indices.size())}},
      {{"phase", "STACKING"}, {"status", "ok"},
       {"method", stack.method}, {"frames_used", stack.frames_used}},
  });
  for (const auto& phase : post.phases) phases.push_back(phase);
  phases.push_back({{"phase", "REPORT"}, {"status", "ok"}});

  int manual_override_count = 0;
  for (const auto& record : qa.records) {
    if (record.manual_override) ++manual_override_count;
  }

  json report = {
      {"run_id", run_id},
      {"status", "ok"},
      {"process", "preprocessing"},
      {"report_contract", "tile_compile_artifacts_v1"},
      {"generated_by", "tile_compile_preprocessing_report"},
      {"effective_config", config_to_json(cfg)},
      {"phases", phases},
      {"input", {
          {"frames_total", static_cast<int>(pp.effective_frames.size())},
          {"color_mode", pp.color_mode == tile_compile::ColorMode::OSC ? "OSC" :
                         pp.color_mode == tile_compile::ColorMode::RGB ? "RGB" : "MONO"},
          {"bayer_pattern", pp.bayer_pattern},
          {"canvas_width", pp.canvas_width},
          {"canvas_height", pp.canvas_height},
      }},
      {"reference", {
          {"strategy", pp.reference_selection_strategy},
          {"index", pp.reference_frame_index},
      }},
      {"quality", {
          {"frames_total", static_cast<int>(qa.records.size())},
          {"accepted", static_cast<int>(qa.accepted_indices.size())},
          {"rejected", static_cast<int>(qa.rejected_indices.size())},
          {"manual_overrides", manual_override_count},
          {"frame_quality_csv", qa.csv_path.string()},
          {"rejected_frames", rejected_path.string()},
      }},
      {"stacking", {
          {"method", stack.method},
          {"weighting", stack.weighting},
          {"frames_used", stack.frames_used},
          {"frames_rejected_before_stack", stack.rejected_input_frames},
          {"sample_rejections_or_winsor_clamps", stack.rejected_samples},
          {"normalization", cfg.stacking.normalization},
          {"stacked_linear", stack.stacked_linear_path.string()},
          {"stacked_rgb", stack.stacked_rgb_path.empty() ? json(nullptr) : json(stack.stacked_rgb_path.string())},
          {"diagnostics", stack.diagnostics_path.string()},
      }},
      {"postprocess", {
          {"astrometry", cfg.postprocess.astrometry},
          {"bge", cfg.postprocess.bge},
          {"pcc", cfg.postprocess.pcc},
          {"hypermetric_stretch", cfg.postprocess.hypermetric_stretch},
          {"astrometry_wcs", post.astrometry_wcs_path.empty() ? json(nullptr) : json(post.astrometry_wcs_path.string())},
          {"bge_rgb", post.bge_rgb_path.empty() ? json(nullptr) : json(post.bge_rgb_path.string())},
          {"bge_diagnostics", post.bge_diag_path.empty() ? json(nullptr) : json(post.bge_diag_path.string())},
          {"pcc_rgb", post.pcc_rgb_path.empty() ? json(nullptr) : json(post.pcc_rgb_path.string())},
          {"pcc_diagnostics", post.pcc_diag_path.empty() ? json(nullptr) : json(post.pcc_diag_path.string())},
          {"hms_rgb", post.hms_rgb_path.empty() ? json(nullptr) : json(post.hms_rgb_path.string())},
          {"hms_diagnostics", post.hms_diag_path.empty() ? json(nullptr) : json(post.hms_diag_path.string())},
      }},
      {"artifacts", json::array({
          {{"type", "frame_quality"}, {"path", qa.csv_path.string()}},
          {{"type", "quality_analysis"}, {"path", (artifact_dir / "quality_analysis.json").string()}},
          {{"type", "rejected_frames"}, {"path", rejected_path.string()}},
          {{"type", "stacked_linear"}, {"path", stack.stacked_linear_path.string()}},
          {{"type", "stacking_diagnostics"}, {"path", stack.diagnostics_path.string()}},
      })},
  };
  if (!stack.stacked_rgb_path.empty()) {
    report["artifacts"].push_back({{"type", "stacked_rgb"}, {"path", stack.stacked_rgb_path.string()}});
  }
  for (const auto& artifact : post.artifacts) {
    report["artifacts"].push_back(artifact);
  }
  core::write_text(report_path, report.dump(2));

  const fs::path markdown_path = artifact_dir / "preprocessing_report.md";
  const fs::path html_path = artifact_dir / "preprocessing_report.html";
  {
    std::ostringstream md;
    md << "# Preprocessing Report\n\n";
    md << "- Run: `" << run_id << "`\n";
    md << "- Status: ok\n";
    md << "- Frames total: " << pp.effective_frames.size() << "\n";
    md << "- Frames accepted: " << qa.accepted_indices.size() << "\n";
    md << "- Frames rejected: " << qa.rejected_indices.size() << "\n";
    md << "- Stack frames used: " << stack.frames_used << "\n\n";
    md << "## Quality\n\n";
    md << "- Manual overrides: " << manual_override_count << "\n";
    md << "- Quality CSV: `" << qa.csv_path.string() << "`\n";
    md << "- Rejected frames: `" << rejected_path.string() << "`\n\n";
    md << "## Artifacts\n\n";
    for (const auto& artifact : report["artifacts"]) {
      md << "- " << artifact.value("type", std::string("artifact"))
         << ": `" << artifact.value("path", std::string()) << "`\n";
    }
    md << "\n## Phases\n\n";
    for (const auto& phase : phases) {
      md << "- " << phase.value("phase", std::string())
         << ": " << phase.value("status", std::string()) << "\n";
    }
    core::write_text(markdown_path, md.str());
  }
  {
    std::ostringstream html;
    html << "<!doctype html><html lang=\"de\"><head><meta charset=\"utf-8\">"
         << "<title>Preprocessing Report</title>"
         << "<style>body{font-family:sans-serif;margin:32px;line-height:1.45}"
         << "code{background:#f3f4f6;padding:2px 4px;border-radius:4px}"
         << "table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:6px 8px}</style>"
         << "</head><body><h1>Preprocessing Report</h1>"
         << "<p>Run: <code>" << run_id << "</code></p>"
         << "<h2>Summary</h2><table><tbody>"
         << "<tr><th>Frames total</th><td>" << pp.effective_frames.size() << "</td></tr>"
         << "<tr><th>Accepted</th><td>" << qa.accepted_indices.size() << "</td></tr>"
         << "<tr><th>Rejected</th><td>" << qa.rejected_indices.size() << "</td></tr>"
         << "<tr><th>Manual overrides</th><td>" << manual_override_count << "</td></tr>"
         << "<tr><th>Stack frames</th><td>" << stack.frames_used << "</td></tr>"
         << "</tbody></table><h2>Artifacts</h2><ul>";
    for (const auto& artifact : report["artifacts"]) {
      html << "<li><strong>" << artifact.value("type", std::string("artifact"))
           << "</strong>: <code>" << artifact.value("path", std::string()) << "</code></li>";
    }
    html << "</ul><h2>Phases</h2><ul>";
    for (const auto& phase : phases) {
      html << "<li><strong>" << phase.value("phase", std::string())
           << "</strong>: " << phase.value("status", std::string()) << "</li>";
    }
    html << "</ul></body></html>";
    core::write_text(html_path, html.str());
  }

  json manifest = {
      {"run_id", run_id},
      {"artifacts", json::array({
          {{"type", "config"}, {"phase", "INPUT_SCAN"}, {"path", (artifact_dir / "effective_config.json").string()}},
          {{"type", "events"}, {"path", (artifact_dir / "events.jsonl").string()}},
          {{"type", "registration"}, {"phase", "REGISTRATION"}, {"path", (artifact_dir / "preprocessing_registration.json").string()}},
          {{"type", "frame_quality"}, {"phase", "QUALITY_ANALYSIS"}, {"path", qa.csv_path.string()}},
          {{"type", "quality_analysis"}, {"phase", "FRAME_FILTERING"}, {"path", (artifact_dir / "quality_analysis.json").string()}},
          {{"type", "rejected_frames"}, {"phase", "FRAME_FILTERING"}, {"path", rejected_path.string()}},
          {{"type", "stacked_linear"}, {"phase", "STACKING"}, {"path", stack.stacked_linear_path.string()}},
          {{"type", "stacking_diagnostics"}, {"phase", "STACKING"}, {"path", stack.diagnostics_path.string()}},
          {{"type", "report"}, {"phase", "REPORT"}, {"path", report_path.string()}},
          {{"type", "report_markdown"}, {"phase", "REPORT"}, {"path", markdown_path.string()}},
          {{"type", "report_html"}, {"phase", "REPORT"}, {"path", html_path.string()}},
      })},
  };
  if (!stack.stacked_rgb_path.empty()) {
    manifest["artifacts"].push_back({{"type", "stacked_rgb"}, {"phase", "STACKING"}, {"path", stack.stacked_rgb_path.string()}});
  }
  for (const auto& artifact : post.artifacts) {
    manifest["artifacts"].push_back(artifact);
  }
  core::write_text(manifest_path, manifest.dump(2));
}

} // namespace

int preprocess_command(const std::string& config_path,
                       const std::string& runs_dir,
                       const std::string& project_root,
                       const std::string& run_id_override,
                       bool config_from_stdin) {
  try {
    const bool use_stdin_config = config_from_stdin || config_path == "-";
    std::string raw_config;
    if (use_stdin_config) {
      raw_config = read_all_stdin();
    } else {
      raw_config = core::read_text(fs::path(config_path));
    }
    if (raw_config.empty()) {
      std::cerr << "Error: preprocessing config is empty\n";
      return 1;
    }
    json parsed = json::parse(raw_config);
    prep::Config cfg = parse_preprocessing_config(parsed);

    const std::string run_id =
        run_id_override.empty() ? ("preprocessing_" + core::get_run_id()) : run_id_override;
    const fs::path runs = fs::path(runs_dir);
    fs::path run_dir;
    try {
      run_dir = fs::absolute(runs / run_id);
    } catch (...) {
      run_dir = (runs / run_id).lexically_normal();
    }
    const fs::path proj_root =
        project_root.empty() ? fs::current_path() : fs::path(project_root);
    const fs::path artifact_dir = run_dir / "artifacts" / "preprocess";
    try {
      fs::create_directories(artifact_dir);
      fs::create_directories(run_dir / "outputs");
    } catch (const std::exception& e) {
      std::cerr << "Error: cannot create preprocessing directories in " << run_dir
                << ": " << e.what() << "\n";
      return 1;
    }

    core::write_text(artifact_dir / "effective_config.json", config_to_json(cfg).dump(2));

    std::ofstream events(artifact_dir / "events.jsonl", std::ios::out | std::ios::trunc);
    if (!events.is_open()) {
      std::cerr << "Error: cannot open preprocessing event log\n";
      return 1;
    }
    runner::TeeBuf tee_buf(std::cout.rdbuf(), events.rdbuf());
    std::ostream event_out(&tee_buf);

    core::EventEmitter emitter;
    emitter.run_start(run_id, {{"process", "preprocessing"}, {"run_dir", run_dir.string()}}, event_out);

    runner::PreprocessPipelineContext pp;
    if (!runner::run_preprocess_pipeline(run_id, cfg, run_dir, proj_root,
                                         emitter, event_out, pp)) {
      return 1;
    }

    const auto star_metrics = measure_stars_for_quality(pp);
    runner::QualityAnalysisContext qa;
    if (!runner::run_quality_analysis(run_id, cfg, pp, star_metrics, run_dir,
                                      emitter, event_out, qa)) {
      return 1;
    }

    const PreprocessStackResult stack =
        run_preprocess_stacking(run_id, cfg, pp, qa, run_dir, emitter, event_out);
    const PreprocessPostprocessResult post =
        run_preprocess_postprocess(run_id, cfg, stack, pp, qa, run_dir, emitter, event_out);

    emitter.phase_start(run_id, prep::phase_to_string(prep::Phase::REPORT), event_out);
    write_report_and_manifest(run_dir, run_id, cfg, pp, qa, stack, post);
    emitter.phase_end(run_id, prep::phase_to_string(prep::Phase::REPORT), "ok",
                      {{"report_json", (artifact_dir / "preprocessing_report.json").string()},
                       {"report_markdown", (artifact_dir / "preprocessing_report.md").string()},
                       {"report_html", (artifact_dir / "preprocessing_report.html").string()}},
                      event_out);
    emitter.run_end(run_id, true, "ok", event_out);

    std::cout << json{
        {"ok", true},
        {"run_id", run_id},
        {"run_dir", run_dir.string()},
        {"artifacts_dir", artifact_dir.string()},
        {"report_json", (artifact_dir / "preprocessing_report.json").string()},
        {"frame_quality_csv", qa.csv_path.string()},
        {"stacked_linear", stack.stacked_linear_path.string()},
        {"stacked_rgb", stack.stacked_rgb_path.empty() ? json(nullptr) : json(stack.stacked_rgb_path.string())},
        {"stacking_diagnostics", stack.diagnostics_path.string()},
        {"report_markdown", (artifact_dir / "preprocessing_report.md").string()},
        {"report_html", (artifact_dir / "preprocessing_report.html").string()},
    }.dump() << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: preprocessing failed: " << e.what() << std::endl;
    return 1;
  }
}
