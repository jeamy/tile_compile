#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/errors.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <sstream>

namespace tile_compile::config {

namespace {

namespace fs = std::filesystem;

/// @brief Replaces backslashes inside double-quoted YAML strings with forward
/// slashes so that Windows paths (e.g. "C:\Users\...") do not cause yaml-cpp
/// to fail with "bad character found while scanning hex number".
/// Only processes characters inside double-quoted scalars; leaves the rest
/// of the YAML text untouched.
static std::string sanitize_yaml_windows_paths(const std::string& yaml) {
    std::string out;
    out.reserve(yaml.size());
    bool in_dq = false;
    for (size_t i = 0; i < yaml.size(); ++i) {
        char c = yaml[i];
        if (!in_dq) {
            if (c == '"') in_dq = true;
            out.push_back(c);
        } else {
            if (c == '\\' && i + 1 < yaml.size()) {
                char next = yaml[i + 1];
                // Keep valid YAML escape sequences (e.g. \n \t \\ \" \/ \uXXXX)
                static const char valid[] = "\"\\0abtnvfrNLP_e/xuU";
                bool valid_esc = false;
                for (char v : valid) { if (next == v) { valid_esc = true; break; } }
                if (!valid_esc) {
                    // Treat as literal backslash in a Windows path → forward slash
                    out.push_back('/');
                } else {
                    out.push_back(c);
                }
            } else {
                if (c == '"') in_dq = false;
                out.push_back(c);
            }
        }
    }
    return out;
}

/// @brief Checks between 0 1.
bool is_between_0_1(float v) { return v >= 0.0f && v <= 1.0f; }

/// @brief Returns true if a YAML node exists and is not null/empty.
/// @details Prevents yaml-cpp "bad conversion" errors when a field is
/// present in the YAML but has no value (e.g. "frames_min:" with nothing
/// after the colon).
bool yaml_has_value(const YAML::Node& node) {
    return node && !node.IsNull();
}

/// @brief Normalizes acceleration backend.
/// @details Part of YAML configuration loading, serialization, schema generation, and validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::string normalize_acceleration_backend(std::string value) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(), not_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(),
              value.end());
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  return value;
}

/// @brief Normalizes the reconstruction method and derived flags.
/// @details Ensures that Config::method is always set and AqmhConfig::enabled is derived from it.
/// This is the single source of truth for method determination. If method is missing, it defaults to "aqmh".
void normalizeMethod(Config &config) {
  if (config.method.empty()) {
    config.method = "aqmh";
  }
  if (config.method == "aqmh") {
    config.aqmh.enabled = true;
  } else if (config.method == "classic_tile_compile") {
    config.aqmh.enabled = false;
  }
}

/// @brief Reads float pair.
/// @details Part of YAML configuration loading, serialization, schema generation, and validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void read_float_pair(const YAML::Node &n, std::array<float, 2> &out) {
  if (n && n.IsSequence() && n.size() == 2) {
    out[0] = n[0].as<float>();
    out[1] = n[1].as<float>();
  }
}

/// @brief Reads int pair.
/// @details Part of YAML configuration loading, serialization, schema generation, and validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void read_int_pair(const YAML::Node &n, std::array<int, 2> &out) {
  if (n && n.IsSequence() && n.size() == 2) {
    out[0] = n[0].as<int>();
    out[1] = n[1].as<int>();
  }
}

/// @brief Implements scalar looks like float.
/// @details Part of YAML configuration loading, serialization, schema generation, and validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool scalar_looks_like_float(const std::string &raw) {
  return raw.find('.') != std::string::npos ||
         raw.find('e') != std::string::npos ||
         raw.find('E') != std::string::npos;
}

/// @brief Implements trim trailing zeros.
/// @details Part of YAML configuration loading, serialization, schema generation, and validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::string trim_trailing_zeros(std::string text) {
  const auto dot = text.find('.');
  if (dot == std::string::npos) {
    return text;
  }
  while (!text.empty() && text.back() == '0') {
    text.pop_back();
  }
  if (!text.empty() && text.back() == '.') {
    text.pop_back();
  }
  if (text == "-0") {
    return "0";
  }
  return text.empty() ? "0" : text;
}

/// @brief Formats config float scalar.
/// @details Part of YAML configuration loading, serialization, schema generation, and validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::string format_config_float_scalar(double value) {
  if (!std::isfinite(value)) {
    return "0";
  }

  const double rounded = std::round(value * 100.0) / 100.0;
  if (rounded == 0.0 && value != 0.0 && std::fabs(value) < 0.01) {
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(2) << value;
    return oss.str();
  }

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << rounded;
  return trim_trailing_zeros(oss.str());
}

/// @brief Implements round yaml numeric scalars inplace.
/// @details Part of YAML configuration loading, serialization, schema generation, and validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void round_yaml_numeric_scalars_inplace(YAML::Node node) {
  if (!node || node.IsNull()) {
    return;
  }
  if (node.IsMap()) {
    for (auto it = node.begin(); it != node.end(); ++it) {
      round_yaml_numeric_scalars_inplace(it->second);
    }
    return;
  }
  if (node.IsSequence()) {
    for (auto it = node.begin(); it != node.end(); ++it) {
      round_yaml_numeric_scalars_inplace(*it);
    }
    return;
  }
  if (!node.IsScalar()) {
    return;
  }

  const std::string raw = node.Scalar();
  if (!scalar_looks_like_float(raw)) {
    return;
  }

  char *end = nullptr;
  errno = 0;
  const double value = std::strtod(raw.c_str(), &end);
  if (errno != 0 || end == raw.c_str() || (end && *end != '\0') ||
      !std::isfinite(value)) {
    return;
  }
  node = format_config_float_scalar(value);
}

} // namespace

/// @brief Implements load.
/// @details Part of YAML configuration loading, serialization, schema generation, and validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Config Config::from_yaml_text(const std::string &yaml_text) {
  const std::string sanitized = sanitize_yaml_windows_paths(yaml_text);
  YAML::Node node = YAML::Load(sanitized);
  return from_yaml(node);
}

Config Config::load(const fs::path &path) {
  if (!fs::exists(path)) {
    throw ConfigError("Config file not found: " + path.string());
  }
  std::ifstream f(path);
  if (!f) throw ConfigError("Cannot open config file: " + path.string());
  std::ostringstream ss;
  ss << f.rdbuf();
  const std::string sanitized = sanitize_yaml_windows_paths(ss.str());
  YAML::Node node = YAML::Load(sanitized);
  return from_yaml(node);
}

/// @brief Implements from yaml.
/// @details Part of YAML configuration loading, serialization, schema generation, and validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Config Config::from_yaml(const YAML::Node &node) {
  Config cfg;

  if (node["method"]) {
    cfg.method = node["method"].as<std::string>();
  }

  if (node["pipeline"]) {
    auto p = node["pipeline"];
    if (yaml_has_value(p["mode"]))
      cfg.pipeline.mode = p["mode"].as<std::string>();
  }

  if (node["output"]) {
    auto o = node["output"];
    if (yaml_has_value(o["registered_dir"]))
      cfg.output.registered_dir = o["registered_dir"].as<std::string>();
    if (yaml_has_value(o["write_registered_frames"]))
      cfg.output.write_registered_frames =
          o["write_registered_frames"].as<bool>();
    if (yaml_has_value(o["crop_to_nonzero_bbox"]))
      cfg.output.crop_to_nonzero_bbox = o["crop_to_nonzero_bbox"].as<bool>();
  }

  if (node["data"]) {
    auto d = node["data"];
    if (yaml_has_value(d["image_width"]))
      cfg.data.image_width = d["image_width"].as<int>();
    if (yaml_has_value(d["image_height"]))
      cfg.data.image_height = d["image_height"].as<int>();
    if (yaml_has_value(d["color_mode"]))
      cfg.data.color_mode = d["color_mode"].as<std::string>();
    if (yaml_has_value(d["bayer_pattern"]))
      cfg.data.bayer_pattern = d["bayer_pattern"].as<std::string>();
    if (yaml_has_value(d["linear_required"]))
      cfg.data.linear_required = d["linear_required"].as<bool>();
  }

  if (node["linearity"]) {
    auto l = node["linearity"];
    if (yaml_has_value(l["enabled"]))
      cfg.linearity.enabled = l["enabled"].as<bool>();
    if (yaml_has_value(l["max_frames"]))
      cfg.linearity.max_frames = l["max_frames"].as<int>();
    if (yaml_has_value(l["min_overall_linearity"]))
      cfg.linearity.min_overall_linearity =
          l["min_overall_linearity"].as<float>();
    if (yaml_has_value(l["strictness"]))
      cfg.linearity.strictness = l["strictness"].as<std::string>();
  }

  if (node["calibration"]) {
    auto c = node["calibration"];
    if (yaml_has_value(c["use_bias"]))
      cfg.calibration.use_bias = c["use_bias"].as<bool>();
    if (yaml_has_value(c["use_dark"]))
      cfg.calibration.use_dark = c["use_dark"].as<bool>();
    if (yaml_has_value(c["use_flat"]))
      cfg.calibration.use_flat = c["use_flat"].as<bool>();
    if (yaml_has_value(c["bias_use_master"]))
      cfg.calibration.bias_use_master = c["bias_use_master"].as<bool>();
    if (yaml_has_value(c["dark_use_master"]))
      cfg.calibration.dark_use_master = c["dark_use_master"].as<bool>();
    if (yaml_has_value(c["dark_already_bias_corrected"])) {
      cfg.calibration.dark_already_bias_corrected =
          c["dark_already_bias_corrected"].as<bool>();
    }
    if (yaml_has_value(c["flat_use_master"]))
      cfg.calibration.flat_use_master = c["flat_use_master"].as<bool>();
    if (yaml_has_value(c["dark_auto_select"]))
      cfg.calibration.dark_auto_select = c["dark_auto_select"].as<bool>();
    if (yaml_has_value(c["dark_match_exposure_tolerance_percent"])) {
      cfg.calibration.dark_match_exposure_tolerance_percent =
          c["dark_match_exposure_tolerance_percent"].as<float>();
    }
    if (yaml_has_value(c["dark_match_use_temp"]))
      cfg.calibration.dark_match_use_temp = c["dark_match_use_temp"].as<bool>();
    if (yaml_has_value(c["dark_match_temp_tolerance_c"]))
      cfg.calibration.dark_match_temp_tolerance_c =
          c["dark_match_temp_tolerance_c"].as<float>();
    if (yaml_has_value(c["bias_dir"]))
      cfg.calibration.bias_dir = c["bias_dir"].as<std::string>();
    if (yaml_has_value(c["darks_dir"]))
      cfg.calibration.darks_dir = c["darks_dir"].as<std::string>();
    if (yaml_has_value(c["flats_dir"]))
      cfg.calibration.flats_dir = c["flats_dir"].as<std::string>();
    if (yaml_has_value(c["bias_master"]))
      cfg.calibration.bias_master = c["bias_master"].as<std::string>();
    if (yaml_has_value(c["dark_master"]))
      cfg.calibration.dark_master = c["dark_master"].as<std::string>();
    if (yaml_has_value(c["flat_master"]))
      cfg.calibration.flat_master = c["flat_master"].as<std::string>();
    if (yaml_has_value(c["pattern"]))
      cfg.calibration.pattern = c["pattern"].as<std::string>();
  }

  if (node["assumptions"]) {
    auto a = node["assumptions"];
    if (yaml_has_value(a["frames_min"]))
      cfg.assumptions.frames_min = a["frames_min"].as<int>();
    if (yaml_has_value(a["frames_reduced_threshold"]))
      cfg.assumptions.frames_reduced_threshold =
          a["frames_reduced_threshold"].as<int>();
    if (yaml_has_value(a["reduced_mode_skip_clustering"])) {
      cfg.assumptions.reduced_mode_skip_clustering =
          a["reduced_mode_skip_clustering"].as<bool>();
    }
    read_int_pair(a["reduced_mode_cluster_range"],
                  cfg.assumptions.reduced_mode_cluster_range);
  }

  if (node["normalization"]) {
    auto n = node["normalization"];
    if (yaml_has_value(n["enabled"]))
      cfg.normalization.enabled = n["enabled"].as<bool>();
    if (yaml_has_value(n["mode"]))
      cfg.normalization.mode = n["mode"].as<std::string>();
    if (yaml_has_value(n["per_channel"]))
      cfg.normalization.per_channel = n["per_channel"].as<bool>();
  }

  if (node["registration"]) {
    auto r = node["registration"];
    if (yaml_has_value(r["engine"]))
      cfg.registration.engine = r["engine"].as<std::string>();
    if (yaml_has_value(r["transform_model"]))
      cfg.registration.transform_model = r["transform_model"].as<std::string>();
    if (yaml_has_value(r["enable_star_pair_fallback"]))
      cfg.registration.enable_star_pair_fallback =
          r["enable_star_pair_fallback"].as<bool>();
    if (yaml_has_value(r["allow_rotation"]))
      cfg.registration.allow_rotation = r["allow_rotation"].as<bool>();
    if (yaml_has_value(r["star_topk"]))
      cfg.registration.star_topk = r["star_topk"].as<int>();
    if (yaml_has_value(r["star_min_inliers"]))
      cfg.registration.star_min_inliers = r["star_min_inliers"].as<int>();
    if (yaml_has_value(r["star_inlier_tol_px"]))
      cfg.registration.star_inlier_tol_px = r["star_inlier_tol_px"].as<float>();
    if (yaml_has_value(r["star_dist_bin_px"]))
      cfg.registration.star_dist_bin_px = r["star_dist_bin_px"].as<float>();
    if (yaml_has_value(r["reject_outliers"]))
      cfg.registration.reject_outliers = r["reject_outliers"].as<bool>();
    if (yaml_has_value(r["reject_cc_min_abs"]))
      cfg.registration.reject_cc_min_abs = r["reject_cc_min_abs"].as<float>();
    if (yaml_has_value(r["reject_shift_px_min"]))
      cfg.registration.reject_shift_px_min = r["reject_shift_px_min"].as<float>();
    if (yaml_has_value(r["reject_shift_median_multiplier"]))
      cfg.registration.reject_shift_median_multiplier =
          r["reject_shift_median_multiplier"].as<float>();
    if (yaml_has_value(r["reject_scale_min"]))
      cfg.registration.reject_scale_min = r["reject_scale_min"].as<float>();
    if (yaml_has_value(r["reject_scale_max"]))
      cfg.registration.reject_scale_max = r["reject_scale_max"].as<float>();
    if (yaml_has_value(r["auto_engine"]))
      cfg.registration.auto_engine = r["auto_engine"].as<bool>();
    if (yaml_has_value(r["auto_engine_rotation_threshold_deg"]))
      cfg.registration.auto_engine_rotation_threshold_deg =
          r["auto_engine_rotation_threshold_deg"].as<float>();
    // Neue Blind-Chain Parameter (§4.1, §8.B)
    if (yaml_has_value(r["max_blind_chain_depth"]))
      cfg.registration.max_blind_chain_depth = r["max_blind_chain_depth"].as<int>();
    if (yaml_has_value(r["blind_chain_strong_anchor_cc"]))
      cfg.registration.blind_chain_strong_anchor_cc = r["blind_chain_strong_anchor_cc"].as<float>();
    if (yaml_has_value(r["blind_chain_drift_threshold_px"]))
      cfg.registration.blind_chain_drift_threshold_px = r["blind_chain_drift_threshold_px"].as<float>();
    // Astrometric rescue (§4.13)
    if (yaml_has_value(r["use_astrometry"]))
      cfg.registration.use_astrometry = r["use_astrometry"].as<bool>();
    // Local background subtraction (§4.4, §8.D)
    if (yaml_has_value(r["enable_local_background_subtraction"]))
      cfg.registration.enable_local_background_subtraction = r["enable_local_background_subtraction"].as<bool>();
    if (yaml_has_value(r["star_shift_radius_px"]))
      cfg.registration.star_shift_radius_px = r["star_shift_radius_px"].as<float>();
    if (yaml_has_value(r["affine_refinement_enabled"]))
      cfg.registration.affine_refinement_enabled =
          r["affine_refinement_enabled"].as<bool>();
    if (yaml_has_value(r["smooth_local_refinement_enabled"]))
      cfg.registration.smooth_local_refinement_enabled =
          r["smooth_local_refinement_enabled"].as<bool>();
  }

  if (node["dithering"]) {
    auto d = node["dithering"];
    if (yaml_has_value(d["enabled"]))
      cfg.dithering.enabled = d["enabled"].as<bool>();
    if (yaml_has_value(d["min_shift_px"]))
      cfg.dithering.min_shift_px = d["min_shift_px"].as<float>();
  }

  if (node["tile_denoise"]) {
    auto td = node["tile_denoise"];
    if (yaml_has_value(td["soft_threshold"])) {
      auto st = td["soft_threshold"];
      if (yaml_has_value(st["enabled"]))
        cfg.tile_denoise.soft_threshold.enabled = st["enabled"].as<bool>();
      if (yaml_has_value(st["blur_kernel"]))
        cfg.tile_denoise.soft_threshold.blur_kernel = st["blur_kernel"].as<int>();
      if (yaml_has_value(st["alpha"]))
        cfg.tile_denoise.soft_threshold.alpha = st["alpha"].as<float>();
      if (yaml_has_value(st["skip_star_tiles"]))
        cfg.tile_denoise.soft_threshold.skip_star_tiles = st["skip_star_tiles"].as<bool>();
    }
    if (yaml_has_value(td["wiener"])) {
      auto w = td["wiener"];
      if (yaml_has_value(w["enabled"]))
        cfg.tile_denoise.wiener.enabled = w["enabled"].as<bool>();
      if (yaml_has_value(w["snr_threshold"]))
        cfg.tile_denoise.wiener.snr_threshold = w["snr_threshold"].as<float>();
      if (yaml_has_value(w["q_min"]))
        cfg.tile_denoise.wiener.q_min = w["q_min"].as<float>();
      if (yaml_has_value(w["q_max"]))
        cfg.tile_denoise.wiener.q_max = w["q_max"].as<float>();
      if (yaml_has_value(w["q_step"]))
        cfg.tile_denoise.wiener.q_step = w["q_step"].as<float>();
      if (yaml_has_value(w["min_snr"]))
        cfg.tile_denoise.wiener.min_snr = w["min_snr"].as<float>();
      if (yaml_has_value(w["max_iterations"]))
        cfg.tile_denoise.wiener.max_iterations = w["max_iterations"].as<int>();
    }
  }

  if (node["chroma_denoise"]) {
    auto cd = node["chroma_denoise"];
    if (yaml_has_value(cd["enabled"]))
      cfg.chroma_denoise.enabled = cd["enabled"].as<bool>();
    if (yaml_has_value(cd["color_space"]))
      cfg.chroma_denoise.color_space = cd["color_space"].as<std::string>();
    if (yaml_has_value(cd["apply_stage"]))
      cfg.chroma_denoise.apply_stage = cd["apply_stage"].as<std::string>();
    if (yaml_has_value(cd["protect_luma"]))
      cfg.chroma_denoise.protect_luma = cd["protect_luma"].as<bool>();
    if (yaml_has_value(cd["luma_guard_strength"]))
      cfg.chroma_denoise.luma_guard_strength = cd["luma_guard_strength"].as<float>();

    if (yaml_has_value(cd["star_protection"])) {
      auto sp = cd["star_protection"];
      if (yaml_has_value(sp["enabled"]))
        cfg.chroma_denoise.star_protection.enabled = sp["enabled"].as<bool>();
      if (yaml_has_value(sp["threshold_sigma"]))
        cfg.chroma_denoise.star_protection.threshold_sigma =
            sp["threshold_sigma"].as<float>();
      if (yaml_has_value(sp["dilate_px"]))
        cfg.chroma_denoise.star_protection.dilate_px = sp["dilate_px"].as<int>();
    }

    if (yaml_has_value(cd["structure_protection"])) {
      auto st = cd["structure_protection"];
      if (yaml_has_value(st["enabled"]))
        cfg.chroma_denoise.structure_protection.enabled = st["enabled"].as<bool>();
      if (yaml_has_value(st["gradient_percentile"]))
        cfg.chroma_denoise.structure_protection.gradient_percentile =
            st["gradient_percentile"].as<float>();
    }

    if (yaml_has_value(cd["chroma_wavelet"])) {
      auto cw = cd["chroma_wavelet"];
      if (yaml_has_value(cw["enabled"]))
        cfg.chroma_denoise.chroma_wavelet.enabled = cw["enabled"].as<bool>();
      if (yaml_has_value(cw["levels"]))
        cfg.chroma_denoise.chroma_wavelet.levels = cw["levels"].as<int>();
      if (yaml_has_value(cw["threshold_scale"]))
        cfg.chroma_denoise.chroma_wavelet.threshold_scale =
            cw["threshold_scale"].as<float>();
      if (yaml_has_value(cw["soft_k"]))
        cfg.chroma_denoise.chroma_wavelet.soft_k = cw["soft_k"].as<float>();
    }

    if (yaml_has_value(cd["chroma_bilateral"])) {
      auto cb = cd["chroma_bilateral"];
      if (yaml_has_value(cb["enabled"]))
        cfg.chroma_denoise.chroma_bilateral.enabled = cb["enabled"].as<bool>();
      if (yaml_has_value(cb["sigma_spatial"]))
        cfg.chroma_denoise.chroma_bilateral.sigma_spatial =
            cb["sigma_spatial"].as<float>();
      if (yaml_has_value(cb["sigma_range"]))
        cfg.chroma_denoise.chroma_bilateral.sigma_range =
            cb["sigma_range"].as<float>();
    }

    if (yaml_has_value(cd["blend"])) {
      auto b = cd["blend"];
      if (yaml_has_value(b["mode"]))
        cfg.chroma_denoise.blend.mode = b["mode"].as<std::string>();
      if (yaml_has_value(b["amount"]))
        cfg.chroma_denoise.blend.amount = b["amount"].as<float>();
    }
  }

  if (node["global_metrics"]) {
    auto gm = node["global_metrics"];
    if (yaml_has_value(gm["adaptive_weights"]))
      cfg.global_metrics.adaptive_weights = gm["adaptive_weights"].as<bool>();
    if (yaml_has_value(gm["weights"])) {
      auto w = gm["weights"];
      if (yaml_has_value(w["background"]))
        cfg.global_metrics.weights.background = w["background"].as<float>();
      if (yaml_has_value(w["noise"]))
        cfg.global_metrics.weights.noise = w["noise"].as<float>();
      if (yaml_has_value(w["gradient"]))
        cfg.global_metrics.weights.gradient = w["gradient"].as<float>();
      if (yaml_has_value(w["fwhm"]))
        cfg.global_metrics.weights.fwhm = w["fwhm"].as<float>();
      if (yaml_has_value(w["roundness"]))
        cfg.global_metrics.weights.roundness = w["roundness"].as<float>();
      if (yaml_has_value(w["star_count"]))
        cfg.global_metrics.weights.star_count = w["star_count"].as<float>();
    }
    read_float_pair(gm["clamp"], cfg.global_metrics.clamp);
    if (yaml_has_value(gm["weight_exponent_scale"]))
      cfg.global_metrics.weight_exponent_scale = gm["weight_exponent_scale"].as<float>();
  }

  if (node["tile"]) {
    auto t = node["tile"];
    if (yaml_has_value(t["size_factor"]))
      cfg.tile.size_factor = t["size_factor"].as<int>();
    if (yaml_has_value(t["min_size"]))
      cfg.tile.min_size = t["min_size"].as<int>();
    if (yaml_has_value(t["max_divisor"]))
      cfg.tile.max_divisor = t["max_divisor"].as<int>();
    if (yaml_has_value(t["overlap_fraction"]))
      cfg.tile.overlap_fraction = t["overlap_fraction"].as<float>();
    if (yaml_has_value(t["star_min_count"]))
      cfg.tile.star_min_count = t["star_min_count"].as<int>();
    if (yaml_has_value(t["star_soft_count"]))
      cfg.tile.star_soft_count = t["star_soft_count"].as<int>();
    else
      cfg.tile.star_soft_count = cfg.tile.star_min_count;
  }

  if (node["local_metrics"]) {
    auto lm = node["local_metrics"];
    read_float_pair(lm["clamp"], cfg.local_metrics.clamp);
    if (yaml_has_value(lm["neighborhood_normalization"])) {
      auto nn = lm["neighborhood_normalization"];
      if (yaml_has_value(nn["enabled"]))
        cfg.local_metrics.neighborhood_normalization.enabled =
            nn["enabled"].as<bool>();
      if (yaml_has_value(nn["radius"]))
        cfg.local_metrics.neighborhood_normalization.radius =
            nn["radius"].as<int>();
      if (yaml_has_value(nn["blend"]))
        cfg.local_metrics.neighborhood_normalization.blend =
            nn["blend"].as<float>();
    }
    if (yaml_has_value(lm["spatial_regularization"])) {
      auto sr = lm["spatial_regularization"];
      if (yaml_has_value(sr["enabled"]))
        cfg.local_metrics.spatial_regularization.enabled =
            sr["enabled"].as<bool>();
      if (yaml_has_value(sr["lambda"]))
        cfg.local_metrics.spatial_regularization.lambda =
            sr["lambda"].as<float>();
      if (yaml_has_value(sr["passes"]))
        cfg.local_metrics.spatial_regularization.passes =
            sr["passes"].as<int>();
      if (yaml_has_value(sr["tau_local"]))
        cfg.local_metrics.spatial_regularization.tau_local =
            sr["tau_local"].as<float>();
    }
    if (lm["star_mode"] && lm["star_mode"]["weights"]) {
      auto w = lm["star_mode"]["weights"];
      if (yaml_has_value(w["fwhm"]))
        cfg.local_metrics.star_mode.weights.fwhm = w["fwhm"].as<float>();
      if (yaml_has_value(w["roundness"]))
        cfg.local_metrics.star_mode.weights.roundness =
            w["roundness"].as<float>();
      if (yaml_has_value(w["contrast"]))
        cfg.local_metrics.star_mode.weights.contrast =
            w["contrast"].as<float>();
    }
    if (yaml_has_value(lm["structure_mode"])) {
      auto sm = lm["structure_mode"];
      if (yaml_has_value(sm["background_weight"]))
        cfg.local_metrics.structure_mode.background_weight =
            sm["background_weight"].as<float>();
      if (yaml_has_value(sm["metric_weight"]))
        cfg.local_metrics.structure_mode.metric_weight =
            sm["metric_weight"].as<float>();
    }
    if (yaml_has_value(lm["k_local"]))
      cfg.local_metrics.k_local = lm["k_local"].as<float>();
  }

  if (node["aqmh"]) {
    auto a = node["aqmh"];
    if (yaml_has_value(a["enabled"]))
      cfg.aqmh.enabled = a["enabled"].as<bool>();
    if (yaml_has_value(a["pyramid"])) {
      auto p = a["pyramid"];
      if (yaml_has_value(p["scales"]))
        cfg.aqmh.pyramid.scales = p["scales"].as<int>();
      if (yaml_has_value(p["base_window_px"]))
        cfg.aqmh.pyramid.base_window_px = p["base_window_px"].as<int>();
      if (yaml_has_value(p["w_sharp"]))
        cfg.aqmh.pyramid.w_sharp = p["w_sharp"].as<float>();
      if (yaml_has_value(p["w_snr"]))
        cfg.aqmh.pyramid.w_snr = p["w_snr"].as<float>();
      if (yaml_has_value(p["score_scale"]))
        cfg.aqmh.pyramid.score_scale = p["score_scale"].as<float>();
      if (yaml_has_value(p["k_artifact"]))
        cfg.aqmh.pyramid.k_artifact = p["k_artifact"].as<float>();
      if (yaml_has_value(p["frac_artifact_max"]))
        cfg.aqmh.pyramid.frac_artifact_max =
            p["frac_artifact_max"].as<float>();
    }
    if (yaml_has_value(a["storage"])) {
      auto s = a["storage"];
      if (yaml_has_value(s["resolution_divisor"]))
        cfg.aqmh.storage.resolution_divisor =
            s["resolution_divisor"].as<int>();
      if (yaml_has_value(s["dtype"]))
        cfg.aqmh.storage.dtype = s["dtype"].as<std::string>();
      if (yaml_has_value(s["max_resident_maps"]))
        cfg.aqmh.storage.max_resident_maps =
            s["max_resident_maps"].as<int>();
    }
    if (yaml_has_value(a["global_quality"])) {
      auto g = a["global_quality"];
      if (yaml_has_value(g["g_floor"])) cfg.aqmh.global_quality.g_floor = g["g_floor"].as<float>();
      if (yaml_has_value(g["g_w_sharp"])) cfg.aqmh.global_quality.g_w_sharp = g["g_w_sharp"].as<float>();
      if (yaml_has_value(g["g_w_snr"])) cfg.aqmh.global_quality.g_w_snr = g["g_w_snr"].as<float>();
      if (yaml_has_value(g["g_w_background_penalty"]))
        cfg.aqmh.global_quality.g_w_background_penalty =
            g["g_w_background_penalty"].as<float>();
      if (yaml_has_value(g["g_k_scale"]))
        cfg.aqmh.global_quality.g_k_scale = g["g_k_scale"].as<float>();
    }
    if (yaml_has_value(a["cherry_pick"])) {
      auto cp = a["cherry_pick"];
      if (yaml_has_value(cp["enabled"]))
        cfg.aqmh.cherry_pick.enabled = cp["enabled"].as<bool>();
      if (yaml_has_value(cp["mode"]))
        cfg.aqmh.cherry_pick.mode = cp["mode"].as<std::string>();
      if (yaml_has_value(cp["k_frac"]))
        cfg.aqmh.cherry_pick.k_frac = cp["k_frac"].as<float>();
      if (yaml_has_value(cp["k_min_required"]))
        cfg.aqmh.cherry_pick.k_min_required = cp["k_min_required"].as<int>();
      if (yaml_has_value(cp["margin_min"]))
        cfg.aqmh.cherry_pick.margin_min = cp["margin_min"].as<float>();
      if (yaml_has_value(cp["reject_below_best_fraction"]))
        cfg.aqmh.cherry_pick.reject_below_best_fraction =
            cp["reject_below_best_fraction"].as<float>();
      if (yaml_has_value(cp["min_keep_fraction"]))
        cfg.aqmh.cherry_pick.min_keep_fraction =
            cp["min_keep_fraction"].as<float>();
      if (yaml_has_value(cp["tiered_k_frac"])) {
        cfg.aqmh.cherry_pick.tiered_k_frac.clear();
        for (const auto &item : cp["tiered_k_frac"]) {
          AqmhCherryPickConfig::Tier tier;
          if (yaml_has_value(item["min_n_rankable"]))
            tier.min_n_rankable = item["min_n_rankable"].as<int>();
          if (yaml_has_value(item["k_frac"]))
            tier.k_frac = item["k_frac"].as<float>();
          cfg.aqmh.cherry_pick.tiered_k_frac.push_back(tier);
        }
      }
    }
    if (yaml_has_value(a["diagnostics"])) {
      auto d = a["diagnostics"];
      if (yaml_has_value(d["tau_artifact"]))
        cfg.aqmh.diagnostics.tau_artifact = d["tau_artifact"].as<float>();
      if (yaml_has_value(d["q_region"]))
        cfg.aqmh.diagnostics.q_region = d["q_region"].as<float>();
      if (yaml_has_value(d["r_morph_canvas_px"]))
        cfg.aqmh.diagnostics.r_morph_canvas_px =
            d["r_morph_canvas_px"].as<int>();
      // NEW FIELDS:
      if (yaml_has_value(d["enabled"]))
        cfg.aqmh.diagnostics.enabled = d["enabled"].as<bool>();
      if (yaml_has_value(d["level"]))
        cfg.aqmh.diagnostics.level = d["level"].as<std::string>();
      if (yaml_has_value(d["per_frame_blocks"]))
        cfg.aqmh.diagnostics.per_frame_blocks = d["per_frame_blocks"].as<bool>();
      if (yaml_has_value(d["heatmaps"]))
        cfg.aqmh.diagnostics.heatmaps = d["heatmaps"].as<bool>();
      if (yaml_has_value(d["regions"]))
        cfg.aqmh.diagnostics.regions = d["regions"].as<bool>();
      if (yaml_has_value(d["format"]))
        cfg.aqmh.diagnostics.format = d["format"].as<std::string>();
      if (yaml_has_value(d["binary_block_size_px"]))
        cfg.aqmh.diagnostics.binary_block_size_px = d["binary_block_size_px"].as<int>();
    }
    if (yaml_has_value(a["reconstruction"])) {
      auto r = a["reconstruction"];
      const bool has_clip_sigma = static_cast<bool>(r["clip_sigma"]);
      const bool has_clip_sigma_low = static_cast<bool>(r["clip_sigma_low"]);
      const bool has_clip_sigma_high = static_cast<bool>(r["clip_sigma_high"]);
      if (has_clip_sigma) {
        cfg.aqmh.reconstruction.clip_sigma = r["clip_sigma"].as<float>();
        if (!has_clip_sigma_low)
          cfg.aqmh.reconstruction.clip_sigma_low =
              cfg.aqmh.reconstruction.clip_sigma;
        if (!has_clip_sigma_high)
          cfg.aqmh.reconstruction.clip_sigma_high =
              cfg.aqmh.reconstruction.clip_sigma;
      }
      if (yaml_has_value(r["clip_sigma_low"]))
        cfg.aqmh.reconstruction.clip_sigma_low =
            r["clip_sigma_low"].as<float>();
      if (yaml_has_value(r["clip_sigma_high"]))
        cfg.aqmh.reconstruction.clip_sigma_high =
            r["clip_sigma_high"].as<float>();
      if (yaml_has_value(r["clip_iterations"])) cfg.aqmh.reconstruction.clip_iterations = r["clip_iterations"].as<int>();
      if (yaml_has_value(r["min_fraction"])) cfg.aqmh.reconstruction.min_fraction = r["min_fraction"].as<float>();
      if (yaml_has_value(r["min_n_eff"])) cfg.aqmh.reconstruction.min_n_eff = r["min_n_eff"].as<float>();
      // NEW FIELDS:
      if (yaml_has_value(r["chunk_rows"]))
        cfg.aqmh.reconstruction.chunk_rows = r["chunk_rows"].as<int>();
      if (yaml_has_value(r["memory_budget_mb"]))
        cfg.aqmh.reconstruction.memory_budget_mb = r["memory_budget_mb"].as<size_t>();
      if (yaml_has_value(r["delete_prewarped_cache_after_run"]))
        cfg.aqmh.reconstruction.delete_prewarped_cache_after_run =
            r["delete_prewarped_cache_after_run"].as<bool>();
      if (yaml_has_value(r["prewarp_interpolation"]))
        cfg.aqmh.reconstruction.prewarp_interpolation =
            r["prewarp_interpolation"].as<std::string>();
      if (yaml_has_value(r["debayer_first"]))
        cfg.aqmh.reconstruction.debayer_first =
            r["debayer_first"].as<bool>();
      if (yaml_has_value(r["pre_debayer_method"]))
        cfg.aqmh.reconstruction.pre_debayer_method =
            r["pre_debayer_method"].as<std::string>();
      if (yaml_has_value(r["rgb_q_map_mode"]))
        cfg.aqmh.reconstruction.rgb_q_map_mode =
            r["rgb_q_map_mode"].as<std::string>();
      if (yaml_has_value(r["rgb_memory_strategy"]))
        cfg.aqmh.reconstruction.rgb_memory_strategy =
            r["rgb_memory_strategy"].as<std::string>();
      if (yaml_has_value(r["registration_weight_guard"]))
        cfg.aqmh.reconstruction.registration_weight_guard =
            r["registration_weight_guard"].as<bool>();
      if (yaml_has_value(r["registration_weight_floor"]))
        cfg.aqmh.reconstruction.registration_weight_floor =
            r["registration_weight_floor"].as<float>();
      if (yaml_has_value(r["registration_cc_floor"]))
        cfg.aqmh.reconstruction.registration_cc_floor =
            r["registration_cc_floor"].as<float>();
      if (yaml_has_value(r["registration_cc_full"]))
        cfg.aqmh.reconstruction.registration_cc_full =
            r["registration_cc_full"].as<float>();
      if (yaml_has_value(r["registration_sequential_factor"]))
        cfg.aqmh.reconstruction.registration_sequential_factor =
            r["registration_sequential_factor"].as<float>();
      if (yaml_has_value(r["registration_predicted_factor"]))
        cfg.aqmh.reconstruction.registration_predicted_factor =
            r["registration_predicted_factor"].as<float>();
      if (yaml_has_value(r["registration_chain_depth_penalty"]))
        cfg.aqmh.reconstruction.registration_chain_depth_penalty =
            r["registration_chain_depth_penalty"].as<float>();
      if (yaml_has_value(r["registration_chain_depth_max_penalty"]))
        cfg.aqmh.reconstruction.registration_chain_depth_max_penalty =
            r["registration_chain_depth_max_penalty"].as<float>();
      if (yaml_has_value(r["structure_mask_low_q"]))
        cfg.aqmh.reconstruction.structure_mask_low_q =
            r["structure_mask_low_q"].as<float>();
      if (yaml_has_value(r["structure_mask_high_q"]))
        cfg.aqmh.reconstruction.structure_mask_high_q =
            r["structure_mask_high_q"].as<float>();
      if (yaml_has_value(r["structure_mask_blur_sigma_px"]))
        cfg.aqmh.reconstruction.structure_mask_blur_sigma_px =
            r["structure_mask_blur_sigma_px"].as<float>();
    }
    if (yaml_has_value(a["validation"])) {
      auto v = a["validation"];
      if (yaml_has_value(v["max_seam_score_regression"])) cfg.aqmh.validation.max_seam_score_regression = v["max_seam_score_regression"].as<float>();
      if (yaml_has_value(v["max_fwhm_regression"])) cfg.aqmh.validation.max_fwhm_regression = v["max_fwhm_regression"].as<float>();
      if (yaml_has_value(v["max_background_rms_regression"])) cfg.aqmh.validation.max_background_rms_regression = v["max_background_rms_regression"].as<float>();
      if (yaml_has_value(v["max_tail11_abs_regression"])) cfg.aqmh.validation.max_tail11_abs_regression = v["max_tail11_abs_regression"].as<float>();
      if (yaml_has_value(v["max_elongation_regression"])) cfg.aqmh.validation.max_elongation_regression = v["max_elongation_regression"].as<float>();
    }
  }

  if (node["synthetic"]) {
    auto s = node["synthetic"];
    if (yaml_has_value(s["weighting"]))
      cfg.synthetic.weighting = s["weighting"].as<std::string>();
    if (yaml_has_value(s["frames_min"]))
      cfg.synthetic.frames_min = s["frames_min"].as<int>();
    if (yaml_has_value(s["frames_max"]))
      cfg.synthetic.frames_max = s["frames_max"].as<int>();
    if (yaml_has_value(s["clustering"])) {
      auto cl = s["clustering"];
      if (yaml_has_value(cl["mode"]))
        cfg.synthetic.clustering.mode = cl["mode"].as<std::string>();
      read_int_pair(cl["cluster_count_range"],
                    cfg.synthetic.clustering.cluster_count_range);
    }
  }

  if (node["astrometry"]) {
    auto a = node["astrometry"];
    if (yaml_has_value(a["enabled"]))
      cfg.astrometry.enabled = a["enabled"].as<bool>();
    if (a["astap_bin"] && !a["astap_bin"].IsNull())
      cfg.astrometry.astap_bin = a["astap_bin"].as<std::string>();
    if (a["astap_data_dir"] && !a["astap_data_dir"].IsNull())
      cfg.astrometry.astap_data_dir = a["astap_data_dir"].as<std::string>();
    if (yaml_has_value(a["search_radius"]))
      cfg.astrometry.search_radius = a["search_radius"].as<int>();
  }

  if (node["bge"]) {
    auto b = node["bge"];
    const bool has_method = static_cast<bool>(b["method"]);
    if (yaml_has_value(b["enabled"]))
      cfg.bge.enabled = b["enabled"].as<bool>();
    if (has_method) {
      cfg.bge.method = b["method"].as<std::string>();
      cfg.bge.enabled = (cfg.bge.method != "none");
    } else {
      cfg.bge.method = cfg.bge.enabled ? "classic" : "none";
    }
    if (yaml_has_value(b["autobge"])) {
      auto a = b["autobge"];
      if (yaml_has_value(a["num_sample_points"]))
        cfg.bge.autobge.num_sample_points = a["num_sample_points"].as<int>();
      if (yaml_has_value(a["poly_degree"]))
        cfg.bge.autobge.poly_degree = a["poly_degree"].as<int>();
      if (yaml_has_value(a["rbf_smooth"]))
        cfg.bge.autobge.rbf_smooth = a["rbf_smooth"].as<float>();
      if (yaml_has_value(a["downsample_scale"]))
        cfg.bge.autobge.downsample_scale = a["downsample_scale"].as<int>();
      if (yaml_has_value(a["patch_size"]))
        cfg.bge.autobge.patch_size = a["patch_size"].as<int>();
      if (yaml_has_value(a["patch_estimator"]))
        cfg.bge.autobge.patch_estimator = a["patch_estimator"].as<std::string>();
      if (yaml_has_value(a["stretch_mode"]))
        cfg.bge.autobge.stretch_mode = a["stretch_mode"].as<std::string>();
      if (yaml_has_value(a["stretch_target_median"]))
        cfg.bge.autobge.stretch_target_median =
            a["stretch_target_median"].as<float>();
      if (yaml_has_value(a["border_margin"]))
        cfg.bge.autobge.border_margin = a["border_margin"].as<int>();
      if (yaml_has_value(a["bright_exclusion_fraction"]))
        cfg.bge.autobge.bright_exclusion_fraction =
            a["bright_exclusion_fraction"].as<float>();
      if (yaml_has_value(a["gradient_descent_max_iters"]))
        cfg.bge.autobge.gradient_descent_max_iters =
            a["gradient_descent_max_iters"].as<int>();
      if (yaml_has_value(a["random_seed"]))
        cfg.bge.autobge.random_seed = a["random_seed"].as<int>();
      if (yaml_has_value(a["normalize_between_stages"]))
        cfg.bge.autobge.normalize_between_stages =
            a["normalize_between_stages"].as<bool>();
      if (yaml_has_value(a["apply_guards"]))
        cfg.bge.autobge.apply_guards = a["apply_guards"].as<bool>();
      if (yaml_has_value(a["mono_mode"]))
        cfg.bge.autobge.mono_mode = a["mono_mode"].as<std::string>();
      auto read_autobge_point = [](const YAML::Node &point_node,
                                   const char *field_name)
          -> std::array<float, 2> {
        float x = 0.0f;
        float y = 0.0f;
        if (point_node.IsSequence() && point_node.size() == 2) {
          x = point_node[0].as<float>();
          y = point_node[1].as<float>();
        } else if (point_node.IsMap() && point_node["x"] && point_node["y"]) {
          x = point_node["x"].as<float>();
          y = point_node["y"].as<float>();
        } else {
          throw ValidationError(std::string(field_name) +
                                " points must be [x,y] or {x,y}");
        }
        if (x < 0.0f || x > 1.0f || y < 0.0f || y > 1.0f) {
          throw ValidationError(std::string(field_name) +
                                " coordinates must be in [0,1]");
        }
        return {x, y};
      };
      if (yaml_has_value(a["exclusion_polygons"])) {
        cfg.bge.autobge.exclusion_polygons.clear();
        for (const auto &polygon_node : a["exclusion_polygons"]) {
          std::vector<std::array<float, 2>> polygon;
          for (const auto &point_node : polygon_node) {
            polygon.push_back(read_autobge_point(
                point_node, "bge.autobge.exclusion_polygons"));
          }
          cfg.bge.autobge.exclusion_polygons.push_back(std::move(polygon));
        }
      }
      if (yaml_has_value(a["user_sample_points"])) {
        cfg.bge.autobge.user_sample_points.clear();
        for (const auto &point_node : a["user_sample_points"]) {
          cfg.bge.autobge.user_sample_points.push_back(read_autobge_point(
              point_node, "bge.autobge.user_sample_points"));
        }
      }
    }
    if (yaml_has_value(b["sample_quantile"]))
      cfg.bge.sample_quantile = b["sample_quantile"].as<float>();
    if (yaml_has_value(b["sample_estimator"]))
      cfg.bge.sample_estimator = b["sample_estimator"].as<std::string>();
    if (yaml_has_value(b["min_sample_bg_value"]))
      cfg.bge.min_sample_bg_value = b["min_sample_bg_value"].as<float>();
    if (yaml_has_value(b["structure_thresh_percentile"]))
      cfg.bge.structure_thresh_percentile = b["structure_thresh_percentile"].as<float>();
    if (yaml_has_value(b["min_tiles_per_cell"]))
      cfg.bge.min_tiles_per_cell = b["min_tiles_per_cell"].as<int>();
    if (yaml_has_value(b["min_valid_sample_fraction_for_apply"]))
      cfg.bge.min_valid_sample_fraction_for_apply =
          b["min_valid_sample_fraction_for_apply"].as<float>();
    if (yaml_has_value(b["min_valid_samples_for_apply"]))
      cfg.bge.min_valid_samples_for_apply =
          b["min_valid_samples_for_apply"].as<int>();
    
    if (yaml_has_value(b["mask"])) {
      auto m = b["mask"];
      if (yaml_has_value(m["star_dilate_px"]))
        cfg.bge.mask.star_dilate_px = m["star_dilate_px"].as<int>();
      if (yaml_has_value(m["sat_dilate_px"]))
        cfg.bge.mask.sat_dilate_px = m["sat_dilate_px"].as<int>();
    }
    
    if (yaml_has_value(b["grid"])) {
      auto g = b["grid"];
      if (yaml_has_value(g["N_g"]))
        cfg.bge.grid.N_g = g["N_g"].as<int>();
      if (yaml_has_value(g["G_min_px"]))
        cfg.bge.grid.G_min_px = g["G_min_px"].as<int>();
      if (yaml_has_value(g["G_max_fraction"]))
        cfg.bge.grid.G_max_fraction = g["G_max_fraction"].as<float>();
      if (yaml_has_value(g["insufficient_cell_strategy"]))
        cfg.bge.grid.insufficient_cell_strategy = g["insufficient_cell_strategy"].as<std::string>();
    }
    
    if (yaml_has_value(b["fit"])) {
      auto f = b["fit"];
      if (yaml_has_value(f["method"]))
        cfg.bge.fit.method = f["method"].as<std::string>();
      if (yaml_has_value(f["robust_loss"]))
        cfg.bge.fit.robust_loss = f["robust_loss"].as<std::string>();
      if (yaml_has_value(f["huber_delta"]))
        cfg.bge.fit.huber_delta = f["huber_delta"].as<float>();
      if (yaml_has_value(f["irls_max_iterations"]))
        cfg.bge.fit.irls_max_iterations = f["irls_max_iterations"].as<int>();
      if (yaml_has_value(f["irls_tolerance"]))
        cfg.bge.fit.irls_tolerance = f["irls_tolerance"].as<float>();
      if (yaml_has_value(f["polynomial_order"]))
        cfg.bge.fit.polynomial_order = f["polynomial_order"].as<int>();
      if (yaml_has_value(f["rbf_phi"]))
        cfg.bge.fit.rbf_phi = f["rbf_phi"].as<std::string>();
      if (yaml_has_value(f["rbf_mu_factor"]))
        cfg.bge.fit.rbf_mu_factor = f["rbf_mu_factor"].as<float>();
      if (yaml_has_value(f["rbf_lambda"]))
        cfg.bge.fit.rbf_lambda = f["rbf_lambda"].as<float>();
      if (yaml_has_value(f["rbf_epsilon"]))
        cfg.bge.fit.rbf_epsilon = f["rbf_epsilon"].as<float>();
    }

    if (yaml_has_value(b["autotune"])) {
      auto a = b["autotune"];
      if (yaml_has_value(a["enabled"]))
        cfg.bge.autotune.enabled = a["enabled"].as<bool>();
      if (yaml_has_value(a["max_evals"]))
        cfg.bge.autotune.max_evals = a["max_evals"].as<int>();
      if (yaml_has_value(a["holdout_fraction"]))
        cfg.bge.autotune.holdout_fraction = a["holdout_fraction"].as<float>();
      if (yaml_has_value(a["alpha_flatness"]))
        cfg.bge.autotune.alpha_flatness = a["alpha_flatness"].as<float>();
      if (yaml_has_value(a["beta_roughness"]))
        cfg.bge.autotune.beta_roughness = a["beta_roughness"].as<float>();
      if (yaml_has_value(a["strategy"]))
        cfg.bge.autotune.strategy = a["strategy"].as<std::string>();
    }
    if (yaml_has_value(b["tile_weight_lambda_structure"]))
      cfg.bge.tile_weight_lambda_structure =
          b["tile_weight_lambda_structure"].as<float>();
  }

  if (node["pcc"]) {
    auto p = node["pcc"];
    if (yaml_has_value(p["enabled"]))
      cfg.pcc.enabled = p["enabled"].as<bool>();
    if (yaml_has_value(p["source"]))
      cfg.pcc.source = p["source"].as<std::string>();
    if (yaml_has_value(p["mag_limit"]))
      cfg.pcc.mag_limit = p["mag_limit"].as<float>();
    if (yaml_has_value(p["mag_bright_limit"]))
      cfg.pcc.mag_bright_limit = p["mag_bright_limit"].as<float>();
    if (yaml_has_value(p["aperture_radius_px"]))
      cfg.pcc.aperture_radius_px = p["aperture_radius_px"].as<float>();
    if (yaml_has_value(p["annulus_inner_px"]))
      cfg.pcc.annulus_inner_px = p["annulus_inner_px"].as<float>();
    if (yaml_has_value(p["annulus_outer_px"]))
      cfg.pcc.annulus_outer_px = p["annulus_outer_px"].as<float>();
    if (yaml_has_value(p["min_stars"]))
      cfg.pcc.min_stars = p["min_stars"].as<int>();
    if (yaml_has_value(p["sigma_clip"]))
      cfg.pcc.sigma_clip = p["sigma_clip"].as<float>();
    if (yaml_has_value(p["background_model"]))
      cfg.pcc.background_model = p["background_model"].as<std::string>();
    if (yaml_has_value(p["max_condition_number"]))
      cfg.pcc.max_condition_number = p["max_condition_number"].as<float>();
    if (yaml_has_value(p["max_residual_rms"]))
      cfg.pcc.max_residual_rms = p["max_residual_rms"].as<float>();
    if (yaml_has_value(p["radii_mode"]))
      cfg.pcc.radii_mode = p["radii_mode"].as<std::string>();
    if (yaml_has_value(p["aperture_fwhm_mult"]))
      cfg.pcc.aperture_fwhm_mult = p["aperture_fwhm_mult"].as<float>();
    if (yaml_has_value(p["annulus_inner_fwhm_mult"]))
      cfg.pcc.annulus_inner_fwhm_mult = p["annulus_inner_fwhm_mult"].as<float>();
    if (yaml_has_value(p["annulus_outer_fwhm_mult"]))
      cfg.pcc.annulus_outer_fwhm_mult = p["annulus_outer_fwhm_mult"].as<float>();
    if (yaml_has_value(p["min_aperture_px"]))
      cfg.pcc.min_aperture_px = p["min_aperture_px"].as<float>();
    if (yaml_has_value(p["siril_catalog_dir"]))
      cfg.pcc.siril_catalog_dir = p["siril_catalog_dir"].as<std::string>();

    if (yaml_has_value(p["apply_attenuation"]))
      cfg.pcc.apply_attenuation = p["apply_attenuation"].as<bool>();
    if (yaml_has_value(p["chroma_strength"]))
      cfg.pcc.chroma_strength = p["chroma_strength"].as<float>();
    if (yaml_has_value(p["k_max"]))
      cfg.pcc.k_max = p["k_max"].as<float>();
    if (yaml_has_value(p["background_neutralization_mode"]))
      cfg.pcc.background_neutralization_mode =
          p["background_neutralization_mode"].as<std::string>();
  }

  if (node["hypermetric_stretch"]) {
    auto h = node["hypermetric_stretch"];
    if (yaml_has_value(h["enabled"]))
      cfg.hypermetric_stretch.enabled = h["enabled"].as<bool>();
    if (yaml_has_value(h["require_successful_pcc"]))
      cfg.hypermetric_stretch.require_successful_pcc =
          h["require_successful_pcc"].as<bool>();
    if (yaml_has_value(h["mode"]))
      cfg.hypermetric_stretch.mode = h["mode"].as<std::string>();
    if (yaml_has_value(h["sensor_profile"]))
      cfg.hypermetric_stretch.sensor_profile =
          h["sensor_profile"].as<std::string>();
    if (yaml_has_value(h["fallback_profile"]))
      cfg.hypermetric_stretch.fallback_profile =
          h["fallback_profile"].as<std::string>();
    if (yaml_has_value(h["adaptive_anchor"]))
      cfg.hypermetric_stretch.adaptive_anchor =
          h["adaptive_anchor"].as<bool>();
    if (yaml_has_value(h["target_bg"]))
      cfg.hypermetric_stretch.target_bg = h["target_bg"].as<float>();
    if (yaml_has_value(h["protect_b"]))
      cfg.hypermetric_stretch.protect_b = h["protect_b"].as<float>();
    if (yaml_has_value(h["convergence_power"]))
      cfg.hypermetric_stretch.convergence_power =
          h["convergence_power"].as<float>();
    if (yaml_has_value(h["log_d_mode"]))
      cfg.hypermetric_stretch.log_d_mode = h["log_d_mode"].as<std::string>();
    if (yaml_has_value(h["fixed_log_d"]))
      cfg.hypermetric_stretch.fixed_log_d = h["fixed_log_d"].as<float>();
    if (yaml_has_value(h["color_strategy"]))
      cfg.hypermetric_stretch.color_strategy =
          h["color_strategy"].as<std::string>();
    if (yaml_has_value(h["fixed_color_strategy"]))
      cfg.hypermetric_stretch.fixed_color_strategy =
          h["fixed_color_strategy"].as<float>();
    if (yaml_has_value(h["color_grip"]))
      cfg.hypermetric_stretch.color_grip = h["color_grip"].as<float>();
    if (yaml_has_value(h["shadow_convergence"]))
      cfg.hypermetric_stretch.shadow_convergence =
          h["shadow_convergence"].as<float>();
    if (yaml_has_value(h["linear_expansion"]))
      cfg.hypermetric_stretch.linear_expansion =
          h["linear_expansion"].as<float>();
    if (yaml_has_value(h["write_channels"]))
      cfg.hypermetric_stretch.write_channels = h["write_channels"].as<bool>();
    if (yaml_has_value(h["output_rgb"]))
      cfg.hypermetric_stretch.output_rgb = h["output_rgb"].as<std::string>();
  }

  if (node["stacking"]) {
    auto st = node["stacking"];
    if (yaml_has_value(st["method"]))
      cfg.stacking.method = st["method"].as<std::string>();
    if (yaml_has_value(st["common_overlap_required_fraction"]))
      cfg.stacking.common_overlap_required_fraction =
          st["common_overlap_required_fraction"].as<float>();
    if (yaml_has_value(st["tile_common_valid_min_fraction"]))
      cfg.stacking.tile_common_valid_min_fraction =
          st["tile_common_valid_min_fraction"].as<float>();
    if (yaml_has_value(st["sigma_clip"])) {
      auto sc = st["sigma_clip"];
      if (yaml_has_value(sc["sigma_low"]))
        cfg.stacking.sigma_clip.sigma_low = sc["sigma_low"].as<float>();
      if (yaml_has_value(sc["sigma_high"]))
        cfg.stacking.sigma_clip.sigma_high = sc["sigma_high"].as<float>();
      if (yaml_has_value(sc["max_iters"]))
        cfg.stacking.sigma_clip.max_iters = sc["max_iters"].as<int>();
      if (yaml_has_value(sc["min_fraction"]))
        cfg.stacking.sigma_clip.min_fraction = sc["min_fraction"].as<float>();
    }
    if (yaml_has_value(st["cluster_quality_weighting"])) {
      auto cqw = st["cluster_quality_weighting"];
      if (yaml_has_value(cqw["enabled"]))
        cfg.stacking.cluster_quality_weighting.enabled =
            cqw["enabled"].as<bool>();
      if (yaml_has_value(cqw["kappa_cluster"]))
        cfg.stacking.cluster_quality_weighting.kappa_cluster =
            cqw["kappa_cluster"].as<float>();
      if (yaml_has_value(cqw["cap_enabled"]))
        cfg.stacking.cluster_quality_weighting.cap_enabled =
            cqw["cap_enabled"].as<bool>();
      if (yaml_has_value(cqw["cap_ratio"]))
        cfg.stacking.cluster_quality_weighting.cap_ratio =
            cqw["cap_ratio"].as<float>();
    }
    if (yaml_has_value(st["output_stretch"]))
      cfg.stacking.output_stretch = st["output_stretch"].as<bool>();
    if (yaml_has_value(st["cosmetic_correction"]))
      cfg.stacking.cosmetic_correction = st["cosmetic_correction"].as<bool>();
    if (yaml_has_value(st["cosmetic_correction_sigma"]))
      cfg.stacking.cosmetic_correction_sigma = st["cosmetic_correction_sigma"].as<float>();
    if (yaml_has_value(st["per_frame_cosmetic_correction"]))
      cfg.stacking.per_frame_cosmetic_correction = st["per_frame_cosmetic_correction"].as<bool>();
    if (yaml_has_value(st["per_frame_cosmetic_correction_sigma"]))
      cfg.stacking.per_frame_cosmetic_correction_sigma = st["per_frame_cosmetic_correction_sigma"].as<float>();
  }

  if (node["validation"]) {
    auto v = node["validation"];
    if (yaml_has_value(v["min_fwhm_improvement_percent"])) {
      cfg.validation.min_fwhm_improvement_percent =
          v["min_fwhm_improvement_percent"].as<float>();
    }
    if (yaml_has_value(v["max_background_rms_increase_percent"])) {
      cfg.validation.max_background_rms_increase_percent =
          v["max_background_rms_increase_percent"].as<float>();
    }
    if (yaml_has_value(v["min_tile_weight_variance"]))
      cfg.validation.min_tile_weight_variance =
          v["min_tile_weight_variance"].as<float>();
    if (yaml_has_value(v["require_no_tile_pattern"]))
      cfg.validation.require_no_tile_pattern =
          v["require_no_tile_pattern"].as<bool>();
  }

  if (node["runtime_limits"]) {
    auto rl = node["runtime_limits"];
    if (yaml_has_value(rl["tile_analysis_max_factor_vs_stack"])) {
      cfg.runtime_limits.tile_analysis_max_factor_vs_stack =
          rl["tile_analysis_max_factor_vs_stack"].as<float>();
    }
    if (yaml_has_value(rl["hard_abort_hours"]))
      cfg.runtime_limits.hard_abort_hours = rl["hard_abort_hours"].as<float>();
    if (yaml_has_value(rl["allow_emergency_mode"]))
      cfg.runtime_limits.allow_emergency_mode =
          rl["allow_emergency_mode"].as<bool>();
    if (yaml_has_value(rl["parallel_workers"]))
      cfg.runtime_limits.parallel_workers = rl["parallel_workers"].as<int>();
    if (yaml_has_value(rl["memory_budget"]))
      cfg.runtime_limits.memory_budget = rl["memory_budget"].as<int>();
    if (yaml_has_value(rl["acceleration_backend"])) {
      cfg.runtime_limits.acceleration_backend =
          normalize_acceleration_backend(
              rl["acceleration_backend"].as<std::string>());
    }
    if (yaml_has_value(rl["tile_reconstruction_diagnostics"])) {
      cfg.runtime_limits.tile_reconstruction_diagnostics =
          rl["tile_reconstruction_diagnostics"].as<std::string>();
    }
    if (yaml_has_value(rl["tile_boundary_diagnostics_enabled"])) {
      cfg.runtime_limits.tile_boundary_diagnostics_enabled =
          rl["tile_boundary_diagnostics_enabled"].as<bool>();
    } else {
      // Derive from legacy string field: "full" or "minimal" → enabled, "off" → disabled.
      cfg.runtime_limits.tile_boundary_diagnostics_enabled =
          (cfg.runtime_limits.tile_reconstruction_diagnostics != "off");
    }
  }

  normalizeMethod(cfg);

  return cfg;
}

/// @brief Implements save.
/// @details Part of YAML configuration loading, serialization, schema generation, and validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void Config::save(const fs::path &path) const {
  YAML::Node node = to_yaml();
  round_yaml_numeric_scalars_inplace(node);
  std::ofstream out(path);
  if (!out) {
    throw ConfigError("Cannot write config file: " + path.string());
  }
  out << node;
}

/// @brief Converts yaml.
/// @details Part of YAML configuration loading, serialization, schema generation, and validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
YAML::Node Config::to_yaml() const {
  YAML::Node node;

  node["method"] = method;

  node["pipeline"]["mode"] = pipeline.mode;

  node["output"]["registered_dir"] = output.registered_dir;
  node["output"]["write_registered_frames"] = output.write_registered_frames;
  node["output"]["crop_to_nonzero_bbox"] = output.crop_to_nonzero_bbox;

  node["data"]["image_width"] = data.image_width;
  node["data"]["image_height"] = data.image_height;
  node["data"]["color_mode"] = data.color_mode;
  node["data"]["bayer_pattern"] = data.bayer_pattern;
  node["data"]["linear_required"] = data.linear_required;

  node["linearity"]["enabled"] = linearity.enabled;
  node["linearity"]["max_frames"] = linearity.max_frames;
  node["linearity"]["min_overall_linearity"] = linearity.min_overall_linearity;
  node["linearity"]["strictness"] = linearity.strictness;

  node["calibration"]["use_bias"] = calibration.use_bias;
  node["calibration"]["use_dark"] = calibration.use_dark;
  node["calibration"]["use_flat"] = calibration.use_flat;
  node["calibration"]["bias_use_master"] = calibration.bias_use_master;
  node["calibration"]["dark_use_master"] = calibration.dark_use_master;
  node["calibration"]["dark_already_bias_corrected"] =
      calibration.dark_already_bias_corrected;
  node["calibration"]["flat_use_master"] = calibration.flat_use_master;
  node["calibration"]["dark_auto_select"] = calibration.dark_auto_select;
  node["calibration"]["dark_match_exposure_tolerance_percent"] =
      calibration.dark_match_exposure_tolerance_percent;
  node["calibration"]["dark_match_use_temp"] = calibration.dark_match_use_temp;
  node["calibration"]["dark_match_temp_tolerance_c"] =
      calibration.dark_match_temp_tolerance_c;
  node["calibration"]["bias_dir"] = calibration.bias_dir;
  node["calibration"]["darks_dir"] = calibration.darks_dir;
  node["calibration"]["flats_dir"] = calibration.flats_dir;
  node["calibration"]["bias_master"] = calibration.bias_master;
  node["calibration"]["dark_master"] = calibration.dark_master;
  node["calibration"]["flat_master"] = calibration.flat_master;
  node["calibration"]["pattern"] = calibration.pattern;

  node["assumptions"]["frames_min"] = assumptions.frames_min;
  node["assumptions"]["frames_reduced_threshold"] =
      assumptions.frames_reduced_threshold;
  node["assumptions"]["reduced_mode_skip_clustering"] =
      assumptions.reduced_mode_skip_clustering;
  node["assumptions"]["reduced_mode_cluster_range"].push_back(
      assumptions.reduced_mode_cluster_range[0]);
  node["assumptions"]["reduced_mode_cluster_range"].push_back(
      assumptions.reduced_mode_cluster_range[1]);

  node["normalization"]["enabled"] = normalization.enabled;
  node["normalization"]["mode"] = normalization.mode;
  node["normalization"]["per_channel"] = normalization.per_channel;

  node["registration"]["engine"] = registration.engine;
  node["registration"]["transform_model"] = registration.transform_model;
  node["registration"]["enable_star_pair_fallback"] =
      registration.enable_star_pair_fallback;
  node["registration"]["allow_rotation"] = registration.allow_rotation;
  node["registration"]["star_topk"] = registration.star_topk;
  node["registration"]["star_min_inliers"] = registration.star_min_inliers;
  node["registration"]["star_inlier_tol_px"] = registration.star_inlier_tol_px;
  node["registration"]["star_dist_bin_px"] = registration.star_dist_bin_px;
  node["registration"]["reject_outliers"] = registration.reject_outliers;
  node["registration"]["reject_cc_min_abs"] = registration.reject_cc_min_abs;
  node["registration"]["reject_shift_px_min"] = registration.reject_shift_px_min;
  node["registration"]["reject_shift_median_multiplier"] =
      registration.reject_shift_median_multiplier;
  node["registration"]["reject_scale_min"] = registration.reject_scale_min;
  node["registration"]["reject_scale_max"] = registration.reject_scale_max;
  node["registration"]["auto_engine"] = registration.auto_engine;
  node["registration"]["auto_engine_rotation_threshold_deg"] =
      registration.auto_engine_rotation_threshold_deg;
  // Neue Blind-Chain Parameter (§4.1, §8.B)
  node["registration"]["max_blind_chain_depth"] = registration.max_blind_chain_depth;
  node["registration"]["blind_chain_strong_anchor_cc"] = registration.blind_chain_strong_anchor_cc;
  node["registration"]["blind_chain_drift_threshold_px"] = registration.blind_chain_drift_threshold_px;
  // Astrometric rescue (§4.13)
  node["registration"]["use_astrometry"] = registration.use_astrometry;
  // Local background subtraction (§4.4, §8.D)
  node["registration"]["enable_local_background_subtraction"] = registration.enable_local_background_subtraction;
  node["registration"]["star_shift_radius_px"] = registration.star_shift_radius_px;
  node["registration"]["affine_refinement_enabled"] =
      registration.affine_refinement_enabled;
  node["registration"]["smooth_local_refinement_enabled"] =
      registration.smooth_local_refinement_enabled;

  node["dithering"]["enabled"] = dithering.enabled;
  node["dithering"]["min_shift_px"] = dithering.min_shift_px;

  node["tile_denoise"]["soft_threshold"]["enabled"] = tile_denoise.soft_threshold.enabled;
  node["tile_denoise"]["soft_threshold"]["blur_kernel"] = tile_denoise.soft_threshold.blur_kernel;
  node["tile_denoise"]["soft_threshold"]["alpha"] = tile_denoise.soft_threshold.alpha;
  node["tile_denoise"]["soft_threshold"]["skip_star_tiles"] = tile_denoise.soft_threshold.skip_star_tiles;
  node["tile_denoise"]["wiener"]["enabled"] = tile_denoise.wiener.enabled;
  node["tile_denoise"]["wiener"]["snr_threshold"] = tile_denoise.wiener.snr_threshold;
  node["tile_denoise"]["wiener"]["q_min"] = tile_denoise.wiener.q_min;
  node["tile_denoise"]["wiener"]["q_max"] = tile_denoise.wiener.q_max;
  node["tile_denoise"]["wiener"]["q_step"] = tile_denoise.wiener.q_step;
  node["tile_denoise"]["wiener"]["min_snr"] = tile_denoise.wiener.min_snr;
  node["tile_denoise"]["wiener"]["max_iterations"] = tile_denoise.wiener.max_iterations;

  node["chroma_denoise"]["enabled"] = chroma_denoise.enabled;
  node["chroma_denoise"]["color_space"] = chroma_denoise.color_space;
  node["chroma_denoise"]["apply_stage"] = chroma_denoise.apply_stage;
  node["chroma_denoise"]["protect_luma"] = chroma_denoise.protect_luma;
  node["chroma_denoise"]["luma_guard_strength"] = chroma_denoise.luma_guard_strength;
  node["chroma_denoise"]["star_protection"]["enabled"] =
      chroma_denoise.star_protection.enabled;
  node["chroma_denoise"]["star_protection"]["threshold_sigma"] =
      chroma_denoise.star_protection.threshold_sigma;
  node["chroma_denoise"]["star_protection"]["dilate_px"] =
      chroma_denoise.star_protection.dilate_px;
  node["chroma_denoise"]["structure_protection"]["enabled"] =
      chroma_denoise.structure_protection.enabled;
  node["chroma_denoise"]["structure_protection"]["gradient_percentile"] =
      chroma_denoise.structure_protection.gradient_percentile;
  node["chroma_denoise"]["chroma_wavelet"]["enabled"] =
      chroma_denoise.chroma_wavelet.enabled;
  node["chroma_denoise"]["chroma_wavelet"]["levels"] =
      chroma_denoise.chroma_wavelet.levels;
  node["chroma_denoise"]["chroma_wavelet"]["threshold_scale"] =
      chroma_denoise.chroma_wavelet.threshold_scale;
  node["chroma_denoise"]["chroma_wavelet"]["soft_k"] =
      chroma_denoise.chroma_wavelet.soft_k;
  node["chroma_denoise"]["chroma_bilateral"]["enabled"] =
      chroma_denoise.chroma_bilateral.enabled;
  node["chroma_denoise"]["chroma_bilateral"]["sigma_spatial"] =
      chroma_denoise.chroma_bilateral.sigma_spatial;
  node["chroma_denoise"]["chroma_bilateral"]["sigma_range"] =
      chroma_denoise.chroma_bilateral.sigma_range;
  node["chroma_denoise"]["blend"]["mode"] = chroma_denoise.blend.mode;
  node["chroma_denoise"]["blend"]["amount"] = chroma_denoise.blend.amount;

  node["global_metrics"]["adaptive_weights"] = global_metrics.adaptive_weights;
  node["global_metrics"]["weight_exponent_scale"] = global_metrics.weight_exponent_scale;
  node["global_metrics"]["weights"]["background"] =
      global_metrics.weights.background;
  node["global_metrics"]["weights"]["noise"] = global_metrics.weights.noise;
  node["global_metrics"]["weights"]["gradient"] =
      global_metrics.weights.gradient;
  node["global_metrics"]["weights"]["fwhm"] = global_metrics.weights.fwhm;
  node["global_metrics"]["weights"]["roundness"] =
      global_metrics.weights.roundness;
  node["global_metrics"]["weights"]["star_count"] =
      global_metrics.weights.star_count;
  node["global_metrics"]["clamp"].push_back(global_metrics.clamp[0]);
  node["global_metrics"]["clamp"].push_back(global_metrics.clamp[1]);

  node["tile"]["size_factor"] = tile.size_factor;
  node["tile"]["min_size"] = tile.min_size;
  node["tile"]["max_divisor"] = tile.max_divisor;
  node["tile"]["overlap_fraction"] = tile.overlap_fraction;
  node["tile"]["star_min_count"] = tile.star_min_count;
  node["tile"]["star_soft_count"] = tile.star_soft_count;

  node["local_metrics"]["clamp"].push_back(local_metrics.clamp[0]);
  node["local_metrics"]["clamp"].push_back(local_metrics.clamp[1]);
  node["local_metrics"]["neighborhood_normalization"]["enabled"] =
      local_metrics.neighborhood_normalization.enabled;
  node["local_metrics"]["neighborhood_normalization"]["radius"] =
      local_metrics.neighborhood_normalization.radius;
  node["local_metrics"]["neighborhood_normalization"]["blend"] =
      local_metrics.neighborhood_normalization.blend;
  node["local_metrics"]["spatial_regularization"]["enabled"] =
      local_metrics.spatial_regularization.enabled;
  node["local_metrics"]["spatial_regularization"]["lambda"] =
      local_metrics.spatial_regularization.lambda;
  node["local_metrics"]["spatial_regularization"]["passes"] =
      local_metrics.spatial_regularization.passes;
  node["local_metrics"]["spatial_regularization"]["tau_local"] =
      local_metrics.spatial_regularization.tau_local;
  node["local_metrics"]["star_mode"]["weights"]["fwhm"] =
      local_metrics.star_mode.weights.fwhm;
  node["local_metrics"]["star_mode"]["weights"]["roundness"] =
      local_metrics.star_mode.weights.roundness;
  node["local_metrics"]["star_mode"]["weights"]["contrast"] =
      local_metrics.star_mode.weights.contrast;
  node["local_metrics"]["structure_mode"]["background_weight"] =
      local_metrics.structure_mode.background_weight;
  node["local_metrics"]["structure_mode"]["metric_weight"] =
      local_metrics.structure_mode.metric_weight;
  node["local_metrics"]["k_local"] = local_metrics.k_local;

  node["aqmh"]["enabled"] = aqmh.enabled;
  node["aqmh"]["pyramid"]["scales"] = aqmh.pyramid.scales;
  node["aqmh"]["pyramid"]["base_window_px"] = aqmh.pyramid.base_window_px;
  node["aqmh"]["pyramid"]["w_sharp"] = aqmh.pyramid.w_sharp;
  node["aqmh"]["pyramid"]["w_snr"] = aqmh.pyramid.w_snr;
  node["aqmh"]["pyramid"]["score_scale"] = aqmh.pyramid.score_scale;
  node["aqmh"]["pyramid"]["k_artifact"] = aqmh.pyramid.k_artifact;
  node["aqmh"]["pyramid"]["frac_artifact_max"] =
      aqmh.pyramid.frac_artifact_max;
  node["aqmh"]["storage"]["resolution_divisor"] =
      aqmh.storage.resolution_divisor;
  node["aqmh"]["storage"]["dtype"] = aqmh.storage.dtype;
  node["aqmh"]["storage"]["max_resident_maps"] =
      aqmh.storage.max_resident_maps;
  node["aqmh"]["global_quality"]["g_floor"] = aqmh.global_quality.g_floor;
  node["aqmh"]["global_quality"]["g_w_sharp"] = aqmh.global_quality.g_w_sharp;
  node["aqmh"]["global_quality"]["g_w_snr"] = aqmh.global_quality.g_w_snr;
  node["aqmh"]["global_quality"]["g_w_background_penalty"] =
      aqmh.global_quality.g_w_background_penalty;
  node["aqmh"]["global_quality"]["g_k_scale"] = aqmh.global_quality.g_k_scale;
  node["aqmh"]["cherry_pick"]["enabled"] = aqmh.cherry_pick.enabled;
  node["aqmh"]["cherry_pick"]["mode"] = aqmh.cherry_pick.mode;
  node["aqmh"]["cherry_pick"]["k_frac"] = aqmh.cherry_pick.k_frac;
  node["aqmh"]["cherry_pick"]["k_min_required"] = aqmh.cherry_pick.k_min_required;
  node["aqmh"]["cherry_pick"]["margin_min"] = aqmh.cherry_pick.margin_min;
  node["aqmh"]["cherry_pick"]["reject_below_best_fraction"] =
      aqmh.cherry_pick.reject_below_best_fraction;
  node["aqmh"]["cherry_pick"]["min_keep_fraction"] =
      aqmh.cherry_pick.min_keep_fraction;
  if (aqmh.cherry_pick.tiered_k_frac.empty()) {
    node["aqmh"]["cherry_pick"]["tiered_k_frac"] = YAML::Node(YAML::NodeType::Sequence);
  } else {
    for (const auto &tier : aqmh.cherry_pick.tiered_k_frac) {
      YAML::Node item;
      item["min_n_rankable"] = tier.min_n_rankable;
      item["k_frac"] = tier.k_frac;
      node["aqmh"]["cherry_pick"]["tiered_k_frac"].push_back(item);
    }
  }
  node["aqmh"]["diagnostics"]["tau_artifact"] =
      aqmh.diagnostics.tau_artifact;
  node["aqmh"]["diagnostics"]["q_region"] = aqmh.diagnostics.q_region;
  node["aqmh"]["diagnostics"]["r_morph_canvas_px"] =
      aqmh.diagnostics.r_morph_canvas_px;
  // NEW FIELDS:
  node["aqmh"]["diagnostics"]["enabled"] = aqmh.diagnostics.enabled;
  node["aqmh"]["diagnostics"]["level"] = aqmh.diagnostics.level;
  node["aqmh"]["diagnostics"]["per_frame_blocks"] = aqmh.diagnostics.per_frame_blocks;
  node["aqmh"]["diagnostics"]["heatmaps"] = aqmh.diagnostics.heatmaps;
  node["aqmh"]["diagnostics"]["regions"] = aqmh.diagnostics.regions;
  node["aqmh"]["diagnostics"]["format"] = aqmh.diagnostics.format;
  node["aqmh"]["diagnostics"]["binary_block_size_px"] = aqmh.diagnostics.binary_block_size_px;
  node["aqmh"]["reconstruction"]["clip_sigma"] = aqmh.reconstruction.clip_sigma;
  node["aqmh"]["reconstruction"]["clip_sigma_low"] =
      aqmh.reconstruction.clip_sigma_low;
  node["aqmh"]["reconstruction"]["clip_sigma_high"] =
      aqmh.reconstruction.clip_sigma_high;
  node["aqmh"]["reconstruction"]["clip_iterations"] = aqmh.reconstruction.clip_iterations;
  node["aqmh"]["reconstruction"]["min_fraction"] = aqmh.reconstruction.min_fraction;
  node["aqmh"]["reconstruction"]["min_n_eff"] = aqmh.reconstruction.min_n_eff;
  // NEW FIELDS:
  node["aqmh"]["reconstruction"]["chunk_rows"] = aqmh.reconstruction.chunk_rows;
  node["aqmh"]["reconstruction"]["memory_budget_mb"] = aqmh.reconstruction.memory_budget_mb;
  node["aqmh"]["reconstruction"]["delete_prewarped_cache_after_run"] =
      aqmh.reconstruction.delete_prewarped_cache_after_run;
  node["aqmh"]["reconstruction"]["prewarp_interpolation"] =
      aqmh.reconstruction.prewarp_interpolation;
  node["aqmh"]["reconstruction"]["debayer_first"] =
      aqmh.reconstruction.debayer_first;
  node["aqmh"]["reconstruction"]["pre_debayer_method"] =
      aqmh.reconstruction.pre_debayer_method;
  node["aqmh"]["reconstruction"]["rgb_q_map_mode"] =
      aqmh.reconstruction.rgb_q_map_mode;
  node["aqmh"]["reconstruction"]["rgb_memory_strategy"] =
      aqmh.reconstruction.rgb_memory_strategy;
  node["aqmh"]["reconstruction"]["registration_weight_guard"] =
      aqmh.reconstruction.registration_weight_guard;
  node["aqmh"]["reconstruction"]["registration_weight_floor"] =
      aqmh.reconstruction.registration_weight_floor;
  node["aqmh"]["reconstruction"]["registration_cc_floor"] =
      aqmh.reconstruction.registration_cc_floor;
  node["aqmh"]["reconstruction"]["registration_cc_full"] =
      aqmh.reconstruction.registration_cc_full;
  node["aqmh"]["reconstruction"]["registration_sequential_factor"] =
      aqmh.reconstruction.registration_sequential_factor;
  node["aqmh"]["reconstruction"]["registration_predicted_factor"] =
      aqmh.reconstruction.registration_predicted_factor;
  node["aqmh"]["reconstruction"]["registration_chain_depth_penalty"] =
      aqmh.reconstruction.registration_chain_depth_penalty;
  node["aqmh"]["reconstruction"]["registration_chain_depth_max_penalty"] =
      aqmh.reconstruction.registration_chain_depth_max_penalty;
  node["aqmh"]["reconstruction"]["structure_mask_low_q"] =
      aqmh.reconstruction.structure_mask_low_q;
  node["aqmh"]["reconstruction"]["structure_mask_high_q"] =
      aqmh.reconstruction.structure_mask_high_q;
  node["aqmh"]["reconstruction"]["structure_mask_blur_sigma_px"] =
      aqmh.reconstruction.structure_mask_blur_sigma_px;
  node["aqmh"]["validation"]["max_seam_score_regression"] = aqmh.validation.max_seam_score_regression;
  node["aqmh"]["validation"]["max_fwhm_regression"] = aqmh.validation.max_fwhm_regression;
  node["aqmh"]["validation"]["max_background_rms_regression"] = aqmh.validation.max_background_rms_regression;
  node["aqmh"]["validation"]["max_tail11_abs_regression"] =
      aqmh.validation.max_tail11_abs_regression;
  node["aqmh"]["validation"]["max_elongation_regression"] =
      aqmh.validation.max_elongation_regression;

  node["synthetic"]["weighting"] = synthetic.weighting;
  node["synthetic"]["frames_min"] = synthetic.frames_min;
  node["synthetic"]["frames_max"] = synthetic.frames_max;
  node["synthetic"]["clustering"]["mode"] = synthetic.clustering.mode;
  node["synthetic"]["clustering"]["cluster_count_range"].push_back(
      synthetic.clustering.cluster_count_range[0]);
  node["synthetic"]["clustering"]["cluster_count_range"].push_back(
      synthetic.clustering.cluster_count_range[1]);

  node["astrometry"]["enabled"] = astrometry.enabled;
  node["astrometry"]["astap_bin"] = astrometry.astap_bin;
  node["astrometry"]["astap_data_dir"] = astrometry.astap_data_dir;
  node["astrometry"]["search_radius"] = astrometry.search_radius;

  node["bge"]["enabled"] = bge.enabled;
  node["bge"]["method"] = bge.method;
  node["bge"]["autobge"]["num_sample_points"] =
      bge.autobge.num_sample_points;
  node["bge"]["autobge"]["poly_degree"] = bge.autobge.poly_degree;
  node["bge"]["autobge"]["rbf_smooth"] = bge.autobge.rbf_smooth;
  node["bge"]["autobge"]["downsample_scale"] = bge.autobge.downsample_scale;
  node["bge"]["autobge"]["patch_size"] = bge.autobge.patch_size;
  node["bge"]["autobge"]["patch_estimator"] =
      bge.autobge.patch_estimator;
  node["bge"]["autobge"]["stretch_mode"] = bge.autobge.stretch_mode;
  node["bge"]["autobge"]["stretch_target_median"] =
      bge.autobge.stretch_target_median;
  node["bge"]["autobge"]["border_margin"] = bge.autobge.border_margin;
  node["bge"]["autobge"]["bright_exclusion_fraction"] =
      bge.autobge.bright_exclusion_fraction;
  node["bge"]["autobge"]["gradient_descent_max_iters"] =
      bge.autobge.gradient_descent_max_iters;
  node["bge"]["autobge"]["random_seed"] = bge.autobge.random_seed;
  node["bge"]["autobge"]["normalize_between_stages"] =
      bge.autobge.normalize_between_stages;
  node["bge"]["autobge"]["apply_guards"] = bge.autobge.apply_guards;
  node["bge"]["autobge"]["mono_mode"] = bge.autobge.mono_mode;
  if (bge.autobge.exclusion_polygons.empty()) {
    node["bge"]["autobge"]["exclusion_polygons"] = YAML::Node(YAML::NodeType::Sequence);
  } else {
    for (const auto &polygon : bge.autobge.exclusion_polygons) {
      YAML::Node polygon_node(YAML::NodeType::Sequence);
      for (const auto &point : polygon) {
        YAML::Node point_node(YAML::NodeType::Sequence);
        point_node.push_back(point[0]);
        point_node.push_back(point[1]);
        polygon_node.push_back(point_node);
      }
      node["bge"]["autobge"]["exclusion_polygons"].push_back(polygon_node);
    }
  }
  if (bge.autobge.user_sample_points.empty()) {
    node["bge"]["autobge"]["user_sample_points"] = YAML::Node(YAML::NodeType::Sequence);
  } else {
    for (const auto &point : bge.autobge.user_sample_points) {
      YAML::Node point_node(YAML::NodeType::Sequence);
      point_node.push_back(point[0]);
      point_node.push_back(point[1]);
      node["bge"]["autobge"]["user_sample_points"].push_back(point_node);
    }
  }
  node["bge"]["sample_quantile"] = bge.sample_quantile;
  node["bge"]["sample_estimator"] = bge.sample_estimator;
  node["bge"]["min_sample_bg_value"] = bge.min_sample_bg_value;
  node["bge"]["structure_thresh_percentile"] = bge.structure_thresh_percentile;
  node["bge"]["min_tiles_per_cell"] = bge.min_tiles_per_cell;
  node["bge"]["min_valid_sample_fraction_for_apply"] =
      bge.min_valid_sample_fraction_for_apply;
  node["bge"]["min_valid_samples_for_apply"] =
      bge.min_valid_samples_for_apply;
  node["bge"]["mask"]["star_dilate_px"] = bge.mask.star_dilate_px;
  node["bge"]["mask"]["sat_dilate_px"] = bge.mask.sat_dilate_px;
  node["bge"]["grid"]["N_g"] = bge.grid.N_g;
  node["bge"]["grid"]["G_min_px"] = bge.grid.G_min_px;
  node["bge"]["grid"]["G_max_fraction"] = bge.grid.G_max_fraction;
  node["bge"]["grid"]["insufficient_cell_strategy"] = bge.grid.insufficient_cell_strategy;
  node["bge"]["fit"]["method"] = bge.fit.method;
  node["bge"]["fit"]["robust_loss"] = bge.fit.robust_loss;
  node["bge"]["fit"]["huber_delta"] = bge.fit.huber_delta;
  node["bge"]["fit"]["irls_max_iterations"] = bge.fit.irls_max_iterations;
  node["bge"]["fit"]["irls_tolerance"] = bge.fit.irls_tolerance;
  node["bge"]["fit"]["polynomial_order"] = bge.fit.polynomial_order;
  node["bge"]["fit"]["rbf_phi"] = bge.fit.rbf_phi;
  node["bge"]["fit"]["rbf_mu_factor"] = bge.fit.rbf_mu_factor;
  node["bge"]["fit"]["rbf_lambda"] = bge.fit.rbf_lambda;
  node["bge"]["fit"]["rbf_epsilon"] = bge.fit.rbf_epsilon;
  node["bge"]["autotune"]["enabled"] = bge.autotune.enabled;
  node["bge"]["autotune"]["max_evals"] = bge.autotune.max_evals;
  node["bge"]["autotune"]["holdout_fraction"] = bge.autotune.holdout_fraction;
  node["bge"]["autotune"]["alpha_flatness"] = bge.autotune.alpha_flatness;
  node["bge"]["autotune"]["beta_roughness"] = bge.autotune.beta_roughness;
  node["bge"]["autotune"]["strategy"] = bge.autotune.strategy;
  node["bge"]["tile_weight_lambda_structure"] =
      bge.tile_weight_lambda_structure;

  node["pcc"]["enabled"] = pcc.enabled;
  node["pcc"]["source"] = pcc.source;
  node["pcc"]["mag_limit"] = pcc.mag_limit;
  node["pcc"]["mag_bright_limit"] = pcc.mag_bright_limit;
  node["pcc"]["aperture_radius_px"] = pcc.aperture_radius_px;
  node["pcc"]["annulus_inner_px"] = pcc.annulus_inner_px;
  node["pcc"]["annulus_outer_px"] = pcc.annulus_outer_px;
  node["pcc"]["min_stars"] = pcc.min_stars;
  node["pcc"]["sigma_clip"] = pcc.sigma_clip;
  node["pcc"]["background_model"] = pcc.background_model;
  node["pcc"]["max_condition_number"] = pcc.max_condition_number;
  node["pcc"]["max_residual_rms"] = pcc.max_residual_rms;
  node["pcc"]["radii_mode"] = pcc.radii_mode;
  node["pcc"]["aperture_fwhm_mult"] = pcc.aperture_fwhm_mult;
  node["pcc"]["annulus_inner_fwhm_mult"] = pcc.annulus_inner_fwhm_mult;
  node["pcc"]["annulus_outer_fwhm_mult"] = pcc.annulus_outer_fwhm_mult;
  node["pcc"]["min_aperture_px"] = pcc.min_aperture_px;
  node["pcc"]["siril_catalog_dir"] = pcc.siril_catalog_dir;
  node["pcc"]["apply_attenuation"] = pcc.apply_attenuation;
  node["pcc"]["chroma_strength"] = pcc.chroma_strength;
  node["pcc"]["k_max"] = pcc.k_max;
  node["pcc"]["background_neutralization_mode"] =
      pcc.background_neutralization_mode;

  node["hypermetric_stretch"]["enabled"] = hypermetric_stretch.enabled;
  node["hypermetric_stretch"]["require_successful_pcc"] =
      hypermetric_stretch.require_successful_pcc;
  node["hypermetric_stretch"]["mode"] = hypermetric_stretch.mode;
  node["hypermetric_stretch"]["sensor_profile"] =
      hypermetric_stretch.sensor_profile;
  node["hypermetric_stretch"]["fallback_profile"] =
      hypermetric_stretch.fallback_profile;
  node["hypermetric_stretch"]["adaptive_anchor"] =
      hypermetric_stretch.adaptive_anchor;
  node["hypermetric_stretch"]["target_bg"] = hypermetric_stretch.target_bg;
  node["hypermetric_stretch"]["protect_b"] = hypermetric_stretch.protect_b;
  node["hypermetric_stretch"]["convergence_power"] =
      hypermetric_stretch.convergence_power;
  node["hypermetric_stretch"]["log_d_mode"] = hypermetric_stretch.log_d_mode;
  node["hypermetric_stretch"]["fixed_log_d"] = hypermetric_stretch.fixed_log_d;
  node["hypermetric_stretch"]["color_strategy"] =
      hypermetric_stretch.color_strategy;
  node["hypermetric_stretch"]["fixed_color_strategy"] =
      hypermetric_stretch.fixed_color_strategy;
  node["hypermetric_stretch"]["color_grip"] = hypermetric_stretch.color_grip;
  node["hypermetric_stretch"]["shadow_convergence"] =
      hypermetric_stretch.shadow_convergence;
  node["hypermetric_stretch"]["linear_expansion"] =
      hypermetric_stretch.linear_expansion;
  node["hypermetric_stretch"]["write_channels"] =
      hypermetric_stretch.write_channels;
  node["hypermetric_stretch"]["output_rgb"] = hypermetric_stretch.output_rgb;

  node["stacking"]["method"] = stacking.method;
  node["stacking"]["common_overlap_required_fraction"] =
      stacking.common_overlap_required_fraction;
  node["stacking"]["tile_common_valid_min_fraction"] =
      stacking.tile_common_valid_min_fraction;
  node["stacking"]["sigma_clip"]["sigma_low"] = stacking.sigma_clip.sigma_low;
  node["stacking"]["sigma_clip"]["sigma_high"] = stacking.sigma_clip.sigma_high;
  node["stacking"]["sigma_clip"]["max_iters"] = stacking.sigma_clip.max_iters;
  node["stacking"]["sigma_clip"]["min_fraction"] =
      stacking.sigma_clip.min_fraction;
  node["stacking"]["cluster_quality_weighting"]["enabled"] =
      stacking.cluster_quality_weighting.enabled;
  node["stacking"]["cluster_quality_weighting"]["kappa_cluster"] =
      stacking.cluster_quality_weighting.kappa_cluster;
  node["stacking"]["cluster_quality_weighting"]["cap_enabled"] =
      stacking.cluster_quality_weighting.cap_enabled;
  node["stacking"]["cluster_quality_weighting"]["cap_ratio"] =
      stacking.cluster_quality_weighting.cap_ratio;
  node["stacking"]["output_stretch"] = stacking.output_stretch;
  node["stacking"]["cosmetic_correction"] =
      stacking.cosmetic_correction;
  node["stacking"]["cosmetic_correction_sigma"] =
      stacking.cosmetic_correction_sigma;
  node["stacking"]["per_frame_cosmetic_correction"] =
      stacking.per_frame_cosmetic_correction;
  node["stacking"]["per_frame_cosmetic_correction_sigma"] =
      stacking.per_frame_cosmetic_correction_sigma;

  node["validation"]["min_fwhm_improvement_percent"] =
      validation.min_fwhm_improvement_percent;
  node["validation"]["max_background_rms_increase_percent"] =
      validation.max_background_rms_increase_percent;
  node["validation"]["min_tile_weight_variance"] =
      validation.min_tile_weight_variance;
  node["validation"]["require_no_tile_pattern"] =
      validation.require_no_tile_pattern;

  node["runtime_limits"]["tile_analysis_max_factor_vs_stack"] =
      runtime_limits.tile_analysis_max_factor_vs_stack;
  node["runtime_limits"]["hard_abort_hours"] = runtime_limits.hard_abort_hours;
  node["runtime_limits"]["allow_emergency_mode"] =
      runtime_limits.allow_emergency_mode;
  node["runtime_limits"]["parallel_workers"] =
      runtime_limits.parallel_workers;
  node["runtime_limits"]["memory_budget"] = runtime_limits.memory_budget;
  node["runtime_limits"]["acceleration_backend"] =
      runtime_limits.acceleration_backend;
  node["runtime_limits"]["tile_reconstruction_diagnostics"] =
      runtime_limits.tile_reconstruction_diagnostics;

  return node;
}

/// @brief Implements validate.
/// @details Part of YAML configuration loading, serialization, schema generation, and validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void Config::validate() const {
  if (method != "aqmh" && method != "classic_tile_compile") {
    throw ValidationError("method must be 'aqmh' or 'classic_tile_compile'");
  }

  if (pipeline.mode != "production" && pipeline.mode != "test") {
    throw ValidationError("pipeline.mode must be 'production' or 'test'");
  }

  if (data.image_width < 0 || data.image_height < 0) {
    throw ValidationError(
        "data.image_width and data.image_height must be >= 0");
  }
  if (data.color_mode != "OSC" && data.color_mode != "MONO" &&
      data.color_mode != "RGB") {
    throw ValidationError("data.color_mode must be OSC, MONO, or RGB");
  }
  if (data.linear_required && data.color_mode == "RGB") {
    throw ValidationError(
        "data.linear_required should be false for already debayered RGB data");
  }

  if (linearity.max_frames < 1) {
    throw ValidationError("linearity.max_frames must be >= 1");
  }
  if (!is_between_0_1(linearity.min_overall_linearity)) {
    throw ValidationError("linearity.min_overall_linearity must be in [0,1]");
  }
  if (linearity.strictness != "strict" && linearity.strictness != "moderate" &&
      linearity.strictness != "permissive") {
    throw ValidationError(
        "linearity.strictness must be 'strict', 'moderate', or 'permissive'");
  }

  if (calibration.dark_match_exposure_tolerance_percent < 0.0f) {
    throw ValidationError(
        "calibration.dark_match_exposure_tolerance_percent must be >= 0");
  }
  if (calibration.dark_match_temp_tolerance_c < 0.0f) {
    throw ValidationError(
        "calibration.dark_match_temp_tolerance_c must be >= 0");
  }
  if (calibration.use_bias && calibration.bias_dir.empty() &&
      calibration.bias_master.empty()) {
    throw ValidationError(
        "calibration.use_bias requires calibration.bias_dir or calibration.bias_master");
  }
  if (calibration.use_dark && calibration.darks_dir.empty() &&
      calibration.dark_master.empty()) {
    throw ValidationError(
        "calibration.use_dark requires calibration.darks_dir or calibration.dark_master");
  }
  if (calibration.use_flat && calibration.flats_dir.empty() &&
      calibration.flat_master.empty()) {
    throw ValidationError(
        "calibration.use_flat requires calibration.flats_dir or calibration.flat_master");
  }

  if (assumptions.frames_min < 1)
    throw ValidationError("assumptions.frames_min must be >= 1");
  if (assumptions.frames_reduced_threshold < assumptions.frames_min) {
    throw ValidationError("assumptions.frames_reduced_threshold must be >= "
                          "assumptions.frames_min");
  }
  if (assumptions.reduced_mode_cluster_range[0] < 1 ||
      assumptions.reduced_mode_cluster_range[1] <
          assumptions.reduced_mode_cluster_range[0]) {
    throw ValidationError("assumptions.reduced_mode_cluster_range must be "
                          "[min,max] with min>=1 and max>=min");
  }

  if (!normalization.enabled) {
    throw ValidationError("normalization.enabled must be true for Methodik v3");
  }
  if (normalization.mode != "background" && normalization.mode != "median") {
    throw ValidationError(
        "normalization.mode must be 'background' or 'median'");
  }

  if (registration.engine != "hybrid_phase_ecc" &&
      registration.engine != "robust_phase_ecc" &&
      registration.engine != "star_similarity" &&
      registration.engine != "triangle_star_matching") {
    throw ValidationError(
        "registration.engine must be 'triangle_star_matching', "
        "'star_similarity', 'hybrid_phase_ecc', or 'robust_phase_ecc'");
  }
  if (registration.auto_engine_rotation_threshold_deg <= 0.0f) {
    throw ValidationError(
        "registration.auto_engine_rotation_threshold_deg must be > 0");
  }
  if (registration.transform_model != "similarity" &&
      registration.transform_model != "affine") {
    throw ValidationError(
        "registration.transform_model must be 'similarity' or 'affine'");
  }
  if (registration.star_topk < 3) {
    throw ValidationError("registration.star_topk must be >= 3");
  }
  if (registration.star_min_inliers < 2) {
    throw ValidationError("registration.star_min_inliers must be >= 2");
  }
  if (registration.star_inlier_tol_px <= 0.0f ||
      registration.star_dist_bin_px <= 0.0f) {
    throw ValidationError(
        "registration.star_inlier_tol_px and star_dist_bin_px must be > 0");
  }
  if (registration.reject_cc_min_abs < 0.0f ||
      registration.reject_cc_min_abs > 1.0f) {
    throw ValidationError("registration.reject_cc_min_abs must be in [0,1]");
  }
  if (registration.reject_shift_px_min < 0.0f ||
      registration.reject_shift_median_multiplier <= 0.0f) {
    throw ValidationError(
        "registration.reject_shift_px_min must be >= 0 and "
        "registration.reject_shift_median_multiplier must be > 0");
  }
  if (registration.reject_scale_min <= 0.0f ||
      registration.reject_scale_max < registration.reject_scale_min) {
    throw ValidationError(
        "registration.reject_scale_min must be > 0 and "
        "registration.reject_scale_max must be >= reject_scale_min");
  }
  // Neue Parameter Validierung (§4.1, §4.4, §4.13)
  if (registration.max_blind_chain_depth < 0 || registration.max_blind_chain_depth > 100) {
    throw ValidationError("registration.max_blind_chain_depth must be in [0, 100]");
  }
  if (registration.blind_chain_strong_anchor_cc < 0.01f ||
      registration.blind_chain_strong_anchor_cc > 0.5f) {
    throw ValidationError("registration.blind_chain_strong_anchor_cc must be in [0.01, 0.5]");
  }
  if (registration.blind_chain_drift_threshold_px < 0.5f ||
      registration.blind_chain_drift_threshold_px > 10.0f) {
    throw ValidationError("registration.blind_chain_drift_threshold_px must be in [0.5, 10.0]");
  }
  if (registration.star_shift_radius_px < 10.0f ||
      registration.star_shift_radius_px > 2000.0f) {
    throw ValidationError("registration.star_shift_radius_px must be in [10, 2000]");
  }

  if (dithering.min_shift_px < 0.0f) {
    throw ValidationError("dithering.min_shift_px must be >= 0");
  }

  if (tile_denoise.soft_threshold.blur_kernel < 3) {
    throw ValidationError("tile_denoise.soft_threshold.blur_kernel must be >= 3");
  }
  if (tile_denoise.soft_threshold.alpha <= 0.0f) {
    throw ValidationError("tile_denoise.soft_threshold.alpha must be > 0");
  }
  if (tile_denoise.wiener.q_max < 0.0f || tile_denoise.wiener.q_max > 1.0f) {
    throw ValidationError("tile_denoise.wiener.q_max must be in [0,1]");
  }
  if (tile_denoise.wiener.q_min < -1.0f ||
      tile_denoise.wiener.q_min > tile_denoise.wiener.q_max) {
    throw ValidationError("tile_denoise.wiener.q_min must be <= q_max and >= -1");
  }
  if (tile_denoise.wiener.q_step <= 0.0f) {
    throw ValidationError("tile_denoise.wiener.q_step must be > 0");
  }
  if (tile_denoise.wiener.max_iterations < 1) {
    throw ValidationError("tile_denoise.wiener.max_iterations must be >= 1");
  }

  if (chroma_denoise.color_space != "ycbcr_linear" &&
      chroma_denoise.color_space != "opponent_linear") {
    throw ValidationError(
        "chroma_denoise.color_space must be 'ycbcr_linear' or 'opponent_linear'");
  }
  if (chroma_denoise.apply_stage != "pre_stack_tiles" &&
      chroma_denoise.apply_stage != "post_stack_linear" &&
      chroma_denoise.apply_stage != "post_pcc") {
    throw ValidationError(
        "chroma_denoise.apply_stage must be 'pre_stack_tiles', 'post_stack_linear' or 'post_pcc'");
  }
  if (!is_between_0_1(chroma_denoise.luma_guard_strength)) {
    throw ValidationError("chroma_denoise.luma_guard_strength must be in [0,1]");
  }
  if (chroma_denoise.star_protection.threshold_sigma <= 0.0f) {
    throw ValidationError(
        "chroma_denoise.star_protection.threshold_sigma must be > 0");
  }
  if (chroma_denoise.star_protection.dilate_px < 0) {
    throw ValidationError("chroma_denoise.star_protection.dilate_px must be >= 0");
  }
  if (chroma_denoise.structure_protection.gradient_percentile < 0.0f ||
      chroma_denoise.structure_protection.gradient_percentile > 100.0f) {
    throw ValidationError(
        "chroma_denoise.structure_protection.gradient_percentile must be in [0,100]");
  }
  if (chroma_denoise.chroma_wavelet.levels < 1) {
    throw ValidationError("chroma_denoise.chroma_wavelet.levels must be >= 1");
  }
  if (chroma_denoise.chroma_wavelet.threshold_scale <= 0.0f) {
    throw ValidationError(
        "chroma_denoise.chroma_wavelet.threshold_scale must be > 0");
  }
  if (chroma_denoise.chroma_wavelet.soft_k <= 0.0f) {
    throw ValidationError("chroma_denoise.chroma_wavelet.soft_k must be > 0");
  }
  if (chroma_denoise.chroma_bilateral.sigma_spatial <= 0.0f ||
      chroma_denoise.chroma_bilateral.sigma_range <= 0.0f) {
    throw ValidationError(
        "chroma_denoise.chroma_bilateral sigma values must be > 0");
  }
  if (chroma_denoise.blend.mode != "chroma_only") {
    throw ValidationError("chroma_denoise.blend.mode must be 'chroma_only'");
  }
  if (!is_between_0_1(chroma_denoise.blend.amount)) {
    throw ValidationError("chroma_denoise.blend.amount must be in [0,1]");
  }

  auto check_weight_sum = [](std::initializer_list<float> weights,
                             const char *name) {
    float sum = 0.0f;
    for (const float w : weights)
      sum += w;
    if (std::fabs(sum - 1.0f) > 1.0e-3f) {
      throw ValidationError(std::string(name) + " must sum to 1.0");
    }
  };

  if (!is_between_0_1(global_metrics.weights.background) ||
      !is_between_0_1(global_metrics.weights.noise) ||
      !is_between_0_1(global_metrics.weights.gradient) ||
      !is_between_0_1(global_metrics.weights.fwhm) ||
      !is_between_0_1(global_metrics.weights.roundness) ||
      !is_between_0_1(global_metrics.weights.star_count)) {
    throw ValidationError("global_metrics.weights.* must be between 0 and 1");
  }
  check_weight_sum({global_metrics.weights.background,
                    global_metrics.weights.noise,
                    global_metrics.weights.gradient,
                    global_metrics.weights.fwhm,
                    global_metrics.weights.roundness,
                    global_metrics.weights.star_count},
                   "global_metrics.weights");
  if (global_metrics.clamp[0] >= global_metrics.clamp[1]) {
    throw ValidationError(
        "global_metrics.clamp must be [min,max] with min < max");
  }
  if (global_metrics.weight_exponent_scale <= 0.0f) {
    throw ValidationError(
        "global_metrics.weight_exponent_scale must be > 0");
  }

  if (tile.size_factor <= 0)
    throw ValidationError("tile.size_factor must be positive");
  if (tile.min_size <= 0)
    throw ValidationError("tile.min_size must be positive");
  if (tile.max_divisor <= 0)
    throw ValidationError("tile.max_divisor must be positive");
  if (tile.overlap_fraction < 0 || tile.overlap_fraction > 0.5f) {
    throw ValidationError("tile.overlap_fraction must be between 0 and 0.5");
  }
  if (tile.star_min_count < 0)
    throw ValidationError("tile.star_min_count must be >= 0");
  if (tile.star_soft_count < 0)
    throw ValidationError("tile.star_soft_count must be >= 0");

  if (local_metrics.clamp[0] >= local_metrics.clamp[1]) {
    throw ValidationError(
        "local_metrics.clamp must be [min,max] with min < max");
  }
  if (local_metrics.neighborhood_normalization.radius < 0) {
    throw ValidationError(
        "local_metrics.neighborhood_normalization.radius must be >= 0");
  }
  if (local_metrics.neighborhood_normalization.blend < 0.0f ||
      local_metrics.neighborhood_normalization.blend > 1.0f) {
    throw ValidationError(
        "local_metrics.neighborhood_normalization.blend must be between 0 and 1");
  }
  if (local_metrics.spatial_regularization.lambda < 0.0f ||
      local_metrics.spatial_regularization.lambda > 1.0f) {
    throw ValidationError(
        "local_metrics.spatial_regularization.lambda must be between 0 and 1");
  }
  if (local_metrics.spatial_regularization.passes < 0) {
    throw ValidationError(
        "local_metrics.spatial_regularization.passes must be >= 0");
  }
  if (local_metrics.spatial_regularization.tau_local <= 0.0f) {
    throw ValidationError(
        "local_metrics.spatial_regularization.tau_local must be > 0");
  }
  check_weight_sum({local_metrics.star_mode.weights.fwhm,
                    local_metrics.star_mode.weights.roundness,
                    local_metrics.star_mode.weights.contrast},
                   "local_metrics.star_mode.weights");
  if (std::fabs(local_metrics.structure_mode.background_weight +
                local_metrics.structure_mode.metric_weight - 1.0f) > 1.0e-3f) {
    throw ValidationError(
        "local_metrics.structure_mode weights must sum to 1.0");
  }
  if (local_metrics.k_local <= 0.0f) {
    throw ValidationError("local_metrics.k_local must be > 0");
  }

  if (aqmh.pyramid.scales < 1 || aqmh.pyramid.scales > 8) {
    throw ValidationError("aqmh.pyramid.scales must be in [1,8]");
  }
  if (aqmh.pyramid.base_window_px < 1) {
    throw ValidationError("aqmh.pyramid.base_window_px must be >= 1");
  }
  if (aqmh.pyramid.w_sharp < 0.0f || aqmh.pyramid.w_snr < 0.0f ||
      aqmh.pyramid.w_sharp + aqmh.pyramid.w_snr <= 0.0f) {
    throw ValidationError(
        "aqmh.pyramid.w_sharp and w_snr must be non-negative with positive sum");
  }
  if (aqmh.pyramid.score_scale <= 0.0f) {
    throw ValidationError("aqmh.pyramid.score_scale must be > 0");
  }
  if (aqmh.pyramid.k_artifact <= 0.0f) {
    throw ValidationError("aqmh.pyramid.k_artifact must be > 0");
  }
  if (!is_between_0_1(aqmh.pyramid.frac_artifact_max) ||
      aqmh.pyramid.frac_artifact_max <= 0.0f) {
    throw ValidationError("aqmh.pyramid.frac_artifact_max must be in (0,1]");
  }
  if (aqmh.storage.resolution_divisor != 1 &&
      aqmh.storage.resolution_divisor != 2 &&
      aqmh.storage.resolution_divisor != 4) {
    throw ValidationError("aqmh.storage.resolution_divisor must be 1, 2, or 4");
  }
  if (aqmh.storage.dtype != "float32" && aqmh.storage.dtype != "uint16" &&
      aqmh.storage.dtype != "uint8") {
    throw ValidationError(
        "aqmh.storage.dtype must be 'float32', 'uint16', or 'uint8'");
  }
  if (aqmh.storage.max_resident_maps < 0 ||
      aqmh.storage.max_resident_maps > 16) {
    throw ValidationError("aqmh.storage.max_resident_maps must be in [0,16]");
  }
  if (!is_between_0_1(aqmh.cherry_pick.k_frac) ||
      aqmh.cherry_pick.k_frac <= 0.0f) {
    throw ValidationError("aqmh.cherry_pick.k_frac must be in (0,1]");
  }
  if (aqmh.cherry_pick.mode != "auto_reject" &&
      aqmh.cherry_pick.mode != "top_k") {
    throw ValidationError(
        "aqmh.cherry_pick.mode must be 'auto_reject' or 'top_k'");
  }
  if (aqmh.cherry_pick.k_min_required < 1) {
    throw ValidationError("aqmh.cherry_pick.k_min_required must be >= 1");
  }
  if (aqmh.cherry_pick.margin_min < 0.0f || aqmh.cherry_pick.margin_min > 1.0f) {
    throw ValidationError("aqmh.cherry_pick.margin_min must be in [0,1]");
  }
  if (!is_between_0_1(aqmh.cherry_pick.reject_below_best_fraction) ||
      aqmh.cherry_pick.reject_below_best_fraction <= 0.0f) {
    throw ValidationError(
        "aqmh.cherry_pick.reject_below_best_fraction must be in (0,1]");
  }
  if (!is_between_0_1(aqmh.cherry_pick.min_keep_fraction) ||
      aqmh.cherry_pick.min_keep_fraction <= 0.0f) {
    throw ValidationError(
        "aqmh.cherry_pick.min_keep_fraction must be in (0,1]");
  }
  int last_min = -1;
  for (const auto &tier : aqmh.cherry_pick.tiered_k_frac) {
    if (tier.min_n_rankable < 0 || tier.min_n_rankable <= last_min ||
        tier.k_frac <= 0.0f || tier.k_frac > 1.0f) {
      throw ValidationError("aqmh.cherry_pick.tiered_k_frac must be strictly ordered with fractions in (0,1]");
    }
    last_min = tier.min_n_rankable;
  }
  if (!(aqmh.global_quality.g_floor > 0.0f && aqmh.global_quality.g_floor < 1.0f) ||
      aqmh.global_quality.g_w_sharp < 0.0f ||
      aqmh.global_quality.g_w_snr < 0.0f ||
      aqmh.global_quality.g_w_background_penalty < 0.0f ||
      aqmh.global_quality.g_w_sharp + aqmh.global_quality.g_w_snr +
              aqmh.global_quality.g_w_background_penalty <=
          0.0f ||
      !(aqmh.global_quality.g_k_scale > 0.0f) ||
      !std::isfinite(aqmh.global_quality.g_k_scale)) {
    throw ValidationError("aqmh.global_quality values are invalid");
  }
  if (aqmh.reconstruction.clip_sigma <= 0.0f ||
      aqmh.reconstruction.clip_sigma_low <= 0.0f ||
      aqmh.reconstruction.clip_sigma_high <= 0.0f ||
      aqmh.reconstruction.clip_iterations < 0 ||
      aqmh.reconstruction.min_fraction <= 0.0f || aqmh.reconstruction.min_fraction > 1.0f ||
      aqmh.reconstruction.min_n_eff < 1.0f) {
    throw ValidationError("aqmh.reconstruction values are invalid");
  }
  if (!is_between_0_1(aqmh.diagnostics.tau_artifact)) {
    throw ValidationError("aqmh.diagnostics.tau_artifact must be in [0,1]");
  }
  if (!is_between_0_1(aqmh.diagnostics.q_region)) {
    throw ValidationError("aqmh.diagnostics.q_region must be in [0,1]");
  }
  if (aqmh.diagnostics.r_morph_canvas_px < 1) {
    throw ValidationError(
        "aqmh.diagnostics.r_morph_canvas_px must be >= 1");
  }
  // NEW VALIDATIONS:
  if (aqmh.diagnostics.level != "none" &&
      aqmh.diagnostics.level != "summary" &&
      aqmh.diagnostics.level != "full") {
    throw ValidationError("aqmh.diagnostics.level must be none, summary, or full");
  }
  if (aqmh.diagnostics.format != "json" && aqmh.diagnostics.format != "binary") {
    throw ValidationError("aqmh.diagnostics.format must be json or binary");
  }
  if (aqmh.diagnostics.binary_block_size_px < 0) {
    throw ValidationError("aqmh.diagnostics.binary_block_size_px must be >= 0");
  }
  if (aqmh.reconstruction.chunk_rows < 0) {
    throw ValidationError("aqmh.reconstruction.chunk_rows must be >= 0");
  }
  if (aqmh.reconstruction.prewarp_interpolation != "linear" &&
      aqmh.reconstruction.prewarp_interpolation != "cubic" &&
      aqmh.reconstruction.prewarp_interpolation != "lanczos4") {
    throw ValidationError(
        "aqmh.reconstruction.prewarp_interpolation must be linear, cubic, or lanczos4");
  }
  if (aqmh.reconstruction.pre_debayer_method != "bilinear" &&
      aqmh.reconstruction.pre_debayer_method != "nearest" &&
      aqmh.reconstruction.pre_debayer_method != "vng" &&
      aqmh.reconstruction.pre_debayer_method != "edge_aware") {
    throw ValidationError(
        "aqmh.reconstruction.pre_debayer_method must be bilinear, nearest, vng, or edge_aware");
  }
  if (aqmh.reconstruction.rgb_q_map_mode != "shared_luma") {
    throw ValidationError(
        "aqmh.reconstruction.rgb_q_map_mode must be shared_luma");
  }
  if (aqmh.reconstruction.rgb_memory_strategy != "sequential") {
    throw ValidationError(
        "aqmh.reconstruction.rgb_memory_strategy must be sequential");
  }
  if (!is_between_0_1(aqmh.reconstruction.registration_weight_floor) ||
      !is_between_0_1(aqmh.reconstruction.registration_cc_floor) ||
      !is_between_0_1(aqmh.reconstruction.registration_cc_full) ||
      aqmh.reconstruction.registration_cc_full <=
          aqmh.reconstruction.registration_cc_floor ||
      !is_between_0_1(aqmh.reconstruction.registration_sequential_factor) ||
      !is_between_0_1(aqmh.reconstruction.registration_predicted_factor) ||
      aqmh.reconstruction.registration_chain_depth_penalty < 0.0f ||
      aqmh.reconstruction.registration_chain_depth_penalty > 0.5f ||
      !is_between_0_1(aqmh.reconstruction.registration_chain_depth_max_penalty)) {
    throw ValidationError(
        "aqmh.reconstruction registration weight guard values are invalid");
  }
  if (!is_between_0_1(aqmh.reconstruction.structure_mask_low_q) ||
      !is_between_0_1(aqmh.reconstruction.structure_mask_high_q) ||
      aqmh.reconstruction.structure_mask_high_q <=
          aqmh.reconstruction.structure_mask_low_q ||
      aqmh.reconstruction.structure_mask_blur_sigma_px < 0.0f ||
      !std::isfinite(aqmh.reconstruction.structure_mask_blur_sigma_px)) {
    throw ValidationError(
        "aqmh.reconstruction structure mask values are invalid");
  }

  if (assumptions.frames_reduced_threshold < assumptions.frames_min) {
    throw ValidationError(
        "assumptions.frames_reduced_threshold must be >= assumptions.frames_min "
        "(i.e. N_red >= frames_min, enforcing N >= max(N_red, frames_min) for clustering)");
  }

  if (synthetic.clustering.cluster_count_range[0] < 1 ||
      synthetic.clustering.cluster_count_range[1] <
          synthetic.clustering.cluster_count_range[0]) {
    throw ValidationError("synthetic.clustering.cluster_count_range must be "
                          "[min,max] with min>=1 and max>=min");
  }
  if (synthetic.clustering.mode != "kmeans" &&
      synthetic.clustering.mode != "quantile") {
    throw ValidationError(
        "synthetic.clustering.mode must be 'kmeans' or 'quantile'");
  }
  if (synthetic.weighting != "global" &&
      synthetic.weighting != "tile_weighted") {
    throw ValidationError(
        "synthetic.weighting must be 'global' or 'tile_weighted'");
  }
  if (synthetic.frames_min < 1)
    throw ValidationError("synthetic.frames_min must be at least 1");
  if (synthetic.frames_max < synthetic.frames_min) {
    throw ValidationError("synthetic.frames_max must be >= frames_min");
  }

  if (bge.method != "none" && bge.method != "classic" &&
      bge.method != "autobge") {
    throw ValidationError("bge.method must be one of: none|classic|autobge");
  }
  if (bge.method == "autobge") {
    if (bge.autobge.num_sample_points < 0 || bge.autobge.num_sample_points > 3000) {
      throw ValidationError("bge.autobge.num_sample_points must be in [0,3000]");
    }
    if (bge.autobge.poly_degree < 1 || bge.autobge.poly_degree > 6) {
      throw ValidationError("bge.autobge.poly_degree must be in [1,6]");
    }
    if (bge.autobge.rbf_smooth < 0.0f || bge.autobge.rbf_smooth > 10.0f) {
      throw ValidationError("bge.autobge.rbf_smooth must be in [0,10]");
    }
    if (bge.autobge.downsample_scale < 1 || bge.autobge.downsample_scale > 8) {
      throw ValidationError("bge.autobge.downsample_scale must be in [1,8]");
    }
    if (bge.autobge.patch_size < 3 || bge.autobge.patch_size > 101 ||
        (bge.autobge.patch_size % 2) == 0) {
      throw ValidationError("bge.autobge.patch_size must be odd and in [3,101]");
    }
    if (bge.autobge.patch_estimator != "median" &&
        bge.autobge.patch_estimator != "sigma_clipped_median") {
      throw ValidationError(
          "bge.autobge.patch_estimator must be one of: median|sigma_clipped_median");
    }
    if (bge.autobge.stretch_mode != "none" &&
        bge.autobge.stretch_mode != "linear" &&
        bge.autobge.stretch_mode != "mtf") {
      throw ValidationError(
          "bge.autobge.stretch_mode must be one of: none|linear|mtf");
    }
    if (bge.autobge.stretch_target_median < 0.01f ||
        bge.autobge.stretch_target_median > 0.99f) {
      throw ValidationError(
          "bge.autobge.stretch_target_median must be in [0.01,0.99]");
    }
    if (bge.autobge.border_margin < 0 || bge.autobge.border_margin > 250) {
      throw ValidationError("bge.autobge.border_margin must be in [0,250]");
    }
    if (bge.autobge.bright_exclusion_fraction < 0.01f ||
        bge.autobge.bright_exclusion_fraction > 0.99f) {
      throw ValidationError(
          "bge.autobge.bright_exclusion_fraction must be in [0.01,0.99]");
    }
    if (bge.autobge.gradient_descent_max_iters < 1 ||
        bge.autobge.gradient_descent_max_iters > 500) {
      throw ValidationError(
          "bge.autobge.gradient_descent_max_iters must be in [1,500]");
    }
    if (bge.autobge.mono_mode != "rgb_duplicate" &&
        bge.autobge.mono_mode != "disabled") {
      throw ValidationError(
          "bge.autobge.mono_mode must be one of: rgb_duplicate|disabled");
    }
    for (const auto &polygon : bge.autobge.exclusion_polygons) {
      if (polygon.size() < 3)
        throw ValidationError("bge.autobge.exclusion_polygons require at least 3 points");
      for (const auto &point : polygon) {
        if (!std::isfinite(point[0]) || !std::isfinite(point[1]) ||
            point[0] < 0.0f || point[0] > 1.0f ||
            point[1] < 0.0f || point[1] > 1.0f)
          throw ValidationError("bge.autobge.exclusion_polygons coordinates must be in [0,1]");
      }
    }
  }

  if (bge.tile_weight_lambda_structure <= 0.0f) {
    throw ValidationError("bge.tile_weight_lambda_structure must be > 0");
  }
  if (bge.sample_quantile <= 0.0f || bge.sample_quantile > 0.5f) {
    throw ValidationError("bge.sample_quantile must be in (0,0.5]");
  }
  if (bge.sample_estimator != "quantile" &&
      bge.sample_estimator != "sigma_clipped_median" &&
      bge.sample_estimator != "sextractor_mode" &&
      bge.sample_estimator != "biweight") {
    throw ValidationError(
        "bge.sample_estimator must be one of: quantile|sigma_clipped_median|sextractor_mode|biweight");
  }
  if (bge.min_sample_bg_value < 0.0f) {
    throw ValidationError("bge.min_sample_bg_value must be >= 0");
  }
  if (bge.structure_thresh_percentile < 0.0f ||
      bge.structure_thresh_percentile > 1.0f) {
    throw ValidationError("bge.structure_thresh_percentile must be in [0,1]");
  }
  if (bge.min_tiles_per_cell < 1) {
    throw ValidationError("bge.min_tiles_per_cell must be >= 1");
  }
  if (bge.min_valid_sample_fraction_for_apply <= 0.0f ||
      bge.min_valid_sample_fraction_for_apply > 1.0f) {
    throw ValidationError(
        "bge.min_valid_sample_fraction_for_apply must be in (0,1]");
  }
  if (bge.min_valid_samples_for_apply < 1) {
    throw ValidationError("bge.min_valid_samples_for_apply must be >= 1");
  }
  if (bge.grid.N_g < 1 || bge.grid.G_min_px < 1 ||
      bge.grid.G_max_fraction <= 0.0f || bge.grid.G_max_fraction > 1.0f) {
    throw ValidationError("bge.grid parameters are out of range");
  }
  if (bge.fit.irls_max_iterations < 1 || bge.fit.irls_tolerance <= 0.0f ||
      bge.fit.huber_delta <= 0.0f ||
      bge.fit.rbf_mu_factor <= 0.0f ||
      bge.fit.rbf_lambda <= 0.0f || bge.fit.rbf_epsilon <= 0.0f) {
    throw ValidationError("bge.fit parameters are out of range");
  }
  if (bge.fit.method != "poly" && bge.fit.method != "spline" &&
      bge.fit.method != "bicubic" && bge.fit.method != "rbf" &&
      bge.fit.method != "modeled_mask_mesh") {
    throw ValidationError(
        "bge.fit.method must be one of: poly|spline|bicubic|rbf|modeled_mask_mesh");
  }
  if (bge.fit.robust_loss != "huber" && bge.fit.robust_loss != "tukey") {
    throw ValidationError("bge.fit.robust_loss must be 'huber' or 'tukey'");
  }
  if (bge.fit.rbf_phi != "thinplate" && bge.fit.rbf_phi != "multiquadric" &&
      bge.fit.rbf_phi != "gaussian") {
    throw ValidationError(
        "bge.fit.rbf_phi must be one of: thinplate|multiquadric|gaussian");
  }
  if (bge.autotune.max_evals < 1 ||
      bge.autotune.holdout_fraction < 0.05f ||
      bge.autotune.holdout_fraction > 0.50f ||
      bge.autotune.alpha_flatness < 0.0f ||
      bge.autotune.beta_roughness < 0.0f) {
    throw ValidationError("bge.autotune parameters are out of range");
  }
  if (bge.autotune.strategy != "conservative" &&
      bge.autotune.strategy != "extended") {
    throw ValidationError(
        "bge.autotune.strategy must be 'conservative' or 'extended'");
  }

  if (pcc.aperture_radius_px <= 0.0f || pcc.annulus_inner_px <= 0.0f ||
      pcc.annulus_outer_px <= 0.0f) {
    throw ValidationError("pcc aperture and annulus radii must be > 0");
  }
  if (pcc.min_stars < 3 || pcc.sigma_clip <= 0.0f) {
    throw ValidationError("pcc.min_stars must be >= 3 and sigma_clip > 0");
  }
  if (pcc.background_model != "median" && pcc.background_model != "plane") {
    throw ValidationError("pcc.background_model must be 'median' or 'plane'");
  }
  if (pcc.max_condition_number < 1.0f || pcc.max_residual_rms <= 0.0f) {
    throw ValidationError(
        "pcc.max_condition_number must be >= 1 and max_residual_rms > 0");
  }
  if (pcc.radii_mode != "fixed" && pcc.radii_mode != "auto_fwhm") {
    throw ValidationError("pcc.radii_mode must be 'fixed' or 'auto_fwhm'");
  }
  if (pcc.aperture_fwhm_mult <= 0.0f || pcc.annulus_inner_fwhm_mult <= 0.0f ||
      pcc.annulus_outer_fwhm_mult <= 0.0f || pcc.min_aperture_px <= 0.0f) {
    throw ValidationError("pcc adaptive radii parameters must be > 0");
  }
  if (pcc.chroma_strength < 0.0f || pcc.chroma_strength > 1.0f) {
    throw ValidationError("pcc.chroma_strength must be in [0,1]");
  }
  if (pcc.k_max <= 0.0f) {
    throw ValidationError("pcc.k_max must be > 0");
  }
  if (pcc.background_neutralization_mode != "always" &&
      pcc.background_neutralization_mode != "auto" &&
      pcc.background_neutralization_mode != "off") {
    throw ValidationError(
        "pcc.background_neutralization_mode must be 'always', 'auto', or 'off'");
  }

  if (hypermetric_stretch.mode != "ready_to_use" &&
      hypermetric_stretch.mode != "scientific") {
    throw ValidationError(
        "hypermetric_stretch.mode must be 'ready_to_use' or 'scientific'");
  }
  if (hypermetric_stretch.target_bg < 0.05f ||
      hypermetric_stretch.target_bg > 0.50f) {
    throw ValidationError("hypermetric_stretch.target_bg must be in [0.05,0.50]");
  }
  if (hypermetric_stretch.protect_b < 0.1f) {
    throw ValidationError("hypermetric_stretch.protect_b must be >= 0.1");
  }
  if (hypermetric_stretch.convergence_power < 1.0f ||
      hypermetric_stretch.convergence_power > 10.0f) {
    throw ValidationError(
        "hypermetric_stretch.convergence_power must be in [1,10]");
  }
  if (hypermetric_stretch.log_d_mode != "auto" &&
      hypermetric_stretch.log_d_mode != "fixed") {
    throw ValidationError(
        "hypermetric_stretch.log_d_mode must be 'auto' or 'fixed'");
  }
  if (hypermetric_stretch.fixed_log_d < 0.0f ||
      hypermetric_stretch.fixed_log_d > 7.0f) {
    throw ValidationError("hypermetric_stretch.fixed_log_d must be in [0,7]");
  }
  if (hypermetric_stretch.color_strategy != "auto" &&
      hypermetric_stretch.color_strategy != "fixed") {
    throw ValidationError(
        "hypermetric_stretch.color_strategy must be 'auto' or 'fixed'");
  }
  if (hypermetric_stretch.fixed_color_strategy < -1.0f ||
      hypermetric_stretch.fixed_color_strategy > 1.0f) {
    throw ValidationError(
        "hypermetric_stretch.fixed_color_strategy must be in [-1,1]");
  }
  if (hypermetric_stretch.color_grip < 0.0f ||
      hypermetric_stretch.color_grip > 1.0f) {
    throw ValidationError("hypermetric_stretch.color_grip must be in [0,1]");
  }
  if (hypermetric_stretch.shadow_convergence < 0.0f) {
    throw ValidationError(
        "hypermetric_stretch.shadow_convergence must be >= 0");
  }
  if (hypermetric_stretch.linear_expansion < 0.0f ||
      hypermetric_stretch.linear_expansion > 1.0f) {
    throw ValidationError(
        "hypermetric_stretch.linear_expansion must be in [0,1]");
  }
  if (hypermetric_stretch.output_rgb.empty()) {
    throw ValidationError("hypermetric_stretch.output_rgb must not be empty");
  }

  if (stacking.method != "average" && stacking.method != "rej") {
    throw ValidationError("stacking.method must be 'average' or 'rej'");
  }
  if (!is_between_0_1(stacking.common_overlap_required_fraction) ||
      stacking.common_overlap_required_fraction <= 0.0f) {
    throw ValidationError(
        "stacking.common_overlap_required_fraction must be in (0,1]");
  }
  if (!is_between_0_1(stacking.tile_common_valid_min_fraction) ||
      stacking.tile_common_valid_min_fraction <= 0.0f) {
    throw ValidationError(
        "stacking.tile_common_valid_min_fraction must be in (0,1]");
  }
  if (stacking.sigma_clip.sigma_low <= 0.0f ||
      stacking.sigma_clip.sigma_high <= 0.0f) {
    throw ValidationError("stacking.sigma_clip.sigma_low/high must be > 0");
  }
  if (stacking.sigma_clip.max_iters < 1) {
    throw ValidationError("stacking.sigma_clip.max_iters must be >= 1");
  }
  if (!is_between_0_1(stacking.sigma_clip.min_fraction)) {
    throw ValidationError("stacking.sigma_clip.min_fraction must be in [0,1]");
  }
  if (stacking.cluster_quality_weighting.kappa_cluster <= 0.0f) {
    throw ValidationError(
        "stacking.cluster_quality_weighting.kappa_cluster must be > 0");
  }
  if (stacking.cluster_quality_weighting.cap_enabled &&
      stacking.cluster_quality_weighting.cap_ratio <= 0.0f) {
    throw ValidationError("stacking.cluster_quality_weighting.cap_ratio must be "
                          "> 0 when cap_enabled=true");
  }
  if (stacking.cosmetic_correction_sigma <= 0.0f) {
    throw ValidationError("stacking.cosmetic_correction_sigma must be > 0");
  }
  if (runtime_limits.tile_analysis_max_factor_vs_stack <= 0.0f) {
    throw ValidationError(
        "runtime_limits.tile_analysis_max_factor_vs_stack must be > 0");
  }
  if (runtime_limits.hard_abort_hours <= 0.0f) {
    throw ValidationError("runtime_limits.hard_abort_hours must be > 0");
  }
  if (runtime_limits.parallel_workers < 1) {
    throw ValidationError("runtime_limits.parallel_workers must be >= 1");
  }
  if (runtime_limits.memory_budget < 1) {
    throw ValidationError("runtime_limits.memory_budget must be >= 1");
  }
  const std::string backend =
      normalize_acceleration_backend(runtime_limits.acceleration_backend);
  if (backend != "auto" && backend != "cpu" && backend != "opencv_cuda" &&
      backend != "opencv_opencl" && backend != "opencl" && backend != "cuda") {
    throw ValidationError(
        "runtime_limits.acceleration_backend must be auto, cpu, opencv_cuda, opencv_opencl, opencl, or cuda");
  }
  if (runtime_limits.tile_reconstruction_diagnostics != "full" &&
      runtime_limits.tile_reconstruction_diagnostics != "minimal" &&
      runtime_limits.tile_reconstruction_diagnostics != "off") {
    throw ValidationError(
        "runtime_limits.tile_reconstruction_diagnostics must be full, minimal, or off");
  }
}

/// @brief Implements get schema json.
/// @details Part of YAML configuration loading, serialization, schema generation, and validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::string getEffectiveMethod(const Config& config) {
    const char* forceClassic = std::getenv("FORCE_CLASSIC");
    if (forceClassic && std::string(forceClassic) == "1") {
        return "classic_tile_compile";
    }
    return config.method;
}

/// @details Part of YAML configuration loading, serialization, schema generation, and validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::string get_schema_json() {
  for (const fs::path &candidate : {
           fs::path("tile_compile.schema.json"),
           fs::path("tile_compile_cpp") / "tile_compile.schema.json",
           fs::path("..") / "tile_compile.schema.json",
           fs::path("..") / "tile_compile_cpp" / "tile_compile.schema.json",
       }) {
    std::error_code ec;
    if (!fs::exists(candidate, ec) || ec) {
      continue;
    }
    std::ifstream in(candidate);
    if (!in) {
      continue;
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();
    if (!text.empty()) {
      return text;
    }
  }

  return R"({
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "tile_compile v3 config",
  "type": "object",
  "properties": {
    "pipeline": { "type":"object",
                      "properties": { "mode":{"type":"string","enum":["production","test"]} } },
    "output": { "type":"object",
      "properties": { "registered_dir":{"type":"string"},
                      "write_registered_frames":{"type":"boolean"},
                      "crop_to_nonzero_bbox":{"type":"boolean"} } },
    "data": { "type":"object",
      "properties": { "image_width":{"type":"integer","minimum":0},
                      "image_height":{"type":"integer","minimum":0},
                      "color_mode":{"type":"string","enum":["OSC","MONO","RGB"]},
                      "bayer_pattern":{"type":"string"},
                      "linear_required":{"type":"boolean","deprecated":true,
                                         "description":"Deprecated: non-linear frames are warn-only in the runner and are no longer removed."} } },
    "linearity": { "type":"object",
      "properties": { "enabled":{"type":"boolean"},
                      "max_frames":{"type":"integer","minimum":1},
                      "min_overall_linearity":{"type":"number","minimum":0,"maximum":1},
                      "strictness":{"type":"string","enum":["strict","moderate","permissive"]} } },
    "calibration": { "type":"object",
      "properties": { "use_bias":{"type":"boolean"}, "use_dark":{"type":"boolean"}, "use_flat":{"type":"boolean"},
                      "bias_use_master":{"type":"boolean"}, "dark_use_master":{"type":"boolean"}, "dark_already_bias_corrected":{"type":"boolean"}, "flat_use_master":{"type":"boolean"},
                      "dark_auto_select":{"type":"boolean"},
                      "dark_match_exposure_tolerance_percent":{"type":"number","minimum":0},
                      "dark_match_use_temp":{"type":"boolean"},
                      "dark_match_temp_tolerance_c":{"type":"number","minimum":0},
                      "bias_dir":{"type":"string"}, "darks_dir":{"type":"string"}, "flats_dir":{"type":"string"},
                      "bias_master":{"type":"string"}, "dark_master":{"type":"string"}, "flat_master":{"type":"string"},
                      "pattern":{"type":"string"} } },
    "assumptions": { "type":"object",
      "properties": { "frames_min":{"type":"integer","minimum":1},
                      "frames_reduced_threshold":{"type":"integer","minimum":1},
                      "reduced_mode_skip_clustering":{"type":"boolean"},
                      "reduced_mode_cluster_range":{"type":"array","items":{"type":"integer","minimum":1},"minItems":2,"maxItems":2} } },
    "normalization": { "type":"object",
      "properties": { "enabled":{"type":"boolean"},
                      "mode":{"type":"string","enum":["background","median"]},
                      "per_channel":{"type":"boolean"} } },
    "registration": { "type":"object",
      "properties": { "engine":{"type":"string","enum":["triangle_star_matching","star_similarity","hybrid_phase_ecc","robust_phase_ecc"]},
                      "transform_model":{"type":"string","enum":["similarity","affine"]},
                      "enable_star_pair_fallback":{"type":"boolean"},
                      "allow_rotation":{"type":"boolean"},
                      "star_topk":{"type":"integer","minimum":3},
                      "star_min_inliers":{"type":"integer","minimum":2},
                      "star_inlier_tol_px":{"type":"number","exclusiveMinimum":0},
                      "star_dist_bin_px":{"type":"number","exclusiveMinimum":0},
                      "reject_outliers":{"type":"boolean"},
                      "reject_cc_min_abs":{"type":"number","minimum":0,"maximum":1},
                      "reject_shift_px_min":{"type":"number","minimum":0},
                      "reject_shift_median_multiplier":{"type":"number","exclusiveMinimum":0},
                      "reject_scale_min":{"type":"number","exclusiveMinimum":0},
                      "reject_scale_max":{"type":"number","exclusiveMinimum":0},
                      "max_blind_chain_depth":{"type":"integer","minimum":0,"maximum":100},
                      "blind_chain_strong_anchor_cc":{"type":"number","minimum":0.01,"maximum":0.5},
                      "blind_chain_drift_threshold_px":{"type":"number","minimum":0.5,"maximum":10.0},
                      "use_astrometry":{"type":"boolean"},
                      "enable_local_background_subtraction":{"type":"boolean"},
                      "star_shift_radius_px":{"type":"number","minimum":10,"maximum":2000} } },
    "dithering": { "type":"object",
      "properties": { "enabled":{"type":"boolean"},
                      "min_shift_px":{"type":"number","minimum":0} } },
    "tile_denoise": { "type":"object",
      "properties": {
        "soft_threshold": { "type":"object",
          "properties": { "enabled":{"type":"boolean"},
                          "blur_kernel":{"type":"integer","minimum":3},
                          "alpha":{"type":"number","exclusiveMinimum":0},
                          "skip_star_tiles":{"type":"boolean"} } },
        "wiener": { "type":"object",
          "properties": { "enabled":{"type":"boolean"},
                          "snr_threshold":{"type":"number","minimum":0},
                          "q_min":{"type":"number","minimum":-1},
                          "q_max":{"type":"number","minimum":0,"maximum":1},
                          "q_step":{"type":"number","exclusiveMinimum":0},
                          "min_snr":{"type":"number","minimum":0},
                          "max_iterations":{"type":"integer","minimum":1} } } } },
    "chroma_denoise": { "type":"object",
      "properties": { "enabled":{"type":"boolean"},
                      "color_space":{"type":"string","enum":["ycbcr_linear","opponent_linear"]},
                      "apply_stage":{"type":"string","enum":["pre_stack_tiles","post_stack_linear","post_pcc"]},
                      "protect_luma":{"type":"boolean"},
                      "luma_guard_strength":{"type":"number","minimum":0,"maximum":1},
                      "star_protection":{"type":"object","properties":{
                        "enabled":{"type":"boolean"},
                        "threshold_sigma":{"type":"number","exclusiveMinimum":0},
                        "dilate_px":{"type":"integer","minimum":0}}},
                      "structure_protection":{"type":"object","properties":{
                        "enabled":{"type":"boolean"},
                        "gradient_percentile":{"type":"number","minimum":0,"maximum":100}}},
                      "chroma_wavelet":{"type":"object","properties":{
                        "enabled":{"type":"boolean"},
                        "levels":{"type":"integer","minimum":1},
                        "threshold_scale":{"type":"number","exclusiveMinimum":0},
                        "soft_k":{"type":"number","exclusiveMinimum":0}}},
                      "chroma_bilateral":{"type":"object","properties":{
                        "enabled":{"type":"boolean"},
                        "sigma_spatial":{"type":"number","exclusiveMinimum":0},
                        "sigma_range":{"type":"number","exclusiveMinimum":0}}},
                      "blend":{"type":"object","properties":{
                        "mode":{"type":"string","enum":["chroma_only"]},
                        "amount":{"type":"number","minimum":0,"maximum":1}}} } },
    "global_metrics": { "type":"object",
      "properties": { "adaptive_weights":{"type":"boolean"},
                      "weight_exponent_scale":{"type":"number","exclusiveMinimum":0,"description":"Exponent scale k for G_f = exp(k * Q_f). k=1.0 (default) is standard, k>1 increases differentiation between good/bad frames."},
                      "weights":{"type":"object","properties":{"background":{"type":"number","minimum":0,"maximum":1},"noise":{"type":"number","minimum":0,"maximum":1},"gradient":{"type":"number","minimum":0,"maximum":1},"fwhm":{"type":"number","minimum":0,"maximum":1},"roundness":{"type":"number","minimum":0,"maximum":1},"star_count":{"type":"number","minimum":0,"maximum":1}}},
                      "clamp":{"type":"array","items":{"type":"number"},"minItems":2,"maxItems":2} } },
    "tile": { "type":"object",
      "properties": { "size_factor":{"type":"integer","minimum":1},
                      "min_size":{"type":"integer","minimum":1},
                      "max_divisor":{"type":"integer","minimum":1},
                      "overlap_fraction":{"type":"number","minimum":0,"maximum":0.5},
                      "star_min_count":{"type":"integer","minimum":0},
                      "star_soft_count":{"type":"integer","minimum":0} } },
    "local_metrics": { "type":"object",
      "properties": { "clamp":{"type":"array","items":{"type":"number"},"minItems":2,"maxItems":2},
                      "neighborhood_normalization":{"type":"object","properties":{"enabled":{"type":"boolean"},"radius":{"type":"integer","minimum":0},"blend":{"type":"number","minimum":0,"maximum":1}}},
                      "spatial_regularization":{"type":"object","properties":{"enabled":{"type":"boolean"},"lambda":{"type":"number","minimum":0,"maximum":1},"passes":{"type":"integer","minimum":0},"tau_local":{"type":"number","exclusiveMinimum":0}}},
                      "star_mode":{"type":"object","properties":{"weights":{"type":"object","properties":{"fwhm":{"type":"number","minimum":0,"maximum":1},"roundness":{"type":"number","minimum":0,"maximum":1},"contrast":{"type":"number","minimum":0,"maximum":1}}}}},
                      "structure_mode":{"type":"object","properties":{"background_weight":{"type":"number","minimum":0,"maximum":1},"metric_weight":{"type":"number","minimum":0,"maximum":1}}} } },
    "aqmh": { "type":"object",
      "properties": { "enabled":{"type":"boolean"},
                      "pyramid":{"type":"object","properties":{"scales":{"type":"integer","minimum":1,"maximum":8,"default":4},"base_window_px":{"type":"integer","minimum":1,"default":4},"w_sharp":{"type":"number","minimum":0,"default":0.6},"w_snr":{"type":"number","minimum":0,"default":0.4},"score_scale":{"type":"number","exclusiveMinimum":0,"default":1.8},"k_artifact":{"type":"number","exclusiveMinimum":0,"default":3.0},"frac_artifact_max":{"type":"number","exclusiveMinimum":0,"maximum":1,"default":0.25}}},
                      "storage":{"type":"object","properties":{"resolution_divisor":{"type":"integer","enum":[1,2,4],"default":2,"description":"Downsamples stored AQMH quality maps. 1 keeps full resolution, 2 stores half width/height (~1/4 pixels), 4 stores quarter width/height. HARD RULE: if recommending cherry_pick.enabled=true in the same analysis or effective config, recommend resolution_divisor=1. Never recommend cherry_pick.enabled=true together with resolution_divisor=2 or 4."},"dtype":{"type":"string","enum":["float32","uint16","uint8"],"default":"uint16","description":"Storage data type for AQMH quality maps. float32 is exact; uint16 is recommended for lower disk and I/O cost; uint8 is smallest but coarser."},"max_resident_maps":{"type":"integer","minimum":0,"maximum":16,"default":2,"description":"Maximum number of full-resolution AQMH quality maps kept in RAM by the reconstruction read cache. 0 disables the read cache."}}},
                      "global_quality":{"type":"object","properties":{"g_floor":{"type":"number","exclusiveMinimum":0,"exclusiveMaximum":1,"default":0.03},"g_w_sharp":{"type":"number","minimum":0,"default":0.55},"g_w_snr":{"type":"number","minimum":0,"default":0.30},"g_w_background_penalty":{"type":"number","minimum":0,"default":0.25},"g_k_scale":{"type":"number","exclusiveMinimum":0,"default":1.5,"description":"Sigmoid temperature for global AQMH quality. The resulting frame weight remains bounded to [g_floor, 1]."}}},
                      "cherry_pick":{"type":"object","properties":{"enabled":{"type":"boolean","default":false,"description":"Enables per-pixel AQMH frame selection during reconstruction."},"mode":{"type":"string","enum":["auto_reject","top_k"],"default":"auto_reject","description":"auto_reject keeps most locally usable frames and rejects only clear low-score outliers; top_k is the legacy fixed best-k selection."},"k_frac":{"type":"number","exclusiveMinimum":0,"maximum":1,"default":0.30,"description":"Fraction retained in legacy mode=top_k."},"k_min_required":{"type":"integer","minimum":1,"default":20},"margin_min":{"type":"number","minimum":0,"maximum":1,"default":0.02},"reject_below_best_fraction":{"type":"number","exclusiveMinimum":0,"maximum":1,"default":0.25,"description":"In mode=auto_reject, reject samples only when their local score is below this fraction of the local best score."},"min_keep_fraction":{"type":"number","exclusiveMinimum":0,"maximum":1,"default":0.9,"description":"In mode=auto_reject, retain at least this fraction of locally rankable samples."},"tiered_k_frac":{"type":"array","default":[],"items":{"type":"object","properties":{"min_n_rankable":{"type":"integer","minimum":0},"k_frac":{"type":"number","exclusiveMinimum":0,"maximum":1}},"required":["min_n_rankable","k_frac"]}}}},
                      "reconstruction":{"type":"object","properties":{"clip_sigma":{"type":"number","exclusiveMinimum":0,"default":2.0},"clip_sigma_low":{"type":"number","exclusiveMinimum":0,"default":2.0},"clip_sigma_high":{"type":"number","exclusiveMinimum":0,"default":2.0},"clip_iterations":{"type":"integer","minimum":0,"default":4},"min_fraction":{"type":"number","exclusiveMinimum":0,"maximum":1,"default":0.4},"min_n_eff":{"type":"number","minimum":1,"default":2.0},"chunk_rows":{"type":"integer","minimum":0,"default":0},"memory_budget_mb":{"type":"integer","minimum":0,"default":0},"delete_prewarped_cache_after_run":{"type":"boolean","default":true,"description":"Controls deletion of the disk-backed cache/prewarped_frames directory after a successful run. true saves disk space but prevents direct resume from AQMH_RECONSTRUCTION or STACKING; false retains registered and prewarped frames for those resumes without repeating registration and PREWARP. The cache can require several tens of gigabytes."},"prewarp_interpolation":{"type":"string","enum":["linear","cubic","lanczos4"],"default":"linear","description":"Interpolation kernel used when prewarping registered frames onto the common canvas before AQMH reconstruction and stacking. linear is the conservative default; cubic and lanczos4 are explicit tuning options that can preserve more high-frequency detail but may increase background noise or ringing."},"registration_weight_guard":{"type":"boolean","default":true},"registration_weight_floor":{"type":"number","minimum":0,"maximum":1,"default":0.30},"registration_cc_floor":{"type":"number","minimum":0,"maximum":1,"default":0.35},"registration_cc_full":{"type":"number","minimum":0,"maximum":1,"default":0.8},"registration_sequential_factor":{"type":"number","minimum":0,"maximum":1,"default":0.92},"registration_predicted_factor":{"type":"number","minimum":0,"maximum":1,"default":0.50},"registration_chain_depth_penalty":{"type":"number","minimum":0,"maximum":0.5,"default":0.03},"registration_chain_depth_max_penalty":{"type":"number","minimum":0,"maximum":1,"default":0.15},"structure_mask_low_q":{"type":"number","minimum":0,"maximum":1,"default":0.40},"structure_mask_high_q":{"type":"number","minimum":0,"maximum":1,"default":0.90},"structure_mask_blur_sigma_px":{"type":"number","minimum":0,"default":4.0}}},
                      "validation":{"type":"object","properties":{"max_seam_score_regression":{"type":"number","minimum":0,"default":0.05},"max_fwhm_regression":{"type":"number","minimum":0,"default":0.02},"max_background_rms_regression":{"type":"number","minimum":0,"default":0.05},"max_tail11_abs_regression":{"type":"number","minimum":0,"default":0.10},"max_elongation_regression":{"type":"number","minimum":0,"default":0.08}}},
                      "diagnostics":{"type":"object","properties":{"enabled":{"type":"boolean","default":true},"level":{"type":"string","enum":["none","summary","full"],"default":"full"},"per_frame_blocks":{"type":"boolean","default":true},"heatmaps":{"type":"boolean","default":true},"regions":{"type":"boolean","default":true},"format":{"type":"string","enum":["json","binary"],"default":"json"},"binary_block_size_px":{"type":"integer","minimum":0,"default":0},"tau_artifact":{"type":"number","minimum":0,"maximum":1,"default":0.20},"q_region":{"type":"number","minimum":0,"maximum":1,"default":0.75},"r_morph_canvas_px":{"type":"integer","minimum":1,"default":6}}} } },
    "synthetic": { "type":"object",
      "properties": { "weighting":{"type":"string","enum":["global","tile_weighted"]},
                      "frames_min":{"type":"integer","minimum":1},
                      "frames_max":{"type":"integer","minimum":1},
                      "clustering":{"type":"object","properties":{"mode":{"type":"string","enum":["kmeans","quantile"]},"cluster_count_range":{"type":"array","items":{"type":"integer","minimum":1},"minItems":2,"maxItems":2}}} } },
    "astrometry": { "type":"object",
      "properties": { "enabled":{"type":"boolean"},
                      "astap_bin":{"type":"string"},
                      "astap_data_dir":{"type":"string"},
                      "search_radius":{"type":"integer","minimum":1,"maximum":360} } },
    "bge": { "type":"object",
      "properties": { "enabled":{"type":"boolean"},
                      "method":{"type":"string","enum":["none","classic","autobge"],"default":"none"},
                      "autobge":{"type":"object","properties":{"num_sample_points":{"type":"integer","minimum":0,"default":0},"poly_degree":{"type":"integer","minimum":1,"maximum":6,"default":2},"rbf_smooth":{"type":"number","minimum":0,"default":0.1},"downsample_scale":{"type":"integer","minimum":1,"default":4},"patch_size":{"type":"integer","minimum":3,"default":15},"patch_estimator":{"type":"string","enum":["median","sigma_clipped_median"],"default":"median"},"stretch_mode":{"type":"string","enum":["none","linear","mtf"],"default":"linear"},"stretch_target_median":{"type":"number","exclusiveMinimum":0,"maximum":1,"default":0.25},"border_margin":{"type":"integer","minimum":0,"default":10},"bright_exclusion_fraction":{"type":"number","exclusiveMinimum":0,"exclusiveMaximum":1,"default":0.5},"gradient_descent_max_iters":{"type":"integer","minimum":1,"default":100},"random_seed":{"type":"integer","default":42},"normalize_between_stages":{"type":"boolean","default":true},"apply_guards":{"type":"boolean","default":true},"mono_mode":{"type":"string","enum":["rgb_duplicate","disabled"],"default":"rgb_duplicate"}}},
                      "tile_weight_lambda_structure":{"type":"number","minimum":0},
                      "sample_quantile":{"type":"number","exclusiveMinimum":0,"maximum":0.5},
                      "sample_estimator":{"type":"string","enum":["quantile","sigma_clipped_median","sextractor_mode","biweight"]},
                      "min_sample_bg_value":{"type":"number","minimum":0},
                      "structure_thresh_percentile":{"type":"number","minimum":0,"maximum":1},
                      "min_tiles_per_cell":{"type":"integer","minimum":1},
                      "min_valid_sample_fraction_for_apply":{"type":"number","exclusiveMinimum":0,"maximum":1},
                      "min_valid_samples_for_apply":{"type":"integer","minimum":1},
                      "mask":{"type":"object","properties":{"star_dilate_px":{"type":"integer","minimum":0},"sat_dilate_px":{"type":"integer","minimum":0}}},
                      "grid":{"type":"object","properties":{"N_g":{"type":"integer","minimum":1},"G_min_px":{"type":"integer","minimum":1},"G_max_fraction":{"type":"number","exclusiveMinimum":0,"maximum":1},"insufficient_cell_strategy":{"type":"string","enum":["discard","nearest","radius_expand"]}}},
                      "fit":{"type":"object","properties":{"method":{"type":"string","enum":["poly","spline","bicubic","rbf","modeled_mask_mesh"]},"robust_loss":{"type":"string","enum":["huber","tukey"]},"huber_delta":{"type":"number","exclusiveMinimum":0},"irls_max_iterations":{"type":"integer","minimum":1},"irls_tolerance":{"type":"number","exclusiveMinimum":0},"polynomial_order":{"type":"integer","enum":[2,3]},"rbf_phi":{"type":"string","enum":["thinplate","multiquadric","gaussian"]},"rbf_mu_factor":{"type":"number","exclusiveMinimum":0},"rbf_lambda":{"type":"number","exclusiveMinimum":0},"rbf_epsilon":{"type":"number","exclusiveMinimum":0}}},
                      "autotune":{"type":"object","properties":{"enabled":{"type":"boolean"},"max_evals":{"type":"integer","minimum":1},"holdout_fraction":{"type":"number","minimum":0.05,"maximum":0.5},"alpha_flatness":{"type":"number","minimum":0},"beta_roughness":{"type":"number","minimum":0},"strategy":{"type":"string","enum":["conservative","extended"]}}} } },
    "pcc": { "type":"object",
      "properties": { "enabled":{"type":"boolean"},
                      "source":{"type":"string","enum":["auto","siril","vizier_gaia","vizier_apass"]},
                      "mag_limit":{"type":"number","minimum":1,"maximum":22},
                      "mag_bright_limit":{"type":"number","minimum":0,"maximum":15},
                      "aperture_radius_px":{"type":"number","exclusiveMinimum":0},
                      "annulus_inner_px":{"type":"number","exclusiveMinimum":0},
                      "annulus_outer_px":{"type":"number","exclusiveMinimum":0},
                      "min_stars":{"type":"integer","minimum":3},
                      "sigma_clip":{"type":"number","exclusiveMinimum":0},
                      "background_model":{"type":"string","enum":["median","plane"]},
                      "max_condition_number":{"type":"number","minimum":1},
                      "max_residual_rms":{"type":"number","exclusiveMinimum":0},
                      "radii_mode":{"type":"string","enum":["fixed","auto_fwhm"]},
                      "aperture_fwhm_mult":{"type":"number","exclusiveMinimum":0},
                      "annulus_inner_fwhm_mult":{"type":"number","exclusiveMinimum":0},
                      "annulus_outer_fwhm_mult":{"type":"number","exclusiveMinimum":0},
                      "min_aperture_px":{"type":"number","exclusiveMinimum":0},
                      "siril_catalog_dir":{"type":"string"},
                      "apply_attenuation":{"type":"boolean"},
                      "chroma_strength":{"type":"number","minimum":0,"maximum":1},
                      "background_neutralization_mode":{"type":"string","enum":["always","auto","off"]},
                      "k_max":{"type":"number","exclusiveMinimum":0} } },
    "hypermetric_stretch": { "type":"object",
      "properties": { "enabled":{"type":"boolean"},
                      "require_successful_pcc":{"type":"boolean"},
                      "mode":{"type":"string","enum":["ready_to_use","scientific"]},
                      "sensor_profile":{"type":"string"},
                      "fallback_profile":{"type":"string"},
                      "adaptive_anchor":{"type":"boolean"},
                      "target_bg":{"type":"number","minimum":0.05,"maximum":0.50},
                      "protect_b":{"type":"number","minimum":0.1},
                      "convergence_power":{"type":"number","minimum":1.0,"maximum":10.0},
                      "log_d_mode":{"type":"string","enum":["auto","fixed"]},
                      "fixed_log_d":{"type":"number","minimum":0,"maximum":7},
                      "color_strategy":{"type":"string","enum":["auto","fixed"]},
                      "fixed_color_strategy":{"type":"number","minimum":-1,"maximum":1},
                      "color_grip":{"type":"number","minimum":0,"maximum":1},
                      "shadow_convergence":{"type":"number","minimum":0},
                      "linear_expansion":{"type":"number","minimum":0,"maximum":1},
                      "write_channels":{"type":"boolean"},
                      "output_rgb":{"type":"string"} } },
    "stacking": { "type":"object",
      "properties": { "method":{"type":"string","enum":["rej","average"]},
                      "sigma_clip":{"type":"object","properties":{"sigma_low":{"type":"number","exclusiveMinimum":0},"sigma_high":{"type":"number","exclusiveMinimum":0},"max_iters":{"type":"integer","minimum":1},"min_fraction":{"type":"number","minimum":0,"maximum":1}}},
                      "cluster_quality_weighting":{"type":"object","properties":{"enabled":{"type":"boolean"},"kappa_cluster":{"type":"number","exclusiveMinimum":0,"description":"Quality-weight exponent for synthetic-cluster aggregation: w_k = exp(kappa_cluster * Q_k)."},"cap_enabled":{"type":"boolean"},"cap_ratio":{"type":"number","exclusiveMinimum":0,"description":"Optional dominance cap ratio for cluster weights: w_k <= cap_ratio * median_j(w_j)."}}},
                      "output_stretch":{"type":"boolean"},
                      "cosmetic_correction":{"type":"boolean"},
                      "cosmetic_correction_sigma":{"type":"number","exclusiveMinimum":0},
                      "per_frame_cosmetic_correction":{"type":"boolean"},
                      "per_frame_cosmetic_correction_sigma":{"type":"number","exclusiveMinimum":0} } },
    "validation": { "type":"object",
      "properties": { "min_fwhm_improvement_percent":{"type":"number"},
                      "max_background_rms_increase_percent":{"type":"number"},
                      "min_tile_weight_variance":{"type":"number","minimum":0},
                      "require_no_tile_pattern":{"type":"boolean"} } },
    "runtime_limits": { "type":"object",
      "properties": { "tile_analysis_max_factor_vs_stack":{"type":"number","exclusiveMinimum":0},
                      "hard_abort_hours":{"type":"number","exclusiveMinimum":0},
                      "allow_emergency_mode":{"type":"boolean"},
                      "parallel_workers":{"type":"integer","minimum":1},
                      "memory_budget":{"type":"integer","minimum":1},
                      "acceleration_backend":{"type":"string","enum":["auto","cpu","opencv_cuda","opencv_opencl","opencl","cuda"],"description":"Beschleunigungs-Backend fuer PREWARP, TILE_RECONSTRUCTION und STACKING. AQMH_MAPS und AQMH_RECONSTRUCTION sind CPU-only, weil die M42-Messungen auf GPU instabile oder langsamere Laufzeiten gezeigt haben."},
                      "tile_reconstruction_diagnostics":{"type":"string","enum":["full","minimal","off"]} } }
  }
})";
}

} // namespace tile_compile::config
