#include "tile_compile/config/configuration.hpp"
#include "tile_compile/config/legacy_config_migration.hpp"
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

Config Config::from_yaml_text_migrated(const std::string &yaml_text,
                                      ConfigMigrationReport &report) {
  const std::string sanitized = sanitize_yaml_windows_paths(yaml_text);
  YAML::Node node = YAML::Load(sanitized);
  migrate_legacy_config_node(node, report);  // may throw ConfigError
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
    if (yaml_has_value(r["prewarp_interpolation"]))
      cfg.registration.prewarp_interpolation =
          r["prewarp_interpolation"].as<std::string>();
    if (yaml_has_value(r["debayer_first"]))
      cfg.registration.debayer_first = r["debayer_first"].as<bool>();
    if (yaml_has_value(r["pre_debayer_method"]))
      cfg.registration.pre_debayer_method =
          r["pre_debayer_method"].as<std::string>();
  }

  if (node["dithering"]) {
    auto d = node["dithering"];
    if (yaml_has_value(d["enabled"]))
      cfg.dithering.enabled = d["enabled"].as<bool>();
    if (yaml_has_value(d["min_shift_px"]))
      cfg.dithering.min_shift_px = d["min_shift_px"].as<float>();
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
    if (yaml_has_value(cd["adaptation_reference_ratio"]))
      cfg.chroma_denoise.adaptation_reference_ratio =
          cd["adaptation_reference_ratio"].as<float>();

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

    if (yaml_has_value(cd["extended_source_protection"])) {
      auto es = cd["extended_source_protection"];
      if (yaml_has_value(es["enabled"]))
        cfg.chroma_denoise.extended_source_protection.enabled = es["enabled"].as<bool>();
      if (yaml_has_value(es["luma_sigma"]))
        cfg.chroma_denoise.extended_source_protection.luma_sigma =
            es["luma_sigma"].as<float>();
      if (yaml_has_value(es["dilate_px"]))
        cfg.chroma_denoise.extended_source_protection.dilate_px =
            es["dilate_px"].as<int>();
    }

    if (yaml_has_value(cd["large_scale_bias"])) {
      auto lb = cd["large_scale_bias"];
      if (yaml_has_value(lb["enabled"]))
        cfg.chroma_denoise.large_scale_bias.enabled = lb["enabled"].as<bool>();
      if (yaml_has_value(lb["block_size"]))
        cfg.chroma_denoise.large_scale_bias.block_size =
            lb["block_size"].as<int>();
      if (yaml_has_value(lb["blur_sigma"]))
        cfg.chroma_denoise.large_scale_bias.blur_sigma =
            lb["blur_sigma"].as<float>();
      if (yaml_has_value(lb["strength"]))
        cfg.chroma_denoise.large_scale_bias.strength =
            lb["strength"].as<float>();
    }
  }

  if (node["luma_denoise"]) {
    auto ld = node["luma_denoise"];
    if (yaml_has_value(ld["enabled"]))
      cfg.luma_denoise.enabled = ld["enabled"].as<bool>();
    if (yaml_has_value(ld["luma_guard_strength"]))
      cfg.luma_denoise.luma_guard_strength = ld["luma_guard_strength"].as<float>();
    if (yaml_has_value(ld["blend_amount"]))
      cfg.luma_denoise.blend_amount = ld["blend_amount"].as<float>();

    if (yaml_has_value(ld["star_protection"])) {
      auto sp = ld["star_protection"];
      if (yaml_has_value(sp["enabled"]))
        cfg.luma_denoise.star_protection.enabled = sp["enabled"].as<bool>();
      if (yaml_has_value(sp["threshold_sigma"]))
        cfg.luma_denoise.star_protection.threshold_sigma =
            sp["threshold_sigma"].as<float>();
      if (yaml_has_value(sp["dilate_px"]))
        cfg.luma_denoise.star_protection.dilate_px = sp["dilate_px"].as<int>();
    }

    if (yaml_has_value(ld["structure_protection"])) {
      auto st = ld["structure_protection"];
      if (yaml_has_value(st["enabled"]))
        cfg.luma_denoise.structure_protection.enabled = st["enabled"].as<bool>();
      if (yaml_has_value(st["gradient_percentile"]))
        cfg.luma_denoise.structure_protection.gradient_percentile =
            st["gradient_percentile"].as<float>();
    }

    if (yaml_has_value(ld["wavelet"])) {
      auto w = ld["wavelet"];
      if (yaml_has_value(w["enabled"]))
        cfg.luma_denoise.wavelet.enabled = w["enabled"].as<bool>();
      if (yaml_has_value(w["levels"]))
        cfg.luma_denoise.wavelet.levels = w["levels"].as<int>();
      if (yaml_has_value(w["threshold_scale"]))
        cfg.luma_denoise.wavelet.threshold_scale = w["threshold_scale"].as<float>();
      if (yaml_has_value(w["soft_k"]))
        cfg.luma_denoise.wavelet.soft_k = w["soft_k"].as<float>();
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

  // Single-method reconstruction contract (plan 6.1). Parsed and validated but
  // not yet consumed by any phase (built in M2+).
  if (node["reconstruction"]) {
    auto r = node["reconstruction"];
    auto &rc = cfg.reconstruction;
    if (yaml_has_value(r["delete_source_cache_after_run"]))
      rc.delete_source_cache_after_run =
          r["delete_source_cache_after_run"].as<bool>();
    if (yaml_has_value(r["keep_profile_cache_after_run"]))
      rc.keep_profile_cache_after_run =
          r["keep_profile_cache_after_run"].as<bool>();
    if (yaml_has_value(r["common_overlap_required_fraction"]))
      rc.common_overlap_required_fraction =
          r["common_overlap_required_fraction"].as<float>();
    if (yaml_has_value(r["diagnostics"]) &&
        yaml_has_value(r["diagnostics"]["level"]))
      rc.diagnostics.level = r["diagnostics"]["level"].as<std::string>();
    if (yaml_has_value(r["diagnostics"]) &&
        yaml_has_value(r["diagnostics"]["preview_forward_drizzle_uniform"]))
      rc.diagnostics.preview_forward_drizzle_uniform =
          r["diagnostics"]["preview_forward_drizzle_uniform"].as<bool>();
    if (yaml_has_value(r["diagnostics"]) &&
        yaml_has_value(r["diagnostics"]["persist_forward_drizzle_uniform_store"]))
      rc.diagnostics.persist_forward_drizzle_uniform_store =
          r["diagnostics"]["persist_forward_drizzle_uniform_store"].as<bool>();
    if (yaml_has_value(r["drizzle"])) {
      auto d = r["drizzle"];
      if (yaml_has_value(d["internal_scale"]))
        rc.drizzle.internal_scale = d["internal_scale"].as<int>();
      if (yaml_has_value(d["output_scale"]))
        rc.drizzle.output_scale = d["output_scale"].as<int>();
      if (yaml_has_value(d["kernel"]))
        rc.drizzle.kernel = d["kernel"].as<std::string>();
      if (yaml_has_value(d["pixfrac"]))
        rc.drizzle.pixfrac = d["pixfrac"].as<float>();
      if (yaml_has_value(d["robust_passes"]))
        rc.drizzle.robust_passes = d["robust_passes"].as<int>();
      if (yaml_has_value(d["min_clip_contributors"]))
        rc.drizzle.min_clip_contributors = d["min_clip_contributors"].as<int>();
      if (yaml_has_value(d["chunk_rows"]))
        rc.drizzle.chunk_rows = d["chunk_rows"].as<int>();
      if (yaml_has_value(d["chunk_halo_rows"]))
        rc.drizzle.chunk_halo_rows = d["chunk_halo_rows"].as<int>();
      if (yaml_has_value(d["memory_budget_mb"]))
        rc.drizzle.memory_budget_mb = d["memory_budget_mb"].as<size_t>();
    }
    if (yaml_has_value(r["clipping"])) {
      auto c = r["clipping"];
      if (yaml_has_value(c["clip_sigma_low"]))
        rc.clipping.clip_sigma_low = c["clip_sigma_low"].as<float>();
      if (yaml_has_value(c["clip_sigma_high"]))
        rc.clipping.clip_sigma_high = c["clip_sigma_high"].as<float>();
      if (yaml_has_value(c["min_fraction"]))
        rc.clipping.min_fraction = c["min_fraction"].as<float>();
      if (yaml_has_value(c["min_n_eff"]))
        rc.clipping.min_n_eff = c["min_n_eff"].as<float>();
      if (yaml_has_value(c["guard_fallback"]))
        rc.clipping.guard_fallback = c["guard_fallback"].as<bool>();
      if (yaml_has_value(c["shared_frame_rejection"]))
        rc.clipping.shared_frame_rejection =
            c["shared_frame_rejection"].as<bool>();
      if (yaml_has_value(c["shared_frame_rejection_consensus"]))
        rc.clipping.shared_frame_rejection_consensus =
            c["shared_frame_rejection_consensus"].as<float>();
    }
    if (yaml_has_value(r["coverage_gate"])) {
      auto g = r["coverage_gate"];
      if (yaml_has_value(g["min_frames"]))
        rc.coverage_gate.min_frames = g["min_frames"].as<int>();
      if (yaml_has_value(g["min_supported_fraction"]))
        rc.coverage_gate.min_supported_fraction =
            g["min_supported_fraction"].as<float>();
      if (yaml_has_value(g["min_channel_n_eff_floor"]))
        rc.coverage_gate.min_channel_n_eff_floor =
            g["min_channel_n_eff_floor"].as<float>();
      if (yaml_has_value(g["min_channel_n_eff_fraction"]))
        rc.coverage_gate.min_channel_n_eff_fraction =
            g["min_channel_n_eff_fraction"].as<float>();
      if (yaml_has_value(g["min_analysis_pixels"]))
        rc.coverage_gate.min_analysis_pixels =
            g["min_analysis_pixels"].as<int>();
      if (yaml_has_value(g["max_internal_hole_area_px"]))
        rc.coverage_gate.max_internal_hole_area_px =
            g["max_internal_hole_area_px"].as<long long>();
    }
    if (yaml_has_value(r["quality"]) &&
        yaml_has_value(r["quality"]["pyramid"])) {
      auto q = r["quality"]["pyramid"];
      if (yaml_has_value(q["scales"]))
        rc.quality.pyramid.scales = q["scales"].as<int>();
      if (yaml_has_value(q["base_window_px"]))
        rc.quality.pyramid.base_window_px = q["base_window_px"].as<int>();
      if (yaml_has_value(q["sharpness_weight"]))
        rc.quality.pyramid.sharpness_weight =
            q["sharpness_weight"].as<float>();
      if (yaml_has_value(q["snr_weight"]))
        rc.quality.pyramid.snr_weight = q["snr_weight"].as<float>();
      if (yaml_has_value(q["score_scale"]))
        rc.quality.pyramid.score_scale = q["score_scale"].as<float>();
      if (yaml_has_value(q["artifact_sigma"]))
        rc.quality.pyramid.artifact_sigma = q["artifact_sigma"].as<float>();
      if (yaml_has_value(q["max_artifact_fraction"]))
        rc.quality.pyramid.max_artifact_fraction =
            q["max_artifact_fraction"].as<float>();
    }
    if (yaml_has_value(r["multiband"])) {
      auto m = r["multiband"];
      if (yaml_has_value(m["enabled"]))
        rc.multiband.enabled = m["enabled"].as<bool>();
      if (yaml_has_value(m["levels"]))
        rc.multiband.levels = m["levels"].as<int>();
      if (yaml_has_value(m["alpha_cap"]))
        rc.multiband.alpha_cap = m["alpha_cap"].as<float>();
      if (yaml_has_value(m["fine_quality_exponent"]))
        rc.multiband.fine_quality_exponent =
            m["fine_quality_exponent"].as<float>();
      if (yaml_has_value(m["medium_quality_exponent"]))
        rc.multiband.medium_quality_exponent =
            m["medium_quality_exponent"].as<float>();
      if (yaml_has_value(m["min_quality_separation"]))
        rc.multiband.min_quality_separation =
            m["min_quality_separation"].as<float>();
      if (yaml_has_value(m["full_quality_separation"]))
        rc.multiband.full_quality_separation =
            m["full_quality_separation"].as<float>();
      if (yaml_has_value(m["min_effective_samples"]))
        rc.multiband.min_effective_samples =
            m["min_effective_samples"].as<float>();
      if (yaml_has_value(m["full_effective_samples"]))
        rc.multiband.full_effective_samples =
            m["full_effective_samples"].as<float>();
    }
    if (yaml_has_value(r["multiband_validation"])) {
      auto v = r["multiband_validation"];
      if (yaml_has_value(v["fwhm_ratio_max"]))
        rc.multiband_validation.fwhm_ratio_max =
            v["fwhm_ratio_max"].as<double>();
      if (yaml_has_value(v["p90_fwhm_ratio_max"]))
        rc.multiband_validation.p90_fwhm_ratio_max =
            v["p90_fwhm_ratio_max"].as<double>();
      if (yaml_has_value(v["tail_ratio_max"]))
        rc.multiband_validation.tail_ratio_max =
            v["tail_ratio_max"].as<double>();
      if (yaml_has_value(v["elongation_ratio_max"]))
        rc.multiband_validation.elongation_ratio_max =
            v["elongation_ratio_max"].as<double>();
      if (yaml_has_value(v["background_rms_ratio_max"]))
        rc.multiband_validation.background_rms_ratio_max =
            v["background_rms_ratio_max"].as<double>();
      if (yaml_has_value(v["seam_ratio_max"]))
        rc.multiband_validation.seam_ratio_max =
            v["seam_ratio_max"].as<double>();
      if (yaml_has_value(v["min_stars_fwhm"]))
        rc.multiband_validation.min_stars_fwhm =
            v["min_stars_fwhm"].as<int>();
      if (yaml_has_value(v["min_stars_p90_tail_elongation"]))
        rc.multiband_validation.min_stars_p90_tail_elongation =
            v["min_stars_p90_tail_elongation"].as<int>();
      if (yaml_has_value(v["max_fwhm_ci_relative_width"]))
        rc.multiband_validation.max_fwhm_ci_relative_width =
            v["max_fwhm_ci_relative_width"].as<double>();
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
    // `bge.enabled` was a legacy on/off mirror of `bge.method` ("none" ==
    // disabled) that could silently disagree with `method` -- whichever was
    // written most recently by a given caller won, which meant e.g. a
    // config with enabled:false but a stale method:classic still ran BGE.
    // The field has been removed; reject it explicitly instead of quietly
    // ignoring or reinterpreting it, so an old config or preset that still
    // sets it fails loudly with an actionable message rather than behaving
    // unpredictably.
    if (yaml_has_value(b["enabled"])) {
      throw ValidationError(
          "bge.enabled is no longer supported; use bge.method: "
          "none|classic|autobge|auto instead (method is the sole on/off "
          "switch -- \"none\" disables BGE).");
    }
    if (b["method"])
      cfg.bge.method = b["method"].as<std::string>();
    if (yaml_has_value(b["auto_detect"])) {
      auto ad = b["auto_detect"];
      if (yaml_has_value(ad["gradient_threshold"]))
        cfg.bge.auto_detect.gradient_threshold =
            ad["gradient_threshold"].as<float>();
      if (yaml_has_value(ad["extended_source_sigma"]))
        cfg.bge.auto_detect.extended_source_sigma =
            ad["extended_source_sigma"].as<float>();
      if (yaml_has_value(ad["extended_source_dilate_px"]))
        cfg.bge.auto_detect.extended_source_dilate_px =
            ad["extended_source_dilate_px"].as<int>();
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
    if (yaml_has_value(h["highlight_ceiling_percentile"]))
      cfg.hypermetric_stretch.highlight_ceiling_percentile =
          h["highlight_ceiling_percentile"].as<float>();
    if (yaml_has_value(h["write_channels"]))
      cfg.hypermetric_stretch.write_channels = h["write_channels"].as<bool>();
    if (yaml_has_value(h["output_rgb"]))
      cfg.hypermetric_stretch.output_rgb = h["output_rgb"].as<std::string>();
  }

  if (node["stacking"]) {
    auto st = node["stacking"];
    if (yaml_has_value(st["per_frame_cosmetic_correction"]))
      cfg.stacking.per_frame_cosmetic_correction = st["per_frame_cosmetic_correction"].as<bool>();
    if (yaml_has_value(st["per_frame_cosmetic_correction_sigma"]))
      cfg.stacking.per_frame_cosmetic_correction_sigma = st["per_frame_cosmetic_correction_sigma"].as<float>();
  }

  if (node["runtime_limits"]) {
    auto rl = node["runtime_limits"];
    if (yaml_has_value(rl["hard_abort_hours"]))
      cfg.runtime_limits.hard_abort_hours = rl["hard_abort_hours"].as<float>();
    if (yaml_has_value(rl["parallel_workers"]))
      cfg.runtime_limits.parallel_workers = rl["parallel_workers"].as<int>();
    if (yaml_has_value(rl["memory_budget"]))
      cfg.runtime_limits.memory_budget = rl["memory_budget"].as<int>();
    if (yaml_has_value(rl["acceleration_backend"])) {
      cfg.runtime_limits.acceleration_backend =
          normalize_acceleration_backend(
              rl["acceleration_backend"].as<std::string>());
    }
  }


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
  node["registration"]["prewarp_interpolation"] =
      registration.prewarp_interpolation;
  node["registration"]["debayer_first"] = registration.debayer_first;
  node["registration"]["pre_debayer_method"] = registration.pre_debayer_method;

  node["dithering"]["enabled"] = dithering.enabled;
  node["dithering"]["min_shift_px"] = dithering.min_shift_px;

  node["chroma_denoise"]["enabled"] = chroma_denoise.enabled;
  node["chroma_denoise"]["color_space"] = chroma_denoise.color_space;
  node["chroma_denoise"]["apply_stage"] = chroma_denoise.apply_stage;
  node["chroma_denoise"]["protect_luma"] = chroma_denoise.protect_luma;
  node["chroma_denoise"]["luma_guard_strength"] = chroma_denoise.luma_guard_strength;
  node["chroma_denoise"]["adaptation_reference_ratio"] =
      chroma_denoise.adaptation_reference_ratio;
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
  node["chroma_denoise"]["extended_source_protection"]["enabled"] =
      chroma_denoise.extended_source_protection.enabled;
  node["chroma_denoise"]["extended_source_protection"]["luma_sigma"] =
      chroma_denoise.extended_source_protection.luma_sigma;
  node["chroma_denoise"]["extended_source_protection"]["dilate_px"] =
      chroma_denoise.extended_source_protection.dilate_px;
  node["chroma_denoise"]["large_scale_bias"]["enabled"] =
      chroma_denoise.large_scale_bias.enabled;
  node["chroma_denoise"]["large_scale_bias"]["block_size"] =
      chroma_denoise.large_scale_bias.block_size;
  node["chroma_denoise"]["large_scale_bias"]["blur_sigma"] =
      chroma_denoise.large_scale_bias.blur_sigma;
  node["chroma_denoise"]["large_scale_bias"]["strength"] =
      chroma_denoise.large_scale_bias.strength;

  node["luma_denoise"]["enabled"] = luma_denoise.enabled;
  node["luma_denoise"]["luma_guard_strength"] = luma_denoise.luma_guard_strength;
  node["luma_denoise"]["blend_amount"] = luma_denoise.blend_amount;
  node["luma_denoise"]["star_protection"]["enabled"] =
      luma_denoise.star_protection.enabled;
  node["luma_denoise"]["star_protection"]["threshold_sigma"] =
      luma_denoise.star_protection.threshold_sigma;
  node["luma_denoise"]["star_protection"]["dilate_px"] =
      luma_denoise.star_protection.dilate_px;
  node["luma_denoise"]["structure_protection"]["enabled"] =
      luma_denoise.structure_protection.enabled;
  node["luma_denoise"]["structure_protection"]["gradient_percentile"] =
      luma_denoise.structure_protection.gradient_percentile;
  node["luma_denoise"]["wavelet"]["enabled"] = luma_denoise.wavelet.enabled;
  node["luma_denoise"]["wavelet"]["levels"] = luma_denoise.wavelet.levels;
  node["luma_denoise"]["wavelet"]["threshold_scale"] =
      luma_denoise.wavelet.threshold_scale;
  node["luma_denoise"]["wavelet"]["soft_k"] = luma_denoise.wavelet.soft_k;

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

  node["astrometry"]["enabled"] = astrometry.enabled;
  node["astrometry"]["astap_bin"] = astrometry.astap_bin;
  node["astrometry"]["astap_data_dir"] = astrometry.astap_data_dir;
  node["astrometry"]["search_radius"] = astrometry.search_radius;

  node["bge"]["method"] = bge.method;
  node["bge"]["auto_detect"]["gradient_threshold"] =
      bge.auto_detect.gradient_threshold;
  node["bge"]["auto_detect"]["extended_source_sigma"] =
      bge.auto_detect.extended_source_sigma;
  node["bge"]["auto_detect"]["extended_source_dilate_px"] =
      bge.auto_detect.extended_source_dilate_px;
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
  node["hypermetric_stretch"]["highlight_ceiling_percentile"] =
      hypermetric_stretch.highlight_ceiling_percentile;
  node["hypermetric_stretch"]["write_channels"] =
      hypermetric_stretch.write_channels;
  node["hypermetric_stretch"]["output_rgb"] = hypermetric_stretch.output_rgb;

  // Single-method reconstruction contract (plan 6.1).
  {
    const auto &rc = reconstruction;
    auto r = node["reconstruction"];
    r["delete_source_cache_after_run"] = rc.delete_source_cache_after_run;
    r["keep_profile_cache_after_run"] = rc.keep_profile_cache_after_run;
    r["common_overlap_required_fraction"] = rc.common_overlap_required_fraction;
    r["diagnostics"]["level"] = rc.diagnostics.level;
    r["diagnostics"]["preview_forward_drizzle_uniform"] =
        rc.diagnostics.preview_forward_drizzle_uniform;
    r["diagnostics"]["persist_forward_drizzle_uniform_store"] =
        rc.diagnostics.persist_forward_drizzle_uniform_store;
    r["drizzle"]["internal_scale"] = rc.drizzle.internal_scale;
    r["drizzle"]["output_scale"] = rc.drizzle.output_scale;
    r["drizzle"]["kernel"] = rc.drizzle.kernel;
    r["drizzle"]["pixfrac"] = rc.drizzle.pixfrac;
    r["drizzle"]["robust_passes"] = rc.drizzle.robust_passes;
    r["drizzle"]["min_clip_contributors"] = rc.drizzle.min_clip_contributors;
    r["drizzle"]["chunk_rows"] = rc.drizzle.chunk_rows;
    r["drizzle"]["chunk_halo_rows"] = rc.drizzle.chunk_halo_rows;
    r["drizzle"]["memory_budget_mb"] = rc.drizzle.memory_budget_mb;
    r["clipping"]["clip_sigma_low"] = rc.clipping.clip_sigma_low;
    r["clipping"]["clip_sigma_high"] = rc.clipping.clip_sigma_high;
    r["clipping"]["min_fraction"] = rc.clipping.min_fraction;
    r["clipping"]["min_n_eff"] = rc.clipping.min_n_eff;
    r["clipping"]["guard_fallback"] = rc.clipping.guard_fallback;
    r["clipping"]["shared_frame_rejection"] = rc.clipping.shared_frame_rejection;
    r["clipping"]["shared_frame_rejection_consensus"] =
        rc.clipping.shared_frame_rejection_consensus;
    r["coverage_gate"]["min_frames"] = rc.coverage_gate.min_frames;
    r["coverage_gate"]["min_supported_fraction"] =
        rc.coverage_gate.min_supported_fraction;
    r["coverage_gate"]["min_channel_n_eff_floor"] =
        rc.coverage_gate.min_channel_n_eff_floor;
    r["coverage_gate"]["min_channel_n_eff_fraction"] =
        rc.coverage_gate.min_channel_n_eff_fraction;
    r["coverage_gate"]["min_analysis_pixels"] =
        rc.coverage_gate.min_analysis_pixels;
    r["coverage_gate"]["max_internal_hole_area_px"] =
        rc.coverage_gate.max_internal_hole_area_px;
    r["quality"]["pyramid"]["scales"] = rc.quality.pyramid.scales;
    r["quality"]["pyramid"]["base_window_px"] =
        rc.quality.pyramid.base_window_px;
    r["quality"]["pyramid"]["sharpness_weight"] =
        rc.quality.pyramid.sharpness_weight;
    r["quality"]["pyramid"]["snr_weight"] = rc.quality.pyramid.snr_weight;
    r["quality"]["pyramid"]["score_scale"] = rc.quality.pyramid.score_scale;
    r["quality"]["pyramid"]["artifact_sigma"] =
        rc.quality.pyramid.artifact_sigma;
    r["quality"]["pyramid"]["max_artifact_fraction"] =
        rc.quality.pyramid.max_artifact_fraction;
    r["multiband"]["enabled"] = rc.multiband.enabled;
    r["multiband"]["levels"] = rc.multiband.levels;
    r["multiband"]["alpha_cap"] = rc.multiband.alpha_cap;
    r["multiband"]["fine_quality_exponent"] =
        rc.multiband.fine_quality_exponent;
    r["multiband"]["medium_quality_exponent"] =
        rc.multiband.medium_quality_exponent;
    r["multiband"]["min_quality_separation"] =
        rc.multiband.min_quality_separation;
    r["multiband"]["full_quality_separation"] =
        rc.multiband.full_quality_separation;
    r["multiband"]["min_effective_samples"] =
        rc.multiband.min_effective_samples;
    r["multiband"]["full_effective_samples"] =
        rc.multiband.full_effective_samples;
    r["multiband_validation"]["fwhm_ratio_max"] =
        rc.multiband_validation.fwhm_ratio_max;
    r["multiband_validation"]["p90_fwhm_ratio_max"] =
        rc.multiband_validation.p90_fwhm_ratio_max;
    r["multiband_validation"]["tail_ratio_max"] =
        rc.multiband_validation.tail_ratio_max;
    r["multiband_validation"]["elongation_ratio_max"] =
        rc.multiband_validation.elongation_ratio_max;
    r["multiband_validation"]["background_rms_ratio_max"] =
        rc.multiband_validation.background_rms_ratio_max;
    r["multiband_validation"]["seam_ratio_max"] =
        rc.multiband_validation.seam_ratio_max;
    r["multiband_validation"]["min_stars_fwhm"] =
        rc.multiband_validation.min_stars_fwhm;
    r["multiband_validation"]["min_stars_p90_tail_elongation"] =
        rc.multiband_validation.min_stars_p90_tail_elongation;
    r["multiband_validation"]["max_fwhm_ci_relative_width"] =
        rc.multiband_validation.max_fwhm_ci_relative_width;
  }

  node["stacking"]["per_frame_cosmetic_correction"] =
      stacking.per_frame_cosmetic_correction;
  node["stacking"]["per_frame_cosmetic_correction_sigma"] =
      stacking.per_frame_cosmetic_correction_sigma;

  node["runtime_limits"]["hard_abort_hours"] = runtime_limits.hard_abort_hours;
  node["runtime_limits"]["parallel_workers"] =
      runtime_limits.parallel_workers;
  node["runtime_limits"]["memory_budget"] = runtime_limits.memory_budget;
  node["runtime_limits"]["acceleration_backend"] =
      runtime_limits.acceleration_backend;

  return node;
}

/// @brief Implements validate.
/// @details Part of YAML configuration loading, serialization, schema generation, and validation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void ReconstructionConfig::validate() const {
  auto req = [](bool cond, const std::string &msg) {
    if (!cond) throw ValidationError("reconstruction: " + msg);
  };
  req(drizzle.internal_scale == 1 || drizzle.internal_scale == 2,
      "drizzle.internal_scale must be 1 or 2");
  req(drizzle.output_scale == 1 || drizzle.output_scale == 2,
      "drizzle.output_scale must be 1 or 2");
  req(drizzle.output_scale <= drizzle.internal_scale,
      "drizzle.output_scale must be <= drizzle.internal_scale");
  req(drizzle.kernel == "square", "drizzle.kernel must be 'square' in the MVP");
  req(drizzle.pixfrac > 0.0f && drizzle.pixfrac <= 1.0f,
      "drizzle.pixfrac must be in (0, 1]");
  req(drizzle.robust_passes >= 1 && drizzle.robust_passes <= 6,
      "drizzle.robust_passes must be in [1, 6]");
  req(drizzle.min_clip_contributors >= 2,
      "drizzle.min_clip_contributors must be >= 2");
  req(drizzle.chunk_rows >= 0, "drizzle.chunk_rows must be >= 0");
  req(drizzle.chunk_halo_rows == -1 || drizzle.chunk_halo_rows >= 1,
      "drizzle.chunk_halo_rows must be -1 (auto) or >= 1");

  req(clipping.clip_sigma_low > 0.0f, "clipping.clip_sigma_low must be > 0");
  req(clipping.clip_sigma_high > 0.0f, "clipping.clip_sigma_high must be > 0");
  req(clipping.min_fraction > 0.0f && clipping.min_fraction <= 1.0f,
      "clipping.min_fraction must be in (0, 1]");
  req(clipping.min_n_eff >= 1.0f, "clipping.min_n_eff must be >= 1");
  req(clipping.shared_frame_rejection_consensus > 0.0f &&
          clipping.shared_frame_rejection_consensus <= 1.0f,
      "clipping.shared_frame_rejection_consensus must be in (0, 1]");

  req(diagnostics.level == "summary" || diagnostics.level == "full",
      "diagnostics.level must be 'summary' or 'full'");

  req(quality.pyramid.scales >= 1 && quality.pyramid.scales <= 8,
      "quality.pyramid.scales must be in [1, 8]");
  req(quality.pyramid.base_window_px >= 1,
      "quality.pyramid.base_window_px must be >= 1");
  req(quality.pyramid.sharpness_weight >= 0.0f &&
          quality.pyramid.snr_weight >= 0.0f &&
          quality.pyramid.sharpness_weight + quality.pyramid.snr_weight > 0.0f,
      "quality.pyramid.sharpness_weight and snr_weight must be non-negative "
      "with positive sum");
  req(quality.pyramid.score_scale > 0.0f,
      "quality.pyramid.score_scale must be > 0");
  req(quality.pyramid.artifact_sigma > 0.0f,
      "quality.pyramid.artifact_sigma must be > 0");
  req(quality.pyramid.max_artifact_fraction > 0.0f &&
          quality.pyramid.max_artifact_fraction <= 1.0f,
      "quality.pyramid.max_artifact_fraction must be in (0, 1]");

  req(coverage_gate.min_frames >= 2, "coverage_gate.min_frames must be >= 2");
  req(coverage_gate.min_supported_fraction > 0.0f &&
          coverage_gate.min_supported_fraction <= 1.0f,
      "coverage_gate.min_supported_fraction must be in (0, 1]");
  req(coverage_gate.min_channel_n_eff_floor >= 1.0f,
      "coverage_gate.min_channel_n_eff_floor must be >= 1");
  req(coverage_gate.min_channel_n_eff_fraction > 0.0f &&
          coverage_gate.min_channel_n_eff_fraction <= 1.0f,
      "coverage_gate.min_channel_n_eff_fraction must be in (0, 1]");
  req(coverage_gate.min_analysis_pixels >= 1,
      "coverage_gate.min_analysis_pixels must be >= 1");
  req(coverage_gate.max_internal_hole_area_px >= 0,
      "coverage_gate.max_internal_hole_area_px must be >= 0");

  req(multiband.levels >= 1 && multiband.levels <= 4,
      "multiband.levels must be in [1, 4]");
  req(multiband.levels < 2 || quality.pyramid.scales >= 2,
      "multiband.levels >= 2 requires quality.pyramid.scales >= 2");
  req(multiband.alpha_cap >= 0.0f && multiband.alpha_cap <= 1.0f,
      "multiband.alpha_cap must be in [0, 1]");
  req(multiband.fine_quality_exponent >= 0.0f,
      "multiband.fine_quality_exponent must be >= 0");
  req(multiband.medium_quality_exponent >= 0.0f,
      "multiband.medium_quality_exponent must be >= 0");
  req(multiband.min_quality_separation >= 0.0f &&
          multiband.min_quality_separation < multiband.full_quality_separation &&
          multiband.full_quality_separation <= 1.0f,
      "multiband quality separation must satisfy "
      "0 <= min < full <= 1");
  req(multiband.min_effective_samples >= 1.0f &&
          multiband.min_effective_samples < multiband.full_effective_samples,
      "multiband effective samples must satisfy 1 <= min < full");

  req(multiband_validation.fwhm_ratio_max > 0.0,
      "multiband_validation.fwhm_ratio_max must be > 0");
  req(multiband_validation.p90_fwhm_ratio_max > 0.0,
      "multiband_validation.p90_fwhm_ratio_max must be > 0");
  req(multiband_validation.tail_ratio_max > 0.0,
      "multiband_validation.tail_ratio_max must be > 0");
  req(multiband_validation.elongation_ratio_max > 0.0,
      "multiband_validation.elongation_ratio_max must be > 0");
  req(multiband_validation.background_rms_ratio_max > 0.0,
      "multiband_validation.background_rms_ratio_max must be > 0");
  req(multiband_validation.seam_ratio_max > 0.0,
      "multiband_validation.seam_ratio_max must be > 0");
  req(multiband_validation.min_stars_fwhm >= 0,
      "multiband_validation.min_stars_fwhm must be >= 0");
  req(multiband_validation.min_stars_p90_tail_elongation >= 0,
      "multiband_validation.min_stars_p90_tail_elongation must be >= 0");
  req(multiband_validation.max_fwhm_ci_relative_width > 0.0,
      "multiband_validation.max_fwhm_ci_relative_width must be > 0");

  req(common_overlap_required_fraction >= 0.0f &&
          common_overlap_required_fraction <= 1.0f,
      "common_overlap_required_fraction must be in [0, 1]");
}

void Config::validate() const {

  reconstruction.validate();

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
  if (registration.prewarp_interpolation != "linear" &&
      registration.prewarp_interpolation != "cubic" &&
      registration.prewarp_interpolation != "lanczos4") {
    throw ValidationError(
        "registration.prewarp_interpolation must be linear, cubic, or lanczos4");
  }
  if (registration.pre_debayer_method != "bilinear" &&
      registration.pre_debayer_method != "nearest" &&
      registration.pre_debayer_method != "vng" &&
      registration.pre_debayer_method != "edge_aware") {
    throw ValidationError(
        "registration.pre_debayer_method must be bilinear, nearest, vng, or edge_aware");
  }

  if (dithering.min_shift_px < 0.0f) {
    throw ValidationError("dithering.min_shift_px must be >= 0");
  }

  if (chroma_denoise.color_space != "ycbcr_linear" &&
      chroma_denoise.color_space != "opponent_linear") {
    throw ValidationError(
        "chroma_denoise.color_space must be 'ycbcr_linear' or 'opponent_linear'");
  }
  if (chroma_denoise.apply_stage != "pre_stack_tiles" &&
      chroma_denoise.apply_stage != "post_stack_linear" &&
      chroma_denoise.apply_stage != "post_pcc" &&
      chroma_denoise.apply_stage != "both") {
    throw ValidationError(
        "chroma_denoise.apply_stage must be 'pre_stack_tiles', 'post_stack_linear', 'post_pcc' or 'both'");
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
  if (chroma_denoise.extended_source_protection.luma_sigma < 1.0f ||
      chroma_denoise.extended_source_protection.luma_sigma > 5.0f) {
    throw ValidationError(
        "chroma_denoise.extended_source_protection.luma_sigma must be in [1,5]");
  }
  if (chroma_denoise.extended_source_protection.dilate_px < 0 ||
      chroma_denoise.extended_source_protection.dilate_px > 100) {
    throw ValidationError(
        "chroma_denoise.extended_source_protection.dilate_px must be in [0,100]");
  }
  if (chroma_denoise.adaptation_reference_ratio <= 0.0f) {
    throw ValidationError(
        "chroma_denoise.adaptation_reference_ratio must be > 0");
  }
  if (chroma_denoise.large_scale_bias.block_size < 4) {
    throw ValidationError(
        "chroma_denoise.large_scale_bias.block_size must be >= 4");
  }
  if (chroma_denoise.large_scale_bias.blur_sigma < 0.0f) {
    throw ValidationError(
        "chroma_denoise.large_scale_bias.blur_sigma must be >= 0");
  }
  if (!is_between_0_1(chroma_denoise.large_scale_bias.strength)) {
    throw ValidationError(
        "chroma_denoise.large_scale_bias.strength must be in [0,1]");
  }

  if (!is_between_0_1(luma_denoise.luma_guard_strength)) {
    throw ValidationError("luma_denoise.luma_guard_strength must be in [0,1]");
  }
  if (!is_between_0_1(luma_denoise.blend_amount)) {
    throw ValidationError("luma_denoise.blend_amount must be in [0,1]");
  }
  if (luma_denoise.star_protection.threshold_sigma <= 0.0f) {
    throw ValidationError(
        "luma_denoise.star_protection.threshold_sigma must be > 0");
  }
  if (luma_denoise.star_protection.dilate_px < 0) {
    throw ValidationError("luma_denoise.star_protection.dilate_px must be >= 0");
  }
  if (luma_denoise.structure_protection.gradient_percentile < 0.0f ||
      luma_denoise.structure_protection.gradient_percentile > 100.0f) {
    throw ValidationError(
        "luma_denoise.structure_protection.gradient_percentile must be in [0,100]");
  }
  if (luma_denoise.wavelet.levels < 1) {
    throw ValidationError("luma_denoise.wavelet.levels must be >= 1");
  }
  if (luma_denoise.wavelet.threshold_scale <= 0.0f) {
    throw ValidationError("luma_denoise.wavelet.threshold_scale must be > 0");
  }
  if (luma_denoise.wavelet.soft_k <= 0.0f) {
    throw ValidationError("luma_denoise.wavelet.soft_k must be > 0");
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

  if (bge.method != "none" && bge.method != "classic" &&
      bge.method != "autobge" && bge.method != "auto") {
    throw ValidationError("bge.method must be one of: none|classic|autobge|auto");
  }
  if (bge.auto_detect.gradient_threshold < 0.01f ||
      bge.auto_detect.gradient_threshold > 0.5f) {
    throw ValidationError(
        "bge.auto_detect.gradient_threshold must be in [0.01,0.5]");
  }
  if (bge.auto_detect.extended_source_sigma < 1.0f ||
      bge.auto_detect.extended_source_sigma > 6.0f) {
    throw ValidationError(
        "bge.auto_detect.extended_source_sigma must be in [1,6]");
  }
  if (bge.auto_detect.extended_source_dilate_px < 0 ||
      bge.auto_detect.extended_source_dilate_px > 200) {
    throw ValidationError(
        "bge.auto_detect.extended_source_dilate_px must be in [0,200]");
  }
  if (bge.method == "autobge" || bge.method == "auto") {
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
  if (hypermetric_stretch.highlight_ceiling_percentile < 90.0f ||
      hypermetric_stretch.highlight_ceiling_percentile > 100.0f) {
    throw ValidationError(
        "hypermetric_stretch.highlight_ceiling_percentile must be in "
        "[90,100]");
  }
  if (hypermetric_stretch.output_rgb.empty()) {
    throw ValidationError("hypermetric_stretch.output_rgb must not be empty");
  }

  if (stacking.per_frame_cosmetic_correction_sigma <= 0.0f) {
    throw ValidationError(
        "stacking.per_frame_cosmetic_correction_sigma must be > 0");
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
                      "star_shift_radius_px":{"type":"number","minimum":10,"maximum":2000},
                      "prewarp_interpolation":{"type":"string","enum":["linear","cubic","lanczos4"],"default":"cubic"},
                      "debayer_first":{"type":"boolean","default":true},
                      "pre_debayer_method":{"type":"string","enum":["bilinear","nearest","vng","edge_aware"],"default":"edge_aware"} } },
    "dithering": { "type":"object",
      "properties": { "enabled":{"type":"boolean"},
                      "min_shift_px":{"type":"number","minimum":0} } },
    "chroma_denoise": { "type":"object",
      "properties": { "enabled":{"type":"boolean"},
                      "color_space":{"type":"string","enum":["ycbcr_linear","opponent_linear"]},
                      "apply_stage":{"type":"string","enum":["pre_stack_tiles","post_stack_linear","post_pcc","both"]},
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
                        "amount":{"type":"number","minimum":0,"maximum":1}}},
                      "extended_source_protection":{"type":"object","properties":{
                        "enabled":{"type":"boolean"},
                        "luma_sigma":{"type":"number","minimum":1.0,"maximum":5.0},
                        "dilate_px":{"type":"integer","minimum":0,"maximum":100}}} } },
    "luma_denoise": { "type":"object",
      "description": "Luminance noise reduction on the post-stack linear RGB, before BGE/PCC/HMS. Off by default (opt-in): the CFA-forward-drizzle pipeline has no luma denoise stage of its own; chroma_denoise only ever touches chroma. A multi-level wavelet soft-threshold denoise on the derived luma, reconstructed by adding the same smooth per-pixel brightness delta to R, G and B, which preserves every channel DIFFERENCE (chroma) exactly while only brightness moves -- avoiding the noise amplification a per-channel RATIO reconstruction causes in faint or partially-protected regions.",
      "properties": { "enabled":{"type":"boolean","default":false},
                      "luma_guard_strength":{"type":"number","minimum":0,"maximum":1},
                      "blend_amount":{"type":"number","minimum":0,"maximum":1},
                      "star_protection":{"type":"object","properties":{
                        "enabled":{"type":"boolean"},
                        "threshold_sigma":{"type":"number","exclusiveMinimum":0},
                        "dilate_px":{"type":"integer","minimum":0}}},
                      "structure_protection":{"type":"object","properties":{
                        "enabled":{"type":"boolean"},
                        "gradient_percentile":{"type":"number","minimum":0,"maximum":100}}},
                      "wavelet":{"type":"object","properties":{
                        "enabled":{"type":"boolean"},
                        "levels":{"type":"integer","minimum":1},
                        "threshold_scale":{"type":"number","exclusiveMinimum":0},
                        "soft_k":{"type":"number","exclusiveMinimum":0}}} } },
    "global_metrics": { "type":"object",
      "properties": { "adaptive_weights":{"type":"boolean"},
                      "weight_exponent_scale":{"type":"number","exclusiveMinimum":0,"description":"Exponent scale k for G_f = exp(k * Q_f). k=1.0 (default) is standard, k>1 increases differentiation between good/bad frames."},
                      "weights":{"type":"object","properties":{"background":{"type":"number","minimum":0,"maximum":1},"noise":{"type":"number","minimum":0,"maximum":1},"gradient":{"type":"number","minimum":0,"maximum":1},"fwhm":{"type":"number","minimum":0,"maximum":1},"roundness":{"type":"number","minimum":0,"maximum":1},"star_count":{"type":"number","minimum":0,"maximum":1}}},
                      "clamp":{"type":"array","items":{"type":"number"},"minItems":2,"maxItems":2} } },
    "reconstruction": { "type":"object", "description":"Single-method CFA-forward-drizzle + multiband reconstruction contract (plan sections 6.1-6.3). Parsed and validated in M0; consumed from M2 onward.",
      "properties": {
        "delete_source_cache_after_run":{"type":"boolean","default":false,"description":"Delete the normalized CFA source cache and source quality maps after a fully committed final image. Default false. Setting it true frees disk immediately but DISABLES resume-reconstruction for this run (a later resume would have to re-normalize every frame); the run report announces this as resume_reconstruction_disabled."},
        "keep_profile_cache_after_run":{"type":"boolean","default":false,"description":"Keep the internal transactional U/R/F/M drizzle profile store after a committed final image. Default false (deleted: it is a reconstruction cache, never a downstream-resume predecessor). Setting it true keeps it as a hashed cache so a re-fuse can skip the forward-drizzle pass, at the cost of disk."},
        "common_overlap_required_fraction":{"type":"number","minimum":0,"maximum":1,"default":1.0,"description":"Fraction of accepted dense frame footprints defining the analysis region, independent of sparse CFA support; (0,1], default 1."},
        "diagnostics":{"type":"object","properties":{"level":{"type":"string","enum":["summary","full"],"default":"summary"},"preview_forward_drizzle_uniform":{"type":"boolean","default":false,"description":"Opt-in bounded stripe Uniform diagnostic; default false. Writes summary statistics only. No complete reconstruction phase, clipping, multiband or resume contract."},"persist_forward_drizzle_uniform_store":{"type":"boolean","default":false,"description":"Opt-in, independent of preview. Streams unclipped Uniform planes into immutable generations under artifacts/forward_drizzle_uniform_store/. current.json atomically commits the complete generation after checksum and FITS validation. Includes 8 MiB IO reserve plus one float row in the drizzle budget; old generations are retained. Diagnostic only, not a pipeline resume entry."}}},
        "drizzle":{"type":"object","properties":{"internal_scale":{"type":"integer","enum":[1,2],"default":2},"output_scale":{"type":"integer","enum":[1,2],"default":1},"kernel":{"type":"string","enum":["square"],"default":"square"},"pixfrac":{"type":"number","exclusiveMinimum":0,"maximum":1,"default":0.8},"robust_passes":{"type":"integer","minimum":1,"maximum":6,"default":2},"min_clip_contributors":{"type":"integer","minimum":2,"default":5},"chunk_rows":{"type":"integer","minimum":0,"default":0,"description":"Target stripe rows; 0 selects at most 256 rows within the memory budget. Explicit values exceeding the budget fail closed."},"chunk_halo_rows":{"type":"integer","minimum":-1,"default":-1,"description":"Compatibility field (-1 auto, >=0 explicit). Exact source-footprint enumeration completes each target stripe without an output halo; the CPU Uniform/coverage path does not duplicate halo rows."},"memory_budget_mb":{"type":"integer","minimum":0,"default":0,"description":"Phase allocation budget in MiB, including retained masks/output, one source plus transient load copy and stripe scratch. 0 inherits runtime_limits.memory_budget (library default 512 MiB). Available host/cgroup headroom may reduce it. Fails before allocation if one row does not fit."}}},
        "clipping":{"type":"object","properties":{"clip_sigma_low":{"type":"number","exclusiveMinimum":0,"default":3.0},"clip_sigma_high":{"type":"number","exclusiveMinimum":0,"default":3.0},"min_fraction":{"type":"number","exclusiveMinimum":0,"maximum":1,"default":0.4},"min_n_eff":{"type":"number","minimum":1,"default":3.0},"guard_fallback":{"type":"boolean","default":false,"description":"When the min_fraction/min_n_eff veto (plan 11.8 step 8) fails, use the sigma-clip survivors (or, if none survived, every original candidate unclipped) instead of leaving the pixel/channel unsupported. Default false preserves the exact reviewed 8-step procedure."},"shared_frame_rejection":{"type":"boolean","default":false,"description":"Off by default. The CFA forward-drizzle kernel clips per (pixel, color channel) independently, since R/G/B come from different native Bayer sub-pixel positions; in faint regions this makes the surviving frame set an effectively independent coin-flip per channel, producing anti-correlated per-pixel color noise. When true, a frame is rejected for ALL channels at a pixel only if shared_frame_rejection_consensus or more of the channels that had a candidate from it flag it as an outlier (per-channel values/weights/geometry are untouched). CPU-only: forces CPU execution even with acceleration_backend=cuda."},"shared_frame_rejection_consensus":{"type":"number","exclusiveMinimum":0,"maximum":1,"default":0.5,"description":"Fraction of channels that must flag a frame as an outlier before shared_frame_rejection drops it for all of them. 0.5 = majority."}}},
        "coverage_gate":{"type":"object","properties":{"min_frames":{"type":"integer","minimum":2,"default":2},"min_supported_fraction":{"type":"number","exclusiveMinimum":0,"maximum":1,"default":0.995},"min_channel_n_eff_floor":{"type":"number","minimum":1,"default":3.0},"min_channel_n_eff_fraction":{"type":"number","exclusiveMinimum":0,"maximum":1,"default":0.15},"min_analysis_pixels":{"type":"integer","minimum":1,"default":1024},"max_internal_hole_area_px":{"type":"integer","minimum":0,"default":0}}},
        "quality":{"type":"object","properties":{"pyramid":{"type":"object","properties":{"scales":{"type":"integer","minimum":1,"maximum":8,"default":4,"description":"Number of Source Quality Map pyramid levels."},"base_window_px":{"type":"integer","minimum":1,"default":4,"description":"Base local window size in pixels for Source Quality Map statistics."},"sharpness_weight":{"type":"number","minimum":0,"default":0.6,"description":"Weight of the local sharpness term in the Source Quality Map score; sharpness_weight+snr_weight must be > 0."},"snr_weight":{"type":"number","minimum":0,"default":0.4,"description":"Weight of the local signal-to-noise term in the Source Quality Map score."},"score_scale":{"type":"number","exclusiveMinimum":0,"default":1.8,"description":"Scaling factor applied to the combined Source Quality Map score."},"artifact_sigma":{"type":"number","exclusiveMinimum":0,"default":3.0,"description":"Sigma threshold above which a Source Quality Map region counts as artifact."},"max_artifact_fraction":{"type":"number","exclusiveMinimum":0,"maximum":1,"default":0.25,"description":"Maximum tolerated artifact fraction in the Source Quality Map."}}}}},
        "multiband":{"type":"object","properties":{"enabled":{"type":"boolean","default":true},"levels":{"type":"integer","minimum":1,"maximum":4,"default":3},"alpha_cap":{"type":"number","minimum":0,"maximum":1,"default":1.0},"fine_quality_exponent":{"type":"number","minimum":0,"default":4.0},"medium_quality_exponent":{"type":"number","minimum":0,"default":2.0},"min_quality_separation":{"type":"number","minimum":0,"maximum":1,"default":0.05},"full_quality_separation":{"type":"number","minimum":0,"maximum":1,"default":0.20},"min_effective_samples":{"type":"number","minimum":1,"default":8.0},"full_effective_samples":{"type":"number","minimum":1,"default":24.0}}} } },
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
                      "highlight_ceiling_percentile":{"type":"number","minimum":90,"maximum":100},
                      "write_channels":{"type":"boolean"},
                      "output_rgb":{"type":"string"} } },
    "stacking": { "type":"object",
      "properties": { "per_frame_cosmetic_correction":{"type":"boolean"},
                      "per_frame_cosmetic_correction_sigma":{"type":"number","exclusiveMinimum":0} } },
    "runtime_limits": { "type":"object",
      "properties": { "hard_abort_hours":{"type":"number","exclusiveMinimum":0},
                      "parallel_workers":{"type":"integer","minimum":1},
                      "memory_budget":{"type":"integer","minimum":1},
                      "acceleration_backend":{"type":"string","enum":["auto","cpu","opencv_cuda","opencv_opencl","opencl","cuda"],"description":"Acceleration backend for PREWARP and FORWARD_DRIZZLE."} } }
  }
})";
}

} // namespace tile_compile::config
