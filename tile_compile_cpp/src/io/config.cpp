#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/errors.hpp"

#include <cmath>
#include <fstream>

namespace tile_compile::config {

namespace {

bool is_between_0_1(float v) { return v >= 0.0f && v <= 1.0f; }

void read_float_pair(const YAML::Node &n, std::array<float, 2> &out) {
  if (n && n.IsSequence() && n.size() == 2) {
    out[0] = n[0].as<float>();
    out[1] = n[1].as<float>();
  }
}

void read_int_pair(const YAML::Node &n, std::array<int, 2> &out) {
  if (n && n.IsSequence() && n.size() == 2) {
    out[0] = n[0].as<int>();
    out[1] = n[1].as<int>();
  }
}

} // namespace

Config Config::load(const fs::path &path) {
  if (!fs::exists(path)) {
    throw ConfigError("Config file not found: " + path.string());
  }
  YAML::Node node = YAML::LoadFile(path.string());
  return from_yaml(node);
}

Config Config::from_yaml(const YAML::Node &node) {
  Config cfg;

  if (node["pipeline"]) {
    auto p = node["pipeline"];
    if (p["mode"])
      cfg.pipeline.mode = p["mode"].as<std::string>();
    if (p["abort_on_fail"])
      cfg.pipeline.abort_on_fail = p["abort_on_fail"].as<bool>();
  }

  if (node["output"]) {
    auto o = node["output"];
    if (o["registered_dir"])
      cfg.output.registered_dir = o["registered_dir"].as<std::string>();
    if (o["artifacts_dir"])
      cfg.output.artifacts_dir = o["artifacts_dir"].as<std::string>();
    if (o["write_registered_frames"])
      cfg.output.write_registered_frames =
          o["write_registered_frames"].as<bool>();
    if (o["write_global_metrics"])
      cfg.output.write_global_metrics = o["write_global_metrics"].as<bool>();
    if (o["write_global_registration"])
      cfg.output.write_global_registration =
          o["write_global_registration"].as<bool>();
  }

  if (node["data"]) {
    auto d = node["data"];
    if (d["image_width"])
      cfg.data.image_width = d["image_width"].as<int>();
    if (d["image_height"])
      cfg.data.image_height = d["image_height"].as<int>();
    if (d["frames_min"])
      cfg.data.frames_min = d["frames_min"].as<int>();
    if (d["frames_target"])
      cfg.data.frames_target = d["frames_target"].as<int>();
    if (d["color_mode"])
      cfg.data.color_mode = d["color_mode"].as<std::string>();
    if (d["bayer_pattern"])
      cfg.data.bayer_pattern = d["bayer_pattern"].as<std::string>();
    if (d["linear_required"])
      cfg.data.linear_required = d["linear_required"].as<bool>();
  }

  if (node["linearity"]) {
    auto l = node["linearity"];
    if (l["enabled"])
      cfg.linearity.enabled = l["enabled"].as<bool>();
    if (l["max_frames"])
      cfg.linearity.max_frames = l["max_frames"].as<int>();
    if (l["min_overall_linearity"])
      cfg.linearity.min_overall_linearity =
          l["min_overall_linearity"].as<float>();
    if (l["strictness"])
      cfg.linearity.strictness = l["strictness"].as<std::string>();
  }

  if (node["calibration"]) {
    auto c = node["calibration"];
    if (c["use_bias"])
      cfg.calibration.use_bias = c["use_bias"].as<bool>();
    if (c["use_dark"])
      cfg.calibration.use_dark = c["use_dark"].as<bool>();
    if (c["use_flat"])
      cfg.calibration.use_flat = c["use_flat"].as<bool>();
    if (c["bias_use_master"])
      cfg.calibration.bias_use_master = c["bias_use_master"].as<bool>();
    if (c["dark_use_master"])
      cfg.calibration.dark_use_master = c["dark_use_master"].as<bool>();
    if (c["flat_use_master"])
      cfg.calibration.flat_use_master = c["flat_use_master"].as<bool>();
    if (c["dark_auto_select"])
      cfg.calibration.dark_auto_select = c["dark_auto_select"].as<bool>();
    if (c["dark_match_exposure_tolerance_percent"]) {
      cfg.calibration.dark_match_exposure_tolerance_percent =
          c["dark_match_exposure_tolerance_percent"].as<float>();
    }
    if (c["dark_match_use_temp"])
      cfg.calibration.dark_match_use_temp = c["dark_match_use_temp"].as<bool>();
    if (c["dark_match_temp_tolerance_c"])
      cfg.calibration.dark_match_temp_tolerance_c =
          c["dark_match_temp_tolerance_c"].as<float>();
    if (c["bias_dir"])
      cfg.calibration.bias_dir = c["bias_dir"].as<std::string>();
    if (c["darks_dir"])
      cfg.calibration.darks_dir = c["darks_dir"].as<std::string>();
    if (c["flats_dir"])
      cfg.calibration.flats_dir = c["flats_dir"].as<std::string>();
    if (c["bias_master"])
      cfg.calibration.bias_master = c["bias_master"].as<std::string>();
    if (c["dark_master"])
      cfg.calibration.dark_master = c["dark_master"].as<std::string>();
    if (c["flat_master"])
      cfg.calibration.flat_master = c["flat_master"].as<std::string>();
    if (c["pattern"])
      cfg.calibration.pattern = c["pattern"].as<std::string>();
  }

  if (node["assumptions"]) {
    auto a = node["assumptions"];
    if (a["frames_min"])
      cfg.assumptions.frames_min = a["frames_min"].as<int>();
    if (a["frames_optimal"])
      cfg.assumptions.frames_optimal = a["frames_optimal"].as<int>();
    if (a["frames_reduced_threshold"])
      cfg.assumptions.frames_reduced_threshold =
          a["frames_reduced_threshold"].as<int>();
    if (a["exposure_time_tolerance_percent"]) {
      cfg.assumptions.exposure_time_tolerance_percent =
          a["exposure_time_tolerance_percent"].as<float>();
    }
    if (a["reduced_mode_skip_clustering"]) {
      cfg.assumptions.reduced_mode_skip_clustering =
          a["reduced_mode_skip_clustering"].as<bool>();
    }
    read_int_pair(a["reduced_mode_cluster_range"],
                  cfg.assumptions.reduced_mode_cluster_range);
  }

  if (node["normalization"]) {
    auto n = node["normalization"];
    if (n["enabled"])
      cfg.normalization.enabled = n["enabled"].as<bool>();
    if (n["mode"])
      cfg.normalization.mode = n["mode"].as<std::string>();
    if (n["per_channel"])
      cfg.normalization.per_channel = n["per_channel"].as<bool>();
  }

  if (node["registration"]) {
    auto r = node["registration"];
    if (r["engine"])
      cfg.registration.engine = r["engine"].as<std::string>();
    if (r["allow_rotation"])
      cfg.registration.allow_rotation = r["allow_rotation"].as<bool>();
    if (r["star_topk"])
      cfg.registration.star_topk = r["star_topk"].as<int>();
    if (r["star_min_inliers"])
      cfg.registration.star_min_inliers = r["star_min_inliers"].as<int>();
    if (r["star_inlier_tol_px"])
      cfg.registration.star_inlier_tol_px = r["star_inlier_tol_px"].as<float>();
    if (r["star_dist_bin_px"])
      cfg.registration.star_dist_bin_px = r["star_dist_bin_px"].as<float>();
  }

  if (node["tile_denoise"]) {
    auto td = node["tile_denoise"];
    if (td["soft_threshold"]) {
      auto st = td["soft_threshold"];
      if (st["enabled"])
        cfg.tile_denoise.soft_threshold.enabled = st["enabled"].as<bool>();
      if (st["blur_kernel"])
        cfg.tile_denoise.soft_threshold.blur_kernel = st["blur_kernel"].as<int>();
      if (st["alpha"])
        cfg.tile_denoise.soft_threshold.alpha = st["alpha"].as<float>();
      if (st["skip_star_tiles"])
        cfg.tile_denoise.soft_threshold.skip_star_tiles = st["skip_star_tiles"].as<bool>();
    }
    if (td["wiener"]) {
      auto w = td["wiener"];
      if (w["enabled"])
        cfg.tile_denoise.wiener.enabled = w["enabled"].as<bool>();
      if (w["snr_threshold"])
        cfg.tile_denoise.wiener.snr_threshold = w["snr_threshold"].as<float>();
      if (w["q_min"])
        cfg.tile_denoise.wiener.q_min = w["q_min"].as<float>();
      if (w["q_max"])
        cfg.tile_denoise.wiener.q_max = w["q_max"].as<float>();
      if (w["q_step"])
        cfg.tile_denoise.wiener.q_step = w["q_step"].as<float>();
      if (w["min_snr"])
        cfg.tile_denoise.wiener.min_snr = w["min_snr"].as<float>();
      if (w["max_iterations"])
        cfg.tile_denoise.wiener.max_iterations = w["max_iterations"].as<int>();
    }
  }

  // Legacy: parse old "wiener_denoise" key into tile_denoise.wiener
  if (node["wiener_denoise"]) {
    auto w = node["wiener_denoise"];
    if (w["enabled"])
      cfg.tile_denoise.wiener.enabled = w["enabled"].as<bool>();
    if (w["snr_threshold"])
      cfg.tile_denoise.wiener.snr_threshold = w["snr_threshold"].as<float>();
    if (w["q_min"])
      cfg.tile_denoise.wiener.q_min = w["q_min"].as<float>();
    if (w["q_max"])
      cfg.tile_denoise.wiener.q_max = w["q_max"].as<float>();
    if (w["q_step"])
      cfg.tile_denoise.wiener.q_step = w["q_step"].as<float>();
    if (w["min_snr"])
      cfg.tile_denoise.wiener.min_snr = w["min_snr"].as<float>();
    if (w["max_iterations"])
      cfg.tile_denoise.wiener.max_iterations = w["max_iterations"].as<int>();
  }
  // Sync legacy alias
  cfg.wiener_denoise = cfg.tile_denoise.wiener;

  if (node["global_metrics"]) {
    auto gm = node["global_metrics"];
    if (gm["adaptive_weights"])
      cfg.global_metrics.adaptive_weights = gm["adaptive_weights"].as<bool>();
    if (gm["weights"]) {
      auto w = gm["weights"];
      if (w["background"])
        cfg.global_metrics.weights.background = w["background"].as<float>();
      if (w["noise"])
        cfg.global_metrics.weights.noise = w["noise"].as<float>();
      if (w["gradient"])
        cfg.global_metrics.weights.gradient = w["gradient"].as<float>();
    }
    read_float_pair(gm["clamp"], cfg.global_metrics.clamp);
    if (gm["weight_exponent_scale"])
      cfg.global_metrics.weight_exponent_scale = gm["weight_exponent_scale"].as<float>();
  }

  if (node["tile"]) {
    auto t = node["tile"];
    if (t["size_factor"])
      cfg.tile.size_factor = t["size_factor"].as<int>();
    if (t["min_size"])
      cfg.tile.min_size = t["min_size"].as<int>();
    if (t["max_divisor"])
      cfg.tile.max_divisor = t["max_divisor"].as<int>();
    if (t["overlap_fraction"])
      cfg.tile.overlap_fraction = t["overlap_fraction"].as<float>();
    if (t["star_min_count"])
      cfg.tile.star_min_count = t["star_min_count"].as<int>();
  }

  if (node["local_metrics"]) {
    auto lm = node["local_metrics"];
    read_float_pair(lm["clamp"], cfg.local_metrics.clamp);
    if (lm["star_mode"] && lm["star_mode"]["weights"]) {
      auto w = lm["star_mode"]["weights"];
      if (w["fwhm"])
        cfg.local_metrics.star_mode.weights.fwhm = w["fwhm"].as<float>();
      if (w["roundness"])
        cfg.local_metrics.star_mode.weights.roundness =
            w["roundness"].as<float>();
      if (w["contrast"])
        cfg.local_metrics.star_mode.weights.contrast =
            w["contrast"].as<float>();
    }
    if (lm["structure_mode"]) {
      auto sm = lm["structure_mode"];
      if (sm["background_weight"])
        cfg.local_metrics.structure_mode.background_weight =
            sm["background_weight"].as<float>();
      if (sm["metric_weight"])
        cfg.local_metrics.structure_mode.metric_weight =
            sm["metric_weight"].as<float>();
    }
  }

  if (node["synthetic"]) {
    auto s = node["synthetic"];
    if (s["weighting"])
      cfg.synthetic.weighting = s["weighting"].as<std::string>();
    if (s["frames_min"])
      cfg.synthetic.frames_min = s["frames_min"].as<int>();
    if (s["frames_max"])
      cfg.synthetic.frames_max = s["frames_max"].as<int>();
    if (s["clustering"]) {
      auto cl = s["clustering"];
      if (cl["mode"])
        cfg.synthetic.clustering.mode = cl["mode"].as<std::string>();
      read_int_pair(cl["cluster_count_range"],
                    cfg.synthetic.clustering.cluster_count_range);
    }
  }

  if (node["reconstruction"]) {
    auto r = node["reconstruction"];
    if (r["weighting_function"])
      cfg.reconstruction.weighting_function =
          r["weighting_function"].as<std::string>();
    if (r["window_function"])
      cfg.reconstruction.window_function =
          r["window_function"].as<std::string>();
  }

  if (node["debayer"])
    cfg.debayer = node["debayer"].as<bool>();

  if (node["astrometry"]) {
    auto a = node["astrometry"];
    if (a["enabled"])
      cfg.astrometry.enabled = a["enabled"].as<bool>();
    if (a["astap_bin"])
      cfg.astrometry.astap_bin = a["astap_bin"].as<std::string>();
    if (a["astap_data_dir"])
      cfg.astrometry.astap_data_dir = a["astap_data_dir"].as<std::string>();
    if (a["search_radius"])
      cfg.astrometry.search_radius = a["search_radius"].as<int>();
  }

  if (node["pcc"]) {
    auto p = node["pcc"];
    if (p["enabled"])
      cfg.pcc.enabled = p["enabled"].as<bool>();
    if (p["source"])
      cfg.pcc.source = p["source"].as<std::string>();
    if (p["mag_limit"])
      cfg.pcc.mag_limit = p["mag_limit"].as<float>();
    if (p["mag_bright_limit"])
      cfg.pcc.mag_bright_limit = p["mag_bright_limit"].as<float>();
    if (p["aperture_radius_px"])
      cfg.pcc.aperture_radius_px = p["aperture_radius_px"].as<float>();
    if (p["annulus_inner_px"])
      cfg.pcc.annulus_inner_px = p["annulus_inner_px"].as<float>();
    if (p["annulus_outer_px"])
      cfg.pcc.annulus_outer_px = p["annulus_outer_px"].as<float>();
    if (p["min_stars"])
      cfg.pcc.min_stars = p["min_stars"].as<int>();
    if (p["sigma_clip"])
      cfg.pcc.sigma_clip = p["sigma_clip"].as<float>();
    if (p["siril_catalog_dir"])
      cfg.pcc.siril_catalog_dir = p["siril_catalog_dir"].as<std::string>();
  }

  if (node["stacking"]) {
    auto st = node["stacking"];
    if (st["method"])
      cfg.stacking.method = st["method"].as<std::string>();
    if (st["sigma_clip"]) {
      auto sc = st["sigma_clip"];
      if (sc["sigma_low"])
        cfg.stacking.sigma_clip.sigma_low = sc["sigma_low"].as<float>();
      if (sc["sigma_high"])
        cfg.stacking.sigma_clip.sigma_high = sc["sigma_high"].as<float>();
      if (sc["max_iters"])
        cfg.stacking.sigma_clip.max_iters = sc["max_iters"].as<int>();
      if (sc["min_fraction"])
        cfg.stacking.sigma_clip.min_fraction = sc["min_fraction"].as<float>();
    }
    if (st["output_stretch"])
      cfg.stacking.output_stretch = st["output_stretch"].as<bool>();
    if (st["cosmetic_correction"])
      cfg.stacking.cosmetic_correction = st["cosmetic_correction"].as<bool>();
  }

  if (node["validation"]) {
    auto v = node["validation"];
    if (v["min_fwhm_improvement_percent"]) {
      cfg.validation.min_fwhm_improvement_percent =
          v["min_fwhm_improvement_percent"].as<float>();
    }
    if (v["max_background_rms_increase_percent"]) {
      cfg.validation.max_background_rms_increase_percent =
          v["max_background_rms_increase_percent"].as<float>();
    }
    if (v["min_tile_weight_variance"])
      cfg.validation.min_tile_weight_variance =
          v["min_tile_weight_variance"].as<float>();
    if (v["require_no_tile_pattern"])
      cfg.validation.require_no_tile_pattern =
          v["require_no_tile_pattern"].as<bool>();
  }

  if (node["runtime_limits"]) {
    auto rl = node["runtime_limits"];
    if (rl["tile_analysis_max_factor_vs_stack"]) {
      cfg.runtime_limits.tile_analysis_max_factor_vs_stack =
          rl["tile_analysis_max_factor_vs_stack"].as<float>();
    }
    if (rl["hard_abort_hours"])
      cfg.runtime_limits.hard_abort_hours = rl["hard_abort_hours"].as<float>();
    if (rl["allow_emergency_mode"])
      cfg.runtime_limits.allow_emergency_mode =
          rl["allow_emergency_mode"].as<bool>();
  }

  return cfg;
}

void Config::save(const fs::path &path) const {
  YAML::Node node = to_yaml();
  std::ofstream out(path);
  if (!out) {
    throw ConfigError("Cannot write config file: " + path.string());
  }
  out << node;
}

YAML::Node Config::to_yaml() const {
  YAML::Node node;

  node["pipeline"]["mode"] = pipeline.mode;
  node["pipeline"]["abort_on_fail"] = pipeline.abort_on_fail;

  node["output"]["registered_dir"] = output.registered_dir;
  node["output"]["artifacts_dir"] = output.artifacts_dir;
  node["output"]["write_registered_frames"] = output.write_registered_frames;
  node["output"]["write_global_metrics"] = output.write_global_metrics;
  node["output"]["write_global_registration"] =
      output.write_global_registration;

  node["data"]["image_width"] = data.image_width;
  node["data"]["image_height"] = data.image_height;
  node["data"]["frames_min"] = data.frames_min;
  node["data"]["frames_target"] = data.frames_target;
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
  node["assumptions"]["frames_optimal"] = assumptions.frames_optimal;
  node["assumptions"]["frames_reduced_threshold"] =
      assumptions.frames_reduced_threshold;
  node["assumptions"]["exposure_time_tolerance_percent"] =
      assumptions.exposure_time_tolerance_percent;
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
  node["registration"]["allow_rotation"] = registration.allow_rotation;
  node["registration"]["star_topk"] = registration.star_topk;
  node["registration"]["star_min_inliers"] = registration.star_min_inliers;
  node["registration"]["star_inlier_tol_px"] = registration.star_inlier_tol_px;
  node["registration"]["star_dist_bin_px"] = registration.star_dist_bin_px;

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

  node["global_metrics"]["adaptive_weights"] = global_metrics.adaptive_weights;
  node["global_metrics"]["weight_exponent_scale"] = global_metrics.weight_exponent_scale;
  node["global_metrics"]["weights"]["background"] =
      global_metrics.weights.background;
  node["global_metrics"]["weights"]["noise"] = global_metrics.weights.noise;
  node["global_metrics"]["weights"]["gradient"] =
      global_metrics.weights.gradient;
  node["global_metrics"]["clamp"].push_back(global_metrics.clamp[0]);
  node["global_metrics"]["clamp"].push_back(global_metrics.clamp[1]);

  node["tile"]["size_factor"] = tile.size_factor;
  node["tile"]["min_size"] = tile.min_size;
  node["tile"]["max_divisor"] = tile.max_divisor;
  node["tile"]["overlap_fraction"] = tile.overlap_fraction;
  node["tile"]["star_min_count"] = tile.star_min_count;

  node["local_metrics"]["clamp"].push_back(local_metrics.clamp[0]);
  node["local_metrics"]["clamp"].push_back(local_metrics.clamp[1]);
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

  node["synthetic"]["weighting"] = synthetic.weighting;
  node["synthetic"]["frames_min"] = synthetic.frames_min;
  node["synthetic"]["frames_max"] = synthetic.frames_max;
  node["synthetic"]["clustering"]["mode"] = synthetic.clustering.mode;
  node["synthetic"]["clustering"]["cluster_count_range"].push_back(
      synthetic.clustering.cluster_count_range[0]);
  node["synthetic"]["clustering"]["cluster_count_range"].push_back(
      synthetic.clustering.cluster_count_range[1]);

  node["reconstruction"]["weighting_function"] =
      reconstruction.weighting_function;
  node["reconstruction"]["window_function"] = reconstruction.window_function;

  node["debayer"] = debayer;

  node["astrometry"]["enabled"] = astrometry.enabled;
  node["astrometry"]["astap_bin"] = astrometry.astap_bin;
  node["astrometry"]["astap_data_dir"] = astrometry.astap_data_dir;
  node["astrometry"]["search_radius"] = astrometry.search_radius;

  node["pcc"]["enabled"] = pcc.enabled;
  node["pcc"]["source"] = pcc.source;
  node["pcc"]["mag_limit"] = pcc.mag_limit;
  node["pcc"]["mag_bright_limit"] = pcc.mag_bright_limit;
  node["pcc"]["aperture_radius_px"] = pcc.aperture_radius_px;
  node["pcc"]["annulus_inner_px"] = pcc.annulus_inner_px;
  node["pcc"]["annulus_outer_px"] = pcc.annulus_outer_px;
  node["pcc"]["min_stars"] = pcc.min_stars;
  node["pcc"]["sigma_clip"] = pcc.sigma_clip;
  node["pcc"]["siril_catalog_dir"] = pcc.siril_catalog_dir;

  node["stacking"]["method"] = stacking.method;
  node["stacking"]["sigma_clip"]["sigma_low"] = stacking.sigma_clip.sigma_low;
  node["stacking"]["sigma_clip"]["sigma_high"] = stacking.sigma_clip.sigma_high;
  node["stacking"]["sigma_clip"]["max_iters"] = stacking.sigma_clip.max_iters;
  node["stacking"]["sigma_clip"]["min_fraction"] =
      stacking.sigma_clip.min_fraction;
  node["stacking"]["output_stretch"] = stacking.output_stretch;
  node["stacking"]["cosmetic_correction"] =
      stacking.cosmetic_correction;

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

  return node;
}

void Config::validate() const {
  if (pipeline.mode != "production" && pipeline.mode != "test") {
    throw ValidationError("pipeline.mode must be 'production' or 'test'");
  }

  if (data.image_width < 1 || data.image_height < 1) {
    throw ValidationError(
        "data.image_width and data.image_height must be >= 1");
  }
  if (data.frames_min < 1) {
    throw ValidationError("data.frames_min must be >= 1");
  }
  if (data.frames_target < 0) {
    throw ValidationError("data.frames_target must be >= 0");
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

  if (assumptions.frames_min < 1)
    throw ValidationError("assumptions.frames_min must be >= 1");
  if (assumptions.frames_optimal < 1)
    throw ValidationError("assumptions.frames_optimal must be >= 1");
  if (assumptions.frames_reduced_threshold < assumptions.frames_min) {
    throw ValidationError("assumptions.frames_reduced_threshold must be >= "
                          "assumptions.frames_min");
  }
  if (assumptions.exposure_time_tolerance_percent < 0.0f) {
    throw ValidationError(
        "assumptions.exposure_time_tolerance_percent must be >= 0");
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
      registration.engine != "star_similarity" &&
      registration.engine != "triangle_star_matching") {
    throw ValidationError(
        "registration.engine must be 'triangle_star_matching', "
        "'star_similarity', or 'hybrid_phase_ecc'");
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

  auto check_weight_sum = [](float a, float b, float c, const char *name) {
    const float sum = a + b + c;
    if (std::fabs(sum - 1.0f) > 1.0e-3f) {
      throw ValidationError(std::string(name) + " must sum to 1.0");
    }
  };

  if (!is_between_0_1(global_metrics.weights.background) ||
      !is_between_0_1(global_metrics.weights.noise) ||
      !is_between_0_1(global_metrics.weights.gradient)) {
    throw ValidationError("global_metrics.weights.* must be between 0 and 1");
  }
  check_weight_sum(global_metrics.weights.background,
                   global_metrics.weights.noise,
                   global_metrics.weights.gradient, "global_metrics.weights");
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

  if (local_metrics.clamp[0] >= local_metrics.clamp[1]) {
    throw ValidationError(
        "local_metrics.clamp must be [min,max] with min < max");
  }
  check_weight_sum(local_metrics.star_mode.weights.fwhm,
                   local_metrics.star_mode.weights.roundness,
                   local_metrics.star_mode.weights.contrast,
                   "local_metrics.star_mode.weights");
  if (std::fabs(local_metrics.structure_mode.background_weight +
                local_metrics.structure_mode.metric_weight - 1.0f) > 1.0e-3f) {
    throw ValidationError(
        "local_metrics.structure_mode weights must sum to 1.0");
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

  if (reconstruction.weighting_function != "linear") {
    throw ValidationError("reconstruction.weighting_function must be 'linear'");
  }
  if (reconstruction.window_function != "hanning") {
    throw ValidationError("reconstruction.window_function must be 'hanning'");
  }

  if (stacking.method != "average" && stacking.method != "rej") {
    throw ValidationError("stacking.method must be 'average' or 'rej'");
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

  if (validation.min_tile_weight_variance < 0.0f) {
    throw ValidationError("validation.min_tile_weight_variance must be >= 0");
  }
  if (runtime_limits.tile_analysis_max_factor_vs_stack <= 0.0f) {
    throw ValidationError(
        "runtime_limits.tile_analysis_max_factor_vs_stack must be > 0");
  }
  if (runtime_limits.hard_abort_hours <= 0.0f) {
    throw ValidationError("runtime_limits.hard_abort_hours must be > 0");
  }
}

std::string get_schema_json() {
  return R"({
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "tile_compile v3 config",
  "type": "object",
  "properties": {
    "pipeline": { "type":"object",
      "properties": { "mode":{"type":"string","enum":["production","test"]},
                      "abort_on_fail":{"type":"boolean"} } },
    "output": { "type":"object",
      "properties": { "registered_dir":{"type":"string"},
                      "artifacts_dir":{"type":"string"},
                      "write_registered_frames":{"type":"boolean"},
                      "write_global_metrics":{"type":"boolean"},
                      "write_global_registration":{"type":"boolean"} } },
    "data": { "type":"object",
      "properties": { "image_width":{"type":"integer","minimum":1},
                      "image_height":{"type":"integer","minimum":1},
                      "frames_min":{"type":"integer","minimum":1},
                      "frames_target":{"type":"integer","minimum":0},
                      "color_mode":{"type":"string","enum":["OSC","MONO","RGB"]},
                      "bayer_pattern":{"type":"string"},
                      "linear_required":{"type":"boolean"} } },
    "linearity": { "type":"object",
      "properties": { "enabled":{"type":"boolean"},
                      "max_frames":{"type":"integer","minimum":1},
                      "min_overall_linearity":{"type":"number","minimum":0,"maximum":1},
                      "strictness":{"type":"string","enum":["strict","moderate","permissive"]} } },
    "calibration": { "type":"object",
      "properties": { "use_bias":{"type":"boolean"}, "use_dark":{"type":"boolean"}, "use_flat":{"type":"boolean"},
                      "bias_use_master":{"type":"boolean"}, "dark_use_master":{"type":"boolean"}, "flat_use_master":{"type":"boolean"},
                      "dark_auto_select":{"type":"boolean"},
                      "dark_match_exposure_tolerance_percent":{"type":"number","minimum":0},
                      "dark_match_use_temp":{"type":"boolean"},
                      "dark_match_temp_tolerance_c":{"type":"number","minimum":0},
                      "bias_dir":{"type":"string"}, "darks_dir":{"type":"string"}, "flats_dir":{"type":"string"},
                      "bias_master":{"type":"string"}, "dark_master":{"type":"string"}, "flat_master":{"type":"string"},
                      "pattern":{"type":"string"} } },
    "assumptions": { "type":"object",
      "properties": { "frames_min":{"type":"integer","minimum":1},
                      "frames_optimal":{"type":"integer","minimum":1},
                      "frames_reduced_threshold":{"type":"integer","minimum":1},
                      "exposure_time_tolerance_percent":{"type":"number","minimum":0},
                      "reduced_mode_skip_clustering":{"type":"boolean"},
                      "reduced_mode_cluster_range":{"type":"array","items":{"type":"integer","minimum":1},"minItems":2,"maxItems":2} } },
    "normalization": { "type":"object",
      "properties": { "enabled":{"type":"boolean"},
                      "mode":{"type":"string","enum":["background","median"]},
                      "per_channel":{"type":"boolean"} } },
    "registration": { "type":"object",
      "properties": { "engine":{"type":"string","enum":["triangle_star_matching","star_similarity","hybrid_phase_ecc"]},
                      "allow_rotation":{"type":"boolean"},
                      "star_topk":{"type":"integer","minimum":3},
                      "star_min_inliers":{"type":"integer","minimum":2},
                      "star_inlier_tol_px":{"type":"number","exclusiveMinimum":0},
                      "star_dist_bin_px":{"type":"number","exclusiveMinimum":0} } },
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
    "wiener_denoise": { "type":"object",
      "properties": { "enabled":{"type":"boolean"},
                      "snr_threshold":{"type":"number","minimum":0},
                      "q_min":{"type":"number","minimum":-1},
                      "q_max":{"type":"number","minimum":0,"maximum":1},
                      "q_step":{"type":"number","exclusiveMinimum":0},
                      "min_snr":{"type":"number","minimum":0},
                      "max_iterations":{"type":"integer","minimum":1} } },
    "global_metrics": { "type":"object",
      "properties": { "adaptive_weights":{"type":"boolean"},
                      "weight_exponent_scale":{"type":"number","exclusiveMinimum":0,"description":"Exponent scale k for G_f = exp(k * Q_f). k=1.0 (default) is standard, k>1 increases differentiation between good/bad frames."},
                      "weights":{"type":"object","properties":{"background":{"type":"number","minimum":0,"maximum":1},"noise":{"type":"number","minimum":0,"maximum":1},"gradient":{"type":"number","minimum":0,"maximum":1}}},
                      "clamp":{"type":"array","items":{"type":"number"},"minItems":2,"maxItems":2} } },
    "tile": { "type":"object",
      "properties": { "size_factor":{"type":"integer","minimum":1},
                      "min_size":{"type":"integer","minimum":1},
                      "max_divisor":{"type":"integer","minimum":1},
                      "overlap_fraction":{"type":"number","minimum":0,"maximum":0.5},
                      "star_min_count":{"type":"integer","minimum":0} } },
    "local_metrics": { "type":"object",
      "properties": { "clamp":{"type":"array","items":{"type":"number"},"minItems":2,"maxItems":2},
                      "star_mode":{"type":"object","properties":{"weights":{"type":"object","properties":{"fwhm":{"type":"number","minimum":0,"maximum":1},"roundness":{"type":"number","minimum":0,"maximum":1},"contrast":{"type":"number","minimum":0,"maximum":1}}}}},
                      "structure_mode":{"type":"object","properties":{"background_weight":{"type":"number","minimum":0,"maximum":1},"metric_weight":{"type":"number","minimum":0,"maximum":1}}} } },
    "synthetic": { "type":"object",
      "properties": { "weighting":{"type":"string","enum":["global","tile_weighted"]},
                      "frames_min":{"type":"integer","minimum":1},
                      "frames_max":{"type":"integer","minimum":1},
                      "clustering":{"type":"object","properties":{"mode":{"type":"string","enum":["kmeans","quantile"]},"cluster_count_range":{"type":"array","items":{"type":"integer","minimum":1},"minItems":2,"maxItems":2}}} } },
    "reconstruction": { "type":"object",
      "properties": { "weighting_function":{"type":"string","enum":["linear"]},
                      "window_function":{"type":"string","enum":["hanning"]} } },
    "debayer": {"type":"boolean"},
    "astrometry": { "type":"object",
      "properties": { "enabled":{"type":"boolean"},
                      "astap_bin":{"type":"string"},
                      "astap_data_dir":{"type":"string"},
                      "search_radius":{"type":"integer","minimum":1,"maximum":360} } },
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
                      "siril_catalog_dir":{"type":"string"} } },
    "stacking": { "type":"object",
      "properties": { "method":{"type":"string","enum":["rej","average"]},
                      "sigma_clip":{"type":"object","properties":{"sigma_low":{"type":"number","exclusiveMinimum":0},"sigma_high":{"type":"number","exclusiveMinimum":0},"max_iters":{"type":"integer","minimum":1},"min_fraction":{"type":"number","minimum":0,"maximum":1}}},
                      "output_stretch":{"type":"boolean"},
                      "cosmetic_correction":{"type":"boolean"} } },
    "validation": { "type":"object",
      "properties": { "min_fwhm_improvement_percent":{"type":"number"},
                      "max_background_rms_increase_percent":{"type":"number"},
                      "min_tile_weight_variance":{"type":"number","minimum":0},
                      "require_no_tile_pattern":{"type":"boolean"} } },
    "runtime_limits": { "type":"object",
      "properties": { "tile_analysis_max_factor_vs_stack":{"type":"number","exclusiveMinimum":0},
                      "hard_abort_hours":{"type":"number","exclusiveMinimum":0},
                      "allow_emergency_mode":{"type":"boolean"} } }
  }
})";
}

} // namespace tile_compile::config
