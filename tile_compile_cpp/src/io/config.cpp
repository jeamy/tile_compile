#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/errors.hpp"

#include <cmath>
#include <fstream>
#include <sstream>

namespace tile_compile::config {

static bool is_odd(int v) {
    return (v % 2) != 0;
}

static void read_float_pair(const YAML::Node& n, std::array<float, 2>& out) {
    if (n && n.IsSequence() && n.size() == 2) {
        out[0] = n[0].as<float>();
        out[1] = n[1].as<float>();
    }
}

static void read_int_pair(const YAML::Node& n, std::array<int, 2>& out) {
    if (n && n.IsSequence() && n.size() == 2) {
        out[0] = n[0].as<int>();
        out[1] = n[1].as<int>();
    }
}

Config Config::load(const fs::path& path) {
    if (!fs::exists(path)) {
        throw ConfigError("Config file not found: " + path.string());
    }
    
    YAML::Node node = YAML::LoadFile(path.string());
    return from_yaml(node);
}

Config Config::from_yaml(const YAML::Node& node) {
    Config cfg;

    if (node["pipeline"]) {
        auto p = node["pipeline"];
        if (p["mode"]) cfg.pipeline.mode = p["mode"].as<std::string>();
        if (p["abort_on_fail"]) cfg.pipeline.abort_on_fail = p["abort_on_fail"].as<bool>();
    }

    if (node["data"]) {
        auto d = node["data"];
        if (d["image_width"]) cfg.data.image_width = d["image_width"].as<int>();
        if (d["image_height"]) cfg.data.image_height = d["image_height"].as<int>();
        if (d["frames_min"]) cfg.data.frames_min = d["frames_min"].as<int>();
        if (d["frames_target"]) cfg.data.frames_target = d["frames_target"].as<int>();
        if (d["color_mode"]) cfg.data.color_mode = d["color_mode"].as<std::string>();
        if (d["bayer_pattern"]) cfg.data.bayer_pattern = d["bayer_pattern"].as<std::string>();
        if (d["linear_required"]) cfg.data.linear_required = d["linear_required"].as<bool>();
    }

    if (node["calibration"]) {
        auto c = node["calibration"];
        if (c["use_bias"]) cfg.calibration.use_bias = c["use_bias"].as<bool>();
        if (c["use_dark"]) cfg.calibration.use_dark = c["use_dark"].as<bool>();
        if (c["use_flat"]) cfg.calibration.use_flat = c["use_flat"].as<bool>();
        if (c["bias_use_master"]) cfg.calibration.bias_use_master = c["bias_use_master"].as<bool>();
        if (c["dark_use_master"]) cfg.calibration.dark_use_master = c["dark_use_master"].as<bool>();
        if (c["flat_use_master"]) cfg.calibration.flat_use_master = c["flat_use_master"].as<bool>();
        if (c["dark_auto_select"]) cfg.calibration.dark_auto_select = c["dark_auto_select"].as<bool>();
        if (c["dark_match_exposure_tolerance_percent"]) {
            cfg.calibration.dark_match_exposure_tolerance_percent = c["dark_match_exposure_tolerance_percent"].as<float>();
        }
        if (c["dark_match_use_temp"]) cfg.calibration.dark_match_use_temp = c["dark_match_use_temp"].as<bool>();
        if (c["dark_match_temp_tolerance_c"]) cfg.calibration.dark_match_temp_tolerance_c = c["dark_match_temp_tolerance_c"].as<float>();

        if (c["bias_dir"]) cfg.calibration.bias_dir = c["bias_dir"].as<std::string>();
        if (c["darks_dir"]) cfg.calibration.darks_dir = c["darks_dir"].as<std::string>();
        if (c["flats_dir"]) cfg.calibration.flats_dir = c["flats_dir"].as<std::string>();
        if (c["bias_master"]) cfg.calibration.bias_master = c["bias_master"].as<std::string>();
        if (c["dark_master"]) cfg.calibration.dark_master = c["dark_master"].as<std::string>();
        if (c["flat_master"]) cfg.calibration.flat_master = c["flat_master"].as<std::string>();
        if (c["pattern"]) cfg.calibration.pattern = c["pattern"].as<std::string>();
    }

    if (node["assumptions"]) {
        auto a = node["assumptions"];
        if (a["frames_min"]) cfg.assumptions.frames_min = a["frames_min"].as<int>();
        if (a["frames_optimal"]) cfg.assumptions.frames_optimal = a["frames_optimal"].as<int>();
        if (a["frames_reduced_threshold"]) cfg.assumptions.frames_reduced_threshold = a["frames_reduced_threshold"].as<int>();
        if (a["exposure_time_tolerance_percent"]) {
            cfg.assumptions.exposure_time_tolerance_percent = a["exposure_time_tolerance_percent"].as<float>();
        }
        if (a["warp_variance_warn"]) cfg.assumptions.warp_variance_warn = a["warp_variance_warn"].as<float>();
        if (a["warp_variance_max"]) cfg.assumptions.warp_variance_max = a["warp_variance_max"].as<float>();
        if (a["elongation_warn"]) cfg.assumptions.elongation_warn = a["elongation_warn"].as<float>();
        if (a["elongation_max"]) cfg.assumptions.elongation_max = a["elongation_max"].as<float>();
        if (a["tracking_error_max_px"]) cfg.assumptions.tracking_error_max_px = a["tracking_error_max_px"].as<float>();
        if (a["reduced_mode_skip_clustering"]) {
            cfg.assumptions.reduced_mode_skip_clustering = a["reduced_mode_skip_clustering"].as<bool>();
        }
        read_int_pair(a["reduced_mode_cluster_range"], cfg.assumptions.reduced_mode_cluster_range);
    }

    if (node["v4"]) {
        auto v = node["v4"];
        if (v["iterations"]) cfg.v4.iterations = v["iterations"].as<int>();
        if (v["beta"]) cfg.v4.beta = v["beta"].as<float>();
        if (v["min_valid_tile_fraction"]) cfg.v4.min_valid_tile_fraction = v["min_valid_tile_fraction"].as<float>();
        if (v["parallel_tiles"]) cfg.v4.parallel_tiles = v["parallel_tiles"].as<int>();
        if (v["debug_tile_registration"]) cfg.v4.debug_tile_registration = v["debug_tile_registration"].as<bool>();

        if (v["adaptive_tiles"]) {
            auto at = v["adaptive_tiles"];
            if (at["enabled"]) cfg.v4.adaptive_tiles.enabled = at["enabled"].as<bool>();
            if (at["max_refine_passes"]) cfg.v4.adaptive_tiles.max_refine_passes = at["max_refine_passes"].as<int>();
            if (at["refine_variance_threshold"]) {
                cfg.v4.adaptive_tiles.refine_variance_threshold = at["refine_variance_threshold"].as<float>();
            }
            if (at["min_tile_size_px"]) cfg.v4.adaptive_tiles.min_tile_size_px = at["min_tile_size_px"].as<int>();
            if (at["use_warp_probe"]) cfg.v4.adaptive_tiles.use_warp_probe = at["use_warp_probe"].as<bool>();
            if (at["use_hierarchical"]) cfg.v4.adaptive_tiles.use_hierarchical = at["use_hierarchical"].as<bool>();
            if (at["initial_tile_size"]) cfg.v4.adaptive_tiles.initial_tile_size = at["initial_tile_size"].as<int>();
            if (at["probe_window"]) cfg.v4.adaptive_tiles.probe_window = at["probe_window"].as<int>();
            if (at["num_probe_frames"]) cfg.v4.adaptive_tiles.num_probe_frames = at["num_probe_frames"].as<int>();
            if (at["gradient_sensitivity"]) cfg.v4.adaptive_tiles.gradient_sensitivity = at["gradient_sensitivity"].as<float>();
            if (at["split_gradient_threshold"]) cfg.v4.adaptive_tiles.split_gradient_threshold = at["split_gradient_threshold"].as<float>();
            if (at["hierarchical_max_depth"]) cfg.v4.adaptive_tiles.hierarchical_max_depth = at["hierarchical_max_depth"].as<int>();
        }

        if (v["convergence"]) {
            auto cc = v["convergence"];
            if (cc["enabled"]) cfg.v4.convergence.enabled = cc["enabled"].as<bool>();
            if (cc["epsilon_rel"]) cfg.v4.convergence.epsilon_rel = cc["epsilon_rel"].as<float>();
        }

        if (v["memory_limits"]) {
            auto ml = v["memory_limits"];
            if (ml["rss_warn_mb"]) cfg.v4.memory_limits.rss_warn_mb = ml["rss_warn_mb"].as<int>();
            if (ml["rss_abort_mb"]) cfg.v4.memory_limits.rss_abort_mb = ml["rss_abort_mb"].as<int>();
        }

        if (v["diagnostics"]) {
            auto di = v["diagnostics"];
            if (di["enabled"]) cfg.v4.diagnostics.enabled = di["enabled"].as<bool>();
            if (di["warp_field"]) cfg.v4.diagnostics.warp_field = di["warp_field"].as<bool>();
            if (di["tile_invalid_map"]) cfg.v4.diagnostics.tile_invalid_map = di["tile_invalid_map"].as<bool>();
            if (di["warp_variance_hist"]) cfg.v4.diagnostics.warp_variance_hist = di["warp_variance_hist"].as<bool>();
        }
    }

    if (node["normalization"]) {
        auto n = node["normalization"];
        if (n["enabled"]) cfg.normalization.enabled = n["enabled"].as<bool>();
        if (n["mode"]) cfg.normalization.mode = n["mode"].as<std::string>();
        if (n["per_channel"]) cfg.normalization.per_channel = n["per_channel"].as<bool>();
    }

    if (node["registration"]) {
        auto r = node["registration"];
        if (r["mode"]) cfg.registration.mode = r["mode"].as<std::string>();
        if (r["local_tiles"]) {
            auto lt = r["local_tiles"];
            if (lt["max_warp_delta_px"]) cfg.registration.local_tiles.max_warp_delta_px = lt["max_warp_delta_px"].as<float>();
            if (lt["ecc_cc_min"]) cfg.registration.local_tiles.ecc_cc_min = lt["ecc_cc_min"].as<float>();
            if (lt["min_valid_frames"]) cfg.registration.local_tiles.min_valid_frames = lt["min_valid_frames"].as<int>();
            if (lt["temporal_smoothing_window"]) {
                cfg.registration.local_tiles.temporal_smoothing_window = lt["temporal_smoothing_window"].as<int>();
            }
            if (lt["variance_window_sigma"]) cfg.registration.local_tiles.variance_window_sigma = lt["variance_window_sigma"].as<float>();
        }
    }

    if (node["wiener_denoise"]) {
        auto w = node["wiener_denoise"];
        if (w["enabled"]) cfg.wiener_denoise.enabled = w["enabled"].as<bool>();
        if (w["snr_threshold"]) cfg.wiener_denoise.snr_threshold = w["snr_threshold"].as<float>();
        if (w["q_min"]) cfg.wiener_denoise.q_min = w["q_min"].as<float>();
        if (w["q_max"]) cfg.wiener_denoise.q_max = w["q_max"].as<float>();
        if (w["q_step"]) cfg.wiener_denoise.q_step = w["q_step"].as<float>();
        if (w["min_snr"]) cfg.wiener_denoise.min_snr = w["min_snr"].as<float>();
        if (w["max_iterations"]) cfg.wiener_denoise.max_iterations = w["max_iterations"].as<int>();
    }

    if (node["global_metrics"]) {
        auto gm = node["global_metrics"];
        if (gm["adaptive_weights"]) cfg.global_metrics.adaptive_weights = gm["adaptive_weights"].as<bool>();
        if (gm["weights"]) {
            auto w = gm["weights"];
            if (w["background"]) cfg.global_metrics.weights.background = w["background"].as<float>();
            if (w["noise"]) cfg.global_metrics.weights.noise = w["noise"].as<float>();
            if (w["gradient"]) cfg.global_metrics.weights.gradient = w["gradient"].as<float>();
        }
        read_float_pair(gm["clamp"], cfg.global_metrics.clamp);
    }

    if (node["tile"]) {
        auto t = node["tile"];
        if (t["size_factor"]) cfg.tile.size_factor = t["size_factor"].as<int>();
        if (t["min_size"]) cfg.tile.min_size = t["min_size"].as<int>();
        if (t["max_divisor"]) cfg.tile.max_divisor = t["max_divisor"].as<int>();
        if (t["overlap_fraction"]) cfg.tile.overlap_fraction = t["overlap_fraction"].as<float>();
        if (t["star_min_count"]) cfg.tile.star_min_count = t["star_min_count"].as<int>();
    }

    if (node["local_metrics"]) {
        auto lm = node["local_metrics"];
        read_float_pair(lm["clamp"], cfg.local_metrics.clamp);
        if (lm["star_mode"] && lm["star_mode"]["weights"]) {
            auto w = lm["star_mode"]["weights"];
            if (w["fwhm"]) cfg.local_metrics.star_mode.weights.fwhm = w["fwhm"].as<float>();
            if (w["roundness"]) cfg.local_metrics.star_mode.weights.roundness = w["roundness"].as<float>();
            if (w["contrast"]) cfg.local_metrics.star_mode.weights.contrast = w["contrast"].as<float>();
        }
        if (lm["structure_mode"]) {
            auto sm = lm["structure_mode"];
            if (sm["background_weight"]) cfg.local_metrics.structure_mode.background_weight = sm["background_weight"].as<float>();
            if (sm["metric_weight"]) cfg.local_metrics.structure_mode.metric_weight = sm["metric_weight"].as<float>();
        }
    }

    if (node["clustering"]) {
        auto cl = node["clustering"];
        if (cl["mode"]) cfg.clustering.mode = cl["mode"].as<std::string>();
        if (cl["k_selection"]) cfg.clustering.k_selection = cl["k_selection"].as<std::string>();
        if (cl["use_silhouette"]) cfg.clustering.use_silhouette = cl["use_silhouette"].as<bool>();
        if (cl["fallback_quantiles"]) cfg.clustering.fallback_quantiles = cl["fallback_quantiles"].as<int>();
        if (cl["vector"] && cl["vector"].IsSequence()) {
            cfg.clustering.vector.clear();
            for (const auto& it : cl["vector"]) {
                cfg.clustering.vector.push_back(it.as<std::string>());
            }
        }
        read_int_pair(cl["cluster_count_range"], cfg.clustering.cluster_count_range);
    }

    if (node["synthetic"]) {
        auto s = node["synthetic"];
        if (s["weighting"]) cfg.synthetic.weighting = s["weighting"].as<std::string>();
        if (s["frames_min"]) cfg.synthetic.frames_min = s["frames_min"].as<int>();
        if (s["frames_max"]) cfg.synthetic.frames_max = s["frames_max"].as<int>();
    }

    if (node["reconstruction"]) {
        auto r = node["reconstruction"];
        if (r["weighting_function"]) cfg.reconstruction.weighting_function = r["weighting_function"].as<std::string>();
        if (r["window_function"]) cfg.reconstruction.window_function = r["window_function"].as<std::string>();
        if (r["tile_rescale"]) cfg.reconstruction.tile_rescale = r["tile_rescale"].as<std::string>();
    }

    if (node["debayer"]) cfg.debayer = node["debayer"].as<bool>();

    if (node["stacking"]) {
        auto st = node["stacking"];
        if (st["method"]) cfg.stacking.method = st["method"].as<std::string>();
        if (st["input_dir"]) cfg.stacking.input_dir = st["input_dir"].as<std::string>();
        if (st["input_pattern"]) cfg.stacking.input_pattern = st["input_pattern"].as<std::string>();
        if (st["output_file"]) cfg.stacking.output_file = st["output_file"].as<std::string>();
        if (st["sigma_clip"]) {
            auto sc = st["sigma_clip"];
            if (sc["sigma_low"]) cfg.stacking.sigma_clip.sigma_low = sc["sigma_low"].as<float>();
            if (sc["sigma_high"]) cfg.stacking.sigma_clip.sigma_high = sc["sigma_high"].as<float>();
            if (sc["max_iters"]) cfg.stacking.sigma_clip.max_iters = sc["max_iters"].as<int>();
            if (sc["min_fraction"]) cfg.stacking.sigma_clip.min_fraction = sc["min_fraction"].as<float>();
        }
    }

    if (node["validation"]) {
        auto v = node["validation"];
        if (v["min_fwhm_improvement_percent"]) {
            cfg.validation.min_fwhm_improvement_percent = v["min_fwhm_improvement_percent"].as<float>();
        }
        if (v["max_background_rms_increase_percent"]) {
            cfg.validation.max_background_rms_increase_percent = v["max_background_rms_increase_percent"].as<float>();
        }
        if (v["min_tile_weight_variance"]) cfg.validation.min_tile_weight_variance = v["min_tile_weight_variance"].as<float>();
        if (v["require_no_tile_pattern"]) cfg.validation.require_no_tile_pattern = v["require_no_tile_pattern"].as<bool>();
    }

    if (node["runtime_limits"]) {
        auto rl = node["runtime_limits"];
        if (rl["tile_analysis_max_factor_vs_stack"]) {
            cfg.runtime_limits.tile_analysis_max_factor_vs_stack = rl["tile_analysis_max_factor_vs_stack"].as<float>();
        }
        if (rl["hard_abort_hours"]) cfg.runtime_limits.hard_abort_hours = rl["hard_abort_hours"].as<float>();
    }

    return cfg;
}

void Config::save(const fs::path& path) const {
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

    node["data"]["image_width"] = data.image_width;
    node["data"]["image_height"] = data.image_height;
    node["data"]["frames_min"] = data.frames_min;
    node["data"]["frames_target"] = data.frames_target;
    node["data"]["color_mode"] = data.color_mode;
    node["data"]["bayer_pattern"] = data.bayer_pattern;
    node["data"]["linear_required"] = data.linear_required;

    node["calibration"]["use_bias"] = calibration.use_bias;
    node["calibration"]["use_dark"] = calibration.use_dark;
    node["calibration"]["use_flat"] = calibration.use_flat;
    node["calibration"]["bias_use_master"] = calibration.bias_use_master;
    node["calibration"]["dark_use_master"] = calibration.dark_use_master;
    node["calibration"]["flat_use_master"] = calibration.flat_use_master;
    node["calibration"]["dark_auto_select"] = calibration.dark_auto_select;
    node["calibration"]["dark_match_exposure_tolerance_percent"] = calibration.dark_match_exposure_tolerance_percent;
    node["calibration"]["dark_match_use_temp"] = calibration.dark_match_use_temp;
    node["calibration"]["dark_match_temp_tolerance_c"] = calibration.dark_match_temp_tolerance_c;
    node["calibration"]["bias_dir"] = calibration.bias_dir;
    node["calibration"]["darks_dir"] = calibration.darks_dir;
    node["calibration"]["flats_dir"] = calibration.flats_dir;
    node["calibration"]["bias_master"] = calibration.bias_master;
    node["calibration"]["dark_master"] = calibration.dark_master;
    node["calibration"]["flat_master"] = calibration.flat_master;
    node["calibration"]["pattern"] = calibration.pattern;

    node["assumptions"]["frames_min"] = assumptions.frames_min;
    node["assumptions"]["frames_optimal"] = assumptions.frames_optimal;
    node["assumptions"]["frames_reduced_threshold"] = assumptions.frames_reduced_threshold;
    node["assumptions"]["exposure_time_tolerance_percent"] = assumptions.exposure_time_tolerance_percent;
    node["assumptions"]["warp_variance_warn"] = assumptions.warp_variance_warn;
    node["assumptions"]["warp_variance_max"] = assumptions.warp_variance_max;
    node["assumptions"]["elongation_warn"] = assumptions.elongation_warn;
    node["assumptions"]["elongation_max"] = assumptions.elongation_max;
    node["assumptions"]["tracking_error_max_px"] = assumptions.tracking_error_max_px;
    node["assumptions"]["reduced_mode_skip_clustering"] = assumptions.reduced_mode_skip_clustering;
    node["assumptions"]["reduced_mode_cluster_range"].push_back(assumptions.reduced_mode_cluster_range[0]);
    node["assumptions"]["reduced_mode_cluster_range"].push_back(assumptions.reduced_mode_cluster_range[1]);

    node["v4"]["iterations"] = v4.iterations;
    node["v4"]["beta"] = v4.beta;
    node["v4"]["min_valid_tile_fraction"] = v4.min_valid_tile_fraction;
    node["v4"]["parallel_tiles"] = v4.parallel_tiles;
    node["v4"]["debug_tile_registration"] = v4.debug_tile_registration;
    node["v4"]["adaptive_tiles"]["enabled"] = v4.adaptive_tiles.enabled;
    node["v4"]["adaptive_tiles"]["max_refine_passes"] = v4.adaptive_tiles.max_refine_passes;
    node["v4"]["adaptive_tiles"]["refine_variance_threshold"] = v4.adaptive_tiles.refine_variance_threshold;
    node["v4"]["adaptive_tiles"]["min_tile_size_px"] = v4.adaptive_tiles.min_tile_size_px;
    node["v4"]["adaptive_tiles"]["use_warp_probe"] = v4.adaptive_tiles.use_warp_probe;
    node["v4"]["adaptive_tiles"]["use_hierarchical"] = v4.adaptive_tiles.use_hierarchical;
    node["v4"]["adaptive_tiles"]["initial_tile_size"] = v4.adaptive_tiles.initial_tile_size;
    node["v4"]["adaptive_tiles"]["probe_window"] = v4.adaptive_tiles.probe_window;
    node["v4"]["adaptive_tiles"]["num_probe_frames"] = v4.adaptive_tiles.num_probe_frames;
    node["v4"]["adaptive_tiles"]["gradient_sensitivity"] = v4.adaptive_tiles.gradient_sensitivity;
    node["v4"]["adaptive_tiles"]["split_gradient_threshold"] = v4.adaptive_tiles.split_gradient_threshold;
    node["v4"]["adaptive_tiles"]["hierarchical_max_depth"] = v4.adaptive_tiles.hierarchical_max_depth;
    node["v4"]["convergence"]["enabled"] = v4.convergence.enabled;
    node["v4"]["convergence"]["epsilon_rel"] = v4.convergence.epsilon_rel;
    node["v4"]["memory_limits"]["rss_warn_mb"] = v4.memory_limits.rss_warn_mb;
    node["v4"]["memory_limits"]["rss_abort_mb"] = v4.memory_limits.rss_abort_mb;
    node["v4"]["diagnostics"]["enabled"] = v4.diagnostics.enabled;
    node["v4"]["diagnostics"]["warp_field"] = v4.diagnostics.warp_field;
    node["v4"]["diagnostics"]["tile_invalid_map"] = v4.diagnostics.tile_invalid_map;
    node["v4"]["diagnostics"]["warp_variance_hist"] = v4.diagnostics.warp_variance_hist;

    node["normalization"]["enabled"] = normalization.enabled;
    node["normalization"]["mode"] = normalization.mode;
    node["normalization"]["per_channel"] = normalization.per_channel;

    node["registration"]["mode"] = registration.mode;
    node["registration"]["local_tiles"]["max_warp_delta_px"] = registration.local_tiles.max_warp_delta_px;
    node["registration"]["local_tiles"]["ecc_cc_min"] = registration.local_tiles.ecc_cc_min;
    node["registration"]["local_tiles"]["min_valid_frames"] = registration.local_tiles.min_valid_frames;
    node["registration"]["local_tiles"]["temporal_smoothing_window"] = registration.local_tiles.temporal_smoothing_window;
    node["registration"]["local_tiles"]["variance_window_sigma"] = registration.local_tiles.variance_window_sigma;

    node["wiener_denoise"]["enabled"] = wiener_denoise.enabled;
    node["wiener_denoise"]["snr_threshold"] = wiener_denoise.snr_threshold;
    node["wiener_denoise"]["q_min"] = wiener_denoise.q_min;
    node["wiener_denoise"]["q_max"] = wiener_denoise.q_max;
    node["wiener_denoise"]["q_step"] = wiener_denoise.q_step;
    node["wiener_denoise"]["min_snr"] = wiener_denoise.min_snr;
    node["wiener_denoise"]["max_iterations"] = wiener_denoise.max_iterations;

    node["global_metrics"]["adaptive_weights"] = global_metrics.adaptive_weights;
    node["global_metrics"]["weights"]["background"] = global_metrics.weights.background;
    node["global_metrics"]["weights"]["noise"] = global_metrics.weights.noise;
    node["global_metrics"]["weights"]["gradient"] = global_metrics.weights.gradient;
    node["global_metrics"]["clamp"].push_back(global_metrics.clamp[0]);
    node["global_metrics"]["clamp"].push_back(global_metrics.clamp[1]);

    node["tile"]["size_factor"] = tile.size_factor;
    node["tile"]["min_size"] = tile.min_size;
    node["tile"]["max_divisor"] = tile.max_divisor;
    node["tile"]["overlap_fraction"] = tile.overlap_fraction;
    node["tile"]["star_min_count"] = tile.star_min_count;

    node["local_metrics"]["clamp"].push_back(local_metrics.clamp[0]);
    node["local_metrics"]["clamp"].push_back(local_metrics.clamp[1]);
    node["local_metrics"]["star_mode"]["weights"]["fwhm"] = local_metrics.star_mode.weights.fwhm;
    node["local_metrics"]["star_mode"]["weights"]["roundness"] = local_metrics.star_mode.weights.roundness;
    node["local_metrics"]["star_mode"]["weights"]["contrast"] = local_metrics.star_mode.weights.contrast;
    node["local_metrics"]["structure_mode"]["background_weight"] = local_metrics.structure_mode.background_weight;
    node["local_metrics"]["structure_mode"]["metric_weight"] = local_metrics.structure_mode.metric_weight;

    node["clustering"]["mode"] = clustering.mode;
    node["clustering"]["k_selection"] = clustering.k_selection;
    node["clustering"]["use_silhouette"] = clustering.use_silhouette;
    node["clustering"]["fallback_quantiles"] = clustering.fallback_quantiles;
    node["clustering"]["cluster_count_range"].push_back(clustering.cluster_count_range[0]);
    node["clustering"]["cluster_count_range"].push_back(clustering.cluster_count_range[1]);
    for (const auto& key : clustering.vector) {
        node["clustering"]["vector"].push_back(key);
    }

    node["synthetic"]["weighting"] = synthetic.weighting;
    node["synthetic"]["frames_min"] = synthetic.frames_min;
    node["synthetic"]["frames_max"] = synthetic.frames_max;

    node["reconstruction"]["weighting_function"] = reconstruction.weighting_function;
    node["reconstruction"]["window_function"] = reconstruction.window_function;
    node["reconstruction"]["tile_rescale"] = reconstruction.tile_rescale;

    node["debayer"] = debayer;

    node["stacking"]["method"] = stacking.method;
    node["stacking"]["input_dir"] = stacking.input_dir;
    node["stacking"]["input_pattern"] = stacking.input_pattern;
    node["stacking"]["output_file"] = stacking.output_file;
    node["stacking"]["sigma_clip"]["sigma_low"] = stacking.sigma_clip.sigma_low;
    node["stacking"]["sigma_clip"]["sigma_high"] = stacking.sigma_clip.sigma_high;
    node["stacking"]["sigma_clip"]["max_iters"] = stacking.sigma_clip.max_iters;
    node["stacking"]["sigma_clip"]["min_fraction"] = stacking.sigma_clip.min_fraction;

    node["validation"]["min_fwhm_improvement_percent"] = validation.min_fwhm_improvement_percent;
    node["validation"]["max_background_rms_increase_percent"] = validation.max_background_rms_increase_percent;
    node["validation"]["min_tile_weight_variance"] = validation.min_tile_weight_variance;
    node["validation"]["require_no_tile_pattern"] = validation.require_no_tile_pattern;

    node["runtime_limits"]["tile_analysis_max_factor_vs_stack"] = runtime_limits.tile_analysis_max_factor_vs_stack;
    node["runtime_limits"]["hard_abort_hours"] = runtime_limits.hard_abort_hours;

    return node;
}

void Config::validate() const {
    if (pipeline.mode != "production" && pipeline.mode != "test") {
        throw ValidationError("pipeline.mode must be 'production' or 'test'");
    }

    if (data.image_width < 1 || data.image_height < 1) {
        throw ValidationError("data.image_width and data.image_height must be >= 1");
    }
    if (data.frames_min < 1) {
        throw ValidationError("data.frames_min must be >= 1");
    }
    if (data.frames_target < 0) {
        throw ValidationError("data.frames_target must be >= 0");
    }
    if (!data.linear_required) {
        throw ValidationError("data.linear_required must be true (Methodik v4)");
    }

    if (assumptions.frames_min < 1) {
        throw ValidationError("assumptions.frames_min must be >= 1");
    }
    if (assumptions.frames_optimal < 1) {
        throw ValidationError("assumptions.frames_optimal must be >= 1");
    }
    if (assumptions.frames_reduced_threshold < 1) {
        throw ValidationError("assumptions.frames_reduced_threshold must be >= 1");
    }
    if (assumptions.frames_reduced_threshold < assumptions.frames_min) {
        throw ValidationError("assumptions.frames_reduced_threshold must be >= assumptions.frames_min");
    }
    if (assumptions.exposure_time_tolerance_percent < 0) {
        throw ValidationError("assumptions.exposure_time_tolerance_percent must be >= 0");
    }
    if (assumptions.warp_variance_warn < 0 || assumptions.warp_variance_max < 0) {
        throw ValidationError("assumptions.warp_variance_warn/max must be >= 0");
    }
    if (assumptions.elongation_warn < 0 || assumptions.elongation_warn > 1 || assumptions.elongation_max < 0 || assumptions.elongation_max > 1) {
        throw ValidationError("assumptions.elongation_warn/max must be in [0,1]");
    }
    if (assumptions.reduced_mode_cluster_range[0] < 1 || assumptions.reduced_mode_cluster_range[1] < assumptions.reduced_mode_cluster_range[0]) {
        throw ValidationError("assumptions.reduced_mode_cluster_range must be [min,max] with min>=1 and max>=min");
    }

    if (v4.iterations < 1 || v4.iterations > 10) {
        throw ValidationError("v4.iterations must be in [1,10]");
    }
    if (v4.beta < 0.0f || v4.beta > 20.0f) {
        throw ValidationError("v4.beta must be in [0,20]");
    }
    if (v4.min_valid_tile_fraction < 0.0f || v4.min_valid_tile_fraction > 1.0f) {
        throw ValidationError("v4.min_valid_tile_fraction must be in [0,1]");
    }
    if (v4.parallel_tiles < 1 || v4.parallel_tiles > 32) {
        throw ValidationError("v4.parallel_tiles must be in [1,32]");
    }

    if (v4.adaptive_tiles.max_refine_passes < 0 || v4.adaptive_tiles.max_refine_passes > 5) {
        throw ValidationError("v4.adaptive_tiles.max_refine_passes must be in [0,5]");
    }
    if (v4.adaptive_tiles.refine_variance_threshold < 0.0f) {
        throw ValidationError("v4.adaptive_tiles.refine_variance_threshold must be >= 0");
    }
    if (v4.adaptive_tiles.min_tile_size_px < 32 || v4.adaptive_tiles.min_tile_size_px > 512) {
        throw ValidationError("v4.adaptive_tiles.min_tile_size_px must be in [32,512]");
    }
    if (v4.adaptive_tiles.initial_tile_size < 64 || v4.adaptive_tiles.initial_tile_size > 512) {
        throw ValidationError("v4.adaptive_tiles.initial_tile_size must be in [64,512]");
    }
    if (v4.adaptive_tiles.probe_window < 64 || v4.adaptive_tiles.probe_window > 512) {
        throw ValidationError("v4.adaptive_tiles.probe_window must be in [64,512]");
    }
    if (v4.adaptive_tiles.num_probe_frames < 3 || v4.adaptive_tiles.num_probe_frames > 10) {
        throw ValidationError("v4.adaptive_tiles.num_probe_frames must be in [3,10]");
    }
    if (v4.adaptive_tiles.gradient_sensitivity < 0.1f || v4.adaptive_tiles.gradient_sensitivity > 10.0f) {
        throw ValidationError("v4.adaptive_tiles.gradient_sensitivity must be in [0.1,10]");
    }
    if (v4.adaptive_tiles.split_gradient_threshold < 0.0f || v4.adaptive_tiles.split_gradient_threshold > 1.0f) {
        throw ValidationError("v4.adaptive_tiles.split_gradient_threshold must be in [0,1]");
    }
    if (v4.adaptive_tiles.hierarchical_max_depth < 1 || v4.adaptive_tiles.hierarchical_max_depth > 5) {
        throw ValidationError("v4.adaptive_tiles.hierarchical_max_depth must be in [1,5]");
    }
    if (v4.convergence.epsilon_rel <= 0.0f) {
        throw ValidationError("v4.convergence.epsilon_rel must be > 0");
    }
    if (v4.memory_limits.rss_warn_mb < 1 || v4.memory_limits.rss_abort_mb < v4.memory_limits.rss_warn_mb) {
        throw ValidationError("v4.memory_limits.rss_warn_mb must be >=1 and rss_abort_mb must be >= rss_warn_mb");
    }

    if (!normalization.enabled) {
        throw ValidationError("normalization.enabled must be true (Methodik v4)");
    }
    if (normalization.mode != "background" && normalization.mode != "median") {
        throw ValidationError("normalization.mode must be 'background' or 'median'");
    }

    if (registration.mode != "local_tiles") {
        throw ValidationError("registration.mode must be 'local_tiles' (Methodik v4)");
    }
    if (registration.local_tiles.max_warp_delta_px < 0.0f) {
        throw ValidationError("registration.local_tiles.max_warp_delta_px must be >= 0");
    }
    if (registration.local_tiles.ecc_cc_min < 0.0f || registration.local_tiles.ecc_cc_min > 1.0f) {
        throw ValidationError("registration.local_tiles.ecc_cc_min must be in [0,1]");
    }
    if (registration.local_tiles.min_valid_frames < 1) {
        throw ValidationError("registration.local_tiles.min_valid_frames must be >= 1");
    }
    if (registration.local_tiles.temporal_smoothing_window < 3 || !is_odd(registration.local_tiles.temporal_smoothing_window)) {
        throw ValidationError("registration.local_tiles.temporal_smoothing_window must be odd and >= 3");
    }
    if (registration.local_tiles.variance_window_sigma < 0.0f) {
        throw ValidationError("registration.local_tiles.variance_window_sigma must be >= 0");
    }

    if (wiener_denoise.q_max < 0.0f || wiener_denoise.q_max > 1.0f) {
        throw ValidationError("wiener_denoise.q_max must be in [0,1]");
    }
    if (wiener_denoise.q_min < -1.0f || wiener_denoise.q_min > wiener_denoise.q_max) {
        throw ValidationError("wiener_denoise.q_min must be <= q_max and >= -1");
    }
    if (wiener_denoise.q_step <= 0.0f) {
        throw ValidationError("wiener_denoise.q_step must be > 0");
    }
    if (wiener_denoise.max_iterations < 1) {
        throw ValidationError("wiener_denoise.max_iterations must be >= 1");
    }

    if (global_metrics.weights.background < 0 || global_metrics.weights.background > 1) {
        throw ValidationError("global_metrics.weights.background must be between 0 and 1");
    }
    if (global_metrics.weights.noise < 0 || global_metrics.weights.noise > 1) {
        throw ValidationError("global_metrics.weights.noise must be between 0 and 1");
    }
    if (global_metrics.weights.gradient < 0 || global_metrics.weights.gradient > 1) {
        throw ValidationError("global_metrics.weights.gradient must be between 0 and 1");
    }
    {
        const float sum = global_metrics.weights.background + global_metrics.weights.noise + global_metrics.weights.gradient;
        if (std::fabs(sum - 1.0f) > 1.0e-3f) {
            throw ValidationError("global_metrics.weights.* must sum to 1.0");
        }
    }
    if (global_metrics.clamp[0] >= global_metrics.clamp[1]) {
        throw ValidationError("global_metrics.clamp must be [min,max] with min < max");
    }

    if (tile.size_factor <= 0) {
        throw ValidationError("tile.size_factor must be positive");
    }
    if (tile.min_size <= 0) {
        throw ValidationError("tile.min_size must be positive");
    }
    if (tile.max_divisor <= 0) {
        throw ValidationError("tile.max_divisor must be positive");
    }
    if (tile.overlap_fraction < 0 || tile.overlap_fraction > 0.5f) {
        throw ValidationError("tile.overlap_fraction must be between 0 and 0.5");
    }
    if (tile.star_min_count < 1) {
        throw ValidationError("tile.star_min_count must be >= 1");
    }

    if (local_metrics.clamp[0] >= local_metrics.clamp[1]) {
        throw ValidationError("local_metrics.clamp must be [min,max] with min < max");
    }
    {
        const float sum = local_metrics.star_mode.weights.fwhm + local_metrics.star_mode.weights.roundness + local_metrics.star_mode.weights.contrast;
        if (std::fabs(sum - 1.0f) > 1.0e-3f) {
            throw ValidationError("local_metrics.star_mode.weights.* must sum to 1.0");
        }
    }
    {
        const float sum = local_metrics.structure_mode.background_weight + local_metrics.structure_mode.metric_weight;
        if (std::fabs(sum - 1.0f) > 1.0e-3f) {
            throw ValidationError("local_metrics.structure_mode weights must sum to 1.0");
        }
    }

    if (clustering.cluster_count_range[0] < 1 || clustering.cluster_count_range[1] < clustering.cluster_count_range[0]) {
        throw ValidationError("clustering.cluster_count_range must be [min,max] with min>=1 and max>=min");
    }
    if (clustering.k_selection != "auto" && clustering.k_selection != "fixed") {
        throw ValidationError("clustering.k_selection must be 'auto' or 'fixed'");
    }

    if (synthetic.weighting != "global" && synthetic.weighting != "tile_weighted") {
        throw ValidationError("synthetic.weighting must be 'global' or 'tile_weighted'");
    }
    if (synthetic.frames_min < 1) {
        throw ValidationError("synthetic.frames_min must be at least 1");
    }
    if (synthetic.frames_max < synthetic.frames_min) {
        throw ValidationError("synthetic.frames_max must be >= frames_min");
    }

    if (reconstruction.weighting_function != "exponential") {
        throw ValidationError("reconstruction.weighting_function must be 'exponential'");
    }
    if (reconstruction.window_function != "hanning") {
        throw ValidationError("reconstruction.window_function must be 'hanning'");
    }
    if (reconstruction.tile_rescale != "median_after_background_subtraction") {
        throw ValidationError("reconstruction.tile_rescale must be 'median_after_background_subtraction'");
    }

    if (stacking.method != "average" && stacking.method != "rej") {
        throw ValidationError("stacking.method must be 'average' or 'rej'");
    }
    if (stacking.sigma_clip.sigma_low <= 0.0f || stacking.sigma_clip.sigma_high <= 0.0f) {
        throw ValidationError("stacking.sigma_clip.sigma_low/high must be > 0");
    }
    if (stacking.sigma_clip.max_iters < 1) {
        throw ValidationError("stacking.sigma_clip.max_iters must be >= 1");
    }
    if (stacking.sigma_clip.min_fraction < 0.0f || stacking.sigma_clip.min_fraction > 1.0f) {
        throw ValidationError("stacking.sigma_clip.min_fraction must be in [0,1]");
    }

    if (validation.min_tile_weight_variance < 0.0f) {
        throw ValidationError("validation.min_tile_weight_variance must be >= 0");
    }
    if (runtime_limits.tile_analysis_max_factor_vs_stack <= 0.0f) {
        throw ValidationError("runtime_limits.tile_analysis_max_factor_vs_stack must be > 0");
    }
    if (runtime_limits.hard_abort_hours <= 0.0f) {
        throw ValidationError("runtime_limits.hard_abort_hours must be > 0");
    }
}

std::string get_schema_json() {
    return R"({
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "pipeline": {
      "type": "object",
      "properties": {
        "mode": {"type": "string", "enum": ["production", "test"]},
        "abort_on_fail": {"type": "boolean"}
      }
    },
    "data": {
      "type": "object",
      "properties": {
        "image_width": {"type": "integer", "minimum": 1},
        "image_height": {"type": "integer", "minimum": 1},
        "frames_min": {"type": "integer", "minimum": 1},
        "frames_target": {"type": "integer", "minimum": 0},
        "color_mode": {"type": "string"},
        "bayer_pattern": {"type": "string"},
        "linear_required": {"type": "boolean"}
      }
    },
    "v4": {
      "type": "object",
      "properties": {
        "iterations": {"type": "integer", "minimum": 1, "maximum": 10},
        "beta": {"type": "number", "minimum": 0, "maximum": 20},
        "min_valid_tile_fraction": {"type": "number", "minimum": 0, "maximum": 1},
        "parallel_tiles": {"type": "integer", "minimum": 1, "maximum": 32},
        "debug_tile_registration": {"type": "boolean"}
      }
    },
    "global_metrics": {
      "type": "object",
      "properties": {
        "adaptive_weights": {"type": "boolean"},
        "weights": {
          "type": "object",
          "properties": {
            "background": {"type": "number", "minimum": 0, "maximum": 1},
            "noise": {"type": "number", "minimum": 0, "maximum": 1},
            "gradient": {"type": "number", "minimum": 0, "maximum": 1}
          }
        },
        "clamp": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
      }
    },
    "tile": {
      "type": "object",
      "properties": {
        "size_factor": {"type": "integer", "minimum": 1},
        "min_size": {"type": "integer", "minimum": 1},
        "max_divisor": {"type": "integer", "minimum": 1},
        "overlap_fraction": {"type": "number", "minimum": 0, "maximum": 0.5},
        "star_min_count": {"type": "integer", "minimum": 1}
      }
    },
    "registration": {
      "type": "object",
      "properties": {
        "mode": {"type": "string", "enum": ["local_tiles"]},
        "local_tiles": {
          "type": "object",
          "properties": {
            "max_warp_delta_px": {"type": "number", "minimum": 0},
            "ecc_cc_min": {"type": "number", "minimum": 0, "maximum": 1},
            "min_valid_frames": {"type": "integer", "minimum": 1},
            "temporal_smoothing_window": {"type": "integer", "minimum": 3},
            "variance_window_sigma": {"type": "number", "minimum": 0}
          }
        }
      }
    }
  }
})";
}

} // namespace tile_compile::config
