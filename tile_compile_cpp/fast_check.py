import os, re

words = [
    "abort_on_fail", "acceleration_backend", "adaptive_weights", "allow_emergency_mode", "allow_rotation",
    "alpha", "alpha_flatness", "amount", "annulus_inner_fwhm_mult", "annulus_inner_px",
    "annulus_outer_fwhm_mult", "annulus_outer_px", "aperture_fwhm_mult", "aperture_radius_px", "apply_attenuation",
    "apply_stage", "astap_bin", "astap_data_dir", "astrometry", "autotune",
    "background", "background_model", "background_weight", "bayer_pattern", "beta_roughness",
    "bias_dir", "bias_master", "bias_use_master", "blend", "blur_kernel",
    "cap_enabled", "cap_ratio", "chroma_bilateral", "chroma_denoise", "chroma_strength",
    "chroma_wavelet", "clamp", "cluster_count_range", "cluster_quality_weighting", "clustering",
    "color_mode", "color_space", "contrast", "cosmetic_correction", "cosmetic_correction_sigma",
    "crop_to_nonzero_bbox", "dark_auto_select", "dark_master", "dark_match_exposure_tolerance_percent", "dark_match_temp_tolerance_c",
    "dark_match_use_temp", "dark_use_master", "darks_dir", "debayer", "dilate_px",
    "dithering", "enable_star_pair_fallback", "enabled", "engine", "fit",
    "flat_master", "flat_use_master", "flats_dir", "frames_max", "frames_min",
    "frames_reduced_threshold", "fwhm", "G_max_fraction", "G_min_px", "global_metrics",
    "gradient", "gradient_percentile", "grid", "hard_abort_hours", "holdout_fraction",
    "huber_delta", "image_height", "image_width", "insufficient_cell_strategy", "irls_max_iterations",
    "irls_tolerance", "k_local", "k_max", "kappa_cluster", "lambda",
    "levels", "linear_required", "local_metrics", "luma_guard_strength", "mag_bright_limit",
    "mag_limit", "max_condition_number", "max_divisor", "max_evals", "max_frames",
    "max_iters", "max_residual_rms", "memory_budget", "method", "metric_weight",
    "min_aperture_px", "min_fraction", "min_fwhm_improvement_percent", "min_overall_linearity", "min_shift_px",
    "min_size", "min_snr", "min_stars", "min_tile_weight_variance", "min_tiles_per_cell",
    "min_valid_sample_fraction_for_apply", "min_valid_samples_for_apply", "mode", "N_g",
    "neighborhood_normalization", "noise", "normalization", "output_stretch", "overlap_fraction",
    "parallel_workers", "passes", "pattern", "per_channel", "per_frame_cosmetic_correction",
    "per_frame_cosmetic_correction_sigma", "polynomial_order", "protect_luma", "q_max", "q_min",
    "q_step", "radii_mode", "radius", "rbf_epsilon", "rbf_lambda",
    "rbf_mu_factor", "rbf_phi", "reduced_mode_cluster_range", "reduced_mode_skip_clustering", "registered_dir",
    "reject_cc_min_abs", "reject_outliers", "reject_scale_max", "reject_scale_min", "reject_shift_median_multiplier",
    "reject_shift_px_min", "require_no_tile_pattern", "robust_loss", "roundness", "sample_quantile",
    "sat_dilate_px", "search_radius", "sigma_clip", "sigma_high", "sigma_low",
    "sigma_range", "sigma_spatial", "siril_catalog_dir", "size_factor", "skip_star_tiles",
    "snr_threshold", "soft_k", "soft_threshold", "source", "spatial_regularization",
    "star_dilate_px", "star_dist_bin_px", "star_inlier_tol_px", "star_min_count", "star_min_inliers",
    "star_mode", "star_protection", "star_topk", "strategy", "strictness",
    "structure_mode", "structure_protection", "structure_thresh_percentile", "synthetic", "threshold_scale",
    "threshold_sigma", "tile_analysis_max_factor_vs_stack", "tile_denoise", "tile_weight_lambda_structure", "transform_model",
    "use_bias", "use_dark", "use_flat", "weight_exponent_scale", "weighting",
    "weights", "wiener", "write_registered_frames"
]

files_to_check = []
for d in ["src", "include", "apps"]:
    for r, dirs, fs in os.walk(d):
        for f in fs:
            if f.endswith(".cpp") or f.endswith(".hpp"):
                p = os.path.join(r, f)
                if "config.cpp" not in p and "configuration.hpp" not in p and "cli_main.cpp" not in p:
                    files_to_check.append(p)

contents = ""
for f in files_to_check:
    try: contents += open(f).read() + "\n"
    except: pass

unused = []
for w in words:
    if not re.search(r'\b' + w + r'\b', contents):
        unused.append(w)

print("UNUSED PROPERTIES:")
for u in unused: print(" -", u)
if not unused: print("ALL ARE USED!")
