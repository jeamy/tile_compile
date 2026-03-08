window.PARAM_EDITOR_INDEX = [
  {
    "category": "assumptions",
    "minimum": 0,
    "path": "assumptions.exposure_time_tolerance_percent",
    "source": "schema",
    "type": "number",
    "yaml_default": 8.0
  },
  {
    "category": "assumptions",
    "minimum": 1,
    "path": "assumptions.frames_min",
    "source": "schema",
    "type": "integer",
    "yaml_default": 30
  },
  {
    "category": "assumptions",
    "minimum": 1,
    "path": "assumptions.frames_optimal",
    "source": "schema",
    "type": "integer",
    "yaml_default": 300
  },
  {
    "category": "assumptions",
    "description": "Schwelle fuer Full Mode: 50..(threshold-1) = Reduced Mode, ab threshold = Full Mode.",
    "minimum": 1,
    "path": "assumptions.frames_reduced_threshold",
    "source": "schema",
    "type": "integer",
    "yaml_default": 200
  },
  {
    "category": "assumptions",
    "description": "Pipeline behavior profile: practical keeps compatibility defaults, strict enforces v3.3.6 normative constraints.",
    "enum": [
      "practical",
      "strict"
    ],
    "path": "assumptions.pipeline_profile",
    "source": "schema",
    "type": "string",
    "yaml_default": "strict"
  },
  {
    "category": "assumptions",
    "path": "assumptions.reduced_mode_cluster_range",
    "source": "schema",
    "type": "array",
    "yaml_default": [
      3,
      10
    ]
  },
  {
    "category": "assumptions",
    "description": "Wenn true, werden STATE_CLUSTERING und SYNTHETIC_FRAMES im Reduced-/Emergency-Mode uebersprungen.",
    "path": "assumptions.reduced_mode_skip_clustering",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "astrometry",
    "path": "astrometry.astap_bin",
    "source": "schema",
    "type": "string",
    "yaml_default": ""
  },
  {
    "category": "astrometry",
    "path": "astrometry.astap_data_dir",
    "source": "schema",
    "type": "string",
    "yaml_default": "/media/data/Astro/astap"
  },
  {
    "category": "astrometry",
    "path": "astrometry.enabled",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "astrometry",
    "maximum": 360,
    "minimum": 1,
    "path": "astrometry.search_radius",
    "source": "schema",
    "type": "integer",
    "yaml_default": 183
  },
  {
    "category": "bge",
    "minimum": 0,
    "path": "bge.autotune.alpha_flatness",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.3
  },
  {
    "category": "bge",
    "minimum": 0,
    "path": "bge.autotune.beta_roughness",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.1
  },
  {
    "category": "bge",
    "path": "bge.autotune.enabled",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "bge",
    "maximum": 0.5,
    "minimum": 0.05,
    "path": "bge.autotune.holdout_fraction",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.2
  },
  {
    "category": "bge",
    "minimum": 1,
    "path": "bge.autotune.max_evals",
    "source": "schema",
    "type": "integer",
    "yaml_default": 24
  },
  {
    "category": "bge",
    "enum": [
      "conservative",
      "extended"
    ],
    "path": "bge.autotune.strategy",
    "source": "schema",
    "type": "string",
    "yaml_default": "conservative"
  },
  {
    "category": "bge",
    "path": "bge.enabled",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "bge",
    "exclusiveMinimum": 0,
    "path": "bge.fit.huber_delta",
    "source": "schema",
    "type": "number",
    "yaml_default": 1.5
  },
  {
    "category": "bge",
    "minimum": 1,
    "path": "bge.fit.irls_max_iterations",
    "source": "schema",
    "type": "integer",
    "yaml_default": 10
  },
  {
    "category": "bge",
    "exclusiveMinimum": 0,
    "path": "bge.fit.irls_tolerance",
    "source": "schema",
    "type": "number",
    "yaml_default": "1e-4"
  },
  {
    "category": "bge",
    "enum": [
      "poly",
      "spline",
      "bicubic",
      "rbf",
      "modeled_mask_mesh"
    ],
    "path": "bge.fit.method",
    "source": "schema",
    "type": "string",
    "yaml_default": "rbf"
  },
  {
    "category": "bge",
    "enum": [
      2,
      3
    ],
    "path": "bge.fit.polynomial_order",
    "source": "schema",
    "type": "integer",
    "yaml_default": 2
  },
  {
    "category": "bge",
    "exclusiveMinimum": 0,
    "path": "bge.fit.rbf_epsilon",
    "source": "schema",
    "type": "number",
    "yaml_default": 1.0
  },
  {
    "category": "bge",
    "exclusiveMinimum": 0,
    "path": "bge.fit.rbf_lambda",
    "source": "schema",
    "type": "number",
    "yaml_default": "1e-2"
  },
  {
    "category": "bge",
    "exclusiveMinimum": 0,
    "path": "bge.fit.rbf_mu_factor",
    "source": "schema",
    "type": "number",
    "yaml_default": 1.5
  },
  {
    "category": "bge",
    "enum": [
      "thinplate",
      "multiquadric",
      "gaussian"
    ],
    "path": "bge.fit.rbf_phi",
    "source": "schema",
    "type": "string",
    "yaml_default": "multiquadric"
  },
  {
    "category": "bge",
    "enum": [
      "huber",
      "tukey"
    ],
    "path": "bge.fit.robust_loss",
    "source": "schema",
    "type": "string",
    "yaml_default": "huber"
  },
  {
    "category": "bge",
    "exclusiveMinimum": 0,
    "maximum": 1,
    "path": "bge.grid.G_max_fraction",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.25
  },
  {
    "category": "bge",
    "minimum": 1,
    "path": "bge.grid.G_min_px",
    "source": "schema",
    "type": "integer",
    "yaml_default": 64
  },
  {
    "category": "bge",
    "minimum": 1,
    "path": "bge.grid.N_g",
    "source": "schema",
    "type": "integer",
    "yaml_default": 32
  },
  {
    "category": "bge",
    "enum": [
      "discard",
      "nearest",
      "radius_expand"
    ],
    "path": "bge.grid.insufficient_cell_strategy",
    "source": "schema",
    "type": "string",
    "yaml_default": "radius_expand"
  },
  {
    "category": "bge",
    "minimum": 0,
    "path": "bge.mask.sat_dilate_px",
    "source": "schema",
    "type": "integer",
    "yaml_default": 6
  },
  {
    "category": "bge",
    "minimum": 0,
    "path": "bge.mask.star_dilate_px",
    "source": "schema",
    "type": "integer",
    "yaml_default": 6
  },
  {
    "category": "bge",
    "minimum": 1,
    "path": "bge.min_tiles_per_cell",
    "source": "schema",
    "type": "integer",
    "yaml_default": 3
  },
  {
    "category": "bge",
    "path": "bge.min_valid_sample_fraction_for_apply",
    "source": "yaml_only",
    "type": "number",
    "yaml_default": 0.28
  },
  {
    "category": "bge",
    "path": "bge.min_valid_samples_for_apply",
    "source": "yaml_only",
    "type": "integer",
    "yaml_default": 96
  },
  {
    "category": "bge",
    "exclusiveMinimum": 0,
    "maximum": 0.5,
    "path": "bge.sample_quantile",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.15
  },
  {
    "category": "bge",
    "maximum": 1,
    "minimum": 0,
    "path": "bge.structure_thresh_percentile",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.8
  },
  {
    "category": "calibration",
    "path": "calibration.bias_dir",
    "source": "schema",
    "type": "string",
    "yaml_default": ""
  },
  {
    "category": "calibration",
    "path": "calibration.bias_master",
    "source": "schema",
    "type": "string",
    "yaml_default": ""
  },
  {
    "category": "calibration",
    "path": "calibration.bias_use_master",
    "source": "schema",
    "type": "boolean",
    "yaml_default": false
  },
  {
    "category": "calibration",
    "path": "calibration.dark_auto_select",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "calibration",
    "path": "calibration.dark_master",
    "source": "schema",
    "type": "string",
    "yaml_default": ""
  },
  {
    "category": "calibration",
    "minimum": 0,
    "path": "calibration.dark_match_exposure_tolerance_percent",
    "source": "schema",
    "type": "number",
    "yaml_default": 8.0
  },
  {
    "category": "calibration",
    "minimum": 0,
    "path": "calibration.dark_match_temp_tolerance_c",
    "source": "schema",
    "type": "number",
    "yaml_default": 3.0
  },
  {
    "category": "calibration",
    "path": "calibration.dark_match_use_temp",
    "source": "schema",
    "type": "boolean",
    "yaml_default": false
  },
  {
    "category": "calibration",
    "path": "calibration.dark_use_master",
    "source": "schema",
    "type": "boolean",
    "yaml_default": false
  },
  {
    "category": "calibration",
    "path": "calibration.darks_dir",
    "source": "schema",
    "type": "string",
    "yaml_default": ""
  },
  {
    "category": "calibration",
    "path": "calibration.flat_master",
    "source": "schema",
    "type": "string",
    "yaml_default": ""
  },
  {
    "category": "calibration",
    "path": "calibration.flat_use_master",
    "source": "schema",
    "type": "boolean",
    "yaml_default": false
  },
  {
    "category": "calibration",
    "path": "calibration.flats_dir",
    "source": "schema",
    "type": "string",
    "yaml_default": ""
  },
  {
    "category": "calibration",
    "path": "calibration.pattern",
    "source": "schema",
    "type": "string",
    "yaml_default": "*.fit;*.fits;*.fts;*.fit.fz;*.fits.fz;*.fts.fz"
  },
  {
    "category": "calibration",
    "path": "calibration.use_bias",
    "source": "schema",
    "type": "boolean",
    "yaml_default": false
  },
  {
    "category": "calibration",
    "path": "calibration.use_dark",
    "source": "schema",
    "type": "boolean",
    "yaml_default": false
  },
  {
    "category": "calibration",
    "path": "calibration.use_flat",
    "source": "schema",
    "type": "boolean",
    "yaml_default": false
  },
  {
    "category": "chroma_denoise",
    "enum": [
      "pre_stack_tiles",
      "post_stack_linear"
    ],
    "path": "chroma_denoise.apply_stage",
    "source": "schema",
    "type": "string",
    "yaml_default": "post_stack_linear"
  },
  {
    "category": "chroma_denoise",
    "maximum": 1,
    "minimum": 0,
    "path": "chroma_denoise.blend.amount",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.95
  },
  {
    "category": "chroma_denoise",
    "enum": [
      "chroma_only"
    ],
    "path": "chroma_denoise.blend.mode",
    "source": "schema",
    "type": "string",
    "yaml_default": "chroma_only"
  },
  {
    "category": "chroma_denoise",
    "path": "chroma_denoise.chroma_bilateral.enabled",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "chroma_denoise",
    "exclusiveMinimum": 0,
    "path": "chroma_denoise.chroma_bilateral.sigma_range",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.065
  },
  {
    "category": "chroma_denoise",
    "exclusiveMinimum": 0,
    "path": "chroma_denoise.chroma_bilateral.sigma_spatial",
    "source": "schema",
    "type": "number",
    "yaml_default": 1.5
  },
  {
    "category": "chroma_denoise",
    "path": "chroma_denoise.chroma_wavelet.enabled",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "chroma_denoise",
    "minimum": 1,
    "path": "chroma_denoise.chroma_wavelet.levels",
    "source": "schema",
    "type": "integer",
    "yaml_default": 3
  },
  {
    "category": "chroma_denoise",
    "exclusiveMinimum": 0,
    "path": "chroma_denoise.chroma_wavelet.soft_k",
    "source": "schema",
    "type": "number",
    "yaml_default": 1.0
  },
  {
    "category": "chroma_denoise",
    "exclusiveMinimum": 0,
    "path": "chroma_denoise.chroma_wavelet.threshold_scale",
    "source": "schema",
    "type": "number",
    "yaml_default": 1.8
  },
  {
    "category": "chroma_denoise",
    "enum": [
      "ycbcr_linear",
      "opponent_linear"
    ],
    "path": "chroma_denoise.color_space",
    "source": "schema",
    "type": "string",
    "yaml_default": "ycbcr_linear"
  },
  {
    "category": "chroma_denoise",
    "path": "chroma_denoise.enabled",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "chroma_denoise",
    "maximum": 1,
    "minimum": 0,
    "path": "chroma_denoise.luma_guard_strength",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.75
  },
  {
    "category": "chroma_denoise",
    "path": "chroma_denoise.protect_luma",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "chroma_denoise",
    "minimum": 0,
    "path": "chroma_denoise.star_protection.dilate_px",
    "source": "schema",
    "type": "integer",
    "yaml_default": 2
  },
  {
    "category": "chroma_denoise",
    "path": "chroma_denoise.star_protection.enabled",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "chroma_denoise",
    "exclusiveMinimum": 0,
    "path": "chroma_denoise.star_protection.threshold_sigma",
    "source": "schema",
    "type": "number",
    "yaml_default": 2.5
  },
  {
    "category": "chroma_denoise",
    "path": "chroma_denoise.structure_protection.enabled",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "chroma_denoise",
    "maximum": 100,
    "minimum": 0,
    "path": "chroma_denoise.structure_protection.gradient_percentile",
    "source": "schema",
    "type": "number",
    "yaml_default": 87
  },
  {
    "category": "data",
    "path": "data.bayer_pattern",
    "source": "schema",
    "type": "string"
  },
  {
    "category": "data",
    "enum": [
      "OSC",
      "MONO",
      "RGB"
    ],
    "path": "data.color_mode",
    "source": "schema",
    "type": "string"
  },
  {
    "category": "data",
    "minimum": 0,
    "path": "data.image_height",
    "source": "schema",
    "type": "integer"
  },
  {
    "category": "data",
    "minimum": 0,
    "path": "data.image_width",
    "source": "schema",
    "type": "integer"
  },
  {
    "category": "data",
    "deprecated": true,
    "description": "Deprecated: non-lineare Frames werden im Runner nur noch gewarnt (warn_only), nicht entfernt.",
    "path": "data.linear_required",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "debayer",
    "path": "debayer",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "dithering",
    "path": "dithering.enabled",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "dithering",
    "minimum": 0,
    "path": "dithering.min_shift_px",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.7
  },
  {
    "category": "global_metrics",
    "path": "global_metrics.adaptive_weights",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "global_metrics",
    "path": "global_metrics.clamp",
    "source": "schema",
    "type": "array",
    "yaml_default": [
      -3.0,
      3.0
    ]
  },
  {
    "category": "global_metrics",
    "description": "Exponent scale k for G_f = exp(k * Q_f). k=1.0 (default) is standard, k>1 increases differentiation between good/bad frames. E.g. k=1.5 gives typical ratio ~20:1, k=2.0 gives ~54:1.",
    "exclusiveMinimum": 0,
    "path": "global_metrics.weight_exponent_scale",
    "source": "schema",
    "type": "number",
    "yaml_default": 1.2
  },
  {
    "category": "global_metrics",
    "maximum": 1,
    "minimum": 0,
    "path": "global_metrics.weights.background",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.4
  },
  {
    "category": "global_metrics",
    "maximum": 1,
    "minimum": 0,
    "path": "global_metrics.weights.gradient",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.25
  },
  {
    "category": "global_metrics",
    "maximum": 1,
    "minimum": 0,
    "path": "global_metrics.weights.noise",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.35
  },
  {
    "category": "input_scan",
    "path": "input.max_frames",
    "source": "yaml_only",
    "type": "integer",
    "yaml_default": 0
  },
  {
    "category": "input_scan",
    "path": "input.pattern",
    "source": "yaml_only",
    "type": "string",
    "yaml_default": "./data/*.fits"
  },
  {
    "category": "input_scan",
    "path": "input.sort",
    "source": "yaml_only",
    "type": "string",
    "yaml_default": "numeric"
  },
  {
    "category": "linearity",
    "path": "linearity.enabled",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "linearity",
    "minimum": 1,
    "path": "linearity.max_frames",
    "source": "schema",
    "type": "integer",
    "yaml_default": 8
  },
  {
    "category": "linearity",
    "maximum": 1,
    "minimum": 0,
    "path": "linearity.min_overall_linearity",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.9
  },
  {
    "category": "linearity",
    "enum": [
      "strict",
      "moderate",
      "permissive"
    ],
    "path": "linearity.strictness",
    "source": "schema",
    "type": "string",
    "yaml_default": "moderate"
  },
  {
    "category": "local_metrics",
    "path": "local_metrics.clamp",
    "source": "schema",
    "type": "array",
    "yaml_default": [
      -3.0,
      3.0
    ]
  },
  {
    "category": "local_metrics",
    "maximum": 1,
    "minimum": 0,
    "path": "local_metrics.star_mode.weights.contrast",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.2
  },
  {
    "category": "local_metrics",
    "maximum": 1,
    "minimum": 0,
    "path": "local_metrics.star_mode.weights.fwhm",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.6
  },
  {
    "category": "local_metrics",
    "maximum": 1,
    "minimum": 0,
    "path": "local_metrics.star_mode.weights.roundness",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.2
  },
  {
    "category": "local_metrics",
    "maximum": 1,
    "minimum": 0,
    "path": "local_metrics.structure_mode.background_weight",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.3
  },
  {
    "category": "local_metrics",
    "maximum": 1,
    "minimum": 0,
    "path": "local_metrics.structure_mode.metric_weight",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.7
  },
  {
    "category": "system",
    "path": "log_level",
    "source": "yaml_only",
    "type": "string",
    "yaml_default": "info"
  },
  {
    "category": "normalization",
    "path": "normalization.enabled",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "normalization",
    "enum": [
      "background",
      "median"
    ],
    "path": "normalization.mode",
    "source": "schema",
    "type": "string",
    "yaml_default": "background"
  },
  {
    "category": "normalization",
    "path": "normalization.per_channel",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "output",
    "path": "output.crop_to_nonzero_bbox",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "output",
    "path": "output.registered_dir",
    "source": "schema",
    "type": "string",
    "yaml_default": "registered"
  },
  {
    "category": "output",
    "path": "output.write_registered_frames",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "pcc",
    "exclusiveMinimum": 0,
    "path": "pcc.annulus_inner_fwhm_mult",
    "source": "schema",
    "type": "number",
    "yaml_default": 3.0
  },
  {
    "category": "pcc",
    "exclusiveMinimum": 0,
    "path": "pcc.annulus_inner_px",
    "source": "schema",
    "type": "number",
    "yaml_default": 10
  },
  {
    "category": "pcc",
    "exclusiveMinimum": 0,
    "path": "pcc.annulus_outer_fwhm_mult",
    "source": "schema",
    "type": "number",
    "yaml_default": 5.0
  },
  {
    "category": "pcc",
    "exclusiveMinimum": 0,
    "path": "pcc.annulus_outer_px",
    "source": "schema",
    "type": "number",
    "yaml_default": 16
  },
  {
    "category": "pcc",
    "exclusiveMinimum": 0,
    "path": "pcc.aperture_fwhm_mult",
    "source": "schema",
    "type": "number",
    "yaml_default": 1.8
  },
  {
    "category": "pcc",
    "exclusiveMinimum": 0,
    "path": "pcc.aperture_radius_px",
    "source": "schema",
    "type": "number",
    "yaml_default": 8
  },
  {
    "category": "pcc",
    "path": "pcc.apply_attenuation",
    "source": "yaml_only",
    "type": "boolean",
    "yaml_default": false
  },
  {
    "category": "pcc",
    "enum": [
      "median",
      "plane"
    ],
    "path": "pcc.background_model",
    "source": "schema",
    "type": "string",
    "yaml_default": "plane"
  },
  {
    "category": "pcc",
    "path": "pcc.chroma_strength",
    "source": "yaml_only",
    "type": "number",
    "yaml_default": 0.85
  },
  {
    "category": "pcc",
    "path": "pcc.enabled",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "pcc",
    "path": "pcc.k_max",
    "source": "yaml_only",
    "type": "number",
    "yaml_default": 2.4
  },
  {
    "category": "pcc",
    "maximum": 15,
    "minimum": 0,
    "path": "pcc.mag_bright_limit",
    "source": "schema",
    "type": "number",
    "yaml_default": 6
  },
  {
    "category": "pcc",
    "maximum": 22,
    "minimum": 1,
    "path": "pcc.mag_limit",
    "source": "schema",
    "type": "number",
    "yaml_default": 14
  },
  {
    "category": "pcc",
    "minimum": 1,
    "path": "pcc.max_condition_number",
    "source": "schema",
    "type": "number",
    "yaml_default": 3.0
  },
  {
    "category": "pcc",
    "exclusiveMinimum": 0,
    "path": "pcc.max_residual_rms",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.7
  },
  {
    "category": "pcc",
    "exclusiveMinimum": 0,
    "path": "pcc.min_aperture_px",
    "source": "schema",
    "type": "number",
    "yaml_default": 4.0
  },
  {
    "category": "pcc",
    "minimum": 3,
    "path": "pcc.min_stars",
    "source": "schema",
    "type": "integer",
    "yaml_default": 10
  },
  {
    "category": "pcc",
    "enum": [
      "fixed",
      "auto_fwhm"
    ],
    "path": "pcc.radii_mode",
    "source": "schema",
    "type": "string",
    "yaml_default": "auto_fwhm"
  },
  {
    "category": "pcc",
    "exclusiveMinimum": 0,
    "path": "pcc.sigma_clip",
    "source": "schema",
    "type": "number",
    "yaml_default": 2.5
  },
  {
    "category": "pcc",
    "path": "pcc.siril_catalog_dir",
    "source": "schema",
    "type": "string",
    "yaml_default": ""
  },
  {
    "category": "pcc",
    "enum": [
      "auto",
      "siril",
      "vizier_gaia",
      "vizier_apass"
    ],
    "path": "pcc.source",
    "source": "schema",
    "type": "string",
    "yaml_default": "siril"
  },
  {
    "category": "pipeline",
    "path": "pipeline.abort_on_fail",
    "source": "schema",
    "type": "boolean",
    "yaml_default": false
  },
  {
    "category": "pipeline",
    "enum": [
      "production",
      "test"
    ],
    "path": "pipeline.mode",
    "source": "schema",
    "type": "string",
    "yaml_default": "production"
  },
  {
    "category": "registration",
    "path": "registration.allow_rotation",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "registration",
    "description": "Enable additional star-pair fallback between primary triangle and normative fallback cascade. strict profile disables this.",
    "path": "registration.enable_star_pair_fallback",
    "source": "schema",
    "type": "boolean",
    "yaml_default": false
  },
  {
    "category": "registration",
    "description": "Primary registration engine. robust_phase_ecc recommended for nebula/clouds (LoG gradient preprocessing). triangle_star_matching for clear skies (faster, rotation-invariant).",
    "enum": [
      "triangle_star_matching",
      "star_similarity",
      "hybrid_phase_ecc",
      "robust_phase_ecc"
    ],
    "path": "registration.engine",
    "source": "schema",
    "type": "string",
    "yaml_default": "triangle_star_matching"
  },
  {
    "category": "registration",
    "exclusiveMinimum": 0,
    "path": "registration.reject_cc_mad_multiplier",
    "source": "schema",
    "type": "number",
    "yaml_default": 4.0
  },
  {
    "category": "registration",
    "maximum": 1,
    "minimum": 0,
    "path": "registration.reject_cc_min_abs",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.3
  },
  {
    "category": "registration",
    "path": "registration.reject_outliers",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "registration",
    "exclusiveMinimum": 0,
    "path": "registration.reject_scale_max",
    "source": "schema",
    "type": "number",
    "yaml_default": 1.08
  },
  {
    "category": "registration",
    "exclusiveMinimum": 0,
    "path": "registration.reject_scale_min",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.92
  },
  {
    "category": "registration",
    "exclusiveMinimum": 0,
    "path": "registration.reject_shift_median_multiplier",
    "source": "schema",
    "type": "number",
    "yaml_default": 5.0
  },
  {
    "category": "registration",
    "minimum": 0,
    "path": "registration.reject_shift_px_min",
    "source": "schema",
    "type": "number",
    "yaml_default": 100.0
  },
  {
    "category": "registration",
    "exclusiveMinimum": 0,
    "path": "registration.star_dist_bin_px",
    "source": "schema",
    "type": "number",
    "yaml_default": 5.0
  },
  {
    "category": "registration",
    "exclusiveMinimum": 0,
    "path": "registration.star_inlier_tol_px",
    "source": "schema",
    "type": "number",
    "yaml_default": 4.0
  },
  {
    "category": "registration",
    "minimum": 2,
    "path": "registration.star_min_inliers",
    "source": "schema",
    "type": "integer",
    "yaml_default": 4
  },
  {
    "category": "registration",
    "minimum": 3,
    "path": "registration.star_topk",
    "source": "schema",
    "type": "integer",
    "yaml_default": 150
  },
  {
    "category": "system",
    "path": "run_dir",
    "source": "yaml_only",
    "type": "string",
    "yaml_default": "./run_smart_telescope"
  },
  {
    "category": "runtime_limits",
    "description": "Erlaubt Emergency-Mode fuer Datensaetze mit <50 nutzbaren Frames; sonst wird der Lauf kontrolliert abgebrochen.",
    "path": "runtime_limits.allow_emergency_mode",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "runtime_limits",
    "exclusiveMinimum": 0,
    "path": "runtime_limits.hard_abort_hours",
    "source": "schema",
    "type": "number",
    "yaml_default": 6.0
  },
  {
    "category": "runtime_limits",
    "description": "Speicherbudget in MiB fuer OSC-Worker-Cap bei Tile-Rekonstruktion.",
    "minimum": 1,
    "path": "runtime_limits.memory_budget",
    "source": "schema",
    "type": "integer",
    "yaml_default": 2048
  },
  {
    "category": "runtime_limits",
    "description": "Anzahl paralleler Worker fuer Tile-Phasen; wird zusaetzlich durch CPU-Kerne und OSC-Speicherbudget begrenzt.",
    "minimum": 1,
    "path": "runtime_limits.parallel_workers",
    "source": "schema",
    "type": "integer",
    "yaml_default": 8
  },
  {
    "category": "runtime_limits",
    "exclusiveMinimum": 0,
    "path": "runtime_limits.tile_analysis_max_factor_vs_stack",
    "source": "schema",
    "type": "number",
    "yaml_default": 3.0
  },
  {
    "category": "stacking",
    "description": "Aktiviert optionales Dominanz-Cap fuer Clustergewichte.",
    "path": "stacking.cluster_quality_weighting.cap_enabled",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "stacking",
    "description": "Nur mit cap_enabled=true: begrenzt Gewicht auf w_k <= cap_ratio * median_j(w_j).",
    "exclusiveMinimum": 0,
    "path": "stacking.cluster_quality_weighting.cap_ratio",
    "source": "schema",
    "type": "number",
    "yaml_default": 20.0
  },
  {
    "category": "stacking",
    "description": "Aktiviert v3.2.2 Cluster-Qualitaetsgewichtung im finalen Stack: w_k = exp(kappa_cluster * Q_k).",
    "path": "stacking.cluster_quality_weighting.enabled",
    "source": "schema",
    "type": "boolean",
    "yaml_default": false
  },
  {
    "category": "stacking",
    "description": "Exponentfaktor fuer Cluster-Qualitaet Q_k (Q_k in [-3,+3]).",
    "exclusiveMinimum": 0,
    "path": "stacking.cluster_quality_weighting.kappa_cluster",
    "source": "schema",
    "type": "number",
    "yaml_default": 1.0
  },
  {
    "category": "stacking",
    "description": "Optionale Hotpixel-/Artefakt-Kosmetik nach dem Stacken (Post-Processing, standardmaessig deaktiviert).",
    "path": "stacking.cosmetic_correction",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "stacking",
    "description": "Schwellwert in Vielfachen der MAD-Sigma fuer die Hotpixel-Erkennung in cosmetic_correction. Niedrigerer Wert = aggressiver. Default: 5.0, empfohlen fuer OSC/Seestar: 10.0.",
    "exclusiveMinimum": 0,
    "path": "stacking.cosmetic_correction_sigma",
    "source": "schema",
    "type": "number",
    "yaml_default": 4.0
  },
  {
    "category": "stacking",
    "enum": [
      "rej",
      "average"
    ],
    "path": "stacking.method",
    "source": "schema",
    "type": "string",
    "yaml_default": "rej"
  },
  {
    "category": "stacking",
    "description": "Optionaler linearer Anzeige-Stretch auf den Ausgabe-Daten (Post-Processing, nicht Teil des linearen Pflichtkerns).",
    "path": "stacking.output_stretch",
    "source": "schema",
    "type": "boolean",
    "yaml_default": false
  },
  {
    "category": "stacking",
    "description": "Hotpixel-Korrektur pro Frame VOR dem Stacken. Entfernt fixe Sensordefekte (immer gleiche Position), die Sigma-Clipping nicht erkennt. Empfohlen fuer OSC-Kameras mit fixen Hot Pixeln.",
    "path": "stacking.per_frame_cosmetic_correction",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "stacking",
    "description": "Sigma-Schwelle fuer per_frame_cosmetic_correction. Niedrigerer Wert = aggressiver. Default: 5.0. Fuer Seestar/OSC empfohlen: 5.0.",
    "exclusiveMinimum": 0,
    "path": "stacking.per_frame_cosmetic_correction_sigma",
    "source": "schema",
    "type": "number",
    "yaml_default": 2.5
  },
  {
    "category": "stacking",
    "minimum": 1,
    "path": "stacking.sigma_clip.max_iters",
    "source": "schema",
    "type": "integer",
    "yaml_default": 4
  },
  {
    "category": "stacking",
    "maximum": 1,
    "minimum": 0,
    "path": "stacking.sigma_clip.min_fraction",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.4
  },
  {
    "category": "stacking",
    "exclusiveMinimum": 0,
    "path": "stacking.sigma_clip.sigma_high",
    "source": "schema",
    "type": "number",
    "yaml_default": 1.8
  },
  {
    "category": "stacking",
    "exclusiveMinimum": 0,
    "path": "stacking.sigma_clip.sigma_low",
    "source": "schema",
    "type": "number",
    "yaml_default": 1.8
  },
  {
    "category": "synthetic",
    "path": "synthetic.clustering.cluster_count_range",
    "source": "schema",
    "type": "array",
    "yaml_default": [
      3,
      12
    ]
  },
  {
    "category": "synthetic",
    "enum": [
      "kmeans",
      "quantile"
    ],
    "path": "synthetic.clustering.mode",
    "source": "schema",
    "type": "string",
    "yaml_default": "kmeans"
  },
  {
    "category": "synthetic",
    "minimum": 1,
    "path": "synthetic.frames_max",
    "source": "schema",
    "type": "integer",
    "yaml_default": 20
  },
  {
    "category": "synthetic",
    "minimum": 1,
    "path": "synthetic.frames_min",
    "source": "schema",
    "type": "integer",
    "yaml_default": 4
  },
  {
    "category": "synthetic",
    "description": "global = nur G_f; tile_weighted = W_f,t = G_f*L_f,t mit tileweiser OLA-Rekonstruktion.",
    "enum": [
      "global",
      "tile_weighted"
    ],
    "path": "synthetic.weighting",
    "source": "schema",
    "type": "string",
    "yaml_default": "tile_weighted"
  },
  {
    "category": "tile",
    "minimum": 1,
    "path": "tile.max_divisor",
    "source": "schema",
    "type": "integer",
    "yaml_default": 6
  },
  {
    "category": "tile",
    "minimum": 1,
    "path": "tile.min_size",
    "source": "schema",
    "type": "integer",
    "yaml_default": 48
  },
  {
    "category": "tile",
    "maximum": 0.5,
    "minimum": 0,
    "path": "tile.overlap_fraction",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.3
  },
  {
    "category": "tile",
    "minimum": 1,
    "path": "tile.size_factor",
    "source": "schema",
    "type": "integer",
    "yaml_default": 24
  },
  {
    "category": "tile",
    "minimum": 0,
    "path": "tile.star_min_count",
    "source": "schema",
    "type": "integer",
    "yaml_default": 5
  },
  {
    "category": "tile_denoise",
    "exclusiveMinimum": 0,
    "path": "tile_denoise.soft_threshold.alpha",
    "source": "schema",
    "type": "number",
    "yaml_default": 1.6
  },
  {
    "category": "tile_denoise",
    "minimum": 3,
    "path": "tile_denoise.soft_threshold.blur_kernel",
    "source": "schema",
    "type": "integer",
    "yaml_default": 21
  },
  {
    "category": "tile_denoise",
    "path": "tile_denoise.soft_threshold.enabled",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "tile_denoise",
    "path": "tile_denoise.soft_threshold.skip_star_tiles",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "tile_denoise",
    "path": "tile_denoise.wiener.enabled",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  },
  {
    "category": "tile_denoise",
    "minimum": 1,
    "path": "tile_denoise.wiener.max_iterations",
    "source": "schema",
    "type": "integer",
    "yaml_default": 10
  },
  {
    "category": "tile_denoise",
    "minimum": 0,
    "path": "tile_denoise.wiener.min_snr",
    "source": "schema",
    "type": "number",
    "yaml_default": 2.0
  },
  {
    "category": "tile_denoise",
    "maximum": 1,
    "minimum": 0,
    "path": "tile_denoise.wiener.q_max",
    "source": "schema",
    "type": "number",
    "yaml_default": 1.0
  },
  {
    "category": "tile_denoise",
    "minimum": -1,
    "path": "tile_denoise.wiener.q_min",
    "source": "schema",
    "type": "number",
    "yaml_default": -0.5
  },
  {
    "category": "tile_denoise",
    "exclusiveMinimum": 0,
    "path": "tile_denoise.wiener.q_step",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.1
  },
  {
    "category": "tile_denoise",
    "minimum": 0,
    "path": "tile_denoise.wiener.snr_threshold",
    "source": "schema",
    "type": "number",
    "yaml_default": 4.0
  },
  {
    "category": "validation",
    "path": "validation.max_background_rms_increase_percent",
    "source": "schema",
    "type": "number",
    "yaml_default": 2.0
  },
  {
    "category": "validation",
    "path": "validation.min_fwhm_improvement_percent",
    "source": "schema",
    "type": "number",
    "yaml_default": 3.0
  },
  {
    "category": "validation",
    "minimum": 0,
    "path": "validation.min_tile_weight_variance",
    "source": "schema",
    "type": "number",
    "yaml_default": 0.05
  },
  {
    "category": "validation",
    "path": "validation.require_no_tile_pattern",
    "source": "schema",
    "type": "boolean",
    "yaml_default": true
  }
];
