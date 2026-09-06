# Practical Configuration Examples for tile_compile

**[🇩🇪 Deutsche Version](configuration_examples_practical_de.md)**

This guide complements the configuration reference with practical examples, edge cases, and use cases based on methodology v3.3.

## Update Status (2026-07-18)

- AQMH (`aqmh.*`) fully documented with practical examples.
- HyperMetric Stretch (`hypermetric_stretch.*`) is documented as an optional post-PCC phase with `ready_to_use` and `scientific` modes.
- `bge.fit.robust_loss` and `bge.fit.huber_delta` are available again as user-facing parameters.
- New BGE apply guards `bge.min_valid_sample_fraction_for_apply` and `bge.min_valid_samples_for_apply` are documented.
- PCC examples were aligned with the current parameter set (without `pcc.method`).
- Assumptions examples were aligned with the active runtime fields (`frames_min`, `frames_reduced_threshold`, reduced-mode controls).
- Added `registration.enable_star_pair_fallback` to control the optional non-normative star-pair stage.
- `bge.tile_weight_lambda_structure` was aligned to the current default `1.0`.
- `stacking.common_overlap_required_fraction` and `stacking.tile_common_valid_min_fraction` are now documented with the current strict defaults `1.0 / 1.0`.
- The baseline snippet was updated to the strict `v3.3.9` profile.
- AQMH examples aligned with the object-agnostic v0.2.1 baseline: bounded global sigmoid weights, `resolution_divisor: 2`, `dtype: uint16`, asymmetric `2.0 / 1.5` sigma clipping with four iterations, and dual validation against the uniform control and raw AQMH baseline.

**Strict v3.3.9 baseline snippet:**

```yaml
assumptions:
  frames_min: 50
  frames_reduced_threshold: 200

registration:
  engine: triangle_star_matching
  enable_star_pair_fallback: false

stacking:
  common_overlap_required_fraction: 1.0
  tile_common_valid_min_fraction: 1.0
```

---

## AQMH (Adaptive Quality Map Harvesting) - Experimental

**When to enable:**
- High-quality sessions with strongly varying frame quality (seeing, clouds)
- When tile seams or OLA artifacts are visible
- As an alternative to the classic tile-OLA reconstruction

**Standard configuration (recommended):**

```yaml
registration:
  affine_refinement_enabled: true       # apply only after all residual/NCC/overlap gates pass
  smooth_local_refinement_enabled: true # adds held-out/Jacobian guards; otherwise preserves the prior warp
aqmh:
  enabled: true
  pyramid:
    scales: 4
    base_window_px: 4
    w_sharp: 0.6        # sharpness weight in quality index
    w_snr: 0.4          # SNR weight in quality index
    score_scale: 1.8    # local AQMH quality-map selectivity
    k_artifact: 3.0     # MAD multiplier for artifact detection
    frac_artifact_max: 0.25  # max artifact fraction per window
  storage:
    resolution_divisor: 2   # robust default; use 1 for cherry-pick/reference runs
    dtype: uint16           # use float32 for cherry-pick/reference runs
    max_resident_maps: 2
  global_quality:
    g_floor: 0.03
    g_w_sharp: 0.55
    g_w_snr: 0.30
    g_w_background_penalty: 0.25
    g_k_scale: 1.5         # bounded sigmoid temperature
  reconstruction:
    delete_prewarped_cache_after_run: true  # false to retain cache/prewarped_frames for resume
    prewarp_interpolation: cubic             # evidence-based sharpness default; linear is the low-noise fallback
    debayer_first: true                      # OSC: demosaic before PREWARP/AQMH and reconstruct RGB directly
    pre_debayer_method: edge_aware           # try bilinear for very low-SNR data if chroma artifacts appear
    rgb_q_map_mode: shared_luma
    rgb_memory_strategy: sequential
    clip_sigma: 2.0
    clip_sigma_low: 2.0
    clip_sigma_high: 2.0
    clip_iterations: 4
    min_fraction: 0.4
    min_n_eff: 2.0
    registration_weight_guard: true
    registration_weight_floor: 0.30
    registration_sequential_factor: 0.92
    registration_predicted_factor: 0.50
    structure_mask_low_q: 0.40
    structure_mask_high_q: 0.90
    structure_mask_blur_sigma_px: 4.0
  cherry_pick:
    enabled: false
  validation:
    max_seam_score_regression: 0.05
    max_fwhm_regression: 0.02
    max_background_rms_regression: 0.05
    max_tail11_abs_regression: 0.10
    max_elongation_regression: 0.08
  diagnostics:
    level: full
    tau_artifact: 0.20
    q_region: 0.75
    r_morph_canvas_px: 6
    binary_block_size_px: 64
```

**More tolerant of artifacts (satellites, clouds):**

```yaml
aqmh:
  enabled: true
  pyramid:
    k_artifact: 5.0
    frac_artifact_max: 0.35
```

**Cherry-pick auto-reject (keep most frames, reject only extreme cases):**

```yaml
aqmh:
  enabled: true
  storage:
    resolution_divisor: 1
    dtype: float32
  cherry_pick:
    enabled: true
    mode: auto_reject
    k_min_required: 20  # run-level gate and per-pixel sample floor
    reject_below_best_fraction: 0.25
    min_keep_fraction: 0.90
```

**Memory-efficient (large sessions, limited RAM):**

```yaml
aqmh:
  enabled: true
  storage:
    resolution_divisor: 4   # quarter-resolution maps
    dtype: uint8            # 8-bit quantisation
    max_resident_maps: 2
```

**Disable AQMH (revert to classic tile-OLA):**

```yaml
aqmh:
  enabled: false
```

---

## Background Gradient Extraction (BGE) - NEW in v3.3

**When to enable:**
- Visible background gradients (light pollution, moonlight)
- PCC shows color shifts across the field
- Urban/suburban imaging sites

**Recommended configuration:**

```yaml
bge:
  enabled: true
  method: classic
  autotune:
    enabled: false
    strategy: conservative
    max_evals: 24
    holdout_fraction: 0.25
    alpha_flatness: 0.25
    beta_roughness: 0.10
  tile_weight_lambda_structure: 1.0  # Current default: moderate down-weighting for structure-rich tiles
  sample_quantile: 0.20  # Conservative, resistant to faint objects
  min_valid_sample_fraction_for_apply: 0.30  # Per-channel apply guard (fraction)
  min_valid_samples_for_apply: 96  # Per-channel apply guard (absolute count)
  fit:
    method: rbf  # Flexible, recommended baseline
    robust_loss: huber  # huber | tukey
    huber_delta: 1.5
    rbf_phi: multiquadric  # Good compromise
    rbf_mu_factor: 1.0  # Standard smoothing
```

**For strong gradients (e.g. city outskirts):**

```yaml
bge:
  enabled: true
  method: classic
  sample_quantile: 0.15  # More conservative
  structure_thresh_percentile: 0.95  # Exclude more tiles
  min_valid_sample_fraction_for_apply: 0.30
  min_valid_samples_for_apply: 96
  fit:
    method: rbf
    robust_loss: tukey  # stronger outlier suppression
    rbf_phi: multiquadric
    rbf_mu_factor: 0.8  # Less smoothing for detail
```

**For large diffuse foreground objects (e.g. M31 / M42):**

```yaml
bge:
  enabled: true
  method: classic
  min_valid_sample_fraction_for_apply: 0.28  # More tolerant for dense nebulosity/star fields
  min_valid_samples_for_apply: 96
  fit:
    method: modeled_mask_mesh  # Foreground-aware mesh sky model
```

**For weak gradients (e.g. moonlight):**

```yaml
bge:
  enabled: true
  method: classic
  sample_quantile: 0.25  # Less conservative
  min_valid_sample_fraction_for_apply: 0.30
  min_valid_samples_for_apply: 96
  fit:
    method: poly  # Simpler for weak gradients
    polynomial_order: 2
```

**Select AutoBGE explicitly (planned, opt-in):**

```yaml
bge:
  enabled: true       # Legacy compatibility; method is authoritative
  method: autobge    # none | classic | autobge
  autobge:
    num_sample_points: 0
    poly_degree: 2
    rbf_smooth: 0.1
    downsample_scale: 4
    patch_size: 15
    patch_estimator: median
    stretch_mode: linear  # none | linear | mtf
    stretch_target_median: 0.25
    border_margin: 10
    bright_exclusion_fraction: 0.5
    gradient_descent_max_iters: 100
    random_seed: 42
    normalize_between_stages: true
    apply_guards: true
    mono_mode: rgb_duplicate
```

**Important:** BGE runs **before** PCC. When BGE is enabled, PCC should produce better results afterward.

**PCC v3.3.6 options (recommended with BGE):**

```yaml
pcc:
  background_model: plane      # median | plane
  max_condition_number: 3.0
  max_residual_rms: 0.35
  radii_mode: auto_fwhm        # fixed | auto_fwhm
  aperture_fwhm_mult: 1.8
  annulus_inner_fwhm_mult: 3.0
  annulus_outer_fwhm_mult: 5.0
  min_aperture_px: 4.0
  apply_attenuation: false
  chroma_strength: 1.0
  background_neutralization_mode: auto  # always | auto | off
  k_max: 3.2
```

`chroma_strength` limits the PCC color gains, not background neutralization. `auto` fully neutralizes a spatially coherent global color cast while protecting locally varying nebulosity or field color.

---

## HyperMetric Stretch after PCC

HMS is optional and runs after PCC. Keep it disabled when you only need the linear calibrated output; enable it when the run should also produce a directly viewable VeraLux-stretched RGB file.

**Ready-to-use output:**

```yaml
hypermetric_stretch:
  enabled: true
  require_successful_pcc: true
  mode: ready_to_use
  adaptive_anchor: true
  target_bg: 0.15
  log_d_mode: auto
  color_strategy: fixed
  fixed_color_strategy: 0
  output_rgb: stacked_rgb_hms.fits
```

`ready_to_use` follows the VeraLux GUI default: Auto LogD, adaptive output scaling to the target background, and final soft clip. This is the recommended mode for normal final RGB output.

**Scientific mode:**

```yaml
hypermetric_stretch:
  enabled: true
  mode: scientific
  log_d_mode: auto
  linear_expansion: 0.25
  color_grip: 1.0
  shadow_convergence: 0.0
```

`scientific` skips the final ready-to-use scaling/soft clip and allows `linear_expansion`. Use it when you want a less polished, more controlled stretch for later processing.

---

## Common overlap after PREWARP (`stacking.common_overlap_*`)

**Current sensible defaults:**

```yaml
stacking:
  common_overlap_required_fraction: 1.0
  tile_common_valid_min_fraction: 1.0
```

- `common_overlap_required_fraction: 1.0` enforces strict intersection across all usable frames.
- `tile_common_valid_min_fraction: 1.0` means a tile is only valid when its full area lies inside `COMMON_OVERLAP`.
- The tile coverage ratio is computed over the full tile area, not just the in-bounds remainder.

**Recommendations by setup:**

- **Alt/Az with field rotation:** keep `1.0 / 1.0` (recommended)
- **EQ with very stable tracking:** keep `1.0 / 1.0` when you want neutral border/background statistics
- **Only when intentionally accepting more edge area:** for example `0.98 / 0.95` or `0.95 / 0.90`

**Important:** Lower values re-admit partially covered edge pixels and edge tiles into local metrics, BGE/PCC, and background statistics.

---

## Diagnose visible tile boundaries (artifacts)

There is currently no dedicated seam-correction config block.

If you see visible tile structure, inspect `artifacts/tile_reconstruction.json` after the run and focus on:

- `tile_boundary_raw_pair_mean_abs_diff_p95`
- `tile_boundary_normalized_pair_mean_abs_diff_p95`
- `tile_boundary_pair_mean_abs_diff_p95`
- `tile_boundary_post_background_delta_p95_abs`
- `tile_boundary_post_snr_delta_p95_abs`
- `tile_boundary_top_pairs`
- `tile_norm_scale`

Interpretation:

- high `tile_boundary_raw_pair_mean_abs_diff_*` values indicate that neighboring tiles already differ before the optional tile normalization
- if `tile_boundary_normalized_pair_mean_abs_diff_*` is much higher than the raw value, the per-tile normalization is amplifying the seam
- high `tile_boundary_post_background_delta_*` values indicate strong tile-to-tile background drift
- high `tile_boundary_post_snr_delta_*` values suggest support / quality divergence between neighboring tiles
- `tile_boundary_top_pairs` shows the worst offending neighbors including tile indices, grid positions, valid counts, fallback flags, and post metrics
- inspect `tile_norm_scale` and `tile_norm_bg_*` at those tile indices to see whether the normalization itself is splitting the tile population

If the tile pattern is visible and these boundary diagnostics are also high, check first:

- `tile.overlap_fraction`
- `tile_denoise.*`
- `stacking.output_stretch`
- downstream differences introduced by `BGE` or `PCC`

---

## Hot pixels / RGB single-pixel artifacts (fixed sensor defects)

If the final image still shows **isolated red/green/blue single pixels**, these are typically **fixed hot pixels** (sensor defects) that occur at the same coordinates in every frame. They can survive stack sigma clipping because they are not outliers across frames.

**Recommendation:** Correct hot pixels **per frame before stacking**.

```yaml
stacking:
  per_frame_cosmetic_correction: true
  per_frame_cosmetic_correction_sigma: 5.0
```

Optionally keep an additional very conservative post-stack cosmetic pass:

```yaml
stacking:
  cosmetic_correction: true
  cosmetic_correction_sigma: 10.0
```

---

## Audit Note on Legacy Parameters

During the code/schema audit, several outdated example parameters were removed or replaced.

Removed legacy keys included:
- `tile.size`, `tile.overlap`, `tile.min_valid_fraction`
- `registration.method`, `registration.max_rotation_deg`, `registration.fallback_to_identity`, `registration.identity_correlation_threshold`, `registration.trail_endpoint_enabled`
- `global_metrics.fwhm_percentile`, `global_metrics.fwhm_outlier_sigma`, `global_metrics.use_robust_background`
- `local_metrics.sharpness_method`, `local_metrics.sharpness_kernel_size`, `local_metrics.sharpness_percentile`, `local_metrics.contrast_percentile`
- the old standalone `reconstruction.*` block
- `runtime.min_frames`, `runtime.allow_reduced_mode`, `runtime.max_memory_gb`, `runtime.use_disk_cache`
- `data.mode`
- `output.write_tile_weights`, `output.write_quality_maps`

The practical examples below now use only parameters that are active in the current code and schema.

---

## Tile Generation (`tile.*`)

Tile generation is now **adaptive**. Instead of a fixed `tile.size`, the runner derives tiles from `tile.size_factor`, `tile.min_size`, `tile.max_divisor`, and `tile.overlap_fraction`.

**Short focal length / good seeing:**
```yaml
tile:
  size_factor: 24
  min_size: 48
  max_divisor: 6
  overlap_fraction: 0.30
```

**General-purpose / close to defaults:**
```yaml
tile:
  size_factor: 32
  min_size: 64
  max_divisor: 6
  overlap_fraction: 0.25
```

**Long focal length / large structures / poor seeing:**
```yaml
tile:
  size_factor: 40
  min_size: 96
  max_divisor: 5
  overlap_fraction: 0.30
```

**Alt/Az with strict edge handling:**
```yaml
tile:
  size_factor: 24
  min_size: 48
  max_divisor: 6
  overlap_fraction: 0.30

stacking:
  common_overlap_required_fraction: 1.0
  tile_common_valid_min_fraction: 1.0
```

---

## Registration (`registration.*`)

The active key is `registration.engine`, not `registration.method`.

**Strict / methodology-aligned:**
```yaml
registration:
  engine: triangle_star_matching
  enable_star_pair_fallback: false
  allow_rotation: true
```

**Alt/Az / field rotation / difficult star fields:**
```yaml
registration:
  engine: triangle_star_matching
  allow_rotation: true
  enable_star_pair_fallback: true
  star_topk: 150
  star_min_inliers: 4
  star_inlier_tol_px: 4.0
  star_dist_bin_px: 5.0
  max_shift_px: 80
  reject_outliers: true
  reject_cc_min_abs: 0.25
  reject_shift_px_min: 100.0
  reject_shift_median_multiplier: 5.0
  reject_scale_min: 0.92
  reject_scale_max: 1.08
  # Legacy compatibility; no effect with independent_global_consensus_v2
  max_blind_chain_depth: 0
  blind_chain_strong_anchor_cc: 0.08
  blind_chain_drift_threshold_px: 2.0
  use_astrometry: true            # Astrometric rescue when needed
  enable_local_background_subtraction: false
  star_shift_radius_px: 200       # Alt/Az: 200-400, equatorial: 60
  affine_refinement_enabled: true  # gated; rejection preserves the global warp
  smooth_local_refinement_enabled: true # held-out/Jacobian/NCC-gated; MONO or debayer-first
```

**Star-poor / nebula-heavy / cloudy data:**
```yaml
registration:
  engine: robust_phase_ecc
  allow_rotation: true
  max_shift_px: 80
  reject_outliers: true
  # Legacy compatibility; no effect with independent_global_consensus_v2
  max_blind_chain_depth: 0
  blind_chain_strong_anchor_cc: 0.08
  blind_chain_drift_threshold_px: 2.0
  use_astrometry: true
  enable_local_background_subtraction: true  # For moonlight/gradients
  star_shift_radius_px: 200
```

**Well-tracked equatorial mount:**
```yaml
registration:
  engine: triangle_star_matching
  allow_rotation: true
  max_shift_px: 30
  # New parameters (v2.0) — defaults
  max_blind_chain_depth: 0
  blind_chain_strong_anchor_cc: 0.08
  blind_chain_drift_threshold_px: 2.0
  use_astrometry: true
  enable_local_background_subtraction: false
  star_shift_radius_px: 60        # Equatorial with good tracking
```

**Practical profile: M104 / Alt-Az / somewhat stronger rotation / poor seeing:**
```yaml
registration:
  engine: triangle_star_matching
  auto_engine: true
  transform_model: affine
  enable_star_pair_fallback: true
  allow_rotation: true
  star_topk: 150
  star_min_inliers: 4
  star_inlier_tol_px: 4.0
  star_shift_radius_px: 200
  reject_outliers: true
  reject_cc_min_abs: 0.25
  use_astrometry: true
  enable_local_background_subtraction: true

global_metrics:
  adaptive_weights: true
  weight_exponent_scale: 1.3
  clamp: [-2.5, 2.5]
```

- Full example file: [`m104.example.yaml`](https://github.com/jeamy/tile_compile/blob/master/tile_compile_cpp/examples/m104.example.yaml)
- Intent of this profile: keep the multi-anchor Alt/Az registration path active, retain weak frames, but weight clearly better frames more strongly in the global ranking.

---

## Global Weighting (`global_metrics.*`)

Global weighting now uses the three metric weights `background`, `noise`, `gradient` plus `adaptive_weights`, `clamp`, and `weight_exponent_scale`.

**Balanced / near-default:**
```yaml
global_metrics:
  adaptive_weights: true
  weight_exponent_scale: 1.2
  weights:
    background: 0.40
    noise: 0.35
    gradient: 0.25
  clamp: [-3.0, 3.0]
```

**Stronger separation between good and bad frames:**
```yaml
global_metrics:
  adaptive_weights: true
  weight_exponent_scale: 1.3
  weights:
    background: 0.40
    noise: 0.35
    gradient: 0.25
  clamp: [-2.5, 2.5]
```

- Recommended when seeing or transparency varies noticeably across the session.
- This stronger separation is also used in [`m104.example.yaml`](https://github.com/jeamy/tile_compile/blob/master/tile_compile_cpp/examples/m104.example.yaml).

**Softer weighting for homogeneous sessions:**
```yaml
global_metrics:
  adaptive_weights: false
  weight_exponent_scale: 0.8
```

---

## Local Weighting (`local_metrics.*`)

Instead of old sharpness-kernel and percentile controls, the live knobs are `k_local`, neighborhood normalization, spatial regularization, and the STAR/STRUCTURE weight splits.

**Default-like / robust:**
```yaml
local_metrics:
  clamp: [-3.0, 3.0]
  k_local: 1.0
  neighborhood_normalization:
    enabled: true
    radius: 1
    blend: 0.5
  spatial_regularization:
    enabled: true
    lambda: 0.35
    passes: 1
```

**Stronger local differentiation:**
```yaml
local_metrics:
  k_local: 1.5
```

**Softer local weighting:**
```yaml
local_metrics:
  k_local: 0.7
```

**Favor star-driven local scoring:**
```yaml
local_metrics:
  star_mode:
    weights:
      fwhm: 0.7
      roundness: 0.2
      contrast: 0.1
```

**Favor diffuse-structure scoring:**
```yaml
local_metrics:
  structure_mode:
    metric_weight: 0.7
    background_weight: 0.3
```

---

## Frame Count and Modes (`assumptions.*`, `synthetic.*`, `runtime_limits.*`)

Mode switching is now controlled by `assumptions.frames_min` and `assumptions.frames_reduced_threshold`, not by an older `runtime.min_frames` block.

**Full mode (N >= 200):**
```yaml
assumptions:
  frames_min: 50
  frames_reduced_threshold: 200
  reduced_mode_skip_clustering: false

synthetic:
  weighting: tile_weighted
  frames_min: 4
  frames_max: 20
  clustering:
    mode: kmeans
    cluster_count_range: [3, 12]
```

**Reduced mode (50 <= N < 200):**
```yaml
assumptions:
  frames_min: 50
  frames_reduced_threshold: 200
  reduced_mode_skip_clustering: true
  reduced_mode_cluster_range: [5, 10]
```

**Emergency mode (intentional only):**
```yaml
runtime_limits:
  allow_emergency_mode: true

stacking:
  common_overlap_required_fraction: 1.0
  tile_common_valid_min_fraction: 1.0
  sigma_clip:
    sigma_low: 2.5
    sigma_high: 2.5
    max_iters: 2
```

**Warning:** `allow_emergency_mode` is for rescue/test runs, not normal production.

---

## Camera-Specific Notes (`data.*`, `pcc.*`)

The active color-mode key is `data.color_mode`, not `data.mode`.

**OSC / Bayer camera:**
```yaml
data:
  color_mode: OSC
  bayer_pattern: auto

pcc:
  enabled: true
  source: auto
  background_model: plane
  radii_mode: auto_fwhm
```

**Mono:**
```yaml
data:
  color_mode: MONO
```

---

## Performance Optimization (`pipeline.*`, `runtime_limits.*`, `output.*`)

**Fast debug run:**
```yaml
pipeline:
  mode: test

linearity:
  max_frames: 4

runtime_limits:
  parallel_workers: 2
  memory_budget: 256
  acceleration_backend: cpu

output:
  write_registered_frames: false
```

**Production / high quality:**
```yaml
pipeline:
  mode: production

runtime_limits:
  parallel_workers: 8
  memory_budget: 4096
  acceleration_backend: auto
  hard_abort_hours: 6.0

output:
  write_registered_frames: true
```

**Memory-limited:**
```yaml
runtime_limits:
  parallel_workers: 2
  memory_budget: 256
  acceleration_backend: cpu

output:
  write_registered_frames: false
```

---

## Summary: Typical Setups

### DWARF II / Seestar S50

```yaml
data:
  color_mode: OSC
  bayer_pattern: auto

tile:
  size_factor: 24
  min_size: 48
  max_divisor: 6
  overlap_fraction: 0.30

registration:
  engine: triangle_star_matching
  enable_star_pair_fallback: true
  allow_rotation: true
  max_shift_px: 80
  star_shift_radius_px: 200       # Alt/Az: shift search radius for multi-hour sessions

stacking:
  common_overlap_required_fraction: 1.0
  tile_common_valid_min_fraction: 1.0
  per_frame_cosmetic_correction: true
  per_frame_cosmetic_correction_sigma: 2.5

pcc:
  enabled: true
  source: auto
```

### DSLR on equatorial mount

```yaml
data:
  color_mode: OSC
  bayer_pattern: auto

tile:
  size_factor: 36
  min_size: 96
  max_divisor: 6
  overlap_fraction: 0.35

registration:
  engine: triangle_star_matching
  allow_rotation: true
  max_shift_px: 40

global_metrics:
  adaptive_weights: false
  weight_exponent_scale: 1.0

pcc:
  enabled: true
```

Ready-to-use repository profiles:
- `tile_compile_cpp/examples/ic434.example.yaml`
- `tile_compile_cpp/examples/m31_background_gradient_balanced.example.yaml`

### Mono on a large telescope

```yaml
data:
  color_mode: MONO

tile:
  size_factor: 40
  min_size: 96
  max_divisor: 5
  overlap_fraction: 0.30

registration:
  engine: triangle_star_matching
  allow_rotation: true
  max_shift_px: 20

local_metrics:
  k_local: 1.2
  structure_mode:
    metric_weight: 0.7
    background_weight: 0.3
```

## Raw Stack / Preprocessing

Raw Stack uses a separate preprocessing configuration through the GUI/API, not the normal `tile_compile.yaml` main pipeline. Input directories and calibration frames are selected in the GUI with the same controls as `Input & Scan`.

### CFA/OSC with calibration and default postprocess

```json
{
  "mode": "linear_prestack",
  "lights_dir": "/data/session/lights",
  "bias_dir": "/data/session/bias",
  "darks_dir": "/data/session/darks",
  "flats_dir": "/data/session/flats",
  "input_mode": "cfa_osc",
  "raw_formats": "tile_compile",
  "bayer_pattern": "auto",
  "cfa_mode": "tile_compile",
  "calibration": {
    "use_bias": true,
    "use_dark": true,
    "use_flat": true,
    "dark_auto_select": true
  },
  "quality_filter": {
    "mode": "auto",
    "min_stars": 30,
    "max_fwhm_sigma": 2.0,
    "max_eccentricity": 0.65,
    "min_correlation": 0.75
  },
  "rejection": {
    "method": "sigma",
    "low": 3.0,
    "high": 3.0
  },
  "stacking": {
    "normalization": "addscale",
    "weighting": "quality"
  },
  "postprocess": {
    "astrometry": true,
    "bge": true,
    "pcc": true,
    "hypermetric_stretch": true
  },
  "hypermetric_stretch": {
    "require_successful_pcc": true,
    "mode": "ready_to_use",
    "sensor_profile": "rec709",
    "fallback_profile": "rec709",
    "target_bg": 0.15,
    "output_rgb": "stacked_rgb_hms.fits"
  },
  "report": {
    "detailed": true,
    "formats": ["json", "markdown", "html"]
  }
}
```

### Mono without calibration frames

```json
{
  "mode": "linear_prestack",
  "lights_dir": "/data/session/mono_lights",
  "input_mode": "mono",
  "raw_formats": "tile_compile",
  "bayer_pattern": "auto",
  "mono_mode": "auto",
  "quality_filter": {
    "mode": "relaxed",
    "min_stars": 15,
    "min_correlation": 0.65
  },
  "stacking": {
    "normalization": "median",
    "weighting": "quality"
  },
  "postprocess": {
    "astrometry": true,
    "bge": true,
    "pcc": true,
    "hypermetric_stretch": true
  }
}
```

---

These examples now reflect the active parameters in code and schema (`v3.3.9` status) and stay closer to the maintained repository profiles.

Adjust values to your specific hardware and conditions.

## Forward drizzle: streaming and memory budget (development, 2026-09-05)

The new CPU coverage/Uniform path processes target stripes instead of full-canvas
accumulators per frame or worker. It is not yet a released full reconstruction or
resume pipeline. The preview remains disabled by default.

| Parameter | Units, range and default | Behavior |
|---|---|---|
| `reconstruction.drizzle.memory_budget_mb` | MiB, integer >=0, default 0 | 0 inherits `runtime_limits.memory_budget`; direct library calls use 512 MiB. Accounts for retained output/masks, one source plus transient load copy, stripe scratch and reserve. Available host/cgroup headroom can further reduce the budget. |
| `reconstruction.drizzle.chunk_rows` | internal target rows, integer >=0, default 0 | Auto selects at most 256 rows within budget. Oversized explicit values fail; if one row cannot fit, allocation is rejected before large buffers are created. |
| `reconstruction.drizzle.chunk_halo_rows` | rows, integer >=-1, default -1 | Compatibility field. Exact source-footprint enumeration includes droplets crossing stripe boundaries; CPU Uniform/coverage does not need duplicate output halo rows. |
| `reconstruction.common_overlap_required_fraction` | fraction, (0,1], default 1 | Fraction of accepted dense frame footprints defining an independent analysis region, not intersection of sparse R/G/B droplets. |
| `reconstruction.diagnostics.preview_forward_drizzle_uniform` | boolean, default false | Streaming summary diagnostic, not a finished stack or resume commit. |

Coverage and Uniform share the polygon kernel. `n_eff=(sum B)^2/sum(B^2)` uses
geometric frame weights; missing support counts as zero within the analysis region.
An empty analysis region fails its gate. Production coverage retains only two full
byte masks; frame buffers are striped. Exact percentiles use temporary float spools
and a bounded read buffer, at most approximately `4 * active_channels * internal_pixels`
bytes on disk, with an additional 64 MiB free-space requirement. Hole detection
uses two scanlines; FITS mask export uses one float row instead of a float image.

The CPU reference uses one worker and fixed frame order. Sources may be reloaded
per stripe; affine source rows are geometrically bounded, local warps conservatively
revisited. Extra I/O trades for bounded RAM. Diagnostics report `estimated_peak_bytes`,
`resolved_chunk_rows` and `workers_used`. This is an allocation estimate, not measured
whole-process RSS; existing registration data and concurrent processes require
separate accounting. There is no automatic method or scale fallback. See
`tile_compile_cpp/examples/forward_drizzle_streaming.example.yaml`.

For shared Uniform/Raw library calls, also prefer `chunk_rows: 0`: candidate storage grows with the number of frames, so more frames may require smaller stripes at the same image dimensions. If one row cannot fit, the call fails early. The streaming API avoids retaining both complete outputs in RAM; its sink must consume stripes immediately. This API is not yet a complete new runner path.

`reconstruction.diagnostics.persist_forward_drizzle_uniform_store` (boolean, default `false`) is independent of preview. When enabled, it streams unclipped Uniform planes into `artifacts/forward_drizzle_uniform_store/generation-…/`; `current.json` publishes the complete verified generation atomically. The existing drizzle budget includes an additional 8 MiB FITS/metadata reserve and one float row. Insufficient memory fails before source loading; insufficient free disk fails before plane writing. A failed diagnostic does not fail the legacy run. Old generations are retained and consume disk; there is no automatic cleanup. This is a diagnostic store, not a resumable pipeline phase. Read `current.json` and validate it against the expected source, sampling and algorithm identity; old flat stores are not implicitly accepted or rewritten. The shared clipped Uniform/Raw library store uses the same transaction but is not yet wired into a new runner phase.

The checked predecessor library API uses an explicit source-quality MiB budget (512 MiB by default). It may reject large native frames under its conservative scratch estimate; do not bypass that check. Cache manifests identify existing normalized raw float files and do not perform calibration. Commit schema 2 binds cache and quality-plan hashes. Production runner cache retention and resume integration remain pending.
