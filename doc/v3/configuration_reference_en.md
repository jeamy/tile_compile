# Tile-Compile C++ Configuration Reference

This documentation describes all configuration options for `tile_compile.yaml` based on the C++ implementation in `configuration.hpp` and the schema files `tile_compile.schema.json` / `tile_compile.schema.yaml`.

**Source of truth for defaults:** `include/tile_compile/config/configuration.hpp`  
**Schema version:** v3  
**Reference:** Methodology v3.2

**💡 For practical examples and use cases, see:** [Configuration Examples & Best Practices](configuration_examples_practical_en.md)

## Table of Contents

1. [Pipeline](#1-pipeline)
2. [Output](#2-output)
3. [Data](#3-data)
4. [Linearity](#4-linearity)
5. [Calibration](#5-calibration)
6. [Assumptions](#6-assumptions)
7. [Normalization](#7-normalization)
8. [Registration](#8-registration)
9. [Tile Denoise](#9-tile-denoise)
10. [Global Metrics](#10-global-metrics)
11. [Tile](#11-tile)
12. [Local Metrics](#12-local-metrics)
13. [Synthetic](#13-synthetic)
14. [Reconstruction](#14-reconstruction)
15. [Debayer](#15-debayer)
16. [Astrometry](#16-astrometry)
17. [PCC](#17-pcc)
18. [Stacking](#18-stacking)
19. [Validation](#19-validation)
20. [Runtime Limits](#20-runtime-limits)

---

## 1. Pipeline

Basic pipeline control.

### `pipeline.mode`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `production`, `test` |
| **Default** | `"production"` |

**Purpose:** Determines the execution mode of the pipeline.

- **`production`**: Complete processing with all quality checks and phases
- **`test`**: Reduced processing for testing purposes (may skip some validations)

### `pipeline.abort_on_fail`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Stop immediately when a critical phase fails.

- **`true`**: production-safe behavior (recommended)
- **`false`**: continue for diagnostics/debug runs

---

## 2. Output

Output file and directory configuration.

### `output.registered_dir`

| Property | Value |
|----------|-------|
| **Type** | string |
| **Default** | `"registered"` |

**Purpose:** Subdirectory for registered frames (under `runs/<run_id>/outputs/`).

### `output.artifacts_dir`

| Property | Value |
|----------|-------|
| **Type** | string |
| **Default** | `"artifacts"` |

**Purpose:** Subdirectory for JSON artifacts and reports.

### `output.write_registered_frames`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Persist registered frames as FITS (`reg_XXXXX.fit`).

### `output.write_global_metrics`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Write `global_metrics.json`.

### `output.write_global_registration`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Write `global_registration.json` (warp + cc per frame).

---

## 3. Data

Input data configuration.

### `data.image_width`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `0` (auto-detected) |

**Purpose:** Optional expected image width in pixels.

### `data.image_height`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `0` (auto-detected) |

**Purpose:** Optional expected image height in pixels.

### `data.frames_min`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `0` |

**Purpose:** Expected minimum number of input frames (`0` disables this check).

### `data.frames_target`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `0` |

**Purpose:** Informational target number of frames.

### `data.color_mode`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `OSC`, `MONO`, `RGB` |
| **Default** | `"OSC"` |

**Purpose:** Expected camera color mode.

### `data.bayer_pattern`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `RGGB`, `BGGR`, `GRBG`, `GBRG`, `NONE` |
| **Default** | `"RGGB"` |

**Purpose:** Bayer matrix pattern for color filter arrays. `NONE` for monochrome data.

### `data.linear_required`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Require linear (unstretched) input data.

---

## 4. Linearity

Linearity correction settings.

### `linearity.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enable/disable linearity correction.

### `linearity.max_frames`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `8` |

**Purpose:** Number of frames used for linearity diagnostics.

### `linearity.min_overall_linearity`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `0.9` |

**Purpose:** Minimum acceptable global linearity score.

### `linearity.strictness`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `strict`, `warn`, `off` |
| **Default** | `"strict"` |

**Purpose:** Controls whether linearity violations fail, warn, or are ignored.

---

## 5. Calibration

Calibration frame processing.

### `calibration.use_bias` / `calibration.use_dark` / `calibration.use_flat`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enable per-frame-type calibration stages.

### `calibration.bias_use_master` / `calibration.dark_use_master` / `calibration.flat_use_master`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Use prebuilt master calibration files instead of stacking directories.

### `calibration.dark_auto_select`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Auto-select matching darks by exposure (and optionally temperature).

### `calibration.bias_dir` / `calibration.darks_dir` / `calibration.flats_dir`

| Property | Value |
|----------|-------|
| **Type** | string |
| **Default** | `""` (disabled) |

**Purpose:** Input directories for calibration stacks.

### `calibration.bias_master` / `calibration.dark_master` / `calibration.flat_master`

| Property | Value |
|----------|-------|
| **Type** | string |
| **Default** | `""` (disabled) |

**Purpose:** Paths to precomputed master calibration files.

---

## 6. Assumptions

Physical assumptions about the data.

### `assumptions.frames_min`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `50` |

**Purpose:** Minimum usable frame count for normal mode.

### `assumptions.frames_optimal`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `800` |

**Purpose:** Informational target for best quality/stability.

### `assumptions.frames_reduced_threshold`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `200` |

**Purpose:** Threshold for reduced mode decisions.

### `assumptions.exposure_time_tolerance_percent`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `5.0` |

**Purpose:** Allowed exposure mismatch within one sequence.

### `assumptions.reduced_mode_skip_clustering`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Skip expensive clustering in reduced mode.

### `assumptions.reduced_mode_cluster_range`

| Property | Value |
|----------|-------|
| **Type** | array [2 integers] |
| **Default** | `[5, 20]` |

**Purpose:** Cluster range fallback when reduced mode still runs clustering.

---

## 7. Normalization

Frame normalization settings.

### `normalization.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enable/disable frame normalization.

### `normalization.mode`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `background`, `none` |
| **Default** | `"background"` |

**Purpose:** Normalization method.

- **`background`**: robust background matching (recommended)
- **`none`**: disabled (not recommended for production)

### `normalization.per_channel`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Normalize channels independently for OSC/RGB data.

---

## 8. Registration

Image registration settings.

### `registration.engine`

| Property | Value |
|----------|---------|
| **Type** | string (enum) |
| **Values** | `triangle_star_matching`, `star_similarity`, `hybrid_phase_ecc`, `robust_phase_ecc` |
| **Default** | `"triangle_star_matching"` |

**Purpose:** Primary registration engine. The runner always runs a **6-stage fallback cascade**; this key selects the preferred first stage.

| Engine | Description | Strength |
|--------|-------------|----------|
| **`triangle_star_matching`** | Triangle asterism matching | **Rotation-invariant**, ideal for Alt/Az, clear sky |
| **`star_similarity`** | Star-pair distance matching | Fast for small offsets |
| **`hybrid_phase_ecc`** | Phase correlation + ECC | No star detection needed, for nebulae |
| **`robust_phase_ecc`** | LoG gradient preprocessing + pyramid Phase+ECC | **Recommended for clouds/nebula**, removes gradients before correlation |

**Cascade (always):** Triangle Stars → Star Pairs → Trail Endpoints → AKAZE Features → Robust Phase+ECC → Hybrid Phase+ECC → Identity fallback

**Temporal-Smoothing (v3.2.3+, automatically active):** When direct registration `i→ref` fails, the runner automatically tries:
1. `i→(i-1)→ref` — register to previous frame, then chain warps
2. `i→(i+1)→ref` — register to next frame, then chain warps

All chained warps are validated with NCC against the reference frame. Particularly effective for continuous field rotation (Alt/Az near pole) and clouds/nebula. Logs: `[REG-TEMPORAL]`

**Adaptive Star Detection (v3.2.3+, automatically active):** When fewer than `star_topk / 2` stars are detected, a second detection pass with a lower threshold (2.5σ instead of 3.5σ) is automatically performed.

### `registration.allow_rotation`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Allow rotation in global registration (required for Alt/Az).

### `registration.star_topk`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `120` |

**Purpose:** Number of strongest stars used for star-based matching.

### `registration.star_min_inliers`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `6` |

**Purpose:** Minimum inlier matches required for acceptance.

### `registration.star_inlier_tol_px`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `2.5` |

**Purpose:** Inlier tolerance in pixels for transformed star matches.

### `registration.star_dist_bin_px`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `2.5` |

**Purpose:** Distance histogram bin size in `star_similarity`.

### `registration.reject_outliers`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enable robust post-checks for implausible global warps.

### `registration.reject_cc_min_abs`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `0.35` |

**Purpose:** Absolute minimum correlation coefficient threshold.

### `registration.reject_cc_mad_multiplier`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `4.0` |

**Purpose:** MAD-based robustness for CC outlier threshold.

### `registration.reject_shift_px_min`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `25.0` |

**Purpose:** Fixed minimum shift threshold for rejection logic.

### `registration.reject_shift_median_multiplier`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `3.0` |

**Purpose:** Multiplier for robust shift threshold (`multiplier * median_shift`).

### `registration.reject_scale_min` / `registration.reject_scale_max`

| Property | Value |
|----------|-------|
| **Type** | number / number |
| **Default** | `0.92` / `1.08` |

**Purpose:** Allowed similarity scale range. Warps outside `[reject_scale_min, reject_scale_max]` or with negative determinant (reflection) are rejected.

---

## 8b. Dithering

### `dithering.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Mark session as dithered. Enables dither diagnostics (`detected_count`/`fraction`) in `global_registration.json`.

### `dithering.min_shift_px`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `0.5` |

**Purpose:** Minimum shift in pixels to count a frame as dithered.

---

## 9. Tile Denoise

Tile-based denoising settings.

### `tile_denoise.soft_threshold.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enable soft-threshold denoising (default active path).

### `tile_denoise.soft_threshold.blur_kernel`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `31` |

**Purpose:** Blur radius used for local noise estimate.

### `tile_denoise.soft_threshold.alpha`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `1.5` |

**Purpose:** Soft-threshold aggressiveness.

### `tile_denoise.soft_threshold.skip_star_tiles`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Avoid denoising in STAR tiles to protect stellar detail.

### `tile_denoise.wiener.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Optional Wiener denoise branch (off by default).

### `tile_denoise.wiener.snr_threshold`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `5.0` |

**Purpose:** SNR threshold; tiles above this are typically not filtered.

### `tile_denoise.wiener.q_min` / `tile_denoise.wiener.q_max`

| Property | Value |
|----------|-------|
| **Type** | number / number |
| **Default** | `-0.5` / `1.0` |

**Purpose:** Quality parameter search range for Wiener optimization.

### `tile_denoise.wiener.q_step`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `0.1` |

**Purpose:** Step size for q-parameter optimization.

### `tile_denoise.wiener.min_snr`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `2.0` |

**Purpose:** Minimum SNR for stable Wiener estimation.

### `tile_denoise.wiener.max_iterations`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `10` |

**Purpose:** Maximum iterations for Wiener optimization.

---

## 10. Global Metrics

Global quality metrics.

### `global_metrics.weights.background`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `0.4` |

**Purpose:** Weight for background penalty term.

### `global_metrics.weights.noise`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `0.3` |

**Purpose:** Weight for noise penalty term.

### `global_metrics.weights.gradient`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `0.3` |

**Purpose:** Weight for structure/sharpness term.

### `global_metrics.clamp`

| Property | Value |
|----------|-------|
| **Type** | array [2 numbers] |
| **Default** | `[-3.0, 3.0]` |

**Purpose:** Clamp range before exponential weighting.

### `global_metrics.adaptive_weights`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Auto-adjust global metric weights by variance.

### `global_metrics.weight_exponent_scale`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `1.0` |

**Purpose:** Exponential scaling factor for global weight separation.

---

## 11. Tile

Tile processing configuration.

### `tile.size_factor`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `32` |

**Purpose:** Base tile size factor (`T0 = size_factor * FWHM`).

### `tile.min_size`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `64` |

**Purpose:** Minimum tile size in pixels.

### `tile.max_divisor`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `6` |

**Purpose:** Upper tile size bound via shorter side divisor.

### `tile.overlap_fraction`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `0.25` |

**Purpose:** Fractional overlap for overlap-add blending.

### `tile.star_min_count`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `10` |

**Purpose:** Threshold for STAR vs STRUCTURE tile classification.

---

## 12. Local Metrics

Local quality metrics.

### `local_metrics.clamp`

| Property | Value |
|----------|-------|
| **Type** | array [2 numbers] |
| **Default** | `[-3.0, 3.0]` |

**Purpose:** Clamp range before local exponential weighting.

### `local_metrics.star_mode.weights.fwhm`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `0.6` |

**Purpose:** Weight of FWHM in STAR-tile local quality.

### `local_metrics.star_mode.weights.roundness`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `0.2` |

**Purpose:** Weight of roundness in STAR-tile local quality.

### `local_metrics.star_mode.weights.contrast`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `0.2` |

**Purpose:** Weight of contrast in STAR-tile local quality.

### `local_metrics.structure_mode.metric_weight`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `0.7` |

**Purpose:** Weight of structure metric in STRUCTURE-tile mode.

### `local_metrics.structure_mode.background_weight`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `0.3` |

**Purpose:** Weight of background penalty in STRUCTURE-tile mode.

---

## 13. Synthetic

Synthetic frame generation.

### `synthetic.weighting`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `global`, `tile_weighted` |
| **Default** | `"global"` |

**Purpose:** Weighting strategy for synthetic frame creation.

### `synthetic.frames_min`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `5` |

**Purpose:** Minimum cluster size required to generate synthetic output.

### `synthetic.frames_max`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `30` |

**Purpose:** Upper limit for generated synthetic frames.

### `synthetic.clustering.mode`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `kmeans`, `quantile` |
| **Default** | `"kmeans"` |

**Purpose:** Clustering method for synthetic generation.

### `synthetic.clustering.cluster_count_range`

| Property | Value |
|----------|-------|
| **Type** | array [2 integers] |
| **Default** | `[5, 30]` |

**Purpose:** Min/max cluster count range.

---

## 14. Reconstruction

Image reconstruction settings.

### `reconstruction.weighting_function`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `linear` |
| **Default** | `"linear"` |

**Purpose:** Tile blending weighting function (fixed in current runner).

### `reconstruction.window_function`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `hanning` |
| **Default** | `"hanning"` |

**Purpose:** Window function for overlap-add reconstruction.

---

## 15. Debayer

Debayering settings for OSC data.

### `debayer`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Debayer final CFA stack (OSC). For MONO data, this phase is skipped.

---

## 16. Astrometry

Astrometry solving settings.

### `astrometry.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Enable/disable astrometry solving.

### `astrometry.astap_bin`

| Property | Value |
|----------|-------|
| **Type** | string |
| **Default** | `""` |

**Purpose:** Path to ASTAP binary (empty = system/default path).

### `astrometry.astap_data_dir`

| Property | Value |
|----------|-------|
| **Type** | string |
| **Default** | `""` |

**Purpose:** Path to ASTAP data directory (empty = default path).

### `astrometry.search_radius`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Range** | `1` to `360` |
| **Default** | `180` |

**Purpose:** Search radius in degrees (`180` = blind solve).

---

## 17. PCC

Photometric Color Calibration settings.

### `pcc.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Enable/disable photometric color calibration.

### `pcc.source`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `auto`, `siril`, `vizier_gaia`, `vizier_apass` |
| **Default** | `"auto"` |

**Purpose:** Source catalog/provider for PCC.

### `pcc.mag_limit`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `14.0` |

**Purpose:** Faint magnitude limit.

### `pcc.mag_bright_limit`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `6.0` |

**Purpose:** Bright-star upper magnitude cutoff.

### `pcc.aperture_radius_px`, `pcc.annulus_inner_px`, `pcc.annulus_outer_px`

| Key | Type | Default |
|-----|------|---------|
| `pcc.aperture_radius_px` | number | `8.0` |
| `pcc.annulus_inner_px` | number | `12.0` |
| `pcc.annulus_outer_px` | number | `18.0` |

**Purpose:** Aperture/annulus geometry for star photometry.

### `pcc.min_stars`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `10` |

**Purpose:** Minimum number of valid stars required for PCC.

### `pcc.sigma_clip`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `2.5` |

**Purpose:** Outlier rejection threshold in PCC fitting.

### `pcc.siril_catalog_dir`

| Property | Value |
|----------|-------|
| **Type** | string |
| **Default** | `""` |

**Purpose:** Optional local Siril catalog path.

---

## 18. Stacking

Final stacking settings.

### `stacking.method`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `rej`, `average` |
| **Default** | `"rej"` |

**Purpose:** Final stacking method.

### `stacking.sigma_clip.sigma_low` / `stacking.sigma_clip.sigma_high`

| Property | Value |
|----------|---------|
| **Type** | number |
| **Default** | `2.0` / `2.0` |

**Purpose:** Lower/upper sigma thresholds for rejection. Pixel rejected when `z < -sigma_low` or `z > sigma_high`.

### `stacking.sigma_clip.max_iters`

| Property | Value |
|----------|---------|
| **Type** | integer |
| **Default** | `3` |

**Purpose:** Maximum sigma-clipping iterations.

### `stacking.sigma_clip.min_fraction`

| Property | Value |
|----------|---------|
| **Type** | number |
| **Default** | `0.5` |

**Purpose:** Minimum surviving frame fraction per pixel. Falls back to unclipped mean if violated.

### `stacking.cluster_quality_weighting.enabled`

| Property | Value |
|----------|---------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enable cluster-quality weighting (`w_k = exp(kappa_cluster * Q_k)`) in final combination.

### `stacking.cluster_quality_weighting.kappa_cluster`

| Property | Value |
|----------|---------|
| **Type** | number |
| **Default** | `1.0` |

**Purpose:** Exponent factor for cluster quality influence. Larger values increase separation between good/bad clusters.

### `stacking.cluster_quality_weighting.cap_enabled` / `stacking.cluster_quality_weighting.cap_ratio`

| Property | Value |
|----------|---------|
| **Type** | boolean / number |
| **Default** | `false` / `20.0` |

**Purpose:** Optional dominance cap: `w_k ≤ cap_ratio * median(w_j)` (only active when `cap_enabled=true`).

### `stacking.common_overlap_required_fraction`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `(0, 1]` |
| **Default** | `1.0` |

**Purpose:** Defines the required per-pixel frame coverage for post-PREWARP calculations.

- `1.0` (recommended default): strict common intersection (pixel must be valid in all usable frames)
- `< 1.0`: allows partially covered edge/canvas regions into statistics and tile processing

**Recommendation:** keep `1.0` for rotating-field data (Alt/Az) to avoid geometry-driven bias and stripe/grid artifacts.

### `stacking.tile_common_valid_min_fraction`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `(0, 1]` |
| **Default** | `0.9` |

**Purpose:** Tile-level acceptance threshold after common-overlap masking.

- A tile is used only if at least this fraction of its pixels belongs to the common overlap
- Higher values are stricter and reduce edge contamination

**Recommendation:** `0.9` for production, `0.75-0.85` only when deliberately accepting more edge coverage.

### `stacking.output_stretch`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Optional output stretch for preview-style output.

### `stacking.cosmetic_correction`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Optional cosmetic correction after stacking.

---

## 19. Validation

Validation and quality control.

### `validation.min_fwhm_improvement_percent`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `0.0` |

**Purpose:** Required minimum FWHM improvement in %.

### `validation.max_background_rms_increase_percent`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `0.0` |

**Purpose:** Maximum allowed background RMS increase in %.

### `validation.min_tile_weight_variance`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `0.1` |

**Purpose:** Minimum local tile-weight variance sanity threshold.

### `validation.require_no_tile_pattern`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enforce tile-pattern detector check.

---

## 20. Runtime Limits

Runtime and resource limits.

### `runtime_limits.parallel_workers`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `4` |

**Purpose:** Max worker threads for tile-heavy phases.

### `runtime_limits.memory_budget`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `512` |
| **Units** | MiB |

**Purpose:** Memory cap that can reduce effective worker parallelism (especially for OSC).

### `runtime_limits.tile_analysis_max_factor_vs_stack`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `3.0` |

**Purpose:** Warn when tile analysis exceeds this factor vs baseline stack time.

### `runtime_limits.hard_abort_hours`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `6.0` |

**Purpose:** Hard upper runtime limit in hours.

### `runtime_limits.allow_emergency_mode`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Allow processing very small datasets in emergency mode.

---

## Appendix A — Functional details for all options

This appendix provides a compact but explicit **runtime behavior** description for every configuration key.

### A.1 Pipeline / Output / Data

- `pipeline.mode`: selects production vs test control flow (same core phases, different strictness/debug posture).
- `pipeline.abort_on_fail`: controls whether `phase_end(error)` aborts run immediately.
- `output.registered_dir`: target folder name for registered frame outputs.
- `output.artifacts_dir`: target folder name for JSON artifacts (`global_metrics.json`, `tile_reconstruction.json`, etc.).
- `output.write_registered_frames`: writes per-frame registered FITS; increases IO and disk usage significantly.
- `output.write_global_metrics`: enables writing global metric vectors (frame quality diagnostics).
- `output.write_global_registration`: enables writing global warp/CC diagnostics.
- `data.image_width`, `data.image_height`: optional expected dimensions; normally auto-detected from FITS headers.
- `data.frames_min`: pre-run sanity threshold for minimum input count.
- `data.frames_target`: informational target only; does not enforce rejection by itself.
- `data.color_mode`: expected acquisition mode; runtime auto-detection can override with warning.
- `data.bayer_pattern`: CFA layout for OSC processing and color reconstruction consistency.
- `data.linear_required`: enables strict linearity requirement policy coupling with linearity diagnostics.

### A.2 Linearity / Calibration / Assumptions

- `linearity.enabled`: enables linearity diagnostics in scan/early validation.
- `linearity.max_frames`: sample size for linearity checks (tradeoff speed vs certainty).
- `linearity.min_overall_linearity`: pass/fail threshold for linearity score.
- `linearity.strictness`: policy mapping (fail/warn/ignore behavior).
- `calibration.use_bias`, `use_dark`, `use_flat`: activate master-frame correction stages.
- `calibration.bias_use_master`, `dark_use_master`, `flat_use_master`: use explicit master files vs building from directories.
- `calibration.dark_auto_select`: auto-match dark masters by exposure (and optional temperature).
- `calibration.dark_match_exposure_tolerance_percent`: allowed exposure mismatch for dark matching.
- `calibration.dark_match_use_temp`: toggles temperature-aware dark matching.
- `calibration.dark_match_temp_tolerance_c`: allowed temperature mismatch when temp matching is active.
- `calibration.bias_dir`, `darks_dir`, `flats_dir`: source folders for calibration frame discovery.
- `calibration.bias_master`, `dark_master`, `flat_master`: explicit master calibration file paths.
- `calibration.pattern`: file glob for calibration frame loading.
- `assumptions.frames_min`: minimum frame count expectation for stable methodology.
- `assumptions.frames_optimal`: target count for full-quality behavior.
- `assumptions.frames_reduced_threshold`: switch point between reduced and full pipeline behavior.
- `assumptions.exposure_time_tolerance_percent`: acceptable sub-exposure dispersion.
- `assumptions.reduced_mode_skip_clustering`: disables expensive state clustering in reduced mode.
- `assumptions.reduced_mode_cluster_range`: bounded cluster search if clustering still runs in reduced mode.

### A.3 Normalization / Registration / Dithering

- `normalization.enabled`: mandatory in methodology-driven runs (normally must stay enabled).
- `normalization.mode`: background-centric vs median-centric normalization strategy.
- `normalization.per_channel`: per-channel (OSC/RGB) normalization preserving channel balance.
- `registration.engine`: preferred first engine; runtime still executes multi-stage fallback cascade.
- `registration.allow_rotation`: permits rotational components in global warps (required for Alt/Az).
- `registration.star_topk`: number of strongest stars used by star-based engines.
- `registration.star_min_inliers`: minimum accepted inlier correspondences.
- `registration.star_inlier_tol_px`: geometric tolerance for inlier acceptance.
- `registration.star_dist_bin_px`: distance histogram quantization for star-similarity engine.
- `registration.reject_outliers`: enables robust rejection of implausible warps after matching.
- `registration.reject_cc_min_abs`: absolute NCC floor in outlier logic.
- `registration.reject_cc_mad_multiplier`: robust CC threshold scaling from MAD statistic.
- `registration.reject_shift_px_min`: absolute shift floor for shift-outlier rejection.
- `registration.reject_shift_median_multiplier`: relative shift threshold scale from median shift.
- `registration.reject_scale_min`, `reject_scale_max`: accepted similarity scale band.
- `dithering.enabled`: enables dither diagnostics output in registration artifacts.
- `dithering.min_shift_px`: minimum frame-to-frame shift to count as dither.

### A.4 Tile denoise / Chroma denoise

- `tile_denoise.soft_threshold.enabled`: enables spatial highpass soft-threshold denoise.
- `tile_denoise.soft_threshold.blur_kernel`: background estimation kernel size for residual extraction.
- `tile_denoise.soft_threshold.alpha`: denoise aggressiveness (`tau = alpha * sigma`).
- `tile_denoise.soft_threshold.skip_star_tiles`: bypass denoise on star-dominant tiles.
- `tile_denoise.wiener.enabled`: enables frequency-domain Wiener branch.
- `tile_denoise.wiener.snr_threshold`: Wiener gate; low-SNR tiles are filtered more likely.
- `tile_denoise.wiener.q_min`, `q_max`, `q_step`: internal Wiener quality search range and step.
- `tile_denoise.wiener.min_snr`: minimum accepted SNR for stable Wiener parameterization.
- `tile_denoise.wiener.max_iterations`: iterative Wiener tuning loop bound.
- `chroma_denoise.enabled`: enables chroma-focused denoise (OSC path).
- `chroma_denoise.color_space`: chroma/luma transform (`ycbcr_linear` or `opponent_linear`).
- `chroma_denoise.apply_stage`: execute before tile OLA or after final linear stack.
- `chroma_denoise.protect_luma`: protects luminance structures from chroma denoise side effects.
- `chroma_denoise.luma_guard_strength`: strength of luma protection mask.
- `chroma_denoise.star_protection.enabled`: star-mask protection for color cores/halos.
- `chroma_denoise.star_protection.threshold_sigma`: detection threshold for star mask creation.
- `chroma_denoise.star_protection.dilate_px`: star mask growth radius.
- `chroma_denoise.structure_protection.enabled`: edge/structure-aware chroma protection.
- `chroma_denoise.structure_protection.gradient_percentile`: gradient cutoff for structure mask.
- `chroma_denoise.chroma_wavelet.enabled`: enables wavelet-domain chroma attenuation.
- `chroma_denoise.chroma_wavelet.levels`: number of wavelet decomposition levels.
- `chroma_denoise.chroma_wavelet.threshold_scale`: wavelet threshold multiplier.
- `chroma_denoise.chroma_wavelet.soft_k`: softness of wavelet shrinkage.
- `chroma_denoise.chroma_bilateral.enabled`: enables bilateral smoothing on chroma components.
- `chroma_denoise.chroma_bilateral.sigma_spatial`: spatial bilateral radius/strength.
- `chroma_denoise.chroma_bilateral.sigma_range`: color-distance bilateral selectivity.
- `chroma_denoise.blend.mode`: currently chroma-only blending mode.
- `chroma_denoise.blend.amount`: blend fraction between original and denoised chroma.

### A.5 Global/local metrics / Tile / Synthetic / Reconstruction

- `global_metrics.weights.background`, `noise`, `gradient`: weighted terms composing per-frame global quality score.
- `global_metrics.clamp`: hard bounds before exponential weight mapping.
- `global_metrics.adaptive_weights`: auto-adapt metric weights from observed dispersion.
- `global_metrics.weight_exponent_scale`: controls separation strength in `exp(k*Q)` mapping.
- `tile.size_factor`: base tile size scaling from measured seeing/FWHM.
- `tile.min_size`: lower bound preventing too-small unstable tiles.
- `tile.max_divisor`: upper bound via image dimension divisor.
- `tile.overlap_fraction`: overlap ratio for overlap-add blending smoothness.
- `tile.star_min_count`: threshold for STAR vs STRUCTURE tile class.
- `local_metrics.clamp`: local quality clamp before weight conversion.
- `local_metrics.star_mode.weights.fwhm`, `roundness`, `contrast`: STAR tile quality composition.
- `local_metrics.structure_mode.metric_weight`, `background_weight`: STRUCTURE tile quality composition.
- `synthetic.weighting`: synthetic frame generation method (`global` vs `tile_weighted`).
- `synthetic.frames_min`: minimum cluster size to emit synthetic frame.
- `synthetic.frames_max`: maximum number of synthetic outputs.
- `synthetic.clustering.mode`: clustering backend for state grouping.
- `synthetic.clustering.cluster_count_range`: allowed K-search window.
- `reconstruction.weighting_function`: reconstruction weighting model (currently linear).
- `reconstruction.window_function`: overlap-add window kernel (currently Hanning).

### A.6 Debayer / Astrometry / PCC / Stacking / Validation / Runtime

- `debayer`: enables OSC CFA-to-RGB final conversion stage.
- `astrometry.enabled`: enables plate-solving stage.
- `astrometry.astap_bin`: ASTAP executable path.
- `astrometry.astap_data_dir`: ASTAP star catalog/data path.
- `astrometry.search_radius`: blind vs constrained solve radius.
- `pcc.enabled`: enables photometric color calibration.
- `pcc.source`: catalog/provider selection.
- `pcc.mag_limit`, `mag_bright_limit`: star selection magnitude limits.
- `pcc.aperture_radius_px`, `annulus_inner_px`, `annulus_outer_px`: photometric aperture geometry.
- `pcc.min_stars`: minimum matched stars for stable PCC fit.
- `pcc.sigma_clip`: outlier rejection in PCC regression.
- `pcc.siril_catalog_dir`: local Siril catalog path override.
- `stacking.method`: final combine mode (`rej` sigma-clip vs `average`).
- `stacking.sigma_clip.sigma_low`, `sigma_high`: lower/upper rejection thresholds.
- `stacking.sigma_clip.max_iters`: clipping iteration cap.
- `stacking.sigma_clip.min_fraction`: minimum retained sample ratio fallback guard.
- `stacking.cluster_quality_weighting.enabled`: enables synthetic-cluster quality weighting.
- `stacking.cluster_quality_weighting.kappa_cluster`: weighting exponent.
- `stacking.cluster_quality_weighting.cap_enabled`: explicit dominance cap toggle.
- `stacking.cluster_quality_weighting.cap_ratio`: dominance cap level when enabled.
- **Runtime safeguard:** for synthetic stacking, a default dominance cap is applied even when `cap_enabled=false` to avoid diffuse-signal collapse from a few dominant clusters.
- `stacking.output_stretch`: optional display-oriented post-scale to 16-bit span.
- `stacking.cosmetic_correction`: optional hot-pixel style correction after stacking.
- `stacking.cosmetic_correction_sigma`: detection threshold for cosmetic correction.
- `validation.min_fwhm_improvement_percent`: required sharpness improvement check.
- `validation.max_background_rms_increase_percent`: background degradation guard.
- `validation.min_tile_weight_variance`: sanity check against degenerate local weighting.
- `validation.require_no_tile_pattern`: checker/grid artifact detector gate.
- `runtime_limits.parallel_workers`: upper bound for worker threads.
- `runtime_limits.memory_budget`: memory budget that can cap effective parallelism.
- `runtime_limits.tile_analysis_max_factor_vs_stack`: performance anomaly warning threshold.
- `runtime_limits.hard_abort_hours`: absolute runtime safety stop.
- `runtime_limits.allow_emergency_mode`: permits processing below normal assumptions.
