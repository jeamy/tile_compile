# Tile-Compile C++ Configuration Reference

This documentation describes all configuration options for `tile_compile.yaml` based on the C++ implementation in `configuration.hpp` and the schema files `tile_compile.schema.json` / `tile_compile.schema.yaml`.

**Source of truth for defaults:** `include/tile_compile/config/configuration.hpp`  
**Schema version:** v3  
**Reference:** Methodology v3.3

**Documentation status (2026-06-09):**
- `aqmh.*` fully documented (all 4 sub-blocks: `pyramid`, `storage`, `cherry_pick`, `diagnostics`).
- `bge.fit.robust_loss` and `bge.fit.huber_delta` are documented and user-configurable.
- `bge.min_valid_sample_fraction_for_apply` and `bge.min_valid_samples_for_apply` are documented as BGE channel-apply guards.
- PCC coverage includes active stability and apply controls (`max_condition_number`, `max_residual_rms`, `apply_attenuation`, `chroma_strength`, `k_max`).
- `TILE_RECONSTRUCTION` boundary diagnostics are documented as runtime artifacts; there is currently no dedicated seam-correction config block.
- `bge.tile_weight_lambda_structure` is aligned with the current default `1.0`.
- `stacking.common_overlap_required_fraction` and `stacking.tile_common_valid_min_fraction` are documented as active stacking controls with strict defaults `1.0 / 1.0`.

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
12b. [AQMH (Adaptive Quality Map Harvesting)](#12b-aqmh-adaptive-quality-map-harvesting) **NEW**
13. [Synthetic](#13-synthetic)
14. [Reconstruction](#14-reconstruction)
15. [Debayer (automatic phase)](#15-debayer-automatic-phase)
16. [Astrometry](#16-astrometry)
17. [BGE (Background Gradient Extraction)](#17-bge-background-gradient-extraction) **NEW in v3.3**
18. [PCC](#18-pcc)
19. [HyperMetric Stretch](#19-hypermetric-stretch)
20. [Stacking](#20-stacking)
21. [Validation](#21-validation)
22. [Runtime Limits](#22-runtime-limits)
23. [Raw Stack / Preprocessing](#23-raw-stack--preprocessing)

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

### `output.write_registered_frames`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Persist registered frames as FITS (`reg_XXXXX.fit`).

### `output.crop_to_nonzero_bbox`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Crop final stack to bounding box of all non-zero pixels.

- **`true`**: Removes empty borders from final image. Only pixels with values > 0 are kept. Reduces file size and removes unnecessary black borders.
- **`false`**: Keeps full canvas size, including empty borders.

**Note:** Applied after stacking phase but before debayer (for OSC). Tile offsets are adjusted accordingly. Invalid canvas areas remain masked and are not intended to contribute to downstream debayer/BGE/PCC calculations.

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
| **Default** | `"GBRG"` |

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
| **Values** | `strict`, `moderate`, `permissive` |
| **Default** | `"strict"` |

**Purpose:** Controls whether linearity violations fail, warn, or are ignored.

---

## 5. Calibration

Calibration frame processing.

### `calibration.use_bias` / `calibration.use_dark` / `calibration.use_flat`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` (all three) |

**Purpose:** Enable per-frame-type calibration stages.

**Runtime behavior:** Each enabled stage requires at least one configured
source:
- directory input via `*_dir`
- or explicit master via `*_master`

### `calibration.bias_use_master` / `calibration.dark_use_master` / `calibration.flat_use_master`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Use prebuilt master calibration files instead of stacking directories.

### `calibration.dark_already_bias_corrected`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Marks a master dark as already bias-corrected. When `false` and `use_bias: true`, the runner subtracts the bias from the dark internally before applying calibration so the light-frame offset is not removed twice.

### `calibration.dark_auto_select`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Auto-select matching darks by exposure (and optionally temperature).

### `calibration.dark_match_exposure_tolerance_percent`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | `0` |
| **Default** | `5.0` |

**Purpose:** Maximum allowed exposure mismatch for dark matching in percent.

### `calibration.dark_match_use_temp` / `calibration.dark_match_temp_tolerance_c`

| Property | Value |
|----------|-------|
| **Type** | boolean / number |
| **Default** | `false` / `2.0` |

**Purpose:** When `dark_match_use_temp=true`, sensor temperature is also considered for dark matching.

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

### `calibration.pattern`

| Property | Value |
|----------|-------|
| **Type** | string |
| **Default** | `"*.fit;*.fits;*.fts;*.fit.fz;*.fits.fz;*.fts.fz"` |

**Purpose:** Glob pattern used for calibration file discovery.

---

## 6. Assumptions

Frame-count assumptions and reduced-mode behavior.

### `assumptions.frames_min`
| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | `1` |
| **Default** | `50` |

**Purpose:** Minimum usable frame count before the run aborts or drops into emergency reduced mode.

### `assumptions.frames_reduced_threshold`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | `1` |
| **Default** | `200` |

**Purpose:** Switch point between reduced and full pipeline behavior.

Runtime uses `frames_min` and `frames_reduced_threshold` directly:

| Frame count | Mode |
|-------------|------|
| `< frames_min` | abort / emergency reduced mode |
| `frames_min <= N < frames_reduced_threshold` | **Reduced Mode** |
| `N >= frames_reduced_threshold` | **Full Mode** |

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
| **Default** | `[5, 10]` |

**Purpose:** Cluster search range used when reduced mode still runs clustering.

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

**Cascade:**

- with `registration.enable_star_pair_fallback=true`:
  Triangle Stars → Star Pairs → Trail Endpoints → AKAZE Features → Robust Phase+ECC → Hybrid Phase+ECC → Identity fallback
- with `registration.enable_star_pair_fallback=false`:
  Triangle Stars → Trail Endpoints → AKAZE Features → Robust Phase+ECC → Hybrid Phase+ECC → Identity fallback

### `registration.enable_star_pair_fallback`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enable/disable the extra Star-Pairs fallback stage between Triangle Stars and Trail Endpoints.

Set to `false` to disable the Star-Pairs stage for a stricter fallback policy.

**Note (Strict v3.3.9):** Set `registration.enable_star_pair_fallback: false` for the strict profile.

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
| **Default** | `150` |

**Purpose:** Number of strongest stars used for star-based matching.

### `registration.star_min_inliers`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `4` |

**Purpose:** Minimum inlier matches required for acceptance.

### `registration.star_inlier_tol_px`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `4.0` |

**Purpose:** Inlier tolerance in pixels for transformed star matches.

### `registration.star_dist_bin_px`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `5.0` |

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
| **Default** | `0.25` |

**Purpose:** Absolute minimum correlation coefficient threshold.

### `registration.reject_shift_px_min`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `100.0` |

**Purpose:** Fixed minimum shift threshold for rejection logic.

### `registration.reject_shift_median_multiplier`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `5.0` |

**Purpose:** Multiplier for robust shift threshold (`multiplier * median_shift`).

### `registration.reject_scale_min` / `registration.reject_scale_max`

| Property | Value |
|----------|-------|
| **Type** | number / number |
| **Default** | `0.92` / `1.08` |

**Purpose:** Allowed similarity scale range. Warps outside `[reject_scale_min, reject_scale_max]` or with negative determinant (reflection) are rejected.

----

### `registration.max_blind_chain_depth`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | 0 |
| **Maximum** | 100 |
| **Default** | `0` |

**Purpose:** Maximum depth for blind-chain rescue. `0` = automatic (N/10, min 12, max 50).

- **Value 0 (auto):** Automatic calculation based on frame count N. Formula: `clamp(N/10, 12, 50)`
- **Value >0:** Manual override of maximum chain depth

**Note:** Higher values allow longer chains during cloud blocks but increase drift error risk.

----

### `registration.blind_chain_strong_anchor_cc`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | 0.01 |
| **Maximum** | 0.5 |
| **Default** | `0.08` |

**Purpose:** CC threshold for "strong anchors" in blind chains. Frames with higher CC can start deeper chains.

**Note:** Lower values allow more frames as strong anchors (aggressive), higher values are more conservative.

----

### `registration.blind_chain_drift_threshold_px`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | 0.5 |
| **Maximum** | 10.0 |
| **Default** | `2.0` |

**Purpose:** Maximum allowed drift in pixels per frame within a blind chain.

- Chain is aborted when cumulative drift exceeds this value
- Protects against accumulating errors in long chains

----

### `registration.use_astrometry`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enable astrometric rescue for frames that fail all other registration algorithms.

**Requirements:**
- ASTAP binary must be available (see `astrometry.astap_bin`)
- Local star catalog must be present (see `astrometry.astap_data_dir`)

**Note:** Set to `false` for very bright stars (e.g., Capella) as ASTAP has issues with overexposed centers.

----

### `registration.enable_local_background_subtraction`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Enable local background subtraction before star detection.

**Recommended for:**
- Strong moonlight with gradients
- Bright background structures (nebulae, galaxies)
- Uneven background from flat correction errors

---

### `registration.star_shift_radius_px`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `200.0` |
| **Minimum** | `10` |
| **Maximum** | `2000` |

**Purpose:** Search radius for the shift-consistency filter in `triangle_star_matching` (pixels on the proxy image at half resolution). After triangle voting, each star pair is checked against others to find how many imply a similar shift (within this radius). The cluster with the highest support becomes the anchor; all inconsistent pairs are discarded. The radius must cover the **maximum expected inter-frame shift**.

**When to adjust:**
- **Equatorial mount** with good tracking (small shifts): `60`
- **Alt/Az mount** (DWARF II, Seestar, multi-hour session): `200–400`
- **Very long Alt/Az session** (>4h, large shift range): `400–600`

> ⚠️ Too small a radius (e.g. 60 px on Alt/Az) causes false-match clusters to win the anchor vote over the real shift cluster → all frames fail triangle matching.

----

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

### `local_metrics.neighborhood_normalization.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enables neighborhood-pooled robust normalization of local tile metrics before local score composition.

### `local_metrics.neighborhood_normalization.radius`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | `0` |
| **Default** | `1` |

**Purpose:** Tile-neighborhood radius on the tile grid used to pool metric statistics for robust local z-score normalization.

### `local_metrics.neighborhood_normalization.blend`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `0 .. 1` |
| **Default** | `0.5` |

**Purpose:** Blend factor between tile-local robust z-scores and neighborhood-pooled robust z-scores.

### `local_metrics.spatial_regularization.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enables neighbor-aware regularization of local tile quality scores before exponential weight mapping.

### `local_metrics.spatial_regularization.lambda`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `0 .. 1` |
| **Default** | `0.35` |

**Purpose:** Coupling strength between a tile-local score and the mean score of its direct tile neighbors.

### `local_metrics.spatial_regularization.passes`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | `0` |
| **Default** | `1` |

**Purpose:** Number of regularization passes over the tile-neighborhood graph before `L_f,t = exp(Q_f,t)`.

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

### `local_metrics.k_local`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `> 0` |
| **Default** | `1.0` |

**Purpose:** Exponent scale for local weight `L_{f,t} = exp(k_local * Q_local)`. Default `1.0`; values `> 1` increase local differentiation, `< 1` soften it. Symmetric to `global_metrics.weight_exponent_scale`.

---

## 12b. AQMH (Adaptive Quality Map Harvesting)

AQMH is an independent, per-pixel reconstruction path that can replace the tile-based OLA stacking. For each frame a quality map is computed combining sharpness and SNR information. Reconstruction is performed as a per-pixel weighted average using AQMH weights.

> **Experimental.** When `aqmh.enabled: true`, AQMH fully replaces the tile-OLA reconstruction. Logs appear under `[AQMH]`.

### `aqmh.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enables the AQMH reconstruction path. When `false`, the classic tile-OLA reconstruction is used.

---

### `aqmh.pyramid.*` — Pyramid quality metrics

Controls the Laplacian pyramid used to derive per-frame sharpness and SNR.

#### `aqmh.pyramid.scales`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Range** | 1 – 8 |
| **Default** | `4` |

**Purpose:** Number of pyramid levels for the multi-scale analysis. More levels capture more spatial frequencies at the cost of compute time.

---

#### `aqmh.pyramid.base_window_px`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | 1 |
| **Default** | `4` |

**Purpose:** Window size in pixels at the lowest pyramid level for local metric computation.

---

#### `aqmh.pyramid.w_sharp`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | 0 |
| **Default** | `0.6` |

**Purpose:** Weight of the sharpness metric in the combined quality index `Q = w_sharp * Q_sharp + w_snr * Q_snr`. Together with `w_snr` it controls the relative importance of sharpness vs. SNR.

---

#### `aqmh.pyramid.w_snr`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | 0 |
| **Default** | `0.4` |

**Purpose:** Weight of the SNR metric in the combined quality index. Increase for heavily noisy data with large inter-frame quality spread.

---

#### `aqmh.pyramid.k_artifact`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | >0 |
| **Default** | `3.0` |

**Purpose:** MAD multiplier for artifact detection. Pixels whose local variance exceeds `k_artifact * MAD` are flagged as artifacts and receive reduced AQMH weight.

- **Higher (e.g. 7–10):** More tolerant of outliers — more pixels receive normal weight
- **Lower (e.g. 3–4):** More aggressive artifact suppression

---

#### `aqmh.pyramid.frac_artifact_max`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | >0 – 1 |
| **Default** | `0.25` |

**Purpose:** Maximum tolerated artifact fraction per evaluation window. Windows with more artifacts than `frac_artifact_max` are discarded entirely (no AQMH contribution).

- **Increase (e.g. 0.30–0.40):** When known tolerable artifacts are present (e.g. satellite trails)
- **Decrease:** Stricter quality gates

---

### `aqmh.storage.*` — Quality map storage

#### `aqmh.storage.resolution_divisor`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Values** | `1`, `2`, `4` |
| **Default** | `2` |

**Purpose:** Resolution reduction factor for stored quality maps. `2` = half resolution (recommended, reduces storage by factor 4). `1` = full resolution.

---

#### `aqmh.storage.dtype`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `float32`, `uint8` |
| **Default** | `"float32"` |

**Purpose:** Data type for cached quality maps. `float32` is precise; `uint8` saves disk space (8-bit quantisation of quality values).

---

#### `aqmh.storage.max_resident_maps`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Range** | 0 – 16 |
| **Default** | `2` |

**Purpose:** Maximum number of quality maps held simultaneously in RAM. Caps memory use during streaming operation. `0` = unlimited.

---

### `aqmh.cherry_pick.*` — Frame selection

#### `aqmh.cherry_pick.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Enables selective stacking — only the highest-quality frames are used for AQMH reconstruction. Useful for large datasets with strongly varying quality (e.g. sessions interrupted by clouds).

---

#### `aqmh.cherry_pick.k_min`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | 1 |
| **Default** | `3` |

**Purpose:** Minimum number of frames always included even in cherry-pick mode. Prevents under-determination on small datasets.

---

#### `aqmh.cherry_pick.k_frac`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | >0 – 1 |
| **Default** | `0.30` |

**Purpose:** Fraction of best frames (sorted by AQMH quality) used for cherry-pick stacking. `0.30` = best 30% of frames.

---

### `aqmh.diagnostics.*` — Diagnostic outputs

#### `aqmh.diagnostics.tau_artifact`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | 0 – 1 |
| **Default** | `0.20` |

**Purpose:** Threshold for artifact diagnostics written to `artifacts/aqmh.json`. Pixels with artifact probability > `tau_artifact` are flagged as problematic in the diagnostic output.

---

#### `aqmh.diagnostics.q_region`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | 0 – 1 |
| **Default** | `0.75` |

**Purpose:** Quantile for regional quality statistics in the AQMH diagnostic output. `0.75` = 75th percentile of quality values.

---

#### `aqmh.diagnostics.r_morph_canvas_px`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | 1 |
| **Default** | `6` |

**Purpose:** Morphological radius in canvas pixels for the regional diagnostic map. Controls spatial smoothing when generating the diagnostic quality-map overview.

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

The current C++ config has **no standalone `reconstruction:` block**.

Weighted tile reconstruction, Hanning OLA, and boundary diagnostics are runtime behavior of the runner, not separate top-level config keys. Relevant knobs currently live under:

- `synthetic.*`
- `stacking.*`
- `tile.*`
- `tile_denoise.*`

---

## 15. Debayer (automatic phase)

There is no standalone `debayer` config key anymore.

Debayering is automatic:
- `OSC`: the runner always debayers the final CFA stack into RGB outputs after stacking.
- `MONO`: the phase is a no-op and ends as `ok/MONO`.

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

## 17. BGE (Background Gradient Extraction)

**NEW in v3.3** - Optional background gradient extraction before PCC (Methodology v3.3 §6.3)

BGE removes large-scale background gradients (light pollution, moonlight, airglow) **before** photometric color calibration to avoid color bias from spectrally non-uniform gradients.

**Implementation note (v3.3.6):** BGE directly uses tile-quality data from `LOCAL_METRICS` for sample selection/weighting:
- `type` + `star_count`: star-dense STAR tiles are conservatively excluded or down-weighted.
- `fwhm`: scales effective star-mask dilation per tile.
- `quality_score`: applied as an additional tile-sample reliability factor.

**Key BGE parameters:**
- `bge.enabled`: Enable/disable (default: `false`)
- `bge.tile_weight_lambda_structure`: Lambda in tile reliability weight `w_t = exp(-lambda * structure_score_t) * (1 - masked_fraction_t)` (range `> 0`, default `1.0`)
- `bge.sample_quantile`: Tile background quantile (range `(0, 0.5]`, default `0.20`)
- `bge.min_valid_sample_fraction_for_apply`: Minimum valid tile-sample fraction required per channel before BGE apply (range `(0, 1]`, default `0.30`)
- `bge.min_valid_samples_for_apply`: Minimum absolute valid tile-sample count required per channel before BGE apply (minimum `1`, default `96`)
- `bge.fit.method`: Surface fitting method - `rbf`, `poly`, `spline`, `bicubic`, `modeled_mask_mesh` (default `rbf`)
- `bge.fit.robust_loss`: Robust IRLS loss - `huber` or `tukey` (default `huber`)
- `bge.fit.huber_delta`: Huber transition parameter (default `1.5`, used when `robust_loss=huber`)
- `bge.fit.rbf_phi`: RBF kernel - `multiquadric`, `thinplate`, `gaussian` (default `multiquadric`)
- `bge.autotune.enabled`: Deterministic test-adjust-test autotune (default `false`)
- `bge.autotune.strategy`: `conservative|extended` (default `conservative`)
- `bge.autotune.max_evals`: hard cap for tested candidates (minimum `1`, default `24`)
- `bge.autotune.holdout_fraction`: deterministic CV split fraction (range `[0.05, 0.50]`, default `0.25`)
- `bge.autotune.alpha_flatness`: objective weight for flatness term (minimum `0`, default `0.25`)
- `bge.autotune.beta_roughness`: objective weight for roughness term (minimum `0`, default `0.10`)

**Recommendation:** Enable with `bge.enabled: true` when gradients are visible (urban light pollution, moonlight) or when PCC shows color shifts across the field.

### `bge.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Enable or disable BGE before PCC.

### `bge.tile_weight_lambda_structure`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `> 0` |
| **Default** | `1.0` |

**Purpose:** Lambda in tile reliability weight `w_t = exp(-lambda * structure_score_t) * (1 - masked_fraction_t)`. Higher values down-weight structure-rich tiles more aggressively; `1.0` is the current moderate baseline.

### `bge.sample_quantile`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `(0, 0.5]` |
| **Default** | `0.20` |

**Purpose:** Tile background quantile used to estimate robust background samples.

### `bge.structure_thresh_percentile`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `0 – 1` |
| **Default** | `0.90` |

**Purpose:** Structure-threshold percentile used to reject structure-rich tiles or pixels from BGE sampling.

### `bge.min_tiles_per_cell`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | `1` |
| **Default** | `3` |

**Purpose:** Minimum number of valid tiles required per BGE grid cell.

### `bge.mask.star_dilate_px`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | `0` |
| **Default** | `4` |

**Purpose:** Star-mask dilation radius in pixels for BGE sample protection.

### `bge.mask.sat_dilate_px`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | `0` |
| **Default** | `4` |

**Purpose:** Saturation-mask dilation radius in pixels for BGE sample protection.

### `bge.grid.N_g`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | `1` |
| **Default** | `32` |

**Purpose:** Target BGE grid density / cell count driver.

### `bge.grid.G_min_px`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | `1` |
| **Default** | `64` |

**Purpose:** Minimum pixel size of a BGE grid cell.

### `bge.grid.G_max_fraction`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `(0, 1]` |
| **Default** | `0.25` |

**Purpose:** Upper bound for grid-cell size relative to the image extent.

### `bge.grid.insufficient_cell_strategy`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `discard`, `nearest`, `radius_expand` |
| **Default** | `"discard"` |

**Purpose:** Fallback strategy for grid cells with too few valid samples.

### `bge.fit.method`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `rbf`, `poly`, `spline`, `bicubic`, `modeled_mask_mesh` |
| **Default** | `"rbf"` |

**Purpose:** Surface fitting method used by BGE.

### `bge.fit.irls_max_iterations`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | `1` |
| **Default** | `10` |

**Purpose:** Maximum number of IRLS iterations for robust BGE fitting.

### `bge.fit.irls_tolerance`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | `> 0` |
| **Default** | `1e-4` |

**Purpose:** Convergence tolerance for IRLS-based BGE fitting.

### `bge.fit.polynomial_order`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Values** | `2`, `3` |
| **Default** | `2` |

**Purpose:** Polynomial order when `bge.fit.method=poly`.

### `bge.fit.rbf_mu_factor`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | `> 0` |
| **Default** | `1.0` |

**Purpose:** Scale factor for the adaptive RBF smoothing parameter.

### `bge.fit.rbf_lambda`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | `> 0` |
| **Default** | `1e-6` |

**Purpose:** Regularization strength for RBF fitting.

### `bge.fit.rbf_epsilon`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | `> 0` |
| **Default** | `1e-10` |

**Purpose:** Epsilon / support parameter for the RBF kernel.

### `bge.min_valid_sample_fraction_for_apply`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `(0.0, 1.0]` |
| **Default** | `0.30` |

**Purpose:** Per-channel safety gate for BGE application. If `valid_tile_samples / total_tile_samples` is below this value, BGE is skipped for that channel.

### `bge.min_valid_samples_for_apply`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | `1` |
| **Default** | `96` |

**Purpose:** Absolute per-channel safety gate for BGE application. If valid robust tile samples are fewer than this value, BGE is skipped for that channel.

### `bge.fit.robust_loss`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `huber`, `tukey` |
| **Default** | `"huber"` |

**Purpose:** Robust loss function used in BGE IRLS fitting.

### `bge.fit.huber_delta`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | `> 0` |
| **Default** | `1.5` |

**Purpose:** Huber transition parameter (active when `bge.fit.robust_loss=huber`).

---

## 18. PCC

Photometric Color Calibration settings.

**Implementation note (v3.3.6):** If tile metrics and tile geometry are available and size-consistent, PCC automatically uses them for robust per-star weighting:
- `quality_score`: exponential per-star weight factor (tile-based).
- `gradient_energy/noise`: structure penalty and reject for highly structured tiles.
- `star_count`: mild down-weighting for very star-dense tiles.

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

### `pcc.background_model`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `median`, `plane` |
| **Default** | `"plane"` |

**Purpose:** Local sky-annulus background model for stellar photometry (`plane` recommended under gradients, fallback to `median` if plane fit fails).

### `pcc.max_condition_number`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | `>= 1.0` |
| **Default** | `3.0` |

**Purpose:** Upper bound for PCC matrix condition number; rejects numerically unstable solutions.

### `pcc.max_residual_rms`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | `> 0` |
| **Default** | `0.35` |

**Purpose:** Upper bound for robust fit residual RMS; rejects noisy/unstable PCC fits.

### `pcc.radii_mode`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `fixed`, `auto_fwhm` |
| **Default** | `"auto_fwhm"` |

**Purpose:** Radius handling mode. `auto_fwhm` derives aperture/annulus radii from seeing FWHM using the multipliers below.

### `pcc.aperture_fwhm_mult`, `pcc.annulus_inner_fwhm_mult`, `pcc.annulus_outer_fwhm_mult`, `pcc.min_aperture_px`

| Key | Type | Default |
|-----|------|---------|
| `pcc.aperture_fwhm_mult` | number (>0) | `1.8` |
| `pcc.annulus_inner_fwhm_mult` | number (>0) | `3.0` |
| `pcc.annulus_outer_fwhm_mult` | number (>0) | `5.0` |
| `pcc.min_aperture_px` | number (>0) | `4.0` |

**Purpose:** Conservative FWHM-adaptive radius controls (v3.3.6 §6.4.2).

### `pcc.siril_catalog_dir`

| Property | Value |
|----------|-------|
| **Type** | string |
| **Default** | `""` |

**Purpose:** Optional local Siril catalog path.

### `pcc.apply_attenuation`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Enables adaptive attenuation during PCC matrix application (helps in deep shadows/highlights).

### `pcc.background_neutralization_mode`

| Property | Value |
|----------|-------|
| **Type** | string |
| **Default** | `auto` |
| **Allowed Values** | `always`, `auto`, `off` |

**Purpose:** Controls the post-PCC background neutralization step. `always` forces neutral background offsets, `off` disables the step, and `auto` attenuates or skips it when the measured "background" looks nebulosity-dominated rather than truly neutral sky.

### `pcc.chroma_strength`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `1.0` |

**Purpose:** Global strength factor for PCC chroma correction during apply.

### `pcc.k_max`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `3.2` |

**Purpose:** Upper bound for linear PCC apply strength (limits over-correction in bright structures).

---

## 19. HyperMetric Stretch

VeraLux HyperMetric Stretch (HMS) is an optional final RGB stretch phase after PCC. It reads the PCC RGB result and writes `outputs/stacked_rgb_hms.fits` by default.

| Key | Type | Default | Constraint |
|-----|------|---------|------------|
| `hypermetric_stretch.enabled` | boolean | `true` | |
| `hypermetric_stretch.require_successful_pcc` | boolean | `true` | |
| `hypermetric_stretch.mode` | string | `ready_to_use` | `ready_to_use`, `scientific` |
| `hypermetric_stretch.sensor_profile` | string | `rec709` | |
| `hypermetric_stretch.fallback_profile` | string | `rec709` | |
| `hypermetric_stretch.adaptive_anchor` | boolean | `true` | |
| `hypermetric_stretch.target_bg` | number | `0.20` | 0.05 - 0.50 |
| `hypermetric_stretch.protect_b` | number | `6.0` | >= 0.1 |
| `hypermetric_stretch.convergence_power` | number | `3.5` | 1.0 - 10.0 |
| `hypermetric_stretch.log_d_mode` | string | `auto` | `auto`, `fixed` |
| `hypermetric_stretch.fixed_log_d` | number | `2.0` | 0 - 7 |
| `hypermetric_stretch.color_strategy` | string | `fixed` | `auto`, `fixed` |
| `hypermetric_stretch.fixed_color_strategy` | number | `0.0` | -1 - 1 |
| `hypermetric_stretch.color_grip` | number | `1.0` | 0 - 1 |
| `hypermetric_stretch.shadow_convergence` | number | `0.0` | >= 0 |
| `hypermetric_stretch.linear_expansion` | number | `0.0` | 0 - 1 |
| `hypermetric_stretch.write_channels` | boolean | `false` | |
| `hypermetric_stretch.output_rgb` | string | `stacked_rgb_hms.fits` | non-empty |

`ready_to_use` follows the VeraLux GUI default with output scaling to `target_bg` and final soft clip. `scientific` skips the ready-to-use final scaling/soft clip and allows `linear_expansion`.

The default is the explicit `rec709` profile. Concrete VeraLux profile names can be set directly. `auto` remains accepted for compatibility and currently uses `fallback_profile`, but it is no longer recommended as the default. The sensor profiles are defined in `tile_compile_cpp/src/image/hypermetric_stretch.cpp` in `profiles()`. Profile matching is normalized, so case, spaces, and punctuation are tolerated; the recommended YAML values are:

| YAML value | R | G | B |
|------------|---|---|---|
| `rec709` | 0.2126 | 0.7152 | 0.0722 |
| `Rec.709 (Recommended)` | 0.2126 | 0.7152 | 0.0722 |
| `Sony IMX571 (ASI2600/QHY268)` | 0.2944 | 0.5021 | 0.2035 |
| `Sony IMX455 (ASI6200/QHY600)` | 0.2987 | 0.5001 | 0.2013 |
| `Sony IMX410 (ASI2400)` | 0.3015 | 0.5050 | 0.1935 |
| `Sony IMX269 (Altair/ToupTek)` | 0.3040 | 0.5010 | 0.1950 |
| `Sony IMX294 (ASI294)` | 0.3068 | 0.5008 | 0.1925 |
| `Sony IMX533 (ASI533)` | 0.2910 | 0.5072 | 0.2018 |
| `Sony IMX676 (ASI676)` | 0.2880 | 0.5100 | 0.2020 |
| `Sony IMX585 (ASI585) - STARVIS 2` | 0.3431 | 0.4822 | 0.1747 |
| `Sony IMX662 (ASI662) - STARVIS 2` | 0.3430 | 0.4821 | 0.1749 |
| `Sony IMX678 (ASI678) - STARVIS 2` | 0.3426 | 0.4825 | 0.1750 |
| `Sony IMX415 (DWARF II)` | 0.2703 | 0.5405 | 0.1892 |
| `Sony IMX462 (ASI462)` | 0.3333 | 0.4866 | 0.1801 |
| `Sony IMX715 (ASI715)` | 0.3410 | 0.4840 | 0.1750 |
| `Sony IMX482 (ASI482)` | 0.3150 | 0.4950 | 0.1900 |
| `Sony IMX183 (ASI183)` | 0.2967 | 0.4983 | 0.2050 |
| `Sony IMX178 (ASI178)` | 0.2346 | 0.5206 | 0.2448 |
| `Sony IMX224 (ASI224)` | 0.3402 | 0.4765 | 0.1833 |
| `Canon EOS (Modern - 60D/600D/500D)` | 0.2600 | 0.5200 | 0.2200 |
| `Canon EOS (Legacy - 300D/40D/20D)` | 0.2450 | 0.5350 | 0.2200 |
| `Nikon DSLR (Modern - D5100/D7200)` | 0.2650 | 0.5100 | 0.2250 |
| `Nikon DSLR (Legacy - D3/D300/D90)` | 0.2500 | 0.5300 | 0.2200 |
| `Fujifilm X-Trans 5 HR` | 0.2800 | 0.5100 | 0.2100 |
| `Panasonic MN34230 (ASI1600)` | 0.2650 | 0.5250 | 0.2100 |
| `ZWO Seestar S50` | 0.3333 | 0.4866 | 0.1801 |
| `ZWO Seestar S30` | 0.2928 | 0.5053 | 0.2019 |
| `Narrowband HOO` | 0.5000 | 0.2500 | 0.2500 |
| `Narrowband SHO` | 0.3333 | 0.3400 | 0.3267 |

---

## 20. Stacking

Final stacking settings.

### `stacking.method`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `rej`, `average` |
| **Default** | `"rej"` |

**Purpose:** Final stacking method.

### `stacking.common_overlap_required_fraction`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `(0, 1]` |
| **Default** | `1.0` |

**Purpose:** Required fraction of usable frames in which a pixel must be valid to belong to `COMMON_OVERLAP`.

- **`1.0`**: strict intersection across all usable frames
- **`< 1.0`**: allows edge pixels that are present in only a subset of frames

**Note:** Lower values increase border area but can bias background and color statistics through uneven edge coverage.

**Note (Strict v3.3.9):** Keep this at `1.0`.

### `stacking.tile_common_valid_min_fraction`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `(0, 1]` |
| **Default** | `1.0` |

**Purpose:** Minimum fraction of the **full tile area** that must lie inside `COMMON_OVERLAP` for a tile to remain valid for local metrics and downstream processing.

- **`1.0`**: only tiles fully inside the support mask are valid
- **`< 1.0`**: allows partially covered edge tiles

**Note:** The ratio is computed over the full tile area, not only the in-bounds remainder.

**Note (Strict v3.3.9):** Keep this at `1.0`.

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

### Boundary Diagnostics in `TILE_RECONSTRUCTION`

There is currently **no dedicated seam-correction parameter block** in the active C++ config.

Visible tile boundaries are instead analyzed through runtime artifacts written by `TILE_RECONSTRUCTION`, especially:

- `tile_boundary_raw_pair_mean_abs_diff_p95`
- `tile_boundary_normalized_pair_mean_abs_diff_p95`
- `tile_boundary_pair_count`
- `tile_boundary_observation_count`
- `tile_boundary_pair_mean_abs_diff_mean`
- `tile_boundary_pair_mean_abs_diff_p95`
- `tile_boundary_post_background_delta_p95_abs`
- `tile_boundary_top_pairs`
- `tile_norm_bg_r` / `tile_norm_bg_g` / `tile_norm_bg_b`
- `tile_norm_scale`

`tile_boundary_raw_*` measures the mismatch before the optional per-tile normalization, `tile_boundary_normalized_*` on the actual OLA input. These diagnostics use the common canvas-valid mask, describe the actual mismatch of neighboring tiles at the OLA input stage, and do not modify the reconstruction result.

### `stacking.output_stretch`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Optional linear post-scaling of the output data from `0..max` to the full `0..65535` range.

### `stacking.cosmetic_correction`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Optional cosmetic correction after stacking.

---

### `stacking.cosmetic_correction_sigma`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | >0 |
| **Default** | `5.0` |

**Purpose:** MAD-sigma threshold for `stacking.cosmetic_correction`.

- Lower value = more aggressive.
- **Note:** In the stacked image, bright object cores can have high local contrast. Too aggressive settings can treat real signal peaks as hot pixels.

**Recommendation:**

- MONO / calibrated data: `5.0`
- OSC / smart telescope without darks: `10.0` (more conservative)

---

### `stacking.per_frame_cosmetic_correction`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Hot-pixel correction **per frame before PREWARP/stacking**.

This targets **fixed sensor defects** (RGB single-pixel speckles) that appear at the same coordinates in every frame and therefore can survive stack sigma clipping.

---

### `stacking.per_frame_cosmetic_correction_sigma`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | >0 |
| **Default** | `5.0` |

**Purpose:** MAD-sigma threshold for `stacking.per_frame_cosmetic_correction`.

**Recommendation:** `5.0` (often suitable for OSC/Seestar/DWARF).

---

## 21. Validation

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

## 22. Runtime Limits

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

**Runtime behavior:** The runner writes the measured ratio to
`artifacts/runtime_limits.json` and emits a warning when the configured
threshold is exceeded. This parameter does not stop the run by itself.

### `runtime_limits.hard_abort_hours`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `6.0` |

**Purpose:** Hard upper runtime limit in hours.

**Runtime behavior:** Checked after major phase boundaries in the main run and
resume path. Exceeding the limit aborts the run with
`runtime_limit_exceeded`.

### `runtime_limits.allow_emergency_mode`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Allow processing very small datasets in emergency mode.

---

### `runtime_limits.acceleration_backend`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Default** | `auto` |
| **Valid values** | `auto`, `cpu`, `opencv_cuda`, `opencv_opencl`, `opencl`, `cuda` |

**Purpose:** GPU acceleration backend selection for PREWARP, TILE_RECONSTRUCTION, and STACKING phases.

**Options:**
- `auto` (default): Automatically detects available GPU backends at runtime. Priority: CUDA → OpenCL → CPU. Falls back gracefully if hardware unavailable.
- `opencv_cuda`: Forces NVIDIA CUDA backend (requires CUDA-enabled OpenCV build and NVIDIA GPU).
- `opencv_opencl` / `opencl`: Forces OpenCL backend (requires OpenCL-enabled OpenCV build; works with AMD Radeon, Intel iGPU, NVIDIA GPUs).
- `cpu`: Disables GPU acceleration entirely.
- `cuda`: Experimental native CUDA backend (not yet implemented).

**Hardware compatibility:**
- **NVIDIA GPUs:** Both `opencv_cuda` (recommended for best performance) and `opencv_opencl` work.
- **AMD GPUs (Radeon RX 470/480/570/580/590, Vega, RDNA):** Use `opencv_opencl` or `auto`.
- **Intel integrated GPUs:** Use `opencv_opencl` or `auto`.

**Build requirements:**
- CUDA backend: OpenCV built with `WITH_CUDA=ON` and modules `opencv2/core/cuda.hpp`, `opencv2/cudawarping.hpp`, `opencv2/cudaarithm.hpp`.
- OpenCL backend: OpenCV built with `WITH_OPENCL=ON` and module `opencv2/core/ocl.hpp`.

**Note:** If requested backend is unavailable (missing OpenCV modules or hardware), the pipeline falls back to CPU with a warning.

---

## 23. Raw Stack / Preprocessing

Raw Stack is a separate preprocessing process and is not part of the normal `tile_compile.yaml` main pipeline. Its configuration is used through the preprocessing API and the Raw Stack parameter editor:

- `GET /api/tools/preprocessing/defaults`
- `GET /api/tools/preprocessing/parameters`
- `PATCH /api/tools/preprocessing/parameters`
- `POST /api/tools/preprocessing/run`

The process shares code and algorithms with Tile-Compile but does not appear in the normal Run Studio or the normal Parameter Studio.

### `preprocessing.mode`

| Property | Value |
|----------|-------|
| **Type** | string |
| **Default** | `linear_prestack` |
| **Values** | `linear_prestack` |

**Purpose:** Enables the classic linear pre-stack path without tile grid generation, tile reconstruction, synthetic frames, or state clustering.

### Input and CFA/Mono

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `lights_dir` | string | `""` | Light/raw input directory; in the GUI this is handled by the same controls as `Input & Scan`. |
| `bias_dir`, `darks_dir`, `flats_dir`, `darkflats_dir` | string | `""` | Calibration directories. |
| `input_mode` | string | `auto` | `auto`, `cfa_osc`, `mono`. |
| `raw_formats` | string | `tile_compile` | Uses the same raw/FITS input scope as Tile-Compile. |
| `bayer_pattern` | string | `auto` | Header-derived Bayer pattern or explicit `RGGB`, `GBRG`, `GRBG`, `BGGR`. |
| `cfa_mode` | string | `tile_compile` | CFA/OSC handling through Tile-Compile logic. |
| `mono_mode` | string | `auto` | Mono path without forced RGB/Bayer assumptions. |
| `registration_reference` | string | `best_quality` | Reference frame selection. |

### `calibration.*`

| Parameter | Type | Default |
|-----------|------|---------|
| `calibration.use_bias` | boolean | `false` |
| `calibration.use_dark` | boolean | `false` |
| `calibration.use_flat` | boolean | `false` |
| `calibration.bias_use_master`, `dark_use_master`, `flat_use_master`, `darkflat_use_master` | boolean | `false` |
| `calibration.dark_auto_select` | boolean | `true` |
| `calibration.dark_match_use_temp` | boolean | `false` |
| `calibration.dark_match_exposure_tolerance_percent` | number | `8.0` |
| `calibration.dark_match_temp_tolerance_c` | number | `3.0` |
| `calibration.bias_master`, `dark_master`, `flat_master`, `darkflat_master` | string | `""` |
| `calibration.pattern` | string | `*.fit;*.fits;*.fts;*.fit.fz;*.fits.fz;*.fts.fz` |

### Quality and Stacking

| Parameter | Type | Default | Values |
|-----------|------|---------|--------|
| `quality_filter.mode` | string | `auto` | `auto`, `strict`, `relaxed`, `off` |
| `quality_filter.min_stars` | integer | `30` | >= 0 |
| `quality_filter.max_fwhm_sigma` | number | `2.0` | > 0 |
| `quality_filter.max_eccentricity` | number | `0.65` | 0 - 1 |
| `quality_filter.min_correlation` | number | `0.75` | 0 - 1 |
| `quality_filter.manual_overrides` | object | `{}` | Optional frame overrides by index or filename, e.g. `"0": {"include": false}`. |
| `rejection.method` | string | `sigma` | `sigma`, `median`, `winsor` |
| `rejection.low`, `rejection.high` | number | `3.0` | > 0 |
| `stacking.normalization` | string | `addscale` | `addscale`, `background`, `median`, `none` |
| `stacking.weighting` | string | `quality` | `quality`, `uniform` |

### Postprocess and HMS

| Parameter | Type | Default |
|-----------|------|---------|
| `postprocess.astrometry` | boolean | `true` |
| `postprocess.bge` | boolean | `true` |
| `postprocess.pcc` | boolean | `true` |
| `postprocess.hypermetric_stretch` | boolean | `true` |

HMS is enabled by default. Its detailed parameters match the normal Tile-Compile HMS contract and are only editable in the Raw Stack parameter editor:

| Parameter | Type | Default |
|-----------|------|---------|
| `hypermetric_stretch.require_successful_pcc` | boolean | `true` |
| `hypermetric_stretch.mode` | string | `ready_to_use` |
| `hypermetric_stretch.sensor_profile` | string | `rec709` |
| `hypermetric_stretch.fallback_profile` | string | `rec709` |
| `hypermetric_stretch.adaptive_anchor` | boolean | `true` |
| `hypermetric_stretch.target_bg` | number | `0.15` |
| `hypermetric_stretch.protect_b` | number | `6.0` |
| `hypermetric_stretch.convergence_power` | number | `3.5` |
| `hypermetric_stretch.log_d_mode` | string | `auto` |
| `hypermetric_stretch.fixed_log_d` | number | `2.0` |
| `hypermetric_stretch.color_strategy` | string | `fixed` |
| `hypermetric_stretch.fixed_color_strategy` | number | `0.0` |
| `hypermetric_stretch.color_grip` | number | `1.0` |
| `hypermetric_stretch.shadow_convergence` | number | `0.0` |
| `hypermetric_stretch.linear_expansion` | number | `0.0` |
| `hypermetric_stretch.write_channels` | boolean | `false` |
| `hypermetric_stretch.output_rgb` | string | `stacked_rgb_hms.fits` |

### Report

| Parameter | Type | Default |
|-----------|------|---------|
| `report.detailed` | boolean | `true` |
| `report.formats` | list | `[json, markdown, html]` |

Raw Stack writes report data under `artifacts/preprocess/`:

- `preprocessing_report.json`
- `preprocessing_report.md`
- `preprocessing_report.html`
- `frame_quality.csv`
- `rejected_frames.txt`
- `events.jsonl`
- `artifacts_manifest.json`

## Appendix A — Functional details for all options

This appendix provides a compact but explicit **runtime behavior** description for every configuration key.

### A.1 Pipeline / Output / Data

- `pipeline.mode`: selects production vs test control flow (same core phases, different strictness/debug posture).
- `pipeline.abort_on_fail`: controls whether `phase_end(error)` aborts run immediately.
- `output.registered_dir`: target folder name for registered frame outputs.
- `output.write_registered_frames`: writes per-frame registered FITS; increases IO and disk usage significantly.
- `output.crop_to_nonzero_bbox`: crops the final stack to its non-empty bounding box.
- `data.image_width`, `data.image_height`: optional expected dimensions; normally auto-detected from FITS headers.
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
- `assumptions.frames_min`: minimum usable frame count before the run aborts or enters emergency reduced mode.
- `assumptions.frames_reduced_threshold`: switch point between reduced and full pipeline behavior.
- `assumptions.reduced_mode_skip_clustering`: disables expensive state clustering in reduced mode.
- `assumptions.reduced_mode_cluster_range`: reduced-mode cluster search range when clustering still runs.

### A.3 Normalization / Registration / Dithering

- `normalization.enabled`: mandatory in methodology-driven runs (normally must stay enabled).
- `normalization.mode`: background-centric vs median-centric normalization strategy.
- `normalization.per_channel`: per-channel (OSC/RGB) normalization preserving channel balance.
- `registration.engine`: preferred first engine; runtime still executes multi-stage fallback cascade.
- `registration.enable_star_pair_fallback`: enables/disables the extra non-normative Star-Pairs fallback stage.
- `registration.allow_rotation`: permits rotational components in global warps (required for Alt/Az).
- `registration.star_topk`: number of strongest stars used by star-based engines.
- `registration.star_min_inliers`: minimum accepted inlier correspondences.
- `registration.star_inlier_tol_px`: geometric tolerance for inlier acceptance.
- `registration.star_dist_bin_px`: distance histogram quantization for star-similarity engine.
- `registration.reject_outliers`: enables robust rejection of implausible warps after matching.
- `registration.reject_cc_min_abs`: absolute NCC floor in outlier logic.
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
- `local_metrics.neighborhood_normalization.enabled`, `radius`, `blend`: stabilize local metric normalization by blending tile-local and neighborhood-pooled robust z-scores.
- `local_metrics.spatial_regularization.enabled`, `lambda`, `passes`: regularize local tile scores across neighboring tiles before exponential local weighting.
- `local_metrics.star_mode.weights.fwhm`, `roundness`, `contrast`: STAR tile quality composition.
- `local_metrics.structure_mode.metric_weight`, `background_weight`: STRUCTURE tile quality composition.
- `synthetic.weighting`: synthetic frame generation method (`global` vs `tile_weighted`).
- `synthetic.frames_min`: minimum cluster size to emit synthetic frame.
- `synthetic.frames_max`: maximum number of synthetic outputs.
- `synthetic.clustering.mode`: clustering backend for state grouping.
- `synthetic.clustering.cluster_count_range`: allowed K-search window.
- Reconstruction/OLA is currently internal runner behavior without a standalone `reconstruction:` config block.

### A.6 Debayer / Astrometry / PCC / HMS / Stacking / Validation / Runtime

- Debayer is an automatic OSC pipeline phase and no longer a standalone config key.
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
- `pcc.background_model`: local annulus background model.
- `pcc.max_condition_number`, `pcc.max_residual_rms`: matrix/fit stability limits.
- `pcc.radii_mode`, `pcc.aperture_fwhm_mult`, `pcc.annulus_inner_fwhm_mult`, `pcc.annulus_outer_fwhm_mult`, `pcc.min_aperture_px`: adaptive radius controls.
- `pcc.siril_catalog_dir`: local Siril catalog path override.
- `pcc.apply_attenuation`, `pcc.background_neutralization_mode`, `pcc.chroma_strength`, `pcc.k_max`: optional PCC apply/background-neutralization controls.
- `hypermetric_stretch.enabled`: enables VeraLux HyperMetric Stretch after PCC.
- `hypermetric_stretch.require_successful_pcc`: requires successful PCC artifacts before HMS.
- `hypermetric_stretch.mode`: `ready_to_use` final output mode or `scientific` controlled stretch mode.
- `hypermetric_stretch.sensor_profile`, `fallback_profile`: VeraLux luminance weights.
- `hypermetric_stretch.adaptive_anchor`, `target_bg`, `protect_b`, `convergence_power`: anchor, background target, and stretch controls.
- `hypermetric_stretch.log_d_mode`, `fixed_log_d`: automatic or fixed stretch strength.
- `hypermetric_stretch.color_strategy`, `fixed_color_strategy`, `color_grip`, `shadow_convergence`: color strategy and hybrid grip controls.
- `hypermetric_stretch.linear_expansion`: scientific-mode-only linear expansion.
- `hypermetric_stretch.write_channels`, `output_rgb`: HMS output controls.
- `stacking.method`: final combine mode (`rej` sigma-clip vs `average`).
- `stacking.common_overlap_required_fraction`: required pixel coverage across usable frames for `COMMON_OVERLAP`.
- `stacking.tile_common_valid_min_fraction`: minimum `COMMON_OVERLAP` coverage over the full tile area.
- `stacking.sigma_clip.sigma_low`, `sigma_high`: lower/upper rejection thresholds.
- `stacking.sigma_clip.max_iters`: clipping iteration cap.
- `stacking.sigma_clip.min_fraction`: minimum retained sample ratio fallback guard.
- `stacking.cluster_quality_weighting.enabled`: enables synthetic-cluster quality weighting.
- `stacking.cluster_quality_weighting.kappa_cluster`: weighting exponent.
- `stacking.cluster_quality_weighting.cap_enabled`: explicit dominance cap toggle.
- `stacking.cluster_quality_weighting.cap_ratio`: dominance cap level when enabled.
- **Runtime safeguard:** for synthetic stacking, a default dominance cap is applied even when `cap_enabled=false` to avoid diffuse-signal collapse from a few dominant clusters.
- `stacking.output_stretch`: optional linear post-scale of the output data to the full 16-bit range.
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
