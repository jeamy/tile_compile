# Tile-Compile C++ Configuration Reference

This documentation describes all configuration options for `tile_compile.yaml` based on the C++ implementation in `configuration.hpp` and the schema files `tile_compile.schema.json` / `tile_compile.schema.yaml`.

**Source of truth for defaults:** `include/tile_compile/config/configuration.hpp`  
**Schema version:** v3  
**Reference:** Methodology v3.3

**Documentation status:**
- Single-method pipeline: CFA Forward Drizzle + Multiband is the only
  reconstruction method. The old `aqmh.*`, Classic and `method:` blocks no
  longer exist.
- `data.bayer_pattern` default is `auto`; FITS header (`BAYERPAT`/`COLORTYP`) takes precedence over the config value.
- Removed `data.linear_required` (deprecated; non-linear frames are now only warned about).
- `bge.fit.robust_loss` and `bge.fit.huber_delta` are documented and user-configurable.
- `bge.min_valid_sample_fraction_for_apply` and `bge.min_valid_samples_for_apply` are documented as BGE channel-apply guards.
- PCC coverage includes active stability and apply controls (`max_condition_number`, `max_residual_rms`, `apply_attenuation`, `chroma_strength`, `k_max`).
- `bge.tile_weight_lambda_structure` is aligned with the current default `1.0`.
- The classic/AQMH stacking parameters were removed; `stacking.common_overlap_required_fraction` lives on as `reconstruction.common_overlap_required_fraction` (§1, §14).

**💡 For practical examples and use cases, see:** [Configuration Examples & Best Practices](configuration_examples_practical_en.md)

## Table of Contents

1. [Pipeline](#1-pipeline)
2. [Output](#2-output)
3. [Data](#3-data)
4. [Linearity](#4-linearity)
5. [Calibration](#5-calibration)
7. [Normalization](#7-normalization)
8. [Registration](#8-registration)
9b. [Chroma Denoise](#chroma-denoise)
10. [Global Metrics](#10-global-metrics)
14. [Reconstruction](#14-reconstruction)
15. [Debayer](#15-debayer)
16. [Astrometry](#16-astrometry)
17. [BGE (Background Gradient Extraction)](#17-bge-background-gradient-extraction)
18. [PCC](#18-pcc)
19. [HyperMetric Stretch](#19-hypermetric-stretch)
20. [Stacking](#20-stacking)
22. [Runtime Limits](#22-runtime-limits)
23. [Raw Stack / Preprocessing](#raw-stack-preprocessing)

---

## 1. Pipeline

The pipeline is fixed to a single reconstruction method:
**CFA Forward Drizzle + Multiband** (`tile_compile_runner reconstruct`).
There is no method selector anymore.

### Removed legacy keys (migration)

The following keys were removed by the single-method cutover. When a config
still contains them they are discarded with a warning on load;
`tile_compile_cli migrate-config <in> <out>` writes a cleaned-up
configuration. Semantic method selectors are rejected fail-closed.

| Old key | Status |
|---------|--------|
| `method` | **Fail-closed** — there is exactly one method; remove the key or migrate |
| `reconstruction.engine` | **Fail-closed** — method selector |
| `aqmh.*` | removed; the whole block is discarded with a warning |
| `pipeline.*` | removed (unused) |
| `assumptions.*` | removed (reduced/emergency mode gating dropped) |
| `tile.*`, `tile_denoise.*`, `local_metrics.*`, `synthetic.*`, `validation.*` | removed (classic tile path) |
| `runtime_limits.allow_emergency_mode` | removed |
| `runtime_limits.tile_analysis_max_factor_vs_stack`, `runtime_limits.tile_reconstruction_diagnostics` | removed |
| `stacking.method`, `stacking.sigma_clip.*`, `stacking.cluster_quality_weighting.*`, `stacking.output_stretch`, `stacking.tile_common_valid_min_fraction`, `stacking.cosmetic_correction*` | removed (STACKING is a pass-through) |

**Mapped (migrated) keys:**

| Old | New |
|-----|-----|
| `stacking.common_overlap_required_fraction` | `reconstruction.common_overlap_required_fraction` |

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
| **Default** | `true` |

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
| **Values** | `auto`, `RGGB`, `BGGR`, `GRBG`, `GBRG`, `NONE` |
| **Default** | `"auto"` |

**Purpose:** Bayer matrix pattern for color filter arrays. `NONE` for monochrome data, `auto` to derive the pattern from the FITS header.

**Runtime behaviour:** For OSC data, the runner first reads `BAYERPAT` and `COLORTYP` from the FITS header. If present, that value is used and the config value is ignored. If the header contains no Bayer metadata, the configured `bayer_pattern` is used as a fallback. With `bayer_pattern: auto` and no header metadata, the run aborts with an error instead of guessing a camera-specific pattern.


## 4. Linearity

Input-frame linearity diagnostics. The check looks for evidence of non-linear preprocessing (stretch, curves, hard compression), but it is not a direct camera-response linearity test.

### `linearity.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enable/disable linearity diagnostics in phase 0 (SCAN_INPUT).

**Behavior:** Flagged frames are logged and kept by the current runner in warn-only mode.

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

**Purpose:** Minimum acceptable diagnostic score. If the sampled frames fall below this value, a warning is emitted.

### `linearity.strictness`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `strict`, `moderate`, `permissive` |
| **Default** | `"strict"` |

**Purpose:** Selects the diagnostic thresholds.

The hard decision is conservative and object-agnostic: robust distribution shape plus obvious hard clipping/compression. Spectral, gradient, and variance metrics remain diagnostic because linear frames can legitimately vary strongly across objects (empty fields, star clusters, nebulosity, galaxy cores, CFA texture).

- **`strict`**: Tightest diagnostic thresholds, recommended for raw/calibrated linear data.
- **`moderate`**: More tolerant diagnostics for lightly preprocessed data or difficult fields.
- **`permissive`**: High tolerance for known problematic data.

---

## 5. Calibration

Calibration frame processing.

### `calibration.use_bias` / `calibration.use_dark` / `calibration.use_flat`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` (all three) |

**Purpose:** Enable per-frame-type calibration stages.

- **`use_bias`**: Subtracts readout noise (offset pedestal) from every light frame.
- **`use_dark`**: Subtracts thermal noise (dark current) from every light frame. Requires matching exposure time.
- **`use_flat`**: Divides by flat-field to correct vignetting and dust shadows.

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
| **Values** | `background`, `median` |
| **Default** | `"background"` |

**Purpose:** Normalization method.

- **`background`**: robust background matching (recommended)
- **`median`**: median-based normalization (use for extended objects filling >10% of frame)

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

**Compatibility:** Historical parameter for the disabled blind-chain rescue. It is still parsed and serialized so existing configurations remain valid, but it does not affect the current `independent_global_consensus_v2` strategy. `0` still resolves to `clamp(N/10, 12, 50)` when legacy artifacts or tools inspect the value.

**Fallback behavior:** Unresolved frames are not recovered through chain depth. They are independently optimized by seeded ECC directly against the master reference, then fall back to astrometry or the global transform model.

----

### `registration.blind_chain_strong_anchor_cc`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | 0.01 |
| **Maximum** | 0.5 |
| **Default** | `0.08` |

**Compatibility:** Historical CC threshold for strong blind-chain anchors. The value remains schema- and parser-compatible but is not used by `independent_global_consensus_v2` for warp selection.

----

### `registration.blind_chain_drift_threshold_px`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | 0.5 |
| **Maximum** | 10.0 |
| **Default** | `2.0` |

**Compatibility:** Historical per-frame drift limit in pixels. Because current registration does not create neighboring-frame chains, this metric is non-applicable and the value has no effect on new runs.

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

### `registration.affine_refinement_enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Default-enabled conservative per-frame affine fine registration after the normal global registration. It detects stars on the reference and already warped proxy, builds mutual nearest-neighbor matches, and fits a small affine correction with RANSAC. The correction is applied only when the inlier count and spatial coverage are sufficient, scale/shear/rotation/center displacement remain conservative, matched-star median and p90 residuals both improve without RMS regression, and NCC/overlap do not regress. Otherwise the original warp is preserved.

**Units and applicability:** All residual, match-radius, and center-displacement gates use proxy pixels (normally half-resolution for OSC). Frames below the internal p90 trigger, the reference frame, and frames without enough distributed stars are non-applicable and remain unchanged. Set it to `false` for an unrefined control or diagnostic run.

---

### `registration.smooth_local_refinement_enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enable the default-gated smooth local per-frame correction after global registration and, when accepted, after affine refinement. Mutual nearest-neighbor star residuals fit a regularized 4x4 Gaussian inverse displacement field. A deterministic 25% held-out star set must improve in median, p90, and RMS; the full matched set, spatial coverage, maximum displacement, dense Jacobian/local-scale sampling, common-support NCC, and overlap must also pass.

**Units, limits, and interactions:** Residuals and the internal maximum displacement use proxy pixels; the displacement is scaled to full resolution during prewarp. Internal conservative limits include at least 32 matches, at least 24 training and 8 held-out stars, 15% convex-hull coverage, 1.5 proxy-pixel maximum displacement, Jacobian determinant 0.94–1.06, and local singular values 0.96–1.04. The model tapers to zero near the image boundary and is composed directly with the inverse global/affine map, avoiding a second full-resolution resampling. It currently applies only to MONO and OSC with `registration.debayer_first` and a known Bayer pattern; CFA-mosaic and unsupported color paths keep the unchanged global/affine warp. Any failed gate, non-applicable frame, or exception preserves the unchanged global/affine warp. Per-frame evidence is written to `global_registration.json`. Set it to `false` for a global/affine-only control or diagnostic run.

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

### `registration.prewarp_interpolation`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `bilinear`, `cubic`, `lanczos4` |
| **Default** | `"lanczos4"` |

**Purpose:** Interpolation method for the warp/resample steps in REGISTRATION (prewarp of frames into the shared output geometry). `lanczos4` is the sharpest/recommended method; `bilinear` is faster but softer.

---

### `registration.debayer_first`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** OSC only. When `true`, frames are demosaiced before the prewarp and the R/G/B planes are reconstructed directly. When `false`, a CFA prewarp is used and the final CFA stack is demosaiced after reconstruction. See §15.

---

### `registration.pre_debayer_method`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `linear`, `vng`, `edge_aware`, `mask` |
| **Default** | `"linear"` |

**Purpose:** Demosaicing method used on the `debayer_first` path.

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

## 9b. Chroma Denoise {#chroma-denoise}

Chroma (color) noise denoise for OSC/RGB data. Removes color noise blotches while preserving luma detail and star colors. Operates in a transformed color space (YCbCr or opponent) to isolate chroma from luma. Disabled by default (`enabled: false`); the examples below show it turned on.

### `chroma_denoise.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Enable chroma (color) noise denoise. Removes color noise blotches while preserving luma detail. Recommended for OSC data.

### `chroma_denoise.color_space`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `ycbcr_linear`, `opponent_linear` |
| **Default** | `"ycbcr_linear"` |

**Purpose:** Color space transform used to isolate chroma channels. `ycbcr_linear` is the project-specific invertible 0.25/0.5/0.25 linear transform, not a Rec.601/Rec.709 encoding. `opponent_linear` uses an opponent color space which may perform better for certain sensors.

### `chroma_denoise.apply_stage`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `post_stack_linear`, `post_pcc` |
| **Default** | `"post_pcc"` |

**Purpose:** Single pipeline stage for chroma denoise. `post_stack_linear` runs before BGE and PCC. `post_pcc` runs after photometric color calibration, catches noise re-amplified by PCC gains, and is the default.

### `chroma_denoise.protect_luma`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enables luma-derived star, structure and extended-source masks. They reduce chroma denoise strength and exclude foreground from the large-scale-bias estimate. Linear luma is always preserved exactly.

### `chroma_denoise.luma_guard_strength`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `0 – 1` |
| **Default** | `0.75` |

**Purpose:** Strength of the protection-mask guard. 0 disables mask influence; 1 keeps original chroma in fully protected pixels.

### `chroma_denoise.blend.mode`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `chroma_only` |
| **Default** | `"chroma_only"` |

**Purpose:** Blending mode between original and denoised image. Currently only `chroma_only` is supported.

### `chroma_denoise.blend.amount`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `0 – 1` |
| **Default** | `0.95` |

**Purpose:** Blend fraction between original and denoised chroma. 0 = no denoise, 1 = full denoise. Range 0–1. Recommended: 0.90–0.95.

### `chroma_denoise.star_protection.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enable star protection mask for chroma denoise. Prevents color noise removal from degrading star colors and halos.

### `chroma_denoise.star_protection.threshold_sigma`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `> 0` |
| **Default** | `2.5` |

**Purpose:** Sigma threshold for star detection in protection mask. Lower values detect more stars (more aggressive protection). Range > 0. Recommended: 2.5–3.5.

### `chroma_denoise.star_protection.dilate_px`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Range** | `>= 0` |
| **Default** | `2` |

**Purpose:** Dilation radius for star protection mask in pixels. Expands the protection zone around detected stars to cover halos. Range >= 0. Recommended: 2–4.

### `chroma_denoise.structure_protection.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enable structure protection for chroma denoise. Prevents denoise from blurring nebula edges, galaxy arms, and other fine structures.

### `chroma_denoise.structure_protection.gradient_percentile`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `0 – 100` |
| **Default** | `87` |

**Purpose:** Gradient percentile threshold for structure protection. Regions with gradient above this percentile are protected from denoise. Lower values protect more structure. Range 0–100. Recommended: 85–90.

### `chroma_denoise.chroma_wavelet.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enable wavelet-based chroma denoise. Decomposes chroma into wavelet levels and applies soft-thresholding to remove noise.

### `chroma_denoise.chroma_wavelet.levels`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Range** | `>= 1` |
| **Default** | `3` |

**Purpose:** Number of wavelet decomposition levels. More levels capture larger-scale noise patterns. Range >= 1. Recommended: 3.

### `chroma_denoise.chroma_wavelet.threshold_scale`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `> 0` |
| **Default** | `1.8` |

**Purpose:** Scale factor for wavelet noise threshold. Higher values remove more color noise. Range > 0. Recommended: 1.5–2.0.

### `chroma_denoise.chroma_wavelet.soft_k`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `> 0` |
| **Default** | `1.0` |

**Purpose:** Soft-thresholding parameter k. Controls smoothness of threshold transition. Higher values = smoother transition. Range > 0. Recommended: 1.0.

### `chroma_denoise.chroma_bilateral.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enable bilateral filter for chroma denoise. Edge-preserving smoothing of color channels — smooths color noise while keeping color edges intact.

### `chroma_denoise.chroma_bilateral.sigma_range`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `> 0` |
| **Default** | `2.0` |

**Purpose:** Multiplier on the measured unprotected background chroma sigma used as the bilateral range sigma. This is invariant to normalized versus ADU-valued inputs. Recommended: 2.0.

### `chroma_denoise.chroma_bilateral.sigma_spatial`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `> 0` |
| **Default** | `1.5` |

**Purpose:** Spatial sigma for bilateral filter. Controls the spatial extent of smoothing. Larger values smooth over larger areas. Range > 0. Recommended: 1.5–3.0.

### `chroma_denoise.extended_source_protection.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Protects extended smooth sources (galaxy disks, large nebulae, diffuse halos) from chroma denoising. Detection uses a heavily blurred luma map; pixels whose smoothed luma exceeds `sky_median + luma_sigma * sky_sigma` are blended back toward their original chroma. Enable whenever the field contains a galaxy or large nebula -- without it the wavelet threshold treats the object's smooth color excess as noise and erases it (observed as a green cast after stretching).

### `chroma_denoise.extended_source_protection.luma_sigma`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `1.0 – 5.0` |
| **Default** | `2.5` |

**Purpose:** Detection threshold in sky-sigma units above the smoothed sky median. Lower values protect more area (also faint halos); higher values protect only the brightest core. Keep the protected footprint small: protected pixels receive only `(1 - luma_guard_strength)` of the denoise and therefore keep near-raw chroma noise.

### `chroma_denoise.extended_source_protection.dilate_px`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Range** | `0 – 100` |
| **Default** | `30` |

**Purpose:** Dilation radius (pixels) applied after detection to cover the object's faint outskirts and PSF halos. Keep it tight -- every protected sky pixel keeps near-raw chroma noise.

### `chroma_denoise.large_scale_bias.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Removes large-scale chroma bias (background color gradients wider than the coarsest wavelet level) before the wavelet/bilateral stages. Estimates a coarse block-median surface from unprotected pixels only, fills blocks without unprotected coverage via normalized-convolution fill, smooths it, and subtracts `strength` of its deviation from the surface's own global background median (the overall background chroma level is preserved; only spatial variation is flattened). The correction is faded out inside the protection mask, where the surface is only interpolated and may be contaminated by unprotected faint halo. Requires a sufficiently dilated `extended_source_protection` mask for large objects.

### `chroma_denoise.large_scale_bias.block_size`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Range** | `>= 4` |
| **Default** | `32` |

**Purpose:** Grid cell size in pixels for the block-median bias estimate.

### `chroma_denoise.large_scale_bias.blur_sigma`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `>= 0` |
| **Default** | `24.0` |

**Purpose:** Gaussian sigma in pixels used to interpolate the block grid into a continuous surface. `0` disables smoothing.

### `chroma_denoise.large_scale_bias.strength`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `0 – 1` |
| **Default** | `1.0` |

**Purpose:** Fraction of the estimated bias deviation subtracted from the chroma planes.

---

## 10. Global Metrics

Global quality metrics.

### `global_metrics.weights.background`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `0.4` |

**Purpose:** Weight for background penalty term in the global quality score. Higher values penalize frames with uneven or elevated backgrounds more strongly. Range 0–1. Recommended: 0.4.

### `global_metrics.weights.noise`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `0.3` |

**Purpose:** Weight for noise penalty term in the global quality score. Higher values penalize noisy frames more strongly. Range 0–1. Recommended: 0.3.

### `global_metrics.weights.gradient`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `0.3` |

**Purpose:** Weight for structure/sharpness penalty term in the global quality score. Higher values favor frames with better sharpness/structure. Range 0–1. Recommended: 0.3.

### `global_metrics.clamp`

| Property | Value |
|----------|-------|
| **Type** | array [2 numbers] |
| **Default** | `[-3.0, 3.0]` |

**Purpose:** Clamp range for the composite global quality z-score before exponential weight mapping. Prevents extreme outliers from dominating. Recommended: [-3.0, 3.0].

### `global_metrics.adaptive_weights`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Auto-adjust global metric weights based on observed variance across frames. When true, weights are normalized by the inverse variance of each metric, giving less variable metrics more influence. Recommended: false (manual weights are more predictable).

### `global_metrics.weight_exponent_scale`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `1.0` |

**Purpose:** Exponential scaling factor for global weight separation. Controls how strongly quality differences translate into weight differences via `W_f = exp(k * Q_f)`. Values > 1 increase separation (good frames get more weight), < 1 soften it. Range > 0. Recommended: 1.0.

---

## 14. Reconstruction

The `reconstruction:` block groups the parameters of the CFA forward-drizzle reconstruction with multiband fusion. Besides `quality.pyramid` it contains `drizzle.*` (kernel, pixfrac, chunking, memory budget), `clipping.*` (robust sample clipping), `coverage_gate.*` (coverage gates), and `multiband.*` (band fusion); the full field list is in `tile_compile.schema.yaml`.

### `reconstruction.quality.pyramid.*` — Local quality maps

Controls the Laplacian pyramid for the local quality maps (Source Quality Maps) that Forward Drizzle uses to weight every image region of the input frames by sharpness and SNR. Produced in the `SOURCE_QUALITY_MAPS` phase.

#### `reconstruction.quality.pyramid.scales`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Range** | 1 – 8 |
| **Default** | `4` |

**Purpose:** Number of pyramid levels for the multi-scale analysis. More levels capture more spatial frequencies at the cost of compute time.

---

#### `reconstruction.quality.pyramid.base_window_px`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | 1 |
| **Default** | `4` |

**Purpose:** Window size in pixels at the lowest pyramid level for local metric computation.

---

#### `reconstruction.quality.pyramid.sharpness_weight`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | 0 |
| **Default** | `0.6` |

**Purpose:** Weight of the sharpness metric in the combined quality index `Q = sharpness_weight * Q_sharp + snr_weight * Q_snr`. Together with `snr_weight` it controls the relative importance of sharpness vs. SNR; the sum of both weights must be greater than 0.

---

#### `reconstruction.quality.pyramid.snr_weight`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | 0 |
| **Default** | `0.4` |

**Purpose:** Weight of the SNR metric in the combined quality index. Increase for heavily noisy data with large inter-frame quality spread.

---

#### `reconstruction.quality.pyramid.score_scale`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | >0 |
| **Default** | `1.8` |

**Purpose:** Scales the combined local quality score before the sigmoid. Higher values increase per-pixel quality-map selectivity so sharper frames are favored more strongly; the map remains bounded to `[0,1]`.

---

#### `reconstruction.quality.pyramid.artifact_sigma`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | >0 |
| **Default** | `3.0` |

**Purpose:** MAD multiplier for artifact detection. Pixels whose local deviation exceeds `artifact_sigma * MAD` are flagged as artifacts and receive reduced quality weight.

- **Higher (e.g. 7–10):** More tolerant of outliers — more pixels receive normal weight
- **Lower (e.g. 3–4):** More aggressive artifact suppression

---

#### `reconstruction.quality.pyramid.max_artifact_fraction`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | >0 – 1 |
| **Default** | `0.25` |

**Purpose:** Maximum tolerated artifact fraction per evaluation window. Windows with more artifacts than `max_artifact_fraction` are discarded entirely (no quality contribution).

- **Increase (e.g. 0.30–0.40):** When known tolerable artifacts are present (e.g. satellite trails)
- **Decrease:** Stricter quality gates

---

#### Further `reconstruction.*` keys

| Key | Type | Default | Purpose |
|-----|------|---------|---------|
| `reconstruction.common_overlap_required_fraction` | number | `1.0` | Minimum fraction of usable frames in which a pixel must be valid to belong to `COMMON_OVERLAP` (migrated from `stacking.common_overlap_required_fraction`). |
| `reconstruction.delete_source_cache_after_run` | boolean | `false` | Deletes the source/normalization cache after a successful run (disk space). |
| `reconstruction.keep_profile_cache_after_run` | boolean | `false` | Keeps the `drizzle_profile` cache for reuse/final-render paths. |
| `reconstruction.diagnostics.level` | enum | `summary` | Diagnostics verbosity: `summary`, `full`. |
| `reconstruction.diagnostics.preview_forward_drizzle_uniform` | boolean | `false` | Best-effort uniform-control diagnostic after SAMPLING_GEOMETRY (`forward_drizzle_uniform_diagnostic.json`). |
| `reconstruction.diagnostics.persist_forward_drizzle_uniform_store` | boolean | `false` | Transactionally persists unclipped uniform planes under `artifacts/forward_drizzle_uniform_store/`. |
| `reconstruction.drizzle.internal_scale` | integer | `2` | Internal working scale of the drizzle grid (`1` or `2`). |
| `reconstruction.drizzle.output_scale` | integer | `1` | Output raster relative to the internal grid (`1` or `2`, `<= internal_scale`). |
| `reconstruction.drizzle.kernel` | enum | `square` | Drizzle kernel; currently only `square`. |
| `reconstruction.drizzle.pixfrac` | number | `0.8` | Drop-shrink factor of contributions `(0, 1]`. |
| `reconstruction.drizzle.robust_passes` | integer | `2` | Robust reprojection passes `[1, 6]` (migrated from `clip_iterations`). |
| `reconstruction.drizzle.min_clip_contributors` | integer | `5` | Minimum contributors before sigma/MAD clipping activates. |
| `reconstruction.drizzle.full_frame_estimator` | boolean | `false` | Pilot + full-frame estimator: the hash-selected reservoir frames (at most 64) define the clip bounds, then every frame's folded candidate is tested against them and accumulated, so all frames determine the value. Runs on CPU and CUDA and reads quality maps for all frames; needs wide symmetric clip bounds (about `clip_sigma_low` = `clip_sigma_high` = 4) and `reconstruction.multiband.enabled`. Schema default `false`; the shipped `tile_compile.yaml` sets `true` (with clip 4/4). |
| `reconstruction.drizzle.chunk_rows` | integer | `0` | `0` = budgeted stripes (<=256 rows); `>0` is budget-checked. |
| `reconstruction.drizzle.chunk_halo_rows` | integer | `-1` | Compatibility field; exact footprint enumeration needs no output halo. |
| `reconstruction.drizzle.memory_budget_mb` | integer | `0` | MiB; `0` inherits `runtime_limits.memory_budget`. |
| `reconstruction.clipping.clip_sigma_low` / `clip_sigma_high` | number | `3.0` / `3.0` | MAD sigma thresholds of the robust contribution control. |
| `reconstruction.clipping.min_fraction` | number | `0.4` | Minimum usable sample fraction per pixel `(0, 1]`. |
| `reconstruction.clipping.min_n_eff` | number | `3.0` | Minimum effective contribution count for a valid output pixel (`>= 1`). |
| `reconstruction.clipping.guard_fallback` | boolean | `false` | `false` = strict veto (pixel stays unassigned); `true` = survivor/unclipped fallback instead of a black pixel. |
| `reconstruction.clipping.shared_frame_rejection` | boolean | `false` | Runs on both the CPU and CUDA backends. Reconciles the per-channel independent sigma-clip decisions across R/G/B: rejects a candidate frame in a channel that would otherwise have kept it, if enough other channels independently rejected it (see `shared_frame_rejection_consensus`). Addresses CFA-driven chroma noise (anti-correlated clip decisions across channels, since R/G/B are sampled from disjoint sensor pixels). A frame seen as a candidate by only one channel is left exactly as that channel's own clip decided. |
| `reconstruction.clipping.shared_frame_rejection_consensus` | number | `0.5` | `(0, 1]`. Fraction of the channels that saw a frame as a candidate that must have independently rejected it before `shared_frame_rejection` rejects it in every one of those channels. `1.0` effectively disables the consensus revision (bit-identical to `shared_frame_rejection: false`). |
| `reconstruction.clipping.bimodal_veto` | boolean | `false` | Runs on both the CPU and CUDA backends. After the asymmetric median/MAD clip pass converges, checks the accepted set for a coherent second population the ordinary bound alone did not separate from the majority (e.g. a subset of frames with a small but real registration/tracking offset at this exact pixel) -- more likely with wide `clip_sigma_low`/`clip_sigma_high`, which are deliberately loosened to preserve genuine signal variance and so also admit a coherent minority more readily. If the largest gap between consecutive accepted values exceeds `bimodal_veto_gap_sigma` times the pass's own MAD and splits off a minority side of at least 2 candidates holding less than half the accepted weight, that minority is dropped. Also refreshes the full-frame estimator's frozen pilot bounds when it fires, so the rejected population's non-pilot frames are not re-admitted later. |
| `reconstruction.clipping.bimodal_veto_gap_sigma` | number | `2.5` | `> 0`. Multiple of the clip pass's MAD the largest accepted-value gap must exceed before `bimodal_veto` treats the split as a coherent second population rather than ordinary spread. Lower values fire more readily. |
| `reconstruction.coverage_gate.min_frames` | integer | `2` | Minimum number of usable frames. |
| `reconstruction.coverage_gate.min_supported_fraction` | number | `0.995` | Minimum supported output-pixel fraction `(0, 1]`. |
| `reconstruction.coverage_gate.min_channel_n_eff_floor` | number | `3.0` | N_eff floor for a channel to count as covered. |
| `reconstruction.coverage_gate.min_channel_n_eff_fraction` | number | `0.15` | Minimum fraction of channels above the N_eff floor. |
| `reconstruction.coverage_gate.min_analysis_pixels` | integer | `1024` | Minimum number of analyzable pixels. |
| `reconstruction.coverage_gate.max_internal_hole_area_px` | integer | `0` | Maximum internal hole area in the footprint. |
| `reconstruction.multiband.enabled` | boolean | `true` | Enable multiband fusion. |
| `reconstruction.multiband.levels` | integer | `3` | Band levels `[1, 4]`; `>=2` requires `quality.pyramid.scales >= 2`. |
| `reconstruction.multiband.alpha_cap` | number | `1.0` | Upper bound of the blend alpha `[0, 1]`. |
| `reconstruction.multiband.fine_quality_exponent` / `medium_quality_exponent` | number | `4.0` / `2.0` | Quality exponents of the fine/medium band weights. |
| `reconstruction.multiband.min_quality_separation` / `full_quality_separation` | number | `0.05` / `0.20` | Quality separation thresholds of the band fusion. |
| `reconstruction.multiband.min_effective_samples` / `full_effective_samples` | number | `8.0` / `24.0` | N_eff thresholds for minimal/full band usage. |
| `reconstruction.multiband_validation.fwhm_ratio_max` | number | `0.95` | `> 0`; multiband median FWHM must be `<= x *` raw. |
| `reconstruction.multiband_validation.p90_fwhm_ratio_max` | number | `1.0` | `> 0`; multiband p90 FWHM vs. raw. |
| `reconstruction.multiband_validation.tail_ratio_max` | number | `1.1` | `> 0`; multiband star-tail metric vs. raw. |
| `reconstruction.multiband_validation.elongation_ratio_max` | number | `1.08` | `> 0`; multiband star elongation vs. raw. |
| `reconstruction.multiband_validation.background_rms_ratio_max` | number | `1.05` | `> 0`; background RMS vs. `drizzle_uniform` (raw veto and multiband promotion gate). Raise it when the sampled "background" contains real faint structure (e.g. a field-filling galaxy) whose higher RMS is signal, not noise. |
| `reconstruction.multiband_validation.seam_ratio_max` | number | `1.05` | `> 0`; seam contrast vs. `drizzle_uniform` at the support boundary. |
| `reconstruction.multiband_validation.min_stars_fwhm` | integer | `20` | `>= 0`; matched stars required for the FWHM metric. |
| `reconstruction.multiband_validation.min_stars_p90_tail_elongation` | integer | `30` | `>= 0`; matched stars for p90/tail/elongation metrics. |
| `reconstruction.multiband_validation.max_fwhm_ci_relative_width` | number | `0.1` | `> 0`; veto when the bootstrap 95% CI of the FWHM ratio is wider than this relative width. |

Fail-closed gates: `FORWARD_DRIZZLE` and `MULTIBAND` abort in a controlled
way when coverage/confidence gates or validation thresholds are not met.
When a multiband-validation gate rejects a candidate the run keeps the
`drizzle_uniform` control instead of failing.

---

## 15. Debayer

There is no standalone `debayer` config key.

- `OSC` with `registration.debayer_first: true` (default): frames are demosaiced before the prewarp; the R/G/B planes are reconstructed directly.
- `OSC` with `debayer_first: false`: CFA prewarp; the final CFA stack is demosaiced after reconstruction.
- `MONO`: no demosaicing.

---

## 16. Astrometry

Astrometry solving settings.

### `astrometry.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Enable/disable astrometry solving.

**Fallback behaviour:** ASTAP is tried first. If its quad matcher does not
produce a WCS, the runner performs an offline near-solve against the
Siril/Gaia DR3 catalogue already installed for PCC. The RGB stack must contain
`RA`, `DEC`, `FOCALLEN`, and either `XPIXSZ` or `YPIXSZ`. The fallback produces
a linear TAN/CD WCS only; it does not create SIP distortion terms. If the local
Gaia catalogue is unavailable or the star match is not robust, the existing
`solve_failed` behaviour is retained.

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

**Implementation note:** BGE uses tile-quality data derived from the reconstructed image for sample selection/weighting:
- `type` + `star_count`: star-dense STAR tiles are conservatively excluded or down-weighted.
- `fwhm`: scales effective star-mask dilation per tile.
- `quality_score`: applied as an additional tile-sample reliability factor.

**Key BGE parameters:**
- `bge.method`: Engine selector `none|classic|autobge|auto` (default: `none`). Sole on/off switch -- `none` disables BGE. `auto` measures the background gradient first and only runs AutoBGE when the gradient exceeds `auto_detect.gradient_threshold`.
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

**Recommendation:** Use `bge.method: auto` as the general production default — it skips BGE on flat fields (no gradient → no color corruption) and runs AutoBGE with extended-source exclusion on fields with real gradients. Use `none` to unconditionally skip, or `autobge` to unconditionally run.

### `bge.method`

| Property | Value |
|----------|-------|
| **Type** | string |
| **Values** | `none`, `classic`, `autobge`, `auto` |
| **Default** | `none` |

**Purpose:** Selects the BGE engine. `none` disables BGE, `classic` uses the existing grid/tile BGE implementation, `autobge` selects the two-stage poly+RBF AutoBGE implementation, and `auto` programmatically detects the background gradient and only triggers AutoBGE when the gradient amplitude exceeds `bge.auto_detect.gradient_threshold` — with an automatically derived extended-source exclusion mask so galaxy disks and large nebulae are excluded from the BGE fit. This is the sole on/off switch for BGE.

> **Migration note:** `bge.enabled` (a legacy boolean mirror of this field) has been removed. It could silently disagree with `bge.method` -- e.g. `enabled: false` next to a stale `method: classic` still ran BGE, because `method` was always authoritative whenever present. A config that still sets `bge.enabled` now fails to load with a validation error naming `bge.method`. Replace `enabled: true` with `method: classic` (or `autobge`), and `enabled: false` with `method: none`.

### `bge.auto_detect.gradient_threshold`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | `0` |
| **Default** | `0.05` |

**Purpose:** Minimum relative gradient amplitude to trigger AutoBGE when `bge.method: auto`. Amplitude = `(max − min of linear sky-plane fit) / sky_median` over the G channel. Values below this skip BGE. Range 0.01–0.50; recommended 0.05.

### `bge.auto_detect.extended_source_sigma`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | `0` |
| **Default** | `3.0` |

**Purpose:** Detection threshold for extended sources in the block-median grid. Blocks exceeding `sky_median + N × sky_sigma` are excluded from the AutoBGE sample set, so the poly/RBF fit uses only real sky pixels. Range 1.0–6.0; recommended 3.0.

### `bge.auto_detect.extended_source_dilate_px`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | `0` |
| **Default** | `50` |

**Purpose:** Morphological dilation (pixels) applied to the detected extended-source exclusion mask. Provides a safety margin around galaxy halos and nebula edges. Range 0–200; recommended 50.

### `bge.autobge.num_sample_points`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | `0` |
| **Default** | `0` |

**Purpose:** Number of AutoBGE sample points. `0` lets the implementation derive a deterministic image-size-based count (~1 point per 800 downsampled pixels, clamped to 200–3000). Higher values produce denser spatial sampling but increase runtime. Recommended: `0` (auto) for most cases, `800–1500` for large images with complex gradients.

### `bge.autobge.poly_degree`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Range** | `1 – 6` |
| **Default** | `2` |

**Purpose:** Degree of the first-stage polynomial fit. `2` = quadratic (captures broad gradients), `3` = cubic (captures complex asymmetric gradients). Higher degrees risk overfitting to image structure. Recommended: `2` for most datasets, `3` only for very strong asymmetric gradients.

### `bge.autobge.rbf_smooth`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | `0` |
| **Default** | `0.1` |

**Purpose:** RBF smoothing factor for the second-stage residual fit. Higher values produce a smoother background model but may underfit local gradients. Range `0.01–1.0`. Recommended: `0.1` for typical images, `0.5` for smooth gradients.

### `bge.autobge.downsample_scale`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | `1` |
| **Default** | `4` |

**Purpose:** Integer downscaling factor for the working image used in AutoBGE sampling and fitting. `4` = 4x downsample (16x fewer pixels). Higher values speed up computation but reduce spatial resolution of the background model. Range `1–8`. Recommended: `4` for full-resolution images, `2` for small images.

### `bge.autobge.patch_size`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | `3` |
| **Default** | `15` |

**Purpose:** Odd patch size used for local background sampling. Each sample point measures the background in a patch of this size. Larger patches are more robust but average over more structure. Range `3–31`. Recommended: `15`.

### `bge.autobge.patch_estimator`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `median`, `sigma_clipped_median` |
| **Default** | `"median"` |

**Purpose:** Local background estimator for each sample patch. `median` is fast and robust for most images. `sigma_clipped_median` iteratively rejects outliers and is better for fields with many stars or cosmic rays.

### `bge.autobge.stretch_mode`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `none`, `linear`, `mtf` |
| **Default** | `"linear"` |

**Purpose:** Working-space transform used only for AutoBGE sampling/fitting. `linear` is the conservative default because the later HyperMetric Stretch phase runs independently and BGE should preserve the additive linear-image contract. `mtf` is intended only for AutoBGE parity/experiments.

### `bge.autobge.stretch_target_median`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `(0, 1]` |
| **Default** | `0.25` |

**Purpose:** Target median for the working-space stretch (only used when `stretch_mode=mtf`). Controls the brightness of the stretched image used for sampling. Lower values sample darker background regions. Range `0.1–0.5`.

### `bge.autobge.border_margin`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | `0` |
| **Default** | `10` |

**Purpose:** Pixel margin excluded from sampling at the image border. Border pixels often contain stacking artifacts or vignetting. Increase for wide-field images with strong edge effects. Range `0–100`. Recommended: `10–30`.

### `bge.autobge.bright_exclusion_fraction`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `(0, 1)` |
| **Default** | `0.5` |

**Purpose:** Fraction of brightest pixels excluded from background sampling. `0.5` excludes the top 50% (conservative, good for nebula-rich fields). Lower values (`0.2–0.3`) include more pixels but risk contaminating the background model with bright structures. Range `0.1–0.8`.

### `bge.autobge.gradient_descent_max_iters`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | `1` |
| **Default** | `100` |

**Purpose:** Maximum iterations for the gradient-descent sample-point placement algorithm. Each iteration moves sample points toward dimmer local regions. Higher values find deeper background spots but increase runtime. Range `20–500`. Recommended: `100`.

### `bge.autobge.random_seed`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `42` |

**Purpose:** Random seed for deterministic sample-point generation. Same seed + same image = identical results. Change to get alternative sample placements for comparison.

### `bge.autobge.normalize_between_stages`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** When true, normalizes the residual image between the polynomial first stage and the RBF second stage. Prevents the RBF from re-fitting large-scale gradients already captured by the polynomial. Recommended: `true`.

### `bge.autobge.apply_guards`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** When true, AutoBGE uses the shared outer BGE apply guards (flatness/slope check) before mutating RGB output. Prevents degradation from poor fits. Recommended: `true`.

### `bge.autobge.mono_mode`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `rgb_duplicate`, `disabled` |
| **Default** | `"rgb_duplicate"` |

**Purpose:** Handling of mono (single-channel) images. `rgb_duplicate` copies the mono channel to R/G/B before BGE, allowing per-channel correction. `disabled` processes only the single channel. Use `rgb_duplicate` for OSC images debayered to mono.

### `bge.tile_weight_lambda_structure`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `> 0` |
| **Default** | `1.0` |

**Purpose:** Lambda in tile reliability weight `w_t = exp(-lambda * structure_score_t) * (1 - masked_fraction_t)`. Higher values down-weight structure-rich tiles more aggressively. Range `0.5–3.0`. Recommended: `1.0` for moderate fields, `2.0+` for dense nebulosity.

### `bge.sample_quantile`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `(0, 0.5]` |
| **Default** | `0.20` |

**Purpose:** Tile background quantile used to estimate robust background samples. Lower values (`0.10–0.15`) are more conservative, resistant to nebula contamination. `0.50` = median, suitable for heavily masked fields. Range `(0, 0.5]`.

### `bge.structure_thresh_percentile`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `0 – 1` |
| **Default** | `0.90` |

**Purpose:** Structure-threshold percentile used to reject structure-rich tiles or pixels from BGE sampling. `0.80` = moderate (excludes top 20%), `0.90` = strict (excludes top 10%). Lower values preserve more samples but risk structure contamination. Range `0.5–0.95`.

### `bge.min_tiles_per_cell`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | `1` |
| **Default** | `3` |

**Purpose:** Minimum number of valid tiles required per BGE grid cell. Cells with fewer valid tiles trigger the `insufficient_cell_strategy`. Range `1–10`. Recommended: `3`.

### `bge.mask.star_dilate_px`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | `0` |
| **Default** | `4` |

**Purpose:** Star-mask dilation radius in pixels. Expands the exclusion zone around detected stars to prevent star halos from contaminating background samples. Range `0–20`. Recommended: `4–6` for typical seeing, `8–12` for wide-field with bright stars.

### `bge.mask.sat_dilate_px`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | `0` |
| **Default** | `4` |

**Purpose:** Saturation-mask dilation radius in pixels. Expands the exclusion zone around saturated pixels/cores. Range `0–20`. Recommended: `4–6`, increase for sensors with strong blooming.

### `bge.grid.N_g`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | `1` |
| **Default** | `32` |

**Purpose:** Target BGE grid density. Grid cell size `G = min(W,H) / N_g`. Higher values create finer grids for better gradient capture but require more samples per cell. Range `16–64`. Recommended: `32–36` for typical DSO images, `48+` for wide-field.

### `bge.grid.G_min_px`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | `1` |
| **Default** | `64` |

**Purpose:** Minimum pixel size of a BGE grid cell. Prevents cells from becoming too small on large images. Range `32–128`. Recommended: `56–64`.

### `bge.grid.G_max_fraction`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `(0, 1]` |
| **Default** | `0.25` |

**Purpose:** Upper bound for grid-cell size relative to the image extent. Prevents cells from becoming too large on small images. Range `0.1–0.5`. Recommended: `0.25`.

### `bge.grid.insufficient_cell_strategy`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `discard`, `nearest`, `radius_expand` |
| **Default** | `"discard"` |

**Purpose:** Fallback strategy for grid cells with too few valid samples. `discard` excludes the cell from fitting (conservative). `nearest` fills from nearest valid cell. `radius_expand` enlarges the search radius to find more samples. Recommended: `radius_expand` for border cells.

### `bge.fit.method`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `rbf`, `poly`, `spline`, `bicubic`, `modeled_mask_mesh` |
| **Default** | `"rbf"` |

**Purpose:** Surface fitting method for the background model. `rbf` = Radial Basis Functions (flexible, recommended for most gradients). `poly` = robust polynomial (good for broad smooth gradients, faster). `spline` = thin-plate spline. `bicubic` = bicubic spline. `modeled_mask_mesh` = segmentation-based mesh fit for large nebulae.

### `bge.fit.irls_max_iterations`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | `1` |
| **Default** | `10` |

**Purpose:** Maximum IRLS (Iteratively Reweighted Least Squares) iterations. Higher values allow better convergence but increase runtime. Range `5–20`. Recommended: `10`.

### `bge.fit.irls_tolerance`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | `> 0` |
| **Default** | `1e-4` |

**Purpose:** Convergence tolerance for IRLS. Iteration stops when the parameter change falls below this value. Range `1e-6–1e-3`. Recommended: `1e-4`.

### `bge.fit.polynomial_order`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Values** | `2`, `3` |
| **Default** | `2` |

**Purpose:** Polynomial order when `bge.fit.method=poly`. `2` = quadratic (broad gradients, safe default). `3` = cubic (complex asymmetric gradients, higher overfitting risk). Recommended: `2`.

### `bge.fit.rbf_mu_factor`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | `> 0` |
| **Default** | `1.0` |

**Purpose:** RBF shape parameter `μ = rbf_mu_factor × G` (grid spacing). Controls the width of the basis functions. Higher values produce smoother surfaces. Range `0.5–3.0`. Recommended: `1.0–1.5`.

### `bge.fit.rbf_lambda`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | `> 0` |
| **Default** | `1e-6` |

**Purpose:** RBF regularization parameter λ. Prevents overfitting by penalizing large coefficients. Higher values = smoother but may underfit. Range `1e-6–0.1`. Recommended: `0.01–0.1`.

### `bge.fit.rbf_epsilon`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | `> 0` |
| **Default** | `1e-10` |

**Purpose:** Numerical stabilization epsilon for thin-plate RBF at d=0. Prevents division by zero. Range `1e-10–1.0`. Recommended: `1e-10` for thinplate, `1.0` for multiquadric.

### `bge.min_valid_sample_fraction_for_apply`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Range** | `(0.0, 1.0]` |
| **Default** | `0.30` |

**Purpose:** Per-channel safety gate. If `valid_tile_samples / total_tile_samples` falls below this fraction, BGE is skipped for that channel. Prevents poor-quality correction from sparse sampling. Range `0.1–0.5`. Recommended: `0.25–0.30`.

### `bge.min_valid_samples_for_apply`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Minimum** | `1` |
| **Default** | `96` |

**Purpose:** Absolute per-channel safety gate. If fewer than this many valid robust tile samples are available, BGE is skipped for that channel. Range `24–200`. Recommended: `96`.

### `bge.fit.robust_loss`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `huber`, `tukey` |
| **Default** | `"huber"` |

**Purpose:** Robust loss function for IRLS fitting. `huber` = quadratic for small residuals, linear for large (moderate outlier rejection). `tukey` = completely rejects large residuals (aggressive outlier rejection). Recommended: `huber`.

### `bge.fit.huber_delta`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Minimum** | `> 0` |
| **Default** | `1.5` |

**Purpose:** Huber loss transition parameter δ. Residuals below δ are quadratic, above δ are linear. Smaller δ rejects more outliers. Range `0.5–3.0`. Recommended: `1.5`.

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

**Purpose:** Enable/disable photometric color calibration. Matches catalog star colors to calibrate the RGB color balance of the stacked image.

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

**Purpose:** Faint magnitude limit for PCC catalog star matching. Higher values include fainter stars. CAUTION: for small sensors or dense star fields, mag_limit > 15 can include stars below detection threshold. Range 1–22. Recommended: 14.

### `pcc.mag_bright_limit`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `6.0` |

**Purpose:** Bright magnitude limit for PCC catalog star matching. Stars brighter than this are excluded (saturated stars give unreliable photometry). Range 0–15. Recommended: 6.

### `pcc.aperture_radius_px`, `pcc.annulus_inner_px`, `pcc.annulus_outer_px`

| Key | Type | Default |
|-----|------|---------|
| `pcc.aperture_radius_px` | number | `8.0` |
| `pcc.annulus_inner_px` | number | `12.0` |
| `pcc.annulus_outer_px` | number | `18.0` |

**Purpose:** Aperture/annulus geometry for star photometry. `aperture_radius_px` is the photometric aperture radius, `annulus_inner_px` and `annulus_outer_px` define the sky annulus for local background estimation. Used when `radii_mode=fixed`.

### `pcc.min_stars`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `10` |

**Purpose:** Minimum number of matched catalog stars required for PCC to proceed. Below this, PCC is skipped. Range >= 3. Recommended: 10.

### `pcc.sigma_clip`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `2.5` |

**Purpose:** Sigma clipping threshold for PCC outlier rejection. Stars with residuals > sigma_clip × std are rejected. Range > 0. Recommended: 2.5.

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

**Purpose:** Controls post-PCC background neutralization independently of `chroma_strength` and matrix type. `always` forces neutral background offsets, `off` preserves the per-channel backgrounds, and `auto` fully neutralizes a spatially coherent global color cast while attenuating or skipping correction for locally varying nebulosity or field structure. Diagonal PCC gains are applied around each channel background and therefore do not neutralize it implicitly.

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
| `hypermetric_stretch.highlight_ceiling_percentile` | number | `100.0` | 90 - 100 |
| `hypermetric_stretch.color_cast_correction.enabled` | boolean | `false` | Adaptive average-neutral green-cast correction after HMS (off by default). |
| `hypermetric_stretch.color_cast_correction.max_amount` | number | `1.0` | Upper limit of the automatic strength, (0, 1]. |
| `hypermetric_stretch.color_cast_correction.target_ratio` | number | `1.0` | Median G/((R+B)/2) on object pixels the correction aims for, (0.5, 1.5]. |
| `hypermetric_stretch.color_cast_correction.min_excess` | number | `1.02` | No change if the measured ratio is at or below this value, [1, 2] (protects green/teal objects). |
| `hypermetric_stretch.color_cast_correction.brightness_bins` | integer | `8` | Brightness classes of the object pixels, [1, 32]; the strength is chosen per class and interpolated over the pixel brightness (1 = one global strength). |
| `hypermetric_stretch.color_cast_correction.neutralize_sky` | boolean | `false` | Also shift G so the sky is neutral: `G += (sky R + sky B)/2 - sky G` (skipped above 25 % offset). |
| `hypermetric_stretch.color_cast_correction.object_sigma` | number | `3.0` | Object pixels: blurred luminance above sky + k sigma, (0, 50]; brightest 1 % excluded. |
| `hypermetric_stretch.write_channels` | boolean | `false` | |
| `hypermetric_stretch.output_rgb` | string | `stacked_rgb_hms.fits` | non-empty |

`ready_to_use` follows the VeraLux GUI default with output scaling to `target_bg` and final soft clip. `scientific` skips the ready-to-use final scaling/soft clip and allows `linear_expansion`.

**`highlight_ceiling_percentile`:** in `ready_to_use`, `adaptive_output_scaling` computes the final contrast scale as `min(contrast_scale, physical_scale)`, where `physical_scale` by default (`100`) is chosen so the **true brightest real pixel** never exceeds 1.0. On a target with one very bright, compact highlight (e.g. a nebula core), this caps the contrast of the **entire** frame far below what the rest of the image could otherwise use — on a real M42 run this left `physical_scale` at only ~0.6% of `contrast_scale`, even though `black_clip_percent`/`white_clip_percent` both stayed exactly `0.0`. A lower value (e.g. `99.9`) replaces the exact max pixel with a percentile, deliberately clipping a small, bounded fraction of the brightest pixels in exchange for materially more contrast in stars/highlight regions. Important: this only moves the top ~1-2% of the tone range — the median/background stays pinned exactly at `target_bg` (the final MTF match anchors it there regardless of the ceiling value). To also brighten the dark/mid-tone body (sky, faint nebulosity), raise `target_bg` as well — both levers are independent and additive, not alternatives.

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

In the forward-drizzle pipeline the classic stacking step is a
pass-through: the reconstructed image comes from FORWARD_DRIZZLE/MULTIBAND;
no stack methods, sigma clips or cluster weightings are executed anymore.
Only the optional per-frame cosmetic correction before the warp remains.

### `stacking.per_frame_cosmetic_correction`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Hot-pixel correction **per frame before the warp**.

This targets **fixed sensor defects** (RGB single-pixel speckles) that appear at the same coordinates in every frame and therefore are not reliably removed by the robust contribution control in FORWARD_DRIZZLE.

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

**Autogrow:** The configured value is a starting point, not a hard ceiling. When the planned working set (e.g. retained registration proxies, source images, stripe scratch) does not fit, the memory planner raises the effective budget in 1 GiB steps while free RAM remains (capped at ~80% of currently available memory). Each increase is logged as a run warning. If the working set still does not fit at the headroom cap, the run keeps failing closed with `DRIZZLE_MEMORY_BUDGET`.

### `runtime_limits.hard_abort_hours`

| Property | Value |
|----------|-------|
| **Type** | number |
| **Default** | `6.0` |

**Purpose:** Hard upper runtime limit in hours.

**Runtime behavior:** Checked after major phase boundaries in the main run and
resume path. Exceeding the limit aborts the run with
`runtime_limit_exceeded`.

### `runtime_limits.acceleration_backend`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Default** | `auto` |
| **Valid values** | `auto`, `cpu`, `opencv_cuda`, `opencv_opencl`, `opencl`, `cuda` |

**Purpose:** GPU acceleration backend selection for the warp/resample steps (REGISTRATION prewarp).

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

## 23. Raw Stack / Preprocessing {#raw-stack-preprocessing}

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
| `hypermetric_stretch.highlight_ceiling_percentile` | number | `100.0` |
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

- `output.registered_dir`: target folder name for registered frame outputs.
- `output.write_registered_frames`: writes per-frame registered FITS; increases IO and disk usage significantly.
- `output.crop_to_nonzero_bbox`: crops the final stack to its non-empty bounding box.
- `data.image_width`, `data.image_height`: optional expected dimensions; normally auto-detected from FITS headers.
- `data.color_mode`: expected acquisition mode; runtime auto-detection can override with warning.
- `data.bayer_pattern`: CFA layout for OSC processing and color reconstruction consistency.

### A.2 Linearity / Calibration

- `linearity.enabled`: enables linearity diagnostics in scan/early validation.
- `linearity.max_frames`: sample size for linearity checks (tradeoff speed vs certainty).
- `linearity.min_overall_linearity`: warning threshold for the linearity diagnostic score.
- `linearity.strictness`: threshold preset for robust distribution and clipping diagnostics.
- `calibration.use_bias`, `use_dark`, `use_flat`: activate master-frame correction stages.
- `calibration.bias_use_master`, `dark_use_master`, `flat_use_master`: use explicit master files vs building from directories.
- `calibration.dark_auto_select`: auto-match dark masters by exposure (and optional temperature).
- `calibration.dark_match_exposure_tolerance_percent`: allowed exposure mismatch for dark matching.
- `calibration.dark_match_use_temp`: toggles temperature-aware dark matching.
- `calibration.dark_match_temp_tolerance_c`: allowed temperature mismatch when temp matching is active.
- `calibration.bias_dir`, `darks_dir`, `flats_dir`: source folders for calibration frame discovery.
- `calibration.bias_master`, `dark_master`, `flat_master`: explicit master calibration file paths.
- `calibration.pattern`: file glob for calibration frame loading.

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
- `registration.prewarp_interpolation`: interpolation method of the warp/resample steps.
- `registration.debayer_first`: OSC — demosaic before prewarp instead of CFA prewarp.
- `registration.pre_debayer_method`: demosaicing method for the `debayer_first` path.
- `dithering.enabled`: enables dither diagnostics output in registration artifacts.
- `dithering.min_shift_px`: minimum frame-to-frame shift to count as dither.

### A.4 Chroma denoise


- `chroma_denoise.enabled`: enables chroma-focused denoise (OSC path). Default: `false` (opt-in).
- `chroma_denoise.color_space`: chroma/luma transform (`ycbcr_linear` or `opponent_linear`).
- `chroma_denoise.apply_stage`: selects one supported stage: after the final linear stack or after PCC.
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
- `chroma_denoise.extended_source_protection.enabled`: protect extended smooth sources (galaxy disks, large nebulae) from chroma denoising. Enable whenever the field contains a galaxy or large nebula to prevent the wavelet threshold from erasing real object color.
- `chroma_denoise.extended_source_protection.luma_sigma`: detection threshold above sky background in sigma units (range 1.0–5.0). Lower values protect more area.
- `chroma_denoise.extended_source_protection.dilate_px`: dilation radius (pixels) after detection, to cover PSF halos.
- `chroma_denoise.large_scale_bias.enabled`: removes large-scale chroma bias (sky color gradients wider than the coarsest wavelet level) before the wavelet/bilateral stages.
- `chroma_denoise.large_scale_bias.block_size`: block size in pixels for the block-median bias-surface estimate (>= 4).
- `chroma_denoise.large_scale_bias.blur_sigma`: smoothing sigma in pixels that interpolates the block grid into a continuous surface (>= 0).
- `chroma_denoise.large_scale_bias.strength`: fraction of the estimated deviation subtracted ([0,1]; the global background median is preserved, only spatial variation is flattened). The estimate uses unprotected pixels only, and the correction is faded out inside the protection mask -- there the surface is only interpolated and may be contaminated by unprotected faint halo. Large extended sources therefore require a sufficiently dilated `extended_source_protection` mask.

### A.4b Luma denoise

Standalone luminance denoise stage between post-stack and multiband
(default: disabled). Addresses noise already present in the reconstructed
luminance signal before it reaches multiband fusion -- unlike
`chroma_denoise`, which only smooths the color components.

- `luma_denoise.enabled`: enables the luma wavelet denoise stage. Default: `false` (opt-in).
- `luma_denoise.luma_guard_strength`: strength of the luma guard against detail loss ([0,1]).
- `luma_denoise.blend_amount`: blend fraction between original and denoised luma ([0,1]).
- `luma_denoise.star_protection.enabled`: star-mask protection so star cores/halos stay sharp.
- `luma_denoise.star_protection.threshold_sigma`: detection threshold for star mask creation.
- `luma_denoise.star_protection.dilate_px`: star mask growth radius.
- `luma_denoise.structure_protection.enabled`: edge/structure-aware protection of fine detail (e.g. faint nebula structure).
- `luma_denoise.structure_protection.gradient_percentile`: gradient cutoff for the structure mask.
- `luma_denoise.wavelet.enabled`: wavelet soft-threshold denoise.
- `luma_denoise.wavelet.levels`: number of wavelet decomposition levels.
- `luma_denoise.wavelet.threshold_scale`: wavelet threshold multiplier.
- `luma_denoise.wavelet.soft_k`: softness of wavelet shrinkage.
- `luma_denoise.extended_source_protection.enabled`: protects diffuse nebulae/large galaxies from luma denoising, mirroring `chroma_denoise.extended_source_protection`. Default: `false`. Needed because neither `star_protection` (point sources only) nor `structure_protection` (steep local gradients only) covers broad, low-contrast but real nebulosity.
- `luma_denoise.extended_source_protection.luma_sigma`: detection threshold above sky background, in sigma. Recommended: 2.5.
- `luma_denoise.extended_source_protection.dilate_px`: dilation radius after detection. Recommended: 30.
- `luma_denoise.bilateral.enabled`: bilateral filter applied after the wavelet stage, mirroring `chroma_denoise.chroma_bilateral`. Default: `false`. The wavelet reconstruction always adds its coarsest approximation level back unmodified -- that residual carries real background noise no wavelet strength removes; bilateral targets it directly.
- `luma_denoise.bilateral.sigma_spatial`: spatial sigma. Recommended: 1.5; for background noise closer to a DWARF-onboard-style live-stack, 6-10.
- `luma_denoise.bilateral.sigma_range`: multiplier of the measured background luma sigma. Recommended: 2.0.

Reconstruction is additive (`R_new = R + (Y_denoised - Y)` etc.), not
ratio-based -- a ratio reconstruction amplifies noise in faint/partially
protected regions and produces dark pixels/chroma fringing at star edges.

### A.5 Global metrics / Reconstruction

- `global_metrics.weights.background`, `noise`, `gradient`, `fwhm`, `roundness`, `star_count`: weighted terms composing the per-frame global quality score.
- `global_metrics.clamp`: hard bounds before exponential weight mapping.
- `global_metrics.adaptive_weights`: auto-adapt metric weights from observed dispersion.
- `global_metrics.weight_exponent_scale`: controls separation strength in `exp(k*Q)` mapping.
- `reconstruction.common_overlap_required_fraction`: coverage threshold for `COMMON_OVERLAP`.
- `reconstruction.delete_source_cache_after_run`, `keep_profile_cache_after_run`: cache lifecycle after the run.
- `reconstruction.diagnostics.level`: diagnostics verbosity (`summary`/`full`).
- `reconstruction.diagnostics.preview_forward_drizzle_uniform`, `persist_forward_drizzle_uniform_store`: uniform-control diagnostics/persistence.
- `reconstruction.drizzle.internal_scale`, `output_scale`: internal vs. output drizzle raster.
- `reconstruction.drizzle.kernel`, `pixfrac`: kernel and drop-shrink of forward-drizzle contributions.
- `reconstruction.drizzle.robust_passes`: robust reprojection passes.
- `reconstruction.drizzle.min_clip_contributors`: minimum contributors for sample clipping.
- `reconstruction.drizzle.full_frame_estimator`: all frames instead of only the reservoir determine the value (pilot clip bounds frozen, default off).
- `reconstruction.drizzle.chunk_rows`, `chunk_halo_rows`, `memory_budget_mb`: streaming/memory profile of the reprojector.
- `reconstruction.clipping.clip_sigma_low`, `clip_sigma_high`: MAD rejection thresholds.
- `reconstruction.clipping.min_fraction`, `min_n_eff`: coverage gates per pixel.
- `reconstruction.clipping.guard_fallback`: fallback instead of strict veto on clip failure.
- `reconstruction.clipping.shared_frame_rejection`, `shared_frame_rejection_consensus`: cross-channel consensus against anti-correlated CFA chroma noise (CPU and CUDA); see the §14 table above.
- `reconstruction.clipping.bimodal_veto`, `bimodal_veto_gap_sigma`: rejects a coherent minority population (e.g. a registration-drift-affected subset of frames) the ordinary wide clip bound alone admits; see the §14 table above.
- `reconstruction.coverage_gate.*`: fail-closed coverage/quality gates before FORWARD_DRIZZLE.
- `reconstruction.multiband.*`: band-fusion control (levels, alpha cap, quality/N_eff thresholds).
- `reconstruction.quality.pyramid.*`: local quality maps (see §14).

### A.6 Debayer / Astrometry / PCC / HMS / Stacking / Runtime

- Debayer is controlled via `registration.debayer_first`/`pre_debayer_method` (see §15); there is no standalone `debayer` block.
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
- `hypermetric_stretch.highlight_ceiling_percentile`: percentile ceiling (instead of the true max pixel) for ready_to_use's highlight-protection scale; lower than 100 trades bounded clipping for more contrast, without moving the target_bg-pinned background.
- `hypermetric_stretch.write_channels`, `output_rgb`: HMS output controls.
- `runtime_limits.parallel_workers`: upper bound for worker threads.
- `runtime_limits.memory_budget`: memory budget that can cap effective parallelism.
- `runtime_limits.hard_abort_hours`: absolute runtime safety stop.

## Forward Drizzle v2: streaming and memory budget

FORWARD_DRIZZLE exclusively uses the transactional v2 band path; there is no
method or environment-variable switch to the old producer. Source, quality and
launched-sample regions are band-bounded, one CPU/CUDA workspace is reused across
all bands, and MULTIBAND requires the published v2 store fail-closed. CUDA device
failures restart the entire unpublished phase on `cpu_v2`. The separate CPU
coverage/Uniform preview remains a diagnostic disabled by default.

| Parameter | Units, range and default | Behavior |
|---|---|---|
| `reconstruction.drizzle.memory_budget_mb` | MiB, integer >=0, default 0 | 0 inherits `runtime_limits.memory_budget`; direct library calls use 512 MiB. Accounts for retained output/masks, one source plus transient load copy, stripe scratch and reserve. Available host/cgroup headroom can further reduce the budget. When undersized it autogrows in 1 GiB steps (logged as warnings) up to the headroom cap; see `runtime_limits.memory_budget`. |
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

The shared M3 Uniform/Raw library path additionally budgets one clipping candidate per frame, pixel and channel in the worst case, plus two output stripes. Its materializing wrapper also charges both complete outputs. A check after building candidate lists cannot replace this preflight. The current diagnostic profile store still materializes Uniform, but exports its planes without extra full-image copies.

`reconstruction.diagnostics.persist_forward_drizzle_uniform_store` (boolean, default `false`) is independent of preview. When enabled, it streams unclipped Uniform planes into `artifacts/forward_drizzle_uniform_store/generation-…/`; `current.json` publishes the complete verified generation atomically. The existing drizzle budget includes an additional 8 MiB FITS/metadata reserve and one float row. Insufficient memory fails before source loading; insufficient free disk fails before plane writing. A failed diagnostic does not fail the production run. Old generations are retained and consume disk; there is no automatic cleanup. This is a diagnostic store, not a resumable pipeline phase. Read `current.json` and validate it against the expected source, sampling and algorithm identity; old flat stores are not implicitly accepted or rewritten. The shared clipped Uniform/Raw library store is now a dev/regression oracle only and is not wired into production.

The checked source-quality library path has an explicit MiB budget and preflights a conservative 128 bytes per source pixel plus source/load buffers, metadata and metric scratch. Large native images may fail early. The shared drizzle budget also includes the supplied dense per-source quality weight vector. Store commit schema 2 additionally binds normalized-cache and quality-plan hashes; older diagnostic stores are not accepted implicitly as checked predecessors. These library APIs do not enable pipeline resume.
