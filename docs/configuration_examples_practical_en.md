# Practical Configuration Examples for tile_compile

**[🇩🇪 Deutsche Version](configuration_examples_practical_de.md)**

This guide complements the configuration reference with practical examples, edge cases, and use cases based on methodology v3.3.

## Update Status (single-method cutover)

- The pipeline is fixed to **CFA Forward Drizzle + Multiband** (`tile_compile_runner reconstruct`); there is no method selector anymore.
- `method`, `pipeline.mode`, `aqmh.*`, `tile.*`, `tile_denoise.*`, `local_metrics.*`, `synthetic.*`, `validation.*`, `assumptions.*` and the classic `stacking.*` fields were removed; the old-to-new key mapping is in configuration reference §1.
- The former `aqmh.reconstruction.*` tuning values live on under `registration.*` (prewarp/debayer) and `reconstruction.drizzle.*` / `reconstruction.clipping.*`.
- `aqmh.pyramid.*` is now `reconstruction.quality.pyramid.*`.
- `stacking.common_overlap_required_fraction` is now `reconstruction.common_overlap_required_fraction`.

**Base snippet (single method):**

```yaml
registration:
  engine: triangle_star_matching
  enable_star_pair_fallback: true
  prewarp_interpolation: lanczos4
  debayer_first: true
  pre_debayer_method: linear

reconstruction:
  common_overlap_required_fraction: 1.0
  diagnostics:
    level: full
  drizzle:
    robust_passes: 4
  clipping:
    clip_sigma_low: 2.0
    clip_sigma_high: 4.0
    min_fraction: 0.4
    min_n_eff: 2.0
```

---

## Reconstruction (CFA Forward Drizzle + Multiband)

Reconstruction is the only method and always active. The most relevant
tunables are the robust contribution control (`clipping.*`), the
streaming/memory controls (`drizzle.*`) and the local quality maps
(`quality.pyramid.*`).

**Production profile (recommended, aligned with `tile_compile.yaml`):**

```yaml
registration:
  affine_refinement_enabled: true       # applied only when all residual/NCC/overlap gates pass
  smooth_local_refinement_enabled: true # extra held-out/Jacobian guard; atomic warp fallback otherwise
  prewarp_interpolation: lanczos4       # sharpest interpolation; cubic/linear are faster fallbacks
  debayer_first: true                   # OSC: demosaic before prewarp, reconstruct RGB directly
  pre_debayer_method: linear            # demosaicing method of the debayer_first path

reconstruction:
  delete_source_cache_after_run: true   # delete cache after a successful run (disk space)
  diagnostics:
    level: full
  drizzle:
    robust_passes: 4                    # robust reprojection passes
  clipping:
    clip_sigma_low: 2.0                 # lower MAD threshold (more aggressive)
    clip_sigma_high: 4.0                # upper MAD threshold (more tolerant)
    min_fraction: 0.4                   # minimum usable sample fraction
    min_n_eff: 2.0                      # min. effective contribution count per pixel
    guard_fallback: false               # false = strict veto on clip failure
  quality:
    pyramid:
      scales: 4
      base_window_px: 4
      sharpness_weight: 0.6   # sharpness weight in the quality index
      snr_weight: 0.4         # SNR weight in the quality index
      score_scale: 1.8        # selectivity of the local quality maps
      artifact_sigma: 3.0     # MAD multiplier for artifact detection
      max_artifact_fraction: 0.25  # max. artifact fraction per window
```

**More tolerant of artifacts (satellites, clouds):**

```yaml
reconstruction:
  quality:
    pyramid:
      artifact_sigma: 5.0
      max_artifact_fraction: 0.35
  clipping:
    clip_sigma_low: 1.5
```

**Memory-saving (large sessions, low RAM):**

```yaml
reconstruction:
  drizzle:
    memory_budget_mb: 1024   # explicit budget; 0 inherits runtime_limits.memory_budget
    chunk_rows: 0            # 0 = budgeted stripes (<=256 rows)

runtime_limits:
  parallel_workers: 2
  memory_budget: 1024
```

**Conservative against black artifact pixels:**

```yaml
reconstruction:
  clipping:
    guard_fallback: true     # use survivor/unclipped value instead of the veto
```

**Field-filling objects (e.g. large galaxies):**

The `background_rms` gate measures noise in regions that contain real faint
structure for field-filling objects. The weighted stack may then show more
"RMS" while actually preserving more signal - and the gate rejects it in
favor of the unweighted control (detail loss). Diagnosis: check the report /
`forward_drizzle.json` whether `selected_candidate` fell back to
`drizzle_uniform` although `background_rms` only barely exceeded the
threshold.

```yaml
reconstruction:
  multiband_validation:
    background_rms_ratio_max: 1.15   # more tolerant when "background" holds structure
```

Loosen the remaining `multiband_validation.*` gates (FWHM, elongation,
seam) only with evidence from the validation artifacts.

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

Since the forward-drizzle cutover, AutoBGE anchors all three channels to a
shared pedestal (darkest model median) instead of each channel's own model
median. This equalizes channel-dependent background pedestals before
PCC/HMS. If the slope guard still fires (typically because the model failed
to absorb a channel offset), a re-anchored apply is tested that equalizes
the residual medians of all channels exactly; it is only accepted when the
channel-level spread measurably decreases and the residual tilt stays
bounded (ratio bound or absolute amplitude <= 25% of the removed spread).
This is recorded as `guard_override: "level_equalization"` in
`artifacts/bge.json` (per channel
`guard_reason: "slope_worsened_but_level_equalized"`). Without a level
improvement the guard still discards the correction entirely.

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

## Chroma Denoise / Background Color Bias (`chroma_denoise.*`)

Default: disabled (opt-in), like every denoise stage (`chroma_denoise.enabled: false`).

**When to enable:**
- Chroma noise ("confetti") in the background after the stack
- Broad color casts/blotches that `chroma_wavelet`/`chroma_bilateral` alone
  don't remove

**`large_scale_bias` — only for objects whose full extent is covered by the mask:**

`large_scale_bias` fits a smooth "background" color surface from every pixel
**outside** `extended_source_protection` (plus star/structure protection)
and subtracts it. This is safe only when that mask covers the visible
object completely — otherwise real object color outside the mask gets read
as bias and removed.

```yaml
# Compact object (e.g. a galaxy like M31): a genuinely flat sky remains
# outside the object, so extended_source_protection can cover it fully.
chroma_denoise:
  extended_source_protection:
    enabled: true
    luma_sigma: 2.5
    dilate_px: 15
  large_scale_bias:
    enabled: true
    block_size: 32
    blur_sigma: 24.0
    strength: 1.0
```

```yaml
# Large diffuse emission nebula (M42-class): the nebula fills most of the
# frame and fades gradually into the sky — no luma threshold isolates
# "just the nebula" from background. Keep large_scale_bias off.
chroma_denoise:
  extended_source_protection:
    enabled: true
    luma_sigma: 2.5
    dilate_px: 15
  large_scale_bias:
    enabled: false
```

- **Background:** on a real M42 run, `luma_sigma: 2.5` (the schema default)
  covered only `extended_source_protected_fraction ≈ 0.01`
  (`artifacts/chroma_denoise.json`) — just the bright Trapezium core. The
  remaining ~99% of the frame, mostly real colored nebulosity, was treated
  as background and subtracted, leaving a blue ring at the mask boundary
  and yellow/green blotches across the nebula
  (`large_scale_bias_removed_rms_c1`/`_c2` was clearly > 0). A `luma_sigma`
  scan on the same frame found no safe middle ground either: `1.0` → ~7%
  coverage, `0.75` → already ~49% — there is no threshold between "misses
  the nebula" and "protects half the frame".
- **Diagnose it:** compare `extended_source_protected_fraction` in
  `artifacts/chroma_denoise.json` against the target's true visual extent
  (not just against `extended_source_sky_sigma`). A large gap plus a
  nonzero `large_scale_bias_removed_rms_c1`/`_c2` is this failure mode.
- Both the C++ struct default and the schema default for
  `large_scale_bias.enabled` are `false` (opt-in); only enable it for
  compact targets where mask coverage is verified.
- With `reconstruction.diagnostics.level: full`, the runner writes the star,
  structure and extended-source component masks plus the effective denoise
  amount map as FITS artifacts in addition to the combined protection mask.
  This separates halo detection from mask-transition effects.

---

## Luminance denoise (`luma_denoise.*`)

**When to enable:** fine-grained luminance noise in the reconstructed image
that is already visible on the reconstructed linear image before BGE/PCC (not just color noise --
that's `chroma_denoise`'s job).

```yaml
luma_denoise:
  enabled: true
  luma_guard_strength: 0.85
  blend_amount: 0.85
  star_protection:
    enabled: true
    threshold_sigma: 6
    dilate_px: 8
  structure_protection:
    enabled: true
    gradient_percentile: 90
  wavelet:
    enabled: true
    levels: 3
    threshold_scale: 1.5
    soft_k: 1.0
```

- **Background:** the previous architecture had a luminance denoise stage
  that was removed in a cutover and never replaced. `luma_denoise` runs by
  default **on the reconstructed linear image before BGE/PCC** -- ahead of `chroma_denoise`,
  which only smooths the color components.
- Reconstruction is additive (`R_new = R + (Y_denoised - Y)` etc.), not
  ratio-based. An earlier implementation used `R * (Y_denoised / Y)`; that
  amplifies noise in faint or partially protected regions and produced dark
  single pixels and chroma fringing at star edges in real M42 test runs.
  The additive form is exact for the 0.25/0.5/0.25 luma weighting and
  preserves every color difference.
- `star_protection`/`structure_protection` keep star sharpness and fine,
  faint nebula detail (e.g. in M42) from being blurred by the wavelet
  soft-thresholding -- check a crop preview before enabling on
  structure-rich targets.
- Default: disabled (opt-in), like every denoise stage.

---

## Cross-channel CFA consensus against chroma noise (`reconstruction.clipping.shared_frame_rejection`)

**When to enable:** fine color speckle/blotching around stars or in
structure-rich regions that persists after `chroma_denoise`/`luma_denoise`
and correlates with dropping cross-channel correlation (a *negative*
`corr(R,B)` in the crop is the characteristic signal, not positive -- real
point sources correlate positively across channels).

```yaml
reconstruction:
  clipping:
    shared_frame_rejection: true
    shared_frame_rejection_consensus: 0.5
```

- **Root cause:** R/G/B are reconstructed from disjoint sensor pixels
  (CFA-aware forward drizzle, without prior debayering: R≈1/4, G≈1/2, B≈1/4
  of pixels). The sigma clip in `finalize()` decides which frames are
  outliers independently per (pixel, channel) -- a deliberate architectural
  tradeoff, but one that lets a frame be rejected in one channel and kept in
  another even though both sample the same physical scene at slightly
  offset sensor positions. That produces anti-correlated noise between
  channels that looks like color speckle.
- `shared_frame_rejection` reconciles that decision across channels: a
  frame is rejected in a channel even if that channel's own clip pass kept
  it, if the fraction of channels that independently rejected it exceeds
  `shared_frame_rejection_consensus` (default `0.5` = majority). A frame
  seen as a candidate by only one channel is left untouched by the
  consensus rule -- there is nothing to vote against.
- Runs on both the CPU and CUDA backends (CUDA uses a separate three-kernel
  implementation -- build, cross-channel vote, reduce -- verified bit-exact
  against the original, non-SFR CUDA kernel at `consensus: 1.0`).
- `shared_frame_rejection_consensus: 1.0` effectively disables the
  consensus revision (bit-identical to `shared_frame_rejection: false`) --
  useful as a control run.
- Default: disabled (opt-in).

---

## Color noise around stars (`chroma_denoise`/`luma_denoise` protection mask)

**Symptom:** visible colored rings/halos around stars and elevated color
noise inside the protection zones (`star_protection`, `structure_
protection`, `extended_source_protection`), with a noticeable transition
right at the mask boundary.

**Cause:** `star_protection`/`extended_source_protection` reduce denoise
strength inside their mask (`luma_guard_strength`) but don't drop it to 0 --
the intent is "weaker inside, not off". Two separate issues contributed to
the visible transition:

1. **Hard mask edge:** unlike the other two mask components, `structure_
   protection` was never feathered -- a plain per-pixel threshold. As of
   this session all three mask components are feathered at their own
   scale before being combined.
2. **Large strength jump:** with the defaults (`blend.amount: 1.0`,
   `luma_guard_strength: 0.85`), denoise strength drops from ~100% outside
   to ~15% inside a fully protected zone -- a big step that a feathered
   mask edge alone doesn't fully hide.

**Pull both ends closer together:**

```yaml
chroma_denoise:
  blend:
    amount: 0.9        # was 1.0 -- slightly less aggressive outside
  luma_guard_strength: 0.5   # was 0.85 -- notably more effect inside
luma_denoise:
  blend_amount: 0.8     # was 0.85
  luma_guard_strength: 0.5   # was 0.85
```

- **Measured** (real M42 test runs, color-noise standard deviation R-G/B-G
  in a ring around a test star, same run, only these parameters changed):

  | Configuration | std(R-G) | std(B-G) |
  |---|---|---|
  | Default (`amount: 1.0`, `guard: 0.85`) | 21.05 | 25.26 |
  | Aggressive (`amount: 0.8`, `guard: 0.5`) | 7.92 | 9.24 |
  | Recommended (`amount: 0.9`, `guard: 0.5`) | 8.22 | 9.58 |

- **Trade-off:** a lower `blend.amount` reduces denoise strength
  *everywhere*, not just at the transition -- in the aggressive case,
  isolated dark speckle pixel count in a test crop rose from 65 to 206
  (only to 151 at `amount: 0.9`). `amount: 0.85-0.9` is usually a good
  middle ground between ring reduction and background noise.
- Default is unchanged (`blend.amount: 1.0`, `luma_guard_strength: 0.85`) --
  these values are opt-in tuning for targets with visible star halos, not
  a new baseline.

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

**More dynamics/"punch" (consumer-stack-style look):**

In `ready_to_use` mode, `adaptive_output_scaling` computes the final contrast scale as `min(contrast_scale, physical_scale)`, where `physical_scale` by default is chosen so the **single brightest real pixel** (e.g. a very bright, compact nebula core) never exceeds 1.0. On a real M42 run this made `physical_scale` only **0.6%** of `contrast_scale` — the rest of the frame was compressed into a tiny fraction of the achievable contrast, even though `black_clip_percent`/`white_clip_percent` both stayed exactly `0.0`. Consumer stacks (e.g. DWARF II's onboard processing) go the opposite way: they deliberately blow out the core to give the rest more contrast.

```yaml
hypermetric_stretch:
  enabled: true
  mode: ready_to_use
  target_bg: 0.20                    # lifts sky/faint nebulosity uniformly
  highlight_ceiling_percentile: 99.9  # 100 = never clip (default); lower = deliberate, bounded clipping of the brightest pixels for more contrast
```

Important, backed by numbers from the same simulation:
- `highlight_ceiling_percentile` alone moves **only the top ~1-2%** of the brightness distribution (stars, core edge) — the median/background stays pinned exactly at `target_bg` (the final MTF match anchors it there regardless of the ceiling value). p20/p50/p90 percentiles change by < 2%.
- To also brighten/"fill out" the **dark/mid-tone** area (sky, faint nebula wisps), `target_bg` must be raised as well — it scales p20/p50/p90 almost proportionally.
- Both levers are independent and **additive**, not alternatives.
- Remaining gap to a strongly saturated consumer look (e.g. DWARF II): pure **color saturation** — there is currently no HMS parameter for that; `color_grip`/`chroma_strength` only control how strongly color is pulled into the stretch, not the overall saturation afterward.
- `highlight_ceiling_percentile` is clamped to `[90, 100]` (validated); values below that would clip too large a share of the frame.

---

## Large-scale contrast (`hypermetric_stretch.large_scale_contrast`)

**When to enable:** The stretched image looks monotone in the background although the data carry large-scale structure (dust lanes, reflection
nebulae, glow around bright stars). The global stretch curve squeezes such faint, extended structure into a narrow band; a local contrast would amplify
the noise with it. This stage lifts only the large-scale component: pixel noise, star profiles and fine detail stay unchanged. Default: off.

```yaml
hypermetric_stretch:
  large_scale_contrast:
    enabled: true
    amount: 2.0          # 1 = double, 2 = triple the visible structure (the gain at sigma_px 48 is about 60 % of that)
    sigma_px: 48.0       # structure smaller than about this value is not lifted
    chroma_amount: 2.0   # large-scale colour differences R-G / B-G; 0 = colour unchanged
    remove_vignette: true
```

- **Measured** (IC4605, stretch-step output): pixel noise x1.000, star width x1.000, star signal x1.000; large-scale sky structure x1.8 / x2.5 / x3.3 for
  `amount` 1 / 2 / 3. The stage can be tested without a new reconstruction: `resume-reconstruction --from-phase HYPERMETRIC_STRETCH`
  (only the `hypermetric_stretch` section may change).
- **Vignette:** Without flat-field calibration a vignette is part of the large-scale structure. `remove_vignette: true` (default) removes a radially symmetric
  component before the boost; turn it off for centred, radially symmetric objects (e.g. a large central nebula).
- **Gradients:** Residual light-pollution gradients are lifted too. BGE remains responsible for them; note that `bge.method: classic` also removes two thirds of the
  large-scale nebula structure (see `pi_jev_effektgroessen_nachgelagert_20260926.md`), so: BGE only for real gradients, large-scale contrast afterwards if needed.
- **Diagnostics:** The `phase_end` event of `HYPERMETRIC_STRETCH` carries `large_scale_contrast` with `status`, `span_before`, `span_after` (p5-p95 of the coarse
  map) and `vignette_removed`.

---

## Common overlap (`reconstruction.common_overlap_required_fraction`)

**Current sensible default:**

```yaml
reconstruction:
  common_overlap_required_fraction: 1.0
```

- `1.0` enforces the strict intersection of all usable frames.
- Lower values re-admit partially covered edge pixels into metrics, BGE/PCC, and background statistics.

**Recommendations by setup:**

- **Alt/Az with field rotation:** keep `1.0` (recommended)
- **EQ with very stable tracking:** keep `1.0` when you want neutral border/background statistics
- **Only when intentionally accepting more edge area:** for example `0.98` or `0.95`

---

## Diagnose visible boundaries / artifacts

The forward-drizzle pipeline no longer produces tile seams (no
overlap-add stacking). When visible artifacts appear, check:

- `reconstruction.coverage_gate.*` — coverage gates before FORWARD_DRIZZLE
- `reconstruction.clipping.*` — too-aggressive sigma values can discard
  signal; too-loose values keep outliers
- `reconstruction.clipping.guard_fallback` — `false` leaves unassignable
  pixels black; `true` falls back to survivor/unclipped values
- `reconstruction.diagnostics.level: full` — maximum diagnostics artifacts
- downstream differences from `BGE` or `PCC`

---

## Hot pixels / RGB single-pixel artifacts (fixed sensor defects)

If the final image still shows **isolated red/green/blue single pixels**, these are typically **fixed hot pixels** (sensor defects) that occur at the same coordinates in every frame. They can survive stack sigma clipping because they are not outliers across frames.

**Recommendation:** Correct hot pixels **per frame before stacking**.

```yaml
stacking:
  per_frame_cosmetic_correction: true
  per_frame_cosmetic_correction_sigma: 5.0
```

---

## Audit Note on Removed Parameters

With the single-method cutover the blocks `aqmh.*`, `tile.*`,
`tile_denoise.*`, `local_metrics.*`, `synthetic.*`, `validation.*`,
`assumptions.*`, `pipeline.*`, `method` and the classic `stacking.*` fields
(`method`, `sigma_clip.*`, `cluster_quality_weighting.*`, `output_stretch`,
`tile_common_valid_min_fraction`, `cosmetic_correction*`) were removed.
The mapping of migrated keys is in configuration reference §1.

The practical examples below now use only parameters that are active in the current code and schema.

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

- Full example file: [`reconstruction_tuning.example.yaml`](https://github.com/jeamy/tile_compile/blob/master/tile_compile_cpp/examples/reconstruction_tuning.example.yaml)
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
- This stronger separation is also shown in [`reconstruction_tuning.example.yaml`](https://github.com/jeamy/tile_compile/blob/master/tile_compile_cpp/examples/reconstruction_tuning.example.yaml).

**Softer weighting for homogeneous sessions:**
```yaml
global_metrics:
  adaptive_weights: false
  weight_exponent_scale: 0.8
```

---

## Local Quality Maps (`reconstruction.quality.pyramid.*`)

Local per-pixel weighting is driven by the source-quality pyramid
(phase `SOURCE_QUALITY_MAPS`), not by tile metrics anymore.

**Near-default / robust:**
```yaml
reconstruction:
  quality:
    pyramid:
      scales: 4
      base_window_px: 4
      sharpness_weight: 0.6
      snr_weight: 0.4
      score_scale: 1.8
      artifact_sigma: 3.0
      max_artifact_fraction: 0.25
```

**Favor sharpness (seeing-limited sessions):**
```yaml
reconstruction:
  quality:
    pyramid:
      sharpness_weight: 0.7
      snr_weight: 0.3
      score_scale: 2.5
```

**Favor SNR (noisy, heterogeneous sessions):**
```yaml
reconstruction:
  quality:
    pyramid:
      sharpness_weight: 0.4
      snr_weight: 0.6
```

**Capture more spatial frequencies:**
```yaml
reconstruction:
  quality:
    pyramid:
      scales: 6
      base_window_px: 4
```

---

## Runtime Limits (`runtime_limits.*`)

The former reduced/emergency mode gating (`assumptions.*`,
`runtime_limits.allow_emergency_mode`) is gone; the pipeline has a single
mode. The runtime limits remain relevant:

```yaml
runtime_limits:
  parallel_workers: 8        # parallel workers
  memory_budget: 4096        # MiB; additionally caps parallelism
  hard_abort_hours: 6.0      # hard runtime limit
  acceleration_backend: auto # auto | cpu | opencv_cuda | opencv_opencl | opencl
```

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

## Performance Optimization (`runtime_limits.*`, `output.*`)

**Fast debug run:**
```yaml
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

```

### DSLR on equatorial mount

```yaml
data:
  color_mode: OSC
  bayer_pattern: auto

```

Ready-to-use repository profiles:
- `tile_compile_cpp/examples/reconstruction_tuning.example.yaml` (OSC, canonical)
- `tile_compile_cpp/examples/mono.example.yaml` (MONO)

### Mono on a large telescope

```yaml
data:
  color_mode: MONO

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

The CPU coverage/Uniform path processes target stripes instead of full-canvas
accumulators per frame or worker. The preview remains disabled by default.

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

For shared Uniform/Raw library calls, also prefer `chunk_rows: 0`: candidate storage grows with the number of frames, so more frames may require smaller stripes at the same image dimensions. If one row cannot fit, the call fails early. The streaming API avoids retaining both complete outputs in RAM; its sink must consume stripes immediately.
`reconstruction.diagnostics.persist_forward_drizzle_uniform_store` (boolean, default `false`) is independent of preview. When enabled, it streams unclipped Uniform planes into `artifacts/forward_drizzle_uniform_store/generation-…/`; `current.json` publishes the complete verified generation atomically. The existing drizzle budget includes an additional 8 MiB FITS/metadata reserve and one float row. Insufficient memory fails before source loading; insufficient free disk fails before plane writing. A failed diagnostic does not fail the run. Old generations are retained and consume disk; there is no automatic cleanup. This is a diagnostic store, not a resumable pipeline phase. Read `current.json` and validate it against the expected source, sampling and algorithm identity; old flat stores are not implicitly accepted or rewritten.

The checked predecessor library API uses an explicit source-quality MiB budget (512 MiB by default). It may reject large native frames under its conservative scratch estimate; do not bypass that check. Cache manifests identify existing normalized raw float files and do not perform calibration. Commit schema 2 binds cache and quality-plan hashes.
