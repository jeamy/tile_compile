# tile_compile_cpp example profiles

The example files in this folder document configuration for the only
reconstruction method: **CFA Forward Drizzle + Multiband**.

## Files

- `reconstruction_tuning.example.yaml`
  Canonical full OSC reference configuration, kept in sync with the production
  `tile_compile.yaml` tuning: `drizzle.pixfrac: 0.6`,
  `registration.prewarp_interpolation: cubic`,
  `chroma_denoise` configured for the single `post_pcc` stage with
  `star_protection.threshold_sigma: 4` / `dilate_px: 8`
  (the dilate radius must cover the full PSF including wings — a too-small
  radius lets the chroma wavelet erode star wings into green halos — while
  the sigma threshold must stay sparse: at `threshold_sigma: 2` the dilated
  mask covers ~94% of a dense field and the denoiser degenerates to a no-op),
  `chroma_denoise.large_scale_bias.enabled: false` (default off — see
  tuning note 4 below before turning it on),
  `luma_denoise.enabled: false` (default off — new luminance denoise stage,
  runs once on the post-stack linear RGB before BGE/PCC/HMS; the pipeline
  had no luma denoise of its own before this),
  `runtime_limits.memory_budget: 16384`.
  Use this file to diff your own configuration against current production
  values.

  **Shipped defaults:** `tile_compile.yaml` and this file now enable
  `reconstruction.drizzle.full_frame_estimator: true` with
  `reconstruction.clipping.clip_sigma_low/high: 4/4` (code and schema default
  stay `false` and `3.0`, so configs that omit the keys keep the legacy
  behaviour). The mode needs `reconstruction.multiband.enabled`, costs about
  +65 to +70 % FORWARD_DRIZZLE time and is ~25x slower on the host kernel, so
  keep it for CUDA runs; the runner warns when no CUDA device is used.

- `mono.example.yaml`
  Full reference for MONO datasets: `data.color_mode: MONO`,
  `chroma_denoise.enabled: false`, `pcc.enabled: false`.
  Shows only the overrides that differ from schema defaults.

- `m42_dwarf2_full_frame.example.yaml`, `m31_dwarf2_full_frame.example.yaml`
  Complete DWARF II (OSC) profiles of the two reference datasets (M42, 610 x
  10 s, gain 60; M31, 645 lights, gain 80) with the pilot + full-frame
  estimator enabled: `reconstruction.drizzle.full_frame_estimator: true` and
  the wide, symmetric clip bounds it needs (`reconstruction.clipping.
  clip_sigma_low` / `clip_sigma_high` both `4`). All frames, not only the
  hash-selected reservoir (about 64 per pixel), determine the pixel value.
  Against a matched control with the same config and
  `full_frame_estimator: false`: M42 sky sigma 1.47 to 0.57, faint-nebula
  contrast in the HMS product 5.9/6.4 to 12.6/14.0; M31 sky sigma 2.26 to
  0.95, faint outer-arm contrast 4.2/4.0 to 8.4/8.0. The mode costs about
  +65 to +70 % FORWARD_DRIZZLE time (Q maps for all frames) and is off by
  default. Adapt the placeholder paths (`/path/to/...` for the dark
  master and the astap/siril data) before running. Background and
  measurements: `docs/dynamic_boost_implementierungsplan_2026-09-19_de.md`.

- `m42_dwarf2_full_frame_luma.example.yaml`, `m31_dwarf2_full_frame_luma.example.yaml`
  The two profiles above with the recommended `luma_denoise` stage enabled:
  wavelet + `extended_source_protection`, bilateral off. Each `luma_denoise`
  block is commented parameter by parameter. Against the same profile without
  luma denoise: sky sigma x0.44 on both targets; faint-structure contrast in
  the HMS product 12.6/14.0 -> 16.7/18.8 (M42, above the DWARF reference
  15.2/16.8) and 8.4/8.0 -> 11.8/11.2 (M31); star flux and size unchanged; no
  new sky structure. The bilateral filter and stronger wavelet settings were
  not better (details in the plan). The stage is applied to the working image
  only; `outputs/stacked_rgb.fits` stays the plain reconstruction.

- Colour-cast correction (`hypermetric_stretch.color_cast_correction`):
  optional adaptive average-neutral (SCNR-style) correction of a green cast,
  applied to the stretched RGB after HMS. The strength is chosen automatically
  per brightness class (`brightness_bins`, default 8) from the object pixels,
  so a cast that is stronger in a bright core than in faint arms is corrected
  in both; a class without a measured cast is left unchanged (protects
  genuinely green/teal objects). It corrects the excess ABOVE the sky and does
  not change the sky level unless `neutralize_sky: true` is set, which also shifts G so the sky itself is neutral (a tinted sky otherwise stays tinted). Enabled in the M31
  profiles (green excess above the sky: arms 1.73 -> 1.19, core 1.30 -> 1.02),
  disabled in the M42 profiles (measured 0.78, unchanged either way); off in
  `tile_compile.yaml`.

- `forward_drizzle_streaming.example.yaml`
  Minimal fragment documenting the bounded drizzle streaming/memory options
  (`chunk_rows`, `chunk_halo_rows`, `memory_budget_mb`, diagnostic store).
  Merge into a complete configuration; it is not standalone.

## Removed legacy profiles

The old per-scenario profiles (m104, ic434, M42, M45, m66, bright_star,
canon_*, small_n, medium_n, large_n, mono_small_n_*,
smart_telescope_*, very_bright_star_anti_seam,
ic434_background_gradient, m31_background_gradient_balanced) were removed.
They targeted the retired scenario system (`scenario_profile`, `run_dir`,
`log_level`, `input` top-level keys) that the current schema rejects, and
carried AQMH-era tuning values. Their content survives in git history if a
specific tuning needs to be consulted.

## Generic tuning notes

The snippets below are configuration fragments, not standalone files.

### 1) Equatorial (Canon/DSLR, guided)

Use stricter rejection to prevent false matches:

```yaml
registration:
  transform_model: similarity
  star_topk: 120
  star_inlier_tol_px: 2.5
  reject_cc_min_abs: 0.30
  reject_shift_px_min: 40.0
  reject_shift_median_multiplier: 3.0
reconstruction:
  clipping:
    clip_sigma_low: 2.0
    clip_sigma_high: 2.0
    min_fraction: 0.5
```

- **Why:** EQ sequences usually have small drift/rotation; stricter gates
  reduce misregistration risk.

### 2) Alt/Az (smart telescope, rotation-heavy)

Use tolerant rejection because large shift/rotation is physically expected:

```yaml
registration:
  transform_model: affine
  star_topk: 150
  star_inlier_tol_px: 4.0
  reject_cc_min_abs: 0.25
  reject_shift_px_min: 100.0
  reject_shift_median_multiplier: 5.0
reconstruction:
  clipping:
    clip_sigma_low: 1.8
    clip_sigma_high: 1.8
    min_fraction: 0.2
```

- **Why:** Alt/Az near pole can show wide shift distributions; too strict
  settings reject too many usable frames.
- For no-dark OSC sessions with fixed hot pixels, prefer enabling
  `stacking.per_frame_cosmetic_correction` (pre-stack) instead of trying to
  force this via more aggressive sigma clipping.

### 3) Small-N MONO anti-grid

Prefer seam stability over aggressive enhancement:

```yaml
reconstruction:
  clipping:
    clip_sigma_low: 2.5
    clip_sigma_high: 2.5
    min_fraction: 0.5
    min_n_eff: 3.0
  drizzle:
    min_clip_contributors: 3
global_metrics:
  adaptive_weights: false
```

- **Why:** with low N, a higher survivor fraction and lower contributor
  requirements avoid empty pixels while still rejecting outliers.

### 4) Diffuse emission nebula vs. compact object (`chroma_denoise.large_scale_bias`)

`large_scale_bias` fits a smooth "background" color surface from every pixel
*outside* `extended_source_protection` (plus star/structure protection) and
subtracts it. It is safe only when that mask covers the target's full visual
extent — otherwise real object color outside the mask gets read as bias and
removed.

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

- **Why:** on a real M42 run, `luma_sigma: 2.5` (the schema default) covered
  only `extended_source_protected_fraction ≈ 0.01` (`artifacts/chroma_denoise.json`)
  — just the bright Trapezium core. The remaining ~99% of the frame, mostly
  real colored nebulosity, was treated as background and subtracted, leaving
  a blue ring at the mask boundary and yellow/green blotches across the
  nebula (`large_scale_bias_removed_rms_c1/c2` was clearly > 0 in the
  artifact). A `luma_sigma` scan on the same frame found no safe middle
  ground either: `1.0` → ~7% coverage, `0.75` → already ~49% — there is no
  threshold between "misses the nebula" and "protects half the frame".
- **Diagnose it:** compare `extended_source_protected_fraction` in
  `artifacts/chroma_denoise.json` against the target's true visual extent
  (not just against `extended_source_sky_sigma`). A large gap plus a
  nonzero `large_scale_bias_removed_rms_c1`/`_c2` is this failure mode.
- Both C++ struct default and schema default for `large_scale_bias.enabled`
  are `false` (opt-in); only turn it on for compact targets where the
  mask coverage is verified.

### Forward-drizzle streaming (diagnostics)

`memory_budget_mb: 0` inherits the runner budget; `chunk_rows: 0` chooses at
most 256 target rows within it. Large materialized outputs fail before
allocation; the preview uses a streaming summary sink. Coverage retains two
byte masks and spools exact weighted-`n_eff` quantiles to temporary disk.

`reconstruction.diagnostics.persist_forward_drizzle_uniform_store` (boolean,
default `false`) is independent of preview. When enabled, it streams unclipped
Uniform planes into
`artifacts/forward_drizzle_uniform_store/generation-…/`; `current.json`
publishes the complete verified generation atomically. The existing drizzle
budget includes an additional 8 MiB FITS/metadata reserve and one float row.
Insufficient memory fails before source loading; insufficient free disk fails
before plane writing. A failed diagnostic does not fail the run. Old
generations are retained and consume disk; there is no automatic cleanup.
This is a diagnostic store, not a resumable pipeline phase.
