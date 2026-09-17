# tile_compile_cpp example profiles

The example files in this folder document configuration for the only
reconstruction method: **CFA Forward Drizzle + Multiband**.

## Files

- `reconstruction_tuning.example.yaml`
  Canonical full OSC reference configuration, kept in sync with the production
  `tile_compile.yaml` tuning: `drizzle.pixfrac: 0.6`,
  `registration.prewarp_interpolation: cubic`,
  `chroma_denoise` active at `apply_stage: both` (pre-BGE and post-PCC) with
  `star_protection.threshold_sigma: 4` / `dilate_px: 8`
  (the dilate radius must cover the full PSF including wings — a too-small
  radius lets the chroma wavelet erode star wings into green halos — while
  the sigma threshold must stay sparse: at `threshold_sigma: 2` the dilated
  mask covers ~94% of a dense field and the denoiser degenerates to a no-op),
  `chroma_denoise.large_scale_bias.enabled: false` (default off — see
  tuning note 4 below before turning it on),
  `runtime_limits.memory_budget: 16384`.
  Use this file to diff your own configuration against current production
  values.

- `mono.example.yaml`
  Full reference for MONO datasets: `data.color_mode: MONO`,
  `chroma_denoise.enabled: false`, `pcc.enabled: false`.
  Shows only the overrides that differ from schema defaults.

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
