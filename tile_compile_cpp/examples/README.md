# tile_compile_cpp example profiles (Methodik v3.3)

All example files in this folder are **complete standalone configurations** and include
all currently available config options with inline explanations.

They are kept in sync with v3.3 runner/config parser defaults, including:

- `dithering.*`
  - documents acquisition dithering expectation and shift threshold used for diagnostics.
- `chroma_denoise.*`
  - includes full OSC-oriented chroma-noise reduction block with inline comments.
  - contains preset hints directly in YAML comments:
    - conservative
    - balanced
    - aggressive
  - in MONO profile this block is intentionally present but disabled for completeness.
- `stacking.per_frame_cosmetic_correction*`
  - optional per-frame hot-pixel correction before stacking.
  - recommended when fixed sensor defects (RGB single-pixel speckles) survive sigma clipping.
- `stacking.cluster_quality_weighting.*`
  - includes full v3.2.2 cluster-quality weighting block:
    - `enabled`
    - `kappa_cluster`
    - `cap_enabled`
    - `cap_ratio`
  - documents the weighting model used in final cluster aggregation:
    - `w_k = exp(kappa_cluster * Q_k)`
  - includes optional dominance cap semantics:
    - `w_k <= cap_ratio * median_j(w_j)` (only when `cap_enabled: true`)
- `stacking.common_overlap_*`
  - includes strict common-overlap defaults for all profiles:
    - `common_overlap_required_fraction: 1.0`
    - `tile_common_valid_min_fraction: 0.90`
  - keeps post-PREWARP statistics and tile processing on shared valid regions only.
  - helps avoid edge-driven bias and tile/grid artifacts in rotating-field datasets.
- `registration.reject_*` outlier filtering
  - includes configurable global-registration outlier rejection:
    - `reject_outliers`
    - `reject_cc_min_abs`
    - `reject_cc_mad_multiplier`
    - `reject_shift_px_min`
    - `reject_shift_median_multiplier`
    - `reject_scale_min`
    - `reject_scale_max`
  - rejected registration frames are logged as `warning` events in
    `logs/run_events.jsonl` and summarized in `phase_end(REGISTRATION)` extras.
- `bge.*` (Background Gradient Extraction, v3.3 §6.3)
  - **NEW in v3.3**: Optional pre-PCC background gradient removal
  - Removes large-scale gradients (light pollution, moonlight, airglow) before color calibration
  - Applied directly to RGB channels before PCC (not diagnostics-only)
  - Tile-based sampling with configurable quantile (default: 20th percentile)
  - Coarse grid aggregation to avoid overfitting small-scale structure
  - Multiple surface fitting methods:
    - `rbf`: Radial Basis Functions (Multiquadric, Thin-plate, Gaussian)
    - `poly`: Robust polynomial (order 2-3)
    - `spline`: Thin-plate spline
    - `modeled_mask_mesh`: Foreground-aware mesh sky model for large diffuse targets (e.g. M31/M42)
  - Weighted regularized robust regression (IRLS with Huber/Tukey loss)
  - Adaptive grid spacing scales with image dimensions
  - Writes `artifacts/bge.json` with per-channel diagnostics (samples, grid cells, residual stats)
  - Included in `generate_report.py` as BGE section with dedicated plots
  - Includes deterministic autotune block:
    - `bge.autotune.enabled`
    - `bge.autotune.strategy` (`conservative|extended`)
    - `bge.autotune.max_evals`
    - `bge.autotune.holdout_fraction` (`0.05..0.50`)
    - `bge.autotune.alpha_flatness`
    - `bge.autotune.beta_roughness`
  - **Disabled by default** - enable with `bge.enabled: true` when gradients are present
  - Recommended for urban/suburban imaging or when PCC shows color bias across the field
- `pcc.*` (Photometric Color Calibration, v3.3.6 §6.4)
  - Includes local annulus background model:
    - `pcc.background_model` (`median|plane`)
  - Includes FWHM-adaptive radii controls:
    - `pcc.radii_mode` (`fixed|auto_fwhm`)
    - `pcc.aperture_fwhm_mult`
    - `pcc.annulus_inner_fwhm_mult`
    - `pcc.annulus_outer_fwhm_mult`
    - `pcc.min_aperture_px`

## Profiles

- `tile_compile.full_mode.example.yaml`
  - For datasets with enough usable frames for full mode.
  - Chroma denoise profile: balanced (reference values).
- `tile_compile.reduced_mode.example.yaml`
  - For 50..(frames_reduced_threshold-1) usable frames.
  - Chroma denoise profile: moderately conservative (detail-preserving).
- `tile_compile.emergency_mode.example.yaml`
  - Enables emergency reduced mode for datasets with <50 usable frames.
  - Chroma denoise profile: aggressive (strong chroma noise suppression).
- `tile_compile.smart_telescope_dwarf_seestar.example.yaml`
  - Suggested full config for DWARF / ZWO Seestar OSC stacks.
  - Chroma denoise profile: balanced (moderate chroma noise reduction, works with and without darks).
- `tile_compile.canon_low_n_high_quality.example.yaml`
  - Suggested OSC config for Canon-style datasets with low frame count but high/consistent quality.
  - Anti-grid focus for reduced/emergency operation: larger tiles, higher overlap, conservative weighting.
- `tile_compile.canon_equatorial_balanced.example.yaml`
  - Suggested OSC config for Canon/DSLR on equatorial mount (well-tracked, balanced quality/safety).
  - Registration is intentionally stricter than Alt/Az while still compatible with modeled fallback for failed direct registrations.
- `tile_compile.mono_full_mode.example.yaml`
  - Suggested full config for MONO datasets in full mode.
  - Chroma denoise block included for completeness, but disabled by default.
- `tile_compile.mono_small_n_anti_grid.example.yaml`
  - Suggested MONO config for small frame counts (e.g. 10..40) where tile seams/patterns can appear.
  - Anti-grid focus: larger tiles, higher overlap, conservative local weighting, reduced denoise aggressiveness.
- `tile_compile.mono_small_n_ultra_conservative.example.yaml`
  - Suggested MONO config for very small frame counts (e.g. 8..25) where maximum seam stability is preferred over aggressive enhancement.
  - Ultra-conservative focus: tile denoise disabled, very soft local weighting, larger tiles and higher overlap.

## Usage

1. Copy one example file and adapt paths/device-specific values:
   - `run_dir`
   - `input.pattern`
   - `data.image_width` / `data.image_height`
   - `data.bayer_pattern`
   - (optional) tune `dithering.min_shift_px` to your mount behavior
   - (optional) tune `registration.reject_*` thresholds if frames are over- or under-rejected
   - (optional, Alt/Az near polar region) prefer `tile_compile.smart_telescope_altaz_polar_near.example.yaml` as baseline
   - (optional, OSC) tune `chroma_denoise.blend.amount` and `chroma_denoise.apply_stage`
   - (optional) tune `stacking.cluster_quality_weighting.*` if cluster weighting is too strong/weak
   - (optional) tune `stacking.common_overlap_*` only if you intentionally want more edge coverage
   - (optional, v3.3) enable `bge.enabled: true` if gradients are visible (urban light pollution, moonlight)
   - (optional, v3.3) if mild red cast remains: reduce `bge.structure_thresh_percentile` (e.g. 0.90 -> 0.75) and use `bge.fit.robust_loss: tukey`
2. Run directly with:

```bash
./build/tile_compile_runner --config examples/tile_compile.full_mode.example.yaml
```

### Optional: merge workflow (overlay style)

If you still prefer overlay-style usage, merge your base config with a profile
using your YAML merge tool of choice, then run with the merged file.

## Recommended values by mount type (why)

### 1) Equatorial (Canon/DSLR, guided)

Use stricter rejection to prevent false matches:

```yaml
registration:
  star_topk: 120
  star_inlier_tol_px: 2.5
  reject_cc_min_abs: 0.35
  reject_shift_px_min: 40.0
  reject_shift_median_multiplier: 3.0
stacking:
  sigma_clip:
    sigma_low: 2.0
    sigma_high: 2.0
    min_fraction: 0.5
```

- **Why:** EQ sequences usually have small drift/rotation; stricter gates reduce misregistration risk.

### 2) Alt/Az (smart telescope, rotation-heavy)

Use tolerant rejection because large shift/rotation is physically expected:

```yaml
registration:
  star_topk: 150
  star_inlier_tol_px: 4.0
  reject_cc_min_abs: 0.30
  reject_shift_px_min: 100.0
  reject_shift_median_multiplier: 5.0
stacking:
  sigma_clip:
    sigma_low: 1.8
    sigma_high: 1.8
    min_fraction: 0.2
```

- **Why:** Alt/Az near pole can show wide shift distributions; too strict settings reject too many usable frames.
  - For no-dark OSC sessions with fixed hot pixels, prefer enabling
    `stacking.per_frame_cosmetic_correction` (pre-stack) instead of trying to
    force this via more aggressive sigma clipping.

### 3) Small-N MONO anti-grid

Prefer seam stability over aggressive enhancement:

```yaml
tile:
  min_size: 128
  overlap_fraction: 0.40
stacking:
  method: average
global_metrics:
  adaptive_weights: false
```

- **Why:** with low N, conservative blending and larger overlaps reduce tile pattern artifacts.
