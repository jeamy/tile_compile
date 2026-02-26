# Practical Configuration Examples for tile_compile

**[🇩🇪 Deutsche Version](configuration_examples_practical_de.md)**

This guide complements the configuration reference with practical examples, edge cases, and use cases based on methodology v3.3.

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
  autotune:
    enabled: false
    strategy: conservative
    max_evals: 24
    holdout_fraction: 0.25
    alpha_flatness: 0.25
    beta_roughness: 0.10
  sample_quantile: 0.20  # Conservative, resistant to faint objects
  fit:
    method: rbf  # Flexible, recommended
    rbf_phi: multiquadric  # Good compromise
    rbf_mu_factor: 1.0  # Standard smoothing
```

**For strong gradients (e.g. city outskirts):**

```yaml
bge:
  enabled: true
  sample_quantile: 0.15  # More conservative
  structure_thresh_percentile: 0.95  # Exclude more tiles
  fit:
    method: rbf
    rbf_phi: multiquadric
    rbf_mu_factor: 0.8  # Less smoothing for detail
```

**For weak gradients (e.g. moonlight):**

```yaml
bge:
  enabled: true
  sample_quantile: 0.25  # Less conservative
  fit:
    method: poly  # Simpler for weak gradients
    polynomial_order: 2
```

**Important:** BGE runs **before** PCC. When BGE is enabled, PCC should produce better results afterward.

**PCC v3.3.6 options (recommended with BGE):**

```yaml
pcc:
  background_model: plane      # median | plane
  radii_mode: auto_fwhm        # fixed | auto_fwhm
  aperture_fwhm_mult: 1.8
  annulus_inner_fwhm_mult: 3.0
  annulus_outer_fwhm_mult: 5.0
  min_aperture_px: 4.0
```

---

## Common overlap after PREWARP (`stacking.common_overlap_*`)

**New sensible defaults:**

```yaml
stacking:
  common_overlap_required_fraction: 1.0
  tile_common_valid_min_fraction: 0.90
```

- `common_overlap_required_fraction: 1.0` enforces strict intersection across all usable frames.
- `tile_common_valid_min_fraction: 0.90` prevents edge-heavy tiles from biasing local metrics.

**Recommendations by setup:**

- **Alt/Az with field rotation:** keep `1.0 / 0.90` (recommended)
- **EQ with very stable tracking:** `1.0 / 0.85-0.90`
- **Only when intentionally accepting more edge area:** `0.95 / 0.80-0.85`

**Important:** Lower values can reintroduce dynamic-range/background bias from uneven edge coverage.

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

## Tile Size (`tile.size`)

**Default:** `256`  
**Range:** `64` - `512`  
**Methodology requirement:** Must be large enough for local sharpness metrics, small enough for spatial resolution

### Use Cases:

**Short focal length (< 200mm), good seeing:**
```yaml
tile:
  size: 128
  overlap: 32
```
- Smaller tiles capture local quality differences better
- With good seeing, structures are more finely distributed
- Example: DWARF II (f=100mm), Seestar S50 (f=250mm)

**Medium focal length (200-800mm), normal seeing:**
```yaml
tile:
  size: 256  # Default
  overlap: 64
```
- Standard for most applications
- Good compromise between resolution and computation time
- Example: 80mm refractor, 8" SCT

**Long focal length (> 800mm), poor seeing:**
```yaml
tile:
  size: 384
  overlap: 96
```
- Larger tiles avoid tile artifacts with large structures
- With poor seeing, local quality differences are coarser
- Example: 12" SCT (f=2000mm), large refractors

**Alt/Az mount with field rotation:**
```yaml
tile:
  size: 320
  overlap: 80
  min_valid_fraction: 0.6  # More tolerant with rotation
```
- Larger tiles compensate rotation effects better
- Higher overlap for smoother transitions

---

## Registration (`registration.*`)

### `registration.method`

**Default:** `"triangle_star_matching"`  
**Alternatives:** `star_similarity`, `hybrid_phase_ecc`, `robust_phase_ecc`

**Star-rich fields (> 50 stars):**
```yaml
registration:
  method: triangle_star_matching
  min_stars: 15
  max_shift_px: 50
  max_rotation_deg: 5.0
```
- Triangle matching is robust and precise
- Works even with rotation and translation

**Star-poor fields (< 20 stars), nebulae:**
```yaml
registration:
  method: robust_phase_ecc
  fallback_to_identity: true
  identity_correlation_threshold: 0.3
```
- Phase correlation uses gradient structures
- Works even with diffuse nebulae
- Fallback prevents abort on difficult frames

**Alt/Az with field rotation (current):**
```yaml
registration:
  engine: triangle_star_matching
  allow_rotation: true  # REQUIRED for Alt/Az near pole
  star_topk: 150  # More stars for robust solution
  star_min_inliers: 4
  star_inlier_tol_px: 4.0  # More tolerant for drift/field rotation
  star_dist_bin_px: 5.0
  
  reject_outliers: true
  reject_cc_min_abs: 0.30
  reject_cc_mad_multiplier: 4.0
  reject_shift_px_min: 100.0
  reject_shift_median_multiplier: 5.0
  reject_scale_min: 0.92
  reject_scale_max: 1.08

  # Frames with failed direct registration are predicted using
  # a polynomial field-rotation model, so all frames remain usable.
```
**Note:** This behavior matches the current Alt/Az example profiles.

### `registration.max_shift_px`

**Default:** `50`  
**Range:** `10` - `200`

**Well-tracked (equatorial):**
```yaml
registration:
  max_shift_px: 30
```
- Low drift expected
- Stricter limits prevent misregistrations

**Alt/Az without field derotator:**
```yaml
registration:
  max_shift_px: 100
```
- Higher drift due to field rotation
- More tolerance needed

**Smart telescope (DWARF, Seestar) - short exposures:**
```yaml
registration:
  max_shift_px: 80
  max_rotation_deg: 8.0
```
- Moderate drift from tracking inaccuracies
- Rotation from Alt/Az mount

---

## Global Metrics (`global_metrics.*`)

### `global_metrics.fwhm_percentile`

**Default:** `0.5` (median)  
**Range:** `0.1` - `0.9`

**Good seeing (FWHM < 2.5"):**
```yaml
global_metrics:
  fwhm_percentile: 0.3  # Use best 30% of stars
  fwhm_outlier_sigma: 2.5
```
- With good seeing, best stars are very sharp
- Lower percentile focuses on peak values

**Poor seeing (FWHM > 4"):**
```yaml
global_metrics:
  fwhm_percentile: 0.7  # Use majority of stars
  fwhm_outlier_sigma: 3.5
```
- With poor seeing, large scatter
- Higher percentile avoids outlier dominance

**Turbulent seeing (highly variable):**
```yaml
global_metrics:
  fwhm_percentile: 0.5
  fwhm_outlier_sigma: 4.0  # Very tolerant
  use_robust_background: true
```

---

## Local Metrics (`local_metrics.*`)

### `local_metrics.sharpness_method`

**Default:** `"gradient_energy"`  
**Alternatives:** `laplacian_variance`, `tenengrad`

**High-resolution data (sampling < 1"/px):**
```yaml
local_metrics:
  sharpness_method: tenengrad
  sharpness_kernel_size: 5
```
- Tenengrad is more sensitive to fine details
- Smaller kernel for high resolution

**Low-resolution data (sampling > 3"/px):**
```yaml
local_metrics:
  sharpness_method: gradient_energy
  sharpness_kernel_size: 7
```
- Gradient energy more robust with coarse sampling
- Larger kernel for low resolution

**Smart telescopes (DWARF: 5.57"/px, Seestar: 3.97"/px):**
```yaml
local_metrics:
  sharpness_method: gradient_energy
  sharpness_kernel_size: 5
  contrast_percentile: 0.7
```

---

## Reconstruction (`reconstruction.*`)

### `reconstruction.ola_window`

**Default:** `"hann"`  
**Alternatives:** `bartlett`, `blackman`, `hamming`

**Many frames (N > 500), good SNR:**
```yaml
reconstruction:
  ola_window: hann
  ola_normalize_per_tile: true
```
- Hann window: good compromise
- Per-tile normalization safe with high SNR

**Few frames (50 < N < 200), low SNR:**
```yaml
reconstruction:
  ola_window: blackman  # Smoother transitions
  ola_normalize_per_tile: false
  sigma_clip_threshold: 4.0  # More tolerant
```
- Blackman reduces tile edges
- No tile normalization avoids noise amplification

**Emergency mode (N < 50):**
```yaml
reconstruction:
  ola_window: blackman
  ola_normalize_per_tile: false
  sigma_clip_threshold: 5.0
  min_frames_per_pixel: 3  # Very low
```

---

## Frame Count and Modes

**Methodology v3.2.2 requirements:**
- **Full mode:** N ≥ 200 (clustering + synthetic frames active)
- **Reduced mode:** 50 ≤ N < 200 (clustering disabled)
- **Emergency mode:** N < 50 (only with `runtime.allow_emergency_mode: true`)

### Full Mode (N ≥ 200)

```yaml
runtime:
  min_frames: 200
  allow_reduced_mode: false
  
synthetic:
  enabled: true
  min_cluster_size: 20
  max_clusters: 10
```

### Reduced Mode (50 ≤ N < 200)

```yaml
runtime:
  min_frames: 50
  allow_reduced_mode: true
  
synthetic:
  enabled: false  # Automatically disabled
```

### Emergency Mode (N < 50) - Testing only!

```yaml
runtime:
  min_frames: 10
  allow_emergency_mode: true  # WARNING!
  
tile:
  size: 384  # Larger tiles
  min_valid_fraction: 0.4  # Very tolerant
  
reconstruction:
  sigma_clip_threshold: 5.0
  min_frames_per_pixel: 2
```

**⚠️ Warning:** Emergency mode is not suitable for production!

---

## Focal Length Specific Recommendations

### Short Focal Length (< 200mm)

**Example: DWARF II (100mm f/4.4), Seestar S50 (250mm f/5)**

```yaml
tile:
  size: 128
  overlap: 32
  
registration:
  method: triangle_star_matching
  min_stars: 20  # Many stars in field
  max_shift_px: 60
  
local_metrics:
  sharpness_kernel_size: 5
  contrast_percentile: 0.7
```

### Medium Focal Length (200-800mm)

**Example: 80mm refractor (480mm f/6), 8" SCT (2000mm f/10)**

```yaml
tile:
  size: 256
  overlap: 64
  
registration:
  method: triangle_star_matching
  min_stars: 10
  max_shift_px: 40
  
local_metrics:
  sharpness_kernel_size: 5
  contrast_percentile: 0.5
```

### Long Focal Length (> 800mm)

**Example: 12" SCT (3000mm f/10), large refractors**

```yaml
tile:
  size: 384
  overlap: 96
  
registration:
  method: triangle_star_matching
  min_stars: 5  # Fewer stars in field
  max_shift_px: 30  # Precise guiding expected
  max_rotation_deg: 2.0
  
local_metrics:
  sharpness_kernel_size: 7
  contrast_percentile: 0.3
```

---

## Seeing Conditions

### Excellent Seeing (FWHM < 2")

```yaml
global_metrics:
  fwhm_percentile: 0.2
  fwhm_outlier_sigma: 2.0
  
local_metrics:
  sharpness_percentile: 0.3
  
reconstruction:
  quality_weight_exponent: 2.0  # Stronger weighting
```

### Good Seeing (FWHM 2-3")

```yaml
global_metrics:
  fwhm_percentile: 0.4
  fwhm_outlier_sigma: 2.5
  
local_metrics:
  sharpness_percentile: 0.5
  
reconstruction:
  quality_weight_exponent: 1.5
```

### Moderate Seeing (FWHM 3-4")

```yaml
global_metrics:
  fwhm_percentile: 0.5
  fwhm_outlier_sigma: 3.0
  
local_metrics:
  sharpness_percentile: 0.6
  
reconstruction:
  quality_weight_exponent: 1.0  # Default
```

### Poor Seeing (FWHM > 4")

```yaml
global_metrics:
  fwhm_percentile: 0.7
  fwhm_outlier_sigma: 3.5
  use_robust_background: true
  
local_metrics:
  sharpness_percentile: 0.7
  
reconstruction:
  quality_weight_exponent: 0.8  # Weaker weighting
  sigma_clip_threshold: 4.0
```

---

## Mount-Specific Settings

### Equatorial Mount (well-tracked)

```yaml
registration:
  method: triangle_star_matching
  max_shift_px: 30
  max_rotation_deg: 2.0
  allow_reflection: false
  
tile:
  min_valid_fraction: 0.8  # Strict
```

### Alt/Az without derotator

```yaml
registration:
  method: triangle_star_matching
  max_shift_px: 100
  max_rotation_deg: 15.0
  trail_endpoint_enabled: true
  
tile:
  size: 320  # Larger due to rotation
  overlap: 80
  min_valid_fraction: 0.6  # More tolerant
```

### Alt/Az with derotator (DWARF, Seestar)

```yaml
registration:
  method: triangle_star_matching
  max_shift_px: 60
  max_rotation_deg: 8.0
  
tile:
  size: 256
  overlap: 64
  min_valid_fraction: 0.7
```

---

## Camera-Specific Settings

### OSC (One-Shot Color)

```yaml
data:
  mode: OSC
  bayer_pattern: RGGB  # Camera-dependent!
  
debayer:
  enabled: true
  method: bilinear
  preserve_bayer_normalization: true
  
pcc:
  enabled: true
  source: auto
  method: proportion
```

### Monochrome

```yaml
data:
  mode: MONO
  
debayer:
  enabled: false
  
# No PCC for mono (only for RGB composite)
```

---

## Performance Optimization

### Fast test run

```yaml
pipeline:
  mode: test
  max_frames: 50
  
tile:
  size: 256
  
output:
  write_registered_frames: false
  write_tile_weights: false
```

### Production (maximum quality)

```yaml
pipeline:
  mode: production
  
tile:
  size: 256
  overlap: 64
  
reconstruction:
  ola_normalize_per_tile: true
  
output:
  write_registered_frames: true
  write_tile_weights: true
  write_quality_maps: true
```

### Memory-limited

```yaml
runtime:
  max_memory_gb: 8.0
  use_disk_cache: true
  
tile:
  size: 192  # Smaller = less RAM
  
output:
  write_registered_frames: false
```

---

## Summary: Typical Setups

### DWARF II / Seestar S50

```yaml
data:
  mode: OSC
  bayer_pattern: RGGB
  
tile:
  size: 128
  overlap: 32
  
registration:
  method: triangle_star_matching
  max_shift_px: 80
  max_rotation_deg: 8.0
  
global_metrics:
  fwhm_percentile: 0.5
  
local_metrics:
  sharpness_method: gradient_energy
  sharpness_kernel_size: 5
  
reconstruction:
  ola_window: hann
  quality_weight_exponent: 1.0
  
debayer:
  enabled: true
  method: bilinear
  
pcc:
  enabled: true
  source: auto
```

### DSLR on Equatorial Mount

```yaml
data:
  mode: OSC
  bayer_pattern: RGGB  # Canon usually RGGB, Nikon usually GBRG
  
tile:
  size: 256
  overlap: 64
  
registration:
  method: triangle_star_matching
  max_shift_px: 40
  max_rotation_deg: 3.0
  
global_metrics:
  fwhm_percentile: 0.4
  
reconstruction:
  quality_weight_exponent: 1.5
  
debayer:
  enabled: true
  method: bilinear
  
pcc:
  enabled: true
```

Ready-to-use profile:
- `tile_compile_cpp/examples/tile_compile.canon_equatorial_balanced.example.yaml`

### Mono CCD on Large Telescope

```yaml
data:
  mode: MONO
  
tile:
  size: 384
  overlap: 96
  
registration:
  method: triangle_star_matching
  min_stars: 5
  max_shift_px: 20
  max_rotation_deg: 1.0
  
global_metrics:
  fwhm_percentile: 0.3
  
local_metrics:
  sharpness_kernel_size: 7
  
reconstruction:
  quality_weight_exponent: 2.0
```

---

These examples are based on:
- Methodology v3.2.2 requirements (linearity, no frame selection, tile-based reconstruction)
- Practical experience with various setups
- Physical constraints (seeing, focal length, mount type)

Adjust values to your specific hardware and conditions!
