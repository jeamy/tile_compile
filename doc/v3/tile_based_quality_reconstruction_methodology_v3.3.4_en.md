# Tile-Based Quality Reconstruction for DSO - Methodology v3.3

**Status:** Normative reference specification  
**Version:** v3.3 (2026-02-25)  
**Applies to:** `tile_compile.yaml`

---

## 0. Objective of v3.3

Core objectives:

1. mathematical consistency (notation, formulas, edge cases)
2. clear separation of **mandatory core** vs. **optional extensions**
3. precise semantics for
   - linearity,
   - no frame selection,
   - robust pixel outlier handling

---

## 1. Principles and Definitions

### 1.1 Physical Objective

From fully registered, linear short-exposure frames, a spatially and temporally optimally weighted signal is reconstructed.

The method models two orthogonal quality axes:

- **global** (atmosphere): transparency, sky brightness, noise
- **local** (tile): sharpness, structural support, local background level

### 1.2 No Frame Selection (Invariant)

**Forbidden:** Removal of entire frames based on quality.  
**Permitted:** Pixel-wise outlier rejection (sigma clipping), provided that

- it acts only at pixel level,
- it uses deterministic parameters,
- and it includes a documented fallback to the unchanged mean.

### 1.3 Linearity Semantics (Clarified)

"Strictly linear" in v3.3 means:

1. **Photometric signal mapping** remains linear (no global nonlinear tone curves such as stretch, asinh, log).
2. Linear reconstruction steps (scaling, weighted mean, overlap-add) are mandatory.
3. Robust/statistical nonlinearities (MAD, clipping, sigma clipping, adaptive gating decisions) are allowed as **auxiliary steps**.

---

## 2. Assumptions and Operating Modes

### 2.1 Hard Assumptions (Violation -> Abort)

- Input data are linear (no stretch, no tone curves)
- Uniform exposure time (tolerance +-5%)
- Per-channel processing after channel separation
- No quality-based frame selection
- Registered geometry is expressed in the same pixel reference

### 2.2 Soft Assumptions

| Assumption | Optimal | Minimum | Action if violated |
|---|---:|---:|---|
| Number of frames N | >= 800 | >= 50 | Reduced mode for 50..199 |
| Registration residual | < 0.3 px | < 1.0 px | Warning at > 0.5 px |
| Star elongation | < 0.2 | < 0.4 | Warning at > 0.3 |

### 2.3 Reduced Mode (Unambiguous)

- **Valid only for:** `50 <= N <= 199`
- Steps 8-9 (clustering + synthetic frames) are skipped
- Final output is the reconstruction from phase 7

### 2.4 Below Minimum

- **N < 50:** no reduced mode
- Standard action: controlled abort with diagnostics
- Optional only via explicit `runtime.allow_emergency_mode: true`: emergency mode with warning status

---

## 3. Pipeline Overview (Normative)

1. Registration and geometric harmonization
2. Channel separation
3. Global linear normalization
4. Global frame metrics and global weights
5. Tile geometry
6. Local tile metrics and local weights
7. Tile reconstruction (overlap-add)
8. State-based clustering (full mode only)
9. Synthetic frames (full mode only)
10. Final linear stacking
11. Post-processing (optional, not part of the quality core)

Mandatory core: 1-10.  
Optional/feature-gated: local denoisers, sigma-clipping variants, WCS/PCC.

---

## 4. Registration and Channel Separation up to Phase 2 (Normative)

Up to and including phase 2, the CFA-based registration and channel-separation path applies.
From phase 3 onward, the shared core applies.

### 4.1 CFA-Based Registration Path

- Registration on a CFA luminance proxy
- CFA-aware warp by subplanes (`warp_cfa_mosaic_via_subplanes`)
- Channel separation afterwards

### 4.2 Registration Cascade

Per frame:

1. configurable primary method (`triangle_star_matching` default)
2. fixed fallback order:
   - `trail_endpoint_registration`
   - `feature_registration_similarity` (AKAZE)
   - `robust_phase_ecc`
   - `hybrid_phase_ecc`
   - identity fallback with warning

Acceptance criterion per attempt:

- `NCC(warped, ref) > NCC(identity, ref) + delta_ncc`
- Default `delta_ncc = 0.01`

---

## 5. Shared Core from Phase 3 Onward

## 5.1 Notation (Binding)

- `f` frame index, `t` tile index, `c` channel index, `p` pixel
- `I_{f,c}(p)` normalized input image per frame/channel
- `B_{f,c}` global background (before normalization)
- `sigma_{f,c}` global noise (after normalization)
- `E_{f,c}` global gradient energy (after normalization)
- `Q_{f,c}` global quality index
- `G_{f,c}` global weight
- `Q_{f,t,c}^{local}` local quality index
- `L_{f,t,c}` local weight
- `W_{f,t,c}` effective weight

**From this point onward, channel index `c` is used consistently.**

---

## 5.2 Global Linear Normalization (Mandatory)

Order:

1. Background from raw data:
   - `B_{f,c} = median(I_{f,c}^{raw})`
2. Linear scaling:
   - `I_{f,c} = I_{f,c}^{raw} / max(B_{f,c}, eps_bg)`
3. Metrics on normalized data:
   - `sigma_{f,c}`, `E_{f,c}`

Forbidden: global nonlinear tone curves.

Recommended default:

- `eps_bg = 1e-6`

---

## 5.3 Global Metrics and Weights

### 5.3.1 Robust Metric Normalization

For metric sequence `x`:

`z(x_i) = (x_i - median(x)) / max(1.4826 * MAD(x), eps_mad)`

with `eps_mad = 1e-6`.

### 5.3.2 Global Quality Index

`Q_{f,c} = alpha*(-z(B_{f,c})) + beta*(-z(sigma_{f,c})) + gamma*z(E_{f,c})`

Constraint: `alpha + beta + gamma = 1`

Defaults:

- `alpha=0.4, beta=0.3, gamma=0.3`

Clamping before exponential:

`Q_{f,c}^{clamped} = clip(Q_{f,c}, -3, +3)`

Global weight:

`G_{f,c} = exp(k_global * Q_{f,c}^{clamped})`

with `k_global > 0`, default `k_global=1.0`.

### 5.3.3 Optional Adaptive Weighting

If `global_metrics.adaptive_weights=true`:

- Variances are computed on robustly normalized metrics:
  - `Var(z(B))`, `Var(z(sigma))`, `Var(z(E))`
- Raw weights:
  - `alpha' ~ Var(z(B))`, `beta' ~ Var(z(sigma))`, `gamma' ~ Var(z(E))`
- Clip each weight to [0.1, 0.7], then renormalize to sum 1
- Fallback to static defaults for degenerate total variance

---

## 5.4 Tile Geometry

Parameters:

- Image size `W,H`
- Robust seeing estimate `F` (FWHM in pixels)
- `s = tile.size_factor`
- `T_min = tile.min_size`
- `D = tile.max_divisor`
- `o = tile.overlap_fraction`, `0 <= o <= 0.5`

Formulas:

Formulas:

`T0 = s * F`

**Overlap enforcement (binding):**  
`o_clipped = clip(o, 0, 0.5)`

`T = floor(clip(T0, T_min, floor(min(W,H)/D)))`

`O = floor(o_clipped * T)`

`S = T - O`


Guards (binding):

1. if `F <= 0` -> `F = 3.0`
2. `T_min >= 16`
3. if `S <= 0` -> set `o_clipped=0.25`, recompute `O,S` (and keep `o_clipped` within [0,0.5])
4. if `min(W,H) < T` -> `T=min(W,H)`, `O=0`

---

## 5.5 Local Tile Metrics

### 5.5.1 Classification

- **STAR tile:** `star_count >= tile.star_min_count`
- **STRUCTURE tile:** otherwise

### 5.5.2 STAR Tile Metrics

- `FWHM_{f,t,c}`
- `R_{f,t,c}` (roundness)
- `C_{f,t,c}` (contrast)

Local index:

`Q_{f,t,c}^{star} = 0.6*(-z(FWHM)) + 0.2*z(R) + 0.2*z(C)`

### 5.5.3 STRUCTURE Tile Metrics

- `(E/sigma)_{f,t,c}`
- `B_{f,t,c}`

Local index:

`Q_{f,t,c}^{struct} = 0.7*z(E/sigma) - 0.3*z(B)`

### 5.5.4 Local Weight

`Q_{f,t,c}^{local} = clip(Q_{f,t,c}^{star|struct}, -3, +3)`

`L_{f,t,c} = exp(Q_{f,t,c}^{local})`

---

## 5.6 Effective Weight

`W_{f,t,c} = G_{f,c} * L_{f,t,c}`

Semantics:

- `G`: global atmospheric quality
- `L`: local structure/sharpness quality

---

## 5.7 Tile Reconstruction (Consolidated)

For pixel `p` in tile `t`:

`D_{t,c} = sum_f W_{f,t,c}`

If `D_{t,c} >= eps_weight`:

`R_{t,c}(p) = sum_f W_{f,t,c} * I_{f,c}(p) / D_{t,c}`

If `D_{t,c} < eps_weight`:

`R_{t,c}(p) = (1/N) * sum_f I_{f,c}(p)`

and `fallback_used=true` for this tile.

Default `eps_weight = 1e-6`.

### 5.7.1 Tile Normalization before OLA (Binding)

For reconstructed tile `R_{t,c}`:

1. `bg_t = median(R_{t,c})`
2. `X_t = R_{t,c} - bg_t`
3. `m_t = median(abs(X_t))`
4. if `m_t >= eps_median`: `Y_t = X_t / m_t`, otherwise `Y_t = X_t`

Default `eps_median = 1e-6`.
#### 5.7.1a Photometric Preservation after OLA (Recommended)

The normalization `Y_t = (R_{t,c} - bg_t)/m_t` equalizes local structure but can alter absolute photometric scale if left uncorrected.
To preserve a consistent global affine flux scale, accumulate per-tile metadata during reconstruction and restore a global scale/offset after OLA:

- Per tile (already computed): `bg_t`, `m_t`
- Global restoration factors (robust):
  - `m_global = median_t(m_t)`
  - `bg_global = median_t(bg_t)`

After overlap-add produces `I_rec`, restore:

`I_final = I_rec * m_global + bg_global`

This keeps the core reconstruction linear in pixel values (the restoration is a global affine transform) while preventing systematic tile-to-tile photometric drift.

### 5.7.2 Windowing and Overlap-Add

2D window separable with discrete Hann function:

`hann(i,N) = 0.5*(1 - cos(2*pi*i/(N-1)))`, `i=0..N-1`

Special case: `N=1 -> hann=1`.

`w(x,y) = hann(x,W_t) * hann(y,H_t)`

Reconstruction image:

- numerator accumulator: `A`
- window-sum accumulator: `S`

`A += w * Y_t`, `S += w`, result `I_rec = A / max(S, eps_weight)`

Optionally, after OLA a global robust tile-background offset may be restored (median over `bg_t`).

---

## 5.8 Optional Local Denoisers (Explicitly Optional)

These steps are **not part of the mandatory mathematical core**, but are admissible extensions.

### 5.8.1 Soft-Threshold High-Pass

- Background via box blur
- Residual
- `tau = alpha_d * sigma_tile`
- Soft shrinkage
- Reconstruction

### 5.8.2 Wiener in the Frequency Domain

- Reflection padding
- FFT
- Wiener transfer function
- IFFT and crop

Apply only if gating conditions are met (SNR/quality/tile type).

---

## 5.9 State-Based Clustering (Full Mode)

Active only for `N >= 200`.

State vector per frame/channel (per-channel or channel-aggregated, configurable):

`v_f = (G_{f,*}, mean_t(Q_{f,t,*}^{local}), var_t(Q_{f,t,*}^{local}), B_{f,*}, sigma_{f,*})`

Number of clusters:

`K = clip(floor(N/10), K_min, K_max)`

Defaults: `K_min=5`, `K_max=30`.

---

## 5.10 Synthetic Frames

### 5.10.1 Default (global)

For cluster `k`:

`S_{k,c} = sum_{f in k} G_{f,c} * I_{f,c} / sum_{f in k} G_{f,c}`

### 5.10.2 Optional (tile_weighted)

If `synthetic.weighting=tile_weighted`:

- reconstruct per tile/channel with `W_{f,t,c}`
- assemble to `S_{k,c}` via OLA

### 5.10.3 Semantics of Phase 7 vs 9

- Full mode with `global`: phase 7 primarily provides local quality modeling/diagnostics; the final product is generated from phases 9+10.
- Full mode with `tile_weighted`: local tile quality is explicitly propagated into synthetic frames.
- Reduced mode: the output from phase 7 is the direct final product.

---

## 5.11 Final Linear Stacking

### 5.11.1 Cluster Quality Definition (Binding)

For each cluster `k`, define a robust cluster-level quality index:

`Q_k = median_{f in k}(Q_{f,c}^{clamped})`

where `Q_{f,c}^{clamped}` is the global frame quality index already limited to `[-3,+3]`.

### 5.11.2 Quality-Weighted Cluster Aggregation (Binding)

Clusters are aggregated using exponential quality weighting:

`w_k = exp(kappa_cluster * Q_k)`

with:

- `kappa_cluster > 0` (recommended default: `kappa_cluster = 1.0`)
- `Q_k` already clamped to `[-3,+3]`

Optional stability cap (recommended):

`w_k = min(w_k, r_cap * median_j(w_j))`

with recommended `r_cap` in `[10, 50]`.

Final result per channel:

`R_c = sum_k (w_k * S_{k,c}) / sum_k w_k`

### 5.11.3 Semantics

- Better atmospheric states (higher `Q_k`) receive stronger influence.
- All clusters remain included (no hard state selection).
- The estimator remains linear in synthetic frames.
- Dominance is bounded via optional weight capping.


## 6. Post-Processing (Not Part of the Mandatory Core)

### 6.1 RGB/LRGB Combination

Interchangeable, outside the reconstruction core.

### 6.2 Astrometry (WCS)

Permissible downstream step, without feedback into core weights.


### 6.3 Pre-PCC Background Gradient Extraction (BGE) (Optional, Recommended)

Background gradients (e.g. artificial light pollution, moonlight, airglow) can bias Photometric Color Calibration (PCC), especially when gradients are spectrally non-uniform across channels.  
To mitigate this, an additive Background Gradient Extraction (BGE) step may be applied **before PCC**.

#### 6.3.1 Principle

For each channel `c`, estimate a smooth large-scale background model `B_c(x,y)` and subtract:

`I'_c(x,y) = I_c(x,y) - B_c(x,y)`

BGE must be:
- strictly additive,
- channel-wise,
- independent of frame weighting logic,
- and must not introduce nonlinear tone transforms.

#### 6.3.2 Tile-Driven Sampling Grid (Binding)

The reconstruction tiles are reused as background sampling units. The goal is to obtain **object-free** background samples per tile.

##### (a) Background Mask Definition (Binding)

For each tile `t` and channel `c`, define a binary mask `M_bg` that marks pixels admissible as background samples. `M_bg` must exclude:

1. **Stars:** pixels in `M_star` (from star detection or segmentation), optionally dilated by `mask.star_dilate_px` (recommended default: 2–6 px).
2. **High-structure pixels:** pixels where `structure_metric(p) > structure_thresh`, where `structure_metric` is derived from local gradients (e.g. high-pass energy) and `structure_thresh` is configurable.
3. **Saturated pixels:** pixels with `I >= sat_level` and optionally a dilation margin `mask.sat_dilate_px`.
4. **Optional object mask:** if available (nebula/galaxy mask), exclude it to prevent bias in extended-object fields.

If no star detection is available, `M_star` may be approximated by thresholding a bandpass/DoG response and dilating; this approximation must be deterministic.

##### (b) Robust Tile Background Sample (Binding, Configurable)

Compute one robust background sample per tile using a configurable quantile:

`b_{t,c} = quantile_q(R_{t,c}[M_bg])`

with:
- `q = bge.sample_quantile` in `(0, 0.5]`
- **default:** `q = 0.20` (20% quantile)
- median is obtained by setting `q = 0.50`

Rationale: the lower quantile is more resistant to residual faint object contamination and imperfect masks, while the median is acceptable in sparse fields with strong masking.

Associate each sample with the tile center `(x_t, y_t)`.

##### (c) Tile Reliability Weight (Optional, Recommended)

Tiles may be assigned a reliability weight for later fitting:

`w_t = exp(-lambda * structure_score_t) * (1 - masked_fraction_t)`

where `structure_score_t` is computed from `E/sigma` or similar local structure metrics, and `masked_fraction_t` is the excluded pixel fraction in the tile.


#### 6.3.3 Coarse Grid Aggregation (Binding)

To avoid overfitting small-scale structure, tile samples are aggregated to a **coarser** grid before surface fitting.

##### (a) Grid Definition

Given grid spacing `G` (see 6.3.8), define axis-aligned grid cells over the image plane. Each grid cell is a rectangle of size `G x G`.

##### (b) Assigning Tiles to Grid Cells (Binding)

Each tile sample `(x_t, y_t, b_{t,c}, w_t)` is assigned to exactly one grid cell via integer binning of its center:

`cell_x = floor(x_t / G)`  
`cell_y = floor(y_t / G)`

(All tiles whose centers fall inside the same `G x G` cell belong to that cell.)

##### (c) Per-Cell Aggregation (Binding)

For each cell and channel `c`, aggregate all tile samples assigned to the cell:

- Value: `b_cell = median({b_{t,c}})` (robust)
- Weight: `w_cell = median({w_t})` (or sum, implementation choice; must be documented)

##### (d) Insufficient Samples (Binding)

A grid cell is considered **insufficient** if it contains fewer than:

`n_cell < bge.min_tiles_per_cell`

Recommended default: `bge.min_tiles_per_cell = 3`.

Insufficient cells must be handled deterministically by one of:

1. **Discard (default):** exclude the cell from the fit, or
2. **Nearest-neighbor fill:** replace `(b_cell, w_cell)` by the nearest sufficient cell (by Euclidean distance between cell centers), or
3. **Radius expansion:** iteratively include tiles from neighboring cells within radius `r = k*G` until `n_cell >= min_tiles_per_cell` (deterministic traversal order required).

The chosen strategy must be configurable and recorded in diagnostics.


#### 6.3.4 Surface Fitting

Fit a smooth background surface per channel using:

- Robust 2D polynomial (order 2–3 recommended), or
- Thin-plate spline, or
- Bicubic spline with robust loss, or
- Radial Basis Function (RBF) surface with smoothing (recommended only with explicit regularization), or
- Foreground-aware modeled-mask mesh sky surface (`modeled_mask_mesh`) for scenes with large diffuse foreground structures.

Optional weights:

`w_t = exp(-lambda * structure_score_t)`

Use robust loss (Huber/Tukey).

#### 6.3.5 Subtraction

`I'_c(x,y) = I_c(x,y) - B_c(x,y)`

No multiplicative correction permitted.

#### 6.3.6 Validation Requirements

When BGE is enabled:

1. Background RMS must decrease or remain stable.
2. No artificial curvature across tile boundaries.
3. Stellar flux ratios must remain stable within tolerance.
4. PCC residuals must improve or remain stable vs. no-BGE baseline.

BGE must not modify core weights (`G`, `L`, `W`).

### 6.3.7 Mathematical Surface Model (Binding)

Let the background samples be defined as:

`(x_i, y_i, b_i, w_i)`  for i = 1..M

where:
- `(x_i, y_i)` are grid cell centers,
- `b_i` is the robust background estimate,
- `w_i` optional reliability weight.

A robust polynomial surface of order d (recommended d = 2 or 3) is defined as:

`B_c(x,y) = sum_{m+n <= d} a_{mn} x^m y^n`

The coefficients `a_{mn}` are obtained by minimizing:

`argmin_a sum_i w_i * rho( b_i - B_c(x_i,y_i) )`

where `rho` is a robust loss function, e.g.:

Huber loss:

`rho(r) = 0.5 r^2           if |r| <= delta`
`rho(r) = delta(|r| - 0.5 delta)  otherwise`

or Tukey biweight loss.

The fit must be solved via Iteratively Reweighted Least Squares (IRLS) or equivalent deterministic robust optimization.

Thin-plate spline alternative:

`B_c = argmin_B sum_i w_i (b_i - B(x_i,y_i))^2 + lambda * integral |D^2 B|^2 dx dy`

with regularization parameter `lambda` controlling smoothness.

RBF surface alternative (binding when method=rbf):

For Radial Basis Function interpolation:

`B_c(x,y) = sum_i u_i * phi(||r_i - r||; mu)`

where:
- `r_i = (x_i, y_i)` are grid cell centers
- `r = (x,y)` is the evaluation point
- `u_i` are the RBF coefficients (to be solved)
- `phi(d; mu)` is the RBF kernel
- `mu > 0` is the shape/scale parameter

Supported kernels (binding):

1. **Multiquadric:** `phi(d; mu) = sqrt(d^2 + mu^2)`
2. **Thin-plate:** `phi(d) = d^2 * log(d)` for `d > 0`, with numerical stabilization `phi(d; epsilon) = d^2 * log(d + epsilon)` where `epsilon` is a small constant (e.g. `1e-10`) to avoid singularity at `d=0`
3. **Gaussian:** `phi(d; mu) = exp(-d^2 / (2*mu^2))`

Coefficients `u_i` obtained via weighted regularized robust regression:

`argmin_u sum_j w_j * rho(b_j - sum_i u_i * phi(||r_i - r_j||; mu)) + lambda * ||u||^2`

where:
- `w_j` are the optional reliability weights from tile sampling (see 6.3.2c)
- `rho` = Huber or Tukey loss (as specified above)
- `lambda >= 0` is the regularization parameter (prevents overfitting, controls smoothness)
- `mu > 0` is the shape/scale parameter (controls kernel width)

Recommended defaults:
- `mu = G` (grid spacing) for Multiquadric and Gaussian
- `epsilon = 1e-10` for Thin-plate numerical stabilization
- `lambda = 1e-6` (weak regularization)

Parameter semantics:
- **Gaussian:** `mu` is the bandwidth (standard deviation sigma)
- **Multiquadric:** `mu` controls the transition from linear to quadratic behavior
- **Thin-plate:** no shape parameter (scale-invariant); `epsilon` is only for numerical stability

Only large-scale (low-frequency) components are permitted; overfitting is forbidden.

#### 6.3.8 Adaptive Grid Definition (Binding)

Grid spacing `G` must scale with image dimensions to avoid under- or overfitting.

Define:

`G = clip( max(2*T, min(W,H)/N_g), G_min, G_max )`

Recommended defaults:

- `N_g = 32` (target grid resolution across smallest image axis)
- `G_min = 64 px`
- `G_max = min(W,H)/4`

This ensures:

- background model captures only large-scale gradients,
- grid density adapts to sensor resolution,
- small images are not over-parameterized,
- large mosaics retain sufficient spatial sampling.

Implementations must guarantee that grid resolution is coarser than tile resolution (`G >= 2*T`).


### 6.4 PCC

Permissible downstream step, applied to linear data.

---

## 7. Validation and Abort

## 7.1 Success Criteria

- FWHM improvement over the reference stack according to `validation.min_fwhm_improvement_percent`
- Background RMS not worse than reference
- No systematic tile seams
- Stable weight distributions

## 7.2 Abort Criteria

- Data integrity violated (nonlinear, unreadable, inconsistent)
- Registration failure across large portions of the dataset
- Numerical instability despite fallbacks

## 7.3 Minimum Tests (Normative)

1. `alpha+beta+gamma=1`
2. clamping before `exp`
3. tile monotonicity in `F`
4. overlap consistency (`0<=o<=0.5`, explicit `o_clipped=clip(o,0,0.5)`, integer O,S)
5. low-weight fallback without NaN/Inf
6. no channel coupling
7. no quality-based frame selection
8. deterministic reproducibility
9. registration cascade including identity fallback
10. CFA phase preservation
11. cluster aggregation quality-weighted (exp(kappa_cluster * Q_k)) with optional dominance cap
12. WCS round-trip error below threshold
13. PCC stability: positive determinant, bounded condition number, residuals below threshold

Note: The legacy PCC test "no negative matrix element" is **no longer** required as a hard criterion in v3.3.

---

## 8. Recommended Numerical Defaults

- `eps_bg = 1e-6`
- `eps_mad = 1e-6`
- `eps_weight = 1e-6`
- `eps_median = 1e-6`
- `delta_ncc = 0.01`
- `Q` clamp global/local: `[-3, +3]`

---

## 9. Scope Boundary: Mandatory Core vs Extension

### Mandatory Core

- CFA-based registration path up to channel separation
- global normalization
- global/local metrics and weights
- tile reconstruction including consolidated fallbacks
- clustering/synthesis/final stack depending on operating mode

### Optional Extension

- soft-threshold / Wiener
- alternative sigma-clipping strategies
- WCS/PCC
- specialized performance backends (GPU, queue workers)

### 9.1 Practical Configuration Profiles (tile_compile_cpp)

For operational use, complete reference configurations are provided:

- `tile_compile_cpp/examples/tile_compile.full_mode.example.yaml`
- `tile_compile_cpp/examples/tile_compile.reduced_mode.example.yaml`
- `tile_compile_cpp/examples/tile_compile.emergency_mode.example.yaml`
- `tile_compile_cpp/examples/tile_compile.smart_telescope_dwarf_seestar.example.yaml`

All profiles include **all available configuration options** with inline comments.
Procedure:

1. copy the appropriate profile,
2. adapt `run_dir`, `input.pattern`, and sensor parameters (`image_width/height`, `bayer_pattern`),
3. launch the runner with this file.

---

## 9.2 Strict ML Optimization Extension (Optional, Non-Core)

This extension introduces machine-learning (ML) modules **only** to improve the estimation of weights and state descriptors while preserving the mandatory core invariants:

### 9.2.1 Binding Invariants (Hard Constraints)

1. **No frame selection:** Entire frames must not be removed based on quality (unchanged from v3.3.x).
2. **Strict photometric linearity of the reconstruction core:** The final reconstruction must remain a weighted linear estimator over input frames (and/or synthetic frames), i.e. of the form

   `R(p) = sum_i w_i(p) * X_i(p) / sum_i w_i(p)`

   with `w_i(p) >= 0` and deterministic fallbacks.
3. **Determinism:** ML inference must be deterministic (fixed model weights, fixed preprocessing, fixed seeds where applicable).
4. **No hallucinated content:** ML modules must not generate new spatial structures. ML outputs are restricted to **weights, masks, metrics, and state labels**.
5. **Channel separation preserved:** ML modules must operate per-channel or on explicitly defined channel-aggregated features; no implicit cross-channel coupling in the core estimator.

### 9.2.2 Allowed ML Outputs (Strict)

ML may output any of the following, provided outputs are deterministic and bounded:

- Global quality score per frame/channel: `Q̂_{f,c}` (dimensionless, mapped/clamped to `[-3,+3]`)
- Global weight per frame/channel: `Ĝ_{f,c} = exp(k_global * Q̂_{f,c})`
- Local tile quality score: `q̂_{f,t,c}` (dimensionless, clamped to `[-3,+3]`)
- Local tile weight: `L̂_{f,t,c} = exp(q̂_{f,t,c})`
- Pixel reliability mask (soft, not hard rejection): `M̂_{f,t,c}(p) in [m_min, 1]` with recommended `m_min = 0.05`
- State descriptor for clustering (per frame): `v̂_f` (feature vector)
- State labels (clusters) and/or transition probabilities (HMM), used only to form synthetic frames

Forbidden ML outputs:

- Direct prediction of reconstructed pixel intensities (end-to-end image generation)
- Super-resolution or inpainting that creates spatial detail not supported by the input
- Any stochastic sampling at inference time

### 9.2.3 ML-Driven Effective Weight (Binding)

If ML modules are enabled, the effective weight may be extended to pixel level:

`Ŵ_{f,t,c}(p) = Ĝ_{f,c} * L̂_{f,t,c} * M̂_{f,t,c}(p)`

The tile reconstruction remains a weighted mean:

`R_{t,c}(p) = sum_f Ŵ_{f,t,c}(p) * I_{f,c}(p) / sum_f Ŵ_{f,t,c}(p)`

Fallback rule remains unchanged: if denominator < `eps_weight`, fall back to the unweighted mean.

### 9.2.4 Recommended Learning Paradigms (Non-Binding Guidance)

Because ground truth is typically unavailable, prioritize:

- **Self-supervised learning:** consistency across random frame subsets, Noise2Self/Noise2Void style objectives for masks/denoising proxies (note: denoising must still output masks/weights, not pixels in strict mode).
- **Weak supervision via proxies:** optimize weights to improve deterministic metrics (FWHM, ellipticity, background RMS, seam score) on validation sets.
- **Uncertainty-aware models:** output confidence to avoid overconfident downweighting; uncertainty must be mapped into bounded masks/weights.

### 9.2.5 Models That Fit the Strict Output Constraint (Non-Binding)

- Global weights: gradient-boosted trees (GBM), small MLPs on frame metrics
- Tile quality: small CNN encoders / lightweight ViT-tiny (only if data volume sufficient)
- Pixel reliability masks: U-Net-lite producing `M̂(p)` in `[m_min,1]`

LLMs are admissible only for **configuration synthesis, validation report interpretation, and test generation**, not for pixel-level reconstruction.

### 9.2.6 Validation Requirements for ML Extension (Binding)

When ML is enabled, all mandatory core validation tests still apply, plus:

1. **Bounded outputs:** enforce `Q̂ in [-3,+3]`, `M̂ in [m_min,1]`
2. **Deterministic inference:** identical inputs yield identical weights/masks
3. **No structural synthesis:** correlation of residuals must not show non-physical high-frequency injection; seams and ringing must not increase vs. non-ML baseline
4. **Photometric consistency:** flux ratios of calibration stars remain within tolerance (configurable) compared to baseline core
5. **Ablation:** report baseline (non-ML) vs ML-enabled improvements on the same dataset

### 9.2.7 Configuration Hooks (Normative Names)

Suggested (non-exhaustive) configuration keys:

- `ml.enable: true|false`
- `ml.global_model.path`
- `ml.tile_model.path`
- `ml.mask_model.path`
- `ml.mask.m_min`
- `ml.inference.device: cpu|gpu`
- `ml.inference.deterministic: true`

Implementations must treat missing ML models as a controlled fallback to the non-ML core.


## 10. Change History

| Date | Version | Change |
|---|---|---|
| 2026-02-25 | v3.3.6 | RBF mathematical corrections: separated reliability weights (w_j) from RBF coefficients (u_i), clarified mu as shape/scale parameter (not regularization), corrected Thin-plate kernel to standard form d^2*log(d) with epsilon for numerical stability only, added explicit parameter semantics for each kernel type |
| 2026-02-25 | v3.3.5 | BGE corrections: removed duplicate section headers, added complete RBF mathematical formulation in 6.3.7 (3 kernel types with regularized robust regression), corrected rbf_mu_factor in YAML example, fixed PCC section numbering (6.3→6.4), enhanced key mapping for RBF parameters |
| 2026-02-17 | v3.3.4 | Added spec-conform BGE YAML example and key-mapping; extended RBF basis options to include gaussian (with explicit scale parameter) |
| 2026-02-17 | v3.3.3 | BGE clarified: configurable sample quantile (median allowed), explicit background mask requirements, deterministic coarse-grid assignment with min 3 tiles per cell, and optional regularized RBF surface fitting |
| 2026-02-17 | v3.3.2 | Added formal mathematical surface model for BGE (robust polynomial/spline with IRLS) and adaptive grid definition tied to image scale |
| 2026-02-17 | v3.3 | Added strict ML optimization extension: ML restricted to weights/masks/state descriptors; core remains deterministic and strictly linear |
| 2026-02-15 | v3.2.2 | Replaced cluster-size weighted final stacking with quality-weighted cluster aggregation (exp(kappa_cluster * Q_k)) including optional dominance cap |
| 2026-02-15 | v3.2.1 | Enforced overlap clipping in tile geometry; added photometric restoration after OLA; replaced uniform per-cluster averaging with cluster-size weighted final stack |
| 2026-02-13 | v3.2 | Path A removed; CFA-based registration and channel-separation path defined as the only normative path up to phase 2 |
| 2026-02-13 | v3.2 | Consolidation after mathematical diagnostics |
| 2026-02-13 | v3.2 | Linearity semantics clarified |
| 2026-02-13 | v3.2 | Reduced-mode boundaries made explicit |
| 2026-02-13 | v3.2 | Notation unified to `f,t,c` |
| 2026-02-13 | v3.2 | Tile reconstruction/fallbacks merged into a consistent block |
| 2026-02-13 | v3.2 | Discrete Hann definition fixed normatively |
| 2026-02-13 | v3.2 | PCC test criterion replaced with a technically robust version |


---

## 11. Core Statement

The method replaces rigid search for "best frames" with robust spatio-temporal quality modeling, uses all frames without quality-based selection, and reconstructs signal where it is physically and statistically most reliable.
