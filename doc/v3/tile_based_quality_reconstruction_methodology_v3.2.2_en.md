# Tile-Based Quality Reconstruction for DSO - Methodology v3.2.2

**Status:** Normative reference specification  
**Version:** v3.2.2 (2026-02-13)  

---

## 0. Objective of v3.2.2

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

### 1.2 No Quality-Based Frame Selection (Invariant)

**Forbidden:** Removal of entire frames for atmospheric/photometric quality ranking ("best-N" style selection).  
**Permitted:**

1. Pixel-wise outlier rejection (sigma clipping), provided that it
   - acts only at pixel level,
   - uses deterministic parameters,
   - includes a documented fallback to the unchanged mean.
2. Geometric registration validity gating before phase-3 weighting, i.e. rejecting frames whose estimated warp is physically implausible or clearly failed (e.g. reflection warp, extreme scale, catastrophic shift outlier, very low registration correlation).

Interpretation: all frames that pass geometric registration validity are retained for the quality-weighted reconstruction; no additional quality-based frame culling is allowed downstream.

### 1.3 Linearity Semantics (Clarified)

"Strictly linear" in v3.2.2 means:

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

### 4.3 Registration Validity Gating (Normative)

After registration candidate selection, each frame SHALL pass a deterministic geometric validity gate before entering the shared core (phase 3+).

Recommended default gate dimensions:

- registration correlation lower bound (`reject_cc_min_abs` and optional robust MAD-based bound)
- shift magnitude outlier bound (absolute floor + robust median multiplier)
- similarity scale bounds (`reject_scale_min`, `reject_scale_max`)
- reflection rejection (`det(warp) < 0` => reject)

Semantics:

- This step is **not** quality ranking; it is failure/outlier suppression for invalid geometry.
- Rejected frames MUST be explicitly logged with reason(s) and diagnostics.
- The invariant in 1.2 remains intact: no post-gating quality-based frame selection.

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

1. `m_t = median(R_{t,c})`  ← computed **before** background subtraction
2. `bg_t = m_t`
3. `X_t = R_{t,c} - bg_t`
4. if `m_t >= eps_median`: `Y_t = X_t / m_t`, otherwise `Y_t = X_t`

Default `eps_median = 1e-6`.

> **Rationale (EN):** `m_t` is defined as the absolute median of the tile *before* subtraction (≈ normalised background level, ~1.0 for globally normalised frames). Using `median(abs(X_t))` after subtraction yields values near zero for sky-dominated tiles, which causes §5.7.1a to collapse the dynamic range of the final image to ~0.1 % (observed: M31 core at 0.9 % instead of ~90 %).

#### 5.7.1a Photometric Preservation after OLA (Recommended)

The normalization `Y_t = (R_{t,c} - bg_t)/m_t` equalizes local structure but can alter absolute photometric scale if left uncorrected.
To preserve a consistent global affine flux scale, accumulate per-tile metadata during reconstruction and restore a global scale/offset after OLA:

- Per tile (already computed): `bg_t = m_t` (absolute median before subtraction)
- Global restoration factors (robust):
  - `m_global = median_t(m_t)`  ← ≈ 1.0 for globally normalised frames
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

Practical sensitivity of `kappa_cluster` (assuming `Q_k` span approximately `[-3,+3]`):

| κ (`kappa_cluster`) | max weight ratio (≈ `e^{6κ}`) | Character |
|---:|---:|---|
| 0.3 | ~ `e^{1.8}` ≈ 6 | very mild |
| 0.5 | ~ `e^{3}` ≈ 20 | moderate |
| 1.0 | ~ `e^{6}` ≈ 403 | strong |
| 1.5 | ~ `e^{9}` ≈ 8103 | very aggressive |
| 2.0 | ~ `e^{12}` ≈ 162k | practically winner-takes-most |

Recommendation (astrophotographic datasets):

- Default: `κ = 0.5 ... 1.0`
- `κ = 1.2` only if intentionally targeting lucky-imaging-like behavior
- `κ >= 1.5` is typically unstable (numerically and statistically)

Practical ranges for `r_cap`:

| `r_cap` | Behavior |
|---:|---|
| 5 | very conservative |
| 10 | mildly bounded |
| 20 | moderate |
| 50 | little intervention |
| >100 | effectively disabled |

Recommendation:

- Conservatively stable: `r_cap = 10`
- Balanced: `r_cap = 20-30`
- Nearly unbounded: `r_cap >= 50`

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

### 6.3 PCC

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

Note: The legacy PCC test "no negative matrix element" is **no longer** required as a hard criterion in v3.2.2.

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

## 10. Change History

| Date | Version | Change |
|---|---|---|
| 2026-02-18 | v3.2.2.4 | §5.7.1: `m_t` redefined as `median(R_{t,c})` before background subtraction (was `median(abs(X_t))` after subtraction). The old formulation yielded `m_t ≈ 0` for sky-dominated tiles, causing §5.7.1a to crush dynamic range to ~0.1 % (M31 core observed at 0.9 % instead of ~90 %). / §5.7.1: `m_t` neu definiert als `median(R_{t,c})` vor der Hintergrundsubtraktion (vorher `median(abs(X_t))` nach Subtraktion). Die alte Formulierung lieferte `m_t ≈ 0` für himmeldominierte Tiles und komprimierte den Dynamikbereich in §5.7.1a auf ~0,1 % (M31-Kern bei 0,9 % statt ~90 % beobachtet). |
| 2026-02-15 | v3.2.2.3 | Clarified invariant semantics: no quality-based frame selection while allowing deterministic registration validity gating for geometrically invalid frames |
| 2026-02-15 | v3.2.2.2 | Replaced cluster-size weighted final stacking with quality-weighted cluster aggregation (exp(kappa_cluster * Q_k)) including optional dominance cap |
| 2026-02-15 | v3.2.2.1 | Enforced overlap clipping in tile geometry; added photometric restoration after OLA; replaced uniform per-cluster averaging with cluster-size weighted final stack |
| 2026-02-13 | v3.2.2 | Path A removed; CFA-based registration and channel-separation path defined as the only normative path up to phase 2 |
| 2026-02-13 | v3.2.2 | Consolidation after mathematical diagnostics |
| 2026-02-13 | v3.2.2 | Linearity semantics clarified |
| 2026-02-13 | v3.2.2 | Reduced-mode boundaries made explicit |
| 2026-02-13 | v3.2.2 | Notation unified to `f,t,c` |
| 2026-02-13 | v3.2.2 | Tile reconstruction/fallbacks merged into a consistent block |
| 2026-02-13 | v3.2.2 | Discrete Hann definition fixed normatively |
| 2026-02-13 | v3.2.2 | PCC test criterion replaced with a technically robust version |


---

## 11. Core Statement

The method replaces rigid search for "best frames" with robust spatio-temporal quality modeling, retains all geometrically valid frames without downstream quality-based culling, and reconstructs signal where it is physically and statistically most reliable.
