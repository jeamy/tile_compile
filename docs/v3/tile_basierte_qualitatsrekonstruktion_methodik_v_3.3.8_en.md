# Tile-Based Quality Reconstruction for DSO - Methodology v3.3.8

**Status:** Normative reference specification  
**Version:** v3.3.8 (2026-03-15)
**Applies to:** `tile_compile.yaml`

---

## 0. Objective of v3.3.8

Core objectives:

1. mathematical consistency (notation, formulas, edge cases)
2. clear separation of **mandatory core** vs. **optional extensions**
3. precise semantics for
   - linearity,
   - no frame selection,
   - robust pixel outlier handling,
   - runtime-configured operating modes,
   - and the active local-score / tile-reconstruction estimator

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
- and it includes a documented fallback to the unclipped valid weighted mean.

### 1.3 Linearity Semantics (Clarified)

"Strictly linear" in v3.3.6 means:

1. **Photometric signal mapping** remains linear (no global nonlinear tone curves such as stretch, asinh, log).
2. Linear reconstruction steps (scaling, weighted mean, overlap-add) are mandatory.
3. Robust/statistical nonlinearities (MAD, clipping, sigma clipping, adaptive gating decisions) are allowed as **auxiliary steps**.

---

## 2. Assumptions and Operating Modes

### 2.1 Hard Assumptions (Violation -> Abort)

- Input data are linear (no stretch, no tone curves)
- No quality-based frame selection
- Registered geometry is expressed in the same pixel reference
- From phase 3 onward, channel semantics remain deterministic and channel-consistent in the shared core

### 2.2 Soft Assumptions and Runtime Defaults

| Assumption | Optimal | Minimum | Action if violated |
|---|---:|---:|---|
| Number of usable frames `N` | >= 800 | `>= assumptions.frames_min` | Reduced mode for `frames_min .. frames_reduced_threshold-1`; below `frames_min` abort or emergency mode |
| Exposure-time consistency | constant within one series | dataset-dependent | mixed series require explicit handling/calibration; not a guaranteed core invariant |
| Registration residual | < 0.3 px | < 1.0 px | Warning at > 0.5 px |
| Star elongation | < 0.2 | < 0.4 | Warning at > 0.3 |

### 2.3 Runtime-Configured Operating Modes (Binding)

Let

- `N_min = assumptions.frames_min`
- `N_red = assumptions.frames_reduced_threshold`, with `N_red >= N_min`

Then the active runtime uses:

- **Full mode:** `N >= N_red`
- **Reduced mode:** `N_min <= N < N_red`
- **Below minimum:** `N < N_min`

Reduced-mode consequences:

- `STATE_CLUSTERING` and `SYNTHETIC_FRAMES` are skipped if `assumptions.reduced_mode_skip_clustering = true`
- if reduced-mode clustering remains enabled, `assumptions.reduced_mode_cluster_range = [K_min, K_max]` limits the cluster search
- when clustering/synthesis are skipped, the final output reuses the reconstruction result directly

### 2.4 Below Minimum

- **N < assumptions.frames_min:** no regular reduced mode
- Standard action: controlled abort with diagnostics
- Optional only via explicit `runtime_limits.allow_emergency_mode: true`: emergency mode with warning status

### 2.5 Shared-Core Channel-Semantic Variants (Binding)

Earlier methodology text used the labels `strict` and `practical`. In v3.3.8 these are treated as **implementation variants**, not as active `tile_compile.yaml` operating-profile keys.

Two conformant shared-core variants are allowed:

- **Explicit per-channel variant**
  - channel separation is completed by phase 2
  - phases 3-10 operate per channel
- **CFA-proxy-equivalent variant**
  - the shared core may operate on CFA-proxy data before explicit RGB formation

For the CFA-proxy-equivalent variant, all of the following remain mandatory:

1. linear and deterministic reconstruction behavior in the shared core,
2. channel-equivalent weighting/estimation semantics (no hidden cross-channel coupling in the core estimator),
3. CFA phase preservation for geometric operations,
4. explicit RGB domain before color-calibration extensions (BGE/PCC), with unchanged canvas-mask exclusion policy.

---

## 3. Pipeline Overview (Normative)

1. Registration and geometric harmonization
2. Channel separation (explicit or deferred via a CFA-proxy-equivalent core)
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
Optional/feature-gated: local denoisers, alternative post-stack clipping policies, WCS/PCC.

---

## 4. Registration and Channel Separation up to Phase 2 (Normative)

Up to and including phase 2, the CFA-based registration and channel-separation path applies.
From phase 3 onward, one of the binding shared-core variants from §2.5 applies:
- explicit per-channel core,
- CFA-proxy-equivalent core.

### 4.1 CFA-Based Registration Path

- Registration on a CFA luminance proxy
- CFA-aware warp by subplanes (`warp_cfa_mosaic_via_subplanes`)
- Channel separation afterwards (explicit per-channel variant) or deferred split at channel-stack stage (CFA-proxy-equivalent variant)

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

### 4.3 CFA-Proxy Core Path (Binding)

- Global/local metrics and tile reconstruction may operate on CFA-proxy inputs instead of early explicit RGB planes.
- This is conformant only if the channel semantics and linearity constraints from §2.5 are preserved.
- Explicit RGB data are still required before BGE/PCC and for final RGB outputs.

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

`Q_{f,t,c}^{star} = 0.6*(-z_tilde(FWHM)) + 0.2*z_tilde(R) + 0.2*z_tilde(C)`

### 5.5.3 STRUCTURE Tile Metrics

- `(E/sigma)_{f,t,c}`
- `B_{f,t,c}`

Local index:

`Q_{f,t,c}^{struct} = 0.7*z_tilde(E/sigma) - 0.3*z_tilde(B)`

### 5.5.4 Neighborhood-Aware Metric Normalization (Binding in v3.3.8)

For each scalar tile metric family `m` and tile `t`, first compute the tile-local robust z-score over usable frames:

`z_local(m_{f,t,c}) = robust_z(m_{f,t,c}; {m_{f',t,c}}_{f' in F_t})`

If neighborhood normalization is enabled, additionally compute a pooled robust location/scale over neighboring tiles and all usable frames:

- `N_r(t)`: Manhattan-radius tile neighborhood with radius `r = local_metrics.neighborhood_normalization.radius`
- `P_t(m) = { m_{f',u,c} | u in N_r(t), f' usable }`

`z_pool(m_{f,t,c}) = (m_{f,t,c} - median(P_t(m))) / max(1.4826*MAD(P_t(m)), eps_mad)`

The metric z-score used by the local quality model is then

`z_tilde(m_{f,t,c}) = (1 - b_local) * z_local(m_{f,t,c}) + b_local * z_pool(m_{f,t,c})`

with `b_local = local_metrics.neighborhood_normalization.blend`.

If neighborhood normalization is disabled or `P_t(m)` is empty, use:

`z_tilde(m_{f,t,c}) = z_local(m_{f,t,c})`

Normative default parameters:

- `local_metrics.neighborhood_normalization.enabled = true`
- `local_metrics.neighborhood_normalization.radius = 1`
- `local_metrics.neighborhood_normalization.blend = 0.5`

Binding constraints:

1. normalization is metric-local and frame-order independent,
2. only finite metric samples may contribute to pooled neighborhood statistics,
3. neighborhood normalization changes scores only through metric normalization, never through direct pixel manipulation.

### 5.5.5 Spatial Regularization of Local Scores (Binding in v3.3.8)

First compute the unregularized local score:

`Q_{f,t,c}^{raw} = clip(Q_{f,t,c}^{star|struct}, q_min, q_max)`

with default local clamp range `[q_min, q_max] = [-3, +3]`.

To prevent neighboring tiles from diverging into incompatible local regimes, the local score field is regularized on the tile-neighborhood graph before exponential weighting.

Let `N(t)` be the 4-neighborhood of tile `t` on the tile grid.

For each frame `f`, tile `t`, and pass `k`:

`Q_{f,t,c}^{(k+1)} = (1 - lambda_local) * Q_{f,t,c}^{(k)} + lambda_local * mean_{u in N(t)} Q_{f,u,c}^{(k)}`

with initialization:

`Q_{f,t,c}^{(0)} = Q_{f,t,c}^{raw}`

and final regularized score after `P` passes:

`Q_{f,t,c}^{reg} = Q_{f,t,c}^{(P)}`

Normative default parameters:

- `local_metrics.spatial_regularization.enabled = true`
- `local_metrics.spatial_regularization.lambda = 0.35`
- `local_metrics.spatial_regularization.passes = 1`

Binding constraints:

1. only valid/common tiles may participate,
2. regularization is frame-local and must not couple different frames,
3. tiles without valid neighbors keep `Q_{f,t,c}^{reg} = Q_{f,t,c}^{raw}`,
4. regularization acts only on local quality scores, never directly on pixel values.

### 5.5.6 Local Weight

`Q_{f,t,c}^{local} = clip(Q_{f,t,c}^{reg}, q_min, q_max)`

`L_{f,t,c} = exp(Q_{f,t,c}^{local})`

---

## 5.6 Effective Weight

`W_{f,t,c} = G_{f,c} * L_{f,t,c}`

Semantics:

- `G`: global atmospheric quality
- `L`: local structure/sharpness quality

---

## 5.7 Tile Reconstruction (Consolidated)

Let the tile-level weights be

`w_{f,t,c} = W_{f,t,c}`.

If

`sum_f max(w_{f,t,c}, 0) <= eps_weight`

the tile enters a deterministic weight fallback and all positive finite tile weights are replaced by `1`.

For each pixel `p` in tile `t`, define the valid sample set

`V_{t,c}(p) = { f | I_{f,c}(p) is finite and > 0 }`.

If `|V_{t,c}(p)| = 0`, set

`R_{t,c}(p) = 0`.

For `|V_{t,c}(p)| <= 2` (or if clipping iterations are disabled), use the valid weighted mean directly:

`R_{t,c}(p) = sum_{f in V_{t,c}(p)} w_{f,t,c} I_{f,c}(p) / sum_{f in V_{t,c}(p)} w_{f,t,c}`.

Otherwise apply iterative weighted sigma clipping on the active set `A^{(0)} = V_{t,c}(p)`:

`mu^{(k)} = sum_{f in A^{(k)}} w_{f,t,c} I_{f,c}(p) / sum_{f in A^{(k)}} w_{f,t,c}`

`sigma^{(k)} = sqrt( sum_{f in A^{(k)}} w_{f,t,c}(I_{f,c}(p)-mu^{(k)})^2 / (V1 - V2/V1) )`

with

- `V1 = sum_{f in A^{(k)}} w_{f,t,c}`
- `V2 = sum_{f in A^{(k)}} w_{f,t,c}^2`

and update

`A^{(k+1)} = { f in A^{(k)} | mu^{(k)} - sigma_low*sigma^{(k)} <= I_{f,c}(p) <= mu^{(k)} + sigma_high*sigma^{(k)} }`

subject to the keep-floor

`|A^{(k+1)}| >= ceil(min_fraction * |V_{t,c}(p)|)`.

The final reconstruction value is the weighted mean over the final accepted set. If clipping empties the accepted set numerically, fall back to the unclipped valid weighted mean over `V_{t,c}(p)`.

Default `eps_weight = 1e-6`.

### 5.7.1 Tile Normalization before OLA (Binding)

For reconstructed tile `R_{t,c}`, let

`V_t^+ = { p | R_{t,c}(p) is finite and > 0 inside valid reconstruction support }`.

Then estimate:

1. `bg_t = median_{p in V_t^+} R_{t,c}(p)`
2. `X_t(p) = R_{t,c}(p) - bg_t`
3. `m_t = median_{p in V_t^+} |X_t(p)|`
4. if `m_t >= eps_median`: `Y_t(p) = X_t(p) / m_t`, otherwise `Y_t(p) = X_t(p)`

Default `eps_median = 1e-6`.

#### 5.7.1a Robust Tile-Normalization Guard (Binding)

Implementations must prevent pathological amplification when `m_t` is estimated from too few valid pixels or collapses far below the dataset-wide tile scale.

Required deterministic guard:

1. estimate `bg_t` and `m_t` only from finite, strictly positive tile samples inside the valid reconstruction support
2. require a minimum valid sample count per tile:
   - `n_min = max(64, ceil(0.05 * N_t))`
3. compute robust global references over valid tiles:
   - `bg_global = median_t(bg_t)`
   - `m_global = median_t(m_t)`
4. if a tile does not meet `n_min`, replace its local normalization metadata with the global references
5. clamp valid local scales to
   - `m_t in [0.5 * m_global, 2.0 * m_global]`

This guard is part of the linear affine normalization path. It does not introduce a nonlinear tone curve; it only prevents unstable tile-wise gain explosions from dominating the OLA input.

#### 5.7.1b Photometric Preservation after OLA (Binding in v3.3.8)

The normalization `Y_t = (R_{t,c} - bg_t)/m_t` equalizes local structure but can alter absolute photometric scale if left uncorrected.
To preserve a consistent global affine flux scale, accumulate per-tile metadata during reconstruction and restore a global scale/offset after OLA:

- Per tile (already computed): `bg_{t,c}`, `m_t`
- Global restoration factors (robust):
  - `m_global = median_t(m_t)`
  - `bg_global,c = median_t(bg_{t,c})`

After overlap-add produces `I_rec,c`, restore:

`I_final,c = I_rec,c * m_global + bg_global,c`

For monochrome data this reduces to the single-channel case. This keeps the reconstruction core affine in pixel values while preventing systematic tile-to-tile photometric drift.

### 5.7.2 Windowing and Overlap-Add

2D window separable with discrete Hann function:

`hann(i,N) = 0.5*(1 - cos(2*pi*i/(N-1)))`, `i=0..N-1`

Special case: `N=1 -> hann=1`.

`w(x,y) = hann(x,W_t) * hann(y,H_t)`

Reconstruction image:

- numerator accumulator: `A`
- window-sum accumulator: `S`

`A += w * Y_t`, `S += w`, result `I_rec = A / max(S, eps_weight)`

After OLA, the active runtime may apply the global affine restoration from §5.7.1b before later output scaling or downstream post-processing.

### 5.7.3 Boundary Diagnostics (Recommended, Non-Invasive)

To diagnose visible tile boundaries without changing the reconstruction result, implementations may evaluate neighboring tiles on the actual OLA input `Y_t`.

Recommended practice is to emit these diagnostics twice:

- once on the raw reconstructed tiles before the optional per-tile normalization
- once on the normalized OLA input `Y_t`

For each neighboring tile pair `(a,b)` with overlap domain `Omega_ab`, define:

`Delta_ab(p) = Y_b(p) - Y_a(p)`, for `p in Omega_ab`

Only samples inside the common canvas-valid domain may contribute. Masked canvas zones must be excluded rather than treated as valid zero-valued pixels.

Recommended pair diagnostics:

- `mean_abs_diff_ab = mean_p |Delta_ab(p)|`
- `p95_abs_diff_ab = p95_p |Delta_ab(p)|`
- `mean_signed_diff_ab = mean_p Delta_ab(p)`
- `n_ab = |Omega_ab|` valid finite overlap samples

Additionally, implementations may summarize per-pair differences in:

- valid frame support,
- post-reconstruction background metrics,
- post-reconstruction SNR proxies,
- post-reconstruction mean correlation proxies,
- and fallback mismatch flags.

Binding semantics:

1. These diagnostics must be **read-only** and must not alter `Y_t`, `A`, `S`, or the final OLA result.
2. They may be emitted as runtime artifacts for analysis and regression testing.
3. Because they do not feed back into the estimator, they do **not** change the linearity semantics of the reconstruction core.

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

Given grid spacing `G` (see 6.3.9), define axis-aligned grid cells over the image plane. Each grid cell is a rectangle of size `G x G`.

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

#### 6.3.7 Auto-Tuned BGE (Optional, Conservative) (Binding When Enabled)

This optional extension enables deterministic **test–adjust–test** tuning of BGE parameters to improve robustness under varying gradient conditions (light pollution gradients, moon gradients, airglow). The reconstruction core remains unchanged; BGE remains strictly additive and downstream.

##### 6.3.7.1 Objective (Binding)

For a given channel, define a deterministic objective:

`J = E_cv + alpha * E_flat + beta * E_rough`

with:
- `E_cv`: holdout RMS of background sample residuals evaluated on a deterministic validation split,
- `E_flat`: large-scale gradient energy of the fitted background model,
- `E_rough`: second-derivative energy of the fitted model (penalizes overfit waviness).

All terms must be computed deterministically from the same grid-cell set.

##### 6.3.7.2 Deterministic Holdout Split (Binding)

Grid cells must be sorted by `(cell_y, cell_x)` and split deterministically by selecting every k-th cell as validation, where `k = round(1/holdout_fraction)`.

`holdout_fraction` must be clamped to `[0.05, 0.50]` before split generation.

##### 6.3.7.3 Candidate Search (Conservative Defaults)

When enabled, implementations must evaluate a bounded set of candidate parameters (hard cap `max_evals`) and select the candidate with minimal `J` using deterministic tie-break rules (prefer lower roughness, then coarser effective model).

Conservative candidate families (implementation-defined but must be documented):
- `sample_quantile`: `{q0, 0.35, 0.50}`
- `structure_thresh_percentile`: `{p0, 0.85}`
- `rbf_mu_factor`: `{m0, 1.4}`
- `rbf_lambda`: may still apply internal smoothing preference (select smoothest accepted λ).

Grid spacing `G` should remain unchanged in conservative mode unless explicitly enabled by a non-conservative strategy.

`max_evals` is a hard upper bound on evaluated candidates and must be `>= 1`.

##### 6.3.7.4 Configuration Hooks (Normative Names)

- `bge.autotune.enabled: true|false`
- `bge.autotune.max_evals`
- `bge.autotune.holdout_fraction`
- `bge.autotune.alpha_flatness`
- `bge.autotune.beta_roughness`
- `bge.autotune.strategy: conservative|extended`

When `enabled=true`, the chosen parameter set and objective components must be included in diagnostics.

Minimum diagnostic fields (binding):

- `autotune.enabled`
- `autotune.strategy`
- `autotune.max_evals`
- `autotune.evals_performed`
- `autotune.best.sample_quantile`
- `autotune.best.structure_thresh_percentile`
- `autotune.best.rbf_mu_factor`
- `autotune.best.objective`
- `autotune.best.cv_rms`
- `autotune.best.flatness`
- `autotune.best.roughness`
- `autotune.fallback_used`

##### 6.3.7.5 Robustness and Fallback Semantics (Binding)

Auto-tuning must be fail-safe and deterministic:

1. If the candidate fit cannot produce sufficient valid cells/metrics, that candidate is marked failed.
2. If no candidate succeeds, the implementation must fall back to the user/base BGE configuration unchanged.
3. Tie-break for equal objective values must be deterministic (prefer lower roughness, then coarser effective model).
4. Auto-tuning must not alter core reconstruction weights (`G`, `L`, `W`) and remains strictly additive.

### 6.3.8 Mathematical Surface Model (Binding)

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

Only large-scale (low-frequency) components are permitted; overfitting is forbidden.

#### 6.3.9 Adaptive Grid Definition (Binding)

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

#### 6.3.10 AutoBGE: Two-Stage Polynomial + RBF Background Extraction (Alternative to §6.3.2–6.3.9)

When `bge.method = autobge`, an alternative background extraction algorithm is used instead of the tile-grid-based pipeline described in §6.3.2–6.3.9. AutoBGE is designed for datasets with complex, non-polynomial gradients that are poorly modeled by a single surface fit.

**Configuration.** The following parameters control AutoBGE behavior:

| Parameter | Default | Description |
|---|---|---|
| `bge.autobge.num_sample_points` | `0` (auto) | Number of sample points; 0 = auto from `max(100, downsampled_area / 10000)` |
| `bge.autobge.poly_degree` | `2` | Polynomial degree for first-stage fit |
| `bge.autobge.rbf_smooth` | `0.1` | Multiquadric RBF smoothing parameter |
| `bge.autobge.downsample_scale` | `4` | Area-based downsampling factor |
| `bge.autobge.patch_size` | `15` | Odd patch size for local median estimation |
| `bge.autobge.patch_estimator` | `median` | Patch estimator: `median` or `sigma_clipped_median` |
| `bge.autobge.stretch_mode` | `linear` | Working-space stretch: `none`, `linear`, `mtf` |
| `bge.autobge.stretch_target_median` | `0.25` | Target median for MTF stretch |
| `bge.autobge.border_margin` | `10` | Pixel margin excluded from sampling |
| `bge.autobge.bright_exclusion_fraction` | `0.5` | Fraction of brightest pixels excluded |
| `bge.autobge.gradient_descent_max_iters` | `100` | Max iterations for dim-spot gradient descent |
| `bge.autobge.mono_mode` | `rgb_duplicate` | Mono handling: `rgb_duplicate` or `disabled` |

**Algorithm (Binding).**

1. **Working-Space Transform.** Each channel is optionally stretched to enhance background visibility:
   - `none`: no transform.
   - `linear`: percentile-based linear stretch `(v - p01) / (p99 - p01)`.
   - `mtf`: unlinked non-linear stretch targeting `stretch_target_median`.
   The transform parameters are recorded per-channel for inverse transform of the background model.

2. **Downsampling.** The stretched image is downsampled by `downsample_scale` using area-based averaging. This reduces computation and suppresses noise.

3. **Sample Point Generation.** Sample points are placed on a regular grid within the usable image area (excluding `border_margin`). Bright pixels above the `bright_exclusion_fraction` percentile are excluded. Each grid point is refined by **gradient descent toward dimmer regions**: iteratively moving to the neighbor with the lowest local patch median. Duplicate points (converged to the same location) are removed.

4. **Two-Stage Fitting.**
   - **Stage 1 (Polynomial):** A 2D polynomial of degree `poly_degree` is fit to the sample point values via least-squares. The polynomial background is rendered at full resolution via Lanczos-4 upscaling.
   - **Stage 2 (RBF on Residuals):** The polynomial background is subtracted from the downsampled image. New sample points are generated on the residual. A thin-plate spline RBF with linear affine term is fit to the residual samples. The RBF background is rendered at full resolution via Lanczos-4 upscaling.
   - **Combination:** The total background model is `bg = bg_poly + bg_rbf`.

5. **Inverse Transform.** The combined background model is transformed back from working space to the original data space using the inverse of the per-channel stretch.

6. **Subtraction.** The background model is subtracted from each channel. Results are clamped to non-negative values.

**Mono Handling.** If all three RGB channels are identical (mono input), AutoBGE processes only one channel. When `mono_mode = rgb_duplicate`, the single-channel model is applied to all three channels. When `mono_mode = disabled`, AutoBGE is skipped for mono input.

**Compatibility.** AutoBGE does not use tile metrics or the tile grid. It operates directly on the stacked RGB image. The `bge.method` parameter is mutually exclusive: `none` disables BGE entirely, `classic` uses the tile-grid pipeline (§6.3.2–6.3.9), and `autobge` uses this section.


### 6.4 PCC

This specification recommends applying BGE prior to PCC when spatial background gradients are present.

#### 6.4.1 Local Background Modeling in the Sky Annulus (Binding)

PCC star photometry must subtract a local background estimated in the sky annulus. To reduce gradient bias, the background model may be:

- `median`: constant median over annulus (legacy), or
- `plane`: robust plane fit `bg(dx,dy)=a + b*dx + c*dy` over annulus pixels (recommended under gradients).

If `plane` fit fails, implementations must fall back deterministically to `median`.

#### 6.4.2 FWHM-Adaptive Radii (Optional, Recommended)

When a global seeing estimate `FWHM` is available, PCC radii may be set automatically:

- `r_ap = max(min_aperture_px, aperture_fwhm_mult * FWHM)`
- `r_in = max(r_ap + 1, annulus_inner_fwhm_mult * FWHM)`
- `r_out = max(r_in + 2, annulus_outer_fwhm_mult * FWHM)`

If `FWHM <= 0` or unavailable, implementations must deterministically fall back to `FWHM = 0`, yielding:

- `r_ap = min_aperture_px`
- `r_in = max(r_ap + 1, annulus_inner_fwhm_mult * 0) = r_ap + 1`
- `r_out = max(r_in + 2, annulus_outer_fwhm_mult * 0) = r_in + 2`

Recommended conservative defaults:
- `aperture_fwhm_mult = 1.8`
- `annulus_inner_fwhm_mult = 3.0`
- `annulus_outer_fwhm_mult = 5.0`

These changes must preserve determinism.

#### 6.4.3 Configuration Hooks (Normative Names)

- `pcc.background_model: median|plane`
- `pcc.radii_mode: fixed|auto_fwhm`
- `pcc.aperture_fwhm_mult`
- `pcc.annulus_inner_fwhm_mult`
- `pcc.annulus_outer_fwhm_mult`
- `pcc.min_aperture_px`

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

Note: The legacy PCC test "no negative matrix element" is **no longer** required as a hard criterion in v3.3+.

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

- CFA-based registration path up to explicit or deferred (shared-core-variant-dependent) channel separation
- global normalization
- global/local metrics and weights
- tile reconstruction including consolidated fallbacks
- clustering/synthesis/final stack depending on operating mode

### Optional Extension

- soft-threshold / Wiener
- alternative sigma-clipping strategies
- WCS/PCC
- specialized performance backends (GPU, queue workers)

### 9.1 Operational Example Configurations (tile_compile_cpp)

For operational use, complete reference configurations are provided:

- `tile_compile_cpp/examples/full_mode.example.yaml`
- `tile_compile_cpp/examples/reduced_mode.example.yaml`
- `tile_compile_cpp/examples/emergency_mode.example.yaml`
- `tile_compile_cpp/examples/smart_telescope_dwarf_seestar.example.yaml`

All example files include the active configuration surface with inline comments.
They expose runtime thresholds such as `assumptions.frames_min` and `assumptions.frames_reduced_threshold` explicitly.
The channel-semantic shared-core variant is an implementation property, not a user-facing `assumptions.pipeline_profile` switch.

Procedure:

1. copy the appropriate profile,
2. adapt `run_dir`, `input.pattern`, and sensor parameters (`image_width/height`, `bayer_pattern`),
3. launch the runner with this file.

---

## 9.2 Constrained ML Optimization Extension (Optional, Non-Core)

This extension introduces machine-learning (ML) modules **only** to improve the estimation of weights and state descriptors while preserving the mandatory core invariants:

### 9.2.1 Binding Invariants (Hard Constraints)

1. **No frame selection:** Entire frames must not be removed based on quality (unchanged from v3.2.x).
2. **Strict photometric linearity of the reconstruction core:** The final reconstruction must remain a weighted linear estimator over input frames (and/or synthetic frames), i.e. of the form

   `R(p) = sum_i w_i(p) * X_i(p) / sum_i w_i(p)`

   with `w_i(p) >= 0` and deterministic fallbacks.
3. **Determinism:** ML inference must be deterministic (fixed model weights, fixed preprocessing, fixed seeds where applicable).
4. **No hallucinated content:** ML modules must not generate new spatial structures. ML outputs are restricted to **weights, masks, metrics, and state labels**.
5. **Channel separation preserved:** ML modules must operate per-channel or on explicitly defined channel-aggregated features; no implicit cross-channel coupling in the core estimator.

### 9.2.2 Allowed ML Outputs (Constrained)

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

- **Self-supervised learning:** consistency across random frame subsets, Noise2Self/Noise2Void style objectives for masks/denoising proxies (note: denoising must still output masks/weights, not pixels inside the mandatory reconstruction core).
- **Weak supervision via proxies:** optimize weights to improve deterministic metrics (FWHM, ellipticity, background RMS, seam score) on validation sets.
- **Uncertainty-aware models:** output confidence to avoid overconfident downweighting; uncertainty must be mapped into bounded masks/weights.

### 9.2.5 Models That Fit the Constrained Output Constraint (Non-Binding)

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


#### RBF Surface (Binding, when `bge.fit.method = rbf`)

Let grid-cell samples be `(r_j, b_j, ω_j)` for `j = 1..J`, where:

- `r_j = (x_j, y_j)` are grid cell centers
- `b_j` is the aggregated background value
- `ω_j ≥ 0` is the cell reliability weight

Define the RBF surface with affine trend term:

`B_c(r) = sum_{i=1..J} u_i * φ(||r - r_i||; μ) + a0 + a1*x + a2*y`

where:

- `u_i` are RBF coefficients (unknown)
- `(a0, a1, a2)` is an affine term (recommended; improves extrapolation stability)
- `μ > 0` is the kernel shape/scale parameter
- `φ` is the chosen radial kernel

##### Supported Kernels (Binding)

1. Multiquadric:

   `φ(d; μ) = sqrt(d^2 + μ^2)`

2. Thin-plate spline:

   `φ(d) = d^2 * log(d + ε)`

   with small `ε > 0` for numerical stability (recommended: `ε = 1e-6 * G`).

3. Gaussian:

   `φ(d; μ) = exp(-d^2 / (2 * μ^2))`

   For Gaussian, `μ` acts as bandwidth (`σ`).

##### Robust Regularized Fit (Binding)

Solve for parameters `θ = (u, a)` by minimizing:

`argmin_θ sum_{j=1..J} ω_j * ρ(b_j - B_c(r_j)) + λ * ||u||_2^2`

where:

- `ρ` is the configured robust loss (Huber or Tukey)
- `λ > 0` is mandatory regularization when using RBF
- Optimization must be deterministic (IRLS or equivalent).

RBF without regularization (`λ = 0`) is forbidden.

##### Recommended Defaults

- `μ = G` (grid spacing)
- `λ = 1e-4` (tune within `[1e-6, 1e-2]` depending on gradient strength)
- Include affine term by default.

---

## 10. Core Statement

The method replaces rigid search for "best frames" with robust spatio-temporal quality modeling, uses all frames without quality-based selection, and reconstructs signal where it is physically and statistically most reliable.
