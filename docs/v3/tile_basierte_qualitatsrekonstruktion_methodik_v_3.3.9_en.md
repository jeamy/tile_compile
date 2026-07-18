# Tile-Based Quality Reconstruction for DSO - Methodology v3.3.9

**Status:** Normative reference specification  
**Version:** v3.3.9 (2026-03-27)  
**Last revised:** 2026-03-27 
**Applies to:** `tile_compile.yaml`

---

## 0. Objective of v3.3.9

Core objectives:

1. mathematical consistency (notation, formulas, edge cases)
2. photometrically clean separation of additive background vs. multiplicative photometric scaling
3. a mandatory reconstruction core that remains linear in pixel values and does **not** use tile-wise nonlinear renormalization before OLA
4. soft local quality blending instead of a hard STAR/STRUCTURE switch
5. mass-preserving cluster aggregation in full mode
6. support-aware overlap-add semantics, including deterministic boundary coverage

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

"Strictly linear" (as established in v3.3.6, unchanged in v3.3.9) means:

1. **Photometric signal mapping** remains linear (no global nonlinear tone curves such as stretch, asinh, log).
| `aqmh.storage.dtype` | `float32` | Cache-Datentyp (`float32`, `uint16` oder `uint8`) |
| `aqmh.storage.max_resident_maps` | `2` | Max. gleichzeitig im RAM gehaltene Qualitätskarten |
| `aqmh.cherry_pick.enabled` | `false` | Nur beste Frames stacken |
| `aqmh.cherry_pick.k_frac` | `0.30` | Anteil bester Frames (0.30 = beste 30%) |
| `aqmh.cherry_pick.k_min_required` | `20` | Lauf-Gate und Untergrenze Samples pro Pixel |
| `aqmh.diagnostics.enabled` | `true` | AQMH-Diagnosephase aktivieren |
| `aqmh.diagnostics.level` | `full` | Detaillierungsgrad: `none`, `summary` oder `full` |

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

Earlier methodology text used the labels `strict` and `practical`. In v3.3.9 these are treated as **implementation variants**, not as active `tile_compile.yaml` operating-profile keys.

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

> **Two reconstruction paths:** The classic tile-based path (steps 1–10 below) and the AQMH path (§5.12). AQMH (Adaptive Quality Map Harvesting) is the **default** since v0.3.0; it replaces steps 6–9 with pixel-wise quality maps and independent reconstruction. The classic path is available via `method: classic_tile_compile`.

1. Registration and geometric harmonization
2. Channel separation (explicit or deferred via a CFA-proxy-equivalent core)
3. Global linear normalization
4. Global frame metrics and global weights
5. Tile geometry
6. Local tile metrics and local weights *(classic only; AQMH: quality map computation, see §5.12)*
7. Tile reconstruction (overlap-add) *(classic only; AQMH: pixel-wise weighted reconstruction, see §5.12)*
8. State-based clustering (full mode only) *(classic only; AQMH: skipped)*
9. Synthetic frames (full mode only) *(classic only; AQMH: skipped)*
10. Final linear stacking *(classic only; AQMH: pass-through)*
11. Post-processing (optional, not part of the quality core)

Mandatory core: 1-10 (classic) or 1-5 + §5.12 (AQMH).  
Optional/feature-gated: local denoisers, deterministic defect-pixel suppression / cosmetic correction, alternative post-stack clipping policies, WCS/PCC, post-PCC isolated chroma-speckle suppression.

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

#### 4.1.1 Optional Pre-Warp CFA Defect-Pixel Suppression (Feature-Gated)

Before CFA-aware warp, implementations may apply a deterministic per-frame CFA cosmetic-correction step to suppress isolated defect pixels on the raw mosaic.

Binding conditions for this optional step:

1. CFA phase and sample geometry must remain unchanged; no demosaic or cross-channel resampling is permitted before warp.
2. Corrections must remain local and sparse, limited to isolated same-parity outliers (for example hot / cold / chromatically unstable sensels) or an equivalent deterministic defect-pixel signature.
3. Replacement values must be derived from finite same-parity CFA neighbors only, or from an equivalent CFA-phase-preserving local estimator.
4. Compact real image structure must be protected by a deterministic structure guard; implementations must not erase supported multi-pixel image content under the guise of defect suppression.
5. This step is optional and is not part of the mandatory linear quality core from phases 3-10.

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

**Edge cases (binding):**

1. **Reference frame:** `NCC(identity, ref) = 1.0` for the reference frame itself, making the criterion unsatisfiable. Implementations must accept the identity transform for the reference frame unconditionally without cascade attempts.
2. **Near-perfect alignment:** If `NCC(identity, ref) >= 1 - delta_ncc`, no registration method can satisfy the criterion. Implementations must accept the identity transform directly and log a diagnostic.
3. In both cases the identity fallback is the correct outcome and must not be counted as a cascade failure.

### 4.3 CFA-Proxy Core Path (Binding)

- Global/local metrics and tile reconstruction may operate on CFA-proxy inputs instead of early explicit RGB planes.
- This is conformant only if the channel semantics and linearity constraints from §2.5 are preserved.
- Explicit RGB data are still required before BGE/PCC and for final RGB outputs.

---

## 5. Shared Core from Phase 3 Onward

### 5.1 Notation (Binding)

- `f` frame index, `t` tile index, `c` channel index, `p` pixel
- `I_{f,c}^{raw}(p)` raw linear input image per frame/channel
- `I_{f,c}(p)` normalized input image per frame/channel
- `B_{f,c}` global additive background (before normalization)
- `P_{f,c}` global photometric scale (before normalization)
- `sigma_{f,c}` global noise (after normalization)
- `E_{f,c}` global gradient energy (after normalization)
- `Q_{f,c}` global quality index
- `G_{f,c}` global weight
- `eta_t` tile star-support blend factor
- `U_{f,t,c}` local metric confidence
- `Q_{f,t,c}^{local}` local quality index
- `L_{f,t,c}` local weight
- `W_{f,t,c}` effective weight
- `M_{k,c}` cluster mass in full mode
- `M_{t,c}(p)` per-channel support mask for OLA (1 = valid canvas pixel with at least one valid frame, 0 = canvas-invalid or no valid frames)
- `k_local` local weight scale factor (§5.5.6)
- `k_global` global weight scale factor (§5.3.2)

**From this point onward, channel index `c` is used consistently.**

---

## 5.2 Global Linear Normalization (Mandatory)

Order:

1. Additive background from raw data:
   - `B_{f,c} = median(I_{f,c}^{raw})`
2. Background subtraction:
   - `J_{f,c}(p) = I_{f,c}^{raw}(p) - B_{f,c}`
3. Photometric scale from deterministic throughput/flux reference:
   - `P_{f,c} = photometric_scale(J_{f,c})`

   **Definition of `photometric_scale` (binding):** Implementations must use one of the following deterministic methods, in priority order:

   a. **Ensemble stellar flux scaling (recommended):** `P_{f,c} = median_s(flux_{f,c,s}) / median_s(flux_{ref,c,s})`, where `flux_{f,c,s}` is the instrumental aperture flux of non-saturated reference star `s` in frame `f`, channel `c`. The star set must be fixed (determined from the reference frame) and deterministic.

   b. **Exposure-time ratio:** If all frames share identical optical throughput and only exposure times differ: `P_{f,c} = t_f / t_ref`, where `t_ref` is the reference frame exposure time.

   c. **Deterministic fallback (`P_{f,c} = 1`):** When neither stellar fluxes nor exposure metadata are reliably available, and throughput is already consistent across frames (see constraint 5 below).

   Constraint: `P_{f,c}` must not be derived solely from the sky-background level `B_{f,c}`.
4. Linear scaling:
   - `I_{f,c}(p) = J_{f,c}(p) / max(P_{f,c}, eps_scale)`
5. Metrics on normalized data:
   - `sigma_{f,c}`, `E_{f,c}`

Binding constraints:

1. additive background subtraction and multiplicative photometric scaling must remain separate operations,
2. `P_{f,c}` must not be defined solely by the sky-background level,
3. valid negative or zero pixel values after subtraction/scaling remain admissible samples,
4. global nonlinear tone curves remain forbidden,
5. if no reliable photometric scale can be estimated and exposure throughput is already consistent, implementations may deterministically use `P_{f,c} = 1`.

Recommended default:

- `eps_bg = 1e-6`
- `eps_scale = 1e-6`

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

**Notation note:** The symbols `alpha`, `beta`, `gamma` used here refer exclusively to the global quality index weights. The BGE auto-tuning objective (§6.3.7.1) uses the distinct symbols `alpha_f` (flatness weight) and `beta_r` (roughness weight) to avoid collision.

**Note on mixed normalization:** `B_{f,c}` is a pre-normalization quantity while `sigma_{f,c}` and `E_{f,c}` are post-normalization (§5.2 step 5). The z-score operation makes them scale-independent, so the combination is statistically valid. In mixed-exposure datasets, `z(B_{f,c})` may reflect exposure duration rather than sky brightness; in such cases the `alpha` weight should be reduced or `B` z-score recomputed on exposure-normalized data.

Clamping before exponential:

`Q_{f,c}^{clamped} = clip(Q_{f,c}, -3, +3)`

Global weight:

`G_{f,c} = exp(k_global * Q_{f,c}^{clamped})`

with `k_global > 0`, default `k_global=1.0`.

### 5.3.3 Optional Adaptive Weighting

The mandatory core uses the static weights from §5.3.2.

If `global_metrics.adaptive_weights=true`, adaptive weights remain an optional extension and must satisfy all of the following:

1. they are derived from a deterministic predictive-utility criterion, not merely from `Var(z(.))` of already normalized metrics,
2. they are clipped to `[0.1, 0.7]` and renormalized to sum 1,
3. they fall back to the static defaults for degenerate or uninformative utility estimates,
4. the utility target and tie-break rules must be documented.

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

`T0 = s * F`

`o_clipped = clip(o, 0, 0.5)`

`T_hi = floor(min(W,H)/D)`

Size selection (binding):

1. if `F <= 0` -> set `F = 3.0`
2. require `T_min >= 16`
3. require `D >= 1`
4. if `T_hi < T_min`, enter deterministic compact-tile mode:
   - `T = min(W,H)`
   - `O = 0`
   - `S = T`
5. otherwise:
   - `T = floor(clip(T0, T_min, T_hi))`
   - `O = floor(o_clipped * T)`
   - `S = T - O`

Additional guards (binding):

1. if `S <= 0` -> set `o_clipped = 0.25`, recompute `O,S`
2. if `O >= T` -> set `O = floor(0.25 * T)`, `S = T - O`
3. if `T <= 0` -> abort with diagnostics

**Note:** Guards 1 and 2 are invariant under the stated preconditions (`o_clipped ∈ [0, 0.5]` and `T ≥ 1` guaranteed by guard 3), so they will never trigger in normal operation. They are retained as defensive safeguards against implementation deviations.

---

## 5.5 Local Tile Metrics

**Canvas exclusion (binding for all of §5.5):** All local tile metric computations — FWHM, roundness, contrast, `E/sigma`, background `B_{f,t,c}` — must operate exclusively on pixels that are both finite and within valid canvas support for that frame. Canvas-invalid pixels must be excluded from all metric accumulators. If the canvas-valid pixel count in a tile falls below a minimum (recommended: 16 px), the tile's metrics for that frame are marked as unavailable and the fallback from §5.5.1 applies.

### 5.5.1 Soft Local Support Blend (Binding in v3.3.9)

The mandatory core does **not** use a hard STAR/STRUCTURE switch.

Instead, each tile receives a deterministic star-support blend factor:

`eta_t = clip(star_count_t / max(tile.star_soft_count, 1), 0, 1)`

with normative default:

- `tile.star_soft_count = tile.star_min_count`

Semantics:

- `eta_t = 1`: fully star-supported tile
- `eta_t = 0`: purely structure-driven tile
- intermediate values blend both local models continuously

If star metrics are unavailable or non-finite for a given frame/tile/channel term, the effective star contribution for that term is treated as unavailable and the blend falls back to the structure component.

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

### 5.5.4 Neighborhood-Aware Metric Normalization (Binding in v3.3.9)

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

### 5.5.5 Spatial Regularization of Local Scores (Binding in v3.3.9)

First compute the unregularized local score:

`Q_{f,t,c}^{blend} = eta_t * Q_{f,t,c}^{star} + (1 - eta_t) * Q_{f,t,c}^{struct}`

If one component is unavailable, use the other available component directly.

`Q_{f,t,c}^{raw} = clip(Q_{f,t,c}^{blend}, q_min, q_max)`

with default local clamp range `[q_min, q_max] = [-3, +3]`.

To prevent neighboring tiles from diverging into incompatible local regimes while preserving reliable local differences, the local score field is regularized on the tile-neighborhood graph before exponential weighting.

Let `N(t)` be the 4-neighborhood of tile `t` on the tile grid.

Define a local confidence term

`U_{f,t,c} = clip(n_valid_metrics_{f,t,c} / max(n_expected_metrics_{f,t,c}, 1), 0, 1)`

and an edge-aware neighbor affinity

`A_{t,u}^{(k)} = exp(-|Q_{f,t,c}^{(k)} - Q_{f,u,c}^{(k)}| / tau_local)`

For each frame `f`, tile `t`, and pass `k`:

`lambda_{f,t,c}^{eff} = lambda_local * (1 - U_{f,t,c})`

Let `S_A^{(k)} = sum_{u in N(t)} A_{t,u}^{(k)}`.

**Affinity-zero guard (binding):** If `S_A^{(k)} < eps_affinity`, set `Q_{f,t,c}^{(k+1)} = Q_{f,t,c}^{(k)}` (no regularization for this tile/pass). Otherwise:

`Q_{f,t,c}^{(k+1)} = (1 - lambda_{f,t,c}^{eff}) * Q_{f,t,c}^{(k)} + lambda_{f,t,c}^{eff} * sum_{u in N(t)} A_{t,u}^{(k)} Q_{f,u,c}^{(k)} / S_A^{(k)}`

Default `eps_affinity = 1e-6`.

with initialization:

`Q_{f,t,c}^{(0)} = Q_{f,t,c}^{raw}`

and final regularized score after `P` passes:

`Q_{f,t,c}^{reg} = Q_{f,t,c}^{(P)}`

Normative default parameters:

- `local_metrics.spatial_regularization.enabled = true`
- `local_metrics.spatial_regularization.lambda = 0.35`
- `local_metrics.spatial_regularization.passes = 1`
- `local_metrics.spatial_regularization.tau_local = 1.0`
- `local_metrics.spatial_regularization.eps_affinity = 1e-6`

Binding constraints:

1. only valid/common tiles may participate,
2. regularization is frame-local and must not couple different frames,
3. tiles without valid neighbors or with `lambda_{f,t,c}^{eff} <= 0` keep `Q_{f,t,c}^{reg} = Q_{f,t,c}^{raw}`,
4. regularization acts only on local quality scores, never directly on pixel values.

### 5.5.6 Local Weight

`Q_{f,t,c}^{local} = clip(Q_{f,t,c}^{reg}, q_min, q_max)`

`L_{f,t,c} = exp(k_local * Q_{f,t,c}^{local})`

with `k_local > 0`, default `k_local = 1.0`.

**Symmetry with global weight:** The scale factors `k_global` (§5.3.2) and `k_local` independently control the sensitivity of global and local exponential weighting. Both default to 1.0. Increasing `k_local` sharpens the local quality contrast; decreasing it toward 0 equalizes local weights across tiles.

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

`V_{t,c}(p) = { f | I_{f,c}(p) is finite and inside valid canvas support }`.

If `|V_{t,c}(p)| = 0`, set

`R_{t,c}(p) = 0`.

For `|V_{t,c}(p)| <= 2` (or if clipping iterations are disabled), use the valid weighted mean directly:

`R_{t,c}(p) = sum_{f in V_{t,c}(p)} w_{f,t,c} I_{f,c}(p) / sum_{f in V_{t,c}(p)} w_{f,t,c}`.

Otherwise apply iterative weighted sigma clipping on the active set `A^{(0)} = V_{t,c}(p)`:

`mu^{(k)} = sum_{f in A^{(k)}} w_{f,t,c} I_{f,c}(p) / sum_{f in A^{(k)}} w_{f,t,c}`

with

- `V1 = sum_{f in A^{(k)}} w_{f,t,c}`
- `V2 = sum_{f in A^{(k)}} w_{f,t,c}^2`
- `N_eff = V1^2 / max(V2, eps_weight)`
- `D_eff = V1 - V2 / max(V1, eps_weight)`

If `N_eff <= 2 + eps_neff` or `D_eff <= eps_var`, terminate clipping and use the weighted mean over the current active set.

Otherwise:

`sigma^{(k)} = sqrt( sum_{f in A^{(k)}} w_{f,t,c}(I_{f,c}(p)-mu^{(k)})^2 / D_eff )`

and update

`A_prop^{(k+1)} = { f in A^{(k)} | mu^{(k)} - sigma_low*sigma^{(k)} <= I_{f,c}(p) <= mu^{(k)} + sigma_high*sigma^{(k)} }`

subject to the keep-floor

`|A_prop^{(k+1)}| >= ceil(min_fraction * |V_{t,c}(p)|)`

with `min_fraction in (0, 1)`, normative default `min_fraction = 0.5`.

Deterministic keep-floor rule:

- if the proposed set satisfies the keep-floor, accept it:
  - `A^{(k+1)} = A_prop^{(k+1)}`
- otherwise terminate clipping and keep the previous active set:
  - `A^{(*)} = A^{(k)}`

The final reconstruction value is the weighted mean over the final accepted set. If clipping empties the accepted set numerically, fall back to the unclipped valid weighted mean over `V_{t,c}(p)`.

Default `eps_weight = 1e-6`.

Recommended guards:

- `eps_neff = 1e-6`
- `eps_var = 1e-12`
- `min_fraction = 0.5` (configurable via `tile_reconstruction.sigma_clip.min_fraction`)

### 5.7.1 Support-Aware Windowing and Overlap-Add (Binding in v3.3.9)

The mandatory core does **not** apply tile-wise median/MAD renormalization before overlap-add.

Instead, the OLA input is the reconstructed tile itself:

`Y_{t,c}(p) = R_{t,c}(p)`

Let

- `M_{t,c}(p) = 1` if `|V_{t,c}(p)| > 0` AND `R_{t,c}(p)` is finite AND pixel `p` is inside valid canvas support, otherwise `0`
- `omega_t(p) >= 0` be the deterministic blend window for tile `t`

**Canvas semantics (binding):** `M_{t,c}(p) = 0` in all of the following cases:
1. `|V_{t,c}(p)| = 0` (no valid frames contribute to pixel `p` in tile `t`, channel `c`); the convention `R_{t,c}(p) = 0` is a fill value only and must NOT enter the OLA accumulator.
2. Pixel `p` is outside valid canvas support (canvas-masked region), regardless of whether frames nominally cover it.
3. `R_{t,c}(p)` is non-finite (NaN or Inf).

The support mask carries a channel index `c` because in CFA-proxy-equivalent processing the number of valid frames per pixel can differ per channel.

**Note on fill-zero exclusion:** Setting `M_{t,c}(p) = 0` when `|V| = 0` prevents fill-zero values from diluting the OLA denominator `S` at coverage boundaries. Without this, `S` would accumulate window weights from tiles with no valid data, biasing `I_rec` toward zero in partially covered regions.

Binding OLA semantics:

1. `omega_t` must depend only on tile geometry, overlap geometry, and boundary position, never on pixel values,
2. `omega_t` must be non-negative,
3. for every valid canvas pixel `p`, implementations must guarantee
   - `sum_t omega_t(p) * M_{t,c}(p) > 0`,
4. for interior pixels with full support, the windows must form a partition of unity up to numeric tolerance:
   - `sum_t omega_t(p) * M_{t,c}(p) in [1 - tol_ola, 1 + tol_ola]`,
5. tiles touching an outer image boundary must use one-sided boundary-aware windows or an equivalent support-aware renormalization; symmetric zero-at-both-ends windows are not permitted on the outer canvas boundary unless rules 3 and 4 remain satisfied by construction.

Recommended defaults:

- `tol_ola = 1e-6`
- interior tiles: separable cosine/Tukey partition windows
- boundary tiles: one-sided cosine ramps only across actual overlap zones

Accumulation:

- numerator accumulator: `A`
- support accumulator: `S`

`A += omega_t * M_{t,c} * Y_{t,c}`

`S += omega_t * M_{t,c}`

`I_rec = A / max(S, eps_weight)`

**Canvas guarantee:** Only pixels with `M_{t,c}(p) = 1` contribute to `A` and `S`. Canvas-invalid pixels contribute nothing to either accumulator and thus do not enter any calculation. The final `I_rec` at a canvas-invalid pixel will be `0 / eps_weight`, which must be subsequently masked out in the output (not interpreted as a valid reconstruction value).

### 5.7.2 Boundary Diagnostics (Recommended, Non-Invasive)

To diagnose visible tile boundaries without changing the reconstruction result, implementations may evaluate neighboring tiles on the actual OLA input `R_{t,c}` and on the windowed input `omega_t * R_{t,c}`.

Recommended practice is to emit these diagnostics twice:

- once on the raw reconstructed tiles `R_{t,c}`
- once on the windowed OLA input `omega_t * R_{t,c}`

For each neighboring tile pair `(a,b)` with overlap domain `Omega_ab`, define:

`Delta_ab(p) = R_{b,c}(p) - R_{a,c}(p)`, for `p in Omega_ab`

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

1. These diagnostics must be **read-only** and must not alter `R_{t,c}`, `omega_t`, `A`, `S`, or the final OLA result.
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

Active only in **full mode** (`N >= N_red`, see §2.3). Additionally, clustering requires a minimum of 50 usable frames to yield at least `K_min = 5` meaningful clusters (`K = clip(floor(N/10), K_min, K_max)` yields K ≥ 5 for N ≥ 50). If `N_red < 50`, the effective threshold is `max(N_red, 50)`.

**Removed:** The previously stated fixed threshold `N >= 200` is superseded by the configurable mode framework. Implementations must not hardcode 200 as a clustering gate.

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

**Zero-denominator fallback:** If `sum_{f in k} G_{f,c} <= eps_weight` (degenerate cluster), use the unweighted mean: `S_{k,c} = sum_{f in k} I_{f,c} / |k|`.

### 5.10.2 Optional (tile_weighted)

If `synthetic.weighting=tile_weighted`:

- reconstruct per tile/channel with `W_{f,t,c}`
- assemble to `S_{k,c}` via the same support-aware OLA semantics from §5.7.1
- implementations must monitor the tile-boundary diagnostics from §5.7/§11; if tile-weighted synthesis shows simultaneous boundary regression and cross-tile weight disagreement, they must deterministically fall back to §5.10.1 (`global`) for synthetic-frame formation instead of reintroducing nonlinear per-tile renormalization
- the requested and effective synthetic weighting modes must be recorded in diagnostics

### 5.10.3 Semantics of Phase 7 vs 9

- Full mode with `global`: phase 7 primarily provides local quality modeling/diagnostics; the final product is generated from phases 9+10.
- Full mode with `tile_weighted`: local tile quality is explicitly propagated into synthetic frames unless the seam guard above falls back to `global`.
- Reduced mode: the output from phase 7 is the direct final product.

---

## 5.11 Final Linear Stacking

### 5.11.1 Cluster Quality and Mass Definition (Binding in v3.3.9)

For each cluster `k` and channel `c`, define:

`Q_{k,c} = median_{f in k}(Q_{f,c}^{clamped})`

`M_{k,c} = sum_{f in k} G_{f,c}`

where `Q_{f,c}^{clamped}` is the global frame quality index already limited to `[-3,+3]`.

If `M_{k,c} <= eps_weight`, replace it deterministically by

`M_{k,c} = |k|`.

### 5.11.2 Mass-Preserving Quality-Weighted Cluster Aggregation (Binding in v3.3.9)

Clusters are aggregated using exponential quality weighting **and** cluster mass preservation:

`Q_{k,c}^{rel} = clip(Q_{k,c} - median_j(Q_{j,c}), -3, +3)`

`w_{k,c}^{raw} = M_{k,c} * exp(kappa_cluster * Q_{k,c}^{rel})`

with:

- `kappa_cluster > 0` (recommended default: `kappa_cluster = 1.0`)
- `Q_{k,c}^{rel}` already clamped to `[-3,+3]`

Optional stability cap (recommended):

`w_{k,c} = min(w_{k,c}^{raw}, r_cap * median_j(w_{j,c}^{raw}))`

with recommended `r_cap` in `[10, 50]`.

If weight capping is disabled, use:

`w_{k,c} = w_{k,c}^{raw}`

Final result per channel:

`R_c = sum_k (w_{k,c} * S_{k,c}) / sum_k w_{k,c}`

**Zero-denominator fallback:** If `sum_k w_{k,c} <= eps_weight` (degenerate weight distribution), fall back to the equal-weight mean: `R_c = sum_k S_{k,c} / K`. This fallback must be logged as a warning.

### 5.11.3 Semantics

- Better atmospheric states (higher `Q_{k,c}`) receive stronger influence.
- All clusters remain included (no hard state selection).
- Cluster support is preserved through `M_{k,c}`.
- The estimator remains linear in synthetic frames.
- Dominance is bounded via optional weight capping.


## 5.12 AQMH — Adaptive Quality Map Harvesting (Alternative Reconstruction Path)

> **Status:** Implementation extension (v0.3.0+). AQMH is the **default reconstruction method** in the C++ implementation. It replaces the classic tile-based reconstruction path (§5.5–§5.11) with a pixel-wise quality-map-driven approach. The classic path remains available via `method: classic_tile_compile`.

### 5.12.1 Motivation and Scope

AQMH (Adaptive Quality Map Harvesting) replaces the tile-grid-based local quality evaluation (§5.5) and tile-based overlap-add reconstruction (§5.7) with a **pixel-wise** quality assessment using Laplacian pyramid decomposition.

Key differences from the classic path:

| Aspect | Classic (§5.5–§5.11) | AQMH (§5.12) |
|--------|----------------------|--------------|
| Quality evaluation | Tile-grid-based, per (frame, tile) | Pixel-wise, multi-scale pyramid |
| Reconstruction unit | Tile with overlap-add (OLA) | Pixel-wise weighted mean |
| Clustering (§5.9) | Active in full mode | **Skipped** |
| Synthetic frames (§5.10) | Active in full mode | **Skipped** |
| Final stacking (§5.11) | Sigma-clip on synthetic frames | Pass-through of AQMH output |
| Phase display | `LOCAL_METRICS` | `AQMH_QUALITY_MAPS` |

### 5.12.2 Quality Map Computation

For each frame `f`, a quality map `Q_f(x, y)` is computed using a Laplacian pyramid with `S` scales (default `S = 4`).

**Per-pixel quality index:**

```
Q_f(x, y) = w_sharp · Q_sharp_f(x, y) + w_snr · Q_snr_f(x, y)
```

where:

- `Q_sharp_f` — normalized sharpness from Laplacian pyramid level responses
- `Q_snr_f` — normalized signal-to-noise ratio from local statistics
- `w_sharp` (default 0.6), `w_snr` (default 0.4) — configurable weights, must be non-negative with positive sum

**Artefact suppression:** Pixels whose local variance exceeds `k_artifact · MAD` are flagged as artefacts. Windows with more than `frac_artifact_max` artefact fraction are discarded entirely.

**Quality map storage:** Maps are cached to disk (`runs/<id>/cache/aqmh/`) with configurable resolution divisor (1/2/4) and data type (float32/uint16/uint8). A resident-memory cache (`max_resident_maps`) limits RAM usage during reconstruction.

### 5.12.3 Pixel-Wise Weighted Reconstruction

The reconstruction replaces the tile-based OLA (§5.7) with a direct pixel-wise weighted combination:

```
recon(x, y) = Σ_f W_f(x, y) · I_f(x, y) / Σ_f W_f(x, y)
```

where the per-pixel weight combines the AQMH quality map with the global frame weight:

```
W_f(x, y) = G_f · Q_f(x, y)
```

- `G_f` — global frame weight from §5.3 (unchanged)
- `Q_f(x, y)` — AQMH quality map value for frame `f` at pixel `(x, y)`

**Sigma-clip rejection** is applied at pixel level before the weighted mean, using the same `sigma_low`, `sigma_high`, `min_fraction` parameters as classic stacking.

### 5.12.4 Cherry-Pick Frame Selection (Optional)

When `aqmh.cherry_pick.enabled: true`, only the top `k` frames per pixel are used in the weighted mean:

```
k = max(k_min, floor(k_frac · N))
```

- `k_min` (default 3) — minimum frames per pixel, prevents underdetermination
- `k_frac` (default 0.30) — fraction of best frames to select

This implements a **per-pixel** quality-based selection that respects the "no frame selection" invariant (§1.2) — no entire frames are removed; selection is purely pixel-local.

### 5.12.5 Skipped Phases

When AQMH is active (`method: aqmh`):

- **§5.5 Local Tile Metrics** — replaced by quality map computation (§5.12.2)
- **§5.9 State-Based Clustering** — skipped (`reason: aqmh_independent_reconstruction`)
- **§5.10 Synthetic Frames** — skipped (`reason: aqmh_independent_reconstruction`)
- **§5.11 Final Linear Stacking** — pass-through; AQMH output is used directly

### 5.12.6 Conformance with Core Invariants

AQMH preserves the mandatory core invariants:

1. **Linearity** (§1.3): The weighted mean is linear in pixel values. Sigma-clipping is an auxiliary statistical nonlinearity, as permitted.
2. **No frame selection** (§1.2): All frames contribute to the quality maps. Cherry-pick selects per-pixel, not per-frame.
3. **Deterministic execution**: Quality map computation and weighted reconstruction are deterministic for identical inputs.
4. **Global weights preserved**: `G_f` from §5.3 is used unchanged.

### 5.12.7 Configuration Parameters

| Parameter | Type | Default | Range | Purpose |
|-----------|------|---------|-------|---------|
| `method` | string | `"aqmh"` | `aqmh`, `classic_tile_compile` | Selects reconstruction method |
| `aqmh.enabled` | bool | `true` | — | Auto-derived from `method` |
| `aqmh.pyramid.scales` | int | `4` | 1–8 | Laplacian pyramid levels |
| `aqmh.pyramid.base_window_px` | int | `16` | ≥1 | Base window size |
| `aqmh.pyramid.w_sharp` | float | `0.6` | ≥0 | Sharpness weight |
| `aqmh.pyramid.w_snr` | float | `0.4` | ≥0 | SNR weight |
| `aqmh.pyramid.k_artifact` | float | `3.0` | >0 | Artefact MAD multiplier |
| `aqmh.pyramid.frac_artifact_max` | float | `0.25` | (0,1] | Max artefact fraction |
| `aqmh.storage.resolution_divisor` | int | `2` | 1,2,4 | Map resolution divisor |
| `aqmh.storage.dtype` | string | `"float32"` | float32,uint16,uint8 | Map data type |
| `aqmh.storage.max_resident_maps` | int | `2` | 0–16 | Max resident maps in RAM |
| `aqmh.cherry_pick.enabled` | bool | `false` | — | Per-pixel frame selection |
| `aqmh.cherry_pick.k_min_required` | int | `20` | ≥1 | Run gate and min retained samples per pixel |
| `aqmh.cherry_pick.k_frac` | float | `0.30` | (0,1] | Best-frame fraction |
| `aqmh.diagnostics.tau_artifact` | float | `0.1` | — | Artefact diagnostic threshold |

### 5.12.8 Artifact Output

AQMH produces `aqmh_metrics.json` (in addition to `tile_reconstruction.json` with `"method": "aqmh"`):

- Per-frame quality map statistics (sharpness p50, SNR p50, scene-dependent SNR)
- Omitted pyramid scales
- Cache statistics (bytes read/written, hits/misses)
- Cherry-pick diagnostics (active fraction, mean/median k, k-heatmap)


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

1. **Canvas-invalid pixels:** pixels outside valid canvas support for the reconstructed image `I_rec`. Canvas-masked regions must never enter BGE calculations regardless of their pixel values. This is a binding constraint.
2. **Stars:** pixels in `M_star` (from star detection or segmentation), optionally dilated by `mask.star_dilate_px` (recommended default: 2–6 px).
3. **High-structure pixels:** pixels where `structure_metric(p) > structure_thresh`, where `structure_metric` is derived from local gradients (see definition below) and `structure_thresh` is configurable.
4. **Saturated pixels:** pixels with `I >= sat_level` and optionally a dilation margin `mask.sat_dilate_px`.
5. **Optional object mask:** if available (nebula/galaxy mask), exclude it to prevent bias in extended-object fields.

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

`w_t = exp(-lambda_structure * structure_score_t) * (1 - masked_fraction_t)`

where:
- `masked_fraction_t = (number of M_bg=0 pixels in tile t) / (total canvas-valid pixels in tile t)` — the fraction of canvas-valid pixels excluded by the background mask.
- `structure_score_t` is defined as the dimensionless relative high-pass energy:

  `structure_score_t = median_{p in M_bg_t} ( hp(R_{t,c}(p)) / max(sigma_{t,c}^{bg}, eps_sigma) )^2`

  where `hp(x)` is the high-pass residual after box-blur with radius `bge.structure_blur_px` (recommended default: 5 px), applied only to canvas-valid pixels within the tile, and `sigma_{t,c}^{bg}` is the local background noise scale on the same tile/channel. Preferred source: the local STRUCTURE noise proxy from §5.5.3 when available; deterministic fallback: a robust MAD-based estimate on the unmasked background pixels `M_bg_t`. If no `M_bg` pixels are available in the tile, `structure_score_t = +inf` and `w_t = 0`.

- `lambda_structure > 0` is a damping factor (recommended default: `lambda_structure = 2.0`).

This normalization is binding: a global multiplicative intensity rescaling of the same image content must not, by itself, change `structure_score_t` or `w_t` apart from floating-point noise.

This formula is binding and applies to both §6.3.2(c) and §6.3.4. **Canvas-invalid pixels must not contribute to `masked_fraction_t` or `structure_score_t`.**


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
- Weight: `w_cell = median({w_t})` (normative default)

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

Optional weights: use `w_t` as defined in §6.3.2(c) (binding formula: `exp(-lambda_structure * structure_score_t) * (1 - masked_fraction_t)`, with dimensionless `structure_score_t`). Canvas-invalid pixels must be excluded from both `structure_score_t` and `masked_fraction_t` as specified there.

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

For a given channel, define a deterministic brightness-normalized objective:

`B_ref = max(abs(median_i b_i), eps_bg)`

`J = E_cv / B_ref + alpha_f * E_flat / B_ref^2 + beta_r * E_rough / B_ref`

with:
- `E_cv`: holdout RMS of background sample residuals evaluated on a deterministic validation split,
- `E_flat`: large-scale gradient energy of the fitted background model,
- `E_rough`: second-derivative energy of the fitted model (penalizes overfit waviness).
- `B_ref`: robust background amplitude from the same candidate's grid-cell values `{b_i}`.
- `eps_bg > 0`: zero-safe normalization floor (recommended default: `1e-12` in normalized data units).

**Notation:** `alpha_f` (flatness weight) and `beta_r` (roughness weight) are distinct from the global quality index weights `alpha`, `beta`, `gamma` (§5.3.2).

All terms must be computed deterministically from the same grid-cell set. This normalization is binding: a global multiplicative rescaling of the input intensities must not change candidate ranking except for negligible floating-point differences.

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
- `bge.autotune.alpha_f` (flatness weight, corresponds to `alpha_f` in objective)
- `bge.autotune.beta_r` (roughness weight, corresponds to `beta_r` in objective)
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
- `autotune.best.objective` (binding normalized objective `J`)
- `autotune.best.objective_raw` (recommended auxiliary diagnostic)
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

`rho(r) = 0.5 * r^2                        if |r| <= delta`
`rho(r) = delta * (|r| - 0.5 * delta)      otherwise`

(Equivalently: `delta * |r| - 0.5 * delta^2` for the linear branch.)

or Tukey biweight loss.

The fit must be solved via Iteratively Reweighted Least Squares (IRLS) or equivalent deterministic robust optimization.

Thin-plate spline alternative:

`B_c = argmin_B sum_i w_i (b_i - B(x_i,y_i))^2 + lambda * J_TPS(B)`

where the standard 2D TPS bending energy is:

`J_TPS(B) = integral [ (d^2B/dx^2)^2 + 2*(d^2B/dxdy)^2 + (d^2B/dy^2)^2 ] dx dy`

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

Implementations must guarantee that grid resolution is coarser than tile resolution (`G >= 2*T`) **except** in compact-tile mode.

**Compact-tile mode exception (binding):** When compact-tile mode is active (§5.4 step 4, `T = min(W,H)`), the constraint `G >= 2*T` cannot be satisfied because `G_max = min(W,H)/4 < 2*T`. In this case, BGE **must** be disabled (`bge.enabled = false` implicitly) or the grid spacing is fixed at `G = G_max` with an explicit diagnostic warning. Implementations must document which behavior is active and must not silently violate the `G >= 2*T` constraint.

---

#### 6.3.10 RBF Surface Mathematical Specification (Binding, when `bge.fit.method = rbf`)

Let grid-cell samples be `(r_j, b_j, w_j)` for `j = 1..J`, where:

- `r_j = (x_j, y_j)` are grid cell centers
- `b_j` is the aggregated background value
- `w_j >= 0` is the cell reliability weight (as defined in §6.3.2(c))

Define the RBF surface with affine trend term:

`B_c(r) = sum_{i=1..J} u_i * phi(||r - r_i||; mu) + a0 + a1*x + a2*y`

where:

- `u_i` are RBF coefficients (unknown)
- `(a0, a1, a2)` is an affine term (recommended; improves extrapolation stability)
- `mu > 0` is the kernel shape/scale parameter
- `phi` is the chosen radial kernel

##### Supported Kernels (Binding)

1. Multiquadric:

   `phi(d; mu) = sqrt(d^2 + mu^2)`

2. Thin-plate spline kernel:

   `phi(d) = d^2 * log(d + eps_rbf)`

   with small `eps_rbf > 0` for numerical stability (recommended: `eps_rbf = 1e-6 * G`).

3. Gaussian:

   `phi(d; mu) = exp(-d^2 / (2 * mu^2))`

   For Gaussian, `mu` acts as bandwidth.

##### Robust Regularized Fit (Binding)

Solve for parameters `theta = (u, a)` by minimizing:

`argmin_theta sum_{j=1..J} w_j * rho(b_j - B_c(r_j)) + lambda * ||u||_2^2`

where:

- `rho` is the configured robust loss (Huber or Tukey, as defined in §6.3.8)
- `lambda > 0` is mandatory regularization when using RBF
- Optimization must be deterministic (IRLS or equivalent).

**RBF without regularization (`lambda = 0`) is forbidden.**

Canvas-invalid pixels must not contribute to the sample set `(r_j, b_j, w_j)` used for fitting.

##### Recommended Defaults

- `mu = G` (grid spacing)
- `lambda = 1e-4` (tune within `[1e-6, 1e-2]` depending on gradient strength)
- Include affine term by default.

---

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

#### 6.4.4 Optional Post-PCC Isolated Chroma-Speckle Suppression

After successful PCC, implementations may apply a deterministic cleanup step for isolated chromatic speckles before writing the final RGB outputs.

Binding conditions for this optional step:

1. The step operates only in the explicit RGB domain after PCC.
2. It must be restricted to pixels inside valid canvas support.
3. A correction is permitted only when exactly one channel is an isolated local outlier relative to neighboring valid RGB samples, or an equivalent deterministic single-channel chroma-outlier condition is satisfied.
4. Bright stellar cores, compact real structures, and supported multi-pixel chromatic image content must be protected by deterministic luma and/or structure guards.
5. Replacement values must be derived from robust local neighborhood statistics of the affected channel, or from an equivalent deterministic local estimator.
6. This step belongs to optional post-processing and is not part of the mandatory linear quality core.

---

## 7. Validation and Abort

### 7.1 Success Criteria

- FWHM improvement over the reference stack according to `validation.min_fwhm_improvement_percent`
- Background RMS not worse than reference
- No systematic tile seams
- Stable stellar flux ratios and photometric scaling
- Stable weight distributions

### 7.2 Abort Criteria

- Data integrity violated (nonlinear, unreadable, inconsistent)
- Registration failure across large portions of the dataset
- Numerical instability despite fallbacks

### 7.3 Minimum Tests (Normative)

1. `alpha+beta+gamma=1` (global quality index weights, §5.3.2)
2. clamping before `exp` (global and local weights)
3. additive background subtraction and multiplicative photometric scaling remain separate
4. valid negative or zero pixels remain admissible samples
5. tile monotonicity in `F`, including deterministic compact-tile fallback for small images
6. overlap consistency (`0<=o<=0.5`, explicit `o_clipped=clip(o,0,0.5)`, integer `O,S`)
7. low-weight fallback without NaN/Inf
8. sigma-clipping guards on `N_eff` and `D_eff`; `min_fraction` keep-floor active
9. soft local-score blending remains continuous around the star-support threshold
10. support-aware OLA covers all valid boundary pixels and satisfies partition-of-unity tolerance with `M_{t,c}(p)` as defined in §5.7.1
11. no channel coupling
12. no quality-based frame selection
13. deterministic reproducibility
14. registration cascade including identity fallback; reference-frame unconditional identity acceptance
15. CFA phase preservation
16. cluster aggregation is mass-preserving (`M_{k,c}` included) with optional dominance cap
17. WCS round-trip error below `validation.wcs_roundtrip_max_arcsec` (recommended default: `0.5` arcsec) only when WCS is enabled
18. PCC stability: positive determinant, condition number below `validation.pcc_max_condition_number` (recommended default: `1000`), residuals below `validation.pcc_max_residual_mag` (recommended default: `0.05` mag) only when PCC is enabled
19. STAR metric coefficient sum: `0.6 + 0.2 + 0.2 = 1.0` (§5.5.2)
20. STRUCTURE metric coefficient sum (unsigned): `0.7 + 0.3 = 1.0` (§5.5.3)
21. canvas-invalid pixels contribute zero to OLA accumulators `A` and `S` (§5.7.1), zero to local metrics (§5.5), and zero to BGE sampling (§6.3.2)
22. clustering threshold matches mode framework: active iff `N >= max(N_red, 50)` (§5.9)
23. BGE tile reliability weights remain stable under global multiplicative intensity rescaling when the local noise scale is rescaled consistently (§6.3.2(c))
24. BGE autotune candidate ranking uses the normalized objective `J`; `autotune.best.objective` reports that normalized value (§6.3.7.1)

Note: The legacy PCC test "no negative matrix element" is **no longer** required as a hard criterion in v3.3+.

---

## 8. Recommended Numerical Defaults

- `eps_bg = 1e-6`
- `eps_scale = 1e-6`
- `eps_mad = 1e-6`
- `eps_weight = 1e-6`
- `eps_neff = 1e-6`
- `eps_var = 1e-12`
- `tol_ola = 1e-6`
- `eps_affinity = 1e-6`
- `delta_ncc = 0.01`
- `Q` clamp global/local: `[-3, +3]`
- `k_global = 1.0`
- `k_local = 1.0`
- `min_fraction = 0.5` (sigma-clip keep-floor)
- `lambda_bge = 1.0` (BGE structure score damping)
- `bge.structure_blur_px = 5`
- `validation.wcs_roundtrip_max_arcsec = 0.5`
- `validation.pcc_max_condition_number = 1000`
- `validation.pcc_max_residual_mag = 0.05`

---

## 9. Scope Boundary: Mandatory Core vs Extension

### Mandatory Core

- CFA-based registration path up to explicit or deferred (shared-core-variant-dependent) channel separation
- global normalization
- global/local metrics and weights
- tile reconstruction including consolidated fallbacks
- clustering/synthesis/final stack depending on operating mode

### Optional Extension

- deterministic CFA defect-pixel suppression / cosmetic correction
- soft-threshold / Wiener
- alternative sigma-clipping strategies
- WCS/PCC
- post-PCC isolated chroma-speckle suppression
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

`Ŵ_{f,t,c}(p) = Ĝ_{f,c} * L̂_{f,t,c} * M̂_{f,t,c}(p)`

The tile reconstruction remains a weighted mean:

`R_{t,c}(p) = sum_f Ŵ_{f,t,c}(p) * I_{f,c}(p) / sum_f Ŵ_{f,t,c}(p)`

Fallback rule remains unchanged: if denominator < `eps_weight`, fall back to the unweighted mean.

**ML mask and sigma-clipping integration (binding):**

1. **Valid sample set:** The valid sample set `V_{t,c}(p)` (§5.7) remains defined by the canvas/finiteness criterion. The soft mask `M̂_{f,t,c}(p) in [m_min, 1]` does NOT change membership in `V_{t,c}(p)`. Instead it scales the frame's effective weight at pixel level.

2. **Sigma-clipping with ML weights:** When sigma-clipping is active (§5.7), the per-iteration weighted statistics must use the pixel-level weight `Ŵ_{f,t,c}(p)` instead of the tile-level weight `w_{f,t,c}`. All other sigma-clipping semantics (keep-floor `min_fraction`, fallback to unclipped mean, `N_eff` guard, `D_eff` guard) apply unchanged.

3. **Keep-floor with soft masks:** The keep-floor `ceil(min_fraction * |V_{t,c}(p)|)` counts all frames in `V_{t,c}(p)` regardless of their soft mask value. This prevents near-zero mask values from trivially shrinking the active set.

4. **Canvas invariant:** The canvas exclusion defined in §5.7.1 applies unconditionally. `M̂_{f,t,c}(p) > 0` for a canvas-invalid pixel does not make that pixel a valid sample. Canvas invalidity overrides any ML mask value.

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


*The RBF surface mathematical specification has been relocated to **§6.3.10** within the BGE section where it belongs.*

---

## 10. Core Statement

The method replaces rigid search for "best frames" with robust spatio-temporal quality modeling, uses all frames without quality-based selection, and reconstructs signal where it is physically and statistically most reliable.
