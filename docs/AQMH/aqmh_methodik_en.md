# Adaptive Quality Mask Harvesting (AQMH) — Methodology v0.1.0

**Status:** Exploratory specification — not yet normative  
**Version:** v0.1.0 (2026-06-05)  
**Last revised:** 2026-06-05  
**Relation to core:** Proposed extension to the mandatory core of v3.3.9

---

## 0. Motivation and Objective

### 0.1 Limitation of the Fixed Tile Grid

The mandatory reconstruction core (v3.3.9, §5.4–§5.7) partitions every registered frame into a **regular rectangular tile grid**. For each tile `t` and frame `f`, a single scalar quality score `Q_{f,t,c}^{local}` and a single weight `L_{f,t,c}` are assigned to the entire tile.

This design has two well-understood structural limitations:

1. **Intra-tile heterogeneity:** A single satellite trail, cloud edge, or hot-pixel cluster that covers only 5% of a tile degrades the quality score for the entire tile, discarding the remaining 95% of good data at full weight.

2. **Tile boundary artifacts:** Because the quality weight is constant within a tile, reconstruction quality can jump discontinuously at tile boundaries if neighboring tiles receive significantly different weights, even with OLA windowing.

### 0.2 AQMH Objective

The Adaptive Quality Mask Harvesting method replaces the per-tile scalar weight with a **continuous, pixel-resolved quality weight field** `Q_map_{f,c}(x,y)` computed for every frame. The reconstruction then performs a **pixel-wise weighted mean** instead of a tile-wise weighted mean.

Core objectives:

1. Extract the usable fraction of every frame at pixel resolution — not at tile granularity.
2. Eliminate tile boundary weight discontinuities from the final stack.
3. Model quality heterogeneity within tiles via a multi-scale analysis pyramid.
4. Remain compatible with the deterministic weighted-mean reconstruction invariants of the v3.3.9 mandatory core in default AQMH mode.
5. Degrade gracefully to the existing tile-level method when the dense map provides no additional information.

### 0.3 Relation to the Mandatory Core

AQMH is an **optional extension** in the sense of §9 of v3.3.9. It does not replace:

- Registration (phases 1–2)
- Global normalization (phase 4)
- Global metrics and weights `G_{f,c}` (phase 5)
- Tile geometry definition (phase 6, still used as a spatial unit for metric computation)
- Common-overlap mask definition (phase 7)
- State-based clustering and synthetic frames (phases 10–11)

AQMH extends **phase 8 (local metrics)** and **phase 9 (tile reconstruction)** only. The output of AQMH — a pixel-resolved weight field per frame — feeds into a modified reconstruction accumulator that replaces the constant `L_{f,t,c}` but preserves the support-aware reconstruction invariants.

---

## 1. Principles and Definitions

### 1.1 Physical Objective

The method models per-pixel observational quality as the product of two separable components:

- **Frame-level quality:** the global atmospheric state of frame `f`, captured by `G_{f,c}` (unchanged from v3.3.9 §5.3).
- **Spatial quality field:** the continuous quality distribution within the frame, captured by `Q_map_{f,c}(x,y)`.

The effective pixel weight is:

`W_{f,c}^{dense}(x,y) = G_{f,c} * Q_map_{f,c}(x,y)`

### 1.2 Invariants (Binding)

The following invariants from v3.3.9 remain binding for default AQMH reconstruction modes (`dense_map`, `tile`, and `hybrid`). The optional cherry-pick mode is an explicit opt-in deviation and is governed separately by §5.3.

1. **No frame selection:** Entire frames must not be removed based on quality.
2. **Conditional photometric linearity:** Once the deterministic weights have been computed, the final reconstruction remains `R(p) = sum_f w_f(p) * I_f(p) / sum_f w_f(p)` with `w_f(p) >= 0`. AQMH must not apply nonlinear intensity transforms to the samples entering the accumulator.
3. **Determinism:** All quality map computations must be deterministic and reproducible.
4. **Canvas exclusion:** Canvas-invalid pixels contribute zero to all accumulators regardless of their `Q_map` value.
5. **No hallucination:** AQMH outputs are weights and masks only. It does not generate or predict pixel intensities.

### 1.3 Notation

- `f` frame index
- `c` channel index
- `(x, y)` pixel coordinates in the registered canvas
- `t` tile index (retained from v3.3.9 for spatial reference)
- `Q_map_{f,c}(x,y)` per-pixel quality field, `∈ [0, 1]`
- `D_s` downscale factor at pyramid scale `s`
- `P` configured maximum number of pyramid scales
- `S_actual` ordered set of pyramid scales actually computed after the small-image omission rule
- `P_actual = |S_actual|` actual number of computed pyramid scales used for fusion
- `R_s` spatial radius of the local analysis window at scale `s`
- `Psi_s(x,y)` quality contribution at scale `s`
- `W_{f,c}^{dense}(x,y)` effective dense pixel weight
- `B_s(x,y; R)` masked local background operator at scale `s`

### 1.4 Deterministic Statistics Convention

All medians, MADs, and quantiles in AQMH are computed over finite values only and over the explicitly stated valid support (for example `W_s_valid` or canvas-valid pixels). If the support is empty, the statistic is invalid and the fallback rules in §2.3.4 apply.

For deterministic reproducibility, sort finite values in ascending numeric order. The median is the middle value for odd sample counts and the arithmetic mean of the two middle values for even sample counts. MAD uses the same median convention on `|x - median(x)|`. Quantiles use linear interpolation between sorted samples with index `q * (n - 1)`, clamped to `[0, n-1]`; for `n = 1`, the only sample is returned.

---

## 2. Dense Quality Map Computation

### 2.1 Overview

For each frame `f` and channel `c`, a dense quality map `Q_map_{f,c}` is computed from the prewarped, normalized frame `I_{f,c}`. The map is computed in a **multi-scale pyramid** with `P` scales. Each per-scale quality contribution is upscaled to canvas resolution, then the upscaled contributions are fused via geometric mean.

### 2.2 Input Data

The input to AQMH is the normalized frame `I_{f,c}(x,y)` as defined in v3.3.9 §5.2. The common-overlap canvas mask from §5.5 (binding) applies: canvas-invalid pixels are excluded from all quality map accumulators.

### 2.3 Multi-Scale Pyramid

#### 2.3.1 Scale Definition

Define `P` analysis scales with downscale factors `D_s` and window radii `R_s`:

| Scale `s` | Downscale `D_s` | Window `R_s` | Captured structure |
|---|---|---|---|
| 0 | 1  | 4 px  | sub-tile, pixel-near defects, hot pixels |
| 1 | 4  | 4 px  | tile-comparable (≈ 16 px window) |
| 2 | 16 | 4 px  | coarse regions (≈ 64 px window) |
| 3 | 64 | 4 px  | global frame quality context (≈ 256 px window) |

Normative defaults: `P = 4`, `D_s = 4^s`, `R_s = 4` (in downscaled pixels).

A scale `s` is **omitted** when `D_s > min(W, H) / 16`; the configured maximum `P` remains unchanged, while `S_actual` and `P_actual` are reduced accordingly. Equivalently, scale `s` requires `min(W, H) >= 16 * D_s`. Worked examples for the normative defaults: scale 1 (`D=4`) requires `min(W,H) >= 64`; scale 2 (`D=16`) requires `>= 256`; scale 3 (`D=64`) requires `>= 1024`. Thus small images automatically drop the coarsest scales (e.g. a 512 px image keeps only scales 0–2).

#### 2.3.2 Per-Scale Signal Computation

At scale `s`, compute a downscaled version of the input:

`I_s(x,y) = downsample(I_{f,c}, D_s)`

using area-averaging with canvas-mask-aware denominator (canvas-invalid pixels excluded from the area mean, not replaced by zero).

For each pixel `(x,y)` in the downscaled domain, compute the following **three quality signals** over a local window `W_s(x,y)` of radius `R_s`:

##### (a) Local Sharpness Signal `Phi_sharp`

`Phi_sharp_s(x,y) = Var_{p in W_s_valid(x,y)}(Lap(I_s)(p))`

where `Lap` is the Laplacian response and `Var` is the local variance of finite, valid Laplacian values. Result is clamped to `[0, +inf)` (non-negative).

No explicit global rescaling of `Phi_sharp_s` is applied here. The per-scale robust z-score in §2.3.3 is invariant to any global multiplicative scaling (`z(a*Phi) = z(Phi)` for `a > 0`), so a `sigma_Lap`-based normalization at this step would have no effect on `Psi_s` and is intentionally omitted. Implementations that need a well-conditioned intermediate may rescale `Phi_sharp_s` locally for numerical reasons, but must not rely on it changing the result.

##### (b) Local SNR Signal `Phi_snr`

`b_s(x,y) = B_s(x,y; R_s) = median_{p in W_s_valid(x,y)} I_s(p)`  
`mu_s(x,y) = mean_{p in W_s_valid(x,y)}(max(I_s(p) - b_s(x,y), 0))`  
`sigma_s(x,y) = MAD_{p in W_s_valid(x,y)}(I_s(p)) * 1.4826`

`Phi_snr_s(x,y) = mu_s(x,y) / max(sigma_s(x,y), eps_aqmh)`

The clamp is applied only for the mean term; MAD uses the raw signed residuals to preserve noise scale estimation accuracy. `B_s` is a deterministic masked median over the same valid local support `W_s_valid(x,y)` used by the other local statistics. If `W_s_valid(x,y)` is empty, the signal is invalid under §2.3.4. If fewer than three valid pixels are available, implementations may fall back to `mean(max(I_s(p), 0))`, but must set the diagnostic flag `scene_dependent_snr = true`.

Scene-dependence guard: `Phi_snr_s` is a local support-quality proxy, not a source detector. The background-centered definition is intended to reduce source-content bias; fallback to the non-centered mean is allowed only as a diagnostic-marked degraded path.

##### (c) Artifact Anomaly Score `Phi_artifact`

`Phi_artifact_s(x,y)` detects local outlier gradients that indicate satellite trails, cosmic rays, or cloud edges:

1. Compute the high-pass residual `hp_s(x,y) = I_s(x,y) - blur(I_s, R_s)(x,y)`, where `blur(I_s, R_s)(x,y)` is the masked local mean of `I_s` over `W_s_valid(x,y)`.
2. Compute the local robust scale `tau_s(x,y) = max(1.4826 * MAD_{p in W_s_valid(x,y)}(hp_s(p)), eps_aqmh)`.
3. Compute the local outlier fraction: `frac_out_s(x,y) = |{p in W_s_valid(x,y) : |hp_s(p)| > k_artifact * tau_s(x,y)}| / |W_s_valid(x,y)|`  
   with normative default `k_artifact = 5.0`.
4. `Phi_artifact_s(x,y) = 1 - clip(frac_out_s(x,y) / frac_artifact_max, 0, 1)`  
   with normative default `frac_artifact_max = 0.10`.

`Phi_artifact_s = 1` indicates a clean region; `Phi_artifact_s = 0` indicates that at least `frac_artifact_max` (default 10%) of pixels in the window are outliers.

#### 2.3.3 Per-Scale Quality Map

The per-scale quality map `Psi_s(x,y)` is defined as:

`Psi_s(x,y) = sigmoid(w_sharp * z(Phi_sharp_s) + w_snr * z(Phi_snr_s)) * Phi_artifact_s(x,y)`

where:
- `z(Phi)` is the robust z-score normalization: `z(Phi)(x,y) = (Phi(x,y) - median(Phi_s)) / max(1.4826 * MAD(Phi_s), eps_aqmh)`
  applied over all finite, canvas-valid pixels at scale `s`
- `sigmoid(v) = 1 / (1 + exp(-v))`
- `w_sharp`, `w_snr` are configurable weights (normative defaults: `w_sharp = 0.6`, `w_snr = 0.4`)
- The artifact term `Phi_artifact_s` acts as a multiplicative gate: it suppresses any region with excessive outlier density regardless of its sharpness or SNR.

Binding constraint: `Psi_s(x,y) ∈ [0, 1]` for all finite inputs. The sigmoid term is strictly positive, but the multiplicative artifact gate may set `Psi_s` to exactly zero.

**Within-frame relativity (binding clarification):** Because `z(Phi_sharp_s)` and `z(Phi_snr_s)` are normalized per frame (median/MAD computed over that frame's pixels at scale `s`), the **sigmoid factor of `Q_map` is a within-frame relative quality field**, not an absolute cross-frame quality measurement. Two frames of differing global seeing produce similarly distributed sigmoid factors, each self-normalized. Consequently, in the dense reconstruction weight `W_{f,c}^{dense} = G_{f,c} * Q_map_{f,c}` (§1.1, §4.3):

- **Between-frame** discrimination at a given pixel is carried by the global weight `G_{f,c}` and by the **absolute** artifact gate `Phi_artifact_s` (which is not z-scored and can drive `Q_map` toward zero in any frame independently of other frames).
- **Within-frame** spatial discrimination (which regions of a single frame are sharper / cleaner) is carried by the sigmoid factor.

This division of labor is intentional and binding: the sigmoid factor must not be interpreted as an absolute photometric quality and must not be used to rank whole frames.

### 2.3.4 Boundary and Empty-Window Rules

All local statistics are computed over finite, canvas-valid pixels only. Let `W_s_valid(x,y)` be the valid subset of the analysis window.

If `|W_s_valid(x,y)| = 0`, all per-scale signals at `(x,y)` are marked invalid and the fused `Q_map` value is later set to zero by the canvas guard. If `|W_s_valid(x,y)| > 0` but fewer than three valid pixels are available, robust scale estimates fall back to `eps_aqmh` and local variance estimates fall back to zero. For the `Phi_snr_s` signal, the background-centering fallback rule in §2.3.2(b) is more specific and takes precedence.

The default boundary mode for convolution-like operations (`Lap`, `blur`, and local windows) is valid-only masked evaluation. Implementations must not mirror or replicate canvas-invalid pixels into the statistic.

### 2.4 Multi-Scale Fusion

Let `S_actual` be the ordered set of scales that are actually computed after applying the omission rule in §2.3.1, and let `P_actual = |S_actual|`. Upsample each computed `Psi_s` to the full canvas resolution using bilinear interpolation:

`Psi_s^{up}(x,y) = upsample(Psi_s, D_s)`

Fuse via **geometric mean** over the `P_actual` computed scales:

`Q_map_{f,c}(x,y) = ( prod_{s in S_actual} Psi_s^{up}(x,y) )^{1/P_actual}`

The configured `P` is an upper bound. It must not be used as the exponent denominator when one or more scales are omitted. If `P_actual = 0`, `Q_map` is defined as zero everywhere after the canvas guard.

Geometric mean is chosen over arithmetic mean because it requires **all scales to agree** on high quality. A single scale signaling an artifact suppresses the fused map regardless of other scales. This implements a conservative "all-clear" fusion philosophy.

**Zero-scale guard:** If `Psi_s^{up}(x,y) = 0` for any scale `s`, then `Q_map_{f,c}(x,y) = 0` exactly (one bad scale vetos the pixel).

**Canvas guard (binding):** For all canvas-invalid pixels `p`, set `Q_map_{f,c}(p) = 0` unconditionally after fusion, overriding any computed value.

### 2.5 Relationship to Tile-Level Quality Scores

For compatibility and diagnostics, the per-tile quality score `Q_{f,t,c}^{aqmh}` is derived as the spatial median of `Q_map_{f,c}` over the tile `t`:

`Q_{f,t,c}^{aqmh} = median_{p in t, canvas-valid} Q_map_{f,c}(p)`

This value is reported in diagnostics alongside the v3.3.9 tile score `Q_{f,t,c}^{local}` for comparison. It does **not** replace the v3.3.9 score in the mandatory core.

---

## 3. Quality Map Storage and Memory Model

### 3.1 Storage Format

Conceptually, `Q_map_{f,c}` is a full-canvas quality field, one per frame and channel. The persisted representation may be lower resolution or quantized according to §3.2. For multi-channel CFA inputs processed via the CFA-proxy-equivalent core variant, a single luminance-channel map may be used instead of per-channel maps (configurable).

Recommended storage: on the existing `DiskCacheFrameStore` or a parallel quality-map disk cache with identical indexing semantics.

### 3.2 Memory Budget

At full resolution, one map requires `W * H * 4` bytes. For a 24 Mpx sensor with 300 frames and 3 channels, the total budget is approximately `24e6 * 4 * 300 * 3 ≈ 86 GB`.

Therefore, the following compression strategies are supported:

| Strategy | Description | Normative? |
|---|---|---|
| Full resolution float32 | No compression | Optional |
| 1/4 area float32 | Downscale by 2 in each axis | **Default** |
| uint8 quantization | Map scaled to `[0, 255]` | Optional |
| Tile-compressed float16 | Per-tile float16 sub-blocks | Future optional; only valid when explicitly implemented |

**Normative default:** Store `Q_map` at `1/4` area resolution (factor-2 downscale in each axis, `resolution_divisor = 2`). The map is upscaled to full resolution on demand during reconstruction via bilinear interpolation.

### 3.3 Disk Cache Lifecycle

Quality maps are written during phase 8.AQMH (see pipeline overview §4) and consumed during phase 9.AQMH. Maps are invalidated when the source prewarped frame is invalidated, when the common-overlap mask changes, or when any **map-affecting** AQMH configuration changes (`pyramid`, `storage`, or map format version). Reconstruction-only settings such as `reconstruction.mode` and `fallback_to_tile` must not invalidate map cache entries. Implementations should store and validate a cache metadata hash covering only map-affecting inputs.

---

## 4. Pipeline Integration

### 4.1 Modified Pipeline Phases

The AQMH extension modifies the pipeline in two phases:

```
Phase 8:   LOCAL_METRICS
  8.a  [AQMH] Dense quality map computation     ← new
  8.b  [v3.3.9] Tile metrics, z-scores, reg.    ← unchanged

Phase 9:   TILE_RECONSTRUCTION
  9.a  [AQMH] Pixel-wise weighted stacking      ← replaces constant-weight tile reconstruction
  9.b  [v3.3.9] Support-aware OLA               ← unchanged (windowing, accumulators)
```

All other phases (0–7 and 10–18) are unchanged.

### 4.2 Phase 8.a: Dense Map Computation

For each frame `f` and channel `c` with `frame_has_data[f] = true`:

1. Load the prewarped frame `I_{f,c}` from `DiskCacheFrameStore`.
2. Apply the common-overlap canvas mask: set pixels outside mask to NaN (excluded from all window statistics).
3. For `s = 0, ..., P-1`:
   a. Compute `I_s` via area-averaged downscaling with mask-aware denominator.
   b. Compute `Phi_sharp_s`, `Phi_snr_s`, `Phi_artifact_s` (§2.3.2).
   c. Compute `Psi_s` (§2.3.3).
4. Upsample all `Psi_s` to canvas resolution (§2.4).
5. Compute fused `Q_map_{f,c}` via geometric mean (§2.4).
6. Apply canvas guard: set canvas-invalid pixels to zero.
7. Write `Q_map_{f,c}` to quality-map disk cache (at configured storage resolution).

### 4.3 Phase 9.a: Pixel-Wise Weighted Reconstruction

For each tile `t` and pixel `p` in the canvas-valid support:

Define the finite intensity sample set:

`V_{t,c}^{I}(p) = { f | I_{f,c}(p) is finite AND canvas-valid }`

Define the map-available sample set:

`V_{t,c}^{map}(p) = { f in V_{t,c}^{I}(p) | Q_map_{f,c}(p) is finite }`

For each `f in V_{t,c}^{I}(p)`, the effective pixel weight in `dense_map` mode is:

`w_{f,c}^{dense}(p) = G_{f,c} * Q_map_{f,c}(p)` when `f in V_{t,c}^{map}(p)`

`w_{f,c}^{dense}(p) = G_{f,c} * L_{f,t,c}` when the map is unavailable and `fallback_to_tile = true`

`w_{f,c}^{dense}(p) = 0` when the map is unavailable and `fallback_to_tile = false`

The reconstructed pixel value is:

`R_{t,c}^{dense}(p) = sum_{f in V^I} w_{f,c}^{dense}(p) * I_{f,c}(p) / sum_{f in V^I} w_{f,c}^{dense}(p)`

**Weight fallback (binding):** Before applying the `eps_weight` fallback, implementations must distinguish finite map samples from unavailable map samples. A finite zero is a valid map sample and is an explicit veto, not a missing value. If `sum_f max(w_{f,c}^{dense}(p), 0) <= eps_weight`, fallback behavior depends on why the sum is zero:

1. If at least one finite map sample exists at `p` (`V_{t,c}^{map}(p) != empty`) and all dense weights are zero because the available maps explicitly veto the pixel (`Q_map = 0`), the output pixel is marked unsupported/zero for that reconstruction tile. Do **not** replace the explicit zero-veto by an unweighted mean.
2. If no finite map sample exists at `p`, or all nonzero weights are unavailable because of IO/cache failure, replace all weights over `V_{t,c}^{I}(p)` by 1 and fall back to the unweighted valid mean (identical semantics to v3.3.9 §5.7 tile-level fallback).
3. If sigma clipping removes all samples after a nonzero pre-clip weight sum, use the existing v3.3.9 sigma-clipping keep-floor and fallback semantics.

**Sigma clipping:** Iterative weighted sigma clipping (v3.3.9 §5.7) applies with `w_{f,c}^{dense}(p)` in place of `w_{f,t,c}`. The keep-floor `min_fraction` and `N_eff` / `D_eff` guards are unchanged.

### 4.4 Map-Unavailability Fallback (All Modes)

When AQMH is enabled but the quality map for a given frame/channel is unavailable (IO error, invalid cache entry, frame skipped), reconstruction falls back to the v3.3.9 tile-level weight for that frame/channel/tile, **independently of the reconstruction mode**:

`w_{f,c}(p) = G_{f,c} * L_{f,t,c}   (AQMH map unavailable)`

This ensures AQMH never blocks reconstruction.

Partial maps are handled at pixel granularity: if a map is available but `Q_map_{f,c}(p)` is non-finite for a canvas-valid pixel, that sample uses the tile-level fallback weight at `p`; finite map values continue to use dense weights. A finite zero is not “unavailable”; it is an explicit veto.

### 4.5 Reconstruction Modes (Binding)

The three default (non-cherry-pick) reconstruction modes differ only in how the per-pixel weight is formed from `Q_map` and the tile weight `L_{f,t,c}`. In all modes the global weight `G_{f,c}` is applied and the unavailability fallback of §4.4 holds.

| Mode | Per-pixel weight (map available and finite) | Purpose |
|---|---|---|
| `tile` | `w_{f,c}(p) = G_{f,c} * L_{f,t,c}` | v3.3.9-equivalent; ignores `Q_map` (useful for A/B baselines while still emitting AQMH diagnostics) |
| `dense_map` | `w_{f,c}(p) = G_{f,c} * Q_map_{f,c}(p)` | full pixel-resolved weighting |
| `hybrid` | `w_{f,c}(p) = G_{f,c} * max(Q_map_{f,c}(p), L_{f,t,c})` | diagnostic/conservative mode; tile weight as a **lower bound** on the dense weight |

**Distinctness (binding):** `hybrid` is **not** equivalent to `dense_map` with `fallback_to_tile = true`. In `dense_map`, the tile weight is used *only* when the map sample is missing/non-finite; where a finite map sample exists it fully replaces the tile weight (and may suppress a pixel toward zero). In `hybrid`, a finite map sample never drops the weight below the tile baseline `G_{f,c} * L_{f,t,c}`; the `max` guarantees that AQMH can only *raise* confidence in already tile-trusted regions, never veto them below the tile level.

**Artifact caveat (binding):** Because `hybrid` prevents dense-map down-weighting below the tile baseline, it does **not** provide AQMH artifact veto semantics. It is a conservative diagnostic/A-B mode for early validation, not the default artifact-suppression mode. Artifact-sensitive runs must use `dense_map`.

---

## 5. Adaptive Region Extraction (Optional)

### 5.1 Motivation

In addition to the continuous weight map, AQMH can generate **binary quality regions** for diagnostic reporting and for the optional cherry-pick stacking mode.

### 5.2 Quality Contour Extraction

From the fused `Q_map_{f,c}`, extract binary regions by thresholding:

1. Compute the per-frame threshold: `tau_f = quantile(Q_map_{f,c}, q_region)` over finite, canvas-valid pixels only, with normative default `q_region = 0.75`.
2. Binary mask: `M_f(x,y) = 1 iff Q_map_{f,c}(x,y) >= tau_f AND canvas-valid`.
3. Apply morphological opening with radius `r_morph_canvas_px` to remove isolated noise regions. The normative default is a **canvas-equivalent radius of 6 px**, which corresponds to 3 px at the default `resolution_divisor = 2`. If region extraction is run on a stored/downscaled map, use `r_morph_map = max(1, round(r_morph_canvas_px / resolution_divisor))`.
4. Extract connected components; label each component with:
   - `Area_r`: pixel count
   - `MeanQ_r`: mean quality score over the region
   - `Compactness_r = 4*pi*Area_r / Perimeter_r^2` (Polsby-Popper score)
5. Rank regions by `Score_r = MeanQ_r * log(1 + Area_r)`.

These regions are reported in the diagnostic artifact `aqmh_regions.json` per frame.

### 5.3 Optional Cherry-Pick Stacking Mode

When `aqmh.cherry_pick.enabled = true`, per-pixel stacking uses only the **top-K frames** by quality, rather than all frames:

`K(p) = min(N_valid(p), max(k_min, floor(k_frac * N_valid(p))))`

where `N_valid(p) = |V_{t,c}^{I}(p)|` is the number of finite intensity samples at pixel `p` (the intensity sample set of §4.3), and with normative defaults `k_min = 3`, `k_frac = 0.3`.

For each pixel `p`, sort `V_{t,c}^{I}(p)` by the cross-frame calibrated score `S_f(p) = G_{f,c} * Q_map_{f,c}(p)` descending and retain only the top-`K(p)` frames. Frames with unavailable/non-finite maps receive `S_f(p) = G_{f,c} * L_{f,t,c}` when `fallback_to_tile = true`, otherwise `S_f(p) = 0`. Weighted reconstruction proceeds over this reduced set.

**Warning (binding):** Cherry-pick mode violates the v3.3.9 no-frame-selection invariant at pixel level, even though it does not discard entire frames. It must only be used when explicitly enabled by the user and must be clearly flagged in diagnostic output. Default is `disabled`.

---

## 6. Quality Map Diagnostics

### 6.1 Per-Frame Diagnostics

For each processed frame `f`, the following scalar diagnostics are written to `aqmh_metrics.json`:

| Field | Definition |
|---|---|
| `map_mean` | Mean of `Q_map_{f,c}` over canvas-valid pixels |
| `map_p10` | 10th percentile over canvas-valid pixels |
| `map_p90` | 90th percentile over canvas-valid pixels |
| `artifact_frac` | Fraction of canvas-valid pixels with `Q_map_{f,c} < tau_artifact` (normative default: `tau_artifact = 0.2`) |
| `sharpness_p50` | Median of the **pre-z-score** `Phi_sharp_0` at scale 0 |
| `snr_p50` | Median of the **pre-z-score** `Phi_snr_1` at scale 1 |
| `n_regions` | Number of quality regions (§5.2) above threshold |

If the referenced diagnostic scale is omitted by the small-image scale rule (§2.3.1), the corresponding diagnostic field is written as `NaN` or `null` and the artifact must also record that the scale was unavailable. With normative defaults, `sharpness_p50` is normally available because scale 0 has `D=1`; `snr_p50` may be unavailable when scale 1 is omitted.

### 6.2 Per-Tile Diagnostics

For each tile `t`, the following values are reported alongside the existing v3.3.9 tile metrics:

- `aqmh_q_median`: `Q_{f,t,c}^{aqmh}` as defined in §2.5
- `aqmh_q_p10`, `aqmh_q_p90`: 10th and 90th percentile within the tile
- `aqmh_artifact_frac`: fraction of pixels in the tile with `Q_map < tau_artifact`
- `aqmh_vs_tile_delta`: `Q_{f,t,c}^{aqmh} - Q_{f,t,c}^{local}` (diagnostic for intra-tile heterogeneity)

### 6.3 Heatmaps

For integration into the existing report generator (`tile_compile_cpp/scripts/generate_report.py`, function `_gen_local_metrics`), AQMH emits additional spatial heatmap entries for the `aqmh_metrics.json` artifact:

- Mean `Q_map` per tile, per frame (available as a new tab in the local metrics report section)
- Artifact fraction heatmap per tile
- `aqmh_vs_tile_delta` heatmap (reveals tiles with high intra-tile variance)

---

## 7. Configuration

### 7.1 Top-Level Switch

```yaml
aqmh:
  enabled: false        # default: disabled until validated
```

When `enabled: false`, all AQMH computations are skipped and the pipeline is identical to v3.3.9.

### 7.2 Pyramid Configuration

```yaml
aqmh:
  pyramid:
    scales: 4           # number of scales P (default: 4)
    base_window_px: 4   # window radius R_s in downscaled pixels (default: 4)
    w_sharp: 0.6        # sharpness weight in per-scale sigmoid (default: 0.6)
    w_snr: 0.4          # SNR weight in per-scale sigmoid (default: 0.4)
    k_artifact: 5.0     # outlier detection threshold (default: 5.0)
    frac_artifact_max: 0.10  # artifact gate threshold (default: 0.10)
```

### 7.3 Storage Configuration

```yaml
aqmh:
  storage:
    resolution_divisor: 2   # linear divisor per axis: 1=full, 2=half-width/height (1/4 area), 4=quarter-width/height (1/16 area)
    dtype: float32          # float32 | float16 | uint8 (default: float32)
    max_resident_maps: 2    # bounded read-through cache during reconstruction; 0 disables
```

The storage default (`resolution_divisor = 2`, `dtype = float32`) corresponds to the **1/4-area float32** strategy in §3.2. `max_resident_maps` bounds how many full-resolution maps may be held in RAM simultaneously during phase 9.a; it must not scale with frame count.

### 7.4 Reconstruction Configuration

```yaml
aqmh:
  reconstruction:
    mode: dense_map         # dense_map | tile | hybrid (default: dense_map)
    fallback_to_tile: true  # fall back to tile weights when map unavailable (default: true)
```

### 7.5 Cherry-Pick Mode

```yaml
aqmh:
  cherry_pick:
    enabled: false      # must be explicitly enabled; breaks no-frame-selection invariant at pixel level
    k_min: 3
    k_frac: 0.30
```

### 7.6 Diagnostics Configuration

```yaml
aqmh:
  diagnostics:
    tau_artifact: 0.20  # quality threshold for the artifact_frac diagnostic (default: 0.20)
    q_region: 0.75      # quantile threshold for quality-region extraction (default: 0.75)
    r_morph_canvas_px: 6 # canvas-equivalent radius for quality-region morphology (default: 6)
```

`tau_artifact` is a **diagnostic-only** threshold (see §6.1, §6.2). It does not affect reconstruction weights or the per-scale artifact gate `Phi_artifact_s` (which is governed by `k_artifact` and `frac_artifact_max` in §7.2).

---

## 8. Numerical Defaults

All `eps_aqmh` constants default to `1e-6` unless otherwise specified.

| Parameter | Default | Description |
|---|---|---|
| `eps_aqmh` | `1e-6` | Denominator guard for all AQMH divisions |
| `k_artifact` | `5.0` | Outlier sigma multiplier |
| `frac_artifact_max` | `0.10` | Maximum tolerated outlier fraction per window |
| `w_sharp` | `0.6` | Sharpness weight in per-scale quality sigmoid |
| `w_snr` | `0.4` | SNR weight in per-scale quality sigmoid |
| `P` | `4` | Maximum number of pyramid scales; actual count may be lower due to the omission rule in §2.3.1 |
| `R_s` | `4` | Window radius at each scale (in downscaled pixels) |
| `q_region` | `0.75` | Quality quantile threshold for region extraction |
| `r_morph_canvas_px` | `6` | Morphological opening radius in canvas pixels; map-space radius is `max(1, round(r_morph_canvas_px / resolution_divisor))` |
| `k_frac` | `0.30` | Cherry-pick frame fraction |
| `k_min` | `3` | Minimum frames in cherry-pick mode |
| `tau_artifact` | `0.20` | Quality threshold for artifact fraction diagnostic |
| `max_resident_maps` | `2` | Max full-resolution maps held in RAM during reconstruction |
| `resolution_divisor` | `2` | Storage downscale factor per axis (1/4-area default) |

---

## 9. Validation Requirements

When AQMH is enabled, all mandatory v3.3.9 validation tests (§7.3) remain binding. Additionally:

1. **Map range:** `Q_map_{f,c}(p) ∈ [0, 1]` for all finite canvas-valid pixels.
2. **Canvas guard:** `Q_map_{f,c}(p) = 0` for all canvas-invalid pixels.
3. **Determinism:** Identical registered frames and canvas masks produce identical quality maps.
4. **Fallback coverage:** Every pixel with `V_{t,c}^{I}(p) = ∅` returns zero/unsupported; every pixel with no finite map samples but finite intensity samples returns the weight-fallback unweighted mean; no NaN/Inf in output.
5. **Explicit zero-veto:** If finite maps exist at a pixel and all available dense-map weights are zero, the output remains unsupported/zero and must not be replaced by an unweighted mean.
6. **Tile compatibility:** `Q_{f,t,c}^{aqmh}` matches `median(Q_map)` within floating-point tolerance.
7. **No structural injection:** Seam scores and FWHM must not worsen vs. the v3.3.9 tile-weight baseline on the same dataset.
8. **Artifact detection:** Known satellite-contaminated frames show elevated `artifact_frac > 0.01` for at least the contaminated tiles.
9. **Scale omission:** For an input where `P_actual < P` (for example `min(W,H) < 64` with defaults), fusion uses `P_actual` as the geometric-mean denominator, omitted scales are recorded in diagnostics, and unavailable diagnostic scales are written as `NaN`/`null`.
10. **Cherry-pick flag:** When `cherry_pick.enabled = true`, the output artifact `aqmh_metrics.json` must contain `cherry_pick_active: true` and the pipeline log must emit a `WARNING` level message.

---

## 10. Scope Boundary

### Mandatory Core (Unchanged)

- Registration, normalization, global metrics, tile geometry, tile reconstruction OLA semantics, clustering, final stack

### AQMH Extension (Optional)

- Dense quality map computation (phase 8.a)
- Pixel-wise weighted reconstruction (phase 9.a, when `mode = dense_map` or `mode = hybrid`; diagnostics-only AQMH map generation when `mode = tile`)
- Adaptive region extraction (§5)
- Cherry-pick stacking (§5.3, explicit opt-in only)
- AQMH diagnostic artifacts

---

## 11. Core Statement

AQMH preserves deterministic weighted-mean reconstruction, canvas exclusion, and the non-hallucination invariants of the v3.3.9 mandatory core. It extends the expressiveness of the local weight model from a per-tile scalar field to a continuous per-pixel quality field. Every pixel that enters the reconstruction accumulator does so with a deterministic non-negative weight that reflects both global atmospheric conditions and local spatial quality — without artificial tile boundaries and without discarding usable data due to intra-tile heterogeneity.
