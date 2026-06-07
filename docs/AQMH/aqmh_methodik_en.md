# Adaptive Quality Mask Harvesting (AQMH) — Methodology v0.1.0

**Status:** Exploratory specification — not yet normative  
**Version:** v0.1.0 (2026-06-05)  
**Last revised:** 2026-06-05  
**Relation to core:** Independent reconstruction method; may reuse shared preprocessing infrastructure

---

## 0. Motivation and Objective

### 0.1 Motivation

AQMH is a separate quality-map-based stacking method. It is motivated by the observation that local image quality is often spatially heterogeneous: a satellite trail, cloud edge, registration remnant, or hot-pixel cluster can affect a small region of a registered frame while the remaining frame area is still useful.

Any method that assigns one local scalar quality value to a large spatial block has two structural limitations:

1. **Intra-region heterogeneity:** A small contaminated area can influence the quality assigned to a much larger region.

2. **Block-boundary discontinuities:** If weights are constant inside coarse spatial blocks, reconstruction quality can jump at block boundaries.

### 0.2 AQMH Objective

The Adaptive Quality Mask Harvesting method computes a **continuous, pixel-resolved quality weight field** `Q_map_{f,c}(x,y)` for every frame. The reconstruction then performs a **pixel-wise weighted mean** using AQMH weights only.

Core objectives:

1. Extract the usable fraction of every frame at pixel resolution.
2. Avoid spatial block-boundary weight discontinuities in the final stack.
3. Model local quality heterogeneity via a multi-scale analysis pyramid.
4. Preserve deterministic weighted-mean reconstruction, canvas exclusion, and non-hallucination invariants.
5. Function independently of Classic Tile Compile. Classic outputs may be used only as external comparison baselines, not as AQMH inputs or fallbacks.

### 0.3 Independence and Shared Infrastructure

AQMH is an **independent reconstruction method**. It may reuse shared pipeline infrastructure, but its quality model and reconstruction weights are not derived from Classic Tile Compile local/tile metrics.

Shared infrastructure may include:

- input scan and frame selection
- calibration and registration/prewarping
- global photometric normalization
- common-overlap/canvas-valid mask
- run management, logging, artifacts, reports, and UI plumbing

The AQMH algorithm itself consists of:

- AQMH dense quality-map computation
- AQMH pixel-wise weighted reconstruction
- AQMH diagnostics and optional region extraction

Classic Tile Compile and AQMH must be runnable independently. Enabling or disabling one method must not change the mathematical definition of the other.

---

## 1. Principles and Definitions

### 1.1 Physical Objective

The method models per-pixel observational quality as the product of two separable components:

- **Frame-level quality:** the global atmospheric state of frame `f`, captured by `G_{f,c}`. `G_{f,c}` is an AQMH input derived from shared global frame diagnostics and normalization; it is not a Classic tile/local metric.
- **Spatial quality field:** the continuous quality distribution within the frame, captured by `Q_map_{f,c}(x,y)`.

The effective pixel weight is:

`W_{f,c}^{aqmh}(x,y) = G_{f,c} * Q_map_{f,c}(x,y)`

### 1.2 Invariants (Binding)

The following invariants are binding for AQMH reconstruction. The optional cherry-pick mode is an explicit opt-in deviation and is governed separately by §5.3.

1. **No frame selection:** Entire frames must not be removed based on quality.
2. **Conditional photometric linearity:** Once the deterministic weights have been computed, the final reconstruction remains `R(p) = sum_f w_f(p) * I_f(p) / sum_f w_f(p)` with `w_f(p) >= 0`. AQMH must not apply nonlinear intensity transforms to the samples entering the accumulator.
3. **Determinism:** All quality map computations must be deterministic and reproducible.
4. **Canvas exclusion:** Canvas-invalid pixels are excluded from all AQMH accumulators and statistics. They are written as zero/unsupported only in final output arrays.
5. **No hallucination:** AQMH outputs are weights and masks only. It does not generate or predict pixel intensities.

### 1.3 Notation

- `f` frame index
- `c` channel index
- `(x, y)` pixel coordinates in the registered canvas
- `Q_map_{f,c}(x,y)` per-pixel quality field, `∈ [0, 1]`
- `D_s` downscale factor at pyramid scale `s`
- `P` configured maximum number of pyramid scales
- `S_actual` ordered set of pyramid scales actually computed after the small-image omission rule
- `P_actual = |S_actual|` actual number of computed pyramid scales used for fusion
- `R_s` spatial radius of the local analysis window at scale `s`
- `Psi_s(x,y)` quality contribution at scale `s`
- `W_{f,c}^{aqmh}(x,y)` effective AQMH pixel weight
- `B_s(x,y; R)` masked local background operator at scale `s`

### 1.4 Deterministic Statistics Convention

All medians, MADs, and quantiles in AQMH are computed over finite values only and over the explicitly stated valid support (for example `W_s_valid` or canvas-valid pixels). If the support is empty, the statistic is invalid and the fallback rules in §2.3.4 apply.

For deterministic reproducibility, sort finite values in ascending numeric order. The median is the middle value for odd sample counts and the arithmetic mean of the two middle values for even sample counts. MAD uses the same median convention on `|x - median(x)|`. Quantiles use linear interpolation between sorted samples with index `q * (n - 1)`, clamped to `[0, n-1]`; for `n = 1`, the only sample is returned.

### 1.5 Canvas Exclusion Contract

Canvas-invalid pixels are outside the observed data domain. They are not zero-valued samples, not background samples, not low-quality samples, and not padding. They must not influence AQMH in any phase.

Binding rules:

1. Canvas-invalid source pixels are converted to invalid/NaN before AQMH map computation.
2. Downsampling uses a valid-count denominator; invalid pixels do not contribute value or weight.
3. Local statistics, filters, medians, MADs, Laplacian responses, artifact fractions, z-score populations, and quantiles operate only on finite canvas-valid support.
4. Upsampling from scale space to canvas space is mask-aware; invalid scale samples do not interpolate into valid canvas pixels.
5. Reconstruction iterates only over canvas-valid output pixels, and the frame sample set `V_c^I(p)` contains only finite, canvas-valid source samples.
6. Diagnostics and region extraction use canvas-valid support only. Raw invalid-canvas area may be present in arrays for shape compatibility, but it must be excluded from all statistics.
7. The final `Q_map` canvas guard sets invalid canvas pixels to exactly zero as an output convention only. That zero must never be fed back as a data sample into AQMH statistics.

---

## 2. Dense Quality Map Computation

### 2.1 Overview

For each frame `f` and channel `c`, a dense quality map `Q_map_{f,c}` is computed from the prewarped, normalized frame `I_{f,c}`. The map is computed in a **multi-scale pyramid** with `P` scales. Each per-scale quality contribution is upscaled to canvas resolution, then the upscaled contributions are fused via geometric mean.

### 2.2 Input Data

The input to AQMH is the registered, prewarped, photometrically normalized frame `I_{f,c}(x,y)` produced by shared preprocessing. The common-overlap canvas mask applies: canvas-invalid pixels are excluded from all quality map accumulators.

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

`Phi_sharp_s(x,y) = Var_{p in W_s_valid(x,y)}(Lap_valid(I_s)(p))`

where `Lap_valid` is the masked valid-support Laplacian response and `Var` is the local variance of finite, valid Laplacian values. `Lap_valid` must not use mirrored, replicated, zero-filled, or canvas-invalid neighbors. If the center pixel is invalid or the finite stencil support is insufficient for a deterministic Laplacian estimate, the response is invalid. Result is clamped to `[0, +inf)` (non-negative).

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

**Within-frame relativity (binding clarification):** Because `z(Phi_sharp_s)` and `z(Phi_snr_s)` are normalized per frame (median/MAD computed over that frame's pixels at scale `s`), the **sigmoid factor of `Q_map` is a within-frame relative quality field**, not an absolute cross-frame quality measurement. Two frames of differing global seeing produce similarly distributed sigmoid factors, each self-normalized. Consequently, in the AQMH reconstruction weight `W_{f,c}^{aqmh} = G_{f,c} * Q_map_{f,c}` (§1.1, §4.3):

- **Between-frame** discrimination at a given pixel is carried by the global weight `G_{f,c}` and by the **absolute** artifact gate `Phi_artifact_s` (which is not z-scored and can drive `Q_map` toward zero in any frame independently of other frames).
- **Within-frame** spatial discrimination (which regions of a single frame are sharper / cleaner) is carried by the sigmoid factor.

This division of labor is intentional and binding: the sigmoid factor must not be interpreted as an absolute photometric quality and must not be used to rank whole frames.

### 2.3.4 Boundary and Empty-Window Rules

All local statistics are computed over finite, canvas-valid pixels only. Let `W_s_valid(x,y)` be the valid subset of the analysis window.

If `|W_s_valid(x,y)| = 0`, all per-scale signals at `(x,y)` are marked invalid and the fused `Q_map` value is later set to zero by the canvas guard. If `|W_s_valid(x,y)| > 0` but fewer than three valid pixels are available, robust scale estimates fall back to `eps_aqmh` and local variance estimates fall back to zero. For the `Phi_snr_s` signal, the background-centering fallback rule in §2.3.2(b) is more specific and takes precedence.

The default boundary mode for convolution-like operations (`Lap_valid`, `blur`, local windows, and morphology support masks) is valid-only masked evaluation. Implementations must not mirror, replicate, zero-fill, or otherwise synthesize canvas-invalid pixels into the statistic. If a library primitive cannot express this support rule directly, implementations must compute numerator and valid-support denominator separately or use an explicit masked operator.

### 2.4 Multi-Scale Fusion

Let `S_actual` be the ordered set of scales that are actually computed after applying the omission rule in §2.3.1, and let `P_actual = |S_actual|`. Upsample each computed `Psi_s` to the full canvas resolution using mask-aware bilinear interpolation:

`Psi_s^{up}(x,y) = upsample_valid(Psi_s, valid_s, D_s)`

where `valid_s` is the finite valid-support mask of `Psi_s`. `upsample_valid` interpolates the numerator `Psi_s * valid_s` and the support mask `valid_s` separately, then divides by the interpolated support. If the interpolated support is zero at a canvas pixel, `Psi_s^{up}` is invalid at that pixel. Invalid scale samples must not be treated as zero during interpolation because that would depress neighboring valid canvas pixels.

Fuse via **geometric mean** over the `P_actual` computed scales:

`Q_map_{f,c}(x,y) = ( prod_{s in S_actual} Psi_s^{up}(x,y) )^{1/P_actual}`

The configured `P` is an upper bound. It must not be used as the exponent denominator when one or more scales are omitted. If `P_actual = 0`, `Q_map` is defined as zero everywhere after the canvas guard.

Geometric mean is chosen over arithmetic mean because it requires **all scales to agree** on high quality. A single scale signaling an artifact suppresses the fused map regardless of other scales. This implements a conservative "all-clear" fusion philosophy.

**Zero-scale guard:** If `Psi_s^{up}(x,y) = 0` for any scale `s`, then `Q_map_{f,c}(x,y) = 0` exactly (one bad scale vetos the pixel).

**Canvas guard (binding):** For all canvas-invalid pixels `p`, set `Q_map_{f,c}(p) = 0` unconditionally after fusion, overriding any computed value.

If any computed scale is invalid at a canvas-valid pixel because there is no valid scale support after mask-aware upsampling, that scale contributes a zero-veto at that pixel. This is distinct from canvas-invalid pixels, which are excluded and zeroed only by the final canvas guard.

### 2.5 Block-Level Diagnostic Summaries

For reports and visual summaries, AQMH may derive block-level diagnostic values by aggregating `Q_map_{f,c}` over a display block `b`:

`Q_{f,b,c}^{aqmh} = median_{p in b, canvas-valid} Q_map_{f,c}(p)`

The block grid is a reporting/visualization aid only. It is not part of the AQMH reconstruction weight model, and it must not introduce block-constant weights into the AQMH accumulator.

---

## 3. Quality Map Storage and Memory Model

### 3.1 Storage Format

Conceptually, `Q_map_{f,c}` is a full-canvas quality field, one per frame and channel. The persisted representation may be lower resolution or quantized according to §3.2. For multi-channel CFA inputs processed via the CFA-proxy-equivalent core variant, a single luminance-channel map may be used instead of per-channel maps (configurable).

Recommended storage: on the existing `DiskCacheFrameStore` or a parallel quality-map disk cache with identical indexing semantics.

### 3.2 Memory Budget

At full resolution, one map requires `W * H * 4` bytes. For a 24 Mpx sensor with 300 frames and 3 channels, the full uncompressed **on-disk working set** would be approximately `24e6 * 4 * 300 * 3 ≈ 86 GB`.

This number is **not** a permitted RAM budget. AQMH must never assume that all frames, all prewarped frames, or all quality maps are resident in memory. Like the rest of Tile Compile, AQMH is designed for hundreds of frames and must be implemented as a streaming, disk-cache-backed method in every stage.

Binding memory invariant:

1. At AQMH map-computation time, each worker may hold only the current source frame, its current pyramid temporaries, and the current output map.
2. After a frame's `Q_map` has been computed, it must be written to the AQMH map cache promptly and its full-resolution working buffers must be released.
3. AQMH reconstruction must read frames and maps through bounded providers/caches. The number of resident source frames and resident full-resolution maps must be bounded by explicit memory limits and must not scale with frame count.
4. A valid implementation must be able to process hundreds of frames without OOM by trading memory for disk IO.

Therefore, the following compression strategies are supported:

| Strategy | Description | Normative? |
|---|---|---|
| Full resolution float32 | No compression | Optional |
| 1/4 area float32 | Downscale by 2 in each axis | **Default** |
| uint8 quantization | Map scaled to `[0, 255]` | Optional |
| Block-compressed float16 | Per-block float16 sub-blocks | Future optional; only valid when explicitly implemented |

**Normative default:** Store `Q_map` at `1/4` area resolution (factor-2 downscale in each axis, `resolution_divisor = 2`). The map is upscaled to full resolution on demand during reconstruction via bilinear interpolation.

### 3.3 Disk Cache Lifecycle

Quality maps are written during AQMH map computation and consumed during AQMH reconstruction. Maps are invalidated when the source prewarped frame is invalidated, when the common-overlap mask changes, or when any **map-affecting** AQMH configuration changes (`pyramid`, `storage`, or map format version). Reconstruction-only settings must not invalidate map cache entries. Implementations should store and validate a cache metadata hash covering only map-affecting inputs.

The cache is not an optimization; it is part of the AQMH execution model. Implementations must use cache-backed access for all large per-frame data products:

| Stage | Large data | Required access pattern |
|---|---|---|
| Shared preprocessing | calibrated/registered/prewarped frames | disk-backed frame store; bounded resident frame set |
| AQMH map computation | source frame, pyramid buffers, output map | one frame per worker; write-through map cache |
| AQMH reconstruction | source frames and `Q_map` files | bounded frame/map read cache; no full-run preload |
| AQMH diagnostics/report | metrics and summaries | aggregate JSON/statistics; raw maps remain cache artifacts |

---

## 4. Pipeline Integration

### 4.1 AQMH Processing Stages

AQMH has its own algorithmic stages. A concrete application may schedule these stages inside existing runner phases for engineering convenience, but that scheduling is not part of the mathematical method.

```
AQMH_MAPS
  Compute dense per-frame quality maps Q_map

AQMH_RECONSTRUCTION
  Perform pixel-wise weighted stacking with W_aqmh = G * Q_map

AQMH_DIAGNOSTICS
  Emit quality-map, reconstruction, and optional region artifacts
```

Shared preprocessing and postprocessing stages may be reused, but Classic Tile Compile local metrics and tile reconstruction are not AQMH stages.

### 4.2 AQMH Map Computation

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
7. Write `Q_map_{f,c}` to the AQMH quality-map disk cache (at configured storage resolution).

### 4.3 AQMH Pixel-Wise Weighted Reconstruction

For each pixel `p` in the canvas-valid support:

Define the finite intensity sample set:

`V_c^{I}(p) = { f | I_{f,c}(p) is finite AND canvas-valid }`

Define the map-available sample set:

`V_c^{map}(p) = { f in V_c^{I}(p) | Q_map_{f,c}(p) is finite }`

For each `f in V_c^{I}(p)`, the effective AQMH pixel weight is:

`w_{f,c}^{aqmh}(p) = G_{f,c} * Q_map_{f,c}(p)` when `f in V_c^{map}(p)`

`w_{f,c}^{aqmh}(p) = 0` when the map sample is unavailable or non-finite

No Classic Tile Compile local/tile weight is used as an AQMH fallback.

Canvas-invalid output pixels are not reconstructed. They are written as unsupported/zero without evaluating frame samples, map samples, sigma clipping, or denominator fallback. Canvas-invalid source pixels are never members of `V_c^I(p)`.

The reconstructed pixel value is:

`R_c^{aqmh}(p) = sum_{f in V^I} w_{f,c}^{aqmh}(p) * I_{f,c}(p) / sum_{f in V^I} w_{f,c}^{aqmh}(p)`

**Unsupported-pixel handling (binding):** Before applying any numerical denominator guard, implementations must distinguish finite map samples from unavailable map samples. A finite zero is a valid map sample and is an explicit veto, not a missing value. If `sum_f max(w_{f,c}^{aqmh}(p), 0) <= eps_weight`, fallback behavior depends on why the sum is zero:

1. If at least one finite map sample exists at `p` (`V_c^{map}(p) != empty`) and all AQMH weights are zero because the available maps explicitly veto the pixel (`Q_map = 0`), the output pixel is marked unsupported/zero. Do **not** replace the explicit zero-veto by an unweighted mean.
2. If no finite map sample exists at `p`, or all map samples are unavailable because of IO/cache failure, the output pixel is marked unsupported/zero and the run emits an AQMH cache/map-availability warning. AQMH must not silently switch to Classic Tile Compile weights.
3. If sigma clipping removes all samples after a nonzero pre-clip weight sum, use the AQMH sigma-clipping keep-floor and denominator-guard semantics defined for pixel-wise weighted reconstruction.

**Sigma clipping:** Iterative weighted sigma clipping applies with `w_{f,c}^{aqmh}(p)`. The keep-floor `min_fraction` and `N_eff` / `D_eff` guards are AQMH reconstruction parameters and must be deterministic.

---

## 5. Adaptive Region Extraction (Optional)

### 5.1 Motivation

In addition to the continuous weight map, AQMH can generate **binary quality regions** for diagnostic reporting and for the optional cherry-pick stacking mode.

### 5.2 Quality Contour Extraction

From the fused `Q_map_{f,c}`, extract binary regions by thresholding:

1. Compute the per-frame threshold: `tau_f = quantile(Q_map_{f,c}, q_region)` over finite, canvas-valid pixels only, with normative default `q_region = 0.75`.
2. Binary mask: `M_f(x,y) = 1 iff Q_map_{f,c}(x,y) >= tau_f AND canvas-valid`.
3. Apply morphological opening with radius `r_morph_canvas_px` to remove isolated noise regions. The normative default is a **canvas-equivalent radius of 6 px**, which corresponds to 3 px at the default `resolution_divisor = 2`. If region extraction is run on a stored/downscaled map, use `r_morph_map = max(1, round(r_morph_canvas_px / resolution_divisor))`. Morphology is constrained to the canvas-valid support: invalid canvas pixels are outside the domain, not zero-valued background pixels, and the final region mask is always intersected with the canvas-valid mask.
4. Extract connected components; label each component with:
   - `Area_r`: pixel count
   - `MeanQ_r`: mean quality score over the region
   - `Compactness_r = 4*pi*Area_r / Perimeter_r^2` (Polsby-Popper score)
5. Rank regions by `Score_r = MeanQ_r * log(1 + Area_r)`.

These regions are reported in the diagnostic artifact `aqmh_regions.json` per frame.

### 5.3 Optional Cherry-Pick Stacking Mode

When `aqmh.cherry_pick.enabled = true`, per-pixel stacking uses only the **top-K frames** by quality, rather than all frames:

`K(p) = min(N_valid(p), max(k_min, floor(k_frac * N_valid(p))))`

where `N_valid(p) = |V_c^{I}(p)|` is the number of finite intensity samples at pixel `p` (the intensity sample set of §4.3), and with normative defaults `k_min = 3`, `k_frac = 0.3`.

For each pixel `p`, sort `V_c^{I}(p)` by the cross-frame calibrated score `S_f(p) = G_{f,c} * Q_map_{f,c}(p)` descending and retain only the top-`K(p)` frames. Frames with unavailable/non-finite maps receive `S_f(p) = 0`. Weighted reconstruction proceeds over this reduced set.

**Warning (binding):** Cherry-pick mode violates the default AQMH no-frame-selection invariant at pixel level, even though it does not discard entire frames. It must only be used when explicitly enabled by the user and must be clearly flagged in diagnostic output. Default is `disabled`.

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

### 6.2 Block-Level Diagnostics

For each report block `b`, the following values may be reported:

- `aqmh_q_median`: `Q_{f,b,c}^{aqmh}` as defined in §2.5
- `aqmh_q_p10`, `aqmh_q_p90`: 10th and 90th percentile within the block
- `aqmh_artifact_frac`: fraction of pixels in the block with `Q_map < tau_artifact`

### 6.3 Heatmaps

For integration into the report generator, AQMH emits spatial heatmap entries for the `aqmh_metrics.json` artifact:

- Mean `Q_map` per report block, per frame
- Artifact fraction heatmap per report block
- Optional AQMH-vs-Classic comparison heatmaps only when both methods were run separately on the same input set

---

## 7. Configuration

### 7.1 Top-Level Switch

```yaml
method: aqmh              # optional explicit method key: classic_tile_compile | aqmh
aqmh:
  enabled: false        # default: disabled until validated
```

When `aqmh.enabled: false`, all AQMH computations are skipped. If the implementation does not yet support an explicit top-level `method` key, runtime status must still expose the derived method: `aqmh.enabled = false` means `classic_tile_compile`, and `aqmh.enabled = true` means `aqmh`.

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

The storage default (`resolution_divisor = 2`, `dtype = float32`) corresponds to the **1/4-area float32** strategy in §3.2. `max_resident_maps` bounds how many full-resolution maps may be held in RAM simultaneously during AQMH reconstruction; it must not scale with frame count.

### 7.4 Cherry-Pick Mode

```yaml
aqmh:
  cherry_pick:
    enabled: false      # must be explicitly enabled; breaks no-frame-selection invariant at pixel level
    k_min: 3
    k_frac: 0.30
```

### 7.5 Diagnostics Configuration

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

When AQMH is enabled, the following validation requirements apply:

1. **Map range:** `Q_map_{f,c}(p) ∈ [0, 1]` for all finite canvas-valid pixels.
2. **Canvas guard:** `Q_map_{f,c}(p) = 0` for all canvas-invalid pixels.
3. **Determinism:** Identical registered frames and canvas masks produce identical quality maps.
4. **Unsupported coverage:** Every pixel with `V_c^{I}(p) = empty` returns zero/unsupported; every pixel with finite intensity samples but no finite AQMH map samples returns zero/unsupported with an AQMH warning; no NaN/Inf in output.
5. **Explicit zero-veto:** If finite maps exist at a pixel and all available AQMH weights are zero, the output remains unsupported/zero and must not be replaced by an unweighted mean.
6. **Block diagnostic consistency:** `Q_{f,b,c}^{aqmh}` matches `median(Q_map over b)` within floating-point tolerance.
7. **No structural injection:** Seam scores, FWHM, and background RMS must not regress against an AQMH-disabled control run beyond the documented validation tolerance on the same dataset.
8. **Artifact detection:** Known satellite-contaminated frames show elevated `artifact_frac > 0.01` for at least the contaminated report blocks.
9. **Scale omission:** For an input where `P_actual < P` (for example `min(W,H) < 64` with defaults), fusion uses `P_actual` as the geometric-mean denominator, omitted scales are recorded in diagnostics, and unavailable diagnostic scales are written as `NaN`/`null`.
10. **Cherry-pick flag:** When `cherry_pick.enabled = true`, the output artifact `aqmh_metrics.json` must contain `cherry_pick_active: true` and the pipeline log must emit a `WARNING` level message.

---

## 10. Scope Boundary

### Shared Infrastructure

- Input scan, calibration, registration/prewarping, global normalization, common-overlap mask, run management, logging, reports

### AQMH Method

- Dense quality map computation
- Pixel-wise weighted AQMH reconstruction
- Adaptive region extraction (§5)
- Cherry-pick stacking (§5.3, explicit opt-in only)
- AQMH diagnostic artifacts

---

## 11. Core Statement

AQMH is an independent deterministic weighted-mean reconstruction method. It uses a continuous per-pixel quality field rather than block-constant local weights. Every pixel that enters the reconstruction accumulator does so with a deterministic non-negative AQMH weight that reflects both global atmospheric conditions and local spatial quality, without artificial block boundaries and without relying on Classic Tile Compile weights or fallback behavior.
