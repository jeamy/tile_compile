# Adaptive Quality Mask Harvesting (AQMH) — Methodology v0.2.0

**Status:** Draft specification — refined, pending empirical validation
**Version:** v0.2.0 (2026-07-03)
**Last revised:** 2026-07-03
**Relation to core:** Independent reconstruction method; may reuse shared preprocessing infrastructure

---

## 0. Motivation and Objective

### 0.1 Motivation

AQMH is a separate quality-map-based stacking method. It is motivated by the observation that local image quality is often spatially heterogeneous: a satellite trail, cloud edge, registration remnant, or hot-pixel cluster can affect a small region of a registered frame while the remaining frame area is still useful.

Any method that assigns one local scalar quality value to a large spatial block has two structural limitations:

1. **Intra-region heterogeneity:** A small contaminated area can influence the quality assigned to a much larger region.

2. **Block-boundary discontinuities:** If weights are constant inside coarse spatial blocks, reconstruction quality can jump at block boundaries.

### 0.2 AQMH Objective

The Adaptive Quality Mask Harvesting method computes a **continuous per-pixel quality weight field** `Q_map_{f,c}(x,y)` for every frame. The reconstruction then performs a **pixel-wise quality-weighted mean** using AQMH weights only: each output pixel is reconstructed from the frame samples at that same pixel position, weighted by the corresponding AQMH quality value.

Core objectives:

1. Extract the usable fraction of every frame at pixel resolution.
2. Avoid spatial block-boundary weight discontinuities in the final stack.
3. Model local quality heterogeneity via a multi-scale analysis pyramid.
4. Preserve deterministic weighted-mean reconstruction, canvas exclusion, and non-hallucination invariants.
5. Function independently of Classic Tile Compile. Classic outputs may be used only as external comparison baselines, not as AQMH inputs or fallbacks.
6. Guard any optional deviation from full-frame-set reconstruction (cherry-pick, §5.3) with explicit sample-count thresholds and diagnostics. These safeguards limit statistical risk; they do not by themselves guarantee a quality improvement.

### 0.3 Independence and Shared Infrastructure

AQMH is an **independent reconstruction method**. It may reuse shared pipeline infrastructure, but its quality model and reconstruction weights are not derived from Classic Tile Compile local/tile metrics.

Shared infrastructure may include:

- input scan and non-quality eligibility filtering (readability, calibration availability, and explicit user exclusions only)
- calibration and registration/prewarping
- global photometric normalization
- global output-canvas mask and frame-specific registered-valid masks
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

- **Frame-level quality:** the global atmospheric state of frame `f`, captured by the strictly positive AQMH factor `G_{f,c}` defined below.
- **Spatial quality field:** the continuous quality distribution within the frame, captured by `Q_map_{f,c}(x,y)`.

For every frame/channel, derive the global pre-z-score summaries from its AQMH maps:

`g_sharp_{f,c} = median_{p source-valid}(Phi_sharp_0(p))`

`g_snr_{f,c} = median_{p source-valid}(Phi_snr_1(p))`, using the finest available scale if scale 1 is omitted.

Normalize each summary across frames of the same channel with the deterministic robust z-score convention of §1.4. Invalid summaries receive cross-frame z-score zero and are flagged. Then define:

`G_{f,c} = g_floor + (1 - g_floor) * sigmoid(g_w_sharp * z_f(g_sharp) + g_w_snr * z_f(g_snr))`

with normative defaults `g_floor = 0.05`, `g_w_sharp = 0.6`, and `g_w_snr = 0.4`. For finite inputs, `G_{f,c} in (g_floor, 1)`; an otherwise eligible frame cannot be removed by assigning global quality zero. The definition is AQMH-native and deterministic; no Classic local/tile metric is an input.

The effective pixel weight is:

`W_{f,c}^{aqmh}(x,y) = G_{f,c} * Q_map_{f,c}(x,y)`

### 1.2 Invariants (Binding)

The following invariants are binding for AQMH reconstruction. The optional cherry-pick mode is an explicit opt-in deviation and is governed separately by §5.3, including hard sample-count floors that can force it off at runtime.

1. **No frame selection:** Entire frames must not be removed based on quality.
2. **Conditional photometric linearity:** Once the deterministic weights have been computed, the final reconstruction remains `R(p) = sum_f w_f(p) * I_f(p) / sum_f w_f(p)` with `w_f(p) >= 0`. AQMH must not apply nonlinear intensity transforms to the samples entering the accumulator.
3. **Determinism:** All quality map computations must be deterministic and reproducible.
4. **Canvas exclusion:** Canvas-invalid pixels are excluded from all AQMH accumulators and statistics. They are written as zero/unsupported only in final output arrays.
5. **No hallucination:** AQMH outputs are weights and masks only. It does not generate or predict pixel intensities.
6. **Sample-count sufficiency for optional selection modes:** Any mode that reduces the per-pixel sample count below the full valid set (currently: cherry-pick, §5.3) must enforce a documented minimum retained-sample count, below which the mode is disabled automatically.

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
- `C(p)` global output-canvas mask; pixels with `C(p) = 0` are outside the reconstruction domain
- `M_f(p)` frame-specific registered-valid mask inside `C`; it may vary with dithering, saturation, registration support, or missing data
- `N_valid(p)` number of finite intensity samples at pixel `p` (`= |V_c^{I}(p)|`, §4.3)
- `N_rankable(p)` number of finite intensity samples at pixel `p` with a finite, strictly positive AQMH selection score (§5.3.1)
- `k_frac(p)` effective cherry-pick fraction at pixel `p`: the greater of the base fraction and the matching tier fraction when tiering is enabled, otherwise the base `k_frac` (§5.3.4)
- `K_nominal(p) = floor(k_frac(p) * N_rankable(p))` unclamped nominal retained count at pixel `p` (§5.3.2)
- `K_nominal_median` run-level median of `K_nominal(p)` over pixels with `C(p) = 1`, used for the cherry-pick eligibility gate (§5.3.2)
- `K(p)` number of retained frames at pixel `p` under cherry-pick mode (§5.3.3)

### 1.4 Deterministic Statistics Convention

All medians, MADs, and quantiles in AQMH are computed over finite values only and over the explicitly stated valid support. “Source-valid” means `C(p) = 1`, `M_f(p) = 1`, and a finite source sample. If the support is empty, the statistic is invalid and the fallback rules in §2.3.4 apply.

For deterministic reproducibility, sort finite values in ascending numeric order. The median is the middle value for odd sample counts and the arithmetic mean of the two middle values for even sample counts. MAD uses the same median convention on `|x - median(x)|`. Quantiles use linear interpolation between sorted samples with index `q * (n - 1)`, clamped to `[0, n-1]`; for `n = 1`, the only sample is returned.

For a finite population `X`, define the dimension-preserving numerical floor

`eps_scale(X) = max(nextafter(0, 1), eps_rel * max(median(|X|), MAD(X)))`

with `eps_rel = 1e-6`. If all values are exactly zero, `nextafter(0,1)` is used only as a division guard. Robust z-scores use

`z_X(x) = (x - median(X)) / max(1.4826 * MAD(X), eps_scale(X))`.

If `MAD(X) = 0`, the population is treated as statistically degenerate and `z_X(x) = 0` for all `x`; implementations must not amplify sub-floor differences. This convention preserves physical dimensions and makes the non-degenerate z-score invariant to positive global rescaling.

For local noise denominators, use the centered floor

`eps_noise(X) = max(nextafter(0, 1), eps_rel * median(|X - median(X)|))`.

Unlike `eps_scale`, `eps_noise` is independent of a constant background offset and therefore cannot suppress a small valid noise estimate merely because the scene has a large absolute intensity level.

### 1.5 Canvas and Frame-Validity Contract

Pixels outside `C` are outside the output domain. Pixels inside `C` can still be unavailable in an individual frame when `M_f(p) = 0`. Neither class is a zero-valued, background, or low-quality sample.

Binding rules:

1. Source samples with `C(p) = 0` or `M_f(p) = 0` are converted to invalid/NaN before AQMH map computation.
2. Downsampling uses a valid-count denominator; output-domain-invalid and frame-invalid pixels do not contribute value or weight.
3. Local statistics, filters, medians, MADs, Laplacian responses, artifact fractions, z-score populations, and quantiles operate only on finite source-valid support.
4. Upsampling from scale space to canvas space is mask-aware; invalid scale samples do not interpolate into valid canvas pixels.
5. Reconstruction iterates only where `C(p) = 1`, and `V_c^I(p)` contains only finite samples with `M_f(p) = 1`.
6. Per-frame diagnostics and region extraction use source-valid support only. Run-level spatial aggregates use `C`; raw invalid areas must be excluded from all statistics.
7. The final `Q_map` guard sets pixels with `C(p) = 0` or `M_f(p) = 0` to exactly zero as an output convention only. That zero must never be fed back as a data sample into AQMH statistics.

---

## 2. Dense Quality Map Computation

### 2.1 Overview

For each frame `f` and channel `c`, a dense quality map `Q_map_{f,c}` is computed from the prewarped, normalized frame `I_{f,c}`. The map is computed in a **multi-scale pyramid** with `P` scales. Each per-scale quality contribution is upscaled to canvas resolution, then the upscaled contributions are fused via geometric mean.

### 2.2 Input Data

The input to AQMH is the registered, prewarped, photometrically normalized frame `I_{f,c}(x,y)` produced by shared preprocessing. Both `C` and `M_f` apply; only source-valid pixels enter quality-map statistics.

### 2.3 Multi-Scale Pyramid

#### 2.3.1 Scale Definition

Define `P` analysis scales with downscale factors `D_s` and window radii `R_s`:

| Scale `s` | Downscale `D_s` | Window `R_s` | Captured structure |
|---|---|---|---|
| 0 | 1  | 4 px  | sub-tile, pixel-near defects, hot pixels |
| 1 | 4  | 4 px  | tile-comparable (≈ 16 px canvas-space radius) |
| 2 | 16 | 4 px  | coarse regions (≈ 64 px canvas-space radius) |
| 3 | 64 | 4 px  | global frame quality context (≈ 256 px canvas-space radius) |

Normative defaults: `P = 4`, `D_s = 4^s`, `R_s = 4` (in downscaled pixels).

A scale `s` is **omitted** when `D_s > min(W, H) / 16`; the configured maximum `P` remains unchanged, while `S_actual` and `P_actual` are reduced accordingly. Equivalently, scale `s` requires `min(W, H) >= 16 * D_s`. Worked examples for the normative defaults: scale 1 (`D=4`) requires `min(W,H) >= 64`; scale 2 (`D=16`) requires `>= 256`; scale 3 (`D=64`) requires `>= 1024`. Thus small images automatically drop the coarsest scales (e.g. a 512 px image keeps only scales 0–2).

#### 2.3.2 Per-Scale Signal Computation

At scale `s`, compute a downscaled version of the input:

`I_s(x,y) = downsample(I_{f,c}, D_s)`

using area-averaging with a validity-aware denominator (samples outside `C` or `M_f` are excluded from the area mean, not replaced by zero).

For each pixel `(x,y)` in the downscaled domain, compute the following **three quality signals** over the square local window `W_s(x,y) = {(u,v): |u-x| <= R_s AND |v-y| <= R_s}` clipped to the scale domain:

##### (a) Local Sharpness Signal `Phi_sharp`

`Phi_sharp_s(x,y) = Var_{p in W_s_valid(x,y)}(Lap_valid(I_s)(p))`

where `Lap_valid(I)(p) = I(p) - mean(I(q))` over source-valid 4-connected axial neighbors `q` of `p`. It requires a valid center and at least two valid axial neighbors; otherwise it is invalid. `Var` is the population variance `sum_i (x_i - mean(x))^2 / n` of finite Laplacian responses in the window. No mirrored, replicated, or zero-filled samples are used. The result is clamped to `[0, +inf)`.

No explicit global rescaling of `Phi_sharp_s` is applied here. For a non-degenerate population, the robust z-score in §1.4 is invariant to positive global scaling. Degenerate populations map to zero z-score by definition. Therefore a `sigma_Lap` normalization at this step would not change `Psi_s` and is omitted.

##### (b) Local SNR Signal `Phi_snr`

`b_s(x,y) = B_s(x,y; R_s) = median_{p in W_s_valid(x,y)} I_s(p)`
`mu_s(x,y) = mean_{p in W_s_valid(x,y)}(max(I_s(p) - b_s(x,y), 0))`
`sigma_s(x,y) = MAD_{p in W_s_valid(x,y)}(I_s(p)) * 1.4826`

`Phi_snr_s(x,y) = mu_s(x,y) / max(sigma_s(x,y), eps_noise(I_s over W_s_valid(x,y)))`

The clamp is applied only for the mean term. The SNR noise scale `sigma_s` uses the raw finite pixel values `I_s(p)` in the local support, not the background-subtracted or clamped signal term, so noise scale estimation is not biased by positive-signal clipping. `B_s` is a deterministic masked median over the same valid local support `W_s_valid(x,y)` used by the other local statistics. If `W_s_valid(x,y)` is empty, the signal is invalid under §2.3.4. If fewer than three valid pixels are available, implementations may fall back to `mean(max(I_s(p), 0))`, but must set the diagnostic flag `scene_dependent_snr = true`.

Scene-dependence guard: `Phi_snr_s` is a local support-quality proxy, not a source detector. The background-centered definition is intended to reduce source-content bias; fallback to the non-centered mean is allowed only as a diagnostic-marked degraded path.

##### (c) Artifact Anomaly Score `Phi_artifact`

`Phi_artifact_s(x,y)` detects local outlier gradients that indicate satellite trails, cosmic rays, or cloud edges:

1. Compute the high-pass residual `hp_s(x,y) = I_s(x,y) - blur(I_s, R_s)(x,y)`, where `blur(I_s, R_s)(x,y)` is the masked local mean of `I_s` over `W_s_valid(x,y)`.
2. Define `H_s_valid(x,y) = {p in W_s_valid(x,y) | hp_s(p) is finite}`. Compute `tau_s(x,y) = max(1.4826 * MAD_{p in H_s_valid(x,y)}(hp_s(p)), eps_scale(hp_s over H_s_valid(x,y)))`.
3. Compute the local outlier fraction: `frac_out_s(x,y) = |{p in H_s_valid(x,y) : |hp_s(p)| > k_artifact * tau_s(x,y)}| / |H_s_valid(x,y)|`
   with normative default `k_artifact = 3.0`.
4. `Phi_artifact_s(x,y) = 1 - clip(frac_out_s(x,y) / frac_artifact_max, 0, 1)`
   with normative default `frac_artifact_max = 0.25`.

`Phi_artifact_s = 1` indicates a clean region; `Phi_artifact_s = 0` indicates that at least `frac_artifact_max` (default 25%) of pixels in the window are outliers.

If `H_s_valid(x,y)` is empty, `Phi_artifact_s(x,y)` is invalid. If it contains fewer than three samples, `Phi_artifact_s(x,y) = 1` and the insufficient-support condition is recorded; sparse support must not be converted into a false artifact veto.

Using `|H_s_valid|` rather than `|W_s_valid|` is binding: only samples for which an outlier decision can actually be made belong in the denominator. This differs intentionally from v0.1.0 and avoids treating an invalid high-pass response as an inlier.

#### 2.3.3 Per-Scale Quality Map

The per-scale quality map `Psi_s(x,y)` is defined as:

`Psi_s(x,y) = sigmoid(w_sharp * z(Phi_sharp_s) + w_snr * z(Phi_snr_s)) * Phi_artifact_s(x,y)`

where:
- `z(Phi)` is the deterministic robust z-score from §1.4
  applied over all finite, source-valid pixels at scale `s`
- `sigmoid(v) = 1 / (1 + exp(-v))`
- `w_sharp`, `w_snr` are configurable weights (normative defaults: `w_sharp = 0.6`, `w_snr = 0.4`)
- The artifact term `Phi_artifact_s` acts as a multiplicative gate: it suppresses any region with excessive outlier density regardless of its sharpness or SNR.

Binding constraint: `Psi_s(x,y) ∈ [0, 1]` for all finite inputs. The sigmoid term is strictly positive, but the multiplicative artifact gate may set `Psi_s` to exactly zero.

**Within-frame relativity (binding clarification):** Because `z(Phi_sharp_s)` and `z(Phi_snr_s)` are normalized per frame (median/MAD computed over that frame's pixels at scale `s`), the **sigmoid factor of `Q_map` is a within-frame relative quality field**, not an absolute cross-frame quality measurement. Two frames of differing global seeing produce similarly distributed sigmoid factors, each self-normalized. Consequently, in the AQMH reconstruction weight `W_{f,c}^{aqmh} = G_{f,c} * Q_map_{f,c}` (§1.1, §4.3):

- **Between-frame** discrimination at a given pixel is carried by the global weight `G_{f,c}` and by the **absolute** artifact gate `Phi_artifact_s` (which is not z-scored and can drive `Q_map` toward zero in any frame independently of other frames).
- **Within-frame** spatial discrimination (which regions of a single frame are sharper / cleaner) is carried by the sigmoid factor.

This division of labor is intentional and binding: the sigmoid factor must not be interpreted as an absolute photometric quality and must not be used to rank whole frames.

#### 2.3.4 Boundary and Empty-Window Rules

All local statistics are computed over finite, source-valid pixels only. Let `W_s_valid(x,y)` be the valid subset of the analysis window.

If `|W_s_valid(x,y)| = 0`, all per-scale signals at `(x,y)` are marked invalid and the fused `Q_map` value is later set to zero by the output guard. If `|W_s_valid(x,y)| > 0` but fewer than three valid pixels are available, robust scale estimates use the dimension-preserving `eps_scale` guard and local variance estimates fall back to zero. For `Phi_snr_s`, the background-centering fallback rule in §2.3.2(b) is more specific and takes precedence.

The default boundary mode for convolution-like operations (`Lap_valid`, `blur`, local windows, and morphology support masks) is valid-only masked evaluation. Implementations must not mirror, replicate, zero-fill, or otherwise synthesize samples outside `C` or `M_f` into a statistic. If a library primitive cannot express this support rule directly, implementations must compute numerator and valid-support denominator separately or use an explicit masked operator.

### 2.4 Multi-Scale Fusion

Let `S_actual` be the ordered set of scales that are actually computed after applying the omission rule in §2.3.1, and let `P_actual = |S_actual|`. Upsample each computed `Psi_s` to the full canvas resolution using mask-aware bilinear interpolation:

`Psi_s^{up}(x,y) = upsample_valid(Psi_s, valid_s, D_s)`

where `valid_s` is the finite valid-support mask of `Psi_s`. `upsample_valid` interpolates the numerator `Psi_s * valid_s` and the support mask `valid_s` separately, then divides by the interpolated support. If the interpolated support is zero at a canvas pixel, `Psi_s^{up}` is invalid at that pixel. Invalid scale samples must not be treated as zero during interpolation because that would depress neighboring valid canvas pixels.

Fuse via **geometric mean** over the `P_actual` computed scales:

`Q_map_{f,c}(x,y) = ( prod_{s in S_actual} Psi_s^{up}(x,y) )^{1/P_actual}`

For numerical evaluation, first handle exact zeros as specified below; otherwise compute `Q_map = exp(mean_s(log(Psi_s^{up})))` to avoid product underflow.

The configured `P` is an upper bound. It must not be used as the exponent denominator when one or more scales are omitted. If `P_actual = 0`, `Q_map` is defined as zero everywhere after the output guard.

Geometric mean is chosen over arithmetic mean because it strongly penalizes disagreement between scales: high fused quality requires every scale to be at least moderately high. A single low scale can drive the result near zero, and an exact zero produces the hard veto below; nonzero disagreement is still a soft penalty rather than a logical all-scales predicate.

**Zero-scale guard:** If `Psi_s^{up}(x,y) = 0` for any scale `s`, then `Q_map_{f,c}(x,y) = 0` exactly (one bad scale vetos the pixel).

**Output guard (binding):** Set `Q_map_{f,c}(p) = 0` wherever `C(p) = 0` or `M_f(p) = 0`, overriding any computed value.

If any computed scale is invalid at a source-valid pixel because there is no valid scale support after mask-aware upsampling, that scale contributes a zero-veto at that pixel. Pixels outside `C` or `M_f` are excluded and zeroed only by the final output guard.

### 2.5 Block-Level Diagnostic Summaries

For reports and visual summaries, AQMH may derive block-level diagnostic values by aggregating `Q_map_{f,c}` over a display block `b`:

`Q_{f,b,c}^{aqmh} = median_{p in b, source-valid} Q_map_{f,c}(p)`

The block grid is a reporting/visualization aid only. It is not part of the AQMH reconstruction weight model, and it must not introduce block-constant weights into the AQMH accumulator.

---

## 3. Quality Map Storage and Memory Model

### 3.1 Storage Format

Conceptually, `Q_map_{f,c}` is a full-canvas quality field, one per frame and analysis channel. The persisted representation may be lower resolution or quantized according to §3.2. In an explicitly configured CFA-proxy mode, the proxy is the sole analysis channel: its `Q_map` and `G` are reused unchanged for every reconstructed output channel. Artifacts must record `analysis_channel: proxy` and the output-channel mapping; they must not present copied values as independently measured channel diagnostics.

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
| Full resolution float32 | No spatial loss; preserves pixel-local values and exact zero-vetos | **Default** |
| 1/4 area float32 | Downscale by 2 in each axis | Optional approximate mode |
| uint16 quantization | Map scaled to `[0, 65535]` | Optional, recommended performance format when bit-identical float32 cache values are not required |
| uint8 quantization | Map scaled to `[0, 255]` | Optional, smaller cache with higher quantization error |
| Block-compressed float16 | Per-block float16 sub-blocks | Future optional; only valid when explicitly implemented |

**Normative default:** Store `Q_map` at full resolution (`resolution_divisor = 1`). This is required for the stated pixel-resolution objective and preserves pixel-local defects and exact zero-vetos.

`resolution_divisor > 1` is an explicitly approximate storage mode. It must use mask-aware area downsampling and mask-aware bilinear reconstruction, and must additionally store a full-resolution one-bit zero-veto mask. After upsampling, vetoed pixels are reset to exactly zero. Reports must record `spatially_approximate_maps: true`; validation must quantify loss of isolated-defect detection against the full-resolution mode.

### 3.3 Disk Cache Lifecycle

Quality maps are written during AQMH map computation and consumed during AQMH reconstruction. Maps are invalidated when the source prewarped frame, `C`, `M_f`, or any **map-affecting** AQMH configuration changes (`pyramid`, `storage`, or map format version). Reconstruction-only settings must not invalidate map cache entries. Implementations should store and validate a cache metadata hash covering only map-affecting inputs.

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

AQMH_GLOBAL_QUALITY
  Finalize cross-frame AQMH global factors G from map diagnostics

AQMH_RECONSTRUCTION
  Perform pixel-wise weighted stacking with W_aqmh = G * Q_map

AQMH_DIAGNOSTICS
  Emit quality-map, reconstruction, and optional region artifacts

AQMH_NATIVE_BGE_INPUTS
  Optional postprocessing support: derive BGE sampling helpers from the AQMH
  reconstruction output and global output mask C, not from Classic Tile Compile local
  metrics.
```

The stage order `AQMH_MAPS -> AQMH_GLOBAL_QUALITY -> AQMH_RECONSTRUCTION` is binding. Global-quality finalization must not start until every eligible frame has emitted its required pre-z-score summaries, and reconstruction must not start until all `G_{f,c}` values are finalized and persisted.

Shared preprocessing and postprocessing stages may be reused, but Classic Tile Compile local metrics and tile reconstruction are not AQMH stages. If BGE is enabled for an AQMH run, BGE tile-sampling helpers must be derived from the AQMH reconstruction output and `C`. They may contain per-tile background, robust noise, and gradient/structure estimates for BGE sampling only; they are not AQMH reconstruction weights and must not be read from `local_metrics.json`.

### 4.2 AQMH Map Computation

For each frame `f` and channel `c` with `frame_has_data[f] = true`:

1. Load the prewarped frame `I_{f,c}` from `DiskCacheFrameStore`.
2. Apply `C` and the frame-specific `M_f`: set every sample with `C(p) = 0` or `M_f(p) = 0` to NaN.
3. For each `s in S_actual` after applying the omission rule in §2.3.1:
   a. Compute `I_s` via area-averaged downscaling with mask-aware denominator.
   b. Compute `Phi_sharp_s`, `Phi_snr_s`, `Phi_artifact_s` (§2.3.2).
   c. Compute `Psi_s` (§2.3.3).
4. Upsample all `Psi_s` to canvas resolution (§2.4).
5. Compute fused `Q_map_{f,c}` via geometric mean (§2.4).
6. Apply the output guard: set pixels outside `C` or `M_f` to zero.
7. Write `Q_map_{f,c}` to the AQMH quality-map disk cache (at configured storage resolution).

After all eligible frames have produced the pre-z-score summaries required by §1.1, compute `G_{f,c}` across frames for each analysis channel and persist both the summaries and finalized factors before reconstruction starts.

### 4.3 AQMH Pixel-Wise Weighted Reconstruction

For each pixel `p` with `C(p) = 1`:

Define the finite intensity sample set:

`V_c^{I}(p) = { f | C(p) = 1 AND M_f(p) = 1 AND I_{f,c}(p) is finite }`

Define the map-available sample set:

`V_c^{map}(p) = { f in V_c^{I}(p) | Q_map_{f,c}(p) is finite }`

For each `f in V_c^{I}(p)`, the effective AQMH pixel weight is:

`w_{f,c}^{aqmh}(p) = G_{f,c} * Q_map_{f,c}(p)` when `f in V_c^{map}(p)`

`w_{f,c}^{aqmh}(p) = 0` when the map sample is unavailable or non-finite

No Classic Tile Compile local/tile weight is used as an AQMH fallback.

Pixels with `C(p) = 0` are not reconstructed. They are written as unsupported/zero without evaluating frame samples, map samples, sigma clipping, or denominator fallback. Frame-invalid source pixels are never members of `V_c^I(p)`.

Define the reconstruction sample set `A_c(p)`. If cherry-pick mode is active at `p` (§5.3), `A_c(p)` is the retained top-`K(p)` subset of `V_c^{rank}(p)`; otherwise `A_c(p) = V_c^{I}(p)`. All sums, clipping operations, and denominator guards below operate on `A_c(p)`.

The reconstructed pixel value is:

`R_c^{aqmh}(p) = sum_{f in A_c(p)} w_{f,c}^{aqmh}(p) * I_{f,c}(p) / sum_{f in A_c(p)} w_{f,c}^{aqmh}(p)`

**Unsupported-pixel handling (binding):** Before applying any numerical denominator guard, implementations must distinguish finite map samples from unavailable map samples. A finite zero is a valid map sample and is an explicit veto, not a missing value. Let `D = sum_{f in A_c(p)} w_f`, `w_max = max_{f in A_c(p)} w_f`, and `eps_weight(p) = |A_c(p)| * machine_epsilon * w_max`. If `D <= eps_weight(p)`, fallback behavior depends on why the sum is zero:

1. If at least one finite map sample exists at `p` (`V_c^{map}(p) != empty`) and all AQMH weights are zero because the available maps explicitly veto the pixel (`Q_map = 0`), the output pixel is marked unsupported/zero. Do **not** replace the explicit zero-veto by an unweighted mean.
2. If no finite map sample exists at `p`, or all map samples are unavailable because of IO/cache failure, the output pixel is marked unsupported/zero and the run emits an AQMH cache/map-availability warning. AQMH must not silently switch to Classic Tile Compile weights.
3. Sigma clipping cannot remove all positive-weight samples because of the binding keep-floor below. If the post-clip denominator nevertheless fails its numerical guard, the pixel is unsupported/zero and a numerical warning is recorded.

#### 4.3.1 Deterministic Weighted Sigma Clipping

Sigma clipping is enabled by default with `clip_sigma = 3.0`, `clip_iterations = 3`, `min_fraction = 0.5`, and `min_n_eff = 2.0`. It operates only on finite samples in `A_c(p)` with strictly positive weights; zero-weight samples never influence clipping statistics.

For each iteration:

1. Compute the deterministic weighted median `m`: after sorting by `(intensity, frame_index)`, choose the first intensity whose cumulative weight is at least half the total weight.
2. Compute `sigma = 1.4826 * weighted_median(|I_f - m|)` using the same ordering convention. If `sigma <= eps_scale({I_f})`, retain samples with `I_f = m`; otherwise retain samples satisfying `|I_f - m| <= clip_sigma * sigma`.
3. Let `n0` be the positive-weight sample count before the first iteration and `n_keep_min = min(n0, max(1, ceil(min_fraction * n0)))`. If the rule would retain fewer than `n_keep_min`, retain the `n_keep_min` samples with smallest `(abs(I_f - m) / max(sigma, eps_scale({I_f})), frame_index)` instead.
4. Stop early when the retained set no longer changes.

After clipping, compute `D_eff = sum w_f` and `N_eff = D_eff^2 / sum w_f^2` over retained samples. The output is unsupported/zero if `D_eff <= n_retained * machine_epsilon * max(w_f)` or `N_eff < min_n_eff`; the failure reason is recorded. Otherwise apply the weighted-mean formula in §4.3 to the retained set. Cherry-pick selection occurs before clipping, so `k_min_required` is a pre-clip selection floor; `min_fraction` and `min_n_eff` independently govern the post-clip set.

---

## 5. Adaptive Region Extraction (Optional)

### 5.1 Motivation

In addition to the continuous weight map, AQMH can generate **binary quality regions** for diagnostic reporting and for the optional cherry-pick stacking mode.

### 5.2 Quality Contour Extraction

From the fused `Q_map_{f,c}`, extract binary regions by thresholding:

1. Compute the per-frame/channel threshold: `tau_{f,c} = quantile(Q_map_{f,c}, q_region)` over finite, source-valid pixels only, with normative default `q_region = 0.75`.
2. Binary region mask: `M_region_{f,c}(x,y) = 1 iff Q_map_{f,c}(x,y) >= tau_{f,c} AND C(x,y) = 1 AND M_f(x,y) = 1`.
3. Apply morphological opening with radius `r_morph_canvas_px` to remove isolated noise regions. The normative default is a canvas-equivalent radius of 6 px. If region extraction is run on a stored/downscaled map, use `r_morph_map = max(1, round(r_morph_canvas_px / resolution_divisor))`. Morphology is constrained to the valid support; the final region mask is intersected with both `C` and `M_f`.
4. Extract connected components; label each component with:
   - `Area_r`: pixel count
   - `MeanQ_r`: mean quality score over the region
   - `Compactness_r = 4*pi*Area_r / Perimeter_r^2` (Polsby-Popper score)
5. Rank regions by `Score_r = MeanQ_r * log(1 + Area_r)`.

These regions are reported in `aqmh_regions.json` per frame and channel.

### 5.3 Cherry-Pick Stacking Mode

When `aqmh.cherry_pick.enabled = true`, per-pixel stacking may use only the **top-K frames** by quality, rather than all frames. Cherry-pick trades the no-frame-selection invariant (§1.2.1) for a potential per-pixel quality gain. A retained-sample floor reduces variance and rejection risk, but cannot prove a net quality gain or full-frame-equivalent defect rejection; that requires empirical validation (§9).

Below a minimum sample count, cherry-pick has an increased risk of **fragment artifacts**: visible per-frame noise grain, incompletely rejected hot pixels or cosmic ray remnants, and a non-constant local background level, because fewer independent samples remain to average down photon and read noise or to reject a residual defect. This subsection defines conservative binding thresholds intended to reduce this risk.

#### 5.3.1 Baseline Top-K Selection

For each pixel `p`, define `S_f(p) = G_{f,c} * Q_map_{f,c}(p)`. A frame is rankable only if its intensity and score are finite and `S_f(p) > 0`; let `V_c^{rank}(p)` be this set and `N_rankable(p) = |V_c^{rank}(p)|`. Frames with unavailable/non-finite maps or non-positive scores are not rankable. Sort `V_c^{rank}(p)` by `S_f(p)` descending, using ascending frame index as the deterministic tie-breaker. Weighted reconstruction proceeds over the retained top-`K(p)` frames only, as defined in §5.3.3. If cherry-pick degrades to full-set reconstruction at a pixel, reconstruction uses `V_c^{I}(p)` with the ordinary zero-weight and unavailable-map semantics of §4.3.

#### 5.3.2 Selection-Size Eligibility Floor

Eligibility for cherry-pick must not be gated on the total number of input frames scheduled for the run, because that number does not by itself tell us how many frames would actually survive top-K selection at each pixel. Instead, eligibility is gated directly on the **nominal number of frames that selection would actually retain**.

Define, for each pixel with `C(p) = 1`, the unclamped nominal retained count:

`K_nominal(p) = floor(k_frac(p) * N_rankable(p))`

and the run-level aggregate:

`K_nominal_median = median_{p: C(p)=1}(K_nominal(p))`

with the normative binding minimum:

`k_min_required = 20`

**Binding rule:** If cherry-pick is configured and `K_nominal_median < k_min_required`, it is forced disabled for the entire run. The run must record `cherry_pick_forced_disabled: true` and the computed `k_nominal_median` in `aqmh_metrics.json`, and emit a `WARNING` log message. Reconstruction then proceeds identically to a `cherry_pick.enabled = false` run. If cherry-pick was not configured, `cherry_pick_forced_disabled` is `false`.

**Rationale:** `k_min_required = 20` is a conservative empirical policy floor, not a consequence of a universal MAD error formula. Sampling error depends on the estimator and the underlying distribution; the common `1/sqrt(K)` scaling alone equals about 22% at `K = 20`, not 16%. Gating on `K_nominal_median` instead of on total input frame count is deliberate: two runs with the same total frame count can have very different numbers of rankable samples per pixel (dithering coverage, local saturation, missing maps, or zero quality scores), so the input count alone is not a reliable predictor of whether selection is safe. With tiering disabled, a run with 30 rankable frames and base `k_frac = 0.30` yields `K_nominal(p) = floor(0.30 * 30) = 9`, below the floor; with `k_frac = 0.80`, it yields 24 and may pass the aggregate gate.

#### 5.3.3 Per-Pixel Effective-Sample Floor

Even when the run-level selection-size floor (§5.3.2) is satisfied on aggregate, per-pixel data loss can locally reduce `N_rankable(p)` well below the pixels used for `K_nominal_median`. The top-K rule is therefore:

`K(p) = max(k_min_required, K_nominal(p))` when `N_rankable(p) >= k_min_required`; otherwise cherry-pick is inactive at `p` and all samples in `V_c^I(p)` are used.

Because `K_nominal(p) <= N_rankable(p)` for the required configuration range `0 < k_frac(p) <= 1`, this rule retains between the floor and all rankable samples. Whenever cherry-pick is active, at least `k_min_required` strictly positive-weight samples contribute. If fewer rankable samples exist, cherry-pick degrades to the full AQMH weighted mean at that pixel.

`k_min_required = 20` remains a normative, empirically chosen default that must be validated against representative datasets. It must not be presented as a mathematically sufficient threshold for artifact-free reconstruction.

#### 5.3.4 Tiered `k_frac` Schedule (Optional Refinement)

Applying a single global `k_frac` together with the `k_min_required` floor can force near-100% retention at the low end of the eligible sample-count range (defeating the purpose of cherry-pick) while being unnecessarily conservative at the high end. Implementations may configure a tiered schedule based on `N_rankable(p)`:

| `N_rankable(p)` range | `k_frac(p)` (example tier) | Typical `K(p)` |
|---|---|---|
| 0 – 19 | cherry-pick inactive at this pixel; use full set (§5.3.3) | n/a |
| 20 – 59 | base `k_frac` (no matching tier) | exactly 20 at the default `k_frac = 0.30`; potentially more with a higher configured base |
| 60 – 99 | 0.60 | >= 36 |
| 100 – 199 | 0.45 | >= 45 |
| 200 – 399 | 0.35 | >= 70 |
| >= 400 | 0.30 (base default) | >= 120 |

Tiering is optional and disabled when `tiered_k_frac` is absent or empty. If disabled, the base `k_frac` applies uniformly. If enabled, choose the matching tier with the greatest `min_n_rankable <= N_rankable(p)` and set `k_frac(p) = max(base_k_frac, tier_k_frac)`; if no tier matches, the base value applies. Tiering never overrides the floor. All fractions must satisfy `0 < k_frac <= 1`.

#### 5.3.5 Rank-Separation Diagnostic

At small `K(p)`, adjacent-rank score differences can be dominated by measurement noise in `Q_map`, producing a selection that is not meaningfully different from a random choice among near-tied frames, which manifests as pixel-to-pixel inconsistent inclusion (visible seam-like or salt-and-pepper texture). For each pixel where cherry-pick is active, define the rank margin:

`margin(p) = (S_{(K)}(p) - S_{(K+1)}(p)) / S_{(1)}(p)`

the score gap between the last retained and first excluded rankable frames, normalized by the best score at that pixel. This makes the margin dimensionless and invariant to a common rescaling of `G`. `margin(p)` is undefined/skipped if `K(p) = N_rankable(p)` or cherry-pick is inactive at `p`. If `median_p(margin(p))` over pixels with `C(p) = 1` where cherry-pick is active falls below `margin_min` (normative default: `0.02`), the run records `low_rank_separation: true` in `aqmh_metrics.json`. This is diagnostic-only: it does not block reconstruction, but signals that cherry-pick is unlikely to provide a meaningful quality gain over the full weighted mean for this dataset.

#### 5.3.6 Worked Examples

**30-rankable-frame run, tiering disabled, default `k_frac = 0.30`:** `K_nominal(p) = floor(0.30 * 30) = 9` at pixels with full rankable coverage, so `K_nominal_median = 9 < k_min_required = 20`. Cherry-pick is forced disabled and the run falls back to the standard full pixel-wise weighted mean (§4.3).

**30-rankable-frame run, tiering disabled, `k_frac = 0.80`:** `K_nominal(p) = floor(0.80 * 30) = 24`, so the aggregate may pass the floor. The retained count is 24 at a fully covered pixel; whether this improves the result remains an empirical question.

**250-frame run:** for a pixel with `N_rankable(p) = 240`, the example tiered schedule (§5.3.4) selects `k_frac(p) = 0.35`, giving `K_nominal(p) = 84` and therefore `K(p) = 84`.

**Edge pixel in a large, eligible run:** if `N_rankable(p) = 15`, cherry-pick is inactive at that pixel and reconstruction uses the full `V_c^I(p)` set under §4.3, including its ordinary handling of zero-weight or unavailable-map samples.

#### 5.3.7 Binding Summary

Cherry-pick mode violates the default AQMH no-frame-selection invariant at pixel level, even though it does not discard entire frames. It must only be used when explicitly enabled by the user, is subject to the run-level selection-size floor (§5.3.2) and the per-pixel floor (§5.3.3) — both driven by the single constant `k_min_required` — and must be clearly flagged in diagnostic output. Default is `disabled`.

---

## 6. Quality Map Diagnostics

### 6.1 Per-Frame Diagnostics

For each processed `(frame f, channel c)` pair, the following scalar diagnostics are written to `aqmh_metrics.json`. Implementations may additionally report a clearly named channel aggregate, but it must not replace the per-channel values:

| Field | Definition |
|---|---|
| `map_mean` | Mean of `Q_map_{f,c}` over source-valid pixels |
| `map_p10` | 10th percentile over source-valid pixels |
| `map_p90` | 90th percentile over source-valid pixels |
| `artifact_frac` | Fraction of source-valid pixels with `Q_map_{f,c} < tau_artifact` (normative default: `tau_artifact = 0.2`) |
| `sharpness_p50` | Median of the **pre-z-score** `Phi_sharp_0` at scale 0 |
| `snr_p50` | Median of the **pre-z-score** `Phi_snr_1` at scale 1 |
| `n_regions` | Number of quality regions (§5.2) above threshold |
| `global_quality` | `G_{f,c}` from §1.1 |
| `global_sharpness_input`, `global_snr_input` | Pre-z-score summaries used to derive `G_{f,c}` |
| `global_quality_input_invalid` | `true` if either global summary required the zero-z-score fallback |

The following run-level fields are also written to `aqmh_metrics.json` (they are not per-frame scalars):

| Field | Definition |
|---|---|
| `cherry_pick_forced_disabled` | `true` if the run-level selection-size floor (§5.3.2) forced cherry-pick off despite being configured |
| `cherry_pick_active` | Effective run-level state after the eligibility gate; `true` only when selection can be active at eligible pixels |
| `k_nominal_median` | Run-level `K_nominal_median` computed for the eligibility check (§5.3.2) |
| `k_effective_p10`, `k_effective_p50`, `k_effective_p90` | 10th/50th/90th percentile of `K(p)` over pixels with `C(p) = 1` where cherry-pick is active; `null` if there are none |
| `low_rank_separation` | `true` if the rank-margin diagnostic (§5.3.5) fell below `margin_min`; `null` if no margins are defined |

If the referenced diagnostic scale is omitted by the small-image scale rule (§2.3.1), the corresponding diagnostic field is written as `NaN` or `null` and the artifact must also record that the scale was unavailable. With normative defaults, `sharpness_p50` is normally available because scale 0 has `D=1`; `snr_p50` remains unavailable when scale 1 is omitted. This differs intentionally from computational `g_snr`, which must use its §1.1 fallback so that `G` remains defined.

### 6.2 Block-Level Diagnostics

For each report block `b`, the following values may be reported:

- `aqmh_q_median`: `Q_{f,b,c}^{aqmh}` as defined in §2.5
- `aqmh_q_p10`, `aqmh_q_p90`: 10th and 90th percentile within the block
- `aqmh_artifact_frac`: fraction of pixels in the block with `Q_map < tau_artifact`

### 6.3 Heatmaps

For integration into the report generator, AQMH emits spatial heatmap entries for the `aqmh_metrics.json` artifact:

- Mean `Q_map` per report block, per frame and analysis channel
- Artifact fraction heatmap per report block, per frame and analysis channel
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
    k_artifact: 3.0     # outlier detection threshold (default: 3.0)
    frac_artifact_max: 0.25  # artifact gate threshold (default: 0.25)
  global_quality:
    g_floor: 0.05       # strictly positive frame-level weight floor
    g_w_sharp: 0.6      # global sharpness contribution
    g_w_snr: 0.4        # global SNR contribution
```

### 7.3 Storage Configuration

```yaml
aqmh:
  storage:
    resolution_divisor: 1   # default: full resolution; values >1 are explicitly approximate
    dtype: float32          # float32 | uint16 | uint8 (default: float32)
    max_resident_maps: 2    # bounded read-through cache during reconstruction; 0 disables
```

The storage default (`resolution_divisor = 1`, `dtype = float32`) is spatially lossless. `max_resident_maps` bounds how many full-resolution maps may be held in RAM simultaneously during AQMH reconstruction; it must not scale with frame count.
`uint16` is the recommended performance cache format when bit-identical float32 cache values are not required. It quantizes `Q_map` to `[0,65535]`, preserves exact zero-veto and full-quality endpoints, and has a maximum quantization error of about `7.7e-6`.

**Migration from v0.1.0:** The default changed from `resolution_divisor = 2` to `1`. Existing configurations that explicitly specify `2` retain approximate quarter-area behavior and must satisfy the veto-mask requirements of §3.2. Existing v0.1.0 map caches lack that mask and must be invalidated and rebuilt; the v0.2.0 map-format version must therefore differ from v0.1.0.

### 7.4 Cherry-Pick Mode

```yaml
aqmh:
  cherry_pick:
    enabled: false             # must be explicitly enabled; breaks no-frame-selection invariant at pixel level
    k_frac: 0.30
    k_min_required: 20         # hard floor on retained samples, used both as run-level gate and per-pixel floor (default: 20)
    margin_min: 0.02           # rank-separation diagnostic threshold (default: 0.02)
    tiered_k_frac: []          # optional; example schedule shown in §5.3.4
```

If the run-level `K_nominal_median` (§5.3.2) falls below `k_min_required`, cherry-pick is force-disabled at runtime even if `enabled: true` is configured.

### 7.5 Diagnostics Configuration

```yaml
aqmh:
  diagnostics:
    tau_artifact: 0.20  # quality threshold for the artifact_frac diagnostic (default: 0.20)
    q_region: 0.75      # quantile threshold for quality-region extraction (default: 0.75)
    r_morph_canvas_px: 6 # canvas-equivalent radius for quality-region morphology (default: 6)
```

`tau_artifact` is a **diagnostic-only** threshold (see §6.1, §6.2). It does not affect reconstruction weights or the per-scale artifact gate `Phi_artifact_s` (which is governed by `k_artifact` and `frac_artifact_max` in §7.2).

### 7.6 Reconstruction Configuration

```yaml
aqmh:
  reconstruction:
    clip_sigma: 3.0
    clip_iterations: 3
    min_fraction: 0.50
    min_n_eff: 2.0
```

These parameters have the binding semantics defined in §4.3.1. Required ranges are `clip_sigma > 0`, integer `clip_iterations >= 0`, `0 < min_fraction <= 1`, and `min_n_eff >= 1`.

### 7.7 Validation Configuration

```yaml
aqmh:
  validation:
    max_seam_score_regression: 0.02
    max_fwhm_regression: 0.02
    max_background_rms_regression: 0.02
```

These are maximum relative regressions against a uniform-weight control reconstructed from the same preprocessed samples with the same validity masks and sigma-clipping rules. For a lower-is-better metric `m`, define `regression = (m_aqmh - m_control) / max(abs(m_control), eps_scale({m_aqmh, m_control}))`. The validation passes when the regression does not exceed its configured limit. The control is AQMH-native test infrastructure, not a Classic Tile Compile output.

---

## 8. Numerical Defaults

AQMH uses the dimension-preserving `eps_scale(X)` convention from §1.4 and a separate floating-point weight-sum guard from §4.3. It does not use one dimensionless constant as the denominator floor for unrelated physical quantities.

| Parameter | Default | Description |
|---|---|---|
| `eps_rel` | `1e-6` | Relative factor in the dimension-preserving `eps_scale(X)` guard |
| `k_artifact` | `3.0` | Outlier sigma multiplier |
| `frac_artifact_max` | `0.25` | Maximum tolerated outlier fraction per window |
| `w_sharp` | `0.6` | Sharpness weight in per-scale quality sigmoid |
| `w_snr` | `0.4` | SNR weight in per-scale quality sigmoid |
| `g_floor` | `0.05` | Strictly positive lower bound for `G_{f,c}` |
| `g_w_sharp` | `0.6` | Global sharpness contribution to `G_{f,c}` |
| `g_w_snr` | `0.4` | Global SNR contribution to `G_{f,c}` |
| `P` | `4` | Maximum number of pyramid scales; actual count may be lower due to the omission rule in §2.3.1 |
| `R_s` | `4` | Window radius at each scale (in downscaled pixels) |
| `q_region` | `0.75` | Quality quantile threshold for region extraction |
| `r_morph_canvas_px` | `6` | Morphological opening radius in canvas pixels; map-space radius is `max(1, round(r_morph_canvas_px / resolution_divisor))` |
| `k_frac` | `0.30` | Cherry-pick frame fraction (base default; see tiered schedule §5.3.4) |
| `k_min_required` | `20` | Hard floor on retained samples: gates run-level eligibility via `K_nominal_median` (§5.3.2) and clamps per-pixel `K(p)` (§5.3.3) |
| `margin_min` | `0.02` | Rank-separation diagnostic threshold for cherry-pick (§5.3.5) |
| `tau_artifact` | `0.20` | Quality threshold for artifact fraction diagnostic |
| `max_resident_maps` | `2` | Max full-resolution maps held in RAM during reconstruction |
| `resolution_divisor` | `1` | Full-resolution map storage; values above 1 enable approximate mode |
| `clip_sigma` | `3.0` | Weighted-MAD clipping threshold |
| `clip_iterations` | `3` | Maximum clipping iterations |
| `min_fraction` | `0.50` | Minimum retained fraction after clipping |
| `min_n_eff` | `2.0` | Minimum post-clip effective sample count |
| `max_seam_score_regression` | `0.02` | Maximum relative seam-score regression against the §7.7 control |
| `max_fwhm_regression` | `0.02` | Maximum relative FWHM regression against the §7.7 control |
| `max_background_rms_regression` | `0.02` | Maximum relative background-RMS regression against the §7.7 control |

---

## 9. Validation Requirements

When AQMH is enabled, the following validation requirements apply:

1. **Map range:** `Q_map_{f,c}(p) ∈ [0, 1]` for all finite source-valid pixels.
2. **Output guard:** `Q_map_{f,c}(p) = 0` wherever `C(p) = 0` or `M_f(p) = 0`.
3. **Determinism:** Identical registered frames, `C`, and `M_f` masks produce identical quality maps.
4. **Unsupported coverage:** Every pixel with `V_c^{I}(p) = empty` returns zero/unsupported; every pixel with finite intensity samples but no finite AQMH map samples returns zero/unsupported with an AQMH warning; no NaN/Inf in output.
5. **Explicit zero-veto:** If finite maps exist at a pixel and all available AQMH weights are zero, the output remains unsupported/zero and must not be replaced by an unweighted mean.
6. **Block diagnostic consistency:** `Q_{f,b,c}^{aqmh}` matches `median(Q_map over b)` within floating-point tolerance.
7. **No structural injection:** Seam score, FWHM, and background RMS must satisfy the explicit control-run definition and configured limits in §7.7.
8. **Artifact detection:** Known satellite-contaminated frames show elevated `artifact_frac > 0.01` for at least the contaminated report blocks.
9. **Scale omission:** For an input where `P_actual < P` (for example `min(W,H) < 64` with defaults), fusion uses `P_actual` as the geometric-mean denominator, omitted scales are recorded in diagnostics, and unavailable diagnostic scales are written as `NaN`/`null`.
10. **Cherry-pick flag:** When `cherry_pick.enabled = true`, `aqmh_metrics.json` must contain `cherry_pick_active`, reflecting the effective runtime state. If it is active, the pipeline log must emit a `WARNING` level message.
11. **Cherry-pick selection-size floor:** If cherry-pick is configured and `K_nominal_median < k_min_required`, cherry-pick must be forced off internally, `cherry_pick_forced_disabled: true` and `k_nominal_median` must be recorded, and reconstruction must be bit-for-bit equivalent to a `cherry_pick.enabled = false` run.
12. **Cherry-pick effective-sample floor:** For every pixel with `C(p) = 1` where cherry-pick is active, `K(p) >= k_min_required` and all retained scores are finite and strictly positive.
13. **Cherry-pick graceful degradation:** Where `N_rankable(p) < k_min_required`, cherry-pick is inactive at that pixel and reconstruction is identical to the ordinary full-set AQMH reconstruction in §4.3.
14. **Global quality:** Every eligible frame has finite `G_{f,c}` with `g_floor < G_{f,c} < 1`; identical global inputs produce identical `G`, and no frame is removed through `G = 0`.
15. **Storage fidelity:** The default full-resolution cache reproduces computed `Q_map` values exactly within float32 roundoff. Any `resolution_divisor > 1` run records approximate mode, preserves the full-resolution zero-veto mask exactly, and reports isolated-defect recall relative to full resolution.
16. **Channel diagnostics:** Every processed `(f,c)` pair has distinct map and global-quality diagnostics; any aggregate is separately named.

---

## 10. Scope Boundary

### Shared Infrastructure

- Input scan, calibration, registration/prewarping, global normalization, global output mask `C`, frame-valid masks `M_f`, run management, logging, reports

### AQMH Method

- Dense quality map computation
- Pixel-wise weighted AQMH reconstruction
- Adaptive region extraction (§5)
- Cherry-pick stacking with run-level and per-pixel sample-count floors (§5.3)
- AQMH diagnostic artifacts

### Explicitly Out of Scope

The following are not part of this specification and remain candidate future extensions, not covered by any binding rule here:

- Sub-pixel super-resolution or drizzle-style reconstruction.
- Cross-frame PSF matching or deconvolution prior to fusion.
- An absolute (non-frame-relative) spatial quality measurement independent of `G_{f,c}`.
- Joint multi-frame statistical modeling beyond independent per-pixel weighted combination and sigma clipping.

---

## 11. Core Statement

AQMH is an independent deterministic weighted-mean reconstruction method. It uses a continuous per-pixel quality field rather than block-constant local weights. Every pixel that enters the reconstruction accumulator does so with a deterministic non-negative AQMH weight that reflects both global atmospheric conditions and local spatial quality, without artificial block boundaries and without relying on Classic Tile Compile weights or fallback behavior. Its one optional deviation from full-frame-set reconstruction, cherry-pick stacking, is bounded by explicit run-level and per-pixel sample-count floors and diagnostics; improvement over full-set reconstruction remains an empirical validation requirement.
