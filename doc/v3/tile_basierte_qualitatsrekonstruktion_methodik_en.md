# Tile‑based Quality Reconstruction for DSO – Methodology v3

**Status:** Reference specification (Single Source of Truth)
**Version:** v3.1 (2026-01-09)
**Replaces:** Methodology v2
**Objective:** Clear, unambiguous workflows for two allowed registration + preprocessing paths
**Applies to:** `tile_compile.proc` (Clean Break) + `tile_compile.yaml`

---

## 0. Motivation for v3

Methodology v2 defined the quality and reconstruction logic precisely, but left the **preprocessing path (OSC, registration, channel handling)** implicit.

Methodology v3 makes this **explicit** and separates two **equivalent, but different** paths:

- **A – Siril-based path** (proven, low risk)
- **B – CFA-based path** (methodically maximally clean, higher implementation effort)

From **Phase 2 (tile generation)** both paths are **identical**.

---

## 1. Objective

The goal is to reconstruct a **spatially and temporally optimally weighted signal** from fully registered, linear short‑exposure frames of astronomical deep‑sky objects.

The method explicitly models two orthogonal influences:

- **global atmospheric quality** (transparency, haze, background drift)
- **local seeing‑ and structure‑driven quality** (sharpness, detail carrying capacity)

There is **no frame selection**. Every frame contributes according to its physical information content.

---

## 2. Assumptions (mandatory)

### 2.1 Hard assumptions (violation → abort)

- Data are **linear** (no stretch, no non‑linear operators)
- **No frame selection** (pixel-level artifact rejection is allowed)
- Channel-separated processing (no channel coupling)
- Strictly linear pipeline (no feedback loops)
- Uniform exposure time across frames (tolerance: ±5%)

### 2.2 Soft assumptions (with tolerances)

| Assumption | Optimal | Minimum | Reduced mode |
|-----------|---------|---------|--------------|
| Frame count | ≥ 800 | ≥ 50 | 50–199 |
| Registration residual | < 0.3 px | < 1.0 px | warning if > 0.5 px |
| Star elongation | < 0.2 | < 0.4 | warning if > 0.3 |

### 2.3 Implicit assumptions (now explicit)

- Stable optical configuration (focus, field curvature)
- Tracking error < 1 pixel per exposure
- No systematic drift during the session

### 2.4 Reduced mode (50–199 frames)

If the frame count is below optimal but above minimum:

- State-based clustering is skipped
- Synthetic frames are skipped
- Emit a validation warning in the report

**Reduced Mode workflow (binding):**

1. Steps 1–7 are executed normally (incl. tile-based reconstruction)
2. Steps 8–9 are skipped
3. Step 10 stacks the **reconstructed result from step 7** directly

```
R_c = reconstructed image from step 7
```

Alternative (only if clustering is explicitly kept enabled):

- reduce cluster count to 5–10
- generate synthetic frames with reduced cluster count

**Gradual degradation (instead of hard abort):**

| Severity | Action | Example |
|----------|--------|---------|
| warning | continue with note | registration residual 0.5–1.0 px |
| degraded | enable fallback mode | < 50 frames → reduced mode without clustering |
| critical | abort with explanation | no stars found, data not linear |

Only **critical** errors (data integrity violated) abort the run.

---

## 3. Full pipeline (normative)

1. Registration of raw frames
2. Channel split (R/G/B or mono)
3. **Global linear normalization (mandatory, once)**
4. Computation of global frame metrics
5. Seeing‑adaptive tile geometry
6. Local tile metrics and weighting
7. Tile‑wise reconstruction (overlap‑add)
8. **State‑based clustering of frames** (optional; skipped in reduced mode)
9. Reconstruction of synthetic quality frames (optional; skipped in reduced mode)
10. Final linear stacking (per channel)
11. **Combination (RGB / LRGB) – outside the methodology**
12. Validation and abort decision

The pipeline is **strictly linear**. There are no feedback loops.

---

# A. Siril-based path (reference, recommended)

## A.1 Purpose and positioning

The Siril path uses Siril’s **proven registration and debayer logic**. The methodology applies **only afterwards**.

This path is:

- stable
- reproducible
- low-risk
- recommended for production

---

## A.2 Steps A.1–A.2 (Siril)

### A.2.1 Debayer + registration (Siril)

- input: raw OSC frames
- Siril performs:
  - debayer (interpolation)
  - star detection
  - transform estimation
  - rotation / translation

**Result:** registered, debayered RGB frames.

---

### A.2.2 Channel split (after registration)

- RGB → R / G / B
- from here: **no cross-channel operations**

Rationale:

> Cross-channel stacking coherently adds color-dependent resampling residuals.

---

## A.3 Hand-off to the common core

From here all rules from methodology v2 apply unchanged, but **per channel**.

Input:

```
R_frames[f][x,y]
G_frames[f][x,y]
B_frames[f][x,y]
```

---

# B. CFA-based path (optional, experimental)

## B.1 Purpose and positioning

The CFA path avoids **any color-dependent interpolation before tile analysis**.

It is methodically ideal, but:

- more complex
- more implementation-heavy
- currently experimental

---

## B.2 Steps B.1–B.2 (CFA)

### B.2.1 Registration on CFA luminance

- CFA luminance derived from real samples (e.g., G-dominant or sum)
- estimate exactly **one** transform per frame
- robust methods (RANSAC / ECC)

Important:

> The transform is color-independent, but interpolation must be **CFA-aware**.

---

### B.2.2 CFA-aware transform

- split CFA mosaic into 4 subplanes (R, G1, G2, B)
- apply the identical transform to each subplane
- **no interpolation across Bayer phases**
- re-interleave to CFA

Result: registered CFA frames without Bayer-phase mixing.

---

### B.2.3 Channel split

- CFA → R / G / B (or G-only)
- from here: identical to path A

---

# 3. Common core from phase 3 (A == B)

## 3.1 Global normalization (mandatory)

### Purpose

Decouple photometric transparency fluctuations from quality metrics.

### Requirements

- global
- linear
- exactly once
- **before any metric computation** (except B_f used for normalization)
- separated per color channel

### Allowed methods

- background‑based scaling (masked, robust)
- fallback: scaling by global median

**Binding order:**

1. compute B_f,c (background level) on **raw** data
2. normalize: I'_f,c = I_f,c / B_f,c
3. compute σ_f,c and E_f,c on **normalized** data

Formally:

```
B_f,c = median(I_f,c)            # BEFORE normalization
I'_f,c = I_f,c / B_f,c           # normalization
σ_f,c = std(I'_f,c)              # AFTER normalization
E_f,c = gradient_energy(I'_f,c)  # AFTER normalization
```

### Forbidden

- histogram stretch
- asinh / log
- local/adaptive normalization before tile analysis

---

## 3.2 Global frame metrics

For each registered, normalized frame *f*, compute:

- **B_f** – global background level (robust, masked)
- **σ_f** – global noise
- **E_f** – gradient energy (large‑scale structure)

### Normalization

All metrics are robustly scaled using **median + MAD**.

Formally (for a metric value `x`):

[
\tilde x = \frac{x - \mathrm{median}(x)}{1.4826 \cdot \mathrm{MAD}(x)}
]

### Global quality score

[
Q_f = \alpha(-\tilde B_f) + \beta(-\tilde\sigma_f) + \gamma\tilde E_f
]

with:

- α + β + γ = 1 (mandatory)
- default: α = 0.4, β = 0.3, γ = 0.3

`Q_f,c` is clamped to **[−3, +3]** before exp(·).

### Global weight

[
G_f,c = \exp(Q_f,c)
]

### Adaptive weights

If the data characteristics deviate strongly from typical conditions, weights can be adapted.

**Algorithm (variance-based):**

```
1. Compute metric variances:
   Var(B), Var(σ), Var(E)

2. Weights from variance:
   α' = Var(B) / (Var(B) + Var(σ) + Var(E))
   β' = Var(σ) / (Var(B) + Var(σ) + Var(E))
   γ' = Var(E) / (Var(B) + Var(σ) + Var(E))

3. Apply constraints:
   α', β', γ' = clip(α', β', γ', 0.1, 0.7)

4. Renormalize:
   α', β', γ' = normalize(α', β', γ') so that Σ = 1
```

**Properties:**

- higher variance → higher weight
- min 0.1, max 0.7 per weight (prevents extremes)
- sum guaranteed = 1.0
- fallback to default weights if Var = 0

**Config:**

```yaml
global_metrics:
  adaptive_weights: true
  weights:
    background: 0.4
    noise: 0.3
    gradient: 0.3
```

**Semantics:** `G_f` encodes only global atmospheric quality.

---

## 3.3 Tile geometry (seeing‑adaptive)

> Tiles are generated AFTER registration and channel split, but BEFORE any channel combination.

### FWHM estimation

- from **registered frames**
- robust sampling across many stars and frames
- outliers (high ellipticity, low SNR) are rejected

### Tile size

Definitions:

- `W`, `H` – image width/height in pixels
- `F` – robust FWHM estimate in pixels (e.g. median across many stars and frames)
- `s = tile.size_factor` – dimensionless scale factor
- `T_min = tile.min_size`
- `D = tile.max_divisor`
- `o = tile.overlap_fraction` with `0 ≤ o ≤ 0.5`

Derivation (compact, normative):

1. A seeing‑limited star has a characteristic spatial scale `F` (FWHM). To measure local seeing/focus and structure robustly, a tile must cover **multiple PSF scales**.
2. Therefore we choose the tile edge length proportional to `F`:

```
T_0 = s · F
```

3. We enforce lower/upper bounds for numerical stability and locality.

Normative tile geometry:

```
T = floor(clip(T_0, T_min, floor(min(W, H) / D)))
O = floor(o · T)
S = T − O
```

where `clip(x,a,b) = min(max(x,a),b)`.

**Boundary checks (binding):**

Before tile computation, enforce:

1. `F > 0`: if FWHM not measurable, use default `F = 3.0`
2. `T_min ≥ 16`: absolute lower bound for tile size
3. `T ≥ T_min`: if `T < T_min` after computation, set `T = T_min`
4. `S > 0`: if `S ≤ 0` (extreme overlap), set `o = 0.25` and recompute
5. `min(W, H) ≥ T`: if image smaller than tile, use `T = min(W, H)` and `O = 0`

---

## 3.3.1 Tile‑based Noise Reduction (optional)

**Purpose:** Before computing local metrics, adaptive noise reduction can be applied at the tile level. This reduces background noise while preserving stars and structures.

**Algorithm: Highpass + Soft‑Threshold**

For each tile *t* in frame *f*:

1. **Background estimation:** Box blur with kernel size *k*
   ```
   B_t = box_blur(T_t, k)
   ```

2. **Residual (highpass):**
   ```
   R_t = T_t − B_t
   ```

3. **Robust noise estimation (MAD):**
   ```
   σ_t = 1.4826 · median(|R_t − median(R_t)|)
   ```

4. **Soft threshold:**
   ```
   τ = α · σ_t
   R'_t = sign(R_t) · max(|R_t| − τ, 0)
   ```

5. **Reconstruction:**
   ```
   T'_t = B_t + R'_t
   ```

**Parameters:**

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `tile_denoising.enabled` | Enable denoising | false | true |
| `tile_denoising.kernel_size` | Box blur kernel size *k* (odd) | 15 | 31 |
| `tile_denoising.alpha` | Threshold multiplier *α* | 2.0 | 1.5 |

**Overlap blending:**

Since tiles overlap, denoised tiles are blended using linear weights:
```
w(x, y) = ramp(x) · ramp(y)
```
where `ramp` linearly increases from 0 (edge) to 1 (center).

**Typical results (empirical):**

| kernel | alpha | Noise reduction | Star preservation |
|--------|-------|-----------------|-------------------|
| 15 | 2.0 | ~75% | ~91% |
| 31 | 1.5 | ~89% | ~93% |
| 31 | 2.0 | ~89% | ~91% |

**Recommendation:** `kernel_size=31, alpha=1.5` provides the best balance between noise reduction and signal preservation.

---

## 3.4 Local tile metrics

For each tile *t* and each frame *f*:

### Case A – stars present in the tile

**Measured quantities:**

- FWHM_f,t
- roundness R_f,t
- local contrast C_f,t

**Quality score (standard):**

[
Q_{star} = 0.6,(−\widetilde{\mathrm{FWHM}}) + 0.2,\tilde R + 0.2,\tilde C
]

### Case B – no stars in the tile

**Measured quantities:**

- gradient energy E_f,t
- local standard deviation σ_f,t
- local background level B_f,t

**Quality score (default):**

[
Q_{struct} = 0.7,\widetilde{(E/\sigma)} − 0.3,\tilde B
]

This metric is largely brightness‑invariant.

### Local weight

All local quality scores are clamped to **[−3, +3]**.

[
L_{f,t} = \exp(Q_{local})
]

---

## 3.5 Effective weight

[
W_{f,t} = G_f \cdot L_{f,t}
]

`G_f` and `L_f,t` represent orthogonal information axes.

---

## 3.6 Tile reconstruction

For each pixel *p* in tile *t*:

[
I_t(p) = \frac{\sum_f W_{f,t} I_f(p)}{\sum_f W_{f,t}}
]

### Stability rules

Define the denominator:

[
D_t = \sum_f W_{f,t}
]

with a small constant `ε > 0`.

- If `D_t ≥ ε`: normal weighted reconstruction.
- If `D_t < ε` (e.g. all weights numerically ~0):
  1. reconstruct the tile using an **unweighted mean over all frames** (no frame selection):

     [
     I_t(p) = \frac{1}{N}\sum_f I_f(p)
     ]

  2. mark the tile as `fallback_used=true` (used by validation/abort decisions).

### Overlap-add with window function (binding)

- window function: **Hanning** (2D, separable)
- definition: `w(x,y) = hann(x) · hann(y)` with `hann(t) = 0.5 · (1 - cos(2πt))`

### Edge handling

- overlap‑add (no hard seams)

### Tile normalization (binding)

Before overlap-add, normalize each tile:

1. subtract background: `T'_t = T_t - median(T_t)`
2. normalize: `T''_t = T'_t / median(|T'_t|)` (if median > ε)

No feedback into quality metrics.

---

## 3.7 State-based clustering

### Principle

A synthetic frame represents a **physically coherent observing state**, not a time interval.

### State vector

For each frame *f*:

[
v_f = (G_f, \langle Q_{tile} \rangle, \mathrm{Var}(Q_{tile}), B_f, \sigma_f)
]

### Clustering

- cluster **frames**, not tiles

**Dynamic cluster count (binding):**

```
K = clip(floor(N / 10), K_min, K_max)
```

where:

- K_min = 5
- K_max = 30
- N = number of frames

Examples:

- N = 50 → K = 5
- N = 200 → K = 20
- N = 500 → K = 30
- N = 800 → K = 30 (capped)

Fallbacks (only under explicitly stable conditions):

- quantiles by G_f
- time buckets

---

## 3.8 Synthetic frames and final stacking

**Synthetic frames (binding):**

For each cluster k:

```
S_k,c = Σ_{f∈Cluster_k} G_f,c · I_f,c / Σ_{f∈Cluster_k} G_f,c
```

where I_f,c are the **original frames** (not reconstructed).

**Optional (tile‑based, for quality propagation):**

If tile‑level quality improvements (e.g. via tile noise filtering in §3.3.1) should be preserved all the way to the final stack, synthetic frame generation can optionally be performed tile‑wise. For each tile *t* inside a cluster, use the effective weights

```
W_f,t,c = G_f,c · L_f,t,c
```

and assemble the synthetic frame using the same overlap‑add reconstruction principle as in §3.6.

Enable via:

`synthetic.weighting: tile_weighted` (default: `global`).

Result: 15–30 synthetic frames per channel (matching cluster count).

**Final stacking (binding, Python-only):**

Synthetic frames are stacked purely linearly in the backend. Optionally, a
pixel-wise **sigma-clipping** step may be applied before computing the mean
in order to suppress extreme outliers (e.g. cosmic rays). The normative
result is:

```
R_c = mean(S_c) = (1/K) · Σ_k S_k,c
```

with:

- linear stacking (unweighted in the state space – all weights are already
  encoded in S_k,c)
- **no drizzle**
- **no additional global weighting** in the final stacking step

Whenever sigma-clipping is enabled, it must fall back to the **plain mean**
where too few samples remain, thus preserving linearity and the “no frame
selection” invariant.

---

## 3.9 Combination (explicitly outside the methodology)

RGB / LRGB combination is:

- **not part** of tile-based quality reconstruction
- a separate post-processing step
- fully interchangeable

---

## 4. Validation and abort

### Success criteria

- median FWHM ↓ ≥ 5–10%
- field homogeneity ↑
- background RMS ≤ classical stacking
- no systematic tile artifacts

### Abort criteria

- < 30% of **signal‑carrying tiles** usable
- very low spread of tile weights
- visible tiling / seam artifacts
- violation of normalization rules

---

## 5. Key statement

The method replaces the search for “best frames” with a **spatio‑temporal quality map**, using each piece of information exactly where it is physically valid.

This specification is **normative**. Deviations require explicit versioning.

---

## 6. Test cases (normative)

The following test cases are mandatory. A run is methodology‑conform only if these tests (automated or reproducible manual) are satisfied.

1. **Global weight normalization**
   - **Given**: α, β, γ from configuration
   - **Then**: α + β + γ = 1 (hard error otherwise)

2. **Clamping before exponential**
   - **Given**: metric values including outliers
   - **Then**: `Q_f` and `Q_local` are clamped to [−3, +3] before `exp(·)`

3. **Tile size monotonicity**
   - **Given**: two seeing estimates `F1 < F2`
   - **Then**: `T(F1) ≤ T(F2)` (subject to clamping)

4. **Overlap determinism**
   - **Then**: `0 ≤ overlap_fraction ≤ 0.5` and `O = floor(o·T)`, `S = T−O` are integer and deterministic

5. **Low‑weight tile fallback**
   - **Given**: a tile with `D_t < ε`
   - **Then**: reconstruction uses the unweighted mean over all frames; output contains no NaNs/Infs

6. **Channel separation / no channel coupling**
   - **Then**: no metric, weight, or reconstruction step mixes information between R/G/B

7. **No frame selection (invariant)**
   - **Then**: every reconstruction uses all frames; violations abort the run

8. **Determinism**
   - **Given**: identical inputs (frames + config)
   - **Then**: stable outputs within a defined numeric tolerance and identical tile geometry

---

## 7. Change history

| Date | Version | Changes |
|-------|---------|---------|
| 2026-01-09 | v3.1 | Boundary checks for tile geometry (§3.3) |
| 2026-01-09 | v3.1 | Dynamic cluster count K = clip(N/10, 5, 30) (§3.7) |
| 2026-01-09 | v3.1 | Adaptive global weights as optional extension (§3.2) |
| 2026-01-09 | v3.1 | Gradual degradation instead of hard abort (§2.4 / §4) |
| 2026-01-09 | v3.1 | Explicit Q_local formula with MAD normalization (§3.4) |
| 2026-01-09 | v3.1 | Hanning window function and ε=1e-6 specified (§3.6) |
| 2026-01-09 | v3.1 | Synthetic frame formula explicitly documented (§3.8) |
| 2026-01-09 | v3.0 | Initial v3 specification with path A/B |

---

## Appendix A – Implementation notes (non‑normative, but strongly recommended)

This appendix refines computational and algorithmic details to ensure **reproducible, robust implementations**. It extends the methodology without changing its semantics.

### A.1 Background estimation (global and local)

**Goal:** robust separation of signal and atmospheric haze.

Recommended procedure:

- coarse object mask (e.g., sigma‑clip + dilation)
- compute background from remaining pixels
- robust statistic (median or biweight location)

Note:

> The background must **not** contain structural gradients that later leak into E or E/σ.

---

### A.2 Noise estimation σ

**Global:**

- robust standard deviation from background‑masked pixels
- no smoothing before estimation

**Local (tile):**

- same method but restricted to tile
- σ is explicitly used as **normalization** for structure metrics

---

### A.3 Gradient energy E

**Recommended definition:**

E = mean(|∇I|²)

More robust alternative:

E = median(|∇I|²)

Implementation notes:

- Sobel or Scharr operator
- optional light pre‑smoothing (σ ≤ 1 px), but consistent globally & locally
- discard border pixels

Important:

> Different gradient definitions change the scale, **not** the methodology – scaling is absorbed by MAD normalization.

---

### A.4 Star selection for FWHM

Recommended criteria:

- SNR > threshold
- ellipticity < 0.4
- no saturation

FWHM:

- measured via PSF fit or radial profile
- do not apply a log transform; use **MAD-normalized FWHM** directly as \widetilde{\mathrm{FWHM}}

---

### A.5 Normalization (median + MAD)

For each metric x:

x̃ = (x − median(x)) / (1.4826 · MAD(x))

Notes:

- separate per metric
- separate for global vs local
- do not mix scales

---

### A.6 Tile normalization before overlap‑add

Procedure:

1. estimate and subtract local background
2. scale tile to common median
3. apply window function
4. overlap‑add

Guard:

- if |median(tile_bgfree)| < ε_median, do **not** scale (scale = 1.0)

Goal:

- avoid patchwork brightness
- do not influence quality metrics

---

### A.7 Clustering

Recommendations:

- standard: k‑means or GMM
- standardize feature vector first
- multiple initializations; choose best inertia/LLH

Warning:

> Time‑based clustering is **not** a substitute for state clustering.

---

### A.8 Numerical stability

- explicitly set ε in the tile reconstruction denominator
- clamp exp(Q) (e.g., Q ∈ [−3, 3])
- prefer double precision

Recommended defaults:

- ε = 1e−6
- ε_median = 1e−6

---

### A.9 Debug and diagnostic artifacts (recommended)

During development store:

- histograms of Q_f and Q_local
- 2D maps of tile weights
- difference image reconstructed − classical

These artifacts are not part of production but are essential for verification.

---

## Appendix B – Validation plots (formally specified)

This appendix defines **mandatory validation artifacts** used to decide whether a run is **successful**, **borderline**, or **failed**. All plots must be generated from **production‑relevant data**.

### B.1 FWHM distribution (before / after)

**Type:** histogram + boxplot

**Inputs:**

- classical reference (stack or single frames)
- synthetic quality frames

**Metrics:**

- median FWHM
- interquartile range

**Acceptance:**

- median FWHM reduction ≥ `validation.min_fwhm_improvement_percent`

---

### B.2 FWHM field map (2D)

**Type:** heatmap across image coordinates

**Inputs:**

- local FWHM measurements from star tiles

**Goal:**

- field homogenization
- reduction of edge seeing/rotation artifacts

**Warning signal:**

- hard transitions along tile borders

---

### B.3 Global background vs time

**Type:** line plot

**Inputs:**

- B_f (raw) = before global normalization (registered but not yet scaled frames)
- effective contribution after weighting

**Goal:**

- correct down‑weighting of cloudy phases

---

### B.4 Global and local weights over time

**Type:** scatter/line

**Inputs:**

- G_f
- ⟨L_f,t⟩ per frame

**Goal:**

- clear separation of observing states

---

### B.5 Tile weight distribution

**Type:** histogram

**Inputs:**

- W_f,t for all tiles

**Acceptance:**

- variance ≥ `validation.min_tile_weight_variance`

---

### B.6 Difference image

**Type:** image + histogram

**Definition:**

difference = reconstruction − classical stacking

**Goal:**

- visible detail gain
- no large‑scale systematic patterns

**Abort:**

- periodic tile patterns

---

### B.7 SNR vs resolution

**Type:** scatter

**Inputs:**

- local SNR
- local FWHM

**Goal:**

- physically plausible trade‑off
- no artificial over‑sharpening

---

## Appendix C – Complexity and performance budget

This appendix supports planning and scaling of production runs.

### C.1 Computational complexity (rough order)

Let:

- F = number of frames
- T = number of tiles
- P = pixels per tile

**Global metrics:** O(F · N_pixels)

**Tile analysis:** O(F · T · P)

**Reconstruction:** O(T · F · P)

Tile analysis dominates runtime.

---

### C.2 Memory requirements

- one frame in RAM (float32): ~4 · W · H bytes
- tile buffers: ~T · P · sizeof(float)

**Recommendation:**

- stream per tile
- do not keep a full frame matrix in RAM

---

### C.3 I/O strategy

- registration: single read/write pass
- tile analysis: prefer sequential access
- synthetic frames: explicitly persist

Avoid:

- random tile access on rotating disks

---

### C.4 Parallelization

Suitable levels:

- tiles (embarrassingly parallel)
- frames within a tile (optional)

Notes:

- global normalization is independent per frame and can be parallelized (I/O may limit)
- state clustering is typically not the bottleneck; parallelization is optional

Option: RabbitMQ‑based parallelization

This option is intended for later implementation and enables horizontal scaling across multiple workers.

- task queue: RabbitMQ
- granularity:
  - preferred: **tile tasks** (one task = tile t across all frames f)
  - optional: frame tasks within a tile (only if local I/O is fast)
- results:
  - reconstructed tile block + summary statistics (e.g., ΣW, tile median after bg subtraction)
  - separate channel/queue for diagnostic artifacts (histograms, QA maps)
- aggregation:
  - master collects tile results and performs deterministic overlap‑add
  - deterministic seeds/sorting for reproducibility
- fault tolerance:
  - idempotent tasks (tile can be recomputed)
  - dead‑letter queue for failed tiles

---

### C.5 Runtime estimate

For typical values:

- F ≈ 1000
- T ≈ 200–400
- P ≈ (64–256)²

Expectation:

- CPU (8–16 cores): hours
- GPU acceleration: optional

---

### C.6 Abort on runtime limit

The following limits are binding:

- `runtime_limits.tile_analysis_max_factor_vs_stack`
- `runtime_limits.hard_abort_hours`

If exceeded, perform a controlled abort.
