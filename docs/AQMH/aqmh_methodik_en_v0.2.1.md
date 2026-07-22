# Adaptive Quality Map Hyperstacking (AQMH) Methodology v0.2.1

**Document status:** Normative for the `v0.2.1` configuration family.  
**Supersedes:** `aqmh_methodik_en_v0.2.0.md`.  
**Language:** English. For the German version see `aqmh_methodik_de_v0.2.1.md`.

---

## 0. Document History and Scope

### 0.1 Changes from v0.2.0

Version `v0.2.1` is a strict superset of `v0.2.0`. All mathematical definitions,
invariants, normative defaults and diagnostic requirements of `v0.2.0` remain
binding. `v0.2.1` formalises three main additions that were introduced in the
reference implementation after the `v0.2.0` baseline:

1. **Background-gradient penalty in global frame quality (§1.5, §4.2).**  
   A third global-quality signal penalises frames with strong large-scale
   background gradients (e.g. light pollution, moon glow). The signal is
   derived from a quadrant-based sky-gradient estimator, not from the AQMH
   quality map itself. It is therefore documented as a permitted
   **cross-infrastructure extension** that remains compatible with the AQMH
   independence principle (§0.4).

2. **Registration-weight guard (§4.3).**  
   Before the per-pixel weighted reconstruction, each frame's global AQMH
   weight can be damped by a registration-confidence factor derived from the
   shared global-registration artifact. This guard is purely a robustness
   measure; it does not change the reconstruction formula and it remains a
   frame-level multiplier.

3. **Adaptive low-frequency neutralisation and structure-masked detail
   blending (§6.3, §6.4).**  
   After the ordinary AQMH reconstruction, an optional post-processing stage
   removes low-frequency "veil" residuals by subtracting a heavily blurred
   difference between the AQMH output and a uniform-control (unweighted) mean.
   Detail is then selectively reintroduced in high-structure regions. A
   validation gate decides whether the neutralised/blended result is kept; if
   not, the raw AQMH output is preserved. Uniform control must never silently
   replace or attenuate raw AQMH.

The goal of these additions is to preserve the v0.2.0 pixel-quality model
while reducing three practical failure modes:

- Large-scale background gradients that would otherwise dominate the final
  stack because every frame shares a similar additive gradient.
- Residual registration weakness (low correlation or deep chain-predicted frames)
  being amplified by high AQMH quality scores.
- Low-frequency "veil" or background non-linearity introduced by the
  weighted reconstruction itself.

### 0.2 Gate Applicability Clarification (2026-07-18)

The validation gates introduced by `v0.2.1` are only meaningful for metrics
that are finite and comparable in both the AQMH candidate and the uniform
control. A metric whose control value is non-finite, missing, or numerically
degenerate must be reported as `not_applicable`; it must not be converted into
an artificial hard failure.

This clarification is object-agnostic. Implementations must not special-case
individual targets or object names. The same applicability rules must work for
bright emission nebulae, faint galaxies, large galaxy arms, star clusters,
high-latitude dust/IFN fields and sparse star fields. Scene content may affect
whether a metric is applicable; it must not change the mathematical meaning of
the metric.

In particular:

- Relative regression must not be computed from a zero or near-zero control
  denominator without an explicit absolute fallback threshold.
- `null`/`NaN` seam scores are diagnostic values, not valid veto values.
- Star-tail and elongation gates require enough comparable star samples in both
  candidates.
- Final selection may use asymmetric tradeoffs: a clear primary improvement
  (e.g. FWHM or elongation) may tolerate bounded secondary regressions
  (background, seam, tail) when those secondary metrics are applicable and
  reported.
- Structure preservation must be evaluated over more than the brightest
  high-gradient pixels. Faint galaxy arms, dust lanes and diffuse nebula
  structures are valid signal and must not be treated as background solely
  because their gradients fall below a high global quantile.

---

## 0.3 Independence and Shared Infrastructure

AQMH remains an **independent reconstruction method**. It may reuse shared
pipeline infrastructure, but its quality model and reconstruction weights are
not derived from Classic Tile Compile local/tile metrics.

Shared infrastructure may include:

- input scan and non-quality eligibility filtering (readability, calibration
  availability, explicit user exclusions);
- calibration and registration/prewarping;
- global photometric normalization;
- global output-canvas mask `C` and frame-specific registered-valid masks `M_f`;
- run management, logging, artifacts, reports, UI plumbing.

The AQMH algorithm itself consists of:

- AQMH dense quality-map computation;
- AQMH pixel-wise weighted reconstruction;
- AQMH diagnostics and optional region extraction.

Classic Tile Compile and AQMH must remain runnable independently. Enabling or
disabling one method must not change the mathematical definition of the other.

### 0.4 Cross-Infrastructure Extensions

`v0.2.1` explicitly allows the following carefully scoped inputs from shared
infrastructure, provided they are documented and do not replace AQMH-native
signals:

- **Sky-gradient summary** for the background penalty (§1.5).  
  This is a scalar per frame, derived from the background-masked input image
  after registration. It is used only inside the global-quality sigmoid and
  remains orthogonal to the within-frame `Q_map` computation.

- **Registration confidence metadata** for the weight guard (§4.3).  
  This is a scalar multiplier per frame, derived from the existing global
  registration artifact. It adjusts the global factor `G_f` but does not
  influence `Q_map`.

- **Uniform-control mean** for neutralisation validation (§6.3).  
  The unweighted mean of the registered frames is computed as a side product
  of AQMH reconstruction (`compute_uniform_control = true`). It is used only
  for post-reconstruction validation, not as a primary quality input or an
  output candidate.

- **PSF-FWHM sharpness proxy** for the global sharpness summary `g_sharp`
  (§1.1, §4.2).  
  When per-frame star metrics are available from the shared metrics phase,
  the runner may substitute `1 / wfwhm` (weighted FWHM, inverted so that
  smaller FWHM yields a higher sharpness value) for the AQMH-map
  `g_sharp` summary.  This is a scalar per frame, derived from PSF star
  detection on the registered frame.  It is used only inside the
  global-quality sigmoid and remains orthogonal to the within-frame `Q_map`
  computation.  The AQMH-map `g_sharp` serves as fallback when star metrics
  are unavailable or invalid.  The selected source is recorded per frame as
  `global_sharpness_source` (`"laplacian_variance"` or `"psf_wfwhm_inverted"`)
  in `aqmh_metrics.json`.

  **Rationale.** The AQMH-map sharpness signal `Phi_sharp_0` is based on
  `local_variance(laplacian)`, which measures local high-frequency variance.
  On stellar-dominated fields this is an excellent sharpness proxy.  On
  extended-emission targets (galaxies, nebulae) the same signal is dominated
  by nebular texture rather than stellar PSF width, and can correlate
  *positively* with seeing FWHM — causing an inversion where worse frames
  receive higher `g_sharp` and thus higher weights.  The PSF-FWHM proxy is
  immune to this inversion because it measures stellar image width directly,
  independent of extended-scene content.

---

## 1. Principles and Definitions

### 1.1 Physical Objective

The method models per-pixel observational quality as the product of two
separable components:

- **Frame-level quality:** the global atmospheric/technical state of frame `f`,
  captured by the strictly positive AQMH factor `G_{f,c}`.
- **Spatial quality field:** the continuous quality distribution within the
  frame, captured by `Q_map_{f,c}(x,y)`.

For every frame/channel, derive the global pre-z-score summaries from its
AQMH maps:

```
g_sharp_{f,c}  = median_{p source-valid}(Phi_sharp_0(p))
g_snr_{f,c}    = median_{p source-valid}(Phi_snr_1(p))
```

using the finest available scale if scale 1 is omitted.

**Post-v0.2.1 extension — PSF-FWHM sharpness proxy.** When per-frame star
metrics are available, the runner may replace `g_sharp` with `1 / wfwhm`
(inverted weighted FWHM) from the shared PSF star-detection phase.  This
avoids a known inversion of `local_variance(laplacian)` on
extended-emission targets (§0.4).  The AQMH-map `g_sharp` is used as
fallback when star metrics are unavailable.  The selected source is recorded
per frame as `global_sharpness_source` (`"laplacian_variance"` or
`"psf_wfwhm_inverted"`) in `aqmh_metrics.json`.

### 1.2 Effective Pixel Weight

The effective per-pixel reconstruction weight is:

```
W_{f,c}^{aqmh}(x,y) = G_{f,c} * Q_map_{f,c}(x,y)
```

`v0.2.1` additionally allows a per-frame registration-confidence multiplier
`R_f ∈ [r_floor, 1]` (§4.3). When enabled, the weight becomes:

```
W_{f,c}^{aqmh}(x,y) = G_{f,c} * R_f * Q_map_{f,c}(x,y)
```

`R_f` is a shared-infrastructure guard, not part of the AQMH quality model. It
must be reported in `aqmh_reconstruction.json`.

### 1.3 Invariants (Binding)

The following invariants remain binding from v0.2.0:

1. **No frame selection:** Entire frames must not be removed based on quality.
2. **Conditional photometric linearity:** Once deterministic weights are
   computed, the final reconstruction remains `R(p) = sum_f w_f(p) * I_f(p) /
   sum_f w_f(p)` with `w_f(p) >= 0`. AQMH must not apply non-linear intensity
   transforms to the samples entering the accumulator.
3. **Determinism:** All quality-map computations must be deterministic and
   reproducible.
4. **Canvas exclusion:** Canvas-invalid pixels are excluded from all AQMH
   accumulators and statistics.
5. **No hallucination:** AQMH outputs are weights and masks only. It does not
   generate or predict pixel intensities.
6. **Sample-count sufficiency for optional selection modes:** Cherry-pick and
   any other per-pixel selection mode must enforce a documented minimum
   retained-sample count, below which the mode is disabled automatically.

`v0.2.1` adds the following derived invariants for the new extensions:

7. **Cross-infrastructure extensions must be monotonic frame-level guards.**  
   The registration-weight guard and the background penalty may only multiply
   the global frame weight by a non-negative scalar. They must not alter
   `Q_map`, must not introduce pixel-dependent weights, and must be
   independently disableable.

8. **Neutralisation must be validation-gated.**  
   Any post-reconstruction neutralisation or detail blending must be compared
   against the ordinary AQMH output and/or the uniform-control mean. The run
   must record which candidate was selected and why. The final output must not
   be worse, by the documented regression metrics, than the ordinary AQMH
   output.

### 1.4 Deterministic Statistics Convention

(Identical to v0.2.0 §1.4.)

Robust z-score is used throughout: `z(x) = (x - med) / (1.4826 * MAD)` over
finite, source-valid samples at the same aggregation level. Degenerate sets
(MAD = 0 or fewer than three finite samples) map to z-score 0 for valid
inputs and are flagged invalid otherwise.

### 1.5 Global Quality with Background Penalty (v0.2.1 addition)

`v0.2.1` introduces an optional third global-quality signal:

```
g_background_{f,c} = sky_gradient_f
```

where `sky_gradient_f` is a dimensionless relative large-scale gradient:

```
sky_gradient_f = (q_max - q_min) / background_f
```

- `background_f` is the masked median background of frame `f`.
- The image is divided into four quadrants. For each quadrant `q` the masked
  median background `b_q` is computed.
- `q_min = min_q(b_q)`, `q_max = max_q(b_q)`.

If `background_f <= 0` or fewer than four valid quadrant values exist, the
summary is invalid.

The background penalty enters global quality as a **subtractive** term:

```
z_s,f = robust_zscore(g_sharp)
z_n,f = robust_zscore(g_snr)
z_b,f = robust_zscore(g_background)

score_f = w_sharp * z_s,f + w_snr * z_n,f - w_background * z_b,f
G_f     = g_floor + (1 - g_floor) * sigmoid(clamp(g_k_scale * score_f, -8, 8))
```

with `sigmoid(v) = 1 / (1 + exp(-v))`.

**Post-v0.2.1 extension — Sigmoid temperature and score clipping.** The
implementation multiplies `score_f` by a configurable temperature factor
`g_k_scale` (default `1.5`) before applying the sigmoid, and clips the scaled
score to `[-8, 8]` for numerical stability.  The v0.2.1 reference formula uses
an implicit scale of `1.0` and no clipping.  The temperature factor is
necessary because `robust_zscore` (MAD-based) typically produces smaller
absolute values than standard z-scores; without scaling, all `G_f` values
cluster near `g_floor + 0.5 * (1 - g_floor)`, making the weighted stack
nearly unweighted.  The clipping is a safety net that is never active in
practice because individual z-scores are already clamped to `[-5, 5]`, but it
prevents `exp` overflow on extreme inputs.

The weights `w_sharp`, `w_snr`, `w_background` are configured directly as
`g_w_sharp`, `g_w_snr`, `g_w_background_penalty`. At runtime they are converted
to **effective weights** `w_*_eff`:

- If a configured weight is `<= 0`, its effective weight is `0`.
- If fewer than three finite positive summary values are available, the
  effective weight is `0`.
- If the coefficient of variation `CV = MAD / median` of the finite positive
  summary values is below `0.01`, the effective weight is `0`. This prevents
  near-constant signals from amplifying noise and inverting the quality
  ranking.

The remaining effective weights are renormalised so that:

```
w_sharp_norm + w_snr_norm + w_background_norm = 1
```

if at least one effective weight is positive. If all effective weights are
zero, every frame receives `G_f = g_floor`.

**Normative defaults (v0.2.1):**

The v0.2.1 reference defaults are:

- `g_floor = 0.05`
- `g_w_sharp = 0.6`
- `g_w_snr = 0.4`
- `g_w_background_penalty = 0.3`

The current operating defaults (post-v0.2.1 revision) are:

- `g_floor = 0.03`
- `g_w_sharp = 0.55`
- `g_w_snr = 0.30`
- `g_w_background_penalty = 0.25`

These were tuned empirically: the lower floor and more balanced weights
prevent a small number of high-quality frames from dominating the stack on
typical deep-sky datasets where frame-to-frame quality variation is moderate.

Post-v0.2.1 extension default:

- `g_k_scale = 1.5` (sigmoid temperature; `1.0` restores the v0.2.1 reference
  formula)

Setting `g_w_background_penalty = 0.0` restores the exact v0.2.0 global
quality definition.

**Rationale.** A strong large-scale gradient means that the background model
used by the SNR signal (`b_s` in §2.3.2(b)) is a biased local estimate. Frames
with very flat backgrounds tend to be cleaner and more linear; frames with a
strong moon-glow or light-pollution gradient receive a lower global weight,
reducing their influence on the final stack without removing them entirely.

**Compatibility note.** The background penalty is not derived from `Q_map`. It
is therefore an **extension** of v0.2.0, not a contradiction. It must be
clearly reported in `aqmh_metrics.json` as the per-frame field
`global_background_penalty_input`, with source
`global_background_penalty_source: "sky_gradient"`. The configured penalty
weight is part of the AQMH global-quality configuration, not the
`registration_weight_guard` artifact.

---

## 2. Quality Map Computation

### 2.1 Overview

Identical to v0.2.0. AQMH computes a dense quality map for each frame/channel
pair by fusing multi-scale local sharpness, SNR and artifact-anomaly signals.

### 2.2 Shared Preprocessing

Identical to v0.2.0. All frames must be calibrated, registered and prewarped
to a common output canvas before AQMH map computation.

### 2.3 Multi-Scale Pyramid

Identical to v0.2.0 §2.3. The four scales, omission rule, sharpness, SNR and
artifact computations, and the geometric-mean fusion are unchanged.

### 2.4 Multi-Scale Fusion

Identical to v0.2.0 §2.4.

```
Q_map_{f,c}(x,y) = ( prod_{s in S_actual} Psi_s^{up}(x,y) )^{1/P_actual}
```

implemented as `exp(mean_s(log(Psi_s^{up})))` with the exact-zero veto and
output guard.

### 2.5 Block-Level Diagnostic Summaries

Identical to v0.2.0 §2.5.

---

## 3. Quality Map Storage and Memory Model

The default storage mode is full-resolution `float32` with
`resolution_divisor = 1`. An optional performance mode uses `uint16` with
`resolution_divisor = 2` for reduced memory footprint; this is not the
default and must be explicitly reported. Cherry-pick requires the reference
mode because a reduced or quantised map can change the per-pixel ranking.
Each run must report resolution and data type.

The background-penalty signal does not affect quality-map storage; it is a
per-frame scalar computed during or after map computation and stored in
`aqmh_metrics.json`.

---

## 4. Pipeline Integration

### 4.1 AQMH Processing Stages

Identical to v0.2.0:

```
AQMH_MAPS
AQMH_GLOBAL_QUALITY
AQMH_RECONSTRUCTION
AQMH_DIAGNOSTICS
AQMH_NATIVE_BGE_INPUTS
```

The output of `AQMH_RECONSTRUCTION` is additionally persisted unchanged as
`outputs/aqmh_reconstructed_raw.fit`. For OSC data this artifact still contains
the CFA pattern and is the only valid input for a resume from `STACKING`.
`reconstructed_L.fit` is not a resume artifact: `STACKING` replaces it with the
scaled luminance output. If an older run lacks the raw artifact, the runner must
rerun `AQMH_RECONSTRUCTION` from the map and prewarp caches. Already stacked
luminance must never be debayered again as CFA data.
Set `aqmh.reconstruction.delete_prewarped_cache_after_run: false` to retain the
`cache/prewarped_frames` directory for later resumes; the default `true` continues to reclaim the
disk space after the run.

### 4.2 Global Quality Stage (v0.2.1 update)

The stage now computes, for each frame, the three summary vectors
`g_sharp`, `g_snr`, `g_background` and passes them to
`compute_aqmh_global_quality`. The result is a vector `G_f` per frame.

All three inputs are recorded per frame in `aqmh_metrics.json`:

- `global_sharpness_input`
- `global_sharpness_source: "laplacian_variance"` or `"psf_wfwhm_inverted"` (post-v0.2.1 extension)
- `global_snr_input`
- `global_background_penalty_input`
- `global_background_penalty_source: "sky_gradient"`
- `global_quality_input_invalid`
- `global_quality` (the final `G_f`)

When the PSF-FWHM sharpness proxy is used, `global_sharpness_input` already
contains the `1 / wfwhm` value.

### 4.3 Registration-Weight Guard (v0.2.1 addition)

Before reconstruction, the runner may optionally apply a
**registration-weight guard** to `G_f`. It reads the shared
`global_registration.json` artifact, which must contain per-frame fields
`cc` (cross-correlation confidence), `source` (registration source type) and
optionally `chain_depth`.

For each frame `f`:

1. **Cross-correlation mapping.** Direct and reference solutions are not
   continuously damped above the floor:

   ```
   r_cc = cc_f >= cc_floor ? 1.0 : r_floor
   ```

   Sequential, predicted, interpolated, and unknown solutions use:

   ```
   t = clamp( (cc_f - cc_floor) / (cc_full - cc_floor), 0, 1 )
   r_cc = r_floor + (1 - r_floor) * t
   ```

   with defaults `r_floor = 0.30`, `cc_floor = 0.35`, `cc_full = 0.80`.

2. **Registration-source damping.**

   - If `source == "sequential_refined"`: `r_f *= sequential_factor`
     (default `0.92`).
   - If `source` contains `"predicted"`, `"interpolated"`, or is `"unknown"`:
     `r_f *= predicted_factor` (default `0.50`).

3. **Chain-depth damping for non-direct solutions.**

   ```
   depth_penalty = min( depth_max, max(0, depth_f - 1) * depth_penalty_per_step )
   r_f *= (1 - depth_penalty)
   ```

   with defaults `depth_penalty_per_step = 0.03`, `depth_max = 0.15`.

4. **Clamp.**

   ```
   R_f = clamp(r_f, r_floor, 1.0)
   ```

The effective global weight used in reconstruction is:

```
G_f^{eff} = G_f * R_f
```

The guard is enabled by default (`registration_weight_guard: true`) and
reported in `aqmh_reconstruction.json` under `registration_weight_guard`.

**Rationale.** Deep chain-predicted or weakly-correlated frames can survive
eligibility filtering but still carry lower registration accuracy. The guard
prevents high AQMH quality scores from amplifying registration errors. It is
a shared-infrastructure multiplier and is therefore documented as a
**robustness extension**, not as part of the core AQMH quality model.

### 4.4 Registration NCC preprocessing (v0.2.1 addition)

Before registration NCC comparisons, the normalized proxy images are prepared
for robust correlation as follows:

1. Clamp the proxy image to nonnegative values to remove negative values caused
   by background subtraction.
2. Apply a Gaussian blur with `sigma = 1.5 px` to reduce the influence of hot
   pixels and isolated defects.
3. Compute both identity-overlap NCC and warped-overlap NCC from these prepared
   images.

A near-identity result is accepted only when all of the following hold:

- total shift is smaller than `star_inlier_tol_px`;
- absolute rotation is below `0.1°`;
- warped NCC is no more than `0.02` below identity-overlap NCC; and
- identity-overlap NCC is greater than `0.7`.

The identity-NCC condition is essential: a near-zero warp with low correlation
must not be accepted merely because the optimizer found no meaningful shift.
This registration fix is upstream of AQMH and does not alter `Q_map`; it
improves the reliability of the `cc`, `source`, and `chain_depth` metadata used
by the registration-weight guard.

---

## 5. Pixel-Wise Weighted Reconstruction

### 5.1 Reconstruction Formula

Identical to v0.2.0. For each output channel:

```
R(p) = sum_{f in V_c^I(p)} w_f(p) * I_f(p) / sum_{f in V_c^I(p)} w_f(p)
```

where `w_f(p) = G_f^{eff} * Q_map_{f,c}(p)` and `V_c^I(p)` is the set of frames
with finite intensity and finite quality-map value at pixel `p`.

The geometric support of each registered frame is part of `V_c^I(p)`.
Prewarp pixels outside that support must remain invalid (`NaN` plus the
per-frame support mask); zero-filled warp borders must never enter AQMH as real
intensity samples. Three mask roles remain separate:

- The frame-support mask states which pixels are supplied by one frame.
- The common-overlap mask is used only for analysis, validation, and
  calibration at the configured minimum coverage.
- The output mask is the union of all frame-support masks and preserves every
  genuinely reconstructable edge structure.

The common-overlap mask must not crop AQMH reconstruction or BGE, PCC, and HMS
outputs to the common core. Pixels that do not satisfy `min_n_eff` remain
diagnosable as insufficiently supported, but they must not cause valid nearby
pixels from a few frames to be diluted by nonexistent zero samples.

### 5.2 Sigma Clipping and Effective-Sample Sufficiency

Identical to v0.2.0. Iterative sigma clipping with `clip_sigma_low` and
`clip_sigma_high`, and a minimum effective sample count `min_n_eff` and
minimum fraction `min_fraction`.

### 5.3 Cherry-Pick Stacking Mode

Identical to v0.2.0 §5.3. The run-level `K_nominal_median` floor,
per-pixel `K(p) = max(k_min_required, K_nominal(p))` rule, tiering and
rank-separation diagnostics are unchanged.

---

## 6. Post-Reconstruction Robustness Extensions (v0.2.1)

### 6.1 Uniform Control Mean

When `compute_uniform_control: true`, the reconstruction routine also computes
the unweighted mean of the registered frames over the same valid-pixel set. This
mean is called the **uniform-control** output `U(p)`:

```
U(p) = sum_{f in V_c^I(p)} I_f(p) / |V_c^I(p)|
```

`U(p)` is not a quality-weighted result; it is a diagnostic reference.

### 6.2 Validation Metrics

Both the raw AQMH output `A(p)` and any post-processed candidate are compared
to `U(p)` using the same metrics as v0.2.0 validation:

- `seam_score_regression`
- `fwhm_regression`
- `background_rms_regression`
- `tail11_abs_regression`
- `elongation_regression`

Here, `background_rms` is not the global intensity spread of the image. It is
estimated robustly from horizontal and vertical pixel differences
(`1.4826 * MAD(diff / sqrt(2))`). Slowly varying nebula, galaxy, and IFN signal
is therefore not classified as background noise.

These comparisons are recorded in `aqmh_reconstruction.json`. Each metric must
also record an applicability state:

```
status = pass | fail | not_applicable
```

`not_applicable` is required when either side of the comparison is missing,
non-finite, has too few samples, or would require a relative regression against
a zero/near-zero control denominator. A `not_applicable` metric is diagnostic
only and must not by itself trigger fallback to uniform control.

### 6.3 Adaptive Low-Frequency Neutralisation

If the uniform control `U(p)` is available, an optional neutralised candidate
`N(p)` is computed:

```
L(p) = GaussianBlur( A(p) - U(p), sigma = 96 px )
N(p) = A(p) - L(p)
```

The blur uses a reflect-101 border mode. The sigma of 96 px was chosen
empirically to target large-scale "veil" residuals without affecting
structure on smaller scales.

A validation comparison `compare(N, U)` is computed. `N(p)` may be selected as
the base over `A(p)` only when the background RMS metric is applicable and its
regression is strictly smaller:

```
background_rms_regression(N, U) < background_rms_regression(A, U)
```

If the raw AQMH candidate already passes applicable background gates, the
neutralised candidate must not replace it unless it also preserves the primary
image-quality metrics used by the final selector. The configured background
threshold is recorded as a diagnostic gate at this stage; the final structure
candidate and final output are subject to the applicable validation gates in
§6.4 and §6.5. The selected base image is called `B(p)` (either `A(p)` or
`N(p)`).

**Rationale.** The per-pixel weighted mean can, in rare cases, create a
low-frequency residual that is smoother than the true background. Subtracting
the blurred difference to the unweighted mean removes this additive
low-frequency component while preserving higher-frequency detail.

### 6.4 Structure-Masked Detail Blending

After selecting the base `B(p)`, a second optional candidate tries to preserve
AQMH detail in high-structure regions while keeping the smoother background
from `B(p)`:

1. Compute the gradient magnitude `grad(U)` of the uniform-control mean.
2. Build a soft structure mask `M_s(p)` by mapping the gradient magnitude from
   the `low_q = 0.40` quantile to the `high_q = 0.90` quantile to `[0, 1]` and
   then Gaussian-blurring with `sigma = 4 px`.
   The v0.2.1 reference values were `low_q = 0.70`, `high_q = 0.97`,
   `sigma = 2 px`; the current operating defaults were widened to preserve
   mid-gradient structure (spiral arms, dust lanes) that the original narrow
   range suppressed.
3. Compute the detail candidate:

   ```
   D(p) = U(p) + M_s(p) * (B(p) - U(p))
   ```

   In high-structure regions (`M_s ≈ 1`) `D(p)` follows the AQMH base; in
   smooth regions (`M_s ≈ 0`) it follows the uniform control.

The structure mask is a scene-independent signal-confidence estimate, not a
nebula-specific heuristic. Implementations may extend it with additional
deterministic multi-scale structure measures (for example DoG/Laplacian or
background-mask-aware faint-structure confidence), provided that all components
are reported and no object-specific rules are used. This is important for
galaxies, where scientifically relevant spiral arms and dust lanes often occupy
mid-gradient percentile ranges rather than only the top gradient quantiles.

`D(p)` is accepted if it passes all applicable validation thresholds:

```
background_rms_regression(D, U) <= max_background_rms_regression
fwhm_regression(D, U)            <= max_fwhm_regression
seam_score_regression(D, U)      <= max_seam_score_regression
tail11_abs_regression(D, U)      <= max_tail11_abs_regression
elongation_regression(D, U)      <= max_elongation_regression
```

and it improves at least one of FWHM or seam score relative to `U`.

If `D(p)` fails, an **alpha-blended attenuation** search is performed over
`alpha ∈ [0, 1]`:

```
D_alpha(p) = U(p) + alpha * M_s(p) * (B(p) - U(p))
```

The largest `alpha` that satisfies the applicable validation thresholds and improves
FWHM or seam score is selected. If no `alpha > 0` passes, the base `B(p)` is
kept.

### 6.5 Final Protection Gate

After all post-processing, the selected output `O(p)` is validated one more
time against `U(p)` and the immutable raw AQMH baseline `A(p)`. If:

```
background_rms_regression(O, U) > max_background_rms_regression
```

or any other applicable configured threshold is exceeded, the runner rejects
the post-processed candidate and emits `A(p)`. `U(p)` remains a diagnostic
reference only. Blending toward `U(p)` is forbidden because it can erase faint
diffuse signal while one-sided noise metrics appear to improve.

### 6.6 Reporting Requirements

`aqmh_reconstruction.json` must contain:

- `low_frequency_neutralization_applied` (`true`/`false`)
- `low_frequency_neutralization` (comparison metrics plus `sigma_px`)
- `structure_masked_detail_applied` (`true`/`false`)
- `structure_masked_detail_alpha` (final blend alpha if attenuation was used)
- `structure_masked_detail_validation`
- `uniform_control_gate_triggered` (`true`/`false`)
- `raw_aqmh_validation`
- `final_vs_raw_aqmh_validation`
- `raw_aqmh_preserved_by_guard` (`true`/`false`)
- `selected_candidate`

---

## 7. Validation

### 7.1 Regression Validation Against Uniform Control

Identical to v0.2.0 §9. The reconstructed output is compared to the
uniform-control mean. Regression thresholds:

Current operating defaults (post-v0.2.1 revision):

- `max_seam_score_regression = 0.05`
- `max_fwhm_regression = 0.02`
- `max_background_rms_regression = 0.05`
- `max_tail11_abs_regression = 0.10`
- `max_elongation_regression = 0.08`

The v0.2.1 reference values were `0.02` for seam/FWHM/background and `0.05`
for tail/elongation. The current defaults are wider for seam, background,
tail, and elongation to reduce false rejections on real-world datasets where
sub-threshold regressions are common and do not indicate a meaningful quality
degradation.

These thresholds apply only to metrics whose applicability state is `pass` or
`fail`. Metrics reported as `not_applicable` are excluded from the hard veto and
must carry a reason string in the diagnostic artifact.

### 7.2 Zero-Veto Test

Identical to v0.2.0. When all weights are zero, the output must be zero, not
an unweighted mean.

### 7.3 No Structural Injection

Identical to v0.2.0. The AQMH output must not introduce structure that exceeds
the uniform-control output by the configured regression thresholds.

---

## 8. Diagnostics

Identical to v0.2.0 §6 and §7, with the addition of the fields listed in
§6.6.

Per-frame diagnostics in `aqmh_metrics.json` include:

- `map_mean`, `map_p10`, `map_p90`
- `artifact_frac`
- `sharpness_p50`, `snr_p50`
- `n_regions`
- `global_quality`
- `global_sharpness_input`, `global_snr_input`
- `global_sharpness_source: "laplacian_variance"` or `"psf_wfwhm_inverted"` (post-v0.2.1 extension)
- `global_background_penalty_input` and
  `global_background_penalty_source: "sky_gradient"` (v0.2.1)
- `global_quality_input_invalid`

---

## 9. Configuration Summary

### 9.1 Data and Pipeline

- `data.color_mode`: `OSC` for one-shot-color data.
- `data.bayer_pattern`: default `auto`. The FITS header values `BAYERPAT` and
  `COLORTYP` take precedence; a config value is used only as a fallback when
  the header has no Bayer metadata. This change was introduced to prevent
  example-config defaults from overriding camera metadata.

### 9.2 AQMH Core Parameters (unchanged from v0.2.0)

- `aqmh.pyramid.scales = 4`
- `aqmh.pyramid.base_window_px = 4`
- `aqmh.pyramid.w_sharp = 0.6`
- `aqmh.pyramid.w_snr = 0.4`
- `aqmh.pyramid.k_artifact = 3.0`
- `aqmh.pyramid.frac_artifact_max = 0.25`
- `aqmh.storage.resolution_divisor = 1`
- `aqmh.storage.dtype = "float32"`

### 9.3 Global Quality (v0.2.1)

- `aqmh.global_quality.g_floor = 0.03`
- `aqmh.global_quality.g_w_sharp = 0.55`
- `aqmh.global_quality.g_w_snr = 0.30`
- `aqmh.global_quality.g_w_background_penalty = 0.25`
- `aqmh.global_quality.g_k_scale = 1.5` (post-v0.2.1 extension; `1.0` restores
  the v0.2.1 reference formula)

The v0.2.1 reference defaults were `g_floor=0.05`, `g_w_sharp=0.6`,
`g_w_snr=0.4`, `g_w_background_penalty=0.3`.

Set `g_w_background_penalty = 0.0` to obtain the exact v0.2.0 behaviour.

### 9.4 Registration-Weight Guard (v0.2.1)

- `aqmh.reconstruction.registration_weight_guard = true`
- `aqmh.reconstruction.registration_weight_floor = 0.30`
- `aqmh.reconstruction.registration_cc_floor = 0.35`
- `aqmh.reconstruction.registration_cc_full = 0.80`
- `aqmh.reconstruction.registration_sequential_factor = 0.92`
- `aqmh.reconstruction.registration_predicted_factor = 0.50`
- `aqmh.reconstruction.registration_chain_depth_penalty = 0.03`
- `aqmh.reconstruction.registration_chain_depth_max_penalty = 0.15`

### 9.5 Neutralisation and Blending (v0.2.1)

- Neutralisation blur sigma: `96 px` (hard-coded).
- Structure-mask quantiles: `low_q = 0.40`, `high_q = 0.90`.
- Structure-mask blur sigma: `4 px`.
- Validation thresholds are taken from `aqmh.validation`.

The v0.2.1 reference values were `low_q=0.70`, `high_q=0.97`, blur sigma
`2 px`; the current operating defaults are wider to preserve mid-gradient
structure.

### 9.6 Cherry-Pick (unchanged)

- `aqmh.cherry_pick.enabled = false` by default.
- `k_frac = 0.30`
- `k_min_required = 20`
- `margin_min = 0.02`

---

## 10. Conformance Statement

A run is **v0.2.1-conformant** if and only if:

1. It follows all binding invariants of v0.2.0 (§1.3).
2. It computes `Q_map` according to §2 and §3 with full-resolution float32
   storage unless an explicitly approximate mode is selected and reported.
3. It computes global quality using the extended formula of §1.5 when
   `g_w_background_penalty > 0`, or using the v0.2.0 formula when it is `0`.
4. It applies the registration-weight guard as described in §4.3 when enabled,
   and reports the result.
5. It performs post-reconstruction validation and optional neutralisation as
   described in §6, and records the selected candidate and any fallback.
6. It produces `aqmh_metrics.json`, `aqmh_reconstruction.json` and all
   required diagnostic fields.

A run is **strict v0.2.0-conformant** if `g_w_background_penalty = 0`,
`registration_weight_guard = false`, and all post-reconstruction extensions
are disabled. Such a run is a valid subset of v0.2.1.

---

## 11. References

- `aqmh_methodik_en_v0.2.0.md` — baseline methodology.
- `tile_compile_cpp/include/tile_compile/config/configuration.hpp` —
  configuration structs.
- `tile_compile_cpp/src/metrics/aqmh_global_quality.cpp` — global quality
  computation.
- `tile_compile_cpp/apps/runner_phase_aqmh_reconstruction.cpp` —
  registration-weight guard, neutralisation and blending.
- `tile_compile_cpp/src/metrics/aqmh_quality_map.cpp` — quality-map
  computation.
- `tile_compile_cpp/src/reconstruction/aqmh_reconstruction.cpp` —
  weighted reconstruction and cherry-pick logic.
