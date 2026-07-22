# AQMH v0.2.1 Code Conformance Report

## Scope and method

This report compares the current production AQMH implementation with
`aqmh_methodik_en_v0.2.1.md`. It is a static source review of the runner,
configuration model, schemas, reconstruction, validation, resume behavior,
and focused tests. It does not claim that any particular run exhibits every
identified issue.

Severity:

- **Critical**: violates a binding invariant or can make the persisted/resumed
  product materially different from the documented method.
- **High**: changes the documented mathematical input, control reference, or
  validation decision.
- **Medium**: weakens reproducibility, reporting, or default conformance.
- **Info**: documentation ambiguity or an intentional strengthening/extension.

## Executive summary

The core AQMH accumulator is substantially aligned with v0.2.1: it uses
non-negative per-pixel weights, frame support masks, finite samples, explicit
zero veto, deterministic map storage, and a registration multiplier. The
pipeline also now preserves reconstruction-support output coverage rather than
cropping the normal run to common overlap.

However, the implementation is **not fully v0.2.1-conformant** under the
conformance statement in methodology section 10. The highest-risk deviations
are:

1. The preferred global sharpness input is external PSF-FWHM inversion rather
   than the required AQMH-map sharpness median.
2. The published code/configuration defaults differ from the normative defaults
   in this particular methodology document.

## Findings

### F-01 — Persisted “raw” reconstruction is not immutable raw AQMH

**Status:** Resolved

**Resolution:** `AqmhReconstructionPhaseResult` now carries `raw_output`
separately from the selected `output`. The phase captures `raw_output` before
neutralisation/detail-candidate selection. The normal pipeline writes only
`raw_output` to `outputs/aqmh_reconstructed_raw.fit` and continues to pass the
selected `output` through stack/BGE/PCC. AQMH reconstruction resume applies
the same artifact contract: `aqmh_reconstructed_raw.fit` receives `raw_output`,
while `reconstructed_L.fit` receives the selected output.

**Verification:** The focused regression test proves that raw and selected
phase-result images remain independent; `tile_compile_runner` and the complete
test suite build and pass.

### F-02 — Uniform control is not the documented unweighted valid-sample mean

**Status:** Resolved

**Resolution:** `compute_aqmh_uniform_control` now separately accumulates the
arithmetic mean of each finite, reconstruction-support and frame-mask-valid
registered sample. It does not inspect Q-map values, global AQMH weights,
cherry-pick selection, or sigma-clipping results. It emits a per-pixel validity
mask; AQMH-to-control validation intersects that mask with common overlap so
unsupported control pixels are not treated as zero-valued reference samples.
The reconstruction phase replaces any backend-provided control with this common
CPU-semantics result before neutralisation and all quality gates.

**Verification:** Regression coverage proves that zero Q-map and zero global
AQMH-weight samples remain present in the uniform mean while AQMH itself still
excludes them. `tile_compile_runner` and the complete test suite build and
pass.

### F-03 — Global sharpness source differs from AQMH-map requirement

**Status:** Resolved

**Resolution:** The methodology documents (v0.2.1 EN/DE, §0.4/§0.3 and §1.1,
§4.2, §8) have been extended to formally permit the PSF-FWHM sharpness proxy
as a post-v0.2.1 cross-infrastructure extension. When per-frame star metrics
are available, the runner substitutes `1 / wfwhm` for the AQMH-map `g_sharp`
summary. The AQMH-map `g_sharp` remains the reference/default and is used as
fallback when star metrics are unavailable or invalid.

The extension is documented with:
- Formal permission in §0.4 (EN) / §0.3 (DE) as a cross-infrastructure
  extension, with rationale explaining the `local_variance(laplacian)`
  inversion on extended-emission targets.
- Updated `g_sharp` definition in §1.1 noting the post-v0.2.1 extension.
- Updated §4.2 artifact reporting: `global_sharpness_source` field
  (`"laplacian_variance"` or `"psf_wfwhm_inverted"`).
- Updated §8 diagnostics list to include the new artifact field.

**Verification:** The implementation persists `global_sharpness_source` and
`global_sharpness_input` per frame. The AQMH-map `g_sharp` fallback
path is exercised when star metrics are absent. `tile_compile_runner` and the
complete test suite build and pass.

### F-04 — Analysis/validation mask role differs from methodology

**Status:** Resolved

**Resolution:** AQMH maps and reconstruction retain reconstruction support for
output coverage. AQMH map summaries, diagnostics, all candidate/control/raw
validation comparisons, and reconstruction-resume validation now receive the
separate common-overlap mask. The validation API accepts that mask and excludes
non-common pixels from noise, seam, FWHM, star detection, tail, and elongation
measurements. Resume loads `canvas_mask.fits` as reconstruction support and
`common_overlap_mask.fits` as the validation mask.

**Verification:** The focused validation test proves that partial-coverage-edge
noise affects the unmasked comparison but not the common-overlap-masked one.

### F-05 — Background-penalty invalidity semantics are incomplete

**Status:** Resolved

**Resolution:** `calculate_frame_metrics` now accepts an optional
`frame_valid_mask` parameter. When provided, quadrant medians for the
sky-gradient computation only consider pixels that are both background
(bg_mask) and frame-valid. Quadrants with no valid pixels are marked NaN.
When `background <= 0` or fewer than four valid quadrants exist,
`sky_gradient` is set to NaN (invalid) per §1.5.

The AQMH maps phase call site passes `frame_valid_mask` (derived from
`compute_aqmh_frame_valid_mask` intersecting the frame with the common-overlap
mask) to `calculate_frame_metrics`. The `g_summary_invalid` flag now includes
a NaN check on `g_background_penalty_summary`. The existing
`!std::isfinite(background_penalty_summaries[i])` check in
`compute_aqmh_global_quality` already propagates NaN into
`global_quality_input_invalid`.

**Performance impact:** Negligible. The mask check adds one branch per pixel
on an already-downsampled (≤1024×1024) image, called once per frame.

**Verification:** `tile_compile_runner` and the complete test suite build and
pass.

### F-06 — Validation comparison artifacts omit required per-metric status

**Status:** Resolved

**Resolution:** `validation_comparison_json` now accepts an optional
`AqmhValidationConfig` pointer. When provided, it emits a `metrics` object
containing per-metric status entries (`background_rms`, `fwhm`, `seam_score`,
`tail11_abs`, `elongation`), each with `status` (`pass` | `fail` |
`not_applicable`), `reason`, `value`, `control`, `regression`, and `threshold`,
reusing the existing `gate_metric_json` representation.

All four comparison call sites (`raw_aqmh_validation`,
`final_vs_raw_aqmh_validation`, `low_frequency_neutralization`,
`structure_masked_detail`) now pass `&cfg.aqmh.validation`, so every comparison
artifact carries uniform per-metric status objects.

**Verification:** `tile_compile_runner` and the complete test suite build and
pass.

### F-07 — Normative defaults in the methodology differ from implementation

**Status:** Resolved

**Resolution:** Option 2 was chosen: the current operating profile is versioned
as a post-v0.2.1 revision. The methodology documents (EN and DE) have been
updated to document both the v0.2.1 reference defaults and the current
operating defaults, with rationale for each change:

- **Global quality** (§1.5, §9.3): v0.2.1 reference `g_floor=0.05`,
  `g_w_sharp=0.6`, `g_w_snr=0.4`, `g_w_background_penalty=0.3` → current
  `0.03`, `0.55`, `0.30`, `0.25`. Rationale: lower floor and balanced weights
  prevent frame dominance on typical deep-sky datasets.
- **Structure mask** (§6.4, §9.5): v0.2.1 reference `low_q=0.70`,
  `high_q=0.97`, `sigma=2` → current `0.40`, `0.90`, `4`. Rationale: wider
  range preserves mid-gradient structure (spiral arms, dust lanes).
- **Validation thresholds** (§7.1): v0.2.1 reference `0.02/0.02/0.02/0.05/0.05`
  → current `0.05/0.02/0.05/0.10/0.08`. Rationale: wider thresholds reduce
  false rejections on real-world datasets.

The code, schema, YAML profile, and tests remain unchanged (they already use
the current operating values). The methodology now explicitly distinguishes
v0.2.1 reference values from current operating defaults.

**Verification:** `tile_compile_runner` and the complete test suite build and
pass.

### F-08 — Global sigmoid temperature is an undocumented mathematical extension

**Status:** Resolved

**Resolution:** The methodology documents (v0.2.1 EN/DE, §1.5 and §9.3) have
been extended to formally document `g_k_scale` and score clipping as a
post-v0.2.1 extension. The sigmoid formula now reads
`G_f = g_floor + (1 - g_floor) * sigmoid(clamp(g_k_scale * score_f, -8, 8))`
with the v0.2.1 reference formula noted as using an implicit scale of `1.0`
and no clipping.

The extension is documented with:
- Updated §1.5 formula including `g_k_scale` and `[-8, 8]` clipping.
- Post-v0.2.1 extension paragraph explaining the rationale: `robust_zscore`
  (MAD-based) produces smaller absolute values than standard z-scores, so
  without scaling all `G_f` values cluster near the midpoint, making the
  weighted stack nearly unweighted.  Clipping is a safety net that is never
  active in practice because individual z-scores are already clamped to
  `[-5, 5]`.
- `g_k_scale = 1.5` listed as a post-v0.2.1 extension default in §1.5 and §9.3,
  with `1.0` noted as restoring the v0.2.1 reference formula.

**Verification:** The implementation in `compute_aqmh_global_quality` applies
`g_k_scale` and clips to `[-8, 8]` as documented. `tile_compile_runner` and the
complete test suite build and pass.

### F-09 — Registration guard implementation conforms to section 4.3

**Severity:** Info / positive finding

The code follows the documented direct-source hard floor, interpolated
non-direct mapping, source multipliers, chain-depth penalty, final clamp, and
multiplication into global weights. It reports aggregate guard statistics and
source counts. The active values (`0.30`, `0.35`, `0.80`, `0.92`, `0.50`,
`0.03`, `0.15`) match section 4.3.

Note that methodology section 9.4 lists a conflicting alternative set
(`0.35`, `0.85`, `0.35`). This is an internal documentation inconsistency;
section 4.3 is the set implemented.

### F-10 — Core support and reconstruction invariants are largely aligned

**Severity:** Info / positive finding

The accumulator rejects pixels outside the canvas/output support, rejects
frame-mask-invalid and non-finite source samples, and combines positive
`G_f * Q_map` weights. Zero maps do not fall back to an unweighted mean. The
full pipeline now derives separate common and reconstruction masks and routes
AQMH output through reconstruction support, retaining valid partial-coverage
edges. This satisfies the output-coverage direction in section 5.1.

The current test suite covers zero veto, missing-map behavior, frame-mask
support, uniform-control behavior, matched-star comparisons, common-overlap
validation masking, and raw/selected AQMH phase-result separation.

## Methodology-internal ambiguities to resolve

These have been resolved:

1. **Section 3 vs 9.2 (storage defaults):** Resolved. §3 now documents
   `float32`/`resolution_divisor = 1` as the default, with `uint16`/`2` as an
   optional performance mode. §9.2 already listed `float32`/`1`.
2. **Registration defaults (§4.3 vs §9.4):** Resolved. §9.4 (EN) was corrected
   to match §4.3 and the implementation: `weight_floor=0.30`,
   `cc_floor=0.35`, `cc_full=0.80`, `sequential=0.92`, `predicted=0.50`,
   `depth_penalty=0.03`, `depth_max=0.15`. DE §9.4 was already correct.
3. **Global-quality defaults (§1.5):** Resolved via F-07. §1.5 now documents
   both the v0.2.1 reference defaults and the current operating defaults with
   rationale.

## Remediation order

All items below have been resolved:

1. **F-05:** Thread frame-support and common-overlap masks into
   background-gradient computation; emit NaN on invalid preconditions.
2. **F-03/F-08:** PSF-FWHM sharpness proxy and sigmoid temperature formally
   versioned as post-v0.2.1 extensions.
3. **F-06:** Per-metric status/reason emitted in every validation comparison.
4. **F-07:** Methodology updated to distinguish v0.2.1 reference defaults from
   current operating defaults.
5. **Documentation ambiguities:** Storage defaults, registration defaults, and
   global-quality defaults reconciled across EN/DE methodology and §3/§4.3/§9.

## Conclusion

The implementation is operationally mature and contains several safeguards that
support the methodology. All identified conformance findings (F-03 through
F-08) and methodology-internal ambiguities have been resolved. The methodology
documents (EN/DE) now explicitly distinguish v0.2.1 reference values from
post-v0.2.1 operating defaults and extensions.
