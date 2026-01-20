# Methodik v3 – Assumptions Analysis and Plausibility Review

**Document Type:** Critical Review
**Reference:** `tile_basierte_qualitatsrekonstruktion_methodik_en.md` v3

---

## 1. Core Assumptions Assessment

### 1.1 Linear Data (§2)

**Assumption:** Data are linear (no stretch, no non-linear operators)

**Plausibility:** ✅ **Fully Justified**

**Rationale:**
- Weighted averaging is only physically meaningful for linear data
- Non-linear transforms (asinh, log, histogram stretch) distort SNR relationships
- Quality metrics (B, σ, E) lose interpretability on stretched data

**Implementation Status:** A linearity validator exists (e.g. `tile_compile_backend.linearity`), but enforcement depends on the active runner/pipeline path.

**Recommendation:** No change needed.

---

### 1.2 OSC Data, Channel-Separated Processing (§2)

**Assumption:** OSC data; processing is separated per color channel

**Plausibility:** ✅ **Fully Justified**

**Rationale:**
- Different wavelengths have different atmospheric effects (dispersion, scattering)
- Debayer interpolation artifacts are wavelength-specific
- Channel coupling would mix independent noise sources

**Edge Cases:**
- Mono cameras: Methodology applies directly (single channel)
- LRGB: Each filter treated as independent channel

**Recommendation:** Clarify mono camera handling explicitly.

---

### 1.3 Full Registration (§2)

**Assumption:** Full registration (translation + rotation, no residuals)

**Plausibility:** ⚠️ **Idealized**

**Reality:**
- Sub-pixel residuals are unavoidable
- Atmospheric dispersion causes wavelength-dependent shifts
- Field rotation (alt-az mounts) may have interpolation artifacts

**Impact of Violation:**
- Tile metrics become position-dependent artifacts
- Star roundness metric degrades
- FWHM estimates may be biased

**Recommended Change:**
```yaml
assumptions:
  registration_residual_warn_px: 0.5
  registration_residual_max_px: 1.0
```

**New Validation:**
- Compute residual statistics from star matches
- Warn if residual > `registration_residual_warn_px`
- Abort if residual > `registration_residual_max_px`

---

### 1.4 Large Number of Frames (≥800)

**Assumption:** Large number of short exposures (typically ≥800 frames)

**Plausibility:** ⚠️ **Conservative**

**Analysis:**

| Frame Count | Statistical Power | Recommendation |
|-------------|-------------------|----------------|
| ≥800 | Full methodology | Normal operation |
| 200-799 | Reduced but valid | Reduce cluster count (10-20) |
| 50-199 | Limited | Simple mode (no clustering) |
| <50 | Insufficient | Abort or fall back to median |

**Impact of Low Frame Count:**
- Cluster statistics unreliable (15-30 clusters need ≥500 samples)
- Local metric variance high
- Fallback tiles increase

**Recommended Change:**
```yaml
assumptions:
  frames_min: 50
  frames_optimal: 800
  frames_reduced_threshold: 200
```

**Reduced Mode Behavior:**
- Skip state clustering
- Skip synthetic frame generation
- Proceed deterministically with direct tile-weighted reconstruction
- Output a validation warning in the report

---

### 1.5 No Frame Selection (§2)

**Assumption:** No frame selection – every frame contributes

**Plausibility:** ✅ **Fully Justified**

**Rationale:**
- Information-theoretically optimal
- Weight-based contribution is selection via continuous function
- Hard selection loses information at threshold boundary

**Edge Case – Transient Artifacts:**
- Cosmic rays, satellites, aircraft should still be rejected
- This is **artifact rejection**, not frame selection
- Sigma-clipping pre-processing is compatible

**Recommendation:** Clarify distinction:
- Frame selection (forbidden): Discarding frames based on quality
- Artifact rejection (allowed): Pixel-level transient removal

---

## 2. Implicit Assumptions (Not Explicitly Stated)

### 2.1 Uniform Exposure Time

**Implicit Assumption:** All frames have identical exposure time

**If Violated:**
- Background levels not comparable
- Noise levels not comparable
- Normalization may not fully compensate

**Recommendation:** Add explicit check and normalize by exposure time.

### 2.2 Stable Optical Configuration

**Implicit Assumption:** Focus, field rotation, optical train unchanged

**If Violated:**
- FWHM trends may be systematic, not atmospheric
- Tile weights may encode equipment drift

**Recommendation:** Add optional focus tracking metric.

### 2.3 No Significant Tracking Errors

**Implicit Assumption:** Mount tracking is accurate

**If Violated:**
- Star trails affect all metrics
- FWHM becomes meaningless

**Recommendation:** Detect via elongation metric and abort/warn.

---

## 3. Recommended Specification Changes

### 3.1 Add to §2 (Assumptions)

```markdown
- Uniform exposure time across all frames
- Stable optical configuration (focus, field curvature)
- Tracking errors < 1 pixel per exposure
```

### 3.2 Add Tolerance Configuration

```yaml
assumptions:
  exposure_time_tolerance_percent: 5
  registration_residual_warn_px: 0.5
  registration_residual_max_px: 1.0
  elongation_warn: 0.3
  elongation_max: 0.4
  reduced_mode_skip_clustering: true
  reduced_mode_cluster_range: [5, 10]
```

### 3.3 Add Reduced Mode Specification

```markdown
## Reduced Mode (50-199 frames)

When frame count is below optimal but above minimum:
- Skip state-based clustering (§10)
- No synthetic frame generation
- Direct tile-weighted stacking
- Output validation warning
```

---

## 4. Summary

| Assumption | Plausibility | Change Needed |
|------------|--------------|---------------|
| Linear data | ✅ Correct | None |
| Channel separation | ✅ Correct | Clarify mono |
| Full registration | ⚠️ Idealized | Add tolerance |
| ≥800 frames | ⚠️ Conservative | Add reduced mode |
| No frame selection | ✅ Correct | Clarify artifacts |
| Uniform exposure | ❓ Implicit | Make explicit |
| Stable optics | ❓ Implicit | Make explicit |
| Good tracking | ❓ Implicit | Make explicit |

---

## 5. Conclusion

The core methodology assumptions are **scientifically sound**. Recommended changes are:

1. **Add tolerance thresholds** for registration and tracking
2. **Define reduced mode** for smaller datasets
3. **Make implicit assumptions explicit** in specification
4. **Clarify artifact rejection** vs. frame selection

These changes improve robustness without altering the fundamental methodology.
