# Deep Analysis of the Current AQMH/Classic Process Flow

## Executive Summary

This analysis evaluates the current `tile_compile_cpp` implementation for
astronomical images. AQMH is the default reconstruction method; Classic Tile
Compile remains a separate alternative. The shared flow includes registration,
prewarp/canvas harmonization, normalization, global metrics, common-overlap
masking, stacking, debayer, astrometry, BGE, PCC and HMS.

The authoritative AQMH execution path is:

```text
SCAN_INPUT -> REGISTRATION -> PREWARP -> CHANNEL_SPLIT
-> NORMALIZATION -> GLOBAL_METRICS -> TILE_GRID -> COMMON_OVERLAP
-> AQMH_MAPS(19) -> AQMH_GLOBAL_QUALITY(20)
-> AQMH_RECONSTRUCTION(21) -> AQMH_DIAGNOSTICS(22)
-> STACKING -> DEBAYER -> ASTROMETRY -> BGE -> PCC
-> HYPERMETRIC_STRETCH -> DONE
```

The most relevant remaining risks are operational tuning, numerical edge
cases, support handling at frame borders, and throughput on slower storage.

## 1. Core Architecture Analysis

The codebase is organized into modular components that roughly match the phases described in the specification:

```
src/
├── astrometry/     # Astrometric solving and photometric calibration
├── core/           # Basic types and utilities
├── image/          # Image processing (CFA, normalization)
├── io/             # Input/output operations
├── metrics/        # Quality metrics calculation
├── pipeline/       # Pipeline orchestration
├── reconstruction/ # Tile-based reconstruction
└── registration/   # Image registration
```

The implementation follows a linear pipeline approach as required by the specification, with no feedback loops detected in the codebase.

The shared core enum sequence is defined in
`include/tile_compile/core/types.hpp`. AQMH uses extension enum values 19–22;
they are not replacements with the old enum values 8 and 9:

`0 SCAN_INPUT -> 1 REGISTRATION -> 2 PREWARP -> 3 CHANNEL_SPLIT -> 4 NORMALIZATION -> 5 GLOBAL_METRICS -> 6 TILE_GRID -> 7 COMMON_OVERLAP -> 19 AQMH_MAPS -> 20 AQMH_GLOBAL_QUALITY -> 21 AQMH_RECONSTRUCTION -> 22 AQMH_DIAGNOSTICS -> 12 STACKING -> 13 DEBAYER -> 14 ASTROMETRY -> 15 BGE -> 16 PCC -> 17 HYPERMETRIC_STRETCH -> 18 DONE`

## 2. Registration Analysis

### 2.1 Observations

The registration module uses a cascaded fallback strategy with CFA-aware full-frame prewarping. The current pipeline performs registration before channel processing and keeps CFA structure intact until debayer.

### 2.2 Risks and Observations

1. **CFA Subplane Warping**: The implementation in `warp_cfa_mosaic_via_subplanes()` correctly handles the Bayer phase preservation, but there's a potential issue with the boundary handling when pixel phases are misaligned at image edges.

2. **Residual Monitoring**: A dedicated residual-quality KPI (with hard thresholds) should be exposed more prominently in artifacts/reports for faster diagnostics on difficult sessions.

3. **Fallback Transparency**: In difficult data, fallback-heavy registration can still finish "ok"; clearer fallback summaries improve operator decisions.

### 2.3 Mathematical Correctness

The registration approach using ECC is mathematically sound, but the delta shifts for the CFA subplanes need careful consideration:

```cpp
// Current implementation in cfa_processing.cpp
float new_tx = t_x + (a2_00 * dx + a2_01 * dy) - dx;
float new_ty = t_y + (a2_10 * dx + a2_11 * dy) - dy;
```

This correctly accounts for the sub-pixel shift needed for each Bayer phase, but in rare cases with high rotations (> 10°), the approximation may introduce very small phase errors.

### 2.4 Comparison with Existing Solutions

The CFA-based approach (Path B) is more methodologically sound than many existing solutions which typically debayer first (Path A). In the current `v3.3.9` reference this is complemented by the optional but tightly constrained pre-warp CFA defect-pixel suppression step, which remains outside the mandatory linear core.

## 3. Debayering and Color Processing

### 3.1 Observations

The production pipeline keeps CFA structure during registration/reconstruction and performs OSC debayering at the dedicated DEBAYER phase.

### 3.2 Issues Identified

1. **Nearest Neighbor Trade-off**: The current nearest-neighbor method is fast and deterministic but can show aliasing in high-frequency structures.

2. **Sensor-Specific Response**: G1/G2 are combined with a simple average; this is generally acceptable but may be suboptimal on sensors with asymmetric green response.

3. **Photometric QA Scope**: Linearity is preserved in the core, but additional photometric QA metrics after PCC would improve traceability.

### 3.3 Mathematical Correctness

The CFA processing is generally mathematically correct. However, the handling of green channels deserves attention:

```cpp
// Current implementation in cfa_processing.cpp
float g1 = mosaic(y * 2 + g1_row, x * 2 + g1_col);
float g2 = mosaic(y * 2 + g2_row, x * 2 + g2_col);
channels.G(y, x) = 0.5f * (g1 + g2);
```

This assumes equal response in both green filters, which is generally valid but may introduce slight inaccuracies with some sensors.

### 3.4 Comparison with Existing Solutions

The implementation prioritizes predictable linear processing over visually optimized debayering variants (for example, VNG/AHD).

## 4. Normalization and Metric Calculation

### 4.1 Observations

The normalization step is now specified more strictly than in earlier methodology revisions:
- additive background subtraction
- deterministic photometric scaling
- per-channel normalization
- robust statistical measures (median, MAD)

### 4.2 Risks and Observations

1. **Potential Numerical Instability**: In `metrics.cpp`, there are several places where division could lead to numerical instability if denominators are very small:

```cpp
// Potential division by zero if sum is very small
weights /= sum;
```

2. **Parameter Sensitivity**: Clamp/exponent settings can materially change frame discrimination and should be tuned per dataset class.

### 4.3 Mathematical Correctness

The normalization approach is mathematically sound, with good use of robust statistics. In `v3.3.9`, the key methodological requirement is the clean separation of additive background `B_f` and multiplicative photometric scale `P_f`; any documentation or implementation description that collapses this back into pure `1/background` scaling is outdated.

The gradient energy calculation could still be improved:

```cpp
// Current implementation calculates median of gradient magnitude squared
m.gradient_energy = grad_vals.empty() ? 0.0f : core::median_of(grad_vals);
```

This is reasonable, but the specification suggests `E = mean(|∇I|²)`, which would be more sensitive to structural content.

### 4.4 Comparison with Existing Solutions

The use of MAD-based normalization is state-of-the-art and superior to many existing implementations that use simple mean/stddev measures, which are more susceptible to outliers.

## 5. Tile Generation and Analysis

### 5.1 Observations

The `TILE_GRID` phase is shared infrastructure and is primarily consumed by
Classic. AQMH still receives the common canvas and support masks, but its
quality analysis is performed by `AQMH_MAPS`, not by tile metrics. The Classic
branch follows the tile-specific behavior described below.

The Classic implementation follows the specification by:
- using adaptive tile sizes based on FWHM
- using support-aware overlap-add semantics
- blending star-based and structure-based local metrics

### 5.2 Risks and Observations

1. **FWHM Estimation**: The FWHM estimation in `metrics.cpp` uses a 1D Gaussian approximation which may not be optimal for all seeing conditions, particularly for elliptical PSFs.

2. **Tile Boundary Logic**: The code in `adaptive_tile_grid.cpp` has edge cases where tile coverage might be incomplete:

```cpp
// Could miss edge pixels in rare cases
if (!xs.empty() && xs.back() + tile_size < image_width) xs.push_back(image_width - tile_size);
```

3. **Star Detection Thresholding**: The current implementation uses fixed parameters for star detection, which might not be optimal for all types of astronomical data.

### 5.3 Mathematical Correctness

The tile analysis implementation is generally correct, but the old hard STAR/STRUCTURE interpretation is no longer the right methodological framing for `v3.3.9`:

```cpp
Q_blend = eta_t * Q_star + (1 - eta_t) * Q_struct
```

The specification requires MAD normalization of all metrics before combining.

### 5.4 Comparison with Existing Solutions

The tile-based approach is more sophisticated than common global selection methods used in many stacking programs. It's conceptually similar to Astro Pixel Processor's weighted analysis, but with more robust mathematical foundations.

## 6. Reconstruction

### 6.1 Observations

In the AQMH branch, reconstruction is pixel-wise and occurs in
`AQMH_RECONSTRUCTION` (21), followed by independent `AQMH_DIAGNOSTICS` (22).
The Classic branch implements:
- tile-wise weighted averaging
- optional tile denoise (soft-threshold + optional Wiener)
- support-aware overlap-add with explicit valid-support semantics

### 6.2 Issues Identified

1. **Wiener Filter Implementation**: The current Wiener filter implementation has fixed parameters that may not adapt well to different noise characteristics:

```cpp
// Parameters could be more adaptive
cv::Mat H = power - sigma_sq;
cv::threshold(H, H, 0.0, 0.0, cv::THRESH_TOZERO);
```

2. **I/O Throughput Dependency**: Disk-backed frame caching reduces RAM pressure but shifts performance sensitivity to storage throughput/latency.

3. **Sigma Clipping**: The sigma clipping implementation can be computationally intensive as it processes each pixel independently.

### 6.3 Mathematical Correctness

The overlap-add reconstruction is mathematically correct, using proper normalization, but the key `v3.3.9` point is that tiles must no longer be median/MAD-renormalized before OLA:

```cpp
// Proper normalization in the overlap-add process
for (int i = 0; i < result.size(); ++i) {
    if (weight_sum.data()[i] > 0) {
        result.data()[i] /= weight_sum.data()[i];
    }
}
```

The Wiener filter implementation could still be improved with adaptive parameters based on local SNR, but that is now clearly outside the mandatory quality core.

### 6.4 Comparison with Existing Solutions

The implementation's approach to Wiener filtering is comparable to standard implementations in image processing libraries, but lacks the adaptivity found in more advanced denoising methods like BM3D or Non-Local Means.

## 7. Clustering and Synthetic Frames

### 7.1 Observations

State-based clustering and synthetic-frame generation are Classic-only in the
current flow. AQMH does not enter these phases because it has already produced
the final pixel-wise reconstruction.

### 7.2 Issues Identified

1. **Operational Risk**: Cluster quality can degrade on heterogeneous datasets when cluster count and eligibility thresholds are aggressive.

2. **Feature Stability**: State-vector features should be monitored for drift across long sessions (cloud passages, gradients) to avoid unstable cluster assignments.

### 7.3 Mathematical Correctness

The implemented clustering follows the intended state-vector approach and supports mode-dependent skip behavior in reduced/emergency paths. In `v3.3.9`, the old fixed gate `N >= 200` is superseded by the mode framework with the effective lower bound `N >= max(N_red, 50)`, and the downstream aggregation is explicitly mass-preserving through `M_{k,c}`.

```
v_f = (G_f, ⟨Q_local⟩, Var(Q_local), CC̄_tiles, WarpVar̄, invalid_frac)
```

### 7.4 Comparison with Existing Solutions

State-based clustering is an advanced approach compared to traditional methods like simple frame selection or global weighting. It's conceptually similar to approaches in Lucky Imaging but more mathematically rigorous.

## 8. Astrometry and Photometric Calibration

### 8.1 Observations

The implementation includes production components for:
- Astrometric solving
- Background Gradient Extraction (BGE)
- Photometric color calibration
- Catalog matching

### 8.2 Issues Identified

1. **External Dependencies**: Astrometry relies on ASTAP binaries/catalogs and is therefore environment-sensitive.

2. **Catalog Availability**: PCC quality depends on available catalog source/data quality (local Siril catalog or online fallback).

### 8.3 Mathematical Correctness

The PCC approach is mathematically consistent for diagonal color correction in linear space. In `v3.3.9`, BGE before PCC and the optional post-PCC isolated chroma-speckle suppression are explicitly documented as downstream extensions outside the mandatory linear core. Result quality remains dependent on catalog quality/coverage and robust star matching.

### 8.4 Comparison with Existing Solutions

The approach is comparable to solutions in popular tools like SIRIL, ASTAP, and AstroImageJ and is fully integrated into the runner phases (ASTROMETRY + BGE + PCC).

## 9. Performance and Scalability

### 9.1 Observations

The implementation uses OpenCV for many image processing operations, which provides good performance but could be further optimized.

### 9.2 Risks and Observations

1. **Disk I/O Pressure**: Disk-backed frame caching reduces RAM pressure but shifts bottlenecks to storage throughput/latency.

2. **Parallel Efficiency Ceiling**: Core tile processing is parallelized, but overall runtime still depends on I/O and phase-specific serial sections.

3. **I/O Strategy**: The implementation doesn't follow the I/O strategy recommendations in Appendix C.3 of the specification.

### 9.3 Comparison with Existing Solutions

Most astronomical image processing tools face similar challenges with large datasets. Tools like PixInsight and AstroPixelProcessor have more mature memory management strategies.

## 10. Recommendations

### 10.0 Implementation Status (2026-02-13)

The following high-impact quality actions are implemented in the current code path:

1. [implemented] **Canvas-/Parity-safe registration flow**
   - Dedicated `PREWARP` phase with full-frame canvas harmonization.
   - CFA-aware subplane warp for OSC and parity-safe offset handling.

2. [implemented] **COMMON_OVERLAP gating**
   - Explicit common-overlap phase and tile/pixel validity filtering to avoid
     edge/empty-canvas contamination.

3. [implemented] **Robust PCC fit guardrails**
   - PCC uses robust log-chromaticity deltas with sigma-clipped location,
     bounded channel correction factors, and stronger resistance to pathological
     field color casts.

4. [implemented] **PCC photometry contamination filter**
   - Aperture/annulus photometry rejects unstable annuli (IQR gate) to reduce
     nebula/gradient contamination in star color fitting.

### 10.1 Priority Improvements

1. **Observability & QA**:
   - Expose clearer residual/fallback diagnostics in report artifacts
   - Add compact phase-level quality KPIs for faster triage

2. **Metric Tuning Discipline**:
   - Define per-target presets for weight/clamp/exponent ranges
   - Add regression checks for scaling stability under aggressive stretch

3. **Tile Artifact Prevention**:
   - Keep strict regression checks for tile-pattern indicators
   - Validate synthetic tile-weighted behavior against bright-core edge cases
   - Continue enforcing support-aware OLA invariants at coverage boundaries

### 10.2 Algorithmic Enhancements

1. **Debayering**:
   - [current] Deterministic nearest-neighbor demosaic in production path.
   - [offen] Optional higher-order demosaic variants remain future work if needed.

2. **Denoise Strategy**:
   - [erledigt] Dataset-aware chroma denoise scaling is implemented.
   - [teilweise offen] Continue expanding practical presets and optional structure-preserving alternatives.

3. **FWHM Estimation**:
   - [current] Robust FWHM-based tile sizing and frame-level quality usage in production.
   - [offen] Optional next step: stricter PSF-model validation/fit variants per dataset class.

### 10.3 Performance Optimizations

1. **Memory Management**:
   - Implement tile-based processing with streaming I/O
   - Follow the recommendations in Appendix C.3 of the specification

2. **Pipeline Throughput**:
   - [current] CUDA/OpenCL acceleration covers PREWARP, AQMH maps, classic tile reconstruction, synthetic reconstruction, and STACKING.
   - [current] CUDA additionally accelerates streaming AQMH_RECONSTRUCTION; Cherry-Pick remains CPU-only.
   - [current] Per-worker CUDA streams, cached AQMH filters, and concurrent RGB stacking reduce synchronization and allocation overhead.
   - [open] Continue profiling CPU/GPU/I/O phase splits; storage throughput can remain the limiting factor.

### 10.4 Validation and Testing

1. **Synthetic Test Cases**:
   - Develop synthetic test images with known properties
   - Validate the reconstruction against ground truth

2. **Comparative Analysis**:
   - Compare results with established tools like SIRIL and PixInsight
   - Quantify improvements in FWHM, SNR, and detail preservation

## 11. Conclusion

The tile_compile_cpp implementation provides a solid v3.2 production foundation for tile-based quality reconstruction. The approach is mathematically consistent and addresses key limitations of traditional global-only stacking.

The remaining work is primarily operational hardening: better diagnostics, stricter regression protection against scaling/tile artifacts, and performance tuning for large runs on mixed storage systems.

The most significant improvements now come from:
1. stronger run-time observability and report-level KPIs,
2. safer tuning presets for metric/weight parameters,
3. regression tests focused on bright-core scaling and tile imprinting,
4. targeted throughput optimization based on measured bottlenecks.

With these improvements, the implementation would provide state-of-the-art performance for astronomical image processing, particularly for datasets with variable seeing conditions and atmospheric transparency.
