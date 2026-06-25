# AutoBGE Integration Proposal — Replacing the Current BGE Algorithm

> **Status:** Proposal / Design Document  
> **Date:** 2026-06-24  
> **Author:** Cascade  
> **Related files:**  
> - `tile_compile_cpp/src/image/background_extraction.cpp` (current C++ BGE, ~4600 lines)  
> - `tile_compile_cpp/include/tile_compile/image/background_extraction.hpp`  
> - `tile_compile_cpp/include/tile_compile/config/configuration.hpp` (`BGEConfig`)  
> - `tile_compile_cpp/src/io/config.cpp` (BGE config parsing)  
> - `tile_compile_cpp/apps/runner_pipeline.cpp` (BGE phase orchestration)  
> - Reference: `AutoBGE.py` v2.0.2 (Siril script by Adrian Knagg-Baugh)

---

## 1. Executive Summary

The current C++ BGE implementation uses a **grid-based tile sampling** approach with single-stage polynomial or RBF surface fitting, robust estimators, autotune, and extensive safety guards. The Siril **AutoBGE.py** script uses a fundamentally different approach: **two-stage gradient removal** (polynomial → RBF) with **smart sample point generation** (gradient descent to dim spots, quartile-based distribution, border emphasis) and **per-channel image stretching** before processing.

This document proposes integrating the AutoBGE algorithm's core ideas into the C++ pipeline as a **new BGE method** (`bge.method: autobge`), replacing the current default while preserving the existing implementation as a fallback option.

---

## 2. Algorithm Comparison

### 2.1 Current C++ BGE

| Aspect | Implementation |
|--------|---------------|
| **Sampling** | Grid-based tile sampling: image divided into tiles, each tile evaluated for structure/foreground, background value estimated per tile |
| **Sample point placement** | Deterministic grid cells; tiles within each cell aggregated to cell median |
| **Foreground masking** | Structure score (DoG + gradient threshold), star mask dilation, saturation mask |
| **Fit method** | Single-stage: either polynomial (`fit.method: poly`) or RBF (`fit.method: rbf`), not both |
| **Robust estimation** | Sigma-clipped median, biweight, SExtractor mode; IRLS with Huber/Tukey weights |
| **Stretching** | None — operates on linear pixel values directly |
| **Downsampling** | None — operates at full resolution |
| **Autotune** | Cross-validation grid search over fit method, sample estimator, grid spacing, RBF mu |
| **Guards** | Flatness guard, slope guard, chroma guard, partial-channel revert, safety fallbacks |
| **Config complexity** | ~30+ parameters across `bge.*` |
| **Code size** | ~4600 lines |

### 2.2 AutoBGE.py

| Aspect | Implementation |
|--------|---------------|
| **Sampling** | Smart point generation: border points + quartile-based random points |
| **Sample point placement** | Gradient descent to dimmest local spot (15×15 patch median), avoids bright regions (top 50% excluded per quartile) |
| **Foreground masking** | Exclusion polygons (user-drawn); bright region exclusion via percentile |
| **Fit method** | **Two-stage**: (1) polynomial gradient removal → (2) RBF residual gradient removal |
| **Robust estimation** | Patch median (15×15) at each sample point |
| **Stretching** | Unlinked **non-linear** stretch per channel (target median 0.25) before processing, unstretch after |
| **Downsampling** | 4× area downsampling for speed, Lanczos4 upscaling of background model |
| **Autotune** | None |
| **Guards** | None — simple clip to [0,1] |
| **Config complexity** | 3 parameters: `npoints`, `polydegree`, `rbfsmooth` |
| **Code size** | ~1176 lines |

### 2.3 Key Algorithmic Differences

```
Current C++ BGE pipeline:
  Image → Tile grid → Structure mask → Per-tile bg estimate → 
  Coarse grid aggregation → [Poly OR RBF] fit → Guard checks → Apply

AutoBGE pipeline:
  Image → Stretch → Downsample 4× → 
  Stage 1: Sample points (gradient descent to dim) → Poly fit → Subtract → Normalize →
  Stage 2: Re-sample points → RBF fit → Subtract → Normalize →
  Unstretch → Apply
```

---

## 3. What AutoBGE Does Better

1. **Two-stage removal**: Polynomial captures large-scale gradients, RBF captures residual small-scale gradients. The current C++ BGE only does one or the other.

2. **Smart sample placement**: Gradient descent to dimmest local spot is more effective than grid-based sampling at avoiding nebulae and galaxy light. The current C++ BGE uses structure scores but still samples on a fixed grid.

3. **Image stretching**: Working on stretched data makes the gradient more visible and separable from signal. The current C++ BGE operates on linear data where gradients can be subtle.

4. **Simplicity**: 3 parameters vs 30+. Easier to configure and harder to misconfigure.

5. **Speed**: 4× downsampling + simple patch medians vs full-resolution tile metrics + autotune.

---

## 4. What the Current C++ BGE Does Better

1. **Robust statistics**: Sigma-clipped median, biweight, SExtractor mode, IRLS with Huber/Tukey weights — much more robust than simple patch medians.

2. **Safety guards**: Flatness, slope, and chroma guards prevent over-correction and color casts. AutoBGE has no guards.

3. **Autotune**: Cross-validation automatically selects best parameters. AutoBGE requires manual tuning.

4. **Foreground masking**: Structure score with DoG + gradient threshold is more sophisticated than percentile-based bright region exclusion.

5. **Canvas mask awareness**: Handles warped canvas borders correctly. AutoBGE assumes full-frame data.

6. **Per-channel atomic apply**: Prevents color casts from partial channel application. AutoBGE processes channels independently without cross-channel guards.

---

## 5. Proposed Integration: Hybrid Two-Stage BGE

### 5.1 Design Goals

- Adopt AutoBGE's **two-stage approach** (poly → RBF) as the new default
- Adopt AutoBGE's **smart sample point placement** (gradient descent to dim spots) with existing structure mask as automatic exclusion mask
- Adopt AutoBGE's **image stretching** before processing (non-linear per-channel stretch)
- Adopt AutoBGE's **downsampling** for speed
- **Keep** the current C++ BGE's robust statistics, safety guards, canvas mask awareness, and atomic RGB apply
- **Keep** autotune as optional (off by default with new method)
- **Reduce** config complexity — new method has fewer parameters

### 5.2 New Config Section

```yaml
bge:
  enabled: true
  method: autobge          # NEW: autobge | classic | auto
                           # autobge = two-stage poly+RBF (new default)
                           # classic = current grid-based single-stage
                           # auto = try autobge, fall back to classic on guard failure
  
  # --- AutoBGE parameters (new) ---
  autobge:
    num_sample_points: 100      # Total sample points for gradient fitting
    poly_degree: 2              # Polynomial degree for stage 1
    rbf_smooth: 0.1             # RBF smoothing for stage 2
    downsample_scale: 4         # Downsample factor (1=no downsample, 4=default)
    patch_size: 15              # Patch size for median estimation at sample points
    patch_estimator: "median"   # median | sigma_clipped_median (improvement over AutoBGE)
    stretch_target_median: 0.25 # Target median for pre-processing stretch
    border_margin: 10           # Margin in pixels for border sample points
    bright_exclusion_fraction: 0.5  # Exclude brightest N% of pixels per quartile
    gradient_descent_max_iters: 100  # Max iterations for dim-spot search
    random_seed: 42                 # Seed for deterministic quartile point selection
    
    # Safety (reuse existing guards)
    apply_guards: true          # Enable flatness/slope/chroma guards
    normalize_between_stages: true  # Match median of stretched image between poly and RBF stages
    # Mono handling: duplicate single channel to RGB, process, average back
    mono_mode: "rgb_duplicate"  # rgb_duplicate | disabled
  
  # --- Classic parameters (existing, used when method: classic) ---
  fit:
    method: rbf                 # poly | rbf (single-stage)
    # ... existing parameters remain unchanged
```

### 5.3 Algorithm: `autobge` Method

```
Input: R, G, B channels (Matrix2Df), canvas_valid_mask, BGEConfig

Phase A: Preprocessing
  1. For each channel c:
     a. Apply canvas_valid_mask (zero out invalid pixels)
     b. Stretch: unlinked **non-linear** stretch per channel (target_median=0.25)
        - Uses MTF-like formula per channel (not linear):
          ```
          shifted = channel - channel_min
          median_shifted = median(shifted)
          numerator   = (median_shifted - 1) * target_median * shifted
          denominator = median_shifted * (target_median + shifted - 1) - target_median * shifted
          stretched = numerator / (denominator + eps)   # eps = 1e-6 where denom==0
          ```
        - Record per-channel min and median for unstretch
     c. Downsample by downsample_scale using area interpolation
     d. Store stretched+downsampled image

Phase B: Stage 1 — Polynomial Gradient Removal
  2. Generate sample points on downsampled image:
     a. Add border points (corners + 5 per edge), with border_margin
     b. Divide into 4 quartiles
     c. Per quartile: exclude brightest 50%, randomly select N/4 points
     d. For each point: gradient descent to dimmest local spot (patch median)
     e. Filter points against canvas_valid_mask and existing structure mask (downsampled)
     f. Deduplicate points (border corners overlap with edge linspace points)
  3. For each channel c:
     a. Estimate patch median (patch_size×patch_size) at each sample point
     b. Fit polynomial surface (degree=poly_degree) via least squares
     c. Upscale polynomial surface to full resolution (Lanczos4)
     d. Subtract: image_after_poly = stretched_image - poly_background
     e. Normalize: shift median back to median of stretched_image

Phase C: Stage 2 — RBF Gradient Removal
  4. Re-generate sample points on image_after_poly (same procedure as step 2)
  5. For each channel c:
     a. Estimate patch median at each sample point
     b. Fit RBF surface (multiquadric, smooth=rbf_smooth) 
        - Use Eigen for RBF interpolation (replacing scipy.interpolate.Rbf)
     c. Upscale RBF surface to full resolution (Lanczos4)
     d. Subtract: corrected = image_after_poly - rbf_background
     e. Normalize: shift median back to median of stretched_image

Phase D: Postprocessing
  6. Unstretch corrected image (revert stretch from Phase A):
     - For each channel c:
       ```
       median_stretched = median(corrected[..., c])      # current median after BGE
       original_median  = stored medians[c]
       original_min     = stored mins[c]
       numerator   = (median_stretched - 1) * original_median * corrected[..., c]
       denominator = median_stretched * (original_median + corrected[..., c] - 1)
                       - original_median * corrected[..., c]
       unstretched = numerator / (denominator + eps) + original_min
       ```
  7. Unstretch total background (poly_background + rbf_background) with the same formula
  8. For each channel:
     a. Apply canvas_valid_mask
     b. Clamp model to [q05-guard_pad, q95+guard_pad] (reuse existing clamp)
  
Phase E: Guards (reuse existing)
  9. Flatness guard: compare spatial_background_spread before/after
  10. Slope guard: compare coarse_background_plane_slope before/after
  11. Chroma guard: compare log_chroma_std_background R/G and B/G
  12. If any guard fails:
      a. If method == auto: fall back to classic BGE
      b. If method == autobge: revert to pre-BGE state, log warning
  
Phase F: Apply
  13. Atomic RGB apply (reuse existing):
      corrected = channel_before - bg_model + pedestal
      where pedestal = model_stats.median (preserves background level)
```

### 5.4 Implementation Plan

#### Step 1: New C++ Functions in `background_extraction.cpp`

```cpp
// --- New functions for autobge method ---

/// Stretch a channel using unlinked non-linear stretch (per-channel)
struct StretchParams {
    std::vector<float> original_mins;   // per-channel min
    std::vector<float> original_medians; // per-channel median
    bool was_single_channel;
};
Matrix2Df stretch_channel(const Matrix2Df& channel, float target_median, 
                          StretchParams* params);
Matrix2Df unstretch_channel(const Matrix2Df& channel, const StretchParams& params);

/// Process mono images by duplicating to 3-channel, processing, then averaging back
Matrix2Df ensure_mono_from_processed_rgb(const Matrix2Df& R, const Matrix2Df& G, const Matrix2Df& B);

/// Downsample image by integer factor using area interpolation
Matrix2Df downsample_area(const Matrix2Df& image, int scale);

/// Upscale background model to target size using Lanczos4
Matrix2Df upscale_lanczos4(const Matrix2Df& background, int target_rows, int target_cols);

/// Gradient descent to dimmest local spot
struct SamplePoint { int x; int y; };
SamplePoint gradient_descent_to_dim(const Matrix2Df& luminance, 
                                     int start_x, int start_y,
                                     int max_iters, int patch_size);

/// Generate smart sample points (border + quartile + gradient descent)
std::vector<SamplePoint> generate_autobge_sample_points(
    const Matrix2Df& image, int num_points, int border_margin,
    float bright_exclusion_fraction, int patch_size,
    int max_descent_iters,
    const std::vector<uint8_t>* valid_mask_downsampled);

/// Fit polynomial surface via least squares (reuse existing polynomial code)
Matrix2Df fit_polynomial_autobge(
    const Matrix2Df& image_downsampled,
    const std::vector<SamplePoint>& points, int degree, int patch_size);

/// Fit RBF surface (multiquadric with smooth parameter)
Matrix2Df fit_rbf_autobge(
    const Matrix2Df& image_downsampled,
    const std::vector<SamplePoint>& points, float smooth, int patch_size);

/// Main entry point for autobge method
bool apply_autobge_extraction(
    Matrix2Df& R, Matrix2Df& G, Matrix2Df& B,
    const BGEConfig& config,
    BGEDiagnostics* diagnostics);
```

#### Step 2: RBF Implementation (Eigen)

The current C++ BGE already has RBF fitting with Eigen (`solve_rbf_model`). The key difference is:
- AutoBGE uses scipy's `Rbf(x, y, z, function='multiquadric', smooth=smooth, epsilon=1.0)` 
- scipy's `smooth` parameter adds `smooth * I` to the diagonal of the RBF matrix
- scipy's `epsilon=1.0` is fixed (our `mu` is configurable)

**Adaptation:** Reuse `solve_rbf_model` with:
- `rbf_phi = "multiquadric"` (already supported)
- `rbf_mu_factor` → set so that `mu ≈ 1.0` (fixed, matching scipy's `epsilon=1.0`)
- `rbf_lambda` → set to `rbf_smooth` value (maps to scipy's `smooth`)
- Skip IRLS (set `irls_max_iterations = 1`)
- **Important:** `epsilon` is fixed at `1.0` in AutoBGE; do not expose it as a tunable parameter for `autobge`

#### Step 3: Polynomial Fit (Eigen)

The current C++ BGE has `fit_polynomial_surface`. AutoBGE uses numpy's `lstsq` for polynomial fitting with all monomials `x^i * y^j` where `i + j <= degree`. The existing polynomial fit code can be reused with:
- `polynomial_order = poly_degree` (config mapping)
- Ensure the basis includes all mixed terms `x^i * y^j` for `i + j <= degree`
- Skip robust weighting (set `irls_max_iterations = 1`)

#### Step 4: Integration into `apply_background_extraction()`

```cpp
bool apply_background_extraction(Matrix2Df& R, Matrix2Df& G, Matrix2Df& B,
                                 const std::vector<TileMetrics>& tile_metrics,
                                 const TileGrid& tile_grid,
                                 const BGEConfig& config,
                                 BGEDiagnostics* diagnostics) {
    if (config.method == "autobge" || config.method == "auto") {
        // autobge does not need tile_metrics/tile_grid; classic path still uses them.
        bool ok = apply_autobge_extraction(R, G, B, config, diagnostics);
        if (ok || config.method == "autobge") {
            return ok;
        }
        // method == "auto" and autobge failed → fall through to classic
        std::cout << "[BGE] autobge failed; falling back to classic method\n";
        // Reset channels to pre-BGE state
        R = R_input; G = G_input; B = B_input;
    }
    
    // ... existing classic BGE code ...
}
```

#### Step 5: Config Parsing (`config.cpp`)

Add parsing for `bge.method` and `bge.autobge.*`:

```cpp
// In read_yaml():
if (bge_node && bge_node["method"]) {
    config.bge.method = bge_node["method"].as<std::string>("autobge");
}
if (bge_node && bge_node["autobge"]) {
    const auto& ab = bge_node["autobge"];
    config.bge.autobge.num_sample_points = ab["num_sample_points"].as<int>(100);
    config.bge.autobge.poly_degree = ab["poly_degree"].as<int>(2);
    config.bge.autobge.rbf_smooth = ab["rbf_smooth"].as<float>(0.1f);
    config.bge.autobge.downsample_scale = ab["downsample_scale"].as<int>(4);
    config.bge.autobge.patch_size = ab["patch_size"].as<int>(15);
    config.bge.autobge.patch_estimator = ab["patch_estimator"].as<std::string>("median");
    config.bge.autobge.stretch_target_median = ab["stretch_target_median"].as<float>(0.25f);
    config.bge.autobge.border_margin = ab["border_margin"].as<int>(10);
    config.bge.autobge.bright_exclusion_fraction = ab["bright_exclusion_fraction"].as<float>(0.5f);
    config.bge.autobge.gradient_descent_max_iters = ab["gradient_descent_max_iters"].as<int>(100);
    config.bge.autobge.random_seed = ab["random_seed"].as<int>(42);
    config.bge.autobge.normalize_between_stages = ab["normalize_between_stages"].as<bool>(true);
    config.bge.autobge.apply_guards = ab["apply_guards"].as<bool>(true);
}
```

#### Step 6: Config Structs

Two `BGEConfig` structs exist and must be kept in sync:
- `tile_compile_cpp/include/tile_compile/image/background_extraction.hpp` (runtime BGE config)
- `tile_compile_cpp/include/tile_compile/config/configuration.hpp` (YAML-serializable config)

Add to **both** structs:

```cpp
struct AutoBGEConfig {
    int num_sample_points = 100;
    int poly_degree = 2;
    float rbf_smooth = 0.1f;
    int downsample_scale = 4;
    int patch_size = 15;
    std::string patch_estimator = "median";  // median | sigma_clipped_median
    float stretch_target_median = 0.25f;
    int border_margin = 10;
    float bright_exclusion_fraction = 0.5f;
    int gradient_descent_max_iters = 100;
    int random_seed = 42;
    bool normalize_between_stages = true;
    bool apply_guards = true;
    std::string mono_mode = "rgb_duplicate";
};

struct BGEConfig {
    std::string method = "autobge";  // NEW: autobge | classic | auto
    AutoBGEConfig autobge;            // NEW
    // ... existing fields remain (including common_valid_mask in image/background_extraction.hpp) ...
};

// Note: `common_valid_mask` is already present in image/background_extraction.hpp
// and is used for canvas exclusion. No change needed there.
```

#### Step 7: OpenCV Usage

The current C++ BGE does not use OpenCV directly (uses Eigen + custom box blur). AutoBGE.py uses OpenCV for:
- `cv2.resize` (area interpolation for downsampling, Lanczos4 for upscaling)
- `cv2.fillPoly` (exclusion mask — not needed in C++ pipeline)

**Adaptation:** Use OpenCV (`cv::resize`) for downsampling and upscaling since OpenCV is already a dependency. Alternatively, implement area downsampling with the existing integral-image approach and Lanczos4 with Eigen.

#### Step 8: Tests

```cpp
// test_autobge.cpp
// 1. Synthetic gradient: create image with known polynomial gradient, verify removal
// 2. Two-stage: create image with poly + RBF gradient, verify both removed
// 3. Stretch/unstretch roundtrip: verify image is unchanged after stretch+unstretch
// 4. Sample point generation: verify points avoid bright regions
// 5. Guard integration: verify guards trigger on over-correction
// 6. Canvas mask: verify invalid pixels are handled
// 7. Comparison: run autobge vs classic on test image, compare flatness
```

#### Step 9: Deterministic Sampling

AutoBGE.py uses `np.random.choice` without a fixed seed. For reproducibility in the C++ pipeline:
- Use `std::mt19937` with a deterministic seed (`config.bge.autobge.random_seed`)
- Config option: `autobge.random_seed` (default `42`)
- This ensures identical results across runs on the same input

#### Step 10: Validation Rules

Add to `tile_compile_cpp/src/io/config.cpp` validation:

```cpp
if (bge.method != "autobge" && bge.method != "classic" && bge.method != "auto") {
  throw ValidationError("bge.method must be one of: autobge|classic|auto");
}
if (bge.autobge.num_sample_points < 10) {
  throw ValidationError("bge.autobge.num_sample_points must be >= 10");
}
if (bge.autobge.poly_degree < 1 || bge.autobge.poly_degree > 6) {
  throw ValidationError("bge.autobge.poly_degree must be in [1,6]");
}
if (bge.autobge.rbf_smooth < 0.0f) {
  throw ValidationError("bge.autobge.rbf_smooth must be >= 0");
}
if (bge.autobge.downsample_scale < 1) {
  throw ValidationError("bge.autobge.downsample_scale must be >= 1");
}
if (bge.autobge.patch_size < 3 || bge.autobge.patch_size % 2 == 0) {
  throw ValidationError("bge.autobge.patch_size must be odd and >= 3");
}
if (bge.autobge.patch_estimator != "median" &&
    bge.autobge.patch_estimator != "sigma_clipped_median") {
  throw ValidationError(
      "bge.autobge.patch_estimator must be one of: median|sigma_clipped_median");
}
if (bge.autobge.bright_exclusion_fraction <= 0.0f ||
    bge.autobge.bright_exclusion_fraction >= 1.0f) {
  throw ValidationError("bge.autobge.bright_exclusion_fraction must be in (0,1)");
}
if (bge.autobge.mono_mode != "rgb_duplicate" && bge.autobge.mono_mode != "disabled") {
  throw ValidationError("bge.autobge.mono_mode must be one of: rgb_duplicate|disabled");
}
```

---

## 6. Migration Strategy

### Phase 1: Implementation (non-breaking)
- Add `autobge` method alongside existing code
- Default `bge.method` stays `classic` during development
- Add `bge.method: auto` for automatic fallback

### Phase 2: Testing
- Run A/B comparison on real datasets (M31, M42, NGC281)
- Compare: flatness, chroma std, visual inspection, star photometry
- Test stretch/no-stretch to verify necessity of non-linear stretch
- Test `patch_estimator: median` vs `sigma_clipped_median` on star-rich fields
- Test mono images via `mono_mode: rgb_duplicate`
- Tune defaults

### Phase 3: Switch Default
- Change `bge.method` default to `autobge`
- Keep `classic` available for users who need it
- Update documentation

### Phase 4: Cleanup (optional, later)
- If `autobge` proves superior, consider deprecating `classic`
- Remove autotune code if `autobge` doesn't need it

---

## 7. Risk Analysis

| Risk | Mitigation |
|------|-----------|
| AutoBGE's simple patch median is less robust than current estimators | Add `patch_estimator: sigma_clipped_median` option; default can stay `median` for exact compatibility |
| No exclusion zones in C++ pipeline (no user interaction) | Rely on gradient descent + bright exclusion + structure mask from existing code |
| Stretch/unstretch could introduce numerical errors | Use float32 throughout, verify roundtrip on test images |
| RBF with many sample points (100+) could be slow | Downsample 4× reduces grid size; RBF matrix is N×N where N=100, trivial for Eigen |
| Guards might reject autobge more often (different correction profile) | Tune guard thresholds for autobge; use `method: auto` fallback |
| Per-channel stretch could affect color balance | Unstretch restores original values; guards catch chroma issues |

---

## 8. Expected Performance

| Metric | Current C++ BGE | AutoBGE (estimated) |
|--------|-----------------|---------------------|
| Sample generation | O(tiles × tile_pixels) full-res | O(num_points × patch_size² × descent_iters) on 4× downsampled |
| Polynomial fit | O(grid_cells × poly_terms²) | O(100 × poly_terms²) — similar |
| RBF fit | O(M² × irls_iters) where M=grid_cells | O(100² × 1) — faster (no IRLS) |
| Surface rendering | Full-res polynomial/RBF evaluation | Lanczos4 upscale of downsampled surface |
| Autotune | O(candidates × full_pipeline) | N/A (disabled by default) |
| Total (estimated) | 5-30s with autotune | 1-5s without autotune |

---

## 9. Open Questions

1. **Stretch necessity:** The AutoBGE stretch is **non-linear** (MTF-like), not linear. Is it essential for quality, or can we skip it and work on linear data (like the current C++ BGE)? The stretch makes gradients more visible but adds complexity and numerical risk. **Recommendation:** Implement the exact non-linear stretch first, then test with a simpler linear stretch or no stretch.

2. **Sample point count:** AutoBGE defaults to 100 points. The current C++ BGE uses a grid with potentially hundreds of cells. Should we increase the default for larger images? **Recommendation:** Make it proportional to image area: `num_points = max(100, image_area / 10000)`.

3. **Exclusion zones:** AutoBGE supports user-drawn exclusion polygons. The C++ pipeline has no user interaction during BGE. **Recommendation:** Use the existing `build_modeled_foreground_mask` as a mandatory exclusion mask for sample point generation. Without it, gradient descent will still move points away from bright stars, but nebulae/galaxy halos may be sampled incorrectly.

4. **RBF epsilon:** AutoBGE uses `epsilon=1.0` (fixed). The current C++ BGE computes `mu = rbf_mu_factor * grid_spacing`. Which is better for the autobge method? **Recommendation:** Use `mu = 1.0` (matching AutoBGE) since sample points are in downsampled pixel coordinates.

5. **Guard compatibility:** The existing guards were tuned for the classic BGE. Should we use different thresholds for autobge? **Recommendation:** Start with existing thresholds, adjust based on A/B testing.

---

## 10. Summary

The AutoBGE algorithm offers a simpler, potentially more effective approach to background gradient extraction through its two-stage poly→RBF design and smart sample point placement. The integration proposal combines the best of both approaches:

- **From AutoBGE:** Two-stage removal, gradient-descent sample placement, image stretching, downsampling, simplicity
- **From current C++ BGE:** Robust statistics, safety guards, canvas mask awareness, atomic RGB apply, autotune (optional)

The implementation is additive (new method alongside existing), non-breaking, and can be validated through A/B comparison before becoming the default.
