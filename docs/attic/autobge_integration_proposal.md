# AutoBGE Integration Proposal — Dual-Method BGE (Classic + AutoBGE)

> **Status:** Proposal / Design Document  
> **Date:** 2026-06-24 (updated 2026-06-26)  
> **Author:** Cascade  
> **Related files:**  
> - `tile_compile_cpp/src/image/background_extraction.cpp` (current C++ BGE, ~4600 lines)  
> - `tile_compile_cpp/include/tile_compile/image/background_extraction.hpp`  
> - `tile_compile_cpp/include/tile_compile/config/configuration.hpp` (`BGEConfig`)  
> - `tile_compile_cpp/src/io/config.cpp` (BGE config parsing)  
> - `tile_compile_cpp/apps/runner_pipeline.cpp` (BGE phase orchestration)  
> - `tile_compile_cpp/include/tile_compile/core/types.hpp` (Phase enum)  
> - `web_frontend_v3/js/components/phase-list.js` (Run Monitor phase list)  
> - Reference: `AutoBGE.py` v2.0.2 (Siril script by Adrian Knagg-Baugh)

---

## 1. Executive Summary

The current C++ BGE implementation uses a **grid-based tile sampling** approach with single-stage polynomial or RBF surface fitting, robust estimators, autotune, and extensive safety guards. The Siril **AutoBGE.py** script uses a fundamentally different approach: **two-stage gradient removal** (polynomial → RBF) with **smart sample point generation** (gradient descent to dim spots, quartile-based distribution, border emphasis) and **per-channel image stretching** before processing.

This document proposes adding AutoBGE as a **second BGE method** alongside the existing classic implementation. Users choose via `bge.method: none | classic | autobge`. The options are **mutually exclusive** — selecting one automatically disables the others. The default is **`none`** so BGE remains opt-in and existing runs are not changed by adding the new method selector. The Run Monitor displays the active method in the phase label: **"BGE (Skipped)"**, **"BGE (Classic)"**, or **"BGE (AutoBGE)"**.

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

C++ AutoBGE pipeline (proposed):
  Image → Working transform (none/linear/mtf) → Downsample 4× → 
  Stage 1: Sample points (gradient descent to dim) → Poly fit → Subtract → Normalize →
  Stage 2: Re-sample points → RBF fit → Subtract → Normalize →
  Inverse transform → Derive linear model (without pedestal) → Apply (+ pedestal)
```

---

## 3. What AutoBGE Does Better

1. **Two-stage removal**: Polynomial captures large-scale gradients, RBF captures residual small-scale gradients. The current C++ BGE only does one or the other.

2. **Smart sample placement**: Gradient descent to dimmest local spot is more effective than grid-based sampling at avoiding nebulae and galaxy light. The current C++ BGE uses structure scores but still samples on a fixed grid.

3. **Image stretching**: AutoBGE.py uses an MTF-style non-linear stretch that makes gradients more visible and separable from signal. The C++ implementation generalizes this to `stretch_mode: none | linear | mtf`; the conservative default is `linear` to preserve the additive background contract. Full MTF parity is available as experimental `stretch_mode: mtf`.

4. **Simpler core controls**: AutoBGE.py exposes only `npoints`, `polydegree`, and
   `rbfsmooth`. The C++ integration intentionally adds safety, reproducibility,
   masking, and working-space options, so it is not a strict 3-parameter port.
   The user-facing surface should still remain smaller than classic BGE.

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

- Provide **three** BGE options: `bge.method: none | classic | autobge`
- **Mutual exclusion**: when `autobge` is selected, classic parameters are ignored; when `classic` is selected, autobge parameters are ignored; `none` disables BGE entirely (equivalent to `bge.enabled: false`).
- **Default disabled**: `none` is the default so BGE remains an explicit user choice.
- Adopt AutoBGE's **two-stage approach** (poly → RBF)
- Adopt AutoBGE's **smart sample point placement** (gradient descent to dim spots) with existing structure mask as automatic exclusion mask
- Make AutoBGE's pre-fit working stretch configurable: `none | linear | mtf`. Use
  `linear` as the conservative default and treat the original AutoBGE-style MTF
  stretch as an experimental mode to validate against `none` and `linear`.
- Adopt AutoBGE's **downsampling** for speed
- **Keep** the current C++ BGE's robust statistics, safety guards, canvas mask awareness, and atomic RGB apply (applied to both methods)
- **Keep** autotune as optional (only for classic method; autobge has no autotune)
- **Reduce** config complexity — autobge method has fewer parameters
- **Run Monitor** shows the active method in the phase label: "BGE (AutoBGE)", "BGE (Classic)", or "BGE (Skipped)"

**Naming note:** the current code already uses `bge.fit.method` for the classic surface fitter (`poly | rbf | modeled_mask_mesh | ...`). The new `bge.method` is intentionally a higher-level BGE engine selector (`none | classic | autobge`), not a replacement for `bge.fit.method`. Implementation must keep diagnostics explicit:
- `diagnostics.bge_method` or equivalent: `none | classic | autobge`
- `diagnostics.fit_method`: classic fit method or AutoBGE stage method (`poly`, `rbf`, `poly+rbf`)

### 5.2 New Config Section

```yaml
bge:
  enabled: false           # Legacy compatibility flag; method=none is authoritative for new configs.
  method: none             # NEW default: none | classic | autobge (mutually exclusive)
                           # none = BGE disabled entirely (no gradient extraction)
                           # autobge = two-stage poly+RBF with smart sampling
                           # classic = current grid-based single-stage (poly OR rbf)
                           # When method=autobge, all classic bge.fit.* params are ignored.
                           # When method=classic, all bge.autobge.* params are ignored.
                           # When method=none, all BGE params are ignored.
                           # `method` is authoritative when present.
                           # `enabled` is legacy-only when method is absent:
                           #   enabled=true -> method=classic
                           #   enabled=false/missing -> method=none
  
  # --- AutoBGE parameters (new) ---
  autobge:
    num_sample_points: 0        # 0=auto; >0 explicit total sample points
    poly_degree: 2              # Polynomial degree for stage 1
    rbf_smooth: 0.1             # RBF smoothing for stage 2
    downsample_scale: 4         # Downsample factor (1=no downsample, 4=default)
    patch_size: 15              # Patch size for median estimation at sample points
    patch_estimator: "median"   # median | sigma_clipped_median (improvement over AutoBGE)
    stretch_mode: "linear"      # none | linear | mtf; mtf matches AutoBGE-style nonlinear stretch
    stretch_target_median: 0.25 # Target median for mtf mode only
    border_margin: 10           # Margin in pixels for border sample points
    bright_exclusion_fraction: 0.5  # Exclude brightest N% of pixels per quartile
    gradient_descent_max_iters: 100  # Max iterations for dim-spot search
    random_seed: 42                 # Seed for deterministic quartile point selection
    
    # Safety (reuse existing guards)
    apply_guards: true          # Enable flatness/slope/chroma guards
    normalize_between_stages: true  # Match median of working image between poly and RBF stages
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
     b. Build the AutoBGE working image according to `stretch_mode`:
        - `none`: use the linear channel values directly.
        - `linear`: apply only an affine per-channel normalization for numerical
          conditioning; preserve additive relationships so the background model
          remains linearly interpretable. Compute scale/offset from valid,
          positive, finite canvas pixels only:
            p01 = valid 1st percentile, p99 = valid 99th percentile
            working = (channel - p01) / max(p99 - p01, eps)
          Do not clip working values before fitting; store p01 and scale for
          exact inverse mapping. If too few valid pixels exist or p99≈p01,
          fall back to `none` for that channel and record the fallback in
          diagnostics.
        - `mtf`: use the AutoBGE-style unlinked non-linear stretch per channel
          (target_median=0.25):
          ```
          shifted = channel - channel_min
          median_shifted = median(shifted)
          numerator   = (median_shifted - 1) * target_median * shifted
          denominator = median_shifted * (target_median + shifted - 1) - target_median * shifted
          stretched = numerator / (denominator + eps)   # eps = 1e-6 where denom==0
          ```
        - Record per-channel transform parameters for inverse mapping. For `none`
          this is identity; for `linear` this is affine; for `mtf` this is the MTF
          inverse.
     c. Downsample by downsample_scale using area interpolation
     d. Store working+downsampled image

Phase B: Stage 1 — Polynomial Gradient Removal
  2. Generate sample points on downsampled image:
     a. Add border points (corners + 5 per edge), with border_margin
     b. Divide into 4 quartiles
     c. Per quartile: exclude brightest `bright_exclusion_fraction` (default 0.5 = 50%), randomly select N/4 points
     d. For each point: gradient descent to dimmest local spot (patch median)
     e. Filter points against canvas_valid_mask and existing structure mask (downsampled)
     f. Deduplicate points (border corners overlap with edge linspace points)
  3. For each channel c:
     a. Estimate patch median (patch_size×patch_size) at each sample point
     b. Fit polynomial surface (degree=poly_degree) via least squares
     c. Upscale polynomial surface to full resolution (Lanczos4)
     d. Subtract: image_after_poly = working_image - poly_background
     e. Normalize: shift median back to median of working_image

Phase C: Stage 2 — RBF Gradient Removal
  4. Re-generate sample points on the *downsampled* `image_after_poly` (same procedure as step 2; coordinates remain in downsampled space)
  5. For each channel c:
     a. Estimate patch median at each sample point
     b. Fit RBF surface (multiquadric, smooth=rbf_smooth) 
        - Use Eigen for RBF interpolation (replacing scipy.interpolate.Rbf)
     c. Upscale RBF surface to full resolution (Lanczos4)
     d. Subtract: corrected = image_after_poly - rbf_background
     e. Normalize: shift median back to median of working_image

Phase D: Postprocessing
  6. Map the corrected working image back to the original linear pixel domain:
     - For each channel c:
       - `none`: corrected_linear = corrected
       - `linear`: apply the inverse affine transform
       - `mtf`: apply the inverse MTF transform:
       ```
       median_stretched = median(corrected[..., c])      # current median after BGE
       original_median  = stored medians[c]
       original_min     = stored mins[c]
       numerator   = (median_stretched - 1) * original_median * corrected[..., c]
       denominator = median_stretched * (original_median + corrected[..., c] - 1)
                       - original_median * corrected[..., c]
       unstretched = numerator / (denominator + eps) + original_min
       ```
  7. Derive a linear-domain background model from the corrected linear result:
       bg_model_linear = channel_before - corrected_linear
     **Pedestal is NOT baked into the model** — it is added at application time in Phase F,
     consistent with the classic pipeline convention. `model_stats.median = median(bg_model_linear)`
     over valid background pixels serves as the pedestal in Phase F.
     Deriving `bg_model_linear` this way is mandatory for `mtf`, because a non-linear
     transform does not preserve additive background differences; the model must always
     be computed in the linear domain.
  8. For each channel:
     a. Apply canvas_valid_mask
     b. Clamp model to [q05-guard_pad, q95+guard_pad] (reuse existing clamp)
  
Phase E: Guards (reuse existing)
  9. Flatness guard: compare spatial_background_spread before/after
  10. Slope guard: compare coarse_background_plane_slope before/after
  11. Chroma guard: compare log_chroma_std_background R/G and B/G
  12. If any guard fails:
      a. Revert to pre-BGE state (R/G/B = R_input/G_input/B_input)
      b. Log warning with guard reason
      c. Do not switch to the other BGE engine automatically — user's choice is explicit
      d. Classic internal fallback fits remain available only when method=classic;
         autobge failures revert to pre-BGE unless a future autobge-specific fallback
         is explicitly designed.
  
Phase F: Apply
  13. Atomic RGB apply using the linear-domain model from Phase D:
      corrected = channel_before - bg_model + pedestal
      where pedestal = model_stats.median (preserves background level)
```

**Important output contract:** AutoBGE must not be a separate final-apply path.
The AutoBGE implementation returns an `AutoBGEResult` containing one
linear-domain `BackgroundModel` per processed channel plus diagnostics. The outer
`apply_background_extraction()` then runs the shared guard, atomic RGB apply, and
diagnostic finalization. AutoBGE may return a clean failure result; the outer path
then reverts/no-ops exactly as it does for failed classic models. It must not both
unstretch a corrected image and then apply the same stretched-space model a
second time.

**`apply_guards` wiring:** `apply_guards` lives in `AutoBGEConfig` but the guard code
runs in the outer `apply_background_extraction()`, not inside the AutoBGE model
builder. Explicit wiring is required: `build_autobge_models()` returns an
`AutoBGEResult`, and the outer function checks `config.autobge.apply_guards`
before running the guard block. Without this, `apply_guards: false` is silently
ignored.

**AutoBGE diagnostics adapter:** AutoBGE must materialize its sample points and
per-stage model statistics into the existing diagnostic shape. The adapter should
populate at least `sample_bg_values`, `sample_weight_values` (use `1.0` for
unweighted AutoBGE samples), `grid_cells` or an equivalent sample-cell list,
`model_stats`, `residual_stats`, and per-channel guard fields. Guards that need
classic grid cells must either consume this adapter output or fall back to
image-wide metrics only.

### 5.4 Implementation Plan

#### Step 1: New C++ Functions in `background_extraction.cpp`

```cpp
// --- New functions for autobge method ---

/// Working-space transform for AutoBGE sampling/fitting.
struct StretchParams {
    std::vector<float> original_mins;   // per-channel min
    std::vector<float> original_medians; // per-channel median
    std::vector<float> linear_offsets;  // p01 per channel for linear mode
    std::vector<float> linear_scales;   // max(p99-p01, eps) per channel
    std::string mode;                   // none | linear | mtf
    bool was_single_channel;
};
Matrix2Df transform_to_autobge_working_space(
    const Matrix2Df& channel, const AutoBGEConfig& config, StretchParams* params);
Matrix2Df transform_from_autobge_working_space(
    const Matrix2Df& channel, const StretchParams& params);

/// Mono handling is selected by explicit upstream color mode/metadata, not by
/// testing whether RGB channels happen to be numerically identical.
enum class AutoBGEMonoMode { RGBDuplicate, Disabled };

struct AutoBGEResult {
    bool success = false;
    bool mono_input = false;
    std::array<BackgroundModel, 3> channel_models;
    std::vector<BGEChannelDiagnostics> channel_diagnostics;
};

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

/// Main AutoBGE model builder. Does not mutate/apply final RGB.
AutoBGEResult build_autobge_models(
    const Matrix2Df& R, const Matrix2Df& G, const Matrix2Df& B,
    const BGEConfig& config);
```

#### Step 2: RBF Implementation (Eigen)

The current C++ BGE already has RBF fitting with Eigen (`solve_rbf_model`). The key difference is:
- AutoBGE uses scipy's `Rbf(x, y, z, function='multiquadric', smooth=smooth, epsilon=1.0)` 
- scipy's `smooth` parameter adds `smooth * I` to the diagonal of the RBF matrix
- scipy's `epsilon=1.0` is fixed (our `mu` is configurable)

**Adaptation:** Reuse `solve_rbf_model`, but do not assume SciPy's parameters map
1:1 until this is verified. RBF behavior is sensitive to coordinate scale and the
regularization convention.

Initial implementation rules:
- Normalize sample coordinates to a stable range, preferably `[0,1]` in both axes,
  before solving. This keeps `epsilon`/`mu` independent of image size and
  downsample factor.
- `rbf_phi = "multiquadric"` (already supported)
- `rbf_mu_factor` / `epsilon` → start with an AutoBGE-compatible value on normalized
  coordinates, then validate against `AutoBGE.py` on a small reference image.
- `rbf_lambda` → start from `rbf_smooth`, but confirm sign and diagonal convention
  against the current Eigen solver before claiming SciPy parity.
- Skip IRLS (set `irls_max_iterations = 1`)
- **Important:** do not expose `epsilon` as a user-facing `autobge` parameter until
  the coordinate normalization and SciPy equivalence test are in place.

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
    if (config.method == "none") {
        // BGE disabled — no-op, return false (not applied)
        if (diagnostics) diagnostics->success = false;
        return false;
    }
    if (config.method == "autobge") {
        // autobge does not need tile_metrics/tile_grid for fitting, but still
        // needs the common canvas mask and diagnostic/guard adapter data.
        // It builds models only; shared guard + atomic apply happens below.
        AutoBGEResult auto_res = build_autobge_models(R, G, B, config);
        return finalize_bge_from_channel_models(
            R, G, B, auto_res.channel_models, auto_res.channel_diagnostics,
            config, diagnostics);
    }
    
    // method == "classic" → existing classic BGE code
    // ... existing classic BGE code ...
}
```

#### Step 5: Config Parsing (`config.cpp`)

Add parsing for `bge.method` and `bge.autobge.*`:

```cpp
// In read_yaml():
if (bge_node && bge_node["method"]) {
    config.bge.method = bge_node["method"].as<std::string>("none");
} else {
    // Legacy normalization: no method field present
    bool legacy_enabled = bge_node && bge_node["enabled"]
                          ? bge_node["enabled"].as<bool>(false) : false;
    config.bge.method = legacy_enabled ? "classic" : "none";
}
// Normalize enabled flag to match method (method is authoritative when present):
config.bge.enabled = (config.bge.method != "none");
// Edge case: if user writes method=autobge AND enabled=false, method wins -> enabled=true.
if (bge_node && bge_node["autobge"]) {
    const auto& ab = bge_node["autobge"];
    config.bge.autobge.num_sample_points = ab["num_sample_points"].as<int>(0);
    config.bge.autobge.poly_degree = ab["poly_degree"].as<int>(2);
    config.bge.autobge.rbf_smooth = ab["rbf_smooth"].as<float>(0.1f);
    config.bge.autobge.downsample_scale = ab["downsample_scale"].as<int>(4);
    config.bge.autobge.patch_size = ab["patch_size"].as<int>(15);
    config.bge.autobge.patch_estimator = ab["patch_estimator"].as<std::string>("median");
    config.bge.autobge.stretch_mode = ab["stretch_mode"].as<std::string>("linear");
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
    int num_sample_points = 0;  // 0 = automatic sizing, >0 = explicit override
    int poly_degree = 2;
    float rbf_smooth = 0.1f;
    int downsample_scale = 4;
    int patch_size = 15;
    std::string patch_estimator = "median";  // median | sigma_clipped_median
    std::string stretch_mode = "linear";      // none | linear | mtf
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
    std::string method = "none";     // NEW: none | classic | autobge (mutually exclusive)
    AutoBGEConfig autobge;            // NEW (used only when method == "autobge")
    // ... existing fields remain (used only when method == "classic") ...
    // ... including common_valid_mask in image/background_extraction.hpp ...
};

// Note: `common_valid_mask` is already present in image/background_extraction.hpp
// and is used for canvas exclusion. No change needed there.
```

#### Step 7: OpenCV Usage

The current C++ BGE does not use OpenCV directly (uses Eigen + custom box blur). AutoBGE.py uses OpenCV for:
- `cv2.resize` (area interpolation for downsampling, Lanczos4 for upscaling)
- `cv2.fillPoly` (exclusion mask — not needed in C++ pipeline)

**Adaptation:** Prefer existing project image utilities or a small local
implementation for area downsampling and model upscaling. Use OpenCV
(`cv::resize`) only if the relevant runner target already links OpenCV or the
additional dependency is accepted explicitly.

Before choosing OpenCV, verify that the target containing
`background_extraction.cpp` already links OpenCV in CMake. If not, prefer the
existing custom/image utility path unless adding OpenCV to the runner target is
acceptable for packaging and CI. The implementation plan must include the CMake
change or explicitly state that no new dependency is introduced.

#### Step 8: Tests

```cpp
// test_autobge.cpp
// 1. Synthetic gradient: create image with known polynomial gradient, verify removal
// 2. Two-stage: create image with poly + RBF gradient, verify both removed
// 3. Working-space roundtrip: verify none/linear/mtf transform+inverse
//    recover the original image within tolerance
// 4. Sample point generation: verify points avoid bright regions
// 5. Guard integration: verify guards trigger on over-correction
// 6. Canvas mask: verify invalid pixels are handled
// 7. Comparison: run autobge vs classic on test image, compare flatness
// 8. Reference parity: compare RBF/poly stage outputs against AutoBGE.py on a
//    tiny deterministic fixture within tolerance
// 9. Sparse sampling: verify graceful failure/retry when masks leave too few points
```

#### Step 9: Deterministic Sampling

AutoBGE.py uses `np.random.choice` without a fixed seed. For reproducibility in the C++ pipeline:
- Use `std::mt19937` with a deterministic seed (`config.bge.autobge.random_seed`)
- Config option: `autobge.random_seed` (default `42`)
- This ensures identical results across runs on the same input

Sample count semantics:
- `num_sample_points = 0` means automatic sizing:
  `num_points = max(100, downsampled_image_area / 10000)`.
- `num_sample_points > 0` is an explicit override.
- After canvas/foreground/bright-region filtering, each stage must require enough
  distinct points for the model: at minimum the polynomial term count plus margin
  for stage 1, and at least 16 points for RBF stage 2.
- If too few points remain, retry once with a relaxed
  `bright_exclusion_fraction` and without random down-selection from valid
  background candidates. If still insufficient, fail AutoBGE cleanly and revert
  to pre-BGE RGB.

#### Step 10: Validation Rules

Add to `tile_compile_cpp/src/io/config.cpp` validation:

```cpp
if (bge.method != "autobge" && bge.method != "classic" && bge.method != "none") {
  throw ValidationError("bge.method must be one of: none|classic|autobge");
}
// autobge.* validation only when method == autobge
if (bge.method == "autobge") {
  if (bge.autobge.num_sample_points != 0 && bge.autobge.num_sample_points < 10) {
    throw ValidationError("bge.autobge.num_sample_points must be 0 or >= 10");
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
  if (bge.autobge.stretch_mode != "none" &&
      bge.autobge.stretch_mode != "linear" &&
      bge.autobge.stretch_mode != "mtf") {
    throw ValidationError("bge.autobge.stretch_mode must be one of: none|linear|mtf");
  }
  if (bge.autobge.bright_exclusion_fraction <= 0.0f ||
      bge.autobge.bright_exclusion_fraction >= 1.0f) {
    throw ValidationError("bge.autobge.bright_exclusion_fraction must be in (0,1)");
  }
  if (bge.autobge.mono_mode != "rgb_duplicate" && bge.autobge.mono_mode != "disabled") {
    throw ValidationError("bge.autobge.mono_mode must be one of: rgb_duplicate|disabled");
  }
}
```

---

## 6. Run Monitor Integration

### 6.1 Phase Labeling

The Run Monitor phase list (`web_frontend_v3/js/components/phase-list.js`) currently shows a single `BGE` phase entry. With two methods, the phase label must reflect the active method.

**Display behavior:**

| `bge.method` | Phase label in Run Monitor | Phase enum (C++) |
|--------------|---------------------------|-------------------|
| `none` | **BGE (Skipped)** | `Phase::BGE` (unchanged, emitted as skipped) |
| `classic` | **BGE (Classic)** | `Phase::BGE` (unchanged) |
| `autobge` | **BGE (AutoBGE)** | `Phase::BGE` (unchanged) |

The C++ `Phase::BGE` enum stays the same — only the **display label** changes in the frontend.

### 6.2 Frontend Changes (`phase-list.js`)

```js
// In getPhasesForConfig(): keep the stable phase id as "BGE", but add a
// method-aware display label. Resume and status updates must use phaseId, not label.
function getBgeLabel(configDraft) {
  const bgeMethod = configDraft?.bge?.method || "none";
  if (bgeMethod === "none") return "BGE (Skipped)";
  if (bgeMethod === "classic") return "BGE (Classic)";
  return "BGE (AutoBGE)";
}

export function getPhasesForConfig(configDraft) {
  if (!configDraft || typeof configDraft !== "object") return DEFAULT_PHASES;
  const method = configDraft.method;
  const aqmhEnabled = configDraft.aqmh && configDraft.aqmh.enabled;
  const basePhases = (method === "classic_tile_compile" || aqmhEnabled === false)
    ? CLASSIC_PHASES : AQMH_PHASES;
  return basePhases.map(p => p === "BGE"
    ? { phase: "BGE", label: getBgeLabel(configDraft), bgeMethod: configDraft?.bge?.method || "none" }
    : { phase: p, label: p });
}
```

Implementation detail for `createPhaseItem()` / `setPhaseList()`:
- `getPhasesForConfig` now returns **objects for all phases**: `{ phase: p, label: p }` for standard
  phases and `{ phase: "BGE", label: "BGE (AutoBGE|Classic|Skipped)", bgeMethod: ... }` for BGE.
- `setPhaseList` already handles both strings and objects (`typeof phase === "string"`), but
  **`createPhaseList` does not** — it must be updated to call `createPhaseItem(phase.phase, ...)` and
  render `phase.label` as display text, not the raw object.
- Change `createPhaseItem` to accept a display label:
  `createPhaseItem(phaseId, state, pct = 0, label = phaseId)`.
- Change `setPhaseList` to call:
  `createPhaseItem(name, state, pct, phase.label || name)`.
- Change `createPhaseList` to normalize each item first:
  `const phaseId = typeof p === "string" ? p : p.phase; const label = typeof p === "string" ? p : (p.label || p.phase);`
- Change `updatePhaseState` to accept an optional label:
  `updatePhaseState(phaseName, status, pct, label)` and update the visible
  `.tc-phase-label` text when `label` is provided.
- Store the canonical id in `data-phase`, e.g. `data-phase="BGE"`.
- Render `phase.label` as visible text (not `phase.phase`).
- Keep `RESUMABLE_PHASES` as canonical ids only: `"BGE"`, not `"BGE (AutoBGE)"`.
- `updatePhaseState()` should match by canonical phase id from SSE (`phase: "BGE"`)
  and may update the visible label from `event.label` if present.

### 6.3 Backend Changes (`runner_pipeline.cpp`, `runner_resume.cpp`, `runner_preprocess.cpp`)

**All three runner files** emit BGE phase events and require the same label change:
- `tile_compile_cpp/apps/runner_pipeline.cpp` — main pipeline (Phase 11.5)
- `tile_compile_cpp/apps/runner_resume.cpp` — resume path (lines ~1780ff)
- `tile_compile_cpp/apps/runner_preprocess.cpp` — preprocessing path (lines ~1305ff)
- `tile_compile_cpp/src/core/events.cpp` / `events.hpp` — `EventEmitter` must accept
  optional phase metadata, or provide a BGE-specific overload/helper, so runners
  can emit `label` and `bge_method` without abusing the canonical phase name.

The `emitter.phase_start()` and `emitter.phase_end()` calls currently use the string `"BGE"`. Keep the canonical phase unchanged and add the method only as metadata/display label:

```cpp
// In runner_pipeline.cpp, BGE phase section:
const std::string bge_phase_label =
    (cfg.bge.method == "none")   ? "BGE (Skipped)" :
    (cfg.bge.method == "classic") ? "BGE (Classic)" :
                                     "BGE (AutoBGE)";

emitter.phase_start(run_id, Phase::BGE, "BGE", log_file,
                    {{"label", bge_phase_label},
                     {"bge_method", cfg.bge.method}});
// Event payload:
//   phase: "BGE"                    // stable id for state/resume
//   label: bge_phase_label          // display only
//   bge_method: cfg.bge.method      // none|classic|autobge
// phase_progress_counts and phase_end continue to use Phase::BGE / "BGE" as id.
```

If the existing `label` parameter is retained, rename it internally to
`display_label` or document that it never replaces the canonical `phase` field.

### 6.4 Phase Data in SSE Events

The SSE event `phase_start` sends a `label` field. The frontend `run-monitor.js` should use this label for display instead of hardcoding "BGE". This ensures the Run Monitor always shows the correct method even if the config was changed between runs.

### 6.5 Resume / History

- **Resume phase identity**: `RESUMABLE_PHASES` remains `{ "BGE", ... }`. Do not add `"BGE (Classic)"` or `"BGE (AutoBGE)"`; those are labels only.
- **Resume command**: selecting the BGE row always sends `from_phase=BGE` / `--from-phase BGE`.
- **Selected method on resume**: the BGE method comes from the resume configuration revision loaded in the Run Monitor editor. If that revision has `bge.method: classic`, resume from BGE reruns Classic BGE. If it has `bge.method: autobge`, it reruns AutoBGE. If it has `bge.method: none`, the BGE phase is emitted as skipped and the pipeline continues to PCC/HMS from the pre-BGE stack.
- **Revision safety**: when the user changes `bge.method` in the resume editor, the UI should show a small notice near the Resume button: "Resume from BGE will use method: Classic/AutoBGE/Disabled from the selected config revision." This avoids assuming the method from the old historical run.
- **Artifacts**: BGE resume should overwrite/regenerate method-specific BGE diagnostics for the resumed run. Diagnostics must include `bge_method` so history can show which engine produced the artifact.
- **Run History**: Store both `phase: "BGE"` and `label: "BGE (Classic|AutoBGE|Skipped)"` in phase results JSON. History display should show the stored label but route clicks/resume by `phase`.

### 6.6 Config UI (Parameter Studio)

The Parameter Studio should show a **dropdown** for `bge.method`:

| UI Element | Details |
|------------|---------|
| **Field** | `bge.method` |
| **Type** | Select dropdown |
| **Options** | `none` (label: "Disabled") / `classic` (label: "Classic (Grid-Based)") / `autobge` (label: "AutoBGE (Two-Stage)") |
| **Default** | `none` |
| **Behavior** | `none` → hide all BGE parameter sections; `autobge` → show `bge.autobge.*`, hide `bge.fit.*`; `classic` → show `bge.fit.*`, hide `bge.autobge.*` |

---

## 7. Migration Strategy

### Phase 1: Implementation (non-breaking, additive)
- Add `autobge` method alongside existing code
- Add `bge.method` config field with default `none`
- Add `bge.autobge.*` config section
- Add Run Monitor phase labeling (frontend + backend)
- Add Parameter Studio dropdown + conditional show/hide
- Both methods fully functional and selectable

### Phase 2: Testing
- Run A/B comparison on real datasets (M31, M42, NGC281)
- Compare: flatness, chroma std, visual inspection, star photometry
- Test `stretch_mode: none`, `linear`, and `mtf`; keep `linear` only if it improves
  robustness over `none`, and promote `mtf` only with clear evidence.
- Test `patch_estimator: median` vs `sigma_clipped_median` on star-rich fields
- Test mono images via `mono_mode: rgb_duplicate`
- Verify Run Monitor shows correct label for each method
- Verify resume works with both methods
- Tune defaults

### Phase 3: Documentation & Schema
- Update `configuration_reference.md` with `bge.method` and `bge.autobge.*` sections
- Update `gui3_user_guide_en.md` / `gui3_user_guide_de.md` with BGE method selection
- Update methodology document with AutoBGE description
- **Update the real schema source of truth**. In the current code this appears to
  be generated in `tile_compile_cpp/src/io/config.cpp`; if a checked-in
  `tile_compile.schema.json` exists, update it from that generator instead of
  hand-editing stale schema output. Add `bge.method` as enum
  `["none","classic","autobge"]` and the full `bge.autobge.*` object block.
- **Update `web_frontend_v3/i18n/en.json` and `de.json`** with BGE method display strings if labels are translated (e.g. `"bge.method.autobge"`, `"bge.method.classic"`, `"bge.method.none"`)
- **Update `web_frontend_v3/js/components/situation-assistant.js`**: the `gradient` profile currently suggests `bge.fit.method: rbf`. Add `bge.method: autobge` to the suggestion list, or replace `bge.fit.method` suggestions with the new engine selector
- Update `mkdocs.yml` if needed

### Phase 4: Long-term (optional)
- Both methods remain available indefinitely
- If `autobge` proves clearly superior in all test cases, consider a deliberate future default change from `none` to `autobge`
- If `classic` becomes unused, consider deprecation (but keep available for compatibility)

---

## 8. Risk Analysis

| Risk | Mitigation |
|------|-----------|
| AutoBGE's simple patch median is less robust than current estimators | Add `patch_estimator: sigma_clipped_median` option; default can stay `median` for exact compatibility |
| No exclusion zones in C++ pipeline (no user interaction) | Rely on gradient descent + bright exclusion + structure mask from existing code |
| Working-space transform could introduce numerical errors | Use float32 throughout, verify roundtrip for `linear` and `mtf` on test images |
| RBF with many sample points (100+) could be slow | Downsample 4× reduces grid size; RBF matrix is N×N where N=100, trivial for Eigen |
| Guards might reject autobge more often (different correction profile) | Tune guard thresholds for autobge; guard failure reverts to pre-BGE state (no auto-fallback) |
| Per-channel working transform could affect color balance | Inverse transform restores original scale; guards catch chroma issues |
| MTF stretch background is not additively invertible | Derive the linear-domain model from `input_linear - corrected_linear + pedestal`; never unstretch the stretched-space background difference directly |
| Existing BGE fallback code could silently switch engines | Gate classic fallbacks behind `method=classic`; AutoBGE guard failure reverts unless an AutoBGE-specific fallback is added |
| RBF parameter mapping differs from SciPy | Normalize coordinates and add a reference parity test against `AutoBGE.py` before claiming equivalent behavior |
| New OpenCV usage could affect packaging/CI | Prefer existing utilities; add OpenCV only if CMake target linkage and CI packaging are explicitly updated |

---

## 9. Expected Performance

| Metric | Current C++ BGE | AutoBGE (estimated) |
|--------|-----------------|---------------------|
| Sample generation | O(tiles × tile_pixels) full-res | O(num_points × patch_size² × descent_iters) on 4× downsampled |
| Polynomial fit | O(grid_cells × poly_terms²) | O(N × poly_terms²) where N=auto-sized (≥100) — similar |
| RBF fit | O(M² × irls_iters) where M=grid_cells | O(N² × 1) where N=auto-sized (≥100) — faster (no IRLS) |
| Surface rendering | Full-res polynomial/RBF evaluation | Lanczos4 upscale of downsampled surface |
| Autotune | O(candidates × full_pipeline) | N/A (disabled by default) |
| Total (estimated) | 5-30s with autotune | 1-5s without autotune |

---

## 10. Decisions

All open questions have been resolved based on the recommendations:

1. **Working-space stretch:** Implement `stretch_mode: none | linear | mtf`. Use
   `linear` as the initial AutoBGE default because the pipeline later applies
   HyperMetric Stretch and BGE should preserve a conservative linear correction
   contract. Treat `mtf` as an experimental AutoBGE-parity mode and compare all
   three modes before changing the default.

2. **Proportional sample point count:** `num_sample_points=0` means automatic sizing with `num_points = max(100, image_area / 10000)` where `image_area` is the downsampled image area in pixels. A positive config value overrides this.

3. **Exclusion mask:** Use the existing modeled foreground-mask logic as a mandatory exclusion mask for sample point generation, but compute it from the same working-space luminance image used by AutoBGE sampling, then downsample it with nearest/majority semantics. This prevents sampling in nebulae and galaxy halos where gradient descent alone is insufficient while keeping mask coordinates aligned with sample generation.

4. **RBF epsilon:** Normalize sample coordinates before solving and validate the Eigen RBF output against `AutoBGE.py`. Do not expose `epsilon` as a tunable parameter for `autobge` until this parity check defines a stable mapping.

5. **Guard thresholds:** Start with existing guard thresholds (flatness, slope, chroma). Adjust based on A/B testing if autobge is rejected more frequently than classic.

---

## 11. Summary

The AutoBGE algorithm offers a simpler, potentially more effective approach to background gradient extraction through its two-stage poly→RBF design and smart sample point placement. This proposal adds it as a **second method** alongside the existing classic BGE, giving users an explicit choice:

- **`bge.method: none`** — BGE disabled entirely (default; no gradient extraction)
- **`bge.method: classic`** — Grid-based single-stage, robust estimators, autotune (existing)
- **`bge.method: autobge`** — Two-stage poly→RBF, smart sampling, configurable working-space transform, downsampling (new)

Both active methods share the same safety guards, canvas mask awareness, and atomic RGB apply. The Run Monitor shows which method is active: **"BGE (Skipped)"**, **"BGE (Classic)"**, or **"BGE (AutoBGE)"**. The implementation is additive and non-breaking because the default remains disabled and both active methods require an explicit user choice.
