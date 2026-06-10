# AQMH — Step-by-Step Implementation Plan

**Version:** v0.1.0 (2026-06-05)  
**Target codebase:** `tile_compile_cpp` (C++20, OpenCV 4.11, Eigen 3.x)  
**Methodology reference:** `docs/AQMH/aqmh_methodik_en.md`  
**Prerequisite:** Familiarity with the runner infrastructure in `tile_compile_cpp/apps/runner_pipeline.cpp`

---

## Overview

The implementation is organized into **7 milestones**, each independently testable. Each milestone can be merged without breaking the existing pipeline (AQMH is always gated behind `aqmh.enabled = false`).

```
Milestone 1  Configuration types and YAML parsing
Milestone 2  Dense quality map computation (core algorithm)
Milestone 3  Quality map disk cache
Milestone 4  AQMH map-computation integration
Milestone 5  AQMH pixel-wise reconstruction
Milestone 6  Diagnostics and report integration
Milestone 7  Validation, tests, and documentation
```

### Non-Negotiable Resource Constraint

AQMH must be implemented as a **streaming, cache-backed method**. The pipeline must never keep all prewarped frames or all full-resolution quality maps resident in RAM. For large datasets, AQMH quality maps are potentially as large as the input frames themselves, so the implementation must treat them as disk-backed working data.

Binding implementation rules:

1. Every phase that touches large per-frame data must use a disk-backed provider/cache. No AQMH phase may require all frames or all maps in RAM.
2. Compute at most one frame's AQMH map per worker at a time.
3. Persist the map immediately to `QualityMapCache`.
4. Release the full-resolution working map before the worker advances to the next frame.
5. During reconstruction, load only the frame/map subset needed for the current reconstruction chunk/batch through bounded read caches.
6. The number of resident frames and resident maps must be bounded by configuration-derived memory budget, not by frame count.
7. Cache misses must be handled deterministically inside AQMH: affected samples receive zero AQMH weight, affected pixels become unsupported if no finite AQMH map sample remains, and the run emits a cache/map-availability warning. Cache misses must not trigger whole-run recomputation unless explicitly requested.

For a 24 Mpx image, one float32 quality map is about 96 MB at full resolution, about 24 MB at the default 1/4-area storage resolution, about 12 MB if stored as uint16 at 1/4-area resolution, and about 6 MB if stored as uint8 at 1/4-area resolution. With 300 frames, full-resolution float32 maps would be about 29 GB for one channel and about 86 GB for three channels. Therefore, full in-memory map retention is forbidden.

Per-stage caching requirements:

| Stage | Required cache behavior |
|---|---|
| Input/preprocessing | use existing frame stores; AQMH must not introduce a full-frame preload |
| AQMH map computation | read one source frame per worker, write `Q_map` immediately, release temporaries |
| AQMH reconstruction | stream source frames and maps via bounded providers; no eager full-run preload |
| Diagnostics/report | consume summary artifacts; raw map files remain grouped cache artifacts |

---

## Milestone 1 — Configuration Types and YAML Parsing

**Goal:** All AQMH configuration keys are parsed, validated, and accessible from `config::Config`.  
**Files modified:** 5 (`configuration.hpp`, `src/io/config.cpp`, schema YAML/JSON, default YAML)  
**Files created:** 0  
**Breakage risk:** None (new config subtree, disabled by default)

---

### Step 1.1 — Add `AqmhConfig` struct to configuration header

**File:** `tile_compile_cpp/include/tile_compile/config/configuration.hpp`

Add after the `LocalMetricsConfig` block in namespace `tile_compile::config`:

```cpp
struct AqmhPyramidConfig {
    int    scales              = 4;
    int    base_window_px      = 4;
    float  w_sharp             = 0.6f;
    float  w_snr               = 0.4f;
    float  k_artifact          = 5.0f;
    float  frac_artifact_max   = 0.10f;
};

struct AqmhStorageConfig {
    int         resolution_divisor = 2;   // linear divisor per axis: 1=full, 2=half width/height, 4=quarter width/height
    std::string dtype              = "float32"; // float32 | uint16 | uint8
    int         max_resident_maps  = 2;   // bounded read cache; 0 disables
};

struct AqmhDiagnosticsConfig {
    float tau_artifact       = 0.20f;
    float q_region           = 0.75f;
    int   r_morph_canvas_px  = 6;     // canvas-equivalent radius; map radius = round(radius / resolution_divisor)
};

struct AqmhCherryPickConfig {
    bool  enabled = false;
    int   k_min   = 3;
    float k_frac  = 0.30f;
};

struct AqmhConfig {
    bool                    enabled      = false;
    AqmhPyramidConfig       pyramid;
    AqmhStorageConfig       storage;
    AqmhCherryPickConfig    cherry_pick;
    AqmhDiagnosticsConfig   diagnostics;
};
```

Add `AqmhConfig aqmh;` as a member of `struct Config`.

The first implementation may keep `aqmh.enabled` as the config switch, but run status and artifacts must expose a derived method field:

```cpp
method = cfg.aqmh.enabled ? "aqmh" : "classic_tile_compile";
```

If a future top-level `method` config key is added, it must remain consistent with `aqmh.enabled`.

---

### Step 1.2 — YAML parsing and serialization

**File:** `tile_compile_cpp/src/io/config.cpp`

Add parsing and `to_yaml()` serialization for the new `aqmh:` subtree, mirroring the pattern used for `local_metrics`. All fields must have the defaults from Step 1.1 as fallback. If the `aqmh:` key is absent from the YAML, `AqmhConfig` keeps all defaults (so existing configs remain valid).

For region extraction, use `diagnostics.q_region` as the finite canvas-valid quality quantile threshold. If morphology runs on the full-resolution `Q_map`, use `diagnostics.r_morph_canvas_px` directly. Convert to working-map pixels with `r_morph_work_px = max(1, round(r_morph_canvas_px / storage.resolution_divisor))` only when morphology runs on a stored/downscaled map. This keeps the morphological footprint stable when `resolution_divisor` changes.

Also update:

- `tile_compile_cpp/tile_compile.schema.yaml`
- `tile_compile_cpp/tile_compile.schema.json`
- `tile_compile_cpp/tile_compile.yaml` defaults, with `aqmh.enabled: false`

---

### Step 1.3 — Validation

In the config validator, add:

- `aqmh.pyramid.scales` must be in `[1, 8]`.
- `aqmh.pyramid.base_window_px` must be `>= 1`.
- `aqmh.pyramid.w_sharp >= 0`, `aqmh.pyramid.w_snr >= 0`, and their sum must be `> 0`.
- `aqmh.storage.resolution_divisor` must be one of `{1, 2, 4}`.
- `aqmh.storage.dtype` must be one of `{"float32", "uint16", "uint8"}`. `uint16` is the recommended performance format when bit-identical float32 cache values are not required; it preserves exact zero/one and has a maximum quantization error of approximately `7.7e-6`.
- `aqmh.storage.max_resident_maps` must be in `[0, 16]`; `0` disables the reconstruction read-through LRU cache.
- `aqmh.cherry_pick.k_min >= 1`.
- `aqmh.cherry_pick.k_frac` must be in `(0, 1]`.
- `aqmh.cherry_pick.enabled = true` must emit a runtime `WARNING` log entry when the pipeline starts AQMH work.
- `aqmh.diagnostics.tau_artifact` must be in `[0, 1]`.
- `aqmh.diagnostics.q_region` must be in `[0, 1]`.
- `aqmh.diagnostics.r_morph_canvas_px` must be `>= 1`.

---

### Step 1.4 — Verification

Run the existing config unit tests. Confirm `AqmhConfig` is zero-initialized correctly and that loading any existing `.yaml` file without an `aqmh:` key still parses without error.

---

## Milestone 2 — Dense Quality Map Computation

**Goal:** A standalone, testable C++ module that takes a `Matrix2Df` frame and a canvas-valid mask and returns a `Matrix2Df` quality map `∈ [0,1]`.  
**Files created:** 3 new (`.hpp`, `.cpp` in `src/metrics/` and `include/tile_compile/metrics/`, plus a test source)  
**Files modified:** 0  
**Breakage risk:** None (no existing code modified)

---

### Step 2.1 — Create header

**File:** `tile_compile_cpp/include/tile_compile/metrics/aqmh_quality_map.hpp`

```cpp
#pragma once

#include "tile_compile/core/types.hpp"
#include "tile_compile/config/configuration.hpp"

#include <limits>
#include <vector>

namespace tile_compile::metrics {

inline constexpr float eps_aqmh = 1.0e-6f;

struct AqmhQualityMapDiagnostics {
    float sharpness_p50 = std::numeric_limits<float>::quiet_NaN();
    float snr_p50 = std::numeric_limits<float>::quiet_NaN();
    bool scene_dependent_snr = false;
    std::vector<int> omitted_scales;
};

struct AqmhQualityMapResult {
    Matrix2Df q_map;
    AqmhQualityMapDiagnostics diagnostics;
};

/// @brief Per-pixel quality map in [0,1] for a single normalized frame.
///
/// Input:
///   frame       - prewarped, normalized frame (float32, any size)
///   canvas_mask - 1=valid, 0=canvas-invalid (same spatial size as frame)
///   cfg         - AQMH pyramid configuration
///
/// Output:
///   Quality map at the same spatial resolution as the input frame plus
///   scalar diagnostics needed by aqmh_metrics.json.
///   canvas-invalid pixels are set to exactly 0.0f.
///   All other finite values are in [0, 1].
AqmhQualityMapResult compute_aqmh_quality_map(
    const Matrix2Df&            frame,
    const std::vector<uint8_t>& canvas_mask,
    int                         canvas_mask_width,
    int                         canvas_mask_height,
    const config::AqmhPyramidConfig& cfg);

} // namespace tile_compile::metrics
```

---

### Step 2.2 — Implement `compute_aqmh_quality_map`

**File:** `tile_compile_cpp/src/metrics/aqmh_quality_map.cpp`

Implement in this exact order:

#### Sub-step 2.2.1 — Canvas mask application helper

```cpp
static cv::Mat apply_canvas_mask(const cv::Mat& src,
                                  const std::vector<uint8_t>& mask,
                                  int mw, int mh);
```

Returns a copy of `src` with NaN at all canvas-invalid pixels. Uses the same `Rect`-clipped indexing as `apply_common_overlap_to_tile_inplace_and_check_nonzero` in `runner_phase_local_metrics.cpp`.

#### Sub-step 2.2.2 — Area-average downscale with mask-aware denominator

```cpp
static cv::Mat mask_aware_downsample(const cv::Mat& src,
                                      const cv::Mat& valid_mask_f,
                                      int D);
```

Downscale by integer factor `D` using `cv::resize` with `INTER_AREA`. For the mask-aware denominator: compute the mean only over finite pixels in each `D×D` block. If all pixels in a block are NaN, the output pixel is NaN.

Implementation note: the simplest correct approach is numerator/denominator filtering:
1. Build a numerator image `src_num` where invalid/NaN pixels are temporarily set to `0` only for the numerator sum, then compute `cv::resize(src_num, ..., INTER_AREA)`.
2. Build a `valid_mask_f` image with `1.0f` for finite pixels and `0.0f` otherwise, then compute `cv::resize(valid_mask_f, ..., INTER_AREA)` to get the valid fraction per downsampled pixel.
3. Divide the area-averaged numerator by `max(valid_fraction, eps_aqmh)`; set pixels where `valid_fraction <= 0` to NaN.

The temporary numerator zero is not a data sample. It is valid only because the denominator correction removes invalid pixels from the mean.

#### Sub-step 2.2.3 — Local Laplacian sharpness signal `Phi_sharp`

```cpp
static cv::Mat compute_phi_sharp(const cv::Mat& img_s, int R);
```

1. Compute `lap = masked_laplacian(img_s)`.
2. Compute local variance of `lap` in a `(2R+1)×(2R+1)` window using masked sums, not raw `cv::boxFilter` over NaNs:
   `local_var = E[x^2] - E[x]^2`.
3. Clamp to `[0, +inf)` and return as `Phi_sharp_s`.

Do not call `cv::Laplacian` directly on an image containing NaNs or canvas-filled zeros. OpenCV boundary handling and NaN propagation are not the AQMH support model. `masked_laplacian` must ignore invalid neighbors and return NaN when the center pixel is invalid or the finite stencil support is insufficient.

No global `sigma_lap` rescaling is applied (methodology §2.3.2(a)): the subsequent robust z-score in `robust_zscore_map` (Sub-step 2.2.6) is invariant to global scaling, so such a step would not change the result.

#### Sub-step 2.2.4 — Local SNR signal `Phi_snr`

```cpp
static cv::Mat compute_phi_snr(const cv::Mat& img_s,
                               int R,
                               bool* scene_dependent_snr);
```

1. Estimate the local background `b_s(x,y)` as the masked median of finite pixels in the same `(2R+1)×(2R+1)` window used by the other local statistics.
2. Compute `signal = max(img_s - b_s, 0)` when at least three finite pixels are available; otherwise fall back to `max(img_s, 0)` and set `*scene_dependent_snr = true` because `Phi_snr` is scene-dependent for at least one local support.
3. `mu = masked_mean(signal, R)`.
4. `sigma_map = 1.4826 * local_MAD(img_s, R)` using finite pixels only.
5. `phi_snr = mu / max(sigma_map, eps_aqmh)`.
6. Clamp to `[0, +inf)`.

#### Sub-step 2.2.5 — Artifact anomaly score `Phi_artifact`

```cpp
static cv::Mat compute_phi_artifact(const cv::Mat& img_s, int R,
                                     float k_artifact,
                                     float frac_artifact_max);
```

1. `blur = masked_box_filter(img_s, R)`.
2. `hp = img_s - blur`.
3. `abs_hp = cv::abs(hp)`.
4. Local robust scale of `hp`: `tau_map = max(1.4826 * local_MAD(hp, R), eps_aqmh)`. A mean-absolute-deviation approximation is acceptable for the first implementation only if documented in diagnostics as an approximation.
5. Outlier indicator: `outlier = (abs_hp > k_artifact * tau_map) ? 1.0f : 0.0f`.
6. `frac_out = masked_mean(outlier, R)` over `W_s_valid`, not divided by the full kernel area.
7. `phi_artifact = 1 - clip(frac_out / frac_artifact_max, 0, 1)`.

#### Sub-step 2.2.5a — Shared masked local-statistics helpers

Implement these helpers before the three signal functions:

```cpp
static cv::Mat finite_mask_f32(const cv::Mat& src);
static cv::Mat finite_mask_u8(const cv::Mat& src);
static cv::Mat masked_laplacian(const cv::Mat& src);
static cv::Mat masked_box_sum(const cv::Mat& src, const cv::Mat& valid, int R);
static cv::Mat masked_box_mean(const cv::Mat& src, const cv::Mat& valid, int R);
static cv::Mat masked_local_median(const cv::Mat& src,
                                   const cv::Mat& valid,
                                   int R);
static float finite_median(const cv::Mat& src);
static float finite_canvas_quantile(const Matrix2Df& src,
                                    const CanvasMask& common_valid_mask,
                                    float q);
static cv::Mat mask_aware_upsample(const cv::Mat& src,
                                   const cv::Mat& valid,
                                   int out_w,
                                   int out_h);
static cv::Mat local_mad_approx_or_exact(const cv::Mat& src,
                                         const cv::Mat& center,
                                         const cv::Mat& valid,
                                         int R);
```

All local-map helpers must preserve NaN for pixels where the valid-count denominator is zero. For fewer than three valid pixels, robust scale returns `eps_aqmh` and variance returns zero. `finite_median` is a scalar helper over all finite pixels in a matrix; return NaN when no finite pixels are available. `finite_canvas_quantile` collects only finite values whose canvas mask is valid and applies the deterministic quantile convention from the methodology. `robust_zscore_map` should reuse the same finite-pixel collection logic.

Canvas-invalid pixels must never be represented as numeric zero inside these helpers. Numeric zero is a valid sample value; invalid support is represented only by the finite/valid mask and NaN payloads.

`mask_aware_upsample` must upsample `src * valid` and `valid` separately with `cv::INTER_LINEAR`, then divide numerator by support where support is positive. Output pixels with zero interpolated support are NaN. Do not call `cv::resize(src, ...)` directly on `Psi_s`, because invalid scale samples would otherwise be treated as zeros or propagate NaNs into valid neighbors.

Use one deterministic order-statistics convention everywhere: sort finite values ascending; median is the center value for odd counts and the arithmetic mean of the two center values for even counts; MAD uses that same median convention; quantiles use linear interpolation at `q * (n - 1)`, clamped to the sample range.

#### Sub-step 2.2.6 — Per-scale robust z-score normalization

```cpp
static cv::Mat robust_zscore_map(const cv::Mat& src);
```

Compute `median` and `1.4826 * MAD` over all finite pixels in `src`. Return `(src - median) / max(1.4826*MAD, eps_aqmh)`. NaN pixels stay NaN.

#### Sub-step 2.2.7 — Per-scale sigmoid fusion `Psi_s`

```cpp
static cv::Mat compute_psi_s(const cv::Mat& phi_sharp,
                               const cv::Mat& phi_snr,
                               const cv::Mat& phi_artifact,
                               float w_sharp, float w_snr);
```

1. `z_sharp = robust_zscore_map(phi_sharp)`.
2. `z_snr   = robust_zscore_map(phi_snr)`.
3. `combined = w_sharp * z_sharp + w_snr * z_snr`.
4. `sigmoid_val = 1.0f / (1.0f + exp(-combined))` (element-wise via `cv::exp`).
5. `psi_s = sigmoid_val * phi_artifact`.
6. Clamp finite values to `[0, 1]`. Preserve NaN pixels as invalid support; do not convert them to zero here.

Only a finite numeric zero is an explicit AQMH veto. NaN means unavailable support and is handled by mask-aware upsampling/fusion.

#### Sub-step 2.2.8 — Multi-scale loop and geometric mean fusion

In `compute_aqmh_quality_map`:

```cpp
const int W = frame.cols(), H = frame.rows();
cv::Mat masked = apply_canvas_mask(frame_cv, canvas_mask, ...);
cv::Mat valid_mask_f = finite_mask_f32(masked); // finite pixel mask, float

std::vector<cv::Mat> psi_upscaled;
float sharpness_p50_diag = std::numeric_limits<float>::quiet_NaN();
float snr_p50_diag = std::numeric_limits<float>::quiet_NaN();
bool scene_dependent_snr = false;
std::vector<int> omitted_scales;
for (int s = 0; s < cfg.scales; ++s) {
    int D = 1 << (2 * s);  // D = 1, 4, 16, 64
    if (D > std::min(W, H) / 16) {
        omitted_scales.push_back(s);
        continue;  // skip too-small scales
    }

    cv::Mat img_s = mask_aware_downsample(masked, valid_mask_f, D);
    cv::Mat phi_sharp    = compute_phi_sharp(img_s, cfg.base_window_px);
    cv::Mat phi_snr      = compute_phi_snr(img_s, cfg.base_window_px,
                                           &scene_dependent_snr);
    cv::Mat phi_artifact = compute_phi_artifact(img_s, cfg.base_window_px,
                                                cfg.k_artifact,
                                                cfg.frac_artifact_max);
    if (s == 0)
        sharpness_p50_diag = finite_median(phi_sharp);
    if (s == 1)
        snr_p50_diag = finite_median(phi_snr);
    cv::Mat psi_s = compute_psi_s(phi_sharp, phi_snr, phi_artifact,
                                   cfg.w_sharp, cfg.w_snr);

    cv::Mat psi_valid = finite_mask_f32(psi_s);
    cv::Mat psi_up = mask_aware_upsample(psi_s, psi_valid, W, H);
    psi_upscaled.push_back(psi_up);
}

if (psi_upscaled.empty()) {
    cv::Mat q_map = cv::Mat::zeros(H, W, CV_32F);
    // apply canvas guard and return; this only happens for extremely small inputs
    AqmhQualityMapResult result;
    result.q_map = matrix_from_cv(q_map);
    result.diagnostics.sharpness_p50 = sharpness_p50_diag;
    result.diagnostics.snr_p50 = snr_p50_diag;
    result.diagnostics.scene_dependent_snr = scene_dependent_snr;
    result.diagnostics.omitted_scales = omitted_scales;
    return result;
}

// Geometric mean with exact zero-veto semantics
cv::Mat log_sum = cv::Mat::zeros(H, W, CV_32F);
cv::Mat zero_veto = cv::Mat::zeros(H, W, CV_8U);
for (const auto& p : psi_upscaled) {
    cv::Mat finite = finite_mask_u8(p);
    // OpenCV comparisons return a CV_8U mask with 255 for true pixels.
    // Bitwise OR and setTo(mask) intentionally use nonzero-as-true semantics.
    zero_veto |= (~finite) | (p <= 0.0f);
    cv::Mat log_p;
    cv::log(cv::max(p, eps_aqmh), log_p);
    log_p.setTo(0.0f, ~finite); // invalid samples are vetoed via zero_veto, not accumulated as NaN
    log_sum += log_p;
}
log_sum /= static_cast<float>(psi_upscaled.size());
cv::Mat q_map;
cv::exp(log_sum, q_map);
q_map.setTo(0.0f, zero_veto);

// Canvas guard: zero out invalid pixels
// (apply canvas mask: set invalid to 0)
for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
        if (!canvas_pixel_valid(x, y, canvas_mask, ...))
            q_map.at<float>(y,x) = 0.0f;

// Convert to Matrix2Df and return
AqmhQualityMapResult result;
result.q_map = matrix_from_cv(q_map);
result.diagnostics.sharpness_p50 = sharpness_p50_diag;
result.diagnostics.snr_p50 = snr_p50_diag;
result.diagnostics.scene_dependent_snr = scene_dependent_snr;
result.diagnostics.omitted_scales = omitted_scales;
return result;
```

---

### Step 2.3 — Unit test for `compute_aqmh_quality_map`

**File:** `tile_compile_cpp/tests/test_aqmh_quality_map.cpp` (create new)

Write the following tests (using the existing test framework — check `tile_compile_cpp/tests/` for the pattern):

1. **Synthetic flat frame:** Uniform frame → `Q_map` should be spatially near-uniform, all finite canvas-valid values `∈ [0,1]`.
2. **Injected hot pixel:** Frame with one bright outlier → pixels in the outlier's `R_s`-neighborhood at scale 0 should have `Q_map < Q_map` of the clean region at that scale.
3. **Canvas mask:** All-invalid canvas mask → entire `Q_map = 0`.
4. **Half-canvas mask:** Left half invalid → left half of `Q_map = 0`, right half `> 0`.
5. **Canvas non-infiltration:** Left half canvas-invalid contains extreme values (`1e9`), right half is constant valid data. The right-half `Q_map` must match the same run with the invalid half replaced by any other values, within float tolerance.
6. **Masked Laplacian boundary:** A valid region adjacent to canvas-invalid pixels must not show artificial sharpness solely because the invalid side contains NaN/zero/extreme values.
7. **Mask-aware upsample:** Invalid coarse-scale samples must not depress neighboring valid full-resolution `Psi_s^{up}` values; only zero finite samples may veto.
8. **Satellite stripe:** Frame with a bright horizontal stripe → `Q_map` reduced in stripe region (`artifact_frac` elevated).
9. **Determinism:** Two calls with identical inputs return bit-identical results.
10. **Zero-veto:** A scale map with exact finite zero produces exact zero in the fused map.
11. **Tiny valid window:** Fewer than three valid pixels never produce NaN/Inf.

---

## Milestone 3 — Quality Map Disk Cache

**Goal:** A `QualityMapCache` class that writes and reads `Q_map` arrays using the same DiskCache infrastructure as the existing frame store.  
**Files created:** 2  
**Files modified:** 0  
**Breakage risk:** None

---

### Step 3.1 — Create `QualityMapCache` header

**File:** `tile_compile_cpp/include/tile_compile/metrics/aqmh_quality_map_cache.hpp`

```cpp
#pragma once

#include "tile_compile/core/types.hpp"
#include "tile_compile/config/configuration.hpp"

#include <filesystem>
#include <cstddef>
#include <mutex>

namespace tile_compile::metrics {

/// Disk cache for per-frame AQMH quality maps.
/// Maps are stored at reduced resolution and upscaled on read.
class QualityMapCache {
public:
    QualityMapCache(const std::filesystem::path& cache_dir,
                    std::string map_stream_id,
                    int full_width, int full_height,
                    const config::AqmhStorageConfig& cfg);

    /// Write quality map for frame fi. Downscales to storage resolution.
    bool write(size_t fi, const Matrix2Df& q_map);

    /// Read quality map for frame fi. Upscales to full resolution.
    /// Returns empty matrix on cache miss or IO error.
    Matrix2Df read(size_t fi) const;

    /// Optional bounded read-through cache for reconstruction.
    /// Implementations must enforce max_resident_maps and evict LRU entries.
    Matrix2Df read_cached(size_t fi);

    /// Returns true if the cache entry for fi exists and is readable.
    bool has(size_t fi) const;

    /// Remove all cached files (called on run invalidation).
    void clear();

private:
    std::filesystem::path cache_dir_;
    std::string map_stream_id_;  // e.g. "luma", "R", "G", "B"; part of cache identity
    int full_w_, full_h_, stored_w_, stored_h_;
    config::AqmhStorageConfig cfg_;
    mutable std::mutex cache_mutex_;

    std::filesystem::path entry_path(size_t fi) const;
};

} // namespace tile_compile::metrics
```

---

### Step 3.2 — Implement `QualityMapCache`

**File:** `tile_compile_cpp/src/metrics/aqmh_quality_map_cache.cpp`

- Use `resolution_divisor` to compute `stored_w = ceil(full_w / D)`, `stored_h = ceil(full_h / D)` so odd image sizes round-trip to full dimensions.
- Storage format for `dtype = "float32"`: raw binary `.bin` file, header-free, row-major float32.
- `write`: clamp `q_map` to `[0,1]`, downsample to `stored_w × stored_h` using `cv::INTER_AREA`, write to `{cache_dir}/aqmh_{map_stream_id}_{fi:06d}.bin`.
- `read`: read binary, reshape to `stored_w × stored_h`, upsample to `full_w × full_h` using `cv::INTER_LINEAR`, clamp to `[0,1]`, return as `Matrix2Df`.
- `read_cached`: keep at most `cfg.max_resident_maps` full-resolution maps resident. Evict least-recently-used entries before inserting a new map. If `max_resident_maps = 0`, delegate directly to `read`.
- `uint16` support: quantize `[0,1]` → `[0,65535]` on write using nearest-integer rounding, dequantize on read with division by `65535.0`. This is not bit-identical to float32, but zero-veto (`0`) and full-quality (`1`) values remain exact.
- `uint8` support: quantize `[0,1]` → `[0,255]` on write, dequantize on read.
- `float16` support: not accepted until IEEE half conversion is implemented and tested. Do not silently write float32 when `dtype=float16`.
- Handle all file errors with a `false` return (never throw).
- Include a tiny sidecar metadata file (`aqmh_cache.json`) containing full size, stored size, dtype, resolution divisor, map stream id (`luma` for the first OSC/proxy implementation or channel id for per-channel maps), AQMH map format version, hash of the AQMH pyramid/storage config, and hash of the common-overlap mask. Do not include reconstruction-only settings or cherry-pick settings in the map-cache hash. `read` must reject entries whose map-affecting metadata do not match the current run.

---

### Step 3.3 — Unit tests for cache

**File:** `tile_compile_cpp/tests/test_aqmh_quality_map_cache.cpp` (create new)

1. **Write/read round-trip:** Write a synthetic map, read back, max absolute difference `< 1e-5` (float32 and uint16) or `< 0.005` (uint8).
2. **Missing file:** `read(999)` on empty cache returns empty `Matrix2Df`.
3. **Resolution round-trip:** Write at `resolution_divisor=2`, check that read-back map is `full_w × full_h`.
4. **Metadata mismatch:** Changing full dimensions, dtype, or divisor causes `read` to return an empty matrix rather than silently using stale data.
5. **Config/mask mismatch:** Changing pyramid config, storage config, map format version, or common-overlap mask invalidates old cache entries. Changing reconstruction-only or diagnostics-only settings does not.
6. **LRU bound:** Reading many maps through `read_cached` never leaves more than `max_resident_maps` maps resident.

---

## Milestone 4 — AQMH Map-Stage Integration

**Goal:** the runner computes and stores `Q_map` for each frame when `aqmh.enabled = true`. The first implementation may host this in `run_phase_local_metrics` for infrastructure reuse, but AQMH map computation remains a separate method stage.
**Files modified:** `runner_phase_local_metrics.cpp`, `runner_phase_local_metrics.hpp`  
**Files created:** 0  
**Breakage risk:** Low — all new code is gated behind `cfg.aqmh.enabled`

---

### Step 4.1 — Expose the AQMH quality-map cache to downstream stages

**File:** `tile_compile_cpp/apps/runner_phase_local_metrics.hpp`

Add output parameter:

```cpp
bool run_phase_local_metrics(
    ...,                                          // all existing parameters unchanged
    std::unique_ptr<metrics::QualityMapCache>& out_aqmh_cache  // new: nullptr if disabled
);
```

Prefer placing `QualityMapCache` in namespace `tile_compile::metrics` and forward-declaring it in the header to avoid pulling OpenCV/cache implementation details into the runner header.

Add `#include <memory>` to `runner_phase_local_metrics.hpp` if it is not already present.

---

### Step 4.2 — Add AQMH map computation block

**File:** `tile_compile_cpp/apps/runner_phase_local_metrics.cpp`

Inside the AQMH map computation block, after loading the prewarped frame for `fi`:

```cpp
if (cfg.aqmh.enabled && frame_has_data[fi] && aqmh_cache) {
    // Load full prewarped frame (single-channel luminance or per-channel)
    Matrix2Df full_frame;
    if (prewarped_frames.load_frame(fi, full_frame)) {
        metrics::AqmhQualityMapResult aqmh_result =
            metrics::compute_aqmh_quality_map(
                full_frame,
                common_valid_mask,
                common_mask_width,
                common_mask_height,
                cfg.aqmh.pyramid);
        aqmh_cache->write(fi, aqmh_result.q_map);
        aqmh_frame_diag[fi]["sharpness_p50"] = aqmh_result.diagnostics.sharpness_p50;
        aqmh_frame_diag[fi]["snr_p50"] = aqmh_result.diagnostics.snr_p50;
        aqmh_frame_diag[fi]["scene_dependent_snr"] =
            aqmh_result.diagnostics.scene_dependent_snr;
        aqmh_frame_diag[fi]["omitted_scales"] = aqmh_result.diagnostics.omitted_scales;
    }
}
```

The diagnostic medians `sharpness_p50` and `snr_p50` must be computed while the corresponding pre-z-score maps are still in local scope and returned through `AqmhQualityMapResult::diagnostics`. Do not retain full `phi_sharp` / `phi_snr` maps after the pyramid loop solely for diagnostics; store only the scalar medians in the per-frame diagnostic record. If scale 1 is omitted for a small image, `snr_p50` remains NaN and the artifact must record that the scale was unavailable.

Thread-safety note: `QualityMapCache::write` must be thread-safe. One file per frame index avoids write-write conflicts, but directory creation, metadata writes, and diagnostic vectors still need either pre-initialization or a mutex.

Channel note: for MONO this computes one map per frame. For OSC, the first implementation should compute one luminance/proxy map per frame and reuse it for RGB reconstruction after debayer/channel split; per-channel maps can be added later.

Also update stale comments in this file so they describe AQMH map computation as its own method stage.

---

### Step 4.3 — Initialize `QualityMapCache` before the worker loop

Before the parallel worker launch:

```cpp
std::unique_ptr<metrics::QualityMapCache> aqmh_cache;
if (cfg.aqmh.enabled) {
    aqmh_cache = std::make_unique<metrics::QualityMapCache>(
        run_dir / "cache" / "aqmh",
        "luma",  // first implementation uses one luminance/proxy stream per frame
        common_mask_width,
        common_mask_height,
        cfg.aqmh.storage);
}
```

---

### Step 4.4 — Update call site in `runner_pipeline.cpp`

Pass `aqmh_cache` through to the reconstruction phase (declare it at pipeline scope, pass by reference). If `cfg.aqmh.enabled = false`, the pointer remains null and all downstream AQMH code is skipped.

---

### Step 4.5 — Integration test

Add a mini-run integration test (using existing test infrastructure):

1. Run a 5-frame synthetic dataset with `aqmh.enabled = true`.
2. Verify that `aqmh_{map_stream_id}_{000000..000004}.bin` files exist in the cache directory, for example `aqmh_luma_000000.bin` in the first OSC/proxy implementation.
3. Read them back, verify values are in `[0, 1]`.
4. Run with `aqmh.enabled = false`, verify no cache files are written.

---

## Milestone 5 — AQMH Pixel-Wise Reconstruction

**Goal:** When `aqmh.enabled = true`, reconstruction uses AQMH per-pixel weights from `Q_map`. It must not use Classic Tile Compile local/tile weights as a mode, lower bound, or fallback.
**Files modified:** `runner_pipeline.cpp`, reconstruction headers as needed
**Files created:** `src/reconstruction/reconstruction_aqmh.cpp`
**Breakage risk:** Medium — adds an independent AQMH reconstruction path gated by `aqmh.enabled`

---

### Step 5.1 — New function signature for AQMH reconstruction

**File:** `tile_compile_cpp/include/tile_compile/reconstruction/reconstruction.hpp`

Add:

```cpp
namespace tile_compile::metrics {
class QualityMapCache;
}
namespace tile_compile::io {
class FrameProvider;
}

/// Pixel-wise AQMH reconstruction using per-frame dense quality maps.
/// frame_provider provides prewarped/normalized source frames on demand.
/// q_map_cache provides full-canvas AQMH maps on demand.
/// Missing or invalid map samples receive zero AQMH weight.
ReconstructTilesResult reconstruct_aqmh_weighted(
    io::FrameProvider&                     frame_provider,
    metrics::QualityMapCache*              q_map_cache,    // nullable AQMH maps
    const std::vector<float>&              global_weights, // G_{f,c}
    const CanvasMask&                      common_valid_mask,
    const ReconstructionConfig&            cfg);
```

`FrameProvider` may be an adapter around the existing prewarped frame store. It must expose bounded, cache-backed frame access and must not materialize all frames in memory.

---

### Step 5.2 — Implement `reconstruct_aqmh_weighted`

**File:** `tile_compile_cpp/src/reconstruction/reconstruction_aqmh.cpp` (new file)

Use tiles or strips only as work-partitioning chunks. The AQMH weight model itself is pixel-wise and must not consume Classic tile weights.

```
For each reconstruction chunk (parallel):
  For each pixel (x,y) in the chunk:
    if !common_valid_mask(x,y): R(x,y) = 0; continue
    V = { f | I_f(x,y) is finite AND canvas-valid }
    if V is empty: R(x,y) = 0; continue

    bool any_finite_map_sample = false
    For each f in V:
      q_map = q_map_for_frame_f_loaded_for_this_chunk_or_batch
      if q_map is full-size AND q_map(x,y) is finite:
        any_finite_map_sample = true
        w_f(x,y) = global_weights[f] * q_map(x,y)  // q_map may be 0: explicit veto
      else:
        w_f(x,y) = 0

    if sum(w_f) <= eps_weight:
      if any_finite_map_sample:
        R(x,y) = 0  // explicit AQMH zero-veto; do not unweighted-fallback
        continue
      else:
        R(x,y) = 0  // no AQMH map support; do not fall back to Classic weights
        mark pixel/run diagnostic as unsupported due to missing AQMH maps
        continue

    R(x,y) = weighted_sigma_clip_pixel(I_f(x,y), w_f, V, sigma_clip_cfg)
```

`weighted_sigma_clip_pixel` may reuse deterministic sigma-clipping helper logic from existing reconstruction code, but it must be parameterized as pixel-wise AQMH code and must not depend on Classic tile weights. Extract a helper if useful:

```cpp
static float sigma_clip_pixel(
    const std::vector<float>& values,
    const std::vector<float>& weights,
    const SigmaClipConfig& cfg);
```

The helper must preserve deterministic `min_fraction`, `N_eff` / `D_eff`, and `eps_weight` semantics after explicit AQMH zero-veto has been handled. Do not introduce a separate clipping policy for AQMH unless the methodology is updated.

Performance note: do not call frame or map cache reads inside the innermost pixel loop. At the start of a chunk or reconstruction batch, resolve the needed source-frame and quality-map references through bounded caches. The pixel loop may then perform direct lookups into those resident chunk-local references.

---

### Step 5.3 — Provide frames and Q-maps without unbounded preload

**File:** `tile_compile_cpp/apps/runner_pipeline.cpp`

Before AQMH reconstruction, do **not** eagerly load all frames or all maps. The required path is a bounded provider/cache path:

1. Pass a cache-backed frame provider for prewarped/normalized source frames.
2. Pass `QualityMapCache*` for AQMH maps.
3. Let reconstruction request the frame/map subset needed for the current chunk or batch.
4. Enforce resident frame and resident map limits independently.

Required pseudocode:

```cpp
recon_result = reconstruction::reconstruct_aqmh_weighted(
    prewarped_frame_provider, aqmh_cache.get(), global_weights,
    common_valid_mask, recon_cfg);
```

---

### Step 5.4 — Select the independent AQMH reconstruction path

```cpp
if (cfg.aqmh.enabled) {
    if (!aqmh_cache) {
        throw std::runtime_error("AQMH enabled but AQMH quality-map cache is unavailable");
    }
    recon_result = reconstruction::reconstruct_aqmh_weighted(
        prewarped_frame_provider, aqmh_cache.get(), global_weights,
        common_valid_mask, aqmh_recon_cfg);
}
```

Classic Tile Compile remains in its own runner branch. It is not an AQMH mode,
fallback, or lower-bound path.

---

### Step 5.5 — Reconstruction integration tests

1. **Synthetic artifact test:** Frame 0 has a bright stripe; `Q_map[0]` has zeros in stripe region. AQMH result must show reduced stripe contribution vs. equal-weight stack.
2. **Missing-map support test:** All cache reads miss or return invalid maps; AQMH output pixels become unsupported/zero and the run emits an AQMH warning. The implementation must not fall back to Classic tile weights.
3. **Canvas guard:** Canvas-invalid pixel in every frame -> output `R(x,y) = 0`.
4. **AQMH zero-veto:** If finite maps exist at a pixel and all AQMH weights are zero, the output stays unsupported/zero and must not fall through to an unweighted fallback.
5. **Finite-map weighting:** With finite map samples, per-frame weights equal `G_f * Q_map_f(x,y)` exactly within float tolerance.
6. **Memory bound:** Reconstruction with hundreds of frames must stay below the configured resident-map limit.

---

## Performance Expectations

AQMH computes dense per-frame maps plus per-pixel weight lookup during reconstruction. Classic Tile Compile can be run separately as a performance and quality baseline, but it is not an AQMH mode or fallback.

Expected cost relative to a separate Classic Tile Compile baseline:

| Stage | Classic baseline | AQMH path | Expected difference |
|---|---:|---:|---|
| Local quality analysis | Per frame × report block/tile | Per frame × pyramid scale × local windows | AQMH usually `2x-6x` slower for the quality-analysis stage |
| Reconstruction weighting | One scalar weight per frame/block | One map lookup and branch per frame/pixel | AQMH reconstruction usually `1.2x-2.5x` slower |
| Disk IO | Prewarped frames and artifacts | Plus AQMH map writes/reads | Can dominate if storage is slow |
| RAM pressure | Mostly frame/tile batches | Frame/tile batches plus bounded map cache | Must remain bounded by `max_resident_maps` |
| Total pipeline | Baseline | Baseline plus AQMH overhead | Typical end-to-end `1.3x-3x`; worst case `>4x` on slow disks or high worker counts |

For a rough 24 Mpx, 300-frame mono run:

- Classic baseline local metrics may be on the order of minutes, depending on block count and star detection cost.
- AQMH map generation adds several full-frame image-processing passes per frame and scale. With four scales, expect roughly `10-25` full-frame-equivalent passes per frame after downscaling is accounted for.
- Default 1/4-area float32 map storage writes about `24 MB * 300 = 7.2 GB` for mono, or about `21.6 GB` for three independent channel maps.
- If reconstruction reads every map repeatedly per spatial chunk, IO can become catastrophic. Therefore reconstruction must either use a bounded resident-map cache or schedule work frame-major/map-batch-aware enough to avoid rereading the same map for every chunk.

Performance acceptance targets for the first implementation:

1. AQMH disabled: no measurable slowdown beyond config parsing noise.
2. AQMH enabled, mono 24 Mpx / 300 frames: no OOM with `resolution_divisor=2`, `max_resident_maps=2`, and the configured reconstruction memory budget.
3. Map generation throughput should be reported as frames/minute and megapixels/second.
4. Reconstruction diagnostics should report AQMH cache hit rate, cache miss count, bytes read, bytes written, and max resident maps.
5. If AQMH total runtime exceeds `3x` a separate Classic baseline runtime on the same dataset, diagnostics must identify whether CPU map generation, reconstruction map reads, or disk IO is the bottleneck.

---

## Milestone 6 — Diagnostics and Report Integration

**Goal:** AQMH metrics appear in `aqmh_metrics.json` artifact and as new heatmap charts in the generated report.  
**Files modified:** `runner_phase_local_metrics.cpp`, `tile_compile_cpp/scripts/generate_report.py`  
**Files created:** 0

---

### Step 6.1 — Write `aqmh_metrics.json` artifact

**File:** `tile_compile_cpp/apps/runner_phase_local_metrics.cpp`

After the AQMH map computation for each frame, collect the scalar diagnostics defined in §6.1 of the methodology:

```cpp
core::json aqmh_frame_diag;
aqmh_frame_diag["fi"]           = static_cast<int>(fi);
aqmh_frame_diag["map_mean"]     = compute_finite_mean(aqmh_result.q_map);
aqmh_frame_diag["map_p10"]      = compute_percentile(aqmh_result.q_map, 10.0f);
aqmh_frame_diag["map_p90"]      = compute_percentile(aqmh_result.q_map, 90.0f);
aqmh_frame_diag["artifact_frac"]= compute_fraction_below(aqmh_result.q_map,
                                                         cfg.aqmh.diagnostics.tau_artifact);
aqmh_frame_diag["sharpness_p50"] = aqmh_result.diagnostics.sharpness_p50;
aqmh_frame_diag["snr_p50"]       = aqmh_result.diagnostics.snr_p50;
aqmh_frame_diag["scene_dependent_snr"] =
    aqmh_result.diagnostics.scene_dependent_snr;
aqmh_frame_diag["omitted_scales"] = aqmh_result.diagnostics.omitted_scales;
```

Collect report-block derived metrics (§6.2):

```cpp
for (size_t bi = 0; bi < report_blocks.size(); ++bi) {
    float aqmh_median = block_median_of_q_map(aqmh_result.q_map,
                                              report_blocks[bi]);
    aqmh_block_diag[bi]["aqmh_q_median"] = aqmh_median;
}
```

Extract quality regions from `aqmh_result.q_map` for diagnostics whenever `cfg.aqmh.enabled = true`, using methodology §5.2:

```cpp
float tau_region = finite_canvas_quantile(aqmh_result.q_map,
                                          common_valid_mask,
                                          cfg.aqmh.diagnostics.q_region);
int r_morph_px = cfg.aqmh.diagnostics.r_morph_canvas_px;
auto regions = extract_aqmh_quality_regions(aqmh_result.q_map,
                                            common_valid_mask,
                                            tau_region,
                                            r_morph_px);
aqmh_frame_diag["n_regions"] = regions.size();
aqmh_regions_artifact["frames"][fi]["regions"] = regions;
```

This code uses the full-resolution `aqmh_result.q_map`, so the morphology radius is the canvas-space radius directly. Divide by `storage.resolution_divisor` only if region extraction is deliberately run on a stored/downscaled map read from cache.

Write `aqmh_regions.json` alongside `aqmh_metrics.json`. If a future config disables region extraction, write `n_regions = 0` or omit the field consistently in both metrics and report generation.

Write to `run_dir / "artifacts" / "aqmh_metrics.json"` at the end of the phase.

The artifact should contain:

- `schema_version`
- `config` subset (`storage`, `pyramid`, `cherry_pick`, `diagnostics`)
- `frames[]`
- `report_blocks[]`
- block-indexed arrays sufficient for heatmap generation
- `cache_dir`
- `map_storage_resolution`
- `cache_stats` (`bytes_written`, `bytes_read`, `read_count`, `cache_hits`, `cache_misses`, `max_resident_maps_observed`)
- `timing` (`map_compute_s`, `map_write_s`, `map_read_s`, `aqmh_reconstruction_s`)

---

### Step 6.2 — Add cherry-pick warning to artifact

If `cfg.aqmh.cherry_pick.enabled`:

```cpp
artifact["cherry_pick_active"] = true;
log_file << "[AQMH] WARNING: cherry_pick mode enabled — pixel-level frame "
            "selection active, no-frame-selection invariant relaxed.\n";
```

---

### Step 6.3 — Report generator: new AQMH section

**File:** `tile_compile_cpp/scripts/generate_report.py`

Add a new `_gen_aqmh_metrics` function. It reads `aqmh_metrics.json` and generates:

1. **AQMH quality heatmap:** mean `aqmh_q_median` per report block (colormap `"viridis"`).
2. **Artifact fraction heatmap:** per-block `artifact_frac` (colormap `"inferno"`, reversed — high artifact = red).
3. **Optional comparison heatmap:** AQMH-vs-Classic deltas only when both methods were run separately and a comparison artifact exists.
4. **Per-frame timeseries:** `map_mean` and `artifact_frac` over frame index.

Integrate the section into the main report generation function as an AQMH section, not as part of local/tile metrics.

If `aqmh_metrics.json` is absent, report generation must silently skip the AQMH section so baseline runs remain unchanged.

---

### Step 6.3 AQMH-Native BGE Inputs

When `aqmh.enabled = true`, BGE must not depend on Classic `local_metrics.json`.
If BGE is enabled after AQMH reconstruction:

1. Use the reconstruction output canvas mask (`outputs/canvas_mask.fits`) as the BGE support domain.
2. Use the post-reconstruction tile grid only as a sampling partition.
3. Build BGE tile helpers directly from the reconstructed RGB/luma output:
   - per-tile background = finite canvas-valid luma median,
   - per-tile noise = MAD-based robust luma sigma,
   - per-tile structure = mean local luma gradient over canvas-valid pixels.
4. Mark the BGE artifact with `tile_metrics_source = "aqmh_output"`.
5. Preserve `have_local_metrics = false` so reports do not imply Classic local metrics were used.

For Classic runs, the existing `local_metrics.json` path remains unchanged and
must report `tile_metrics_source = "classic_local_metrics"`.


## Milestone 7 — Validation, Tests, and Documentation

**Goal:** All validation criteria from §9 of the methodology are verifiable by automated tests. Documentation is complete.

---

### Step 7.1 — Validation tests

Create `tile_compile_cpp/tests/test_aqmh_validation.cpp`:

| Test | Assertion |
|---|---|
| Map range | `all(Q_map ∈ [0,1])` for all finite canvas-valid pixels |
| Canvas guard | `all(Q_map == 0)` at canvas-invalid pixels |
| Canvas exclusion | Changing source values only in canvas-invalid pixels does not change any canvas-valid AQMH map value, diagnostic statistic, region, or reconstructed pixel |
| Determinism | Two calls return bit-identical results |
| Missing-map and zero-veto coverage | Missing/non-finite maps with finite intensities produce unsupported/zero AQMH output with a warning; finite all-zero maps produce unsupported/zero output without NaN/Inf |
| Block diagnostic consistency | `Q_{f,b}^{aqmh}` matches `median(Q_map over report block)` within `1e-5` |
| No structural injection | FWHM of point source does not increase vs. no-AQMH baseline |
| Artifact detection | Satellite frame has `artifact_frac > 0.01` in contaminated report blocks |
| Scale omission | Small input with `P_actual < P` records omitted scales, writes unavailable scale diagnostics as NaN/null, and fuses with denominator `P_actual` |
| Morph radius scaling | Changing `resolution_divisor` converts `r_morph_canvas_px` to the expected map-space radius |
| Cherry-pick flag | `cherry_pick_active=true` present in artifact JSON when enabled |
| Explicit zero-veto | Finite all-zero maps at a pixel produce unsupported/zero output, not unweighted mean |
| No Classic fallback | Missing/non-finite maps with finite intensities never fall back to Classic tile weights or unweighted mean |
| Cherry-pick ranking | Cherry-pick sorts by `G_f * Q_map_f(p)`, not raw frame-relative `Q_map` alone |

---

### Step 7.2 — Regression test against separate baseline runs

Add to the existing CI / mini-run test:

1. Run the same dataset with `aqmh.enabled = false` as a separate baseline run -> record FWHM, background RMS, and seam score.
2. Run with `aqmh.enabled = true`.
3. Assert: FWHM does not increase by more than 5%; background RMS does not increase; seam score does not increase.
4. (Aspirational, not blocking) Assert: FWHM decreases by at least 1% on a dataset with known intra-frame artifacts.

---

### Step 7.3 — CMake integration

**File:** `tile_compile_cpp/CMakeLists.txt`

Add the new source files to the library target:

```cmake
target_sources(tile_compile_lib PRIVATE
    src/metrics/aqmh_quality_map.cpp
    src/metrics/aqmh_quality_map_cache.cpp
    src/reconstruction/reconstruction_aqmh.cpp
)
```

In the current `CMakeLists.txt`, the library target is named `tile_compile_lib`; use that target name unless the build file is refactored.

Add AQMH test sources to the existing monolithic `tests` executable:

```cmake
add_executable(tests
    ...
    tests/test_aqmh_quality_map.cpp
    tests/test_aqmh_quality_map_cache.cpp
    tests/test_aqmh_validation.cpp
)
```

This repository currently uses one Catch2 test target named `tests`, so do not create separate test executables unless the test layout is refactored first.

---

### Step 7.4 — Documentation updates

1. **`README.md` / `README_de.md`:** Add a brief AQMH section under "Extensions" with a pointer to `docs/AQMH/`.
2. **`docs/AQMH/`:** The methodology and this plan are already the documentation.
3. **`tile_compile_cpp/examples/`:** Add `aqmh_enabled.example.yaml` with `aqmh.enabled: true` and all pyramid defaults filled in with inline comments.

---

## Dependency Graph

```
Milestone 1  (config)
    |
Milestone 2  (quality map algorithm)
    |
    +---> Milestone 3  (disk cache)
    |           |
    |     Milestone 4  (AQMH map integration)
    |           |
    |     Milestone 5  (AQMH reconstruction)
    |           |
    |     Milestone 6  (diagnostics & report)
    |           |
    +-----------+----> Milestone 7  (validation & tests)
```

Milestones 2 and 3 are independent and can be developed in parallel.  
Milestones 4 and 5 depend on both 2 and 3.  
Milestone 6 depends on 4 and 5.  
Milestone 7 depends on all previous milestones.

---

## Estimated File Inventory

| File | Action | Milestone |
|---|---|---|
| `include/tile_compile/config/configuration.hpp` | Modify (add structs) | 1 |
| `src/io/config.cpp` | Modify (add YAML parsing and serialization) | 1 |
| `tile_compile.schema.yaml` | Modify (schema) | 1 |
| `tile_compile.schema.json` | Modify (schema) | 1 |
| `tile_compile.yaml` | Modify (disabled default config subtree) | 1 |
| `include/tile_compile/metrics/aqmh_quality_map.hpp` | Create | 2 |
| `src/metrics/aqmh_quality_map.cpp` | Create | 2 |
| `tests/test_aqmh_quality_map.cpp` | Create | 2 |
| `include/tile_compile/metrics/aqmh_quality_map_cache.hpp` | Create | 3 |
| `src/metrics/aqmh_quality_map_cache.cpp` | Create | 3 |
| `tests/test_aqmh_quality_map_cache.cpp` | Create | 3 |
| `apps/runner_phase_local_metrics.hpp` | Modify (new output param) | 4 |
| `apps/runner_phase_local_metrics.cpp` | Modify (AQMH worker block) | 4 |
| `apps/runner_pipeline.cpp` | Modify (cache init, branch) | 4, 5 |
| `include/tile_compile/reconstruction/reconstruction.hpp` | Modify (new overload) | 5 |
| `src/reconstruction/reconstruction_aqmh.cpp` | Create | 5 |
| `tests/test_aqmh_validation.cpp` | Create | 7 |
| `tile_compile_cpp/examples/aqmh_enabled.example.yaml` | Create | 7 |
| `tile_compile_cpp/scripts/generate_report.py` | Modify (`_gen_aqmh_metrics`) | 6 |
| `tile_compile_cpp/CMakeLists.txt` | Modify | 7 |

**Total new files:** 9  
**Total modified files:** 11  
**No existing tests deleted or weakened.**
