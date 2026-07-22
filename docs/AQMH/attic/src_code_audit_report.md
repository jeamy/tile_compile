# Source Code Audit Report — `tile_compile_cpp/src/`

**Date:** 2026-07-21
**Scope:** 45 `.cpp` files, ~30 800 lines across 9 subdirectories

---

## 1. Duplicate Code

### 1.1 — `fit_elliptical_psf_2d` (HIGH)

**Files:**
- `@/media/data/programming/tile_compile/tile_compile_cpp/src/metrics/metrics.cpp:471-534`
- `@/media/data/programming/tile_compile/tile_compile_cpp/src/metrics/tile_metrics.cpp:26-87`

**Description:** The function `fit_elliptical_psf_2d(const cv::Mat& patch, float bg)` is **byte-for-byte identical** in both files. It computes weighted second central moments to fit an elliptical 2D Gaussian proxy. Both are `static` in anonymous namespaces, so each TU gets its own copy.

**Recommendation:** Move to a shared header/source in `metrics/` (e.g. `psf_fit.cpp`/`psf_fit.hpp`) and include from both TUs.

### 1.2 — Median/MAD/percentile implementations (MEDIUM)

At least **5 independent median implementations** exist:

| Location | Variant | Notes |
|---|---|---|
| `core/utils.cpp:361` | `median_of(std::vector<float>&)` | Canonical, modifies input, returns 0.0 on empty |
| `reconstruction/reconstruction.cpp:40` | `median_inplace(std::vector<float>&)` | Same algorithm, different name, returns 0.0 on empty |
| `metrics/aqmh_quality_map.cpp:90` | `median_of(std::vector<float>)` | By-value, returns NaN on empty |
| `reconstruction/aqmh_sigma_clip.cpp:58` | `median_select(std::vector<float>&)` | Uses `max_element` for even-N, not `nth_element` |
| `image/autobge.cpp:105` | `median_from_values(std::vector<float>)` | Full sort, not `nth_element`, returns `vals[N/2]` (not averaged for even N) |

Similarly, `robust_sigma_mad` is reimplemented in:
- `core/utils.cpp:404` (canonical, modifies input)
- `reconstruction/reconstruction.cpp:51` (`robust_sigma_mad_from_mat`, from `cv::Mat`)
- `metrics/metrics.cpp:99` (`masked_sigma_mad`, from `Matrix2Df` + mask)

And `percentile` functions:
- `core/utils.cpp:416` — `percentile_from_sorted` (canonical, 0-100 scale)
- `core/utils.cpp:428` — `percentile_of` (sorts then calls above)
- `reconstruction/reconstruction.cpp:69` — `percentile_from_mat` (from `cv::Mat`, 0-100 scale)
- `image/autobge.cpp:82` — `percentile_of_valid` (0-1 scale, no clamping)
- `metrics/aqmh_quality_map.cpp` — inline `median_of` variant

**Impact:** Inconsistent edge-case behavior: some return 0.0 on empty, others NaN. `median_from_values` in `autobge.cpp` does **not average the two middle elements for even N**, giving a biased median. The `1.4826f * mad` constant is duplicated 20+ times across files.

**Recommendation:** Consolidate to `core::median_of`, `core::mad_of`, `core::robust_sigma_mad`, `core::percentile_from_sorted` in `core/utils.cpp`. Add overloads for `cv::Mat` and `Matrix2Df` inputs. Fix `autobge.cpp`'s even-N median bug.

### 1.3 — CFA mosaic warp (MEDIUM)

**Files:**
- `core/acceleration.cpp:346-434` — `cuda_warp_cfa_mosaic`
- `core/acceleration.cpp:475-543` — `opencl_warp_cfa_mosaic`
- `image/cfa_processing.cpp:446` — `warp_cfa_mosaic_via_subplanes`

**Description:** All three implement the same sub-plane decomposition (split 2x2 Bayer pattern into 4 half-resolution planes, warp each, reassemble). The CUDA and OpenCL variants share ~80% identical logic (sub-plane extraction, `make_warp` lambda, reassembly loop). The CPU variant in `cfa_processing.cpp` is structurally similar but uses different warp offset conventions.

**Recommendation:** Extract shared sub-plane extraction/reassembly into a helper, leaving only the warp dispatch (CUDA/OpenCL/CPU) in the backend-specific code.

### 1.4 — OpenCL sigma-clip implementations (MEDIUM)

**File:** `core/acceleration.cpp:549-833` and `839-1055`

Two large OpenCL kernel functions (`opencl_sigma_clip_weighted_tile_impl` and `opencl_sigma_clip_stack_impl`) share ~70% structural similarity: the keep-mask update loop, min-keep computation, active_mask update, and final accumulation are nearly identical. The main difference is weighted vs. unweighted accumulation.

**Recommendation:** Factor out the shared clipping loop into a template/helper that accepts a weight policy.

---

## 2. Dead Code

### 2.1 — `glob()` wrapper (LOW)

**File:** `@/media/data/programming/tile_compile/tile_compile_cpp/src/core/utils.cpp:351-353`

```cpp
std::vector<fs::path> glob(const fs::path& dir, const std::string& pattern) {
    return discover_frames(dir, pattern);
}
```

This is a one-line forwarding wrapper to `discover_frames`. If no external consumer calls `glob()` directly, it is dead code.

### 2.2 — `estimate_background_sigma_clip` (LOW)

**File:** `@/media/data/programming/tile_compile/tile_compile_cpp/src/core/utils.cpp:437-456`

This function uses `stddev_of` (not MAD-based) for sigma clipping, which is inconsistent with the robust MAD-based approach used everywhere else in the codebase. It may be a legacy implementation superseded by the MAD-based functions. Verify whether any caller still uses it.

### 2.3 — `stddev_of` (LOW)

**File:** `@/media/data/programming/tile_compile/tile_compile_cpp/src/core/utils.cpp:386-398`

Standard deviation function. Only used by `estimate_background_sigma_clip` (above). If that function is dead, this one is too. The rest of the codebase uses `1.4826 * MAD` for robust sigma.

### 2.4 — `opencv_cuda_headers_available` unreachable case (LOW)

**File:** `@/media/data/programming/tile_compile/tile_compile_cpp/src/core/acceleration.cpp:58-75`

When `TILE_COMPILE_HAS_OPENCV_CUDA_HEADERS` is defined, the switch covers all `AccelerationPhase` values, then falls through to `(void)phase; return false;`. When the macro is undefined, the switch is skipped and the same `return false` executes. The `(void)phase` suppresses unused-variable warnings but is only reachable if the switch doesn't return — which can't happen if all enum cases are covered. This is not harmful but is dead code.

---

## 3. Logical Problems

### 3.1 — `read_bytes` negative size (MEDIUM)

**File:** `@/media/data/programming/tile_compile/tile_compile_cpp/src/core/utils.cpp:136-151`

```cpp
auto size = file.tellg();
file.seekg(0, std::ios::beg);
std::vector<uint8_t> buffer(size);
```

`tellg()` returns `-1` on failure (e.g. for certain special files or pipes). This would be cast to `size_t` and create a huge allocation, likely crashing. No guard checks `size > 0`.

**Recommendation:** Add `if (size <= 0) throw IOError("Empty or unreadable file: " + path.string());`

### 3.2 — `median_from_values` even-N bias (MEDIUM)

**File:** `@/media/data/programming/tile_compile/tile_compile_cpp/src/image/autobge.cpp:105-108`

```cpp
float median_from_values(std::vector<float> vals) {
  if (vals.empty()) return 0.0f;
  std::sort(vals.begin(), vals.end());
  return vals[vals.size() / 2];
}
```

For even N, this returns the upper-middle element instead of the average of the two middle elements. Every other median implementation in the codebase correctly averages. This biases background estimation in `autobge.cpp`.

**Recommendation:** Fix to average the two middle elements, or replace with `core::median_of`.

### 3.3 — `percentile_of_valid` no bounds clamping (LOW)

**File:** `@/media/data/programming/tile_compile/tile_compile_cpp/src/image/autobge.cpp:82-85`

```cpp
float percentile_of_valid(const std::vector<float>& sorted_vals, float pct) {
  if (sorted_vals.empty()) return 0.0f;
  const size_t idx = static_cast<size_t>(pct * (sorted_vals.size() - 1));
  return sorted_vals[idx];
}
```

No clamping of `pct` to `[0, 1]`. If `pct > 1.0f` is passed, `idx` exceeds the array bounds. Currently only called with 0.01, 0.50, 0.99, so safe in practice, but fragile.

### 3.4 — OMP critical in hot loop (MEDIUM)

**File:** `@/media/data/programming/tile_compile/tile_compile_cpp/src/reconstruction/aqmh_reconstruction.cpp:383-388`

```cpp
#pragma omp critical
{
    effective_k.push_back(static_cast<float>(selected.size()));
    if (margin >= 0.0f) margins.push_back(margin);
}
```

This `critical` section is inside the per-pixel OMP loop. While cherry-pick is disabled by default, when enabled it serializes all threads on every cherry-picked pixel. A lock-free approach (e.g. per-thread buffers merged after the loop) would be better.

### 3.5 — `keep_count` using float equality (LOW)

**File:** `@/media/data/programming/tile_compile/tile_compile_cpp/src/reconstruction/aqmh_sigma_clip.cpp:144`

```cpp
for (const auto &s : samples) keep_count += s.value == center;
```

Exact float equality comparison. This is intentional (keeping only samples exactly equal to the median when MAD is degenerate), but float equality is fragile. If the median came from `weighted_median_select` which returns a pivot value from the data, this is safe. But if it ever changes to return an interpolated value, this breaks silently.

### 3.6 — `source_masked_frame` mask shape validation (LOW)

**File:** `@/media/data/programming/tile_compile/tile_compile_cpp/src/metrics/aqmh_quality_map.cpp:150-174`

When `mask_shape_valid` or `frame_mask_shape_valid` is false, the function silently writes NaN for all pixels (because `valid` is false). No error or warning is logged. This could mask configuration errors where the mask dimensions don't match the frame.

### 3.7 — `sanitize_yaml_windows_paths` escape table (LOW)

**File:** `@/media/data/programming/tile_compile/tile_compile_cpp/src/io/config.cpp:37`

```cpp
static const char valid[] = "\"\\0abtnvfrNLP_e \t/x u U";
```

This string contains space, tab, `/`, `x`, `u`, `U` as individual characters. The intent is to list valid YAML escape characters, but including bare space and tab means a backslash followed by whitespace is treated as a valid escape (preserving the backslash), when it should probably be treated as a Windows path separator. Edge case, unlikely in practice.

---

## 4. Style / Maintainability Issues

### 4.1 — Verbose `@brief`/`@details` boilerplate (LOW)

Nearly every function has a two-line `@brief`/`@details` comment that follows the pattern "Part of X helpers; this helper keeps the implementation localized in this translation unit and preserves the surrounding phase, artifact, and error-handling semantics expected by callers." This adds no information and inflates file sizes by ~15-20%. Consider removing or shortening.

### 4.2 — `1.4826f` magic constant (LOW)

The MAD-to-sigma conversion factor `1.4826f` appears 20+ times across 8 files. It should be a named `constexpr float kMadToSigma = 1.4826f;` in a shared header.

### 4.3 — Inconsistent NaN-vs-zero on empty input (LOW)

Some functions return `0.0f` on empty input (`core::median_of`, `reconstruction::median_inplace`), others return `NaN` (`aqmh_quality_map::median_of`, `aqmh_quality_map::mad_of`). This inconsistency can propagate through the pipeline: a `0.0f` from an empty median becomes a valid-looking value, while `NaN` correctly signals "no data."

**Recommendation:** Standardize on NaN for empty-input cases in statistical functions, or document the convention per function.

---

## 5. Summary by Severity

| Severity | Count | Items |
|---|---|---|
| **HIGH** | 1 | 1.1 (duplicate `fit_elliptical_psf_2d`) |
| **MEDIUM** | 6 | 1.2 (median/MAD proliferation), 1.3 (CFA warp), 1.4 (OpenCL sigma-clip), 3.1 (`read_bytes`), 3.2 (even-N median bias), 3.4 (OMP critical) |
| **LOW** | 8 | 2.1-2.4 (dead code), 3.3, 3.5-3.7 (logical), 4.1-4.3 (style) |

## 6. Recommended Action Priority

1. **Fix 3.2** (even-N median bias in `autobge.cpp`) — potential correctness bug
2. **Fix 3.1** (`read_bytes` negative size) — potential crash
3. **Fix 1.1** (extract shared `fit_elliptical_psf_2d`) — eliminates copy-paste drift risk
4. **Consolidate 1.2** (median/MAD/percentile) — largest maintainability win
5. **Address 3.4** (OMP critical) — performance, only if cherry-pick is used on large images
6. Remaining items as time permits
