# AQMH v0.2.0 — Implementation Audit & Performance Bottleneck Report

**Date:** 2026-07-06  
**Auditor:** Cascade  
**Scope:** Full audit of `aqmh_v0.2.0_implementation_plan.md` and `aqmh_v0.2.0_optimization_plan.md` implementation status, with focus on performance bottlenecks in AQMH* phases and GPU integration.

---

## 1. Executive Summary

The implementation plan was **largely implemented** (config structs, pipeline disentanglement, cache routing fix, prefetch coordinator, CUDA/OpenCL kernels, binary diagnostics). However, there are **critical performance bottlenecks** in the GPU paths and **one safety-gate violation** that together explain why AQMH phases remain extremely slow.

### Severity Ranking

| # | Issue | Severity | Impact |
|---|-------|----------|--------|
| 1 | GPU path loads full frames instead of regions | **Critical** | ~74× wasted I/O per chunk |
| 2 | GPU path loads full Q-maps instead of regions | **Critical** | Same I/O amplification |
| 3 | `gpu_reconstruction` default is `"auto"` not `"disabled"` | **Critical** | Activates broken GPU path by default |
| 4 | CUDA kernel local memory spill (~16KB/thread) | **High** | Occupancy 1-2 threads/SM |
| 5 | Bitonic sort always processes 1024 elements | **High** | ~500K comparisons/pixel regardless of frame count |
| 6 | `invalidate_mapping` after each load forces re-mmap | **High** | ~48K mmap syscalls per run |
| 7 | Cherry-pick gate doubles I/O before reconstruction | **Medium** | Full O(N×W×H) pre-pass |
| 8 | No async I/O overlap in GPU host code | **Medium** | Fully synchronous upload/download |
| 9 | Legacy `AccelerationOps::reconstruct_aqmh` forces CPU fallback | **Low** | Dead code for AQMH pipeline |

---

## 2. Plan Compliance Audit

### 2.1 Implementation Plan (`aqmh_v0.2.0_implementation_plan.md`)

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0: Pipeline Disentanglement | ✅ Implemented | `runner_aqmh_pipeline.cpp`, separate phase files |
| Phase 1: Config & Numerical Infrastructure | ✅ Implemented | All config structs, `aqmh_eps.cpp` |
| Phase 2: Quality Map Computation | ✅ Implemented | Source-valid mask, `phi_snr`, `phi_artifact` |
| Phase 3: Global Quality Factor G | ✅ Implemented | `aqmh_global_quality.cpp`, `AQMH_GLOBAL_QUALITY` phase |
| Phase 4: Storage & Cache | ✅ Implemented | Format version 2, mask-aware downsampling |
| Phase 5: Reconstruction | ✅ Implemented | `aqmh_sigma_clip.cpp`, weighted median+MAD |
| Phase 6: Cherry-Pick v0.2.0 | ✅ Implemented | `aqmh_cherry_pick.cpp`, tiered k_frac, gate |
| Phase 7: Diagnostics | ✅ Implemented | 11 fields/frame, block-level, heatmaps |
| Phase 8: Region Extraction | ✅ Implemented | Threshold, morphology, connected components |
| Phase 9: Validation | ✅ Implemented | `aqmh_validation.cpp`, control run |
| Phase 10: Integration | ✅ Implemented | CMakeLists, pipeline orchestrator |

### 2.2 Optimization Plan (`aqmh_v0.2.0_optimization_plan.md`)

| Item | Status | Notes |
|------|--------|-------|
| 2.1: Diagnostics config flags | ✅ Implemented | `enabled`, `level`, `per_frame_blocks`, `heatmaps`, `regions`, `format` |
| 2.1.2: Reconstruction chunk config | ✅ Implemented | `chunk_rows`, `memory_budget_mb`, `gpu_reconstruction` |
| 2.1.5: `chunk_rows` in reconstruction | ✅ Implemented | Auto-sizing formula matches plan |
| 3.0: AQMH_MAPS profiling | ⚠️ Not profiled | No profiling results documented |
| 3.1: Prefetch coordinator | ✅ Implemented | `AqmhPrefetchCoordinator`, Option C |
| 3.2: Cache routing fix | ✅ Implemented | `read_region()` routes through LRU |
| 3.2: `max_resident_maps` default | ✅ Implemented | Changed from 2 to 4 |
| 4.1: CUDA backend | ✅ Implemented | Kernel, row-chunking, bitonic sort |
| 4.1.3: Backend arbitration | ✅ Implemented | `opencv_cuda` excluded, `cuda > opencv_opencl > cpu` |
| 4.1.4: `gpu_reconstruction` gate | ⚠️ **Violated** | Default is `"auto"`, plan requires `"disabled"` |
| 4.2: OpenCL backend | ✅ Implemented | Mirrors CUDA kernel |
| 4.3: GPU row-chunk sizing | ✅ Implemented | Device-memory-based sizing |
| 4.4: Binary diagnostics | ✅ Implemented | `aqmh_binary_diagnostics.cpp/.hpp` |

---

## 3. Critical Performance Bottlenecks

### 3.1 GPU Path Loads Full Frames Instead of Regions (CRITICAL)

**Location:** `src/reconstruction/aqmh_reconstruction_cuda.cu:768-769`, `src/reconstruction/aqmh_reconstruction_opencl.cpp:677-679`

The CUDA and OpenCL paths load entire frames via `load_frame(fi, full_frame)` for each row-chunk, despite only needing `chunk_rows` rows out of `height` total. The CPU path correctly uses `load_frame_region(fi, y0, rows, frame)` which calls `extract_tile_into()` and only copies the needed rows from the mmap'd file.

**Impact:** For a 2310×3924 canvas with 645 frames and `chunk_rows=53`:
- Each chunk loads 645 full frames = 645 × 2310 × 3924 × 4 bytes ≈ **23 GB**
- Only 53/3924 ≈ 1.35% is used → **98.65% wasted I/O**
- With ~74 chunks: total I/O = **~1.7 TB** instead of ~23 GB

**Fix:** Pass `AqmhFrameRegionLoader` and `AqmhMaskRegionLoader` to the GPU functions and use them instead of full-frame loaders. The GPU function signatures already accept `AqmhFrameLoader` — extend them to accept region loaders, or wrap the region loader into the existing interface.

### 3.2 GPU Path Loads Full Q-Maps Instead of Regions (CRITICAL)

**Location:** `src/reconstruction/aqmh_reconstruction_cuda.cu:776`, `src/reconstruction/aqmh_reconstruction_opencl.cpp:684`

Both GPU paths call `q_map_cache->read_cached(fi)` which loads the **entire** Q-map for every frame, for every chunk. The CPU path uses `q_map_cache->read_region(fi, y0, rows)`.

**Impact:** Same I/O amplification as §3.1 — full Q-maps are loaded 74× instead of once.

**Fix:** Use `read_region(fi, y0, rows)` in the GPU host-side staging loop. The `QualityMapCache::read_region()` now routes through the LRU cache (§3.2 fix is implemented), so this is a drop-in change.

### 3.3 `gpu_reconstruction` Default Violates Safety Gate (CRITICAL)

**Location:** `include/tile_compile/config/configuration.hpp:267`

```cpp
std::string gpu_reconstruction = "auto";  // "disabled" | "auto" | "force"
```

The optimization plan (§4.1.4) explicitly requires:
> defaulting to `disabled` until determinism is validated

With `"auto"` as default, the GPU path is activated on any CUDA-capable machine. Given the I/O issues in §3.1–3.2, this makes GPU reconstruction **slower than CPU** while also risking numerical divergence that hasn't been validated.

**Fix:** Change default to `"disabled"`:
```cpp
std::string gpu_reconstruction = "disabled";
```

### 3.4 CUDA Kernel Local Memory Spill (HIGH)

**Location:** `src/reconstruction/aqmh_reconstruction_cuda.cu:431-433, 244-246, 479`

Each thread declares:
- `float values[1024]` = 4 KB
- `float weights[1024]` = 4 KB
- `float scores[1024]` = 4 KB
- `int sort_indices[1024]` = 4 KB (inside `sigma_clip`)
- `float deviations[1024]` = 4 KB (inside `sigma_clip`)
- `float sorted_values[1024]` = 4 KB (inside `sigma_clip`)
- Plus `control_values[1024]`, `control_weights[1024]` if uniform control

Total: **~24-32 KB per thread** in local memory. With 16×16 = 256 threads/block, that's **~6-8 MB/block**. CUDA local memory spills to L2 cache-backed global memory, limiting occupancy to **1-2 warps per SM**.

**Impact:** GPU utilization is extremely low. The kernel is functionally correct but performance is likely worse than CPU for typical frame counts.

**Fix:** The plan's Option A (streaming/online weighted statistics) is the correct long-term fix. Short-term: reduce `kMaxFramesCompile` to a realistic maximum (e.g., 768) and use smaller block sizes (8×8 = 64 threads) to reduce per-block local memory pressure.

### 3.5 Bitonic Sort Always Processes 1024 Elements (HIGH)

**Location:** `src/reconstruction/aqmh_reconstruction_cuda.cu:85-108`

The `bitonic_sort_indices` function always iterates over `kMaxFramesCompile = 1024` elements, even when only `n` < 1024 samples are valid. The outer loops are:
```cuda
for (int k = 2; k <= kMaxFramesCompile; k *= 2)      // 10 iterations
  for (int j = k / 2; j > 0; j /= 2)                  // log2(k) iterations
    for (int i = 0; i < kMaxFramesCompile; ++i)        // 1024 iterations
```

Total: ~10 × 10 × 1024 / 2 ≈ **50K comparisons per sort**.

Per pixel, sorts are called: cherry-pick (1×), weighted median (1×), weighted MAD (1×), noise floor (2×), keep-floor (1-2×) = **~6 sorts per sigma-clip iteration**, × 3 iterations = **~18 sorts/pixel** = ~900K comparisons/pixel.

For a 2310×3924 canvas (~9M pixels): **~8 trillion comparisons**.

**Impact:** The bitonic sort is O(N log²N) with N=1024 regardless of actual sample count. For typical 50-200 valid samples, an insertion sort (O(N²)) would be faster: 200² = 40K vs 50K per sort.

**Fix:** Use adaptive sort size: round up `n` to the next power of 2, cap at 1024. For `n ≤ 64`, use insertion sort. For `n > 64`, use bitonic sort on `next_pow2(n)` elements only.

### 3.6 `invalidate_mapping` Forces Re-mmap on Every Chunk (HIGH)

**Location:** `apps/runner_phase_aqmh_reconstruction.cpp:93`

The `aqmh_frame_loader` lambda calls `prewarped_frames.invalidate_mapping(fi)` after each frame load. This unmaps the frame's mmap view, forcing a re-mmap on the next access.

In the CPU path with region streaming, each frame is accessed once per chunk (e.g., 74 chunks for 3924/53 rows). The `extract_tile_into` method calls `mapped_frame_ptr` which re-mmaps if the view is null. This results in:
- 645 frames × 74 chunks = **~48K mmap syscalls** per run
- Each mmap involves `open()`, `mmap()`, `close()` = 3 syscalls

**Impact:** ~144K syscalls for frame loading that could be zero (keep mmap views resident).

**Fix:** Remove `invalidate_mapping(fi)` from the `aqmh_frame_loader` lambda. The mmap views are read-only and don't need invalidation during reconstruction. Call `clear_mappings()` once after the entire reconstruction phase if memory pressure is a concern.

### 3.7 Cherry-Pick Gate Doubles I/O (MEDIUM)

**Location:** `src/reconstruction/aqmh_reconstruction.cpp:78-143`

Before the main reconstruction loop, the cherry-pick gate scans all frames and all pixels to count rankable samples and compute `k_nominal_median`. This is a full O(frame_count × W × H) pass that reads all Q-maps and frame masks.

**Impact:** Doubles the Q-map I/O: once for the gate, once for reconstruction. For 645 frames and 2310×3924 pixels, this is ~645 × 9M = ~5.8B operations plus 645 Q-map region reads.

**Mitigation:** The gate uses 128-row slabs with `read_region` (line 82-108), which is already efficient. The I/O cost is inherent to the gate decision. Consider caching the rankable count from the AQMH_MAPS phase to avoid re-reading.

### 3.8 No Async I/O Overlap in GPU Host Code (MEDIUM)

**Location:** `src/reconstruction/aqmh_reconstruction_cuda.cu:744-869`

Despite creating a CUDA stream (line 742), the host-side code is fully synchronous:
1. Load all frames for chunk (CPU, serial) → upload (async) → launch kernel (async) → download (async) → `cudaStreamSynchronize` (blocking)
2. No overlap between chunk N's download and chunk N+1's upload/compute

**Fix:** Double-buffer: while the GPU processes chunk N, load and upload chunk N+1's data. Use two sets of device buffers and alternate.

---

## 4. GPU Integration Audit

### 4.1 Backend Arbitration

**Status:** ✅ Correctly implemented

- `opencv_cuda` is excluded for `aqmh_reconstruction` in `phase_supports_backend()` (line 85-89: not listed)
- `cuda` and `opencv_opencl` are enabled (line 93-97)
- `choose_auto_backend()` iterates `{cuda, opencv_cuda, opencv_opencl, cpu}` and skips `opencv_cuda` via `phase_supports_backend` → correct priority: `cuda > opencv_opencl > cpu`

### 4.2 Rollout Safety Gate

**Status:** ⚠️ Violated

- Config default is `"auto"` (should be `"disabled"`)
- Gate logic in `runner_phase_aqmh_reconstruction.cpp:78-85` is correct:
  - `"force"` → always use GPU
  - `"auto"` → use GPU if backend selected and not CPU
  - `"disabled"` → always CPU
- But with `"auto"` default, GPU is active by default

### 4.3 Legacy Fallback Path

**Location:** `src/core/acceleration.cpp:2552-2572`

`AccelerationOps::reconstruct_aqmh()` still calls the CPU path and unconditionally sets `acceleration_fallback = true` when `using_gpu` and phase is `aqmh_reconstruction`. This is only called from `tests/test_aqmh_reconstruction.cpp:322` — not from the production pipeline. It's dead code for production but should be removed or updated to avoid confusion.

### 4.4 CUDA Kernel Correctness

**Status:** Functionally correct, performance-crippled

- Row-chunking: ✅ Correct (chunks along rows, all frames per chunk)
- Deterministic tiebreaker: ✅ `(value, frame_index)` sort order
- Sigma-clip: ✅ Matches CPU logic (weighted median, MAD, keep-floor)
- Cherry-pick: ✅ Top-K by score, deterministic
- Uniform control: ✅ Reuse optimization when weights are equal

### 4.5 OpenCL Kernel

**Status:** Mirrors CUDA kernel, same issues

- Same full-frame loading problem (§3.1)
- Same full Q-map loading problem (§3.2)
- Same fixed-size local array approach (§3.4)
- Same bitonic sort over 1024 elements (§3.5)

---

## 5. AQMH Phase Performance Analysis

### 5.1 AQMH_MAPS Phase

**Current implementation:** Multi-threaded worker pool with per-frame quality map computation.

**Performance characteristics:**
- Each frame is processed independently → good parallelism
- `compute_aqmh_quality_map` does 4 scales × 4 separable filter passes = 16 O(W×H×R) operations
- Cache write is synchronous per frame
- Prefetch coordinator publishes each frame after write → overlaps Q-map I/O with continued map computation

**Bottleneck:** The 2.2× regression (565→1215s) mentioned in the optimization plan (§3.0) was never profiled. Likely contributors:
- `source_masked_frame` creates a full W×H copy per frame
- `downsample_valid_mean` creates a full downsampled copy per scale
- `masked_laplacian`, `local_variance`, `local_mean_and_count`, `phi_snr`, `phi_artifact` each create full W×H matrices
- `mask_aware_bilinear_upsample` creates a full W×H upsampled copy per scale
- Total temporary allocations per frame: ~8-12 × W × H × sizeof(float)

**Recommendation:** Profile with `perf record` or built-in timers. Consider fusing the per-scale pipeline stages to reduce temporary allocations.

### 5.2 AQMH_RECONSTRUCTION Phase

**Current implementation:** CPU path with row-chunked region streaming, OpenMP parallelism.

**Performance characteristics:**
- Row chunks of ~53 rows (auto-sized from memory budget)
- Per chunk: load all 645 frames' rows + Q-map regions → gather samples → sigma-clip per pixel
- Cherry-pick gate: full pre-pass before main loop
- OpenMP parallel over pixels with `schedule(dynamic, 64)`

**Bottleneck:** The cherry-pick gate pre-pass (§3.7) and the per-pixel sigma-clip with weighted median/MAD sort. For 645 frames, each pixel sorts 645 samples 3× (3 clip iterations) = ~1935 sort operations per pixel.

**GPU path bottleneck:** All issues in §3.1–3.5 combine to make the GPU path slower than CPU.

### 5.3 AQMH_DIAGNOSTICS Phase

**Current implementation:** Post-reconstruction, reads Q-maps from cache for block statistics.

**Performance characteristics:**
- Re-reads Q-maps from cache (additional I/O after reconstruction)
- Block-level statistics, heatmaps, region extraction
- Configurable via `level`, `per_frame_blocks`, `heatmaps`, `regions`, `format`

**Bottleneck:** Re-reading Q-maps that were already loaded during reconstruction. The prefetch coordinator only prefetches into the LRU — if `max_resident_maps=4` and 645 frames, most maps will have been evicted by the time diagnostics runs.

**Recommendation:** Consider computing block-level diagnostics inline during the AQMH_MAPS phase (per-frame) rather than re-reading Q-maps post-reconstruction.

---

## 6. Recommended Fix Priority

### Immediate (safety + quick wins)

1. **Fix `gpu_reconstruction` default** to `"disabled"` — 1-line change in `configuration.hpp:267`
2. **Remove `invalidate_mapping(fi)`** from `aqmh_frame_loader` lambda — 1-line deletion in `runner_phase_aqmh_reconstruction.cpp:93`

### Short-term (GPU I/O fix)

3. **Pass region loaders to GPU functions** — extend `reconstruct_aqmh_weighted_cuda()` and `_opencl()` to accept `AqmhFrameRegionLoader` and `AqmhMaskRegionLoader`, use them instead of full-frame loaders
4. **Use `read_region` for Q-maps in GPU path** — replace `read_cached(fi)` with `read_region(fi, y0, rows)`

### Medium-term (GPU kernel optimization)

5. **Adaptive sort size** — use `next_pow2(n)` instead of always 1024 in bitonic sort
6. **Reduce `kMaxFramesCompile`** to 768 or use template parameter
7. **Double-buffered chunk processing** — overlap chunk N download with chunk N+1 upload

### Long-term (algorithmic)

8. **Online/streaming weighted statistics** — replace per-pixel sort with incremental algorithm
9. **Fuse AQMH_MAPS pipeline stages** — reduce temporary allocations
10. **Inline block diagnostics** — compute during AQMH_MAPS instead of post-reconstruction

---

## 7. Conclusion

The implementation plan was faithfully implemented in terms of functionality. The performance issues stem from **two categories**:

1. **GPU I/O mismatch**: The GPU paths load full frames and full Q-maps per chunk instead of using region streaming. This is the single largest performance issue and makes GPU reconstruction ~74× slower than it should be for I/O-bound workloads.

2. **GPU kernel inefficiency**: Fixed-size 1024-element arrays and bitonic sort over all 1024 elements regardless of actual sample count, combined with massive local memory spill, make the kernel compute-bound in a way that negates GPU parallelism.

The CPU path is well-optimized with region streaming, OpenMP parallelism, and the cache routing fix. The `gpu_reconstruction` safety gate default violation means the broken GPU path is active by default, which likely explains why "AQMH phases still take extremely long" — the system is using the GPU path (which is slower than CPU due to I/O issues) instead of the optimized CPU path.

**Fixing items 1-2 (default to `"disabled"` + remove `invalidate_mapping`) will immediately restore CPU-path performance.** Fixing items 3-4 will make the GPU path viable. Items 5-8 will make it competitive.
