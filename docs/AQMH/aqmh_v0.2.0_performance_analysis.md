# AQMH v0.2.0 Performance Regression Analysis

**Date:** 2026-07-05  
**Analyzed Runs:**
- `m31-gpu1_20260628_210413` (v0.1.0 / legacy AQMH reconstruction)
- `M31_AQMH2_20260704_233605` (v0.2.0 / new AQMH pipeline)

**Goal:** Understand why v0.2.0 is significantly slower than v0.1.0 for the same dataset (645 M31 frames, 3840x2160, OSC) and propose concrete optimizations.

---

## 1. Executive Summary

| Metric | v0.1.0 | v0.2.0 | Delta |
|--------|--------|--------|-------|
| **Total wall time** | **29.7 min** | **56.6 min** | **+90 %** |
| Reconstruction + maps time | 901.9 s | 2 117.0 s | **+2.3x** |
| Artifact size (`aqmh_metrics.json`) | 256 KB | 178 MB | **+694x** |
| Additional artifacts | none | `aqmh_regions.json` (100 MB) | new |
| Reconstruction backend | **OpenCV CUDA GPU** | **CPU exact (GPU fallback)** | regression |
| Cache reads | 1 290 | 28 380 | **+22x** |

The v0.2.0 pipeline restructures the previous `AQMH_QUALITY_MAPS` + `TILE_RECONSTRUCTION` steps into three sequential phases (`AQMH_MAPS`/`AQMH_RECONSTRUCTION`/`AQMH_DIAGNOSTICS`). The biggest regressions are:

1. **CPU-only reconstruction** replacing the v0.1.0 OpenCV CUDA kernel (901.7 s vs 336.9 s). The v0.2.0 artifact shows `acceleration_fallback: true` — the GPU backend was selected but fell back to CPU because no GPU implementation exists for the v0.2.0 weighted-MAD algorithm.
2. **Slower AQMH_MAPS phase** (formerly `AQMH_QUALITY_MAPS`/`LOCAL_METRICS`, 565.0 s in v0.1.0) now costs 1 215.3 s — a 2.2x increase.
3. **AQMH_DIAGNOSTICS phase** writes 278 MB of diagnostic JSON and takes 318.6 s with no opt-out.

**Recommended actions:**
- Re-enable GPU acceleration for the v0.2.0 reconstruction path.
- Make the `AQMH_DIAGNOSTICS` phase configurable or compute it only on demand.
- Reduce JSON diagnostic payload size (replace full per-frame block arrays with summaries).
- Consider fusing `AQMH_MAPS` with reconstruction again, or at least overlap their execution.

---

## 2. Dataset and Environment

Both runs processed the same input directory:

```
/media/tc_ssd/M31_ligths_all
```

Common parameters:

| Parameter | Value |
|-----------|-------|
| Frames | 645 |
| Image size | 3840 x 2160 |
| Color mode | OSC (GBRG) |
| Canvas size | 3924 x 2310 |
| Parallel workers | 8 |
| Acceleration | `auto` → CUDA available |
| AQMH enabled | `true` |

Configuration differences are minimal (mostly BGE-related settings) and do not explain the performance gap.

---

## 3. Phase-by-Phase Timing

| Phase | v0.1.0 | v0.2.0 | Delta |
|-------|--------|--------|-------|
| SCAN_INPUT | 0.5 s | 1.3 s | +0.8 s |
| CHANNEL_SPLIT | 0.0 s | 0.0 s | — |
| NORMALIZATION | 96.5 s | 111.4 s | +14.9 s |
| GLOBAL_METRICS | 77.2 s | — | removed/merged |
| TILE_GRID | 0.3 s | — | removed/merged |
| REGISTRATION | 281.6 s | 315.1 s | +33.5 s |
| PREWARP | 293.8 s | 328.4 s | +34.6 s |
| COMMON_OVERLAP | 0.3 s | 0.0 s | −0.3 s |
| AQMH_QUALITY_MAPS / LOCAL_METRICS | **565.0 s** | — | renamed to AQMH_MAPS |
| AQMH_MAPS | — | **1 215.3 s** | renamed/restructured from AQMH_QUALITY_MAPS |
| AQMH_GLOBAL_QUALITY | — | 0.0 s | new |
| AQMH_RECONSTRUCTION | — | **901.7 s** | replaces TILE_RECONSTRUCTION |
| AQMH_DIAGNOSTICS | — | **318.6 s** | new |
| TILE_RECONSTRUCTION | **336.9 s** | — | removed |
| STATE_CLUSTERING | 0.0 s | 0.0 s | — |
| SYNTHETIC_FRAMES | 0.0 s | 0.0 s | — |
| STACKING | 1.3 s | 3.2 s | +1.9 s |
| DEBAYER | 0.5 s | 0.4 s | −0.1 s |
| ASTROMETRY | 8.0 s | 8.1 s | +0.1 s |
| BGE | 60.1 s | 60.3 s | +0.2 s |
| PCC | 56.5 s | 55.3 s | −1.2 s |
| HYPERMETRIC_STRETCH | 1.2 s | 1.1 s | −0.1 s |
| **Total** | **29.7 min** | **56.6 min** | **+26.9 min** |

The v0.1.0 run had two separate phases: `AQMH_QUALITY_MAPS`/`LOCAL_METRICS` (565.0 s) for quality-map computation and `TILE_RECONSTRUCTION` (336.9 s) for weighted reconstruction on the GPU. The v0.2.0 pipeline renames and restructures these into three sequential phases: `AQMH_MAPS` (renamed from `AQMH_QUALITY_MAPS`), `AQMH_RECONSTRUCTION` (replaces `TILE_RECONSTRUCTION`), and `AQMH_DIAGNOSTICS` (new). Together they consume 2 435.6 s, compared to 901.9 s for the two v0.1.0 phases — a 2.7x increase.

---

## 4. Root Cause Analysis

### 4.1 Reconstruction Backend: GPU (v0.1.0) → CPU (v0.2.0)

The most critical regression is the reconstruction backend.

**v0.1.0 artifact (`tile_reconstruction.json`):**
```json
{
  "acceleration_used": true,
  "cache_stats": {
    "bytes_read": 5846563800,
    "bytes_written": 2923281900,
    "read_count": 1290,
    "write_count": 645,
    "max_resident_maps_observed": 2
  }
}
```

**v0.2.0 artifact (`aqmh_reconstruction.json`):**
```json
{
  "acceleration": {
    "selected_backend": "opencv_cuda",
    "using_gpu": true
  },
  "acceleration_fallback": true,
  "acceleration_used": false,
  "execution_backend": "cpu_exact_v0_2",
  "region_streaming": true,
  "chunk_count": 44,
  "chunk_rows": 53,
  "cache_stats": {
    "bytes_read": 3085264620,
    "bytes_written": 2923281900,
    "read_count": 28380,
    "write_count": 645,
    "max_resident_maps_observed": 0
  }
}
```

Observations:
- v0.1.0 used the OpenCV CUDA kernel (`cv::cuda::GpuMat` operations, not native NVIDIA CUDA) and kept up to 2 quality maps resident in memory.
- v0.2.0 **selected** `opencv_cuda` as the backend (and `acceleration_context.json` reports `using_gpu: true`), but the artifact shows `acceleration_fallback: true` and `acceleration_used: false`. This means the GPU backend was selected by the acceleration framework but the v0.2.0 reconstruction code has no GPU implementation for the exact weighted-MAD algorithm, so it fell back to the "exact CPU region-streaming backend" with 44 chunks of 53 rows each.
- v0.2.0 reads **less total bytes** (3.09 GB vs 5.85 GB) but performs **22x more read calls** (28 380 vs 1 290). The small chunk size and CPU streaming create massive overhead.
- The log explicitly states: `v0.2 weighted-MAD uses the exact CPU region-streaming backend; the legacy v0.1 CUDA kernel is intentionally not used.` (The log label "CUDA kernel" here means the v0.1.0 **OpenCV CUDA** path, not a native NVIDIA CUDA kernel.)

Impact: The reconstruction step alone is **2.7x slower** (901.7 s vs 336.9 s), and it is the single largest contributor to the slowdown.

### 4.2 Slower `AQMH_MAPS` Phase (Renamed from `AQMH_QUALITY_MAPS`/`LOCAL_METRICS`)

The quality-map functionality existed in v0.1.0 as a separate phase called `AQMH_QUALITY_MAPS` (which logged its end event as `LOCAL_METRICS`). It took **565.0 s** (19:16:44–19:26:09). In v0.2.0 it was renamed to `AQMH_MAPS` and takes **1 215.3 s** — a 2.2x increase. The phase was not "embedded inside TILE_RECONSTRUCTION" as previously stated; it was always a separate phase.

```
[AQMH] requested=auto selected=opencv_cuda execution=GPU (auto-detected) [OpenCV CUDA]
[AQMH] Using 8 parallel workers for quality-map computation cpu_workers=8 gpu=yes backend=opencv_cuda
```
(Note: `selected=opencv_cuda` means the phase runs via **OpenCV CUDA**, not native NVIDIA CUDA kernels.)

Duration: **1 215.3 s** (≈1.88 s/frame), up from 565.0 s (≈0.88 s/frame) in v0.1.0. The GPU is selected, but the phase still dominates the pipeline. Possible reasons for the 2.2x slowdown:
- Each frame is read from disk cache, preprocessed, downsampled across pyramid scales, and the per-pixel quality map is computed and written back to disk.
- The CPU workers (8) may be the bottleneck for the pyramid/combination steps even though the heavy convolutions are on GPU.
- There is no overlap with reconstruction; the whole pipeline waits for all 645 maps to be written before reconstruction starts.

### 4.3 `AQMH_DIAGNOSTICS` Phase: Mandatory and Heavy

v0.2.0 adds a mandatory `AQMH_DIAGNOSTICS` phase (318.6 s) that:
- Computes block-level statistics and heatmaps for every frame.
- Writes them to `aqmh_metrics.json` (178 MB) and `aqmh_regions.json` (100 MB).

Size breakdown of `aqmh_metrics.json`:

| Field | Size | Share |
|-------|------|-------|
| `frames` | 107.5 MB | 99.7 % |
| `diagnostics` | 0.37 MB | 0.3 % |
| all other | <0.01 MB | <0.1 % |

Each frame entry is 164 KB of `block_diagnostics` (9 blocks per frame). The diagnostics phase has no configuration flag to disable or reduce its output.

Impact: This phase adds **5.3 minutes** and **278 MB of disk I/O** to every run, regardless of whether the user needs the detailed per-block heatmaps.

### 4.4 Increased I/O and Cache Pressure

- v0.2.0 writes more intermediate artifacts (178 MB + 100 MB diagnostic JSON).
- The CPU region-streaming reconstruction performs 28 380 cache reads, compared to 1 290 in the GPU path.
- `persistent_mmap_cache_views` and `region_streaming` are enabled, but the high read-count suggests the small chunk size (53 rows) thrashes the cache.

### 4.5 Memory Observations

System memory during a comparable v0.2.0 run (M42):
- 55/62 GB RAM used
- 28/31 GB swap used

The CPU-streaming backend likely keeps many frame chunks in memory or swap, while the v0.1.0 GPU path kept only 2 resident maps.

### 4.6 Are `parallel_workers` Respected?

**Yes.** I verified both the runtime logs and the source code:

- **AQMH_MAPS** log entries show `workers=8 cpu_workers=8 gpu=yes backend=opencv_cuda`.
- **AQMH_RECONSTRUCTION** prints `cpu_workers=8` and passes `cfg.runtime_limits.parallel_workers` into `reconstruction::AqmhReconstructionConfig::parallel_workers` (`@/media/data/programming/tile_compile/tile_compile_cpp/apps/runner_phase_aqmh_reconstruction.cpp:66`).
- The CPU reconstruction then uses OpenMP `#pragma omp parallel num_threads(...)` with `schedule(dynamic, 64)` over the pixels of each chunk (`@/media/data/programming/tile_compile/tile_compile_cpp/src/reconstruction/aqmh_reconstruction.cpp:237-249`).

**Why it still feels slow:** The v0.2.0 backend is CPU-only and **row-streaming**. Each chunk is only 53 rows, so the per-chunk work is small and the 8 threads spend much of their time waiting on cache I/O and synchronization. More workers cannot overcome the fact that the algorithm is no longer running on the GPU.

There is also a cap on the reconstruction chunk size: the code computes the chunk rows from `memory_budget / 2` clamped to `[128, 1536]` MB. For the M31 run with `memory_budget: 2048`, the chunk size was exactly 53 rows (`2 048 / 2 = 1024` MB). Raising `memory_budget` to 4 096 MB would only increase the chunk size to about 79 rows (clamped at `4 096 / 2 = 2048` → 1536 MB), not a dramatic change. To use larger chunks, the clamp would need to be relaxed.

---

## 5. Quantified Impact

| Bottleneck | Time Added | Share of Extra Time |
|------------|------------|---------------------|
| CPU-only reconstruction (GPU fallback) | 564.8 s | 35 % |
| Slower AQMH_MAPS phase (565→1 215 s) | 650.3 s | 41 % |
| AQMH_DIAGNOSTICS phase (new) | 318.6 s | 20 % |
| Removed GLOBAL_METRICS + TILE_GRID | −77.5 s | −5 % |
| Other small changes | ~35 s | 2 % |
| **Total extra** | **~1 491 s** | **~93 %** |

(Shares are of the total extra phase time. The net wall-time increase is larger — 26.9 min = 1 614 s — due to inter-phase overhead and gaps not captured by individual phase timings.)

Net wall-time increase: 26.9 min (90 %).

---

## 6. Optimization Recommendations

### 6.1 Re-enable GPU Reconstruction (Highest Priority)

**Three-tier backend strategy:**

| Tier | Backend | Target Platform | Status |
|------|---------|-----------------|--------|
| 1 | **Native CUDA** (`cuda`) | NVIDIA GPUs (Linux/Win) | not implemented — `cuda` backend returns `false` for all phases |
| 2 | **OpenCL** (`opencv_opencl`) | AMD/Intel GPUs (Win/Mac/Linux) | not implemented — `aqmh_reconstruction` not in allowed list |
| 3 | **CPU exact** (`cpu`) | All platforms (fallback) | implemented — current default via `acceleration_fallback` |

The v0.1.0 OpenCV CUDA algorithm is **not** re-used — the goal is to run the v0.2.0 exact weighted-MAD on the GPU. The v0.2.0 code currently selects `opencv_cuda` but falls back to CPU (`acceleration_fallback: true`) because no GPU implementation exists.

**Backend availability in the codebase:**
- `opencv_opencl` is already integrated in the build (`TILE_COMPILE_HAS_OPENCV_OPENCL`, `cv::ocl`) and used by other phases (prewarp, stacking, tile_reconstruction).
- `aqmh_reconstruction` is currently **not** listed for `opencv_opencl` (`@/media/data/programming/tile_compile/tile_compile_cpp/src/core/acceleration.cpp:91-94`).
- The native `cuda` backend returns `false` for all phases (`@/media/data/programming/tile_compile/tile_compile_cpp/src/core/acceleration.cpp:96-97`).
- Both must be activated for `aqmh_reconstruction`.

**Implementation plan:**

1. **Native NVIDIA CUDA backend (Tier 1 — highest priority)**
   - Add a new translation unit, e.g. `src/reconstruction/aqmh_reconstruction_cuda.cu`.
   - Implement a real per-pixel `weighted_mad_sigma_clip` `__global__` kernel (not OpenCV CUDA wrappers):
     - Gather `frame_count` samples per pixel (value, weight = `gw * q`), honoring `canvas_mask` and frame masks.
     - Apply cherry-pick if enabled (`aqmh_select_top_k` logic on device).
     - Compute weighted median / MAD, iterative sigma-clipping, and `min_n_eff` / `min_fraction` checks.
     - Return weighted mean, weight sum, and optional uniform-control image.
   - Activate the `cuda` backend for `aqmh_reconstruction` in `phase_supports_backend()`.
   - Keep the CPU path as a deterministic fallback when `tile_compile_with_cuda` is false or the CUDA kernel reports a failure.
   - Hook it into `runner_phase_aqmh_reconstruction.cpp` so the phase selects the **native CUDA** backend when `acceleration.selected_backend == cuda` (distinct from `opencv_cuda`).

2. **OpenCL backend (Tier 2 — cross-platform fallback)**
   - Add `AccelerationPhase::aqmh_reconstruction` to the `opencv_opencl` case in `@/media/data/programming/tile_compile/tile_compile_cpp/src/core/acceleration.cpp:91-94`.
   - Implement `opencl_aqmh_weighted_reconstruction_impl` using **real OpenCL C kernels** (via `cv::ocl::ProgramSource` or direct OpenCL API), **not** `cv::UMat` reductions:
     - `cv::UMat` operations (`cv::mean`, `cv::multiply`, `cv::compare`) cannot express per-pixel weighted-MAD with iterative sigma-clipping and quickselect — this requires custom kernel code.
     - The OpenCL kernel mirrors the CUDA kernel logic 1:1: gather samples, cherry-pick, weighted median/MAD, sigma-clipping, `min_n_eff`/`min_fraction` checks.
     - Cherry-pick can remain on the CPU as a pre-pass if it simplifies the kernel; the heavy per-pixel weighted-MAD is the part that must run on the GPU.
   - This enables GPU acceleration on **AMD/Intel GPUs** (Windows/Mac/Linux) where CUDA is unavailable.
   - On Mac, OpenCL is deprecated by Apple but still functional; a future Metal backend could replace it if needed.

3. **Avoid duplicating region streaming for GPU paths**
   - The current CPU path streams 53-row chunks because of memory limits. A GPU path can load full frames into device memory once (or in larger batches) and process the whole canvas on the GPU.
   - Remove or bypass the `chunk_rows` loop when a GPU backend is active.

Expected gain: 2–3x faster reconstruction, reducing 901.7 s to roughly 300–450 s.

### 6.2 Make `AQMH_DIAGNOSTICS` Configurable

Add a configuration flag to control the diagnostics phase:

```yaml
aqmh:
  diagnostics:
    enabled: true        # default: true
    level: "summary"     # options: "summary" | "full" | "none"
    per_frame_blocks: false
    heatmaps: false
```

Behavior:
- `none`: Skip the diagnostics phase entirely; write only the lightweight `diagnostics` array (0.37 MB).
- `summary`: Keep the `diagnostics` array and global metrics, but omit the per-frame `frames` block.
- `full`: Current behavior (178 MB + 100 MB).

Expected gain: Up to 318.6 s saved and 278 MB less disk I/O when `none` or `summary` is selected.

### 6.3 Reduce Diagnostic Payload Size

If `full` diagnostics are required:
- Store block heatmaps in a compact binary format (e.g., FITS or compressed binary) instead of JSON.
- Reduce block count or resolution (`r_morph_canvas_px`, `q_region` defaults could be relaxed for production runs).
- Write per-frame diagnostics asynchronously or after the main pipeline completes.

### 6.4 Overlap AQMH_MAPS with Reconstruction

Currently `AQMH_MAPS` must complete before `AQMH_RECONSTRUCTION` starts. For streaming/region-based reconstruction, this is not strictly necessary. Options:
- Start reconstructing rows as soon as their quality maps are available (producer-consumer pipeline).
- Process frames in batches: compute maps for N frames, then immediately reconstruct them, freeing their cache.

Expected gain: 20–40 % of the combined maps+reconstruction time, because the two phases are currently almost perfectly sequential.

### 6.5 Increase Region-Streaming Chunk Size

The current chunk size is 53 rows (44 chunks). Larger chunks would reduce the 28 380 cache read calls:

```yaml
runtime_limits:
  memory_budget: 4096        # increases chunk size, but capped at 1536 MB effective
```

The chunk size is computed as `memory_budget / 2` clamped to `[128, 1536]` MB (`@/media/data/programming/tile_compile/tile_compile_cpp/src/reconstruction/aqmh_reconstruction.cpp:159-164`). For the M31 run this gave 53 rows; with 4 096 MB it would only grow to about 79 rows. To get substantially larger chunks, either:
- raise or remove the 1536 MB clamp in the code, or
- add an explicit `aqmh.reconstruction.chunk_rows` config that bypasses the budget formula.

Trade-off: higher memory usage, but modern systems (62 GB RAM) can handle it. Expected gain: 20–30 % reduction in reconstruction time.

### 6.6 Cache Quality Maps in Memory

v0.2.0 uses `persistent_mmap_cache_views` and `max_resident_maps_observed: 0`, meaning maps are read from disk for every chunk. Keeping 2–4 maps resident (like v0.1.0) would eliminate repeated disk reads for the same frame across chunks.

Expected gain: significant reduction in the 3.09 GB of cache reads and the 28 380 read calls.

### 6.7 Investigate AQMH_MAPS CPU Bottlenecks

Even with `gpu=yes`, the phase takes 1.88 s/frame. Profiling should focus on:
- `aqmh_quality_map.cpp` pyramid construction and downsample/combine steps.
- CPU post-processing after GPU kernels (bilinear upsample, artifact mask, etc.).
- Disk I/O for writing the cache.

Quick win: parallelize the per-frame write-back or use asynchronous writes.

---

## 7. Suggested Implementation Plan

1. **Immediate (no code changes):**
   - Document that v0.2.0 is intentionally slower for the sake of algorithmic correctness/diagnostics.
   - Advise users to run production stacks with a config that disables BGE guards if they want the fastest path (already done in M31_AQMH2).

2. **Short-term (config only):**
   - Add `aqmh.diagnostics.enabled` and `aqmh.diagnostics.level` flags.
   - Add `aqmh.reconstruction.chunk_rows` to tune the region-streaming granularity.

3. **Medium-term (code changes):**
   - Overlap AQMH_MAPS and AQMH_RECONSTRUCTION using a frame batch queue.
   - Increase default resident quality maps to 2–4.

4. **Long-term (code changes):**
   - Port the exact v0.2.0 weighted-MAD reconstruction to native NVIDIA CUDA.
   - Replace JSON diagnostics with a compact binary format.

---

## 8. Conclusion

The v0.2.0 pipeline is **90 % slower** than v0.1.0 for this dataset because it:

1. **Replaced the fast GPU reconstruction with an exact CPU region-streaming backend** (GPU was selected but fell back to CPU because no v0.2.0 GPU implementation exists).
2. **Slowed down the `AQMH_QUALITY_MAPS`/`LOCAL_METRICS` phase** (565→1 215 s) by renaming it to `AQMH_MAPS` and making it 2.2x slower.
3. **Added a mandatory AQMH_DIAGNOSTICS phase** that writes 278 MB of per-frame block diagnostics.

The GPU is still used for AQMH_MAPS, but the CPU reconstruction and diagnostic I/O dominate the runtime. The most impactful fixes are re-enabling GPU reconstruction and making the diagnostics phase optional.

---

## Appendix A: Relevant Files

- `tile_compile_cpp/src/reconstruction/aqmh_reconstruction.cpp` — v0.2.0 CPU reconstruction backend.
- `tile_compile_cpp/apps/runner_phase_aqmh_reconstruction.cpp` — phase orchestration.
- `tile_compile_cpp/apps/runner_phase_aqmh_maps.cpp` — thin wrapper around local metrics.
- `tile_compile_cpp/apps/runner_phase_aqmh_diagnostics.cpp` — diagnostic JSON generation.
- `tile_compile_cpp/apps/runner_phase_local_metrics.cpp` — quality map computation.
- `tile_compile_cpp/src/metrics/aqmh_quality_map.cpp` — quality map algorithm.
- `tile_compile_cpp/include/tile_compile/config/configuration.hpp` — AQMH config structs.

## Appendix B: Raw Timing Data

```
v0.1.0 (m31-gpu1_20260628_210413)
  SCAN_INPUT                  0.5 s
  NORMALIZATION              96.5 s
  GLOBAL_METRICS             77.2 s
  TILE_GRID                   0.3 s
  REGISTRATION              281.6 s
  PREWARP                   293.8 s
  COMMON_OVERLAP               0.3 s
  AQMH_QUALITY_MAPS/LOCAL_METRICS 565.0 s
  TILE_RECONSTRUCTION       336.9 s
  STACKING                    1.3 s
  DEBAYER                     0.5 s
  ASTROMETRY                  8.0 s
  BGE                        60.1 s
  PCC                        56.5 s
  HYPERMETRIC_STRETCH         1.2 s
  Total:                    29.7 min

v0.2.0 (M31_AQMH2_20260704_233605)
  SCAN_INPUT                 1.3 s
  NORMALIZATION            111.4 s
  REGISTRATION             315.1 s
  PREWARP                  328.4 s
  COMMON_OVERLAP             0.0 s
  AQMH_MAPS              1 215.3 s
  AQMH_GLOBAL_QUALITY        0.0 s
  AQMH_RECONSTRUCTION      901.7 s
  AQMH_DIAGNOSTICS         318.6 s
  STACKING                   3.2 s
  DEBAYER                    0.4 s
  ASTROMETRY                 8.1 s
  BGE                       60.3 s
  PCC                       55.3 s
  HYPERMETRIC_STRETCH        1.1 s
  Total:                   56.6 min
```

## Appendix C: Artifact Size Data

```
v0.1.0 aqmh_metrics.json: 256 KB
  - diagnostics: 206 KB (99.6 %)
  - all other:  < 1 KB

v0.2.0 aqmh_metrics.json: 178 MB
  - frames:   107.5 MB (99.7 %)
  - diagnostics: 0.37 MB (0.3 %)
  - all other:  < 1 MB

v0.2.0 aqmh_regions.json: 100 MB
```
