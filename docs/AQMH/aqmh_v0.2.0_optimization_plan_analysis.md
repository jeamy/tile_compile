# AQMH v0.2.0 — Optimization Plan Analysis

**Author:** Mistral Vibe  
**Date:** 2026-07-05  
**Status:** Final Analysis  
**Source:** Analysis of `docs/AQMH/aqmh_v0.2.0_optimization_plan.md` (revised version)  

---

## 📋 Executive Summary

### Overall Assessment: **🟡 GOOD BUT NOT YET IMPLEMENTABLE**

| **Criteria**          | **Score** | **Status**                                                                                     |
|----------------------|-----------|----------------------------------------------------------------------------------------------|
| **Completeness**     | 85%       | Most aspects covered, but **critical technical details missing**                            |
| **Logical Correctness** | 70%    | **Design decisions fixed**, but **implementation blocks remain**                             |
| **Consistency**      | 90%       | **Major design inconsistencies resolved**, minor details need clarification                   |
| **Implementability** | 60%       | **Cannot be implemented as-is** – code snippets won't compile or have undefined behavior     |

---

### ✅ Major Improvements from First Version

1. **🎯 Critical Design Flaws Fixed:**
   - Overlap architecture now correctly implements **I/O-only prefetch (Option C)** with realistic 5–15% gain estimates (vs. previous 20–40%)
   - CUDA kernel uses **row-chunking** instead of frame-batching (prevents correctness bugs from partial frame sets)
   - Cache bug **root cause identified** (routing bug in `read_region()`, not missing feature)
   - **Backend arbitration rules** added (`cuda > opencv_opencl > cpu` for `aqmh_reconstruction`)
   - **Rollout safety gate** added (`gpu_reconstruction: disabled` default)

2. **📈 New Critical Items:**
   - **Item 3.0:** AQMH_MAPS profiling (650s/41% of regression) – now highest priority in Section 3
   - Correctness constraints for GPU kernels (row-chunking requirement)

3. **📉 Realistic Estimates:**
   - Overlap: **5–15%** (corrected from 20–40%)
   - Binary format: **~6 MB** (corrected from 1.5 MB)

---

### ❌ Remaining Critical Issues

| **Severity** | **Issue** | **Location** | **Impact** | **Status** |
|--------------|-----------|--------------|------------|------------|
| 🔴 **BLOCKER** | `memory_budget_mb` missing from `AqmhReconstructionConfig` | Item 2.1.2, 2.1.5 | Code won't compile | **MUST FIX** |
| 🔴 **BLOCKER** | `denom` undefined in auto formula | Item 2.1.5 | Code won't compile | **MUST FIX** |
| 🔴 **BLOCKER** | `MAX_FRAMES` undefined in CUDA kernel | Item 4.1.2 | Code won't compile | **MUST FIX** |
| 🔴 **BLOCKER** | Backend priority not implemented | Item 4.1.3 | Wrong backend may be selected | **MUST FIX** |
| 🔴 **BLOCKER** | Binary format size calculation incorrect | Item 4.4.1 | Documentation error | **MUST FIX** |
| 🟡 **HIGH** | `read_region()` routing fix not detailed | Item 3.2 | Implementation risk | **SHOULD FIX** |
| 🟡 **HIGH** | Thread management in `AqmhPrefetchCoordinator` unclear | Item 3.1.1 | Implementation risk | **SHOULD FIX** |
| 🟡 **HIGH** | Local memory spill in CUDA kernel | Item 4.1.2 | Performance risk | **SHOULD FIX** |

---

## 🔍 Detailed Analysis

---

## 1. Item 2: Short-Term (Config Only)

### ✅ Strengths
- Config structure extensions are well-defined
- New fields have sensible defaults
- Parsing, serialization, and validation are covered
- Schema JSON updates are correct

### ❌ Critical Issues

#### 1.1 `memory_budget_mb` Missing from Config (BLOCKER)
**Location:** Item 2.1.2 (Config Struct) + Item 2.1.5 (Implementation)  
**Problem:**
```cpp
// In Item 2.1.5:
const size_t target_mb = static_cast<size_t>(std::clamp(
    cfg.memory_budget_mb / 2, 128, 1536));
```
`cfg` is of type `AqmhReconstructionConfig`, which **does not contain** `memory_budget_mb`.  

**Impact:** Code will not compile.  

**🔧 Solution:**
```cpp
// Option A: Add to AqmhReconstructionConfig (RECOMMENDED)
struct AqmhReconstructionConfig {
  // ... existing fields ...
  int chunk_rows = 0;
  size_t memory_budget_mb = 0;  // 0 = use global config
};

// Option B: Pass as parameter to reconstruct_aqmh_weighted()
AqmhReconstructionResult reconstruct_aqmh_weighted(
    size_t frame_count,
    const AqmhFrameLoader& load_frame,
    metrics::QualityMapCache* q_map_cache,
    const VectorXf& global_weights,
    size_t memory_budget_mb,  // NEW PARAMETER
    // ... existing parameters ...
);
```

#### 1.2 `denom` Undefined in Auto Formula (BLOCKER)
**Location:** Item 2.1.5  
**Problem:**
```cpp
const size_t target_bytes = target_mb * 1024u * 1024u;
const size_t denom = std::max<size_t>(1, ...);  // ❌ What is "..."?
chunk_rows = std::max(1, std::min(height, static_cast<int>(target_bytes / denom)));
```

**Impact:** Code will not compile.  

**🔧 Solution:**
```cpp
// Correct formula:
const size_t bytes_per_pixel = sizeof(float);  // 4 bytes
const size_t bytes_per_frame_per_chunk = width * chunk_rows * bytes_per_pixel;
const size_t bytes_per_qmap_per_chunk = bytes_per_frame_per_chunk;
const size_t bytes_per_chunk = (bytes_per_frame_per_chunk + bytes_per_qmap_per_chunk) * frame_count;
const size_t denom = bytes_per_chunk;
```

#### 1.3 Redundancy Between `enabled` and `level`
**Location:** Item 2.1.1  
**Problem:** Both `enabled: false` and `level: "none"` skip the diagnostics phase → **user confusion**.  

**Impact:** Poor UX, potential for inconsistent configurations.  

**🔧 Solution:**
```cpp
// Option A: Remove 'enabled', use only 'level' (RECOMMENDED)
struct AqmhDiagnosticsConfig {
  std::string level = "full";  // "none" = disabled, "summary", "full"
  // ... other fields ...
};

// Option B: Add validation to enforce consistency
void validate_and_normalize_config(AqmhDiagnosticsConfig& cfg) {
  if (!cfg.enabled) {
    cfg.level = "none";
    cfg.per_frame_blocks = false;
    cfg.heatmaps = false;
    cfg.regions = false;
  } else if (cfg.level == "none") {
    cfg.enabled = false;
  }
}
```

#### 1.4 Flag Hierarchy Unclear
**Location:** Item 2.1.1, 2.1.4  
**Problem:** `per_frame_blocks`, `heatmaps`, `regions` are **defined in config** but **only effective when `level == "full"`** (see Item 2.1.4: `if (full_mode && ...)').  

**Impact:** Misleading for users who set these flags with `level: "summary"`.  

**🔧 Solution:**
- **Document explicitly:** These flags are **only effective when `level == "full"`**
- **Add validation:** Automatically disable these flags when `level != "full"`

#### 1.5 Missing Tests for Flag Combinations
**Location:** Item 2.1.8  
**Problem:** No tests for:
- `level: "summary"` + `per_frame_blocks: true` (should ignore flag)
- `enabled: false` + `level: "full"` (should skip phase)
- `level: "none"` + any flags (should skip phase)

**🔧 Solution:**
```cpp
// Add to test_aqmh_validation.cpp
TEST(AqmhDiagnosticsConfig, LevelSummaryIgnoresFlags) {
  AqmhDiagnosticsConfig cfg;
  cfg.level = "summary";
  cfg.per_frame_blocks = true;
  cfg.heatmaps = true;
  cfg.regions = true;
  validate_and_normalize_config(cfg);
  EXPECT_FALSE(cfg.per_frame_blocks);
  EXPECT_FALSE(cfg.heatmaps);
  EXPECT_FALSE(cfg.regions);
}

TEST(AqmhDiagnosticsConfig, DisabledSkipsPhase) {
  AqmhDiagnosticsConfig cfg;
  cfg.enabled = false;
  cfg.level = "full";  // Should be overridden
  validate_and_normalize_config(cfg);
  EXPECT_EQ(cfg.level, "none");
}
```

---

## 2. Item 3: Medium-Term (Code Changes)

### ✅ Strengths
- **Item 3.0 added:** AQMH_MAPS profiling (650s/41% of regression) – **critical for prioritization**
- **Overlap design corrected:** Now **I/O-only prefetch** (Option C) with realistic estimates
- **Cache bug root cause identified:** Routing bug in `read_region()` (not missing feature)
- **Implementation order adjusted:** Profiling (3.0) → Cache fix (3.2) → Overlap (3.1)

### ❌ Critical Issues

#### 2.1 `AqmhPrefetchCoordinator` Thread Management Unclear
**Location:** Item 3.1.1  
**Problem:** 
```cpp
// From plan:
"publish_frame(fi) enqueues a background prefetch task"
```
- **Who manages the threads?**
- **How are tasks executed?**
- **What if all threads are busy?**

**Impact:** Implementation risk – could lead to:
- Blocking behavior
- Thread leaks
- Deadlocks

**🔧 Solution:**
```cpp
// Option A: Use existing ThreadPool (RECOMMENDED if available)
class AqmhPrefetchCoordinator {
public:
  explicit AqmhPrefetchCoordinator(size_t frame_count, ThreadPool& pool)
    : pool_(pool), frame_count_(frame_count) {}
  
  void publish_frame(size_t fi) {
    pool_.enqueue([this, fi] { prefetch_frame(fi); });
  }
  
  void wait_all_prefetched() {
    pool_.wait_for_completion();
  }
private:
  ThreadPool& pool_;
  size_t frame_count_;
};

// Option B: Internal thread pool
class AqmhPrefetchCoordinator {
public:
  explicit AqmhPrefetchCoordinator(size_t frame_count, size_t thread_count = 4)
    : frame_count_(frame_count) {
    for (size_t i = 0; i < thread_count; ++i) {
      threads_.emplace_back(&AqmhPrefetchCoordinator::worker, this);
    }
  }
  
  ~AqmhPrefetchCoordinator() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      shutdown_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_) {
      t.join();
    }
  }
  
  void publish_frame(size_t fi) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push(fi);
    }
    cv_.notify_one();
  }
  
  void wait_all_prefetched() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return prefetched_count_ == frame_count_; });
  }

private:
  void worker() {
    while (true) {
      size_t fi;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
        if (shutdown_ && queue_.empty()) break;
        fi = queue_.front();
        queue_.pop();
      }
      prefetch_frame(fi);
    }
  }
  
  std::vector<std::thread> threads_;
  std::queue<size_t> queue_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::atomic<size_t> prefetched_count_{0};
  bool shutdown_{false};
};
```

#### 2.2 `read_region()` Routing Fix Not Detailed
**Location:** Item 3.2  
**Problem:** Root cause identified (bypasses LRU cache) but **no implementation details**.  

**Impact:** High risk of incorrect implementation.  

**🔧 Solution:**
```cpp
// In QualityMapCache::read_region():
std::vector<float> QualityMapCache::read_region(size_t frame_idx, const Rect& region) {
  // 1. Check if full map is in LRU cache
  if (resident_.count(frame_idx)) {
    const auto& full_map = resident_[frame_idx];
    // 2. Extract region from full map
    return extract_region(full_map, region);
  } else {
    // 3. Fall back to direct disk read (existing behavior)
    return read_direct_from_disk(frame_idx, region);
  }
}

// Helper function:
std::vector<float> QualityMapCache::extract_region(
    const MatrixXf& full_map, const Rect& region) {
  std::vector<float> result(region.width * region.height);
  for (int y = 0; y < region.height; ++y) {
    for (int x = 0; x < region.width; ++x) {
      result[y * region.width + x] = full_map(region.y + y, region.x + x);
    }
  }
  return result;
}
```

#### 2.3 Expected Gain for Overlap May Be Optimistic
**Location:** Item 3.1.3  
**Problem:** 
- Estimated gain: **5–15% of combined AQMH_MAPS + AQMH_RECONSTRUCTION time**
- **But:** If `AQMH_GLOBAL_QUALITY` is very fast, prefetch may complete **before** reconstruction starts → **no gain**

**Impact:** Users may expect higher improvements than actually achieved.  

**🔧 Solution:**
- **Add timeout to `wait_all_prefetched()`:**
  ```cpp
  void wait_all_prefetched(std::chrono::seconds timeout = std::chrono::seconds(5)) {
    if (all_prefetched()) return;
    std::unique_lock<std::mutex> lock(mutex_);
    if (!cv_.wait_for(lock, timeout, [this] { return all_prefetched(); })) {
      log_warning("Prefetch timeout, continuing with available frames");
    }
  }
  ```
- **OR: Lazy loading (RECOMMENDED):**
  - Reconstruction **starts immediately** and loads missing frames on-demand
  - No blocking, no wasted time

#### 2.4 Missing Tests for Cache Fix
**Location:** Item 3.5  
**Problem:** No tests for:
- `read_region()` actually uses LRU cache
- `max_resident_maps_observed` > 0 after fix
- Cache hit/miss statistics

**🔧 Solution:**
```cpp
TEST(QualityMapCache, ReadRegionUsesLRU) {
  QualityMapCache cache(4);  // max_resident_maps = 4
  
  // Load a frame into cache
  auto map = cache.read_cached(0);  // Should populate LRU
  
  // Read region should hit cache
  auto region = cache.read_region(0, Rect{0, 0, 100, 100});
  
  // Verify cache was used (not disk)
  EXPECT_TRUE(cache.was_last_read_from_cache());
}

TEST(QualityMapCache, ResidentMapsObserved) {
  QualityMapCache cache(4);
  
  // Read multiple frames
  for (int i = 0; i < 10; ++i) {
    cache.read_region(i, Rect{0, 0, 100, 100});
  }
  
  // Should have <= 4 resident maps
  EXPECT_GT(cache.get_max_resident_maps_observed(), 0);
  EXPECT_LE(cache.get_max_resident_maps_observed(), 4);
}
```

---

## 3. Item 4: Long-Term (CUDA/OpenCL + Binary)

### ✅ Strengths
- **Correctness constraint fixed:** Row-chunking instead of frame-batching
- **Backend arbitration added:** Priority order for `aqmh_reconstruction`
- **Rollout safety gate:** `gpu_reconstruction: disabled` default
- **Chunking strategy clarified:** GPU uses row-chunks (not whole canvas)

### ❌ Critical Issues

#### 3.1 `MAX_FRAMES` Undefined in CUDA Kernel (BLOCKER)
**Location:** Item 4.1.2  
**Problem:**
```cuda
float values[MAX_FRAMES];  // ❌ MAX_FRAMES not defined
float weights[MAX_FRAMES];
```
- CUDA requires **compile-time constants** for local arrays
- **645 frames × 8 bytes (2 arrays) = ~5 KB per thread** → **Local Memory Spill**

**Impact:** Kernel will not compile.  

**🔧 Solution:**

**Option A: Template Parameter (for small frame counts):**
```cuda
template<int MAX_FRAMES>
__global__ void weighted_mad_sigma_clip_kernel(...) {
  float values[MAX_FRAMES];
  float weights[MAX_FRAMES];
  // ...
}

// Call:
weighted_mad_sigma_clip_kernel<645><<<grid, block>>>(...);
```

**Option B: Shared Memory (RECOMMENDED for large frame counts):**
```cuda
__global__ void weighted_mad_sigma_clip_kernel(...) {
  extern __shared__ float shared_buffer[];
  float* values = shared_buffer;
  float* weights = shared_buffer + blockDim.x * blockDim.y * MAX_FRAMES_PER_BLOCK;
  
  // MAX_FRAMES_PER_BLOCK = 64 (fits in shared memory)
  // Process frames in batches of 64
  // ...
}

// Call:
size_t shared_mem_size = blockDim.x * blockDim.y * 64 * 2 * sizeof(float);
weighted_mad_sigma_clip_kernel<<<grid, block, shared_mem_size>>>(...);
```

#### 3.2 Local Memory Spill Performance Risk
**Location:** Item 4.1.2  
**Problem:**
- **~5 KB/thread local memory** for 645 frames → **spills to global memory**
- **Occupancy drops** → fewer concurrent thread blocks
- **Performance unclear** until prototyped

**Impact:** "Expected gain: 2-3x faster" **cannot be guaranteed** without testing.  

**🔧 Solution:**
| **Approach** | **Description** | **Pros** | **Cons** |
|--------------|-----------------|----------|----------|
| **A. Accept Spill** | Use local memory, accept performance hit | Simple | Lower occupancy |
| **B. Shared Memory** | Process frames in sub-groups (64 at a time) | No spill, good occupancy | Complex implementation |
| **C. Streaming** | Online weighted-median/MAD update | Minimal memory, scalable | Numerical stability? |
| **D. Template (64)** | Limit to 64 frames per kernel call | Simple, good performance | Multiple kernel launches |

**➡️ RECOMMENDATION:**
1. **Prototype first** with small dataset (10 frames, 64×64)
2. **Measure performance** with different approaches
3. **Choose based on results**

#### 3.3 Backend Priority Not Implemented (BLOCKER)
**Location:** Item 4.1.3  
**Problem:**
- **Priority order defined:** `cuda > opencv_opencl > cpu` (excluding `opencv_cuda`)
- **But:** No code showing **where/How** this is implemented

**Impact:** "auto" may still select `opencv_cuda` (which has no implementation) → **silent fallback to CPU**

**🔧 Solution:**
```cpp
// In src/core/acceleration.cpp:

// 1. Update phase_supports_backend():
bool phase_supports_backend(AccelerationPhase phase, AccelerationBackend backend) {
  switch (phase) {
    // ... existing cases ...
    case AccelerationPhase::aqmh_reconstruction:
      // Only cuda and opencv_opencl are supported (NOT opencv_cuda)
      return backend == AccelerationBackend::cuda ||
             backend == AccelerationBackend::opencv_opencl;
    // ...
  }
}

// 2. Update auto-resolution logic:
AccelerationBackend select_backend_for_phase(
    AccelerationPhase phase, 
    const std::vector<AccelerationBackend>& available_backends) {
  
  if (phase == AccelerationPhase::aqmh_reconstruction) {
    // Priority: cuda > opencv_opencl > cpu
    for (auto backend : {AccelerationBackend::cuda, 
                          AccelerationBackend::opencv_opencl, 
                          AccelerationBackend::cpu}) {
      if (std::find(available_backends.begin(), available_backends.end(), backend) != available_backends.end()) {
        return backend;
      }
    }
  }
  // ... existing logic ...
}
```

#### 3.4 Binary Format Size Calculation Incorrect (BLOCKER)
**Location:** Item 4.4.1  
**Problem:**
- **Claimed size:** ~1.5 MB
- **Actual calculation (2310×3924 canvas, block_size_px=6):**
  - Per-frame records: 645 × 50 bytes = **32.25 KB**
  - Block grid: ceil(3924/6)=654, ceil(2310/6)=385
  - Per array: 654 × 385 × 4 bytes = **~1.02 MB**
  - 6 arrays: 6 × 1.02 MB = **~6.12 MB**
  - **Total: ~6.15 MB** (NOT 1.5 MB)

**Impact:** Wrong documentation, potential memory budgeting issues.  

**🔧 Solution:**
1. **Correct size estimate:** **~6.15 MB** (for 2310×3924 canvas)
2. **Alternative:** Use larger `block_size_px`:
   - `block_size_px=32`: ceil(3924/32)=123, ceil(2310/32)=73
   - Per array: 123 × 73 × 4 = **~35.9 KB**
   - 6 arrays: ~215 KB
   - **Total: ~247 KB** (much smaller!)
3. **Clarify in plan:**
   - Is `block_size_px` for binary format the same as `r_morph_canvas_px` (6)?
   - **RECOMMENDATION:** **Yes** (consistent, simple)

#### 3.5 `block_grid_width`/`block_grid_height` Storage Unclear
**Location:** Item 4.4.1  
**Problem:**
- **Current:** Stores `block_grid_width` and `block_grid_height` in header
- **Alternative:** Can be **calculated** from `width`, `height`, `block_size_px`

**Impact:** Wastes 8 bytes in header.  

**🔧 Solution:**
```cpp
// In header:
struct BinaryDiagnosticsHeader {
  char magic[4] = {'A', 'Q', 'D', 'B'};
  uint32_t version = 1;
  uint32_t frame_count;
  uint32_t width;
  uint32_t height;
  uint32_t block_size_px;  // e.g., 6
  uint8_t has_heatmaps;
  // block_grid_width/height NOT stored (calculated on read)
};

// When reading:
uint32_t block_grid_width = (header.width + header.block_size_px - 1) / header.block_size_px;
uint32_t block_grid_height = (header.height + header.block_size_px - 1) / header.block_size_px;
```

#### 3.6 Frame Index in CUDA Kernel
**Location:** Item 4.1.2  
**Problem:**
```cuda
// Current:
const int idx = fi * chunk_rows * width + y * width + x;
if (frame_masks[idx] == 0) continue;
```
- **Unclear:** Does `frame_masks` have the same layout as `frames`/`q_maps`?
- **Assumption:** `[frame_count][chunk_rows][width]` (flattened, row-major)

**🔧 Solution:**
**Document layout explicitly in plan:**
```
// Memory layout (row-major, flattened):
// - frames:     [frame_count][chunk_rows][width]
// - q_maps:     [frame_count][chunk_rows][width]
// - frame_masks: [frame_count][chunk_rows][width] (uint8)
// - canvas_mask: [chunk_rows][width] (uint8)
// - global_weights: [frame_count] (float32)
```

#### 3.7 Missing Tests for Backend Arbitration
**Location:** Item 4.7  
**Problem:** No tests for:
- `acceleration_backend: auto` selects `cuda` when available
- `acceleration_backend: opencv_opencl` selects OpenCL
- `opencv_cuda` is **never** selected for `aqmh_reconstruction`

**🔧 Solution:**
```cpp
TEST(AccelerationBackend, AqmhReconstructionAutoSelectsCuda) {
  // Mock: CUDA and OpenCL available
  auto available = {AccelerationBackend::cuda, AccelerationBackend::opencv_opencl};
  auto selected = select_backend_for_phase(
      AccelerationPhase::aqmh_reconstruction, available);
  EXPECT_EQ(selected, AccelerationBackend::cuda);
}

TEST(AccelerationBackend, AqmhReconstructionExcludesOpencvCuda) {
  // Mock: Only opencv_cuda available
  auto available = {AccelerationBackend::opencv_cuda};
  auto selected = select_backend_for_phase(
      AccelerationPhase::aqmh_reconstruction, available);
  // Should fall back to CPU (opencv_cuda not supported)
  EXPECT_EQ(selected, AccelerationBackend::cpu);
}
```

#### 3.8 Missing Row-Chunking Correctness Test
**Location:** Item 4.7  
**Problem:** No test for:
- GPU kernel with **≥2 row-chunks** produces same result as single-chunk
- Verifies **no partial-frame statistics** bug

**🔧 Solution:**
```cpp
TEST(AqmhReconstructionCuda, RowChunkingCorrectness) {
  // Force multiple chunks by limiting device memory
  AqmhReconstructionConfig cfg;
  cfg.chunk_rows = 0;  // Auto
  cfg.memory_budget_mb = 1;  // Very small → many chunks
  
  auto result_gpu = reconstruct_aqmh_weighted_cuda(..., cfg);
  auto result_cpu = reconstruct_aqmh_weighted(..., cfg);  // CPU (single-chunk)
  
  // Compare with tolerance
  EXPECT_TRUE(compare_results(result_gpu, result_cpu, 1e-5f));
}
```

---

## 📊 Test Coverage Analysis

### ✅ Existing Tests (Good)
- Basic config parsing/validation
- Phase skip tests for diagnostics
- Chunk rows calculation tests

### ❌ Missing Tests (Critical)

| **Test** | **Location** | **Priority** | **Status** |
|----------|--------------|--------------|------------|
| Config flag combination tests | Item 2.1.8 | 🔴 HIGH | Missing |
| Cache fix regression tests | Item 3.5 | 🔴 HIGH | Missing |
| Backend arbitration tests | Item 4.7 | 🔴 HIGH | Missing |
| Row-chunking correctness tests | Item 4.7 | 🔴 HIGH | Missing |
| Determinism tests (multi-dataset) | Item 4.7 | 🟡 MEDIUM | Partially covered |
| Prefetch coordinator tests | Item 3.5 | 🟡 MEDIUM | Missing |
| Binary format round-trip tests | Item 4.7 | 🟡 MEDIUM | Missing |

---

## 🎯 Implementation Recommendations

---

### Phase 1: Fix Critical Blockers (1–2 days)
**Must be completed before any implementation can begin.**

| **Task** | **Priority** | **Owner** | **Estimate** | **Dependencies** |
|----------|--------------|-----------|--------------|------------------|
| Add `memory_budget_mb` to `AqmhReconstructionConfig` | 🔴 | Plan Author | 1h | None |
| Define `denom` in auto formula | 🔴 | Plan Author | 1h | None |
| Define `MAX_FRAMES` in CUDA kernel | 🔴 | Plan Author | 2h | None |
| Implement backend priority in `acceleration.cpp` | 🔴 | Plan Author | 2h | None |
| Correct binary format size calculation | 🔴 | Plan Author | 1h | None |
| Detail `read_region()` routing fix | 🔴 | Plan Author | 2h | None |

---

### Phase 2: Address High-Priority Issues (1–2 days)
**Should be completed before implementation of respective items.**

| **Task** | **Priority** | **Owner** | **Estimate** | **Dependencies** |
|----------|--------------|-----------|--------------|------------------|
| Clarify thread management in `AqmhPrefetchCoordinator` | 🟡 | Plan Author | 2h | Phase 1 |
| Detail Local Memory Spill solution | 🟡 | Plan Author | 4h | Phase 1 |
| Add missing tests for config flags | 🟡 | Developer | 4h | Phase 1 |
| Add missing tests for backend arbitration | 🟡 | Developer | 4h | Phase 1 |
| Clarify memory layout for GPU kernels | 🟡 | Plan Author | 1h | Phase 1 |
| Decide on `block_grid_width`/`height` storage | 🟡 | Plan Author | 1h | Phase 1 |

---

### Phase 3: Implementation Order (Updated)

#### 1. Item 2: Config Changes (3–5 days)
1. `configuration.hpp` — Add all new fields
2. `config.cpp` — Parsing, serialization, validation
3. `runner_phase_aqmh_diagnostics.cpp` — Respect config flags
4. `aqmh_reconstruction.cpp` + `.hpp` — `chunk_rows` support
5. `runner_phase_aqmh_reconstruction.cpp` — `chunk_rows` passthrough
6. **Fix critical issues from Phase 1** (especially `memory_budget_mb`)
7. All YAML files + schema (use script!)
8. Tests (including new flag combination tests)
9. Documentation updates

#### 2. Item 3: Code Changes (5–10 days)
1. **Item 3.0: Profiling first** (2–3 days)
   - Profile `AQMH_MAPS` phase
   - Identify bottleneck (I/O vs. GPU vs. CPU)
   - **Result determines priority of 3.1 and 3.2**
2. **Item 3.2: Cache Fix** (2–3 days)
   - Implement `read_region()` routing through LRU
   - Add tests for cache hits/misses
   - **Verify `max_resident_maps_observed` > 0**
   - **Then:** Bump `max_resident_maps` default to 4
3. **Item 3.1: Prefetch Overlap** (3–5 days)
   - Implement `AqmhPrefetchCoordinator` (with thread management from Phase 2)
   - Integrate into pipeline
   - Add determinism tests (output must be byte-identical)
4. Config file updates
5. Documentation updates

#### 3. Item 4: Long-Term (4–6 weeks)
1. **Backend arbitration** (1 day)
   - Implement priority order in `acceleration.cpp`
   - Add tests
2. **CUDA kernel** (2–3 weeks)
   - **First:** Prototype with small dataset
   - **Resolve Local Memory Spill** (from Phase 2)
   - Implement row-chunked kernel
   - Add determinism tests
3. **OpenCL kernel** (1–2 weeks)
   - Mirror CUDA implementation
   - Use `cv::UMat` (consistent with existing code)
4. **CMakeLists.txt** updates
5. **Backend dispatch** in runner
6. **Rollout safety gate** (`gpu_reconstruction: disabled`)
7. **Binary format** (2–3 days)
   - Implement read/write
   - Add round-trip tests
8. **Final determinism validation** (1 week)
   - Test across multiple real datasets
   - **Only then:** Set `gpu_reconstruction` default to `auto`

---

## 📝 Checklist for Plan Finalization

### 🔴 Critical (Must Complete)
- [ ] `memory_budget_mb` added to `AqmhReconstructionConfig`
- [ ] `denom` defined in auto formula
- [ ] `MAX_FRAMES` defined in CUDA kernel (template or shared memory)
- [ ] Backend priority implemented in `acceleration.cpp`
- [ ] Binary format size calculation corrected
- [ ] `read_region()` routing fix detailed

### 🟡 High Priority (Should Complete)
- [ ] Thread management in `AqmhPrefetchCoordinator` clarified
- [ ] Local Memory Spill solution selected and documented
- [ ] Memory layout for GPU kernels documented
- [ ] `block_grid_width`/`height` storage decision made
- [ ] Missing tests added (config flags, backend arbitration, row-chunking)

### 🟢 Nice to Have (Could Improve)
- [ ] YAML update script created
- [ ] Determinism tolerance defined (1e-5 relative, 1e-3 absolute)
- [ ] `wait_all_prefetched()` improved (lazy loading or timeout)
- [ ] CUDA kernel prototyped
- [ ] OpenCL API decision made (`cv::UMat` vs `cl::Buffer`)

---

## 📊 Final Assessment

### What Changed Since First Version?

**✅ Fixed:**
1. **Overlap Architecture:** Now correctly implements **I/O-only prefetch** (not compute overlap)
2. **GPU Chunking:** Now uses **row-chunking** (not frame-batching) → **correctness guaranteed**
3. **Cache Bug:** Root cause identified (routing bug, not missing feature)
4. **Backend Arbitration:** Priority order defined
5. **Rollout Safety:** `gpu_reconstruction: disabled` default
6. **New Item:** AQMH_MAPS profiling (3.0) added

**❌ Still Broken:**
1. **`memory_budget_mb` missing** → Code won't compile
2. **`MAX_FRAMES` undefined** → CUDA kernel won't compile
3. **Backend priority not implemented** → Wrong backend may be selected
4. **Binary format size wrong** → Documentation error
5. **Technical details missing** → Implementation risk

### Bottom Line

> **🟡 GOOD BUT NOT YET IMPLEMENTABLE**
> 
> The revised plan has **excellent design decisions** and **fixes all major architectural flaws** from the first version. However, **several technical details are missing or incorrect** that would prevent compilation or cause runtime errors.
> 
> **After fixing the 🔴 critical issues (1–2 days of work), the plan will be ready for implementation.**

---

## 📚 Appendices

---

### Appendix A: Recommended CUDA Kernel Structure (Shared Memory)

```cuda
// aqmh_reconstruction_cuda.cu

// Sub-group size for shared memory (64 frames fits in shared memory)
constexpr int SUB_GROUP_FRAMES = 64;

__global__ void weighted_mad_sigma_clip_kernel(
    const float* __restrict__ frames,
    const float* __restrict__ q_maps,
    const uint8_t* __restrict__ canvas_mask,
    const uint8_t* __restrict__ frame_masks,
    const float* __restrict__ global_weights,
    float* __restrict__ output,
    float* __restrict__ weight_sum,
    float* __restrict__ uniform_control,
    int width, int chunk_rows, int frame_count,
    // ... other parameters ...
) {
    extern __shared__ float shared_buffer[];
    
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= chunk_rows) return;
    if (canvas_mask[y * width + x] == 0) return;
    
    // Shared memory for this block
    float* s_values = shared_buffer;
    float* s_weights = shared_buffer + blockDim.x * blockDim.y * SUB_GROUP_FRAMES;
    
    // Process frames in sub-groups
    float running_values[SUB_GROUP_FRAMES];
    float running_weights[SUB_GROUP_FRAMES];
    int running_n_samples = 0;
    
    for (int group_start = 0; group_start < frame_count; group_start += SUB_GROUP_FRAMES) {
        int group_size = std::min(SUB_GROUP_FRAMES, frame_count - group_start);
        
        // Load sub-group into shared memory (cooperative)
        for (int fi_local = 0; fi_local < group_size; ++fi_local) {
            int fi = group_start + fi_local;
            const int idx = fi * chunk_rows * width + y * width + x;
            
            if (frame_masks[idx] == 0) continue;
            const float v = frames[idx];
            const float q = q_maps[idx];
            
            if (!isfinite(v) || !isfinite(q)) continue;
            
            const float gw = global_weights[fi];
            const float score = gw * max(0.0f, q);
            const float w = score > 0.0f ? score : 0.0f;
            
            if (w > 0.0f) {
                // Store in shared memory via threadIdx
                int store_idx = threadIdx.x * SUB_GROUP_FRAMES + fi_local;
                s_values[store_idx] = v;
                s_weights[store_idx] = w;
            } else {
                s_values[threadIdx.x * SUB_GROUP_FRAMES + fi_local] = 0.0f;
                s_weights[threadIdx.x * SUB_GROUP_FRAMES + fi_local] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Process this sub-group (all threads)
        // ... weighted median/MAD for this sub-group ...
        
        // Update running statistics
        // ...
    }
    
    // Write final result
    output[y * width + x] = weighted_mean;
    weight_sum[y * width + x] = total_weight;
    if (compute_uniform_control) {
        uniform_control[y * width + x] = control_mean;
    }
}

// Kernel launch:
size_t shared_mem_size = blockDim.x * blockDim.y * SUB_GROUP_FRAMES * 2 * sizeof(float);
weighted_mad_sigma_clip_kernel<<<grid, block, shared_mem_size>>>(...);
```

---

### Appendix B: YAML Update Script

```python
#!/usr/bin/env python3
# tools/update_aqmh_configs.py

import yaml
import glob
from pathlib import Path

# New fields for AQMH v0.2.0
NEW_AQMH_FIELDS = {
    "diagnostics": {
        "enabled": True,
        "level": "full",
        "per_frame_blocks": True,
        "heatmaps": True,
        "regions": True,
        "format": "json",
        "tau_artifact": 0.2,
        "q_region": 0.75,
        "r_morph_canvas_px": 6,
    },
    "reconstruction": {
        "clip_sigma": 3.0,
        "clip_iterations": 3,
        "min_fraction": 0.5,
        "min_n_eff": 2.0,
        "chunk_rows": 0,
        "gpu_reconstruction": "disabled",
    },
    "storage": {
        "max_resident_maps": 4,
    }
}

def update_yaml(filepath):
    with open(filepath, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except:
            config = {}
    
    if "aqmh" not in config:
        return False
    
    aqmh = config["aqmh"]
    modified = False
    
    for section, fields in NEW_AQMH_FIELDS.items():
        if section not in aqmh:
            aqmh[section] = {}
            modified = True
        for key, default_value in fields.items():
            if key not in aqmh[section]:
                aqmh[section][key] = default_value
                modified = True
    
    if modified:
        with open(filepath, 'w') as f:
            yaml.dump(config, f, sort_keys=False, default_flow_style=False)
        print(f"Updated: {filepath}")
    return modified

if __name__ == "__main__":
    import sys
    
    updated_count = 0
    for filepath in glob.glob("**/*.yaml", recursive=True):
        if update_yaml(filepath):
            updated_count += 1
    
    print(f"\nTotal files updated: {updated_count}")
    sys.exit(0 if updated_count > 0 else 1)
```

---

### Appendix C: Corrected Binary Format Size Calculation

```
For a canvas of 2310×3924 pixels with block_size_px = 6:

1. Per-Frame Records:
   - 645 frames × 50 bytes/frame = 32,250 bytes = ~32 KB

2. Block Arrays (if has_heatmaps = true):
   - block_grid_width = ceil(3924 / 6) = 654
   - block_grid_height = ceil(2310 / 6) = 385
   - Per array: 654 × 385 × 4 bytes = 1,019,820 bytes = ~1.02 MB
   - 6 arrays (aqmh_q_median, aqmh_q_p10, aqmh_q_p90, aqmh_artifact_frac, q_map_heatmap, artifact_heatmap)
   - Total: 6 × 1,019,820 = 6,118,920 bytes = ~6.12 MB

3. Total Size:
   - Per-frame + Block arrays = ~32 KB + ~6.12 MB = ~6.15 MB
   - (NOT 1.5 MB as previously stated)

For a canvas of 2310×3924 pixels with block_size_px = 32:

1. Per-Frame Records: Same = ~32 KB

2. Block Arrays:
   - block_grid_width = ceil(3924 / 32) = 123
   - block_grid_height = ceil(2310 / 32) = 73
   - Per array: 123 × 73 × 4 = 35,916 bytes = ~35.9 KB
   - 6 arrays: 6 × 35,916 = 215,496 bytes = ~215 KB

3. Total Size:
   - Per-frame + Block arrays = ~32 KB + ~215 KB = ~247 KB
```

---

### Appendix D: Backend Priority Implementation

```cpp
// In src/core/acceleration.cpp

// 1. Define phase-specific backend priorities
std::map<AccelerationPhase, std::vector<AccelerationBackend>> PHASE_BACKEND_PRIORITY = {
    {AccelerationPhase::aqmh_reconstruction, {
        AccelerationBackend::cuda,           // Highest priority
        AccelerationBackend::opencv_opencl,  // Second priority
        AccelerationBackend::cpu             // Fallback
        // Note: opencv_cuda is EXCLUDED for this phase
    }},
    // ... other phases ...
};

// 2. Update select_backend_for_phase()
AccelerationBackend select_backend_for_phase(
    AccelerationPhase phase,
    const std::vector<AccelerationBackend>& available_backends) {
    
    // Check if phase has custom priority
    if (PHASE_BACKEND_PRIORITY.count(phase)) {
        const auto& priority = PHASE_BACKEND_PRIORITY[phase];
        for (auto backend : priority) {
            if (std::find(available_backends.begin(), available_backends.end(), backend) 
                != available_backends.end()) {
                return backend;
            }
        }
    }
    
    // Default behavior for other phases
    // ... existing logic ...
}

// 3. Update phase_supports_backend() to exclude opencv_cuda for aqmh_reconstruction
bool phase_supports_backend(AccelerationPhase phase, AccelerationBackend backend) {
    switch (phase) {
        case AccelerationPhase::aqmh_reconstruction:
            // Only cuda and opencv_opencl are supported
            return backend == AccelerationBackend::cuda ||
                   backend == AccelerationBackend::opencv_opencl;
        // ... other phases ...
    }
}
```

---

*End of Analysis Document*
