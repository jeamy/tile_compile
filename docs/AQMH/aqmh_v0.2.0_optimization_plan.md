# AQMH v0.2.0 — Optimization Implementation Plan

**Source:** `docs/AQMH/aqmh_v0.2.0_performance_analysis.md` Section 7, Items 2–4 (including §6.7's AQMH_MAPS bottleneck, now tracked as §3.0 below)
**Goal:** Detailed implementation plan for short-term (config), medium-term (code — including the AQMH_MAPS regression, the biggest single contributor at 41% of extra time), and long-term (CUDA/OpenCL + binary diagnostics) optimizations.

**Revision note:** this revision fixes several issues found in review: (1) added §3.0 covering the previously-untracked `AQMH_MAPS` 2.2x regression (650s/41% of the gap — the largest single item, previously not addressed by any concrete task); (2) replaced the CUDA/OpenCL kernel design's frame-axis batching with row-axis chunking (§4.1.2/§4.2.2) after identifying that frame-batching would silently compute weighted-MAD statistics from a subset of frames — a correctness bug, not a performance detail; (3) resolved the direct contradiction between §4.1.2 (GPU must batch due to ~23GB memory) and the old §4.3 (GPU should use no chunking at all); (4) narrowed the §3.1 overlap design and its queue API to match the "Option C" I/O-only overlap it settled on, and revised its expected-gain estimate down accordingly; (5) corrected §3.2's description of the quality-map cache bug to the actual confirmed root cause (a routing bug bypassing an existing LRU, not a missing feature); (6) added explicit GPU-backend arbitration rules and a `gpu_reconstruction` rollout safety gate, both previously unaddressed gaps.

---

## Item 2: Short-Term (Config Only)

### 2.1 Make `AQMH_DIAGNOSTICS` Configurable

#### 2.1.1 Config Struct Changes

**File:** `tile_compile_cpp/include/tile_compile/config/configuration.hpp:273-277`

Current `AqmhDiagnosticsConfig`:
```cpp
struct AqmhDiagnosticsConfig {
  float tau_artifact = 0.20f;
  float q_region = 0.75f;
  int r_morph_canvas_px = 6;
};
```

New:
```cpp
struct AqmhDiagnosticsConfig {
  bool enabled = true;                    // master switch
  std::string level = "full";             // "none" | "summary" | "full"
  bool per_frame_blocks = true;           // per-frame block-level diagnostics + heatmaps
  bool heatmaps = true;                   // spatial heatmap arrays
  bool regions = true;                    // region extraction (aqmh_regions.json)
  std::string format = "json";            // "json" | "binary"
  int binary_block_size_px = 0;            // 0 = use r_morph_canvas_px
  float tau_artifact = 0.20f;
  float q_region = 0.75f;
  int r_morph_canvas_px = 6;
};
```

Behavior:
- `enabled: false` → skip `AQMH_DIAGNOSTICS` phase entirely
- `level: "none"` → same as `enabled: false`
- `level: "summary"` → write only run-level cherry-pick fields + lightweight `diagnostics` array (~0.37 MB); skip per-frame `frames` block
- `level: "full"` → current behavior (178 MB + 100 MB)
- `per_frame_blocks: false` → skip `compute_block_diagnostics()` for all frames (**only effective when `level == "full"`**)
- `heatmaps: false` → omit `q_map_heatmap` and `artifact_heatmap` arrays from block diagnostics (**only effective when `level == "full"`**)
- `regions: false` → skip region extraction, do not write `aqmh_regions.json`

**Note on `enabled`/`level` redundancy:** Both `enabled: false` and `level: "none"` disable the phase. The validation step below normalizes them to be consistent: if `enabled == false`, set `level = "none"`; if `level == "none"`, set `enabled = false`. This avoids contradictory states (`enabled: true` + `level: "none"`) without removing either field, since removing `enabled` would be a breaking change for existing configs. The sub-flags `per_frame_blocks`, `heatmaps`, and `regions` are **only meaningful when `level == "full"`** — they are silently ignored for `level: "summary"` and `level: "none"`.

#### 2.1.2 Reconstruction Chunk Size Config

**File:** `tile_compile_cpp/include/tile_compile/config/configuration.hpp:260-265`

Current `AqmhReconstructionConfig`:
```cpp
struct AqmhReconstructionConfig {
  float clip_sigma = 3.0f;
  int clip_iterations = 3;
  float min_fraction = 0.5f;
  float min_n_eff = 2.0f;
};
```

New:
```cpp
struct AqmhReconstructionConfig {
  float clip_sigma = 3.0f;
  int clip_iterations = 3;
  float min_fraction = 0.5f;
  float min_n_eff = 2.0f;
  int chunk_rows = 0;                 // 0 = backend-specific auto sizing, >0 = explicit override
  size_t memory_budget_mb = 0;        // 0 = use global config (passed in from AqmhConfig at callsite)
  std::string gpu_reconstruction = "disabled";  // "disabled" | "auto" | "force"
};
```

**Note (two `AqmhReconstructionConfig` structs):** `configuration.hpp` holds the *user-facing* config; `include/tile_compile/reconstruction/aqmh_reconstruction.hpp:14-28` holds the *internal* `reconstruction::AqmhReconstructionConfig` used by `aqmh_reconstruction.cpp`. Both must gain `chunk_rows` and `memory_budget_mb`. The bridge is in `runner_phase_aqmh_reconstruction.cpp`:
```cpp
aqmh_recon_cfg.chunk_rows        = cfg.aqmh.reconstruction.chunk_rows;
aqmh_recon_cfg.memory_budget_mb  = cfg.aqmh.reconstruction.memory_budget_mb != 0
    ? cfg.aqmh.reconstruction.memory_budget_mb
    : static_cast<size_t>(cfg.memory_budget_mb);  // fall back to global budget
```

#### 2.1.3 Config Parsing (`src/io/config.cpp`)

**File:** `tile_compile_cpp/src/io/config.cpp`

**Parsing (around line 673):**
```cpp
if (a["diagnostics"]) {
  auto d = a["diagnostics"];
  if (d["tau_artifact"])
    cfg.aqmh.diagnostics.tau_artifact = d["tau_artifact"].as<float>();
  if (d["q_region"])
    cfg.aqmh.diagnostics.q_region = d["q_region"].as<float>();
  if (d["r_morph_canvas_px"])
    cfg.aqmh.diagnostics.r_morph_canvas_px = d["r_morph_canvas_px"].as<int>();
  // NEW:
  if (d["enabled"])
    cfg.aqmh.diagnostics.enabled = d["enabled"].as<bool>();
  if (d["level"])
    cfg.aqmh.diagnostics.level = d["level"].as<std::string>();
  if (d["per_frame_blocks"])
    cfg.aqmh.diagnostics.per_frame_blocks = d["per_frame_blocks"].as<bool>();
  if (d["heatmaps"])
    cfg.aqmh.diagnostics.heatmaps = d["heatmaps"].as<bool>();
  if (d["regions"])
    cfg.aqmh.diagnostics.regions = d["regions"].as<bool>();
  if (d["format"])
    cfg.aqmh.diagnostics.format = d["format"].as<std::string>();
  if (d["binary_block_size_px"])
    cfg.aqmh.diagnostics.binary_block_size_px = d["binary_block_size_px"].as<int>();
}
if (a["reconstruction"]) {
  auto r = a["reconstruction"];
  // ... existing fields ...
  // NEW:
  if (r["chunk_rows"])
    cfg.aqmh.reconstruction.chunk_rows = r["chunk_rows"].as<int>();
  if (r["memory_budget_mb"])
    cfg.aqmh.reconstruction.memory_budget_mb = r["memory_budget_mb"].as<size_t>();
  if (r["gpu_reconstruction"])
    cfg.aqmh.reconstruction.gpu_reconstruction = r["gpu_reconstruction"].as<std::string>();
}
```

**Serialization (around line 1312):**
```cpp
node["aqmh"]["diagnostics"]["tau_artifact"] = aqmh.diagnostics.tau_artifact;
node["aqmh"]["diagnostics"]["q_region"] = aqmh.diagnostics.q_region;
node["aqmh"]["diagnostics"]["r_morph_canvas_px"] = aqmh.diagnostics.r_morph_canvas_px;
// NEW:
node["aqmh"]["diagnostics"]["enabled"] = aqmh.diagnostics.enabled;
node["aqmh"]["diagnostics"]["level"] = aqmh.diagnostics.level;
node["aqmh"]["diagnostics"]["per_frame_blocks"] = aqmh.diagnostics.per_frame_blocks;
node["aqmh"]["diagnostics"]["heatmaps"] = aqmh.diagnostics.heatmaps;
node["aqmh"]["diagnostics"]["regions"] = aqmh.diagnostics.regions;
node["aqmh"]["diagnostics"]["format"] = aqmh.diagnostics.format;
node["aqmh"]["diagnostics"]["binary_block_size_px"] = aqmh.diagnostics.binary_block_size_px;
// ...
node["aqmh"]["reconstruction"]["chunk_rows"] = aqmh.reconstruction.chunk_rows;
node["aqmh"]["reconstruction"]["memory_budget_mb"] = aqmh.reconstruction.memory_budget_mb;
node["aqmh"]["reconstruction"]["gpu_reconstruction"] = aqmh.reconstruction.gpu_reconstruction;
```

**Validation (around line 1875):**
```cpp
if (!is_between_0_1(aqmh.diagnostics.tau_artifact)) { ... }
if (!is_between_0_1(aqmh.diagnostics.q_region)) { ... }
if (aqmh.diagnostics.r_morph_canvas_px < 1) { ... }
// NEW:
if (aqmh.diagnostics.level != "none" &&
    aqmh.diagnostics.level != "summary" &&
    aqmh.diagnostics.level != "full") {
  throw ValidationError("aqmh.diagnostics.level must be none, summary, or full");
}
// Normalize enabled/level to be consistent (see §2.1.1 note on redundancy):
if (!aqmh.diagnostics.enabled) aqmh.diagnostics.level = "none";
if (aqmh.diagnostics.level == "none") aqmh.diagnostics.enabled = false;
if (aqmh.diagnostics.format != "json" && aqmh.diagnostics.format != "binary") {
  throw ValidationError("aqmh.diagnostics.format must be json or binary");
}
if (aqmh.diagnostics.binary_block_size_px < 0) {
  throw ValidationError("aqmh.diagnostics.binary_block_size_px must be >= 0");
}
if (aqmh.reconstruction.chunk_rows < 0) {
  throw ValidationError("aqmh.reconstruction.chunk_rows must be >= 0");
}
if (aqmh.reconstruction.gpu_reconstruction != "disabled" &&
    aqmh.reconstruction.gpu_reconstruction != "auto" &&
    aqmh.reconstruction.gpu_reconstruction != "force") {
  throw ValidationError("aqmh.reconstruction.gpu_reconstruction must be disabled, auto, or force");
}
```

**Schema JSON (around line 2368):**
```json
"diagnostics":{"type":"object","properties":{
  "enabled":{"type":"boolean"},
  "level":{"type":"string","enum":["none","summary","full"]},
  "per_frame_blocks":{"type":"boolean"},
  "heatmaps":{"type":"boolean"},
  "regions":{"type":"boolean"},
  "format":{"type":"string","enum":["json","binary"]},
  "binary_block_size_px":{"type":"integer","minimum":0},
  "tau_artifact":{"type":"number","minimum":0,"maximum":1},
  "q_region":{"type":"number","minimum":0,"maximum":1},
  "r_morph_canvas_px":{"type":"integer","minimum":1}
}}
```
Add to `reconstruction` properties:
```json
"chunk_rows":{"type":"integer","minimum":0},
"memory_budget_mb":{"type":"integer","minimum":0},
"gpu_reconstruction":{"type":"string","enum":["disabled","auto","force"]}
```

#### 2.1.4 Diagnostics Phase — Respect Config Flags

**File:** `tile_compile_cpp/apps/runner_phase_aqmh_diagnostics.cpp`

In `run_phase_aqmh_diagnostics()`:

1. **Early exit when disabled:**
```cpp
if (!cfg.aqmh.diagnostics.enabled || cfg.aqmh.diagnostics.level == "none") {
  emitter.phase_start(run_id, Phase::AQMH_DIAGNOSTICS, "AQMH_DIAGNOSTICS", log_file);
  emitter.phase_end(run_id, Phase::AQMH_DIAGNOSTICS, "skipped",
                    {{"reason", "disabled_by_config"}}, log_file);
  return true;
}
```

2. **Summary mode — skip per-frame blocks:**
```cpp
const bool full_mode = cfg.aqmh.diagnostics.level == "full";
if (full_mode && cfg.aqmh.diagnostics.per_frame_blocks && q_map_cache && ...) {
  // existing block diagnostics loop
}
```

3. **Heatmaps flag — omit heatmap arrays:**
In `compute_block_diagnostics()`, add `bool emit_heatmaps` param. When false, skip `heatmap_arr` and `art_heat_arr` construction.

4. **Regions flag — skip region extraction:**
Region extraction code (in `runner_phase_aqmh_diagnostics.cpp` or `runner_aqmh_pipeline.cpp`) must check `cfg.aqmh.diagnostics.regions` before writing `aqmh_regions.json`.

#### 2.1.5 Reconstruction Phase — Respect `chunk_rows`

**File:** `tile_compile_cpp/apps/runner_phase_aqmh_reconstruction.cpp:56-68`

```cpp
aqmh_recon_cfg.chunk_rows = cfg.aqmh.reconstruction.chunk_rows;  // 0 = auto
```

**File:** `tile_compile_cpp/src/reconstruction/aqmh_reconstruction.cpp:159-164`

Current:
```cpp
const size_t target_mb = static_cast<size_t>(std::clamp(
    cfg.memory_budget_mb / 2, 128, 1536));
```

New:
```cpp
int chunk_rows;
if (cfg.chunk_rows > 0) {
  chunk_rows = std::min(height, cfg.chunk_rows);
} else {
  // Global default remains 512 MB. aqmh.reconstruction.memory_budget_mb == 0
  // means "use global memory_budget_mb"; if that is also absent, use 512 MB.
  const size_t budget_mb = cfg.memory_budget_mb > 0 ? cfg.memory_budget_mb : 512u;
  const size_t target_mb = static_cast<size_t>(std::clamp(
      static_cast<int>(budget_mb) / 2, 128, 1536));
  const size_t target_bytes = target_mb * 1024u * 1024u;
  // bytes per row-chunk: one frame slice + one Q-map slice, for all frames
  // layout: [frame_count][chunk_rows][width] × float32, times 2 (frames + q_maps)
  const size_t bytes_per_row_all_frames =
      static_cast<size_t>(width) * frame_count * sizeof(float) * 2;
  const size_t denom = std::max<size_t>(1, bytes_per_row_all_frames);
  chunk_rows = std::max(1, std::min(height, static_cast<int>(target_bytes / denom)));
}
```

Also add `chunk_rows` and `memory_budget_mb` to `reconstruction::AqmhReconstructionConfig` in `aqmh_reconstruction.hpp:14-28` (the *internal* config struct, distinct from the user-facing one in `configuration.hpp` — see §2.1.2 note).

#### 2.1.6 Config File Updates

**`tile_compile_cpp/tile_compile.yaml` (default config, line 301-304):**
```yaml
  diagnostics:
    enabled: true
    level: summary
    per_frame_blocks: true
    heatmaps: false
    regions: true
    format: json
    binary_block_size_px: 0   # 0 = use r_morph_canvas_px
    tau_artifact: 0.2
    q_region: 0.75
    r_morph_canvas_px: 6
```
Add to `aqmh.reconstruction` section (after line 304, before `stacking:`):
```yaml
  reconstruction:
    clip_sigma: 3.0
    clip_iterations: 3
    min_fraction: 0.5
    min_n_eff: 2.0
    chunk_rows: 0          # 0 = backend-specific auto sizing
    memory_budget_mb: 0    # 0 = use global memory_budget_mb
    gpu_reconstruction: disabled   # auto | force | disabled
```

**`tile_compile_cpp/examples/aqmh_tuning.example.yaml` (line 110-113):**
```yaml
  diagnostics:
    enabled: true
    level: summary        # none | summary | full
    per_frame_blocks: false
    heatmaps: false
    regions: false
    format: json
    q_region: 0.75
    r_morph_canvas_px: 6
    tau_artifact: 0.2
```
Add `reconstruction` section with `chunk_rows: 0`, `memory_budget_mb: 0`, and `gpu_reconstruction: disabled`.

**`tile_compile_cpp/examples/M42.global_medium.yaml` (line 284-287):**
```yaml
  diagnostics:
    enabled: true
    level: summary
    per_frame_blocks: true
    heatmaps: false
    regions: true
    format: json
    binary_block_size_px: 0   # 0 = use r_morph_canvas_px
    tau_artifact: 0.20
    q_region: 0.75
    r_morph_canvas_px: 6
```
Add `reconstruction` section with `chunk_rows: 0`, `memory_budget_mb: 0`, and `gpu_reconstruction: disabled`.

**All other example YAMLs** — add `diagnostics.enabled: true`, `diagnostics.level: summary`, `diagnostics.format: json`, `diagnostics.binary_block_size_px: 0`, `reconstruction.chunk_rows: 0`, `reconstruction.memory_budget_mb: 0`, and `reconstruction.gpu_reconstruction: disabled` where `aqmh:` section exists. Files to update:
- `examples/M45_high_altitude_strong_rotation.example.yaml`
- `examples/bright_star.example.yaml`
- `examples/canon_equatorial_balanced.example.yaml`
- `examples/canon_low_n_high_quality.example.yaml`
- `examples/emergency_mode.example.yaml`
- `examples/full_mode.example.yaml`
- `examples/ic434.example.yaml`
- `examples/ic434_background_gradient.example.yaml`
- `examples/m104.example.yaml`
- `examples/m31_background_gradient_balanced.example.yaml`
- `examples/m66_galaxy_background_balanced.example.yaml`
- `examples/mono_full_mode.example.yaml`
- `examples/mono_small_n_anti_grid.example.yaml`
- `examples/mono_small_n_ultra_conservative.example.yaml`
- `examples/reduced_mode.example.yaml`
- `examples/smart_telescope_dwarf_seestar.example.yaml`
- `examples/smart_telescope_very_bright_star.example.yaml`
- `examples/very_bright_star_anti_seam.example.yaml`

**`tile_compile_cpp/tile_compile.schema.json`** — update diagnostics and reconstruction properties to match schema JSON changes above.

#### 2.1.7 Documentation Updates

- `docs/AQMH/aqmh_v0.2.0_performance_analysis.md` Section 6.2 — update YAML example to match final config struct
- `docs/configuration_examples_practical_de.md` — add diagnostics config examples
- `README.md` / `README_de.md` — mention new diagnostics flags in config reference

#### 2.1.8 Tests

**File:** `tile_compile_cpp/tests/test_aqmh_validation.cpp` (or new test file)

- Test that `level: "none"` skips the diagnostics phase (check phase_end status = "skipped")
- Test that `level: "summary"` writes `aqmh_metrics.json` without `frames` array
- Test that `level: "full"` writes complete diagnostics
- Test that `chunk_rows: 100` produces exactly `ceil(height / 100)` chunks
- Test that `chunk_rows: 0` uses the auto formula

---

## Item 3: Medium-Term (Code Changes)

### 3.0 Investigate and Reduce `AQMH_MAPS` Phase Cost (Highest-Priority Item in This Section)

**Why this is here:** Per `aqmh_v0.2.0_performance_analysis.md` §5, the `AQMH_MAPS` phase becoming 2.2x slower (565→1215 s) is the **single largest** contributor to the v0.2.0 regression — 650.3 s / 41% of the extra time, larger than the CPU-reconstruction fallback (35%) or diagnostics (20%). Section 6.7 of the analysis names this as a required investigation with a concrete "quick win" (parallelize per-frame write-back / async writes), but earlier drafts of this plan only addressed it indirectly via the overlap in §3.1, which hides part of this latency behind reconstruction without reducing it. This subsection makes it a first-class item.

**File:** `tile_compile_cpp/src/metrics/aqmh_quality_map.cpp` (pyramid construction, downsample/combine), `tile_compile_cpp/apps/runner_phase_aqmh_maps.cpp` / `runner_phase_local_metrics.cpp` (per-frame orchestration and cache write-back)

**Investigation steps:**
1. Profile a representative run (`perf record` / built-in phase timers) to attribute the 1.88 s/frame cost between: GPU kernel time, CPU pre/post-processing (bilinear upsample, artifact mask), and cache write I/O.
2. Compare against v0.1.0's `AQMH_QUALITY_MAPS`/`LOCAL_METRICS` implementation (0.88 s/frame) to identify what specifically was added or changed in the v0.2.0 pyramid/combination path.
3. Check whether per-frame cache writes are synchronous and serialize with GPU dispatch (the "quick win" from the analysis: make write-back asynchronous, overlapping I/O of frame N with GPU compute of frame N+1).

**Expected gain:** Undetermined until profiling completes — treat as the primary open question of Item 3, not a guaranteed win. If the bottleneck is confirmed to be serialized write-back I/O, the async write-back fix should recover a meaningful fraction of the 650 s gap on its own, independent of the §3.1 overlap.

### 3.1 Overlap AQMH_MAPS with AQMH_RECONSTRUCTION

**Goal:** Start preloading quality maps into the reconstruction-side LRU as soon as each map is available, instead of first touching them during reconstruction after all 645 maps are complete.

**Constraint (established before design):** Reconstruction needs `global_weights` from `AQMH_GLOBAL_QUALITY`, which itself needs summaries from *all* maps. True per-row overlap of the compute-heavy weighted-MAD step is therefore not possible without changing the statistics. Three options were considered:
- **Option A:** Compute G weights incrementally as frames arrive (complex, changes the z-score statistics — rejected).
- **Option B:** Use v0.1.0-style `global_weights` from `GLOBAL_METRICS` for the overlapped portion, then correct with AQMH G weights in a final pass (rejected — introduces a second correctness-sensitive code path).
- **Option C (chosen):** Overlap only Q-map I/O with the tail of the maps phase. Q-map loading can start as soon as a frame's map is published; frame-pixel loading and the per-pixel sigma-clip/weighted-MAD computation still wait for G weights. This is a **Q-map prefetch** overlap, not a batch-reconstruct overlap — the queue design below reflects that. This is the safer option because no existing frame-pixel cache is assumed or invented by the plan.

#### 3.1.1 Prefetch Queue Architecture

**New file:** `tile_compile_cpp/src/reconstruction/aqmh_pipeline_overlap.hpp`

```cpp
namespace tile_compile::reconstruction {

/// Prefetch coordinator for overlapping AQMH_MAPS with Q-map I/O in
/// AQMH_RECONSTRUCTION (Option C — see plan §3.1).
/// This does NOT let reconstruction compute weighted-MAD output before
/// global_weights are available; it only lets Q-map loading start early
/// so maps are resident in QualityMapCache by the time AQMH_GLOBAL_QUALITY
/// finishes. It deliberately does not promise frame-pixel prefetch because
/// no frame-pixel cache is part of this plan.
class AqmhPrefetchCoordinator {
public:
  explicit AqmhPrefetchCoordinator(size_t frame_count);
  ~AqmhPrefetchCoordinator();

  /// Called by the maps phase: mark frame fi's Q-map as written to cache.
  /// Triggers async prefetch of that frame's Q-map into the reconstruction-side
  /// QualityMapCache resident LRU.
  void publish_frame(size_t fi);

  /// Called by the maps phase: signal that all frames are published.
  void finish();

  /// Called by reconstruction before starting the sigma-clip pass: block
  /// until every published Q-map has been prefetched (a no-op if maps already
  /// finished before reconstruction reached this point).
  void wait_all_prefetched();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace
```

**Implementation:** `src/reconstruction/aqmh_pipeline_overlap.cpp`

Thread management: `AqmhPrefetchCoordinator` owns an internal pool of worker threads (1–4, configurable). `publish_frame(fi)` enqueues the frame index into a `std::queue<size_t>` protected by `std::mutex` and signals workers via `std::condition_variable`. Workers call `QualityMapCache::read_cached(fi)` (not `read_region()`) to load the full Q-map into the LRU, then increment an `std::atomic<size_t> prefetched_count_`. `finish()` marks `finished_ = true`, rejects further `publish_frame()` calls, and wakes waiters; it does not imply shutdown while queued work remains. `wait_all_prefetched()` blocks until either `prefetched_count_ == published_count_ && finished_` or an error flag is set. The destructor sets `shutdown_ = true`, wakes workers, and joins all threads. Sketch:

```cpp
// Internal worker loop (simplified):
void worker() {
  while (true) {
    size_t fi;
    {
      std::unique_lock lock(mutex_);
      cv_.wait(lock, [&]{ return !queue_.empty() || shutdown_; });
      if (shutdown_ && queue_.empty()) return;
      fi = queue_.front(); queue_.pop();
    }
    try {
      q_map_cache_->read_cached(fi);  // populates LRU (see §3.2)
      ++prefetched_count_;
    } catch (...) {
      std::lock_guard err_lock(mutex_);
      error_ = std::current_exception();
    }
    cv_.notify_all();               // wakes wait_all_prefetched()
  }
}
```

- Thread-safe via `std::mutex` + `std::condition_variable`.
- `publish_frame(fi)` enqueues a background Q-map prefetch task via `QualityMapCache::read_cached()`, populating the LRU — see §3.2 for why this must go through the resident-cache path.
- `finish()` records that no further frames will be published; it is required so `wait_all_prefetched()` can distinguish "still waiting for maps" from "all maps published".
- `wait_all_prefetched()` blocks the reconstruction thread until every published Q-map has either been prefetched or an error occurred. On prefetch-only errors, it must **not** hard-fail the run: it logs a warning, disables further prefetching, sets `artifact["prefetch_fallback"] = true`, and lets reconstruction continue through the normal sequential `read_region()`/`read_cached()` path. Only if the sequential path also cannot read the Q-map does the run fail with a clear diagnostic.
- No `wait_for_batch()` / batch-of-frame-indices API — Option C never hands reconstruction a partial frame set to compute a final result from, since that would require statistics over a strict subset of frames (see §4.1.2 for why partial-frame computation is unsafe for this algorithm).

#### 3.1.2 Integration into Pipeline

**File:** `tile_compile_cpp/apps/runner_aqmh_pipeline.cpp`

Current flow (sequential):
```
run_phase_aqmh_maps() → run_phase_aqmh_global_quality() → run_phase_aqmh_reconstruction()
```

New flow (Option C — I/O overlap only):
```
Thread A: run_phase_aqmh_maps() → publish each frame → finish() → run_phase_aqmh_global_quality()
Thread B: on each publish_frame() → prefetch Q-map into QualityMapCache LRU (no compute)
Main:     after AQMH_GLOBAL_QUALITY completes → wait_all_prefetched() → run weighted-MAD pass
```

Implementation:
1. Create `AqmhPrefetchCoordinator` before starting the maps phase.
2. In `runner_phase_aqmh_maps.cpp`: after each frame's Q-map is written to cache, call `coordinator.publish_frame(fi)`.
3. After the maps loop publishes the last frame, call `coordinator.finish()` before starting `AQMH_GLOBAL_QUALITY`.
4. `AQMH_GLOBAL_QUALITY` still runs only after all maps are complete (it needs all frame summaries) — unchanged from today.
5. `run_phase_aqmh_reconstruction()` calls `coordinator.wait_all_prefetched()` before its per-pixel sigma-clip loop; in the common case this returns immediately because prefetch has already finished during the `AQMH_GLOBAL_QUALITY` phase.
6. The reconstruction row-chunk loop (`aqmh_reconstruction.cpp`, chunk-of-rows across all frames) is otherwise unchanged — Option C only moves *when* Q-map bytes are pulled into the resident LRU, not *how* the weighted-MAD is computed.

#### 3.1.3 Expected Gain

Only the Q-map I/O-bound portion of reconstruction can be hidden behind the tail of `AQMH_MAPS`; frame-pixel reads and the compute-bound sigma-clip pass are unaffected. Realistic estimate: **2–8% of the combined `AQMH_MAPS` + `AQMH_RECONSTRUCTION` time** until profiling proves a larger Q-map-I/O share, not the 20–40% figure implied by a full producer-consumer overlap — that figure assumed Option A/B-style compute overlap, which was rejected above. Measure the actual Q-map I/O share (via §3.0's profiling) before quoting a tighter number.

### 3.2 Fix Quality-Map Cache Residency (`max_resident_maps_observed: 0`)

**File:** `tile_compile_cpp/include/tile_compile/config/configuration.hpp:248-252`

Current:
```cpp
struct AqmhStorageConfig {
  int resolution_divisor = 1;
  std::string dtype = "float32";
  int max_resident_maps = 2;
};
```

Change default:
```cpp
int max_resident_maps = 4;  // was 2
```

**File:** `tile_compile_cpp/src/metrics/aqmh_quality_map_cache.cpp`

**Root cause (confirmed, not speculative):** an LRU cache already exists — `resident_`, `lru_`, `evict_to_limit_locked()`, and the `max_resident_maps_observed` stat are all implemented, gated on `max_resident_maps`. The bug is a **routing** bug, not a missing feature: `read_cached()` is the only call path that uses the LRU, but the actual hot path used by region-streaming reconstruction calls `read_region()` (`aqmh_reconstruction.cpp:90,192`), which reads directly and never touches `resident_`/`lru_`. This fully explains `max_resident_maps_observed: 0` in the v0.2.0 artifact — it is dead code from the region-streaming caller's perspective, not a config-passthrough issue.

**Fix (more invasive than "verify and adjust"):** route `read_region()` through the resident-cache/LRU, serving full-map reads from `resident_` when a map is already cached, and falling back to direct region reads only on a miss. This changes call-path behavior in the reconstruction hot loop. Treat it as a real code change with its own test coverage (§3.5), not a config-default bump. **Bumping `max_resident_maps` to 4 has zero effect until this routing fix lands.**

Concrete implementation sketch for `QualityMapCache::read_region()`:
```cpp
std::vector<float> QualityMapCache::read_region(size_t frame_idx, const Rect& region) {
  {
    std::lock_guard lock(mutex_);
    // 1. Fast path: if map is resident, extract region from it.
    auto it = resident_.find(frame_idx);
    if (it != resident_.end()) {
      lru_.touch(frame_idx);  // update recency
      return extract_region(it->second, region);
    }
  }

  // 2. Cache miss: load full map WITHOUT holding the cache mutex.
  auto full_map = load_map_from_disk(frame_idx);

  {
    std::lock_guard lock(mutex_);
    // 3. Double-check: another thread may have inserted while we were doing I/O.
    auto it = resident_.find(frame_idx);
    if (it != resident_.end()) {
      lru_.touch(frame_idx);
      return extract_region(it->second, region);
    }

    // 4. Insert, then evict until resident_.size() <= max_resident_maps.
    resident_[frame_idx] = std::move(full_map);
    lru_.push_back(frame_idx);
    evict_to_limit_locked();
    update_max_observed_locked();
    return extract_region(resident_.at(frame_idx), region);
  }
}

// Helper (already has a natural implementation):
std::vector<float> QualityMapCache::extract_region(
    const QualityMap& map, const Rect& r) {
  std::vector<float> out(r.width * r.height);
  for (int y = 0; y < r.height; ++y)
    for (int x = 0; x < r.width; ++x)
      out[y * r.width + x] = map.at(r.y + y, r.x + x);
  return out;
}
```

**Config file updates (apply only after the routing fix is verified to affect `max_resident_maps_observed`):**
- `tile_compile.yaml`: `max_resident_maps: 4`
- `examples/aqmh_tuning.example.yaml`: `max_resident_maps: 4`
- `examples/M42.global_medium.yaml`: `max_resident_maps: 4`
- All other example YAMLs with `aqmh:` section

### 3.3 Config File Updates for Item 3

**`tile_compile.yaml`:**
```yaml
aqmh:
  storage:
    max_resident_maps: 4    # was 2
```

**All example YAMLs** — update `max_resident_maps` to 4 where present.

### 3.4 Documentation Updates for Item 3

- `docs/AQMH/aqmh_v0.2.0_performance_analysis.md` Section 6.4 and 6.6 — update after implementation with measured before/after results; do not mark as implemented before code lands
- `docs/configuration_examples_practical_de.md` — mention overlap mode and `max_resident_maps` tuning

### 3.5 Tests for Item 3

- Test that `AqmhPrefetchCoordinator` correctly triggers Q-map prefetch on `publish_frame()`, that `finish()` unblocks `wait_all_prefetched()` once all published maps are prefetched, and that prefetch-only errors set `artifact["prefetch_fallback"] = true` while preserving byte-identical output through the sequential read path
- Test that reconstruction with the Option C prefetch overlap produces **byte-identical** output to sequential reconstruction (the prefetch must not change which frames/weights contribute to any pixel)
- Test that `read_region()` after the §3.2 routing fix actually populates/consults `resident_`/`lru_` (regression test for the dead-code-path bug, not just an end-to-end stat check)
- Test that `max_resident_maps: 4` results in `max_resident_maps_observed >= 2` in cache stats, run only after the routing fix — this test would currently pass trivially at `max_resident_maps: 2` doing nothing, so assert on the actual read-count reduction (fewer cache reads than the current 28 380), not just the observed-maps counter

---

## Item 4: Long-Term (Code Changes)

### 4.1 Native CUDA Backend for AQMH Reconstruction

#### 4.1.1 CMakeLists.txt Changes

**File:** `tile_compile_cpp/CMakeLists.txt`

**Note:** `TILE_COMPILE_ENABLE_CUDA` (option) and `TILE_COMPILE_WITH_CUDA` (resolved variable, with nvcc detection) already exist around lines 78-102, together with the comment this item removes (see step 4). The steps below extend that **existing** `if(TILE_COMPILE_WITH_CUDA)` block — do not add a second, parallel one.

1. **Enable CUDA language when .cu files are present (inside the existing `TILE_COMPILE_WITH_CUDA` block, not a new one):**
```cmake
# Inside the existing if(TILE_COMPILE_WITH_CUDA) block (~line 78-102):
if(TILE_COMPILE_WITH_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    # Architecture flags — auto-detect or use generic arch
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "75;86;89;90")  # Turing, Ampere, Ada, Hopper
    endif()
endif()
```

2. **Add CUDA source to library:**
```cmake
# In LIB_SOURCES (after line 444):
if(TILE_COMPILE_WITH_CUDA)
    list(APPEND LIB_SOURCES src/reconstruction/aqmh_reconstruction_cuda.cu)
endif()
```

3. **Link CUDA libraries:**
```cmake
# After target_link_libraries for tile_compile_lib:
if(TILE_COMPILE_WITH_CUDA)
    target_link_libraries(tile_compile_lib PUBLIC CUDA::cudart)
endif()
```

4. **Update comment at line 95-98** — remove "There are currently no .cu translation units" since we now have one.

#### 4.1.2 CUDA Kernel Implementation

**New file:** `tile_compile_cpp/src/reconstruction/aqmh_reconstruction_cuda.cu`

**New header:** `tile_compile_cpp/include/tile_compile/reconstruction/aqmh_reconstruction_cuda.hpp`

```cpp
#pragma once
#include "tile_compile/reconstruction/aqmh_reconstruction.hpp"

namespace tile_compile::reconstruction {

/// CUDA-accelerated AQMH weighted-MAD reconstruction.
/// Falls back to CPU if CUDA kernel fails or insufficient device memory.
AqmhReconstructionResult reconstruct_aqmh_weighted_cuda(
    size_t frame_count, const AqmhFrameLoader &load_frame,
    metrics::QualityMapCache *q_map_cache, const VectorXf &global_weights,
    const std::vector<uint8_t> &canvas_mask, int width, int height,
    const AqmhReconstructionConfig &cfg,
    const AqmhMaskLoader &load_frame_valid_mask = {},
    const AqmhProgressCallback &progress = {});

} // namespace
```

**Kernel design (`aqmh_reconstruction_cuda.cu`):**

**Correctness constraint (must be respected by the batching strategy):** weighted-median/MAD sigma-clipping is a statistic computed **jointly over every frame that contributes to a pixel**. Splitting `frame_count` frames into independent batches and computing a final weighted mean/MAD per batch — as an earlier draft of this plan proposed — silently computes each pixel's result from only a subset of the 645 frames, which is a correctness bug, not a performance detail: a naive `d_frames`/`d_q_maps` upload of all 645 frames is ~23 GB, larger than most GPU VRAM, so batching is unavoidable, but it **must not** be done along the frame axis for a single output pass. The existing CPU implementation (`aqmh_reconstruction.cpp`) already solves this correctly by chunking along **rows**, not frames: each row-chunk still sees all `frame_count` frames' data for that row range, so the per-pixel sample set is always complete. The CUDA kernel must use the same strategy — this also removes the earlier conflict with §4.3 ("GPU paths process the whole canvas without a chunk loop"): the GPU chunk loop still exists, just over rows (matching `chunk_rows`), not over frames; §4.3 is revised below to reflect this.

1. **Memory layout (per row-chunk, not per whole canvas):**
   - For a chunk of `chunk_rows` rows, upload all `frame_count` frames' pixel data for those rows as `float* d_frames` of size `frame_count * chunk_rows * width`
   - Upload the matching Q-map slice as `float* d_q_maps`, same shape
   - Upload `canvas_mask` slice, `frame_valid_masks` slice, and the full `global_weights` (`frame_count` floats — small, upload once for the whole run)
   - Output per chunk: `float* d_output`, `float* d_weight_sum`, `float* d_uniform_control`, sized `chunk_rows * width`

2. **Row-chunk sizing (mirrors `chunk_rows` from §2.1.2, not an independent GPU-only formula):**
   - `device_memory_budget = min(0.60 * cuda_free_memory, 4 GiB)`, where `cuda_free_memory` comes from `cudaMemGetInfo(&free_bytes, &total_bytes)`. If `cudaMemGetInfo` fails, or the computed budget cannot fit at least one row, fall back to CPU.
   - `chunk_rows = max(1, min(height, device_memory_budget / estimated_bytes_per_row_all_frames))`
   - This reuses `cfg.aqmh.reconstruction.chunk_rows` when explicitly set (§2.1.2); `0`/auto derives the value above from available device memory instead of `memory_budget_mb`.
   - `estimated_bytes_per_row_all_frames` must include frames, Q-maps, frame masks, output buffers, and a small safety margin; do not use only the `frames + q_maps` term for allocation decisions.
   - Process chunk → write its `chunk_rows` output rows → free device buffers → load next chunk. Every chunk still contains all `frame_count` frames for its row range, so no pixel's sample set is ever partial.

3. **Memory layout (row-major, must be documented explicitly):**
```
 frames:      [frame_count][chunk_rows][width]  float32 — pixel values
 q_maps:      [frame_count][chunk_rows][width]  float32 — quality scores
 frame_masks: [frame_count][chunk_rows][width]  uint8   — per-pixel validity
 canvas_mask: [chunk_rows][width]               uint8   — output mask
 global_weights: [frame_count]                  float32 — small, upload once
 output:      [chunk_rows][width]               float32
 weight_sum:  [chunk_rows][width]               float32
 uniform_control: [chunk_rows][width]           float32 (optional)
```
 All device-side index arithmetic must use this layout: `idx_fi_y_x = fi * chunk_rows * width + y * width + x`.

4. **Per-pixel kernel (operates within one row-chunk; `frame_count` here is the FULL frame count, not a frame-axis batch):**

**`MAX_FRAMES` and local-memory strategy:** `float values[MAX_FRAMES]`/`weights[MAX_FRAMES]` as fixed-size per-thread local arrays for `frame_count` up to 645 is ~5 KB/thread, which will spill to global memory and hurt occupancy. Two viable approaches; the plan mandates resolving this before implementation:

- **Option A — Dynamic shared memory (recommended for ≥ 128 frames):** Allocate `extern __shared__` buffers, partitioned per thread within the block. For a 16×16 block and 645 frames: `16*16*645*2*4 = ~13 MB/block` — **exceeds hardware limits (~100 KB on Ampere)**. Therefore shared memory can only hold a sub-group of frames at a time, using an *online* or *streaming* accumulation of the weighted statistics. This changes the sigma-clip to an incremental algorithm but avoids spill.
- **Option B — Accept local-memory spill (simplest, prototype first):** Declare `float values[MAX_FRAMES]` where `MAX_FRAMES` is a compile-time constant (e.g. `constexpr int MAX_FRAMES = 1024;` or a template parameter `template<int MAX_FRAMES>`). The kernel will spill to L2 cache-backed local memory; occupancy drops but correctness is unaffected. Profile before committing to Option A.

**Chosen approach for this plan:** Prototype with Option B (compile-time constant, accept spill), measure throughput, then switch to an online algorithm if occupancy loss is unacceptable.

```cuda
constexpr int MAX_FRAMES_COMPILE = 1024;  // upper bound; CPU fallback if frame_count exceeds it

__global__ void weighted_mad_sigma_clip_kernel(
    const float* __restrict__ frames,      // [frame_count, chunk_rows, width]
    const float* __restrict__ q_maps,      // [frame_count, chunk_rows, width]
    const uint8_t* __restrict__ canvas_mask,
    const uint8_t* __restrict__ frame_masks, // [frame_count, chunk_rows, width]
    const float* __restrict__ global_weights,
    float* __restrict__ output,
    float* __restrict__ weight_sum,
    float* __restrict__ uniform_control,
    int width, int chunk_rows, int frame_count,
    float clip_sigma, int clip_iterations,
    float min_fraction, float min_n_eff,
    bool cherry_pick_enabled, float cherry_pick_k_frac,
    int cherry_pick_k_min_required,
    bool compute_uniform_control) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;  // row within this chunk
    if (x >= width || y >= chunk_rows) return;
    if (canvas_mask[y * width + x] == 0) return;

    // Gather samples across ALL frame_count frames for this pixel — never a subset
    // (local arrays spill to L2-cached local memory for large frame_count; see note above)
    float values[MAX_FRAMES_COMPILE];
    float weights[MAX_FRAMES_COMPILE];
    int n_samples = 0;

    for (int fi = 0; fi < frame_count; ++fi) {
        const int idx = fi * chunk_rows * width + y * width + x;
        if (frame_masks[idx] == 0) continue;
        const float v = frames[idx];
        const float q = q_maps[idx];
        if (!isfinite(v) || !isfinite(q)) continue;
        const float gw = global_weights[fi];
        const float score = gw * max(0.0f, q);
        if (score > 0.0f) {
            values[n_samples] = v;
            weights[n_samples] = score;
            ++n_samples;
        }
    }

    if (n_samples == 0) { /* unsupported pixel */ return; }

    // Cherry-pick: partial sort to select top-K (on-device)
    // ...

    // Weighted median + MAD + sigma-clip iterations
    // ...

    // Write output
    output[y * width + x] = weighted_mean;
    weight_sum[y * width + x] = total_weight;
    if (compute_uniform_control) uniform_control[y * width + x] = control_mean;
}
```

5. **Shared memory note (not used for per-pixel sample storage):**
   - Do NOT use `__shared__ float s_values[block_size * frame_count]` — for 16×16 blocks and 645 frames this exceeds the hardware shared-memory limit by ~130×.
   - The per-pixel `values[]`/`weights[]` arrays live in per-thread local memory (see Option B above). Shared memory may only be used for unrelated small block-level reductions if profiling identifies such a need; it must not hold the per-pixel frame sample set.
   - Block size: 16×16 or 32×8, tuned after profiling the local-memory-spill cost.

6. **Cherry-pick on device:**
   - Use a per-thread local-memory selection algorithm over `values[]`/`weights[]`; do not use CUB `DeviceRadixSort`, because that is a device-wide primitive and is not appropriate inside a per-pixel kernel.
   - For cherry-pick top-K by score, use bounded insertion/selection over the per-thread arrays, preserving `(score desc, frame_index asc)` as deterministic tiebreaker.
   - For weighted median/MAD, sort the selected per-thread sample list by `(value, frame_index)` using insertion sort for small `n_samples`; prototype bitonic-in-local-memory only if profiling shows insertion sort dominates runtime.
   - Do not store per-pixel sample arrays in shared memory; §4.1.2 step 5 explains why full-frame shared memory is not feasible.

7. **Deterministic tiebreaker:**
   - Sort by `(value, frame_index)` — frame_index is the global frame index (0..frame_count-1), unaffected by row-chunking since every chunk sees the same full frame set
   - This matches the CPU implementation's deterministic ordering

#### 4.1.3 Activate `cuda` Backend for `aqmh_reconstruction`

**File:** `tile_compile_cpp/src/core/acceleration.cpp:96-97`

Current:
```cpp
case AccelerationBackend::cuda:
  return false;
```

New:
```cpp
case AccelerationBackend::cuda:
  return phase == AccelerationPhase::aqmh_reconstruction;
```

**Note on the code below:** `acc.selected` and `acc.request_honored` are real fields on `AccelerationSelection` (`include/tile_compile/core/acceleration.hpp:69-77`), so this snippet is syntactically valid — but as of today `runner_phase_aqmh_reconstruction.cpp` contains **zero** `acc.selected ==` branches; this dispatch logic is entirely new, not a modification of existing branches. The `if (...using_gpu)` log block at lines 38-43 that this item removes is itself gated on `using_gpu`, and becomes dead once real dispatch branches exist — remove the whole conditional, not just the log string.

**Backend arbitration — concrete implementation required in `src/core/acceleration.cpp`:** once both `cuda` (§4.1.3) and `opencv_opencl` (§4.2.1) return `true` for `aqmh_reconstruction`, and `opencv_cuda` already does today (`selected_backend: opencv_cuda`, which is why v0.2.0 currently falls back to CPU), the "auto" resolution must be made explicit. Without an override, "auto" keeps resolving to `opencv_cuda`, silently defeating the entire item.

**Required changes to `src/core/acceleration.cpp`:**

1. **Exclude `opencv_cuda` for this phase** (in `phase_supports_backend()` or equivalent):
```cpp
case AccelerationPhase::aqmh_reconstruction:
  // opencv_cuda has no v0.2.0 GPU implementation — explicitly excluded
  return backend == AccelerationBackend::cuda ||
         backend == AccelerationBackend::opencv_opencl ||
         backend == AccelerationBackend::cpu;
```

2. **Add phase-specific priority for "auto" resolution** (in `select_backend_for_phase()` or wherever `acceleration_backend: auto` is resolved):
```cpp
if (phase == AccelerationPhase::aqmh_reconstruction) {
  // Priority: native cuda > opencv_opencl > cpu (opencv_cuda excluded above)
  for (auto backend : {AccelerationBackend::cuda,
                       AccelerationBackend::opencv_opencl,
                       AccelerationBackend::cpu}) {
    if (phase_supports_backend(phase, backend) &&
        backend_is_available(backend))  // device present + compiled in
      return backend;
  }
}
// ... existing default logic for other phases ...
```

This ensures that on a CUDA-capable machine, "auto" selects native `cuda`; on a non-CUDA machine with OpenCL, it selects `opencv_opencl`; otherwise CPU. The `gpu_reconstruction` rollout gate (see below) is checked *after* backend selection and can suppress dispatch to GPU regardless of what "auto" resolved to.

**Rollout safety gate:** a brand-new CUDA/OpenCL kernel implementing weighted-median/MAD/sigma-clip is high-risk for silent numerical divergence from the CPU reference (see the local-memory-spill and correctness notes in §4.1.2). Add `aqmh.reconstruction.gpu_reconstruction: auto | force | disabled` (independent of the general `acceleration_backend` setting), defaulting to `disabled`. `acceleration_fallback` handles *crashes*; this handles *wrong-but-successful* results, which fallback-on-failure cannot catch.

**Default flip criteria (`disabled` → `auto`):** keep the default disabled until all of the following are true:
- At least 3 real datasets pass determinism tests, covering: small/medium frame count, high frame count, and masks/NaNs/artifacts.
- CUDA/OpenCL output satisfies `abs(cpu - gpu) <= max(1e-3, 1e-5 * abs(cpu))` for reconstructed pixels.
- Unsupported-pixel masks and NaN masks are identical to CPU output.
- `weight_sum` passes the same tolerance family, and mean signed error shows no systematic bias.
- Median reconstruction speedup is at least 2.0× over CPU.
- No validation dataset is slower than CPU by more than 10%.
- Failure/fallback path is covered by tests.

`force` is for manual experiments only and must never become the default.

#### 4.1.4 Hook CUDA Path into Runner

**File:** `tile_compile_cpp/apps/runner_phase_aqmh_reconstruction.cpp`

```cpp
#include "tile_compile/reconstruction/aqmh_reconstruction_cuda.hpp"

// In run_phase_aqmh_reconstruction():
const auto &acc = aqmh_reconstruction_acceleration;

AqmhReconstructionResult aqmh_recon;
if (acc.selected == core::AccelerationBackend::cuda && acc.request_honored) {
  log_file << "[AQMH_RECONSTRUCTION] Using native CUDA backend" << std::endl;
  aqmh_recon = reconstruction::reconstruct_aqmh_weighted_cuda(
      frames.size(), aqmh_frame_loader, aqmh_cache.get(), aqmh_global_weights,
      common_valid_mask, canvas_width, canvas_height, aqmh_recon_cfg,
      aqmh_mask_loader, [&](int d, int t) { /* progress */ });
  artifact["execution_backend"] = "cuda_native";
  artifact["acceleration_used"] = true;
  artifact["acceleration_fallback"] = false;
} else {
  // existing CPU path
  aqmh_recon = reconstruction::reconstruct_aqmh_weighted(...);
  artifact["execution_backend"] = "cpu_exact_v0_2";
  artifact["acceleration_used"] = false;
  artifact["acceleration_fallback"] = acc.using_gpu;
}
```

Remove the whole `if (...using_gpu)` conditional at lines 38-43 (the hardcoded "v0.2 weighted-MAD uses the exact CPU..." log block), not just its log string — it becomes dead code once the dispatch branches above exist.

### 4.2 OpenCL Backend for AQMH Reconstruction

#### 4.2.1 Activate `opencv_opencl` for `aqmh_reconstruction`

**File:** `tile_compile_cpp/src/core/acceleration.cpp:91-94`

Current:
```cpp
case AccelerationBackend::opencv_opencl:
  return phase == AccelerationPhase::prewarp ||
         phase == AccelerationPhase::aqmh_maps ||
         phase == AccelerationPhase::tile_reconstruction ||
         phase == AccelerationPhase::stacking;
```

New:
```cpp
case AccelerationBackend::opencv_opencl:
  return phase == AccelerationPhase::prewarp ||
         phase == AccelerationPhase::aqmh_maps ||
         phase == AccelerationPhase::aqmh_reconstruction ||
         phase == AccelerationPhase::tile_reconstruction ||
         phase == AccelerationPhase::stacking;
```

#### 4.2.2 OpenCL Kernel Implementation

**New file:** `tile_compile_cpp/src/reconstruction/aqmh_reconstruction_opencl.cpp`

**New header:** `tile_compile_cpp/include/tile_compile/reconstruction/aqmh_reconstruction_opencl.hpp`

```cpp
#pragma once
#include "tile_compile/reconstruction/aqmh_reconstruction.hpp"

namespace tile_compile::reconstruction {

AqmhReconstructionResult reconstruct_aqmh_weighted_opencl(
    size_t frame_count, const AqmhFrameLoader &load_frame,
    metrics::QualityMapCache *q_map_cache, const VectorXf &global_weights,
    const std::vector<uint8_t> &canvas_mask, int width, int height,
    const AqmhReconstructionConfig &cfg,
    const AqmhMaskLoader &load_frame_valid_mask = {},
    const AqmhProgressCallback &progress = {});

} // namespace
```

**Implementation approach:**
- Use `cv::ocl::ProgramSource` to compile an OpenCL C kernel at runtime
- The kernel mirrors the CUDA kernel's **row-chunked** logic 1:1 (see §4.1.2's correctness constraint — chunk along rows, keep the full `frame_count` per pixel, never split frames across independent output-producing batches)
- Start with `cv::UMat` + `cv::ocl::Kernel` because OpenCV/OpenCL is already available in the project. If `UMat` prevents a contiguous `[frame_count][chunk_rows][width]` layout or adds measurable overhead, switch this implementation to direct `cl::Buffer` while preserving the same kernel contract.
- Per-pixel kernel: gather across all `frame_count` frames for the chunk's rows, cherry-pick, weighted median/MAD, sigma-clip
- Use the same per-thread local-memory strategy as the CUDA prototype: fixed-size private arrays for `values[]`/`weights[]`, bounded insertion/selection for cherry-pick top-K, and insertion sort by `(value, frame_index)` for weighted median/MAD. Do **not** store `frame_count` samples per work-item in OpenCL local memory; it has the same capacity problem as CUDA shared memory. If private-memory spill is too expensive, prototype the same streaming/online alternative chosen for CUDA rather than diverging algorithmically.

**OpenCL C kernel source** (embedded as string constant or loaded from `.cl` file):
```c
__kernel void weighted_mad_sigma_clip(
    __global const float *frames,      // [frame_count, chunk_rows, width]
    __global const float *q_maps,      // [frame_count, chunk_rows, width]
    __global const uchar *canvas_mask,
    __global const uchar *frame_masks,
    __global const float *global_weights,
    __global float *output,
    __global float *weight_sum,
    __global float *uniform_control,
    int width, int chunk_rows, int frame_count,
    float clip_sigma, int clip_iterations,
    float min_fraction, float min_n_eff,
    int cherry_pick_enabled, float cherry_pick_k_frac,
    int cherry_pick_k_min_required,
    int compute_uniform_control) {

    const int x = get_global_id(0);
    const int y = get_global_id(1);  // row within this chunk
    // ... same logic as CUDA kernel: loop fi in [0, frame_count) — never a frame-axis subset ...
    // Use private arrays with a compile-time MAX_FRAMES_COMPILE equivalent;
    // reject or CPU-fallback when frame_count exceeds that bound.
}
```

#### 4.2.3 Hook OpenCL Path into Runner

**File:** `tile_compile_cpp/apps/runner_phase_aqmh_reconstruction.cpp`

```cpp
#include "tile_compile/reconstruction/aqmh_reconstruction_opencl.hpp"

// In the backend selection:
if (acc.selected == core::AccelerationBackend::cuda && acc.request_honored) {
  // CUDA path
} else if (acc.selected == core::AccelerationBackend::opencv_opencl && acc.request_honored) {
  log_file << "[AQMH_RECONSTRUCTION] Using OpenCL backend" << std::endl;
  aqmh_recon = reconstruction::reconstruct_aqmh_weighted_opencl(...);
  artifact["execution_backend"] = "opencl";
  artifact["acceleration_used"] = true;
  artifact["acceleration_fallback"] = false;
} else {
  // CPU path (existing)
}
```

#### 4.2.4 CMakeLists.txt for OpenCL

OpenCL is already available via OpenCV (`TILE_COMPILE_HAS_OPENCV_OPENCL`). No additional CMake changes needed — the OpenCL implementation uses `cv::ocl` which is part of OpenCV.

Add the new source file:
```cmake
# In LIB_SOURCES:
src/reconstruction/aqmh_reconstruction_opencl.cpp
```

### 4.3 Use Larger, Memory-Sized Row Chunks for GPU Paths (Revised — Superseded "No Chunking" Claim)

**File:** `tile_compile_cpp/src/reconstruction/aqmh_reconstruction_cuda.cu` and `aqmh_reconstruction_opencl.cpp`

**Correction:** an earlier draft of this item said GPU paths should process "the whole canvas without a chunk loop" (`chunk_rows = height`, `chunk_count = 1`). That directly contradicts §4.1.2/§4.2.2: uploading all `frame_count` frames for the *entire* canvas at once is ~23 GB, which does not fit in typical GPU VRAM, which is exactly why §4.1.2 batches by row-chunk in the first place. The two cannot both be true.

**What GPU paths actually do:** keep the row-chunk loop (same shape as the CPU's 53-row chunks), but size chunks by *device* memory rather than the CPU's `memory_budget_mb`-derived clamp — GPU VRAM bandwidth and capacity differ from the CPU host-memory budget, so `chunk_rows` for the GPU path is typically much larger than the CPU's 53 rows (fewer, bigger chunks → fewer read/dispatch round-trips, which is the real target of "avoid region streaming overhead"), but it is not literally 1 chunk. Compute it via the formula in §4.1.2 step 2 (`device_memory_budget / (frame_count * width * sizeof(float) * 2)`), and report the actual resulting `chunk_rows` / `chunk_count` in the artifact — do not hardcode `height` / `1`.

### 4.4 Replace JSON Diagnostics with Compact Binary Format

#### 4.4.1 Binary Format Design

**New file:** `tile_compile_cpp/include/tile_compile/metrics/aqmh_binary_diagnostics.hpp`

Format: Simple tagged binary format (not FITS, to avoid FITS overhead for small structured data):

```
Header:
  magic: "AQDB" (4 bytes)
  version: uint32 (1)
  frame_count: uint32
  canvas_width: uint32   — needed to compute block_grid_width on read
  canvas_height: uint32  — needed to compute block_grid_height on read
  block_size_px: uint32
  has_heatmaps: uint8 (0/1)
  — NOTE: block_grid_width/block_grid_height are NOT stored in the header;
    readers compute them as ceil(canvas_width / block_size_px) and
    ceil(canvas_height / block_size_px), saving 8 bytes and avoiding
    redundant data that could become inconsistent.

Per-frame records (frame_count ×):
  frame_index: uint32
  map_mean: float32
  map_p10: float32
  map_p90: float32
  artifact_frac: float32
  sharpness_p50: float32
  snr_p50: float32
  n_regions: uint32
  global_quality: float32
  global_sharpness_input: float32
  global_snr_input: float32
  global_quality_input_invalid: uint8
  — ~50 bytes/frame total (including alignment padding)

Block arrays (if has_heatmaps), each of size block_grid_width × block_grid_height float32:
  aqmh_q_median
  aqmh_q_p10
  aqmh_q_p90
  aqmh_artifact_frac
  q_map_heatmap
  artifact_heatmap
```

**Size estimate (corrected — the 1.5 MB figure in the earlier draft was wrong):**

For a typical 2310×3924 canvas with `block_size_px = 6` (same as `r_morph_canvas_px`):
- Per-frame records: 645 × 50 bytes = **~32 KB**
- block_grid: ceil(3924/6) = 654, ceil(2310/6) = 385
- Per array: 654 × 385 × 4 bytes = **~1.02 MB**
- 6 arrays: **~6.12 MB**
- **Total: ~6.15 MB** (vs 178 MB JSON — still a 29× reduction, but not the 120× implied by 1.5 MB)

With `block_size_px = 32`: block_grid = 123×73; 6 arrays × ~36 KB = **~247 KB total** — much smaller, at the cost of lower spatial resolution in the heatmaps.

**Recommendation:** Use `block_size_px = r_morph_canvas_px` (same parameter, consistent with diagnostics) as the default. Add `AqmhDiagnosticsConfig::binary_block_size_px = 0`, where `0` means use `r_morph_canvas_px`; values `> 0` override only the binary heatmap grid resolution. This keeps the default semantically identical to existing diagnostics while allowing low-I/O profiles (e.g. `binary_block_size_px: 32`) to reduce binary heatmaps to ~247 KB.

#### 4.4.2 Implementation

**New file:** `tile_compile_cpp/src/metrics/aqmh_binary_diagnostics.cpp`

```cpp
void write_binary_diagnostics(const std::filesystem::path &path,
                              const AqmhBinaryDiagnostics &diag);
AqmhBinaryDiagnostics read_binary_diagnostics(const std::filesystem::path &path);
```

**File:** `tile_compile_cpp/apps/runner_phase_aqmh_diagnostics.cpp`

Add config flag `aqmh.diagnostics.format` (`"json"` | `"binary"`):
- `"binary"`: write `aqmh_metrics.bin` instead of `aqmh_metrics.json`
- `"json"`: current behavior (default for backward compatibility)

#### 4.4.3 Config Addition

`AqmhDiagnosticsConfig::format` is already introduced in §2.1.1 so the config file, schema, serialization, and web/backend work have a single source of truth. Item 4 only changes the runtime behavior behind that field: `json` keeps the existing output path, `binary` writes/reads the new AQDB format.

#### 4.4.4 Web Backend / Frontend

- `web_backend_cpp/`: add endpoint for binary diagnostics download
- `web_frontend_v3/`: add client-side binary parser (or backend converts to JSON on-the-fly for API responses)

### 4.5 Config File Updates for Item 4

**`tile_compile.yaml`:**
```yaml
aqmh:
  diagnostics:
    format: json           # json | binary
    binary_block_size_px: 0 # 0 = use r_morph_canvas_px
  storage:
    max_resident_maps: 4
  reconstruction:
    gpu_reconstruction: disabled   # auto | force | disabled — rollout gate, see §4.1.4; default disabled until determinism-validated
```

**All example YAMLs** — add `format: json` and `binary_block_size_px: 0` to diagnostics section, and `gpu_reconstruction: disabled` to reconstruction section.

### 4.6 Documentation Updates for Item 4

- `docs/AQMH/aqmh_v0.2.0_performance_analysis.md` Section 6.1 — update after implementation with measured CUDA/OpenCL results; do not mark as implemented before code lands
- `docs/AQMH/aqmh_v0.2.0_performance_analysis.md` Section 6.3 — update after implementation with measured binary-format size and load-time results
- `docs/api/cpp/` — document new CUDA/OpenCL reconstruction APIs
- `README.md` / `README_de.md` — mention CUDA/OpenCL backend support for AQMH reconstruction

### 4.7 Tests for Item 4

- **CUDA kernel smoke test:** Run `reconstruct_aqmh_weighted_cuda()` on a small dataset (10 frames, 64×64) and compare output to `reconstruct_aqmh_weighted()` (CPU) within float32 tolerance — this is a smoke test only, not sufficient sign-off (see below)
- **CUDA row-chunking correctness test:** Run with `frame_count` and canvas size forced large enough to require ≥2 GPU row-chunks (e.g. mock a tiny `device_memory_budget`), and assert output is identical to a single-chunk run — this is the regression test for the frame-batching-vs-row-batching bug identified in §4.1.2; without it, a future change could silently reintroduce per-batch partial-frame statistics
- **OpenCL kernel test:** Same as the CUDA smoke + chunking tests, for `reconstruct_aqmh_weighted_opencl()`
- **Backend arbitration tests (unit-level, no real device required — mock `backend_is_available()`):**
  - With `{cuda, opencv_opencl}` available → must select `cuda`
  - With `{opencv_opencl}` available → must select `opencv_opencl`
  - With `{opencv_cuda}` only available → must select `cpu` (opencv_cuda excluded for this phase)
  - With `{cuda, opencv_cuda, opencv_opencl}` available → must select `cuda` (not opencv_cuda)
  - These tests exercise `select_backend_for_phase(aqmh_reconstruction, ...)` and `phase_supports_backend()` directly, without running a full pipeline — they are the regression guard for the §4.1.3 priority-order fix
- **Backend selection test:** Verify that explicit `acceleration_backend: cuda` selects the CUDA dispatch branch, and `opencv_opencl` selects the OpenCL branch, in `runner_phase_aqmh_reconstruction.cpp`
- **Fallback test:** Verify that CUDA/OpenCL crash/failure (simulated via `reconstruct_aqmh_weighted_cuda()` returning an error) falls back to CPU and sets `artifact["acceleration_fallback"] = true`
- **Rollout gate test:** Verify `aqmh.reconstruction.gpu_reconstruction: disabled` never dispatches to CUDA/OpenCL even when the backend is available and selected, and that `force` bypasses backend auto-detection
- **Binary diagnostics test:** Write and read binary diagnostics, verify round-trip integrity
- **Determinism test (multi-dataset, required before flipping the rollout gate default):** Verify CUDA/OpenCL and CPU match across at least 3 real-scale datasets with varying frame counts/masking patterns, not only the 10-frame/64×64 toy case — the toy case cannot exercise the row-chunking path or realistic masking/NaN patterns. Acceptance criteria: `abs(cpu - gpu) <= max(1e-3, 1e-5 * abs(cpu))`, identical unsupported-pixel mask, identical NaN mask, `weight_sum` within the same tolerance family, p99 absolute error <= `2e-4` where applicable, mean absolute error <= `5e-5` where applicable, and no systematic signed bias.

---

## Summary: All Files to Change

### New Files

| File | Item | Purpose |
|------|------|---------|
| `src/reconstruction/aqmh_reconstruction_cuda.cu` | 4.1 | Native CUDA kernel |
| `include/.../aqmh_reconstruction_cuda.hpp` | 4.1 | CUDA header |
| `src/reconstruction/aqmh_reconstruction_opencl.cpp` | 4.2 | OpenCL kernel |
| `include/.../aqmh_reconstruction_opencl.hpp` | 4.2 | OpenCL header |
| `src/reconstruction/aqmh_pipeline_overlap.cpp` | 3.1 | `AqmhPrefetchCoordinator` impl (Q-map I/O-only prefetch, Option C) |
| `include/.../aqmh_pipeline_overlap.hpp` | 3.1 | `AqmhPrefetchCoordinator` header |
| `src/metrics/aqmh_binary_diagnostics.cpp` | 4.4 | Binary diagnostics format |
| `include/.../aqmh_binary_diagnostics.hpp` | 4.4 | Binary diagnostics header |

### Modified Files

| File | Items | Changes |
|------|-------|---------|
| `include/tile_compile/config/configuration.hpp` | 2.1, 3.2, 4.4 | `AqmhDiagnosticsConfig` fields (`enabled`, `level`, `per_frame_blocks`, `heatmaps`, `regions`, `format`, `binary_block_size_px`), `AqmhReconstructionConfig` fields (`chunk_rows`, `memory_budget_mb`, `gpu_reconstruction`), `max_resident_maps` default |
| `src/io/config.cpp` | 2.1, 4.4 | Parse/serialize/validate new fields, schema JSON |
| `apps/runner_phase_aqmh_diagnostics.cpp` | 2.1, 4.4 | Respect `enabled`/`level`/`per_frame_blocks`/`heatmaps`/`regions`/`format` |
| `apps/runner_phase_aqmh_reconstruction.cpp` | 2.1, 4.1, 4.2 | `chunk_rows` passthrough, CUDA/OpenCL backend selection, `cuda > opencv_opencl > cpu` arbitration, `gpu_reconstruction` gate |
| `src/reconstruction/aqmh_reconstruction.cpp` | 2.1 | `chunk_rows` override in chunk calculation |
| `include/.../aqmh_reconstruction.hpp` | 2.1 | Add `chunk_rows` and `memory_budget_mb` to internal `reconstruction::AqmhReconstructionConfig` |
| `src/metrics/aqmh_quality_map_cache.cpp` | 3.2 | Route `read_region()` through the existing LRU (root-cause fix, not a bookkeeping tweak) |
| `src/core/acceleration.cpp` | 4.1, 4.2 | Activate `cuda` and `opencv_opencl` for `aqmh_reconstruction`; add explicit backend-priority rule for "auto" (excluding `opencv_cuda` for this phase) |
| `CMakeLists.txt` | 4.1 | Extend existing `TILE_COMPILE_WITH_CUDA` block: enable CUDA language, add .cu source, link cudart |
| `tile_compile.yaml` | 2.1, 3.2, 4.4 | New config fields |
| `tile_compile.schema.json` | 2.1, 4.4 | Schema updates |
| All 19 example YAMLs | 2.1, 3.2, 4.4 | New diagnostics/reconstruction/storage config fields (`format`, `binary_block_size_px`, `chunk_rows`, `memory_budget_mb`, `gpu_reconstruction`, `max_resident_maps`) |
| `docs/AQMH/aqmh_v0.2.0_performance_analysis.md` | all, plus §6.7 (AQMH_MAPS bottleneck) | Update after implementation with measured results and final status |
| `docs/configuration_examples_practical_de.md` | 2.1 | Config examples |
| `README.md` / `README_de.md` | 4.1, 4.2 | CUDA/OpenCL backend docs |

---

## Implementation Order

1. **Item 2 (short-term, config only)** — no behavioral change, just new config fields + diagnostics gating
   1. `configuration.hpp` — add fields
   2. `config.cpp` — parse/serialize/validate/schema
   3. `runner_phase_aqmh_diagnostics.cpp` — respect flags
   4. `aqmh_reconstruction.cpp` + `aqmh_reconstruction.hpp` — `chunk_rows` override and internal `memory_budget_mb` support
   5. `runner_phase_aqmh_reconstruction.cpp` — passthrough `chunk_rows` + `memory_budget_mb` fallback from global config
   6. All YAML files + schema
   7. Tests
   8. Docs

2. **Item 3 (medium-term, code changes)** — AQMH_MAPS profiling, cache fix, then Q-map I/O prefetch overlap
   1. **§3.0 profiling first** — attribute the 650 s `AQMH_MAPS` regression before writing any overlap/cache code; its outcome may change the priority or necessity of steps below
   2. `aqmh_quality_map_cache.cpp` — fix the `read_region()` routing bug so the existing LRU is actually consulted (§3.2) — do this **before** bumping the config default, since the default has no effect otherwise
   3. `configuration.hpp` — bump `max_resident_maps` default (only after step 2 is verified to change `max_resident_maps_observed`)
   4. `aqmh_pipeline_overlap.hpp/.cpp` — `AqmhPrefetchCoordinator` implementation (Option C, Q-map I/O prefetch only; not a full producer-consumer batch queue and not frame-pixel prefetch)
   5. `runner_aqmh_pipeline.cpp` — integrate prefetch coordinator
   6. YAML files
   7. Tests
   8. Docs

3. **Item 4 (long-term, CUDA/OpenCL + binary)**
   1. `acceleration.cpp` — activate backends for `aqmh_reconstruction`, add the `cuda > opencv_opencl > cpu` priority rule for "auto" resolution (§4.1.4 arbitration note) — excluding `opencv_cuda`
   2. `aqmh_reconstruction_cuda.cu/.hpp` — CUDA kernel, **row-chunked** (not frame-batched — see §4.1.2 correctness constraint)
   3. `aqmh_reconstruction_opencl.cpp/.hpp` — OpenCL kernel, same row-chunking
   4. `CMakeLists.txt` — CUDA build support, extending the existing `TILE_COMPILE_WITH_CUDA` block
   5. `runner_phase_aqmh_reconstruction.cpp` — backend dispatch + `gpu_reconstruction: auto|force|disabled` rollout gate (§4.1.4), defaulting to `disabled` until validated
   6. `aqmh_binary_diagnostics.cpp/.hpp` — binary format
   7. `runner_phase_aqmh_diagnostics.cpp` — binary output
   8. `configuration.hpp` + `config.cpp` — verify already-added `format`, `binary_block_size_px`, and `gpu_reconstruction` fields are wired to the Item 4 runtime behavior
   9. YAML files + schema
   10. Tests, including the cross-dataset determinism validation required before flipping `gpu_reconstruction` to `auto` by default
   11. Docs
