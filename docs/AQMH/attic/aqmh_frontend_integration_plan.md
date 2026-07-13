# AQMH — Frontend Integration Investigation and Implementation Plan

**Version:** v0.2.0 (2026-06-07)
**Scope:** GUI2 / `web_frontend`, C++ web backend / `web_backend_cpp`, report integration  
**Related documents:**
- `docs/AQMH/aqmh_methodik_en.md`
- `docs/AQMH/aqmh_implementation_plan.md`

---

## 1. Executive Summary

AQMH and Classic Tile Compile are separate reconstruction methods. The frontend must treat them as independent methods that can share infrastructure, but not as variants of the same algorithm.

User-facing method choices:

```text
method = classic_tile_compile
method = aqmh
```

There is no combined AQMH/Classic method in the first implementation. AQMH must not expose Classic tile weighting or Classic fallback behavior as AQMH controls.

Shared frontend/backend infrastructure remains useful:

- input scan
- configuration editing and validation
- run start API
- run directory/history model
- artifact list and report generation
- common log/event transport

AQMH-specific frontend behavior is required for:

- method selection
- AQMH config controls
- AQMH run monitor stages
- AQMH cache and memory warnings
- AQMH artifacts and reports

---

## 2. Current UI Investigation

### 2.1 Frontend Surfaces

| UI surface | File | Current responsibility | AQMH impact |
|---|---|---|---|
| Dashboard | `web_frontend/index.html`, `web_frontend/src/app.js` | Start overview, queue, validation, pipeline preview | Add method badge/selector and AQMH storage estimate |
| Input & Scan | `web_frontend/input-scan.html` | Input directory, scan options, scan summary | No input-file changes; optionally show AQMH cache estimate after scan |
| Parameter Studio | `web_frontend/parameter-studio.html` | Full config editor | Add AQMH category and method-aware field visibility |
| Wizard | `web_frontend/wizard.html` | Guided setup | Add explicit method choice: Classic vs AQMH |
| Run Monitor | `web_frontend/run-monitor.html` | Phase progress, logs, artifacts, resume controls | Show method-specific stages/status |
| History / comparison | `web_frontend/history-tools.html` | Run selection and comparison | Tag runs by method and expose AQMH metrics |
| Report generation | `tile_compile_cpp/scripts/generate_report.py` | HTML report from artifacts | Add independent AQMH report section |

### 2.2 Backend Status Model

Status is derived from run directory events and artifacts in:

- `web_backend_cpp/src/services/run_inspector.cpp`
- `web_backend_cpp/include/services/run_inspector.hpp`

The current canonical phase list is Classic-oriented:

```text
SCAN_INPUT
CHANNEL_SPLIT
NORMALIZATION
GLOBAL_METRICS
TILE_GRID
REGISTRATION
PREWARP
COMMON_OVERLAP
LOCAL_METRICS
TILE_RECONSTRUCTION
STATE_CLUSTERING
SYNTHETIC_FRAMES
STACKING
DEBAYER
ASTROMETRY
BGE
PCC
HYPERMETRIC_STRETCH
```

The frontend mirrors this with `RUN_MONITOR_PHASE_ORDER` in `web_frontend/src/app.js` and dashboard grouping via `DASHBOARD_PIPELINE_GROUPS`.

### 2.3 Consequence

AQMH should not be hidden as “extra detail” inside Classic `LOCAL_METRICS` and `TILE_RECONSTRUCTION` rows. That would misrepresent the method and create the false impression that AQMH depends on Classic tile metrics.

The correct model is:

- shared preprocessing stages may be displayed the same way
- method-specific reconstruction stages must be displayed according to the selected method
- AQMH stages should be named AQMH stages in status, monitor, history, and reports

---

## 3. Method Boundary

### 3.1 What Is Shared

The following can remain common UI/backend infrastructure:

- input scan and validation
- FITS/calibration file handling
- registration/prewarp status
- common-overlap/canvas mask status
- run start/resume mechanics
- artifact listing
- report-generation trigger
- history storage

### 3.2 What Is Different

| Area | Classic Tile Compile | AQMH |
|---|---|---|
| Local quality model | block/tile scalar quality | dense per-pixel quality map |
| Reconstruction weight | Classic local/tile weight | `G_f * Q_map_f(x,y)` |
| Missing AQMH maps | not applicable | unsupported/zero AQMH output plus warning |
| Combined AQMH/Classic mode | not part of AQMH | not implemented |
| Cache pressure | mainly frames/intermediates | frame cache plus AQMH map cache |
| Diagnostics | local/tile metrics | map statistics, regions, cache stats |
| Report heatmaps | Classic local metrics | AQMH quality/artifact heatmaps |

### 3.3 What Must Not Happen

- No AQMH tile-weighting mode.
- No combined AQMH/Classic reconstruction mode.
- No AQMH fallback to Classic tile weights.
- No automatic fallback from AQMH reconstruction to Classic tile weights.
- No UI label implying AQMH is “Classic plus dense maps”.

---

## 4. Configuration Model

### 4.1 Top-Level Method

Add a top-level method selector in config:

```yaml
method: classic_tile_compile  # classic_tile_compile | aqmh
```

Alternative if a top-level method key is too invasive for the first implementation:

```yaml
aqmh:
  enabled: false
```

Frontend interpretation must still be method-based:

```text
aqmh.enabled = false -> method classic_tile_compile
aqmh.enabled = true  -> method aqmh
```

### 4.2 AQMH Config

AQMH config exposed to the UI:

```yaml
aqmh:
  enabled: false
  pyramid:
    scales: 4
    base_window_px: 4
    w_sharp: 0.6
    w_snr: 0.4
    k_artifact: 5.0
    frac_artifact_max: 0.10
  storage:
    resolution_divisor: 2
    dtype: uint16
    max_resident_maps: 2
  cherry_pick:
    enabled: false
    k_min: 3
    k_frac: 0.30
  diagnostics:
    tau_artifact: 0.20
    q_region: 0.75
    r_morph_canvas_px: 6
```

There is no AQMH reconstruction mode field in the first implementation. AQMH reconstruction always means AQMH dense-map weighted reconstruction.

### 4.3 Validation Contract

The frontend should rely on backend/schema validation for numeric constraints:

- `aqmh.pyramid.scales in [1,8]`
- `aqmh.storage.resolution_divisor in {1,2,4}`
- `aqmh.storage.dtype in {"float32","uint16","uint8"}`
- `aqmh.storage.max_resident_maps in [0,16]`
- `aqmh.cherry_pick.k_min >= 1`
- `aqmh.cherry_pick.k_frac in (0,1]`
- `aqmh.diagnostics.tau_artifact in [0,1]`
- `aqmh.diagnostics.q_region in [0,1]`
- `aqmh.diagnostics.r_morph_canvas_px >= 1`

---

## 5. Frontend Implementation Plan

### Milestone F1 — Method Selection Contract

**Goal:** The UI and backend can identify whether a run is Classic or AQMH.

Backend contract:

```json
{
  "method": "aqmh",
  "aqmh": {
    "enabled": true
  }
}
```

Rules:

- Missing method on old runs means `classic_tile_compile`.
- `aqmh.enabled=true` means `method=aqmh`.
- The frontend must not infer AQMH from the existence of cache files alone.

### Milestone F2 — Parameter Studio

**Goal:** Make AQMH editable without mixing it into Classic local/tile controls.

Implementation:

1. Add a method/category selector:
   - `Classic Tile Compile`
   - `AQMH`
2. Add AQMH category button when schema contains `aqmh`.
3. Group AQMH fields:

| Group | Fields |
|---|---|
| Core | `enabled` or top-level `method` |
| Storage | `resolution_divisor`, `dtype`, `max_resident_maps` |
| Pyramid | `scales`, `base_window_px`, `w_sharp`, `w_snr`, `k_artifact`, `frac_artifact_max` |
| Diagnostics | `tau_artifact`, `q_region`, `r_morph_canvas_px` |
| Experimental | `cherry_pick.*` |

4. Hide Classic local/tile-only controls when the user is editing an AQMH-only preset, unless they are still needed by shared preprocessing or report-block layout.
5. Do not show combined AQMH/Classic modes, tile-weighting modes, or Classic fallback controls.

### Milestone F3 — Dashboard

**Goal:** Make the selected method obvious before a run starts.

Display:

| Field | Classic | AQMH |
|---|---|---|
| Method badge | `Classic Tile Compile` | `AQMH` |
| Quality model | local/block scalar | dense quality maps |
| Reconstruction | Classic weighted stack | AQMH pixel-wise weighted stack |
| Cache estimate | normal run cache | AQMH map cache estimate |

AQMH cache estimate:

```text
stored_width  = ceil(width / resolution_divisor)
stored_height = ceil(height / resolution_divisor)
bytes_per_map = stored_width * stored_height * dtype_bytes
total_cache   = bytes_per_map * frame_count * map_stream_count
```

First implementation can assume `map_stream_count = 1` (`luma`).

Warnings:

- `resolution_divisor = 1` and many frames: large disk usage.
- `max_resident_maps > 4`: possible RAM pressure.
- `cherry_pick.enabled = true`: pixel-level frame selection active.

### Milestone F4 — Wizard

**Goal:** Provide an explicit, safe method choice.

Wizard step:

```text
Reconstruction method
(*) Classic Tile Compile
( ) AQMH
```

If AQMH is selected:

```text
AQMH storage
(*) Conservative: resolution_divisor=2, float32, max_resident_maps=2
( ) Full resolution: resolution_divisor=1, float32
```

Do not enable AQMH automatically. Do not present any combined AQMH/Classic mode as an option.

### Milestone F5 — Run Monitor

**Goal:** Show method-specific progress without pretending AQMH is Classic tile reconstruction.

Keep shared preprocessing rows where they are truly shared. For method-specific rows, use a method-aware display.

The Run Monitor must make cache-backed execution visible. AQMH is expected to process hundreds of frames; the UI must not imply that frame or map data is resident as a full run. Show per-stage cache/residency signals where available:

| Stage | Required UI signal |
|---|---|
| Shared preprocessing | frame-store/cache progress if available |
| AQMH maps | maps computed/written, bytes written, current worker count |
| AQMH reconstruction | resident maps observed/configured, frame/map cache hits, bytes read |
| AQMH diagnostics | summary artifact size, no raw-map inline rendering |

Recommended AQMH stage labels:

```text
AQMH_MAPS
AQMH_RECONSTRUCTION
AQMH_DIAGNOSTICS
```

If backend phase names cannot be changed immediately, the frontend may temporarily map existing runner events to AQMH labels:

| Backend event | AQMH display label |
|---|---|
| `LOCAL_METRICS` with `method=aqmh` | `AQMH_MAPS` |
| `TILE_RECONSTRUCTION` with `method=aqmh` | `AQMH_RECONSTRUCTION` |

This mapping is a compatibility layer only. The UI text should still say AQMH.

AQMH status payload:

```json
{
  "method": "aqmh",
  "aqmh": {
    "enabled": true,
    "storage": {
      "resolution_divisor": 2,
      "dtype": "uint16",
      "max_resident_maps": 2
    },
    "maps": {
      "computed": 143,
      "total": 300,
      "stream": "luma"
    },
    "cache": {
      "bytes_written": 7200000000,
      "bytes_read": 1800000000,
      "read_count": 1200,
      "cache_hits": 1100,
      "cache_misses": 100,
      "max_resident_maps_observed": 2
    }
  }
}
```

Run Monitor display:

- header badge: `AQMH`
- map progress: `AQMH maps 143/300`
- cache written/read
- cache hit rate
- resident maps observed vs configured
- warnings for cache misses and unsupported pixels

### Milestone F6 — Artifacts and Viewer

AQMH artifacts:

| Artifact | Purpose |
|---|---|
| `artifacts/aqmh_metrics.json` | Per-frame, block-level, cache, and timing diagnostics |
| `artifacts/aqmh_regions.json` | Optional quality-region diagnostics |
| `cache/aqmh/aqmh_luma_000000.bin` | Raw map cache, hidden/grouped by default |

Frontend behavior:

- Show friendly labels: `AQMH Metrics`, `AQMH Regions`.
- Group raw map cache files under `AQMH cache`.
- Avoid inline rendering of huge JSON arrays.
- Prefer backend-provided summaries for large artifacts.

### Milestone F7 — Report Integration

Add an independent AQMH report section.

Charts:

1. AQMH quality heatmap per report block.
2. AQMH artifact fraction heatmap per report block.
3. Per-frame `map_mean`.
4. Per-frame `artifact_frac`.
5. AQMH cache/timing table.
6. Optional AQMH-vs-Classic comparison only when both methods were run as separate runs.

The AQMH report section must not be nested under Classic local metrics.
If BGE ran after AQMH, show `tile_metrics_source = aqmh_output` as an AQMH-native BGE input source. Do not label these helpers as Classic local metrics.

### Milestone F8 — History and Comparison

History tags:

- `Classic Tile Compile`
- `AQMH`
- `AQMH cherry-pick`

Comparison fields:

| Metric | Source |
|---|---|
| method | status/config/artifact metadata |
| AQMH cache size | `aqmh_metrics.json.cache_stats` |
| AQMH map compute time | `aqmh_metrics.json.timing.map_compute_s` |
| AQMH reconstruction time | `aqmh_metrics.json.timing.aqmh_reconstruction_s` |
| mean artifact fraction | `aqmh_metrics.json.frames[].artifact_frac` |

AQMH-vs-Classic comparison is a cross-run comparison, not a single AQMH run mode.

---

## 6. Backend Event Contract

The runner should eventually emit AQMH-specific method stage events:

```json
{
  "type": "phase_progress",
  "phase_name": "AQMH_MAPS",
  "progress": 0.42,
  "payload": {
    "method": "aqmh",
    "aqmh_maps_done": 126,
    "aqmh_maps_total": 300,
    "aqmh_cache_bytes_written": 3020000000
  }
}
```

```json
{
  "type": "phase_progress",
  "phase_name": "AQMH_RECONSTRUCTION",
  "progress": 0.58,
  "payload": {
    "method": "aqmh",
    "aqmh_cache_hits": 2400,
    "aqmh_cache_misses": 180,
    "aqmh_max_resident_maps_observed": 2,
    "aqmh_unsupported_pixels": 0
  }
}
```

Compatibility rule:

- If the backend initially emits `LOCAL_METRICS` / `TILE_RECONSTRUCTION`, the frontend may remap the labels for AQMH runs.
- The payload must still include `method: "aqmh"` or equivalent status metadata.

---

## 7. UX Requirements

Use labels:

- `AQMH`
- `AQMH dense quality maps`
- `AQMH pixel-wise reconstruction`
- `AQMH cache`
- `Resident maps`
- `Pixel-level frame selection`

Avoid labels:

- `AQMH tile mode`
- `AQMH/Classic combined mode`
- `Classic fallback`
- `Dense map mode`

Do not expose internal mathematical symbols such as `Phi_snr`, `Psi_s`, or `P_actual` in normal UI. These belong in advanced diagnostics/reports.

---

## 8. Testing Plan

### 8.1 Frontend Tests

1. Method selector can switch between Classic and AQMH.
2. Parameter Studio shows AQMH fields only in AQMH category/method context.
3. No combined AQMH/Classic, tile-weighting, or Classic-fallback AQMH controls are rendered.
4. Dashboard badge shows `AQMH` when AQMH is enabled.
5. AQMH cache estimate is computed from scan dimensions and frame count.
6. Run Monitor maps AQMH status to AQMH labels.
7. Missing `aqmh` metadata on old runs displays as Classic/unknown without crashing.

### 8.2 Backend Contract Tests

1. Status endpoint includes `method`.
2. AQMH status includes `aqmh.maps`, `aqmh.cache`, and storage config when AQMH is enabled.
3. Artifact listing includes `aqmh_metrics.json` and `aqmh_regions.json`.
4. Raw AQMH cache files are grouped or hidden from the primary artifact list.
5. AQMH cache misses surface as AQMH warnings, not Classic fallback states.

### 8.3 Integration Tests

1. Classic run shows Classic method label and Classic phase labels.
2. AQMH run shows AQMH method label and AQMH stage labels.
3. AQMH run generates report with independent AQMH section.
4. Resume/retry of AQMH map computation updates AQMH map progress.
5. Resume/retry of AQMH reconstruction reuses valid AQMH cache metadata.

---

## 9. Implementation Order

1. Add method metadata to config/status.
2. Remove combined AQMH/Classic, tile-weighting, and Classic-fallback AQMH controls from schema/UI plans.
3. Add AQMH artifact contract.
4. Add status `aqmh` block.
5. Add Parameter Studio AQMH category.
6. Add Dashboard method badge and AQMH cache estimate.
7. Add Wizard method choice.
8. Add Run Monitor AQMH label mapping/status panel.
9. Add AQMH report section.
10. Add History/Comparison tags.

---

## 10. Explicit Non-Goals

Do not implement the following for the first frontend pass:

- Combined AQMH/Classic reconstruction.
- AQMH fallback to Classic Tile Compile.
- AQMH `tile` mode.
- A single run that silently switches between AQMH and Classic reconstruction.
- Direct visualization of raw full AQMH map cache files in the Run Monitor.
- Default-enabled cherry-pick.

---

## 11. Summary

AQMH should feel like a separate reconstruction method inside the same application, not like a Classic Tile Compile option.

The stable frontend contract is:

- shared input and run infrastructure
- explicit method selection
- independent AQMH configuration
- AQMH-specific monitor labels/status
- AQMH-specific artifacts and reports
- Classic only as a separately run comparison baseline
