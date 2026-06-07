# AQMH — Frontend Integration Investigation and Implementation Plan

**Version:** v0.1.0 (2026-06-06)  
**Scope:** GUI2 / `web_frontend`, C++ web backend / `web_backend_cpp`, report integration  
**Related documents:**
- `docs/AQMH/aqmh_methodik_en.md`
- `docs/AQMH/aqmh_implementation_plan.md`

---

## 1. Executive Summary

AQMH should be integrated into the GUI as an **optional mode of the existing Tile Compile pipeline**, not as a separate workflow and not as a separate top-level pipeline type.

The user-facing flow remains:

1. Input scan
2. Configuration / Parameter Studio
3. Run start
4. Run Monitor
5. Stats / report
6. Optional resume

AQMH changes the semantics and diagnostics of two existing phases:

- `LOCAL_METRICS`: computes classic tile metrics and, when enabled, AQMH dense quality maps.
- `TILE_RECONSTRUCTION`: reconstructs tiles with classic tile weights, AQMH dense-map weights, or AQMH hybrid weights depending on `aqmh.reconstruction.mode`.

The Run Monitor should **not** switch to a different phase list. It should show the same canonical phase sequence and add AQMH badges/substatus where applicable.

---

## 2. Investigation Results

### 2.1 Current Frontend Structure

The active GUI consists of static pages under `web_frontend/`, with most behavior centralized in:

- `web_frontend/src/app.js`
- `web_frontend/src/api.js`
- `web_frontend/src/constants.js`

Relevant pages:

| UI surface | File | Current responsibility |
|---|---|---|
| Dashboard | `web_frontend/index.html`, `web_frontend/src/app.js` | Start-oriented overview, input queue, validation guardrails, pipeline preview |
| Input & Scan | `web_frontend/input-scan.html` | Input directory selection, scan options, scan summary |
| Parameter Studio | `web_frontend/parameter-studio.html` | Full config editing, category-filtered dynamic parameter editor, validation |
| Wizard | `web_frontend/wizard.html` | Guided run setup and preset application |
| Run Monitor | `web_frontend/run-monitor.html` | Phase progress, log, artifacts, resume controls, stats/report actions |
| History / comparison | `web_frontend/history-tools.html` and shared app logic | Run selection, status comparison, report navigation |
| Report viewer/generation | `tile_compile_cpp/scripts/generate_report.py`, backend stats endpoints | HTML report generation from artifacts |

### 2.2 Current Backend Status Model

Run status is derived from run directory events and artifacts in:

- `web_backend_cpp/src/services/run_inspector.cpp`
- `web_backend_cpp/include/services/run_inspector.hpp`

The canonical phase list is hard-coded in `PHASE_ORDER`:

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

The frontend has a matching `RUN_MONITOR_PHASE_ORDER` in `web_frontend/src/app.js`. It also groups dashboard pipeline phases via `DASHBOARD_PIPELINE_GROUPS`.

### 2.3 Consequence for AQMH

Adding new top-level phases such as `AQMH_MAPS` or `DENSE_RECONSTRUCTION` would require coordinated changes in:

- backend `PHASE_ORDER`
- frontend `RUN_MONITOR_PHASE_ORDER`
- dashboard grouping
- resume phase handling
- progress computation
- i18n phase labels
- tests for status/resume/progress

This is unnecessary. AQMH is best represented as **substatus inside existing phases**.

---

## 3. What Stays the Same

### 3.1 Pipeline Flow

The frontend should keep the same pipeline flow and phase list for classic and AQMH runs.

No changes are needed to the high-level user journey:

- scan input
- edit/validate config
- start run
- monitor run
- generate stats/report
- resume from existing phases

### 3.2 Input & Scan

AQMH does not require different input files, calibration files, FITS selection, Bayer selection, color mode detection, or scan rules.

The Input & Scan page should remain unchanged for the first implementation. Optional later enhancement: after a scan, show an AQMH storage estimate if `aqmh.enabled = true` in the current config.

### 3.3 Run Start API

Run start should continue to use the existing backend route:

```text
POST /api/runs/start
```

The only AQMH-specific input is the YAML config content already passed to the backend.

### 3.4 Resume Semantics

Resume should continue to use existing phases. AQMH recomputation follows the same affected phase boundaries:

- resume from `LOCAL_METRICS` to recompute AQMH maps and local metrics
- resume from `TILE_RECONSTRUCTION` to reuse maps and rerun reconstruction

No new resume target is required for AQMH.

---

## 4. What Changes With AQMH

### 4.1 Configuration

The config gains:

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
    dtype: float32
    max_resident_maps: 2
  reconstruction:
    mode: dense_map
    fallback_to_tile: true
  cherry_pick:
    enabled: false
    k_min: 3
    k_frac: 0.30
  diagnostics:
    tau_artifact: 0.20
    q_region: 0.75
    r_morph_canvas_px: 6
```

The frontend should not manually hard-code all AQMH fields if the dynamic Parameter Studio can read them from schema/defaults. Hard-coded UI should be limited to curated convenience controls in Dashboard/Wizard.

### 4.2 Artifacts

AQMH adds optional artifacts:

| Artifact | Purpose |
|---|---|
| `aqmh_metrics.json` | Per-frame and per-tile dense-map diagnostics |
| `aqmh_regions.json` | Optional binary quality region diagnostics |
| `cache/aqmh/aqmh_luma_000000.bin` | Map cache files, not usually opened directly by users |
| AQMH report charts | Rendered by stats/report generation |

### 4.3 Monitoring

AQMH should enrich the same phase rows:

| Phase | Classic display | AQMH display |
|---|---|---|
| `LOCAL_METRICS` | tile quality metrics | tile metrics + AQMH map compute/cache write progress |
| `TILE_RECONSTRUCTION` | tile-weighted reconstruction | dense/hybrid/tile AQMH mode, cache read stats, resident-map bound |

### 4.4 Runtime Cost Expectations

The UI should set expectations:

- `LOCAL_METRICS` may become significantly slower because it computes dense maps.
- `TILE_RECONSTRUCTION` may become slower because it performs per-pixel map lookups.
- Disk usage increases because quality maps are cached.
- RAM must remain bounded by `aqmh.storage.max_resident_maps`.

---

## 5. Frontend Implementation Plan

### Milestone F1 — Schema, Defaults, and Validation Contract

**Goal:** Make AQMH config visible and valid through existing config APIs.

Required backend/CLI prerequisites:

1. Add `aqmh.*` to `tile_compile_cpp/tile_compile.yaml`.
2. Add `aqmh.*` to `tile_compile_cpp/tile_compile.schema.yaml`.
3. Add `aqmh.*` to generated/embedded JSON schema.
4. Ensure `tile_compile_cli dump-default-config` includes AQMH defaults.
5. Ensure `tile_compile_cli validate-config` validates:
   - `aqmh.pyramid.scales in [1,8]`
   - `aqmh.storage.resolution_divisor in {1,2,4}`
   - `aqmh.storage.dtype in {"float32","uint8"}` initially
   - `aqmh.storage.max_resident_maps in [0,16]`
   - `aqmh.reconstruction.mode in {"dense_map","tile","hybrid"}`
   - `aqmh.diagnostics.tau_artifact in [0,1]`
   - `aqmh.diagnostics.q_region in [0,1]`
   - `aqmh.diagnostics.r_morph_canvas_px >= 1`

Frontend impact:

- Existing `/api/config/schema`, `/api/config/defaults`, `/api/config/validate` continue to work.
- Parameter Studio can render AQMH fields once schema/defaults exist.

### Milestone F2 — Parameter Studio AQMH Category

**Goal:** Make AQMH editable in the full parameter UI.

Files:

- `web_frontend/parameter-studio.html`
- `web_frontend/src/app.js`
- `web_frontend/i18n/de.json`
- `web_frontend/i18n/en.json`

Implementation:

1. Add category button:

```html
<button id="parameter-category-aqmh"
        data-control="parameter.category.aqmh"
        data-category="aqmh">
  AQMH
</button>
```

2. Ensure dynamic editor groups `aqmh.*` under the AQMH category.
3. Add search keywords:
   - dense map
   - quality map
   - AQMH
   - artifact
   - cache
   - resident maps
4. Add field help text for high-risk fields:
   - `aqmh.enabled`
   - `aqmh.storage.resolution_divisor`
   - `aqmh.storage.max_resident_maps`
   - `aqmh.reconstruction.mode`
   - `aqmh.cherry_pick.enabled`

Recommended UI grouping:

| Group | Fields |
|---|---|
| Core | `enabled`, `reconstruction.mode`, `fallback_to_tile` |
| Storage | `resolution_divisor`, `dtype`, `max_resident_maps` |
| Pyramid | `scales`, `base_window_px`, `w_sharp`, `w_snr`, `k_artifact`, `frac_artifact_max` |
| Diagnostics | `tau_artifact`, `q_region`, `r_morph_canvas_px` |
| Experimental | `cherry_pick.*` |

### Milestone F3 — Dashboard AQMH Summary

**Goal:** Surface AQMH state without making the dashboard more complex.

Files:

- `web_frontend/index.html`
- `web_frontend/src/app.js`

Add a compact dashboard panel or status row:

| Field | Display |
|---|---|
| AQMH state | `Off`, `dense_map`, `hybrid`, `tile` |
| Storage | `1/4-area float32`, `full float32`, etc. |
| Memory bound | `max resident maps: N` |
| Estimated cache | after scan, approximate `map_bytes * frame_count` |

Behavior:

- If `aqmh.enabled = false`, show a quiet `AQMH off` badge.
- If `aqmh.enabled = true`, show an active badge and storage warning if needed.
- If scan data has frame count / dimensions, compute approximate cache size.

Cache estimate:

```text
stored_width  = ceil(width / resolution_divisor)
stored_height = ceil(height / resolution_divisor)
bytes_per_map = stored_width * stored_height * dtype_bytes
total_cache   = bytes_per_map * frame_count * map_stream_count
```

First implementation can assume `map_stream_count = 1` (`luma`).

Guardrail warnings:

- `resolution_divisor = 1` and many frames: warn about disk usage.
- `max_resident_maps > 4`: warn about RAM.
- `cherry_pick.enabled = true`: warn that pixel-level frame selection is active.

### Milestone F4 — Wizard AQMH Step

**Goal:** Provide a safe guided way to enable AQMH.

Files:

- `web_frontend/wizard.html`
- `web_frontend/src/app.js`

Add an optional advanced step:

```text
Quality weighting
[ ] Enable AQMH dense quality maps

Mode:
(*) dense_map  [recommended for artifact suppression]
( ) hybrid     [diagnostic/A-B; does not veto artifacts below tile baseline]
( ) tile       [classic reconstruction while still computing AQMH diagnostics]

Storage:
(*) Conservative: resolution_divisor=2, float32, max_resident_maps=2
( ) Full resolution: resolution_divisor=1, float32
```

Do not enable AQMH automatically. The user should explicitly opt in.

### Milestone F5 — Run Monitor AQMH Awareness

**Goal:** Keep the same phase UI, add AQMH-specific detail.

Files:

- `web_frontend/run-monitor.html`
- `web_frontend/src/app.js`
- `web_backend_cpp/src/services/run_inspector.cpp`
- `web_backend_cpp/include/services/run_inspector.hpp`

Do not change:

- `PHASE_ORDER`
- `RUN_MONITOR_PHASE_ORDER`
- `DASHBOARD_PIPELINE_GROUPS`

Add optional AQMH status payload to `/api/runs/:id/status`:

```json
{
  "aqmh": {
    "enabled": true,
    "mode": "dense_map",
    "storage": {
      "resolution_divisor": 2,
      "dtype": "float32",
      "max_resident_maps": 2
    },
    "cache": {
      "bytes_written": 7200000000,
      "bytes_read": 1800000000,
      "read_count": 1200,
      "cache_hits": 1100,
      "cache_misses": 100,
      "max_resident_maps_observed": 2
    },
    "maps": {
      "computed": 143,
      "total": 300,
      "stream": "luma"
    }
  }
}
```

The backend can derive this from:

- `config.yaml`
- `artifacts/aqmh_metrics.json`
- cache directory stats
- phase event payloads

Frontend display:

1. Header badge:
   - `AQMH off`
   - `AQMH dense_map`
   - `AQMH hybrid`
   - `AQMH tile`
2. `LOCAL_METRICS` row subtext:
   - `AQMH maps 143/300`
   - `cache written 3.4 GB`
3. `TILE_RECONSTRUCTION` row subtext:
   - `dense_map`
   - `cache hits 91%`
   - `resident maps 2/2`
4. Artifact list:
   - show `aqmh_metrics.json`
   - show `aqmh_regions.json`
   - hide raw map cache files by default or group them under “Cache”.

### Milestone F6 — Artifacts and Viewer

**Goal:** Make AQMH artifacts easy to inspect.

Existing artifact list is populated by:

- `GET /api/runs/:id/artifacts`
- `GET /api/runs/:id/artifacts/view`

Implementation:

1. Ensure backend artifact listing includes:
   - `artifacts/aqmh_metrics.json`
   - `artifacts/aqmh_regions.json`
2. Add friendly labels in the frontend:
   - `AQMH Metrics`
   - `AQMH Regions`
3. For large AQMH metrics files, avoid rendering huge arrays inline if they exceed current viewer limits.
4. Optionally add summary extraction in backend:
   - mode
   - map count
   - artifact fraction p50/p90
   - cache bytes

### Milestone F7 — Report Integration

**Goal:** Add AQMH diagnostics to generated stats/report.

Files:

- `tile_compile_cpp/scripts/generate_report.py`
- frontend report i18n files

Add `_gen_aqmh_metrics(artifacts_dir, aqmh, tile_grid)`:

Charts:

1. Mean AQMH quality per tile
2. Artifact fraction per tile
3. `aqmh_vs_tile_delta` heatmap
4. Per-frame `map_mean`
5. Per-frame `artifact_frac`
6. Optional cache/timing table

Behavior:

- If `aqmh_metrics.json` is absent, silently skip section.
- If present, add section after Local Metrics or Reconstruction.
- If `aqmh_regions.json` exists, add region count summary.

### Milestone F8 — History / Run Comparison

**Goal:** Make AQMH runs identifiable in history and comparison tools.

Display tags:

- `classic`
- `AQMH dense_map`
- `AQMH hybrid`
- `AQMH tile diagnostics`

Comparison fields:

| Metric | Source |
|---|---|
| AQMH enabled/mode | `config.yaml` or `aqmh_metrics.json.config` |
| cache size | `aqmh_metrics.json.cache_stats` |
| map compute time | `aqmh_metrics.json.timing.map_compute_s` |
| dense reconstruction time | `aqmh_metrics.json.timing.dense_reconstruction_s` |
| mean artifact fraction | `aqmh_metrics.json.frames[].artifact_frac` |

---

## 6. Backend Contract for Frontend

### 6.1 Status Endpoint

Extend existing status output with optional `aqmh`.

Rules:

- Missing `aqmh` means classic or unknown old run.
- `aqmh.enabled = false` should be shown as classic.
- Frontend must not fail if fields are missing.

### 6.2 Events

The runner should emit ordinary phase events, not new phases.

Recommended additional payload fields:

For `LOCAL_METRICS`:

```json
{
  "type": "phase_progress",
  "phase_name": "LOCAL_METRICS",
  "progress": 0.42,
  "payload": {
    "aqmh_enabled": true,
    "aqmh_maps_done": 126,
    "aqmh_maps_total": 300,
    "aqmh_cache_bytes_written": 3020000000
  }
}
```

For `TILE_RECONSTRUCTION`:

```json
{
  "type": "phase_progress",
  "phase_name": "TILE_RECONSTRUCTION",
  "progress": 0.58,
  "payload": {
    "aqmh_enabled": true,
    "aqmh_mode": "dense_map",
    "aqmh_cache_hits": 2400,
    "aqmh_cache_misses": 180,
    "aqmh_max_resident_maps_observed": 2
  }
}
```

Existing Run Monitor log formatting can summarize these payloads later, but the first version can simply surface them in status/artifact summaries.

### 6.3 Artifacts

`aqmh_metrics.json` should include:

```json
{
  "schema_version": 1,
  "config": {
    "storage": {},
    "pyramid": {},
    "reconstruction": {},
    "cherry_pick": {},
    "diagnostics": {}
  },
  "frames": [],
  "tiles": [],
  "cache_stats": {},
  "timing": {}
}
```

This lets the frontend and report generator avoid re-parsing `config.yaml` for every summary.

---

## 7. UX Requirements

### 7.1 Copy and Labels

Use user-facing labels:

- “AQMH dense quality maps”
- “Dense map weighting”
- “Hybrid AQMH weighting”
- “Classic tile weighting”
- “AQMH cache”
- “Resident maps”

Avoid exposing method terms such as `Phi_snr`, `Psi_s`, or `P_actual` in normal UI. These belong in advanced diagnostics/report.

### 7.2 Warnings

Show warnings for:

- `cherry_pick.enabled = true`
  - text: “Pixel-level frame selection active.”
- `resolution_divisor = 1` with large frame counts
  - text: “Full-resolution AQMH maps may require significant disk space.”
- `max_resident_maps` high enough to exceed likely RAM budget
  - text: “Resident map cache may increase memory use.”

### 7.3 Defaults

Recommended safe UI defaults:

```yaml
aqmh:
  enabled: false
  reconstruction:
    mode: dense_map
    fallback_to_tile: true
  storage:
    resolution_divisor: 2
    dtype: float32
    max_resident_maps: 2
```

Do not default-enable AQMH until validation datasets confirm stable behavior.

---

## 8. Testing Plan

### 8.1 Frontend Unit/DOM Tests

Where existing test infrastructure allows:

1. Parameter Studio renders AQMH category when schema contains `aqmh`.
2. AQMH search terms find AQMH fields.
3. Dashboard badge changes from `AQMH off` to `AQMH dense_map`.
4. Run Monitor renders AQMH status block when status contains `aqmh`.
5. Run Monitor remains classic when status lacks `aqmh`.

### 8.2 Backend Contract Tests

Add tests in `web_backend_cpp/tests`:

1. Status endpoint returns classic-compatible JSON when no AQMH artifacts exist.
2. Status endpoint includes `aqmh.enabled/mode` when config enables AQMH.
3. Artifact listing includes `aqmh_metrics.json` and `aqmh_regions.json`.
4. AQMH cache files are not spammed as primary user artifacts, or are grouped/filtered if listed.

### 8.3 Integration Tests

1. Classic run still shows unchanged phase list.
2. AQMH run shows same phase list plus AQMH badges.
3. AQMH run can generate report with AQMH section.
4. Resume from `LOCAL_METRICS` recomputes AQMH diagnostics.
5. Resume from `TILE_RECONSTRUCTION` reuses existing AQMH cache if metadata matches.

---

## 9. Implementation Order

Recommended order:

1. Backend/CLI schema + config validation for `aqmh.*`
2. `aqmh_metrics.json` and `aqmh_regions.json` artifact contract
3. `/api/runs/:id/status` optional `aqmh` block
4. Parameter Studio AQMH category
5. Run Monitor AQMH badge/substatus
6. Dashboard AQMH summary and storage estimate
7. Wizard AQMH guided toggle
8. Report generator AQMH section
9. History/comparison AQMH tags

This order keeps the frontend dependent on stable backend contracts rather than duplicating inference logic.

---

## 10. Explicit Non-Goals

Do not implement the following for the first frontend pass:

- A separate AQMH run type.
- New top-level phases in the Run Monitor.
- A separate AQMH-only start button.
- Direct visualization of full AQMH map cache files in the Run Monitor.
- Default-enabled AQMH.
- Default-enabled cherry-pick.

---

## 11. Summary

AQMH should feel like a more advanced quality-weighting mode inside Tile Compile, not like a second application.

The stable frontend contract is:

- same input flow
- same phase flow
- same run start
- same resume model
- extra AQMH config
- extra AQMH diagnostics
- extra AQMH report section

The only automatic Run Monitor “switch” should be visual: when `aqmh.enabled` is detected, show AQMH-specific badges and submetrics inside the existing `LOCAL_METRICS` and `TILE_RECONSTRUCTION` phases.
