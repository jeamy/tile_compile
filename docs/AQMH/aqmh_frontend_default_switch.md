# AQMH Frontend Default Switch — Full Migration Plan

**Version:** v0.1.0 (2026-06-12)  
**Scope:** GUI2 / `web_frontend`, `web_backend_cpp`, config defaults, documentation  
**Prerequisite:** `aqmh_frontend_integration_plan.md` milestones F1–F8 implemented  
**Related documents:**
- `docs/AQMH/aqmh_frontend_integration_plan.md`
- `docs/AQMH/aqmh_implementation_plan.md`

---

## 1. Summary

This document defines all changes required to make AQMH the default reconstruction method in Tile Compile. Classic Tile Compile remains fully available as an explicit user choice.

Before this switch, the plan explicitly listed "Default-enabled AQMH" as a non-goal. That constraint is lifted here. All other non-goals from Section 10 of the integration plan remain in effect:

- No combined AQMH/Classic reconstruction.
- No AQMH fallback to Classic.
- No AQMH tile mode.
- No silent method switching within a single run.
- No default-enabled cherry-pick.

---

## 2. Affected Files

### Config / Schema

| File | Change |
|---|---|
| `tile_compile_cpp/tile_compile.yaml` | `aqmh.enabled: false` → `aqmh.enabled: true` |
| `tile_compile_cpp/tile_compile.schema.yaml` | Update description for `aqmh.enabled` default |
| `tile_compile_cpp/tile_compile.schema.json` | Same — update default annotation |
| `tile_compile_cpp/examples/aqmh_enabled.example.yaml` | Remove — this is now the default config; the example becomes `classic_mode.example.yaml` |

### Documentation

| File | Change |
|---|---|
| `docs/AQMH/aqmh_frontend_integration_plan.md` | Remove "Default-enabled AQMH" from Section 10 Non-Goals; update Section 4.1 default comment |
| `docs/AQMH/aqmh_implementation_plan.md` | Update Step 1.1 — `AqmhConfig::enabled` default changes to `true` |

### Frontend

| File | Change |
|---|---|
| `web_frontend/wizard.html` | AQMH is first and selected by default; Classic is second option |
| `web_frontend/src/app.js` | `RUN_MONITOR_PHASE_ORDER`, `DASHBOARD_PIPELINE_GROUPS`, default method badge |
| `web_frontend/parameter-studio.html` | AQMH category opens by default; Classic controls hidden unless Classic is selected |
| `web_frontend/index.html` | Dashboard method badge default: `AQMH` |

---

## 3. Config Changes

### 3.1 `tile_compile.yaml`

```yaml
# Before
aqmh:
  enabled: false

# After
aqmh:
  enabled: true
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

Classic mode is selected by setting `aqmh.enabled: false` explicitly. No other config change is required to run Classic.

### 3.2 `AqmhConfig` Struct Default (C++)

**File:** `tile_compile_cpp/include/tile_compile/config/configuration.hpp`

```cpp
// Before
struct AqmhConfig {
    bool enabled = false;
    ...
};

// After
struct AqmhConfig {
    bool enabled = true;
    ...
};
```

This change affects any code path that constructs `AqmhConfig` without reading a YAML file. All such paths must be audited before merging — particularly test fixtures and synthetic run builders that do not load a real config.

### 3.3 Example Config

Delete `tile_compile_cpp/examples/aqmh_enabled.example.yaml` (this is now the default).

Create `tile_compile_cpp/examples/classic_mode.example.yaml`:

```yaml
# Classic Tile Compile mode
# AQMH is the default. Set aqmh.enabled: false to use Classic Tile Compile.
aqmh:
  enabled: false
```

---

## 4. Frontend Changes

### 4.1 Wizard (`web_frontend/wizard.html`)

**Current state:**

```html
<label>
  <input type="radio" name="method" value="classic_tile_compile" checked>
  Classic Tile Compile
</label>
<label>
  <input type="radio" name="method" value="aqmh">
  AQMH
</label>
```

**After:**

```html
<label>
  <input type="radio" name="method" value="aqmh" checked>
  AQMH
</label>
<label>
  <input type="radio" name="method" value="classic_tile_compile">
  Classic Tile Compile
</label>
```

The AQMH storage sub-section must be visible immediately on wizard open, not deferred until the user selects AQMH:

```html
<!-- Storage section: shown by default, hidden when Classic is selected -->
<div id="wizard-aqmh-storage" class="wizard-method-section">
  <label>
    <input type="radio" name="aqmh_storage" value="conservative" checked>
    Conservative: resolution_divisor=2, uint16, max_resident_maps=2
  </label>
  <label>
    <input type="radio" name="aqmh_storage" value="full_res">
    Full resolution: resolution_divisor=1, float32
  </label>
</div>
```

JavaScript: show `#wizard-aqmh-storage` when `method=aqmh`, hide when `method=classic_tile_compile`.

### 4.2 Dashboard (`web_frontend/src/app.js` and `web_frontend/index.html`)

#### Method badge

The dashboard must derive the active method from the loaded config. If no config is loaded yet, default to `AQMH` (matching the new config default).

```js
// Before
function getMethodLabel(config) {
  return config?.aqmh?.enabled ? 'AQMH' : 'Classic Tile Compile';
}

// After — no change needed IF config is always loaded before render.
// If no config is loaded, the fallback changes:
function getMethodLabel(config) {
  if (!config) return 'AQMH';  // was: 'Classic Tile Compile'
  return config.aqmh?.enabled !== false ? 'AQMH' : 'Classic Tile Compile';
}
```

The condition `!== false` is intentional: `aqmh.enabled` absent from config → AQMH (new default), `aqmh.enabled: false` → Classic, `aqmh.enabled: true` → AQMH.

#### AQMH cache estimate

The cache estimate panel must be visible as the primary post-scan summary, not hidden behind an AQMH toggle:

```js
// Show cache estimate immediately after scan completes if method is AQMH (default)
function onScanComplete(scanResult, config) {
  const method = getMethodLabel(config);
  if (method === 'AQMH') {
    renderAqmhCacheEstimate(scanResult, config.aqmh.storage);
  }
}
```

Cache estimate formula (unchanged from integration plan §5.3):

```js
function computeAqmhCacheBytes(width, height, frameCount, storage) {
  const storedW = Math.ceil(width / storage.resolution_divisor);
  const storedH = Math.ceil(height / storage.resolution_divisor);
  const dtypeBytes = { float32: 4, uint16: 2, uint8: 1 }[storage.dtype] ?? 4;
  const bytesPerMap = storedW * storedH * dtypeBytes;
  return bytesPerMap * frameCount;  // map_stream_count=1 (luma) for first impl
}
```

#### Dashboard warnings

These warnings are shown by default (not after opt-in):

| Condition | Warning |
|---|---|
| `resolution_divisor = 1` and `frameCount > 50` | Large AQMH cache expected — consider `resolution_divisor: 2` |
| `max_resident_maps > 4` | High resident map count — possible RAM pressure |
| `cherry_pick.enabled = true` | Pixel-level frame selection active |

### 4.3 Run Monitor (`web_frontend/src/app.js`)

#### `RUN_MONITOR_PHASE_ORDER`

AQMH-specific stages are inserted as primary stages. Classic stages are listed but only rendered when `method=classic_tile_compile`.

```js
// Primary phase order for AQMH runs (new default)
const RUN_MONITOR_PHASE_ORDER_AQMH = [
  'SCAN_INPUT',
  'CHANNEL_SPLIT',
  'NORMALIZATION',
  'GLOBAL_METRICS',
  'TILE_GRID',
  'REGISTRATION',
  'PREWARP',
  'COMMON_OVERLAP',
  'AQMH_MAPS',
  'AQMH_RECONSTRUCTION',
  'AQMH_DIAGNOSTICS',
  'SYNTHETIC_FRAMES',
  'STACKING',
  'DEBAYER',
  'ASTROMETRY',
  'BGE',
  'PCC',
  'HYPERMETRIC_STRETCH',
];

// Phase order for Classic runs
const RUN_MONITOR_PHASE_ORDER_CLASSIC = [
  'SCAN_INPUT',
  'CHANNEL_SPLIT',
  'NORMALIZATION',
  'GLOBAL_METRICS',
  'TILE_GRID',
  'REGISTRATION',
  'PREWARP',
  'COMMON_OVERLAP',
  'LOCAL_METRICS',
  'TILE_RECONSTRUCTION',
  'STATE_CLUSTERING',
  'SYNTHETIC_FRAMES',
  'STACKING',
  'DEBAYER',
  'ASTROMETRY',
  'BGE',
  'PCC',
  'HYPERMETRIC_STRETCH',
];

function getPhaseOrder(method) {
  return method === 'classic_tile_compile'
    ? RUN_MONITOR_PHASE_ORDER_CLASSIC
    : RUN_MONITOR_PHASE_ORDER_AQMH;
}
```

Compatibility mapping (unchanged from integration plan §5, now applied by default):

```js
function remapPhaseForDisplay(phaseName, method) {
  if (method !== 'aqmh') return phaseName;
  if (phaseName === 'LOCAL_METRICS') return 'AQMH_MAPS';
  if (phaseName === 'TILE_RECONSTRUCTION') return 'AQMH_RECONSTRUCTION';
  return phaseName;
}
```

#### `DASHBOARD_PIPELINE_GROUPS`

```js
// Before: Classic-oriented groups
const DASHBOARD_PIPELINE_GROUPS = {
  preprocessing: ['SCAN_INPUT', 'CHANNEL_SPLIT', 'NORMALIZATION', ...],
  reconstruction: ['LOCAL_METRICS', 'TILE_RECONSTRUCTION', ...],
  ...
};

// After: method-aware groups
function getPipelineGroups(method) {
  const shared = {
    preprocessing: ['SCAN_INPUT', 'CHANNEL_SPLIT', 'NORMALIZATION',
                    'GLOBAL_METRICS', 'TILE_GRID', 'REGISTRATION',
                    'PREWARP', 'COMMON_OVERLAP'],
    postprocessing: ['SYNTHETIC_FRAMES', 'STACKING', 'DEBAYER',
                     'ASTROMETRY', 'BGE', 'PCC', 'HYPERMETRIC_STRETCH'],
  };
  if (method === 'classic_tile_compile') {
    return {
      ...shared,
      reconstruction: ['LOCAL_METRICS', 'TILE_RECONSTRUCTION', 'STATE_CLUSTERING'],
    };
  }
  return {
    ...shared,
    reconstruction: ['AQMH_MAPS', 'AQMH_RECONSTRUCTION', 'AQMH_DIAGNOSTICS'],
  };
}
```

#### Run Monitor AQMH status panel

When `method=aqmh` (default), the AQMH-specific status panel is rendered in the primary monitor view, not in a collapsible "Advanced" section:

- Header badge: `AQMH`
- Map progress: `AQMH maps 143/300`
- Cache written / read
- Cache hit rate
- Resident maps observed vs configured
- Warnings for cache misses and unsupported pixels

### 4.4 Parameter Studio (`web_frontend/parameter-studio.html`)

#### Default category

The AQMH category tab/panel is active on first open. Classic local/tile controls are in a separate `Classic` category that is closed by default.

```js
// Before
const DEFAULT_PARAM_CATEGORY = 'classic';

// After
const DEFAULT_PARAM_CATEGORY = 'aqmh';
```

#### Classic controls visibility

Classic-only controls (tile weighting, block/tile scalar quality settings, local metrics thresholds) are hidden by default. They are shown only when:

1. The user switches to the `Classic` category, or
2. The loaded config has `aqmh.enabled: false`.

```js
function updateParamVisibility(config) {
  const isAqmh = config?.aqmh?.enabled !== false;
  document.querySelectorAll('[data-method="classic"]').forEach(el => {
    el.hidden = isAqmh;
  });
  document.querySelectorAll('[data-method="aqmh"]').forEach(el => {
    el.hidden = !isAqmh;
  });
}
```

---

## 5. History and Comparison (`web_frontend/history-tools.html`)

No logic change needed: tagging is driven by the `method` field in stored run metadata. Newly created runs will naturally carry `method=aqmh` once the config default is flipped.

Confirm the history view sorts/groups correctly when `method=aqmh` is the majority tag. There is no code change required here unless the current sort order hard-codes Classic as the primary bucket.

---

## 6. Backend Compatibility

No breaking changes. The existing compatibility contract holds:

| Run state | Derived method |
|---|---|
| Old run, no `method` field | `classic_tile_compile` (unchanged) |
| `aqmh.enabled: false` in config | `classic_tile_compile` |
| `aqmh.enabled: true` in config | `aqmh` |
| New run, default config | `aqmh` (new) |

The status endpoint must always include `method` in its response. If `method` is absent from a legacy run, the frontend treats it as Classic — not as AQMH. This is the correct and safe behavior: old Classic runs must not be retroactively relabeled.

---

## 7. Test Audit

### Tests that must be updated

Any test that:
- Constructs `AqmhConfig{}` without loading a YAML file will now have `enabled=true` by default. These tests must explicitly set `enabled=false` if they are testing Classic behavior, or be converted to AQMH tests.
- Checks that the dashboard/wizard shows `Classic Tile Compile` as the default label.
- Checks that `RUN_MONITOR_PHASE_ORDER` starts with Classic phase names.
- Checks that the Parameter Studio opens in the Classic category.

### New tests

| Test | Assertion |
|---|---|
| Dashboard no-config state | Method badge shows `AQMH` when no config is loaded |
| Wizard initial state | AQMH radio is checked on first open |
| Wizard storage section | AQMH storage section visible immediately on wizard open |
| Parameter Studio initial category | AQMH category is active on first open |
| Classic controls hidden | Classic-only fields are hidden when `aqmh.enabled=true` |
| Run Monitor phase order | `getPhaseOrder('aqmh')` returns AQMH-ordered list |
| Run Monitor phase order Classic | `getPhaseOrder('classic_tile_compile')` returns Classic-ordered list |
| Cache estimate shown by default | `onScanComplete` renders cache estimate without user interaction when method is AQMH |
| Legacy run label | Run with no `method` field renders as `Classic Tile Compile`, not `AQMH` |

---

## 8. Migration Path for Existing Users

Users with existing Classic configs are not affected. Their configs have `aqmh.enabled: false` explicitly or have no `aqmh:` key at all, which the backend already treats as Classic.

Users who update to the new default config (e.g., by running a new project wizard or resetting to defaults) will get AQMH. The wizard must make this visible with the AQMH radio pre-selected and an information line:

```text
AQMH is the recommended reconstruction method.
To use Classic Tile Compile, select it below.
```

---

## 9. Documentation Update

### `docs/AQMH/aqmh_frontend_integration_plan.md` — Section 10

Remove from the Non-Goals list:

> Default-enabled AQMH.

The remaining non-goals are unchanged.

### `docs/AQMH/aqmh_implementation_plan.md` — Step 1.1

Update the default comment for `AqmhConfig::enabled`:

```cpp
// Before comment:
// Each milestone can be merged without breaking the existing pipeline
// (AQMH is always gated behind `aqmh.enabled = false`).

// After:
// AQMH is enabled by default. Set aqmh.enabled = false to use Classic Tile Compile.
```

---

## 10. Implementation Order

1. Flip `AqmhConfig::enabled` default in `configuration.hpp` — this is the anchor change; all other changes derive from it.
2. Update `tile_compile.yaml` default config.
3. Update schema YAML/JSON descriptions.
4. Rename example config: delete `aqmh_enabled.example.yaml`, create `classic_mode.example.yaml`.
5. Wizard: swap radio button order and default; make AQMH storage section visible immediately.
6. Dashboard: update `getMethodLabel` fallback; make cache estimate primary post-scan view; activate default warnings.
7. Run Monitor: introduce `RUN_MONITOR_PHASE_ORDER_AQMH` / `_CLASSIC` split; update `DASHBOARD_PIPELINE_GROUPS` to be method-aware.
8. Parameter Studio: set `DEFAULT_PARAM_CATEGORY = 'aqmh'`; add `data-method` attributes and visibility logic for Classic controls.
9. Audit and update affected tests.
10. Update documentation (integration plan Section 10, implementation plan Step 1.1).

Steps 1–4 are backend/config. Steps 5–8 are pure frontend. They can be developed in parallel after step 1 is merged.
