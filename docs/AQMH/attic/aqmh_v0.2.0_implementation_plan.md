# AQMH v0.2.0 — Implementation Plan

**Spec:** `docs/AQMH/aqmh_methodik_en_v0.2.0.md`
**Current code:** Partial v0.1.0 in `tile_compile_cpp/`

## Delta: v0.1.0 → v0.2.0

| Area | v0.1.0 (current) | v0.2.0 (target) |
|---|---|---|
| **Pipeline structure** | AQMH as `if`-branch inside Classic phases | **AQMH as independent main path**, separate files |
| Numerical guards | `eps_aqmh = 1e-6` | `eps_scale(X)`, `eps_noise(X)` dimension-preserving |
| Frame quality `G` | External `global_weights` from `GLOBAL_METRICS` | AQMH-native: z-score of map summaries → sigmoid |
| Validity model | Canvas-valid (`C`) | Source-valid: `C AND M_f` |
| `Phi_snr` floor | `eps_aqmh` | `eps_noise(I_s)` |
| `Phi_artifact` denom | `W_s_valid` | `H_s_valid` (finite hp only) |
| `Phi_artifact` <3 samples | Not specified | `= 1` (no false veto) |
| Storage default | `resolution_divisor=2` | `resolution_divisor=1` |
| Veto mask | Not stored | Full-res 1-bit for `divisor>1` |
| Map format version | 1 | 2 (bump) |
| Pipeline stages | MAPS→RECON (inline in Classic) | MAPS→GLOBAL_QUALITY→RECON→DIAGNOSTICS (separate) |
| Sigma clipping | Unspecified | Weighted median+MAD, keep-floor, N_eff guard |
| Cherry-pick | `min(N,max(k_min,floor(k_frac*N)))` | `k_min_required=20`, `K_nominal_median` gate, tiered |
| Diagnostics | 7 fields/frame | 11 fields/frame + run-level cherry-pick stats |
| Validation | 10 reqs | 16 reqs + control-run |

---

## Phase 0: Pipeline Disentanglement — AQMH als Main-Pfad

**Ziel:** AQMH-Code aus Classic-Phasen extrahieren, in eigene Dateien auslagern, zwei klare Pipeline-Pfade schaffen.

### Target File Layout (neue AQMH-Dateien)

```
tile_compile_cpp/
├── apps/
│   ├── runner_pipeline.cpp              # Main dispatcher: aqmh vs classic
│   ├── runner_aqmh_pipeline.cpp         # NEW: AQMH pipeline orchestrator
│   ├── runner_aqmh_pipeline.hpp         # NEW
│   ├── runner_phase_local_metrics.cpp   # Classic-only (nach Extraktion)
│   ├── runner_phase_local_metrics.hpp   # Classic-only
│   ├── runner_phase_aqmh_maps.cpp       # NEW: AQMH quality-map phase
│   ├── runner_phase_aqmh_maps.hpp       # NEW
│   ├── runner_phase_aqmh_global_quality.cpp  # NEW: G factor phase
│   ├── runner_phase_aqmh_global_quality.hpp  # NEW
│   ├── runner_phase_aqmh_reconstruction.cpp  # NEW: AQMH recon phase
│   ├── runner_phase_aqmh_reconstruction.hpp  # NEW
│   ├── runner_phase_aqmh_diagnostics.cpp     # NEW: AQMH diagnostics phase
│   ├── runner_phase_aqmh_diagnostics.hpp     # NEW
│   └── ... (classic files unchanged)
├── src/
│   ├── metrics/
│   │   ├── aqmh_quality_map.cpp         # Existing (updated)
│   │   ├── aqmh_quality_map_cache.cpp   # Existing (updated)
│   │   ├── aqmh_eps.cpp                 # NEW: eps_scale, eps_noise, robust zscore
│   │   ├── aqmh_global_quality.cpp      # NEW: G factor computation
│   │   ├── aqmh_frame_valid_mask.cpp    # NEW: M_f mask derivation
│   │   └── ... (classic metrics unchanged)
│   ├── reconstruction/
│   │   ├── reconstruction.cpp           # Classic-only (nach Extraktion)
│   │   ├── aqmh_reconstruction.cpp      # NEW: AQMH recon (extracted)
│   │   ├── aqmh_sigma_clip.cpp          # NEW: weighted median+MAD clipping
│   │   ├── aqmh_cherry_pick.cpp         # NEW: cherry-pick selection logic
│   │   └── ... (classic recon files unchanged)
│   └── ...
├── include/tile_compile/
│   ├── metrics/
│   │   ├── aqmh_quality_map.hpp         # Existing (updated)
│   │   ├── aqmh_quality_map_cache.hpp   # Existing (updated)
│   │   ├── aqmh_eps.hpp                 # NEW
│   │   ├── aqmh_global_quality.hpp      # NEW
│   │   ├── aqmh_frame_valid_mask.hpp    # NEW
│   │   └── ...
│   ├── reconstruction/
│   │   ├── reconstruction.hpp           # Classic structs only
│   │   ├── aqmh_reconstruction.hpp      # NEW: AQMH recon structs + decls
│   │   ├── aqmh_sigma_clip.hpp          # NEW
│   │   ├── aqmh_cherry_pick.hpp         # NEW
│   │   └── ...
│   ├── core/
│   │   ├── types.hpp                    # Updated: new Phase enum values
│   │   └── ...
│   └── ...
```

### Step 0.1 — Phase-Enum erweitern (`include/tile_compile/core/types.hpp`)

- Neue Phasen hinzufügen:
  ```cpp
  AQMH_MAPS,              // quality map computation
  AQMH_GLOBAL_QUALITY,    // G factor computation
  AQMH_RECONSTRUCTION,    // AQMH pixel-wise reconstruction
  AQMH_DIAGNOSTICS,       // post-recon diagnostics + regions
  ```
- `phase_to_string()` und `phase_from_int()` aktualisieren
- Classic-Phasen bleiben unangetastet (`LOCAL_METRICS`, `TILE_RECONSTRUCTION`, etc.)

### Step 0.2 — `run_phase_local_metrics()` aufteilen

**Aktuell:** Eine Funktion (~994 Zeilen) mit `if (compute_classic_local_metrics)`-Verzweigung.
- Classic-Pfad (tile metrics, local weights, neighborhood normalization, spatial regularization, Zeilen 169–993) → in `runner_phase_local_metrics.cpp` belassen, AQMH-Code entfernen
- AQMH-Pfad (quality map computation, Zeilen 324–555) → extrahieren in **`runner_phase_aqmh_maps.cpp`** + **`runner_phase_aqmh_maps.hpp`**
- Neue Funktion: `run_phase_aqmh_maps()` — nur AQMH-relevante Parameter:
  - `frames`, `prewarped_frames`, `common_valid_mask` (C), `frame_valid_masks` (M_f)
  - `cfg.aqmh.*`, `acceleration`, `emitter`, `log_file`
  - **Keine** `tiles_phase56`, `local_metrics`, `local_weights`, `tile_common_valid`
- `runner_phase_local_metrics.hpp`: Classic-Signatur behalten, AQMH-Signatur in `runner_phase_aqmh_maps.hpp`

### Step 0.3 — TILE_RECONSTRUCTION-Phase aufteilen

**Aktuell:** `runner_pipeline.cpp` Zeilen ~2226–2397 (AQMH) und ~2398–3856 (Classic, ~1458 Zeilen) in einem `if/else`-Block.
- AQMH-Pfad → extrahieren in **`runner_phase_aqmh_reconstruction.cpp`** + `.hpp`
  - Neue Funktion: `run_phase_aqmh_reconstruction()` — ruft `reconstruct_aqmh_weighted()` auf, schreibt `tile_reconstruction.json` artifact
- Classic-Pfad → in `runner_pipeline.cpp` belassen oder in separate Funktion auslagern (optional)
- **`reconstruction.cpp`** aufteilen:
  - Classic tile-weighted reconstruction (`reconstruct_tiles_parallel()` etc.) → bleibt in `reconstruction.cpp`
  - AQMH reconstruction (`reconstruct_aqmh_weighted()`) → extrahieren in **`aqmh_reconstruction.cpp`**
  - AQMH structs (`AqmhReconstructionConfig`, `AqmhReconstructionResult`, `AqmhFrameLoader`) → **`aqmh_reconstruction.hpp`**
  - Classic structs (`ReconstructionConfig`, `ReconstructTilesResult`) → bleiben in `reconstruction.hpp`

### Step 0.4 — `global_weights` Entflechtung

**Aktuell:** `GLOBAL_METRICS`-Phase erzeugt `global_weights` aus `frame_metrics` → beide Pfade nutzen dieselben.
- Classic: `global_weights` aus `frame_metrics` bleibt unangetastet
- AQMH: `AQMH_GLOBAL_QUALITY`-Phase erzeugt `G_{f,c}` → eigene Variable `aqmh_global_weights`
- **`PhaseMetricsContext::global_weights`** wird nicht von AQMH-Code gelesen
- AQMH-Pipeline übergibt `aqmh_global_weights` an `reconstruct_aqmh_weighted()`
- Neue Datei: **`aqmh_global_quality.cpp`** + **`aqmh_global_quality.hpp`** für G-Faktor-Logik

### Step 0.5 — `COMMON_OVERLAP` für AQMH: C-only, M_f ableiten

**Aktuell:** `COMMON_OVERLAP` berechnet `common_valid_mask` (C) **und** `tile_common_valid` (tile-level).
- AQMH braucht nur `C` (pixel-level canvas mask) + `M_f` (frame-level valid mask)
- `COMMON_OVERLAP` für AQMH: nur `common_valid_mask` berechnen, tile-Validierung überspringen
- **M_f (frame-valid mask)**: neue Datei **`aqmh_frame_valid_mask.cpp`** + `.hpp`
  - `M_f(p) = 1 iff prewarped_frame[fi](p) is finite`
  - Effizient: während PREWARP erzeugt und als 1-bit-Maske pro Frame gecacht
  - Neue Datenstruktur: `FrameValidMaskStore` (disk-backed, ähnlich `DiskCacheFrameStore` aber 1-bit)

### Step 0.6 — `TILE_GRID` für AQMH überspringen

- AQMH ist pixel-wise, braucht kein Tile-Grid
- Phase wird für AQMH-Main übersprungen
- `tiles_phase56` wird nicht erzeugt → alle tile-abhängigen Parameter fallen weg
- In `runner_aqmh_pipeline.cpp`: `TILE_GRID`-Phase nicht aufrufen

### Step 0.7 — Post-Reconstruction Phasen bereinigen

- `STATE_CLUSTERING`, `SYNTHETIC_FRAMES`, `STACKING` → für AQMH komplett überspringen
- Aktuell schon via `skip_clustering_for_aqmh` — wird durch Pipeline-Trennung obsolet
- `DEBAYER`, `ASTROMETRY`, `BGE`, `PCC`, `HYPERMETRIC_STRETCH` → gemeinsam (shared infrastructure)
- BGE: v0.2.0 §4.1 spezifiziert `AQMH_NATIVE_BGE_INPUTS` — BGE tile-sampling aus AQMH recon output + C, **nicht** aus `local_metrics.json`

### Step 0.8 — Pipeline-Main-Loop: zwei Pfade

**Aktuell:** `runner_pipeline.cpp` (~6426 Zeilen) mit `if (cfg.aqmh.enabled)` Verzweigungen in jeder Phase.
- Neue Datei: **`runner_aqmh_pipeline.cpp`** + `.hpp` — AQMH Pipeline Orchestrator
- `runner_pipeline.cpp` wird zum Dispatcher:
  ```
  if (cfg.method == "aqmh") → runner::run_aqmh_pipeline(...)
  else → runner::run_classic_pipeline(...)
  ```
- Beide Pfade teilen: `SCAN_INPUT`, `REGISTRATION`, `PREWARP`, `CHANNEL_SPLIT`, `NORMALIZATION`, `COMMON_OVERLAP` (C-only für AQMH)
- AQMH-spezifisch: `AQMH_MAPS` → `AQMH_GLOBAL_QUALITY` → `AQMH_RECONSTRUCTION` → `AQMH_DIAGNOSTICS`
- Classic-spezifisch: `TILE_GRID` → `LOCAL_METRICS` → `TILE_RECONSTRUCTION` → `STATE_CLUSTERING` → `SYNTHETIC_FRAMES` → `STACKING`
- Gemeinsame Post-Phasen: `DEBAYER` → `ASTROMETRY` → `BGE` → `PCC` → `HYPERMETRIC_STRETCH` → `DONE`

### Step 0.9 — `AQMH_DIAGNOSTICS` als eigene Phase

**Aktuell:** Diagnostics inline in `LOCAL_METRICS` (`aqmh_metrics.json`) und `TILE_RECONSTRUCTION` (`tile_reconstruction.json`).
- Neue Datei: **`runner_phase_aqmh_diagnostics.cpp`** + `.hpp`
- Sammelt: per-frame diagnostics, run-level diagnostics, block-level diagnostics, heatmaps, region extraction
- Schreibt: `aqmh_metrics.json` (erweitert), `aqmh_regions.json`
- Wird nach `AQMH_RECONSTRUCTION` ausgeführt

### Step 0.10 — Web Backend / Frontend Anpassung

- Web Backend (`web_backend_cpp/`): AQMH-Phasen in Phase-Definition aufnehmen
- Phase-Namen in i18n-Dateien (`de.json`, `en.json`) ergänzen; remove redundant `phase.aqmh_quality_maps` entry (duplicate of `phase.aqmh_maps`)
- **Fix phase order in `web_frontend_v3/js/components/phase-list.js`**:
  - `CLASSIC_PHASES`: reorder to match `core/types.hpp`: `SCAN_INPUT → REGISTRATION → PREWARP → CHANNEL_SPLIT → NORMALIZATION → GLOBAL_METRICS → TILE_GRID → COMMON_OVERLAP → ...`
  - `AQMH_PHASES`: reorder to match v0.2.0 pipeline: `SCAN_INPUT → REGISTRATION → PREWARP → CHANNEL_SPLIT → NORMALIZATION → COMMON_OVERLAP → AQMH_MAPS → AQMH_GLOBAL_QUALITY → AQMH_RECONSTRUCTION → AQMH_DIAGNOSTICS → ...`
  - Remove Classic-only phases (`GLOBAL_METRICS`, `TILE_GRID`, `LOCAL_METRICS`, `TILE_RECONSTRUCTION`, `STATE_CLUSTERING`, `SYNTHETIC_FRAMES`, `STACKING`) from `AQMH_PHASES`
  - Update `RESUMABLE_PHASES` accordingly
- **Fix phase order in `web_frontend/src/app.js`**:
  - `RUN_MONITOR_PHASE_ORDER`: reorder to match `core/types.hpp`
  - `AQMH_RUN_MONITOR_PHASE_ORDER`: add `AQMH_GLOBAL_QUALITY` and `AQMH_DIAGNOSTICS`, remove Classic-only phases, correct order
  - `getEffectiveDashboardPipelineGroups()`: update AQMH groups to `SCAN → REG → AQMH → DEBAYER → ASTROM → BGE → PCC → HMS`
- **Skip `GLOBAL_METRICS` phase events for AQMH in backend** (`tile_compile_cpp/apps/runner_phase_metrics.cpp`):
  - Add `const bool expose_global_metrics = !cfg.aqmh.enabled;`
  - Skip `phase_start`, `phase_progress`, `phase_end` when AQMH is active
  - Keep the metric computation and `global_metrics.json` artifact — they are still needed for:
    - BGE background-RMS validation (`frame_metrics[i].noise`)
    - Registration model-predicted weight penalty (`global_weights`)
  - This is an interim step until BGE/registration are fully disentangled from Classic metrics
- **Fix backend-status merge in `web_frontend_v3/js/pages/run-monitor.js`**:
  - In `refreshRunStatus()`, merge `status.phases` into `getPhasesForConfig()`-based order instead of using raw backend order
  - This is a defensive guard against any backend phases not relevant to the current method
- API-Endpunkte für `aqmh_metrics.json`, `aqmh_regions.json`

---

## Phase 1: Config & Numerical Infrastructure

### Step 1.1 — Config structs (`include/tile_compile/config/configuration.hpp`)

- `AqmhStorageConfig`: default `resolution_divisor` 2→1
- `AqmhCherryPickConfig`: remove `k_min`, add `k_min_required=20`, `margin_min=0.02`, `tiered_k_frac` (vec of `{min_n_rankable, k_frac}`)
- New `AqmhGlobalQualityConfig`: `g_floor=0.05, g_w_sharp=0.6, g_w_snr=0.4`
- New `AqmhReconstructionConfig`: `clip_sigma=3.0, clip_iterations=3, min_fraction=0.5, min_n_eff=2.0`
- New `AqmhValidationConfig`: `max_seam_score_regression=0.02, max_fwhm_regression=0.02, max_background_rms_regression=0.02`
- New `AqmhDiagnosticsConfig`: `tau_artifact=0.20, q_region=0.75, r_morph_canvas_px=6`
- `AqmhConfig`: add the four new sub-structs

### Step 1.2 — Replace `eps_aqmh` with `eps_scale`/`eps_noise`

**New files:** `src/metrics/aqmh_eps.cpp` + `include/tile_compile/metrics/aqmh_eps.hpp`

- Remove `constexpr float eps_aqmh` from `aqmh_quality_map.hpp`
- `aqmh_eps.hpp`: declare `eps_scale()`, `eps_noise()`, `robust_zscore_eps_scale()`
- `aqmh_eps.cpp`: implement:
  - `eps_scale(X)`: `max(nextafter(0,1), 1e-6 * max(median(|X|), MAD(X)))`
  - `eps_noise(X)`: `max(nextafter(0,1), 1e-6 * median(|X - median(X)|))`
  - `robust_zscore_eps_scale()`: degenerate (MAD=0) → z=0
- `aqmh_quality_map.cpp`: include `aqmh_eps.hpp`, update all call sites: `Phi_snr`→`eps_noise`, `Phi_artifact` tau→`eps_scale(hp)`, `compute_psi`→`robust_zscore_eps_scale`

### Step 1.3 — YAML parsing + schema

- Parse `aqmh.global_quality.*`, `aqmh.reconstruction.*`, `aqmh.validation.*`
- Parse `aqmh.diagnostics.tau_artifact`, `q_region`, `r_morph_canvas_px`
- Parse `cherry_pick.k_min_required`, `margin_min`, `tiered_k_frac`
- Update `tile_compile.schema.json`

---

## Phase 2: Quality Map Computation (§2)

### Step 2.1 — Source-valid mask (C + M_f)

**New files:** `src/metrics/aqmh_frame_valid_mask.cpp` + `include/tile_compile/metrics/aqmh_frame_valid_mask.hpp`

- `aqmh_frame_valid_mask.hpp`: declare `FrameValidMaskStore` (disk-backed 1-bit per-pixel mask), `compute_frame_valid_mask()`
- `aqmh_frame_valid_mask.cpp`: implement M_f derivation from prewarped frames
- `aqmh_quality_map.hpp`: `compute_aqmh_quality_map` signature add `frame_valid_mask` param
- `aqmh_quality_map.cpp`: compute `source_valid = C AND M_f`; replace all canvas-valid checks
- `canvas_masked_frame` → `source_masked_frame` (NaN where C=0 OR M_f=0)
- Update `downsample_valid_mean`, `mask_valid` calls

### Step 2.2 — `Phi_snr` background + denominator (§2.3.2b)

- **Background `b_s`**: spec requires `median_{p in W_s_valid}(I_s(p))` — a **local masked median**, not a local mean. Current implementation uses `local_mean` as background approximation.
  - Correct implementation: per-pixel window median (O(W×H×R²)); or document explicitly as approved performance approximation if the mean is retained.
  - `mu_s = mean(max(I_s(p) - b_s, 0))` — mean of positive-clamped background-subtracted values
  - `sigma_s = 1.4826 * MAD_{p in W_s_valid}(I_s(p))` — MAD over **raw** `I_s`, not background-subtracted
- **`eps_noise` scope**: spec §2.3.2b defines `eps_noise(I_s over W_s_valid(x,y))` as a **per-window** floor. Current implementation computes it once globally over all finite pixels as a performance approximation — this is an intentional deviation. Document as approved approximation: global `eps_noise` computed once per scale over all source-valid pixels, used as scalar floor.
- Both approximations must be noted in code comments referencing §2.3.2b.

### Step 2.3 — `Phi_artifact` updates (§2.3.2c)

- `H_s_valid(x,y)` = `{p in W_s_valid(x,y) | hp_s(p) is finite}` — subset where high-pass response is finite
- `tau_s` uses `max(1.4826 * MAD_{H_s_valid}(hp_s), eps_scale(hp_s over H_s_valid))`
- **`frac_out` denominator = `|H_s_valid|`** (not `|W_s_valid|`) — binding per spec
- `|H_s_valid| < 3` → `Phi_artifact = 1` (insufficient support, no false veto); `|H_s_valid| = 0` → invalid
- **Current separable-path implementation** uses `local_mean(outlier_ind, r)` which implicitly divides by the total window count, not `|H_s_valid|`. This must be corrected:
  - Compute `outlier_count = local_sum(outlier_ind, r)` (sum, not mean)
  - Compute `h_valid_count = local_sum(hp_finite_ind, r)` (support count)
  - `frac_out(p) = outlier_count(p) / h_valid_count(p)` where `h_valid_count > 0`
  - `|H_s_valid| < 3`: use scalar `h_valid_count < 3` guard before assigning 1.0
  - For the `< 3` guard and `tau_s` (which requires local MAD over `H_s_valid`), the separable path cannot compute a true per-pixel local MAD. Document as: use global `eps_scale` floor for tau, and handle `< 3` via the support-count map.

### Step 2.4 — `Lap_valid` explicit definition

- Verify: `I(p) - mean(I(q))` over source-valid 4-connected axial neighbors
- Requires valid center + ≥2 valid neighbors; no mirror/replicate/zero-fill

### Step 2.5 — Geometric mean & output guard

- Log-sum approach already correct; update `eps_aqmh` clamp to `nextafter(0,1)`
- Output guard: `C=0 OR M_f=0` → `Q_map=0` (was C-only)

### Step 2.6 — Pre-z-score diagnostic summaries

- `AqmhQualityMapDiagnostics`: add `g_sharp_summary`, `g_snr_summary`, `g_summary_invalid`
- Compute median of `Phi_sharp_0` and `Phi_snr_1` (or finest available) over source-valid

---

## Phase 3: Global Quality Factor G (§1.1, §4.1)

### Step 3.1 — `AQMH_GLOBAL_QUALITY` stage

**New files:** `src/metrics/aqmh_global_quality.cpp` + `include/tile_compile/metrics/aqmh_global_quality.hpp` + `apps/runner_phase_aqmh_global_quality.cpp` + `.hpp`

- `aqmh_global_quality.hpp`: declare `compute_aqmh_global_quality()` — takes per-frame summaries, returns `G_{f,c}`
- `aqmh_global_quality.cpp`: implement cross-frame robust z-score + sigmoid
- `runner_phase_aqmh_global_quality.cpp`: pipeline phase — collect summaries from all frames, call `compute_aqmh_global_quality()`, persist G values
- `G_{f,c} = g_floor + (1-g_floor) * sigmoid(g_w_sharp*z + g_w_snr*z)`
- Persist G values + summaries before reconstruction

### Step 3.2 — Integration

- `runner_aqmh_pipeline.cpp`: pass `aqmh_global_weights` (not `PhaseMetricsContext::global_weights`) to `reconstruct_aqmh_weighted()`
- Assert `g_floor < G < 1`

---

## Phase 4: Storage & Cache (§3)

### Step 4.1 — Format version bump

- `kAqmhMapFormatVersion` 1→2 (invalidates v0.1.0 caches)

### Step 4.2 — Zero-veto mask for `divisor > 1`

- Store full-res 1-bit veto mask alongside downsampled map
- On read: upsample, then reset vetoed pixels to 0
- Update `write()`, `decode_file()`, metadata

### Step 4.3 — Mask-aware area downsampling

- `downsample_for_storage`: valid-count denominator, invalid pixels excluded

### Step 4.4 — Cache invalidation: M_f hash

- `make_config_hash`: add M_f mask hash (or combined C+M_f hash)

---

## Phase 5: Reconstruction (§4.3)

### Step 5.1 — Sample sets

**File:** `src/reconstruction/aqmh_reconstruction.cpp` (extracted from `reconstruction.cpp`)

- `V_c^I(p)`: add `C=1 AND M_f=1 AND finite` (currently C-only)
- Add `frame_valid_mask` param to `reconstruct_aqmh_weighted`
- `A_c(p)`: cherry-pick subset or `V_c^I(p)`

### Step 5.2 — Unsupported-pixel handling (3 cases)

- `eps_weight(p) = |A_c(p)| * eps_machine * w_max`
- Case 1: finite map, all weights zero → veto, no unweighted mean
- Case 2: no finite map → unsupported + warning
- Case 3: post-clip denom fails → unsupported + numerical warning

### Step 5.3 — Deterministic weighted sigma clipping (§4.3.1)

**New files:** `src/reconstruction/aqmh_sigma_clip.cpp` + `include/tile_compile/reconstruction/aqmh_sigma_clip.hpp`

- `aqmh_sigma_clip.hpp`: declare `weighted_median()`, `weighted_mad()`, `sigma_clip_weighted()`
- `aqmh_sigma_clip.cpp`: implement:
  - Weighted median: sort by `(intensity, frame_index)` — **frame_index as deterministic tiebreaker** — cumweight ≥ half
  - Weighted MAD: `1.4826 * weighted_median(|I_f - m|)` using same sort convention
  - Clip: `|I_f - m| <= clip_sigma * sigma`; degenerate (sigma ≤ eps_scale) → retain samples with `I_f = m`
  - Keep-floor: `n_keep_min = min(n0, max(1, ceil(min_fraction*n0)))`; if would retain fewer, retain the `n_keep_min` samples with smallest `(|I_f - m| / max(sigma, eps_scale({I_f})), frame_index)` — **frame_index as tiebreaker here too**
  - Post-clip: `D_eff = sum(w_f)`, `N_eff = D_eff² / sum(w_f²)`; unsupported if `D_eff <= n_retained * machine_epsilon * max(w_f)` OR `N_eff < min_n_eff`
- `aqmh_reconstruction.cpp`: call `sigma_clip_weighted()` per pixel instead of Welford-only

### Step 5.4 — Config integration

**File:** `apps/runner_phase_aqmh_reconstruction.cpp`

- Map new `AqmhReconstructionConfig` (from `configuration.hpp`) to `reconstruct_aqmh_weighted()` params
- Replace `sigma_low/sigma_high` with `clip_sigma/clip_iterations/min_fraction/min_n_eff`
- Remove old `sigma_low/sigma_high` references from `runner_pipeline.cpp`

---

## Phase 6: Cherry-Pick v0.2.0 (§5.3)

### Step 6.1 — Run-level gate (§5.3.2)

**New files:** `src/reconstruction/aqmh_cherry_pick.cpp` + `include/tile_compile/reconstruction/aqmh_cherry_pick.hpp`

- `aqmh_cherry_pick.hpp`: declare `CherryPickGate`, `compute_k_nominal()`, `evaluate_cherry_pick_gate()`
- `aqmh_cherry_pick.cpp`: implement:
  - `K_nominal(p) = floor(k_frac(p) * N_rankable(p))`
  - `K_nominal_median = median(K_nominal(p))` over C=1 pixels
  - If `< k_min_required`: force disable, record `cherry_pick_forced_disabled`, WARNING

### Step 6.2 — Per-pixel K(p) (§5.3.3)

**File:** `src/reconstruction/aqmh_cherry_pick.cpp`

- `K(p) = max(k_min_required, K_nominal(p))` when `N_rankable >= k_min_required`
- Else: inactive, use full `V_c^I(p)`
- Remove old `min(N, max(k_min, ...))` logic from `aqmh_reconstruction.cpp`

### Step 6.3 — Tiered k_frac (§5.3.4)

**File:** `src/reconstruction/aqmh_cherry_pick.cpp`

- Match tier: greatest `min_n_rankable <= N_rankable(p)`
- `k_frac(p) = max(base, tier)`; no tier → base

### Step 6.4 — Rank-separation diagnostic (§5.3.5)

**File:** `src/reconstruction/aqmh_cherry_pick.cpp`

- `margin(p) = (S_(K) - S_(K+1)) / S_(1)`
- `median(margin)` < `margin_min` → `low_rank_separation: true`

### Step 6.5 — Diagnostics output

**Files:** `src/reconstruction/aqmh_cherry_pick.cpp` (compute) + `apps/runner_phase_aqmh_diagnostics.cpp` (write)

- Add: `cherry_pick_forced_disabled`, `cherry_pick_active`, `k_nominal_median`
- Add: `k_effective_p10/p50/p90`, `low_rank_separation`
- `AqmhReconstructionResult` in `aqmh_reconstruction.hpp`: add cherry-pick diagnostic fields

### Step 6.6 — Remove large-image fallback

- Current `kCherryPickMaxPixels=8Mpx` global fallback is non-spec
- Implement streaming per-pixel selection within memory budget
- All cherry-pick logic in `aqmh_cherry_pick.cpp` (not in `reconstruction.cpp`)

---

## Phase 7: Diagnostics (§6)

**New files:** `apps/runner_phase_aqmh_diagnostics.cpp` + `.hpp`

### Step 7.1 — Per-frame fields (§6.1)

All 11 fields per `(frame, channel)` pair:

- `map_mean` — mean of `Q_map` over source-valid pixels
- `map_p10`, `map_p90` — 10th/90th percentile over source-valid pixels
- `artifact_frac` — fraction with `Q_map < tau_artifact` (from `AqmhDiagnosticsConfig.tau_artifact`)
- `sharpness_p50` — median of pre-z-score `Phi_sharp_0` at scale 0
- `snr_p50` — median of pre-z-score `Phi_snr_1` at scale 1 (or finest available; `NaN`/`null` if scale 1 omitted)
- `n_regions` — number of quality regions from §5.2 above threshold
- `global_quality` — `G_{f,c}` from §1.1
- `global_sharpness_input`, `global_snr_input` — pre-z-score summaries used to derive `G_{f,c}`
- `global_quality_input_invalid` — `true` if either global summary required the zero-z-score fallback
- Update existing: "canvas-valid" → "source-valid" in all computations
- All per-frame diagnostics written by `runner_phase_aqmh_diagnostics.cpp` (not inline in maps/recon phases)

### Step 7.2 — Run-level fields

- `cherry_pick_forced_disabled`, `cherry_pick_active`, `k_nominal_median`
- `k_effective_p10/p50/p90`, `low_rank_separation`
- Written to `aqmh_metrics.json` by `runner_phase_aqmh_diagnostics.cpp`

### Step 7.3 — Per-channel + CFA-proxy

- All diagnostics per `(frame, channel)`; CFA-proxy: record `analysis_channel: proxy`
- Aggregates may be reported additionally but must not replace per-channel values

### Step 7.4 — Block-level diagnostics (§6.2)

**File:** `apps/runner_phase_aqmh_diagnostics.cpp`

Per report block `b`, per `(frame, channel)`:

- `aqmh_q_median` — `Q_{f,b,c}^{aqmh} = median_{p in b, source-valid}(Q_map(p))`
- `aqmh_q_p10`, `aqmh_q_p90` — 10th/90th percentile within block
- `aqmh_artifact_frac` — fraction of block pixels with `Q_map < tau_artifact`

### Step 7.5 — Heatmaps (§6.3)

**File:** `apps/runner_phase_aqmh_diagnostics.cpp`

- Mean `Q_map` per report block, per frame and analysis channel → into `aqmh_metrics.json` spatial heatmap entries
- Artifact fraction heatmap per report block, per frame and analysis channel
- Optional AQMH-vs-Classic comparison heatmaps only when both methods were run separately on the same input set

---

## Phase 8: Region Extraction (§5.2)

**File:** `apps/runner_phase_aqmh_diagnostics.cpp` (integrated into AQMH_DIAGNOSTICS phase)

### Step 8.1 — Threshold + binary mask

- `tau_{f,c} = quantile(Q_map_{f,c}, q_region)` over finite source-valid pixels (from `AqmhDiagnosticsConfig.q_region = 0.75`)
- `M_region_{f,c}(p) = 1 iff Q_map_{f,c}(p) >= tau_{f,c} AND C(p) = 1 AND M_f(p) = 1`

### Step 8.2 — Morphological opening (§5.2, step 3)

- Apply morphological opening with radius `r_morph_map = max(1, round(r_morph_canvas_px / resolution_divisor))` in map-space pixels (canvas default: `r_morph_canvas_px = 6`)
- Opening constrained to valid support (C AND M_f); result intersected with C AND M_f
- Use OpenCV `cv::morphologyEx` with `MORPH_OPEN` on the binary mask

### Step 8.3 — Connected components + region properties (§5.2, steps 4–5)

- Extract connected components via `cv::connectedComponentsWithStats`
- For each component `r`, compute:
  - `Area_r` — pixel count
  - `MeanQ_r` — mean `Q_map` over region pixels
  - `Compactness_r = 4*pi*Area_r / Perimeter_r^2` (Polsby-Popper; perimeter from contour length)
- Rank by `Score_r = MeanQ_r * log(1 + Area_r)`
- Write per `(frame, channel)` to `aqmh_regions.json`

---

## Phase 9: Validation (§7.7, §9)

### Step 9.1 — Uniform-weight control run

- Same samples, masks, sigma-clipping; uniform weights; AQMH-native

### Step 9.2 — Regression metrics

- `regression = (m_aqmh - m_control) / max(abs(m_control), eps_scale(...))`
- Compare seam score, FWHM, background RMS

### Step 9.3 — 16 validation checks

Implement all §9.1–9.16 as testable assertions:

1. Map range `Q_map ∈ [0,1]` for all finite source-valid pixels
2. Output guard: `Q_map = 0` where `C=0` or `M_f=0`
3. Determinism: identical inputs → identical maps
4. Unsupported coverage: no finite intensity + no finite map → unsupported with warning
5. Explicit zero-veto: finite zero maps not replaced by unweighted mean
6. Block diagnostic consistency: `Q_{f,b,c}` matches `median(Q_map over b)` within float32 tolerance
7. No structural injection: seam/FWHM/background-RMS within configured regression limits vs. §7.7 control
8. Artifact detection: contaminated frames show `artifact_frac > 0.01` for contaminated blocks
9. Scale omission: `P_actual < P` → fusion uses `P_actual` denominator; omitted scales recorded; unavailable diagnostics written as `NaN`/`null`
10. Cherry-pick flag + **WARNING log**: when `cherry_pick_active = true`, pipeline log must emit a `WARNING` level message (binding per §9 req.10)
11. Cherry-pick forced-disabled: `K_nominal_median < k_min_required` → `cherry_pick_forced_disabled: true`, reconstruction bit-identical to `enabled=false`
12. Cherry-pick effective-sample floor: `K(p) >= k_min_required` for all active pixels; all retained scores finite and strictly positive
13. Cherry-pick graceful degradation: `N_rankable(p) < k_min_required` → inactive at that pixel, full-set reconstruction
14. Global quality: every eligible frame has finite `G_{f,c}` with `g_floor < G < 1`; no frame removed via `G=0`
15. Storage fidelity: full-res cache exact within float32; `divisor > 1` records approximate mode, preserves zero-veto mask, reports isolated-defect recall
16. Channel diagnostics: every `(f,c)` pair has distinct diagnostics; aggregates separately named

---

## Phase 10: Integration

### Step 10.1 — `runner_aqmh_pipeline.cpp` — Stage ordering

- Enforce binding stage order: `AQMH_MAPS` → `AQMH_GLOBAL_QUALITY` → `AQMH_RECONSTRUCTION` → `AQMH_DIAGNOSTICS`
- Optional post-reconstruction: `AQMH_NATIVE_BGE_INPUTS` (when BGE enabled, see Step 10.5)
- `AQMH_GLOBAL_QUALITY` waits for all frames' pre-z-score summaries
- `AQMH_RECONSTRUCTION` waits for all `G_{f,c}` values persisted
- `AQMH_DIAGNOSTICS` waits for reconstruction output
- All five stages in `runner_aqmh_pipeline.cpp`, nicht in `runner_pipeline.cpp`

### Step 10.2 — Remove `k_min` from config/UI

- Clean up references in config parsing, UI, schema, docs
- Remove `cfg.aqmh.cherry_pick.k_min` from `runner_pipeline.cpp` (Zeile 2257)

### Step 10.5 — `AQMH_NATIVE_BGE_INPUTS` stage (§4.1)

**New files:** `apps/runner_phase_aqmh_bge_inputs.cpp` + `.hpp`

- Only executed when BGE is enabled for an AQMH run
- Derives BGE tile-sampling helpers from **AQMH reconstruction output** + canvas mask `C`
  - Per-tile: background estimate, robust noise, gradient/structure estimates for BGE sampling
  - Must **not** read from `local_metrics.json` (Classic Tile Compile output)
  - Must **not** be used as AQMH reconstruction weights
- Writes BGE-compatible tile-sampling artifact (format TBD, compatible with BGE phase input)
- Phase enum: add `AQMH_BGE_INPUTS` to `core/types.hpp`; add to phase-list and i18n

### Step 10.6 — `CMakeLists.txt` — neue Source-Files

- Alle neuen `.cpp`-Dateien in `tile_compile_cpp/CMakeLists.txt` aufnehmen:
  - `apps/runner_aqmh_pipeline.cpp`
  - `apps/runner_phase_aqmh_maps.cpp`
  - `apps/runner_phase_aqmh_global_quality.cpp`
  - `apps/runner_phase_aqmh_reconstruction.cpp`
  - `apps/runner_phase_aqmh_diagnostics.cpp`
  - `apps/runner_phase_aqmh_bge_inputs.cpp`
  - `src/metrics/aqmh_eps.cpp`
  - `src/metrics/aqmh_global_quality.cpp`
  - `src/metrics/aqmh_frame_valid_mask.cpp`
  - `src/reconstruction/aqmh_reconstruction.cpp`
  - `src/reconstruction/aqmh_sigma_clip.cpp`
  - `src/reconstruction/aqmh_cherry_pick.cpp`

### Step 10.7 — Build & test

- Build with C++20, CUDA 13, OpenCV 4.11
- Run mini dataset (15 frames) → verify maps, G, reconstruction
- Run full dataset → verify diagnostics, cherry-pick gate, performance
- Verify Classic pipeline still works (regression test)

---

## Suggested Implementation Order

1. **Phase 0.1** (Phase-Enum incl. `AQMH_BGE_INPUTS`) — foundation for all subsequent work
2. **Phase 1** (config + eps incl. `AqmhDiagnosticsConfig`) — config structs, `aqmh_eps.cpp/.hpp`, no behavioral change
3. **Phase 4.1** (format bump) — do early to avoid stale cache issues during dev
4. **Phase 0.2** (split `run_phase_local_metrics`) — extract `runner_phase_aqmh_maps.cpp`
5. **Phase 0.3** (split TILE_RECONSTRUCTION) — extract `runner_phase_aqmh_reconstruction.cpp` + `aqmh_reconstruction.cpp/.hpp`
6. **Phase 0.5** (M_f mask) — `aqmh_frame_valid_mask.cpp/.hpp`
7. **Phase 2** (quality map) — core algorithm changes in `aqmh_quality_map.cpp`; document approved approximations (Step 2.2, 2.3)
8. **Phase 0.4 + Phase 3** (global quality) — `aqmh_global_quality.cpp/.hpp` + `runner_phase_aqmh_global_quality.cpp/.hpp`
9. **Phase 5** (reconstruction) — `aqmh_sigma_clip.cpp/.hpp` incl. keep-floor tiebreaker, rewrite inner loop in `aqmh_reconstruction.cpp`
10. **Phase 6** (cherry-pick) — `aqmh_cherry_pick.cpp/.hpp`, builds on Phase 5
11. **Phase 0.8** (pipeline split) — `runner_aqmh_pipeline.cpp/.hpp`, dispatcher in `runner_pipeline.cpp`
12. **Phase 0.9 + Phase 7+8** (diagnostics + regions incl. block-level, heatmaps, morphology, CC) — `runner_phase_aqmh_diagnostics.cpp/.hpp`
13. **Phase 10.5** (`AQMH_NATIVE_BGE_INPUTS`) — `runner_phase_aqmh_bge_inputs.cpp/.hpp`
14. **Phase 0.10** (web backend/frontend) — phase definitions, i18n
15. **Phase 9** (validation incl. WARNING log check) — test infrastructure
16. **Phase 10** (integration) — CMakeLists, final wiring, build & test

### New File Summary (18 files)

| File | Purpose |
|---|---|
| `apps/runner_aqmh_pipeline.cpp/.hpp` | AQMH pipeline orchestrator (5 stages) |
| `apps/runner_phase_aqmh_maps.cpp/.hpp` | Quality-map computation phase |
| `apps/runner_phase_aqmh_global_quality.cpp/.hpp` | G factor phase |
| `apps/runner_phase_aqmh_reconstruction.cpp/.hpp` | AQMH reconstruction phase |
| `apps/runner_phase_aqmh_diagnostics.cpp/.hpp` | Diagnostics + block-level + heatmaps + region extraction phase |
| `apps/runner_phase_aqmh_bge_inputs.cpp/.hpp` | BGE tile-sampling helpers from AQMH recon output (§4.1) |
| `src/metrics/aqmh_eps.cpp` + `include/.../aqmh_eps.hpp` | eps_scale, eps_noise, robust zscore |
| `src/metrics/aqmh_global_quality.cpp` + `include/.../aqmh_global_quality.hpp` | G factor computation |
| `src/metrics/aqmh_frame_valid_mask.cpp` + `include/.../aqmh_frame_valid_mask.hpp` | M_f mask derivation + storage |
| `src/reconstruction/aqmh_reconstruction.cpp` + `include/.../aqmh_reconstruction.hpp` | AQMH recon (extracted from reconstruction.cpp) |
| `src/reconstruction/aqmh_sigma_clip.cpp` + `include/.../aqmh_sigma_clip.hpp` | Weighted median+MAD sigma clipping with deterministic tiebreakers |
| `src/reconstruction/aqmh_cherry_pick.cpp` + `include/.../aqmh_cherry_pick.hpp` | Cherry-pick selection logic |

---

## Known Approved Approximations (deviations from spec with rationale)

| Spec requirement | Approved approximation | Location | Rationale |
|---|---|---|---|
| `b_s = median_{W_s_valid}(I_s)` per pixel (§2.3.2b) | `b_s = local_mean(I_s, r)` via separable O(W×H) boxfilter | `aqmh_quality_map.cpp` `phi_snr` | O(W×H×R²) per-pixel median is ~100× slower; mean is a close approximation for smooth backgrounds |
| `eps_noise(I_s over W_s_valid(x,y))` per window (§2.3.2b) | Global `eps_noise` computed once over all source-valid pixels per scale | `aqmh_quality_map.cpp` `phi_snr` | Per-window eps_noise requires O(N) alloc+sort per pixel; global floor is numerically equivalent for the floor's purpose |
| `frac_out = outliers / |H_s_valid|` (§2.3.2c) | `local_mean_and_count(outlier_ind, r)` — `.count` = `|H_s_valid|`, `.mean * count` = outlier sum; `frac = mean` is already `outlier_count / |H_s_valid|` | `aqmh_quality_map.cpp` `phi_artifact` | **Implemented** — `local_mean_and_count` naturally divides by finite-support count, which equals `\|H_s_valid\|` since `outlier_ind` is NaN iff `hp` is NaN |
