# Release Note: Raw Stack / Preprocessing

## Scope

Raw Stack adds a standalone preprocessing menu item to Tile-Compile for the classical path from light frames to a stacked image:

```
Input Scan -> Calibration -> CFA/Mono Prep -> Reference Selection
  -> Registration -> Quality Analysis -> Frame Filtering -> Stacking
  -> Astrometry (opt.) -> BGE (opt.) -> PCC (opt.) -> HyperMetric Stretch (opt.) -> Report
```

The process is **not** part of the normal Tile-Compile Run Studio. It shares algorithms (registration, stacking, BGE, PCC, HyperMetric Stretch) and infrastructure (Run Monitor, artifact API, report generator), but launches as a separate tool runner and does not appear in the normal phase list or Parameter Studio.

---

## New Components

### Backend

- `POST /api/tools/preprocessing/run` – Starts a Raw Stack run as a background job.
- `GET /api/tools/preprocessing/status?job_id=...` – Returns phase status, artifacts, and metadata.
- `GET /api/tools/preprocessing/report?job_id=...` – Returns paths to report artifacts and `run_id`.
- `GET /api/tools/preprocessing/defaults` – Returns the effective default configuration.
- `GET /api/tools/preprocessing/parameters` – Returns parameter groups for the editor.
- `PATCH /api/tools/preprocessing/parameters` – Validates and merges configuration overrides.
- `POST /api/tools/preprocessing/scan` – Starts an input scan without a full run.
- `POST /api/tools/preprocessing/cancel` – Cancels a running job.

The existing generic run artifact endpoints are also used:

- `GET /api/runs/{run_id}/artifacts` – Lists all artifacts of the run.
- `GET /api/runs/{run_id}/artifacts/view?path=...` – Loads artifact content as text or JSON.
- `GET /api/runs/{run_id}/artifacts/raw/<path>` – Serves artifact directly with Content-Type (HTML, JSON, binary).

### Frontend

- New menu item `Raw Stack` in the header and sidebar, at the same level as `Astrometry`, `BGE`, and `PCC`.
- `raw-stack.html` with sections: Input, Run Queue, Calibration, Quality, Stack, Postprocess, Parameters, Monitor.
- i18n binding via `src/i18n.js` for all section titles, intro, footer, and navigation.
- Inline monitor in `raw-stack.html` shows phase status, log, and artifact links after the run starts.
- Run Monitor integration: `run-monitor.html?preprocessing_job_id=<id>` switches to preprocessing mode.

### Runner

- `tile_compile_runner preprocess` – New subcommand for the complete preprocessing run.
- Phases 1–10 as standalone functions in `runner_preprocess.cpp`, `runner_phase_preprocess_pipeline.cpp`, `runner_phase_quality_analysis.cpp`.
- String-based phase events (no Tile-Compile Phase enum) for clean separation from normal run phases.

---

## Implemented Phases

| Phase | Status |
|-------|--------|
| `INPUT_SCAN` | Complete |
| `CALIBRATION` | Complete (Bias, Dark with Auto-Select, Flat) |
| `CFA_CHANNEL_PREP` | Complete (OSC Bayer, Mono) |
| `REFERENCE_SELECTION` | Complete |
| `REGISTRATION` | Complete (triangle star matching) |
| `QUALITY_ANALYSIS` | Complete (stars, FWHM, eccentricity, correlation, saturation) |
| `FRAME_FILTERING` | Complete (auto + mode thresholds) |
| `STACKING` | Complete (Sigma/Median/Winsor, addscale/background/median/none, quality/uniform) |
| `ASTROMETRY` | Complete (ASTAP, WCS) |
| `BGE` | Complete |
| `PCC` | Complete |
| `HYPERMETRIC_STRETCH` | Complete (`run_hypermetric_stretch_rgb`, diagnostics JSON) |
| `REPORT` | Complete (JSON, Markdown, HTML) |

---

## Defaults

| Parameter | Value |
|-----------|-------|
| `input_mode` | `auto` |
| `raw_formats` | `tile_compile` |
| `calibration.dark_auto_select` | `true` |
| `calibration.dark_match_exposure_tolerance_percent` | `8.0` |
| `calibration.dark_match_use_temp` | `false` |
| `quality_filter.mode` | `auto` |
| `quality_filter.min_stars` | `30` |
| `quality_filter.max_fwhm_sigma` | `2.0` |
| `quality_filter.max_eccentricity` | `0.65` |
| `quality_filter.min_correlation` | `0.75` |
| `rejection.method` | `sigma` |
| `rejection.low` / `rejection.high` | `3.0` |
| `stacking.normalization` | `addscale` |
| `stacking.weighting` | `quality` |
| `postprocess.astrometry` | `true` |
| `postprocess.bge` | `true` |
| `postprocess.pcc` | `true` |
| `postprocess.hypermetric_stretch` | `true` |
| `hypermetric_stretch.require_successful_pcc` | `true` |
| `hypermetric_stretch.mode` | `ready_to_use` |
| `hypermetric_stretch.sensor_profile` | `rec709` |
| `hypermetric_stretch.target_bg` | `0.15` |
| `hypermetric_stretch.protect_b` | `6.0` |
| `hypermetric_stretch.convergence_power` | `3.5` |
| `report.detailed` | `true` |

---

## Artifacts

### Diagnostics under `<run_dir>/artifacts/preprocess/`

| File | Description |
|------|-------------|
| `effective_config.json` | Effective configuration of the run |
| `frame_quality.csv` | Quality metrics per frame |
| `rejected_frames.txt` | Excluded frames with reason |
| `stacking_diagnostics.json` | Stacking parameters and weights |
| `bge_diagnostics.json` | BGE result |
| `pcc_diagnostics.json` | PCC result (color correction factors) |
| `hms_diagnostics.json` | HMS result (anchor, log-D, profile) |
| `events.jsonl` | All phase events as JSONL |
| `artifacts_manifest.json` | Manifest of all artifacts |
| `preprocessing_report.json` | Machine-readable overall report |
| `preprocessing_report.md` | Markdown summary |
| `preprocessing_report.html` | HTML report |

### Image outputs under `<run_dir>/outputs/`

| File | Description |
|------|-------------|
| `stacked_linear.fits` | Linear mono or L-channel stack |
| `stacked_rgb.fits` | Linear RGB stack (OSC) |
| `stacked_rgb_bge.fits` | RGB after BGE (if active) |
| `stacked_rgb_pcc.fits` | RGB after PCC (if active) |
| `stacked_rgb_hms.fits` | Stretched RGB stack (if HMS active) |
| `calibrated/cal_NNNNN.fit` | Calibrated individual frames (if calibration active) |

---

## Known Limitations

- **No Resume**: Raw Stack jobs are not resumable. There is no phase resume function.
- **Manual Frame Overrides**: The data model is prepared; the interactive frame table remains a next development step.
- **HMS without PCC**: With `require_successful_pcc: true` (default), HMS is skipped if PCC fails. Can be set to `false` in the Parameter Editor to run HMS independently.
- **Darkflats**: `darkflats_dir` is available as a configuration field but is not yet processed at the same depth as Bias/Dark/Flat.
