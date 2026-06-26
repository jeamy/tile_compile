# Raw Stack – GUI Documentation

Raw Stack is a standalone menu item in the Tile-Compile GUI for linear preprocessing of FITS light frames through to a stacked image. The process runs completely separately from the normal Tile-Compile Run Studio. It shares algorithms, artifact infrastructure, the Run Monitor, and Parameter Studio interaction patterns, but launches as a separate tool runner and does not appear in the normal Tile-Compile phase list.

Parameter input happens in the Raw Stack menu through visible form fields and the **Parameters** section. This section is structured like Parameter Studio: grouped parameters, editable values, JSON editor, Validate, and Reset. Raw Stack uses its own preprocessing configuration; relevant parameters can still be imported from a loaded Tile-Compile YAML.

---

## Overview: Processing Pipeline

```
Input Scan
  -> Calibration (Bias / Dark / Flat)
  -> CFA/Mono Prep (normalization, Bayer)
  -> Reference Selection
  -> Registration (global affine warp matrices)
  -> Quality Analysis (stars, FWHM, eccentricity, correlation)
  -> Frame Filtering (auto-rejection, manual overrides)
  -> Stacking (Sigma/Median/Winsor, normalization, weighting)
  -> Astrometry (ASTAP, optional)
  -> BGE – Background Gradient Extraction (optional)
  -> PCC – Photometric Color Calibration (optional)
  -> HyperMetric Stretch (optional)
  -> Report
```

Every phase emits a `phase_start` and `phase_end` event to `artifacts/preprocess/events.jsonl`. Phases that are not configured or whose prerequisites are missing are ended as `skipped` – they do not abort the run.

---

## Section: Input & Scan

### Input Directory

The folder containing the light frames (FITS files). Multiple folders can be added via the Run Queue. For a single session, one folder in the **Input Directory** field is sufficient.

### File Pattern

Filters which files within the folder are treated as light frames. Default: `*.fits`. Simple glob patterns are supported, e.g. `*.fit`, `light_*.fits`.

### Output Directory (Runs Dir)

Parent directory under which Raw Stack creates the run folder. The full path is assembled from output directory + run name + timestamp: `<runs_dir>/<run_name>_YYYYMMDD_HHMMSS`.

### Run Name

Optional custom prefix for the run folder. If left empty, `run` is used.

### Frames Minimum / Max Frames

- **Frames Minimum**: Guardrail threshold. A warning is logged if the frame count falls below this value.
- **Max Frames**: Limits the number of processed frames. `0` means unlimited.

### Color Mode

- **OSC**: CFA/Bayer sensor (e.g. DSLR, color camera). The Bayer debayer path is activated.
- **MONO**: Monochrome sensor. No Bayer handling, no RGB stack.
- Default: `auto` – detected from the FITS header (`COLORTYP`, `BAYERPAT`, `NAXIS`).

### Bayer Pattern

Only used for OSC. `auto (from FITS header)` reads `BAYERPAT` or `COLORTYP`. Explicit values: `RGGB`, `GBRG`, `GRBG`, `BGGR`. An incorrect pattern will produce color artifacts in the stack.

### Checksums

Optional MD5/SHA scan during the input scan. Increases scan time; recommended only for integrity verification of large archives.

---

## Section: Run Queue

The Run Queue allows combining multiple input folders with different filters (L, R, G, B, Ha, ...) into a single Raw Stack run. Each queue entry consists of:

- **Filter**: Channel label, e.g. `L`, `R`, `G`, `B`, `Ha`. For OSC datasets, `OSC` is set automatically.
- **Input Dir**: Directory with the light frames for this channel.
- **Pattern**: Optional file pattern for this channel.
- **Run Label**: Optional label for the queue entry.
- **Active**: Deactivates an entry without deleting it.

New entries are added via the `+` button from the current input directory field. Entries can be removed with `-`.

> **Note:** Clicking `Start` directly without a queue entry starts a single-folder run with the currently entered input directory.

---

## Section: Calibration

Calibration corrects sensor artifacts before registration. All three stages can be activated independently.

### Bias

Corrects the constant electronics offset of the sensor (readout noise pedestal). Applied before dark and flat.

- **Checkbox**: Enable bias calibration.
- **Folder or Master**: Either specify a folder with bias frames (an average master is built automatically) or a single master bias FITS file.
- **Browse**: Select folder or file via dialog.

### Dark

Corrects thermal noise and hot pixels. If bias is also enabled, the dark master is automatically bias-corrected before being subtracted from the lights.

- **Dark Auto-Select** (in Parameter Editor): With `calibration.dark_auto_select: true`, darks are automatically filtered by exposure time from the FITS header. Tolerance configurable via `calibration.dark_match_exposure_tolerance_percent` (default: 8 %). Optionally also by sensor temperature (`calibration.dark_match_use_temp`, tolerance: `calibration.dark_match_temp_tolerance_c`).

### Flat

Corrects vignetting and pixel sensitivity variations. The flat master is normalized to its median (pixel / median), so that lights retain unchanged average brightness after division.

> **Tip:** Bias, Dark, and Flat are all optional. If no calibration directories are specified, light frames are processed directly (phase is marked as `skipped`, no error).

---

## Section: Quality

Controls automatic frame selection based on quality metrics.

| Field | Description | Default |
|-------|-------------|---------|
| **Mode** | `auto` – FWHM k = configured `max_fwhm_sigma`; `strict` – FWHM k = 1.5 (tight); `relaxed` – FWHM k = 3.0 (loose); `off` – no filtering | `auto` |
| **Min stars** | Minimum number of detected stars per frame | `30` |
| **Min correlation** | Minimum cross-correlation with the reference frame (0–1) | `0.75` |
| **Max FWHM sigma** | Maximum deviation of FWHM from the median in sigma units | `2.0` |
| **Max eccentricity** | Maximum mean eccentricity of stars (0 = circle, 1 = line) | `0.65` |

Rejected frames are recorded in `artifacts/preprocess/rejected_frames.txt` and in `frame_quality.csv`. In the Run Monitor they are marked as `excluded`.

---

## Section: Stack

Configures rejection, normalization, and weighting for the stacking step.

| Field | Description | Default |
|-------|-------------|---------|
| **Rejection** | Per-pixel rejection method | `sigma` |
| **Low / High** | Sigma thresholds (for sigma rejection) | `3.0 / 3.0` |
| **Normalization** | Frame normalization before stacking: `addscale` (additive + scaling), `background` (background only), `median` (median), `none` | `addscale` |
| **Weighting** | Frame weighting: `quality` (proportional to quality score), `uniform` | `quality` |

---

## Section: Postprocess

Four optional post-processing phases applied to the stacked image after stacking. All are active by default.

### Astrometry

Solves the WCS (World Coordinate System) of the stacked image via ASTAP. Result: `artifacts/preprocess/preprocessing_registration.json` with WCS metadata. Prerequisite: ASTAP binary and local star catalog (configured via `astrometry.astap_bin` / `astrometry.astap_data_dir` in the Parameter Editor).

### BGE – Background Gradient Extraction

Extracts and subtracts background gradients from the RGB stack. Produces `outputs/stacked_rgb_bge.fits` and `artifacts/preprocess/bge_diagnostics.json`. Only active when an RGB stack is available.

BGE uses the same BGE configuration (`bge.*`) and seeing-based tile geometry (`tile.*`) known from Tile Compile. For OSC/CFA data, FWHM is mapped to the full debayer scale so the sampling geometry remains comparable to Tile Compile. The effective BGE and tile parameters are written to `bge_diagnostics.json`.

### PCC – Photometric Color Calibration

Photometrically calibrates the color channels of the RGB stack based on star colors from the WCS. Requires successful astrometry. Produces `outputs/stacked_rgb_pcc.fits` and `artifacts/preprocess/pcc_diagnostics.json`.

### HyperMetric Stretch (HMS)

Non-linearly stretches the linear RGB stack into the visually displayable range. HMS uses the best available input: PCC output > BGE output > linear RGB stack.

**Output:** `outputs/stacked_rgb_hms.fits`<br>
**Diagnostics:** `artifacts/preprocess/hms_diagnostics.json`

**HMS detail parameters** (in Parameter Editor under `hypermetric_stretch.*`):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `require_successful_pcc` | Only run HMS if PCC was successful | `true` |
| `mode` | Operating mode: `ready_to_use` (visual) or `scientific` | `ready_to_use` |
| `sensor_profile` | Color profile for luminance weights (e.g. `rec709`, `veralux`) | `rec709` |
| `fallback_profile` | Fallback if `sensor_profile` is not found | `rec709` |
| `adaptive_anchor` | Determine anchor automatically from image histogram | `true` |
| `target_bg` | Target value for background after stretch (0–1) | `0.15` |
| `protect_b` | Protection factor for the blue channel (color balance) | `6.0` |
| `convergence_power` | Convergence exponent of the stretch curve | `3.5` |
| `log_d_mode` | How to determine the Log-D value: `auto` or `fixed` | `auto` |
| `color_strategy` | Color strategy: `fixed` or `adaptive` | `fixed` |
| `color_grip` | Color saturation strength (0 = neutral) | `1.0` |
| `output_rgb` | Output filename in the `outputs/` folder | `stacked_rgb_hms.fits` |

---

## Section: Parameters (Parameter Editor)

The Parameters section displays the preprocessing parameters grouped by backend schema and includes the JSON editor for the complete effective configuration. It allows editing all parameters that are not exposed as individual form fields. Interaction and structure follow Parameter Studio, but the stored values belong to the separate Raw Stack configuration.

**Buttons:**
- **Defaults**: Loads the default configuration from the backend (`GET /api/tools/preprocessing/defaults`).
- **Validate**: Sends the current JSON configuration to the backend for validation (`PATCH /api/tools/preprocessing/parameters`).
- **Reset**: Resets the editor to the stored defaults.

**Priority at start**: Values from visible form fields (postprocess toggles, quality, stacking, calibration) override the corresponding JSON values in the editor. HMS details, BGE/tile parameters, dark matching tolerances, report formats, and other advanced parameters come exclusively from the editor.

### Import from loaded Tile-Compile YAML

When Raw Stack defaults are loaded, the GUI reads the currently loaded Tile-Compile configuration and imports only parameters that are meaningful and implemented for the Raw Stack process. Visible Raw Stack fields can still override these values afterwards.

| Source in Tile Compile | Target in Raw Stack | Notes |
|------------------------|---------------------|-------|
| `runtime_limits.parallel_workers` | `runtime_limits.parallel_workers` | controls parallelism |
| `runtime_limits.memory_budget` | `runtime_limits.memory_budget` | controls memory-aware sub-batches |
| `normalization.mode` | `stacking.normalization` | only `background`, `median`, `addscale`, `none` |
| `stacking.weighting` | `stacking.weighting` | only `quality`, `uniform` |
| `stacking.cosmetic_correction` | `stacking.cosmetic_correction` | final cosmetic correction |
| `stacking.cosmetic_correction_sigma` | `stacking.cosmetic_correction_sigma` | sigma for final correction |
| `stacking.per_frame_cosmetic_correction` | `stacking.per_frame_cosmetic_correction` | cosmetic correction before warp/stack |
| `stacking.per_frame_cosmetic_correction_sigma` | `stacking.per_frame_cosmetic_correction_sigma` | sigma for per-frame correction |
| `stacking.sigma_clip.sigma_low` | `rejection.low` | lower sigma rejection |
| `stacking.sigma_clip.sigma_high` | `rejection.high` | upper sigma rejection |
| `stacking.sigma_clip.max_iters` | `rejection.max_iters` | iterative sigma clipping |
| `stacking.sigma_clip.min_fraction` | `rejection.min_fraction` | minimum remaining sample fraction |
| `astrometry.*` | `astrometry.*` | includes `enabled`, ASTAP paths, search radius |
| `astrometry.enabled` | `postprocess.astrometry` | presets the postprocess toggle |
| `bge.*` | `bge.*` | complete BGE configuration |
| `bge.method` | `postprocess.bge` | `none` disables the postprocess toggle, `classic`/`autobge` enable it |
| `bge.enabled` | `postprocess.bge` | legacy fallback when `bge.method` is absent |
| `tile.*` | `tile.*` | tile geometry for BGE sampling |
| `pcc.*` | `pcc.*` | complete PCC configuration |
| `pcc.enabled` | `postprocess.pcc` | presets the postprocess toggle |
| `hypermetric_stretch.*` | `hypermetric_stretch.*` | complete HMS configuration |
| `hypermetric_stretch.enabled` | `postprocess.hypermetric_stretch` | if present, presets the HMS toggle |

Tile-Compile-specific phase parameters such as Tile Reconstruction, State Clustering, Synthetic Frames, Common Overlap, or normal Run Studio resume/template metadata are not imported. Raw Stack only uses the shared algorithms and the compatible parameters listed above.

**All parameter groups** (visible in the editor and in the separate Parameter Studio tab):

| Group | Contents |
|-------|----------|
| `input` | `lights_dir`, `bias_dir`, `darks_dir`, `flats_dir`, `darkflats_dir`, `input_mode`, `raw_formats` |
| `calibration` | `use_bias`, `use_dark`, `use_flat`, master options, dark auto-select, tolerances |
| `cfa_mono` | `input_mode`, `bayer_pattern`, `cfa_mode`, `mono_mode` |
| `registration` | `registration_reference` |
| `quality_filter` | `mode`, `min_stars`, `max_fwhm_sigma`, `max_eccentricity`, `min_correlation` |
| `stacking` | `rejection.method`, `rejection.low/high`, `stacking.normalization`, `stacking.weighting` |
| `postprocess` | `astrometry`, `bge`, `pcc`, `hypermetric_stretch` |
| `bge_tile` | `bge.*`, `tile.*` for BGE sampling and surface fitting |
| `hypermetric_stretch` | all HMS detail parameters |
| `report` | `report.detailed`, `report.formats` |
| `runtime_limits` | `runtime_limits.parallel_workers`, `runtime_limits.memory_budget` |

---

## Section: Monitor (inline)

The inline Monitor section in `raw-stack.html` shows after starting the run:

- **Phase list**: Status of each phase (ok / skipped / error) as compact chips.
- **Log**: Latest events from `events.jsonl` as a scrollable terminal window.
- **Artifacts**: Links to generated artifacts (CSV, JSON, HTML report).
- **Report button**: Opens `preprocessing_report.html` directly in the browser.
- **Run Monitor button**: Opens `run-monitor.html?preprocessing_job_id=<id>` for the full monitor view.

---

## Run Monitor (full view)

The normal Run Monitor (`run-monitor.html`) detects the URL parameter `preprocessing_job_id` and switches to preprocessing mode:

- **Phase list**: Shows all preprocessing phases with status.
- **Live Log**: Loads `artifacts/preprocess/events.jsonl` via `/api/runs/{run_id}/artifacts/view`.
- **Artifact list**: All files under `artifacts/preprocess/` via `/api/runs/{run_id}/artifacts`.
- **Artifact viewer**: Clicking a file opens its content directly (JSON, CSV, JSONL, HTML).
- **Raw serve**: HTML reports are served directly as HTML via `/api/runs/{run_id}/artifacts/raw/...`.

> **Important:** Resume, revision, and template functions of the normal Run Monitor are disabled for Raw Stack jobs. Raw Stack is not a resumable run but a standalone tool runner.

---

## Phase Reference

| Phase | Description | Skippable |
|-------|-------------|-----------|
| `INPUT_SCAN` | Discover frames, check dimensions, determine color mode | No |
| `CALIBRATION` | Apply Bias / Dark / Flat | Yes (if no calibration directory configured) |
| `CFA_CHANNEL_PREP` | Bayer normalization for OSC; single-channel normalization for Mono | No |
| `REFERENCE_SELECTION` | Select reference frame (best quality or temporal center) | No |
| `REGISTRATION` | Compute affine warp matrices via triangle star matching | No |
| `QUALITY_ANALYSIS` | Count stars, measure FWHM, eccentricity, correlation, saturation | No |
| `FRAME_FILTERING` | Exclude frames by quality (auto + manual overrides) | Yes (mode=off) |
| `STACKING` | Stack calibrated, registered frames | No |
| `ASTROMETRY` | Solve WCS via ASTAP | Yes (if disabled or ASTAP missing) |
| `BGE` | Extract and subtract background gradient | Yes (if disabled or no RGB stack) |
| `PCC` | Photometric color calibration | Yes (if disabled or no WCS) |
| `HYPERMETRIC_STRETCH` | Non-linear stretch of the RGB stack | Yes (if disabled or no RGB stack) |
| `REPORT` | Generate JSON / Markdown / HTML report | No |

---

## Artifact Reference

### Diagnostics under `artifacts/preprocess/`

| File | Contents |
|------|----------|
| `effective_config.json` | Effective configuration of the run |
| `frame_quality.csv` | Quality metrics per frame (stars, FWHM, eccentricity, correlation, status) |
| `rejected_frames.txt` | List of excluded frames with exclusion reason |
| `stacking_diagnostics.json` | Stacking parameters, weights, frames used |
| `bge_diagnostics.json` | BGE result (success/failure, extracted background gradient) |
| `pcc_diagnostics.json` | PCC result (color correction factors, number of reference stars used) |
| `hms_diagnostics.json` | HMS result (anchor, log-D, profile, color strategy) |
| `events.jsonl` | All phase events as JSONL (one JSON object per line) |
| `artifacts_manifest.json` | Manifest of all generated artifacts (type, phase, path) |
| `preprocessing_report.json` | Machine-readable overall report |
| `preprocessing_report.md` | Human-readable Markdown summary |
| `preprocessing_report.html` | HTML report with phase overview and metrics |

### Image outputs under `outputs/`

| File | Contents |
|------|----------|
| `stacked_linear.fits` | Linear stacked mono or L channel |
| `stacked_rgb.fits` | Linear RGB stack (OSC after debayer) |
| `stacked_rgb_bge.fits` | RGB stack after background correction (if BGE active) |
| `stacked_rgb_pcc.fits` | RGB stack after photometric color calibration (if PCC active) |
| `stacked_rgb_hms.fits` | Stretched RGB stack (if HMS active and successful) |
| `calibrated/cal_NNNNN.fit` | Calibrated individual frames (if calibration active) |

---

## Known Limitations and Next Steps

- **No Resume**: Raw Stack jobs are not resumable runs. There is no phase resume function.
- **No Tile Processing**: Tile grid, tile reconstruction, synthetic frames, and state clustering are not started. Raw Stack produces a simple stacked frame, not a tile-reconstructed one.
- **Manual Frame Overrides**: The data model is prepared. The interactive frame table with per-frame override is a next development step.
- **Dark Auto-Select**: Recommended when the dark folder contains frames with very different exposure times. For homogeneous darks, a simple folder without auto-select is sufficient.
- **HMS and PCC**: HMS uses `require_successful_pcc: true` by default. If PCC fails (no WCS, no reference stars), HMS is marked as `skipped`. This value can be set to `false` in the Parameter Editor to run HMS independently of PCC.
