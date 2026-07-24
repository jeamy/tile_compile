# GUI3 User Guide (English)

This guide describes the practical workflow with the GUI3 web interface: from installation through scanning input data, adjusting parameters, to starting and monitoring a run.

## Overview

GUI3 consists of three main areas:

| Tab | Sub-Tabs | Purpose |
|-----|----------|---------|
| **Processing** | Input & Scan, Parameter, Run Monitor | Main workflow: scan input, adjust parameters, start and monitor runs |
| **Tools** | Raw Stack, Astrometry, PCC | Standalone tools for preprocessing, plate solving, and color calibration |
| **History** | Run History | Browse, compare, and generate reports for completed runs |

---

## 1. Installation and Startup

### Release Bundle

1. Download the release ZIP from [GitHub Releases](https://github.com/jeamy/tile_compile/releases).
2. Extract and run the launcher:
   - **Linux**: `./start_gui3.sh`
   - **macOS**: Double-click `start_gui3.command` (or run `./start_gui3.command` in Terminal)
   - **Windows**: Double-click `start_gui3.bat`
3. On first launch, all application files are copied to `~/tilecompile/` (or `%USERPROFILE%\tilecompile\` on Windows).
4. The browser opens automatically at `http://127.0.0.1:8080/ui/`.

> **macOS note:** If Gatekeeper blocks the launcher: open `System Settings → Privacy & Security`, scroll down, and explicitly allow the blocked entry.

> **After installation:** The downloaded archive and extracted folder can be deleted. All files have been copied to your user directory. On updates, only application files are replaced — user data (runs, catalogs) is preserved.

---

## 2. Scanning Input (Input & Scan)

The first step is scanning the FITS input data.

### Steps

1. Select **Tab Processing → Sub-Tab Input & Scan**.
2. **Input folder**: Enter the path to the directory containing FITS light frames (e.g., `/data/M31/lights`).
3. **File pattern**: Default is `*.fits`. Alternatively `*.fit` or `light_*.fits`.
4. **Output folder (Runs-Dir)**: Target directory for run outputs. A directory under `~/tilecompile/runs/` is suggested by default.
5. **Run name**: Optional name for the run folder (e.g., `M31_altaz_test`).
6. **Parameters**:
   - **Frames Minimum**: Minimum number of frames, otherwise abort (default: 50).
   - **Max. Frames**: 0 = use all frames.
   - **Sort order**: `numeric`, `alphabetic`, or `timestamp`.
   - **Color mode**: `OSC` (One-Shot-Color / Bayer) or `MONO`.
   - **Compute checksums**: Optional, for integrity verification.
7. **Calibration** (optional): Specify bias, dark, and flat folders and enable the respective checkboxes.
8. Click **Start Scan**.

### Scan Result

After completion, the scan result card shows:
- Number of frames found
- Resolution and color mode of the frames
- Frame sizes and checksums (if enabled)

### Run Queue (for MONO)

MONO users can add multiple input folders (L/R/G/B) via the Run Queue. Each queue entry is processed as a separate channel.

---

## 3. Adjusting Parameters (Parameter Studio)

> **Tip:** For your first steps and getting familiar with the interface, the default settings can be kept as-is. This step can be skipped.

After scanning, switch to the **Parameter** sub-tab.

### Loading Configuration

- **Load example config**: Choose one of the bundled example YAMLs via the configuration selector (e.g., `M42.global_medium.yaml`).
- **Empty configuration**: Start with default values.
- **Upload existing config**: Upload your own YAML file.

### Parameter Categories

The Parameter Studio is organized into categories:

| Category | Key Parameters |
|----------|---------------|
| **Pipeline** | `pipeline.mode` (production/preview), `method` (aqmh/classic_tile_compile) |
| **Registration** | `registration.allow_rotation`, `registration.transform_model` (similarity/affine), `registration.engine` |
| **AQMH** | `aqmh.enabled`, `aqmh.pyramid.scales`, `aqmh.cherry_pick.enabled`, `aqmh.storage.max_resident_maps` |
| **Reconstruction** | Tile geometry, reconstruction mode |
| **Stacking** | `stacking.method` (sigma_clip/median/average), sigma-clip parameters |
| **Debayer** | `data.bayer_pattern` (RGGB/BGGR/GBRG/GRBG) |
| **Astrometry** | `astrometry.astap_bin`, `astrometry.astap_data_dir` |
| **BGE** | `bge.method`, `bge.enabled` (legacy), `bge.fit.method`, `bge.autobge`, `bge.autotune` |
| **PCC** | `pcc.source` (auto/siril/vizier_gaia/vizier_apass), `pcc.mag_limit`, `pcc.siril_catalog_dir` |
| **HyperMetric Stretch** | `hypermetric_stretch.enabled`, color strategy, convergence |
| **Calibration** | Bias/Dark/Flat paths |
| **Assumptions** | `frames_min`, `frames_reduced_threshold` |
| **Runtime Limits** | `hard_abort_hours`, `acceleration_backend` |

### Workflow

1. Expand a parameter category and adjust values.
2. Click **Validate** to check the configuration against the schema.
3. On validation errors: review error messages and correct.
4. Click **Save** to save the configuration.

> **Tip:** For initial attempts, the default configuration with `method: aqmh` works well. The most important adjustments are `registration.allow_rotation` (keep `true` for Alt/Az mounts) and `data.bayer_pattern` (cross-check with FITS headers).

### Explain Panel

Click on any parameter label to see a detailed explanation in the **Explain** panel on the right side of the grid. The panel shows the parameter's purpose, permitted range, default value, and interactions with other parameters.

### Presets

Use the **Preset** selector in the editor toolbar to load bundled example configurations. Click **Reload** to refresh the preset list and **Apply** to load the selected preset into the current draft.

### AI Recommendations

The Parameter page has a second tab: **AI Recommendations**. This provides AI-powered configuration recommendations based on scan data, equipment info, and imaging conditions.

1. Switch to the **AI Recommendations** tab.
2. Fill in the form: object name, camera type, mount type, telescope, conditions, and goals.
3. Under **Tools → AI & API**, select a **Provider** and **Model**. PI supports many providers (Anthropic, OpenAI, Google, DeepSeek, Groq, Mistral, xAI, OpenRouter, and more). Enter the key for the chosen provider, select **Save Key**, then use **Fetch status** to verify it. A key is required before analysis can run.
4. Return to **AI Recommendations** and click **Create AI Analysis**. The request runs in the background — tab switching is safe.
5. Review the recommendations: each suggestion shows the parameter path, new value, and a brief rationale.
6. Select individual recommendations with checkboxes, or use **Apply All** to apply all.
7. Use **PI Preview** to validate the planned changes as a YAML diff without saving.
8. Use **Apply PI** to save the last valid PI Preview as a new config revision.

> The AI sidecar (`tile_compile_pi_agent`) must be running for analysis to work. The key is kept in local PI AuthStorage, not in pipeline configuration, run data, or logs. Alternatively, it can be provided through a provider variable in a local `.env` file; the complete current list of supported key and credential variables is in [`.env.example`](../.env.example). A `401 invalid x-api-key` error means the provider rejected the key.

### Situation Assistant

Below the parameter grid, the **Situation-Assistent** panel offers scenario-based quick adjustments:

1. Select one or more imaging scenarios (e.g., "Alt/Az mount", "Light pollution", "Short exposure").
2. The assistant calculates parameter deltas for each selected scenario.
3. Click **Apply** to merge the deltas into the current draft.
4. The changes appear immediately in the parameter grid and YAML diff.

---

## 4. Starting and Monitoring a Run (Run Monitor)

### Starting a Run

1. Switch from the **Parameter** sub-tab to **Run Monitor**, or click **Next ▶** directly from the Input & Scan tab.
2. In the Run Monitor, click **Start Run**.
3. The runner starts as a background process. The Run Monitor shows progress in real time.

### Run Monitor

The Run Monitor displays:

- **Current phase**: Which pipeline phase is running (SCAN_INPUT → REGISTRATION → ... → PCC → DONE)
- **Phase progress**: Status of each phase (pending, running, ok, skipped, error)
- **Warnings and errors**: Prominently displayed in the warning banner
- **Run statistics**: Processed frames, estimated time remaining (if available)
- **Image preview**: Shows the latest output image (stacked, BGE, PCC, or HMS) during and after the run

Below the phase list, three tabs provide additional tools:

#### Resume Tab

The Resume tab contains the phase-based resume mechanism (see [Resume](#resume) below).

#### Run-Chat Tab

The Run-Chat tab provides AI-assisted diagnosis of the current run:

1. Type a question or describe a problem (e.g., "stars are elongated" or "background is uneven").
2. The AI analyzes run artifacts, event logs, and phase results.
3. The response includes image observations, likely causes, checks to perform, and recommendations.
4. If the AI suggests config changes, a PI Preview can be generated and applied directly to the Resume YAML.
5. A resume recommendation may suggest continuing from a specific phase.

> Run-Chat history is persisted per run and restored when switching back to the Run Monitor.

#### Live Log Tab

The Live Log tab shows real-time log output from the runner process, including phase events, warnings, and error messages.

### Aborting a Run

- Click **Stop** to abort the running run.

### Resume

After completion or an abort, a run can be resumed from a selected phase. The
existing run directory and its intermediate artifacts are reused.

1. In the phase list, click the phase from which processing should continue,
   for example `ASTROMETRY`, `BGE`, `PCC`, or `HYPERMETRIC_STRETCH`.
2. The **Resume** section opens and shows the selected phase in a badge on the
   right.
3. Review the YAML in **Config YAML**. Use **Load current config** to restore
   the run configuration or select and load a stored **Config Revision**.
4. Click **Resume** in the Resume section.
5. GUI3 stores the selected configuration as a revision, starts the runner at
   the selected phase, and returns to live monitoring.

Phases before the selected resume point are not recomputed. Their existing
artifacts must therefore still be present and compatible with the selected
configuration.

> Common resume points: `ASTROMETRY` re-runs plate solving, `BGE` re-runs
> background extraction, `PCC` re-runs photometric color calibration, and
> `HYPERMETRIC_STRETCH` creates a new stretched output from the existing linear
> PCC result.

### Configuring HyperMetric Stretch During Resume

GUI3 provides an interactive HMS preview when `HYPERMETRIC_STRETCH` is selected
as the resume phase.

1. Click `HYPERMETRIC_STRETCH` in the phase list.
2. In the Resume section, click **Configure HMS**. The button is displayed
   immediately to the left of the **HyperMetric Stretch** phase badge.
3. Wait for the first preview. GUI3 normally uses
   `outputs/stacked_rgb_pcc.fits`. If that file is unavailable, it uses the
   complete `pcc_R.fit`, `pcc_G.fit`, and `pcc_B.fit` channel set.
4. Adjust the HMS parameters. A new proxy preview is calculated automatically
   after each valid change.
5. Inspect the preview, histogram, resolved source, calculated log D, anchor,
   and black/white clipping percentages.
6. Click **Apply & start resume** to write the displayed HMS values into the
   Resume YAML and immediately resume from `HYPERMETRIC_STRETCH`.

Use **Reset** to restore the values that were loaded when the dialog opened.
Use **Cancel** or the close button to discard changes without starting a
resume.

#### Preview Navigation

- Drag the image to pan.
- Use the mouse wheel to zoom.
- Double-click the image to fit it into the preview area.
- The RGB histogram is displayed below the image on a logarithmic scale.

The preview operates on a downsampled proxy with a maximum long edge of 1600
pixels. The final resume runs HMS on the full-resolution PCC data.

#### HMS Parameters

Hover over a parameter label or its information icon to see a description and
the permitted range.

| Parameter | Effect |
|-----------|--------|
| **Mode** | `Ready to use` applies display-oriented output scaling and highlight soft clipping. `Scientific` avoids those display-oriented finishing operations. |
| **Sensor profile** | Selects RGB luminance weights used for anchor calculation, automatic log D, star-pressure estimation, and output scaling. `Automatic` uses the fallback profile. |
| **Fallback profile** | Profile used when the primary profile is `Automatic`. An unknown explicit profile falls back to Rec.709 in the core. |
| **Adaptive anchor** | Derives the stretch anchor from image content instead of using the statistical anchor path. |
| **Target background** | Desired background brightness after stretching. Higher values produce a brighter background. |
| **Protect B** | Shapes the hyperbolic curve. Higher values compress and protect bright regions more strongly. |
| **Convergence power** | Controls how quickly RGB channels converge in brighter regions. |
| **log-D mode** | `Automatic` calculates the stretch strength from the proxy. `Fixed` enables the fixed log-D value. |
| **Fixed log D** | Manual logarithmic stretch strength. Higher values reveal faint structures more strongly. |
| **Color strategy** | Selects automatic or fixed color handling. |
| **Fixed color strategy** | In ready-to-use mode, negative values increase shadow convergence; positive values reduce color grip. |
| **Color grip** | Controls preservation of the original color ratios in scientific mode. |
| **Shadow convergence** | Blends dark regions more strongly toward per-channel stretching in scientific mode. |
| **Linear expansion** | Adds low-end linear expansion in scientific mode. |

Numeric fields reject values outside their displayed ranges. An invalid value
does not start a preview and cannot be applied. If focus leaves such a field,
GUI3 limits it to the nearest allowed boundary.

#### Applying HMS Values

**Apply & start resume** changes only the known fields under
`hypermetric_stretch` in the currently loaded Resume YAML. Other configuration
sections remain unchanged. GUI3 then uses the normal Resume mechanism, which:

- snapshots the previous run configuration,
- creates a configuration revision,
- starts the runner with `--from-phase HYPERMETRIC_STRETCH`, and
- displays the new job in the Run Monitor.

The preview itself never modifies `config.yaml` or any FITS artifact. If no
complete PCC RGB artifact is available, the preview reports an error and no
resume is started.

### Configuring AutoBGE During Resume

Select `BGE` in the phase list and click **Configure BGE** to the left of the
BGE badge. The dialog uses `stacked_rgb_solve.fits`, with `stacked_rgb.fits` as
fallback, together with `canvas_mask.fits`. It never uses an already
BGE-corrected image as input.

The dialog provides **Original**, **Corrected**, and **Background** views. Drag
to pan, use the mouse wheel to zoom, and double-click to fit the image. Colored
overlay points show the samples used by AutoBGE.

All AutoBGE controls include tooltips and enforce their valid ranges. Important
controls include sample count, polynomial degree, RBF smoothing, downsample
scale, patch size and estimator, working-space transform, bright exclusion,
random seed, and safety guards.

To exclude real dark structures such as dark nebulae from background sampling:

1. Click **Draw exclusion**.
2. Click at least three vertices on the image.
3. Double-click to close the polygon and recalculate the preview.
4. Use **Clear exclusions** to remove all polygons.

Exclusions affect sample selection only; they never blank pixels in the output.
They are stored as normalized coordinates in
`bge.autobge.exclusion_polygons`, so the full-resolution resume uses the same
regions.

#### Adding Manual Sample Points

In addition to the automatic sample points, manual points can be placed to
force specific dark background regions into the model:

1. Click **Add points**. The cursor changes to a crosshair.
2. Click on the desired locations in the image. Each click adds a point.
   Points are displayed as white circles with a red outline.
3. Click **Add points** again or double-click to exit point mode.
4. Use **Clear points** to remove all manual points.

Manual points are always included in the sample set and bypass random
downselection. They are stored as normalized float coordinates `[0..1]` under
`bge.autobge.user_sample_points` in the YAML, making them
resolution-independent. Saved points are automatically loaded when reopening
the dialog or resuming.

#### Guard Rejection Reasons

When the safety guards reject the preview, the status line shows the specific
rejection reason along with an actionable hint. Typical reasons are:

| Reason | Meaning | Recommended Action |
|--------|---------|-------------------|
| **Flatness worsened** | The background model is less flat after correction | Lower `poly_degree`, raise `rbf_smooth`, or draw exclusion polygons around structures |
| **Background chroma worsened** | The color distribution in the background has degraded | Raise `rbf_smooth`, lower `poly_degree`, or add more manual points on dark background |
| **Slope worsened** | The gradient in the background model has degraded | Place manual points on the faint gradient, raise `num_sample_points`, or lower `rbf_smooth` |

Additionally, the info line below the image shows which channels (R, G, B)
were rejected. If `apply_guards` is disabled, the preview can be applied
despite rejection — this is not recommended.

Click **Apply & start resume** to set `bge.method: autobge`, update the AutoBGE
configuration in the Resume YAML, and resume from `BGE`. **Reset** restores the
parameters, polygons, and manual points loaded when the dialog opened. Preview
calculations do not modify FITS files or `config.yaml`.

---

## 5. Viewing Results

### Output Files

After completion, results are located under `<runs_dir>/<run_name>/`:

```
runs/<run_id>/
├── outputs/
│   ├── stacked.fits              # Linear sum image
│   ├── reconstructed_L.fit       # MONO reconstruction
│   ├── stacked_rgb.fits          # OSC RGB
│   ├── stacked_rgb_solve.fits    # WCS solved
│   ├── stacked_rgb_bge.fits      # After BGE (before PCC)
│   ├── stacked_rgb_pcc.fits      # After PCC
│   └── stacked_rgb_hms.fits      # After HyperMetric Stretch (optional)
├── artifacts/
│   ├── report.html               # Diagnostic report
│   ├── report.css
│   ├── *.png                     # Diagrams/heatmaps
│   ├── normalization.json
│   ├── global_registration.json
│   ├── bge.json
│   └── ...
├── logs/
│   └── run_events.jsonl          # Event timeline
└── config.yaml                   # Run configuration snapshot
```

### Diagnostic Report (report.html)

An HTML diagnostic report can be generated via the Run Monitor or Run History:

- **Run Monitor**: Click **Generate Stats**.
- **Run History**: Select a run and generate report.
- **CLI**: `./tile_compile_cli generate-report runs/<run_id>`

The report includes:
- Normalization/background trends
- Global quality distributions and weights
- Registration drift/CC/rotation
- Tile and reconstruction heatmaps
- BGE diagnostics
- Validation metrics (including tile-pattern indicators)
- Pipeline timeline and frame-usage funnel

---

## 6. Setting Up Astrometry and PCC

Astrometry (plate solving) and PCC (photometric color calibration) are optional phases that require external data.

### ASTAP (for Astrometry / WCS Plate Solving)

1. **Install ASTAP**: Download the binary from [https://www.hnsky.org/astap.htm](https://www.hnsky.org/astap.htm).
2. **Download star database**: At minimum D50 (for deep-sky) or G18 (for wide-field).
3. **Configure in GUI3**:
   - Tab **Tools → Sub-Tab Astrometry**
   - **ASTAP CLI**: Path to the `astap` binary (e.g., `/usr/local/bin/astap`)
   - **Star Database Dir**: Path to the catalog directory (e.g., `/usr/local/share/astap`)
   - Click **Detect ASTAP** to verify the installation.
   - Alternatively: use **Install CLI** and **Download Catalog** directly from the GUI.

> ASTAP catalogs can also be downloaded directly via GUI3 (Tools → Astrometry → Download Catalog).

### Siril Gaia DR3 XP Catalog (for PCC)

1. **Catalog data**: The Siril Gaia DR3 XP sampled catalog is required for photometric color calibration.
2. **Configure in GUI3**:
   - Tab **Tools → Sub-Tab PCC**
   - **Catalog Dir**: Path to the catalog directory (default: `~/.local/share/siril/siril_cat1_healpix8_xpsamp/`)
   - Click **Download Missing** to download missing catalog chunks (48 chunks, ~2 GB total).
   - The download runs in the background with a progress indicator.

> If the catalog has already been downloaded by [Siril](https://siril.org/), the same folder can be reused.

3. **PCC parameters** (in Parameter Studio):
   - `pcc.source`: `auto` (recommended), `siril`, `vizier_gaia`, or `vizier_apass`
   - `pcc.mag_limit`: Magnitude limit for stars (default: 14.0)
   - `pcc.siril_catalog_dir`: Path to the Siril catalog (if different from default)

### When Catalogs Are Missing

If ASTAP or the Siril catalog are not installed:
- Core reconstruction (registration, AQMH, stacking, debayer) still works.
- The `ASTROMETRY` and `PCC` phases will be marked as `skipped` or fail, depending on configuration.
- BGE (Background Gradient Extraction) works independently of external catalogs.

---

## 7. Raw Stack (Standalone Preprocessing)

The **Raw Stack** sub-tab (under Tools) offers separate linear preprocessing of FITS lights up to a finished stack, independent from the Tile Compile Run Studio.

Pipeline steps:
1. Input Scan
2. Calibration (Bias/Dark/Flat)
3. CFA/Mono Prep
4. Reference Selection
5. Registration
6. Quality Analysis
7. Frame Filtering
8. Stacking (Sigma-Clip/Median/Winsor)
9. Astrometry (optional)
10. BGE (optional)
11. PCC (optional)
12. HyperMetric Stretch (optional)
13. Report

> Detailed documentation: [docs/raw_stack_gui_de.md](raw_stack_gui_de.md)

---

## 8. Run History

Under **Tab History → Sub-Tab Run History**, completed runs can be browsed:

- List of all runs with status, date, frame count
- View run details (configuration, phase status, artifacts)
- Generate reports
- Compare runs
- Open run directory in file manager
