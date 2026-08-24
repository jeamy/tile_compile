# Process Flow – Technical Data Flow of the System

## Pipeline objective

The system turns a set of calibrated astronomical single-frame inputs into a reproducible final product inside a shared geometric and photometric reference space.

From a technical perspective, the pipeline is organized into three major blocks:

- **Preparation and normalization**
  - validate inputs
  - unify geometry
  - normalize intensity levels
- **Quality modeling and reconstruction**
  - compute global metrics and dense AQMH quality maps
  - perform per-pixel AQMH weighted reconstruction
  - optionally use Classic Tile-Compile with local tile metrics, clustering and synthetic frames
- **Post-processing and calibration**
  - debayer
  - astrometry / WCS
  - optional BGE
  - PCC

The primary product is a linear stacked image. Depending on configuration and data mode, the run may also generate debayered, gradient-corrected and photometrically calibrated derivatives, plus structured diagnostics.

## Core terms

- **Run**
  - One full pipeline execution with its own run directory under `runs/<run_id>/`.
- **Phase**
  - One well-defined processing stage such as `REGISTRATION`, `AQMH_MAPS`, or `PCC`.
- **Artifact**
  - Persisted diagnostic or intermediate data, typically written under `artifacts/`.
- **Event timeline**
  - Chronological execution events written to `logs/run_events.jsonl`.
- **Assumptions thresholds**
  - `assumptions.frames_min` and `assumptions.frames_reduced_threshold` control whether the runner aborts, enters reduced mode, or runs the full pipeline.
- **Resume**
  - Existing run directories can be reused for supported downstream phases, especially `STACKING`, `ASTROMETRY`, `BGE`, `PCC`, and `HYPERMETRIC_STRETCH`.

---

## Overall flow

```text
Input frames (FITS)
   -> SCAN_INPUT
   -> REGISTRATION
   -> PREWARP
   -> CHANNEL_SPLIT
   -> NORMALIZATION
   -> GLOBAL_METRICS
   -> TILE_GRID (auxiliary geometry; reconstruction grid for Classic)
   -> COMMON_OVERLAP
   -> AQMH_MAPS (enum 19)
   -> AQMH_GLOBAL_QUALITY (enum 20)
   -> AQMH_RECONSTRUCTION (enum 21)
   -> AQMH_DIAGNOSTICS (enum 22)
      or [Classic] LOCAL_METRICS -> TILE_RECONSTRUCTION
   -> [Classic only, optional] STATE_CLUSTERING
   -> [Classic only, optional] SYNTHETIC_FRAMES
   -> STACKING
   -> [optional / data-dependent] DEBAYER
   -> ASTROMETRY
   -> [optional] BGE
   -> [optional] PCC
   -> [optional] HYPERMETRIC_STRETCH
   -> DONE
```

---

## Why AQMH is the default

Frame-level global scoring alone is usually insufficient for astrophotography
series because quality varies spatially. AQMH therefore computes a dense
per-frame quality map and weights every output pixel independently. This avoids
a fixed tile raster and overlap-add seams while still reacting to:

- location-dependent seeing variations
- local guiding or deformation artifacts
- border artifacts after warp or rotation
- uneven background or noise distributions

The original tile-based method remains available as **Classic Tile-Compile** via
`method: classic_tile_compile`. It approximates local quality with overlapping
tiles and is no longer the default.

---

## Phases in detail

## 0) Validate input (`SCAN_INPUT`)

**Input**

- one input path or multiple input directories
- FITS files with headers and acquisition metadata

**Processing**

- discover and enumerate input files
- validate headers, bit depth, image dimensions and color mode
- classify data as mono or OSC/CFA
- detect obvious exclusion cases
- verify that sufficient storage and workspace capacity are available

**Output**

- cleaned frame list
- scan summary with metadata, warnings and errors
- guardrails used by downstream run-start decisions

---

## 1) Global registration (`REGISTRATION`)

**Goal**

- bring all frames into one common geometric reference system

**Processing**

- select a reference frame
- estimate geometric transforms relative to the reference
- switch through fallback strategies if the primary registration path is not reliable enough
- persist registration metrics and transform parameters
- execute on CPU workers; this phase does not use the GPU

**Output**

- registered transform data per frame
- quality indicators such as correlation, drift, rotation or residual misalignment

---

## 2) Prewarp onto a common canvas (`PREWARP`)

**Goal**

- move all registered frames onto the same target canvas and pixel geometry

**Processing**

- apply the estimated transforms to a shared target area
- for OSC/CFA data: use CFA-safe warping via sub-plane logic so the Bayer pattern stays semantically stable
- enlarge the canvas when field rotation or translation exceeds the original bounds
- track offsets such as `tile_offset_x` and `tile_offset_y`
- use CUDA or OpenCL for full-frame warps when available, otherwise CPU

**Output**

- prewarped frames with unified geometry
- a consistent coordinate domain for AQMH and Classic downstream phases

---

## 3) Establish the channel model (`CHANNEL_SPLIT`)

**Goal**

- define a consistent internal channel model for mono or OSC data

**Processing**

- determine whether subsequent metrics and reconstruction stages operate on mono data, CFA sub-planes, or RGB-compatible representations
- derive channel-related metadata for downstream stages

**Output**

- channel and mode description used by later phases

---

## 4) Normalization (`NORMALIZATION`)

**Goal**

- make signal and background levels comparable across frames

**Processing**

- estimate background and intensity statistics per frame or per channel
- scale data into a shared reference state
- persist normalization parameters

**Output**

- normalized frames or equivalent normalization parameters
- diagnostics about background and signal stability

---

## 5) Global quality metrics (`GLOBAL_METRICS`)

**Goal**

- derive a global quality profile for each frame

**Processing**

- compute global measures such as background level, noise, gradient energy, star metrics or global sharpness indicators
- derive a global frame weight
- in the `strict` profile: evaluate on unified geometry before local stages proceed

**Output**

- per-frame global metrics
- global weights and selection priors

---

## 6) Build the tile grid (`TILE_GRID`)

**Goal**

- provide auxiliary spatial geometry and the reconstruction grid for the Classic path

**Processing**

- generate an overlapping or smoothly composable tile raster
- parameterize tile size, overlap and usable support region

**Output**

- auxiliary tile geometry; in Classic Tile-Compile, also the local-metrics and reconstruction grid

---

## 7) Determine shared overlap (`COMMON_OVERLAP`)

**Goal**

- restrict downstream processing to pixel regions that actually contain reliable warped data

**Processing**

- derive global and tile-local validity masks
- compute usable area fractions after warp, translation and rotation
- mask empty or insufficiently overlapping border regions

**Output**

- global valid fractions
- tile-local validity measures
- robust support mask for reconstruction and stacking

---

## 8) AQMH quality maps (`AQMH_MAPS`, enum 19)

**Goal**

- produce a dense per-pixel quality model for every frame

**Processing**

- calculate multi-scale sharpness and SNR using a Laplacian pyramid
- detect artifact-dominated support and apply the common canvas mask
- cache one `Q_map` per frame for independent reconstruction
- use CUDA/OpenCL filters when available

**Output**

- cached AQMH quality maps and AQMH diagnostics

This is followed by `AQMH_GLOBAL_QUALITY` (enum 20), which computes the global
frame weights. With `method: classic_tile_compile`, `LOCAL_METRICS` (enum 8)
is executed instead and computes local tile metrics and weights `L_f,t`.

---

## 9) Reconstruction (`AQMH_RECONSTRUCTION`, enum 21)

**Goal**

- reconstruct the final linear signal from per-pixel AQMH quality maps (default) or classic local tile contributions

**Processing**

- AQMH: combine each pixel with global frame weights and per-frame quality maps, then apply weighted sigma clipping
- Classic: `TILE_RECONSTRUCTION` (enum 9) fuses weighted tile contributions and
  blends neighboring overlap regions
- use streaming CUDA for AQMH reconstruction when Cherry-Pick is disabled
- use CUDA/OpenCL for classic sigma clipping and overlap-add; fall back to CPU when unavailable

**Output**

- reconstructed image with quality-aware information usage
- AQMH or per-tile reconstruction diagnostics

---

## 10) State clustering (`STATE_CLUSTERING`, Classic Tile-Compile only)

**Goal**

- group frames with similar quality or acquisition states

**Processing**

- cluster in global and/or local feature space
- separate heterogeneous sub-populations within a single acquisition series

**Output**

- cluster assignment per frame
- diagnostics for cluster size and stability

---

## 11) Synthetic frames (`SYNTHETIC_FRAMES`, Classic Tile-Compile only)

**Goal**

- derive robust intermediate representations from clusters

**Processing**

- aggregate frame groups into synthetic representatives
- reduce variance inside a state cluster

**Output**

- synthetic frames as alternative inputs for later aggregation stages

---

## 12) Final stacking (`STACKING`)

**Goal**

- produce the final linear stacked image

**Processing**

- AQMH: pass through the final reconstruction produced in
  `AQMH_RECONSTRUCTION` (phase 21)
- Classic: robustly aggregate reconstructed or synthetic intermediate data
- Classic: suppress outliers such as hot pixels, satellite trails or sporadic defects
- Classic: combine data using the previously derived quality models
- Classic: use CUDA/OpenCL for weighted or sigma-clipped reduction and process OSC RGB channels concurrently

**Output**

- linear final image, typically `outputs/stacked.fits`

---

## 13) Debayer (`DEBAYER`, OSC only)

**Goal**

- convert CFA/OSC data into an RGB representation

**Processing**

- demosaic the stacked or otherwise prepared linear data product
- for mono data: pass through without color interpolation

**Output**

- RGB FITS, typically `outputs/stacked_rgb.fits`

---

## 14) Astrometry (`ASTROMETRY`)

**Goal**

- generate a WCS solution for the final image

**Processing**

- try ASTAP plate solving first; if it does not produce a WCS, match detected stars against the locally installed PCC Gaia DR3 catalog without starting Siril or using the network
- derive or write sky-coordinate context and image scale

**Output**

- WCS-aware image or associated WCS file
- diagnostic artifacts and phase fields describing the selected solver, Gaia star counts, and any fallback error

---

## 15) Background Gradient Extraction (`BGE`, optional)

**Goal**

- reduce large-scale background gradients before color calibration

**Processing**

- estimate a background model per RGB channel
- subtract that model from the RGB image
- persist diagnostics such as `artifacts/bge.json`

**Output**

- gradient-corrected RGB image, typically `outputs/stacked_rgb_bge.fits`
- BGE diagnostics

---

## 16) Photometric Color Calibration (`PCC`)

**Goal**

- calibrate the RGB image towards a more astrophysically plausible color balance

**Processing**

- match stars against catalogs using the available WCS context
- determine and apply color scaling or calibration factors

**Output**

- photometrically calibrated RGB image, typically `outputs/stacked_rgb_pcc.fits`
- PCC diagnostics and possibly auxiliary catalog products

---

## 17) HyperMetric Stretch (`HYPERMETRIC_STRETCH`, optional)

**Goal**

- apply a final, reproducible VeraLux HMS stretch to the PCC-calibrated RGB image

**Processing**

- reads the PCC RGB result, typically `outputs/stacked_rgb_pcc.fits`
- resolves the configured sensor profile, adaptive anchor and Auto-LogD
- applies the HyperMetric stretch curve and color preservation

**Output**

- stretched RGB image, typically `outputs/stacked_rgb_hms.fits`
- with `write_channels: true`, also `hms_R.fit`, `hms_G.fit`, `hms_B.fit`

---

## 18) Finish (`DONE`)

**Goal**

- move the run into a consistent final state

**Processing**

- persist the terminal status such as `ok` or `validation_failed`
- finalize artifacts, logs and the configuration snapshot

**Output**

- reproducible and auditable run state

---

## Typical run structure

A run typically creates `runs/<run_id>/` with the following logical structure:

- `outputs/`
  - final and derived FITS products
  - e.g. `stacked.fits`, `stacked_rgb.fits`, `stacked_rgb_bge.fits`, `stacked_rgb_pcc.fits`, `stacked_rgb_hms.fits`
- `artifacts/`
  - per-phase JSON diagnostics
  - reports and visual assets
- `logs/`
  - `run_events.jsonl` as the run event timeline
- `config.yaml`
  - snapshot of the effective configuration used for this run

The exact filenames may vary by configuration. The stable part is the semantic separation between outputs, artifacts, logs and configuration snapshot.

---

## Resume of post-run phases

The complete resume matrix with the implemented entry points and minimum
dependencies is in
[resume_dependencies_en.md](resume_dependencies_en.md). Its distinction
between direct resume and in-place full rerun is authoritative.

If a run already exists, supported post-processing phases can be re-executed
from the persisted run state:

```text
./tile_compile_runner resume --run-dir runs/<run_id> --from-phase ASTROMETRY
./tile_compile_runner resume --run-dir runs/<run_id> --from-phase HYPERMETRIC_STRETCH
```

The resume path reuses in particular:

- the configuration snapshot `config.yaml`
- outputs and artifacts from earlier phases
- the run directory as the authoritative working context

For direct post-processing resumes this is a controlled continuation based on
persisted run data. Early phases marked as in-place full reruns instead start
the complete pipeline in the same run directory.

---

## Evaluation with the integrated report generator

For technical evaluation and quality assurance, an HTML report can be generated from a run directory:

```text
./tile_compile_cli generate-report runs/<run_id>
```

The report is typically written to `runs/<run_id>/artifacts/report.html` and correlates execution events, diagnostic artifacts and configuration state.

Typical report sections include:

- **Normalization**
  - background trends and intensity-scaling stability
- **Global metrics**
  - background, noise, gradient energy, global weights, distributions
- **Star metrics**
  - FWHM, wFWHM, roundness, star count, correlation plots
- **Registration**
  - drift, rotation, matching or correlation quality
- **Tile analysis**
  - Classic-only tile grid, local metrics and spatial heatmaps
- **AQMH analysis**
  - quality-map statistics, artifact support and reconstruction diagnostics
- **Reconstruction**
  - AQMH per-pixel reconstruction or Classic tile-local usage metrics
- **Clustering and synthetic frames**
  - Classic-only cluster sizes, reduction behavior and synthetic representative usage
- **BGE / PCC**
  - background model, residuals, calibration diagnostics
- **Validation**
  - derived quality indicators and threshold checks
- **Timeline**
  - chronological phase sequence from `run_events.jsonl`

The report also embeds the effective `config.yaml`, which makes each finding directly traceable to the exact parameter state.

---

## Notes on interpretation

1. **Linear images look dark**
   - This is expected. A linear stacked image is not stretched for presentation by default.
2. **`validation_failed` does not automatically mean “useless”**
   - It primarily means that defined validation or guardrail criteria were violated.
3. **Per-pixel AQMH quality is the default principle**
   - The main advantage comes from dense local quality weighting instead of a purely global average. Classic Tile-Compile remains available when tile-based diagnostics or clustering are specifically desired.

---

## Short conclusion

> The pipeline transforms a heterogeneous FITS frame series into a shared geometric and photometric reference space, builds dense AQMH quality maps, reconstructs the signal per pixel, and produces a reproducible final image with diagnostics, WCS metadata and optional color calibration. The former tile-based workflow is retained as Classic Tile-Compile.
