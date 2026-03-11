# Process Flow – Technical Data Flow of the System

## Pipeline objective

The system turns a set of calibrated astronomical single-frame inputs into a reproducible final product inside a shared geometric and photometric reference space.

From a technical perspective, the pipeline is organized into three major blocks:

- **Preparation and normalization**
  - validate inputs
  - unify geometry
  - normalize intensity levels
- **Quality modeling and reconstruction**
  - compute global and local metrics
  - perform tile-based selection and reconstruction
  - optionally cluster acquisition states and derive synthetic frames
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
  - One well-defined processing stage such as `REGISTRATION`, `LOCAL_METRICS`, or `PCC`.
- **Artifact**
  - Persisted diagnostic or intermediate data, typically written under `artifacts/`.
- **Event timeline**
  - Chronological execution events written to `logs/run_events.jsonl`.
- **Methodology profile**
  - `assumptions.pipeline_profile` controls whether the pipeline follows a stricter normative interpretation (`strict`) or a more pragmatic robust profile (`practical`).
- **Resume**
  - Existing run directories can be reused for post-run phases, currently especially from `ASTROMETRY`, `BGE`, or `PCC` onward.

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
   -> TILE_GRID
   -> COMMON_OVERLAP
   -> LOCAL_METRICS
   -> TILE_RECONSTRUCTION
   -> [optional] STATE_CLUSTERING
   -> [optional] SYNTHETIC_FRAMES
   -> STACKING
   -> [optional / data-dependent] DEBAYER
   -> ASTROMETRY
   -> [optional] BGE
   -> [optional] PCC
   -> DONE
```

---

## Why the pipeline is tile-based

Frame-level global scoring alone is usually insufficient for astrophotography series. Local quality varies within a single frame due to factors such as:

- location-dependent seeing variations
- local guiding or deformation artifacts
- border artifacts after warp or rotation
- uneven background or noise distributions

For that reason, the system models the data not only at frame level but also at tile level. This makes it possible to decide, per spatial region, which frames or frame contributions provide the highest usable quality there.

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

**Output**

- prewarped frames with unified geometry
- a consistent coordinate domain for all tile-based downstream phases

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

- partition the image field into locally evaluable regions

**Processing**

- generate an overlapping or smoothly composable tile raster
- parameterize tile size, overlap and usable support region

**Output**

- tile geometry used by local metrics and reconstruction

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

## 8) Local per-tile metrics (`LOCAL_METRICS`)

**Goal**

- model the best local data quality per tile and per frame

**Processing**

- compute local sharpness, contrast, noise or star metrics for each tile
- combine them with global weights and validity masks
- in the `strict` profile: operate on prewarped raw tiles for geometrically consistent comparison

**Output**

- local weights and local quality profiles for each tile/frame combination

---

## 9) Tile reconstruction (`TILE_RECONSTRUCTION`)

**Goal**

- reconstruct a spatially consistent intermediate image from the best local contributions

**Processing**

- select or fuse the strongest tile contributions
- blend transitions between neighboring tiles to avoid seam artifacts
- reconstruct using local quality maps and support weights

**Output**

- reconstructed image with locally optimized information usage
- per-tile reconstruction metrics

---

## 10) State clustering (`STATE_CLUSTERING`, optional)

**Goal**

- group frames with similar quality or acquisition states

**Processing**

- cluster in global and/or local feature space
- separate heterogeneous sub-populations within a single acquisition series

**Output**

- cluster assignment per frame
- diagnostics for cluster size and stability

---

## 11) Synthetic frames (`SYNTHETIC_FRAMES`, optional)

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

- robustly aggregate reconstructed or synthetic intermediate data
- suppress outliers such as hot pixels, satellite trails or sporadic defects
- combine data using the previously derived quality models

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

- perform plate solving against astrometry tools and catalogs
- derive or write sky-coordinate context and image scale

**Output**

- WCS-aware image or associated WCS file
- diagnostic artifacts describing the solving process

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

## 17) Finish (`DONE`)

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
  - e.g. `stacked.fits`, `stacked_rgb.fits`, `stacked_rgb_bge.fits`, `stacked_rgb_pcc.fits`
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

If a run already exists, post-processing phases can be re-executed from the persisted run state:

```text
./tile_compile_runner resume --run-dir runs/<run_id> --from-phase ASTROMETRY
```

The resume path reuses in particular:

- the configuration snapshot `config.yaml`
- outputs and artifacts from earlier phases
- the run directory as the authoritative working context

Resume is therefore not a partially new run, but a controlled continuation based on persisted run data.

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
  - tile grid, local quality maps, spatial heatmaps
- **Reconstruction**
  - tile-local reconstruction metrics and usage maps
- **Clustering and synthetic frames**
  - cluster sizes, reduction behavior, synthetic representative usage
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
3. **Tile-based optimization is the key principle**
   - The main advantage comes from using local quality instead of applying a purely global average quality model to all spatial regions.

---

## Short conclusion

> The pipeline transforms a heterogeneous FITS frame series into a shared geometric and photometric reference space, models data quality globally and locally, reconstructs the signal tile-by-tile, and produces a reproducible final image together with diagnostics, WCS metadata and optional color calibration.
