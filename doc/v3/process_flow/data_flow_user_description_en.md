# Process Flow – How the System Works

## In short

The system processes many astronomical single frames (FITS) into **one clean, sharp, and color-consistent final image**.

The pipeline follows a clear sequence of processing steps: validate input, align frames, evaluate quality, reconstruct the signal tile-by-tile, and finally apply astrometric and color calibration.

Key terms (v3.3):

- **Methodology profile:** `assumptions.pipeline_profile` controls whether the pipeline is aligned strictly with the normative methodology ("strict") or with a more pragmatic profile ("practical").
- **Resume:** Post-run phases can be continued from an existing run directory (currently `ASTROMETRY`, `BGE`, `PCC`).

---

## Sketch: overall flow

```text
Input (many FITS frames)
        |
        v
[1] Validate & prepare
        |
        v
[2] Accurate global alignment
        |
        v
[3] Prewarp onto a common canvas
        |
        v
[4] Channel model + brightness comparability
        |
        v
[5] Split into tiles
        |
        v
[6] Compute common overlap (shared valid data only)
        |
        v
[7] Measure quality per tile + combine the best info per tile
        |
        v
[8] Optional: cluster similar "states" -> synthetic intermediate frames
        |
        v
[9] Final stacking (robust aggregation)
        |
        v
[10] Debayer (OSC), astrometry, optional BGE, color calibration (PCC)
        |
        v
Output: final FITS image (+ RGB/PCC variants, WCS, artifacts, logs)
```

---

## Why tiles?

In astronomical series the quality is often **not uniform across the full frame**:

- seeing changes over time,
- tracking errors can be local,
- some regions can be blurrier or noisier than others.

Therefore the system does not only evaluate the whole frame globally, but also many small regions (tiles).
This enables a local decision: **Which frames are best for this region?**

---

## Step-by-step (user-friendly)

## 0) Validate input (SCAN_INPUT)

What happens:

- All input files are discovered.
- Headers and acquisition mode are checked (mono vs OSC/CFA).
- Obvious problematic frames are filtered.
- Available disk space is checked.

Result:

- A cleaned list of usable frames.

---

## 1) Global registration (REGISTRATION)

What happens:

- A reference frame is selected.
- All other frames are aligned to it.
- If a method does not work reliably, the system switches through fallback strategies.

Result:

- Frames are as geometrically consistent as possible.

Sketch:

```text
before:    *   .*    ..*    *..
after:     *    *      *      *
```

---

## 2) Prewarp onto a common canvas (PREWARP)

What happens:

- After registration, all frames are warped onto a shared target canvas.
- For OSC data, this is done CFA-safe (sub-plane warp) so the Bayer pattern remains consistent.
- If field rotation requires a larger canvas, offsets (`tile_offset_x/y`) are tracked.

Result:

- All subsequent phases operate in the same geometry and coordinate system.

---

## 3) Channel model (CHANNEL_SPLIT)

What happens:

- For OSC/mono, the channel handling strategy is established.
- In practice this is mostly metadata / bookkeeping.

Result:

- A clear channel plan for the next steps.

---

## 4) Brightness normalization (NORMALIZATION)

What happens:

- Frames are normalized to a common brightness level.
- Background and signal levels are made comparable.

Result:

- Differences due to varying acquisition conditions become less disturbing.

---

## 5) Global quality metrics (GLOBAL_METRICS)

What happens:

- Each frame receives global quality values (e.g. sharpness / signal).
- A global weight per frame is derived.

Result:

- Good frames contribute more later, weak frames less.

---

## 6) Build tile grid (TILE_GRID)

What happens:

- The image is split into many slightly overlapping tiles.
- Tile size is chosen to match the data quality.

Result:

- A structure to decide locally (not only globally).

Sketch:

```text
+----+----+----+
| T1 | T2 | T3 |
+----+----+----+
| T4 | T5 | T6 |
+----+----+----+
| T7 | T8 | T9 |
+----+----+----+
```

---

## 7) Determine common overlap (COMMON_OVERLAP)

What happens:

- The system computes which canvas pixels carry valid data across all relevant frames.
- Global and tile-local valid fractions are derived.
- Border/empty areas caused by rotation/translation are masked.

Result:

- Reconstruction and stacking use only robust shared regions.

---

## 8) Local metrics per tile (LOCAL_METRICS)

What happens:

- For each frame and each tile, local quality is evaluated.
- This tells the system which parts of a frame are good.

Result:

- Local weights per tile and frame.

---

## 9) Tile reconstruction (TILE_RECONSTRUCTION)

What happens:

- For each tile, the best information from many frames is combined.
- Transitions are blended smoothly to avoid hard seams.

Result:

- A reconstructed image with locally optimized quality.

Sketch:

```text
Frame A is good on the left tile
Frame B is good on the right tile
=> reconstruction uses more of A on the left, more of B on the right
```

---

## 10) State clustering (STATE_CLUSTERING, optional)

What happens:

- Frames with similar “state” (e.g. similar quality / conditions) are grouped.

Result:

- Better separation of different acquisition conditions.

---

## 11) Synthetic frames (SYNTHETIC_FRAMES, optional)

What happens:

- Stable intermediate images are built from clusters.
- These are often smoother and more robust.

Result:

- A better basis for the final stacking.

---

## 12) Final stacking (STACKING)

What happens:

- Everything is merged into the final result.
- Outliers are handled robustly (e.g. satellite trails, hot pixel spikes).

Result:

- A linear, clean stacked image.

---

## 13) Debayer (DEBAYER, OSC only)

What happens:

- For OSC/CFA, the mosaic is converted into an RGB image.
- For mono, this is usually pass-through.

Result:

- An RGB image (or mono pass-through).

---

## 14) Astrometry (ASTROMETRY)

What happens:

- The image is solved on the sky (WCS / plate solving).

Result:

- Pixels get sky-coordinate context.

---

## 15) Background Gradient Extraction (BGE, optional)

What happens:

- Optionally, before PCC, a large-scale background model per RGB channel is estimated.
- The model is subtracted from the RGB channels.
- Diagnostic data is saved (e.g. `artifacts/bge.json`).

Result:

- Less gradient influence (light pollution / moonlight) on subsequent color calibration.

---

## 16) Photometric Color Calibration (PCC)

What happens:

- Colors are calibrated using a star catalog match.

Result:

- More natural and scientifically plausible colors.

---

## 17) Finish (DONE)

What happens:

- The pipeline ends with a status (`ok` or `validation_failed`).
- Artifacts, logs and outputs are stored in the run directory.

Result:

- A reproducible end state.

---

## Typical output files

A run typically creates a directory `runs/<run_id>/` containing:

- `outputs/`
  - `stacked.fits` (final linear stacked image)
  - `stacked_rgb.fits` (RGB after debayer)
  - `stacked_rgb_bge.fits` (optional: after BGE)
  - `stacked_rgb_pcc.fits` (after PCC)
- `artifacts/`
  - JSON diagnostics per phase (quality, weights, validation, BGE, PCC)
  - report assets (e.g. `report.html`, `report.css`, PNGs)
- `logs/`
  - `run_events.jsonl` (event timeline)
- `config.yaml` (snapshot of the configuration used for this run)

Note: exact filenames depend on your configuration. The key part is the structure (`outputs/`, `artifacts/`, `logs/`, `config.yaml`).

---

## Resume (post-run)

If a run already exists and you only want to redo post-processing phases (e.g. astrometry / BGE / PCC), you can resume:

```text
./tile_compile_runner resume --run-dir runs/<run_id> --from-phase ASTROMETRY
```

This operates from the run directory (including the `config.yaml` snapshot).

---

## Evaluation with the integrated report generator

For structured analysis you can generate an HTML report from a run directory:

```text
./tile_compile_cli generate-report runs/<run_id>
```

The report is written to `runs/<run_id>/artifacts/report.html` and is accompanied by `report.css` and PNG charts.

The report can include:

- **Normalization:** background trend (mono or R/G/B) and stability over time.
- **Global metrics:** background, noise, gradient energy, global frame weights, distributions.
- **Star metrics (Siril-like):** FWHM, wFWHM, roundness, star count, FWHM vs roundness plot.
- **Registration:** translation/drift, rotation trend, CC distribution.
- **Tile analysis:** tile grid, local quality maps, spatial heatmaps.
- **Reconstruction:** reconstruction KPIs (e.g. contrast/background/SNR per tile).
- **Clustering & synthetic frames:** cluster sizes, synthetic usage, reduction behavior.
- **BGE:** per-channel grid cells, residual distributions and background offsets.
- **Validation:** FWHM improvement, tile-pattern indicators, further checks.
- **Pipeline timeline:** chronological phase view derived from `run_events.jsonl`.
- **Frame usage funnel:** from “discovered” to “stacked/synthetic”.

The report also embeds the used `config.yaml` to make evaluation traceable.

---

## Notes on interpretation

1. **Linear images look dark:** a linear astro image will often look flat/dark until stretched.
2. **Validation may fail even if the result is usable:** thresholds are violated, not necessarily the entire result.
3. **Tile-based reconstruction optimizes locally:** this is the main advantage over purely global stacking.

---

## Short conclusion

> Many frames in -> align -> evaluate locally -> combine best signal per tile -> robust stack -> astrometry + color calibration -> final astrophotography output.
