# Process Flow Documentation — AQMH Pipeline (tile_compile_cpp)

## Overview

This document primarily describes the **actual AQMH execution flow** of the C++ implementation (`tile_compile_cpp/apps/runner_pipeline.cpp`). Classic Tile-Compile reconstruction is separated as an alternative at the end.

The current default pipeline processes **FITS frames** (mono or OSC/CFA) and produces a pixel-wise AQMH reconstruction. It does not use classic local tile metrics, clustering, or synthetic frames.

**Implementation:** C++ with Eigen, OpenCV, cfitsio, nlohmann/json, YAML-cpp.

**GUI3 integration:** The productive GUI path uses the web frontend plus the Crow/C++ backend. Crow orchestrates the C++ pipeline by invoking `tile_compile_cli` and `tile_compile_runner`; it does not reimplement the processing logic.

> **Default method: AQMH** — The default reconstruction method is `aqmh` (Adaptive Quality Map Harvesting). AQMH uses phases 19–22 (`AQMH_MAPS`, `AQMH_GLOBAL_QUALITY`, `AQMH_RECONSTRUCTION`, `AQMH_DIAGNOSTICS`) instead of Classic phases 8–9. Clustering and synthetic frames are skipped. See [Pipeline Overview](phase_0_overview.md) for the AQMH vs. Classic comparison.

The authoritative resume entry points and their artifact/cache dependencies are
documented in [Resume Dependencies](resume_dependencies_en.md). In particular,
early resume phases trigger an in-place full rerun; they do not continue from
the named phase.

## Current AQMH phases (C++ implementation)

Source of the phase order: `tile_compile::Phase` in `include/tile_compile/core/types.hpp`.

| ID | Enum | Short description |
|----|------|-------------------|
| 0 | `SCAN_INPUT` | Input scan, header/mode detection, linearity validation, disk-space precheck (`scandir*4`) |
| 1 | `REGISTRATION` | Global registration (cascaded), warp quality / CC |
| 2 | `PREWARP` | Full-frame prewarp onto a common canvas (CFA-safe for OSC) |
| 3 | `CHANNEL_SPLIT` | Metadata phase (OSC/mono channel model; actual channel work happens later) |
| 4 | `NORMALIZATION` | Global linear normalization (additive background + photometric scale) |
| 5 | `GLOBAL_METRICS` | Global frame metrics and weights `G_f` |
| 6 | `TILE_GRID` | Adaptive tile geometry (seeing/FWHM-based) |
| 7 | `COMMON_OVERLAP` | Common valid-data area (global/tile-local) |
| 19 | `AQMH_MAPS` | Pyramid pixel-wise quality maps and per-frame diagnostics in `cache/aqmh/` and `artifacts/aqmh_metrics.json` |
| 20 | `AQMH_GLOBAL_QUALITY` | Global frame weights `G_f` from sharpness, SNR, and background penalty |
| 21 | `AQMH_RECONSTRUCTION` | Pixel-wise weighted reconstruction with support masks, robust clipping, and immutable raw CFA output |
| 22 | `AQMH_DIAGNOSTICS` | Block diagnostics, heatmaps, and reconstruction metrics |
| 12 | `STACKING` | Final linear stacking (including robust pixel outlier handling) |
| 13 | `DEBAYER` | OSC debayering and RGB output (mono: pass-through) |
| 14 | `ASTROMETRY` | Plate solving / WCS |
| 15 | `BGE` | Optional Background Gradient Extraction on RGB before PCC |
| 16 | `PCC` | Photometric Color Calibration |
| 17 | `HYPERMETRIC_STRETCH` | VeraLux HyperMetric Stretch after PCC |
| 18 | `DONE` | Final status (`ok` or `validation_failed`) |

Note: **Validation** is a quality block between `STACKING` and `DEBAYER`, but it is not its own enum phase.

Note: **BGE** is an optional **dedicated phase** between `ASTROMETRY` and `PCC`.

`STATE_CLUSTERING` and `SYNTHETIC_FRAMES` are skipped or unused as executing
AQMH stages.
`AQMH_BGE_INPUTS` is currently defined as an enum value only and is not emitted
as a separate executing phase by the normal runner.

## Document structure

Details of the AQMH-specific extension phases are documented in
[AQMH extensions](phase_8_aqmh_extensions.md).

The **authoritative AQMH execution order** is shared preprocessing,
`AQMH_MAPS`, `AQMH_GLOBAL_QUALITY`, `AQMH_RECONSTRUCTION`,
`AQMH_DIAGNOSTICS`, and shared finalization. Enum values 19–22 are AQMH
phase values, not a continuation of the old Classic phase 8/9 numbering.

High-level mapping:

- Input + mode + linearity + disk precheck -> `SCAN_INPUT`
- Registration / prewarp -> `REGISTRATION`, `PREWARP`
- Normalization + canvas masks -> `NORMALIZATION` through `COMMON_OVERLAP`
- AQMH analysis -> `AQMH_MAPS` -> `AQMH_GLOBAL_QUALITY`
- AQMH reconstruction -> `AQMH_RECONSTRUCTION` -> `AQMH_DIAGNOSTICS`
- Unused Classic stages -> `STATE_CLUSTERING`, `SYNTHETIC_FRAMES` (`skipped`)
- Finalization path -> `STACKING`, `DEBAYER`, `ASTROMETRY`, `BGE` (optional but its own phase), `PCC`, `HYPERMETRIC_STRETCH`, `DONE`

---

## Pipeline flow diagram (C++ implementation, v3.3)

```
┌─────────────────────────────────────────────────────────────┐
│               INPUT: MONO / OSC RAW FITS FRAMES             │
└────────────────────────────┬────────────────────────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 0: SCAN_INPUT         │
              │  • FITS dimensions + header  │
              │  • Color mode (MONO/OSC)     │
              │  • Bayer pattern detection   │
              │  • Linearity validation      │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 1: REGISTRATION       │
              │  • Cascaded fallbacks        │
              │  • CC / warp metrics         │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 2: PREWARP            │
              │  • Full-frame canvas warp    │
              │  • CFA-safe (OSC)            │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 3: CHANNEL_SPLIT      │
              │  (metadata-only)             │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 4: NORMALIZATION      │
              │  • Sigma-clip BG mask        │
              │  • Additive background B_f   │
              │  • Photometric scale P_f     │
              │  • I = (I_raw - B_f) / P_f   │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 5: GLOBAL_METRICS     │
              │  • B_f, σ_f, E_f per frame   │
              │  • MAD-normalize -> z-scores │
              │  • G_f = exp(α·B̃+β·σ̃+γ·Ẽ)    │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 6: TILE_GRID          │
              │  • FWHM probe (central ROI)  │
              │  • T = clip(s·F, min, max)   │
              │  • Overlap + stride calc     │
              │  • Uniform tile grid         │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 7: COMMON_OVERLAP     │
              │  • Pixelwise valid overlap   │
              │  • common_overlap.json       │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 19: AQMH_MAPS          │
              │  • Pyramid quality maps      │
              │  • Per-frame pixel diagnostics│
              │  • cache/aqmh/                │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 20: AQMH_GLOBAL_QUALITY│
              │  • Sharpness/SNR summaries   │
              │  • Background penalty        │
              │  • Global frame weight G_f   │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 21: AQMH_RECONSTRUCTION│
              │  • Pixel-wise weighting      │
              │  • Support/valid masks       │
              │  • Robust Welford/sigma-clip │
              │  • Persist raw CFA artifact  │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 22: AQMH_DIAGNOSTICS   │
              │  • Block diagnostics         │
              │  • Heatmaps / map statistics │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 12: STACKING          │
              │  • Sigma-clip rejection      │
              │  • Or mean of synth frames   │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  VALIDATION                  │
              │  • FWHM improvement check    │
              │  • Tile weight variance      │
              │  • Tile pattern detection    │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 13: DEBAYER           │
              │  • OSC: NN demosaic -> RGB   │
              │  • MONO: pass-through        │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 14: ASTROMETRY       │
              │  • ASTAP solve / WCS        │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 15: BGE              │
              │  • Optional before PCC      │
              │  • Gradient subtraction     │
              │  • artifacts/bge.json       │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 16: PCC              │
              │  • Photometric color cal.   │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 17: HMS              │
              │  • VeraLux HyperMetric      │
              │  • Stretch after PCC        │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 18: DONE             │
              │  • Final status emit        │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  OUTPUTS:                   │
              │  • stacked.fits             │
              │  • reconstructed_L.fit      │
              │  • stacked_rgb.fits (OSC)   │
              │  • stacked_rgb_solve.fits   │
              │  • stacked_rgb_bge.fits     │
              │  • stacked_rgb_pcc.fits     │
              │  • stacked_rgb_hms.fits     │
              │  • R/G/B .fit (OSC)         │
              │  • 12 artifact JSON files   │
              │  • run_events.jsonl         │
              └─────────────────────────────┘
```

## Core principles (C++ implementation)

1. **Core linearity:** phases up to and including PCC stay linear; HMS is an explicit final stretch phase after PCC.
2. **No hard frame selection:** frames are kept; failed registration falls back to identity warp with CC=0.
3. **Mono + OSC:** both modes in one pipeline, CFA-aware for OSC.
4. **Strictly sequential:** no feedback loops; deterministic execution order.
5. **AQMH instead of Classic tiles:** reconstruction is pixel-wise on quality maps, not based on `L_f,t` tile weights.
6. **Global × pixel:** the global frame weight combines with pixel-wise AQMH quality.
7. **Pre-warping:** frames are fully warped onto the common canvas before quality-map computation.
8. **Robust statistics:** median, MAD, sigma-clipping throughout.

## Modes

### AQMH in every mode

- `AQMH_MAPS`, `AQMH_GLOBAL_QUALITY`, `AQMH_RECONSTRUCTION`, and `AQMH_DIAGNOSTICS` form the AQMH analysis and reconstruction path.
- `STATE_CLUSTERING` and `SYNTHETIC_FRAMES` are not used as Classic processing in AQMH mode.
- `STACKING` consumes the AQMH reconstruction, followed by `DEBAYER`, `ASTROMETRY`, `BGE`, `PCC`, and HMS.
- Frame count still controls reduced/emergency gates and resource use.

### Classic alternative

Only `method: classic_tile_compile` executes `LOCAL_METRICS`,
`TILE_RECONSTRUCTION`, `STATE_CLUSTERING`, and `SYNTHETIC_FRAMES`.

## Quality metrics

### AQMH global frame quality (`AQMH_GLOBAL_QUALITY`)

- **B_f:** background level of the normalized frame (lower = better)
- **σ_f:** noise (lower = better)
- **E_f:** gradient energy / Sobel-based (higher = better)
- **Q_f:** weighted score = α·(-B̃) + β·(-σ̃) + γ·Ẽ (MAD-normalized)
- **G_f:** global frame weight from the AQMH summaries, bounded by the configured range

### AQMH quality maps (`AQMH_MAPS`)

- Pyramid local variance/sharpness information per frame and pixel
- Per-frame SNR and background penalty
- Artifact and support masks
- Maps persisted under `cache/aqmh/`

### AQMH reconstruction (`AQMH_RECONSTRUCTION`)
- Pixel-wise combination of valid frames using the AQMH map and `G_f`
- Robust Welford/sigma-clipping reduction
- Support-aware output without Classic tile renormalization
- Used in phase 9 for tile reconstruction.

## Mathematical notation

```
Indices:
  f - frame index (0..N-1)
  p - pixel index on the common canvas

Dimensions:
  N  - number of usable frames after input validation
  W,H - image width/height in pixels

Normalization:
  I_f^raw  - original linear frame
  B_f      - additive background level
  P_f      - photometric scale
  J_f      - background-subtracted frame = I_f^raw - B_f
  I_f      - normalized frame = J_f / P_f

AQMH global inputs:
  g_sharp_f       - summarized sharpness/PSF quality
  g_snr_f         - summarized signal-to-noise quality
  g_background_f  - background penalty

Global frame weight:
  G_f = compute_aqmh_global_quality(g_sharp_f, g_snr_f, g_background_f)

Pixel-wise reconstruction:
  q_f,p = QualityMap(f, p)
  w_f,p = G_f × q_f,p × valid_f,p
  recon_p = robust_weighted_reduce({I_f,p}, {w_f,p})

The robust reduction applies minimum support, effective sample-size,
sigma-clipping, and the configured AQMH gates. Classic quantities `L_f,t`,
tile overlap-add, clustering, and synthetic frames are not part of AQMH.
```

## Artifact files

An AQMH run produces the following central artifacts in
`<run_dir>/artifacts/`; optional diagnostic artifacts may be added:

| File | Phase | Contents |
|------|-------|----------|
| `normalization.json` | 4 | mode, bayer, B_mono / B_r / B_g / B_b per frame |
| `global_metrics.json` | 5 | global normalization/input metrics per frame |
| `tile_grid.json` | 6 | image dimensions, tile list (x,y,w,h), FWHM, overlap |
| `global_registration.json` | 1 | warp matrices (a00,a01,tx,a10,a11,ty) + CC per frame |
| `common_overlap.json` | 7 | global/tile-wise common valid area |
| `aqmh_metrics.json` | 8/9 | quality-map metadata, frame diagnostics, and global AQMH weights |
| `aqmh_reconstruction.json` | 10 | AQMH reconstruction, support, and clipping diagnostics |
| `aqmh_regions.json` / `cache/aqmh_block_diagnostics.jsonl` | 11 | AQMH region, block, and heatmap diagnostics |
| `bge.json` | 15 | per-channel BGE diagnostics (samples, grid cells, residuals) |
| `validation.json` | 12 | method-specific quality and support validation |

`local_metrics.json`, `state_clustering.json`, and `synthetic_frames.json` are
Classic-only artifacts and are not AQMH quality inputs.

### Report generation and readable data

For consolidated analysis, generate the report via the integrated CLI/backend path.

Invocation:

```text
./tile_compile_cli generate-report runs/<run_id>
```

Generated outputs:

- `artifacts/report.html`
- `artifacts/report.css`
- `artifacts/*.png` (charts/heatmaps)

Input data used:

- Artifact JSONs: `normalization.json`, `global_metrics.json`, `tile_grid.json`,
  `global_registration.json`, `common_overlap.json`, `aqmh_metrics.json`,
  `aqmh_reconstruction.json`, `aqmh_regions.json`, AQMH block diagnostics,
  `bge.json`, `validation.json`
- Run events: `logs/run_events.jsonl`
- Run configuration: `config.yaml` (embedded into the report)

Typically readable content:

- Normalization/background trends (mono/RGB)
- AQMH quality-map statistics and global frame weights
- Star metrics (incl. FWHM, wFWHM, roundness, star count)
- Registration evaluation (shift/rotation/correlation)
- AQMH reconstruction and support heatmaps
- Classic clustering and synthetic-frame summaries only for `classic_tile_compile`
- BGE diagnostics (per-channel background models, grid cells, residual histograms)
- Validation results (incl. tile-pattern indicators)
- Pipeline timeline and frame usage funnel

## Directory structure

```
runs/<run_id>/
├── config.yaml           # copy of the run configuration
├── logs/
│   └── run_events.jsonl  # all pipeline events (JSONL)
├── artifacts/
│   ├── normalization.json
│   ├── global_metrics.json
│   ├── tile_grid.json
│   ├── global_registration.json
│   ├── common_overlap.json
│   ├── aqmh_metrics.json
│   ├── aqmh_reconstruction.json
│   ├── aqmh_regions.json
│   ├── bge.json
│   ├── validation.json
│   ├── report.html       # generated via CLI/backend report path
│   ├── *.png             # chart images
├── cache/
│   ├── normalized_frames/
│   ├── prewarped_frames/
│   ├── aqmh/
│   ├── aqmh_masks/
│   └── aqmh_block_diagnostics.jsonl
└── outputs/
    ├── stacked.fits
    ├── reconstructed_L.fit
    ├── stacked_rgb.fits       # (OSC only)
    ├── reconstructed_R.fit    # (OSC only)
    ├── reconstructed_G.fit    # (OSC only)
    ├── reconstructed_B.fit    # (OSC only)
    └── synthetic_*.fit        # (Classic mode only)
```

## Performance optimizations (C++)

- **Eigen matrices:** vectorized pixel operations (SIMD)
- **OpenCV:** optimized image processing (Sobel, Laplacian, warpAffine)
- **Thread parallelism:** AQMH map/reconstruction workers; Classic additionally uses a `TILE_RECONSTRUCTION` worker pool
- **Pre-warping:** warp all frames once instead of per-tile
- **2× downsample:** registration at half resolution (speedup ~4×)
- **Memory-efficient:** frames are loaded from disk per phase
- **cv::setNumThreads(1):** avoids OpenCV thread contention in parallel tiles
- **CUDA worker streams:** one non-default stream per parallel PREWARP/AQMH/tile worker
- **Streaming AQMH CUDA reconstruction:** GPU accumulators remain resident while frames/maps are transferred one at a time; VRAM use is independent of frame count
- **Concurrent RGB stacking:** R/G/B reductions run concurrently with separate CUDA streams

## GPU execution by phase

| Phase | CUDA | OpenCL | Work performed on GPU |
|---|---:|---:|---|
| `PREWARP` | Yes | Yes | Full-frame affine/CFA warps |
| `AQMH_MAPS` | Yes | Yes | Pyramid filters and local variance |
| `AQMH_RECONSTRUCTION` | Yes | No | Weighted Welford statistics, masks, sigma clipping, accumulation |
| Classic `TILE_RECONSTRUCTION` | Yes | Yes | Sigma clipping and overlap-add |
| `SYNTHETIC_FRAMES` | Yes | Yes | Cluster tile reconstruction |
| `STACKING` / resume | Yes | Yes | Weighted/sigma-clipped reduction, concurrent RGB |

`REGISTRATION` remains CPU-only; GPU execution starts in `PREWARP`.

`runtime_limits.acceleration_backend: auto` selects CUDA, then OpenCL, then
CPU per supported phase. AQMH Cherry-Pick uses CPU. CUDA/OpenCL failures fall
back to CPU, and `artifacts/acceleration_context.json` plus live fields
`cpu_workers`, `gpu`, and `backend` record the effective path.

## References

### Normative specification
  - `/docs/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.9_en.md`

### C++ implementation
  - `/tile_compile_cpp/apps/runner_pipeline.cpp`
  - **Configuration:** `/tile_compile_cpp/include/tile_compile/config/configuration.hpp`
  - **Report generator:** `/web_backend_cpp/src/services/report_generator.cpp`

---

**Note:** this document describes the **actual C++ code behavior**. If there are contradictions with the normative specification, the code is the reference for behavior and the specification is the reference for intent.
