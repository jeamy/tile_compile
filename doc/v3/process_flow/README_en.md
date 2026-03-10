# Process Flow Documentation — C++ Pipeline (tile_compile_cpp)

## Overview

This document describes the **actual execution flow of the C++ implementation** (`tile_compile_cpp/apps/runner_pipeline.cpp`) of tile-based quality reconstruction for deep-sky objects according to methodology v3.

The pipeline processes **FITS frames** (mono or OSC/CFA) and produces a weighted, tile-based reconstruction result with optional clustering, synthetic frames and sigma-clipping stacking.

**Implementation:** C++ with Eigen, OpenCV, cfitsio, nlohmann/json, YAML-cpp.

**GUI2 integration:** The productive GUI path uses the web frontend plus the Crow/C++ backend. Crow orchestrates the C++ pipeline by invoking `tile_compile_cli` and `tile_compile_runner`; it does not reimplement the processing logic.

## Current pipeline phases (C++ implementation, v3.3)

Source of the phase order: `tile_compile::Phase` in `include/tile_compile/core/types.hpp`.

| ID | Enum | Short description |
|----|------|-------------------|
| 0 | `SCAN_INPUT` | Input scan, header/mode detection, linearity validation, disk-space precheck (`scandir*4`) |
| 1 | `REGISTRATION` | Global registration (cascaded), warp quality / CC |
| 2 | `PREWARP` | Full-frame prewarp onto a common canvas (CFA-safe for OSC) |
| 3 | `CHANNEL_SPLIT` | Metadata phase (OSC/mono channel model; actual channel work happens later) |
| 4 | `NORMALIZATION` | Global linear normalization (background-based) |
| 5 | `GLOBAL_METRICS` | Global frame metrics and weights `G_f` |
| 6 | `TILE_GRID` | Adaptive tile geometry (seeing/FWHM-based) |
| 7 | `COMMON_OVERLAP` | Common valid-data area (global/tile-local) |
| 8 | `LOCAL_METRICS` | Local tile metrics/classification and local weights `L_f,t` |
| 9 | `TILE_RECONSTRUCTION` | Weighted tile-based reconstruction (overlap-add) |
| 10 | `STATE_CLUSTERING` | State-vector clustering (optional, mode-dependent) |
| 11 | `SYNTHETIC_FRAMES` | Generation of synthetic frames (optional, mode-dependent) |
| 12 | `STACKING` | Final linear stacking (including robust pixel outlier handling) |
| 13 | `DEBAYER` | OSC debayering and RGB output (mono: pass-through) |
| 14 | `ASTROMETRY` | Plate solving / WCS |
| 15 | `BGE` | Optional Background Gradient Extraction on RGB before PCC |
| 16 | `PCC` | Photometric Color Calibration |
| 17 | `DONE` | Final status (`ok` or `validation_failed`) |

Note: **Validation** is a quality block between `STACKING` and `DEBAYER`, but it is not its own enum phase.

Note: **BGE** is an optional **dedicated phase** between `ASTROMETRY` and `PCC`.

## Document structure

The **authoritative phase order** is the v3.3 list above (0..17).

High-level mapping:

- Input + mode + linearity + disk precheck -> `SCAN_INPUT`
- Registration / prewarp -> `REGISTRATION`, `PREWARP`
- Normalization + global/local weights + reconstruction -> `NORMALIZATION` through `TILE_RECONSTRUCTION` (including `COMMON_OVERLAP`)
- Optional full-mode block -> `STATE_CLUSTERING`, `SYNTHETIC_FRAMES`
- Finalization path -> `STACKING`, `DEBAYER`, `ASTROMETRY`, `BGE` (optional but its own phase), `PCC`, `DONE`

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
              │  • OSC: per-channel BG       │
              │  • MONO: single BG           │
              │  • Scale = 1/Background      │
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
              │  PHASE 8: LOCAL_METRICS      │
              │  • Per (frame,tile) metrics  │
              │  • Tile type: STAR/STRUCTURE │
              │  • Robust z-score quality Q  │
              │  • L_f,t = exp(clip(Q))      │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 9: TILE_RECONSTRUCTION│
              │  • W_f,t = G_f × L_f,t       │
              │  • Weighted sum per tile     │
              │  • BG normalization per tile │
              │  • Hanning window + overlap  │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 10: STATE_CLUSTERING  │
              │  • 6D state vector per frame │
              │  • K-Means (K=N/10)          │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 11: SYNTHETIC_FRAMES  │
              │  • Weighted mean per cluster │
              │  • N -> K frame reduction    │
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
              │  PHASE 17: DONE             │
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
              │  • R/G/B .fit (OSC)         │
              │  • 12 artifact JSON files   │
              │  • run_events.jsonl         │
              └─────────────────────────────┘
```

## Core principles (C++ implementation)

1. **Linearity:** no non-linear operations (no stretch). Linearity is checked in phase 0.
2. **No hard frame selection:** frames are kept; failed registration falls back to identity warp with CC=0.
3. **Mono + OSC:** both modes in one pipeline, CFA-aware for OSC.
4. **Strictly sequential:** no feedback loops; deterministic execution order.
5. **Tile-based:** local quality evaluation; seeing-adaptive tile sizing.
6. **Global × local:** effective weight `W_f,t = G_f × L_f,t`.
7. **Pre-warping:** frames are fully warped before tile extraction (avoids CFA artifacts).
8. **Robust statistics:** median, MAD, sigma-clipping throughout.

## Modes

### Full/normal mode (N ≥ frames_reduced_threshold)

- All enum phases 0..17 are executed.
- State clustering is enabled.
- Synthetic frames are generated.
- Sigma-clipping rejection stacking.
- Best noise reduction.

### Reduced mode (N < frames_reduced_threshold)

- Phase 10 (`STATE_CLUSTERING`) is **skipped**.
- Phase 11 (`SYNTHETIC_FRAMES`) is **skipped**.
- Phase 9 (`TILE_RECONSTRUCTION`) produces the final image directly.
- Phase 12 (`STACKING`) passes through the reconstructed image unchanged.
- Validation is still executed.

## Quality metrics

### Global metrics (phase 5: GLOBAL_METRICS)

- **B_f:** background level of the normalized frame (lower = better)
- **σ_f:** noise (lower = better)
- **E_f:** gradient energy / Sobel-based (higher = better)
- **Q_f:** weighted score = α·(-B̃) + β·(-σ̃) + γ·Ẽ (MAD-normalized)
- **G_f:** global weight = exp(Q_f) with clamping [-3, +3]

### Local metrics (phase 8: LOCAL_METRICS)

- **FWHM:** full width half maximum (seeing quality)
- **Roundness:** star roundness (tracking quality)
- **Contrast:** local contrast (signal strength)
- **Sharpness:** sharpness metric
- **Star count:** number of detected stars per tile
- **Tile type:** STAR (enough stars) vs STRUCTURE (too few stars)
- **L_f,t:** local weight = exp(clip(Q_local))

### Effective weight

- **W_f,t = G_f × L_f,t**
- Combines per-frame quality with local tile quality.
- Used in phase 9 for tile reconstruction.

## Mathematical notation

```
Indices:
  f - frame index (0..N-1)
  t - tile index (0..T-1)
  k - cluster index (0..K-1)

Dimensions:
  N  - number of frames (after linearity rejection)
  K  - number of clusters / synthetic frames
  T  - number of tiles in the grid
  W,H - image width/height in pixels
  F  - seeing FWHM in pixels

Normalization:
  I_f      - original frame
  B_f      - background level (OSC: B_r, B_g, B_b separately)
  I'_f     - normalized frame = I_f / B_f
  s_f      - normalization scale = 1 / B_f

Global weights:
  B̃_f, σ̃_f, Ẽ_f - MAD-normalized metrics
  Q_f = α·(-B̃_f) + β·(-σ̃_f) + γ·Ẽ_f
  G_f = exp(clip(Q_f, -3, +3))

Local weights (STAR tiles):
  Q_f,t = w_fwhm·(-FWHM̃) + w_round·R̃ + w_contrast·C̃
  L_f,t = exp(clip(Q_f,t))

Local weights (STRUCTURE tiles):
  Q_f,t = w_metric·(Ẽ/σ̃) + w_bg·(-B̃)
  L_f,t = exp(clip(Q_f,t))

Effective weight:
  W_f,t = G_f × L_f,t

Tile reconstruction:
  tile_t = Σ_f W_f,t · tile_f,t / Σ_f W_f,t
  recon  = overlap_add(hanning(tile_t))

State vector (clustering):
  v_f = [G_f, ⟨Q_local⟩_f, Var(Q_local)_f, CC̄_tiles, WarpVar̄, invalid_frac]

Synthetic frames:
  synth_k = Σ_{f∈cluster_k} G_f · warp(I'_f) / Σ G_f
```

## Artifact files

Each run produces 11 JSON artifact files in `<run_dir>/artifacts/`:

| File | Phase | Contents |
|------|-------|----------|
| `normalization.json` | 4 | mode, bayer, B_mono / B_r / B_g / B_b per frame |
| `global_metrics.json` | 5 | background, noise, gradient, quality, G_f per frame |
| `tile_grid.json` | 6 | image dimensions, tile list (x,y,w,h), FWHM, overlap |
| `global_registration.json` | 1 | warp matrices (a00,a01,tx,a10,a11,ty) + CC per frame |
| `common_overlap.json` | 7 | global/tile-wise common valid area |
| `local_metrics.json` | 8 | tile metrics per frame×tile, tile type, quality, L_f,t |
| `tile_reconstruction.json` | 9 | valid counts, mean CC, post contrast/BG/SNR per tile |
| `state_clustering.json` | 10 | cluster labels, cluster sizes, method |
| `synthetic_frames.json` | 11 | number of synthetic frames, frames_min/max |
| `bge.json` | 15 | per-channel BGE diagnostics (samples, grid cells, residuals) |
| `validation.json` | 12 | FWHM improvement, tile-weight variance, pattern ratio |

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
  `global_registration.json`, `common_overlap.json`, `local_metrics.json`, `tile_reconstruction.json`,
  `state_clustering.json`, `synthetic_frames.json`, `bge.json`, `validation.json`
- Run events: `logs/run_events.jsonl`
- Run configuration: `config.yaml` (embedded into the report)

Typically readable content:

- Normalization/background trends (mono/RGB)
- Global quality metrics and weight distributions
- Star metrics (incl. FWHM, wFWHM, roundness, star count)
- Registration evaluation (shift/rotation/correlation)
- Tile/reconstruction heatmaps and local indicators
- Clustering and synthetic frame summaries
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
│   ├── local_metrics.json
│   ├── tile_reconstruction.json
│   ├── state_clustering.json
│   ├── synthetic_frames.json
│   ├── bge.json
│   ├── validation.json
│   ├── report.html       # generated via CLI/backend report path
│   └── *.png             # chart images
└── outputs/
    ├── stacked.fits
    ├── reconstructed_L.fit
    ├── stacked_rgb.fits       # (OSC only)
    ├── reconstructed_R.fit    # (OSC only)
    ├── reconstructed_G.fit    # (OSC only)
    ├── reconstructed_B.fit    # (OSC only)
    └── synthetic_*.fit        # (normal mode)
```

## Performance optimizations (C++)

- **Eigen matrices:** vectorized pixel operations (SIMD)
- **OpenCV:** optimized image processing (Sobel, Laplacian, warpAffine)
- **Thread parallelism:** TILE_RECONSTRUCTION using a `std::thread` worker pool
- **Pre-warping:** warp all frames once instead of per-tile
- **2× downsample:** registration at half resolution (speedup ~4×)
- **Memory-efficient:** frames are loaded from disk per phase
- **cv::setNumThreads(1):** avoids OpenCV thread contention in parallel tiles

## References

### Normative specification
  - `/doc/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.6_en.md`

### C++ implementation
  - `/tile_compile_cpp/apps/runner_pipeline.cpp`
  - **Configuration:** `/tile_compile_cpp/include/tile_compile/config/configuration.hpp`
  - **Report generator:** `/web_backend_cpp/src/services/report_generator.cpp`

---

**Note:** this document describes the **actual C++ code behavior**. If there are contradictions with the normative specification, the code is the reference for behavior and the specification is the reference for intent.
