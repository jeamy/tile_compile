# Process Flow Documentation — CFA Forward Drizzle + Multiband Pipeline (tile_compile_cpp)

## Overview

This document describes the **actual execution flow** of the C++ implementation
(`tile_compile_cpp/apps/runner_pipeline.cpp` → `runner_forward_drizzle.cpp`).

The pipeline has exactly **one reconstruction method**: **CFA Forward Drizzle +
Multiband**. It processes **FITS frames** (mono or OSC/CFA) and reconstructs the
stack by forward-mapping every calibrated source sample through the measured
registration geometry into a super-resolution output grid — per CFA channel for
OSC data, without any debayering step. The removed AQMH and Classic Tile-Compile
methods no longer exist in the code; a top-level `method:` key in a config is
rejected fail-closed.

**Implementation:** C++20 with Eigen, OpenCV, cfitsio, nlohmann/json, YAML-cpp.

**GUI3 integration:** The productive GUI path uses the web frontend plus the
Crow/C++ backend. Crow orchestrates the C++ pipeline by invoking
`tile_compile_cli` and `tile_compile_runner`; it does not reimplement the
processing logic.

The resume entry point (`resume-reconstruction`) and its artifact/cache
dependencies are documented in
[Resume Dependencies](resume_dependencies_en.md). Resume is supported from
`GLOBAL_QUALITY` and `FORWARD_DRIZZLE` (reconstruction) and from
`ASTROMETRY`, `BGE`, `PCC` and `HYPERMETRIC_STRETCH` (downstream chain on
the persisted outputs).

## Canonical phases (C++ implementation)

Phase order emitted by `tile_compile_runner reconstruct` (canonical order in
`web_backend_cpp/include/services/run_inspector.hpp`):

| # | Phase | Short description |
|---|-------|-------------------|
| 1 | `SCAN_INPUT` | FITS enumeration, header/mode detection (MONO/OSC + Bayer pattern), linearity validation, disk-space precheck |
| 2 | `CHANNEL_SPLIT` | Metadata-only: records color mode, channels and Bayer pattern. No files are split — CFA channel assignment happens per pixel inside FORWARD_DRIZZLE |
| 3 | `NORMALIZATION` | Additive background `B_f` and photometric scale `P_f` per frame (per CFA channel for OSC); writes `cache/normalized_frames/` and `artifacts/normalization.json` |
| 4 | `REGISTRATION` | Global registration (cascaded fallbacks) on registration proxies; affine + optional smooth local warp per frame; writes `global_registration.json` and the `registration_sampling.json` plan |
| 5 | `NORMALIZED_CACHE` | Seals the normalized-frame cache against the sampling plan; required for every downstream stage |
| 6 | `SAMPLING_GEOMETRY` | Output grid, coverage masks, and the local-warp geometry cache (`artifacts/forward_drizzle_geometry/`); writes `sampling_geometry.json` |
| 7 | `COMMON_OVERLAP` | Pixelwise common-valid coverage of all frames; writes `forward_common_overlap.json` and the analysis/reconstruction support masks |
| 8 | `SOURCE_QUALITY_MAPS` | Per-frame pixelwise source-quality maps cached under `cache/source_quality_maps/`; writes `source_quality_plan.json` |
| 9 | `GLOBAL_QUALITY` | Global quality gate: aggregates source maps into frame weights and the final weight plan; a resume entry point |
| 10 | `FORWARD_DRIZZLE` | The core reconstruction: chunked forward-drizzle gather of all source samples into the banded v2 store (`artifacts/forward_drizzle_v2/`); writes `forward_drizzle.json` + checkpoint |
| 11 | `MULTIBAND` | Multiband (low-/high-frequency) reconstruction from the uniform store and band passes; writes `reconstruction_multiband.fits` |
| 12 | `ASTROMETRY` | Optional plate solving / WCS (ASTAP, local catalog fallback) |
| 13 | `BGE` | Optional background-gradient extraction before PCC |
| 14 | `PCC` | Optional photometric color calibration |
| 15 | `HYPERMETRIC_STRETCH` | Optional VeraLux HyperMetric Stretch (explicit non-linear stretch, runs last) |

`STACKING` is emitted as a pass-through marker before the downstream
post-processing phases; there is no classic stacking stage — the drizzle store
*is* the stack.

## Pipeline flow diagram

```
┌─────────────────────────────────────────────────────────────┐
│               INPUT: MONO / OSC RAW FITS FRAMES             │
└────────────────────────────┬────────────────────────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  SCAN_INPUT                  │
              │  • FITS dims + headers       │
              │  • MONO/OSC + Bayer detect   │
              │  • linearity + disk precheck │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  CHANNEL_SPLIT (metadata)    │
              │  • mode + Bayer pattern      │
              │  • deferred to drizzle       │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  NORMALIZATION               │
              │  • B_f / P_f per frame       │
              │  • per CFA channel (OSC)     │
              │  • cache/normalized_frames/  │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  REGISTRATION                │
              │  • cascaded global align     │
              │  • affine + local warp       │
              │  • registration_sampling.json│
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  NORMALIZED_CACHE            │
              │  • seals cache vs. plan      │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  SAMPLING_GEOMETRY           │
              │  • output grid + coverage    │
              │  • local-warp geometry cache │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  COMMON_OVERLAP              │
              │  • common-valid masks        │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  SOURCE_QUALITY_MAPS         │
              │  • per-pixel source quality  │
              │  • cache/source_quality_maps │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  GLOBAL_QUALITY              │
              │  • frame weights + gate      │
              │  • resume entry point        │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  FORWARD_DRIZZLE             │
              │  • chunked CFA gather        │
              │  • banded v2 store           │
              │  • resume entry point        │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  MULTIBAND                   │
              │  • band fusion               │
              │  • reconstruction_*.fits     │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  ASTROMETRY → BGE → PCC →    │
              │  HYPERMETRIC_STRETCH         │
              │  (each optional)             │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  OUTPUTS:                    │
              │  • reconstructed stack FITS  │
              │  • artifacts/*.json          │
              │  • logs/run_events.jsonl     │
              └──────────────────────────────┘
```

## Core principles

1. **Single method:** there is no method selector. Configs carrying `method:` or
   `reconstruction.engine:` are rejected fail-closed; legacy structural blocks
   (`aqmh`, `pipeline`, `tile`, `local_metrics`, `synthetic`, …) are stripped
   with a migration warning.
2. **No debayering:** for OSC input each source pixel is assigned to its CFA
   channel per-pixel inside the drizzle gather
   (`cfa_channel_for_source_pixel` with `cfa_origin` offsets from the sampling
   plan). R/G/B output planes are produced directly.
3. **Forward mapping:** source samples are splatted into the output grid
   (sub-pixel accurate, pixfrac-controlled drop size) instead of inverse-mapping
   output pixels — this is what makes the CFA-aware super-resolution possible.
4. **Bounded memory:** FORWARD_DRIZZLE runs in row chunks with a configurable
   `reconstruction.drizzle.memory_budget_mb`; the banded store is written
   incrementally, input amplification stays low.
5. **Linearity:** all phases up to and including PCC stay linear;
   HYPERMETRIC_STRETCH is the explicit final non-linear step.
6. **Resume contract:** `resume-reconstruction --from-phase` supports
   `GLOBAL_QUALITY` and `FORWARD_DRIZZLE` (reconstruction resume; all
   predecessor artifacts plus `run_provenance.json` config sha256 must
   validate) and `ASTROMETRY`, `BGE`, `PCC`, `HYPERMETRIC_STRETCH`
   (downstream resume on the persisted outputs; only the downstream config
   sections may differ from the run-start config).

## Document structure

| File | Contents |
|------|----------|
| [phase_0_overview.md](phase_0_overview.md) | Phase table, artifact map, configuration surface |
| [phase_1_scan_normalization.md](phase_1_scan_normalization.md) | SCAN_INPUT, CHANNEL_SPLIT, NORMALIZATION |
| [phase_2_registration_geometry.md](phase_2_registration_geometry.md) | REGISTRATION, NORMALIZED_CACHE, SAMPLING_GEOMETRY, COMMON_OVERLAP |
| [phase_3_quality.md](phase_3_quality.md) | SOURCE_QUALITY_MAPS, GLOBAL_QUALITY |
| [phase_4_forward_drizzle.md](phase_4_forward_drizzle.md) | FORWARD_DRIZZLE gather, chunking, store, checkpoint |
| [phase_5_multiband.md](phase_5_multiband.md) | MULTIBAND band fusion |
| [phase_6_postprocessing.md](phase_6_postprocessing.md) | ASTROMETRY, BGE, PCC, HYPERMETRIC_STRETCH |
| [data_flow_user_description_en.md](data_flow_user_description_en.md) | Data-centric walkthrough (EN) |
| [data_flow_user_description_de.md](data_flow_user_description_de.md) | Data-centric walkthrough (DE) |
| [resume_dependencies_en.md](resume_dependencies_en.md) | Resume contract (EN) |
| [resume_dependencies_de.md](resume_dependencies_de.md) | Resume contract (DE) |
| [flow_analysis.md](flow_analysis.md) | Implementation-level notes |

### Normative specification

- `/docs/forward_drizzle_v2_zielarchitektur_2026-09-12_de.md`
- `/docs/AQMH/aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md`
  (the implementation plan the single-method cutover follows)

### C++ implementation

- `/tile_compile_cpp/apps/runner_pipeline.cpp` — scan + early phases
- `/tile_compile_cpp/apps/runner_phase_metrics.cpp` — CHANNEL_SPLIT, NORMALIZATION
- `/tile_compile_cpp/apps/runner_phase_registration.cpp` — REGISTRATION
- `/tile_compile_cpp/apps/runner_forward_drizzle.cpp` — NORMALIZED_CACHE … MULTIBAND + downstream orchestration
- `/tile_compile_cpp/apps/runner_downstream.cpp` — ASTROMETRY, BGE, PCC, HMS
