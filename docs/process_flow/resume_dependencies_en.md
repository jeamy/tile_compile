# Resume Dependencies

This document describes the implemented resume contract of the current
runner. Legacy run layouts and legacy cache paths are outside that contract.

## Core rule

A resume is safe only when every input of the target phase exists as a valid
artifact or cache. `config.yaml` in the run directory is always required.

The CLI accepts several phase names, but they do not all have the same
semantics:

- **In-place full rerun:** Early phases are not continued from the selected
  phase. The runner reads `input_dir` from the latest `run_start` event and
  starts a complete pipeline run in the same run directory.
- **Direct resume:** The phase loads persisted artifacts and starts the
  corresponding downstream path.

## Supported entry points

| Requested phase | Mechanism | Actual entry | Minimum dependencies | Result |
|---|---|---|---|---|
| `SCAN_INPUT`, `CHANNEL_SPLIT`, `NORMALIZATION`, `GLOBAL_METRICS`, `TILE_GRID`, `REGISTRATION`, `PREWARP`, `COMMON_OVERLAP`, `LOCAL_METRICS`, `TILE_RECONSTRUCTION`, `STATE_CLUSTERING`, `SYNTHETIC_FRAMES`, `DEBAYER` | In-place full rerun | new complete `run` | `config.yaml`, valid latest `run_start` event with `input_dir`, readable input frames | all phases are regenerated; existing artifacts are not guaranteed as resume inputs |
| `AQMH_MAPS`, `AQMH_GLOBAL_QUALITY`, `AQMH_RECONSTRUCTION`, `AQMH_DIAGNOSTICS` | Direct AQMH resume | `AQMH_RECONSTRUCTION` | `artifacts/aqmh_metrics.json`, `cache/aqmh/aqmh_cache.json`, valid `cache/prewarped_frames`, `outputs/canvas_mask.fits`; if its dimensions differ, reconstructible `cache/aqmh_masks` is also required | new AQMH reconstruction, diagnostics, then `STACKING`/`DEBAYER` |
| `STACKING` with AQMH | Direct resume or AQMH reconstruction | `STACKING` if the raw artifact exists, otherwise `AQMH_RECONSTRUCTION` | preferably `outputs/aqmh_reconstructed_raw.fit`; if absent, all AQMH dependencies above | stacking, debayer, and later phases |
| `STACKING` with Classic | Direct resume | `STACKING` | at least one valid `outputs/synthetic_*.fit`; optionally `artifacts/synthetic_frames.json`, `artifacts/global_registration.json`, masks | stacking, debayer, and later phases |
| `ASTROMETRY` | Direct post-processing resume | `ASTROMETRY` | `outputs/stacked_rgb_solve.fits` or `outputs/stacked_rgb.fits`; an existing `artifacts/stacked_rgb.wcs` may be used as fallback | astrometry, then BGE and PCC |
| `BGE` | Direct post-processing resume | `BGE` | RGB output as for `ASTROMETRY`, `outputs/canvas_mask.fits`; Classic additionally needs matching `artifacts/local_metrics.json` and `artifacts/tile_grid.json`, while AQMH can derive tile data from RGB | BGE, then PCC |
| `PCC` | Direct post-processing resume | `PCC` | RGB output; optionally `outputs/stacked_rgb_bge_linear.fits`; enabled PCC requires valid WCS or a persisted WCS artifact | PCC |
| `HYPERMETRIC_STRETCH`/`HMS` | Direct post-processing resume | `HYPERMETRIC_STRETCH` | preferably `outputs/pcc_R.fit`, `pcc_G.fit`, `pcc_B.fit` or `outputs/stacked_rgb_pcc.fits`; with `require_successful_pcc: false`, BGE/Solve RGB is an alternative | HMS output |

## Cache dependencies

All current caches are under `cache/`:

| Cache | Produced by | Directly required by |
|---|---|---|
| `cache/normalized_frames` | normalization | in-place full reruns and later pipeline phases of that run |
| `cache/prewarped_frames` | registration/prewarp | `AQMH_RECONSTRUCTION` and AQMH resume entries mapped to it |
| `cache/aqmh` | AQMH quality maps | AQMH reconstruction and diagnostics |
| `cache/aqmh_masks` | AQMH/mask phases | AQMH resume when `outputs/canvas_mask.fits` is not full-canvas sized |
| `cache/phase9_osc_rgb` | RGB/phase-9 processing | only where the concrete downstream implementation uses it |

`aqmh.reconstruction.delete_prewarped_cache_after_run: true` deletes the
prewarp cache after the run. A direct AQMH resume from
`AQMH_RECONSTRUCTION` is then unavailable; an in-place full rerun remains
possible when the input log and input data are valid.

## Unsafe assumptions

- Requesting an early phase does **not** mean that only that phase runs.
- A phase event does not replace a required artifact.
- Caches outside `cache/prewarped_frames`, `cache/normalized_frames`,
  `cache/aqmh`, and `cache/aqmh_masks` are not read.
- For AQMH, `STACKING` without `aqmh_reconstructed_raw.fit` is not a pure
  stacking resume; a valid AQMH reconstruction is required first.

## Sources

- `tile_compile_cpp/apps/runner_resume.cpp`
- `tile_compile_cpp/apps/runner_pipeline.cpp`
- `web_backend_cpp/include/services/run_inspector.hpp`
