# Process Flow – Technical Data Flow of the System

## Pipeline objective

The system turns a set of calibrated astronomical single-frame inputs into
a reproducible final product inside a shared geometric and photometric
reference space, using exactly one reconstruction method:
**CFA Forward Drizzle + Multiband**.

Technically, the pipeline is organized into three major blocks:

- **Preparation and normalization**
  - validate inputs
  - normalize intensity levels (per CFA channel for OSC)
  - measure registration geometry
- **Quality modeling and forward reconstruction**
  - seal the normalized-frame cache
  - derive output grid, coverage masks, and geometry caches
  - compute per-pixel source-quality maps and global frame weights
  - forward-drizzle every source sample into a banded super-resolution
    store
  - fuse the multiband result
- **Post-processing and calibration**
  - astrometry / WCS
  - optional BGE
  - optional PCC
  - optional HyperMetric Stretch

The primary product is a linear reconstructed image (mono or per-channel
RGB for OSC). There is no method selector: AQMH and Classic Tile-Compile
no longer exist, and a `method:` key in a config is rejected fail-closed.

## Core terms

- **Run** — one full pipeline execution with its own run directory under
  `runs/<run_id>/`.
- **Phase** — one well-defined processing stage such as `REGISTRATION`,
  `FORWARD_DRIZZLE`, or `PCC`.
- **Artifact** — persisted diagnostic or intermediate data, typically
  under `artifacts/`.
- **Event timeline** — chronological execution events written to
  `logs/run_events.jsonl`.
- **Sampling plan** — `artifacts/registration_sampling.json`: the frozen
  geometric contract (output grid, affine + local warps, CFA origin) all
  downstream stages obey.
- **Uniform store** — the banded accumulator set under
  `artifacts/forward_drizzle_v2/` produced by the gather; the single
  source for the multiband fusion and the exports.
- **Resume** — `resume-reconstruction` continues a run from
  `GLOBAL_QUALITY` or `FORWARD_DRIZZLE` only.

## Overall flow

```text
Input frames (FITS, MONO or OSC/CFA)
   -> SCAN_INPUT
   -> CHANNEL_SPLIT            (metadata only — no file splits)
   -> NORMALIZATION            (B_f/P_f per frame & CFA channel,
                                cache/normalized_frames/)
   -> REGISTRATION             (affine + optional local warp,
                                registration_sampling.json)
   -> NORMALIZED_CACHE         (seal cache against the plan)
   -> SAMPLING_GEOMETRY        (output grid, coverage masks,
                                forward_drizzle_geometry/ cache)
   -> COMMON_OVERLAP           (common-valid support masks)
   -> SOURCE_QUALITY_MAPS      (cache/source_quality_maps/)
   -> GLOBAL_QUALITY           (frame weights + gates)
   -> FORWARD_DRIZZLE          (chunked CFA gather,
                                artifacts/forward_drizzle_v2/)
   -> MULTIBAND                (band fusion,
                                reconstruction_multiband.fits)
   -> STACKING                 (pass-through marker)
   -> ASTROMETRY (optional)
   -> BGE        (optional)
   -> PCC        (optional)
   -> HYPERMETRIC_STRETCH (optional)
```

## Why forward drizzle instead of classic stacking

- **Sub-pixel accuracy:** each source sample is splatted into the output
  grid through the measured per-frame geometry (affine + optional smooth
  local warp), so dithered data yields true super-resolution.
- **CFA-native:** for OSC input the Bayer pattern is never interpolated —
  `cfa_channel_for_source_pixel` routes every source pixel into the
  matching R/G/B accumulator plane. No debayering artifacts, full
  dither benefit per channel.
- **Quality-weighted:** per-pixel source-quality maps × global frame
  weights replace any global frame rejection; bad pixels lose weight,
  not whole frames.
- **Bounded memory:** row-chunked gather with a configurable memory
  budget; input amplification stays low even for large stacks.

## Phases in detail

### SCAN_INPUT

- Enumerates FITS inputs, reads dimensions and headers (EXPTIME, filter,
  Bayer keys).
- Detects MONO vs. OSC and the Bayer pattern; validates linearity and
  free disk space (`scandir × 4` conservative estimate).
- Mode inconsistencies or unreadable inputs are terminal here.

### CHANNEL_SPLIT (metadata)

- Records the color mode and Bayer pattern in the event stream
  (`note: deferred_to_forward_drizzle_cfa`).
- Emits **no files**: the actual per-pixel channel assignment happens
  inside the drizzle gather.

### NORMALIZATION

- Computes the additive background `B_f` and photometric scale `P_f` per
  frame — per CFA channel for OSC (`B_r/g/b`, `P_r/g/b`).
- Photometric scaling uses `exposure_ratio` when all EXPTIME headers are
  valid, otherwise `identity_fallback`.
- Persists normalized frames to `cache/normalized_frames/` and writes
  `normalization.json` + `global_metrics.json` (early frame weights
  `G_f`).
- `normalization.enabled: false` is terminal.

### REGISTRATION

- Cascaded global registration on downsampled registration proxies
  (CFA-safe for OSC).
- Produces an affine transform and optionally a smooth local warp model
  per frame; failures degrade to identity warp with CC=0 — no hard frame
  rejection.
- Failed or rejected frames receive a model-predicted warp (field-rotation
  polynomial / blended / interpolated / nearest-copy) only when an
  always-on plausibility gate accepts it: the prediction must stay within
  `1 deg + 2x` the measured local/global rotation rate (capped at 5 deg)
  and within `60 px + 2x` the measured shift rate (capped at 400 px) of
  the neighbouring measured anchors, and extrapolation past the anchor
  span is limited to 3 frames. Implausible predictions leave the frame
  `unresolved` (excluded from canvas, prewarp and drizzle). Counts are
  reported in `global_registration.json` under
  `diag.reg_model_predicted_implausible`,
  `..._reasons` and `..._frames`.
- Writes `global_registration.json` and the frozen
  `registration_sampling.json` plan (output dimensions, `internal_scale`,
  `cfa_origin`, per-frame transforms).
- `run_provenance.json` anchors the config sha256 for later resume
  validation.

### NORMALIZED_CACHE

- Seals `cache/normalized_frames/` against the sampling plan. From here
  on the cache is read-only input for every consumer.

### SAMPLING_GEOMETRY

- Builds the output grid and coverage masks
  (`analysis_common_mask`, `reconstruction_support_mask`).
- Materializes the local-warp geometry cache
  `artifacts/forward_drizzle_geometry/` for frames with local models —
  affine-only runs skip it entirely.
- Writes `sampling_geometry.json`; the coverage-geometry hash feeds the
  resume checkpoint.

### COMMON_OVERLAP

- Computes the pixelwise common-valid coverage of all frames.
- Enforces `reconstruction.common_overlap_required_fraction`;
  writes `forward_common_overlap.json` and fixes the support masks for
  the remainder of the run.

### SOURCE_QUALITY_MAPS

- Per-frame, per-pixel quality maps from the
  `reconstruction.quality.pyramid.*` configuration.
- Persisted under `cache/source_quality_maps/`; plan summarized in
  `source_quality_plan.json`.

### GLOBAL_QUALITY

- Aggregates source maps into final frame weights `G_f` and applies the
  coverage/clipping gates (`coverage_gate.*`,
  `common_overlap_required_fraction`, `clipping.min_n_eff`).
- First supported resume entry point.

### FORWARD_DRIZZLE

- The core gather: for each row chunk (`drizzle.chunk_rows` +
  `chunk_halo_rows`) every contributing source sample is splatted with
  `drizzle.kernel`/`drizzle.pixfrac` into `wx`/`w`/`w²` accumulators of
  the banded v2 store — per CFA channel for OSC.
- Robust reduction follows `clipping.*` (`clip_sigma_low/high`,
  `min_clip_contributors`, `robust_passes`, `guard_fallback`).
- Memory is bounded by `drizzle.memory_budget_mb`; the checkpoint
  (`forward_drizzle_checkpoint.json`) records the geometry hash and
  artifact sizes for resume.

### MULTIBAND

- Fuses the banded store into `reconstruction_multiband.fits` per
  `reconstruction.multiband.*`; candidate intermediates spool through
  `artifacts/multiband_candidate_spool/`.
- Always re-runs on resume (not an entry point).

### STACKING (pass-through marker)

- Emitted before downstream processing for status/report continuity.
  The drizzle store already *is* the stack — no classic stacking stage
  exists.

### ASTROMETRY / BGE / PCC / HYPERMETRIC_STRETCH

- Optional downstream phases, fixed order, each independently skippable.
- Everything up to and including PCC stays linear; HMS is the explicit
  non-linear final step.

## Typical run structure

```text
runs/<run_id>/
├── config.yaml                     # frozen run config
├── logs/run_events.jsonl           # event stream
├── artifacts/
│   ├── run_provenance.json         # resume anchor (config sha256)
│   ├── normalization.json
│   ├── global_metrics.json
│   ├── global_registration.json
│   ├── registration_sampling.json
│   ├── sampling_geometry.json
│   ├── forward_common_overlap.json
│   ├── forward_drizzle_geometry/
│   ├── source_quality_plan.json
│   ├── forward_drizzle_v2/
│   ├── forward_drizzle_checkpoint.json
│   ├── forward_drizzle.json
│   ├── reconstruction_multiband.fits
│   └── report.html                 # via POST /api/runs/<id>/stats
├── cache/
│   ├── normalized_frames/
│   └── source_quality_maps/
└── outputs/                        # final FITS products
```

## Resume

Only two entry points exist — `GLOBAL_QUALITY` and `FORWARD_DRIZZLE`.
Both validate provenance (config sha256), the checkpoint (geometry hash,
artifact sizes), and all required caches fail-closed. Downstream phases
always re-run. See [resume_dependencies_en.md](resume_dependencies_en.md).

## Evaluation with the integrated report generator

```text
POST /api/runs/<run_id>/stats  (GUI: Generate Stats)
```

produces `artifacts/report.html` (self-contained, inline SVG) and
`artifacts/stats.json` from the artifact JSONs and `run_events.jsonl` — normalization trends,
registration evaluation, coverage/support heatmaps, drizzle and
multiband diagnostics, downstream (BGE/PCC) results, and the pipeline
timeline.

## Notes on interpretation

- A `skipped` downstream phase (e.g. no astrometric solution) is not a
  failure; check the phase-end payload for the reason.
- `channels` in CHANNEL_SPLIT only describes the channel model — count
  actual per-channel coverage in `forward_drizzle.json`.
- `STACKING` finishing instantly is expected: it is a marker, the real
  work happened in FORWARD_DRIZZLE/MULTIBAND.

## Short conclusion

The pipeline is a single, deterministic CFA-forward-drizzle
reconstruction path: normalize, register, measure per-pixel quality,
splat every sample into a banded super-resolution store, fuse the bands,
then optionally solve, correct, calibrate, and stretch — with a strict
fail-closed resume contract at the two expensive boundaries.
