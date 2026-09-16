# Flow Analysis — CFA Forward Drizzle + Multiband

## Executive Summary

This analysis describes the current `tile_compile_cpp` implementation after
the single-method cutover. There is exactly one reconstruction method:
**CFA Forward Drizzle + Multiband**. AQMH and Classic Tile-Compile have been
removed; a `method:` selector in a config is rejected fail-closed, and the
removed structural config blocks are stripped by the migration layer with
warnings.

The authoritative execution path is:

```text
SCAN_INPUT -> CHANNEL_SPLIT -> NORMALIZATION -> REGISTRATION
-> NORMALIZED_CACHE -> SAMPLING_GEOMETRY -> COMMON_OVERLAP
-> SOURCE_QUALITY_MAPS -> GLOBAL_QUALITY -> FORWARD_DRIZZLE -> MULTIBAND
-> [STACKING marker] -> ASTROMETRY -> BGE -> PCC -> HYPERMETRIC_STRETCH
```

## 1. Core architecture

The runner is split into three orchestration layers:

- `runner_pipeline.cpp` — process entry, SCAN_INPUT, runtime limits,
  memory planning
- `runner_phase_metrics.cpp` / `runner_phase_registration.cpp` — early
  photometric stages and geometry measurement
- `runner_forward_drizzle.cpp` — the stage machine
  (`run_forward_drizzle_stages`) for NORMALIZED_CACHE through MULTIBAND,
  checkpoint/resume validation, cache retention, downstream dispatch
- `runner_downstream.cpp` — optional ASTROMETRY/BGE/PCC/HMS phases

Execution is strictly sequential; there are no feedback loops. Phase
events (`phase_start`/`phase_end`/`phase_progress`) go to
`logs/run_events.jsonl` and drive the backend run inspector and the UI.

## 2. Data and cache design

Two persistent caches feed the reconstruction core:

- `cache/normalized_frames/` — written in NORMALIZATION, sealed in
  NORMALIZED_CACHE. After the seal it is read-only; every consumer
  (quality maps, gather) sees identical pixels.
- `cache/source_quality_maps/` — per-frame quality maps from
  SOURCE_QUALITY_MAPS, reused by GLOBAL_QUALITY and FORWARD_DRIZZLE.

A third, artifact-side cache exists under
`artifacts/forward_drizzle_geometry/`: the materialized local-warp
sampling geometry per geometry variant (`.leaves` + `manifest.json` +
`.rows` index). It is built once in SAMPLING_GEOMETRY and then shared by
coverage, GLOBAL_QUALITY and the gather — no consumer re-runs
`sample_leaves` per stripe. Affine-only runs skip the cache entirely.

Cache retention is explicit: `delete_source_cache_after_run` removes both
source caches at run end (disabling resume, reported in `run_end` as
`cache_retention`); `keep_profile_cache_after_run` controls the profile
cache separately.

## 3. The gather and the store

FORWARD_DRIZZLE is a **forward** mapping: for each output row chunk, all
contributing source samples are splatted into `wx`/`w`/`w²` accumulators.
Key properties:

- **CFA-native:** OSC pixels are routed per-pixel via
  `cfa_channel_for_source_pixel` into R/G/B accumulator planes; the Bayer
  pattern is never interpolated.
- **Banded store:** `artifacts/forward_drizzle_v2/` holds uniform and
  band accumulators; MULTIBAND fuses them without a second pass over the
  sources.
- **Bounded memory:** `drizzle.chunk_rows`/`chunk_halo_rows` bound the
  working set; `drizzle.memory_budget_mb` caps the source LRU and band
  planner. `TC_FORWARD_DRIZZLE_MEMORY_BUDGET_MB` allows budget sweeps
  without invalidating the checkpoint (the committed store is
  budget-invariant and bit-exact on the CPU path).
- **Robust reduction:** `clipping.*` sigma-clipping, minimum contributor
  and `n_eff` gates, `guard_fallback` for under-supported regions.

## 4. Resume contract

`resume-reconstruction --from-phase` supports exactly `GLOBAL_QUALITY`
and `FORWARD_DRIZZLE`. Validation is fail-closed:

- `run_provenance.json` config sha256 must match `config.yaml`
- `forward_drizzle_checkpoint.json` must match the coverage-geometry hash
  and recorded artifact byte sizes
- all required caches must exist and parse

Everything downstream of the resume point (including MULTIBAND and all
optional post-processing) re-runs. Unsupported phase names are rejected
with `FORWARD_STAGE_UNSUPPORTED_RESUME_PHASE`.

## 5. Failure and safety properties

- **Fail-closed config:** `method:`/`reconstruction.engine:` throw
  `UNKNOWN_LEGACY_KEY`; removed structural blocks strip with warnings;
  `validate-config` applies the same migrated parser as the runner, and
  the backend validates the effective run config before snapshotting —
  invalid configs never reach the runner.
- **Terminal early errors:** inconsistent color modes, missing inputs,
  disabled normalization, insufficient common overlap all abort before
  the expensive stages.
- **Skipped vs. error:** optional downstream phases may end `skipped`
  (e.g. no astrometric solution) without failing the run.
- **CPU fallback:** acceleration backends must preserve CPU semantics
  within documented tolerances; the CPU path is the bit-exactness
  reference.

## 6. Remaining risks / observations

1. **Storage-bound gather:** the source-LRU + chunked store shifts the
   bottleneck to disk throughput on large stacks; the memory budget and
   `chunk_rows` are the operational levers.
2. **Resume foot-gun:** `delete_source_cache_after_run: true` silently
   makes resume impossible — it is reported in `run_end`, but operators
   should treat it as a one-way decision.
3. **Local-warp cache size:** the geometry cache scales with
   `local_frames × source_pixels × variants`; the pre-flight estimate in
   SAMPLING_GEOMETRY guards disk space but can reject runs on tight
   volumes.
4. **Provenance strictness:** any post-run edit of `config.yaml`
   invalidates resume — intended, but worth surfacing in the UI.

## Sources

- `tile_compile_cpp/apps/runner_forward_drizzle.cpp`
- `tile_compile_cpp/apps/runner_pipeline.cpp`
- `tile_compile_cpp/src/reconstruction/forward_drizzle_v2*.cpp`
- `docs/forward_drizzle_v2_zielarchitektur_2026-09-12_de.md`
