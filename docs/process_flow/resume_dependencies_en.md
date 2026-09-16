# Resume Dependencies

This document describes the implemented resume contract of the current
single-method runner (`tile_compile_runner resume-reconstruction`).
Legacy run layouts and legacy cache paths are outside that contract.

## Core rule

A resume is safe only when every input of the target phase exists as a
valid artifact or cache. `config.yaml` in the run directory is always
required, and `artifacts/run_provenance.json` must match the run.

The runner validates **fail-closed**: any missing artifact, hash
mismatch, or unsupported phase aborts with an error — it never falls back
to a silent partial run.

## Supported entry points

`resume-reconstruction --from-phase` accepts two groups of phases
(case-insensitive; `HMS` is an alias for `HYPERMETRIC_STRETCH`):

### Reconstruction resume

| Requested phase | Mechanism | Minimum dependencies | What runs |
|---|---|---|---|
| `GLOBAL_QUALITY` | Direct resume | `config.yaml` byte-identical to the run-start config (sha256 in `run_provenance.json`), `registration_sampling.json`, `sampling_geometry.json`, `forward_common_overlap.json`, geometry files + `forward_drizzle_geometry/` cache (when local warps exist), `cache/source_quality_maps/`, `source_quality_plan.json`, `cache/normalized_frames/` | GLOBAL_QUALITY, FORWARD_DRIZZLE, MULTIBAND, STACKING marker, all enabled downstream phases |
| `FORWARD_DRIZZLE` | Direct resume | all of the above plus `forward_drizzle_checkpoint.json` (geometry hash + recorded artifact sizes must match) and a consistent `forward_drizzle_v2/` store | FORWARD_DRIZZLE (from checkpoint), MULTIBAND, all enabled downstream phases |

### Downstream resume

These entry points reuse only the persisted reconstruction outputs and
never touch the M1–M3 predecessors (sampling geometry, forward drizzle):

| Requested phase | Minimum dependencies | Mutable config sections | What runs |
|---|---|---|---|
| `ASTROMETRY` | `outputs/stacked_rgb.fits` or `stacked_rgb_solve.fits` | `astrometry`, `bge`, `pcc`, `chroma_denoise`, `hypermetric_stretch`, `runtime_limits` | ASTROMETRY, BGE, PCC, HYPERMETRIC_STRETCH (as enabled) |
| `BGE` | as above | `bge`, `pcc`, `chroma_denoise`, `hypermetric_stretch`, `runtime_limits` | BGE, PCC, HYPERMETRIC_STRETCH |
| `PCC` | as above; with `bge.method != "none"` also a consistent `stacked_rgb_bge_linear.fits` | `pcc`, `chroma_denoise`, `hypermetric_stretch`, `runtime_limits` | PCC, HYPERMETRIC_STRETCH |
| `HYPERMETRIC_STRETCH` | as above plus `outputs/stacked_rgb_pcc.fits` (the linear HMS input) | `hypermetric_stretch`, `runtime_limits` | HYPERMETRIC_STRETCH |

For a downstream resume the config is compared **per top-level section**
against the run-start config (the revision under
`artifacts/config_revisions/`, else `config.yaml` when its sha256 still
matches provenance): only the sections listed above may differ. Any other
change is rejected with `FORWARD_STAGE_CONFIG_SCOPE_MISMATCH` — this lets
the HMS editor modify `hypermetric_stretch.*` without invalidating the
persisted PCC/reconstruction artifacts.

`MULTIBAND` is not a resume entry point; it always re-runs together with
FORWARD_DRIZZLE. Every other phase name is rejected with
`FORWARD_STAGE_UNSUPPORTED_RESUME_PHASE` — there is no in-place full
rerun disguised as a resume anymore.

## Cache dependencies

| Cache | Produced by | Required by resume |
|---|---|---|
| `cache/normalized_frames` | NORMALIZATION (+ NORMALIZED_CACHE seal) | reconstruction entry points (source-LRU feeds the gather) |
| `cache/source_quality_maps` | SOURCE_QUALITY_MAPS | reconstruction entry points (sample weights) |
| `artifacts/forward_drizzle_geometry` | SAMPLING_GEOMETRY | reconstruction entry points when local warp models exist; skipped entirely for affine-only runs |

`reconstruction.delete_source_cache_after_run: true` deletes
`cache/source_quality_maps` (and the normalized-frame cache) at run end —
a **reconstruction resume** is then impossible; downstream resume from
the persisted outputs keeps working.
`reconstruction.keep_profile_cache_after_run` controls retention of
profile caches independently.

## Environment overrides honored on resume

- `TC_FORWARD_DRIZZLE_MEMORY_BUDGET_MB` — overrides
  `reconstruction.drizzle.memory_budget_mb` without invalidating the
  checkpoint (the committed store is budget-invariant and bit-exact on the
  CPU path).

## Unsafe assumptions

- Requesting an unsupported phase does **not** degrade into a full rerun;
  it fails.
- A phase event does not replace a required artifact — the checkpoint
  verifies artifact byte sizes and the coverage-geometry hash.
- `STACKING` and `MULTIBAND` are not resume entry points. `ASTROMETRY`,
  `BGE`, `PCC`, `HYPERMETRIC_STRETCH` are downstream entry points: they
  need the persisted reconstruction outputs and allow changes only in the
  downstream config sections.
- A `config.yaml` that differs from the run-start config outside the
  allowed scope (or whose sha256 no longer matches `run_provenance.json`
  for a reconstruction resume) aborts the resume.

## Sources

- `tile_compile_cpp/apps/runner_forward_drizzle.cpp`
  (`run_forward_drizzle_stages`, resume validation)
- `tile_compile_cpp/apps/runner_pipeline.cpp`
- `web_backend_cpp/include/services/run_inspector.hpp`
  (`RESUME_FROM_PHASES`)
