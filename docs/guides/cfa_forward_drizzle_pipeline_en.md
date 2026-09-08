# CFA Forward Drizzle + Multiband Reconstruction

This is the **single reconstruction method** of the current `tile_compile`
pipeline. There is no method selector: every run uses CFA-aware forward drizzle
followed by multiband reconstruction and a three-way candidate selection. The
older Classic / AQMH tile-based methods are no longer offered in the product
(their documents under *Professional & Technical* describe historical behaviour).

## What it does

Each calibrated, registered light frame is drizzled **directly from its Bayer
(CFA) samples** onto the output grid with an exact square-droplet kernel. No
frame is debayered or interpolated before it contributes, so colour and
resolution information is carried straight from the sensor into the
reconstruction.

Three candidate images are built on the same geometry and the pipeline picks the
best one automatically:

| Candidate | How it is built | Role |
|---|---|---|
| **`drizzle_uniform`** | Equal per-sample weight | Robust control; the safety floor |
| **`drizzle_raw`** | Quality-weighted per source sample | Sharper where the data supports it |
| **`drizzle_multiband`** | À-trous multi-band fusion of the raw estimate with per-band confidence (`alpha`) | Best detail retention when confidence is high |

Selection is decided on fixed validation metrics (background RMS, seam score,
FWHM, elongation, tail). If the sharper candidate regresses on a mandatory
safety metric, the pipeline falls back — `drizzle_multiband` → `drizzle_raw` →
`drizzle_uniform` — and records the reason in the run report.

## Pipeline phases

The active reconstruction phases, in run order:

| Phase | Description |
|---|---|
| `NORMALIZED_CACHE` | Build / reuse the normalized CFA source cache (metadata-tagged) |
| `SAMPLING_GEOMETRY` | Registration sampling plan; direct rasterized channel support, `n_eff` and hole coverage; the **coverage gate** |
| `COMMON_OVERLAP` | Common valid-data overlap across accepted frames |
| `SOURCE_QUALITY_MAPS` | Per-frame source quality maps (pyramid) consumed by the raw/multiband weights |
| `GLOBAL_QUALITY` | Global per-frame quality weights |
| `FORWARD_DRIZZLE` | CFA forward drizzle → the transactional U/R/F/M profile store (CPU or CUDA, byte-identical) |
| `MULTIBAND` | À-trous fusion to `X_out`, three-way candidate selection, per-channel delivery |

Downstream phases (`STACKING` pass-through, `DEBAYER`, `ASTROMETRY`, `BGE`,
`PCC`, `HYPERMETRIC_STRETCH`) are unchanged.

### Coverage gate

`SAMPLING_GEOMETRY` runs a hard, fail-closed gate before any reconstruction:
valid channel fraction ≥ 0.995, p10 `n_eff` ≥ `max(3.0, 0.15·N)` per channel,
≥ 1024 analysis pixels, and no internal unsupported channel island. On a sparse
CFA field (few frames, weak dither, strong field rotation) an R or B sample
covers only a small fraction of the output grid per frame, so under-dithered
sets fail here rather than producing colour seams or comb artefacts. When
coverage is marginal the documented remedies are a global `pixfrac = 1.0` or
`internal_scale = 1`.

## Scale modes

`internal_scale` / `output_scale` in `reconstruction.drizzle`:

- **1 / 1** — native-resolution reconstruction.
- **2 / 2** — 2× supersampled reconstruction and output.
- **2 / 1** (production default) — reconstruct at internal 2×, then a single
  deterministic 2×2 area-average to native output. This is the recommended mode
  for critically sampled star fields.

## Outputs

In `runs/<run_id>/`:

| File | Contents |
|---|---|
| `outputs/reconstructed_<ch>.fit` | The **selected** candidate, per channel, in output-scale geometry |
| `outputs/forward_drizzle_raw_<ch>.fit` | The immutable Raw baseline, always written |
| `artifacts/reconstruction_multiband.fits` | The fused multiband `X_out` (fuse commit target) |
| `artifacts/forward_drizzle.json` | The reconstruction contract: geometry, coverage gate, candidate selection + validation gates, clipping, flux space, per-band alpha confidence, resources, throughput, and the reference machine |

With `diagnostics.level: full` the uniform and multiband control FITS are also
delivered. The `diagnostics.level` and the profile-cache retention settings
**never change the computed result** — only which extra files are written and
which caches are kept.

All planes are in the **normalized linear working space** (same space as
`reconstruction_multiband.fits`). The STACKING 17.4 normalization undo
(`scale_r/g/b`, pedestal) is applied downstream, not here.

## Caches and resume

| Setting | Default | Effect |
|---|---|---|
| `reconstruction.keep_profile_cache_after_run` | `false` | The internal U/R/F/M profile store is a reconstruction cache, deleted after a committed final image. `true` keeps it (hashed) so a re-fuse can skip the forward-drizzle pass, at the cost of disk. Does not affect results. |
| `reconstruction.delete_source_cache_after_run` | `false` | Keeps the normalized CFA source cache and quality maps so resume-reconstruction stays possible. Setting it `true` frees disk immediately but **disables resume-reconstruction** for the run; the report announces this as `resume_reconstruction_disabled`. |

The run report's **CFA Forward Drizzle / Multiband** section shows the coverage
gate, the candidate decision and its per-candidate validation metrics, the flux
space, the noise diagnostics, and the resources / throughput block — every value
matches the corresponding `forward_drizzle.json` / `sampling_geometry.json`
artifact.

## Related

- [Typical Workflow (GUI3)](workflow.md)
- [GUI3 User Guide](../gui3_user_guide_en.md)
- [Outputs & Artifacts](../reference/outputs.md)
