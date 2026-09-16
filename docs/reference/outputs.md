# Outputs and Artifacts

After a successful run (`runs/<run_id>/`):

## `outputs/`

- `reconstructed_L.fit` (mono) or `reconstructed_R/G/B.fit` (OSC) — the
  selected reconstruction candidate
- `forward_drizzle_raw_*.fit` — raw-candidate exports (per
  `reconstruction.diagnostics.*`)
- `forward_drizzle_uniform_*.fit` / `forward_drizzle_multiband_*.fit` —
  emitted when the corresponding diagnostics flags are enabled
- `pcc_R/G/B.fit` — photometrically calibrated channels (PCC enabled)
- `stacked_rgb*.fits` / WCS artifacts — solved/calibrated RGB products
  from the optional downstream phases
- `live_edit.fits` / `live_image_export_*` — Live-Editor exports

## `artifacts/`

- `run_provenance.json` — resume anchor (config sha256)
- `run_start_config_claim.json`, `config_revisions/` — config snapshots
- `normalization.json` — per-frame `B_*`/`P_*` (per CFA channel for OSC)
- `global_metrics.json` — global frame metrics/weights
- `global_registration.json` — warp matrices, CC, dithering diagnostics
- `registration_sampling.json` — frozen sampling plan
- `sampling_geometry.json`, `sampling_geometry_*_mask.fits` — output grid
  and coverage masks
- `forward_drizzle_geometry/` — local-warp geometry cache
- `forward_common_overlap.json` — common-overlap statistics
- `source_quality_plan.json` — source-quality weight plan
- `forward_drizzle_v2/` — banded uniform store (the actual stack)
- `forward_drizzle_checkpoint.json` — resume checkpoint
- `forward_drizzle.json` — gather summary
- `forward_drizzle_geometry_profile.json` — geometry instrumentation
- `reconstruction_multiband.fits` — fused multiband result
- `forward_downstream_inputs.json` — hand-off to downstream phases
- `bge.json`, `validation.json`, `acceleration_context.json`,
  `runtime_limits.json`, `config_migration.json` (when migration
  stripped legacy keys)
- `pi_run_provenance.json`, `pi_run_quality.json` — PI metadata
- `report.html`, `stats.json` — via `POST /api/runs/<id>/stats` (GUI: Generate Stats)

## `cache/`

- `normalized_frames/` — sealed normalized source frames
- `source_quality_maps/` — per-frame quality maps
- Both are deleted at run end when
  `reconstruction.delete_source_cache_after_run: true` (which disables
  resume).

## Other

- `logs/run_events.jsonl` — phase events, progress, warnings
- `config.yaml` — frozen run configuration snapshot
