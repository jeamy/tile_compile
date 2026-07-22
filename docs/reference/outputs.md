# Outputs and Artifacts

After a successful run (`runs/<run_id>/`):

## `outputs/`

- `stacked.fits`
- `reconstructed_L.fit`
- `stacked_rgb.fits` (OSC)
- `stacked_rgb_solve.fits` / WCS artifacts
- `stacked_rgb_bge.fits` (BGE-only snapshot before PCC)
- `stacked_rgb_pcc.fits`
- `stacked_rgb_hms.fits` (optional VeraLux HyperMetric Stretch output)
- `synthetic_*.fit` (mode-dependent)

## `artifacts/`

- `normalization.json`
- `global_metrics.json`
- `tile_grid.json`
- `global_registration.json`
- `local_metrics.json`
- `tile_reconstruction.json`
- `aqmh.json` (AQMH quality diagnostics — artifact fractions, regional quality stats)
- `state_clustering.json`
- `synthetic_frames.json`
- `bge.json`
- `validation.json`
- `report.html`, `report.css`, `*.png`

## Other

- `logs/run_events.jsonl`
- `config.yaml` (run snapshot)
