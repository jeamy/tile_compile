# CLI Reference

## CLI Runner

```bash
./tile_compile_runner \
  reconstruct \
  --config ../tile_compile.yaml \
  --input-dir /path/to/lights \
  --runs-dir /path/to/runs
```

### Common options

- `--max-frames <n>` limit frames (`0` = no limit)
- `--dry-run` execute validation flow without full processing
- `--run-id <id>` custom run id for grouping
- `--project-root <path>` project root directory
- `--stdin` with `--config -` to read YAML from stdin

### Resume mode

```bash
./tile_compile_runner resume-reconstruction \
  --run-dir /path/to/runs/<run_id> \
  --from-phase FORWARD_DRIZZLE
```

Supported resume phases (case-insensitive; `HMS` is an alias for
`HYPERMETRIC_STRETCH`):

- **Reconstruction resume:** `GLOBAL_QUALITY` and `FORWARD_DRIZZLE`.
  Re-uses the checked M1–M3 predecessors (normalized-frame cache, geometry,
  overlap, quality maps, weights, uniform store) and re-runs
  `FORWARD_DRIZZLE`/`MULTIBAND` plus the downstream phases. `config.yaml`
  must be byte-identical to the run-start config (sha256 in
  `run_provenance.json`).
- **Downstream resume:** `ASTROMETRY`, `BGE`, `PCC`,
  `HYPERMETRIC_STRETCH`. Reuses the persisted reconstruction outputs
  (`outputs/stacked_rgb*.fits`; HMS additionally requires
  `outputs/stacked_rgb_pcc.fits`) and re-runs only the downstream chain —
  the M1–M3 predecessors are never touched. The config may differ from the
  run-start config **only** in the sections consumed at-or-after the entry
  point (e.g. a HYPERMETRIC_STRETCH resume accepts changes to
  `hypermetric_stretch` and `runtime_limits`; a PCC resume additionally
  accepts `pcc` and `chroma_denoise`). Any other changed section aborts
  with `FORWARD_STAGE_CONFIG_SCOPE_MISMATCH`.

`MULTIBAND` is not a resume entry; it always re-runs with
`FORWARD_DRIZZLE`.

`reconstruction.delete_source_cache_after_run: true` deletes the
normalized-frame cache at run end and therefore disables reconstruction
resume (downstream resume from persisted outputs still works).

### Preprocessing pipeline

```bash
./tile_compile_runner preprocess \
  --config /path/to/preprocess.json \
  --runs-dir /path/to/runs
```

## CLI Scan

```bash
./tile_compile_cli scan /path/to/lights --frames-min 30
```

## Other CLI Commands

```bash
# Config handling
./tile_compile_cli get-schema                              # Print JSON schema
./tile_compile_cli dump-default-config                     # Print default config as JSON
./tile_compile_cli load-config <path>                    # Load and display config YAML
./tile_compile_cli save-config <path> [--stdin]            # Save config YAML
./tile_compile_cli validate-config (--path P | --yaml Y | --stdin)

# Run inspection
./tile_compile_cli list-runs /path/to/runs
./tile_compile_cli get-run-status /path/to/runs/<run_id>
./tile_compile_cli get-run-logs /path/to/runs/<run_id> [--tail N]
./tile_compile_cli list-artifacts /path/to/runs/<run_id>

# Input scanning
./tile_compile_cli scan /path/to/lights [--frames-min N]

# FITS analysis
./tile_compile_cli fits-stats /path/to/image.fits

# Photometric color calibration (PCC)
./tile_compile_cli pcc-run <in.fits> <out.fits> --wcs <wcs.fits> [--source vizier|siril]
./tile_compile_cli pcc-apply <in.fits> <out.fits> [--r X] [--g Y] [--b Z]

# GUI state (for external tool integration)
./tile_compile_cli load-gui-state [--path <file>]
./tile_compile_cli save-gui-state [--path <file>] [--stdin | <JSON>]
```

## Diagnostic Report

Generate an HTML quality report from a finished run via GUI3 (**Generate
Stats** in Run Monitor / Run History) or the backend endpoint:

```bash
curl -X POST http://127.0.0.1:8080/api/runs/<run_id>/stats \
  -H 'Content-Type: application/json' -d '{}'
```

Output (single self-contained HTML with inline SVG, plus a JSON summary):

- `runs/<run_id>/artifacts/report.html`
- `runs/<run_id>/artifacts/stats.json`

The report aggregates data from artifact JSON files, `logs/run_events.jsonl`, and `config.yaml`, including:

- normalization/background trends
- global quality distributions and weights
- registration drift/CC/rotation diagnostics
- reconstruction support/coverage heatmaps
- BGE diagnostics (grid cells, residuals, channel shifts)
- validation metrics
- pipeline timeline and frame-usage funnel
