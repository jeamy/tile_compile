# CLI Reference

## CLI Runner

```bash
./tile_compile_runner \
  run \
  --config ../tile_compile.yaml \
  --input-dir /path/to/lights \
  --runs-dir /path/to/runs
```

### Common options

- `--max-frames <n>` limit frames (`0` = no limit)
- `--max-tiles <n>` limit classic tile processing (`0` = no limit)
- `--dry-run` execute validation flow without full processing
- `--run-id <id>` custom run id for grouping
- `--stdin` with `--config -` to read YAML from stdin

### Resume mode

```bash
./tile_compile_runner resume \
  --run-dir /path/to/runs/<run_id> \
  --from-phase BGE
```

Supported resume phases (any phase from 0..17):

- Early: `SCAN_INPUT`, `CHANNEL_SPLIT`, `NORMALIZATION`, `GLOBAL_METRICS`, `TILE_GRID`
- Mid: `REGISTRATION`, `PREWARP`, `COMMON_OVERLAP`, `LOCAL_METRICS`, `TILE_RECONSTRUCTION`
- Late: `STATE_CLUSTERING`, `SYNTHETIC_FRAMES`, `STACKING`, `DEBAYER`, `ASTROMETRY`, `BGE`, `PCC`, `HYPERMETRIC_STRETCH`

Common resume points: `ASTROMETRY` (re-solve), `BGE` (re-extract background), `PCC` (re-calibrate color), `HYPERMETRIC_STRETCH` (rerun final VeraLux stretch), `STACKING` (re-stack from synthetic frames).

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

Generate an HTML quality report from a finished run either via GUI3 or directly through the CLI:

```bash
./tile_compile_cli generate-report runs/<run_id>
```

Output:

- `runs/<run_id>/artifacts/report.html`
- `runs/<run_id>/artifacts/report.css`
- `runs/<run_id>/artifacts/*.png`

The report aggregates data from artifact JSON files, `logs/run_events.jsonl`, and `config.yaml`, including:

- normalization/background trends
- global quality distributions and weights
- registration drift/CC/rotation diagnostics
- tile and reconstruction heatmaps
- clustering/synthetic frame summaries
- BGE diagnostics (grid cells, residuals, channel shifts)
- validation metrics (including tile-pattern indicators)
- pipeline timeline and frame-usage funnel
