# CLI Reference

## tile_compile_runner

Pipeline execution:

```bash
tile_compile_runner run \
  --config <path> \
  --input-dir <path> \
  --runs-dir <path>
```

Resume:

```bash
tile_compile_runner resume \
  --run-dir <path> \
  --from-phase <PHASE>
```

Options:
- `--max-frames <n>` — limit frames (0 = no limit)
- `--max-tiles <n>` — limit classic tile processing (0 = no limit)
- `--dry-run` — validate without full processing
- `--run-id <id>` — custom run ID
- `--stdin` — read config from stdin

Pipeline phases run from `SCAN_INPUT` (0) through `DONE` (18), including
`HYPERMETRIC_STRETCH` (17). Supported resume entry points depend on available
artifacts; common points are `STACKING`, `ASTROMETRY`, `BGE`, `PCC`, and
`HYPERMETRIC_STRETCH`.

GPU selection is configured in YAML, not with a runner flag:

```yaml
runtime_limits:
  acceleration_backend: auto
```

Inspect `artifacts/acceleration_context.json` or the live log fields
`cpu_workers`, `gpu`, and `backend` to confirm the effective backend.

## tile_compile_cli

Config:
- `get-schema` — print JSON schema
- `dump-default-config` — default config as JSON
- `load-config <path>` — load YAML
- `save-config <path>` — save YAML
- `validate-config` — validate

Run inspection:
- `list-runs <dir>`
- `get-run-status <dir>`
- `get-run-logs <dir> [--tail N]`
- `list-artifacts <dir>`

Input:
- `scan <dir> [--frames-min N]`

FITS:
- `fits-stats <path>`

PCC:
- `pcc-run <in.fits> <out.fits> --wcs <wcs.fits> [--source vizier|siril]`
- `pcc-apply <in.fits> <out.fits> [--r X] [--g Y] [--b Z]`

GUI state:
- `load-gui-state [--path <file>]`
- `save-gui-state [--path <file>] [--stdin | <JSON>]`
