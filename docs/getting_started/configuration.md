# Configuration

The pipeline is configured via `tile_compile.yaml`.

## Key Sections

- `data` — input handling, frame limits
- `normalization` — background/scale estimation
- `registration` — alignment engine, anchor strategy, prewarp refinements
- `dithering` — dithering diagnostics gate
- `global_metrics` — global frame-weight computation
- `reconstruction` — the reconstruction core: `drizzle.*` (kernel, pixfrac,
  chunking, memory budget), `clipping.*`, `coverage_gate.*`, `multiband.*`,
  `quality.pyramid.*`, `diagnostics.*`
- `astrometry` — plate solving
- `bge` — background gradient extraction
- `pcc` — photometric color calibration
- `hypermetric_stretch` — final non-linear stretch
- `stacking` — per-frame cosmetic correction
- `runtime_limits` — memory, time, worker limits, acceleration backend

There is no `method` section — CFA Forward Drizzle + Multiband is the only
method. A top-level `method:` key is rejected fail-closed; legacy blocks
(`aqmh`, `pipeline`, `tile`, `local_metrics`, `synthetic`, …) are stripped
with a migration warning. `tile_compile_cli migrate-config <in> <out>`
writes a cleaned file.

## GPU backend

```yaml
runtime_limits:
  acceleration_backend: auto  # auto | opencv_cuda | opencv_opencl | cpu
  parallel_workers: 8
  memory_budget: 2048
```

`auto` prefers CUDA, then OpenCL, then CPU. FORWARD_DRIZZLE has CUDA
acceleration for the geometry/gather hot paths; the CPU path is the
bit-exactness reference and always remains available as fallback.
REGISTRATION remains CPU-only. The effective choice is written to
`artifacts/acceleration_context.json` and shown in live progress logs.

## Schema

Validate with:

```bash
./tile_compile_cli validate-config --path tile_compile.yaml
```

Get schema:

```bash
./tile_compile_cli get-schema
```

## Examples

See `tile_compile_cpp/examples/` for scenario-specific configs:

- `m104.example.yaml` — Alt/Az, strong rotation, poor seeing
- `large_n.example.yaml` / `medium_n.example.yaml` / `small_n.example.yaml` — frame-count tiers
- `mono.example.yaml` — monochrome workflow
- `smart_telescope_dwarf_seestar.example.yaml` — smart telescope
- `canon_equatorial_balanced.example.yaml` — balanced DSLR
- `reconstruction_tuning.example.yaml` — reconstruction.* tuning reference

## Parameter Studio (GUI3)

The web frontend provides a guided parameter editor with:

- Scenario presets (altaz, rotation, bright stars, few frames, gradient)
- Situation Assistant — automated parameter suggestions
- Real-time validation against schema
