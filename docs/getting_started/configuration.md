# Configuration

The pipeline is configured via `tile_compile.yaml`.

## Key Sections

- `input` — source directory, frame limits
- `registration` — alignment engine, anchor strategy, astrometry
- `method` / `aqmh` — AQMH quality maps, reconstruction, storage, Cherry-Pick
- `tile` — classic tile geometry
- `stacking` — weighting, rejection, output format
- `background_extraction` — BGE parameters
- `photometric_color_calibration` — PCC catalog, reference
- `runtime_limits` — memory, time, worker limits, acceleration backend

## GPU backend

```yaml
runtime_limits:
  acceleration_backend: auto  # auto | opencv_cuda | opencv_opencl | cpu
  parallel_workers: 8
  memory_budget: 2048
```

`auto` prefers CUDA, then OpenCL, then CPU. Backend support is phase-specific:
PREWARP, AQMH maps, classic tile reconstruction, synthetic reconstruction, and
STACKING support CUDA/OpenCL; streaming AQMH reconstruction currently supports
CUDA; REGISTRATION remains CPU-only. AQMH Cherry-Pick deliberately uses CPU.
The effective choice is written to `artifacts/acceleration_context.json` and
shown in live progress logs.

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
- `full_mode.example.yaml` — high quality equatorial
- `emergency_mode.example.yaml` — minimal runtime
- `smart_telescope_dwarf_seestar.example.yaml` — smart telescope
- `canon_equatorial_balanced.example.yaml` — balanced DSLR

## Parameter Studio (GUI3)

The web frontend provides a guided parameter editor with:

- Scenario presets (altaz, rotation, bright stars, few frames, gradient)
- Situation Assistant — automated parameter suggestions
- Real-time validation against schema
