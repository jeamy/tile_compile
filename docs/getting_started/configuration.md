# Configuration

The pipeline is configured via `tile_compile.yaml`.

## Key Sections

- `input` — source directory, frame limits
- `registration` — alignment engine, anchor strategy, astrometry
- `tile_reconstruction` — tile geometry, acceleration
- `stacking` — weighting, rejection, output format
- `background_extraction` — BGE parameters
- `photometric_color_calibration` — PCC catalog, reference
- `runtime_limits` — memory, time, worker limits

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

## Parameter Studio (GUI2)

The web frontend provides a guided parameter editor with:

- Scenario presets (altaz, rotation, bright stars, few frames, gradient)
- Situation Assistant — automated parameter suggestions
- Real-time validation against schema
