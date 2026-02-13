# Tile-Compile

Tile-Compile is a toolkit for **tile-based quality reconstruction** of astronomical image stacks (methodology v3.2).

## Documentation (v3.2)

- Normative methodology: `doc/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.2.md`
- Implementation process flow: `doc/v3/process_flow/`
- German README snapshot: `REDME_de.md`

Given a directory of FITS lights, the pipeline can:

- optionally **calibrate** lights (bias/dark/flat)
- **register** frames using a robust fallback cascade
- compute **global and local (tile) quality metrics**
- reconstruct the image via tile-weighted overlap-add
- optionally cluster frame states and build synthetic frames
- **stack** using sigma-clip or weighted averaging
- **debayer** OSC/CFA data
- run **astrometry** (WCS)
- run **photometric color calibration** (PCC)
- write final outputs and detailed diagnostic artifacts

## Active Version

| Version | Directory | Status | Backend |
|---------|-----------|--------|---------|
| C++ | `tile_compile_cpp/` | Active (v3.2) | C++17 + Eigen + OpenCV + cfitsio + yaml-cpp |

## Pipeline Phases

| ID | Phase | Description |
|----|-------|-------------|
| 0 | SCAN_INPUT | Input discovery, mode detection, linearity check, disk-space precheck |
| 1 | REGISTRATION | Cascaded global registration + CFA-aware prewarp |
| 2 | CHANNEL_SPLIT | Metadata phase (channel model) |
| 3 | NORMALIZATION | Linear background-based normalization |
| 4 | GLOBAL_METRICS | Global frame metrics and weights |
| 5 | TILE_GRID | Adaptive tile geometry |
| 6 | LOCAL_METRICS | Local tile metrics and local weights |
| 7 | TILE_RECONSTRUCTION | Weighted overlap-add reconstruction |
| 8 | STATE_CLUSTERING | Optional state clustering |
| 9 | SYNTHETIC_FRAMES | Optional synthetic frame generation |
| 10 | STACKING | Final linear stacking |
| 11 | DEBAYER | OSC demosaic to RGB (MONO pass-through) |
| 12 | ASTROMETRY | Plate solving / WCS |
| 13 | PCC | Photometric color calibration |
| 14 | DONE | Final status (`ok` or `validation_failed`) |

Detailed phase docs: `doc/v3/process_flow/`

## Registration Cascade (Fallback Strategy)

| Stage | Method | Typical use case |
|-------|--------|------------------|
| 1 | Primary engine (`triangle_star_matching`) | Normal star-rich frames |
| 2 | Trail endpoint registration | Star trails / rotation-heavy data |
| 3 | AKAZE feature matching | General feature fallback |
| 4 | Robust phase+ECC | Clouds/nebulosity with larger transforms |
| 5 | Hybrid phase+ECC | Weak star matching cases |
| 6 | Identity fallback | Last resort (CC=0, frame retained) |

## Configuration

- Main config file: `tile_compile.yaml`
- Schemas: `tile_compile.schema.json`, `tile_compile.schema.yaml`
- Reference document: `doc/v3/configuration_reference.md`

## Quickstart (C++)

### Build Requirements

- CMake >= 3.20
- C++17 compiler (GCC 11+ or Clang 14+)
- OpenCV >= 4.5
- Eigen3
- cfitsio
- yaml-cpp
- nlohmann-json

### Build

```bash
cd tile_compile_cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### CLI Runner

```bash
./tile_compile_runner \
  run \
  --config ../tile_compile.yaml \
  --input-dir /path/to/lights \
  --runs-dir /path/to/runs
```

### CLI Scan

```bash
./tile_compile_cli scan \
  --input-dir /path/to/lights \
  --pattern "*.fits"
```

### GUI (Qt6)

```bash
./tile_compile_gui
```

## Outputs

After a successful run (`runs/<run_id>/`):

- `outputs/`
  - `stacked.fits`
  - `reconstructed_L.fit`
  - `stacked_rgb.fits` (OSC)
  - `stacked_rgb_solve.fits` / WCS artifacts
  - `stacked_rgb_pcc.fits`
  - `synthetic_*.fit` (mode-dependent)
- `artifacts/`
  - `normalization.json`
  - `global_metrics.json`
  - `tile_grid.json`
  - `global_registration.json`
  - `local_metrics.json`
  - `tile_reconstruction.json`
  - `state_clustering.json`
  - `synthetic_frames.json`
  - `validation.json`
  - `report.html`, `report.css`, `*.png`
- `logs/run_events.jsonl`
- `config.yaml` (run snapshot)

## Diagnostic Report (`tile_compile_cpp/generate_report.py`)

Generate an HTML quality report from a finished run:

```bash
python tile_compile_cpp/generate_report.py runs/<run_id>
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
- validation metrics (including tile-pattern indicators)
- pipeline timeline and frame-usage funnel

## Calibration (Bias / Dark / Flat)

- Master frames (`bias_master`, `dark_master`, `flat_master`) can be used directly
- Directory-based masters (`bias_dir`, `darks_dir`, `flats_dir`) can be built automatically
- `dark_auto_select: true` matches darks by exposure time (±5%)

## Project Structure

```text
tile_compile/
├── tile_compile_cpp/
│   ├── apps/
│   ├── include/tile_compile/
│   ├── src/
│   ├── gui_cpp/
│   ├── tests/
│   ├── generate_report.py
│   ├── tile_compile.yaml
│   ├── tile_compile.schema.json
│   └── tile_compile.schema.yaml
├── tile_compile_python/  # legacy
├── doc/
│   └── v3/
│       ├── process_flow/
│       └── tile_basierte_qualitatsrekonstruktion_methodik_v_3.2.md
├── runs/
├── README.md
└── REDME_de.md
```

## Tests

```bash
cd tile_compile_cpp/build
ctest --output-on-failure
```