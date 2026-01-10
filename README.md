# Tile-Compile

Tile-Compile is a toolkit for **tile-based quality reconstruction** of astronomical image stacks (Methodik v3).

Given a directory of aligned (or alignable) FITS lights, it:

- optionally **calibrates** lights (bias/dark/flat)
- **registers** frames (Siril or OpenCV)
- splits channels (OSC)
- estimates **global + local (tile) quality metrics**
- reconstructs an improved per-channel image based on tile weights
- optionally clusters frame “states” and generates synthetic frames
- produces a final stacked output plus a set of **diagnostic artifacts** (PNGs/JSON)

## Quickstart

### 1) Install Python dependencies

- Python 3.8+

Use your preferred environment manager. This repo contains helper scripts:

```bash
./setup_venv.sh
```

Then activate your environment:

```bash
source .venv/bin/activate
```

### 2) Run the GUI

```bash
./start_gui.sh
```

In the GUI:

- select input directory
- optionally enable calibration and select bias/darks/flats (dir or master)
- validate config, then run

### 3) Run the CLI

Example:

```bash
python3 tile_compile_runner.py run \
  --config tile_compile.yaml \
  --input-dir /path/to/lights \
  --runs-dir runs
```

There is also a convenience wrapper:

```bash
./run-cli.sh
```

## Calibration (Bias/Darks/Flats)

Calibration happens before registration.

- If you provide a **master file** (`bias_master`, `dark_master`, `flat_master`), it is used directly.
- If you provide a **directory** (`bias_dir`, `darks_dir`, `flats_dir`) and no master is set, Tile-Compile builds a master from the frames in that directory.
- Built masters are written into the run under:
  - `outputs/calibration/master_bias.fit`
  - `outputs/calibration/master_dark.fit`
  - `outputs/calibration/master_flat.fit`

## Outputs

Each run produces a self-contained directory structure (under `--runs-dir`), typically:

- `outputs/`
  - calibrated lights (if enabled)
  - registered frames
  - reconstructed outputs
  - final stack
- `artifacts/`
  - PNGs and JSON diagnostics (quality plots, tile heatmaps, previews)
- `work/`
  - intermediate work directories (e.g. Siril work dirs)

For detailed explanations of the artifacts/PNGs, see `doc/stats/`.

## Tests

```bash
pytest
```

## Project structure

- `runner/`: pipeline implementation and utilities
- `gui/`: Qt GUI (PySide6)
- `siril_scripts/`: default Siril scripts
- `tile_compile_backend/`: shared backend modules
- `tests/`: test suite
- `doc/`: documentation