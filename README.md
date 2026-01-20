# Tile-Compile

Tile-Compile is a toolkit for **tile-based quality reconstruction** of astronomical image stacks (Methodik v3).

Given a directory of aligned (or alignable) FITS lights, it:

- optionally **calibrates** lights (bias/dark/flat)
- **registers** frames (OpenCV ECC)
- splits channels (OSC)
- estimates **global + local (tile) quality metrics**
- reconstructs an improved per-channel image based on tile weights
- optionally clusters frame "states" and generates synthetic frames
- produces a final stacked output plus a set of **diagnostic artifacts** (PNGs/JSON)

## Versionen

Das Projekt enthält zwei unabhängige Implementierungen:

| Version | Verzeichnis | Status | Backend |
|---------|-------------|--------|---------|
| **Python** | `tile_compile_python/` | Stabil | Python + NumPy + OpenCV |
| **C++** | `tile_compile_cpp/` | In Entwicklung | C++ + Eigen + OpenCV |

Beide Versionen sind unabhängig voneinander lauffähig.

## Quickstart (Python-Version)

### System prerequisites (important)

If you use a Python that was built without certain system libraries (common with `pyenv`), you may hit errors like:

- `ModuleNotFoundError: No module named '_bz2'`
- `ModuleNotFoundError: No module named '_ctypes'`

These are **missing stdlib extension modules** and mean your Python interpreter was compiled without the required OS development packages.

#### Linux (Fedora)

Install build dependencies (covers `_bz2`, `_ctypes`, SSL, zlib, lzma, sqlite, tkinter, etc.):

```bash
sudo dnf install -y \
  gcc make \
  bzip2-devel libffi-devel \
  zlib-devel xz-devel \
  openssl-devel \
  readline-devel sqlite-devel tk-devel
```

If you use `pyenv`, rebuild your Python after installing the packages and then recreate the project venv.

#### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  libbz2-dev libffi-dev \
  zlib1g-dev liblzma-dev \
  libssl-dev \
  libreadline-dev libsqlite3-dev tk-dev
```

#### macOS (Homebrew)

```bash
brew install openssl@3 readline sqlite3 xz zlib bzip2 libffi tcl-tk
```

If you use `pyenv` on macOS, ensure `pyenv install` sees these libraries (often works automatically with Homebrew).

#### Windows

- **Recommended**: install Python from python.org (it includes `bz2` and `ctypes`).
- Use a venv and install dependencies via `pip` (see below).
- For PySide6/Qt: ensure you run on 64-bit Python.

### Python setup (venv)

#### Python package dependencies

The Python implementation consists of:

- backend/runner dependencies (scientific stack, FITS IO, registration)
- optional GUI dependency (Qt via PySide6)

At minimum you need the packages from `tile_compile_python/requirements.txt`.

Notes:

- The backend uses OpenCV (`cv2`). If your environment does not already provide it, install `opencv-python`.
- The GUI uses Qt via `PySide6`.

```bash
cd tile_compile_python

# Virtual Environment erstellen
python3 -m venv .venv
source .venv/bin/activate

# Abhängigkeiten installieren
pip install -r requirements.txt

# GUI starten
./start_gui.sh
```

### Python setup (pyenv users)

If you use `pyenv` and hit `_bz2` / `_ctypes` errors, rebuild the interpreter after installing the OS packages above:

```bash
pyenv uninstall 3.12.12
pyenv install 3.12.12
```

Then recreate the project venv (important):

```bash
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart (C++ Version - in Entwicklung)

Siehe `tile_compile_cpp/README.md` und `doc/c-port/` für den Portierungsplan.

## GUI workflow (recommended)

The GUI is organized into tabs. A typical end-to-end workflow looks like this:

### 1) Scan

- Pick an **Input dir** (your lights) and a file **pattern** (default `*.fit*`).
- Click **Scan**.
- Review what was detected (frame count, color mode hints).

### 2) Select directories (optional calibration)

If you want calibration (Bias/Darks/Flats), set it up before starting the run:

- Enable **use_bias / use_dark / use_flat**.
- Either select a **master** (`*_master`) or a **directory** (`*_dir`).
- If you provide a directory and no master is set, Tile-Compile will build a master during the run.

### 3) Configuration

- Load `tile_compile.yaml` (or edit the YAML in the GUI).
- Click **Validate config** to ensure the configuration is consistent.
- Save if you want the file updated on disk.

### 4) Assumptions

- Use the **Assumptions** tab to apply opinionated defaults / heuristics.
- Apply them to the config if desired.

### 5) Run

- Go to **Run** tab.
- Set **Working dir** (where the GUI writes its effective config) and **Runs dir**.
- Click **Start**.
- Use **Stop** to terminate a running pipeline.

### 6) Pipeline Progress

- The **Pipeline Progress** tab shows the current phase and progress.
- For phases with sub-steps, the progress bar shows the active substep (e.g. `calibrate_lights`, `cluster_frames`).

### 7) Logs + Current run + Resume

- The **Live log** tab shows the runner stdout stream.
- The **Current run** tab lets you:
  - refresh status/logs/artifacts
  - inspect the current run dir
  - **Resume from phase...** (re-run from a selected phase onward)

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

### Common PNG artifacts

The exact set depends on config and pipeline path, but typically you will find under `artifacts/`:

- per-phase preview images (inputs/registered/reconstructed)
- global and per-tile quality plots
- tile heatmaps / weight maps
- state clustering diagnostics (when enabled)

Open `artifacts/report.html` for a clickable overview that links to the generated PNG/JSON files.

For detailed explanations of the artifacts/PNGs, see `doc/stats/`.

## Tests

```bash
pytest
```

## Project structure

```
tile-compile/
├── tile_compile_python/    # Python-Implementierung (stabil)
│   ├── gui/                # PyQt6 GUI
│   ├── runner/             # Pipeline-Runner
│   ├── tile_compile_backend/  # Backend-Module
│   ├── tests/              # Unit-Tests
│   └── start_gui.sh        # GUI starten
├── tile_compile_cpp/       # C++ Implementierung (in Entwicklung)
│   ├── gui/                # PyQt6 GUI (ruft C++ Backend)
│   ├── build/              # Build-Verzeichnis
│   └── start_gui.sh        # GUI starten
├── doc/                    # Dokumentation
│   ├── c-port/             # C++ Portierungsplan
│   └── ...
└── README.md
```