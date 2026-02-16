# Tile-Compile

Tile-Compile is a toolkit for **tile-based quality reconstruction** of astronomical image stacks (methodology v3.2).

We present a novel methodology for the reconstruction of high-quality astronomical images from short-exposure deep-sky datasets. Conventional stacking methods often rely on binary frame selection ("lucky imaging"), which discards significant portions of collected frames. Our approach, **Tile-Based Quality Reconstruction (TBQR)**, replaces rigid frame selection with a robust spatio-temporal quality model. By decomposing frames into local tiles and modeling quality along two orthogonal axes—global atmospheric transparency/noise and local structural sharpness—we reconstruct a signal that is physically and statistically optimal at every pixel. We demonstrate that this method preserves the full photometric depth of the dataset while achieving superior resolution improvement compared to traditional reference stacks.

While the methodology was originally conceived to address the specific challenges of short-exposure data from modern smart telescopes (e.g., Dwarf, Seestar), its architectural flexibility makes it equally potent for conventional astronomical setups. The extensive set of tunable parameters—ranging from adaptive tile sizing and cross-correlation thresholds to sophisticated clustering logic—allows the pipeline to be meticulously optimized for a wide array of optical systems and atmospheric conditions.

> **Practical note:** The pipeline is primarily optimized for datasets with many usable frames. With very small frame counts, or with strongly mixed frame quality in one stack, visible tile patterns can occur in difficult cases. This can often be mitigated by testing different configuration settings (especially registration, tile, and reconstruction-related parameters). See the example profiles in `tile_compile_cpp/examples/` and `tile_compile_cpp/examples/README.md`.

> **Note:** This is experimental software primarily developed for processing images from smart telescopes (e.g., DWARF, Seestar, ZWO SeeStar, etc.). While designed for general astronomical image processing, it has been optimized for the specific characteristics and challenges of smart telescope data.

## Documentation (v3.2)

- Normative methodology: `doc/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.2.2_en.md`
- Implementation process flow: `doc/v3/process_flow/`
- English step-by-step guide: `doc/v3/tbqr_step_by_step_en.md`
- German README snapshot: `README_de.md`

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

### Example profiles

Complete standalone example configs are available under `tile_compile_cpp/examples/`:

- `tile_compile.full_mode.example.yaml`
- `tile_compile.reduced_mode.example.yaml`
- `tile_compile.emergency_mode.example.yaml`
- `tile_compile.smart_telescope_dwarf_seestar.example.yaml`
- `tile_compile.canon_low_n_high_quality.example.yaml`
- `tile_compile.mono_full_mode.example.yaml`
- `tile_compile.mono_small_n_anti_grid.example.yaml` (recommended for MONO low-frame datasets, e.g. ~10..40, to reduce tile-pattern risk)

See also: `tile_compile_cpp/examples/README.md`

## Quickstart (C++)

For a full beginner-friendly walkthrough, see:
`doc/v3/tbqr_step_by_step_en.md`

### Build Requirements

- CMake >= 3.20
- C++17 compiler (GCC 11+ or Clang 14+)
- OpenCV >= 4.5
- Eigen3
- cfitsio
- yaml-cpp
- nlohmann-json

#### Package install examples

Linux (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake pkg-config \
  libeigen3-dev libopencv-dev libcfitsio-dev libyaml-cpp-dev nlohmann-json3-dev libssl-dev \
  qt6-base-dev qt6-tools-dev libgl1-mesa-dev
```

macOS (Homebrew, core libs):

```bash
brew install cmake pkg-config eigen opencv cfitsio yaml-cpp nlohmann-json openssl
```

Windows:

- MinGW/MSYS2: `mingw-w64-x86_64-eigen3`, `mingw-w64-x86_64-opencv`, `mingw-w64-x86_64-cfitsio`, `mingw-w64-x86_64-yaml-cpp`, `mingw-w64-x86_64-nlohmann-json`, `mingw-w64-x86_64-openssl`, `mingw-w64-x86_64-pkgconf`
- MSVC/vcpkg: `eigen3`, `opencv`, `cfitsio`, `yaml-cpp`, `nlohmann-json`, `openssl`, `pkgconf`

### Build

```bash
cd tile_compile_cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Release build scripts (portable app bundles)

Inside `tile_compile_cpp/`, platform-specific release scripts are available:

- Linux (native): `build_linux_release.sh`
- Linux (Docker Ubuntu 20.04 / glibc 2.31): `build_linux_release_docker_ubuntu2004.sh`
- macOS: `build_macos_release.sh`
- Windows: `build_windows_release.bat`

Linux Docker wrapper (recommended for broad Linux compatibility):

```bash
cd tile_compile_cpp
bash build_linux_release_docker_ubuntu2004.sh
# optional: skip image rebuild
bash build_linux_release_docker_ubuntu2004.sh --skip-build
```

Release outputs are written to `tile_compile_cpp/dist/`:

- Linux: `dist/linux/` + `dist/tile_compile_cpp-linux-release.zip`
- Windows: `dist/windows/` + `dist/tile_compile_cpp-windows-release.zip`
- macOS: `dist/macos/tile_compile_gui.app` + optional `dist/tile_compile_cpp-macos.dmg`

Included runtime files in release bundles:

- executables (`tile_compile_gui`, `tile_compile_runner`, `tile_compile_cli`)
- GUI runtime files (`gui_cpp/constants.js`, `gui_cpp/styles.qss`)
- config + schemas (`tile_compile.yaml`, `tile_compile.schema.yaml`, `tile_compile.schema.json`)
- example profiles (`examples/`)

Not included by design:

- external Siril catalog data
- external ASTAP binary/data

macOS note:

- On older macOS versions, Homebrew `qt` may require Ventura and fail to install.
- In that case, install Qt6 via the Qt Online Installer (for example under `~/Qt/<version>/macos`) and optionally set:

```bash
export CMAKE_PREFIX_PATH="$HOME/Qt/<version>/macos"
export Qt6_DIR="$HOME/Qt/<version>/macos/lib/cmake/Qt6"
```

### Docker Build + Run (recommended for isolated environments)

A helper script is available at:
`tile_compile_cpp/scripts/docker_compile_and_run.sh`

What it does:

- `build-image`: builds a Docker image and compiles `tile_compile_cpp` inside the container
- `run-shell`: starts an interactive shell in the compiled container
- `run-app`: runs `tile_compile_runner` directly in the container

Default runs volume mapping:

- Host: `tile_compile_cpp/runs`
- Container: `/workspace/tile_compile_cpp/runs`

Examples:

```bash
# build Docker image and compile inside container
./tile_compile_cpp/scripts/docker_compile_and_run.sh build-image

# open interactive shell inside container
./tile_compile_cpp/scripts/docker_compile_and_run.sh run-shell

# run pipeline in container
./tile_compile_cpp/scripts/docker_compile_and_run.sh run-app -- run \
  --config /mnt/config/tile_compile.yaml \
  --input-dir /mnt/input \
  --runs-dir /workspace/tile_compile_cpp/runs
```

Use `run-shell` if you need additional mounts (e.g., config/input directories) and then start the runner manually.

#### Windows start notes (Docker)

Use a Linux shell (WSL2 Ubuntu) to run the helper script:

```bash
bash scripts/docker_compile_and_run.sh build-image
bash scripts/docker_compile_and_run.sh run-app -- run --config /mnt/config/tile_compile.yaml --input-dir /mnt/input --runs-dir /workspace/tile_compile_cpp/runs
```

GUI on Windows:

- Recommended: Windows 11 + WSLg, then run:

```bash
bash scripts/docker_compile_and_run.sh run-gui
```

- Without WSLg, use an X server (e.g. VcXsrv), export `DISPLAY`, and run GUI manually:

```bash
export DISPLAY=host.docker.internal:0.0
docker run --rm -it \
  -e DISPLAY=$DISPLAY \
  -e QT_QPA_PLATFORM=xcb \
  -v "$(pwd)/tile_compile_cpp/runs:/workspace/tile_compile_cpp/runs" \
  -w /workspace/tile_compile_cpp/build \
  tile_compile_cpp:dev \
  ./tile_compile_gui
```

### CLI Runner

```bash
./tile_compile_runner \
  run \
  --config ../tile_compile.yaml \
  --input-dir /path/to/lights \
  --runs-dir /path/to/runs
```

Common options:

- `--max-frames <n>` limit frames (`0` = no limit)
- `--max-tiles <n>` limit tile count for Phase 5/6 (`0` = no limit)
- `--dry-run` execute validation flow without full processing
- `--run-id <id>` custom run id for grouping
- `--stdin` with `--config -` to read YAML from stdin

Resume mode:

```bash
./tile_compile_runner resume \
  --run-dir /path/to/runs/<run_id> \
  --from-phase PCC
```

### CLI Scan

```bash
./tile_compile_cli scan /path/to/lights --frames-min 30
```

### Other CLI Possibilities

```bash
# validate config
./tile_compile_cli validate-config --path ../tile_compile.yaml

# list available runs
./tile_compile_cli list-runs /path/to/runs

# inspect one run
./tile_compile_cli get-run-status /path/to/runs/<run_id>
./tile_compile_cli get-run-logs /path/to/runs/<run_id> --tail 200
./tile_compile_cli list-artifacts /path/to/runs/<run_id>
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

## External Sources (PCC and Astrometry)

For optional color calibration and astrometric solving, the pipeline can use external data/tools:

- **Siril Gaia DR3 XP sampled catalog** (for PCC)
  - Can be reused if already downloaded by Siril.
  - Typical local path: `~/.local/share/siril/siril_cat1_healpix8_xpsamp/`
  - Upstream source (catalog release): `https://zenodo.org/records/14738271`
- **ASTAP** (for astrometry / WCS plate solving)
  - Requires ASTAP plus a star database (e.g., D50 for deep-sky use).
  - Official site/downloads: `https://www.hnsky.org/astap.htm`

If these resources are not installed, core reconstruction still works, but ASTROMETRY/PCC phases may be skipped or fail depending on configuration.

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
│       ├── tbqr_step_by_step_en.md
│       └── tile_basierte_qualitatsrekonstruktion_methodik_v_3.2.2_en.md
├── runs/
├── README.md
└── README_de.md
```

## Tests

```bash
cd tile_compile_cpp/build
ctest --output-on-failure
```