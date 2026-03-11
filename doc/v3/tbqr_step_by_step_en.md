# TBQR Step-by-Step Guide (English)

This guide explains how to run the active C++ pipeline (`tile_compile_cpp`) and how GUI2 interacts with it through the Crow/C++ backend.

**Update note (2026-03-03):**
- Resume supports `ASTROMETRY`, `BGE`, and `PCC`.
- BGE runs before PCC; BGE configuration includes user-facing robust controls (`bge.fit.robust_loss`, `bge.fit.huber_delta`).
- Methodology profile is explicit via `assumptions.pipeline_profile` (`practical` or `strict`).
- Strict-aligned registration cascade can disable Star-Pairs via `registration.enable_star_pair_fallback: false`.

## 1) Prerequisites

This repository currently consists of three relevant runtime/build parts:

- `tile_compile_cpp` for the native runner and CLI
- `web_backend_cpp` for the Crow/C++ backend
- `web_frontend` for the static HTML/CSS/JS GUI2 files

Required core build dependencies:

- CMake >= 3.21 recommended
- C++17 compiler
- Ninja recommended
- pkg-config
- Eigen3
- OpenCV >= 4.5
- cfitsio
- yaml-cpp
- nlohmann-json
- OpenSSL
- libcurl

Backend-specific build dependencies:

- Crow
- Asio

Notes:

- Crow and Asio are fetched by the backend CMake build.
- GUI2 no longer depends on Qt in the active build, runtime, or packaging path.
- The frontend itself does not require a JS build toolchain for normal use; it is shipped as static files.

Platform package examples:

- Linux: `build-essential cmake ninja-build pkg-config libeigen3-dev libopencv-dev libcfitsio-dev libyaml-cpp-dev nlohmann-json3-dev libssl-dev libcurl4-openssl-dev`
- macOS: `xcode-select --install` first, then `brew install cmake ninja pkg-config eigen cfitsio yaml-cpp nlohmann-json openssl curl` and `brew install opencv`
- Windows MSYS2 MinGW64: `mingw-w64-x86_64-toolchain mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja mingw-w64-x86_64-pkgconf mingw-w64-x86_64-eigen3 mingw-w64-x86_64-opencv mingw-w64-x86_64-cfitsio mingw-w64-x86_64-yaml-cpp mingw-w64-x86_64-nlohmann-json mingw-w64-x86_64-openssl mingw-w64-x86_64-curl mingw-w64-x86_64-ntldd`

macOS note:

- Homebrew's default `opencv` formula currently requires a newer macOS release than macOS 12. For the documented Homebrew path, macOS 13+ is therefore the practical baseline unless OpenCV is provided separately.

## 2) Build the C++ tools

From the repository root:

```bash
cd tile_compile_cpp
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

This creates (among others):

- `tile_compile_runner` (main pipeline runner)
- `tile_compile_cli` (utility CLI)

## 2b) Build the Crow/C++ backend

From the repository root:

```bash
cd web_backend_cpp
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF
cmake --build . -j$(nproc)
```

This creates:

- `tile_compile_web_backend` (Crow/C++ API and UI backend)

Notes:

- The backend links against `yaml-cpp`, `OpenSSL`, and `libcurl`.
- On Windows, the backend also needs Winsock system libraries at link time; this is already handled in the backend CMake setup.

## 2a) Build and run with Docker (optional)

If you prefer an isolated environment, use:
`tile_compile_cpp/scripts/docker_compile_and_run.sh`

Supported commands:

- `build-image`: builds Docker image and compiles `tile_compile_cpp` in the container
- `run-shell`: opens an interactive shell in the compiled container
- `run-app`: executes `tile_compile_runner` directly in the container

Default runs volume mapping:

- Host: `tile_compile_cpp/runs`
- Container: `/workspace/tile_compile_cpp/runs`

Example workflow:

```bash
# from repository root
./tile_compile_cpp/scripts/docker_compile_and_run.sh build-image

# optional interactive shell
./tile_compile_cpp/scripts/docker_compile_and_run.sh run-shell

# run pipeline directly in container
./tile_compile_cpp/scripts/docker_compile_and_run.sh run-app -- run \
  --config /mnt/config/tile_compile.yaml \
  --input-dir /mnt/input \
  --runs-dir /workspace/tile_compile_cpp/runs
```

Note: if your config/input files are outside the runs volume, use `run-shell` and start the runner manually with your custom mounts.

## 3) Choose and prepare a config

Start from one of the example profiles in `tile_compile_cpp/examples/`, for example:

- `tile_compile.full_mode.example.yaml`
- `tile_compile.smart_telescope_dwarf_seestar.example.yaml`
- `tile_compile.mono_small_n_anti_grid.example.yaml`

Copy one file and adjust at least:

- `run_dir`
- `input.pattern`
- `data.image_width` / `data.image_height` (if needed)
- `data.bayer_pattern` (for OSC/CFA datasets)
- `assumptions.pipeline_profile` (`strict` for explicit per-channel v3.3.6 alignment, `practical` to allow CFA-proxy core path)
- `registration.enable_star_pair_fallback` (`false` in strict profile)

Example:

```bash
cp ../examples/tile_compile.smart_telescope_dwarf_seestar.example.yaml ./my_config.yaml
```

## 4) Run the pipeline (runner CLI)

Basic run:

```bash
./tile_compile_runner run \
  --config ./my_config.yaml \
  --input-dir /path/to/lights \
  --runs-dir /path/to/runs
```

Useful options:

- `--max-frames N` limit input frames (`0` = no limit)
- `--max-tiles N` limit phase-5/6 tiles (`0` = no limit)
- `--dry-run` validate flow without full processing
- `--run-id <id>` force a run-id (useful to group related runs)
- `--project-root <path>` set explicit project root
- `--stdin` with `--config -` to pass YAML via stdin

## 5) Resume a finished/interrupted run

```bash
./tile_compile_runner resume \
  --run-dir /path/to/runs/<run_id> \
  --from-phase BGE
```

Allowed `--from-phase` values are currently `ASTROMETRY`, `BGE`, or `PCC`.

Notes:

- Resume is intended for the *post-run* phases that operate on run outputs (for example: solving WCS, background extraction, and photometric color calibration).
- Make sure the run directory contains the expected inputs (for example `outputs/stacked_rgb.fits` or whatever your config produces), as well as the run snapshot `config.yaml`.
- If you changed your config after the run, prefer editing `runs/<run_id>/config.yaml` (the run snapshot) so that the resumed phases use consistent settings.

## 6) Use utility CLI commands

### Scan input directory

```bash
./tile_compile_cli scan /path/to/lights --frames-min 30
```

### Validate a config file

```bash
./tile_compile_cli validate-config --path ./my_config.yaml
```

### List runs

```bash
./tile_compile_cli list-runs /path/to/runs
```

### Show status for one run

```bash
./tile_compile_cli get-run-status /path/to/runs/<run_id>
```

### Show recent logs

```bash
./tile_compile_cli get-run-logs /path/to/runs/<run_id> --tail 200
```

### List artifacts

```bash
./tile_compile_cli list-artifacts /path/to/runs/<run_id>
```

## 7) Run GUI2 (recommended UI path)

Development start from repository root:

```bash
./start_backend.sh
```

Then open:

```text
http://127.0.0.1:8080/ui/
```

Release bundles start GUI2 via:

- Linux: `start_gui2.sh`
- macOS: `start_gui2.command`
- Windows: `start_gui2.bat`

GUI2 is not a separate native processing engine. It uses the Crow/C++ backend as the UI/backend layer and delegates all scan, run, resume, astrometry, PCC, and report actions to `tile_compile_cli` and `tile_compile_runner`.

## 8) GUI2 workflow

Use this sequence for a complete run from scan to outputs.

### Step 1: Open GUI2 and inspect runtime defaults

1. Open the Dashboard at `/ui/`.
2. Verify the active runs directory, default config source, and guardrail state.
3. If needed, switch to the Wizard or Input & Scan directly.

### Step 2: Scan the input frames

1. Open **Input & Scan** or **Wizard** step 1.
2. Select one absolute input directory or a serial MONO filter queue.
3. Run scan and verify:
   - detected frame count
   - detected color mode (`OSC` / `MONO`)
   - image size
   - Bayer pattern (for OSC)

### Step 3: Review calibration inputs

1. Configure bias/dark/flat only if the dataset is not already calibrated.
2. Select directories or master files as needed.
3. Keep disabled if you want the pipeline to use already calibrated lights.

### Step 4: Adjust and validate parameters

1. Open **Parameter Studio**.
2. Edit parameters by section, use search, and inspect the Explain panel.
3. Validate the configuration before starting a run.

### Step 5: Start a run

1. Start from **Dashboard**, **Wizard**, or the dedicated run controls.
2. Select the runs directory and run name / label where applicable.
3. Start the run and switch to **Run Monitor**.

### Step 6: Monitor progress and logs

1. Use **Run Monitor** for phase state, phase progress, artifacts, and resume target selection.
2. Use **Live Log** for the streaming log output.
3. For completed or failed runs, inspect:
   - `runs/<run_id>/logs/run_events.jsonl`
   - `runs/<run_id>/artifacts/`
   - `runs/<run_id>/outputs/`

### Step 7: Resume post-run phases if needed

GUI2 supports resume flows through the backend for `ASTROMETRY`, `BGE`, and `PCC`.

CLI equivalent:

```bash
./tile_compile_runner resume \
  --run-dir /path/to/runs/<run_id> \
  --from-phase ASTROMETRY
```

### Step 8: Astrometry, BGE, and PCC

1. Use **Astrometry** to solve the RGB output and write WCS.
2. If enabled, run **BGE** before PCC.
3. Use **PCC** on the solved or BGE-corrected RGB result.
4. Generate the final calibrated output and optional stats/report assets.

### Step 9: Verify final outputs

Check run directory contents:

- `outputs/` for final FITS products
- `artifacts/` for JSON diagnostics and report assets
- `logs/run_events.jsonl` for detailed event history

## 9) Inspect outputs

A successful run creates `runs/<run_id>/` with:

- `outputs/` (stacked and reconstructed FITS files)
- `artifacts/` (JSON diagnostics, report resources)
- `logs/run_events.jsonl`
- `config.yaml` (run snapshot)

## 10) Generate the HTML diagnostic report

From repository root:

```bash
./tile_compile_cli generate-report runs/<run_id>
```

Expected output files:

- `runs/<run_id>/artifacts/report.html`
- `runs/<run_id>/artifacts/report.css`
- `runs/<run_id>/artifacts/*.png`

## 11) Common troubleshooting checks

1. Verify FITS files are linear and readable.
2. Confirm `input.pattern` and `--input-dir` point to the same dataset.
3. If too many frames are rejected in registration, review `registration.reject_*` thresholds.
4. Check `runs/<run_id>/logs/run_events.jsonl` for warnings/errors.
5. Use `tile_compile_cli get-run-status` and `list-artifacts` to verify phase outputs.
