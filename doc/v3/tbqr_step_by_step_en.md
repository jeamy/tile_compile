# TBQR Step-by-Step Guide (English)

This guide explains how to run the active C++ pipeline (`tile_compile_cpp`) from build to finished outputs.

## 1) Prerequisites

Required dependencies:

- CMake >= 3.20
- C++17 compiler (GCC 11+ or Clang 14+)
- OpenCV >= 4.5
- Eigen3
- cfitsio
- yaml-cpp
- nlohmann-json

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
- `tile_compile_gui` (Qt GUI)

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
  --from-phase PCC
```

Allowed `--from-phase` values are currently `ASTROMETRY` or `PCC`.

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

## 7) Run the GUI (optional)

```bash
./tile_compile_gui
```

## 8) GUI step-by-step workflow (English)

Use this sequence for a complete run from scan to outputs.

### Step 1: Open the GUI and load your config

1. Start `tile_compile_gui` from `tile_compile_cpp/build`.
2. Open your YAML config (or start from defaults).
3. Confirm key fields:
   - `run_dir`
   - `input.pattern`
   - calibration toggles (`use_bias`, `use_dark`, `use_flat`)

### Step 2: Scan your input frames

1. Go to the **Scan** tab.
2. Set input directory and pattern.
3. Run scan and verify:
   - detected frame count
   - OSC/MONO detection
   - Bayer pattern (for OSC)

### Step 3: Review or adjust calibration (optional)

1. If needed, enable bias/dark/flat calibration.
2. Set master files or source directories.
3. Keep disabled if your lights are already calibrated.

### Step 4: Validate configuration

1. Open the configuration tab/editor.
2. Validate config before running.
3. Fix any reported schema or value issues.

### Step 5: Start pipeline run

1. Go to the **Run** tab.
2. Set runs directory and output subfolder naming.
3. Start the run and monitor progress by phase.

### Step 6: Monitor logs and status

1. Watch live status/progress in the GUI.
2. If warnings appear, inspect run logs after completion:
   - `runs/<run_id>/logs/run_events.jsonl`
3. Use `tile_compile_cli get-run-status` for a quick status summary if needed.

### Step 7: (Optional) Astrometry and PCC

1. In **Astrometry**, solve `stacked_rgb.fits` (or your selected output) to write WCS.
2. In **PCC**, run photometric color calibration on the solved RGB result.
3. Save calibrated output (for example `stacked_rgb_pcc.fits`).

### Step 8: Verify final outputs

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
python tile_compile_cpp/generate_report.py runs/<run_id>
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
