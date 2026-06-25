# Tile-Compile

Tile-Compile is a toolkit for high-quality astronomical image reconstruction from short-exposure deep-sky datasets. The default reconstruction method is **AQMH (Adaptive Quality Map Harvesting)** — a per-pixel, quality-map-driven approach that replaces tile-based overlap-add stacking with a physically optimal pixel-wise weighted average.

> **Classic Tile-Compile (TBQR):** The original tile-based quality reconstruction methodology is still available and fully supported. See [Classic Tile-Compile README (EN)](README_classic_tile_compile_en.md) and the [Classic Tile-Compile README (DE)](README_classic_tile_compile_de.md). Set `aqmh.enabled: false` to revert to classic TILE_RECONSTRUCTION.

> **Note:** This is experimental software primarily developed for processing images from smart telescopes (e.g., DWARF, Seestar, ZWO SeeStar, etc.). While designed for general astronomical image processing, it has been optimized for the specific characteristics and challenges of smart telescope data.

## AQMH — Adaptive Quality Map Harvesting (Default)

AQMH is the default reconstruction path as of v0.3.0. For each input frame a **dense quality map** `Q_map_{f,c}(x,y)` is computed using a **4-scale Laplacian pyramid**, combining sharpness and SNR metrics into a per-pixel quality value. The final image is reconstructed as a **per-pixel weighted mean** — effective weight `W = G_{f,c} * Q_map_{f,c}(x,y)`, where `G_{f,c}` is the global frame weight from shared preprocessing. No tile grid, no OLA seams.

> **Normative specification:** [AQMH Methodology v0.1.0](docs/AQMH/aqmh_methodik_en.md)

### How it works

```
For each frame f, channel c:
  For each pyramid scale s (D_s = 4^s, window R_s = 4 px in downscaled pixels):
    1. Downsample I_{f,c} by D_s (mask-aware area average)
    2. Compute per-window:
         Phi_sharp = local variance of masked Laplacian (sharpness)
         Phi_snr   = local SNR = mu / max(1.4826*MAD, eps)
         Phi_artifact = 1 - clip(outlier_frac / frac_artifact_max, 0, 1)
    3. Psi_s = sigmoid(w_sharp*z(Phi_sharp) + w_snr*z(Phi_snr)) * Phi_artifact
       (z = robust z-score; artifact gate is multiplicative — one bad scale vetos pixel)
    4. Upsample Psi_s to canvas resolution (mask-aware bilinear)
  Q_map_{f,c} = geometric_mean over scales(Psi_s)  # all scales must agree
  Store Q_map to disk cache (default: 1/4-area float32)

Reconstruction (per canvas-valid pixel p):
  W_{f,c}(p) = G_{f,c} * Q_map_{f,c}(p)
  R_c(p) = sum_f( W_{f,c}(p) * I_{f,c}(p) ) / sum_f( W_{f,c}(p) )
```

### Key parameters (`aqmh.*`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `aqmh.enabled` | `true` | Enable AQMH (false = use classic TILE_RECONSTRUCTION) |
| `aqmh.pyramid.scales` | `4` | Pyramid levels for multi-scale analysis |
| `aqmh.pyramid.base_window_px` | `4` | Window size at lowest pyramid level |
| `aqmh.pyramid.w_sharp` | `0.6` | Sharpness weight in quality index |
| `aqmh.pyramid.w_snr` | `0.4` | SNR weight in quality index |
| `aqmh.pyramid.k_artifact` | `3.0` | MAD multiplier for artifact detection (higher = more tolerant) |
| `aqmh.pyramid.frac_artifact_max` | `0.25` | Max artifact fraction per window before discard |
| `aqmh.storage.resolution_divisor` | `2` | Quality map cache resolution (1/2/4) |
| `aqmh.storage.dtype` | `float32` | Cache data type (`float32` or `uint8`) |
| `aqmh.storage.max_resident_maps` | `2` | Max quality maps in RAM simultaneously |
| `aqmh.cherry_pick.enabled` | `false` | Stack only top-quality frames |
| `aqmh.cherry_pick.k_frac` | `0.30` | Fraction of best frames to use (0.30 = best 30%) |
| `aqmh.cherry_pick.k_min` | `3` | Minimum frames always included |
| `aqmh.diagnostics.tau_artifact` | `0.20` | Artifact threshold for `artifacts/aqmh.json` |
| `aqmh.diagnostics.q_region` | `0.75` | Quantile for regional quality statistics |
| `aqmh.diagnostics.r_morph_canvas_px` | `6` | Morphological radius for diagnostic quality map |

Full parameter documentation: [Configuration Reference — §12b AQMH](docs/configuration_reference_en.md)  
Practical examples: [Configuration Examples — AQMH section](docs/configuration_examples_practical_en.md)  
Normative specification: [AQMH Methodology v0.1.0](docs/AQMH/aqmh_methodik_en.md)

### When to use AQMH vs. Classic

| Situation | Recommendation |
|-----------|----------------|
| Default / most sessions | **AQMH** (enabled by default) |
| Tile seams or OLA artifacts visible | **AQMH** eliminates seams entirely |
| Strongly varying frame quality (seeing, clouds) | **AQMH** with `cherry_pick.enabled: true` |
| Very large sessions, RAM-limited | **AQMH** with `storage.resolution_divisor: 4`, `dtype: uint8` |
| Sessions with satellite trails / cosmetic issues | **AQMH** with `k_artifact: 5.0`, `frac_artifact_max: 0.35` |
| Research requiring TBQR tile-weighted OLA | Classic (`aqmh.enabled: false`) |

### Minimal AQMH config

```yaml
aqmh:
  enabled: true          # default — can be omitted
  pyramid:
    k_artifact: 3.0      # default
    frac_artifact_max: 0.25  # default
```

### Disable AQMH (revert to classic)

```yaml
aqmh:
  enabled: false
```

## Documentation

- **AQMH methodology (normative):** [AQMH Methodology v0.1.0](docs/AQMH/aqmh_methodik_en.md)
- **AQMH parameter reference:** [Configuration Reference — §12b AQMH](docs/configuration_reference_en.md)
- **AQMH practical examples:** [Configuration Examples & Best Practices](docs/configuration_examples_practical_en.md)
- Configuration reference (full): [Configuration Reference (EN)](docs/configuration_reference_en.md)
- German README: [README_de.md](README_de.md)
- GUI3 packaging and launch notes: [GUI3 README](packaging/gui3/README.md)
- Data flow (user-friendly): [Process Flow - How the System Works](docs/process_flow/data_flow_user_description_en.md)
- Full documentation site: [https://jeamy.github.io/tile_compile/](https://jeamy.github.io/tile_compile/)
- **GUI3 User Guide (step-by-step):** [docs/gui3_user_guide_en.md](docs/gui3_user_guide_en.md)
- Raw Stack GUI guide (English): [docs/raw_stack_gui_en.md](docs/raw_stack_gui_en.md)
- Step-by-step guide (Classic/Developer): [Step-by-Step Guide](docs/tbqr_step_by_step_en.md)
- **PI – AI-assisted recommendations:** [docs/PI/pi_ai_recommendations_en.md](docs/PI/pi_ai_recommendations_en.md)

### Classic Tile-Compile (TBQR) documentation

- Classic README (EN): [README_classic_tile_compile_en.md](README_classic_tile_compile_en.md)
- Classic README (DE): [README_classic_tile_compile_de.md](README_classic_tile_compile_de.md)
- Normative TBQR methodology: [Tile-Based Quality Reconstruction Methodology v3.3.9](docs/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.9_en.md)
- Methodology paper PDF v3.3.6: [paper-tile_based_quality_reconstruction_methodology_v_3.3.6_en.pdf](docs/v3/paper-tile_based_quality_reconstruction_methodology_v_3.3.6_en.pdf)
- Implementation process flow: [Process flow (English)](docs/process_flow/README_en.md)

## Typical Workflow (GUI3)

The standard user workflow with GUI3 involves three steps:

1. **Scan input** — Tab *Processing → Input & Scan*: Select input folder with FITS lights, optionally specify calibration frames (bias/dark/flat), start scan. The scan detects frames, resolution, and color mode.
2. **Adjust parameters** — Tab *Processing → Parameter*: Load an example config or adjust values. Key parameters: registration (rotation, transform model), AQMH (cherry-pick, pyramid scales), stacking method, Bayer pattern. Validate and save configuration.
3. **Start and monitor run** — Tab *Processing → Run Monitor*: Start run, track phase progress in real time, abort or resume from a specific phase as needed.

After completion: Results are in `runs/<run_id>/outputs/`. Generate a diagnostic report via *Generate Stats* in the Run Monitor or via Run History.

Full guide: [GUI3 User Guide](docs/gui3_user_guide_en.md)

## Paper Example Data Sources

- M31 lights source for the paper example run (10 GB): [M31 lights](https://wolke.eibrain.org/index.php/s/Z88dmWizEJYjwBe) 
- M31 run source for the paper example run (20 GB): [M31 run](https://wolke.eibrain.org/index.php/s/tfSycSNEzdL7jje)

Given a directory of FITS lights, the pipeline can:

- optionally **calibrate** lights (bias/dark/flat)
- **register** frames using a robust 6-stage fallback cascade
- compute **global quality metrics** (transparency, SNR, weights)
- compute **per-frame AQMH quality maps** (sharpness + SNR, Laplacian pyramid)
- **reconstruct** the image via per-pixel AQMH weighted average (default) or tile-weighted OLA (classic)
- optionally **cherry-pick** only the best frames for AQMH reconstruction
- optionally cluster frame states and build synthetic frames
- **stack** using sigma-clip or weighted averaging
- **debayer** OSC/CFA data
- run **astrometry** (WCS)
- run optional **background gradient extraction** (BGE, pre-PCC)
- run **photometric color calibration** (PCC)
- write final outputs and detailed diagnostic artifacts (including `artifacts/aqmh.json`)

## Active Components

| Component | Directory | Status | Stack |
|-----------|-----------|--------|-------|
| Core pipeline | `tile_compile_cpp/` | Active | C++17 + Eigen + OpenCV + cfitsio + yaml-cpp |
| GUI3 backend | `web_backend_cpp/` | Active | Crow + C++17 |
| GUI3 frontend | `web_frontend_v3/` | Active | HTML + CSS + JavaScript (ESM) |

## Pipeline Phases

In practical use, the overall workflow is intentionally simple: provide the input data and a manageable set of configuration parameters; the pipeline processes the dataset automatically from AQMH reconstruction through astrometry, optional background handling, and PCC to the final result.

| ID | Phase | Description |
|----|-------|-------------|
| 0 | SCAN_INPUT | Input discovery, mode detection, linearity check, disk-space precheck |
| 1 | REGISTRATION | Cascaded global registration |
| 2 | PREWARP | Full-frame canvas prewarp (CFA-safe for OSC) |
| 3 | CHANNEL_SPLIT | Metadata phase (channel model) |
| 4 | NORMALIZATION | Linear background-based normalization |
| 5 | GLOBAL_METRICS | Global frame metrics and weights |
| 6 | TILE_GRID | Adaptive tile geometry (used by classic TILE_RECONSTRUCTION) |
| 7 | COMMON_OVERLAP | Common valid-data overlap (global/tile-local masks) |
| 8 | LOCAL_METRICS | Local tile metrics + **AQMH quality map computation** |
| 9 | TILE_RECONSTRUCTION | **AQMH per-pixel weighted reconstruction** (default) or tile-weighted OLA (classic) |
| 10 | STATE_CLUSTERING | Optional state clustering |
| 11 | SYNTHETIC_FRAMES | Optional synthetic frame generation |
| 12 | STACKING | Final linear stacking |
| 13 | DEBAYER | OSC demosaic to RGB (MONO pass-through) |
| 14 | ASTROMETRY | Plate solving / WCS |
| 15 | BGE | Optional RGB background gradient extraction before PCC |
| 16 | PCC | Photometric color calibration |
| 17 | HYPERMETRIC_STRETCH | Optional VeraLux HyperMetric Stretch after PCC |
| 18 | DONE | Final status (`ok` or `validation_failed`) |

Detailed phase docs: `docs/process_flow/`

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
- Reference document: [Configuration Reference](docs/configuration_reference_en.md)
- Practical examples: [Configuration Examples & Best Practices](docs/configuration_examples_practical_en.md)

### Example profiles

Complete standalone example configs are available under `tile_compile_cpp/examples/`.
The filenames no longer use the old `tile_compile.` prefix.

- `full_mode.example.yaml`
- `reduced_mode.example.yaml`
- `emergency_mode.example.yaml`
- `smart_telescope_dwarf_seestar.example.yaml`
- `smart_telescope_very_bright_star.example.yaml`
- `canon_low_n_high_quality.example.yaml`
- `very_bright_star_anti_seam.example.yaml`
- `canon_equatorial_balanced.example.yaml`
- `mono_full_mode.example.yaml`
- `mono_small_n_anti_grid.example.yaml` (recommended for MONO low-frame datasets, e.g. ~10..40, to reduce tile-pattern risk)
- `mono_small_n_ultra_conservative.example.yaml` (recommended for very small MONO datasets, e.g. ~8..25, when seam stability matters more than aggressive enhancement)

See also: [Examples README](tile_compile_cpp/examples/README.md) for the intended use case and tuning focus of each profile.

## Binary Releases (GUI3)

Pre-built GUI3 release bundles are published via [GitHub Releases](https://github.com/jeamy/tile_compile/releases).

Each bundle contains:

- GUI3 frontend (`web_frontend_v3/`)
- Crow backend (`web_backend_cpp/`)
- native C++ tools (`tile_compile_runner`, `tile_compile_cli`, `tile_compile_web_backend`)
- launchers for Linux, macOS, and Windows
- optional PI AI sidecar (`agent_service/`, requires Node.js >= 20)

At runtime, GUI3 uses the local Crow/C++ backend as the process adapter for the C++ runner/CLI.

## Quickstart

### GUI3 (recommended)

Development start from repository root:

```bash
./start_backend.sh
```

Then open:

```text
http://127.0.0.1:8080/ui/
```

Release bundle start:

- Linux: `start_gui3.sh`
- macOS: `start_gui3.command`
- Windows: `start_gui3.bat`

The launcher copies the bundled payload into a per-user install directory, starts the Crow backend in the foreground, and opens the browser to the local GUI3 URL.

**Installation and update behavior:**

- On first start, the launcher copies all application files to `~/tilecompile/` (Linux/macOS) or `%USERPROFILE%\tilecompile\` (Windows).
- After the first successful start, you can safely delete the downloaded package archive and extracted folder—all data has been copied to your user directory.
- On updates, only the application files (`web_frontend_v3/`, `web_backend_cpp/`, `tile_compile_cpp/`, `agent_service/`) are replaced. Your user data (configurations, runs, ASTAP catalog, PCC database) remains untouched.

macOS install note:

- On macOS 15.x (including Sequoia 15.1), Gatekeeper may no longer offer the older right-click override path for unknown developers. If `start_gui3.command` or other scripts are blocked, open `System Settings -> Privacy & Security`, scroll to the bottom, and explicitly allow the blocked `start_gui3.command` there before starting it again.

Minimum OS versions for the current GUI3 release bundles:

- Linux: x86_64 Linux with `glibc >= 2.39` (Ubuntu 24.04 or equivalent is the safe baseline for the current CI-built ZIPs)
- macOS: macOS 15
- Windows: Windows 10 x64 or newer

Notes:

- macOS release bundles are built with an explicit deployment target and are intended to run from macOS 13 upward.
- Linux bundles do not bundle `glibc`, so older distributions than the current build baseline are not guaranteed to work.
- The optional PI AI sidecar (`agent_service/`) requires **Node.js >= 20**. If Node.js is not installed or too old, the backend starts without the AI sidecar and prints a warning. See [GUI3 README](packaging/gui3/README.md) for details.

### C++ CLI / runner

For a full beginner-friendly walkthrough, see:
[Step-by-Step Guide](docs/tbqr_step_by_step_en.md)

### Build Requirements

- CMake >= 3.21
- C++17 compiler (GCC 11+ or Clang 14+)
- OpenCV >= 4.5
- Eigen3
- cfitsio
- yaml-cpp
- nlohmann-json
- Node.js >= 20 (only for optional PI AI sidecar / `agent_service/`)

#### GPU acceleration requirements

The pipeline supports two GPU backends:

**NVIDIA CUDA (opencv_cuda):**
- Requires OpenCV CUDA modules:
  - `opencv2/core/cuda.hpp`
  - `opencv2/cudawarping.hpp`
  - `opencv2/cudaarithm.hpp`
- At runtime, a CUDA-capable NVIDIA GPU and working CUDA/OpenCV runtime are required.
- `TILE_COMPILE_ENABLE_CUDA` only enables the CUDA hook/build gate.

**AMD/Intel/NVIDIA OpenCL (opencv_opencl):**
- Requires OpenCV OpenCL module:
  - `opencv2/core/ocl.hpp`
- At runtime, an OpenCL-capable GPU (AMD, Intel, NVIDIA) and working OpenCL runtime are required.
- Works with AMD Radeon (Polaris/Vega/RDNA), Intel integrated GPUs, and NVIDIA GPUs.
- Generally easier to set up than CUDA on non-NVIDIA hardware.

**Auto-selection:**
- `acceleration_backend: auto` (default) automatically detects available GPU backends at runtime.
- Priority order: CUDA → OpenCL → CPU
- Falls back gracefully to CPU if no GPU backend is available.

Notes:

- Many default distro/Homebrew/OpenCV packages provide CPU-only builds. GPU acceleration requires OpenCV built with CUDA or OpenCL support.
- For NVIDIA GPUs: CUDA backend typically provides better performance than OpenCL.
- For AMD/Intel GPUs: OpenCL is the only supported GPU backend.
- On macOS: OpenCL support depends on OpenCV build; CUDA is not practical.

#### Package install examples

Linux (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake pkg-config \
  libeigen3-dev libopencv-dev libcfitsio-dev libyaml-cpp-dev nlohmann-json3-dev libssl-dev \
  libcurl4-openssl-dev
```

Linux (Fedora):

```bash
sudo dnf install -y \
  gcc-c++ cmake pkgconf-pkg-config ninja-build \
  eigen3-devel opencv-devel cfitsio-devel yaml-cpp-devel nlohmann-json-devel openssl-devel \
  libcurl-devel
```

macOS (Homebrew, core libs):

```bash
xcode-select --install
brew install cmake ninja pkg-config eigen cfitsio yaml-cpp nlohmann-json openssl curl
brew install opencv
```

Notes:

- `ninja` is required for the local GUI3 packaging scripts.
- On macOS 12, the default Homebrew `opencv` formula is currently not supported. The Homebrew-based path therefore effectively requires macOS 15 for OpenCV, unless you provide a separate working OpenCV installation yourself.
- The package examples above are sufficient for CPU builds. They do not guarantee GPU acceleration, because the OpenCV package on the host may not include CUDA modules.
- If a downloaded GUI3/release bundle is blocked by Gatekeeper with messages such as “developer cannot be identified” or a bundled `.dylib` cannot be opened, remove the quarantine flag from the extracted release folder with `xattr -dr com.apple.quarantine /path/to/extracted_release` and then start the bundle again.

Windows:

- MinGW/MSYS2: `mingw-w64-x86_64-eigen3`, `mingw-w64-x86_64-opencv`, `mingw-w64-x86_64-cfitsio`, `mingw-w64-x86_64-yaml-cpp`, `mingw-w64-x86_64-nlohmann-json`, `mingw-w64-x86_64-openssl`, `mingw-w64-x86_64-curl`, `mingw-w64-x86_64-pkgconf`
- MSVC/vcpkg: `eigen3`, `opencv`, `cfitsio`, `yaml-cpp`, `nlohmann-json`, `openssl`, `curl`, `pkgconf`

### Build

```bash
cd tile_compile_cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Release build + packaging

GUI3 release bundles are built by:

- `.github/workflows/release-tile-compile-gui3.yml`

The workflow builds the Qt-free C++ binaries, bundles `web_backend_cpp/` and `web_frontend_v3/`, adds the GUI3 launchers, and creates ZIP artifacts for Linux, Windows, macOS Apple Silicon, and macOS Intel.

Not included by design:

- external Siril catalog data
- external ASTAP binary/data
  
### Docker Build + Run (recommended for isolated environments)

The Docker image bundles the C++ backend, runner, CLI, web frontend (v3), and the PI AI sidecar (Node.js 22 LTS).
The entrypoint script (`docker/ubuntu24.04/entrypoint.sh`) starts the sidecar and backend automatically.

#### Quick start

**Linux / macOS:**
```bash
./start_gui3_docker.sh
```
Open: http://127.0.0.1:8080/ui/

**Windows:**
```cmd
start_gui3_docker.bat
```

#### Input data and run output

Host directories are mounted into the container:

| Mount | Host (default) | Container path | Purpose |
|-------|----------------|----------------|---------|
| Input | `./tmp/docker-input` | `/data/input` | FITS light frames |
| Runs | `./tmp/docker-runs` | `/data/runs` | Run output, artifacts, logs |
| Extra | `--extra-root <path>` | `/data/extra` | Additional allowed root |

Place your FITS files in `./tmp/docker-input/` (or specify a custom path with `--input-dir`).
In the GUI, enter `/data/input` as the input directory and `/data/runs` as the runs directory.

```bash
# Custom input directory
./start_gui3_docker.sh --input-dir /path/to/my/fits --runs-dir /path/to/runs
```

#### Options

```bash
./start_gui3_docker.sh --help
```

| Option | Description | Default |
|--------|-------------|---------|
| `--image-tag <tag>` | Docker image tag | `tile-compile-web-backend:ubuntu24.04` |
| `--name <name>` | Container name | `tile-compile-web-backend` |
| `--port <port>` | Host port → container 8080 | `8080` |
| `--input-dir <path>` | Host input data mount | `./tmp/docker-input` |
| `--runs-dir <path>` | Host runs output mount | `./tmp/docker-runs` |
| `--env-file <path>` | `.env` file with API keys (mounted read-only) | `./.env` |
| `--no-agent` | Disable PI AI sidecar in container | (enabled by default) |
| `--no-build` | Skip `docker build` | (build by default) |
| `--extra-root <path>` | Additional allowed root at `/data/extra` | — |

#### `.env` file

The `.env` file provides API keys, Docker mount paths, and configuration for the PI AI sidecar.
Copy `.env.example` to `.env` in the project root and fill in your settings:

```bash
cp .env.example .env
# Edit .env – set INPUT_DIR, RUNS_DIR, AI_SCAN_ENABLED, model, API keys, etc.
```

The file is mounted read-only into the container at `/opt/tile_compile/.env`.
The sidecar loads it automatically via `dotenv`.

Docker-relevant settings in `.env`:

| Variable | Description | Default if unset |
|----------|-------------|------------------|
| `INPUT_DIR` | Host directory with FITS light frames (mounted to `/data/input`) | `./tmp/docker-input` |
| `RUNS_DIR` | Host directory for run output (mounted to `/data/runs`) | `./tmp/docker-runs` |
| `HOST_PORT` | Host port mapped to container 8080 | `8080` |
| `IMAGE_TAG` | Docker image tag | `tile-compile-web-backend:ubuntu24.04` |
| `CONTAINER_NAME` | Container name | `tile-compile-web-backend` |
| `EXTRA_ALLOWED_ROOTS` | Additional host path mounted at `/data/extra` | — |

CLI arguments (`--input-dir`, `--runs-dir`, etc.) override `.env` values.

#### Container architecture

```
entrypoint.sh
  ├── PI AI sidecar (node dist/server.js)  →  127.0.0.1:3001
  └── C++ backend (tile_compile_web_backend)  →  0.0.0.0:8080
        └── connects to sidecar at http://127.0.0.1:3001
```

#### Advanced: CLI-only Docker (legacy)

For running just the C++ runner inside a container (without backend/UI), the legacy script is still available:

```bash
./tile_compile_cpp/scripts/docker_compile_and_run.sh build-image
./tile_compile_cpp/scripts/docker_compile_and_run.sh run-app -- run \
  --config /mnt/config/tile_compile.yaml \
  --input-dir /mnt/input \
  --runs-dir /workspace/tile_compile_cpp/runs
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
  --from-phase BGE
```

Supported resume phases (any phase from 0..17):
- Early: `SCAN_INPUT`, `CHANNEL_SPLIT`, `NORMALIZATION`, `GLOBAL_METRICS`, `TILE_GRID`
- Mid: `REGISTRATION`, `PREWARP`, `COMMON_OVERLAP`, `LOCAL_METRICS`, `TILE_RECONSTRUCTION`
- Late: `STATE_CLUSTERING`, `SYNTHETIC_FRAMES`, `STACKING`, `DEBAYER`, `ASTROMETRY`, `BGE`, `PCC`, `HYPERMETRIC_STRETCH`

Common resume points: `ASTROMETRY` (re-solve), `BGE` (re-extract background), `PCC` (re-calibrate color), `HYPERMETRIC_STRETCH` (rerun final VeraLux stretch), `STACKING` (re-stack from synthetic frames).

### CLI Scan

```bash
./tile_compile_cli scan /path/to/lights --frames-min 30
```

### Other CLI Possibilities

```bash
# Config handling
./tile_compile_cli get-schema                              # Print JSON schema
./tile_compile_cli dump-default-config                     # Print default config as JSON
./tile_compile_cli load-config <path>                    # Load and display config YAML
./tile_compile_cli save-config <path> [--stdin]            # Save config YAML
./tile_compile_cli validate-config (--path P | --yaml Y | --stdin)

# Run inspection
./tile_compile_cli list-runs /path/to/runs
./tile_compile_cli get-run-status /path/to/runs/<run_id>
./tile_compile_cli get-run-logs /path/to/runs/<run_id> [--tail N]
./tile_compile_cli list-artifacts /path/to/runs/<run_id>

# Input scanning
./tile_compile_cli scan /path/to/lights [--frames-min N]

# FITS analysis
./tile_compile_cli fits-stats /path/to/image.fits

# Photometric color calibration (PCC)
./tile_compile_cli pcc-run <in.fits> <out.fits> --wcs <wcs.fits> [--source vizier|siril]
./tile_compile_cli pcc-apply <in.fits> <out.fits> [--r X] [--g Y] [--b Z]

# GUI state (for external tool integration)
./tile_compile_cli load-gui-state [--path <file>]
./tile_compile_cli save-gui-state [--path <file>] [--stdin | <JSON>]
```

### GUI3 integration

The recommended UI path is the web-based GUI3:

- backend: `web_backend_cpp/`
- frontend: `web_frontend_v3/`
- orchestration: Crow backend -> `tile_compile_cli` / `tile_compile_runner`

Development start:

```bash
./start_backend.sh
```

Open `http://127.0.0.1:8080/ui/`.

## Outputs

After a successful run (`runs/<run_id>/`):

- `outputs/`
  - `stacked.fits`
  - `reconstructed_L.fit`
  - `stacked_rgb.fits` (OSC)
  - `stacked_rgb_solve.fits` / WCS artifacts
  - `stacked_rgb_bge.fits` (BGE-only snapshot before PCC)
  - `stacked_rgb_pcc.fits`
  - `stacked_rgb_hms.fits` (optional VeraLux HyperMetric Stretch output)
  - `synthetic_*.fit` (mode-dependent)
- `artifacts/`
  - `normalization.json`
  - `global_metrics.json`
  - `tile_grid.json`
  - `global_registration.json`
  - `local_metrics.json`
  - `tile_reconstruction.json`
  - `aqmh.json` (AQMH quality diagnostics — artifact fractions, regional quality stats)
  - `state_clustering.json`
  - `synthetic_frames.json`
  - `bge.json`
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
  - **Download via GUI3**: Tab *Tools → PCC → Download Missing* automatically downloads missing catalog chunks (~2 GB, 48 chunks).
- **ASTAP** (for astrometry / WCS plate solving)
  - Requires ASTAP plus a star database (e.g., D50 for deep-sky use).
  - Official site/downloads: `https://www.hnsky.org/astap.htm`
  - **Download via GUI3**: Tab *Tools → Astrometry → Install CLI* and *Download Catalog* download ASTAP binary and star database directly.

If these resources are not installed, core reconstruction still works, but ASTROMETRY/PCC phases may be skipped or fail depending on configuration. BGE (Background Gradient Extraction) works independently of external catalogs.

## Diagnostic Report (`report.html` via C++ backend)

Generate an HTML quality report from a finished run either via GUI3 or directly through the CLI:

```bash
./tile_compile_cli generate-report runs/<run_id>
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
- BGE diagnostics (grid cells, residuals, channel shifts)
- validation metrics (including tile-pattern indicators)
- pipeline timeline and frame-usage funnel

## Calibration (Bias / Dark / Flat)

- Master frames (`bias_master`, `dark_master`, `flat_master`) can be used directly
- Directory-based masters (`bias_dir`, `darks_dir`, `flats_dir`) can be built automatically
- When `use_bias: true` and `use_dark: true`, raw darks are bias-corrected internally unless `dark_already_bias_corrected: true` is set
- `dark_auto_select: true` matches darks by exposure time (±5%)

## Project Structure

```text
tile_compile/
├── web_frontend_v3/        # GUI3 HTML/CSS/JS frontend
├── web_backend_cpp/        # GUI3 Crow/C++ backend
├── tile_compile_cpp/
│   ├── apps/                # runner/cli entry points
│   ├── include/tile_compile/
│   ├── src/
│   ├── examples/            # example configs
│   ├── scripts/             # helper scripts
│   ├── tests/
│   ├── tile_compile.yaml
│   ├── tile_compile.schema.json
│   └── tile_compile.schema.yaml
├── packaging/gui3/          # GUI3 release launchers/bundle helpers
├── docker/                  # Docker build/runtime images
├── docs/
│   ├── v3/                  # methodology docs
│   └── process_flow/        # implementation process-flow docs
├── start_backend.sh         # dev start for Crow backend + GUI3
├── README.md
└── README_de.md
```

## Tests

```bash
cd tile_compile_cpp/build
ctest --output-on-failure
```

## Attribution

This project was built with assistance from Windsurf, Kiro, Antigravity, GPT 5.*,Claude 4.* Sonnet, Codex, ***. Babysitting by a human in a virtual environment.

The PI (Parameter Intelligence) module uses the following packages:

- **[@earendil-works/pi-coding-agent](https://github.com/earendil-works/pi/tree/main/packages/coding-agent)** — AI agent framework for parameter analysis and recommendations (v0.79.x)

The HyperMetric Stretch (HMS) phase is derived from the VeraLux HyperMetric Stretch Siril script:

- (c) 2025 Riccardo Paterniti
- VeraLux - HyperMetric Stretch
- SPDX-License-Identifier: GPL-3.0-or-later
- Version 1.5.2
- Inspired by the "True Color" methodology of Dr. Roger N. Clark
- Math basis: Inverse Hyperbolic Stretch (IHS) and Vector Color Preservation
- Sensor science: hardware-specific Quantum Efficiency weighting


## Versions

## v0.3.5 (2026-06-22)

**GUI v3 – New Web Frontend:**
- New web frontend (v3) with improved Parameter Studio, Run Monitor, and Input & Scan tab.
- "Save As" dialog with directory browser: save file to selectable directory and filename via modal dialog.
- Warning banner in Run Monitor prominently displays warnings and errors during runs.
- Calibration gain mismatch: Dark/Flat calibration with mismatched gain now produces a warning instead of aborting.

**Astrometry Fix:**
- YAML `~` (null) for `astap_bin` was incorrectly parsed as string "null", causing ASTAP not to be found. Fix: null values are correctly treated as empty string, allowing the default path to be used.

## v0.3.4 (2026-06-16)
- Bug fixes

## v0.3.3 (2026-06-16)

**PI – AI-assisted configuration recommendations:**
- New PI (Parameter Intelligence) module: AI analyses scan results and generates validated parameter recommendations directly in Parameter Studio.
- Frame quality metrics (FWHM, noise, background, roundness, star count) from `scan-metrics` are passed to the AI as measured facts.
- See [docs/PI/pi_ai_recommendations_en.md](docs/PI/pi_ai_recommendations_en.md) for full documentation.

## v0.3.2 (2026-06-13)

**AQMH-First as default:**
- Top-level `method` field as single source of truth for reconstruction method (AQMH or Classic Tile Compile).
- AQMH is now the default with no migration of existing runs. Missing `method` field automatically defaults to `aqmh`.
- Frontend shows AQMH as pre-selected method in Wizard, Dashboard, and Parameter Studio.
- Classic phases (LOCAL_METRICS, STATE_CLUSTERING, SYNTHETIC_FRAMES) are hidden for AQMH runs.
- Rollback mechanisms: `FORCE_CLASSIC=1` environment variable or `--force-classic` CLI flag.

## v0.3.0 (2026-06-09)

**AQMH as default reconstruction method:**
- AQMH (Adaptive Quality Map Harvesting) is now the default reconstruction path (`aqmh.enabled: true`).
- Classic TILE_RECONSTRUCTION is still available via `aqmh.enabled: false`.
- All example profiles updated with `aqmh:` block.
- Full AQMH parameter documentation added to configuration reference and practical examples.
- `k_artifact` default changed to `3.0`, `frac_artifact_max` default changed to `0.25`.

## v0.2.A (2026-05-26)
- Calibration Bug fixes

## v0.2.9 (2026-05-25)

**Raw Stack preprocessing pipeline:**
- New standalone Raw Stack UI: preprocessing from FITS light frames to final stacked image, separate from the Tile-Compile run studio.
- Pipeline covers: Calibration, CFA/Mono Prep, Registration, Quality Filtering, Stacking, Astrometry, BGE, PCC, HyperMetric Stretch.
- All parameters (sigma-clip, rejection, weighting, BGE, PCC, Astrometry, HMS) are taken from the Parameter Studio config — nothing is hardcoded.
- See [docs/raw_stack_gui_en.md](docs/raw_stack_gui_en.md) for GUI documentation.

## v0.2.8 (2026-05-23)

-- HMS Bug fixes

## v0.2.7 (2026-05-22)

**implementationHMS:**
- Added VeraLux HyperMetric Stretch (HMS) as a post-PCC pipeline phase.

## v0.2.6 (2026-05-20)

**Build hardening & Frontend cleanup:**
- Hardened web_backend_cpp build with CUDA 13 + OpenCV 4.11 CUDA 13 configuration
- Frontend refactoring: centralized utilities in `src/utils.js` (escapeHtml, getMessage, getStorageJson, humanizeControlId, etc.)
- Migrated shell.js, parameter-studio-page.js, and tooltips.js to ES6 modules with shared utils.js imports
- Removed dead code

## v0.2.5 (2026-04-26)

- v0.2.5 combines the documentation-system refresh with a BGE robustness update for difficult chromatic gradients such as IC434. BGE sample-estimator selection is now exposed in YAML, schema validation, and Parameter Studio, while autotune can compare robust estimators and reject degenerate flat background models when significant background or chroma spread remains.
- Professional documentation system with MkDocs Material + Doxygen
- Installation instructions for pre-built binaries (Ubuntu, Fedora, Arch)
- Configurable BGE sample estimators: `quantile`, `sigma_clipped_median`, `sextractor_mode`, and `biweight`
- BGE autotune now sweeps sample estimators and applies chroma/background-spread guards for flat or imbalanced correction surfaces
- Reconstruction fallback path hardened: safer shape/weight validation, corrected OLA memory budgeting, tile-local temporary buffers, and removal of ineffective scheduler/config dead code

## v0.2.4 (2026-04-25)

- Registration performance: anchor-promotion rounds now reuse the parallel worker pool and only retry unresolved frames whose nearest anchor changed after promotion, instead of running repeated full-frame single-threaded passes. Added `reg_promotion_retry_frames` diagnostics.

## v0.2.3 (2026-04-24)

- More robust registration: deep-chain outlier rejection (reject long chains with low CC), doubled anchor density for large-N sessions, "hopping" sequential rescue that searches past weak neighbors for better anchors, and ASTAP plate-solving as fallback even for model-interpolated frames.

## v0.2.2 (2026-04-24)

- **Hot/dead-pixel correction fixed** (`cosmetic_correction_cfa`): defective pixels inside star regions were not corrected because `neighbor_threshold` was set too low — star-halo pixels were incorrectly counted as "hot neighbours" and suppressed the correction. The threshold is now aligned with the full global hot-pixel threshold. Additionally: pixels exceeding 5× the local floor are now replaced unconditionally (`extreme_outlier` bypass). Dead/cold-pixel detection added. Works without dark frames.

## v0.2.1 (2026-04-23)

- Registration phase: NCC computation made more robust against background subtraction and hot pixels (clamp + Gaussian blur before NCC); the near-identity bypass condition strengthened with an `ncc_identity > 0.7` guard to prevent false acceptance for frames far from the reference.

## v0.2.0 (2026-04-14)

- Registration for long Alt/Az sessions was expanded substantially: N-scaled multi-anchor reference selection, N-scaled anchor promotion, astrometric registration/rescue for weak or unresolved frames, plus new practical example configs and refreshed process documentation for difficult rotation/seeing cases.

## v0.1.F (2026-04-07)

- TILE_RECONSTRUCTION performance: replaced the memory-driven worker reduction (3 workers instead of 8) with frame sub-batching. Workers now always run at the configured `parallel_workers` count; the memory budget controls how many frames are processed per batch instead of how many threads are active. Expected speedup: ~2.7× for OSC runs with 600+ frames on a 2 GB memory budget.

## v0.1.E (2026-04-06)

- Calibration/UI follow-up: dark+bias calibration now handles raw darks without double bias subtraction, `dark_already_bias_corrected` was propagated through backend, schema, example YAMLs, and Parameter Studio, and the Parameter Studio now shows one consolidated section per selected category instead of a split double view.

## v0.1.D (2026-04-04)

- Added `registration.auto_engine` (default: `true`): automatically detects strong field rotation from a small frame probe before registration and overrides the engine to `triangle_star_matching` + `transform_model: affine` when a rotation-blind engine (`robust_phase_ecc`, `hybrid_phase_ecc`) is configured for an Alt/Az dataset. The override threshold is controlled by `auto_engine_rotation_threshold_deg` (default: `0.05°/frame`).

## v0.1.C (2026-04-03)

- Stabilized tile reconstruction after the recent performance optimization rollout, with follow-up fixes and analysis focused on visible tile-seam artifacts in the final reconstruction output.

## v0.1.B (2026-03-31)

- Fixed the late PCC/output path semantics: `stacked_rgb.fits` remains the stacking output, successful `BGE`/`PCC` snapshots stay separated as `stacked_rgb_bge.fits` / `stacked_rgb_pcc.fits`, and `output_stretch` now uses only a pure linear `0..max -> 0..65535` scaling with obsolete nonlinear/quantile stretch code removed.

## v0.1.A (2026-03-29)

- Stabilized the late RGB/PCC output path after the `v3.3.9` rollout: visible RGB stretching now preserves chroma instead of amplifying weak background channel offsets, PCC background neutralization gained the new `always|auto|off` control with a nebulosity-aware auto guard, and the new parameter was propagated through schema, docs, and all example configs.

## v0.1.9 (2026-03-28)

- Promoted the `v3.3.9` methodology into the active reference state across code, frontend, and documentation: the linear reconstruction core, BGE/PCC semantics, Parameter Studio visibility, and process-flow docs now align to the same runtime baseline; backend startup handling was hardened as well.

## v0.1.8 (2026-03-25)

- Improved Linux packaging scripts to bundle all required shared libraries (OpenCV, CFITSIO, yaml-cpp, etc.) for better cross-distribution compatibility and reduced dependency issues.

## v0.1.7 (2026-03-24)

- Fixed Linux AppImage packaging to export `TILE_COMPILE_INPUT_SEARCH_ROOTS` so directory scanning works correctly in packaged releases.
- Enhanced GUI2 file browser to always show parent directory (..) even when not yet granted, triggering permission dialog on click for seamless navigation.

## v0.1.6 (2026-03-24)

- Reworked GUI2 queue/batch handling and run monitoring: batch tabs in Run Monitor, batch-targeted stats/report actions, timestamped queue-root naming with hours/minutes, and updated EN/DE documentation.

## v0.1.5 (2026-03-23)

- Stabilized `PREWARP` for OpenCL and extended GPU acceleration with OpenCL equivalents for the previously CUDA-only `TILE_RECONSTRUCTION` and `STACKING` paths, including sigma-clipping and overlap-add accumulation.

## v0.1.4 (2026-03-22)

- Added a real artifact-based `STACKING` resume path in the C++ runner so `resume --from-phase STACKING` rebuilds from `synthetic_*.fit`/`canvas_mask.fits` instead of replaying the entire pipeline.
- Fixed one synthetic/tile overlap-add weighting failure mode so zero/invalid pixels no longer contribute Hann weights. This removes that specific darkening mechanism, but residual internal line artifacts may still have other causes.

## v0.1.3 (2026-03-21)

- Added per-frame registration provenance and chain-depth tracking in the C++ registration artifacts, including stricter blind-chain anchor rules to limit drift through weak sequential rescue chains.
- Fixed GUI2 resume/run-monitor status updates so the active phase/status becomes visible immediately without requiring a manual page refresh.

## v0.1.2 (2026-03-20)

- Fixed Alt/Az registration validation to score warps on the actual common overlap instead of the cropped full-frame canvas.
- Relaxed over-aggressive CC outlier rejection for long rotating sessions by keeping the CC threshold absolute instead of run-global MAD-relative.
- Fixed field-rotation model extrapolation outside the span of valid registrations so edge/tail frames use bounded bridge prediction instead of unstable local polynomial blow-up.

## v0.1.1 (2026-03-19)

- Improved GUI2 tool persistence and PCC save handling, including temporary-output based saving and cross-platform temp-path behavior.
- Hardened backend memory usage and significantly reduced BGE autotune runtime on the IC434 reference run while preserving the selected solution behavior.

## v0.1.0 (2026-03-18)

- Fixed Astrometry/PCC tool path inputs being overwritten by backend defaults.

## v0.0.F (2026-03-17)

- Promoted the DSO tile-reconstruction methodology to `v3.3.8` in EN/DE and aligned it with the active runtime semantics.
- Corrected the normative method text for runtime-configured mode thresholds, neighborhood-aware local metric normalization, sigma-clipped tile reconstruction, and affine post-OLA photometric restoration.
- Fixed GUI2 run-name reset so changing the input directory clears the shared `run_name` across dashboard, wizard, and input-scan.
- Added a macOS 15 / Sequoia Gatekeeper note for `start_gui2.command` and blocked package approval via `System Settings -> Privacy & Security`.
- Switched ASTAP `d80` catalog downloads to platform-specific upstream packages: Linux `.deb`, macOS `.pkg`, Windows `.exe`.

## v0.0.E (2026-03-15)

- Wired `assumptions.frames_min` into the active runner mode-gate and `assumptions.reduced_mode_cluster_range` into reduced-mode clustering.engine
- Removed stale `assumptions.pipeline_profile`, `assumptions.frames_optimal`, and `assumptions.exposure_time_tolerance_percent` from the active config/schema/frontend/docs/examples surface.
- Regenerated the C++ schema and synchronized Parameter Studio, Assumptions UI, and methodology/reference docs with the active runtime semantics.

## v0.0.D (2026-03-15)

- Expanded `TILE_RECONSTRUCTION` boundary diagnostics to separate raw vs. normalized tile mismatches and exclude masked canvas zones from the metric.
- Added artifact visibility for `tile_norm_bg_*` and `tile_norm_scale` to diagnose whether per-tile normalization itself amplifies visible seams.
- Synchronized GUI2 `run_name` and `runs_dir` across dashboard, wizard, and input-scan, including direct editing on the input-scan page.

## v0.0.C (2026-03-13)

- GUI2 parameter/config handling synchronized with the current C++ config schema, defaults, and reference docs.
- Added boundary diagnostics for visible tile mismatches in `TILE_RECONSTRUCTION` and removed the ineffective dedicated seam-correction config block.
- Expanded run-monitor resume handling, live-log detail visibility, and config revision/template flows.

## v0.0.B (2026-03-12)

- Added server-side persistence for the GUI2 UI draft state via backend API/state storage.
- Migrated UX-relevant frontend parameters away from local browser storage to a central server-backed UI state.
- Synchronized run names, preset selections, config drafts, validation state, and tool inputs/results more consistently across dashboard, parameter studio, wizard, and tools.

## v0.0.A (2026-03-12)

- Bugfixes

## v0.0.9 (2026-03-11)

- Added Linux AppImage generation to the GitHub Actions release workflow.
- Reworked PCC background-noise handling and connected UI/report updates so current PCC diagnostics are exposed more consistently in the GUI.

## v0.0.8 (2026-03-11)

- zero-copy COMMON_OVERLAP
- Scratch reuse in LOCAL_METRICS
- reduced lock contention in tile_weighted-OLA
- faster sigma-clip kernel
- fewer tile copies in tile_weighted path
- parallel BGE autotune candidate evaluation

## v0.0.7 (2026-03-11)

- Supports now:
  - Linux: x86_64 Linux with `glibc >= 2.39` (Ubuntu 24.04 or equivalent is the safe baseline for the current CI-built ZIPs)
  - macOS: macOS 15
  - Windows: Windows 10 x64 or newer

## v0.0.6 (2026-03-11)

- Completed the productive migration to the Crow/C++ backend.
- Integrated C++ report generation.
- Updated launcher scripts, Docker packaging, and GitHub workflows to build and run the C++ backend directly.

## v0.0.5 (2026-03-09)

- Promoted GUI2 as the recommended interface with a web frontend, FastAPI backend, and cross-platform release bundles.
- Expanded DE/EN i18n coverage in the GUI2 frontend and parameter studio, with aligned docs and backend config handling.
- Moved the previous Qt6 GUI path into `legacy/` and clarified the actively maintained GUI2 packaging/start workflow.

## v0.0.4 (2026-03-06)

- Fixed Alt/Az registration for datasets with large field rotation.

## v0.0.3 (2026-03-05)

- Improved BGE/PCC pipeline with clearer phase visibility, stronger guardrails, and a more consistent config surface.
- Expanded parallel execution in compute-heavy stages.
- Multiple phase optimizations for more stable behavior and lower runtime overhead.

## v0.0.2 (2026-02-16)

- First release with pre-built packages for Windows, Linux, and macOS
- Includes GUI, CLI, and runner executables
- Experimental release for testing purposes

## v0.0.1 (2026-02-15)

- First public release

## Changelog

### (2026-06-22)

**GUI v3, calibration improvements, astrometry fix (`v0.3.5`):**

- **New web frontend (v3):** Improved Parameter Studio, Run Monitor, and Input & Scan tab with modern UI components.
- **"Save As" dialog with directory browser:** Modal dialog to select directory and filename for saving config files. Backend `/api/config/presets` now returns subdirectories (`is_dir` field) alongside YAML files.
- **Calibration gain mismatch:** Dark/Flat calibration with mismatched gain now produces a warning instead of aborting the run. `select_dark_inputs` falls back to all available darks when no exact match is found.
- **Warning banner in Run Monitor:** Warnings and errors from the runner are prominently displayed in a dedicated banner during runs.
- **Astrometry fix:** YAML `~` (null) for `astap_bin` was incorrectly parsed as string `"null"` by yaml-cpp, causing `fs::exists("null")` to fail and ASTAP to be reported as "not found". Fix: `IsNull()` check added before parsing `astap_bin` and `astap_data_dir` in `config.cpp`, so null values are treated as empty string and the default path (`astap_data_dir/astap_cli`) is used.

### (2026-06-20)

**PCC fix, AI prompt enhancements, build and smoke test corrections (`v0.3.4`):**

- **PCC green cast fix:** Restored the missing `if (!matrix_is_diagonal)` guard in the adaptive damping logic in `run_pcc` (`photometric_color_cal.cpp`). Without this guard, damping was applied even for diagonal matrices, causing a green cast in the stacked image. The fix ensures damping is only applied for non-diagonal matrices. Validated across multiple test runs (k3-glm, k6-fix) confirming correct PCC matrix and no green cast.
- **AI prompt enhancements:** Enhanced the AI sidecar prompt in `frameAnalysisService.ts` with stricter rules and session context. The AI now receives session geometry (mount type, field rotation estimate, session duration) alongside scan metrics, enabling more context-aware recommendations.
- **Session geometry in `cli_main.cpp`:** Extended FITS header reading to extract RA/DEC and compute session geometry (mount type detection, approximate field rotation from session duration and declination). This data is forwarded to the AI sidecar for analysis.
- **`ai_routes.cpp` fix:** Fixed a bug where `base_config` was stored as a raw string instead of a parsed JSON object. Added `session_geometry` to the analysis context sent to the AI sidecar.
- **Parameter tuning validation:** Conducted systematic comparison runs (k2, k3-glm, k4, k5, k6-fix, m31-classic, m31-default, m31-cl1) to validate PCC fix and identify optimal parameters: `weight_exponent_scale` (1.2 vs 1.5), `chroma_strength` (0.7–0.9), `apply_attenuation` (true/false), `background_neutralization_mode` (auto/always). Results confirmed the PCC fix across all configurations.
- **Windows build fix:** `timegm` is a GNU extension not available under MinGW. Replaced with `#ifdef _WIN32` → `_mkgmtime` / `#else` → `timegm` in `cli_main.cpp` for cross-platform compatibility.
- **Linux smoke test fix:** Smoke tests in `release-tile-compile-gui2.yml` hung for 1h+ because `kill` only terminated the bash script, not the backend process. Applied the following fixes:
  - Set `TILE_COMPILE_AI_AGENT_AUTOSTART=0` (prevents agent service from starting)
  - Used `setsid` + `kill -- -PID` for process group cleanup
  - Added `pkill -f tile_compile_web_backend` as fallback cleanup
- **macOS smoke test fix:** Applied the same fixes in `build_local_macos.sh`. Replaced `setsid` (Linux-only) with `python3` + `os.setsid()` since `setsid` is not available on macOS.

### (2026-06-16)

**PI – AI-assisted configuration recommendations (`v0.3.3`):**

- Implemented the PI (Parameter Intelligence) module: the AI sidecar receives the full scan result, frame quality metrics aggregate (`scan_metrics`: FWHM, noise, background, roundness, star count), all relevant configuration parameters with descriptions, and the complete schema constraints (`min`, `max`, `enum`) — and produces validated, data-driven configuration recommendations.
- `scan_metrics` are now forwarded from the backend to the AI sidecar in both POST and SSE analysis routes (`ai_routes.cpp`).
- `config_schema` sent to the sidecar now includes `minimum` and `maximum` from the JSON schema; the AI prompt formats these as `min:`/`max:` per parameter line and contains an explicit rule: values MUST stay within `[min, max]`.
- Per-update validation in `validate_updates_against_schema()`: if the combined patch fails schema validation, each update is tested individually and cumulatively; only the offending updates are rejected with `config_validation_failed` — valid recommendations are no longer discarded.
- `start_backend.sh`: agent_service sidecar is rebuilt automatically via `npm run build` whenever any `.ts` source file is newer than `dist/server.js`; `npm install` runs automatically if `node_modules` is missing or `package.json` changed.
- Traffic log: `prompt_length` diagnostic entry added before the prompt log to verify which sections are present in the prompt without relying on the truncated log output.
- Full documentation: [docs/PI/pi_ai_recommendations_en.md](docs/PI/pi_ai_recommendations_en.md)

### (2026-06-13)

**AQMH-First Implementation:**
- Top-level `method` field as single source of truth: `method: aqmh` or `method: classic_tile_compile`.
- Config normalization: Missing `method` automatically defaults to `aqmh`, `aqmh.enabled` is derived from it.
- Schema validation for `method` field with enum values `aqmh` and `classic_tile_compile`.
- Frontend: `currentMethod()` API, AQMH as default in Wizard, Dashboard, Parameter Studio.
- Run Monitor: Classic phases are hidden for AQMH (`LOCAL_METRICS`, `STATE_CLUSTERING`, `SYNTHETIC_FRAMES`).
- Rollback: `FORCE_CLASSIC=1` environment variable and `--force-classic` CLI flag.
- BGE: `tile_metrics_source` set to `aqmh_output` for AQMH runs.
- Reports: Standalone AQMH section with quality map heatmaps.
- History: Method tags for filtering and comparison.
- Documentation: [AQMH First Frontend Transition Plan v0.3.0](docs/AQMH/aqmh_first_frontend_transition_plan.md)

### (2026-06-09)

**Switch to AQMH as default reconstruction method:**
- AQMH (Adaptive Quality Map Harvesting) is now the default reconstruction path. Set `aqmh.enabled: false` to revert to classic TILE_RECONSTRUCTION.
- Normative specification: [AQMH Methodology v0.1.0](docs/AQMH/aqmh_methodik_en.md)
- All example YAML profiles updated with `aqmh:` configuration block.
- Full AQMH parameter documentation added to configuration reference and practical examples.
- `k_artifact` implementation default changed to `3.0`, `frac_artifact_max` to `0.25`.
- Main READMEs restructured; classic TBQR documentation preserved in `README_classic_tile_compile_en.md` / `README_classic_tile_compile_de.md`.

### (2026-05-25)

**Raw Stack preprocessing pipeline (`v0.2.8`):**

- Added a new standalone Raw Stack page in GUI2 for end-to-end preprocessing of FITS light frames through to a stacked and post-processed image, running fully separately from the normal Tile-Compile run studio.
- The pipeline covers all phases: Calibration (Bias/Dark/Flat), CFA/Mono Prep, Registration, Quality Analysis, Frame Filtering, Stacking (Sigma/Median/Winsor), Astrometry (ASTAP), Background Gradient Extraction (BGE), Photometric Color Calibration (PCC), and HyperMetric Stretch.
- All configurable parameters (sigma-clip, rejection method, stacking weighting, BGE, PCC, Astrometry, and HyperMetric Stretch) are taken directly from the Parameter Studio configuration — no hardcoded values.
- Output scaling correctly restores background and scale after stacking to produce accurate pixel values.
- Raw Stack UI cleanup: removed Run Monitor button, added full i18n coverage for all labels and buttons.
- See [docs/raw_stack_gui_en.md](docs/raw_stack_gui_en.md) for the full GUI reference.

### (2026-05-22) 

**implementationHMS:**
- Added VeraLux HyperMetric Stretch (HMS) as a post-PCC pipeline phase.
- HMS now defaults to enabled in the C++ config defaults, `tile_compile.yaml`, and all example YAML profiles.
- The default mode is `ready_to_use`, using adaptive anchor, Auto LogD, target background `0.2`, and output `outputs/stacked_rgb_hms.fits`.
- `mode: scientific` is implemented for controlled stretch output without ready-to-use final scaling/soft clip and with optional `linear_expansion`.
- Resume supports rerunning HMS directly via `--from-phase HYPERMETRIC_STRETCH` for historical runs with existing PCC artifacts.

### (2026-05-20) 

**Build hardening & Frontend cleanup:**
- Fixed RunnerFrameCache build errors: implemented missing `try_load_normalized` and `store_normalized` methods
- Migrated both C++ projects to C++20 (GCC 13+, Clang 16+)
- Hardened web_backend_cpp build with CUDA 13 + OpenCV 4.11 CUDA 13 configuration
- Backend route_utils: fixed incomplete AppState type errors, hardened path validation
- Frontend refactoring: centralized utilities in `src/utils.js` (escapeHtml, getMessage, getStorageJson, humanizeControlId, etc.)
- Migrated shell.js, parameter-studio-page.js, and tooltips.js to ES6 modules with shared utils.js imports
- Eliminated duplicate I18N functions across frontend scripts (message(), textFor(), activeLocale(), getLocale())
- Removed dead code: `param_editor_index.json` (36KB unused duplicate)
- Updated documentation: unified C++20 requirements, release URLs updated to v0.2.5

### (2026-04-26)

**Documentation system and BGE robustness (`v0.2.5`, 2026-04-26):**

- Added professional documentation system using MkDocs Material with Doxygen integration for C++ API reference
- Updated all GitHub Releases documentation with correct binary filenames (tile_compile_gui2-linux-v0.2.4.zip, etc.)
- Added comprehensive installation instructions for pre-built binaries on Ubuntu/Debian, Fedora/RHEL, and Arch/Manjaro
- Restructured navigation with separate User Guide, Configuration, Methodology, and API Reference sections
- Added configurable `bge.sample_estimator` support in YAML configs, schema files, and Parameter Studio (`quantile`, `sigma_clipped_median`, `sextractor_mode`, `biweight`)
- Extended BGE autotune so it can compare sample estimators and penalize or reject flat models when background/chroma spread indicates a real gradient
- Extended RGB chroma guards across BGE methods, including conservative fallbacks for imbalanced per-channel correction surfaces
- Updated `ic434_background_gradient.example.yaml` with robust RBF/`sextractor_mode` settings for IC434-like red/green background gradients
- Added `docs/reconstruction_audit_2026-04-26.md` with the reconstruction audit checklist and implementation notes
- Hardened reconstruction fallback helpers against mismatched frame/tile shapes and missing or invalid tile weights
- Reworked `reconstruct_tiles_parallel()` to use tile-sized temporary OLA buffers instead of full-frame scratch matrices per tile/sub-batch
- Updated reconstruction memory budgeting so it accounts for global overlap-add accumulators plus per-worker tile scratch
- Removed ineffective reconstruction scheduler/config dead code, including the unused GPU batch field, unused `make_hann_1d()` API, and non-functional underutilization detector

### (2026-04-25)

**Registration performance: parallel anchor-promotion retries (`v0.2.4`):**

- Fixed the direct-registration anchor-promotion loop so promoted-anchor retry passes use the configured parallel registration worker pool instead of falling back to a single-threaded `reg_worker()` call.
- Promotion rounds now build a targeted retry list and only revisit unresolved frames whose nearest active anchor is one of the newly promoted anchors, avoiding repeated full 325-frame passes when the anchor set changes.
- Registration progress now reports the actual job count and worker count for each pass, and `global_registration.json` diagnostics include `reg_promotion_retry_frames` for future runtime analysis.

### (2026-04-24)

**Registration robustness: deep-chain rejection + adaptive anchors + hopping rescue + astrometric fallback (`v0.2.3`):**

- Reject chain-validated frames with `chain_depth > max_blind_chain_depth` and `cc < reject_cc_min_abs` as `deep_chain_low_cc` outliers instead of accepting them; this prevents drift from long sequential chains through cloudy blocks.
- Increased adaptive active-anchor target from `min(21, max(3, (N+59)/60))` to `min(32, max(4, (N+29)/30))`, doubling anchor density for large-N sessions (e.g., 325 frames now use ~12 anchors instead of ~6).
- "Hopping" sequential rescue: when the direct neighbor has low CC or cannot anchor a blind chain, search up to 5 frames (for refine) or 8 frames (for rescue) in each direction for a better anchor with CC > 0.3–0.4, dramatically reducing chain depth in scattered-cloud conditions.
- Astrometric rescue moved to run *after* model-based warp prediction (Section 4b), so ASTAP can now also rescue frames that only have interpolated `model_*` provenances; added `weak_model` condition to `should_try_astrometry` so low-CC model frames are eligible for plate-solving.

### (2026-04-24)

**Hot/dead-pixel correction fix + registration code quality (`v0.2.2`):**

- Fixed `cosmetic_correction_cfa` silently skipping defective pixels inside star regions: `neighbor_threshold` was set to `0.5 × global_threshold`, so star-halo pixels (which sit well above that low bar) were counted as "hot neighbours" and blocked correction of genuine hot pixels nearby. The threshold is now raised to the full global hot-pixel threshold, so only pixels that are themselves hot-pixel candidates count as hot neighbours.
- Added `extreme_outlier` bypass: pixels exceeding `local_median + 5 × local_floor` are replaced unconditionally regardless of neighbourhood support. No real star-PSF pixel reaches that level relative to its same-colour neighbours.
- Added dead/cold-pixel correction: `global_candidate_cold` (`< median − σ_threshold × σ`) and `cold_outlier` (`< local_median − local_floor`) are now also replaced with the local same-colour median.
- All three fixes operate on the raw CFA mosaic before warping and require no dark frames.
- Diagnostic keys in `global_reg_extra` moved into a `diag` sub-object (4.2); downstream-facing keys remain at top level.
- Section headers added to `run_phase_registration_prewarp` to mark the seven major processing phases (4.1).

### (2026-04-23)

**Registration NCC robustness + near-identity guard (`v0.2.1`):**

- NCC computation in `try_method` now clamps negative values and applies a Gaussian blur (σ=1.5) before computing `ncc_identity_overlap` and `ncc_warped`. Raw normalized proxy images carry negative background values and hot pixels that caused NCC to collapse from ~0.88 to ~0.05 for sub-pixel shifts, triggering false near-identity rejections.
- Near-identity bypass condition strengthened with an `ncc_identity > 0.7` guard: a near-zero warp is only accepted as a valid near-identity result when the frame is already close to the reference, preventing false bypasses for frames that simply failed to find a shift.

### (2026-04-14)

**Registration v0.2.0: multi-anchor scaling + astrometric registration/rescue:**

- Global registration no longer relies on rigid `1/3/5` reference buckets. It now uses an N-scaled anchor selection with roughly one requested anchor per 80 frames, forced to odd anchor counts and currently capped at 15.
- Anchor promotion after strong direct matches now scales with `N` as well: the active-anchor target is roughly one anchor per 60 frames, while per-round promotions and the number of extra direct passes grow in a controlled way for long sessions.
- This reduces the classic late-reference failure mode on long Alt/Az datasets, because early and late parts of the sequence can attach directly to nearer temporal anchors instead of being forced through one distant master frame.
- Astrometric registration/rescue in the runner was upgraded in practice: ASTAP-based solves are no longer limited to `cc <= 0`, but can also replace weak or deeply chained results, using the nearest active anchor as the astrometric reference basis.
- New registration telemetry was added to `global_registration.json`, including `requested_ref_frames`, `active_ref_frames`, `reg_target_active_anchor_count`, `reg_promote_limit_per_round`, `reg_max_direct_anchor_rounds`, `reg_direct_anchor_rounds`, and `reg_source_counts`.
- Added the new example profile [tile_compile_cpp/examples/m104.example.yaml](tile_compile_cpp/examples/m104.example.yaml) for the concrete problem class "Alt/Az, somewhat stronger rotation, poor seeing, weight better frames more strongly"; the DE/EN practical examples and [docs/process_flow/phase_1_registration.md](docs/process_flow/phase_1_registration.md) were updated to match the current registration flow.

### (2026-04-07)

**TILE_RECONSTRUCTION performance: sub-batch stacking replaces worker reduction (`v0.1.F`):**

- Replaced the memory-budget-driven worker reduction in TILE_RECONSTRUCTION with frame sub-batching. Previously, a 2 GB memory budget capped OSC runs at 3 parallel workers (instead of the configured 8) because the peak RAM estimate assumed all frames loaded simultaneously per worker. Workers now always run at the configured `parallel_workers` count; the budget controls the sub-batch size (frames per batch) instead. For the reference run (610 frames, 475 tiles, 8 workers, 2 GB budget) this yields ~3 batches of ~205 frames each — same quality, ~2.7× faster TILE_RECONSTRUCTION.
- `tile_boundary_diagnostics_enabled` added to `runtime_limits` (default: `false`). Boundary diagnostics are now opt-in; the previous default of always running them added ~5–10 % overhead per production run.
- `tile_grid.json` now includes `estimated_reconstruction_time_s` (calibrated estimate based on tile count, frame count, and worker count) and `coverage_filtered_tiles`.
- `runtime_limits.json` now includes `tile_analysis_to_stack_ratio`; a warning is logged when the ratio exceeds 10.
- `phase_end` event for TILE_RECONSTRUCTION now includes `duration_s`.
- web_backend_cpp code-quality fixes: consolidated three duplicate `utc_now_iso()` implementations into a shared header, fixed SIGKILL being sent on every polling cycle after SIGTERM (now waits ~3 s), fixed FD leak on `fork()` failure, fixed sequential stdout/stderr read deadlock in `run_subprocess()`, and reduced `prune_locked()` call frequency from every mutation to terminal-state transitions only.

### (2026-04-06)

**Calibration fix + Parameter Studio reorganization (`v0.1.E`):**

- Fixed the bias/dark calibration path: when bias and dark are both enabled, a raw dark is now bias-corrected internally before being applied to lights, preventing double subtraction of the bias pedestal.
- Added the new config field `calibration.dark_already_bias_corrected` across runner, schema, docs, defaults, example configs, and GUI2 so pre-bias-corrected master darks can be marked explicitly.
- Reorganized Parameter Studio: selecting a category such as `registration` or `calibration` now shows exactly one consolidated section; missing schema parameters are injected into that same block instead of being rendered in a separate section editor above it.

### (2026-04-05)

**Calibration guardrails and backend persistence for calibration paths:**

- Updated GUI2 calibration-path handling so disabling a calibration stage removes its paths from the active config immediately and re-enabling restores the previously used paths from backend UI state, without using browser storage.
- Added extra calibration guardrails, including warnings for obvious gain mismatches between light frames and calibration files.

### (2026-04-04)

**Auto-engine for Alt/Az field rotation + registration failure fix (`v0.1.D`):**

- Added `registration.auto_engine` (default: `true`): probes a small set of frames before registration starts and automatically overrides the engine to `triangle_star_matching` + `transform_model: affine` when a rotation-blind engine (`robust_phase_ecc`, `hybrid_phase_ecc`) is configured but strong field rotation is detected. Threshold: `auto_engine_rotation_threshold_deg` (default: `0.05°/frame`, covers Alt/Az at any exposure time while staying well below EQ residual rotation).
- Fixed a complete registration failure mode: `engine: robust_phase_ecc` with `allow_rotation: true` on Alt/Az datasets produced NCC ≈ 0 for all frames, causing 469/470 frames to fall back to identity transform (`model_nearest_copy`) with no actual alignment.
- Updated `tile_compile.yaml` default engine to `triangle_star_matching` and `reject_cc_min_abs` to `0.05`.
- Propagated new config fields to all schemas, example configs, and documentation.

### (2026-04-03)

**Tile-reconstruction stabilization after recent optimization rollout (`v0.1.C`):**

- Stabilized tile reconstruction after the recent performance optimization rollout, with follow-up fixes and analysis focused on visible tile-seam artifacts in the final reconstruction output.

### (2026-03-29)

**RGB/PCC output-path stabilization after the `v3.3.9` rollout (`v0.1.A`):**

- Reworked the visible RGB output stretch so it operates luminance-aware and keeps chroma stable instead of exaggerating small background channel offsets into large blue/gray edge bands.
- Added `pcc.background_neutralization_mode = always|auto|off` with a new auto guard that attenuates or suppresses background neutralization when the measured "background" behaves like diffuse field signal rather than neutral sky.
- Synchronized the new PCC control through schema, defaults, reference docs, and all example configurations so the runtime, documentation, and example surface now expose the same behavior.

### (2026-03-28)

**Implementation and rollout of the `v3.3.9` methodology (`v0.1.9`):**

- Moved the key `v3.3.9` methodology changes into the active runtime path: linear reconstruction core without the old pre-OLA tile normalization, cleaner BGE/PCC semantics, more robust support/seam handling, and updated guards/diagnostics.
- Updated the frontend and configuration surface to the current schema/methodology baseline so new `v3.3.9` parameters are exposed more consistently in Parameter Studio and related documentation.
- Refreshed the process-flow, reference, and comparison documents for `v3.3.9`, and additionally hardened Crow/C++ web-backend startup so failures now report a clear fatal error instead of producing a core dump.

### (2026-03-24)

**AppImage packaging fix + file browser navigation enhancement (`v0.1.7`):**

- Fixed Linux AppImage packaging in `packaging/gui2/start_gui2.sh` to export `TILE_COMPILE_INPUT_SEARCH_ROOTS` environment variable, resolving directory scanning failures in packaged releases where relative paths could not be resolved.
- Enhanced GUI2 file browser (`web_frontend/tooltips.js`) to always display parent directory (..) navigation even when the parent path is not yet granted, showing a lock icon (🔒) for restricted paths and triggering the permission grant dialog on click for seamless upward navigation.
- Updated backend file listing route (`web_backend_cpp/src/routes/system_routes.cpp`) to return `parent_allowed` flag alongside `parent` path, enabling frontend to distinguish between accessible and restricted parent directories.

**GUI2 batch/queue run-monitor refresh + docs update (`v0.1.6`):**

- Reworked the GUI2 Run Monitor for queue/batch runs: queue entries now appear as tabs, redundant duplicate batch/filter rows were removed, and top-level batch/structure summary visibility was corrected again for queued runs.
- Enabled batch-targeted post-run actions in the Run Monitor so `Generate Stats`, stats-folder opening, and report opening can operate on the currently selected finished batch tab instead of only the active root/current run.
- Changed unnamed queue-root naming from date-only to `YYYYMMDD_HHMM`, making batch-root directories less collision-prone and aligning the dashboard/wizard path hints with the actual behavior.
- Expanded the EN/DE step-by-step guides with explicit batch/queue usage notes, including the primary MONO multi-filter use case and Run Monitor tab behavior.

### (2026-03-23)

**OpenCL expansion for `PREWARP`, `TILE_RECONSTRUCTION`, and `STACKING` (`v0.1.5`):**

- Stabilized the OpenCL `PREWARP` path for multi-threaded execution by guarding OpenCV OpenCL/T-API access and forcing explicit host copies where needed.
- Extended `tile_compile_cpp/src/core/acceleration.cpp` with OpenCL equivalents for the previously CUDA-only `TILE_RECONSTRUCTION` and `STACKING` paths, including sigma-clipping and overlap-add accumulation/normalization.

### (2026-03-22)

**Real `STACKING` resume + synthetic OLA seam fix (`v0.1.4`):**

- Implemented a true artifact-based `STACKING` resume path in `tile_compile_cpp/apps/runner_resume.cpp`, so `resume --from-phase STACKING` now rebuilds the stacked outputs directly from existing `synthetic_*.fit` plus `canvas_mask.fits` and continues with later phases instead of triggering an in-place full rerun.
- Fixed one overlap-add accumulation failure mode in `tile_compile_cpp/src/core/acceleration.cpp`: zero/invalid tile pixels no longer add Hann weights to `weight_sum`. This removes that specific darkening path, but residual internal seams/lines may still have other causes.

### (2026-03-21)

**Registration provenance/depth diagnostics + resume status visibility (`v0.1.3`):**

- Extended `tile_compile_cpp/apps/runner_phase_registration.cpp` so each frame now carries explicit registration provenance (`direct_global`, `sequential_rescue`, `temporal_rescue`, modeled variants, etc.) plus `chain_depth`, and writes that information into `global_registration.json`.
- Tightened blind sequential chaining: weak `sequential_rescue` frames no longer act as effectively unlimited anchors; anchor reuse is now capped by chain depth unless correlation is strong enough.
- Added aggregate registration diagnostics such as source counts, maximum observed chain depth, and blocked blind-chain-anchor attempts to the registration artifact metadata.
- Fixed GUI2/backend resume status handling so the monitor subtitle and phase state update immediately after `resume`, including the case where the runner has started but the next `resume_start` event has not yet been written to the run log.

### (2026-03-20)

**Registration/field-rotation stabilization for long Alt/Az sessions (`v0.1.2`):**

- Fixed global registration validation in `tile_compile_cpp/` so NCC comparisons are computed only on the actual valid overlap mask of the warped frame instead of on the cropped full-frame canvas. This prevents correct larger-rotation warps from being rejected just because rotated corners fall outside the fixed proxy image.
- Applied the same overlap-masked NCC validation to temporal rescue chaining, so neighbor-to-reference rescue no longer fails for the same cropped-canvas reason.
- Reworked global registration outlier CC filtering for long rotating runs: the `low_cc` reject gate now uses the configured absolute minimum directly instead of a run-global median/MAD threshold that incorrectly rejected many geometrically plausible edge frames.
- Fixed field-rotation model prediction outside the span of real registrations: tail/head frames no longer use unstable local polynomial extrapolation and instead fall back to bounded bridge-style edge prediction, preventing the severe fan-out / wedge artifacts seen on the M66 Alt/Az regression run.

### (2026-03-19)

**GUI2 tool persistence/PCC UX, backend memory guards, and BGE autotune speed-up:**

- Hardened `web_backend_cpp/` against OOM-prone API/tool paths with capped subprocess/stdout capture, bounded scan/report payload retention, streamed event-file inspection, and retained-job limits plus environment-configurable defaults documented for GUI2.
- Added `packaging/gui2/.env.example` and documented the new backend runtime limit environment variables used by GUI2 launchers.
- Fixed GUI2 frontend/backend asset serving and route behavior so `/ui` and direct asset paths resolve reliably instead of producing 404s.
- Improved Astrometry/PCC tool UX: persistent in-progress download state across page switches, corrected download progress calculation, automatic PCC WCS prefill from matching files, and automatic PCC parameter import from a run `config.yaml` with visible traceability in the UI/log.
- Reworked PCC output handling in GUI2 so `Run PCC` writes to a temporary result, `Save Corrected` uses a styled in-app save dialog, copies the RGB result plus `_R/_G/_B` sidecar files from the temp output, and works consistently across Linux/macOS/Windows temp directories.
- Fixed standalone PCC fallback behavior when `canvas_mask` is missing by using a safe full-image fallback instead of aborting the tool run.
- Added BGE phase timing diagnostics to `bge.json` and optimized the real hotspot in `tile_compile_cpp/`: autotune prep now reuses prepared tile analysis across quantile candidates, reducing measured BGE wall time on the IC434 reference run from about `472s` to about `181s` without adding new full-frame memory pressure.

### (2026-03-18)

- Fixed Astrometry data directory input not being respected when user manually changes the path - now uses `shouldKeepAstapSelection` logic to preserve user input.
- Added server-side persistence for Astrometry and PCC tool parameters via UI state API - settings survive server restarts.
- Improved catalog download intelligence: Astrometry catalogs skip download if already installed, PCC Siril only downloads missing chunks.
- Enhanced archive extraction robustness for macOS `.pkg`, Linux `.deb`, and Windows `.exe` formats with better error messages and validation.
- Fixed macOS release bundle library issues by explicitly bundling GCC runtime libraries (`libgcc_s`, `libgfortran`, `libquadmath`, `libgomp`) and preserving `libstdc++` for Homebrew-compiled dependencies.

### (2026-03-17)

**Methodology `v3.3.8` + GUI2 run-name reset (`v0.0.F`):**

- Added new normative methodology documents `docs/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.8_en.md` and `docs/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.8_de.md`.
- Corrected the method specification so it matches the active runtime for operating-mode thresholds, shared-core channel semantics, neighborhood-aware local metric normalization, sigma-clipped tile reconstruction, and affine post-OLA photometric restoration.
- Fixed GUI2 so a changed input directory clears the shared `run_name` across dashboard, wizard, and input-scan.
- Added a short macOS 15 / Sequoia install note for Gatekeeper-blocked `start_gui2.command` launch.
- Changed ASTAP `d80` downloads from the invalid shared ZIP assumption to the real upstream platform packages: Linux `.deb`, macOS `.pkg`, Windows `.exe`.

### (2026-03-15)

**Assumptions runtime/config synchronization (`v0.0.E`):**

- `assumptions.frames_min` is now used by the active C++ runner mode gate instead of the old hardcoded minimum-frame threshold.
- `assumptions.reduced_mode_cluster_range` now affects reduced-mode clustering directly, so the exposed config field is no longer parser-only drift.
- Removed dead assumptions fields from the active config surface: `pipeline_profile`, `frames_optimal`, and `exposure_time_tolerance_percent`.
- Synchronized active C++ config code, generated schemas, example YAMLs, GUI2 Assumptions/Parameter Studio, and DE/EN docs to the remaining runtime-relevant assumptions fields.

### (2026-03-15)

**Boundary diagnostics deepening + GUI2 run-field synchronization:**

- Extended `TILE_RECONSTRUCTION` diagnostics so `tile_reconstruction.json` now exposes raw and normalized tile-boundary metrics separately, plus `tile_norm_bg_r/g/b` and `tile_norm_scale` for direct normalization analysis.
- Corrected tile-boundary analysis to exclude masked `COMMON_OVERLAP` / canvas-invalid zones instead of counting them as valid zero-valued samples.
- Updated the methodology/process/reference/practical docs to reflect the read-only raw/normalized boundary diagnostics and the common-canvas-mask requirement.
- Added `run_name` and `runs_dir` editing to Input&Scan and unified both fields across dashboard, wizard, and input-scan via the shared GUI2 stored state.

### (2026-03-13)

**GUI2 config/studio sync + tile-boundary diagnostics update:**

- Removed the ineffective `stacking.tile_seam_harmonization.*` experiment from the active C++ config surface and replaced it with read-only tile-boundary diagnostics in `TILE_RECONSTRUCTION`.
- Synchronized config code, generated schemas, example configs, and DE/EN reference docs with the active C++ config surface.
- Reworked Parameter Studio so parameter inventory, defaults, ranges, tooltips, and filtering are driven from the current schema/default config instead of stale manual lists.
- Extended GUI2 live-log and run-monitor behavior, including richer phase details, resume config editing/template flows, stored config revisions, and corrected phase status after successful resume.

### (2026-03-12)

**Server-side GUI2 UI-state persistence:**

- Added persistent backend storage plus API access for the GUI2 UI draft state so frontend UX state no longer depends primarily on local browser storage.
- Migrated the major UX-relevant frontend parameters to the shared server-backed UI state, including run naming, preset synchronization, config drafts, validation state, dirty state, queues, and tool path/input settings.
- Restored and synchronized additional tool result state across reloads where useful, while keeping purely ephemeral runtime display state non-persistent.

### (2026-03-11)

**Crow/C++ runtime, release packaging, and PCC update:**

- Finalized the productive GUI2 path around the Crow/C++ backend, including integrated C++ report generation and aligned frontend/backend report handling.
- Updated release packaging, local build/start scripts, and GitHub workflows for Linux, macOS, and Windows, including the documented GUI2 bundle OS baselines.
- Added Linux AppImage creation to the GitHub Actions release workflow so releases now include a portable Linux artifact alongside the ZIP bundle.
- Added date-aware run-directory naming and aligned route/websocket handling plus backend tests for the new naming behavior.
- Reworked PCC background-noise handling and connected UI/report updates so current PCC diagnostics are exposed more consistently in the GUI.

### (2026-03-09)

**GUI2 release + i18n refresh:**

- Promoted the web-based GUI2 stack (`web_frontend/` + `web_backend_cpp/`) to the recommended UI path and updated the top-level docs accordingly.
- Added the dedicated GUI2 release workflow and launcher packaging for Linux, macOS, and Windows under `.github/workflows/release-tile-compile-gui2.yml` and `packaging/gui2/`.
- Expanded frontend localization coverage and parameter-studio translations, with matching backend config contract updates and tests.
- Moved the earlier Qt6 GUI/build-script path into `legacy/` to separate the maintained GUI2 route from the legacy desktop implementation.

### (2026-03-10)

**Python elimination in the productive GUI2 path:**

- Switched GUI2 runtime, packaging, Docker, and CI to the Crow/C++ backend.
- Removed the productive Python dependency for stats/report generation; this now runs via the integrated C++ backend path and CLI support.
- Updated the repository structure and GUI2 documentation to reflect `web_backend_cpp/` as the maintained backend implementation.

### (2026-03-05, later update)

**Strict/Practical runtime unification + verification:**

- Unified the image-processing runtime core path for `assumptions.pipeline_profile: strict|practical`.
- Removed strict-only execution branches in the hot path:
  - no strict-only pre-registration order path,
  - no strict-only reduced/full gate override (`max(200, threshold)`),
  - no strict-only tile re-normalization branch,
  - no strict-only channel re-weighting branch in OSC tile stacking.
- Registration no longer force-overrides `registration.enable_star_pair_fallback=false` in strict mode.
- Updated config reference docs (EN/DE) so profile text matches current runtime behavior.
- Added A/B evidence run pair (`max_frames=80`) confirming same core flow with only minor numeric fit variance.

### (2026-03-05)

**Performance and throughput optimization (large datasets, 1000+ frames):**

- Added adaptive worker selection per phase with I/O-aware caps based on sampled frame size and task count.
- `DiskCacheFrameStore` now uses persistent memory-mapped frame views with rewrite invalidation, reducing repeated open/mmap/unmap overhead for tile access.
- Removed the global PREWARP store mutex so frame-cache writes can proceed concurrently.
- `GLOBAL_METRICS` now runs in a parallel worker pool with thread-safe progress and error aggregation.
- `TILE_RECONSTRUCTION` overlap-add switched from a single global lock to row-stripe locking to reduce contention.
- In OSC tile reconstruction, each valid frame tile is debayered once and cached as R/G/B planes for reuse across channel stacks.
- `LOCAL_METRICS` now skips globally invalid tiles before extraction and limits heavy full-matrix artifact writes for large production runs.

### (2026-03-03)

**Methodology alignment (v3.3.6 strict profile):**

- Added `assumptions.pipeline_profile: practical|strict` to switch between compatibility mode and strict normative behavior.
- In `strict` profile, REGISTRATION/PREWARP is executed before CHANNEL_SPLIT/NORMALIZATION/GLOBAL_METRICS.
- In `strict` profile, reduced/full gating enforces full mode only from `N >= 200`.
- In `strict` profile, phase-7 tile normalization before OLA is always enabled.
- PCC `auto_fwhm` now falls back deterministically to `FWHM=0` when seeing is unavailable.
- Added `registration.enable_star_pair_fallback` (default `true`); strict profile disables it to match the normative cascade order.
- Updated config schema/sample config and v3 reference docs (DE/EN) for these settings.

**BGE/PCC configuration and docs alignment:**

- Restored user-facing BGE fit parameters `bge.fit.robust_loss` and `bge.fit.huber_delta`.
- Added user-facing BGE apply guards `bge.min_valid_sample_fraction_for_apply` and `bge.min_valid_samples_for_apply`.
- Re-enabled parse/serialize/schema support for these keys in the runtime config surface.
- Runner mapping now forwards the configured values (no internal forced override).
- BGE config artifacts in both pipeline and resume paths include `robust_loss` and `huber_delta` again.
- Updated BGE/PCC docs and practical examples (DE/EN) to match current behavior and active parameter set.

### (2026-02-26)

**BGE Phase Visibility / Comparison Outputs:**

- BGE is now emitted as a dedicated pipeline enum phase (`BGE=15`) between `ASTROMETRY` and `PCC`.
- GUI phase progress now shows BGE explicitly, including BGE substep progress updates.
- Added explicit pre-PCC output `outputs/stacked_rgb_bge.fits` for direct BGE-only vs BGE+PCC comparison.
- Configuration docs/examples updated for v3.3.6 option set:
  - `bge.autotune.*` (`enabled`, `strategy`, `max_evals`, `holdout_fraction`, `alpha_flatness`, `beta_roughness`)
  - `pcc.background_model`
  - `pcc.max_condition_number`, `pcc.max_residual_rms`
  - `pcc.radii_mode`
  - `pcc.aperture_fwhm_mult`, `pcc.annulus_inner_fwhm_mult`, `pcc.annulus_outer_fwhm_mult`, `pcc.min_aperture_px`

### (2026-02-25)

**Registration / Canvas / Color-Correctness Fixes:**

- **Bayer parity-safe offsets in registration/prewarp path**: Canvas offsets are now handled consistently to preserve CFA parity across expanded/cropped canvases.
- **Output scaling origin fixes**: Scaling calls now use the correct tile/debayer offsets where required, preventing R/G parity mismatches after crop/canvas transforms.
- **Common-overlap and canvas handling clarified** in process-flow docs and aligned with the current phase model.

**PCC (Photometric Color Calibration) Improvements:**

- **Robust log-chromaticity fit** implemented for PCC matrix estimation (instead of the older proportion-only approach).
- **Guardrails on channel scales** added to avoid extreme global color casts.
- **Aperture annulus contamination filter (IQR gate)** added to reject unstable star measurements in nebulous/gradient-heavy fields.

**Documentation Refresh:**

- `docs/process_flow/*` updated to the current production pipeline state, including `PREWARP`, `COMMON_OVERLAP`, canvas/offset propagation, and current enum phase ordering.

**BGE (Background Gradient Extraction):**

- Added optional pre-PCC BGE stage that directly subtracts modeled background from RGB channels.
- Added foreground-aware BGE fit method `modeled_mask_mesh` for difficult fields with large diffuse objects (e.g. M31/M42) to reduce color-cloud artifacts before PCC.
- Added `artifacts/bge.json` with per-channel diagnostics (tile samples, grid cells, residual statistics).
- Extended report generation to include a dedicated BGE section with summary plots and residual analysis.

### (2026-02-19)

**Calibration Fixes:**

- **GUI dark calibration propagation fixed**: If `use dark` is enabled and either **Darks dir** or **Dark master** is set, these values are now merged into the effective runtime config and applied by the runner. This fixes cases where dark calibration appeared enabled in the GUI but was not present in the run config (`use_dark: false`, empty dark paths).

### (2026-02-17)

**New Registration Features for Alt/Az Mounts Near Pole:**

- **Temporal-Smoothing Registration**: For field rotation, automatically uses neighbor frames (i-1, i+1) for registration when direct registration to reference fails. Chained warps: `i→(i-1)→ref` or `i→(i+1)→ref`. Useful for continuous field rotation (Alt/Az near pole) and clouds/nebula.

- **Adaptive Star Detection**: When too few stars are detected (< topk/2), automatically performs a second pass with lower threshold (2.5σ instead of 3.5σ). This improves star detection in clouds, nebula, or weak frames.

- **New Registration Engine**: `robust_phase_ecc` with LoG gradient preprocessing, optimized for frames with strong nebulae/clouds.

**Field Rotation Support:**

- **Canvas Expansion for Alt/Az Mounts**: Output canvas is now automatically expanded to contain all rotated frames. Previously, stars at the edges were cropped when using Alt/Az mounts near the pole. The bounding box of all warped frames is computed and the canvas is resized accordingly. Log output shows expansion: `"Field rotation detected: expanding canvas from WxH to W'xH'"`.

**Documentation:**

- **New**: [Practical Configuration Examples & Best Practices](docs/configuration_examples_practical_en.md) - Comprehensive guide with use cases for different focal lengths, seeing conditions, mount types, and camera setups (DWARF, Seestar, DSLR, Mono CCD). Includes parameter recommendations based on methodology v3.3.4.
