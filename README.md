# Tile-Compile

Tile-Compile is a toolkit for **tile-based quality reconstruction** of astronomical image stacks (methodology v3.3).

We present a novel methodology for the reconstruction of high-quality astronomical images from short-exposure deep-sky datasets. Conventional stacking methods often rely on binary frame selection ("lucky imaging"), which discards significant portions of collected frames. Our approach, **Tile-Based Quality Reconstruction (TBQR)**, replaces rigid frame selection with a robust spatio-temporal quality model. By decomposing frames into local tiles and modeling quality along two orthogonal axes—global atmospheric transparency/noise and local structural sharpness—we reconstruct a signal that is physically and statistically optimal at every pixel. We demonstrate that this method preserves the full photometric depth of the dataset while achieving superior resolution improvement compared to traditional reference stacks.

While the methodology was originally conceived to address the specific challenges of short-exposure data from modern smart telescopes (e.g., Dwarf, Seestar), its architectural flexibility makes it equally potent for conventional astronomical setups. The extensive set of tunable parameters—ranging from adaptive tile sizing and cross-correlation thresholds to sophisticated clustering logic—allows the pipeline to be meticulously optimized for a wide array of optical systems and atmospheric conditions.

> **Practical note:** The pipeline is primarily optimized for datasets with many usable frames. With very small frame counts, or with strongly mixed frame quality in one stack, visible tile patterns can occur in difficult cases. This can often be mitigated by testing different configuration settings (especially registration, tile, and reconstruction-related parameters). See the example profiles in `tile_compile_cpp/examples/` and `tile_compile_cpp/examples/README.md`.

> **Note:** This is experimental software primarily developed for processing images from smart telescopes (e.g., DWARF, Seestar, ZWO SeeStar, etc.). While designed for general astronomical image processing, it has been optimized for the specific characteristics and challenges of smart telescope data.

## Documentation (v3.3)

- Normative methodology: [Tile-Based Quality Reconstruction Methodology v3.3.6](doc/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.6_en.md)
- Methodology paper PDF v3.3.6: [paper-tile_based_quality_reconstruction_methodology_v_3.3.6_en.pdf](doc/v3/paper-tile_based_quality_reconstruction_methodology_v_3.3.6_en.pdf)
- Implementation process flow: [Process flow (English)](doc/v3/process_flow/README_en.md)
- Implementation process flow (German): [Process flow](doc/v3/process_flow/README_de.md)
- English step-by-step guide: [Step-by-Step Guide](doc/v3/tbqr_step_by_step_en.md)
- German step-by-step guide: [Schritt-für-Schritt-Anleitung](doc/v3/tbqr_step_by_step_de.md)
- German README snapshot: [German README](README_de.md)
- Data flow (user-friendly): [Process Flow – How the System Works](doc/v3/process_flow/data_flow_user_description_en.md)

## Paper Example Data Sources

- M31 lights source for the paper example run (10 GB): [M31 lights](https://wolke.eibrain.org/index.php/s/Z88dmWizEJYjwBe) 
- M31 run source for the paper example run (20 GB): [M31 run](https://wolke.eibrain.org/index.php/s/tfSycSNEzdL7jje)

Given a directory of FITS lights, the pipeline can:

- optionally **calibrate** lights (bias/dark/flat)
- **register** frames using a robust fallback cascade
- compute **global and local (tile) quality metrics**
- reconstruct the image via tile-weighted overlap-add
- optionally cluster frame states and build synthetic frames
- **stack** using sigma-clip or weighted averaging
- **debayer** OSC/CFA data
- run **astrometry** (WCS)
- run optional **background gradient extraction** (BGE, pre-PCC)
- run **photometric color calibration** (PCC)
- write final outputs and detailed diagnostic artifacts

## Active Components

| Component | Directory | Status | Stack |
|-----------|-----------|--------|-------|
| Core pipeline | `tile_compile_cpp/` | Active | C++17 + Eigen + OpenCV + cfitsio + yaml-cpp |
| GUI2 backend | `web_backend_cpp/` | Active | Crow + C++17 |
| GUI2 frontend | `web_frontend/` | Active | HTML + CSS + JavaScript |

## Pipeline Phases

In practical use, the overall workflow is intentionally simple: after you provide the input data and a manageable set of configuration parameters, the pipeline processes the dataset automatically from stacking through astrometry, optional background handling, and PCC to the final result. No complicated manual intermediate steps are required for a normal run. At the same time, the system remains fully configurable in depth, so every stage can still be tuned in fine detail whenever you need tighter control over registration, tiling, reconstruction, stacking, or post-processing behavior.

| ID | Phase | Description |
|----|-------|-------------|
| 0 | SCAN_INPUT | Input discovery, mode detection, linearity check, disk-space precheck |
| 1 | REGISTRATION | Cascaded global registration |
| 2 | PREWARP | Full-frame canvas prewarp (CFA-safe for OSC) |
| 3 | CHANNEL_SPLIT | Metadata phase (channel model) |
| 4 | NORMALIZATION | Linear background-based normalization |
| 5 | GLOBAL_METRICS | Global frame metrics and weights |
| 6 | TILE_GRID | Adaptive tile geometry |
| 7 | COMMON_OVERLAP | Common valid-data overlap (global/tile-local masks) |
| 8 | LOCAL_METRICS | Local tile metrics and local weights |
| 9 | TILE_RECONSTRUCTION | Weighted overlap-add reconstruction |
| 10 | STATE_CLUSTERING | Optional state clustering |
| 11 | SYNTHETIC_FRAMES | Optional synthetic frame generation |
| 12 | STACKING | Final linear stacking |
| 13 | DEBAYER | OSC demosaic to RGB (MONO pass-through) |
| 14 | ASTROMETRY | Plate solving / WCS |
| 15 | BGE | Optional RGB background gradient extraction before PCC |
| 16 | PCC | Photometric color calibration |
| 17 | DONE | Final status (`ok` or `validation_failed`) |

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
- Reference document: [Configuration Reference](doc/v3/configuration_reference_en.md)
- Practical examples: [Configuration Examples & Best Practices](doc/v3/configuration_examples_practical_en.md)

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

## Binary Releases (GUI2)

Pre-built GUI2 release bundles are published via [GitHub Releases](https://github.com/jeamy/tile_compile/releases).

Each bundle contains:

- GUI2 frontend (`web_frontend/`)
- Crow backend (`web_backend_cpp/`)
- native C++ tools (`tile_compile_runner`, `tile_compile_cli`, `tile_compile_web_backend`)
- launchers for Linux, macOS, and Windows

At runtime, GUI2 uses the local Crow/C++ backend as the process adapter for the C++ runner/CLI.

## Quickstart

### GUI2 (recommended)

Development start from repository root:

```bash
./start_backend.sh
```

Then open:

```text
http://127.0.0.1:8080/ui/
```

Release bundle start:

- Linux: `start_gui2.sh`
- macOS: `start_gui2.command`
- Windows: `start_gui2.bat`

The launcher copies the bundled payload into a per-user install directory, starts the Crow backend in the foreground, and opens the browser to the local GUI2 URL.

### C++ CLI / runner

For a full beginner-friendly walkthrough, see:
[Step-by-Step Guide](doc/v3/tbqr_step_by_step_en.md)

### Build Requirements

- CMake >= 3.21
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

- `ninja` is required for the local GUI2 packaging scripts.
- On macOS 12, the default Homebrew `opencv` formula is currently not supported. The Homebrew-based path therefore effectively requires macOS 13+ for OpenCV, unless you provide a separate working OpenCV installation yourself.

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

GUI2 release bundles are built by:

- `.github/workflows/release-tile-compile-gui2.yml`

The workflow builds the Qt-free C++ binaries, bundles `web_backend_cpp/` and `web_frontend/`, adds the GUI2 launchers, and creates ZIP artifacts for Linux, Windows, macOS Apple Silicon, and macOS Intel.

Not included by design:

- external Siril catalog data
- external ASTAP binary/data
  
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

#### Windows notes (Docker / CLI workflow)

Use a Linux shell (WSL2 Ubuntu) to run the helper script:

```bash
bash scripts/docker_compile_and_run.sh build-image
bash scripts/docker_compile_and_run.sh run-app -- run --config /mnt/config/tile_compile.yaml --input-dir /mnt/input --runs-dir /workspace/tile_compile_cpp/runs
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

Supported resume phases: `ASTROMETRY`, `BGE`, `PCC`.

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

### GUI2 integration

The recommended UI path is the web-based GUI2:

- backend: `web_backend_cpp/`
- frontend: `web_frontend/`
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
- **ASTAP** (for astrometry / WCS plate solving)
  - Requires ASTAP plus a star database (e.g., D50 for deep-sky use).
  - Official site/downloads: `https://www.hnsky.org/astap.htm`

If these resources are not installed, core reconstruction still works, but ASTROMETRY/PCC phases may be skipped or fail depending on configuration.

## Diagnostic Report (`report.html` via C++ backend)

Generate an HTML quality report from a finished run either via GUI2 or directly through the CLI:

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
- `dark_auto_select: true` matches darks by exposure time (±5%)

## Project Structure

```text
tile_compile/
├── web_frontend/           # GUI2 HTML/CSS/JS frontend
├── web_backend_cpp/        # GUI2 Crow/C++ backend
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
├── packaging/gui2/          # GUI2 release launchers/bundle helpers
├── docker/                  # Docker build/runtime images
├── doc/
│   ├── v3/                  # methodology and process-flow docs
│   └── gui2/                # GUI2 concept/reference docs
├── start_backend.sh         # dev start for Crow backend + GUI2
├── start_gui2_docker.sh     # run GUI2 in Docker
├── README.md
└── README_de.md
```

## Tests

```bash
cd tile_compile_cpp/build
ctest --output-on-failure
```

## Versions

### v0.0.1 (2026-02-15)

- First public release

### v0.0.2 (2026-02-16)

- First release with pre-built packages for Windows, Linux, and macOS
- Includes GUI, CLI, and runner executables
- Experimental release for testing purposes

### v0.0.3 (2026-03-05)

- Improved BGE/PCC pipeline with clearer phase visibility, stronger guardrails, and a more consistent config surface.
- Expanded parallel execution in compute-heavy stages.
- Multiple phase optimizations for more stable behavior and lower runtime overhead.

### v0.0.4 (2026-03-06)

- Fixed Alt/Az registration for datasets with large field rotation.

### v0.0.5 (2026-03-09)

- Promoted GUI2 as the recommended interface with a web frontend, FastAPI backend, and cross-platform release bundles.
- Expanded DE/EN i18n coverage in the GUI2 frontend and parameter studio, with aligned docs and backend config handling.
- Moved the previous Qt6 GUI path into `legacy/` and clarified the actively maintained GUI2 packaging/start workflow.

### v0.0.6 (2026-03-11)

- Completed the productive migration to the Crow/C++ backend.
- Integrated C++ report generation.
- Updated launcher scripts, Docker packaging, and GitHub workflows to build and run the C++ backend directly.

## Changelog

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

- `doc/v3/process_flow/*` updated to the current production pipeline state, including `PREWARP`, `COMMON_OVERLAP`, canvas/offset propagation, and current enum phase ordering.

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

- **New**: [Practical Configuration Examples & Best Practices](doc/v3/configuration_examples_practical_en.md) - Comprehensive guide with use cases for different focal lengths, seeing conditions, mount types, and camera setups (DWARF, Seestar, DSLR, Mono CCD). Includes parameter recommendations based on methodology v3.3.4.
