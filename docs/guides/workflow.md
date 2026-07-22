# Typical Workflow (GUI3)

The standard user workflow with GUI3 involves three steps:

1. **Scan input** — Tab *Processing → Input & Scan*: Select input folder with FITS lights, optionally specify calibration frames (bias/dark/flat), start scan. The scan detects frames, resolution, and color mode.
2. **Adjust parameters** — Tab *Processing → Parameter*: Load an example config or adjust values. Key parameters: registration (rotation, transform model), AQMH (cherry-pick, pyramid scales), stacking method, Bayer pattern. Validate and save configuration.
3. **Start and monitor run** — Tab *Processing → Run Monitor*: Start run, track phase progress in real time, abort or resume from a specific phase as needed.

After completion: Results are in `runs/<run_id>/outputs/`. Generate a diagnostic report via *Generate Stats* in the Run Monitor or via Run History.

Full guide: [GUI3 User Guide](../gui3_user_guide_en.md)

## Pipeline Phases

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

Detailed phase docs: [Process Flow](../process_flow/phase_0_overview.md)

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
- Reference document: [Configuration Reference](../configuration_reference_en.md)
- Practical examples: [Configuration Examples & Best Practices](../configuration_examples_practical_en.md)

### Example profiles

Complete standalone example configs are available under `tile_compile_cpp/examples/`.

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

See also: [Examples README](https://github.com/jeamy/tile_compile/blob/master/tile_compile_cpp/examples/README.md) for the intended use case and tuning focus of each profile.

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

### GUI3

Development start from repository root (requires [building from source](../getting_started/installation.md)):

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
- The optional PI AI sidecar (`agent_service/`) requires **Node.js >= 20**. If Node.js is not installed or too old, the backend starts without the AI sidecar and prints a warning. See [GUI3 README](https://github.com/jeamy/tile_compile/blob/master/packaging/gui3/README.md) for details.

## Paper Example Data Sources

- M31 lights source for the paper example run (10 GB): [M31 lights](https://wolke.eibrain.org/index.php/s/Z88dmWizEJYjwBe) 
- M31 run source for the paper example run (20 GB): [M31 run](https://wolke.eibrain.org/index.php/s/tfSycSNEzdL7jje)
