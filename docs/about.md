# About Tile-Compile

Tile-Compile is a toolkit for high-quality astronomical image reconstruction from short-exposure deep-sky datasets. It is designed for smart telescope data (DWARF, Seestar, ZWO SeeStar, etc.) but works with any FITS input.

## Two Reconstruction Methods

| Method | Description | Status |
|--------|-------------|--------|
| **AQMH** | Adaptive Quality Map Harvesting — pixel-wise quality maps with Laplacian pyramid, per-pixel weighted reconstruction | ✅ Default (v0.3.0+) |
| **Classic Tile-Compile** | Tile-based quality reconstruction with local metrics, clustering, synthetic frames, and OLA stacking | Optional (`method: classic_tile_compile`) |

## Features

- **GUI3** — Web-based interface with Scan Input, Parameter Studio, Run Monitor, Results, Astrometry/PCC, Raw Stack, and Run History tabs
- **CLI** — Full command-line interface for scripting and automation
- **AQMH** — Per-pixel quality-map-driven reconstruction with cherry-pick frame selection
- **Classic pipeline** — Tile-based reconstruction with state clustering and synthetic frames
- **Calibration** — Bias/Dark/Flat calibration with auto-selection and exposure matching
- **Astrometry** — ASTAP plate solving with WCS output
- **BGE** — Background Gradient Extraction before PCC
- **PCC** — Photometric Color Calibration
- **HyperMetric Stretch** — VeraLux stretch after PCC
- **Raw Stack** — Standalone preprocessing pipeline (calibration → stacking → post-processing)
- **AI-assisted configuration** — Parameter Intelligence (PI) module for data-driven recommendations
- **Reports** — HTML reports with charts, heatmaps, and diagnostics

## Tech Stack

- **C++20** with Eigen, OpenCV (CUDA 13), cfitsio, nlohmann/json, YAML-cpp
- **Crow** C++ backend for the web interface
- **Vanilla JS** frontend (no build step required)
- **Python** scripts for analysis and documentation

## Downloads

Pre-built binaries for Linux, macOS (Apple Silicon + Intel), and Windows are available on [GitHub Releases](https://github.com/jeamy/tile_compile/releases).

## Documentation

- [Installation](getting_started/installation.md)
- [Quick Start](getting_started/quickstart.md)
- [GUI3 User Guide (EN)](gui3_user_guide_en.md)
- [GUI3 Benutzerhandbuch (DE)](gui3_user_guide_de.md)
- [Configuration Reference](configuration_reference.md)
- [Pipeline Overview](process_flow/phase_0_overview.md)
- [Methodology v3.3.9](v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.9_en.md)

## License

See [LICENSE](https://github.com/jeamy/tile_compile/blob/main/LICENSE).

---

## Changelog

### (2026-06-22) — v0.3.5

- **New web frontend (v3):** Improved Parameter Studio, Run Monitor, and Input & Scan tab with modern UI components.
- **"Save As" dialog with directory browser:** Modal dialog to select directory and filename for saving config files.
- **Calibration gain mismatch:** Dark/Flat calibration with mismatched gain now produces a warning instead of aborting.
- **Warning banner in Run Monitor:** Warnings and errors prominently displayed during runs.
- **Astrometry fix:** YAML `~` (null) for `astap_bin` was incorrectly parsed as string `"null"`, causing ASTAP to be reported as "not found".

### (2026-06-20) — v0.3.4

- **PCC green cast fix:** Restored missing `if (!matrix_is_diagonal)` guard in adaptive damping logic.
- **AI prompt enhancements:** AI sidecar receives session geometry (mount type, field rotation, session duration).
- **Windows build fix:** `timegm` replaced with `_mkgmtime` under MinGW.
- **Linux/macOS smoke test fix:** Process group cleanup for backend process.

### (2026-06-16) — v0.3.3

- **PI – AI-assisted configuration recommendations:** AI sidecar receives scan metrics, config parameters, and schema constraints to produce validated recommendations.
- Per-update validation: invalid updates rejected individually, valid ones no longer discarded.
- `start_backend.sh`: agent_service sidecar auto-rebuilt when source changes.

### (2026-06-13) — AQMH-First Implementation

- Top-level `method` field: `aqmh` (default) or `classic_tile_compile`.
- Frontend: AQMH as default in Wizard, Dashboard, Parameter Studio.
- Run Monitor: Classic phases hidden for AQMH runs.
- Reports: Standalone AQMH section with quality map heatmaps.

### (2026-06-09) — AQMH Default

- AQMH (Adaptive Quality Map Harvesting) is now the default reconstruction path.
- All example YAML profiles updated with `aqmh:` configuration block.
- Full AQMH parameter documentation added to configuration reference.
- Classic TBQR documentation preserved in separate READMEs.

### (2026-05-25) — v0.2.8

- **Raw Stack preprocessing pipeline:** Standalone page for end-to-end preprocessing (calibration → stacking → post-processing).
- All parameters taken from Parameter Studio configuration.

### v0.2.4

- First release with pre-built packages for Windows, Linux, and macOS.
- Includes GUI, CLI, and runner executables.

### v0.0.1 (2026-02-15)

- First public release.
