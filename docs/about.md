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
