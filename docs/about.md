# About Tile-Compile

Tile-Compile is a toolkit for high-quality astronomical image reconstruction from short-exposure deep-sky datasets. It is designed for smart telescope data (DWARF, Seestar, ZWO SeeStar, etc.) but works with any FITS input.

## Reconstruction Method

**CFA Forward Drizzle + Multiband** is the single reconstruction method:
drizzle before debayer — every raw CFA sample is drizzled directly into its
channel accumulator plane, followed by à-trous multiband fusion. The earlier
AQMH and Classic Tile-Compile methods have been removed; a top-level `method:`
key is rejected fail-closed.

## Features

- **GUI3** — Web-based interface with Scan Input, Parameter Studio, Run Monitor, Results, Astrometry/PCC, Raw Stack, and Run History tabs
- **CLI** — Full command-line interface for scripting and automation
- **CFA Forward Drizzle** — Direct CFA sample drizzling with support-mask semantics, chunked bounded-memory gather, and transactional resume checkpoints
- **Multiband** — À-trous band fusion with three-way candidate selection
- **Calibration** — Bias/Dark/Flat calibration with auto-selection and exposure matching
- **Astrometry** — ASTAP plate solving with WCS output
- **BGE** — Background Gradient Extraction (classic + AutoBGE)
- **PCC** — Photometric Color Calibration
- **HyperMetric Stretch** — VeraLux stretch after PCC
- **Raw Stack** — Standalone preprocessing pipeline (calibration → stacking → post-processing)
- **AI-assisted configuration** — Parameter Intelligence (PI) module for data-driven recommendations
- **Reports** — HTML reports with charts, heatmaps, and diagnostics
- **GPU acceleration** — CUDA for Forward Drizzle geometry/gather, CUDA + OpenCL for source-quality-map filters; CPU fallback preserved

## Tech Stack

- **C++20** with Eigen, OpenCV (CUDA 13), cfitsio, nlohmann/json, YAML-cpp
- **Crow** C++ backend for the web interface
- **Vanilla JS** frontend (no build step required)
- **Node.js** optional PI AI sidecar

## Downloads

Pre-built binaries for Linux, macOS (Apple Silicon + Intel), and Windows are available on [GitHub Releases](https://github.com/jeamy/tile_compile/releases).

## License

See [LICENSE](https://github.com/jeamy/tile_compile/blob/main/LICENSE).
