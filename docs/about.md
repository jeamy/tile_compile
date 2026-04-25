# Tile-Compile

Tile-Compile is a toolkit for **tile-based quality reconstruction** of astronomical image stacks (methodology v3.3).

## What is TBQR?

**Tile-Based Quality Reconstruction (TBQR)** replaces rigid frame selection with a robust spatio-temporal quality model. By decomposing frames into local tiles and modeling quality along two orthogonal axes—global atmospheric transparency/noise and local structural sharpness—we reconstruct a signal that is physically and statistically optimal at every pixel.

## Quick Links

- [Installation](getting_started/installation.md) – Download or build from source
- [CLI Reference](getting_started/cli_reference.md) – All runner and CLI commands
- [Configuration](configuration_reference.md) – Parameter reference
- [Pipeline Overview](process_flow/phase_0_overview.md) – Phase-by-phase documentation
- [Methodology v3.3.9](v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.9_en.md) – Full specification
- [TBQR Step-by-Step](tbqr_step_by_step_en.md) – Practical guide

## Features

- **Advanced Registration** – Global + sequential alignment with astrometric fallback
- **Tile-based Reconstruction** – Local quality-weighted stacking
- **Background Gradient Extraction (BGE)** – Remove light pollution gradients
- **Photometric Color Calibration (PCC)** – Accurate color using reference stars
- **Quality-driven Clustering** – Separate frames by seeing conditions
- **Smart Telescope Optimized** – For Seestar, Dwarf II, etc.
- **Resume Capability** – Continue from any pipeline phase
- **Docker Support** – Containerized execution
- **GUI2 Web Interface** – Browser-based parameter studio

## Components

| Component | Directory | Technology |
|-----------|-----------|------------|
| Core pipeline | `tile_compile_cpp/` | C++17 + OpenCV + cfitsio |
| GUI2 backend | `web_backend_cpp/` | Crow + C++17 |
| GUI2 frontend | `web_frontend/` | HTML + CSS + JavaScript |

## External Links

- [GitHub Repository](https://github.com/jeamy/tile_compile)
- [GitHub Releases](https://github.com/jeamy/tile_compile/releases)

---

## Changelog

See [GitHub Releases](https://github.com/jeamy/tile_compile/releases) for detailed release notes.

### v0.2.4 (2026-04-25)
- Registration robustness improvements
- Sequential refinement with hopping anchor search
- Better handling of large datasets

### v0.2.3 (2026-04-20)
- Deep-chain low-CC rescue for blind sequential registration
- ASTAP rescue for weak_model frames
- Adaptive active anchor count for large datasets

### v0.2.2 (2026-04-15)
- Orientation trend outlier rejection
- Enhanced NCC computation with clamp+blur
- Improved anchor selection heuristics

---

*Full README available at the [GitHub repository](https://github.com/jeamy/tile_compile)*
