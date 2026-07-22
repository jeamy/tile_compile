# Tile-Compile

Tile-Compile is a toolkit for high-quality astronomical image reconstruction from short-exposure deep-sky datasets. The default reconstruction method is **AQMH (Adaptive Quality Map Hyperstacking)** — a per-pixel, quality-map-driven approach that replaces tile-based overlap-add stacking with a physically optimal pixel-wise weighted average.

> **Classic Tile-Compile (TBQR):** The original tile-based quality reconstruction methodology is still available and fully supported. See [Classic Tile-Compile README (EN)](README_classic_tile_compile_en.md) and [Classic Tile-Compile README (DE)](README_classic_tile_compile_de.md). Set `aqmh.enabled: false` to revert to classic TILE_RECONSTRUCTION.

> **Note:** This is experimental software primarily developed for processing images from smart telescopes (e.g., DWARF, Seestar, ZWO SeeStar, etc.). While designed for general astronomical image processing, it has been optimized for the specific characteristics and challenges of smart telescope data.

## Quick Start

### GUI3

Download a pre-built bundle from [GitHub Releases](https://github.com/jeamy/tile_compile/releases), or build from source (see [Installation](docs/getting_started/installation.md)) and start from repository root:

```bash
./start_backend.sh
```

Then open: http://127.0.0.1:8080/ui/

Release bundle start:

- Linux: `start_gui3.sh`
- macOS: `start_gui3.command`
- Windows: `start_gui3.bat`

### CLI

```bash
./tile_compile_runner run \
  --config tile_compile.yaml \
  --input-dir /path/to/lights \
  --runs-dir /path/to/runs
```

### Docker

```bash
./start_gui3_docker.sh
```

Open: http://127.0.0.1:8080/ui/

## Documentation

Full documentation site: **[https://jeamy.github.io/tile_compile/](https://jeamy.github.io/tile_compile/)**

### Getting Started

- [Quick Start](docs/getting_started/quickstart.md)
- [Installation](docs/getting_started/installation.md)
- [CLI Reference](docs/reference/cli.md)
- [Configuration](docs/getting_started/configuration.md)

### User Guides

- [GUI3 User Guide (EN)](docs/gui3_user_guide_en.md) — Complete step-by-step guide
- [GUI3 Benutzerhandbuch (DE)](docs/gui3_user_guide_de.md) — Deutsche Schritt-für-Schritt-Anleitung
- [Workflow & Pipeline Phases](docs/guides/workflow.md) — Typical GUI3 workflow, phase table, registration cascade
- [Raw Stack GUI](docs/guides/raw_stack_gui.md) — Standalone preprocessing pipeline (not optimized, retained for legacy reasons)
- [PI – AI-Assisted Recommendations](docs/guides/pi_ai.md) — Data-driven parameter recommendations

### AQMH

- [AQMH Overview](docs/guides/aqmh_overview.md) — How it works, key parameters, when to use
- [AQMH Methodology v0.2.1 (normative)](docs/AQMH/aqmh_methodik_en_v0.2.1.md)
- [AQMH v0.2.0 Paper (PDF)](docs/AQMH/zenodo-0.2.0/paper-adaptive_quality_map_hyperstacking_m31_run_20260722_en.pdf)
- [AQMH v0.1.0 Paper](docs/AQMH/zenodo-0.1.0/)

### Configuration

- [Configuration Reference (EN)](docs/configuration_reference_en.md)
- [Configuration Reference (DE)](docs/configuration_reference.md)
- [Practical Examples (EN)](docs/configuration_examples_practical_en.md)
- [Practical Examples (DE)](docs/configuration_examples_practical_de.md)
- Example profiles: `tile_compile_cpp/examples/`

### Reference

- [Build from Source](docs/reference/build.md) — Build requirements, GPU acceleration, package installs
- [Docker](docs/reference/docker.md) — Container build, run, and configuration
- [CLI Reference](docs/reference/cli.md) — Runner, scan, config, resume, report generation
- [Outputs & Artifacts](docs/reference/outputs.md) — Run output directory structure
- [Calibration & External Tools](docs/reference/calibration.md) — Bias/dark/flat, ASTAP, Siril catalog
- [Project Structure](docs/reference/project_structure.md) — Repository layout and components

### Methodology

- [AQMH Methodology v0.2.1](docs/AQMH/aqmh_methodik_en_v0.2.1.md) — Current AQMH normative specification
- [TBQR Methodology v3.3.9 (EN)](docs/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.9_en.md)
- [TBQR Methodology v3.3.9 (DE)](docs/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.9_de.md)
- [Process Flow](docs/process_flow/) — Phase-by-phase implementation docs

### Changelog

- [Release Notes](docs/changelog/releases.md)
- [Detailed Changelog](docs/changelog/detailed_changelog.md)

### Other Languages

- [German README](README_de.md)
- [Classic README (EN)](README_classic_tile_compile_en.md)
- [Classic README (DE)](README_classic_tile_compile_de.md)

## Attribution

This project was built with assistance from Windsurf-Devin, Kiro, Antigravity, GPT, Claude, Codex, ***. Babysitting by a human in a virtual environment.

The PI (Parameter Intelligence) module uses:

- **[@earendil-works/pi-coding-agent](https://github.com/earendil-works/pi/tree/main/packages/coding-agent)** — AI agent framework (v0.80.x)

The HyperMetric Stretch (HMS) phase is derived from the VeraLux HyperMetric Stretch Siril script:

- (c) 2025 Riccardo Paterniti — VeraLux - HyperMetric Stretch — GPL-3.0-or-later — Version 1.5.2

The AutoBGE (Background Gradient Extraction) phase is based on the AutoBGE Siril script:

- (c) Adrian Knagg-Baugh from Franklin Marek SAS code (2025) — GPL-3.0-or-later — Version 2.0.2
