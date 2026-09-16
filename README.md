# Tile-Compile

Tile-Compile is a toolkit for high-quality astronomical image reconstruction from short-exposure deep-sky datasets. The reconstruction method is **CFA Forward Drizzle + Multiband**: every calibrated source sample is forward-mapped through the measured registration geometry into a super-resolution output grid — directly from the Bayer (CFA) data for OSC input, without any debayering step — and fused across frequency bands into the final linear image.

There is exactly one method. AQMH and Classic Tile-Compile have been removed; a `method:` key in a configuration file is rejected fail-closed (`tile_compile_cli migrate-config` cleans legacy configs). Historical AQMH/TBQR methodology documents under `docs/AQMH/` and `docs/v3/` remain as records.

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
./tile_compile_runner reconstruct \
  --config tile_compile.yaml \
  --input-dir /path/to/lights \
  --runs-dir /path/to/runs \
  --project-root /path/to/tile_compile
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
- [Live Image Editor (EN)](docs/guides/live_image_editor_en.md) — Non-destructive FITS editing, previews, undo/redo, repeat, and AI/local fallback behavior

### Forward Drizzle Pipeline

- [CFA Forward Drizzle + Multiband (EN)](docs/guides/cfa_forward_drizzle_pipeline_en.md) — How it works, candidates, key parameters
- [CFA Forward Drizzle + Multiband (DE)](docs/guides/cfa_forward_drizzle_pipeline_de.md) — Deutsche Variante
- [Process Flow](docs/process_flow/) — Phase-by-phase implementation docs (EN/DE)
- [Forward-Drizzle v2 Target Architecture (DE)](docs/forward_drizzle_v2_zielarchitektur_2026-09-12_de.md) — Normative design document

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

### Methodology (historical records)

- [CFA Forward-Drizzle + Multiband Implementation Plan (DE)](docs/AQMH/aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md) — the plan the single-method cutover follows
- [AQMH Methodology v0.2.1](docs/AQMH/aqmh_methodik_en_v0.2.1.md) — superseded method, kept as record
- [TBQR Methodology v3.3.9 (EN)](docs/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.9_en.md) — Classic Tile-Compile, kept as record
- [TBQR Methodology v3.3.9 (DE)](docs/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.9_de.md)

### Changelog

- [Release Notes](docs/changelog/releases.md)
- [Detailed Changelog](docs/changelog/detailed_changelog.md)

### Other Languages

- [German README](README_de.md)

## Attribution

This project was built with assistance from Windsurf-Devin, Kiro, Antigravity, GPT, Claude, Codex, ***. Babysitting by a human in a virtual environment.

The PI (Parameter Intelligence) module uses:

- **[@earendil-works/pi-coding-agent](https://github.com/earendil-works/pi/tree/main/packages/coding-agent)** — AI agent framework (v0.80.x)

The HyperMetric Stretch (HMS) phase is derived from the VeraLux HyperMetric Stretch Siril script:

- (c) 2025 Riccardo Paterniti — VeraLux - HyperMetric Stretch — GPL-3.0-or-later — Version 1.5.2

The AutoBGE (Background Gradient Extraction) phase is based on the AutoBGE Siril script:

- (c) Adrian Knagg-Baugh from Franklin Marek SAS code (2025) — GPL-3.0-or-later — Version 2.0.2
