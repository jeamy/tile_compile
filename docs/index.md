# Tile-Compile

**High-quality astronomical image reconstruction from short-exposure deep-sky datasets.**

---

## What is Tile-Compile?

Tile-Compile is a scientific-grade image stacking pipeline designed for astrophotography. It reconstructs high-quality images from multiple light frames through:

- **AQMH Reconstruction** — Per-pixel quality-map-driven weighted average (default method)
- **Classic Tile-Compile (TBQR)** — Tile-based quality reconstruction with local metrics, clustering, and OLA stacking
- **Advanced Registration** — Cascaded global + sequential alignment with astrometric fallback
- **Background Gradient Extraction (BGE)** — Remove light pollution gradients
- **Photometric Color Calibration (PCC)** — Accurate color using reference stars
- **HyperMetric Stretch (HMS)** — VeraLux-based post-PCC stretch
- **Modern GUI3** — Browser-based interface with full workflow support
- **AI-Assisted Configuration** — Parameter Intelligence (PI) module for data-driven recommendations
- **Live Image Editor** — Non-destructive FITS editing after a run, with live preview, undo/redo, repeatable operations, and optional AI proposals

Designed for smart telescope data (DWARF, Seestar, ZWO SeeStar, etc.) but works with any FITS input.

## Quick Start

### Download GUI3

```bash
# Linux GUI3 (Browser Interface)
curl -L -o tile_compile_gui3.zip \
  https://github.com/jeamy/tile_compile/releases/latest/download/tile_compile_gui3-linux-v0.3.0.zip
unzip tile_compile_gui3.zip && cd tile_compile_gui3-linux-v0.3.0

# Start GUI3 (browser opens automatically)
./start_gui3.sh  # http://127.0.0.1:8080/ui/
```

Also available: [macOS](getting_started/installation.md), [Windows](getting_started/installation.md), or [build from source](reference/build.md).

### Typical GUI3 Workflow

1. **Scan Input** — Select FITS lights folder, optional calibration frames
2. **Adjust Parameters** — Load example config or customize settings
3. **Start & Monitor** — Run with real-time phase progress tracking
4. **View Results** — Stacked images, diagnostic reports, quality metrics

## Find the right documentation

### New users

Start with the [Quick Start](getting_started/quickstart.md) (English) or [Schnellstart](getting_started/quickstart_de.md) (German). The [GUI3 User Guide](gui3_user_guide_en.md) walks through scanning, parameters, running a stack, and viewing results step by step. If you prefer German, use the [GUI3 Benutzerhandbuch](gui3_user_guide_de.md).

Recommended path:

1. Install or download GUI3.
2. Scan a folder of FITS light frames.
3. Keep the example configuration for the first run.
4. Start and monitor the run.
5. Open the result in the [Live Image Editor](guides/live_image_editor_en.md) or inspect the report.

### Experienced users

Use the [workflow guide](guides/workflow.md), [configuration reference](configuration_reference_en.md), and [practical examples](configuration_examples_practical_en.md). The **Professional & Technical** section in the site navigation contains phase internals, data flow, resume contracts, and methodology documents.

### Documentation map

The online help is intentionally split into four levels:

- **Start Here** — installation, quick start, and the complete GUI walkthrough.
- **Workflows & Tools** — day-to-day processing, Raw Stack, PI, and Live Image Editor.
- **Configuration** — beginner configuration, full parameter reference, and examples.
- **Professional & Technical** — algorithms, phase artifacts, data flow, resume dependencies, and normative methodology.

### Getting Started

- [Quick Start](getting_started/quickstart.md)
- [Installation](getting_started/installation.md)
- [CLI Reference](reference/cli.md)
- [Configuration](getting_started/configuration.md)

### User Guides

- [GUI3 User Guide (EN)](gui3_user_guide_en.md)
- [GUI3 Benutzerhandbuch (DE)](gui3_user_guide_de.md)
- [Workflow & Pipeline Phases](guides/workflow.md)
- [AQMH Overview](guides/aqmh_overview.md)
- [Raw Stack GUI](guides/raw_stack_gui.md) — not optimized, retained for legacy reasons
- [PI – AI-Assisted Recommendations](guides/pi_ai.md)
- [Live Image Editor (EN)](guides/live_image_editor_en.md)
- [Live Image Editor (DE)](guides/live_image_editor_de.md)

### Configuration

- [Configuration Reference (EN)](configuration_reference_en.md)
- [Configuration Reference (DE)](configuration_reference.md)
- [Practical Examples (EN)](configuration_examples_practical_en.md)
- [Practical Examples (DE)](configuration_examples_practical_de.md)

### Reference

- [Build from Source](reference/build.md)
- [Docker](reference/docker.md)
- [Outputs & Artifacts](reference/outputs.md)
- [Calibration & External Tools](reference/calibration.md)
- [Project Structure](reference/project_structure.md)

### Methodology

- [AQMH Methodology v0.2.1](AQMH/aqmh_methodik_en_v0.2.1.md)
- [CFA Forward-Drizzle and Multiband as the Single Method — Implementation Plan (DE)](AQMH/aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md)
- [TBQR Methodology v3.3.9 (EN)](v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.9_en.md)
- [Process Flow](process_flow/phase_0_overview.md)

### Changelog

- [Release Notes](changelog/releases.md)
- [Detailed Changelog](changelog/detailed_changelog.md)

---

*[View on GitHub](https://github.com/jeamy/tile_compile)*
