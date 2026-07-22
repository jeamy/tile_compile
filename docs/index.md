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

## Documentation

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
- [TBQR Methodology v3.3.9 (EN)](v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.9_en.md)
- [Process Flow](process_flow/phase_0_overview.md)

### Changelog

- [Release Notes](changelog/releases.md)
- [Detailed Changelog](changelog/detailed_changelog.md)

---

*[View on GitHub](https://github.com/jeamy/tile_compile)*
