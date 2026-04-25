# tile_compile

**Tile-based quality reconstruction for astrophotography**

---

## Overview

tile_compile is a scientific-grade image stacking pipeline designed for astrophotography. It reconstructs high-quality images from multiple light frames through:

- **Advanced Registration** – Global + sequential alignment with astrometric fallback
- **Tile-based Reconstruction** – Local quality-weighted stacking
- **Background Gradient Extraction (BGE)** – Remove light pollution gradients
- **Photometric Color Calibration (PCC)** – Accurate color using reference stars
- **Quality-driven Clustering** – Separate frames by seeing conditions

## Quick Start

### Download Pre-built Binary (No Build Required)

```bash
# Linux GUI2 (CLI + Web Interface)
curl -L -o tile_compile.zip \
  https://github.com/jeamy/tile_compile/releases/latest/download/tile_compile_gui2-linux-v0.2.4.zip
unzip tile_compile.zip && cd tile_compile_gui2-linux-v0.2.4

./tile_compile_runner run \
  --config tile_compile.yaml \
  --input-dir /path/to/lights \
  --runs-dir ./runs

# Or start web UI
./start_backend.sh  # http://localhost:8080/ui/
```

Also available: [AppImage](getting_started/installation.md), [macOS](getting_started/installation.md), [Windows](getting_started/installation.md), or [build from source](getting_started/installation.md).

## Documentation

- [Installation](getting_started/installation.md)
- [CLI Reference](getting_started/cli_reference.md)
- [Configuration Reference](configuration_reference.md)
- [Pipeline Overview](process_flow/phase_0_overview.md)

## Features

- **Multiple Config Modes** – From emergency (fast) to full (maximum quality)
- **Smart Telescopes** – Optimized for Seestar, Dwarf II, etc.
- **Resume Capability** – Continue from any pipeline phase
- **Docker Support** – Containerized execution
- **GUI2 Web Interface** – Browser-based parameter studio

---

*[View on GitHub](https://github.com/jeamy/tile_compile)*
