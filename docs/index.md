# tile_compile

**Tile-based quality reconstruction for astrophotography**

---

## Overview

tile_compile is a scientific-grade image stacking pipeline designed for astrophotography. It reconstructs high-quality images from multiple light frames through:

- **Advanced Registration** – Global + sequential alignment with astrometric fallback
- **Tile-based Reconstruction** – Local quality-weighted stacking (AQMH)
- **Background Gradient Extraction (BGE)** – Remove light pollution gradients
- **Photometric Color Calibration (PCC)** – Accurate color using reference stars
- **Quality-driven Clustering** – Separate frames by seeing conditions
- **Modern GUI3** – Browser-based interface with full workflow support

## Quick Start

### Download GUI3 (Recommended)

```bash
# Linux GUI3 (Browser Interface)
curl -L -o tile_compile_gui3.zip \
  https://github.com/jeamy/tile_compile/releases/latest/download/tile_compile_gui3-linux-v0.3.0.zip
unzip tile_compile_gui3.zip && cd tile_compile_gui3-linux-v0.3.0

# Start GUI3 (browser opens automatically)
./start_gui3.sh  # http://127.0.0.1:8080/ui/
```

Also available: [macOS](getting_started/installation.md), [Windows](getting_started/installation.md), or [build from source](getting_started/installation.md).

### Typical GUI3 Workflow

1. **Scan Input** – Select FITS lights folder, optional calibration frames
2. **Adjust Parameters** – Load example config or customize settings
3. **Start & Monitor** – Run with real-time phase progress tracking
4. **View Results** – Stacked images, diagnostic reports, quality metrics

## Documentation

- **[GUI3 User Guide (EN)](gui3_user_guide_en.md)** – Complete step-by-step guide
- **[GUI3 Benutzerhandbuch (DE)](gui3_user_guide_de.md)** – Deutsche Schritt-für-Schritt-Anleitung
- [Installation](getting_started/installation.md)
- [Configuration Reference](configuration_reference.md)
- [Raw Stack GUI (EN)](raw_stack_gui_en.md)
- [Raw Stack GUI (DE)](raw_stack_gui_de.md)

## Features

- **Modern GUI3** – Browser-based interface with Processing, Tools, and History tabs
- **AQMH Method** – Advanced tile-based quality reconstruction (default)
- **Resume Capability** – Continue from any pipeline phase
- **Smart Telescopes** – Optimized for Seestar, Dwarf II, etc.
- **Astrometry & PCC** – Built-in plate solving and color calibration
- **Docker Support** – Containerized execution

---

*[View on GitHub](https://github.com/jeamy/tile_compile)*
