# Context for ChatGPT / External Review

Project: tile_compile_cpp
Purpose: C++ implementation of
         Tile-basierte Qualitätsrekonstruktion für DSO – Methodik v4

Key points:
- C++17 with Qt6 GUI
- OpenCV for image processing
- cfitsio for FITS file I/O
- Tile-local registration (v4) is authoritative
- Performance-optimized implementation

Entry points:
- apps/runner_main.cpp (CLI runner)
- apps/cli_main.cpp (CLI tool)
- gui_cpp/main.cpp (Qt6 GUI)
- tile_compile.yaml (configuration)

Directory structure:
- include/ - Header files
- src/ - Implementation files
- apps/ - Application entry points
- gui_cpp/ - Qt6 GUI implementation
- tests/ - Unit tests (excluded from snapshot)

Open questions / TODO:
- (fill in if desired)
