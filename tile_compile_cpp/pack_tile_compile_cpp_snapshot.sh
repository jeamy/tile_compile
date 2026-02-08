#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# pack_tile_compile_cpp_snapshot.sh
#
# Creates a clean C++ project snapshot archive for review / analysis.
# Designed for tile_compile C++ implementation (Methodik v3).
# -----------------------------------------------------------------------------

set -euo pipefail

PROJECT_NAME="tile_compile_cpp"
DATE=$(date +%Y%m%d)
OUT="${PROJECT_NAME}_snapshot_${DATE}.tar.gz"

# Sanity check
if [[ ! -f "CMakeLists.txt" ]]; then
  echo "ERROR: run this script from the tile_compile_cpp directory (CMakeLists.txt missing)" >&2
  exit 1
fi

# Optional context file
if [[ ! -f README_CONTEXT.md ]]; then
  cat <<'EOF' > README_CONTEXT.md
# Context for ChatGPT / External Review

Project: tile_compile_cpp
Purpose: C++ implementation of
         Tile-basierte Qualitätsrekonstruktion für DSO – Methodik v3

Key points:
- C++17 with Qt6 GUI
- OpenCV for image processing
- cfitsio for FITS file I/O
- Tile-local registration is supported
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
EOF
  echo "Created README_CONTEXT.md"
fi

# Build archive
tar \
  --exclude='build' \
  --exclude='build/**' \
  --exclude='.cache' \
  --exclude='.cache/**' \
  --exclude='runs' \
  --exclude='runs/**' \
  --exclude='*.o' \
  --exclude='*.a' \
  --exclude='*.so' \
  --exclude='*.log' \
  --exclude='*.tmp' \
  --exclude='.git' \
  --exclude='CMakeFiles' \
  --exclude='cmake_install.cmake' \
  --exclude='Makefile' \
  --exclude='compile_commands.json' \
  -czf "$OUT" \
  CMakeLists.txt \
  README.md \
  README_CONTEXT.md \
  tile_compile.yaml \
  tile_compile.schema.yaml \
  tile_compile.schema.json \
  start_gui.sh \
  include \
  src \
  apps \
  gui_cpp

echo "Snapshot created: $OUT"
echo "You can now upload this archive directly."
