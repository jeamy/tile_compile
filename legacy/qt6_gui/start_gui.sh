#!/usr/bin/env bash
set -e

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "${repo_root}"

echo "Info: starting C++/Qt6 GUI" >&2

# Check if C++ GUI was built
if [ ! -x "build/tile_compile_gui" ]; then
  echo "ERROR: C++ GUI not built yet." >&2
  echo "Please run: cd build && cmake .. && make tile_compile_gui" >&2
  exit 1
fi

# Check if C++ CLI backend (used by GUI tabs) was built
if [ ! -x "build/tile_compile_cli" ]; then
  echo "WARNING: C++ CLI backend not built yet." >&2
  echo "Please run: cd build && cmake .. && make tile_compile_cli" >&2
  echo "Continuing anyway - GUI will show errors for GUI/backend actions." >&2
fi

echo "Info: Backend: C++ (build/tile_compile_cli)" >&2

# Launch from build dir so relative backend paths like ./tile_compile_cli resolve correctly.
cd "${repo_root}/build"
exec ./tile_compile_gui "$@"
