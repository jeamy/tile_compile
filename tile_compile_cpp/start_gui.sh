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

# Check if C++ Backend/Runner was built
if [ ! -x "build/tile_compile_runner" ]; then
  echo "WARNING: C++ backend not built yet." >&2
  echo "Please run: cd build && cmake .. && make" >&2
  echo "Continuing anyway - GUI will show errors when running pipeline." >&2
fi

echo "Info: Backend: C++ (build/tile_compile_runner)" >&2
exec ./build/tile_compile_gui
