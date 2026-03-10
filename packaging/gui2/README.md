# tile_compile GUI2 Release Bundle

This directory contains the launcher scripts and packaging helpers for the GUI2 release bundle. GUI2 consists of the web frontend, the Crow/C++ backend, and the native C++ runner/CLI.

## Bundle Layout

The generated release archive contains:

- `start_gui2.sh` for Linux
- `start_gui2.command` for macOS
- `start_gui2.bat` and `start_gui2.ps1` for Windows
- `payload/` with:
  - `web_backend_cpp/`
  - `web_frontend/`
  - `tile_compile_cpp/build/` with `tile_compile_runner` and `tile_compile_cli`
  - `web_backend_cpp/build/` with `tile_compile_web_backend`
  - `tile_compile_cpp/lib/` with bundled native runtime libraries
  - `tile_compile_cpp/examples/`
  - `tile_compile_cpp/tile_compile.yaml`
  - `tile_compile_cpp/tile_compile.schema.yaml`
  - `tile_compile_cpp/tile_compile.schema.json`

## First Start

The launcher does not run directly from the extracted archive forever. On start it copies the bundled payload into a per-user install directory:

- Linux/macOS: `~/tilecompile`
- Windows: `%USERPROFILE%\\tilecompile`

After that it:

1. copies the bundled payload into the user install directory
2. reuses the bundled native libraries
3. starts the Crow backend in the foreground
4. opens the browser on `/ui/`

No Python runtime, virtual environment, or pip installation is required in the productive release path.

The backend is intentionally started in the foreground so it can be stopped directly with `Ctrl+C` on Linux/macOS or in the launcher console on Windows.

## Build Dependencies

GUI2 itself no longer depends on Qt in the release path. The current native C++ build requirements for the GUI2 release are:

- Linux: `libcurl4-openssl-dev`
- macOS: `curl`
- Windows MSYS2: `mingw-w64-x86_64-curl`

Other core dependencies still include Eigen, OpenCV, cfitsio, yaml-cpp, nlohmann-json and OpenSSL.

## CI Workflow

The GitHub Actions workflow is:

- `.github/workflows/release-tile-compile-gui2.yml`

It builds the Qt-free runner binaries and the Crow backend, bundles GUI2 files, copies native runtime libraries, runs a smoke test, and uploads release ZIP artifacts.
