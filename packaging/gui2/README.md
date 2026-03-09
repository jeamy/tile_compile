# tile_compile GUI2 Release Bundle

This directory contains the launcher scripts and packaging helpers for the GUI2 release bundle. GUI2 consists of the web frontend, the FastAPI backend, and the native C++ runner/CLI.

## Bundle Layout

The generated release archive contains:

- `start_gui2.sh` for Linux
- `start_gui2.command` for macOS
- `start_gui2.bat` and `start_gui2.ps1` for Windows
- `payload/` with:
  - `web_backend/`
  - `web_frontend/`
  - `tile_compile_cpp/build/` with `tile_compile_runner` and `tile_compile_cli`
  - `tile_compile_cpp/lib/` with bundled native runtime libraries
  - `tile_compile_cpp/scripts/`
  - `tile_compile_cpp/examples/`
  - `tile_compile_cpp/tile_compile.yaml`
  - `tile_compile_cpp/tile_compile.schema.yaml`
  - `tile_compile_cpp/tile_compile.schema.json`

## First Start

The launcher does not run directly from the extracted archive forever. On start it copies the bundled payload into a per-user install directory:

- Linux/macOS: `~/tilecompile`
- Windows: `%USERPROFILE%\\tilecompile`

After that it:

1. checks whether Python 3.11+ is available
2. asks whether Python should be installed if it is missing
3. aborts with a clear message if Python is not installed
4. creates `~/.venv` inside the user install directory
5. always installs `web_backend/requirements-backend.txt`
6. starts the FastAPI backend in the foreground
7. opens the browser on `/ui/`

Without Python the app does not work, because GUI2 depends on the FastAPI backend and the report generation path.

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

It builds the Qt-free runner binaries, bundles GUI2 files, copies native runtime libraries, runs a smoke test, and uploads release ZIP artifacts.
