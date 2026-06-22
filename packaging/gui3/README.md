# tile_compile GUI3 Release Bundle

This directory contains the launcher scripts and packaging helpers for the GUI3 release bundle. GUI3 consists of the web frontend v3, the Crow/C++ backend, and the native C++ runner/CLI.

## Bundle Layout

The generated release archive contains:

- `start_gui3.sh` for Linux
- `start_gui3.command` for macOS
- `start_gui3.bat` and `start_gui3.ps1` for Windows
- `payload/` with:
  - `web_backend_cpp/`
  - `web_frontend_v3/`
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

The backend is intentionally started under the launcher shell so the terminal window stays attached after startup and can stop the server directly with `Ctrl+C` on Linux/macOS or in the launcher console on Windows.

## Minimum OS Versions

Current practical minimums for the packaged GUI3 release bundles are:

- Linux: x86_64 Linux with `glibc >= 2.39` (the current release workflow builds on Ubuntu 24.04; Ubuntu 24.04 or equivalent is the safe baseline)
- macOS: macOS 15
- Windows: Windows 10 x64 or newer

Notes:

- macOS release bundles are now built with an explicit deployment target and are intended to run from macOS 13 upward, not only on the exact build host version.
- Linux compatibility below the CI build baseline is not guaranteed for the current ZIP bundles because `glibc` is not bundled.
- Windows packaging is built and smoke-tested on `windows-2022`; Windows 10/11 x64 is the intended baseline.

## Build Dependencies

The current native C++ build requirements for the GUI3 release are:

- Linux: `libcurl4-openssl-dev`
- macOS: `curl`
- Windows MSYS2: `mingw-w64-x86_64-curl`

Other core dependencies still include Eigen, OpenCV, cfitsio, yaml-cpp, nlohmann-json and OpenSSL.

macOS notes:

- `packaging/gui3/build_local_macos.sh` requires `xcode-select --install`, `cmake`, `ninja`, `pkg-config`, and `python3`.
- On macOS 12, Homebrew's default `opencv` formula is not supported. The Homebrew-based packaging path therefore effectively requires macOS 15 unless OpenCV is provided from another working installation.

## CI Workflow

The GitHub Actions workflow is:

- `.github/workflows/release-tile-compile-gui3.yml`

It builds the runner binaries and the Crow backend, bundles GUI3 files, copies native runtime libraries, runs a smoke test, and uploads release ZIP artifacts.

## Local Packaging

To reproduce the release-style packaging locally, use the scripts in this directory:

- Linux: `packaging/gui3/build_local_linux.sh`
- macOS: `packaging/gui3/build_local_macos.sh`
- Windows (MSYS2 MinGW64): `packaging/gui3/build_local_windows_msys2.sh`

They mirror the release workflow closely:

1. build `tile_compile_cpp` (`tile_compile_runner`, `tile_compile_cli`)
2. build `tile_compile_web_backend`
3. assemble the GUI3 bundle with `payload/`
4. collect native runtime libraries
5. run a smoke test against `/api/app/state`
6. create the ZIP artifact in `artifacts/`

Examples:

```bash
packaging/gui3/build_local_linux.sh --tag dev
packaging/gui3/build_local_macos.sh --tag dev
packaging/gui3/build_local_windows_msys2.sh --tag dev
```

Common options:

- `--skip-build` to reuse existing build directories
- `--skip-smoke` to skip the launch test
- `--build-type <type>` to switch CMake configuration
- `--port <port>` to change the smoke-test port

## Environment Example

An example environment file is available at:

- `packaging/gui3/.env.example`

It documents the supported GUI3 launcher variables and the backend memory-guard limits.
The launcher scripts do not auto-load this file; source it manually before starting GUI3 if you want to override defaults.
