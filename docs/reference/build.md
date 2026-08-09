# Build from Source

## Build Requirements

- CMake >= 3.21
- C++20 compiler (GCC 13+, Clang 16+, or MSVC 2022 17.8+)
- OpenCV >= 4.5
- Eigen3
- cfitsio
- yaml-cpp
- nlohmann-json
- Node.js >= 20 (only for optional PI AI sidecar / `agent_service/`)

## GPU Acceleration Requirements

The pipeline supports two GPU backends:

### NVIDIA CUDA (opencv_cuda)

- Requires OpenCV CUDA modules:
  - `opencv2/core/cuda.hpp`
  - `opencv2/cudawarping.hpp`
  - `opencv2/cudaarithm.hpp`
  - `opencv2/cudafilters.hpp`
- At runtime, a CUDA-capable NVIDIA GPU and working CUDA/OpenCV runtime are required.
- `TILE_COMPILE_ENABLE_CUDA` only enables the CUDA hook/build gate.

### AMD/Intel/NVIDIA OpenCL (opencv_opencl)

- Requires OpenCV OpenCL module:
  - `opencv2/core/ocl.hpp`
- At runtime, an OpenCL-capable GPU (AMD, Intel, NVIDIA) and working OpenCL runtime are required.
- Works with AMD Radeon (Polaris/Vega/RDNA), Intel integrated GPUs, and NVIDIA GPUs.
- Generally easier to set up than CUDA on non-NVIDIA hardware.

### Auto-selection

- `acceleration_backend: auto` (default) automatically detects available GPU backends at runtime.
- Priority order: CUDA → OpenCL → CPU
- Falls back gracefully to CPU if no GPU backend is available.

### Accelerated Phases

| Phase | CUDA | OpenCL | GPU work |
|---|---:|---:|---|
| `PREWARP` | Yes | Yes | Full-frame affine warps; four CFA subplanes for OSC |
| `AQMH_MAPS` | Yes | Yes | Pyramid box filters and local-variance/sharpness maps |
| `AQMH_RECONSTRUCTION` | Yes | No | Streaming weighted Welford statistics, masks, sigma clipping, and final accumulation |
| Classic `TILE_RECONSTRUCTION` | Yes | Yes | Weighted sigma clipping, overlap-add, and accumulator normalization |
| `SYNTHETIC_FRAMES` | Yes | Yes | Per-cluster weighted tile reconstruction |
| `STACKING` / resume | Yes | Yes | Sigma-clipped/weighted reduction; RGB channels execute concurrently |

`REGISTRATION` is CPU-only (star detection, matching, transform estimation,
and ECC); GPU processing starts with the subsequent `PREWARP` phase.

CUDA uses one non-default stream per parallel worker. `AQMH_RECONSTRUCTION`
keeps only its accumulators plus one frame/quality map resident, so VRAM usage
does not grow with frame count. AQMH Cherry-Pick `mode: auto_reject` is
implemented for CPU and CUDA; OpenCL currently falls back to CPU for that mode
to preserve semantics. CUDA errors, unsupported operations, or unavailable
runtimes fall back to CPU. Live progress logs report `cpu_workers`, `gpu`, and
the selected `backend`.

### Notes

- Many default distro/Homebrew/OpenCV packages provide CPU-only builds. GPU acceleration requires OpenCV built with CUDA or OpenCL support.
- For NVIDIA GPUs: CUDA backend typically provides better performance than OpenCL.
- For AMD/Intel GPUs: OpenCL is the only supported GPU backend.
- On macOS: OpenCL support depends on OpenCV build; CUDA is not practical.

## Package Install Examples

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake pkg-config \
  libeigen3-dev libopencv-dev libcfitsio-dev libyaml-cpp-dev nlohmann-json3-dev libssl-dev \
  libcurl4-openssl-dev
```

### Linux (Fedora)

```bash
sudo dnf install -y \
  gcc-c++ cmake pkgconf-pkg-config ninja-build \
  eigen3-devel opencv-devel cfitsio-devel yaml-cpp-devel nlohmann-json-devel openssl-devel \
  libcurl-devel
```

### macOS (Homebrew)

```bash
xcode-select --install
brew install cmake ninja pkg-config eigen cfitsio yaml-cpp nlohmann-json openssl curl
brew install opencv
brew upgrade yaml-cpp  # ensure >= 0.8 to avoid linker errors on Intel Macs
```

Notes:

- `ninja` is required for the local GUI3 packaging scripts.
- On macOS 12, the default Homebrew `opencv` formula is currently not supported. The Homebrew-based path therefore effectively requires macOS 15 for OpenCV, unless you provide a separate working OpenCV installation yourself.
- On macOS Intel (x86_64), a linker error referencing `YAML::FpToString` can occur if the installed yaml-cpp is too old. Run `brew upgrade yaml-cpp` to fix it, then delete the build directory and reconfigure.
- The package examples above are sufficient for CPU builds. They do not guarantee GPU acceleration, because the OpenCV package on the host may not include CUDA modules.
- If a downloaded GUI3/release bundle is blocked by Gatekeeper with messages such as "developer cannot be identified" or a bundled `.dylib` cannot be opened, remove the quarantine flag from the extracted release folder with `xattr -dr com.apple.quarantine /path/to/extracted_release` and then start the bundle again.

### Windows

- MinGW/MSYS2: `mingw-w64-x86_64-eigen3`, `mingw-w64-x86_64-opencv`, `mingw-w64-x86_64-cfitsio`, `mingw-w64-x86_64-yaml-cpp`, `mingw-w64-x86_64-nlohmann-json`, `mingw-w64-x86_64-openssl`, `mingw-w64-x86_64-curl`, `mingw-w64-x86_64-pkgconf`
- MSVC/vcpkg: `eigen3`, `opencv`, `cfitsio`, `yaml-cpp`, `nlohmann-json`, `openssl`, `curl`, `pkgconf`

## Build

```bash
cd tile_compile_cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)
```

## Tests

```bash
cd tile_compile_cpp/build
ctest --output-on-failure
```

## Release Build + Packaging

GUI3 release bundles are built by:

- `.github/workflows/release-tile-compile-gui3.yml`

The workflow builds the Qt-free C++ binaries, bundles `web_backend_cpp/` and `web_frontend_v3/`, adds the GUI3 launchers, and creates ZIP artifacts for Linux, Windows, macOS Apple Silicon, and macOS Intel.

Not included by design:

- external Siril catalog data
- external ASTAP binary/data
