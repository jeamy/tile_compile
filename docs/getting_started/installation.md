# Installation

## Option 1: Pre-built Binaries (Recommended)

Download ready-to-use binaries from [GitHub Releases](https://github.com/jeamy/tile_compile/releases):

### GUI3 (Browser Interface — Recommended)

| Platform | Download |
|----------|----------|
| Linux x86_64 (zip) | `tile_compile_gui3-linux-v{version}.zip` |
| macOS Apple Silicon | `tile_compile_gui3-macos-apple-v{version}.zip` |
| macOS Intel | `tile_compile_gui3-macos-intel-v{version}.zip` |
| Windows x64 | `tile_compile_gui3-windows-v{version}.zip` |

### Linux

```bash
# Download latest release
curl -L -o tile_compile.zip \
  https://github.com/jeamy/tile_compile/releases/latest/download/tile_compile_gui3-linux-v0.3.0.zip

# Extract
unzip tile_compile.zip
cd tile_compile_gui3-linux-v0.3.0

# Start GUI3 (browser opens automatically)
./start_gui3.sh  # http://127.0.0.1:8080/ui/
```

### macOS

```bash
# Apple Silicon
curl -L -o tile_compile.zip \
  https://github.com/jeamy/tile_compile/releases/latest/download/tile_compile_gui3-macos-apple-v0.3.0.zip

# Or Intel
curl -L -o tile_compile.zip \
  https://github.com/jeamy/tile_compile/releases/latest/download/tile_compile_gui3-macos-intel-v0.3.0.zip

unzip tile_compile.zip
cd tile_compile_gui3-macos-*/
./start_gui3.command  # Browser opens automatically
```

> **macOS note:** If Gatekeeper blocks the launcher: `System Settings → Privacy & Security`, scroll down and allow the blocked entry.

### Windows

1. Download `tile_compile_gui3-windows-v0.3.0.zip`
2. Extract to desired location
3. Run:
   ```cmd
   start_gui3.bat
   :: Browser opens automatically at http://127.0.0.1:8080/ui/
   ```

> **First launch:** All application files are copied to `~/tilecompile/` (or `%USERPROFILE%\tilecompile\` on Windows). The downloaded archive can be deleted afterwards. On updates, only application files are replaced — user data (runs, catalogs) is preserved.

> Full GUI3 workflow guide: [GUI3 User Guide](../gui3_user_guide_en.md)

---

## Option 2: Build from Source

### Prerequisites

- C++20 compiler (GCC 13+, Clang 16+, MSVC 2022+ 17.8+)
- CMake 3.20+
- OpenCV 4.x
- CFITSIO
- yaml-cpp
- Eigen3
- nlohmann/json
- spdlog (optional)
- CLI11 (optional)
- Catch2 (optional, for tests)

---

## Linux

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y build-essential cmake git \
  libopencv-dev libcfitsio-dev libyaml-cpp-dev \
  libeigen3-dev nlohmann-json3-dev libspdlog-dev \
  libcli11-dev catch2
```

### Fedora / RHEL / Rocky / AlmaLinux

```bash
sudo dnf install -y gcc gcc-c++ cmake git \
  opencv-devel cfitsio-devel yaml-cpp-devel \
  eigen3-devel nlohmann-json-devel spdlog-devel \
  cli11-devel catch2-devel
```

> **Fedora note:** `nlohmann-json-devel` is in the standard repos since Fedora 38. On older releases install via `pip install nlohmann-json` or build from source.

### Arch / Manjaro

```bash
sudo pacman -S base-devel cmake git \
  opencv cfitsio yaml-cpp eigen nlohmann-json \
  spdlog cli11 catch2
```

---

## Build Pipeline

```bash
cd tile_compile_cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
ctest --output-on-failure
```

### CUDA 13 with OpenCV CUDA 13

When using an OpenCV build that was compiled against CUDA 13, configure
`tile_compile_cpp` with the matching OpenCV and CUDA paths. Mixing an
OpenCV-CUDA build for one CUDA version with a different CUDA toolkit can make
CMake fail during `find_package(OpenCV)`.

Example for OpenCV installed in `/opt/opencv-4.11-cuda13` and CUDA installed in
`/usr/local/cuda-13.0`:

```bash
rm -rf tile_compile_cpp/build
cmake -S tile_compile_cpp -B tile_compile_cpp/build \
  -DOpenCV_DIR=/opt/opencv-4.11-cuda13/lib64/cmake/opencv4 \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-13.0 \
  -DCUDA_NVCC_EXECUTABLE=/usr/local/cuda-13.0/bin/nvcc \
  -DTILE_COMPILE_NVCC_EXECUTABLE=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DTILE_COMPILE_ENABLE_CUDA=ON
cmake --build tile_compile_cpp/build -j$(nproc)
```

The configuration summary should show:

```text
TILE_COMPILE_ENABLE_CUDA: ON
TILE_COMPILE_WITH_CUDA: ON
CUDA nvcc: /usr/local/cuda-13.0/bin/nvcc
OpenCV: 4.11.0
```

If CMake reports an unsuitable CUDA version, remove the build directory before
reconfiguring so stale `CUDA_*` cache entries cannot point to an older toolkit.

Install to system:

```bash
sudo cmake --install .
```

---

## Docker

Pre-built environment, no host dependencies needed:

```bash
./tile_compile_cpp/scripts/docker_compile_and_run.sh build-image
./tile_compile_cpp/scripts/docker_compile_and_run.sh run-shell
```

Inside the container:

```bash
./tile_compile_runner run --config tile_compile.yaml --input-dir /mnt/input --runs-dir /mnt/runs
```

---

## Web Backend (Optional, for development)

The GUI3 release bundle includes a pre-built backend. To build the backend manually:

```bash
cd web_backend_cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# Start backend
../../start_backend.sh
```

Open http://127.0.0.1:8080/ui/

---

## Documentation Tools (Optional)

Required only for building the documentation site.

### Python tools (all platforms)

```bash
pip install mkdocs mkdocs-material mike
```

### Doxygen + Graphviz

| Distro | Command |
|--------|---------|
| Ubuntu/Debian | `sudo apt install doxygen graphviz` |
| Fedora/RHEL | `sudo dnf install doxygen graphviz` |
| Arch | `sudo pacman -S doxygen graphviz` |
| macOS | `brew install doxygen graphviz` |
| Windows | `choco install doxygen.portable graphviz` |

### Generate documentation

```bash
# C++ API (Doxygen)
cd tile_compile_cpp
doxygen Doxyfile

# Full site (MkDocs)
cd ..
mkdocs serve    # http://127.0.0.1:8000
mkdocs build    # output: site/
```

---

## Verification

```bash
# Check executables
./tile_compile_runner --help
./tile_compile_cli --help

# Validate config
./tile_compile_cli validate-config --path tile_compile.yaml

# Quick scan
./tile_compile_cli scan /path/to/lights --frames-min 30
```

For the full GUI3 workflow (scan, parameters, run, results), see the [GUI3 User Guide](../gui3_user_guide_en.md).
