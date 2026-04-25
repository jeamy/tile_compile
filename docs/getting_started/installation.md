# Installation

## Option 1: Pre-built Binaries (Recommended)

Download ready-to-use binaries from [GitHub Releases](https://github.com/jeamy/tile_compile/releases):

### GUI2 (CLI + Web Interface)

| Platform | Download |
|----------|----------|
| Linux x86_64 (zip) | `tile_compile_gui2-linux-v{version}.zip` |
| Linux x86_64 (AppImage) | `tile_compile_gui2-linux-x86_64-v{version}.AppImage` |
| macOS Apple Silicon | `tile_compile_gui2-macos-apple-v{version}.zip` |
| macOS Intel | `tile_compile_gui2-macos-intel-v{version}.zip` |
| Windows x64 | `tile_compile_gui2-windows-v{version}.zip` |

### Linux (GUI2 zip)

```bash
# Download latest release (replace v0.2.4 with latest)
curl -L -o tile_compile.zip \
  https://github.com/jeamy/tile_compile/releases/latest/download/tile_compile_gui2-linux-v0.2.4.zip

# Extract
unzip tile_compile.zip
cd tile_compile_gui2-linux-v0.2.4

# Verify
./tile_compile_runner --version
./start_backend.sh  # Start GUI2 backend
```

### Linux (AppImage)

```bash
curl -L -o tile_compile.AppImage \
  https://github.com/jeamy/tile_compile/releases/latest/download/tile_compile_gui2-linux-x86_64-v0.2.4.AppImage
chmod +x tile_compile.AppImage
./tile_compile.AppImage
```

### macOS

```bash
# Apple Silicon
curl -L -o tile_compile.zip \
  https://github.com/jeamy/tile_compile/releases/latest/download/tile_compile_gui2-macos-apple-v0.2.4.zip

# Or Intel
curl -L -o tile_compile.zip \
  https://github.com/jeamy/tile_compile/releases/latest/download/tile_compile_gui2-macos-intel-v0.2.4.zip

unzip tile_compile.zip
cd tile_compile_gui2-macos-*/
./tile_compile_runner --version
```

### Windows

1. Download `tile_compile_gui2-windows-v0.2.4.zip`
2. Extract to desired location
3. Run:
   ```cmd
   tile_compile_runner.exe --version
   start_backend.bat  :: Optional: start GUI2
   ```

---

## Option 2: Build from Source

### Prerequisites

- C++17 compiler (GCC 11+, Clang 14+, MSVC 2022+)
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

## GUI2 Backend (Optional)

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
