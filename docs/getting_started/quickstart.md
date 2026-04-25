# Quick Start

## Installation

### Option 1: Pre-built Binaries (Fastest)

Download from [GitHub Releases](https://github.com/jeamy/tile_compile/releases):

```bash
# Linux (GUI2 with web interface)
curl -L -o tile_compile.zip \
  https://github.com/jeamy/tile_compile/releases/latest/download/tile_compile_gui2-linux-v0.2.4.zip
unzip tile_compile.zip
cd tile_compile_gui2-linux-v0.2.4
```

Or use AppImage (no extraction needed):
```bash
curl -L -o tile_compile.AppImage \
  https://github.com/jeamy/tile_compile/releases/latest/download/tile_compile_gui2-linux-x86_64-v0.2.4.AppImage
chmod +x tile_compile.AppImage && ./tile_compile.AppImage
```

### Option 2: Build from Source

```bash
# Clone repository
git clone https://github.com/jeamy/tile_compile.git
cd tile_compile

# Build C++ pipeline
cd tile_compile_cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# Optional: GUI2 backend
cd ../web_backend_cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

## First Run

```bash
./tile_compile_cpp/build/tile_compile_runner \
  run \
  --config tile_compile_cpp/tile_compile.yaml \
  --input-dir /path/to/lights \
  --runs-dir ./runs
```

## Resume a Run

```bash
./tile_compile_cpp/build/tile_compile_runner \
  resume \
  --run-dir runs/<run_id> \
  --from-phase PCC
```

See [CLI Reference](cli_reference.md) for all commands.
