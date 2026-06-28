# Quick Start

## Installation

### Option 1: Pre-built Binaries (Fastest)

Download from [GitHub Releases](https://github.com/jeamy/tile_compile/releases):

```bash
# Linux (GUI3 with web interface)
curl -L -o tile_compile.zip \
  https://github.com/jeamy/tile_compile/releases/latest/download/tile_compile_gui3-linux-v0.3.9.zip
unzip tile_compile.zip
cd tile_compile_gui3-linux-v0.3.9

# Start GUI3 (browser opens automatically)
./start_gui3.sh  # http://127.0.0.1:8080/ui/
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

# Optional: Web backend (from tile_compile_cpp/build)
cd ../../web_backend_cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

## GUI3 Workflow

1. **Scan Input** — Tab *Processing → Input & Scan*: Select FITS lights folder, optional calibration frames, click *Start Scan*
2. **Adjust Parameters** — Tab *Processing → Parameter*: Load example config or customize, validate and save
3. **Start & Monitor** — Tab *Processing → Run Monitor*: Start run, track phase progress in real time
4. **View Results** — Results in `runs/<run_id>/outputs/`, generate diagnostic report via *Generate Stats*

Full guide: [GUI3 User Guide](../gui3_user_guide_en.md)

When `runtime_limits.acceleration_backend: auto` is used, live progress entries
show the effective execution resources, for example:

```text
PREWARP | progress (50%) | cpu_workers=8 gpu=yes backend=opencv_cuda
REGISTRATION | progress (50%) | cpu_workers=8 gpu=no backend=cpu
```

The GPU is optional; `gpu=no` for REGISTRATION is expected. Verify all selected
phase backends in `runs/<run_id>/artifacts/acceleration_context.json`.

## CLI First Run

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
