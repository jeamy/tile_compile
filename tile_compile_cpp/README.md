# Tile-Compile C++ Version

Dies ist die C++ Implementierung des Tile-Compile Backends (in Entwicklung).

## Status

**In Entwicklung** - Siehe `doc/c-port/` für den Portierungsplan.

## Voraussetzungen

### Fedora (Haupt-Entwicklungsplattform)

```bash
sudo dnf install -y \
    eigen3-devel \
    opencv-devel \
    cfitsio-devel \
    yaml-cpp-devel \
    json-devel \
    openssl-devel \
    cmake \
    gcc-c++ \
    make
```

### Ubuntu/Debian

```bash
sudo apt install -y \
    libeigen3-dev \
    libopencv-dev \
    libcfitsio-dev \
    libyaml-cpp-dev \
    nlohmann-json3-dev \
    libssl-dev \
    cmake \
    g++ \
    make
```

## Build

```bash
cd tile_compile_cpp
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

## GUI starten

Die GUI benötigt Python mit PySide6 (kann aus der Python-Version verwendet werden):

```bash
./start_gui.sh
```

## Optional: Matplotlib für Report-Previews

Für die Artefakt-Reports können (optional) Log-Previews und Histogramme aus FITS erzeugt werden.
Dafür wird `matplotlib` im Python-Umfeld benötigt.

```bash
python3 -m pip install matplotlib
```

## CLI verwenden (nach Build)

```bash
# Pipeline ausführen
./build/tile_compile_runner run \
    --config config.yaml \
    --input-dir /pfad/zu/frames \
    --runs-dir runs

# Run fortsetzen
./build/tile_compile_runner resume \
    --run-dir runs/20240119_123456_abc12345 \
    --from-phase 5
```

## Struktur

```
tile_compile_cpp/
├── gui/                    # PyQt6 GUI (ruft C++ Backend auf)
├── include/                # C++ Header (TODO)
├── src/                    # C++ Implementierung (TODO)
├── apps/                   # CLI-Programme (TODO)
├── tests/                  # Unit-Tests (TODO)
├── build/                  # Build-Verzeichnis
├── CMakeLists.txt          # Build-Konfiguration (TODO)
└── start_gui.sh            # GUI-Startscript
```

