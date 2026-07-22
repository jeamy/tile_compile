# Installation

## Option 1: Vorgefertigte Binaries (Empfohlen)

Gebrauchsfertige Binaries von [GitHub Releases](https://github.com/jeamy/tile_compile/releases) herunterladen:

### GUI3 (Browser-Oberfläche — Empfohlen)

| Plattform | Download |
|-----------|----------|
| Linux x86_64 (zip) | `tile_compile_gui3-linux-v{version}.zip` |
| macOS Apple Silicon | `tile_compile_gui3-macos-apple-v{version}.zip` |
| macOS Intel | `tile_compile_gui3-macos-intel-v{version}.zip` |
| Windows x64 | `tile_compile_gui3-windows-v{version}.zip` |

### Linux

```bash
# Neueste Release herunterladen
curl -L -o tile_compile.zip \
  https://github.com/jeamy/tile_compile/releases/latest/download/tile_compile_gui3-linux-v0.3.9.zip

# Entpacken
unzip tile_compile.zip
cd tile_compile_gui3-linux-v0.3.9

# GUI3 starten (Browser öffnet sich automatisch)
./start_gui3.sh  # http://127.0.0.1:8080/ui/
```

### macOS

```bash
# Apple Silicon
curl -L -o tile_compile.zip \
  https://github.com/jeamy/tile_compile/releases/latest/download/tile_compile_gui3-macos-apple-v0.3.9.zip

# Oder Intel
curl -L -o tile_compile.zip \
  https://github.com/jeamy/tile_compile/releases/latest/download/tile_compile_gui3-macos-intel-v0.3.9.zip

unzip tile_compile.zip
cd tile_compile_gui3-macos-*/
./start_gui3.command  # Browser öffnet sich automatisch
```

> **macOS-Hinweis:** Falls Gatekeeper den Launcher blockiert: `Systemeinstellungen → Datenschutz & Sicherheit`, nach unten scrollen und den blockierten Eintrag erlauben.

### Windows

1. `tile_compile_gui3-windows-v0.3.9.zip` herunterladen
2. An gewünschten Ort entpacken
3. Ausführen:
   ```cmd
   start_gui3.bat
   :: Browser öffnet sich automatisch unter http://127.0.0.1:8080/ui/
   ```

> **Erster Start:** Alle Anwendungsdateien werden nach `~/tilecompile/` (oder `%USERPROFILE%\tilecompile\` auf Windows) kopiert. Das heruntergeladene Archiv kann danach gelöscht werden. Bei Updates werden nur Anwendungsdateien ersetzt — Benutzerdaten (Runs, Kataloge) bleiben erhalten.

> Vollständige GUI3-Workflow-Anleitung: [GUI3 Benutzerhandbuch](../gui3_user_guide_de.md)

---

## Option 2: Aus dem Quellcode bauen

### Voraussetzungen

- C++20-Compiler (GCC 13+, Clang 16+, MSVC 2022+ 17.8+)
- CMake 3.21+
- OpenCV 4.x
- CFITSIO
- yaml-cpp
- Eigen3
- nlohmann/json
- spdlog (optional)
- CLI11 (optional)
- Catch2 (optional, für Tests)

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

> **Fedora-Hinweis:** `nlohmann-json-devel` ist seit Fedora 38 in den Standard-Repos. Bei älteren Releases über `pip install nlohmann-json` oder aus dem Quellcode bauen.

### Arch / Manjaro

```bash
sudo pacman -S base-devel cmake git \
  opencv cfitsio yaml-cpp eigen nlohmann-json \
  spdlog cli11 catch2
```

---

## Pipeline bauen

```bash
cd tile_compile_cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
ctest --output-on-failure
```

### GPU-Beschleunigung — Voraussetzungen

GPU-Beschleunigung ist optional. CPU-only-Builds werden vollständig unterstützt.
Der Laufzeit-Backend wird über `runtime_limits.acceleration_backend` ausgewählt
(`auto`, `opencv_cuda`, `opencv_opencl` oder `cpu`). `auto` versucht CUDA, dann
OpenCL, dann CPU.

Für NVIDIA CUDA muss OpenCV mit `core/cuda`, `cudawarping`, `cudaarithm` und
`cudafilters` gebaut sein; die Installation eines CUDA-Toolkits neben einem
CPU-only-OpenCV-Paket reicht nicht. Für OpenCL muss OpenCV `core/ocl` bieten
und der Host eine funktionierende OpenCL ICD/Runtime bereitstellen.

| Phase | CUDA | OpenCL | Beschleunigte Operation |
|---|---:|---:|---|
| `PREWARP` | Ja | Ja | Full-Frame/CFA-Affine-Warping |
| `AQMH_MAPS` | Ja | Ja | Lokale-Varianz- und Pyramiden-Filter |
| `AQMH_RECONSTRUCTION` | Ja | Nein | Streaming-Welford-Statistiken und Sigma-Clipping |
| Klassische `TILE_RECONSTRUCTION` | Ja | Ja | Sigma-Clipping und Overlap-Add |
| `SYNTHETIC_FRAMES` | Ja | Ja | Cluster-Tile-Rekonstruktion |
| `STACKING` und Resume | Ja | Ja | Gewichtete/Sigma-geclippte Reduktion und paralleles RGB |

`REGISTRATION` bleibt CPU-only; GPU-Verarbeitung beginnt bei `PREWARP`.

AQMH Cherry-Pick fällt derzeit auf CPU zurück. CUDA-Rekonstruktion verarbeitet
ein Frame und eine Quality-Map gleichzeitig, daher skaliert der VRAM-Verbrauch
nicht mit der Anzahl der Input-Frames. Laufzeit-Logs zeigen `cpu_workers`,
`gpu` und `backend`.

### CUDA 13 mit OpenCV CUDA 13

Bei Verwendung eines OpenCV-Builds, der gegen CUDA 13 kompiliert wurde,
`tile_compile_cpp` mit den passenden OpenCV- und CUDA-Pfaden konfigurieren.
Das Mischen eines OpenCV-CUDA-Builds für eine CUDA-Version mit einem anderen
CUDA-Toolkit kann dazu führen, dass CMake bei `find_package(OpenCV)` scheitert.

Beispiel für OpenCV in `/opt/opencv-4.11-cuda13` und CUDA in
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

Die Konfigurationszusammenfassung sollte zeigen:

```text
TILE_COMPILE_ENABLE_CUDA: ON
TILE_COMPILE_WITH_CUDA: ON
CUDA nvcc: /usr/local/cuda-13.0/bin/nvcc
OpenCV: 4.11.0
```

Zur Laufzeit zeichnet `artifacts/acceleration_context.json` das erkannte Gerät
und den gewählten Backend für jede unterstützte Phase auf. Dies ist die
autoritative Methode, um einen echten GPU-Pfad von einem kontrollierten
CPU-Fallback zu unterscheiden.

Falls CMake eine ungeeignete CUDA-Version meldet, das Build-Verzeichnis vor der
Neukonfiguration entfernen, damit veraltete `CUDA_*`-Cache-Einträge nicht auf
ein älteres Toolkit verweisen.

Installation ins System:

```bash
sudo cmake --install .
```

---

## Docker

Vorgefertigte Umgebung, keine Host-Abhängigkeiten erforderlich:

```bash
./tile_compile_cpp/scripts/docker_compile_and_run.sh build-image
./tile_compile_cpp/scripts/docker_compile_and_run.sh run-shell
```

Im Container:

```bash
./tile_compile_runner run --config tile_compile.yaml --input-dir /mnt/input --runs-dir /mnt/runs
```

---

## Web-Backend (Optional, für Entwicklung)

Das GUI3-Release-Bundle enthält ein vorgebautes Backend. Zum manuellen Bauen:

```bash
cd web_backend_cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# Backend starten
../../start_backend.sh
```

http://127.0.0.1:8080/ui/ öffnen

---

## Dokumentations-Tools (Optional)

Nur für den Bau der Dokumentations-Website erforderlich.

### Python-Tools (alle Plattformen)

```bash
pip install mkdocs mkdocs-material mike
```

### Doxygen + Graphviz

| Distro | Befehl |
|--------|--------|
| Ubuntu/Debian | `sudo apt install doxygen graphviz` |
| Fedora/RHEL | `sudo dnf install doxygen graphviz` |
| Arch | `sudo pacman -S doxygen graphviz` |
| macOS | `brew install doxygen graphviz` |
| Windows | `choco install doxygen.portable graphviz` |

### Dokumentation erzeugen

```bash
# C++ API (Doxygen)
cd tile_compile_cpp
doxygen Doxyfile

# Volle Website (MkDocs)
cd ..
mkdocs serve    # http://127.0.0.1:8000
mkdocs build    # Ausgabe: site/
```

---

## Verifikation

```bash
# Executables prüfen
./tile_compile_runner --help
./tile_compile_cli --help

# Konfiguration validieren
./tile_compile_cli validate-config --path tile_compile.yaml

# Schneller Scan
./tile_compile_cli scan /pfad/zu/lights --frames-min 30
```

Für den vollständigen GUI3-Workflow (Scan, Parameter, Run, Ergebnisse) siehe das [GUI3 Benutzerhandbuch](../gui3_user_guide_de.md).
