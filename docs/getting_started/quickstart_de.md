# Schnellstart

## Installation

### Option 1: Vorgefertigte Binaries (Schnellster Weg)

Von [GitHub Releases](https://github.com/jeamy/tile_compile/releases) herunterladen:

```bash
# Linux (GUI3 mit Web-Oberfläche)
curl -L -o tile_compile.zip \
  https://github.com/jeamy/tile_compile/releases/latest/download/tile_compile_gui3-linux-v0.3.9.zip
unzip tile_compile.zip
cd tile_compile_gui3-linux-v0.3.9

# GUI3 starten (Browser öffnet sich automatisch)
./start_gui3.sh  # http://127.0.0.1:8080/ui/
```

### Option 2: Aus dem Quellcode bauen

```bash
# Repository klonen
git clone https://github.com/jeamy/tile_compile.git
cd tile_compile

# C++-Pipeline bauen
cd tile_compile_cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# Optional: Web-Backend (aus tile_compile_cpp/build)
cd ../../web_backend_cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

## GUI3-Workflow

1. **Input scannen** — Tab *Processing → Input & Scan*: FITS-Lights-Ordner auswählen, optional Kalibrierungsframes, auf *Start Scan* klicken
2. **Parameter anpassen** — Tab *Processing → Parameter*: Beispiel-Konfiguration laden oder anpassen, validieren und speichern
3. **Start & Überwachung** — Tab *Processing → Run Monitor*: Run starten, Phasenfortschritt in Echtzeit verfolgen
4. **Ergebnisse ansehen** — Ergebnisse in `runs/<run_id>/outputs/`, Diagnosebericht über *Generate Stats* erzeugen

Vollständige Anleitung: [GUI3 Benutzerhandbuch](../gui3_user_guide_de.md)

Wenn `runtime_limits.acceleration_backend: auto` verwendet wird, zeigen die
Live-Fortschrittsmeldungen die effektiv genutzten Ausführungsressourcen, z.B.:

```text
PREWARP | progress (50%) | cpu_workers=8 gpu=yes backend=opencv_cuda
REGISTRATION | progress (50%) | cpu_workers=8 gpu=no backend=cpu
```

Die GPU ist optional; `gpu=no` bei REGISTRATION ist normal. Alle ausgewählten
Phasen-Backends in `runs/<run_id>/artifacts/acceleration_context.json` überprüfen.

## CLI — Erster Run

```bash
./tile_compile_cpp/build/tile_compile_runner \
  run \
  --config tile_compile_cpp/tile_compile.yaml \
  --input-dir /pfad/zu/lights \
  --runs-dir ./runs
```

## Run fortsetzen (Resume)

```bash
./tile_compile_cpp/build/tile_compile_runner \
  resume \
  --run-dir runs/<run_id> \
  --from-phase PCC
```

Siehe [CLI-Referenz](cli_reference.md) für alle Befehle.
