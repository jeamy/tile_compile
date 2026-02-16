# Tile-Compile

Tile-Compile ist ein Toolkit für **tile-basierte Qualitätsrekonstruktion** astronomischer Bildstapel (Methodik v3.2).

Wir stellen eine neuartige Methodik zur Rekonstruktion hochwertiger astronomischer Bilder aus Kurzzeitbelichtungs-Deep-Sky-Datensätzen vor. Konventionelle Stacking-Methoden beruhen häufig auf einer binären Frame-Auswahl ("Lucky Imaging"), wodurch erhebliche Teile der gesammelten Frames verworfen werden. Unser Ansatz, **Tile-Based Quality Reconstruction (TBQR)**, ersetzt diese starre Frame-Auswahl durch ein robustes räumlich-zeitliches Qualitätsmodell. Indem wir Frames in lokale Tiles zerlegen und die Qualität entlang zweier orthogonaler Achsen modellieren — globale atmosphärische Transparenz/Rauschen und lokale strukturelle Schärfe — rekonstruieren wir ein Signal, das an jedem Pixel physikalisch und statistisch optimal ist. Wir zeigen, dass diese Methode die volle photometrische Tiefe des Datensatzes bewahrt und zugleich eine überlegene Auflösungsverbesserung gegenüber traditionellen Referenz-Stacks erzielt.

Während die Methodik ursprünglich entwickelt wurde, um die spezifischen Herausforderungen von Kurzzeitbelichtungsdaten moderner Smart-Teleskope (z.B. DWARF, Seestar) zu adressieren, macht ihre architektonische Flexibilität sie ebenso leistungsfähig für konventionelle astronomische Setups. Der umfangreiche Satz abstimmbarer Parameter — von adaptiver Tile-Größe und Kreuzkorrelationsschwellen bis hin zu ausgefeilter Clustering-Logik — ermöglicht eine präzise Optimierung der Pipeline für ein breites Spektrum optischer Systeme und atmosphärischer Bedingungen.

> **Praxis-Hinweis:** Die Pipeline ist in erster Linie für Datensätze mit vielen nutzbaren Frames optimiert. Bei sehr kleinen Frame-Anzahlen oder bei stark gemischter Frame-Qualität innerhalb eines Stacks können in schwierigen Fällen sichtbare Kachelmuster auftreten. Dem kann man häufig entgegenwirken, indem man verschiedene Konfigurationseinstellungen testet (insbesondere Parameter für Registrierung, Tile-Geometrie und Rekonstruktion). Siehe dazu die Beispielprofile unter `tile_compile_cpp/examples/` sowie `tile_compile_cpp/examples/README.md`.

> **Hinweis:** Dies ist experimentelle Software, die primär für die Verarbeitung von Bildern von Smart-Teleskopen entwickelt wurde (z.B. DWARF, Seestar, ZWO SeeStar, usw.). Obwohl sie für die allgemeine astronomische Bildverarbeitung konzipiert ist, wurde sie für die spezifischen Eigenschaften und Herausforderungen von Smart-Teleskop-Daten optimiert.

## Dokumentation (v3.2)

- Methodik (normativ): `doc/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.2.2_en.md`
- Prozessfluss (Implementierung): `doc/v3/process_flow/`
- Englische Schritt-für-Schritt-Anleitung: `doc/v3/tbqr_step_by_step_en.md`
- Englisches Haupt-README: `README.md`

Aus einem Verzeichnis mit FITS-Lights kann die Pipeline:

- Lights optional **kalibrieren** (Bias/Dark/Flat)
- Frames mit robuster 6-stufiger Kaskade **registrieren**
- **globale und lokale (Tile-)Qualitätsmetriken** berechnen
- Bild via tile-gewichteter Overlap-Add-Rekonstruktion erzeugen
- optional Frame-"Zustände" clustern und synthetische Frames erstellen
- Ergebnis via **Sigma-Clip** stacken
- OSC-Daten **debayern**
- **Astrometrie** (ASTAP/WCS) ausführen
- **photometrische Farbkalibrierung** (PCC) anwenden
- finale Ausgaben plus **Diagnose-Artefakte** (JSON) schreiben

## Aktive Version

| Version | Verzeichnis | Status | Backend |
|---------|-------------|--------|---------|
| C++ | `tile_compile_cpp/` | Aktiv (v3.2) | C++17 + Eigen + OpenCV + cfitsio + yaml-cpp |

## Pipeline-Phasen

| ID | Phase | Beschreibung |
|----|-------|-------------|
| 0 | SCAN_INPUT | Input-Erkennung, Modus-Erkennung, Linearitätsprüfung, Festplattenplatz-Precheck |
| 1 | REGISTRATION | Kaskadierte globale Registrierung + CFA-bewusstes Prewarp |
| 2 | CHANNEL_SPLIT | Metadatenphase (Kanalmodell) |
| 3 | NORMALIZATION | Lineare hintergrundbasierte Normalisierung |
| 4 | GLOBAL_METRICS | Globale Frame-Metriken und Gewichte |
| 5 | TILE_GRID | Adaptive Tile-Geometrie |
| 6 | LOCAL_METRICS | Lokale Tile-Metriken und lokale Gewichte |
| 7 | TILE_RECONSTRUCTION | Gewichtete Overlap-Add Rekonstruktion |
| 8 | STATE_CLUSTERING | Optionale Zustands-Clustering |
| 9 | SYNTHETIC_FRAMES | Optionale Erzeugung synthetischer Frames |
| 10 | STACKING | Finales lineares Stacking |
| 11 | DEBAYER | OSC-Demosaicing zu RGB (MONO-Pass-Through) |
| 12 | ASTROMETRY | Astrometrisches Solving / WCS |
| 13 | PCC | Photometrische Farbkalibrierung |
| 14 | DONE | Finaler Status (`ok` oder `validation_failed`) |

Detaillierte Phasen-Dokumentation: `doc/v3/process_flow/`

## Registrierungskaskade (Fallback-Strategie)

| Stufe | Methode | Typischer Anwendungsfall |
|-------|--------|------------------|
| 1 | Primäre Engine (`triangle_star_matching`) | Normale sternreiche Frames |
| 2 | Trail-Endpoint-Registrierung | Startrails / rotationsstarke Daten |
| 3 | AKAZE-Feature-Matching | Allgemeiner Feature-Fallback |
| 4 | Robust Phase+ECC | Wolken/Nebel mit größeren Transformationen |
| 5 | Hybrid Phase+ECC | Fälle mit schwachem Stern-Matching |
| 6 | Identity-Fallback | Letzter Ausweg (CC=0, Frame wird beibehalten) |

## Konfiguration

- Hauptkonfigurationsdatei: `tile_compile.yaml`
- Schemas: `tile_compile.schema.json`, `tile_compile.schema.yaml`
- Referenzdokument: `doc/v3/configuration_reference.md`

### Beispielprofile

Vollständige eigenständige Beispielkonfigurationen sind verfügbar unter `tile_compile_cpp/examples/`:

- `tile_compile.full_mode.example.yaml`
- `tile_compile.reduced_mode.example.yaml`
- `tile_compile.emergency_mode.example.yaml`
- `tile_compile.smart_telescope_dwarf_seestar.example.yaml`
- `tile_compile.canon_low_n_high_quality.example.yaml`
- `tile_compile.mono_full_mode.example.yaml`
- `tile_compile.mono_small_n_anti_grid.example.yaml` (empfohlen für MONO-Datensätze mit geringer Frame-Anzahl, z.B. ~10..40, zur Reduzierung von Tile-Muster-Risiko)

Siehe auch: `tile_compile_cpp/examples/README.md`

## Schnellstart (C++)

Für eine vollständige anfängerfreundliche Anleitung siehe:
`doc/v3/tbqr_step_by_step_en.md`

### Build-Voraussetzungen

- CMake >= 3.20
- C++17 Compiler (GCC 11+ oder Clang 14+)
- OpenCV >= 4.5
- Eigen3
- cfitsio
- yaml-cpp
- nlohmann-json

#### Paket-Installationsbeispiele

Linux (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake pkg-config \
  libeigen3-dev libopencv-dev libcfitsio-dev libyaml-cpp-dev nlohmann-json3-dev libssl-dev \
  qt6-base-dev qt6-tools-dev libgl1-mesa-dev
```

macOS (Homebrew, Kernbibliotheken):

```bash
brew install cmake pkg-config eigen opencv cfitsio yaml-cpp nlohmann-json openssl
```

Windows:

- MinGW/MSYS2: `mingw-w64-x86_64-eigen3`, `mingw-w64-x86_64-opencv`, `mingw-w64-x86_64-cfitsio`, `mingw-w64-x86_64-yaml-cpp`, `mingw-w64-x86_64-nlohmann-json`, `mingw-w64-x86_64-openssl`, `mingw-w64-x86_64-pkgconf`
- MSVC/vcpkg: `eigen3`, `opencv`, `cfitsio`, `yaml-cpp`, `nlohmann-json`, `openssl`, `pkgconf`

### Kompilieren

```bash
cd tile_compile_cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Release-Build-Skripte (portable App-Bundles)

> **Warnung:** Die Build-Skripte sind experimentelle Versionn. Nutzung auf eigene Gefahr.

Im Verzeichnis `tile_compile_cpp/` stehen plattformspezifische Release-Skripte bereit:

- Linux (nativ): `build_linux_release.sh`
- Linux (Docker Ubuntu 20.04 / glibc 2.31): `build_linux_release_docker_ubuntu2004.sh`
- Linux (AppImage): `build_linux_appimage.sh`
- Linux (AppImage Docker): `build_linux_appimage_docker.sh`
- macOS: `build_macos_release.sh`
- Windows: `build_windows_release.bat` (erkennt MSYS2 automatisch, falls installiert)

**Linux-Docker-Wrapper (empfohlen für breite Linux-Kompatibilität):**

```bash
cd tile_compile_cpp
bash build_linux_release_docker_ubuntu2004.sh
# optional: Image-Build überspringen
bash build_linux_release_docker_ubuntu2004.sh --skip-build
```

**Linux AppImage (portable Single-File-Executable):**

```bash
cd tile_compile_cpp
# Docker 
bash build_linux_appimage_docker.sh
```

Das AppImage ist eine portable Single-File-Executable, die auf den meisten Linux-Distributionen läuft (ohne Installation).

Die Release-Ausgaben liegen unter `tile_compile_cpp/dist/`:

- Linux: `dist/linux/` + `dist/tile_compile_cpp-linux-release.zip`
- Windows: `dist/windows/` + `dist/tile_compile_cpp-windows-release.zip`
- macOS: `dist/macos/tile_compile_gui.app` + optional `dist/tile_compile_cpp-macos.dmg`

Enthaltene Laufzeitdateien in den Release-Bundles:

- Executables (`tile_compile_gui`, `tile_compile_runner`, `tile_compile_cli`)
- GUI-Laufzeitdateien (`gui_cpp/constants.js`, `gui_cpp/styles.qss`)
- Konfiguration + Schemas (`tile_compile.yaml`, `tile_compile.schema.yaml`, `tile_compile.schema.json`)
- Beispielprofile (`examples/`)

**Hinweis:** Falls im Release-Paket keine YAML-Konfigurationen enthalten sind, verwende die Beispielprofile unter `examples/` als Vorlage und übernimm die gewünschten Optionen in deine eigene `tile_compile.yaml`.

Bewusst nicht enthalten:

- externe Siril-Katalogdaten
- externe ASTAP-Binary/Daten

Windows-Hinweis:

- Das Build-Script erkennt MSYS2-Installationen unter `C:\msys64\mingw64` (oder `ucrt64`/`clang64`) automatisch und setzt `CMAKE_PREFIX_PATH` entsprechend.
- Falls MSYS2 nicht installiert ist, Abhängigkeiten installieren via:
  - **Option A (MinGW)**: MSYS2 von https://www.msys2.org/ installieren, dann in der MSYS2 MinGW64-Shell:
    ```bash
    pacman -S --needed mingw-w64-x86_64-toolchain mingw-w64-x86_64-cmake mingw-w64-x86_64-pkgconf
    pacman -S --needed mingw-w64-x86_64-eigen3 mingw-w64-x86_64-opencv mingw-w64-x86_64-cfitsio mingw-w64-x86_64-yaml-cpp mingw-w64-x86_64-nlohmann-json mingw-w64-x86_64-openssl
    ```
  - **Option B (MSVC)**: vcpkg installieren und `VCPKG_ROOT` setzen, dann:
    ```bat
    vcpkg install eigen3:x64-windows opencv4:x64-windows cfitsio:x64-windows yaml-cpp:x64-windows nlohmann-json:x64-windows openssl:x64-windows qt6:x64-windows
    ```

macOS-Hinweis:

- Auf älteren macOS-Versionen kann Homebrew-`qt` mindestens Ventura voraussetzen und die Installation fehlschlagen.
- In diesem Fall Qt6 über den Qt Online Installer installieren (z.B. unter `~/Qt/<version>/macos`) und optional setzen:

```bash
export CMAKE_PREFIX_PATH="$HOME/Qt/<version>/macos"
export Qt6_DIR="$HOME/Qt/<version>/macos/lib/cmake/Qt6"
```

### Docker Build + Run (empfohlen für isolierte Umgebungen)

Ein Hilfsskript ist verfügbar unter:
`tile_compile_cpp/scripts/docker_compile_and_run.sh`

Was es tut:

- `build-image`: baut ein Docker-Image und kompiliert `tile_compile_cpp` im Container
- `run-shell`: startet eine interaktive Shell im kompilierten Container
- `run-app`: führt `tile_compile_runner` direkt im Container aus

Standard-Volume-Mapping für Runs:

- Host: `tile_compile_cpp/runs`
- Container: `/workspace/tile_compile_cpp/runs`

Beispiele:

```bash
# Docker-Image bauen und im Container kompilieren
./tile_compile_cpp/scripts/docker_compile_and_run.sh build-image

# interaktive Shell im Container öffnen
./tile_compile_cpp/scripts/docker_compile_and_run.sh run-shell

# Pipeline im Container ausführen
./tile_compile_cpp/scripts/docker_compile_and_run.sh run-app -- run \
  --config /mnt/config/tile_compile.yaml \
  --input-dir /mnt/input \
  --runs-dir /workspace/tile_compile_cpp/runs
```

Verwende `run-shell`, wenn du zusätzliche Mounts benötigst (z.B. Config/Input-Verzeichnisse) und starte den Runner dann manuell.

#### Windows-Start-Hinweise (Docker)

Führe das Hilfsskript in einer Linux-Shell (WSL2 Ubuntu) aus:

```bash
bash scripts/docker_compile_and_run.sh build-image
bash scripts/docker_compile_and_run.sh run-app -- run --config /mnt/config/tile_compile.yaml --input-dir /mnt/input --runs-dir /workspace/tile_compile_cpp/runs
```

GUI unter Windows:

- Empfohlen: Windows 11 + WSLg, dann:

```bash
bash scripts/docker_compile_and_run.sh run-gui
```

- Ohne WSLg: X-Server (z.B. VcXsrv) starten, `DISPLAY` setzen und GUI manuell starten:

```bash
export DISPLAY=host.docker.internal:0.0
docker run --rm -it \
  -e DISPLAY=$DISPLAY \
  -e QT_QPA_PLATFORM=xcb \
  -v "$(pwd)/tile_compile_cpp/runs:/workspace/tile_compile_cpp/runs" \
  -w /workspace/tile_compile_cpp/build \
  tile_compile_cpp:dev \
  ./tile_compile_gui
```

### CLI-Runner

```bash
./tile_compile_runner \
  run \
  --config ../tile_compile.yaml \
  --input-dir /path/to/lights \
  --runs-dir /path/to/runs
```

Häufige Optionen:

- `--max-frames <n>` Frames begrenzen (`0` = keine Begrenzung)
- `--max-tiles <n>` Tile-Anzahl für Phase 5/6 begrenzen (`0` = keine Begrenzung)
- `--dry-run` Validierungsablauf ohne vollständige Verarbeitung ausführen
- `--run-id <id>` benutzerdefinierte Run-ID für Gruppierung
- `--stdin` mit `--config -` um YAML von stdin zu lesen

Fortsetzungsmodus (Resume):

```bash
./tile_compile_runner resume \
  --run-dir /path/to/runs/<run_id> \
  --from-phase PCC
```

### CLI Scan (Frame-Erkennung)

```bash
./tile_compile_cli scan /path/to/lights --frames-min 30
```

### Weitere CLI-Möglichkeiten

```bash
# Konfiguration validieren
./tile_compile_cli validate-config --path ../tile_compile.yaml

# verfügbare Runs auflisten
./tile_compile_cli list-runs /path/to/runs

# einen Run inspizieren
./tile_compile_cli get-run-status /path/to/runs/<run_id>
./tile_compile_cli get-run-logs /path/to/runs/<run_id> --tail 200
./tile_compile_cli list-artifacts /path/to/runs/<run_id>
```

### GUI (Qt6)

```bash
./tile_compile_gui
```

## Ausgaben

Nach einem erfolgreichen Lauf (`runs/<run_id>/`):

- `outputs/`
  - `stacked.fits`
  - `reconstructed_L.fit`
  - `stacked_rgb.fits` (OSC)
  - `stacked_rgb_solve.fits` / WCS-Artefakte
  - `stacked_rgb_pcc.fits`
  - `synthetic_*.fit` (modusabhängig)
- `artifacts/`
  - `normalization.json`
  - `global_metrics.json`
  - `tile_grid.json`
  - `global_registration.json`
  - `local_metrics.json`
  - `tile_reconstruction.json`
  - `state_clustering.json`
  - `synthetic_frames.json`
  - `validation.json`
  - `report.html`, `report.css`, `*.png`
- `logs/run_events.jsonl`
- `config.yaml` (Run-Snapshot)

## Externe Quellen (PCC und Astrometrie)

Für optionale Farbkalibrierung und astrometrisches Solving kann die Pipeline externe Daten und Tools verwenden:

- **Siril Gaia DR3 XP sampled catalog** (für PCC)
  - Kann wiederverwendet werden, falls bereits von Siril heruntergeladen.
  - Typischer lokaler Pfad: `~/.local/share/siril/siril_cat1_healpix8_xpsamp/`
  - Upstream-Quelle (Katalog-Release): `https://zenodo.org/records/14738271`
- **ASTAP** (für Astrometrie / WCS Plate Solving)
  - Benötigt ASTAP plus eine Sterndatenbank (z.B. D50 für Deep-Sky-Nutzung).
  - Offizielle Seite/Downloads: `https://www.hnsky.org/astap.htm`

Wenn diese Ressourcen nicht installiert sind, funktioniert die Kernrekonstruktion weiterhin, aber ASTROMETRY- und PCC-Phasen können je nach Konfiguration übersprungen werden oder fehlschlagen.

## Diagnosebericht (`tile_compile_cpp/generate_report.py`)

Erzeuge einen HTML-Qualitätsbericht aus einem abgeschlossenen Lauf:

```bash
python tile_compile_cpp/generate_report.py runs/<run_id>
```

Ausgabe:

- `runs/<run_id>/artifacts/report.html`
- `runs/<run_id>/artifacts/report.css`
- `runs/<run_id>/artifacts/*.png`

Der Bericht aggregiert Daten aus Artifact-JSON-Dateien, `logs/run_events.jsonl` und `config.yaml`, einschließlich:

- Normalisierung/Hintergrund-Trends
- Globale Qualitätsverteilungen und Gewichte
- Registrierungs-Drift/CC/Rotation-Diagnosen
- Tile- und Rekonstruktions-Heatmaps
- Clustering- und Zusammenfassungen synthetischer Frames
- Validierungsmetriken (einschließlich Tile-Pattern-Indikatoren)
- Pipeline-Timeline und Frame-Usage-Funnel

## Kalibrierung (Bias / Dark / Flat)

- Master-Frames (`bias_master`, `dark_master`, `flat_master`) können direkt verwendet werden
- Verzeichnis-basierte Master (`bias_dir`, `darks_dir`, `flats_dir`) können automatisch erstellt werden
- `dark_auto_select: true` ordnet Darks nach Belichtungszeit zu (±5%)

## Projektstruktur

```text
tile_compile/
├── tile_compile_cpp/
│   ├── apps/
│   ├── include/tile_compile/
│   ├── src/
│   ├── gui_cpp/
│   ├── tests/
│   ├── generate_report.py
│   ├── tile_compile.yaml
│   ├── tile_compile.schema.json
│   └── tile_compile.schema.yaml
├── tile_compile_python/  # legacy
├── doc/
│   └── v3/
│       ├── process_flow/
│       ├── tbqr_step_by_step_en.md
│       └── tile_basierte_qualitatsrekonstruktion_methodik_v_3.2.2_en.md
├── runs/
├── README.md
└── README_de.md
```

## Tests

```bash
cd tile_compile_cpp/build
ctest --output-on-failure
```