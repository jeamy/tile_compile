# Tile-Compile

Tile-Compile ist ein Toolkit für **tile-basierte Qualitätsrekonstruktion** astronomischer image stacks (Methodik v3.3).

Wir stellen eine neuartige Methodik zur Rekonstruktion hochwertiger astronomischer Bilder aus Kurzzeitbelichtungs-Deep-Sky-Datensätzen vor. Konventionelle Stacking-Methoden beruhen häufig auf einer binären Frame-Auswahl ("Lucky Imaging"), wodurch erhebliche Teile der gesammelten Frames verworfen werden. Unser Ansatz, **Tile-Based Quality Reconstruction (TBQR)**, ersetzt diese starre Frame-Auswahl durch ein robustes räumlich-zeitliches Qualitätsmodell. Indem wir Frames in lokale Tiles zerlegen und die Qualität entlang zweier orthogonaler Achsen modellieren — globale atmosphärische Transparenz/Rauschen und lokale strukturelle Schärfe — rekonstruieren wir ein Signal, das an jedem Pixel physikalisch und statistisch optimal ist. Wir zeigen, dass diese Methode die volle photometrische Tiefe des Datensatzes bewahrt und zugleich eine überlegene Auflösungsverbesserung gegenüber traditionellen Referenz-Stacks erzielt.

Während die Methodik ursprünglich entwickelt wurde, um die spezifischen Herausforderungen von Kurzzeitbelichtungsdaten moderner Smart-Teleskope (z.B. DWARF, Seestar) zu adressieren, macht ihre architektonische Flexibilität sie ebenso leistungsfähig für konventionelle astronomische Setups. Der umfangreiche Satz abstimmbarer Parameter — von adaptiver Tile-Größe und Kreuzkorrelationsschwellen bis hin zu ausgefeilter Clustering-Logik — ermöglicht eine präzise Optimierung der Pipeline für ein breites Spektrum optischer Systeme und atmosphärischer Bedingungen.

> **Praxis-Hinweis:** Die Pipeline ist in erster Linie für Datensätze mit vielen nutzbaren Frames optimiert. Bei sehr kleinen Frame-Anzahlen oder bei stark gemischter Frame-Qualität innerhalb eines Stacks können in schwierigen Fällen sichtbare Kachelmuster auftreten. Dem kann man häufig entgegenwirken, indem man verschiedene Konfigurationseinstellungen testet (insbesondere Parameter für Registrierung, Tile-Geometrie und Rekonstruktion). Siehe dazu die Beispielprofile unter `tile_compile_cpp/examples/` sowie `tile_compile_cpp/examples/README.md`.

> **Hinweis:** Dies ist experimentelle Software, die primär für die Verarbeitung von Bildern von Smart-Teleskopen entwickelt wurde (z.B. DWARF, Seestar, ZWO SeeStar, usw.). Obwohl sie für die allgemeine astronomische Bildverarbeitung konzipiert ist, wurde sie für die spezifischen Eigenschaften und Herausforderungen von Smart-Teleskop-Daten optimiert.

## Dokumentation (v3.3)

- Methodik (normativ): [Tile-Based Quality Reconstruction Methodology v3.3.6](doc/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.6_en.md)
- Methodik-Paper PDF v3.3.6: [paper-tile_based_quality_reconstruction_methodology_v_3.3.6_en.pdf](doc/v3/paper-tile_based_quality_reconstruction_methodology_v_3.3.6_en.pdf)
- Prozessfluss (Implementierung): [Process flow (German)](doc/v3/process_flow/README_de.md)
- Deutsche Schritt-für-Schritt-Anleitung: [Schritt-für-Schritt-Anleitung](doc/v3/tbqr_step_by_step_de.md)
- Englische Schritt-für-Schritt-Anleitung: [Step-by-Step Guide](doc/v3/tbqr_step_by_step_en.md)
- Englisches Haupt-README: [English README](README.md)
- Ablaufplan (verständliche Kurzbeschreibung): [Ablaufplan – Funktionsweise des Systems](doc/v3/process_flow/data_flow_user_description_de.md)

## Datenquellen Für Das Paper-Beispiel

- M31-Lights für den Paper-Beispiellauf: [M31 lights](https://wolke.eibrain.org/index.php/s/Z88dmWizEJYjwBe)
- M31-Run für den Paper-Beispiellauf: [M31 run](https://wolke.eibrain.org/index.php/s/tfSycSNEzdL7jje)

Aus einem Verzeichnis mit FITS-Lights kann die Pipeline:

- Lights optional **kalibrieren** (Bias/Dark/Flat)
- Frames mit robuster 6-stufiger Kaskade **registrieren**
- **globale und lokale (Tile-)Qualitätsmetriken** berechnen
- Bild via tile-gewichteter Overlap-Add-Rekonstruktion erzeugen
- optional Frame-"Zustände" clustern und synthetische Frames erstellen
- Ergebnis via **Sigma-Clip** stacken
- OSC-Daten **debayern**
- **Astrometrie** (ASTAP/WCS) ausführen
- optionale **Background Gradient Extraction** (BGE, vor PCC) ausführen
- **photometrische Farbkalibrierung** (PCC) anwenden
- finale Ausgaben plus **Diagnose-Artefakte** (JSON) schreiben

## Aktive Komponenten

| Komponente | Verzeichnis | Status | Stack |
|-----------|-------------|--------|-------|
| Kernpipeline | `tile_compile_cpp/` | Aktiv | C++17 + Eigen + OpenCV + cfitsio + yaml-cpp |
| GUI2 Backend | `web_backend_cpp/` | Aktiv | Crow + C++17 |
| GUI2 Frontend | `web_frontend/` | Aktiv | HTML + CSS + JavaScript |

## Pipeline-Phasen

Im praktischen Einsatz ist der Gesamtworkflow bewusst einfach gehalten: Nach der Auswahl der Eingabedaten und einiger überschaubarer Konfigurationsparameter arbeitet die Pipeline den Datensatz automatisch vom Stacking über Astrometrie und optionale Hintergrundbehandlung bis hin zum PCC-Endergebnis ab. Für einen normalen Lauf sind keine komplizierten manuellen Zwischenschritte erforderlich. Gleichzeitig bleibt das System bis ins Detail konfigurierbar, sodass sich jede Phase bei Bedarf sehr fein anpassen lässt, etwa für Registrierung, Tile-Geometrie, Rekonstruktion, Stacking oder die nachgelagerte Verarbeitung.

| ID | Phase | Beschreibung |
|----|-------|-------------|
| 0 | SCAN_INPUT | Input-Erkennung, Modus-Erkennung, Linearitätsprüfung, Festplattenplatz-Precheck |
| 1 | REGISTRATION | Kaskadierte globale Registrierung |
| 2 | PREWARP | Vollbild-Canvas-Prewarp (CFA-sicher bei OSC) |
| 3 | CHANNEL_SPLIT | Metadatenphase (Kanalmodell) |
| 4 | NORMALIZATION | Lineare hintergrundbasierte Normalisierung |
| 5 | GLOBAL_METRICS | Globale Frame-Metriken und Gewichte |
| 6 | TILE_GRID | Adaptive Tile-Geometrie |
| 7 | COMMON_OVERLAP | Gemeinsamer datentragender Overlap (globale/tile-lokale Masken) |
| 8 | LOCAL_METRICS | Lokale Tile-Metriken und lokale Gewichte |
| 9 | TILE_RECONSTRUCTION | Gewichtete Overlap-Add Rekonstruktion |
| 10 | STATE_CLUSTERING | Optionales Zustands-Clustering |
| 11 | SYNTHETIC_FRAMES | Optionale Erzeugung synthetischer Frames |
| 12 | STACKING | Finales lineares Stacking |
| 13 | DEBAYER | OSC-Demosaicing zu RGB (MONO-Pass-Through) |
| 14 | ASTROMETRY | Astrometrisches Solving / WCS |
| 15 | BGE | Optionale RGB-Hintergrund-Gradientenentfernung vor PCC |
| 16 | PCC | Photometrische Farbkalibrierung |
| 17 | DONE | Finaler Status (`ok` oder `validation_failed`) |

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
- Referenzdokument: [Konfigurationsreferenz](doc/v3/configuration_reference.md)
- Praktische Beispiele: [Konfigurationsbeispiele & Best Practices](doc/v3/configuration_examples_practical_de.md)

### Beispielprofile

Vollständige eigenständige Beispielkonfigurationen sind verfügbar unter `tile_compile_cpp/examples/`.
Die Dateinamen verwenden nicht mehr das ältere Präfix `tile_compile.`.

- `full_mode.example.yaml`
- `reduced_mode.example.yaml`
- `emergency_mode.example.yaml`
- `smart_telescope_dwarf_seestar.example.yaml`
- `smart_telescope_very_bright_star.example.yaml`
- `canon_low_n_high_quality.example.yaml`
- `very_bright_star_anti_seam.example.yaml`
- `canon_equatorial_balanced.example.yaml`
- `mono_full_mode.example.yaml`
- `mono_small_n_anti_grid.example.yaml` (empfohlen für MONO-Datensätze mit geringer Frame-Anzahl, z.B. ~10..40, zur Reduzierung von Tile-Muster-Risiko)
- `mono_small_n_ultra_conservative.example.yaml` (empfohlen für sehr kleine MONO-Datensätze, z.B. ~8..25, wenn Nahtstabilität wichtiger ist als aggressive Verstärkung)

Siehe auch: [Examples README](tile_compile_cpp/examples/README.md) für Einsatzzweck und Tuning-Schwerpunkt der einzelnen Profile.

## Binary Releases (GUI2)

Vorkompilierte GUI2-Release-Bundles werden über [GitHub Releases](https://github.com/jeamy/tile_compile/releases) veröffentlicht.

Jedes Bundle enthält:

- GUI2 Frontend (`web_frontend/`)
- Crow-Backend (`web_backend_cpp/`)
- native C++ Werkzeuge (`tile_compile_runner`, `tile_compile_cli`, `tile_compile_web_backend`)
- Starter für Linux, macOS und Windows

Zur Laufzeit arbeitet GUI2 über das lokale Crow/C++-Backend als Adapter auf den C++ Runner und die C++ CLI.

## Schnellstart

### GUI2 (empfohlen)

Entwicklungsstart aus dem Repository-Root:

```bash
./start_backend.sh
```

Danach im Browser:

```text
http://127.0.0.1:8080/ui/
```

Release-Bundle-Start:

- Linux: `start_gui2.sh`
- macOS: `start_gui2.command`
- Windows: `start_gui2.bat`

Der Starter kopiert die gebündelte Payload in ein benutzerspezifisches Installationsverzeichnis, startet das Crow-Backend im Vordergrund und öffnet den Browser auf die lokale GUI2-URL.

Mindestbetriebssysteme für die aktuellen GUI2-Release-Bundles:

- Linux: x86_64-Linux mit `glibc >= 2.39` (Ubuntu 24.04 oder äquivalent ist die sichere Basis für die derzeitigen CI-ZIP-Builds)
- macOS: macOS 13+
- Windows: Windows 10 x64 oder neuer

Hinweise:

- macOS ist derzeit ab Version 13 vorgesehen. Es ist also nicht erst macOS 15+ nötig, aber macOS 12 und älter sind nicht die dokumentierte Release-Basis.
- Linux-Bundles enthalten keine `glibc`; ältere Distributionen als die aktuelle Build-Basis sind daher nicht garantiert lauffähig.

### C++ CLI / Runner

Für eine vollständige anfängerfreundliche Anleitung siehe:
[Step-by-Step Guide](doc/v3/tbqr_step_by_step_en.md)

### Build-Voraussetzungen

- CMake >= 3.21
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
  libcurl4-openssl-dev
```

Linux (Fedora):

```bash
sudo dnf install -y \
  gcc-c++ cmake pkgconf-pkg-config ninja-build \
  eigen3-devel opencv-devel cfitsio-devel yaml-cpp-devel nlohmann-json-devel openssl-devel \
  libcurl-devel
```

macOS (Homebrew, Kernbibliotheken):

```bash
brew install cmake pkg-config eigen opencv cfitsio yaml-cpp nlohmann-json openssl curl
```

Hinweise:

- Wenn ein heruntergeladenes GUI2-/Release-Bundle von Gatekeeper mit Meldungen wie „Entwickler kann nicht identifiziert werden“ blockiert wird oder eine mitgelieferte `.dylib` nicht geöffnet werden kann, entferne das Quarantine-Flag am entpackten Release-Ordner mit `xattr -dr com.apple.quarantine /pfad/zum/entpackten_release` und starte das Bundle danach erneut.

Windows:

- MinGW/MSYS2: `mingw-w64-x86_64-eigen3`, `mingw-w64-x86_64-opencv`, `mingw-w64-x86_64-cfitsio`, `mingw-w64-x86_64-yaml-cpp`, `mingw-w64-x86_64-nlohmann-json`, `mingw-w64-x86_64-openssl`, `mingw-w64-x86_64-curl`, `mingw-w64-x86_64-pkgconf`
- MSVC/vcpkg: `eigen3`, `opencv`, `cfitsio`, `yaml-cpp`, `nlohmann-json`, `openssl`, `curl`, `pkgconf`

### Kompilieren

```bash
cd tile_compile_cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Release-Build und Packaging

GUI2-Release-Bundles werden gebaut über:

- `.github/workflows/release-tile-compile-gui2.yml`

Der Workflow baut die Qt-freien C++-Binaries, bündelt `web_backend_cpp/` und `web_frontend/`, ergänzt die GUI2-Starter und erzeugt ZIP-Artefakte für Linux, macOS und Windows.

Bewusst nicht enthalten:

- externe Siril-Katalogdaten
- externe ASTAP-Binary/Daten

Windows-Hinweis (Docker / CLI-Workflow):

- Das Build-Script erkennt MSYS2-Installationen unter `C:\msys64\mingw64` (oder `ucrt64`/`clang64`) automatisch und setzt `CMAKE_PREFIX_PATH` entsprechend.
- Falls MSYS2 nicht installiert ist, Abhängigkeiten installieren via:
  - **Option A (MinGW)**: MSYS2 von https://www.msys2.org/ installieren, dann in der MSYS2 MinGW64-Shell:
    ```bash
    pacman -S --needed mingw-w64-x86_64-toolchain mingw-w64-x86_64-cmake mingw-w64-x86_64-pkgconf
    pacman -S --needed mingw-w64-x86_64-eigen3 mingw-w64-x86_64-opencv mingw-w64-x86_64-cfitsio mingw-w64-x86_64-yaml-cpp mingw-w64-x86_64-nlohmann-json mingw-w64-x86_64-openssl
    ```
  - **Option B (MSVC)**: vcpkg installieren und `VCPKG_ROOT` setzen, dann:
    ```bat
    vcpkg install eigen3:x64-windows opencv4:x64-windows cfitsio:x64-windows yaml-cpp:x64-windows nlohmann-json:x64-windows openssl:x64-windows curl:x64-windows
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
  --from-phase BGE
```

Unterstützte Resume-Phasen: `ASTROMETRY`, `BGE`, `PCC`.

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

### GUI2-Integration

Der empfohlene UI-Pfad ist die webbasierte GUI2:

- Backend: `web_backend_cpp/`
- Frontend: `web_frontend/`
- Orchestrierung: Crow-Backend -> `tile_compile_cli` / `tile_compile_runner`

Entwicklungsstart:

```bash
./start_backend.sh
```

Danach `http://127.0.0.1:8080/ui/` öffnen.

## Ausgaben

Nach einem erfolgreichen Lauf (`runs/<run_id>/`):

- `outputs/`
  - `stacked.fits`
  - `reconstructed_L.fit`
  - `stacked_rgb.fits` (OSC)
  - `stacked_rgb_solve.fits` / WCS-Artefakte
  - `stacked_rgb_bge.fits` (BGE-only Snapshot vor PCC)
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
  - `bge.json`
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

## Diagnosebericht (`report.html` über C++-Backend)

Erzeuge einen HTML-Qualitätsbericht aus einem abgeschlossenen Lauf entweder über GUI2 oder direkt über die CLI:

```bash
./tile_compile_cli generate-report runs/<run_id>
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
- BGE-Diagnostik (Grid-Zellen, Residuen, Kanalverschiebungen)
- Validierungsmetriken (einschließlich Tile-Pattern-Indikatoren)
- Pipeline-Timeline und Frame-Usage-Funnel

## Kalibrierung (Bias / Dark / Flat)

- Master-Frames (`bias_master`, `dark_master`, `flat_master`) können direkt verwendet werden
- Verzeichnis-basierte Master (`bias_dir`, `darks_dir`, `flats_dir`) können automatisch erstellt werden
- `dark_auto_select: true` ordnet Darks nach Belichtungszeit zu (±5%)

## Projektstruktur

```text
tile_compile/
├── web_frontend/           # GUI2 HTML/CSS/JS Frontend
├── web_backend_cpp/        # GUI2 Crow/C++ Backend
├── tile_compile_cpp/
│   ├── apps/                # Runner/CLI Entry-Points
│   ├── include/tile_compile/
│   ├── src/
│   ├── examples/            # Beispielkonfigurationen
│   ├── scripts/             # Hilfsskripte
│   ├── tests/
│   ├── tile_compile.yaml
│   ├── tile_compile.schema.json
│   └── tile_compile.schema.yaml
├── packaging/gui2/          # GUI2 Starter und Bundle-Helfer
├── docker/                  # Docker Build-/Runtime-Images
├── doc/
│   ├── v3/                  # Methodik- und Prozessfluss-Doku
│   └── gui2/                # GUI2 Konzept-/Referenzdokumente
├── start_backend.sh         # Dev-Start fuer Crow-Backend + GUI2
├── start_gui2_docker.sh     # GUI2 in Docker starten
├── README.md
└── README_de.md
```

## Tests

```bash
cd tile_compile_cpp/build
ctest --output-on-failure
```

## Versionen

### v0.0.1 (2026-02-15)

- Erste öffentliche Version

### v0.0.2 (2026-02-16)

- Erste Version mit vorkompilierten Paketen für Windows, Linux und macOS
- Enthält GUI-, CLI- und Runner-Executables
- Experimentelle Version zu Testzwecken

### v0.0.3 (2026-03-05)

- Verbesserte BGE/PCC-Pipeline mit klarerer Phasensichtbarkeit, stärkeren Guardrails und konsistenterer Konfigurationsoberfläche.
- Erweiterte Parallelisierung in rechenintensiven Phasen.
- Mehrere Phasen-Optimierungen für stabileres Verhalten und geringeren Laufzeit-Overhead.

### v0.0.4 (2026-03-06)

- Alt/Az-Registrierung für Datensätze mit großer Feldrotation korrigiert.

### v0.0.5 (2026-03-09)

- GUI2 als empfohlene Oberfläche etabliert, mit Web-Frontend, FastAPI-Backend und plattformübergreifenden Release-Bundles.
- DE/EN-i18n-Abdeckung in GUI2 und Parameter-Studio erweitert; Dokumentation und Backend-Konfigurationshandling darauf abgestimmt.
- Den bisherigen Qt6-GUI-Pfad nach `legacy/` verschoben und den aktiv gepflegten GUI2-Start-/Packaging-Weg klarer dokumentiert.

### v0.0.6 (2026-03-11)

- Produktive Migration auf das Crow/C++-Backend abgeschlossen.
- Integrierte C++-Report-Generierung aktiviert.
- Launcher, Docker-Packaging und GitHub-Workflows auf direkten Start des C++-Backends umgestellt.

## v0.0.7 (2026-03-11)

Unterstützt nun:
- Linux: x86_64 Linux with `glibc >= 2.39` (Ubuntu 24.04 or equivalent is the safe baseline for the current CI-built ZIPs)
- macOS: macOS 13+
- Windows: Windows 10 x64 or newer

## v0.0.8 (2026-03-11)

- zero-copy COMMON_OVERLAP
- Scratch-Reuse in LOCAL_METRICS
- weniger Lock-Contention im tile_weighted-OLA
- schnellerer Sigma-Clip-Kern
- weniger Tile-Kopien im tile_weighted-Pfad
- parallele BGE-Autotune-Kandidatenbewertung

## v0.0.9 (2026-03-11)

- Linux-AppImage-Erzeugung im GitHub-Actions-Release-Workflow ergänzt.
- PCC-Background-Noise-Behandlung überarbeitet und passende UI-/Report-Updates angebunden, damit aktuelle PCC-Diagnostik in der GUI konsistenter sichtbar ist.

## Changelog

### (2026-03-11)

**Crow/C++-Laufzeit, Release-Packaging und PCC-Update:**

- Den produktiven GUI2-Pfad rund um das Crow/C++-Backend finalisiert, inklusive integrierter C++-Report-Erzeugung und abgestimmter Frontend-/Backend-Behandlung der Reports.
- Release-Packaging, lokale Build-/Start-Skripte und GitHub-Workflows für Linux, macOS und Windows aktualisiert, einschließlich der dokumentierten OS-Baselines der GUI2-Bundles.
- Linux-AppImage-Erzeugung im GitHub-Actions-Release-Workflow ergänzt, sodass Releases jetzt neben dem ZIP-Bundle auch ein portables Linux-Artefakt enthalten.
- Datumsbasierte Benennung der Run-Verzeichnisse ergänzt und Route-/WebSocket-Handling sowie Backend-Tests auf dieses Verhalten abgeglichen.
- PCC-Background-Noise-Behandlung überarbeitet und passende UI-/Report-Updates angebunden, damit aktuelle PCC-Diagnostik in der GUI konsistenter sichtbar ist.

### (2026-03-09)

**GUI2-Release + i18n-Refresh:**

- Den webbasierten GUI2-Stack (`web_frontend/` + `web_backend_cpp/`) als empfohlenen UI-Pfad etabliert und die Top-Level-Dokumentation entsprechend aktualisiert.
- Einen dedizierten GUI2-Release-Workflow samt Launcher-Packaging für Linux, macOS und Windows unter `.github/workflows/release-tile-compile-gui2.yml` und `packaging/gui2/` ergänzt.
- Frontend-Lokalisierung und Übersetzungen im Parameter-Studio deutlich erweitert; dazu passende Updates am Backend-Konfigurationsvertrag und an den Tests ergänzt.
- Den früheren Qt6-GUI-/Build-Script-Pfad nach `legacy/` verschoben, damit die gepflegte GUI2-Strecke klar von der Legacy-Desktop-Implementierung getrennt ist.

### (2026-03-10)

**Python-Eliminierung im produktiven GUI2-Pfad:**

- GUI2-Laufzeit, Packaging, Docker und CI auf das Crow/C++-Backend umgestellt.
- Die produktive Python-Abhängigkeit für Stats-/Report-Erzeugung entfernt; diese läuft nun über den integrierten C++-Backendpfad und CLI-Support.
- Repository-Struktur und GUI2-Dokumentation auf `web_backend_cpp/` als gepflegte Backend-Implementierung aktualisiert.

### (2026-03-05, spätere Aktualisierung)

**Strict/Practical Runtime-Vereinheitlichung + Verifikation:**

- Laufzeit-Core-Pfad der Bildverarbeitung für `assumptions.pipeline_profile: strict|practical` vereinheitlicht.
- Strict-spezifische Ausführungszweige im Hot-Path entfernt:
  - kein strict-only Pre-Registration-Reihenfolgepfad mehr,
  - kein strict-only Reduced/Full-Gate-Override (`max(200, threshold)`),
  - kein strict-only Tile-Re-Normalisierungszweig,
  - kein strict-only Kanal-Reweighting-Zweig im OSC-Tile-Stacking.
- Registration erzwingt in strict nicht mehr `registration.enable_star_pair_fallback=false`.
- Konfig-Referenzdoku (DE/EN) auf das aktuelle Runtime-Verhalten der Profile abgeglichen.
- A/B-Evidenzläufe (`max_frames=80`) hinzugefügt; gleicher Core-Flow, nur geringe numerische Fit-Varianz.

### (2026-03-05)

**Performance- und Durchsatz-Optimierungen (große Datensätze, 1000+ Frames):**

- Adaptive Worker-Auswahl je Phase ergänzt, mit I/O-bewusster Obergrenze auf Basis gesampelter Framegröße und Task-Anzahl.
- `DiskCacheFrameStore` nutzt jetzt persistente Memory-Mappings pro Frame mit Invalidation beim Überschreiben; das reduziert wiederholte open/mmap/unmap-Kosten beim Tile-Zugriff.
- Globaler PREWARP-Store-Mutex entfernt, sodass Cache-Schreibvorgänge parallel laufen können.
- `GLOBAL_METRICS` läuft jetzt im parallelen Worker-Pool mit thread-sicherer Progress- und Fehleraggregation.
- `TILE_RECONSTRUCTION`-Overlap-Add von einem globalen Lock auf Row-Stripe-Locking umgestellt, um Lock-Contention zu reduzieren.
- Im OSC-Tile-Rekonstruktionspfad wird jedes valide Frame-Tile nur noch einmal debayert und als R/G/B für die Kanal-Stacks wiederverwendet.
- `LOCAL_METRICS` überspringt global ungültige Tiles jetzt vor der Extraktion und begrenzt bei großen Produktionsläufen das Schreiben sehr großer Voll-Artefakte.

### (2026-03-03)

**Methodik-Angleichung (v3.3.6 Strict-Profil):**

- `assumptions.pipeline_profile: practical|strict` ergänzt (Kompatibilitätsmodus vs. strikt normatives Verhalten).
- Im `strict`-Profil laufen REGISTRATION/PREWARP vor CHANNEL_SPLIT/NORMALIZATION/GLOBAL_METRICS.
- Im `strict`-Profil wird Full-Mode erst ab `N >= 200` erzwungen.
- Im `strict`-Profil ist die Phase-7-Tile-Normalisierung vor OLA immer aktiv.
- PCC `auto_fwhm` fällt bei fehlendem Seeing deterministisch auf `FWHM=0` zurück.
- `registration.enable_star_pair_fallback` ergänzt (Default `true`); im strict-Profil deaktiviert für normativen Cascade-Order.
- Konfig-Schema, Beispielkonfig und v3-Referenzdokumente (DE/EN) entsprechend aktualisiert.

**BGE/PCC Konfigurations- und Doku-Abgleich:**

- Benutzerkonfigurierbare BGE-Fit-Parameter `bge.fit.robust_loss` und `bge.fit.huber_delta` wiederhergestellt.
- Benutzerkonfigurierbare BGE-Apply-Grenzwerte `bge.min_valid_sample_fraction_for_apply` und `bge.min_valid_samples_for_apply` ergänzt.
- Parser/Serializer/Schema-Unterstützung für diese Keys in der Runtime-Konfigurationsoberfläche wieder aktiviert.
- Runner-Mapping übernimmt wieder die konfigurierten Werte (kein internes Erzwingen auf feste Defaults).
- BGE-Konfig-Artefakte enthalten in Pipeline- und Resume-Pfad wieder `robust_loss` und `huber_delta`.
- BGE/PCC-Dokumentation und praktische Beispiele (DE/EN) auf den aktuellen Parametersatz aktualisiert.

### (2026-02-26)

**BGE-Phasensichtbarkeit / Vergleichs-Outputs:**

- BGE wird jetzt als eigene Pipeline-Enum-Phase (`BGE=15`) zwischen `ASTROMETRY` und `PCC` emittiert.
- Die GUI zeigt BGE explizit in der Phasenanzeige, inklusive BGE-Substep-Progress.
- Neuer expliziter Pre-PCC-Output `outputs/stacked_rgb_bge.fits` für direkten Vergleich BGE-only vs. BGE+PCC.
- Konfig-Dokumentation/Beispiele auf v3.3.6-Optionssatz aktualisiert:
  - `bge.autotune.*` (`enabled`, `strategy`, `max_evals`, `holdout_fraction`, `alpha_flatness`, `beta_roughness`)
  - `pcc.background_model`
  - `pcc.radii_mode`
  - `pcc.aperture_fwhm_mult`, `pcc.annulus_inner_fwhm_mult`, `pcc.annulus_outer_fwhm_mult`, `pcc.min_aperture_px`

### (2026-02-25)

**Registration / Canvas / Farbkorrektheits-Fixes:**

- **Bayer-paritätssichere Offsets im Registration/Prewarp-Pfad**: Canvas-Offsets werden jetzt konsistent behandelt, sodass die CFA-Parität über erweiterte/gecoppte Canvas-Bereiche stabil bleibt.
- **Output-Skalierungs-Origin korrigiert**: Skalierungsaufrufe verwenden an den kritischen Stellen die korrekten Tile-/Debayer-Offsets und vermeiden damit R/G-Paritätsfehler nach Crop/Canvas-Transformationen.
- **Common-Overlap- und Canvas-Handling** in der Prozessfluss-Doku präzisiert und auf das aktuelle Phasenmodell abgeglichen.

**PCC (Photometrische Farbkalibrierung) Verbesserungen:**

- **Robuster Log-Chromaticity-Fit** für die PCC-Matrixschätzung implementiert (anstelle des älteren rein proportion-basierten Ansatzes).
- **Guardrails für Kanal-Skalierungsfaktoren** ergänzt, um extreme globale Farbstiche zu verhindern.
- **Annulus-Kontaminationsfilter (IQR-Gate)** in der Apertur-Photometrie ergänzt, um instabile Sternmessungen in Nebel-/Gradient-Feldern zu verwerfen.

**Dokumentations-Refresh:**

- `doc/v3/process_flow/*` auf den aktuellen Produktionsstand gebracht, inkl. `PREWARP`, `COMMON_OVERLAP`, Canvas/Offset-Propagation und aktueller Enum-Phasenreihenfolge.

**BGE (Background Gradient Extraction):**

- Optionale BGE-Stufe vor PCC ergänzt, die den modellierten Hintergrund direkt von den RGB-Kanälen subtrahiert.
- Vordergrundbewusste BGE-Fit-Methode `modeled_mask_mesh` ergänzt, um in schwierigen Feldern mit großflächigen diffusen Objekten (z.B. M31/M42) Farbwolken vor PCC zu reduzieren.
- Neues Artefakt `artifacts/bge.json` mit kanalweisen Diagnosedaten (Tile-Samples, Grid-Zellen, Residuenstatistik).
- Report-Generator um eigenen BGE-Abschnitt mit Zusammenfassungsplots und Residuenanalyse erweitert.

### (2026-02-17)

**Neue Registrierungs-Features für Alt/Az-Montierungen in Polarnähe:**

- **Temporal-Smoothing Registration**: Bei Feldrotation werden automatisch Nachbar-Frames (i-1, i+1) für Registrierungen genutzt, wenn die direkte Registrierung zur Referenz fehlschlägt. Verkettete Warps: `i→(i-1)→ref` oder `i→(i+1)→ref`. Nützlich bei kontinuierlicher Feldrotation (Alt/Az nahe Pol) und Wolken/Nebel.

- **Adaptive Stern-Detektion**: Bei zu wenigen erkannten Sternen (< topk/2) wird automatisch ein zweiter Durchlauf mit niedrigerem Schwellwert (2.5σ statt 3.5σ) durchgeführt. Dies verbessert die Stern-Erkennung bei Wolken, Nebel oder schwachen Frames.

- **Neue Registration Engine**: `robust_phase_ecc` mit LoG-Gradient-Preprocessing, speziell für Frames mit starken Nebeln/Wolken optimiert.

**Feldrotations-Unterstützung:**

- **Canvas-Erweiterung für Alt/Az-Montierungen**: Der Output-Canvas wird jetzt automatisch erweitert, um alle rotierten Frames zu erfassen. Zuvor wurden Sterne an den Rändern abgeschnitten, wenn Alt/Az-Montierungen nahe dem Pol verwendet wurden. Die Bounding Box aller gewarpten Frames wird berechnet und der Canvas entsprechend vergrößert. Log-Ausgabe zeigt Erweiterung: `"Field rotation detected: expanding canvas from WxH to W'xH'"`.

**Dokumentation:**

- **Neu**: [Praktische Konfigurationsbeispiele & Best Practices](doc/v3/configuration_examples_practical_de.md) - Umfassender Leitfaden mit Anwendungsfällen für verschiedene Brennweiten, Seeing-Bedingungen, Montierungstypen und Kamera-Setups (DWARF, Seestar, DSLR, Mono CCD). Enthält Parameter-Empfehlungen basierend auf Methodik v3.3.4.
