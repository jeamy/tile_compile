# Tile-Compile

Tile-Compile ist ein Toolkit für **tile-basierte Qualitätsrekonstruktion** astronomischer image stacks (Methodik v3.3).

Wir stellen eine neuartige Methodik zur Rekonstruktion hochwertiger astronomischer Bilder aus Kurzzeitbelichtungs-Deep-Sky-Datensätzen vor. Konventionelle Stacking-Methoden beruhen häufig auf einer binären Frame-Auswahl ("Lucky Imaging"), wodurch erhebliche Teile der gesammelten Frames verworfen werden. Unser Ansatz, **Tile-Based Quality Reconstruction (TBQR)**, ersetzt diese starre Frame-Auswahl durch ein robustes räumlich-zeitliches Qualitätsmodell. Indem wir Frames in lokale Tiles zerlegen und die Qualität entlang zweier orthogonaler Achsen modellieren — globale atmosphärische Transparenz/Rauschen und lokale strukturelle Schärfe — rekonstruieren wir ein Signal, das an jedem Pixel physikalisch und statistisch optimal ist. Wir zeigen, dass diese Methode die volle photometrische Tiefe des Datensatzes bewahrt und zugleich eine überlegene Auflösungsverbesserung gegenüber traditionellen Referenz-Stacks erzielt.

Während die Methodik ursprünglich entwickelt wurde, um die spezifischen Herausforderungen von Kurzzeitbelichtungsdaten moderner Smart-Teleskope (z.B. DWARF, Seestar) zu adressieren, macht ihre architektonische Flexibilität sie ebenso leistungsfähig für konventionelle astronomische Setups. Der umfangreiche Satz abstimmbarer Parameter — von adaptiver Tile-Größe und Kreuzkorrelationsschwellen bis hin zu ausgefeilter Clustering-Logik — ermöglicht eine präzise Optimierung der Pipeline für ein breites Spektrum optischer Systeme und atmosphärischer Bedingungen.

> **Praxis-Hinweis:** Die Pipeline ist in erster Linie für Datensätze mit vielen nutzbaren Frames optimiert. Bei sehr kleinen Frame-Anzahlen oder bei stark gemischter Frame-Qualität innerhalb eines Stacks können in schwierigen Fällen sichtbare Kachelmuster auftreten. Dem kann man häufig entgegenwirken, indem man verschiedene Konfigurationseinstellungen testet (insbesondere Parameter für Registrierung, Tile-Geometrie und Rekonstruktion). Siehe dazu die Beispielprofile unter `tile_compile_cpp/examples/` sowie `tile_compile_cpp/examples/README.md`.

> **Hinweis:** Dies ist experimentelle Software, die primär für die Verarbeitung von Bildern von Smart-Teleskopen entwickelt wurde (z.B. DWARF, Seestar, ZWO SeeStar, usw.). Obwohl sie für die allgemeine astronomische Bildverarbeitung konzipiert ist, wurde sie für die spezifischen Eigenschaften und Herausforderungen von Smart-Teleskop-Daten optimiert.

## Dokumentation (v3.3)

- Methodik (normativ): [Tile-Based Quality Reconstruction Methodology v3.3.9](docs/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.9_en.md)
- Methodik-Paper PDF v3.3.6: [paper-tile_based_quality_reconstruction_methodology_v_3.3.6_en.pdf](docs/v3/paper-tile_based_quality_reconstruction_methodology_v_3.3.6_en.pdf)
- Prozessfluss (Implementierung): [Process flow (German)](docs/process_flow/README_de.md)
- Deutsche Schritt-für-Schritt-Anleitung: [Schritt-für-Schritt-Anleitung](docs/tbqr_step_by_step_de.md)
- GUI3 Paketierung und Start: [GUI3 README](packaging/gui3/README.md)
- Englisches Haupt-README: [English README](README.md)
- Ablaufplan (verständliche Kurzbeschreibung): [Ablaufplan - Funktionsweise des Systems](docs/process_flow/data_flow_user_description_de.md)
- Vollständige Dokumentation: [https://jeamy.github.io/tile_compile/](https://jeamy.github.io/tile_compile/)
- Raw Stack GUI-Anleitung (Deutsch): [docs/raw_stack_gui_de.md](docs/raw_stack_gui_de.md)

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
| 17 | HYPERMETRIC_STRETCH | Optionaler VeraLux HyperMetric Stretch nach PCC |
| 18 | DONE | Finaler Status (`ok` oder `validation_failed`) |

Detaillierte Phasen-Dokumentation: `docs/process_flow/`

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
- Referenzdokument: [Konfigurationsreferenz](docs/configuration_reference.md)
- Praktische Beispiele: [Konfigurationsbeispiele & Best Practices](docs/configuration_examples_practical_de.md)

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

**Installations- und Update-Verhalten:**

- Beim ersten Start kopiert der Starter alle Anwendungsdateien nach `~/tilecompile/` (Linux/macOS) bzw. `%USERPROFILE%\tilecompile\` (Windows).
- Nach dem ersten erfolgreichen Start können Sie das heruntergeladene Paket-Archiv und den entpackten Ordner bedenkenlos löschen – alle Daten wurden in Ihr Benutzerverzeichnis kopiert.
- Bei Updates werden nur die Anwendungsdateien (`web_frontend/`, `web_backend_cpp/`, `tile_compile_cpp/`) ersetzt. Ihre Benutzerdaten (Konfigurationen, Runs, ASTAP-Katalog, PCC-Datenbank) bleiben unberührt.

Hinweis zur macOS-Installation:

- Unter macOS 15.x, einschließlich Sequoia 15.1, bietet Gatekeeper für unbekannte Entwickler teils nicht mehr den früheren Rechtsklick-Override an. Wenn `start_gui2.command` oder andere scripts blockiert werden, öffne `Systemeinstellungen -> Datenschutz & Sicherheit`, scrolle nach unten und erlaube dort den blockierten Eintrag wie `start_gui2.command` explizit, bevor du ihn erneut startest.

Mindestbetriebssysteme für die aktuellen GUI2-Release-Bundles:

- Linux: x86_64-Linux mit `glibc >= 2.35` (Ubuntu 22.04 oder äquivalent ist die sichere Basis für die derzeitigen CI-ZIP-Builds)
- macOS: macOS 15
- Windows: Windows 10 x64 oder neuer

Hinweise:

- macOS ist derzeit ab Version 13 vorgesehen. Es ist also nicht erst macOS 15+ nötig, aber macOS 12 und älter sind nicht die dokumentierte Release-Basis.
- Linux-Bundles enthalten keine `glibc`; ältere Distributionen als die aktuelle Build-Basis sind daher nicht garantiert lauffähig.

### C++ CLI / Runner

Für eine vollständige anfängerfreundliche Anleitung siehe:
[Step-by-Step Guide](docs/tbqr_step_by_step_en.md)

### Build-Voraussetzungen

- CMake >= 3.21
- C++17 Compiler (GCC 11+ oder Clang 14+)
- OpenCV >= 4.5
- Eigen3
- cfitsio
- yaml-cpp
- nlohmann-json

#### Voraussetzungen für GPU-Beschleunigung

Die Pipeline unterstützt zwei GPU-Backends:

**NVIDIA CUDA (opencv_cuda):**
- Benötigt OpenCV-CUDA-Module:
  - `opencv2/core/cuda.hpp`
  - `opencv2/cudawarping.hpp`
  - `opencv2/cudaarithm.hpp`
- Zur Laufzeit: CUDA-fähige NVIDIA-GPU und funktionierende CUDA-/OpenCV-Runtime erforderlich.
- `TILE_COMPILE_ENABLE_CUDA` aktiviert nur das CUDA-Hook-/Build-Gate.

**AMD/Intel/NVIDIA OpenCL (opencv_opencl):**
- Benötigt OpenCV-OpenCL-Modul:
  - `opencv2/core/ocl.hpp`
- Zur Laufzeit: OpenCL-fähige GPU (AMD, Intel, NVIDIA) und funktionierende OpenCL-Runtime erforderlich.
- Funktioniert mit AMD Radeon (Polaris/Vega/RDNA), Intel integrierten GPUs und NVIDIA-GPUs.
- Generell einfacher einzurichten als CUDA auf Nicht-NVIDIA-Hardware.

**Automatische Auswahl:**
- `acceleration_backend: auto` (Standard) erkennt verfügbare GPU-Backends automatisch zur Laufzeit.
- Prioritätsreihenfolge: CUDA → OpenCL → CPU
- Fällt sauber auf CPU zurück, wenn kein GPU-Backend verfügbar ist.

Hinweise:

- Viele Standard-OpenCV-Pakete aus Distributionen/Homebrew sind CPU-only. GPU-Beschleunigung benötigt OpenCV mit CUDA- oder OpenCL-Unterstützung.
- Für NVIDIA-GPUs: CUDA-Backend bietet typischerweise bessere Performance als OpenCL.
- Für AMD/Intel-GPUs: OpenCL ist das einzige unterstützte GPU-Backend.
- Unter macOS: OpenCL-Unterstützung hängt vom OpenCV-Build ab; CUDA ist nicht praktikabel.

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

- Die obigen Paketbeispiele reichen für CPU-Builds aus. Sie garantieren keine GPU-Beschleunigung, weil das jeweilige OpenCV-Paket auf dem Host die CUDA-Module enthalten muss.
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

Unterstützte Resume-Phasen (alle Phasen 0..17):
- Früh: `SCAN_INPUT`, `CHANNEL_SPLIT`, `NORMALIZATION`, `GLOBAL_METRICS`, `TILE_GRID`
- Mitte: `REGISTRATION`, `PREWARP`, `COMMON_OVERLAP`, `LOCAL_METRICS`, `TILE_RECONSTRUCTION`
- Spät: `STATE_CLUSTERING`, `SYNTHETIC_FRAMES`, `STACKING`, `DEBAYER`, `ASTROMETRY`, `BGE`, `PCC`, `HYPERMETRIC_STRETCH`

Häufige Resume-Punkte: `ASTROMETRY` (neu lösen), `BGE` (Hintergrund neu extrahieren), `PCC` (Farbe neu kalibrieren), `HYPERMETRIC_STRETCH` (finalen VeraLux-Stretch neu ausführen), `STACKING` (neu stacken aus synthetischen Frames).

### CLI Scan (Frame-Erkennung)

```bash
./tile_compile_cli scan /path/to/lights --frames-min 30
```

### Weitere CLI-Möglichkeiten

```bash
# Konfigurationshandling
./tile_compile_cli get-schema                              # JSON-Schema ausgeben
./tile_compile_cli dump-default-config                     # Default-Config als JSON
./tile_compile_cli load-config <pfad>                    # Config YAML laden und anzeigen
./tile_compile_cli save-config <pfad> [--stdin]            # Config YAML speichern
./tile_compile_cli validate-config (--path P | --yaml Y | --stdin)

# Run-Inspektion
./tile_compile_cli list-runs /pfad/zu/runs
./tile_compile_cli get-run-status /pfad/zu/runs/<run_id>
./tile_compile_cli get-run-logs /pfad/zu/runs/<run_id> [--tail N]
./tile_compile_cli list-artifacts /pfad/zu/runs/<run_id>

# Input-Scanning
./tile_compile_cli scan /pfad/zu/lights [--frames-min N]

# FITS-Analyse
./tile_compile_cli fits-stats /pfad/zu/bild.fits

# Photometrische Farbkalibrierung (PCC)
./tile_compile_cli pcc-run <in.fits> <out.fits> --wcs <wcs.fits> [--source vizier|siril]
./tile_compile_cli pcc-apply <in.fits> <out.fits> [--r X] [--g Y] [--b Z]

# GUI-State (für externe Tool-Integration)
./tile_compile_cli load-gui-state [--path <datei>]
./tile_compile_cli save-gui-state [--path <datei>] [--stdin | <JSON>]
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
  - `stacked_rgb_hms.fits` (optionale VeraLux HyperMetric Stretch Ausgabe)
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
- Wenn `use_bias: true` und `use_dark: true`, werden rohe Darks intern bias-korrigiert, außer `dark_already_bias_corrected: true` ist gesetzt
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
├── packaging/gui3/          # GUI3 Starter und Bundle-Helfer
├── docker/                  # Docker Build-/Runtime-Images
├── docs/
│   ├── v3/                  # Methodik-Dokumente
│   └── process_flow/        # Implementierungs-Prozessfluss
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

## Danksagung

Dieses Projekt wurde mit Unterstützung von Windsurf, Kiro, Antigravity, GPT 5.*,Claude 4.* Sonnet, Codex, ***. Überwachung durch einen Menschen in einer virtuellen Umgebung.

Die HyperMetric-Stretch-Phase (HMS) wurde aus dem VeraLux HyperMetric Stretch Siril-Skript übernommen:

- (c) 2025 Riccardo Paterniti
- VeraLux - HyperMetric Stretch
- SPDX-License-Identifier: GPL-3.0-or-later
- Version 1.5.2
- Inspiriert von der "True Color"-Methodik von Dr. Roger N. Clark
- Mathematische Basis: Inverse Hyperbolic Stretch (IHS) und Vector Color Preservation
- Sensorik-Basis: hardware-spezifische Quantum-Efficiency-Gewichtung


## Versionen

## v0.2.A (2026-05-26)
- Calibration Bug fixes

## v0.2.9 (25.05.2026)

**Raw-Stack-Preprocessing-Pipeline:**
- Neue eigenständige Raw-Stack-Oberfläche: Preprocessing von FITS-Light-Frames bis zum fertigen Stack, vollständig getrennt vom Tile-Compile-Run-Studio.
- Pipeline umfasst: Kalibrierung, CFA/Mono-Prep, Registrierung, Quality-Filtering, Stacking, Astrometrie, BGE, PCC, HyperMetric Stretch.
- Alle Parameter (Sigma-Clip, Rejection, Gewichtung, BGE, PCC, Astrometrie, HMS) werden aus der Parameter-Studio-Konfiguration übernommen – keine hartkodierten Werte.
- Siehe [docs/raw_stack_gui_de.md](docs/raw_stack_gui_de.md) für die GUI-Dokumentation.

## v0.2.8 (2026-05-23)

-- HMS Bug fixes

## v0.2.7 (22.05.2026)

**implementationHMS:**
- VeraLux HyperMetric Stretch (HMS) als Post-PCC-Pipelinephase ergänzt.

## v0.2.6 (20.05.2026)

**Build-Härtung & Frontend-Bereinigung:**
- web_backend_cpp Build mit CUDA 13 + OpenCV 4.11 CUDA 13 Konfiguration gehärtet
- Frontend-Refactoring: Utilities in `src/utils.js` zentralisiert (escapeHtml, getMessage, getStorageJson, humanizeControlId, etc.)
- shell.js, parameter-studio-page.js und tooltips.js zu ES6-Modulen mit gemeinsamen utils.js-Importen migriert
- Toter Code entfernt

## v0.2.5 (26.04.2026)

- v0.2.5 kombiniert die Überarbeitung des Dokumentationssystems mit einer BGE-Robustheitsrunde für schwierige chromatische Gradienten wie IC434. Die BGE-Sample-Estimator-Auswahl ist jetzt in YAML, Schema-Validierung und Parameter Studio sichtbar; das Autotuning kann robuste Estimatoren vergleichen und degenerierte flache Hintergrundmodelle ablehnen, wenn weiterhin deutlicher Hintergrund- oder Chroma-Spread vorliegt.
- Professionelles Dokumentationssystem mit MkDocs Material + Doxygen
- Installationsanleitungen für vorgefertigte Binärdateien (Ubuntu, Fedora, Arch)
- Konfigurierbare BGE-Sample-Estimatoren: `quantile`, `sigma_clipped_median`, `sextractor_mode` und `biweight`
- BGE-Autotune sweept jetzt Sample-Estimatoren und nutzt Chroma-/Background-Spread-Guards gegen flache oder unausgewogene Korrekturflächen
- Reconstruction-Fallback-Pfad gehärtet: sichere Shape-/Weight-Validierung, korrigiertes OLA-Memory-Budget, tile-lokale temporäre Buffer und Entfernung wirkungsloser Scheduler-/Config-Dead-Code-Pfade

## v0.2.4 (25.04.2026)

- Registrierungs-Performance: Anchor-Promotion-Runden nutzen jetzt wieder den parallelen Worker-Pool und versuchen nur noch ungelöste Frames erneut, deren nächster Anchor sich durch die Promotion geändert hat, statt wiederholte vollständige Single-Thread-Pässe auszuführen. Diagnose `reg_promotion_retry_frames` ergänzt.

## v0.2.3 (2026-04-24)

- Robustere Registrierung: Deep-Chain-Outlier-Rejection (lange Ketten mit niedrigem CC ablehnen), verdoppelte Anker-Dichte für große-N-Sessions, "Hopping" Sequential Rescue das schwache Nachbarn überspringt um bessere Anker zu finden, und ASTAP-Plate-Solving als Fallback auch für modell-interpolierte Frames.

## v0.2.2 (2026-04-24)

- **Hot/Dead-Pixel-Korrektur repariert** (`cosmetic_correction_cfa`): Defekte Pixel in Sternbereichen wurden bisher nicht erkannt, da `neighbor_threshold` zu niedrig gesetzt war — Sternhalo-Pixel wurden fälschlicherweise als "heiße Nachbarn" gewertet und blockierten die Korrektur. Der Threshold wurde auf den vollen globalen Schwellwert angehoben. Zusätzlich: Pixel die das 5-fache des lokalen Floors übersteigen werden jetzt bedingungslos ersetzt (`extreme_outlier`-Bypass). Dead/Cold-Pixel-Erkennung neu hinzugefügt. Funktioniert auch ohne Darks.

## v0.2.1 (2026-04-23)

- Registrierungsphase: NCC-Berechnung vor Registrierung robuster gegen Hintergrundsubtraktion und Hot-Pixel gemacht (Clamp + Gaussblur vor NCC); Near-Identity-Bypass-Bedingung mit `ncc_identity > 0.7`-Guard gestärkt um False-Accepts bei weit vom Referenzframe entfernten Frames zu verhindern.

## v0.2.0 (2026-04-14)

- Registrierung fuer lange Alt/Az-Sessions deutlich erweitert: N-skalierende Multi-Anchor-Referenzwahl, N-skalierende Anchor-Promotion, astrometrische Registrierung/Rescue fuer schwache oder ungelöste Frames sowie neue Praxisbeispiele und aktualisierte Prozessdokumentation fuer schwierige Rotations- und Seeing-Faelle.

## v0.1.F (2026-04-07)

- TILE_RECONSTRUCTION-Performance: Die speicherbedingte Worker-Reduktion (3 statt 8 Worker) wurde durch Frame-Sub-Batching ersetzt. Worker laufen jetzt immer mit dem konfigurierten `parallel_workers`-Wert; das Memory-Budget steuert die Batch-Größe (Frames pro Batch) statt die Thread-Anzahl. Erwarteter Speedup: ~2,7× für OSC-Läufe mit 600+ Frames bei 2 GB Memory-Budget.

## v0.1.E (2026-04-06)

- Calibration-/GUI2-Nachzug: Die Dark/Bias-Kalibrierung behandelt rohe Darks jetzt korrekt ohne doppelten Bias-Abzug, `dark_already_bias_corrected` wurde in Backend, Schema, Beispiel-YAMLs und Parameter Studio ergänzt, und das Parameter Studio zeigt pro gewählter Kategorie nur noch einen zusammenhängenden Abschnitt statt einer getrennten Doppelansicht.

## v0.1.D (2026-04-04)

- `registration.auto_engine` ergänzt (Standard: `true`): Erkennt starke Feldrotation automatisch aus einer kleinen Frame-Probe vor der Registrierung und überschreibt die Engine auf `triangle_star_matching` + `transform_model: affine`, wenn eine rotationsblinde Engine (`robust_phase_ecc`, `hybrid_phase_ecc`) für einen Alt/Az-Datensatz konfiguriert ist. Schwellwert: `auto_engine_rotation_threshold_deg` (Standard: `0.05°/Frame` — greift bei Alt/Az bei jeder Belichtungszeit, liegt sicher unter EQ-Residualrotation).

## v0.1.C (2026-04-03)

- Tile-Rekonstruktion nach dem Rollout der letzten Performance-Optimierungen stabilisiert; der Schwerpunkt lag auf Nachbesserungen und Analyse sichtbarer Kachel- bzw. Nahtartefakte im finalen Rekonstruktionsergebnis.

## v0.1.B (2026-03-31)

- PCC-/Output-Pfad korrigiert: `stacked_rgb.fits` bleibt der reine Stacking-Output, erfolgreiche `BGE`-/`PCC`-Snapshots bleiben sauber getrennt als `stacked_rgb_bge.fits` / `stacked_rgb_pcc.fits`, und `output_stretch` verwendet jetzt ausschliesslich ein lineares `0..max -> 0..65535`-Scaling; obsoletter nichtlinearer/Quantil-Stretch-Code wurde entfernt.

## v0.1.A (2026-03-29)

- Den spaeten RGB-/PCC-Ausgabepfad nach dem `v3.3.9`-Rollout stabilisiert: der sichtbare RGB-Stretch erhaelt jetzt die Chroma statt schwache Hintergrund-Kanalabweichungen aufzublasen, die PCC-Hintergrundneutralisierung besitzt nun die neue Steuerung `always|auto|off` mit einem nebelbewussten Auto-Guard, und der neue Parameter wurde in Schema, Doku und allen Beispielkonfigurationen nachgezogen.

## v0.1.9 (2026-03-28)

- Die `v3.3.9`-Methodik ist jetzt als aktiver Referenzstand in Code, Frontend und Dokumentation durchgezogen: der lineare Rekonstruktionskern, BGE/PCC-Semantik, Parameter-Studio-Sichtbarkeit und die Prozessdokumentation wurden auf denselben Stand gebracht; zusaetzlich wurde das Web-Backend bei Startfehlern gehaertet.

## v0.1.8 (2026-03-25)

- Linux-Paketierungs-Skripte verbessert: Alle benötigten Shared Libraries (OpenCV, CFITSIO, yaml-cpp, etc.) werden jetzt gebündelt für bessere Kompatibilität über verschiedene Distributionen hinweg und weniger Abhängigkeitsprobleme.

## v0.1.7 (2026-03-24)

- Linux-AppImage-Paketierung korrigiert: `TILE_COMPILE_INPUT_SEARCH_ROOTS` wird jetzt exportiert, sodass Verzeichnis-Scans in gepackten Releases funktionieren.
- GUI2-Dateibrowser verbessert: Übergeordnetes Verzeichnis (..) wird immer angezeigt, auch wenn noch nicht freigegeben, und öffnet bei Klick den Freigabe-Dialog für nahtlose Navigation.

## v0.1.6 (2026-03-24)

- GUI2-Queue-/Batch-Handling und Run-Monitor überarbeitet: Batch-Tabs im Run Monitor, batchbezogene Stats-/Report-Aktionen, Queue-Root-Benennung mit Stunden/Minuten sowie aktualisierte DE/EN-Dokumentation.

## v0.1.5 (2026-03-23)

- `PREWARP` für OpenCL stabilisiert und die GPU-Beschleunigung um OpenCL-Äquivalente für die bisher CUDA-exklusiven Pfade in `TILE_RECONSTRUCTION` und `STACKING` erweitert, einschließlich Sigma-Clipping und Overlap-Add-Akkumulation.

## v0.1.4 (2026-03-22)

- Echten artefaktbasierten `STACKING`-Resume-Pfad im C++-Runner ergänzt, sodass `resume --from-phase STACKING` aus `synthetic_*.fit`/`canvas_mask.fits` neu aufbaut statt die gesamte Pipeline erneut abzuspielen.
- Einen konkreten Fehler in der OLA-Gewichtung für Synthetic-/Tile-Überlagerung korrigiert: Null-/Invalid-Pixel tragen keine Hann-Gewichte mehr bei. Dieser spezielle Abdunklungspfad ist damit behoben, verbleibende innere Linienartefakte können aber weiterhin andere Ursachen haben.

## v0.1.3 (2026-03-21)

- Pro-Frame-Tracking für Registration-Herkunft und Kettentiefe in den C++-Registrierungsartefakten ergänzt, inklusive strengerer Blind-Chain-Ankerregeln zur Driftbegrenzung in schwachen sequentiellen Rescue-Ketten.
- GUI2-Resume-/Run-Monitor-Statusaktualisierung korrigiert, sodass aktive Phase und Status ohne manuelles Seiten-Refresh sofort sichtbar werden.

## v0.1.2 (2026-03-20)

- Alt/Az-Registrierungsvalidierung korrigiert: Warps werden jetzt auf dem tatsächlichen gemeinsamen Überlapp bewertet statt auf dem beschnittenen Vollbild-Canvas.
- Zu aggressive CC-Outlier-Verwerfung für lange rotierende Sessions entschärft: die CC-Schwelle ist jetzt absolut und nicht mehr relativ zur globalen Run-MAD-Verteilung.
- Extrapolation des Feldrotationsmodells außerhalb des Bereichs echter Registrierungen korrigiert: Rand-/Tail-Frames verwenden jetzt eine begrenzte Bridge-Vorhersage statt instabiler lokaler Polynom-Explosion.

## v0.1.1 (2026-03-19)

- GUI2-Tool-Persistenz und PCC-Speicherworkflow verbessert, einschließlich temp-basierter Speicherung und plattformübergreifend konsistentem Verhalten bei Temp-Pfaden.
- Backend-Speichernutzung gehärtet und die BGE-Autotune-Laufzeit auf dem IC434-Referenzlauf deutlich reduziert, ohne das gewählte Lösungsverhalten zu verändern.

## v0.1.0 (2026-03-18)

- Astrometry/PCC-Tool-Pfadeingaben werden nicht mehr durch Backend-Defaults überschrieben.

## v0.0.F (2026-03-17)

- DSO-Tile-Rekonstruktionsmethodik auf `v3.3.8` in DE/EN angehoben und auf die aktive Runtime-Semantik abgeglichen.
- Normativen Methodiktext für runtime-konfigurierte Modusgrenzen, nachbarschaftsbewusste lokale Metrik-Normalisierung, sigma-geclippte Tile-Rekonstruktion und affine photometrische Restaurierung nach OLA korrigiert.
- GUI2-Run-Name-Reset korrigiert: beim Wechsel des Eingabeordners wird der gemeinsame `run_name` nun in Dashboard, Wizard und Input&Scan geleert.
- Kurzen macOS-15-/Sequoia-Hinweis für Gatekeeper-blockierte `start_gui2.command` ergänzt.
- ASTAP-`d80`-Katalogdownload auf die realen Upstream-Pakete je Plattform umgestellt: Linux `.deb`, macOS `.pkg`, Windows `.exe`.

## v0.0.E (2026-03-15)

- `assumptions.frames_min` im aktiven Runner-Mode-Gate verdrahtet und `assumptions.reduced_mode_cluster_range` an das Reduced-Mode-Clustering angebunden.
- Veraltete `assumptions.pipeline_profile`, `assumptions.frames_optimal` und `assumptions.exposure_time_tolerance_percent` aus aktiver Config-/Schema-/Frontend-/Doku-/Beispieloberfläche entfernt.
- C++-Schema neu erzeugt und Parameter-Studio, Assumptions-UI sowie Methodik-/Referenzdokus mit der aktiven Runtime-Semantik synchronisiert.

## v0.0.D (2026-03-15)

- `TILE_RECONSTRUCTION`-Boundary-Diagnostik um getrennte Raw-/Normalized-Metriken erweitert und maskierte Canvas-Zonen aus der Metrik ausgeschlossen.
- Artefakt-Sichtbarkeit für `tile_norm_bg_*` und `tile_norm_scale` ergänzt, damit erkennbar wird, ob die Tile-Normierung sichtbare Nähte selbst verstärkt.
- GUI2-`run_name` und `runs_dir` zwischen Dashboard, Wizard und Input&Scan synchronisiert, inklusive direkter Bearbeitung auf der Input&Scan-Seite.

## v0.0.C (2026-03-13)

- GUI2-Parameter- und Konfigurationshandling mit aktuellem C++-Config-Schema, Defaults und Referenzdokus synchronisiert.
- Boundary-Diagnostik für sichtbare Tile-Mismatches in `TILE_RECONSTRUCTION` ergänzt und der ineffektive dedizierte Seam-Korrektur-Config-Block wieder entfernt.
- Run-Monitor um Resume-Config-/Template-/Revisions-Flows, detailliertere Live-Logs und robustere Statuskorrektur nach erfolgreichem Resume erweitert.

## v0.0.B (2026-03-12)

- Serverseitige Persistenz für den GUI2-UI-Draft-State über Backend-API und Statusspeicher ergänzt.
- UX-relevante Frontend-Parameter aus dem lokalen Browser-Storage in einen zentralen servergestützten UI-State migriert.
- Run-Namen, Preset-Auswahl, Config-Drafts, Validierungsstatus sowie Tool-Eingaben/-Ergebnisse konsistenter zwischen Dashboard, Parameter-Studio, Wizard und Tools synchronisiert.

## v0.0.A (2026-03-12)

- Bufixes

## v0.0.9 (2026-03-11)

- Linux-AppImage-Erzeugung im GitHub-Actions-Release-Workflow ergänzt.
- PCC-Background-Noise-Behandlung überarbeitet und passende UI-/Report-Updates angebunden, damit aktuelle PCC-Diagnostik in der GUI konsistenter sichtbar ist.

## v0.0.8 (2026-03-11)

- zero-copy COMMON_OVERLAP
- Scratch-Reuse in LOCAL_METRICS
- weniger Lock-Contention im tile_weighted-OLA
- schnellerer Sigma-Clip-Kern
- weniger Tile-Kopien im tile_weighted-Pfad
- parallele BGE-Autotune-Kandidatenbewertung

## v0.0.7 (2026-03-11)

- Unterstützt nun:
  - Linux: x86_64 Linux with `glibc >= 2.39` (Ubuntu 24.04 or equivalent is the safe baseline for the current CI-built ZIPs)
  - macOS: macOS 15
  - Windows: Windows 10 x64 or newer

## v0.0.6 (2026-03-11)

- Produktive Migration auf das Crow/C++-Backend abgeschlossen.
- Integrierte C++-Report-Generierung aktiviert.
- Launcher, Docker-Packaging und GitHub-Workflows auf direkten Start des C++-Backends umgestellt.

## v0.0.5 (2026-03-09)

- GUI2 als empfohlene Oberfläche etabliert, mit Web-Frontend, FastAPI-Backend und plattformübergreifenden Release-Bundles.
- DE/EN-i18n-Abdeckung in GUI2 und Parameter-Studio erweitert; Dokumentation und Backend-Konfigurationshandling darauf abgestimmt.
- Den bisherigen Qt6-GUI-Pfad nach `legacy/` verschoben und den aktiv gepflegten GUI2-Start-/Packaging-Weg klarer dokumentiert.

## v0.0.4 (2026-03-06)

- Alt/Az-Registrierung für Datensätze mit großer Feldrotation korrigiert.

## v0.0.3 (2026-03-05)

- Verbesserte BGE/PCC-Pipeline mit klarerer Phasensichtbarkeit, stärkeren Guardrails und konsistenterer Konfigurationsoberfläche.
- Erweiterte Parallelisierung in rechenintensiven Phasen.
- Mehrere Phasen-Optimierungen für stabileres Verhalten und geringeren Laufzeit-Overhead.

## v0.0.2 (2026-02-16)

- Erste Version mit vorkompilierten Paketen für Windows, Linux und macOS
- Enthält GUI-, CLI- und Runner-Executables
- Experimentelle Version zu Testzwecken

## v0.0.1 (2026-02-15)

- Erste öffentliche Version

## Changelog

### (25.05.2026)

**Raw-Stack-Preprocessing-Pipeline (`v0.2.8`):**

- Neue eigenständige Raw-Stack-Seite in GUI2 für die vollständige Vorverarbeitung von FITS-Light-Frames bis zum fertig gestackten und nachbearbeiteten Bild, läuft vollständig getrennt vom normalen Tile-Compile-Run-Studio.
- Die Pipeline deckt alle Phasen ab: Kalibrierung (Bias/Dark/Flat), CFA/Mono-Prep, Registrierung, Quality-Analyse, Frame-Filterung, Stacking (Sigma/Median/Winsor), Astrometrie (ASTAP), Background Gradient Extraction (BGE), Photometrische Farbkalibrierung (PCC) und HyperMetric Stretch.
- Alle konfigurierbaren Parameter (Sigma-Clip, Rejection-Methode, Stacking-Gewichtung, BGE, PCC, Astrometrie und HyperMetric Stretch) werden direkt aus der Parameter-Studio-Konfiguration übernommen – keine hartkodierten Werte.
- Output-Skalierung stellt Hintergrund und Skala nach dem Stacking korrekt wieder her für akkurate Pixelwerte.
- Raw-Stack-UI-Bereinigung: Run-Monitor-Button entfernt, vollständige i18n-Abdeckung für alle Labels und Buttons ergänzt.
- Siehe [docs/raw_stack_gui_de.md](docs/raw_stack_gui_de.md) für die vollständige GUI-Referenz.

### (22.05.2026) 

**implementationHMS:**
- VeraLux HyperMetric Stretch (HMS) als Post-PCC-Pipelinephase ergänzt.
- HMS ist jetzt in den C++-Config-Defaults, in `tile_compile.yaml` und in allen Beispiel-YAML-Profilen standardmäßig aktiviert.
- Der Default-Modus ist `ready_to_use` mit Adaptive Anchor, Auto LogD, Zielhintergrund `0.2` und Ausgabe `outputs/stacked_rgb_hms.fits`.
- `mode: scientific` ist implementiert für kontrollierte Stretch-Ausgaben ohne Ready-to-Use-Final-Scaling/Soft-Clip und mit optionaler `linear_expansion`.
- Resume unterstützt direktes erneutes Ausführen von HMS über `--from-phase HYPERMETRIC_STRETCH` für historische Runs mit vorhandenen PCC-Artefakten.

### (20.05.2026) 

**Build-Härtung & Frontend-Bereinigung:**
- RunnerFrameCache Build-Fehler behoben: fehlende Methoden `try_load_normalized` und `store_normalized` implementiert
- Beide C++-Projekte auf C++20 migriert (GCC 13+, Clang 16+)
- web_backend_cpp Build mit CUDA 13 + OpenCV 4.11 CUDA 13 Konfiguration gehärtet
- Backend route_utils: unvollständige AppState-Typ-Fehler behoben, Pfadvalidierung gehärtet
- Frontend-Refactoring: Utilities in `src/utils.js` zentralisiert (escapeHtml, getMessage, getStorageJson, humanizeControlId, etc.)
- shell.js, parameter-studio-page.js und tooltips.js zu ES6-Modulen mit gemeinsamen utils.js-Importen migriert
- Doppelte I18N-Funktionen eliminiert (message(), textFor(), activeLocale(), getLocale())
- Toter Code entfernt: `param_editor_index.json` (36KB ungenutztes Duplikat)
- Dokumentation aktualisiert: Einheitliche C++20-Anforderungen, Release-URLs auf v0.2.5 aktualisiert

### (26.04.2026)

**Dokumentationssystem und BGE-Robustheit (`v0.2.5`, 26.04.2026):**

- Professionelles Dokumentationssystem mit MkDocs Material und Doxygen-Integration für C++ API-Referenz hinzugefügt
- GitHub Releases Dokumentation mit korrekten Binärdateinamen (tile_compile_gui2-linux-v0.2.4.zip, etc.) aktualisiert
- Umfassende Installationsanleitungen für vorgefertigte Binärdateien auf Ubuntu/Debian, Fedora/RHEL und Arch/Manjaro hinzugefügt
- Navigation mit separaten Abschnitten für User Guide, Configuration, Methodology und API Reference restrukturiert
- Konfigurierbares `bge.sample_estimator` in YAML-Konfigurationen, Schema-Dateien und Parameter Studio ergänzt (`quantile`, `sigma_clipped_median`, `sextractor_mode`, `biweight`)
- BGE-Autotune erweitert: Sample-Estimatoren werden verglichen, flache Modelle werden bei verbleibendem Background-/Chroma-Spread penalisiert oder abgelehnt
- RGB-Chroma-Guards auf BGE-Methoden ausgeweitet, inklusive konservativer Fallbacks für unausgewogene kanalweise Korrekturflächen
- `ic434_background_gradient.example.yaml` mit robusten RBF-/`sextractor_mode`-Parametern für IC434-ähnliche rot/grüne Hintergrundgradienten aktualisiert
- `docs/reconstruction_audit_2026-04-26.md` mit Reconstruction-Audit-Checkliste und Umsetzungsnotizen ergänzt
- Reconstruction-Fallback-Helper gegen abweichende Frame-/Tile-Größen sowie fehlende oder ungültige Tile-Weights gehärtet
- `reconstruct_tiles_parallel()` auf tile-große temporäre OLA-Buffer umgestellt statt full-frame Scratch-Matrizen pro Tile/Sub-Batch zu allokieren
- Reconstruction-Memory-Budget aktualisiert: globale Overlap-Add-Akkumulatoren plus per-worker Tile-Scratch werden jetzt berücksichtigt
- Wirkungslosen Reconstruction-Scheduler-/Config-Dead-Code entfernt, inklusive ungenutztem GPU-Batch-Feld, ungenutzter `make_hann_1d()`-API und nicht funktionaler Underutilization-Erkennung

### (25.04.2026)

**Registrierungs-Performance: parallele Anchor-Promotion-Retries (`v0.2.4`):**

- Die Direct-Registration-Anchor-Promotion-Schleife nutzt für Retry-Pässe jetzt den konfigurierten parallelen Registration-Worker-Pool, statt auf einen single-threaded `reg_worker()`-Aufruf zurückzufallen.
- Promotion-Runden erzeugen nun eine gezielte Retry-Liste und besuchen nur ungelöste Frames erneut, deren nächster aktiver Anchor einer der neu promoteten Anchors ist; dadurch entfallen wiederholte vollständige 325-Frame-Pässe bei geändertem Anchor-Set.
- Der Registration-Fortschritt meldet nun die tatsächliche Job-Anzahl und Worker-Anzahl je Pass, und die `global_registration.json`-Diagnose enthält `reg_promotion_retry_frames` für künftige Laufzeitanalysen.

### (2026-04-24)

**Registrierungs-Robustheit: Deep-Chain-Rejection + adaptive Anchors + Hopping-Rescue + astrometrische Fallback (`v0.2.3`):**

- Chain-validierte Frames mit `chain_depth > max_blind_chain_depth` und `cc < reject_cc_min_abs` werden jetzt als `deep_chain_low_cc`-Outlier rejected statt akzeptiert; verhindert Drift durch lange sequentielle Ketten über Wolkenfelder.
- Adaptive aktive-Anchor-Zielgröße von `min(21, max(3, (N+59)/60))` auf `min(32, max(4, (N+29)/30))` erhöht, verdoppelt Anker-Dichte für große-N-Sessions (z.B. 325 Frames nutzen jetzt ~12 statt ~6 Anker).
- "Hopping" Sequential Rescue: wenn der direkte Nachbar niedriges CC hat oder keine Blind-Chain ankern kann, werden bis zu 5 Frames (für Refine) bzw. 8 Frames (für Rescue) in jede Richtung nach einem besseren Anker mit CC > 0.3–0.4 gesucht, reduziert Ketten-Tiefe dramatisch bei Streifwolken-Bedingungen.
- Astrometrisches Rescue wird jetzt *nach* der modellbasierten Warp-Vorhersage ausgeführt (Section 4b), sodass ASTAP auch Frames retten kann die nur `model_*`-Provenances haben; `weak_model`-Bedingung zu `should_try_astrometry` hinzugefügt damit low-CC Model-Frames für Plate-Solving berechtigt sind.

### (2026-04-24)

**Hot/Dead-Pixel-Korrektur repariert + Registrierungs-Code-Qualitaet (`v0.2.2`):**

- `cosmetic_correction_cfa` korrigierte defekte Pixel in Sternbereichen bisher stillschweigend nicht: `neighbor_threshold` war auf `0.5 × global_threshold` gesetzt, sodass Sternhalo-Pixel (die deutlich ueber dieser niedrigen Grenze liegen) als "heiße Nachbarn" gezählt wurden und die Korrektur echter hot Pixel in ihrer Naehe blockierten. Der Schwellwert wurde auf den vollen globalen Hot-Pixel-Threshold angehoben — nur noch Pixel die selbst Hot-Pixel-Kandidaten waeren zaehlen als heiße Nachbarn.
- `extreme_outlier`-Bypass hinzugefuegt: Pixel die `local_median + 5 × local_floor` ueberschreiten werden bedingungslos ersetzt, unabhaengig vom Neighborhood-Support. Kein echtes Sternprofil-Pixel erreicht diesen Wert relativ zu seinen gleichfarbigen Nachbarn.
- Dead/Cold-Pixel-Erkennung hinzugefuegt: `global_candidate_cold` (`< median − σ_threshold × σ`) und `cold_outlier` (`< local_median − local_floor`) werden jetzt ebenfalls durch den lokalen gleichfarbigen Median ersetzt.
- Alle drei Fixes arbeiten auf dem rohen CFA-Mosaik vor dem Warping und benoetigen keine Dark Frames.
- Diagnostische Keys in `global_reg_extra` in ein `diag`-Subobjekt verschoben (4.2); downstream-relevante Keys bleiben auf oberster Ebene.
- Section-Header in `run_phase_registration_prewarp` fuer die sieben Hauptphasen eingefuegt (4.1).

### (2026-04-23)

**Registrierungs-NCC-Robustheit + Near-Identity-Guard (`v0.2.1`):**

- NCC-Berechnung in `try_method` klemmt negative Werte jetzt ab und wendet einen Gaussblur (σ=1.5) vor der Berechnung von `ncc_identity_overlap` und `ncc_warped` an. Rohe normalisierte Proxy-Bilder enthalten negative Hintergrundwerte und Hot Pixel, die die NCC von ~0.88 auf ~0.05 bei Sub-Pixel-Verschiebungen kollabieren liessen und damit falsche Near-Identity-Ablehnungen ausloesten.
- Near-Identity-Bypass-Bedingung mit `ncc_identity > 0.7`-Guard gestaerkt: ein nahezu-null Warp wird nur dann als gueltiges Near-Identity-Ergebnis akzeptiert wenn der Frame bereits nah am Referenzframe liegt — verhindert falsche Bypasses bei Frames die schlicht keinen Shift gefunden haben.

### (2026-04-14)

**Registrierung v0.2.0: Multi-Anchor-Skalierung + astrometrische Registrierung/Rescue:**

- Die globale Registrierung verwendet nicht mehr nur starre `1/3/5`-Referenz-Buckets, sondern eine N-skalierende Anchor-Auswahl mit ungefaehr einem angeforderten Anchor pro 80 Frames, auf ungerade Anchor-Zahlen erzwungen und aktuell auf 15 begrenzt.
- Die Anchor-Promotion nach starken direkten Treffern skaliert jetzt ebenfalls mit `N`: Zielgroesse der aktiven Anchors ist ungefaehr ein Anchor pro 60 Frames, Promotions pro Runde und die Zahl der zusaetzlichen Direktpaesse wachsen fuer lange Sessions kontrolliert mit.
- Direkte Registrierung laeuft damit auf langen Alt/Az-Datensaetzen deutlich weniger in späte Ein-Referenz-Fallen; fruehe und spaete Sequenzbereiche koennen direkt an naeheren zeitlichen Anchors haengen.
- Astrometrische Registrierung/Rescue wurde im Registrierungs-Runner praktisch aufgewertet: ASTAP-basierte Loesungen greifen nicht mehr nur bei `cc <= 0`, sondern auch fuer schwache bzw. tief verkettete Ergebnisse und verwenden den naechsten aktiven Anchor als Referenzbasis.
- Neue Registration-Telemetrie in `global_registration.json`: unter anderem `requested_ref_frames`, `active_ref_frames`, `reg_target_active_anchor_count`, `reg_promote_limit_per_round`, `reg_max_direct_anchor_rounds`, `reg_direct_anchor_rounds` und `reg_source_counts`.
- Neues Beispielprofil [tile_compile_cpp/examples/m104.example.yaml](tile_compile_cpp/examples/m104.example.yaml) fuer den Problemfall "Alt/Az, etwas staerkere Rotation, schlechtes Seeing, gute Frames staerker gewichten" hinzugefuegt; DE/EN-Praxisbeispiele sowie [docs/process_flow/phase_1_registration.md](docs/process_flow/phase_1_registration.md) auf den aktuellen Registrierungsablauf aktualisiert.

### (2026-04-07)

**TILE_RECONSTRUCTION-Performance: Sub-Batch-Stacking ersetzt Worker-Reduktion (`v0.1.F`):**

- Die speicherbedingte Worker-Reduktion in TILE_RECONSTRUCTION durch Frame-Sub-Batching ersetzt. Bisher begrenzte ein 2-GB-Memory-Budget OSC-Läufe auf 3 parallele Worker (statt der konfigurierten 8), weil die Peak-RAM-Schätzung alle Frames gleichzeitig pro Worker annahm. Worker laufen jetzt immer mit dem konfigurierten `parallel_workers`-Wert; das Budget steuert die Sub-Batch-Größe (Frames pro Batch). Für den Referenzlauf (610 Frames, 475 Tiles, 8 Worker, 2 GB Budget) ergeben sich ~3 Batches à ~205 Frames — gleiche Qualität, ~2,7× schnellere TILE_RECONSTRUCTION.
- `tile_boundary_diagnostics_enabled` in `runtime_limits` ergänzt (Standard: `false`). Boundary-Diagnostik ist jetzt opt-in; der bisherige Standard, sie immer auszuführen, verursachte ~5–10 % Overhead pro Produktionslauf.
- `tile_grid.json` enthält jetzt `estimated_reconstruction_time_s` (kalibrierte Schätzung auf Basis von Tile-Anzahl, Frame-Anzahl und Worker-Anzahl) und `coverage_filtered_tiles`.
- `runtime_limits.json` enthält jetzt `tile_analysis_to_stack_ratio`; eine Warnung wird geloggt wenn der Wert 10 überschreitet.
- `phase_end`-Ereignis für TILE_RECONSTRUCTION enthält jetzt `duration_s`.
- web_backend_cpp Code-Qualitätsfixes: drei doppelte `utc_now_iso()`-Implementierungen in einen gemeinsamen Header konsolidiert, SIGKILL-Versand bei jedem Polling-Zyklus nach SIGTERM korrigiert (wartet jetzt ~3 s), FD-Leak bei `fork()`-Fehler behoben, sequentiellen stdout/stderr-Deadlock in `run_subprocess()` behoben und `prune_locked()`-Aufrufhäufigkeit von jeder Mutation auf Terminal-Zustandsübergänge reduziert.

### (2026-04-06)

**Calibration-Fix + Parameter-Studio-Umorganisation (`v0.1.E`):**

- Fehler in der Bias/Dark-Verarbeitung korrigiert: bei aktivem Bias und Dark wird ein rohes Dark nun intern bias-korrigiert, bevor es auf Lights angewendet wird; damit wird der Bias-Pedestal nicht mehr doppelt subtrahiert.
- Neues Config-Feld `calibration.dark_already_bias_corrected` in Runner, Schema, Doku, Defaults, Beispielkonfigurationen und GUI2 aufgenommen, damit bereits bias-korrigierte Master-Darks explizit markiert werden können.
- Parameter Studio umorganisiert: bei Auswahl einer Kategorie wie `registration` oder `calibration` wird nur noch ein einziger zusammenhängender Abschnitt angezeigt; fehlende Schema-Parameter werden in denselben Block ergänzt statt separat im Abschnittseditor dargestellt.

### (2026-04-05)

**Calibration-Guardrails und Backend-Persistenz für Kalibrierpfade:**

- Kalibrierpfade in GUI2 beim Deaktivieren einer Calibration-Stufe aus der aktiven Config entfernt und beim erneuten Aktivieren aus dem serverseitigen UI-State wiederhergestellt, ohne Browser-Storage zu verwenden.
- Zusätzliche Guardrails für Kalibrationsdaten ergänzt, unter anderem Warnungen bei offensichtlichen Gain-Mismatches zwischen Lights und Calibration-Dateien.

### (2026-04-04)

**Auto-Engine für Alt/Az-Feldrotation + Registrierungsausfall-Fix (`v0.1.D`):**

- `registration.auto_engine` ergänzt (Standard: `true`): Sondiert vor der Registrierung einige Frames und überschreibt die Engine automatisch auf `triangle_star_matching` + `transform_model: affine`, wenn eine rotationsblinde Engine konfiguriert ist und starke Feldrotation erkannt wird. Schwellwert: `auto_engine_rotation_threshold_deg` (Standard: `0.05°/Frame`).
- Vollständigen Registrierungsausfall behoben: `engine: robust_phase_ecc` mit `allow_rotation: true` auf Alt/Az-Datensätzen lieferte NCC ≈ 0 für alle Frames — 469/470 Frames fielen auf Identity-Transform zurück ohne echte Ausrichtung.
- Neue Config-Felder in alle Schemas, Beispielkonfigurationen und Dokumentation übertragen.

### (2026-04-03)

**Stabilisierung der Tile-Rekonstruktion nach dem letzten Optimierungs-Rollout (`v0.1.C`):**

- Tile-Rekonstruktion nach dem Rollout der letzten Performance-Optimierungen stabilisiert; der Schwerpunkt lag auf Nachbesserungen und Analyse sichtbarer Kachel- bzw. Nahtartefakte im finalen Rekonstruktionsergebnis.

### (2026-03-29)

**Stabilisierung des RGB-/PCC-Ausgabepfads nach dem `v3.3.9`-Rollout (`v0.1.A`):**

- Den sichtbaren RGB-Output-Stretch so umgestellt, dass er luminanzbewusst arbeitet und die Chroma stabil haelt, statt kleine Hintergrund-Kanalabweichungen zu grossen blauen/grauen Randflaechen aufzubauschen.
- `pcc.background_neutralization_mode = always|auto|off` ergaenzt und mit einem neuen Auto-Guard versehen, der die Hintergrundneutralisierung abschwaecht oder unterdrueckt, wenn der gemessene "Hintergrund" eher diffuses Feldsignal als neutraler Himmel ist.
- Die neue PCC-Steuerung durch Schema, Defaults, Referenzdokumentation und alle Beispielkonfigurationen synchronisiert, sodass Runtime, Doku und Beispieloberflaeche jetzt denselben Stand zeigen.

### (2026-03-28)

**Implementierung und Durchzug der `v3.3.9`-Methodik (`v0.1.9`):**

- Die in `v3.3.9` definierten Kernpunkte in den aktiven Codepfad uebernommen: linearer Rekonstruktionskern ohne alte Tile-Vor-OLA-Normalisierung, sauberere BGE-/PCC-Semantik, robustere Support-/Seam-Behandlung und aktualisierte Guard-/Diagnostikpfade.
- Frontend und Konfigurationsoberflaeche auf den aktiven Schema-/Methodikstand nachgezogen, sodass neue Parameter aus `v3.3.9` im Parameter Studio und in der Dokumentation konsistenter verfuegbar sind.
- Die Prozessfluss-, Referenz- und Vergleichsdokumente auf `v3.3.9` aktualisiert und das Crow/C++-Web-Backend bei Startfehlern zusaetzlich gehaertet, sodass statt Core Dumps klare Fehlermeldungen erscheinen.

### (2026-03-24)

**AppImage-Paketierung-Fix + Dateibrowser-Navigation-Verbesserung (`v0.1.7`):**

- Linux-AppImage-Paketierung in `packaging/gui2/start_gui2.sh` korrigiert: Die Umgebungsvariable `TILE_COMPILE_INPUT_SEARCH_ROOTS` wird jetzt exportiert, wodurch Verzeichnis-Scan-Fehler in gepackten Releases behoben werden, bei denen relative Pfade nicht aufgelöst werden konnten.
- GUI2-Dateibrowser (`web_frontend/tooltips.js`) verbessert: Das übergeordnete Verzeichnis (..) wird jetzt immer angezeigt, auch wenn der Parent-Pfad noch nicht freigegeben ist. Gesperrte Pfade zeigen ein Schloss-Icon (🔒) und öffnen bei Klick den Freigabe-Dialog für nahtlose Aufwärts-Navigation.
- Backend-Dateilisten-Route (`web_backend_cpp/src/routes/system_routes.cpp`) aktualisiert: Liefert jetzt das `parent_allowed`-Flag zusammen mit dem `parent`-Pfad, sodass das Frontend zwischen zugänglichen und gesperrten Parent-Verzeichnissen unterscheiden kann.

**GUI2-Batch-/Queue-Run-Monitor-Refresh + Doku-Update (`v0.1.6`):**

- Den GUI2-Run-Monitor für Queue-/Batch-Läufe überarbeitet: Queue-Einträge erscheinen jetzt als Tabs, redundante doppelte Batch-/Filter-Zeilen wurden entfernt, und die obere Batch-/Verzeichnisstruktur-Zusammenfassung wird bei Queue-Runs wieder korrekt angezeigt.
- Batchbezogene Post-Run-Aktionen im Run Monitor aktiviert, sodass `Stats erstellen`, Stats-Ordner und Report auf den aktuell ausgewählten beendeten Batch-Tab arbeiten können, statt nur auf den aktiven Root-/Current-Run.
- Die Benennung unbenannter Queue-Root-Verzeichnisse von reinem Datum auf `YYYYMMDD_HHMM` umgestellt; dadurch sinkt das Kollisionsrisiko und die Dashboard-/Wizard-Hinweise stimmen wieder mit dem tatsächlichen Verhalten überein.
- Die EN-/DE-Schritt-für-Schritt-Anleitungen um explizite Batch-/Queue-Hinweise ergänzt, inklusive des primären MONO-Mehrfilter-Anwendungsfalls und des Tab-Verhaltens im Run Monitor.

### (2026-03-23)

**OpenCL-Ausbau für `PREWARP`, `TILE_RECONSTRUCTION` und `STACKING` (`v0.1.5`):**

- Den OpenCL-`PREWARP`-Pfad für die Multi-Thread-Ausführung stabilisiert, indem OpenCV-OpenCL-/T-API-Zugriffe geschützt und bei Bedarf explizite Host-Kopien erzwungen werden.
- `tile_compile_cpp/src/core/acceleration.cpp` um OpenCL-Äquivalente für die bisher CUDA-exklusiven Pfade in `TILE_RECONSTRUCTION` und `STACKING` erweitert, einschließlich Sigma-Clipping sowie Overlap-Add-Akkumulation und -Normalisierung.

### (2026-03-22)

**Echter `STACKING`-Resume + Synthetic-OLA-Seam-Fix (`v0.1.4`):**

- In `tile_compile_cpp/apps/runner_resume.cpp` einen echten artefaktbasierten `STACKING`-Resume-Pfad implementiert: `resume --from-phase STACKING` baut die Stacking-Ausgaben jetzt direkt aus vorhandenen `synthetic_*.fit` plus `canvas_mask.fits` neu auf und läuft danach mit den späteren Phasen weiter, statt einen vollständigen In-Place-Rerun anzustoßen.
- Einen konkreten Fehler in der Overlap-Add-Akkumulation in `tile_compile_cpp/src/core/acceleration.cpp` korrigiert: Null-/Invalid-Pixel eines Tiles addieren keine Hann-Gewichte mehr in `weight_sum`. Dieser spezielle Abdunklungspfad ist damit behoben, verbleibende innere Nähte/Linien können aber weiterhin andere Ursachen haben.

### (2026-03-21)

**Registration-Provenance/Kettentiefe-Diagnostik + Resume-Statussichtbarkeit (`v0.1.3`):**

- `tile_compile_cpp/apps/runner_phase_registration.cpp` erweitert: Jeder Frame trägt jetzt eine explizite Registration-Herkunft (`direct_global`, `sequential_rescue`, `temporal_rescue`, modellierte Varianten usw.) plus `chain_depth`, und diese Informationen werden in `global_registration.json` geschrieben.
- Blindes sequentielles Chaining verschärft: Schwache `sequential_rescue`-Frames dienen nicht mehr praktisch unbegrenzt als neue Anker; die Weiterverwendung wird jetzt über Kettentiefe begrenzt, außer die Korrelation ist stark genug.
- Zusätzliche Registration-Diagnostik in die Artefakt-Metadaten aufgenommen, darunter Source-Counts, maximale beobachtete Kettentiefe und blockierte Blind-Chain-Ankerversuche.
- GUI2-/Backend-Resume-Statuspfad korrigiert, sodass Monitor-Untertitel und Phasenstatus direkt nach `resume` aktualisiert werden, auch wenn der Runner schon läuft, aber das nächste `resume_start`-Event noch nicht im Run-Log steht.

### (2026-03-20)

**Stabilisierung von Registrierung/Feldrotation für lange Alt/Az-Sessions (`v0.1.2`):**

- Die globale Registrierungsvalidierung in `tile_compile_cpp/` so korrigiert, dass NCC-Vergleiche nur noch auf der tatsächlich gültigen Überlappungsmaske des gewarpten Frames berechnet werden statt auf dem beschnittenen Vollbild-Canvas. Dadurch werden korrekt größere Rotationswarps nicht mehr allein wegen abgeschnittener Ecken im festen Proxy-Bild verworfen.
- Dieselbe overlap-maskierte NCC-Validierung auch auf das temporale Rescue-Chaining angewendet, sodass Nachbar-zu-Referenz-Rescues nicht mehr aus demselben Crop-Canvas-Grund scheitern.
- Die CC-basierte Outlier-Verwerfung für lange rotierende Läufe überarbeitet: das `low_cc`-Reject-Gate verwendet jetzt direkt die konfigurierte absolute Mindestschwelle statt einer run-globalen Median/MAD-Schwelle, die viele geometrisch plausible Randframes fälschlich verworfen hat.
- Die Vorhersage des Feldrotationsmodells außerhalb des Bereichs echter Registrierungen korrigiert: Tail-/Head-Frames nutzen keine instabile lokale Polynom-Extrapolation mehr, sondern fallen auf eine begrenzte Bridge-artige Randvorhersage zurück. Das verhindert die massiven Fächer-/Keil-Artefakte aus der M66-Alt/Az-Regression.

### (2026-03-19)

**GUI2-Tool-Persistenz/PCC-UX, Backend-Memory-Guards und BGE-Autotune-Beschleunigung:**

- `web_backend_cpp/` gegen OOM-anfällige API-/Tool-Pfade gehärtet: begrenzte Subprozess-/Stdout-Captures, beschränkte Scan-/Report-Payloads, gestreamte Event-Datei-Auswertung sowie Limits für behaltene Jobs mit per Environment konfigurierbaren Defaults für GUI2.
- `packaging/gui2/.env.example` ergänzt und die neuen Backend-Runtime-Limit-Umgebungsvariablen für die GUI2-Starter dokumentiert.
- GUI2-Frontend-/Backend-Asset-Auslieferung und Routenverhalten korrigiert, sodass `/ui` und direkte Asset-Pfade zuverlässig aufgelöst werden statt 404 zu liefern.
- Astrometry-/PCC-Tool-UX verbessert: persistenter Downloadstatus über Seitenwechsel hinweg, korrigierte Download-Fortschrittsberechnung, automatische PCC-WCS-Vorbelegung aus gleichnamigen Dateien und automatische PCC-Parameterübernahme aus einer Run-`config.yaml` mit sichtbarer Herkunft im UI/Log.
- PCC-Output-Handling in GUI2 überarbeitet: `Run PCC` schreibt zuerst in ein temporäres Ergebnis, `Save Corrected` nutzt einen GUI2-internen Speichern-Dialog, kopiert das RGB-Ergebnis plus `_R/_G/_B`-Sidecars aus dem Temp-Output und arbeitet konsistent über Linux/macOS/Windows-Temp-Verzeichnisse hinweg.
- Standalone-PCC so korrigiert, dass bei fehlender `canvas_mask` ein sicherer Full-Image-Fallback verwendet wird statt den Tool-Lauf abzubrechen.
- BGE-Phasen-Timings in `bge.json` ergänzt und den echten Hotspot in `tile_compile_cpp/` optimiert: Autotune-Prep verwendet vorbereitete Tile-Analyse jetzt über mehrere Quantil-Kandidaten hinweg wieder, wodurch die gemessene BGE-Laufzeit auf dem IC434-Referenzlauf von etwa `472s` auf etwa `181s` sank, ohne neuen Vollbild-Speicherdruck zu erzeugen.

### (2026-03-18)

- Astrometry-Datenverzeichnis-Eingabe wird nun korrekt respektiert wenn User den Pfad manuell ändert - nutzt `shouldKeepAstapSelection`-Logik zur Bewahrung der User-Eingabe.
- Serverseitige Persistenz für Astrometry- und PCC-Tool-Parameter via UI-State-API hinzugefügt - Einstellungen überleben Server-Neustarts.
- Intelligentere Katalog-Downloads: Astrometry-Kataloge überspringen Download wenn bereits installiert, PCC Siril lädt nur fehlende Chunks herunter.
- Robustere Archiv-Extraktion für macOS `.pkg`, Linux `.deb` und Windows `.exe` mit besseren Fehlermeldungen und Validierung.
- macOS-Release-Bundle-Library-Probleme behoben durch explizites Bundling von GCC-Runtime-Libraries (`libgcc_s`, `libgfortran`, `libquadmath`, `libgomp`) und Beibehaltung von `libstdc++` für Homebrew-kompilierte Dependencies.

### (2026-03-17)

**Methodik `v3.3.8` + GUI2-Run-Name-Reset (`v0.0.F`):**

- Neue normative Methodikdokumente `docs/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.8_en.md` und `docs/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.8_de.md` ergänzt.
- Die Methodikspezifikation so korrigiert, dass sie zur aktiven Runtime bei Modusgrenzen, Shared-Core-Kanal-Semantik, nachbarschaftsbewusster lokaler Metrik-Normalisierung, sigma-geclippter Tile-Rekonstruktion und affiner photometrischer Restaurierung nach OLA passt.
- GUI2 so korrigiert, dass ein geänderter Eingabeordner den gemeinsamen `run_name` in Dashboard, Wizard und Input&Scan löscht.
- Kurzen macOS-15-/Sequoia-Hinweis für Gatekeeper-blockierte `start_gui2.command` Start ergänzt.
- ASTAP-`d80`-Downloads von der falschen gemeinsamen ZIP-Annahme auf die tatsächlichen Upstream-Pakete je Plattform umgestellt: Linux `.deb`, macOS `.pkg`, Windows `.exe`.

### (2026-03-15)

**Assumptions-Runtime-/Config-Synchronisierung (`v0.0.E`):**

- `assumptions.frames_min` wird jetzt im aktiven C++-Runner tatsächlich für das Mode-Gate verwendet statt über einen fest verdrahteten Mindestwert.
- `assumptions.reduced_mode_cluster_range` wirkt jetzt direkt auf das Reduced-Mode-Clustering und ist damit kein reiner Parser-/Schema-Rest mehr.
- Tote Assumptions-Felder aus der aktiven Konfigurationsoberfläche entfernt: `pipeline_profile`, `frames_optimal` und `exposure_time_tolerance_percent`.
- Aktiven C++-Config-Code, generierte Schemas, Beispiel-YAMLs, GUI2-Assumptions-/Parameter-Studio und DE/EN-Dokumentation auf die verbleibenden runtime-relevanten Assumptions-Felder zusammengezogen.

### (2026-03-15)

**Boundary-Diagnostik vertieft + GUI2-Run-Felder synchronisiert:**

- `TILE_RECONSTRUCTION` so erweitert, dass `tile_reconstruction.json` Raw- und Normalized-Tile-Boundary-Metriken getrennt ausgibt, zusätzlich zu `tile_norm_bg_r/g/b` und `tile_norm_scale` für die direkte Analyse der Tile-Normierung.
- Die Tile-Boundary-Analyse korrigiert, sodass maskierte `COMMON_OVERLAP`-/canvas-ungültige Zonen nicht mehr als gültige Nullsamples in die Diagnose eingehen.
- Methodik-, Prozessfluss-, Referenz- und Praxisdokus auf die read-only Raw-/Normalized-Boundary-Diagnostik und die Pflicht zur Common-Canvas-Maskierung aktualisiert.
- `run_name` und `runs_dir` auch auf Input&Scan eingebunden und beide Felder über Dashboard, Wizard und Input&Scan auf einen gemeinsamen GUI2-Zustand vereinheitlicht.

### (2026-03-13)

**GUI2-Config-/Studio-Sync + Tile-Boundary-Diagnostik:**

- Das ineffektive `stacking.tile_seam_harmonization.*`-Experiment aus der aktiven C++-Konfigurationsoberfläche entfernt und durch reine Tile-Boundary-Diagnostik in `TILE_RECONSTRUCTION` ersetzt.
- Config-Code, generierte Schemas, Beispiel-Configs und DE/EN-Referenzdokus mit der aktiven C++-Konfigurationsoberfläche synchronisiert.
- Das Parameter-Studio so überarbeitet, dass Parameterbestand, Defaults, Wertebereiche, Tooltips und Filterung aus aktuellem Schema und Default-Config stammen statt aus veralteten manuellen Listen.
- GUI2-Live-Log und Run-Monitor erweitert, inklusive detaillierterer Phaseninformationen, Resume-Config-Editing/Template-Flows, gespeicherter Config-Revisionen und korrekter Phasenstatus-Anhebung nach erfolgreichem Resume.

### (2026-03-12)

**Serverseitige GUI2-UI-State-Persistenz:**

- Persistenten Backend-Speicher samt API-Zugriff für den GUI2-UI-Draft-State ergänzt, sodass der Frontend-UX-State nicht mehr primär von lokalem Browser-Storage abhängt.
- Die wesentlichen UX-relevanten Frontend-Parameter in den gemeinsamen servergestützten UI-State migriert, darunter Run-Benennung, Preset-Synchronisation, Config-Drafts, Validierungsstatus, Dirty-State, Queues sowie Tool-Pfad-/Input-Einstellungen.
- Zusätzlichen sinnvollen Tool-Ergebniszustand über Reloads hinweg wiederherstellbar gemacht, während rein ephemerer Laufzeit-Anzeigezustand bewusst nicht persistent bleibt.

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

- `docs/process_flow/*` auf den aktuellen Produktionsstand gebracht, inkl. `PREWARP`, `COMMON_OVERLAP`, Canvas/Offset-Propagation und aktueller Enum-Phasenreihenfolge.

**BGE (Background Gradient Extraction):**

- Optionale BGE-Stufe vor PCC ergänzt, die den modellierten Hintergrund direkt von den RGB-Kanälen subtrahiert.
- `bge.method` ist der aktive Selektor: `none` (Default), `classic` oder `autobge`; das alte `bge.enabled` bleibt nur Kompatibilität, wenn `method` fehlt.
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

- **Neu**: [Praktische Konfigurationsbeispiele & Best Practices](docs/configuration_examples_practical_de.md) - Umfassender Leitfaden mit Anwendungsfällen für verschiedene Brennweiten, Seeing-Bedingungen, Montierungstypen und Kamera-Setups (DWARF, Seestar, DSLR, Mono CCD). Enthält Parameter-Empfehlungen basierend auf Methodik v3.3.4.
