# TBQR Schritt-für-Schritt-Anleitung (Deutsch)

Diese Anleitung erklärt, wie die aktive C++-Pipeline (`tile_compile_cpp`) ausgeführt wird und wie GUI2 über das Crow/C++-Backend mit ihr zusammenarbeitet.

**Update-Hinweis (2026-03-03):**
- Resume unterstützt `ASTROMETRY`, `BGE` und `PCC`.
- BGE läuft vor PCC; die BGE-Konfiguration enthält benutzernahe robuste Regler (`bge.fit.robust_loss`, `bge.fit.huber_delta`).
- Das Methodikprofil wird explizit über `assumptions.pipeline_profile` (`practical` oder `strict`) gesteuert.
- Die streng ausgerichtete Registrierungskaskade kann Star-Pairs über `registration.enable_star_pair_fallback: false` deaktivieren.

## 1) Voraussetzungen

Dieses Repository besteht derzeit aus drei relevanten Runtime-/Build-Bausteinen:

- `tile_compile_cpp` für den nativen Runner und die CLI
- `web_backend_cpp` für das Crow/C++-Backend
- `web_frontend` für die statischen HTML/CSS/JS-Dateien von GUI2

Erforderliche Core-Build-Abhängigkeiten:

- CMake >= 3.21 empfohlen
- C++17-Compiler
- Ninja empfohlen
- pkg-config
- Eigen3
- OpenCV >= 4.5
- cfitsio
- yaml-cpp
- nlohmann-json
- OpenSSL
- libcurl

Backend-spezifische Build-Abhängigkeiten:

- Crow
- Asio

Hinweise:

- Crow und Asio werden über den CMake-Build des Backends eingebunden.
- Das Frontend selbst benötigt für den normalen Betrieb keine JS-Build-Toolchain; es wird als statischer Dateisatz ausgeliefert.

Beispielpakete je Plattform:

- Linux: `build-essential cmake ninja-build pkg-config libeigen3-dev libopencv-dev libcfitsio-dev libyaml-cpp-dev nlohmann-json3-dev libssl-dev libcurl4-openssl-dev`
- macOS: zuerst `xcode-select --install`, dann `brew install cmake ninja pkg-config eigen cfitsio yaml-cpp nlohmann-json openssl curl` und zusätzlich `brew install opencv`
- Windows MSYS2 MinGW64: `mingw-w64-x86_64-toolchain mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja mingw-w64-x86_64-pkgconf mingw-w64-x86_64-eigen3 mingw-w64-x86_64-opencv mingw-w64-x86_64-cfitsio mingw-w64-x86_64-yaml-cpp mingw-w64-x86_64-nlohmann-json mingw-w64-x86_64-openssl mingw-w64-x86_64-curl mingw-w64-x86_64-ntldd`

macOS-Hinweis:

- Die Standard-`opencv`-Formel von Homebrew setzt derzeit ein neueres macOS als macOS 12 voraus. Für den dokumentierten Homebrew-Pfad ist macOS 13+ daher praktisch die sinnvolle Basis, sofern OpenCV nicht separat bereitgestellt wird.

## 2) Die C++-Werkzeuge bauen

Ausgehend vom Repository-Root:

```bash
cd tile_compile_cpp
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

Dadurch entstehen unter anderem:

- `tile_compile_runner` (Haupt-Runner der Pipeline)
- `tile_compile_cli` (Hilfs-CLI)

## 2b) Das Crow/C++-Backend bauen

Ausgehend vom Repository-Root:

```bash
cd web_backend_cpp
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF
cmake --build . -j$(nproc)
```

Dadurch entsteht:

- `tile_compile_web_backend` (Crow/C++ API- und UI-Backend)

Hinweise:

- Das Backend linkt gegen `yaml-cpp`, `OpenSSL` und `libcurl`.
- Unter Windows benötigt das Backend zusätzlich Winsock-Systembibliotheken zur Link-Zeit; das ist bereits im Backend-CMake berücksichtigt.

## 2a) Mit Docker bauen und ausführen (optional)

Wenn du eine isolierte Umgebung bevorzugst, verwende:
`tile_compile_cpp/scripts/docker_compile_and_run.sh`

Unterstützte Befehle:

- `build-image`: baut das Docker-Image und kompiliert `tile_compile_cpp` im Container
- `run-shell`: öffnet eine interaktive Shell im bereits kompilierten Container
- `run-app`: führt `tile_compile_runner` direkt im Container aus

Standard-Mapping für Runs:

- Host: `tile_compile_cpp/runs`
- Container: `/workspace/tile_compile_cpp/runs`

Beispielablauf:

```bash
# aus dem Repository-Root
./tile_compile_cpp/scripts/docker_compile_and_run.sh build-image

# optional interaktive Shell
./tile_compile_cpp/scripts/docker_compile_and_run.sh run-shell

# Pipeline direkt im Container ausführen
./tile_compile_cpp/scripts/docker_compile_and_run.sh run-app -- run \
  --config /mnt/config/tile_compile.yaml \
  --input-dir /mnt/input \
  --runs-dir /workspace/tile_compile_cpp/runs
```

Hinweis: Wenn sich Konfigurations- oder Eingabedateien außerhalb des gemappten Runs-Volumes befinden, verwende `run-shell` und starte den Runner manuell mit deinen eigenen Mounts.

## 3) Konfiguration auswählen und vorbereiten

Starte mit einem der Beispielprofile in `tile_compile_cpp/examples/`, zum Beispiel:

- `tile_compile.full_mode.example.yaml`
- `tile_compile.smart_telescope_dwarf_seestar.example.yaml`
- `tile_compile.mono_small_n_anti_grid.example.yaml`

Kopiere eine Datei und passe mindestens Folgendes an:

- `run_dir`
- `input.pattern`
- `data.image_width` / `data.image_height` (falls nötig)
- `data.bayer_pattern` (für OSC/CFA-Datensätze)
- `assumptions.pipeline_profile` (`strict` für explizite kanalweise v3.3.6-Ausrichtung, `practical` für den CFA-Proxy-Core-Pfad)
- `registration.enable_star_pair_fallback` (`false` im strikten Profil)

Beispiel:

```bash
cp ../examples/tile_compile.smart_telescope_dwarf_seestar.example.yaml ./my_config.yaml
```

## 4) Die Pipeline ausführen (Runner-CLI)

Einfacher Lauf:

```bash
./tile_compile_runner run \
  --config ./my_config.yaml \
  --input-dir /path/to/lights \
  --runs-dir /path/to/runs
```

Nützliche Optionen:

- `--max-frames N` begrenzt die Eingabeframes (`0` = kein Limit)
- `--max-tiles N` begrenzt die Tiles in Phase 5/6 (`0` = kein Limit)
- `--dry-run` validiert den Ablauf ohne vollständige Verarbeitung
- `--run-id <id>` erzwingt eine Run-ID (nützlich, um verwandte Läufe zu gruppieren)
- `--project-root <path>` setzt explizit den Projekt-Root
- `--stdin` zusammen mit `--config -`, um YAML über stdin zu übergeben

## 5) Einen beendeten oder unterbrochenen Lauf fortsetzen

```bash
./tile_compile_runner resume \
  --run-dir /path/to/runs/<run_id> \
  --from-phase BGE
```

Aktuell erlaubte `--from-phase`-Werte sind `ASTROMETRY`, `BGE` oder `PCC`.

Hinweise:

- Resume ist für die *Post-Run*-Phasen gedacht, die auf bereits erzeugten Run-Ergebnissen arbeiten, zum Beispiel WCS-Solving, Hintergrundextraktion oder photometrische Farbkalibrierung.
- Stelle sicher, dass das Run-Verzeichnis die erwarteten Eingaben enthält, etwa `outputs/stacked_rgb.fits` oder die von deiner Konfiguration erzeugten äquivalenten Dateien, sowie den Run-Snapshot `config.yaml`.
- Wenn du die Konfiguration nach dem Lauf geändert hast, bearbeite vorzugsweise `runs/<run_id>/config.yaml` direkt, damit die fortgesetzten Phasen konsistente Einstellungen verwenden.

## 6) Hilfsbefehle der CLI verwenden

### Eingabeverzeichnis scannen

```bash
./tile_compile_cli scan /path/to/lights --frames-min 30
```

### Konfigurationsdatei validieren

```bash
./tile_compile_cli validate-config --path ./my_config.yaml
```

### Runs auflisten

```bash
./tile_compile_cli list-runs /path/to/runs
```

### Status eines Runs anzeigen

```bash
./tile_compile_cli get-run-status /path/to/runs/<run_id>
```

### Letzte Logs anzeigen

```bash
./tile_compile_cli get-run-logs /path/to/runs/<run_id> --tail 200
```

### Artefakte auflisten

```bash
./tile_compile_cli list-artifacts /path/to/runs/<run_id>
```

## 7) GUI2 starten (empfohlener UI-Pfad)

Entwicklungsstart aus dem Repository-Root:

```bash
./start_backend.sh
```

Danach im Browser öffnen:

```text
http://127.0.0.1:8080/ui/
```

Release-Bundles starten GUI2 über:

- Linux: `start_gui2.sh`
- macOS: `start_gui2.command`
- Windows: `start_gui2.bat`

GUI2 ist keine eigenständige native Processing-Engine. Es nutzt das Crow/C++-Backend als UI-/Backend-Schicht und delegiert alle Scan-, Run-, Resume-, Astrometrie-, PCC- und Report-Aktionen an `tile_compile_cli` und `tile_compile_runner`.

## 8) GUI2-Workflow

Verwende diese Reihenfolge für einen vollständigen Lauf vom Scan bis zu den Endergebnissen.

### Schritt 1: GUI2 öffnen und Laufzeit-Defaults prüfen

1. Öffne das Dashboard unter `/ui/`.
2. Prüfe aktives Runs-Verzeichnis, Standard-Konfigurationsquelle und Guardrail-Status.
3. Wechsle bei Bedarf direkt zum Wizard oder zu **Input & Scan**.

### Schritt 2: Eingabeframes scannen

1. Öffne **Input & Scan** oder **Wizard** Schritt 1.
2. Wähle entweder ein absolutes Eingabeverzeichnis oder eine serielle MONO-Filter-Queue.
3. Starte den Scan und prüfe:
   - erkannte Frame-Anzahl
   - erkannten Farbmodus (`OSC` / `MONO`)
   - Bildgröße
   - Bayer-Muster (bei OSC)

### Schritt 3: Kalibrier-Eingaben prüfen

1. Konfiguriere Bias/Dark/Flat nur dann, wenn der Datensatz noch nicht kalibriert ist.
2. Wähle nach Bedarf Verzeichnisse oder Master-Dateien.
3. Lasse die Kalibrierung deaktiviert, wenn die Pipeline bereits kalibrierte Lights verwenden soll.

### Schritt 4: Parameter anpassen und validieren

1. Öffne **Parameter Studio**.
2. Bearbeite die Parameter nach Bereichen, verwende die Suche und nutze das Explain-Panel.
3. Validiere die Konfiguration, bevor du den Lauf startest.

### Schritt 5: Einen Lauf starten

1. Starte den Lauf aus dem **Dashboard**, dem **Wizard** oder über die dedizierten Run-Steuerungen.
2. Wähle, wo relevant, das Runs-Verzeichnis und den Run-Namen bzw. das Label.
3. Starte den Lauf und wechsle anschließend in den **Run Monitor**.

### Schritt 6: Fortschritt und Logs überwachen

1. Nutze den **Run Monitor** für Phasenstatus, Phasenfortschritt, Artefakte und die Auswahl eines Resume-Ziels.
2. Verwende **Live Log** für das gestreamte Log.
3. Prüfe bei abgeschlossenen oder fehlgeschlagenen Läufen:
   - `runs/<run_id>/logs/run_events.jsonl`
   - `runs/<run_id>/artifacts/`
   - `runs/<run_id>/outputs/`

### Schritt 7: Post-Run-Phasen bei Bedarf fortsetzen

GUI2 unterstützt über das Backend Resume-Abläufe für `ASTROMETRY`, `BGE` und `PCC`.

CLI-Äquivalent:

```bash
./tile_compile_runner resume \
  --run-dir /path/to/runs/<run_id> \
  --from-phase ASTROMETRY
```

### Schritt 8: Astrometrie, BGE und PCC

1. Nutze **Astrometry**, um das RGB-Ergebnis zu lösen und WCS zu schreiben.
2. Falls aktiviert, führe **BGE** vor PCC aus.
3. Verwende **PCC** auf dem gelösten oder BGE-korrigierten RGB-Ergebnis.
4. Erzeuge das finale kalibrierte Ergebnis sowie optionale Stats-/Report-Artefakte.

### Schritt 9: Finale Ausgaben prüfen

Prüfe den Inhalt des Run-Verzeichnisses:

- `outputs/` für finale FITS-Produkte
- `artifacts/` für JSON-Diagnosen und Report-Artefakte
- `logs/run_events.jsonl` für die detaillierte Ereignishistorie

## 9) Ausgaben prüfen

Ein erfolgreicher Lauf erzeugt `runs/<run_id>/` mit:

- `outputs/` (gestackte und rekonstruierte FITS-Dateien)
- `artifacts/` (JSON-Diagnosen, Report-Ressourcen)
- `logs/run_events.jsonl`
- `config.yaml` (Run-Snapshot)

## 10) Den HTML-Diagnosereport erzeugen

Aus dem Repository-Root:

```bash
./tile_compile_cli generate-report runs/<run_id>
```

Erwartete Ausgabedateien:

- `runs/<run_id>/artifacts/report.html`
- `runs/<run_id>/artifacts/report.css`
- `runs/<run_id>/artifacts/*.png`

## 11) Häufige Troubleshooting-Prüfungen

1. Prüfe, ob die FITS-Dateien linear und lesbar sind.
2. Stelle sicher, dass `input.pattern` und `--input-dir` auf denselben Datensatz zeigen.
3. Wenn bei der Registrierung zu viele Frames verworfen werden, überprüfe die Schwellen `registration.reject_*`.
4. Prüfe `runs/<run_id>/logs/run_events.jsonl` auf Warnungen und Fehler.
5. Verwende `tile_compile_cli get-run-status` und `list-artifacts`, um Phasenausgaben zu verifizieren.
