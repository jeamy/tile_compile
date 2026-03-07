# HTML Frontend + FastAPI Backend (Desktop)

Stand: 2026-03-07

## 1) Ziel und Scope

Diese Doku beschreibt die technische Zielarchitektur fuer GUI2 als:

- HTML/CSS/JS Frontend im Browser (Desktop, 1920-Baseline)
- FastAPI Backend als API- und Prozess-Adapter
- bestehende C++ Engine weiterhin ueber `tile_compile_runner` und `tile_compile_cli`
- kein Qt6 in Build, Runtime und Deployment des Web-Stacks

Nicht im Scope:

- Mobile Layouts
- Ersatz der C++ Pipeline-Logik

## 2) Zielarchitektur

## 2.1 Komponenten

- `Frontend (HTML/CSS/JS)`:
  - statische UI-Dateien, i18n-Strings, Wizard, Parameter Studio, Run Monitor
  - kommuniziert nur ueber HTTP/WebSocket mit FastAPI
- `FastAPI Backend`:
  - REST-Endpunkte fuer Scan/Config/Run/History/Tools
  - WebSocket fuer Live-Log und Phase-Status
  - Prozessmanager fuer Runner/CLI
- `C++ Binaries`:
  - `tile_compile_runner` fuer `run` und `resume`
  - `tile_compile_cli` fuer Schema/Validation/Scan/History/FITS/PCC-Tools
- `Dateisystem-Store`:
  - `runs_dir` (frei waehlen)
  - pro Run Konfig-Revisionen (Versionierung fuer Resume)
  - Artefakte/Logs/Reports

## 2.2 Laufzeitfluss

1. Browser oeffnet `GET /` und laedt statisches Frontend.
2. Frontend holt initialen Zustand (`/api/app/state`).
3. Benutzer startet Aktionen (Scan, Validate, Run, Resume, Stats).
4. FastAPI startet CLI/Runner-Prozesse, streamt Status zur UI.
5. Ergebnisse/Artefakte werden in Run-Verzeichnissen persistiert.

## 2.3 Empfohlene Repo-Struktur (neu)

Diese Struktur ist fuer die Web-Migration neu anzulegen:

```text
tile_compile/
  web_backend/
    app.py
    api/
    services/
    requirements-backend.in
    requirements-backend.txt
    requirements-stats.in
    requirements-stats.txt
    requirements-dev.in
    requirements-dev.txt
  web_frontend/
    index.html
    assets/
    js/
    css/
  tile_compile_cpp/
    build/
    tile_compile_runner
    tile_compile_cli
```

Frontend-Referenz fuer Scope/Flows:

- `doc/gui2/clickdummy/*.html`
- insbesondere:
  - `dashboard.html`
  - `input-scan.html`
  - `parameter-studio.html`
  - `run-monitor.html`
  - `history-tools.html`
  - `wizard.html`

## 3) Was wird benoetigt

## 3.1 C++ Kernabhaengigkeiten (Qt6-frei)

Fuer HTML+FastAPI werden nur Runner/CLI benoetigt:

- CMake (>= 3.21 empfohlen)
- C++17 Compiler
- Eigen3
- OpenCV (>= 4.5)
- CFITSIO
- yaml-cpp
- nlohmann-json
- OpenSSL

Nicht erforderlich im Web-Stack:

- Qt6
- Qt Runtime/Plugins

## 3.1.1 Migrationsanforderung im bestehenden Repo

Der aktuelle `tile_compile_cpp/CMakeLists.txt` bindet Qt6 global ein. Fuer das Zielbild ohne Qt6 sind diese Schritte Pflicht:

1. `find_package(Qt6 ...)` nur noch ausfuehren, wenn `BUILD_QT_GUI=ON`.
2. Runner/CLI duerfen nur gegen eine Core-Lib ohne Qt6 linken.
3. Qt-abhaengiger Code (insb. Netzwerkanteile) aus Core/Runner/CLI entfernen oder hinter austauschbare Adapter legen.
4. CI-Profil `cpp-no-qt` als Standard fuer Web-Deployment einfuehren.

## 3.2 Python Backend-Abhaengigkeiten

Pflicht (Backend):

- `fastapi`
- `uvicorn[standard]`
- `pydantic`
- `pydantic-settings`
- `python-multipart`
- `PyYAML`
- `orjson` (optional, aber empfohlen fuer schnelle JSON-Antworten)

Pflicht (Stats-Button `generate_report.py`):

- `numpy`
- `matplotlib`
- `PyYAML`

Optional (nur fuer Diagnose-Skripte):

- `astropy`

Empfohlene Entwicklungs-Tools:

- `pytest`
- `httpx`
- `ruff`
- `mypy`

## 3.3 Frontend-Abhaengigkeiten

Bei Vanilla-HTML keine Build-Toolchain noetig:

- Browser (Chromium/Firefox/Safari)
- lokale Fonts als WOFF2 im Repo einbetten fuer identisches Rendering auf macOS/Windows/Linux

## 4) Installation pro Betriebssystem

Wichtig:

- Die folgenden C++ Build-Kommandos zeigen das Zielprofil `-DBUILD_QT_GUI=OFF`.
- Dieses Profil muss im aktuellen CMake noch verbindlich umgesetzt werden (siehe Abschnitt 3.1.1).

## 4.1 Linux (Ubuntu/Debian)

Systempakete:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake ninja-build pkg-config \
  libeigen3-dev libopencv-dev libcfitsio-dev libyaml-cpp-dev nlohmann-json3-dev libssl-dev \
  python3 python3-venv python3-pip
```

C++ Build (Qt-freies Zielprofil):

```bash
cd /media/data/programming/tile_compile/tile_compile_cpp
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_QT_GUI=OFF
cmake --build build -j
```

Python `venv` + Backend:

```bash
cd /media/data/programming/tile_compile
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r web_backend/requirements-backend.txt
pip install -r web_backend/requirements-stats.txt
```

## 4.2 macOS

Grundsetup:

```bash
xcode-select --install
brew install cmake ninja pkg-config eigen opencv cfitsio yaml-cpp nlohmann-json openssl python
```

C++ Build (Qt-freies Zielprofil):

```bash
cd /media/data/programming/tile_compile/tile_compile_cpp
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_QT_GUI=OFF
cmake --build build -j
```

Python `venv` + Backend:

```bash
cd /media/data/programming/tile_compile
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r web_backend/requirements-backend.txt
pip install -r web_backend/requirements-stats.txt
```

## 4.3 Windows (MSYS2 + PowerShell)

MSYS2 Pakete (MinGW64):

```bash
pacman -S --needed mingw-w64-x86_64-toolchain mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja mingw-w64-x86_64-pkgconf
pacman -S --needed mingw-w64-x86_64-eigen3 mingw-w64-x86_64-opencv mingw-w64-x86_64-cfitsio
pacman -S --needed mingw-w64-x86_64-yaml-cpp mingw-w64-x86_64-nlohmann-json mingw-w64-x86_64-openssl
```

C++ Build (MSYS2 MinGW64 Shell, Qt-freies Zielprofil):

```bash
cd /media/data/programming/tile_compile/tile_compile_cpp
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_QT_GUI=OFF
cmake --build build -j
```

Python `venv` (PowerShell):

```powershell
cd C:\path\to\tile_compile
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r web_backend\requirements-backend.txt
pip install -r web_backend\requirements-stats.txt
```

Windows Besonderheiten:

- Execution-Policy kann Aktivierung blockieren:
  - `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- Runner/CLI-Pfade im Backend immer absolut konfigurieren.

## 5) `venv` und Python-Paketmanagement

Empfehlung:

- pro Repo genau eine virtuelle Umgebung (`.venv`)
- keine globale Paketinstallation
- getrennte Requirements-Dateien

Empfohlenes Layout:

- `web_backend/requirements-backend.in`
- `web_backend/requirements-stats.in`
- `web_backend/requirements-dev.in`
- gelockte Ableitungen:
  - `requirements-backend.txt`
  - `requirements-stats.txt`
  - `requirements-dev.txt`

Lock-Management:

```bash
pip install pip-tools
pip-compile web_backend/requirements-backend.in -o web_backend/requirements-backend.txt
pip-compile web_backend/requirements-stats.in -o web_backend/requirements-stats.txt
pip-compile web_backend/requirements-dev.in -o web_backend/requirements-dev.txt
```

## 6) API/Backend-Integration mit Runner/CLI

## 6.1 Aktuelle CLI-Kommandos

`tile_compile_cli`:

- `get-schema`
- `load-gui-state`, `save-gui-state`
- `load-config`, `save-config`, `validate-config`
- `scan`
- `list-runs`, `get-run-status`, `get-run-logs`, `list-artifacts`
- `fits-stats`
- `pcc-apply`

`tile_compile_runner`:

- `run`
- `resume`

## 6.2 Endpunkt-Mapping (empfohlen)

| HTTP | Endpoint | Backend-Aktion | C++ Aufruf |
|---|---|---|---|
| `GET` | `/api/config/schema` | Schema lesen | `tile_compile_cli get-schema` |
| `POST` | `/api/config/validate` | YAML validieren | `tile_compile_cli validate-config ...` |
| `POST` | `/api/scan` | Input scan | `tile_compile_cli scan ...` |
| `POST` | `/api/runs/start` | Run starten | `tile_compile_runner run ...` |
| `POST` | `/api/runs/{run_id}/resume` | Resume ab Phase | `tile_compile_runner resume ...` |
| `GET` | `/api/runs` | Run-Liste | `tile_compile_cli list-runs ...` |
| `GET` | `/api/runs/{run_id}/status` | Status | `tile_compile_cli get-run-status ...` |
| `GET` | `/api/runs/{run_id}/logs` | Logs | `tile_compile_cli get-run-logs ...` |
| `GET` | `/api/runs/{run_id}/artifacts` | Artefakte | `tile_compile_cli list-artifacts ...` |
| `POST` | `/api/runs/{run_id}/stats` | Report erstellen | `python3 tile_compile_cpp/scripts/generate_report.py <run_dir>` |

## 6.3 Resume + Config-Versionierung

Fuer `resume` nach Parameteraenderung:

- vor jeder Aenderung Snapshot erzeugen:
  - `run_dir/config_revisions/<ts>_<rev>.yaml`
  - `run_dir/config_revisions/index.json`
- Resume request enthaelt:
  - `from_phase`
  - `config_revision_id`
- Backend stellt Restore-Endpunkt bereit:
  - `POST /api/runs/{run_id}/config-revisions/{rev_id}/restore`

Dadurch bleibt alte Konfiguration immer wiederherstellbar.

## 7) MONO Multi-Filter Queue (seriell)

Nicht-OSC Ablauf:

- UI erfasst Filterliste (z. B. `L`, `R`, `G`, `B`, `Ha`) mit jeweils:
  - `filter_name`
  - `input_dir`
  - optional `pattern`
  - `enabled`
  - optional `run_label`
- Backend validiert jede Queue-Position.
- Runner startet seriell:
  - Filter 1 komplett
  - danach Filter 2 usw.
- Gesamtstatus:
  - aktiver Filterindex
  - verbleibende Queue
  - Teilergebnisse pro Filter

Empfohlene Endpunkte:

- `POST /api/mono-queue/validate`
- `POST /api/runs/start-mono-queue`
- `GET /api/runs/{run_id}/queue-status`

## 8) Build-Management

## 8.1 C++ Build

- Buildsystem: CMake + Ninja
- Build-Typen:
  - `Debug`
  - `Release`
- Output:
  - `tile_compile_runner`
  - `tile_compile_cli`
  - kein `tile_compile_gui` im Web-Deployment

Verbindliche Build-Regel:

- `option(BUILD_QT_GUI "Build Qt GUI" OFF)` als Default im Web-Zweig
- bei `BUILD_QT_GUI=OFF`:
  - kein `find_package(Qt6 REQUIRED ...)`
  - nur Runner/CLI bauen/installieren
  - keine Qt6-Pakete als Installationsvoraussetzung

Vorteil:

- deutlich einfachere Installation fuer Web-Deployment
- weniger Abhaengigkeiten auf macOS/Windows/Linux

## 8.2 Backend Build

- kein klassischer Compile-Step
- "Build" bedeutet:
  - `venv` erstellen
  - Dependencies installieren
  - statische Frontend-Dateien bereitstellen

Start lokal:

```bash
source .venv/bin/activate
uvicorn web_backend.app:app --host 127.0.0.1 --port 8080 --reload
```

## 8.3 Release/Packaging

Empfohlenes Artefaktmodell:

- `dist/cpp/<os>/` fuer Runner/CLI (+ noetige DLL/SO/DYLIB)
- `dist/web/` fuer Backend-Code + Requirements + Frontend-Assets
- gemeinsamer Launcher pro OS:
  - startet FastAPI
  - oeffnet Browser-URL
  - setzt Pfade zu Runner/CLI

Versionierung:

- gemeinsame App-Version aus Git-Tag
- `/api/version` liefert:
  - backend version
  - runner version
  - cli version

## 9) CI/CD (empfohlen)

Pipeline-Matrix:

- Linux, Windows, macOS

Jobs:

- C++ Build + Smoke-Test (`runner --help`, `cli`)
- Python Lint/Test (`ruff`, `pytest`)
- API-Integrationstest (Mock-Run-Verzeichnis)
- Paket-Erzeugung (`dist/`)
- Artefakt-Upload pro OS

Gates vor Release:

- alle drei OS Builds gruen
- API-Endpunktvertrag unveraendert oder versioniert
- Wizard/Run/Resume E2E-Smoke erfolgreich

## 10) Installation-Checkliste (schnell)

1. C++ Toolchain + Libraries installiert.
2. `tile_compile_runner` und `tile_compile_cli` gebaut und startbar.
3. Python `venv` erstellt, Backend-Pakete installiert.
4. Stats-Dependencies (`numpy`, `matplotlib`) vorhanden.
5. Backend kann Runner/CLI-Pfade lesen.
6. Frontend laedt unter `http://127.0.0.1:8080/`.
7. Scan, Validate, Run, Resume, History, Generate Stats erfolgreich getestet.

## 11) Risiko- und Migrationshinweise

- Prioritaet fuer Migration:
  1. CMake entkoppeln, so dass Runner/CLI wirklich ohne Qt6 gebaut werden.
  2. FastAPI Adapter stabilisieren (subprocess, timeouts, error mapping).
  3. End-to-End Tests fuer Wizard + Resume + Queue.
  4. Danach Packaging verfeinern.

Damit bleibt die C++ Engine unveraendert nutzbar, waehrend die GUI auf ein wartbares HTML/FastAPI-Modell umgestellt wird.
