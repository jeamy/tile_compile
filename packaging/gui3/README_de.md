# tile_compile GUI3 Release-Bundle

Dieses Verzeichnis enthält die Starter-Skripte und Paketierungshilfen für das GUI3-Release-Bundle. GUI3 besteht aus dem Web-Frontend v3, dem Crow/C++-Backend und dem nativen C++-Runner/CLI.

## Bundle-Layout

Das generierte Release-Archiv enthält:

- `start_gui3.sh` für Linux
- `start_gui3.command` für macOS
- `start_gui3.bat` und `start_gui3.ps1` für Windows
- `payload/` mit:
  - `web_backend_cpp/`
  - `web_frontend_v3/`
  - `tile_compile_cpp/build/` mit `tile_compile_runner` und `tile_compile_cli`
  - `web_backend_cpp/build/` mit `tile_compile_web_backend`
  - `tile_compile_cpp/lib/` mit gebündelten nativen Runtime-Bibliotheken
  - `tile_compile_cpp/examples/`
  - `tile_compile_cpp/tile_compile.yaml`
  - `tile_compile_cpp/tile_compile.schema.yaml`
  - `tile_compile_cpp/tile_compile.schema.json`

## Erster Start

Der Starter läuft nicht dauerhaft direkt aus dem entpackten Archiv. Beim Start kopiert er die gebündelte Payload in ein benutzerspezifisches Installationsverzeichnis:

- Linux/macOS: `~/tilecompile`
- Windows: `%USERPROFILE%\\tilecompile`

Anschließend:

1. kopiert er die gebündelte Payload in das Benutzerverzeichnis
2. verwendet die gebündelten nativen Bibliotheken weiter
3. startet das Crow-Backend im Vordergrund
4. öffnet den Browser auf `/ui/`

Es wird keine Python-Runtime, keine virtuelle Umgebung und keine pip-Installation im produktiven Release-Pfad benötigt.

Das Backend wird bewusst unter der Starter-Shell gestartet, damit das Terminal-Fenster nach dem Start verbunden bleibt und der Server direkt mit `Ctrl+C` auf Linux/macOS oder im Starter-Konsolenfenster auf Windows gestoppt werden kann.

## Mindestbetriebssysteme

Aktuelle praktische Mindestvoraussetzungen für die gepackten GUI3-Release-Bundles:

- Linux: x86_64-Linux mit `glibc >= 2.39` (der aktuelle Release-Workflow baut auf Ubuntu 24.04; Ubuntu 24.04 oder äquivalent ist die sichere Basis)
- macOS: macOS 15
- Windows: Windows 10 x64 oder neuer

Hinweise:

- macOS-Release-Bundles werden mit explizitem Deployment-Target gebaut und sind ab macOS 13 lauffähig, nicht nur auf der exakten Build-Host-Version.
- Linux-Kompatibilität unterhalb der CI-Build-Basis ist für die aktuellen ZIP-Bundles nicht garantiert, da `glibc` nicht gebündelt wird.
- Windows-Paketierung wird auf `windows-2022` gebaut und per Smoke-Test geprüft; Windows 10/11 x64 ist die vorgesehene Basis.

## Build-Abhängigkeiten

Die aktuellen nativen C++-Build-Voraussetzungen für das GUI3-Release sind:

- Linux: `libcurl4-openssl-dev`
- macOS: `curl`
- Windows MSYS2: `mingw-w64-x86_64-curl`

Weitere Kern-Abhängigkeiten umfassen Eigen, OpenCV, cfitsio, yaml-cpp, nlohmann-json und OpenSSL.

macOS-Hinweise:

- `packaging/gui3/build_local_macos.sh` erfordert `xcode-select --install`, `cmake`, `ninja`, `pkg-config` und `python3`.
- Auf macOS 12 wird die Standard-`opencv`-Formel von Homebrew nicht unterstützt. Der Homebrew-basierte Paketierungspfad erfordert daher faktisch macOS 15, sofern OpenCV nicht aus einer anderen funktionierenden Installation bereitgestellt wird.

## CI-Workflow

Der GitHub-Actions-Workflow ist:

- `.github/workflows/release-tile-compile-gui3.yml`

Er baut die Runner-Binaries und das Crow-Backend, bündelt die GUI3-Dateien, kopiert native Runtime-Bibliotheken, führt einen Smoke-Test durch und lädt die Release-ZIP-Artefakte hoch.

## Lokale Paketierung

Um das Release-Paketieren lokal zu reproduzieren, verwenden Sie die Skripte in diesem Verzeichnis:

- Linux: `packaging/gui3/build_local_linux.sh`
- macOS: `packaging/gui3/build_local_macos.sh`
- Windows (MSYS2 MinGW64): `packaging/gui3/build_local_windows_msys2.sh`

Sie spiegeln den Release-Workflow eng:

1. `tile_compile_cpp` bauen (`tile_compile_runner`, `tile_compile_cli`)
2. `tile_compile_web_backend` bauen
3. GUI3-Bundle mit `payload/` zusammenstellen
4. native Runtime-Bibliotheken sammeln
5. Smoke-Test gegen `/api/app/state` ausführen
6. ZIP-Artefakt in `artifacts/` erzeugen

Beispiele:

```bash
packaging/gui3/build_local_linux.sh --tag dev
packaging/gui3/build_local_macos.sh --tag dev
packaging/gui3/build_local_windows_msys2.sh --tag dev
```

Allgemeine Optionen:

- `--skip-build` – vorhandene Build-Verzeichnisse wiederverwenden
- `--skip-smoke` – Start-Test überspringen
- `--build-type <type>` – CMake-Konfiguration wechseln
- `--port <port>` – Port für den Smoke-Test ändern

## Umgebungsvariablen-Beispiel

Eine Beispiel-Umgebungsdatei ist verfügbar unter:

- `packaging/gui3/.env.example`

Sie dokumentiert die unterstützten GUI3-Starter-Variablen und die Backend-Speicher-Guard-Limits.
Die Starter-Skripte laden diese Datei nicht automatisch; sourcen Sie sie manuell vor dem Start von GUI3, wenn Sie Defaults überschreiben möchten.
