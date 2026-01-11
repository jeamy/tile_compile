# Portierung: Python/Qt -> C++/Qt (tile-compile)

## Ziel

- **GUI**: von Python/Qt (aktuelles `gui/`) auf **C++/Qt6 (Widgets)** umstellen.
- **Repo-Struktur**: 
  - **Python**-Code + Skripte, die Python bedienen, nach `python/` verschieben.
  - **C++/Qt**-Code + Build/Dependencies nach `cpp/` anlegen.
- **Portierungsumfang**: **Full Port** (GUI + Backend/Runner/Algorithmen nach C++).
- **Übergang**: kontrollierte Migration mit Parallelbetrieb (Python-Implementierung bleibt temporär als Referenz/Regression-Oracle lauffähig).

## Nicht-Ziele / Annahmen

- Keine funktionalen Änderungen „nebenbei“ (nur Portierung, Bugfixes separat).
- Keine lokalen Installationsschritte ohne explizite Aufforderung; bevorzugt Docker/CI.

## Ausgangslage (kurz)

- `gui/` enthält die aktuelle Python/Qt GUI.
- Der Kern/Backend-Teil scheint Python-basiert (`tile_compile_backend/`, `runner/`, CLI/Runner-Skripte).
- Es gibt zusätzlich `gui-tauri-legacy/` (historisch, derzeit nicht Ziel der Portierung).

## Ziel-Repo-Struktur (Vorschlag)

```
/ (repo root)
  python/
    gui/                 # bisheriges gui/ (Python/Qt)
    runner/
    tile_compile_backend/
    siril_scripts/
    tests/
    scripts/             # *.sh, *.py, helper tooling
    requirements.txt / pyproject.toml (optional: spiegeln oder referenzieren)

  cpp/
    CMakeLists.txt
    cmake/
    third_party/         # optional, falls vendored deps nötig
    app/
      src/
      include/
      resources/         # qrc, icons, translations
    tests/

  doc/
    port_to_cpp.md
```

### Verschiebe-Regel (konkret)

- **Nach `python/`**
  - Alles, was *direkt* Python-Code ist.
  - Alle Skripte/Tools, die *primär* Python ausführen/steuern (z.B. `run-cli.sh`, `tile_compile_runner.py`, `start_gui.*` sofern Python-GUI).
  - Python-Tests.

- **Nach `cpp/`**
  - Neue C++/Qt-App (GUI) inkl. Buildsystem.
  - C++-Tests.
  - Qt-Ressourcen (Icons, `.qrc`, Übersetzungen).

- **Im Root lassen** (typisch)
  - Top-level README, Lizenz, ggf. CI/Container-Konfig.
  - Gemeinsame Doku.

## Architektur-Entscheidung (wichtig): Was wird portiert?

Festgelegt:

1. **Qt**: **Qt6**
2. **UI**: **Widgets**
3. **Scope**: **Full Port** (GUI + Backend/Runner/Algorithmen nach C++)

### Übergangs-/Kompatibilitätsstrategie (Python als Referenz)

Auch beim Full Port ist Python während der Migration nützlich als:

- **Referenz-Implementierung** für Ergebnisvergleich (Regression/Golden Master).
- **Fallback** für Teilstrecken, solange C++-Module noch nicht vollständig sind.

Pragmatischer Übergang (falls nötig):

- **CLI/Process-Boundary**: C++ ruft temporär Python-Runs auf (Konfig-Datei rein, Artefakte/Logs raus).
- Ziel ist, diese Boundary **schrittweise zu eliminieren**, sobald die jeweilige C++-Komponente portiert ist.

## Migrationsphasen (Portierungsplan)

### Phase 0: Vorbereitung (Analyse & Stabilisierung)

- **Inventar**
  - Welche GUI-Views/Flows existieren in `gui/`?
  - Welche Backend-Aufrufe triggert die GUI?
  - Welche Config-Formate sind „Source of Truth“ (z.B. `tile_compile.yaml`, Schema-Dateien)?

- **Schnittstellen definieren**
  - Eine eindeutige Beschreibung: Inputs/Outputs, Status/Progress, Fehlercodes.
  - Entscheiden, ob Status über stdout JSON-lines, Logfile oder API läuft.

- **Testbasis**
  - Kritische End-to-End-Pfade identifizieren (Happy path, typische Fehlerfälle).

**Artefakte**:
- `doc/port_to_cpp.md` (dieses Dokument)
- ggf. `doc/cpp_backend_interface.md` (optional)

### Phase 1: Repo-Umstrukturierung (ohne funktionale Änderungen)

- `gui/` -> `python/gui/`
- `runner/` -> `python/runner/`
- `tile_compile_backend/` -> `python/tile_compile_backend/`
- `tests/` -> `python/tests/` (oder bei getrennter Teststrategie im Root belassen, aber sauber konfigurieren)
- Skripte, die Python starten, nach `python/scripts/`.

**Wichtig**: 
- Imports/Module-Pfade anpassen (z.B. `PYTHONPATH`/Package-Struktur).
- Startskripte/README aktualisieren (aber in separatem Commit).

### Phase 2: C++/Qt Projektgerüst erstellen

- `cpp/CMakeLists.txt` mit:
  - Qt6
  - `AUTOMOC`, `AUTOUIC`, `AUTORCC`
  - Separate Targets: `app`, `libcore` (Backend-Logik) und optional `libcli`/`cli`.

- Basis-App:
  - MainWindow
  - Routing/Navigation (falls mehrere Views)
  - Settings/Config Reader

- Ressourcen:
  - `resources.qrc`
  - Styling: Qt Stylesheets.

### Phase 3: Backend/Runner-Portierung (Full Port - Kern zuerst)

Ziel: den ausführbaren Kern (ohne GUI) in C++ nachbauen, damit die GUI später nur noch „Client“ ist.

- Portierungsreihenfolge (empfohlen):
  - **Config/Schema/IO** (YAML/JSON, Pfade, Validierung)
  - **Orchestrierung** (Runner/Phasenmodell)
  - **Algorithmen/Module** (z.B. Clustering/Stacking/Artefakte)
  - **Siril-Integration** (Script-Calls, Prozesssteuerung, Log parsing)

**Qualitätssicherung**:
- Für jede portierte Einheit: Vergleich gegen Python-Referenz (Golden Master / toleranzbasierte Metriken).
- Tests: Unit + kleine Integrationsläufe.

### Phase 4: CLI & Headless Betrieb in C++

- Ein C++ CLI-Entry-Point, der die gleiche Pipeline wie Python ausführt (ohne GUI).
- Damit kann CI/Regression unabhängig von GUI laufen.

### Phase 5: GUI-Portierung (Qt6 Widgets)

Für jede GUI-Funktion:

- UI in C++/Qt Widgets nachbauen.
- Aufrufe an den C++-Kern (`libcore`) anbinden.
- Progress/Cancel/Retry sauber abbilden.
- Fehlerdarstellung: technische Fehler in userfreundliche Messages mappen.

### Phase 6: Paralleler Betrieb & Umschalten

- Beide GUIs lauffähig:
  - `python/gui` bleibt als Fallback.
  - `cpp/app` wird default, sobald Feature-Parität erreicht.

- Smoke-Tests / Regression:
  - Reproduzierbare Testdaten / kleine Dataset-Samples.

### Phase 7: Python-Deprecation (nach Full Port)

- Python bleibt optional als Referenz/Debug-Tool.
- Produktivpfad: C++ only.
- Langfristig: Python-Verzeichnisse nur noch für Tests/Legacy, oder komplett entfernen (separates Projekt/Branch).

## Build/Packaging/Deployment

### C++/Qt

- CMake + Ninja (typisch)
- Packaging:
  - Linux: AppImage / deb / rpm (später)
  - Qt deployment tools (`linuxdeployqt`) *erst später*, wenn nötig.

### Python

- Entweder weiterhin via venv/Docker
- Oder als „embedded“ Runtime-Bundle (nur wenn unbedingt erforderlich)

## Risiken & Stolpersteine

- **Qt6 Widgets**: UI ist „klassisch“; modernes UX braucht sauberes Layout/Stylesheet-Design.
- **Threading/Progress**: C++ UI darf nicht blockieren; Worker-Threads + Signals.
- **Fehler-/Loghandling**: Einheitliche Fehlercodes/Struktur wichtig.
- **Pfad-/Config-Management**: Absolut/relativ, Working Directory, portability.

## Empfohlene Commit-Strategie

- Kleine, nachvollziehbare Commits:
  - 1) Nur Verschieben + Import-Fixes
  - 2) C++ Skeleton ohne Feature
  - 3) Feature-Portierung in Scheiben

## Festgelegt / Nächste Schritte

Festgelegt:

1. **Qt**: Qt6
2. **UI**: Widgets
3. **Scope**: Full Port

Nächste Schritte (konkret):

1. Repo-Umstrukturierung nach `python/` und `cpp/` (nur Move + Import/Pfade fixen).
2. `cpp/` Skeleton mit CMake + minimalem Qt6 Widgets App-Start.
3. `libcore` anlegen und zuerst Config/IO/Runner in C++ portieren.
4. C++ CLI bauen (headless) und Python-Referenztests etablieren.
5. GUI portieren und final auf C++ default umschalten.
