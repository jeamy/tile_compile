# Tile-Compile

Tile-Compile ist ein Toolkit für hochwertige astronomische Bildrekonstruktion aus Kurzzeitbelichtungs-Deep-Sky-Datensätzen. Die Standard-Rekonstruktionsmethode ist **AQMH (Adaptive Quality Map Hyperstacking)** — ein pixelgenauer, qualitätskarten-basierter Ansatz, der das tile-basierte Overlap-Add-Stacking durch einen physikalisch optimalen pixelweisen gewichteten Mittelwert ersetzt.

> **Classic Tile-Compile (TBQR):** Die ursprüngliche tile-basierte Qualitätsrekonstruktions-Methodik ist weiterhin verfügbar und vollständig unterstützt. Siehe [Classic Tile-Compile README (EN)](README_classic_tile_compile_en.md) und [Classic Tile-Compile README (DE)](README_classic_tile_compile_de.md). Mit `aqmh.enabled: false` wird auf die klassische TILE_RECONSTRUCTION zurückgefallen.

> **Hinweis:** Dies ist experimentelle Software, die primär für die Verarbeitung von Bildern von Smart-Teleskopen entwickelt wurde (z.B. DWARF, Seestar, ZWO SeeStar, usw.). Obwohl sie für die allgemeine astronomische Bildverarbeitung konzipiert ist, wurde sie für die spezifischen Eigenschaften und Herausforderungen von Smart-Teleskop-Daten optimiert.

## Schnellstart

### GUI3

Pre-built Bundle von [GitHub Releases](https://github.com/jeamy/tile_compile/releases) herunterladen, oder aus dem Quellcode bauen (siehe [Installation](docs/getting_started/installation_de.md)) und aus dem Repository starten:

```bash
./start_backend.sh
```

Dann öffnen: http://127.0.0.1:8080/ui/

Release-Bundle starten:

- Linux: `start_gui3.sh`
- macOS: `start_gui3.command`
- Windows: `start_gui3.bat`

### CLI

```bash
./tile_compile_runner run \
  --config tile_compile.yaml \
  --input-dir /pfad/zu/lights \
  --runs-dir /pfad/zu/runs
```

### Docker

```bash
./start_gui3_docker.sh
```

Öffnen: http://127.0.0.1:8080/ui/

## Dokumentation

Vollständige Dokumentation: **[https://jeamy.github.io/tile_compile/](https://jeamy.github.io/tile_compile/)**

### Erste Schritte

- [Schnellstart](docs/getting_started/quickstart.md)
- [Installation](docs/getting_started/installation.md)
- [CLI-Referenz](docs/reference/cli.md)
- [Konfiguration](docs/getting_started/configuration.md)

### Benutzerhandbücher

- [GUI3 Benutzerhandbuch (DE)](docs/gui3_user_guide_de.md) — Vollständige Schritt-für-Schritt-Anleitung
- [GUI3 User Guide (EN)](docs/gui3_user_guide_en.md)
- [Workflow & Pipeline-Phasen](docs/guides/workflow.md) — Typischer GUI3-Workflow, Phasentabelle, Registrierungskaskade
- [Raw Stack GUI](docs/guides/raw_stack_gui.md) — Eigenständige Vorverarbeitungs-Pipeline (nicht optimiert, nur aus historischen Gründen vorhanden)
- [PI – KI-gestützte Empfehlungen](docs/guides/pi_ai.md) — Datengetriebene Parameterempfehlungen
- [Live Image Editor (DE)](docs/guides/live_image_editor_de.md) — Nicht-destruktive FITS-Bearbeitung, Preview, Undo/Redo, Wiederholen und KI-/lokaler Fallback

### AQMH

- [AQMH-Übersicht](docs/guides/aqmh_overview.md) — Funktionsweise, Hauptparameter, Einsatzgebiete
- [AQMH-Methodik v0.2.1 (normativ)](docs/AQMH/aqmh_methodik_en_v0.2.1.md)
- [AQMH v0.2.0 Paper (PDF)](docs/AQMH/zenodo-0.2.0/paper-adaptive_quality_map_hyperstacking_m31_run_20260722_en.pdf)
- [AQMH v0.1.0 Paper](docs/AQMH/zenodo-0.1.0/)

### Konfiguration

- [Konfigurationsreferenz (DE)](docs/configuration_reference.md)
- [Konfigurationsreferenz (EN)](docs/configuration_reference_en.md)
- [Praxisbeispiele (DE)](docs/configuration_examples_practical_de.md)
- [Praxisbeispiele (EN)](docs/configuration_examples_practical_en.md)
- Beispielprofile: `tile_compile_cpp/examples/`

### Referenz

- [Aus dem Source bauen](docs/reference/build.md) — Build-Anforderungen, GPU-Beschleunigung, Paketinstallation
- [Docker](docs/reference/docker.md) — Container bauen, ausführen und konfigurieren
- [CLI-Befehle](docs/reference/cli.md) — Runner, Scan, Config, Resume, Report-Generierung
- [Outputs & Artifacts](docs/reference/outputs.md) — Run-Output-Verzeichnisstruktur
- [Kalibrierung & externe Tools](docs/reference/calibration.md) — Bias/Dark/Flat, ASTAP, Siril-Katalog
- [Projektstruktur](docs/reference/project_structure.md) — Repository-Layout und Komponenten

### Methodik

- [AQMH-Methodik v0.2.1](docs/AQMH/aqmh_methodik_en_v0.2.1.md) — Aktuelle AQMH-normative Spezifikation
- [TBQR-Methodik v3.3.9 (DE)](docs/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.9_de.md)
- [TBQR-Methodik v3.3.9 (EN)](docs/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.9_en.md)
- [Process Flow](docs/process_flow/) — Phasenweise Implementierungsdokumente

### Changelog

- [Release Notes](docs/changelog/releases.md)
- [Detailliertes Changelog](docs/changelog/detailed_changelog.md)

### Weitere Sprachen

- [English README](README.md)
- [Classic README (EN)](README_classic_tile_compile_en.md)
- [Classic README (DE)](README_classic_tile_compile_de.md)

## Attribution

Dieses Projekt wurde mit Unterstützung von Windsurf-Devin, Kiro, Antigravity, GPT, Claude, Codex, *** erstellt. Babysitting by a human in a virtual environment.

Das PI (Parameter Intelligence) Modul verwendet:

- **[@earendil-works/pi-coding-agent](https://github.com/earendil-works/pi/tree/main/packages/coding-agent)** — KI-Agent-Framework (v0.80.x)

Die HyperMetric Stretch (HMS) Phase basiert auf dem VeraLux HyperMetric Stretch Siril-Skript:

- (c) 2025 Riccardo Paterniti — VeraLux - HyperMetric Stretch — GPL-3.0-or-later — Version 1.5.2

Die AutoBGE (Background Gradient Extraction) Phase basiert auf dem AutoBGE Siril-Skript:

- (c) Adrian Knagg-Baugh from Franklin Marek SAS code (2025) — GPL-3.0-or-later — Version 2.0.2
