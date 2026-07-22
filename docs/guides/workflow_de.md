# Typischer Workflow (GUI3)

Der Standard-Workflow mit GUI3 umfasst drei Schritte:

1. **Input scannen** — Tab *Processing → Input & Scan*: Input-Ordner mit FITS-Lights auswählen, optional Kalibrierungsframes (Bias/Dark/Flat) angeben, Scan starten. Der Scan erkennt Frames, Auflösung und Farbmodus.
2. **Parameter anpassen** — Tab *Processing → Parameter*: Beispiel-Konfiguration laden oder Werte anpassen. Wichtige Parameter: Registration (Rotation, Transformationsmodell), AQMH (Cherry-Pick, Pyramiden-Skalen), Stacking-Methode, Bayer-Pattern. Konfiguration validieren und speichern.
3. **Run starten und überwachen** — Tab *Processing → Run Monitor*: Run starten, Phasenfortschritt in Echtzeit verfolgen, bei Bedarf abbrechen oder ab einer bestimmten Phase fortsetzen.

Nach Abschluss: Ergebnisse liegen in `runs/<run_id>/outputs/`. Diagnosebericht über *Generate Stats* im Run Monitor oder über die Run-History erzeugen.

Vollständige Anleitung: [GUI3 Benutzerhandbuch](../gui3_user_guide_de.md)

## Pipeline-Phasen

| ID | Phase | Beschreibung |
|----|-------|-------------|
| 0 | SCAN_INPUT | Input-Erkennung, Moduserkennung, Linearitätsprüfung, Speicherplatz-Precheck |
| 1 | REGISTRATION | Kaskadierte globale Registrierung |
| 2 | PREWARP | Full-Frame-Canvas-Prewarp (CFA-sicher für OSC) |
| 3 | CHANNEL_SPLIT | Metadaten-Phase (Kanalmodell) |
| 4 | NORMALIZATION | Lineare hintergrundbasierte Normalisierung |
| 5 | GLOBAL_METRICS | Globale Frame-Metriken und Gewichte |
| 6 | TILE_GRID | Adaptive Tile-Geometrie (für klassische TILE_RECONSTRUCTION) |
| 7 | COMMON_OVERLAP | Gemeinsamer gültiger Daten-Overlap (globale/tile-lokale Masken) |
| 8 | LOCAL_METRICS | Lokale Tile-Metriken + **AQMH-Qualitätskarten-Berechnung** |
| 9 | TILE_RECONSTRUCTION | **AQMH pixelgenaue gewichtete Rekonstruktion** (Standard) oder Tile-gewichtete OLA (klassisch) |
| 10 | STATE_CLUSTERING | Optionales State-Clustering |
| 11 | SYNTHETIC_FRAMES | Optionale synthetische Frame-Erzeugung |
| 12 | STACKING | Finale lineare Stacking |
| 13 | DEBAYER | OSC-Demosaic zu RGB (MONO-Durchlauf) |
| 14 | ASTROMETRY | Plate Solving / WCS |
| 15 | BGE | Optionale RGB-Hintergrund-Gradient-Extraktion vor PCC |
| 16 | PCC | Photometrische Farbkalibrierung |
| 17 | HYPERMETRIC_STRETCH | Optionaler VeraLux HyperMetric Stretch nach PCC |
| 18 | DONE | Endstatus (`ok` oder `validation_failed`) |

Detaillierte Phasen-Dokumentation: [Process Flow](../process_flow/phase_0_overview.md)

## Registrierungs-Kaskade (Fallback-Strategie)

| Stufe | Methode | Typischer Anwendungsfall |
|-------|---------|--------------------------|
| 1 | Primäre Engine (`triangle_star_matching`) | Normale sternreiche Frames |
| 2 | Trail-Endpoint-Registrierung | Sternspuren / rotationsreiche Daten |
| 3 | AKAZE-Feature-Matching | Allgemeiner Feature-Fallback |
| 4 | Robuste Phase+ECC | Wolken/Nebel mit größeren Transformationen |
| 5 | Hybride Phase+ECC | Schwache Stern-Matching-Fälle |
| 6 | Identity-Fallback | Letzter Ausweg (CC=0, Frame behalten) |

## Konfiguration

- Haupt-Konfigurationsdatei: `tile_compile.yaml`
- Schemas: `tile_compile.schema.json`, `tile_compile.schema.yaml`
- Referenzdokument: [Konfigurationsreferenz](../configuration_reference.md)
- Praktische Beispiele: [Konfigurationsbeispiele & Best Practices](../configuration_examples_practical_de.md)

### Beispielprofile

Vollständige eigenständige Beispiel-Konfigurationen unter `tile_compile_cpp/examples/`.

- `full_mode.example.yaml`
- `reduced_mode.example.yaml`
- `emergency_mode.example.yaml`
- `smart_telescope_dwarf_seestar.example.yaml`
- `smart_telescope_very_bright_star.example.yaml`
- `canon_low_n_high_quality.example.yaml`
- `very_bright_star_anti_seam.example.yaml`
- `canon_equatorial_balanced.example.yaml`
- `mono_full_mode.example.yaml`
- `mono_small_n_anti_grid.example.yaml` (empfohlen für MONO-Datensätze mit wenigen Frames, z.B. ~10..40, zur Reduzierung des Tile-Muster-Risikos)
- `mono_small_n_ultra_conservative.example.yaml` (empfohlen für sehr kleine MONO-Datensätze, z.B. ~8..25, wenn Nahtstabilität wichtiger ist als aggressive Verbesserung)

Siehe auch: [Examples README](https://github.com/jeamy/tile_compile/blob/master/tile_compile_cpp/examples/README.md) für den vorgesehenen Anwendungsfall und Abstimmungsschwerpunkt jedes Profils.

## Binary-Releases (GUI3)

Vorgefertigte GUI3-Release-Bundles werden über [GitHub Releases](https://github.com/jeamy/tile_compile/releases) veröffentlicht.

Jedes Bundle enthält:

- GUI3-Frontend (`web_frontend_v3/`)
- Crow-Backend (`web_backend_cpp/`)
- Native C++-Tools (`tile_compile_runner`, `tile_compile_cli`, `tile_compile_web_backend`)
- Launcher für Linux, macOS und Windows
- Optionaler PI-AI-Sidecar (`agent_service/`, benötigt Node.js >= 20)

Zur Laufzeit verwendet GUI3 das lokale Crow/C++-Backend als Prozess-Adapter für den C++-Runner/CLI.

## Schnellstart

### GUI3

Entwicklungsstart vom Repository-Root:

```bash
./start_backend.sh
```

Dann öffnen:

```text
http://127.0.0.1:8080/ui/
```

Release-Bundle-Start:

- Linux: `start_gui3.sh`
- macOS: `start_gui3.command`
- Windows: `start_gui3.bat`

Der Launcher kopiert die gebündelten Dateien in ein benutzerspezifisches Installationsverzeichnis, startet das Crow-Backend im Vordergrund und öffnet den Browser mit der lokalen GUI3-URL.

**Installations- und Update-Verhalten:**

- Beim ersten Start kopiert der Launcher alle Anwendungsdateien nach `~/tilecompile/` (Linux/macOS) oder `%USERPROFILE%\tilecompile\` (Windows).
- Nach dem ersten erfolgreichen Start kann das heruntergeladene Paketarchiv und der entpackte Ordner sicher gelöscht werden — alle Daten wurden ins Benutzerverzeichnis kopiert.
- Bei Updates werden nur die Anwendungsdateien (`web_frontend_v3/`, `web_backend_cpp/`, `tile_compile_cpp/`, `agent_service/`) ersetzt. Benutzerdaten (Konfigurationen, Runs, ASTAP-Katalog, PCC-Datenbank) bleiben unangetastet.

macOS-Installationshinweis:

- Auf macOS 15.x (inklusive Sequoia 15.1) bietet Gatekeeper möglicherweise nicht mehr den älteren Rechtsklick-Override-Pfad für unbekannte Entwickler. Wenn `start_gui3.command` oder andere Skripte blockiert werden, `Systemeinstellungen → Datenschutz & Sicherheit` öffnen, nach unten scrollen und die blockierte `start_gui3.command` dort explizit erlauben, bevor sie erneut gestartet wird.

Mindest-Betriebssystemversionen für die aktuellen GUI3-Release-Bundles:

- Linux: x86_64 Linux mit `glibc >= 2.39` (Ubuntu 24.04 oder äquivalent ist die sichere Basis für die aktuellen CI-gebauten ZIPs)
- macOS: macOS 15
- Windows: Windows 10 x64 oder neuer

Hinweise:

- macOS-Release-Bundles sind mit einem expliziten Deployment-Target gebaut und sollen ab macOS 13 laufen.
- Linux-Bundles bündeln keine `glibc`, daher werden ältere Distributionen als die aktuelle Build-Basis nicht garantiert unterstützt.
- Der optionale PI-AI-Sidecar (`agent_service/`) benötigt **Node.js >= 20**. Wenn Node.js nicht installiert oder zu alt ist, startet das Backend ohne AI-Sidecar und gibt eine Warnung aus. Siehe [GUI3 README](https://github.com/jeamy/tile_compile/blob/master/packaging/gui3/README.md) für Details.

## Paper-Beispieldaten

- M31-Lights für das Paper-Beispiel (10 GB): [M31 Lights](https://wolke.eibrain.org/index.php/s/Z88dmWizEJYjwBe)
- M31-Run für das Paper-Beispiel (20 GB): [M31 Run](https://wolke.eibrain.org/index.php/s/tfSycSNEzdL7jje)
