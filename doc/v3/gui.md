
# Tile Compile – GUI-Spezifikation (Python/Qt6)

**Status:** Normativ
**Gültigkeit:** kompatibel mit `tile_basierte_qualitatsrekonstruktion_methodik_en.md` (Methodik v3), `tile_compile.proc` (Clean Break), `tile_compile.yaml`

**Implementierung:** Python/Qt6 Desktop‑App in `gui/main.py`

---

## 1. Ziel der GUI

Die GUI dient als **Run-Controller und Monitor** für *Tile Compile* gemäß Methodik v3.
Sie ermöglicht:

* Auswahl von Eingabedaten (Input-Verzeichnis mit Frames)
* Scan und Validierung der Eingabedaten (Frameanzahl, Color Mode, Bayer Pattern)
* Auswahl und **Editieren der Konfiguration vor dem Run**
* Anzeige von **Methodik v3 Assumptions** (Hard/Soft/Implicit)
* Anzeige des **Reduced Mode** Status (50–199 Frames)
* Starten deterministischer Runs
* Überwachung laufender Runs mit **Phasen-Fortschritt** (11 Phasen)
* kontrolliertes Abbrechen von Runs
* Einsehen von Logs, Status und Validierungsartefakten

Die GUI ist **kein interaktives Bildbearbeitungs- oder Analysewerkzeug**.

---

## 2. Grundprinzipien (verbindlich)

1. **Konfiguration ist vor dem Run editierbar, danach read-only**
2. **Keine Eingriffe während der Ausführung**
3. **Jeder Run ist deterministisch und eindeutig identifiziert**
4. **Abbruch ist erlaubt, Resume nicht**
5. **GUI ist Client, nicht Rechenkern**
6. **Methodik v3 Assumptions werden vor Run-Start validiert**

---

## 3. Technologiestack

### 3.1 Frontend

* **Python 3.8+**
* **PySide6** (Qt6 Bindings)
* Styling via `styles.qss`
* Konstanten in `gui/constants.js` (JSON-Payload)

### 3.2 Backend (lokal)

* Python-Backend via `tile_compile_backend_cli.py`
* Aufgaben:
  * Schema-Validierung (`get-schema`, `validate-config`)
  * Input-Scan (`scan`)
  * Siril-Script-Validierung (`validate-siril-scripts`, `validate-ssf`)
  * Run-Management (`list-runs`, `get-run-status`, `get-run-logs`, `list-artifacts`)
  * GUI-State Persistenz (`load-gui-state`, `save-gui-state`)
* Kommunikation:
  * Subprocess-Aufrufe mit JSON-Ausgabe

### 3.3 Runner

* `tile_compile_runner.py` als separater Prozess
* Kommunikation via stdout (JSON-Events)
* Signal-basierter Abbruch (SIGINT)

---

## 4. Methodik v3 Integration

### 4.1 Pipeline-Phasen (normativ)

Die GUI zeigt den Fortschritt der 11 Methodik v3 Phasen:

| Phase | Name | Beschreibung |
|-------|------|--------------|
| 0 | SCAN_INPUT | Eingabe-Validierung |
| 1 | REGISTRATION | Frame-Registrierung |
| 2 | CHANNEL_SPLIT | Kanal-Trennung (R/G/B) |
| 3 | NORMALIZATION | Globale lineare Normalisierung |
| 4 | GLOBAL_METRICS | Globale Frame-Metriken (B, σ, E) |
| 5 | TILE_GRID | Seeing-adaptive Tile-Geometrie |
| 6 | LOCAL_METRICS | Lokale Tile-Metriken |
| 7 | TILE_RECONSTRUCTION | Tile-weise Rekonstruktion |
| 8 | STATE_CLUSTERING | Zustandsbasiertes Clustering |
| 9 | SYNTHETIC_FRAMES | Synthetische Qualitätsframes |
| 10 | STACKING | Finales lineares Stacking |
| 11 | DONE | Abschluss |

### 4.2 Assumptions-Anzeige

Die GUI zeigt vor Run-Start die Methodik v3 Assumptions:

**Hard Assumptions (Verletzung → Abbruch):**
* Lineare Daten (kein Stretch)
* Keine Frame-Selektion
* Kanal-getrennte Verarbeitung
* Einheitliche Belichtungszeit (±5%)

**Soft Assumptions (mit Toleranzen):**
* Frame-Anzahl: optimal ≥800, minimum ≥50
* Registrierungs-Residual: <0.3 px optimal, <1.0 px minimum
* Stern-Elongation: <0.2 optimal, <0.4 maximum

**Reduced Mode (50–199 Frames):**
* Überspringt STATE_CLUSTERING
* Überspringt SYNTHETIC_FRAMES
* Direktes Tile-gewichtetes Stacking
* Validierungs-Warnung im Report

### 4.3 Konfigurationsvalidierung

Vor Run-Start werden validiert:
* Schema-Konformität (`tile_compile.schema.json`)
* Gewichtsnormalisierung (α + β + γ = 1)
* Clamp-Bereiche ([-3, +3])
* Siril-Script-Policy (keine Stretch-Befehle)
* Assumptions-Thresholds (frames_min < frames_reduced_threshold < frames_optimal)

---

## 5. Konfigurationsmodell

### 5.1 Konfigurationsquelle

Die GUI arbeitet kontextbezogen zu einem Input-Verzeichnis.

* Rohframes (*.fit, *.fits, *.fts)
* optional: `tile_compile.yaml`

### 5.2 Konfigurationslogik

1. Falls `tile_compile.yaml` vorhanden → laden und editierbar anzeigen
2. Falls nicht vorhanden → Erstellung aus Template anbieten
3. Nach Run-Start → Konfiguration einfrieren und hashen (`config_hash`)

Ein Run referenziert **immer genau eine Konfiguration**.

### 5.3 Konfigurationseditor

* YAML-Texteditor mit Syntax-Highlighting
* Schema-basierte Validierung
* Methodik-Verstöße blockieren Run-Start
* Assumptions-Tab für strukturierte Bearbeitung

---

## 6. GUI-Tabs

### 6.1 Scan

* Input-Verzeichnis auswählen (Browse)
* Frames-Minimum setzen
* Checksummen optional berechnen
* Color Mode bestätigen (OSC/Mono)
* Anzeige: frames_detected, color_mode, bayer_pattern

### 6.2 Configuration

* YAML-Editor für `tile_compile.yaml`
* Load / Save / Validate Buttons
* Validierungsstatus (ok / invalid / not validated)
* Fehleranzeige bei Validierungsproblemen

### 6.3 Assumptions (NEU)

* Strukturierte Anzeige der Methodik v3 Assumptions
* Hard/Soft/Implicit Assumptions
* Reduced Mode Indikator
* Editierbare Toleranzen:
  * `frames_min`, `frames_optimal`, `frames_reduced_threshold`
  * `registration_residual_warn_px`, `registration_residual_max_px`
  * `elongation_warn`, `elongation_max`
  * `reduced_mode_skip_clustering`, `reduced_mode_cluster_range`

### 6.4 Run

* Working dir / Input dir / Runs dir
* Pattern für Frame-Erkennung
* Dry-run Option
* Start / Abort Buttons
* Status-Anzeige (idle / running / finished)

### 6.5 Pipeline Progress (NEU)

* Visuelle Darstellung der 11 Phasen
* Aktueller Phasen-Status (pending / running / ok / error / skipped)
* Reduced Mode Indikator für Phase 8 + 9
* Phasen-Dauer und Metriken

### 6.6 Current Run

* run_id und run_dir Anzeige
* Status-Refresh
* Log-Viewer mit Filter und Tail
* Artefakt-Liste

### 6.7 Run History

* Tabelle vergangener Runs
* Spalten: created_at, run_id, status, config_hash, frames_manifest_id
* Run-Auswahl für Detail-Ansicht

### 6.8 Live Log

* Echtzeit-Ausgabe des Runner-Prozesses
* JSON-Events formatiert
* Backend-Kommandos geloggt

---

## 7. Einschränkungen

Die GUI darf **keine** Parameter während eines Runs ändern und keine Bildinteraktion erlauben.

Verboten:
* Frame-Selektion
* Stretch/Normalisierung außerhalb der Pipeline
* Manuelle Gewichtsanpassung während des Runs
* Resume nach Abbruch

---

## 8. Zusammenfassung

Die GUI ist ein deterministischer, reproduzierbarer Run-Controller für Methodik v3 mit:
* Vollständiger Phasen-Überwachung (11 Phasen)
* Assumptions-Validierung vor Run-Start
* Reduced Mode Unterstützung (50–199 Frames)
* Kontrollierter Abbruchmöglichkeit
* Keine Eingriffe in die Methodik
