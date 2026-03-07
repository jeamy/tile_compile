# Detailkonzept GUI 2

## 1) Informationsarchitektur

Primare Navigation (persistent):

1. Dashboard
2. Input & Scan
3. Parameter Studio
4. Assumptions
5. Run Monitor
6. History + Tools (History-Fokus)
7. Astrometry
8. PCC
9. Live Log

Damit bleiben alle bisherigen Bereiche erhalten, aber in einem konsistenten Design- und Interaktionssystem.

## 1.1 Desktop-Baseline

- Zielauflosung: mindestens `1920x1080`.
- Optimierung ist explizit auf 1920+ ausgelegt, nicht auf 1440 als Primar-Layout.
- Auf kleineren Breiten kann horizontaler Scroll entstehen; das ist in diesem Konzept akzeptiert.
- Konkrete Raster- und Breitenwerte: `layout_1920_spec.md`.

## 1.2 Referenzgrundlage (HTML-only)

- Dieses Konzept referenziert die klickbaren HTML-Dummies unter `doc/gui2/clickdummy/` als primare UI-Quelle.
- PNG-Mockups sind nicht mehr normative Referenz fuer das Detailkonzept.
- Layout- und Interaktionsregeln werden ueber HTML/CSS-Struktur + `layout_1920_spec.md` beschrieben.

## 2) Screenspezifikation

### Dashboard

- Oberer KPI-Block: Frames, Scan-Qualitaet, Warnungen, letzter Lauf.
  - `Scan-Qualitaet`: normierter Score `0..1` aus Scanindikatoren (z. B. Sternfindung, Hintergrundstabilitaet, Pattern-Konsistenz).
  - `Warnungen`: Anzahl offener, nicht-blockierender Warnungen aus Scan/Validierung/Guardrails.
- Guided-Run-Karte: Eingabedirs, Farbmodus, Preset, Startaktionen.
- Guided-Run-Karte mit Modusschalter:
  - `Einfach`: nur Kernfelder fuer Schnellstart.
  - `Erweitert`: zusaetzlich `runs_dir`, `run_name`, Output-Preview und MONO-Queue.
- Voller Schritt-fuer-Schritt Wizard bleibt auf eigener Seite (`wizard.html`).
- MONO-Queue-Block: Filterliste mit Reihenfolge (`L, R, G, B, Ha, OIII, SII`), je Eintrag Input-Ordner und Status.
- Readiness Guardrails: explizite Startbedingungen als Checkliste.
- Pipeline-Vorschau: erwartete Phasenfolge vor dem Start.

### Parameter Studio

- Suchleiste ueber alle Parameterpfade (`section.key.subkey`).
  - Live-Suche ohne separaten Such-Button.
  - Ergebnisliste erscheint direkt unter dem Suchfeld (editierbare Treffer mit Kategoriehinweis).
  - Enter springt zum ersten Formular-Treffer und markiert das Feld kurz.
- Preset-Auswahl im Kopfbereich (`Preset` Select) plus Aktion `Preset anwenden`.
  - Preset-Katalog wird aus allen Dateien `tile_compile_cpp/examples/*.example.yaml` gebildet.
- Linke Kategorie-Spalte mit klarer Gruppierung (Pipeline, Registration, BGE, PCC, ...).
- Mittlerer Formbereich als Abschnittseditor:
  - zeigt pro gewaehlter Kategorie alle zugehoerigen Parameter als editierbare Felder.
  - basiert auf dem vereinheitlichten Editor-Index aus Schema + YAML (`param_editor_index.json`).
- Kategorienavigation arbeitet als Filter:
  - Klick auf Kategorie zeigt den vollstaendigen Editierbereich dieser Kategorie.
  - `Alle` zeigt nur Uebersicht/Hinweise, nicht alle Felder gleichzeitig.
- Rechtes Explain-Panel:
  - Kurz-Erklaerung pro Parameter
  - erlaubte Werte / Range
  - Default
  - Risiko-/Einflussinfo
  - YAML-Diff
- `Situation Assistant` im rechten Panel:
  - Alt/Az
  - starke Rotation
  - helle Sterne
  - wenige Frames
  - starker Gradient
  - Delta-Panel aktualisiert sich dynamisch aus den aktiv ausgewaehlten Situationen.
- Aktionen: Preset anwenden, Situation anwenden, validieren, speichern.

### Run Monitor

- Batch-Kontextleiste (`n/m`, aktueller Input, run_id, aktueller Filter und Filterindex).
- Phasenliste in Runner-Reihenfolge mit Statuschips und Prozentanzeige je Phase:
  - `SCAN_INPUT -> CHANNEL_SPLIT -> NORMALIZATION -> GLOBAL_METRICS -> TILE_GRID -> REGISTRATION -> PREWARP -> COMMON_OVERLAP -> LOCAL_METRICS -> TILE_RECONSTRUCTION -> STATE_CLUSTERING -> SYNTHETIC_FRAMES -> STACKING -> DEBAYER -> ASTROMETRY -> BGE -> PCC`.
- Live-Log mit Warnungs-Highlighting.
- `Stats` direkt unter dem Live-Log:
  - `Generate Stats`
  - `Open Stats Folder`
  - Script-Hinweis: Kommando wird aus Konfiguration aufgeloest (siehe §4.5)
- Artefaktliste und Aktionen (Resume, Report, Run-Ordner).

### History + Tools

- Historientabelle mit Selektion und Vergleich.
- Direkte Uebernahme eines Runs als "Current".
- Hinweisbereich mit Deep-Links zu `Astrometry`, `PCC` und `Run Monitor` (Stats).

### Astrometry

- Eigenstaendiger Tool-Screen fuer ASTAP-Workflow.
- Setup-Bereich:
  - ASTAP CLI Pfad
  - ASTAP Data Dir
  - beide Felder mit Dateisystem-Auswahl (`Browse File` / `Browse Dir`)
  - Install/Reinstall ASTAP CLI
  - Installationsstatus
- Star-Database-Bereich:
  - D05/D20/D50/D80 Auswahl
  - Download und Cancel
  - Installations-/Partial-Status
- Plate-Solve-Bereich:
  - FITS-Datei auswaehlen
  - Solve starten
  - Save Solved (WCS in FITS schreiben)
  - Ergebnisfelder (RA/Dec/FOV/Rotation/Scale)
- Log/Progress-Bereich fuer Download und Solve.

### PCC

- Eigenstaendiger Tool-Screen fuer photometrische Farbkalibration.
- Input-Bereich:
  - RGB-FITS
  - WCS-Datei
  - Browse-Aktionen
- Quellen-Bereich:
  - `siril`, `vizier_gaia`, `vizier_apass`
  - Siril-Katalogstatus (`48 Chunks`) und Katalogpfad
  - Katalogpfad mit Verzeichnis-Auswahl (`Browse Catalog Dir`)
  - Download Missing Chunks / Cancel
  - Online-Source-Check fuer VizieR
- Parameter-Bereich:
  - `mag_limit`, `mag_bright_limit`, `min_stars`, `sigma_clip`, Apertur/Annulus
- Aktionen:
  - `Run PCC`
  - `Save Corrected`
- Ergebnis-/Log-Bereich mit Sternzahlen, Residual und Matrix.

## 3) Parametereingabe: vollstaendig und einfach

Grundprinzip:

- Jeder Parameter existiert als Formularfeld.
- Jeder Parameter bleibt parallel im YAML sichtbar.
- Beide Ansichten sind bidirektional synchronisiert.
- Jeder Parameter hat mindestens eine Kurz-Erklaerung (ein Satz).
- MONO-Queue-Eingabe ist als strukturierte Liste editierbar.
- Jede Speicherung erzeugt eine neue, unveraenderliche Config-Revision (append-only).

Bedienebenen:

- Guided Mode:
  - zeigt nur relevante Kernparameter je Workflow-Schritt.
  - empfohlene Defaults und kurze Erlaeuterung.
- Expert Mode:
  - zeigt den kompletten Parameterbaum.
  - volle Feldtiefe inkl. BGE/PCC/Runtime-Limits.

Validierungslogik:

- Sofortvalidierung pro Feld (Typ, Min/Max, Enum).
- Kontextvalidierung fuer Gruppen (z. B. Abhaengigkeiten in Calibration/BGE/PCC).
- Startblockade nur mit konkreter, klickbarer Fehlerliste.
- Tooltips sind fuer alle interaktiven Elemente verpflichtend (Buttons, Felder, Tabellenaktionen, Statuschips).

## 4) Parameter-Knowledge-Layer

### 4.1 Metadatenmodell pro Parameter

Jeder Parameter nutzt folgende Metadaten (de/en):

- `title`
- `short_help`
- `impact`
- `range_hint`
- `default_hint`
- `phase_link`
- `risk_hint`

Quelle der Metadaten:

1. `tile_compile.schema.yaml` (`description`, min/max, enum)
2. manuelle Kurzhilfen fuer fehlende oder zu technische Eintraege
3. Override-Tabelle fuer kritische Parameter (Registration/BGE/PCC)

### 4.2 UI-Verhalten

- Hover/Fokus auf Feld zeigt `short_help` sofort.
- Klick auf Info-Icon oeffnet erweiterten Text (`impact`, `risk_hint`).
- Bei Warnungen zeigt das Feld direkt passende Handlungsvorschlaege.

## 4.3 Deprecated Parameter: data.linear_required

- `data.linear_required` ist als **Deprecated** markiert.
- GUI-Verhalten:
  - Das Feld erscheint im Parameter Studio mit einem `deprecated`-Badge (orangefarbener Chip).
  - Tooltip: "Dieses Feld wird in einer kuenftigen Version entfernt. Wert bitte unveraendert lassen."
  - Guardrail: `warn` wenn Wert explizit auf `false` gesetzt wird (abweichend vom Default).
  - Das Feld ist NICHT aus dem Formular entfernt (bestehende Configs bleiben valid).
  - Kategorie im Parameter Studio: `Data (Deprecated)` als eigene Untergruppe am Ende der `Data`-Sektion.

## 4.3a MONO Multi-Filter Queue (seriell)

### Datenmodell

- `run.filter_queue[]` mit Elementen:
  - `filter_name` (z. B. `L`, `R`, `G`, `B`, `Ha`)
  - `input_dir`
  - `run_label` (optional)
  - `pattern` (optional, fallback auf globales Pattern)
  - `enabled` (bool)
  - `status` (`pending|running|ok|error|skipped`)

### Ausfuehrungslogik

- Queue wird strikt seriell abgearbeitet.
- Naechster Eintrag startet erst nach `run_end` des vorherigen Eintrags.
- Globaler Fortschritt:
  - `Filter i/N`
  - Gesamtdauer und ETA

### Resume-Verhalten

- Resume ist moeglich:
  1. pro Filtereintrag (letzter fehlerhafter/abgebrochener Filter)
  2. pro Phase innerhalb des Filtereintrags
- Resume-Button ist initial deaktiviert und wird erst nach explizitem Phase-Klick aktiv.
- UI zeigt immer den Resume-Target-Kontext: `Filter=Ha, Phase=BGE`.
- Vor jedem Resume bleibt die vorherige Config-Revision erhalten (kein Ueberschreiben).
- Resume kann explizit mit gewaehlter Config-Revision starten.
- Rueckkehr auf aeltere Revision ist jederzeit moeglich (`Restore Revision`), ohne Revisionsverlust.

## 4.4 Config-Revisions-Speicherkonzept

### Speicherort

- Config-Revisionen werden pro Run-Kontext gespeichert in:
  - `<runs_dir>/<run_name>_<YYYYMMDD_HHMMSS>/config_revisions/`
  - Dateiname: `config_rev_<ISO8601_timestamp>.yaml`
  - Beispiel: `config_rev_20260307T221530Z.yaml`
- Append-only: bestehende Dateien werden nie ueberschrieben.
- Zeigerreferenz: `<run_dir>/config_active_revision.txt` enthaelt den Dateinamen der aktuell aktiven Revision.

### Format einer Revision

```yaml
revision_id: 20260307T221530Z
created_at: "2026-03-07T22:15:30Z"
created_by: gui2
comment: "Parameter Studio Save"
config: {}
```

- `config` enthaelt den vollstaendigen Config-Snapshot (kein Delta).
- `revision_id` ist identisch mit dem ISO8601-Suffix.

### GUI-Verhalten

- `monitor.resume.config_revision` laedt alle `.yaml`-Dateien aus `config_revisions/` und listet sie sortiert nach Datum ab.
- `monitor.resume.restore_revision` setzt `config_active_revision.txt` auf eine aeltere Revision und laed deren `config`-Block in den GUI-State.
- Beim naechsten `parameter.save` wird eine **neue** Revision angelegt, die alte bleibt unveraendert.
- Beim GUI-Start: falls `config_active_revision.txt` fehlt, wird die juengste Revision geladen.

## 4.5 generate_report.py: Pfad-Parametrisierung

- Der Pfad zum Stats-Script wird **nicht** hardcodiert.
- Suchstrategie (in dieser Reihenfolge):
  1. GUI-Konfigurationsdatei `~/.config/tile_compile/gui2.json` Schluessel `stats_script_path`
  2. Umgebungsvariable `TILE_COMPILE_STATS_SCRIPT`
  3. Relativ zur Launcher-Binary: `../scripts/generate_report.py`
  4. Fallback-Anzeige: GUI zeigt Fehlermeldung mit Hinweis auf manuelle Konfiguration.
- Das aufgeloeste Kommando wird vor Ausfuehrung im Stats-Panel angezeigt (Readonly).
- `monitor.stats.generate` mappt im Backend auf `runner.stats.generate_report`, der Adapter loest den Pfad zur Laufzeit auf.

## 4.6 Run-Zielverzeichnis und Run-Naming

- `run.runs_dir` ist im Dashboard frei waehlbar (Folder-Picker + manuelle Eingabe).
- `run.run_name` ist frei definierbar (benutzerdefinierter Basisname).
- Effektiver Run-Ordnername wird verbindlich gebildet als:
  - `<run_name>_<YYYYMMDD_HHMMSS>`
- Das Suffix nutzt immer den tatsaechlichen Startzeitpunkt des Runs.
- Beispiel:
  - `M31_altaz_test_20260307_221530`
- Der finale Pfad wird vor Start als Preview angezeigt:
  - `<run.runs_dir>/<run_name>_<YYYYMMDD_HHMMSS>`

## 5) Situation Assistant (Objekt/Szenario-Empfehlungen)

### 5.1 Regelwerk

Regelbasiertes Empfehlungssystem auf Parameter-Deltas:

- Eingang: `capture_mode`, `rotation_level`, `star_brightness`, `frame_count`, `background_gradient_level`
- Ausgang: konkrete Aenderungen (`path`, `recommended_value`, `reason`)

### 5.2 Pflicht-Szenarien

1. Alt/Az
2. Starke Rotation
3. Helle Sterne im Feld
4. Wenige Frames
5. Starker Hintergrundgradient

Detaillierte Werte: `szenario_empfehlungen.md`.

### 5.3 Anwendung im UI

- Benutzer waehlt ein oder mehrere Szenarien.
- GUI zeigt:
  - `Empfohlen`-Badge an betroffenen Feldern
  - Delta-Ansicht vor Anwendung
  - Undo/Redo fuer Preset-Anwendung

## 6) i18n-Konzept

Siehe auch: `i18n_konzept.md`.

Pflichtumfang in GUI 2:

- Laufzeit-Sprachwechsel (`de`, `en`) ohne Neustart.
- Uebersetzung fuer:
  - Navigation, Buttons, Meldungen
  - Parameternamen und Kurz-Erklaerungen
  - Guardrail-Texte
  - Szenario-Empfehlungsgruende
- Einheitliches Key-Schema:
  - `ui.nav.dashboard`
  - `param.pcc.k_max.short_help`
  - `scenario.altaz.reason.rotation`

## 7) Komponentenstandard

- `ParameterField` (Label, Feld, Kurz-Hilfe, Fehlerzustand)
- `SectionCard` (Titel, Collapse, Status)
- `StatusChip` (`ok`, `warn`, `error`, `running`, `pending`, `recommended`)
- `ActionBar` (Validieren, Speichern, Run starten)
- `DiffPanel` (vorher/nachher)
- `ScenarioPanel` (Situation waehlen, Deltas uebernehmen)
- `LocaleSwitch` (DE/EN)
- `TooltipLayer` (einheitliches Verhalten fuer alle Controls)

## 8) Design Tokens 

- Grundfarben:
  - Hintergrund: `#f3f7fc -> #e5eef5`
  - Surface: `#ffffff`
  - Primary: `#15808d`
  - Accent: `#c96f2d`
- Radius:
  - Cards: 16-20 px
  - Inputs/Buttons: 10-12 px
- Typografie:
  - Titel: Inria Serif (oder aehnliche Serif)
  - UI-Text: Inria Sans (oder aehnliche Sans)

## 9) Abdeckung der bisherigen Funktionalitaet

- Scan- und Kalibrier-Logik bleibt erhalten, wird besser strukturiert.
- Config/YAML-Workflow bleibt erhalten, wird um vollstaendige Feldpflege erweitert.
- Run-/Batch-Steuerung bleibt erhalten, bekommt besseren Zustandskontext.
- Astrometry und PCC sind eigene Screens; History+Tools enthaelt die Historie und Deep-Links.
- Stats ist im Run Monitor unter dem Live-Log integriert.
- Live-Log bleibt erhalten, wird im Run Monitor priorisiert.

## 10) Technischer Uebergangspfad (HTML + FastAPI, ohne Qt-GUI)

1. HTML-Komponenten-Set (Cards, Fields, Chips, Tooltips) im Web-Frontend etablieren.
2. FastAPI-Adapter fuer Runner/CLI-Endpunkte und Event-Streaming aufsetzen.
3. Parameter Studio als zentrale Web-Seite mit Such-/Explain-/Diff-Layer umsetzen.
4. Parameter-Knowledge-Layer und i18n-Keys anbinden.
5. MONO Multi-Filter Queue (seriell) im Dashboard und Run Monitor integrieren.
6. Situation Assistant mit regelbasierten Parameter-Deltas einbauen.
7. Bestehende Funktionsbereiche aus der alten GUI in HTML-Seiten ueberfuehren (funktional identisch, visuell modernisiert).
8. Verbindliche Screen-Reihenfolge fuer die Implementierung:
   - Dashboard -> Input&Scan -> Parameter Studio -> Assumptions -> Run Monitor -> History+Tools -> Live Log.

## 11) Detaillierte Ablaeufe (synchron zur Implementierung)

## 11.1 Dashboard-KPI und Startfreigabe

1. Dashboard zeigt `Scan-Qualitaet` und `Warnungen` als KPI.
2. Klick auf KPI oeffnet Detailursachen und Deep-Links.
3. Readiness-Guardrails zeigen `ok/warn/error`.
4. `Run starten` ist nur erlaubt, wenn kein `error` offen ist.

## 11.2 MONO Eingabe mehrerer Input-Dirs

1. Bei `color_mode=MONO` wird `run.filter_queue[]` aktiv.
2. Pro Filter wird ein eigener `input_dir` erfasst.
3. Optional je Filter:
   - `pattern`
   - `run_label`
   - `enabled`
4. Reihenfolge wird strikt seriell abgearbeitet (`Filter i/N`).

## 11.3 Parameter aendern und Resume per Phase-Klick

1. In `Parameter Studio` werden Werte geaendert.
2. `Validieren` prueft Schema + Semantik; bei `error` bleibt Resume blockiert.
3. `Speichern` persistiert die geaenderte Konfiguration als neue Config-Revision fuer den aktiven Run-Kontext (Filter + run_id).
4. Wechsel in `Run Monitor`.
5. Optional gewuenschte Config-Revision aus der Revisionsliste waehlen.
6. Gewuenschte Phase anklicken (`Resume ab Phase X`) und optional Filterkontext waehlen.
7. Resume startet mit gewaehlter Konfiguration ab gewaehlter Phase.
8. UI zeigt den Zielkontext explizit an: `Filter=<name>, Phase=<phase>, Config-Version=<timestamp/hash>`.
9. Bei Bedarf kann per `Restore Revision` auf eine fruehere Config-Version zurueckgekehrt werden.

## 11.4 Trennung PCC und Stats

1. `PCC` liegt auf eigener Seite `PCC` (Run PCC, Save Corrected, Katalogdownload).
2. `Astrometry` liegt auf eigener Seite `Astrometry` (ASTAP Setup, Katalogdownload, Solve).
3. `Stats` liegt im `Run Monitor` direkt unter `Live Log`.
4. Stats-Aufruf nutzt aktiven Run-Kontext und erzeugt Report-Artefakte im Run-Ordner.

## 12) Implementierungsstrategie (synchron zur Funktionsmatrix)

## 12.1 Phase A - Fundament

1. State-Model, Event-Stream, i18n-Layer, Guardrail-Service.
2. Basiskomponenten (`SectionCard`, `ParameterField`, `StatusChip`, `TooltipLayer`).

## 12.2 Phase B - Dashboard + Queue

1. KPI-Aggregator (`Scan-Qualitaet`, `Warnungen`).
2. MONO Queue-Editor (`run.filter_queue[]`).
3. Start-Gating mit hartem Guardrail-Block bei `error`.

## 12.3 Phase C - Parameter Studio

1. Vollstaendige Feldabdeckung + Suche + Kategorie-Navigation.
2. Explain-Layer + Situation Assistant + YAML-Diff.
3. Schema-/Semantikvalidierung vor Persistenz.

## 12.4 Phase D - Run Monitor + Stats

1. Phasenmonitor, Resume, Filterkontext.
2. Stats-Panel unter Live Log (`Generate Stats`, `Open Stats Folder`).
3. Integration von `generate_report.py` inkl. Fehlerpfad ohne Python.

## 12.5 Phase E - History + Astrometry/PCC

1. Historientabelle, Current-Run-Selektion, Report-Zugriff.
2. Eigenstaendige Astrometry-Seite (ASTAP Setup/Download/Solve).
3. Eigenstaendige PCC-Seite (Siril/VizieR, Chunk-Download, Run/Save).
4. Sauberer Wechsel zwischen History, Tool-Screens und Run-Monitor-Stats.

## 12.6 Phase F - Abnahme

1. Control-Coverage gegen Implementierungsmatrix.
2. E2E-Flows fuer OSC, MONO-Queue, Resume, Stats, PCC.
3. Dokumentations-Sync zwischen `detailkonzept.md` und `implementierungsablauf_funktionen.md`.
