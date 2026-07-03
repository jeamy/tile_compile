# GUI3 Benutzerhandbuch (Deutsch)

Dieses Handbuch beschreibt den praktischen Workflow mit der GUI3-Weboberfläche: von der Installation über den Scan der Eingabedaten, die Parametereinstellung bis zum Start und Monitoring eines Runs.

## Übersicht

GUI3 besteht aus drei Hauptbereichen:

| Tab | Sub-Tabs | Zweck |
|-----|----------|------|
| **Processing** | Input & Scan, Parameter, Run Monitor | Hauptworkflow: Eingabe scannen, Parameter einstellen, Run starten und überwachen |
| **Tools** | Raw Stack, Astrometry, PCC | Eigenständige Werkzeuge für Vorverarbeitung, Plate Solving und Farbkalibrierung |
| **History** | Run History | Abgeschlossene Runs durchsuchen, vergleichen und Reports generieren |

---

## 1. Installation und Start

### Release-Bundle

1. Release-ZIP von [GitHub Releases](https://github.com/jeamy/tile_compile/releases) herunterladen.
2. Entpacken und den Starter ausführen:
   - **Linux**: `./start_gui3.sh`
   - **macOS**: Doppelklick auf `start_gui3.command` (oder `./start_gui3.command` im Terminal)
   - **Windows**: Doppelklick auf `start_gui3.bat`
3. Beim ersten Start werden alle Anwendungsdateien nach `~/tilecompile/` (bzw. `%USERPROFILE%\tilecompile\`) kopiert.
4. Der Browser öffnet sich automatisch auf `http://127.0.0.1:8080/ui/`.

> **macOS-Hinweis:** Falls Gatekeeper den Starter blockiert: `Systemeinstellungen → Datenschutz & Sicherheit` öffnen und den blockierten Eintrag explizit erlauben.

> **Nach der Installation:** Das heruntergeladene Archiv und der entpackte Ordner können gelöscht werden. Alle Dateien wurden in das Benutzerverzeichnis kopiert. Bei Updates werden nur Anwendungsdateien ersetzt — Benutzerdaten (Runs, Kataloge) bleiben erhalten.

---

## 2. Eingabe scannen (Input & Scan)

Der erste Schritt ist das Scannen der FITS-Eingabedaten.

### Schritte

1. **Tab Processing → Sub-Tab Input & Scan** auswählen.
2. **Eingabeordner** wählen: Pfad zum Verzeichnis mit den FITS-Light-Frames angeben (z.B. `/data/M31/lights`).
3. **Dateimuster** einstellen: Standard ist `*.fits`. Alternativ `*.fit` oder `light_*.fits`.
4. **Ausgabeordner (Runs-Dir)** wählen: Zielverzeichnis für Run-Outputs. Standardmäßig wird ein Verzeichnis unter `~/tilecompile/runs/` vorgeschlagen.
5. **Run Name** vergeben: Optionaler Name für den Run-Ordner (z.B. `M31_altaz_test`).
6. **Parameter einstellen**:
   - **Frames Minimum**: Mindestanzahl Frames, sonst Abbruch (Standard: 30).
   - **Max. Frames**: 0 = alle Frames verwenden.
   - **Sortierung**: `numeric`, `alphabetic` oder `timestamp`.
   - **Farbmodus**: `OSC` (One-Shot-Color / Bayer) oder `MONO`.
   - **Checksummen berechnen**: Optional, für Integritätsprüfung.
7. **Kalibrierung** (optional): Bias-, Dark- und Flat-Ordner angeben und die jeweiligen Checkboxen aktivieren.
8. **Scan starten** klicken.

### Scan-Ergebnis

Nach Abschluss zeigt die Scan-Result-Karte:
- Anzahl gefundener Frames
- Auflösung und Farbmodus der Frames
- Frames-Größen und Checksummen (falls aktiviert)

### Run-Queue (für MONO)

MONO-Nutzer können mehrere Eingabeordner (L/R/G/B) über die Run-Queue hinzufügen. Jede Queue-Eingabe wird als separater Kanal verarbeitet.

---

## 3. Parameter einstellen (Parameter Studio)

Nach dem Scan wechseln Sie zum Sub-Tab **Parameter**.

### Konfiguration laden

- **Beispielkonfiguration laden**: Über den Konfigurations-Selector eine der mitgelieferten Beispiel-YAMLs wählen (z.B. `M42.global_medium.yaml`).
- **Leere Konfiguration**: Mit Default-Werten starten.
- **Bestehende Konfiguration hochladen**: Eine eigene YAML-Datei hochladen.

### Parameter-Kategorien

Das Parameter Studio ist in Kategorien gegliedert:

| Kategorie | Wichtige Parameter |
|-----------|-------------------|
| **Pipeline** | `pipeline.mode` (production/preview), `method` (aqmh/classic_tile_compile) |
| **Registrierung** | `registration.allow_rotation`, `registration.transform_model` (similarity/affine), `registration.engine` |
| **AQMH** | `aqmh.enabled`, `aqmh.pyramid.scales`, `aqmh.cherry_pick.enabled`, `aqmh.storage.max_resident_maps` |
| **Rekonstruktion** | Tile-Geometrie, Rekonstruktionsmodus |
| **Stacking** | `stacking.method` (sigma_clip/median/average), Sigma-Clip-Parameter |
| **Debayer** | `data.bayer_pattern` (RGGB/BGGR/GBRG/GRBG) |
| **Astrometry** | `astrometry.astap_bin`, `astrometry.astap_data_dir` |
| **BGE** | `bge.method`, `bge.enabled` (Legacy), `bge.fit.method`, `bge.autobge`, `bge.autotune` |
| **PCC** | `pcc.source` (auto/siril/vizier_gaia/vizier_apass), `pcc.mag_limit`, `pcc.siril_catalog_dir` |
| **HyperMetric Stretch** | `hypermetric_stretch.enabled`, Farbstrategie, Konvergenz |
| **Kalibrierung** | Bias/Dark/Flat-Pfade |
| **Assumptions** | `frames_min`, `frames_reduced_threshold` |
| **Runtime Limits** | `hard_abort_hours`, `acceleration_backend` |

### Workflow

1. Parameter-Kategorie aufklappen und Werte anpassen.
2. **Validate** klicken, um die Konfiguration gegen das Schema zu prüfen.
3. Bei Validierungsfehlern: Fehlermeldungen beachten und korrigieren.
4. **Save** klicken, um die Konfiguration zu speichern.

> **Tipp:** Für erste Versuche ist die Default-Konfiguration mit `method: aqmh` bereits gut brauchbar. Die wichtigsten Anpassungen sind `registration.allow_rotation` (bei Alt/Az-Montierung auf `true` belassen) und `data.bayer_pattern` (mit den FITS-Headern abgleichen).

---

## 4. Run starten und überwachen (Run Monitor)

### Run starten

1. Vom Sub-Tab **Parameter** auf **Run Monitor** wechseln, oder direkt vom Input & Scan-Tab auf **Next ▶** klicken.
2. Im Run Monitor auf **Run starten** klicken.
3. Der Runner wird als Hintergrundprozess gestartet. Der Run Monitor zeigt den Fortschritt in Echtzeit.

### Run Monitor

Der Run Monitor zeigt:

- **Aktuelle Phase**: Welche Pipeline-Phase gerade läuft (SCAN_INPUT → REGISTRATION → ... → PCC → DONE)
- **Phasen-Fortschritt**: Status jeder Phase (pending, running, ok, skipped, error)
- **Event-Timeline**: Live-Events aus `run_events.jsonl`
- **Warnungen und Fehler**: Prominent angezeigt im Warnungs-Banner
- **Run-Statistiken**: Verarbeitete Frames, verbleibende Zeit (falls verfügbar)

### Run abbrechen

- **Stop** klicken, um den laufenden Run abzubrechen.

### Resume (Fortsetzen)

Nach Abschluss oder Abbruch kann ein Run ab einer gewählten Phase fortgesetzt
werden. Das bestehende Run-Verzeichnis und seine Zwischenartefakte werden
wiederverwendet.

1. In der Phasenliste die Phase anklicken, ab der die Verarbeitung fortgesetzt
   werden soll, beispielsweise `ASTROMETRY`, `BGE`, `PCC` oder
   `HYPERMETRIC_STRETCH`.
2. Der Abschnitt **Resume** öffnet sich und zeigt die gewählte Phase rechts als
   Badge an.
3. Die YAML unter **Config YAML** prüfen. Mit **Aktuelle Config laden** kann die
   Run-Konfiguration wiederhergestellt werden. Alternativ kann eine gespeicherte
   **Config-Revision** gewählt und geladen werden.
4. Im Resume-Abschnitt auf **Resume** klicken.
5. GUI3 speichert die gewählte Konfiguration als Revision, startet den Runner
   ab der gewählten Phase und kehrt zur Live-Überwachung zurück.

Phasen vor dem gewählten Resume-Punkt werden nicht neu berechnet. Ihre
vorhandenen Artefakte müssen deshalb weiterhin vorhanden und mit der gewählten
Konfiguration kompatibel sein.

> Häufige Resume-Punkte: `ASTROMETRY` führt das Plate Solving erneut aus, `BGE`
> berechnet die Hintergrundextraktion neu, `PCC` wiederholt die photometrische
> Farbkalibrierung und `HYPERMETRIC_STRETCH` erzeugt aus dem vorhandenen
> linearen PCC-Ergebnis eine neue gestreckte Ausgabe.

### HyperMetric Stretch beim Resume konfigurieren

Wenn `HYPERMETRIC_STRETCH` als Resume-Phase gewählt ist, stellt GUI3 eine
interaktive HMS-Vorschau bereit.

1. In der Phasenliste `HYPERMETRIC_STRETCH` anklicken.
2. Im Resume-Abschnitt auf **HMS konfigurieren** klicken. Der Button befindet
   sich unmittelbar links neben dem Badge **HyperMetric Stretch**.
3. Auf die erste Vorschau warten. GUI3 verwendet normalerweise
   `outputs/stacked_rgb_pcc.fits`. Ist diese Datei nicht vorhanden, wird der
   vollständige Kanalsatz `pcc_R.fit`, `pcc_G.fit` und `pcc_B.fit` verwendet.
4. Die HMS-Parameter anpassen. Nach jeder gültigen Änderung wird automatisch
   eine neue Proxy-Vorschau berechnet.
5. Vorschau, Histogramm, verwendete Quelle, berechnetes log D, Anker sowie
   Schwarz- und Weiß-Clipping prüfen.
6. **Übernehmen & Resume starten** klicken. Dadurch werden die angezeigten
   HMS-Werte in die Resume-YAML geschrieben und der Run unmittelbar ab
   `HYPERMETRIC_STRETCH` fortgesetzt.

**Zurücksetzen** stellt die Werte wieder her, die beim Öffnen des Dialogs
geladen wurden. **Abbrechen** oder der Schließen-Button verwirft die Änderungen,
ohne einen Resume zu starten.

#### Navigation in der Vorschau

- Bild ziehen, um den Ausschnitt zu verschieben.
- Mit dem Mausrad zoomen.
- Das Bild doppelt anklicken, um es in den Vorschaubereich einzupassen.
- Unter dem Bild wird das RGB-Histogramm logarithmisch dargestellt.

Die Vorschau verwendet einen verkleinerten Proxy mit maximal 1600 Pixeln an der
langen Kante. Der abschließende Resume führt HMS auf den PCC-Daten in voller
Auflösung aus.

#### HMS-Parameter

Wenn der Mauszeiger über einem Parameternamen oder seinem Info-Symbol steht,
zeigt ein Tooltip die Wirkung und den zulässigen Bereich an.

| Parameter | Wirkung |
|-----------|---------|
| **Modus** | `Anzeigefertig` führt eine darstellungsorientierte Ausgabeskalierung und weiches Highlight-Clipping aus. `Wissenschaftlich` lässt diese abschließenden Darstellungsschritte aus. |
| **Sensorprofil** | Wählt die RGB-Luminanzgewichte für Ankerberechnung, automatische log-D-Bestimmung, Star-Pressure-Schätzung und Ausgabeskalierung. `Automatisch` verwendet das Fallback-Profil. |
| **Fallback-Profil** | Wird verwendet, wenn das primäre Profil auf `Automatisch` steht. Ein unbekanntes explizites Profil fällt im Core auf Rec.709 zurück. |
| **Adaptiver Anker** | Bestimmt den Stretch-Anker aus dem Bildinhalt statt über den statistischen Ankerpfad. |
| **Zielhintergrund** | Gewünschte Hintergrundhelligkeit nach dem Stretch. Höhere Werte erzeugen einen helleren Hintergrund. |
| **Schutz B** | Formt die hyperbolische Kurve. Höhere Werte komprimieren und schützen helle Bereiche stärker. |
| **Konvergenz-Power** | Steuert, wie schnell die RGB-Kanäle in helleren Bereichen konvergieren. |
| **log-D-Modus** | `Automatisch` berechnet die Stretch-Stärke aus dem Proxy. `Fest` aktiviert den festen log-D-Wert. |
| **Fester log D** | Manuelle logarithmische Stretch-Stärke. Höhere Werte machen schwache Strukturen stärker sichtbar. |
| **Farbstrategie** | Wählt automatische oder feste Farbbehandlung. |
| **Feste Farbstrategie** | Im Modus `Anzeigefertig` erhöhen negative Werte die Schattenkonvergenz; positive Werte verringern den Farbgriff. |
| **Farbgriff** | Steuert im wissenschaftlichen Modus, wie stark die ursprünglichen Farbverhältnisse erhalten bleiben. |
| **Schattenkonvergenz** | Blendet dunkle Bereiche im wissenschaftlichen Modus stärker zum kanalweisen Stretch über. |
| **Lineare Expansion** | Fügt im wissenschaftlichen Modus eine lineare Expansion am unteren Ende hinzu. |

Numerische Felder akzeptieren keine Werte außerhalb ihrer angezeigten Bereiche.
Ein ungültiger Wert startet keine Vorschau und kann nicht übernommen werden.
Verlässt der Fokus ein solches Feld, begrenzt GUI3 den Wert auf die
nächstgelegene zulässige Grenze.

#### HMS-Werte übernehmen

**Übernehmen & Resume starten** ändert ausschließlich die bekannten Felder
unter `hypermetric_stretch` in der aktuell geladenen Resume-YAML. Andere
Konfigurationsabschnitte bleiben unverändert. Anschließend verwendet GUI3 den
normalen Resume-Mechanismus. Dieser:

- sichert die vorherige Run-Konfiguration,
- erstellt eine Config-Revision,
- startet den Runner mit `--from-phase HYPERMETRIC_STRETCH` und
- zeigt den neuen Job im Run Monitor an.

Die Vorschau selbst verändert weder `config.yaml` noch FITS-Artefakte. Ist kein
vollständiges PCC-RGB-Artefakt vorhanden, meldet die Vorschau einen Fehler und
es wird kein Resume gestartet.

### AutoBGE beim Resume konfigurieren

In der Phasenliste `BGE` wählen und links neben dem BGE-Badge auf
**BGE konfigurieren** klicken. Der Dialog verwendet `stacked_rgb_solve.fits`
und ersatzweise `stacked_rgb.fits` zusammen mit `canvas_mask.fits`. Ein bereits
BGE-korrigiertes Bild wird niemals erneut als Eingang verwendet.

Der Dialog bietet die Ansichten **Original**, **Korrigiert** und
**Hintergrund**. Ziehen verschiebt das Bild, das Mausrad zoomt und ein
Doppelklick passt das Bild an den Vorschaubereich an. Farbige Punkte markieren
die von AutoBGE verwendeten Samples.

Alle AutoBGE-Parameter besitzen Tooltips und erzwingen ihre zulässigen
Wertebereiche. Dazu gehören Sample-Anzahl, Polynomgrad, RBF-Glättung,
Downsample-Faktor, Patch-Größe und -Schätzer, Arbeitsraumtransformation,
Helligkeitsausschluss, Zufalls-Seed und Schutzprüfungen.

So werden echte dunkle Strukturen wie Dunkelnebel von der Hintergrundabtastung
ausgeschlossen:

1. **Ausschluss zeichnen** anklicken.
2. Mindestens drei Eckpunkte im Bild setzen.
3. Das Polygon mit einem Doppelklick schließen und die Vorschau neu berechnen.
4. **Ausschlüsse löschen** entfernt alle Polygone.

Ausschlüsse beeinflussen nur die Sample-Auswahl und löschen niemals Pixel der
Ausgabe. Sie werden als normalisierte Koordinaten unter
`bge.autobge.exclusion_polygons` gespeichert, sodass der Resume in voller
Auflösung dieselben Bereiche verwendet.

#### Manuelle Sample-Points setzen

Zusätzlich zu den automatischen Sample-Points können manuelle Punkte gesetzt
werden, um gezielt dunkle Hintergrundbereiche für die Modellierung zu
erzwingen:

1. **Punkte hinzufügen** anklicken. Der Cursor wechselt auf ein Fadenkreuz.
2. Auf die gewünschten Stellen im Bild klicken. Jeder Klick fügt einen Punkt
   hinzu. Punkte werden als weiße Kreise mit rotem Rand dargestellt.
3. Ein erneuter Klick auf **Punkte hinzufügen** oder ein Doppelklick beendet
   den Punktmodus.
4. **Punkte löschen** entfernt alle manuellen Punkte.

Manuelle Punkte werden immer in den Sample-Satz aufgenommen und umgehen die
zufällige Downselection. Sie werden als normalisierte Float-Koordinaten
`[0..1]` unter `bge.autobge.user_sample_points` in der YAML gespeichert und
sind damit auflösungsunabhängig. Beim Wiederöffnen des Dialogs oder bei einem
Resume werden gespeicherte Punkte automatisch geladen.

#### Guard-Ablehnungsgründe

Wenn die Schutzprüfungen (Guards) die Vorschau ablehnen, zeigt die
Statuszeile den konkreten Ablehnungsgrund zusammen mit einem
Lösungshinweis an. Typische Gründe sind:

| Grund | Bedeutung | Empfohlene Maßnahme |
|-------|-----------|---------------------|
| **Ebenheit verschlechtert** | Das Hintergrundmodell ist nach der Korrektur unebener als vorher | `poly_degree` verringern, `rbf_smooth` erhöhen, Ausschlusspolygone um Strukturen ziehen |
| **Hintergrund-Chroma verschlechtert** | Die Farbverteilung im Hintergrund hat sich verschlechtert | `rbf_smooth` erhöhen, `poly_degree` verringern, mehr manuelle Punkte auf dunklem Hintergrund setzen |
| **Steigung verschlechtert** | Der Gradient im Hintergrundmodell hat sich verschlechtert | Manuelle Punkte auf den dunklen Gradienten setzen, `num_sample_points` erhöhen, `rbf_smooth` verringern |

Zusätzlich wird unter dem Bild angezeigt, welche Kanäle (R, G, B) jeweils
abgelehnt wurden. Wenn `apply_guards` deaktiviert ist, kann die Vorschau
trotz Ablehnung übernommen werden — dies wird jedoch nicht empfohlen.

**Übernehmen & Resume starten** setzt `bge.method: autobge`, aktualisiert die
AutoBGE-Werte in der Resume-YAML und setzt den Run ab `BGE` fort.
**Zurücksetzen** stellt Parameter, Polygone und manuelle Punkte vom
Öffnungszeitpunkt wieder her. Preview-Berechnungen verändern weder FITS-Dateien
noch `config.yaml`.

---

## 5. Ergebnisse ansehen

### Ausgabedateien

Nach Abschluss liegen die Ergebnisse unter `<runs_dir>/<run_name>/`:

```
runs/<run_id>/
├── outputs/
│   ├── stacked.fits              # Lineares Summenbild
│   ├── reconstructed_L.fit       # MONO-Rekonstruktion
│   ├── stacked_rgb.fits          # OSC-RGB
│   ├── stacked_rgb_solve.fits    # Mit WCS gelöst
│   ├── stacked_rgb_bge.fits      # Nach BGE (vor PCC)
│   ├── stacked_rgb_pcc.fits      # Nach PCC
│   └── stacked_rgb_hms.fits      # Nach HyperMetric Stretch (optional)
├── artifacts/
│   ├── report.html               # Diagnosebericht
│   ├── report.css
│   ├── *.png                     # Diagramme/Heatmaps
│   ├── normalization.json
│   ├── global_registration.json
│   ├── bge.json
│   └── ...
├── logs/
│   └── run_events.jsonl          # Event-Timeline
└── config.yaml                   # Run-Snapshot der Konfiguration
```

### Diagnosebericht (report.html)

Über den Run Monitor oder die Run History kann ein HTML-Diagnosebericht generiert werden:

- **Run Monitor**: Button **Stats erstellen** klicken.
- **Run History**: Run auswählen und Report generieren.
- **CLI**: `./tile_compile_cli generate-report runs/<run_id>`

Der Bericht enthält:
- Normalisierungs-/Hintergrund-Trends
- Globale Qualitätsverteilungen und Gewichte
- Registrierungs-Drift/CC/Rotation
- Tile- und Rekonstruktions-Heatmaps
- BGE-Diagnostik
- Validierungsmetriken (inkl. Tile-Pattern-Indikatoren)
- Pipeline-Zeitachse und Frame-Usage-Funnel

---

## 6. Astrometry und PCC einrichten

Astrometrie (Plate Solving) und PCC (Photometrische Farbkalibrierung) sind optionale Phasen, die externe Daten benötigen.

### ASTAP (für Astrometry / WCS Plate Solving)

1. **ASTAP installieren**: Binary von [https://www.hnsky.org/astap.htm](https://www.hnsky.org/astap.htm) herunterladen.
2. **Sterndatenbank herunterladen**: Mindestens D50 (für Deep-Sky) oder G18 (für Weitwinkel).
3. **In GUI3 einrichten**:
   - Tab **Tools → Sub-Tab Astrometry**
   - **ASTAP CLI**: Pfad zum `astap`-Binary (z.B. `/usr/local/bin/astap`)
   - **Star Database Dir**: Pfad zum Katalogverzeichnis (z.B. `/usr/local/share/astap`)
   - **Detect ASTAP** klicken, um die Installation zu prüfen.
   - Alternativ: **Install CLI** und **Download Catalog** direkt über die GUI.

> Die ASTAP-Kataloge können auch direkt über die GUI3 heruntergeladen werden (Tools → Astrometry → Download Catalog).

### Siril Gaia DR3 XP Katalog (für PCC)

1. **Katalogdaten**: Der Siril Gaia DR3 XP sampled catalog wird für die photometrische Farbkalibrierung benötigt.
2. **In GUI3 einrichten**:
   - Tab **Tools → Sub-Tab PCC**
   - **Catalog Dir**: Pfad zum Katalogverzeichnis (Standard: `~/.local/share/siril/siril_cat1_healpix8_xpsamp/`)
   - **Download Missing** klicken, um fehlende Katalog-Chunks herunterzuladen (48 Chunks, insgesamt ~2 GB).
   - Der Download erfolgt im Hintergrund mit Fortschrittsanzeige.

> Falls der Katalog bereits von [Siril](https://siril.org/) heruntergeladen wurde, kann derselbe Ordner wiederverwendet werden.

3. **PCC-Parameter** (im Parameter Studio):
   - `pcc.source`: `auto` (empfohlen), `siril`, `vizier_gaia` oder `vizier_apass`
   - `pcc.mag_limit`: Grenzgröße für Sterne (Standard: 14.0)
   - `pcc.siril_catalog_dir`: Pfad zum Siril-Katalog (falls abweichend)

### Wenn Kataloge fehlen

Wenn ASTAP oder der Siril-Katalog nicht installiert sind:
- Die Kernrekonstruktion (Registrierung, AQMH, Stacking, Debayer) funktioniert weiterhin.
- Die Phasen `ASTROMETRY` und `PCC` werden als `skipped` markiert oder schlagen fehl, je nach Konfiguration.
- BGE (Background Gradient Extraction) funktioniert unabhängig von externen Katalogen.

---

## 7. Raw Stack (eigenständige Vorverarbeitung)

Der Sub-Tab **Raw Stack** (unter Tools) bietet ein separates lineares Preprocessing von FITS-Lights bis zum fertigen Stack, unabhängig vom Tile-Compile-Run-Studio.

Pipeline-Schritte:
1. Input Scan
2. Calibration (Bias/Dark/Flat)
3. CFA/Mono Prep
4. Reference Selection
5. Registration
6. Quality Analysis
7. Frame Filtering
8. Stacking (Sigma-Clip/Median/Winsor)
9. Astrometry (optional)
10. BGE (optional)
11. PCC (optional)
12. HyperMetric Stretch (optional)
13. Report

> Detaillierte Dokumentation: [docs/raw_stack_gui_de.md](raw_stack_gui_de.md)

---

## 8. Run History

Unter **Tab History → Sub-Tab Run History** können abgeschlossene Runs durchsucht werden:

- Liste aller Runs mit Status, Datum, Frame-Anzahl
- Run-Details einsehen (Konfiguration, Phasen-Status, Artefakte)
- Reports generieren
- Runs vergleichen
- Run-Verzeichnis im Dateimanager öffnen
