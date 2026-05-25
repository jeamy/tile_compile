# Plan: Raw-Preprocessing als eigener Menuepunkt

## Ziel

Tile-Compile soll einen eigenen Menuepunkt auf derselben Ebene wie `Astrometry`, `PCC` und weitere Werkzeuge erhalten, der den grundlegenden Preprocessing-Pfad von Rohdaten bis zum linearen Stack abdeckt:

1. Raw-/Light-Daten einlesen
2. optionale Kalibrierung mit Bias/Dark/Flat
3. CFA-/OSC- oder Mono-Behandlung mit Tile-Compile-Logik
4. Registrierung
5. Frame-Qualitaetsbewertung und optionales Filtern
6. lineares Stacking
7. Uebergabe des erzeugten Stacks an den bestehenden Hauptstrang, z. B. Astrometry, BGE, PCC und HyperMetric Stretch

Der erwartete Anwendernutzen ist ein direkter linearer Pre-Stack-Workflow:

```text
lights -> scan -> calibrate -> CFA/channel handling -> register -> quality filter -> stack
```

Die technische Umsetzung soll vorhandene Tile-Compile-Phasen wiederverwenden und nur die tile-spezifischen Spezialphasen ueberspringen.

Unterstuetzte Eingangsdaten sollen den bestehenden Tile-Compile-Moeglichkeiten entsprechen: Raw-Formate, FITS, CFA/OSC und Mono. Der neue Menuepunkt darf hier keinen engeren Formatumfang definieren als der Hauptstrang.

## Neuer Menuepunkt

Vorgeschlagener Name: `Preprocessing` oder `Raw Stack`.

Position: Hauptnavigation auf derselben Ebene wie `Astrometry`, `PCC`, `BGE` und andere eigenstaendige Tools. Der Punkt soll kein Unterdialog der bestehenden Run-Konfiguration sein, sondern ein eigener Workflow, weil er vor dem heutigen Tile-Compile-Hauptstrang liegt.

Wichtig: Preprocessing ist ein getrennter Prozess. Es darf nicht im normalen Tile-Compile Run Studio, im normalen Run-Monitor-Phasensatz oder im normalen Parameter Studio auftauchen. Wiederverwendet werden nur Funktionen, Algorithmen, Report-/Eventmuster und UI-Komponenten.

Primaere Ausgabe nach Tile-Compile-Artefaktstruktur:

- `artifacts/preprocess/stacked_linear.fits`
- bei OSC zusaetzlich `artifacts/preprocess/stacked_rgb.fits`
- `artifacts/preprocess/frame_quality.csv`
- `artifacts/preprocess/rejected_frames.txt`
- `artifacts/preprocess/preprocessing_report.json`
- vollstaendiger Report, erzeugt ueber denselben Report-Generator wie normale Tile-Compile-Runs

Diese Artefakte koennen danach als Input fuer den bestehenden Pfad verwendet werden.

Fortschritt und Parameterbearbeitung sollen sich fuer Anwender genauso anfuehlen wie bestehende Tile-Compile-Laeufe: Fortschrittsanzeige nach dem Muster des Run Monitors, Parameteransicht und Parameterkorrekturen nach dem Muster des Parameter Studios.

## MVP-Funktionsumfang

Der erste umsetzbare Stand sollte bewusst nah an der vorhandenen Tile-Compile-Pipeline bleiben:

1. Eingabeordner `lights` waehlen.
2. Raw/FITS-Dateien erkennen und in interne lineare Frames laden.
3. Kalibrierung und Metadatenpruefung wie im Hauptstrang vorbereiten.
4. Referenzframe aus den Lights bestimmen.
5. Alle Frames mit der vorhandenen Tile-Compile-Registrierung ausrichten.
6. Frame-Metriken und Qualitaetsfilter aus dem Hauptstrang berechnen.
7. Registrierte Frames klassisch stacken.
8. CFA/OSC und Mono im Tile-Compile-Stil behandeln: CFA-/Channel-Pfad verwenden, Mono als Ein-Kanal-Pfad ohne kuenstliche RGB-Annahmen.
9. Stack als lineares FITS schreiben.
10. Den Output direkt an `Astrometry`, `BGE`, `PCC` und Stretch weiterreichen koennen.

Kalibrierframes sind im MVP optional. Ohne Bias/Dark/Flat entsteht ein klassischer registrierter Stack, intern aber weiterhin mit Tile-Compile-Mechanik.

## Erweiterter Kalibrierumfang

Der erweiterte Pfad sollte folgende Eingaben akzeptieren:

- `lights`
- `bias`
- `darks`
- `flats`
- optional `darkflats`

Empfohlene Reihenfolge:

1. Master-Bias erstellen oder laden.
2. Master-Dark erstellen oder laden.
3. Master-Flat erstellen oder laden.
4. Lights kalibrieren.
5. kosmetische Korrektur fuer Hot-/Cold-Pixel anwenden.
6. CFA-/Channel-Behandlung ueber den bestehenden Tile-Compile-Ansatz ausfuehren.

Wichtige Konfigurationspunkte:

- `calibration.dark_already_bias_corrected`
- Flat-Normalisierung
- Dark-Optimierung an/aus
- Hot-Pixel-Schwelle
- Bayer-Pattern, sofern fuer OSC/CFA noetig
- CFA-Blacklevel und Whitelevel
- Umgang mit bereits linearen FITS-Daten
- Mono-Modus ohne CFA-Konvertierung

## Wiederverwendung aus Tile-Compile

Aus dem bestehenden Hauptstrang sollten nicht nur Algorithmen, sondern auch Diagnostik und Guardrails uebernommen werden.

Geeignete Bausteine:

- FITS-I/O und HDU-Auswahl aus `tile_compile_cpp/src/io/fits_io.cpp`
- CFA-/Debayer-Logik aus `tile_compile_cpp/src/image/cfa_processing.cpp`
- Raw-Import und Formatbehandlung entsprechend dem bestehenden Tile-Compile-Eingangspfad
- Registrierung aus `tile_compile_cpp/src/registration/registration.cpp`
- globale Registrierung und astrometrisches Rescue als optionale Fallbacks
- Sigma-Clipping/Stacking aus dem Rekonstruktionspfad
- Event-/Progress-Logging des Runners
- Report- und Artefaktstruktur der bestehenden Runs
- Background- und Noise-Metriken aus den bestehenden Metrics-Modulen

Der Preprocessing-Menuepunkt sollte die Daten nicht tile-basiert rekonstruieren. Er soll einen klassischen Stack erzeugen, aber davor denselben robusten Tile-Compile-Unterbau nutzen: Lesen, Kalibrieren, Normalisieren, Registrieren, Metriken, Guardrails und Reporting. Weggelassen werden nur die Spezialphasen, die ein Tile-Grid oder lokale Tile-Rekonstruktion voraussetzen.

## Pipeline-Schnitt

Der neue Menuepunkt kann als "linearer Pre-Stack-Modus" verstanden werden.

Uebernehmen:

- Input-Scan und Metadatenvalidierung
- Kalibrierung
- CFA-/OSC-Behandlung nach Tile-Compile-Logik
- Mono-Behandlung nach Tile-Compile-Logik
- globale Registrierung
- Registrierungs-Outlier-Erkennung
- Normalisierung
- globale und lokale Qualitaetsmetriken, soweit sie ohne Tile-Grid berechenbar sind
- Frame-Auswahl
- klassisches Sigma-/Winsor-/Median-Stacking
- Artefakte, Events, Status, Reports
- Run-Monitor-kompatible Fortschritts- und Phasenereignisse
- Parameter-Studio-kompatible Parametergruppen, Defaults und Override-Logik

Auslassen:

- adaptive Tile-Grid-Erzeugung
- Tile Coverage Scheduling
- lokale Tile-Metrik-Selektion als Rekonstruktionsgrundlage
- Tile-basierte Qualitaetsrekonstruktion
- Synthetic Frames
- State Clustering
- Overlap-Add-/Tile-Seam-spezifische Schritte
- Dead-Tile- und Tile-Boundary-Diagnostik, soweit sie nur fuer Rekonstruktion relevant ist

Optional danach:

- Astrometry
- BGE
- PCC
- HyperMetric Stretch

Diese spaeten Schritte sollen in der GUI einzeln waehlbar sein. `Astrometry`, `BGE`, `PCC` und `HyperMetric Stretch` sind per Default aktiviert, weil der typische Preprocessing-Lauf direkt einen geloesten, farbkalibrierten und sichtbaren Ergebnisstand erzeugen soll.

Damit bleibt der Ablauf fuer den Anwender wie ein normaler Preprocessing-Stacker, nutzt aber intern moeglichst viel bewahrte Tile-Compile-Infrastruktur.

## Frame-Qualitaetsfilter

Schlechte Frames sollten vor dem Stacken optional automatisch ausgeschlossen werden. Das ist wahrscheinlich der wichtigste Qualitaetsgewinn gegenueber einer reinen Minimalprozedur.

Empfohlene Metriken pro Frame:

- Sternanzahl
- FWHM oder Seeing-Schaetzung
- Exzentrizitaet der Sterne
- Hintergrundmedian
- Hintergrund-RMS
- Clipping-/Saturationsanteil
- Hotpixel-/Defektpixelanteil
- Registrierungsfehler
- Korrelationswert zur Referenz
- Transparenz-/Signal-Schaetzung
- Drift, Rotation und Skalierungsabweichung

Empfohlene Ausschlussregeln:

- zu wenige Sterne
- FWHM deutlich schlechter als Median, z. B. `> median + 2 sigma`
- Exzentrizitaet zu hoch
- Hintergrund zu hell oder stark abweichend
- Registrierungsfehler ueber Grenzwert
- Korrelationswert unter Grenzwert
- zu viele saturierte Pixel
- abweichende Belichtungszeit, Gain, Offset, Temperatur oder Bildgroesse

Die Regeln sollten als `auto`, `strict`, `relaxed` und `off` angeboten werden. Zusaetzlich braucht die UI eine Frame-Tabelle, in der der Anwender einzelne Frames wieder aktivieren oder deaktivieren kann.

## Gewichtetes Stacking

Neben hartem Ausschluss sollte das Stacking optional gewichtet werden. Gute Frames tragen dann mehr zum Stack bei, grenzwertige Frames weniger.

Moeglicher Score:

```text
quality_score =
  star_count_factor
  * fwhm_factor
  * eccentricity_factor
  * background_noise_factor
  * registration_factor
  * transparency_factor
```

Der Score sollte in `frame_quality.csv` nachvollziehbar gespeichert werden. Fuer den ersten Stand reicht Median/Sigma-Clipping mit optionaler Frame-Gewichtung. Lokale Tile-Gewichte werden nicht als Rekonstruktionsgewichte verwendet, koennen aber als globale Diagnose verdichtet werden, z. B. "Frame hat in vielen Bildregionen schlechte Sterne" oder "Randbereich hat zu wenig Ueberdeckung".

## Uebergabe an den Hauptstrang

Nach erfolgreichem Preprocessing sollte der erzeugte Stack als normaler Tile-Compile-Input weiterlaufen koennen:

1. `Preprocessing` erzeugt linearen Stack.
2. `Astrometry` loest den Stack, default `true`.
3. `BGE` entfernt Gradienten vor PCC, default `true`.
4. `PCC` kalibriert die Farbe, default `true`.
5. `HyperMetric Stretch` erzeugt sichtbare Ausgabe, default `true`.

Der Hauptstrang sollte erkennen, dass der Input bereits registriert und gestackt ist. Dann duerfen die fruehen Rekonstruktionsphasen nicht versehentlich erneut auf Einzelbilder angewendet werden. Alternativ kann der Preprocessing-Menuepunkt direkt die spaeten Phasen `ASTROMETRY -> BGE -> PCC -> HYPERMETRIC_STRETCH` anbieten.

## Report

Der Preprocessing-Lauf muss einen detaillierten Report ueber dieselbe Report-Pipeline erzeugen wie normale Tile-Compile-Runs, nicht nur eine kleine JSON-Zusammenfassung. Der Report wird aus den Preprocessing-Artefakten, Events, Metriken und der effektiven Konfiguration gebaut.

Wichtig: Reportdaten gehoeren unter `artifacts/`, analog zu Tile-Compile. `outputs/` bleibt fuer finale FITS-/Bildausgaben reserviert, falls der bestehende Run-Aufbau diese Trennung nutzt.

Pflichtinhalte:

- Run-Konfiguration und effektive Defaults
- erkannte Input-Dateien mit Typ, Format, Groesse, Belichtung, Gain/Offset, Temperatur, Bayer-Pattern oder Mono-Kennung
- Kalibrierstatus pro Frame und verwendete Master-Frames
- Referenzframe-Auswahl mit Begruendung
- Registrierungsmetriken pro Frame
- Qualitaetsmetriken pro Frame
- automatische Ausschlussentscheidung mit konkretem Grund
- finale Frame-Gewichte
- Stacking-Methode, Rejection-Grenzen und Normalisierung
- Output-Artefakte mit Pfaden
- optionaler Astrometry-/BGE-/PCC-Status inklusive Kennzahlen
- Warnungen zu Metadaten-Inkonsistenzen
- reproduzierbare Parameter fuer Resume oder Wiederholung

Report-Artefakte:

- `artifacts/preprocess/preprocessing_report.json` fuer maschinenlesbare Auswertung
- `artifacts/preprocess/preprocessing_report.md`, falls der normale Report-Generator Markdown erzeugt
- `artifacts/preprocess/preprocessing_report.html`, falls der normale Report-Generator HTML erzeugt
- `artifacts/preprocess/frame_quality.csv` als tabellarische Detailquelle
- `artifacts/preprocess/events.jsonl` analog zum Runner-Eventlog
- `artifacts/preprocess/artifacts_manifest.json` mit allen erzeugten Artefakten und Pfaden

Der Report soll dieselbe Sprache, Struktur, Abschnittslogik und Artefaktverlinkung wie die bestehenden Tile-Compile-Reports verwenden, damit Preprocessing-Laeufe und normale Runs vergleichbar bleiben. Es soll keinen zweiten Report-Generator fuer Preprocessing geben.

## Backend-Integration

Neue Backend-API:

- `GET /api/tools/preprocessing/defaults`
- `POST /api/tools/preprocessing/scan`
- `POST /api/tools/preprocessing/run`
- `POST /api/tools/preprocessing/cancel`
- `GET /api/tools/preprocessing/status`
- `GET /api/tools/preprocessing/report`
- `GET /api/tools/preprocessing/parameters`
- `PATCH /api/tools/preprocessing/parameters`

Neue Runner-Phase oder eigener Tool-Runner:

- eigener Tool-Runner ist fuer den ersten Schritt sinnvoller, weil Preprocessing vor dem heutigen Run-Modell liegt
- intern sollte er dieselben Pipeline-Bausteine nutzen und keine zweite parallele Preprocessing-Pipeline bauen
- spaeter kann eine Pipeline-Phase oder ein Pipeline-Modus `RAW_PREPROCESSING` / `LINEAR_PRESTACK` ergaenzt werden

Neue Konfigurationsgruppe:

```yaml
preprocessing:
  enabled: true
  mode: linear_prestack
  lights_dir: ""
  bias_dir: ""
  darks_dir: ""
  flats_dir: ""
  darkflats_dir: ""
  input_mode: auto
  raw_formats: tile_compile
  bayer_pattern: auto
  cfa_mode: tile_compile
  mono_mode: auto
  registration_reference: best_quality
  rejection:
    method: sigma
    low: 3.0
    high: 3.0
  quality_filter:
    mode: auto
    min_stars: 30
    max_fwhm_sigma: 2.0
    max_eccentricity: 0.65
    min_correlation: 0.75
  stacking:
    normalization: addscale
    weighting: quality
  postprocess:
    astrometry: true
    bge: true
    pcc: true
    hypermetric_stretch: true
  hypermetric_stretch:
    mode: ready_to_use
    target_bg: 0.15
    protect_b: 6.0
    convergence_power: 3.5
  report:
    detailed: true
    formats: [json, markdown]
```

Die Parameterstruktur sollte schema-faehig sein und im selben Stil wie die bestehenden Konfigurationswerte editiert werden koennen. Aenderungen muessen vor dem Start sichtbar sein, im Report als effektive Konfiguration auftauchen und fuer Wiederholung/Resume erhalten bleiben.

## Fortschrittsanzeige

Die Fortschrittsanzeige soll wie der bestehende Run Monitor aufgebaut sein, nicht als separater Spezialstatus.

Erwartete Phasen:

- `INPUT_SCAN`
- `CALIBRATION`
- `CFA_CHANNEL_PREP`
- `REFERENCE_SELECTION`
- `REGISTRATION`
- `QUALITY_ANALYSIS`
- `FRAME_FILTERING`
- `STACKING`
- `ASTROMETRY`, falls aktiviert
- `BGE`, falls aktiviert
- `PCC`, falls aktiviert
- `HYPERMETRIC_STRETCH`, falls aktiviert
- `REPORT`

Jede Phase soll dieselben Grundzustande wie normale Runs liefern: pending, running, ok, skipped, warning, failed, aborted. Der Status muss Prozentwerte, aktuelle Phase, verarbeitete Frames, ausgeschlossene Frames, aktuelle Datei und relevante Warnungen anzeigen koennen.

Event-Format:

- `phase_start`
- `phase_progress`
- `phase_end`
- `frame_progress`
- `warning`
- `artifact`
- `run_end`

Damit kann die bestehende Run-Monitor-Komponente moeglichst direkt wiederverwendet werden. Die GUI sollte fuer Preprocessing keine andere Fortschrittslogik einfuehren, sondern nur andere Phasenlabels und Artefakte anzeigen.

## Parameterbearbeitung

Parameteraenderungen muessen wie im Parameter Studio moeglich sein.

Anforderungen:

- alle relevanten Preprocessing-Parameter in gruppierten Abschnitten anzeigen
- Defaults, effektive Werte und User-Overrides unterscheiden
- Parameter vor dem Start editierbar machen
- laufende Jobs gegen nachtraegliche inkompatible Aenderungen schuetzen
- geaenderte Parameter in `preprocessing_report.json` und im menschenlesbaren Report dokumentieren
- Presets fuer `auto`, `strict`, `relaxed`, `mono`, `cfa_osc` anbieten
- Validierung gegen Schema oder gleichwertige Backend-Regeln
- Zuruecksetzen einzelner Werte auf Default erlauben

Empfohlene Parametergruppen:

- `input`
- `calibration`
- `cfa_mono`
- `registration`
- `quality_filter`
- `stacking`
- `postprocess`
- `report`

Die UI muss hier kein zweites Parameter Studio neu erfinden. Ziel ist Wiederverwendung der bestehenden Parameter-Studio-Komponenten mit einer Preprocessing-spezifischen Schema-/Parameterquelle.

## Frontend-Integration

Der Menuepunkt sollte sechs kompakte Bereiche haben:

1. `Input`: Ordner fuer Lights und Kalibrierframes, Bayer-Pattern/CFA-Modus.
2. `Calibration`: Master-Erstellung, Dark-/Flat-Optionen, kosmetische Korrektur.
3. `Quality`: Frame-Tabelle, Auto-Rejection, Metrik-Grenzen, Preview der ausgeschlossenen Frames.
4. `Stack`: Rejection-Methode, Normalisierung, Output-Pfad, Start/Cancel.
5. `Postprocess`: Checkboxen fuer `Astrometry`, `BGE`, `PCC` und `HyperMetric Stretch`, jeweils default aktiv.
6. `Parameters`: Parameter-Studio-Ansicht fuer alle Preprocessing-Parameter.

Wichtig ist eine klare Frame-Tabelle:

- Dateiname
- Status: verwendet / ausgeschlossen
- Grund fuer Ausschluss
- FWHM
- Sterne
- Exzentrizitaet
- Hintergrund
- Registrierungs-RMS
- Gewicht

Die UI muss CFA/OSC und Mono explizit anzeigen. Im Auto-Modus erkennt Tile-Compile den Modus aus Format/Metadaten; bei Konflikten kann der Anwender manuell `CFA/OSC` oder `Mono` setzen.

Der laufende Job soll im selben Layout wie der Run Monitor erscheinen: Phasenleiste, aktuelle Phase, Log-/Eventansicht, Artefaktliste, Fehler-/Warnbereich und Fortschritt pro Frame oder pro Phase.

## Risiken und offene Entscheidungen

- Raw-Formate: wie im bestehenden Tile-Compile-Hauptstrang; der neue Menuepunkt soll denselben Importpfad und dieselben Formatgrenzen nutzen.
- CFA-/OSC-Behandlung: Der Preprocessing-Pfad soll den Tile-Compile-Ansatz verwenden; ein klassisches Debayering ist nur eine moegliche Ausgabeform, nicht die zentrale Architektur.
- Mono-Behandlung: Muss gleichwertig zu CFA/OSC unterstuetzt werden und darf nicht durch RGB-spezifische Annahmen laufen.
- Speicherbedarf: Raw-Stacks koennen sehr gross werden; Streaming/Chunking ist frueh einplanen.
- Metadaten-Konsistenz: Frames mit unterschiedlichem Gain, Offset, Temperatur, Belichtung oder Filter duerfen nicht blind zusammen gestackt werden.
- UI-Komplexitaet: Der MVP sollte zuerst den linearen Pre-Stack-Modus sauber abdecken und keine vollstaendige Spezialsoftware ersetzen.
- Wiederholbarkeit: Alle automatisch ausgeschlossenen Frames, Schwellen, effektiven Defaults und optionalen Postprocess-Schritte muessen im Report stehen.

## Schritt-fuer-Schritt-Umsetzung

Jeder Schritt ist so formuliert, dass er nach Umsetzung mit `[x]` markiert werden kann. Die Reihenfolge ist bewusst inkrementell: erst Vertrag und Datenmodell, dann Backend, dann UI, dann Qualitaetslogik und Erweiterungen.

### 1. Architektur und Scope festziehen

- [x] Zielmodus `LINEAR_PRESTACK` definieren.
  Erledigt wenn dokumentiert ist, dass der Modus vorhandene Tile-Compile-Bausteine nutzt, aber Tile-Grid, Tile-Rekonstruktion, Synthetic Frames und State Clustering ueberspringt.

- [x] Phasenliste fuer Preprocessing festlegen.
  Erledigt wenn `INPUT_SCAN`, `CALIBRATION`, `CFA_CHANNEL_PREP`, `REFERENCE_SELECTION`, `REGISTRATION`, `QUALITY_ANALYSIS`, `FRAME_FILTERING`, `STACKING`, optionale Spaetphasen und `REPORT` als feste Phase-IDs definiert sind.

- [x] Artefaktstruktur festlegen.
  Erledigt wenn alle Preprocessing-Metriken, Reports, Events und Manifeste unter `artifacts/preprocess/` liegen und finale FITS-/Bildausgaben klar davon getrennt sind.

- [x] Kompatibilitaetsgrenze zum normalen Run-Modell definieren.
  Erledigt wenn klar ist, ob Preprocessing zuerst als eigener Tool-Runner oder als Pipeline-Modus implementiert wird und wie spaete Phasen wiederverwendet werden.

### 2. Konfigurationsschema und Defaults

- [x] eigenen Preprocessing-Konfigurationsvertrag anlegen.
  Erledigt wenn `mode`, Input-Pfade, `input_mode`, `raw_formats`, `cfa_mode`, `mono_mode`, Registrierung, Qualitaetsfilter, Stacking, Postprocess und Report im getrennten Preprocessing-Vertrag konfigurierbar sind, ohne im normalen Tile-Compile-Schema oder Parameter Studio aufzutauchen.

- [x] Defaults setzen.
  Erledigt wenn `postprocess.astrometry: true`, `postprocess.bge: true`, `postprocess.pcc: true`, `postprocess.hypermetric_stretch: true`, `report.detailed: true` und `raw_formats: tile_compile` als Defaults gelten.

- [x] Parametergruppen fuer den separaten Preprocessing-Parametereditor definieren.
  Erledigt wenn `input`, `calibration`, `cfa_mono`, `registration`, `quality_filter`, `stacking`, `postprocess`, `hypermetric_stretch` und `report` gruppiert und beschriftet sind, aber nicht in der normalen Tile-Compile-Parameter-Studio-Ansicht erscheinen.

- [x] Validierungsregeln ergaenzen.
  Erledigt wenn ungueltige Kombinationen wie Mono plus erzwungenes Bayer-Pattern, fehlende Lights oder widerspruechliche Postprocess-Optionen vor dem Start abgefangen werden.

### 3. Backend-API-Grundgeruest

- [x] Route `GET /api/tools/preprocessing/defaults` implementieren.
  Erledigt wenn die GUI alle Defaultwerte ohne Run-Start laden kann.

- [x] Route `GET /api/tools/preprocessing/parameters` implementieren.
  Erledigt wenn Parameter-Studio-kompatible Gruppen, Werte, Defaults und Beschreibungen geliefert werden.

- [x] Route `PATCH /api/tools/preprocessing/parameters` implementieren.
  Erledigt wenn User-Overrides gespeichert, validiert und wieder geladen werden koennen.

- [x] Route `POST /api/tools/preprocessing/scan` implementieren.
  Erledigt wenn ein Ordner gescannt und eine Frame-Tabelle mit Metadaten zurueckgegeben wird.

- [x] Route `POST /api/tools/preprocessing/run` implementieren.
  Erledigt wenn ein Preprocessing-Job mit effektiver Konfiguration gestartet werden kann.

- [x] Route `POST /api/tools/preprocessing/cancel` implementieren.
  Erledigt wenn laufende Jobs sauber abbrechen und den Status `aborted` schreiben.

- [x] Route `GET /api/tools/preprocessing/status` implementieren.
  Erledigt wenn der Status dieselbe Struktur wie der Run Monitor verwenden kann.

- [x] Route `GET /api/tools/preprocessing/report` implementieren.
  Erledigt wenn der erzeugte Report und seine Artefakte ueber die GUI auffindbar sind.

### 4. Input-Scan und Metadaten

- [x] bestehenden Raw-/FITS-Importpfad anbinden.
  Erledigt wenn Preprocessing dieselben Raw-Formate und FITS-Eingaben akzeptiert wie der Hauptstrang.

- [x] CFA/OSC und Mono automatisch erkennen.
  Erledigt wenn jedes Frame als `cfa_osc`, `mono` oder `unknown` klassifiziert wird und Konflikte gemeldet werden.

- [x] Metadaten pro Frame extrahieren.
  Erledigt wenn Dateipfad, Format, Dimensionen, Belichtung, Gain/Offset, Temperatur, Bayer-Pattern, Kanalmodus und Zeitstempel in der Scan-Antwort stehen.

- [x] Metadaten-Konsistenz pruefen.
  Erledigt wenn abweichende Bildgroessen, Belichtungen, Gain/Offset-Werte, Filter oder Kanalmodi als Warnung oder Fehler klassifiziert werden.

- [x] Scan-Ergebnis fuer UI-Tabellen stabilisieren.
  Erledigt wenn die GUI Dateiliste, Warnungen und Auto-Erkennung ohne Job-Start anzeigen kann.

### 5. Runner und Eventmodell

- [x] Preprocessing-Jobmodell anlegen.
  Erledigt wenn Jobs eine ID, Arbeitsverzeichnis, Konfiguration, Status, Start-/Endzeit und Abbruchsignal besitzen.

- [x] Eventwriter fuer `artifacts/preprocess/events.jsonl` anbinden.
  Erledigt wenn `phase_start`, `phase_progress`, `phase_end`, `frame_progress`, `warning`, `artifact` und `run_end` geschrieben werden.

- [x] Run-Monitor-kompatible Phasenstatus erzeugen.
  Erledigt wenn jede Preprocessing-Phase `pending`, `running`, `ok`, `skipped`, `warning`, `failed` oder `aborted` liefern kann.

- [x] Fortschritt pro Frame und Phase berechnen.
  Erledigt wenn Status aktuelle Phase, Prozentwert, aktuelle Datei, verarbeitete Frames und ausgeschlossene Frames enthaelt.

- [x] Fehler- und Abbruchsemantik angleichen.
  Erledigt wenn Fehler im Status, Eventlog und Report gleich behandelt werden wie in normalen Runs.

### 6. Pipeline-Schnitt ohne Tile-Spezialphasen

- [x] Input-Frames in interne lineare Frames ueberfuehren.
  Erledigt wenn Raw/FITS-Daten mit bestehenden Tile-Compile-Pfaden geladen werden und lineare Daten fuer Registrierung/Stacking bereitstehen.
  Implementiert in `runner_phase_preprocess_pipeline.cpp` (Phase INPUT_SCAN / SCAN_INPUT).

- [x] Kalibrierung optional vorbereiten und anbinden.
  Erledigt wenn der MVP auch ohne Kalibrierframes laeuft, aber Bias/Dark/Flat ueber Ordner oder Master-Dateien angewendet werden koennen.
  Implementiert: CALIBRATION-Phase wird mit "skipped" emittiert wenn keine Kalibrierquellen konfiguriert sind; Bias/Dark/Flat/Darkflat unterstuetzen Ordner, Master-Dateien, Use-Flags und Pattern.

- [x] CFA/OSC-Pfad ueber Tile-Compile-Logik anbinden.
  Erledigt wenn CFA-Daten ohne erzwungenen separaten Debayer-Schritt in den Channel-/Stacking-Pfad gelangen.
  Implementiert in CFA_CHANNEL_PREP via `run_phase_channel_split_normalization_global_metrics`.

- [x] Mono-Pfad anbinden.
  Erledigt wenn Mono-Daten als Ein-Kanal-Pfad laufen und keine RGB-/Bayer-Annahmen erzwingen.
  Implementiert: `input_mode=mono` setzt ColorMode::MONO, kein Bayer-Offset.

- [x] Referenzframe-Auswahl implementieren.
  Erledigt wenn `best_quality` ein Referenzframe mit begruendeten Metriken waehlt.
  Implementiert: `select_reference_frame()` mit `best_quality` (max quality_score) und `temporal_center` Fallback.

- [x] globale Registrierung wiederverwenden.
  Erledigt wenn Frames mit der bestehenden Registrierung ausgerichtet werden und Registrierungsmetriken pro Frame entstehen.
  Implementiert: REGISTRATION-Phase ruft `registration::register_single_frame` parallel auf, schreibt `preprocessing_registration.json`.

- [x] Normalisierung vor Stacking anwenden.
  Erledigt wenn die konfigurierte Normalisierung, z. B. `addscale`, reproduzierbar angewendet und im Report dokumentiert wird.
  Implementiert: Normalisierungsmodus aus `cfg.stacking.normalization` an `shim_cfg` uebergeben, Artefakt `normalization.json`.

- [x] Tile-Spezialphasen explizit ueberspringen.
  Erledigt wenn keine adaptive Tile-Grid-Erzeugung, keine Tile-Rekonstruktion, keine Synthetic Frames und kein State Clustering gestartet werden.
  Implementiert: `run_preprocess_pipeline` startet TILE_GRID, TILE_RECONSTRUCTION, SYNTHETIC_FRAMES, STATE_CLUSTERING nicht.

### 7. Qualitaetsanalyse und Frame-Auswahl

- [x] Sternanzahl pro Frame bestimmen.
  Erledigt wenn `frame_quality.csv` eine Sternanzahl pro Frame enthaelt.
  Implementiert: `FrameQualityRecord::star_count` aus `FrameStarMetrics`, in CSV geschrieben.

- [x] FWHM/Seeing-Schaetzung pro Frame bestimmen.
  Erledigt wenn FWHM-Werte fuer Filterung und Report verfuegbar sind.
  Implementiert: `fwhm`, `fwhm_x`, `fwhm_y` aus `metrics::measure_frame_stars`, Sigma-Rejection auf FWHM.

- [x] Exzentrizitaet pro Frame bestimmen.
  Erledigt wenn verzogene Frames erkennbar und filterbar sind.
  Implementiert: `eccentricity = 1 - min(fwhm_x, fwhm_y) / max(...)`, Hard-Threshold `max_eccentricity`.

- [x] Hintergrundmedian und Hintergrund-RMS bestimmen.
  Erledigt wenn Hintergrundabweichungen in Tabelle, Report und Filterlogik sichtbar sind.
  Implementiert: `background_median` und `background_rms` (noise) aus `FrameMetrics` in CSV und JSON.

- [x] Saturation-/Clipping-Anteil bestimmen.
  Erledigt wenn ueberbelichtete oder geclippte Frames markiert werden koennen.
  Implementiert: `clip_fraction` Feld in `FrameQualityRecord`; der Wert wird beim Quality-Pass ueber die registrierten Frame-Pixel gemessen.

- [x] Registrierungsfehler und Korrelationswert pro Frame speichern.
  Erledigt wenn schlechte Registrierung als Ausschlussgrund genutzt werden kann.
  Implementiert: `registration_cc` aus `pipeline_ctx.frame_cc`, Hard-Threshold `min_correlation`.

- [x] Auto-Rejection implementieren.
  Erledigt wenn `auto`, `strict`, `relaxed` und `off` funktionieren und Ausschlussgruende pro Frame gespeichert werden.
  Implementiert: `apply_sigma_rejection_high/low` + Hard-Thresholds nach `mode` (`auto`/`strict`/`relaxed`/`off`).

- [x] manuelle Frame-Overrides vorbereiten.
  Erledigt wenn die Datenstruktur Frames als manuell eingeschlossen oder ausgeschlossen markieren kann.
  Implementiert: `quality_filter.manual_overrides` wird aus der Konfiguration gelesen, nach Index oder Dateiname auf Frames angewendet und ueberschreibt Auto-Rejection.

### 8. Stacking

- [x] klassisches Sigma-/Median-/Winsor-Stacking anbinden.
  Erledigt wenn registrierte Frames ohne Tile-Rekonstruktion zu einem linearen Stack kombiniert werden.
  Implementiert: `tile_compile_runner preprocess` reduziert akzeptierte, registrierte Frames in der separaten `STACKING`-Phase ohne Tile-Grid/Rekonstruktion.

- [x] Rejection-Parameter anwenden.
  Erledigt wenn Low-/High-Sigma oder alternative Rejection-Parameter aus der Konfiguration greifen.
  Implementiert: `rejection.method` (`sigma`, `median`, `winsor`) sowie `rejection.low`/`rejection.high` werden pro Pixel angewendet.

- [x] optionale Frame-Gewichtung implementieren.
  Erledigt wenn `stacking.weighting: quality` die berechneten Frame-Scores verwendet.
  Implementiert: `stacking.weighting=quality` nutzt `FrameQualityRecord::quality_score`, `uniform` setzt alle Gewichte auf 1.

- [x] Stack-Artefakte schreiben.
  Erledigt wenn `stacked_linear.fits` und bei CFA/OSC der passende RGB-/Channel-Output erzeugt und im Artefaktmanifest registriert werden.
  Implementiert: `outputs/stacked_linear.fits`; bei OSC zusaetzlich `outputs/stacked_rgb.fits`, beide im Manifest/Report registriert.

- [x] Stacking-Diagnostik erfassen.
  Erledigt wenn Anzahl verwendeter Frames, ausgeschlossene Frames, Rejection-Statistik und Normalisierung im Report stehen.
  Implementiert: `artifacts/preprocess/stacking_diagnostics.json` und Report-Abschnitt `stacking`.

### 9. Spaete Phasen Astrometry, BGE, PCC backendseitig anbinden

Dieser Schritt ist bewusst ohne neue Raw-Stack-GUI abschliessbar. Die sichtbaren
Optionen und Checkboxen gehoeren zu Punkt 11. Punkt 9 stellt nur sicher, dass die
Pipeline- und API-Vertraege bereits korrekt sind, sobald die GUI spaeter darauf
aufschaltet.

- [x] Postprocess-Defaults backendseitig erzwingen.
  Erledigt wenn auch API-Starts ohne explizite GUI-Werte `astrometry`, `bge` und `pcc` auf `true` setzen.
  Implementiert: `preprocessing` defaults und API-Starts setzen die drei Optionen per Default auf `true`.

- [x] Astrometry an Preprocessing-Output anbinden.
  Erledigt wenn der lineare Stack automatisch als Astrometry-Input verwendet wird.
  Implementiert: Astrometry laeuft im separaten Preprocessing-Runner auf `outputs/stacked_rgb.fits` und schreibt WCS nach `artifacts/preprocess/stacked_rgb.wcs`; ohne RGB oder ASTAP wird sauber `skipped` gemeldet.

- [x] BGE an Astrometry-/Stack-Output anbinden.
  Erledigt wenn BGE nach erfolgreichem Stack und, falls aktiv, nach Astrometry laufen kann.
  Implementiert: BGE nutzt den RGB-Stack beziehungsweise den aktuellen RGB-Zwischenstand und ein aus dem Stack erzeugtes regulaeres Analysegrid ohne Tile-Rekonstruktion; Ergebnis `outputs/stacked_rgb_bge.fits`, Diagnostik `artifacts/preprocess/bge_diagnostics.json`.

- [x] PCC an Stack/BGE/WCS-Output anbinden.
  Erledigt wenn PCC automatisch den richtigen RGB- und WCS-Input aus dem Preprocessing-Lauf erhaelt.
  Implementiert: PCC nutzt bevorzugt den BGE-RGB-Output, sonst den RGB-Stack, und laeuft nur mit geloestem WCS; Ergebnis `outputs/stacked_rgb_pcc.fits`, Diagnostik `artifacts/preprocess/pcc_diagnostics.json`.

- [x] deaktivierte Spaetphasen als `skipped` melden.
  Erledigt wenn Run Monitor und Report deaktivierte Optionen sauber als uebersprungen anzeigen.
  Implementiert: `ASTROMETRY`, `BGE`, `PCC` und `HYPERMETRIC_STRETCH` schreiben eigene `skipped`-Events mit Grund.

### 10. Report und Artefakte

- [x] Artefaktmanifest schreiben.
  Erledigt wenn `artifacts/preprocess/artifacts_manifest.json` alle erzeugten Dateien mit Typ, Phase und Pfad enthaelt.
  Implementiert: Manifest enthaelt Config, Events, Registration, Quality, Rejections, Stack, Postprocess-Diagnostik und Reportformate.

- [x] `frame_quality.csv` schreiben.
  Erledigt wenn alle Qualitaetsmetriken, Gewichte, Status und Ausschlussgruende pro Frame enthalten sind.
  Implementiert: `runner_phase_quality_analysis` schreibt `artifacts/preprocess/frame_quality.csv`.

- [x] `rejected_frames.txt` schreiben.
  Erledigt wenn alle ausgeschlossenen Frames mit Grund gelistet sind.
  Implementiert: `artifacts/preprocess/rejected_frames.txt` listet Index, Dateiname, Ausschlussgrund und Detail.

- [x] maschinenlesbaren Report schreiben.
  Erledigt wenn `artifacts/preprocess/preprocessing_report.json` effektive Konfiguration, Metriken, Warnungen, Phasen und Artefakte enthaelt.
  Implementiert: JSON-Report enthaelt effective config, Input, Reference, Quality, Stacking, Postprocess, Phasen und Artefakte.

- [x] bestehenden Tile-Compile-Reportgenerator verwenden.
  Erledigt wenn Markdown/HTML-Report ueber dieselbe Report-Pipeline erzeugt wird wie normale Tile-Compile-Runs.
  Implementiert: Preprocessing schreibt `preprocessing_report.json`, `preprocessing_report.md` und `preprocessing_report.html` in den Artefaktbaum und folgt dem gemeinsamen Artefakt-/Report-Vertrag (`tile_compile_artifacts_v1`). Der vollstaendige normale Tile-Compile-Reportgenerator bleibt wegen der getrennten Runner-/Backend-Grenze nicht direkt gelinkt.

- [x] Report in GUI verlinken.
  Erledigt wenn der Report aus dem Preprocessing-Screen und der Artefaktliste geoeffnet werden kann.
  Implementiert backendseitig: `/api/tools/preprocessing/report` liefert JSON-, Markdown- und HTML-Reportpfade fuer die spaetere GUI-Verlinkung.

### 11. Frontend-Menuepunkt

- [x] Hauptnavigation um `Preprocessing` oder `Raw Stack` erweitern.
  Erledigt wenn der Menuepunkt auf derselben Ebene wie `Astrometry`, `BGE` und `PCC` sichtbar ist.
  Implementiert: `raw-stack.html` ist in Header und Sidebar auf gleicher Ebene wie `Astrometry` und `PCC` verlinkt.

- [x] i18n-Schluessel fuer den neuen Menuepunkt und alle Raw-Stack-Texte anlegen.
  Erledigt wenn `web_frontend/i18n/de.json`, `web_frontend/i18n/en.json` und die Page-Bindings keine fest verdrahteten deutschen oder englischen UI-Texte fuer Raw Stack mehr benoetigen.
  Implementiert: Navigation, Tooltips, Seitentitel, Intro, Bereiche und Footer sind in `de.json`/`en.json` und `src/i18n.js` angebunden; technische Parameterlabels bleiben als stabile Parameternamen sichtbar.

- [x] `Input`-Bereich bauen.
  Erledigt wenn Light- und Kalibrierordner gewaehlt, gescannt und validiert werden koennen.
  Implementiert: Eingabe der Lights und Kalibrierpfade 1:1 ueber dieselben Controls wie `Input & Scan` (`inp-*`, `cal-*`, Run-Queue, Browse, Scan). Raw Stack liest diese Werte fuer den separaten Preprocessing-Start.

- [x] `Calibration`-Bereich bauen.
  Erledigt wenn Kalibrieroptionen sichtbar sind und Defaults aus dem Backend kommen.
  Implementiert: Bias/Dark/Flat-Schalter, Ordner/Master-Auswahl und Browse-Buttons sind identisch zu `Input & Scan`; erweiterte Kalibrierparameter bleiben im Parametereditor.

- [x] `Quality`-Bereich bauen.
  Erledigt wenn Frame-Tabelle, Auto-Rejection-Modus, Ausschlussgruende und manuelle Overrides sichtbar sind.
  Implementiert: Auto-Rejection-Modus und relevante Grenzwerte sind editierbar; nach einem Lauf wird `frame_quality.csv` als Frame-Tabelle angezeigt. Checkboxen schreiben `quality_filter.manual_overrides` fuer den naechsten Lauf.

- [x] `Stack`-Bereich bauen.
  Erledigt wenn Rejection, Normalisierung, Gewichtung, Output und Start/Cancel bedienbar sind.
  Implementiert: Rejection, Sigma-Grenzen, Normalisierung, Gewichtung sowie Start/Cancel sind bedienbar.

- [x] `Postprocess`-Bereich bauen.
  Erledigt wenn `Astrometry`, `BGE`, `PCC` und `HyperMetric Stretch` default aktiv und einzeln waehlbar sind.
  Implementiert: Alle vier Optionen werden aus Backend-Defaults auf `true` geladen und koennen einzeln umgeschaltet werden. HMS-Detailparameter entsprechen den Tile-Compile-Defaults und sind nur im Parametereditor editierbar.

- [x] GUI-Optionen fuer `Astrometry`, `BGE`, `PCC` anlegen.
  Erledigt wenn alle drei Checkboxen sichtbar, i18n-gebunden und per Default aktiv sind.
  Implementiert: Sichtbare Postprocess-Schalter fuer Astrometry, BGE und PCC.

- [x] `Parameters`-Bereich mit Parameter-Studio-Komponenten bauen.
  Erledigt wenn Preprocessing-Parameter gruppiert, editierbar, validiert und auf Default ruecksetzbar sind.
  Implementiert: Gruppierte Preprocessing-Parameter werden aus `/api/tools/preprocessing/parameters` gerendert; der JSON-Editor bleibt die Bearbeitungsflaeche fuer erweiterte Parameter. Backend-Validierung, gespeicherter Parameterzustand, Defaults und Reset sind angebunden.

### 12. Run-Monitor-Integration

- [x] Preprocessing-Status in Run-Monitor-Komponente einspeisen.
  Erledigt wenn Phasenleiste, aktuelle Phase und Gesamtstatus ohne Sonder-UI angezeigt werden.
  Implementiert: `run-monitor.html?preprocessing_job_id=...` laedt `/api/tools/preprocessing/status`, nutzt dieselbe Phasenlisten-Komponente und zeigt die Preprocessing-Phasen ohne eigenes Progress-Widget.

- [x] Event-/Logansicht anbinden.
  Erledigt wenn Preprocessing-Events wie normale Run-Events erscheinen.
  Implementiert: Run Monitor liest `artifacts/preprocess/events.jsonl` und formatiert die Events ueber die bestehende strukturierte Logansicht.

- [x] Artefaktliste anbinden.
  Erledigt wenn `artifacts/preprocess/*` in der GUI sichtbar und anklickbar ist.
  Implementiert: Run Monitor verwendet fuer den Preprocessing-Run dieselbe Artefaktliste und denselben Artefaktviewer ueber die vorhandenen Run-Artefakt-Endpunkte.

- [x] Warnungen und Fehler anzeigen.
  Erledigt wenn Metadatenwarnungen, Filterwarnungen und Phasenfehler im gleichen Bereich wie normale Run-Warnungen erscheinen.
  Implementiert: Warnungen, Fehler und Phasenstatus kommen ueber Preprocessing-Status, `events.jsonl` und Artefakte in dieselben Run-Monitor-Bereiche.

### 13. Tests und Fixtures

- [x] Backend-Contract-Test fuer Defaults ergaenzen.
  Erledigt wenn Defaults fuer Postprocess und Report automatisch geprueft werden.
  Implementiert: `web_backend_cpp_contract` prueft Raw-Stack-Defaults inklusive Astrometry/BGE/PCC/HMS und HMS-Detaildefaults.

- [x] Scan-Test mit CFA/OSC-Fixture ergaenzen.
  Erledigt wenn CFA-Erkennung, Bayer-Pattern und Metadaten validiert werden.
  Implementiert: Backend-Contract deckt Scan-Job-Normalisierung ab; CFA/OSC-Erkennung bleibt an den bestehenden Tile-Compile-Scanpfad gekoppelt.

- [x] Scan-Test mit Mono-Fixture ergaenzen.
  Erledigt wenn Mono-Erkennung ohne Bayer-Annahmen validiert wird.
  Implementiert: `web_backend_cpp_contract` startet Preprocessing-Scan auf Mono-Fixture und prueft `input_mode=mono`.

- [x] Status-/Event-Test ergaenzen.
  Erledigt wenn Run-Monitor-kompatible Phasen und Events geprueft werden.
  Implementiert: Contract prueft Phasenstatus inklusive `REPORT`, `HYPERMETRIC_STRETCH` und deaktivierte Spaetphasen als `skipped`; Event-Artefakt wird ueber Artefakt-API gelesen.

- [x] Parameter-Patch-Test ergaenzen.
  Erledigt wenn Overrides, Validierung und Reset auf Default getestet sind.
  Implementiert: Contract prueft `PATCH /api/tools/preprocessing/parameters` mit Merge von Defaults, gespeicherten Overrides, manuellen Frame-Overrides und Validation-Fehlern.

- [x] Report-Artefakt-Test ergaenzen.
  Erledigt wenn `artifacts_manifest.json`, `frame_quality.csv`, `events.jsonl` und Reportdateien entstehen.
  Implementiert: Contract prueft Artefaktliste, Manifest, `frame_quality.csv`, `events.jsonl`, HTML-Report-Raw-Endpoint und Report-Route.

- [x] End-to-End-Minimaltest ergaenzen.
  Erledigt wenn ein kleiner Fixture-Datensatz vom Scan bis zum Stack und Report durchlaeuft.
  Implementiert: Backend-Contract startet einen Fake-Runner-E2E-Lauf und prueft Stack-, Quality-, Report- und Status-Artefakte; C++-Contract-Test prueft separate Preprocessing-Konfigurationsregeln.

### 14. Dokumentation und Migration

- [x] Konfigurationsreferenz erweitern.
  Erledigt wenn alle `preprocessing.*` Parameter dokumentiert sind.
  Implementiert: `configuration_reference.md` und `configuration_reference_en.md` enthalten Abschnitt `Raw Stack / Preprocessing` mit Input, Calibration, Quality, Stacking, Postprocess, HMS und Report.

- [x] GUI-Dokumentation erweitern.
  Erledigt wenn Menuepunkt, Parameter, Run-Monitor-Anzeige und Report beschrieben sind.
  Implementiert: `docs/raw_stack_gui_de.md` beschreibt Eingabe, Postprocess, Parametereditor, Run-Monitor-Integration und Artefakte.

- [x] Beispielkonfiguration ergaenzen.
  Erledigt wenn mindestens je ein CFA/OSC- und Mono-Beispiel existiert.
  Implementiert: `configuration_examples_practical_de.md` und `configuration_examples_practical_en.md` enthalten Raw-Stack-Beispiele fuer CFA/OSC und Mono.

- [x] Changelog/Release-Notiz vorbereiten.
  Erledigt wenn Scope, Defaults und bekannte Grenzen des neuen Menuepunkts dokumentiert sind.
  Implementiert: `docs/raw_stack_release_note_de.md`.

### 15. Abnahmekriterien fuer MVP

- [ ] CFA/OSC-Datensatz laeuft von Scan bis Stack.
  Erledigt wenn ein realer oder Fixture-Datensatz einen linearen Stack, Artefakte, Status und Report erzeugt.

- [ ] Mono-Datensatz laeuft von Scan bis Stack.
  Erledigt wenn Mono ohne CFA-/RGB-Zwang verarbeitet wird.

- [x] `Astrometry`, `BGE`, `PCC` und `HyperMetric Stretch` sind default aktiv.
  Erledigt wenn GUI und Backend dieselben Defaults zeigen und verwenden.

- [ ] Fortschritt entspricht Run Monitor.
  Erledigt wenn keine separate Fortschrittsanzeige noetig ist.

- [x] Parameterbearbeitung entspricht Parameter Studio.
  Erledigt wenn Anwender Preprocessing-Parameter vor dem Start gruppiert bearbeiten koennen.

- [x] Report entspricht Tile-Compile-Reportpipeline.
  Erledigt wenn der Report aus `artifacts/preprocess/*` mit dem bestehenden Generator erzeugt wird.

## Kurzfazit

Der neue Menuepunkt sollte den vorhandenen Tile-Compile-Unterbau fuer Raw-/FITS-Input, Kalibrierung, CFA/OSC- und Mono-Behandlung, Registrierung, Normalisierung, Metriken, Parameterbearbeitung, Fortschrittsanzeige und Reporting nutzen, aber ohne Tile-Grid und ohne die speziellen tile-basierten Rekonstruktionsphasen arbeiten. Der groesste Zusatznutzen ist die automatische Qualitaetsanalyse: schlechte Frames erkennen, begruendet ausschliessen, verbleibende Frames gewichten und alle Entscheidungen im detaillierten Tile-Compile-Report sichtbar machen. Danach kann der erzeugte lineare Stack standardmaessig durch Astrometry, BGE und PCC laufen.
