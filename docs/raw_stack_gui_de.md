# Raw Stack – GUI-Dokumentation

Raw Stack ist ein eigenstaendiger Menuepunkt in der Tile-Compile-GUI fuer lineares Preprocessing von FITS-Lights bis zum gestackten Bild. Der Prozess laeuft vollstaendig getrennt vom normalen Tile-Compile Run Studio. Er teilt Algorithmen, Artefaktinfrastruktur, Run-Monitor und Parameter-Studio-Bedienlogik, startet aber als separater Tool-Runner und erscheint nicht in der normalen Tile-Compile-Phasenliste.

Die Parametereingabe erfolgt im Raw-Stack-Menue ueber sichtbare Felder und den Abschnitt **Parameters**. Dieser Abschnitt ist wie das Parameter Studio aufgebaut: gruppierte Parameter, editierbare Werte, JSON-Editor, Validate und Reset. Raw Stack nutzt dabei eine eigene Preprocessing-Konfiguration; Parameter koennen aber aus einer geladenen Tile-Compile-YAML uebernommen werden, sofern sie fuer Raw Stack relevant sind.

---

## Ueberblick: Verarbeitungspfad

```
Input Scan
  -> Calibration (Bias / Dark / Flat)
  -> CFA/Mono Prep (Normalisierung, Bayer)
  -> Reference Selection
  -> Registration (globale affine Warp-Matrizen)
  -> Quality Analysis (Sterne, FWHM, Exzentrizitaet, Korrelation)
  -> Frame Filtering (Auto-Rejection, manuelle Overrides)
  -> Stacking (Sigma/Median/Winsor, Normalisierung, Gewichtung)
  -> Astrometry (ASTAP, optional)
  -> BGE – Background Gradient Extraction (optional)
  -> PCC – Photometric Color Calibration (optional)
  -> HyperMetric Stretch (optional)
  -> Report
```

Jede Phase emittiert einen `phase_start`- und einen `phase_end`-Event in `artifacts/preprocess/events.jsonl`. Phasen, die nicht konfiguriert sind oder deren Voraussetzungen fehlen, werden als `skipped` beendet – sie brechen den Lauf nicht ab.

---

## Bereich: Input & Scan

### Eingabeordner

Der Ordner mit den Light-Frames (FITS-Dateien). Mehrere Ordner koennen ueber die Run-Queue hinzugefuegt werden. Fuer eine einzelne Session genuegt ein Ordner im Feld **Eingabeordner**.

### Dateimuster

Filtert, welche Dateien innerhalb des Ordners als Light-Frames behandelt werden. Standard: `*.fits`. Erlaubt sind einfache Glob-Muster wie `*.fit`, `light_*.fits`.

### Ausgabeordner (Runs-Dir)

Elternverzeichnis, unter dem Raw Stack den Run-Ordner anlegt. Der vollstaendige Pfad wird aus Ausgabeordner + Run-Name + Zeitstempel zusammengesetzt: `<runs_dir>/<run_name>_YYYYMMDD_HHMMSS`.

### Run Name

Optionaler benutzerdefinierter Prafix fuer den Run-Ordner. Leer gelassen wird `run` verwendet.

### Frames Minimum / Max. Frames

- **Frames Minimum**: Guardrail-Schwelle. Wird sie unterschritten, wird im Log gewarnt.
- **Max. Frames**: Begrenzt die Anzahl der verarbeiteten Frames. `0` bedeutet unbegrenzt.

### Farbmodus

- **OSC**: CFA/Bayer-Sensor (z. B. DSLR, Farbkamera). Der Bayer-Debayer-Pfad wird aktiviert.
- **MONO**: Monochrom-Sensor. Kein Bayer-Handling, kein RGB-Stack.
- Default: `auto` – wird aus dem FITS-Header erkannt (`COLORTYP`, `BAYERPAT`, `NAXIS`).

### Bayer-Pattern

Wird nur bei OSC verwendet. `auto (aus FITS-Header)` liest `BAYERPAT` oder `COLORTYP`. Explizite Werte: `RGGB`, `GBRG`, `GRBG`, `BGGR`. Bei falschem Pattern entstehen Farbfehler im Stack.

### Checksummen

Optionaler MD5/SHA-Scan beim Input-Scan. Erhoeht die Scanzeit, empfohlen nur fuer Integritaetspruefungen grosser Archive.

---

## Bereich: Run-Queue

Die Run-Queue erlaubt es, mehrere Eingabeordner mit unterschiedlichen Filtern (L, R, G, B, Ha, ...) zu einem einzigen Raw-Stack-Lauf zu kombinieren. Jeder Queue-Eintrag besteht aus:

- **Filter**: Kanalbezeichnung, z. B. `L`, `R`, `G`, `B`, `Ha`. Bei OSC-Datensaetzen wird automatisch `OSC` gesetzt.
- **Input Dir**: Verzeichnis mit den Light-Frames fuer diesen Kanal.
- **Pattern**: Optionales Dateimuster fuer diesen Kanal.
- **Run Label**: Optionale Bezeichnung des Queue-Eintrags.
- **Aktiv**: Deaktiviert einen Eintrag ohne ihn zu loeschen.

Neue Eintraege werden ueber den `+`-Button aus dem aktuellen Eingabeordner-Feld uebernommen. Eintraege koennen mit `-` entfernt werden.

> **Hinweis:** Ein direkter Klick auf `Start` ohne Queue-Eintrag startet einen Single-Ordner-Lauf mit dem aktuell eingetragenen Eingabeordner.

---

## Bereich: Kalibrierung

Die Kalibrierung korrigiert Sensor-Artefakte vor der Registrierung. Alle drei Stufen sind unabhaengig aktivierbar.

### Bias

Korrigiert den konstanten Elektronik-Offset des Sensors (Readout-Rauschen-Pedestal). Wird vor Dark und Flat angewendet.

- **Checkbox**: Kalibrierung mit Bias aktivieren.
- **Ordner oder Master**: Entweder einen Ordner mit Bias-Frames angeben (Average-Master wird automatisch gebildet) oder eine einzelne Master-Bias-FITS-Datei.
- **Browse**: Ordner oder Datei per Dialog auswaehlen.

### Dark

Korrigiert thermisches Rauschen und Hot-Pixel. Wenn Bias ebenfalls aktiviert ist, wird der Dark-Master automatisch bias-korrigiert bevor er von den Lights subtrahiert wird.

- **Dark-Auto-Select** (im Parametereditor): Bei `calibration.dark_auto_select: true` werden Darks automatisch nach Belichtungszeit aus dem FITS-Header gefiltert. Toleranz konfigurierbar ueber `calibration.dark_match_exposure_tolerance_percent` (Standard: 8 %). Optional auch nach Sensortemperatur (`calibration.dark_match_use_temp`, Toleranz: `calibration.dark_match_temp_tolerance_c`).

### Flat

Korrigiert Vignettierung und Pixelempfindlichkeits-Unterschiede. Der Flat-Master wird auf seinen Median normiert (Pixel / Median), sodass Lights nach Division unveraenderte mittlere Helligkeit behalten.

> **Tipp:** Bias, Dark und Flat sind optional. Werden keine Kalibrierordner angegeben, werden die Lights direkt weiterverarbeitet (Phase wird als `skipped` markiert, kein Fehler).

---

## Bereich: Quality

Steuert die automatische Frame-Selektion nach Qualitaetsmetriken.

| Feld | Beschreibung | Standard |
|------|-------------|---------|
| **Mode** | `auto` – FWHM-k = konfigurierter `max_fwhm_sigma`; `strict` – FWHM-k = 1.5 (eng); `relaxed` – FWHM-k = 3.0 (weit); `off` – kein Filtering | `auto` |
| **Min stars** | Mindestanzahl erkannter Sterne pro Frame | `30` |
| **Min correlation** | Mindest-Kreuzkorrelation mit dem Referenzframe (0–1) | `0.75` |
| **Max FWHM sigma** | Max. Abweichung des FWHM vom Median in Sigma-Einheiten | `2.0` |
| **Max eccentricity** | Maximale mittlere Exzentrizitaet der Sterne (0 = Kreis, 1 = Linie) | `0.65` |

Abgelehnte Frames werden in `artifacts/preprocess/rejected_frames.txt` und in `frame_quality.csv` protokolliert. Im Run-Monitor sind sie als `excluded` markiert.

---

## Bereich: Stack

Konfiguriert Rejection, Normalisierung und Gewichtung des Stacking-Schritts.

| Feld | Beschreibung | Standard |
|------|-------------|---------|
| **Rejection** | Rejection-Methode pro Pixel | `sigma` |
| **Low / High** | Sigma-Grenzen (bei Sigma-Rejection) | `3.0 / 3.0` |
| **Normalization** | Frame-Normalisierung vor dem Stack: `addscale` (additiv + skalierend), `background` (nur Hintergrund), `median` (Median), `none` | `addscale` |
| **Weighting** | Frame-Gewichtung: `quality` (proportional zum Quality-Score), `uniform` | `quality` |

---

## Bereich: Postprocess

Vier optionale Nachbearbeitungsphasen, die nach dem Stacking auf das gestackte Bild angewendet werden. Alle sind per Default aktiv.

### Astrometry

Loest das WCS (World Coordinate System) des gestackten Bilds via ASTAP. Ergebnis: `artifacts/preprocess/preprocessing_registration.json` mit WCS-Metadaten. Voraussetzung: ASTAP-Binary und lokaler Sternenkatalog (konfiguriert ueber `astrometry.astap_bin` / `astrometry.astap_data_dir` im Parametereditor).

### BGE – Background Gradient Extraction

Extrahiert und subtrahiert Hintergrundgradienten aus dem RGB-Stack. Erzeugt `outputs/stacked_rgb_bge.fits` und `artifacts/preprocess/bge_diagnostics.json`. Nur bei vorhandenem RGB-Stack aktiv.

BGE verwendet die aus Tile Compile bekannte BGE-Konfiguration (`bge.*`) und die seeing-basierte Tile-Geometrie (`tile.*`). Bei OSC/CFA wird die FWHM auf die volle Debayer-Skala bezogen, damit die Sampling-Geometrie mit Tile Compile vergleichbar bleibt. Die effektiven BGE- und Tile-Parameter werden in `bge_diagnostics.json` protokolliert.

### PCC – Photometric Color Calibration

Kalibriert die Farbkanaele des RGB-Stacks photometrisch auf Basis von Sternfarben aus dem WCS. Erfordert erfolgreiche Astrometry. Erzeugt `outputs/stacked_rgb_pcc.fits` und `artifacts/preprocess/pcc_diagnostics.json`.

### HyperMetric Stretch (HMS)

Streckt den linearen RGB-Stack nichtlinear in den visuell darstellbaren Bereich. HMS verwendet den besten verfuegbaren Input: PCC-Output > BGE-Output > linearer RGB-Stack.

**Ausgabe:** `outputs/stacked_rgb_hms.fits`<br>
**Diagnostik:** `artifacts/preprocess/hms_diagnostics.json`

**HMS-Detailparameter** (im Parametereditor unter `hypermetric_stretch.*`):

| Parameter | Beschreibung | Standard |
|-----------|-------------|---------|
| `require_successful_pcc` | HMS nur starten wenn PCC erfolgreich war | `true` |
| `mode` | Betriebsmodus: `ready_to_use` (visuell) oder `scientific` | `ready_to_use` |
| `sensor_profile` | Farbprofil fuer Luminanzgewichte (z. B. `rec709`, `veralux`) | `rec709` |
| `fallback_profile` | Fallback wenn `sensor_profile` nicht gefunden | `rec709` |
| `adaptive_anchor` | Anchor automatisch aus Bildhistogramm bestimmen | `true` |
| `target_bg` | Zielwert fuer den Hintergrund nach Stretch (0–1) | `0.15` |
| `protect_b` | Schutzfaktor fuer den Blaukanal (Farbbalance) | `6.0` |
| `convergence_power` | Konvergenzexponent der Stretch-Kurve | `3.5` |
| `log_d_mode` | Bestimmung des Log-D-Werts: `auto` oder `fixed` | `auto` |
| `color_strategy` | Farbstrategie: `fixed` oder `adaptive` | `fixed` |
| `color_grip` | Staerke der Farbsaettigung (0 = neutral) | `1.0` |
| `output_rgb` | Ausgabedateiname im `outputs/`-Ordner | `stacked_rgb_hms.fits` |

---

## Bereich: Parameters (Parametereditor)

Der Parameters-Bereich zeigt die Preprocessing-Parameter gruppiert nach Backend-Schema und enthaelt darunter den JSON-Parametereditor fuer die vollstaendige effektive Konfiguration. Er ermoeglicht die Bearbeitung aller Parameter, die nicht als eigene Formularfelder sichtbar sind. Bedienung und Struktur entsprechen dem Parameter Studio, aber die gespeicherten Werte gehoeren zur separaten Raw-Stack-Konfiguration.

**Schaltflaechen:**
- **Defaults**: Laedt die gespeicherte Preprocessing-Konfiguration vom Backend (`GET /api/tools/preprocessing/parameters`).
- **Validate**: Sendet die aktuelle JSON-Konfiguration zur Validierung ans Backend und speichert sie fuer weitere Raw-Stack-Starts (`PATCH /api/tools/preprocessing/parameters`).
- **Reset**: Setzt den Editor auf die Backend-Defaults zurueck (`GET /api/tools/preprocessing/defaults`).

**Prioritaet beim Start**: Werte aus den sichtbaren Formularfeldern (Postprocess-Schalter, Quality, Stack, Kalibrierung) ueberschreiben die entsprechenden JSON-Werte im Editor. HMS-Details, BGE/Tile-Parameter, Dark-Matching-Toleranzen, Reportformate und andere erweiterte Parameter kommen ausschliesslich aus dem Editor.

### Uebernahme aus geladener Tile-Compile-YAML

Beim Laden der Raw-Stack-Defaults liest die GUI die aktuell geladene Tile-Compile-Konfiguration und uebernimmt nur Parameter, die im Raw-Stack-Prozess sinnvoll und implementiert sind. Sichtbare Raw-Stack-Felder koennen diese Werte anschliessend wieder ueberschreiben.

| Quelle in Tile Compile | Ziel in Raw Stack | Hinweis |
|------------------------|-------------------|---------|
| `runtime_limits.parallel_workers` | `runtime_limits.parallel_workers` | steuert Parallelisierung |
| `runtime_limits.memory_budget` | `runtime_limits.memory_budget` | steuert speicherschonende Sub-Batches |
| `normalization.mode` | `stacking.normalization` | nur `background`, `median`, `addscale`, `none` |
| `stacking.weighting` | `stacking.weighting` | nur `quality`, `uniform` |
| `stacking.cosmetic_correction` | `stacking.cosmetic_correction` | finale kosmetische Korrektur |
| `stacking.cosmetic_correction_sigma` | `stacking.cosmetic_correction_sigma` | Sigma fuer finale Korrektur |
| `stacking.per_frame_cosmetic_correction` | `stacking.per_frame_cosmetic_correction` | kosmetische Korrektur vor Warp/Stack |
| `stacking.per_frame_cosmetic_correction_sigma` | `stacking.per_frame_cosmetic_correction_sigma` | Sigma fuer Per-Frame-Korrektur |
| `stacking.sigma_clip.sigma_low` | `rejection.low` | Sigma-Rejection unten |
| `stacking.sigma_clip.sigma_high` | `rejection.high` | Sigma-Rejection oben |
| `stacking.sigma_clip.max_iters` | `rejection.max_iters` | iterative Sigma-Clips |
| `stacking.sigma_clip.min_fraction` | `rejection.min_fraction` | Mindestanteil verbleibender Samples |
| `astrometry.*` | `astrometry.*` | inkl. `enabled`, ASTAP-Pfade, Suchradius |
| `astrometry.enabled` | `postprocess.astrometry` | setzt den Postprocess-Schalter vor |
| `bge.*` | `bge.*` | vollstaendige BGE-Konfiguration |
| `bge.enabled` | `postprocess.bge` | setzt den Postprocess-Schalter vor |
| `tile.*` | `tile.*` | Tile-Geometrie fuer BGE-Sampling |
| `pcc.*` | `pcc.*` | vollstaendige PCC-Konfiguration |
| `pcc.enabled` | `postprocess.pcc` | setzt den Postprocess-Schalter vor |
| `hypermetric_stretch.*` | `hypermetric_stretch.*` | vollstaendige HMS-Konfiguration |
| `hypermetric_stretch.enabled` | `postprocess.hypermetric_stretch` | falls vorhanden, setzt den HMS-Schalter vor |

Nicht uebernommen werden Tile-Compile-spezifische Phasenparameter wie Tile Reconstruction, State Clustering, Synthetic Frames, Common Overlap oder normale Run-Studio-Resume-/Template-Metadaten. Raw Stack nutzt davon nur die gemeinsamen Algorithmen und die oben genannten, kompatiblen Parameter.

**Alle Parametergruppen** (sichtbar im Editor und im separaten Parameter-Studio-Tab):

| Gruppe | Inhalt |
|--------|--------|
| `input` | `lights_dir`, `bias_dir`, `darks_dir`, `flats_dir`, `darkflats_dir`, `input_mode`, `raw_formats` |
| `calibration` | `use_bias`, `use_dark`, `use_flat`, Masteroptionen, Dark-Auto-Select, Toleranzen |
| `cfa_mono` | `input_mode`, `bayer_pattern`, `cfa_mode`, `mono_mode` |
| `registration` | `registration_reference` |
| `quality_filter` | `mode`, `min_stars`, `max_fwhm_sigma`, `max_eccentricity`, `min_correlation`, `manual_overrides` |
| `stacking` | `rejection.method`, `rejection.low/high`, `stacking.normalization`, `stacking.weighting` |
| `postprocess` | `astrometry`, `bge`, `pcc`, `hypermetric_stretch` |
| `bge_tile` | `bge.*`, `tile.*` fuer BGE-Sampling und Surface-Fit |
| `hypermetric_stretch` | alle HMS-Detailparameter |
| `report` | `report.detailed`, `report.formats` |
| `runtime_limits` | `runtime_limits.parallel_workers`, `runtime_limits.memory_budget` |

---

## Bereich: Monitor (inline)

Der Inline-Monitor-Bereich in `raw-stack.html` zeigt nach dem Start:

- **Phase-Liste**: Status jeder Phase (ok / skipped / error) als kompakte Chips.
- **Log**: Letzte Events aus `events.jsonl` als scrollbares Terminalfenster.
- **Artefakte**: Links auf generierte Artefakte (CSV, JSON, HTML-Report).
- **Report-Button**: Oeffnet `preprocessing_report.html` direkt im Browser.
- **Run Monitor-Button**: Oeffnet `run-monitor.html?preprocessing_job_id=<id>` fuer die vollstaendige Monitor-Ansicht.

---

## Run Monitor (vollstaendige Ansicht)

Der normale Run Monitor (`run-monitor.html`) erkennt den URL-Parameter `preprocessing_job_id` und wechselt in den Preprocessing-Modus:

- **Phasenliste**: Zeigt alle Preprocessing-Phasen mit Status.
- **Live Log**: Laedt `artifacts/preprocess/events.jsonl` ueber `/api/runs/{run_id}/artifacts/view`.
- **Artefaktliste**: Alle Dateien unter `artifacts/preprocess/` ueber `/api/runs/{run_id}/artifacts`.
- **Artefakt-Viewer**: Klick auf eine Datei oeffnet ihren Inhalt direkt (JSON, CSV, JSONL, HTML).
- **Raw-Serve**: HTML-Reports werden ueber `/api/runs/{run_id}/artifacts/raw/...` direkt als HTML ausgegeben.

> **Wichtig:** Resume-, Revision- und Template-Funktionen des normalen Run-Monitors sind fuer Raw-Stack-Jobs deaktiviert. Raw Stack ist kein resumabler Run, sondern ein eigenstaendiger Tool-Runner.

---

## Phasen-Referenz

| Phase | Beschreibung | Skippbar |
|-------|-------------|----------|
| `INPUT_SCAN` | Frames entdecken, Dimensionen pruefen, Farbmodus bestimmen | Nein |
| `CALIBRATION` | Bias / Dark / Flat anwenden | Ja (wenn kein Kalibrierordner konfiguriert) |
| `CFA_CHANNEL_PREP` | Bayer-Normalisierung fuer OSC; einkanalige Normalisierung fuer Mono | Nein |
| `REFERENCE_SELECTION` | Referenzframe waehlen (beste Qualitaet oder zeitlicher Mittelpunkt) | Nein |
| `REGISTRATION` | Affine Warp-Matrizen via Triangle-Star-Matching berechnen | Nein |
| `QUALITY_ANALYSIS` | Sterne zaehlen, FWHM, Exzentrizitaet, Korrelation, Saettigung messen | Nein |
| `FRAME_FILTERING` | Frames nach Qualitaet ausschliessen (Auto + manuelle Overrides) | Ja (mode=off) |
| `STACKING` | Kalibrierte, registrierte Frames stacken | Nein |
| `ASTROMETRY` | WCS loesen via ASTAP | Ja (wenn deaktiviert oder ASTAP fehlt) |
| `BGE` | Hintergrundgradient extrahieren und subtrahieren | Ja (wenn deaktiviert oder kein RGB-Stack) |
| `PCC` | Photometrische Farbkalibrierung | Ja (wenn deaktiviert oder kein WCS) |
| `HYPERMETRIC_STRETCH` | Nichtlineare Streckung des RGB-Stacks | Ja (wenn deaktiviert oder kein RGB-Stack) |
| `REPORT` | JSON / Markdown / HTML Report erzeugen | Nein |

---

## Artefakte-Referenz

### Diagnostik unter `artifacts/preprocess/`

| Datei | Inhalt |
|-------|--------|
| `effective_config.json` | Effektive Konfiguration des Laufs |
| `frame_quality.csv` | Qualitaetsmetriken pro Frame (Sterne, FWHM, Exzentrizitaet, Korrelation, Status) |
| `rejected_frames.txt` | Liste der ausgeschlossenen Frames mit Ausschlussgrund |
| `stacking_diagnostics.json` | Stacking-Parameter, Gewichte, verwendete Frames |
| `bge_diagnostics.json` | BGE-Ergebnis (Erfolg / Fehler, extrahierter Hintergrundgradient) |
| `pcc_diagnostics.json` | PCC-Ergebnis (Farbkorrekturfaktoren, Anzahl verwendeter Sterne) |
| `hms_diagnostics.json` | HMS-Ergebnis (Anchor, Log-D, Profil, Farbstrategie) |
| `events.jsonl` | Alle Phasen-Events als JSONL (ein JSON-Objekt pro Zeile) |
| `artifacts_manifest.json` | Manifest aller erzeugten Artefakte (Typ, Phase, Pfad) |
| `preprocessing_report.json` | Maschinenlesbarer Gesamtreport |
| `preprocessing_report.md` | Lesbare Markdown-Zusammenfassung |
| `preprocessing_report.html` | HTML-Report mit Phasenuebersicht und Metriken |

### Bildausgaben unter `outputs/`

| Datei | Inhalt |
|-------|--------|
| `stacked_linear.fits` | Linearer gestackter Mono- oder L-Kanal |
| `stacked_rgb.fits` | Linearer RGB-Stack (OSC nach Debayer) |
| `stacked_rgb_bge.fits` | RGB-Stack nach Hintergrundkorrektur (wenn BGE aktiv) |
| `stacked_rgb_pcc.fits` | RGB-Stack nach photometrischer Farbkalibrierung (wenn PCC aktiv) |
| `stacked_rgb_hms.fits` | Gestreckter RGB-Stack (wenn HMS aktiv und erfolgreich) |
| `calibrated/cal_NNNNN.fit` | Kalibrierte Einzelframes (wenn Kalibrierung aktiv) |

---

## Bekannte Grenzen und naechste Schritte

- **Kein Resume**: Raw-Stack-Jobs sind keine resumablen Runs. Es gibt keine Phasen-Resume-Funktion.
- **Keine Tile-Verarbeitung**: Tile-Grid, Tile-Rekonstruktion, Synthetic Frames und State Clustering werden nicht gestartet. Raw Stack erzeugt einen einfachen gestackten Frame, keinen Tile-rekonstruierten.
- **Manuelle Frame-Overrides**: Nach einem Lauf zeigt Raw Stack `frame_quality.csv` als Frame-Tabelle. Checkbox-Aenderungen schreiben `quality_filter.manual_overrides` fuer den naechsten Lauf.
- **Dark-Auto-Select**: Wird bei sehr heterogenen Belichtungszeiten im Darkordner empfohlen. Bei homogenen Darks genuegt ein einfacher Ordner ohne Auto-Select.
- **HMS und PCC**: HMS verwendet standardmaessig `require_successful_pcc: true`. Wenn PCC fehlschlaegt (kein WCS, keine Referenzsterne), wird HMS als `skipped` markiert. Dieser Wert kann im Parametereditor auf `false` gesetzt werden um HMS unabhaengig von PCC zu aktivieren.
