# Release-Notiz: Raw Stack / Preprocessing

## Scope

Raw Stack ergaenzt Tile-Compile GUI3 unter `Tools -> Raw Stack` um einen separaten Preprocessing-Workflow fuer den klassischen Pfad von Lights bis linearem Stack:

```
Input Scan -> Calibration -> CFA/Mono Prep -> Reference Selection
  -> Registration -> Quality Analysis -> Frame Filtering -> Stacking
  -> Astrometry (opt.) -> BGE (opt.) -> PCC (opt.) -> HyperMetric Stretch (opt.) -> Report
```

Der Prozess ist **nicht** Teil des normalen Tile-Compile Run Studios. Er teilt Algorithmen (Registrierung, Stacking, BGE, PCC, HyperMetric Stretch) und Infrastruktur (Run-Monitor, Artefakt-API, Report-Generator), startet aber als separater Tool-Runner und erscheint weder in der normalen Phasenliste noch im Parameter Studio.

---

## Neue Komponenten

### Backend

- `POST /api/tools/preprocessing/run` – Startet einen Raw-Stack-Lauf als Hintergrundjob.
- `GET /api/tools/preprocessing/status?job_id=...` – Liefert Phasenstatus, Artefakte und Metadaten.
- `GET /api/tools/preprocessing/report?job_id=...` – Liefert Pfade zu Report-Artefakten und `run_id`.
- `GET /api/tools/preprocessing/defaults` – Liefert die effektive Standardkonfiguration.
- `GET /api/tools/preprocessing/parameters` – Liefert Parametergruppen fuer den Editor.
- `PATCH /api/tools/preprocessing/parameters` – Validiert und merged Konfigurations-Overrides.
- `POST /api/tools/preprocessing/scan` – Startet einen Input-Scan ohne vollstaendigen Lauf.
- `POST /api/tools/preprocessing/cancel` – Bricht einen laufenden Job ab.

Zusaetzlich werden die bestehenden generischen Run-Artefakt-Endpunkte genutzt:

- `GET /api/runs/{run_id}/artifacts` – Listet alle Artefakte des Runs.
- `GET /api/runs/{run_id}/artifacts/view?path=...` – Laedt Artefakt-Inhalt als Text oder JSON.
- `GET /api/runs/{run_id}/artifacts/raw/<path>` – Liefert Artefakt direkt mit Content-Type (HTML, JSON, Binaer).

### Frontend

- GUI3-Navigation: Im Hauptmenue `Tools` den Untermenuepunkt `Raw Stack` waehlen. `Astrometry` und `PCC` sind weitere Untermenuepunkte derselben Tool-Gruppe.
- Die Seite wird innerhalb der GUI3-Single-Page-Anwendung durch `web_frontend_v3/js/pages/raw-stack.js` gerendert; es gibt keine separate `raw-stack.html` und keinen Sidebar-Eintrag.
- Die Raw-Stack-Ansicht enthaelt Eingabe- und Ausgabeordner, Dateimuster, Kalibrierung, Stack-Methode, Sigma-Grenzen sowie Start- und Abbruchaktionen.
- Der Status eines gestarteten Jobs wird direkt in der Raw-Stack-Ansicht angezeigt und alle zwei Sekunden aktualisiert. Beim erneuten Oeffnen wird ein laufender Job automatisch wieder verbunden.
- Die GUI3-i18n-Anbindung erfolgt ueber `web_frontend_v3/js/i18n/i18n.js`.

### Runner

- `tile_compile_runner preprocess` – Neuer Subcommand fuer den vollstaendigen Preprocessing-Lauf.
- Phasen 1–10 als eigenstaendige Funktionen in `runner_preprocess.cpp`, `runner_phase_preprocess_pipeline.cpp`, `runner_phase_quality_analysis.cpp`.
- String-basierte Phase-Events (kein Tile-Compile-Phase-Enum) fuer saubere Trennung von den normalen Run-Phasen.

---

## Implementierte Phasen

| Phase | Status |
|-------|--------|
| `INPUT_SCAN` | Vollstaendig |
| `CALIBRATION` | Vollstaendig (Bias, Dark mit Auto-Select, Flat) |
| `CFA_CHANNEL_PREP` | Vollstaendig (OSC Bayer, Mono) |
| `REFERENCE_SELECTION` | Vollstaendig |
| `REGISTRATION` | Vollstaendig (Triangle-Star-Matching) |
| `QUALITY_ANALYSIS` | Vollstaendig (Sterne, FWHM, Exzentrizitaet, Korrelation, Saettigung) |
| `FRAME_FILTERING` | Vollstaendig (Auto + Modus-Grenzen) |
| `STACKING` | Vollstaendig (Sigma/Median/Winsor, addscale/background/median/none, quality/uniform) |
| `ASTROMETRY` | Vollstaendig (ASTAP, WCS) |
| `BGE` | Vollstaendig |
| `PCC` | Vollstaendig |
| `HYPERMETRIC_STRETCH` | Vollstaendig (run_hypermetric_stretch_rgb, Diagnostik-JSON) |
| `REPORT` | Vollstaendig (JSON, Markdown, HTML) |

---

## Defaults

| Parameter | Wert |
|-----------|------|
| `input_mode` | `auto` |
| `raw_formats` | `tile_compile` |
| `calibration.dark_auto_select` | `true` |
| `calibration.dark_match_exposure_tolerance_percent` | `8.0` |
| `calibration.dark_match_use_temp` | `false` |
| `quality_filter.mode` | `auto` |
| `quality_filter.min_stars` | `30` |
| `quality_filter.max_fwhm_sigma` | `2.0` |
| `quality_filter.max_eccentricity` | `0.65` |
| `quality_filter.min_correlation` | `0.75` |
| `rejection.method` | `sigma` |
| `rejection.low` / `rejection.high` | `3.0` |
| `stacking.normalization` | `addscale` |
| `stacking.weighting` | `quality` |
| `postprocess.astrometry` | `true` |
| `postprocess.bge` | `true` |
| `postprocess.pcc` | `true` |
| `postprocess.hypermetric_stretch` | `true` |
| `hypermetric_stretch.require_successful_pcc` | `true` |
| `hypermetric_stretch.mode` | `ready_to_use` |
| `hypermetric_stretch.sensor_profile` | `rec709` |
| `hypermetric_stretch.target_bg` | `0.15` |
| `hypermetric_stretch.protect_b` | `6.0` |
| `hypermetric_stretch.convergence_power` | `3.5` |
| `report.detailed` | `true` |

---

## Artefakte

### Diagnostik unter `<run_dir>/artifacts/preprocess/`

| Datei | Beschreibung |
|-------|-------------|
| `effective_config.json` | Effektive Konfiguration des Laufs |
| `frame_quality.csv` | Qualitaetsmetriken pro Frame |
| `rejected_frames.txt` | Ausgeschlossene Frames mit Grund |
| `stacking_diagnostics.json` | Stacking-Parameter und Gewichte |
| `bge_diagnostics.json` | BGE-Ergebnis |
| `pcc_diagnostics.json` | PCC-Ergebnis (Farbkorrekturfaktoren) |
| `hms_diagnostics.json` | HMS-Ergebnis (Anchor, Log-D, Profil) |
| `events.jsonl` | Alle Phasen-Events als JSONL |
| `artifacts_manifest.json` | Manifest aller Artefakte |
| `preprocessing_report.json` | Maschinenlesbarer Gesamtreport |
| `preprocessing_report.md` | Markdown-Zusammenfassung |
| `preprocessing_report.html` | HTML-Report |

### Bildausgaben unter `<run_dir>/outputs/`

| Datei | Beschreibung |
|-------|-------------|
| `stacked_linear.fits` | Linearer Mono- oder L-Stack |
| `stacked_rgb.fits` | Linearer RGB-Stack (OSC) |
| `stacked_rgb_bge.fits` | RGB nach BGE (wenn aktiv) |
| `stacked_rgb_pcc.fits` | RGB nach PCC (wenn aktiv) |
| `stacked_rgb_hms.fits` | Gestreckter RGB-Stack (wenn HMS aktiv) |
| `calibrated/cal_NNNNN.fit` | Kalibrierte Einzelframes (wenn Kalibrierung aktiv) |

---

## Bekannte Grenzen

- **Kein Resume**: Raw-Stack-Jobs sind nicht resumabel. Es gibt keine Phasen-Resume-Funktion.
- **Manuelle Frame-Overrides**: Die Frame-Tabelle aus `frame_quality.csv` kann Frames fuer den naechsten Lauf per `quality_filter.manual_overrides` ein- oder ausschliessen.
- **HMS ohne PCC**: Bei `require_successful_pcc: true` (Standard) wird HMS uebersprungen wenn PCC fehlschlaegt. Kann im Parametereditor auf `false` gesetzt werden.
- **Darkflats**: `darkflats_dir` ist als Konfigurationsfeld vorbereitet, wird aber noch nicht in der gleichen Tiefe wie Bias/Dark/Flat verarbeitet.
