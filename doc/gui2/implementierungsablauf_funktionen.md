# Implementierungsablauf und Funktionsmatrix GUI 2

Ziel: lueckenlose Umsetzung aller GUI-2-Funktionen ohne Funktionsverlust gegenueber `gui_cpp`.

## 1) Abnahmeregeln (verbindlich)

1. Jeder interaktive Control-Eintrag hat:
   - `control_id`
   - `screen`
   - `label_key`
   - `tooltip_key`
   - `action`
   - `backend_binding`
2. Kein Control ohne Tooltip.
3. Kein Start (`Run starten`) bei offenen `error`-Guardrails.
4. Jede Aktion schreibt ein Event in den Run/GUI-Event-Stream.
5. Jede Funktion hat einen manuellen und einen automatisierten Abnahmetest.
6. Jede `parameter.save`-Aktion erzeugt eine neue Config-Revision (append-only, keine Ueberschreibung).
7. Der Run-Ordnername endet immer mit Startdatum/-zeit im Format `YYYYMMDD_HHMMSS`.

## 2) Tooltip-Standard (verbindlich)

Fuer alle Controls gilt:

- Buttons: kurzer Zweck + Nebenwirkung.
- Eingabefelder: Bedeutung + erlaubter Wertebereich + Default.
- Tabellenzeilen/Links: was geoeffnet/gestartet wird.
- Statuschips: aktueller Zustand + Ursprung (z. B. letzte Validierung).

Key-Schema:

- `ui.tooltip.<screen>.<control_id>`
- Beispiele:
  - `ui.tooltip.dashboard.run_start`
  - `ui.tooltip.parameter.registration.star_topk`
  - `ui.tooltip.monitor.stats_generate`

HTML-Clickdummy-Umsetzung:

- Primar ueber `title` oder `data-tooltip` pro Control.
- Zusaetzliche Absicherung ueber `doc/gui2/clickdummy/tooltips.js` (setzt Fallback-Tooltips auf alle interaktiven Elemente).
- Control-Mapping im Dummy ueber `data-control=\"<control_id>\"`.

## 3) Migrationsreihenfolge (ohne Funktionsverlust)

1. **Bestandsfreeze**
   - Bestehende `gui_cpp`-Funktionen je Tab inventarisieren.
   - Jede Alt-Funktion bekommt `legacy_id`.
2. **Control-Registry anlegen**
   - Zentrale Datei `gui2_control_registry.yaml` pflegen.
   - Alle Controls aus Abschnitt 4 mit Status (`TODO/IN_PROGRESS/DONE`) fuehren.
3. **i18n + Tooltip-Layer**
   - `label_key` und `tooltip_key` fuer alle Registry-Eintraege.
4. **State-Model**
   - Zentraler GUI-State: `project`, `scan`, `config`, `config_revisions`, `queue`, `run`, `history`, `tools`, `i18n`.
5. **Screen-Implementierung**
   - Reihenfolge: Dashboard -> Input&Scan -> Parameter Studio -> Assumptions -> Run Monitor -> History+Tools -> Astrometry -> PCC -> Live Log.
   - Referenzseiten: `doc/gui2/clickdummy/*.html` (HTML-only, keine PNG-Referenz notwendig).
6. **Funktionsparitaet**
   - Legacy-Feature-Mapping gegen Registry abhaken.
7. **Abnahmesuite**
   - E2E-Flows + Kontrolllisten aus Abschnitt 8.

## 4) Vollstaendige Funktionsmatrix (Controls)

## 4.1 Global / Shell

| control_id | Typ | Aktion | backend_binding |
|---|---|---|---|
| `nav.start` | Link | zur Startseite | client routing |
| `nav.dashboard` | Link | Dashboard oeffnen | client routing |
| `nav.input_scan` | Link | Input&Scan oeffnen | client routing |
| `nav.parameter_studio` | Link | Parameter Studio oeffnen | client routing |
| `nav.assumptions` | Link | Assumptions oeffnen | client routing |
| `nav.run_monitor` | Link | Run Monitor oeffnen | client routing |
| `nav.history_tools` | Link | History+Tools oeffnen | client routing |
| `nav.astrometry` | Link | Astrometry oeffnen | client routing |
| `nav.pcc` | Link | PCC oeffnen | client routing |
| `nav.live_log` | Link | Live Log oeffnen | client routing |
| `nav.layout_1920` | Link | Layout-Review oeffnen | client routing |
| `nav.flow` | Link | Ablaufkarte oeffnen | client routing |
| `locale.de` | Toggle | Sprache auf DE | i18n store update |
| `locale.en` | Toggle | Sprache auf EN | i18n store update |
| `status.run_ready` | Statuschip | Guardrail-Details zeigen | guardrail summary read |

## 4.2 Input & Scan

| control_id | Typ | Aktion | backend_binding |
|---|---|---|---|
| `input_scan.input_dirs` | Feld/Liste | Input-Ordner setzen (Mehrfachauswahl) | project.scan.inputs |
| `input_scan.pattern` | Feld | Dateimuster setzen | project.scan.pattern |
| `input_scan.max_frames` | Feld | Max. Frame-Anzahl setzen | config.input.max_frames |
| `input_scan.sort` | Select | Sortierreihenfolge setzen | config.input.sort |
| `input_scan.color_mode_confirm` | Select | Farbmodus bestaetigen (OSC/MONO) | project.scan.color_mode |
| `input_scan.bayer_pattern` | Select | Bayer-Pattern explizit setzen (RGGB/GBRG/GRBG/BGGR/auto) | config.data.bayer_pattern |
| `input_scan.with_checksums` | Toggle | Checksummen-Scan aktivieren | project.scan.with_checksums |
| `input_scan.scan_run` | Button | Scan ausfuehren | scan service |
| `input_scan.scan_results` | Readonly-Panel | Scan-Ergebnis anzeigen (Frame-Anzahl, Bayer-Info, Warnungen) | scan.results.read |
| `input_scan.calibration.use_bias` | Toggle | Bias-Kalibrierung aktivieren | config.calibration.use_bias |
| `input_scan.calibration.bias_dir` | Feld/Folder | Bias-Ordner setzen | config.calibration.bias_dir |
| `input_scan.calibration.use_dark` | Toggle | Dark-Kalibrierung aktivieren | config.calibration.use_dark |
| `input_scan.calibration.darks_dir` | Feld/Folder | Dark-Ordner setzen | config.calibration.darks_dir |
| `input_scan.calibration.use_flat` | Toggle | Flat-Kalibrierung aktivieren | config.calibration.use_flat |
| `input_scan.calibration.flats_dir` | Feld/Folder | Flat-Ordner setzen | config.calibration.flats_dir |

## 4.3 Dashboard

| control_id | Typ | Aktion | backend_binding |
|---|---|---|---|
| `dashboard.quick.start_wizard` | Button | New Guided Run starten | wizard init |
| `dashboard.guided.mode_simple` | Toggle | Guided Run auf Einfach stellen | dashboard.guided_mode |
| `dashboard.guided.mode_advanced` | Toggle | Guided Run auf Erweitert stellen | dashboard.guided_mode |
| `dashboard.kpi.scan_quality` | KPI/Link | Scan-Qualitaet-Details oeffnen | scan.quality.summary |
| `dashboard.kpi.open_warnings` | KPI/Link | Warnungsliste oeffnen | guardrail.warning.index |
| `dashboard.input_dirs` | Feld | Input-Ordner setzen | project.scan.inputs |
| `dashboard.color_mode` | Feld/Select | Farbmodus bestaetigen | project.scan.color_mode |
| `dashboard.preset` | Select | Preset waehlen | config preset apply |
| `dashboard.run.runs_dir` | Feld/Folder | Run-Ausgabeordner waehlen | run.runs_dir |
| `dashboard.run.name` | Feld | Freien Run-Namen setzen | run.run_name |
| `dashboard.run.path_preview` | Readonly | Finalen Run-Pfad anzeigen | run.output_dir_preview |
| `dashboard.queue.edit` | Button/Link | MONO-Queue Editor oeffnen | queue editor open |
| `dashboard.queue.row.filter_name` | Feld/Select | MONO-Filter setzen (`L/R/G/B/Ha/...`) | run.filter_queue[].filter_name |
| `dashboard.queue.row.input_dir` | Feld | Input-Ordner je Filter setzen | run.filter_queue[].input_dir |
| `dashboard.queue.row.pattern` | Feld | Optionales Pattern je Filter setzen | run.filter_queue[].pattern |
| `dashboard.queue.row.enabled` | Toggle | Filtereintrag aktiv/inaktiv | run.filter_queue[].enabled |
| `dashboard.queue.row.run_label` | Feld | Optionales Label/Subfolder je Filtereintrag | run.filter_queue[].run_label |
| `dashboard.scan_refresh` | Button | Scan neu | scan service |
| `dashboard.open_parameter_studio` | Button | Studio oeffnen | client routing |
| `dashboard.run_start` | Button | Queue/Run starten | runner start |
| `dashboard.guardrail.scan_ok` | Zeile | zu Scan-Fehlern springen | guardrail deep-link |
| `dashboard.guardrail.color_mode` | Zeile | zu Farbmodus springen | guardrail deep-link |
| `dashboard.guardrail.config_valid` | Zeile | zu Parameterfehlern springen | guardrail deep-link |
| `dashboard.guardrail.calibration_paths` | Zeile | zu Kalibrierpfaden springen | guardrail deep-link |
| `dashboard.guardrail.bge_pcc` | Zeile | zu BGE/PCC-Werten springen | guardrail deep-link |

## 4.4 Parameter Studio

Hinweis: Einzelzeilen wie `parameter.registration.*` sind Kernbeispiele; die Vollabdeckung aller Parameter erfolgt ueber den dynamischen Abschnittseditor `parameter.value.*`.

| control_id | Typ | Aktion | backend_binding |
|---|---|---|---|
| `parameter.search` | Suchfeld | Live-Suche ueber `section.key.subkey`, Enter springt zum ersten Formular-Treffer | parameter index |
| `parameter.search.results` | Panel | Trefferliste unter dem Suchfeld anzeigen (editierbare Treffer inkl. Kategorie) | parameter.search.results |
| `parameter.preset.select` | Select | Preset auswaehlen (alle `examples/*.example.yaml`) | preset catalog read |
| `parameter.preset_apply` | Button | Preset anwenden | config patch apply |
| `parameter.situation_apply` | Button | Szenario anwenden | scenario delta engine |
| `parameter.yaml_sync` | Button | YAML laden/sync | yaml parser/serializer |
| `parameter.validate` | Button | Validierung starten | schema + semantic validator |
| `parameter.category.*` | Liste | Kategorie filtern | local ui state |
| `parameter.full_editor` | Panel | alle Parameter der aktiven Kategorie editierbar rendern | parameter.editor_index |
| `parameter.value.*` | Dynamische Felder | Wert fuer jeden Parameterpfad setzen | config path write |
| `parameter.registration.engine` | Feld | Wert setzen | config path write |
| `parameter.registration.allow_rotation` | Feld | Wert setzen | config path write |
| `parameter.registration.star_topk` | Feld | Wert setzen | config path write |
| `parameter.registration.star_inlier_tol_px` | Feld | Wert setzen | config path write |
| `parameter.registration.reject_cc_min_abs` | Feld | Wert setzen | config path write |
| `parameter.bge.enabled` | Feld | Wert setzen | config path write |
| `parameter.bge.fit_method` | Feld | Wert setzen | config path write |
| `parameter.bge.rbf_lambda` | Feld | Wert setzen | config path write |
| `parameter.pcc.source` | Feld | Wert setzen | config path write |
| `parameter.pcc.sigma_clip` | Feld | Wert setzen | config path write |
| `parameter.pcc.k_max` | Feld | Wert setzen | config path write |
| `parameter.reset_default` | Button | auf Defaults zurueck | config reset |
| `parameter.review_changes` | Button | Diff/Impact anzeigen | diff engine |
| `parameter.save` | Button | speichern | config persist |
| `parameter.explain.info` | Panel | Kurzinfo/Range/Risiko | metadata lookup |
| `parameter.situation.altaz` | Toggle | Delta markieren | scenario engine |
| `parameter.situation.rotation` | Toggle | Delta markieren | scenario engine |
| `parameter.situation.bright_stars` | Toggle | Delta markieren | scenario engine |
| `parameter.situation.few_frames` | Toggle | Delta markieren | scenario engine |
| `parameter.situation.gradient` | Toggle | Delta markieren | scenario engine |
| `parameter.diff_panel` | Panel | YAML-Diff anzeigen | diff engine |

## 4.5 Run Monitor

| control_id | Typ | Aktion | backend_binding |
|---|---|---|---|
| `monitor.stop` | Button | Lauf stoppen | runner stop |
| `monitor.phase.scan_input` | Zeile/Action | Resume ab Phase | runner resume |
| `monitor.phase.channel_split` | Zeile/Action | Resume ab Phase | runner resume |
| `monitor.phase.normalization` | Zeile/Action | Resume ab Phase | runner resume |
| `monitor.phase.global_metrics` | Zeile/Action | Resume ab Phase | runner resume |
| `monitor.phase.tile_grid` | Zeile/Action | Resume ab Phase | runner resume |
| `monitor.phase.registration` | Zeile/Action | Resume ab Phase | runner resume |
| `monitor.phase.prewarp` | Zeile/Action | Resume ab Phase | runner resume |
| `monitor.phase.common_overlap` | Zeile/Action | Resume ab Phase | runner resume |
| `monitor.phase.local_metrics` | Zeile/Action | Resume ab Phase | runner resume |
| `monitor.phase.tile_reconstruction` | Zeile/Action | Resume ab Phase | runner resume |
| `monitor.phase.state_clustering` | Zeile/Action | Resume ab Phase | runner resume |
| `monitor.phase.synthetic_frames` | Zeile/Action | Resume ab Phase | runner resume |
| `monitor.phase.stacking` | Zeile/Action | Resume ab Phase | runner resume |
| `monitor.phase.debayer` | Zeile/Action | Resume ab Phase | runner resume |
| `monitor.phase.astrometry` | Zeile/Action | Resume ab Phase | runner resume |
| `monitor.phase.bge` | Zeile/Action | Resume ab Phase | runner resume |
| `monitor.phase.pcc` | Zeile/Action | Resume ab Phase | runner resume |
| `monitor.phase.progress_pct` | Readonly | Prozentfortschritt je Phase anzeigen | run.phase_progress_map |
| `monitor.filter.L` | Chip/Action | Filterkontext setzen | queue state |
| `monitor.filter.R` | Chip/Action | Filterkontext setzen | queue state |
| `monitor.filter.G` | Chip/Action | Filterkontext setzen | queue state |
| `monitor.filter.B` | Chip/Action | Filterkontext setzen | queue state |
| `monitor.filter.Ha` | Chip/Action | Filterkontext setzen | queue state |
| `monitor.filter.OIII` | Chip/Action | Filterkontext setzen | queue state |
| `monitor.filter.SII` | Chip/Action | Filterkontext setzen | queue state |
| `monitor.resume` | Button | Resume starten (erst aktiv nach Phase-Klick) | runner resume |
| `monitor.resume.config_revision` | Select | Config-Revision fuer Resume waehlen | run.resume.config_revision |
| `monitor.resume.restore_revision` | Button | Aeltere Config-Revision wiederherstellen | config.revision.restore |
| `monitor.report` | Button | Reportansicht | report open |
| `monitor.open_run_folder` | Button | Run-Ordner | file-open action |
| `monitor.stats.generate` | Button | Stats erstellen | runner.stats.generate_report (Pfad wird zur Laufzeit aus Konfiguration aufgeloest) |
| `monitor.stats.open_folder` | Button | Stats-Ordner oeffnen | file-open action |

## 4.6 History + Tools

| control_id | Typ | Aktion | backend_binding |
|---|---|---|---|
| `history.row_select` | Tabellenzeile | Run als aktiv waehlen | history state |
| `history.refresh` | Button | Historie aktualisieren | run index refresh |
| `history.set_current` | Button | als Current Run setzen | current run pointer |
| `history.open_report` | Button | Report oeffnen | report open |
| `nav.astrometry` | Link | Astrometry-Toolseite oeffnen | client routing |
| `nav.pcc` | Link | PCC-Toolseite oeffnen | client routing |

## 4.7 Astrometry

| control_id | Typ | Aktion | backend_binding |
|---|---|---|---|
| `tools.astrometry.binary` | Feld | ASTAP-Binary setzen | tools config write |
| `tools.astrometry.data_dir` | Feld | ASTAP-Datenverzeichnis setzen | tools config write |
| `tools.astrometry.detect` | Button | ASTAP-Status pruefen | astrometry.setup.detect |
| `tools.astrometry.install_cli` | Button | ASTAP CLI herunterladen/installieren | astrometry.setup.install_cli |
| `tools.astrometry.catalog` | Feld | Katalog setzen | tools config write |
| `tools.astrometry.download_catalog` | Button | ASTAP-Katalog laden | astrometry.catalog.download |
| `tools.astrometry.cancel_download` | Button | Download abbrechen | astrometry.catalog.cancel |
| `tools.astrometry.file` | Feld | Solve-Datei setzen | tools config write |
| `tools.astrometry.browse_file` | Button | Solve-Datei waehlen | fs.pick.file |
| `tools.astrometry.browse_binary` | Button | ASTAP-CLI-Datei waehlen | fs.pick.file |
| `tools.astrometry.browse_data_dir` | Button | ASTAP-Datenverzeichnis waehlen | fs.pick.dir |
| `tools.astrometry.solve` | Button | Plate Solve starten | astrometry runner |
| `tools.astrometry.save_solved` | Button | FITS mit WCS speichern | astrometry.save_solved |

## 4.8 PCC

| control_id | Typ | Aktion | backend_binding |
|---|---|---|---|
| `tools.pcc.rgb_fits` | Feld | RGB-FITS setzen | tools config write |
| `tools.pcc.wcs_file` | Feld | WCS-Datei setzen | tools config write |
| `tools.pcc.browse_rgb` | Button | RGB-Datei waehlen | fs.pick.file |
| `tools.pcc.browse_wcs` | Button | WCS-Datei waehlen | fs.pick.file |
| `tools.pcc.source` | Feld | PCC source setzen | tools config write |
| `tools.pcc.sigma` | Feld | PCC sigma setzen | tools config write |
| `tools.pcc.min_stars` | Feld | PCC min stars setzen | tools config write |
| `tools.pcc.siril_catalog_dir` | Feld | Siril-Katalogordner setzen | tools config write |
| `tools.pcc.browse_catalog_dir` | Button | Siril-Katalogverzeichnis waehlen | fs.pick.dir |
| `tools.pcc.download_missing` | Button | fehlende Siril-Chunks laden | pcc.catalog.download_missing |
| `tools.pcc.cancel_download` | Button | laufenden Download abbrechen | pcc.catalog.cancel |
| `tools.pcc.check_online` | Button | Online-Quelle pruefen | pcc.catalog.check_online |
| `tools.pcc.run` | Button | PCC Quicktest starten | pcc runner |
| `tools.pcc.save_corrected` | Button | korrigiertes Ergebnis speichern | file writer |

## 4.9 Assumptions

| control_id | Typ | Aktion | backend_binding |
|---|---|---|---|
| `assumptions.pipeline_profile` | Select | Pipeline-Profil setzen (`strict`/`practical`) | config.assumptions.pipeline_profile |
| `assumptions.frames_min` | Feld | Mindestanzahl Frames | config.assumptions.frames_min |
| `assumptions.frames_optimal` | Feld | Optimale Frame-Anzahl | config.assumptions.frames_optimal |
| `assumptions.frames_reduced_threshold` | Feld | Schwelle fuer Reduced-Mode | config.assumptions.frames_reduced_threshold |
| `assumptions.reduced_mode_skip_clustering` | Toggle | Clustering im Reduced-Mode ueberspringen | config.assumptions.reduced_mode_skip_clustering |
| `assumptions.reduced_mode_cluster_range` | Feld | Min/Max Cluster im Reduced-Mode | config.assumptions.reduced_mode_cluster_range |
| `assumptions.exposure_time_tolerance_percent` | Feld | Belichtungszeit-Toleranz in Prozent | config.assumptions.exposure_time_tolerance_percent |
| `assumptions.pipeline_mode_info` | Info-Panel | Aktuellen Modus anzeigen (Full/Reduced/Emergency) | config.assumptions.read |

## 4.10 Wizard (Guided Run)

| control_id | Typ | Aktion | backend_binding |
|---|---|---|---|
| `wizard.input.input_dirs` | Feld/Liste | Input-Ordner setzen | wizard.draft.inputs |
| `wizard.input.pattern` | Feld | Dateimuster setzen | wizard.draft.pattern |
| `wizard.input.color_mode` | Select | Farbmodus setzen | wizard.draft.color_mode |
| `wizard.input.runs_dir` | Feld/Folder | Ausgabeordner setzen | wizard.draft.runs_dir |
| `wizard.input.run_name` | Feld | Run-Name setzen | wizard.draft.run_name |
| `wizard.calibration.use_bias` | Toggle | Bias aktivieren | wizard.draft.calibration.use_bias |
| `wizard.calibration.bias_dir` | Feld/Folder | Bias-Ordner setzen | wizard.draft.calibration.bias_dir |
| `wizard.calibration.use_dark` | Toggle | Dark aktivieren | wizard.draft.calibration.use_dark |
| `wizard.calibration.darks_dir` | Feld/Folder | Dark-Ordner setzen | wizard.draft.calibration.darks_dir |
| `wizard.calibration.use_flat` | Toggle | Flat aktivieren | wizard.draft.calibration.use_flat |
| `wizard.calibration.flats_dir` | Feld/Folder | Flat-Ordner setzen | wizard.draft.calibration.flats_dir |
| `wizard.queue.row.filter_name` | Select | MONO-Filter setzen | wizard.draft.filter_queue[].filter_name |
| `wizard.queue.row.input_dir` | Feld | Input-Ordner je Filter | wizard.draft.filter_queue[].input_dir |
| `wizard.preset.select` | Select | Preset waehlen | wizard.draft.preset_id |
| `wizard.situation.apply` | Button | Situationsdeltas anwenden | wizard.draft.scenarios |
| `wizard.validation.result` | Readonly-Panel | Validierungsergebnis anzeigen | validator.result.read |
| `wizard.start` | Button | Run starten | runner.start |
| `wizard.nav.back` | Button | Vorheriger Schritt | wizard.state.step |
| `wizard.nav.next` | Button | Naechster Schritt | wizard.state.step |

## 4.9 Layout/Flow (Dokuseiten)

| control_id | Typ | Aktion | backend_binding |
|---|---|---|---|
| `layout.hotspot.shell` | Hotspot | Shell-Mapping pruefen | doc navigation |
| `layout.hotspot.main_wrapper` | Hotspot | Wrapper-Mapping pruefen | doc navigation |
| `flow.node.*` | Hotspot | zur Zielseite springen | doc navigation |

## 5) New Guided Run: detaillierter Implementierungsablauf

## 5.1 Benutzerablauf

1. Nutzer klickt `dashboard.quick.start_wizard`.
2. Wizard Schritt `Input`:
   - Input-Ordner erfassen.
   - Pattern setzen.
   - Farbmodus bestaetigen.
   - `runs_dir` waehlen.
   - `run_name` frei setzen.
3. Wizard Schritt `Calibration`:
   - Bias/Dark/Flat Pfade setzen.
4. Wizard Schritt `Queue`:
   - MONO-Filtereintraege erfassen (`filter_name`, `input_dir`, optional `run_label`, `pattern`, `enabled`).
   - Reihenfolge pruefen.
5. Wizard Schritt `Preset + Situation`:
   - Preset waehlen.
   - Szenario-Deltas anwenden (Alt/Az, Rotation, helle Sterne, ...).
6. Wizard Schritt `Validation`:
   - Schema + Semantik + Guardrails.
   - Fehlerliste mit Deep-Links.
7. Wizard Schritt `Review + Start`:
   - YAML-Diff und Startparameter anzeigen.
   - Finalen Ausgabeordner als Preview anzeigen (`<run_name>_<YYYYMMDD_HHMMSS>`).
   - Start bestaetigen.
8. Run Monitor oeffnen.

## 5.2 Technische Pipeline

1. `wizard_init(project_id)`
2. `scan_preview(inputs, pattern)`
3. `build_queue(entries[])`
4. `apply_preset(preset_id)`
5. `apply_scenarios(scenarios[])`
6. `validate_config(config)`
7. `persist_config_revision(config, run_context)`
8. `resolve_run_output_dir(runs_dir, run_name, start_timestamp)`
9. `start_serial_queue(queue, config_revision)`
10. `subscribe_run_events(run_id)`

## 5.3 Datenobjekte

- `GuidedRunDraft`
  - `inputs[]`
  - `calibration`
  - `color_mode`
  - `runs_dir`
  - `run_name`
  - `filter_queue[]`
  - `preset_id`
  - `scenarios[]`
  - `config_patch`
- `RunStartPayload`
  - `config_path`
  - `config_revision`
  - `filter_queue`
  - `resume_policy`
  - `output_dir_name`
  - `working_dir`

## 5.4 Fehlerfaelle (muss implementiert werden)

1. Input-Ordner leer/nicht lesbar.
2. Queue leer oder alle Eintraege `enabled=false`.
3. Konflikt zwischen Preset und manuellen Overrides.
4. Validierung `error` blockiert Start.
5. Runner-Startfehler (Prozessstart).
6. Scriptfehler bei `Generate Stats`.
7. `run.runs_dir` nicht vorhanden oder nicht schreibbar.
8. `run.run_name` ungueltig (leerer/unerlaubter Dateiname).
9. Ausgewaehlte Config-Revision fuer Resume nicht mehr aufloesbar.

## 6) Stats-Button Implementierung (Run Monitor unter Live Log)

## 6.1 Trigger-Regel

- Wenn `python3` verfuegbar:
  - Kommando:
    - Kommando wird zur Laufzeit aufgeloest (siehe Detailkonzept §4.5: `stats_script_path` aus gui2.json, Env-Var `TILE_COMPILE_STATS_SCRIPT`, oder relativ zur Binary)
- Wenn `python3` fehlt:
  - UI zeigt klaren Fehler + Install-Hinweis.

## 6.2 Aufrufe

1. Button `monitor.stats.generate`:
   - startet Stats fuer aktuell selektierten Run.

## 6.3 Ergebnisdarstellung

- Status im Toolpanel:
  - `pending`, `running`, `ok`, `error`
- Bei `ok`:
  - Button `monitor.stats.open_folder` aktiv.
  - Link auf erzeugte Artefakte.

## 7) Paritaets-Checkliste gegen alte Tabs

| Legacy-Bereich (`gui_cpp`) | GUI2-Ziel | Statusfeld fuer Umsetzung |
|---|---|---|
| Scan | Dashboard + Input&Scan | `TODO/IN_PROGRESS/DONE` |
| Configuration | Parameter Studio | `TODO/IN_PROGRESS/DONE` |
| Assumptions | Assumptions | `TODO/IN_PROGRESS/DONE` |
| Run | Dashboard/Run Monitor | `TODO/IN_PROGRESS/DONE` |
| Pipeline Progress | Run Monitor | `TODO/IN_PROGRESS/DONE` |
| Current run | Run Monitor | `TODO/IN_PROGRESS/DONE` |
| Run history | History+Tools | `TODO/IN_PROGRESS/DONE` |
| Astrometry | Astrometry | `TODO/IN_PROGRESS/DONE` |
| PCC | PCC | `TODO/IN_PROGRESS/DONE` |
| Live log | Run Monitor/Live Log | `TODO/IN_PROGRESS/DONE` |

## 8) Abnahmetests (keine Funktion vergessen)

1. **Control Coverage Test**
   - Jede Zeile aus Abschnitt 4 einmal ausfuehren.
2. **Tooltip Coverage Test**
   - Fuer jede `control_id` existiert `tooltip_key`.
3. **Guided Run E2E**
   - Start Wizard bis Run Monitor ohne manuelle YAML-Edits.
4. **MONO Queue E2E**
   - Mindestens 3 Filter, strikt serieller Ablauf.
5. **Resume E2E**
   - Resume ab Phase und Filter.
   - Resume mit expliziter Config-Revision.
6. **Config-Revisionshistorie E2E**
   - Nach mehreren Saves sind alte Revisionen intakt und wiederherstellbar.
7. **Run-Name/Output-Pfad E2E**
   - Frei gesetzter Run-Name + automatisches Datums-Suffix im Zielordner.
8. **Stats E2E**
   - Generate-Button im Run Monitor + Open-Stats-Folder.
9. **i18n E2E**
   - DE/EN fuer Labels + Tooltips + Warnungen.
10. **Parity Audit**
   - Legacy-Tabelle vollstaendig auf `DONE`.
11. **Spacing Gate**
   - `python3 doc/gui2/scripts/check_layout_1920_spacing.py` liefert `Result: OK`.
12. **Theme/Font Gate**
   - Themes funktionieren ueber `?theme=observatory|slate|sand`.
   - Lokale Fonts aus `clickdummy/assets/fonts` werden auf allen Ziel-OS geladen.

## 9) Detaillierte Ablaeufe (Control-gebunden)

## 9.1 Dashboard-KPI und Readiness

1. `dashboard.kpi.scan_quality` zeigt Score `0..1` plus Trend.
2. Klick auf `dashboard.kpi.scan_quality` oeffnet Ursachenliste (Sterne/Gradient/SNR).
3. `dashboard.kpi.open_warnings` zeigt Anzahl nicht-blockierender Warnungen.
4. Klick auf `dashboard.kpi.open_warnings` oeffnet Warnungsliste mit Deep-Links.
5. `dashboard.guardrail.*`-Zeilen springen direkt in den betroffenen Screenbereich.
6. `dashboard.run_start` bleibt gesperrt, solange mindestens ein `error`-Guardrail offen ist.

## 9.2 MONO Queue statt OSC Single-Input

1. `dashboard.color_mode` auf MONO stellen.
2. `dashboard.queue.edit` oeffnen.
3. Pro Filter Eintrag setzen:
   - `dashboard.queue.row.filter_name`
   - `dashboard.queue.row.input_dir`
   - optional `dashboard.queue.row.pattern`
   - optional `dashboard.queue.row.enabled=false` fuer Skip
4. Reihenfolge validieren (`L->R->G->B->Ha...`).
5. `dashboard.run_start` startet serielle Abarbeitung als `run.filter_queue[]`.
6. `monitor.filter.*` und Batch-Leiste zeigen `Filter i/N`.

## 9.3 Parameteraenderung -> Speichern -> Resume per Phase-Klick

1. In `parameter-studio` Werte aendern:
   - z. B. `parameter.registration.star_topk`, `parameter.bge.rbf_lambda`, `parameter.pcc.k_max`.
2. `parameter.validate` ausfuehren:
   - bei `error` kein Resume erlaubt.
   - bei `warn` Resume erlaubt, Warnung im Kontext anzeigen.
3. `parameter.save` persistiert die Aenderung als neue Config-Version fuer den aktiven Run.
4. Nach `run-monitor` wechseln und optional `monitor.filter.*` setzen.
5. Gewuenschte Revision in `monitor.resume.config_revision` waehlen.
6. Gewuenschte Phase per Klick waehlen (`monitor.phase.*`); erst danach wird `monitor.resume` aktiv.
7. Bei Bedarf alte Konfiguration via `monitor.resume.restore_revision` wiederherstellen.
8. Alte Revisionen bleiben immer erhalten (append-only Historie).
9. Runner-Aufruf enthaelt:
   - `run_dir`
   - `filter_context`
   - `from_phase`
   - `config_revision` (gewaehlte Version)
10. Run-Monitor zeigt bestaetigten Resume-Kontext:
   - `Resume: Filter R (2/5) ab BGE mit Config rev <id>`.

## 9.4 Stats, Astrometry und PCC aufgetrennt

1. In `history-tools` bleibt Historie + Deep-Link auf Tools.
2. `astrometry` ist eigene Seite:
   - ASTAP CLI installieren/pruefen
   - Katalog downloaden/abbrechen
   - Solve + Save Solved
3. `pcc` ist eigene Seite:
   - Siril- oder VizieR-Quelle waehlen
   - fehlende Siril-Chunks downloaden/abbrechen
   - PCC run/save corrected
4. In `run-monitor` liegt Stats:
   - `monitor.stats.generate`
   - `monitor.stats.open_folder`
5. `monitor.stats.generate` nutzt den aktiven/selektierten Run-Kontext.
6. Ergebnisartefakte erscheinen im Run-Ordner und Artefaktpanel.

## 10) Implementierungsstrategie (Phasen und Workpackages)

## 10.1 Phase A - Fundament

1. Zentrales State-Model implementieren (`project`, `scan`, `config`, `config_revisions`, `queue`, `run`, `history`, `tools`, `i18n`).
2. Event-Stream etablieren (jede Aktion schreibt `ui_event` + optional `run_event`).
3. Guardrail-Service und Warning-Index als API bereitstellen.

## 10.2 Phase B - Dashboard + Queue

1. KPI-Aggregator fuer `scan_quality` und `open_warnings` implementieren.
2. Queue-Editor fuer `run.filter_queue[]` bauen.
3. Start-Gating (`error` blockiert Start) hart verdrahten.
4. Run-Zielpfad:
   - `run.runs_dir` validieren (existiert/schreibbar).
   - `run.run_name` pruefen/sanitisieren.
   - Zielname immer `<run_name>_<YYYYMMDD_HHMMSS>`.
5. Tests:
   - Queue leer -> Start blockiert.
   - MONO mit 3+ Filtern -> serieller Laufstart korrekt.

## 10.3 Phase C - Parameter Studio

1. Parameterindex + Suche + Kategorien.
2. Bidirektionale YAML-Synchronisierung.
3. Situation Assistant + Explain-Metadaten.
4. Tests:
   - Schema- und Semantikvalidierung.
   - i18n-String-Abdeckung DE/EN.

## 10.4 Phase D - Run Monitor + Stats

1. Phasenliste, Filterkontext, Resume-Flow.
2. Stats-Panel unter Live Log (`monitor.stats.*`).
3. Command-Execution-Adapter fuer `generate_report.py`.
4. Tests:
   - Stats bei fehlendem Python -> klarer Fehler.
   - Stats bei vorhandenem Kontext -> Artefakte erzeugt.
   - Parameteraenderung + Resume ab Phase nutzt neue Config-Version.
   - Restore auf alte Config-Version und erneutes Resume funktioniert.

## 10.5 Phase E - History + Astrometry/PCC

1. Historientabelle mit Selektion/Refresh/Report.
2. Astrometry-Screen (Setup, Download, Solve, Save).
3. PCC-Screen (Siril/VizieR, Download, Run, Save).
4. Tests:
   - ASTAP CLI/Katalog Downloadpfade.
   - PCC Quicktest und Save-Corrected.
   - Uebergang History <-> Astrometry/PCC <-> Run Monitor stabil.

## 10.6 Phase F - Hardening und Abnahme

1. Vollstaendiger Control-Coverage-Test gegen Abschnitt 4.
2. Spacing/Theme/Font Gates.
3. Legacy-Parity-Tabelle auf `DONE`.
4. Finales Dokumentations-Review (`detailkonzept.md` <-> diese Datei).

## 11) Backend-Implementierung (FastAPI)

## 11.1 Zielbild

1. GUI2-Frontend spricht ausschliesslich mit FastAPI (`REST + WebSocket`).
2. FastAPI kapselt alle Aufrufe von `tile_compile_cli` und `tile_compile_runner`.
3. Keine direkte Prozesssteuerung aus dem Browser.
4. Jeder mutierende API-Call schreibt ein `ui_event` (Audit + Replay).

## 11.2 Service-Aufteilung im Backend

1. `ConfigService`
   - Schema lesen, YAML validieren, Preset laden, Config-Revisionsverwaltung.
2. `ScanService`
   - Input-Scan, Guardrail-Readiness, KPI-Berechnung (`scan_quality`, `warnings`).
3. `RunService`
   - Run-Start (OSC/MONO Queue), Resume, Stop, Status-Polling.
4. `HistoryService`
   - Runs auflisten, Current-Run setzen, Artefakte/Reports lesen.
5. `AstrometryService`
   - ASTAP detect/install, Katalogdownload/cancel, solve/save_solved.
6. `PCCService`
   - Siril-Status, Missing-Chunk Download/cancel, Online-Check, run/save_corrected.
7. `StatsService`
   - `generate_report.py` starten, Status und Output-Ordner bereitstellen.
8. `StreamService`
   - WebSocket fuer Live-Log, Phasenfortschritt, Queue-Status, Job-Status.

## 11.3 API-Ablauf (End-to-End)

## 11.3.1 Dashboard Schnellstart

1. Frontend laedt `GET /api/app/state`.
2. Frontend laedt `GET /api/guardrails` und `GET /api/scan/quality`.
3. Bei `POST /api/runs/start`:
   - Backend validiert Guardrails.
   - Backend validiert `runs_dir`/`run_name`.
   - Backend erzeugt Config-Revision.
   - Backend startet Runner-Prozess.
   - Frontend subscribed `WS /api/ws/runs/{run_id}`.

## 11.3.2 Resume per Phase

1. Phase-Klick setzt lokal `selected_phase`.
2. `monitor.resume` erst aktiv bei gesetzter Phase.
3. Frontend sendet `POST /api/runs/{run_id}/resume` mit:
   - `from_phase`
   - `config_revision_id`
   - `filter_context` (bei MONO Queue)
4. Backend prueft Revisionsdatei + Run-Kontext.
5. Backend startet `tile_compile_runner resume ...`.
6. Status/Logs laufen via WebSocket in Run Monitor.

## 11.3.3 Astrometry Tool-Flow

1. `POST /api/tools/astrometry/detect`.
2. Optional `POST /api/tools/astrometry/install-cli`.
3. Optional `POST /api/tools/astrometry/catalog/download` + `.../cancel`.
4. `POST /api/tools/astrometry/solve`.
5. Optional `POST /api/tools/astrometry/save-solved`.

## 11.3.4 PCC Tool-Flow

1. `GET /api/tools/pcc/siril/status`.
2. Optional `POST /api/tools/pcc/siril/download-missing` + `.../cancel`.
3. Optional `POST /api/tools/pcc/check-online`.
4. `POST /api/tools/pcc/run`.
5. Optional `POST /api/tools/pcc/save-corrected`.

## 11.4 Job-/Prozessmodell

1. Lange Tasks laufen als Backend-Jobs (nicht als blockierende HTTP-Requests).
2. Jeder Job hat:
   - `job_id`
   - `type`
   - `state` (`pending|running|ok|error|cancelled`)
   - `started_at`, `ended_at`
   - `run_id` (optional)
3. HTTP startet Job und liefert `202 Accepted` + `job_id`.
4. Fortschritt via:
   - `WS /api/ws/jobs/{job_id}` oder
   - `GET /api/jobs/{job_id}`.
5. Cancel ueber `POST /api/jobs/{job_id}/cancel`.

## 11.5 Fehler- und Sicherheitsregeln

1. Prozessaufrufe nur mit Whitelist-Kommandos + validierten Argumenten.
2. Keine Shell-Konkatenation mit untrusted Input.
3. Dateisystemzugriffe auf erlaubte Root-Pfade begrenzen.
4. Einheitliches Fehlerformat:
   - `code`
   - `message`
   - `hint`
   - `details` (optional)
5. Katalogdownloads sind cancellable und idempotent.

## 11.6 Teststrategie Backend

1. API-Contract-Tests (Request/Response je Endpoint).
2. Prozessadapter-Mocks fuer CLI/Runner.
3. Integrations-Tests mit echten Binaries fuer:
   - Scan
   - Validate
   - Run/Resume
   - Stats
4. Tool-Tests fuer Astrometry/PCC:
   - detect/install/download/cancel
   - run/save
5. WebSocket-Tests fuer Phase-Progress und Live-Log.

## 11.7 Vollstaendiger API-Vertrag (FastAPI v1)

Basis:

1. Prefix: `/api`
2. Content-Type Request/Response: `application/json`
3. Zeitformat: ISO-8601 UTC (`2026-03-07T22:16:08Z`)
4. IDs: `run_id`, `job_id`, `revision_id` als Strings

## 11.7.1 Einheitliches Fehlerformat

Alle Fehlerantworten (`4xx/5xx`) liefern:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "config contains invalid values",
    "hint": "check field pcc.sigma_clip",
    "details": {"path": "pcc.sigma_clip"}
  }
}
```

## 11.7.2 REST Endpunkte (Core)

| HTTP | Endpoint | Zweck | Response |
|---|---|---|---|
| `GET` | `/api/version` | Backend/CLI/Runner Versionen | `{api, backend, cli, runner}` |
| `GET` | `/api/app/state` | Initialer UI-State | `AppState` |
| `GET` | `/api/app/constants` | Enum/Phasen/Defaults | `AppConstants` |
| `GET` | `/api/config/schema` | Schema lesen | JSON-Schema |
| `GET` | `/api/config/current` | Aktuelle Config | `{config, source}` |
| `POST` | `/api/config/validate` | Config validieren | `{ok, errors[], warnings[]}` |
| `POST` | `/api/config/save` | Config speichern + Revision erzeugen | `{revision_id, path}` |
| `GET` | `/api/config/presets` | Preset-Katalog | `{items[]}` |
| `POST` | `/api/config/presets/apply` | Preset anwenden | `{config, applied_paths[]}` |
| `GET` | `/api/config/revisions` | Revisionsliste | `{items[]}` |
| `POST` | `/api/config/revisions/{revision_id}/restore` | Revision aktiv setzen | `{ok, active_revision_id}` |
| `POST` | `/api/scan` | Input-Scan starten | `{job_id}` |
| `GET` | `/api/scan/quality` | KPI Scan-Qualitaet | `{score, factors[]}` |
| `GET` | `/api/guardrails` | Readiness-Zustand | `{status, checks[]}` |
| `GET` | `/api/runs` | Runs listen | `{items[], total}` |
| `GET` | `/api/runs/{run_id}/status` | Runstatus + Phasen | `RunStatus` |
| `GET` | `/api/runs/{run_id}/logs` | Logauszug | `{lines[], cursor}` |
| `GET` | `/api/runs/{run_id}/artifacts` | Artefaktliste | `{items[]}` |
| `POST` | `/api/runs/start` | Run/Queue starten | `{run_id, job_id}` |
| `POST` | `/api/runs/{run_id}/resume` | Resume ab Phase | `{run_id, job_id}` |
| `POST` | `/api/runs/{run_id}/stop` | Lauf stoppen | `{ok}` |
| `POST` | `/api/runs/{run_id}/set-current` | Current Run setzen | `{ok, run_id}` |
| `POST` | `/api/runs/{run_id}/stats` | Stats-Report erzeugen | `{job_id}` |
| `GET` | `/api/runs/{run_id}/stats/status` | Stats-Status | `{state, output_dir, report_path}` |

## 11.7.3 REST Endpunkte (Astrometry/PCC)

| HTTP | Endpoint | Zweck | Response |
|---|---|---|---|
| `POST` | `/api/tools/astrometry/detect` | ASTAP detect | `{installed, binary, data_dir}` |
| `POST` | `/api/tools/astrometry/install-cli` | ASTAP CLI installieren | `{job_id}` |
| `POST` | `/api/tools/astrometry/catalog/download` | ASTAP-Katalog downloaden | `{job_id}` |
| `POST` | `/api/tools/astrometry/catalog/cancel` | ASTAP-Download abbrechen | `{ok}` |
| `POST` | `/api/tools/astrometry/solve` | Plate-Solve starten | `{job_id}` |
| `POST` | `/api/tools/astrometry/save-solved` | FITS mit WCS speichern | `{output_path}` |
| `GET` | `/api/tools/pcc/siril/status` | Siril-Chunkstatus | `{installed, total, missing[]}` |
| `POST` | `/api/tools/pcc/siril/download-missing` | Missing Chunks laden | `{job_id}` |
| `POST` | `/api/tools/pcc/siril/cancel` | PCC-Download abbrechen | `{ok}` |
| `POST` | `/api/tools/pcc/check-online` | VizieR Reachability pruefen | `{ok, latency_ms}` |
| `POST` | `/api/tools/pcc/run` | PCC Run | `{job_id}` |
| `POST` | `/api/tools/pcc/save-corrected` | PCC Result speichern | `{output_rgb, output_channels[]}` |

## 11.7.4 REST Endpunkte (Jobs)

| HTTP | Endpoint | Zweck | Response |
|---|---|---|---|
| `GET` | `/api/jobs/{job_id}` | Jobstatus lesen | `JobStatus` |
| `GET` | `/api/jobs` | Jobs filtern/listen | `{items[]}` |
| `POST` | `/api/jobs/{job_id}/cancel` | Job abbrechen | `{ok}` |

## 11.7.5 WebSocket Endpunkte

| WS | Zweck |
|---|---|
| `/api/ws/runs/{run_id}` | Phase/Lauf/Queue/Log Events fuer Run Monitor |
| `/api/ws/jobs/{job_id}` | Feingranulare Job-Events fuer Tool- und Downloadjobs |
| `/api/ws/system` | optionale globale Events (health/version/warnings) |

## 11.7.6 Event-Vertrag Run-Stream

Pflicht-Events:

1. `phase_start`
2. `phase_progress`
3. `phase_end`
4. `run_end`
5. `queue_progress` (MONO)
6. `log_line`

Beispiele:

```json
{"type":"phase_start","run_id":"20260307_221530_9f2a","filter":"R","phase":"NORMALIZATION","phase_index":3,"phase_count":17,"ts":"2026-03-07T22:16:04Z"}
```

```json
{"type":"phase_progress","run_id":"20260307_221530_9f2a","filter":"R","phase":"NORMALIZATION","current":842,"total":1264,"pct":66.6,"eta_s":501,"ts":"2026-03-07T22:16:08Z"}
```

```json
{"type":"phase_end","run_id":"20260307_221530_9f2a","filter":"R","phase":"NORMALIZATION","status":"ok","duration_ms":183245,"artifacts":["/runs/.../normalization_report.json"],"ts":"2026-03-07T22:19:07Z"}
```

```json
{"type":"run_end","run_id":"20260307_221530_9f2a","status":"ok","duration_ms":2156345,"ts":"2026-03-07T22:52:11Z"}
```

FE-Regel:

1. `monitor.resume` bleibt disabled, bis `selected_phase` gesetzt ist.
2. Bei WS-Disconnect: `GET /api/runs/{run_id}/status` fuer Resync.

## 11.7.7 Wichtige Request-Modelle

`POST /api/runs/start`

```json
{
  "config_revision_id": "config_rev_20260307T221530Z",
  "runs_dir": "/data/tile_runs",
  "run_name": "M31_altaz_test",
  "color_mode": "MONO",
  "filter_queue": [
    {"filter_name":"L","input_dir":"/data/m31/L","pattern":"*.fits","run_label":"L","enabled":true}
  ]
}
```

`POST /api/runs/{run_id}/resume`

```json
{
  "from_phase": "BGE",
  "config_revision_id": "config_rev_20260307T221530Z",
  "filter_context": {"filter_name":"R","index":2}
}
```

`POST /api/tools/astrometry/solve`

```json
{
  "astap_binary": "/home/user/.local/share/tile_compile/astap/astap_cli",
  "astap_data_dir": "/home/user/.local/share/tile_compile/astap",
  "fits_path": "/runs/xyz/stacked_m31.fit",
  "search_radius_deg": 180
}
```

`POST /api/tools/pcc/run`

```json
{
  "rgb_fits": "/runs/xyz/stacked_rgb.fit",
  "wcs_file": "/runs/xyz/stacked_rgb.wcs",
  "source": "siril",
  "siril_catalog_dir": "/home/user/.local/share/siril/siril_cat1_healpix8_xpsamp",
  "params": {"mag_limit":14.0,"mag_bright_limit":6.0,"min_stars":10,"sigma_clip":2.5}
}
```

## 11.7.8 Statuscodes (verbindlich)

1. `200` fuer sync read/success.
2. `202` fuer async Jobstart.
3. `400` fuer Request-Fehler.
4. `404` fuer unbekannte `run_id/job_id/revision_id`.
5. `409` fuer Zustandskonflikte (z. B. Resume ohne Phase, Run bereits aktiv).
6. `422` fuer Schema-/Feldvalidierung.
7. `500` fuer unerwartete Backendfehler.

## 12) FastAPI vs. Drogon (Einschaetzung)

## 12.1 Kurzfazit

1. Fuer dieses Projekt ist **FastAPI klar die pragmatischere Wahl**.
2. Drogon ist technisch stark, aber hier mit hoeherem Integrationsaufwand.

## 12.2 Warum FastAPI hier besser passt

1. Bestehende Stats-/Tool-Skripte sind bereits Python-basiert (`generate_report.py`, Diagnose-Tools).
2. API- und DTO-Entwicklung mit Pydantic ist schneller und wartbarer.
3. WebSocket/SSE + JSON-Serialisierung sind mit FastAPI sofort produktiv nutzbar.
4. Team-/Operations-Aufwand fuer Release und Debugging ist geringer.
5. C++ bleibt auf seine Staerke fokussiert (Runner/CLI/Algorithmen), statt zweiten C++-Webstack einzufuehren.

## 12.3 Wann Drogon sinnvoll waere

1. Wenn ein reiner C++-Stack ohne Python-Abhaengigkeiten strategisch zwingend ist.
2. Wenn sehr hohe API-Last (nicht Runner-Last) die zentrale Bottleneck ist.
3. Wenn ein dediziertes C++-Backend-Team dauerhaft verfuegbar ist.

## 12.4 Empfehlung

1. GUI2 Backend mit FastAPI umsetzen.
2. Runner/CLI als stabile Prozessadapter beibehalten.
3. Entscheidung in einem spaeteren Architektur-Review nur dann neu bewerten, wenn echte Performance-/Betriebsdaten FastAPI als Engpass zeigen.
