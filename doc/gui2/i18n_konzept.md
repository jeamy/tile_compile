# i18n-Konzept fuer GUI 2

## Ziel

GUI 2 soll von Anfang an mehrsprachig sein, ohne spaetere Grossumbauten.

## Sprachumfang Phase 1

- `de` (Default)
- `en`

## Was wird uebersetzt

- Navigation, Buttons, Dialoge
- Validierungs- und Guardrail-Meldungen
- Parameter-Kurz-Erklaerungen
- Szenario-Empfehlungstexte
- Hilfe-/Tooltip-Texte

Nicht uebersetzt bleiben technische Schluessel/Bezeichner wie `pcc.k_max` oder `sigma_clip`.

## Architektur

- Uebersetzungsdateien pro Sprache, z. B.:
  - `i18n/de.json`
  - `i18n/en.json`
- Source of Truth fuer Control-basierte UI-Keys: `gui2_control_registry.yaml` (`label_key`, `tooltip_key`).
- Zusaetzliche (nicht direkt Control-gebundene) Keys kommen aus:
  - `i18n/additional_keys.yaml`
- String-Keys statt Harttext im UI:
  - `ui.nav.dashboard`
  - `ui.nav.input_scan`
  - `ui.nav.assumptions`
  - `ui.nav.live_log`
  - `ui.button.validate`
  - `ui.tooltip.dashboard.run_start`
  - `dashboard.kpi.scan_quality`
  - `dashboard.kpi.open_warnings`
  - `ui.tooltip.dashboard.kpi_scan_quality`
  - `ui.tooltip.dashboard.kpi_open_warnings`
  - `queue.row.filter_name`
  - `queue.row.input_dir`
  - `queue.row.pattern`
  - `queue.row.enabled`
  - `ui.field.runs_dir`
  - `ui.field.run_name`
  - `ui.field.run_path_preview`
  - `ui.field.resume_config_revision`
  - `ui.tooltip.parameter.search`
  - `param.registration.allow_rotation.short_help`
  - `scenario.altaz.title`
  - `queue.filter.title`
  - `queue.filter.progress`
  - `queue.resume.target`
  - `monitor.stats.generate`
  - `monitor.stats.open_folder`
  - `monitor.stats.command_hint`
  - `ui.button.restore_config_revision`
  - `tools.pcc.source`
  - `tools.pcc.sigma`
  - `tools.pcc.min_stars`
  - `ui.tooltip.tools.pcc_run`
  - `scenario.few_frames`
  - `scenario.few_frames.title`
  - `scenario.few_frames.reason.stability`
  - `scenario.gradient`
  - `scenario.gradient.title`
  - `scenario.gradient.reason.background`
  - `queue.filter.OIII`
  - `queue.filter.SII`
  - `queue.row.run_label`
  - `param.data.bayer_pattern.label`
  - `phase.calibration`
  - `phase.prewarp`
  - `phase.local_metrics`
  - `phase.tile_reconstruction`
  - `phase.state_clustering`
  - `phase.synthetic_frames`
  - `phase.astrometry`
  - `param.assumptions.pipeline_profile.label`
  - `param.calibration.use_bias.label`
  - `param.calibration.bias_dir.label`
  - `param.calibration.use_dark.label`
  - `param.calibration.darks_dir.label`
  - `param.calibration.use_flat.label`
  - `param.calibration.flats_dir.label`
  - `assumptions.pipeline_mode_info`
  - `ui.panel.scan_results`
  - `ui.panel.validation_result`
  - `ui.field.pattern`
  - `ui.field.with_checksums`
  - `ui.tooltip.input_scan.*` (Input & Scan Screen)
  - `ui.tooltip.assumptions.*` (Assumptions Screen)
  - `ui.tooltip.wizard.*` (Wizard Screen)

## Aktueller Scope-Stand (synchron)

- Stats liegt i18n-seitig im `Run Monitor` (`monitor.stats.*`).
- PCC bleibt i18n-seitig in `History+Tools` (`tools.pcc.*`).
- MONO-Queue-Eingaben je Filter sind ueber `queue.row.*` abgedeckt, inkl. `queue.row.run_label`.
- Navigation umfasst alle Shell-Ziele (`ui.nav.*`, inkl. `input_scan`, `assumptions`, `live_log`).
- Alle 5 Situation-Szenarien abgedeckt: `altaz`, `rotation`, `bright_stars`, `few_frames`, `gradient`.
- Alle C++-Phasen als Resume-Ziele abgedeckt: inkl. `calibration`, `prewarp`, `local_metrics`, `tile_reconstruction`, `state_clustering`, `synthetic_frames`, `astrometry`.
- Filter-Chips MONO komplett: `L`, `R`, `G`, `B`, `Ha`, `OIII`, `SII`.
- `data.bayer_pattern` als konfigurierbares Feld in Input & Scan abgedeckt (`param.data.bayer_pattern.label`).
- Assumptions-Screen: alle Felder mit `param.assumptions.*` und `ui.tooltip.assumptions.*` abgedeckt.
- Wizard-Screen: alle Steps mit `ui.tooltip.wizard.*` abgedeckt.
- Input & Scan-Screen: alle Felder und Kalibrierungs-Controls mit `ui.tooltip.input_scan.*` abgedeckt.
- `de.json` und `en.json`: alle 304 Keys uebersetzt, kein `TODO::` mehr vorhanden.

## Runtime-Verhalten

- Sprache per Topbar-Switch (`DE` / `EN`) sofort wechselbar.
- Kein Neustart erforderlich.
- Aktive Sprache in GUI-State speichern.

## Parameter-Erklaerungen

- Kurz-Erklaerungen aus Metadaten in beiden Sprachen pflegen.
- Falls Uebersetzung fehlt: Fallback auf Englisch plus Marker `missing translation` im Dev-Mode.

## Qualitaetssicherung

- Lint fuer fehlende Keys.
- Snapshot-Tests fuer kritische Screens in `de` und `en`.
- Platzhalter-Check (`{value}`, `{min}`, `{max}`) auf korrekte Substitution.
- CI-Check: Alle in `gui2_control_registry.yaml` referenzierten `label_key`/`tooltip_key` muessen in `de.json` und `en.json` vorhanden sein.
- CI-Check umfasst ebenfalls `i18n/additional_keys.yaml`.
- Automatisierbar ueber:
  - `python3 doc/gui2/scripts/sync_i18n_from_registry.py --check` (nur pruefen)
  - `python3 doc/gui2/scripts/sync_i18n_from_registry.py` (fehlende Keys anlegen)

## Erweiterungspfad

1. `de`/`en` stabilisieren.
2. spaetere Sprachen (z. B. `fr`, `es`) ueber gleiche Key-Struktur erweitern.
3. optionale locale-spezifische Formatierung (Zahlen, Datumsformat, Einheiten).
