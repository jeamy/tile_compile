# GUI 2 Konzeptpaket (Desktop)

Dieses Verzeichnis enthaelt das komplette Redesign-Paket fuer eine modernisierte GUI mit einfacher Eingabe aller verwendeten Parameter.

## Inhalt

- `mockups/`
  - `gui2_01_styleboard.png`: Designvorschlaege (A/B/C)
  - `gui2_02_dashboard.png`: Dashboard + Guided Run
  - `gui2_03_parameter_studio.png`: Vollstaendige Parametereingabe inkl. Explain/Situation/i18n
  - `gui2_04_run_monitor.png`: Live-Phasen, Logs, Stats, Artefakte
  - `gui2_05_history_tools.png`: Historie + Astrometry/PCC
  - `gui2_07_flow_overview.png`: Ablauf-Blueprint
  - `gui2_08_layout_1920_overlay.png`: Detailliertes 1920er Raster-Overlay (Shell, Spalten, Interaktionszonen)
  - `gui2_09_layout_1920_measurelines.png`: Zweite Overlay-Variante mit expliziten Pixelmasslinien je Screen
- `clickdummy/`
  - klickbare HTML-Prototypen auf Basis der Mockups
  - Einstieg: `clickdummy/index.html`
  - Wizard-Ablauf (detailliert): `clickdummy/wizard.html`
  - Layout-Raster-Ansicht: `clickdummy/layout-1920.html`
  - zusaetzliche Sidebar-Pfade: `clickdummy/input-scan.html`, `clickdummy/assumptions.html`, `clickdummy/live-log.html`
  - globale Theme-/Font-Basis: `clickdummy/theme.css`, `clickdummy/theme.js`, `clickdummy/assets/fonts/*`
- `designvorschlaege.md`
  - Designalternativen, Empfehlung und Modernisierung aller GUI-Bereiche
- `detailkonzept.md`
  - Detaillierte UI-Spezifikation (Layout, Komponenten, Guardrails, Situation Assistant)
  - inkl. MONO Multi-Filter Queue (serielle Abarbeitung)
- `implementierungsablauf_funktionen.md`
  - Vollstaendige Funktionsmatrix (Buttons/Felder), Tooltip-Standard und detaillierter Implementierungsablauf
- `gui2_control_registry.yaml`
  - Zentrale Control-Registry mit Statusfeldern (`TODO/IN_PROGRESS/DONE`) pro Control
- `layout_1920_spec.md`
  - Fixes 1920er Layout-Raster (Spalten, Gutter, Panel-Breiten, Interaktionszonen)
- `szenario_empfehlungen.md`
  - Konkrete Parameter-Empfehlungen fuer Alt/Az, starke Rotation, helle Sterne, etc.
  - inkl. MONO-Beispielprofile aus `tile_compile_cpp/examples/*.example.yaml`
- `i18n_konzept.md`
  - Sprachkonzept fuer DE/EN inkl. Parameter-Kurztexte
- `i18n/`
  - `de.json`, `en.json` als Laufzeit-Locales
  - `additional_keys.yaml` fuer nicht direkt Control-gebundene Texte (z. B. Szenario- oder Command-Hinweise)
- `ablaeufe.md`
  - Klickdummy-Flows und End-to-End-Ablaufe
- `parameter_katalog.md`
  - Vollstaendige Parameterliste aus `tile_compile.yaml` + GUI-Laufzeitparameter
- `scripts/`
  - `generate_mockups.py`: erzeugt alle PNG-Mockups neu
  - `check_layout_1920_spacing.py`: validiert verbindliche Spacing-Tokens und Mindestabstaende
  - `generate_parameter_catalog.py`: erzeugt den Parameterkatalog neu

## Empfohlene Designlinie

Empfohlen ist **Variante B - Parameter Studio** (siehe `mockups/gui2_01_styleboard.png`), weil sie die volle Parameterabdeckung mit schneller Bedienung (Suche, Presets, Guardrails, Kurz-Erklaerungen, Szenario-Empfehlungen) am besten verbindet.

## Nutzung

1. Klickdummy starten: `doc/gui2/clickdummy/index.html` im Browser oeffnen.
2. Designvorschlaege lesen: `doc/gui2/designvorschlaege.md`.
3. Detail- und Ablaufbeschreibung lesen: `doc/gui2/detailkonzept.md` und `doc/gui2/ablaeufe.md`.
4. Szenario- und i18n-Teil pruefen: `doc/gui2/szenario_empfehlungen.md` und `doc/gui2/i18n_konzept.md`.
5. Parameterabdeckung pruefen: `doc/gui2/parameter_katalog.md`.
6. Layout-Raster im Klickdummy pruefen: `doc/gui2/clickdummy/layout-1920.html` und `doc/gui2/layout_1920_spec.md`.
7. i18n-Sync pruefen: `python3 doc/gui2/scripts/sync_i18n_from_registry.py --check`.
8. Wizard-Flow pruefen: `doc/gui2/clickdummy/wizard.html` plus `doc/gui2/ablaeufe.md`.

## Scope

- Die GUI-2-Konzeption ist jetzt explizit auf Desktop optimiert.
- Layout-Baseline: mindestens `1920x1080`.
- Resume arbeitet mit Config-Revisionshistorie (alte Konfigurationen bleiben erhalten und sind wiederherstellbar).
- `runs_dir` ist frei waehlbar; Run-Namen sind frei definierbar, der Zielordner endet immer mit `<YYYYMMDD_HHMMSS>`.

## Themes, Fonts, Monitore

- Themes sind zentral ueber CSS-Variablen in `clickdummy/theme.css` steuerbar.
- Theme schnell wechseln: URL-Parameter `?theme=observatory|slate|sand` (wird per `theme.js` gespeichert).
- Fonts sind lokal eingebettet (`Inria Sans/Serif` in `clickdummy/assets/fonts`), dadurch identisches Rendering auf macOS/Windows/Linux.
- Monitorverhalten:
  - groesser als 1920: Inhalt bleibt auf 1920-Baseline zentriert, Aussenraender wachsen.
  - kleiner als 1920: horizontales Scrollen ist bewusst erlaubt (kein Kompaktlayout).
