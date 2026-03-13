# Klickdummy-Ablaufe

Referenzbilder:

- `mockups/gui2_07_flow_overview.png` (Ablaufkarte)
- `mockups/gui2_02_dashboard.png`
- `mockups/gui2_03_parameter_studio.png`
- `mockups/gui2_04_run_monitor.png`
- `mockups/gui2_05_history_tools.png`

Klickdummy-Einstieg: `clickdummy/index.html`
Wizard-Klickdummy (detailliert): `clickdummy/wizard.html`

## Ablauf 1: Standard-Run (Desktop)

1. `dashboard.html` oeffnen.
2. `runs_dir` waehlen und freien `run_name` setzen.
3. Preview pruefen: `<run_name>_<YYYYMMDD_HHMMSS>`.
4. In "Guided Run" auf "Parameter Studio" klicken.
5. In `parameter-studio.html` Werte pruefen/anpassen, dann "Speichern" klicken.
6. In `run-monitor.html` Lauf beobachten (Phasen/Logs/Artefakte).
7. Unterhalb von `Live Log` optional `Generate Stats` ausfuehren.
8. Ueber "Report" nach `history-tools.html` springen.

Ergebnis: schneller End-to-End-Pfad von Setup bis Analyse.

## Ablauf 1a: MONO Multi-Filter Serienlauf

1. `dashboard.html` oeffnen.
2. Im Guided-Run den MONO-Queue-Editor oeffnen.
3. Nur bei MONO (nicht OSC): pro Filter eigenen Input-Ordner erfassen, z. B.:
   - `L -> /data/m31/L`
   - `R -> /data/m31/R`
   - `G -> /data/m31/G`
   - `B -> /data/m31/B`
   - optional `Ha/OIII/SII`
4. Reihenfolge pruefen und Queue starten.
5. Im `run-monitor.html` seriellen Fortschritt beobachten (`Filter i/N`).
6. Bei Fehler: Resume per Klick auf gewuenschte Phase und Filtereintrag.

Ergebnis: mehrere MONO-Filter-Runs werden ohne manuelle Neustarts seriell abgearbeitet.

## Ablauf 1b: Run-Zielordner und Name festlegen

1. Im Dashboard `runs_dir` per Folder-Picker waehlen.
2. Freien `run_name` eingeben.
3. Pfadvorschau kontrollieren (`<run_name>_<YYYYMMDD_HHMMSS>`).
4. Bei Bedarf `run_name` anpassen, bis der Zielordner passt.
5. Run starten; das Datums-Suffix wird beim tatsaechlichen Startzeitpunkt gesetzt.

Ergebnis: Ausgabeort und Run-Naming sind kontrollierbar, bleiben aber eindeutig ueber das Datums-Suffix.

## Ablauf 2: Parameter-Deep-Dive (Expert)

1. `parameter-studio.html` oeffnen.
2. Suche verwenden (`pcc.`, `bge.`, `runtime_limits.`).
3. Kurz-Erklaerung und Guardrails im Explain-Panel pruefen.
4. YAML-Diff kontrollieren.
5. Validieren und speichern.

Ergebnis: alle verwendeten Parameter bleiben beherrschbar, ohne YAML-only Inseln.

## Ablauf 3: Situation Assistant (Alt/Az, Rotation, helle Sterne)

1. `parameter-studio.html` oeffnen.
2. Situation waehlen (z. B. `Alt/Az`, `Starke Rotation`, `Helle Sterne`).
3. Angezeigte Delta-Empfehlungen prufen.
4. Preset-Deltas uebernehmen.
5. Validieren und speichern.

Ergebnis: schnelle, kontextbezogene Voreinstellungen ohne Blindflug.

## Ablauf 3a: Parameter aendern und ab Phase fortsetzen (Resume)

1. `parameter-studio.html` oeffnen.
2. Relevante Parameter anpassen (z. B. Registration/BGE/PCC).
3. `Validieren` ausfuehren; bei Fehlern erst korrigieren.
4. `Speichern` klicken (neue Config-Version wird aktiv).
5. Zu `run-monitor.html` wechseln.
6. Optional aktiven Filter in der MONO-Queue setzen.
7. Gewuenschte Config-Revision fuer Resume waehlen.
8. Gewuenschte Phase direkt anklicken (`Resume ab Phase X`).
9. Resume starten und im Header pruefen, dass Config-Version + Filter/Phase angezeigt werden.
10. Falls Ergebnis schlechter ist: aeltere Revision wiederherstellen und erneut ab Phase fortsetzen.

Ergebnis: gezielte Korrekturen ohne kompletten Neustart; Fortsetzung ab definierter Phase mit neuer Konfiguration.

## Ablauf 4: Monitoring und Recovery

1. `run-monitor.html` oeffnen.
2. Bei Warnungen betroffene Phase im Log verfolgen.
3. Gewuenschte Phase direkt anklicken (Resume ab Phase X, im Dummy simuliert).
4. Artefakte pruefen (rechts).
5. Unter `Live Log` bei Bedarf `Generate Stats` ausfuehren.
6. "Resume" oder "Report" nutzen.
7. In `history-tools.html` Vergleich mit frueheren Runs durchfuehren.

Ergebnis: operativer Betrieb inklusive Fehlerfall/Recovery bleibt in einer konsistenten UI.

## Ablauf 5: Historie + Tooling

1. `history-tools.html` oeffnen.
2. Run in Historie waehlen.
3. Fuer Plate-Solve auf `astrometry.html` wechseln.
4. Fuer Farbkalibrierung auf `pcc.html` wechseln (`Run PCC`, `Save Corrected`).
5. Bei Bedarf fuer Reports nach `run-monitor.html` wechseln und dort `Generate Stats` ausfuehren.

Ergebnis: Nachbearbeitung und Laufvergleich sind nicht mehr getrennte Funktionsinseln.

## Ablauf 6: i18n Live-Umschaltung

1. Sprache in der Topbar auf `DE` oder `EN` setzen.
2. Navigation, Parameter-Kurztexte und Warnungen aktualisieren sich sofort.
3. Szenario-Empfehlungstexte bleiben fachlich identisch, nur sprachlich umgestellt.

Ergebnis: gleiche Funktion fuer unterschiedliche Nutzergruppen ohne UI-Bruch.

## Ablauf 7: MONO-Szenarioprofil aus Beispiel-YAML laden

1. In `parameter-studio.html` ueber `YAML Sync` eines der Profile laden:
   - `tile_compile_cpp/examples/tile_compile.mono_full_mode.example.yaml`
   - `tile_compile_cpp/examples/tile_compile.mono_small_n_anti_grid.example.yaml`
   - `tile_compile_cpp/examples/tile_compile.mono_small_n_ultra_conservative.example.yaml`
2. `scenario_profile.id` wird erkannt und im Situation Assistant vorausgewaehlt.
3. `scenario_profile.gui2_scenarios` wird als aktive Szenario-Tags gesetzt.
4. Delta-Vorschlaege kontrollieren und uebernehmen.
5. Validieren und speichern.

Ergebnis: reproduzierbarer Startpunkt fuer MONO-Workflows, mit explizit dokumentiertem Szenario-Kontext.

## Ablauf 8: Layout-Review 1920 (detailliert)

1. `clickdummy/layout-1920.html` oeffnen.
2. Overlay 1 (`gui2_08_layout_1920_overlay.png`) auf Shell/Main/Wrapper pruefen.
3. Overlay 2 (`gui2_09_layout_1920_measurelines.png`) auf Pixelmasslinien je Screen pruefen.
4. Shell-Hotspot (Sidebar) anklicken und mit `dashboard.html` querpruefen.
5. Main-Wrapper-Hotspot anklicken und mit `parameter-studio.html` vergleichen.
6. Split-Zeilen nacheinander klicken:
   - Dashboard Split
   - Parameter Split
   - Run Monitor Split
   - History+Tools Split
7. Interaktionszonen gegen `layout_1920_spec.md` Abschnitt 6/13 validieren.
8. Spacing-Gate ausfuehren: `python3 doc/gui2/scripts/check_layout_1920_spacing.py`.
9. Bei Abweichungen Koordinaten/Spacing-Tokens in `scripts/generate_mockups.py` und Hotspots synchron anpassen.

Ergebnis: Mockup, Klickdummy und 1920-Spezifikation bleiben pixelgenau abgestimmt.

## Ablauf 9: Stats im Run Monitor starten

1. `run-monitor.html` oeffnen.
2. Zum Block `Stats` unterhalb von `Live Log` scrollen.
3. `Generate Stats` klicken.
4. Kommandohinweis pruefen: Das aufgeloeste Kommando wird im Stats-Panel angezeigt (Pfad wird aus Konfiguration geladen, siehe §4.5 im Detailkonzept).
5. Danach Report/Run-Monitor fuer Ergebnisfluss oeffnen.

Ergebnis: Stats-Erzeugung ist direkt im operativen Run-Monitor verfuegbar, ohne Wechsel in History+Tools.

## Ablauf 10: Implementierungsvollstaendigkeit pruefen

1. `implementierungsablauf_funktionen.md` oeffnen.
2. `gui2_control_registry.yaml` oeffnen.
3. Abschnitt 4 (Funktionsmatrix) zeilenweise gegen den aktuellen Build pruefen.
4. Registry-Status pro Control (`TODO/IN_PROGRESS/DONE`) aktualisieren.
5. Abschnitt 7 (Legacy-Paritaet) auf `DONE` setzen.
6. Abschnitt 8 (Abnahmetests) als finalen Gate ausfuehren, inkl. Spacing Gate.

Ergebnis: es bleibt keine Funktion unimplementiert oder undokumentiert.

## Klickdummy-Hotspots

- Sidebar-Hotspots sind in den Desktop-Screens vollstaendig aktiv (Dashboard, Input&Scan, Parameter Studio, Assumptions, Run Monitor, History, Astrometry, PCC, Live Log).
- Kernaktionen (Parameter speichern, Run starten, Report) sind als Hotspots hinterlegt.
- Ablaufkarte (`flow.html`) verlinkt Knoten direkt auf die passenden Screens.
