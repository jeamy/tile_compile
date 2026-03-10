# Konsistenzcheck GUI2 Clickdummy (HTML) - 2026-03-07

Gepruefte Quellen:

- `doc/gui2/clickdummy/*.html`
- `doc/gui2/detailkonzept.md`
- `doc/gui2/html_fastapi_architektur.md` (Architekturdokument, heute Crow/C++-Stand)
- `doc/gui2/implementierungsablauf_funktionen.md`
- `doc/gui2/layout_1920_spec.md`
- `doc/gui2/parameter_katalog.md`

## Ergebnis

Gesamtstatus: **synchronisiert** (HTML-only Zielbild, Crow/C++-Backend, Desktop 1920).

## Gepruefte Punkte und Anpassungen

1. Parameter Studio war zu klein gegen Matrix/Katalog
- Vorher: nur wenige Basisfelder.
- Jetzt: Suche, Preset/YAML/Validation, Registration/BGE/PCC-Kernfelder, Deprecated-Block, Explain, Situation Assistant, Diff, Save/Reset/Review.
- Datei: `doc/gui2/clickdummy/parameter-studio.html`.

2. Tooltip-Pflicht fuer alle interaktiven Controls
- Vorher: viele Controls ohne Tooltip.
- Jetzt: direkte Tooltips + globaler Fallback.
- Dateien:
  - `doc/gui2/clickdummy/tooltips.js`
  - alle `doc/gui2/clickdummy/*.html` eingebunden.

3. Run Monitor Funktionsparitaet
- Vorher: reduzierte Phasenliste, wenige Resume-Controls.
- Jetzt: komplette Phasenkette (inkl. Resume per Phase), Filterchips (L/R/G/B/Ha/OIII/SII), Config-Revision, Restore, Stats unter Live Log.
- Datei: `doc/gui2/clickdummy/run-monitor.html`.

4. History+Tools Trennung gegen Stats
- Vorher: uneinheitlich mit Stats-Logik.
- Jetzt: nur History/Astrometry/PCC; kein Generate-Stats auf der Seite.
- Datei: `doc/gui2/clickdummy/history-tools.html`.

5. Wizard-Detaillierung
- Vorher: nur Ablaufkarten.
- Jetzt: detaillierte Steps mit Draft-Feldern und Control-Mapping (Input, Calibration, Queue, Preset/Situation, Validation, Start/Navigation).
- Datei: `doc/gui2/clickdummy/wizard.html`.

6. Input & Scan Matrix-Abgleich
- Ergaenzt: `scan.frames_min` sowie Control-Mapping fuer Scan/Calibration-Felder.
- Datei: `doc/gui2/clickdummy/input-scan.html`.

7. Desktop 1920 Konsistenz
- `theme.css` auf `--layout-min-width: 1920px` gesetzt.
- Datei: `doc/gui2/clickdummy/theme.css`.

8. Detailkonzept ohne Mockup-PNG als Sollquelle
- Ergaenzt: HTML-only Referenzgrundlage.
- Technischer Pfad auf HTML + Crow/C++-Backend (ohne Qt-GUI) umgestellt.
- Datei: `doc/gui2/detailkonzept.md`.

9. Layout-Spec aktualisiert
- Mockup-/PNG-/Qt-Referenzen auf HTML-only und Web/Crow angepasst.
- Datei: `doc/gui2/layout_1920_spec.md`.

10. Implementierungsmatrix + Katalog synchronisiert
- Tooltip-Umsetzung fuer HTML konkretisiert.
- Parameter-Studio-Abgleich in Katalog dokumentiert.
- Dateien:
  - `doc/gui2/implementierungsablauf_funktionen.md`
  - `doc/gui2/parameter_katalog.md`

## Hinweis zur Vollabdeckung

Der Clickdummy zeigt eine funktionsvolle Linie-B Referenz mit Kernfeldern und Such-/Kategoriefluss.
Die produktive Vollabdeckung aller YAML-Keys bleibt verbindlich an `parameter_katalog.md` und `gui2_control_registry.yaml` gebunden.
