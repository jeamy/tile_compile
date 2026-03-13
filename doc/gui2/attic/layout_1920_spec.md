# Layout 1920 Spec (Linie B, Desktop)

Diese Spezifikation definiert ein fixes Desktop-Layout fuer `1920x1080` als Primarziel.

## 1) Basisraster

- Viewport: `1920 x 1080`
- Outer Margin: `24 px` links/rechts
- Topbar Height: `76 px` (y=20..96 in der HTML-Referenz)
- Content Start: `y = 116 px`
- Global Gap (zwischen Hauptbereichen): `24 px`
- Card Radius:
  - Primare Cards: `18-20 px`
  - Unterkarten / Inputs: `10-12 px`

## 2) Shell-Raster

Horizontale Aufteilung (fix):

1. Sidebar
   - `x=24`
   - `width=300 px`
2. Main Area
   - `x=356`
   - `width=1538 px`

Herleitung:

- `24 (margin) + 300 (sidebar) + 32 (gap) + 1538 (main) + 24 (margin) = 1918` (2 px Rundungsreserve durch Borders)

Vertikale Aufteilung:

- Topbar: `y=20..96`
- Workspace: `y=116..1056`

## 3) Main-Area Grid (Linie B)

Standard fuer grosse Inhaltsseiten (`Dashboard`, `Parameter Studio`, `Run Monitor`, `History+Tools`, `Astrometry`, `PCC`):

- Main Card Wrapper: `x=356`, `y=230`, `w=1538`, `h=798`
- Innenabstand Wrapper: `24 px`

Empfohlene Spaltenlogik im Wrapper:

- 12-Spalten-Gefuehl, aber mit festen Panelbreiten statt fluiden Prozenten
- Standard Gutter: `18 px`

## 4) Screen-Spezifische Breiten

## 4.1 Dashboard

- Linkes Hauptpanel (`Guided Run`): `854 px`
- Rechtes Guardrail-Panel: `664 px`
- Unteres Pipeline-Panel: volle Wrapper-Breite

Orientierungswerte aus HTML-Referenz:

- Guided Run: `x=356..1210`
- Guardrails: `x=1230..1894`
- Readiness-Listenlayout innerhalb Guardrails:
  - erste Zeile startet bei `y=470`
  - Zeilenhoehe `46 px`
  - vertikaler Step `56 px`
  - damit bleiben alle 5 Zeilen vollstaendig innerhalb `y=402..784`

## 4.2 Parameter Studio

Drei feste Spalten im Wrapper:

1. Kategorien links: `350 px`
2. Form Mitte: `632 px`
3. Explain/Situation rechts: `470 px`

Zwischenabstand:

- links -> mitte: `16-18 px`
- mitte -> rechts: `18 px`

Orientierungswerte aus HTML-Referenz:

- Links: `x=382..732`
- Mitte: `x=748..1380`
- Rechts: `x=1398..1868`
- Registration-Mikrolayout (Mittelspalte):
  - Row 1: `engine`, `allow_rotation` bei `y=424`
  - Row 2: `star_topk`, `star_inlier_tol_px`, `reject_cc_min_abs` bei `y=516`
  - Horizontaler Gap in Row 2: `20 px` (visuell entspannter als die vorherige enge Variante)

## 4.3 Run Monitor

Drei feste Spalten im Wrapper:

1. Phasen: `648 px`
2. Live Log: `420 px`
3. Artefakte/Actions: `382 px`

Orientierungswerte aus HTML-Referenz:

- Phasen: `x=382..1030`
- Live Log + Stats: `x=1048..1468`
- Artefakte: `x=1486..1868`
- Mittelspalte intern:
  - Live Log oben: `y=424..628`
  - Stats darunter: `y=646..968`

## 4.4 History + Tools

Zweispaltenlayout:

1. History links: `860 px`
2. Tools rechts: `608 px`

Orientierungswerte aus HTML-Referenz:

- History: `x=382..1242`
- Tools: `x=1260..1868`
- Stats-Funktionen sind aus History+Tools in den Run Monitor verlagert.
- Astrometry und PCC sind eigenstaendige Screens; History+Tools enthaelt nur Historie + Deep-Links.

## 5) Typografie-Skala (Desktop 1920)

- App/Screen Titel (Serif): `42 px`
- Card Titel (Serif): `30-34 px`
- Section Titel: `22-26 px`
- UI Text: `18-20 px`
- Secondary Text: `15-17 px`
- Chips: `17 px`

## 6) Interaktionszonen

Mindestgroessen fuer klickbare Ziele:

- Primar-Buttons: `>= 52 px` Hoehe
- Sekundaer-Buttons: `>= 44 px` Hoehe
- Zeileninteraktionen (z. B. Phasen-Resume): `>= 48 px` Hoehe
- Chip/Toggle: `>= 34 px` Hoehe

## 6.1 Verbindliche Spacing-Tabelle (Tokens)

Alle Werte sind als zentrale Sollwerte im Layout-Spec und in den Clickdummy-CSS-Dateien (`theme.css`, `screen-placeholder.css`) zu halten.

| Token | Wert | Bedeutung |
|---|---|---|
| `global.wrapper_inner_padding` | `24` | Innenabstand Main-Wrapper |
| `global.field_label_to_input_gap` | `12` | Abstand Label zu Inputkante |
| `dashboard.readiness_first_row_y` | `470` | Start-y Readiness-Liste |
| `dashboard.readiness_row_h` | `46` | Zeilenhoehe Readiness |
| `dashboard.readiness_row_step` | `56` | Vertikaler Zeilenstep Readiness |
| `parameter_studio.registration_row1_y` | `424` | Y der Registration-Row 1 |
| `parameter_studio.registration_row2_y` | `516` | Y der Registration-Row 2 |
| `parameter_studio.registration_row2_hgap` | `20` | Horizontaler Gap in Registration-Row 2 |
| `parameter_studio.section_title_gap` | `34` | Abstand Input-Row zu naechster Section-Title |
| `run_monitor.artifact_list_first_y` | `480` | Erste Artefaktzeile |
| `run_monitor.artifact_button_row_y` | `804` | Y von `Resume`/`Report` |
| `run_monitor.artifact_secondary_button_y` | `868` | Y von `Run-Ordner offnen` |
| `astrometry.first_input_y` | `356` | Erstes Astrometry-Feld |
| `astrometry.row_step` | `86` | Vertikaler Schritt Astrometry-Felder |
| `astrometry.plate_solve_y` | `528` | Y von `Plate Solve File` + `Solve` |

## 6.2 Spacing-Pruefregeln (automatisch pruefbar)

Muss in CI/Review laufen:

```bash
python3 doc/gui2/scripts/check_layout_1920_spacing.py
```

Der Check validiert:

1. Tokenwerte gegen Sollwerte aus 6.1.
2. Readiness-Liste bleibt innerhalb Guardrail-Panel (`bottom <= 784`).
3. Abstand letzte Artefaktzeile -> Action-Buttons ist mindestens `20 px`.

Nur bei `Result: OK` gilt das Layout als spacing-konform.

## 7) Resizing-Regeln (Desktop-only)

- Primarziel bleibt `1920x1080`.
- Bei groesserem Viewport:
  - Kernbreiten bleiben stabil (1920-Baseline),
  - Content bleibt zentriert,
  - Aussenraender wachsen automatisch.
- Bei kleiner als 1920:
  - horizontales Scrollen ist erlaubt und vorgesehen,
  - kein alternatives Kompaktlayout vorgesehen.

## 7.1 Konkretes Klickdummy-Verhalten

- In `clickdummy/style.css`:
  - `body.min-width = 1920px`
  - `topbar/main.max-width = 1920 - 48`
- Ergebnis:
  - `>1920 px`: mehr Seitenrand links/rechts.
  - `<1920 px`: horizontaler Scroll statt Umbruch auf Kompaktlayout.

## 8) Mapping zu den vorhandenen Assets (HTML-only)

- Primare Referenzseiten liegen unter `doc/gui2/clickdummy/*.html`.
- Das Layout wird direkt ueber HTML-Struktur und CSS definiert, nicht ueber PNG-Overlays.
- Technische Referenz fuer Rasterregeln:
  - `doc/gui2/clickdummy/theme.css`
  - `doc/gui2/clickdummy/screen-placeholder.css`
  - diese Spezifikation

## 9) Implementierungshinweis (Web/Crow)

Fuer die Web-Umsetzung empfiehlt sich:

1. Feste Mindestbreite der App-Shell: `1920`.
2. Spaltenbreiten als zentrale CSS-Tokens und konstante Grid-Definition.
3. Pixel-Snap (int-Rundung) fuer Breiten/Abstaende, um 1px-Artefakte zu vermeiden.
4. API-getriebene Datenbindung ohne Layout-Logik im Backend.

## 10) Koordinatenmatrix (px + Prozent vom 1920x1080 Canvas)

### 10.1 Shell

| Bereich | Pixel (x,y,w,h) | Relativ (%) |
|---|---|---|
| Sidebar | `24,116,300,940` | `left 1.25, top 10.74, width 15.62, height 87.04` |
| Main Area | `356,116,1538,940` | `left 18.54, top 10.74, width 80.10, height 87.04` |
| Main Wrapper | `356,230,1538,798` | `left 18.54, top 21.30, width 80.10, height 73.89` |

### 10.2 Screen-Zonen

| Screen | Bereich | Pixel (x,y,w,h) | Relativ (%) |
|---|---|---|---|
| Dashboard | Guided Run | `356,402,854,382` | `18.54 / 37.22 / 44.48 / 35.37` |
| Dashboard | Guardrails | `1230,402,664,382` | `64.06 / 37.22 / 34.58 / 35.37` |
| Parameter Studio | Kategorien | `382,346,350,650` | `19.90 / 32.04 / 18.23 / 60.19` |
| Parameter Studio | Form | `748,346,632,650` | `38.96 / 32.04 / 32.92 / 60.19` |
| Parameter Studio | Explain | `1398,346,470,650` | `72.81 / 32.04 / 24.48 / 60.19` |
| Run Monitor | Phasen | `382,358,648,638` | `19.90 / 33.15 / 33.75 / 59.07` |
| Run Monitor | Live Log + Stats | `1048,358,420,638` | `54.58 / 33.15 / 21.88 / 59.07` |
| Run Monitor | Artefakte | `1486,358,382,638` | `77.40 / 33.15 / 19.90 / 59.07` |
| History+Tools | Historie | `382,266,860,730` | `19.90 / 24.63 / 44.79 / 67.59` |
| Astrometry | Setup/Katalog/Solve | `382,266,1486,730` | `19.90 / 24.63 / 77.40 / 67.59` |
| PCC | Input/Katalog/Parameter | `382,266,1486,730` | `19.90 / 24.63 / 77.40 / 67.59` |
| Run Monitor | Generate Stats Button | `1084,860,166,52` | `56.46 / 79.63 / 8.65 / 4.81` |
| Run Monitor | Open Stats Folder | `1260,860,174,52` | `65.63 / 79.63 / 9.06 / 4.81` |

## 11) Klickdummy-Mapping (Layout-Review)

Die Seite `clickdummy/layout-1920.html` dokumentiert die Rasterzonen als HTML-Referenz:

| Bereich/Link | Relativ (%) | Ziel |
|---|---|---|
| Shell | `1.25,10.74,15.62,87.04` | `dashboard.html` |
| Main Wrapper | `18.54,21.30,80.10,73.89` | `parameter-studio.html` |
| Dashboard Zone | `34.58,69.81,58.33,3.52` | `dashboard.html` |
| Parameter Zone | `34.58,74.63,58.33,3.52` | `parameter-studio.html` |
| Run Monitor Zone | `34.58,79.44,58.33,3.52` | `run-monitor.html` |
| History+Tools Zone | `34.58,84.26,58.33,3.52` | `history-tools.html` |
| Astrometry Zone | `34.58,89.07,58.33,3.52` | `astrometry.html` |
| PCC Zone | `34.58,93.89,58.33,3.52` | `pcc.html` |

## 12) Referenzmodus der Spezifikation

- Diese Spezifikation wird direkt gegen die HTML-Dummies geprueft.
- PNG-Mockups sind optionales Historienmaterial, aber nicht mehr normative Sollquelle.

## 13) Review-Checkliste (detailliert)

1. Shell passt: `x=24/356`, Breiten `300/1538`, Topbar `76 px`.
2. Wrapper passt: `x=356, y=230, w=1538, h=798`.
3. Screen-Spalten stimmen mit Abschnitt 4/10.
4. Primaraktionen sind mindestens `52 px` hoch.
5. Zeilen-Resumes im Run-Monitor sind mindestens `48 px` hoch.
6. Klickdummy-Zonen und Navigationslinks verweisen konsistent auf die Zielscreens.
