# Designvorschlaege GUI 2

Referenzbilder:

- `mockups/gui2_01_styleboard.png`
- `mockups/gui2_03_parameter_studio.png`

## Ziel

- Moderne, ruhige, produktionsreife Oberflaeche fuer Astrofoto-Workflows.
- Einfache Eingabe **aller** verwendeten Parameter.
- Einheitlicher Ablauf von Scan bis Report, ohne Funktionsinseln.
- Sofort verstaendliche Parameter durch Kurz-Erklaerungen und Kontextempfehlungen.

## Vorschlag A - Observatory Deck

Kurzbild: klare Kartenstruktur mit starkem Fokus auf Schnellstart und Status.

- Staerken:
  - Sehr schneller Einstieg fuer wiederholte Runs.
  - Gute Lesbarkeit fuer Monitoring.
- Risiken:
  - Tiefe Parametrisierung kann schnell auf Unterseiten ausweichen.

## Vorschlag B - Parameter Studio (Empfohlen)

Kurzbild: zentraler Arbeitsraum mit Suchleiste, Presets, Guardrails, YAML-Diff.

- Staerken:
  - Vollstaendige Parametereingabe bleibt dennoch uebersichtlich.
  - Sehr gut fuer Expert:innen und Einsteiger (Guided + Expert Mode).
  - Direkte Rueckmeldung bei ungueltigen oder riskanten Werten.
- Risiken:
  - Braucht klare Informationshierarchie und gute Defaults.

### B.1 Verbindliche Erweiterung: Parameter-Erklaerungen fuer alle Felder

Jedes Feld bekommt einen einheitlichen Info-Block:

- `Kurz`: ein Satz, was der Parameter macht.
- `Wirkung`: welche Pipeline-Phase und welche Bildauswirkung betroffen ist.
- `Range`: erlaubte Werte laut Schema.
- `Default`: Standardwert.
- `Risiko`: was bei zu hohen/niedrigen Werten typischerweise passiert.

Damit sind auch tiefe Schluessel wie `bge.fit.rbf_lambda` oder `pcc.annulus_outer_fwhm_mult` direkt verstaendlich.

### B.2 Verbindliche Erweiterung: Objekt-/Situations-Empfehlungen

Im Parameter Studio kommt ein `Situation Assistant` hinzu. Er zeigt empfohlene Werte als Delta zum aktuellen Profil.

Pflicht-Situationen in GUI 2:

- Alt/Az Setup
- Starke Feldrotation
- Helle Sterne im Feld
- Wenige Frames / kurze Session
- Starker Hintergrundgradient / Lichtverschmutzung

Optional erweiterbar um Objektprofile:

- Galaxie (schwacher Hintergrund, feine Struktur)
- Emissionsnebel (grossflaechige Gradienten)
- Sternhaufen (viele punktfoermige Sterne)

### B.2a Verbindliche Erweiterung: MONO Multi-Filter Queue (seriell)

Fuer MONO-Runs muss die GUI mehrere Filtereingaben als Queue unterstuetzen, z. B. `L`, `R`, `G`, `B`, `Ha`, `OIII`, `SII`.

- Eingabe je Filter:
  - Filtername
  - Input-Ordner
  - optionales Subfolder-/Run-Label
  - optional eigenes Pattern
- Ausfuehrung:
  - strikt seriell (ein Filter nach dem anderen)
  - klarer Queue-Fortschritt (`Filter 2/7`)
  - Resume gezielt pro Queue-Eintrag und Phase
- Nutzen:
  - reproduzierbarer Workflow fuer LRGB/SHO-Daten
  - weniger manuelle Run-Neustarts

### B.3 Verbindliche Erweiterung: i18n von Anfang an

- UI-Sprache umschaltbar (mindestens `de` und `en`) ohne Neustart.
- Uebersetzt werden nicht nur Labels, sondern auch:
  - Parameter-Kurztexte
  - Warnungen und Guardrail-Meldungen
  - Szenario-Empfehlungen und Begruendungen
- Fachbegriffe bleiben konsistent (z. B. `sigma_clip`, `k_max`, `RBF`) und werden bei Bedarf erlaeutert.

## Vorschlag C - Timeline Command

Kurzbild: ablauforientierte Oberflaeche fuer Batch- und Historienarbeit.

- Staerken:
  - Gut fuer Multi-Run-Vergleiche und Team-Review.
- Risiken:
  - Parameterpflege kann verteilt wirken, wenn nicht mit Studio kombiniert.

## Empfehlung

**Vorschlag B** als Hauptlinie, erweitert um Timeline-Elemente aus C im Run Monitor und in der Historie.

## Modernisierung der gesamten bisherigen GUI

Aktueller Zustand (`gui_cpp`): getrennte Tabs (`Scan`, `Configuration`, `Assumptions`, `Run`, `Pipeline Progress`, `Current run`, `Run history`, `Astrometry`, `PCC`, `Live log`).

GUI-2-Zielstruktur:

| Bisher | GUI 2 | Modernisierung |
|---|---|---|
| Scan | Input & Scan | Multi-Ordner-Management, Kalibrierung als klare Karten mit Inline-Validierung |
| Configuration + Assumptions | Parameter Studio | Alle YAML-Keys als Form + Suche + Presets + YAML-Diff + **Kurz-Erklaerung + Situation Assistant** |
| Run | Dashboard (Guided Run) | Startblockaden als sichtbare Checkliste statt versteckter Tooltips |
| Pipeline Progress | Run Monitor | Phasenleiste, Batch-Kontext, Logs und Artefakte in einem Screen |
| Current run | Run Monitor + History Detail | Refresh/Resume/Report in einheitlichem Detailpanel |
| Run history | History | Vergleichsfaehige Tabelle mit Schnellaktionen |
| Astrometry + PCC | History + Tools | Tool-Bereich im selben Kontext wie Run-Historie |
| Stats | Run Monitor | Operative Report-Erzeugung ohne Screenwechsel |
| Live log | Run Monitor (und optional Drawer) | Logfilter, Tail, Event-Highlights, direkt gekoppelt mit Stats |

## Visuelle Leitplanken

- Heller, kontrastreicher "Observatory"-Look (teal/copper Akzente, keine Purple-Lastigkeit).
- Konsistente Kartenkomponenten, grosse klickbare Ziele, klare Statuschips.
- Typografie: Serif fuer Orientierungstitel, Sans fuer Bedien- und Formtexte.
- Desktop-Optimierung: Layout-Baseline mindestens `1920x1080` (nicht fuer 1440 als Primarziel ausgelegt).
- Grossflaechige Mehrspaltenlayouts, dauerhafte Sidebar, hohe Informationsdichte.
- i18n-Regel: Sprache ist globale Session-Einstellung mit sofortigem UI-Refresh.
