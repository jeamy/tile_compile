# M42: Schärfe- und Dynamik-Analyse (2026-09-19)

Zusammenfassung der Untersuchung zu zwei beobachteten Problemen auf der 610-Frame-M42-DWARF-II-Serie (`/media/tc_ssd/M42_02.2026_lights_all`): (1) ein Schärfeunterschied zwischen zwei Commits, (2) eine wahrgenommene Dynamik-/Kontrastlücke gegenüber dem DWARF-II-Onboard-Livestack. Vollständiger Report mit Zahlen, Crop-Vergleichen und allen Zwischenschritten: [M42 Run-Vergleich (Artifact)](https://claude.ai/artifact/K2zaNdFPcS26yZTmfw8vRN).

## Ausgangslage

Zwei reale Runs (`run_20260916_154152`, Commit `ababbfba9`/tatsächlich `b9f6ccc8`, und `m42-canon-c1_20260919_084906`, Commit `083401c1c`) zeigten einen scheinbar großen Schärfeunterschied (Laplace-Varianz −71%) und der neuere Run wirkte gegenüber dem DWARF-II-Onboard-Stack (derselbe Beobachtungsabend, dieselben Rohframes) deutlich kontrastärmer/rauschiger.

## Methodik

Alle Befunde sind **kontrolliert**, nicht aus dem ursprünglichen, konfundierten A/B-Vergleich abgeleitet:

- **Commit-Isolation**: alter Commit-Stand in eigenem Git-Worktree gebaut (CUDA 12.9 + g++-13.4, da die aktuelle Systemtoolchain für den alten CMake-Pin zu neu ist), gegen exakt identische Config wie der neue Commit gerechnet — auf 30 Test-Frames und auf der realen 610-Frame-Serie.
- **Stage-Isolation**: `outputs/stacked_rgb_pcc.fits` zwischen zwei Runs vertauscht und einzeln per `resume-reconstruction --from-phase HYPERMETRIC_STRETCH` neu gerechnet (2×2-Kreuztest), um HMS als Ursache von Reconstruction/PCC zu trennen.
- **Referenz**: DWARF-II-Onboard-Stack (`DWARF_RAW_TELE_M42_..._2026-02-13.../stacked-16_....fits`), dieselbe Session, dieselben Rohframes — als externe, nicht direkt vergleichbare (andere Pipeline) aber aussagekräftige Referenz.
- Alle finalen Behauptungen wurden nach einer ersten, falschen rein-metrikbasierten Bewertung **zusätzlich visuell geprüft** (siehe Korrektur unten).

## Befund 1: Schärfe-Differenz zwischen Commits → `soft_floor()` in HMS

Der 2×2-Kreuztest zeigt eindeutig: Schärfe- **und** Rauschdifferenz folgen ausschließlich dem HMS-Binary, nicht dem PCC-Input. `chroma_denoise` und `shared_frame_rejection` sind als Ursache ausgeschlossen (`stacked_rgb.fits` vor PCC/HMS ist zwischen den Commits bytegleich).

**Mechanismus**: Der bekannte Dark-Speckle-Fix in `hypermetric_stretch.cpp` ersetzt `max(v - anchor, 0)` durch `soft_floor(v, anchor, eps=0.00025)`. Der Anchor wird an anderer Stelle bewusst als `floor - 0.00025` berechnet — **exakt dieselbe Konstante**. Für die meisten Himmelspixel gilt damit `v - anchor ≈ eps`, sie landen also systematisch genau im Bereich der stärksten Glättungswirkung von `soft_floor`. Der Fix wirkt dadurch nicht nur auf seltene Schwarz-Ausreißer, sondern faktisch als milder Weichzeichner über die gesamte Hintergrundfläche.

**Einordnung**: Vermutlich keine reine Regression — ein rauschärmerer Hintergrund ist erwünscht. Laplace-Varianz/Tenengrad können aber nicht zwischen „verlorenem Strukturdetail" und „entferntem Rauschen" unterscheiden; ein Teil des gemessenen „Schärfeverlusts" ist wahrscheinlich echte Rauschunterdrückung.

## Befund 2: Dynamik-Lücke zu DWARF → fehlende Rauschunterdrückung

DWARF's Hintergrund ist ~4–7× rauschärmer (relatives Rauschen, Std/Mittelwert in einem sternarmen Himmelspatch): DWARF 4.9% vs. `tile_compile` 21.7–36.5% je nach Commit. Ein Rauschkorrelations-Test (Std bei k×k-Pixel-Mittelung) zeigt: DWARF's Verhältnisse liegen nahe 1.0 (kaum Reduktion durch Mittelung) — ein klares Signal für aktive räumliche Rauschunterdrückung in der Firmware, nicht nur eine andere Tonwertkurve.

**Direkt getestet**: `luma_denoise` mit Default-Stärke aktiviert senkt das relative Rauschen bereits von 21.8% auf 17.3%, und die Korrelations-Signatur nähert sich DWARF's Muster deutlich an.

## Konfigurations-Tuning: harte Grenze gefunden

Fünf Config-Iterationen (`blend_amount`, `wavelet.levels/threshold_scale` bis Schema-Maximum, `structure_protection` aus, `convergence_power`) senken das relative Rauschen nur bis auf ein Plateau bei ~16%. Code-Ursache identifiziert: Die Wavelet-Rekonstruktion in `luma_denoise.cpp` addiert ihre gröbste Gaußsche-Pyramide-Approximationsebene immer **unverändert** zurück — dieser Rest trägt reales Restrauschen, das keine Wavelet-Stärke entfernt, auch nicht bei `levels=8` (Schema-Maximum).

## Umgesetzte Code-Änderung: `luma_denoise.bilateral` + `extended_source_protection`

Zwei neue, von `chroma_denoise` portierte Bausteine (beide **Default: aus**):

- **`luma_denoise.bilateral`**: kantenerhaltender Bilateral-Filter nach der Wavelet-Stufe, adressiert genau den oben identifizierten strukturellen Rest.
- **`luma_denoise.extended_source_protection`**: schützt breitflächige, kontrastarme aber reale Nebelstruktur (wie bei `chroma_denoise` bereits vorhanden). War der eigentlich fehlende Baustein — ohne ihn hatten Wavelet+Bilateral den Nebelkern-Laplace-Wert von 0.0108 auf ~0.0035 einbrechen lassen (echter Detailverlust). Mit aktivierter Maske: Erholung auf 0.0084.

Vollständig eingebunden (Config-Struct, YAML-Parser/-Writer, Validierung, beide Schema-Dateien, Default-Config, DE/EN-Doku, GUI-i18n, neue Regressionstests). Volle Testsuite: 466 Testfälle, 1.386.683 Assertions, alle grün.

**Validiertes Ergebnis**: Hintergrundrauschen sinkt real von 21.82% auf 16.67–16.79%.

## Korrektur nach visueller Prüfung (wichtig)

Die erste „finale" Empfehlung (`bilateral` + `extended_source_protection` + `highlight_ceiling_percentile: 99.5`) wurde nur anhand von Kennzahlen bewertet — ein Fehler. Direkte Bildkontrolle ergab:

1. **`highlight_ceiling_percentile: 99.5` brennt den Trapez-Kern sichtbar aus** (3.21% der Kern-Crop-Fläche hart auf Weiß geclippt, ~11.600 Pixel). Zurückgesetzt auf den sicheren Default `100` (0.00% Clipping). Auch sanftere Werte (99.8–99.95) clippen noch messbar (0.22–1.18%) — für dieses Ziel gibt es keinen brauchbaren Wert unter 100.
2. **Die Hintergrundrauschen-Verbesserung ist real, aber räumlich eng begrenzt**: In einem reinen Sternfeld-Himmelsausschnitt klar sichtbar weniger Korn. Im **nebeldominierten Kern-Ausschnitt** — dem Bereich, an dem die Bildqualität tatsächlich beurteilt wird — ist der Effekt praktisch nicht wahrnehmbar, weil dort kaum leerer Himmel vorkommt.

**Fazit**: Das ursprüngliche Problem („DWARF's Kern-/Nebelregion wirkt dynamischer") ist durch die umgesetzte Denoise-Erweiterung **nicht gelöst** — sie löst ein reales, aber engeres Problem (Himmelshintergrund-Rauschen).

## Fünf gescheiterte Ansätze für Kern-/Nebelkontrast

Direkte ADU-Messung im linearen `stacked_rgb.fits` (Nutzer-Analyse) lieferte die entscheidende Erklärung: Hintergrund 26.81 ADU, schwache Nebelschwaden 29.21 ADU (+2.40 ADU), hellerer Nebel 32.93 ADU (+6.12 ADU), robustes Hintergrundrauschen 1.64 ADU. **Schwache Strukturen liegen damit nur bei 1.47× Rauschsigma, hellerer Nebel bei 3.74×** — Signal und Rauschen sind pro Pixel nicht sauber trennbar.

Das erklärt, warum fünf verschiedene Versuche, den Kern-/Nebelkontrast gezielt anzuheben, scheiterten:

| # | Ansatz | Ergebnis |
|---|---|---|
| 1 | `highlight_ceiling_percentile` senken | Kern clippt sichtbar (siehe oben) |
| 2 | Neue `hypermetric_stretch.local_contrast`-Stufe, uniformer Boost | Kernstruktur +4–5×, aber Hintergrundrauschen +130% (21.8%→49.6%) |
| 3 | Dieselbe Stufe, Rauschschwellenwert-gegated | Kein sauberer Punkt — zu strikt: kaum Effekt; zu locker: Rauschen leckt durch |
| 4 | Dieselbe Stufe, räumlich gegated (Nebel-Silhouette-Maske) | Hintergrund bleibt exakt auf Denoise-Niveau (16.67%), aber **innerhalb** der Nebelregion nur verstärktes Korn, keine saubere Struktur |
| 5 | `luma_denoise.wavelet.boost` (Verstärkung oberhalb der Rauschschwelle pro Wavelet-Ebene) | Sichtbare dunkle/grüne Ringartefakte um helle Sterne |

**Grund**: Bei diesem Signal-Rausch-Verhältnis gibt es keine per-Pixel- oder Nachbarschafts-basierte Methode (Schwellenwert, Maske, Wavelet-Skala), die Signal und Rauschen sauber trennt — beide überlappen zu stark in Amplitude und Ortsfrequenz für eine skalare Gain-Funktion.

Beide neuen Felder (`hypermetric_stretch.local_contrast`, `luma_denoise.wavelet.boost`) sind im Code vorhanden, aber standardmäßig wirkungslos (`enabled: false` / `boost: 0.0`) — nicht für den produktiven Einsatz empfohlen.

## Offene Punkte / was tatsächlich nötig wäre

Um die volle Lücke zu DWARF zu schließen (Hintergrund 16.7%→4.9%, plus echten Kern-/Nebelkontrast), reicht kein Parameter- oder Ad-hoc-Algorithmus-Tuning mehr aus. Kandidaten für einen größeren Eingriff:

- **Echte Multi-Frame-Rauschunterdrückung** über die ursprünglichen 610 Einzelframes (zeitliche statt räumliche Redundanz) statt Einzelbild-Nachbearbeitung.
- **Kanten-/struktur-tensor-bewusstes Verfahren**, das Sternkanten gesondert behandelt, um Ringing zu vermeiden.
- Akzeptieren, dass DWARF's Firmware für eine ansprechende Sofort-Vorschau reale Signaltreue opfert, während `tile_compile` durchgängig auf Signalerhalt ausgelegt ist (additive statt Ratio-Rekonstruktion, Schutzmasken überall) — ein Teil der Lücke ist vermutlich ein bewusster Zielkonflikt, kein Defizit.

Nicht mehr offen: `weight_exponent_scale` und `shared_frame_rejection` wurden im ursprünglichen (konfundierten) A/B-Vergleich als Nebendarsteller identifiziert, aber durch den kontrollierten Commit-Test als Haupttreiber ausgeschlossen; eine dedizierte Einzel-Isolation ist angesichts der jetzt bekannten HMS-Ursache niedrige Priorität.
