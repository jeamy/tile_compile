# PI Jev M4 — Erste Bewertung an M31 und M42: sind die Empfehlungen verwertbar?

> **Stand:** 2026-09-23. **Art:** explorative Auswertung, keine Freigabe. Reproduzierbar mit
> `web_backend_cpp/tools/pi_decisions_eval.cpp` (Ziel `pi_decisions_eval`) und
> `agent_service/scripts/jev_eval_call.ts` (echter Aufruf über denselben `DecisionsService` wie der Sidecar).
> **Kein Rekonstruktions-Run wurde gestartet, kein Backend/Sidecar-Prozess.** Genutzt wurden nur
> `tile_compile_cli scan`/`scan-metrics` (lesend), die vorhandenen `config.yaml`/`global_metrics.json` der
> Runs `M31-20260921_053613_a1f5c7de` und `m42_20260922_211255` sowie der Live-Endpunkt (5 Aufrufe je Fall,
> ca. 0,00014 USD pro Aufruf, gesamt unter 0,01 USD).

## 1. Ergebnis in einem Satz

Auf diesen beiden Datensätzen gibt es **keine verwertbare Empfehlung mit solider Basis**: Mit der echten
Config ist nichts vorzuschlagen, und im Gegenfall (Kandidat erzwungen) antwortet Jev stabil `keep_current`,
gestützt auf eine Evidenz (relative Streuung), die den Effekt der adaptiven Gewichtung nachweislich **nicht**
vorhersagt. Der Nutzen der bisherigen Umsetzung liegt in der Absicherung (Atomarität, Locks, Enthaltung,
Reproduzierbarkeit), nicht in der Qualität der Empfehlung.

## 2. Aktuelle Config gegen Vorschlag

| | M31 (645 Frames, Gain 80) | M42 (610 Frames, Gain 60) |
|---|---|---|
| `global_metrics.adaptive_weights` in der echten Run-Config | `true` | `true` |
| Kandidat `enable_adaptive_weights` | ausgeschlossen: `already_active` | ausgeschlossen: `already_active` |
| Angebotene Kandidaten | `keep_current`, `insufficient_evidence` | dito |
| Modellaufruf | keiner (nichts zu entscheiden) | keiner |
| Vorschlag | **unverändert** (`no_change`) | **unverändert** (`no_change`) |
| Unterschied Vorschlag / aktuelle Config | keiner | keiner |

Ohne eingefrorene Schwellen (Produktionsstand) wäre der Kandidat ohnehin mit `policy_thresholds_not_frozen`
ausgeschlossen. Beide Konfigurationen setzen die adaptive Gewichtung also schon; die Standardvorgabe im Code
(`false`) ist hier irrelevant.

## 3. Gegenprobe: Entwurf mit `adaptive_weights: false`

Nur um Jevs Verhalten zu prüfen, wurde der Entwurf künstlich auf `false` gesetzt und die Schwellen für dieses
Experiment auf `min_quality_spread=0.001`, `min_measurement_coverage=0.9` gestellt (willkürlich, siehe 6).
Der Kandidat war dann angeboten; je 5 Aufrufe, Modell `typesafe/jev-1.13-20260917`:

| Fall | gewählt | P(keep_current) | P(insufficient_evidence) | P(enable_adaptive_weights) | Provider-`confidence` |
|---|---|---|---|---|---|
| M31, echte Zahlen | 5x `keep_current` | 0,86-0,88 | 0,08-0,09 | 0,04-0,05 | 0,79-0,82 |
| M42, echte Zahlen | 5x `keep_current` | 0,85-0,86 | 0,08-0,09 | 0,05-0,06 | 0,77-0,79 |
| synthetisch: alle Streuungen 0 | 5x `keep_current` | 0,84-0,87 | 0,09-0,12 | 0,03-0,04 | 0,77-0,81 |
| synthetisch: Streuungen 0,35 | 5x `keep_current` | 0,75-0,80 | 0,11-0,17 | 0,08-0,11 | 0,62-0,70 |

Lesart: stabil (die Wahl ändert sich in keinem der 20 Aufrufe), und die Wahrscheinlichkeit für "aktivieren"
steigt mit der Streuung in die plausible Richtung (0,03 -> 0,05 -> 0,10). Aber Jev empfiehlt es in **keinem**
Fall, auch nicht bei extrem uneinheitlichen Frames. Ein Modell, das bei jeder Eingabe dasselbe wählt, liefert
in der Praxis keine Empfehlung, sondern eine Bestätigung des Ist-Zustands.

## 4. Trägt die Evidenz? Nein — die Streuung sagt den Effekt nicht voraus

Die Pre-Rules nutzen als Beleg die relative Streuung (MAD/Median) von `fwhm` und `noise` einer Gruppe. An den
echten Runs lässt sich prüfen, was die adaptive Gewichtung tatsächlich bewirkt hat (`global_metrics.json`,
Feld `global_weight` je Frame):

| | M31 | M42 |
|---|---|---|
| Scan: relative Streuung `fwhm` / `noise` | 0,0048 / 0,0084 | 0,0115 / 0,0212 |
| Gewichte im Run: min / Median / max | 0,26 / 1,11 / 10,2 | 0,03 / 1,13 / 10,0 |
| Variationskoeffizient der Gewichte | 0,92 | 0,84 |
| effektive Stichprobe n_eff / N | **0,54** | **0,59** |

Die Frames sind nach der Streuung fast identisch, die Gewichtung ist aber stark ungleich (fast die Hälfte der
Frames trägt effektiv nichts bei). Der Grund: die Gewichte hängen nicht an der Größe der Streuung, sondern an
der Übereinstimmung der Metriken (`leave_one_out_positive_correlation_squared` mit robuster Normierung).
Diese Übereinstimmung ist im Scan hoch:

| Rangkorrelation (Spearman), Scan-Metriken je Frame | M31 | M42 |
|---|---|---|
| background - noise | 0,93 | 0,97 |
| background - fwhm | 0,88 | 0,75 |
| noise - fwhm | 0,83 | 0,73 |
| Scan-Hintergrund vs. Run-Gewicht | -0,89 | -0,90 |
| Scan-Rauschen vs. Run-Gewicht | -0,84 | -0,92 |

Kohärente Qualitätsunterschiede sind vorhanden (Hintergrund/Rauschen korrelieren fast perfekt, und die
Run-Gewichte folgen ihnen), nur klein in absoluten Zahlen. Die bisherige Evidenz (`quality_spread`) blendet
genau das aus. Jev bekommt also eine Zahl, die für die Entscheidung kaum etwas aussagt, und antwortet
konsequent konservativ.

Nebenbefund: `fwhm` aus dem Scan und `fwhm` aus den Run-Metriken sind praktisch unkorreliert (Spearman 0,05 bzw.
0,04) und haben andere Skalen (Median Scan 9,6 / 9,1 gegen Run 14,2). Das sind verschiedene Messverfahren
(im Zielbild bereits vermerkt); der Scan-FWHM taugt nicht als Stellvertreter für die spätere Registrierungsqualität.

## 5. Bewertung: verwertbar? solide Basis?

- **Als Empfehlung (Qualitätsgewinn): nein.** Es gibt keine Wahrheit, gegen die man sie prüfen könnte: weder für
  M31 noch für M42 existiert ein gepaarter Lauf mit/ohne adaptive Gewichtung auf denselben Frames. Ohne diesen
  Vergleich ist jede Aussage "besser" oder "schlechter" unbelegt, ganz gleich, wie sicher das Modell klingt.
- **Als Bestätigung des Ist-Zustands: ja, aber trivial.** `keep_current` stimmt hier, weil die Optimierung
  schon aktiv ist — das erkennt schon die deterministische Vorbedingung, ohne Modell.
- **Als Schutzgerüst: ja.** Atomarer Kandidat, Locks, geschützte Pfade, Enthaltung, Staleness, Hashes, keine
  Pfade/Geheimnisse zum Provider, reproduzierbare Ablage, und ein Vorschlag, der nie ohne Nutzer in einen Run geht.
- **Konsistenz und Kosten:** stabil über 20 Aufrufe, Kosten vernachlässigbar (ca. 0,00014 USD).

## 6. Grenzen dieser Auswertung

- Zwei Datensätze, gleiche Kamera-Familie; die Schwellen (0,001 / 0,9) sind für das Experiment gewählt, nicht
  begründet. Mit einer anderen Schwelle würde sich nur ändern, ob der Kandidat angeboten wird, nicht Jevs Antwort.
- Die synthetischen Varianten ändern nur die Streuungsfelder des State, nicht die übrigen Größen.
- Es wurde kein Qualitätsmaß der Rekonstruktion verglichen (kein Lauf gestartet, siehe Kopf).

## 7. Konsequenzen für die Planung (Vorschläge, noch nicht umgesetzt)

1. **Evidenz ersetzen:** statt Streuung die Metrik-Übereinstimmung (Rangkorrelationen und daraus abgeleitet eine
   erwartete n_eff/N) als Beleg für `enable_adaptive_weights` definieren; sie ist aus dem Scan berechenbar und
   erklärt die beobachteten Gewichte. Erst danach wären Schwellen begründbar.
2. **Wahrheit schaffen (M5):** gepaarte Läufe mit/ohne Option auf denselben Frames, Sternform an gematchten
   Positionen, Rauschen und Abdeckung — nur mit ausdrücklichem Auftrag für Läufe.
3. **Erst dann klären, ob ein Modell nötig ist:** wenn eine deterministische Regel aus Metrik-Übereinstimmung
   dasselbe leistet, ist Jev für diesen Kandidaten überflüssig. Sinnvoller wäre Jev bei Entscheidungen mit
   mehreren gleichwertigen Alternativen und gemischten Belegen, für die es keine einfache Regel gibt.
4. **Weitere Kandidaten** einzeln nach dem Erweiterungsverfahren des Regelkatalogs; bis dahin bleibt die
   Produktionseinstellung bei den Baselines (Schwellen nicht eingefroren).

## 8. Nachtrag M5 (Teil 1): Metrik-Übereinstimmung im State

Stand 2026-09-23. Der State enthält je Gruppe `metric_agreement`: Spearman-Rangkorrelation für `background~noise`,
`background~fwhm`, `noise~fwhm` und deren Median (nur Frames, in denen beide Metriken gültig sind; `fwhm<=0` wird
ausgeschlossen; unter `min_valid_for_spread` Frames `not_applicable`; konstante Metrik `invalid/constant_metric`, nie 0).
Die Zahlen gehen in die Provider-Projektion. Die Evidenz des Kandidaten `enable_adaptive_weights` (jetzt Version 2)
ist `metric_agreement` statt `quality_spread`; die Schwelle `min_metric_agreement` bleibt ungefroren.

Offline aus denselben Scan-Metriken (kein Run, kein Provider-Aufruf):

| | M31 | M42 |
|---|---|---|
| background~noise | 0,926 | 0,966 |
| background~fwhm | 0,877 | 0,754 |
| noise~fwhm | 0,827 | 0,728 |
| Median | 0,877 | 0,754 |

Die C++-Werte stimmen mit der unabhängigen Python-Auswertung in Abschnitt 4 überein. Noch offen: Eine Schwelle ist
damit nicht begründet (zwei Datensätze, gleiche Kamera, kein gepaarter Lauf), und eine "erwartete n_eff/N" wurde
bewusst nicht implementiert, solange sie nicht gegen das Gewichtungsverfahren des Runners validiert ist.

## 9. Gepaarter Lauf M31 mit/ohne adaptive Gewichtung (M5, Teil 2)

Stand 2026-09-23. Zwei vollständige `reconstruct`-Läufe auf denselben 645 Frames (`/media/tc_ssd/M31_ligths_all`) mit
demselben Binary; die Configs unterscheiden sich nur in `global_metrics.adaptive_weights` (A: `true`
`20260923_204346_5a30a6f0`, B: `false` `20260923_212703_e8a50695`). Die Auswertung war rein lesend.

| | A (an) | B (aus) |
|---|---|---|
| Gewichte min / Median / max | 0,26 / 1,11 / 10,2 | 0,40 / 1,08 / 3,95 |
| n_eff / N | 0,54 | 0,83 |
| Median-FWHM raw (px), 95%-KI | 3,757 [3,726-3,782] | 3,750 [3,728-3,782] |
| Elongation raw | 1,1222 | 1,1204 |
| Hintergrund-RMS raw | 0,734 | 0,718 |
| Rangeschwanz (tail) | 0,480 | 0,481 |
| gewählter Kandidat | `drizzle_raw` | `drizzle_raw` |
| robustes Rauschen im Endbild (rMAD), R/G/B | 1,645 / 1,572 / 1,576 | 1,606 / 1,542 / 1,537 |

`adaptive_weights: false` heißt nicht gleichgewichtet: die statischen Gewichte bleiben aktiv (n_eff/N 0,83).
Die Schärfe ändert sich nicht messbar (Differenz 0,17 % innerhalb der Konfidenzintervalle), die adaptive Variante
hat etwa 2,4 % mehr Hintergrundrauschen, passend zur geringeren effektiven Stichprobe. Fluss der hellsten 1 % Pixel
stimmt auf 0,05 % überein. **Auf M31 gibt es keinen belegten Nutzen der adaptiven Gewichtung.**

Grenzen: ein Datensatz, je ein Lauf pro Arm (kein Wiederholungslauf, also keine Run-zu-Run-Grundstreuung);
die Stern-Stichproben der Validierung sind nicht identisch (247 gegen 248 Sterne), das Ergebnis ist deshalb kein
Vergleich an gematchten Positionen im Sinne der Projektregel; M42 wurde nicht gepaart gelaufen.

## 10. Gepaarte Läufe M42 und IC5070, Jev-Abfragen für vier Datensätze (M5, Teil 3)

Stand 2026-09-24. Gleiches Vorgehen wie Abschnitt 9 (gleiches Binary, nur `global_metrics.adaptive_weights` verschieden,
Auswertung lesend). A = Gewichtung an, B = aus.

| | M42 A / B (610 Frames) | IC5070 A / B (466 Frames) |
|---|---|---|
| Run-IDs | `20260923_223614_ce66cf2d` / `20260923_232436_95ed29f7` | `20260924_001226_8c66f570` / `20260924_004750_bbc7b433` |
| Gewichte min / Median / max | 0,027 / 1,13 / 9,99 gegen 0,12 / 1,01 / 4,27 | 0,027 / 1,05 / 4,70 gegen 0,027 / 0,95 / 2,93 |
| n_eff / N | 0,59 / 0,80 | 0,64 / 0,77 |
| Median-FWHM raw (px) | 3,969 / 3,957 (KI überlappen) | 4,412 / 4,403 (KI überlappen) |
| Elongation raw | 1,0365 / 1,0342 | 1,1238 / 1,1246 |
| Hintergrund-RMS raw | 1,988 / 1,949 | 1,848 / 1,848 |
| Rauschen Endbild rMAD (R/G/B) | 4,63/3,56/4,00 gegen 4,53/3,49/3,92 (A etwa +2 %) | 4,43/3,47/3,91 gegen 4,43/3,48/3,91 (gleich) |
| gewählter Kandidat | `drizzle_raw` / `drizzle_raw` | `drizzle_raw` / `drizzle_raw` |

**Jev** (je 5 Aufrufe, Kandidat `enable_adaptive_weights` durch Setzen des Entwurfs auf `false` angeboten, Testschwellen
`min_measurement_coverage=0,9`, `min_metric_agreement=0,5`, willkürlich; Modell `typesafe/jev-1.13-20260917`; Kosten gesamt 0,0031 USD):

| Datensatz | Median-Übereinstimmung | Wahl | P(keep_current) | P(enable) |
|---|---|---|---|---|
| M31 | 0,877 | 5x `keep_current` | 0,86-0,88 | 0,05-0,06 |
| M42 | 0,754 | 5x `keep_current` | 0,85-0,89 | 0,05-0,08 |
| M66 | 0,685 | 5x `keep_current` | 0,78-0,85 | 0,07-0,12 |
| IC5070 (Schwelle 0) | 0,330 | 5x `keep_current` | 0,84-0,86 | 0,06-0,09 |

Mit Schwelle 0,5 wird der Kandidat bei IC5070 vor dem Modell ausgeschlossen (`evidence_below_threshold:metric_agreement`).

**Lesart:** Auf M31, M42 und IC5070 bringt die adaptive Gewichtung keinen messbaren Schärfegewinn; bei M31 und M42 kostet
sie etwa 2 % Rauschen, bei IC5070 nichts. Jevs stetiges `keep_current` deckt sich damit. Die Übereinstimmung der Metriken
unterscheidet die Datensätze (0,33 bis 0,88), sagt aber keinen Nutzen voraus: eine Schwelle ist auch mit vier
Datensätzen nicht begründbar. Grenzen wie in Abschnitt 9 (ein Lauf je Arm, kein Wiederholungslauf, keine gematchten
Sternpositionen).

### 10.1 M66 (975 Frames)

A (Gewichtung an, `20260924_013435_a39a9e30`) gegen B (aus, `20260924_040232_e0b8cac7`), gleiche Bedingungen.

| | A | B |
|---|---|---|
| Gewichte min / Median / max | 0,027 / 1,19 / 5,29 | 0,027 / 0,96 / 3,39 |
| n_eff / N | 0,63 | 0,73 |
| Median-FWHM raw (px), 95%-KI | 5,820 [5,537-6,052] | 5,779 [5,616-6,155] |
| Elongation raw | 1,0925 | 1,0881 |
| Hintergrund-RMS raw | 1,119 | 1,103 |
| tail | 0,267 | 0,230 |
| Rauschen Endbild rMAD (R/G/B) | 2,22/1,88/2,09 | 2,19/1,85/2,05 (A etwa +1,4 %) |
| gewählter Kandidat | `drizzle_raw` | `drizzle_raw` |

Schärfe: keine messbare Änderung (0,7 %, Konfidenzintervalle überlappen breit). Rauschen: A etwa 1,4 % höher.
**Auffällig:** der Fluss der hellsten 1 % Pixel unterscheidet sich um +9 bis +16 % (A/B: 1,16 / 1,09 / 1,15), bei
M31/M42/IC5070 lag er unter 1,3 %. Die Ursache wurde nicht untersucht (mögliche Gründe: die Gewichte verschieben die
Anteile heller/dunkler Frames bei einem Objekt mit sehr hellem Kern, oder die Validierungs- und Supportmasken
unterscheiden sich); nicht als Qualitätsaussage lesen. Zusammenfassend bleibt es bei allen vier Datensätzen dabei: kein
belegter Vorteil der adaptiven Gewichtung, ein Rauschnachteil von 0 bis 2 %.

**Betrieb (nicht Ergebnis):** Die Geometrie-Vorprüfung der Forward-Drizzle-Phase (`runner_forward_drizzle.cpp`) verlangt
das 1,5-fache von etwa 118 GiB frei; bei M66 mit 975 Frames braucht ein Lauf beim Start rund 250 GB freien Platz
(kalibrierte Frames und Normalisierungs-Cache liegen dann schon auf dem Datenträger).
