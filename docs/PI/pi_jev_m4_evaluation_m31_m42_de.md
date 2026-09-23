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
