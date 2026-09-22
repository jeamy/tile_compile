# PI Jev M0 — Feld-Inventar und Quellenverifikation

> **Stand:** 2026-09-22.
> **Status:** M0-Teilergebnis. Deckt den ersten Checklistenpunkt aus
> [Implementierungsplan Abschnitt 4](pi_jev_implementierungsplan_de.md#4-m0--vertr%C3%A4ge-quellen-und-testgrundlage-einfrieren)
> ab: Quellen und Fehlend-Verhalten pro verwendetem Feld. Die maschinenlesbaren
> Gegenstücke liegen unter `web_backend_cpp/config/pi_decisions/`
> (`protected_paths_v1.json`, `candidates_v1.json`, `schemas/*.schema.json`).

Jede Zeile wurde gegen den tatsächlichen Code verifiziert (Pfad + Zeile/Symbol
in der Spalte Quelle), nicht aus der Schema-Description übernommen.

## 1. Scan-Frame-Felder (`tile_compile_cpp/apps/cli_main.cpp`)

| Feld | Quelle | Einheit | Fehlend-Verhalten |
|---|---|---|---|
| `background` | `cli_main.cpp` Scan-Ergebnisstruktur, `fr.background` | Aufnahme-ADU (roh, unkalibriert) | Frame mit `ok=false` liefert kein Feld; `status=missing` im State |
| `noise` | `fr.noise` | ADU | wie oben |
| `gradient_energy` | `fr.gradient_energy` | dimensionslos | wie oben |
| `sky_gradient` | `fr.sky_gradient` | Verhältnis zum Hintergrund | wie oben |
| `fwhm` | `fr.fwhm`, aus `sm.fwhm` (`metrics.cpp`, `measure_frame_star_metrics`) | Pixel | `<= 0` wird beim Aggregieren verworfen (`if (fr.fwhm > 0) all_fwhm.push_back(...)`), nicht als 0 gezählt |
| `roundness` | `fr.roundness = sm.roundness` (`metrics.hpp:17`: `fwhm_y / fwhm_x`) | Verhältnis, > 0, kann > 1 sein | `<= 0` wird verworfen, gleiche Regel wie `fwhm` |
| `star_count` | `fr.star_count` | Anzahl | immer vorhanden bei `ok=true`, auch `0` |
| `exposure_seconds` | Header `hi.exptime`, nur wenn `>= 0` | Sekunden | fehlt im JSON, wenn Header-Feld ungültig/nicht vorhanden — kein `0`-Default |
| `gain` | Header `hi.gain`, nur wenn `>= 0` | Kamera-Rohwert | wie oben |
| `temperature_c` | Header `hi.ccd_temp`, nur wenn `> -900` | °C | wie oben (Sentinel `-900`, kein `0`) |
| `filter` | Header `hi.filter`, nur wenn nicht leer | Freitext | fehlt im JSON statt leerem String |
| `target` | Header `hi.object` | Freitext | wie oben |
| `camera` | Header `hi.instrume` | Freitext | wie oben; **kein** Substring-Mapping auf Sensor-ID (siehe Pre-Rules 4.9) |
| `ra_deg`/`dec_deg` | Header, nur wenn `> -900` | Grad | Sentinel `-900` |

Aggregation (`agg_stats` in `cli_main.cpp`): `{min, max, mean, median, p10, p90, count}`
über die jeweils gültige Teilmenge. Leerer Vektor liefert `null`, nicht `0`.
`roundness` wird laut Aggregat als Rohwert geführt; die in
`pi_scan_pre_rules_de.md` §2 vorgeschlagene symmetrische Achsenabweichung
(`abs(log(roundness))`) existiert noch nicht im Scan-Code und muss in M1 neu
berechnet werden, nicht aus dem bestehenden Aggregat abgeleitet.

## 2. Bestehende Verträge, die unverändert bleiben

| Vertrag | Quelle | Bezug zu Jev |
|---|---|---|
| `pi.feature-vector.v1` | `web_backend_cpp/src/services/pi/pi_feature_vector.cpp`, `build_scan_feature_vector()` | kNN-Shadow-Pfad, separat von `pi.decision-state.v1`; Jev-State projiziert dieselben Rohquellen zusätzlich, ändert aber nicht diesen Vertrag |
| kNN-Distanz/Vorhersage | `pi_param_model.cpp`, `predict_param_nn()` | bleibt Shadow; `available=true` ist laut Zielbild §1 kein Vorrangkriterium gegenüber Jev |
| Lokales Outcome-/Promotion-System | `docs/PI/pi_local_learning_plan_de.md` §5/§7, `pi_outcome_recorder.cpp::record_run_outcome_if_needed` | **Offener Punkt, siehe Abschnitt 4 unten** — noch nicht mit dem Jev-Proposal-Outcome-Pfad reconciled |
| Recommendation-Validator | `pi_recommendation_validator.cpp`, `validate_recommendation_updates()` | bestehender Teilpatch-fähiger Pfad bleibt für PI unverändert; Jev braucht laut M2 einen expliziten *atomaren* Modus obendrauf, keinen Ersatz |

## 3. Geschützte Pfade (`guard_quality` / `guard_scope`)

Siehe `web_backend_cpp/config/pi_decisions/protected_paths_v1.json` — jeder
Eintrag trägt dort den exakten dotted path plus die Codestelle, gegen die er
verifiziert wurde (`configuration.hpp`, `config.cpp`). Nicht erneut in Prosa
dupliziert, um Drift zwischen Doku und Katalog zu vermeiden.

## 4. Offen: Verhältnis zum lokalen Lernsystem

`pi_jev_implementierungsplan_de.md` §2 listet `pi_outcome_recorder.cpp` als
Eingriffspunkt ("Vorschlag -> tatsächliche Config -> Ergebnis verknüpfen,
Labels sauber trennen"). Das lokale Lernsystem
(`docs/PI/pi_local_learning_plan_de.md`) betreibt in derselben Datei bereits
`record_run_outcome_if_needed()` für sein eigenes `PiMemoryStore`/
Auto-Promotion-Zählverfahren (`N >= 3` positive `quality_delta`-Outcomes).

Vor M2 zu klären, bevor `pi_outcome_recorder.cpp` angefasst wird:

1. Schreibt ein übernommener Jev-Vorschlag einen `PiMemoryStore`-Kandidaten
   (und zählt damit potenziell zur Auto-Promotion-Schwelle des lokalen
   Modells), oder bleibt sein Outcome in einem getrennten Store
   (`pi_decisions/<proposal_id>/...`, Implementierungsplan §3.6) komplett
   isoliert?
2. Falls getrennt: Reicht die reine räumliche Trennung (zwei Verzeichnisse),
   oder braucht `record_run_outcome_if_needed()` einen expliziten
   `source`-Diskriminator (`"pi_recommendation" | "jev_proposal" |
   "local_model"`), damit ein künftiger Auswerteschritt nicht versehentlich
   über Quellen hinweg mittelt?
3. Ist Jev als Zwischenschritt gedacht, bis das lokale Modell laut
   Lernplan-Schritt 7 "scharf geschaltet" wird, oder als dauerhaft parallele
   zweite Quelle (aktuelle Nutzerentscheidung, 2026-09-22: **dauerhaft
   parallel**)? Damit ist Frage 3 beantwortet — Fragen 1/2 bleiben vor M2 zu
   klären, weil "dauerhaft parallel" die Vermischung der Outcome-Zähler noch
   wahrscheinlicher macht als ein Zwischenschritt-Szenario.

Dieser Abschnitt bleibt bis zur Klärung ein Blocker für den
`pi_outcome_recorder.cpp`-Teil von M2, nicht für M1 (State-Builder) oder M3
(Adapter), die beide unabhängig davon sind.
