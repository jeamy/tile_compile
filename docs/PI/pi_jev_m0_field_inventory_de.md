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
| Lokales Outcome-/Promotion-System | `docs/PI/pi_local_learning_plan_de.md` §5/§7, `pi_outcome_recorder.cpp::record_run_outcome_if_needed` | **Entschieden, siehe Abschnitt 4**: Jev nutzt diesen Store und Recorder nicht, eigenes Modul `pi_decision_outcome` |
| Recommendation-Validator | `pi_recommendation_validator.cpp`, `validate_recommendation_updates()` | bestehender Teilpatch-fähiger Pfad bleibt für PI unverändert; Jev braucht laut M2 einen expliziten *atomaren* Modus obendrauf, keinen Ersatz |

## 3. Geschützte Pfade (`guard_quality` / `guard_scope`)

Siehe `web_backend_cpp/config/pi_decisions/protected_paths_v1.json` — jeder
Eintrag trägt dort den exakten dotted path plus die Codestelle, gegen die er
verifiziert wurde (`configuration.hpp`, `config.cpp`). Nicht erneut in Prosa
dupliziert, um Drift zwischen Doku und Katalog zu vermeiden.

## 4. Entschieden: Verhältnis zum lokalen Lernsystem (2026-09-23)

Die drei offenen Fragen (Outcome-Zähler, Diskriminator, Zwischenschritt vs.
parallel) sind gegen den Code geklärt. Vorgabe des Nutzers: Jev ist eine
**dauerhaft parallele, unabhängige** zweite Quelle.

### 4.1 Befunde im bestehenden Code (verifiziert)

| Befund | Stelle | Folge |
|---|---|---|
| Die einzige Quelle von `PiMemoryStore`-Kandidaten mit Run-Bezug ist der Scan-Apply: `type=config_optimization`, `source=scan_ai_apply`, `revision_id`, `config_sha256` | `ai_routes.cpp` `build_apply_candidate_memory()` (Z. 337ff) | Jev-Vorschläge dürfen diesen Builder nicht benutzen |
| Der Recorder joint Run -> Kandidat **ohne Quellenfilter**, primär über `prior_active_config_revision_id == item.revision_id`, sekundär über `config_sha256` | `pi_outcome_recorder.cpp`, `record_run_outcome_if_needed()` | Ein Jev-Eintrag im `PiMemoryStore` würde automatisch gejoint |
| Der Auto-Promotion-Zähler zählt alle Outcomes mit numerischem `quality_delta` an einem Kandidaten (`N >= 3`) | `PiMemoryStore::evaluate_auto_promotion()` | Quellen würden vermischt; heute nur Shadow und `quality_delta` immer `null` |
| Der Retrain-Export nimmt jedes Memory mit `type == config_optimization` **und** `status == accepted`, ohne Quellenprädikat | `scripts/pi_retrain_models.py`, `collect_scan_reference_points()` | Ein akzeptierter Jev-Eintrag würde kNN-Trainingspunkt |
| Die Run-Provenance trägt `config_revision_id`, `prior_active_config_revision_id`, `config_sha256`, `started_at` | `runs_routes.cpp:524`, reales Beispiel geprüft | Ausreichend für eine eigene Verknüpfung, ohne den Recorder zu ändern |
| Revision-Autor unterscheidet Quellen bereits (`pi_scan_ai`, `pi_action_plan`, `save_config`, `config_patch`) | `revision_store.add(..., author)` | Jev-Übernahme braucht einen eigenen Autor `jev_proposal`, falls sie eine Revision erzeugt |

### 4.2 Entscheidungen

1. **Getrennter Speicher, keine gemeinsame Zählung (Frage 1).** Jev-Vorschläge
   und -Outcomes leben ausschließlich unter `pi_decisions/<proposal_id>/`
   (Implementierungsplan §3.6). Jev legt **niemals** ein
   `PiMemoryStore`-Item an, weder als Kandidat noch als Outcome. Damit sind
   Auto-Promotion-Zähler und Retrain-Export strukturell blind für Jev —
   Isolation durch Abwesenheit, nicht durch einen Filter, den man vergessen
   kann.
2. **Kein `source`-Diskriminator im bestehenden Recorder (Frage 2).**
   `record_run_outcome_if_needed()` und `PiMemoryStore` bleiben unverändert.
   Ein Diskriminator wäre nur nötig, wenn beide Quellen denselben Store
   teilten; das ist ausgeschlossen. Stattdessen ein **neues Modul**
   `pi_decision_outcome` mit eigener `record_jev_outcome_if_needed()` und
   eigenem Marker `runs/<run_id>/artifacts/jev_outcome_recorded.json`,
   aufgerufen an denselben zwei Stellen wie der bestehende Recorder
   (Status-Poll, Run-Delete-Route), aber unabhängig davon und mit eigenem
   Fehlerpfad (ein Fehler in einem Recorder darf den anderen nie blockieren).
3. **Dauerhaft parallel (Frage 3).** Jev ist kein Übergangsschritt bis zur
   Scharfschaltung der lokalen Modelle (Lernplan Schritt 7). Wechselwirkung
   ausschließlich offline in M5: das Replay-Werkzeug liest beide Speicher
   **read-only** und vergleicht; weder darf Jev lokale Modelle trainieren noch
   die lokalen Modelle Jev-Kandidaten verändern (Zielbild §5/§9).

### 4.3 Verknüpfung Run -> Jev-Vorschlag und Aussagegrenze

Beim Übernehmen speichert der Vorschlag `applied_at`, die `updates[]` (Pfad,
alter/neuer Wert) und `config_hash_before`/`config_hash_after` des
Entwurfs. Übernahme verändert nur den Entwurf; eine Revision entsteht erst,
wenn der Nutzer speichert (Autor dann `jev_proposal`, wenn der Speicherweg
das tragen kann, sonst `save_config` — dann greift die Werteprüfung unten).

`record_jev_outcome_if_needed()` verknüpft **nicht** über Hash-Gleichheit
(die YAML-Serializer von Apply-Zeit und Run-Start sind laut bestehendem
Recorder-Kommentar nicht garantiert byte-gleich; `effective_config_yaml()`
injiziert Felder). Stattdessen liest es die `config.yaml` des Runs read-only
und prüft, ob jeder `updates[].path` dort den vorgeschlagenen Wert trägt:

| `attribution` | Bedeutung |
|---|---|
| `paths_present` | alle Jev-Pfade haben im Run den vorgeschlagenen Wert |
| `paths_partial` | nur ein Teil (Nutzer hat nachträglich geändert) |
| `paths_absent` | Run nutzt die Jev-Werte nicht — kein Outcome-Eintrag |

Ein Outcome trägt bewusst **kein** `quality_delta` und keine Aussage
"verbessert": `comparison_kind = "unpaired"`, plus die Provenance
(`config_revision_id`, `run_id`, Run-Qualitätsmaße falls vorhanden). Selbst
`paths_present` ist konfundiert — im selben Run können weitere Änderungen
(PI-Aktionsplan, manuelle Edits) stecken. Belastbare Qualitätsaussagen gibt
es erst über den gepaarten Vergleich derselben Frames in M5.

### 4.4 Optionale Härtung (nicht Teil von M2, eigene Entscheidung)

`collect_scan_reference_points()` hat heute kein Quellenprädikat. Ein
`memory.get("source") == "scan_ai_apply"`-Filter wäre eine harmlose
Absicherung gegen künftige Fremdquellen, ändert aber das Verhalten des
bestehenden Systems (jeder bereits akzeptierte Eintrag anderer Herkunft würde
ausgeschlossen). Nicht im Rahmen dieser Klärung geändert; vor einer Änderung
prüfen, welche `source`-Werte in vorhandenen Memory-Stores tatsächlich
`accepted` sind.
