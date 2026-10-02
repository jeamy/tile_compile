# PI Jev M0 — Review der alten „HARD RULE“-Texte (Post-Run)

> **Stand:** 2026-09-22.
> **Status:** M0-Teilergebnis für den Post-Run-Teil des Checklistenpunkts
> "Review der alten „HARD RULE“-Texte" aus
> [Implementierungsplan Abschnitt 4](pi_jev_implementierungsplan_de.md#4-m0--vertr%C3%A4ge-quellen-und-testgrundlage-einfrieren).
> Der Pre-Run-Teil ist bereits in [pi_scan_pre_rules_de.md](pi_scan_pre_rules_de.md)
> erledigt (dortige `[B]`-Kennzeichnung entfernt, jede Regel trägt jetzt
> `source_kind`/`validation_status`). Dieses Dokument macht denselben Review
> für den Post-Run-/Resume-Bereich, relevant für M6.

## 1. Gefundener Text

`docs/PI/attic/pi_run_chat_empfehlungs_chat_datenplan.md` (Entwurf,
2026-07-17, `attic/` — nicht aktiv verlinkt von den drei Jev-Plänen) enthält
unter „Prompt-Vertrag" (Zeile ~486) genau das Muster, das dieser
Checklistenpunkt sucht: hart in einen LLM-Prompt geschriebene Regeln statt
ausführbarer Validierung, z. B. *„Nutze nur Pfade aus
`schema_summary.valid_config_paths`"*, *„`star_pressure` ist nur Diagnose,
kein Config-Pfad"*, *„`stretch.*` ist ungültig"*, *„HMS nur als Resume
empfehlen, wenn ... "*.

**Enforcement-Stelle geprüft:** `min_resume_phase`, `valid_config_paths` und
`invalid_or_diagnostic_paths` kommen im aktuellen Backend-Code
(`web_backend_cpp/src/services/pi/`) **nicht vor** (verifiziert per Grep,
2026-09-22). Der Mechanismus wurde nie implementiert — die Regeln existierten
ausschließlich als Prompt-Text, exakt die in
[pi_scan_pre_rules_de.md §1](pi_scan_pre_rules_de.md#1-grundprinzip) verworfene
Konstruktion ("Eine deterministisch ausgeführte Heuristik bleibt eine
Heuristik" gilt hier noch stärker: das war nicht einmal deterministisch
ausgeführt, sondern reine Prompt-Instruktion ohne Nachprüfung der Ausgabe).

## 2. Warum das mehr als ein Dokumentationsfund ist

Das Dokument beschreibt einen **realen, konkreten Vorfall** (nicht
hypothetisch): ein M42-Lauf, bei dem der bestehende Run-Chat
`stretch.star_pressure`/`stretch.protect_b` als Config-Änderung empfahl
(ungültige Pfade — der wirksame Block heißt `hypermetric_stretch`) und
`HYPERMETRIC_STRETCH` als Resume-Phase vorschlug, obwohl die wahrscheinliche
Ursache vor HMS lag (Crop, AQMH-Fallback, Stacking, Normalisierung oder BGE).

Das ist genau die Fehlerklasse, vor der
[Implementierungsplan M6](pi_jev_implementierungsplan_de.md#10-m6--post-run-beratung-und-pi-%C3%BCbergabe)
bereits schützen will ("Kein frei vom Modell erzeugter Phasenname", G5
Resume-Vertrag). Der Vorfall ist die empirische Begründung für diese Guards,
nicht nur eine abstrakte Vorsichtsmaßnahme — beide Dokumente sollten das
explizit verknüpfen, statt den Vorfall unerwähnt zu lassen.

## 3. Bewertung der vorgeschlagenen Mechanik

Der im Attic-Dokument skizzierte Ansatz (`valid_config_paths` +
`invalid_or_diagnostic_paths` + `min_resume_phase` pro Action, vom Backend vor
Prompt-Erstellung berechnet und dem Modell als Allowlist mitgegeben, statt nur
als Anweisung) ist strukturell **kompatibel** mit dem in
[pi_scan_pre_rules_de.md §3](pi_scan_pre_rules_de.md#3-harte-policy-vor-kandidatenbildung)
etablierten Guard-Muster (`guard_schema`, `guard_scope`) und mit
[Implementierungsplan §3.4](pi_jev_implementierungsplan_de.md#34-vorschlag-piconfig-proposalv1)
(`updates[].path` nur aus einer serverseitig geprüften Menge). Empfehlung für
M6: dieselbe Allowlist-vor-Modellantwort-Architektur wiederverwenden
(`pi_decision_policy` guard_scope, Abschnitt 3 des Regelkatalogs), nicht die
alte Prompt-Text-Variante reaktivieren. `min_resume_phase` als Konzept — jede
Action bindet die früheste Phase, ab der sie gültig ist — ist eine sinnvolle,
bisher fehlende Ergänzung zu M6s bestehendem Resume-Vertrag (G5) und sollte
dort als konkretes Feld aufgenommen werden.

**Nicht übernehmen:** die Prompt-eingebettete Durchsetzung selbst (*"Der
Provider-Prompt muss harte Regeln enthalten"*) — das ist exakt der
`source_kind: prompt_policy`-Fall, der laut Regelkatalog `validation_status:
unvalidated` bleibt, bis eine echte serverseitige Prüfung existiert. Ein LLM,
das angewiesen wird, eine Regel zu befolgen, ist keine Durchsetzung der
Regel.

## 4. Ergebnis für die Checkliste

- [x] Enforcement-Stelle identifiziert: keine (reiner Prompt-Text, nie
      implementiert).
- [x] Keine implizite Änderung von Pipeline-Defaults durch diesen Review
      selbst — reine Dokumentation.
- [ ] Offen: `min_resume_phase` als Feld in `pi.config-proposal.v1`/M6-State
      ergänzen (Implementierungsplan-Änderung, nicht Teil dieses M0-Reviews).
