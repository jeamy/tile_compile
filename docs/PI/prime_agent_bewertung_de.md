# PrimeIntellect prime-agent — Bewertung für tile_compile

> **Status:** Bewertungsdokument, keine Implementierung geplant
> **Datum:** 2026-09-05
> **Betrifft:** `agent_service/` (bestehender `pi-coding-agent`-Sidecar), `web_backend_cpp/src/services/pi/*` (bestehender Memory-Store/Local-Learning-Stack)
> **Verwandte Docs:** [`pi_local_learning_plan_de.md`](pi_local_learning_plan_de.md), [`scan_ai_parameterstudio.md`](scan_ai_parameterstudio.md), [`pi_ki_empfehlungen_de.md`](pi_ki_empfehlungen_de.md), [`pi_context_protocol_compression_plan_de.md`](pi_context_protocol_compression_plan_de.md)

---

## 0. Anlass und verifizierter Kernfakt

Frage: Kann [PrimeIntellect-ai/prime-agent](https://github.com/PrimeIntellect-ai/prime-agent) sinnvoll in tile_compile integriert werden, um KI-Erfahrungen/Memories zu nutzen — parallel zu oder statt der bestehenden `@earendil-works/pi-coding-agent`-Integration (`agent_service/`)?

**Verifiziert gegen `package.json` von prime-agent (nicht nur README-Prosa):**

```json
"dependencies": {
  "@earendil-works/pi-coding-agent": "^0.9.1",
  "get-east-asian-width": "^1.6.0"
}
```

`prime-agent` ist also **kein separates Ökosystem neben** `pi-coding-agent`, sondern eine Anwendung, die **auf derselben Bibliothek aufbaut**, die tile_compile in `agent_service/src/services/frameAnalysisService.ts` bereits direkt einbindet (`AuthStorage`, `ModelRegistry`, `createAgentSession`, `session.subscribe`/`session.prompt`, siehe [`scan_ai_parameterstudio.md`](scan_ai_parameterstudio.md) Abschnitt 8). Das ändert die Fragestellung: Es geht nicht darum, eine zweite Agent-Runtime neben `pi-coding-agent` zu betreiben, sondern darum, ob ein **Nutzungsmuster einer bereits vorhandenen Abhängigkeit** übernehmenswert ist.

`prime-agent` selbst ist ein CLI-Produkt (TypeScript/Python-Hybrid, persistente Python-REPL, `@anthropic-ai/sandbox-runtime` als Dev-Dependency für Code-Ausführung), kein importierbares SDK — es lässt sich nicht als Bibliothek in `agent_service` einbinden, sondern nur als externes Werkzeug parallel installieren und aufrufen.

## 1. Was prime-agent tut

- **Recursive Language Model (RLM):** Kontext wird als Variablen in einer persistenten Python-REPL behandelt; `rlm(...)`-Aufrufe spawnen echte Sub-Agents.
- **Continual Harness:** Speichert Supplemental Prompts, **Memories**, Skill-Beschreibungen und wiederverwendbare Sub-Agent-Spezifikationen als versionierten, persistenten State mit Rollback — unabhängig vom Basis-Systemprompt.
- **`/refine`:** Überprüft die aktuelle Trajektorie und schreibt kleine, evidenzgestützte Updates auf den Harness-State.
- Daemon-Betrieb, Agent-zu-Agent-Kommunikation, automatische Kompression über Turns hinweg.
- Zielgruppe laut eigener Beschreibung: lang laufende, autonome Coding-/Research-Aufgaben — kein Runtime-Baustein für ein Produkt, sondern ein interaktives/autonomes Entwicklerwerkzeug.

## 2. Abgleich mit dem, was tile_compile für „Memories" bereits hat

tile_compile hat mit [`pi_local_learning_plan_de.md`](pi_local_learning_plan_de.md) bereits ein **vollständig implementiertes** (Schritte 1–6), produktionsnahes Memory-/Lernsystem für genau den Anwendungsfall „aus vergangenen Empfehlungen lernen":

- `pi_memory_store`: `candidate → accepted/rejected`, jetzt mit automatischer Promotion via `evaluate_auto_promotion()` (N ≥ 3 positive Outcomes), Shadow-Mode-Log, Export/Dedupe.
- `pi_outcome_recorder`: verknüpft angewendete Config-Diffs über Revisions-Abstammung mit gemessener Run-Qualität (`pi_run_quality.json`), robust gegen Run-Löschung, mit `on_complete`-Hook statt reinem Polling.
- `pi_feature_vector` / `pi_param_model`: gemeinsames Feature-Schema für Scan- und Live-Edit-Domain, lokale k-NN-Modelle in `pi_models/` (versioniert, `config_schema_sha256`-gepinnt, Kill-Switch, Rollout-Schutz per Leave-one-out-CV), bereits mit Unit-Tests verifiziert.
- Für Live-Edit-Ops: Terminalwert-Logik, Bildstatistik-Feature-Vektor direkt aus `cv::Mat`, ohne Vision-Call.

Das ist strukturell **stärker** als `prime-agent`s „Continual Harness"-Memories für diesen konkreten Zweck:

| Anforderung | `pi_local_learning` (bestehend) | `prime-agent` Continual Harness |
|---|---|---|
| Speicherform | strukturierte, typisierte Records (Feature-Vektor, Pfad, Wert, Outcome) | Freitext/Prompt-Fragmente, versioniert |
| Validierung gegen Config-Schema | ja, durchgängig (`config_schema_sha256`-Pinning, Schema-Drift-Erkennung) | nein — Textkontext, keine Schema-Bindung |
| Automatische Promotion/Rejection | ja, messbasiert (`quality_delta`, N ≥ 3) | `/refine` ist evidenzgestützt, aber für Prompt-/Skill-Text, nicht für numerische Parameter |
| Kosten pro Inferenz | lokal, <5 ms, kein externer Call | LLM-Call pro Refine-Zyklus |
| Reproduzierbarkeit/Offline-Fähigkeit | ja (Kill-Switch, deterministischer Fallback) | nein — braucht laufenden Agenten mit Modellzugriff |

**Konsequenz:** Für den vom Nutzer angesprochenen Zweck „Erfahrungen/Memories nutzen" gibt es in tile_compile bereits eine speziell dafür gebaute, verifizierte, günstigere Lösung. Eine zweite, Freitext-basierte Memory-Schicht aus `prime-agent` daneben zu betreiben würde das in [`pi_local_learning_plan_de.md`](pi_local_learning_plan_de.md) Abschnitt 4.3 explizit festgehaltene Prinzip verletzen: *„ein Ort pro Artefakttyp, keine Schatten-Kopien"*. Das wäre Redundanz, kein Zugewinn.

## 3. Wo `prime-agent` trotzdem eine Nische hätte — und wer sie schon besetzt

Der einzige Bereich, in dem `prime-agent`s Stärken (lange, autonome, mehrstufige Untersuchungen über Sub-Agents, persistente REPL, Refine über eine ganze Trajektorie) etwas leisten würden, das die bestehende PI-Infrastruktur nicht abdeckt, ist:

> Offene, mehrstufige **Entwickler-seitige Investigation** über angesammelte Run-/Log-/Memory-Daten hinweg — z. B. das in [`pi_local_learning_plan_de.md`](pi_local_learning_plan_de.md) Abschnitt 9, Punkt 6 offene Altsystem (`evaluate_memory_outcome_payload`, nie verdrahtete `/evaluate-run`-Routen, `stats.json`, das nie geschrieben wird) systematisch aufzuräumen, oder wiederkehrende AutoBGE-Guard-Fehlklassifikationen über viele echte Runs zu untersuchen.

Das hat mit dem eigentlichen Thema dieses Dokuments — Erfahrungen/Memories zur Laufzeit nutzen — nichts zu tun; es ist eine reine Entwickler-Werkzeugfrage, unabhängig vom PI-Memory-System. Es gibt hierfür in tile_compile keine belegte, bereits etablierte Praxis, die diese Rolle besetzt — die README-Attribution nennt lediglich KI-Coding-Assistenten, die beim Bau des Repos geholfen haben (ein einmaliger Entwicklungsvorgang), keine laufende Analyse-/Lernpraxis über Runs oder Logs. Aus dieser Zeile lässt sich also nicht ableiten, dass die Nische von `prime-agent` bereits besetzt wäre.

Der tatsächliche Grund, `prime-agent` hierfür nicht einzuführen, ist schlichter: Diese Nische ist ohnehin nicht das, wonach der Nutzer gefragt hat (Erfahrungen/Memories nutzen), und ein Ersatz für die im Alltag ohnehin schon eingesetzten Coding-Assistenten (Claude Code, Codex etc.) müsste eigenständig begründet werden — dafür fehlt hier die Evidenz. `prime-agent` wäre bestenfalls ein weiteres CLI-Tool zur Auswahl für Entwickler, kein struktureller Zugewinn und kein Baustein des Produkts.

## 4. Sicherheits-/Vertrauensgrenze, falls doch einmal ausprobiert

Sollte `prime-agent` testweise als Entwicklerwerkzeug (nicht als Produktbestandteil) eingesetzt werden, gilt eine andere Vertrauensstufe als beim bestehenden `agent_service`-Sidecar:

- Der heutige Sidecar (`frameAnalysisService.ts`) ist zustandslos, macht reine JSON-Request/Response-Aufrufe, führt keinen Code aus und läuft nur auf `127.0.0.1`. Ein fehlerhafter Aufruf blockiert nachweislich weder Scan noch Parameter Studio (siehe [`scan_ai_parameterstudio.md`](scan_ai_parameterstudio.md) Abschnitt 17).
- `prime-agent` bringt eine **persistente Python-REPL mit Code-Ausführung** (`@anthropic-ai/sandbox-runtime`) und Sub-Agent-Spawning mit — eine grundsätzlich größere Angriffs-/Fehlerfläche.
- Falls genutzt: nur lokal von einem Entwickler gestartet, niemals aus `web_backend_cpp`/`runs_routes.cpp` heraus automatisiert aufgerufen, kein Schreibzugriff auf `pi_memory_store`, `tile_compile.yaml` oder Config-Revisions — Berichte/Vorschläge werden von einem Menschen gelesen und ggf. manuell in bestehende Docs/Issues überführt, genau wie heute bei Claude Code/Codex.

## 5. Wie eine Verwendung aussehen würde (falls gewünscht)

Kein Code-Integrationsschritt, sondern ein Werkzeug-Opt-in für Entwickler:

1. `prime-agent` lokal installieren (npm-Paket, eigener Prozess, außerhalb von `tile_compile`-Runtime-Verzeichnissen).
2. Harness-State/Memories dieses Agenten beziehen sich ausschließlich auf **Entwicklungsaufgaben** (z. B. „räume das Altsystem in `pi_routes.cpp` auf", „analysiere die letzten 50 `shadow_predictions.jsonl`-Einträge auf systematische Abweichungen") — nicht auf Produktions-Parameterempfehlungen.
3. Lesezugriff auf `runs/`, `pi_models/*/shadow_predictions.jsonl`, `docs/PI/*` als Kontext, kein Schreibzugriff auf `pi_memory_store` oder Config.
4. Ergebnis: ein vom Menschen geprüfter Vorschlag/PR, wie bei jedem anderen Agenten-Tool in der Attribution-Liste — keine neue Laufzeitkomponente in `agent_service` oder `web_backend_cpp`.

## 6. Explizit nicht Teil einer Integration

- Kein zweiter Memory-Layer neben `pi_memory_store`.
- Kein Ersatz von `frameAnalysisService.ts`/`liveImageChatService.ts` — beide bleiben der schlanke, zustandslose Sidecar-Pfad.
- Kein automatisierter, aus dem Backend ausgelöster `prime-agent`-Aufruf.
- Keine neue Produktabhängigkeit (`prime-agent` wäre, falls überhaupt, ein Entwickler-Tool, kein `package.json`-Eintrag von `agent_service`).

## 7. Empfehlung

Nicht integrieren. Die konkret genannte Motivation — KI-Erfahrungen/Memories nutzen — ist mit `pi_local_learning_plan_de.md` bereits besser gelöst (strukturiert, validiert, lokal, günstig) als `prime-agent`s Freitext-Harness das leisten würde. Die einzig verbleibende Nische von `prime-agent` (lang laufende autonome Entwickler-Investigation, Abschnitt 3) betrifft ein anderes Thema als Memories/Lernen und rechtfertigt für sich allein keine neue Werkzeugeinführung. Optional: bei Gelegenheit unverbindlich als zusätzliches Entwickler-CLI-Tool ausprobieren (Abschnitt 5), aber ohne Erwartung eines strukturellen Zugewinns gegenüber dem Status quo.
