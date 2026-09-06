# PI Memory-Ablauf — Ist-Zustand (Generierung und Verwendung)

> **Status:** Beschreibung des Ist-Zustands, kein Plan
> **Datum:** 2026-09-05
> **Betrifft:**
> - `web_backend_cpp/src/services/pi/pi_memory_store.cpp` / `include/services/pi/pi_memory_store.hpp`
> - `web_backend_cpp/src/services/pi/pi_ai_request_builder.cpp`
> - `web_backend_cpp/src/services/pi/pi_outcome_recorder.cpp`
> - `web_backend_cpp/src/services/pi/pi_live_edit_recorder.cpp`
> - `web_backend_cpp/src/services/pi/pi_param_model.cpp`
> - `web_backend_cpp/src/routes/ai_routes.cpp` (`build_apply_candidate_memory`, `accepted_pi_memories_for_scan_request`, `negative_pi_memories_for_scan_request`, `compact_memory_for_scan_context`)
> - `web_backend_cpp/src/routes/pi_routes.cpp` (`POST /api/pi/memories/<id>/review` und weitere Memory-Routen)
> - `web_frontend_v3/js/pages/ai-empfehlung.js` (`reviewPiMemory`, Accept/Reject/Deprecate-Buttons), `web_frontend_v3/js/api/endpoints.js` (`memoryReview`)
> - `agent_service/src/services/frameAnalysisService.ts` / `liveImageChatService.ts`
> **Verwandte Docs:** [`pi_local_learning_plan_de.md`](pi_local_learning_plan_de.md) (Design/Implementierungsplan, dieses Dokument beschreibt den *aktuellen Laufzustand* daraus), [`scan_ai_parameterstudio.md`](scan_ai_parameterstudio.md), [`prime_agent_bewertung_de.md`](prime_agent_bewertung_de.md)

---

## 0. Überblick in einem Satz

Ein Memory entsteht automatisch, sobald eine KI-Empfehlung angewendet oder eine Live-Edit-Operation nicht rückgängig gemacht wird; es wird aber erst nach manueller Freigabe (`status: accepted`) für die nächste Anfrage sichtbar — Auto-Promotion ist bereits implementiert, läuft aber nur im Shadow-Mode (berechnet, geloggt, nicht angewendet).

```
Apply / Live-Edit-Close
        │
        ▼
  append_candidate()            status = "candidate"
        │
        ▼  (später, nach Run)
  attach_outcome()               akkumuliert Run-Qualität, ändert Status NICHT
        │
        ▼
  evaluate_auto_promotion()      Shadow-Mode: würde "accepted"/"rejected" vorschlagen,
        │                        wendet es aber nicht an
        ▼
  [manuelles /review]  ◄── einziger aktiver Weg zu status = "accepted"/"rejected"
        │
        ▼
  retrieve() / retrieve_negative()   nur status == "accepted" (positiv) bzw.
        │                            rejected/deprecated (negativ) werden geliefert
        ▼
  session_context["accepted_pi_memories" / "negative_pi_memories"]
        │
        ▼
  build_ai_request_v2()          → positive_memories / negative_memories im JSON-Request
        │
        ▼
  agent_service (frameAnalysisService.ts) — Prompt-Kontext für den nächsten LLM-Call
```

---

## 1. Generierung eines Memory-Kandidaten

Es gibt zwei unabhängige Erzeugungspfade, die in denselben Store schreiben.

### 1.1 Scan-Analyse-Apply (`ai_routes.cpp:337`, `build_apply_candidate_memory`)

Ausgelöst, wenn eine KI-Empfehlung aus der Scan-Analyse im Parameter Studio tatsächlich angewendet wird (nicht schon bei bloßer Anzeige der Empfehlung).

Aufgebaut wird:

- `type: "config_optimization"`, `status: "candidate"`, `privacy_class: "metadata_only"`, `source: "scan_ai_apply"`
- `config_updates`: die **gesamte** Menge der in diesem Apply geänderten `{path, value}`-Paare als **ein** Record (nicht ein Record pro Einzelpfad — siehe `pi_local_learning_plan_de.md` Abschnitt 0.2: eine Kausalzuordnung pro Pfad wäre bei gemeinsam angewendeten Änderungen nicht herstellbar)
- `context_signature`: aus `session_context`/`scan_metrics`/`scan_metadata` gebaut über `build_memory_query_context_signature(...)` — feste Feldliste (siehe Abschnitt 3)
- `scope.applies_when` / `scope.does_not_apply_when`: Textregeln, z. B. `"context_signature_matches_target_acquisition_and_affected_config_paths"` / `"different_target_class_or_acquisition_setup"`, `"contradicting_outcome_memory_exists"`
- `provenance`: `analysis_id`, `revision_id`, `config_sha256` — der Schlüssel, über den `pi_outcome_recorder` später die tatsächlich gemessene Run-Qualität anhängt (Abschnitt 2)
- `persisted`: ob eine Config-Revision daraus entstanden ist

### 1.2 Live-Image-Editor (`pi_live_edit_recorder.cpp`)

Ausgelöst beim Schließen einer Editier-Session (`/api/pi/live-image-chat/close`), nicht bei jedem einzelnen Undo/Redo/Adjust-Schritt.

- Feature-Vektor: Pixelstatistik aus dem **ursprünglichen** Bildzustand (`original_fits`, nicht `current_fits`) — Ziel ist, aus dem Ausgangsbild eine gute Operation vorherzusagen, nicht das Ergebnis zu beschreiben.
- Label-Logik: **Terminalwert pro Op-Typ** am Sessionende. Eine Operation, die im `undo_stack` überlebt, erzeugt einen Kandidaten mit `outcome.retained: true`; eine per Undo vollständig entfernte oder durch eine andere Operation am selben Zielparameter ersetzte erzeugt `outcome.retained: false` — sie wird nicht stillschweigend verworfen, sondern als negatives Beispiel erfasst.
- `type: "live_edit_operation"` — anderer `type`-Wert, aber derselbe Store, derselbe Review-/Retrieval-Mechanismus.

Beide Pfade schreiben über `PiMemoryStore::append_candidate(memory)` in dieselbe append-only Datei `memories_v2.jsonl` (`pi_memory_store.hpp:29`). Es entsteht dabei **kein** externer/LLM-Aufruf — reines lokales Logging aus einem bereits abgeschlossenen Nutzer-/System-Ereignis.

---

## 2. Outcome-Anreicherung (nach der Generierung, vor jeder Bewertung)

`attach_outcome()` (`pi_memory_store.hpp:49`) hängt gemessene Ergebnisqualität an einen bestehenden Kandidaten an, **ohne** dessen `status` zu ändern:

- Für Scan-Analyse-Kandidaten: `pi_outcome_recorder` joint `pi_run_provenance.json` (Revisions-Abstammung, geschrieben bei Run-Start) mit `pi_run_quality.json` (Mittelwert der gültigen AQMH-Frame-Gewichte, geschrieben von `runner_phase_local_metrics.cpp`) und schreibt das Ergebnis an den Memory-Kandidaten.
- Für Live-Edit-Kandidaten ist "behalten oder verworfen" sofort beim Sessionende bekannt — kein zweiphasiges Apply-dann-später-Attach nötig, der Outcome wird direkt mit dem Kandidaten zusammen angelegt.
- Geschrieben wird an **zwei** Stellen: `reviews_path()` (bestehendes "letzter gewinnt"-Overlay, für Abwärtskompatibilität mit Code, der `item["outcome"]` liest) **und** `outcomes_path()`, ein zusätzlicher, akkumulierender Append-only-Log, den `list()` vollständig (nicht latest-wins) zu `item["outcomes"]` zusammenführt — nötig, weil ein Memory über mehrere Runs hinweg mehrere unabhängige Outcomes ansammeln muss, nicht nur den letzten.

`evaluate_auto_promotion()` zählt `item["outcomes"]`-Einträge mit numerischem `quality_delta` (positiv/negativ; `null`-Einträge — heute der Normalfall, siehe `comparison_kind: "unpaired"` — zählen für keine Seite). Ab **N ≥ 3** unabhängigen positiven Outcomes würde die Regel `accepted` vorschlagen, bei überwiegend negativen `rejected`.

**Wichtig:** `evaluate_auto_promotion()` ändert den Status selbst nicht. Das Ergebnis wird nur über `log_auto_promotion_shadow_decision()` in einen separaten Shadow-Log geschrieben (`memory_auto_promotion_shadow_v1.jsonl`, abrufbar über `GET /api/pi/memories/auto-promotion-shadow`) — beobachtbar, aber ohne Wirkung auf `retrieve()`. Grund: eine falsch kalibrierte Promotion-Regel würde sonst sofort und unbemerkt die Empfehlungsqualität verschlechtern; sie soll erst stichprobenartig geprüft werden, bevor sie live geschaltet wird.

**Praktische Konsequenz:** Der einzige heute aktive Weg, einen Kandidaten von `status: "candidate"` zu `"accepted"`/`"rejected"`/`"deprecated"` zu bringen, ist der manuelle `review()`-Aufruf (`/review`-Endpunkt, `allowed_review_status()`: `promotable`, `accepted`, `rejected`, `deprecated`).

### 2.1 Technische Integration des manuellen Review-Wegs

Kein separates Werkzeug, kein Python, kein zweiter Client — eine normale HTTP-Route im selben C++-Backend, aufgerufen aus derselben Web-Oberfläche wie Scan und Parameter Studio.

**Backend-Route** (`pi_routes.cpp:2635`, Crow-Framework):

```cpp
CROW_ROUTE(app, "/api/pi/memories/<string>/review").methods("POST"_method)
([state](const crow::request& req, const std::string& memory_id) {
    auto body = parse_body(req);
    const std::string status   = body->value("status", std::string());      // accepted/rejected/deprecated/promotable
    const std::string reviewer = body->value("reviewer", std::string("user"));
    ...
    tile_compile::pi::PiMemoryStore store(pi_storage_dir(state));
    const auto review = store.review(memory_id, status, reviewer, note, outcome, scope);
    return json_resp({{"ok", true}, {"review", review}});
});
```

`POST /api/pi/memories/<memory_id>/review`, Body `{status, reviewer, note, outcome?, scope?}`. Die Route ruft nur `PiMemoryStore::review(...)` auf, die Schreiblogik selbst liegt vollständig im bereits beschriebenen Store (`reviews_path()`).

**Client: GUI3-Web-Frontend, drei Buttons** (`web_frontend_v3/js/pages/ai-empfehlung.js`):

- Jede im PI-Analyse-Panel gelistete Memory-Karte hat, sofern `canReview` zutrifft, die Buttons **Accept**, **Reject**, **Deprecate** (Zeile ~1160–1162).
- Ein Klick ruft `reviewPiMemory(memoryId, reviewStatus)` (Zeile 1184) auf, das über `API_ENDPOINTS.pi.memoryReview(memoryId)` (`web_frontend_v3/js/api/endpoints.js:60`) genau diese Route mit `POST` anspricht — `reviewer` wird dabei fest auf `"gui3"` gesetzt.
- Es ist also ein von Hand gedrückter Button in derselben App, kein Kommandozeilentool, kein separater Login, keine zweite Oberfläche.

**Zusätzliche interne Aufrufer (kein zweiter HTTP-Weg, sondern direkte C++-Methodenaufrufe):** `store.review(...)` wird auch von anderen Backend-Codepfaden direkt aufgerufen — `pi_negative_learning` (automatische Counterexample-Erzeugung, `pi_routes.cpp:1706`), sowie `pi_run_outcome_evaluator` und `pi_resume_feedback` (`pi_routes.cpp:2735/2859/2968`). Das sind Aufrufe der C++-Methode innerhalb des bereits erwähnten, teils unverdrahteten Altsystems (`/evaluate-run`/`/outcome`/`/promote`, siehe Abschnitt 6) — nicht der HTTP-`/review`-Endpunkt selbst und nicht Teil des aktiv genutzten Wegs. Der `/review`-Endpunkt hat nur einen tatsächlich aktiv verdrahteten Aufrufer: die drei GUI3-Buttons.

---

## 3. Retrieval: welche Memories in eine neue Anfrage einfließen

Vor jeder neuen Scan-Analyse-Anfrage (`ai_routes.cpp:806`, `session_context_with_accepted_memories`):

1. **Query-Aufbau:** `collect_config_leaf_paths(base_config, ...)` sammelt alle Blattpfade der aktuellen Config; zusammen mit `allowed_paths` bilden sie `paths`. `build_memory_query_context_signature(body, base_config, allowed_paths, paths)` baut daraus dieselbe feste `context_signature`-Struktur wie bei der Generierung.
2. **Positiv-Retrieval:** `accepted_pi_memories_for_scan_request()` ruft `store.retrieve({type: "config_optimization", paths, context_signature}, limit=8)` auf. Von den Treffern werden nur die mit `status == "accepted"` übernommen — alles andere (`candidate`, `promotable`, `rejected`, `deprecated`) wird verworfen, selbst wenn der Kontext gut matcht.
3. **Negativ-Retrieval:** `negative_pi_memories_for_scan_request()` ruft `store.retrieve_negative(...)` mit derselben Query auf — liefert Memories, die bereits als nicht funktionierend markiert wurden ("das wurde hier schon probiert").
4. **Scoring (`context_match_score()`, `pi_memory_store.cpp:208`):** je Feld ein Teil-Score, u. a.:
   - Text-Exaktmatch: `target.object_name` (4 Pkt.), `target.object_type` (4), `acquisition.camera_type` (3), `acquisition.color_mode`/`camera_name` (2), `optics.telescope`/`mount.type` (2) …
   - Zahlen mit Toleranzband: `acquisition.frame_count` (±25 % = volle Punkte, ±60 % = halbe), `optics.focal_length_mm` (±15 %/±35 %), `optics.f_ratio` (±15 %/±30 %)
   - Mengen-Overlap: `acquisition.filters`, `pipeline.affected_paths`, `pipeline.phases`, `problem.classes`, `problem.hints` (je Treffer Punkte, gedeckelt)
   - Bool-Match: `target.has_extended_emission`
   - Zusätzlich ein Status-Bonus (`accepted` → +2) beim Ranking.
   - Fehlende Query-Felder werden nicht bestraft, sondern separat als `missing_query_fields` mitgeführt (siehe Abschnitt 4).
5. **Kompaktierung:** `compact_memory_for_scan_context()` (`ai_routes.cpp:480`) reduziert jeden Treffer auf ein schlankes Objekt für den Prompt — `memory_id`, `type`, `source`, `status`, `context_match_score`, `match_explanation`, `summary`, `confidence`, `config_updates`, `context_signature`, `scope`, `evidence`, `outcome`, `review`-Statusfelder — Interna wie Rohdaten oder volle Historie werden nicht mitgeschickt.
6. Beide Listen landen in `session_context["accepted_pi_memories"]` bzw. `["negative_pi_memories"]` (nur wenn nicht leer).

---

## 4. Einbettung in den KI-Request

`build_ai_request_v2()` (`pi_ai_request_builder.cpp:39`):

- Liest `positive_memories`/`negative_memories` entweder direkt aus dem Input oder — falls dort leer — aus `session_context["accepted_pi_memories"]`/`["negative_pi_memories"]` (`positive_memories_from_session_context()`, `negative_memories_from_session_context()`).
- Aggregiert `missing_query_fields` über alle Memories mit `match_coverage` zu `retrieval_coverage_summary.systemically_missing_context_fields` — aber nur Felder, die in **mehr als der Hälfte** der gecoverten Memories fehlen (systemisch, nicht Einzelfall). Damit kann das LLM erkennen: "hier fehlt strukturell Kontext, Confidence senken", statt jeden Memory-Eintrag einzeln zu interpretieren.
- Baut daraus den finalen `pi.ai-request.v2`-JSON-Request mit `positive_memories`, `negative_memories`, `retrieval_coverage_summary`, `context_signature`, `scan_context`, `config`, `allowed_config_paths` etc.

Dieser Request geht unverändert an den Sidecar (`agent_service/src/services/frameAnalysisService.ts`). Der Prompt weist das Modell an, akzeptierte Memories als Präzedenzfälle zu nutzen und negative Memories nicht zu wiederholen — reines Prompt-Konditionieren pro Aufruf, kein Fine-Tuning und kein persistenter Modellzustand auf LLM-Seite.

---

## 5. Parallel dazu, noch nicht scharf: lokale Modelle

Unabhängig vom Memory-Store-Pfad existiert `pi_param_model.cpp`: ein gewichteter k-Nearest-Neighbor auf demselben Feature-Vektor-Schema, für aktuell zwei Testpfade (`bge.method`, `normalization.mode`), abgelegt in `pi_models/scan/<path>/vN/`. Bei jeder abgeschlossenen Scan-Analyse wird zusätzlich eine Vorhersage berechnet und nach `pi_models/scan/<path>/shadow_predictions.jsonl` geloggt (`{feature_vector, model_prediction, llm_value, agrees_with_llm}`) — das sind dieselben Dateien, die aktuell im Arbeitsverzeichnis als geändert markiert sind. Diese Vorhersagen beeinflussen den tatsächlichen `action_plan`/die validierten Updates **nicht** — reines Shadow-Logging. Das Scharfschalten (lokale Modelle als primäre Quelle, LLM nur noch für Rationale/Cold-Start) ist der in `pi_local_learning_plan_de.md` Abschnitt 7 als letzter offener Schritt benannte Punkt und braucht zuerst reale Nutzungsdaten.

---

## 6. Was aktuell *nicht* automatisch passiert (Zusammenfassung der Lücken)

- `candidate → accepted/rejected`: nur manuell über `POST /api/pi/memories/<id>/review` (Details: Abschnitt 2.1), ausgelöst durch die Accept/Reject/Deprecate-Buttons im GUI3-Frontend — Auto-Promotion ist gebaut und verifiziert, aber Shadow-Mode-only.
- `quality_delta`: bleibt für Scan-Kandidaten heute überwiegend `null` (`comparison_kind: "unpaired"`) — belastbare Same-Frames-Vergleiche (z. B. über `bge.autotune`-CV-Holdout) sind laut Plan die vorgesehene, aber noch nicht produktiv genutzte Quelle.
- Lokale Modelle (`pi_param_model`) beeinflussen keine tatsächliche Empfehlung, nur Shadow-Logs.
- Ein zweites, älteres Outcome/Promotion-System in `pi_routes.cpp` (`evaluate_memory_outcome_payload()`, `/evaluate-run`, `/outcome`, `/promote`) existiert parallel, ist aber an keiner Stelle automatisch verdrahtet und liest ein Artefakt (`stats.json`), das der Runner nie schreibt — bewusst unangetastet gelassen, siehe `pi_local_learning_plan_de.md` Abschnitt 9, Punkt 6.
