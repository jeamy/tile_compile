# PI Decision Trace und Unified Assistant — Plan

> **Status:** Plan, nichts davon ist implementiert.
> **Datum:** 2026-09-27, überarbeitet 2026-10-02 (Code-Review: Faktenkorrekturen, Schema- und
> Endpunkt-Klarstellungen, ergänzte Risiken und Prüfpunkte), Analyse-Nachtrag 2026-10-02 (Arbeitskontext-Modell,
> Rationale-Struktur, Jev-Annahmekette, Audit-Beziehung, Backfill, Session-Fortsetzung, P1-Gates), 2. Review
> 2026-10-02 (LLM-Anker-Record `llm_proposal`, `image_id`-Definition, Chat-Lücke Analyse-Kontext,
> `run_uid`-Normalisierung, Löschkaskade, Session-Lock, `origin`-Feld).
> **Betrifft:** `agent_service/src/services/*`, `web_backend_cpp/src/services/pi/*`,
> `web_backend_cpp/src/routes/pi_routes.cpp`, `web_frontend_v3/js/{main.js,pages,components,state}`
> **Verwandt:** [`pi_local_learning_plan_de.md`](pi_local_learning_plan_de.md),
> [`pi_memory_ablauf_de.md`](pi_memory_ablauf_de.md), [`pi_jev_decisions_plan_de.md`](pi_jev_decisions_plan_de.md),
> [`pi_jev_m0_provider_protocol_de.md`](pi_jev_m0_provider_protocol_de.md),
> [`pi_live_image_chat_plan.md`](pi_live_image_chat_plan.md)
>
> Keine Zeit- oder Aufwandsschätzungen (AGENTS.md). Der Plan nennt Umfang, Abhängigkeitsreihenfolge, Risiken und Prüfpunkte.

---

## 0. Ausgangslage (aus dem Code gelesen)

**Lernen / Nachvollziehbarkeit**

- `agent_service` nutzt `@earendil-works/pi-coding-agent` ^1.0.0; vier Services (`runChatService.ts`,
  `liveImageChatService.ts`, `frameAnalysisService.ts`, `modelService.ts`) erzeugen Sessions mit
  `SessionManager.inMemory()`. Das Gespräch, aus dem eine Entscheidung entstand, geht beim Sidecar-Ende
  verloren.
- Das Memory (`pi.memory.v2`, `pi_memory_store.*`) speichert das **Ergebnis** einer Entscheidung (`config_updates`,
  `context_signature`, `provenance` mit `analysis_id`/`revision_id`/`config_sha256`, später `outcome`/`outcomes`). Es
  speichert nicht das **Warum**: Evidenz, Alternativen, Abwägung, Nutzerkorrektur.
- Jev ist ein *Decisions*-Modell (`POST /api/alpha/decisions`, nur `choice`/`noul`/`score`), kein Chat-Modell. Die Antwort
  enthält `choice`, `probabilities`, `confidence`, `model` (datierter Build), `id`, `usage.cost` — aber **keinen
  Begründungstext**. Das Warum einer Jev-Entscheidung ist daher nur über Zustand, Kandidatenmenge, Wahrscheinlichkeiten und
  Policy-Urteil rekonstruierbar (`pi_decision_policy.cpp`, `pi_decision_outcome.cpp`).

**UI**

- `main.js`: Header + Sub-Tab-Leiste + `#content`. Kein globaler Platz für einen Assistenten.
- AI-Funktionen sind verstreut und untereinander nicht verknüpft:
  - `pages/parameter.js` (1031 Zeilen): Tabs *Parameter* / *AI* (`ai-empfehlung.js`, 1491 Zeilen) / *Jev*
    (`jev-empfehlung.js`, 454 Zeilen), beide hinter Feature-Flags.
  - `pages/run-monitor.js` (2685 Zeilen): Run-Chat, `post-run-advice.js`, `situation-assistant.js`, `explain-panel.js`.
  - `components/live-image-viewer.js` (1533 Zeilen): eigener Chat + Bildoperationen (`/api/pi/live-image-chat/*`).
  - `pages/tools.js`: `ai-settings` (AI & API).
- State ist je Funktion getrennt (`ai-state.js` 33 Zeilen, `scan-state.js`, `run-state.js`, `config-state.js`); es gibt
  keinen gemeinsamen "Arbeitskontext".
- Backend-Endpunkte sind je Funktion getrennt: `/api/scan/analysis*`, `/api/pi/run-chat*`, `/api/pi/live-image-chat/*`,
  `/api/pi/assistant/ask`, `/api/pi/action-plans/{validate,preview,apply}`, `/api/pi/memories/*`, `/api/pi/audit`.

---

## 1. Ziele und Nichtziele

**Ziele**

1. Jede Entscheidung (Config-Apply, Ablehnung, Bildoperation, Undo, Jev-Auswahl, Memory-Review) wird als
   nachvollziehbarer **Decision Record** gespeichert: wer, was, auf welcher Evidenz, gegen welche Alternativen, mit welcher
   Begründung — und wie die Begründung epistemisch einzuordnen ist.
2. Das Warum wird beim **Entscheiden** erfasst, nicht nachträglich aus Chat-Text erraten.
3. Ein **zentraler Assistent** als Dock am rechten Rand, auf jeder Seite erreichbar, bündelt Parameteroptimierung,
   Bildnachbearbeitung, Run-Beratung, Erklärung — und Jev gleichberechtigt.
4. Memories verweisen auf Decision Records; Retrieval kann Gründe mitliefern.

**Nichtziele**

- Kein Umbau des Runners; Runner bleibt frei von LLM-/Jev-Abhängigkeiten (Zuständigkeitstabelle Jev-Plan §1).
- Keine Lockerung der Sicherheitsregeln: PI-Tools bleiben mutation-free; Schreiben nur über Action-Plan → Preview →
  `validate-config` → explizites Apply. Ein Decision Record ändert nie einen validierten Patch.
- Keine automatische Memory-Promotion durch Begründungen. Shadow-Mode bleibt wie dokumentiert.
- Der C++-Memory-Store wird nicht ersetzt.

---

## 2. Teil A — Decision Trace

### 2.1 Grundprinzipien

- **Backend ist maßgeblich.** Decision Records liegen im C++-Backend (append-only JSONL neben `memories_v2.jsonl`), nicht
  im Sidecar. Der Sidecar liefert höchstens Verweise.
- **Epistemische Herkunft ist Pflichtfeld.** Jede Begründung trägt eine Quelle:
  `user_stated` | `measured_fact` | `model_probability` (Jev) | `rule` (deterministische Policy) | `llm_hypothesis`.
  Das vom LLM selbst formulierte Reasoning ist nachträgliche Rationalisierung und wird immer als `llm_hypothesis`
  markiert. Es dient der Anzeige, nie als Evidenz für Promotion oder Retrieval-Gewichtung.
- **Fehlender Grund ist ein Wert.** Gibt der Nutzer keinen Grund an, wird `no_reason_given` gespeichert — kein Raten.
- **Metadata-only bleibt.** Records enthalten Pfade, Werte, Hashes, Kennzahlen, IDs; keine Rohbilder, keine
  Dateipfade aus Nutzerdaten. Volltext-Sessions bleiben getrennt und werden nur referenziert. Einzige
  Ausnahme mit Schutzregel: `rationale.user.text` ist Nutzer-Freitext und setzt `privacy_class` auf
  `metadata_plus_user_text`. Er wird längenbegrenzt, das Backend
  entfernt absolute Pfade, und er ist nicht Teil des Standard-Exports (`privacy_class` am Record, s. u.).
- **Append-only mit Overlay.** `decisions_v1.jsonl` wird nie mutiert. Nachträgliche Verknüpfungen
  (`memory_id`, `outcome_refs`, `supersedes`) gehen in eine zweite Datei `decision_links_v1.jsonl` —
  dasselbe Muster wie `reviews_path()`/`outcomes_path()` im Memory-Store; Leser mergen den letzten Stand
  über `decision_id`.
- **Schreibautorität beim Backend.** Records entstehen an den realen Ereignispunkten (Apply-Route,
  Jev-Adapter, Live-Edit-Recorder, Review-Route). Der Client-Endpunkt (§3.4) nimmt nur eingeschränkte
  Nutzer-Payloads entgegen — `kind`, Bezug über `action_plan_id`/`preview_id`, `rationale.user`.
  `actor`, `context_ref`, `evidence`, `rationale.basis` und den `jev`-Block setzt ausschließlich das Backend; vom Client
  behauptete `measured_fact`- oder `model_probability`-Basiseinträge sind unzulässig.

### 2.2 Schema `pi.decision-record.v1`

```json
{
  "schema_version": "pi.decision-record.v1",
  "decision_id": "dec_...",
  "parent_decision_id": null,
  "created_at": "...",
  "kind": "config_apply | config_reject | config_undo | preview_dismissed | config_manual_edit | llm_proposal | jev_choice | jev_override | live_edit_op | live_edit_undo | live_edit_keep | memory_review | no_change",
  "actor": "user | llm | jev | rule",
  "idempotency_key": "",
  "action_plan_id": null,
  "origin": "live | legacy_migration",
  "privacy_class": "metadata_only | metadata_plus_user_text",
  "context_ref": {
    "context_id": "", "analysis_id": "", "revision_id": "", "config_sha256": "",
    "run_uid": "", "image_id": "", "context_signature": {}
  },
  "subject": { "paths": [{"path": "", "from": null, "to": null}], "op": null },
  "evidence": { "fact_ids": [], "metrics_ref": "", "state_sha256": "" },
  "alternatives": [
    { "id": "", "label": "", "probability": null, "selected": false, "rejected_by": null }
  ],
  "rationale": {
    "user": {
      "reason_codes": [],
      "catalog_version": "",
      "text": "",
      "no_reason_given": false
    },
    "basis": [
      { "source": "measured_fact | model_probability | rule | llm_hypothesis",
        "ref": "fact_id | generation_id | rule_id | recommendation_id" }
    ]
  },
  "jev": {
    "model_reported": "", "generation_id": "", "confidence": null,
    "probabilities": {}, "cost_usd": null, "policy_verdict": "accepted | rejected | clamped",
    "policy_reason": ""
  },
  "llm": { "model": "", "usage": { "input_tokens": null, "output_tokens": null, "cost_usd": null } },
  "session_ref": { "session_id": "", "entry_ids": [] },
  "memory_id": null,
  "outcome_refs": []
}
```

Hinweise:

- `jev` ist nur bei Jev-Entscheidungen gefüllt. Es enthält genau die Felder, die die Antwort liefert; die Wahrscheinlichkeiten
  und die Kandidatenmenge **sind** bei Jev die Begründung, ergänzt um das Policy-Urteil.
- `keep_current` und "Ergebnis akzeptabel, keine Änderung" (`no_change`) werden aufgezeichnet. Sie sind laut Jev-Plan ein
  vollwertiger Erfolg und für das Lernen genauso wichtig. **Präzedenzregel:** Wählt Jev `keep_current`, ist der
  Record `jev_choice` mit `keep_current` als ausgewählter Alternative — `no_change` ist Nutzer-/LLM-Entscheidungen
  vorbehalten.
- `config_reject` ist nur das **explizite** Verwerfen eines Vorschlags (Button, optional mit Grund). Ein
  geschlossener oder abgelaufener Preview-Dialog ist `preview_dismissed` und gilt nicht als Negativsignal —
  "nicht angewendet" ist keine Ablehnung. Bei `preview_dismissed` ist `actor=user` für einen bewussten
  Dismiss und `actor=rule` für Timeout/Ablauf.
- **Rationale ist zweigeteilt.** `rationale.user` enthält, was der Nutzer ausdrücklich angab (Chips, Text oder
  `no_reason_given`). `rationale.basis[]` enthält die Systembasis der Entscheidung, jeweils mit eigener Herkunft
  (`measured_fact`, `model_probability`, `rule`, `llm_hypothesis`) und Referenz. Übernimmt ein Nutzer einen LLM-Vorschlag
  und gibt zusätzlich einen Chip an, stehen beide Herkunftsarten nebeneinander statt in einem Einzelwert. `user.*` ist
  immer Nutzerangabe und braucht keine Quellkennung. `no_reason_given` ist ein Boolean in `rationale.user`, keine Quelle.
- `actor` beschreibt, wer die Entscheidung festgelegt hat — bei Übernahme einer LLM- oder Jev-Empfehlung also `user`.
  Serverseitig validierte Regeln: `actor=user` → `rationale.basis` darf alle Herkunftsarten enthalten (die Basis
  beschreibt, worauf sich die Entscheidung stützte), `rationale.user` ist frei; `actor=llm` → `basis` nur `llm_hypothesis`,
  `user` leer; `actor=jev` → `basis` nur `model_probability`/`rule`, `user` leer; `actor=rule` → `basis` nur `rule`,
  `user` leer. Vom Client behauptete `measured_fact`-/`model_probability`-Basiseinträge sind unzulässig (§2.1). Innerhalb
  von `rationale.user` schließen sich `no_reason_given=true` und nicht-leere `reason_codes`/`text` aus.
- **Jev-Annahmekette.** Eine Jev-Auswahl erzeugt `jev_choice` (`actor=jev`). Übernimmt der Nutzer den validierten Patch,
  folgt `config_apply` (`actor=user`, `parent_decision_id` = `jev_choice`). Wählt der Nutzer eine andere Alternative als
  Jev, ist das `jev_override` (`actor=user`, Parent = `jev_choice`) und danach ggf. `config_apply` mit Parent
  `jev_override`. `config_reject` bleibt dem Verwerfen eines **PI/LLM-Vorschlags** vorbehalten; das Verwerfen einer
  Jev-Auswahl ist `jev_override` mit der Alternative `keep_current` oder einer Nutzerwahl.
- **LLM-Anker-Record.** Die PI/LLM-Empfehlung selbst wird als `llm_proposal` (`actor=llm`,
  `rationale.basis=[llm_hypothesis]`, `rationale.user` leer) aufgezeichnet — sie ist das Pendant zu
  `jev_choice` und der `parent_decision_id`-Anker von `config_apply`/`config_reject` im LLM-Pfad.
  `recommendation_id` in `rationale.basis[].ref` verweist auf diesen Record (bei Altbestand auf
  `analysis_id` + Vorschlagsindex). Ohne ihn wäre `actor=llm` unerreichbar und die Kette
  Empfehlung → Apply nur bei Jev rekonstruierbar.
- `evidence.metrics_ref` ist eine artefakt-relative Referenz (relativer Pfad oder Artefakt-Schlüssel, nie
  ein absoluter Pfad — sonst verletzt er metadata-only). `fact_ids` verweisen auf die stabilen `fact_id`s
  aus `pi_context_protocol_compression_plan_de.md`.
- `idempotency_key` (z. B. `action_plan_id` + Ereignis) verhindert Doppel-Records bei Retries und
  Doppelklicks; das Backend dedupliziert darüber.
- `action_plan_id` verknüpft den Record mit dem validierten Patch — ohne ihn ist die Kette Empfehlung →
  Preview → Apply nicht eindeutig rekonstruierbar.
- `llm` trägt Modell und Usage des zugehörigen PI-Calls (analog `jev.cost_usd`), damit Kosten- und
  Qualitätsauswertung über beide Provider symmetrisch ausfallen.
- `memory_id` und `outcome_refs` werden typischerweise später befüllt — über `decision_links_v1.jsonl`,
  nicht durch Umschreiben des Records (§2.1).
- `parent_decision_id` bildet Ketten ab: Empfehlung → Preview → Apply → Undo → neuer Vorschlag.
- `state_sha256` macht Jev-Calls reproduzierbar auswertbar, ohne den Zustand doppelt zu speichern.

### 2.3 Reason-Code-Katalog

Versionierte Datei (analog `protected_paths_v1.json`), Schema `pi.user-reason-codes.v1`, getrennt nach
Bereich. **Namensraum beachten:** `pi.config-proposal.v1` kennt bereits `reason_codes` für Policy-Urteile
(`no_applicable_candidates`, `validated` in `pi_decision_service.cpp`/`pi_decision_policy.cpp`). Der
Nutzer-Grundkatalog ist ein getrennter Katalog; beide Listen dürfen nicht vermischt werden. Chips-Labels
sind i18n-pflichtig: Der Katalog liefert Codes plus stabile Schlüssel, DE/EN-Texte liegen in den
Frontend-i18n-Dateien. Jeder Record speichert die verwendete `catalog_version`, damit alte Codes
interpretierbar bleiben:

- **Config:** `too_aggressive`, `too_conservative`, `artifacts`, `noise_worse`, `sharpness_worse`, `color_off`,
  `runtime_too_long`, `evidence_insufficient`, `outdated_by_new_data`, `other`.
- **Bild:** `too_soft`, `too_harsh`, `halo`, `clipping`, `background_off`, `color_cast`, `taste`, `other`.

Die genaue Liste ist beim Implementieren gegen reale Memories und Live-Edit-Daten zu verifizieren. Neue Codes erhöhen die
Katalogversion; alte Records bleiben lesbar.

### 2.4 Erfassungspunkte

| Ereignis | Wo | Grundangabe |
|---|---|---|
| Config-Apply | `/api/scan/analysis/apply`, `/api/pi/action-plans/apply` | Evidenz, alle `config_updates`, Alternativen aus Preview |
| Config explizit verworfen | Action-Plan-Dialog | `config_reject`, Reason-Chips + Freitext |
| Jev-Auswahl | Jev-Adapter im Backend (`pi_decision_*`) | `jev`-Block komplett, automatisch |
| Jev überstimmt | Jev-Karte | Nutzerwahl ≠ Jev, Reason-Chips |
| Live-Edit-Op, Undo, Close | `pi_live_edit_recorder.cpp` | Op, Retained-Status, Reason bei Undo |
| Memory-Review | `/api/pi/memories/<id>/review` | `note` → `rationale`, Reason-Chips |
| Run-Outcome | `pi_outcome_recorder` | `outcome_refs` über `decision_links_v1.jsonl` (§2.1) |
| Config-Undo | Rücknahme eines Apply (neuer Pfad, optional) | `config_undo`, verweist via `parent_decision_id` |
| Preview geschlossen/abgelaufen | Action-Plan-Dialog (Dismiss, Timeout) | `preview_dismissed` (nur bei realem Dismiss/Timeout, nicht durch Reload), kein Reason-Zwang, kein Negativsignal |
| Manuelle Config-Änderung ohne Empfehlung | Parameter-Tab (optional, ab P5) | `config_manual_edit`, `no_reason_given`-Default |

Reason-Abfrage ist **optional und niedrigschwellig**: ein Klick auf Chips, überspringbar. Bei `Reject`, `Deprecate`,
Undo und Jev-Override wird sie angeboten, bei `Accept`/`Apply` nur als optionales Feld.

**Live-Edit-Granularität (Verhaltensänderung, bewusst):** Heute schreibt `pi_live_edit_recorder` das Memory erst beim
Close und nicht pro Undo/Redo-Schritt (`pi_memory_ablauf_de.md` §1.2). Decision Records weichen davon ab: `live_edit_op` und
`live_edit_undo` entstehen **während** der Session, `live_edit_keep` beim Close mit dem Retained-Terminalwert
(derselbe, den der Recorder für das Memory nutzt). Das Memory-Verhalten bleibt unverändert; nur der Trace ist
feiner. Datenvolumen ist begrenzt, indem Parameter-Adjust-Ströme (Slider) je Op zu einem Record zusammengefasst werden
(letzter Wert beim Loslassen/Commit). Ein Absturz ohne Close hinterlässt eine erkennbar offene Kette, kein implizites
`keep`. `pi_memory_ablauf_de.md` ist bei Umsetzung entsprechend zu ergänzen.

**Beziehung zu `/api/pi/audit`:** Das Audit (Action-Plan-Applies, Scan-AI-Config-Applies, Memory-Reviews) bleibt
bestehen und wird in P5 zu einer **Sicht über Decision Records** umgestellt. Bis dahin laufen beide parallel; die
Quellen werden nicht doppelt gepflegt, sondern der Audit-Endpunkt liest ab P5 aus den Records und fällt für Altfälle
auf seine bisherigen Quellen zurück.

**Backfill:** Bestehende Memories und Audit-Einträge werden **nicht** zu Records umgedeutet. Optional erzeugt ein
einmaliger, explizit ausgelöster Migrationslauf `legacy`-Records (`origin="legacy_migration"`,
`rationale.user.no_reason_given=true`,
`basis=[]`, `kind` aus dem Altereignis) nur dort, wo `analysis_id`/`revision_id` eindeutig belegt sind. Ohne
Backfill zeigt der Warum-Bereich bei Altfällen "Keine Aufzeichnung (vor Einführung)".

### 2.5 Session-Persistenz im Sidecar

Verifiziert gegen `@earendil-works/pi-coding-agent` **1.0.0** (npm `latest`, 2026-10-01; Diff der
Typdefinitionen des npm-Tarballs gegen die installierte 0.87.1):

- **Keine Breaking Changes für den Sidecar.** `createAgentSession`, `SessionManager`,
  `DefaultResourceLoader`, `getAgentDir` und die genutzten `session`-Methoden (`subscribe`, `prompt`,
  `abort`, `dispose`) sind zwischen 0.87.1 und 1.0.0 unverändert. `SessionManager.inMemory(cwd?, options?,
  entries?)` bleibt. Einzige Signaturänderung am Rand: `steer()`/`followUp()` liefern
  `Promise<QueuedInputDisposition>` statt `Promise<void>` — unkritisch für Aufrufer ohne
  Rückgabeauswertung; der Sidecar nutzt beide nicht.
- **Persistenz erfordert nicht zwingend 1.0:** `SessionManager.create(cwd, sessionDir?)`, `.open(path)`,
  `.continueRecent()`, `.findById()`, `.list()` existieren bereits in 0.87.1. Ein Verzeichnis unter dem
  Memory-Verzeichnis wird direkt über `sessionDir` gesetzt (getrennt von `memories_v2.jsonl`).
- `session_id` und `entry_ids` sind verfügbar: `appendMessage()` liefert die Entry-ID, `AgentSession`
  emittiert `entry_appended`-Events, `getEntries()`/`getEntry(id)`/`getLeafId()` lesen den Baum.
  Session-Format: append-only JSONL, `CURRENT_SESSION_VERSION = 3`. Der Sidecar gibt `session_id` und die
  `entry_ids` der relevanten Nachrichten an das Backend zurück; das Backend schreibt sie in `session_ref`.
- **Marker-API heißt `sessionManager.appendCustomEntry(customType, data)`** (nicht `pi.appendEntry()`):
  `CustomEntry` geht nicht in den Modellkontext — genau das geforderte "hier wurde Decision X getroffen"
  ohne zweiten Wahrheitsspeicher. Alternativ `appendLabelChange(targetId, label)` als Lesezeichen.
- Neue Session-Dateien entstehen erst mit der ersten User-/Assistant-Nachricht (pi#10000): leere Threads
  hinterlassen keine Datei — für die Kontext-Verknüpfung relevant, `session_ref` kann leer bleiben.
- **Fortsetzung und Nebenläufigkeit (zu klären vor P3):** Der Sidecar legt heute pro Request eine Session an. Mit
  Persistenz gilt: Die `session_id` wird pro Arbeitskontext (§3.0) im Backend gehalten und bei Folge-Requests
  mitgegeben; der Sidecar öffnet sie mit `SessionManager.open(path)` bzw. `findById()` statt `inMemory()`. Pro Kontext
  läuft höchstens ein aktiver Request; ein zweiter wird abgewiesen oder gequeued (Dock zeigt "beschäftigt").
  Durchgesetzt wird das über ein **per-Session-Lock im Sidecar** — die Session-Datei ist append-only JSONL;
  zwei parallele `AgentSession`s auf derselben Datei wären korruptionsgefährdet. Eine
  nicht auffindbare Session (gelöscht, Dateiverlust) startet eine neue und schreibt einen Hinweis in den Thread; sie
  bricht nie das Apply. Frame-Analyse (`frameAnalysisService`) bleibt ein abgeschlossener Einzellauf je Analyse mit
  eigener Session, kein fortgesetzter Chat.
- Datenschutz: Session-Dateien können Pfade und Nutzertext enthalten. Sie werden nie in Memory-Exports aufgenommen
  (`pi.memories-export` bleibt metadata-only). Löschkonzept und Aufbewahrung sind vor Aktivierung festzulegen
  (§2.8).
- **Nebenbefund, separat zu bewerten:** 0.99.0/1.0.0 bringt einen eingebauten TypeSafe-`jev-latest`-Classifier
  in `ModelRuntime` (`classify()`, codemode-`models.classify()`, Virtual Models inkl.
  `jev-router.ts`-Beispiel). Der Jev-Pfad dieses Projekts bleibt davon unberührt — der backend-seitige
  Direktaufruf des OpenRouter-Decisions-Endpunkts ist maßgeblich —, aber der Sidecar könnte Jev-Auswahlfragen
  künftig auch über den Classifier stellen. Kein Lieferumfang dieses Plans.

### 2.6 Memory v3 (additiv)

- Neue Felder an `pi.memory`: `decision_refs[]`, `reason_codes[]` (aggregiert), `rationale_summary` (kurz, mit Quelle).
- Schema bleibt rückwärtskompatibel (Import/Export, Dedupe nach Signatur). `schema_version` wird erhöht, alte Memories
  haben leere Felder.
- Retrieval liefert **kompakt**: Gründe als Codes plus `rationale_ref`. Details werden nur auf Nachfrage nachgeladen (passt zu
  den Chunk-Nachlade-Regeln in `pi_context_protocol_compression_plan_de.md`).
- `rejected`/`deprecated`-Negativsignale tragen künftig ihren Grund; der Request-Builder kann dadurch "abgelehnt wegen
  `artifacts`" statt nur "abgelehnt" weitergeben.
- Auto-Promotion (Shadow) darf `reason_codes` nur zum **Gruppieren** nutzen (z. B. Outcomes je Grund), nicht als Evidenz
  für Annahme.

### 2.7 Auswertung

- Neue schreibgeschützte Ansicht "Warum" an Memory und im Audit (`/api/pi/audit`): Entscheidungskette, Alternativen,
  Gründe mit Herkunftslabel.
- Auswertung nach Reason-Code: Welche Empfehlungen werden aus welchem Grund abgelehnt? Das ist die Eingabe für spätere
  Regelkalibrierung — ausdrücklich nicht für automatische Änderungen.
- `GET /api/pi/decision-records` mit Filtern (`kind`, `reason_code`, Zeitraum, `context_ref`) als Basis der
  Auswertung; Aggregation (Ablehngründe je Empfehlungstyp, Jev-Kalibrierung, Kosten aus `jev`/`llm`) erfolgt
  lesend darüber.

### 2.8 Aufbewahrung, Export und Größe

- `decisions_v1.jsonl` und `decision_links_v1.jsonl` wachsen append-only. Rotation/Kompaktierung und die
  Aufbewahrungsfrist werden **vor Aktivierung** festgelegt — gemeinsam mit dem Session-Löschkonzept (§2.5).
- `pi.memories-export` bleibt unverändert metadata-only und enthält keine Decision Records. Ob es einen
  separaten, ebenfalls metadata-only Decision-Export gibt (forensisch, als Eingabe der Regelkalibrierung in
  §2.7), ist eine offene Entscheidung; `rationale.user.text` gehört in keinen Fall hinein.

---

## 3. Teil B — Unified Assistant (UI)

### 3.0 Arbeitskontext (Modell)

Dock, Threads und `GET /api/pi/assistant/thread` brauchen eine eindeutige Kontext-ID. Definition:

- **`context_id`** wird vom Backend vergeben (für Runs aus der stabilen `run_uid`, §3.5) und ist Teil von `context_ref` jedes Records. Es gibt drei Ebenen mit
  fester Verschachtelung:
  1. **Analyse-Kontext** (`analysis_id`): Scan-Analyse, Revisionen (`revision_id`), Parameteroptimierung, Pre-Run-Jev.
  2. **Run-Kontext** (`run_uid`): entsteht aus genau einer Analyse-Revision (`config_sha256` verknüpft sie),
     **sofern eine existiert** — Direkt-Starts und Queue-Runs ohne AI-Analyse beginnen die Kette am Run.
     Run-Chat, Post-Run-Beratung, Jev nach dem Run.
  3. **Bild-Kontext** (`image_id`): Live-Image-Session auf einem Ergebnisbild; gehört zu genau einem Run.
- **`image_id` ist neu zu definieren.** Es gibt heute kein Bild-ID-Konzept: `live-image-chat/create` nimmt nur
  `run_id` entgegen und arbeitet auf genau einem Ergebnisbild (`find_output_fits` → `outputs/live_edit.fits`);
  die Historie liegt unter `live_image_chat/<hash(run_id)>.json`. Der Bild-Kontext ist daher faktisch ein Facet
  des Run-Kontexts mit genau einem Slot: `image_id = "<run_uid>:live_edit"`. Bekommt ein Run künftig mehrere
  bearbeitbare Ergebnisbilder, wächst `image_id` um den Output-Slot; die Kontextstruktur bleibt gleich.
- Ein Kontext zeigt im Thread seine **Vorfahren schreibgeschützt mit** (Bild → Run → Analyse), damit die Kette
  Analyse → Run → Bildoperation sichtbar bleibt. Schreiben (Chat, Apply) geschieht nur im aktiven Kontext.
- Die Auswahl des aktiven Kontexts leitet `context-provider.js` aus dem UI-Zustand ab (laufender Run, geöffnetes Bild,
  geladene Analyse) und lässt sie im Kontextband umschalten. Ohne Auswahl gilt der zuletzt aktive Kontext.
- Die bestehenden Historien (`run-chat/history`, `live-image-chat/history`, Scan-Analyse-History) werden über ihre
  jeweilige ID (`run_uid`, Live-Session→`image_id`, `analysis_id`) auf `context_id` abgebildet; `assistant/thread`
  mergt sie über diese Zuordnung. Wo kein Run oder keine Analyse existiert, gibt es keinen Kontext.

### 3.1 Zielbild

Ein **PI-Assistent-Dock** am rechten Rand des Browsers:

- Immer erreichbar (alle Haupt-Tabs, alle Sub-Tabs), ein-/ausklappbar, in der Breite verschiebbar, Zustand in
  `ui-state` persistiert, Tastenkürzel.
- Ein **Kontextband** oben im Dock zeigt, worauf sich der Assistent bezieht: Analyse, Config-Revision, Run, Bild. Der
  Kontext wird aus `scan-state`, `run-state`, `config-state` und dem ausgewählten Bild abgeleitet und kann umgeschaltet
  werden.
- Das Dock hat **einen einzigen Thread pro Arbeitskontext** (Analyse/Run/Bild), keine Unter-Tabs. Der Thread ist
  chat-artig (Freitext möglich) und enthält **Karten**. Jev-Entscheidungen erscheinen als Karten im selben Thread,
  erkennbar an Jev-Label und eigenem Kartenstil. Es gibt keinen separaten Jev-Tab und keinen Verlauf/Warum-Tab.
- Ein Wechsel des Arbeitskontexts wechselt den Thread.

| Karte | Inhalt | Aktionen |
|---|---|---|
| Empfehlung (LLM) | Scan-AI-Empfehlungen, Evidenz, Diff | Preview, Apply, Verwerfen + Grund |
| Jev-Entscheidung | Fragen, Kandidaten mit Wahrscheinlichkeit/Konfidenz, `keep_current`, Policy-Urteil, Diff, Kosten | Preview, Apply, Überstimmen + Grund |
| Run-Beratung | Run-Chat, Post-Run-Advice, Situation-Assistent | Nachfragen, Erklärung |
| Bild-Operation | Vorschau-Thumbnail, Parameter, Op-Historie | Anpassen, Undo/Redo, Reset, Export + Grund |
| Lernen | Memory-Kandidat nach Apply | Accept/Reject/Deprecate + Grund |

**Warum-Bereich an jeder Karte:** Jede Entscheidungskarte hat einen ausklappbaren Abschnitt "Warum" mit der Decision-Kette
(Teil A): Evidenz, Alternativen, Begründung mit Herkunftslabel, Vorgänger-/Folgeentscheidungen, Outcome. Er ist lesend
und lädt Details erst beim Ausklappen nach. Ein globaler Verlauf entfällt; die Kette ist über die Karte und über die
Memory-/Audit-Ansicht erreichbar.

**Jev-Auslöser:** Jev ist kein Chat (Decisions-Modell, nur `choice`-Fragen). Im Eingabebereich des Docks stehen deshalb
neben dem Freitextfeld Schaltflächen "Jev: Pre-Run-Empfehlung" und "Jev: Beratung nach Run" (nur auf Wunsch). Das Ergebnis
kommt als Jev-Karte in den Thread.

- Alle schreibenden Aktionen laufen weiter durch Action-Plan → Preview → `validate-config` → Apply. Das Dock ersetzt die
  Oberfläche, nicht die Sicherheitslogik.
- **Das große Bild bleibt eine eigene Fläche.** Der Live-Image-Viewer braucht Platz. Er wird eine Ansicht im Hauptbereich,
  gebunden an den Dock-Kontext (Bild ausgewählt → Dock zeigt Bild-Operationen). Sein eigener Chat entfällt und zieht ins
  Dock um.
- Schmale Fenster: Dock wird Overlay statt Spalte.

### 3.2 Jev als Karte im PI-Thread, gleiche Behandlung

Jev ist ein Empfehlungs-Provider neben dem LLM-Pfad, kein eigener Tab:

- Gemeinsame Provider-Schnittstelle im Frontend: `{ id, kind: "llm_scan" | "jev_pre" | "jev_post" | "run_chat" | "live_edit",
  run(context) → Karte[] }`.
- Jev-Karten nutzen dieselben Bausteine wie LLM-Karten: Diff (`yaml-diff.js`), Guardrail-Badges, Preview/Apply,
  `reason-picker.js`, Warum-Bereich, Decision Record, Action-Plan-Pipeline. Es gibt keine Sonderwege für Jev.
- **Vergleich im Thread:** Liegen für denselben Kontext ein PI- und ein Jev-Vorschlag vor, verweisen die Karten
  aufeinander ("Jev-Vorschlag abweichend") und der Warum-Bereich zeigt die Abweichung samt bestehendem Shadow-Vergleich.
  Widersprüche werden im Record verknüpft (`parent_decision_id` bzw. gemeinsamer Kontext) und sind Lernmaterial.
- Feature-Flags (`feature-flags.js`: `aiEnabled`, `jevEnabled`) steuern, welche Provider und Auslöser sichtbar sind. Ohne Jev
  bleibt das Dock voll nutzbar; Jev-Kosten (`usage.cost`) stehen in der Karte.
- Zuständigkeitsgrenzen aus dem Jev-Plan bleiben: Jev wählt nur aus Backend-Kandidaten, keine freien Pfade/Werte, keine
  Ausführung. Eine Jev-Karte kann den Patch nur anwenden, wenn die Policy ihn akzeptiert hat.
- Offen: Ob eine Rückfrage im Freitext zu einer Jev-Karte den Jev-Bezug im Prompt bekommt. Der Jev-Plan erlaubt PI nur
  Erläuterungen; der validierte Patch darf sich dadurch nicht ändern.

### 3.3 Architektur im Frontend

Neu: `web_frontend_v3/js/assistant/`

- `dock.js` — Layout, Resize, Collapse, Fokus-Handling.
- `context-provider.js` — leitet den aktiven Kontext aus bestehenden State-Modulen ab; kein neuer Datenhalter für Run/Scan.
- `thread-store.js` — Threads je Kontext, Nachrichten, Karten; Persistenz über Backend-Historie (siehe 3.4).
  Karten werden nach Reload aus Decision Records und Historien rekonstruiert (analog der Run-Monitor-Regel:
  Zustand aus Backend-Artefakten, nicht aus transientem Browser-State). Nicht persistierte Zwischenstände wie
  unangewendete Previews werden **beim Lesen** als "offen/nicht angewendet" dargestellt und nicht als Record
  geschrieben; ein `preview_dismissed` entsteht nur bei einem tatsächlichen Dismiss-Ereignis oder Timeout im Dialog,
  nie rückwirkend durch einen Reload.
- `cards/` — Kartenregistry: `recommendation.js` (PI), `jev-decision.js` (Jev), `run-advice.js`, `image-op.js`, `learning.js`, `why-section.js` (ausklappbarer Warum-Bereich, von allen Entscheidungskarten genutzt).
- `providers/` — Adapter auf bestehende Endpunkte (`llm_scan`, `jev_*`, `run_chat`, `live_edit`).
- `reason-picker.js` — wiederverwendbare Chips + Freitext, Katalog aus Backend (`/api/pi/reason-codes`).

Änderungen an Bestehendem:

- `main.js`: `app-root` bekommt ein Spaltenlayout (`tc-content` | Dock), Kürzel, Dock-Badge (ersetzt den
  `tc-badge-running`-Hack am Parameter-Sub-Tab).
- `layout.css`: Grid für Content + Dock, Dock-Breite als CSS-Variable.
- `parameter.js`: Tabs *AI* und *Jev* verschwinden (Inhalt wandert als Karten in das Dock) (nach Migration); `ai-empfehlung.js`/`jev-empfehlung.js` werden in
  Karten und Provider zerlegt statt gelöscht — die Logik bleibt, die Hülle ändert sich.
- `run-monitor.js`: eingebetteter Run-Chat/Advice wandert ins Dock; Rest unverändert (große Datei, möglichst wenig
  Eingriff).
- `live-image-viewer.js`: Chat-Teil und Op-Liste ins Dock, Canvas bleibt.
- `tools.js` → `ai-settings`: bleibt als Einstellungsseite, aus dem Dock verlinkt.
- i18n: alle neuen Texte in DE und EN (`js/i18n`, `i18n/`).

### 3.4 Backend-Seite des Docks

- Zuerst **keine neuen Fachendpunkte**: Provider-Adapter sprechen die bestehenden Routen an. Das hält die Blast Radius klein.
- Neu nur: `GET /api/pi/reason-codes` (Katalog `pi.user-reason-codes.v1`), `POST /api/pi/decision-records`
  (nur eingeschränkte Nutzer-Payloads, §2.1 Schreibautorität), `GET /api/pi/decision-records?context=...`
  (Kette lesen, §2.7-Filter) sowie `GET /api/pi/assistant/thread?context=...` als **neuer Fachendpunkt**,
  der die vorhandenen Historien (`run-chat/history`, `live-image-chat/history`, Scan-Analyse-History)
  zeitlich mergt und dedupliziert. Der Namensraum `/api/pi/decisions/*` ist bereits durch die Jev-Decision-API
  belegt (`status`/`test`/`log`/`settings` in `pi_decision_routes.cpp`) und bleibt unangetastet.
- **Der Analyse-Kontext braucht einen echten Chat-Pfad.** `/api/pi/assistant/ask` ist ein lokaler
  Keyword-Antwortgeber (`pi_assistant.cpp`, `mode: "local_read_only"`) ohne LLM, Session oder Historie — er
  kann der Freitext-Thread im Analyse-Kontext nicht sein. Entscheidung: `run-chat` wird zu einem
  kontext-parametrierten `context-chat` verallgemeinert (ein Adapter, eine Session-Verwaltung, `context_id`
  statt `run_id`), statt einen dritten Chat-Dienst einzuführen. `assistant/ask` bleibt als lokaler
  Fallback bei Sidecar-Ausfall bestehen.
- Spätere Option: ein gemeinsamer Intent-Router (`/api/pi/assistant/turn`), der deterministisch anhand von Kontext und
  Kartentyp an LLM, Jev oder Bildoperation dispatcht. Erst nach Stabilisierung der Adapter; kein erster Schritt.

### 3.5 Run-Lebenszyklus: Aktivieren, Fortsetzen, Löschen, Verschieben

**Ist-Zustand (aus dem Code):**

- Ein "Archiv" als eigenes Konzept gibt es nicht. Ein früherer Run wird über *Run History* → "Als aktuell setzen"
  aktiviert (`run-history.js: setRunCurrent`): `POST /api/runs/<id>/set-current` setzt im Backend nur
  `current_run_id`/`current_run_dir` (im Speicher), das Frontend setzt `run-state` und wechselt zum Run Monitor.
  Eine AI-Session wird dabei nicht geladen, und der Run-Chat-Verlauf wird erst bei Bedarf gelesen.
- Der Run-Chat-Verlauf liegt **zentral** unter `pi_storage_dir/run_chat/<run_id>_<hash>.json` (Legacy:
  `<run_dir>/artifacts/pi_run_chat_history.json`). Der Schlüssel ist ein Hash des `run_id`-**Strings**.
- Runs können in einem benutzerdefinierten `runs_dir` oder auf Netzlaufwerken liegen; `run_id` kann ein absoluter Pfad sein
  (`resolve_run_dir`). Dasselbe Run kann so unter zwei Strings angesprochen werden (Name und Pfad) und bekäme zwei
  verschiedene Verlaufsdateien. `resolve_run_dir` löst außerdem per Präfix auf (`name.find(run_id) == 0`), was bei
  Namensüberschneidung mehrdeutig sein kann.

**Entscheidungen:**

1. **Stabile `run_uid`.** Jeder Run bekommt eine unveränderliche ID, die nicht vom Anzeigenamen oder Pfad abhängt.
   - Neue Runs: `run_uid` wird in das bereits bei Run-Start geschriebene `pi_run_provenance.json` aufgenommen.
   - Bestehende Runs: **nichts in das Run-Verzeichnis schreiben** (AGENTS.md). Die Zuordnung `run_uid ↔ run_id ↔
     run_dir-Hinweis ↔ config_sha256 ↔ Startzeit` liegt zentral in `run_index_v1.jsonl` (append-only, Overlay).
   - `context_id` (§3.0) des Run-Kontexts wird aus der `run_uid` gebildet. Alle Schlüssel (Records, Verlauf, Session)
     laufen über `run_uid`, nie über den rohen `run_id`-String. Vorhandene `run_chat/*.json` werden über den Index auf
     die `run_uid` abgebildet (lesend; Migration ohne Löschen der Altdatei).
   - Die Auflösung `run_id`/Pfad → `run_uid` ist **exakt**, nicht per Präfix. Da `run_id` auch ein absoluter
     Pfad sein kann, wird vor dem Vergleich normalisiert (kanonischer Pfad, Symlinks, Trailing-Separator,
     Case-Regeln des Dateisystems).
2. **Aktivieren eines Runs** (`set-current`) löst im Dock einen Kontextwechsel auf den Run-Kontext aus:
   - Records, Karten und Chat-Verlauf werden sofort geladen (lesend, §3.3).
   - Die **Session wird nicht beim Aktivieren geöffnet**, sondern lazy mit der ersten Nutzernachricht (`open` per
     `session_id`, §2.5). Aktivieren bleibt dadurch billig und nebenläufigkeitsfrei.
   - Fehlende Teile degradieren sichtbar statt zu scheitern:

     | Zustand | Verhalten |
     |---|---|
     | Records und Session vorhanden | Thread vollständig, Chat setzt fort |
     | Records vorhanden, Session fehlt | Thread lesbar, neue Session beim ersten Senden, Hinweis im Thread |
     | Nur Altverlauf (`run_chat`, ohne Records) | Verlauf lesbar, Warum-Bereich "Keine Aufzeichnung (vor Einführung)" |
     | Run-Verzeichnis nicht erreichbar (Netzlaufwerk offline) | Thread lesbar aus zentralem Speicher; Aktionen, die Artefakte brauchen, deaktiviert |
     | Nichts vorhanden | Leerer Thread, Session entsteht erst mit der ersten Nachricht |

   - *Run History* zeigt je Run ein Kennzeichen "AI-Verlauf vorhanden" (Anzahl Entscheidungen) — ohne Session zu öffnen.
   - Die zuletzt aktive Kontext-Auswahl wird persistiert: `set-current` schreibt heute nur In-Memory-State
     (`runs_routes.cpp`), ein Backend-Restart verliert sie. Ergänzung im zentralen Store — eigener Eintrag in
     `run_index_v1.jsonl` oder der UI-State-Datei — damit "Zustand überlebt Reload/Restart" gilt.
3. **Resume eines Runs** (`/api/runs/<id>/resume`) setzt denselben Run-Kontext fort; es entsteht kein neuer Kontext.
   Das bestehende Resume-Feedback (`/api/pi/memories/resume-feedback`) wird als Record verknüpft (`parent_decision_id`).
4. **Löschen eines Runs** (`/api/runs/<id>/delete`):
   - Decision Records bleiben (Lerndaten, Memory-Verweise) und werden über das Overlay als `run_deleted` markiert; der
     Warum-Bereich zeigt "Run gelöscht".
   - Session und Chat-Verlauf (Nutzertext) folgen dem Löschkonzept (§2.8): Standard ist Mitlöschen oder Anonymisieren; der
     Bestätigungsdialog nennt es ausdrücklich. Die Kaskade gilt für Run- **und** Bild-Kontext (`image_id`
     hängt am `run_uid`); der Analyse-Kontext bleibt unberührt.
   - Memories mit `provenance` auf den Run bleiben bestehen.
5. **Verschieben, Kopieren, Umbenennen:** Weil nichts am Run-Verzeichnis hängt, überleben Records und Verlauf ein
   Verschieben auf demselben System, solange `run_dir` auflösbar bleibt oder über `config_sha256`/Startzeit
   wiedergefunden wird. Wandert ein Run auf eine andere Maschine, ist **optional** ein Export "Run mit Trace" vorgesehen:
   metadata-only Records plus `run_uid`, Session nur auf ausdrückliche Wahl (enthält Nutzertext), Import mit
   Kollisionsprüfung über `run_uid` (Standard: vorhandene Records nicht überschreiben).

---

## 4. Reihenfolge (Abhängigkeiten, keine Zeitangaben)

1. **P0 — Basis:** Upgrade auf `@earendil-works/pi-coding-agent` 1.0.0 (API-Verifikation erledigt, s. §2.5:
   keine Breaking Changes für die genutzten Flächen); Inventar aller Stellen, an denen Entscheidungen
   entstehen (Tabelle 2.4) gegen den Code verifizieren; Reason-Code-Katalog (`pi.user-reason-codes.v1`)
   entwerfen.
2. **P1 — Decision-Record-Backend** (inkl. `run_uid` in `pi_run_provenance.json` neuer Runs und zentralem `run_index_v1.jsonl`, §3.5) (Gate vor Aktivierung: Aufbewahrung/Rotation und Löschkonzept für Records und Sessions entschieden, §2.8): Schema, Store (`decisions_v1.jsonl` + Overlay `decision_links_v1.jsonl`),
   Idempotenz und `actor`×`basis`-Validierung, Schreibpunkte in Apply, Review, Live-Edit-Recorder,
   Jev-Adapter. Noch keine UI-Änderung. Voraussetzung für alles Weitere.
3. **P2 — Reason-Erfassung in bestehender UI:** Reason-Picker in `ai-empfehlung.js`, `jev-empfehlung.js`,
   `live-image-viewer.js`, Memory-Review. Liefert sofort Daten, unabhängig vom Dock.
4. **P3 — Dock-Hülle:** Layout, Kontext-Modell (§3.0) inkl. Backend-Vergabe von `context_id`, Kontext-Provider, Thread-Store, leere Kartenregistry, Feature-Flag. Gate: Session-Fortsetzung und Nebenläufigkeit (§2.5) sowie Run-Lebenszyklus (§3.5) entschieden.
5. **P4 — Migration in Reihenfolge des geringsten Risikos:** Run-Beratung → Scan-AI-Karten → Jev-Karten →
   Bildoperationen (größtes Stück, `live-image-viewer.js`). Die Jev-Karte "Beratung nach Run" setzt Jev-M6
   aus dem Jev-Implementierungsplan voraus; ohne M6 bleibt nur die Pre-Run-Auslösung sichtbar.
6. **P5 — Memory v3 und Retrieval mit Gründen**, Warum-Bereich an den Karten, Audit auf Sicht über Decision Records umstellen (§2.4), optionaler Backfill.
7. **P6 — Alte Tabs/Fenster entfernen**, sobald das Dock Parität hat.
8. **P7 — Optional:** Pilot von Pi Durable für Crash-Resume einzelner Services (nicht Teil der Lernfunktion).

---

## 5. Prüfpunkte

- **Schema:** Golden-Tests für `pi.decision-record.v1` (alle `kind`-Werte), Rückwärtskompatibilität von Memory v2→v3,
  Export/Import/Dedupe.
- **Invarianten:** Ein Record verändert nie einen validierten Patch; Apply läuft nur über Action-Plan/Preview/`validate-config`;
  `llm_hypothesis` fließt nicht in Promotion oder Retrieval-Evidenz; Jev-Kandidaten außerhalb des Gitters bleiben abgelehnt.
- **Jev:** Records enthalten Wahrscheinlichkeiten, Konfidenz, Build-ID, Kosten, Policy-Urteil auch für `keep_current`.
- **Records:** Overlay statt Mutation (`decision_links_v1.jsonl`), Idempotenz bei Apply-Retry/Doppelklick,
  `actor`×`basis`-Regeln serverseitig validiert, `no_reason_given` XOR `reason_codes`/`text`, Jev-Annahmekette über `parent_decision_id`, `rationale.user.text`
  längenbegrenzt und pfadbereinigt, `catalog_version` gespeichert; `llm_proposal` verankert die LLM-Kette
  symmetrisch zu `jev_choice`; `origin` unterscheidet Backfill-Records; `evidence.metrics_ref` bleibt
  artefakt-relativ (keine absoluten Pfade).
- **Namensräume:** keine Kollision mit `/api/pi/decisions/*` (Jev-API) und keine Vermischung mit den
  `reason_codes` aus `pi.config-proposal.v1`.
- **Run-Lebenszyklus:** Aktivieren öffnet keine Session; alle fünf Zustände der Tabelle (§3.5) degradieren ohne Fehler; Zugriff über Name und Pfad liefert denselben Kontext; `run_uid`-Auflösung exakt (kein Präfix) inkl. Pfad-Normalisierung; Löschen eines Runs markiert Records, kaskadiert auf den Bild-Kontext und folgt dem Löschkonzept; die aktive Kontext-Auswahl überlebt einen Backend-Restart; nichts wird in bestehende Run-Verzeichnisse geschrieben.
- **Kontextmodell:** `image_id` ist heute genau `<run_uid>:live_edit`; der Analyse-Kontext nutzt denselben
  verallgemeinerten Chat-Pfad wie der Run (kein dritter Chat-Dienst); der Sidecar hält ein per-Session-Lock
  (max. ein aktiver Request je Kontext).
- **Datenschutz:** Exports enthalten keine Session-Texte; Records enthalten keine Rohdaten/Pfade.
- **UI:** Dock auf allen Tabs, Zustand überlebt Reload, Kontextwechsel aktualisiert den Thread, Warum-Bereich lädt erst beim Ausklappen nach, schmale Fenster, DE/EN,
  Tastatur-Bedienbarkeit, bestehende Shortcuts (`1/2/3`, Pfeiltasten) kollidieren nicht mit Eingabefeldern im Dock.
- **Kein Run-Start, kein Backend-Start** durch Tests oder Migration (AGENTS.md).

---

## 6. Risiken und offene Fragen

- **Reason-Fatigue:** Zu viele Abfragen werden übersprungen. Deshalb optional, Chips, vorbelegt aus Kontext — und
  `no_reason_given` bleibt explizit. Gegenläufiges Risiko: vorbelegte Chips erzeugen Suggestionsbias und
  verfälschen die Grund-Daten; Vorbelegungen als unbestätigt markieren oder verzichten.
- **Sidecar-Ausfall:** Das Dock muss degradiert nutzbar bleiben — Jev- und Backend-Karten funktionieren,
  LLM-Karten zeigen den Ausfall. Records werden weiter geschrieben (Backend ist maßgeblich).
- **Blindleistung manueller Edits:** Ohne `config_manual_edit` sieht das Lernsystem nur KI-vermittelte
  Entscheidungen; manuelle Nutzerkorrekturen sind ein relevantes Gegensignal und werden als optionaler
  Schreibpunkt (§2.4) mitgeführt.
- **Kausalität:** Ein Grund erklärt eine Entscheidung, bewertet aber nicht ihre Wirkung. Wirkung bleibt Sache von
  Outcomes und Run-Qualitätsmessung.
- **Größe von `run-monitor.js` / `live-image-viewer.js`:** Migration nur schrittweise und mit Parität; Dock-Karten
  dürfen die alten Komponenten zunächst einbetten (Wrapper), statt sofort neu geschrieben zu werden.
- **Run-Identität:** Der Verlauf hängt heute am `run_id`-String. Ohne die stabile `run_uid` (§3.5) entstehen bei Name/Pfad-Doppelzugriff getrennte Verläufe, und Präfix-Auflösung kann den falschen Run treffen. Die Zuordnung muss exakt und zentral geführt werden.
- **Zwei Wahrheiten:** Wenn Sidecar-Sessions und Backend-Records auseinanderlaufen, gewinnt das Backend; Sessions sind
  nur Anhänge.
- **Offen:** Aufbewahrung/Löschung von Sessions **und** Decision Records (§2.8); separater Decision-Export
  ja/nein; ob `noul`/`score`-Fragen (Jev) eigene Reason-Felder brauchen
  (empirisch nur `choice` verifiziert, siehe M0-Dokument); ob mehrere Nutzer denselben Kontext sehen sollen (derzeit
  nein, lokales Einzelnutzer-Tool).
- **Pi Durable:** Nach heutigem Stand nicht nötig für Trace oder Dock. Nur P7 als Experiment.
