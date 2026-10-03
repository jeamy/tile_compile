# PI Decision Trace — Plan (Teil A)

> **Status:** Plan, nichts davon ist implementiert. Teilplan des [Gesamtplans](pi_decision_trace_und_unified_assistant_plan_de.md).
> **Datum:** 2026-09-27, überarbeitet 2026-10-02 (Code-Review: Faktenkorrekturen, Schema- und
> Endpunkt-Klarstellungen, ergänzte Risiken und Prüfpunkte), Analyse-Nachtrag 2026-10-02 (Arbeitskontext-Modell,
> Rationale-Struktur, Jev-Annahmekette, Audit-Beziehung, Backfill, Session-Fortsetzung, P1-Gates), 2. Review
> 2026-10-02 (LLM-Anker-Record `llm_proposal`, `image_id`-Definition, Chat-Lücke Analyse-Kontext,
> `run_uid`-Normalisierung, Löschkaskade, Session-Lock, `origin`-Feld), 3. Review 2026-10-02 (Preview-/Plan-IDs,
> Preview-TTL, Live-Session-Eviction, Redaktion im Overlay, `run_key`, Kontextdrift, Granularität `llm_proposal`).
> **Betrifft:** `agent_service/src/services/*`, `web_backend_cpp/src/services/pi/*`, `web_backend_cpp/src/routes/pi_routes.cpp`
> **Verwandt:** [`pi_local_learning_plan_de.md`](pi_local_learning_plan_de.md),
> [`pi_memory_ablauf_de.md`](pi_memory_ablauf_de.md), [`pi_jev_decisions_plan_de.md`](pi_jev_decisions_plan_de.md),
> [`pi_jev_m0_provider_protocol_de.md`](pi_jev_m0_provider_protocol_de.md),
> [`pi_live_image_chat_plan.md`](pi_live_image_chat_plan.md)
>
> Paragraphen-Nummern sind über beide Teilpläne eindeutig: `§2.x` steht im [Trace-Plan](pi_decision_trace_plan_de.md), `§3.x` im
> [Assistant-Plan](pi_unified_assistant_plan_de.md); Verweise ohne Zusatz sind dokumentübergreifend gemeint.
> Keine Zeit- oder Aufwandsschätzungen (AGENTS.md).

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
  "kind": "config_apply | config_reject | config_undo | preview_dismissed | config_manual_edit | llm_proposal | jev_choice | jev_override | live_edit_op | live_edit_undo | live_edit_keep | live_edit_expired | memory_review | no_change",
  "actor": "user | llm | jev | rule",
  "idempotency_key": "",
  "action_plan_id": null,
  "preview_id": null,
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
  geschlossener Preview-Dialog ist `preview_dismissed` (immer `actor=user`, nur bei einem **realen** Dismiss-Ereignis
  im Dialog) und gilt nicht als Negativsignal — "nicht angewendet" ist keine Ablehnung. **Ablauf ist kein Record:**
  Ein Preview, der nie angewendet wurde und dessen TTL (§2.9) abgelaufen ist, wird beim Lesen als "abgelaufen"
  abgeleitet. Ein Sweep schreibt keine Records, und niemand erfindet Timeout-Ereignisse.
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
  **Granularität:** ein `llm_proposal` je Antwort (Vorschlagsmenge), nicht je Einzelempfehlung. Die einzelnen
  Empfehlungen stehen als `alternatives[]` (mit stabiler `id`, `label`, `subject.paths`) — analog zu den Kandidaten
  von `jev_choice`. Das Kind (`config_apply`/`config_reject`) hat genau einen Parent; welche Alternativen
  gewählt wurden, steht dort in `subject.paths` plus `alternatives[].selected`. `rationale.basis[].ref` verweist
  auf `decision_id` + Alternativen-`id` (bei Altbestand auf `analysis_id` + Index). Ein `llm_proposal` ohne
  Kind-Record gilt als "unbeantwortet" (beim Lesen abgeleitet) und ist **kein** Negativsignal. Ohne ihn wäre `actor=llm` unerreichbar und die Kette
  Empfehlung → Apply nur bei Jev rekonstruierbar.
- `evidence.metrics_ref` ist eine artefakt-relative Referenz (relativer Pfad oder Artefakt-Schlüssel, nie
  ein absoluter Pfad — sonst verletzt er metadata-only). `fact_ids` verweisen auf die stabilen `fact_id`s
  aus `pi_context_protocol_compression_plan_de.md`.
- `idempotency_key` (z. B. `action_plan_id` + Ereignis) verhindert Doppel-Records bei Retries und
  Doppelklicks; das Backend dedupliziert darüber.
- `action_plan_id` verknüpft den Record mit dem validierten Patch — ohne ihn ist die Kette Empfehlung →
  Preview → Apply nicht eindeutig rekonstruierbar. **Beide IDs gibt es im Backend heute nicht** (weder in
  `pi_routes.cpp` noch in `services/pi/*`; Preview und Apply sind zustandslos). Sie werden in P1 eingeführt (§2.9).
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
| Preview geschlossen | Action-Plan-Dialog (expliziter Dismiss) | `preview_dismissed` (`actor=user`, nur bei realem Dismiss, nicht durch Reload oder Ablauf), kein Reason-Zwang, kein Negativsignal |
| Live-Session verfällt | `evict_expired` (Alter/Kapazität) | `live_edit_expired` (`actor=rule`) |
| Manuelle Config-Änderung ohne Empfehlung | Parameter-Tab (optional, ab P5) | `config_manual_edit`, `no_reason_given`-Default |

Reason-Abfrage ist **optional und niedrigschwellig**: ein Klick auf Chips, überspringbar. Bei `Reject`, `Deprecate`,
Undo und Jev-Override wird sie angeboten, bei `Accept`/`Apply` nur als optionales Feld.

**Live-Edit-Granularität (Verhaltensänderung, bewusst):** Heute schreibt `pi_live_edit_recorder` das Memory erst beim
Close und nicht pro Undo/Redo-Schritt (`pi_memory_ablauf_de.md` §1.2). Decision Records weichen davon ab: `live_edit_op` und
`live_edit_undo` entstehen **während** der Session, `live_edit_keep` beim Close mit dem Retained-Terminalwert
(derselbe, den der Recorder für das Memory nutzt). Das Memory-Verhalten bleibt unverändert; nur der Trace ist
feiner. Datenvolumen ist begrenzt, indem Parameter-Adjust-Ströme (Slider) je Op zu einem Record zusammengefasst werden
(letzter Wert beim Loslassen/Commit). Ein Absturz ohne Close hinterlässt eine erkennbar offene Kette, kein implizites
`keep`. **Eviction:** Das Backend wirft Live-Image-Sessions nach 1800 s bzw. über 5 Sessions aus dem Speicher
(`LiveImageSessionStore::evict_expired`, aufgerufen in `pi_routes.cpp`). Das ist der häufigste Weg zu einer Kette
ohne Close; die Eviction schreibt `live_edit_expired` (`actor=rule`, Grund `age`/`capacity`, letzter Op-Stand), kein
`live_edit_keep`. Ob und wie das Memory in diesem Fall (heute vermutlich kein Kandidat) entsteht, ist vor P1 am Code
zu verifizieren; das Memory-Verhalten bleibt unverändert. `pi_memory_ablauf_de.md` ist bei Umsetzung entsprechend zu ergänzen.

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
Typdefinitionen des npm-Tarballs gegen die zuvor installierte 0.87.1; das Upgrade auf `^1.0.0` ist im Repo bereits eingespielt, Commit `94f9223b`):

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
  Durchgesetzt wird das über ein **per-Session-Lock im Sidecar** (Haltezeit an die Request-Lebensdauer gebunden, Freigabe in `finally`, bei Abort und bei Client-Disconnect, zusätzlich Watchdog-Timeout = Request-Timeout; bei Ablauf Freigabe mit Hinweis im Thread, nicht still. Ein Sidecar-Neustart leert alle Locks. Mehrere Sidecar-Instanzen auf demselben `sessionDir` sind nicht unterstützt und werden über eine Lock-Datei im `sessionDir` erkannt) — die Session-Datei ist append-only JSONL;
  zwei parallele `AgentSession`s auf derselben Datei wären korruptionsgefährdet. Eine
  nicht auffindbare Session (gelöscht, Dateiverlust) startet eine neue und schreibt einen Hinweis in den Thread; sie
  bricht nie das Apply. Frame-Analyse (`frameAnalysisService`) bleibt ein abgeschlossener Einzellauf je Analyse mit
  eigener Session, kein fortgesetzter Chat.
- **Kontextdrift (Revisionen, neue Analysen):** Jede Session notiert beim letzten Turn ihre `context_basis`
  (`analysis_id`, `revision_id`, `config_sha256`) als `appendCustomEntry`. Beim nächsten Turn vergleicht das Backend mit
  der aktuellen Basis. Bei Abweichung: (a) das Backend liefert dem Modell die aktualisierten Fakten und einen
  Kontextwechsel-Hinweis (nicht still); (b) das Dock zeigt "Basis geändert: Revision X → Y"; (c) frühere
  `llm_proposal`/`jev_choice` auf älterer Basis werden beim Lesen als "veraltet" markiert (kein Record). Apply
  veralteter Vorschläge läuft unverändert durch `validate-config` gegen die aktuelle Config. Wechselt die
  `analysis_id` (neue Analyse), beginnt eine neue Session; die alte bleibt lesbar.
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
- **Redaktion statt Mutation.** `decision_links_v1.jsonl` kennt einen Link-Typ `redact` (Ziel: `decision_id`,
  Feld `rationale.user.text`). Leser unterdrücken den Text, sobald ein `redact`-Link existiert. Physisch entfernt wird er
  nur bei **Kompaktierung**: das Backend schreibt `decisions_v1.jsonl` unter Lock in eine Temp-Datei neu und ersetzt sie
  atomar. Das ist der einzige erlaubte Rewrite. Grund-Codes (Chips) enthalten keine personenbezogenen Daten und
  bleiben erhalten.
- Redaktion wird ausgelöst durch: Löschen eines Runs/Bild-Kontexts (§3.5), den Nutzer (Einzelrecord) und Ablauf der
  Aufbewahrungsfrist.
- `pi.memories-export` bleibt unverändert metadata-only und enthält keine Decision Records. Ob es einen
  separaten, ebenfalls metadata-only Decision-Export gibt (forensisch, als Eingabe der Regelkalibrierung in
  §2.7), ist eine offene Entscheidung; `rationale.user.text` gehört in keinen Fall hinein.

### 2.9 Plan- und Preview-Objekte (neu in P1)

Preview und Apply sind heute zustandslos. Damit Kette, Idempotenz und Dismiss belegbar werden, bekommen sie ein
persistiertes Gegenstück:

- `POST /api/pi/action-plans/preview` legt ein **Preview-Objekt** an (`preview_id`, `action_plan_id`, Hash des Plans,
  Basis-`config_sha256`, `created_at`, `expires_at`) und liefert die IDs mit der bestehenden Antwort zurück (additiv,
  kein Bruch für Altclients). Speicherort zentral, getrennt von den Records.
- `POST /api/pi/action-plans/apply` nimmt optional `preview_id`. Fehlt er (Altcaller, Scan-AI-Apply-Route), erzeugt das
  Backend eine `action_plan_id` und markiert den Record `origin="live"` ohne Preview-Bezug. Ein Apply gegen ein
  abgelaufenes Preview wird nicht blockiert, sondern wie bisher gegen die **aktuelle** Config revalidiert
  (`validate-config`); die Sicherheitslogik ändert sich nicht.
- **TTL:** `expires_at` ist eine Konfiguration des Backends. Ablauf erzeugt keinen Record (§2.2).
- `idempotency_key` = `action_plan_id` + Ereignisart.

---

## 5. Prüfpunkte (Trace)

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
- **Plan-/Preview-Objekte (§2.9):** IDs additiv in der Preview-Antwort; Apply ohne `preview_id` funktioniert
  unverändert; abgelaufenes Preview erzeugt keinen Record und blockiert kein Apply; Doppelklick auf Apply erzeugt
  genau einen Record.
- **Live-Edit:** Eviction schreibt `live_edit_expired` und kein `live_edit_keep`; Memory-Verhalten unverändert.
- **Redaktion:** `redact`-Link unterdrückt `rationale.user.text` beim Lesen; Kompaktierung entfernt ihn physisch
  (atomarer Rewrite unter Lock); Exports enthalten ihn nie.
- **Datenschutz:** Exports enthalten keine Session-Texte; Records enthalten keine Rohdaten/Pfade.

---

## 6. Risiken und offene Fragen (Trace)

- **Reason-Fatigue:** Zu viele Abfragen werden übersprungen. Deshalb optional, Chips, vorbelegt aus Kontext — und
  `no_reason_given` bleibt explizit. Gegenläufiges Risiko: vorbelegte Chips erzeugen Suggestionsbias und
  verfälschen die Grund-Daten; Vorbelegungen als unbestätigt markieren oder verzichten.
- **Blindleistung manueller Edits:** Ohne `config_manual_edit` sieht das Lernsystem nur KI-vermittelte
  Entscheidungen; manuelle Nutzerkorrekturen sind ein relevantes Gegensignal und werden als optionaler
  Schreibpunkt (§2.4) mitgeführt.
- **Kausalität:** Ein Grund erklärt eine Entscheidung, bewertet aber nicht ihre Wirkung. Wirkung bleibt Sache von
  Outcomes und Run-Qualitätsmessung.
- **Zwei Wahrheiten:** Wenn Sidecar-Sessions und Backend-Records auseinanderlaufen, gewinnt das Backend; Sessions sind
  nur Anhänge.
- **Offen:** Aufbewahrung/Löschung von Sessions **und** Decision Records (§2.8); separater Decision-Export
  ja/nein; ob `noul`/`score`-Fragen (Jev) eigene Reason-Felder brauchen
  (empirisch nur `choice` verifiziert, siehe M0-Dokument); ob mehrere Nutzer denselben Kontext sehen sollen (derzeit
  nein, lokales Einzelnutzer-Tool).
