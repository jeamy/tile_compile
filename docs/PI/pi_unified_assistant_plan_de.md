# PI Unified Assistant — Plan (Teil B)

> **Status:** Plan, nichts davon ist implementiert. Teilplan des [Gesamtplans](pi_decision_trace_und_unified_assistant_plan_de.md).
> **Datum:** 2026-09-27, überarbeitet 2026-10-02 (Code-Review: Faktenkorrekturen, Schema- und
> Endpunkt-Klarstellungen, ergänzte Risiken und Prüfpunkte), Analyse-Nachtrag 2026-10-02 (Arbeitskontext-Modell,
> Rationale-Struktur, Jev-Annahmekette, Audit-Beziehung, Backfill, Session-Fortsetzung, P1-Gates), 2. Review
> 2026-10-02 (LLM-Anker-Record `llm_proposal`, `image_id`-Definition, Chat-Lücke Analyse-Kontext,
> `run_uid`-Normalisierung, Löschkaskade, Session-Lock, `origin`-Feld), 3. Review 2026-10-02 (Preview-/Plan-IDs,
> Preview-TTL, Live-Session-Eviction, Redaktion im Overlay, `run_key`, Kontextdrift, Granularität `llm_proposal`).
> **Betrifft:** `web_frontend_v3/js/{main.js,pages,components,state}`, `web_backend_cpp/src/routes/{pi_routes,runs_routes,app_state_routes}.cpp`, `agent_service/src/services/*`
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
  geschrieben; ein `preview_dismissed` entsteht nur bei einem tatsächlichen Dismiss-Ereignis im Dialog, nie
  rückwirkend durch einen Reload und nie durch Ablauf (§2.9).
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
   - Bestehende Runs: **nichts in das Run-Verzeichnis schreiben** (AGENTS.md). Die Zuordnung `run_uid ↔ run_key ↔
     run_id-Aliase ↔ config_sha256 ↔ Startzeit` liegt zentral in `run_index_v1.jsonl` (append-only, Overlay).
   - **Kanonischer Schlüssel ist `run_key`: der aufgelöste, normalisierte Run-Verzeichnispfad.** Ein Name ist nur
     relativ zu einem `runs_dir` eindeutig (Custom-Verzeichnis, Netzlaufwerk); gleiche Namen in verschiedenen
     Roots sind verschiedene Runs. Namen und absolute Pfade sind Aliase auf denselben `run_key`.
     `config_sha256`/Startzeit dienen nur dem **Wiederfinden** eines verschobenen Runs und führen nie automatisch
     zusammen (eine Queue kann mehrere Runs mit derselben Config starten); Mehrdeutigkeit wird dem Nutzer als
     Auswahl angeboten.
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
     (`runs_routes.cpp`), ein Backend-Restart verliert sie. Verbindlich im bestehenden
     UI-State (`runtime_dir/ui_state.json`, `app_state_routes.cpp`), Feld `pi_active_context`. `run_index_v1.jsonl` bleibt
     unveränderliche Zuordnung und enthält keine flüchtige Auswahl — damit "Zustand überlebt Reload/Restart" gilt.
3. **Resume eines Runs** (`/api/runs/<id>/resume`) setzt denselben Run-Kontext fort; es entsteht kein neuer Kontext.
   Das bestehende Resume-Feedback (`/api/pi/memories/resume-feedback`) wird als Record verknüpft (`parent_decision_id`).
4. **Löschen eines Runs** (`/api/runs/<id>/delete`):
   - Decision Records bleiben (Lerndaten, Memory-Verweise) und werden über das Overlay als `run_deleted` markiert; der
     Warum-Bereich zeigt "Run gelöscht".
   - Session und Chat-Verlauf (Nutzertext) folgen dem Löschkonzept (§2.8): Standard ist Mitlöschen oder Anonymisieren; der
     Bestätigungsdialog nennt es ausdrücklich. `rationale.user.text` der Records dieser Kontexte wird über `redact`-Links
     unterdrückt (§2.8); die Grund-Codes bleiben. Die Kaskade gilt für Run- **und** Bild-Kontext (`image_id`
     hängt am `run_uid`); der Analyse-Kontext bleibt unberührt.
   - Memories mit `provenance` auf den Run bleiben bestehen.
5. **Verschieben, Kopieren, Umbenennen:** Weil nichts am Run-Verzeichnis hängt, überleben Records und Verlauf ein
   Verschieben auf demselben System, solange `run_dir` auflösbar bleibt oder über `config_sha256`/Startzeit
   wiedergefunden wird. Wandert ein Run auf eine andere Maschine, ist **optional** ein Export "Run mit Trace" vorgesehen:
   metadata-only Records plus `run_uid`, Session nur auf ausdrückliche Wahl (enthält Nutzertext), Import mit
   Kollisionsprüfung über `run_uid` (Standard: vorhandene Records nicht überschreiben).

---

## 5. Prüfpunkte (Assistant)

- **Run-Lebenszyklus:** Aktivieren öffnet keine Session; alle fünf Zustände der Tabelle (§3.5) degradieren ohne Fehler; Zugriff über Name und Pfad liefert denselben Kontext; `run_uid`-Auflösung exakt (kein Präfix) inkl. Pfad-Normalisierung; Löschen eines Runs markiert Records, kaskadiert auf den Bild-Kontext und folgt dem Löschkonzept; die aktive Kontext-Auswahl überlebt einen Backend-Restart; nichts wird in bestehende Run-Verzeichnisse geschrieben.
- **Kontextmodell:** `image_id` ist heute genau `<run_uid>:live_edit`; der Analyse-Kontext nutzt denselben
  verallgemeinerten Chat-Pfad wie der Run (kein dritter Chat-Dienst); der Sidecar hält ein per-Session-Lock
  (max. ein aktiver Request je Kontext).
- **Run-Identität:** `run_key` ist der normalisierte Pfad; gleiche Namen in verschiedenen Roots bleiben getrennt;
  Wiederfinden über `config_sha256` führt nie automatisch zusammen.
- **Kontextdrift:** Revisionswechsel erzeugt Hinweis im Dock und Modellkontext; veraltete Vorschläge werden beim
  Lesen markiert; neue `analysis_id` beginnt neue Session; Session-Lock wird bei Abort, Disconnect und Timeout
  freigegeben.
- **UI:** Dock auf allen Tabs, Zustand überlebt Reload, Kontextwechsel aktualisiert den Thread, Warum-Bereich lädt erst beim Ausklappen nach, schmale Fenster, DE/EN,
  Tastatur-Bedienbarkeit, bestehende Shortcuts (`1/2/3`, Pfeiltasten) kollidieren nicht mit Eingabefeldern im Dock.

---

## 6. Risiken und offene Fragen (Assistant)

- **Sidecar-Ausfall:** Das Dock muss degradiert nutzbar bleiben — Jev- und Backend-Karten funktionieren,
  LLM-Karten zeigen den Ausfall. Records werden weiter geschrieben (Backend ist maßgeblich).
- **Größe von `run-monitor.js` / `live-image-viewer.js`:** Migration nur schrittweise und mit Parität; Dock-Karten
  dürfen die alten Komponenten zunächst einbetten (Wrapper), statt sofort neu geschrieben zu werden.
- **Run-Identität:** Der Verlauf hängt heute am `run_id`-String. Ohne die stabile `run_uid` (§3.5) entstehen bei Name/Pfad-Doppelzugriff getrennte Verläufe, und Präfix-Auflösung kann den falschen Run treffen. Die Zuordnung muss exakt und zentral geführt werden.
