# PI Decision Trace und Unified Assistant — Gesamtplan

> **Status:** Plan. Umsetzung läuft auf Branch `feature/pi-decision-trace-assistant` (siehe „Umsetzungsstand“ unten). Dieses Dokument ist die Übersicht; Details stehen in zwei Teilplänen.
> **Datum:** 2026-09-27, überarbeitet 2026-10-02 (Code-Review: Faktenkorrekturen, Schema- und
> Endpunkt-Klarstellungen, ergänzte Risiken und Prüfpunkte), Analyse-Nachtrag 2026-10-02 (Arbeitskontext-Modell,
> Rationale-Struktur, Jev-Annahmekette, Audit-Beziehung, Backfill, Session-Fortsetzung, P1-Gates), 2. Review
> 2026-10-02 (LLM-Anker-Record `llm_proposal`, `image_id`-Definition, Chat-Lücke Analyse-Kontext,
> `run_uid`-Normalisierung, Löschkaskade, Session-Lock, `origin`-Feld), 3. Review 2026-10-02 (Preview-/Plan-IDs,
> Preview-TTL, Live-Session-Eviction, Redaktion im Overlay, `run_key`, Kontextdrift, Granularität `llm_proposal`).
> **Teilpläne:**
> - [pi_decision_trace_plan_de.md](pi_decision_trace_plan_de.md) — Teil A: Decision Records, Reason-Codes, Memory v3, Session-Persistenz, Plan-/Preview-Objekte,
>   Aufbewahrung/Redaktion (§0, §2.x, Prüfpunkte, Risiken).
> - [pi_unified_assistant_plan_de.md](pi_unified_assistant_plan_de.md) — Teil B: Dock, Karten, Arbeitskontext, Jev als Karte, Frontend-Architektur, Backend-Seite,
>   Run-Lebenszyklus (§0, §3.x, Prüfpunkte, Risiken).
> **Verwandt:** [`pi_local_learning_plan_de.md`](../pi_local_learning_plan_de.md),
> [`pi_memory_ablauf_de.md`](../pi_memory_ablauf_de.md), [`pi_jev_decisions_plan_de.md`](../pi_jev_decisions_plan_de.md),
> [`pi_jev_m0_provider_protocol_de.md`](../pi_jev_m0_provider_protocol_de.md),
> [`pi_live_image_chat_plan.md`](../pi_live_image_chat_plan.md)
>
> Paragraphen-Nummern sind über beide Teilpläne eindeutig: `§2.x` steht im [Trace-Plan](pi_decision_trace_plan_de.md), `§3.x` im
> [Assistant-Plan](pi_unified_assistant_plan_de.md); Verweise ohne Zusatz sind dokumentübergreifend gemeint.
>
> Keine Zeit- oder Aufwandsschätzungen (AGENTS.md). Der Plan nennt Umfang, Abhängigkeitsreihenfolge, Risiken und Prüfpunkte.

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

## 4. Reihenfolge (Abhängigkeiten, keine Zeitangaben)

1. **P0 — Basis:** Upgrade auf `@earendil-works/pi-coding-agent` 1.0.0 (API-Verifikation erledigt, s. §2.5:
   keine Breaking Changes für die genutzten Flächen); Inventar aller Stellen, an denen Entscheidungen
   entstehen (Tabelle 2.4) gegen den Code verifizieren; Reason-Code-Katalog (`pi.user-reason-codes.v1`)
   entwerfen.
2. **P1 — Decision-Record-Backend** (inkl. `run_uid` in `pi_run_provenance.json` neuer Runs und zentraler Tabelle `run_index` in `pi_store_v2.sqlite`, §3.5) (Gate vor Aktivierung: Aufbewahrung/Rotation und Löschkonzept für Records und Sessions entschieden, §2.8): Schema, SQLite-Speicher `pi_store_v2.sqlite` (Memories **und** Decision Records, keine Altbestandsuebernahme) mit `decision_records` + `decision_links`,
   Plan-/Preview-Objekte mit IDs und TTL (§2.9), Idempotenz und `actor`×`basis`-Validierung, Redaktion (§2.8), Schreibpunkte in Apply, Review, Live-Edit-Recorder,
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

## 5. Übergreifende Prüfpunkte

- **Kein Run-Start, kein Backend-Start** durch Tests oder Migration (AGENTS.md).

Detail-Prüfpunkte: [Trace](pi_decision_trace_plan_de.md#5-prüfpunkte-trace), [Assistant](pi_unified_assistant_plan_de.md#5-prüfpunkte-assistant).

---

## 6. Übergreifende Risiken

- **Pi Durable:** Nach heutigem Stand nicht nötig für Trace oder Dock. Nur P7 als Experiment.

Detail-Risiken: [Trace](pi_decision_trace_plan_de.md#6-risiken-und-offene-fragen-trace), [Assistant](pi_unified_assistant_plan_de.md#6-risiken-und-offene-fragen-assistant).

---

## 7. Umsetzungsstand

| Phase | Stand |
|---|---|
| P0 Upgrade Pi 1.0 | `package.json` auf `^1.0.0` (Commit `94f9223b`), installiertes SDK `1.0.0` verifiziert; Sidecar-TS-Build und 68 Tests bestanden. |
| P1 Speicher/Records | `PiDatabase`: SQLite/WAL, `pi_store_v2.sqlite`, Schema v6, kein Import alter JSONL-/v1-Dateien. Memories und Decision Records mit Validierung, Idempotenz, Overlay-Links und logischer Redaktion. System-SQLite oder gepinnter FetchContent-Fallback. Record-Leseendpunkte implementiert; **Schreibpunkte noch deaktiviert**. |
| P1 Preview/Apply | Persistente Preview-IDs, stabile Plan-IDs, Config-Hash, TTL (Default 1800 s, konfigurierbar), read-time Expiry, Lookup/Dismiss-Routen. Apply revalidiert die aktuelle Config, prueft den Plan-Bezug und liefert bei erfolgreicher Wiederholung dieselbe Revision. Config-Lock umfasst Revalidierung und Save. Crash-Recovery zwischen Datei und SQLite noch offen. |
| P1 Nutzergruende | Versionierter Katalog, DE/EN-Labels, Validator und `GET /api/pi/reason-codes`; Anbindung an echte Decision-Schreibpunkte und Dock-Chips noch offen. |
| P1 Jev | `PiJevStore`: Status, State, Source, Candidates, Request, Response, Proposal, Events, Revisions-/Outcome-Bezuege und Verarbeitungsmarker in derselben SQLite-Datei. Batch-Apply transaktional; keine neuen Vorschlagsdateien. Menschlicher Policy-Override und Offline-Evaluationsbundles bleiben Konfigurations-/Testdateien. |
| P1 Run-Identitaet | `PiRunIndex` mit exakten normalisierten Pfaden, UID und Alias-Tabelle. Neue Runs erhalten UID in Provenance; historische Runs nur zentrale Zuordnung, ohne Artifact-Rewrite. Fingerprints schlagen Kandidaten vor, mergen nie. `set-current` liefert Kontext und persistiert `pi_active_context`; Lookup-Endpunkt vorhanden. Explizite Neuverknuepfung verschobener Runs per bestaetigtem Relink-Endpunkt implementiert (bekannte UID, erhaltene Alt-Aliase, Konfliktschutz, keine Artefakt-Aenderung). Relink-UI, Kontextwechsel im Dock und session-/history-UID-Anbindung noch offen. |
| P1 Run-Lerndaten | Dauerhafte, versionierte Run-Snapshots mit Config (inkl. passender Runner-Defaults), Light-/Kalibrationsherkunft, Statistik und relevanten Phasenereignissen. Erfassung bei Start/Abschluss/Resume/Report und vor Dateiloeschung. SQLite-Fehler blockieren Loeschung; Datei-Lifecycle und expliziter Lern-Ausschluss sind getrennt. Roh-FITS werden nicht gespeichert; optional begrenztes PNG. [Details und Grenzen](pi_run_learning_archive_de.md). Archiv-/Loeschdialog-UI noch offen. |
| P1 Tests | 22 ausgewaehlte Backendtests (PI, Contract, AI-Routen, Run-Start/-Stop und Archiv) sowie 19 C++-FITS-/Config-Testfaelle mit 139 Assertions bestanden. Backend/Runner gebaut; Modell-Testdaten isoliert. Zusaetzlicher unveraenderter Resume-Status-Test scheitert am Feldnamen-Widerspruch `duration_seconds` versus `duration_s` (kein separat gebauter Basisstand-Test). |
| P2–P7 | offen |

Gates: Aufbewahrung/Loeschkonzept (§2.8) ist vor *Aktivierung* der Schreibpunkte zu entscheiden.
[Konkrete Entscheidungsvorlage](pi_retention_policy_draft_de.md) liegt vor, ist nicht freigegeben.
Die Record-Leseendpunkte sind bereits angeschlossen, permanente Decision-Schreibpunkte noch nicht.
P2–P7 und der Unified Assistant Dock sind weiterhin offen; die Backend-Bausteine sind kein fertiger UI-Umbau.
