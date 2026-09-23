# PI Jev — Detaillierter Implementierungsplan

> **Stand:** 2026-09-22.
> **Status:** M0 abgeschlossen, M1 (State-Builder) umgesetzt und getestet; M2-M7 offen; noch keine Routen-/UI-Verdrahtung.
> **Verbindliche Reihenfolge:** Pre-Run-Beratung zuerst, Post-Run-Beratung danach.
> **Lieferumfang:** Vorschläge; kein automatischer Run/Resume und kein zweiter Bildeditor.

Grundlagen: [Zielbild](pi_jev_decisions_plan_de.md), [korrigierter Regelkatalog](pi_scan_pre_rules_de.md), [lokaler Lernplan](pi_local_learning_plan_de.md), [M0-Feld-Inventar](pi_jev_m0_field_inventory_de.md), [M0-Provider-Protokoll](pi_jev_m0_provider_protocol_de.md), [M0-HARD-RULE-Review](pi_jev_m0_hard_rule_review_de.md).

## 1. Abgrenzung und Abhängigkeiten

M0 -> M1 -> M2 -> M3 -> M4 ergibt eine vollständige, zunächst experimentelle Pre-Run-Beratung. M5 entscheidet über die Freigabe einzelner Kandidaten für reguläre Vorschläge. M6 ergänzt Post-Run. M7 schließt Integration, Betriebsdokumentation und Regression ab.

M3 kann nach eingefrorenem M0-Vertrag unabhängig von der internen Implementierung des Builders entwickelt werden; die Integration wartet auf M2. Alle neuen Dateien und Schnittstellen unten sind geplant, sofern nicht ausdrücklich als bestehend bezeichnet.

Keine Veränderung der Rekonstruktionsmethode, keine neue Runner-Netzwerkabhängigkeit, kein Training eines neuen numerischen Optimierers. kNN bleibt Shadow. Live Image Chat und GUI-Kurven bleiben im bisherigen PI-Pfad.

## 2. Konkrete Eingriffspunkte

| Bestehende Datei/Komponente | Aufgabe in dieser Umsetzung |
|---|---|
| `web_backend_cpp/src/routes/ai_routes.cpp` | Vorhandene Scan-Analyse/Apply- und Completion-Analyse-Flows anbinden; Jev-Services aufrufen, keine neue Entscheidungslogik im Routenmonolith |
| `web_backend_cpp/src/routes/scan_routes.cpp` | Vorhandene Scan-Resultate/IDs als Quellen; keine zweite Scan-Ausführung durch Beratung |
| `web_backend_cpp/src/services/pi/pi_feature_vector.cpp` | Bestehenden kNN-Vertrag bewahren; Messquellen für neuen Builder wiederverwenden |
| `web_backend_cpp/src/services/pi/pi_recommendation_validator.cpp` | Expliziten atomaren Validierungspfad ergänzen, Legacy-Teilpatchverhalten nicht versehentlich global ändern |
| `web_backend_cpp/src/services/pi/pi_parameter_catalog.cpp` | Metadaten referenzieren; keine automatische Freigabe aus Description/Enum |
| `web_backend_cpp/src/services/pi/pi_action_plan.cpp` und `pi_action_validator.cpp` | Jev-Vorschlag ohne Ausführungsaktion darstellen; Kandidaten-/Evidenzbindung prüfen |
| `web_backend_cpp/src/services/pi/pi_storage_paths.cpp` | Persistenz unter bestehendem Backend-State-Root; keine impliziten Writes in fremde Runs |
| `web_backend_cpp/src/services/pi/pi_outcome_recorder.cpp` | Vorschlag -> tatsächliche Config -> Ergebnis verknüpfen, Labels sauber trennen |
| `web_backend_cpp/src/services/config_revisions.cpp`, `run_inspector.cpp`, `routes/runs_routes.cpp` | Aktuelle Config und bestehende Resume-Machbarkeit wiederverwenden |
| `agent_service/src/config.ts`, `types.ts`, `server.ts` | Optionalen Decisions-Adapter konfigurieren und anbieten; nutzt den bereits vorhandenen `.env`-Key `JEV_OPENROUTER_API_KEY`, siehe 2.1 |
| `agent_service/src/services/frameAnalysisService.ts` | Bestehende PI-Beratung als unabhängige Vergleichs-/Fallback-Option erhalten |
| `web_frontend_v3/js/pages/input-scan.js`, `js/state/scan-state.js` | Vorschläge, Status, Wiederherstellung und atomare Übernahme |
| `web_frontend_v3/js/pages/tools.js`, `js/pages/ai-empfehlung.js` | Neue, eigenständige Jev-Konfigurationskarte neben der bestehenden AI-&-API-Karte (siehe 2.1); kein zusätzlicher Eintrag im bestehenden Provider-Dropdown |
| `web_frontend_v3/js/pages/parameter.js` | Dritten Sub-Tab „Jev-Empfehlungen“ neben `Parameter`/`AI Empfehlung` ergänzen (bestehendes `switchView`-Muster) |
| `web_frontend_v3/js/pages/run-monitor.js` | Später Post-Run-Befunde und geprüfte Resume-Vorbereitung |
| `web_frontend_v3/i18n/de.json`, `en.json` | Alle neuen Nutzertexte in beiden Sprachen |
| `web_backend_cpp/CMakeLists.txt`, `agent_service/package.json` | Neue Quellen und passende Tests registrieren |

Neue Backend-Module jeweils mit Header unter `include/services/pi/` und Implementierung unter `src/services/pi/`:

- `pi_decision_state`: Quellenvalidierung und kanonischer State.
- `pi_pre_rules`: Voraussetzungen, Befunde, freigegebene Kandidaten.
- `pi_decision_policy`: Allowlist, Locks, Auswahl-/Enthaltungsprüfung, atomare Validierung.
- `pi_decision_service`: Orchestrierung, Sidecar-Aufruf, Status und Persistenz.
- `pi_post_run_advisor`: erst M6; Zwischenstandsdiagnose und Resume-Vorschlag.

Adapter neu: `agent_service/src/services/decisionsService.ts`. Versionierte Anwendungsschemas und Kandidaten-/Fragenkataloge neu unter `web_backend_cpp/config/pi_decisions/`. Diese sind Source-Artefakte; Laufzeitantworten gehören nicht in diesen Ordner.

### 2.1 Eigenständige Jev-Konfigurationsoberfläche (Nutzerentscheidung 2026-09-22)

Jev wird als zweite, parallele, von der bestehenden PI-Beratung unabhängige Quelle implementiert — nicht als weitere Option in deren bestehendem Auswahlpfad. Grund: `web_frontend_v3/js/pages/ai-empfehlung.js` (Karte unter Tools -> AI & API) verwaltet heute genau **einen** globalen Provider/Modell/Key-Slot (`persistAiProviderModelConfig()` patcht `API_ENDPOINTS.ai.config`, kein Rollen-/Zweck-Feld). Ein zusätzlicher Eintrag im dortigen Provider-Dropdown würde PI und Jev gegenseitig exklusiv machen, statt beide gleichzeitig aktiv zu halten — das widerspricht dem Unabhängigkeitsziel aus Abschnitt 1 direkt.

Konsequenzen für die Umsetzung:

- **Tools -> AI & API**: eine zweite, eigenständige Karte „Jev (Decisions API)“ neben der bestehenden AI-&-API-Karte, mit eigenem `PI_DECISIONS_MODE`-Schalter (`off|shadow|suggest`), eigenem Modellfeld (gepinnte Kennung, Abschnitt 7/M3) und eigenem API-Key-Feld. Eigener Backend-Endpoint statt Wiederverwendung von `ai.config`, damit Umschalten der bestehenden PI-Provider-Auswahl den Jev-Zustand nicht mitbewegt.
- **Key-Namensraum**: Jev läuft über OpenRouters Alpha-Decisions-Endpunkt (empirisch verifiziert, [M0 Provider-Protokoll](pi_jev_m0_provider_protocol_de.md)), Modell-ID `typesafe/jev-1.13`. `.env` enthält dafür bereits einen eigenständigen, vom generischen `OPENROUTER_API_KEY` getrennten Key `JEV_OPENROUTER_API_KEY` — `agent_service/src/config.ts` braucht einen entsprechenden `jev_openrouter`-Allowlist-Eintrag (oder passende Umbenennung), nicht den bestehenden `openrouter`-Eintrag. Die Karten-Trennung aus 2.1 (eigene Jev-Karte statt Dropdown-Eintrag in der bestehenden AI-&-API-Karte) bleibt nötig, weil diese Karte weiterhin nur einen einzigen aktiven Provider/Modell-Slot verwaltet und ein zusätzlicher Dropdown-Eintrag Jev und die bestehende PI-Beratung gegenseitig exklusiv machen würde.
- **Parameter-Tab**: `web_frontend_v3/js/pages/parameter.js` hat heute exakt zwei Sub-Tabs (`Parameter`, `AI Empfehlung`) über ein einfaches `switchView(view, page, paramTab, aiTab)`-Muster (Klassen-/Sichtbarkeits-Toggle, lazy gemountete Containerseite). Ein dritter Tab „Jev-Empfehlungen“ ist eine direkte Erweiterung desselben Musters (neuer Button + neuer lazy gemounteter Container + zusätzlicher `switchView`-Zweig), keine strukturelle Änderung nötig.
- Über diese drei Stellen hinaus (Jev-Karte unter Tools, dritter Parameter-Tab, spätere Run-Monitor-Anbindung in M6) wird kein weiterer UI-Einstiegspunkt ergänzt, um den in Abschnitt 1 festgelegten Scope nicht zu überschreiten.

## 3. Daten- und Schnittstellenverträge

### 3.1 State: `pi.decision-state.v1`

```json
{
  "schema_version": "pi.decision-state.v1",
  "domain": "pre_run",
  "identity": {
    "dataset_fingerprint": "sha256:...",
    "scan_id": "...",
    "scan_hash": "sha256:...",
    "config_hash": "sha256:...",
    "locks_hash": "sha256:...",
    "software_version": "..."
  },
  "measurement_contract": {
    "method_version": "scan-metrics:...",
    "pixel_grid": "source",
    "calibration_state": "unknown"
  },
  "coverage": {"detected": 120, "measured": 120, "read_ok": 118},
  "groups": [],
  "session_facts": {},
  "capabilities": {},
  "base_config": {},
  "locked_paths": [],
  "blocking_findings": []
}
```

Beispiel ist eine Formskizze, kein vollständiger zulässiger Beratungsfall. Schema fordert verwendete Gruppenmetriken und alle kandidatenabhängigen Pflichtfelder.

Messwertvertrag: `{value, status, unit, source_ref, method_version, valid_count, total_count}`. `status` ist `valid|missing|invalid|not_applicable`; bei nicht gültigem Status ist `value=null`. Rohdaten, Diagnosewerte und daraus abgeleitete Quotienten getrennt benennen.

Dataset-Identität: kanonisch sortiertes Manifest der tatsächlich selektierten Inputs einschließlich relativer ID, Größe, mtime und verfügbarem Inhaltsdigest plus Ausschlussliste. Ist kein Inhaltsdigest vorhanden, kennzeichnet `identity_strength=metadata` die Grenze: keine Behauptung inhaltlicher Identität. Vor Apply Manifest/Scan-Revision erneut prüfen; für wissenschaftliche Paarvergleiche Inhaltsidentität nachweisen. Absolute Quellpfade bleiben lokal.

JSON-Kanonisierung fest definieren: sortierte Objektkeys, stabile Arrayreihenfolge, keine NaN/Infinity, keine volatil generierten Zeitfelder im semantischen Hash. Sampling-/Detektor- und Gruppierungsrevision gehören in den Hash.

### 3.2 Kandidaten- und Fragenkatalog

Katalogversion enthält exakte Pfade/Werte, atomare Gruppe, Preconditions, Evidenzreferenzen, Textschlüssel, erwarteten Mechanismus, Risiko und Freigabestatus. Erste Kandidaten: `keep_current`, `insufficient_evidence`, `enable_adaptive_weights` gemäß Regelkatalog.

`decision_questions.yaml` enthält Anwendungsspezifikation, keine ungeprüft direkt weitergeleiteten Provider-Objekte. Der Adapter übersetzt explizit in den getesteten Provider-Vertrag. Fragen werden nicht blind aus Parameterbeschreibungen erzeugt.

Keine freien Config-Werte in der Modellantwort. Auswahl einer ID, deren Voraussetzungen nicht erfüllt sind, ergibt `invalid_response`. Unabhängige Szenariofragen dürfen nur bei vorhandener Evidenz gestellt werden und sind nicht Voraussetzung für den ersten Kandidaten.

### 3.3 Interner Sidecar-Vertrag

Geplant: `POST /decisions` im vorhandenen Sidecar, nur durch Backend benutzt.

Request: `pi.decisions.request.v1` mit `request_id`, `state_hash`, `question_set_version`, providergeeigneter State-Projektion und zulässigen Fragen/Kandidaten. Kein Client-seitig vorgegebener Ziel-URL oder API-Key.

Response: `pi.decisions.response.v1` mit gebundenen IDs/Hashes, `status`, normalisierten Antworten, angefragter Modellkennung und tatsächlich vom Provider gelieferter Modellkennung, falls vorhanden. Fehlende Provider-Version bleibt ausdrücklich unbekannt. `status=ok|unavailable|invalid_response`.

Normalisierte Antwort enthält Auswahl und getrennte Rohwahrscheinlichkeiten. Keine erfundene `confidence`, falls der Provider sie nicht so liefert; Ableitungen wie Maximum/Marge werden als solche benannt.

### 3.4 Vorschlag: `pi.config-proposal.v1`

Pflichtfelder: `proposal_id`, `domain`, Identitätsblock, `state_hash`, `candidate_id/version`, `question_set_version`, `policy_version`, `model`, `status`, `updates`, `evidence_refs`, `reason_codes`, `review_required`, `validation`, `created_at`.

`updates` enthält `path`, `old_value`, `value`, `group_id`. Leere Updates bei `no_change`, `abstain`, `blocked`, `unavailable` oder `stale`. Die Anwendung darf diese Zustände nicht als leere erfolgreiche Mutation verbuchen.

Validierungsstatus enthält separat Schema-, Policy-, Evidenz- und Gesamtconfig-Prüfung. Modell-Confidence wird nicht als Validierungsstatus missbraucht.

### 3.5 Backend-API und vorhandener Apply-Pfad

Geplante neue Beratungsschnittstellen:

- `POST /api/scan/decisions`: vorhandene Scan-ID, aktueller Config-Entwurf und Locks -> asynchroner Beratungsjob/ID; kein Scan- oder Runner-Start.
- `GET /api/scan/decisions/<proposal_id>`: persistierter Status und Ergebnis; bestehende Auth-/Scope-Regeln übernehmen.
- Vorhandenes `POST /api/scan/analysis/apply`: additiver Jev-Zweig anhand `proposal_id` mit erwarteten Dataset-/Scan-/Config-/Lock-Hashes. Nur ganze Kandidatengruppe übernehmen; bestehende PI-Aufrufer unverändert unterstützen.

State wird serverseitig aus den Scanquellen rekonstruiert. Browser liefert keine autoritativen Messwerte. Bei veraltetem Entwurf `409 PROPOSAL_STALE`; bei Policy-/Config-Fehler strukturierte Ablehnung; keine Rückkehr auf eine alte vollständige Config.

Apply führt Compare-and-swap auf der Entwurfsrevision aus und validiert unmittelbar vor der Mutation. Wiederholte gleiche Übernahme liefert dieselbe Revision; ein inzwischen geänderter Entwurf führt zum Konflikt. Kein Job vom Typ Run/Resume wird erzeugt.

### 3.6 Persistenz und Replay

Unter Backend-State-Root: `pi_decisions/<proposal_id>/state.json`, `proposal.json`, normalisierte `response.json` und append-only Ereignisse. Atomare Dateiersetzung für Status; Replay braucht gespeicherten State, Kandidaten- und Policy-Version, nicht einen neuen Modellcall.

Große Rohantworten nur größenbegrenzt und ohne Secrets speichern. Request-Hash umfasst State, Modellkennung, Fragen- und Kandidatenkatalog. Gleichzeitige gleiche Anfragen deduplizieren; Cache nicht über Versionen hinweg verwenden.

Run-Outcomes referenzieren später `proposal_id` und die tatsächlich gestartete Config. Vorhandene Runs bei Beratung nicht umschreiben. Für neue Runs darf die normale Run-Erzeugung eine Referenz übernehmen; nachträgliche Beratungsdaten bleiben im Backend-State-Store.

## 4. M0 — Verträge, Quellen und Testgrundlage einfrieren

**Status:** inhaltlich abgeschlossen (2026-09-22). **Abhängigkeit:** keine.

Arbeit:

- [x] Aktuelle Scan-JSONs, Feature-Vektor, Schema und ausführbare Validatoren inventarisieren; Herkunft jedes State-Feldes tabellarisch hinterlegen. -> [Feld-Inventar](pi_jev_m0_field_inventory_de.md), gegen `cli_main.cpp`/`metrics.cpp`/`pi_feature_vector.cpp` verifiziert.
- [x] Geschützte Pfade, Nutzer-Locks und erste Kandidaten-Allowlist im maschinenlesbaren Katalog definieren. -> `web_backend_cpp/config/pi_decisions/protected_paths_v1.json`, `candidates_v1.json`.
- [x] Anwendungsschemas für State, Vorschlag, Adapter und Katalog erstellen; Status-/Fehlercodes festlegen. -> `web_backend_cpp/config/pi_decisions/schemas/pi.decision-state.v1.schema.json`, `pi.config-proposal.v1.schema.json`, `pi.decisions.request.v1.schema.json`, `pi.decisions.response.v1.schema.json` (JSON Schema draft-07, wie `tile_compile.schema.json`).
- [x] Provider-Protokoll aus offiziellen Quellen verifizieren; redigierte Request-/Response-/Fehlerfixtures erstellen. Live-Verifikation ausdrücklich von Mock-Tests unterscheiden. -> [M0 Provider-Protokoll](pi_jev_m0_provider_protocol_de.md), per echtem Aufruf gegen `openrouter.ai/api/alpha/decisions` mit dem vorhandenen `JEV_OPENROUTER_API_KEY` verifiziert (2026-09-22); Wire-Schemas `schemas/openrouter.decisions.request.v1`/`.response.v1`, Fixtures unter `fixtures/`. Eine erste, rein dokumentbasierte Fassung dieses Punkts kam fälschlich zu "läuft nicht über OpenRouter" — per echtem API-Aufruf widerlegt und korrigiert; SDK-Frage per §5 des Provider-Protokolls entschieden (eigener HTTP-Client, kein Fremd-SDK für den Alpha-Endpunkt).
- [x] Review der alten „HARD RULE“-Texte: tatsächliche Enforcement-Stelle oder offene Policy markieren. Keine implizite Änderung von Pipeline-Defaults. Pre-Run bereits in `pi_scan_pre_rules_de.md` erledigt; Post-Run-Teil jetzt in [M0 HARD-RULE-Review](pi_jev_m0_hard_rule_review_de.md) — fand `docs/PI/attic/pi_run_chat_empfehlungs_chat_datenplan.md` (nie implementiert, reiner Prompt-Text) und den realen M42-Vorfall, der die bestehenden M6-Guards begründet; empfiehlt `min_resume_phase` als neues Feld für M6 (siehe dortige Checkliste).
- [x] Lokale Fixtures aus synthetischen Daten oder freigegebenen anonymisierten Scans anlegen; bestehende Run-Dateien nicht verändern. -> `web_backend_cpp/config/pi_decisions/fixtures/scan_metrics_synthetic.json`, rein synthetisch, Aggregatwerte gegenrechnet; plus die Provider-Fixtures aus dem vorigen Punkt.

**Abnahme:** Jedes verwendete Feld hat Quelle, Einheit, Fehlend-Verhalten und Testfixture; jede als hart deklarierte Bedingung besitzt eine geplante ausführbare Prüfung. Unbekannte Provider-Details blockieren Adapter-Freigabe, nicht State-/Policy-Arbeit. **M0 ist damit inhaltlich vollständig** (Katalog, Schemas, Provider-Vertrag, HARD-RULE-Review, Fixtures); es existiert noch kein Backend-/Frontend-Code. M1 (State-Builder) kann auf Basis der vorliegenden Schemas/Fixtures beginnen; M3 (Adapter) kann auf Basis des verifizierten Provider-Vertrags beginnen, sobald die offene SDK-Entscheidung (Provider-Protokoll §5) getroffen ist.

## 5. M1 — State-Builder und Gruppenstatistik

**Status:** umgesetzt und getestet (2026-09-23), noch nirgends in Routen/UI verdrahtet (das ist M4). **Abhängigkeit:** M0.

Umgesetzte API (`web_backend_cpp/include/services/pi/pi_decision_state.hpp`, `src/services/pi/pi_decision_state.cpp`): `build_pre_run_decision_state(PreRunDecisionInputs)` -> `PreRunDecisionResult{state, findings, state_hash, provider_projection}`. Rein funktional: kein I/O, keine Uhr, keine Modellaufrufe; keine `AppState`-Abhängigkeit. Zusätzlich exportiert: `canonical_json_dump`, `sha256_prefixed`, `axis_asymmetry`.

- [x] Gruppen nach kompatiblen Aufnahmebedingungen (Kamera, Filter, Belichtung, Gain) bilden; ein unbekannter (fehlender/leerer) Headerwert ist ein eigener Bucket und wird nie einem bekannten gleichgesetzt. Farbmodus/Bayer sind Session-Fakten aus dem Scan-Ergebnis (nicht pro Frame gemessen).
- [x] Messabdeckung pro Metrik/Gruppe: `valid_count`/`total_count` je Messwert plus `metric_coverage{missing, invalid}`; nicht lesbare Frames zählen als `frames_read_failed`, nie als Messwert 0. `fwhm`/`roundness` <= 0 (Scan-Sentinel -1) gelten als ungültig, wie in `cli_main.cpp`.
- [x] Vergleichbare Spreads deterministisch (`median`, `p10`, `p90`, `mad`, `relative_spread`) je Gruppe aus den Frame-Werten, nicht aus dem globalen Scan-Aggregat. Nullnenner -> `invalid`/`zero_denominator`; unter `policy.min_valid_for_spread` gültigen Frames -> `not_applicable`/`insufficient_sample`. **Der Default 5 ist ein Platzhalter, keine Freigabepolicy** (bleibt M5-Thema); er steht im State (`policy`) und damit im Hash.
- [x] Symmetrische Achsenabweichung `abs(ln(roundness))` pro Frame, danach aggregiert (gegenläufige Elongationen verdecken sich nicht: Test mit 0.5/2 -> Median der Rohrundheit 1.25, Median der Achsenabweichung ln 2); Roh-`roundness` bleibt unverändert im State.
- [x] Session-Schätzungen (`field_rotation_deg`, als `kind=estimate`) getrennt von Nutzerangaben (`user_stated`, nur mit `source=user`, sonst Finding `session_context_ignored`). Keine Montierung/Objektklasse abgeleitet.
- [x] State-/Dataset-/Config-/Lock-Hashes (kanonisches JSON: sortierte Keys, `3` == `3.0`, Frame-Reihenfolge und flüchtige Zeitstempel nicht semantisch, NaN/Infinity -> `invalid_argument` im Kanonisierer bzw. Finding `non_finite_input` im Builder) und Provider-Allowlist-Projektion (kein Kamera-/Zielname, keine Scan-ID/Pfade/Secrets, Filter und Nutzerfakten nur als kurze Klartext-Token, Capabilities nur Booleans; danach defensive Leak-Prüfung). Projizierte Config-Pfade sind bis M2 eine Konstante (`global_metrics.adaptive_weights`).
- [x] Bestehendes `pi.feature-vector.v1`/kNN unberührt: `pi_feature_vector.*` und `pi_param_model.*` nicht angefasst, `web_backend_cpp_pi_param_model` läuft unverändert grün.

Blockierende Befunde (Codes in `state.blocking_findings`): `color_mode_unresolved`, `bayer_pattern_missing` (OSC), `scan_errors_present`, `no_readable_frames`, `metrics_missing`, `base_config_invalid`. Nicht blockierend (`findings`): `mixed_acquisition_groups`, `coverage_partial`, `identity_strength_metadata`, `scan_id_unspecified`, `non_finite_input`, Scan-Warnungen, ignorierte/zurückgehaltene Eingaben.

Schema-Anpassungen gegenüber M0 (`pi.decision-state.v1.schema.json`): Gruppen erhielten `gain`, `frames_read_failed`, `metric_coverage`; `metrics` ist je Metrik eine `metricSummary`; `measurement` hat ein optionales `reason`; neuer Top-Level-Block `policy`; `session_facts` ist `{measured, user_stated}`.

Tests: `tests/test_pi_decision_state.cpp` (CMake-Target `web_backend_cpp_pi_decision_state`). Enthält einen kleinen Draft-07-Teilprüfer, der den Builder-Output gegen das committete Schema prüft, damit Code und Schema nicht auseinanderlaufen. Abgedeckt: vollständig (synthetisches Scan-Fixture), leer, einzelne Metrik fehlend, NaN/Infinity/falscher Typ, Nullnenner, gemischte Filter/Belichtungen inkl. unbekanntem Bucket, OSC/MONO/UNKNOWN/unbekannter String, Roundness 0.5/1/2, gegenläufige Elongationen, Hash-Stabilität und -Änderung (Sperren, Config, Metrik, Policy, Digest-Stärke), Projektion ohne Geheimnisse/Pfade.

**Abnahme:** Gleiche semantische Inputs ergeben denselben State-Hash (getestet: Key-/Frame-/Lock-Reihenfolge, 100 vs 100.0, Zeitstempel). Unzureichende Inputs erzeugen nur blockierende Befunde bzw. `missing`/`not_applicable`-Messwerte, keinen scheinbar gültigen Wert. Nicht Teil dieser Abnahme: Verdrahtung in Routen (M4), Kandidatenbildung (M2).

## 6. M2 — Pre-Rules, Kandidaten und atomare Validierung

**Status:** offen. **Abhängigkeit:** M1.

Geplante APIs:

- `build_pre_run_candidates(state, catalog, policy)` -> zulässige Kandidaten und Ausschlussgründe.
- `validate_decision_candidate(candidate, state, current_config)` -> validierter Gesamtpatch oder Ablehnung.
- `resolve_decision(answer, candidates, policy)` -> Vorschlag, Beibehalten oder Enthaltung.

Arbeit:

- [ ] Erste Kandidaten aus Abschnitt 3.2 implementieren. Bereits aktive adaptive Gewichtung ergibt keinen Änderungs-Patch.
- [ ] Experimentelle Zulässigkeit und reguläre Freigabe trennen; unbelegte numerische Confidence-Schwellen nicht als Defaults einbauen.
- [ ] Exakte Allowlist und geschützte Pfade vor Schema-Validierung prüfen.
- [ ] Vollständigen Kandidaten auf Ausgangsconfig anwenden und per bestehendem `validate-config` prüfen; keine Rettung einzelner Werte aus gescheitertem Kandidaten.
- [ ] Locks, Evidenz-Referenzen, `old_value`, aktuelle Versionen und Abhängigkeiten prüfen.
- [ ] Bestehenden Validator um expliziten atomaren Modus/Wrapper ergänzen; bestehende Aufrufer behalten ihren Vertrag.
- [ ] Rationale aus versionierten Textbausteinen und echten Messreferenzen erzeugen; keine LLM-Erklärung nötig, um einen Vorschlag darzustellen.

Tests neu: `test_pi_pre_rules.cpp`, `test_pi_decision_policy.cpp`; bestehend `test_ai_routes.cpp`, `test_pi_action_plan.cpp`. Ein Gate-Patch, ein ungültiger Wert oder eine Lock-Verletzung verwirft die ganze Gruppe. Reihenfolge der Regeln darf das Ergebnis nicht ändern.

**Abnahme:** Kein Modellresultat kann einen neuen Pfad/Wert einschleusen. `keep_current` und Enthaltung sind überall darstellbar. Aktuelle Config bleibt bei jeder Ablehnung byte-/semantikgleich gemäß Speichervertrag.

## 7. M3 — Jev-Adapter, Fehlerbehandlung und Betriebsmodus

**Status:** offen. **Abhängigkeit:** M0-Vertrag, für Integration M2.

- [ ] `decisionsService.ts` mit injizierbarem HTTP-Transport für Tests erstellen; Endpoint/Modell aus zentraler Konfiguration.
- [ ] `PI_DECISIONS_MODE=off|shadow|suggest` vorsehen, Default `off`; vorhandenen `JEV_OPENROUTER_API_KEY` aus bestehendem Secret-Mechanismus nutzen, eigener Allowlist-Eintrag getrennt vom generischen `openrouter`-Eintrag (Abschnitt 2.1). Keine neuen Top-Level-Pipelineparameter.
- [ ] `allow_experimental_suggestions` mit Default false ergänzen: nur in `suggest` wirksam, Kandidaten immer sichtbar experimentell markieren; nicht mit regulärer Qualitätsfreigabe verwechseln.
- [ ] Gepinnte Modellkennung verlangen für freigegebene Policy; Aliaswechsel invalidiert Kalibrierungsfreigabe.
- [ ] Request- und Response-Schemas prüfen, unbekannte Kandidaten/Fragen und nicht endliche oder unzulässige Wahrscheinlichkeiten ablehnen.
- [ ] Gesamtdeadline und Größenlimit konfigurieren; höchstens ein Retry bei vorübergehendem Fehler innerhalb derselben Deadline, keiner bei Schema-/Authfehlern; `Retry-After` nur innerhalb Budget.
- [ ] Abbruch, parallele Deduplizierung und begrenzte Parallelität implementieren. Keine Run-Slots durch unbeschränkte Requests blockieren.
- [ ] Im Sidecar Fehler in stabile Anwendungscodes übersetzen; bestehende Traffic-Logs dürfen für diesen Pfad keinen vollständigen State/API-Key ausgeben.
- [ ] `pi_decision_service` baut auf M1/M2 auf, persistiert Status und verwirft verspätete Antworten zu veralteten Inputs.

Tests neu: `agent_service/tests/decisionsService.test.ts`; Testskript mit vorhandenem TypeScript-/Node-Werkzeug ergänzen. Fake-Transport für Timeout, 429, 5xx, 401, ungültiges JSON, übergroße Antwort, falsche ID, NaN-artige Werte, Versionsabweichung und Abbruch.

**Abnahme:** `off` erzeugt null Requests; `shadow` erzeugt keine anwendbaren UI-Patches; Provider-Ausfall ändert keine Config. Mock-Vertrag und optionaler realer Provider-Smoke werden getrennt ausgewiesen.

## 8. M4 — Vollständiger Pre-Run-Workflow

**Status:** offen. **Abhängigkeit:** M1–M3.

- [ ] Asynchrone Beratungsroute und Statusabruf nach Abschnitt 3.5 anbinden; vorhandene Scan-Ergebnisse wiederverwenden.
- [ ] In `input-scan.js` bestehende Komponenten nutzen: Empfehlung anfordern, läuft, Vorschlag, keine Änderung, fehlende Evidenz, nicht verfügbar, veraltet.
- [ ] Eigenständige Jev-Karte unter Tools -> AI & API (`tools.js`/`ai-empfehlung.js`) und dritten Parameter-Sub-Tab „Jev-Empfehlungen“ (`parameter.js`) gemäß Abschnitt 2.1 umsetzen; Mode-/Key-/Modellstatus dort, Vorschlagsdarstellung/-übernahme weiterhin im Scan-Flow.
- [ ] Tabelle mit alter/neuer Einstellung, Evidenz, Risiko und experimentellem/freigegebenem Status anzeigen. Keine Formulierung „verbessert“, solange nur erwarteter Nutzen vorliegt.
- [ ] Ganze Kandidatengruppe auswählen; keine Checkboxen, die abhängige Einzelwerte auseinandernehmen.
- [ ] Übernahme nur in Config-Entwurf; kein impliziter Start. Lock-Auswahl und Revision serverseitig prüfen.
- [ ] Persistierten Status nach Reload, Navigation und History wiederherstellen. Veraltete Responses dürfen neuere Auswahl nicht überschreiben.
- [ ] DE/EN-Texte ergänzen; gemeinsame Styles und mobile Darstellung erhalten.
- [ ] Bestehende PI-Beratung als getrennt bezeichnete Alternative erhalten. Keine automatische zweite Provider-Anfrage als versteckter Fallback.

Tests: Backend-Route mit Fake-Sidecar; bestehendes Apply ohne Jev bleibt kompatibel. UI-Fixtures für alle Zustände bei Desktop/Mobil. Rennen zwischen Config-Edit und Response/Apply, Reload und doppelte Übernahme testen. Prüfen, dass kein Run-/Resume-Job entsteht.

**Abnahme:** Eine synthetische geeignete Eingabe kann einen validierten, als experimentell markierten Vorschlag bis in den Entwurf führen. Ein ungeeigneter Fall bleibt unverändert mit nachvollziehbarem Grund. Dies ist funktionale Abnahme, noch kein Nachweis besserer Bilder.

## 9. M5 — Evaluation und Freigabe pro Kandidat

**Status:** offen. **Abhängigkeit:** M4; Bildqualitätsfreigabe benötigt reale Vergleichsevidenz.

Arbeit:

- [ ] Neues Replay-Werkzeug `web_backend_cpp/scripts/evaluate_pi_decisions.py` und maschinenlesbaren Evaluationsbericht mit State-/Policy-/Modelldigests erstellen; Vorschlagsauswertung ohne Runner-Start ermöglichen.
- [ ] Datensatzregister mit Session-, Geräte-, Filter- und Qualitätsgruppen aufbauen. Kalibrierung und Test auf Sessionebene trennen; gleiche Frames nicht in beide Mengen aufnehmen.
- [ ] Vier Baselines protokollieren: aktuelle Config, Regeln allein, bestehende PI-Beratung, Regeln plus Jev. kNN optional separat ausweisen.
- [ ] Nutzerzustimmung, angewendete Config und Qualitätslabel in getrennten Feldern speichern. Abgebrochene/fehlgeschlagene Runs nicht als negative Bildqualität ohne Ursache etikettieren.
- [ ] Für Qualitätsnachweise gepaarte Ergebnisse derselben Inputs und vergleichbarer Ausgabezustände verwenden; zusätzliche Runs ausschließlich nach ausdrücklichem Auftrag.
- [ ] Messplan vor Evaluation einfrieren: Sternform an gematchten Positionen, Rauschen auf gültigen vergleichbaren Flächen, Signalerhaltung, Abdeckung und bestehende Gate-Entscheidungen.
- [ ] Numerische Nichtunterlegenheits-/Verbesserungsgrenzen, Stichprobenanforderungen und Auswahl-/Enthaltungsschwellen pro Kandidat in einer versionierten Freigabepolicy festlegen. Diese Werte sind aktuell offen; ohne sie keine reguläre Freigabe.
- [ ] Rate ungültiger Vorschläge, Enthaltungsrate, Coverage, Kalibrierung, Qualitätsdeltas und Unsicherheitsintervalle getrennt berichten. Agreement ist nur Diagnose.

**Abnahme:** Null verbotene/ungültige anwendbare Patches in der festgelegten Testsuite; keine Verletzung bestehender Qualitätsgates. Reguläre Freigabe nur bei erfüllter vorab definierter Qualitäts- und Coverage-Policy auf getrennten Testdaten. Fehlender Mehrwert hält Kandidaten im Shadow-/explizit experimentellen Modus.

Ein erster Kandidat kann häufig `keep_current` liefern. Das rechtfertigt keine Absenkung der Freigabegrenzen; zusätzliche sinnvolle Kandidaten werden nach dem Erweiterungsverfahren des Regelkatalogs einzeln eingeführt.

## 10. M6 — Post-Run-Beratung und PI-Übergabe

**Status:** offen. **Abhängigkeit:** M4, M5-Verfahren für jede neue Empfehlung.

Zunächst die tatsächlichen Producer/Consumer von Qualitätsartefakten inventarisieren. Ein Reader für `pi_run_quality.json` beweist weder vollständige Erzeugung noch Verfügbarkeit der benötigten Zwischenstandsmetriken. Vorhandenen Completion-Analyse-Pfad nutzen; keine parallele konkurrierende Ergebniswahrheit.

- [ ] `pi.post-run-decision-state.v1` definieren: Run-/Config-/Artefaktidentität, ausgeführte Phasen, Raw/Uniform/Multiband/selected, Gate-Ergebnisse und gültige Zwischenstände.
- [ ] Fehlende Messungen explizit abbilden; bei Bedarf separaten offline/deterministischen Metrik-Producer planen. Keine Modellcalls im Runner.
- [ ] `pi_post_run_advisor` mit vier Ergebnissen `no_change`, `diagnose`, `suggest_downstream`, `suggest_reconstruction` implementieren.
- [ ] Erster diagnostischer Fall: validiert gemessener Unterschied zwischen benannten Vor-/Nachbearbeitungsständen. Ohne lokalisierbare Ursache nur Diagnosehinweis.
- [ ] Pipelineparameter und sichtbare PI-Bildoperationen getrennt halten. Ein PI-Edit darf nicht ungeprüft ein Trainingslabel für Run-Config werden.
- [ ] Resume-Vorschlag anhand vorhandener Config-Scope-, Cache-, Masken- und Provenienzprüfung erstellen. Kein frei vom Modell erzeugter Phasenname. Jede vorgeschlagene Action trägt `min_resume_phase` (serverseitig aus der Pfad-Phase-Zuordnung berechnet, nie vom Modell) — Konzept aus [M0 HARD-RULE-Review](pi_jev_m0_hard_rule_review_de.md) übernommen, dort erstmals (unimplementiert) skizziert.
- [ ] Bisheriges Ergebnis, Patch-ID und Ablehnungshistorie erhalten; kein identischer erneut verworfener Patch ohne neue Evidenz.
- [ ] Im Run-Monitor Befund, Unsicherheit und früheste geprüfte Phase anzeigen. Übernahme bereitet Entwurf vor; Start bleibt separate Aktion.
- [ ] Optional strukturierte Diagnose an vorhandenen PI-Kontext übergeben; PI entscheidet im bestehenden Nachbearbeitungsworkflow mit dem Nutzer.

Tests: fehlendes `canvas_mask.fits`, fehlende Vorgängercaches, falsche Provenienz, nur finale gestreckte Metrik, ungültige Sternpopulation, abgelehnter Multiband-Kandidat, unbekannte PCC-Ursache, falsche Resume-Phase, keine Änderung und wiederholter Vorschlag. Bestehende Run-Artefakte bleiben bei Beratung unverändert.

**Abnahme:** Ohne benötigte Evidenz keine scheinpräzise Reparatur. Kein Jev-Score überschreibt Gates. Resume-Machbarkeit ist separat geprüft; kein automatischer Runstart.

## 11. M7 — Integration, Dokumentation und Regression

**Status:** offen. **Abhängigkeit:** jeweiliger freizugebender Produktabschnitt.

- [ ] Neue Backend-Quellen, Testtargets und auszuliefernde Katalogdateien in CMake/Packaging aufnehmen; aus installiertem Arbeitsverzeichnis testen, nicht nur aus Repository-CWD.
- [ ] Sidecar-Konfiguration, Modi, Fehlercodes, Datenübermittlung und Rückschaltung auf `off` dokumentieren.
- [ ] DE/EN-Nutzertexte, Reporttexte bei Änderungen und API-Verträge konsistent halten.
- [ ] Bei tatsächlicher Änderung eines Pipelineparameters die lokale Skill-Anweisung `.devin/skills/update-param-doc/SKILL.md` anwenden: Structs, Parser, Schemas, Beispiele und DE/EN-Referenzen gemeinsam aktualisieren. Dieser Plan selbst ändert keine Pipelineparameter.
- [ ] Bekannte Grenzen dokumentieren: begrenzte Kandidatenabdeckung, Messproxies, fehlende Evidenz und separate empirische Freigabe.
- [ ] Release-Nachweis enthält Codeversion, Katalog-/Policy-/Modellversion und genaue Test-/Evaluationsresultate.

Prüfreihenfolge während der Umsetzung:

1. Neue State-/Policy-/Adapter-Tests gezielt ausführen.
2. Backend-Testtargets und Backend ausführbar bauen; betroffene Route-/Action-/Resume-Verträge testen.
3. `agent_service`: TypeScript-Build und neue Tests; keine unbeabsichtigten Dependency-Upgrades.
4. Frontend statisch und an Fixtures bzw. vorhandenem Dienst bei Desktop/Mobil prüfen; kein npm-Build für v3 erfinden.
5. Falls `tile_compile_cpp/apps/` oder Messcode geändert wird: Runner und relevante C++-Tests bauen, danach vollständige Suite soweit praktisch; CUDA nur bei tatsächlicher CUDA-Änderung.
6. JSON/YAML, Dokumentationslinks und `git diff --check` prüfen.

Alle Terminalausgaben nach `/tmp/out_<zweck>.txt` umleiten und separat lesen. Tests dürfen isolierte Backend-Fixtures verwenden; keine persistenten Backend-/Sidecar-Prozesse ohne Auftrag starten. Keine Bildverarbeitungsruns als impliziten Test beginnen. Vor Abschluss müssen gestartete Test-/Build-Sessions beendet sein.

## 12. Abnahmematrix und offene fachliche Entscheidungen

| Gate | Nachweis | Blockiert |
|---|---|---|
| G0 Vertragsklarheit | Quellen, Einheiten, Status, Schemas und Provider-Fixtures | Adapterintegration |
| G1 Zulässigkeit | Negative State-/Lock-/Gate-/Atomizitätstests | Jede anwendbare Empfehlung |
| G2 Funktionaler Pre-Run-Pfad | Scan -> Vorschlag -> Entwurf; kein Runstart; Reload/Race-Tests | Experimentelle UI-Freigabe |
| G3 Empirischer Nutzen | Vorab definierte Policy und getrennte gepaarte Evaluation | Reguläre Kandidatenfreigabe |
| G4 Post-Run-Evidenz | Zwischenstände, Gate-Herkunft, gültige Metriken | Konkreter Post-Run-Patch |
| G5 Resume-Vertrag | Vorgänger, Cache, Maske, Scope, Provenienz | Resume-Empfehlung |
| G6 Auslieferung | Build, Tests, Ressourcen, DE/EN-Dokumentation | Release |

Offen und ausdrücklich vor der jeweiligen Freigabe zu lösen:

- Exakter Provider-Vertrag und verfügbare stabile Modellkennung (M0/M3).
- Mindestmessabdeckung und experimentelle Zulässigkeit für adaptive Gewichtung (M0/M2).
- Empirische Nutzen-/Nichtunterlegenheitsgrenzen und Mindeststichprobe pro Kandidat (M5).
- Verfügbarkeit und Producer aller erforderlichen Post-Run-Zwischenstandsmetriken (M6).

Diese offenen Fachwerte sind keine Erlaubnis zu geraten. Technische Implementierung, funktionale Tests, Provider-Smoke und wissenschaftliche Qualitätsfreigabe werden jeweils separat als bestanden, offen oder fehlgeschlagen ausgewiesen.
