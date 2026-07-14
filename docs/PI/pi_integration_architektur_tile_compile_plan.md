# PI-Integration Architekturplan fuer Tile Compile

Status: rekonstruiert und neu geschrieben am 2026-07-14  
Zieloberflaeche: GUI3 (`web_frontend_v3`). Die alte GUI (`web_frontend`) ist nicht Ziel dieser Integration.

## Ausgangspunkt

Tile Compile hat PI bereits als Parameter-Optimierung im Scan-AI-/Empfehlungsfluss. Die neue Integration erweitert das zu einer kontrollierten PI-Schicht, die Kontext versteht, Empfehlungen in Action-Plans ueberfuehrt, Aenderungen vorab validiert, eine sichere Vorschau liefert, explizit angewendet wird und aus geprueften Optimierungen ueber Sessions hinweg lernen kann.

Die Architektur ist an die Xyona-Idee angelehnt, aber fuer Tile Compile optimiert:

- Tile Compile arbeitet config- und run-zentriert, daher ist `validate-config` die zentrale Sicherheitsgrenze.
- PI-Antworten werden nicht direkt geschrieben, sondern als `pi.action-plan.v1` modelliert.
- Memories lernen nicht automatisch "Wahrheiten", sondern speichern reviewbare Optimierungs-Erfahrungen.
- GUI3 bleibt der einzige neue Bedienpfad.

## Zielbild

PI wird in Tile Compile zu einer Assistenz- und Orchestrierungsschicht:

- Kontext lesen: Runtime, aktuelle Config, Scan-Ergebnisse, Jobs, Artefakte, Reports.
- Vorschlagen: Scan-AI-Empfehlungen und spaeter weitere PI-Tools erzeugen strukturierte Action-Plans.
- Pruefen: Action-Plans werden serverseitig formal und config-semantisch validiert.
- Vorschau: GUI3 zeigt mutationfreie YAML-Diffs.
- Anwenden: Schreibende Aktionen brauchen explizite Bestaetigung und erzeugen Config-Revisionen.
- Lernen: Erfolgreich angewendete Optimierungen koennen als Memory-Kandidaten gespeichert und reviewed werden.

## Sicherheitsregeln

- Kein Blind-Write: Jede Config-Aenderung laeuft ueber Action-Plan, Preview und Validierung.
- Kein ungeprueftes Lernen: Memories starten als `candidate`; erst Review macht sie fuer spaetere Sessions belastbar.
- Keine alte GUI: Neue PI-Funktionalitaet wird nur in `web_frontend_v3` eingebaut.
- Privacy by default: Memories speichern Metadaten, Config-Pfade, Gruende und Validierung, aber keine Bilddaten.
- Bestehende Tile-Compile-Kommandos bleiben die Autoritaet fuer Config-Gueltigkeit.

## Rekonstruierter Implementierungsstand

- [x] Plan-Datei in `docs/PI/pi_integration_architektur_tile_compile_plan.md` rekonstruiert.
- [x] Backend-PI-Routen rekonstruiert und in `main.cpp` registriert.
- [x] PI-Service-Dateien fuer Action-Plans, Validator, Context, Tools, Assistant und Memories rekonstruiert.
- [x] CMake-Ziele und Tests fuer PI-Komponenten rekonstruiert.
- [x] Scan-AI-Routen um PI Action-Plan-Anreicherung erweitert.
- [x] Scan-AI Apply um `learn=true` Memory-Kandidaten erweitert.
- [x] GUI3-Endpunkte und AI-Empfehlungsseite um PI Preview, PI Apply und Memories erweitert.
- [x] Alte GUI von versehentlichen PI-Aenderungen bereinigt.
- [x] Agent-Service Traffic-Logging zentralisiert und mit Redaction versehen.

Verifikation:

- [x] `node --check web_frontend_v3/js/pages/ai-empfehlung.js`
- [x] `node --check web_frontend_v3/js/api/endpoints.js`
- [x] `npm run build` in `agent_service`
- [x] Backend Build fuer `tile_compile_web_backend`, `fake_tile_compile_cli`, PI-Tests und AI-Routen-Test
- [x] `ctest --output-on-failure -R 'web_backend_cpp_(pi_routes|pi_memory_store|pi_action_plan|ai_routes)'`

## Phase 0 - Fundament

Ziel: Gemeinsame Sprache und Sicherheitsmodell fuer PI schaffen.

- [x] Action-Plan Schema `pi.action-plan.v1` definieren.
- [x] `config.set` als erste sichere Action implementieren.
- [x] Action-Plan Shape-Validator implementieren.
- [x] Scan-AI `validated_updates` in Action-Plans umwandeln.
- [x] Backend-Tests fuer Action-Plan-Erzeugung und Validierung anlegen.
- [x] PI-Traffic-Log aus Agent-Service duplizierten Stellen herausziehen.
- [x] Secrets, Bearer Tokens, API Keys und lokale Projektpfade im Traffic-Log redacten.

Abnahmekriterien:

- [x] Action-Plan ohne Schema oder ohne gueltige Actions wird abgelehnt.
- [x] Scan-AI-Antworten enthalten `action_plan` und `action_plan_validation`.
- [x] Logging-Fehler brechen Analyse nicht ab.

## Phase 1 - PI Kontext, Tools und Assistant

Ziel: PI kann Tile-Compile-Kontext kontrolliert lesen, ohne Schreibrechte.

- [x] `/api/pi/context` implementieren.
- [x] `/api/pi/tools` implementieren.
- [x] `/api/pi/tools/call` implementieren.
- [x] `/api/pi/assistant/ask` implementieren.
- [x] Tool `context.overview` bereitstellen.
- [x] Tool `config.schema` bereitstellen.
- [x] Tools fuer Artefakt-/Report-Zusammenfassung bereitstellen.
- [x] Vorschau-Tools fuer BGE/HMS als planende Read-/Preview-Tools bereitstellen.

Abnahmekriterien:

- [x] Kontextantwort enthaelt Runtime-, State-, Job- und Scan-Informationen.
- [x] Unbekannte Tools werden abgelehnt.
- [x] Assistant antwortet ohne schreibende Nebenwirkung.
- [x] PI-Routen-Test deckt Kontext, Tools und Assistant ab.

## Phase 2 - Action-Plan Preview und kontrolliertes Anwenden

Ziel: PI-Vorschlaege koennen sicher vorab gesehen und erst dann angewendet werden.

- [x] `/api/pi/action-plans/validate` implementieren.
- [x] `/api/pi/action-plans/preview` implementieren.
- [x] Preview setzt Config-Aenderungen mutationfrei auf eine Basiskonfig.
- [x] Preview prueft Ergebnis mit `validate-config --stdin`.
- [x] `/api/pi/action-plans/apply` implementieren.
- [x] Apply erfordert `confirmed=true`.
- [x] Apply kann `expected_patched_yaml` gegen Preview-Ergebnis pruefen.
- [x] Apply speichert Config-Revision und UI-Event.
- [x] GUI3 AI-Seite erzeugt Action-Plans aus ausgewaehlten Empfehlungen.
- [x] GUI3 zeigt PI Preview als YAML-Diff.
- [x] GUI3 erlaubt explizites PI-Anwenden.

Abnahmekriterien:

- [x] Unbestaetigtes Apply wird abgelehnt.
- [x] Preview erzeugt YAML-Diff ohne Config-Datei zu schreiben.
- [x] Apply erzeugt Revision.
- [x] Tests decken Preview, Apply und invaliden Config-Fall ab.

## Phase 3 - Memories und Lernen ueber Sessions

Ziel: Tile Compile merkt sich nuetzliche Optimierungsentscheidungen reviewbar und sessionuebergreifend.

- [x] PI Memory Store als JSONL-Dateien unter `runs/.pi_memory` implementieren.
- [x] Memory Schema `pi.memory.v1` verwenden.
- [x] Kandidaten mit `append_candidate` speichern.
- [x] `/api/pi/memories` fuer Liste und Statusfilter implementieren.
- [x] `/api/pi/memories/:id/review` fuer `accepted`, `rejected`, `deprecated` implementieren.
- [x] `/api/pi/memories/retrieve` fuer einfache pfad-/typbasierte Suche implementieren.
- [x] Scan-AI Apply kann bei `learn=true` ein `config_optimization` Memory erzeugen.
- [x] GUI3 AI-Seite zeigt Memory-Liste und Review-Aktionen.

Abnahmekriterien:

- [x] Memory Store Test deckt Append, List, Review und Retrieve ab.
- [x] PI-Routen-Test deckt Memory-List, Review und Retrieve ab.
- [x] AI-Routen-Test prueft Memory-Kandidat nach `learn=true`.

Noch offen in Phase 3:

- [x] Akzeptierte Memories beim naechsten Scan-AI-Request automatisch als Kontext beilegen.
- [x] GUI3 deutlicher anzeigen, welche Memories nur Kandidaten sind und welche accepted/deprecated sind.
- [x] Memory-Deduplikation einbauen, damit identische Config-Optimierungen nicht mehrfach wachsen.

## Phase 4 - Memory Retrieval im Optimierungsfluss

Ziel: PI nutzt akzeptierte Erfahrungen, ohne sie unkritisch zu kopieren.

- [x] Beim Aufbau des Scan-AI Request relevante Memories anhand von Config-Pfaden abrufen.
- [x] Nur `accepted` Memories als starken Kontext verwenden; `candidate` wird nicht in den Request-Kontext uebernommen.
- [x] Memory-Kontext im Prompt klar als "historische Erfahrung" markieren.
- [x] Ablehnungsstatus (`rejected`, `deprecated`) als Negativsignal nutzbar machen.
- [x] Memory-Retrieval mit Tests gegen Fehluebernahme absichern.

Abnahmekriterien:

- [x] Ein neuer Scan-AI-Request enthaelt passende accepted Memories.
- [x] Rejected Memories werden nicht als Empfehlungskontext genutzt.
- [x] Tests pruefen, dass Memories keine Schema-/Config-Validierung umgehen.

## Phase 5 - Outcome-Metriken und Qualitaetsfeedback

Ziel: Lernen wird besser als blosses "wurde angewendet".

- [ ] Nach Runs relevante Outcome-Metriken erfassen: Validierung, Artefakte, Warnungen, Report-Status, ggf. Qualitaetsmetriken.
- [x] Memory-Kandidaten um Outcome-Felder erweitern.
- [x] GUI3 Review zeigt angewendete Pfade, Gruende, Validierung und Outcomes.
- [x] Accepted-Memories nach positiver Outcome-Evidenz hoeher gewichten.
- [x] Deprecated-Memories fuer verschlechterte oder ueberholte Optimierungen unterstuetzen.

## Phase 6 - Erweiterte PI-Werkzeuge

Ziel: PI kann weitere Tile-Compile-Arbeitsablaeufe planen, aber weiterhin kontrolliert.

- [x] BGE-Plan-Tool mit echten Run-/Config-Daten vertiefen.
- [x] HMS-/Mosaik-Plan-Tool mit echten Projektparametern vertiefen.
- [x] Resume-/Run-Planung als read-only Plan erzeugen.
- [x] Schreibende Tools erst nach Action-Plan/Preview/Apply freischalten.
- [x] Tool-Registry versionieren und dokumentieren.

## Phase 7 - Audit, Export und Betrieb

Ziel: PI-Aktionen bleiben nachvollziehbar und wartbar.

- [x] GUI3 Audit-Ansicht fuer Action-Plans, Applies, Revisionen und Memory-Reviews.
- [x] Export/Import fuer PI Memories mit Privacy-Filter.
- [x] CLI-/Backend-Werkzeug zum Aufraeumen und Deduplizieren von Memories.
- [x] Regressionstests fuer typische Optimierungsfaelle: OSC/MONO, BGE, HMS, AQMH, PCC.
- [x] Nutzer-Dokumentation fuer Workflow: Empfehlung, Preview, Apply, Learn, Review.

## Memory-Konzept im Detail

Ein Memory ist eine reviewbare Erfahrung, keine automatische Regel.

Typischer Ablauf:

1. Scan-AI erzeugt Empfehlungen.
2. GUI3 zeigt Empfehlungen und PI Preview.
3. Nutzer wendet validierte Aenderungen an.
4. Wenn `learn=true` gesetzt ist, speichert Tile Compile einen Memory-Kandidaten.
5. Nutzer reviewed den Kandidaten als `accepted`, `rejected` oder `deprecated`.
6. Spaetere Sessions duerfen akzeptierte Memories als Kontext verwenden, muessen aber weiterhin Schema und Config validieren.

Beispiel fuer `config_optimization`:

- `type`: `config_optimization`
- `source`: `scan_ai_apply`
- `status`: `candidate`
- `privacy_class`: `metadata_only`
- `analysis_id`
- `config_path_name`
- `config_updates`
- `summary`
- `confidence`
- `detected_scenarios`
- `warnings`
- `validation`

## Naechster sinnvoller Schritt

Als naechstes sollte Phase 4 begonnen werden: akzeptierte Memories beim Aufbau neuer Scan-AI-Requests abrufen und als explizit gekennzeichneten historischen Kontext an den Agent-Service uebergeben. Das ist der Punkt, an dem "Lernen ueber Sessions" praktisch wirksam wird.
