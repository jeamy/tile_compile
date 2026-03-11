# Vergleich Crow/C++ Backend vs. FastAPI-Backend (`master`)

Stand: 2026-03-11

## Update nach Umsetzungsrunden

Seit der Erstfassung dieses Reports wurden im C++-Backend bereits mehrere Prio-1/Prio-2-Punkte umgesetzt:

- Multi-Input-Scan wird jetzt wie im FastAPI-Backend aggregiert und liefert `input_dirs`, `per_dir_results`, summierte `frames_detected` sowie gemeinsames `color_mode`/`color_mode_candidates`.
- `app/constants` und der Run-Monitor wurden auf das FastAPI-Phasenmodell umgestellt (`phases`, `resume_from`, `OSC/MONO/RGB`).
- Artifact-View/Raw sind jetzt auf Dateien innerhalb des jeweiligen Run-Verzeichnisses begrenzt.
- `astrometry/solve` liefert jetzt `wcs_path` und geparste Solve-Ergebnisse (`ra_deg`, `dec_deg`, Pixel-Scale, Rotation, FOV).
- Subprozess-Jobs haben jetzt echte PID-Verfolgung und echte Prozessbeendigung bei `cancel`.
- Tool-Download-Jobs liefern jetzt laufende Fortschrittsdaten auch im `data`-Payload; PCC-Downloads fuehren `current_chunk`, `pending_chunks`, `completed_chunks`, `stage` und `progress`.
- `pcc/run` reicht jetzt den erweiterten FastAPI-Parameterumfang an die CLI weiter (`mag_bright_limit`, Aperture/Annulus, `chroma_strength`, `k_max`, `apply_attenuation`).
- `runs/{run_id}/stop` ist jetzt naeher am FastAPI-Verhalten und liefert `ok=false`, wenn nichts getroffen wurde, sowie belegte `killed_pids`.

Mit der aktuellen Umsetzungsrunde wurden auch die zuvor offenen Prio-Punkte geschlossen:

- `jobs/{job_id}/cancel` wirkt jetzt generisch fuer Subprozess-, Download- und sonstige Custom-Jobs.
- Download-Jobs liefern Retry-/Resume-Metadaten inkl. `attempt`, `status_code`, `resumed`, `existing_bytes`, `bytes_*` und nutzen HTTP-Resume via Range/`CURLOPT_RESUME_FROM_LARGE`.
- UI-Events liegen jetzt im reicheren FastAPI-Schema vor, werden als JSONL persistiert und beim Start fuer Replay wieder eingelesen.
- Config-Routen nutzen jetzt den CLI-Pfad (`load-config`, `save-config`, `validate-config`) und die Pfadauflosung wurde auf robuste kanonische Unterpfadpruefung mit relativer Aufloesung angeglichen.
- Unter `web_backend_cpp/tests` existiert jetzt eine eigene Contract-/Integrationstest-Suite, die per `ctest` laeuft und Kernvertraege fuer Config, Runs, Tools/Jobs und UI-Serving absichert.
- Der PCC-Ergebnisvertrag wird jetzt nicht nur unter `job.data.result`, sondern fuer relevante Felder auch auf Top-Level im Job-`data` gespiegelt.
- Einfache Fehlerpfade liefern jetzt ebenfalls strukturierte Error-Envelopes mit `code`, `message` und `details`.

Offen bleiben damit nur noch kleinere UI-/Content-Feinschliffe ausserhalb des eigentlichen Backend-Vertrags.

## Scope

Verglichen wurden:

- neues Backend: `web_backend_cpp`
- Referenz: FastAPI-Implementierung aus Branch `master` unter `web_backend`
- Frontend-Nutzung: `web_frontend`

Zusaetzlich wurde das C++-Backend erfolgreich gebaut und getestet:

- `cmake --build /tmp/tile_compile_web_backend_build -j2`
- `ctest --output-on-failure` in `/tmp/tile_compile_web_backend_build` lief erfolgreich (`4/4` Tests)

## Kurzfazit

Auf Route-Ebene ist das Crow/C++-Backend nahezu vollstaendig: die im FastAPI-Backend vorhandenen API- und WebSocket-Endpunkte sind im C++-Backend ebenfalls angelegt.

Auf Vertrags- und Verhaltens-Ebene ist die Portierung inzwischen weitgehend auf FastAPI-Niveau:

- erledigt: Multi-Input-Scan, Run-Monitor-Phasenmodell und Artifact-Pfadsicherheit
- erledigt: generische Job-Cancellation inkl. Download-/Custom-Jobs
- erledigt: Download-Retry/Resume-Vertrag inkl. Resume-Metadaten und HTTP-Resume
- erledigt: UI-Events inkl. Eventnamen, Top-Level-Feldern und JSONL-Replay
- erledigt: Config-Routen ueber CLI-Vertrag sowie robuste Pfadauflosung
- erledigt: automatisierte C++-Contract-/Integrationstests sind vorhanden

Fazit: "FastAPI-Verhalten und Frontend-Vertrag vollstaendig ersetzt" ist fuer den derzeit sichtbaren Implementierungsstand backendseitig erreicht. Verbleibende Restpunkte liegen aktuell eher im UI-Feinschliff als im API-/Backend-Vertrag.

## 1. Route-Abdeckung

### 1.1 API-/WS-Routen

Die folgenden funktionalen Gruppen sind im C++-Backend vorhanden:

- System: `health`, `version`, `fs/*`
- App-State: `app/state`, `app/constants`, `app/ui-events`
- Config: `current`, `validate`, `save`, `patch`, `presets`, `presets/apply`, `revisions`, `restore`, `schema`
- Scan: `scan`, `scan/latest`, `scan/quality`, `guardrails`
- Runs: `runs`, `start`, `status`, `stop`, `resume`, `logs`, `artifacts`, `artifact view/raw`, `delete`, `set-current`, `stats`, `stats/status`, `config revision restore`
- Tools: Astrometry- und PCC-Endpunkte inkl. Retry-/Cancel-Routen
- WebSockets: `ws/runs/{id}`, `ws/jobs/{id}`, `ws/system`

Bewertung:

- Route-Matrix gegen FastAPI: weitgehend vollstaendig
- kein grober "Endpoint fehlt komplett"-Befund

### 1.2 Nicht vom Frontend genutzte oder derzeit sekundaere Routen

Im aktuellen `web_frontend` werden einige vorhandene Backend-Routen nicht oder kaum genutzt:

- `/api/health`
- `/api/version`
- `/api/config/schema`
- `/api/ws/jobs/{id}`
- `/api/ws/system`

Das ist nicht falsch, aber potentiell ungenutzte Oberflaeche.

## 2. Kritische Abweichungen zur FastAPI-Referenz

### 2.1 Multi-Input-Scan ist funktional nicht gleichwertig

Status: ERLEDIGT

FastAPI aggregiert mehrere Eingabeverzeichnisse aktiv selbst und erzeugt:

- `input_dirs`
- `per_dir_results`
- aufsummierte `frames_detected`
- gemeinsames `color_mode`/`color_mode_candidates`

Dieser Punkt ist inzwischen im C++-Backend umgesetzt. `/api/scan` aggregiert mehrere Eingabeverzeichnisse jetzt selbst und erzeugt ebenfalls:

- `input_dirs`
- `per_dir_results`
- summierte `frames_detected`
- gemeinsames `color_mode`/`color_mode_candidates`

Referenz:

- `web_backend_cpp/src/routes/scan_routes.cpp`
- `master:web_backend/app/api/scan.py`

Bewertung: erledigt.

### 2.2 Run-Monitor-Phasenmodell ist nicht kompatibel zum Frontend

Status: ERLEDIGT

Das Frontend zeigt und aktualisiert Phasen wie:

- `SCAN_INPUT`
- `CHANNEL_SPLIT`
- `NORMALIZATION`
- `GLOBAL_METRICS`
- `TILE_GRID`
- `REGISTRATION`
- `...`
- `ASTROMETRY`
- `BGE`
- `PCC`

Siehe:

- `web_frontend/run-monitor.html`
- `web_frontend/src/app.js`

Die FastAPI-Referenz nutzt exakt dieses Uppercase-Phasenmodell:

- `master:web_backend/app/services/run_inspector.py`

Dieser Punkt wurde inzwischen umgestellt. Das C++-Backend liefert jetzt das FastAPI-/Frontend-Modell ueber:

- `web_backend_cpp/include/services/run_inspector.hpp`
- `web_backend_cpp/src/routes/app_state_routes.cpp`

Erledigt wurden dabei auch:

- `app/constants` liefert `phases`
- `resume_from` ist auf die FastAPI-Resume-Phasen angepasst
- `color_modes` ist auf `OSC/MONO/RGB` angeglichen

Bewertung: erledigt.

### 2.3 Prozessabbruch/Cancellation ist im C++-Backend nicht sauber implementiert

Status: ERLEDIGT

Inzwischen umgesetzt:

- `SubprocessManager::cancel()` verfolgt jetzt echte PIDs und beendet laufende Prozesse
- `POST /api/jobs/{job_id}/cancel` markiert Jobs generisch im `job_store` als `cancelled` und wirkt dadurch auch fuer Download-/Custom-Threads
- Download-/Custom-Jobs pollen `job_store.is_cancelled(job_id)` und brechen generisch ab
- `runs/{run_id}/stop` liefert nicht mehr pauschal Erfolg, kann `killed_pids` befuellen und besitzt jetzt einen Orphan-Fallback (`run.stop.orphan`)

Relevante Stellen:

- `web_backend_cpp/src/subprocess_manager.cpp`
- `web_backend_cpp/src/routes/jobs_routes.cpp`
- `web_backend_cpp/src/routes/runs_routes.cpp`
- `web_backend_cpp/src/routes/tools_routes.cpp`

Bewertung: fachlich auf FastAPI-Niveau angeglichen.

### 2.4 Pfadpruefung ist deutlich schwaecher als in FastAPI

Status: ERLEDIGT

Die fruehere String-Prefix-Logik wurde ersetzt. Das C++-Backend nutzt jetzt:

- `web_backend_cpp/src/backend_runtime.cpp`

- kanonische Unterpfadpruefung statt Prefix-Stringvergleich
- Standard-Allowlist inkl. Projektwurzel, `runs`, `$HOME`, `/tmp`, `/media`
- `resolve_input_path()` fuer relative Pfade und `input_search_roots`
- sichere Aufloesung auch fuer Tool- und Config-Pfade

Zusätzlich bleiben die Artifact-Routen weiterhin explizit auf das jeweilige Run-Verzeichnis begrenzt:

- `web_backend_cpp/src/routes/runs_routes.cpp`

Bewertung: die vormals offene Pfadpolicy ist funktional angeglichen.

### 2.5 Tool-Job-Vertrag ist nicht auf FastAPI-Niveau

Status: ERLEDIGT

Im FastAPI-Backend und im Contract-Dokument fuer Tool-Jobs sind u. a. vorgesehen:

- `stage`
- `bytes_received`
- `bytes_total`
- `progress`
- `retry_count`
- `resume_enabled`
- `attempt`
- `status_code`
- `resumed`
- `existing_bytes`
- job-spezifische Felder fuer Downloads/Solve/PCC

Siehe:

- `master:web_backend/fe_contract_tools_jobs.md`
- `master:web_backend/app/api/tools.py`

Im C++-Backend ist dieser Vertrag jetzt ebenfalls umgesetzt:

- laufende `progress`-Updates werden jetzt im Job-`data` mitgefuehrt
- Download-Jobs fuehren jetzt `bytes_received`, `bytes_total`, `attempt`, `status_code`, `resumed`, `existing_bytes`, `retry_count`, `resume_enabled`
- PCC-Downloads fuehren zusaetzlich `current_chunk`, `pending_chunks`, `completed_chunks`, `stage`
- generische Subprozess-Jobs behalten initiale Vertragsdaten auch nach Abschluss
- Retry-Backoff und HTTP-Resume sind im `download_manager` implementiert; Teil-Downloads bleiben fuer spaeteres Resume erhalten

Relevante Stellen:

- `web_backend_cpp/src/services/download_manager.cpp`
- `web_backend_cpp/src/routes/tools_routes.cpp`
- `web_backend_cpp/src/subprocess_manager.cpp`

Bewertung: fuer den sichtbaren Job-Vertrag an FastAPI angeglichen.

### 2.6 Astrometry-Solve liefert nicht den erwarteten Ergebnisvertrag

Status: ERLEDIGT

FastAPI liefert nach `astrometry/solve` im Job-Datensatz u. a.:

- `wcs_path`
- geparste WCS-Zusammenfassung (`ra_deg`, `dec_deg`, `pixel_scale_arcsec`, `rotation_deg`, `fov_*`)

Siehe:

- `master:web_backend/app/api/tools.py`

Das Frontend nutzt genau diese Felder:

- `web_frontend/src/app.js`

Dieser Punkt ist inzwischen umgesetzt. Das C++-Backend liefert jetzt:

- `web_backend_cpp/src/routes/tools_routes.cpp`
- `web_backend_cpp/src/subprocess_manager.cpp`

- `wcs_path`
- geparstes Solve-Ergebnis (`ra_deg`, `dec_deg`, `pixel_scale_arcsec`, `rotation_deg`, `fov_*`)
- Nutzung von `search_radius_deg`

Bewertung: erledigt.

### 2.7 PCC-Run ist nur teilweise portiert

Status: TEILWEISE ERLEDIGT

FastAPI unterstuetzt mehr Parameter und liefert Ergebnisdaten fuer das Frontend:

- `mag_bright_limit`
- `aperture_radius_px`
- `annulus_inner_px`
- `annulus_outer_px`
- `chroma_strength`
- `k_max`
- `apply_attenuation`
- Ergebnisdaten wie Sterne/Matrix/Output-Pfade

Das Frontend sendet mehrere dieser Felder und erwartet Ergebnisdaten:

- `web_frontend/src/app.js`

Das C++-Backend unterstuetzt inzwischen nicht mehr nur den Minimalumfang. Nachgezogen wurden:

- `mag_limit`
- `min_stars`
- `sigma_clip`
- `mag_bright_limit`
- `aperture_radius_px`
- `annulus_inner_px`
- `annulus_outer_px`
- `chroma_strength`
- `k_max`
- `apply_attenuation`

Fuer die aktuell vom Frontend gelesenen Kernfelder ist der Stand deutlich besser als in der Erstfassung:

- der CLI-Output von `tile_compile_cli pcc-run` enthaelt `stars_matched`, `stars_used`, `residual_rms` und `matrix`
- `SubprocessManager` uebernimmt JSON-stdout in `job.data.result`, so dass diese Felder im Frontend ankommen koennen

Inzwischen ebenfalls erledigt:

- `web_backend_cpp/src/routes/tools_routes.cpp`
- `web_backend_cpp/src/subprocess_manager.cpp`
- `tile_compile_cpp/apps/cli_main.cpp`

Bewertung: Parameterseite weitgehend portiert; Ergebnisvertrag ist fuer die aktuelle UI direkt nutzbar und jetzt auch per C++-Contract-Test abgesichert.

### 2.8 App-Constants/UI-Events weichen vom Vertrag ab

Status: ERLEDIGT

Erledigt:

- FastAPI: `/api/app/constants` liefert `phases`
- FastAPI: `color_modes = ["OSC","MONO","RGB"]`

Siehe:

- `master:web_backend/app/api/app_state.py`
- `web_backend_cpp/src/routes/app_state_routes.cpp`

Erledigt bei UI-Events:

- reichere Eventstruktur mit `seq`, `ts`, `event`, `source`, `run_id`, `job_id`, `payload`
- angeglichene Eventnamen wie `scan.start`, `run.start`, `run.stop`, `run.resume`, `config.save`, `config.patch.save`
- JSONL-Persistenz unter `web_backend_cpp/runtime/ui_events.jsonl`
- Replay beim Start durch Einlesen des JSONL-Logs in den In-Memory-Store

Siehe:

- `web_backend_cpp/src/ui_event_store.cpp`
- `web_backend_cpp/src/routes/runs_routes.cpp`
- `web_backend_cpp/src/routes/config_routes.cpp`
- `web_backend_cpp/src/routes/tools_routes.cpp`

Bewertung: `app/constants` und `ui-events` sind vertraglich angeglichen.

### 2.9 Config-Endpunkte sind funktional einfacher als FastAPI

Status: ERLEDIGT

Die frueheren Abweichungen wurden in `web_backend_cpp/src/routes/config_routes.cpp` geschlossen:

- `/api/config/schema` liefert das geparste Schemaobjekt ueber den CLI-Pfad
- `/api/config/current` nutzt `load-config` und faellt nur noch explizit mit `fallback=file_read` zurueck
- `/api/config/validate` akzeptiert `path`, `yaml` und `config` und reicht `warnings` durch
- `/api/config/save` und `/api/config/patch` nutzen `save-config --stdin`
- `/api/config/presets/apply` nutzt `load-config`
- Revisions enthalten jetzt `revision_id`, `path`, `source`, `created_at`, `run_id`
- relative Config-Pfade werden robust aufgeloest und gegen die globale Pfadpolicy geprueft

Siehe:

- `web_backend_cpp/src/routes/config_routes.cpp`

Bewertung: der Config-Vertrag ist funktional auf FastAPI-Niveau angeglichen.

## 3. Frontend-Abdeckung durch das neue Backend

## 3.1 Route-seitig

Ja, fast alle vom Frontend verwendeten API-Pfade existieren im C++-Backend.

## 3.2 Funktions-seitig

Weitgehend ja.

### Dashboard / Wizard / Input & Scan

Funktional abgedeckt:

- Guardrails, App-State, Config, Run-Start, Presets und FS-Browser sind vorhanden.
- Multi-Input-Scan inkl. Aggregation ist auf den Frontend-Vertrag angeglichen.
- Queue-/Scan-Verhalten ist durch die aktuelle C++-Testsuite grundlegend mit abgesichert.

### Run Monitor

Funktional abgedeckt:

- Run-Status, Logs, Artifacts, Stats und Resume-Routen existieren.
- Phasenmodell und `resume_from` passen zum Frontend.
- UI-Event-/Stream-Details sind auf den FastAPI-Vertrag normalisiert.
- `app/ui-events` liefert Replay-faehige Eventdaten mit `latest_seq`.

### History + Tools

Funktional abgedeckt:

- Run-Liste, Set-Current, Delete und Stats sind vorhanden.
- Tool-Seiten sind durch die vorhandenen Contract-/Integrationstests deutlich besser gegen Vertragsregressionen abgesichert.

### Astrometry

Funktional abgedeckt:

- Detect, Install, Catalog-Download, Solve und Save-Solved existieren.
- `solve` liefert die vom Frontend benoetigten WCS-/Summary-Felder.
- Download-Retry/Resume-Vertrag ist backend-seitig vorhanden.

### PCC

Funktional abgedeckt:

- Status, Download, Online-Check, Run und Save-Corrected existieren.
- `pcc/run` akzeptiert den erweiterten Parametersatz.
- die aktuell von der UI gelesenen Ergebnisfelder werden ueber CLI-JSON durchgereicht und sind per C++-Contract-Test mit abgesichert.

### Live Log

Funktional abgedeckt:

- Logs und Run-WebSocket existieren.
- Event-/Log-Pfade sind durch die aktuelle Testsuite und den normalisierten UI-Event-Store deutlich robuster abgesichert.

## 4. Tote/ungenutzte Teile, Demo-Daten, technische Altlasten

### 4.1 Offensichtlich ungenutzter Frontend-Code

Status: ERLEDIGT

`web_frontend/src/main.js` existiert nicht mehr.

### 4.2 Demo-/Placeholder-Daten in HTML-Seiten

Status: WEITGEHEND BEREINIGT

Die groben Demo-/Mock-Inhalte wurden aus den HTML-Seiten entfernt:

- vorbelegte Queue-Zeilen fuer `L/R/G/B/Ha` wurden auf einzelne neutrale Leerzeilen reduziert
- statische Beispiel-Dateinamen in Astrometry/PCC wurden entfernt
- statische Demo-Resultate in Astrometry/PCC wurden entfernt
- konkrete Beispielpfade wie `/data/calib/darks` wurden durch generische Platzhalter ersetzt

Verbleibend sind vor allem normale Formular-Defaults und Startwerte, z. B.:

- neutrale leere Ergebnisfelder
- fachliche Default-Parameter in `parameter-studio.html`, `pcc.html` und `assumptions.html`
- initiale Status-/Leerwerte wie `-` in History- und Monitor-Feldern

Beispiele:

- `web_frontend/index.html`
- `web_frontend/dashboard.html`
- `web_frontend/input-scan.html`
- `web_frontend/wizard.html`
- `web_frontend/parameter-studio.html`
- `web_frontend/run-monitor.html`
- `web_frontend/astrometry.html`
- `web_frontend/pcc.html`
- `web_frontend/assumptions.html`

Das ist funktional unkritisch; der verbliebene Rest ist eher normales UI-Bootstrapping als echter Demo-/Mock-Zustand.

### 4.3 C++-Backend ohne eigene Test-Suite

Status: ERLEDIGT

Unter `web_backend_cpp/tests` gibt es jetzt eine eigene Test-Suite mit Contract-/Tool-/Run-Abdeckung. Verifiziert wurde sie in `/tmp/tile_compile_web_backend_build` mit `ctest` (`4/4` Tests erfolgreich).

Referenzrahmen aus dem FastAPI-Backend:

- `master:web_backend/tests/test_backend_contract.py`
- `master:web_backend/tests/test_tools_api_jobs.py`
- `master:web_backend/tests/test_runs_queue.py`

Bewertung:

- zentrale Vertraege werden jetzt automatisiert geprueft
- API-Portierungsregressionen sind damit deutlich besser abgesichert

### 4.4 Kleine technische Auffaelligkeiten im C++-Code

- erledigt: `download_manager.cpp` nutzt `on_progress` jetzt
- erledigt: `runs/stop` kann `killed_pids` jetzt befuellen
- erledigt: `download_manager.cpp` kennt jetzt HTTP-Resume sowie Attempt-/Status-Metadatenmodell
- erledigt: `ui_event_store.cpp` persistiert JSONL und liefert das reichere Eventschema
- erledigt: `backend_runtime.cpp` prueft Pfade ueber robuste kanonische Nachfahr-Pruefung
- einfache Fehlerpfade liefern jetzt ebenfalls strukturierte Error-Envelopes

## 5. Priorisierte To-do-Liste

### Prio 1

- erledigt: Run-Inspector/`PHASE_ORDER` auf FastAPI-/Frontend-Modell umstellen
- erledigt: echte Prozessbeendigung fuer `cancel`, `jobs/{id}/cancel`, `runs/{id}/stop` fuer Subprozess-Jobs
- erledigt: sichere Pfadaufloesung fuer Artifact-View/Raw und globale Allowlist-Checks
- erledigt: generische Cancel-Semantik fuer Custom-/Download-Jobs ueber `jobs/{id}/cancel`
- erledigt: Multi-Input-Scan wie FastAPI aggregierend implementieren

### Prio 2

- erledigt: Astrometry-Solve-Ergebnisvertrag portieren (`wcs_path`, RA/DEC, Scale, Rotation, FOV)
- erledigt: PCC-Run-Optionen portieren; Ergebnisvertrag auf Top-Level spiegeln und per Test absichern
- erledigt: Tool-Download-Contract inkl. `bytes_*`, `attempt`, `status_code`, `existing_bytes`, HTTP-Resume/Retry sauber implementieren
- erledigt: `ui-events` inkl. Eventnamen, Top-Level-Feldern und Persistenz auf FastAPI-Vertrag angleichen

### Prio 3

- erledigt: `app/constants` und Config-Endpunkte angeglichen; einfache Fehler-Envelopes sind strukturiert
- weitgehend erledigt: statische Demo-/Placeholder-Inhalte in HTML reduziert; verblieben sind nur normale Formular-Defaults und neutrale Leerzustande

## 6. Gesamtbewertung

Das Crow/C++-Backend ist als funktionaler Ersatz fuer das FastAPI-Backend im frontend-relevanten Umfang erreicht:

- gleiche Endpunktlandschaft
- erfolgreicher Build
- erfolgreiche C++-Contract-/Integrationstests
- zentrale Frontend-Kernflows auf Referenzniveau portiert

Empfehlung:

Das C++-Backend kann fuer die bisher verglichene Frontend-Funktionalitaet als 1:1-Ersatz der FastAPI-Referenz bewertet werden. Restarbeit liegt aktuell eher in kleinen UI-/Content-Feinschliffen und optionaler weiterer Testvertiefung, nicht mehr in sichtbaren Vertragsluecken des Backends.
