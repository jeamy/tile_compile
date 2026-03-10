# Vergleich Crow/C++ Backend vs. FastAPI-Backend (`master`)

Stand: 2026-03-10

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

Offen bleiben damit vor allem die noch nicht voll portierten Bereiche rund um reichere UI-Events, vollstaendige Download-Retry/Resume-Semantik auf FastAPI-Niveau und die letzten Vertragsdetails einzelner Tool-Jobs.

## Scope

Verglichen wurden:

- neues Backend: `web_backend_cpp`
- Referenz: FastAPI-Implementierung aus Branch `master` unter `web_backend`
- Frontend-Nutzung: `web_frontend`

Zusätzlich wurde das C++-Backend erfolgreich gebaut:

- `cmake -S web_backend_cpp -B /tmp/tile_compile_web_backend_build`
- `cmake --build /tmp/tile_compile_web_backend_build -j2`

## Kurzfazit

Auf Route-Ebene ist das Crow/C++-Backend nahezu vollstaendig: die im FastAPI-Backend vorhandenen API- und WebSocket-Endpunkte sind im C++-Backend ebenfalls angelegt.

Auf Vertrags- und Verhaltens-Ebene ist die Portierung aber noch nicht vollstaendig. Es gibt mehrere relevante Abweichungen, durch die das Frontend nur teilweise korrekt abgedeckt ist:

- erledigt: Multi-Input-Scan wurde auf FastAPI-aehnliche Aggregation umgestellt.
- erledigt: Run-Monitor/Phasenmodell wurde auf den Frontend-Vertrag angeglichen.
- teilweise offen: Tool-Jobs (Astrometry/PCC/Downloads) sind deutlich naeher am Vertrag, aber Retry/Resume/UI-Event-Details fehlen noch.
- teilweise offen: Cancellation ist fuer Subprozesse jetzt echt implementiert; Download-/Custom-Jobs sind noch nicht voll generisch an `jobs/{id}/cancel` angekoppelt.
- teilweise offen: Artifact-Pfadsicherheit wurde korrigiert; die globale Pfadpolicy ist weiterhin einfacher als im FastAPI-Backend.

Fazit: "Route vorhanden" ist weitgehend erreicht, "FastAPI-Verhalten und Frontend-Vertrag vollstaendig ersetzt" noch nicht.

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

### 1.2 Nicht vom Frontend genutzte oder derzeit sekundäre Routen

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

Rest-Risiko:

- keine eigene C++-Contract-Testabdeckung vorhanden

Bewertung: erledigt, fachlich deutlich angenaehert.

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

Status: TEILWEISE ERLEDIGT

Der kritische Teil dieses Punkts ist inzwischen umgesetzt:

- `SubprocessManager::cancel()` verfolgt jetzt echte PIDs und beendet laufende Prozesse
- `runs/stop` liefert nicht mehr pauschal Erfolg und kann `killed_pids` befuellen

Relevante Stellen:

- `web_backend_cpp/src/subprocess_manager.cpp`
- `web_backend_cpp/src/routes/runs_routes.cpp`

Weiter offen:

- Download-/Custom-Threads (Astrometry/PCC Downloads) hoeren nicht auf `POST /api/jobs/{job_id}/cancel`, sondern nur auf separate globale Cancel-Flags je Toolgruppe.

Bewertung: deutlich verbessert, aber noch nicht voll FastAPI-aequivalent.

### 2.4 Pfadpruefung ist deutlich schwaecher als in FastAPI

Status: TEILWEISE ERLEDIGT

Das C++-Backend nutzt fuer Allowlisting im Kern nur einen Prefix-Stringvergleich:

- `web_backend_cpp/src/backend_runtime.cpp`

Das ist gegenueber FastAPI (`ensure_path_allowed`, `resolve_input_path`) deutlich schwaecher.

Besonders problematisch war:

- `/api/runs/{run_id}/artifacts/view`
- `/api/runs/{run_id}/artifacts/raw/...`

Dieser Artifact-spezifische Sicherheitsmangel wurde inzwischen behoben. Die C++-Routen erzwingen jetzt, dass der Zielpfad innerhalb des jeweiligen Run-Verzeichnisses bleibt:

- `web_backend_cpp/src/routes/runs_routes.cpp`

Die FastAPI-Version schuetzt genau diesen Fall explizit:

- `master:web_backend/app/api/runs.py`

Weiter offen:

- die globale Allowlist-/Pfadpolicy des C++-Backends ist insgesamt immer noch einfacher als die FastAPI-Variante

Bewertung: Artifact-Pfade erledigt, globale Pfadpolicy teilweise offen.

### 2.5 Tool-Job-Vertrag ist nicht auf FastAPI-Niveau

Status: TEILWEISE ERLEDIGT

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

Im C++-Backend wurde hier inzwischen ein relevanter Teil nachgezogen:

- laufende `progress`-Updates werden jetzt im Job-`data` mitgefuehrt
- Download-Jobs fuehren jetzt u. a. `stage`, `current_chunk`, `pending_chunks`, `completed_chunks`
- generische Subprozess-Jobs behalten initiale Vertragsdaten auch nach Abschluss

Relevante Stellen:

- `web_backend_cpp/src/services/download_manager.cpp`
- `web_backend_cpp/src/routes/tools_routes.cpp`
- `web_backend_cpp/src/subprocess_manager.cpp`

Weiter offen:

- `bytes_received`
- `bytes_total`
- `attempt`
- `status_code`
- `existing_bytes`
- echte HTTP-Resume-/Retry-Semantik auf FastAPI-Niveau

Bewertung: verbessert, aber noch offen.

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

Weiter offen ist vor allem die vollstaendige Gleichheit des Ergebnisobjekts:

- `web_backend_cpp/src/routes/tools_routes.cpp`

Bewertung: Parameterseite stark verbessert, Ergebnisvertrag noch nicht voll abgeschlossen.

### 2.8 App-Constants/UI-Events weichen vom Vertrag ab

Status: TEILWEISE ERLEDIGT

Erledigt:

- FastAPI: `/api/app/constants` liefert `phases`
- FastAPI: `color_modes = ["OSC","MONO","RGB"]`

Siehe:

- `master:web_backend/app/api/app_state.py`
- `web_backend_cpp/src/routes/app_state_routes.cpp`

Weiter offen bei UI-Events:

- FastAPI nutzt reichere Eventstrukturen und Eventnamen wie `scan.start`, `run.start`, `run.resume`
- C++-Events enthalten nur `{seq,event,data}` statt der reicheren Top-Level-Felder

Siehe:

- `web_backend_cpp/src/ui_event_store.cpp`
- `master:web_backend/tests/test_backend_contract.py`

Bewertung: `app/constants` erledigt, UI-Events noch offen.

### 2.9 Config-Endpunkte sind funktional einfacher als FastAPI

Abweichungen im C++-Backend:

- `/api/config/current` liest Datei direkt; kein `load-config`-CLI-Pfad, keine `fallback=file_read`-Semantik
- nicht vorhandene Datei fuehrt effektiv zu leerem YAML mit `200`, statt sauberem Fehlerbild
- `/api/config/validate` akzeptiert praktisch nur `yaml`, nicht gleichwertig `path` oder `config`
- `warnings` aus CLI werden nicht durchgereicht
- `/api/config/schema` liefert Rohtext statt geparstes Schemaobjekt

Siehe:

- `web_backend_cpp/src/routes/config_routes.cpp`
- `master:web_backend/app/api/config.py`

Bewertung: mittel.

## 3. Frontend-Abdeckung durch das neue Backend

## 3.1 Route-seitig

Ja, fast alle vom Frontend verwendeten API-Pfade existieren im C++-Backend.

## 3.2 Funktions-seitig

Nein, nicht vollstaendig.

### Dashboard / Wizard / Input & Scan

Teilweise abgedeckt:

- Guardrails, App-State, Config, Run-Start, Presets, FS-Browser sind vorhanden.

Nicht vollstaendig:

- MONO-Queue/Run-Queue ist vorhanden, aber Cancellation/Status ist noch nicht voll auf Referenzniveau.

### Run Monitor

Nur teilweise abgedeckt:

- Run-Status, Logs, Artifacts, Stats, Resume-Routen existieren.

Wesentliche Restluecke:

- UI-Event-/Stream-Details sind noch nicht voll auf FastAPI-Niveau.

### History + Tools

Teilweise abgedeckt:

- Run-Liste, Set-Current, Delete, Stats vorhanden.

Risiken:

- Job-/Run-Cancel ist verbessert, aber fuer alle Custom-/Download-Jobs noch nicht voll einheitlich

### Astrometry

Nur teilweise abgedeckt:

- Detect, Install, Catalog-Download, Solve, Save-Solved existieren.

Aber:

- Download-Retry/Resume-Vertrag ist noch nicht vollstaendig

### PCC

Nur teilweise abgedeckt:

- Status, Download, Online-Check, Run, Save-Corrected existieren.

Aber:

- Ergebnisvertrag fuer Frontend unvollstaendig

### Live Log

Teilweise abgedeckt:

- Logs und Run-WebSocket existieren.

Aber:

- wegen Event-Normalisierung und UI-Event-Vertrag noch nicht auf Referenzniveau

## 4. Tote/ungenutzte Teile, Demo-Daten, technische Altlasten

### 4.1 Offensichtlich ungenutzter Frontend-Code

`web_frontend/src/main.js` ist im aktuellen HTML nicht eingebunden.

Bewertung:

- wahrscheinlich Alt-/Prototyp-Code
- kann bei Wartung und Analyse stoeren

### 4.2 Demo-/Placeholder-Daten in HTML-Seiten

Mehrere Seiten enthalten noch statische Demo- oder Placeholder-Inhalte:

- vorkonfigurierte Preset-Optionen in HTML
- vorbelegte Queue-Zeilen fuer `L/R/G/B/Ha`
- statische Default-Statuszeilen
- Beispiel-Revision in `run-monitor.html`
- Beispielpfade wie `/data/calib/darks`

Beispiele:

- `web_frontend/index.html`
- `web_frontend/dashboard.html`
- `web_frontend/input-scan.html`
- `web_frontend/wizard.html`
- `web_frontend/parameter-studio.html`
- `web_frontend/run-monitor.html`

Das ist nicht zwingend falsch, wirkt aber teils wie Demo-/Mock-Zustand statt rein datengetriebener UI.

### 4.3 C++-Backend ohne eigene Test-Suite

Unter `web_backend_cpp` gibt es derzeit keine eigene Test-Suite.

Im FastAPI-Backend existieren dagegen Contract-/Tool-/Queue-Tests:

- `master:web_backend/tests/test_backend_contract.py`
- `master:web_backend/tests/test_tools_api_jobs.py`
- `master:web_backend/tests/test_runs_queue.py`

Bewertung:

- grosse funktionale Unterschiede bleiben derzeit unautomatisiert
- besonders riskant bei API-Vertragsportierung

### 4.4 Kleine technische Auffaelligkeiten im C++-Code

- erledigt: `download_manager.cpp` nutzt `on_progress` jetzt
- erledigt: `runs/stop` kann `killed_pids` jetzt befuellen
- mehrere Fehler-Envelopes sind inkonsistent detailarm im Vergleich zum FastAPI-Backend

## 5. Priorisierte To-do-Liste

### Prio 1

- erledigt: Run-Inspector/`PHASE_ORDER` auf FastAPI-/Frontend-Modell umstellen
- erledigt: echte Prozessbeendigung fuer `cancel`, `jobs/{id}/cancel`, `runs/{id}/stop` fuer Subprozess-Jobs
- teilweise erledigt: sichere Pfadauflösung fuer Artifact-View/Raw und globale Allowlist-Checks
- erledigt: Multi-Input-Scan wie FastAPI aggregierend implementieren

### Prio 2

- erledigt: Astrometry-Solve-Ergebnisvertrag portieren (`wcs_path`, RA/DEC, Scale, Rotation, FOV)
- teilweise erledigt: PCC-Run-Optionen und Ergebnisvertrag vollstaendig portieren
- teilweise erledigt: Tool-Download-Contract inkl. Progress/Resume/Retry sauber implementieren

### Prio 3

- teilweise erledigt: `app/constants`, `ui-events`, Fehler-Envelopes auf FastAPI-Vertrag angleichen
- Config-Endpunkte fachlich an FastAPI annähern
- ungenutzten Frontend-Code (`src/main.js`) entfernen oder klar markieren
- statische Demo-/Placeholder-Inhalte in HTML reduzieren

## 6. Gesamtbewertung

Das Crow/C++-Backend ist als struktureller Ersatz fuer das FastAPI-Backend schon weit fortgeschritten:

- gleiche Endpunktlandschaft
- erfolgreicher Build
- zentrale Features grundsaetzlich vorhanden

Aber:

- die vertragstreue Portierung ist noch nicht abgeschlossen
- mehrere Frontend-Kernflows sind inzwischen funktional deutlich naeher an der FastAPI-Referenz
- vor allem UI-Events, Download-Retry/Resume-Details, Config-Vertrag und Teile des PCC-Ergebnisvertrags sind noch nicht auf Referenzniveau

Empfehlung:

Das C++-Backend ist funktional deutlich naeher an einem echten Ersatz fuer die FastAPI-Referenz als zu Beginn der Analyse. Vor einem Switch sollten aber die noch offenen Restpunkte aus Prio 2/3 und vor allem eine neue C++-Contract-Test-Suite folgen.
