# Vergleich Crow/C++ Backend vs. FastAPI-Backend (`master`)

Stand: 2026-03-10

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

- Multi-Input-Scan ist nicht auf FastAPI-Niveau umgesetzt.
- Run-Monitor/Phasenmodell ist inkompatibel zum aktuellen Frontend.
- Tool-Jobs (Astrometry/PCC/Downloads) liefern nicht den erwarteten Live- und Ergebnisvertrag.
- Cancellation ist fuer Subprozesse und Download-Threads funktional unvollstaendig.
- Es gibt sicherheitsrelevante Unterschiede bei Pfadpruefungen.

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

FastAPI aggregiert mehrere Eingabeverzeichnisse aktiv selbst und erzeugt:

- `input_dirs`
- `per_dir_results`
- aufsummierte `frames_detected`
- gemeinsames `color_mode`/`color_mode_candidates`

Im C++-Backend wird bei `/api/scan` dagegen einfach ein einzelner CLI-Aufruf mit mehreren Pfaden aufgebaut:

- `web_backend_cpp/src/routes/scan_routes.cpp`

Das ist nicht aequivalent zur FastAPI-Logik aus:

- `master:web_backend/app/api/scan.py`

Risiko:

- Dashboard/Wizard/Input-Scan mit mehreren Input-Ordnern kann fachlich falsche oder unvollstaendige Ergebnisse liefern.
- Felder wie `per_dir_results` werden vom C++-Code erwartet, aber nicht aktiv erzeugt.

Bewertung: kritisch.

### 2.2 Run-Monitor-Phasenmodell ist nicht kompatibel zum Frontend

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

Das C++-Backend nutzt dagegen ein altes/anderes Phasenmodell:

- `scan`
- `local_metrics`
- `metrics`
- `registration`
- `stacking`
- `clustering`
- `bge`
- `pcc`
- `validation`

Siehe:

- `web_backend_cpp/include/services/run_inspector.hpp`

Folgen:

- viele Phase-Buttons im Run Monitor werden nie korrekt aktualisiert
- Resume-Freigaben und Fortschrittsdarstellung sind nur teilweise konsistent
- `app/constants` liefert zudem `phase_order` statt `phases`

Bewertung: kritisch fuer `run-monitor.html` und `live-log.html`.

### 2.3 Prozessabbruch/Cancellation ist im C++-Backend nicht sauber implementiert

`SubprocessManager::cancel()` markiert Jobs als `cancelled`, beendet aber den gestarteten OS-Prozess nicht sichtbar:

- `web_backend_cpp/src/subprocess_manager.cpp`

Auch `runs/stop` arbeitet ueber dieses Modell und liefert immer ein leeres `killed_pids`-Array:

- `web_backend_cpp/src/routes/runs_routes.cpp`

Zusatzproblem:

- Download-/Custom-Threads (Astrometry/PCC Downloads) hoeren nicht auf `POST /api/jobs/{job_id}/cancel`, sondern nur auf separate globale Cancel-Flags je Toolgruppe.

Folgen:

- UI zeigt "cancelled", waehrend Prozess/Download real weiterlaufen kann
- generisches Job-Cancel ist fuer Download-Jobs fachlich unzuverlaessig

Bewertung: kritisch.

### 2.4 Pfadpruefung ist deutlich schwaecher als in FastAPI

Das C++-Backend nutzt fuer Allowlisting im Kern nur einen Prefix-Stringvergleich:

- `web_backend_cpp/src/backend_runtime.cpp`

Das ist gegenueber FastAPI (`ensure_path_allowed`, `resolve_input_path`) deutlich schwaecher.

Besonders problematisch:

- `/api/runs/{run_id}/artifacts/view`
- `/api/runs/{run_id}/artifacts/raw/...`

Dort wird im C++-Backend kein robuster Check erzwungen, dass der Pfad innerhalb des Run-Verzeichnisses bleibt:

- `web_backend_cpp/src/routes/runs_routes.cpp`

Die FastAPI-Version schuetzt genau diesen Fall explizit:

- `master:web_backend/app/api/runs.py`

Bewertung: kritisch, sicherheitsrelevant.

### 2.5 Tool-Job-Vertrag ist nicht auf FastAPI-Niveau

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

Im C++-Backend fehlt ein Grossteil davon oder wird nie aktualisiert.

Besonders deutlich:

- `web_backend_cpp/src/services/download_manager.cpp`

Der Parameter `on_progress` wird zwar uebergeben, im Download-Code aber nie aufgerufen. Retry-/Resume-Semantik aus FastAPI existiert praktisch nicht; die `/retry`-Routen setzen nur Flags im Request, ohne echte HTTP-Range-/Resume-Implementierung.

Bewertung: kritisch fuer Tool-UX und Vertragsgleichheit.

### 2.6 Astrometry-Solve liefert nicht den erwarteten Ergebnisvertrag

FastAPI liefert nach `astrometry/solve` im Job-Datensatz u. a.:

- `wcs_path`
- geparste WCS-Zusammenfassung (`ra_deg`, `dec_deg`, `pixel_scale_arcsec`, `rotation_deg`, `fov_*`)

Siehe:

- `master:web_backend/app/api/tools.py`

Das Frontend nutzt genau diese Felder:

- `web_frontend/src/app.js`

Das C++-Backend startet nur einen Subprozess und liefert am Ende nur generisches `stdout/stderr/exit_code`:

- `web_backend_cpp/src/routes/tools_routes.cpp`
- `web_backend_cpp/src/subprocess_manager.cpp`

Es fehlen:

- `wcs_path`
- geparstes Solve-Ergebnis
- Nutzung von `search_radius_deg`

Folgen:

- Astrometry-Seite kann Solve als "ok" sehen, aber Felder/Resultate nicht korrekt fuellen
- `save-solved` bekommt u. U. keinen belastbaren `wcs_path`

Bewertung: kritisch fuer `astrometry.html`.

### 2.7 PCC-Run ist nur teilweise portiert

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

Das C++-Backend unterstuetzt nur einen Teil der Optionen:

- `mag_limit`
- `min_stars`
- `sigma_clip`

und liefert kein gleichwertiges Ergebnisobjekt:

- `web_backend_cpp/src/routes/tools_routes.cpp`

Folgen:

- PCC-Seite kann Jobs starten, aber Resultatdarstellung und Folgeaktionen sind nur teilweise belastbar
- `save-corrected` bekommt oft keine sinnvollen `output_channels`

Bewertung: kritisch fuer `pcc.html`.

### 2.8 App-Constants/UI-Events weichen vom Vertrag ab

Abweichungen:

- FastAPI: `/api/app/constants` liefert `phases`
- C++: liefert `phase_order`
- FastAPI: `color_modes = ["OSC","MONO","RGB"]`
- C++: `color_modes = ["OSC","MONO","NARROW"]`

Siehe:

- `master:web_backend/app/api/app_state.py`
- `web_backend_cpp/src/routes/app_state_routes.cpp`

Bei UI-Events:

- FastAPI nutzt reichere Eventstrukturen und Eventnamen wie `scan.start`, `run.start`, `run.resume`
- C++ nutzt vereinfachte Namen wie `scan_started`, `run_started`, `run_resumed`
- C++-Events enthalten nur `{seq,event,data}` statt der reicheren Top-Level-Felder

Siehe:

- `web_backend_cpp/src/ui_event_store.cpp`
- `master:web_backend/tests/test_backend_contract.py`

Bewertung: mittel bis hoch. Das aktuelle Frontend nutzt UI-Events nur begrenzt, aber der Referenzvertrag ist nicht erreicht.

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

- Multi-Input-Scan ist nicht auf FastAPI-Niveau portiert.
- MONO-Queue/Run-Queue ist vorhanden, aber Cancellation/Status ist nicht robust wie im Referenzbackend.

### Run Monitor

Nur teilweise abgedeckt:

- Run-Status, Logs, Artifacts, Stats, Resume-Routen existieren.

Wesentliche Luecke:

- Phasenmodell passt nicht zum Frontend.

Damit ist eine Kernfunktion des Run Monitors aktuell nicht vertragstreu abgedeckt.

### History + Tools

Teilweise abgedeckt:

- Run-Liste, Set-Current, Delete, Stats vorhanden.

Risiken:

- Artifact-View/Raw sind sicherheitlich schwach
- Job-/Run-Cancel nicht robust

### Astrometry

Nur teilweise abgedeckt:

- Detect, Install, Catalog-Download, Solve, Save-Solved existieren.

Aber:

- Solve-Ergebnisvertrag fehlt
- Download-Live-Status fehlt weitgehend

### PCC

Nur teilweise abgedeckt:

- Status, Download, Online-Check, Run, Save-Corrected existieren.

Aber:

- Parameterumfang kleiner als FastAPI
- Ergebnisvertrag fuer Frontend unvollstaendig

### Live Log

Teilweise abgedeckt:

- Logs und Run-WebSocket existieren.

Aber:

- wegen Phasenmodell und Event-Normalisierung nicht auf Referenzniveau

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

- `download_manager.cpp`: `on_progress` wird nicht genutzt
- `runs/stop`: `killed_pids` wird zurueckgegeben, aber nie real befuellt
- mehrere Fehler-Envelopes sind inkonsistent detailarm im Vergleich zum FastAPI-Backend

## 5. Priorisierte To-do-Liste

### Prio 1

- Run-Inspector/`PHASE_ORDER` auf FastAPI-/Frontend-Modell umstellen
- echte Prozessbeendigung fuer `cancel`, `jobs/{id}/cancel`, `runs/{id}/stop`
- sichere Pfadauflösung fuer Artifact-View/Raw und globale Allowlist-Checks
- Multi-Input-Scan wie FastAPI aggregierend implementieren

### Prio 2

- Astrometry-Solve-Ergebnisvertrag portieren (`wcs_path`, RA/DEC, Scale, Rotation, FOV)
- PCC-Run-Optionen und Ergebnisvertrag vollstaendig portieren
- Tool-Download-Contract inkl. Progress/Resume/Retry sauber implementieren

### Prio 3

- `app/constants`, `ui-events`, Fehler-Envelopes auf FastAPI-Vertrag angleichen
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
- mehrere Frontend-Kernflows sind nur teilweise korrekt abgedeckt
- vor allem Run Monitor, Multi-Input-Scan, Tool-Jobs und Cancellation sind noch nicht auf Referenzniveau

Empfehlung:

Das C++-Backend sollte noch nicht als vollwertiger Ersatz fuer die FastAPI-Referenz betrachtet werden. Vor einem Switch sollten mindestens die Punkte aus Prio 1 und Prio 2 erledigt und durch eine neue C++-Contract-Test-Suite abgesichert werden.
