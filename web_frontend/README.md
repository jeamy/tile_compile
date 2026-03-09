# GUI2 Web Frontend

Produktives HTML-Frontend fuer GUI2. Das Frontend wird vom FastAPI-Backend unter `/ui/` ausgeliefert und steuert die C++-Werkzeuge ausschliesslich ueber HTTP und WebSocket.

## Rolle im System

- `web_frontend/`: produktive Oberflaeche
- `web_backend/`: API- und Prozess-Adapter
- `tile_compile_cpp/`: Runner und CLI fuer Scan, Runs, Resume, Astrometry, PCC und Reports

Startseite ist das Dashboard (`/ui/` -> `index.html`).

## Start

Entwicklungsstart aus dem Repository-Root:

```bash
./start_backend.sh
```

Dann im Browser:

```text
http://127.0.0.1:8080/ui/
```

Release-Bundles starten ueber die jeweiligen GUI2-Starter und liefern dieses Frontend ebenfalls unter `/ui/` aus.

## Produktive Screens

- `index.html` / `dashboard.html`
- `input-scan.html`
- `wizard.html`
- `parameter-studio.html`
- `assumptions.html`
- `run-monitor.html`
- `history-tools.html`
- `astrometry.html`
- `pcc.html`
- `live-log.html`

## Hinweise

- Die HTML-Dateien sind keine Standalone-App; ohne Backend funktionieren API-Aufrufe, Scan, Run-Monitor, Resume, Astrometry, PCC und Report-Start nicht.
- Die veralteten HTML-Dummys unter `doc/gui2/clickdummy/` sind nur Konzept-/Referenzmaterial und nicht Teil der produktiven Auslieferung.
