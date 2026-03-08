# GUI2 FastAPI Backend

## Start (Dev)

```bash
cd /media/data/programming/tile_compile
source .venv/bin/activate
pip install -r web_backend/requirements-backend.txt
uvicorn app.main:app --app-dir web_backend --reload --port 8080
```

## Konfiguration (Runtime)

Optionale Env-Variablen:

- `TILE_COMPILE_CLI` (Pfad zu `tile_compile_cli`)
- `TILE_COMPILE_RUNNER` (Pfad zu `tile_compile_runner`)
- `TILE_COMPILE_RUNS_DIR` (Default Runs-Verzeichnis)
- `TILE_COMPILE_CONFIG_PATH` (Default Config-Datei)
- `TILE_COMPILE_STATS_SCRIPT` (Default: `tile_compile_cpp/scripts/generate_report.py`)
- `TILE_COMPILE_ALLOWED_ROOTS` (Pfadliste, getrennt mit `:`; beschraenkt erlaubte Dateisystem-Zugriffe)

Falls nicht gesetzt, werden Standardpfade unter `tile_compile_cpp/build/...` verwendet.

## API-Status

Die API in `web_backend/app/api` ist auf den GUI2-Vertrag verdrahtet:
- App State/Constants
- UI Event Audit/Replay (`GET /api/app/ui-events`)
- Config (Schema, current, validate, save, presets, revisions)
- Scan/Quality/Guardrails
- Runs (list/status/logs/artifacts/start/resume/stop/stats)
- Tools (Astrometry + PCC Endpunkte)
- Jobs (list/get/cancel)
- WebSocket Streams (`runs`, `jobs`, `system`)

Hinweis: Tool-Aktionen fuer ASTAP/Siril sind als echte Download-/Install-/Solve-Jobs implementiert.
Abhaengigkeiten auf dem Host: `unzip` (ASTAP zip), optional `dpkg-deb` fuer ASTAP D80 `.deb`-Katalog.
Download-Jobs unterstuetzen Retry/Resume (`retry_count`, `retry_backoff_sec`, `resume`, `timeout_s`, `force_restart`).
Zusaetzliche Retry-Aliase:
- `POST /api/tools/astrometry/install-cli/retry`
- `POST /api/tools/astrometry/catalog/download/retry`
- `POST /api/tools/pcc/siril/download-missing/retry`

Run-Monitoring nutzt primär `logs/run_events.jsonl` (mit Fallback auf `events.jsonl`), inkl. Phase-Status und Gesamt-Progress.
`POST /api/runs/start` unterstützt neben `input_dir` auch serielle Queues über `queue[]` oder `input_dirs[]`.
`POST /api/runs/start` blockiert bei Guardrail-`error` (HTTP `409`).
`POST /api/runs/{run_id}/resume` erfordert `from_phase` + `config_revision_id` (HTTP `409`/`404` bei Verstoß).
Zusätzlicher Restore-Endpunkt für Resume-Flows:
- `POST /api/runs/{run_id}/config-revisions/{revision_id}/restore`

Asynchrone Job-Starts liefern `202 Accepted`, u. a.:
- `/api/scan`
- `/api/runs/start`, `/api/runs/{run_id}/resume`, `/api/runs/{run_id}/stats`
- Tool-Job-Endpunkte (`install/download/solve/run`)

Run-WebSocket (`/api/ws/runs/{run_id}`) streamt vertragliche Events:
- `phase_start`, `phase_progress`, `phase_end`, `run_end`, `queue_progress`, `log_line`
- plus `run_status` als Resync-Fallback.

Sicherheitsregeln im Backend:
- Command-Whitelist: nur freigegebene Executables (`tile_compile_cli`, `tile_compile_runner`, Stats-Script via Python, ASTAP/dpkg-deb).
- Path-Policy: API-Pfade werden gegen `TILE_COMPILE_ALLOWED_ROOTS` validiert (`PATH_NOT_ALLOWED`, `PATH_NOT_FOUND` als `422`).
- Einheitliches Fehlerformat fuer `4xx/5xx`: immer `{ \"error\": { code, message, hint?, details? } }`.

FE-Job-Contract (Live-Felder in `job.data`):  
[fe_contract_tools_jobs.md](/media/data/programming/tile_compile/web_backend/fe_contract_tools_jobs.md)

Jobstatus (`GET /api/jobs`, `GET /api/jobs/{job_id}`) liefert:
- `created_at`, `updated_at`
- `started_at`, `ended_at`
- `run_id` (falls vorhanden)

## Tests

```bash
python3 -m pytest -q web_backend/tests
```

Optionale echte Binary-Integrationstests:

```bash
WEB_BACKEND_ENABLE_BINARY_TESTS=1 python3 -m pytest -q -rs web_backend/tests/test_integration_binaries.py
```
