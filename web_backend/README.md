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

Falls nicht gesetzt, werden Standardpfade unter `tile_compile_cpp/build/...` verwendet.

## API-Status

Die API in `web_backend/app/api` ist auf den GUI2-Vertrag verdrahtet:
- App State/Constants
- Config (Schema, current, validate, save, presets, revisions)
- Scan/Quality/Guardrails
- Runs (list/status/logs/artifacts/start/resume/stop/stats)
- Tools (Astrometry + PCC Endpunkte)
- Jobs (list/get/cancel)
- WebSocket Streams (`runs`, `jobs`, `system`)

Hinweis: Tool-Aktionen fuer ASTAP/Siril sind als echte Download-/Install-/Solve-Jobs implementiert.
Abhaengigkeiten auf dem Host: `unzip` (ASTAP zip), optional `dpkg-deb` fuer ASTAP D80 `.deb`-Katalog.

Run-Monitoring nutzt primär `logs/run_events.jsonl` (mit Fallback auf `events.jsonl`), inkl. Phase-Status und Gesamt-Progress.
`POST /api/runs/start` unterstützt neben `input_dir` auch serielle Queues über `queue[]` oder `input_dirs[]`.

## Tests

```bash
python3 -m pytest -q web_backend/tests
```
