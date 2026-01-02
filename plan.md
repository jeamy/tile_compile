# Tile Compile GUI + Backend Plan (local single-host)

## Ziel
Eine lokale Desktop-GUI (Tauri), die eine deterministische Tile-Compile Pipeline orchestriert.
Backend läuft lokal (Python), wird via IPC (Tauri Commands) aufgerufen und streamt Events/Logs.

## Scope (jetzt)
- Transport: IPC (Tauri Commands) gemäß `backend_api.md`
- Schema: `tile_compile.schema.json` ist die Quelle für Config-Validation und GUI-Editor
- Local-only Runs im Filesystem (Run-Folder Contract)

## Milestones

### M1 — Python Backend Skeleton (Library + CLI)
Deliverables:
- `tile_compile_backend/` Package (scan/config/validate/runs/artifacts)
- `tile_compile_backend_cli.py` mit Subcommands (JSON stdout)
- `pyproject.toml` (Dependencies: pyyaml, astropy, jsonschema)

### M2 — Input Scan (FITS) nach `input_scan_spec.md`
Deliverables:
- `scan_input(input_path)`:
  - erkennt Frames (stable sort)
  - validiert `NAXIS1/NAXIS2`
  - bestimmt `frames_detected`, `image_width`, `image_height`, `color_mode` (best-effort)
  - erzeugt `frames_manifest.json` + `frames_manifest_id` (sha256)

### M3 — Config IO + Validation
Deliverables:
- `load_config(path)` / `save_config(path,yaml)`
- `validate_config(yaml_text, schema)`:
  - JSON Schema validation (`tile_compile.schema.json`)
  - Cross-field checks (weights sums, clamp order, min/max constraints)

### M4 — Run Management (local)
Deliverables:
- `start_run(...)` / `abort_run(run_id)` via Tauri (Rust) -> Python runner subprocess
- `list_runs(runs_dir)` / `get_run_status(run_id)` / `get_run_logs(...)` / `list_artifacts(run_id)`
- Stabiler Run Folder Contract:
  - `runs/<run_id>/config.yaml`
  - `runs/<run_id>/events.jsonl`
  - `runs/<run_id>/status.json` (optional, derived)
  - `runs/<run_id>/artifacts/` (plots, outputs)

### M5 — GUI Screens (Tauri Frontend)
Deliverables:
- Input Screen (Scan + derived metadata read-only)
- Config Screen (Schema-driven editor + raw YAML)
- Run Screen (Start/Abort + Live Events + Artifacts)
- Run History Screen

## Nicht-Ziele (jetzt)
- RabbitMQ / verteilte Worker
- Remote backend/server mode
- Interaktive Bildansicht/Processing in der GUI
