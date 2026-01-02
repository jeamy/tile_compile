# Tile Compile – Backend API Specification

**Zweck:** Definition der Schnittstelle zwischen GUI (Tauri) und Backend

---

## 1. Allgemeines

* API ist lokal
* synchron für Steuerung, asynchron für Status
* alle Antworten sind JSON

Hinweis:

* Diese Spezifikation ist **transport-neutral**.
* Die Implementierung erfolgt zunächst über **IPC (Tauri Commands)**.
* Die HTTP-Endpunkte dienen als **logisches API-Modell**.

---

## 2. Endpunkte

### 2.1 Input Scan

```
POST /scan
```

Request:

```json
{ "input_path": "/path/to/frames" }
```

Response:

```json
{
  "image_width": 3840,
  "image_height": 2160,
  "frames_detected": 1032,
  "color_mode": "OSC",
  "frames_manifest_id": "sha256"
}
```

---

### 2.2 Start Run

```
POST /runs
```

Request:

```json
{
  "input_path": "/path",
  "config_path": "/path/tile_compile.yaml",
  "execution_mode": "local"
}
```

Response:

```json
{ "run_id": "uuid" }
```

---

### 2.2.1 List Runs (History)

```
GET /runs
```

Response:

```json
[
  {
    "run_id": "uuid",
    "status": "SUCCESS",
    "created_at": "2026-01-02T18:01:02Z",
    "run_dir": "/abs/path/runs/<ts>_<run_id>",
    "config_hash": "sha256",
    "frames_manifest_id": "sha256"
  }
]
```

---

### 2.3 Run Status

```
GET /runs/{run_id}
```

Response:

```json
{
  "status": "RUNNING",
  "phase": "TILE_ANALYSIS",
  "progress": { "tiles_done": 203, "tiles_total": 312 }
}
```

Erweiterungen (optional, empfohlen):

```json
{
  "status": "RUNNING",
  "phase": "TILE_ANALYSIS",
  "phase_id": 5,
  "phase_name": "local_tile_metrics",
  "progress": { "tiles_done": 203, "tiles_total": 312 },
  "paths": {
    "run_dir": "/abs/path/runs/<ts>_<run_id>",
    "logs": "/abs/path/.../logs",
    "artifacts": "/abs/path/.../artifacts",
    "outputs": "/abs/path/.../outputs"
  },
  "config_hash": "sha256",
  "frames_manifest_id": "sha256"
}
```

---

### 2.4 Abort Run

```
POST /runs/{run_id}/abort
```

Response:

```json
{ "status": "ABORTING" }
```

---

### 2.5 Logs

```
GET /runs/{run_id}/logs
```

Optional (für Live-UI):

```
GET /runs/{run_id}/logs?tail=500
```

---

### 2.6 Artefakte

```
GET /runs/{run_id}/artifacts
```

---

### 2.7 Schema

```
GET /schema
```

Response:

* JSON Schema (Draft 2020-12), Datei: `tile_compile.schema.json`

---

### 2.8 Config IO

#### 2.8.1 Load Config

```
GET /config
```

Query:

```json
{ "path": "/path/tile_compile.yaml" }
```

Response:

```json
{ "yaml_text": "..." }
```

#### 2.8.2 Save Config (nur vor Run)

```
PUT /config
```

Request:

```json
{ "path": "/path/tile_compile.yaml", "yaml_text": "..." }
```

Response:

```json
{ "status": "OK" }
```

---

### 2.9 Config Validation

```
POST /config/validate
```

Request:

```json
{
  "yaml_text": "...",
  "input_path": "/path/to/frames"
}
```

Response:

```json
{
  "valid": true,
  "errors": [],
  "warnings": []
}
```

Validation umfasst:

* JSON Schema Validation
* Methodik-Gates (z. B. `normalization.enabled = true`)
* Cross-Field Checks (z. B. Gewichtssummen)

---

## 3. Events / Log-Model (asynchron)

Die Ausführung publiziert Events als JSON (z. B. JSONL), die die GUI live anzeigen kann.

Minimal:

* `run_start` (run_id, config_hash, frames_manifest_id, frame_count)
* `phase_start` (phase_id, phase_name)
* `phase_end` (phase_id, phase_name, status)
* `run_end` (status)
* `run_stop_requested`

Optional:

* `progress` (tiles_done/tiles_total)
* `warning` / `error`

## 4. Statuswerte

* PENDING
* RUNNING
* ABORTING
* ABORTED
* FAILED
* SUCCESS

---

## 5. Abbruch-Semantik

* Abort ist final
* kein Resume
* teilfertige Artefakte werden nicht weiterverwendet

---

## 6. Fehlercodes

* 400 – ungültige Konfiguration
* 404 – Run nicht gefunden
* 409 – Run nicht abbrechbar
* 500 – interner Fehler

---

## 7. Transport: HTTP → IPC Mapping (Tauri Commands)

Die IPC-Commands spiegeln die HTTP-Endpunkte semantisch.

| HTTP (logisch)                 | IPC Command (Tauri)        |
| ----------------------------- | -------------------------- |
| `POST /scan`                  | `scan_input(input_path)`   |
| `GET /schema`                 | `get_schema()`             |
| `GET /config`                 | `load_config(path)`        |
| `PUT /config`                 | `save_config(path,yaml)`   |
| `POST /config/validate`       | `validate_config(...)`     |
| `POST /runs`                  | `start_run(...)`           |
| `GET /runs`                   | `list_runs(runs_dir)`      |
| `GET /runs/{run_id}`          | `get_run_status(run_id)`   |
| `POST /runs/{run_id}/abort`   | `abort_run(run_id)`        |
| `GET /runs/{run_id}/logs`     | `get_run_logs(run_id,...)` |
| `GET /runs/{run_id}/artifacts`| `list_artifacts(run_id)`   |
