# REST API Reference

The GUI2 backend (`web_backend_cpp`) exposes a REST API via Crow.

## Base URL

```
http://localhost:8080
```

## Endpoints

### System

- `GET /api/system/info` — backend version, capabilities
- `GET /api/system/health` — health check

### Jobs

- `POST /api/jobs/run` — start pipeline run
- `POST /api/jobs/resume` — resume existing run
- `GET /api/jobs` — list jobs
- `GET /api/jobs/:id` — job status
- `DELETE /api/jobs/:id` — cancel job

### Runs

- `GET /api/runs` — list runs
- `GET /api/runs/:id` — run details
- `GET /api/runs/:id/logs` — run logs
- `GET /api/runs/:id/artifacts` — artifact list
- `GET /api/runs/:id/report` — quality report

### Config

- `POST /api/config/validate` — validate config YAML
- `POST /api/config/suggest` — parameter suggestions
- `GET /api/config/schema` — config schema

### File Browser

- `GET /api/files/list` — list directory
- `GET /api/files/grant` — request path access
- `POST /api/files/scan` — scan for frames

### PCC

- `POST /api/pcc/run` — run photometric calibration
- `POST /api/pcc/apply` — apply calibration matrix

For detailed endpoint parameters, see `web_backend_cpp/src/routes/`.
