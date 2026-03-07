# FE Contract: Tool Jobs (ASTAP/PCC)

Diese Datei definiert die Live-Contract-Felder fuer `GET /api/jobs/{job_id}` und `WS /api/ws/jobs/{job_id}`.

## 1) Gemeinsame Job-Felder

Jeder Job liefert:
- `job_id`
- `type`
- `state` (`pending|running|ok|error|cancelled`)
- `pid` (optional)
- `exit_code` (optional)
- `created_at`, `updated_at`
- `data` (job-spezifische Live-Felder)

## 2) Live-Felder in `data` (Download-Jobs)

Diese Felder gelten fuer:
- `astrometry_install_cli`
- `astrometry_catalog_download`
- `pcc_siril_download`

### Progress/Transfer
- `stage`: `download|extract|decompress|done`
- `bytes_received`: aktuell empfangene Bytes
- `bytes_total`: Gesamtbytes (falls vom Server bekannt)
- `progress`: `0..1` (falls `bytes_total > 0`)

### Retry/Resume
- `retry_count`: konfigurierte Retry-Anzahl
- `resume_enabled`: ob Resume aktiv ist
- `attempt`: aktuelle Attempt-Nummer
- `retrying`: `true|false` (nur bei Fehlern/Retry-Zyklus)
- `error`: letzte Fehlermeldung (bei Fehler)
- `status_code`: letzter HTTP-Status (z. B. `200`, `206`)
- `resumed`: `true|false` ob letzter Download per HTTP-Range fortgesetzt wurde
- `existing_bytes`: bereits vorhandene Dateigroesse vor Attempt

### Job-spezifisch
- `astrometry_install_cli`:
  - `url`, `data_dir`, `archive`, `binary`
- `astrometry_catalog_download`:
  - `catalog_id`, `url`, `archive`, `installed`
- `pcc_siril_download`:
  - `catalog_dir`
  - `pending_chunks` (Liste)
  - `total_chunks`
  - `current_chunk`
  - `current_index`
  - `completed_chunks`
  - `missing_after`

## 3) Run/Tool Jobs ohne HTTP-Download

- `astrometry_solve`:
  - `command`
  - `wcs_path` (abgeleiteter erwarteter WCS-Pfad)
- `pcc_run`:
  - `command`

## 4) FE Polling/WS Empfehlung

## Primär
- `WS /api/ws/jobs/{job_id}` fuer Statuswechsel und `pid/exit_code`.

## Sekundär
- Alle `0.5..1.0s` `GET /api/jobs/{job_id}`, um `data.*` (Progress/Stage/Chunk) zu aktualisieren.

## Abbruch
- Button `Cancel` -> `POST /api/jobs/{job_id}/cancel`.
- FE setzt UI sofort auf "Cancelling...", bis `state == cancelled|ok|error`.

## 5) Retry/Resume Parameter (Request-Seite)

Unterstuetzt fuer:
- `POST /api/tools/astrometry/install-cli`
- `POST /api/tools/astrometry/catalog/download`
- `POST /api/tools/pcc/siril/download-missing`
- sowie `/retry`-Alias-Endpunkte

Request-Optionen:
- `retry_count: int` (Default `2`)
- `retry_backoff_sec: float` (Default `1.5`)
- `resume: bool` (Default `true`)
- `timeout_s: int` (Endpoint-spezifischer Default)
- `force_restart: bool` (Default `false`, loescht existierende Teilarchive vor neuem Download)

## 6) Fehlerdarstellung FE

Bei `state == error`:
- Primaer `data.error`
- Optional Transferkontext anzeigen:
  - `attempt`, `status_code`, `url`, `archive/current_chunk`
