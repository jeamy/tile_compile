# PI – AI-Assisted Configuration Recommendations

> 🇩🇪 [Deutsche Version](pi_ki_empfehlungen_de.md)

**Status:** Production (v0.3.3, 2026-06-16)  
**Module:** `agent_service` (TypeScript sidecar) + `web_backend_cpp` AI routes  

---

## Overview

The **PI** (Parameter Intelligence) module analyses the result of a scan job and generates
AI-assisted configuration recommendations for `tile_compile`. Recommendations are validated
against the JSON schema, tested individually if needed (per-update validation), and can be
applied directly in the Parameter Studio.

---

## Architecture

```
Frontend (app.js)
  ├── POST /api/scan/metrics        → scan-metrics CLI → image quality aggregates
  ├── POST /api/scan/analysis       → web_backend_cpp AI route
  │       ├── scan_result           (frame metadata: count, gain, exposure …)
  │       ├── scan_metrics          (FWHM, SNR, noise, roundness, star count – aggregates)
  │       ├── base_config           (current tile_compile.yaml, flattened)
  │       ├── config_schema         (path → type, enum, min, max, description)
  │       └── allowed_config_paths  (paths permitted for recommendations)
  │
  └── agent_service (Node/TypeScript)
        ├── buildPrompt()           → structured prompt with all data sections
        ├── Claude / Anthropic API  → JSON response (schema_version: pi.scan-analysis.v1)
        └── validate_updates_against_schema()
              ├── full-patch validation (fast path)
              └── per-update validation (isolates invalid recommendations)
```

---

## Prompt Structure

The prompt contains six sections:

| Section | Content |
|---------|---------|
| **System instruction** | Role, output format (JSON), strict rules |
| **CONFIG SCHEMA** | All allowed paths with `type`, `enum`, `min`, `max`, description |
| **CURRENT CONFIG** | Current values as `path = value` |
| **SCAN RESULT** | Frame metadata (count, camera, gain, exposure, errors …) |
| **IMAGE QUALITY METRICS** | Aggregates from `scan-metrics` (FWHM, noise, roundness, SNR, star count) |
| **FRAME STATISTICS** | Per-frame distribution (if available) |

### Strict AI Rules in the Prompt

- The recommended value must differ from the current value.
- Type must match the schema (`integer`, `number`, `boolean`, `string`).
- `enum` values: only allowed values are permitted.
- **`min`/`max` constraints: value MUST be within the allowed range.**
- No `object` or `array` paths.
- No file or directory paths.
- No CRITICAL warnings for assumptions or generic astrophotography advice.

---

## Validation

### Full-patch validation (fast path)

All recommended updates are applied as a patch to the current config and validated via
`tile_compile_cli validate-config --stdin`. If successful, all updates are accepted.

### Per-update validation (fallback)

If the full patch fails, each update is tested **individually and cumulatively**:

```
for each update u:
  trial = current_base + u
  validate(trial)
  → OK:   current_base = trial, u is marked "applicable: true"
  → FAIL: u is rejected (reject_reason: "config_validation_failed")
```

This prevents a single invalid update from blocking all other valid recommendations.

### Common rejection reasons

| Reason | Cause |
|--------|-------|
| `config_validation_failed` | Recommended value violates a schema constraint (e.g. `max: 16`) |
| `unknown_path` | Path does not exist in the schema |
| `type_mismatch` | Wrong data type |
| `same_value` | New value is identical to the current value |

---

## Image Quality Metrics (`scan_metrics`)

The `scan-metrics` phase computes the following aggregates per dataset:

| Metric | Meaning |
|--------|---------|
| `fwhm` | Star sharpness (Full Width at Half Maximum) in pixels |
| `noise` | Background noise (σ) |
| `background` | Background brightness (median) |
| `roundness` | Star roundness (1.0 = perfectly round) |
| `star_count` | Detected stars per frame |

Aggregates include `median`, `mean`, `std`, `min`, `max`, `p10`, `p90`.

The AI uses these values as **measured facts** — not assumptions — for recommendations
regarding `aqmh.cherry_pick`, `local_metrics`, `global_metrics.weights`, and sigma-clip parameters.

### Metrics Cache

`POST /api/scan/metrics` reuses previously computed image statistics when the cache key matches:

```
input_path | normalized object_name | frame_count
```

Successful results are stored both in the backend job store and on disk under:

```
runs/.pi_memory/scan_metrics_cache/
```

On a cache hit the endpoint returns immediately:

```json
{
  "cached": true,
  "state": "ok",
  "result": {
    "cache_hit": true,
    "cache_source": "job_store | disk",
    "cache_source_job_id": "...",
    "cache_key": "..."
  }
}
```

The frontend logs this as `Image statistics cache hit` and skips the `scan-metrics` job. A new
metrics job is started only when no matching result exists, or when the input path, object name,
or frame count changes.

### Sidecar Timeouts

Backend calls to the PI sidecar use a short connection timeout and a long analysis timeout:

| Env variable | Default | Purpose |
|--------------|---------|---------|
| `AI_AGENT_CONNECT_TIMEOUT_MS` | `10000` | Fail quickly only when the sidecar cannot be reached. |
| `AI_AGENT_ANALYSIS_TIMEOUT_MS` | `1200000` | Minimum timeout for `/analyze` calls; prevents long provider calls from being reported as sidecar unavailable. |

`AI_SCAN_TIMEOUT_MS` still configures the general AI timeout, but `/analyze` uses the larger of
`AI_SCAN_TIMEOUT_MS` and `AI_AGENT_ANALYSIS_TIMEOUT_MS`.

---

## Output Format

```json
{
  "schema_version": "pi.scan-analysis.v1",
  "summary": "…",
  "confidence": 0.83,
  "detected_scenarios": ["osc_short_exposure", "large_frame_count", …],
  "recommendations": [
    {
      "id": "rec_sigma_high",
      "path": "stacking.sigma_clip.sigma_high",
      "value": 2.5,
      "current_value": 3.0,
      "confidence": 0.91,
      "review_required": false,
      "rationale": "…"
    }
  ],
  "warnings": ["…"]
}
```

Saved to `.ai_analyses/<Target>_<Date>.json`.

---

## Configuration

| Environment variable | Meaning | Default |
|---|---|---|
| `TILE_COMPILE_AI_AGENT_AUTOSTART` | Auto-start sidecar | `1` |
| `TILE_COMPILE_AI_MODEL` | AI model | `claude-sonnet-4-6` |
| `TILE_COMPILE_BACKEND_SUBPROCESS_CAPTURE_BYTES` | Max stdout for CLI subprocesses | `1048576` (1 MB) |
| `ANTHROPIC_API_KEY` | Anthropic API key (in `.env`) | – |

---

## Implementation Files

| File | Role |
|---|---|
| `agent_service/src/services/frameAnalysisService.ts` | Prompt construction, AI call, traffic log |
| `web_backend_cpp/src/routes/ai_routes.cpp` | Schema export, payload construction, per-update validation |
| `web_backend_cpp/src/services/ai_service.cpp` | Sidecar HTTP client |
| `web_frontend/src/app.js` | scan-metrics fetch, analysis trigger, UI integration |
| `tile_compile_cpp/tile_compile.schema.json` | Authoritative schema source (min/max/enum/desc) |

---

## Related Documents

- [PI Scan-AI Implementation Plan](scan_ai_implementierungsplan.md)
- [PI Parameter Studio](scan_ai_parameterstudio.md)
- [Configuration Reference](../configuration_reference_en.md)
- [AQMH Methodology](../AQMH/aqmh_methodik_en.md)
- 🇩🇪 [Deutsche Version](pi_ki_empfehlungen_de.md)
