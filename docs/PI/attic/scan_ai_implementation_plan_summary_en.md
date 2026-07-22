# PI Scan-AI — Implementation Plan Summary

**Status:** Summary of the implementation plan  
**Goal:** Integrate PI Scan-AI optionally into scan and Parameter Studio.  
**Key point:** AI is off by default and requires explicit activation plus credentials.

---

## Core Architecture

The PI integration is implemented as a local Node/TypeScript sidecar because `@earendil-works/pi-coding-agent` can be used directly there:

```text
Frontend -> C++ Backend -> Node PI Sidecar -> @earendil-works/pi-coding-agent -> external AI
```

The C++ backend remains responsible for scan, jobs, config patches, and validation. The sidecar handles provider/model management, AuthStorage, `.env`, and the actual PI agent session.

---

## What Already Exists

| Area | Available |
|---|---|
| Scan | `/api/scan`, `/api/scan/latest`, JobStore |
| Config | `/api/config/schema`, `/api/config/defaults`, `/api/config/patch`, `/api/config/validate` |
| Revisions | `ConfigRevisionStore` |
| Frontend | `web_frontend/src/api.js`, `src/app.js`, `input-scan.html`, `parameter-studio.html` |
| Validation | CLI/Backend Config-Validate |

---

## Fixed Requirements

- AI is off by default.
- External requests happen only after explicit activation or confirmed one-shot.
- API keys come from PI `AuthStorage`, process environment, `agent_service/.env`, or project root `.env`.
- Secrets are never written to config, revisions, JobStore, logs, or UI state.
- All PI providers/models come from `ModelRegistry`.
- Internally, `updates: [{path, value}]` is the canonical patch format.
- Every AI recommendation is validated against schema and config validation.

---

## Target Files

### New: Sidecar

```text
agent_service/
  package.json
  tsconfig.json
  .env.example
  src/
    server.ts
    types.ts
    config.ts
    services/
      authService.ts
      modelService.ts
      frameAnalysisService.ts
```

### New: C++ Backend

```text
web_backend_cpp/include/routes/ai_routes.hpp
web_backend_cpp/src/routes/ai_routes.cpp
web_backend_cpp/include/services/ai_service.hpp
web_backend_cpp/src/services/ai_service.cpp
web_backend_cpp/tests/test_ai_routes.cpp
web_backend_cpp/tests/test_ai_service.cpp
```

### To Modify

```text
web_backend_cpp/src/main.cpp
web_backend_cpp/CMakeLists.txt
web_backend_cpp/include/backend_runtime.hpp
web_backend_cpp/src/backend_runtime.cpp
web_frontend/src/constants.js
web_frontend/src/api.js
web_frontend/src/app.js
web_frontend/input-scan.html
web_frontend/parameter-studio.html
web_frontend/i18n/de.json
web_frontend/i18n/en.json
web_frontend/style.css
web_frontend/layout-panels.css
.gitignore
```

---

## New APIs

### C++ Backend

| Route | Purpose |
|---|---|
| `GET /api/ai/config` | Read AI settings without secrets |
| `PATCH /api/ai/config` | Save AI settings without secrets |
| `GET /api/ai/models` | Read PI providers/models/auth status |
| `POST /api/ai/auth` | Store API key for provider |
| `POST /api/ai/test` | Test connection/model |
| `DELETE /api/ai/auth/<provider>` | Remove stored key |
| `POST /api/scan/analysis` | Analyse scan |
| `GET /api/scan/analysis/latest` | Retrieve latest analysis |
| `POST /api/scan/analysis/apply` | Apply validated recommendations |

### Node Sidecar

| Route | Purpose |
|---|---|
| `GET /health` | Sidecar health |
| `GET /models` | PI ModelRegistry |
| `POST /auth` | Store key in PI AuthStorage |
| `DELETE /auth/:provider` | Remove stored key |
| `POST /test` | Test prompt |
| `POST /analyze` | Scan-AI analysis |

---

## AI Configuration

Default:

```yaml
ai:
  scan_analysis:
    enabled: false
    mode: manual
    provider: ""
    model: ""
    temperature: 0
    max_tokens: 8000
    timeout_ms: 120000
    send_paths: false
    persist_recommendations: false
```

Credential priority:

1. PI `AuthStorage`
2. Process environment
3. `agent_service/.env`
4. Project root `.env`

Typical `.env` keys:

```dotenv
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
GROQ_API_KEY=...
OPENROUTER_API_KEY=...
```

Secrets are never written to config, revisions, JobStore, logs, or UI state.

---

## Implementation Roadmap

### Phase 0: Setup

- Define Node version.
- Create `agent_service`.
- Include `@earendil-works/pi-coding-agent`.
- Sidecar port `127.0.0.1:3001` as default.

### Phase 1: Sidecar

- `/health`
- Load `.env`
- PI `AuthStorage`
- PI `ModelRegistry`
- `/models`, `/auth`, `/test`

### Phase 2: Analysis

- `FrameAnalysisService`
- PI session with `createAgentSession`
- Prompt + JSON-only contract
- `/analyze`

### Phase 3: C++ Backend

- `ai_routes`
- `ai_service` as sidecar client
- `/api/ai/*`
- `/api/scan/analysis*`
- Report missing sidecar structurally

### Phase 4: Validation

- AI response is untrusted.
- Only allow known schema paths.
- Canonical `updates: [{path, value}]`.
- Validate patch against config schema and `validate-config`.
- Only make validated updates applicable.

### Phase 5: Frontend

- AI settings panel.
- Provider/model selection.
- API key input/test.
- Scan analysis panel.
- Recommendations with confidence/risk/warnings.
- Apply selected recommendations.

### Phase 6: Tests

- Default off.
- `.env` detected.
- No secret leaks.
- Sidecar unavailable.
- AI disabled.
- Unknown path rejected.
- Wrong type rejected.
- Apply creates revision `pi_scan_ai`.

---

## Must-Have Acceptance Criteria

- [ ] Scan works unchanged without AI.
- [ ] AI is off by default.
- [ ] No external requests without activation/credentials.
- [ ] All PI models from `ModelRegistry` can be displayed.
- [ ] API keys from AuthStorage, environment, or `.env` are detected.
- [ ] Secrets are never output or persisted.
- [ ] `/api/scan/analysis` delivers validated recommendations.
- [ ] Invalid recommendations are discarded.
- [ ] Apply creates validated config and optional revision.
- [ ] Parameter Studio shows confidence, risk, rationale, and warnings.

---

## Key Risks

| Risk | Mitigation |
|---|---|
| Unexpected external AI costs | `enabled=false`, `mode=manual`, visible UI action |
| Secret leak | Redaction, no secret responses, tests |
| Invalid AI config | Schema filter + config validate |
| Sidecar down | Backend remains usable |
| Wrong patch format | only use `updates[]` internally |
| Absolute paths to AI | `send_paths=false` default |

---

## Rollback

1. Disable AI.
2. Do not start sidecar.
3. Optionally remove `register_ai_routes(...)`.
4. Scan, config, and Parameter Studio continue as before.

---

**Full plan:** [`scan_ai_implementation_plan_en.md`](./scan_ai_implementation_plan_en.md)  
**Specification:** [`scan_ai_parameter_studio_en.md`](../scan_ai_parameter_studio_en.md)
