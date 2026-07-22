# PI Scan-AI Implementation Plan

**Status:** Implementation begun, backend groundwork complete  
**Date:** 2026-06-15  
**Goal:** Integrate PI Scan-AI into `tile_compile` so scan results can be translated into validated Parameter Studio recommendations.  
**Principle:** AI is off by default. Without explicit activation and valid credentials, no external AI requests are executed.

---

## 0. Implementation Status

Stand: 2026-06-15.

Completed:

- [x] Optional `agent_service` Node/TypeScript project created.
- [x] `@earendil-works/pi-coding-agent` included as dependency.
- [x] `.env.example` and `.env` loading from project root and `agent_service/.env`.
- [x] Sidecar routes `/health`, `/models`, `/auth`, `DELETE /auth/:provider`, `/test`, `/analyze`.
- [x] C++ AI service and sidecar HTTP client created.
- [x] C++ backend routes `/api/ai/*` and `/api/scan/analysis*` registered.
- [x] AI is off by default.
- [x] Backend works without running sidecar.
- [x] `/api/ai/models` reports missing sidecar structurally as `AI_AGENT_UNAVAILABLE`.
- [x] API keys are not stored in backend config, UI state, or responses.
- [x] Frontend API constants for AI/scan analysis added.
- [x] Release/packaging scripts bundle the optional `agent_service`.
- [x] Backend test for default-off and sidecar-unavailable created.
- [x] `cmake --build web_backend_cpp/build` successful.
- [x] `ctest --output-on-failure -R web_backend_cpp_ai_routes` successful.

---

## 1–10. Architecture, APIs, Configuration, Validation, Frontend, Tests

The full implementation plan covers:

- **Architecture:** Frontend → C++ Backend → Node PI Sidecar → `@earendil-works/pi-coding-agent` → external AI
- **Sidecar:** `agent_service/` with `server.ts`, `authService.ts`, `modelService.ts`, `frameAnalysisService.ts`
- **Backend routes:** `/api/ai/config`, `/api/ai/models`, `/api/ai/auth`, `/api/ai/test`, `/api/scan/analysis`, `/api/scan/analysis/latest`, `/api/scan/analysis/apply`
- **Sidecar routes:** `/health`, `/models`, `/auth`, `/auth/:provider`, `/test`, `/analyze`
- **AI config defaults:** `enabled: false`, `mode: manual`, `temperature: 0`, `max_tokens: 8000`, `timeout_ms: 120000`, `send_paths: false`, `persist_recommendations: false`
- **Credential priority:** PI AuthStorage → process env → `agent_service/.env` → project root `.env`
- **Validation:** AI response is untrusted; only known schema paths allowed; canonical `updates: [{path, value}]`; patch validated against config schema and `validate-config`
- **Frontend:** AI settings panel, provider/model selection, API key input/test, scan analysis panel, recommendations with confidence/risk/warnings, apply selected recommendations
- **Tests:** Default-off, `.env` detected, no secret leaks, sidecar unavailable, AI disabled, unknown path rejected, wrong type rejected, apply creates revision `pi_scan_ai`

See the [German full version](scan_ai_implementierungsplan.md) for the complete detailed plan.

---

## 11. Rollback

Rollback must remain trivial:

1. Do not start sidecar.
2. AI config `enabled=false`.
3. `register_ai_routes(...)` can be removed if necessary.
4. Backend and frontend continue to work with scan, config, and Parameter Studio.

Since AI is optional, no existing stacking or scan path may depend on AI.

---

## 12. Implementation Order (Compact)

1. Sidecar skeleton + `/health`.
2. `.env` + AuthStorage + ModelRegistry.
3. `/models`, `/auth`, `/test`.
4. `FrameAnalysisService` + `/analyze`.
5. C++ AI backend client.
6. C++ AI routes.
7. Schema/config validation. `[done in backend]`
8. JobStore/revision integration. `[done in backend]`
9. Frontend settings. `[basis done]`
10. Frontend scan analysis panel. `[basis done]`
11. Apply flow in Parameter Studio. `[basis done]`
12. Tests and secret redaction audit.
