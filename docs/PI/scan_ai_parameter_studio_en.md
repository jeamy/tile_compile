# PI Scan-AI for Parameter Studio

**Status:** Draft for implementation  
**Goal:** During input scan, frames should not only be technically detected but qualitatively evaluated. From this analysis, a suitable, validated configuration for the Parameter Studio should emerge, so the stacking process starts with the best possible initial configuration.

## 1. Target Vision

The current scan already provides basic data:

- Frame count
- Image size
- `color_mode`
- `bayer_pattern`
- Warnings and errors
- Frame list

The new PI analysis tool extends this step with a qualitative frame analysis and generates concrete configuration suggestions from it. The AI does not make unchecked final decisions. It creates explained recommendations that are validated by the backend against schema and guardrails and can be applied or adjusted in the Parameter Studio as suggestions.

## 2. Technical Approach

Since `tile_compile` currently uses a C++ backend and a static frontend, but the referenced package `@earendil-works/pi-coding-agent` is TypeScript/Node, the AI integration should be implemented as a small local sidecar service.

Recommended structure:

```text
tile_compile/
  agent_service/
    package.json
    tsconfig.json
    src/
      server.ts
      services/
        authService.ts
        modelService.ts
        frameAnalysisService.ts
```

## 3–14. Detailed Sections

The full specification covers:

- **Scan metrics:** FWHM, noise, background, roundness, star count, aggregates (median, mean, std, min, max, p10, p90)
- **AI prompt structure:** System instruction, config schema, current config, scan result, image quality metrics, frame statistics
- **Strict AI rules:** Recommended value must differ from current; type must match schema; enum/min/max constraints; no object/array paths; no file paths; no critical warnings for assumptions
- **Validation:** Full-patch validation (fast path) and per-update validation (fallback); common rejection reasons
- **Output format:** `pi.scan-analysis.v1` with summary, confidence, detected scenarios, recommendations, warnings
- **Security:** No API keys in config/revisions/logs; AI off by default; sidecar on `127.0.0.1` only; timeouts to prevent blocking; optional job treatment
- **Fallback without AI:** Deterministic scenario deltas from existing rules; UI shows "AI disabled"; Parameter Studio remains usable with manual scenario chips

See the [German full version](scan_ai_parameterstudio.md) for the complete detailed specification.

## 15. Fallback Without AI

When AI is disabled or no model is available:

- Backend delivers deterministic scenario deltas from existing rules.
- UI shows `AI disabled` or `AI not configured`.
- Parameter Studio remains usable with manual scenario chips.

This is important for reproducible runs and installations without cloud access.

## 16. Implementation Order

1. Create `agent_service` with `FrameAnalysisService` and local HTTP endpoint.
2. Implement sidecar endpoints for `/models`, `/auth`, `/test` and auth status.
3. Connect backend routes `GET/PATCH /api/ai/config`, `GET /api/ai/models`, `POST /api/ai/auth`, `POST /api/ai/test`.
4. Connect backend route `POST /api/scan/analysis`.
5. Implement JSON contract and validation.
6. Extend Parameter Studio with AI settings, PI analysis panel, and apply buttons.
7. Gradually extend scan metrics: header aggregates, image statistics, star/gradient metrics.
8. Store config revisions with source `pi_scan_ai`.
9. Tests for default-off, JSON validation, schema filter, apply behaviour, and missing API key.

## 17. Acceptance Criteria

- Scan works unchanged without AI configuration.
- AI is off by default.
- All PI-registered providers/models can be displayed and selected.
- API keys/credentials can be set up and tested for external AIs.
- API keys are detected from PI `AuthStorage`, process environment, or `.env` if present.
- API keys are not stored in config, revisions, logs, or job data.
- With configured PI agent, `/api/scan/analysis` produces a structured recommendation.
- Only schema-valid parameters are suggested or applied.
- Parameter Studio shows deltas, confidence, and rationale.
- Apply produces a validated config and optionally a revision.
- Colour mode/Bayer pattern are not automatically finalised on ambiguous evidence.
- Errors in agent service do not block scan or manual Parameter Studio.
