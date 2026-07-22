# PI Integration Architecture Plan for Tile Compile

Status: reconstructed and rewritten 2026-07-14  
Target surface: GUI3 (`web_frontend_v3`). The old GUI (`web_frontend`) is not a target of this integration.

Reset decision: Previously stored AI/PI memory data does not need to be migrated or
compatibly read back. The next memory iteration starts from zero with a new store
and schema contract.

## Starting Point

Tile Compile already has PI as parameter optimisation in the Scan-AI/recommendation flow. The new integration extends this into a controlled PI layer that understands context, translates recommendations into action plans, pre-validates changes, provides a safe preview, applies changes explicitly, and can learn from reviewed optimisations across sessions.

- Tile Compile works config- and run-centrically, so `validate-config` is the central safety boundary.
- PI responses are not written directly but modelled as `pi.action-plan.v1`.
- Memories do not automatically learn "truths" but store reviewable optimisation experiences.
- Memories are stored globally and retrieved across projects. Project/run IDs serve only as evidence references, not as storage boundaries.
- Memory retrieval uses a detailed context signature from object, target type, acquisition device, optics, camera, filter, exposures, calibration, mount, sky/quality, and pipeline configuration.
- Previous AI/memory files are not a source of truth for the new architecture and may be ignored.
- GUI3 remains the only new interaction path.

## Target Vision

PI becomes an assistance and orchestration layer in Tile Compile:

- **Read context:** Runtime, current config, scan results, jobs, artifacts, reports.
- **Propose:** Scan-AI recommendations and later PI tools generate structured action plans.
- **Validate:** Action plans are formally and config-semantically validated server-side.
- **Preview:** GUI3 shows mutation-free YAML diffs.
- **Apply:** Write actions require explicit confirmation and produce config revisions.
- **Learn:** Successfully applied optimisations can be stored as memory candidates and reviewed.
- **Recall:** Accepted memories are searched semantically and structurally, weighted by evidence and scope, and given to the AI as bounded context.

## Safety Rules

- **No blind write:** Every config change goes through action plan, preview, and validation.
- **No unreviewed learning:** Memories start as `candidate`; only review makes them reliable for future sessions.
- **No old GUI:** New PI functionality is only built into `web_frontend_v3`.
- **Privacy by default:** Memories store metadata, config paths, rationales, and validation, but no image data.
- **Global memory policy:** Learning is not project-specific. No absolute image paths, raw images, or confidential local paths are stored as knowledge. Instead, stable, normalised metadata and optionally hashed run/artifact references are used.
- **No legacy obligation:** There is no migration path for previous AI/memory data. An old store is ignored for the new layer and not automatically evaluated.
- **No unexplained AI authority:** Every memory use must be marked in the prompt as historical experience with scope, confidence, evidence, and possible counterexamples.
- Existing Tile Compile commands remain the authority for config validity.

## Reconstructed Implementation Status

- [x] Plan file in `docs/PI/pi_integration_architektur_tile_compile_plan.md` reconstructed.
- [x] Backend PI routes reconstructed and registered in `main.cpp`.
- [x] PI service files for action plans, validator, context, tools, assistant, and memories reconstructed.
- [x] CMake targets and tests for PI components reconstructed.
- [x] Scan-AI routes extended with PI action plan enrichment.
- [x] Scan-AI apply extended with `learn=true` memory candidates.
- [x] GUI3 endpoints and AI recommendation page extended with PI Preview, PI Apply, and memories.
- [x] Old GUI cleaned of accidental PI changes.
- [x] Agent service traffic logging centralised and redacted.

Verification:

- [x] `node --check web_frontend_v3/js/pages/ai-empfehlung.js`
- [x] `node --check web_frontend_v3/js/api/endpoints.js`
- [x] `npm run build` in `agent_service`
- [x] Backend build for `tile_compile_web_backend`, `fake_tile_compile_cli`, PI tests, and AI routes test
- [x] `ctest --output-on-failure -R 'web_backend_cpp_(pi_routes|pi_memory_store|pi_action_plan|ai_routes)'`

## Phase 0 — Foundation

Goal: Create a shared language and safety model for PI.

- [x] Define action plan schema `pi.action-plan.v1`.
- [x] Implement `config.set` as the first safe action.
- [x] Implement action plan shape validator.
- [x] Convert Scan-AI `validated_updates` into action plans.
- [x] Create backend tests for action plan generation and validation.
- [x] Extract PI traffic log from duplicated locations in agent service.
- [x] Redact secrets, bearer tokens, API keys, and local project paths in traffic log.

Acceptance criteria:

- [x] Action plan without schema or without valid actions is rejected.
- [x] Scan-AI responses contain `action_plan` and `action_plan_validation`.
- [x] Logging errors do not abort analysis.

## Phase 1 — PI Context, Tools, and Assistant

Goal: PI can read Tile Compile context in a controlled way, without write access.

- [x] Implement `/api/pi/context`.
- [x] Implement `/api/pi/tools`.
- [x] Implement `/api/pi/tools/call`.
- [x] Implement `/api/pi/assistant/ask`.
- [x] Provide tool `context.overview`.
- [x] Provide tool `config.schema`.
- [x] Provide tools for artifact/report summary.
- [x] Provide preview tools for BGE/HMS as planning read/preview tools.

Acceptance criteria:

- [x] Context response contains runtime, state, job, and scan information.
- [x] Unknown tools are rejected.
- [x] Assistant responds without write side effects.
- [x] PI routes test covers context, tools, and assistant.

## Phase 2 — Action Plan Preview and Controlled Application

Goal: PI suggestions can be safely previewed before being applied.

- [x] Implement `/api/pi/action-plans/validate`.
- [x] Implement `/api/pi/action-plans/preview`.
- [x] Preview applies config changes mutation-free to a base config.
- [x] Preview validates result with `validate-config --stdin`.
- [x] Implement `/api/pi/action-plans/apply`.
- [x] Apply requires `confirmed=true`.
- [x] Apply can check `expected_patched_yaml` against preview result.
- [x] Apply stores config revision and UI event.
- [x] GUI3 AI page generates action plans from selected recommendations.
- [x] GUI3 shows PI Preview as YAML diff.
- [x] GUI3 allows explicit PI Apply.

Acceptance criteria:

- [x] Unconfirmed apply is rejected.
- [x] Preview produces YAML diff without writing config file.
- [x] Apply produces revision.
- [x] Tests cover preview, apply, and invalid config case.

## Phase 3 — Memories and Cross-Session Learning

Goal: Tile Compile remembers useful optimisation decisions in a reviewable and cross-session manner.

- [x] Implement PI memory store as JSONL files under a configurable PI storage.
- [x] First memory storage implemented; ignored for the new global memory layer.
- [x] Store candidates with `append_candidate`.
- [x] Implement `/api/pi/memories` for list and status filter.
- [x] Implement `/api/pi/memories/:id/review` for `accepted`, `rejected`, `deprecated`.
- [x] Implement `/api/pi/memories/retrieve` for simple path/type-based search.
- [x] Scan-AI apply can generate a `config_optimization` memory with `learn=true`.
- [x] GUI3 AI page shows memory list and review actions.
- [x] Sharpen memory storage from "project-specific" to "globally usable": project/run only as provenance, not as retrieval boundary. (covered by Phase 9: `memories_v2.jsonl`, global indexes)
- [x] Enrich memory candidates with a complete astro context signature on creation. (covered by Phase 9: `pi.context_signature.v1` builder)
- [x] Initialise new memory store without regard for old stored AI/memory data. (covered by Phase 9: store ignores `memories.jsonl`, reads only `memories_v2.jsonl`)
- [x] Only mark memory candidates as worth learning (`promotable`) when outcome evidence or user feedback is available. (Automatic promotion logic: `verdict=improved` sets `promotable`, promote endpoint for `promotable` → `accepted`)

Acceptance criteria:

- [x] Memory store test covers append, list, review, and retrieve.
- [x] PI routes test covers memory list, review, and retrieve.
- [x] AI routes test checks memory candidate after `learn=true`.
- [x] Memory JSON contains no absolute local image/project paths. (covered by Phase 9: tests in test_pi_routes.cpp and test_ai_routes.cpp)
- [x] Memory JSON contains at least `context_signature`, `evidence`, `scope`, `outcome`, `review`, and `provenance`. (covered by Phase 9: `pi.memory.v2` mandatory fields)
- [x] GUI3 clearly shows why a memory is globally reusable or why it should only apply locally/restricted. (covered by Phase 9: memory detail with scope, `does_not_apply_when`, scope editor)

Still open in Phase 3:

- [x] Automatically include accepted memories as context in the next Scan-AI request.
- [x] More clearly indicate in GUI3 which memories are only candidates and which are accepted/deprecated.
- [x] Add memory deduplication so identical config optimisations do not grow multiple times.

## Phase 4 — Memory Retrieval in the Optimisation Flow

Goal: PI uses accepted experiences without uncritically copying them.

- [x] Retrieve relevant memories based on config paths when building the Scan-AI request.
- [x] Only use `accepted` memories as strong context; `candidate` is not included in request context.
- [x] Mark memory context in the prompt clearly as "historical experience".
- [x] Use rejection status (`rejected`, `deprecated`) as a negative signal.
- [x] Secure memory retrieval with tests against false adoption.
- [x] Perform retrieval not only by config paths but by a weighted context signature. (`pi_memory_store.cpp`: 18 fields with `compare_text`, `compare_number_close`, `compare_set_overlap`)
- [x] Retrieve positive and negative memories together: accepted as possible strategy, rejected/deprecated as warning against repetition. (`positive_memories` and `negative_memories` in the `pi.ai-request.v2` container)
- [x] Include retrieval explanation in AI context: why this memory fits, which fields match, which do not. (`match_explanation` and `match_coverage` in `accepted_pi_memories` and `negative_pi_memories`)
- [x] Limit retrieval: diversity cap by object/camera classes, max 2 memories per class with context query, max 3 without. (`apply_diversity_cap()` in `pi_memory_store.cpp`)
- [x] Pass coverage fields from retrieval explicitly to the AI prompt: `retrieval_coverage_summary` with `systemically_missing_context_fields` in the `pi.ai-request.v2` container. (`pi_ai_request_builder.cpp`)
- [x] Mismatch penalty for actively wrong context fields: `object_type` (-6) and `camera_type` (-5) prevent cross-contamination even with config path overlap. (`pi_memory_store.cpp`)

Acceptance criteria:

- [x] A new Scan-AI request contains matching accepted memories.
- [x] Rejected memories are not used as recommendation context.
- [x] Tests verify that memories do not bypass schema/config validation.
- [x] Tests verify that a memory for e.g. "M42, nebula, OSC, extended emission" is not blindly applied to "M104, galaxy, mono/LRGB". (`test_pi_memory_store.cpp`: cross-contamination, galaxy_matches=0, path_overlap=0, nebula_positive=1)
- [x] Tests verify that a rejected memory appears as an explicit negative signal in the prompt for similar context. (`test_pi_memory_store.cpp`: rejected-signal, retrieve_negative returns 1 match with `retrieval_warning` and `match_explanation`)

## Phase 5 — Outcome Metrics and Quality Feedback

Goal: Learning becomes better than just "was applied".

- [x] Automatically capture relevant outcome metrics after runs: post-run trigger `/api/pi/memories/evaluate-run` reads run artifacts (stats.json) and writes outcome data into open memory candidates. (`pi_routes.cpp`)
- [x] Extend memory candidates with outcome fields.
- [x] GUI3 review shows applied paths, rationales, validation, and outcomes.
- [x] Weight accepted memories higher after positive outcome evidence.
- [x] Support deprecated memories for degrading or outdated optimisations.
- [x] Store outcome delta instead of single value: before/after for FWHM, background gradient, star count, report warnings, artifact status, resume phase, and user rating. (`evaluate_memory_outcome_payload` in `pi_routes.cpp`)
- [x] Only mark memory as `promotable` when at least one positive evidence is present: `verdict=improved` or `user_rating>=4` sets status to `promotable`. Promote endpoint `/api/pi/memories/:id/promote` for `promotable` → `accepted` transition.
- [x] Support negative learning: On `verdict=worse/unchanged`, `/api/pi/memories/:id/outcome` automatically creates a `counterexample` candidate with the same context and sets it as `rejected`. (`pi_routes.cpp`)

## Phase 6 — Extended PI Tools

Goal: PI can plan additional Tile Compile workflows, but still controlled.

- [x] Deepen BGE planning tool with real run/config data.
- [x] Deepen HMS/mosaic planning tool with real project parameters.
- [x] Generate resume/run planning as read-only plan.
- [x] Only enable write tools after action plan/preview/apply.
- [x] Version and document tool registry.

## Phase 7 — Audit, Export, and Operations

Goal: PI actions remain traceable and maintainable.

- [x] GUI3 audit view for action plans, applies, revisions, and memory reviews.
- [x] Export/import for PI memories with privacy filter.
- [x] CLI/backend tool for cleaning up and deduplicating memories.
- [x] Regression tests for typical optimisation cases: OSC/mono, BGE, HMS, AQMH, PCC.
- [x] User documentation for workflow: recommendation, preview, apply, learn, review.

## Phase 8 — Run Chat and Natural Quality Feedback

Goal: After a completed run, the user can describe in natural language what seems wrong with the image. PI connects this description with run context, artifacts, reports, config, and memories and produces traceable diagnosis and optimisation suggestions.

Example from `runs/run_20260714_091851`:

> Stars at the top have black cores. The nebula at the top is not included but cropped and barely visible. What can be done?

Planned workflow:

1. User opens a completed run in GUI3.
2. Chat panel offers an input field for natural problem description.
3. Backend builds a `pi.run-chat-context.v1` from run status, config revision, report stats, artifacts, phase events, scan metrics, and relevant memories.
4. PI responds in natural user language: likely causes, artifacts to check, concrete next steps.
5. If useful, PI additionally generates a `pi.action-plan.v1` for safe config changes.
6. GUI3 shows response, evidence, artifact links, and optionally PI preview/apply.
7. User can save the chat outcome as a memory candidate if a later run confirms the improvement.

Implementation steps:

- [x] Add GUI3 run chat panel in Run Monitor; history remains without chat elements.
- [x] Implement `/api/pi/run-chat` as a read-only diagnosis endpoint.
- [x] Build run context builder for completed runs: report, artifacts, config/preview context, relevant metrics and memories.
- [x] Translate natural user description into structured problem hints without treating them as hard truth.
- [x] Define response format: `summary`, `likely_causes`, `checks`, `recommendations`, `evidence`, optional `action_plan`.
- [x] Model typical image problems as controlled hints: black star cores, cropped nebula, too-dark nebula portions, background gradient, colour cast, tile patterns, soft stars.
- [x] Connect chat responses with existing PI preview; apply remains deliberately separate and review-required.
- [x] Show last run image in Run History and Run Monitor above the chat.
- [x] Only generate image preview on first page load if no preview exists; further regeneration only via explicit refresh button.
- [x] Support follow-up questions in Run Monitor chat with local chat history.
- [x] PI suggests an appropriate resume start phase; user selects it explicitly.
- [x] Create tests with fixture run and example problems.

Acceptance criteria:

- [x] Chat works without write access and without storing image data in memories.
- [x] Response names concrete run artifacts or report facts as evidence.
- [x] Recommendations can optionally be validated and previewed as an action plan.
- [x] Preview refresh is no longer forced by polling, resume status, or terminal events.
- [x] Run History no longer contains chat controls or run chat action plan elements.
- [x] User text like "stars have black cores" leads to traceable checks instead of blind parameter guessing.

## Phase 9 — Global AI Memory Layer

Goal: PI becomes a professional, globally learning knowledge layer for
astro optimisations. What is learned is not "this project had this config"
but "under this acquisition/object/pipeline context, this strategy was
sensible or not sensible with this evidence".

Principle:

- **Global, not project-specific:** Memories reside in central PI storage and are usable across all runs/projects.
- **Context, not anecdote:** Every memory must describe its professional validity condition.
- **Evidence, not gut feeling:** Every memory needs provenance, outcome information, and review status.
- **Retrieval, not blind prompt appending:** Only matching, bounded, explained memories are given to the AI.
- **Negative experiences are valuable:** Unsuccessful suggestions are stored as counterexamples.

### 9.1 Memory Data Types

New or sharpened memory types:

- `config_optimization`: A concrete parameter strategy was sensible under a certain context. (implemented: generation via `scan_ai_apply` and `run_chat`)
- `artifact_diagnosis`: A visible problem was connected to causes/checks/phases. (implemented: generation via `run_chat`)
- `resume_strategy`: A resume phase was sensible or not sensible for a problem class. (implemented: generation via `run_chat`; `source: resume_feedback` is defined as a value, but no dedicated resume feedback endpoint sets this `source` value explicitly — currently runs via the general outcome endpoint)
- `provider_prompt_pattern`: A prompt/context pattern led to better structured AI responses. (Type defined, but no generation path implemented; **deferred** — only relevant when provider-specific prompt optimisation becomes empirically necessary. Until then, the static prompt structure in `pi.ai-request.v2` suffices.)
- `counterexample`: A recommendation was not helpful despite similar context. (implemented: negative outcome sets `verdict=worse` and `review_recommendation=rejected`; automatic counterexample candidate via `/api/pi/memories/:id/outcome` with `negative_learning=true`)
- `user_preference`: User preferences for display/stretch/detail level, if explicitly confirmed. (Type defined, but no GUI3 element generates this type; **deferred** — requires GUI3 to implement a dedicated "save preference" dialog. Can be added in a later phase with a `/api/pi/memories/preference` endpoint.)

### 9.2 Global Memory Schema `pi.memory.v2`

`pi.memory.v2` is the new starting point. Earlier drafts or previously
stored AI data are not migrated and not compatibly read back. Old files
are ignored for the new memory layer or only considered as manually
exported reference, never as automatically trusted source.

Mandatory fields:

- `schema_version`: `pi.memory.v2`
- `id`: stable ID
- `type`: memory type
- `status`: `candidate`, `promotable`, `accepted`, `rejected`, `deprecated`
- `privacy_class`: e.g. `metadata_only`
- `created_at`, `updated_at`
- `source`: `scan_ai_apply`, `run_chat`, `resume_feedback`, `manual_review`, `outcome_evaluator`
- `summary`: short professional statement
- `recommendation`: structured recommendation or warning
- `context_signature`: normalised context signature
- `scope`: validity range and boundaries
- `evidence`: provenance and supporting evidence
- `outcome`: outcome observation and before/after deltas
- `review`: human review information
- `retrieval`: search/ranking aids

Example structure:

```json
{
  "schema_version": "pi.memory.v2",
  "type": "config_optimization",
  "status": "candidate",
  "privacy_class": "metadata_only",
  "summary": "For extended nebulae with OSC data, use BGE conservatively because faint emission can be removed as background.",
  "context_signature": {
    "target": {
      "object_name": "M42",
      "object_type": "emission_nebula",
      "angular_size_class": "large",
      "has_extended_emission": true
    },
    "acquisition": {
      "camera_type": "OSC",
      "filters": ["dual_narrowband"],
      "exposure_seconds_median": 180,
      "frame_count": 120,
      "total_integration_minutes": 360,
      "calibration": {
        "darks": true,
        "flats": true,
        "bias": false
      }
    },
    "optics": {
      "telescope": "unknown_or_redacted",
      "focal_length_mm": null,
      "f_ratio": null,
      "pixel_scale_arcsec": null
    },
    "mount": {
      "type": "EQ",
      "tracking_quality": "unknown"
    },
    "pipeline": {
      "affected_paths": ["bge.enabled", "stretch.target_background"],
      "phases": ["BGE", "HYPERMETRIC_STRETCH"]
    },
    "quality": {
      "gradient_class": "medium",
      "star_count_class": "high",
      "fwhm_class": "normal"
    }
  },
  "scope": {
    "applies_when": [
      "target has large diffuse emission",
      "background extraction may confuse nebulosity with background"
    ],
    "does_not_apply_when": [
      "compact galaxy target",
      "strong measured gradient dominates the field"
    ],
    "confidence": 0.68
  },
  "recommendation": {
    "action_plan_fragment": {
      "actions": [
        {"type": "config.set", "path": "bge.enabled", "value": false}
      ]
    },
    "explanation": "Try disabling BGE or configuring it more conservatively and re-evaluate stretch."
  },
  "evidence": {
    "run_refs": [
      {"run_id_hash": "sha256:...", "artifact_refs": ["report", "stacked_rgb_hms_preview"]}
    ],
    "human_feedback": "Nebula became more visible after BGE off.",
    "ai_observation": "Preview showed cropped/weak nebula structure."
  },
  "outcome": {
    "before": {"nebula_visibility": "weak", "warnings": ["faint_nebula"]},
    "after": {"nebula_visibility": "improved"},
    "delta": {"user_rating": 1},
    "verified": false
  },
  "review": {
    "reviewed_by": null,
    "reviewed_at": null,
    "notes": ""
  },
  "retrieval": {
    "keywords": ["nebula", "BGE", "extended emission", "OSC"],
    "embedding_text": "extended nebula OSC BGE removes faint emission",
    "negative": false
  }
}
```

Note on `embedding_text`: The field is reserved for later vector search and is currently only stored as human-readable text. The current retrieval uses exclusively the structured `keywords` array and the weighted `context_signature` similarity. As long as no embedding backend is connected, `embedding_text` must not be interpreted as a retrieval signal.

### 9.3 Context Signature for Tile Compile

The context signature is normalised from available sources. Missing values
remain `null` or `unknown` but are not invented.

Sources:

- FITS headers: object name, camera, filter, exposure time, gain, temperature, date, telescope/focal length if available.
- Scan result: frame count, colour mode, Bayer pattern, file groups, errors/warnings.
- Scan metrics: FWHM, star count, background, noise, gradient, roundness, session geometry.
- Config: relevant paths, enabled pipeline phases, BGE/PCC/HMS/AQMH/stacking/registration parameters.
- Run report: phase status, artifacts, warnings, quality metrics.
- User input: mount, object class, camera, calibration, notes.
- Run chat: natural problem description and image observations.

Normalised fields:

- `target.object_name`, `target.object_type`, `target.angular_size_class`, `target.has_extended_emission`
- `acquisition.camera_name`, `acquisition.camera_type`, `acquisition.color_mode`, `acquisition.filters`
- `acquisition.exposure_seconds_min/median/max`, `frame_count`, `total_integration_minutes`
- `acquisition.gain`, `sensor_temperature_c`, `date_range`
- `calibration.darks/flats/bias/dark_flats`, `calibration.quality_warnings`
- `optics.telescope`, `focal_length_mm`, `aperture_mm`, `f_ratio`, `reducer`, `pixel_scale_arcsec`
- `mount.type`, `tracking_quality`, `field_rotation_risk`
- `sky.moon_phase`, `moon_distance`, `bortle`, `transparency` if later available
- `quality.fwhm_class`, `gradient_class`, `noise_class`, `star_count_class`
- `pipeline.phases`, `pipeline.affected_paths`, `pipeline.resume_phase`

### 9.4 Professional AI Context

Scan-AI and Run Chat may no longer send just "recommendations plus a few
memories". The request must be professionally structured:

- `task`: clear objective, e.g. `scan_config_optimization`, `run_quality_diagnosis`, `resume_strategy`.
- `current_context`: normalised context signature.
- `current_evidence`: scan/run/report/artifact data, with image preview if the model supports vision.
- `candidate_memories`: matching positive memories with match explanation.
- `negative_memories`: rejected/deprecated/counterexamples with warning.
- `constraints`: allowed config paths, safety rules, no paths/secrets, no unvalidated writes.
- `required_output_schema`: e.g. `pi.scan-analysis.v2` or `pi.run-chat-answer.v2`.
- `uncertainty_policy`: AI must mark missing data as missing and must not guess.
- `memory_write_policy`: AI may suggest memory candidates but must not automatically accept them.

The AI response must become more professional:

- Recommendations need `rationale`, `evidence_refs`, `expected_effect`, `risk`, `confidence`, `scope`.
- Parameter suggestions need `path`, `current_value`, `suggested_value`, `why_now`, `why_safe`.
- For image questions, it must be clear whether an image was sent and which observation comes from the image.
- For follow-up questions, the previous chat/run context must be included.
- Repeated ineffective suggestions must not be offered again; instead the AI must formulate a different hypothesis or counterexample.

### 9.5 Memory Generation

Memory candidates arise from multiple sources:

- `learn=true` after apply: only generates `candidate`, not yet accepted truth.
- Run outcome evaluator after resume/run: supplements outcome deltas.
- Run chat with user feedback: generates `artifact_diagnosis` or `resume_strategy`.
- User marks "helped" or "did not help": generates positive or negative evidence.
- Repeated success in similar context: candidate becomes `promotable`.

Storage rules:

- No memory without context signature.
- No accepted memory without review.
- No global memory without scope and `does_not_apply_when`.
- No memory with image data; only preview/artifact references and optionally hashes.
- No memory with absolute local paths in knowledge fields.
- Every memory change generates an audit event.

### 9.6 Retrieval and Ranking

Ranking signals:

- Config path overlap.
- Object class and target size.
- Camera/filter/colour mode similarity.
- Pipeline phase and problem class.
- Quality metrics and artifact class.
- Outcome quality and review status.
- Recency and deprecation.
- Counterexamples for similar contexts.

Retrieval result:

- `matches`: accepted memories with score and match fields.
- `warnings`: negative/deprecated memories with reason.
- `coverage`: which context fields are missing and therefore lower confidence.
- `prompt_budget`: limit for AI context.

### 9.7 GUI3 Requirements

- Memory detail view shows context signature, scope, evidence, outcome, review, and retrieval hits.
- During review, the user can edit scope: "applies to nebulae", "not for galaxies", "only OSC", "only dualband".
- Run chat shows which memories the AI used and why.
- AI recommendation shows positive and negative memory hints separately.
- "Learn from this optimisation" becomes more precise: `save learning candidate`, then review/outcome required.
- GUI3 allows global memory search by object, camera, filter, config path, problem class, and status.

### 9.8 Implementation Steps

- [x] Define `pi.memory.v2` schema as new baseline contract; no legacy/draft migration.
- [x] Define store reset behaviour: new global store, old AI/memory files visibly ignored, no automatic adoption.
- [x] Implement `pi.context_signature.v1` builder from Scan-AI context, GUI context, config paths, and scan metrics.
- [x] Extend `pi.context_signature.v1` with deeper FITS header/run report extraction for object, telescope, filter, exposure, and acquisition date.
- [x] Extend memory store with global indexes: `by_type`, `by_status`, `by_path`, `by_target`, `by_camera`, `by_filter`, `by_problem`.
- [x] Implement retrieval service with scoring, accepted-only matches, and negative warnings.
- [x] Build retrieval match explanation and coverage fields for Scan-AI context and store retrieval.
- [x] Extend Scan-AI request with structured positive and negative memory contexts.
- [x] Define common professional context container `pi.ai-request.v2` and embed compatibly in Scan-AI requests.
- [x] Extend Run Chat request compatibly with the same `pi.ai-request.v2` container, including image status and chat history.
- [x] Switch Scan-AI and Run Chat internally fully to `pi.ai-request.v2` as the primary sidecar contract.
- [x] Implement outcome evaluator for run/resume and update memory candidates with before/after deltas.
- [x] Implement negative learning from user feedback and ineffective resume attempts.
- [x] Extend GUI3 memory detail/review view with context, scope, evidence, outcome, and `promotable`.
- [x] Extend export/import for `pi.memory.v2` with privacy filter.
- [x] Create tests for global retrieval cases, scope boundaries, negative memories, and legacy ignore.
- [x] Tests verify that old AI/memory data is not loaded, migrated, or used as retrieval context.
- [x] Implement diversity cap in retrieval: max 2 memories per class with context query, max 3 without. (`apply_diversity_cap()`)
- [x] Pass coverage from retrieval explicitly as prompt section: `retrieval_coverage_summary` with `systemically_missing_context_fields` in `pi.ai-request.v2`. (`pi_ai_request_builder.cpp`)
- [x] Mismatch penalty for actively wrong fields: prevents cross-contamination even with config path overlap. (`pi_memory_store.cpp`)
- [x] Tests verify context cross-contamination: Memory for "M42, nebula, OSC" is not blindly applied to "M104, galaxy, mono/LRGB". (`test_pi_memory_store.cpp`)
- [x] Tests verify that a rejected memory appears as an explicit negative signal in the AI prompt for similar context. (`test_pi_memory_store.cpp`)
- [x] Post-run trigger for automatic outcome capture: `/api/pi/memories/evaluate-run` reads run artifacts and writes outcome data into open candidates. (`pi_routes.cpp`)
- [x] `resume_feedback` as dedicated `source` value: `/api/pi/memories/resume-feedback` sets `source=resume_feedback` and generates `resume_strategy` or `counterexample` memory.
- [x] `provider_prompt_pattern` and `user_preference` memory types documented as **deferred**: no implementation needed until provider-specific prompt optimisation or GUI3 preference dialog becomes necessary.

Acceptance criteria:

- [x] A new global memory from a previous run is found in a different project when object/acquisition/pipeline context matches.
- [x] The same memory is not found or only with low confidence when the context does not professionally match.
- [x] AI prompts contain explicitly positive memories, negative memories, match explanation, and missing context fields.
- [x] Memory candidates contain object/acquisition data if available in Scan-AI/GUI context: object class, camera, frame count, calibration, mount, quality metrics.
- [x] Memory candidates additionally contain deeply extracted FITS fields: object name, telescope, filter, exposure, acquisition date.
- [x] GUI3 allows review with scope adjustment.
- [x] No memory stores raw image data, API keys, or absolute local image paths.
- [x] Tests ensure that accepted memories still do not bypass config validation.
- [x] An empty new PI storage starts deterministically without legacy and without automatic migration.
- [x] Tests verify context cross-contamination: Memory for "M42, nebula, OSC" is not blindly applied to "M104, galaxy, mono/LRGB".
- [x] Tests verify that a rejected memory appears as an explicit negative signal in the AI prompt for similar context.

## Memory Concept in Detail

A memory is a reviewable, globally usable experience, not an automatic
rule and not a project-specific note.

Typical flow:

1. Scan-AI generates recommendations.
2. GUI3 shows recommendations and PI Preview.
3. User applies validated changes.
4. If `learn=true` is set, Tile Compile stores a memory candidate with context signature.
5. After run/resume, outcome data is supplemented: what got better, same, worse, or remained unclear?
6. User reviews the candidate as `accepted`, `rejected`, or `deprecated` and can edit the scope.
7. Later sessions may use matching accepted memories as context but must still validate schema and config.
8. Rejected/deprecated memories are used as counterexamples so the AI does not repeat the same unsuccessful strategy.

Example for `config_optimization`:

- `type`: `config_optimization`
- `source`: `scan_ai_apply`
- `status`: `candidate`
- `privacy_class`: `metadata_only`
- `analysis_id`
- `provenance`: analysis/run/artifact references without absolute local image paths
- `config_updates`
- `context_signature`: object, target type, camera, telescope/optics, filter, exposures, frame count, calibration, mount, quality classes, relevant pipeline phases
- `scope`: when this experience applies and when not
- `summary`
- `confidence`
- `detected_scenarios`
- `warnings`
- `validation`
- `outcome`: before/after deltas, user feedback, report/artifact status
- `review`: status, reviewer, notes, scope changes

Memory quality rules:

- A memory without `context_signature` may not be `accepted`.
- A memory without outcome or user review remains `candidate`.
- A memory with only a single case gets low confidence.
- A memory with contradictory later result becomes `deprecated` or gets a `counterexample`.
- Memories may act globally but only within their professionally described scope.
- The AI never receives memories as commands but as historical evidence with match score and uncertainty.

## Professionalisation of AI Function

The AI function should no longer work like a simple recommendation dialog
but like a structured diagnosis and optimisation assistant.

Mandatory requirements for AI requests:

- Complete context container instead of loose prompt.
- Explicit task: analysis, diagnosis, resume plan, memory evaluation, or outcome evaluation.
- Normalised context signature.
- Relevant positive and negative memories with match justification.
- Current config and allowed config paths.
- Report/artifact/quality data with source references.
- Image preview only if provider/model supports vision; otherwise clear note `image_context=false`.
- Strict response schema with recommendations, evidence, risks, confidence, and optional action plan.

Mandatory requirements for AI responses:

- No assertion without evidence reference or uncertainty marking.
- No repetition of identical parameter recommendations if they have already been tested without improvement in the same run.
- Every parameter recommendation explains expected effect, risk, affected phase, and smallest sensible resume phase.
- For memory suggestions, the AI must include scope, evidence, and possible counterexamples.
- For image problems, the AI must distinguish between "observed in image", "derived from report", and "only user description".

## Next Sensible Step

All phases 0–9 are implemented and tested. The remaining deferred items are not blockers:

**Deferred items (only when needed):**
- `provider_prompt_pattern`: Becomes relevant when provider-specific prompt tuning is empirically necessary.
- `user_preference`: Needs a GUI3 "save preference" dialog and a `/api/pi/memories/preference` endpoint.
- Vector search via `embedding_text`: Reserved for later semantic retrieval extension; currently no embedding backend connected.

**Recommended next work (outside PI architecture):**
- GUI3: Trigger post-run evaluator automatically as client call after run completion event (currently manual via `/api/pi/memories/evaluate-run`).
- GUI3: Offer resume feedback dialog in Run Monitor after resume completion.
- GUI3: Promote button for `promotable` memories in memory review panel.
