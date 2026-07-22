# PI Run-Chat Recommendation Chat: Data and Validation Plan

Status: Draft, 2026-07-17  
Goal: The run-specific PI chat should deliver reliable, schema-valid, and
pipeline-correct recommendations. This covers diagnosis, concrete config
changes, action plans, and resume recommendations.

## Initial Problem

The recommendation chat can currently produce plausible but technically
incorrect recommendations. A concrete example from an M42 run:

- Parameters like `stretch.star_pressure` and `stretch.protect_b` were
  recommended.
- These paths are not effective config paths for `tile_compile`.
  The relevant config block is called `hypermetric_stretch`.
- `star_pressure` is a diagnostic value from HyperMetric Stretch, but
  not a normal config parameter.
- `HYPERMETRIC_STRETCH` was recommended as the resume start, even though
  the likely causes were before HMS: crop, AQMH fallback, stacking,
  normalisation, or BGE.

This is not purely a model problem. An LLM can only reliably recommend
when it knows the technical boundaries:

- which config paths exist,
- which values are allowed,
- which phase is influenced by which parameter,
- which artifacts and diagnostic values belong to which pipeline stage,
- which values are only measurements and cannot be set.

The central design rule is therefore:

> The PIAI may interpret and prioritise. Schema validity,
> action plan validity, and resume phase are deterministically validated
> and computed in the backend.

## Target Vision

The run chat does not produce free advice but a structured result:

1. Diagnosis with evidence references.
2. List of missing evidence if a cause cannot be substantiated.
3. Concrete recommendations with validity range.
4. Schema-valid action plan.
5. Per action a minimal resume phase.
6. A final computed resume recommendation.
7. Warnings for removed or corrected provider suggestions.

The user should then be able to understand:

- why a parameter is recommended,
- which artifacts or comparison values support it,
- which changes are actually applicable,
- from which phase a resume is sensible,
- which assumptions are still uncertain.

## Data Needed Per Run

The chat needs a compact but complete run context. Raw data does not
need to be sent to the PIAI. Structured summaries, diagnostic values,
and image context are important.

Recommended container:

```json
{
  "schema_version": "pi.run-recommendation-context.v1",
  "run_id": "M42-pi_20260717_154431",
  "run_context": {
    "target": {},
    "status": {},
    "phase_status": [],
    "phase_order": []
  },
  "config": {
    "effective": {},
    "raw_path": "redacted-or-relative",
    "schema_summary": {},
    "path_phase_map": {},
    "diagnostic_only_fields": {}
  },
  "artifacts": {},
  "diagnostics": {},
  "image_context": {},
  "comparison_runs": [],
  "previous_turns": [],
  "memories": {}
}
```

### Rationale

A recommendation chat is only robust if it does not only see the final image.
Many visible errors originate earlier in the pipeline and are only made
visible by later phases. Without phase and artifact context, the
recommendation lands too quickly at stretching because stretching is close
to the final image.

## Effective Config

The PIAI needs the effective config, not just the raw YAML text.

Required:

- current values after defaults and normalisation,
- only non-sensitive values,
- stable paths in dot-path format,
- marking of unknown or ignored paths.

Example:

```json
{
  "config": {
    "effective": {
      "normalization.mode": "background",
      "output.crop_to_nonzero_bbox": true,
      "tile.min_size": 64,
      "tile.overlap_fraction": 0.25,
      "hypermetric_stretch.protect_b": 6.0
    },
    "unknown_or_ignored_paths": [
      {
        "path": "stretch.star_pressure",
        "reason": "not present in tile_compile schema; ignored by pipeline"
      }
    ]
  }
}
```

### Rationale

If the chat only sees raw YAML, it can treat ignored or wrong paths as
effective. This is exactly how recommendations like
`stretch.star_pressure` arise. The effective config must be the authority.

## Schema Summary for the PIAI

The PIAI does not need a complete schema file with all descriptions.
It needs a compact, machine-readable summary of allowed recommendation
paths.

Example:

```json
{
  "schema_summary": {
    "valid_config_paths": {
      "hypermetric_stretch.protect_b": {
        "type": "number",
        "minimum": 0.1,
        "maximum": 5.0,
        "default": 6.0,
        "recommendation_allowed": true,
        "min_resume_phase": "HYPERMETRIC_STRETCH",
        "description": "B-channel protection during HMS"
      },
      "normalization.mode": {
        "type": "string",
        "enum": ["median", "background"],
        "recommendation_allowed": true,
        "min_resume_phase": "NORMALIZATION"
      },
      "output.crop_to_nonzero_bbox": {
        "type": "boolean",
        "recommendation_allowed": true,
        "min_resume_phase": "COMMON_OVERLAP"
      }
    },
    "invalid_or_diagnostic_paths": {
      "stretch.*": {
        "kind": "invalid",
        "reason": "No effective tile_compile config group named stretch for HMS tuning."
      },
      "star_pressure": {
        "kind": "diagnostic_only",
        "reason": "Estimated by HMS diagnostics; not a configurable input."
      }
    }
  }
}
```

### Rationale

LLMs are good at generating plausible parameter names. That is a risk here.
The prompt must therefore explicitly say: Only paths from
`valid_config_paths` may go into the action plan. Everything else belongs
at most as diagnosis or warning in free text.

## Phase and Resume Mapping

Every recommendable config path needs a minimal resume phase.

Example:

```json
{
  "path_phase_map": {
    "data.*": "INPUT_SCAN",
    "calibration.*": "CALIBRATION",
    "registration.*": "REGISTRATION",
    "quality_filter.*": "QUALITY_FILTER",
    "normalization.*": "NORMALIZATION",
    "stacking.*": "STACKING",
    "output.crop_to_nonzero_bbox": "COMMON_OVERLAP",
    "tile.*": "TILE_RECONSTRUCTION",
    "aqmh.pyramid.*": "AQMH_MAPS",
    "aqmh.storage.*": "AQMH_MAPS",
    "aqmh.global_quality.*": "AQMH_GLOBAL_QUALITY",
    "aqmh.reconstruction.*": "AQMH_RECONSTRUCTION",
    "bge.*": "BGE",
    "pcc.*": "PCC",
    "hypermetric_stretch.*": "HYPERMETRIC_STRETCH"
  }
}
```

The final resume phase is not taken from the model. It is computed from all
validated actions:

1. Remove invalid actions.
2. Determine the minimal resume phase for each action.
3. Select the earliest phase according to pipeline order.
4. If no action is present, deliver only a diagnostic recommendation
   with low confidence.

Example:

```json
{
  "actions": [
    {"path": "hypermetric_stretch.protect_b", "value": 3.5},
    {"path": "output.crop_to_nonzero_bbox", "value": false},
    {"path": "normalization.mode", "value": "median"}
  ],
  "computed_resume_phase": "COMMON_OVERLAP"
}
```

### Rationale

When multiple parameters are changed, the latest visible phase is not
decisive. The earliest affected pipeline stage is decisive.
An HMS resume recommendation would be wrong as soon as a change to crop,
normalisation, stacking, AQMH, BGE, or PCC is involved.

Important: AQMH is not a single resume stage. The runner knows at least
these AQMH-related phases:

- `AQMH_MAPS`: Recompute quality maps.
- `AQMH_GLOBAL_QUALITY`: Re-evaluate global weights from existing maps.
- `AQMH_RECONSTRUCTION`: Reconstruct from existing prewarped frames,
  existing AQMH maps, and existing canvas mask.
- `AQMH_DIAGNOSTICS`: Diagnose the reconstructed output.

`AQMH_RECONSTRUCTION` is therefore only the correct start phase when the
changed parameters exclusively affect reconstruction from already valid
AQMH maps. If maps, pyramid settings, storage resolution,
canvas/crop, prewarp, registration, or common overlap are affected,
`AQMH_RECONSTRUCTION` is too late.

## Run Artifacts and Diagnostic Values

The chat needs a structured summary per pipeline stage.

### Crop and Canvas

Required:

```json
{
  "crop": {
    "crop_to_nonzero_bbox": true,
    "crop_x": 448,
    "crop_y": 588,
    "output_width": 3858,
    "output_height": 2194,
    "canvas_width": 4754,
    "canvas_height": 3370
  }
}
```

Rationale: When nebula is missing at the edge or top, crop is an early and
frequent prime suspect. Without these values, the chat can falsely attribute
the problem to stretching.

### AQMH

Required:

```json
{
  "aqmh": {
    "enabled": true,
    "fallback_to_uniform_control": true,
    "uniform_control_blend_accepted": false,
    "uniform_control_blend_alpha": 0.0,
    "background_rms_regression": 637.23,
    "structure_masked_detail_applied": false
  }
}
```

Rationale: AQMH fallbacks change the local reconstruction. If a
nebula region appears weak or flat, the chat must know whether the local
reconstruction was accepted or rejected.

### BGE

Required:

```json
{
  "bge": {
    "attempted": true,
    "applied": false,
    "success": false,
    "skip_reason": "background_chroma_worsened",
    "method": "autobge"
  }
}
```

Rationale: BGE can weaken real nebula but also improve gradients.
A blanket recommendation `bge.enabled=false` is only sensible
if the run context shows that BGE was problematic or the comparison run
without BGE was better.

### PCC

Required:

```json
{
  "pcc": {
    "success": true,
    "stars_matched": 343,
    "stars_used": 305,
    "residual_rms": 0.3559,
    "condition_number": 1.701,
    "matrix_diagonal": [1.0689, 1.0, 1.7012]
  }
}
```

Rationale: PCC parameters must not be set from general rules.
If a good comparison run had residuals around 0.32, a recommendation like
`pcc.max_residual_rms=0.05` is obviously too strict.

### Stacking and Rejection

Required:

```json
{
  "stacking": {
    "method": "rej",
    "sigma_low": 3.0,
    "sigma_high": 3.0,
    "cosmetic_correction_enabled": true,
    "cosmetic_correction_sigma": 10.0,
    "per_frame_cosmetic_correction_sigma": 5.0,
    "valid_mask_fraction": 0.94
  }
}
```

Rationale: Black star cores can be caused by rejection or cosmetic
correction. Without stacking context, `HYPERMETRIC_STRETCH` as a
resume phase is just guessing.

### HyperMetric Stretch

Required:

```json
{
  "hypermetric_stretch": {
    "enabled": true,
    "input_stage": "pcc",
    "protect_b": 6.0,
    "star_pressure": 0.745,
    "black_clip_percent": 0.0197,
    "white_clip_percent": 0.0160,
    "log_d": 3.951
  }
}
```

Rationale: HMS diagnostics are important but not automatically causal.
`star_pressure` may be used as evidence but not as a config target.
HMS resume is only correct when the changes exclusively affect HMS
or a pre-HMS artifact comparison shows that the problem only arises in HMS.

## Comparison Run Context

When the user mentions a good or bad comparison run, the chat must
receive a structured diff.

Example:

```json
{
  "comparison_runs": [
    {
      "run_id": "m42_20260703_083337",
      "role": "better_reference",
      "config_diff": {
        "normalization.mode": {
          "current": "background",
          "reference": "median"
        },
        "tile.min_size": {
          "current": 64,
          "reference": 48
        },
        "tile.overlap_fraction": {
          "current": 0.25,
          "reference": 0.4
        }
      },
      "diagnostic_diff": {
        "crop_y": {
          "current": 588,
          "reference": 8
        },
        "hypermetric_stretch.star_pressure": {
          "current": 0.745,
          "reference": 0.760
        }
      }
    }
  ]
}
```

### Rationale

A comparison run is stronger than general astrophotography heuristics. In
the M42 example, the comparison run refutes the simple thesis
"high star_pressure causes black cores" because the better run had a
higher `star_pressure`.

## Image Data for the PIAI

The PIAI should not only receive a final preview. Small, targeted image
contexts are sensible:

- final preview,
- comparison run preview,
- crop of the conspicuous star core region,
- crop of the weak nebula region,
- optional pre-HMS preview,
- optional post-PCC/pre-HMS preview,
- optional BGE preview,
- optional AQMH reconstruction preview.

Each image needs metadata:

```json
{
  "image_id": "final_preview",
  "stage": "HYPERMETRIC_STRETCH",
  "width": 3858,
  "height": 2194,
  "crop_x": 448,
  "crop_y": 588,
  "source_artifact": "outputs/stacked_rgb_hms.png"
}
```

### Rationale

A final image can show what looks wrong but not when it went wrong.
Pre-HMS and intermediate stage images separate display errors from
pipeline errors.

## Prompt Contract

The provider prompt must contain hard rules.

Mandatory rules:

- Answer exactly as a JSON object.
- Only use paths from `schema_summary.valid_config_paths` for
  `action_plan.actions`.
- Do not generate actions for paths from `invalid_or_diagnostic_paths`.
- `star_pressure` is diagnostic only, not a config path.
- `stretch.*` is invalid.
- Every concrete recommendation with a value must be encoded as an action.
- Every action must include `evidence_ref` and `min_resume_phase`.
- If evidence is missing, write it in `missing_evidence` instead of
  asserting a cause.
- Only recommend HMS as resume when all validated actions concern
  `hypermetric_stretch.*` or a pre-HMS comparison proves that the
  problem only arises in HMS.
- Comparison run diffs take precedence over general heuristics.

Example prompt section:

```text
CONFIG VALIDITY RULES:
- You may only place paths from valid_config_paths into action_plan.actions.
- Never use stretch.*.
- Never use star_pressure as an action path. It is diagnostic-only.
- If you mention a diagnostic-only field, mark it as evidence, not as a setting.

RESUME RULES:
- For every action, copy min_resume_phase from valid_config_paths.
- Do not choose HYPERMETRIC_STRETCH if any action requires an earlier phase.
- If the observed problem may originate before HMS and no pre-HMS image proves
  otherwise, prefer the earliest plausible diagnostic phase.
```

## PIAI Response Schema

Recommended schema:

```json
{
  "schema_version": "pi.run-chat-answer.v1",
  "summary": "string",
  "diagnosis": [
    {
      "text": "string",
      "confidence": "low|medium|high",
      "evidence_ref": "string"
    }
  ],
  "missing_evidence": [
    {
      "text": "string",
      "would_disambiguate": "string"
    }
  ],
  "recommendations": [
    {
      "text": "string",
      "confidence": "low|medium|high",
      "evidence_ref": "string"
    }
  ],
  "action_plan": {
    "schema_version": "pi.action-plan.v1",
    "source": "pi.run-chat.provider",
    "mutation_free": true,
    "actions": [
      {
        "id": "string",
        "type": "config.set",
        "path": "string",
        "value": "any",
        "min_resume_phase": "string",
        "rationale": "string",
        "evidence_ref": "string",
        "confidence": "low|medium|high"
      }
    ]
  },
  "resume_recommendation": {
    "from_phase": "string",
    "confidence": "low|medium|high",
    "reason": "string"
  },
  "warnings": []
}
```

### Rationale

`recommendations` may be explanatory. `action_plan.actions` must be
executable. This separation prevents vague hints from directly becoming
config mutations.

## Backend Validation After Provider Response

After the provider response, the backend must deterministically validate.

Algorithm:

```text
parse provider JSON
validate response schema shape

for each action in action_plan.actions:
  require type == "config.set"
  require path in valid_config_paths
  require value matches type, enum, min, max
  require path is not diagnostic-only
  attach canonical min_resume_phase from backend map

drop invalid actions
record validation warnings for dropped actions

computed_resume_phase = earliest(min_resume_phase of remaining actions)

if provider resume phase is later than computed_resume_phase:
  override with computed_resume_phase
  add warning

if no valid actions remain:
  mark action_plan as diagnostic_only
  do not offer one-click apply
```

### Rationale

The provider can make mistakes despite the prompt. The safety boundary must
not be in the prompt but in the backend.

## Resume Correction

The resume recommendation must be computed from validated actions.

Example:

```json
{
  "provider_resume": "HYPERMETRIC_STRETCH",
  "valid_actions": [
    {
      "path": "output.crop_to_nonzero_bbox",
      "value": false,
      "min_resume_phase": "COMMON_OVERLAP"
    },
    {
      "path": "normalization.mode",
      "value": "median",
      "min_resume_phase": "NORMALIZATION"
    }
  ],
  "computed_resume": "COMMON_OVERLAP",
  "warning": "Provider resume phase HYPERMETRIC_STRETCH is too late for the validated actions."
}
```

### Rationale

A resume phase that is too late is dangerous because it promises the user a
fast recalculation that does not actually make the real change effective.
The result then looks unchanged, and PI may learn false counterexamples.

## Memory Rules

PI memory should not only store successful recommendations but also
counterexamples.

Example:

```json
{
  "schema_version": "pi.memory.v2",
  "type": "counterexample",
  "source": "run_chat_feedback",
  "problem": {
    "classes": ["black_star_cores", "faint_nebula", "cropped_nebula"]
  },
  "bad_recommendation": {
    "actions": [
      {"path": "stretch.star_pressure", "value": 0.4}
    ],
    "resume_phase": "HYPERMETRIC_STRETCH"
  },
  "why_wrong": [
    "stretch.star_pressure is not schema-valid",
    "star_pressure is diagnostic-only",
    "comparison run had higher star_pressure but better visual result",
    "crop and AQMH diagnostics indicated earlier pipeline cause"
  ],
  "better_rule": "For this symptom set, inspect crop, AQMH, normalization and stacking before HMS."
}
```

### Rationale

Without negative memories, the chat repeats plausible errors. Run chat
problems in particular benefit from counterexamples because they often
arise from visual misinterpretation.

## M42 Example as Expected Behaviour

When the user reports:

- stars in the centre are black,
- nebula at top barely visible,
- comparison run looks better,

then the chat should prioritise the following data:

1. Crop diff:
   - bad run: `crop_y=588`
   - good run: `crop_y=8`
2. AQMH diagnostics:
   - bad run: fallback to uniform control
3. Config diff:
   - bad run: `normalization.mode=background`
   - good run: `normalization.mode=median`
   - bad run: `tile.min_size=64`, `overlap_fraction=0.25`
   - good run: `tile.min_size=48`, `overlap_fraction=0.4`
4. HMS diagnostics:
   - bad run: `star_pressure≈0.745`
   - good run: `star_pressure≈0.760`

Correct conclusion:

- `star_pressure` is not the main evidence.
- `stretch.star_pressure` must not be recommended.
- HMS resume is too late for the main correction.
- First sensible corrections concern crop, normalisation, tile/AQMH, or
  stacking.

Example of a valid action plan:

```json
{
  "actions": [
    {
      "type": "config.set",
      "path": "output.crop_to_nonzero_bbox",
      "value": false,
      "min_resume_phase": "COMMON_OVERLAP",
      "rationale": "The bad run cropped away much more top canvas than the reference run."
    },
    {
      "type": "config.set",
      "path": "normalization.mode",
      "value": "median",
      "min_resume_phase": "NORMALIZATION",
      "rationale": "The better reference run used median normalization."
    },
    {
      "type": "config.set",
      "path": "tile.overlap_fraction",
      "value": 0.4,
      "min_resume_phase": "TILE_RECONSTRUCTION",
      "rationale": "The better reference run used more overlap."
    }
  ],
  "computed_resume_phase": "COMMON_OVERLAP"
}
```

Optional HMS A/B test:

```json
{
  "type": "config.set",
  "path": "hypermetric_stretch.protect_b",
  "value": 3.5,
  "min_resume_phase": "HYPERMETRIC_STRETCH",
  "rationale": "Fast display-only A/B test, not the main correction."
}
```

This HMS test must not override the main resume recommendation when
other actions concern earlier phases.

## Implementation Plan

### Phase 1: Generate Schema and Phase Context

- Create backend function `build_run_recommendation_schema_summary()`.
- Generate a list of allowed dot-paths from the config schema.
- Include type, enum, min/max, and default per path.
- Store minimal resume phase per path.
- Explicitly mark diagnostic fields like `star_pressure` as not settable.
- Mark known invalid aliases like `stretch.*`.

Acceptance criteria:

- `hypermetric_stretch.protect_b` is allowed.
- `stretch.protect_b` is forbidden.
- `stretch.star_pressure` is forbidden.
- `star_pressure` is diagnostic-only.

### Phase 2: Extend Run Context

- Extend `run.report.summary` and `run.artifacts.summary` with compact
  diagnostic areas.
- Normalise crop, AQMH, BGE, PCC, stacking, and HMS diagnostics.
- Deliver effective config with unknown/ignored paths.
- Generate comparison run diffs when a comparison run is named or detected.

Acceptance criteria:

- M42 context contains `crop_y`, AQMH fallback, BGE skip, PCC stats, and
  HMS diagnostics.
- Comparison run diff shows concrete parameter and diagnostic differences.

### Phase 3: Harden Prompt

- Extend `build_provider_run_chat_prompt()` with schema and resume rules.
- Include `schema_summary`, `path_phase_map`, and `diagnostic_only_fields`
  in the AI request.
- Add prompt rules for `stretch.*`, `star_pressure`, and HMS resume.
- Require provider to explicitly name missing evidence.

Acceptance criteria:

- Prompt contains a machine-readable list of allowed paths.
- Prompt forbids `stretch.*`.
- Prompt requires `min_resume_phase` per action.

### Phase 4: Validate Provider Action Plan

- After provider response, check every action path against `valid_config_paths`.
- Check values against type, enum, min/max.
- Remove invalid actions and report as warnings.
- Override `min_resume_phase` from backend mapping.
- Do not allow one-click apply when no valid actions remain.

Acceptance criteria:

- Provider response with `stretch.star_pressure` is not applicable.
- Provider response with `hypermetric_stretch.protect_b` remains valid.
- Invalid actions appear in `warnings` or `rejected_actions`.

### Phase 5: Deterministically Compute Resume

- Compute the earliest phase from all validated actions.
- Use provider resume only as a hint.
- When provider resume is too late, override with backend resume.
- When action plan is empty, do not output a seemingly safe resume phase.

Acceptance criteria:

- `output.crop_to_nonzero_bbox=false` does not lead to `HYPERMETRIC_STRETCH`.
- Combination of `normalization.mode` and `hypermetric_stretch.protect_b`
  leads to `NORMALIZATION`.
- Combination of `output.crop_to_nonzero_bbox` and HMS change leads to
  `COMMON_OVERLAP`.
- AQMH reconstruction parameters lead to `AQMH_RECONSTRUCTION`, but
  AQMH map/pyramid changes lead to `AQMH_MAPS` or earlier.

### Phase 5b: Make Resume Errors Clearly Visible

The runner must not abort after `resume_start` without `resume_end`. For
every resume phase:

- write `resume_end success=false` on early validation errors,
- if a phase has already started, also write `phase_end status=error`,
- mirror `stderr` errors as structured event with `reason`,
- UI state must not get stuck in a running resume.

Specifically for `AQMH_RECONSTRUCTION`, these errors must be logged
structurally:

- missing `artifacts/aqmh_metrics.json`,
- missing `cache/aqmh/aqmh_cache.json`,
- invalid AQMH metadata,
- missing or dimensionally wrong `outputs/canvas_mask.fits`,
- missing `cache/aqmh_masks` when rebuilding the full-canvas mask,
- missing `cache/prewarped_frames` frames,
- error persisting `reconstructed_L.fit` or `synthetic_0.fit`,
- errors in `AQMH_DIAGNOSTICS`.

Rationale: Without `resume_end`, the run looks like it is running or
hanging. The recommendation chat cannot then learn that the resume phase
was not practical.

### Phase 6: Memory and Feedback

- Store counterexample memories for rejected or marked-wrong
  recommendations.
- Store resume feedback with run context and phase.
- Only generate positive memories from successful, validated, and
  user-confirmed results.

Acceptance criteria:

- Wrong recommendation `stretch.star_pressure` is learnable as a counterexample.
- Later prompts receive relevant negative memories.
- Memories contain no absolute raw image paths and no secrets.

### Phase 7: Tests

Minimally required tests:

- `run_chat_rejects_invalid_stretch_paths`
- `run_chat_accepts_hypermetric_stretch_paths`
- `run_chat_treats_star_pressure_as_diagnostic_only`
- `run_chat_computes_resume_from_valid_actions`
- `run_chat_overrides_provider_hms_resume_when_actions_need_earlier_phase`
- `run_chat_keeps_hms_resume_for_hms_only_action`
- `run_chat_m42_context_prefers_crop_or_stacking_over_hms`
- `run_chat_records_counterexample_for_rejected_provider_action`

### Rationale

These tests cover exactly the error class that became visible in the M42
run: valid-sounding wrong paths and too-late resume phases.

## Security and Privacy Rules

- No API keys, tokens, or local secrets in prompts.
- No absolute raw image paths in memories.
- Send images only as reduced previews or crops.
- Config paths and diagnostic values may be sent.
- Never write provider responses directly.
- All write actions remain preview- and apply-required.

## Summary

A functioning PI recommendation chat needs three layers:

1. Good evidence: run artifacts, diagnostic values, images, and comparison run diffs.
2. Hard boundaries: schema-valid paths, diagnostic-only fields, and
   phase mapping.
3. Deterministic post-processing: action plan validation and
   resume computation in the backend.

This prevents the PIAI from recommending pseudo-technical parameters or
choosing a resume phase that is too late. The AI remains responsible for
interpretation and prioritisation; the pipeline rules remain in the code.
