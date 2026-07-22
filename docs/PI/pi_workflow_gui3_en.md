# PI Workflow in GUI3

Status: 2026-07-14

## Recommendation to Apply

1. Open the AI Recommendation page in GUI3.
2. Generate a Scan-AI analysis or load an existing analysis.
3. Select recommendations.
4. Run `PI Preview`.
5. Review the YAML diff and validation.
6. Execute `Apply PI` only if the preview is plausible.

PI never writes directly. Configuration changes go through Action-Plan, Preview, `validate-config`, and explicit Apply.

## Learning with Memories

- `Learn from this optimisation` stores a memory candidate after a successful apply.
- Candidates are not automatically trusted.
- A memory becomes relevant for future sessions only after review as `accepted`.
- `rejected` and `deprecated` memories are used as negative signals.

## Review

In `PI Memories`:

- `Accept`: approve a good optimisation for future sessions.
- `Reject`: reject a false or inappropriate optimisation.
- `Deprecate`: mark a previously useful optimisation that is now outdated or degrading.

Outcome fields show which paths were applied and whether config validation succeeded.

## Audit

`PI Audit` shows:

- PI Action-Plan Applies
- Scan-AI Config Applies
- Memory Reviews

The view is read-only and serves traceability.

## Export, Import, Dedupe

- `Export` saves a `pi.memories-export.v1` JSON bundle with metadata-only memories and reviews.
- `Import` reads such a bundle and skips duplicates.
- `Dedupe` removes existing duplicate memory entries by signature and creates a backend backup.

## Safety

- Accepted memories are historical context, not rules.
- Every new recommendation must still pass schema, current evidence, and `validate-config`.
- PI tools are read-only/mutation-free. Write actions run exclusively through Action-Plan, Preview, and Apply.
