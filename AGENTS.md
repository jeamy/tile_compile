# AGENTS.md

This file applies to the entire repository. More specific `AGENTS.md` files in
subdirectories may add or override rules for their subtree.

## Non-Negotiable Rules

- Never start `tile_compile_web_backend`, `start_backend.sh`, Docker GUI/backend
  services, or any equivalent backend process unless the user explicitly asks
  for it in the current request.
- Never start or resume an image-processing run unless the user explicitly asks
  for that run operation. Builds and tests are not run operations.
- Treat `runs/` and external run directories as user data. Analysis is read-only
  by default. Do not rewrite artifacts, configs, caches, logs, or outputs in an
  existing run unless explicitly requested.
- Do not revert unrelated worktree changes. Work with existing edits and keep
  changes scoped to the request.
- Do not use destructive Git or filesystem commands without explicit approval.

## Running Services

- Assume `tile_compile_web_backend` is always running and available at
  `http://127.0.0.1:8080/ui` (Crow server: `http://127.0.0.1:8080`).
- Assume the `tile_compile_pi_agent` sidecar is always running and listening at
  `http://127.0.0.1:3001`.
- Use these existing services when needed; do not start additional backend or
  sidecar processes.

## Terminal Output

**Alle Terminalkommandos nach `/tmp/out*.txt` umleiten und das Ergebnis aus
dieser Datei lesen!**

Apply this rule as follows:

1. Redirect both stdout and stderr of every diagnostic, build, test, search, or
   execution command to a uniquely named `/tmp/out_<purpose>.txt` file.
2. Inspect the result in a separate read-back command from that file. The
   read-back command is the only exception to the redirection requirement.
3. Do not rely on direct tool output from the original command.
4. Use a new descriptive output file for unrelated commands so results cannot be
   confused with stale output.

Example:

```bash
cmake --build build --target tests -j2 > /tmp/out_cpp_build.txt 2>&1
sed -n '1,240p' /tmp/out_cpp_build.txt
```

For long-running commands, keep the command attached until it exits, then read
the complete output file. Do not leave build, test, runner, or server sessions
running when finishing a task.

## Repository Map

- `tile_compile_cpp/`: C++20 image-processing library, CLI, runner, schemas,
  examples, and Catch2 tests.
- `web_backend_cpp/`: separate C++ HTTP backend and backend contract tests.
- `web_frontend_v3/`: active static frontend (`index.html`, CSS, JavaScript,
  i18n JSON). It has no npm build step.
- `web_frontend/`: legacy/static frontend. Do not update it automatically when
  changing v3 unless the behavior is shared or the user requests parity.
- `agent_service/`: optional Node.js PI AI sidecar.
- `docs/`: methodology, configuration references, process descriptions, and
  practical examples.
- `runs/`: run data and generated artifacts; not source code.

## Architecture Boundaries

- AQMH and Classic Tile Compile are independent reconstruction methods. Do not
  feed Classic local/tile quality metrics into AQMH weights.
- Shared scan, calibration, registration, prewarp, normalization, masks, logging,
  and run-management infrastructure may be reused by both methods.
- AQMH post-processing candidates must be validated against both the uniform
  control and the immutable raw AQMH baseline. If no candidate passes, preserve
  raw AQMH.
- Comparative star-tail and elongation metrics must use matched star positions,
  not independently detected candidate/control populations.
- Preserve resume contracts and phase artifacts when changing pipeline phases.
  A phase must not claim resumability unless every required predecessor artifact
  and cache is present and validated.
- GPU paths must preserve CPU semantics within documented tolerances and retain a
  tested CPU fallback. Do not assume CUDA is absent merely because a sandboxed
  process cannot see the device.

## C++ Build And Tests

Use the existing build tree when compatible. Configure from `tile_compile_cpp/`:

```bash
cmake -S . -B build -DBUILD_TESTS=ON > /tmp/out_cpp_configure.txt 2>&1
cmake --build build --target tile_compile_runner tests -j2 > /tmp/out_cpp_build.txt 2>&1
./build/tests > /tmp/out_cpp_tests.txt 2>&1
```

- For a narrow change, run the relevant Catch2 filter first, then the complete
  suite when practical.
- Build `tile_compile_runner` when modifying files under `tile_compile_cpp/apps/`;
  the library-only test target does not compile every runner phase.
- Run native CUDA tests with actual GPU access when CUDA behavior changed. Keep
  environment/sandbox failures distinct from code failures.
- Backend tests may build and invoke backend fixtures. They must not be confused
  with permission to start a persistent backend service.

## Frontend Work

- Keep tabs visually and behaviorally consistent across all pages. Reuse shared
  tab classes/components instead of page-specific button styling.
- Tabs must look like tabs, not ordinary action buttons. Preserve selected,
  hover, focus-visible, disabled, and responsive states.
- Run-monitor state must be restored from backend/run artifacts after reload,
  navigation, and history activation. Do not rely solely on transient browser
  state for phase or log displays.
- Update both `web_frontend_v3/i18n/de.json` and `en.json` for user-visible text.
  Keep report translations in `report_de.json` and `report_en.json` aligned when
  report labels change.
- Use existing CSS tokens and shared components. Avoid one-off inline styles.
- Verify frontend changes at desktop and mobile sizes. Do not start the backend
  for this verification unless explicitly requested; use static fixtures or an
  already-running service.

## Configuration And Documentation

Configuration defaults must agree across:

- C++ configuration structs;
- parser serialization and validation;
- `tile_compile_cpp/tile_compile.schema.json`;
- `tile_compile_cpp/tile_compile.schema.yaml`;
- `tile_compile_cpp/tile_compile.yaml` where the value is explicitly present;
- relevant tests and example profiles.

When a parameter is added, modified, renamed, or removed, follow
`.devin/skills/update-param-doc/SKILL.md`. Update at least:

- `docs/configuration_reference.md` and `configuration_reference_en.md`;
- `docs/configuration_examples_practical_de.md` and
  `configuration_examples_practical_en.md`;
- applicable files under `tile_compile_cpp/examples/` and their `README.md`;
- active methodology documents when semantics or invariants changed.

Historical documents under an `attic/` directory and explicitly versioned older
methodologies are records. Do not silently rewrite them to current defaults.

Validate JSON and YAML after edits. Parameter documentation must explain units,
ranges, defaults, interactions, fallback behavior, and whether a metric can be
non-applicable.

## Coding Standards

- Follow existing C++20, JavaScript, CSS, and documentation conventions.
- Prefer existing helpers and structured parsers over ad hoc string handling.
- Keep edits narrow. Add an abstraction only when it removes real duplication or
  matches an established local pattern.
- Add tests for behavior changes and regression-prone contracts. Tests should
  assert externally relevant behavior, not only implementation details.
- Use concise comments only where intent or a non-obvious invariant needs to be
  preserved.
- Keep source files ASCII unless the file already uses Unicode or localized text
  requires it.

## Run Analysis

- Compare runs using their effective config, phase events, metrics, validation
  artifacts, and final output measurements. Do not infer causality from final
  FWHM alone.
- Distinguish raw AQMH, neutralized, structure-masked, blended, and final selected
  candidates. Report which gate selected or rejected each candidate.
- Recommendations must be object-agnostic by default. Object-specific overrides
  require evidence and must not weaken global safety invariants.
- State clearly when conclusions are inferred from artifacts rather than verified
  by a new run.

## Completion Checklist

- Relevant code, schemas, examples, and docs agree.
- Formatting and syntax checks pass.
- Relevant tests and required executable targets build successfully.
- No backend or run was started without an explicit request.
- No required terminal session remains active.
- Final response summarizes changed behavior, verification, and any residual risk.
