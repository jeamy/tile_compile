# XYONA Lab - AI Integration Architecture

**Last Updated:** 2026-07-13
**Status:** implemented v2 product/safety baseline; outer runtime cutover governed by active roadmap
**Scope:** LLM-backed Lab assistant, validated graph actions, verified autonomous
agent sessions, agent gateway (MCP), privacy gates, evals, and the audio-ML
extension track  
**Primary target:** Phases 0-6; Phase 0 is unblocked (operator module
naming/descriptor refactor is merged)

---

## Current Runtime Authority

The implemented Action Bus, `ToolRegistry`, privacy, validation, preview/apply,
undo/rollback, verification, journals, grants, telemetry, MCP contract core,
and UI described here remain the product baseline.

The provider/session target in this document is no longer the future runtime
authority. The current code still contains `AiProvider`, `AnthropicProvider`,
and C++-owned conversation/tool loops, but the reviewed clean-break target is
one optional Pi SDK sidecar. Pi will own the outer conversation/provider/tool
loop; XYONA will keep all product and mutation authority. The current execution
source is
[`ROADMAP_AI_INTEGRATION.md`](../../../docs/roadmaps/active/AI_INTEGRATION/ROADMAP_AI_INTEGRATION.md),
with the frozen product boundary in
[`SPEC_PIR_0_RUNTIME_CONTRACT.md`](../../../docs/roadmaps/active/AI_INTEGRATION/SPEC_PIR_0_RUNTIME_CONTRACT.md). The
old direct-provider roadmap is historical under
`docs/roadmaps/done/AI_INTEGRATION/` in the workspace repository.

Until that roadmap is implemented, later sections naming the direct provider and
old loop describe the current as-built baseline, not a second supported target
and not a compatibility requirement.

---

## Overview

XYONA Lab will get a high-end AI integration in three capability tiers that
build on one shared foundation:

1. **Assistant (read/explain):** answers questions about the current graph,
   operators, diagnostics, and signal flow. Read-only.
2. **Copilot (propose/preview/apply):** proposes validated single- and
   multi-action patches that the user previews and accepts as one undo
   transaction.
3. **Agent (verified autonomy):** works on a goal independently inside the
   system - plans, patches, verifies its own result in a sandbox (and
   optionally by offline render + analysis), iterates, and journals every step.
   Autonomy is explicitly granted, capability-gated, budgeted, and always
   reversible.

The foundation is not the cloud provider. The foundation is a local,
deterministic **tool and action surface** that lets any model - today's or a
future one - read, explain, diagnose, propose, and (when granted) apply changes
without ever owning project mutation. Providers, models, and even frontends
(embedded chat, MCP clients, headless automation) are replaceable. The XYONA
tool surface, descriptors, and validation path are not. That is what makes the
integration future-proof.

The first production-quality milestone is unchanged from v1:

```text
MockProvider fixture
    -> ActionPlan JSON (versioned schema)
    -> ActionPlanValidator
    -> existing canvas UndoableAction commands
    -> GraphBuilder preflight in GraphMutationSandbox
    -> apply as one Undo transaction
    -> Undo restores the pre-state
```

No LLM provider is integrated before this loop works without network calls.
No autonomy tier ships before the tier below it has eval baselines.

---

## Non-Negotiable Principles

### 1. AI Never Mutates Project State Directly

AI output is data. It produces typed, schema-versioned action plans with
rationales and confidence scores. Lab validates those plans and applies them
through the existing command path.

Canonical mutation surface:

- `src/app/lab/canvas/commands/CreateNodeCommand.*`
- `src/app/lab/canvas/commands/CreateCoreNodeCommand.*`
- `src/app/lab/canvas/commands/DeleteNodeCommand.*`
- `src/app/lab/canvas/commands/CreateConnectionCommand.*`
- `src/app/lab/canvas/commands/DeleteConnectionCommand.*`
- `src/app/lab/canvas/commands/SetParameterCommand.*`
- `src/app/lab/canvas/commands/SetParameterEnumCommand.*`
- `src/app/lab/canvas/commands/MoveNodeCommand.*`
- `src/app/lab/canvas/commands/ResizeNodeCommand.*`
- `src/app/lab/canvas/commands/SetNodeColorCommand.*`

All AI-applied changes are grouped into a single `juce::UndoManager`
transaction. Partial graph mutation from streaming text is forbidden. This
rule holds for every tier including autonomous sessions: an agent step that
mutates the graph is one named undo transaction.

### 2. AI Never Runs on the Realtime Audio Thread

LLM calls, tool execution, context building, validation, analysis, offline
verification renders, and future ML inference run outside the realtime audio
callback.

- AI work uses a dedicated `AiJobRunner` (message-thread-marshalled via
  `juce::MessageManager::callAsync`).
- The removed offline worker-pool implementation was a render-internal
  primitive and must not be recreated as the async AI job queue.

### 3. Descriptors and Facts Are the Product Surface

LLM quality is bounded by what Lab can tell the model about operators.
`xyona::OpDesc` is already richer than a typical plugin catalog (provider,
family, moduleName, domain, materialization, capabilities, parameters,
`HelpMeta`, slot and routing metadata). Phase 0 adds optional AI metadata.

In addition, the workspace help system already defines an evidence-bound fact
surface that the AI integration must reuse instead of duplicating:

- `../../../docs/contracts/HELP_COMPANION_FACTS.md` - deterministic per-operator
  `<locale>.help.json` fact projections (ports, params, ranges, defaults,
  related operators, fingerprints). These are the primary data source for the
  operator tools in the Tool Registry.
- `../../../docs/contracts/HELP_AUTHORING_BUNDLE.md` and
  `../../../docs/contracts/OPERATOR_HELP_STANDARD.md` - the evidence model:
  agents never invent operator facts; every claim is grounded in descriptor or
  bundle evidence.

### 4. One Tool Surface for Every Frontend

There is exactly one `ToolRegistry`. The embedded assistant UI, autonomous
agent sessions, the MCP gateway, and headless automation all call the same
tools with the same validation, the same privacy gate, and the same permission
checks. No frontend gets a privileged side channel into project mutation.

This is the central future-proofing rule: new frontends (a future MCP client,
a scripting console, a batch CLI) and new providers plug into the existing
surface; they never extend the mutation path.

### 5. Read and Explain Before Generate, Generate Before Autonomy

Capability tiers ship strictly in order:

- read-only workflows (explain graph, inspect node, find operator, summarize
  diagnostics)
- single-action suggestions with preview/apply/reject
- multi-action patch building with sandbox preflight
- autonomous sessions with self-verification

Each tier requires eval baselines from the tier below before it is exposed to
users.

### 6. Privacy From Phase 1

Privacy is a project/runtime gate, not a UI disclaimer.

Privacy modes:

- `LocalOnly`: no cloud provider calls
- `MetadataOnly`: send graph and descriptor metadata, no file paths or audio
- `CloudAllowed`: cloud LLM calls allowed within budget
- `NeverSendAudio`: audio bytes and audio-derived digests require explicit
  per-job opt-in or are blocked

All outbound provider requests pass through one privacy chokepoint:
`PrivacyGate::canSend(PayloadCategory)`. Providers and gateway transports must
not bypass it.

### 7. Autonomy Is Granted, Bounded, and Reversible

Autonomous operation is opt-in per session and capability-gated:

- `L0 ReadOnly`: tools that read; no mutation tools available
- `L1 Propose`: mutation tools produce plans; user applies via preview
- `L2 ApplyWithReview`: agent applies validated plans; every step is one undo
  transaction; user reviews the journal and can roll back to any checkpoint
- `L3 AutonomousSession`: agent runs a bounded goal loop (steps, tokens, time,
  cost caps) including self-verification; same journaling and rollback rules

The permission level is enforced in the Tool Registry (tools are simply not
available below their level), not in prompts. Prompts are guidance; the
registry is the gate.

### 8. Versioned Contracts Everywhere

Every machine-readable surface carries an explicit schema version from day
one:

- `ActionPlan.schemaVersion` (starts at 1)
- `AiMeta` metadata schema in `xyona-operator-v1` (an `ai.schema` field)
- Tool definitions (name, semver-ish `toolVersion`, JSON schema for input)
- Agent journal entries

Old versions are parsed forever or rejected loudly; silent reinterpretation is
forbidden. This is what allows model upgrades, provider swaps, and MCP clients
of different generations to coexist.

### 9. Evals Gate Capability, Not Vibes

The integration maintains golden eval task sets (read-only Q&A, single-action
suggestions, multi-action patch builds, autonomous goals). Metrics:

- plan validity rate (validator pass on first attempt / after repair loop)
- task success against machine-checkable post-conditions
- undo/rollback integrity
- token and cost per task

Evals run with replay fixtures in CI and against live providers manually on
model/provider upgrades. A capability tier is enabled by default only when its
eval baseline is green.

---

## Architecture Bands

The AI integration is one feature module (see Code & Project Structure) with
clear internal bands. Dependency direction is strictly one-way: `ai/` depends
on canvas commands, DiscoveryService, GraphBuilder, and help facts - nothing
outside `ai/` depends on `ai/` internals (only on the `AiCenter` facade).

### Action Bus

Location: `src/app/lab/ai/actions/`

- parse provider output into typed, schema-versioned `ActionPlan`
- validate operator IDs, node IDs, ports, parameter IDs, ranges, and enums
- simulate graph effects in the `GraphMutationSandbox` before apply
- return structured validation errors to the provider/tool loop
- apply accepted plans through existing canvas commands

The Action Bus is the only path from AI output to project mutation - for the
embedded assistant, agent sessions, and the MCP gateway alike.

### Verification Band

Location: `src/app/lab/ai/verify/`

What makes autonomy trustworthy. Three verification levels, used by validator,
preview, and agent sessions:

1. **Static preflight** (Phase 1): `GraphMutationSandbox` applies the plan to
   a decoupled graph model and runs the same `GraphBuilder` validation path;
   produces diff + diagnostics.
2. **Post-conditions** (Phase 3): machine-checkable assertions attached to a
   plan (node/connection count deltas, no error-severity diagnostics, required
   output reachable).
3. **Audio verification** (Phase 4): optional offline render of the affected
   subgraph plus analyzer metrics (loudness, spectral summary, silence/clip
   detection) compared against goal criteria. Never on the RT thread. The
   metric computation is thread-agnostic; the production render seam
   (`OfflineSubgraphRenderer`) is Canvas-owned and therefore runs on the
   message thread, duration-capped by the verifier
   (`AudioVerifier::kMaxRenderDurationSeconds`) so one verify blocks the UI
   for at most one bounded render. Moving the production render off the
   message thread requires a snapshot-fed renderer (separate follow-up).

### Job Lane

Location: `src/app/lab/ai/jobs/`

- run provider conversations and agent steps off the message thread and audio
  thread (`AiJobRunner`)
- manage cancellation, progress, timeouts, retry state, and background
  execution of long agent sessions
- marshal UI updates through `juce::MessageManager::callAsync`
- keep action application atomic and message-thread-owned

### Knowledge Band

Location: `src/app/lab/ai/context/` and `src/app/lab/ai/knowledge/`

- `ContextBuilder`/`ContextSerializer`: deterministic, token-budgeted
  serialization of graph state. Large graphs are serialized hierarchically:
  a stable top-level summary (node/connection counts per family, named
  regions, IO endpoints) plus on-demand detail via tools, never a full dump.
  Node addressing is stable across a conversation.
- `knowledge/`: read adapters for descriptor AI metadata, help companion
  facts (`<locale>.help.json`), the idioms library, and (Phase 6) project
  memory. All operator facts flow from these evidence sources.

### Session Band

Location: `src/app/lab/ai/sessions/`

- `AgentSession`: goal, autonomy level, budgets (steps/tokens/cost/time),
  current plan, state machine (planning -> acting -> verifying -> done/failed/
  awaiting-user)
- `AgentJournal`: append-only, persisted record of every step - tool calls,
  plans, validation results, verification results, applied undo transaction
  names, costs. The journal is the user-facing audit trail.
- `Checkpoints`: named undo anchor points per agent step; "roll back to step
  N" maps to the undo history. A session abort never leaves a half-applied
  step. Two consequences of sharing the project undo history: manual canvas
  edits during a running L2/L3 session pause the session with a warning
  (rolling back a checkpoint would also revert those edits — the UI says
  so), and the session step budget must stay within the configured
  `UndoManager` capacity (Lab does not call `setMaxNumberOfStoredUnits`
  today — configure and test it).

### Provider Layer

Location: `src/app/lab/ai/providers/`

- hide provider-specific request/response formats behind `AiProvider`
- expose capabilities (streaming, tool use, context size, prompt caching) via
  a queryable `ProviderCapabilities` struct - callers adapt to capabilities,
  never to provider names
- return tool calls and text deltas to the assistant runtime

One real provider first. Recommended: Anthropic, because native tool use is
the main requirement. The interface must not assume Anthropic-specific
concepts. Model names are settings (`planningModel`, `fastExplainModel`,
`agentModel`), resolved by the provider implementation - never hardcoded.

### Gateway Band

Location: `src/app/lab/ai/gateway/`

Phase 5. Exposes the same Tool Registry to the outside world:

- **MCP server**: MCP `2025-11-25` local stdio tools-only protocol core (no
  open port). The current slice implements `initialize`, `ping`,
  notifications, `tools/list`, and `tools/call`; Resources, Prompts,
  Streamable HTTP with token auth, OAuth/auth discovery, and process lifecycle
  wrappers are later slices. External agents (IDE agents, Claude-family
  desktop clients, custom harnesses) drive Lab through the identical
  tool/validation/privacy path
- **Headless automation**: batch/CLI entry for scripted patch generation and
  verification without UI

The gateway adds transports, not capabilities: autonomy levels, privacy gate,
budgets, and journaling apply unchanged.

---

## Code & Project Structure

The AI integration follows the Lab feature-module standard
(`docs/architecture/FEATURE_MODULES.md`): facade + PIMPL, module-local docs,
compiled into `xyona_lab_lib`, tests mirrored under `tests/`.

```text
src/app/lab/ai/
├── AiCenter.h/cpp               <- Facade (public API, PIMPL; the only header
│                                   other Lab code includes)
├── README.md                    <- Quick start (required)
├── docs/
│   ├── OVERVIEW.md              <- Module architecture (kept current)
│   ├── TOOLS.md                 <- Tool Registry reference (generated or
│   │                               hand-maintained, versioned)
│   └── SESSIONS.md              <- Autonomy levels, journal, rollback
├── actions/
│   ├── ActionPlan.h/cpp         <- typed plan, schemaVersion, plan-local IDs
│   ├── ActionPlanValidator.h/cpp
│   └── ActionPlanApplier.h/cpp
├── verify/
│   ├── GraphMutationSandbox.h/cpp
│   ├── PostConditions.h/cpp
│   └── AudioVerifier.h/cpp      <- Phase 4
├── context/
│   ├── ContextBuilder.h/cpp
│   └── ContextSerializer.h/cpp
├── knowledge/
│   ├── OperatorFacts.h/cpp      <- OpDesc.ai + companion-facts adapter
│   └── IdiomLibrary.h/cpp
├── tools/
│   ├── ToolRegistry.h/cpp       <- single registry, permission-gated
│   └── tools/*.h/cpp            <- one file per tool
├── layout/                      <- Phase 3
│   └── AutoLayout.h/cpp
├── jobs/
│   ├── AiJob.h/cpp
│   └── AiJobRunner.h/cpp
├── sessions/                    <- Phase 4
│   ├── AgentSession.h/cpp
│   ├── AgentJournal.h/cpp
│   └── Checkpoints.h/cpp
├── providers/
│   ├── AiProvider.h             <- interface + ProviderCapabilities
│   ├── MockProvider.h/cpp
│   └── AnthropicProvider.h/cpp  <- Phase 2
├── gateway/                     <- Phase 5
│   ├── McpServer.h              <- AI-38 stdio protocol core (header-only)
│   └── HeadlessRunner.h         <- AI-39 CLI argument/result core (header-only)
├── privacy/
│   └── PrivacyGate.h/cpp
├── telemetry/
│   └── AiTelemetry.h/cpp
└── ui/
    ├── AiPanel.h/cpp            <- Phase 2 side panel
    ├── AiChatView.h/cpp
    ├── AiActionPreview.h/cpp
    ├── AiSessionView.h/cpp      <- Phase 4 journal/checkpoint UI
    └── AiPrivacyIndicator.h/cpp
```

Supporting locations:

```text
src/app/lab/debugbar/AiDebugPanel.h/cpp     <- Phase 1 debug-only UI
src/app/lab/preferences/AiPreferences.h/cpp <- Phase 2 (new preferences dir)
tests/ai/                                   <- mirrors module layout
tests/ai/fixtures/provider_replay/          <- VCR-style replay fixtures
tests/ai/evals/                             <- golden eval task sets
resources/ai/prompts/                       <- system prompts, per locale
resources/ai/idioms_library.json
docs/ai/test_plans/                         <- manual UX test plans
```

Layering rules (enforced in review, checkable by include lint):

1. Only `ActionPlanApplier` touches `juce::UndoManager` and canvas commands.
2. Only `AiJobRunner` spawns threads; everything else is thread-agnostic.
3. Only providers and the gateway perform I/O with the outside world, and
   both sit behind `PrivacyGate`.
4. `ui/` depends on the rest of the module; nothing in the module depends on
   `ui/`.
5. No JUCE UI types outside `ui/` and the debug panel. Core logic
   (actions/verify/context/knowledge/tools) is headless and unit-testable.
6. i18n: all user-visible strings in `ui/` go through the existing `i18n/`
   mechanism; assistant output language follows the app locale via prompt
   configuration, with English fallback.

Naming follows module conventions (`AiCenter` facade like `ParameterCenter`/
`HelpCenter`). All files compile into `xyona_lab_lib`; the core logic must not
require a display and stays testable in CI.

---

## Phase 0 - Descriptor AI Metadata

**Duration:** 3-5 weeks  
**Precondition:** operator module naming/descriptor refactor - **merged, done**  
**User-visible AI:** none

### Goal

Make operators legible to an assistant without changing runtime behavior.

### Deliverables

Core descriptor additions (in `../xyona-core/include/xyona/types.hpp`):

```cpp
struct AiParamExample {
    std::string scenario;
    std::unordered_map<std::string, double> numericValues;
    std::unordered_map<std::string, std::string> enumValues;
};

struct AiIdiomaticPatch {
    std::string name;
    std::string description;
    std::vector<std::string> beforeOps;
    std::vector<std::string> afterOps;
};

struct AiMeta {
    int schema = 1;                      // versioned from day one
    std::vector<std::string> semanticTags;
    std::string purpose;
    std::string antiPurpose;
    std::vector<AiParamExample> paramExamples;
    std::vector<AiIdiomaticPatch> idiomaticPatches;
    std::vector<std::string> relatedOperators;
    std::string dangerNotes;
    bool nondeterministic = false;
};

struct OpDesc {
    ...
    AiMeta ai;
};
```

Schema additions for `xyona-operator-v1`:

```yaml
ai:
  schema: 1
  semanticTags: [gain, loudness, utility]
  purpose: "Use this for transparent linear gain changes."
  antiPurpose: "Do not use this for dynamics control or clipping protection."
  relatedOperators: [cdp.modify.loudness_dbgain]
  dangerNotes: "Large gain values may clip downstream processors."
```

Tooling:

- `../xyona-core/specs/ai_semantic_tags.yaml` (controlled vocabulary)
- extend `../xyona-core/tools/operator_modules/validate_operator_modules.py`
  with an optional `ai` block validation (follows the existing vocabulary
  validation pattern; no refactoring required)
- optional seed script that drafts AI metadata from existing help companion
  facts, reviewed by hand

Parser note (verified against current code): `packs_api.cpp` extracts pack
metadata with small manual JSON helpers. Nested `ai` metadata must either be
stored as raw JSON and handed to Lab unparsed, or parsed with a real JSON
parser. Do not grow ad hoc string extraction for nested arrays and objects.

Descriptor wiring reality (verified 2026-07-05): only the CDP pack has an
op.yaml -> descriptor pipeline
(`xyona-cdp-pack/scripts/generate_operator_metadata.py` generates checked-in
JSON headers under `src/generated/`, consumed via the packs API). Core
operators build `OpDesc` by hand in `adapter/core_operator.cpp`
(`buildDescriptor`), Lab operators in `CustomOperatorMacros.h`;
`lab-public.op.yaml` is a validated spec that no runtime or build step
consumes. An `ai:` block in op.yaml therefore does NOT reach Core/Lab
descriptors automatically. Rule: op.yaml stays the single source; Core/Lab
adopt a small codegen following the CDP pattern (script emits checked-in
`ai` metadata headers included by `buildDescriptor`/macros). Hand-mirroring
is acceptable only as a documented transition with a mandatory drift check.
This is roadmap decision AI-D9.

### Initial Coverage

Minimum seed set (coverage gate warning-only until Phase 2 completes):

- 5-8 representative CDP pack operators
- 3-5 Lab public operators
- 3-5 Core operators

### Verification

- validator accepts optional `ai` blocks; malformed tags fail against the
  controlled vocabulary
- `DiscoveryService::get(id)` returns descriptors with `ai` data
- at least one Core, one Lab, and one CDP operator expose AI metadata in a test

---

## Phase 1 - Action Loop Without LLM

**Duration:** 6-8 weeks  
**User-visible AI:** debug-only

### Goal

Build and test the full action loop with deterministic fixtures and no network
calls. This phase creates the module skeleton exactly as specified in Code &
Project Structure (facade, actions, verify, context, tools, jobs, providers/
mock, privacy, telemetry).

### Action Types

Initial actions: `CreateNodeAction`, `DeleteNodeAction`,
`CreateConnectionAction`, `DeleteConnectionAction`, `SetParameterAction`,
`SetParameterEnumAction`, `MoveNodeAction`.

Each action includes `rationale`, `confidence`, and stable plan-local IDs for
newly created nodes (generated nodes must be referenceable by later actions
before Lab assigns real `NodeId` values).

```json
{
  "schemaVersion": 1,
  "actions": [
    {
      "type": "create_node",
      "tempNodeId": "n1",
      "operatorId": "cdp.modify.loudness_gain",
      "position": { "x": 320, "y": 120 },
      "rationale": "Transparent gain stage before output.",
      "confidence": 0.91
    },
    {
      "type": "connect",
      "fromNodeId": "existing_input",
      "fromPort": "out",
      "toNodeId": "n1",
      "toPort": "in",
      "rationale": "Route source into gain processor.",
      "confidence": 0.88
    }
  ]
}
```

### Validation

`ActionPlanValidator` checks:

- schema version is known
- operator exists in `DiscoveryService`
- node IDs and temp node IDs resolve
- parameter IDs exist; numeric values in range; enum values allowed
- ports exist and types are compatible
- plan action count does not exceed a hard cap, initially 64
- graph preflight succeeds in the `GraphMutationSandbox`

**Sandbox decision (resolved):** the sandbox uses a lightweight decoupled
graph model (not the live `Canvas`, not a hidden canvas instance) that feeds
the same `GraphBuilder` validation path as the live graph. Decoupling the
builder input from the live canvas is the architecturally hardest part of this
phase - spike it first, before committing to the sandbox API.

### Application

`ActionPlanApplier`:

- maps plan-local IDs to real `NodeId` values as commands complete
- applies commands through `juce::UndoManager::perform` inside one named
  transaction
- aborts and rolls back if any command fails mid-plan
- never applies partial streaming output

### Tool Registry (defined here, used everywhere later)

Phase 1 defines the registry and the read-only tool set against fixtures:

- `list_operators`, `search_operators_by_tag`, `get_operator_descriptor`
  (backed by descriptor AI metadata + help companion facts)
- `get_graph_summary` (hierarchical, token-budgeted), `get_graph_diagnostics`,
  `inspect_node`
- `propose_actions`, `dry_run_validate`

Every tool declares: name, `toolVersion`, JSON input schema, required autonomy
level, payload categories it may emit (for the privacy gate).

### Project State

```text
ProjectState.ai
  privacyMode
  monthlyBudget
  conversationSummaryRefs
```

No raw API keys in `.xyona` project files. Provider credentials belong in app
settings or platform secret storage.

### UI

Debug-only: `src/app/lab/debugbar/AiDebugPanel.h/cpp` shows current job state,
last action plan, validation result, rejection reasons, telemetry counters.

### Verification

Tests under `tests/ai/`:

- `MockProviderActionRoundTripTest`
- `ValidatorRejectionTest`
- `GraphBuilderPreflightTest`
- `JobLifecycleTest`
- `PlanLocalIdResolutionTest`
- `AtomicUndoTransactionTest`
- `ToolRegistryPermissionTest` (mutation tools invisible at L0)

Acceptance: fixture -> plan -> validate -> apply -> undo restores the previous
state, in CI, without network.

---

## Phase 2 - One Real Provider and Read-Only UX

**Duration:** 8-10 weeks  
**User-visible AI:** yes, read-first side panel (tier: Assistant, entering
Copilot)

### Goal

Introduce one real LLM provider and make it useful before it can build complex
patches.

`src/app/lab/ai/providers/AnthropicProvider.h/cpp` implements the Phase 1
`AiProvider` interface. No second provider until a concrete release or
compliance requirement exists.

**HTTP stack decision (resolved recommendation):** the project currently has
no HTTP client at all (`vcpkg.json`: eigen3, libmysofa only). Adopt libcurl
(or cpr) via vcpkg with SSE streaming support; `juce::URL` is acceptable only
for a non-streaming first slice. Decide and spike this at the start of the
phase - it is the largest new dependency of the whole plan.

### Use Cases

Supported first (grounded in companion facts and descriptor AI metadata):

- "Explain this graph." / "Why is there no signal at the output?"
- "Find an operator for this task." / "Inspect the selected node."
- "Summarize current GraphBuilder diagnostics."

Limited mutation: single-action suggestions only, preview/apply/reject, Phase 1
validation mandatory.

### Tool Errors Drive the Repair Loop

Tool errors are structured and sent back to the provider:

```json
{
  "kind": "unknown_operator_id",
  "got": "gain_boost",
  "suggestions": ["xyona.core.gain", "cdp.modify.loudness_gain"]
}
```

After a bounded number of failed repair attempts, the conversation aborts the
plan and reports the validation failure.

### UI

`src/app/lab/ai/ui/`: `AiPanel`, `AiChatView`, `AiActionPreview`,
`AiPrivacyIndicator`. Streaming text is allowed; action plans apply only after
fully assembled validation; reject leaves the graph untouched; accept creates
one undo transaction. Panel strings via `i18n/`; assistant responds in the app
locale.

### Preferences and Secrets

`src/app/lab/preferences/AiPreferences.h/cpp`: provider, planning model, fast
explain model, privacy default, budget caps, API key alias/status.

Credentials: platform secret store (Keychain / DPAPI / libsecret) with an
`ANTHROPIC_API_KEY` environment fallback for development. Never in project
files, never fully logged.

### Eval Harness v1

`tests/ai/evals/` gets its first golden sets:

- 20+ read-only Q&A tasks over fixture projects with machine-checkable
  expectations (mentions the right operator ID, cites the real diagnostic)
- 10+ single-action suggestion tasks with post-condition checks

CI runs them against replay fixtures; a manual job runs them live before any
model/provider change is rolled out.

### Verification

- VCR-style provider replay fixtures
- `LocalOnly` blocks cloud calls; `MetadataOnly` strips paths and
  audio-sensitive fields
- token/cost budget stops cleanly at a low artificial cap
- action reject leaves state unchanged; accepted action undoable
- eval baseline recorded (this baseline gates Phase 3 exposure)

---

## Phase 3 - Generative Patch Assistant

**Duration:** 8-10 weeks  
**User-visible AI:** multi-action patch building (tier: Copilot, complete)

### Goal

The assistant proposes complete patch changes while Lab remains the sole
validator and mutator.

Example prompts:

- "Create a simple mastering chain on the selected bus."
- "Build a granular-style texture chain from this source."
- "Add a diagnostic tap before the output."
- "Replace this gain staging with a safer loudness normalization path."

### Deliverables

- extended actions: `ResizeNodeAction`, `SetNodeColorAction`, optional grouped
  layout metadata
- `src/app/lab/ai/layout/AutoLayout.h/cpp` - deterministic placement without
  overlapping existing canvas content
- `resources/ai/prompts/system_planner.md`, `resources/ai/idioms_library.json`

### Plan Simulation Becomes a Hard Gate

```text
ActionPlan
  -> apply to GraphMutationSandbox
  -> run GraphBuilder preflight
  -> check declared post-conditions
  -> produce diff and diagnostics
  -> only then allow preview/apply
```

Post-conditions (introduced here, reused by Phase 4 verification):

- expected node count delta
- expected connection count delta
- no GraphBuilder diagnostics of severity error
- required output reachable

### Preview

`AiActionPreview`: textual diff, action-by-action list, validation warnings,
canvas ghost layer for new nodes/connections if feasible (preview-only; never
real canvas nodes).

### Verification

- golden action-sequence fixtures; corrupted-plan mutation tests
- multi-action apply and undo roundtrip; plan cap enforcement
- failed mid-apply command rolls back previous actions
- eval set extended with 10+ multi-action build tasks (post-condition checked);
  baseline gates Phase 4
- manual UX test plans in `docs/ai/test_plans/`

---

## Phase 4 - Verified Autonomy (Agent Sessions)

**Duration:** 10-12 weeks  
**User-visible AI:** goal-driven agent sessions (tier: Agent)

### Goal

The agent works on a goal independently inside the system: plan -> act ->
verify -> iterate, with every step validated, journaled, budgeted, and
reversible. This is the "works and patches autonomously" end state.

### The Agent Loop

```text
AgentSession(goal, autonomyLevel, budgets)
  loop:
    plan next step        (provider call with tools)
    act                   (ActionPlan -> validator -> sandbox -> apply
                           as one named undo transaction = checkpoint)
    verify                (post-conditions; optionally AudioVerifier:
                           offline render of affected subgraph + analyzer
                           metrics vs. goal criteria)
    if verified: continue or finish
    if failed:   revert step (undo to checkpoint), replan (bounded retries)
  until: goal met | budget exhausted | user abort | needs user decision
```

### Deliverables

- `sessions/AgentSession`, `sessions/AgentJournal` (append-only, persisted,
  human-readable), `sessions/Checkpoints` (named undo anchors per step)
- `verify/AudioVerifier`: offline render + analyzer metrics (loudness,
  spectral summary, silence/clipping), reusing existing analyzer
  infrastructure and `MaterializedAudioStore` buffer identity for caching;
  threading per the Verification Band note above (message-thread render
  seam, duration-capped, never RT); subject to `NeverSendAudio`
  (verification is local; only derived scalar metrics may ever be
  summarized to the provider, gated by privacy mode)
- autonomy levels L0-L3 enforced by the Tool Registry; L2/L3 require explicit
  per-session user grant
- budget enforcement: max steps, max tokens, max cost, max wall time; hard
  stop with a clean journal entry
- `ui/AiSessionView`: live journal, per-step diffs, "roll back to step N",
  abort button, budget display
- background execution via `AiJobRunner`; sessions survive panel close, not
  app close (session resume from journal is a later extension)

### Safety Rails

- prompt-injection defense: descriptor text, help content, file names, and
  node labels are untrusted input; they are data in the context, never
  executable instructions; tool results are sanitized (no control tokens);
  the tool allowlist per autonomy level is enforced registry-side
- no recursive self-granting: an agent cannot raise its own autonomy level or
  budgets
- destructive breadth guard: plans that delete more than N nodes (initially 5)
  or touch more than M% of the graph require L3 plus an explicit per-plan
  confirmation

### Verification

- scripted end-to-end autonomous runs against fixture projects with
  MockProvider decision scripts (deterministic, CI)
- kill/abort mid-step leaves a consistent, fully undoable state
- journal replay reconstructs the exact command sequence
- eval set: 10+ autonomous goal tasks with machine-checkable end states
  (graph shape + audio metrics); baseline gates default-on exposure

---

## Phase 5 - Agent Gateway (MCP + Headless)

**Duration:** 6-8 weeks  
**User-visible AI:** external agents and automation drive Lab

### Goal

Expose the proven tool surface to the outside world so that external agent
harnesses (IDE agents, desktop AI clients, custom pipelines) can operate Lab -
with identical validation, privacy, budgets, and journaling. This decouples
XYONA from any single provider's chat UI and is the second pillar of
future-proofing: as external agent ecosystems evolve, Lab only maintains its
tool surface.

### Deliverables

- `gateway/McpServer`: MCP `2025-11-25` tools-only protocol core over local
  stdio first; localhost with token auth is the second transport slice. Serves
  the Tool Registry 1:1 under an autonomy level granted in the Lab UI, never by
  the client. It advertises only the MCP `tools` capability; Resources,
  Prompts, Streamable HTTP, OAuth/auth discovery, and process lifecycle wrappers
  are separate later slices
- `gateway/HeadlessRunner`: CLI batch contract core (`--goal`, `--project`,
  `--autonomy`, `--max-steps`, `--budget-micros`, `--journal`) for scripted
  generation/verification; exits with machine-readable result and journal
  path. Trust model: invoking the CLI on the machine is a local-operator grant
  (equivalent to granting in the app UI), capped by a `maxHeadlessAutonomy`
  preference (default L1); MCP clients never get this path and can never
  self-grant a level
- session security: gateway disabled by default; enabling it is an explicit
  preference; remote transports out of scope until a concrete need exists

### Verification

- MCP conformance tests against a reference client
- permission tests: client requests above granted level are rejected
  registry-side
- privacy tests: `LocalOnly`/`MetadataOnly` gate the gateway exactly like the
  embedded assistant
- one end-to-end demo: external agent builds and verifies a patch via MCP on a
  fixture project

---

## Phase 6 - Intelligence Deepening (Audio Analysis, ML Packs, Generation)

**Not scheduled; begins only after Phase 4 is stable.** Formerly the deferred
track; now harmonized with the workspace ideas backlog.

### 6a - Audio Analysis and Cache

- `ai/analysis/AnalysisService`: offline spectral fingerprint,
  onset/transient analysis, loudness profile (reusing existing loudness
  analyzer code)
- content-addressed cache keyed by `MaterializedAudioStore` buffer identity
  (`dependencySignature`/`fileFingerprint`), analyzer ID, analyzer version
- feeds both the assistant ("what is on this bus?") and `AudioVerifier`

### 6b - ML Runtime and First ML Operators

- decision stands: pack-owned ONNX runtime in a separate `xyona-ml-pack`
  (keeps `xyona-core` small, preserves the dynamic pack boundary; a core-level
  facade only if proven necessary)
- first operator: denoise (broad utility, low licensing risk); source
  separation later (higher impact, license-sensitive, heavier)
- nondeterministic operators declare `ai.nondeterministic = true`

### 6c - Embeddings, Search, and Generation

- audio embeddings + ANN index ("find similar audio", "find audio by
  description")
- generative audio nodes: this is where the ideas-backlog item **OP-003 "AI
  Texture Node"** (`docs/ideas/IDEAS_BACKLOG.md`) lands - a
  descriptor-declared operator calling an `AITextureService` (local model or
  remote API) through the same privacy gate and job lane; buffers land in the
  managed store with prompt/seed metadata for reproducible rerenders
- requires explicit product, privacy, licensing, and storage decisions before
  implementation

---

## Cross-Cutting Requirements

### Privacy

Outbound payload categories: descriptor metadata, graph metadata, file paths,
waveform summaries, audio digests, raw audio bytes. Every provider or gateway
request passes `PrivacyGate::canSend(PayloadCategory)`. No bypass.

### Security

- prompt injection: all project-derived text is untrusted data (see Phase 4
  safety rails); system prompts live in `resources/ai/prompts/` and are the
  only instruction source
- the model never receives or produces anything that is executed directly -
  no eval, no shell, no file writes outside the Action Bus
- gateway transports are local and disabled by default; stdio has no open
  port, while the later localhost transport must be token-authenticated
- secrets in platform secret storage; never in project files or logs

### Cost Guardrails

From Phase 2: per-conversation token budget, per-project monthly budget,
per-session budgets (Phase 4), visible cost estimate, soft warning near limit,
hard stop at limit, deterministic token/size-capped context serializer.

### Context Management

Large graphs never get fully serialized. `ContextBuilder` emits a stable
hierarchical summary within a fixed token budget; detail is pulled through
tools (`inspect_node`, `get_graph_summary` with scope). Node references stay
stable within a conversation. Long conversations are summarized into
`conversationSummaryRefs` rather than replayed verbatim.

### Telemetry

Local by default. Counters: provider requests, input/output/cached tokens,
estimated cost, first-token latency, full-response latency, proposed actions,
validator rejections by reason, user accepts/rejects, apply failures, agent
steps, verification failures, rollbacks. Any anonymous export is opt-in and
disabled by default.

### Determinism

CI uses `MockProvider` and replay fixtures only (`tests/ai/fixtures/
provider_replay/`). Generated patch layout is deterministic. Agent-loop CI
tests use scripted MockProvider decisions. Nondeterministic ML operators
declare it in descriptor metadata.

### Versioning and Future-Proofing

- `ActionPlan.schemaVersion`, `AiMeta.schema`, `toolVersion` from day one
- provider capabilities are queried, never assumed; new models are a settings
  change, not a code change
- frontends (panel, MCP, headless) share one tool surface; new frontends add
  transports only
- eval suites re-run on every model/provider upgrade; regressions block the
  upgrade, not the release

---

## Process Requirements

The original implementation roadmap is complete/superseded under
`docs/roadmaps/done/AI_INTEGRATION/` in the workspace repository. Its
implemented product/safety baseline remains current code. Future work follows
`docs/roadmaps/active/AI_INTEGRATION/ROADMAP_AI_INTEGRATION.md`; it must not
restart the old direct-provider phases or treat this document's original
provider sequence as a compatibility requirement.

Implementation reports remain under `docs/reports/implementation/` and review
reports under `docs/reports/review/` or `docs/reports/technical-review/`.

---

## Open Decisions

Resolved in v2 (previous open decisions 1, 3, 4, 5):

1. Sandbox backing model: lightweight decoupled graph model + shared
   GraphBuilder path (spike A validates).
2. HTTP implementation: libcurl/cpr via vcpkg with SSE; `juce::URL` only for a
   non-streaming first slice (spike B validates).
3. First provider: Anthropic; no second provider without concrete need.
4. Phase 0 AI metadata parsing: raw JSON pass-through or real JSON parser;
   never nested manual string extraction.

Still open:

1. Secret storage order of implementation per platform (Keychain first vs.
   DPAPI first) - decide with spike B.
2. Journal persistence format (JSONL recommended) and retention policy.
3. MCP remote/HTTP wrapper detail - the current Stage 5 slice is pinned to MCP
   2025-11-25 local stdio/tools-only; localhost+token remains the second slice
   for clients without stdio.
4. Session resume after app restart (out of scope for Phase 4; revisit for
   Phase 5 headless runs).
5. Eval result storage and history (in-repo vs. local artifacts).

---

## Acceptance Criteria

### Milestone 1 - ready for a real provider (end of Phase 1)

1. `MockProvider` produces a valid action plan fixture.
2. `ActionPlanValidator` rejects invalid operator IDs, bad parameter IDs,
   out-of-range values, type mismatches, and graph preflight failures.
3. `ActionPlanApplier` applies valid plans through existing canvas commands.
4. A multi-command plan is one undo transaction; undo restores the previous
   graph state; plan-local IDs map correctly to real `NodeId` values.
5. No AI code runs on the realtime audio thread.
6. `LocalOnly` blocks provider calls; CI needs no network.
7. Tool Registry enforces autonomy levels (mutation tools invisible at L0).

### Milestone 2 - ready for autonomy (end of Phase 3)

1. Multi-action plans pass sandbox preflight with post-conditions.
2. Preview/apply/reject is stable; reject never mutates.
3. Eval baselines exist for read-only, single-action, and multi-action tasks.

### Milestone 3 - autonomous and future-proof (end of Phase 5)

1. An agent session completes a golden goal end-to-end: plan, apply, verify
   (including audio verification), journal, within budget - and "roll back to
   step N" restores exactly that state.
2. Abort at any point leaves a consistent, undoable project.
3. An external MCP client performs the same task through the gateway with the
   same validation, privacy, and permission behavior.
4. Swapping the model name in settings (same provider) requires no code change
   and passes the eval suite.
5. All machine-readable surfaces carry schema versions; an old journal and an
   old ActionPlan fixture still parse.
