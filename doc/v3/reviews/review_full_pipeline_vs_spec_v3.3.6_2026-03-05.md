# Full Pipeline Conformance Review vs Methodology v3.3.6

Date: 2026-03-05  
Scope: full `tile_compile_cpp` pipeline (core + BGE + PCC) against `doc/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.6_en.md`

Profiles reviewed:
- `strict` profile semantics
- `practical` profile semantics (including CFA-proxy core path)

## Verdict

- `practical`: **largely conformant** for the implemented core+BGE+PCC path.
- `strict`: **partial conformance**.
- Remaining conformance risk is concentrated in:
  1. strict interpretation of explicit channel split timing, and
  2. missing normative automated tests for BGE/PCC/mask invariants.

## Evidence Base

- Pipeline code paths in:
  - `apps/runner_pipeline.cpp`
  - `apps/runner_phase_metrics.cpp`
  - `apps/runner_phase_local_metrics.cpp`
  - `apps/runner_phase_registration.cpp`
- BGE/PCC algorithm code in:
  - `src/image/background_extraction.cpp`
  - `src/astrometry/photometric_color_cal.cpp`
- Mode/assumption enforcement in:
  - `src/core/mode_gating.cpp`
  - `src/io/config.cpp`
- Run-level evidence:
  - `build/runs/20260305_111455_39fe162a/M42_02.2026_lights_all/logs/run_events.jsonl`
  - `build/runs/20260305_111455_39fe162a/M42_02.2026_lights_all/artifacts/chroma_bins_bge.json`
  - `build/runs/20260305_111455_39fe162a/M42_02.2026_lights_all/artifacts/chroma_bins_pcc.json`

## Full Clause Checklist (Strict vs Practical)

Status legend: `conform`, `partial`, `deviation`, `n/a`.

| Spec clause | Strict | Practical | Implementation evidence | Notes |
|---|---|---|---|---|
| §0 Objective of v3.3.6 | partial | conform | `apps/runner_pipeline.cpp` phase flow + artifacts | Objective is high-level; no single hard assertion in code. |
| §1.1 Physical objective | conform | conform | `apps/runner_pipeline.cpp:2427-2564` | Linear weighted reconstruction implemented. |
| §1.2 No frame selection invariant | partial | conform | `apps/runner_phase_registration.cpp:425-557, 560-683` | Temporary reject state exists; downstream behavior keeps frames via predicted/fallback warps. |
| §1.3 Linearity semantics | conform | conform | `apps/runner_pipeline.cpp:2568-2624` | Core stays linear; output stretch is post-core optional output step. |
| §2.1 Hard assumptions | partial | conform | `apps/runner_phase_metrics.cpp:30-43`, `apps/runner_pipeline.cpp:1135-1189` | Strict explicit early split remains incomplete; practical CFA-proxy semantics are documented/implemented. |
| §2.2 Soft assumptions | conform | conform | `src/io/config.cpp:901-918`, `src/core/mode_gating.cpp:5-31` | Thresholds and gating behavior are implemented. |
| §2.3 Reduced mode (50..199) | conform | partial | `apps/runner_pipeline.cpp:883-889`, `src/core/mode_gating.cpp:5-31` | Strict enforces `max(200, threshold)` full-mode gate; practical keeps configurable threshold semantics. |
| §2.4 Below minimum / emergency | conform | conform | `apps/runner_pipeline.cpp:891-900`, `src/core/mode_gating.cpp:9-16` | Abort below minimum unless emergency mode is explicitly enabled. |
| §2.5 Profile-dependent channel semantics | partial | conform | `apps/runner_pipeline.cpp:455-475, 1091-1094, 1190+` | Practical CFA-proxy path present; strict explicit split by phase 2 is not yet end-to-end. |
| §3 Pipeline overview (phase presence/order) | conform | conform | `apps/runner_pipeline.cpp` phase start/end events | Required phases and order are present. |
| §4 Reg+channel split up to phase 2 | partial | conform | `apps/runner_pipeline.cpp:455-475, 657+` | Same strict caveat: explicit split timing. |
| §4.1 CFA-based registration path | partial | conform | `apps/runner_pipeline.cpp` + prewarp path | Practical path conforms; strict explicit split timing remains partial. |
| §4.2 Registration cascade | conform | conform | `apps/runner_phase_registration.cpp:266-275` | Canonical fallback cascade used. |
| §4.3 CFA-proxy path in practical profile | n/a | conform | `apps/runner_pipeline.cpp:1135-1189` | Practical profile permits deferred split with channel-equivalent semantics. |
| §5.2 Global linear normalization | conform | conform | `apps/runner_phase_metrics.cpp:89-217, 294-298` | Per-channel scale estimation and output background medians in place. |
| §5.3 Global metrics and weights | conform | conform | `apps/runner_phase_metrics.cpp:306-451` | Global metrics/weights implemented. |
| §5.4 Tile geometry | conform | conform | `apps/runner_pipeline.cpp:680-717` | Canvas-aware tile grid and artifact output. |
| §5.5 Local tile metrics | conform | conform | `apps/runner_phase_local_metrics.cpp:124-260` | STAR/STRUCTURE split and local weighting implemented. |
| §5.6 Effective weight `W=G*L` | conform | conform | `apps/runner_pipeline.cpp:1150-1160, 1250-1261, 1310-1317` | Implemented as specified. |
| §5.7 Tile reconstruction + fallback | conform | conform | `apps/runner_pipeline.cpp:1268-1275, 1325-1332` | Sigma-clip + fallback present. |
| §5.7.1 Tile norm before OLA | conform | partial | `apps/runner_pipeline.cpp:1067-1069, 1413-1484` | Strict always-on; practical can disable in reduced/emergency mode. |
| §5.7.2 Windowing + OLA | conform | conform | `apps/runner_pipeline.cpp:1076-1080, 1503-1513, 1562-1580` | Hann windows + overlap-add + normalization. |
| §5.8 Optional denoisers | conform | conform | `apps/runner_pipeline.cpp:1342-1399` | Implemented as optional gated stage. |
| §5.9 State-based clustering | conform | conform | `apps/runner_pipeline.cpp:1721-1964` | Deterministic kmeans path with fallback. |
| §5.10 Synthetic frames | conform | conform | `apps/runner_pipeline.cpp:2045-2420` | Full/reduced semantics implemented. |
| §5.11 Final linear stacking | conform | conform | `apps/runner_pipeline.cpp:2427-2564` | Quality-weighted final linear estimator implemented. |
| §6.1 RGB/LRGB combination | n/a | n/a | Not a mandatory core criterion in this review | Out of scope for strict core conformance decision. |
| §6.2 Astrometry (WCS) | conform | conform | `apps/runner_pipeline.cpp:3023-3140` | Optional stage implemented with artifact persistence. |
| §6.3 BGE before PCC + robust tile model | conform | conform | `apps/runner_pipeline.cpp:3164-3351`, `src/image/background_extraction.cpp:2925-3015` | Ordering and robust apply/fallback semantics implemented. |
| §6.3 canvas-mask exclusion policy | conform | conform | `src/image/background_extraction.cpp:2567-2579, 3162-3163` | Hard mask validation + mask-enforced apply. |
| §6.3.8 Mathematical surface model | partial | partial | `src/image/background_extraction.cpp:2989-2991` + fit methods | Surface fitting exists; proof-level clause-by-clause math trace not fully documented in this review. |
| §6.4 PCC (annulus, stability, mask) | conform | conform | `src/astrometry/photometric_color_cal.cpp:293-358, 735-746, 1797-2160` | Robust star photometry + matrix guards + mandatory canvas-mask policy. |
| §7.1 Success criteria | partial | partial | `apps/runner_pipeline.cpp:2681-2840` | Criteria are computed/written, but failures do not immediately abort run. |
| §7.2 Abort criteria | partial | partial | `apps/runner_pipeline.cpp:891-900, 3416-3424` | Major abort paths exist; not all validation failures are hard-abort by design. |
| §7.3 Minimum tests (normative) | deviation | deviation | `tests/` inventory vs spec list | Dedicated normative BGE/PCC/mask tests missing. |
| §8 Recommended numerical defaults | partial | partial | Config/runtime constants spread across code | Defaults exist but no single explicit compliance lock to spec default list. |
| §9 Mandatory core vs optional extension boundary | partial | conform | Core path in `runner_pipeline.cpp` | Strict profile still has explicit split caveat; practical profile is aligned. |
| §9.1 Practical configuration profiles | conform | conform | `tile_compile_cpp/examples/*.yaml` | Profiles are present and complete. |
| §9.2 ML optimization extension | n/a | n/a | No ML inference path in current pipeline | Not implemented; extension clauses are inactive. |
| §10 Core statement | partial | conform | Aggregate from rows above | Strict remains partial due explicit split caveat + normative test gap. |

## PCC Color-Cast Evidence (Run-Specific)

Run:
- `build/runs/20260305_111455_39fe162a/M42_02.2026_lights_all`

Matrix behavior in logs:
- Earlier no-op case (identity):
  - `logs/run_events.jsonl:1450` shows matrix `[[1,0,0],[0,1,0],[0,0,1]]`
- Corrected PCC application:
  - `logs/run_events.jsonl:1466` shows matrix `[[1.047972,0,0],[0,1,0],[0,0,1.658271]]`

Chroma-bin evidence in artifacts:
- `artifacts/chroma_bins_bge.json`
- `artifacts/chroma_bins_pcc.json`

Key bins (`ratio_of_means`):
- `mid_50_90`: `B/G 0.973089 -> 1.000320`
- `bright_90_99`: `B/G 0.961640 -> 0.997729`
- `core_99_99.9`: `B/G 0.846165 -> 0.979119`

Interpretation:
- The dominant cast issue in bright/core regions was tied to PCC collapsing to identity in earlier behavior.
- With corrected PCC matrix application, bright/core chroma is significantly closer to neutral.

## Remaining Findings (Actionable)

1. Strict-profile explicit channel separation timing is still not fully end-to-end; practical profile is conformant with CFA-proxy semantics.
2. Normative automated tests from §7.3 remain incomplete, especially for mask exclusion invariants and PCC bright-core neutrality regression.
3. Validation criteria are computed and persisted, but not universally hard-abort enforced; decide whether strict mode should enforce hard abort on all failed success criteria.
4. Add explicit traceability tests for §6.3.8 surface-fit behavior (model/robustness guarantees), beyond implementation-level evidence.

## Dead Code / Duplicate Logic (BGE/PCC Focus)

- `apply_common_overlap_to_tile` duplicate hot-path logic has been consolidated into shared helper:
  - `apps/runner_shared.hpp:36-72`
  - used from `runner_phase_local_metrics.cpp` and `runner_pipeline.cpp`
- Residual cleanup candidate (non-blocking): `reg_reject_orientation_outliers` appears reported but not actively incremented in current rejection logic.
