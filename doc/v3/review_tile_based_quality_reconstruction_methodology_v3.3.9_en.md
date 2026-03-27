# Review: Tile-Based Quality Reconstruction Methodology v3.3.9

**Document under review:** `tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.9_en.md`  
**Review date:** 2026-03-27  
**Reviewer:** Cascade (automated analysis)  
**Scope:** Mathematical correctness, logical consistency, notation, specification gaps, structural issues

> **Status: ALL 27 ISSUES FIXED.** All identified issues have been corrected directly in the methodology document. See §0.1 of the methodology document for the complete fix log.

Issues are categorized by severity:
- **[CRITICAL]** — Formula produces wrong or undefined results
- **[MAJOR]** — Logic error, unreachable code, specification gap that breaks conformance
- **[MINOR]** — Notation ambiguity, redundancy, cosmetic issue
- **[INFO]** — Non-normative observation

---

## 1. Mathematical Errors

### 1.1 [CRITICAL] §5.7.1: Support mask `M_t(p)` incorrectly marks empty-reconstruction pixels as valid

**Location:** §5.7.1, line ~534

**Definition in document:**
```
M_t(p) = 1  if R_{t,c}(p) is finite and inside valid tile/canvas support
         0  otherwise
```

**Problem:** When `|V_{t,c}(p)| = 0` (no valid frames at pixel p), the document sets `R_{t,c}(p) = 0` (line ~477). The value `0` is finite, so `M_t(p) = 1`. This means the zero-fill value contributes `omega_t * 0 = 0` to the OLA numerator `A` but `omega_t * 1` to the support accumulator `S`. The final result `I_rec = A / S` is thereby diluted at coverage edges, producing biased output rather than coverage exclusion.

The mask must distinguish "valid zero-valued reconstruction" from "fill value due to empty sample set". Correct definition:
```
M_t(p) = 1  if |V_{t,c}(p)| > 0 AND R_{t,c}(p) is finite and inside valid canvas support
         0  otherwise
```

**Additional sub-issue:** `M_t(p)` has no channel index `c`, but `R_{t,c}(p)` is channel-dependent. In a CFA-proxy-equivalent variant, the number of valid frames per pixel can differ per channel. `M_t` should be `M_{t,c}(p)`.

---

### 1.2 [CRITICAL] §6.3.2(c) vs §6.3.4: Inconsistent tile reliability weight formula

**Location:** §6.3.2(c), line ~778 vs §6.3.4, line ~836

**§6.3.2(c):**
```
w_t = exp(-lambda * structure_score_t) * (1 - masked_fraction_t)
```

**§6.3.4:**
```
w_t = exp(-lambda * structure_score_t)
```

The factor `(1 - masked_fraction_t)` appears in §6.3.2(c) (where the weight is defined for background sampling) but is absent in §6.3.4 (where the same weight is used in the surface fit). These are contradictory. A tile with `masked_fraction_t = 0.9` receives a significantly reduced weight in §6.3.2(c) but full weight in §6.3.4. One of the two formulas is incorrect.

---

### 1.3 [CRITICAL] §6.3.9: `G >= 2*T` constraint violated in compact-tile mode

**Location:** §6.3.9, line ~973 and §5.4, line ~296

**§6.3.9 asserts:**
```
G = clip( max(2*T, min(W,H)/N_g), G_min, G_max )
```
with `G_max = min(W,H)/4`, and claims (line ~988) "Implementations must guarantee that grid resolution is coarser than tile resolution (`G >= 2*T`)."

**§5.4 compact-tile mode** sets `T = min(W,H)` when `T_hi < T_min`.

In compact-tile mode: `2*T = 2*min(W,H)`, but `G_max = min(W,H)/4`. Since the clip upper bound is `G_max = min(W,H)/4 << 2*T`, the result `G <= G_max < 2*T`. The stated constraint `G >= 2*T` is violated. This is not a defensive gap — it is an unconditional violation when compact-tile mode is active.

**Fix required:** Either exclude BGE from compact-tile mode, or define a separate `G_max` branch for this case.

---

### 1.4 [MAJOR] §5.5.5: Spatial regularization drifts to zero when neighbor affinities are near-zero

**Location:** §5.5.5, lines ~410–416

**Formula:**
```
Q^{(k+1)} = (1 - lambda_eff) * Q^{(k)} + lambda_eff * sum_u A_{t,u}^{(k)} Q_u^{(k)} / max(sum_u A_{t,u}^{(k)}, eps_weight)
```

**Problem:** When all neighbor affinities `A_{t,u}^{(k)}` are numerically very small (not zero, since `exp > 0` always, but they can be `~exp(-100/tau)` ≈ 0), the neighbor term becomes:
```
lambda_eff * sum_u(~0 * Q_u) / max(~0, eps) ≈ lambda_eff * 0 / eps ≈ 0
```

The update becomes `Q^{(k+1)} ≈ (1 - lambda_eff) * Q^{(k)}`, which decays toward zero rather than preserving `Q^{(k)}`.

**Binding constraint §5.5.5 point 3** states: "tiles without valid neighbors ... keep `Q_{f,t,c}^{reg} = Q_{f,t,c}^{raw}`". However, the formula does NOT implement this for the numerically-near-zero affinity case — only for a structurally empty N(t), which cannot occur for any grid larger than 1×1 since every tile has at least 2 neighbors.

**Fix:** Add an explicit guard: if `sum_u A_{t,u}^{(k)} < eps_affinity`, set `Q^{(k+1)} = Q^{(k)}`.

---

### 1.5 [MAJOR] §4.2: NCC acceptance criterion becomes unsatisfiable for near-identity frames

**Location:** §4.2, line ~163

**Criterion:**
```
NCC(warped, ref) > NCC(identity, ref) + delta_ncc
```
with `delta_ncc = 0.01`.

**Problem:** For the reference frame itself, `NCC(identity, ref) = 1.0`. The criterion requires `NCC > 1.01`, which is mathematically impossible (NCC ∈ [-1, 1]). Similarly, any frame with `NCC(identity, ref) > 0.99` (well-aligned frame) makes the criterion unsatisfiable. The identity fallback will be triggered — which is the correct outcome — but the document never documents this expected path or states that the criterion is intentionally unsatisfiable in such cases.

Additionally, there is no stated exception for the reference frame itself.

---

### 1.6 [MAJOR] §6.3.8: Ambiguous Huber loss notation — `delta(...)` looks like Dirac delta

**Location:** §6.3.8, line ~952–953

**As written:**
```
rho(r) = 0.5 r^2                       if |r| <= delta
rho(r) = delta(|r| - 0.5 delta)        otherwise
```

The expression `delta(|r| - 0.5 delta)` is mathematically ambiguous: `delta(x)` conventionally denotes the **Dirac delta distribution**, not multiplication. The correct meaning is `delta * (|r| - 0.5 * delta)`. The missing `*` operator makes this look like a function application.

The standard Huber form `rho(r) = delta * |r| - 0.5 * delta^2` in the linear branch is equivalent, uses standard notation, and is unambiguous.

---

### 1.7 [MINOR] §6.3.8: Non-standard thin-plate spline bending energy notation

**Location:** §6.3.8, line ~961

**As written:**
```
B_c = argmin_B sum_i w_i (b_i - B(x_i,y_i))^2 + lambda * integral |D^2 B|^2 dx dy
```

`|D^2 B|^2` is non-standard. The standard 2D TPS bending energy is:
```
J(B) = integral [ (d^2B/dx^2)^2 + 2*(d^2B/dxdy)^2 + (d^2B/dy^2)^2 ] dx dy
```
It is unclear whether `|D^2 B|^2` means the squared Frobenius norm of the Hessian, the trace of the squared Hessian, or the standard TPS form above. These differ by a factor of 2 on the cross-term.

---

## 2. Logic Errors

### 2.1 [MAJOR] §5.9: Hard threshold `N >= 200` conflicts with configurable mode framework

**Location:** §5.9, line ~627

**Document states:** "Active only for N >= 200."

**Conflict with §2.3:** Full mode is defined as `N >= N_red` where `N_red = assumptions.frames_reduced_threshold` is user-configurable. If a user sets `N_red = 100` and has `N = 150`, the system is in full mode by §2.3 but clustering is inactive by §5.9's hard threshold. These two conditions are contradictory.

Either the threshold `N >= 200` should be replaced with `N >= N_red` (consistent with the mode framework), or the relationship between the 200-frame threshold and the configurable thresholds must be explicitly documented.

---

### 2.2 [MAJOR] §5.7: `min_fraction` used in keep-floor without a normative default

**Location:** §5.7, line ~506

**Formula:**
```
|A_prop^{(k+1)}| >= ceil(min_fraction * |V_{t,c}(p)|)
```

`min_fraction` is used as a parameter but no normative default value, valid range, or configuration key is given anywhere in the document. This is a critical specification gap for any conformant implementation.

---

### 2.3 [MAJOR] §5.2: `photometric_scale()` function undefined

**Location:** §5.2 step 3, line ~207

**As written:**
```
P_{f,c} = photometric_scale(J_{f,c})
```

The function `photometric_scale` is never defined. The accompanying text says "from deterministic throughput/flux reference" but provides no formula, algorithm, or reference. Constraint 5 of §5.2 permits `P_{f,c} = 1` as a fallback, but the primary case is completely unspecified. This is a critical gap in a normative specification.

---

### 2.4 [MINOR] §5.4: Guard rules 1 and 2 are unreachable dead code

**Location:** §5.4, lines ~306–307

**Guard rule 1:** "if `S <= 0` → set `o_clipped = 0.25`, recompute `O,S`"  
**Guard rule 2:** "if `O >= T` → set `O = floor(0.25 * T)`, `S = T - O`"

**Analysis:** In step 5, `o_clipped = clip(o, 0, 0.5)`, so `o_clipped ∈ [0, 0.5]`. Therefore:
- `O = floor(o_clipped * T) ≤ floor(0.5 * T) ≤ T - 1` for any integer `T ≥ 1`
- `S = T - O ≥ T - (T-1) = 1 > 0`

Both conditions `S <= 0` and `O >= T` are thus mathematically impossible given the preceding clamping, for any `T >= 1` (which is guaranteed by guard rule 3 aborting for `T <= 0`). These guards are dead code under the specified logic.

They could be removed, or a comment should explain under which historical or future conditions they could trigger.

---

### 2.5 [MAJOR] §5.11.2: No fallback when `sum_k w_{k,c} = 0` in final stacking

**Location:** §5.11.2, line ~705

**Formula:**
```
R_c = sum_k (w_{k,c} * S_{k,c}) / sum_k w_{k,c}
```

No fallback is specified when the denominator `sum_k w_{k,c} = 0`. Unlike §5.7 (tile reconstruction) which has an explicit weight fallback, §5.11.2 is silent on this case. While it is unlikely given `M_{k,c} > 0` and `exp > 0`, the document should specify a deterministic fallback (e.g., equal-weight mean over all synthetic frames) for completeness and conformance testing.

---

### 2.6 [MAJOR] §9.2.3: ML soft mask integration with sigma-clipping algorithm unspecified

**Location:** §9.2.3, lines ~1169–1175

**Problem:** The ML-extended weight `Ŵ_{f,t,c}(p) = Ĝ * L̂ * M̂` is a pixel-level soft weight. However, §5.7's sigma-clipping algorithm uses frame-level tile weights `w_{f,t,c}` (not pixel-level). The ML extension does not specify:

1. Whether sigma-clipping in §5.7 uses `w_{f,t,c}` or `Ŵ_{f,t,c}(p)`.
2. How the soft mask `M̂_{f,t,c}(p) ∈ [m_min, 1]` interacts with the binary valid sample set `V_{t,c}(p)` defined by a finite-check. A frame with `M̂ = 0.05` is nearly excluded, but V still includes it, giving a near-zero weight that participates in the denominator.
3. Whether the keep-floor `ceil(min_fraction * |V|)` counts soft-masked frames.

This is a specification gap that makes ML-mode behavior non-deterministic across implementations.

---

## 3. Notation and Consistency Errors

### 3.1 [MAJOR] §5.3.2 / §6.3.7.1: Symbol collision — `alpha, beta` reused with different meanings

**§5.3.2:**
```
Q_{f,c} = alpha*(-z(B)) + beta*(-z(sigma)) + gamma*z(E)
alpha = 0.4, beta = 0.3, gamma = 0.3
```

**§6.3.7.1:**
```
J = E_cv + alpha * E_flat + beta * E_rough
alpha = bge.autotune.alpha_flatness
beta = bge.autotune.beta_roughness
```

The same Greek letters `alpha` and `beta` are used for entirely unrelated quantities in different sections. The configuration key names (`alpha_flatness`, `beta_roughness`) differ from the global metric defaults in §5.3.2, but the in-document math uses the same unqualified symbols. A reader who holds both sections in context will encounter ambiguity. The BGE autotune symbols should use different notation (e.g., `alpha_f`, `beta_r`) or be renamed entirely.

---

### 3.2 [MAJOR] §5.3.2 vs §5.5.6: Asymmetric weight formulas — global has `k_global`, local does not

**Global weight (§5.3.2):**
```
G_{f,c} = exp(k_global * Q_{f,c}^{clamped})     k_global = 1.0 (configurable)
```

**Local weight (§5.5.6):**
```
L_{f,t,c} = exp(Q_{f,t,c}^{local})              (no k_local)
```

There is no `k_local` scale factor for the local weight. This asymmetry is not documented, explained, or justified. There is no way to independently tune the sensitivity of local weights without modifying Q values. Either a `k_local` parameter is missing from the local weight formula, or the document should explicitly state why `k_local = 1` is hardcoded and not configurable.

---

### 3.3 [MINOR] §1.3: Version reference to v3.3.6 in a v3.3.9 document

**Location:** §1.3, line ~44

**As written:** `"Strictly linear" in v3.3.6 means:`

This is a v3.3.9 document. The phrasing implies this definition was introduced in v3.3.6 and carried forward unchanged. If that is the intent, it should read: `"Strictly linear" (as established in v3.3.6, unchanged in v3.3.9) means:`. As written, it looks like a copy-paste artifact.

---

### 3.4 [MINOR] §6.3.3(c): "or sum, implementation choice" violates determinism requirement

**Location:** §6.3.3(c), line ~805

**As written:** "Weight: `w_cell = median({w_t})` (or sum, implementation choice; must be documented)"

`median` and `sum` of reliability weights have substantially different behaviors: sum accumulates mass (larger cells get more influence) while median is robust to outlier tiles. The document's core emphasis on determinism and reproducibility (§1.3, §7.3 test 13) is undermined by leaving this choice open. The chosen default must be normatively specified, not left as "implementation choice."

---

### 3.5 [MINOR] §5.2: `B_{f,c}` used in quality index is pre-normalization, while `sigma` and `E` are post-normalization

**Location:** §5.3.2, line ~240

```
Q_{f,c} = alpha*(-z(B_{f,c})) + beta*(-z(sigma_{f,c})) + gamma*z(E_{f,c})
```

From §5.1: `B_{f,c}` is "global additive background (before normalization)".  
From §5.2 step 5: `sigma_{f,c}` and `E_{f,c}` are "metrics on normalized data".

The quality index thus mixes z-scores of pre-normalization data (`B`) with post-normalization data (`sigma`, `E`). These are on different physical scales and distributions. While this is not necessarily incorrect (z-scores are scale-independent), the inconsistency is nowhere acknowledged or justified. If `B` varies systematically with exposure time but `sigma` is normalized away, the z-score combination may weight frames suboptimally in mixed-exposure datasets.

---

## 4. Structural / Document Integrity Issues

### 4.1 [MAJOR] RBF surface specification is severely misplaced — appears after §9 instead of within §6.3.4

**Location:** Lines ~1218–1273 (after §9.2.7)

The section "RBF Surface (Binding, when `bge.fit.method = rbf`)" is placed after the ML extension §9.2.7 — far from §6.3.4 where surface fitting methods are introduced and RBF is listed as one option. The document references "RBF surface with smoothing" at line ~831 but the binding specification for it appears ~390 lines later outside any section hierarchy.

The section has no heading number (the heading starts with `####` but is not numbered), making it unreferenceable. It belongs as §6.3.10 or §6.3.4.1 at minimum.

**Impact:** A conformant implementation that reads §6.3.4 for RBF details will not find the binding specification without reading the entire document, including post-scope material.

---

### 4.2 [MAJOR] §6.3.9 references grid spacing `G` and the `G >= 2*T` constraint before RBF (which uses `G` and `μ = G` default) is defined

**Location:** §6.3.9 precedes the RBF section by ~250 lines

The RBF default `μ = G` depends on the adaptive grid spacing defined in §6.3.9, but the binding RBF definition appears after §9. This creates a forward-reference dependency. Combined with issue 4.1, the BGE specification is fragmented across the document in a way that violates its own "normative reference" status.

---

### 4.3 [MINOR] §7.1 and §7.2 use `##` heading level (same as §7), not `###`

**Location:** Lines ~1040, ~1048

```
## 7. Validation and Abort    ← H2
## 7.1 Success Criteria        ← should be H3 (###)
## 7.2 Abort Criteria          ← should be H3 (###)
```

§7.1 and §7.2 appear as siblings of §7 in the heading hierarchy rather than as children. This is a Markdown structural error that will render incorrectly in table-of-contents generators and document viewers.

---

### 4.4 [MINOR] §5.1 uses `##` heading level (same as §5) instead of `###`

**Location:** Line ~176

```
## 5. Shared Core from Phase 3 Onward   ← H2
## 5.1 Notation (Binding)               ← should be H3 (###)
```

Same structural issue as §7.1/7.2.

---

### 4.5 [MINOR] §6.3.8 "Mathematical Surface Model" contains polynomial and TPS but not RBF, which is split out

**Location:** §6.3.8 vs the unnumbered RBF section after §9

§6.3.8 contains the polynomial surface model and TPS alternative, while RBF (also listed in §6.3.4 as a surface method) is separately defined outside the section hierarchy. The three surface methods (polynomial, TPS, RBF) should be co-located within §6.3, ideally as §6.3.8.1, §6.3.8.2, §6.3.8.3.

---

## 5. Specification Gaps

### 5.1 [MAJOR] `structure_score_t` undefined for the weight formula

**Location:** §6.3.2(c) line ~780, §6.3.4 line ~836

Both formulas reference `structure_score_t` computed "from `E/sigma` or similar local structure metrics." For a normative specification, "or similar" is not acceptable. The exact formula (or its normative equivalent) must be defined. Without this, implementations will differ.

---

### 5.2 [MAJOR] §7.3 tests 17 and 18 reference undefined thresholds

**Test 17:** "WCS round-trip error below threshold" — no threshold value or parameter name defined anywhere in the document.

**Test 18:** "PCC residuals below threshold" — same problem.

These tests cannot be implemented as written. Normative threshold values or configuration keys must be given.

---

### 5.3 [MINOR] §5.10.1: No fallback when cluster weight denominator is zero

**Location:** §5.10.1, line ~647

```
S_{k,c} = sum_{f in k} G_{f,c} * I_{f,c} / sum_{f in k} G_{f,c}
```

No fallback is specified if `sum G = 0`. In practice this cannot happen (G = exp(...) > 0), but the pattern is inconsistent with other reconstruction formulas in the document which always include explicit fallbacks.

---

### 5.4 [MINOR] §7.3: No test for STAR/STRUCTURE coefficient sums

§7.3 test 1 checks `alpha + beta + gamma = 1` for the global metric. There is no analogous test for the STAR metric coefficients (`0.6 + 0.2 + 0.2 = 1.0`) or the STRUCTURE metric coefficients (unsigned sum `0.7 + 0.3 = 1.0`). These should be added as normative tests for completeness.

---

## 6. Summary Table

| ID | Section | Severity | Category | Short Description |
|---|---|---|---|---|
| 1.1 | §5.7.1 | CRITICAL | Math | `M_t(p)=1` for fill-zero R; no channel index |
| 1.2 | §6.3.2(c)/6.3.4 | CRITICAL | Math | Inconsistent tile weight formula (with/without masked_fraction) |
| 1.3 | §6.3.9/5.4 | CRITICAL | Math | `G >= 2*T` constraint violated in compact-tile mode |
| 1.4 | §5.5.5 | MAJOR | Math | Regularization drifts to zero with near-zero affinities |
| 1.5 | §4.2 | MAJOR | Math | NCC acceptance unsatisfiable for near-perfect / reference frame |
| 1.6 | §6.3.8 | MAJOR | Math | Huber loss: `delta(...)` looks like Dirac delta (missing `*`) |
| 1.7 | §6.3.8 | MINOR | Math | TPS bending energy `\|D^2 B\|^2` is non-standard/ambiguous |
| 2.1 | §5.9 | MAJOR | Logic | Hard `N >= 200` threshold conflicts with configurable mode framework |
| 2.2 | §5.7 | MAJOR | Logic | `min_fraction` used but never given a normative default |
| 2.3 | §5.2 | MAJOR | Logic | `photometric_scale()` function undefined |
| 2.4 | §5.4 | MINOR | Logic | Guard rules 1 and 2 are unreachable dead code |
| 2.5 | §5.11.2 | MAJOR | Logic | No fallback for zero-denominator in final stack formula |
| 2.6 | §9.2.3 | MAJOR | Logic | ML soft mask integration with sigma-clipping unspecified |
| 3.1 | §5.3.2/6.3.7.1 | MAJOR | Notation | `alpha`, `beta` reused for different quantities |
| 3.2 | §5.3.2/5.5.6 | MAJOR | Notation | Asymmetric `k_global`/missing `k_local` undocumented |
| 3.3 | §1.3 | MINOR | Notation | References v3.3.6 in v3.3.9 document |
| 3.4 | §6.3.3(c) | MAJOR | Notation | "or sum, implementation choice" violates determinism |
| 3.5 | §5.3.2 | MINOR | Notation | Quality index mixes pre/post-normalization z-scores |
| 4.1 | §6.3/§9 | MAJOR | Structure | RBF binding spec misplaced after §9 instead of in §6.3 |
| 4.2 | §6.3.9 | MAJOR | Structure | §6.3.9 forward-references RBF `G` 250 lines before its definition |
| 4.3 | §7.1/7.2 | MINOR | Structure | `##` heading level should be `###` |
| 4.4 | §5.1 | MINOR | Structure | `##` heading level should be `###` |
| 4.5 | §6.3.8 | MINOR | Structure | Polynomial/TPS in §6.3.8, RBF outside — should be co-located |
| 5.1 | §6.3.2(c)/6.3.4 | MAJOR | Gap | `structure_score_t` formula undefined |
| 5.2 | §7.3 tests 17/18 | MAJOR | Gap | Threshold values for WCS/PCC tests undefined |
| 5.3 | §5.10.1 | MINOR | Gap | No fallback for zero cluster weight denominator |
| 5.4 | §7.3 | MINOR | Gap | No test for STAR/STRUCTURE coefficient sums |

---

## 7. Priority Fix Recommendations

### Immediate (block conformant implementation)
1. **1.1** — Fix `M_t(p)` to exclude fill-zero pixels; add channel index `c`
2. **1.2** — Unify tile reliability weight formula in §6.3.2(c) and §6.3.4
3. **2.2** — Add normative default for `min_fraction` (e.g., `min_fraction = 0.5`)
4. **2.3** — Define `photometric_scale()` or provide a normative formula
5. **5.2** — Add threshold values for WCS/PCC validation tests (§7.3 items 17, 18)

### High Priority (correctness / cross-section consistency)
6. **1.3** — Document or fix `G >= 2*T` in compact-tile mode
7. **1.4** — Add affinity-zero guard to regularization update
8. **2.1** — Replace hard `N >= 200` with reference to `N_red`
9. **3.1** — Rename `alpha`/`beta` in §6.3.7.1 to avoid symbol collision
10. **3.2** — Add `k_local` parameter to local weight formula, or justify its absence
11. **4.1** — Move RBF binding spec into §6.3 as a numbered subsection

### Normal Priority (clarity, robustness)
12. **1.5** — Document NCC acceptance behavior for reference frame / near-1.0 case
13. **1.6** — Fix Huber loss notation: `delta * (|r| - 0.5 * delta)`
14. **2.5** — Add explicit fallback for zero denominator in §5.11.2
15. **2.6** — Specify ML mask / sigma-clipping integration in §9.2.3
16. **3.4** — Specify normative default for cell weight aggregation (median or sum, not both)
17. **5.1** — Define `structure_score_t` formula

---

*End of review.*
