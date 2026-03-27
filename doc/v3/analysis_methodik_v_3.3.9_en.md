# Analysis of Mathematical and Logical Issues in Methodik v3.3.9

The methodology document was assessed for mathematical consistency. As corroborated by expert review, several issues from the initial analysis have been downgraded or reclassified as valid design choices. The refined overview highlights true mathematical flaws versus edge-case aesthetics and document hygiene.

## 1. Definitive Mathematical and Logical Errors

### 1.1 Impossible Bounds during Adaptive Weighting (Logic Error)
**Location:** §5.3.3 Optional Adaptive Weighting  
**Issue:** The rules require adaptive weights to be "clipped to [0.1, 0.7] and renormalized to sum 1".
- **Assessment:** This is perfectly impossible as an algebraic end state for larger stacks ($N > 10$). If we have $\ge 11$ frames with a minimum bound of $0.1$, the initial sum is strictly $\ge 1.1$. Renormalizing them to sum $1$ will algebraically divide all elements downward, instantly breaking the strict $0.1$ minimum floor. This constraint is mathematically unfulfillable for normal batch dimensions.

### 1.2 Scale and Dimensionality Errors in Reliability Weights (Math Error)
**Location:** §6.3.2(c) Tile Reliability Weight  
**Issue:** The scale parameter $\lambda$ directly scales the high-pass variance metric:  
`w_t = exp(-lambda * structure_score_t)` where `structure_score_t = median(hp(R_{t,c}(p))^2)`
- **Assessment:** `structure_score_t` operates on local variances, meaning its magnitude heavily scales with the absolute intensity representation (e.g. $[0, 1]$ floats versus $[0, 65535]$ integers). Using a fixed default scalar like $\lambda = 1.0$ completely breaks across different execution backends unless the metric is relatively normalized (e.g., divided by the baseline local noise variance $\sigma_{t,c}^2$).

## 2. Theoretical / Edge-Case Discontinuities

### 2.1 Discontinuity in Cluster Mass Fallback
**Location:** §5.11.1 Cluster Quality and Mass Definition  
**Issue:** `If M_{k,c} <= eps_weight, replace it deterministically by M_{k,c} = |k|`.
- **Assessment:** While mathematically discontinuous (an abrupt replacement with a raw absolute count), it is extremely unlikely to manifest as a catastrophic failure under standard parameters. Because the global quality $Q$ is clamped to $[-3, 3]$ before exponentiation $G_{f,c} = \exp(k_{global} \times Q_{f,c}^{clamped})$, the individual weights generally do not hit $10^{-6}$ unless $k_{global}$ is set unreasonably high. It is an aesthetic/theoretical weakness rather than an immediate hazard.

## 3. Ambiguities and Document Hygiene

### 3.1 Dimensional Inconsistency in BGE Auto-tuning Objective
**Location:** §6.3.7.1 Objective (Binding)  
**Issue:** `J = E_cv + alpha_f * E_flat + beta_r * E_rough`
- **Assessment:** Unifying RMS ($E_{cv}$, unit Intensity) linearly with raw energies ($E_{flat}, E_{rough}$, implicit unit $Intensity^2$) without specifying scaling creates a dimensionally fragile formulation. Since those energy terms are heavily underspecified in their text, formal proof of mathematical damage is somewhat shielded. However, it requires a definitive clarification to avoid backend implementations building scale-dependent auto-tuners.

### 3.2 Unreachable Sigma-Clipping Fallback (Dead Code)
**Location:** §5.7 Tile Reconstruction  
**Issue:** "If clipping empties the accepted set numerically, fall back to the unclipped valid weighted mean."
- **Assessment:** The condition `|A_prop| >= ceil(min_fraction * |V|)`, and its direct consequence to identically adopt the predecessor iteration on violation, mathematically guarantees an active set will never empty itself dynamically ($\min(1, ceil(\geq 1)) \geq 1$). The fallback is dead code and merely harmless document hygiene.

## 4. Reclassified Design Choices (False Positives)

### 4.1 BGE Grid Aggregation (Valid Architecture)
**Location:** §6.3.3 Coarse Grid Aggregation
- **Assessment:** Initially marked as a masking logic break, computing the robust cell state via an unweighted median `b_cell = median({b_{t,c}})` alongside `w_cell = median({w_t})` is a fully intentional design split. The unweighted nature allows the robust filter to operate agnostically locally, while the generated $w_j$ weights are strictly applied inside the actual fitting objective `argmin sum w_j * rho(...)` downstream (§6.3.10). An explicit weighted median is an alternative, but the existing path operates correctly.

### 4.2 Blind Diffusion of Low Quality Confidence (Valid Architecture)
**Location:** §5.5.5 Spatial Regularization of Local Scores
- **Assessment:** Target-side throttling logic (`lambda_{eff}` exclusively controlled by $U_{f,t,c}$) intentionally regulates adoption but assumes neighborhood metric equality as a structural prior. Factoring in source tile confidence dynamically would require re-scoping the diffusion matrix heavily, which qualifies as a feature evolution rather than correcting a logical inconsistency.
