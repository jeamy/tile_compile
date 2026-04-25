# Differences between Methodology v3.3.8 and v3.3.9

**Status:** Comparison document  
**Compared files:**  
- `doc/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.8_en.md`
- `doc/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.9_en.md`

---

## 1. Summary

`v3.3.9` is not just an editorial revision of `v3.3.8`; it is a substantive tightening of the methodology core. The most important changes are:

1. clean separation of additive background `B_{f,c}` and multiplicative photometric scaling `P_{f,c}`
2. removal of tile-wise nonlinear renormalization before OLA from the mandatory core
3. soft local STAR/STRUCTURE blending instead of hard switching
4. support-aware overlap-add semantics with explicit exclusion of canvas-invalid or unsupported pixels
5. mass-preserving cluster aggregation in full mode
6. stronger normative definitions for BGE, PCC, validation, and ML-mask integration

---

## 2. Structural Changes

### 2.1 Version and Objective Definition

- `v3.3.8` was updated to `v3.3.9`.
- The objective section was materially refocused.
- Newly emphasized in `v3.3.9`:
  - separation of additive background and photometric scaling
  - prohibition of tile-wise nonlinear renormalization in the mandatory core
  - soft local quality blending
  - mass-preserving cluster aggregation
  - support-aware OLA with deterministic boundary coverage

### 2.2 New or Relocated Subsections

New in `v3.3.9`:

- `4.1.1 Optional Pre-Warp CFA Defect-Pixel Suppression (Feature-Gated)`
- `6.3.10 RBF Surface Mathematical Specification`
- `6.4.4 Optional Post-PCC Isolated Chroma-Speckle Suppression`

Removed or replaced in `v3.3.9`:

- `5.7.1 Tile Normalization before OLA`
- `5.7.1a Robust Tile-Normalization Guard`
- `5.7.1b Photometric Preservation after OLA`
- the old RBF appendix in `9.2.7` was removed and relocated to `6.3.10`

---

## 3. Changes in the Reconstruction Core

### 3.1 Linearity Semantics

`v3.3.9` adds explicitly:

- tile-wise data-dependent renormalization of reconstructed pixel values before OLA is **not** part of the mandatory linear core

This is a central methodological change relative to `v3.3.8`, where that normalization was still normative in `5.7.1`.

### 3.2 Global Normalization

`v3.3.8`:

- derived linear normalization in the core from the global background level
- `I = I_raw / max(B, eps_bg)`

`v3.3.9`:

- introduces a mandatory two-step path:
  - additive background subtraction `J = I_raw - B`
  - photometric scaling `I = J / max(P, eps_scale)`
- defines `photometric_scale()` bindingly:
  - ensemble stellar-flux scaling
  - exposure-time ratio
  - deterministic fallback `P = 1`
- explicitly forbids deriving `P` solely from sky background `B`

### 3.3 Adaptive Global Weights

`v3.3.8`:

- allowed optional adaptive weighting from variances of z-normalized metrics

`v3.3.9`:

- makes the static weights the mandatory core
- keeps adaptive weights only as an optional extension
- requires documented utility/tie-break semantics instead of a simple `Var(z(.))` heuristic

### 3.4 Tile Geometry

`v3.3.9` tightens tile geometry:

- explicit `T_hi = floor(min(W,H)/D)`
- deterministic compact-tile mode when `T_hi < T_min`
- additional guards for `S <= 0`, `O >= T`, `T <= 0`
- explicit note that some guards are purely defensive under valid preconditions

### 3.5 Local Tile Metrics

`v3.3.8`:

- used hard STAR/STRUCTURE classification

`v3.3.9`:

- replaces classification with `eta_t` as a soft STAR/STRUCTURE blend factor
- adds binding canvas exclusion for all local metrics
- extends regularization with:
  - local confidence `U_{f,t,c}`
  - edge-aware neighbor affinity `A_{t,u}`
  - `eps_affinity` guard
- introduces `k_local` as an explicit local weight scale factor

### 3.6 Tile Reconstruction and OLA

This is one of the largest differences between the two versions.

`v3.3.8`:

- defined tile-wise median/MAD normalization before OLA
- added global photometric restoration after OLA

`v3.3.9`:

- removes that entire normalization path from the mandatory core
- defines support-aware OLA instead:
  - valid samples by finiteness plus valid canvas support
  - `M_{t,c}(p)` as an explicit support mask
  - canvas-invalid pixels and `|V| = 0` do not contribute to the OLA denominator
  - binding partition-of-unity and boundary rules
- adds sigma-clipping guards:
  - `N_eff`
  - `D_eff`
  - `min_fraction` keep-floor

### 3.7 State-Based Clustering and Synthetic Frames

`v3.3.8`:

- active only for `N >= 200`

`v3.3.9`:

- replaces the fixed threshold with the mode framework:
  - active iff `N >= max(N_red, 50)`
- adds for `synthetic.weighting=tile_weighted`:
  - use of the same support-aware OLA semantics
  - deterministic fallback to `global` when boundary regression and cross-tile weight disagreement are detected

### 3.8 Final Stacking

`v3.3.8`:

- aggregated clusters by quality weighting only

`v3.3.9`:

- introduces `M_{k,c}` as cluster mass
- makes aggregation mass-preserving:
  - `w_{k,c}^{raw} = M_{k,c} * exp(kappa_cluster * Q_{k,c}^{rel})`
- adds zero-denominator fallbacks at both cluster and final aggregation level

---

## 4. Changes in BGE and PCC

### 4.1 BGE Sampling and Tile Reliability

`v3.3.9` substantially tightens BGE:

- canvas-invalid pixels must be explicitly excluded
- `structure_score_t` is bindingly defined as dimensionless relative high-pass energy
- `w_t` is normatively defined by
  - `exp(-lambda_structure * structure_score_t) * (1 - masked_fraction_t)`
- stability under global intensity rescaling is explicitly required

### 4.2 Coarse Grid and Surface Fit

`v3.3.9`:

- makes `w_cell = median({w_t})` the normative default
- tightens coarse-grid semantics
- directly binds the weight formula from `6.3.2(c)` into `6.3.4`

### 4.3 BGE Autotuning

`v3.3.8`:

- objective `J = E_cv + alpha * E_flat + beta * E_rough`
- configuration names `alpha_flatness`, `beta_roughness`

`v3.3.9`:

- brightness-normalized objective
  - `B_ref = max(abs(median_i b_i), eps_bg)`
  - `J = E_cv / B_ref + alpha_f * E_flat / B_ref^2 + beta_r * E_rough / B_ref`
- new normative configuration names:
  - `bge.autotune.alpha_f`
  - `bge.autotune.beta_r`
- `autotune.best.objective` is now explicitly the normalized objective
- `autotune.best.objective_raw` is added as an auxiliary diagnostic

### 4.4 Adaptive Grid and RBF

`v3.3.9`:

- adds the compact-tile-mode exception for `G >= 2*T`
- relocates the RBF specification from the ML appendix to the proper BGE section `6.3.10`
- makes regularization and canvas exclusion explicit in the RBF fit

### 4.5 PCC

New in `v3.3.9`:

- `6.4.4 Optional Post-PCC Isolated Chroma-Speckle Suppression`

This is the first normative description of:

- RGB-only processing after PCC
- restriction to valid canvas support
- isolated single-channel chroma-outlier correction only
- structure/luma guards

---

## 5. Changes in Validation and Defaults

### 5.1 Validation

`v3.3.8`:

- 13 minimum normative tests

`v3.3.9`:

- 24 minimum normative tests
- newly added tests include, among others:
  - separation of additive and multiplicative normalization
  - admissibility of negative and zero pixels
  - sigma-clipping guards
  - support-aware OLA
  - mass-preserving cluster aggregation
  - STAR/STRUCTURE coefficient sums
  - canvas-invalid exclusion
  - clustering gate according to the mode framework
  - BGE stability under intensity rescaling
  - normalized BGE autotune objective

### 5.2 Recommended Numerical Defaults

`v3.3.9` adds relative to `v3.3.8`:

- `eps_scale`
- `eps_neff`
- `eps_var`
- `tol_ola`
- `eps_affinity`
- `k_global`
- `k_local`
- `min_fraction`
- `lambda_bge`
- `bge.structure_blur_px`
- `validation.wcs_roundtrip_max_arcsec`
- `validation.pcc_max_condition_number`
- `validation.pcc_max_residual_mag`

Removed from the defaults list:

- `eps_median`

---

## 6. Changes in Scope and Optional Extensions

### 6.1 Optional Extensions

New in `v3.3.9`:

- deterministic CFA defect-pixel suppression / cosmetic correction
- post-PCC isolated chroma-speckle suppression

### 6.2 ML Extension

`v3.3.9` adds bindingly in `9.2.3`:

- how ML masks interact with the valid sample set
- that soft masks do not change membership in `V_{t,c}(p)`
- that sigma-clipping must operate at pixel level with `Ŵ_{f,t,c}(p)`
- that canvas invalidity overrides ML masks

Also:

- the old RBF section was removed from the ML part
- it is replaced there by a reference to `§6.3.10`

---

## 7. Main Net Effects

Overall, `v3.3.9` shifts the methodology in the following direction:

- fewer implicit nonlinear tile heuristics inside the reconstruction core
- clearer photometric semantics
- more robust boundary and support handling
- more strongly normalized BGE/PCC integration
- more precise validation and diagnostics requirements

The largest substantive differences are therefore:

1. `v3.3.8` still allowed or required tile-wise pre-OLA normalization, while `v3.3.9` removes it from the mandatory core.
2. `v3.3.8` used a hard STAR/STRUCTURE switch, while `v3.3.9` replaces it with soft blending plus confidence-aware regularization.
3. `v3.3.8` aggregated clusters by quality weighting, while `v3.3.9` additionally preserves cluster mass.
4. `v3.3.9` makes canvas support and OLA semantics explicit binding mathematical constraints.

---

## 8. Note on the Nature of This Document

This document is **not a raw line-by-line diff**. It is a structured, content-oriented summary of the differences between `v3.3.8` and `v3.3.9`.
