    # Tile-Based Quality Reconstruction for DSO – Methodology v4

**Status:** Normative Reference (experimental reference implementation)\
**Replaces:** Methodology v2, v3\
**Valid for:** `tile_compile` (local, non-global reconstruction)

---

## 0. Guiding Principle (new in v4)

> There is **no globally consistent coordinate system**. Every reconstruction is **local, temporally consistent, and self-correcting**.

This methodology no longer describes a classical image stacker, but a **tile-local multi-frame reconstruction operator**.

---

## 1. Core Assumptions (mandatory)

- Raw data is **linear**
- No global registration
- No global reference geometry
- Motion can be **spatially and temporally dependent**
- All geometric corrections are **tile-local**
- Every decision must be locally verifiable

Violation leads to **run abortion**.

---

## 2. Overall Pipeline (v4, normative)

```
Phase 0:  SCAN_INPUT         – Load frames, extract metadata
Phase 1:  CHANNEL_SPLIT      – OSC → R/G/B channel separation (CFA)
Phase 2:  NORMALIZATION      – Global coarse normalization
Phase 3:  GLOBAL_METRICS     – Compute frame quality
Phase 4:  TILE_GRID          – Define tile geometry
Phase 5:  LOCAL_METRICS      – Local tile quality (pre-warp)
Phase 6:  TILE_RECONSTRUCTION_TLR – Local registration + reconstruction
Phase 7:  STATE_CLUSTERING   – State-based clustering
Phase 8:  SYNTHETIC_FRAMES   – Synthetic quality frames
Phase 9:  STACKING           – Final linear stacking
Phase 10: DEBAYER            – Color reconstruction (OSC)
Phase 11: DONE               – Completion + validation
```

**Important:** Registration is **no longer a separate step**, but integrated into Phase 6 (TILE_RECONSTRUCTION_TLR).

---

## 3. Global Coarse Normalization (mandatory)

Purpose: Decoupling photometric transparency.

For each frame *f*:

```
I'_f = I_f / B_f
```

- B\_f: robust global background
- Applied once only
- No local adjustment

---

## 4. Tile Geometry (adaptive)

Initial tile size:

```
T_0 = clip(32 · FWHM, 64, max_tile_size)
```

- `max_tile_size`: Configurable (default: 128)
- Overlap ≥ 25%

**Recursive Refinement (implemented):**
- Tiles are automatically refined at high warp variance
- Refinement criterion:
  - `warp_variance > refinement_variance_threshold`
  - `mean_correlation < 0.5`
- Configurable: `enable_recursive_refinement`, `refinement_max_depth`

---

## 5. Local Registration (core of v4)

### 5.1 Motion Model

Minimal model:

```
p' = p + (dx, dy)
```

Optional (experimental, regularized):

```
p' = p + v + J(p − c)
```

with ||J|| ≪ 1.

---

### 5.2 Iterative Reference Formation (new, mandatory)

For each tile *t*:

1. Initial reference R\_t⁽⁰⁾ from median frame
2. Local registration of all frames
3. Reconstruction I\_t⁽¹⁾
4. New reference R\_t := I\_t⁽¹⁾
5. Repeat until convergence (typically 2–3 iterations)

---

### 5.3 Temporal Smoothing of Warps (mandatory)

For each tile *t*:

```
Â_{f,t} = smooth_time(A_{f−k…f+k,t})
```

Recommended:

- Savitzky–Golay filter
- Robust median filter

---

## 6. Local Quality Metrics

**Implementation:**
- Phase 5 (LOCAL_METRICS): Metrics **before** registration
- Phase 6 (TLR): Post-warp metrics via `compute_post_warp_metrics()`
  - Contrast (Laplacian variance)
  - Background (robust median)
  - SNR proxy

### 6.1 Star Tiles

- log(FWHM)
- Roundness
- Local contrast

### 6.2 Structure Tiles

- E / σ (Edge-to-Noise Ratio)
- Local background

All metrics:

- Robustly normalized (median + MAD)
- Clamped to [−3, +3]

---

## 7. Weights (extended)

Global weight:

```
G_f = exp(Q_f)
```

Local weight:

```
L_{f,t} = exp(Q_{local})
```

Registration quality:

```
R_{f,t} = exp(β · (cc_{f,t} − 1))
```

Effective weight:

```
W_{f,t} = G_f · L_{f,t} · R_{f,t}
```

---

## 8. Tile Reconstruction

For each tile:

```
I_t(p) = Σ_f W_{f,t} · I_f(Â_{f,t}(p)) / Σ_f W_{f,t}
```

Stability rules:

- ΣW < ε → Tile invalid
- < N\_min valid frames → Tile invalid

---

## 9. Overlap-Add

**Implementation (complete):**

```
w_t(p) = hann(p) · ψ(var(Â_{f,t}))
```

with:

```
ψ(v) = exp(-v / (2·σ²))
```

- High warp variance → reduced window weight
- Configurable: `variance_window_sigma` (default: 2.0)

---

## 10. State-Based Clustering

**Implementation (extended):**

State vector per frame:

```
v_f = (G_f, ⟨Q_{tile}⟩, Var(Q_{tile}), ⟨cc⟩, Var(Â), invalid_tile_fraction)
```

Extended metadata from TLR:
- `mean_correlation`: ⟨cc⟩ across all tiles
- `warp_variance`: Var(Â) of translations
- `invalid_tile_fraction`: Fraction of invalid tiles

Clustering:

- k = 15–30 (configurable)
- One synthetic frame per cluster

---

## 11. Final Stacking

- Linear stacking of synthetic frames
- No additional weighting
- No geometric transformation

---

## 12. Validation (v4)

### Local (mandatory)

- FWHM heatmaps
- Warp vector fields
- Tile invalid maps

### Global (secondary)

- SNR distribution
- Background RMS

Abortion criteria:

- < 30% valid tiles
- Large-scale systematic warp patterns

---

## 13. Configuration Parameters (v4)

```yaml
v4:
  iterations: 3                        # Iterative reference refinement
  beta: 5.0                            # Registration quality weight β
  min_valid_tile_fraction: 0.3         # Minimum valid tile fraction
  
  adaptive_tiles:
    enabled: true                      # Enable adaptive refinement
    max_refine_passes: 2               # Maximum refinement passes
    refine_variance_threshold: 0.25    # Variance threshold for splitting
    min_tile_size_px: 64               # Minimum tile size
  
  convergence:
    enabled: false                     # Early stopping on convergence
    epsilon_rel: 1.0e-3                # Relative L2 norm threshold
  
  memory_limits:
    rss_warn_mb: 4096                  # Soft limit (warning)
    rss_abort_mb: 8192                 # Hard limit (abort)
  
  diagnostics:
    enabled: true                      # Generate diagnostic artifacts
    warp_field: true                   # Save warp vector field
    tile_invalid_map: true             # Save invalid tile map
    warp_variance_hist: true           # Save variance statistics

registration:
  mode: local_tiles
  local_tiles:
    ecc_cc_min: 0.2                    # Minimum ECC correlation
    min_valid_frames: 10               # Minimum valid frames per tile
    temporal_smoothing_window: 11      # Savitzky-Golay window (odd)
    variance_window_sigma: 2.0         # ψ(var) scale parameter
```

---

## 14. Core Statement v4

> Methodology v4 replaces global geometry with **locally consistent, temporally smoothed reconstruction**.

It is:

- Correct for Alt/Az & EQ mounts
- Robust against field rotation
- Experimentally maximally flexible
- Scientifically rigorously justified

---

**End of normative specification v4**
