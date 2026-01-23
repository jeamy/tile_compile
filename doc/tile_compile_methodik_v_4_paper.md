# Tile‑basierte Qualitätsrekonstruktion für DSO – Methodik v4

## A Local, Iterative, Memory‑Safe Reconstruction Framework

---

## Abstract

We present **Methodik v4**, an experimental reference framework for the reconstruction of deep‑sky astronomical images from large sets of short‑exposure frames. In contrast to classical stacking pipelines, Methodik v4 **eliminates global image registration entirely** and replaces it with a **tile‑local, temporally consistent reconstruction operator**. The method is robust against field rotation (Alt/Az mounts, polar regions), scalable to thousands of frames, and memory‑safe by construction through streaming‑based processing. We describe the algorithmic architecture, mathematical model, validation strategy, and practical implications for modern high‑frame‑count astrophotography.

---

## 1. Motivation and Limitations of Classical Pipelines

Traditional astrophotography pipelines rely on the assumption that all frames can be mapped into a **single global coordinate system** via translation, rotation, or affine transforms. This assumption breaks down in several common scenarios:

- Alt/Az mounts without derotators
- Observations near the celestial poles
- Long acquisition sequences with time‑dependent field rotation
- Local seeing variability across wide fields

Global registration becomes either numerically unstable or physically invalid, leading to spatially varying residuals, degraded point spread functions (PSFs), and systematic artefacts.

**Methodik v4 discards the global coordinate assumption altogether.**

---

## 2. Core Principle of Methodik v4

> *There exists no globally consistent geometry.*

Instead, each spatial region is reconstructed **locally**, using only the information that is physically valid at that location and time.

Key consequences:

- Registration is **tile‑local**, not global
- Registration is **integrated into reconstruction**, not a preprocessing step
- All quality metrics are evaluated **after local geometric correction**
- The pipeline is inherently robust to spatially and temporally varying distortions

---

## 3. High‑Level Architecture

### 3.1 Conceptual Pipeline

```
Input Frames (on disk)
        │
        ▼
Global Coarse Normalization
        │
        ▼
Initial Tile Grid
        │
        ▼
┌───────────────────────────────┐
│  TileProcessor (per Tile)     │
│                               │
│  • Iterative reference build  │
│  • Local registration         │
│  • Temporal warp smoothing    │
│  • Post‑warp quality metrics  │
│  • Weighted accumulation      │
└───────────────────────────────┘
        │
        ▼
Adaptive Tile Refinement (optional)
        │
        ▼
Overlap‑Add Reconstruction
        │
        ▼
State‑based Clustering
        │
        ▼
Synthetic Frames
        │
        ▼
Final Linear Stack
```

There is **exactly one active execution path**. Legacy global‑registration pipelines are not part of Methodik v4.

---

## 4. TileProcessor: The Central Operator

The **TileProcessor** is the fundamental unit of computation. All algorithmic complexity is localized here.

### 4.1 Responsibilities

For a single tile *t*:

- read tile windows from disk (streaming)
- build an initial reference from the median of warped tiles
- iteratively refine the reference
- estimate local motion per frame
- smooth motion temporally
- compute post‑warp quality metrics
- accumulate contributions with physically motivated weights

No global state is required.

---

## 5. Mathematical Model

### 5.1 Local Motion Model

Minimal model (default):

\[
\mathbf{p}' = \mathbf{p} + (\Delta x, \Delta y)
\]

This approximation is valid locally even under strong field rotation.

---

### 5.2 Iterative Reference Update

For iteration *k*:

\[
R_t^{(k+1)}(p) = \frac{\sum_f W_{f,t}^{(k)} \, I_f(A_{f,t}^{(k)}(p))}{\sum_f W_{f,t}^{(k)}}
\]

Iterations continue until either:

- a fixed iteration count is reached, or
- the relative change of the reference falls below a threshold.

---

### 5.3 Effective Weighting

\[
W_{f,t} = G_f \cdot L_{f,t} \cdot R_{f,t}
\]

Where:

- \(G_f\): global atmospheric quality
- \(L_{f,t}\): local structural or stellar quality
- \(R_{f,t}\): registration confidence derived from ECC correlation

---

## 6. Adaptive Tile Refinement

Tiles are not assumed to be optimal a priori.

After an initial reconstruction pass, tiles exhibiting high warp variance are **recursively subdivided**, subject to a minimum tile size constraint. This allows the method to locally increase spatial resolution where the motion model begins to break down.

---

## 7. Memory‑Safe Streaming Architecture

Methodik v4 is designed for **bounded memory usage**.

### Key properties:

- Frames are never fully loaded into RAM
- Only tile windows are read via FITS memory mapping
- Memory complexity is independent of frame count

\[
\mathcal{O}(w_t \cdot h_t \cdot k)
\]

where *k* is the number of iterations.

---

## 8. Validation and Diagnostics

Validation is **local by design**.

Mandatory diagnostic artefacts:

- local warp vector fields
- tile invalidity maps
- warp variance histograms

Global metrics (e.g. median FWHM) are retained only as secondary diagnostics.

---

## 9. Relation to Existing Approaches

Methodik v4 is conceptually related to:

- lucky imaging
- multi‑frame super‑resolution
- software‑based adaptive optics

However, it differs in its **explicit rejection of global geometry** and its integration of registration, quality assessment, and reconstruction into a single local operator.

---

## 10. Conclusion

Methodik v4 replaces the classical notion of image stacking with a **local, iterative reconstruction paradigm**. By eliminating global registration and enforcing memory‑safe streaming, it enables robust, physically consistent reconstruction in regimes where traditional pipelines fail.

The framework is intended as an **experimental reference implementation**, prioritizing correctness, transparency, and methodological clarity over computational efficiency.

---

*End of paper‑ready Methodik v4 description.*

