# Methodology v4 – Limitations & Future Work

This document complements the **reference implementation of Methodology v4** with an **explicit, reviewer‑suitable analysis of limitations** and a **structured outlook on future work**.

It is explicitly intended to demonstrate **methodological honesty** and to anticipate potential reviewer concerns.

---

## 1. Current Limitations (deliberately accepted)

### 1.1 Local Motion Model is Translation-Based

**Status:** intentionally minimal

- Current model:
  \[
  p' = p + (\Delta x, \Delta y)
  \]
- Locally correct for small tiles
- Approximates rotation and distortion only indirectly

**Impact:**
- For very large tiles or extreme field rotation, warp variance increases
- Currently compensated by **adaptive tile refinement**

**Assessment:**
- Methodologically acceptable
- No correctness error

---

### 1.2 Temporal Warp Smoothing is Median-Based

**Status:** minimalist reference

- Robust median filter
- No explicit motion dynamics

**Impact:**
- Suboptimal smoothing for continuous acceleration

**Assessment:**
- Correct, but conservative

---

### 1.3 Convergence Criterion is Purely Photometric

**Status:** intentionally simple

- Termination based on:
  \[
  \|R_{k} - R_{k-1}\| / \|R_{k}\|
  \]

**Not considered:**
- PSF stability
- Warp variance plateaus

---

### 1.4 Overlap-Add Uses Fixed Window Functions

**Status:** deterministic

- Hann/Hanning windows
- No adaptive edge weighting

**Impact:**
- Local artifacts possible with abrupt changes in warp stability

---

### 1.5 No Explicit Modeling of PSF Anisotropy

**Status:** out of scope

- PSF considered only implicitly through local quality metrics

---

## 2. Explicitly Unaddressed Problem Classes

These points are **deliberately excluded**:

- Absolute astrometric accuracy
- Photometric calibration to catalog level
- Global distortion models
- Real-time processing

Methodology v4 is a **reconstruction pipeline, not an astrometry pipeline**.

---

## 3. Future Work – Short to Medium Term

### 3.1 Extended Local Motion Models

- Affine models with strong regularization
- Local Jacobian estimation
- Adaptive model selection per tile

---

### 3.2 Physically Motivated Warp Smoothing

- Savitzky–Golay filters
- Kalman filters with smooth field rotation assumptions

---

### 3.3 Adaptive Overlap Windows

- Window weight dependent on warp variance
- Reduction of edge artifacts

---

### 3.4 Tile Failure Taxonomy

- Explicit error codes per tile
- Better diagnostics and statistical evaluation

---

### 3.5 GPU-Accelerated Tile Processing

- GPU-based local registration
- Tile batching
- **Without abandoning the streaming model**

---

## 4. Long-Term Perspective

Methodology v4 builds a bridge between:

- Classical lucky imaging
- Multi-frame super-resolution
- Software-based adaptive optics

In the long term, it opens the possibility of a **purely software-based field-AO approximation** for amateur and semi-professional astronomy.

---

## 5. Reviewer Positioning (Explicit)

> *The presented framework deliberately trades global geometric consistency for local physical validity. This choice is not a limitation of the implementation, but a methodological decision aligned with the underlying observational conditions.*

---

**End: Limitations & Future Work (Methodology v4)**
