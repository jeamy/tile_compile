# Referenz‑Pseudocode pro GitHub‑Issue (v4‑konform)

**Ziel:** Jede Issue‑Nummer erhält eine **normative, sprachunabhängige Referenz‑Implementierung**.  
**Status:** Methodisch vollständig, nicht optimiert, deterministisch.

---

## Issue 1 – Globale lineare Normalisierung

```text
for each frame f:
    mask = build_background_mask(frame f)
    B_f = median(frame f pixels where mask == background)
    assert B_f > epsilon
    frame_norm = frame / B_f
    store frame_norm
```

---

## Issue 2 – Robuste Hintergrundschätzung B_f

```text
pixels = frame pixels excluding object mask
B_f = median(pixels)
```

Optional robuster:

```text
B_f = biweight_location(pixels)
```

---

## Issue 3 – Globale Rauschschätzung σ_f

```text
pixels = frame_norm pixels excluding object mask
sigma_f = 1.4826 * MAD(pixels)
```

---

## Issue 4 – Gradientenergie E_f

```text
I = frame_norm
Gx, Gy = sobel(I)
E_f = median(Gx^2 + Gy^2)
```

---

## Issue 5 – MAD‑Normalisierung globaler Metriken

```text
for metric in {B, sigma, E}:
    med = median(metric over all frames)
    mad = MAD(metric over all frames)
    metric_tilde = (metric - med) / (1.4826 * mad)
```

---

## Issue 6 – Globaler Qualitätsindex Q_f und G_f

```text
Q_f = alpha*(-B_tilde) + beta*(-sigma_tilde) + gamma*(E_tilde)
Q_f = clip(Q_f, -3, +3)
G_f = exp(Q_f)
```

---

## Issue 7 – Seeing‑adaptive Tile‑Geometrie

```text
F = median(FWHM over valid stars)
T = clip(32 * F, 64, min(W,H)/6)
overlap = 0.25 * T
step = T - overlap
```

---

## Issue 8 – Stern‑ vs. Struktur‑Tiles

```text
if star_count(tile) >= threshold:
    tile_type = STAR
else:
    tile_type = STRUCTURE
```

---

## Issue 9 – Lokale Qualitätsindizes Q_local

### Stern‑Tiles

```text
Q_star = 0.6*(-log(FWHM_tilde)) + 0.2*R_tilde + 0.2*C_tilde
Q_star = clip(Q_star, -3, +3)
L_ft = exp(Q_star)
```

### Struktur‑Tiles

```text
Q_struct = 0.7*(E_tilde / sigma_tilde) - 0.3*B_tilde
Q_struct = clip(Q_struct, -3, +3)
L_ft = exp(Q_struct)
```

---

## Issue 10 – Effektive Gewichtung W_f,t

```text
W_f,t = G_f * L_f,t
```

---

## Issue 11 – Stabile Tile‑Rekonstruktion

```text
numerator = sum_f(W_f,t * I_f(p))
denominator = sum_f(W_f,t)
if denominator < epsilon:
    pixel = mean(I_f(p))
else:
    pixel = numerator / denominator
```

---

## Issue 12 – Zustandsvektor v_f

```text
v_f = (
    G_f,
    mean(Q_local over tiles),
    var(Q_local over tiles),
    B_f,
    sigma_f
)
```

---

## Issue 13 – Zustands‑Clusterung

```text
X = standardize(all v_f)
clusters = kmeans(X, k)
```

Fallback:

```text
clusters = quantiles(G_f, k)
```

---

## Issue 14 – Synthetische Qualitätsframes

```text
for each cluster c:
    frames_c = frames with cluster == c
    synthetic_frame = tile_reconstruction(frames_c)
```

---

## Issue 15 – Validierung & Abbruch

```text
if median_fwhm_improvement < threshold:
    abort_run()
if tile_weight_variance < threshold:
    abort_run()
else:
    accept_result()
```

---

## Abschluss

Dieser Pseudocode ist **normativ**: jede v4‑konforme Implementierung muss äquivalentes Verhalten zeigen, unabhängig von Sprache oder Optimierung.

---

