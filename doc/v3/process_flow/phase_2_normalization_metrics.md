# NORMALIZATION + GLOBAL_METRICS — Hintergrund-Normalisierung und globale Gewichtung

> **C++ Implementierung:** `runner_main.cpp` Zeilen 385–717
> **Phase-Enums:** `Phase::NORMALIZATION` (L385–L634), `Phase::GLOBAL_METRICS` (L636–L717)

## Übersicht

Diese beiden Phasen berechnen für jeden Frame die Hintergrund-Normalisierung und daraus abgeleitete globale Qualitätsmetriken. Das Ergebnis sind Normalisierungsfaktoren (`NormalizationScales`) und globale Frame-Gewichte (`G_f`).

```
┌──────────────────────────────────────────────────────┐
│  NORMALIZATION (Phase 2)                             │
│                                                      │
│  Für jeden Frame f:                                  │
│  1. Frame laden (FITS → Matrix2Df)                   │
│  2. Grob-Normalisierung (Median aller Pixel)         │
│  3. Sigma-Clip Background-Maske erstellen            │
│  4. OSC: Kanalgetrennte BG-Schätzung (R,G,B)        │
│     MONO: Einzelne BG-Schätzung                      │
│  5. Scale = 1 / Background                           │
│                                                      │
│  Output: NormalizationScales[N], B_mono/B_r/B_g/B_b  │
└──────────────────────────┬───────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────┐
│  GLOBAL_METRICS (Phase 3)                            │
│                                                      │
│  Für jeden Frame f:                                  │
│  1. Frame laden + normalisieren                      │
│  2. calculate_frame_metrics() → B_f, σ_f, E_f       │
│  3. calculate_global_weights() → G_f                 │
│                                                      │
│  Output: FrameMetrics[N], VectorXf global_weights    │
└──────────────────────────────────────────────────────┘
```

## Phase 2: NORMALIZATION — Detaillierter Ablauf

### Schritt 1: Frame laden

```cpp
auto frame_pair = io::read_fits_float(path);
const Matrix2Df &img = frame_pair.first;
```

Jeder Frame wird einzeln von Disk geladen (Memory-effizient). Die Normalisierung berechnet nur die Skalierungsfaktoren — die eigentliche Normalisierung wird **lazy** bei Bedarf angewendet.

### Schritt 2: Grob-Normalisierung für Masken-Berechnung

```cpp
const float b0 = core::median_of(all);  // Median aller Pixel
const float coarse_scale = (b0 > eps_b) ? (1.0f / b0) : 1.0f;
Matrix2Df coarse_norm = img * coarse_scale;
```

- Median aller Pixel als grober Background-Schätzer
- Division durch Median → grob auf ~1.0 normalisiert
- Nur für die Berechnung der Sigma-Clip-Maske verwendet

### Schritt 3: Sigma-Clipping Background-Maske

```cpp
const cv::Mat1b bg_mask = metrics::build_background_mask_sigma_clip(coarse_cv, 3.0f, 3);
```

- **3σ Sigma-Clipping** mit 3 Iterationen
- Maske markiert Pixel die zum Background gehören (1) vs. Sterne/Objekte (0)
- Wird auf dem grob-normalisierten Bild berechnet

### Schritt 4a: OSC — Kanalgetrennte Background-Schätzung

Bei OSC-Daten wird der Background **pro Bayer-Kanal** geschätzt:

```cpp
int r_row, r_col, b_row, b_col;
image::bayer_offsets(detected_bayer_str, r_row, r_col, b_row, b_col);

// Pixel nach Bayer-Position sortieren, nur Background-Maske berücksichtigen
for (int y = 0; y < img.rows(); ++y) {
    const int py = y & 1;
    for (int x = 0; x < img.cols(); ++x) {
        if (bg_mask(y,x) == 0) continue;
        const int px = x & 1;
        if (py == r_row && px == r_col)      pr_bg.push_back(v);  // Red
        else if (py == b_row && px == b_col) pb_bg.push_back(v);  // Blue
        else                                  pg_bg.push_back(v);  // Green
    }
}

float br = core::median_of(pr_bg);  // Background Red
float bg = core::median_of(pg_bg);  // Background Green
float bb = core::median_of(pb_bg);  // Background Blue
```

- **Bayer-Offsets** bestimmen welche Pixel R, G oder B sind
- Background wird als **Median der maskierten Pixel** pro Kanal berechnet
- **Fallback**: Wenn Median ≤ ε, wird `estimate_background_sigma_clip()` auf alle Kanal-Pixel angewendet
- **Fehler**: Wenn Background für irgendeinen Kanal ≤ ε → Pipeline bricht ab

Skalierungsfaktoren:
```cpp
s.scale_r = 1.0f / br;
s.scale_g = 1.0f / bg;
s.scale_b = 1.0f / bb;
```

### Schritt 4b: MONO — Einzelne Background-Schätzung

```cpp
// Nur Background-maskierte Pixel verwenden
float b = core::median_of(p_bg);
// Fallback: Sigma-Clip auf alle Pixel
if (!(b > eps_b))
    b = core::estimate_background_sigma_clip(all_pixels);
s.scale_mono = 1.0f / b;
```

### Schritt 5: Output-Skalierung vorbereiten

Nach der Normalisierung werden Median-Background-Werte für die spätere Output-Skalierung berechnet:

```cpp
const float output_pedestal = 32768.0f;
const float output_bg_mono = median_finite_positive(B_mono, 1.0f);
const float output_bg_r    = median_finite_positive(B_r, 1.0f);
const float output_bg_g    = median_finite_positive(B_g, 1.0f);
const float output_bg_b    = median_finite_positive(B_b, 1.0f);
```

Diese Werte werden in Phase 10 (STACKING) und Phase 11 (DEBAYER) verwendet, um die normalisierten Daten zurück in physikalische Einheiten zu konvertieren.

### Lazy Normalisierung

Die eigentliche Normalisierung wird **nicht** sofort auf die Frames angewendet. Stattdessen wird die Lambda-Funktion `load_frame_normalized()` verwendet:

```cpp
auto load_frame_normalized = [&](size_t frame_index) -> pair<Matrix2Df, FitsHeader> {
    auto frame_pair = io::read_fits_float(frames[frame_index]);
    Matrix2Df img = frame_pair.first;
    image::apply_normalization_inplace(img, norm_scales[frame_index],
                                       detected_mode, detected_bayer_str, 0, 0);
    return {img, frame_pair.second};
};
```

`apply_normalization_inplace` multipliziert jeden Pixel mit dem entsprechenden Skalierungsfaktor (OSC: kanalgetrennt nach Bayer-Offset, MONO: einheitlich).

## Phase 3: GLOBAL_METRICS — Detaillierter Ablauf

### Frame-Metriken berechnen

```cpp
for (size_t i = 0; i < frames.size(); ++i) {
    auto frame_pair = io::read_fits_float(path);
    Matrix2Df img = frame_pair.first;
    image::apply_normalization_inplace(img, norm_scales[i], ...);
    frame_metrics[i] = metrics::calculate_frame_metrics(img);
}
```

`calculate_frame_metrics()` berechnet pro Frame:

| Metrik | Symbol | Beschreibung | Berechnung |
|--------|--------|-------------|------------|
| **Background** | B_f | Hintergrundniveau | Median nach Normalisierung |
| **Noise** | σ_f | Rausch-Level | Robust σ (MAD-basiert) |
| **Gradient Energy** | E_f | Strukturenergie | Sobel-Gradient Magnitude |
| **Quality Score** | Q_f | Qualitätsindex | Intern berechnet |

### Globale Gewichte berechnen

```cpp
VectorXf global_weights = metrics::calculate_global_weights(
    frame_metrics,
    cfg.global_metrics.weights.background,  // α (default 0.4)
    cfg.global_metrics.weights.noise,       // β (default 0.3)
    cfg.global_metrics.weights.gradient,    // γ (default 0.3)
    cfg.global_metrics.clamp[0],            // clamp_lo (default -3)
    cfg.global_metrics.clamp[1]);           // clamp_hi (default +3)
```

#### Gewichtsberechnung

1. **MAD-Normalisierung** jeder Metrik über alle Frames:
   ```
   x̃_f = (x_f - median(x)) / (1.4826 · MAD(x))
   ```

2. **Qualitäts-Score** als gewichtete Linearkombination:
   ```
   Q_f = α · (-B̃_f) + β · (-σ̃_f) + γ · Ẽ_f
   ```
   - Niedriger Background = besser (negiert)
   - Niedriges Rauschen = besser (negiert)
   - Hohe Gradient-Energie = besser (nicht negiert)

3. **Clamping und Exponential-Mapping**:
   ```
   G_f = exp(clip(Q_f, clamp_lo, clamp_hi))
   ```
   - Clamp verhindert extreme Gewichte
   - exp() stellt sicher, dass alle Gewichte > 0

## Konfigurationsparameter

| Parameter | Beschreibung | Default |
|-----------|-------------|---------|
| `normalization.enabled` | Normalisierung aktivieren (Pflicht) | `true` |
| `global_metrics.weights.background` | α — Background-Gewicht | 0.4 |
| `global_metrics.weights.noise` | β — Noise-Gewicht | 0.3 |
| `global_metrics.weights.gradient` | γ — Gradient-Gewicht | 0.3 |
| `global_metrics.clamp` | [lo, hi] Clamping-Bereich | [-3, +3] |
| `global_metrics.adaptive_weights` | Adaptive Gewichtung | `false` |

**Gewichts-Normierung:** α + β + γ = 1.0 (wird von `cfg.validate()` geprüft)

## Artifact: `normalization.json`

```json
{
  "mode": "OSC",
  "bayer_pattern": "RGGB",
  "B_mono": [0.0, 0.0, ...],
  "B_r": [1234.5, 1230.1, ...],
  "B_g": [1567.2, 1560.8, ...],
  "B_b": [1100.3, 1098.7, ...]
}
```

## Artifact: `global_metrics.json`

```json
{
  "metrics": [
    {
      "background": 1.002,
      "noise": 0.0045,
      "gradient_energy": 0.123,
      "quality_score": 0.85,
      "global_weight": 2.34
    },
    ...
  ],
  "weights": {"background": 0.4, "noise": 0.3, "gradient": 0.3},
  "clamp": [-3.0, 3.0],
  "adaptive_weights": false
}
```

## Fehlerbehandlung

| Fehler | Verhalten |
|--------|-----------|
| Normalisierung disabled | phase_end(error), Pipeline-Abbruch |
| Background ≤ ε (Kanal) | phase_end(error), Pipeline-Abbruch |
| Frame nicht lesbar | phase_end(error), Pipeline-Abbruch |
| Leerer Frame | Warnung, Dummy-Metriken (B=0, σ=0, E=0, Q=1) |

## Nächste Phase

→ **Phase 4: TILE_GRID — Seeing-adaptive Tile-Erzeugung**