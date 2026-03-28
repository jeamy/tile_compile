# NORMALIZATION + GLOBAL_METRICS — Lineare Normierung und globale Gewichtung

> **C++ Implementierung:** `runner_pipeline.cpp`
> **Phase-Enums:** `Phase::NORMALIZATION` (4), `Phase::GLOBAL_METRICS` (5)

## Übersicht

Diese beiden Phasen berechnen für jeden Frame die lineare Normierung gemäß Methodik `v3.3.9` und daraus abgeleitete globale Qualitätsmetriken. Der verbindliche Ablauf trennt den **additiven Hintergrund** `B_{f,c}` von der **multiplikativen photometrischen Skala** `P_{f,c}`. Das Ergebnis sind Normierungsparameter und globale Frame-Gewichte (`G_f`).

```
┌──────────────────────────────────────────────────────┐
│  NORMALIZATION (Phase 4)                             │
│                                                      │
│  Für jeden Frame f:                                  │
│  1. Frame laden (FITS → Matrix2Df)                   │
│  2. Additives Hintergrundniveau B_f schätzen         │
│  3. Hintergrundsubtraktion J = I_raw - B_f           │
│  4. Photometrische Skala P_f deterministisch         │
│     bestimmen oder P_f = 1 Fallback                  │
│  5. I = J / P_f                                      │
│                                                      │
│  Output: NormalizationScales[N], B_mono/B_r/B_g/B_b  │
└──────────────────────────┬───────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────┐
│  GLOBAL_METRICS (Phase 5)                            │
│                                                      │
│  Für jeden Frame f:                                  │
│  1. Frame laden + normalisieren                      │
│  2. calculate_frame_metrics() → B_f, σ_f, E_f        │
│  3. calculate_global_weights() → G_f                 │
│                                                      │
│  Output: FrameMetrics[N], VectorXf global_weights    │
└──────────────────────────────────────────────────────┘
```

## Phase 4: NORMALIZATION — Detaillierter Ablauf

Die in älteren Beschreibungen verwendete Kurzform "`Scale = 1 / Background`" ist für `v3.3.9` **nicht mehr korrekt**. Der bindende Kern ist:

1. `B_{f,c}` additiv schätzen,
2. `J_{f,c} = I_{f,c}^{raw} - B_{f,c}`,
3. `P_{f,c}` photometrisch bestimmen,
4. `I_{f,c} = J_{f,c} / max(P_{f,c}, eps_scale)`.

### Schritt 1: Frame laden

```cpp
auto frame_pair = io::read_fits_float(path);
const Matrix2Df &img = frame_pair.first;
```

Jeder Frame wird einzeln von Disk geladen (Memory-effizient). Die Normalisierung berechnet nur die Skalierungsfaktoren — die eigentliche Normalisierung wird **lazy** bei Bedarf angewendet.

### Schritt 2: Hintergrundmaske und additive Hintergrundschätzung

```cpp
const float b0 = core::median_of(all);  // grobe additive Hintergrundschätzung
Matrix2Df coarse_centered = img.array() - b0;
```

- Median aller Pixel als grober additiver Background-Schätzer
- nur zur Vorbereitung der Background-Maske / robusten Hintergrundschätzung
- keine normative photometrische Skalierung an dieser Stelle

### Schritt 3: Sigma-Clipping Background-Maske

```cpp
const cv::Mat1b bg_mask = metrics::build_background_mask_sigma_clip(coarse_cv, 3.0f, 3);
```

- **3σ Sigma-Clipping** mit 3 Iterationen
- Maske markiert Pixel, die zum Background gehören (1) vs. Sterne/Objekte (0)
- die Maske dient der robusteren Schätzung von `B_{f,c}` und ggf. von lokalen Rauschgrößen

### Schritt 4a: OSC — Kanalgetrennte additive Hintergrundschätzung

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
- Background wird als **Median der maskierten Pixel** pro Kanal geschätzt
- die Werte `br`, `bg`, `bb` sind additive Pedestals `B_r`, `B_g`, `B_b`
- **Fallback**: Wenn die robuste Schätzung instabil ist, darf deterministisch auf eine konservative Ersatzschätzung zurückgefallen werden

Danach folgt kanalweise:

```cpp
J_r = I_r_raw - B_r;
J_g = I_g_raw - B_g;
J_b = I_b_raw - B_b;
```

Die photometrische Skalierung `P_{f,c}` ist konzeptionell **ein separater Schritt** und nicht identisch mit `1 / B_{f,c}`.

### Schritt 4b: MONO — Einzelne additive Hintergrundschätzung

```cpp
// Nur Background-maskierte Pixel verwenden
float b = core::median_of(p_bg);
// Fallback: Sigma-Clip auf alle Pixel
if (!std::isfinite(b))
    b = core::estimate_background_sigma_clip(all_pixels);
// danach: J = I_raw - b
```

### Schritt 5: Photometrische Skala und Output-Skalierung vorbereiten

Nach der additiven Hintergrundschätzung werden für die spätere Output-Skalierung robuste Referenzwerte gespeichert. Zusätzlich ist in `v3.3.9` eine deterministische photometrische Skala `P_{f,c}` vorgesehen:

- bevorzugt aus Ensemble-Sternflussen,
- alternativ aus Belichtungszeit-Verhältnissen,
- sonst deterministisch `P_{f,c} = 1`.

Wichtig: `P_{f,c}` darf nicht ausschließlich aus `B_{f,c}` abgeleitet werden.

Für die Ausgabe werden weiterhin robuste Hintergrund-Referenzwerte für die Rückskalierung gespeichert:

```cpp
const float output_pedestal = 32768.0f;
const float output_bg_mono = median_finite_positive(B_mono, 1.0f);
const float output_bg_r    = median_finite_positive(B_r, 1.0f);
const float output_bg_g    = median_finite_positive(B_g, 1.0f);
const float output_bg_b    = median_finite_positive(B_b, 1.0f);
```

Diese Werte werden in Phase 12 (STACKING) und Phase 13 (DEBAYER) verwendet, um die linearen Daten deterministisch in den Output-Skalierungsbereich zu konvertieren.

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

Konzeptionell gilt für `v3.3.9`: `apply_normalization_inplace` muss additive Hintergrundsubtraktion und photometrische Skalierung getrennt behandeln. Eine reine Multiplikation mit `1/background` ist methodisch nicht mehr die normative Beschreibung.

## Phase 5: GLOBAL_METRICS — Detaillierter Ablauf

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
| **Background** | B_f | Hintergrundniveau | additive Schätzung vor photometrischer Skalierung |
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
   G_f = exp(k_global · clip(Q_f, clamp_lo, clamp_hi))
   ```
   - Clamp verhindert extreme Gewichte
   - exp() stellt sicher, dass alle Gewichte > 0
   - in `v3.3.9` ist `k_global` der explizite globale Gewichtsskalenfaktor (Default 1.0)

## Konfigurationsparameter

| Parameter | Beschreibung | Default |
|-----------|-------------|---------|
| `normalization.enabled` | Normalisierung aktivieren (Pflicht) | `true` |
| `global_metrics.weights.background` | α — Background-Gewicht | 0.4 |
| `global_metrics.weights.noise` | β — Noise-Gewicht | 0.3 |
| `global_metrics.weights.gradient` | γ — Gradient-Gewicht | 0.3 |
| `global_metrics.clamp` | [lo, hi] Clamping-Bereich | [-3, +3] |
| `global_metrics.adaptive_weights` | Adaptive Gewichtung | `false` |
| `eps_scale` | Schutz gegen Division durch sehr kleine photometrische Skalen | `1e-6` |

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
| Kein stabiler additiver Hintergrund / keine stabile photometrische Skala | phase_end(error), Pipeline-Abbruch oder deterministischer Fallback |
| Frame nicht lesbar | phase_end(error), Pipeline-Abbruch |
| Leerer Frame | Warnung, Dummy-Metriken (B=0, σ=0, E=0, Q=1) |

## Nächste Phase

→ **Phase 6: TILE_GRID — Seeing-adaptive Tile-Erzeugung**
