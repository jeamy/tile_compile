# LOCAL_METRICS — Lokale Tile-Metriken und Qualitäts-Scoring

> **C++ Implementierung:** `runner_pipeline.cpp`
> **Phase-Enum:** `Phase::LOCAL_METRICS`

## Übersicht

Phase 8 berechnet für jede Kombination aus Frame × Tile eine Reihe von Qualitätsmetriken. Anschließend werden Tiles in **STAR** oder **STRUCTURE** klassifiziert und ein lokales Gewicht `L_f,t` berechnet, das die **lokale Seeing-Qualität** des jeweiligen Tiles in diesem Frame beschreibt.

```
┌──────────────────────────────────────────────────────┐
│  Für jeden Frame f, für jedes Tile t:                │
│                                                      │
│  1. Tile aus pre-warped Frame extrahieren            │
│  2. calculate_tile_metrics(tile_img) →               │
│     FWHM, Roundness, Contrast, Sharpness,            │
│     Background, Noise, Gradient Energy, Star Count   │
│                                                      │
│  Danach pro Tile t (über alle Frames):               │
│  3. Tile-Typ bestimmen (STAR vs STRUCTURE)           │
│  4. Robuste z-Score Normalisierung (Median + MAD)    │
│  5. Qualitäts-Score Q_f,t berechnen                  │
│  6. L_f,t = exp(clip(Q_f,t))                         │
└──────────────────────────────────────────────────────┘
```

## 1. Tile-Extraktion und Metriken

```cpp
for (size_t fi = 0; fi < frames.size(); ++fi) {
    for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
        const Tile &t = tiles_phase56[ti];
        Matrix2Df tile_img = extract_tile(prewarped_frames[fi], t);
        TileMetrics tm = metrics::calculate_tile_metrics(tile_img);
        local_metrics[fi].push_back(tm);
        local_weights[fi].push_back(1.0f);  // Platzhalter, wird in Schritt 5 überschrieben
    }
}
```

- Tiles werden aus den **pre-warped Frames** extrahiert (bereits registriert in Phase 1)
- `extract_tile()` schneidet den Bildbereich `(x, y, width, height)` aus
- **Leere Tiles** (z.B. am Bildrand): Erhalten Dummy-Metriken und Gewicht 1.0

### Metriken pro Tile

| Metrik | Typ | Beschreibung |
|--------|-----|-------------|
| `fwhm` | float | Full Width Half Maximum der Sterne im Tile |
| `roundness` | float | Sternrundheit (1.0 = perfekt rund) |
| `contrast` | float | Lokaler Kontrast (Laplacian-basiert) |
| `sharpness` | float | Schärfe-Metrik |
| `background` | float | Lokaler Hintergrund (Median) |
| `noise` | float | Lokales Rauschen (MAD-basiert) |
| `gradient_energy` | float | Sobel-Gradient-Magnitude |
| `star_count` | int | Anzahl erkannter Sterne |
| `type` | enum | STAR oder STRUCTURE (wird in Schritt 3 gesetzt) |
| `quality_score` | float | Qualitäts-Score Q_f,t (wird in Schritt 5 gesetzt) |

## 2. Tile-Typ-Klassifikation

```cpp
const int star_thr = cfg.tile.star_min_count;

for (size_t ti = 0; ti < n_tiles; ++ti) {
    // Median Star Count über alle Frames für dieses Tile
    std::vector<float> sc_tmp = star_counts;
    float sc_med = core::median_of(sc_tmp);
    const TileType tile_type = (sc_med >= star_thr) ? TileType::STAR : TileType::STRUCTURE;
}
```

| Tile-Typ | Bedingung | Typisches Szenario |
|----------|-----------|-------------------|
| **STAR** | median(star_count) ≥ `star_min_count` | Sternfeld, offener Bereich |
| **STRUCTURE** | median(star_count) < `star_min_count` | Nebel, dunkle Wolken, Randbereich |

Der Tile-Typ bestimmt, **welche Metrik-Kombination** für das Qualitäts-Scoring verwendet wird.

## 3. Robuste z-Score Normalisierung

Pro Tile werden die Metriken über alle Frames **robust normalisiert**:

```cpp
auto robust_tilde = [&](const std::vector<float> &v, std::vector<float> &out) {
    float med = core::median_of(tmp);
    for (float &x : tmp) x = std::fabs(x - med);
    float mad = core::median_of(tmp);
    float sigma = 1.4826f * mad;
    for (size_t i = 0; i < v.size(); ++i) {
        out[i] = (v[i] - med) / sigma;
    }
};
```

- **Median** als Lagemaß (robust gegen Ausreißer)
- **MAD** (Median Absolute Deviation) als Streuungsmaß
- **1.4826** = Normierungskonstante für Konsistenz mit σ bei Normalverteilung
- Ergebnis: z-Scores mit Median=0, σ≈1

Normalisierte Metriken pro Tile:
- `fwhm_t` (log-transformiert vor Normalisierung)
- `r_t` (Roundness)
- `c_t` (Contrast)
- `b_t` (Background)
- `s_t` (Noise)
- `e_t` (Gradient Energy)

## 4. Qualitäts-Score Berechnung

### STAR-Tiles

```cpp
if (tile_type == TileType::STAR) {
    q = cfg.local_metrics.star_mode.weights.fwhm      * (-fwhm_t[fi])
      + cfg.local_metrics.star_mode.weights.roundness  * (r_t[fi])
      + cfg.local_metrics.star_mode.weights.contrast   * (c_t[fi]);
}
```

```
Q_f,t = w_fwhm · (-FWHM̃_f,t) + w_round · R̃_f,t + w_contrast · C̃_f,t
```

- **Niedriger FWHM** = besser (schärfere Sterne) → negiert
- **Hohe Roundness** = besser (runde Sterne, gutes Tracking)
- **Hoher Contrast** = besser (helle Sterne, gutes Signal)

### STRUCTURE-Tiles

```cpp
else {
    float denom = s_t[fi];
    float ratio = (std::fabs(denom) > eps) ? (e_t[fi] / denom) : 0.0f;
    q = cfg.local_metrics.structure_mode.metric_weight    * ratio
      + cfg.local_metrics.structure_mode.background_weight * (-b_t[fi]);
}
```

```
Q_f,t = w_metric · (Ẽ_f,t / σ̃_f,t) + w_bg · (-B̃_f,t)
```

- **ENR (Energy/Noise Ratio)**: Hohe Gradientenergie relativ zum Rauschen = besser
- **Niedriger Background** = besser → negiert

## 5. Clamping und Gewicht

```cpp
q = clip3(q);  // clip(q, clamp[0], clamp[1])
tm.quality_score = q;
local_weights[fi][ti] = std::exp(q);
```

```
L_f,t = exp(clip(Q_f,t, clamp_lo, clamp_hi))
```

- Clamp verhindert extreme Werte (default: [-3, +3])
- exp() stellt sicher, dass L_f,t > 0
- Bereich von L_f,t: [exp(-3) ≈ 0.05, exp(+3) ≈ 20.1]

## 6. Post-Processing: Median-Qualität und FWHM pro Tile

Nach der Gewichtsberechnung werden Zusammenfassungsstatistiken pro Tile berechnet:

```cpp
// Median Quality Score pro Tile
for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
    for (size_t fi = 0; fi < local_metrics.size(); ++fi)
        qs.push_back(local_metrics[fi][ti].quality_score);
    tile_quality_median[ti] = core::median_of(qs);
    tile_is_star[ti] = (local_metrics[0][ti].type == TileType::STAR) ? 1 : 0;
}

// Median FWHM pro Tile
for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
    for (size_t fi = 0; fi < local_metrics.size(); ++fi)
        fwhms.push_back(local_metrics[fi][ti].fwhm);
    tile_fwhm_median[ti] = core::median_of(fwhms);
}
```

Diese werden für:
- **Validation**: Tile-FWHM-Heatmap
- **Wiener-Denoise-Gating** (falls aktiviert)

## Konfigurationsparameter

| Parameter | Beschreibung | Default |
|-----------|-------------|---------|
| `local_metrics.clamp` | [lo, hi] Clamping-Bereich | [-3, +3] |
| `local_metrics.star_mode.weights.fwhm` | FWHM-Gewicht (STAR) | 0.5 |
| `local_metrics.star_mode.weights.roundness` | Roundness-Gewicht (STAR) | 0.25 |
| `local_metrics.star_mode.weights.contrast` | Contrast-Gewicht (STAR) | 0.25 |
| `local_metrics.structure_mode.metric_weight` | ENR-Gewicht (STRUCTURE) | 0.7 |
| `local_metrics.structure_mode.background_weight` | BG-Gewicht (STRUCTURE) | 0.3 |
| `tile.star_min_count` | Min. Sterne für STAR-Typ | 3 |

## Artifact: `local_metrics.json`

```json
{
  "num_frames": 100,
  "num_tiles": 1200,
  "tile_metrics": [
    [
      {
        "fwhm": 3.2,
        "roundness": 0.85,
        "contrast": 0.045,
        "sharpness": 0.12,
        "background": 1.001,
        "noise": 0.004,
        "gradient_energy": 0.08,
        "star_count": 5,
        "tile_type": "STAR",
        "quality_score": 0.72,
        "local_weight": 2.05
      },
      ...
    ],
    ...
  ]
}
```

Struktur: `tile_metrics[frame_index][tile_index]`

## Nächste Phase

→ **Phase 9: TILE_RECONSTRUCTION — Parallele Tile-Rekonstruktion**