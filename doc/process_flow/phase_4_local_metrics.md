# LOCAL_METRICS — Lokale Tile-Metriken und Qualitäts-Scoring

> **C++ Implementierung:** `runner_pipeline.cpp`
> **Phase-Enum:** `Phase::LOCAL_METRICS`

## Übersicht

Phase 8 berechnet für jede Kombination aus Frame × Tile eine Reihe von Qualitätsmetriken. In `v3.3.9` werden Tiles **nicht mehr hart** in STAR oder STRUCTURE umgeschaltet, sondern über einen weichen STAR-Support-Faktor `eta_t` kontinuierlich zwischen beiden lokalen Modellen geblendet. Daraus wird das lokale Gewicht `L_{f,t}` berechnet.

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
│  3. STAR-Support eta_t bestimmen                     │
│  4. Robuste z-Score Normalisierung (Median + MAD)    │
│  5. Q_star / Q_struct / Q_blend berechnen            │
│  6. L_f,t = exp(k_local · clip(Q_f,t))               │
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
| `type` | legacy/info | historischer Typindikator, nicht mehr der bindende Kernschalter |
| `quality_score` | float | Qualitäts-Score Q_f,t (wird in Schritt 5 gesetzt) |

## 2. Weicher STAR-Support statt harter Klassifikation

```cpp
const int star_soft = std::max(cfg.tile.star_soft_count, 1);
for (size_t ti = 0; ti < n_tiles; ++ti) {
    float sc_med = core::median_of(star_counts);
    float eta_t = std::clamp(sc_med / static_cast<float>(star_soft), 0.0f, 1.0f);
}
```

Semantik:

| Bereich | Bedingung | Bedeutung |
|---------|-----------|-----------|
| `eta_t = 1` | starke Sternunterstützung | STAR-dominiert |
| `0 < eta_t < 1` | gemischt | kontinuierliches Blending |
| `eta_t = 0` | keine Sternunterstützung | STRUCTURE-dominiert |

Ein hartes Umschalten ist in `v3.3.9` nicht mehr die normative Beschreibung.

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

Zusätzlich gilt in `v3.3.9`: canvas-ungültige Pixel dürfen nicht in lokale Metrikakkumulatoren eingehen. Wenn ein Tile zu wenige canvas-gültige Pixel enthält, werden dessen lokale Metriken als nicht verfügbar behandelt und das Tile fällt deterministisch auf konservative Defaults bzw. auf das verfügbare Teilmodell zurück.

## 4. Qualitäts-Score Berechnung

### STAR-Tiles

```cpp
q_star = 0.6f * (-fwhm_t[fi])
       + 0.2f * (r_t[fi])
       + 0.2f * (c_t[fi]);
```

```
Q_f,t^star = 0.6 · (-FWHM̃_f,t) + 0.2 · R̃_f,t + 0.2 · C̃_f,t
```

- **Niedriger FWHM** = besser (schärfere Sterne) → negiert
- **Hohe Roundness** = besser (runde Sterne, gutes Tracking)
- **Hoher Contrast** = besser (helle Sterne, gutes Signal)

### STRUCTURE-Komponente

```cpp
float denom = s_t[fi];
float ratio = (std::fabs(denom) > eps) ? (e_t[fi] / denom) : 0.0f;
q_struct = 0.7f * ratio + 0.3f * (-b_t[fi]);
```

```
Q_f,t^struct = 0.7 · (Ẽ_f,t / σ̃_f,t) + 0.3 · (-B̃_f,t)
```

- **ENR (Energy/Noise Ratio)**: Hohe Gradientenergie relativ zum Rauschen = besser
- **Niedriger Background** = besser → negiert

### Weiches Blending und Regularisierung

In `v3.3.9` wird daraus zuerst

```
Q_blend = eta_t · Q_star + (1 - eta_t) · Q_struct
```

gebildet.

Danach darf der lokale Score räumlich regularisiert werden. Die neue normative Semantik ist:

- Confidence-aware Gewichtung über `U_{f,t}`
- edge-aware Nachbarschaftsaffinitäten
- kein Glätten, wenn die Affinitätssumme unter `eps_affinity` fällt

## 5. Clamping und Gewicht

```cpp
q = clip3(q_blend_or_reg);  // clip(q, clamp[0], clamp[1])
tm.quality_score = q;
local_weights[fi][ti] = std::exp(k_local * q);
```

```
L_f,t = exp(k_local · clip(Q_f,t, clamp_lo, clamp_hi))
```

- Clamp verhindert extreme Werte (default: [-3, +3])
- exp() stellt sicher, dass L_f,t > 0
- Bereich von L_f,t bei `k_local=1`: [exp(-3) ≈ 0.05, exp(+3) ≈ 20.1]
- `k_local` ist in `v3.3.9` der explizite lokale Gewichtsskalenfaktor

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
| `tile.star_soft_count` | Soft-Count für STAR-Support `eta_t` | entspricht typischerweise `tile.star_min_count` |
| `local_metrics.spatial_regularization.eps_affinity` | Affinitäts-Guard | `1e-6` |
| `k_local` | Lokaler Exponential-Skalierungsfaktor | `1.0` |

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
