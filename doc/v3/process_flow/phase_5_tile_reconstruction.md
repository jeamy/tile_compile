# TILE_RECONSTRUCTION — Parallele gewichtete Tile-Rekonstruktion

> **C++ Implementierung:** `runner_pipeline.cpp`
> **Phase-Enum:** `Phase::TILE_RECONSTRUCTION`

## Übersicht

Phase 9 ist das **Herzstück der Pipeline**. Jedes Tile wird separat rekonstruiert als gewichtetes Mittel über alle Frames, wobei das effektive Gewicht `W_f,t = G_f × L_f,t` die Frame-Qualität (global) und die lokale Tile-Qualität kombiniert. Danach folgt die normative **Tile-Normalisierung vor OLA** sowie eine **Boundary-Diagnostik** der tatsächlichen OLA-Eingangstiles. Erst dann werden die Tiles mittels **Hanning-Overlap-Add** zu einem Gesamtbild zusammengefügt.

```
┌──────────────────────────────────────────────────────┐
│  Für jedes Tile t (parallel mit N Threads):          │
│                                                      │
│  1. Tile aus jedem pre-warped Frame extrahieren      │
│  2. Effektives Gewicht W_f,t = G_f × L_f,t           │
│  3. Gewichtetes Mittel:                              │
│     tile_rec = Σ W_f,t · tile_f / Σ W_f,t            │
│  4. Post-Metriken (Contrast, BG, SNR)                │
│  5. Tile-Normalisierung vor OLA                      │
│  6. Boundary-Diagnostik der Nachbar-Tiles            │
│  7. Normalisiertes Tile → Hanning-Overlap-Add        │
│                                                      │
│  Danach:                                             │
│  8. recon(y,x) /= weight_sum(y,x)                    │
└──────────────────────────────────────────────────────┘
```

## 1. Parallele Tile-Verarbeitung

```cpp
const int prev_cv_threads = cv::getNumThreads();
cv::setNumThreads(1);  // Verhindert OpenCV Thread-Contention

int parallel_tiles = 4;
int cpu_cores = std::thread::hardware_concurrency();
if (parallel_tiles > cpu_cores) parallel_tiles = cpu_cores;

std::vector<std::thread> workers;
std::atomic<size_t> next_tile{0};
for (int w = 0; w < parallel_tiles; ++w) {
    workers.emplace_back([&]() {
        while (true) {
            size_t ti = next_tile.fetch_add(1);
            if (ti >= tiles_phase56.size()) break;
            process_tile(ti);
        }
    });
}
```

- **Worker-Pool**: Bis zu `parallel_tiles` std::threads (default 4, capped auf CPU-Cores)
- **Work-Stealing**: Atomarer Tile-Index-Counter
- **OpenCV-Threads deaktiviert**: `cv::setNumThreads(1)` verhindert verschachtelte Parallelisierung
- **Thread-Safety**: `recon_mutex` schützt den globalen Overlap-Add-Accumulator

## 2. Tile-Extraktion und Gewichtung

```cpp
auto process_tile = [&](size_t ti) {
    const Tile &t = tiles_phase56[ti];

    for (size_t fi = 0; fi < frames.size(); ++fi) {
        Matrix2Df tile_img = extract_tile(prewarped_frames[fi], t);
        if (tile_img.rows() != t.height || tile_img.cols() != t.width) continue;

        warped_tiles.push_back(tile_img);
        float G_f = global_weights[fi];
        float L_ft = local_weights[fi][ti];
        weights.push_back(G_f * L_ft);
    }
};
```

- Tiles werden aus **pre-warped Frames** extrahiert (Phase 1)
- Nur Tiles mit korrekten Dimensionen werden akzeptiert
- **Effektives Gewicht**: `W_f,t = G_f × L_f,t`
  - `G_f`: Globales Frame-Gewicht (Phase 5)
  - `L_f,t`: Lokales Tile-Gewicht (Phase 8)

## 3. Gewichtetes Mittel

```cpp
float wsum = 0.0f;
for (float w : weights) wsum += w;
if (wsum <= 0.0f) wsum = 1.0f;

Matrix2Df tile_rec = Matrix2Df::Zero(t.height, t.width);
for (size_t i = 0; i < warped_tiles.size(); ++i) {
    tile_rec += warped_tiles[i] * (weights[i] / wsum);
}
```

```
tile_t = Σ_f (W_f,t · tile_f,t) / Σ_f W_f,t
```

- Gewichtete Summe, normiert durch Gesamtgewicht
- Bei `wsum = 0`: Fallback auf gleichmäßige Gewichtung (1.0)
- Alle Frames werden verwendet (v3: keine Frame-Selektion)

## 4. Post-Warp-Metriken

```cpp
auto [contrast, background, snr] = compute_post_warp_metrics(tile_rec);
tile_post_contrast[ti] = contrast;
tile_post_background[ti] = background;
tile_post_snr[ti] = snr;
```

| Metrik | Berechnung | Bedeutung |
|--------|-----------|-----------|
| **Contrast** | Varianz des Laplacian | Struktur-Detailgehalt |
| **Background** | Median aller Pixel | Hintergrundniveau |
| **SNR** | (P99 − Median) / MAD | Signal-to-Noise Proxy |

Diese Metriken werden im Artifact `tile_reconstruction.json` gespeichert und in der Report-Heatmap visualisiert.

## 5. Boundary-Diagnostik vor OLA

Die tatsächlich in OLA eingehenden Tiles werden paarweise über ihre realen Overlaps verglichen, ohne das Ergebnis zu verändern.

Für jedes benachbarte Tile-Paar werden auf dem normalisierten OLA-Eingang unter anderem bestimmt:

- `mean_abs_diff`
- `p95_abs_diff`
- `mean_signed_diff`
- `sample_count`

Damit wird gemessen, wie stark benachbarte Tiles bereits **vor** dem Hanning-Overlap-Add auseinanderlaufen.

### 5.1 Zusätzliche Paar-Metadaten

Neben den direkten Overlap-Differenzen werden pro Nachbarpaar auch tilebezogene Abweichungen zusammengefasst:

- Differenz der `tile_valid_counts`
- Differenz der `tile_post_background`-Werte
- Differenz der `tile_post_snr_proxy`-Werte
- Differenz der `tile_mean_correlations`
- Fallback-Mismatch (`fallback_used` links/rechts unterschiedlich)

Diese Diagnose erklärt oft besser als ein reiner Bildvergleich, warum sichtbare Kachelgrenzen entstehen.

### 5.2 Nicht-invasive Diagnostik

Die Boundary-Diagnostik ist bewusst **nicht-invasiv**:

- keine Änderung der rekonstruierten Tiles
- keine zusätzliche Tile-Korrektur
- keine Rückkopplung in Gewichte, Clipping oder OLA

Damit bleibt die lineare Rekonstruktionssemantik unverändert.

## 6. Hanning-Overlap-Add

```cpp
for (int yy = 0; yy < tile.rows(); ++yy) {
    for (int xx = 0; xx < tile.cols(); ++xx) {
        float win = hann_y[yy] * hann_x[xx];
        recon(iy, ix) += tile(yy, xx) * win;
        weight_sum(iy, ix) += win;
    }
}
```

### Hanning-Fenster (1D)

```cpp
w[i] = 0.5 × (1 − cos(2π × i / (n−1)))
```

- **2D separabel**: `win(y,x) = hann_y[y] × hann_x[x]`
- Werte am Rand ≈ 0, in der Mitte = 1
- **Overlap-Add**: Überlappende Tiles summieren sich zu ~1.0

### Normalisierung

```cpp
for (int i = 0; i < recon.size(); ++i) {
    float ws = weight_sum.data()[i];
    if (ws > 1.0e-12f) {
        recon.data()[i] /= ws;
    } else {
        recon.data()[i] = first_img.data()[i];  // Fallback: Referenz-Frame
    }
}
```

- Division durch akkumulierte Fenstergewichte
- **Fallback**: Pixel ohne Tile-Abdeckung (z.B. schmale Ränder) → Referenz-Frame-Pixel

## Datenstrukturen

| Variable | Typ | Beschreibung |
|----------|-----|-------------|
| `recon` | Matrix2Df | Rekonstruiertes Gesamtbild (W×H) |
| `weight_sum` | Matrix2Df | Akkumulierte Fenstergewichte |
| `reconstructed_tiles[ti]` | Matrix2Df | Fertig rekonstruierte Einzeltiles vor OLA |
| `tile_valid_counts[ti]` | int | Anzahl gültiger Frames pro Tile |
| `tile_mean_correlations[ti]` | float | Mittlere Korrelation (=1.0, global warp) |
| `tile_post_contrast[ti]` | float | Post-Contrast pro Tile |
| `tile_post_background[ti]` | float | Post-Background pro Tile |
| `tile_post_snr[ti]` | float | Post-SNR pro Tile |
| `tile_norm_bg_*[ti]` | float | Tile-Background für die normative Vor-OLA-Normalisierung |
| `tile_norm_scale[ti]` | float | gemeinsamer Tile-Scale-Faktor für die Vor-OLA-Normalisierung |

## Artifact: `tile_reconstruction.json`

```json
{
  "num_frames": 100,
  "num_tiles": 1200,
  "tile_boundary_analysis_uses_common_canvas_mask": true,
  "tile_boundary_raw_analysis_input": "pre_ola_raw",
  "tile_boundary_normalized_analysis_input": "pre_ola_normalized",
  "tile_boundary_analysis_input": "pre_ola_normalized",
  "tile_boundary_raw_pair_mean_abs_diff_p95": 0.0048,
  "tile_boundary_normalized_pair_mean_abs_diff_p95": 0.0064,
  "tile_boundary_pair_count": 2200,
  "tile_boundary_observation_count": 380,
  "tile_boundary_sample_count": 184000,
  "tile_boundary_pair_mean_abs_diff_mean": 0.0021,
  "tile_boundary_pair_mean_abs_diff_p95": 0.0064,
  "tile_boundary_post_background_delta_p95_abs": 0.0017,
  "tile_boundary_post_snr_delta_p95_abs": 8.3,
  "tile_valid_counts": [100, 100, 99, ...],
  "tile_norm_bg_r": [1.001, 1.004, ...],
  "tile_norm_bg_g": [1.002, 1.005, ...],
  "tile_norm_bg_b": [0.998, 1.001, ...],
  "tile_norm_scale": [0.98, 1.01, ...],
  "tile_mean_correlations": [1.0, 1.0, 1.0, ...],
  "tile_post_contrast": [0.0012, 0.0015, ...],
  "tile_post_background": [1.001, 1.002, ...],
  "tile_post_snr_proxy": [45.2, 38.7, ...],
  "tile_boundary_raw_top_pairs": [
    {
      "lhs_index": 17,
      "rhs_index": 18,
      "sample_count": 2048,
      "mean_abs_diff": 0.0068
    }
  ],
  "tile_boundary_top_pairs": [
    {
      "lhs_index": 17,
      "rhs_index": 18,
      "lhs_row": 1,
      "lhs_col": 4,
      "rhs_row": 1,
      "rhs_col": 5,
      "sample_count": 2048,
      "mean_abs_diff": 0.0091,
      "p95_abs_diff": 0.0174
    }
  ]
}
```

Die Boundary-Diagnostik verwendet nur Pixel innerhalb der `COMMON_OVERLAP`-/Canvas-Maske. Ausmaskierte Canvas-Zonen werden bei den Boundary-Metriken nicht als Nullwerte mitgezählt.

## Fehlerbehandlung

| Situation | Verhalten |
|-----------|-----------|
| Leeres Tile (keine gültigen Frames) | `tiles_failed++`, Tile übersprungen |
| Zu wenig gültige Overlap-Samples | Nachbarpaar liefert keine Boundary-Beobachtung |
| weight_sum = 0 an Pixel | Pixel wird als `NaN` markiert |
| Auffällige Nachbargrenzen | erscheinen nur in der Boundary-Diagnostik, nicht als Tile-Eingriff |

## Nächste Phase

→ **Phase 10: STATE_CLUSTERING — Zustandsbasierte Clusterung**
