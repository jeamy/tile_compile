# TILE_RECONSTRUCTION — Parallele gewichtete Tile-Rekonstruktion

> **C++ Implementierung:** `runner_pipeline.cpp`
> **Phase-Enum:** `Phase::TILE_RECONSTRUCTION`

## Übersicht

Phase 9 ist das **Herzstück der Pipeline**. Jedes Tile wird separat rekonstruiert als gewichtetes Mittel über alle Frames, wobei das effektive Gewicht `W_f,t = G_f × L_f,t` die Frame-Qualität (global) und die lokale Tile-Qualität kombiniert. Danach folgt eine **tile-übergreifende Seam-Harmonisierung auf Basis realer Tile-Overlaps**, bevor die Tiles mittels **Hanning-Overlap-Add** zu einem Gesamtbild zusammengefügt werden.

```
┌──────────────────────────────────────────────────────┐
│  Für jedes Tile t (parallel mit N Threads):          │
│                                                      │
│  1. Tile aus jedem pre-warped Frame extrahieren      │
│  2. Effektives Gewicht W_f,t = G_f × L_f,t           │
│  3. Gewichtetes Mittel:                              │
│     tile_rec = Σ W_f,t · tile_f / Σ W_f,t            │
│  4. Post-Metriken (Contrast, BG, SNR)                │
│  5. Overlap-basierte Seam-Beobachtungen              │
│  6. Globaler Offset-/Scale-Solver über Nachbarn      │
│  7. Korrigiertes Tile → Hanning-Overlap-Add          │
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

## 5. Overlap-basierte Seam-Harmonisierung

Die eigentliche Kachelunterdrückung passiert jetzt **nicht mehr nur tile-intern**, sondern über reale Überlappungen benachbarter Tiles.

### 5.1 Hintergrundmaske pro Tile

```cpp
mask = compute_tile_background_mask(tile_rec, seam_cfg, finite_count, sample_count);
```

- Selektiert nur **dunkle, glatte** Pixel
- Schwellen:
  - `sample_quantile`
  - `gradient_quantile`
  - `min_sample_fraction`
  - `min_samples`
- Ziel: Nur robuste Hintergrundpixel für Seam-Schätzung verwenden

### 5.2 Beobachtungen aus Nachbar-Overlaps

```cpp
obs = estimate_tile_overlap_observation(
    lhs_tile, rhs_tile,
    lhs_image, rhs_image,
    lhs_mask, rhs_mask,
    pair_min_samples);
```

Für jedes Nachbarpaar werden in der gemeinsamen Overlap-Zone geschätzt:

- `background_delta = bg_rhs - bg_lhs`
- `log_scale_delta = log(scale_rhs) - log(scale_lhs)`
- `weight = sample_count`

Dabei wird nur auf Pixeln gearbeitet, die in **beiden** Tiles durch die Hintergrundmaske akzeptiert wurden.

### 5.3 Globaler Solver

```cpp
field = solve_tile_seam_field(tile_count, observations, solve_scale);
```

- Löst ein globales Gleichungssystem über alle Tile-Nachbarschaften
- Ergebnis:
  - relatives Background-Feld pro Tile
  - relatives Log-Scale-Feld pro Tile
- Die Felder werden anschließend auf Median 0 zentriert

Damit werden nicht nur einzelne Tiles „auf ein Niveau gezogen“, sondern die **gesamte Tile-Nachbarschaft konsistent ausgeglichen**.

### 5.4 Anwendung der Korrektur

Für MONO:

```cpp
tile = (tile - base_bg) * scale_factor + base_bg + bg_corr;
```

Für OSC:

- Background-Korrektur pro Kanal `R/G/B`
- gemeinsamer Scale-Faktor aus der Luma-Schätzung

Die Stärke wird über `stacking.tile_seam_harmonization.strength` geblendet. Der Scale-Faktor wird zusätzlich durch

- `scale_floor_factor`
- `scale_ceil_factor`

begrenzt.

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
| `tile_norm_bg_*[ti]` | float | angewendete Seam-Background-Korrektur |
| `tile_norm_scale[ti]` | float | angewendeter gemeinsamer Tile-Scale-Faktor |

## Artifact: `tile_reconstruction.json`

```json
{
  "num_frames": 100,
  "num_tiles": 1200,
  "tile_seam_mode": "overlap_global",
  "tile_seam_pair_count": 2200,
  "tile_seam_bg_observations": 2174,
  "tile_seam_scale_observations": 2158,
  "tile_valid_counts": [100, 100, 99, ...],
  "tile_mean_correlations": [1.0, 1.0, 1.0, ...],
  "tile_post_contrast": [0.0012, 0.0015, ...],
  "tile_post_background": [1.001, 1.002, ...],
  "tile_post_snr_proxy": [45.2, 38.7, ...],
  "tile_seam_bg_correction": [-0.03, -0.01, 0.00, ...],
  "tile_seam_scale_factor": [0.98, 1.01, 1.00, ...]
}
```

## Fehlerbehandlung

| Situation | Verhalten |
|-----------|-----------|
| Leeres Tile (keine gültigen Frames) | `tiles_failed++`, Tile übersprungen |
| Zu wenig gültige Hintergrundsamples im Tile | Tile bleibt ohne Seam-Beobachtung |
| Keine belastbaren Overlap-Beobachtungen | Seam-Solver liefert neutrale Korrektur |
| weight_sum = 0 an Pixel | Fallback: Referenz-Frame-Pixel |
| Problematische Scale-Schätzung | Korrektur durch `scale_floor_factor` / `scale_ceil_factor` geklemmt |

## Nächste Phase

→ **Phase 10: STATE_CLUSTERING — Zustandsbasierte Clusterung**
