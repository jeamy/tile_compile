# TILE_RECONSTRUCTION — Parallele gewichtete Tile-Rekonstruktion

> **C++ Implementierung:** `runner_pipeline.cpp`
> **Phase-Enum:** `Phase::TILE_RECONSTRUCTION`

## Übersicht

Phase 9 ist das **Herzstück der Pipeline**. Jedes Tile wird separat rekonstruiert als gewichtetes Mittel über alle Frames, wobei das effektive Gewicht `W_f,t = G_f × L_f,t` die Frame-Qualität (global) und die lokale Tile-Qualität kombiniert. Die rekonstruierten Tiles werden mittels **Hanning-Overlap-Add** zu einem nahtlosen Gesamtbild zusammengefügt.

```
┌──────────────────────────────────────────────────────┐
│  Für jedes Tile t (parallel mit N Threads):          │
│                                                      │
│  1. Tile aus jedem pre-warped Frame extrahieren      │
│  2. Effektives Gewicht W_f,t = G_f × L_f,t           │
│  3. Gewichtetes Mittel:                              │
│     tile_rec = Σ W_f,t · tile_f / Σ W_f,t            │
│  4. Per-Tile Background-Normalisierung               │
│  5. Post-Metriken (Contrast, BG, SNR)                │
│  6. Hanning-Window × tile_rec → Overlap-Add          │
│                                                      │
│  Danach:                                             │
│  7. recon(y,x) /= weight_sum(y,x)                    │
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

## 4. Per-Tile Background-Normalisierung

```cpp
// Methodik v3 §A.6
std::vector<float> rec_px(tile_rec.size());
std::memcpy(rec_px.data(), tile_rec.data(), sizeof(float) * rec_px.size());
float tile_bg = core::median_of(rec_px);

Matrix2Df ref_tile = extract_tile(prewarped_frames[global_ref_idx], t);
float ref_bg = core::median_of(ref_px);

float offset = ref_bg - tile_bg;
if (std::isfinite(offset) && std::fabs(offset) < 0.5f) {
    for (Eigen::Index k = 0; k < tile_rec.size(); ++k)
        tile_rec.data()[k] += offset;
}
```

- **Zweck**: Verhindert sichtbare Tile-Grenzen durch unterschiedliche Background-Levels
- **Referenz**: Background des Referenz-Frames im gleichen Tile-Bereich
- **Offset**: Differenz Referenz-BG − Rekonstruktions-BG → addiert
- **Safety**: Nur wenn Offset endlich und < 0.5 (verhindert Artefakte bei problematischen Tiles)

## 5. Post-Warp-Metriken

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

## 6. Hanning-Overlap-Add

```cpp
std::vector<float> hann_x = make_hann_1d(t.width);
std::vector<float> hann_y = make_hann_1d(t.height);

{
    std::lock_guard<std::mutex> lock(recon_mutex);
    for (int yy = 0; yy < tile_rec.rows(); ++yy) {
        for (int xx = 0; xx < tile_rec.cols(); ++xx) {
            int iy = y0 + yy;
            int ix = x0 + xx;
            float win = hann_y[yy] * hann_x[xx];
            recon(iy, ix) += tile_rec(yy, xx) * win;
            weight_sum(iy, ix) += win;
        }
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
| `tile_valid_counts[ti]` | int | Anzahl gültiger Frames pro Tile |
| `tile_mean_correlations[ti]` | float | Mittlere Korrelation (=1.0, global warp) |
| `tile_post_contrast[ti]` | float | Post-Contrast pro Tile |
| `tile_post_background[ti]` | float | Post-Background pro Tile |
| `tile_post_snr[ti]` | float | Post-SNR pro Tile |

## Artifact: `tile_reconstruction.json`

```json
{
  "num_frames": 100,
  "num_tiles": 1200,
  "tile_valid_counts": [100, 100, 99, ...],
  "tile_mean_correlations": [1.0, 1.0, 1.0, ...],
  "tile_post_contrast": [0.0012, 0.0015, ...],
  "tile_post_background": [1.001, 1.002, ...],
  "tile_post_snr_proxy": [45.2, 38.7, ...]
}
```

## Fehlerbehandlung

| Situation | Verhalten |
|-----------|-----------|
| Leeres Tile (keine gültigen Frames) | `tiles_failed++`, Tile übersprungen |
| weight_sum = 0 an Pixel | Fallback: Referenz-Frame-Pixel |
| offset nicht finite oder > 0.5 | BG-Normalisierung übersprungen |

## Nächste Phase

→ **Phase 10: STATE_CLUSTERING — Zustandsbasierte Clusterung**