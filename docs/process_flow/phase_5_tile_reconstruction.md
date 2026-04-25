# TILE_RECONSTRUCTION — Parallele gewichtete Tile-Rekonstruktion

> **C++ Implementierung:** `runner_pipeline.cpp`
> **Phase-Enum:** `Phase::TILE_RECONSTRUCTION`

## Übersicht

Phase 9 ist das **Herzstück der Pipeline**. Jedes Tile wird separat rekonstruiert als gewichtetes Mittel über alle Frames, wobei das effektive Gewicht `W_{f,t} = G_f × L_{f,t}` die Frame-Qualität (global) und die lokale Tile-Qualität kombiniert. In `v3.3.9` werden die Tiles anschließend **ohne tileweise nichtlineare Vor-OLA-Normalisierung** über eine support-aware Overlap-Add-Semantik zusammengefügt.

```
┌──────────────────────────────────────────────────────┐
│  Für jedes Tile t (parallel mit N Threads):          │
│                                                      │
│  1. Tile aus jedem pre-warped Frame extrahieren      │
│  2. Effektives Gewicht W_f,t = G_f × L_f,t           │
│  3. Support-aware Sigma-Clip / Mittel                │
│     tile_rec = Σ W_f,t · tile_f / Σ W_f,t            │
│  4. Post-Metriken (Contrast, BG, SNR)                │
│  5. Tile → support-aware OLA                         │
│                                                      │
│  Danach:                                             │
│  6. recon(y,x) /= weight_sum(y,x)                    │
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

## 3. Gewichtetes Mittel und support-aware Sigma-Clipping

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
- Bei `wsum = 0`: deterministischer Fallback auf konservatives Mittel
- Alle Frames werden verwendet; Ausreißerbehandlung ist rein pixelweise

In `v3.3.9` ist zusätzlich verbindlich:

- gültige Samples werden über **Finitheit + validen Canvas-Support** definiert,
- negative oder nullige lineare Pixelwerte bleiben zulässig,
- Sigma-Clipping verwendet `N_eff`-/`D_eff`-Guards und `min_fraction`-Keep-Floor.

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

## 5. Support-Aware Overlap-Add

```cpp
for (int yy = 0; yy < tile.rows(); ++yy) {
    for (int xx = 0; xx < tile.cols(); ++xx) {
        float win = window_y[yy] * window_x[xx];
        if (!support_mask(yy, xx)) continue;
        recon(iy, ix) += tile(yy, xx) * win;
        weight_sum(iy, ix) += win;
    }
}
```

### Fenster- und Support-Semantik

```cpp
w[i] = deterministic_partition_window(i, n)
```

- **2D separabel**: zulässig sind z.B. Cosine-/Tukey-Partition-Fenster
- im verpflichtenden Kern zählt nicht "Hanning" als spezifische Wahl, sondern:
  - Nichtnegativität
  - support-aware Partition of Unity
  - korrekte Randbehandlung
- Pixel mit `|V| = 0`, canvas-ungültige Pixel oder nicht-finite Tile-Werte dürfen **nicht** in den OLA-Nenner eingehen

### Normalisierung

```cpp
for (int i = 0; i < recon.size(); ++i) {
    float ws = weight_sum.data()[i];
    if (ws > 1.0e-12f) {
        recon.data()[i] /= ws;
    } else {
        recon.data()[i] = std::numeric_limits<float>::quiet_NaN();
    }
}
```

- Division durch akkumulierte Fenstergewichte
- Pixel ohne gültige Tile-Abdeckung bleiben außerhalb des Supports und werden nicht künstlich mit Referenzbildwerten aufgefüllt

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
| `tile_support_masks[ti]` | Matrix/diagnostics | Support-Information für support-aware OLA |
| `tile_boundary_*` | diagnostics | Boundary- und Seam-Diagnostik ohne Feedback in den Kern |

## Artifact: `tile_reconstruction.json`

```json
{
  "num_frames": 100,
  "num_tiles": 1200,
  "tile_boundary_analysis_uses_common_canvas_mask": true,
  "tile_boundary_raw_analysis_input": "pre_ola_raw",
  "tile_boundary_analysis_input": "pre_ola_raw",
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
| weight_sum = 0 an Pixel | Pixel bleibt außerhalb des gültigen Supports und wird als `NaN` markiert |
| Auffällige Nachbargrenzen | erscheinen nur in der Boundary-Diagnostik, nicht als Tile-Eingriff |

## Tile-Ausschluss und Canvas-Bounds

### Dead-Tile-Detection

Vor der parallelen Verarbeitung klassifiziert `detect_dead_tiles()` jedes Tile basierend auf der Canvas-Überdeckung:

| Kategorie | Bedingung | Ergebnis |
|-----------|-----------|----------|
| **Vollständig außerhalb** | `x1 <= x0 \|\| y1 <= y0` | `dead_mask[ti] = true` |
| **Unter Mindest-Coverage** | `coverage < dead_tile_min_coverage_fraction` (Standard: 1%) | Tile wird übersprungen |
| **Teilweise außerhalb** | Tile überlappt Canvas nur teilweise | Nur sichtbarer Teil wird berechnet |

**Bounds-Klammerung:**
```cpp
const int x0 = std::max(0, t.x);
const int y0 = std::max(0, t.y);
const int x1 = std::min(canvas_width,  t.x + t.width);
const int y1 = std::min(canvas_height, t.y + t.height);
```

### Bounds-Checking in Code-Pfaden

| Code-Pfad | Implementierung | Beispiel |
|-----------|-----------------|----------|
| **CPU Reconstruction** | Schleifen-Bounds | `y < tile.y + tile.height && y < h` |
| **CUDA Acceleration** | Clip-Berechnung | `clip_w = std::min(tile.cols(), accum.cols() - x0)` |
| **OpenCL Acceleration** | Clip-Berechnung | Gleiches Pattern wie CUDA |
| **Runner Pipeline** | Null-Check | `if (iy < 0 \|\| iy >= canvas_height) continue` |

### Konfigurationsparameter

| Parameter | Standard | Beschreibung |
|-----------|----------|--------------|
| `dead_tile_min_coverage_fraction` | 0.01 (1%) | Mindest-Überdeckung für lebendige Tiles |
| `tile_boundary_diagnostics_enabled` | false | Diagnose zu Tile-Grenzen aktivieren |

Der Tile-Scheduler verarbeitet **nur** Tiles, die nicht in `dead_tile_mask` markiert sind. Tiles außerhalb des Canvas werden somit gar nicht berechnet.

## Nächste Phase

→ **Phase 10: STATE_CLUSTERING — Zustandsbasierte Clusterung**
