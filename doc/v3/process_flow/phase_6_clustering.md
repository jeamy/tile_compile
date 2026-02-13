# STATE_CLUSTERING — Zustandsbasierte Frame-Clusterung

> **C++ Implementierung:** `runner_main.cpp` Zeilen 1649–1892
> **Phase-Enum:** `Phase::STATE_CLUSTERING`

## Übersicht

Phase 8 gruppiert Frames nach ihrem **Qualitätszustand** mittels K-Means-Clusterung auf einem 6-dimensionalen Zustandsvektor. Frames mit ähnlicher Qualität werden einem gemeinsamen Cluster zugeordnet. In der nächsten Phase (SYNTHETIC_FRAMES) wird pro Cluster ein synthetischer Frame erzeugt — dies reduziert die Frame-Anzahl von N auf K bei gleichzeitiger Rauschreduktion.

**Reduced Mode:** Bei `N < frames_reduced_threshold` wird diese Phase **übersprungen**.

```
┌──────────────────────────────────────────────────────┐
│  1. Zustandsvektor pro Frame berechnen               │
│     v_f = [G_f, mean_Q, var_Q, CC̄, WarpVar̄, inv_f]  │
│                                                      │
│  2. z-Score Normalisierung (6 Dimensionen)           │
│                                                      │
│  3. K bestimmen: K = clip(N/10, k_min, k_max)       │
│                                                      │
│  4. K-Means Clusterung (20 Iterationen)              │
│     • Initialisierung: gleichmäßig verteilte Frames  │
│     • Assign-Labels → Update-Centers → repeat        │
│                                                      │
│  5. Degenerations-Check                              │
│     • Leere Cluster? → Quantile-Fallback             │
│                                                      │
│  Output: cluster_labels[N], cluster_sizes[K]         │
└──────────────────────────────────────────────────────┘
```

## Reduced Mode

```cpp
const bool reduced_mode = (frames.size() < cfg.assumptions.frames_reduced_threshold);
const bool skip_clustering = (reduced_mode && cfg.assumptions.reduced_mode_skip_clustering);

if (skip_clustering) {
    use_synthetic_frames = false;
    emitter.phase_end(run_id, Phase::STATE_CLUSTERING, "skipped",
                      {{"reason", "reduced_mode"}, ...});
}
```

Wenn `reduced_mode_skip_clustering = true` und N < Threshold:
- Phase wird als "skipped" markiert
- Alle Frames erhalten Label 0 (ein Cluster)
- Synthetische Frames werden **nicht** erzeugt
- TILE_RECONSTRUCTION-Ergebnis wird direkt als finales Bild verwendet

## 1. Zustandsvektor (6D)

Pro Frame wird ein 6-dimensionaler Zustandsvektor berechnet:

```cpp
state_vectors[fi] = {
    G_f,                    // Globales Gewicht (Phase 4)
    mean_local,             // Mittelwert lokaler Qualitäts-Scores
    var_local,              // Varianz lokaler Qualitäts-Scores
    mean_cc_tiles,          // Mittlere Tile-Korrelation (global)
    mean_warp_var_tiles,    // Mittlere Warp-Varianz (global)
    frame_invalid_fraction  // Anteil ungültiger Tiles
};
```

| Dimension | Symbol | Quelle | Beschreibung |
|-----------|--------|--------|-------------|
| 0 | G_f | Phase 4 | Globales Frame-Gewicht |
| 1 | ⟨Q_local⟩_f | Phase 6 | Mittelwert der lokalen Tile-Quality-Scores |
| 2 | Var(Q_local)_f | Phase 6 | Varianz der lokalen Tile-Quality-Scores |
| 3 | CC̄_tiles | Phase 7 | Mittlere Tile-Korrelation (über alle Tiles) |
| 4 | WarpVar̄ | Phase 7 | Mittlere Warp-Varianz (über alle Tiles) |
| 5 | inv_frac_f | Phase 7 | Anteil ungültiger Tiles am Gesamtgrid |

### Mean/Varianz lokaler Qualität

```cpp
float mean_local = 0.0f, var_local = 0.0f;
for (const auto &tm : local_metrics[fi])
    mean_local += tm.quality_score;
mean_local /= local_metrics[fi].size();
for (const auto &tm : local_metrics[fi]) {
    float diff = tm.quality_score - mean_local;
    var_local += diff * diff;
}
var_local /= local_metrics[fi].size();
```

## 2. z-Score Normalisierung

```cpp
const size_t D = 6;
std::vector<float> means(D, 0.0f);
std::vector<float> stds(D, 0.0f);

// Mean + Std pro Dimension
for (size_t d = 0; d < D; ++d) {
    means[d] = sum(X[*][d]) / N;
    stds[d] = sqrt(var(X[*][d]));
}

// Normalisierung
for (size_t i = 0; i < X.size(); ++i)
    for (size_t d = 0; d < D; ++d)
        X[i][d] = (stds[d] > eps) ? ((X[i][d] - means[d]) / stds[d]) : 0.0f;
```

- Alle 6 Dimensionen werden auf Mittelwert=0, Standardabweichung=1 normalisiert
- Verhindert, dass eine Dimension die Clusterung dominiert
- Bei std=0 (konstante Dimension): wird auf 0 gesetzt

## 3. Cluster-Anzahl K

```cpp
int k_min = cfg.synthetic.clustering.cluster_count_range[0];
int k_max = cfg.synthetic.clustering.cluster_count_range[1];
int k_default = std::max(k_min, std::min(k_max, n_frames / 10));
n_clusters = std::min(k_default, n_frames);
```

```
K = clip(floor(N / 10), k_min, k_max)
```

| N Frames | k_min=3, k_max=30 | K |
|----------|-------------------|---|
| 50 | clip(5, 3, 30) | 5 |
| 100 | clip(10, 3, 30) | 10 |
| 200 | clip(20, 3, 30) | 20 |
| 500 | clip(50, 3, 30) | 30 |

## 4. K-Means Clusterung

```cpp
// Initialisierung: gleichmäßig verteilte Frames als Zentren
for (int c = 0; c < n_clusters; ++c) {
    int idx = (c * n_frames) / n_clusters;
    centers[c] = X[idx];
}

// 20 Iterationen
for (int iter = 0; iter < 20; ++iter) {
    // Assign: jeder Frame zum nächsten Zentrum
    for (size_t fi = 0; fi < X.size(); ++fi) {
        float best_dist = MAX;
        for (int c = 0; c < n_clusters; ++c) {
            float dist = euclidean_distance_sq(X[fi], centers[c]);
            if (dist < best_dist) { best_dist = dist; cluster_labels[fi] = c; }
        }
    }
    // Update: neue Zentren als Mittelwert der Cluster-Mitglieder
    for (int c = 0; c < n_clusters; ++c) {
        centers[c] = mean(X[fi] where cluster_labels[fi] == c);
    }
}
```

- **Initialisierung**: Gleichmäßig verteilte Frames (nicht k-means++)
- **Distanzmetrik**: Euklidische Distanz im 6D-Raum (nach z-Normalisierung)
- **20 Iterationen** (fest, kein Konvergenzcheck)
- **Methode**: `"kmeans"`

## 5. Degenerations-Fallback

```cpp
bool degenerate = false;
for (int c = 0; c < n_clusters; ++c) {
    if (counts[c] <= 0) { degenerate = true; break; }
}

if (degenerate && n_clusters > 1) {
    clustering_method = "quantile";
    // Sortiere Frames nach G_f, verteile gleichmäßig auf Cluster
    std::sort(order.begin(), order.end(), by_global_weight);
    for (size_t r = 0; r < order.size(); ++r) {
        int label = (r * n_clusters) / order.size();
        cluster_labels[order[r].second] = label;
    }
}
```

Wenn K-Means zu **leeren Clustern** führt (z.B. bei sehr homogenen Daten):
- **Quantile-Fallback**: Frames werden nach globalem Gewicht `G_f` sortiert und gleichmäßig auf K Cluster verteilt
- Jeder Cluster enthält dann N/K Frames
- Methode wird als `"quantile"` im Artifact vermerkt

## Konfigurationsparameter

| Parameter | Beschreibung | Default |
|-----------|-------------|---------|
| `synthetic.clustering.cluster_count_range` | [k_min, k_max] | [3, 30] |
| `assumptions.frames_reduced_threshold` | Threshold für Reduced Mode | 200 |
| `assumptions.reduced_mode_skip_clustering` | Clustering in Reduced Mode überspringen | `true` |

## Artifact: `state_clustering.json`

```json
{
  "n_clusters": 10,
  "k_min": 3,
  "k_max": 30,
  "method": "kmeans",
  "cluster_labels": [0, 0, 1, 2, 0, 3, ...],
  "cluster_sizes": [12, 8, 15, 10, ...]
}
```

## Nächste Phase

→ **Phase 9: SYNTHETIC_FRAMES — Synthetische Frame-Erzeugung**