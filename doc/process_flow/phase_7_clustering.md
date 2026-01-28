# Phase 7: State‑Clustering (C++)

## Ziel

Frames werden anhand eines Zustandsvektors in Cluster eingeteilt. Jeder Cluster repräsentiert eine ähnliche Qualität/Bewegung, um daraus **synthetische Frames** zu erzeugen.

## C++‑Implementierung

**Referenz:** `tile_compile_cpp/apps/runner_main.cpp` (Phase 7)

### Zustandsvektor (C++)

```
[G_f,
 mean_local_quality,
 var_local_quality,
 mean_cc_tiles,
 mean_warp_var_tiles,
 invalid_tile_fraction]
```

- `G_f`: globales Gewicht
- `mean_local_quality`, `var_local_quality`: aus lokalen Tile‑Metriken
- `mean_cc_tiles`: Mittelwert der Tile‑ECC‑Korrelationen
- `mean_warp_var_tiles`: Mittelwert der Tile‑Warp‑Varianz
- `invalid_tile_fraction`: Anteil ungültiger Tiles pro Frame

### Clustering

- K‑Means
- `K = clip(floor(N/10), K_min, K_max)`
- Parameter: `synthetic.clustering.cluster_count_range`

### Reduced Mode

- Wenn `frames < frames_reduced_threshold` und `reduced_mode_skip_clustering=true`, wird Phase 7 übersprungen und Phase 8 deaktiviert.

## C++‑Skizze

```cpp
X = zscore(state_vectors);
labels = kmeans(X, K);
```

## Output

- `artifacts/state_clustering.json`
- `cluster_labels[frame]`

