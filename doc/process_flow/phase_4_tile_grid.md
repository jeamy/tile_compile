# Phase 4: Adaptive Tile‑Grid (C++)

## Ziel

Erzeuge ein Tile‑Grid, das lokale Bewegungen und Struktur‑Gradienten berücksichtigt und unnötige Tiles vermeidet.

## C++‑Implementierung

**Referenzen:**
- `tile_compile_cpp/src/pipeline/adaptive_tile_grid.cpp`
- `tile_compile_cpp/apps/runner_main.cpp`

### Schritte

1. **Initialisierung**
   - Basierend auf `tile.size_factor`, `tile.min_size`, Bildgröße.

2. **Warp‑Probe (optional)**
   - Grobe Warp‑Analyse mit wenigen Frames (`num_probe_frames`).
   - Erzeugt ein **Gradient‑Field** für adaptive Dichte.

3. **Hierarchische Tile‑Erzeugung**
   - Rekursive Unterteilung bis `hierarchical_max_depth`.
   - Splits, wenn Gradient über `split_gradient_threshold` liegt.

4. **Adaptive Verfeinerung (später in Phase 6)**
   - Tiles mit hoher Warp‑Varianz oder niedriger Korrelation werden gesplittet.

## C++‑Skizze

```cpp
TileGrid grid = build_adaptive_tile_grid(cfg, grad_field);
// später in Phase 6: split tiles by warp_variance/mean_cc
```

## Parameter (Auszug)

- `v4.adaptive_tiles.enabled`
- `v4.adaptive_tiles.use_warp_probe`
- `v4.adaptive_tiles.use_hierarchical`
- `v4.adaptive_tiles.initial_tile_size`
- `v4.adaptive_tiles.min_tile_size_px`
- `v4.adaptive_tiles.gradient_sensitivity`
- `v4.adaptive_tiles.split_gradient_threshold`

