# Phase 5: Lokale Tile‑Metriken (C++)

## Ziel

Ermittle pro Tile und pro Frame lokale Qualitätsmetriken. Diese Werte bestimmen die **lokalen Gewichte** `L_f,t`.

## C++‑Implementierung

**Referenzen:**
- `tile_compile_cpp/apps/runner_main.cpp` (LOCAL_METRICS Abschnitt)

### Berechnete Metriken

- FWHM‑Schätzung
- Roundness
- Kontrast / Gradient‑Energie
- Background / Noise
- Star‑Count (für STAR/STRUCTURE‑Klassifikation)

### Klassifikation

- **STAR‑Tile**: genügend Sterne → FWHM/Roundness/Contrast dominieren
- **STRUCTURE‑Tile**: keine/kaum Sterne → ENR und Background dominieren

### Gewichtung

- **STAR:** `Q = w_fwhm·(-FWHM̃) + w_round·R̃ + w_con·C̃`
- **STRUCTURE:** `Q = w_metric·ENR̃ − w_bg·B̃`
- Clamping auf `local_metrics.clamp`
- `L_f,t = exp(Q)`

## C++‑Skizze

```cpp
TileMetrics tm = compute_tile_metrics(tile);
float q = (star_tile) ? q_star(tm) : q_struct(tm);
local_weights[f][t] = std::exp(clamp(q));
```

## Output

- `local_metrics[frame][tile]`
- `local_weights[frame][tile]`

## Parameter (Auszug)

- `local_metrics.star_mode.weights.{fwhm,roundness,contrast}`
- `local_metrics.structure_mode.{metric_weight,background_weight}`
- `tile.star_min_count`

