# Phase 3: Globale Frame‑Metriken (C++)

## Ziel

Für jedes Frame wird ein globales Qualitätsgewicht `G_f` bestimmt. Dieses Gewicht fließt in alle späteren Tile‑Rekonstruktionen ein.

## C++‑Implementierung

**Referenzen:**
- `tile_compile_cpp/src/metrics/metrics.cpp`
- `tile_compile_cpp/include/tile_compile/metrics/metrics.hpp`

### Schritte

1. **Downsample (optional)**
   - Bei sehr großen Frames wird auf max 1024px Kantenlänge skaliert, um Speicher zu sparen.

2. **Background‑Mask (Sigma‑Clipping)**
   - Maske über Sigma‑Clip im Hintergrundbereich.

3. **Metriken**
   - `background`: Median der Hintergrund‑Pixel
   - `noise`: MAD‑basiertes Sigma der Hintergrund‑Pixel
   - `gradient_energy`: Median von Sobel‑Gradienten (quadratischer Betrag)

4. **Robuste Normalisierung**
   - Median + MAD → robust z‑Score pro Metrik.

5. **Score & Gewicht**
   - `Q = w_bg·(-bg_n) + w_noise·(-noise_n) + w_grad·(grad_n)`
   - Clamping auf `global_metrics.clamp`
   - `G_f = exp(Q)`
   - **Normierung**: Σ G_f = 1

## C++‑Skizze

```cpp
FrameMetrics m = calculate_frame_metrics(frame);
VectorXf G = calculate_global_weights(metrics, w_bg, w_noise, w_grad, clamp_lo, clamp_hi);
```

## Output

- `frame_metrics`: background/noise/gradient
- `global_weights`: Normalisierte Gewichte `G_f`

## Wichtige Parameter

- `global_metrics.weights.{background,noise,gradient}` (Summe = 1.0)
- `global_metrics.clamp` (z. B. `[-3, 3]`)

