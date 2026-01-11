# Stats / PNG artifacts

Dieses Verzeichnis dokumentiert die im Runner generierten **PNG-Artefakte** (Plots/Heatmaps/Previews) und erklärt **welche Daten** dargestellt werden und **wie sie zu interpretieren sind**.

## Aufbau

- `data/`:
  - kopierte PNGs aus einem run (bzw. aus dem tatsächlichen Arbeits-/SSD-Pfad, falls ausgelagert)
  - `README.md`: run-spezifische, phasen-sortierte Erklärung + eingebettete Bilder

## Namenskonventionen

- Suffix `_R/_G/_B`: Kanal-spezifisch (OSC → demosaiced channels).
- `*_timeseries.png`: x-Achse ist typischerweise `frame_index` (0-basiert im Plot, nicht unbedingt Dateiname).
- `*_heatmap_*.png`: 2D-Grid, Achsen sind `tile_x` und `tile_y` (Index im Tile-Grid).

## Begriffe / Größen

- `B_f`: Background-Level pro Frame (hier: **Median** des Frames), vor Normalisierung gemessen.
- `σ_f`: Noise-Level pro Frame (hier: **Standardabweichung** der Pixel nach Normalisierung).
- `E_f`: Gradient-Energy pro Frame (hier: Mittelwert der Gradientenmagnituden nach Normalisierung).
- MAD-Normalisierung (robust):
  - `x̃ = (x - median(x)) / (1.4826 * MAD(x))`
  - `MAD(x) = median(|x - median(x)|)`
- Globaler Quality/Weight:
  - `Q_f,c = α*(-B̃) + β*(-σ̃) + γ*(Ẽ)`
  - `G_f,c = exp(clamp(Q_f,c, -3, +3))`
- Lokaler (Tile) Quality/Weight:
  - `Q_local = w_fwhm*(-FWHM̃) + w_round*(R̃) + w_con*(C̃)`
  - `L_f,t,c = exp(clamp(Q_local, -3, +3))`

## Runs

- [`data`](data/README.md)
