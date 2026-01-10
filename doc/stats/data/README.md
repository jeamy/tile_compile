# Run 20260109_223321_68ebfdd3 — Stats/Artifacts (phasensortiert)

Quelle der Artefakte in diesem Run:

- primär erzeugt unter: `/media/tc_ssd/20260109_223321_68ebfdd3/artifacts/`
- im Repo-Verzeichnis ist `doc/stats/data/` die kopierte Ausgabe (Run wurde offenbar auf SSD ausgelagert)

Diese Seite erklärt **welche Daten** in den PNGs stecken und **wie du sie interpretierst**. Die Reihenfolge ist nach Pipeline-Phasen (siehe `runner/phases_impl.py`) sortiert.

## Allgemeine Konventionen

- x-Achse `frame_index`: Reihenfolge der Frames innerhalb des Runs (entspricht typischerweise Sortierung der Eingabeframes nach Dateiname/Index; 0-basiert im Plot).
- Kanal-Suffix `_R/_G/_B`: jeweiliger Farbkanal nach Channel-Split.
- Heatmaps:
  - x-Achse `tile_x`: Tile-Index in x-Richtung
  - y-Achse `tile_y`: Tile-Index in y-Richtung
  - Farbskala: Wert der jeweiligen Metrik pro Tile.

## Phase NORMALIZATION

### `normalization_background_timeseries.png`

- **Was wird geplottet**
  - Pro Kanal `R/G/B`: Zeitreihe der **Pre-normalization Backgrounds** `B_f`.
- **Datenquelle (Code)**
  - `B_f` wird als **Median** des jeweiligen Frames berechnet, *bevor* die Normalisierung die Frames verändert.
- **Achsen**
  - x: `frame_index`
  - y: `B_f` (Pixel-Intensity, in den FITS-Datenwerten; keine physikalischen Units)
- **Interpretation**
  - Eine ruhige, langsame Drift ist z.B. Mond-/Dunst-/Gradientenänderung.
  - Große Sprünge deuten auf wechselnde Bedingungen oder fehlerhafte Frames.

![](./normalization_background_timeseries.png)

## Phase GLOBAL_METRICS

### `global_weight_timeseries.png`

- **Was wird geplottet**
  - Pro Kanal `R/G/B`: Zeitreihe der globalen Gewichte `G_f,c`.
- **Datenquelle (Code/Definition)**
  - Hintergrund `B_f`: Median pro Frame (aus Phase NORMALIZATION, vor der Normalisierung gemessen)
  - Rauschen `σ_f`: Standardabweichung pro Frame (auf normalisierten Frames)
  - Gradient `E_f`: Mittelwert der Gradienten-Magnitude pro Frame (auf normalisierten Frames)
  - Robuste Normierung per MAD:
    - `x̃ = (x - median(x)) / (1.4826 * MAD(x))`
  - Globaler Quality:
    - `Q_f,c = α*(-B̃) + β*(-σ̃) + γ*(Ẽ)`
  - Clamp + Gewicht:
    - `Q_f,c := clamp(Q_f,c, -3, +3)`
    - `G_f,c = exp(Q_f,c)`
- **Achsen**
  - x: `frame_index`
  - y: `G_f,c` (positiver, unitless Weight)
- **Interpretation**
  - Höher = Frame hat in Summe bessere globale Qualität (weniger Background/Noise, „besserer“ Gradient).
  - Sehr niedrige Werte: Frames, die in späteren Schritten weniger beitragen sollten.

![](./global_weight_timeseries.png)

### `global_weight_hist.png`

- **Was wird geplottet**
  - Histogramm der `G_f,c` pro Kanal.
- **Achsen**
  - x: `G_f,c`
  - y: `count`
- **Interpretation**
  - Breite Verteilung = starke Qualitätsschwankungen.
  - Mehrgipflig kann auf Zustandswechsel (Seeing/Transparenz) hindeuten.

![](./global_weight_hist.png)

## Phase TILE_GRID

### `tile_grid_overlay_R.png`, `tile_grid_overlay_G.png`, `tile_grid_overlay_B.png`

- **Was wird geplottet**
  - Repräsentatives Frame pro Kanal als Graubild, mit darübergelegtem Tile-Grid.
- **Achsen**
  - Keine numerischen Achsen (Bildkoordinaten); Grid-Linien zeigen die Tile-Kachelung.
- **Interpretation**
  - Prüfen, ob Tile-Größe/Overlap sinnvoll ist:
    - Tiles sollten groß genug sein, um Struktur/Stars zu enthalten.
    - Zu kleine Tiles → noisy Metriken; zu große Tiles → schlechte Lokalität.

![](./tile_grid_overlay_R.png)
![](./tile_grid_overlay_G.png)
![](./tile_grid_overlay_B.png)

### `tile_grid.json`

- **Was ist das**
  - JSON-Metadaten zur Grid-Konfiguration (tile_size, step_size, etc.) und abgeleiteten Grid-Metadaten.

`tile_grid.json` wird zusammen mit den Overlays geschrieben.

## Phase LOCAL_METRICS (Tile-Metriken)

Die LOCAL_METRICS Phase berechnet pro Frame und Tile `Q_local` und daraus `L_f,t,c = exp(Q_local)`.

- `Q_local = w_fwhm*(-FWHM̃) + w_round*(R̃) + w_con*(C̃)`
- Clamp: `Q_local := clamp(Q_local, -3, +3)`
- Tile-Weight: `L_f,t,c = exp(Q_local)`

### `tile_quality_heatmap_*.png` (mean Q_local)

- **Was wird geplottet**
  - Heatmap der Tile-Qualität: `mean(Q_local)` über alle Frames.
- **Interpretation**
  - Hotspots können Bereiche mit systematisch „besserer“ Stern-/Strukturqualität sein.

![](./tile_quality_heatmap_R.png)
![](./tile_quality_heatmap_G.png)
![](./tile_quality_heatmap_B.png)

### `tile_quality_var_heatmap_*.png` (var Q_local)

- **Was wird geplottet**
  - Heatmap der Varianz: `var(Q_local)` über Frames.
- **Interpretation**
  - Hohe Varianz = Tile reagiert stark auf Seeing/Tracking/Clouds → Rekonstruktion wird dort adaptiver.

![](./tile_quality_var_heatmap_R.png)
![](./tile_quality_var_heatmap_G.png)
![](./tile_quality_var_heatmap_B.png)

### `tile_weight_heatmap_*.png` (mean L_local)

- **Was wird geplottet**
  - Heatmap der mittleren Tile-Gewichte: `mean(L_f,t,c)`.
- **Interpretation**
  - Höhere Werte = dieser Bildbereich wird im Mittel stärker gewichtet.

![](./tile_weight_heatmap_R.png)
![](./tile_weight_heatmap_G.png)
![](./tile_weight_heatmap_B.png)

### `tile_weight_var_heatmap_*.png` (var L_local)

- **Was wird geplottet**
  - Heatmap der Varianz: `var(L_f,t,c)`.
- **Interpretation**
  - Zeigt, wie stark die Gewichtung je Tile über die Zeit schwankt.

![](./tile_weight_var_heatmap_R.png)
![](./tile_weight_var_heatmap_G.png)
![](./tile_weight_var_heatmap_B.png)

## Phase TILE_RECONSTRUCTION

### `reconstruction_weight_sum_*.png`

- **Was wird geplottet**
  - `log(1 + weight_sum)` als Bild.
  - `weight_sum` ist die aufsummierte (Windowed) Gewichtung pro Pixel über alle Frames/Tiles.
- **Interpretation**
  - Dunkle Bereiche: wenig effektive Abdeckung/Gewicht → höhere Unsicherheit.
  - Helle Bereiche: viel Gewicht/Überdeckung → stabilere Rekonstruktion.

![](./reconstruction_weight_sum_R.png)
![](./reconstruction_weight_sum_G.png)
![](./reconstruction_weight_sum_B.png)

### `reconstruction_preview_*.png`

- **Was wird geplottet**
  - Vorschau der rekonstruierten Kanäle (uint8-visualisiert).
- **Interpretation**
  - Schneller sanity check: Fokus, Strukturen, Artefakte, Seam-Lines.

![](./reconstruction_preview_R.png)
![](./reconstruction_preview_G.png)
![](./reconstruction_preview_B.png)

### `reconstruction_absdiff_vs_mean_*.png`

- **Was wird geplottet**
  - `abs(reconstruction - mean(first N frames))` für `N = min(25, num_frames)`.
- **Interpretation**
  - Große Differenzen zeigen, wo die Rekonstruktion stark vom einfachen Mittel abweicht (meist durch Gewichtung / Tile-Normalisierung / lokale Selektion).

![](./reconstruction_absdiff_vs_mean_R.png)
![](./reconstruction_absdiff_vs_mean_G.png)
![](./reconstruction_absdiff_vs_mean_B.png)

## Phase CLUSTERING

### `clustering_summary_*.png`

- **Was wird geplottet**
  - Links: Cluster-Größen (Anzahl Frames pro Cluster)
  - Rechts: Boxplot von `G_f,c` pro Cluster
- **Interpretation**
  - Cluster sollen „ähnliche Zustände“ gruppieren. Große Unterschiede im Boxplot können auf starke Qualitätswechsel hinweisen.

![](./clustering_summary_R.png)
![](./clustering_summary_G.png)
![](./clustering_summary_B.png)

## Phase STACKING

### `quality_analysis_combined.png`

- **Was wird geplottet**
  - Pro vorhandenem Kanal eine Zeile mit bis zu 6 Panels:
    1. Histogramm `G_f,c` (mit Median-Linie)
    2. Histogramm `σ_f` (Noise)
    3. Histogramm `B_f` (Background)
    4. Histogramm `E_f` (Gradient)
    5. Histogramm `mean(Q_local)`
    6. Scatter: bevorzugt `σ` vs `G_f,c` (oder fallback `mean(Q_local)` vs `var(Q_local)`)
- **Interpretation**
  - Kompakte Übersicht, um Verteilungen/Outlier zu sehen.

![](./quality_analysis_combined.png)

## Hinweise zum Kopieren der Artefakte ins Repo

Damit die Bilder in dieser Markdown-Seite angezeigt werden, müssen die Dateien aus
`/media/tc_ssd/20260109_223321_68ebfdd3/artifacts/` nach `doc/stats/data/` kopiert werden.
