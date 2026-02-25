# TILE_GRID — Seeing-adaptive Tile-Erzeugung

> **C++ Implementierung:** `runner_pipeline.cpp`
> **Phase-Enum:** `Phase::TILE_GRID`

## Übersicht

Phase 6 erzeugt ein **einheitliches reguläres Tile-Grid** über das gesamte Bild. Die Tile-Größe wird **adaptiv** an das gemessene Seeing (FWHM) angepasst. Größeres Seeing → größere Tiles, um genügend Sternpixel pro Tile zu haben.

```
┌──────────────────────────────────────────────────────┐
│  1. FWHM-Messung                                     │
│     • Zentrales 1024×1024 ROI                        │
│     • Bis zu 5 gleichmäßig verteilte Probeframes     │
│     • metrics::measure_fwhm_from_image()             │
│     • Erster gültiger FWHM-Wert wird übernommen      │
│                                                      │
│  2. Tile-Größe berechnen                             │
│     T = clip(size_factor × FWHM, T_min, T_max)       │
│     T_max = min(W, H) / max_divisor                  │
│                                                      │
│  3. Overlap berechnen                                │
│     overlap_px = floor(overlap_fraction × T)         │
│     stride = T − overlap_px                          │
│                                                      │
│  4. Grid erzeugen                                    │
│     build_initial_tile_grid(W, H, T, overlap_frac)   │
│     → Liste von Tile{x, y, width, height}            │
└──────────────────────────────────────────────────────┘
```

## 1. FWHM-Messung

```cpp
const size_t n_probe = std::min<size_t>(5, frames.size());
for (size_t pi = 0; pi < n_probe; ++pi) {
    size_t fi = (n_probe <= 1) ? 0 : (pi * (frames.size() - 1)) / (n_probe - 1);
    const int roi_w = std::min(width, 1024);
    const int roi_h = std::min(height, 1024);
    const int roi_x0 = std::max(0, (width - roi_w) / 2);
    const int roi_y0 = std::max(0, (height - roi_h) / 2);

    Matrix2Df img = io::read_fits_region_float(frames[fi], roi_x0, roi_y0, roi_w, roi_h);
    image::apply_normalization_inplace(img, norm_scales[fi], ...);
    float fwhm = metrics::measure_fwhm_from_image(img);
    if (fwhm > 0.0f) { seeing_fwhm_med = fwhm; break; }
}
```

- **Probeframes**: Bis zu 5 Frames, gleichmäßig über die Sequenz verteilt (Anfang, 1/4, Mitte, 3/4, Ende)
- **ROI**: Zentrales 1024×1024 Fenster (oder kleiner bei kleinen Bildern)
- **Normalisiert**: Frame wird vor FWHM-Messung normalisiert
- **Early Exit**: Erster gültiger FWHM-Wert wird übernommen
- **Fallback**: Wenn kein FWHM messbar → Default = 3.0 Pixel

`measure_fwhm_from_image()` findet Sterne, fittet 2D-Gauss-Profile und gibt den Median-FWHM zurück.

## 2. Tile-Größe berechnen

```cpp
float F = seeing_fwhm_med;
if (!(F > 0.0f) || !std::isfinite(F)) F = 3.0f;

const int tmin = std::max(16, cfg.tile.min_size);
const int D = std::max(1, cfg.tile.max_divisor);
int tmax = std::max(1, std::min(width, height) / D);
if (tmax < tmin) tmax = tmin;

const float t0 = static_cast<float>(cfg.tile.size_factor) * F;
const float tc = std::min(std::max(t0, (float)tmin), (float)tmax);
seeing_tile_size = static_cast<int>(std::floor(tc));
```

### Formel

```
T = floor(clip(size_factor × FWHM, T_min, T_max))

wobei:
  T_min = max(16, config.tile.min_size)
  T_max = min(W, H) / max_divisor
```

### Typische Werte

| Seeing (FWHM) | size_factor=20 | Tile-Größe |
|---------------|----------------|------------|
| 2.0 px | 20 × 2.0 = 40 | 40 px |
| 3.0 px | 20 × 3.0 = 60 | 60 px |
| 5.0 px | 20 × 5.0 = 100 | 100 px |
| 8.0 px | 20 × 8.0 = 160 | 160 px |

## 3. Overlap berechnen

```cpp
overlap_fraction = std::min(0.5f, std::max(0.0f, overlap_fraction));
overlap_px = static_cast<int>(std::floor(overlap_fraction * seeing_tile_size));
stride_px = seeing_tile_size - overlap_px;
if (stride_px <= 0) {
    overlap_fraction = 0.25f;
    overlap_px = static_cast<int>(std::floor(0.25f * seeing_tile_size));
    stride_px = seeing_tile_size - overlap_px;
}
```

- **overlap_fraction** wird auf [0.0, 0.5] geclampt
- **stride** = tile_size − overlap → Schrittweite in Pixeln
- **Safety**: Wenn stride ≤ 0, wird overlap auf 25% zurückgesetzt
- Overlap sorgt für **glatte Übergänge** bei der Hanning-Overlap-Add-Rekonstruktion

## 4. Grid erzeugen

```cpp
std::vector<Tile> tiles = pipeline::build_initial_tile_grid(
    width, height, uniform_tile_size, overlap_fraction);
```

Die Funktion erzeugt ein reguläres Grid:
- Tiles werden in Raster-Reihenfolge (links→rechts, oben→unten) erzeugt
- Jedes Tile hat Position `(x, y)` und Dimensionen `(width, height)`
- **Randtiles** können abweichende Dimensionen haben (Bild-Kante)
- Tiles am rechten/unteren Rand werden so angepasst, dass sie das Bild vollständig abdecken

### Tile-Limitierung

```cpp
std::vector<Tile> tiles_phase56 = tiles;
if (max_tiles > 0 && tiles_phase56.size() > max_tiles)
    tiles_phase56.resize(max_tiles);
```

Für Debug/Test: `--max-tiles` limitiert die Anzahl der Tiles im Tile-Pfad (ab Phase 6).

## Konfigurationsparameter

| Parameter | Beschreibung | Default |
|-----------|-------------|---------|
| `tile.size_factor` | Multiplikator für FWHM → Tile-Größe | 20 |
| `tile.min_size` | Minimale Tile-Größe (px) | 32 |
| `tile.max_divisor` | Maximale Tile-Größe = min(W,H) / Divisor | 4 |
| `tile.overlap_fraction` | Overlap-Anteil (0.0–0.5) | 0.25 |
| `tile.star_min_count` | Min. Sterne für STAR-Klassifikation | 3 |

## Artifact: `tile_grid.json`

```json
{
  "image_width": 4656,
  "image_height": 3520,
  "num_tiles": 1200,
  "overlap_fraction": 0.25,
  "seeing_fwhm_median": 3.5,
  "seeing_tile_size": 70,
  "seeing_overlap_px": 17,
  "stride_px": 53,
  "tile_config": {
    "size_factor": 20,
    "min_size": 32,
    "max_divisor": 4,
    "overlap_fraction": 0.25
  },
  "uniform_tile_size": 70,
  "tiles": [
    {"x": 0, "y": 0, "width": 70, "height": 70},
    {"x": 53, "y": 0, "width": 70, "height": 70},
    ...
  ]
}
```

## Nächste Phase

→ **Phase 7: COMMON_OVERLAP — Gemeinsamer Datenbereich**, danach **Phase 8: LOCAL_METRICS — Lokale Metriken**

(Hinweis: `REGISTRATION` läuft in v3.2 bereits als Phase 1 vor `CHANNEL_SPLIT`/`NORMALIZATION`/`GLOBAL_METRICS`.)