# Phase 3: Tile-Erzeugung (FWHM-basiert)

## Übersicht

Phase 3 erzeugt ein einheitliches Tile-Grid für alle Frames und Kanäle. Die Tile-Größe wird **adaptiv** basierend auf dem gemessenen Seeing (FWHM) berechnet.

## Ziele

1. FWHM (Full Width Half Maximum) aller Sterne messen
2. Robuste FWHM-Schätzung über alle Frames
3. Seeing-adaptive Tile-Größe berechnen
4. Einheitliches Tile-Grid erzeugen
5. Overlap-Strategie definieren

## Schritt 3.1: FWHM-Messung

### Was ist FWHM?

```
Stern-Profil (1D-Schnitt):

Intensität
    │
    │      ╱‾‾‾╲
    │     ╱     ╲
Max │────●───────●────  ← Maximum
    │   ╱         ╲
1/2 │──●───────────●──  ← Half Maximum
    │ ╱             ╲
    │╱               ╲___
    └──────────────────────► Position
       │←── FWHM ──→│
       
FWHM = Breite bei halber Maximalhöhe
     = Maß für Seeing-Qualität
```

**Physikalische Bedeutung:**
- FWHM ≈ 2.355 × σ (für Gaussian PSF)
- Typische Werte: 2-6 Pixel
- Niedriges FWHM = gutes Seeing (scharfe Sterne)
- Hohes FWHM = schlechtes Seeing (unscharfe Sterne)

### Messmethode

```
┌─────────────────────────────────────────┐
│  Normalized Frame I'_f,c[x,y]           │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Step 1: Sternfindung                   │
│                                         │
│  • Threshold: I' > μ + 5σ               │
│  • Lokale Maxima                        │
│  • Mindestabstand zwischen Sternen      │
│  • Keine saturierten Sterne             │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Step 2: PSF-Fitting pro Stern          │
│                                         │
│  Gaussian 2D:                           │
│  PSF(x,y) = A·exp(-((x-x₀)²+(y-y₀)²)    │
│                    /(2σ²))              │
│                                         │
│  Fit-Parameter: x₀, y₀, σ, A            │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Step 3: FWHM-Berechnung                │
│                                         │
│  FWHM = 2.355 × σ                       │
│                                         │
│  Qualitätsfilter:                       │
│  • 1.0 < FWHM < 15.0 Pixel              │
│  • Rundheit > 0.6                       │
│  • Fit-Residuum < 0.1                   │
└─────────────────────────────────────────┘
```

### Visualisierung: 2D Gaussian PSF

```
Stern-Cutout (15x15 Pixel):

    ░░░░░░░░░░░░

░░░
    ░░░░░░░░░░░░░░░
    ░░░░░▒▒▒▒▒░░░░░
    ░░░░▒▓▓▓▓▓▒░░░░
    ░░░▒▓████▓▒░░░░  ← Stern-Zentrum
    ░░░▒▓████▓▒░░░░
    ░░░░▒▓▓▓▓▓▒░░░░
    ░░░░░▒▒▒▒▒░░░░░
    ░░░░░░░░░░░░░░░
    
    │←─ FWHM ─→│
    
Kontur bei halber Höhe:
    
    ░░░░░░░░░░░░░░░
    ░░░░░●●●●░░░░░░
    ░░░░●░░░░●░░░░░
    ░░░●░░░░░░●░░░░
    ░░●░░░░░░░░●░░░  ← FWHM-Kontur
    ░░░●░░░░░░●░░░░
    ░░░░●░░░░●░░░░░
    ░░░░░●●●●░░░░░░
    ░░░░░░░░░░░░░░░
```

### Robuste FWHM-Schätzung

```
┌─────────────────────────────────────────┐
│  FWHM-Messungen:                        │
│  • Pro Frame: 50-500 Sterne             │
│  • Pro Stern: 1 FWHM-Wert               │
│  • Gesamt: N_frames × N_stars Werte     │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Robuste Statistik                      │
│                                         │
│  1. Sammle alle FWHM-Werte              │
│  2. Entferne Ausreißer (IQR-Methode)    │
│  3. Berechne Median                     │
│                                         │
│  F = median(FWHM_valid)                 │
└─────────────────────────────────────────┘
```

**IQR-Methode (Interquartile Range):**
```python
Q1 = percentile(FWHM, 25)
Q3 = percentile(FWHM, 75)
IQR = Q3 - Q1

# Ausreißer-Grenzen
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Filtern
FWHM_valid = FWHM[(FWHM >= lower) & (FWHM <= upper)]

# Robuste Schätzung
F = median(FWHM_valid)
```

### Beispiel-Verteilung

```
FWHM-Histogramm über alle Frames:

Count
  │
80│     ╱‾╲
  │    ╱   ╲
60│   ╱     ╲
  │  ╱       ╲___
40│ ╱            ╲___
  │╱                 ╲___
20│                      ╲___
  │                          ╲___
  └────────────────────────────────► FWHM [px]
  1   2   3   4   5   6   7   8   9
      │       │
      Q1      Q3
          │
        Median F ≈ 3.2 px
```

## Schritt 3.2: Tile-Größen-Berechnung

### Normative Formel

```
Gegeben:
  F      - robuste FWHM-Schätzung [Pixel]
  s      - tile.size_factor (dimensionslos)
  T_min  - tile.min_size [Pixel]
  W, H   - Bildbreite/-höhe [Pixel]
  D      - tile.max_divisor (dimensionslos)

Berechnung:
  T_0 = s × F                          (seeing-adaptiv)
  T_max = floor(min(W, H) / D)         (Lokalitäts-Constraint)
  T = floor(clip(T_0, T_min, T_max))   (finale Tile-Größe)
```

### Schritt-für-Schritt

```
┌─────────────────────────────────────────┐
│  Input:                                 │
│  F = 3.2 px (FWHM)                      │
│  s = 8.0                                │
│  T_min = 64 px                          │
│  W = 4096 px, H = 2048 px               │
│  D = 8                                  │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Step 1: Seeing-adaptive Größe          │
│                                         │
│  T_0 = s × F                            │
│      = 8.0 × 3.2                        │
│      = 25.6 px                          │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Step 2: Maximale Größe                 │
│                                         │
│  T_max = floor(min(W, H) / D)           │
│        = floor(min(4096, 2048) / 8)     │
│        = floor(2048 / 8)                │
│        = 256 px                         │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Step 3: Clamping                       │
│                                         │
│  T_0 = 25.6 px                          │
│  T_min = 64 px                          │
│  T_max = 256 px                         │
│                                         │
│  T = floor(clip(25.6, 64, 256))         │
│    = floor(64)                          │
│    = 64 px                              │
└─────────────────────────────────────────┘
```

### Warum diese Formel?

```
1. Seeing-Adaptivität (s × F):
   ┌────────────────────────────────┐
   │ Schlechtes Seeing (F=6px)      │
   │ → T_0 = 8×6 = 48 px            │
   │ → Größere Tiles nötig          │
   │   (mehr Struktur pro Tile)     │
   └────────────────────────────────┘
   
   ┌────────────────────────────────┐
   │ Gutes Seeing (F=2px)           │
   │ → T_0 = 8×2 = 16 px            │
   │ → Kleinere Tiles möglich       │
   │   (höhere Lokalität)           │
   └────────────────────────────────┘

2. Untere Schranke (T_min):
   ┌────────────────────────────────┐
   │ Verhindert zu kleine Tiles     │
   │ • Zu wenig Sterne/Struktur     │
   │ • Hohe statistische Varianz    │
   │ • Instabile Metriken           │
   └────────────────────────────────┘

3. Obere Schranke (T_max):
   ┌────────────────────────────────┐
   │ Verhindert zu große Tiles      │
   │ • Verlust der Lokalität        │
   │ • Keine lokale Seeing-Varianz  │
   │ • Nähert sich globalem Stack   │
   └────────────────────────────────┘
```

## Schritt 3.3: Overlap-Berechnung

### Formel

```
Gegeben:
  T  - Tile-Größe [Pixel]
  o  - tile.overlap_fraction (0 ≤ o ≤ 0.5)

Berechnung:
  O = floor(o × T)    (Overlap in Pixel)
  S = T - O           (Stride/Schritt in Pixel)
```

### Beispiel

```
T = 64 px
o = 0.25

O = floor(0.25 × 64) = 16 px
S = 64 - 16 = 48 px

Visualisierung:

Tile 0:     ┌────────────────┐
            │                │
            │    64 × 64     │
            │                │
            └────────────────┘
                    ↓ Stride = 48 px
Tile 1:             ┌────────────────┐
                    │ ←─ O=16 ─→     │
                    │    64 × 64     │
                    │                │
                    └────────────────┘
                            ↓ Stride = 48 px
Tile 2:                     ┌────────────────┐
                            │ ←─ O=16 ─→     │
                            │    64 × 64     │
                            │                │
                            └────────────────┘
```

### Warum Overlap?

```
Ohne Overlap:                Mit Overlap:
┌────┬────┬────┐            ┌────┬────┬────┐
│    │    │    │            │  ┌─┼─┬─┼─┬  │
│ T0 │ T1 │ T2 │            │  │ │ │ │ │  │
│    │    │    │            │  └─┼─┴─┼─┘  │
└────┴────┴────┘            └────┴────┴────┘
     ↑                           ↑
  Harte Kanten              Weiche Übergänge
  → Artefakte               → Glatte Rekonstruktion
```

## Schritt 3.4: Tile-Grid-Erzeugung

### Algorithmus

```
┌─────────────────────────────────────────┐
│  Input:                                 │
│  W, H - Bildgröße                       │
│  T    - Tile-Größe                      │
│  S    - Stride                          │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Berechne Tile-Positionen               │
│                                         │
│  x_positions = [0, S, 2S, 3S, ...]      │
│    bis x + T ≤ W                        │
│                                         │
│  y_positions = [0, S, 2S, 3S, ...]      │
│    bis y + T ≤ H                        │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Erzeuge Tile-Grid                      │
│                                         │
│  tiles = []                             │
│  for y in y_positions:                  │
│    for x in x_positions:                │
│      tile = {                           │
│        'id': len(tiles),                │
│        'x': x, 'y': y,                  │
│        'w': T, 'h': T                   │
│      }                                  │
│      tiles.append(tile)                 │
└─────────────────────────────────────────┘
```

### Visualisierung: Tile-Grid

```
Bild (W=256, H=192):
T=64, S=48 (O=16)

  0   48  96  144 192 240
0  ┌───┬───┬───┬───┐
   │ 0 │ 1 │ 2 │ 3 │
48 ├───┼───┼───┼───┤
   │ 4 │ 5 │ 6 │ 7 │
96 ├───┼───┼───┼───┤
   │ 8 │ 9 │10 │11 │
144└───┴───┴───┴───┘

Tile-Count:
  N_x = ceil((W - T) / S) + 1 = 4
  N_y = ceil((H - T) / S) + 1 = 3
  N_total = N_x × N_y = 12 Tiles
```

### Detaillierte Tile-Struktur

```python
tile = {
    'id': 5,           # Eindeutige ID
    'x': 48,           # Linke obere Ecke (x)
    'y': 48,           # Linke obere Ecke (y)
    'w': 64,           # Breite
    'h': 64,           # Höhe
    'x_end': 112,      # x + w
    'y_end': 112,      # y + h
    'center_x': 80,    # x + w/2
    'center_y': 80,    # y + h/2
    'area': 4096,      # w × h
}
```

## Schritt 3.5: Tile-Klassifikation

### Stern-Tiles vs. Struktur-Tiles

```
┌─────────────────────────────────────────┐
│

  Pro Tile t:                            │
│  1. Zähle Sterne im Tile                │
│  2. Berechne Struktur-Energie           │
│  3. Klassifiziere                       │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Klassifikation:                        │
│                                         │
│  if star_count ≥ 3:                     │
│    type = 'star'                        │
│  elif structure_energy > threshold:     │
│    type = 'structure'                   │
│  else:                                  │
│    type = 'background'                  │
└─────────────────────────────────────────┘
```

### Visualisierung

```
Bild mit Tile-Grid und Klassifikation:

┌─────────────────────────────────────┐
│ ★  ★ │ ★    │       │       │     │
│       │   ★  │       │       │     │
│   ★   │      │       │       │     │
├───────┼───────┼───────┼───────┼─────┤
│       │ ░░░░  │ ░░░░░ │       │     │
│       │ ░███░ │ ░███░ │       │     │
│       │ ░░░░  │ ░░░░░ │       │     │
├───────┼───────┼───────┼───────┼─────┤
│       │       │       │       │     │
│       │       │       │       │     │
│       │       │       │       │     │
└─────────────────────────────────────┘

Legende:
  ★ = Stern-Tile (≥3 Sterne)
  █ = Struktur-Tile (Nebel)
  ░ = Hintergrund-Tile
```

## Schritt 3.6: Validierung

```python
def validate_tile_grid(tiles, W, H, T, S, O):
    # Check 1: Overlap-Konsistenz
    assert 0 <= O <= T/2, "Overlap must be ≤ T/2"
    assert S == T - O, "Stride must equal T - O"
    
    # Check 2: Tile-Size-Monotonie
    # (wird bei Berechnung garantiert)
    
    # Check 3: Vollständige Abdeckung
    covered = np.zeros((H, W), dtype=bool)
    for tile in tiles:
        x, y, w, h = tile['x'], tile['y'], tile['w'], tile['h']
        covered[y:y+h, x:x+w] = True
    
    coverage = covered.sum() / (W * H)
    assert coverage > 0.95, f"Coverage {coverage:.1%} < 95%"
    
    # Check 4: Determinismus
    # Gleiche Inputs → gleiche Tile-Positionen
    tiles_check = generate_tiles(W, H, T, S)
    assert tiles == tiles_check, "Non-deterministic tile generation"
    
    # Check 5: Kanalunabhängigkeit
    # Gleiches Grid für R, G, B
    assert same_grid_for_all_channels()
```

## Output-Datenstruktur

```python
# Phase 3 Output
{
    'fwhm': {
        'measurements': List[float],  # Alle FWHM-Werte
        'robust_estimate': float,     # F (Median)
        'std': float,                 # Streuung
        'frame_medians': List[float], # Pro Frame
    },
    'tile_config': {
        'size': int,          # T
        'overlap': int,       # O
        'stride': int,        # S
        'size_factor': float, # s
        'min_size': int,      # T_min
        'max_divisor': int,   # D
    },
    'tiles': [
        {
            'id': int,
            'x': int, 'y': int,
            'w': int, 'h': int,
            'type': str,  # 'star', 'structure', 'background'
            'star_count': int,
        },
        ...
    ],
    'grid_stats': {
        'total_tiles': int,
        'tiles_x': int,
        'tiles_y': int,
        'coverage': float,  # 0.0-1.0
        'star_tiles': int,
        'structure_tiles': int,
        'background_tiles': int,
    }
}
```

## Beispiel-Konfiguration

```yaml
tile:
  size_factor: 8.0      # s (Multiplikator für FWHM)
  min_size: 64          # T_min (Pixel)
  max_divisor: 8        # D (Bild / D = max Tile-Größe)
  overlap_fraction: 0.25 # o (25% Overlap)

# Beispiel-Berechnung:
# FWHM = 3.5 px
# T_0 = 8.0 × 3.5 = 28 px
# T = clip(28, 64, 256) = 64 px
# O = 0.25 × 64 = 16 px
# S = 64 - 16 = 48 px
```

## Performance-Hinweise

```python
# Effiziente FWHM-Messung
def measure_fwhm_batch(frames):
    # Parallele Sternfindung
    with ThreadPoolExecutor() as executor:
        star_lists = executor.map(find_stars, frames)
    
    # Vektorisierte PSF-Fits
    fwhm_values = []
    for stars in star_lists:
        fwhms = fit_psf_batch(stars)  # GPU-beschleunigt
        fwhm_values.extend(fwhms)
    
    # Robuste Statistik
    F = robust_median(fwhm_values)
    return F

# Tile-Grid caching
@lru_cache(maxsize=1)
def get_tile_grid(W, H, T, S):
    # Grid wird nur einmal berechnet
    return generate_tiles(W, H, T, S)
```

## Nächste Phase

→ **Phase 4: Lokale Tile-Metriken und Qualitätsanalyse**
