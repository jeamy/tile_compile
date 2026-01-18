# Phase 4: Lokale Tile-Metriken und Qualitätsanalyse

## Übersicht

Phase 4 berechnet für jedes Tile in jedem Frame lokale Qualitätsmetriken. Diese ermöglichen eine **räumlich adaptive** Rekonstruktion, die lokale Seeing-Variationen berücksichtigt.

## Ziele

1. Lokale Metriken pro Tile berechnen (FWHM, Rundheit, Kontrast)
2. Tile-Typ-spezifische Analysen (Stern vs. Struktur)
3. Lokale Qualitätsindizes berechnen
4. Effektive Gewichte kombinieren (global × lokal)

## Input

```python
# Aus Phase 2:
normalized_frames[c][f][x, y]  # Normalisierte Frames
global_weights[c][f]           # G_f,c

# Aus Phase 3:
tiles = [                      # Tile-Grid
    {'id': 0, 'x': 0, 'y': 0, 'w': 64, 'h': 64, 'type': 'star'},
    ...
]
```

## Schritt 4.1: Tile-Extraktion

### Prozess

```
┌─────────────────────────────────────────┐
│  Frame f, Kanal c, Tile t               │
│  I'_f,c[x,y] (normalized)               │
│  tile = {x, y, w, h}                    │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Tile-Cutout extrahieren                │
│                                         │
│  tile_data = I'_f,c[y:y+h, x:x+w]       │
│                                         │
│  Shape: (h, w) = (64, 64)               │
└─────────────────────────────────────────┘
```

### Visualisierung

```
Gesamtbild (512×512):          Tile-Cutout (64×64):

┌────────────────────┐         ┌──────────┐
│                    │         │  ★       │
│    ┌──────────┐    │         │     ★    │
│    │  ★       │    │   →     │          │
│    │     ★    │    │         │    ★    │
│    │          │    │         │          │
│    │    ★     │    │         │  ★   ★  │
│    │          │    │         │          │
│    │  ★   ★   │    │        │      ★  │
│    │          │    │         └──────────┘
│    │      ★   │    │
│    └──────────┘    │
│                    │
└────────────────────┘
```

## Schritt 4.1.1: Tile-basierte Rauschunterdrückung (optional)

Falls `tile_denoising.enabled = true`, wird vor der Metrik-Berechnung eine adaptive Rauschunterdrückung auf Tile-Ebene angewendet.

### Algorithmus: Highpass + Soft-Threshold

```
┌─────────────────────────────────────────┐
│  Tile T_t (z.B. 64×64)                  │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  1. Background-Schätzung                │
│     B_t = box_blur(T_t, k)              │
│     k = tile_denoising.kernel_size      │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  2. Residuum (Highpass)                 │
│     R_t = T_t − B_t                     │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  3. Robuste Rauschschätzung (MAD)       │
│     σ_t = 1.4826 · median(|R_t - med|)  │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  4. Soft-Threshold                      │
│     τ = α · σ_t                         │
│     R'_t = sign(R_t) · max(|R_t| − τ, 0)│
│     α = tile_denoising.alpha            │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  5. Rekonstruktion                      │
│     T'_t = B_t + R'_t                   │
└─────────────────────────────────────────┘
```

### Overlap-Blending

Da Tiles überlappen, werden die denoisten Tiles mit linearen Gewichten geblendet:

```
Gewichtsfunktion:           Blending-Beispiel:

  1 ┤   ╱──╲                ┌───┬───┬───┐
    │  ╱    ╲               │ A │A+B│ B │
  0 ┼─╱      ╲──            ├───┼───┼───┤
    0       64              │A+C│ALL│B+D│
                            ├───┼───┼───┤
  w(x,y) = ramp(x)·ramp(y)  │ C │C+D│ D │
                            └───┴───┴───┘
```

### Konfiguration

| Parameter | Beschreibung | Default | Empfohlen |
|-----------|--------------|---------|-----------|
| `tile_denoising.enabled` | Aktivierung | false | true |
| `tile_denoising.kernel_size` | Box-Blur Kernelgröße | 15 | 31 |
| `tile_denoising.alpha` | Threshold-Multiplikator | 2.0 | 1.5 |

### Typische Ergebnisse

| kernel | alpha | Noise-Red. | Stern-Erhalt |
|--------|-------|------------|--------------|
| 15 | 2.0 | ~75% | ~91% |
| **31** | **1.5** | **~89%** | **~93%** |
| 31 | 2.0 | ~89% | ~91% |

## Schritt 4.2: Stern-Tile-Metriken

### Für Tiles mit type='star'

```
┌─────────────────────────────────────────┐
│  Tile-Cutout mit Sternen                │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Metrik 1: FWHM (lokales Seeing)        │
│                                         │
│  1. Finde Sterne im Tile                │
│  2. Fitte PSF pro Stern                 │
│  3. Berechne FWHM pro Stern             │
│  4. Median über alle Sterne im Tile     │
│                                         │
│  FWHM_f,t,c = median(FWHM_stars)        │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Metrik 2: Rundheit (Tracking-Qualität) │
│                                         │
│  Pro Stern:                             │
│    Berechne Elliptizität e = 1 - b/a    │
│    wobei a, b = Halbachsen              │
│                                         │
│  roundness_f,t,c = 1 - median(e)        │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Metrik 3: Kontrast (SNR)               │
│                                         │
│  Pro Stern:                             │
│    peak = max(PSF)                      │
│    background = median(tile_data)       │
│    noise = std(background_pixels)       │
│                                         │
│  contrast_f,t,c = (peak - bg) / noise   │
│                 = median(contrasts)     │
└─────────────────────────────────────────┘
```

### Detaillierte FWHM-Messung

```
Stern im Tile:

Intensität (1D-Schnitt):
    │
    │      ╱‾‾‾╲
    │     ╱     ╲
Max │────●───────●────
    │   ╱         ╲
1/2 │──●───────────●──
    │ ╱             ╲
    │╱               ╲___
    └──────────────────────► Position
       │←── FWHM ──→│

PSF-Fit (2D Gaussian):
  PSF(x,y) = A·exp(-((x-x₀)²/σₓ² + (y-y₀)²/σᵧ²)/2)
  
  FWHM_x = 2.355 × σₓ
  FWHM_y = 2.355 × σᵧ
  FWHM = √(FWHM_x × FWHM_y)
```

### Rundheits-Berechnung

```
Elliptische PSF:

    ╱───╲       a = große Halbachse
   │  ●  │      b = kleine Halbachse
    ╲───╱       
    │←a→│       Elliptizität: e = 1 - b/a
      ↕         Rundheit: r = b/a = 1 - e
      b

Perfekt rund:  a = b  →  e = 0, r = 1.0
Elongiert:     a > b  →  e > 0, r < 1.0

Beispiele:
  r = 1.0  →  Perfekt rund (ideales Tracking)
  r = 0.8  →  Leicht elongiert
  r = 0.5  →  Stark elongiert (schlechtes Tracking)
```

### Kontrast-Berechnung

```
Tile mit Stern:

┌────────────────────┐
│ ░░░░░░░░░░░░░░░░░░ │  ← Hintergrund (bg)
│ ░░░░░░░░░░░░░░░░░░ │
│ ░░░░░░░███░░░░░░░░ │  ← Stern (peak)
│ ░░░░░░█████░░░░░░░ │
│ ░░░░░░░███░░░░░░░░ │
│ ░░░░░░░░░░░░░░░░░░ │
└────────────────────┘

Kontrast = (peak - bg) / noise

Hoher Kontrast:
  • Heller Stern
  • Niedriges Rauschen
  → Gute Qualität

Niedriger Kontrast:
  • Schwacher Stern
  • Hohes Rauschen
  → Schlechte Qualität
```

## Schritt 4.3: Struktur-Tile-Metriken

### Für Tiles mit type='structure' (Nebel)

```
┌─────────────────────────────────────────┐
│  Tile-Cutout mit Nebelstruktur          │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Metrik 1: Energie/Rausch-Verhältnis    │
│                                         │
│  1. Berechne Gradientenergie E          │
│     (wie in Phase 2, aber lokal)        │
│                                         │
│  2. Schätze lokales Rauschen σ          │
│                                         │
│  3. ENR_f,t,c = E / σ                   │
│     (Energy-to-Noise Ratio)             │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Metrik 2: Lokaler Hintergrund          │
│                                         │
│  B_local_f,t,c = median(tile_data)      │
│                                         │
│  (Hintergrundniveau im Tile)            │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Metrik 3: Strukturvarianz              │
│                                         │
│  var_f,t,c = std(tile_data)             │
│                                         │
│  (Variabilität der Struktur)            │
└─────────────────────────────────────────┘
```

### Visualisierung: Nebel-Tile

```
Tile mit Nebelstruktur:

┌────────────────────┐
│ ░░░░░▒▒▒░░░░░░░░░░ │  ← Schwache Struktur
│ ░░░░▒▓▓▓▒░░░░░░░░░ │
│ ░░░▒▓███▓▒░░░░░░░░ │  ← Helle Nebelregion
│ ░░░▒▓▓▓▓▒░░░░░░░░░ │
│ ░░░░▒▒▒░░░░░░░░░░░ │
│ ░░░░░░░░░░░░░░░░░░ │  ← Hintergrund
└────────────────────┘

Gradienten (Sobel):

┌────────────────────┐
│ ░░░░░████░░░░░░░░░ │  ← Hohe Gradienten
│ ░░░░██░░██░░░░░░░░ │     an Kanten
│ ░░░██░░░░██░░░░░░░ │
│ ░░░██░░░░██░░░░░░░ │
│ ░░░░██░░██░░░░░░░░ │
│ ░░░░░████░░░░░░░░░ │
└────────────────────┘

E = mean(gradient²)  →  Strukturenergie
σ = std(background)  →  Rauschen
ENR = E / σ²         →  Signal-to-Noise
```

## Schritt 4.4: Lokaler Qualitätsindex

### Formel (Stern-Tiles)

```
Q_local_f,t,c = 0.6 · (−FWHM̃_f,t,c) + 0.2 · r̃_f,t,c + 0.2 · C̃_f,t,c

wobei (robuste Skalierung mit Median + MAD über alle Stern-Tiles):
  FWHM̃_f,t,c = (FWHM_f,t,c − median(FWHM))
                / (1.4826 · MAD(FWHM))
  r̃_f,t,c     = (roundness_f,t,c − median(roundness))
                / (1.4826 · MAD(roundness))
  C̃_f,t,c     = (contrast_f,t,c − median(contrast))
                / (1.4826 · MAD(contrast))

  Kleinere FWHM̃ sind besser (daher Vorzeichen − vor FWHM̃ im Q_local).
```

### Formel (Struktur-Tiles)

```
Q_local_f,t,c = 0.7 · ENR̃_f,t,c − 0.3 · B̃_local,f,t,c

wobei (robuste Skalierung mit Median + MAD über alle Struktur-Tiles):
  ENR̃_f,t,c      = (ENR_f,t,c − median(ENR))
                    / (1.4826 · MAD(ENR))
  B̃_local,f,t,c  = (B_local_f,t,c − median(B_local))
                    / (1.4826 · MAD(B_local))

  Ein expliziter Varianz-Term geht in v3.1 **nicht** in Q_local ein.
```

### Prozess

```
┌─────────────────────────────────────────┐
│  Metriken für alle Tiles:               │
│  FWHM_f,t,c, roundness_f,t,c, ...       │
│  (f = frames, t = tiles, c = channels)  │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Robuste Normalisierung (pro Metrik)    │
│                                          │
│  median_FWHM = median(FWHM_f,t,c)       │
│  MAD_FWHM    = MAD(FWHM_f,t,c)          │
│  FWHM̃       = (FWHM − median_FWHM)
│                / (1.4826 · MAD_FWHM)    │
│                                          │
│  (analog für andere Metriken mit
│   Median + MAD)                          │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Gewichtete Kombination                 │
│                                          │
│  Q_local = Σ wᵢ · metric̃ᵢ              │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Clamping (Stabilität)                  │
│                                          │
│  Q_local = clip(Q_local, -3, +3)        │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Exponential Mapping                    │
│                                          │
│  L_f,t,c = exp(Q_local_f,t,c)           │
│                                          │
│  Wertebereich: [e⁻³, e³] ≈ [0.05, 20.1]│
└─────────────────────────────────────────┘
```

## Schritt 4.5: Effektives Gewicht

### Kombination: Global × Lokal

```
W_f,t,c = G_f,c × L_f,t,c

wobei:
  G_f,c   - Globales Frame-Gewicht (aus Phase 2)
  L_f,t,c - Lokales Tile-Gewicht (aus Phase 4)
  W_f,t,c - Effektives Gewicht für Tile t in Frame f
```

### Visualisierung

```
Frame f=10, Kanal R:

Global Weight: G_10,R = 5.0  (gutes Frame)

Lokale Weights (pro Tile):

Tile-Grid mit L_10,t,R:

┌──────┬──────┬──────┬──────┐
│ 0.5  │ 2.0  │ 3.5  │ 1.0  │  ← Schlechtes Seeing links
├──────┼──────┼──────┼──────┤
│ 1.2  │ 8.0  │ 7.5  │ 2.5  │  ← Gutes Seeing Mitte
├──────┼──────┼──────┼──────┤
│ 0.8  │ 3.0  │ 4.0  │ 1.5  │
└──────┴──────┴──────┴──────┘

Effektive Weights: W = G × L

┌──────┬──────┬──────┬──────┐
│ 2.5  │ 10.0 │ 17.5 │ 5.0  │
├──────┼──────┼──────┼──────┤
│ 6.0  │ 40.0 │ 37.5 │ 12.5 │  ← Höchstes Gewicht
├──────┼──────┼──────┼──────┤
│ 4.0  │ 15.0 │ 20.0 │ 7.5  │
└──────┴──────┴──────┴──────┘

Interpretation:
  • Tile (1,1): W = 40.0 → Beste Qualität
  • Tile (0,0): W = 2.5  → Schlechteste Qualität
```

### Warum Global × Lokal?

```
Nur Global (G):
  ┌─────────────────────────────┐
  │ Gesamtes Frame gleich       │
  │ gewichtet                   │
  │ → Keine lokale Adaptivität  │
  └─────────────────────────────┘

Nur Lokal (L):
  ┌─────────────────────────────┐
  │ Frame-Qualität ignoriert    │
  │ → Schlechte Frames können   │
  │   hohe lokale Gewichte haben│
  └─────────────────────────────┘

Global × Lokal (W):
  ┌─────────────────────────────┐
  │ Beste Kombination:          │
  │ • Gute Frames bevorzugt     │
  │ • Lokale Variationen        │
  │   berücksichtigt            │
  └─────────────────────────────┘
```

## Schritt 4.6: Heatmap-Visualisierung

### Gewichts-Heatmap pro Frame

```python
def visualize_tile_weights(frame_id, channel, tiles, weights):
    # Erstelle Heatmap
    heatmap = np.zeros((H, W))
    
    for tile, weight in zip(tiles, weights[frame_id]):
        x, y, w, h = tile['x'], tile['y'], tile['w'], tile['h']
        heatmap[y:y+h, x:x+w] = weight
    
    # Visualisierung
    plt.imshow(heatmap, cmap='hot')
    plt.colorbar(label='Weight W_f,t,c')
    plt.title(f'Frame {frame_id}, Channel {channel}')
```

### Beispiel-Heatmap

```
Frame 42, R-Kanal:

    Gewicht W_42,t,R
    
    │ 40│████████████░░░░░░░░░░░░
    │ 35│████████████░░░░░░░░░░░░
    │ 30│████████████████░░░░░░░░
    │ 25│████████████████░░░░░░░░
    │ 20│████████████████████░░░░
    │ 15│████████████████████░░░░
    │ 10│████████████████████████
    │  5│████████████████████████
    │  0│████████████████████████
    └───┴────────────────────────
    
    Legende:
    █ = Hohe Gewichte (gute Qualität)
    ░ = Niedrige Gewichte (schlechte Qualität)
    
    → Rechts oben: Schlechtes lokales Seeing
    → Mitte/Links: Gutes lokales Seeing
```

## Schritt 4.7: Validierung

```python
def validate_local_metrics(tiles, metrics, weights):
    # Check 1: Metrik-Vollständigkeit
    for f in range(N_frames):
        for t in range(N_tiles):
            for c in ['R', 'G', 'B']:
                assert (f, t, c) in metrics, \
                    f"Missing metrics for frame {f}, tile {t}, channel {c}"
    
    # Check 2: Positive Gewichte
    for w in weights.values():
        assert np.all(w > 0), "All weights must be positive"
    
    # Check 3: Clamping
    for Q in [m['Q_local'] for m in metrics.values()]:
        assert -3.0 <= Q <= 3.0, "Q_local not clamped"
    
    # Check 4: Kanalunabhängigkeit
    # Keine Metrik mischt R/G/B
    for tile_type in ['star', 'structure']:
        assert metrics_independent_per_channel(tile_type)
    
    # Check 5: Determinismus
    # Gleiche Inputs → gleiche Metriken
    metrics_check = compute_local_metrics(frames, tiles)
    assert metrics == metrics_check
```

## Output-Datenstruktur

```python
# Phase 4 Output
{
    'local_metrics': {
        'R': {  # Pro Kanal
            (f, t): {  # Pro Frame f, Tile t
                'tile_id': int,
                'frame_id': int,
                'type': str,  # 'star' or 'structure'
                
                # Stern-Metriken (wenn type='star')
                'fwhm': float,
                'roundness': float,
                'contrast': float,
                'star_count': int,
                
                # Struktur-Metriken (wenn type='structure')
                'enr': float,
                'background_local': float,
                'variance': float,
                
                # Qualitätsindex
                'Q_local': float,  # clamped [-3, 3]
                'L': float,        # exp(Q_local)
            },
            ...
        },
        'G': {...},
        'B': {...},
    },
    'effective_weights': {
        'R': np.ndarray,  # shape: (N_frames, N_tiles)
        'G': np.ndarray,  # W_f,t,c = G_f,c × L_f,t,c
        'B': np.ndarray,
    },
    'statistics': {
        'R': {
            'fwhm_mean': float, 'fwhm_std': float,
            'roundness_mean': float, 'roundness_std': float,
            'L_mean': float, 'L_std': float,
            'W_mean': float, 'W_std': float,
        },
        'G': {...},
        'B': {...},
    }
}
```

## Performance-Hinweise

```python
# Parallele Tile-Metrik-Berechnung
def compute_tile_metrics_parallel(frames, tiles, channel):
    from concurrent.futures import ProcessPoolExecutor
    
    # Chunking für bessere Cache-Nutzung
    def process_chunk(chunk):
        results = {}
        for (f, t) in chunk:
            tile_data = extract_tile(frames[f], tiles[t])
            metrics = compute_metrics(tile_data, tiles[t]['type'])
            results[(f, t)] = metrics
        return results
    
    # Parallel processing
    chunks = create_chunks(N_frames, N_tiles, chunk_size=1000)
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_chunk, chunks)
    
    # Merge results
    all_metrics = {}
    for result in results:
        all_metrics.update(result)
    
    return all_metrics

# Memory-effiziente Speicherung
# Sparse matrix für Gewichte (viele Tiles, wenige Frames)
from scipy.sparse import csr_matrix

weights_sparse = csr_matrix(weights)  # Komprimiert
```

## Nächste Phase

→ **Phase 5: Tile-basierte Rekonstruktion**
