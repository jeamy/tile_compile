# Phase 5: Tile-basierte Rekonstruktion

## Übersicht

Phase 5 ist das **Herzstück** der Methodik: Jedes Tile wird separat rekonstruiert durch gewichtetes Stacking aller Frames, wobei die effektiven Gewichte W_f,t,c die Qualität jedes Frames für jedes Tile widerspiegeln.

## Ziele

1. Gewichtete Rekonstruktion pro Tile
2. Overlap-Add für glatte Übergänge
3. Fensterfunktion zur Artefakt-Vermeidung
4. Fallback-Strategie für degenerierte Tiles
5. Kanalweise Verarbeitung (keine Farbmischung)

## Input

```python
# Aus Phase 2:
normalized_frames[c][f][x, y]  # Normalisierte Frames

# Aus Phase 3:
tiles = [...]  # Tile-Grid

# Aus Phase 4:
effective_weights[c][f, t]  # W_f,t,c = G_f,c × L_f,t,c
```

## Schritt 5.1: Gewichtete Tile-Rekonstruktion

### Normative Formel

```
Für jedes Tile t, Kanal c, Pixel p innerhalb des Tiles:

I_t,c(p) = Σ_f [W_f,t,c · I'_f,c(p)] / Σ_f W_f,t,c

wobei:
  I_t,c(p)   - Rekonstruiertes Tile an Position p
  W_f,t,c    - Effektives Gewicht (Frame f, Tile t, Kanal c)
  I'_f,c(p)  - Normalisierter Frame f an Position p
  Σ_f        - Summe über alle Frames
```

### Prozess

```
┌─────────────────────────────────────────┐
│  Tile t, Kanal c                        │
│  Position: (x, y, w, h)                 │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Step 1: Initialisierung                │
│                                          │
│  numerator = zeros(h, w)                │
│  denominator = zeros(h, w)              │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Step 2: Akkumulation über alle Frames  │
│                                          │
│  for f in range(N_frames):              │
│    tile_data = I'_f,c[y:y+h, x:x+w]    │
│    weight = W_f,t,c                     │
│                                          │
│    numerator += weight × tile_data      │
│    denominator += weight                │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Step 3: Division                       │
│                                          │
│  I_t,c = numerator / denominator        │
│                                          │
│  (elementweise Division)                │
└─────────────────────────────────────────┘
```

### Visualisierung: Gewichtetes Stacking

```
Frame 0 (W=2.0):        Frame 1 (W=5.0):        Frame 2 (W=1.0):
┌──────────┐            ┌──────────┐            ┌──────────┐
│  ★       │            │  ★       │            │  ★       │
│     ★    │            │     ★    │            │     ★    │
│          │  ×2.0      │          │  ×5.0      │          │  ×1.0
│    ★     │            │    ★     │            │    ★     │
│          │            │          │            │          │
│  ★   ★   │            │  ★   ★   │            │  ★   ★   │
└──────────┘            └──────────┘            └──────────┘
      │                       │                       │
      └───────────┬───────────┴───────────┬───────────┘
                  │                       │
                  ▼                       ▼
          Numerator = 2.0×F0 + 5.0×F1 + 1.0×F2
          Denominator = 2.0 + 5.0 + 1.0 = 8.0
                  │
                  ▼
          Rekonstruktion = Numerator / 8.0
                  │
                  ▼
            ┌──────────┐
            │  ★       │  ← Frame 1 dominiert
            │     ★    │     (höchstes Gewicht)
            │          │
            │    ★     │
            │          │
            │  ★   ★   │
            └──────────┘
```

### Beispiel-Berechnung (einzelnes Pixel)

```
Pixel (32, 32) in Tile t=5, Kanal R:

Frame  │ I'_f,R(32,32) │ W_f,5,R │ Beitrag
───────┼───────────────┼─────────┼──────────
  0    │    0.0234     │   2.0   │  0.0468
  1    │    0.0256     │   5.0   │  0.1280
  2    │    0.0198     │   1.0   │  0.0198
  3    │    0.0245     │   3.5   │  0.0858
  ...  │     ...       │   ...   │   ...
 799   │    0.0221     │   2.8   │  0.0619
───────┴───────────────┴─────────┴──────────
Summe  │               │  2847.3 │ 67.234

Rekonstruktion:
I_5,R(32,32) = 67.234 / 2847.3 = 0.0236
```

## Schritt 5.2: Fallback für degenerierte Tiles

### Problem

```
Degeneriertes Tile:
  • Alle Gewichte ≈ 0
  • Denominator ≈ 0
  • Division durch Null!
  
Ursachen:
  • Tile außerhalb des Objekts (nur Hintergrund)
  • Alle Frames haben schlechte Qualität in diesem Tile
  • Artefakte (Satellit, Flugzeug)
```

### Lösung (normativ)

```
┌─────────────────────────────────────────┐
│  Berechne Denominator D_t,c             │
│  D_t,c = Σ_f W_f,t,c                    │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Check: D_t,c ≥ ε ?                     │
│  (ε = kleine Konstante, z.B. 1e-6)      │
└────────────┬────────────────────────────┘
             │
       ┌─────┴─────┐
       │           │
    Ja │           │ Nein
       ▼           ▼
┌──────────┐  ┌──────────────────────┐
│ Normale  │  │ Fallback:            │
│ Rekon-   │  │ Ungewichtetes Mittel │
│ struktion│  │                      │
│          │  │ I_t,c = (1/N)·Σ_f I_f│
│          │  │                      │
│          │  │ Markiere:            │
│          │  │ fallback_used = true │
└──────────┘  └──────────────────────┘
```

### Implementierung

```python
def reconstruct_tile(frames, weights, tile, epsilon=1e-6):
    """
    Rekonstruiert ein einzelnes Tile.
    
    Args:
        frames: Liste von normalisierten Frames
        weights: Array von Gewichten W_f,t,c
        tile: Tile-Dictionary mit Position
        epsilon: Schwellwert für Fallback
    
    Returns:
        reconstructed: Rekonstruiertes Tile
        fallback_used: Boolean, ob Fallback verwendet wurde
    """
    h, w = tile['h'], tile['w']
    x, y = tile['x'], tile['y']
    
    # Initialisierung
    numerator = np.zeros((h, w), dtype=np.float32)
    denominator = 0.0
    
    # Akkumulation
    for f, frame in enumerate(frames):
        tile_data = frame[y:y+h, x:x+w]
        weight = weights[f]
        
        numerator += weight * tile_data
        denominator += weight
    
    # Check für Fallback
    if denominator >= epsilon:
        # Normale Rekonstruktion
        reconstructed = numerator / denominator
        fallback_used = False
    else:
        # Fallback: Ungewichtetes Mittel
        reconstructed = np.zeros((h, w), dtype=np.float32)
        for frame in frames:
            tile_data = frame[y:y+h, x:x+w]
            reconstructed += tile_data
        reconstructed /= len(frames)
        fallback_used = True
    
    return reconstructed, fallback_used
```

## Schritt 5.3: Fensterfunktion (Windowing)

### Zweck

```
Problem ohne Fensterfunktion:

Tile-Grenzen:
┌────────┬────────┐
│  Tile0 │ Tile1  │  ← Harte Kante bei Overlap
└────────┴────────┘
         ↑
    Sichtbare Artefakte!

Lösung mit Fensterfunktion:

┌────────┬────────┐
│  Tile0 │ Tile1  │  ← Weicher Übergang
└────────┴────────┘
    ╲    ╱
     ╲  ╱  ← Gewichtete Mischung
      ╲╱
```

### Cosine-Fenster (Tukey-Fenster)

```
Fensterfunktion w(x):

1.0 │    ┌───────────────┐

    │   ╱                 ╲
0.5 │  ╱                   ╲
    │ ╱                     ╲
0.0 └─────────────────────────
    0   O    T-O   T
    
    │←─ Fade-in ─→│←─ Plateau ─→│←─ Fade-out ─→│
    
Formel:
  w(x) = 0.5 × (1 + cos(π × (x - O) / O))     für 0 ≤ x < O
  w(x) = 1.0                                   für O ≤ x < T-O
  w(x) = 0.5 × (1 + cos(π × (x - (T-O)) / O)) für T-O ≤ x < T
```

### 2D-Fensterfunktion

```python
def create_window_2d(tile_size, overlap):
    """
    Erstellt 2D-Fensterfunktion (Cosine-Taper).
    
    Args:
        tile_size: Tile-Größe T
        overlap: Overlap-Größe O
    
    Returns:
        window: 2D-Array (T, T) mit Fensterfunktion
    """
    # 1D-Fenster
    window_1d = np.ones(tile_size)
    
    # Fade-in (linker Rand)
    for i in range(overlap):
        alpha = i / overlap
        window_1d[i] = 0.5 * (1 + np.cos(np.pi * (1 - alpha)))
    
    # Fade-out (rechter Rand)
    for i in range(overlap):
        alpha = i / overlap
        window_1d[tile_size - 1 - i] = 0.5 * (1 + np.cos(np.pi * (1 - alpha)))
    
    # 2D durch äußeres Produkt
    window_2d = np.outer(window_1d, window_1d)
    
    return window_2d
```

### Visualisierung: 2D-Fenster

```
2D-Fensterfunktion (64×64, O=16):

    1.0 │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        │ ░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░
        │ ░░▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒░░
        │ ░░▒▓████████████████████████▓▒░░
        │ ░░▒▓████████████████████████▓▒░░
        │ ░░▒▓████████████████████████▓▒░░
        │ ░░▒▓████████████████████████▓▒░░
        │ ░░▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒░░
        │ ░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░
    0.0 │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    
    Legende:
    █ = 1.0 (volle Gewichtung)
    ▓ = 0.75
    ▒ = 0.5
    ░ = 0.25
```

## Schritt 5.4: Overlap-Add

### Prozess

```
┌─────────────────────────────────────────┐
│  Rekonstruierte Tiles (mit Fenster)     │
│  I_t,c × window_t                       │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Initialisierung                        │
│                                          │
│  output = zeros(H, W)                   │
│  weight_sum = zeros(H, W)               │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Für jedes Tile t:                      │
│                                          │
│  x, y, w, h = tile['x'], ...            │
│  tile_recon = I_t,c × window            │
│                                          │
│  output[y:y+h, x:x+w] += tile_recon     │
│  weight_sum[y:y+h, x:x+w] += window     │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Normalisierung                         │
│                                          │
│  final = output / weight_sum            │
│                                          │
│  (elementweise Division)                │
└─────────────────────────────────────────┘
```

### Visualisierung: Overlap-Add

```
Schritt 1: Tile 0 platzieren
┌────────────────┐
│ ░░▒▒▓▓████▓▓▒▒ │  ← Mit Fenster
│ ░░▒▒▓▓████▓▓▒▒ │
└────────────────┘

Schritt 2: Tile 1 addieren (mit Overlap)
┌────────────────┐
│ ░░▒▒▓▓████▓▓▒▒ │
│ ░░▒▒▓▓████▓▓▒▒ │
└────────────────┘
        ┌────────────────┐
        │ ░░▒▒▓▓████▓▓▒▒ │  ← Überlappend
        │ ░░▒▒▓▓████▓▓▒▒ │
        └────────────────┘

Overlap-Region:
┌────────────────┬────────────────┐
│ ░░▒▒▓▓████▓▓▒▒ │ ░░▒▒▓▓████▓▓▒▒ │
│ ░░▒▒▓▓████▓▓▒▒ │ ░░▒▒▓▓████▓▓▒▒ │
└────────────────┴────────────────┘
        │←─ O ─→│
        
In Overlap-Region:
  output = Tile0 × w0 + Tile1 × w1
  weight_sum = w0 + w1
  final = output / weight_sum
  
  → Weicher Übergang!
```

### Implementierung

```python
def overlap_add(tiles_reconstructed, tiles, tile_size, overlap, H, W):
    """
    Kombiniert rekonstruierte Tiles mit Overlap-Add.
    
    Args:
        tiles_reconstructed: Liste von rekonstruierten Tiles
        tiles: Tile-Positionen
        tile_size: T
        overlap: O
        H, W: Output-Dimensionen
    
    Returns:
        final: Finales rekonstruiertes Bild (H, W)
    """
    # Fensterfunktion
    window = create_window_2d(tile_size, overlap)
    
    # Initialisierung
    output = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)
    
    # Overlap-Add
    for tile, tile_recon in zip(tiles, tiles_reconstructed):
        x, y, w, h = tile['x'], tile['y'], tile['w'], tile['h']
        
        # Fenster anwenden
        tile_windowed = tile_recon * window
        
        # Addieren
        output[y:y+h, x:x+w] += tile_windowed
        weight_sum[y:y+h, x:x+w] += window
    
    # Normalisierung
    # Verhindere Division durch Null (sollte nicht vorkommen)
    weight_sum = np.maximum(weight_sum, 1e-10)
    final = output / weight_sum
    
    return final
```

## Schritt 5.5: Tile-Normalisierung (optional)

### Hintergrundsubtraktion pro Tile

```
Problem:
  • Tiles können unterschiedliche Hintergrundniveaus haben
  • Führt zu Artefakten bei Overlap-Add

Lösung:
  • Subtrahiere lokalen Hintergrund NACH Rekonstruktion
  • Vor Overlap-Add
```

### Prozess

```python
def normalize_tile_background(tile_recon, tile_type):
    """
    Normalisiert Hintergrund eines rekonstruierten Tiles.
    
    Args:
        tile_recon: Rekonstruiertes Tile
        tile_type: 'star', 'structure', oder 'background'
    
    Returns:
        normalized: Normalisiertes Tile
    """
    if tile_type == 'background':
        # Hintergrund-Tiles: Median subtrahieren
        bg = np.median(tile_recon)
        normalized = tile_recon - bg + 1.0  # +1.0 für Konsistenz
    else:
        # Stern/Struktur-Tiles: Robuste Hintergrundschätzung
        bg = sigma_clipped_mean(tile_recon, sigma=3, maxiters=3)
        normalized = tile_recon - bg + 1.0
    
    return normalized
```

## Schritt 5.6: Validierung

```python
def validate_reconstruction(reconstructed, tiles, metadata):
    # Check 1: Keine NaN/Inf
    assert not np.any(np.isnan(reconstructed)), "NaN in reconstruction"
    assert not np.any(np.isinf(reconstructed)), "Inf in reconstruction"
    
    # Check 2: Positive Werte
    assert np.all(reconstructed >= 0), "Negative values in reconstruction"
    
    # Check 3: Fallback-Statistik
    fallback_count = sum(m['fallback_used'] for m in metadata)
    fallback_ratio = fallback_count / len(tiles)
    
    if fallback_ratio > 0.1:
        print(f"⚠ Warning: {fallback_ratio:.1%} tiles used fallback")
    
    # Check 4: Kanalunabhängigkeit
    # Rekonstruktion von R, G, B ist unabhängig
    assert no_channel_mixing()
    
    # Check 5: Determinismus
    # Gleiche Inputs → gleiche Rekonstruktion
    recon_check = reconstruct(frames, weights, tiles)
    assert np.allclose(reconstructed, recon_check, rtol=1e-5)
```

## Output-Datenstruktur

```python
# Phase 5 Output
{
    'reconstructed': {
        'R': np.ndarray,  # shape: (H, W), dtype: float32
        'G': np.ndarray,
        'B': np.ndarray,
    },
    'tile_metadata': [
        {
            'tile_id': int,
            'fallback_used': bool,
            'weight_sum': float,
            'mean_value': float,
            'std_value': float,
        },
        ...
    ],
    'statistics': {
        'R': {
            'mean': float,
            'std': float,
            'min': float,
            'max': float,
            'fallback_tiles': int,
            'fallback_ratio': float,
        },
        'G': {...},
        'B': {...},
    }
}
```

## Performance-Hinweise

```python
# Vektorisierte Rekonstruktion
def reconstruct_all_tiles_vectorized(frames, weights, tiles):
    """
    Rekonstruiert alle Tiles parallel (vektorisiert).
    """
    N_frames, H, W = frames.shape
    N_tiles = len(tiles)
    T = tiles[0]['w']  # Annahme: alle Tiles gleich groß
    
    # Pre-allocate
    tiles_recon = np.zeros((N_tiles, T, T), dtype=np.float32)
    
    # Batch-Verarbeitung
    for t, tile in enumerate(tiles):
        x, y = tile['x'], tile['y']
        
        # Extrahiere alle Tile-Daten auf einmal
        tile_stack = frames[:, y:y+T, x:x+T]  # (N_frames, T, T)
        
        # Gewichtete Summe (vektorisiert)
        w = weights[:, t]  # (N_frames,)
        numerator = np.sum(w[:, None, None] * tile_stack, axis=0)
        denominator = np.sum(w)
        
        # Rekonstruktion
        if denominator > 1e-6:
            tiles_recon[t] = numerator / denominator
        else:
            tiles_recon[t] = np.mean(tile_stack, axis=0)
    
    return tiles_recon

# GPU-Beschleunigung (optional)
import cupy as cp

def reconstruct_tiles_gpu(frames, weights, tiles):
    """
    GPU-beschleunigte Tile-Rekonstruktion.
    """
    # Transfer zu GPU
    frames_gpu = cp.asarray(frames)
    weights_gpu = cp.asarray(weights)
    
    # Rekonstruktion auf GPU
    tiles_recon_gpu = reconstruct_all_tiles_vectorized(
        frames_gpu, weights_gpu, tiles
    )
    
    # Transfer zurück zu CPU
    tiles_recon = cp.asnumpy(tiles_recon_gpu)
    
    return tiles_recon
```

## Beispiel-Workflow

```python
# Kompletter Rekonstruktions-Workflow
def phase5_reconstruction(frames, global_weights, local_weights, tiles, config):
    """
    Phase 5: Tile-basierte Rekonstruktion.
    """
    H, W = frames[0].shape
    T = config['tile_size']
    O = config['overlap']
    
    # Effektive Gewichte
    W_eff = global_weights[:, None] * local_weights  # (N_frames, N_tiles)
    
    # Rekonstruiere alle Tiles
    tiles_recon = []
    metadata = []
    
    for t, tile in enumerate(tiles):
        recon, fallback = reconstruct_tile(
            frames, W_eff[:, t], tile
        )
        
        # Optional: Hintergrund-Normalisierung
        if config.get('normalize_tile_background', False):
            recon = normalize_tile_background(recon, tile['type'])
        
        tiles_recon.append(recon)
        metadata.append({
            'tile_id': t,
            'fallback_used': fallback,
            'weight_sum': np.sum(W_eff[:, t]),
        })
    
    # Overlap-Add
    final = overlap_add(tiles_recon, tiles, T, O, H, W)
    
    # Validierung
    validate_reconstruction(final, tiles, metadata)
    
    return final, metadata
```

## Nächste Phase

→ **Phase 6: Zustandsbasierte Clusterung und synthetische Frames**
