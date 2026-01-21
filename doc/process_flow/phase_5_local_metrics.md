# Phase 5: Lokale Metriken (computed during TLR)

## Übersicht

In v4 gibt es **keine separate Phase** für lokale Metriken. Diese werden **während der Tile-Local Registration (Phase 6)** berechnet.

Dies ist ein weiterer Unterschied zu v3, wo lokale Metriken in einer separaten Phase vor der Tile-Rekonstruktion berechnet wurden.

## Warum in TLR integriert?

### 1. Metriken sind Nebenprodukte der Rekonstruktion

Die wichtigsten lokalen Metriken in v4 sind:
- **Warp-Varianz**: Wird während Warp-Schätzung berechnet
- **Cross-Correlation**: Wird für Frame-Gewichtung benötigt
- **Frame-Gewichte R_{f,t}**: Direkt aus CC abgeleitet

Alle diese Metriken entstehen **natürlich** während der iterativen Rekonstruktion.

### 2. Keine Redundanz

**v3 Approach:**
```
Phase 4: Compute local metrics (FWHM, roundness, etc.)
  → Requires loading all tiles
  → Separate processing pass

Phase 5: Tile reconstruction
  → Loads tiles again
  → Uses metrics from Phase 4
```

**v4 Approach:**
```
Phase 6: TLR
  → Load tiles once
  → Compute metrics during reconstruction
  → Use metrics immediately
```

### 3. Iterative Verfeinerung

In v4 ändern sich die Metriken **während der Iterationen**:
- Iteration 1: Initiale Warp-Schätzung → initiale CC
- Iteration 2: Verbesserte Warps → bessere CC
- Iteration 3-4: Konvergenz

Separate Metrik-Berechnung würde nur Iteration 1 erfassen.

## Metriken in TLR

### 1. Warp-Varianz

**Berechnung:** Siehe Phase 6 (TLR)

```python
def compute_warp_variance(warps: List[np.ndarray]) -> float:
    """
    Varianz der Warp-Vektoren über alle Frames.
    
    Hohe Varianz → Frames haben sehr unterschiedliche Transformationen
                 → Tile sollte gesplittet werden
    """
    translations = np.array([
        [warp[0, 2], warp[1, 2]]
        for warp in warps
    ])
    
    var_x = np.var(translations[:, 0])
    var_y = np.var(translations[:, 1])
    
    return var_x + var_y
```

**Verwendung:**
- **Adaptive Refinement**: Tiles mit variance > threshold werden gesplittet
- **Qualitätskontrolle**: Hohe Varianz → schwierige Region (Feldrotation, etc.)

### 2. Cross-Correlation

**Berechnung:** Siehe Phase 6 (TLR)

```python
def compute_cross_correlation(reference: np.ndarray, target: np.ndarray) -> float:
    """
    Normalisierte Cross-Correlation.
    
    CC ∈ [0, 1]
    CC = 1 → perfekte Übereinstimmung
    CC < 0.5 → schlechte Übereinstimmung
    """
    # ... (siehe Phase 6)
```

**Verwendung:**
- **Frame-Gewichtung**: R_{f,t} = exp(β·(cc - 1))
- **Qualitätskontrolle**: Niedrige CC → Frame passt nicht gut zu Tile

### 3. Frame-Gewichte R_{f,t}

**Berechnung:**
```python
R_{f,t} = exp(β·(cc_{f,t} - 1))
```

**Verwendung:**
- **Rekonstruktion**: Kombiniert mit globalem Gewicht G_f,c
- **Effektives Gewicht**: W_{f,t,c} = G_f,c · R_{f,t}

### 4. Iterationskonvergenz

**Berechnung:**
```python
def check_convergence(
    reconstruction_prev: np.ndarray,
    reconstruction_curr: np.ndarray,
    epsilon: float = 1e-3
) -> bool:
    """
    Prüft Konvergenz zwischen Iterationen.
    
    Konvergiert wenn: ||I_i - I_{i-1}||_2 / ||I_i||_2 < ε
    """
    diff = reconstruction_curr - reconstruction_prev
    norm_diff = np.linalg.norm(diff)
    norm_curr = np.linalg.norm(reconstruction_curr)
    
    if norm_curr > 0:
        relative_change = norm_diff / norm_curr
    else:
        relative_change = 0.0
    
    return relative_change < epsilon
```

**Verwendung:**
- **Early Stopping**: Stoppe Iterationen wenn konvergiert
- **Qualitätskontrolle**: Nicht-Konvergenz → problematisches Tile

## Metrik-Aggregation

Nach TLR werden Metriken pro Tile gespeichert:

```python
# TLR Output Metadata
{
    'tile_id': int,
    'bbox': Tuple[int, int, int, int],
    'warp_variance': float,
    'cross_correlations': List[float],  # per frame
    'mean_cc': float,
    'frame_weights': List[float],       # R_{f,t}
    'iterations': int,
    'converged': bool,
    'final_warps': List[np.ndarray],
}
```

### Globale Statistiken

```python
def aggregate_tile_metrics(metadata_list: List[Dict]) -> Dict:
    """
    Aggregiert Metriken über alle Tiles.
    """
    warp_variances = [m['warp_variance'] for m in metadata_list]
    mean_ccs = [m['mean_cc'] for m in metadata_list]
    convergence_rates = [m['converged'] for m in metadata_list]
    
    return {
        'mean_warp_variance': np.mean(warp_variances),
        'max_warp_variance': np.max(warp_variances),
        'mean_cc': np.mean(mean_ccs),
        'min_cc': np.min(mean_ccs),
        'convergence_rate': np.mean(convergence_rates),
        'num_tiles': len(metadata_list),
    }
```

## Vergleich v3 vs v4

### v3 Lokale Metriken

```python
# Phase 4 (v3): Separate local metrics computation
for tile in tiles:
    for frame in frames:
        # Load tile region
        tile_region = load_tile(frame, tile.bbox)
        
        # Compute metrics
        fwhm = measure_fwhm(tile_region)
        roundness = measure_roundness(tile_region)
        contrast = measure_contrast(tile_region)
        
        # Store
        local_metrics[tile, frame] = {
            'fwhm': fwhm,
            'roundness': roundness,
            'contrast': contrast,
        }

# Phase 5 (v3): Use metrics for reconstruction
for tile in tiles:
    weights = []
    for frame in frames:
        # Retrieve stored metrics
        metrics = local_metrics[tile, frame]
        
        # Compute weight
        weight = compute_weight(metrics)
        weights.append(weight)
    
    # Reconstruct
    tile_reconstructed = weighted_sum(tiles, weights)
```

### v4 Lokale Metriken (in TLR)

```python
# Phase 6 (v4): Metrics computed during reconstruction
for tile in tiles:
    # Initialize
    reference = median(all_frames[tile.bbox])
    
    for iteration in range(4):
        # Estimate warps (metric: warp vectors)
        warps = [estimate_warp(reference, frame[tile.bbox]) 
                 for frame in frames]
        
        # Compute CC (metric: cross-correlation)
        ccs = [compute_cc(reference, warp(frame[tile.bbox], w))
               for frame, w in zip(frames, warps)]
        
        # Compute weights (metric: frame weights)
        weights = [exp(beta * (cc - 1)) for cc in ccs]
        
        # Reconstruct (using just-computed weights)
        reference = weighted_sum(
            [warp(frame[tile.bbox], w) for frame, w in zip(frames, warps)],
            weights
        )
    
    # Metrics are natural byproducts
    warp_variance = compute_variance(warps)
    mean_cc = np.mean(ccs)
```

## Vorteile der Integration

1. **Effizienz**: Keine separate Metrik-Berechnung nötig
2. **Konsistenz**: Metriken basieren auf finalen Warps, nicht initialen
3. **Adaptivität**: Metriken ändern sich während Iterationen
4. **Memory**: Keine Speicherung großer Metrik-Arrays

## Nächste Phase

→ **Phase 6: Tile-Local Registration & Reconstruction (TLR)** (wo die Metriken berechnet werden)
