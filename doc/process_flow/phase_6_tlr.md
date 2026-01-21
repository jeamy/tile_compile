# Phase 6: Tile-Local Registration & Reconstruction (TLR)

## Übersicht

**TLR (Tile-Local Registration)** ist das Herzstück der v4-Methodik. Jedes Tile registriert sich **lokal und unabhängig** von anderen Tiles, wodurch Feldrotation, differentielle Refraktion und lokale Seeing-Variationen korrekt behandelt werden.

## Kernkonzept

```
Für jedes Tile t:
  Für jede Iteration i = 1..4:
    1. Schätze Warp-Vektoren w_f,t für alle Frames f
    2. Berechne Cross-Correlation cc_f,t zwischen Frame f und Referenz
    3. Berechne Frame-Gewicht: R_{f,t} = exp(β·(cc_f,t - 1))
    4. Rekonstruiere Tile: I_t = Σ (G_f,c · R_{f,t}) · warp(I_f,t,c) / Σ weights
    5. Verwende rekonstruiertes Tile als neue Referenz für nächste Iteration
```

## Warum Tile-Local Registration?

### Problem: Globale Registration versagt bei Feldrotation

**Alt-Az-Montierung mit Feldrotation:**
```
Frame 0 (t=0min):     Frame 100 (t=50min):
    ★                      ★  ★
  ★   ★                  ★      ★
    ★                      ★

Globale Transformation:
  • Rotation um Bildmitte
  • Sterne an Rändern werden elongiert
  • ECC konvergiert schlecht
```

**Tile-Local Registration:**
```
Tile A (Bildmitte):       Tile B (Rand):
  Rotation: 0°              Rotation: 15°
  Translation: (0, 0)       Translation: (50, 30)
  
Jedes Tile hat eigene Transformation!
```

### Vorteile

1. **Feldrotation**: Jedes Tile kann unterschiedlich rotieren
2. **Differentielle Refraktion**: Atmosphärische Effekte variieren lokal
3. **Lokales Seeing**: Turbulenz ist räumlich inhomogen
4. **Tracking-Fehler**: Drift kann lokal kompensiert werden

## Iterative Reconstruction

### Algorithmus

```python
def tile_local_registration(
    tile_bbox: Tuple[int, int, int, int],
    frame_paths: List[str],
    global_weights: np.ndarray,  # G_f,c
    cfg: TileProcessorConfig,
    iterations: int = 4,
    beta: float = 6.0
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Tile-Local Registration & Reconstruction.
    
    Args:
        tile_bbox: (x0, y0, x1, y1) Tile-Bounding-Box
        frame_paths: Liste der Frame-Pfade
        global_weights: Globale Frame-Gewichte G_f,c
        cfg: Konfiguration
        iterations: Anzahl Iterationen (default: 4)
        beta: CC-Sensitivität (default: 6.0)
    
    Returns:
        (reconstructed_tile, metadata_list)
    """
    N = len(frame_paths)
    
    # Load all tile regions (debayered, normalized)
    tiles = []
    for f in range(N):
        tile = load_tile_normalized(
            frame_paths[f],
            tile_bbox,
            channel=cfg.channel,
            bayer_pattern=cfg.bayer_pattern,
            background=cfg.backgrounds[f]
        )
        tiles.append(tile)
    
    # Initialize reference (median of all tiles)
    reference = np.median(tiles, axis=0)
    
    # Iterative refinement
    warps_history = []
    cc_history = []
    
    for iteration in range(iterations):
        # Step 1: Estimate warp vectors for all frames
        warps = []
        for f in range(N):
            warp = estimate_warp_ecc(
                reference=reference,
                target=tiles[f],
                max_iterations=50,
                epsilon=1e-3
            )
            warps.append(warp)
        
        warps_history.append(warps)
        
        # Step 2: Compute cross-correlation
        cross_correlations = []
        for f in range(N):
            warped = apply_warp(tiles[f], warps[f])
            cc = compute_cross_correlation(reference, warped)
            cross_correlations.append(cc)
        
        cc_history.append(cross_correlations)
        
        # Step 3: Compute frame weights
        # R_{f,t} = exp(β·(cc - 1))
        frame_weights = np.array([
            np.exp(beta * (cc - 1.0))
            for cc in cross_correlations
        ])
        
        # Combine with global weights
        # W_{f,t,c} = G_f,c · R_{f,t}
        combined_weights = global_weights * frame_weights
        
        # Normalize weights
        weight_sum = np.sum(combined_weights)
        if weight_sum > 0:
            combined_weights /= weight_sum
        
        # Step 4: Weighted reconstruction
        reconstruction = np.zeros_like(reference)
        for f in range(N):
            warped = apply_warp(tiles[f], warps[f])
            reconstruction += combined_weights[f] * warped
        
        # Update reference for next iteration
        reference = reconstruction
    
    # Compute final metadata
    final_warps = warps_history[-1]
    final_cc = cc_history[-1]
    
    # Warp variance (splitting criterion)
    warp_variance = compute_warp_variance(final_warps)
    
    metadata = {
        'warp_variance': warp_variance,
        'cross_correlations': final_cc,
        'mean_cc': np.mean(final_cc),
        'iterations': iterations,
        'converged': iteration < iterations - 1,
        'warps': final_warps,
    }
    
    return reconstruction, metadata
```

### Warp-Schätzung (ECC)

```python
def estimate_warp_ecc(
    reference: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 50,
    epsilon: float = 1e-3,
    motion_type: str = 'euclidean'
) -> np.ndarray:
    """
    Schätzt Warp-Transformation mittels ECC (Enhanced Correlation Coefficient).
    
    ECC maximiert:
      ρ(T) = Σ I_ref(x) · I_target(T(x)) / √(Σ I_ref² · Σ I_target²)
    
    Args:
        reference: Referenz-Tile
        target: Zu registrierendes Tile
        max_iterations: Max ECC-Iterationen
        epsilon: Konvergenz-Schwelle
        motion_type: 'translation' oder 'euclidean'
    
    Returns:
        Warp-Matrix (2×3 für Affine)
    """
    # OpenCV ECC implementation
    if motion_type == 'translation':
        warp_mode = cv2.MOTION_TRANSLATION
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    else:  # euclidean
        warp_mode = cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    # Normalize images for better convergence
    ref_norm = normalize_for_ecc(reference)
    tgt_norm = normalize_for_ecc(target)
    
    # Run ECC
    try:
        _, warp_matrix = cv2.findTransformECC(
            templateImage=ref_norm,
            inputImage=tgt_norm,
            warpMatrix=warp_matrix,
            motionType=warp_mode,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                max_iterations,
                epsilon
            )
        )
    except cv2.error:
        # ECC failed, return identity
        pass
    
    return warp_matrix


def normalize_for_ecc(image: np.ndarray) -> np.ndarray:
    """
    Normalisiert Bild für ECC (0-255 uint8).
    """
    # Clip to percentiles to avoid outliers
    p1, p99 = np.percentile(image, [1, 99])
    clipped = np.clip(image, p1, p99)
    
    # Scale to 0-255
    normalized = ((clipped - p1) / (p99 - p1) * 255).astype(np.uint8)
    
    return normalized
```

### Cross-Correlation

```python
def compute_cross_correlation(
    reference: np.ndarray,
    target: np.ndarray
) -> float:
    """
    Berechnet normalisierte Cross-Correlation zwischen zwei Bildern.
    
    CC = Σ (I_ref - μ_ref) · (I_tgt - μ_tgt) / (σ_ref · σ_tgt · N)
    
    Returns:
        Cross-Correlation im Bereich [0, 1]
    """
    # Flatten images
    ref_flat = reference.flatten()
    tgt_flat = target.flatten()
    
    # Compute means
    mu_ref = np.mean(ref_flat)
    mu_tgt = np.mean(tgt_flat)
    
    # Compute standard deviations
    sigma_ref = np.std(ref_flat)
    sigma_tgt = np.std(tgt_flat)
    
    if sigma_ref == 0 or sigma_tgt == 0:
        return 0.0
    
    # Compute correlation
    cc = np.sum((ref_flat - mu_ref) * (tgt_flat - mu_tgt))
    cc /= (sigma_ref * sigma_tgt * len(ref_flat))
    
    # Clamp to [0, 1]
    cc = np.clip(cc, 0.0, 1.0)
    
    return cc
```

### Frame-Gewichtung

```python
def compute_frame_weight(cc: float, beta: float = 6.0) -> float:
    """
    Berechnet Frame-Gewicht basierend auf Cross-Correlation.
    
    R_{f,t} = exp(β·(cc - 1))
    
    Args:
        cc: Cross-Correlation [0, 1]
        beta: Sensitivitätsparameter (default: 6.0)
    
    Returns:
        Frame-Gewicht
    
    Beispiele (β=6.0):
        cc=1.00 → R=1.000 (perfekt)
        cc=0.95 → R=0.741 (5% Abweichung → 26% weniger Gewicht)
        cc=0.90 → R=0.549 (10% Abweichung → 45% weniger Gewicht)
        cc=0.80 → R=0.301 (20% Abweichung → 70% weniger Gewicht)
    """
    return np.exp(beta * (cc - 1.0))
```

## Multi-Pass Adaptive Refinement

### Konzept

Nach jedem Pass werden Tiles mit hoher **Warp-Varianz** in 4 Sub-Tiles gesplittet und erneut verarbeitet.

```
Pass 0: Initial Grid (z.B. 904 Tiles)
  → Process all tiles
  → Compute warp_variance per tile
  → Split tiles where warp_variance > threshold (z.B. 0.15)
  → Result: 3520 Tiles

Pass 1: Refined Grid (3520 Tiles)
  → Process all tiles (including new sub-tiles)
  → Compute warp_variance per tile
  → Split tiles where warp_variance > threshold
  → Result: 3580 Tiles (nur 60 neue)

Pass 2: Further Refinement (3580 Tiles)
  → Process all tiles
  → No tiles exceed threshold
  → Converged!
```

### Warp-Varianz

```python
def compute_warp_variance(warps: List[np.ndarray]) -> float:
    """
    Berechnet Varianz der Warp-Vektoren.
    
    Hohe Varianz → Frames haben sehr unterschiedliche Transformationen
                 → Tile sollte gesplittet werden
    
    Args:
        warps: Liste von Warp-Matrizen (2×3)
    
    Returns:
        Warp-Varianz
    """
    # Extract translation components
    translations = np.array([
        [warp[0, 2], warp[1, 2]]  # (dx, dy)
        for warp in warps
    ])
    
    # Compute variance of translations
    var_x = np.var(translations[:, 0])
    var_y = np.var(translations[:, 1])
    
    # Total variance
    warp_variance = var_x + var_y
    
    return warp_variance
```

### Tile-Splitting

```python
def split_tile(bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
    """
    Splittet ein Tile in 4 Sub-Tiles.
    
    ┌─────────┬─────────┐
    │  TL     │  TR     │
    ├─────────┼─────────┤
    │  BL     │  BR     │
    └─────────┴─────────┘
    
    Args:
        bbox: (x0, y0, x1, y1)
    
    Returns:
        Liste von 4 Sub-Tile-Bboxen
    """
    x0, y0, x1, y1 = bbox
    xm = (x0 + x1) // 2
    ym = (y0 + y1) // 2
    
    # Check minimum size
    min_size = 64
    if (xm - x0) < min_size or (ym - y0) < min_size:
        return [bbox]  # Don't split
    
    return [
        (x0, y0, xm, ym),  # Top-left
        (xm, y0, x1, ym),  # Top-right
        (x0, ym, xm, y1),  # Bottom-left
        (xm, ym, x1, y1),  # Bottom-right
    ]
```

### Refinement-Loop

```python
def adaptive_tile_refinement(
    initial_tiles: List[Tuple[int, int, int, int]],
    frame_paths: List[str],
    global_weights: np.ndarray,
    cfg: TileProcessorConfig,
    max_passes: int = 3,
    variance_threshold: float = 0.15
) -> Dict[Tuple, np.ndarray]:
    """
    Multi-Pass Adaptive Tile Refinement.
    
    Args:
        initial_tiles: Initiales Tile-Grid
        frame_paths: Frame-Pfade
        global_weights: Globale Gewichte
        cfg: Konfiguration
        max_passes: Max Refinement-Passes (default: 3)
        variance_threshold: Splitting-Schwelle (default: 0.15)
    
    Returns:
        Dict mapping bbox → reconstructed tile
    """
    tiles = initial_tiles.copy()
    all_results = {}
    
    for refine_pass in range(max_passes + 1):
        print(f"Pass {refine_pass}: Processing {len(tiles)} tiles...")
        
        # Process all tiles
        results = {}
        variances = {}
        
        for bbox in tiles:
            tile, metadata = tile_local_registration(
                bbox, frame_paths, global_weights, cfg
            )
            results[bbox] = tile
            variances[bbox] = metadata['warp_variance']
        
        # Update results
        all_results.update(results)
        
        # Adaptive refinement: split high-variance tiles
        if refine_pass < max_passes:
            new_tiles = []
            split_count = 0
            
            for bbox in tiles:
                if variances[bbox] > variance_threshold:
                    # Split this tile
                    sub_tiles = split_tile(bbox)
                    if len(sub_tiles) > 1:
                        new_tiles.extend(sub_tiles)
                        split_count += 1
                        # Remove old tile from results
                        del all_results[bbox]
                    else:
                        new_tiles.append(bbox)
                else:
                    # Keep this tile
                    new_tiles.append(bbox)
            
            if split_count == 0:
                print(f"Pass {refine_pass}: Converged (no tiles split)")
                break
            
            print(f"Pass {refine_pass}: Split {split_count} tiles "
                  f"({len(tiles)} → {len(new_tiles)})")
            tiles = new_tiles
        else:
            print(f"Pass {refine_pass}: Max passes reached")
    
    return all_results
```

## Parallele Verarbeitung

```python
def process_tiles_parallel(
    tiles: List[Tuple[int, int, int, int]],
    frame_paths: List[str],
    global_weights: np.ndarray,
    cfg: TileProcessorConfig,
    num_workers: int = 8
) -> Dict[Tuple, np.ndarray]:
    """
    Verarbeitet Tiles parallel.
    
    Args:
        tiles: Liste von Tile-Bboxen
        frame_paths: Frame-Pfade
        global_weights: Globale Gewichte
        cfg: Konfiguration
        num_workers: Anzahl paralleler Workers
    
    Returns:
        Dict mapping bbox → reconstructed tile
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    results = {}
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tiles
        futures = {
            executor.submit(
                tile_local_registration,
                bbox, frame_paths, global_weights, cfg
            ): bbox
            for bbox in tiles
        }
        
        # Collect results
        for future in as_completed(futures):
            bbox = futures[future]
            tile, metadata = future.result()
            results[bbox] = tile
    
    return results
```

## Validierung

```python
def validate_tlr(results: Dict, metadata: List[Dict]):
    """
    Validiert TLR-Ergebnisse.
    """
    checks = []
    
    # Check 1: All tiles reconstructed
    assert len(results) > 0
    checks.append(f"✓ {len(results)} tiles reconstructed")
    
    # Check 2: Warp variance convergence
    variances = [m['warp_variance'] for m in metadata]
    mean_var = np.mean(variances)
    max_var = np.max(variances)
    checks.append(f"✓ Warp variance: mean={mean_var:.3f}, max={max_var:.3f}")
    
    # Check 3: Cross-correlation quality
    mean_ccs = [m['mean_cc'] for m in metadata]
    overall_cc = np.mean(mean_ccs)
    assert overall_cc > 0.5, f"Low CC: {overall_cc}"
    checks.append(f"✓ Mean CC: {overall_cc:.3f}")
    
    # Check 4: No NaN/Inf
    for bbox, tile in results.items():
        assert not np.any(np.isnan(tile)), f"NaN in tile {bbox}"
        assert not np.any(np.isinf(tile)), f"Inf in tile {bbox}"
    checks.append("✓ No NaN/Inf in tiles")
    
    return checks
```

## Konfiguration

```yaml
v4:
  iterations: 4                        # Iterationen pro Tile
  beta: 6.0                            # CC-Sensitivität
  parallel_tiles: 8                    # Parallele Workers
  
  adaptive_tiles:
    enabled: true
    max_refine_passes: 3               # Max Refinement-Passes
    refine_variance_threshold: 0.15    # Splitting-Schwelle
    min_tile_size_px: 64               # Min Tile-Größe
```

## Nächste Phase

→ **Phase 7: Zustandsbasierte Clusterung**
