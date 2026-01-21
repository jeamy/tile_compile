# Phase 8: Synthetische Frames

## Übersicht

Phase 8 ist **identisch zu v3**. Aus jedem Cluster wird ein **synthetisches Frame** erzeugt durch gewichtetes Stacking der Cluster-Mitglieder.

Diese Phase wird nur im **Normal Mode** (N ≥ 200 Frames) ausgeführt.

## Ziele

1. Frame-Reduktion: N → K (typisch 200 → 20)
2. Rauschreduktion durch Cluster-Stacking
3. Gewichtserhaltung
4. Vorbereitung für finales Stacking (Phase 9)

## Konzept

```
Cluster k mit Frames {f₁, f₂, ..., f_m}:
  
  Synthetisches Frame:
    I_synth,k = Σ W_f · I_f / Σ W_f
    
  Synthetisches Gewicht:
    W_synth,k = Σ W_f
```

## Implementierung

### Synthetisches Frame pro Cluster

```python
def create_synthetic_frame(
    cluster_frames: List[np.ndarray],
    cluster_weights: np.ndarray,
    weighting_mode: str = 'global'
) -> Tuple[np.ndarray, float]:
    """
    Erzeugt synthetisches Frame aus Cluster.
    
    Args:
        cluster_frames: Liste von Frames im Cluster
        cluster_weights: Gewichte der Frames
        weighting_mode: 'global' oder 'tile_weighted'
    
    Returns:
        (synthetic_frame, synthetic_weight)
    """
    # Normalize weights
    weights_norm = cluster_weights / np.sum(cluster_weights)
    
    # Weighted stack
    synthetic = np.zeros_like(cluster_frames[0], dtype=np.float32)
    for frame, weight in zip(cluster_frames, weights_norm):
        synthetic += weight * frame
    
    # Synthetic weight (sum of original weights)
    synthetic_weight = np.sum(cluster_weights)
    
    return synthetic, synthetic_weight
```

### Alle Cluster verarbeiten

```python
def create_synthetic_frames(
    frames: np.ndarray,           # [N × H × W]
    cluster_labels: np.ndarray,   # [N]
    global_weights: np.ndarray,   # [N]
    K: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Erzeugt synthetische Frames für alle Cluster.
    
    Args:
        frames: Original-Frames
        cluster_labels: Cluster-Zuordnungen
        global_weights: Globale Frame-Gewichte
        K: Anzahl Cluster
    
    Returns:
        (synthetic_frames, synthetic_weights)
          synthetic_frames: [K × H × W]
          synthetic_weights: [K]
    """
    H, W = frames.shape[1], frames.shape[2]
    
    synthetic_frames = np.zeros((K, H, W), dtype=np.float32)
    synthetic_weights = np.zeros(K, dtype=np.float32)
    
    for k in range(K):
        # Get frames in this cluster
        mask = cluster_labels == k
        cluster_frame_indices = np.where(mask)[0]
        
        # Extract frames and weights
        cluster_frames = frames[cluster_frame_indices]
        cluster_weights = global_weights[cluster_frame_indices]
        
        # Create synthetic frame
        synthetic, synth_weight = create_synthetic_frame(
            cluster_frames,
            cluster_weights
        )
        
        synthetic_frames[k] = synthetic
        synthetic_weights[k] = synth_weight
        
        print(f"Cluster {k}: {len(cluster_frame_indices)} frames "
              f"→ synthetic (weight: {synth_weight:.2f})")
    
    return synthetic_frames, synthetic_weights
```

## Weighting Modes

### Mode 1: Global Weighting (Standard)

```python
# Verwende globale Gewichte G_f,c aus Phase 3
synthetic_weight = Σ G_f,c  (für alle f in Cluster k)
```

**Vorteil:** Einfach, konsistent mit globaler Qualitätsbewertung

### Mode 2: Tile-Weighted

```python
# Verwende mittlere effektive Gewichte aus TLR
W_eff,f = mean over tiles (G_f,c · R_{f,t})
synthetic_weight = Σ W_eff,f  (für alle f in Cluster k)
```

**Vorteil:** Berücksichtigt lokale Qualität (TLR-Ergebnisse)

## Rauschreduktion

### SNR-Verbesserung

```
SNR_synth = SNR_single · √m

wobei:
  m = Anzahl Frames im Cluster
  SNR_single = Signal-to-Noise eines einzelnen Frames
```

**Beispiel:**
- Cluster mit 10 Frames: SNR-Verbesserung = √10 ≈ 3.16×
- 20 Cluster à 10 Frames: Gesamt-SNR ≈ 3.16× besser als Einzelframe

### Effektive Integration Time

```
t_eff,synth = Σ (W_f · t_exp,f) / W_synth

wobei:
  t_exp,f = Belichtungszeit von Frame f
  W_f = Gewicht von Frame f
```

## Validierung

```python
def validate_synthetic_frames(
    synthetic_frames: np.ndarray,
    synthetic_weights: np.ndarray,
    original_weights: np.ndarray,
    K: int
):
    """
    Validiert synthetische Frames.
    """
    checks = []
    
    # Check 1: K synthetic frames created
    assert len(synthetic_frames) == K
    assert len(synthetic_weights) == K
    checks.append(f"✓ {K} synthetic frames created")
    
    # Check 2: Weight conservation
    total_original_weight = np.sum(original_weights)
    total_synthetic_weight = np.sum(synthetic_weights)
    weight_ratio = total_synthetic_weight / total_original_weight
    
    assert 0.95 < weight_ratio < 1.05, f"Weight not conserved: {weight_ratio}"
    checks.append(f"✓ Weight conserved: {weight_ratio:.3f}")
    
    # Check 3: No NaN/Inf
    for k in range(K):
        assert not np.any(np.isnan(synthetic_frames[k]))
        assert not np.any(np.isinf(synthetic_frames[k]))
    checks.append("✓ No NaN/Inf in synthetic frames")
    
    # Check 4: Reasonable dynamic range
    for k in range(K):
        min_val = np.min(synthetic_frames[k])
        max_val = np.max(synthetic_frames[k])
        assert min_val >= 0, f"Negative values in synthetic frame {k}"
        assert max_val < 65536, f"Overflow in synthetic frame {k}"
    checks.append("✓ Reasonable dynamic range")
    
    return checks
```

## Visualisierung

```python
def visualize_synthetic_frames(
    synthetic_frames: np.ndarray,
    synthetic_weights: np.ndarray,
    output_dir: str
):
    """
    Visualisiert synthetische Frames.
    """
    import matplotlib.pyplot as plt
    
    K = len(synthetic_frames)
    
    # Grid layout
    cols = int(np.ceil(np.sqrt(K)))
    rows = int(np.ceil(K / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten() if K > 1 else [axes]
    
    for k in range(K):
        # Normalize for display
        frame = synthetic_frames[k]
        p1, p99 = np.percentile(frame, [1, 99])
        frame_norm = np.clip((frame - p1) / (p99 - p1), 0, 1)
        
        # Plot
        axes[k].imshow(frame_norm, cmap='gray')
        axes[k].set_title(f'Cluster {k}\nWeight: {synthetic_weights[k]:.1f}')
        axes[k].axis('off')
    
    # Hide unused subplots
    for k in range(K, len(axes)):
        axes[k].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/synthetic_frames.png', dpi=150)
```

## Modi

### Normal Mode (N ≥ 200)

```python
if N >= 200 and cluster_labels is not None:
    # Create synthetic frames
    synthetic_frames, synthetic_weights = create_synthetic_frames(
        frames, cluster_labels, global_weights, K
    )
    
    print(f"Created {K} synthetic frames from {N} original frames")
    print(f"Frame reduction: {N/K:.1f}×")
```

### Reduced Mode (50 ≤ N < 200)

```python
if 50 <= N < 200:
    # No clustering, no synthetic frames
    # Use original frames directly in Phase 9
    synthetic_frames = frames
    synthetic_weights = global_weights
    
    print(f"Reduced Mode: Using {N} original frames (no synthetic)")
```

## Output-Datenstruktur

```python
# Phase 8 Output
{
    'synthetic_frames': {
        'R': np.ndarray,  # [K × H × W]
        'G': np.ndarray,  # [K × H × W]
        'B': np.ndarray,  # [K × H × W]
    },
    'synthetic_weights': {
        'R': np.ndarray,  # [K]
        'G': np.ndarray,  # [K]
        'B': np.ndarray,  # [K]
    },
    'num_synthetic': int,  # K
}
```

## Konfiguration

```yaml
synthetic:
  weighting: global  # or 'tile_weighted'
  
  # Nur im Normal Mode (N >= 200)
  enabled: true
```

## Nächste Phase

→ **Phase 9: Finales Stacking** (stackt K synthetische Frames mit Sigma-Clipping)
