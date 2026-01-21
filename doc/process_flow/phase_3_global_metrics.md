# Phase 3: Globale Frame-Metriken

## Übersicht

Phase 3 berechnet globale Qualitätsmetriken für jeden Frame und Kanal. Diese Metriken werden verwendet für:
1. **Normalisierung** (Background-Werte für Phase 6)
2. **Globale Gewichtung** G_f,c (kombiniert mit lokalen Gewichten in Phase 6)

Im Gegensatz zu v3 erfolgt hier **keine Normalisierung der Frames** - nur die Berechnung der Metriken. Die eigentliche Normalisierung geschieht on-the-fly in Phase 6.

## Metriken

### 1. Hintergrundniveau B_f,c

**Ziel:** Robuste Schätzung des Himmelshintergrunds

```python
def estimate_background(frame: np.ndarray) -> float:
    """
    Schätzt Hintergrundniveau mittels Sigma-Clipping.
    
    Algorithmus:
    1. Initiale Schätzung: median(frame)
    2. Iteratives Sigma-Clipping:
       - Berechne μ, σ
       - Verwerfe Pixel außerhalb [μ - 3σ, μ + 3σ]
       - Wiederhole bis Konvergenz
    3. Finale Schätzung: median(clipped_pixels)
    
    Returns:
        Background-Level B_f,c
    """
    pixels = frame.flatten()
    
    # Iterative sigma-clipping
    for iteration in range(5):
        mu = np.median(pixels)
        sigma = np.std(pixels)
        
        # Clip outliers
        mask = np.abs(pixels - mu) < 3 * sigma
        pixels_clipped = pixels[mask]
        
        # Check convergence
        if len(pixels_clipped) == len(pixels):
            break
        
        pixels = pixels_clipped
    
    # Final estimate
    background = np.median(pixels)
    
    return background
```

**Interpretation:**
- **Niedriger Background** = weniger Lichtverschmutzung = besser
- Typische Werte: 50-500 ADU (abhängig von Kamera/Belichtung)

### 2. Rauschen σ_f,c

**Ziel:** Schätzung des Rauschlevels

```python
def estimate_noise(frame: np.ndarray, background: float) -> float:
    """
    Schätzt Rauschen mittels MAD (Median Absolute Deviation).
    
    MAD ist robuster als Standardabweichung gegenüber Ausreißern (Sterne).
    
    σ = 1.4826 · median(|frame - median(frame)|)
    
    Returns:
        Noise level σ_f,c
    """
    # Subtract background
    frame_bg_subtracted = frame - background
    
    # Compute MAD
    median_val = np.median(frame_bg_subtracted)
    mad = np.median(np.abs(frame_bg_subtracted - median_val))
    
    # Convert MAD to standard deviation
    sigma = 1.4826 * mad
    
    return sigma
```

**Interpretation:**
- **Niedriges Rauschen** = bessere Bildqualität = besser
- Typische Werte: 5-20 ADU

### 3. Gradientenergie E_f,c

**Ziel:** Maß für Schärfe/Details im Bild

```python
def estimate_gradient_energy(frame: np.ndarray) -> float:
    """
    Berechnet Gradientenergie mittels Sobel-Operator.
    
    Hohe Gradientenergie = viele scharfe Kanten = gutes Seeing
    
    Returns:
        Gradient energy E_f,c
    """
    # Sobel gradients
    grad_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
    
    # Gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Energy (mean of squared gradients)
    energy = np.mean(grad_mag**2)
    
    return energy
```

**Interpretation:**
- **Hohe Energie** = scharfe Sterne = gutes Seeing = besser
- Typische Werte: 100-10000 (stark abhängig von Objekt)

## Globaler Qualitätsindex

### Berechnung

```python
def compute_quality_index(
    background: float,
    noise: float,
    gradient_energy: float,
    alpha: float = 0.3,
    beta: float = 0.3,
    gamma: float = 0.4
) -> float:
    """
    Berechnet globalen Qualitätsindex.
    
    Q_f,c = α·(-B̃_f,c) + β·(-σ̃_f,c) + γ·Ẽ_f,c
    
    wobei:
      B̃, σ̃, Ẽ = normalisierte Metriken (Median + MAD)
      α + β + γ = 1
    
    Args:
        background: Hintergrundniveau
        noise: Rauschen
        gradient_energy: Gradientenergie
        alpha, beta, gamma: Gewichtungsfaktoren
    
    Returns:
        Quality index Q_f,c
    """
    # Normalization (done globally over all frames)
    # B̃ = (B - median(B)) / MAD(B)
    # (assumes this is called after collecting all metrics)
    
    # Quality index (negative for background/noise, positive for energy)
    Q = alpha * (-background) + beta * (-noise) + gamma * gradient_energy
    
    return Q
```

### Normalisierung der Metriken

```python
def normalize_metrics(
    backgrounds: np.ndarray,
    noises: np.ndarray,
    energies: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalisiert Metriken robust mittels Median + MAD.
    
    X̃ = (X - median(X)) / (1.4826 · MAD(X))
    
    Args:
        backgrounds: Array von Background-Werten [N_frames × N_channels]
        noises: Array von Noise-Werten
        energies: Array von Energy-Werten
    
    Returns:
        (backgrounds_norm, noises_norm, energies_norm)
    """
    def robust_normalize(values: np.ndarray) -> np.ndarray:
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        sigma_robust = 1.4826 * mad
        
        if sigma_robust > 0:
            normalized = (values - median) / sigma_robust
        else:
            normalized = np.zeros_like(values)
        
        return normalized
    
    backgrounds_norm = robust_normalize(backgrounds)
    noises_norm = robust_normalize(noises)
    energies_norm = robust_normalize(energies)
    
    return backgrounds_norm, noises_norm, energies_norm
```

### Exponential Mapping

```python
def compute_global_weights(
    quality_indices: np.ndarray,
    clamp_range: Tuple[float, float] = (-3.0, 3.0)
) -> np.ndarray:
    """
    Konvertiert Qualitätsindizes zu Gewichten mittels Exponentialfunktion.
    
    G_f,c = exp(Q_f,c)
    
    mit Clamping vor Exponential um numerische Stabilität zu garantieren.
    
    Args:
        quality_indices: Array von Q_f,c
        clamp_range: Clamping-Bereich (default: [-3, 3])
    
    Returns:
        Global weights G_f,c
    """
    # Clamp to avoid overflow/underflow
    Q_clamped = np.clip(quality_indices, clamp_range[0], clamp_range[1])
    
    # Exponential mapping
    weights = np.exp(Q_clamped)
    
    # Normalize (optional, depends on use case)
    # weights /= np.sum(weights)
    
    return weights
```

## Streaming-Implementierung

Da v4 Frames nicht komplett lädt, erfolgt die Metrik-Berechnung **streaming**:

```python
def compute_global_metrics_streaming(
    frame_paths: List[str],
    bayer_pattern: str,
    channels: List[str] = ['R', 'G', 'B']
) -> Dict[str, np.ndarray]:
    """
    Berechnet globale Metriken im Streaming-Modus.
    
    Args:
        frame_paths: Liste der Frame-Pfade
        bayer_pattern: Bayer-Pattern
        channels: Zu verarbeitende Kanäle
    
    Returns:
        Dict mit Metriken:
          'backgrounds': [N_frames × N_channels]
          'noises': [N_frames × N_channels]
          'energies': [N_frames × N_channels]
          'quality_indices': [N_frames × N_channels]
          'global_weights': [N_frames × N_channels]
    """
    N = len(frame_paths)
    C = len(channels)
    
    # Initialize arrays
    backgrounds = np.zeros((N, C))
    noises = np.zeros((N, C))
    energies = np.zeros((N, C))
    
    # Process each frame
    for f, frame_path in enumerate(frame_paths):
        # Load CFA frame
        cfa_frame = load_fits(frame_path)
        
        # Debayer
        rgb_frame = debayer_vng(cfa_frame, bayer_pattern)
        
        # Process each channel
        for c, channel in enumerate(channels):
            channel_idx = {'R': 0, 'G': 1, 'B': 2}[channel]
            frame_channel = rgb_frame[:, :, channel_idx]
            
            # Compute metrics
            backgrounds[f, c] = estimate_background(frame_channel)
            noises[f, c] = estimate_noise(frame_channel, backgrounds[f, c])
            energies[f, c] = estimate_gradient_energy(frame_channel)
        
        # Progress
        if (f + 1) % 10 == 0:
            print(f"Processed {f+1}/{N} frames...")
    
    # Normalize metrics
    backgrounds_norm, noises_norm, energies_norm = normalize_metrics(
        backgrounds, noises, energies
    )
    
    # Compute quality indices
    alpha, beta, gamma = 0.3, 0.3, 0.4
    quality_indices = (
        alpha * (-backgrounds_norm) +
        beta * (-noises_norm) +
        gamma * energies_norm
    )
    
    # Compute global weights
    global_weights = compute_global_weights(quality_indices)
    
    return {
        'backgrounds': backgrounds,
        'noises': noises,
        'energies': energies,
        'backgrounds_norm': backgrounds_norm,
        'noises_norm': noises_norm,
        'energies_norm': energies_norm,
        'quality_indices': quality_indices,
        'global_weights': global_weights,
    }
```

## Validierung

```python
def validate_global_metrics(metrics: Dict):
    """
    Validiert globale Metriken.
    """
    checks = []
    
    N, C = metrics['backgrounds'].shape
    
    # Check 1: All backgrounds positive
    assert np.all(metrics['backgrounds'] > 0)
    checks.append("✓ All backgrounds > 0")
    
    # Check 2: All noises positive
    assert np.all(metrics['noises'] > 0)
    checks.append("✓ All noises > 0")
    
    # Check 3: All energies positive
    assert np.all(metrics['energies'] > 0)
    checks.append("✓ All energies > 0")
    
    # Check 4: Normalized metrics have mean ≈ 0, std ≈ 1
    for key in ['backgrounds_norm', 'noises_norm', 'energies_norm']:
        mean = np.mean(metrics[key])
        std = np.std(metrics[key])
        assert abs(mean) < 0.1, f"{key} mean {mean} not ≈ 0"
        assert 0.8 < std < 1.2, f"{key} std {std} not ≈ 1"
    checks.append("✓ Normalized metrics: mean ≈ 0, std ≈ 1")
    
    # Check 5: Global weights positive
    assert np.all(metrics['global_weights'] > 0)
    checks.append("✓ All global weights > 0")
    
    # Check 6: No NaN/Inf
    for key in metrics:
        assert not np.any(np.isnan(metrics[key]))
        assert not np.any(np.isinf(metrics[key]))
    checks.append("✓ No NaN/Inf in metrics")
    
    return checks
```

## Output-Datenstruktur

```python
# Phase 3 Output
{
    'backgrounds': np.ndarray,      # [N_frames × N_channels]
    'noises': np.ndarray,           # [N_frames × N_channels]
    'energies': np.ndarray,         # [N_frames × N_channels]
    'backgrounds_norm': np.ndarray, # normalized
    'noises_norm': np.ndarray,      # normalized
    'energies_norm': np.ndarray,    # normalized
    'quality_indices': np.ndarray,  # Q_f,c
    'global_weights': np.ndarray,   # G_f,c = exp(Q_f,c)
}
```

## Konfiguration

```yaml
metrics:
  # Gewichtungsfaktoren (müssen zu 1 summieren)
  alpha: 0.3  # Background-Gewicht
  beta: 0.3   # Noise-Gewicht
  gamma: 0.4  # Gradient-Gewicht
  
  # Clamping vor Exponential
  clamp_min: -3.0
  clamp_max: 3.0
  
normalization:
  enabled: true
  mode: background
  per_channel: true
  target_median: 209.0
```

## Nächste Phase

→ **Phase 4: Adaptive Tile-Grid-Erzeugung**
