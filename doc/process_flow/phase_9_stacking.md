# Phase 9: Finales Stacking

## Übersicht

Phase 9 ist **identisch zu v3**. Die K synthetischen Frames (oder N Original-Frames im Reduced Mode) werden mittels **Sigma-Clipping Rejection Stacking** zu einem finalen Bild kombiniert.

## Ziele

1. Ausreißer-Entfernung (Satelliten, Flugzeuge, kosmische Strahlung)
2. Finales gewichtetes Stacking
3. FITS-Speicherung mit Metadaten
4. Qualitätskontrolle

## Sigma-Clipping Rejection Stacking

### Algorithmus

```python
def sigma_clipping_stack(
    frames: np.ndarray,           # [K × H × W]
    weights: np.ndarray,          # [K]
    sigma_low: float = 3.0,
    sigma_high: float = 3.0,
    max_iters: int = 5,
    min_fraction: float = 0.5
) -> np.ndarray:
    """
    Sigma-Clipping Rejection Stacking.
    
    Algorithmus:
    1. Berechne gewichteten Median und MAD
    2. Verwerfe Pixel außerhalb [μ - σ_low·σ, μ + σ_high·σ]
    3. Wiederhole bis Konvergenz oder max_iters
    4. Finales gewichtetes Mittel der verbleibenden Pixel
    
    Args:
        frames: Frames zu stacken
        weights: Frame-Gewichte
        sigma_low: Untere Sigma-Schwelle
        sigma_high: Obere Sigma-Schwelle
        max_iters: Max Iterationen
        min_fraction: Min Fraktion verbleibender Frames pro Pixel
    
    Returns:
        Gestacktes Bild [H × W]
    """
    K, H, W = frames.shape
    
    # Initialize mask (all frames valid)
    mask = np.ones((K, H, W), dtype=bool)
    
    # Iterative sigma-clipping
    for iteration in range(max_iters):
        # Compute weighted statistics per pixel
        weighted_median = np.zeros((H, W))
        weighted_mad = np.zeros((H, W))
        
        for y in range(H):
            for x in range(W):
                # Get valid frames for this pixel
                valid_frames = frames[mask[:, y, x], y, x]
                valid_weights = weights[mask[:, y, x]]
                
                if len(valid_frames) == 0:
                    continue
                
                # Weighted median
                sorted_indices = np.argsort(valid_frames)
                sorted_frames = valid_frames[sorted_indices]
                sorted_weights = valid_weights[sorted_indices]
                cumsum = np.cumsum(sorted_weights)
                median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
                weighted_median[y, x] = sorted_frames[median_idx]
                
                # MAD (Median Absolute Deviation)
                deviations = np.abs(valid_frames - weighted_median[y, x])
                mad = np.median(deviations)
                weighted_mad[y, x] = mad
        
        # Convert MAD to sigma
        sigma = 1.4826 * weighted_mad
        
        # Update mask (reject outliers)
        new_mask = mask.copy()
        for k in range(K):
            lower = weighted_median - sigma_low * sigma
            upper = weighted_median + sigma_high * sigma
            new_mask[k] = (frames[k] >= lower) & (frames[k] <= upper)
        
        # Check convergence
        if np.array_equal(mask, new_mask):
            print(f"Sigma-clipping converged after {iteration+1} iterations")
            break
        
        mask = new_mask
    
    # Final weighted mean
    stacked = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)
    
    for k in range(K):
        stacked += weights[k] * frames[k] * mask[k]
        weight_sum += weights[k] * mask[k]
    
    # Avoid division by zero
    valid_pixels = weight_sum > 0
    stacked[valid_pixels] /= weight_sum[valid_pixels]
    
    # Check minimum fraction
    fraction_used = np.sum(mask, axis=0) / K
    if np.any(fraction_used < min_fraction):
        low_fraction_pixels = np.sum(fraction_used < min_fraction)
        print(f"⚠ {low_fraction_pixels} pixels with < {min_fraction*100}% frames used")
    
    return stacked
```

### Optimierte Implementierung (Vektorisiert)

```python
def sigma_clipping_stack_vectorized(
    frames: np.ndarray,
    weights: np.ndarray,
    sigma_low: float = 3.0,
    sigma_high: float = 3.0,
    max_iters: int = 5,
    min_fraction: float = 0.5
) -> np.ndarray:
    """
    Vektorisierte Sigma-Clipping-Implementierung (schneller).
    """
    K, H, W = frames.shape
    
    # Initialize mask
    mask = np.ones((K, H, W), dtype=bool)
    
    # Normalize weights
    weights_norm = weights / np.sum(weights)
    weights_3d = weights_norm[:, np.newaxis, np.newaxis]
    
    for iteration in range(max_iters):
        # Masked frames
        frames_masked = np.where(mask, frames, np.nan)
        
        # Weighted median (approximation: use weighted mean)
        # True weighted median is expensive; weighted mean is good approximation
        weighted_sum = np.nansum(frames_masked * weights_3d, axis=0)
        weight_sum = np.nansum(mask * weights_3d, axis=0)
        weighted_mean = weighted_sum / (weight_sum + 1e-10)
        
        # MAD
        deviations = np.abs(frames - weighted_mean[np.newaxis, :, :])
        mad = np.nanmedian(np.where(mask, deviations, np.nan), axis=0)
        sigma = 1.4826 * mad
        
        # Update mask
        lower = weighted_mean - sigma_low * sigma
        upper = weighted_mean + sigma_high * sigma
        new_mask = (frames >= lower[np.newaxis, :, :]) & (frames <= upper[np.newaxis, :, :])
        
        # Convergence check
        if np.array_equal(mask, new_mask):
            print(f"Converged after {iteration+1} iterations")
            break
        
        mask = new_mask
    
    # Final weighted mean
    frames_masked = np.where(mask, frames, 0)
    stacked = np.sum(frames_masked * weights_3d, axis=0)
    weight_sum = np.sum(mask * weights_3d, axis=0)
    stacked = stacked / (weight_sum + 1e-10)
    
    return stacked
```

## FITS-Speicherung

### Metadaten

```python
def save_stacked_fits(
    stacked: np.ndarray,
    output_path: str,
    metadata: Dict
):
    """
    Speichert gestacktes Bild als FITS mit Metadaten.
    """
    from astropy.io import fits
    
    # Create FITS HDU
    hdu = fits.PrimaryHDU(stacked.astype(np.float32))
    
    # Add metadata to header
    hdu.header['NFRAMES'] = (metadata['num_frames'], 'Number of input frames')
    hdu.header['NSYNTHETIC'] = (metadata['num_synthetic'], 'Number of synthetic frames')
    hdu.header['METHOD'] = ('TLR_v4', 'Tile-Local Registration v4')
    hdu.header['SIGMALOW'] = (metadata['sigma_low'], 'Sigma-clipping lower threshold')
    hdu.header['SIGMAHIGH'] = (metadata['sigma_high'], 'Sigma-clipping upper threshold')
    hdu.header['EXPTIME'] = (metadata['total_exposure'], 'Total exposure time (s)')
    hdu.header['CHANNEL'] = (metadata['channel'], 'Color channel (R/G/B)')
    
    # Add processing timestamp
    from datetime import datetime
    hdu.header['DATE'] = (datetime.utcnow().isoformat(), 'Processing date (UTC)')
    
    # Write FITS
    hdu.writeto(output_path, overwrite=True)
    print(f"Saved: {output_path}")
```

### Pro Kanal

```python
def save_all_channels(
    stacked_r: np.ndarray,
    stacked_g: np.ndarray,
    stacked_b: np.ndarray,
    output_dir: str,
    metadata: Dict
):
    """
    Speichert alle Kanäle als separate FITS.
    """
    channels = {
        'R': stacked_r,
        'G': stacked_g,
        'B': stacked_b,
    }
    
    for channel, data in channels.items():
        output_path = f"{output_dir}/stacked_{channel}.fits"
        metadata_channel = metadata.copy()
        metadata_channel['channel'] = channel
        save_stacked_fits(data, output_path, metadata_channel)
```

## Qualitätskontrolle

### Statistiken

```python
def compute_stack_statistics(stacked: np.ndarray) -> Dict:
    """
    Berechnet Statistiken des gestackten Bildes.
    """
    # Basic statistics
    mean = np.mean(stacked)
    median = np.median(stacked)
    std = np.std(stacked)
    
    # Percentiles
    p1, p99 = np.percentile(stacked, [1, 99])
    
    # Dynamic range
    min_val = np.min(stacked)
    max_val = np.max(stacked)
    
    # SNR estimate (rough)
    # SNR ≈ median / MAD
    mad = np.median(np.abs(stacked - median))
    snr_estimate = median / (1.4826 * mad) if mad > 0 else 0
    
    return {
        'mean': mean,
        'median': median,
        'std': std,
        'min': min_val,
        'max': max_val,
        'p1': p1,
        'p99': p99,
        'snr_estimate': snr_estimate,
    }
```

### Validierung

```python
def validate_stacked_image(stacked: np.ndarray, metadata: Dict):
    """
    Validiert gestacktes Bild.
    """
    checks = []
    
    # Check 1: No NaN/Inf
    assert not np.any(np.isnan(stacked)), "NaN in stacked image"
    assert not np.any(np.isinf(stacked)), "Inf in stacked image"
    checks.append("✓ No NaN/Inf")
    
    # Check 2: Reasonable dynamic range
    min_val = np.min(stacked)
    max_val = np.max(stacked)
    assert min_val >= 0, f"Negative values: {min_val}"
    assert max_val < 65536, f"Overflow: {max_val}"
    checks.append(f"✓ Dynamic range: [{min_val:.1f}, {max_val:.1f}]")
    
    # Check 3: Non-zero pixels
    non_zero = np.sum(stacked > 0)
    total = stacked.size
    fraction = non_zero / total
    assert fraction > 0.9, f"Too many zero pixels: {fraction*100:.1f}%"
    checks.append(f"✓ Non-zero pixels: {fraction*100:.1f}%")
    
    # Check 4: Statistics
    stats = compute_stack_statistics(stacked)
    checks.append(f"✓ Median: {stats['median']:.1f}, SNR: {stats['snr_estimate']:.1f}")
    
    return checks
```

## Modi

### Normal Mode (N ≥ 200)

```python
# Stack K synthetic frames
stacked = sigma_clipping_stack(
    synthetic_frames,
    synthetic_weights,
    sigma_low=3.0,
    sigma_high=3.0,
    max_iters=5
)
```

### Reduced Mode (50 ≤ N < 200)

```python
# Stack N original frames (no synthetic)
stacked = sigma_clipping_stack(
    original_frames,
    global_weights,
    sigma_low=3.0,
    sigma_high=3.0,
    max_iters=5
)
```

## Output-Datenstruktur

```python
# Phase 9 Output
{
    'stacked': {
        'R': np.ndarray,  # [H × W]
        'G': np.ndarray,  # [H × W]
        'B': np.ndarray,  # [H × W]
    },
    'statistics': {
        'R': Dict,
        'G': Dict,
        'B': Dict,
    },
    'files': {
        'R': 'stacked_R.fits',
        'G': 'stacked_G.fits',
        'B': 'stacked_B.fits',
    }
}
```

## Konfiguration

```yaml
stacking:
  method: sigma_clipping
  sigma_low: 3.0
  sigma_high: 3.0
  max_iterations: 5
  min_fraction: 0.5
  
  # Output format
  output_format: fits
  output_bitdepth: 32  # float32
```

## Finalisierung

Nach Phase 9 ist die v4-Pipeline abgeschlossen. Die finalen Outputs sind:
- `stacked_R.fits`
- `stacked_G.fits`
- `stacked_B.fits`

Optional kann ein RGB-Composite erstellt werden (außerhalb der Methodik).

## Nächste Schritte (außerhalb v4)

- **RGB-Kombination**: Combine R, G, B zu RGB-Bild
- **Stretching**: Nichtlineare Tonwertkurve
- **Color Calibration**: Weißabgleich, Farbbalance
- **Post-Processing**: Deconvolution, Denoise, etc.
