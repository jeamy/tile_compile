"""
Tile-wise Local Registration (TLR) v4 - Methodik v4 compliant.

Implements iterative local registration with temporal warp smoothing.

Key principles (Methodik v4):
- No global reference frame
- Iterative reference refinement (2-3 iterations)
- Temporal warp smoothing (Savitzky-Golay)
- Registration quality weighting R_{f,t} based on ECC correlation
- Translation-only model
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from astropy.io import fits
from scipy.signal import savgol_filter

try:
    import cv2
except Exception:
    cv2 = None

from .opencv_registration import (
    opencv_prepare_ecc_image,
    opencv_phasecorr_translation,
    opencv_ecc_warp,
)


def _read_fits_tile(
    path: Path,
    tile_bounds: Tuple[int, int, int, int]
) -> Tuple[Optional[np.ndarray], Optional[fits.Header]]:
    """Read a tile region from a FITS file."""
    try:
        with fits.open(str(path)) as hdul:
            data = hdul[0].data
            hdr = hdul[0].header
            
            if data is None:
                return None, None
            
            y0, y1, x0, x1 = tile_bounds
            tile = data[y0:y1, x0:x1].astype(np.float32, copy=True)
            return tile, hdr
    except Exception:
        return None, None


def register_tile(
    moving_tile: np.ndarray,
    ref_tile: np.ndarray,
    ecc_cc_min: float = 0.2
) -> Tuple[Optional[np.ndarray], float]:
    """Register a tile against reference (Methodik v4 §5.1).
    
    Translation-only model: p' = p + (dx, dy)
    Uses Phase Correlation for initial estimate, then ECC refinement.
    
    Args:
        moving_tile: Tile to register
        ref_tile: Reference tile
        ecc_cc_min: Minimum ECC correlation threshold
        
    Returns:
        (warp_matrix, correlation) or (None, 0.0) if failed
        warp_matrix is 2x3 translation-only: [[1, 0, dx], [0, 1, dy]]
    """
    if cv2 is None:
        return None, 0.0
    
    # Preprocess both tiles (ECC preprocessing)
    mov_prep = opencv_prepare_ecc_image(moving_tile)
    ref_prep = opencv_prepare_ecc_image(ref_tile)
    
    # Phase correlation for initial translation
    dx, dy = opencv_phasecorr_translation(mov_prep, ref_prep)
    
    # Initialize translation-only warp
    init_warp = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    
    # ECC refinement (TRANSLATION mode only)
    try:
        warp, cc = opencv_ecc_warp(
            mov_prep, ref_prep,
            allow_rotation=False,  # Translation only
            init_warp=init_warp
        )
        
        # Validate correlation
        if cc < ecc_cc_min:
            return None, 0.0
        
        return warp, cc
        
    except Exception:
        # ECC failed, use phase correlation result
        if abs(dx) < 50 and abs(dy) < 50:  # Sanity check
            return init_warp, 0.5  # Assign moderate confidence
        return None, 0.0


def smooth_warps_temporal(
    warps: List[Optional[np.ndarray]],
    window_length: int = 11,
    polyorder: int = 3
) -> List[Optional[np.ndarray]]:
    """Apply temporal smoothing to warp sequences (Methodik v4 §5.3).
    
    Uses Savitzky-Golay filter for smooth, derivative-preserving interpolation.
    
    Args:
        warps: List of 2x3 warp matrices (or None for invalid)
        window_length: Filter window (must be odd, >= polyorder+2)
        polyorder: Polynomial order
        
    Returns:
        Smoothed warp sequence
    """
    n_frames = len(warps)
    
    # Extract translation components
    tx = np.zeros(n_frames, dtype=np.float32)
    ty = np.zeros(n_frames, dtype=np.float32)
    valid = np.zeros(n_frames, dtype=bool)
    
    for i, warp in enumerate(warps):
        if warp is not None:
            tx[i] = warp[0, 2]
            ty[i] = warp[1, 2]
            valid[i] = True
    
    # If too few valid warps, return original
    if np.sum(valid) < window_length:
        return warps
    
    # Interpolate invalid values
    if not np.all(valid):
        valid_idx = np.where(valid)[0]
        tx_interp = np.interp(np.arange(n_frames), valid_idx, tx[valid_idx])
        ty_interp = np.interp(np.arange(n_frames), valid_idx, ty[valid_idx])
    else:
        tx_interp = tx
        ty_interp = ty
    
    # Apply Savitzky-Golay filter
    try:
        tx_smooth = savgol_filter(tx_interp, window_length, polyorder, mode='nearest')
        ty_smooth = savgol_filter(ty_interp, window_length, polyorder, mode='nearest')
    except Exception:
        # Fallback: simple moving average
        kernel = np.ones(window_length) / window_length
        tx_smooth = np.convolve(tx_interp, kernel, mode='same')
        ty_smooth = np.convolve(ty_interp, kernel, mode='same')
    
    # Reconstruct warp matrices
    smoothed_warps = []
    for i in range(n_frames):
        if warps[i] is not None:
            warp_smooth = np.array([
                [1.0, 0.0, tx_smooth[i]],
                [0.0, 1.0, ty_smooth[i]]
            ], dtype=np.float32)
            smoothed_warps.append(warp_smooth)
        else:
            smoothed_warps.append(None)
    
    return smoothed_warps


def registration_quality_weight(cc: float, beta: float = 5.0) -> float:
    """Compute registration quality weight R_{f,t} (Methodik v4 §7).
    
    R_{f,t} = exp(β · (cc_{f,t} − 1))
    
    Args:
        cc: ECC correlation coefficient [0, 1]
        beta: Sensitivity parameter (higher = more sensitive to poor registration)
        
    Returns:
        Quality weight in [0, 1]
    """
    return float(np.exp(beta * (cc - 1.0)))


def compute_warp_variance(warps: List[Optional[np.ndarray]]) -> float:
    """Compute variance of warp translations (Methodik v4 §9, §10).
    
    Used for:
    - Variance-weighted overlap-add window
    - Extended state vector for clustering
    
    Args:
        warps: List of 2x3 warp matrices
        
    Returns:
        Combined variance of (dx, dy) translations
    """
    tx_list = []
    ty_list = []
    for warp in warps:
        if warp is not None:
            tx_list.append(warp[0, 2])
            ty_list.append(warp[1, 2])
    
    if len(tx_list) < 2:
        return 0.0
    
    var_x = float(np.var(tx_list))
    var_y = float(np.var(ty_list))
    return var_x + var_y


def variance_window_weight(warp_variance: float, sigma: float = 2.0) -> float:
    """Compute variance-based window weight ψ(var(Â)) (Methodik v4 §9).
    
    ψ(v) = exp(-v / (2·σ²))
    
    Low variance → high confidence → full window weight
    High variance → low confidence → reduced window weight
    
    Args:
        warp_variance: Combined (dx, dy) variance
        sigma: Scale parameter
        
    Returns:
        Weight factor in (0, 1]
    """
    return float(np.exp(-warp_variance / (2.0 * sigma * sigma)))


def tile_local_register_and_reconstruct_iterative(
    frames: List[Path],
    tile_bounds: Tuple[int, int, int, int],
    tile_idx: int,
    weights_global: np.ndarray,
    weights_local: np.ndarray,
    config: dict,
    max_iterations: int = 3
) -> Tuple[Optional[np.ndarray], Dict]:
    """Perform iterative tile-local registration and reconstruction (Methodik v4 §5.2).
    
    Iterative refinement:
    1. Initial reference from median frame
    2. Register all frames → reconstruct
    3. Use reconstruction as new reference
    4. Repeat until convergence (2-3 iterations)
    
    Args:
        frames: List of frame paths
        tile_bounds: (y_start, y_end, x_start, x_end)
        tile_idx: Tile index for weight lookup
        weights_global: Global weights G_f per frame
        weights_local: Local weights L_{f,t} per frame
        config: Configuration dict
        max_iterations: Maximum refinement iterations
        
    Returns:
        (reconstructed_tile, metadata) or (None, {})
    """
    if cv2 is None:
        return None, {}
    
    # Get config
    tlr_cfg = config.get("registration", {}).get("local_tiles", {})
    ecc_cc_min = float(tlr_cfg.get("ecc_cc_min", 0.2))
    min_valid_frames = int(tlr_cfg.get("min_valid_frames", 10))
    beta = float(tlr_cfg.get("registration_quality_beta", 5.0))
    
    n_frames = len(frames)
    y0, y1, x0, x1 = tile_bounds
    th, tw = y1 - y0, x1 - x0
    
    # Initial reference: median frame
    ref_idx = n_frames // 2
    ref_tile, _ = _read_fits_tile(frames[ref_idx], tile_bounds)
    
    if ref_tile is None:
        return None, {}
    
    # Iterative refinement
    for iteration in range(max_iterations):
        # Register all frames against current reference
        warps = []
        correlations = []
        
        for f_idx in range(n_frames):
            if f_idx == ref_idx and iteration == 0:
                # Reference frame: identity
                warps.append(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32))
                correlations.append(1.0)
                continue
            
            mov_tile, _ = _read_fits_tile(frames[f_idx], tile_bounds)
            if mov_tile is None:
                warps.append(None)
                correlations.append(0.0)
                continue
            
            warp, cc = estimate_tile_local_translation(mov_tile, ref_tile, ecc_cc_min)
            warps.append(warp)
            correlations.append(cc)
        
        # Temporal smoothing of warps (Methodik v4 §5.3)
        warps_smooth = smooth_warps_temporal(warps, window_length=11, polyorder=3)
        
        # Reconstruct tile with smoothed warps
        tile_sum = np.zeros((th, tw), dtype=np.float64)
        weight_sum = np.zeros((th, tw), dtype=np.float64)
        valid_frames = 0
        
        for f_idx in range(n_frames):
            warp = warps_smooth[f_idx]
            if warp is None:
                continue
            
            # Compute effective weight: W_{f,t} = G_f · L_{f,t} · R_{f,t}
            G_f = float(weights_global[f_idx]) if f_idx < len(weights_global) else 1.0
            L_ft = float(weights_local[f_idx]) if f_idx < len(weights_local) else 1.0
            R_ft = registration_quality_weight(correlations[f_idx], beta)
            W_ft = G_f * L_ft * R_ft
            
            if W_ft < 1e-6:
                continue
            
            # Load and warp tile
            mov_tile, _ = _read_fits_tile(frames[f_idx], tile_bounds)
            if mov_tile is None:
                continue
            
            try:
                warped = cv2.warpAffine(mov_tile, warp, (tw, th), flags=cv2.INTER_LINEAR)
                tile_sum += W_ft * warped.astype(np.float64)
                weight_sum += W_ft
                valid_frames += 1
            except Exception:
                continue
        
        # Check validity
        if valid_frames < min_valid_frames:
            return None, {"error": "insufficient_valid_frames", "valid": valid_frames}
        
        # Normalize
        mask = weight_sum > 1e-6
        tile_reconstructed = np.zeros((th, tw), dtype=np.float32)
        tile_reconstructed[mask] = (tile_sum[mask] / weight_sum[mask]).astype(np.float32)
        
        # Update reference for next iteration
        if iteration < max_iterations - 1:
            ref_tile = tile_reconstructed.copy()
    
    # Compute extended metadata for clustering (Methodik v4 §10)
    valid_correlations = [c for c in correlations if c > 0]
    warp_var = compute_warp_variance(warps_smooth)
    invalid_fraction = 1.0 - (valid_frames / n_frames) if n_frames > 0 else 1.0
    
    metadata = {
        "iterations": max_iterations,
        "valid_frames": valid_frames,
        "mean_correlation": float(np.mean(valid_correlations)) if valid_correlations else 0.0,
        "warp_variance": warp_var,
        "invalid_tile_fraction": invalid_fraction,
        "correlations": correlations,
        "warps": warps_smooth,
    }
    
    return tile_reconstructed, metadata


def tile_local_reconstruct_all_channels_v4(
    frames_by_channel: Dict[str, List[Path]],
    tile_grid: Dict,
    weights_global: Dict[str, np.ndarray],
    weights_local: Dict[str, np.ndarray],
    config: dict
) -> Dict[str, np.ndarray]:
    """Reconstruct all channels using iterative tile-local registration (Methodik v4).
    
    Args:
        frames_by_channel: {"R": [paths], "G": [paths], "B": [paths]}
        tile_grid: Tile grid definition with "tiles" list
        weights_global: Global weights per channel
        weights_local: Local weights per channel (2D array: frames x tiles)
        config: Configuration
        
    Returns:
        {"R": array, "G": array, "B": array}
    """
    results = {}
    tiles = tile_grid.get("tiles", [])
    
    # Get image dimensions from first frame
    first_channel = next(iter(frames_by_channel.values()))
    if not first_channel:
        return results
    
    with fits.open(str(first_channel[0])) as hdul:
        h, w = hdul[0].data.shape
    
    for channel in ["R", "G", "B"]:
        if channel not in frames_by_channel or not frames_by_channel[channel]:
            continue
        
        print(f"[TLR v4] Reconstructing channel {channel} with {len(tiles)} tiles")
        
        # Initialize output
        reconstructed = np.zeros((h, w), dtype=np.float32)
        overlap_count = np.zeros((h, w), dtype=np.float32)
        
        frames = frames_by_channel[channel]
        W_global = weights_global.get(channel, np.ones(len(frames), dtype=np.float32))
        W_local = weights_local.get(channel, np.ones((len(frames), len(tiles)), dtype=np.float32))
        
        # Process each tile
        for tile_idx, tile in enumerate(tiles):
            y0, y1 = tile["y_start"], tile["y_end"]
            x0, x1 = tile["x_start"], tile["x_end"]
            tile_bounds = (y0, y1, x0, x1)
            
            # Get local weights for this tile
            L_tile = W_local[:, tile_idx] if tile_idx < W_local.shape[1] else np.ones(len(frames), dtype=np.float32)
            
            # Iterative reconstruction
            tile_recon, metadata = tile_local_register_and_reconstruct_iterative(
                frames,
                tile_bounds,
                tile_idx,
                W_global,
                L_tile,
                config,
                max_iterations=3
            )
            
            if tile_recon is None:
                print(f"[WARN] Tile {tile_idx} failed: {metadata.get('error', 'unknown')}")
                continue
            
            # Overlap-add with variance-weighted Hanning window (Methodik v4 §9)
            th, tw = tile_recon.shape
            hann_1d_y = np.hanning(th).astype(np.float32)
            hann_1d_x = np.hanning(tw).astype(np.float32)
            hann_2d = np.outer(hann_1d_y, hann_1d_x)
            
            # Apply variance-based window weight: w_t(p) = hann(p) · ψ(var(Â))
            warp_var = metadata.get("warp_variance", 0.0)
            psi = variance_window_weight(warp_var, sigma=2.0)
            weighted_hann = hann_2d * psi
            
            # Add to output
            reconstructed[y0:y1, x0:x1] += tile_recon * weighted_hann
            overlap_count[y0:y1, x0:x1] += weighted_hann
        
        # Normalize by overlap count
        mask = overlap_count > 1e-6
        reconstructed[mask] /= overlap_count[mask]
        
        results[channel] = reconstructed
        print(f"[TLR v4] Channel {channel} complete")
    
    return results


def should_refine_tile(metadata: Dict, variance_threshold: float = 4.0) -> bool:
    """Check if a tile should be recursively refined (Methodik v4 §4).
    
    Refinement criteria:
    - High warp variance
    - High PSF inhomogeneity (low mean correlation)
    
    Args:
        metadata: Tile metadata from reconstruction
        variance_threshold: Max acceptable warp variance
        
    Returns:
        True if tile should be split into subtiles
    """
    warp_var = metadata.get("warp_variance", 0.0)
    mean_cc = metadata.get("mean_correlation", 1.0)
    
    # High variance → unstable registration → refine
    if warp_var > variance_threshold:
        return True
    
    # Low correlation → poor PSF match → refine
    if mean_cc < 0.5:
        return True
    
    return False


def split_tile_bounds(
    tile_bounds: Tuple[int, int, int, int],
    min_tile_size: int = 32
) -> List[Tuple[int, int, int, int]]:
    """Split a tile into 4 sub-tiles (Methodik v4 §4 recursive refinement).
    
    Args:
        tile_bounds: (y0, y1, x0, x1)
        min_tile_size: Minimum size below which no further splitting
        
    Returns:
        List of 4 sub-tile bounds, or original if too small
    """
    y0, y1, x0, x1 = tile_bounds
    th, tw = y1 - y0, x1 - x0
    
    # Don't split if already at minimum size
    if th < 2 * min_tile_size or tw < 2 * min_tile_size:
        return [tile_bounds]
    
    # Split into 4 quadrants
    y_mid = y0 + th // 2
    x_mid = x0 + tw // 2
    
    return [
        (y0, y_mid, x0, x_mid),      # top-left
        (y0, y_mid, x_mid, x1),      # top-right
        (y_mid, y1, x0, x_mid),      # bottom-left
        (y_mid, y1, x_mid, x1),      # bottom-right
    ]


def compute_post_warp_metrics(
    warped_tile: np.ndarray,
    ref_tile: np.ndarray
) -> Dict[str, float]:
    """Compute quality metrics after warp application (Methodik v4 §6).
    
    These metrics are computed on the warped (motion-corrected) tile,
    providing more accurate quality assessment than pre-warp metrics.
    
    Args:
        warped_tile: Tile after warp transformation
        ref_tile: Reference tile for comparison
        
    Returns:
        Dict with metric values
    """
    if warped_tile is None or ref_tile is None:
        return {"fwhm": 0.0, "contrast": 0.0, "background": 0.0}
    
    # Local contrast (edge strength / noise)
    try:
        if cv2 is not None:
            laplacian = cv2.Laplacian(warped_tile, cv2.CV_64F)
            edge_strength = float(np.var(laplacian))
        else:
            edge_strength = 0.0
    except Exception:
        edge_strength = 0.0
    
    # Local background (robust median)
    background = float(np.median(warped_tile))
    
    # Signal-to-noise proxy: (peak - background) / MAD
    peak = float(np.percentile(warped_tile, 99))
    mad = float(np.median(np.abs(warped_tile - background)))
    snr_proxy = (peak - background) / (mad + 1e-6)
    
    return {
        "contrast": edge_strength,
        "background": background,
        "snr_proxy": snr_proxy,
    }


def tile_local_reconstruct_with_refinement(
    frames: List[Path],
    tile_bounds: Tuple[int, int, int, int],
    tile_idx: int,
    weights_global: np.ndarray,
    weights_local: np.ndarray,
    config: dict,
    max_depth: int = 2
) -> Tuple[Optional[np.ndarray], Dict]:
    """Tile reconstruction with recursive refinement (Methodik v4 §4).
    
    If a tile has high warp variance or poor correlation,
    it is split into sub-tiles and processed recursively.
    
    Args:
        frames: List of frame paths
        tile_bounds: (y0, y1, x0, x1)
        tile_idx: Tile index
        weights_global: Global weights
        weights_local: Local weights
        config: Configuration
        max_depth: Maximum recursion depth
        
    Returns:
        (reconstructed_tile, metadata)
    """
    # First attempt: standard reconstruction
    tile_recon, metadata = tile_local_register_and_reconstruct_iterative(
        frames, tile_bounds, tile_idx, weights_global, weights_local, config
    )
    
    if tile_recon is None:
        return None, metadata
    
    # Check if refinement is needed
    tlr_cfg = config.get("registration", {}).get("local_tiles", {})
    enable_refinement = tlr_cfg.get("enable_recursive_refinement", True)
    variance_threshold = float(tlr_cfg.get("refinement_variance_threshold", 4.0))
    
    if not enable_refinement or max_depth <= 0:
        return tile_recon, metadata
    
    if not should_refine_tile(metadata, variance_threshold):
        return tile_recon, metadata
    
    # Split tile and process sub-tiles
    y0, y1, x0, x1 = tile_bounds
    th, tw = y1 - y0, x1 - x0
    sub_bounds = split_tile_bounds(tile_bounds, min_tile_size=32)
    
    if len(sub_bounds) == 1:
        # Can't split further
        return tile_recon, metadata
    
    print(f"[TLR v4] Refining tile {tile_idx} (variance={metadata.get('warp_variance', 0):.2f})")
    
    # Reconstruct each sub-tile
    refined = np.zeros((th, tw), dtype=np.float32)
    overlap = np.zeros((th, tw), dtype=np.float32)
    
    for sub_idx, sub_bound in enumerate(sub_bounds):
        sy0, sy1, sx0, sx1 = sub_bound
        sub_h, sub_w = sy1 - sy0, sx1 - sx0
        
        # Recursive call with reduced depth
        sub_recon, sub_meta = tile_local_reconstruct_with_refinement(
            frames, sub_bound, tile_idx * 4 + sub_idx,
            weights_global, weights_local, config, max_depth - 1
        )
        
        if sub_recon is None:
            continue
        
        # Create Hanning window for sub-tile
        hann_y = np.hanning(sub_h).astype(np.float32)
        hann_x = np.hanning(sub_w).astype(np.float32)
        hann_sub = np.outer(hann_y, hann_x)
        
        # Variance-weighted window
        sub_var = sub_meta.get("warp_variance", 0.0)
        psi = variance_window_weight(sub_var, sigma=2.0)
        weighted_hann = hann_sub * psi
        
        # Map sub-tile coords to parent tile coords
        rel_y0, rel_x0 = sy0 - y0, sx0 - x0
        rel_y1, rel_x1 = rel_y0 + sub_h, rel_x0 + sub_w
        
        refined[rel_y0:rel_y1, rel_x0:rel_x1] += sub_recon * weighted_hann
        overlap[rel_y0:rel_y1, rel_x0:rel_x1] += weighted_hann
    
    # Normalize
    mask = overlap > 1e-6
    if np.any(mask):
        refined[mask] /= overlap[mask]
        metadata["refined"] = True
        metadata["sub_tiles"] = len(sub_bounds)
        return refined, metadata
    
    # Refinement failed, return original
    return tile_recon, metadata
