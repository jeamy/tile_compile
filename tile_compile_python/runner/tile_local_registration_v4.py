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


def estimate_tile_local_translation(
    moving_tile: np.ndarray,
    ref_tile: np.ndarray,
    ecc_cc_min: float = 0.2
) -> Tuple[Optional[np.ndarray], float]:
    """Estimate translation-only transformation for a tile.
    
    Uses Phase Correlation for initial estimate, then ECC refinement.
    
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
    
    # Return final reconstruction
    metadata = {
        "iterations": max_iterations,
        "valid_frames": valid_frames,
        "mean_correlation": float(np.mean([c for c in correlations if c > 0]))
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
            
            # Overlap-add with Hanning window (Methodik v4 §9)
            th, tw = tile_recon.shape
            hann_1d_y = np.hanning(th).astype(np.float32)
            hann_1d_x = np.hanning(tw).astype(np.float32)
            hann_2d = np.outer(hann_1d_y, hann_1d_x)
            
            # Add to output
            reconstructed[y0:y1, x0:x1] += tile_recon * hann_2d
            overlap_count[y0:y1, x0:x1] += hann_2d
        
        # Normalize by overlap count
        mask = overlap_count > 1e-6
        reconstructed[mask] /= overlap_count[mask]
        
        results[channel] = reconstructed
        print(f"[TLR v4] Channel {channel} complete")
    
    return results
