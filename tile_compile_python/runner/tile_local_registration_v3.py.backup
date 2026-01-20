"""
Tile-wise Local Registration (TLR) for astronomical image reconstruction.

This module implements local registration at the tile level, replacing global
frame registration. Each tile is independently registered across all frames,
making it suitable for Alt/Az mounts with field rotation and EQ mounts alike.

Key principles:
- No global reference frame
- Each tile has its own local transformations A_{f,t} per frame
- Registration is integrated into reconstruction
- Translation-only model (rotation is absorbed by local approximation)
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from astropy.io import fits

try:
    import cv2
except Exception:
    cv2 = None

from .opencv_registration import (
    opencv_prepare_ecc_image,
    opencv_phasecorr_translation,
    opencv_ecc_warp,
)


def select_tile_reference_frame(
    frames: List[Path],
    tile_bounds: Tuple[int, int, int, int],
    method: str = "median_time"
) -> int:
    """Select reference frame for a tile.
    
    Args:
        frames: List of frame paths
        tile_bounds: (y_start, y_end, x_start, x_end)
        method: "median_time" or "min_gradient"
        
    Returns:
        Index of reference frame
    """
    if method == "median_time":
        # Simple: use temporal median
        return len(frames) // 2
    
    elif method == "min_gradient":
        # Select frame with minimal gradient energy in this tile
        # (indicates best seeing for this region)
        min_energy = float('inf')
        best_idx = len(frames) // 2
        
        # Sample every 10th frame to avoid full scan
        for i in range(0, len(frames), max(1, len(frames) // 20)):
            try:
                data, _ = _read_fits_tile(frames[i], tile_bounds)
                if data is None:
                    continue
                
                # Compute gradient energy
                gy, gx = np.gradient(data.astype(np.float32))
                energy = float(np.sum(gx**2 + gy**2))
                
                if energy < min_energy:
                    min_energy = energy
                    best_idx = i
            except Exception:
                continue
        
        return best_idx
    
    else:
        return len(frames) // 2


def _read_fits_tile(
    path: Path,
    tile_bounds: Tuple[int, int, int, int]
) -> Tuple[Optional[np.ndarray], Optional[fits.Header]]:
    """Read a tile region from a FITS file.
    
    Args:
        path: Path to FITS file
        tile_bounds: (y_start, y_end, x_start, x_end)
        
    Returns:
        (tile_data, header) or (None, None) if failed
    """
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
    
    Args:
        moving_tile: Tile from moving frame
        ref_tile: Tile from reference frame
        ecc_cc_min: Minimum ECC correlation to accept
        
    Returns:
        (warp_matrix, correlation) or (None, 0.0) if failed
        warp_matrix is 2x3 translation-only: [[1, 0, dx], [0, 1, dy]]
    """
    if cv2 is None:
        return None, 0.0
    
    # Preprocess both tiles identically (ECC preprocessing)
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


def tile_local_register_and_reconstruct(
    frames: List[Path],
    tile_grid: Dict,
    weights_global: np.ndarray,
    weights_local: Dict[str, np.ndarray],
    config: dict,
    channel: str = "R"
) -> Optional[np.ndarray]:
    """Perform tile-wise local registration and reconstruction for one channel.
    
    This is the core TLR function that replaces global registration.
    
    Args:
        frames: List of frame paths (already channel-split)
        tile_grid: Dict with tile definitions
        weights_global: Global weights per frame (G_f)
        weights_local: Local weights per tile per frame (L_{f,t})
        config: Configuration dict with registration.local_tiles settings
        channel: Channel name for weights_local lookup
        
    Returns:
        Reconstructed channel image or None if failed
    """
    if not frames:
        return None
    
    # Get config
    tlr_cfg = config.get("registration", {}).get("local_tiles", {})
    ecc_cc_min = float(tlr_cfg.get("ecc_cc_min", 0.2))
    min_valid_frames = int(tlr_cfg.get("min_valid_frames", 10))
    ref_method = str(tlr_cfg.get("reference_method", "median_time"))
    
    # Get image dimensions from first frame
    with fits.open(str(frames[0])) as hdul:
        h, w = hdul[0].data.shape
    
    # Initialize output
    reconstructed = np.zeros((h, w), dtype=np.float32)
    
    # Get tiles
    tiles = tile_grid.get("tiles", [])
    
    print(f"[TLR] Processing {len(tiles)} tiles for channel {channel}")
    
    for tile_idx, tile in enumerate(tiles):
        y0, y1 = tile["y_start"], tile["y_end"]
        x0, x1 = tile["x_start"], tile["x_end"]
        tile_bounds = (y0, y1, x0, x1)
        
        # Select reference frame for this tile
        ref_idx = select_tile_reference_frame(frames, tile_bounds, method=ref_method)
        ref_tile, _ = _read_fits_tile(frames[ref_idx], tile_bounds)
        
        if ref_tile is None:
            print(f"[WARN] Tile {tile_idx}: failed to read reference")
            continue
        
        # Accumulate weighted contributions
        tile_sum = np.zeros_like(ref_tile, dtype=np.float64)
        weight_sum = np.zeros_like(ref_tile, dtype=np.float64)
        valid_frames = 0
        
        for f_idx, frame_path in enumerate(frames):
            # Get weights
            W_f = weights_global[f_idx]
            L_ft = weights_local[channel][f_idx, tile_idx] if channel in weights_local else 1.0
            W_ft = W_f * L_ft
            
            if W_ft < 1e-6:
                continue
            
            # Read tile
            mov_tile, _ = _read_fits_tile(frame_path, tile_bounds)
            if mov_tile is None:
                continue
            
            # Estimate local translation
            if f_idx == ref_idx:
                # Reference frame: identity transform
                warp = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
                cc = 1.0
            else:
                warp, cc = estimate_tile_local_translation(mov_tile, ref_tile, ecc_cc_min)
                
                if warp is None:
                    # Registration failed for this tile-frame
                    continue
            
            # Apply transformation
            try:
                th, tw = ref_tile.shape
                warped = cv2.warpAffine(mov_tile, warp, (tw, th), flags=cv2.INTER_LINEAR)
                
                # Accumulate
                tile_sum += W_ft * warped.astype(np.float64)
                weight_sum += W_ft
                valid_frames += 1
                
            except Exception:
                continue
        
        # Validate and normalize
        if valid_frames < min_valid_frames:
            print(f"[WARN] Tile {tile_idx}: only {valid_frames} valid frames (< {min_valid_frames})")
            continue
        
        # Avoid division by zero
        mask = weight_sum > 1e-6
        tile_reconstructed = np.zeros_like(ref_tile)
        tile_reconstructed[mask] = (tile_sum[mask] / weight_sum[mask]).astype(np.float32)
        
        # Write to output (handle overlap by averaging)
        reconstructed[y0:y1, x0:x1] += tile_reconstructed
    
    return reconstructed


def tile_local_reconstruct_all_channels(
    frames_by_channel: Dict[str, List[Path]],
    tile_grid: Dict,
    weights_global: np.ndarray,
    weights_local: Dict[str, np.ndarray],
    config: dict
) -> Dict[str, np.ndarray]:
    """Reconstruct all channels using tile-wise local registration.
    
    Args:
        frames_by_channel: {"R": [paths], "G": [paths], "B": [paths]}
        tile_grid: Tile grid definition
        weights_global: Global weights per frame
        weights_local: Local weights per channel per tile
        config: Configuration
        
    Returns:
        {"R": array, "G": array, "B": array}
    """
    results = {}
    
    for channel in ["R", "G", "B"]:
        if channel not in frames_by_channel:
            continue
        
        print(f"[TLR] Reconstructing channel {channel}")
        reconstructed = tile_local_register_and_reconstruct(
            frames_by_channel[channel],
            tile_grid,
            weights_global,
            weights_local,
            config,
            channel=channel
        )
        
        if reconstructed is not None:
            results[channel] = reconstructed
    
    return results
