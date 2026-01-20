"""
TileProcessor v4 – Central Reconstruction Operator (Methodik v4)

Implements tile-centric reconstruction with:
- Iterative reference refinement (§5.2)
- Temporal warp smoothing (§5.3)
- Registration quality weighting R_{f,t} (§7)
- Translation-only model (§5.1)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

try:
    import cv2
except Exception:
    cv2 = None

from .tile_local_registration_v4 import (
    register_tile,
    compute_warp_variance,
    variance_window_weight,
)
from astropy.io import fits

EPS = 1e-6


def smooth_warps_translation(warps: List[np.ndarray], window: int = 5) -> List[np.ndarray]:
    """Temporal smoothing for translation-only warps (Methodik v4 §5.3).
    
    Uses robust median filter for temporal coherence.
    
    Args:
        warps: List of 2x3 warp matrices
        window: Smoothing window size
        
    Returns:
        Smoothed warp sequence
    """
    if len(warps) < window:
        return warps
    
    smoothed = []
    half = window // 2
    
    for i in range(len(warps)):
        xs, ys = [], []
        for j in range(max(0, i - half), min(len(warps), i + half + 1)):
            w = warps[j]
            xs.append(w[0, 2])
            ys.append(w[1, 2])
        w = warps[i].copy()
        w[0, 2] = float(np.median(xs))
        w[1, 2] = float(np.median(ys))
        smoothed.append(w)
    
    return smoothed


class TileProcessorConfig:
    """Configuration container for TileProcessor (Methodik v4 §13)."""
    
    def __init__(self, cfg: Dict[str, Any]):
        v4_cfg = cfg.get("v4", {})
        reg_cfg = cfg.get("registration", {}).get("local_tiles", {})
        
        # v4 specific
        self.iterations = int(v4_cfg.get("iterations", 3))
        self.beta = float(v4_cfg.get("beta", 5.0))
        
        # Registration
        self.ecc_cc_min = float(reg_cfg.get("ecc_cc_min", 0.2))
        self.min_valid_frames = int(reg_cfg.get("min_valid_frames", 10))
        self.temporal_smoothing_window = int(reg_cfg.get("temporal_smoothing_window", 5))
        
        # Variance window (§9)
        self.variance_window_sigma = float(reg_cfg.get("variance_window_sigma", 2.0))


class TileProcessor:
    """Tile-local reconstruction operator (Methodik v4).
    
    Processes a single tile through iterative registration and reconstruction.
    No global coordinate system - each tile is self-contained.
    """
    
    def __init__(
        self,
        tile_id: int,
        bbox: Tuple[int, int, int, int],
        frame_paths: List[Path],
        global_weights: List[float],
        cfg: TileProcessorConfig,
    ):
        """Initialize TileProcessor.
        
        Args:
            tile_id: Unique tile identifier
            bbox: Tile bounding box (x0, y0, w, h)
            frame_paths: List of FITS file paths (disk streaming)
            global_weights: Global quality weights G_f per frame
            cfg: TileProcessorConfig instance
        """
        self.tile_id = tile_id
        self.bbox = bbox  # (x0, y0, w, h)
        self.frame_paths = frame_paths
        self.global_weights = global_weights
        self.cfg = cfg
        
        self.reference = None
        self.valid = True
        self.warp_variance = 0.0
        self.mean_correlation = 0.0
        self.warps: List[Optional[np.ndarray]] = []
        self.correlations: List[float] = []
    
    def _load_tile(self, path: Path) -> Optional[np.ndarray]:
        """Load tile region from FITS file (disk streaming with memmap)."""
        try:
            with fits.open(str(path), memmap=True) as hdul:
                data = hdul[0].data
                if data is None:
                    return None
                x0, y0, w, h = self.bbox
                tile = data[y0:y0 + h, x0:x0 + w].astype(np.float32, copy=False)
                return tile
        except (ValueError, OSError) as e:
            # Fallback for FITS with BZERO/BSCALE/BLANK keywords
            if "memmap" in str(e).lower() or "BZERO" in str(e) or "BSCALE" in str(e):
                # Only warn once per TileProcessor instance
                if not hasattr(self, '_memmap_warning_shown'):
                    print(f"[WARNING] Tile {self.tile_id}: FITS has BZERO/BSCALE - using memmap=False (higher RAM)")
                    self._memmap_warning_shown = True
                try:
                    with fits.open(str(path), memmap=False) as hdul:
                        data = hdul[0].data
                        if data is None:
                            return None
                        x0, y0, w, h = self.bbox
                        tile = data[y0:y0 + h, x0:x0 + w].astype(np.float32, copy=True)
                        return tile
                except Exception:
                    return None
            return None
        except Exception:
            return None
    
    def _apply_warp(self, tile: np.ndarray, warp: np.ndarray) -> np.ndarray:
        """Apply warp transformation to tile."""
        if cv2 is None:
            return tile
        h, w = tile.shape[:2]
        return cv2.warpAffine(tile, warp, (w, h), flags=cv2.INTER_LINEAR)
    
    def _initial_reference(self) -> np.ndarray:
        """Compute initial reference from temporal median (§5.2)."""
        tiles = []
        for path in self.frame_paths:
            tile = self._load_tile(path)
            if tile is not None:
                tiles.append(tile)
        if not tiles:
            return np.zeros((self.bbox[3], self.bbox[2]), dtype=np.float32)
        stack = np.stack(tiles, axis=0)
        return np.median(stack, axis=0).astype(np.float32)
    
    def run(self) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
        """Execute iterative tile reconstruction (Methodik v4 §5.2, §8).
        
        Returns:
            (reconstructed_tile, warps) or (None, []) if invalid
        """
        if cv2 is None:
            self.valid = False
            return None, []
        
        # Initial reference (median of streamed tiles)
        tiles = []
        for path in self.frame_paths:
            tile = self._load_tile(path)
            if tile is not None:
                tiles.append(tile)
        if not tiles:
            self.valid = False
            return None, []
        ref = np.median(np.stack(tiles, axis=0), axis=0).astype(np.float32)
        del tiles
        
        # Iterative refinement
        final_warps = []
        for iteration in range(self.cfg.iterations):
            warped_tiles = []
            warps = []
            weights = []
            correlations = []
            
            for path, Gf in zip(self.frame_paths, self.global_weights):
                tile = self._load_tile(path)
                if tile is None:
                    continue
                
                warp, cc = register_tile(
                    tile,
                    ref,
                    ecc_cc_min=self.cfg.ecc_cc_min,
                )
                
                if warp is None:
                    continue
                
                warped = self._apply_warp(tile, warp)
                warped_tiles.append(warped)
                warps.append(warp)
                correlations.append(cc)
                
                # W_{f,t} = G_f · R_{f,t} where R_{f,t} = exp(β·(cc-1))
                R_ft = float(np.exp(self.cfg.beta * (cc - 1.0)))
                weights.append(Gf * R_ft)
            
            # Check minimum valid frames (§8 stability)
            if len(warped_tiles) < self.cfg.min_valid_frames:
                self.valid = False
                return None, []
            
            # Temporal smoothing of warps (§5.3)
            warps = smooth_warps_translation(warps, self.cfg.temporal_smoothing_window)
            final_warps = warps
            
            # Weighted reconstruction (§8)
            stack = np.stack(warped_tiles, axis=0)
            w = np.asarray(weights, dtype=np.float64)
            w /= max(EPS, np.sum(w))
            ref = np.sum(stack * w[:, None, None], axis=0).astype(np.float32)
            
            # Store for metadata
            self.warps = warps
            self.correlations = correlations
        
        # Compute metadata for clustering (§10)
        self.warp_variance = compute_warp_variance(self.warps)
        self.mean_correlation = float(np.mean(self.correlations)) if self.correlations else 0.0
        
        self.reference = ref
        return ref, final_warps
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get tile metadata for diagnostics and clustering."""
        return {
            "tile_id": self.tile_id,
            "bbox": self.bbox,
            "valid": self.valid,
            "warp_variance": self.warp_variance,
            "mean_correlation": self.mean_correlation,
            "valid_frames": len(self.warps),
        }


def overlap_add(
    results: List[Tuple[Tuple[int, int, int, int], np.ndarray, float]],
    output_shape: Tuple[int, int],
    variance_sigma: float = 2.0,
) -> np.ndarray:
    """Overlap-add tile reconstruction with Hanning window (Methodik v4 §9).
    
    w_t(p) = hann(p) · ψ(var(Â_{f,t}))
    
    Args:
        results: List of (bbox, tile, warp_variance)
        output_shape: (height, width) of output
        variance_sigma: σ for variance window weight
        
    Returns:
        Reconstructed full image
    """
    h, w = output_shape
    output = np.zeros((h, w), dtype=np.float64)
    weight_sum = np.zeros((h, w), dtype=np.float64)
    
    for bbox, tile, warp_var in results:
        x0, y0, tw, th = bbox
        
        # 2D Hanning window
        hann_y = np.hanning(th).astype(np.float64)
        hann_x = np.hanning(tw).astype(np.float64)
        hann_2d = np.outer(hann_y, hann_x)
        
        # Variance-based window weight: ψ(v) = exp(-v/(2σ²))
        psi = variance_window_weight(warp_var, variance_sigma)
        weighted_hann = hann_2d * psi
        
        # Add to output
        output[y0:y0 + th, x0:x0 + tw] += tile.astype(np.float64) * weighted_hann
        weight_sum[y0:y0 + th, x0:x0 + tw] += weighted_hann
    
    # Normalize
    mask = weight_sum > EPS
    output[mask] /= weight_sum[mask]
    
    return output.astype(np.float32)


def build_initial_tile_grid(
    shape: Tuple[int, int],
    cfg: Dict[str, Any],
) -> List[Tuple[int, int, int, int]]:
    """Build initial tile grid (Methodik v4 §4).
    
    T_0 = clip(32·FWHM, 64, max_tile_size)
    Overlap ≥ 25%
    
    Args:
        shape: Image (height, width)
        cfg: Configuration dict
        
    Returns:
        List of tile bboxes (x0, y0, w, h)
    """
    h, w = shape
    
    grid_cfg = cfg.get("tile_grid", {})
    reg_cfg = cfg.get("registration", {}).get("local_tiles", {})
    
    # Get tile size
    fwhm = float(grid_cfg.get("fwhm", 3.0))
    max_tile_size = int(reg_cfg.get("max_tile_size", 128))
    min_tile_size = int(grid_cfg.get("min_size", 64))
    
    # T_0 = clip(32·FWHM, 64, max_tile_size)
    tile_size = int(np.clip(32 * fwhm, min_tile_size, max_tile_size))
    
    # Overlap ≥ 25%
    overlap_fraction = float(grid_cfg.get("overlap_fraction", 0.25))
    overlap_px = int(tile_size * overlap_fraction)
    step = tile_size - overlap_px
    
    tiles = []
    for y0 in range(0, h - tile_size + 1, step):
        for x0 in range(0, w - tile_size + 1, step):
            tiles.append((x0, y0, tile_size, tile_size))
    
    return tiles


def refine_tiles(
    tiles: List[Tuple[int, int, int, int]],
    warp_variances: List[float],
    threshold: float = 4.0,
    min_size: int = 64,
) -> List[Tuple[int, int, int, int]]:
    """Adaptive tile refinement (Methodik v4 §4).
    
    Split tiles with excessive warp variance.
    
    Args:
        tiles: List of (x0, y0, w, h)
        warp_variances: Variance per tile
        threshold: Variance threshold for splitting
        min_size: Minimum tile dimension (no split below this)
        
    Returns:
        Refined tile list
    """
    refined = []
    
    for (x0, y0, w, h), var in zip(tiles, warp_variances):
        if var < threshold or w <= min_size or h <= min_size:
            refined.append((x0, y0, w, h))
            continue
        
        # Split into 4 quadrants
        hw, hh = w // 2, h // 2
        refined.extend([
            (x0, y0, hw, hh),
            (x0 + hw, y0, w - hw, hh),
            (x0, y0 + hh, hw, h - hh),
            (x0 + hw, y0 + hh, w - hw, h - hh),
        ])
    
    return refined


def global_coarse_normalize(
    frames: List[np.ndarray],
    cfg: Dict[str, Any],
) -> List[np.ndarray]:
    """Global coarse normalization (Methodik v4 §3).
    
    I'_f = I_f / B_f
    
    Args:
        frames: List of input frames
        cfg: Configuration
        
    Returns:
        Normalized frames
    """
    normalized = []
    
    for frame in frames:
        # Robust background estimation
        B_f = np.median(frame)
        if B_f < EPS:
            B_f = 1.0
        
        normalized.append((frame / B_f).astype(np.float32))
    
    return normalized


def compute_global_weights(
    frames: List[np.ndarray],
    cfg: Dict[str, Any],
) -> List[float]:
    """Compute global quality weights G_f (Methodik v4 §7).
    
    G_f = exp(Q_f)
    
    Simple implementation: use inverse of background variance.
    
    Args:
        frames: Normalized frames
        cfg: Configuration
        
    Returns:
        List of global weights
    """
    weights = []
    
    for frame in frames:
        # Simple quality: inverse variance (higher = better)
        bg = np.median(frame)
        noise = np.median(np.abs(frame - bg))
        Q_f = -np.log(max(noise, EPS))  # Higher Q for lower noise
        G_f = float(np.exp(Q_f))
        weights.append(G_f)
    
    # Normalize weights
    total = sum(weights)
    if total > EPS:
        weights = [w / total for w in weights]
    
    return weights
