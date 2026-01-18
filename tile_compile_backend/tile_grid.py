import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import cv2


def compute_tile_size_v3(
    image_width: int,
    image_height: int,
    fwhm: float,
    size_factor: float = 8.0,
    min_size: int = 32,
    max_divisor: int = 4
) -> int:
    """
    Compute seeing-adaptive tile size per Methodik v3 §3.3.
    
    Formula:
        T_0 = s · F
        T = floor(clip(T_0, T_min, floor(min(W, H) / D)))
    
    Args:
        image_width: Image width in pixels (W)
        image_height: Image height in pixels (H)
        fwhm: Robust FWHM estimate in pixels (F)
        size_factor: Dimensionless scaling factor (s), default 8.0
        min_size: Minimum tile size (T_min), default 32
        max_divisor: Maximum divisor for upper bound (D), default 4
    
    Returns:
        Computed tile size T
    """
    # T_0 = s · F
    T_0 = size_factor * fwhm
    
    # Upper bound: floor(min(W, H) / D)
    T_max = int(min(image_width, image_height) // max_divisor)
    
    # T = floor(clip(T_0, T_min, T_max))
    T = int(np.floor(np.clip(T_0, min_size, T_max)))
    
    return T


def compute_overlap_v3(tile_size: int, overlap_fraction: float = 0.25) -> Tuple[int, int]:
    """
    Compute overlap and step size per Methodik v3 §3.3.
    
    Formula:
        O = floor(o · T)
        S = T - O
    
    Args:
        tile_size: Tile size T
        overlap_fraction: Overlap fraction o, must be in [0, 0.5]
    
    Returns:
        Tuple of (overlap O, step S)
    """
    if not 0 <= overlap_fraction <= 0.5:
        raise ValueError(f"overlap_fraction must be in [0, 0.5], got {overlap_fraction}")
    
    O = int(np.floor(overlap_fraction * tile_size))
    S = tile_size - O
    
    return O, S


class TileGridGenerator:
    @classmethod
    def generate_adaptive_grid(
        cls, 
        frame: np.ndarray, 
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate an adaptive tile grid per Methodik v3 §3.3.
        
        Args:
            frame: Input astronomical frame
            config: Grid generation configuration with keys:
                - fwhm: FWHM estimate in pixels (required for v3 mode)
                - size_factor: Scaling factor s (default 8.0)
                - min_size: Minimum tile size T_min (default 32)
                - max_divisor: Upper bound divisor D (default 4)
                - overlap_fraction: Overlap o in [0, 0.5] (default 0.25)
                - legacy_mode: If True, use old adaptive logic (default False)
        
        Returns:
            Tile grid configuration and metadata
        """
        config = config or {}
        legacy_mode = config.get('legacy_mode', False)
        
        h, w = frame.shape[:2]
        
        # Analyze frame characteristics
        frame_analysis = cls._analyze_frame_characteristics(frame)
        
        if legacy_mode or 'fwhm' not in config:
            # Legacy adaptive mode (backwards compatibility)
            min_tile_size = config.get('min_tile_size', 32)
            max_tile_size = config.get('max_tile_size', 256)
            base_overlap = config.get('overlap', 0.25)
            
            tile_size = cls._compute_adaptive_tile_size(
                frame.shape, 
                frame_analysis, 
                min_tile_size, 
                max_tile_size
            )
            overlap_frac = cls._compute_adaptive_overlap(
                frame_analysis, 
                base_overlap
            )
            overlap_px = int(overlap_frac * tile_size)
        else:
            # Methodik v3 seeing-adaptive mode
            fwhm = float(config['fwhm'])
            size_factor = float(config.get('size_factor', 8.0))
            min_size = int(config.get('min_size', 32))
            max_divisor = int(config.get('max_divisor', 4))
            overlap_fraction = float(config.get('overlap_fraction', 0.25))
            
            tile_size = compute_tile_size_v3(
                w, h, fwhm, size_factor, min_size, max_divisor
            )
            overlap_px, step = compute_overlap_v3(tile_size, overlap_fraction)
            overlap_frac = overlap_fraction
        
        # Generate grid
        tiles, grid_metadata = cls._generate_grid(
            frame, 
            tile_size, 
            overlap_frac
        )
        
        return {
            'tiles': tiles,
            'tile_size': tile_size,
            'overlap': overlap_frac,
            'overlap_px': overlap_px,
            'frame_metadata': frame_analysis,
            'grid_metadata': grid_metadata
        }
    
    @staticmethod
    def _analyze_frame_characteristics(
        frame: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze frame to guide grid generation
        """
        # Compute gradient for complexity estimation
        gy, gx = np.gradient(frame)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        
        # Detect stars
        stars = cv2.goodFeaturesToTrack(
            np.asarray(frame, dtype=np.float32),
            maxCorners=100, 
            qualityLevel=0.01, 
            minDistance=10
        )
        
        return {
            'mean_intensity': np.mean(frame),
            'std_intensity': np.std(frame),
            'gradient_complexity': np.mean(gradient_magnitude),
            'star_density': len(stars) if stars is not None else 0,
            'shape': frame.shape
        }
    
    @staticmethod
    def _compute_adaptive_tile_size(
        frame_shape: Tuple[int, int], 
        frame_analysis: Dict[str, Any],
        min_tile_size: int,
        max_tile_size: int
    ) -> int:
        """
        Compute adaptive tile size based on frame characteristics
        """
        height, width = frame_shape
        
        # Base tile size computation
        base_size = min(height, width) // 8
        
        # Adjust based on star density
        star_density_factor = frame_analysis['star_density'] / 100
        complexity_factor = frame_analysis['gradient_complexity'] / np.mean(frame_shape)
        
        adaptive_size = int(base_size * (1 + star_density_factor + complexity_factor))
        
        # Constrain tile size
        return max(min(adaptive_size, max_tile_size), min_tile_size)
    
    @staticmethod
    def _compute_adaptive_overlap(
        frame_analysis: Dict[str, Any], 
        base_overlap: float
    ) -> float:
        """
        Compute adaptive overlap based on frame complexity
        """
        complexity_adjustment = frame_analysis['gradient_complexity'] / 100
        star_density_adjustment = frame_analysis['star_density'] / 1000
        
        return min(
            base_overlap * (1 + complexity_adjustment + star_density_adjustment), 
            0.5
        )
    
    @classmethod
    def _generate_grid(
        cls,
        frame: np.ndarray, 
        tile_size: int, 
        overlap: float
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Generate grid of tiles with specified overlap
        """
        height, width = frame.shape
        
        # Compute step size considering overlap
        step = int(tile_size * (1 - overlap))
        
        tiles = []
        tile_coordinates = []
        
        for y in range(0, height - tile_size + 1, step):
            for x in range(0, width - tile_size + 1, step):
                tile = frame[y:y+tile_size, x:x+tile_size]
                tiles.append(tile)
                tile_coordinates.append((x, y, tile_size, tile_size))
        
        grid_metadata = {
            'total_tiles': len(tiles),
            'tile_coordinates': tile_coordinates,
            'coverage_percentage': cls._compute_grid_coverage(
                height, width, tile_size, step
            )
        }
        
        return tiles, grid_metadata
    
    @staticmethod
    def _compute_grid_coverage(
        height: int, 
        width: int, 
        tile_size: int, 
        step: int
    ) -> float:
        """
        Compute grid coverage percentage
        """
        total_area = height * width
        covered_area = len(range(0, height - tile_size + 1, step)) * \
                       len(range(0, width - tile_size + 1, step)) * \
                       (tile_size ** 2)

        return min((covered_area / total_area) * 100, 100.0)

def generate_multi_channel_grid(
    channels: Dict[str, np.ndarray], 
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate tile grids for multiple channels
    
    Args:
        channels: Dictionary of channel frames
        config: Grid generation configuration
    
    Returns:
        Tile grid configurations for each channel
    """
    channel_grids = {}
    
    for channel_name, frame in channels.items():
        channel_grids[channel_name] = TileGridGenerator.generate_adaptive_grid(
            frame, 
            config
        )
    
    return channel_grids