import numpy as np
from typing import Dict, List, Tuple

class MetricsCalculator:
    @staticmethod
    def calculate_global_metrics(frames: List[np.ndarray]) -> Dict[str, float]:
        """
        Calculate global metrics for a channel
        
        Metrics:
        - Background level (B)
        - Noise level (σ)
        - Gradient energy
        """
        metrics = {
            'background_level': np.median([np.median(frame) for frame in frames]),
            'noise_level': np.median([np.std(frame) for frame in frames]),
            'gradient_energy': np.median([MetricsCalculator._calculate_gradient_energy(frame) for frame in frames])
        }
        return metrics
    
    @staticmethod
    def _calculate_gradient_energy(frame: np.ndarray) -> float:
        """
        Calculate gradient energy for a single frame
        """
        # Sobel filters for x and y gradients
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = sobel_x.T
        
        grad_x = np.abs(np.convolve(frame.flatten(), sobel_x.flatten(), mode='valid'))
        grad_y = np.abs(np.convolve(frame.flatten(), sobel_y.flatten(), mode='valid'))
        
        # Combined gradient magnitude
        return np.mean(np.sqrt(grad_x**2 + grad_y**2))

class TileMetricsCalculator:
    """
    Calculate local metrics per tile
    """
    def __init__(self, tile_size: int = 64, overlap: float = 0.25):
        self.tile_size = tile_size
        self.overlap = overlap
    
    def calculate_tile_metrics(self, frame: np.ndarray) -> Dict[str, List[float]]:
        """
        Calculate metrics for each tile in a frame
        """
        tiles = self._generate_tiles(frame)
        
        tile_metrics = {
            'fwhm': [],           # Full Width at Half Maximum
            'roundness': [],      # Star roundness
            'contrast': [],       # Tile contrast
            'background_level': [],
            'noise_level': []
        }
        
        for tile in tiles:
            tile_metrics['fwhm'].append(self._calculate_fwhm(tile))
            tile_metrics['roundness'].append(self._calculate_roundness(tile))
            tile_metrics['contrast'].append(self._calculate_contrast(tile))
            tile_metrics['background_level'].append(np.median(tile))
            tile_metrics['noise_level'].append(np.std(tile))
        
        return tile_metrics
    
    def _generate_tiles(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Generate overlapping tiles from a frame
        """
        h, w = frame.shape
        step = int(self.tile_size * (1 - self.overlap))
        
        tiles = []
        for y in range(0, h - self.tile_size + 1, step):
            for x in range(0, w - self.tile_size + 1, step):
                tile = frame[y:y+self.tile_size, x:x+self.tile_size]
                tiles.append(tile)
        
        return tiles
    
    def _calculate_fwhm(self, tile: np.ndarray) -> float:
        """
        Estimate Full Width at Half Maximum
        """
        # Simplified FWHM estimation
        peak = np.max(tile)
        half_max = peak / 2
        
        # Count pixels above half max
        above_half_max = np.sum(tile >= half_max)
        return np.sqrt(above_half_max / np.pi)
    
    def _calculate_roundness(self, tile: np.ndarray) -> float:
        """
        Calculate star roundness
        """
        # Simplified roundness metric
        max_val = np.max(tile)
        max_indices = np.argwhere(tile == max_val)
        
        # Compute spread of maximum points
        spread = np.std(max_indices, axis=0)
        return 1 / (1 + spread.mean())
    
    def _calculate_contrast(self, tile: np.ndarray) -> float:
        """
        Calculate local contrast
        """
        return (np.max(tile) - np.min(tile)) / (np.max(tile) + np.min(tile) + 1e-8)

def compute_channel_metrics(channels: Dict[str, List[np.ndarray]]) -> Dict[str, Dict]:
    """
    Compute metrics for all channels
    """
    channel_metrics = {}
    tile_calculator = TileMetricsCalculator()
    
    for channel_name, frames in channels.items():
        # Global metrics
        global_metrics = MetricsCalculator.calculate_global_metrics(frames)
        
        # Tile metrics (using first frame as representative)
        tile_metrics = tile_calculator.calculate_tile_metrics(frames[0])
        
        channel_metrics[channel_name] = {
            'global': global_metrics,
            'tiles': tile_metrics
        }
    
    return channel_metrics